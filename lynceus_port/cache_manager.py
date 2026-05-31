"""
lynceus_port/cache_manager.py — 移植版分页索引缓存管理器.

改写 ≈ 20%:
  - LRU 改用 LRU-K (K=2): 需要两次访问才算"热", 防止一次性扫描污染
  - CacheStats 增加 eviction_cost_us 追踪 (模拟逐出惩罚)
  - dump_heatmap: 按表的块分布打印热力图
  - _table_identity 使用 table_name 显式字段 (修复 rstrip 塌缩 bug)
"""
from __future__ import annotations
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .cost_model import QueryDescriptor
from .schema import HardwareKind, HardwareTopology
from . import _dbg

DEFAULT_BLOCK_BYTES: int = 2 * (1 << 20)


@dataclass(frozen=True)
class BlockKey:
    table: str
    index_name: str
    block_index: int


@dataclass
class CacheBlock:
    block_id: int
    key: Optional[BlockKey] = None
    ref_count: int = 0
    access_count: int = 0  # ★ LRU-K: 总访问次数


class _FreeBlockPool:
    """LRU-K (K=2) 驱逐池.

    ★ 改写: 块需要 ≥2 次访问才被视为"热", 放入 LRU 尾部;
    只访问过一次的块优先被驱逐 (防扫描污染).
    """

    LRU_K = 2  # ★ K 值

    def __init__(self, num_blocks: int):
        self.blocks: List[CacheBlock] = [
            CacheBlock(block_id=i) for i in range(num_blocks)]
        self._free: List[int] = list(range(num_blocks))
        self._lru_cold: "OrderedDict[int, None]" = OrderedDict()  # access < K
        self._lru_hot: "OrderedDict[int, None]" = OrderedDict()   # access >= K

    def has_free(self) -> bool:
        return bool(self._free) or bool(self._lru_cold) or bool(self._lru_hot)

    def acquire(self) -> Optional[int]:
        if self._free:
            return self._free.pop()
        # ★ 优先驱逐冷块 (访问 < K 次)
        if self._lru_cold:
            victim, _ = self._lru_cold.popitem(last=False)
            return victim
        if self._lru_hot:
            victim, _ = self._lru_hot.popitem(last=False)
            return victim
        return None

    def mark_idle(self, block_id: int) -> None:
        blk = self.blocks[block_id]
        if blk.access_count >= self.LRU_K:
            self._lru_hot[block_id] = None
            self._lru_hot.move_to_end(block_id, last=True)
        else:
            self._lru_cold[block_id] = None
            self._lru_cold.move_to_end(block_id, last=True)

    def mark_busy(self, block_id: int) -> None:
        self._lru_cold.pop(block_id, None)
        self._lru_hot.pop(block_id, None)

    def touch(self, block_id: int) -> None:
        blk = self.blocks[block_id]
        blk.access_count += 1
        # 升级: 冷→热
        if blk.access_count >= self.LRU_K and block_id in self._lru_cold:
            self._lru_cold.pop(block_id, None)
            self._lru_hot[block_id] = None
        if block_id in self._lru_hot:
            self._lru_hot.move_to_end(block_id, last=True)
        elif block_id in self._lru_cold:
            self._lru_cold.move_to_end(block_id, last=True)


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    eviction_cost_us: float = 0.0  # ★ 驱逐惩罚累计

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def dump_snapshot(self) -> str:
        return (f"CacheStats(hits={self.hits}, misses={self.misses}, "
                f"evictions={self.evictions}, hit_rate={self.hit_rate:.2%}, "
                f"evict_cost={self.eviction_cost_us:.0f}µs)")


class IndexCacheManager:
    EVICTION_PENALTY_US: float = 5.0  # ★ 每次驱逐的模拟惩罚

    def __init__(self, device_id: str, capacity_bytes: int,
                 block_bytes: int = DEFAULT_BLOCK_BYTES):
        if block_bytes <= 0:
            raise ValueError("block_bytes must be > 0")
        self.device_id = device_id
        self.block_bytes = block_bytes
        self.num_blocks = max(1, capacity_bytes // block_bytes)
        self._pool = _FreeBlockPool(self.num_blocks)
        self._table: Dict[BlockKey, int] = {}
        self.stats = CacheStats()

    def required_blocks(self, query: QueryDescriptor) -> List[BlockKey]:
        if query.index_available:
            accessed = max(self.block_bytes,
                           query.estimated_data_bytes or self.block_bytes)
            index_name = "idx"
        else:
            accessed = max(self.block_bytes, query.full_table_bytes)
            index_name = "heap"
        n = max(1, -(-accessed // self.block_bytes))
        n = min(n, self.num_blocks)
        table = self._table_identity(query)
        return [BlockKey(table, index_name, i) for i in range(n)]

    @staticmethod
    def _table_identity(query: QueryDescriptor) -> str:
        """显式 table_name 优先 — 修复 rstrip 塌缩."""
        if query.table_name:
            return query.table_name
        stem = query.query_id.split("::", 1)[0]
        m = re.match(r"^(.*?)_\d+$", stem)
        if m and m.group(1):
            return m.group(1)
        return stem or "t"

    def is_resident(self, query: QueryDescriptor) -> bool:
        return all(k in self._table for k in self.required_blocks(query))

    def lookup(self, blocks: List[BlockKey]) -> Tuple[int, int]:
        hits = misses = 0
        for key in blocks:
            bid = self._table.get(key)
            if bid is not None:
                blk = self._pool.blocks[bid]
                if blk.ref_count == 0:
                    self._pool.mark_busy(bid)
                blk.ref_count += 1
                self._pool.touch(bid)
                hits += 1
            else:
                bid = self._admit(key)
                if bid is not None:
                    self._pool.blocks[bid].ref_count += 1
                misses += 1
        self.stats.hits += hits
        self.stats.misses += misses
        return hits, misses

    def _admit(self, key: BlockKey) -> Optional[int]:
        bid = self._pool.acquire()
        if bid is None:
            return None
        blk = self._pool.blocks[bid]
        if blk.key is not None and blk.key in self._table:
            del self._table[blk.key]
            self.stats.evictions += 1
            self.stats.eviction_cost_us += self.EVICTION_PENALTY_US
        blk.key = key
        blk.access_count = 1  # 新块首次访问
        self._table[key] = bid
        return bid

    def release(self, blocks: List[BlockKey]) -> None:
        for key in blocks:
            bid = self._table.get(key)
            if bid is None:
                continue
            blk = self._pool.blocks[bid]
            if blk.ref_count > 0:
                blk.ref_count -= 1
            if blk.ref_count == 0:
                self._pool.mark_idle(bid)

    @property
    def resident_blocks(self) -> int:
        return len(self._table)

    def reset(self) -> None:
        self._pool = _FreeBlockPool(self.num_blocks)
        self._table.clear()
        self.stats = CacheStats()

    def dump_heatmap(self) -> str:
        """断点辅助: 按表打印块分布热力图."""
        table_blocks: Dict[str, int] = {}
        for key in self._table:
            table_blocks[key.table] = table_blocks.get(key.table, 0) + 1
        lines = [f"┌── Cache Heatmap ({self.device_id}) ──",
                 f"│ capacity: {self.num_blocks} blocks, "
                 f"resident: {self.resident_blocks}"]
        for tbl, cnt in sorted(table_blocks.items(), key=lambda x: -x[1]):
            bar = "█" * min(40, cnt)
            lines.append(f"│ {tbl:>15}: {bar} ({cnt})")
        lines.append(f"│ {self.stats.dump_snapshot()}")
        lines.append("└──────────────────────────────")
        return "\n".join(lines)


class TopologyCacheManager:
    def __init__(self, topology: HardwareTopology,
                 cache_fraction: float = 0.5,
                 block_bytes: int = DEFAULT_BLOCK_BYTES):
        if not (0.0 < cache_fraction <= 1.0):
            raise ValueError("cache_fraction must be in (0, 1]")
        self.caches: Dict[str, IndexCacheManager] = {}
        for node_id, node in topology.nodes.items():
            if node.kind == HardwareKind.GPU and node.memory_bytes > 0:
                cap = int(node.memory_bytes * cache_fraction)
                self.caches[node_id] = IndexCacheManager(
                    node_id, cap, block_bytes)

    def get(self, device_id: str) -> Optional[IndexCacheManager]:
        return self.caches.get(device_id)

    def is_resident(self, device_id: str, query: QueryDescriptor) -> bool:
        c = self.caches.get(device_id)
        return c.is_resident(query) if c else False

    def aggregate_hit_rate(self) -> float:
        h = sum(c.stats.hits for c in self.caches.values())
        l = sum(c.stats.lookups for c in self.caches.values())
        return h / l if l else 0.0

    def reset(self) -> None:
        for c in self.caches.values():
            c.reset()
