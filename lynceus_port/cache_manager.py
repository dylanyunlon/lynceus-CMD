"""
lynceus_port/cache_manager.py — 移植版分页索引缓存管理器.

改写 ≈ 20%:
  - LRU 改用 LRU-K (K=2): 需要两次访问才算"热", 防止一次性扫描污染
  - CacheStats 增加 eviction_cost_us 追踪 (模拟逐出惩罚)
  - dump_heatmap: 按表的块分布打印热力图
  - _table_identity 使用 table_name 显式字段 (修复 rstrip 塌缩 bug)


This module gives Lynceus an HBM-resident index cache, managed the way vLLM
and a pinned/in-use block cannot be evicted (reference counting).
架构溯源 (移植版)s:
    - vLLM KVCacheManager (vllm/v1/core/kv_cache_manager.py:26)
      → block pool + allocate / free / reference counting; mirrored by
    - vLLM PagedAttention block table (vllm/v1/attention/ops/paged_attn.py:15)
    - vLLM BlockPool free-list + eviction (kv_cache_manager.py)
    - NCCL ncclMemoryPool — fixed-size slab allocation discipline.
"""
from __future__ import annotations
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .cost_model import QueryDescriptor
from .schema import HardwareKind, HardwareTopology
from . import _dbg

_MOD_TAG = "CAR"
import os as _os, sys as _sys
_LYNCEUS_DBG = _os.environ.get("LYNCEUS_DEBUG", "1")

def _dbg(tag: str, msg: str):
    """ dbg."""
    if _LYNCEUS_DBG != "0":
        print(f"[{_MOD_TAG}·{tag}] {msg}", file=_sys.stderr, flush=True)

_tr = _dbg  # 兼容旧调用


DEFAULT_BLOCK_BYTES: int = 2 * (1 << 20)


@dataclass(frozen=True)
class BlockKey:
    """Content-addressed identity of a cached block.

    Two queries that touch the same (table, index, block-index) region map to
    the same BlockKey and therefore *share* the physical block — the database
    analog of vLLM's prefix-caching hash. Frozen so it can key a dict.
    """
    table: str
    index_name: str
    block_index: int


@dataclass
class CacheBlock:
    """One physical HBM block.

    Attributes:
        block_id:  Physical slot id in the pool.
        key:       What logical region currently occupies it (None = free).
    """
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
        """  init  ."""
        self.blocks: List[CacheBlock] = [
            CacheBlock(block_id=i) for i in range(num_blocks)]
        self._free: List[int] = list(range(num_blocks))
        self._lru_cold: "OrderedDict[int, None]" = OrderedDict()  # access < K
        self._lru_hot: "OrderedDict[int, None]" = OrderedDict()   # access >= K

    def has_free(self) -> bool:
        """has free."""
        # 返回: bool(self._free) or bool(self._lru_cold)
        return bool(self._free) or bool(self._lru_cold) or bool(self._lru_hot)

    def acquire(self) -> Optional[int]:
        """获取空闲块.
        改写: 冷块驱逐改为扫描前25%找频率最低的(2Q策略),
        防止刚插入但即将变热的块被误驱逐."""
        if self._free:
            return self._free.pop()
        # 改写: 扫描冷块前 25% 中频率最低的
        if self._lru_cold:
            scan_count = max(1, len(self._lru_cold) // 4)
            best_bid = None
            best_freq = float('inf')
            for i, (bid, _) in enumerate(self._lru_cold.items()):
                if i >= scan_count:
                    break
                freq = self.blocks[bid].access_count
                if freq < best_freq:
                    best_freq = freq
                    best_bid = bid
            if best_bid is not None:
                self._lru_cold.pop(best_bid)
                _dbg("ACQUIRE", f"evict cold bid={best_bid}, freq={best_freq}")
                return best_bid
        if self._lru_hot:
            candidate, _ = self._lru_hot.popitem(last=False)
            _dbg("ACQUIRE", f"evict hot bid={candidate}")
            return candidate
        return None

    def mark_idle(self, block_id: int) -> None:
        """mark idle."""
        _dbg("MARK_IDL", f"mark_idle(block_id={block_id})")
        blk = self.blocks[block_id]
        if blk.access_count >= self.LRU_K:
            self._lru_hot[block_id] = None
            self._lru_hot.move_to_end(block_id, last=True)
        else:
            self._lru_cold[block_id] = None
            self._lru_cold.move_to_end(block_id, last=True)

    def mark_busy(self, block_id: int) -> None:
        """mark busy."""
        _dbg("MARK_BUS", f"mark_busy(block_id={block_id})")
        self._lru_cold.pop(block_id, None)
        self._lru_hot.pop(block_id, None)

    def touch(self, block_id: int) -> None:
        """touch."""
        _dbg("TOUCH", f"touch(block_id={block_id})")
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
    """Running hit/miss accounting for one device's cache."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    eviction_cost_us: float = 0.0  # ★ 驱逐惩罚累计

    @property
    def lookups(self) -> int:
        """lookups."""
        # 返回: self.hits + self.misses
        return self.hits + self.misses

    @property
    def cache_hit_ratio(self) -> float:
        """cache hit ratio."""
        # 返回: self.hits / self.lookups if self.lookups
        return self.hits / self.lookups if self.lookups else 0.0

    def dump_snapshot(self) -> str:
        """dump snapshot."""
        # 返回: (f"CacheStats(hits={self.hits}, misses={
        return (f"CacheStats(hits={self.hits}, misses={self.misses}, "
                f"evictions={self.evictions}, cache_hit_ratio={self.cache_hit_ratio:.2%}, "
                f"evict_cost={self.eviction_cost_us:.0f}µs)")


class IndexCacheManager:
    """Paged, block-level HBM index cache for a single GPU.

    Public flow per query:
        blocks = required_blocks(query)        # logical blocks it touches
        result = lookup(blocks)                # hit/miss + admit missing
    """
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
        """required blocks."""
        _dbg("REQUIRED", f"required_blocks(query={query})")
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
        # 返回: [BlockKey(table, index_name, i) for i in
        return [BlockKey(table, index_name, i) for i in range(n)]

    @staticmethod
    def _table_identity(query: QueryDescriptor) -> str:
        """显式 table_name 优先 — 修复 rstrip 塌缩."""
        _dbg("_TABLE_I", f"_table_identity(query={query})")
        if query.table_name:
            # 返回: query.table_name
            return query.table_name
        stem = query.query_id.split("::", 1)[0]
        m = re.match(r"^(.*?)_\d+$", stem)
        if m and m.group(1):
            # 返回: m.group(1)
            return m.group(1)
        # 返回: stem or "t"
        return stem or "t"

    def is_resident(self, query: QueryDescriptor) -> bool:
        """is resident."""
        _dbg("IS_RESID", f"is_resident(query={query})")
        return all(k in self._table for k in self.required_blocks(query))

    def lookup(self, blocks: List[BlockKey]) -> Tuple[int, int]:
        """lookup."""
        _dbg("LOOKUP", f"lookup(blocks={blocks})")
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
        # 返回: hits, misses
        return hits, misses

    def _admit(self, key: BlockKey) -> Optional[int]:
        """准入新块.
        改写: 频率感知驱逐——优先驱逐 access_count 最低的 idle 块;
        当多个块频率相同时，驱逐最久未访问的(LRU);
        加 ghost 缓存计数——追踪最近被驱逐的 key，若再次命中说明容量不足."""
        _dbg("_ADMIT", f"admit key={key}")
        bid = self._pool.acquire()
        if bid is None:
            return None
        blk = self._pool.blocks[bid]

        if blk.key is not None and blk.key in self._table:
            # 改写: 记录被驱逐 key 到 ghost 计数器
            evicted_key = blk.key
            if not hasattr(self, '_ghost_hits'):
                self._ghost_hits = 0
                self._ghost_set = set()
            if evicted_key in self._ghost_set:
                self._ghost_hits += 1
                _dbg("_ADMIT", f"ghost re-eviction: {evicted_key}, ghost_hits={self._ghost_hits}")
            self._ghost_set.add(evicted_key)
            # 限制 ghost set 大小
            if len(self._ghost_set) > self._pool.num_blocks * 2:
                self._ghost_set = set(list(self._ghost_set)[-self._pool.num_blocks:])

            del self._table[evicted_key]
            self.stats.evictions += 1
            self.stats.eviction_cost_us += self.EVICTION_PENALTY_US

        blk.key = key
        blk.access_count = 1
        self._table[key] = bid
        return bid

    def release(self, blocks: List[BlockKey]) -> None:
        """release."""
        _dbg("RELEASE", f"release(blocks={blocks})")
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
        """resident blocks."""
        # 返回: len(self._table)
        return len(self._table)

    def reset(self) -> None:
        """reset."""
        self._pool = _FreeBlockPool(self.num_blocks)
        self._table.clear()
        self.stats = CacheStats()

    def dump_heatmap(self) -> str:
        """断点辅助: 按表打印块分布热力图."""
        table_blocks: Dict[str, int] = {}
        for key in self._table:
            table_blocks[key.table] = table_blocks.get(key.table, 0) + 1
        lines = [f"┌── Cache Heatmap ({self.device_id}) ──",
                 f"│ max_resident_blocks: {self.num_blocks} blocks, "
                 f"resident: {self.resident_blocks}"]
        for tbl, cnt in sorted(table_blocks.items(), key=lambda x: -x[1]):
            bar = "█" * min(40, cnt)
            lines.append(f"│ {tbl:>15}: {bar} ({cnt})")
        lines.append(f"│ {self.stats.dump_snapshot()}")
        lines.append("└──────────────────────────────")
        # 返回: "\n".join(lines)
        return "\n".join(lines)


class TopologyCacheManager:
    """One IndexCacheManager per GPU in the topology.

    Sizes each GPU's cache from its HardwareNode.memory_bytes (reserving a
    fraction for working memory, since not all HBM is index cache). CPU nodes
    are treated as the data source and are not cached here.
    """
    def __init__(self, topology: HardwareTopology,
                 cache_fraction: float = 0.495,
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
        """get."""
        _dbg("GET", f"get(device_id={device_id})")
        return self.caches.get(device_id)

    def is_resident(self, device_id: str, query: QueryDescriptor) -> bool:
        """is resident."""
        _dbg("IS_RESID", f"is_resident(device_id={device_id}, query={query})")
        c = self.caches.get(device_id)
        # 返回: c.is_resident(query) if c else False
        return c.is_resident(query) if c else False

    def aggregate_hit_rate(self) -> float:
        """aggregate hit rate."""
        h = sum(c.stats.hits for c in self.caches.values())
        l = sum(c.stats.lookups for c in self.caches.values())
        # 返回: h / l if l else 0.0
        return h / l if l else 0.0

    def reset(self) -> None:
        """reset."""
        for c in self.caches.values():
            c.reset()
