"""
lynceus_port/cache_manager.py — 分页索引缓存管理器。

移植自 lynceus/cache_manager.py，修改约20%:
  - LRU 逐出改为 LRU-K(2)：记录最近2次访问时间，按倒数第2次排序
  - IndexCacheManager 新增 debug_snapshot()
  - TopologyCacheManager.aggregate_stats() 返回详细统计
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .cost_model import QueryDescriptor
from .schema import HardwareKind, HardwareTopology, _dbg

DEFAULT_BLOCK_BYTES: int = 2 * (1 << 20)  # 2 MiB


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
    # ── LRU-K(2)：记录最近两次访问的序号 ──
    access_history: List[int] = field(default_factory=list)


class _FreeBlockPool:
    """空闲池 + LRU-K(2) 逐出"""

    def __init__(self, num_blocks: int):
        self.blocks: List[CacheBlock] = [
            CacheBlock(block_id=i) for i in range(num_blocks)]
        self._free: List[int] = list(range(num_blocks))
        self._lru: "OrderedDict[int, None]" = OrderedDict()
        self._access_counter = 0

    def has_free(self) -> bool:
        return bool(self._free) or bool(self._lru)

    def acquire(self) -> Optional[int]:
        if self._free:
            return self._free.pop()
        if self._lru:
            # ── LRU-K(2): 选倒数第2次访问最早的块 ──
            best_victim = None
            best_k2_time = float('inf')
            for bid in self._lru:
                blk = self.blocks[bid]
                k2 = (blk.access_history[-2]
                       if len(blk.access_history) >= 2
                       else -1)
                if k2 < best_k2_time:
                    best_k2_time = k2
                    best_victim = bid
            if best_victim is not None:
                del self._lru[best_victim]
                return best_victim
        return None

    def mark_idle(self, block_id: int) -> None:
        self._lru[block_id] = None
        self._lru.move_to_end(block_id, last=True)

    def mark_busy(self, block_id: int) -> None:
        self._lru.pop(block_id, None)

    def touch(self, block_id: int) -> None:
        self._access_counter += 1
        blk = self.blocks[block_id]
        blk.access_history.append(self._access_counter)
        if len(blk.access_history) > 3:
            blk.access_history = blk.access_history[-3:]
        if block_id in self._lru:
            self._lru.move_to_end(block_id, last=True)


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


class IndexCacheManager:
    """单 GPU 的分页索引缓存"""

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
                    self._pool.touch(bid)
                misses += 1
        self.stats.hits += hits
        self.stats.misses += misses
        _dbg("Cache",
             f"{self.device_id}: lookup {len(blocks)} blocks, "
             f"hits={hits} misses={misses}")
        return hits, misses

    def _admit(self, key: BlockKey) -> Optional[int]:
        bid = self._pool.acquire()
        if bid is None:
            return None
        blk = self._pool.blocks[bid]
        if blk.key is not None and blk.key in self._table:
            del self._table[blk.key]
            self.stats.evictions += 1
        blk.key = key
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

    def debug_snapshot(self) -> str:
        s = (f"Cache({self.device_id}): "
             f"{self.resident_blocks}/{self.num_blocks} blocks, "
             f"hit_rate={self.stats.hit_rate:.3f} "
             f"({self.stats.hits}h/{self.stats.misses}m/{self.stats.evictions}e)")
        _dbg("Cache", s)
        return s


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

    def debug_snapshot(self) -> str:
        lines = ["=== TopologyCacheManager ==="]
        for dev_id, c in sorted(self.caches.items()):
            lines.append(f"  {c.debug_snapshot()}")
        lines.append(f"  aggregate_hit_rate = {self.aggregate_hit_rate():.3f}")
        s = "\n".join(lines)
        _dbg("TopoCacheMgr", s)
        return s
