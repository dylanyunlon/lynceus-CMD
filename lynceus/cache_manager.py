"""
lynceus/cache_manager.py — Paged index cache manager.

算法改动:
    1. LRU → LRU-K (K=2): 淘汰时按第 K 次最旧的访问时间排序,
       一次性扫描不会把热数据挤出。
    2. lookup 加 prefetch hint: 当 miss_rate > 60% 时,
       自动预取连续 block (提前 admit 后面 2 个 block)。
"""
from __future__ import annotations
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .costing import QueryDescriptor
from .schema import HardwareKind, HardwareTopology

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
    access_times: List[float] = field(default_factory=list)  # 改动: LRU-K history


class _FreeBlockPool:
    """改动: LRU-K (K=2) 淘汰策略。

    原版: OrderedDict 做纯 LRU, 按最后访问时间淘汰
    新版: 淘汰时比较每个 block 的 backward K-distance
          (即第 K 次倒数的访问时间), 最大的优先淘汰。
          K=2 意味着只看"倒数第二次"访问时间 — 只被访问过一次的 block
          其 K-distance 为 +inf, 一定先被淘汰。
    """
    K = 2

    def __init__(self, num_blocks: int):
        self.blocks: List[CacheBlock] = [
            CacheBlock(block_id=i) for i in range(num_blocks)]
        self._free: List[int] = list(range(num_blocks))
        # 仍然用 OrderedDict 维护 idle 集合, 但淘汰逻辑改用 K-distance
        self._idle: Dict[int, None] = {}
        self._time_counter: int = 0  # 逻辑时钟

    def tick(self) -> int:
        self._time_counter += 1
        return self._time_counter

    def has_free(self) -> bool:
        return bool(self._free) or bool(self._idle)

    def acquire(self) -> Optional[int]:
        if self._free:
            return self._free.pop()
        if not self._idle:
            return None
        # LRU-K: 找 backward K-distance 最大的 idle block
        worst_bid = None
        worst_k_dist = -1
        for bid in self._idle:
            blk = self.blocks[bid]
            if len(blk.access_times) < self.K:
                # 访问次数不足K次 → K-distance = +inf, 优先淘汰
                k_dist = float('inf')
            else:
                # K-distance = 当前时间 - 倒数第K次访问时间
                k_dist = self._time_counter - blk.access_times[-self.K]
            if worst_bid is None or k_dist > worst_k_dist:
                worst_bid = bid
                worst_k_dist = k_dist
        if worst_bid is not None:
            del self._idle[worst_bid]
            return worst_bid
        return None

    def mark_idle(self, block_id: int) -> None:
        self._idle[block_id] = None

    def mark_busy(self, block_id: int) -> None:
        self._idle.pop(block_id, None)

    def touch(self, block_id: int) -> None:
        """记录一次访问, 维护 access_times history (最多保留 K+1 条)。"""
        blk = self.blocks[block_id]
        t = self.tick()
        blk.access_times.append(t)
        # 只保留最近 K+1 条
        if len(blk.access_times) > self.K + 1:
            blk.access_times = blk.access_times[-(self.K + 1):]


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    prefetches: int = 0  # 改动: 预取次数统计
    @property
    def lookups(self) -> int:
        return self.hits + self.misses
    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


class IndexCacheManager:
    """改动: LRU-K 淘汰 + miss-triggered prefetch。"""

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
        self._prefetch_threshold = 0.6  # miss_rate 超过此值触发预取

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
        from ._debug import dbg
        dbg('Cache.lookup', device=self.device_id, n_blocks=len(blocks),
            resident=self.resident_blocks)
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

        # 改动: miss-triggered prefetch
        # 当这轮 miss 比例 > threshold, 预取后续连续 block
        n_total = hits + misses
        if n_total > 0 and misses / n_total > self._prefetch_threshold:
            self._prefetch_ahead(blocks, count=2)

        return hits, misses

    def _prefetch_ahead(self, blocks: List[BlockKey], count: int = 2):
        """预取: 对最后一个 block 之后的连续 count 个 block 做 admit (不加 ref)。"""
        if not blocks:
            return
        last = blocks[-1]
        for offset in range(1, count + 1):
            future_key = BlockKey(last.table, last.index_name,
                                  last.block_index + offset)
            if future_key not in self._table:
                bid = self._admit(future_key)
                if bid is not None:
                    self.stats.prefetches += 1
                    # 不 pin (ref_count=0), 纯预取
                    self._pool.mark_idle(bid)

    def _admit(self, key: BlockKey) -> Optional[int]:
        bid = self._pool.acquire()
        if bid is None:
            return None
        blk = self._pool.blocks[bid]
        if blk.key is not None and blk.key in self._table:
            del self._table[blk.key]
            self.stats.evictions += 1
        blk.key = key
        blk.access_times = []  # 重置 access history
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
                self.caches[node_id] = IndexCacheManager(node_id, cap, block_bytes)

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
