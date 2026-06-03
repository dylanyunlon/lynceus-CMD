"""
lynceus_port/cache_manager.py — M017-M018: Paged index cache manager.

Architecture references:
    - vLLM KVCacheManager (vllm/v1/core/kv_cache_manager.py:26)
    - vLLM PagedAttention block table (vllm/v1/attention/ops/paged_attn.py:15)
    - vLLM BlockPool free-list + eviction (kv_cache_manager.py)
    - NCCL ncclMemoryPool — fixed-size slab allocation discipline.

改写 ~20%:
  - _FreeBlockPool: 纯 LRU → LRU-2 (记录最近两次访问时间,
    按第 2 次最老的淘汰; 比纯 LRU 抗 scan 冲刷)
  - required_blocks: index scan 的 block 数加 B-tree fanout
    衰减系数 (深层节点命中率递减, 不再线性 ceil-div)
  - CacheStats: 全局命中率 → EWMA 滑动命中率 (半衰期可调,
    反映近期趋势而非全生命周期平均)
  - aggregate_hit_rate: 简单平均 → 按 lookup 量加权
"""

from __future__ import annotations

import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .cost_model import QueryDescriptor
from .schema import HardwareKind, HardwareTopology
from . import _dbg, _dump_obj, _snapshot, _Timer, LYNCEUS_DEBUG

_T = "CAC"

DEFAULT_BLOCK_BYTES: int = 2 * (1 << 20)  # 2 MiB per block


# ---------------------------------------------------------------------------
# Block identity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BlockKey:
    """Content-addressed identity of a cached block."""
    table: str
    index_name: str
    block_index: int


@dataclass
class CacheBlock:
    """One physical HBM block."""
    block_id: int
    key: Optional[BlockKey] = None
    ref_count: int = 0
    # 改写: LRU-2 需要记录最近两次访问的逻辑时间戳
    access_hist: List[int] = field(default_factory=lambda: [0, 0])


# ---------------------------------------------------------------------------
# Free pool + LRU-2
# ---------------------------------------------------------------------------

class _FreeBlockPool:
    """Free list with LRU-2 eviction.

    改写: 原版纯 LRU 对 sequential scan 脆弱 — 一次全表扫描把整个 cache
    刷成冷数据。LRU-2 记录每个 block 的倒数第二次访问时间戳, 淘汰时选
    second-oldest 最小的 block, 而不是 last-access 最老的。效果: 只被摸过
    一次的 scan block (access_hist[0]==0) 天然排在淘汰队列前面。

    参考: O'Neil, O'Neil, Weikum "The LRU-K Page Replacement Algorithm
    for Database Disk Buffering" (SIGMOD 1993), K=2 的特例。
    """

    def __init__(self, num_blocks: int) -> None:
        self.blocks: List[CacheBlock] = [
            CacheBlock(block_id=i) for i in range(num_blocks)
        ]
        self._free: List[int] = list(range(num_blocks))
        # 改写: 不再用 OrderedDict 维护 LRU 顺序, 改为在 evict 时
        # 扫描 idle set 取 access_hist[0] 最小的 (LRU-2 语义)
        self._idle: set[int] = set()
        self._clock: int = 0  # 逻辑时钟, 每次 touch 递增

        _dbg(_T, f"pool init: {num_blocks} blocks, block_size=CacheBlock")

    @property
    def logical_clock(self) -> int:
        return self._clock

    def has_free(self) -> bool:
        return bool(self._free) or bool(self._idle)

    def acquire(self) -> Optional[int]:
        """Get a physical slot. 改写: evict 按 LRU-2 而非纯 LRU."""
        if self._free:
            bid = self._free.pop()
            _dbg(_T, f"acquire free slot {bid}, remaining_free={len(self._free)}")
            return bid
        if self._idle:
            # 改写核心: 找 access_hist[0] (倒数第二次) 最小的 block
            victim = min(self._idle,
                         key=lambda b: self.blocks[b].access_hist[0])
            self._idle.discard(victim)
            _dbg(_T, f"evict LRU-2 victim={victim}, "
                 f"hist={self.blocks[victim].access_hist}, "
                 f"idle_remaining={len(self._idle)}")
            return victim
        _dbg(_T, "acquire failed: all blocks pinned")
        return None

    def mark_idle(self, block_id: int) -> None:
        self._idle.add(block_id)

    def mark_busy(self, block_id: int) -> None:
        self._idle.discard(block_id)

    def touch(self, block_id: int) -> None:
        """改写: 更新 LRU-2 的双时间戳, 而非只 move_to_end."""
        self._clock += 1
        blk = self.blocks[block_id]
        # shift: 上一次的 "最近" 变成 "倒数第二", 当前时钟写入 "最近"
        blk.access_hist[0] = blk.access_hist[1]
        blk.access_hist[1] = self._clock


# ---------------------------------------------------------------------------
# Per-device cache stats (EWMA)
# ---------------------------------------------------------------------------

@dataclass
class CacheStats:
    """改写: 命中率统计从简单计数改为 EWMA (指数加权移动平均).

    全局计数器保留用于总量审计, 但 hit_rate 返回的是半衰期
    为 halflife 次 lookup 的滑动值, 让 router 对近期 cache 行为
    更敏感, 不被开头的冷启动 miss 拖死。
    """
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    # EWMA 状态
    _ewma_rate: float = 0.0
    _ewma_alpha: float = 0.0  # 在 __post_init__ 里算

    def __init__(self, halflife: int = 200):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        # alpha = 1 - exp(-ln2 / halflife), 即半衰期对应的衰减系数
        self._ewma_alpha = 1.0 - math.exp(-math.log(2) / max(1, halflife))
        self._ewma_rate = 0.0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """改写: 返回 EWMA 滑动命中率而非全局比率."""
        return self._ewma_rate

    @property
    def lifetime_hit_rate(self) -> float:
        """全局比率, 保留用于审计."""
        return self.hits / self.lookups if self.lookups else 0.0

    def record(self, batch_hits: int, batch_misses: int) -> None:
        """改写: 每次 lookup 后调用, 同时更新计数和 EWMA."""
        self.hits += batch_hits
        self.misses += batch_misses
        total = batch_hits + batch_misses
        if total > 0:
            sample_rate = batch_hits / total
            # EWMA update
            self._ewma_rate += self._ewma_alpha * (sample_rate - self._ewma_rate)

    def dump_state(self) -> str:
        return (f"hits={self.hits} miss={self.misses} evict={self.evictions} "
                f"ewma={self._ewma_rate:.4f} lifetime={self.lifetime_hit_rate:.4f}")


# ---------------------------------------------------------------------------
# Per-device cache
# ---------------------------------------------------------------------------

class IndexCacheManager:
    """Paged, block-level HBM index cache for a single GPU."""

    def __init__(self, device_id: str, capacity_bytes: int,
                 block_bytes: int = DEFAULT_BLOCK_BYTES,
                 ewma_halflife: int = 200):
        if block_bytes <= 0:
            raise ValueError("block_bytes must be > 0")
        self.device_id = device_id
        self.block_bytes = block_bytes
        self.num_blocks = max(1, capacity_bytes // block_bytes)
        self._pool = _FreeBlockPool(self.num_blocks)
        self._table: Dict[BlockKey, int] = {}
        self.stats = CacheStats(halflife=ewma_halflife)

        _dbg(_T, f"IndexCacheManager({device_id}): "
             f"capacity={capacity_bytes/(1<<20):.1f}MiB, "
             f"block={block_bytes/(1<<20):.1f}MiB, "
             f"num_blocks={self.num_blocks}")

    # -- block math --------------------------------------------------------

    def required_blocks(self, query: QueryDescriptor) -> List[BlockKey]:
        """改写: index scan 加 B-tree fanout 衰减.

        原版: ceil(accessed_bytes / block_bytes) — 把 index 当连续数组切块。
        改写: 对 index scan, 实际读的 block 数 ≈ result_blocks + depth 层
        内部节点各 1 block (B-tree 自顶向下, 每层最多读一个节点页)。
        这比线性切割更贴近真实 InnoDB/B-tree I/O 模式。
        """
        if query.index_available:
            result_bytes = max(self.block_bytes,
                               query.estimated_data_bytes or self.block_bytes)
            leaf_blocks = max(1, -(-result_bytes // self.block_bytes))
            # 改写: 加 B-tree 内部节点开销, 每层 1 block
            internal_blocks = max(0, query.index_depth - 1)
            n = leaf_blocks + internal_blocks
            index_name = "idx"
        else:
            accessed = max(self.block_bytes, query.full_table_bytes)
            n = max(1, -(-accessed // self.block_bytes))
            index_name = "heap"

        n = min(n, self.num_blocks)
        table = self._table_identity(query)
        keys = [BlockKey(table, index_name, i) for i in range(n)]

        _dbg(_T, f"required_blocks({query.query_id}): "
             f"index={query.index_available}, n={n}, "
             f"table_ident={table}")
        return keys

    @staticmethod
    def _table_identity(query: QueryDescriptor) -> str:
        if query.table_name:
            return query.table_name
        stem = query.query_id.split("::", 1)[0]
        m = re.match(r"^(.*?)_\d+$", stem)
        if m and m.group(1):
            return m.group(1)
        return stem or "t"

    # -- core operations ---------------------------------------------------

    def is_resident(self, query: QueryDescriptor) -> bool:
        blocks = self.required_blocks(query)
        resident = all(k in self._table for k in blocks)
        _dbg(_T, f"is_resident({query.query_id}): {resident}, "
             f"need={len(blocks)}, cached={sum(1 for k in blocks if k in self._table)}")
        return resident

    def lookup(self, blocks: List[BlockKey]) -> Tuple[int, int]:
        """Resolve blocks, pin hits, admit misses."""
        with _Timer(f"lookup@{self.device_id}[{len(blocks)}blk]") as t:
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

        # 改写: 用 EWMA record 替代直接 += 计数
        self.stats.record(hits, misses)

        _dbg(_T, f"lookup done: hits={hits} miss={misses}, "
             f"ewma_rate={self.stats.hit_rate:.4f}, "
             f"pool_clock={self._pool.logical_clock}, "
             f"resident={self.resident_blocks}/{self.num_blocks}")
        return hits, misses

    def _admit(self, key: BlockKey) -> Optional[int]:
        bid = self._pool.acquire()
        if bid is None:
            return None
        blk = self._pool.blocks[bid]
        if blk.key is not None and blk.key in self._table:
            del self._table[blk.key]
            self.stats.evictions += 1
            _dbg(_T, f"evict {blk.key.table}:{blk.key.index_name}[{blk.key.block_index}] "
                 f"from slot {bid}")
        blk.key = key
        blk.access_hist = [0, 0]  # 新入住, 历史清零
        self._table[key] = bid
        return bid

    def release(self, blocks: List[BlockKey]) -> None:
        released = 0
        for key in blocks:
            bid = self._table.get(key)
            if bid is None:
                continue
            blk = self._pool.blocks[bid]
            if blk.ref_count > 0:
                blk.ref_count -= 1
            if blk.ref_count == 0:
                self._pool.mark_idle(bid)
                released += 1
        _dbg(_T, f"release: {released}/{len(blocks)} blocks now idle")

    @property
    def resident_blocks(self) -> int:
        return len(self._table)

    def dump_state(self) -> str:
        """完整状态快照, 用于断点调试."""
        pinned = sum(1 for b in self._pool.blocks if b.ref_count > 0)
        return (f"IndexCacheManager({self.device_id}): "
                f"resident={self.resident_blocks}/{self.num_blocks}, "
                f"pinned={pinned}, idle={len(self._pool._idle)}, "
                f"free={len(self._pool._free)}, "
                f"clock={self._pool.logical_clock}, "
                f"stats=[{self.stats.dump_state()}]")

    def reset(self) -> None:
        _dbg(_T, f"reset {self.device_id}: was {self.stats.dump_state()}")
        self._pool = _FreeBlockPool(self.num_blocks)
        self._table.clear()
        self.stats = CacheStats()


# ---------------------------------------------------------------------------
# Topology-wide manager
# ---------------------------------------------------------------------------

class TopologyCacheManager:
    """One IndexCacheManager per GPU in the topology."""

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
                    node_id, cap, block_bytes
                )
        _dbg(_T, f"TopologyCacheManager: {len(self.caches)} GPU caches, "
             f"fraction={cache_fraction}")

    def get(self, device_id: str) -> Optional[IndexCacheManager]:
        return self.caches.get(device_id)

    def is_resident(self, device_id: str, query: QueryDescriptor) -> bool:
        c = self.caches.get(device_id)
        return c.is_resident(query) if c else False

    def aggregate_hit_rate(self) -> float:
        """改写: 按 lookup 量加权平均, 不再等权.

        原版把一个做了 10 次 lookup 的 GPU 和做了 10000 次的等权平均,
        小样本 GPU 的随机波动会拉偏全局数字。加权后大户说了算。
        """
        total_h = total_l = 0
        for c in self.caches.values():
            total_h += c.stats.hits
            total_l += c.stats.lookups
        rate = total_h / total_l if total_l else 0.0
        _dbg(_T, f"aggregate_hit_rate: {rate:.4f} "
             f"(total_hits={total_h}, total_lookups={total_l})")
        return rate

    def dump_all(self) -> str:
        """打印所有 GPU cache 的完整状态."""
        lines = ["=== TopologyCacheManager state ==="]
        for dev, c in self.caches.items():
            lines.append(f"  {c.dump_state()}")
        lines.append(f"  aggregate_hit_rate={self.aggregate_hit_rate():.4f}")
        return "\n".join(lines)

    def reset(self) -> None:
        _dbg(_T, "resetting all GPU caches")
        for c in self.caches.values():
            c.reset()
