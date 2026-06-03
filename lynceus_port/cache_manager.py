"""
lynceus/cache_manager.py — M017-M018: Paged index cache manager.

The GPU cost model (cost_model.py:218) charges a PCIe transfer every time a
query's data is not already resident in HBM. For a workload that touches the
same indexes and tables over and over — which is exactly what an OLAP query
stream does — paying that PCIe tax on every query is wasteful. If the hot
index blocks already sit in HBM, the transfer term drops to zero and the GPU
path wins on far more queries.

This module gives Lynceus an HBM-resident index cache, managed the way vLLM
manages KV cache: in fixed-size *blocks* rather than per-byte. Blocks are the
unit of allocation, eviction, and sharing. A table's index is chopped into
blocks; queries pull the blocks they need; an LRU manager evicts cold blocks
when HBM fills. Two queries over the same index share blocks (prefix reuse),
and a pinned/in-use block cannot be evicted (reference counting).

Architecture references:
    - vLLM KVCacheManager (vllm/v1/core/kv_cache_manager.py:26)
      → block pool + allocate / free / reference counting; mirrored by
        IndexCacheManager.
    - vLLM PagedAttention block table (vllm/v1/attention/ops/paged_attn.py:15)
      → logical-block → physical-block indirection; mirrored by BlockTable.
    - vLLM BlockPool free-list + eviction (kv_cache_manager.py)
      → free_block_queue / LRU; mirrored by _FreeBlockPool.
    - NCCL ncclMemoryPool — fixed-size slab allocation discipline.

What it does NOT do: move real bytes. Like the rest of Lynceus, this is a
cost-model layer. It tracks *which* blocks are resident so the router can ask
"is this query's data already on the GPU?" and the existing cost model can
skip the transfer term accordingly.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .cost_model import QueryDescriptor
from .schema import HardwareKind, HardwareTopology

from . import _dbg, _dump_obj, _snapshot, _Timer, LYNCEUS_DEBUG
_T = "CAC"



# Default HBM block size for index/table data. vLLM uses 16 tokens/block;
# the database analog is a span of rows whose serialized size is one slab.
DEFAULT_BLOCK_BYTES: int = 2 * (1 << 20)  # 2 MiB per block


# ---------------------------------------------------------------------------
# Block identity
# ---------------------------------------------------------------------------

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
        ref_count: How many in-flight queries pin this block. >0 means it is
                   not evictable, exactly like vLLM's block ref counting.
    """
    block_id: int
    key: Optional[BlockKey] = None
    ref_count: int = 0


# ---------------------------------------------------------------------------
# Free pool + LRU
# ---------------------------------------------------------------------------

class _FreeBlockPool:
    """Free list with LRU eviction over *resident-but-unpinned* blocks.

    Models vLLM's BlockPool: a fixed set of physical blocks, a free queue for
    untouched slots, and an LRU ordering over cached-but-idle blocks so the
    coldest victim is evicted first when the pool is exhausted.
    """

    def __init__(self, num_blocks: int):
        _dbg(_T, "__init__()")
        self.blocks: List[CacheBlock] = [
            CacheBlock(block_id=i) for i in range(num_blocks)
        ]
        # Physically-empty slots, FIFO.
        self._free: List[int] = list(range(num_blocks))
        # Resident, ref_count==0 blocks, ordered oldest→newest (LRU front).
        self._lru: "OrderedDict[int, None]" = OrderedDict()

    def has_free(self) -> bool:
        _dbg(_T, "has_free()")
        return bool(self._free) or bool(self._lru)

    def acquire(self) -> Optional[int]:
        """Get a physical slot, evicting the LRU resident block if needed.

        Returns the block_id, or None if every block is pinned (ref_count>0).
        """
        _dbg(_T, "acquire()")
        if self._free:
            return self._free.pop()
        if self._lru:
            victim, _ = self._lru.popitem(last=False)  # oldest
            # NOTE: we leave blk.key intact so the caller (_admit) can evict
            # the stale table entry before rebinding the slot.
            return victim
        return None  # fully pinned — cannot make room

    def mark_idle(self, block_id: int) -> None:
        """Block hit ref_count 0: becomes an LRU eviction candidate (newest)."""
        _dbg(_T, "mark_idle()")
        self._lru[block_id] = None
        self._lru.move_to_end(block_id, last=True)

    def mark_busy(self, block_id: int) -> None:
        """Block pinned again: remove from eviction candidates."""
        _dbg(_T, "mark_busy()")
        self._lru.pop(block_id, None)

    def touch(self, block_id: int) -> None:
        """Mark a resident idle block as most-recently-used."""
        _dbg(_T, "touch()")
        if block_id in self._lru:
            self._lru.move_to_end(block_id, last=True)


# ---------------------------------------------------------------------------
# Per-device cache
# ---------------------------------------------------------------------------

@dataclass
class CacheStats:
    """Running hit/miss accounting for one device's cache."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def lookups(self) -> int:
        _dbg(_T, "lookups()")
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        _dbg(_T, "hit_rate()")
        return self.hits / self.lookups if self.lookups else 0.0


class IndexCacheManager:
    """Paged, block-level HBM index cache for a single GPU.

    Public flow per query:
        blocks = required_blocks(query)        # logical blocks it touches
        result = lookup(blocks)                # hit/miss + admit missing
        ... query executes, data now resident ...
        release(blocks)                        # unpin

    The router consults `is_resident(query)` to decide whether the GPU cost
    model should skip its PCIe transfer term.
    """

    def __init__(self, device_id: str, capacity_bytes: int,
                 block_bytes: int = DEFAULT_BLOCK_BYTES):
        _dbg(_T, "__init__()")
        if block_bytes <= 0:
            raise ValueError("block_bytes must be > 0")
        self.device_id = device_id
        self.block_bytes = block_bytes
        self.num_blocks = max(1, capacity_bytes // block_bytes)
        self._pool = _FreeBlockPool(self.num_blocks)
        # Logical key → resident physical block_id (the "block table").
        self._table: Dict[BlockKey, int] = {}
        self.stats = CacheStats()

    # -- block math --------------------------------------------------------

    def required_blocks(self, query: QueryDescriptor) -> List[BlockKey]:
        """Logical blocks a query touches.

        Block count = ceil(accessed_bytes / block_bytes), where accessed_bytes
        is the index footprint when an index is available, else the scanned
        data footprint. table/index names are synthesized from the query id
        so repeated queries over the same data collide on the same keys (and
        therefore share blocks).
        """
        _dbg(_T, "required_blocks()")
        if query.index_available:
            # Index footprint ≈ index_depth levels of fanout over the result.
            accessed = max(self.block_bytes,
                           query.estimated_data_bytes or self.block_bytes)
            index_name = "idx"
        else:
            accessed = max(self.block_bytes, query.full_table_bytes)
            index_name = "heap"
        n = max(1, -(-accessed // self.block_bytes))  # ceil div
        # Cap a single query's footprint at the whole cache so one giant scan
        # cannot request more blocks than exist (it would just stream/thrash).
        n = min(n, self.num_blocks)
        table = self._table_identity(query)
        return [BlockKey(table, index_name, i) for i in range(n)]

    @staticmethod
    def _table_identity(query: QueryDescriptor) -> str:
        """Resolve the logical table a query reads, for cache-block keying.

        Prefer the explicit QueryDescriptor.table_name. Only if it is empty do
        we fall back to deriving an identity from query_id — and even then we
        strip ONLY a trailing run-id suffix of the form '_<digits>' (e.g.
        'orders_42' -> 'orders'), via an exact suffix match. We must NOT use
        str.rstrip(digits), which removes every trailing digit/underscore
        character regardless of structure and collapses unrelated ids:
        'q_00000' -> 'q' would merge thousands of distinct queries onto one
        bogus table and inflate the cache hit rate.
        """
        _dbg(_T, "_table_identity()")
        if query.table_name:
            return query.table_name
        stem = query.query_id.split("::", 1)[0]
        m = re.match(r"^(.*?)_\d+$", stem)
        if m and m.group(1):
            return m.group(1)
        return stem or "t"

    # -- core operations ---------------------------------------------------

    def is_resident(self, query: QueryDescriptor) -> bool:
        """True iff *every* block the query needs is currently in HBM.

        This is the question the router asks to zero out the PCIe term.
        """
        _dbg(_T, "is_resident()")
        return all(k in self._table for k in self.required_blocks(query))

    def lookup(self, blocks: List[BlockKey]) -> Tuple[int, int]:
        """Resolve a query's blocks, admitting any that miss.

        Pins every resolved block (ref_count++) so it cannot be evicted while
        the query runs. Returns (hits, misses).
        """
        _dbg(_T, "lookup()")
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
        """Bring a missing block into HBM, evicting LRU if necessary."""
        _dbg(_T, "_admit()")
        bid = self._pool.acquire()
        if bid is None:
            return None  # pool fully pinned; block streams without caching
        blk = self._pool.blocks[bid]
        if blk.key is not None and blk.key in self._table:
            # The slot we reclaimed was holding someone else's block.
            del self._table[blk.key]
            self.stats.evictions += 1
        blk.key = key
        self._table[key] = bid
        return bid

    def release(self, blocks: List[BlockKey]) -> None:
        """Unpin a query's blocks; idle ones become eviction candidates."""
        _dbg(_T, "release()")
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
        _dbg(_T, "resident_blocks()")
        return len(self._table)

    def reset(self) -> None:
        _dbg(_T, "reset()")
        self._pool = _FreeBlockPool(self.num_blocks)
        self._table.clear()
        self.stats = CacheStats()


# ---------------------------------------------------------------------------
# Topology-wide manager
# ---------------------------------------------------------------------------

class TopologyCacheManager:
    """One IndexCacheManager per GPU in the topology.

    Sizes each GPU's cache from its HardwareNode.memory_bytes (reserving a
    fraction for working memory, since not all HBM is index cache). CPU nodes
    are treated as the data source and are not cached here.
    """

    def __init__(self, topology: HardwareTopology,
                 cache_fraction: float = 0.5,
                 block_bytes: int = DEFAULT_BLOCK_BYTES):
        _dbg(_T, "__init__()")
        if not (0.0 < cache_fraction <= 1.0):
            raise ValueError("cache_fraction must be in (0, 1]")
        self.caches: Dict[str, IndexCacheManager] = {}
        for node_id, node in topology.nodes.items():
            if node.kind == HardwareKind.GPU and node.memory_bytes > 0:
                cap = int(node.memory_bytes * cache_fraction)
                self.caches[node_id] = IndexCacheManager(
                    node_id, cap, block_bytes
                )

    def get(self, device_id: str) -> Optional[IndexCacheManager]:
        _dbg(_T, "get()")
        return self.caches.get(device_id)

    def is_resident(self, device_id: str, query: QueryDescriptor) -> bool:
        _dbg(_T, "is_resident()")
        c = self.caches.get(device_id)
        return c.is_resident(query) if c else False

    def aggregate_hit_rate(self) -> float:
        _dbg(_T, "aggregate_hit_rate()")
        h = sum(c.stats.hits for c in self.caches.values())
        l = sum(c.stats.lookups for c in self.caches.values())
        return h / l if l else 0.0

    def reset(self) -> None:
        _dbg(_T, "reset()")
        for c in self.caches.values():
            c.reset()
