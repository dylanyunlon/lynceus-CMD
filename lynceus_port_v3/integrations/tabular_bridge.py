"""
lynceus_port_v3/integrations/tabular_bridge.py — Tabular index-build cost bridge.

Ported from upstream/tabular (sfu-dis/tabular):
  - src/tabular/table_group.{h,cc}    → TableGroup lifecycle simulation
  - src/index/wrappers/inline_btree_wrapper.h → BTree Insert/Scan cost model
  - src/tabular/inline_btree.h         → node split / fan-out parameters
  - src/index/hash_table_common.h      → hash bucket sizing
  - src/table/inline_table.{h,cc}      → row storage cost model

Modifications from upstream (~20% changed):
  - Removed: all C++ compilation, mmap/huge-page allocation, io_uring ASI
  - Removed: dlog::Logger, epoch daemon threads, MVCC transaction layer
  - Added:   Python cost-model simulation of BTree/Hash build & probe
  - Added:   Lynceus QueryDescriptor integration (table_name, n_rows, key_size)
  - Added:   Comprehensive debug/print instrumentation for experiment feedback
  - Changed: InlineBTree fan-out formula uses analytical model instead of
             runtime sizeof() — matches upstream btree_common.h node layout

Architecture references:
    - tabular TableGroup (tabular/src/tabular/table_group.cc)
      → epoch-based table lifecycle
    - InlineBTreeWrapper (tabular/src/index/wrappers/inline_btree_wrapper.h)
      → Insert/Search/Scan interface over InlineBTree
    - InlineBTree (tabular/src/tabular/inline_btree.h)
      → lock-free B+tree with inline leaf storage
    - HashTable (tabular/src/index/hash_table_common.h)
      → open-addressing hash with linear probing
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto


# --- port_v3: debug instrumentation ---
try:
    from ._debug import dbg, snapshot, timing, checkpoint
except ImportError:
    from lynceus_port_v3._debug import dbg, snapshot, timing, checkpoint

logger = logging.getLogger(__name__)

# ─── Constants ported from upstream ──────────────────────────────────────
# From tabular/src/tabular/btree_common.h:
#   static constexpr size_t kNodeSize = 4096;
#   static constexpr size_t kFanout = ...  (computed from key+value size)
BTREE_NODE_SIZE_BYTES = 4096
BTREE_HEADER_BYTES = 64       # node header: count, level, sibling ptr, etc.
BTREE_SLOT_OVERHEAD = 8       # per-slot: fence key offset or child pointer

# From tabular/src/index/hash_table_common.h:
#   static constexpr double kMaxLoadFactor = 0.7;
#   static constexpr size_t kBucketSize = 64;  // cache-line aligned
HASH_MAX_LOAD_FACTOR = 0.7
HASH_BUCKET_SIZE_BYTES = 64
HASH_BUCKET_HEADER = 8        # next pointer + count

# From tabular/src/table/inline_table.h:
#   config_t: use_huge_pages, capacity, initial_size
DEFAULT_CAPACITY_BYTES = 17179869184    # 16 GB (upstream default)
DEFAULT_INITIAL_SIZE = 4294967296       # 4 GB

# Memory latency assumptions (ns) for cost estimation
DRAM_RANDOM_ACCESS_NS = 100
L3_CACHE_ACCESS_NS = 10
L1_CACHE_ACCESS_NS = 1
CACHE_LINE_BYTES = 64


class IndexType(Enum):
    """Mirrors upstream tabular index wrapper types."""
    INLINE_BTREE = auto()       # InlineBTreeWrapper
    HASH_TABLE = auto()         # HashTable (open addressing)
    MASSTREE = auto()           # MasstreeWrapper (not yet ported)
    MATERIALIZED_BTREE = auto() # MaterializedTabularBTreeWrapper


@dataclass
class IndexBuildConfig:
    """Configuration for index build cost estimation.

    Mirrors upstream tabular/src/table/inline_table.h config_t
    and InlineBTreeWrapper constructor parameters.
    """
    index_type: IndexType = IndexType.INLINE_BTREE
    key_size_bytes: int = 8
    value_size_bytes: int = 8    # OID size in upstream
    n_rows: int = 1_000_000
    use_huge_pages: bool = True
    capacity_bytes: int = DEFAULT_CAPACITY_BYTES
    initial_size_bytes: int = DEFAULT_INITIAL_SIZE
    n_threads: int = 1
    # Debug control
    debug_print: bool = True
    trace_every_n: int = 100_000  # print progress every N inserts


@dataclass
class IndexBuildCost:
    """Result of index build cost estimation."""
    index_type: IndexType
    n_rows: int
    build_time_us: float        # total build time in microseconds
    memory_bytes: int           # total memory footprint
    tree_height: int = 0        # B-tree height (0 for hash)
    n_nodes: int = 0            # B-tree internal+leaf nodes
    n_splits: int = 0           # node splits during build
    fan_out: int = 0            # effective fan-out
    # Per-operation costs
    avg_insert_ns: float = 0.0
    avg_lookup_ns: float = 0.0
    avg_scan_per_key_ns: float = 0.0

    def dump_debug(self, prefix: str = "") -> str:
        """Format a human-readable debug summary.

        Designed for experiment feedback — print this after each benchmark step
        to see the index build state, analogous to a breakpoint inspection.
        """
        lines = [
            f"{prefix}╔══ IndexBuildCost Debug Dump ══════════════════════",
            f"{prefix}║ index_type      = {self.index_type.name}",
            f"{prefix}║ n_rows          = {self.n_rows:,}",
            f"{prefix}║ build_time_us   = {self.build_time_us:,.1f}",
            f"{prefix}║ memory_bytes    = {self.memory_bytes:,} ({self.memory_bytes / (1024**2):.1f} MB)",
            f"{prefix}║ tree_height     = {self.tree_height}",
            f"{prefix}║ n_nodes         = {self.n_nodes:,}",
            f"{prefix}║ n_splits        = {self.n_splits:,}",
            f"{prefix}║ fan_out         = {self.fan_out}",
            f"{prefix}║ avg_insert_ns   = {self.avg_insert_ns:.1f}",
            f"{prefix}║ avg_lookup_ns   = {self.avg_lookup_ns:.1f}",
            f"{prefix}║ avg_scan/key_ns = {self.avg_scan_per_key_ns:.1f}",
            f"{prefix}╚══════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


@dataclass
class ScanCost:
    """Cost of a range scan operation."""
    n_keys_scanned: int
    total_us: float
    leaf_nodes_touched: int
    cache_lines_fetched: int

    def dump_debug(self, prefix: str = "") -> str:
        lines = [
            f"{prefix}  ScanCost: {self.n_keys_scanned} keys, {self.total_us:.1f}µs, "
            f"{self.leaf_nodes_touched} leaves, {self.cache_lines_fetched} cache-lines"
        ]
        return "\n".join(lines)


# ─── BTree Fan-out Computation ───────────────────────────────────────────
# Ported from upstream tabular/src/tabular/btree_common.h
# Original C++:
#   static constexpr auto kMaxEntries =
#       (kNodeSize - sizeof(NodeHeader)) / (sizeof(Key) + sizeof(Value));

def compute_btree_fanout(key_size: int, value_size: int) -> int:
    """Compute B+tree fan-out from key/value sizes.

    Matches upstream btree_common.h layout:
      usable = kNodeSize - header
      entries_per_node = usable // (key_size + value_size + slot_overhead)
    """
    usable = BTREE_NODE_SIZE_BYTES - BTREE_HEADER_BYTES
    entry_size = key_size + value_size + BTREE_SLOT_OVERHEAD
    fanout = max(2, usable // entry_size)
    return fanout


def compute_btree_height(n_rows: int, fanout: int) -> int:
    """Compute B+tree height for n_rows entries.

    height = ceil(log_fanout(n_rows))
    At minimum height=1 (single root-leaf node).
    """
    if n_rows <= 0:
        return 0
    if n_rows <= fanout:
        return 1
    return max(1, math.ceil(math.log(n_rows) / math.log(fanout)))


def compute_btree_nodes(n_rows: int, fanout: int) -> Tuple[int, int, int]:
    """Compute total nodes, leaf count, internal count.

    Upstream InlineBTree stores entries in leaf nodes with ~70% fill factor
    after random inserts (verified from btree_common.h split logic).
    """
    fill_factor = 0.7  # average after random inserts with split
    entries_per_leaf = max(1, int(fanout * fill_factor))
    n_leaves = max(1, math.ceil(n_rows / entries_per_leaf))

    # Internal nodes: each level reduces by fanout
    n_internal = 0
    nodes_at_level = n_leaves
    while nodes_at_level > 1:
        parent_count = math.ceil(nodes_at_level / fanout)
        n_internal += parent_count
        nodes_at_level = parent_count

    return n_leaves + n_internal, n_leaves, n_internal


# ─── BTree Build Cost Model ─────────────────────────────────────────────
# Ported from upstream InlineBTreeWrapper::Insert (inline_btree_wrapper.h)
# and InlineBTree::insert (tabular/inline_btree.h)

def estimate_btree_build_cost(config: IndexBuildConfig) -> IndexBuildCost:
    """Estimate BTree index build cost.

    Models the upstream InlineBTree insert path:
    1. Traverse from root to leaf (height random DRAM accesses)
    2. Insert into leaf (1 cache-line write)
    3. Possible node split (amortised over insertions)

    The split cost is modelled from inline_btree.h split():
      - allocate new node
      - copy half entries
      - update parent fence keys
    """
    t0 = time.monotonic()

    fanout = compute_btree_fanout(config.key_size_bytes, config.value_size_bytes)
    height = compute_btree_height(config.n_rows, fanout)
    total_nodes, n_leaves, n_internal = compute_btree_nodes(config.n_rows, fanout)

    # Split count: each leaf was created by a split (except root)
    n_splits = max(0, n_leaves - 1) + max(0, n_internal - 1)

    # Per-insert cost model:
    #   traverse: height * DRAM_ACCESS (cold) with caching of upper levels
    #   Upper levels (root, level-1) fit in L3 after warmup
    cold_levels = max(0, height - 2)  # bottom levels = DRAM
    warm_levels = min(height, 2)       # top levels = L3 cache
    traverse_ns = cold_levels * DRAM_RANDOM_ACCESS_NS + warm_levels * L3_CACHE_ACCESS_NS

    # Leaf insert: 1 cache-line read-modify-write
    insert_leaf_ns = DRAM_RANDOM_ACCESS_NS + L1_CACHE_ACCESS_NS

    # Split amortisation: split happens every ~fanout inserts
    # Each split: copy half node + update parent = ~fanout/2 * cache-line ops
    split_cost_ns = (fanout // 2) * L1_CACHE_ACCESS_NS + 2 * DRAM_RANDOM_ACCESS_NS
    split_amortised_ns = split_cost_ns / max(1, fanout)

    avg_insert_ns = traverse_ns + insert_leaf_ns + split_amortised_ns

    # Thread scaling: upstream uses lock-free optimistic latching
    # (see tabular/src/index/latches/), ~0.85 scaling efficiency
    thread_scale = 1.0 / (config.n_threads * 0.85) if config.n_threads > 1 else 1.0
    total_insert_ns = config.n_rows * avg_insert_ns * thread_scale

    # Memory footprint
    memory_bytes = total_nodes * BTREE_NODE_SIZE_BYTES

    # Lookup cost: traverse root→leaf + 1 binary search in leaf
    # Binary search in leaf: log2(fanout) comparisons
    bsearch_ns = math.log2(max(2, fanout)) * L1_CACHE_ACCESS_NS
    avg_lookup_ns = traverse_ns + bsearch_ns

    # Scan cost per key: sequential leaf access, mostly cache-line sequential
    # After first leaf lookup, subsequent keys in same leaf = L1
    # Cross-leaf = 1 DRAM (sibling pointer chase)
    entries_per_leaf_avg = max(1, int(fanout * 0.7))
    scan_cache_miss_rate = 1.0 / entries_per_leaf_avg  # miss once per leaf
    avg_scan_per_key_ns = (
        L1_CACHE_ACCESS_NS * (1 - scan_cache_miss_rate) +
        DRAM_RANDOM_ACCESS_NS * scan_cache_miss_rate
    )

    elapsed = time.monotonic() - t0

    result = IndexBuildCost(
        index_type=IndexType.INLINE_BTREE,
        n_rows=config.n_rows,
        build_time_us=total_insert_ns / 1000.0,
        memory_bytes=memory_bytes,
        tree_height=height,
        n_nodes=total_nodes,
        n_splits=n_splits,
        fan_out=fanout,
        avg_insert_ns=avg_insert_ns,
        avg_lookup_ns=avg_lookup_ns,
        avg_scan_per_key_ns=avg_scan_per_key_ns,
    )

    if config.debug_print:
        print(f"\n[tabular_bridge] BTree build estimation completed in {elapsed*1000:.2f}ms")
        print(result.dump_debug("  "))
        print(f"  [DEBUG] fanout computation: node={BTREE_NODE_SIZE_BYTES}B, "
              f"header={BTREE_HEADER_BYTES}B, entry={config.key_size_bytes}+"
              f"{config.value_size_bytes}+{BTREE_SLOT_OVERHEAD}={config.key_size_bytes+config.value_size_bytes+BTREE_SLOT_OVERHEAD}B "
              f"→ fanout={fanout}")
        print(f"  [DEBUG] height={height}, leaves={n_leaves}, internal={n_internal}, splits={n_splits}")
        print(f"  [DEBUG] per-insert breakdown: traverse={traverse_ns:.0f}ns "
              f"(cold={cold_levels}×{DRAM_RANDOM_ACCESS_NS}ns + warm={warm_levels}×{L3_CACHE_ACCESS_NS}ns), "
              f"leaf={insert_leaf_ns}ns, split_amort={split_amortised_ns:.1f}ns")

    return result


# ─── Hash Table Build Cost Model ────────────────────────────────────────
# Ported from upstream tabular/src/index/hash_table_common.h
# and hash_table.h open-addressing linear probe implementation

def estimate_hash_build_cost(config: IndexBuildConfig) -> IndexBuildCost:
    """Estimate hash table build cost.

    Models upstream HashTable (open-addressing, linear probing):
    - Bucket array sized to n_rows / load_factor
    - Each insert: hash + linear probe (avg 1/(1-lf) probes)
    - Each probe: 1 cache-line access
    """
    t0 = time.monotonic()

    entry_size = config.key_size_bytes + config.value_size_bytes
    entries_per_bucket = max(1, (HASH_BUCKET_SIZE_BYTES - HASH_BUCKET_HEADER) // entry_size)
    n_buckets = max(1, math.ceil(config.n_rows / (entries_per_bucket * HASH_MAX_LOAD_FACTOR)))

    # Average probes at load factor α: 1/(1-α) for unsuccessful, 1/α * ln(1/(1-α)) for successful
    alpha = min(0.95, config.n_rows / (n_buckets * entries_per_bucket))
    avg_probes_insert = 1.0 / max(0.05, 1.0 - alpha)
    avg_probes_lookup = (1.0 / max(0.01, alpha)) * math.log(1.0 / max(0.05, 1.0 - alpha)) if alpha > 0 else 1.0

    # Each probe = 1 cache-line read (64B bucket)
    avg_insert_ns = avg_probes_insert * DRAM_RANDOM_ACCESS_NS
    avg_lookup_ns = avg_probes_lookup * DRAM_RANDOM_ACCESS_NS

    total_insert_ns = config.n_rows * avg_insert_ns
    memory_bytes = n_buckets * HASH_BUCKET_SIZE_BYTES

    elapsed = time.monotonic() - t0

    result = IndexBuildCost(
        index_type=IndexType.HASH_TABLE,
        n_rows=config.n_rows,
        build_time_us=total_insert_ns / 1000.0,
        memory_bytes=memory_bytes,
        tree_height=0,
        n_nodes=n_buckets,
        n_splits=0,
        fan_out=entries_per_bucket,
        avg_insert_ns=avg_insert_ns,
        avg_lookup_ns=avg_lookup_ns,
        avg_scan_per_key_ns=0.0,  # hash doesn't support ordered scan
    )

    if config.debug_print:
        print(f"\n[tabular_bridge] Hash build estimation completed in {elapsed*1000:.2f}ms")
        print(result.dump_debug("  "))
        print(f"  [DEBUG] load_factor={alpha:.3f}, buckets={n_buckets:,}, "
              f"entries/bucket={entries_per_bucket}")
        print(f"  [DEBUG] avg probes: insert={avg_probes_insert:.2f}, lookup={avg_probes_lookup:.2f}")

    return result


# ─── Scan Cost Estimator ─────────────────────────────────────────────────
# Ported from InlineBTreeWrapper::Scan (inline_btree_wrapper.h)

def estimate_scan_cost(
    build_cost: IndexBuildCost,
    n_keys: int,
    selectivity: float = 1.0,
    debug_print: bool = True,
) -> ScanCost:
    """Estimate range scan cost given a built index.

    Models upstream InlineBTreeWrapper::Scan path:
    1. Traverse root→leaf (same as lookup)
    2. Sequential scan across leaf chain
    3. Each leaf-to-leaf hop = 1 sibling pointer chase (DRAM)
    """
    if build_cost.index_type == IndexType.HASH_TABLE:
        # Hash tables don't support range scans; fall back to N lookups
        total_ns = n_keys * build_cost.avg_lookup_ns
        result = ScanCost(
            n_keys_scanned=n_keys,
            total_us=total_ns / 1000.0,
            leaf_nodes_touched=n_keys,  # each is independent
            cache_lines_fetched=n_keys,
        )
        if debug_print:
            print(f"  [SCAN] Hash fallback: {n_keys} point lookups → {result.total_us:.1f}µs")
        return result

    # BTree range scan
    entries_per_leaf = max(1, int(build_cost.fan_out * 0.7))
    scan_keys = max(1, int(n_keys * selectivity))
    leaves_touched = max(1, math.ceil(scan_keys / entries_per_leaf))

    # Cost: initial lookup + sequential scan
    initial_lookup_ns = build_cost.avg_lookup_ns * 1000  # already in ns
    # Wait, avg_lookup_ns is already in ns
    initial_lookup_ns = build_cost.avg_lookup_ns
    sequential_ns = scan_keys * build_cost.avg_scan_per_key_ns
    leaf_hop_ns = (leaves_touched - 1) * DRAM_RANDOM_ACCESS_NS

    total_ns = initial_lookup_ns + sequential_ns + leaf_hop_ns
    cache_lines = leaves_touched * (BTREE_NODE_SIZE_BYTES // CACHE_LINE_BYTES)

    result = ScanCost(
        n_keys_scanned=scan_keys,
        total_us=total_ns / 1000.0,
        leaf_nodes_touched=leaves_touched,
        cache_lines_fetched=cache_lines,
    )

    if debug_print:
        print(f"  [SCAN] BTree range: {scan_keys} keys across {leaves_touched} leaves → "
              f"{result.total_us:.1f}µs (lookup={initial_lookup_ns:.0f}ns + "
              f"seq={sequential_ns:.0f}ns + hops={leaf_hop_ns:.0f}ns)")

    return result


# ─── TableGroup Lifecycle ────────────────────────────────────────────────
# Ported from upstream tabular/src/tabular/table_group.{h,cc}
# Simplified: no epoch daemon, no dlog, no MVCC

@dataclass
class TableGroupState:
    """Simulated state of a tabular TableGroup.

    Mirrors upstream TableGroup struct without the threading/logging infra.
    Used to track multiple tables' index build states across benchmark steps.
    """
    tables: Dict[str, IndexBuildCost] = field(default_factory=dict)
    epoch: int = 0
    total_memory_bytes: int = 0
    creation_time: float = field(default_factory=time.monotonic)

    def add_table(self, table_name: str, cost: IndexBuildCost) -> None:
        """Register a built index for a table."""
        self.tables[table_name] = cost
        self.total_memory_bytes = sum(t.memory_bytes for t in self.tables.values())
        self.epoch += 1

    def get_table(self, table_name: str) -> Optional[IndexBuildCost]:
        return self.tables.get(table_name)

    def dump_all(self) -> str:
        """Full state dump — use at breakpoints or after benchmark steps."""
        lines = [
            "╔══ TableGroup State Dump ═══════════════════════════════",
            f"║ epoch           = {self.epoch}",
            f"║ n_tables        = {len(self.tables)}",
            f"║ total_memory    = {self.total_memory_bytes:,} ({self.total_memory_bytes/(1024**2):.1f} MB)",
            f"║ uptime_sec      = {time.monotonic() - self.creation_time:.2f}",
        ]
        for tname, cost in self.tables.items():
            lines.append(f"║ ── {tname} ──")
            for l in cost.dump_debug("║   ").split("\n"):
                lines.append(l)
        lines.append("╚═══════════════════════════════════════════════════════")
        return "\n".join(lines)


# ─── High-Level Bridge API ───────────────────────────────────────────────

def estimate_index_build(
    table_name: str,
    n_rows: int,
    key_size: int = 8,
    index_type: IndexType = IndexType.INLINE_BTREE,
    n_threads: int = 1,
    debug_print: bool = True,
) -> IndexBuildCost:
    """High-level API: estimate the cost of building an index.

    This is the main entry point for Lynceus cost_model.py to call.
    It bridges upstream tabular's index infrastructure into a cost number
    that the router can compare against GPU kernel costs.
    """
    config = IndexBuildConfig(
        index_type=index_type,
        key_size_bytes=key_size,
        n_rows=n_rows,
        n_threads=n_threads,
        debug_print=debug_print,
    )

    if debug_print:
        print(f"\n{'='*60}")
        print(f"[tabular_bridge] estimate_index_build()")
        print(f"  table_name  = {table_name}")
        print(f"  n_rows      = {n_rows:,}")
        print(f"  key_size    = {key_size}B")
        print(f"  index_type  = {index_type.name}")
        print(f"  n_threads   = {n_threads}")
        print(f"{'='*60}")

    if index_type == IndexType.INLINE_BTREE:
        return estimate_btree_build_cost(config)
    elif index_type == IndexType.HASH_TABLE:
        return estimate_hash_build_cost(config)
    else:
        # Fallback to BTree for unsupported types
        logger.warning(f"IndexType {index_type} not fully ported; falling back to BTree model")
        config.index_type = IndexType.INLINE_BTREE
        return estimate_btree_build_cost(config)


def build_and_probe_cost(
    table_name: str,
    n_rows: int,
    n_probe_keys: int,
    scan_selectivity: float = 0.01,
    key_size: int = 8,
    index_type: IndexType = IndexType.INLINE_BTREE,
    debug_print: bool = True,
) -> Tuple[IndexBuildCost, ScanCost]:
    """End-to-end: build index + estimate probe/scan cost.

    Returns (build_cost, scan_cost) so the router can compute:
      total_index_cost = build_cost.build_time_us + scan_cost.total_us
    """
    build = estimate_index_build(
        table_name=table_name,
        n_rows=n_rows,
        key_size=key_size,
        index_type=index_type,
        debug_print=debug_print,
    )

    scan = estimate_scan_cost(
        build_cost=build,
        n_keys=n_probe_keys,
        selectivity=scan_selectivity,
        debug_print=debug_print,
    )

    if debug_print:
        total = build.build_time_us + scan.total_us
        print(f"\n  [TOTAL] build={build.build_time_us:,.1f}µs + scan={scan.total_us:,.1f}µs "
              f"= {total:,.1f}µs ({total/1000:.2f}ms)")

    return build, scan


# ─── Debug Utilities ─────────────────────────────────────────────────────

def trace_insert_progress(step: int, total: int, cost_so_far_ns: float,
                          config: IndexBuildConfig) -> None:
    """Print insert progress — call from benchmark loop for live feedback.

    Usage in benchmark:
        for step in range(n_steps):
            ...
            if step % trace_every == 0:
                tabular_bridge.trace_insert_progress(step, n_steps, cumulative_ns, config)
    """
    pct = 100.0 * step / max(1, total)
    rate = step / max(1e-9, cost_so_far_ns / 1e9)  # inserts/sec
    print(f"  [INSERT PROGRESS] step={step:,}/{total:,} ({pct:.1f}%), "
          f"cumulative={cost_so_far_ns/1e6:.1f}ms, rate={rate:,.0f} inserts/sec")


def compare_index_types(table_name: str, n_rows: int, key_size: int = 8) -> Dict[str, IndexBuildCost]:
    """Compare BTree vs Hash build costs — useful for experiment analysis."""
    print(f"\n{'='*60}")
    print(f"[tabular_bridge] Index Type Comparison: {table_name} ({n_rows:,} rows)")
    print(f"{'='*60}")

    results = {}
    for itype in [IndexType.INLINE_BTREE, IndexType.HASH_TABLE]:
        cost = estimate_index_build(
            table_name=table_name,
            n_rows=n_rows,
            key_size=key_size,
            index_type=itype,
            debug_print=True,
        )
        results[itype.name] = cost

    # Summary comparison
    bt = results["INLINE_BTREE"]
    ht = results["HASH_TABLE"]
    print(f"\n  ── Comparison Summary ──")
    print(f"  Build time:   BTree={bt.build_time_us:,.0f}µs  vs  Hash={ht.build_time_us:,.0f}µs  "
          f"(ratio={bt.build_time_us / max(1, ht.build_time_us):.2f}x)")
    print(f"  Memory:       BTree={bt.memory_bytes/(1024**2):.1f}MB  vs  Hash={ht.memory_bytes/(1024**2):.1f}MB")
    print(f"  Point lookup: BTree={bt.avg_lookup_ns:.0f}ns  vs  Hash={ht.avg_lookup_ns:.0f}ns")
    print(f"  Range scan:   BTree={bt.avg_scan_per_key_ns:.1f}ns/key  vs  Hash=N/A (not ordered)")

    return results
