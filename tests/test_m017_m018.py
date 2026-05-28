"""
tests/test_m017_m018.py — Validation for M017-M018 deliverables.

Tests:
    1.  required_blocks: count = ceil(footprint / block_bytes)
    2.  required_blocks: capped at cache size
    3.  required_blocks: same query id → same keys (sharing)
    4.  lookup: cold cache is all misses, then admitted
    5.  lookup: warm cache hits (second pass after release)
    6.  is_resident: false before, true after admission
    7.  Pinning: in-use block is never evicted
    8.  Eviction: LRU victim chosen when pool full
    9.  Eviction: eviction counter increments
    10. Stats: hit_rate computed correctly
    11. release: idle block becomes eviction candidate
    12. reset: clears table and stats
    13. Topology: one cache per GPU, none for CPU
    14. Topology: cache sized from memory_bytes * fraction
    15. Integration: repeated workload converges to high hit rate
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lynceus.cost_model import (
    QueryDescriptor, QueryType, create_default_topology,
)
from lynceus.schema import HardwareKind
from lynceus.cache_manager import (
    IndexCacheManager, TopologyCacheManager, BlockKey,
)


def _q(qid="qA", rows=10_000, width=200, index=True, table_rows=1_000_000):
    return QueryDescriptor(
        query_id=qid, query_type=QueryType.INDEX_SCAN,
        estimated_rows=rows, estimated_width_bytes=width,
        selectivity=min(1.0, rows / table_rows), table_rows=table_rows,
        index_available=index, num_predicates=1,
    )


def test_01_required_block_count():
    c = IndexCacheManager("gpu0", capacity_bytes=100 * (1 << 20),
                          block_bytes=1 << 20)  # 1 MiB blocks, 100 slots
    q = _q(rows=10_000, width=1000)  # ~10 MB footprint
    blocks = c.required_blocks(q)
    assert len(blocks) == max(1, -(-(10_000 * 1000) // (1 << 20)))


def test_02_required_blocks_capped():
    c = IndexCacheManager("gpu0", capacity_bytes=4 * (1 << 20),
                          block_bytes=1 << 20)  # only 4 slots
    q = _q(rows=10_000_000, width=1000, index=False)  # huge
    assert len(c.required_blocks(q)) <= c.num_blocks


def test_03_same_query_same_keys():
    c = IndexCacheManager("gpu0", 100 * (1 << 20), 1 << 20)
    a = c.required_blocks(_q("orders_1"))
    b = c.required_blocks(_q("orders_2"))
    assert a == b  # trailing digits stripped → same table


def test_04_cold_all_miss():
    c = IndexCacheManager("gpu0", 100 * (1 << 20), 1 << 20)
    blocks = c.required_blocks(_q())
    hits, misses = c.lookup(blocks)
    assert hits == 0 and misses == len(blocks)


def test_05_warm_hits():
    c = IndexCacheManager("gpu0", 100 * (1 << 20), 1 << 20)
    blocks = c.required_blocks(_q())
    c.lookup(blocks); c.release(blocks)
    hits, misses = c.lookup(blocks)
    assert misses == 0 and hits == len(blocks)


def test_06_is_resident():
    c = IndexCacheManager("gpu0", 100 * (1 << 20), 1 << 20)
    q = _q()
    assert not c.is_resident(q)
    c.lookup(c.required_blocks(q))
    assert c.is_resident(q)


def test_07_pinned_not_evicted():
    # 2 slots; pin 2 blocks, then try to admit a 3rd → cannot evict pinned.
    c = IndexCacheManager("gpu0", 2 * (1 << 20), 1 << 20)
    a = [BlockKey("t", "idx", 0), BlockKey("t", "idx", 1)]
    c.lookup(a)  # pins both
    new = [BlockKey("t", "idx", 2)]
    c.lookup(new)  # miss, but no evictable slot
    assert a[0] in c._table and a[1] in c._table  # both pinned, still resident


def test_08_lru_victim():
    c = IndexCacheManager("gpu0", 2 * (1 << 20), 1 << 20)
    k0 = [BlockKey("t", "idx", 0)]
    k1 = [BlockKey("t", "idx", 1)]
    k2 = [BlockKey("t", "idx", 2)]
    c.lookup(k0); c.release(k0)   # idle, oldest
    c.lookup(k1); c.release(k1)   # idle, newer
    c.lookup(k2)                  # needs slot → evicts k0 (LRU)
    assert k0[0] not in c._table
    assert k1[0] in c._table and k2[0] in c._table


def test_09_eviction_counter():
    c = IndexCacheManager("gpu0", 1 * (1 << 20), 1 << 20)  # 1 slot
    a = [BlockKey("t", "idx", 0)]
    b = [BlockKey("t", "idx", 1)]
    c.lookup(a); c.release(a)
    c.lookup(b)  # evicts a
    assert c.stats.evictions >= 1


def test_10_hit_rate():
    c = IndexCacheManager("gpu0", 100 * (1 << 20), 1 << 20)
    blocks = c.required_blocks(_q())
    c.lookup(blocks); c.release(blocks)  # n misses
    c.lookup(blocks)                     # n hits
    assert abs(c.stats.hit_rate - 0.5) < 1e-9


def test_11_release_makes_evictable():
    c = IndexCacheManager("gpu0", 1 * (1 << 20), 1 << 20)
    a = [BlockKey("t", "idx", 0)]
    b = [BlockKey("t", "idx", 1)]
    c.lookup(a)            # pinned, sole slot
    c.lookup(b)            # cannot evict pinned a
    assert b[0] not in c._table
    c.release(a)           # now a is evictable
    c.lookup(b)            # evicts a, admits b
    assert b[0] in c._table


def test_12_reset():
    c = IndexCacheManager("gpu0", 100 * (1 << 20), 1 << 20)
    c.lookup(c.required_blocks(_q()))
    c.reset()
    assert c.resident_blocks == 0 and c.stats.lookups == 0


def test_13_one_cache_per_gpu():
    tcm = TopologyCacheManager(create_default_topology())
    assert set(tcm.caches.keys()) == {"gpu0", "gpu1", "gpu2", "gpu3"}
    assert tcm.get("cpu0") is None


def test_14_cache_sized_from_hbm():
    topo = create_default_topology()
    tcm = TopologyCacheManager(topo, cache_fraction=0.5,
                               block_bytes=2 * (1 << 20))
    hbm = topo.get_node("gpu0").memory_bytes
    expected = int(hbm * 0.5) // (2 * (1 << 20))
    assert tcm.get("gpu0").num_blocks == expected


def test_15_workload_converges():
    c = IndexCacheManager("gpu0", 200 * (1 << 20), 1 << 20)
    blocks = c.required_blocks(_q())
    for _ in range(20):
        c.lookup(blocks)
        c.release(blocks)
    # First pass missed, the other 19 hit → high steady-state hit rate.
    assert c.stats.hit_rate > 0.9


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
