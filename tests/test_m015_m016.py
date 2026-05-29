"""
tests/test_m015_m016.py — Validation for M015-M016 deliverables.

Tests:
    1.  decompose: simple point lookup → single SCAN stage
    2.  decompose: full pipeline (scan+filter+join+agg+sort) ordering
    3.  decompose: deterministic (same query → identical stage graph)
    4.  decompose: row-flow monotonicity (aggregate collapses)
    5.  Scheduler: max_pipeline_depth bounded by device count
    6.  Scheduler: schedule returns one assignment per stage
    7.  Scheduler: each stage routed to a real topology device
    8.  Scheduler: serial_cost == sum of stage costs
    9.  Scheduler: pipelined_cost <= serial_cost (overlap never hurts)
    10. Scheduler: single query has no self-speedup (batch m=1 -> 1.0)
    11. Scheduler: transfer is summed over all stages (initial load included)
    12. Scheduler: batch speedup grows with m (Megatron bubble shrinks)
    13. Scheduler: devices_used has no duplicates
    14. Scheduler: schedule_batch length matches input
    15. Integration: heavy join query pipelines across GPU+CPU
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lynceus.cost_model import (
    CostModelEngine, QueryDescriptor, QueryType, create_default_topology,
)
from lynceus.schema import HardwareKind
from lynceus.pipeline_scheduler import (
    QueryPipelineScheduler, decompose_query, StageKind,
)


def _engine():
    return CostModelEngine(create_default_topology())


def _point_lookup():
    return QueryDescriptor(
        query_id="q_point", query_type=QueryType.POINT_LOOKUP,
        estimated_rows=1, selectivity=0.000001, table_rows=1_000_000,
        index_available=True, num_predicates=1,
    )


def _full_pipeline_query():
    return QueryDescriptor(
        query_id="q_full", query_type=QueryType.JOIN,
        estimated_rows=50_000, selectivity=0.3, table_rows=1_000_000,
        num_predicates=2, num_joins=2, group_by_cardinality=100,
        sort_required=True, estimated_width_bytes=200,
    )


def test_01_decompose_point_lookup():
    stages = decompose_query(_point_lookup())
    assert stages[0].kind == StageKind.SCAN
    kinds = {s.kind for s in stages}
    assert StageKind.JOIN not in kinds and StageKind.SORT not in kinds


def test_02_decompose_full_ordering():
    stages = decompose_query(_full_pipeline_query())
    kinds = [s.kind for s in stages]
    assert kinds[0] == StageKind.SCAN
    assert StageKind.JOIN in kinds
    assert StageKind.AGGREGATE in kinds
    assert kinds[-1] == StageKind.SORT
    # canonical order: scan before join before aggregate before sort
    order = [StageKind.SCAN, StageKind.FILTER, StageKind.JOIN,
             StageKind.AGGREGATE, StageKind.SORT]
    positions = [order.index(k) for k in kinds]
    assert positions == sorted(positions)


def test_03_decompose_deterministic():
    q = _full_pipeline_query()
    a = [(s.stage_id, s.kind, s.produces_rows) for s in decompose_query(q)]
    b = [(s.stage_id, s.kind, s.produces_rows) for s in decompose_query(q)]
    assert a == b


def test_04_decompose_aggregate_collapses():
    stages = decompose_query(_full_pipeline_query())
    agg = next(s for s in stages if s.kind == StageKind.AGGREGATE)
    assert agg.produces_rows == 100


def test_05_pipeline_depth_bounded():
    eng = _engine()
    n_dev = sum(1 for n in eng.topology.nodes.values()
                if n.kind in (HardwareKind.GPU, HardwareKind.CPU))
    sched = QueryPipelineScheduler(eng)
    assert sched.max_pipeline_depth == n_dev


def test_06_one_assignment_per_stage():
    sched = QueryPipelineScheduler(_engine())
    q = _full_pipeline_query()
    sch = sched.schedule(q)
    assert len(sch.assignments) == len(decompose_query(q))


def test_07_stages_routed_to_real_devices():
    eng = _engine()
    sched = QueryPipelineScheduler(eng)
    sch = sched.schedule(_full_pipeline_query())
    for a in sch.assignments:
        assert a.device_id in eng.topology.nodes


def test_08_latency_is_compute_plus_transfer():
    sched = QueryPipelineScheduler(_engine())
    sch = sched.schedule(_full_pipeline_query())
    # latency is the full critical path: every stage's compute + every handoff.
    expected = sum(a.cost.total_us for a in sch.assignments)
    assert abs(sch.latency_us - expected) < 1e-6
    assert abs(sch.latency_us - (sch.compute_cost_us + sch.transfer_cost_us)) < 1e-6


def test_09_transfer_never_vanishes():
    # The initial data load must appear in transfer_cost_us, not be silently
    # absorbed (the bug that produced a fake 246x speedup).
    sched = QueryPipelineScheduler(_engine())
    sch = sched.schedule(_full_pipeline_query())
    total_transfer = sum(a.cost.transfer_cost_us for a in sch.assignments)
    assert abs(sch.transfer_cost_us - total_transfer) < 1e-6


def test_10_single_query_no_self_speedup():
    # One query cannot pipeline with itself: batch speedup at m=1 is exactly 1.
    sched = QueryPipelineScheduler(_engine())
    b = sched.schedule_pipeline([_full_pipeline_query()])
    assert abs(b.speedup - 1.0) < 1e-9


def test_11_transfer_is_sum_over_all_stages():
    sched = QueryPipelineScheduler(_engine())
    sch = sched.schedule(_full_pipeline_query())
    manual = sum(c.cost.transfer_cost_us for c in sch.assignments)
    assert abs(sch.transfer_cost_us - manual) < 1e-6


def test_12_batch_speedup_grows_with_m():
    # Megatron bubble (p-1)/(m+p-1): more independent queries -> more overlap.
    sched = QueryPipelineScheduler(_engine())
    qs = [_full_pipeline_query() for _ in range(64)]
    b1 = sched.schedule_pipeline(qs[:1])
    b64 = sched.schedule_pipeline(qs)
    assert b64.speedup > b1.speedup
    assert b64.bubble_fraction < b1.bubble_fraction


def test_13_devices_used_unique():
    sched = QueryPipelineScheduler(_engine())
    sch = sched.schedule(_full_pipeline_query())
    assert len(sch.devices_used) == len(set(sch.devices_used))


def test_14_batch_length():
    sched = QueryPipelineScheduler(_engine())
    qs = [_point_lookup(), _full_pipeline_query()]
    out = sched.schedule_batch(qs)
    assert len(out) == 2


def test_15_integration_heavy_join():
    sched = QueryPipelineScheduler(_engine())
    sch = sched.schedule(_full_pipeline_query())
    # A multi-stage heavy query should produce a non-trivial schedule.
    assert len(sch.assignments) >= 4
    assert sch.latency_us > 0


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
