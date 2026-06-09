"""
tests/test_m001_m002_validation.py — Validation for lynceus_port M001-M002 deliverables.

Same tests as test_m001_m002.py but imports from lynceus_port.
"""

import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lynceus_port.schema import (
    BenchmarkOutput, HardwareKind, HardwareNode, HardwareTopology,
    MetricKind, MethodResult, PanelResult, RoutingStrategy, SeedCurve,
    TopologyEdge,
)
from lynceus_port.costing import (
    CostBreakdown, CostModelEngine, CPUCostModel, GPUCostModel,
    QueryDescriptor, QueryType, create_default_topology,
)
from lynceus_port.benchmark import (
    WorkloadConfig, generate_query_sequence, run_benchmark,
    StrategyExecutor,
)


def test_seed_curve_append():
    sc = SeedCurve(seed_id=0)
    sc.append(1.0)
    sc.append(2.0)
    assert len(sc.values) == 2
    assert sc.values == [1.0, 2.0]
    print("  PASS: SeedCurve append")


def test_method_result_statistics():
    mr = MethodResult(
        strategy=RoutingStrategy.GPU_ONLY,
        num_steps=3,
        num_seeds=2,
    )
    s0 = mr.add_seed()
    s0.values = [10.0, 20.0, 30.0]
    s1 = mr.add_seed()
    s1.values = [12.0, 22.0, 28.0]

    mr.compute_statistics()
    assert len(mr.mean) == 3
    assert abs(mr.mean[0] - 11.0) < 1e-9
    assert abs(mr.mean[1] - 21.0) < 1e-9
    assert abs(mr.mean[2] - 29.0) < 1e-9
    assert mr.std[0] > 0
    assert mr.reported_final == mr.mean[-1]
    print("  PASS: MethodResult statistics")


def test_method_result_serialization():
    mr = MethodResult(
        strategy=RoutingStrategy.CPU_ONLY,
        num_steps=2,
        num_seeds=1,
    )
    s0 = mr.add_seed()
    s0.values = [5.0, 10.0]
    mr.total_cost = 15.0

    d = mr.to_dict()
    assert "seed_0" in d
    assert "mean" in d
    assert "std" in d
    assert d["total_cost"] == 15.0
    print("  PASS: MethodResult serialization")


def test_benchmark_output_save_load():
    output = BenchmarkOutput(description="test", source="unit_test")
    panel = output.add_panel("test_panel", MetricKind.LATENCY_MS)
    mr = panel.add_method(RoutingStrategy.GPU_ONLY, num_steps=5, num_seeds=1)
    sc = mr.add_seed()
    sc.values = [1.0, 2.0, 3.0, 4.0, 5.0]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        output.save(path)
        with open(path) as f:
            loaded = json.load(f)

        assert "metadata" in loaded
        assert "panels" in loaded
        assert loaded["metadata"]["n_per_seed"] == 5
        assert loaded["metadata"]["total_data_points"] == 5
        assert "test_panel" in loaded["panels"]
        panel_data = loaded["panels"]["test_panel"]
        assert "GPU-Only" in panel_data
        assert panel_data["GPU-Only"]["seed_0"] == [1.0, 2.0, 3.0, 4.0, 5.0]
        print("  PASS: BenchmarkOutput save/load")
    finally:
        os.unlink(path)


def test_topology_transfer_cost():
    topo = create_default_topology()

    cost = topo.get_transfer_cost("cpu0", "cpu0", 1000)
    assert cost == 0.0

    cost = topo.get_transfer_cost("cpu0", "gpu0", 1_000_000)
    assert cost > 0
    assert cost < 1e9

    cost = topo.get_transfer_cost("cpu0", "nonexistent", 1000)
    assert cost == float("inf")
    print("  PASS: Topology transfer cost")


def test_cpu_costing_basic():
    topo = create_default_topology()
    model = CPUCostModel()
    node = topo.get_node("cpu0")

    q_point = QueryDescriptor(
        query_id="point", query_type=QueryType.POINT_LOOKUP,
        estimated_rows=1, selectivity=0.000001,
        table_rows=1_000_000, index_available=True,
    )
    cb_point = model.estimate(q_point, node)
    assert cb_point.total_us > 0

    q_scan = QueryDescriptor(
        query_id="scan", query_type=QueryType.FULL_TABLE_SCAN,
        estimated_rows=1_000_000, selectivity=1.0,
        table_rows=1_000_000,
    )
    cb_scan = model.estimate(q_scan, node)
    assert cb_scan.total_us > cb_point.total_us
    print("  PASS: CPU cost model (point < scan)")


def test_gpu_costing_basic():
    topo = create_default_topology()
    model = GPUCostModel()
    node = topo.get_node("gpu0")

    q_small = QueryDescriptor(
        query_id="small", query_type=QueryType.POINT_LOOKUP,
        estimated_rows=10, selectivity=0.00001,
        table_rows=1_000_000,
    )
    cb_small = model.estimate(q_small, node, data_resident_on_gpu=False)
    assert cb_small.transfer_cost_us > 0

    q_large = QueryDescriptor(
        query_id="large", query_type=QueryType.FULL_TABLE_SCAN,
        estimated_rows=1_000_000, selectivity=1.0,
        table_rows=1_000_000, estimated_width_bytes=100,
    )
    cb_large = model.estimate(q_large, node, data_resident_on_gpu=True)
    assert cb_large.transfer_cost_us == 0.0
    print("  PASS: GPU cost model (transfer + no-transfer)")


def test_costing_routing_logic():
    topo = create_default_topology()
    engine = CostModelEngine(topo)

    q_large = QueryDescriptor(
        query_id="large", query_type=QueryType.FULL_TABLE_SCAN,
        estimated_rows=5_000_000, selectivity=1.0,
        table_rows=5_000_000, estimated_width_bytes=200,
    )
    device, cb = engine.recommend(q_large, "cpu0")
    assert device is not None
    assert cb.total_us > 0

    q_tiny = QueryDescriptor(
        query_id="tiny", query_type=QueryType.POINT_LOOKUP,
        estimated_rows=1, selectivity=0.0000001,
        table_rows=10_000_000, index_available=True,
        estimated_width_bytes=50,
    )
    device, cb = engine.recommend(q_tiny, "cpu0")
    assert device.startswith("cpu"), f"Expected CPU for tiny query, got {device}"
    print("  PASS: Routing logic (large→any, tiny→CPU)")


def test_seed_reproducibility():
    config = WorkloadConfig(num_steps=100, num_seeds=1)
    q1 = generate_query_sequence(config, seed=42)
    q2 = generate_query_sequence(config, seed=42)

    assert len(q1) == len(q2)
    for a, b in zip(q1, q2):
        assert a.query_type == b.query_type
        assert a.estimated_rows == b.estimated_rows
        assert a.selectivity == b.selectivity

    q3 = generate_query_sequence(config, seed=999)
    differs = any(
        a.estimated_rows != b.estimated_rows
        for a, b in zip(q1, q3)
    )
    assert differs, "Different seeds should produce different queries"
    print("  PASS: Seed reproducibility")


def test_zero_row_query():
    topo = create_default_topology()
    engine = CostModelEngine(topo)

    q = QueryDescriptor(
        query_id="zero", query_type=QueryType.POINT_LOOKUP,
        estimated_rows=0, selectivity=0.0,
        table_rows=1_000_000,
    )
    device, cb = engine.recommend(q, "cpu0")
    assert cb.total_us >= 0
    print("  PASS: Zero-row query edge case")


def test_bug1_seed_overflow():
    mr = MethodResult(
        strategy=RoutingStrategy.GPU_ONLY, num_steps=3, num_seeds=2,
    )
    mr.add_seed()
    mr.add_seed()
    try:
        mr.add_seed()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "num_seeds is 2" in str(e)
    print("  PASS: Bug #1 fix — seed overflow prevented")


def test_bug2_mismatched_curve_length():
    mr = MethodResult(
        strategy=RoutingStrategy.CPU_ONLY, num_steps=3, num_seeds=2,
    )
    s0 = mr.add_seed()
    s0.values = [1, 2, 3]
    s1 = mr.add_seed()
    s1.values = [4, 5]
    try:
        mr.compute_statistics()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "expected 3" in str(e)
    print("  PASS: Bug #2 fix — mismatched curve length rejected")


def test_bug3_negative_rows():
    try:
        QueryDescriptor(
            query_id="neg", query_type=QueryType.POINT_LOOKUP,
            estimated_rows=-5,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "estimated_rows" in str(e)
    print("  PASS: Bug #3 fix — negative rows rejected")


if __name__ == "__main__":
    print("Running lynceus_port M001-M002 tests...")
    test_seed_curve_append()
    test_method_result_statistics()
    test_method_result_serialization()
    test_benchmark_output_save_load()
    test_topology_transfer_cost()
    test_cpu_costing_basic()
    test_gpu_costing_basic()
    test_costing_routing_logic()
    test_seed_reproducibility()
    test_zero_row_query()
    test_bug1_seed_overflow()
    test_bug2_mismatched_curve_length()
    test_bug3_negative_rows()
    print("\nAll lynceus_port M001-M002 tests PASSED.")
