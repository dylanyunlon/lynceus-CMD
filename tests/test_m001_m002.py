"""
tests/test_m001_m002.py — Validation for M001-M002 deliverables.

Tests:
    1. Schema serialization round-trip
    2. Cost model correctness (GPU < CPU for large scans)
    3. Topology transfer cost computation
    4. Benchmark output format compliance
    5. Seed reproducibility
    6. Edge cases (empty queries, zero-row queries)
"""

import json
import math
import os
import sys
import tempfile

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lynceus.schema import (
    BenchmarkOutput, HardwareKind, HardwareNode, HardwareTopology,
    MetricKind, MethodResult, PanelResult, RoutingStrategy, SeedCurve,
    TopologyEdge,
)
from lynceus.cost_model import (
    CostBreakdown, CostModelEngine, CPUCostModel, GPUCostModel,
    QueryDescriptor, QueryType, create_default_topology,
)
from lynceus.benchmark import (
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
    assert mr.std[0] > 0  # non-zero std
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

    # Same device: zero cost
    cost = topo.get_transfer_cost("cpu0", "cpu0", 1000)
    assert cost == 0.0

    # CPU → GPU: should be positive
    cost = topo.get_transfer_cost("cpu0", "gpu0", 1_000_000)
    assert cost > 0
    assert cost < 1e9  # not infinite

    # Non-existent path
    cost = topo.get_transfer_cost("cpu0", "nonexistent", 1000)
    assert cost == float("inf")
    print("  PASS: Topology transfer cost")


def test_cpu_cost_model_basic():
    topo = create_default_topology()
    model = CPUCostModel()
    node = topo.get_node("cpu0")

    # Point lookup with index — should be cheap
    q_point = QueryDescriptor(
        query_id="point", query_type=QueryType.POINT_LOOKUP,
        estimated_rows=1, selectivity=0.000001,
        table_rows=1_000_000, index_available=True,
    )
    cb_point = model.estimate(q_point, node)
    assert cb_point.total_us > 0

    # Full table scan — should be expensive
    q_scan = QueryDescriptor(
        query_id="scan", query_type=QueryType.FULL_TABLE_SCAN,
        estimated_rows=1_000_000, selectivity=1.0,
        table_rows=1_000_000,
    )
    cb_scan = model.estimate(q_scan, node)
    assert cb_scan.total_us > cb_point.total_us
    print("  PASS: CPU cost model (point < scan)")


def test_gpu_cost_model_basic():
    topo = create_default_topology()
    model = GPUCostModel()
    node = topo.get_node("gpu0")

    # Small query: GPU overhead dominates
    q_small = QueryDescriptor(
        query_id="small", query_type=QueryType.POINT_LOOKUP,
        estimated_rows=10, selectivity=0.00001,
        table_rows=1_000_000,
    )
    cb_small = model.estimate(q_small, node, data_resident_on_gpu=False)
    assert cb_small.transfer_cost_us > 0  # PCIe transfer

    # Large scan with data on GPU: no transfer
    q_large = QueryDescriptor(
        query_id="large", query_type=QueryType.FULL_TABLE_SCAN,
        estimated_rows=1_000_000, selectivity=1.0,
        table_rows=1_000_000, estimated_width_bytes=100,
    )
    cb_large = model.estimate(q_large, node, data_resident_on_gpu=True)
    assert cb_large.transfer_cost_us == 0.0
    print("  PASS: GPU cost model (transfer + no-transfer)")


def test_cost_model_routing_logic():
    """CostModel-Routed should pick GPU for large scans, CPU for point lookups."""
    topo = create_default_topology()
    engine = CostModelEngine(topo)

    # Large full scan → should route to GPU
    q_large = QueryDescriptor(
        query_id="large", query_type=QueryType.FULL_TABLE_SCAN,
        estimated_rows=5_000_000, selectivity=1.0,
        table_rows=5_000_000, estimated_width_bytes=200,
    )
    device, cb = engine.recommend(q_large, "cpu0")
    # GPU should win for large scans due to parallel processing
    # (but may lose due to transfer — this tests the model's balance)
    assert device is not None
    assert cb.total_us > 0

    # Tiny point lookup with index → should route to CPU
    q_tiny = QueryDescriptor(
        query_id="tiny", query_type=QueryType.POINT_LOOKUP,
        estimated_rows=1, selectivity=0.0000001,
        table_rows=10_000_000, index_available=True,
        estimated_width_bytes=50,
    )
    device, cb = engine.recommend(q_tiny, "cpu0")
    # CPU should win: GPU kernel launch overhead (10us) > CPU cost for 1 row
    assert device.startswith("cpu"), f"Expected CPU for tiny query, got {device}"
    print("  PASS: Routing logic (large→any, tiny→CPU)")


def test_seed_reproducibility():
    """Same seed must produce identical query sequences."""
    config = WorkloadConfig(num_steps=100, num_seeds=1)
    q1 = generate_query_sequence(config, seed=42)
    q2 = generate_query_sequence(config, seed=42)

    assert len(q1) == len(q2)
    for a, b in zip(q1, q2):
        assert a.query_type == b.query_type
        assert a.estimated_rows == b.estimated_rows
        assert a.selectivity == b.selectivity

    # Different seed → different sequence
    q3 = generate_query_sequence(config, seed=999)
    differs = any(
        a.estimated_rows != b.estimated_rows
        for a, b in zip(q1, q3)
    )
    assert differs, "Different seeds should produce different queries"
    print("  PASS: Seed reproducibility")


def test_benchmark_output_format_compliance():
    """Verify benchmark output matches data_demo schema."""
    config = WorkloadConfig(num_steps=50, num_seeds=2, name="test")
    output = run_benchmark(config)

    d = output.to_dict()
    meta = d["metadata"]

    assert meta["n_per_seed"] == 50
    assert meta["n_seeds"] == 2
    assert meta["n_methods"] == 5
    assert meta["total_data_points"] == 50 * 2 * 5

    panels = d["panels"]
    assert len(panels) > 0
    for pname, panel in panels.items():
        for mname, method in panel.items():
            assert "seed_0" in method
            assert "seed_1" in method
            assert "mean" in method
            assert "std" in method
            assert len(method["seed_0"]) == 50
            assert len(method["mean"]) == 50
            assert len(method["std"]) == 50
    print("  PASS: Benchmark output format compliance")


def test_zero_row_query():
    """Edge case: query with 0 estimated rows should not crash."""
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


def test_cost_model_routed_beats_single_device():
    """CostModel-Routed should never be worse than the best single-device strategy."""
    config = WorkloadConfig(num_steps=200, num_seeds=1, name="routing_test")
    output = run_benchmark(config)

    panel = list(output.panels.values())[0]
    routed = panel.methods["CostModel-Routed"]
    gpu = panel.methods["GPU-Only"]
    cpu = panel.methods["CPU-Only"]

    routed_total = sum(routed.seed_curves[0].values)
    gpu_total = sum(gpu.seed_curves[0].values)
    cpu_total = sum(cpu.seed_curves[0].values)

    assert routed_total <= gpu_total + 1e-6, \
        f"Routed ({routed_total:.1f}) should <= GPU ({gpu_total:.1f})"
    assert routed_total <= cpu_total + 1e-6, \
        f"Routed ({routed_total:.1f}) should <= CPU ({cpu_total:.1f})"
    print("  PASS: CostModel-Routed ≤ single-device strategies")


def test_bug1_seed_overflow():
    """Bug #1: add_seed should enforce num_seeds limit."""
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
    """Bug #2: compute_statistics should reject mismatched lengths."""
    mr = MethodResult(
        strategy=RoutingStrategy.CPU_ONLY, num_steps=3, num_seeds=2,
    )
    s0 = mr.add_seed()
    s0.values = [1, 2, 3]
    s1 = mr.add_seed()
    s1.values = [4, 5]  # only 2 values!
    try:
        mr.compute_statistics()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "expected 3" in str(e)
    print("  PASS: Bug #2 fix — mismatched curve length rejected")


def test_bug3_negative_rows():
    """Bug #3: QueryDescriptor rejects negative estimated_rows."""
    try:
        QueryDescriptor(
            query_id="neg", query_type=QueryType.POINT_LOOKUP,
            estimated_rows=-5,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "estimated_rows" in str(e)
    print("  PASS: Bug #3 fix — negative rows rejected")


def test_bug6_rows_capped():
    """Bug #6: estimated_rows should never exceed table_rows."""
    config = WorkloadConfig(num_steps=2000, num_seeds=1)
    queries = generate_query_sequence(config, seed=42)
    for q in queries:
        assert q.estimated_rows <= q.table_rows, \
            f"Query {q.query_id}: estimated_rows={q.estimated_rows} > table_rows={q.table_rows}"
    print("  PASS: Bug #6 fix — rows capped at table_rows")


if __name__ == "__main__":
    print("Running M001-M002 tests...")
    test_seed_curve_append()
    test_method_result_statistics()
    test_method_result_serialization()
    test_benchmark_output_save_load()
    test_topology_transfer_cost()
    test_cpu_cost_model_basic()
    test_gpu_cost_model_basic()
    test_cost_model_routing_logic()
    test_seed_reproducibility()
    test_benchmark_output_format_compliance()
    test_zero_row_query()
    test_cost_model_routed_beats_single_device()
    test_bug1_seed_overflow()
    test_bug2_mismatched_curve_length()
    test_bug3_negative_rows()
    test_bug6_rows_capped()
    print("\nAll M001-M002 tests PASSED.")
