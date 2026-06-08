"""
tests/test_m003_m004.py — Validation for M003-M004 deliverables.

Tests:
    1.  Router: registration and lookup
    2.  Router: duplicate registration rejected
    3.  Router: set_active dispatch
    4.  Router: create_default registers all 6 strategies
    5.  Router: run_all_strategies returns all results
    6.  Strategy: GPUOnlyStrategy routes to gpu0
    7.  Strategy: CPUOnlyStrategy routes to cpu0
    8.  Strategy: HybridStaticStrategy threshold logic
    9.  Strategy: CostModelRoutedStrategy == CostModelEngine.recommend
    10. Strategy: PAR2QOEnhancedStrategy robustness margin
    11. Strategy: AdaptiveStrategy warmup phase
    12. Strategy: AdaptiveStrategy EMA bias correction
    13. Strategy: AdaptiveStrategy load balancing across GPUs
    14. Backward compat: StrategyExecutor still works
    15. Backward compat: benchmark output unchanged for 5 original strategies
    16. Integration: full benchmark with 6 strategies
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lynceus.schema import RoutingStrategy
from lynceus.costing import (
    CostModelEngine, QueryDescriptor, QueryType, create_default_topology,
)
from lynceus.router import Router
from lynceus.strategies import (
    RoutingDecision, RoutingStrategyBase,
    GPUOnlyStrategy, CPUOnlyStrategy, HybridStaticStrategy,
    CostModelRoutedStrategy, PAR2QOEnhancedStrategy,
    AdaptiveStrategy,
)
from lynceus.benchmark import (
    WorkloadConfig, generate_query_sequence, run_benchmark, StrategyExecutor,
)


def _make_engine():
    return CostModelEngine(create_default_topology())


def _make_query(rows=100_000, qtype=QueryType.RANGE_SCAN, idx=True):
    return QueryDescriptor(
        query_id="test", query_type=qtype,
        estimated_rows=rows, selectivity=rows / 6_000_000,
        table_rows=6_000_000, index_available=idx,
    )


# ------------------------------------------------------------------
# Router tests
# ------------------------------------------------------------------

def test_router_register_lookup():
    engine = _make_engine()
    router = Router(engine)
    router.register(GPUOnlyStrategy(engine))
    assert "GPU-Only" in router.registered_names
    s = router.get("GPU-Only")
    assert s.name == "GPU-Only"
    print("  PASS: Router register/lookup")


def test_router_duplicate_rejected():
    engine = _make_engine()
    router = Router(engine)
    router.register(GPUOnlyStrategy(engine))
    try:
        router.register(GPUOnlyStrategy(engine))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "already registered" in str(e)
    print("  PASS: Router duplicate rejected")


def test_router_set_active():
    engine = _make_engine()
    router = Router.create_default(engine)
    router.set_active("CostModel-Routed")
    assert router.active_name == "CostModel-Routed"
    q = _make_query()
    d = router.route_one(q, "cpu0")
    assert isinstance(d, RoutingDecision)
    assert d.cost.total_us > 0
    print("  PASS: Router set_active dispatch")


def test_router_create_default():
    engine = _make_engine()
    router = Router.create_default(engine)
    names = router.registered_names
    assert len(names) == 6
    expected = {"GPU-Only", "CPU-Only", "Hybrid-Static",
                "CostModel-Routed", "PAR2QO-Enhanced", "Adaptive"}
    assert set(names) == expected, f"Got {names}"
    print("  PASS: Router create_default has 6 strategies")


def test_router_run_all():
    engine = _make_engine()
    router = Router.create_default(engine)
    queries = [_make_query(rows=r) for r in [100, 10_000, 500_000]]
    results = router.run_all_strategies(queries, "cpu0")
    assert len(results) == 6
    for name, decisions in results.items():
        assert len(decisions) == 3, f"{name} has {len(decisions)} decisions"
    print("  PASS: Router run_all_strategies")


# ------------------------------------------------------------------
# Strategy tests
# ------------------------------------------------------------------

def test_gpu_only_routes_to_gpu():
    engine = _make_engine()
    s = GPUOnlyStrategy(engine)
    d = s.route_one(_make_query(), "cpu0")
    assert d.device_id == "gpu0"
    assert d.metadata["reason"] == "fixed_gpu"
    print("  PASS: GPUOnlyStrategy → gpu0")


def test_cpu_only_routes_to_cpu():
    engine = _make_engine()
    s = CPUOnlyStrategy(engine)
    d = s.route_one(_make_query(), "cpu0")
    assert d.device_id == "cpu0"
    print("  PASS: CPUOnlyStrategy → cpu0")


def test_hybrid_threshold():
    engine = _make_engine()
    s = HybridStaticStrategy(engine, gpu_threshold_rows=50_000)
    # Below threshold → CPU
    d1 = s.route_one(_make_query(rows=1_000), "cpu0")
    assert d1.device_id == "cpu0"
    assert d1.metadata["reason"] == "rows_below_threshold"
    # Above threshold → GPU
    d2 = s.route_one(_make_query(rows=200_000), "cpu0")
    assert d2.device_id == "gpu0"
    assert d2.metadata["reason"] == "rows_above_threshold"
    print("  PASS: HybridStaticStrategy threshold")


def test_costing_routed_matches_engine():
    engine = _make_engine()
    s = CostModelRoutedStrategy(engine)
    q = _make_query(rows=500_000)
    d = s.route_one(q, "cpu0")
    # Should match engine.recommend()
    expected_dev, expected_cb = engine.recommend(q, "cpu0")
    assert d.device_id == expected_dev
    assert abs(d.cost.total_us - expected_cb.total_us) < 1e-9
    print("  PASS: CostModelRoutedStrategy == engine.recommend")


def test_par2qo_robustness():
    engine = _make_engine()
    s = PAR2QOEnhancedStrategy(engine, robustness_margin=0.20)
    # Tiny query: CPU should win (GPU has kernel launch overhead)
    q_tiny = _make_query(rows=1, qtype=QueryType.POINT_LOOKUP)
    d = s.route_one(q_tiny, "cpu0")
    assert d.device_id.startswith("cpu"), f"Expected CPU for tiny, got {d.device_id}"
    assert d.metadata["reason"] == "cpu_robust_choice"
    print("  PASS: PAR2QOEnhancedStrategy robustness")


def test_adaptive_warmup():
    engine = _make_engine()
    s = AdaptiveStrategy(engine, warmup_steps=10)
    q = _make_query()
    # First 10 queries should have low confidence
    for i in range(10):
        d = s.route_one(q, "cpu0")
        assert d.confidence == 0.5, f"Step {i}: confidence={d.confidence}"
        assert d.metadata["reason"] == "warmup"
    # 11th query should be post-warmup
    d = s.route_one(q, "cpu0")
    assert d.metadata["reason"] in ("min_adjusted_cost", "load_balanced")
    assert d.confidence > 0.5
    print("  PASS: AdaptiveStrategy warmup phase")


def test_adaptive_ema_correction():
    engine = _make_engine()
    s = AdaptiveStrategy(engine, ema_alpha=0.5, warmup_steps=0)
    # Simulate: gpu0 is actually 2x slower than estimated
    s.observe_with_estimate("gpu0", estimated_us=100.0, actual_us=200.0)
    bias = s._bias_ema["gpu0"]
    # EMA: 0.5 * (200/100) + 0.5 * 1.0 = 1.5
    assert abs(bias - 1.5) < 1e-9, f"Expected bias=1.5, got {bias}"
    # After correction, gpu0 costs should be inflated by 1.5x
    q = _make_query(rows=500_000)
    d = s.route_one(q, "cpu0")
    # The decision may or may not change, but adjusted cost should differ
    assert d.cost.total_us > 0
    print("  PASS: AdaptiveStrategy EMA correction")


def test_adaptive_load_balance():
    engine = _make_engine()
    s = AdaptiveStrategy(engine, warmup_steps=0, load_balance_margin=1.0)
    # With margin=1.0, ALL devices within 2x of best are eligible
    q = _make_query(rows=500_000)
    devices_seen = set()
    for _ in range(20):
        d = s.route_one(q, "cpu0")
        devices_seen.add(d.device_id)
    # Should have distributed across multiple devices
    assert len(devices_seen) > 1, f"Only used {devices_seen}"
    print("  PASS: AdaptiveStrategy load balancing")


# ------------------------------------------------------------------
# Backward compatibility
# ------------------------------------------------------------------

def test_strategy_executor_backward_compat():
    engine = _make_engine()
    executor = StrategyExecutor(engine)
    queries = [_make_query(rows=r) for r in [100, 50_000, 500_000]]
    # All original methods should still work
    for strategy in [RoutingStrategy.GPU_ONLY, RoutingStrategy.CPU_ONLY,
                     RoutingStrategy.HYBRID_STATIC,
                     RoutingStrategy.COST_MODEL_ROUTED,
                     RoutingStrategy.PAR2QO_ENHANCED]:
        latencies = executor.execute_strategy(strategy, queries, "cpu0")
        assert len(latencies) == 3
        assert all(lat > 0 for lat in latencies)
    print("  PASS: StrategyExecutor backward compat")


def test_benchmark_output_5_original():
    config = WorkloadConfig(num_steps=30, num_seeds=1, name="compat")
    original_5 = [
        RoutingStrategy.GPU_ONLY, RoutingStrategy.CPU_ONLY,
        RoutingStrategy.HYBRID_STATIC, RoutingStrategy.COST_MODEL_ROUTED,
        RoutingStrategy.PAR2QO_ENHANCED,
    ]
    output = run_benchmark(config, strategies=original_5)
    d = output.to_dict()
    assert d["metadata"]["n_methods"] == 5
    panel = list(d["panels"].values())[0]
    assert set(panel.keys()) == {
        "GPU-Only", "CPU-Only", "Hybrid-Static",
        "CostModel-Routed", "PAR2QO-Enhanced",
    }
    print("  PASS: Benchmark with 5 original strategies unchanged")


def test_full_benchmark_6_strategies():
    config = WorkloadConfig(num_steps=50, num_seeds=2, name="full6")
    output = run_benchmark(config)
    d = output.to_dict()
    assert d["metadata"]["n_methods"] == 6
    panel = list(d["panels"].values())[0]
    assert "Adaptive" in panel
    adaptive = panel["Adaptive"]
    assert len(adaptive["seed_0"]) == 50
    assert len(adaptive["mean"]) == 50
    print("  PASS: Full benchmark with 6 strategies")


if __name__ == "__main__":
    print("Running M003-M004 tests...")
    test_router_register_lookup()
    test_router_duplicate_rejected()
    test_router_set_active()
    test_router_create_default()
    test_router_run_all()
    test_gpu_only_routes_to_gpu()
    test_cpu_only_routes_to_cpu()
    test_hybrid_threshold()
    test_costing_routed_matches_engine()
    test_par2qo_robustness()
    test_adaptive_warmup()
    test_adaptive_ema_correction()
    test_adaptive_load_balance()
    test_strategy_executor_backward_compat()
    test_benchmark_output_5_original()
    test_full_benchmark_6_strategies()
    print("\nAll M003-M004 tests PASSED.")
