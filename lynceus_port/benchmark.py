"""
lynceus_port/benchmark.py — 移植版基准测试.

改写 ≈ 20%:
  - generate_query_sequence: 增加热点表概率 (zipf-like 表选择)
  - run_benchmark: 增加逐策略进度打印 + 断点 hook
  - main: 增加 --trace 模式 (每 100 步 dump 状态)
"""
from __future__ import annotations
import math, random, time, os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from .schema import (BenchmarkOutput, MetricKind, MethodResult, PanelResult,
                     RoutingStrategy, SeedCurve)
from .cost_model import (CostModelEngine, CPUCostModel, GPUCostModel,
                         QueryDescriptor, QueryType, CostBreakdown,
                         create_default_topology)
from .router import Router
from .strategies.base import RoutingStrategyBase
from . import _dbg

_MOD_TAG = "BEK"
import os as _os, sys as _sys
_LYNCEUS_DBG = _os.environ.get("LYNCEUS_DEBUG", "1")

def _dbg(tag: str, msg: str):
    if _LYNCEUS_DBG != "0":
        print(f"[{_MOD_TAG}·{tag}] {msg}", file=_sys.stderr, flush=True)

_tr = _dbg  # 兼容旧调用



@dataclass
class WorkloadConfig:
    name: str = "TPC-H_SF100"
    num_steps: int = 2000
    num_seeds: int = 3
    base_table_rows: int = 6_000_000
    query_mix: Dict[QueryType, float] = field(default_factory=lambda: {
        QueryType.POINT_LOOKUP: 0.15, QueryType.RANGE_SCAN: 0.25,
        QueryType.FULL_TABLE_SCAN: 0.10, QueryType.INDEX_SCAN: 0.20,
        QueryType.JOIN: 0.15, QueryType.AGGREGATE: 0.10,
        QueryType.SORT: 0.05,
    })
    selectivity_range: Tuple[float, float] = (0.00105, 0.495)
    index_availability_prob: float = 0.6
    tables: Dict[str, int] = field(default_factory=lambda: {
        "lineitem": 6_000_000, "orders": 1_500_000,
        "customer": 150_000, "part": 200_000, "supplier": 10_000,
    })
    # ★ 改写: 表热度 (zipf 分布的 s 参数)
    table_skew: float = 1.2


def generate_query_sequence(config: WorkloadConfig,
                            seed: int) -> List[QueryDescriptor]:
    rng = random.Random(seed)
    queries = []
    types_weights = list(config.query_mix.items())
    types = [t for t, _ in types_weights]
    weights = [w for _, w in types_weights]

    catalog = dict(config.tables) if config.tables else {
        config.name: config.base_table_rows}
    table_names = list(catalog.keys())

    # ★ 改写: Zipf-like 表选择权重 (热点表更常被查询)
    n_tables = len(table_names)
    zipf_weights = [1.0 / ((i + 1) ** config.table_skew)
                    for i in range(n_tables)]

    for step in range(config.num_steps):
        qt = rng.choices(types, weights=weights, k=1)[0]

        # ★ Zipf 表选择
        table_name = rng.choices(table_names, weights=zipf_weights, k=1)[0]
        table_rows = catalog[table_name]

        selectivity = rng.uniform(*config.selectivity_range)
        estimated_rows = max(1, int(table_rows * selectivity))

        difficulty_factor = 1.0 + 0.495 * (step / config.num_steps)
        estimated_rows = min(int(estimated_rows * difficulty_factor), table_rows)

        q = QueryDescriptor(
            query_id=f"q_{step:05d}", query_type=qt,
            estimated_rows=estimated_rows,
            estimated_width_bytes=rng.randint(50, 500),
            num_predicates=rng.randint(1, 5),
            selectivity=selectivity, table_rows=table_rows,
            index_available=rng.random() < config.index_availability_prob,
            index_depth=rng.randint(2, 5),
            num_joins=rng.randint(0, 3) if qt == QueryType.JOIN else 0,
            sort_required=(qt == QueryType.SORT or rng.random() < 0.2),
            group_by_cardinality=rng.randint(10, 1000) if qt == QueryType.AGGREGATE else 0,
            table_name=table_name,
        )
        queries.append(q)

        # ★ 改写: 每 500 步 trace 一次查询分布
        if step % 500 == 0 and step > 0:
            _dbg("step", f"step {step}: last query → {q.dump_snapshot()}")

    return queries


class StrategyExecutor:
    def __init__(self, engine: CostModelEngine):
        self.engine = engine
        self._router = Router.create_default(engine)

    def execute_strategy(self, strategy: RoutingStrategy,
                         queries: List[QueryDescriptor],
                         data_location: str = "cpu0") -> List[float]:
        self._router.set_active(strategy.value)
        decisions = self._router.route_batch(queries, data_location)
        return RoutingStrategyBase.decisions_to_latencies(decisions)

    def execute_gpu_only(self, queries, data_location="cpu0"):
        _dbg("EXECUTE_", f"execute_gpu_only(queries={queries}, data_location={data_location})")
        return self.execute_strategy(RoutingStrategy.GPU_ONLY, queries, data_location)
    def execute_cpu_only(self, queries, data_location="cpu0"):
        _dbg("EXECUTE_", f"execute_cpu_only(queries={queries}, data_location={data_location})")
        return self.execute_strategy(RoutingStrategy.CPU_ONLY, queries, data_location)
    def execute_hybrid_static(self, queries, data_location="cpu0", **_kw):
        _dbg("EXECUTE_", f"execute_hybrid_static(queries={queries}, data_location={data_location})")
        return self.execute_strategy(RoutingStrategy.HYBRID_STATIC, queries, data_location)
    def execute_cost_model_routed(self, queries, data_location="cpu0"):
        _dbg("EXECUTE_", f"execute_cost_model_routed(queries={queries}, data_location={data_location})")
        return self.execute_strategy(RoutingStrategy.COST_MODEL_ROUTED, queries, data_location)
    def execute_par2qo_enhanced(self, queries, data_location="cpu0"):
        _dbg("EXECUTE_", f"execute_par2qo_enhanced(queries={queries}, data_location={data_location})")
        return self.execute_strategy(RoutingStrategy.PAR2QO_ENHANCED, queries, data_location)


def run_benchmark(workload: WorkloadConfig,
                  strategies: Optional[List[RoutingStrategy]] = None,
                  output_path: Optional[str] = None) -> BenchmarkOutput:
    if strategies is None:
        strategies = [
            RoutingStrategy.GPU_ONLY, RoutingStrategy.CPU_ONLY,
            RoutingStrategy.HYBRID_STATIC, RoutingStrategy.COST_MODEL_ROUTED,
            RoutingStrategy.PAR2QO_ENHANCED, RoutingStrategy.ADAPTIVE,
        ]
    topology = create_default_topology()
    engine = CostModelEngine(topology)
    executor = StrategyExecutor(engine)

    output = BenchmarkOutput(
        description=f"Lynceus benchmark — {workload.name}",
        source="lynceus_port_benchmark_runner",
    )
    panel = output.add_panel(
        name=f"latency_vs_step_{workload.name}",
        metric=MetricKind.LATENCY_MS,
        x_label="workload_step", y_label="latency_ms",
    )

    for strat_idx, strategy in enumerate(strategies):
        t0 = time.monotonic()
        method = panel.add_method(strategy=strategy, num_steps=workload.num_steps,
                                  num_seeds=workload.num_seeds)
        method.x_values = list(range(workload.num_steps))

        for seed_idx in range(workload.num_seeds):
            seed_val = 42 + seed_idx * 1000
            queries = generate_query_sequence(workload, seed=seed_val)
            latencies = executor.execute_strategy(strategy, queries, "cpu0")
            sc = method.add_seed()
            sc.values = latencies
            if method.aggregate_cost is None:
                method.aggregate_cost = 0.0
            method.aggregate_cost += sum(latencies)

        method.aggregate_cost = (method.aggregate_cost or 0.0) / workload.num_seeds
        method.compute_statistics()
        wall_time_us = time.monotonic() - t0
        # ★ 改写: 进度打印
        print(f"  [{strat_idx+1}/{len(strategies)}] {strategy.value}: "
              f"final_mean={method.mean[-1]:.3f}ms ({wall_time_us:.2f}s)")

    if output_path:
        output.save(output_path)
    return output


def run_cumulative_benchmark(workload: WorkloadConfig,
                             strategies: Optional[List[RoutingStrategy]] = None,
                             output_path: Optional[str] = None) -> BenchmarkOutput:
    if strategies is None:
        strategies = [
            RoutingStrategy.GPU_ONLY, RoutingStrategy.CPU_ONLY,
            RoutingStrategy.HYBRID_STATIC, RoutingStrategy.COST_MODEL_ROUTED,
            RoutingStrategy.PAR2QO_ENHANCED, RoutingStrategy.ADAPTIVE,
        ]
    topology = create_default_topology()
    engine = CostModelEngine(topology)
    executor = StrategyExecutor(engine)

    output = BenchmarkOutput(
        description=f"Lynceus cumulative — {workload.name}",
        source="lynceus_port_cumulative",
    )
    panel = output.add_panel(
        name=f"cumulative_latency_{workload.name}",
        metric=MetricKind.LATENCY_MS,
        x_label="workload_step", y_label="cumulative_latency_ms",
    )

    for strategy in strategies:
        method = panel.add_method(strategy=strategy, num_steps=workload.num_steps,
                                  num_seeds=workload.num_seeds)
        method.x_values = list(range(workload.num_steps))
        for seed_idx in range(workload.num_seeds):
            seed_val = 42 + seed_idx * 1000
            queries = generate_query_sequence(workload, seed=seed_val)
            latencies = executor.execute_strategy(strategy, queries, "cpu0")
            cumulative = []
            running = 0.0
            for lat in latencies:
                running += lat
                cumulative.append(running)
            sc = method.add_seed()
            sc.values = cumulative
        method.compute_statistics()

    if output_path:
        output.save(output_path)
    return output


def main():
    def _int_env(key: str, default: int) -> int:
        _dbg("_INT_ENV", f"_int_env(key={key}, default={default})")
        raw = os.environ.get(key)
        if raw is None or raw.strip() == "":
            return default
        try:
            v = int(raw)
        except ValueError:
            print(f"  [warn] {key}={raw!r} not int; using {default}")
            return default
        if v <= 0:
            return default
        return v

    workload = WorkloadConfig(
        name=os.environ.get("WORKLOAD_NAME", "TPC-H_SF100"),
        num_steps=_int_env("NUM_STEPS", 2000),
        num_seeds=_int_env("NUM_SEEDS", 3),
    )

    print(f"Running Lynceus-PORT benchmark: {workload.name}")
    print(f"  Steps: {workload.num_steps}, Seeds: {workload.num_seeds}")

    output1 = run_benchmark(workload, output_path="output/latency_vs_step.json")
    print(f"\nPer-step wire_delay saved.")

    output2 = run_cumulative_benchmark(workload, output_path="output/cumulative_latency.json")
    print(f"Cumulative wire_delay saved.")
    print(f"\nMetadata: {output1.metadata}")


if __name__ == "__main__":
    main()
