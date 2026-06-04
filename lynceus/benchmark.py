"""
lynceus/benchmark.py — Benchmark runner.

算法改动:
    1. generate_query_sequence: difficulty_factor 用 logistic ramp 代替线性增长
       效果: 前半程缓慢爬升, 后半程陡峭, 更接近真实workload的phase transition
    2. 新增 table locality burst: 连续 burst_len 条 query 倾向命中同一张表
       效果: 缓存命中率曲线有明显的 phase 结构, 不像原版那样均匀随机
    3. run_benchmark 在每个 strategy 上自动跑 tournament 并附带 Elo
"""
from __future__ import annotations
import math
import random
import time
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


@dataclass
class WorkloadConfig:
    name: str = "TPC-H_SF100"
    num_steps: int = 2000
    num_seeds: int = 3
    base_table_rows: int = 6_000_000
    query_mix: Dict[QueryType, float] = field(default_factory=lambda: {
        QueryType.POINT_LOOKUP: 0.15, QueryType.RANGE_SCAN: 0.25,
        QueryType.FULL_TABLE_SCAN: 0.10, QueryType.INDEX_SCAN: 0.20,
        QueryType.JOIN: 0.15, QueryType.AGGREGATE: 0.10, QueryType.SORT: 0.05,
    })
    selectivity_range: Tuple[float, float] = (0.001, 0.45)
    index_availability_prob: float = 0.6
    tables: Dict[str, int] = field(default_factory=lambda: {
        "lineitem": 6_000_000, "orders": 1_500_000,
        "customer": 150_000, "part": 200_000, "supplier": 10_000,
    })
    # 改动: table locality burst 参数
    burst_len_range: Tuple[int, int] = (5, 25)  # 连续多少条query命中同一张表
    # logistic ramp 参数
    ramp_midpoint: float = 0.65   # 相对位置(0~1), difficulty陡增的中心点
    ramp_steepness: float = 8.0   # logistic曲线陡度


def generate_query_sequence(config: WorkloadConfig,
                            seed: int) -> List[QueryDescriptor]:
    rng = random.Random(seed)
    queries = []
    types_weights = list(config.query_mix.items())
    types = [t for t, _ in types_weights]
    weights = [w for _, w in types_weights]
    catalog = dict(config.tables) if config.tables else {config.name: config.base_table_rows}
    table_names = list(catalog.keys())

    # 改动: 预生成 table-locality burst 序列
    # 连续 burst_len 条 query 命中同一张表, 然后切换
    table_sequence: List[str] = []
    while len(table_sequence) < config.num_steps:
        burst_len = rng.randint(*config.burst_len_range)
        chosen_table = rng.choice(table_names)
        table_sequence.extend([chosen_table] * burst_len)
    table_sequence = table_sequence[:config.num_steps]

    for step in range(config.num_steps):
        qt = rng.choices(types, weights=weights, k=1)[0]
        table_name = table_sequence[step]
        table_rows = catalog[table_name]
        selectivity = rng.uniform(*config.selectivity_range)
        estimated_rows = max(1, int(table_rows * selectivity))

        # 改动: logistic ramp 代替线性增长
        # 原版: difficulty_factor = 1.0 + 0.55 * (step / num_steps)  — 线性
        # 新版: logistic S曲线, 前半程缓慢, 后半程陡峭
        progress = step / max(1, config.num_steps)
        logistic_z = config.ramp_steepness * (progress - config.ramp_midpoint)
        if abs(logistic_z) < 20:
            logistic_val = 1.0 / (1.0 + math.exp(-logistic_z))
        else:
            logistic_val = 1.0 if logistic_z > 0 else 0.0
        difficulty_factor = 1.0 + 0.55 * logistic_val

        estimated_rows = min(int(estimated_rows * difficulty_factor), table_rows)

        q = QueryDescriptor(
            query_id=f"q_{step:05d}",
            query_type=qt,
            estimated_rows=estimated_rows,
            estimated_width_bytes=rng.randint(50, 500),
            num_predicates=rng.randint(1, 5),
            selectivity=selectivity,
            table_rows=table_rows,
            index_available=rng.random() < config.index_availability_prob,
            index_depth=rng.randint(2, 5),
            num_joins=rng.randint(0, 3) if qt == QueryType.JOIN else 0,
            sort_required=(qt == QueryType.SORT or rng.random() < 0.2),
            group_by_cardinality=rng.randint(10, 1000) if qt == QueryType.AGGREGATE else 0,
            table_name=table_name,
        )
        queries.append(q)

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
        return self.execute_strategy(RoutingStrategy.GPU_ONLY, queries, data_location)
    def execute_cpu_only(self, queries, data_location="cpu0"):
        return self.execute_strategy(RoutingStrategy.CPU_ONLY, queries, data_location)
    def execute_hybrid_static(self, queries, data_location="cpu0", **_kw):
        return self.execute_strategy(RoutingStrategy.HYBRID_STATIC, queries, data_location)
    def execute_cost_model_routed(self, queries, data_location="cpu0"):
        return self.execute_strategy(RoutingStrategy.COST_MODEL_ROUTED, queries, data_location)
    def execute_par2qo_enhanced(self, queries, data_location="cpu0"):
        return self.execute_strategy(RoutingStrategy.PAR2QO_ENHANCED, queries, data_location)


def run_benchmark(
    workload: WorkloadConfig,
    strategies: Optional[List[RoutingStrategy]] = None,
    output_path: Optional[str] = None,
) -> BenchmarkOutput:
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
        source="lynceus_benchmark_runner",
    )
    panel = output.add_panel(
        name=f"latency_vs_step_{workload.name}",
        metric=MetricKind.LATENCY_MS,
        x_label="workload_step", y_label="latency_ms",
    )
    from ._debug import dbg
    dbg('Benchmark.run', n_strategies=len(strategies), n_steps=workload.num_steps)
    for strategy in strategies:
        method = panel.add_method(strategy=strategy,
            num_steps=workload.num_steps, num_seeds=workload.num_seeds)
        method.x_values = list(range(workload.num_steps))
        for seed_idx in range(workload.num_seeds):
            seed_val = 42 + seed_idx * 1000
            queries = generate_query_sequence(workload, seed=seed_val)
            latencies = executor.execute_strategy(strategy, queries, data_location="cpu0")
            sc = method.add_seed()
            sc.values = latencies
            if method.total_cost is None:
                method.total_cost = 0.0
            method.total_cost += sum(latencies)
        method.total_cost = (method.total_cost or 0.0) / workload.num_seeds
        method.compute_statistics()
    if output_path:
        output.save(output_path)
    return output


def run_cumulative_benchmark(
    workload: WorkloadConfig,
    strategies: Optional[List[RoutingStrategy]] = None,
    output_path: Optional[str] = None,
) -> BenchmarkOutput:
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
        description=f"Lynceus cumulative latency — {workload.name}",
        source="lynceus_benchmark_runner_cumulative",
    )
    panel = output.add_panel(
        name=f"cumulative_latency_{workload.name}",
        metric=MetricKind.LATENCY_MS,
        x_label="workload_step", y_label="cumulative_latency_ms",
    )
    for strategy in strategies:
        method = panel.add_method(strategy=strategy,
            num_steps=workload.num_steps, num_seeds=workload.num_seeds)
        method.x_values = list(range(workload.num_steps))
        for seed_idx in range(workload.num_seeds):
            seed_val = 42 + seed_idx * 1000
            queries = generate_query_sequence(workload, seed=seed_val)
            latencies = executor.execute_strategy(strategy, queries, data_location="cpu0")
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
    import os
    def _int_env(key: str, default: int) -> int:
        raw = os.environ.get(key)
        if raw is None or raw.strip() == "":
            return default
        try:
            v = int(raw)
        except ValueError:
            print(f"  [warn] {key}={raw!r} is not an integer; using {default}")
            return default
        if v <= 0:
            print(f"  [warn] {key}={v} must be > 0; using {default}")
            return default
        return v

    workload = WorkloadConfig(
        name=os.environ.get("WORKLOAD_NAME", "TPC-H_SF100"),
        num_steps=_int_env("NUM_STEPS", 2000),
        num_seeds=_int_env("NUM_SEEDS", 3),
    )
    print(f"Running Lynceus benchmark: {workload.name}")
    print(f"  Steps: {workload.num_steps}, Seeds: {workload.num_seeds}")

    output1 = run_benchmark(workload, output_path="output/latency_vs_step.json")
    print(f"\nPer-step latency data saved.")
    for name, panel in output1.panels.items():
        for mname, mr in panel.methods.items():
            print(f"  {mname}: final_mean={mr.mean[-1]:.3f}ms, total_cost={mr.total_cost:.1f}ms")

    output2 = run_cumulative_benchmark(workload, output_path="output/cumulative_latency.json")
    print(f"\nCumulative latency data saved.")
    for name, panel in output2.panels.items():
        for mname, mr in panel.methods.items():
            print(f"  {mname}: final_cumulative={mr.mean[-1]:.1f}ms")

    # 改动: 跑 tournament 并输出 Elo
    topology = create_default_topology()
    engine = CostModelEngine(topology)
    router = Router.create_default(engine)
    queries = generate_query_sequence(workload, seed=42)
    tourney = router.tournament(queries, data_location="cpu0")
    print(f"\nTournament Elo ratings:")
    for name, rating in sorted(tourney["elo"].items(), key=lambda x: -x[1]):
        print(f"  {name}: {rating:.0f}")

    print(f"\nMetadata: {output1.metadata}")


if __name__ == "__main__":
    main()
