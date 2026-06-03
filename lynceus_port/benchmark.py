"""
lynceus_port/benchmark.py — 移植版 benchmark runner.

算法改写:
  - generate_query_sequence: selectivity 分布从均匀改为 Zipf-like
    (paretovariate), 模拟真实 workload 中热数据高频访问的模式.
  - generate_query_sequence: difficulty_factor 从线性 (1+0.5*t/N) 改为
    对数渐进 (1+0.5*log1p(t)/log1p(N)), 早期上升更陡、后期趋稳,
    更接近真实系统随时间变化的 workload 演化.
  - run_benchmark: 增加 CV 均衡检查——计算每个策略跨 seed 的
    变异系数 (CV=std/mean), 若 CV>0.5 说明种子间方差过大, 警告.
  - run_cumulative_benchmark: cumulative sum 改为 Kahan 补偿求和,
    避免 2000+ 步浮点累积误差.

溯源同原版 (Megatron pipeline / DeepSpeed warm-up / vLLM scheduler).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .schema import (
    BenchmarkOutput,
    MetricKind,
    MethodResult,
    PanelResult,
    RoutingStrategy,
    SeedCurve,
)
from .cost_model import (
    CostModelEngine,
    CPUCostModel,
    GPUCostModel,
    QueryDescriptor,
    QueryType,
    CostBreakdown,
    create_default_topology,
)
from .router import Router
from .strategies.base import RoutingStrategyBase

from . import _dbg, _dump_obj, _snapshot, _Timer, LYNCEUS_DEBUG
_T = "BEK"


# --- Workload ---

@dataclass
class WorkloadConfig:
    name: str = "TPC-H_SF100"
    num_steps: int = 2000
    num_seeds: int = 3
    base_table_rows: int = 6_000_000
    query_mix: Dict[QueryType, float] = field(default_factory=lambda: {
        QueryType.POINT_LOOKUP: 0.15,
        QueryType.RANGE_SCAN: 0.25,
        QueryType.FULL_TABLE_SCAN: 0.10,
        QueryType.INDEX_SCAN: 0.20,
        QueryType.JOIN: 0.15,
        QueryType.AGGREGATE: 0.10,
        QueryType.SORT: 0.05,
    })
    selectivity_range: Tuple[float, float] = (0.001, 0.5)
    index_availability_prob: float = 0.6
    tables: Dict[str, int] = field(default_factory=lambda: {
        "lineitem": 6_000_000,
        "orders": 1_500_000,
        "customer": 150_000,
        "part": 200_000,
        "supplier": 10_000,
    })


def generate_query_sequence(config: WorkloadConfig,
                            seed: int) -> List[QueryDescriptor]:
    """[PORT] Zipf selectivity + log warmup."""
    _dbg(_T, f"generate_query_sequence(seed={seed}, steps={config.num_steps})")
    rng = random.Random(seed)
    queries = []

    types_weights: List[Tuple[QueryType, float]] = list(config.query_mix.items())
    types = [t for t, _ in types_weights]
    weights = [w for _, w in types_weights]

    catalog = dict(config.tables) if config.tables else {
        config.name: config.base_table_rows
    }
    table_names = list(catalog.keys())

    for step in range(config.num_steps):
        qt = rng.choices(types, weights=weights, k=1)[0]
        table_name = rng.choice(table_names)
        table_rows = catalog[table_name]

        # [PORT] Zipf-like selectivity 替代均匀分布
        # paretovariate(alpha=1.5) 产生重尾分布, 大多数 query 命中热数据
        lo, hi = config.selectivity_range
        raw_pareto = rng.paretovariate(1.5)
        selectivity = lo + (hi - lo) * min(1.0, raw_pareto / 10.0)

        estimated_rows = max(1, int(table_rows * selectivity))

        # [PORT] 对数渐进 warmup 替代线性
        # 线性: 1 + 0.5 * (step / N) → 步速恒定
        # 对数: 1 + 0.5 * log1p(step) / log1p(N) → 早期快升, 后期趋稳
        difficulty_factor = 1.0 + 0.5 * math.log1p(step) / math.log1p(config.num_steps)

        estimated_rows = min(
            int(estimated_rows * difficulty_factor),
            table_rows,
        )

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


# --- Strategy executor ---

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
        return self.execute_strategy(
            RoutingStrategy.GPU_ONLY, queries, data_location)

    def execute_cpu_only(self, queries, data_location="cpu0"):
        return self.execute_strategy(
            RoutingStrategy.CPU_ONLY, queries, data_location)

    def execute_hybrid_static(self, queries, data_location="cpu0", **_kw):
        return self.execute_strategy(
            RoutingStrategy.HYBRID_STATIC, queries, data_location)

    def execute_cost_model_routed(self, queries, data_location="cpu0"):
        return self.execute_strategy(
            RoutingStrategy.COST_MODEL_ROUTED, queries, data_location)

    def execute_par2qo_enhanced(self, queries, data_location="cpu0"):
        return self.execute_strategy(
            RoutingStrategy.PAR2QO_ENHANCED, queries, data_location)


# --- Benchmark runner ---

def run_benchmark(
    workload: WorkloadConfig,
    strategies: Optional[List[RoutingStrategy]] = None,
    output_path: Optional[str] = None,
) -> BenchmarkOutput:
    _dbg(_T, f"run_benchmark({workload.name})")

    if strategies is None:
        strategies = [
            RoutingStrategy.GPU_ONLY,
            RoutingStrategy.CPU_ONLY,
            RoutingStrategy.HYBRID_STATIC,
            RoutingStrategy.COST_MODEL_ROUTED,
            RoutingStrategy.PAR2QO_ENHANCED,
            RoutingStrategy.ADAPTIVE,
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
        x_label="workload_step",
        y_label="latency_ms",
    )

    for strategy in strategies:
        method = panel.add_method(
            strategy=strategy,
            num_steps=workload.num_steps,
            num_seeds=workload.num_seeds,
        )
        method.x_values = list(range(workload.num_steps))

        for seed_idx in range(workload.num_seeds):
            seed_val = 42 + seed_idx * 1000
            queries = generate_query_sequence(workload, seed=seed_val)
            latencies = executor.execute_strategy(
                strategy, queries, data_location="cpu0"
            )

            sc = method.add_seed()
            sc.values = latencies

            if method.total_cost is None:
                method.total_cost = 0.0
            method.total_cost += sum(latencies)

        method.total_cost = (method.total_cost or 0.0) / workload.num_seeds
        method.compute_statistics()

        # [PORT] CV 均衡检查: 种子间变异系数过大则警告
        if method.std and method.mean:
            # 取最后 10% 步的平均 CV
            tail = max(1, len(method.mean) // 10)
            tail_cvs = []
            for i in range(-tail, 0):
                m = method.mean[i]
                s = method.std[i]
                if m > 0:
                    tail_cvs.append(s / m)
            if tail_cvs:
                avg_cv = sum(tail_cvs) / len(tail_cvs)
                if avg_cv > 0.5:
                    _dbg(_T, f"WARNING: {strategy.value} tail CV={avg_cv:.3f} > 0.5, "
                         f"seeds 间方差过大, 建议增加 num_seeds")

    if output_path:
        output.save(output_path)

    return output


# --- Cumulative latency ---

def run_cumulative_benchmark(
    workload: WorkloadConfig,
    strategies: Optional[List[RoutingStrategy]] = None,
    output_path: Optional[str] = None,
) -> BenchmarkOutput:
    _dbg(_T, f"run_cumulative_benchmark({workload.name})")

    if strategies is None:
        strategies = [
            RoutingStrategy.GPU_ONLY,
            RoutingStrategy.CPU_ONLY,
            RoutingStrategy.HYBRID_STATIC,
            RoutingStrategy.COST_MODEL_ROUTED,
            RoutingStrategy.PAR2QO_ENHANCED,
            RoutingStrategy.ADAPTIVE,
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
        x_label="workload_step",
        y_label="cumulative_latency_ms",
    )

    for strategy in strategies:
        method = panel.add_method(
            strategy=strategy,
            num_steps=workload.num_steps,
            num_seeds=workload.num_seeds,
        )
        method.x_values = list(range(workload.num_steps))

        for seed_idx in range(workload.num_seeds):
            seed_val = 42 + seed_idx * 1000
            queries = generate_query_sequence(workload, seed=seed_val)
            latencies = executor.execute_strategy(
                strategy, queries, data_location="cpu0"
            )

            # [PORT] Kahan 补偿求和替代 naive cumsum
            # 避免 2000+ 步的浮点累积误差
            cumulative = []
            kahan_sum = 0.0
            kahan_comp = 0.0  # 补偿项
            for lat in latencies:
                y = lat - kahan_comp
                t = kahan_sum + y
                kahan_comp = (t - kahan_sum) - y
                kahan_sum = t
                cumulative.append(kahan_sum)

            sc = method.add_seed()
            sc.values = cumulative

        method.compute_statistics()

    if output_path:
        output.save(output_path)

    return output


# --- CLI ---

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

    output1 = run_benchmark(
        workload,
        output_path="output/latency_vs_step.json",
    )
    print(f"\nPer-step latency data saved.")
    for name, panel in output1.panels.items():
        for mname, mr in panel.methods.items():
            print(f"  {mname}: final_mean={mr.mean[-1]:.3f}ms, "
                  f"total_cost={mr.total_cost:.1f}ms")

    output2 = run_cumulative_benchmark(
        workload,
        output_path="output/cumulative_latency.json",
    )
    print(f"\nCumulative latency data saved.")
    for name, panel in output2.panels.items():
        for mname, mr in panel.methods.items():
            print(f"  {mname}: final_cumulative={mr.mean[-1]:.1f}ms")

    print(f"\nMetadata: {output1.metadata}")


if __name__ == "__main__":
    main()
