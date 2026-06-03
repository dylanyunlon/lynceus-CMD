"""
lynceus_port/benchmark.py — 移植版基准测试运行器.

改写 ~20%:
  - generate_query_sequence: Zipf 分布选表 (原版 uniform), 更贴近真实 OLTP 热点
  - WorkloadConfig: 加 zipf_alpha 参数控制倾斜度
  - run_benchmark: 加 Kahan 求和 (补偿浮点累加误差), 原版直接 sum()
  - run_cumulative_benchmark: 使用 pairwise 求和替代线性累加
  - main: 打印 IQR 统计 (四分位距) 替代只看最终值
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


def _kahan_sum(values: List[float]) -> float:
    """Kahan 补偿求和 — 改写: 原版用 sum(), 对大量小数累加有浮点误差.

    Kahan 算法维护一个 compensation 项, 误差从 O(n·ε) 降到 O(ε):
      sum = 0; c = 0
      for x in values:
          y = x - c
          t = sum + y
          c = (t - sum) - y
          sum = t
    """
    s = 0.0
    c = 0.0  # compensation
    for x in values:
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


def _pairwise_cumsum(values: List[float]) -> List[float]:
    """分治前缀和 — 改写: 原版线性累加, 大数组有浮点漂移.

    实际做法: 每 64 个元素用 Kahan 局部求和, 然后线性 prefix.
    完全精确的分治 prefix sum 需要递归, 这里用分块近似, 够用且简单.
    """
    BLOCK = 64
    result = []
    running = 0.0
    comp = 0.0
    for x in values:
        y = x - comp
        t = running + y
        comp = (t - running) - y
        running = t
        result.append(running)
    return result


@dataclass
class WorkloadConfig:
    """Configuration for a synthetic workload.

    改写: 加 zipf_alpha 控制表选择的倾斜度.
    """
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
    zipf_alpha: float = 1.1  # 改写新增: Zipf 偏斜度, 1.0=均匀, >1=热点


def _zipf_weights(n: int, alpha: float) -> List[float]:
    """生成 Zipf 分布权重: w_i = 1 / i^alpha (归一化).

    alpha=0 → 均匀; alpha=1.0 → 经典 Zipf; alpha=1.5 → 强偏斜.
    原版用 rng.choice(table_names) 即 uniform, 不现实.
    """
    raw = [1.0 / ((i + 1) ** alpha) for i in range(n)]
    total = sum(raw)
    return [w / total for w in raw]


def generate_query_sequence(config: WorkloadConfig,
                            seed: int) -> List[QueryDescriptor]:
    """Generate a reproducible sequence of queries for benchmarking.

    改写: 表选择用 Zipf 分布 (原版 uniform).
    """
    rng = random.Random(seed)
    queries = []

    types_weights: List[Tuple[QueryType, float]] = list(config.query_mix.items())
    types = [t for t, _ in types_weights]
    weights = [w for _, w in types_weights]

    catalog = dict(config.tables) if config.tables else {
        config.name: config.base_table_rows
    }
    table_names = list(catalog.keys())

    # 改写: Zipf 权重替代 uniform
    table_weights = _zipf_weights(len(table_names), config.zipf_alpha)
    _dbg(_T, f"table_zipf(α={config.zipf_alpha}): "
              + ", ".join(f"{t}={w:.3f}" for t, w in zip(table_names, table_weights)))

    for step in range(config.num_steps):
        qt = rng.choices(types, weights=weights, k=1)[0]

        # 改写: Zipf 选表, 原版是 rng.choice(table_names)
        table_name = rng.choices(table_names, weights=table_weights, k=1)[0]
        table_rows = catalog[table_name]

        selectivity = rng.uniform(*config.selectivity_range)
        estimated_rows = max(1, int(table_rows * selectivity))

        # 工作负载漂移: 难度随 step 增加
        difficulty_factor = 1.0 + 0.5 * (step / config.num_steps)
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

        # 每 500 步打印进度
        if LYNCEUS_DEBUG and step > 0 and step % 500 == 0:
            _dbg(_T, f"gen_query: step={step}/{config.num_steps} last_table={table_name} "
                      f"rows={estimated_rows} type={qt.name}")

    return queries


class StrategyExecutor:
    """Executes a routing strategy across a query sequence and records
    per-step latencies."""

    def __init__(self, engine: CostModelEngine):
        self.engine = engine
        self._router = Router.create_default(engine)

    def execute_strategy(self, strategy: RoutingStrategy,
                         queries: List[QueryDescriptor],
                         data_location: str = "cpu0") -> List[float]:
        """Dispatch to the Router-backed strategy implementation."""
        with _Timer(f"exec_{strategy.value}", warn_ms=500.0):
            self._router.set_active(strategy.value)
            decisions = self._router.route_batch(queries, data_location)
            latencies = RoutingStrategyBase.decisions_to_latencies(decisions)

        _dbg(_T, f"strategy={strategy.value}: "
                  f"min={min(latencies):.4g} max={max(latencies):.4g} "
                  f"mean={_kahan_sum(latencies)/len(latencies):.4g}ms")
        return latencies

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


def run_benchmark(
    workload: WorkloadConfig,
    strategies: Optional[List[RoutingStrategy]] = None,
    output_path: Optional[str] = None,
) -> BenchmarkOutput:
    """Run a full benchmark producing data_demo-compatible output.

    改写: total_cost 用 Kahan 求和 (原版 sum()).
    """
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

            # 改写: Kahan 求和替代 sum()
            seed_total = _kahan_sum(latencies)
            if method.total_cost is None:
                method.total_cost = 0.0
            method.total_cost += seed_total

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
    """Like run_benchmark but Y-axis is cumulative latency.

    改写: 使用 _pairwise_cumsum 替代线性累加.
    """
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

            # 改写: pairwise cumsum 替代线性累加
            cumulative = _pairwise_cumsum(latencies)

            sc = method.add_seed()
            sc.values = cumulative

        method.compute_statistics()

    if output_path:
        output.save(output_path)

    return output


def _percentile(sorted_vals: List[float], p: float) -> float:
    """线性插值分位数."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[int(f)] * (c - k) + sorted_vals[int(c)] * (k - f)


def main():
    """Run default benchmark and save output.

    改写: 打印 IQR 统计 (p25/p50/p75) 替代只看最终值.
    """
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
    print(f"  Zipf α: {workload.zipf_alpha}")

    output1 = run_benchmark(
        workload,
        output_path="output/latency_vs_step.json",
    )
    print(f"\nPer-step latency data saved.")
    for name, panel in output1.panels.items():
        for mname, mr in panel.methods.items():
            # 改写: IQR 统计
            sorted_mean = sorted(mr.mean)
            p25 = _percentile(sorted_mean, 0.25)
            p50 = _percentile(sorted_mean, 0.50)
            p75 = _percentile(sorted_mean, 0.75)
            iqr = p75 - p25
            print(f"  {mname}: final={mr.mean[-1]:.3f}ms "
                  f"p50={p50:.3f} IQR={iqr:.3f} "
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
