"""
lynceus/benchmark.py — Benchmark runner producing data_demo-compatible output.

Generates X-axis granularity data (2000+ steps, 3+ seeds, 5+ methods)
that matches the schema of:
    - reversed_figure_data.json  (panels → methods → curves)
    - gradient_norm_24k_data.json (steps + methods with seed_N arrays)
    - ppl_vs_time_1B_30k_data.json (time_hours x-axis)

Architecture references:
    - Megatron forward_backward_pipelining_with_interleaving (schedules.py:896)
      → the benchmark "steps" are analogous to pipeline micro-batches
    - DeepSpeed InferenceEngine (inference/engine.py:40)
      → warm-up + measurement phases
    - vLLM Scheduler (vllm/v1/core/sched/scheduler.py:64)
      → workload scheduling across steps
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


# ---------------------------------------------------------------------------
# Workload generators
# ---------------------------------------------------------------------------

@dataclass
class WorkloadConfig:
    """Configuration for a synthetic workload.

    Each "step" is one query from the workload; the benchmark measures
    the routing strategy's cumulative latency across steps.
    """
    name: str = "TPC-H_SF100"
    num_steps: int = 2000
    num_seeds: int = 3
    base_table_rows: int = 6_000_000  # TPC-H lineitem SF100
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


def generate_query_sequence(config: WorkloadConfig,
                            seed: int) -> List[QueryDescriptor]:
    """Generate a reproducible sequence of queries for benchmarking.

    Each call with the same seed produces identical queries (like
    setting NCCL_SEED for reproducible topology search).
    """
    rng = random.Random(seed)
    queries = []

    # Build weighted query type list
    types_weights: List[Tuple[QueryType, float]] = list(config.query_mix.items())
    types = [t for t, _ in types_weights]
    weights = [w for _, w in types_weights]

    for step in range(config.num_steps):
        qt = rng.choices(types, weights=weights, k=1)[0]

        selectivity = rng.uniform(*config.selectivity_range)
        estimated_rows = max(1, int(config.base_table_rows * selectivity))

        # Workload shift: as step increases, queries get harder
        # (simulates realistic workload evolution)
        difficulty_factor = 1.0 + 0.5 * (step / config.num_steps)
        estimated_rows = min(
            int(estimated_rows * difficulty_factor),
            config.base_table_rows,  # clamp: cannot exceed table size
        )

        q = QueryDescriptor(
            query_id=f"q_{step:05d}",
            query_type=qt,
            estimated_rows=estimated_rows,
            estimated_width_bytes=rng.randint(50, 500),
            num_predicates=rng.randint(1, 5),
            selectivity=selectivity,
            table_rows=config.base_table_rows,
            index_available=rng.random() < config.index_availability_prob,
            index_depth=rng.randint(2, 5),
            num_joins=rng.randint(0, 3) if qt == QueryType.JOIN else 0,
            sort_required=(qt == QueryType.SORT or rng.random() < 0.2),
            group_by_cardinality=rng.randint(10, 1000) if qt == QueryType.AGGREGATE else 0,
        )
        queries.append(q)

    return queries


# ---------------------------------------------------------------------------
# Routing strategy implementations
# ---------------------------------------------------------------------------

class StrategyExecutor:
    """Executes a routing strategy across a query sequence and records
    per-step latencies.

    This is the analog of Megatron's forward_backward schedule — each
    "micro-batch" (query) is routed to hardware and its cost recorded.
    """

    def __init__(self, engine: CostModelEngine):
        self.engine = engine

    def execute_gpu_only(self, queries: List[QueryDescriptor],
                         data_location: str = "cpu0") -> List[float]:
        """All queries go to gpu0 regardless."""
        latencies = []
        for q in queries:
            cb = self.engine.estimate_on_device(q, "gpu0", data_location)
            latencies.append(cb.total_ms)
        return latencies

    def execute_cpu_only(self, queries: List[QueryDescriptor],
                         data_location: str = "cpu0") -> List[float]:
        """All queries go to cpu0."""
        latencies = []
        for q in queries:
            cb = self.engine.estimate_on_device(q, "cpu0", data_location)
            latencies.append(cb.total_ms)
        return latencies

    def execute_hybrid_static(self, queries: List[QueryDescriptor],
                              data_location: str = "cpu0",
                              gpu_threshold_rows: int = 100_000
                              ) -> List[float]:
        """Static threshold: big queries → GPU, small → CPU."""
        latencies = []
        for q in queries:
            if q.estimated_rows > gpu_threshold_rows:
                cb = self.engine.estimate_on_device(q, "gpu0", data_location)
            else:
                cb = self.engine.estimate_on_device(q, "cpu0", data_location)
            latencies.append(cb.total_ms)
        return latencies

    def execute_cost_model_routed(self, queries: List[QueryDescriptor],
                                  data_location: str = "cpu0"
                                  ) -> List[float]:
        """Full cost-model routing: choose min-cost device per query.

        This is the Lynceus core — analogous to NCCL's ncclTopoCompute
        choosing the best ring/tree/collnet algorithm.
        """
        latencies = []
        for q in queries:
            device_id, cb = self.engine.recommend(q, data_location)
            latencies.append(cb.total_ms)
        return latencies

    def execute_par2qo_enhanced(self, queries: List[QueryDescriptor],
                                data_location: str = "cpu0"
                                ) -> List[float]:
        """Cost-model routing + PAR2QO penalty-aware robustness.

        Adds a robustness margin: if GPU cost is within 20% of CPU cost,
        prefer CPU to avoid PCIe transfer variance. This models PAR2QO's
        parametric penalty approach (diagram.py:46 pqoByFeatureCollection).
        """
        latencies = []
        robustness_margin = 0.20
        for q in queries:
            estimates = self.engine.estimate_all_devices(q, data_location)
            gpu_ids = [k for k, n in self.engine.topology.nodes.items()
                       if n.kind.name == "GPU" and k in estimates]
            cpu_ids = [k for k, n in self.engine.topology.nodes.items()
                       if n.kind.name == "CPU" and k in estimates]

            if not gpu_ids or not cpu_ids:
                # Fallback to basic routing
                device_id, cb = self.engine.recommend(q, data_location)
                latencies.append(cb.total_ms)
                continue

            best_gpu = min(gpu_ids, key=lambda k: estimates[k].total_us)
            best_cpu = min(cpu_ids, key=lambda k: estimates[k].total_us)

            gpu_cost = estimates[best_gpu].total_us
            cpu_cost = estimates[best_cpu].total_us

            # PAR2QO penalty: prefer CPU when GPU advantage is marginal
            if gpu_cost < cpu_cost * (1.0 - robustness_margin):
                latencies.append(estimates[best_gpu].total_ms)
            else:
                latencies.append(estimates[best_cpu].total_ms)

        return latencies

    def execute_strategy(self, strategy: RoutingStrategy,
                         queries: List[QueryDescriptor],
                         data_location: str = "cpu0") -> List[float]:
        """Dispatch to strategy-specific executor."""
        dispatch = {
            RoutingStrategy.GPU_ONLY: self.execute_gpu_only,
            RoutingStrategy.CPU_ONLY: self.execute_cpu_only,
            RoutingStrategy.HYBRID_STATIC: self.execute_hybrid_static,
            RoutingStrategy.COST_MODEL_ROUTED: self.execute_cost_model_routed,
            RoutingStrategy.PAR2QO_ENHANCED: self.execute_par2qo_enhanced,
        }
        executor = dispatch.get(strategy)
        if executor is None:
            raise ValueError(f"Unsupported strategy: {strategy}")
        return executor(queries, data_location)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    workload: WorkloadConfig,
    strategies: Optional[List[RoutingStrategy]] = None,
    output_path: Optional[str] = None,
) -> BenchmarkOutput:
    """Run a full benchmark producing data_demo-compatible output.

    For each strategy × seed, generates a full query sequence and
    records per-step latency, producing the same schema as
    ppl_vs_time_1B_30k_data.json.

    Args:
        workload: Workload configuration.
        strategies: List of strategies to benchmark. Default: all 5.
        output_path: If provided, save JSON output to this path.

    Returns:
        BenchmarkOutput with panels/methods/seeds populated.
    """
    if strategies is None:
        strategies = [
            RoutingStrategy.GPU_ONLY,
            RoutingStrategy.CPU_ONLY,
            RoutingStrategy.HYBRID_STATIC,
            RoutingStrategy.COST_MODEL_ROUTED,
            RoutingStrategy.PAR2QO_ENHANCED,
        ]

    # Initialize
    topology = create_default_topology()
    engine = CostModelEngine(topology)
    executor = StrategyExecutor(engine)

    output = BenchmarkOutput(
        description=f"Lynceus benchmark — {workload.name}",
        source="lynceus_benchmark_runner",
    )

    # Panel: latency vs step
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

            # Accumulate total cost
            if method.total_cost is None:
                method.total_cost = 0.0
            method.total_cost += sum(latencies)

        method.total_cost = (method.total_cost or 0.0) / workload.num_seeds
        method.compute_statistics()

    # Save if path provided
    if output_path:
        output.save(output_path)

    return output


# ---------------------------------------------------------------------------
# Cumulative latency panel (PPL-vs-time analog)
# ---------------------------------------------------------------------------

def run_cumulative_benchmark(
    workload: WorkloadConfig,
    strategies: Optional[List[RoutingStrategy]] = None,
    output_path: Optional[str] = None,
) -> BenchmarkOutput:
    """Like run_benchmark but Y-axis is cumulative latency (total time).

    Analogous to ppl_vs_time_1B_30k_data.json where X = time_hours.
    """
    if strategies is None:
        strategies = [
            RoutingStrategy.GPU_ONLY,
            RoutingStrategy.CPU_ONLY,
            RoutingStrategy.HYBRID_STATIC,
            RoutingStrategy.COST_MODEL_ROUTED,
            RoutingStrategy.PAR2QO_ENHANCED,
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

            # Cumulative sum
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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Run default benchmark and save output."""
    import sys

    workload = WorkloadConfig(
        name="TPC-H_SF100",
        num_steps=2000,
        num_seeds=3,
    )

    print(f"Running Lynceus benchmark: {workload.name}")
    print(f"  Steps: {workload.num_steps}, Seeds: {workload.num_seeds}")

    # Per-step latency
    output1 = run_benchmark(
        workload,
        output_path="output/latency_vs_step.json",
    )
    print(f"\nPer-step latency data saved.")
    for name, panel in output1.panels.items():
        for mname, mr in panel.methods.items():
            print(f"  {mname}: final_mean={mr.mean[-1]:.3f}ms, "
                  f"total_cost={mr.total_cost:.1f}ms")

    # Cumulative latency
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
