#!/usr/bin/env python3
"""multi_workload_bench.py – 多负载×多策略论文数据生成

3 workloads × 6 strategies × N steps × K seeds
Outputs output/multi_workload.json with fig1–fig4 data blocks.
"""

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lynceus.costing import (
    CostModelEngine,
    QueryDescriptor,
    QueryType,
    create_default_topology,
)
from lynceus.pipeline_scheduler import QueryPipelineScheduler
from lynceus.cache_manager import TopologyCacheManager
from lynceus.strategies.static import (
    GPUOnlyStrategy,
    CPUOnlyStrategy,
    HybridStaticStrategy,
)
from lynceus.strategies.cost_driven import (
    CostModelRoutedStrategy,
    PAR2QOEnhancedStrategy,
)
from lynceus.strategies.adaptive import AdaptiveStrategy

# ─────────────────────────── numerical helpers ───────────────────────────


@dataclass
class WelfordAccumulator:
    """Welford's online algorithm for streaming mean / variance."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def variance(self) -> float:
        return self.m2 / self.n if self.n >= 2 else 0.0

    def stddev(self) -> float:
        return math.sqrt(self.variance())


@dataclass
class KahanSum:
    """Kahan compensated summation – reduces floating-point drift."""

    total: float = 0.0
    comp: float = 0.0  # running compensation

    def add(self, x: float) -> float:
        y = x - self.comp
        t = self.total + y
        self.comp = (t - self.total) - y
        self.total = t
        return self.total


def shannon_entropy(counts: Dict[str, int]) -> float:
    """Shannon entropy H over a distribution given by raw counts."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def smape(a: float, b: float) -> float:
    """Symmetric Mean Absolute Percentage Error (single-pair)."""
    denom = abs(a) + abs(b)
    if denom == 0.0:
        return 0.0
    return 2.0 * abs(a - b) / denom


def gini_coefficient(values: List[float]) -> float:
    """Gini coefficient measuring inequality in *values* (0 = equal, 1 = max)."""
    n = len(values)
    if n == 0:
        return 0.0
    sorted_v = sorted(values)
    cumsum = 0.0
    weighted_sum = 0.0
    for i, v in enumerate(sorted_v):
        cumsum += v
        weighted_sum += (2 * (i + 1) - n - 1) * v
    total = cumsum
    if total == 0.0:
        return 0.0
    return weighted_sum / (n * total)


# ─────────────────────────── debug helpers ───────────────────────────


def _dbg(msg: str) -> None:
    sys.stderr.write(msg + "\n")


def state_dump(label: str, kvs: List[Tuple[str, str]]) -> None:
    """Print a ┌│└ STATE DUMP block to stderr."""
    _dbg(f"┌─── STATE DUMP: {label}")
    for k, v in kvs:
        _dbg(f"│  {k}: {v}")
    _dbg(f"└─── END {label}")


# ─────────────────────────── workload definitions ────────────────────────

WORKLOAD_PROFILES: Dict[str, Dict[QueryType, float]] = {
    "TPC-H": {
        QueryType.RANGE_SCAN: 0.25,
        QueryType.FULL_TABLE_SCAN: 0.25,
        QueryType.JOIN: 0.25,
        QueryType.AGGREGATE: 0.15,
        QueryType.INDEX_SCAN: 0.10,
    },
    "TPC-DS": {
        QueryType.RANGE_SCAN: 0.15,
        QueryType.FULL_TABLE_SCAN: 0.15,
        QueryType.JOIN: 0.30,
        QueryType.AGGREGATE: 0.20,
        QueryType.INDEX_SCAN: 0.20,
    },
    "YCSB": {
        QueryType.RANGE_SCAN: 0.05,
        QueryType.FULL_TABLE_SCAN: 0.05,
        QueryType.JOIN: 0.10,
        QueryType.AGGREGATE: 0.10,
        QueryType.INDEX_SCAN: 0.70,
    },
}

TABLE_NAMES = ["lineitem", "orders", "customer", "part", "supplier", "nation"]

STRATEGY_SPECS = [
    ("GPU-Only", GPUOnlyStrategy),
    ("CPU-Only", CPUOnlyStrategy),
    ("Hybrid-Static", HybridStaticStrategy),
    ("CostModel-Routed", CostModelRoutedStrategy),
    ("PAR2QO-Enhanced", PAR2QOEnhancedStrategy),
    ("Adaptive", AdaptiveStrategy),
]


def _weighted_choice(rng: random.Random, dist: Dict[QueryType, float]) -> QueryType:
    """Pick a QueryType according to the workload probability distribution."""
    r = rng.random()
    cumulative = 0.0
    for qt, prob in dist.items():
        cumulative += prob
        if r <= cumulative:
            return qt
    return list(dist.keys())[-1]


def _make_query(
    rng: random.Random,
    step: int,
    idx: int,
    qt: QueryType,
) -> QueryDescriptor:
    """Synthesise a realistic QueryDescriptor for the given type."""
    table = rng.choice(TABLE_NAMES)
    table_rows = rng.randint(10_000, 10_000_000)

    is_scan = qt in (QueryType.RANGE_SCAN, QueryType.FULL_TABLE_SCAN)
    is_join = qt == QueryType.JOIN
    is_agg = qt == QueryType.AGGREGATE

    selectivity = rng.uniform(0.0001, 0.01) if not is_scan else rng.uniform(0.01, 0.5)
    est_rows = max(1, int(table_rows * selectivity))
    width = rng.randint(40, 400)
    n_pred = rng.randint(1, 6)
    idx_avail = qt in (QueryType.INDEX_SCAN, QueryType.POINT_LOOKUP) or rng.random() < 0.3
    idx_depth = rng.randint(2, 5) if idx_avail else 0
    n_joins = rng.randint(2, 5) if is_join else 0
    sort_req = rng.random() < 0.4
    gb_card = rng.randint(2, 500) if is_agg else 0

    return QueryDescriptor(
        query_id=f"s{step}_q{idx}",
        query_type=qt,
        estimated_rows=est_rows,
        estimated_width_bytes=width,
        num_predicates=n_pred,
        selectivity=selectivity,
        table_rows=table_rows,
        index_available=idx_avail,
        index_depth=idx_depth,
        num_joins=n_joins,
        sort_required=sort_req,
        group_by_cardinality=gb_card,
        table_name=table,
    )


# ─────────────────────────── instantiation helpers ───────────────────────


def _build_strategy(cls, engine, topo, cache_mgr, scheduler):
    """Try a few common constructor signatures to build a strategy instance."""
    # Each strategy class has slightly different ctor args; try most-specific first.
    for args in [
        (engine, topo, cache_mgr, scheduler),
        (engine, topo, cache_mgr),
        (engine, topo, scheduler),
        (engine, topo),
        (engine,),
        (topo,),
        (),
    ]:
        try:
            return cls(*args)
        except TypeError:
            continue
    raise RuntimeError(f"Cannot instantiate {cls.__name__}")


# ─────────────────────────── main benchmark ──────────────────────────────


def run_benchmark(steps: int, seeds: int) -> dict:
    topo = create_default_topology()
    engine = CostModelEngine(topo)
    scheduler = QueryPipelineScheduler(engine)
    cache_mgr = TopologyCacheManager(topo, cache_fraction=0.5)

    # result accumulators
    fig1: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    fig2: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    fig3: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    fig4: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    total_combos = len(WORKLOAD_PROFILES) * len(STRATEGY_SPECS) * seeds
    done = 0

    for wl_name, wl_dist in WORKLOAD_PROFILES.items():
        for strat_name, strat_cls in STRATEGY_SPECS:
            # accumulators across seeds (Welford per step)
            step_welford: Dict[int, WelfordAccumulator] = defaultdict(WelfordAccumulator)
            seed_cost_curves: List[List[float]] = []
            seed_cache_curves: List[List[float]] = []
            seed_device_counts: Dict[str, int] = defaultdict(int)

            for seed_idx in range(seeds):
                rng = random.Random(42 + seed_idx)
                strategy = _build_strategy(strat_cls, engine, topo, cache_mgr, scheduler)

                kahan = KahanSum()
                cost_curve: List[float] = []
                cache_hits = 0
                cache_total = 0
                cache_curve: List[float] = []

                state_dump(
                    f"{wl_name}/{strat_name}/seed={seed_idx}",
                    [
                        ("steps", str(steps)),
                        ("workload_dist", str({k.name: v for k, v in wl_dist.items()})),
                        ("strategy_class", strat_cls.__name__),
                    ],
                )

                for step in range(steps):
                    # generate a small batch per step (1-3 queries)
                    batch_size = rng.randint(1, 3)
                    step_latency = 0.0

                    for qi in range(batch_size):
                        qt = _weighted_choice(rng, wl_dist)
                        q = _make_query(rng, step, qi, qt)

                        decision = strategy.route_one(q, data_location="cpu0")
                        device_id = decision.device_id if decision else "cpu0"
                        cost = decision.cost.total_us if decision and decision.cost else 0.0

                        # 用策略选择的device计算真实延迟（含transfer）
                        # 而非scheduler重新做路由决策覆盖策略
                        try:
                            dev_cb = engine.estimate_on_device(q, device_id, "cpu0")
                            lat = dev_cb.total_us
                        except Exception:
                            lat = cost
                        # 添加噪声模拟真实执行波动 (±3%)
                        noise_factor = 1.0 + (rng.gauss(0, 0.015))
                        lat *= max(0.9, noise_factor)

                        step_latency += lat
                        kahan.add(lat)
                        seed_device_counts[device_id] += 1

                        # cache simulation
                        cache_total += 1
                        try:
                            cached = cache_mgr.lookup(q.table_name, q.query_id)
                            if cached:
                                cache_hits += 1
                            else:
                                cache_mgr.insert(q.table_name, q.query_id, cost)
                        except Exception:
                            pass

                    avg_lat = step_latency / batch_size
                    step_welford[step].update(avg_lat)
                    cost_curve.append(kahan.total)
                    cache_curve.append(
                        cache_hits / cache_total if cache_total else 0.0
                    )

                seed_cost_curves.append(cost_curve)
                seed_cache_curves.append(cache_curve)
                done += 1
                _dbg(f"│  progress {done}/{total_combos}")

            # ── aggregate across seeds ──
            key = f"{wl_name}/{strat_name}"

            # fig1: latency mean per step (from Welford)
            fig1_curve: List[float] = []
            for s in range(steps):
                fig1_curve.append(step_welford[s].mean)
            fig1[wl_name][strat_name] = fig1_curve

            # fig2: cumulative cost (average across seeds)
            avg_cost: List[float] = []
            for s in range(steps):
                vals = [sc[s] for sc in seed_cost_curves if s < len(sc)]
                avg_cost.append(sum(vals) / len(vals) if vals else 0.0)
            fig2[wl_name][strat_name] = avg_cost

            # fig3: routing distribution
            total_routed = sum(seed_device_counts.values())
            dist_pct: Dict[str, float] = {}
            if total_routed > 0:
                for dev, cnt in seed_device_counts.items():
                    dist_pct[dev] = cnt / total_routed
            fig3[wl_name][strat_name] = dist_pct

            # fig4: cache hit rate (average across seeds)
            avg_cache: List[float] = []
            for s in range(steps):
                vals = [cc[s] for cc in seed_cache_curves if s < len(cc)]
                avg_cache.append(sum(vals) / len(vals) if vals else 0.0)
            fig4[wl_name][strat_name] = avg_cache

            # ── derived metrics (stderr) ──
            final_costs = [sc[-1] for sc in seed_cost_curves if sc]
            gini = gini_coefficient(list(seed_device_counts.values()))

            # entropy of routing distribution
            int_counts = {k: int(v) for k, v in seed_device_counts.items()}
            ent = shannon_entropy(int_counts)

            # SMAPE between first and last Welford means
            first_mean = step_welford[0].mean if 0 in step_welford else 0.0
            last_mean = step_welford[steps - 1].mean if (steps - 1) in step_welford else 0.0
            sm = smape(first_mean, last_mean)

            # final step Welford stddev
            final_std = step_welford[steps - 1].stddev() if (steps - 1) in step_welford else 0.0

            state_dump(
                f"METRICS {key}",
                [
                    ("welford_final_mean", f"{last_mean:.4f}"),
                    ("welford_final_std", f"{final_std:.4f}"),
                    ("kahan_total_cost_avg", f"{avg_cost[-1]:.4f}" if avg_cost else "N/A"),
                    ("shannon_entropy", f"{ent:.4f}"),
                    ("smape_first_last", f"{sm:.4f}"),
                    ("gini_device", f"{gini:.4f}"),
                    ("cache_hit_final", f"{avg_cache[-1]:.4f}" if avg_cache else "N/A"),
                ],
            )

    # ── assemble output ──
    result = {
        "metadata": {
            "steps": steps,
            "seeds": seeds,
            "workloads": list(WORKLOAD_PROFILES.keys()),
            "strategies": [s[0] for s in STRATEGY_SPECS],
        },
        "fig1_data": {
            wl: {st: curve for st, curve in strats.items()}
            for wl, strats in fig1.items()
        },
        "fig2_data": {
            wl: {st: curve for st, curve in strats.items()}
            for wl, strats in fig2.items()
        },
        "fig3_data": {
            wl: {st: dist for st, dist in strats.items()}
            for wl, strats in fig3.items()
        },
        "fig4_data": {
            wl: {st: curve for st, curve in strats.items()}
            for wl, strats in fig4.items()
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-workload × multi-strategy paper data generation"
    )
    parser.add_argument("--steps", type=int, default=500, help="simulation steps")
    parser.add_argument("--seeds", type=int, default=2, help="random seeds")
    args = parser.parse_args()

    _dbg(f"┌─── multi_workload_bench  steps={args.steps}  seeds={args.seeds}")
    _dbg(f"│  workloads: {list(WORKLOAD_PROFILES.keys())}")
    _dbg(f"│  strategies: {[s[0] for s in STRATEGY_SPECS]}")
    _dbg(f"└─── starting benchmark")

    result = run_benchmark(args.steps, args.seeds)

    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", "multi_workload.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    _dbg(f"┌─── DONE")
    _dbg(f"│  wrote {out_path}  ({os.path.getsize(out_path)} bytes)")
    _dbg(f"└───")


if __name__ == "__main__":
    main()
