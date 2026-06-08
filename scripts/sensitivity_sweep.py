#!/usr/bin/env python3
"""
scripts/sensitivity_sweep.py — 参数敏感性扫描 (M261-M270)

对3个关键路由参数做网格搜索 (每参数5个值, 200步×1种子):
  1. HybridStaticStrategy.gpu_threshold_rows
  2. AdaptiveStrategy.initial_temperature
  3. PAR2QOEnhancedStrategy.base_margin

每组参数独立实例化策略，用 phase-shift 负载跑完 200 步后
汇聚延迟均值、P95、路由熵、收敛CV。

输出: output/sensitivity_sweep.json

算法复用 strategy_comparison.py:
  - build_phaseshift_workload: 三阶段负载生成
  - WelfordAccumulator / KahanSum: 在线统计
  - shannon_entropy / percentile / check_convergence

用法:
    python scripts/sensitivity_sweep.py
    python scripts/sensitivity_sweep.py --steps 200 --seed 0
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LYNCEUS_DEBUG", "0")

from lynceus.costing import (
    CostModelEngine, QueryDescriptor, QueryType, create_default_topology,
)
from lynceus.pipeline_scheduler import QueryPipelineScheduler
from lynceus.cache_manager import TopologyCacheManager
from lynceus.strategies.static import HybridStaticStrategy
from lynceus.strategies.cost_driven import PAR2QOEnhancedStrategy
from lynceus.strategies.adaptive import AdaptiveStrategy

# 复用 strategy_comparison 的基础设施
from scripts.strategy_comparison import (
    build_phaseshift_workload,
    WelfordAccumulator,
    KahanSum,
    shannon_entropy,
    percentile,
    check_convergence,
)


# ═══════════════════════════════════════════════════════════════
# 调试基础设施
# ═══════════════════════════════════════════════════════════════

def _dbg(msg: str, **kw) -> None:
    """条件调试打印, 全部到 stderr."""
    print(f"  │ {msg}" + (": " + ", ".join(
        f"{k}={v}" for k, v in kw.items()) if kw else ""), file=sys.stderr)


def _dump_header(title: str) -> None:
    print(f"\n┌─{'─'*60}", file=sys.stderr)
    print(f"│ SENSITIVITY SWEEP: {title}", file=sys.stderr)
    print(f"├─{'─'*60}", file=sys.stderr)


def _dump_footer(summary: str) -> None:
    print(f"├─{'─'*60}", file=sys.stderr)
    print(f"└─ {summary}", file=sys.stderr)
    print(file=sys.stderr)


# ═══════════════════════════════════════════════════════════════
# 参数网格定义
# ═══════════════════════════════════════════════════════════════

# HybridStaticStrategy.gpu_threshold_rows: 默认 90_000
# 扫描范围覆盖从非常保守(少量路由到GPU)到激进(大量路由到GPU)
THRESHOLD_GRID = [10_000, 50_000, 90_000, 200_000, 500_000]

# AdaptiveStrategy.initial_temperature: 默认 50.0
# 低温=更确定性选择, 高温=更随机探索
TEMPERATURE_GRID = [5.0, 20.0, 50.0, 100.0, 200.0]

# PAR2QOEnhancedStrategy.base_margin: 默认 0.18
# 低margin=GPU更容易被选中, 高margin=CPU更保守被偏好
MARGIN_GRID = [0.02, 0.10, 0.18, 0.30, 0.50]


# ═══════════════════════════════════════════════════════════════
# 单策略单参数点运行
# ═══════════════════════════════════════════════════════════════

def run_single_config(
    strategy,
    scheduler: QueryPipelineScheduler,
    topo,
    workload: List[QueryDescriptor],
) -> Dict:
    """运行单策略配置, 返回统计摘要 (不含原始延迟列表以节省内存)."""
    cache_mgr = TopologyCacheManager(topo, cache_fraction=0.5)
    cache_mgr.reset()

    latencies = []
    route_counts = defaultdict(int)
    welford = WelfordAccumulator()
    cumulative = KahanSum()

    for q in workload:
        decision = strategy.route_one(q, data_location="cpu0")
        dev = decision.device_id

        sched = scheduler.schedule(q, data_location="cpu0")
        lat = sched.latency_us

        latencies.append(lat)
        route_counts[dev] += 1
        welford.update(lat)
        cumulative.add(lat)

        # 缓存模拟
        gpu_cache = cache_mgr.get(dev)
        if gpu_cache:
            blocks = gpu_cache.required_blocks(q)
            gpu_cache.lookup(blocks)
            gpu_cache.release(blocks)

        # Adaptive 策略反馈
        if hasattr(strategy, 'observe_with_estimate') and decision.cost:
            strategy.observe_with_estimate(
                dev, decision.cost.total_us, lat)

    sorted_all = sorted(latencies)
    converged, cv = check_convergence(latencies)

    return {
        "mean": welford.mean,
        "std": welford.std,
        "total_cost": cumulative.total,
        "p50": percentile(sorted_all, 50),
        "p95": percentile(sorted_all, 95),
        "p99": percentile(sorted_all, 99),
        "route_distribution": dict(route_counts),
        "shannon_entropy": shannon_entropy(dict(route_counts)),
        "cache_hit_rate": cache_mgr.aggregate_hit_rate(),
        "converged": converged,
        "tail_cv": cv,
    }


# ═══════════════════════════════════════════════════════════════
# 敏感性扫描: 每策略×参数网格
# ═══════════════════════════════════════════════════════════════

def sweep_hybrid_threshold(
    engine: CostModelEngine,
    scheduler: QueryPipelineScheduler,
    topo,
    workload: List[QueryDescriptor],
) -> Dict:
    """扫描 HybridStaticStrategy.gpu_threshold_rows."""
    _dump_header("HybridStaticStrategy — gpu_threshold_rows")
    results = {}

    for val in THRESHOLD_GRID:
        strategy = HybridStaticStrategy(engine, gpu_threshold_rows=val)
        data = run_single_config(strategy, scheduler, topo, workload)
        results[str(val)] = data
        _dbg(f"threshold={val:>7d}",
             mean=f"{data['mean']:.1f}µs",
             p95=f"{data['p95']:.1f}µs",
             entropy=f"{data['shannon_entropy']:.3f}",
             routes=data['route_distribution'])

    # 找最优参数点 (按mean排序)
    best_key = min(results, key=lambda k: results[k]["mean"])
    _dump_footer(
        f"Best threshold={best_key} → mean={results[best_key]['mean']:.1f}µs")

    return {
        "parameter": "gpu_threshold_rows",
        "strategy": "HybridStaticStrategy",
        "grid_values": THRESHOLD_GRID,
        "default_value": 90_000,
        "results": results,
        "best_value": int(best_key),
        "best_mean": results[best_key]["mean"],
    }


def sweep_adaptive_temperature(
    engine: CostModelEngine,
    scheduler: QueryPipelineScheduler,
    topo,
    workload: List[QueryDescriptor],
) -> Dict:
    """扫描 AdaptiveStrategy.initial_temperature."""
    _dump_header("AdaptiveStrategy — initial_temperature")
    results = {}

    for val in TEMPERATURE_GRID:
        strategy = AdaptiveStrategy(engine, initial_temperature=val)
        data = run_single_config(strategy, scheduler, topo, workload)
        results[str(val)] = data
        _dbg(f"temperature={val:>6.1f}",
             mean=f"{data['mean']:.1f}µs",
             p95=f"{data['p95']:.1f}µs",
             entropy=f"{data['shannon_entropy']:.3f}",
             routes=data['route_distribution'])

    best_key = min(results, key=lambda k: results[k]["mean"])
    _dump_footer(
        f"Best temperature={best_key} → mean={results[best_key]['mean']:.1f}µs")

    return {
        "parameter": "initial_temperature",
        "strategy": "AdaptiveStrategy",
        "grid_values": TEMPERATURE_GRID,
        "default_value": 50.0,
        "results": results,
        "best_value": float(best_key),
        "best_mean": results[best_key]["mean"],
    }


def sweep_par2qo_margin(
    engine: CostModelEngine,
    scheduler: QueryPipelineScheduler,
    topo,
    workload: List[QueryDescriptor],
) -> Dict:
    """扫描 PAR2QOEnhancedStrategy.base_margin."""
    _dump_header("PAR2QOEnhancedStrategy — base_margin")
    results = {}

    for val in MARGIN_GRID:
        strategy = PAR2QOEnhancedStrategy(engine, base_margin=val)
        data = run_single_config(strategy, scheduler, topo, workload)
        results[str(val)] = data
        _dbg(f"margin={val:.2f}",
             mean=f"{data['mean']:.1f}µs",
             p95=f"{data['p95']:.1f}µs",
             entropy=f"{data['shannon_entropy']:.3f}",
             routes=data['route_distribution'])

    best_key = min(results, key=lambda k: results[k]["mean"])
    _dump_footer(
        f"Best margin={best_key} → mean={results[best_key]['mean']:.1f}µs")

    return {
        "parameter": "base_margin",
        "strategy": "PAR2QOEnhancedStrategy",
        "grid_values": MARGIN_GRID,
        "default_value": 0.18,
        "results": results,
        "best_value": float(best_key),
        "best_mean": results[best_key]["mean"],
    }


# ═══════════════════════════════════════════════════════════════
# 主实验: 3参数 × 5值 网格扫描
# ═══════════════════════════════════════════════════════════════

def run_sensitivity_sweep(
    n_steps: int = 200,
    seed: int = 0,
    output_dir: str = "output",
) -> Dict:
    """运行完整的三参数敏感性扫描。"""
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    # 构建共享基础设施
    topo = create_default_topology()
    engine = CostModelEngine(topo)
    scheduler = QueryPipelineScheduler(engine)

    _dump_header("EXPERIMENT CONFIG")
    _dbg("n_steps", val=n_steps)
    _dbg("seed", val=seed)
    _dbg("grid_sizes",
         threshold=f"{len(THRESHOLD_GRID)} values",
         temperature=f"{len(TEMPERATURE_GRID)} values",
         margin=f"{len(MARGIN_GRID)} values")
    total_runs = len(THRESHOLD_GRID) + len(TEMPERATURE_GRID) + len(MARGIN_GRID)
    _dbg("total_runs", val=total_runs)
    _dump_footer(
        f"3 parameters × 5 values = {total_runs} runs × {n_steps} steps")

    # 生成共享负载 (所有扫描使用同一负载, 确保可比性)
    workload = build_phaseshift_workload(n_steps, seed=seed)
    _dbg(f"workload generated: {len(workload)} queries, seed={seed}")

    # ─── 三参数扫描 ───
    sweep_threshold = sweep_hybrid_threshold(
        engine, scheduler, topo, workload)
    sweep_temperature = sweep_adaptive_temperature(
        engine, scheduler, topo, workload)
    sweep_margin = sweep_par2qo_margin(
        engine, scheduler, topo, workload)

    elapsed = time.time() - t0

    # ═══════════════════════════════════════════════════════════
    # 交叉分析: 参数敏感度排名
    # ═══════════════════════════════════════════════════════════

    _dump_header("SENSITIVITY RANKING")

    sensitivity_scores = {}
    for sweep in [sweep_threshold, sweep_temperature, sweep_margin]:
        means = [v["mean"] for v in sweep["results"].values()]
        # 敏感度 = max-min 的范围 / 均值, 即相对变幅
        range_val = max(means) - min(means)
        avg_val = sum(means) / len(means) if means else 1.0
        relative_sensitivity = range_val / avg_val if avg_val > 1e-9 else 0.0

        param_name = sweep["parameter"]
        sensitivity_scores[param_name] = {
            "range_us": round(range_val, 2),
            "mean_us": round(avg_val, 2),
            "relative_sensitivity": round(relative_sensitivity, 4),
            "best_value": sweep["best_value"],
            "best_mean_us": round(sweep["best_mean"], 2),
            "default_value": sweep["default_value"],
        }
        _dbg(f"{param_name}",
             range=f"{range_val:.2f}µs",
             rel=f"{relative_sensitivity:.4f}",
             best=sweep["best_value"])

    ranked_params = sorted(
        sensitivity_scores.keys(),
        key=lambda k: sensitivity_scores[k]["relative_sensitivity"],
        reverse=True)

    for rank, param in enumerate(ranked_params):
        s = sensitivity_scores[param]
        _dbg(f"Rank #{rank+1}: {param}",
             relative_sensitivity=s["relative_sensitivity"],
             range=f"{s['range_us']:.2f}µs")

    _dump_footer(
        f"Most sensitive: {ranked_params[0]} "
        f"(rel={sensitivity_scores[ranked_params[0]]['relative_sensitivity']:.4f})")

    # ═══════════════════════════════════════════════════════════
    # 输出 JSON
    # ═══════════════════════════════════════════════════════════

    output = {
        "metadata": {
            "panel": "Sensitivity Sweep — 3 Parameters × 5 Values Grid Search",
            "n_steps": n_steps,
            "seed": seed,
            "total_runs": total_runs,
            "elapsed_seconds": round(elapsed, 2),
            "workload": "phase-shift 3-phase (SCAN→JOIN→AGG)",
            "parameters_swept": {
                "gpu_threshold_rows": {
                    "strategy": "HybridStaticStrategy",
                    "grid": THRESHOLD_GRID,
                    "default": 90_000,
                },
                "initial_temperature": {
                    "strategy": "AdaptiveStrategy",
                    "grid": TEMPERATURE_GRID,
                    "default": 50.0,
                },
                "base_margin": {
                    "strategy": "PAR2QOEnhancedStrategy",
                    "grid": MARGIN_GRID,
                    "default": 0.18,
                },
            },
        },
        "sweeps": {
            "gpu_threshold_rows": sweep_threshold,
            "initial_temperature": sweep_temperature,
            "base_margin": sweep_margin,
        },
        "sensitivity_ranking": ranked_params,
        "sensitivity_scores": sensitivity_scores,
    }

    out_path = os.path.join(output_dir, "sensitivity_sweep.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    _dump_header("SWEEP COMPLETE")
    _dbg("elapsed", seconds=f"{elapsed:.1f}")
    _dbg("output", path=out_path)
    _dbg("runs", total=total_runs)
    for param in ranked_params:
        s = sensitivity_scores[param]
        _dbg(f"  {param}",
             best=s["best_value"],
             mean=f"{s['best_mean_us']:.1f}µs",
             sensitivity=s["relative_sensitivity"])
    _dump_footer(f"File: {out_path}")

    return output


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Lynceus Parameter Sensitivity Sweep (M261-M270)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Number of workload steps per run (default: 200)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for workload generation (default: 0)")
    parser.add_argument("--output-dir", default="output",
                        help="Output directory (default: output)")
    args = parser.parse_args()

    run_sensitivity_sweep(args.steps, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
