#!/usr/bin/env python3
"""
scripts/ablation_study.py — 消融实验 (第三位Claude, M141-M160)

基于子模型(opus 4.6)生成的骨架，由监督Claude修复API兼容性。
作者: dylanyunlon <dogechat@163.com>

7项消融: 逐项关闭算法改写，测量 baseline vs ablated 的 delta。

算法改写 (~20%):
  1. Cohen's d 效应量: d = (m1-m2) / pooled_std
  2. Bootstrap 置信区间: 1000次重采样的 [2.5%, 97.5%]
  3. Welford 在线方差 (复用第二位Claude的)
  4. Kahan 补偿求和 (复用第二位Claude的)
  5. 非参数 Wilcoxon 符号秩近似 (z-score)

用法:
    python scripts/ablation_study.py --steps 500 --seeds 2
    python scripts/ablation_study.py --steps 50 --seeds 1   # smoke test
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("LYNCEUS_DEBUG", "0")

from lynceus.costing import (
    CostModelEngine, QueryDescriptor, QueryType, create_default_topology,
)
from lynceus.pipeline_scheduler import QueryPipelineScheduler
from lynceus.cache_manager import TopologyCacheManager
from lynceus.strategies.static import GPUOnlyStrategy, CPUOnlyStrategy, HybridStaticStrategy
from lynceus.strategies.cost_driven import CostModelRoutedStrategy, PAR2QOEnhancedStrategy
from lynceus.strategies.adaptive import AdaptiveStrategy


# ═══════════════════════════════════════════════════════════════
# 调试基础设施
# ═══════════════════════════════════════════════════════════════

def _dbg(msg: str, **kw) -> None:
    print(f"  │ {msg}" + (": " + ", ".join(
        f"{k}={v}" for k, v in kw.items()) if kw else ""), file=sys.stderr)

def _dump_header(title: str) -> None:
    print(f"\n┌─{'─'*60}", file=sys.stderr)
    print(f"│ STATE DUMP: {title}", file=sys.stderr)
    print(f"├─{'─'*60}", file=sys.stderr)

def _dump_footer(summary: str) -> None:
    print(f"├─{'─'*60}", file=sys.stderr)
    print(f"└─ {summary}", file=sys.stderr)
    print(file=sys.stderr)


# ═══════════════════════════════════════════════════════════════
# 算法工具 (复用+扩展第二位Claude的)
# ═══════════════════════════════════════════════════════════════

class WelfordAccumulator:
    """Welford单pass在线方差。"""
    __slots__ = ('n', 'mean', '_m2')
    def __init__(self):
        self.n = 0; self.mean = 0.0; self._m2 = 0.0
    def update(self, x: float) -> None:
        self.n += 1; delta = x - self.mean
        self.mean += delta / self.n; self._m2 += delta * (x - self.mean)
    @property
    def variance(self) -> float:
        return self._m2 / max(1, self.n - 1) if self.n > 1 else 0.0
    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


def kahan_sum(values: List[float]) -> float:
    """Kahan补偿求和。"""
    s, c = 0.0, 0.0
    for v in values:
        y = v - c; t = s + y; c = (t - s) - y; s = t
    return s


def naive_sum(values: List[float]) -> float:
    """朴素累加（消融对照）。"""
    s = 0.0
    for v in values:
        s += v
    return s


def naive_variance(values: List[float]) -> Tuple[float, float]:
    """朴素two-pass方差（消融对照）。"""
    if len(values) < 2:
        return (values[0] if values else 0.0, 0.0)
    m = sum(values) / len(values)
    v = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return m, math.sqrt(v)


# ═══════════════════════════════════════════════════════════════
# 算法改写 #1: Cohen's d 效应量
# ═══════════════════════════════════════════════════════════════

def cohens_d(group_a: List[float], group_b: List[float]) -> float:
    """Cohen's d = (mean_a - mean_b) / pooled_std.
    
    效应量判定: |d|<0.2 小, 0.2-0.8 中, >0.8 大
    """
    if not group_a or not group_b:
        return 0.0
    wa, wb = WelfordAccumulator(), WelfordAccumulator()
    for v in group_a: wa.update(v)
    for v in group_b: wb.update(v)
    na, nb = wa.n, wb.n
    pooled_var = ((na - 1) * wa.variance + (nb - 1) * wb.variance) / max(1, na + nb - 2)
    pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1e-15
    return (wa.mean - wb.mean) / pooled_std


def effect_size_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2: return "negligible"
    elif ad < 0.5: return "small"
    elif ad < 0.8: return "medium"
    else: return "large"


# ═══════════════════════════════════════════════════════════════
# 算法改写 #2: Bootstrap 置信区间
# ═══════════════════════════════════════════════════════════════

def bootstrap_ci(values: List[float], n_resamples: int = 1000,
                 ci: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap置信区间。返回 (lower, mean, upper)。
    
    1000次有放回重采样，取排序后的 [2.5%, 97.5%] 分位。
    """
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(kahan_sum(sample) / n)
    means.sort()
    alpha = (1 - ci) / 2
    lo_idx = max(0, int(alpha * n_resamples))
    hi_idx = min(n_resamples - 1, int((1 - alpha) * n_resamples))
    return (means[lo_idx], kahan_sum(means) / len(means), means[hi_idx])


# ═══════════════════════════════════════════════════════════════
# 算法改写 #3: Wilcoxon 符号秩近似
# ═══════════════════════════════════════════════════════════════

def wilcoxon_approx(diffs: List[float]) -> Tuple[float, float]:
    """Wilcoxon符号秩检验的正态近似。
    
    返回 (z_score, approx_p_value)。
    非参数配对检验，不假设正态分布。
    """
    nonzero = [(abs(d), 1 if d > 0 else -1) for d in diffs if abs(d) > 1e-15]
    if len(nonzero) < 5:
        return (0.0, 1.0)
    nonzero.sort(key=lambda x: x[0])
    n = len(nonzero)
    # 赋秩
    w_plus = 0.0
    for rank, (_, sign) in enumerate(nonzero, 1):
        if sign > 0:
            w_plus += rank
    # 正态近似
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sigma < 1e-15:
        return (0.0, 1.0)
    z = (w_plus - mu) / sigma
    # 双侧 p-value 近似 (标准正态CDF)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return (z, p)


# ═══════════════════════════════════════════════════════════════
# 负载生成器
# ═══════════════════════════════════════════════════════════════

TABLES = [
    ("lineitem", 6_000_000, 100),
    ("orders",   1_500_000, 80),
    ("partsupp",   800_000, 60),
    ("customer",   150_000, 40),
]
ZIPF_WEIGHTS = [0.6, 0.2, 0.1, 0.1]

# Phase-shift 权重: [SCAN, JOIN, INDEX_SCAN, AGGREGATE]
PHASE_WEIGHTS = [
    [0.60, 0.20, 0.10, 0.10],  # phase 1: scan heavy
    [0.20, 0.50, 0.10, 0.20],  # phase 2: join heavy
    [0.15, 0.15, 0.15, 0.55],  # phase 3: aggregate heavy
]
FIXED_WEIGHTS = [0.25, 0.25, 0.25, 0.25]  # ablated: uniform


def build_workload(n_steps: int, seed: int, use_phase_shift: bool = True) -> List[QueryDescriptor]:
    """构建TPC-H负载。use_phase_shift=False时用固定均匀权重。"""
    queries = []
    one_third = n_steps // 3
    two_third = 2 * n_steps // 3

    for i in range(n_steps):
        h = int(hashlib.md5(f"{seed}:{i}".encode()).hexdigest()[:8], 16)
        # table选择
        u_table = (h & 0xFFFF) / 0xFFFF
        cum, tidx = 0.0, 0
        for wi, w in enumerate(ZIPF_WEIGHTS):
            cum += w
            if u_table < cum: tidx = wi; break
        tname, trows, width = TABLES[tidx]

        # query type权重
        if use_phase_shift:
            if i < one_third: qw = PHASE_WEIGHTS[0]
            elif i < two_third: qw = PHASE_WEIGHTS[1]
            else: qw = PHASE_WEIGHTS[2]
        else:
            qw = FIXED_WEIGHTS

        u_type = ((h >> 16) & 0xFFFF) / 0xFFFF
        cum_qt, qt_idx = 0.0, 0
        for qi, pw in enumerate(qw):
            cum_qt += pw
            if u_type < cum_qt: qt_idx = qi; break

        if qt_idx == 0:
            q = QueryDescriptor(query_id=f"q_{i:05d}", query_type=QueryType.FULL_TABLE_SCAN,
                estimated_rows=max(1, int(trows * 0.1)), table_rows=trows,
                selectivity=0.1, estimated_width_bytes=width, table_name=tname)
        elif qt_idx == 1:
            q = QueryDescriptor(query_id=f"q_{i:05d}", query_type=QueryType.JOIN,
                estimated_rows=max(1, int(trows * 0.03)), table_rows=trows,
                selectivity=0.03, num_joins=2, sort_required=True,
                group_by_cardinality=500, estimated_width_bytes=width, table_name=tname)
        elif qt_idx == 2:
            q = QueryDescriptor(query_id=f"q_{i:05d}", query_type=QueryType.INDEX_SCAN,
                estimated_rows=max(1, int(trows * 0.002)), table_rows=trows,
                selectivity=0.002, index_available=True, index_depth=4,
                estimated_width_bytes=width, table_name=tname)
        else:
            q = QueryDescriptor(query_id=f"q_{i:05d}", query_type=QueryType.AGGREGATE,
                estimated_rows=max(1, int(trows * 0.03)), table_rows=trows,
                selectivity=0.03, group_by_cardinality=7,
                estimated_width_bytes=width, table_name=tname)
        queries.append(q)
    return queries


# ═══════════════════════════════════════════════════════════════
# 模拟运行器
# ═══════════════════════════════════════════════════════════════

def run_simulation(engine, scheduler, cache_mgr, workload,
                   strategy=None) -> List[float]:
    """跑一轮workload，返回延迟列表。"""
    latencies = []
    for q in workload:
        if strategy:
            decision = strategy.route_one(q, data_location="cpu0")
        sched = scheduler.schedule(q, data_location="cpu0")
        latencies.append(sched.latency_us)
        dev = sched.assignments[0].device_id if sched.assignments else "cpu0"
        gpu_cache = cache_mgr.get(dev)
        if gpu_cache:
            blocks = gpu_cache.required_blocks(q)
            gpu_cache.lookup(blocks)
            gpu_cache.release(blocks)
    return latencies


def collect_latencies(n_steps: int, seeds: List[int],
                      topo_factory=None, strategy_factory=None,
                      workload_factory=None) -> List[float]:
    """跨seed收集所有延迟值。"""
    all_lats = []
    for seed in seeds:
        topo = topo_factory() if topo_factory else create_default_topology()
        engine = CostModelEngine(topo)
        scheduler = QueryPipelineScheduler(engine)
        cache_mgr = TopologyCacheManager(topo, cache_fraction=0.5)
        strategy = strategy_factory(engine) if strategy_factory else CostModelRoutedStrategy(engine)
        wl = workload_factory(n_steps, seed) if workload_factory else build_workload(n_steps, seed)
        lats = run_simulation(engine, scheduler, cache_mgr, wl, strategy)
        all_lats.extend(lats)
    return all_lats


# ═══════════════════════════════════════════════════════════════
# 消融结果
# ═══════════════════════════════════════════════════════════════

@dataclass
class AblationResult:
    name: str
    baseline_mean: float = 0.0
    baseline_std: float = 0.0
    ablated_mean: float = 0.0
    ablated_std: float = 0.0
    delta_mean: float = 0.0
    delta_pct: float = 0.0
    cohens_d: float = 0.0
    effect_label: str = ""
    bootstrap_ci: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    wilcoxon_z: float = 0.0
    wilcoxon_p: float = 1.0

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "ablated_mean": self.ablated_mean,
            "ablated_std": self.ablated_std,
            "delta_mean": self.delta_mean,
            "delta_pct": self.delta_pct,
            "cohens_d": self.cohens_d,
            "effect_label": self.effect_label,
            "bootstrap_ci_lower": self.bootstrap_ci[0],
            "bootstrap_ci_mean": self.bootstrap_ci[1],
            "bootstrap_ci_upper": self.bootstrap_ci[2],
            "wilcoxon_z": self.wilcoxon_z,
            "wilcoxon_p": self.wilcoxon_p,
        }


def build_result(name: str, baseline: List[float], ablated: List[float]) -> AblationResult:
    """从两组延迟计算完整消融结果。"""
    wa, wb = WelfordAccumulator(), WelfordAccumulator()
    for v in baseline: wa.update(v)
    for v in ablated: wb.update(v)

    d = cohens_d(baseline, ablated)
    n_paired = min(len(baseline), len(ablated))
    diffs = [baseline[i] - ablated[i] for i in range(n_paired)]
    bci = bootstrap_ci(diffs, n_resamples=1000)
    wz, wp = wilcoxon_approx(diffs)
    delta = wa.mean - wb.mean
    denom = abs(wa.mean) + abs(wb.mean)
    delta_pct = 200 * abs(delta) / denom if denom > 1e-15 else 0.0

    result = AblationResult(
        name=name,
        baseline_mean=wa.mean, baseline_std=wa.std,
        ablated_mean=wb.mean, ablated_std=wb.std,
        delta_mean=delta, delta_pct=delta_pct,
        cohens_d=d, effect_label=effect_size_label(d),
        bootstrap_ci=bci, wilcoxon_z=wz, wilcoxon_p=wp,
    )

    # 调试dump
    _dump_header(f"ABLATION: {name}")
    _dbg("baseline", mean=f"{wa.mean:.2f}", std=f"{wa.std:.2f}", n=wa.n)
    _dbg("ablated", mean=f"{wb.mean:.2f}", std=f"{wb.std:.2f}", n=wb.n)
    _dbg("delta", mean=f"{delta:+.2f}", pct=f"{delta_pct:.2f}%")
    _dbg("Cohen's d", d=f"{d:+.4f}", label=result.effect_label)
    _dbg("Bootstrap CI(95%)", lower=f"{bci[0]:+.2f}", upper=f"{bci[2]:+.2f}")
    _dbg("Wilcoxon", z=f"{wz:+.3f}", p=f"{wp:.4f}")
    _dump_footer(f"{name}: d={d:+.4f} ({result.effect_label}), Δ={delta:+.2f}µs ({delta_pct:.1f}%)")

    return result


# ═══════════════════════════════════════════════════════════════
# 7项消融实验
# ═══════════════════════════════════════════════════════════════

def ablation_1_phase_shift(n_steps, seeds):
    """消融#1: phase-shift负载 vs 固定均匀Zipf。"""
    baseline = collect_latencies(n_steps, seeds,
        workload_factory=lambda n, s: build_workload(n, s, use_phase_shift=True))
    ablated = collect_latencies(n_steps, seeds,
        workload_factory=lambda n, s: build_workload(n, s, use_phase_shift=False))
    return build_result("phase_shift_vs_fixed_zipf", baseline, ablated)


def ablation_2_numa(n_steps, seeds):
    """消融#2: NUMA全连通 vs cpu1断连GPU（通过不同data_location模拟）。"""
    # baseline: 正常拓扑
    baseline = collect_latencies(n_steps, seeds)
    # ablated: 强制所有数据从cpu1出发(跨NUMA)
    all_lats = []
    for seed in seeds:
        topo = create_default_topology()
        engine = CostModelEngine(topo)
        scheduler = QueryPipelineScheduler(engine)
        cache_mgr = TopologyCacheManager(topo, cache_fraction=0.5)
        wl = build_workload(n_steps, seed)
        lats = []
        for q in wl:
            sched = scheduler.schedule(q, data_location="cpu1")  # 强制跨NUMA
            lats.append(sched.latency_us)
        all_lats.extend(lats)
    return build_result("numa_local_vs_cross", baseline, all_lats)


def ablation_3_welford(n_steps, seeds):
    """消融#3: Welford在线方差 vs naive two-pass。"""
    lats = collect_latencies(n_steps, seeds)
    # Welford
    wf = WelfordAccumulator()
    for v in lats: wf.update(v)
    welford_vals = [wf.mean, wf.std, wf.variance]
    # naive
    nm, ns = naive_variance(lats)
    naive_vals = [nm, ns, ns * ns]
    # 差异
    diffs = [abs(a - b) for a, b in zip(welford_vals, naive_vals)]
    _dump_header("ABLATION: welford_vs_naive_variance")
    _dbg("welford", mean=f"{wf.mean:.10f}", std=f"{wf.std:.10f}")
    _dbg("naive", mean=f"{nm:.10f}", std=f"{ns:.10f}")
    _dbg("abs_diff", mean=f"{diffs[0]:.2e}", std=f"{diffs[1]:.2e}")
    _dump_footer(f"Numerical precision delta: mean={diffs[0]:.2e}, std={diffs[1]:.2e}")
    return AblationResult(name="welford_vs_naive_variance",
        baseline_mean=wf.mean, baseline_std=wf.std,
        ablated_mean=nm, ablated_std=ns,
        delta_mean=diffs[0], delta_pct=0.0,
        cohens_d=0.0, effect_label="precision_test")


def ablation_4_kahan(n_steps, seeds):
    """消融#4: Kahan求和 vs naive累加。"""
    lats = collect_latencies(n_steps, seeds)
    kahan_total = kahan_sum(lats)
    naive_total = naive_sum(lats)
    diff = abs(kahan_total - naive_total)
    rel_diff = diff / max(abs(kahan_total), 1e-15)
    _dump_header("ABLATION: kahan_vs_naive_sum")
    _dbg("kahan_total", val=f"{kahan_total:.10f}")
    _dbg("naive_total", val=f"{naive_total:.10f}")
    _dbg("abs_diff", val=f"{diff:.2e}")
    _dbg("rel_diff", val=f"{rel_diff:.2e}")
    _dump_footer(f"Kahan vs naive: abs_diff={diff:.2e}, rel_diff={rel_diff:.2e}")
    return AblationResult(name="kahan_vs_naive_sum",
        baseline_mean=kahan_total, ablated_mean=naive_total,
        delta_mean=diff, delta_pct=rel_diff * 100,
        effect_label="precision_test")


def ablation_5_sigmoid(n_steps, seeds):
    """消融#5: sigmoid soft threshold vs 硬阈值。"""
    # baseline: sigmoid (steepness=2.5 默认)
    baseline = collect_latencies(n_steps, seeds,
        strategy_factory=lambda e: HybridStaticStrategy(e, steepness=2.5))
    # ablated: 近似硬阈值 (steepness=100 → 极陡sigmoid ≈ step function)
    ablated = collect_latencies(n_steps, seeds,
        strategy_factory=lambda e: HybridStaticStrategy(e, steepness=100.0))
    return build_result("sigmoid_soft_vs_hard_threshold", baseline, ablated)


def ablation_6_softmax(n_steps, seeds):
    """消融#6: softmax退火 vs min-cost贪心。"""
    # baseline: 正常退火 (temperature从50降到2)
    baseline = collect_latencies(n_steps, seeds,
        strategy_factory=lambda e: AdaptiveStrategy(e, initial_temperature=50.0, min_temperature=2.0))
    # ablated: 极低温度 ≈ greedy (temperature=0.01 → 几乎总选最低cost)
    ablated = collect_latencies(n_steps, seeds,
        strategy_factory=lambda e: AdaptiveStrategy(e, initial_temperature=0.01, min_temperature=0.01))
    return build_result("softmax_anneal_vs_greedy", baseline, ablated)


def ablation_7_margin(n_steps, seeds):
    """消融#7: margin自适应 vs 固定margin。"""
    # baseline: 自适应margin
    baseline = collect_latencies(n_steps, seeds,
        strategy_factory=lambda e: PAR2QOEnhancedStrategy(e, base_margin=0.18, margin_adapt_rate=0.08))
    # ablated: 固定margin (adapt_rate=0 → margin永不变)
    ablated = collect_latencies(n_steps, seeds,
        strategy_factory=lambda e: PAR2QOEnhancedStrategy(e, base_margin=0.18, margin_adapt_rate=0.0))
    return build_result("adaptive_margin_vs_fixed", baseline, ablated)


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Lynceus Ablation Study (M141-M160)")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()
    seeds = list(range(args.seeds))

    _dump_header("ABLATION EXPERIMENT CONFIG")
    _dbg("steps", val=args.steps)
    _dbg("seeds", val=args.seeds)
    _dbg("ablations", val=7)
    _dump_footer(f"Total runs: 7 ablations × 2 conditions × {args.seeds} seeds = {7*2*args.seeds}")

    ablations = [
        ablation_1_phase_shift,
        ablation_2_numa,
        ablation_3_welford,
        ablation_4_kahan,
        ablation_5_sigmoid,
        ablation_6_softmax,
        ablation_7_margin,
    ]

    results = []
    for i, abl_fn in enumerate(ablations):
        _dbg(f"Running ablation [{i+1}/7]: {abl_fn.__name__}")
        result = abl_fn(args.steps, seeds)
        results.append(result)

    # 汇总输出
    elapsed = time.time() - t0
    output = {
        "metadata": {
            "panel": "Ablation Study — Lynceus M141-M160",
            "n_steps": args.steps,
            "n_seeds": args.seeds,
            "n_ablations": len(results),
            "elapsed_seconds": round(elapsed, 2),
            "algorithms": {
                "effect_size": "Cohen's d",
                "ci": "Bootstrap 1000× resampling, 95%",
                "test": "Wilcoxon signed-rank (z-approx)",
                "variance": "Welford single-pass",
                "summation": "Kahan compensated",
            },
        },
        "ablations": [r.to_dict() for r in results],
    }

    out_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # 最终dump
    _dump_header("ABLATION COMPLETE")
    _dbg("elapsed", seconds=f"{elapsed:.1f}")
    _dbg("output", path=out_path)
    print(file=sys.stderr)
    # 排序表: 按效应量大小
    ranked = sorted(results, key=lambda r: abs(r.cohens_d), reverse=True)
    for r in ranked:
        _dbg(f"  {r.name}",
             d=f"{r.cohens_d:+.4f}",
             label=r.effect_label,
             delta=f"{r.delta_mean:+.2f}µs",
             pct=f"{r.delta_pct:.1f}%",
             p=f"{r.wilcoxon_p:.4f}")
    _dump_footer(f"Largest effect: {ranked[0].name} (d={ranked[0].cohens_d:+.4f})")


if __name__ == "__main__":
    main()
