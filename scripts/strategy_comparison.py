#!/usr/bin/env python3
"""
scripts/strategy_comparison.py — 6策略×多seed×多步 对比实验

第二位Claude (M121–M140) 交付。
作者: dylanyunlon <dogechat@163.com>

算法改写 (~20%, 非字符串/docstring替换):
  1. 负载生成: phase-shift 三阶段查询分布漂移
     (前1/3 SCAN密集 → 中间1/3 JOIN密集 → 后1/3 AGGREGATE密集)
  2. 统计汇聚: Welford 单pass在线方差 (替代 two-pass)
  3. 累积延迟: Kahan 补偿求和 (替代 naive +=)
  4. 路由多样性: Shannon 熵 H = -Σ p·ln(p) 度量路由集中度
  5. 误差度量: SMAPE 对称百分比误差 2|a-b|/(|a|+|b|)
  6. 统计快照: P50/P95/P99 百分位 (排序选取, 非近似)
  7. 收敛检测: 滑窗尾部CV (变异系数 <5% → 认为收敛)

断点调试:
  所有调试输出到 stderr，stdout 保持干净。
  格式: ┌─ STATE DUMP │ 过程日志 └─ 结果总结

用法:
    python scripts/strategy_comparison.py                    # 默认 2000步×3种子
    python scripts/strategy_comparison.py --steps 100 --seeds 1  # 快速smoke
    LYNCEUS_DEBUG=1 python scripts/strategy_comparison.py --steps 50 --seeds 1
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LYNCEUS_DEBUG", "0")

from lynceus.costing import (
    CostModelEngine, QueryDescriptor, QueryType, create_default_topology,
)
from lynceus.pipeline_scheduler import QueryPipelineScheduler
from lynceus.cache_manager import TopologyCacheManager
from lynceus.strategies.static import (
    GPUOnlyStrategy, CPUOnlyStrategy, HybridStaticStrategy,
)
from lynceus.strategies.cost_driven import (
    CostModelRoutedStrategy, PAR2QOEnhancedStrategy,
)
from lynceus.strategies.adaptive import AdaptiveStrategy


# ═══════════════════════════════════════════════════════════════
# 调试基础设施
# ═══════════════════════════════════════════════════════════════

def _dbg(msg: str, **kw) -> None:
    """条件调试打印, 全部到 stderr."""
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
# 算法改写 #1: Phase-Shift 负载生成
# ═══════════════════════════════════════════════════════════════

def build_phaseshift_workload(n_queries: int, seed: int) -> List[QueryDescriptor]:
    """TPC-H负载生成器 + phase-shift三阶段分布漂移。

    原版: 全程使用固定Zipf权重选择query_type
    改写: 把workload分三个phase:
      phase-1 (0~33%):   SCAN密集   (60% scan, 20% join, 10% agg, 10% index)
      phase-2 (33%~66%): JOIN密集   (20% scan, 50% join, 20% agg, 10% index)
      phase-3 (66%~100%):AGGREGATE密集 (15% scan, 15% join, 55% agg, 15% index)
    这模拟真实OLAP负载的阶段性变化: ETL扫描→分析连接→聚合报表
    """
    queries = []
    tables = [
        ("lineitem", 6_000_000, 100),
        ("orders",   1_500_000, 80),
        ("partsupp",   800_000, 60),
        ("customer",   150_000, 40),
    ]
    zipf_weights = [0.6, 0.2, 0.1, 0.1]

    # phase-shift的query_type权重矩阵
    # 行: [SCAN, JOIN, INDEX_SCAN, AGGREGATE]
    phase_weights = [
        [0.60, 0.20, 0.10, 0.10],  # phase-1: scan密集
        [0.20, 0.50, 0.10, 0.20],  # phase-2: join密集
        [0.15, 0.15, 0.15, 0.55],  # phase-3: aggregate密集
    ]

    one_third = n_queries // 3
    two_third = 2 * n_queries // 3

    for i in range(n_queries):
        h = int(hashlib.md5(f"{seed}:{i}".encode()).hexdigest()[:8], 16)

        # table 选择 (保留原版Zipf)
        u_table = (h & 0xFFFF) / 0xFFFF
        cum = 0.0
        tidx = 0
        for wi, w in enumerate(zipf_weights):
            cum += w
            if u_table < cum:
                tidx = wi
                break
        tname, trows, width = tables[tidx]

        # phase-shift: 根据step位置选择query_type概率
        if i < one_third:
            qw = phase_weights[0]
        elif i < two_third:
            qw = phase_weights[1]
        else:
            qw = phase_weights[2]

        u_type = ((h >> 16) & 0xFFFF) / 0xFFFF
        cum_qt = 0.0
        qt_idx = 0
        for qi, pw in enumerate(qw):
            cum_qt += pw
            if u_type < cum_qt:
                qt_idx = qi
                break

        # 构造QueryDescriptor
        if qt_idx == 0:  # SCAN
            q = QueryDescriptor(
                query_id=f"q_{i:05d}", query_type=QueryType.FULL_TABLE_SCAN,
                estimated_rows=max(1, int(trows * 0.1)),
                table_rows=trows, selectivity=0.1,
                estimated_width_bytes=width, table_name=tname,
            )
        elif qt_idx == 1:  # JOIN
            q = QueryDescriptor(
                query_id=f"q_{i:05d}", query_type=QueryType.JOIN,
                estimated_rows=max(1, int(trows * 0.03)),
                table_rows=trows, selectivity=0.03,
                num_joins=2, sort_required=True, group_by_cardinality=500,
                estimated_width_bytes=width, table_name=tname,
            )
        elif qt_idx == 2:  # INDEX_SCAN
            q = QueryDescriptor(
                query_id=f"q_{i:05d}", query_type=QueryType.INDEX_SCAN,
                estimated_rows=max(1, int(trows * 0.002)),
                table_rows=trows, selectivity=0.002,
                index_available=True, index_depth=4,
                estimated_width_bytes=width, table_name=tname,
            )
        else:  # AGGREGATE
            q = QueryDescriptor(
                query_id=f"q_{i:05d}", query_type=QueryType.AGGREGATE,
                estimated_rows=max(1, int(trows * 0.03)),
                table_rows=trows, selectivity=0.03,
                group_by_cardinality=7,
                estimated_width_bytes=width, table_name=tname,
            )
        queries.append(q)

    return queries


# ═══════════════════════════════════════════════════════════════
# 算法改写 #2: Welford 在线方差
# ═══════════════════════════════════════════════════════════════

class WelfordAccumulator:
    """Welford单pass在线方差算法。

    原版: two-pass — 先算mean, 再扫一遍算variance
    改写: 单pass在线, 每个新样本O(1)更新, 数值更稳定
    参考: Welford(1962), Knuth TAOCP Vol.2 §4.2.2
    """
    __slots__ = ('n', 'mean', '_m2')

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._m2 += delta * delta2

    @property
    def variance(self) -> float:
        return self._m2 / max(1, self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


# ═══════════════════════════════════════════════════════════════
# 算法改写 #3: Kahan 补偿求和
# ═══════════════════════════════════════════════════════════════

class KahanSum:
    """Kahan补偿求和 — 减少浮点累加误差。

    原版: naive sum += val
    改写: 维护补偿项c, 每次加法修正舍入误差
    在2000+步的累积延迟计算中,差异可达 ~1e-10 级
    """
    __slots__ = ('_sum', '_c')

    def __init__(self):
        self._sum = 0.0
        self._c = 0.0

    def add(self, val: float) -> float:
        y = val - self._c
        t = self._sum + y
        self._c = (t - self._sum) - y
        self._sum = t
        return self._sum

    @property
    def total(self) -> float:
        return self._sum


# ═══════════════════════════════════════════════════════════════
# 算法改写 #4: Shannon 路由熵
# ═══════════════════════════════════════════════════════════════

def shannon_entropy(route_counts: Dict[str, int]) -> float:
    """计算路由选择的Shannon熵 H = -Σ p·ln(p)。

    衡量路由多样性:
      H=0     → 所有查询路由到同一设备 (GPU-Only/CPU-Only预期)
      H=ln(n) → 均匀分布到n个设备 (最大熵,完全多样)
    Adaptive和PAR2QO策略应有中等熵值。
    """
    total = sum(route_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in route_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)
    return entropy


# ═══════════════════════════════════════════════════════════════
# 算法改写 #5: SMAPE 对称误差
# ═══════════════════════════════════════════════════════════════

def smape(predicted: float, actual: float) -> float:
    """对称平均绝对百分比误差 SMAPE = 2|a-b|/(|a|+|b|)。

    原版: relative_error = |pred - actual| / actual (actual=0时除零)
    改写: SMAPE ∈ [0, 2], 对称处理, 当两者都为0时返回0
    """
    denom = abs(predicted) + abs(actual)
    if denom < 1e-15:
        return 0.0
    return 2.0 * abs(predicted - actual) / denom


# ═══════════════════════════════════════════════════════════════
# 算法改写 #6: 百分位计算
# ═══════════════════════════════════════════════════════════════

def percentile(sorted_vals: List[float], p: float) -> float:
    """精确百分位 (排序选取, 非近似)。p ∈ [0,100]."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[int(f)] * (c - k) + sorted_vals[int(c)] * (k - f)


# ═══════════════════════════════════════════════════════════════
# 算法改写 #7: 滑窗尾部收敛检测
# ═══════════════════════════════════════════════════════════════

def check_convergence(latencies: List[float], window: int = 200,
                      cv_threshold: float = 0.05) -> Tuple[bool, float]:
    """检查尾部window内延迟是否收敛 (CV < threshold)。

    CV = std / mean, 变异系数
    CV < 5% 视为收敛 (查询延迟已稳定)
    """
    if len(latencies) < window:
        return False, float('inf')
    tail = latencies[-window:]
    acc = WelfordAccumulator()
    for v in tail:
        acc.update(v)
    if acc.mean < 1e-9:
        return True, 0.0
    cv = acc.std / acc.mean
    return cv < cv_threshold, cv


# ═══════════════════════════════════════════════════════════════
# 核心: 单策略运行
# ═══════════════════════════════════════════════════════════════

def run_strategy_seed(
    strategy, scheduler, cache_mgr, workload, seed_id: int,
    strategy_name: str, engine=None,
) -> Dict:
    """单策略单seed运行。返回包含延迟曲线和诊断信息的字典。"""
    import random as _rng_mod
    rng = _rng_mod.Random(42 + seed_id)
    latencies = []
    route_counts = defaultdict(int)
    cumulative = KahanSum()
    cum_curve = []
    welford = WelfordAccumulator()

    n = len(workload)
    for idx, q in enumerate(workload):
        # 路由: 由策略决定device
        decision = strategy.route_one(q, data_location="cpu0")
        dev = decision.device_id

        # 用策略选定的device计算真实延迟 (含transfer)
        # 而非scheduler重新路由覆盖策略决策
        if engine is not None:
            try:
                dev_cb = engine.estimate_on_device(q, dev, "cpu0")
                lat = dev_cb.total_us
            except Exception:
                lat = decision.cost.total_us if decision.cost else 0.0
        else:
            lat = decision.cost.total_us if decision.cost else 0.0
        # 执行噪声 (±2%)
        lat *= (1.0 + rng.gauss(0, 0.01))

        latencies.append(lat)
        route_counts[dev] += 1
        welford.update(lat)
        cum_curve.append(cumulative.add(lat))

        # 缓存模拟
        gpu_cache = cache_mgr.get(dev)
        if gpu_cache:
            blocks = gpu_cache.required_blocks(q)
            gpu_cache.lookup(blocks)
            gpu_cache.release(blocks)

        # Adaptive策略: 反馈observe
        if hasattr(strategy, 'observe_with_estimate') and decision.cost:
            strategy.observe_with_estimate(
                dev, decision.cost.total_us, lat)

        # 每500步: 调试快照
        if (idx + 1) % 500 == 0:
            sorted_so_far = sorted(latencies)
            p50 = percentile(sorted_so_far, 50)
            p95 = percentile(sorted_so_far, 95)
            hit_rate = cache_mgr.aggregate_hit_rate()
            ent = shannon_entropy(dict(route_counts))
            converged, cv = check_convergence(latencies)
            _dbg(f"[{strategy_name}] seed={seed_id} step={idx+1}/{n}",
                 avg=f"{welford.mean:.1f}µs",
                 P50=f"{p50:.1f}", P95=f"{p95:.1f}",
                 cache_hit=f"{hit_rate:.1%}",
                 entropy=f"{ent:.3f}",
                 cv=f"{cv:.4f}",
                 converged=converged,
                 routes=dict(route_counts))

    # 最终统计
    sorted_all = sorted(latencies)
    result = {
        "latencies": latencies,
        "cumulative": cum_curve,
        "mean": welford.mean,
        "std": welford.std,
        "total_cost": cumulative.total,
        "p50": percentile(sorted_all, 50),
        "p95": percentile(sorted_all, 95),
        "p99": percentile(sorted_all, 99),
        "route_distribution": dict(route_counts),
        "cache_hit_rate": cache_mgr.aggregate_hit_rate(),
        "shannon_entropy": shannon_entropy(dict(route_counts)),
    }

    # 收敛检测
    converged, cv = check_convergence(latencies)
    result["converged"] = converged
    result["tail_cv"] = cv

    return result


# ═══════════════════════════════════════════════════════════════
# 核心: 多策略×多seed对比
# ═══════════════════════════════════════════════════════════════

def run_comparison(n_steps: int, n_seeds: int, output_dir: str) -> Dict:
    """运行6种策略的完整对比实验。"""
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    # 构建共享基础设施
    topo = create_default_topology()
    engine = CostModelEngine(topo)
    scheduler = QueryPipelineScheduler(engine)

    _dump_header("EXPERIMENT CONFIG")
    _dbg("n_steps", val=n_steps)
    _dbg("n_seeds", val=n_seeds)
    _dbg("topology", nodes=sorted(topo.nodes.keys()))
    _dump_footer(f"6 strategies × {n_seeds} seeds × {n_steps} steps = {6*n_seeds*n_steps} total queries")

    # 策略实例化
    strategy_factories = [
        ("GPU-Only",          lambda e: GPUOnlyStrategy(e)),
        ("CPU-Only",          lambda e: CPUOnlyStrategy(e)),
        ("Hybrid-Static",     lambda e: HybridStaticStrategy(e)),
        ("CostModel-Routed",  lambda e: CostModelRoutedStrategy(e)),
        ("PAR2QO-Enhanced",   lambda e: PAR2QOEnhancedStrategy(e)),
        ("Adaptive",          lambda e: AdaptiveStrategy(e)),
    ]

    all_results = {}

    for strat_idx, (strat_name, factory) in enumerate(strategy_factories):
        _dump_header(f"STRATEGY [{strat_idx+1}/6]: {strat_name}")

        seed_data = {}
        cross_seed_welford = WelfordAccumulator()  # 跨seed汇聚
        all_lats_flat = []

        for seed_id in range(n_seeds):
            # 关键: 每个策略×seed 独立reset
            strategy = factory(engine)
            cache_mgr = TopologyCacheManager(topo, cache_fraction=0.5)
            cache_mgr.reset()

            _dbg(f"seed={seed_id} starting", strategy=strat_name)

            workload = build_phaseshift_workload(n_steps, seed=seed_id + strat_idx * 1000)
            result = run_strategy_seed(
                strategy, scheduler, cache_mgr, workload,
                seed_id, strat_name, engine=engine)

            seed_data[str(seed_id)] = result
            all_lats_flat.extend(result["latencies"])

            # Welford跨seed: 用每个seed的mean做一次update
            cross_seed_welford.update(result["mean"])

            _dbg(f"seed={seed_id} done",
                 mean=f"{result['mean']:.1f}µs",
                 total=f"{result['total_cost']:.1f}µs",
                 p95=f"{result['p95']:.1f}µs",
                 entropy=f"{result['shannon_entropy']:.3f}",
                 cache=f"{result['cache_hit_rate']:.1%}")

        # 跨seed聚合
        sorted_flat = sorted(all_lats_flat)
        agg = {
            "grand_mean": cross_seed_welford.mean,
            "grand_std": cross_seed_welford.std,
            "grand_p50": percentile(sorted_flat, 50),
            "grand_p95": percentile(sorted_flat, 95),
            "grand_p99": percentile(sorted_flat, 99),
            "grand_total_cost": sum(
                seed_data[str(s)]["total_cost"] for s in range(n_seeds)),
        }

        # 逐step的mean/std (Welford)
        step_means = []
        step_stds = []
        for i in range(n_steps):
            sw = WelfordAccumulator()
            for s in range(n_seeds):
                sw.update(seed_data[str(s)]["latencies"][i])
            step_means.append(sw.mean)
            step_stds.append(sw.std)

        # 累积延迟曲线 (Kahan)
        cum_kahan = KahanSum()
        cum_means = []
        for m in step_means:
            cum_means.append(cum_kahan.add(m))

        # 汇聚route_distribution
        merged_routes = defaultdict(int)
        for s in range(n_seeds):
            for dev, cnt in seed_data[str(s)]["route_distribution"].items():
                merged_routes[dev] += cnt

        all_results[strat_name] = {
            "seeds": seed_data,
            "aggregate": agg,
            "step_mean": step_means,
            "step_std": step_stds,
            "cumulative_mean": cum_means,
            "route_distribution": dict(merged_routes),
            "shannon_entropy": shannon_entropy(dict(merged_routes)),
        }

        # 策略完成dump
        _dump_footer(
            f"{strat_name}: mean={agg['grand_mean']:.1f}µs "
            f"std={agg['grand_std']:.1f} "
            f"P95={agg['grand_p95']:.1f}µs "
            f"entropy={all_results[strat_name]['shannon_entropy']:.3f} "
            f"routes={dict(merged_routes)}")

    # ═══════════════════════════════════════════════════════════
    # 策略间交叉分析
    # ═══════════════════════════════════════════════════════════

    _dump_header("CROSS-STRATEGY ANALYSIS")

    # SMAPE: 每对策略间的误差距离
    strat_names = list(all_results.keys())
    smape_matrix = {}
    for i, s1 in enumerate(strat_names):
        for j, s2 in enumerate(strat_names):
            if i < j:
                val = smape(
                    all_results[s1]["aggregate"]["grand_mean"],
                    all_results[s2]["aggregate"]["grand_mean"])
                smape_matrix[f"{s1} vs {s2}"] = round(val, 4)
                _dbg(f"SMAPE({s1}, {s2})", val=f"{val:.4f}")

    # 排名
    ranked = sorted(strat_names,
                    key=lambda s: all_results[s]["aggregate"]["grand_mean"])
    for rank, s in enumerate(ranked):
        _dbg(f"Rank #{rank+1}", strategy=s,
             mean=f"{all_results[s]['aggregate']['grand_mean']:.1f}µs")

    _dump_footer(f"Best: {ranked[0]} ({all_results[ranked[0]]['aggregate']['grand_mean']:.1f}µs)")

    # ═══════════════════════════════════════════════════════════
    # 输出JSON
    # ═══════════════════════════════════════════════════════════

    elapsed = time.time() - t0

    output = {
        "metadata": {
            "panel": f"Strategy Comparison — TPC-H SF100 (phase-shift)",
            "n_steps": n_steps,
            "n_seeds": n_seeds,
            "n_strategies": len(strat_names),
            "total_queries": n_steps * n_seeds * len(strat_names),
            "elapsed_seconds": round(elapsed, 2),
            "algorithms": {
                "workload": "phase-shift 3-phase (SCAN→JOIN→AGG)",
                "variance": "Welford single-pass online",
                "cumulative": "Kahan compensated summation",
                "diversity": "Shannon entropy H=-Σp·ln(p)",
                "error": "SMAPE 2|a-b|/(|a|+|b|)",
                "convergence": "tail-window CV < 5%",
            },
        },
        "methods": {},
        "ranking": ranked,
        "smape_matrix": smape_matrix,
    }

    for sname in strat_names:
        r = all_results[sname]
        output["methods"][sname] = {
            "step_mean": r["step_mean"],
            "step_std": r["step_std"],
            "cumulative_mean": r["cumulative_mean"],
            "total_cost": r["aggregate"]["grand_total_cost"],
            "grand_mean": r["aggregate"]["grand_mean"],
            "grand_std": r["aggregate"]["grand_std"],
            "p50": r["aggregate"]["grand_p50"],
            "p95": r["aggregate"]["grand_p95"],
            "p99": r["aggregate"]["grand_p99"],
            "route_distribution": r["route_distribution"],
            "shannon_entropy": r["shannon_entropy"],
            "seeds": {},
        }
        for sid in range(n_seeds):
            sd = r["seeds"][str(sid)]
            output["methods"][sname]["seeds"][str(sid)] = {
                "mean": sd["mean"],
                "std": sd["std"],
                "total_cost": sd["total_cost"],
                "p50": sd["p50"],
                "p95": sd["p95"],
                "p99": sd["p99"],
                "cache_hit_rate": sd["cache_hit_rate"],
                "converged": sd["converged"],
                "tail_cv": sd["tail_cv"],
                "route_distribution": sd["route_distribution"],
            }

    out_path = os.path.join(output_dir, "strategy_comparison.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    _dump_header("EXPERIMENT COMPLETE")
    _dbg("elapsed", seconds=f"{elapsed:.1f}")
    _dbg("output", path=out_path)
    for sname in ranked:
        _dbg(f"  {sname}",
             mean=f"{output['methods'][sname]['grand_mean']:.1f}µs",
             P95=f"{output['methods'][sname]['p95']:.1f}µs",
             entropy=f"{output['methods'][sname]['shannon_entropy']:.3f}")
    _dump_footer(f"Winner: {ranked[0]} | File: {out_path}")

    return output


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Lynceus 6-Strategy Comparison Benchmark (M121-M140)")
    parser.add_argument("--steps", type=int,
                        default=int(os.environ.get("NUM_STEPS", "2000")))
    parser.add_argument("--seeds", type=int,
                        default=int(os.environ.get("NUM_SEEDS", "3")))
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    run_comparison(args.steps, args.seeds, args.output_dir)


if __name__ == "__main__":
    main()
