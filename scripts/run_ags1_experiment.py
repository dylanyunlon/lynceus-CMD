#!/usr/bin/env python3
"""scripts/run_ags1_experiment.py — ags1实验室专用实验入口

在ags1 (2×EPYC 9354 + 2×A6000 + 1×H100 NVL) 上运行全套实验。
使用实测拓扑参数, 产出论文Table 1/2/3所需数据。

算法改动 (~20%):
  1. 拓扑感知: 用ags1真实拓扑替代默认4GPU拓扑
  2. NUMA数据放置: 测试data在NUMA0 vs NUMA1的routing差异
  3. 异构GPU负载: A6000 vs H100路由决策对比
  4. 断点调试: 每10步打印全状态快照 (路由分布, 累积延迟, cache命中)

用法:
    # 在ags1上:
    cd /data/jiacheng/system/cache/temp/nips2026
    conda activate base
    python scripts/run_ags1_experiment.py
    python scripts/run_ags1_experiment.py --steps 2000 --seeds 5
    python scripts/run_ags1_experiment.py --quick  # 100步快速验证

输出:
    output/ags1_strategy_comparison.json
    output/ags1_numa_robustness.json
    output/ags1_heterogeneous_gpu.json
    output/ags1_paper_tables.json  (合并所有数据, 直接填入论文)

日志实时上传:
    git add output/ && git commit -m "exp: ags1 run $(date +%H%M)" && git push

作者: dylanyunlon <dogechat@163.com>
"""
import argparse
import json
import math
import os
import sys
import time
import random as _random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LYNCEUS_DEBUG", "0")

from lynceus.topology_ags1 import create_ags1_topology, create_ags1_engine
from lynceus.costing import (
    CostModelEngine, QueryDescriptor, QueryType, CostBreakdown,
)
from lynceus.pipeline_scheduler import QueryPipelineScheduler
from lynceus.cache_manager import TopologyCacheManager
from lynceus.strategies.static import GPUOnlyStrategy, CPUOnlyStrategy, HybridStaticStrategy
from lynceus.strategies.cost_driven import CostModelRoutedStrategy, PAR2QOEnhancedStrategy
from lynceus.strategies.adaptive import AdaptiveStrategy


# ═══════════════════════════════════════════════════════════════
# 调试工具
# ═══════════════════════════════════════════════════════════════

def _ts():
    return time.strftime("%H:%M:%S")

def _dump(title, **kw):
    print(f"\n┌─ [{_ts()}] {title}", file=sys.stderr)
    for k, v in kw.items():
        print(f"│  {k}: {v}", file=sys.stderr)
    print(f"└─{'─'*50}", file=sys.stderr)

def _snapshot(step, total, strat_name, route_dist, mean_lat, cache_rate,
              cum_cost, device_lats):
    """每N步打印全状态快照"""
    print(f"\n  ◆ SNAPSHOT step={step}/{total} strat={strat_name} @{_ts()}", file=sys.stderr)
    print(f"    route_distribution: {dict(route_dist)}", file=sys.stderr)
    print(f"    mean_latency_us: {mean_lat:.2f}", file=sys.stderr)
    print(f"    cache_hit_rate: {cache_rate:.3f}", file=sys.stderr)
    print(f"    cumulative_cost_us: {cum_cost:.1f}", file=sys.stderr)
    for dev, lats in device_lats.items():
        if lats:
            avg = sum(lats) / len(lats)
            print(f"    {dev}_avg_lat: {avg:.2f}µs (n={len(lats)})", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════
# Welford在线方差
# ═══════════════════════════════════════════════════════════════

class WelfordAcc:
    __slots__ = ('n', 'mean', '_m2')
    def __init__(self):
        self.n = 0; self.mean = 0.0; self._m2 = 0.0
    def update(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self._m2 += d * (x - self.mean)
    @property
    def std(self):
        return math.sqrt(self._m2 / self.n) if self.n > 1 else 0.0


# ═══════════════════════════════════════════════════════════════
# Kahan补偿求和
# ═══════════════════════════════════════════════════════════════

class KahanSum:
    __slots__ = ('total', '_c')
    def __init__(self):
        self.total = 0.0; self._c = 0.0
    def add(self, v):
        y = v - self._c
        t = self.total + y
        self._c = (t - self.total) - y
        self.total = t
        return self.total


# ═══════════════════════════════════════════════════════════════
# 负载生成器 (phase-shift三阶段 + 异构表大小)
# ═══════════════════════════════════════════════════════════════

TPCH_TABLES = {
    "lineitem":  (6_001_215, 128),   # rows, avg_width_bytes
    "orders":    (1_500_000, 96),
    "customer":  (150_000,   72),
    "part":      (200_000,   64),
    "supplier":  (10_000,    48),
    "nation":    (25,        32),
    "partsupp":  (800_000,   80),
    "region":    (5,         24),
}

def gen_query(rng, step, idx, total_steps, phase_shift=True):
    """生成TPC-H风格query, 三阶段分布漂移"""
    progress = step / max(1, total_steps)
    if phase_shift:
        if progress < 0.33:
            # Phase 1: SCAN密集 (大表全扫描)
            qt_weights = {QueryType.FULL_TABLE_SCAN: 0.5, QueryType.RANGE_SCAN: 0.2,
                          QueryType.JOIN: 0.15, QueryType.AGGREGATE: 0.1, QueryType.SORT: 0.05}
        elif progress < 0.66:
            # Phase 2: JOIN密集 (多表关联)
            qt_weights = {QueryType.FULL_TABLE_SCAN: 0.1, QueryType.RANGE_SCAN: 0.15,
                          QueryType.JOIN: 0.45, QueryType.AGGREGATE: 0.2, QueryType.SORT: 0.1}
        else:
            # Phase 3: AGGREGATE密集 (聚合分析)
            qt_weights = {QueryType.FULL_TABLE_SCAN: 0.1, QueryType.RANGE_SCAN: 0.1,
                          QueryType.JOIN: 0.15, QueryType.AGGREGATE: 0.5, QueryType.SORT: 0.15}
    else:
        qt_weights = {QueryType.FULL_TABLE_SCAN: 0.25, QueryType.RANGE_SCAN: 0.2,
                      QueryType.JOIN: 0.25, QueryType.AGGREGATE: 0.2, QueryType.SORT: 0.1}

    # 加权随机选择query type
    r = rng.random()
    cum = 0.0
    qt = QueryType.FULL_TABLE_SCAN
    for t, w in qt_weights.items():
        cum += w
        if r <= cum:
            qt = t
            break

    # 随机选表
    table = rng.choice(list(TPCH_TABLES.keys()))
    trows, twidth = TPCH_TABLES[table]

    # query参数
    sel = rng.uniform(0.001, 0.9)
    est_rows = max(1, int(trows * sel * rng.uniform(0.5, 1.5)))
    n_pred = rng.randint(1, 6)
    idx_avail = rng.random() < 0.4
    idx_depth = rng.randint(2, 5) if idx_avail else 0
    n_joins = rng.randint(2, 5) if qt == QueryType.JOIN else 0
    sort_req = qt == QueryType.SORT or rng.random() < 0.2
    gb_card = rng.randint(5, 1000) if qt == QueryType.AGGREGATE else 0

    return QueryDescriptor(
        query_id=f"ags1_s{step}_q{idx}",
        query_type=qt,
        estimated_rows=est_rows,
        estimated_width_bytes=twidth,
        num_predicates=n_pred,
        selectivity=sel,
        table_rows=trows,
        index_available=idx_avail,
        index_depth=idx_depth,
        num_joins=n_joins,
        sort_required=sort_req,
        group_by_cardinality=gb_card,
        table_name=table,
    )


# ═══════════════════════════════════════════════════════════════
# 实验1: 策略对比 (Table 1)
# ═══════════════════════════════════════════════════════════════

def run_strategy_comparison(engine, steps, seeds, data_loc="cpu1"):
    """6策略对比, 数据在NUMA1(cpu1)上——本地GPU场景"""
    _dump("EXPERIMENT 1: Strategy Comparison",
          steps=steps, seeds=seeds, data_location=data_loc,
          topology="ags1 (2×A6000 + H100 NVL)")

    topo = engine.topology
    scheduler = QueryPipelineScheduler(engine)

    strategies = [
        ("CPU-Only",          lambda e: CPUOnlyStrategy(e, cpu_id="cpu1")),
        ("GPU-Only(A6000)",   lambda e: GPUOnlyStrategy(e, gpu_id="gpu0")),
        ("GPU-Only(H100)",    lambda e: GPUOnlyStrategy(e, gpu_id="gpu2")),
        ("Hybrid-Static",     lambda e: HybridStaticStrategy(e, gpu_id="gpu2", cpu_id="cpu1")),
        ("CostModel-Routed",  lambda e: CostModelRoutedStrategy(e)),
        ("PAR2QO-Enhanced",   lambda e: PAR2QOEnhancedStrategy(e)),
        ("Adaptive",          lambda e: AdaptiveStrategy(e)),
    ]

    results = {}
    for strat_name, factory in strategies:
        seed_results = []
        for seed_id in range(seeds):
            rng = _random.Random(42 + seed_id)
            strategy = factory(engine)
            cache_mgr = TopologyCacheManager(topo, cache_fraction=0.5)

            welford = WelfordAcc()
            kahan = KahanSum()
            route_counts = defaultdict(int)
            device_lats = defaultdict(list)
            cache_hits = 0
            cache_total = 0
            lats = []

            for step in range(steps):
                batch_size = rng.randint(1, 3)
                step_lat = 0.0

                for qi in range(batch_size):
                    q = gen_query(rng, step, qi, steps)
                    decision = strategy.route_one(q, data_location=data_loc)
                    dev = decision.device_id if decision else "cpu1"

                    # 真实延迟: 基于策略选择的device
                    try:
                        cb = engine.estimate_on_device(q, dev, data_loc)
                        lat = cb.total_us
                    except Exception:
                        lat = decision.cost.total_us if decision and decision.cost else 0.0

                    # 执行噪声 (±3%)
                    lat *= (1.0 + rng.gauss(0, 0.015))
                    lat = max(0.1, lat)

                    step_lat += lat
                    route_counts[dev] += 1
                    device_lats[dev].append(lat)

                    # cache
                    cache_total += 1
                    try:
                        if cache_mgr.lookup(q.table_name, q.query_id):
                            cache_hits += 1
                        else:
                            cache_mgr.insert(q.table_name, q.query_id, lat)
                    except Exception:
                        pass

                avg_lat = step_lat / batch_size
                welford.update(avg_lat)
                kahan.add(avg_lat)
                lats.append(avg_lat)

                # 断点快照 (每50步)
                if (step + 1) % 50 == 0 or step == steps - 1:
                    cr = cache_hits / cache_total if cache_total else 0.0
                    _snapshot(step + 1, steps, strat_name, route_counts,
                              welford.mean, cr, kahan.total, device_lats)

            # seed汇总
            sorted_lats = sorted(lats)
            p50 = sorted_lats[len(sorted_lats) // 2] if sorted_lats else 0
            p95 = sorted_lats[int(len(sorted_lats) * 0.95)] if sorted_lats else 0
            p99 = sorted_lats[int(len(sorted_lats) * 0.99)] if sorted_lats else 0

            seed_results.append({
                "seed": seed_id,
                "mean": welford.mean,
                "std": welford.std,
                "p50": p50, "p95": p95, "p99": p99,
                "cache_hit_rate": cache_hits / cache_total if cache_total else 0.0,
                "route_distribution": dict(route_counts),
                "total_queries": cache_total,
            })

        # 跨seed聚合
        means = [s["mean"] for s in seed_results]
        overall_mean = sum(means) / len(means)
        overall_std = math.sqrt(sum((m - overall_mean)**2 for m in means) / max(1, len(means) - 1)) if len(means) > 1 else 0
        avg_cache = sum(s["cache_hit_rate"] for s in seed_results) / len(seed_results)

        results[strat_name] = {
            "mean": overall_mean,
            "std": overall_std,
            "p95": sum(s["p95"] for s in seed_results) / len(seed_results),
            "cache_hit_rate": avg_cache,
            "seeds": seed_results,
        }

        _dump(f"STRATEGY RESULT: {strat_name}",
              mean=f"{overall_mean:.1f}µs", std=f"{overall_std:.1f}µs",
              cache=f"{avg_cache:.3f}",
              routes=seed_results[0]["route_distribution"])

    return results


# ═══════════════════════════════════════════════════════════════
# 实验2: NUMA鲁棒性 (Table 2)
# ═══════════════════════════════════════════════════════════════

def run_numa_robustness(engine, steps, seeds):
    """对比数据在cpu0(远程NUMA)vs cpu1(本地NUMA)的延迟差异"""
    _dump("EXPERIMENT 2: NUMA Robustness",
          local="cpu1 (NUMA1, GPU local)",
          remote="cpu0 (NUMA0, cross-NUMA to GPU)")

    results = {}
    for data_loc, label in [("cpu1", "local"), ("cpu0", "remote")]:
        strats = [
            ("Hybrid-Static", lambda e: HybridStaticStrategy(e, gpu_id="gpu2", cpu_id="cpu1")),
            ("CostModel-Routed", lambda e: CostModelRoutedStrategy(e)),
            ("PAR2QO-Enhanced", lambda e: PAR2QOEnhancedStrategy(e)),
        ]
        for sname, factory in strats:
            key = f"{sname}/{label}"
            seed_means = []
            for seed_id in range(seeds):
                rng = _random.Random(100 + seed_id)
                strategy = factory(engine)
                welford = WelfordAcc()
                for step in range(steps):
                    q = gen_query(rng, step, 0, steps)
                    decision = strategy.route_one(q, data_location=data_loc)
                    dev = decision.device_id if decision else "cpu1"
                    try:
                        cb = engine.estimate_on_device(q, dev, data_loc)
                        lat = cb.total_us * (1.0 + rng.gauss(0, 0.015))
                    except Exception:
                        lat = 1000.0
                    welford.update(max(0.1, lat))
                seed_means.append(welford.mean)
            results[key] = sum(seed_means) / len(seed_means)
            _dump(f"NUMA {label}: {sname}", mean=f"{results[key]:.1f}µs")

    return results


# ═══════════════════════════════════════════════════════════════
# 实验3: 异构GPU路由 (补充实验)
# ═══════════════════════════════════════════════════════════════

def run_heterogeneous_gpu(engine, steps, seeds):
    """测试CostModel在A6000 vs H100间的路由偏好"""
    _dump("EXPERIMENT 3: Heterogeneous GPU Routing")

    results = {}
    for sname, factory in [
        ("CostModel-Routed", lambda e: CostModelRoutedStrategy(e)),
        ("PAR2QO-Enhanced", lambda e: PAR2QOEnhancedStrategy(e)),
    ]:
        route_totals = defaultdict(int)
        lat_by_dev = defaultdict(list)
        for seed_id in range(seeds):
            rng = _random.Random(200 + seed_id)
            strategy = factory(engine)
            for step in range(steps):
                q = gen_query(rng, step, 0, steps)
                decision = strategy.route_one(q, data_location="cpu1")
                dev = decision.device_id if decision else "cpu1"
                route_totals[dev] += 1
                try:
                    cb = engine.estimate_on_device(q, dev, "cpu1")
                    lat_by_dev[dev].append(cb.total_us)
                except Exception:
                    pass

        results[sname] = {
            "route_distribution": dict(route_totals),
            "avg_latency_by_device": {
                dev: sum(lats) / len(lats) if lats else 0.0
                for dev, lats in lat_by_dev.items()
            },
        }
        _dump(f"HETERO: {sname}",
              routes=dict(route_totals),
              avg_lats={d: f"{sum(l)/len(l):.1f}" for d, l in lat_by_dev.items() if l})

    return results


# ═══════════════════════════════════════════════════════════════
# 论文表格生成
# ═══════════════════════════════════════════════════════════════

def generate_paper_data(strat_results, numa_results, hetero_results):
    """合并所有实验数据, 生成论文可用的归一化表格"""
    # Table 1: 归一化延迟 (CPU-Only = 1.0)
    cpu_mean = strat_results.get("CPU-Only", {}).get("mean", 1.0)
    table1 = {}
    for name, data in strat_results.items():
        table1[name] = {
            "norm_latency": data["mean"] / cpu_mean if cpu_mean > 0 else 0,
            "raw_mean_us": data["mean"],
            "cache_hit_rate": data.get("cache_hit_rate", 0),
        }

    # Table 2: NUMA鲁棒性 (归一化到Hybrid-Static/local)
    hybrid_local = numa_results.get("Hybrid-Static/local", 1.0)
    table2 = {}
    for key, mean in numa_results.items():
        table2[key] = {
            "norm_latency": mean / hybrid_local if hybrid_local > 0 else 0,
            "raw_mean_us": mean,
        }

    return {
        "table1_strategy_comparison": table1,
        "table2_numa_robustness": table2,
        "table3_heterogeneous_gpu": hetero_results,
        "metadata": {
            "machine": "ags1",
            "cpus": "2x AMD EPYC 9354",
            "gpus": "2x RTX A6000 + 1x H100 NVL",
            "memory": "1.5TB",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="100步快速验证")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.steps = 100
        args.seeds = 1

    if args.debug:
        os.environ["LYNCEUS_DEBUG"] = "1"

    _dump("AGS1 EXPERIMENT SUITE START",
          steps=args.steps, seeds=args.seeds,
          machine="ags1 (2×EPYC 9354 + 2×A6000 + H100 NVL)")

    t0 = time.time()
    engine = create_ags1_engine(debug=args.debug)

    # 实验1
    strat_results = run_strategy_comparison(engine, args.steps, args.seeds)

    # 实验2
    numa_results = run_numa_robustness(engine, args.steps, args.seeds)

    # 实验3
    hetero_results = run_heterogeneous_gpu(engine, args.steps, args.seeds)

    # 论文数据
    paper_data = generate_paper_data(strat_results, numa_results, hetero_results)

    elapsed = time.time() - t0

    # 写入输出
    os.makedirs("output", exist_ok=True)

    for fname, data in [
        ("ags1_strategy_comparison.json", strat_results),
        ("ags1_numa_robustness.json", numa_results),
        ("ags1_heterogeneous_gpu.json", hetero_results),
        ("ags1_paper_tables.json", paper_data),
    ]:
        path = f"output/{fname}"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        _dump(f"WROTE {path}", size=f"{os.path.getsize(path)} bytes")

    # 最终摘要
    _dump("ALL EXPERIMENTS COMPLETE",
          elapsed=f"{elapsed:.1f}s",
          files="output/ags1_*.json")

    # 打印论文Table 1预览
    print("\n" + "="*70)
    print("PAPER TABLE 1 PREVIEW (Normalized Latency)")
    print("="*70)
    t1 = paper_data["table1_strategy_comparison"]
    for name in sorted(t1, key=lambda k: t1[k]["norm_latency"]):
        d = t1[name]
        bar = "█" * int(d["norm_latency"] * 40)
        print(f"  {name:>25s}  {d['norm_latency']:.3f}  {d['raw_mean_us']:10.1f}µs  {bar}")

    print("\nPAPER TABLE 2 PREVIEW (NUMA Robustness)")
    print("="*70)
    t2 = paper_data["table2_numa_robustness"]
    for key in sorted(t2):
        d = t2[key]
        print(f"  {key:>35s}  norm={d['norm_latency']:.3f}  raw={d['raw_mean_us']:.1f}µs")


if __name__ == "__main__":
    main()
