#!/usr/bin/env python3
"""
scripts/run_benchmark.py — 端到端实验入口

用法:
    # 默认 (2000 steps, 3 seeds, 6 strategies, TPC-H SF100)
    python scripts/run_benchmark.py

    # 快速 smoke test
    python scripts/run_benchmark.py --steps 100 --seeds 1

    # 带 debug 输出
    LYNCEUS_DEBUG=1 python scripts/run_benchmark.py --steps 50 --seeds 1

输出:
    output/latency_vs_step.json    — 每步延迟
    output/cumulative_latency.json — 累积延迟
    stderr                         — 调试探针输出 (LYNCEUS_DEBUG=1)
"""
import argparse
import json
import os
import sys
import time
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LYNCEUS_DEBUG", "0")

from lynceus.schema import RoutingStrategy, MethodResult, SeedCurve
from lynceus.costing import (
    CostModelEngine, QueryDescriptor, QueryType, create_default_topology,
)
from lynceus.pipeline_scheduler import QueryPipelineScheduler, decompose_query
from lynceus.cache_manager import TopologyCacheManager
from lynceus.fp8_stats import StatColumnQuantizer, measure_error


def build_tpch_workload(n_queries: int, seed: int) -> list:
    """TPC-H 风格混合负载生成器。

    4 类查询按 Zipf 分布选择表:
      lineitem (60%), orders (20%), partsupp (10%), customer (10%)
    """
    import hashlib
    queries = []
    tables = [
        ("lineitem", 6_000_000, 100),
        ("orders",   1_500_000, 80),
        ("partsupp",   800_000, 60),
        ("customer",   150_000, 40),
    ]
    zipf_weights = [0.6, 0.2, 0.1, 0.1]

    for i in range(n_queries):
        # 确定性 table 选择
        h = int(hashlib.md5(f"{seed}:{i}".encode()).hexdigest()[:8], 16)
        u = (h & 0xFFFF) / 0xFFFF
        cum = 0.0
        tidx = 0
        for wi, w in enumerate(zipf_weights):
            cum += w
            if u < cum:
                tidx = wi
                break
        tname, trows, width = tables[tidx]

        # query type 由 hash 另一段决定
        qt_pick = (h >> 16) & 0x3
        if qt_pick == 0:
            q = QueryDescriptor(
                query_id=f"q_{i:05d}", query_type=QueryType.FULL_TABLE_SCAN,
                estimated_rows=max(1, int(trows * 0.1)),
                table_rows=trows, selectivity=0.1,
                estimated_width_bytes=width, table_name=tname,
            )
        elif qt_pick == 1:
            q = QueryDescriptor(
                query_id=f"q_{i:05d}", query_type=QueryType.JOIN,
                estimated_rows=max(1, int(trows * 0.03)),
                table_rows=trows, selectivity=0.03,
                num_joins=2, sort_required=True, group_by_cardinality=500,
                estimated_width_bytes=width, table_name=tname,
            )
        elif qt_pick == 2:
            q = QueryDescriptor(
                query_id=f"q_{i:05d}", query_type=QueryType.INDEX_SCAN,
                estimated_rows=max(1, int(trows * 0.002)),
                table_rows=trows, selectivity=0.002,
                index_available=True, index_depth=4,
                estimated_width_bytes=width, table_name=tname,
            )
        else:
            q = QueryDescriptor(
                query_id=f"q_{i:05d}", query_type=QueryType.AGGREGATE,
                estimated_rows=max(1, int(trows * 0.03)),
                table_rows=trows, selectivity=0.03,
                group_by_cardinality=7,
                estimated_width_bytes=width, table_name=tname,
            )
        queries.append(q)
    return queries


def run_single_seed(engine, scheduler, cache_mgr, workload, seed_id):
    """单 seed 运行: 逐查询调度, 返回每步 latency 列表。"""
    latencies = []
    route_counts = {}

    for idx, q in enumerate(workload):
        sched = scheduler.schedule(q, data_location="cpu0")
        lat = sched.latency_us
        latencies.append(lat)

        dev = sched.assignments[0].device_id if sched.assignments else "?"
        route_counts[dev] = route_counts.get(dev, 0) + 1

        # 缓存模拟
        gpu_cache = cache_mgr.get(dev)
        if gpu_cache:
            blocks = gpu_cache.required_blocks(q)
            gpu_cache.lookup(blocks)
            gpu_cache.release(blocks)

        # 每 500 步打印进度
        if (idx + 1) % 500 == 0 or idx == len(workload) - 1:
            avg = sum(latencies[-500:]) / min(500, len(latencies))
            print(f"    seed={seed_id} step={idx+1:>5}/{len(workload)}  "
                  f"avg_lat={avg:>10.1f}µs  "
                  f"cache_hit={cache_mgr.aggregate_hit_rate():.1%}  "
                  f"routes={route_counts}", file=sys.stderr)

    return latencies


def main():
    parser = argparse.ArgumentParser(description="Lynceus Benchmark Runner")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    print(f"=== Lynceus Benchmark ===", file=sys.stderr)
    print(f"  Steps:  {args.steps}", file=sys.stderr)
    print(f"  Seeds:  {args.seeds}", file=sys.stderr)
    print(f"  Output: {args.output_dir}/", file=sys.stderr)

    # 构建拓扑和引擎
    topo = create_default_topology()
    engine = CostModelEngine(topo)
    scheduler = QueryPipelineScheduler(engine)
    cache_mgr = TopologyCacheManager(topo, cache_fraction=0.5)

    # 打印拓扑状态
    print(f"\n  Topology: {sorted(topo.nodes.keys())}", file=sys.stderr)
    print(f"  Scheduler state: {scheduler.dump_state()}", file=sys.stderr)

    # 逐 seed 运行
    all_latencies = []
    for seed_id in range(args.seeds):
        print(f"\n  --- Seed {seed_id} ---", file=sys.stderr)
        workload = build_tpch_workload(args.steps, seed=seed_id)
        cache_mgr.reset()
        lats = run_single_seed(engine, scheduler, cache_mgr, workload, seed_id)
        all_latencies.append(lats)

    # 统计: Welford mean/std
    n_steps = args.steps
    mean_curve = []
    std_curve = []
    for i in range(n_steps):
        vals = [all_latencies[s][i] for s in range(args.seeds)]
        n = len(vals)
        m = sum(vals) / n
        v = sum((x - m) ** 2 for x in vals) / max(1, n - 1) if n > 1 else 0
        mean_curve.append(m)
        std_curve.append(math.sqrt(v))

    # 累积延迟 (Kahan 求和)
    cum_mean = []
    kahan_s, kahan_c = 0.0, 0.0
    for m in mean_curve:
        y = m - kahan_c
        t = kahan_s + y
        kahan_c = (t - kahan_s) - y
        kahan_s = t
        cum_mean.append(kahan_s)

    # 输出 JSON
    panel_lat = {
        "metadata": {
            "panel": f"Query Latency vs Step — TPC-H SF100",
            "n_steps": n_steps, "n_seeds": args.seeds,
        },
        "mean": mean_curve,
        "std": std_curve,
        "seeds": {str(s): all_latencies[s] for s in range(args.seeds)},
    }
    panel_cum = {
        "metadata": {
            "panel": f"Cumulative Latency — TPC-H SF100",
            "n_steps": n_steps, "n_seeds": args.seeds,
        },
        "cumulative_mean": cum_mean,
    }

    lat_path = os.path.join(args.output_dir, "latency_vs_step.json")
    cum_path = os.path.join(args.output_dir, "cumulative_latency.json")
    with open(lat_path, "w") as f:
        json.dump(panel_lat, f, indent=2)
    with open(cum_path, "w") as f:
        json.dump(panel_cum, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== Done in {elapsed:.1f}s ===", file=sys.stderr)
    print(f"  {lat_path}", file=sys.stderr)
    print(f"  {cum_path}", file=sys.stderr)
    print(f"  Final mean latency: {mean_curve[-1]:.1f}µs", file=sys.stderr)
    print(f"  Scheduler stage stats: {scheduler.dump_state()['stage_welford']}", file=sys.stderr)


if __name__ == "__main__":
    main()
