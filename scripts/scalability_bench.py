#!/usr/bin/env python3
"""
scripts/scalability_bench.py — 扩展性基准: 5表规模 × 5批次

M281–M290 交付。

测试CostModel-Routed策略在不同规模下的扩展性:
  表大小:   [100K, 1M, 10M, 50M, 100M] 行
  查询批次:  [10, 50, 100, 500, 1000] 步

对每个组合记录:
  - total_latency_us:  总延迟 (µs)
  - mean_latency_us:   平均延迟 (µs)
  - throughput_qps:     吞吐量 (queries/sec)

算法亮点:
  1. Kahan补偿求和: 累积总延迟 (避免大数吞小数)
  2. Welford在线方差: 单pass计算mean/std
  3. 确定性哈希负载: 固定seed保证可重复
  4. TPC-H四表混合: SCAN/JOIN/INDEX_SCAN/AGGREGATE

输出: output/scalability_bench.json

用法:
    python scripts/scalability_bench.py                  # 完整 5×5 矩阵
    LYNCEUS_DEBUG=1 python scripts/scalability_bench.py  # 调试输出
"""
import hashlib
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
from lynceus.strategies.cost_driven import CostModelRoutedStrategy


# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

TABLE_SIZES  = [100_000, 1_000_000, 10_000_000, 50_000_000, 100_000_000]
BATCH_SIZES  = [10, 50, 100, 500, 1000]

# TPC-H四表模板 — table_rows由TABLE_SIZES参数化, 宽度固定
TABLE_TEMPLATES = [
    ("lineitem",  100),
    ("orders",     80),
    ("partsupp",   60),
    ("customer",   40),
]

# 表选择的Zipf权重 (lineitem最热)
ZIPF_WEIGHTS = [0.6, 0.2, 0.1, 0.1]

# query_type概率 (混合负载)
QUERY_TYPE_WEIGHTS = [0.30, 0.30, 0.15, 0.25]  # SCAN, JOIN, INDEX_SCAN, AGGREGATE

SEED = 42


# ═══════════════════════════════════════════════════════════════
# 调试基础设施
# ═══════════════════════════════════════════════════════════════

_DEBUG = os.environ.get("LYNCEUS_DEBUG", "0") != "0"


def _dbg(msg: str, **kw) -> None:
    if _DEBUG:
        suffix = ": " + ", ".join(f"{k}={v}" for k, v in kw.items()) if kw else ""
        print(f"  │ {msg}{suffix}", file=sys.stderr)


def _dump_header(title: str) -> None:
    if _DEBUG:
        print(f"\n┌─{'─' * 60}", file=sys.stderr)
        print(f"│ {title}", file=sys.stderr)
        print(f"├─{'─' * 60}", file=sys.stderr)


def _dump_footer(summary: str) -> None:
    if _DEBUG:
        print(f"├─{'─' * 60}", file=sys.stderr)
        print(f"└─ {summary}", file=sys.stderr)
        print(file=sys.stderr)


# ═══════════════════════════════════════════════════════════════
# Kahan 补偿求和
# ═══════════════════════════════════════════════════════════════

class KahanSum:
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
# Welford 在线方差
# ═══════════════════════════════════════════════════════════════

class WelfordAccumulator:
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
# 负载生成
# ═══════════════════════════════════════════════════════════════

def build_workload(n_queries: int, table_rows: int, seed: int) -> List[QueryDescriptor]:
    """生成TPC-H混合负载, table_rows参数化表大小。

    所有四张表的row count按table_rows等比缩放:
      lineitem: table_rows (主表)
      orders:   table_rows × 0.25
      partsupp: table_rows × 0.133
      customer: table_rows × 0.025
    """
    scale_ratios = [1.0, 0.25, 0.133, 0.025]
    queries = []

    for i in range(n_queries):
        h = int(hashlib.md5(f"{seed}:{i}".encode()).hexdigest()[:8], 16)

        # 表选择 (Zipf)
        u_table = (h & 0xFFFF) / 0xFFFF
        cum = 0.0
        tidx = 0
        for wi, w in enumerate(ZIPF_WEIGHTS):
            cum += w
            if u_table < cum:
                tidx = wi
                break
        tname, width = TABLE_TEMPLATES[tidx]
        trows = max(1, int(table_rows * scale_ratios[tidx]))

        # query_type选择
        u_type = ((h >> 16) & 0xFFFF) / 0xFFFF
        cum_qt = 0.0
        qt_idx = 0
        for qi, pw in enumerate(QUERY_TYPE_WEIGHTS):
            cum_qt += pw
            if u_type < cum_qt:
                qt_idx = qi
                break

        # 构造QueryDescriptor
        if qt_idx == 0:   # FULL_TABLE_SCAN
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
        else:              # AGGREGATE
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
# 单组合运行
# ═══════════════════════════════════════════════════════════════

def run_one_combo(
    engine: CostModelEngine,
    scheduler: QueryPipelineScheduler,
    topo,
    table_rows: int,
    batch_size: int,
) -> Dict:
    """运行单个 (table_rows, batch_size) 组合, 返回指标字典。"""

    strategy = CostModelRoutedStrategy(engine)
    cache_mgr = TopologyCacheManager(topo, cache_fraction=0.5)
    cache_mgr.reset()

    workload = build_workload(batch_size, table_rows, seed=SEED)

    kahan = KahanSum()
    welford = WelfordAccumulator()
    route_counts = defaultdict(int)
    latencies = []

    wall_start = time.perf_counter()

    for q in workload:
        decision = strategy.route_one(q, data_location="cpu0")
        dev = decision.device_id

        sched = scheduler.schedule(q, data_location="cpu0")
        lat = sched.latency_us

        latencies.append(lat)
        kahan.add(lat)
        welford.update(lat)
        route_counts[dev] += 1

        # 缓存模拟
        gpu_cache = cache_mgr.get(dev)
        if gpu_cache:
            blocks = gpu_cache.required_blocks(q)
            gpu_cache.lookup(blocks)
            gpu_cache.release(blocks)

    wall_elapsed = time.perf_counter() - wall_start

    total_latency_us = kahan.total
    mean_latency_us = welford.mean
    std_latency_us = welford.std

    # 吞吐量: queries / (模拟总延迟转换为秒)
    # 用wall-clock时间计算实际吞吐
    throughput_qps = batch_size / wall_elapsed if wall_elapsed > 0 else 0.0

    # 也计算模拟吞吐 (基于cost model延迟)
    total_latency_sec = total_latency_us / 1e6
    simulated_throughput_qps = batch_size / total_latency_sec if total_latency_sec > 0 else 0.0

    return {
        "table_rows": table_rows,
        "batch_size": batch_size,
        "total_latency_us": round(total_latency_us, 2),
        "mean_latency_us": round(mean_latency_us, 2),
        "std_latency_us": round(std_latency_us, 2),
        "throughput_qps": round(throughput_qps, 2),
        "simulated_throughput_qps": round(simulated_throughput_qps, 2),
        "wall_clock_sec": round(wall_elapsed, 4),
        "route_distribution": dict(route_counts),
        "cache_hit_rate": round(cache_mgr.aggregate_hit_rate(), 4),
    }


# ═══════════════════════════════════════════════════════════════
# 主实验
# ═══════════════════════════════════════════════════════════════

def run_scalability_bench(output_dir: str = "output") -> Dict:
    """运行 5表规模 × 5批次 的完整扩展性基准。"""
    os.makedirs(output_dir, exist_ok=True)

    topo = create_default_topology()
    engine = CostModelEngine(topo)
    scheduler = QueryPipelineScheduler(engine)

    n_combos = len(TABLE_SIZES) * len(BATCH_SIZES)
    _dump_header("SCALABILITY BENCH CONFIG")
    _dbg("table_sizes", val=TABLE_SIZES)
    _dbg("batch_sizes", val=BATCH_SIZES)
    _dbg("strategy", val="CostModel-Routed")
    _dbg("total_combos", val=n_combos)
    _dump_footer(f"{len(TABLE_SIZES)} table sizes × {len(BATCH_SIZES)} batch sizes = {n_combos} combos")

    t0 = time.time()
    results_matrix = []
    combo_idx = 0

    for trows in TABLE_SIZES:
        for bsz in BATCH_SIZES:
            combo_idx += 1
            _dbg(f"[{combo_idx}/{n_combos}] table_rows={trows:,}, batch={bsz}")

            result = run_one_combo(engine, scheduler, topo, trows, bsz)
            results_matrix.append(result)

            _dbg(f"  done",
                 total=f"{result['total_latency_us']:.1f}µs",
                 mean=f"{result['mean_latency_us']:.1f}µs",
                 throughput=f"{result['throughput_qps']:.1f} q/s",
                 sim_throughput=f"{result['simulated_throughput_qps']:.1f} q/s")

    elapsed = time.time() - t0

    # 构建汇总: 按表大小聚合
    by_table_size = {}
    for trows in TABLE_SIZES:
        subset = [r for r in results_matrix if r["table_rows"] == trows]
        by_table_size[str(trows)] = {
            "mean_of_means": round(
                sum(r["mean_latency_us"] for r in subset) / len(subset), 2),
            "max_throughput": round(
                max(r["simulated_throughput_qps"] for r in subset), 2),
            "batch_results": subset,
        }

    # 按批次大小聚合
    by_batch_size = {}
    for bsz in BATCH_SIZES:
        subset = [r for r in results_matrix if r["batch_size"] == bsz]
        by_batch_size[str(bsz)] = {
            "mean_of_means": round(
                sum(r["mean_latency_us"] for r in subset) / len(subset), 2),
            "total_latency_range": [
                round(min(r["total_latency_us"] for r in subset), 2),
                round(max(r["total_latency_us"] for r in subset), 2),
            ],
        }

    # 找出最佳和最差组合
    best = min(results_matrix, key=lambda r: r["mean_latency_us"])
    worst = max(results_matrix, key=lambda r: r["mean_latency_us"])

    # 扩展性比率: 100M行 vs 100K行 的mean latency倍数 (固定batch=1000)
    lat_100k = next(
        (r["mean_latency_us"] for r in results_matrix
         if r["table_rows"] == 100_000 and r["batch_size"] == 1000), None)
    lat_100m = next(
        (r["mean_latency_us"] for r in results_matrix
         if r["table_rows"] == 100_000_000 and r["batch_size"] == 1000), None)
    scalability_ratio = round(lat_100m / lat_100k, 2) if lat_100k and lat_100m else None

    output = {
        "metadata": {
            "panel": "Scalability Benchmark — CostModel-Routed",
            "strategy": "CostModel-Routed",
            "table_sizes": TABLE_SIZES,
            "batch_sizes": BATCH_SIZES,
            "n_combos": n_combos,
            "total_queries_executed": sum(r["batch_size"] for r in results_matrix),
            "elapsed_seconds": round(elapsed, 2),
            "seed": SEED,
            "algorithms": {
                "cumulative": "Kahan compensated summation",
                "variance": "Welford single-pass online",
                "workload": "TPC-H 4-table mixed (SCAN/JOIN/INDEX/AGG)",
            },
        },
        "matrix": results_matrix,
        "by_table_size": by_table_size,
        "by_batch_size": by_batch_size,
        "summary": {
            "best_combo": {
                "table_rows": best["table_rows"],
                "batch_size": best["batch_size"],
                "mean_latency_us": best["mean_latency_us"],
            },
            "worst_combo": {
                "table_rows": worst["table_rows"],
                "batch_size": worst["batch_size"],
                "mean_latency_us": worst["mean_latency_us"],
            },
            "scalability_ratio_1000x_rows": scalability_ratio,
        },
    }

    out_path = os.path.join(output_dir, "scalability_bench.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # 打印简洁的ASCII矩阵到stderr
    _print_summary_table(results_matrix, elapsed, out_path)

    return output


def _format_rows(n: int) -> str:
    """把行数格式化为可读字符串: 100000 → '100K', 1000000 → '1M' 等。"""
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    return f"{n // 1_000}K"


def _print_summary_table(results: List[Dict], elapsed: float, out_path: str) -> None:
    """打印美观的ASCII汇总矩阵。"""
    print(f"\n{'═' * 72}", file=sys.stderr)
    print(f"  SCALABILITY BENCH — CostModel-Routed Strategy", file=sys.stderr)
    print(f"{'═' * 72}", file=sys.stderr)

    # mean latency矩阵
    print(f"\n  Mean Latency (µs):", file=sys.stderr)
    header = f"  {'Rows':>8s}"
    for bsz in BATCH_SIZES:
        header += f"  {f'B={bsz}':>12s}"
    print(header, file=sys.stderr)
    print(f"  {'─' * 8}" + f"  {'─' * 12}" * len(BATCH_SIZES), file=sys.stderr)

    for trows in TABLE_SIZES:
        row_str = f"  {_format_rows(trows):>8s}"
        for bsz in BATCH_SIZES:
            r = next(x for x in results
                     if x["table_rows"] == trows and x["batch_size"] == bsz)
            row_str += f"  {r['mean_latency_us']:>12.1f}"
        print(row_str, file=sys.stderr)

    # throughput矩阵 (simulated)
    print(f"\n  Simulated Throughput (queries/sec):", file=sys.stderr)
    print(header, file=sys.stderr)
    print(f"  {'─' * 8}" + f"  {'─' * 12}" * len(BATCH_SIZES), file=sys.stderr)

    for trows in TABLE_SIZES:
        row_str = f"  {_format_rows(trows):>8s}"
        for bsz in BATCH_SIZES:
            r = next(x for x in results
                     if x["table_rows"] == trows and x["batch_size"] == bsz)
            row_str += f"  {r['simulated_throughput_qps']:>12.1f}"
        print(row_str, file=sys.stderr)

    print(f"\n{'─' * 72}", file=sys.stderr)
    print(f"  Elapsed: {elapsed:.1f}s | Output: {out_path}", file=sys.stderr)
    print(f"{'═' * 72}\n", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

def main():
    run_scalability_bench()


if __name__ == "__main__":
    main()
