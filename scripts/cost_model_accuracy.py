#!/usr/bin/env python3
"""
scripts/cost_model_accuracy.py — CostModel 预测准确性评估 (M291-M300)

生成 1000 个随机查询，对每个查询:
  1. cost_model.recommend() → 推荐设备 + 预估延迟
  2. scheduler.schedule()   → 真实延迟 (pipeline critical path)
  3. 对比推荐 vs 非推荐设备的延迟差距

输出指标:
  - 推荐正确率 (recommend accuracy): 推荐设备是否是 scheduler 选出的最低延迟设备
  - MAE  (Mean Absolute Error): |预估 - 真实| 的均值 (µs)
  - MAPE (Mean Absolute Percentage Error): |预估 - 真实| / 真实 的均值 (%)
  - 推荐 vs 非推荐延迟差距: 推荐设备的真实延迟 相对 非推荐设备均值的节省

用法:
    python scripts/cost_model_accuracy.py
    python scripts/cost_model_accuracy.py --num-queries 500 --seed 42

输出: output/cost_model_accuracy.json
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
os.environ["LYNCEUS_DBG"] = "0"  # 关闭 debug 噪声

from lynceus.cost_model import (
    CostModelEngine, CostBreakdown, QueryDescriptor, QueryType,
    create_default_topology,
)
from lynceus.pipeline_scheduler import QueryPipelineScheduler


# ---------------------------------------------------------------------------
# 随机查询生成器
# ---------------------------------------------------------------------------

ALL_QUERY_TYPES = list(QueryType)

def gen_random_query(rng: random.Random, idx: int) -> QueryDescriptor:
    """生成一个参数范围合理的随机查询。

    参数空间覆盖:
      - 7 种 query type (POINT_LOOKUP ~ SORT)
      - estimated_rows: 1 ~ 5M (对数均匀)
      - selectivity: 0.0001 ~ 1.0 (对数均匀)
      - num_predicates: 0 ~ 8
      - num_joins: 0 ~ 4
      - sort / group_by: 随机触发
      - index_available: 40% 概率有索引
    """
    qtype = rng.choice(ALL_QUERY_TYPES)

    # 对数均匀: 覆盖小查询(10行)到大查询(5M行)
    table_rows = int(10 ** rng.uniform(3, 7))  # 1K ~ 10M
    selectivity = 10 ** rng.uniform(-4, 0)     # 0.0001 ~ 1.0
    selectivity = min(1.0, selectivity)
    estimated_rows = max(1, int(table_rows * selectivity))

    num_predicates = rng.randint(0, 8)
    num_joins = rng.randint(0, 4) if qtype in (QueryType.JOIN,) else rng.randint(0, 2)
    sort_required = rng.random() < 0.35
    group_by_card = rng.randint(1, min(estimated_rows, 10000)) if rng.random() < 0.3 else 0
    index_available = rng.random() < 0.4
    width = rng.choice([8, 32, 64, 100, 200, 512, 1024])

    return QueryDescriptor(
        query_id=f"q{idx:04d}",
        query_type=qtype,
        estimated_rows=estimated_rows,
        estimated_width_bytes=width,
        num_predicates=num_predicates,
        selectivity=selectivity,
        table_rows=table_rows,
        index_available=index_available,
        index_depth=rng.randint(2, 5),
        num_joins=num_joins,
        sort_required=sort_required,
        group_by_cardinality=group_by_card,
        table_name=f"t{idx % 20}",
    )


# ---------------------------------------------------------------------------
# Welford 在线统计
# ---------------------------------------------------------------------------

class WelfordAccumulator:
    """在线 mean/variance，避免二次遍历。"""
    __slots__ = ("n", "_m", "_s")

    def __init__(self):
        self.n = 0
        self._m = 0.0
        self._s = 0.0

    def push(self, x: float):
        self.n += 1
        if self.n == 1:
            self._m = x
            self._s = 0.0
        else:
            prev = self._m
            self._m += (x - prev) / self.n
            self._s += (x - prev) * (x - self._m)

    @property
    def mean(self) -> float:
        return self._m if self.n > 0 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self._s / (self.n - 1)) if self.n > 1 else 0.0


# ---------------------------------------------------------------------------
# 主评估逻辑
# ---------------------------------------------------------------------------

def evaluate(num_queries: int = 1000, seed: int = 2026,
             data_location: str = "cpu0") -> Dict[str, Any]:
    """运行完整评估流程，返回结果字典。"""

    rng = random.Random(seed)
    topo = create_default_topology()
    engine = CostModelEngine(topo)
    scheduler = QueryPipelineScheduler(engine)

    # 获取所有可用设备 (CPU + GPU)
    from lynceus.schema import HardwareKind
    all_devices = sorted(
        nid for nid, node in topo.nodes.items()
        if node.kind in (HardwareKind.GPU, HardwareKind.CPU)
    )

    # 累积统计器
    correct_count = 0
    mae_acc = WelfordAccumulator()
    mape_acc = WelfordAccumulator()
    latency_gap_acc = WelfordAccumulator()   # 推荐设备 vs 非推荐设备均值
    per_query_results: List[Dict[str, Any]] = []

    # 按设备统计推荐频次
    recommend_freq: Dict[str, int] = {d: 0 for d in all_devices}
    # 按 query type 统计正确率
    type_correct: Dict[str, int] = {qt.name: 0 for qt in QueryType}
    type_total: Dict[str, int] = {qt.name: 0 for qt in QueryType}

    t0 = time.perf_counter()

    for i in range(num_queries):
        query = gen_random_query(rng, i)

        # --- CostModel 推荐 ---
        rec_device, rec_cost = engine.recommend(query, data_location)
        predicted_us = rec_cost.total_us
        recommend_freq[rec_device] = recommend_freq.get(rec_device, 0) + 1

        # --- Scheduler 真实延迟 (pipeline critical path) ---
        schedule = scheduler.schedule(query, data_location)
        actual_us = schedule.latency_us

        # --- 确定"真实最优设备": 对每个设备单独做 schedule ---
        # scheduler.schedule 内部会对每个 stage 调 recommend()，设备选择
        # 是逐 stage 的。为了评估 recommend() 的全局推荐是否正确，
        # 我们比较: recommend 选的设备 vs 在所有设备上直接用 estimate_on_device
        # 得到的最低 total cost 设备。
        all_estimates = engine.estimate_all_devices(query, data_location)
        true_best_device = min(all_estimates, key=lambda d: all_estimates[d].total_us)
        true_best_us = all_estimates[true_best_device].total_us

        is_correct = (rec_device == true_best_device)
        if is_correct:
            correct_count += 1

        # 按 query type 统计
        qt_name = query.query_type.name
        type_total[qt_name] += 1
        if is_correct:
            type_correct[qt_name] += 1

        # --- MAE / MAPE: 预估 vs 真实 (estimate_on_device vs schedule latency) ---
        abs_err = abs(predicted_us - actual_us)
        mae_acc.push(abs_err)
        if actual_us > 0:
            pct_err = abs_err / actual_us * 100.0
            mape_acc.push(pct_err)

        # --- 推荐 vs 非推荐延迟差距 ---
        non_rec_costs = [
            cb.total_us for d, cb in all_estimates.items() if d != rec_device
        ]
        if non_rec_costs:
            avg_non_rec = sum(non_rec_costs) / len(non_rec_costs)
            if avg_non_rec > 0:
                gap_pct = (avg_non_rec - predicted_us) / avg_non_rec * 100.0
                latency_gap_acc.push(gap_pct)
            else:
                gap_pct = 0.0
        else:
            avg_non_rec = 0.0
            gap_pct = 0.0

        per_query_results.append({
            "query_id": query.query_id,
            "query_type": qt_name,
            "estimated_rows": query.estimated_rows,
            "recommended_device": rec_device,
            "true_best_device": true_best_device,
            "correct": is_correct,
            "predicted_us": round(predicted_us, 2),
            "schedule_latency_us": round(actual_us, 2),
            "true_best_estimate_us": round(true_best_us, 2),
            "abs_error_us": round(abs_err, 2),
            "gap_vs_non_rec_pct": round(gap_pct, 2),
        })

    elapsed = time.perf_counter() - t0

    # --- 汇总指标 ---
    accuracy = correct_count / num_queries * 100.0 if num_queries > 0 else 0.0

    per_type_accuracy = {}
    for qt in QueryType:
        n = type_total[qt.name]
        c = type_correct[qt.name]
        per_type_accuracy[qt.name] = {
            "total": n,
            "correct": c,
            "accuracy_pct": round(c / n * 100.0, 2) if n > 0 else 0.0,
        }

    summary = {
        "num_queries": num_queries,
        "seed": seed,
        "data_location": data_location,
        "devices": all_devices,
        "elapsed_s": round(elapsed, 3),
        "accuracy": {
            "recommend_correct": correct_count,
            "recommend_total": num_queries,
            "recommend_accuracy_pct": round(accuracy, 2),
        },
        "error_metrics": {
            "MAE_us": round(mae_acc.mean, 4),
            "MAE_std_us": round(mae_acc.std, 4),
            "MAPE_pct": round(mape_acc.mean, 4),
            "MAPE_std_pct": round(mape_acc.std, 4),
        },
        "latency_gap": {
            "mean_gap_pct": round(latency_gap_acc.mean, 4),
            "std_gap_pct": round(latency_gap_acc.std, 4),
            "description": "推荐设备 vs 非推荐设备均值的延迟节省百分比 (正值=推荐更快)",
        },
        "recommend_frequency": recommend_freq,
        "per_type_accuracy": per_type_accuracy,
    }

    return {
        "summary": summary,
        "per_query": per_query_results,
    }


# ---------------------------------------------------------------------------
# ASCII 摘要打印
# ---------------------------------------------------------------------------

def print_summary(result: Dict[str, Any]):
    s = result["summary"]
    acc = s["accuracy"]
    err = s["error_metrics"]
    gap = s["latency_gap"]

    print("\n" + "=" * 64)
    print("  CostModel Accuracy Evaluation")
    print("=" * 64)
    print(f"  Queries: {s['num_queries']}  |  Seed: {s['seed']}  |  "
          f"Data: {s['data_location']}  |  Time: {s['elapsed_s']}s")
    print(f"  Devices: {', '.join(s['devices'])}")
    print("-" * 64)

    print(f"\n  Recommend Accuracy:  {acc['recommend_correct']}/{acc['recommend_total']}"
          f"  ({acc['recommend_accuracy_pct']:.2f}%)")
    print(f"  MAE:   {err['MAE_us']:.2f} ± {err['MAE_std_us']:.2f} µs")
    print(f"  MAPE:  {err['MAPE_pct']:.2f} ± {err['MAPE_std_pct']:.2f} %")
    print(f"  Gap:   {gap['mean_gap_pct']:.2f} ± {gap['std_gap_pct']:.2f} %"
          f"  (positive = recommended is faster)")

    print(f"\n  Per-type accuracy:")
    for qt_name, info in s["per_type_accuracy"].items():
        if info["total"] > 0:
            bar_len = int(info["accuracy_pct"] / 100 * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"    {qt_name:<16s} [{bar}] "
                  f"{info['correct']:>3d}/{info['total']:<3d} "
                  f"({info['accuracy_pct']:6.2f}%)")

    print(f"\n  Recommend frequency:")
    for dev, cnt in s["recommend_frequency"].items():
        pct = cnt / s["num_queries"] * 100
        bar_len = int(pct / 100 * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"    {dev:<6s} [{bar}] {cnt:>4d} ({pct:5.1f}%)")

    print("=" * 64)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CostModel prediction accuracy evaluation")
    parser.add_argument("--num-queries", type=int, default=1000,
                        help="Number of random queries (default: 1000)")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Random seed (default: 2026)")
    parser.add_argument("--data-location", type=str, default="cpu0",
                        help="Initial data location (default: cpu0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: output/cost_model_accuracy.json)")
    args = parser.parse_args()

    out_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "output",
        "cost_model_accuracy.json")

    result = evaluate(
        num_queries=args.num_queries,
        seed=args.seed,
        data_location=args.data_location,
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print_summary(result)
    print(f"\n  Output saved to: {out_path}")


if __name__ == "__main__":
    main()
