#!/usr/bin/env python3
"""
scripts/cache_effectiveness.py — 缓存效果分析 (M301-M310)

测试 TopologyCacheManager 的缓存效果:
  1. 冷启动 vs 预热后的命中率变化曲线
  2. 不同 cache_fraction (0.1, 0.3, 0.5, 0.7, 0.9) 下的表现
  3. 重复查询 vs 随机查询的命中率对比

输出: output/cache_effectiveness.json

算法改写:
  - 使用 Welford 在线算法计算多 seed 的 hit_rate 均值/标准差
  - PRNG 基于 seed 确定性生成查询序列, 保证可复现性
  - 滑窗命中率 (window=20) 刻画预热曲线的局部趋势
  - 构造受限拓扑 (小显存 GPU) 使缓存压力可见

用法:
    python scripts/cache_effectiveness.py
    python scripts/cache_effectiveness.py --steps 200 --seeds 5
    python scripts/cache_effectiveness.py --steps 50 --seeds 1   # smoke test
"""
import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("LYNCEUS_DBG", "0")

from lynceus.costing import QueryDescriptor, QueryType
from lynceus.cache_manager import TopologyCacheManager, DEFAULT_BLOCK_BYTES
from lynceus.schema import (
    HardwareKind, HardwareNode, HardwareTopology, TopologyEdge,
)


# ═══════════════════════════════════════════════════════════════
# 受限拓扑: 小显存 GPU 使缓存压力可见
# ═══════════════════════════════════════════════════════════════

def create_constrained_topology(gpu_memory_mb: int = 8) -> HardwareTopology:
    """构造受限拓扑: 4 个小显存 GPU + 2 CPU。

    默认 gpu_memory = 8 MB → 在 cache_fraction=0.1 时仅有 ~0.8MB
    ≈ 0 个 2MB block, 有效模拟缓存压力。在 fraction=0.9 时
    ≈ 7.2MB = 3 个 block, 足以观察命中率随 fraction 变化。
    """
    topo = HardwareTopology()
    mem = gpu_memory_mb * (1 << 20)

    cpu = HardwareNode(
        node_id="cpu0", kind=HardwareKind.CPU,
        compute_capacity=1.0, memory_bytes=256 * (1 << 30),
        scan_cost_per_row=1.0, seek_cost=4.0, compute_cost_per_op=0.01,
    )
    topo.add_node(cpu)

    for i in range(4):
        gpu = HardwareNode(
            node_id=f"gpu{i}", kind=HardwareKind.GPU,
            compute_capacity=110.0, memory_bytes=mem,
            bandwidth_gbps=2000.0,
            scan_cost_per_row=0.001, seek_cost=0.01, compute_cost_per_op=0.0001,
        )
        topo.add_node(gpu)
        topo.add_edge(TopologyEdge(
            src="cpu0", dst=gpu.node_id,
            bandwidth_gbps=32.0, latency_us=1.0,
            link_type=HardwareKind.PCIE,
        ))

    return topo


# ═══════════════════════════════════════════════════════════════
# 查询模板 — 使用小 block_bytes 匹配小显存
# ═══════════════════════════════════════════════════════════════

BLOCK_BYTES = 64 * 1024   # 64KB blocks (vs default 2MB)

QUERY_TEMPLATES = [
    # (query_type, index_available, selectivity, estimated_rows, table_rows, table_name, width)
    (QueryType.POINT_LOOKUP,    True,  0.001,    100,  100_000, "orders",   120),
    (QueryType.RANGE_SCAN,      True,  0.05,    5000,  100_000, "orders",   120),
    (QueryType.INDEX_SCAN,      True,  0.02,    2000,  100_000, "lineitem", 80),
    (QueryType.FULL_TABLE_SCAN, False, 1.0,   100000,  100_000, "lineitem", 80),
    (QueryType.POINT_LOOKUP,    True,  0.0005,    50,  100_000, "customer", 200),
    (QueryType.RANGE_SCAN,      True,  0.1,   10000,  100_000, "customer", 200),
    (QueryType.INDEX_SCAN,      True,  0.01,    1000,  100_000, "part",     100),
    (QueryType.AGGREGATE,       False, 0.5,   50000,  100_000, "part",     100),
    (QueryType.POINT_LOOKUP,    True,  0.002,    200,  100_000, "supplier", 150),
    (QueryType.RANGE_SCAN,      True,  0.08,    8000,  100_000, "supplier", 150),
]

CACHE_FRACTIONS = [0.1, 0.3, 0.5, 0.7, 0.9]


def make_query(template_idx: int, seq: int) -> QueryDescriptor:
    """从模板创建一条 QueryDescriptor。"""
    qt, idx_avail, sel, est_rows, tbl_rows, tbl_name, width = \
        QUERY_TEMPLATES[template_idx]
    return QueryDescriptor(
        query_id=f"{tbl_name}::{qt.name}_{seq}",
        query_type=qt,
        estimated_rows=est_rows,
        estimated_width_bytes=width,
        selectivity=sel,
        table_rows=tbl_rows,
        index_available=idx_avail,
        table_name=tbl_name,
    )


def generate_repeated_sequence(rng: random.Random, n_steps: int,
                               n_templates: int, repeat_ratio: float = 0.7
                               ) -> List[int]:
    """生成带重复的查询模板索引序列。

    repeat_ratio 比例的查询从 hot set (前 30% 模板) 中选取,
    模拟真实负载中少量高频查询主导缓存命中的 Zipf 分布。
    """
    hot_size = max(1, int(n_templates * 0.3))
    hot_set = list(range(hot_size))
    cold_set = list(range(n_templates))
    seq = []
    for _ in range(n_steps):
        if rng.random() < repeat_ratio:
            seq.append(rng.choice(hot_set))
        else:
            seq.append(rng.choice(cold_set))
    return seq


def generate_random_sequence(rng: random.Random, n_steps: int,
                             n_templates: int) -> List[int]:
    """生成均匀随机查询模板索引序列 — 无 hot set 偏向。"""
    return [rng.randrange(n_templates) for _ in range(n_steps)]


def sliding_window_hit_rate(hits_timeline: List[int], window: int = 20
                            ) -> List[float]:
    """滑窗命中率: 每步输出最近 window 次查询的命中率。"""
    rates = []
    buf = []
    for h in hits_timeline:
        buf.append(h)
        if len(buf) > window:
            buf.pop(0)
        rates.append(sum(buf) / len(buf))
    return rates


class WelfordAggregator:
    """Welford 在线算法, 逐步追加值, 最终输出均值/标准差。"""

    def __init__(self, length: int):
        self.n = 0
        self.length = length
        self._mean = [0.0] * length
        self._m2 = [0.0] * length

    def push(self, values: List[float]):
        assert len(values) == self.length
        self.n += 1
        for i in range(self.length):
            delta = values[i] - self._mean[i]
            self._mean[i] += delta / self.n
            delta2 = values[i] - self._mean[i]
            self._m2[i] += delta * delta2

    @property
    def mean(self) -> List[float]:
        return list(self._mean)

    @property
    def std(self) -> List[float]:
        if self.n < 2:
            return [0.0] * self.length
        return [math.sqrt(self._m2[i] / (self.n - 1)) for i in range(self.length)]


# ═══════════════════════════════════════════════════════════════
# 运行单轮实验的核心逻辑
# ═══════════════════════════════════════════════════════════════

def run_single_trial(topology: HardwareTopology, cache_fraction: float,
                     query_sequence: List[int], n_steps: int
                     ) -> Dict[str, Any]:
    """在给定拓扑/fraction/查询序列下运行单轮, 返回逐步统计。"""
    cm = TopologyCacheManager(topology, cache_fraction=cache_fraction,
                              block_bytes=BLOCK_BYTES)
    device_ids = list(cm.caches.keys())
    if not device_ids:
        raise RuntimeError("No GPU caches in topology")

    hits_timeline = []  # 每步: 1=全命中, 0=有miss
    total_hits = 0
    total_lookups = 0
    cumulative_rates = []

    for step in range(n_steps):
        tmpl_idx = query_sequence[step]
        q = make_query(tmpl_idx, step)
        dev = device_ids[step % len(device_ids)]
        cache = cm.get(dev)
        blocks = cache.required_blocks(q)
        h, m = cache.lookup(blocks)
        cache.release(blocks)

        hits_timeline.append(1 if m == 0 else 0)
        total_hits += h
        total_lookups += h + m
        cumulative_rates.append(
            total_hits / total_lookups if total_lookups else 0.0)

    total_evictions = sum(c.stats.evictions for c in cm.caches.values())
    total_prefetches = sum(c.stats.prefetches for c in cm.caches.values())
    total_resident = sum(c.resident_blocks for c in cm.caches.values())

    return {
        "cumulative_rates": cumulative_rates,
        "hits_timeline": hits_timeline,
        "final_hit_rate": cm.aggregate_hit_rate(),
        "evictions": total_evictions,
        "prefetches": total_prefetches,
        "resident_blocks": total_resident,
    }


# ═══════════════════════════════════════════════════════════════
# 实验 1: 冷启动 vs 预热后命中率曲线
# ═══════════════════════════════════════════════════════════════

def experiment_warmup_curve(topology, n_steps: int, n_seeds: int,
                            cache_fraction: float = 0.5) -> Dict[str, Any]:
    """跟踪每一步的累积命中率和滑窗命中率, 观测冷启动→稳态的预热过程。"""
    agg_cumulative = WelfordAggregator(n_steps)
    agg_sliding = WelfordAggregator(n_steps)

    for seed in range(n_seeds):
        rng = random.Random(42 + seed)
        seq = generate_repeated_sequence(rng, n_steps, len(QUERY_TEMPLATES))
        trial = run_single_trial(topology, cache_fraction, seq, n_steps)

        agg_cumulative.push(trial["cumulative_rates"])
        sliding = sliding_window_hit_rate(trial["hits_timeline"], window=20)
        agg_sliding.push(sliding)

    return {
        "description": "冷启动→预热命中率曲线 (cache_fraction=%.1f)" % cache_fraction,
        "n_steps": n_steps,
        "n_seeds": n_seeds,
        "cumulative_hit_rate": {
            "mean": [round(v, 6) for v in agg_cumulative.mean],
            "std": [round(v, 6) for v in agg_cumulative.std],
        },
        "sliding_window_hit_rate": {
            "window": 20,
            "mean": [round(v, 6) for v in agg_sliding.mean],
            "std": [round(v, 6) for v in agg_sliding.std],
        },
    }


# ═══════════════════════════════════════════════════════════════
# 实验 2: 不同 cache_fraction 下的表现
# ═══════════════════════════════════════════════════════════════

def experiment_cache_fraction_sweep(topology, n_steps: int, n_seeds: int
                                    ) -> Dict[str, Any]:
    """扫描 cache_fraction ∈ {0.1, 0.3, 0.5, 0.7, 0.9}, 对比最终命中率。"""
    results = {}

    for frac in CACHE_FRACTIONS:
        agg_final = WelfordAggregator(1)
        agg_curve = WelfordAggregator(n_steps)

        for seed in range(n_seeds):
            rng = random.Random(42 + seed)
            seq = generate_repeated_sequence(rng, n_steps, len(QUERY_TEMPLATES))
            trial = run_single_trial(topology, frac, seq, n_steps)

            agg_final.push([trial["final_hit_rate"]])
            agg_curve.push(trial["cumulative_rates"])

        # Report block capacity at this fraction
        cm_probe = TopologyCacheManager(topology, cache_fraction=frac,
                                        block_bytes=BLOCK_BYTES)

        results[str(frac)] = {
            "cache_fraction": frac,
            "final_hit_rate_mean": round(agg_final.mean[0], 6),
            "final_hit_rate_std": round(agg_final.std[0], 6),
            "curve_mean": [round(v, 6) for v in agg_curve.mean],
            "curve_std": [round(v, 6) for v in agg_curve.std],
            "num_blocks_per_device": {
                dev: c.num_blocks for dev, c in cm_probe.caches.items()
            },
        }

    return {
        "description": "cache_fraction 扫描: 最终命中率 vs 缓存比例",
        "fractions": CACHE_FRACTIONS,
        "results": results,
    }


# ═══════════════════════════════════════════════════════════════
# 实验 3: 重复查询 vs 随机查询的命中率对比
# ═══════════════════════════════════════════════════════════════

def experiment_repeated_vs_random(topology, n_steps: int, n_seeds: int,
                                  cache_fraction: float = 0.5
                                  ) -> Dict[str, Any]:
    """对比带 hot-set 的重复查询和均匀随机查询在同一缓存配置下的命中率差异。"""
    generators = {
        "repeated_70pct_hot": lambda rng, ns: generate_repeated_sequence(
            rng, ns, len(QUERY_TEMPLATES), repeat_ratio=0.7),
        "uniform_random": lambda rng, ns: generate_random_sequence(
            rng, ns, len(QUERY_TEMPLATES)),
    }
    output = {}

    for mode_name, gen_fn in generators.items():
        agg_curve = WelfordAggregator(n_steps)
        agg_final = WelfordAggregator(1)
        agg_stats = WelfordAggregator(3)

        for seed in range(n_seeds):
            rng = random.Random(42 + seed)
            seq = gen_fn(rng, n_steps)
            trial = run_single_trial(topology, cache_fraction, seq, n_steps)

            agg_curve.push(trial["cumulative_rates"])
            agg_final.push([trial["final_hit_rate"]])
            agg_stats.push([
                float(trial["evictions"]),
                float(trial["prefetches"]),
                float(trial["resident_blocks"]),
            ])

        output[mode_name] = {
            "final_hit_rate_mean": round(agg_final.mean[0], 6),
            "final_hit_rate_std": round(agg_final.std[0], 6),
            "cumulative_curve_mean": [round(v, 6) for v in agg_curve.mean],
            "cumulative_curve_std": [round(v, 6) for v in agg_curve.std],
            "evictions_mean": round(agg_stats.mean[0], 2),
            "prefetches_mean": round(agg_stats.mean[1], 2),
            "resident_blocks_mean": round(agg_stats.mean[2], 2),
        }

    r_hit = output["repeated_70pct_hot"]["final_hit_rate_mean"]
    u_hit = output["uniform_random"]["final_hit_rate_mean"]
    delta = r_hit - u_hit

    return {
        "description": "重复查询(70% hot) vs 均匀随机 — 命中率对比",
        "cache_fraction": cache_fraction,
        "n_steps": n_steps,
        "n_seeds": n_seeds,
        "modes": output,
        "delta_hit_rate": round(delta, 6),
        "delta_pct": round(delta * 100, 2),
    }


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="缓存效果分析 — 命中率 × cache_fraction")
    parser.add_argument("--steps", type=int, default=300,
                        help="每轮查询步数 (default: 300)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="随机种子数 (default: 3)")
    parser.add_argument("--gpu-memory-mb", type=int, default=8,
                        help="受限 GPU 显存 MB (default: 8)")
    parser.add_argument("--output", type=str,
                        default="output/cache_effectiveness.json",
                        help="输出 JSON 路径")
    args = parser.parse_args()

    print("=" * 60)
    print("  缓存效果分析 — TopologyCacheManager")
    print(f"  steps={args.steps}  seeds={args.seeds}  "
          f"gpu_mem={args.gpu_memory_mb}MB  block={BLOCK_BYTES//1024}KB")
    print("=" * 60)

    t0 = time.time()
    topology = create_constrained_topology(gpu_memory_mb=args.gpu_memory_mb)

    # 实验 1: 冷启动 vs 预热后命中率曲线
    print("\n[1/3] 冷启动 vs 预热后命中率曲线 ...")
    warmup = experiment_warmup_curve(topology, args.steps, args.seeds)
    warmup_mean = warmup["cumulative_hit_rate"]["mean"]
    avg_first = sum(warmup_mean[:10]) / min(10, len(warmup_mean))
    avg_last = sum(warmup_mean[-10:]) / min(10, len(warmup_mean))
    print(f"      冷启动前10步平均命中率: {avg_first:.4f}")
    print(f"      稳态后10步平均命中率:   {avg_last:.4f}")

    # 实验 2: cache_fraction 扫描
    print("\n[2/3] cache_fraction 扫描 ...")
    fraction_sweep = experiment_cache_fraction_sweep(
        topology, args.steps, args.seeds)
    for frac in CACHE_FRACTIONS:
        r = fraction_sweep["results"][str(frac)]
        blocks = list(r["num_blocks_per_device"].values())
        b0 = blocks[0] if blocks else 0
        print(f"      fraction={frac:.1f}  "
              f"hit_rate={r['final_hit_rate_mean']:.4f} "
              f"± {r['final_hit_rate_std']:.4f}  "
              f"blocks/dev={b0}")

    # 实验 3: 重复 vs 随机
    print("\n[3/3] 重复查询 vs 随机查询 ...")
    rep_vs_rand = experiment_repeated_vs_random(
        topology, args.steps, args.seeds)
    for mode, info in rep_vs_rand["modes"].items():
        print(f"      {mode:25s}  "
              f"hit_rate={info['final_hit_rate_mean']:.4f} "
              f"± {info['final_hit_rate_std']:.4f}  "
              f"evictions={info['evictions_mean']:.0f}")
    print(f"      delta (repeated - random): "
          f"{rep_vs_rand['delta_pct']:+.2f}%")

    elapsed = time.time() - t0

    # 汇总输出
    output = {
        "metadata": {
            "script": "scripts/cache_effectiveness.py",
            "task": "M301-M310",
            "description": "缓存效果分析 — 命中率 × cache_fraction",
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "params": {
                "steps": args.steps,
                "seeds": args.seeds,
                "gpu_memory_mb": args.gpu_memory_mb,
                "block_bytes": BLOCK_BYTES,
                "cache_fractions": CACHE_FRACTIONS,
                "query_templates": len(QUERY_TEMPLATES),
            },
            "elapsed_seconds": round(elapsed, 2),
        },
        "warmup_curve": warmup,
        "cache_fraction_sweep": fraction_sweep,
        "repeated_vs_random": rep_vs_rand,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  完成! 耗时 {elapsed:.1f}s")
    print(f"  输出: {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
