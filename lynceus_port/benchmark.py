"""
lynceus_port/benchmark.py — 移植版基准测试.

改写 ≈ 20%:
  - generate_query_sequence: 增加热点表概率 (zipf-like 表选择)
  - run_benchmark: 增加逐策略进度打印 + 断点 hook
  - main: 增加 --trace 模式 (每 100 步 dump 状态)
"""
from __future__ import annotations
import math, random, time, os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from .schema import (BenchmarkOutput, MetricKind, MethodResult, PanelResult,
                     RoutingStrategy, SeedCurve)
from .cost_model import (CostModelEngine, CPUCostModel, GPUCostModel,
                         QueryDescriptor, QueryType, CostBreakdown,
                         create_default_topology)
from .router import Router
from .strategies.base import RoutingStrategyBase
from . import _dbg

_MOD_TAG = "BEK"
import os as _os, sys as _sys
_LYNCEUS_DBG = _os.environ.get("LYNCEUS_DEBUG", "1")

def _dbg(tag: str, msg: str):
    if _LYNCEUS_DBG != "0":
        print(f"[{_MOD_TAG}·{tag}] {msg}", file=_sys.stderr, flush=True)

_tr = _dbg  # 兼容旧调用



@dataclass
# ─── 工作负载配置 ────────────────────────────────────────────────
# 参数空间定义: 查询类型混合比例, 选择率范围, 表行数,
# 索引可用概率, 实验步数与种子数.
class WorkloadConfig:
    name: str = "TPC-H_SF100"
    num_steps: int = 2000
    num_seeds: int = 3
    base_table_rows: int = 6_000_000
    query_mix: Dict[QueryType, float] = field(default_factory=lambda: {
        QueryType.POINT_LOOKUP: 0.15, QueryType.RANGE_SCAN: 0.25,
        QueryType.FULL_TABLE_SCAN: 0.10, QueryType.INDEX_SCAN: 0.20,
        QueryType.JOIN: 0.15, QueryType.AGGREGATE: 0.10,
        QueryType.SORT: 0.05,
    })
    selectivity_range: Tuple[float, float] = (0.00105, 0.495)
    index_availability_prob: float = 0.6
    tables: Dict[str, int] = field(default_factory=lambda: {
        "lineitem": 6_000_000, "orders": 1_500_000,
        "customer": 150_000, "part": 200_000, "supplier": 10_000,
    })
    # ★ 改写: 表热度 (zipf 分布的 s 参数)
    table_skew: float = 1.2


# ─── 查询序列生成 ────────────────────────────────────────────────
# 改编自 PAR2QO utility.py evenly_sample_card.
# 原版均匀采样基数; 移植版使用 zipf 分布更贴近真实工作负载.
# 表选择概率按访问频率加权, 选择率从配置的范围内随机采样.
def generate_query_sequence(config: WorkloadConfig,
                            seed: int) -> List[QueryDescriptor]:
    rng = random.Random(seed)
    queries = []
    types_weights = list(config.query_mix.items())
    types = [t for t, _ in types_weights]
    weights = [w for _, w in types_weights]

    catalog = dict(config.tables) if config.tables else {
        config.name: config.base_table_rows}
    table_names = list(catalog.keys())

    # ★ 改写: Zipf-like 表选择权重 (热点表更常被查询)
    n_tables = len(table_names)
    zipf_weights = [1.0 / ((i + 1) ** config.table_skew)
                    for i in range(n_tables)]

    for step in range(config.num_steps):
        qt = rng.choices(types, weights=weights, k=1)[0]

        # ★ Zipf 表选择
        table_name = rng.choices(table_names, weights=zipf_weights, k=1)[0]
        table_rows = catalog[table_name]

        selectivity = rng.uniform(*config.selectivity_range)
        estimated_rows = max(1, int(table_rows * selectivity))

        difficulty_factor = 1.0 + 0.495 * (step / config.num_steps)
        estimated_rows = min(int(estimated_rows * difficulty_factor), table_rows)

        q = QueryDescriptor(
            query_id=f"q_{step:05d}", query_type=qt,
            estimated_rows=estimated_rows,
            estimated_width_bytes=rng.randint(50, 500),
            num_predicates=rng.randint(1, 5),
            selectivity=selectivity, table_rows=table_rows,
            index_available=rng.random() < config.index_availability_prob,
            index_depth=rng.randint(2, 5),
            num_joins=rng.randint(0, 3) if qt == QueryType.JOIN else 0,
            sort_required=(qt == QueryType.SORT or rng.random() < 0.2),
            group_by_cardinality=rng.randint(10, 1000) if qt == QueryType.AGGREGATE else 0,
            table_name=table_name,
        )
        queries.append(q)

        # ★ 改写: 每 500 步 trace 一次查询分布
        if step % 500 == 0 and step > 0:
            _dbg("step", f"step {step}: last query → {q.dump_snapshot()}")

    return queries


# ─── 策略执行器 ──────────────────────────────────────────────────
# 管理多种路由策略 (CPU-only / GPU-only / Cost-driven / Adaptive)
# 的生命周期. 每个策略实现 RoutingStrategyBase 接口.
# 改编自 PAR2QO postgres.py 中对多个 plan candidate 的评估循环.
class StrategyExecutor:
    def __init__(self, engine: CostModelEngine):
        self.engine = engine
        self._router = Router.create_default(engine)

    def execute_strategy(self, strategy: RoutingStrategy,
                         queries: List[QueryDescriptor],
                         data_location: str = "cpu0") -> List[float]:
        self._router.set_active(strategy.value)
        decisions = self._router.route_batch(queries, data_location)
        return RoutingStrategyBase.decisions_to_latencies(decisions)

    def execute_gpu_only(self, queries, data_location="cpu0"):
        _dbg("EXECUTE_", f"execute_gpu_only(queries={queries}, data_location={data_location})")
        return self.execute_strategy(RoutingStrategy.GPU_ONLY, queries, data_location)
    def execute_cpu_only(self, queries, data_location="cpu0"):
        _dbg("EXECUTE_", f"execute_cpu_only(queries={queries}, data_location={data_location})")
        return self.execute_strategy(RoutingStrategy.CPU_ONLY, queries, data_location)
    def execute_hybrid_static(self, queries, data_location="cpu0", **_kw):
        _dbg("EXECUTE_", f"execute_hybrid_static(queries={queries}, data_location={data_location})")
        return self.execute_strategy(RoutingStrategy.HYBRID_STATIC, queries, data_location)
    def execute_cost_model_routed(self, queries, data_location="cpu0"):
        _dbg("EXECUTE_", f"execute_cost_model_routed(queries={queries}, data_location={data_location})")
        return self.execute_strategy(RoutingStrategy.COST_MODEL_ROUTED, queries, data_location)
    def execute_par2qo_enhanced(self, queries, data_location="cpu0"):
        _dbg("EXECUTE_", f"execute_par2qo_enhanced(queries={queries}, data_location={data_location})")
        return self.execute_strategy(RoutingStrategy.PAR2QO_ENHANCED, queries, data_location)


# ─── 基准测试主入口 ──────────────────────────────────────────────
# 流程: 初始化拓扑 → 创建 cost model engine → 注册策略 →
# 逐步执行 → 收集 predicted vs observed → 多种子聚合.
# 改编自 PAR2QO postgres.py get_real_latency 的中位数计时.
def run_benchmark(workload: WorkloadConfig,
                  strategies: Optional[List[RoutingStrategy]] = None,
                  output_path: Optional[str] = None) -> BenchmarkOutput:
    if strategies is None:
        strategies = [
            RoutingStrategy.GPU_ONLY, RoutingStrategy.CPU_ONLY,
            RoutingStrategy.HYBRID_STATIC, RoutingStrategy.COST_MODEL_ROUTED,
            RoutingStrategy.PAR2QO_ENHANCED, RoutingStrategy.ADAPTIVE,
        ]
    topology = create_default_topology()
    engine = CostModelEngine(topology)
    executor = StrategyExecutor(engine)

    output = BenchmarkOutput(
        description=f"Lynceus benchmark — {workload.name}",
        source="lynceus_port_benchmark_runner",
    )
    panel = output.add_panel(
        name=f"latency_vs_step_{workload.name}",
        metric=MetricKind.LATENCY_MS,
        x_label="workload_step", y_label="latency_ms",
    )

    for strat_idx, strategy in enumerate(strategies):
        t0 = time.monotonic()
        method = panel.add_method(strategy=strategy, num_steps=workload.num_steps,
                                  num_seeds=workload.num_seeds)
        method.x_values = list(range(workload.num_steps))

        for seed_idx in range(workload.num_seeds):
            seed_val = 42 + seed_idx * 1000
            queries = generate_query_sequence(workload, seed=seed_val)
            latencies = executor.execute_strategy(strategy, queries, "cpu0")
            sc = method.add_seed()
            sc.values = latencies
            if method.aggregate_cost is None:
                method.aggregate_cost = 0.0
            method.aggregate_cost += sum(latencies)

        method.aggregate_cost = (method.aggregate_cost or 0.0) / workload.num_seeds
        method.compute_statistics()
        wall_time_us = time.monotonic() - t0
        # ★ 改写: 进度打印
        print(f"  [{strat_idx+1}/{len(strategies)}] {strategy.value}: "
              f"final_mean={method.mean[-1]:.3f}ms ({wall_time_us:.2f}s)")

    if output_path:
        output.save(output_path)
    return output


# ─── 累积基准测试 ────────────────────────────────────────────────
# 逐步增加查询数量, 观察各策略的收敛行为.
# 改编自 PAR2QO diagram.py 的 savePlot (画 convergence 图).
# 移植版不画图, 而是输出 JSON 数据供离线分析.
def run_cumulative_benchmark(workload: WorkloadConfig,
                             strategies: Optional[List[RoutingStrategy]] = None,
                             output_path: Optional[str] = None) -> BenchmarkOutput:
    if strategies is None:
        strategies = [
            RoutingStrategy.GPU_ONLY, RoutingStrategy.CPU_ONLY,
            RoutingStrategy.HYBRID_STATIC, RoutingStrategy.COST_MODEL_ROUTED,
            RoutingStrategy.PAR2QO_ENHANCED, RoutingStrategy.ADAPTIVE,
        ]
    topology = create_default_topology()
    engine = CostModelEngine(topology)
    executor = StrategyExecutor(engine)

    output = BenchmarkOutput(
        description=f"Lynceus cumulative — {workload.name}",
        source="lynceus_port_cumulative",
    )
    panel = output.add_panel(
        name=f"cumulative_latency_{workload.name}",
        metric=MetricKind.LATENCY_MS,
        x_label="workload_step", y_label="cumulative_latency_ms",
    )

    for strategy in strategies:
        method = panel.add_method(strategy=strategy, num_steps=workload.num_steps,
                                  num_seeds=workload.num_seeds)
        method.x_values = list(range(workload.num_steps))
        for seed_idx in range(workload.num_seeds):
            seed_val = 42 + seed_idx * 1000
            queries = generate_query_sequence(workload, seed=seed_val)
            latencies = executor.execute_strategy(strategy, queries, "cpu0")
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


# ─── 命令行入口 ──────────────────────────────────────────────────
# 环境变量:
#   LYNCEUS_STEPS — 每种子步数 (默认 2000)
#   LYNCEUS_SEEDS — 随机种子数 (默认 5)
#   LYNCEUS_OUTDIR — 输出目录 (默认 ./results)
def main():
    def _int_env(key: str, default: int) -> int:
        _dbg("_INT_ENV", f"_int_env(key={key}, default={default})")
        raw = os.environ.get(key)
        if raw is None or raw.strip() == "":
            return default
        try:
            v = int(raw)
        except ValueError:
            print(f"  [warn] {key}={raw!r} not int; using {default}")
            return default
        if v <= 0:
            return default
        return v

    workload = WorkloadConfig(
        name=os.environ.get("WORKLOAD_NAME", "TPC-H_SF100"),
        num_steps=_int_env("NUM_STEPS", 2000),
        num_seeds=_int_env("NUM_SEEDS", 3),
    )

    print(f"Running Lynceus-PORT benchmark: {workload.name}")
    print(f"  Steps: {workload.num_steps}, Seeds: {workload.num_seeds}")

    output1 = run_benchmark(workload, output_path="output/latency_vs_step.json")
    print(f"\nPer-step wire_delay saved.")

    output2 = run_cumulative_benchmark(workload, output_path="output/cumulative_latency.json")
    print(f"Cumulative wire_delay saved.")
    print(f"\nMetadata: {output1.metadata}")


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════
# 断点调试辅助工具 — 不影响主流程, 仅用于脚本级调试
# ═══════════════════════════════════════════════════════════════════════

class BenchmarkDiagnostics:
    """基准诊断工具 — 改编自 PAR2QO diagram.py 的统计分析部分.
    
    原版用 matplotlib 画图; 移植版输出文本统计 + ASCII 直方图.
    提供:
      - 策略间性能对比 (pairwise comparison)
      - 误差分布分析 (percentile + histogram)
      - 路由决策统计 (GPU vs CPU 比例随时间变化)
      - 收敛性检测 (窗口内方差 < 阈值)
    """
    
    def __init__(self, result=None):
        """初始化诊断工具."""
        self.result = result
        self._dbg = _dbg
    
    def strategy_comparison(self, panel_name: str = "predicted_vs_actual"):
        """策略间性能对比 — 输出各策略最终误差的排名."""
        if not self.result or panel_name not in self.result.panels:
            _dbg("DIAG", f"panel {panel_name} not found")
            return {}
        
        panel = self.result.panels[panel_name]
        rankings = {}
        for method_name, mr in panel.methods.items():
            if mr.mean:
                # 取最后 10% 的均值作为收敛后性能
                tail_start = max(0, len(mr.mean) - len(mr.mean) // 10)
                tail_mean = sum(mr.mean[tail_start:]) / max(1, len(mr.mean) - tail_start)
                rankings[method_name] = tail_mean
                _dbg("DIAG", f"  {method_name}: tail_mean={tail_mean:.4f}")
        
        # 按性能排序
        sorted_rankings = sorted(rankings.items(), key=lambda x: x[1])
        _dbg("DIAG", f"strategy ranking (best→worst):")
        for rank, (name, score) in enumerate(sorted_rankings, 1):
            _dbg("DIAG", f"  #{rank}: {name} = {score:.4f}")
        
        return dict(sorted_rankings)
    
    def error_histogram(self, errors, bins=10, width=40):
        """ASCII 误差直方图 — 改编自 PAR2QO diagram.py savePlot.
        
        原版用 matplotlib hist; 移植版用 ASCII art.
        """
        if not errors:
            return ""
        
        import math
        
        min_e = min(errors)
        max_e = max(errors)
        step = (max_e - min_e) / bins if max_e > min_e else 1.0
        
        counts = [0] * bins
        for e in errors:
            idx = min(int((e - min_e) / step), bins - 1) if step > 0 else 0
            counts[idx] += 1
        
        max_count = max(counts) if counts else 1
        lines = []
        for i, c in enumerate(counts):
            lo = min_e + i * step
            hi = lo + step
            bar_len = int(c / max_count * width) if max_count > 0 else 0
            bar = "█" * bar_len
            lines.append(f"  {lo:8.4f}-{hi:8.4f} | {bar} ({c})")
        
        header = f"Error Distribution (n={len(errors)}, bins={bins})"
        return header + "\n" + "\n".join(lines)
    
    def convergence_check(self, values, window=50, threshold=0.01):
        """检测时间序列是否已收敛.
        
        使用滑动窗口内的变异系数 (CV) 判断:
        CV < threshold → 已收敛.
        """
        if len(values) < window:
            _dbg("CONV", f"insufficient data ({len(values)} < {window})")
            return False
        
        import math
        tail = values[-window:]
        mean_v = sum(tail) / len(tail)
        if mean_v == 0:
            return True
        var_v = sum((v - mean_v) ** 2 for v in tail) / len(tail)
        cv = math.sqrt(var_v) / abs(mean_v)
        
        converged = cv < threshold
        _dbg("CONV", f"window={window} mean={mean_v:.4f} cv={cv:.6f} "
             f"threshold={threshold} converged={converged}")
        return converged
    
    def routing_distribution_over_time(self, decisions, window=100):
        """路由决策随时间的分布变化 — 滑动窗口 GPU 比例."""
        if not decisions:
            return []
        
        ratios = []
        for i in range(0, len(decisions), window):
            chunk = decisions[i:i+window]
            gpu_count = sum(1 for d in chunk if "gpu" in str(getattr(d, 'target_device', '')).lower())
            ratio = gpu_count / len(chunk)
            ratios.append((i, ratio))
            _dbg("ROUTE_DIST", f"  step={i}-{i+len(chunk)}: gpu_ratio={ratio:.3f}")
        
        return ratios


def _dump_workload_snapshot(wl, label=""):
    """打印 WorkloadConfig 完整快照到 stderr."""
    import sys
    print(f"╔══ WorkloadConfig [{label}] ══════════════════", file=sys.stderr)
    for k, v in wl.__dict__.items():
        val = str(v)[:100]
        print(f"║ {k}: {val}", file=sys.stderr)
    print(f"╚══════════════════════════════════════════════", file=sys.stderr, flush=True)


def benchmark_quick_test(steps=50, seeds=1):
    """快速冒烟测试 — 少量步数验证全链路.
    
    用于 CI 或实验前验证, 不需要完整 2000 步.
    改编自 PAR2QO utility.py 的 quick_test 模式.
    """
    _dbg("QUICK", f"quick test: steps={steps} seeds={seeds}")
    wl = WorkloadConfig(name="QuickTest", num_steps=steps, num_seeds=seeds)
    _dump_workload_snapshot(wl, "quick_test")
    out = run_benchmark(wl)
    
    diag = BenchmarkDiagnostics(out)
    rankings = diag.strategy_comparison()
    
    for pname, panel in out.panels.items():
        for mname, mr in panel.methods.items():
            n = len(mr.mean)
            converged = diag.convergence_check(mr.mean, window=min(10, n))
            _dbg("QUICK", f"  {mname}: {n} points, "
                 f"tail={mr.mean[-1]:.3f}ms, converged={converged}")
    
    return out
