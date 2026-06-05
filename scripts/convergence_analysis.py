#!/usr/bin/env python3
"""
scripts/convergence_analysis.py — 收敛分析 (M251-M260)

对 output/strategy_comparison.json 中每个策略的延迟时间序列做四维收敛分析:

  1. Welford滑窗 (窗口100) 在线 mean/std
     环形缓冲 + Welford增量更新，O(1)每步，无需保留全部历史。
     输出: sliding_mean[], sliding_std[]

  2. CUSUM 变点检测 (Cumulative Sum Control Chart)
     双侧CUSUM检测均值偏移: S⁺ₜ = max(0, S⁺ₜ₋₁ + (xₜ - μ₀ - k))
                            S⁻ₜ = min(0, S⁻ₜ₋₁ + (xₜ - μ₀ + k))
     k = 0.5σ (松弛), h = 4σ (阈值)
     输出: cusum_pos[], cusum_neg[], changepoints[]

  3. EWMA 平滑 (Exponentially Weighted Moving Average)
     Eₜ = α·xₜ + (1-α)·Eₜ₋₁, α=0.05 (长记忆平滑)
     输出: ewma[]

  4. 收敛步数 (CV < 5%)
     在 Welford 滑窗的 mean/std 上计算 CV = std/mean，
     首次连续50步 CV < 5% 的起始步即为收敛步数。
     输出: convergence_step (int | null), cv_at_convergence

作者: dylanyunlon <dogechat@163.com>
用法:
    python scripts/convergence_analysis.py
    python scripts/convergence_analysis.py --input output/strategy_comparison.json
    python scripts/convergence_analysis.py --window 100 --alpha 0.05 --cv-thresh 0.05
"""
import argparse
import json
import math
import os
import sys
from collections import deque
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════
# 1. Welford 滑窗在线 mean/std
# ═══════════════════════════════════════════════════════════════

class WelfordSlidingWindow:
    """Welford增量算法 + 环形缓冲滑窗。

    维护固定窗口内的在线 mean/std，每步 O(1)：
    - 新值进入窗口:  Welford add
    - 旧值离开窗口:  Welford remove (逆更新)

    逆更新公式 (remove oldest x_old):
        n     ← n - 1
        delta ← x_old - mean
        mean  ← mean - delta / n
        M2    ← M2 - delta * (x_old - mean)

    数值稳定性: 窗口固定100, 不会积累大漂移; 若M2因浮点
    误差变为微小负值, 截断为0。
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._buf: deque = deque()
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0

    def _add(self, x: float) -> None:
        """Welford在线更新 — 添加一个样本。"""
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._m2 += delta * delta2

    def _remove(self, x: float) -> None:
        """Welford逆更新 — 移除最旧样本。"""
        if self._n <= 1:
            self._n = 0
            self._mean = 0.0
            self._m2 = 0.0
            return
        delta = x - self._mean
        self._mean = (self._mean * self._n - x) / (self._n - 1)
        delta2 = x - self._mean
        self._m2 -= delta * delta2
        self._n -= 1
        # 浮点截断保护
        if self._m2 < 0.0:
            self._m2 = 0.0

    def push(self, x: float) -> Tuple[float, float]:
        """压入新值, 返回当前窗口 (mean, std)。"""
        self._buf.append(x)
        self._add(x)
        if len(self._buf) > self.window_size:
            old = self._buf.popleft()
            self._remove(old)
        return self.mean, self.std

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        if self._n < 2:
            return 0.0
        return math.sqrt(self._m2 / (self._n - 1))


# ═══════════════════════════════════════════════════════════════
# 2. CUSUM 变点检测
# ═══════════════════════════════════════════════════════════════

def cusum_detect(
    series: List[float],
    k_factor: float = 0.5,
    h_factor: float = 4.0,
) -> Dict[str, Any]:
    """双侧CUSUM变点检测。

    参数:
        series:   时间序列
        k_factor: 松弛参数 k = k_factor × σ
        h_factor: 决策阈值 h = h_factor × σ

    算法:
        μ₀ = 全序列均值 (基线)
        σ  = 全序列标准差
        k  = k_factor × σ     (允许的自然波动幅度)
        h  = h_factor × σ     (触发报警的累积偏移)

        S⁺ₜ = max(0, S⁺ₜ₋₁ + (xₜ - μ₀) - k)   检测向上偏移
        S⁻ₜ = min(0, S⁻ₜ₋₁ + (xₜ - μ₀) + k)   检测向下偏移

        当 S⁺ > h 或 |S⁻| > h 时, 记录变点并重置。

    返回:
        cusum_pos[]:    正向累积和
        cusum_neg[]:    负向累积和
        changepoints[]: 变点列表 [{step, direction, magnitude}]
    """
    n = len(series)
    if n < 2:
        return {
            "cusum_pos": list(series),
            "cusum_neg": [0.0] * n,
            "changepoints": [],
            "mu0": 0.0,
            "sigma": 0.0,
        }

    # 基线统计 — Welford单pass
    w_mean = 0.0
    w_m2 = 0.0
    for i, x in enumerate(series):
        d = x - w_mean
        w_mean += d / (i + 1)
        w_m2 += d * (x - w_mean)
    mu0 = w_mean
    sigma = math.sqrt(w_m2 / (n - 1)) if n > 1 else 0.0

    k = k_factor * sigma
    h = h_factor * sigma

    s_pos = 0.0
    s_neg = 0.0
    cusum_pos_list = []
    cusum_neg_list = []
    changepoints = []

    for t, x in enumerate(series):
        z = x - mu0
        s_pos = max(0.0, s_pos + z - k)
        s_neg = min(0.0, s_neg + z + k)
        cusum_pos_list.append(round(s_pos, 6))
        cusum_neg_list.append(round(s_neg, 6))

        if s_pos > h:
            changepoints.append({
                "step": t,
                "direction": "up",
                "magnitude": round(s_pos, 4),
            })
            s_pos = 0.0  # 重置
        elif abs(s_neg) > h:
            changepoints.append({
                "step": t,
                "direction": "down",
                "magnitude": round(abs(s_neg), 4),
            })
            s_neg = 0.0  # 重置

    return {
        "cusum_pos": cusum_pos_list,
        "cusum_neg": cusum_neg_list,
        "changepoints": changepoints,
        "mu0": round(mu0, 4),
        "sigma": round(sigma, 4),
        "k": round(k, 4),
        "h": round(h, 4),
    }


# ═══════════════════════════════════════════════════════════════
# 3. EWMA 平滑
# ═══════════════════════════════════════════════════════════════

def ewma_smooth(series: List[float], alpha: float = 0.05) -> List[float]:
    """指数加权移动平均 EWMA。

    Eₜ = α · xₜ + (1 - α) · Eₜ₋₁
    α = 0.05 → 有效窗口 ≈ 2/α - 1 = 39步, 长记忆平滑。
    E₀ = x₀ (初始化为首个观测值)。
    """
    if not series:
        return []
    ewma_vals = [series[0]]
    for x in series[1:]:
        ewma_vals.append(alpha * x + (1.0 - alpha) * ewma_vals[-1])
    return [round(v, 6) for v in ewma_vals]


# ═══════════════════════════════════════════════════════════════
# 4. 收敛步数检测 (CV < 5%, 连续50步确认)
# ═══════════════════════════════════════════════════════════════

def find_convergence_step(
    sliding_means: List[float],
    sliding_stds: List[float],
    cv_threshold: float = 0.05,
    confirm_steps: int = 50,
) -> Tuple[Optional[int], Optional[float]]:
    """找到首次连续 confirm_steps 步 CV < cv_threshold 的起始步。

    CV = std / mean (变异系数)
    要求连续50步都满足 CV < 5%, 避免单点波动误判。

    返回: (convergence_step, cv_at_convergence)
           若未收敛: (None, None)
    """
    n = len(sliding_means)
    if n < confirm_steps:
        return None, None

    run_start = None
    run_length = 0

    for i in range(n):
        mean_i = sliding_means[i]
        std_i = sliding_stds[i]

        if abs(mean_i) < 1e-12:
            cv = 0.0
        else:
            cv = std_i / abs(mean_i)

        if cv < cv_threshold:
            if run_start is None:
                run_start = i
                run_length = 1
            else:
                run_length += 1

            if run_length >= confirm_steps:
                # 收敛! 返回连续段的起始步
                conv_cv = std_i / abs(mean_i) if abs(mean_i) > 1e-12 else 0.0
                return run_start, round(conv_cv, 6)
        else:
            run_start = None
            run_length = 0

    return None, None


# ═══════════════════════════════════════════════════════════════
# 主分析流程
# ═══════════════════════════════════════════════════════════════

def analyze_strategy(
    name: str,
    step_mean: List[float],
    window: int,
    alpha: float,
    cv_threshold: float,
) -> Dict[str, Any]:
    """对单个策略的 step_mean 时间序列执行完整收敛分析。"""

    # --- 1. Welford 滑窗 ---
    wsw = WelfordSlidingWindow(window_size=window)
    sliding_means = []
    sliding_stds = []
    for x in step_mean:
        m, s = wsw.push(x)
        sliding_means.append(round(m, 6))
        sliding_stds.append(round(s, 6))

    # --- 2. CUSUM 变点检测 ---
    cusum_result = cusum_detect(step_mean)

    # --- 3. EWMA 平滑 ---
    ewma_vals = ewma_smooth(step_mean, alpha=alpha)

    # --- 4. 收敛步数 ---
    conv_step, conv_cv = find_convergence_step(
        sliding_means, sliding_stds,
        cv_threshold=cv_threshold,
    )

    # --- 滑窗 CV 序列 (供下游可视化) ---
    cv_series = []
    for m, s in zip(sliding_means, sliding_stds):
        if abs(m) < 1e-12:
            cv_series.append(0.0)
        else:
            cv_series.append(round(s / abs(m), 6))

    return {
        "welford_sliding": {
            "window_size": window,
            "mean": sliding_means,
            "std": sliding_stds,
            "cv": cv_series,
        },
        "cusum": {
            "mu0": cusum_result["mu0"],
            "sigma": cusum_result["sigma"],
            "k": cusum_result["k"],
            "h": cusum_result["h"],
            "cusum_pos": cusum_result["cusum_pos"],
            "cusum_neg": cusum_result["cusum_neg"],
            "changepoints": cusum_result["changepoints"],
            "n_changepoints": len(cusum_result["changepoints"]),
        },
        "ewma": {
            "alpha": alpha,
            "values": ewma_vals,
        },
        "convergence": {
            "cv_threshold": cv_threshold,
            "confirm_steps": 50,
            "converged": conv_step is not None,
            "convergence_step": conv_step,
            "cv_at_convergence": conv_cv,
        },
    }


def run_convergence_analysis(
    input_path: str,
    output_path: str,
    window: int = 100,
    alpha: float = 0.05,
    cv_threshold: float = 0.05,
) -> Dict[str, Any]:
    """读取 strategy_comparison.json，执行全策略收敛分析。"""

    with open(input_path, "r") as f:
        data = json.load(f)

    methods = data.get("methods", {})
    if not methods:
        print("ERROR: no 'methods' in input JSON", file=sys.stderr)
        sys.exit(1)

    print(f"┌─{'─'*60}", file=sys.stderr)
    print(f"│ CONVERGENCE ANALYSIS", file=sys.stderr)
    print(f"├─{'─'*60}", file=sys.stderr)
    print(f"  │ input:        {input_path}", file=sys.stderr)
    print(f"  │ window:       {window}", file=sys.stderr)
    print(f"  │ ewma_alpha:   {alpha}", file=sys.stderr)
    print(f"  │ cv_threshold: {cv_threshold}", file=sys.stderr)
    print(f"  │ strategies:   {len(methods)}", file=sys.stderr)

    results = {}
    summary_table = []

    for strat_name, strat_data in methods.items():
        step_mean = strat_data.get("step_mean", [])
        if not step_mean:
            print(f"  │ SKIP {strat_name}: no step_mean data", file=sys.stderr)
            continue

        analysis = analyze_strategy(
            strat_name, step_mean,
            window=window, alpha=alpha, cv_threshold=cv_threshold,
        )
        results[strat_name] = analysis

        # 摘要行
        conv = analysis["convergence"]
        cusum = analysis["cusum"]
        conv_str = (f"step {conv['convergence_step']}"
                    if conv["converged"] else "NOT converged")
        summary_table.append({
            "strategy": strat_name,
            "converged": conv["converged"],
            "convergence_step": conv["convergence_step"],
            "cv_at_convergence": conv["cv_at_convergence"],
            "n_changepoints": cusum["n_changepoints"],
            "cusum_mu0": cusum["mu0"],
            "cusum_sigma": cusum["sigma"],
        })

        print(f"  │ {strat_name:20s} → {conv_str}, "
              f"CUSUM变点={cusum['n_changepoints']}, "
              f"σ={cusum['sigma']:.1f}",
              file=sys.stderr)

    # 构建输出
    output = {
        "metadata": {
            "panel": "Convergence Analysis — Welford滑窗+CUSUM变点+EWMA",
            "source": input_path,
            "parameters": {
                "welford_window": window,
                "ewma_alpha": alpha,
                "cv_threshold": cv_threshold,
                "confirm_steps": 50,
                "cusum_k_factor": 0.5,
                "cusum_h_factor": 4.0,
            },
            "algorithms": {
                "welford_sliding": (
                    "O(1)环形缓冲Welford增量: "
                    "add时delta更新mean/M2, remove时逆delta更新"
                ),
                "cusum": (
                    "双侧CUSUM: S⁺=max(0,S⁺+(x-μ₀)-k), "
                    "S⁻=min(0,S⁻+(x-μ₀)+k), |S|>h时报变点"
                ),
                "ewma": "Eₜ=α·xₜ+(1-α)·Eₜ₋₁, α=0.05→有效窗口≈39步",
                "convergence": (
                    "滑窗CV=std/mean, 首次连续50步CV<5%即为收敛步"
                ),
            },
            "n_strategies": len(results),
        },
        "strategies": results,
        "summary": summary_table,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"├─{'─'*60}", file=sys.stderr)
    print(f"└─ output: {output_path} "
          f"({os.path.getsize(output_path) / 1024:.0f} KB)",
          file=sys.stderr)
    print(file=sys.stderr)

    return output


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Lynceus Convergence Analysis (M251-M260)")
    parser.add_argument(
        "--input", default="output/strategy_comparison.json",
        help="输入的strategy_comparison.json路径")
    parser.add_argument(
        "--output", default="output/convergence_analysis.json",
        help="输出的convergence_analysis.json路径")
    parser.add_argument(
        "--window", type=int, default=100,
        help="Welford滑窗大小 (默认100)")
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="EWMA平滑系数 (默认0.05)")
    parser.add_argument(
        "--cv-thresh", type=float, default=0.05,
        help="收敛CV阈值 (默认0.05即5%%)")
    args = parser.parse_args()

    run_convergence_analysis(
        args.input, args.output,
        window=args.window, alpha=args.alpha, cv_threshold=args.cv_thresh,
    )


if __name__ == "__main__":
    main()
