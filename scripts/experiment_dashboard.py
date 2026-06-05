#!/usr/bin/env python3
"""
scripts/experiment_dashboard.py — ASCII 实验仪表盘 (全景汇总)

M271-M280 交付。
作者: dylanyunlon <dogechat@163.com>

读取 output/ 下所有 JSON 文件，在终端打印统一仪表盘:
  ┌ Panel 1 ┐  策略排名 + 进度条   (strategy_comparison.json)
  ├ Panel 2 ┤  消融效应量           (ablation_results.json)
  ├ Panel 3 ┤  多负载对比           (multi_workload.json)
  ├ Panel 4 ┤  延迟 sparkline       (latency_vs_step.json)
  └ Panel 5 ┘  累积延迟 sparkline   (cumulative_latency.json)

缺失的 JSON 自动跳过该面板。

用法:
    python scripts/experiment_dashboard.py                     # 默认 output/
    python scripts/experiment_dashboard.py --output-dir mydir  # 自定义目录
"""
import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════

SPARK_CHARS = "▁▂▃▄▅▆▇█"
BOARD_WIDTH = 78  # 面板内容宽度
BAR_MAX_LEN = 30  # 进度条最大长度

# ANSI 颜色 (仅在 tty 下使用)
_IS_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    """条件着色: 仅在 tty 时添加 ANSI 转义。"""
    if not _IS_TTY:
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(t: str) -> str:
    return _c("1", t)


def _green(t: str) -> str:
    return _c("32", t)


def _yellow(t: str) -> str:
    return _c("33", t)


def _red(t: str) -> str:
    return _c("31", t)


def _cyan(t: str) -> str:
    return _c("36", t)


def _dim(t: str) -> str:
    return _c("2", t)


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def load_json(path: str) -> Optional[Dict]:
    """安全加载 JSON, 失败返回 None."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def sparkline(values: List[float], width: int = 40) -> str:
    """将数值列表转换为定宽 sparkline 字符串。

    算法: 等距采样 → 映射到 SPARK_CHARS 的 8 级。
    """
    if not values:
        return ""
    # 降采样到 width 个点
    n = len(values)
    if n <= width:
        sampled = values
    else:
        sampled = [values[int(i * (n - 1) / (width - 1))] for i in range(width)]

    lo = min(sampled)
    hi = max(sampled)
    rng = hi - lo if hi != lo else 1.0

    chars = []
    for v in sampled:
        idx = int((v - lo) / rng * (len(SPARK_CHARS) - 1))
        idx = max(0, min(idx, len(SPARK_CHARS) - 1))
        chars.append(SPARK_CHARS[idx])
    return "".join(chars)


def progress_bar(ratio: float, width: int = BAR_MAX_LEN) -> str:
    """生成 ASCII 进度条 ████░░░░░."""
    ratio = max(0.0, min(1.0, ratio))
    filled = int(round(ratio * width))
    return "█" * filled + "░" * (width - filled)


def format_us(val: float) -> str:
    """格式化微秒值。"""
    if abs(val) >= 1e6:
        return f"{val / 1e6:.1f}s"
    if abs(val) >= 1e3:
        return f"{val / 1e3:.1f}ms"
    return f"{val:.1f}µs"


def effect_color(label: str) -> str:
    """根据效应量等级着色。"""
    label_lower = label.lower()
    if "large" in label_lower:
        return _red(label)
    if "medium" in label_lower:
        return _yellow(label)
    if "small" in label_lower:
        return _green(label)
    return _dim(label)


# ═══════════════════════════════════════════════════════════════
# 面板渲染
# ═══════════════════════════════════════════════════════════════

def box_top(title: str) -> str:
    pad = BOARD_WIDTH - len(title) - 4
    return f"╔═ {_bold(title)} {'═' * max(pad, 1)}╗"


def box_mid() -> str:
    return f"╠{'═' * BOARD_WIDTH}╣"


def box_bot() -> str:
    return f"╚{'═' * BOARD_WIDTH}╝"


def box_line(text: str) -> str:
    """左对齐填充到 BOARD_WIDTH, 补右边框。"""
    # 计算纯文本长度 (去掉 ANSI 转义)
    import re
    plain = re.sub(r"\033\[[0-9;]*m", "", text)
    padding = BOARD_WIDTH - len(plain) - 2
    if padding < 0:
        padding = 0
    return f"║ {text}{' ' * padding}║"


def box_empty() -> str:
    return box_line("")


# ───────────────────────────────────────────────────────────────
# Panel 1: 策略排名 + 进度条
# ───────────────────────────────────────────────────────────────

def panel_strategy_ranking(data: Dict) -> List[str]:
    """策略排名面板: 名称 | 进度条 | 延迟均值 | P95 | 熵。"""
    lines: List[str] = []
    lines.append(box_top("PANEL 1 · Strategy Ranking"))

    methods = data.get("methods", {})
    ranking = data.get("ranking", list(methods.keys()))

    if not methods:
        lines.append(box_line(_dim("  (no strategy data)")))
        lines.append(box_bot())
        return lines

    # 计算归一化: 以最优策略的 grand_mean 为基准
    means = {s: methods[s].get("grand_mean", 0) for s in ranking}
    best_mean = min(means.values()) if means else 1.0
    worst_mean = max(means.values()) if means else 1.0

    # 表头
    header = f"  {'#':>2} {'Strategy':<22} {'Bar':<{BAR_MAX_LEN+2}} {'Mean':>9} {'P95':>9} {'H':>6}"
    lines.append(box_line(_dim(header)))
    lines.append(box_line(_dim("  " + "─" * 72)))

    for rank, sname in enumerate(ranking, 1):
        m = methods[sname]
        gm = m.get("grand_mean", 0)
        p95 = m.get("p95", 0)
        entropy = m.get("shannon_entropy", 0)

        # 进度条: 最优=满, 最差=短
        if worst_mean > best_mean:
            ratio = 1.0 - (gm - best_mean) / (worst_mean - best_mean)
        else:
            ratio = 1.0
        bar = progress_bar(ratio)

        # 颜色标记
        if rank == 1:
            tag = _green(f"{'★ ' + sname:<22}")
        elif rank <= 3:
            tag = _cyan(f"  {sname:<20}")
        else:
            tag = f"  {sname:<20}"

        row = f"  {rank:>2} {tag} {bar}  {format_us(gm):>9} {format_us(p95):>9} {entropy:>6.3f}"
        lines.append(box_line(row))

    # 元数据
    meta = data.get("metadata", {})
    lines.append(box_empty())
    lines.append(box_line(_dim(
        f"  steps={meta.get('n_steps', '?')}  "
        f"seeds={meta.get('n_seeds', '?')}  "
        f"queries={meta.get('total_queries', '?')}  "
        f"elapsed={meta.get('elapsed_seconds', '?')}s"
    )))
    lines.append(box_bot())
    return lines


# ───────────────────────────────────────────────────────────────
# Panel 2: 消融效应量
# ───────────────────────────────────────────────────────────────

def panel_ablation(data: Dict) -> List[str]:
    """消融面板: 名称 | Cohen's d 条 | Δ% | p-value。"""
    lines: List[str] = []
    lines.append(box_top("PANEL 2 · Ablation Effect Sizes"))

    ablations = data.get("ablations", [])
    if not ablations:
        lines.append(box_line(_dim("  (no ablation data)")))
        lines.append(box_bot())
        return lines

    # 按 |d| 降序排列
    sorted_ab = sorted(ablations, key=lambda a: abs(a.get("cohens_d", 0)), reverse=True)

    max_d = max(abs(a.get("cohens_d", 0)) for a in sorted_ab) or 1.0

    header = f"  {'Ablation':<35} {'|d|':>6} {'Bar':<16} {'Δ%':>7} {'p':>8} {'Effect':<10}"
    lines.append(box_line(_dim(header)))
    lines.append(box_line(_dim("  " + "─" * 72)))

    for a in sorted_ab:
        name = a.get("name", "?")[:33]
        d = a.get("cohens_d", 0)
        delta_pct = a.get("delta_pct", 0)
        p_val = a.get("wilcoxon_p", 1.0)
        label = a.get("effect_label", "?")

        # 效应量进度条 (14 宽)
        ratio = min(abs(d) / max_d, 1.0)
        bar_len = 14
        filled = int(round(ratio * bar_len))
        if d >= 0:
            bar = _green("▓" * filled) + "░" * (bar_len - filled)
        else:
            bar = _red("▓" * filled) + "░" * (bar_len - filled)

        sig = "***" if p_val < 0.001 else "** " if p_val < 0.01 else "*  " if p_val < 0.05 else "   "

        row = (
            f"  {name:<35} {abs(d):>6.3f} {bar} {delta_pct:>+6.1f}% "
            f"{p_val:>7.4f}{sig} {effect_color(label)}"
        )
        lines.append(box_line(row))

    lines.append(box_bot())
    return lines


# ───────────────────────────────────────────────────────────────
# Panel 3: 多负载对比
# ───────────────────────────────────────────────────────────────

def panel_multi_workload(data: Dict) -> List[str]:
    """多负载面板: 各负载 × 各策略终态延迟 + 路由分布。"""
    lines: List[str] = []
    lines.append(box_top("PANEL 3 · Multi-Workload Comparison"))

    meta = data.get("metadata", {})
    workloads = meta.get("workloads", [])
    strategies = meta.get("strategies", [])
    fig1 = data.get("fig1_data", {})  # latency curves
    fig3 = data.get("fig3_data", {})  # routing distribution

    if not workloads or not fig1:
        lines.append(box_line(_dim("  (no multi-workload data)")))
        lines.append(box_bot())
        return lines

    # 子面板 3a: 终态延迟热力图 (每负载 × 每策略)
    lines.append(box_line(_bold("  3a · Final Mean Latency (µs)")))
    lines.append(box_empty())

    # 收集终态延迟
    final_lats: Dict[str, Dict[str, float]] = {}
    global_max = 0.0
    for wl in workloads:
        final_lats[wl] = {}
        for st in strategies:
            curve = fig1.get(wl, {}).get(st, [])
            val = curve[-1] if curve else 0.0
            final_lats[wl][st] = val
            if val > global_max:
                global_max = val

    # 缩短策略名
    short_names = [s[:12] for s in strategies]
    col_w = 13

    # 表头
    hdr = f"  {'Workload':<10}" + "".join(f"{sn:>{col_w}}" for sn in short_names)
    lines.append(box_line(_dim(hdr)))
    lines.append(box_line(_dim("  " + "─" * 72)))

    for wl in workloads:
        cols = []
        for st in strategies:
            val = final_lats[wl][st]
            txt = format_us(val)
            # 找此负载最优
            wl_vals = list(final_lats[wl].values())
            wl_min = min(wl_vals) if wl_vals else 0
            if val == wl_min and val > 0:
                txt = _green(f"{txt:>{col_w}}")
            else:
                txt = f"{txt:>{col_w}}"
            cols.append(txt)
        row = f"  {wl:<10}" + "".join(cols)
        lines.append(box_line(row))

    # 子面板 3b: Sparkline 延迟趋势 (每负载取最优策略)
    lines.append(box_empty())
    lines.append(box_line(_bold("  3b · Latency Trend (best strategy per workload)")))
    lines.append(box_empty())

    for wl in workloads:
        # 找最优策略 (终态最低)
        best_st = min(strategies, key=lambda s: final_lats[wl].get(s, float("inf")))
        curve = fig1.get(wl, {}).get(best_st, [])
        spark = sparkline(curve, width=40) if curve else _dim("(no data)")
        lines.append(box_line(
            f"  {wl:<8} [{best_st:<16}] {spark} {format_us(curve[-1]) if curve else '?'}"
        ))

    # 子面板 3c: 路由分布
    if fig3:
        lines.append(box_empty())
        lines.append(box_line(_bold("  3c · Routing Distribution")))
        lines.append(box_empty())

        for wl in workloads:
            wl_routes = fig3.get(wl, {})
            if not wl_routes:
                continue
            lines.append(box_line(_dim(f"  [{wl}]")))
            for st in strategies:
                dist = wl_routes.get(st, {})
                if not dist:
                    continue
                parts = []
                for dev, pct in sorted(dist.items(), key=lambda x: -x[1]):
                    pct_val = pct * 100 if pct <= 1.0 else pct
                    bar_w = int(pct_val / 5)  # 每5%一个块
                    parts.append(f"{dev}={'█' * bar_w}{pct_val:.0f}%")
                route_str = "  ".join(parts)
                sname = st[:16]
                lines.append(box_line(f"    {sname:<16} {route_str}"))

    lines.append(box_bot())
    return lines


# ───────────────────────────────────────────────────────────────
# Panel 4: 延迟 Sparkline
# ───────────────────────────────────────────────────────────────

def panel_latency_sparkline(data: Dict) -> List[str]:
    """每步延迟 sparkline + 统计摘要。"""
    lines: List[str] = []
    lines.append(box_top("PANEL 4 · Latency vs Step"))

    mean_curve = data.get("mean", [])
    std_curve = data.get("std", [])
    seeds = data.get("seeds", {})
    meta = data.get("metadata", {})

    if not mean_curve:
        lines.append(box_line(_dim("  (no latency data)")))
        lines.append(box_bot())
        return lines

    n = len(mean_curve)
    final_mean = mean_curve[-1]
    final_std = std_curve[-1] if std_curve else 0

    # 总体 sparkline
    lines.append(box_line(f"  Mean  {sparkline(mean_curve, 50)}  {format_us(final_mean)}"))
    if std_curve:
        lines.append(box_line(f"  Std   {sparkline(std_curve, 50)}  {format_us(final_std)}"))

    # 各 seed sparkline
    if seeds:
        lines.append(box_empty())
        lines.append(box_line(_dim("  Per-seed traces:")))
        for sid in sorted(seeds.keys(), key=lambda x: int(x)):
            sv = seeds[sid]
            if not sv:
                continue
            sfinal = sv[-1] if sv else 0
            lines.append(box_line(
                f"  seed {sid:>2}  {sparkline(sv, 45)}  {format_us(sfinal)}"
            ))

    # 统计
    lines.append(box_empty())
    p50_idx = n // 2
    p95_idx = int(n * 0.95)
    sorted_mean = sorted(mean_curve)
    p50 = sorted_mean[min(p50_idx, n - 1)]
    p95 = sorted_mean[min(p95_idx, n - 1)]

    lines.append(box_line(_dim(
        f"  n={n}  P50={format_us(p50)}  P95={format_us(p95)}  "
        f"min={format_us(min(mean_curve))}  max={format_us(max(mean_curve))}"
    )))
    lines.append(box_bot())
    return lines


# ───────────────────────────────────────────────────────────────
# Panel 5: 累积延迟 Sparkline
# ───────────────────────────────────────────────────────────────

def panel_cumulative_sparkline(data: Dict) -> List[str]:
    """累积延迟 sparkline。"""
    lines: List[str] = []
    lines.append(box_top("PANEL 5 · Cumulative Latency"))

    cum = data.get("cumulative_mean", [])
    meta = data.get("metadata", {})

    if not cum:
        lines.append(box_line(_dim("  (no cumulative data)")))
        lines.append(box_bot())
        return lines

    final = cum[-1]
    lines.append(box_line(f"  Cumul  {sparkline(cum, 50)}  {format_us(final)}"))

    # 增长率 sparkline (每步增量)
    if len(cum) > 1:
        deltas = [cum[i] - cum[i - 1] for i in range(1, len(cum))]
        lines.append(box_line(f"  Delta  {sparkline(deltas, 50)}  {format_us(deltas[-1])}/step"))

    lines.append(box_empty())
    lines.append(box_line(_dim(
        f"  steps={len(cum)}  final={format_us(final)}  "
        f"growth={format_us(final / len(cum))}/step avg"
    )))
    lines.append(box_bot())
    return lines


# ═══════════════════════════════════════════════════════════════
# 大标题横幅
# ═══════════════════════════════════════════════════════════════

def banner() -> List[str]:
    b = [
        "",
        _bold("  ╦  ╦ ╔╗╔ ╔═╗ ╔═╗ ╦ ╦ ╔═╗   ┌──────────────────────────────────┐"),
        _bold("  ║  ║ ║║║ ║   ╠╣  ║ ║ ╚═╗   │  Experiment Dashboard  (v1.0)   │"),
        _bold("  ╩═╝╩ ╝╚╝ ╚═╝ ╚═╝ ╚═╝ ╚═╝   └──────────────────────────────────┘"),
        "",
    ]
    return b


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ASCII Experiment Dashboard — 全景汇总"
    )
    parser.add_argument("--output-dir", default="output",
                        help="JSON 输出目录 (default: output)")
    args = parser.parse_args()

    od = args.output_dir

    # 定义所有面板: (文件名, 渲染函数)
    panels = [
        ("strategy_comparison.json", panel_strategy_ranking),
        ("ablation_results.json", panel_ablation),
        ("multi_workload.json", panel_multi_workload),
        ("latency_vs_step.json", panel_latency_sparkline),
        ("cumulative_latency.json", panel_cumulative_sparkline),
    ]

    # 尝试加载所有 JSON
    loaded = 0
    skipped = 0

    all_lines: List[str] = banner()

    for fname, render_fn in panels:
        path = os.path.join(od, fname)
        data = load_json(path)
        if data is None:
            skipped += 1
            all_lines.append(_dim(f"  ⊘ {fname} — not found, skipping"))
            all_lines.append("")
            continue

        loaded += 1
        try:
            panel_lines = render_fn(data)
            all_lines.extend(panel_lines)
            all_lines.append("")  # 面板间留白
        except Exception as e:
            all_lines.append(_red(f"  ✗ {fname} — render error: {e}"))
            all_lines.append("")

    # 尾注
    all_lines.append(_dim(f"  ── dashboard complete: {loaded} panels rendered, {skipped} skipped ──"))
    all_lines.append("")

    # 一次性输出
    print("\n".join(all_lines))


if __name__ == "__main__":
    main()
