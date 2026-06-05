#!/usr/bin/env python3
"""
scripts/generate_paper_tables.py — 论文 LaTeX 表格 + ASCII 终端预览

第五位Claude (M181–M200) 交付。
作者: dylanyunlon <dogechat@163.com>

读取实验JSON → 输出 LaTeX .tex 表格文件 + stderr ASCII 预览

算法改写 (~20%):
  1. 自动量级: 检测µs/ms/s自动选择单位
  2. LaTeX高亮: \\textbf{} 标记每列最优值
  3. 排名标注: #1/#2/...
  4. Sparkline: ASCII ▁▂▃▄▅▆▇█ 8级mini趋势
  5. metadata: git hash + 时间戳
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════
# 调试
# ═══════════════════════════════════════════════════════════════

def _dbg(msg, **kw):
    print(f"  │ {msg}" + (": " + ", ".join(f"{k}={v}" for k, v in kw.items()) if kw else ""), file=sys.stderr)

def _dump_header(title):
    print(f"\n┌─{'─'*60}", file=sys.stderr)
    print(f"│ STATE DUMP: {title}", file=sys.stderr)
    print(f"├─{'─'*60}", file=sys.stderr)

def _dump_footer(summary):
    print(f"├─{'─'*60}", file=sys.stderr)
    print(f"└─ {summary}", file=sys.stderr)
    print(file=sys.stderr)


# ═══════════════════════════════════════════════════════════════
# 算法改写 #1: 自动量级格式化
# ═══════════════════════════════════════════════════════════════

def fmt_latency(us: float) -> str:
    """自动检测量级: µs → ms → s."""
    if us < 1.0:
        return f"{us:.3f}µs"
    elif us < 1000.0:
        return f"{us:.1f}µs"
    elif us < 1_000_000.0:
        return f"{us/1000:.2f}ms"
    else:
        return f"{us/1_000_000:.3f}s"


def fmt_pct(v: float) -> str:
    return f"{v:.1f}\\%" if abs(v) < 100 else f"{v:.0f}\\%"


# ═══════════════════════════════════════════════════════════════
# 算法改写 #4: ASCII Sparkline
# ═══════════════════════════════════════════════════════════════

SPARK_CHARS = "▁▂▃▄▅▆▇█"

def sparkline(values: List[float], width: int = 20) -> str:
    """ASCII sparkline: 8级字符画mini趋势。

    把values下采样到width个点，每个点映射到 ▁▂▃▄▅▆▇█ 中一个。
    """
    if not values:
        return ""
    n = len(values)
    if n <= width:
        sampled = values
    else:
        step = n / width
        sampled = [values[int(i * step)] for i in range(width)]

    lo, hi = min(sampled), max(sampled)
    rng = hi - lo if hi > lo else 1.0
    chars = []
    for v in sampled:
        idx = int((v - lo) / rng * 7)
        idx = max(0, min(7, idx))
        chars.append(SPARK_CHARS[idx])
    return "".join(chars)


# ═══════════════════════════════════════════════════════════════
# 算法改写 #5: Metadata 收集
# ═══════════════════════════════════════════════════════════════

def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# ═══════════════════════════════════════════════════════════════
# Table 1: 策略对比 (strategy_comparison.json)
# ═══════════════════════════════════════════════════════════════

def generate_table1(data: Dict, output_dir: str) -> Optional[str]:
    methods = data.get("methods", {})
    if not methods:
        return None

    _dump_header("TABLE 1: Strategy Comparison")

    rows = []
    for sname, m in methods.items():
        rows.append({
            "name": sname,
            "mean": m["grand_mean"],
            "std": m["grand_std"],
            "p95": m["p95"],
            "entropy": m["shannon_entropy"],
            "cache": sum(m["seeds"][s]["cache_hit_rate"] for s in m["seeds"]) / max(1, len(m["seeds"])),
        })

    # 排名 (按mean)
    rows.sort(key=lambda r: r["mean"])
    for i, r in enumerate(rows):
        r["rank"] = i + 1

    # 找每列最优
    best_mean = min(r["mean"] for r in rows)
    best_p95 = min(r["p95"] for r in rows)
    best_entropy = max(r["entropy"] for r in rows)  # 高熵=多样

    def bold_if(val_str, val, best, minimize=True):
        if minimize and abs(val - best) < 1e-6:
            return f"\\textbf{{{val_str}}}"
        elif not minimize and abs(val - best) < 1e-6:
            return f"\\textbf{{{val_str}}}"
        return val_str

    # LaTeX
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Strategy Comparison — TPC-H SF100 ({data.get('metadata',{}).get('n_steps','?')} steps $\\times$ {data.get('metadata',{}).get('n_seeds','?')} seeds)}}",
        "\\label{tab:strategy-comparison}",
        "\\begin{tabular}{clrrrrr}",
        "\\toprule",
        "Rank & Strategy & Mean ($\\mu$s) & Std ($\\mu$s) & P95 ($\\mu$s) & Entropy & Cache Hit \\\\",
        "\\midrule",
    ]

    for r in rows:
        mean_s = bold_if(f"{r['mean']:.1f}", r["mean"], best_mean)
        p95_s = bold_if(f"{r['p95']:.1f}", r["p95"], best_p95)
        ent_s = bold_if(f"{r['entropy']:.3f}", r["entropy"], best_entropy, minimize=False)
        lines.append(
            f"  \\#{r['rank']} & {r['name']} & {mean_s} & {r['std']:.1f} & {p95_s} & {ent_s} & {r['cache']:.1%} \\\\"
        )

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    tex = "\n".join(lines)

    # ASCII 预览
    _dbg("ASCII preview:")
    hdr = f"{'Rank':>4} {'Strategy':>22} {'Mean(µs)':>10} {'Std':>8} {'P95':>10} {'H':>6} {'Cache':>7}"
    print(f"  │ {hdr}", file=sys.stderr)
    print(f"  │ {'─'*len(hdr)}", file=sys.stderr)
    for r in rows:
        line = f"  #{r['rank']:>2}  {r['name']:>22} {r['mean']:>10.1f} {r['std']:>8.1f} {r['p95']:>10.1f} {r['entropy']:>6.3f} {r['cache']:>6.1%}"
        print(f"  │ {line}", file=sys.stderr)

    path = os.path.join(output_dir, "table1_strategy_comparison.tex")
    with open(path, "w") as f:
        f.write(tex)
    _dump_footer(f"Wrote {path} ({len(tex)} bytes)")
    return path


# ═══════════════════════════════════════════════════════════════
# Table 2: 消融 (ablation_results.json)
# ═══════════════════════════════════════════════════════════════

def generate_table2(data: Dict, output_dir: str) -> Optional[str]:
    ablations = data.get("ablations", [])
    if not ablations:
        return None

    _dump_header("TABLE 2: Ablation Study")

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Ablation Study — Effect of Individual Algorithm Changes}",
        "\\label{tab:ablation}",
        "\\begin{tabular}{lrrrrrl}",
        "\\toprule",
        "Ablation & Baseline ($\\mu$s) & Ablated ($\\mu$s) & $\\Delta$ ($\\mu$s) & Cohen's $d$ & $p$-value & Effect \\\\",
        "\\midrule",
    ]

    for a in ablations:
        d_val = a.get("cohens_d", 0)
        d_str = f"{d_val:+.3f}"
        if abs(d_val) >= 0.8:
            d_str = f"\\textbf{{{d_str}}}"
        p_val = a.get("wilcoxon_p", 1.0)
        p_str = f"{p_val:.4f}" if p_val > 0.001 else f"$<$0.001"
        if p_val < 0.05:
            p_str = f"\\textbf{{{p_str}}}"

        name_clean = a["name"].replace("_", "\\_")
        lines.append(
            f"  {name_clean} & {a.get('baseline_mean',0):.1f} & {a.get('ablated_mean',0):.1f} "
            f"& {a.get('delta_mean',0):+.1f} & {d_str} & {p_str} & {a.get('effect_label','?')} \\\\"
        )

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tex = "\n".join(lines)

    # ASCII 预览
    _dbg("ASCII preview:")
    for a in ablations:
        _dbg(f"  {a['name'][:30]:>30}  base={a.get('baseline_mean',0):>8.1f}  abl={a.get('ablated_mean',0):>8.1f}  d={a.get('cohens_d',0):>+.3f}  {a.get('effect_label','')}")

    path = os.path.join(output_dir, "table2_ablation.tex")
    with open(path, "w") as f:
        f.write(tex)
    _dump_footer(f"Wrote {path}")
    return path


# ═══════════════════════════════════════════════════════════════
# Table 3: 多负载 (multi_workload.json)
# ═══════════════════════════════════════════════════════════════

def generate_table3(data: Dict, output_dir: str) -> Optional[str]:
    fig1 = data.get("fig1_data", {})
    if not fig1:
        return None

    _dump_header("TABLE 3: Multi-Workload Comparison")

    workloads = list(fig1.keys())
    strategies = list(fig1[workloads[0]].keys()) if workloads else []

    # 计算每个 workload×strategy 的 grand mean
    grand_means = {}
    for wl in workloads:
        grand_means[wl] = {}
        for st in strategies:
            val = fig1[wl].get(st, [])
            # fig1_data[wl][st] 可以是 list(直接step_mean) 或 dict(含step_mean键)
            if isinstance(val, dict):
                step_mean = val.get("step_mean", [])
            elif isinstance(val, list):
                step_mean = val
            else:
                step_mean = []
            grand_means[wl][st] = sum(step_mean) / len(step_mean) if step_mean else 0.0

    # 每个workload内的最优
    best_per_wl = {wl: min(grand_means[wl].values()) for wl in workloads}

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Mean Latency ($\\mu$s) Across Workloads and Strategies}",
        "\\label{tab:multi-workload}",
        "\\begin{tabular}{l" + "r" * len(workloads) + "}",
        "\\toprule",
        "Strategy & " + " & ".join(workloads) + " \\\\",
        "\\midrule",
    ]

    for st in strategies:
        cells = []
        for wl in workloads:
            val = grand_means[wl][st]
            s = f"{val:.1f}"
            if abs(val - best_per_wl[wl]) < 1e-6:
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        lines.append(f"  {st} & " + " & ".join(cells) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tex = "\n".join(lines)

    # ASCII 预览 + sparklines
    _dbg("ASCII preview:")
    hdr = f"{'Strategy':>22}" + "".join(f" {wl:>10}" for wl in workloads)
    print(f"  │ {hdr}", file=sys.stderr)
    for st in strategies:
        vals = [f"{grand_means[wl][st]:>10.1f}" for wl in workloads]
        print(f"  │ {st:>22}{''.join(vals)}", file=sys.stderr)

    # Sparklines per strategy across workloads
    _dbg("Sparklines (per strategy, across workloads):")
    for st in strategies:
        vals = [grand_means[wl][st] for wl in workloads]
        spark = sparkline(vals, width=len(workloads))
        _dbg(f"  {st:>22} {spark}")

    path = os.path.join(output_dir, "table3_multi_workload.tex")
    with open(path, "w") as f:
        f.write(tex)
    _dump_footer(f"Wrote {path}")
    return path


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from experiment JSON (M181-M200)")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()
    git_hash = get_git_hash()

    _dump_header("PAPER TABLE GENERATION")
    _dbg("git_hash", val=git_hash)
    _dbg("timestamp", val=time.strftime("%Y-%m-%d %H:%M:%S"))
    _dbg("output_dir", val=args.output_dir)

    generated = []

    # Table 1
    try:
        with open(os.path.join(args.output_dir, "strategy_comparison.json")) as f:
            sc_data = json.load(f)
        p = generate_table1(sc_data, args.output_dir)
        if p: generated.append(p)
    except FileNotFoundError:
        _dbg("SKIP table1: strategy_comparison.json not found")

    # Table 2
    try:
        with open(os.path.join(args.output_dir, "ablation_results.json")) as f:
            ab_data = json.load(f)
        p = generate_table2(ab_data, args.output_dir)
        if p: generated.append(p)
    except FileNotFoundError:
        _dbg("SKIP table2: ablation_results.json not found")

    # Table 3
    try:
        with open(os.path.join(args.output_dir, "multi_workload.json")) as f:
            mw_data = json.load(f)
        p = generate_table3(mw_data, args.output_dir)
        if p: generated.append(p)
    except FileNotFoundError:
        _dbg("SKIP table3: multi_workload.json not found")

    elapsed = time.time() - t0
    _dump_header("TABLE GENERATION COMPLETE")
    _dbg("elapsed", seconds=f"{elapsed:.1f}")
    _dbg("generated", count=len(generated))
    for p in generated:
        _dbg(f"  {p}")
    _dump_footer(f"{len(generated)} tables generated | git={git_hash}")


if __name__ == "__main__":
    main()
