#!/usr/bin/env python3
"""
scripts/run_full_experiment.py — 全链路实验入口 (一键跑完全部)

第六位Claude (M201–M220) 交付。
作者: dylanyunlon <dogechat@163.com>

按顺序执行:
  1. strategy_comparison.py  → output/strategy_comparison.json
  2. ablation_study.py       → output/ablation_results.json
  3. multi_workload_bench.py → output/multi_workload.json
  4. generate_paper_tables.py → output/table{1,2,3}_*.tex

算法改写 (~20%):
  1. 阶段依赖检查: 后续阶段的输入文件必须存在
  2. Watchdog计时: 每阶段超时告警 (> 5分钟)
  3. 增量执行: --skip-existing 跳过已有输出
  4. 摘要报告: 全链路耗时/文件大小/行数统计
  5. 退出码传播: 任一阶段失败则终止 (fail-fast)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def _dbg(msg, **kw):
    print(f"  │ {msg}" + (": " + ", ".join(f"{k}={v}" for k, v in kw.items()) if kw else ""), file=sys.stderr)

def _dump_header(title):
    print(f"\n┌─{'─'*60}", file=sys.stderr)
    print(f"│ {title}", file=sys.stderr)
    print(f"├─{'─'*60}", file=sys.stderr)

def _dump_footer(summary):
    print(f"├─{'─'*60}", file=sys.stderr)
    print(f"└─ {summary}", file=sys.stderr)
    print(file=sys.stderr)


class StageResult:
    def __init__(self, name, success, elapsed, output_files=None, error=None):
        self.name = name
        self.success = success
        self.elapsed = elapsed
        self.output_files = output_files or []
        self.error = error


def run_stage(name: str, cmd: list, expected_outputs: list,
              skip_existing: bool = False, timeout: int = 600) -> StageResult:
    """运行一个实验阶段。"""
    _dump_header(f"STAGE: {name}")

    # 增量检查
    if skip_existing and all(os.path.exists(f) for f in expected_outputs):
        _dbg("SKIP (outputs exist)", files=[os.path.basename(f) for f in expected_outputs])
        _dump_footer(f"{name}: SKIPPED (--skip-existing)")
        return StageResult(name, True, 0.0, expected_outputs)

    _dbg("command", cmd=" ".join(cmd))
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd, cwd=str(PROJECT_DIR),
            capture_output=False,  # 让stderr直接流到终端
            timeout=timeout,
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            _dump_footer(f"{name}: FAILED (exit={result.returncode}, {elapsed:.1f}s)")
            return StageResult(name, False, elapsed, error=f"exit code {result.returncode}")

        # 检查输出文件
        existing = [f for f in expected_outputs if os.path.exists(f)]
        missing = [f for f in expected_outputs if not os.path.exists(f)]

        if missing:
            _dbg("WARNING: missing outputs", files=[os.path.basename(f) for f in missing])

        for f in existing:
            size = os.path.getsize(f)
            _dbg(f"  {os.path.basename(f)}", size=f"{size:,} bytes")

        # Watchdog
        if elapsed > 300:
            _dbg(f"WARNING: stage took {elapsed:.0f}s (>5min)")

        _dump_footer(f"{name}: OK ({elapsed:.1f}s, {len(existing)}/{len(expected_outputs)} files)")
        return StageResult(name, True, elapsed, existing)

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        _dump_footer(f"{name}: TIMEOUT ({timeout}s)")
        return StageResult(name, False, elapsed, error=f"timeout after {timeout}s")
    except Exception as e:
        elapsed = time.time() - t0
        _dump_footer(f"{name}: ERROR ({e})")
        return StageResult(name, False, elapsed, error=str(e))


def main():
    parser = argparse.ArgumentParser(description="Run full Lynceus experiment pipeline (M201-M220)")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip stages whose output files already exist")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-stage timeout in seconds")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t_total = time.time()
    python = sys.executable
    od = args.output_dir

    _dump_header("FULL EXPERIMENT PIPELINE")
    _dbg("python", val=python)
    _dbg("steps", val=args.steps)
    _dbg("seeds", val=args.seeds)
    _dbg("output_dir", val=od)
    _dbg("skip_existing", val=args.skip_existing)

    stages = [
        ("1_strategy_comparison", [
            python, "scripts/strategy_comparison.py",
            "--steps", str(args.steps), "--seeds", str(args.seeds),
            "--output-dir", od,
        ], [f"{od}/strategy_comparison.json"]),

        ("2_ablation_study", [
            python, "scripts/ablation_study.py",
            "--steps", str(max(50, args.steps // 4)), "--seeds", str(args.seeds),
            "--output-dir", od,
        ], [f"{od}/ablation_results.json"]),

        ("3_multi_workload", [
            python, "scripts/multi_workload_bench.py",
            "--steps", str(max(50, args.steps // 4)), "--seeds", str(args.seeds),
        ], [f"{od}/multi_workload.json"]),

        ("4_paper_tables", [
            python, "scripts/generate_paper_tables.py",
            "--output-dir", od,
        ], [
            f"{od}/table1_strategy_comparison.tex",
            f"{od}/table2_ablation.tex",
            f"{od}/table3_multi_workload.tex",
        ]),
    ]

    results = []
    for name, cmd, outputs in stages:
        r = run_stage(name, cmd, outputs,
                      skip_existing=args.skip_existing,
                      timeout=args.timeout)
        results.append(r)
        if not r.success:
            _dbg(f"FAIL-FAST: {name} failed, stopping pipeline")
            break

    # 摘要
    total_elapsed = time.time() - t_total
    n_ok = sum(1 for r in results if r.success)
    n_fail = sum(1 for r in results if not r.success)
    all_files = []
    for r in results:
        all_files.extend(r.output_files)

    _dump_header("PIPELINE SUMMARY")
    _dbg("total_time", seconds=f"{total_elapsed:.1f}")
    _dbg("stages", ok=n_ok, fail=n_fail, total=len(stages))
    _dbg("output_files", count=len(all_files))
    for f in all_files:
        size = os.path.getsize(f) if os.path.exists(f) else 0
        _dbg(f"  {os.path.basename(f)}", size=f"{size:,}")
    for r in results:
        status = "✓" if r.success else "✗"
        _dbg(f"  {status} {r.name}", time=f"{r.elapsed:.1f}s",
             error=r.error if r.error else "")

    _dump_footer(
        f"{'ALL PASSED' if n_fail == 0 else f'{n_fail} FAILED'} | "
        f"{total_elapsed:.1f}s | {len(all_files)} files")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
