#!/usr/bin/env bash
# ================================================================
#  scripts/run_and_push.sh — Run experiment, auto-push results to git
#  Usage: bash scripts/run_and_push.sh [quick|full]
#  Server: conda activate walking3 && bash scripts/run_and_push.sh
# ================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

MODE="${1:-quick}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p "$LOG_DIR" output

echo "==========================================="
echo "  Lynceus-CMD Experiment Runner"
echo "  Mode: $MODE | Time: $TIMESTAMP"
echo "==========================================="

# ── Git config ──
git config user.name "dylanyunlon"
git config user.email "dogechat@163.com"

# ── System info ──
echo ""
echo "=== System Info ==="
python3 --version 2>&1
uname -r 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU"
echo ""

# ── Run SOTA comparison ──
echo "=== Running SOTA Comparison ==="
if [ "$MODE" = "quick" ]; then
    python3 scripts/sota_comparison.py --quick 2>&1 | \
        grep -E "SNAPSHOT|PAPER TABLE|Method|------|GPU-Only|CPU-Only|Hybrid|CostModel|PAR2QO|Adaptive|Published|PostgreSQL|Bao|Neo|Balsa|Kendall|Saved" | \
        tee "$LOG_DIR/sota_${TIMESTAMP}.log"
else
    python3 scripts/sota_comparison.py 2>&1 | \
        grep -E "SNAPSHOT|PAPER TABLE|Method|------|GPU-Only|CPU-Only|Hybrid|CostModel|PAR2QO|Adaptive|Published|PostgreSQL|Bao|Neo|Balsa|Kendall|Saved" | \
        tee "$LOG_DIR/sota_${TIMESTAMP}.log"
fi

# ── Run main benchmark ──
echo ""
echo "=== Running Main Benchmark ==="
python3 -c "
import sys; sys.path.insert(0, '.')
import lynceus._debug as dbg; dbg.ENABLED = False
from lynceus.benchmark import run_benchmark, WorkloadConfig
steps = 200 if '$MODE' == 'quick' else 2000
seeds = 2 if '$MODE' == 'quick' else 3
wl = WorkloadConfig(name='TPC-H_SF100', num_steps=steps, num_seeds=seeds)
r = run_benchmark(wl)
r.save('output/latency_vs_step.json')
print(f'Benchmark complete: {steps} steps x {seeds} seeds')
print(f'Saved to output/latency_vs_step.json')
" 2>&1 | grep -v "^\[DBG\|^$\|^──\|CHECKPOINT" | tee -a "$LOG_DIR/bench_${TIMESTAMP}.log"

# ── Auto-push results ──
echo ""
echo "=== Pushing Results ==="
git add -f output/sota_comparison.json output/latency_vs_step.json 2>/dev/null || true
git add "$LOG_DIR/sota_${TIMESTAMP}.log" "$LOG_DIR/bench_${TIMESTAMP}.log" 2>/dev/null || true

if git diff --cached --quiet 2>/dev/null; then
    echo "No changes to push"
else
    git commit -m "experiment: ${MODE} run ${TIMESTAMP}

Results:
$(grep -E 'CostModel-Routed|PAR2QO-Enhanced' "$LOG_DIR/sota_${TIMESTAMP}.log" 2>/dev/null | head -2)
" --author="dylanyunlon <dogechat@163.com>"
    
    git push origin main 2>&1 || echo "Push failed (check PAT)"
    echo "Results pushed to main"
fi

echo ""
echo "==========================================="
echo "  Experiment complete: $TIMESTAMP"
echo "  Logs: $LOG_DIR/sota_${TIMESTAMP}.log"
echo "==========================================="
