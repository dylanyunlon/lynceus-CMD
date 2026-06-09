#!/usr/bin/env bash
# ============================================================
#  run_server_experiment.sh — ags1 服务器端实验 + 自动push
#
#  在 ags1 上: conda activate walking3 && bash scripts/run_server_experiment.sh
#
#  流程:
#    1. 检测硬件 (GPU/CPU/NUMA topology)
#    2. 运行全套实验 (SOTA comparison, ablation, scalability, cache, cost accuracy)
#    3. 生成 LaTeX 表格数据
#    4. 自动 commit + push 到 main
#
#  硬件: 2x EPYC 9354 (128 cores), 2x A6000 (49GB) + 1x H100 NVL (96GB)
#  NUMA: node0=cpu0-31,64-95  node1=cpu32-63,96-127 (GPUs on node1)
#  Conda: walking3 (Python 3.10, PyTorch 2.4.1, CUDA 12.1 -> nvcc 11.5)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR" output

R='\033[0m'; G='\033[32m'; Y='\033[33m'; C='\033[36m'; D='\033[90m'

log() { echo -e "${G}[$(date +%H:%M:%S)]${R} $*"; }
warn() { echo -e "${Y}[$(date +%H:%M:%S)] WARN:${R} $*"; }

# ── 1. 硬件探测 ─────────────────────────────────────────────
log "硬件探测..."
{
    echo "=== Hardware Probe ${TIMESTAMP} ==="
    echo "--- CPU ---"
    lscpu | grep -E "Model name|Socket|Core|Thread|NUMA|CPU\(s\):|Architecture" 2>/dev/null || true
    echo "--- Memory ---"
    free -h 2>/dev/null || cat /proc/meminfo | head -5
    echo "--- GPU ---"
    nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current,compute_cap \
        --format=csv,noheader 2>/dev/null || echo "no GPU"
    echo "--- NUMA Topology ---"
    nvidia-smi topo -m 2>/dev/null || echo "topo not available"
    numactl --hardware 2>/dev/null | head -20 || true
    echo "--- NVLink ---"
    nvidia-smi nvlink -s 2>/dev/null || echo "no nvlink"
    echo "--- Software ---"
    echo "kernel: $(uname -r)"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true
    nvcc --version 2>/dev/null | tail -1 || echo "nvcc not found"
    python3 --version 2>/dev/null
    python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')" 2>/dev/null || true
} > "$LOG_DIR/hardware_${TIMESTAMP}.txt" 2>&1
log "硬件信息: $LOG_DIR/hardware_${TIMESTAMP}.txt"

# ── 2. 环境检查 ─────────────────────────────────────────────
log "Python环境检查..."
python3 -c "
import sys
sys.path.insert(0, '.')
from lynceus.costing import CostModelEngine, create_default_topology
from lynceus.benchmark import WorkloadConfig
from lynceus.router import Router
print(f'  Python {sys.version_info.major}.{sys.version_info.minor}')
print(f'  Lynceus core: OK')
topo = create_default_topology()
print(f'  Topology: {len(topo.nodes)} nodes, {len(topo.edges)} edges')
" 2>/dev/null
log "环境OK"

# ── 3. 全套实验 ─────────────────────────────────────────────
export LYNCEUS_DBG=0  # 关闭debug输出以获得准确的timing

# 3a. SOTA比较 (2000步 x 3种子 = 主实验)
log "实验1/5: SOTA Comparison (2000x3)..."
LYNCEUS_DBG=0 python3 scripts/sota_comparison.py \
    2>&1 | tee "$LOG_DIR/sota_${TIMESTAMP}.txt" | tail -20
log "SOTA完成 → output/sota_comparison.json"

# 3b. 消融实验
log "实验2/5: Ablation Study (500x2)..."
LYNCEUS_DBG=0 python3 scripts/ablation_study.py --steps 500 --seeds 2 \
    2>&1 | tee "$LOG_DIR/ablation_${TIMESTAMP}.txt" | tail -15
log "Ablation完成 → output/ablation_results.json"

# 3c. 可扩展性
log "实验3/5: Scalability Benchmark..."
LYNCEUS_DBG=0 python3 scripts/scalability_bench.py \
    2>&1 | tee "$LOG_DIR/scalability_${TIMESTAMP}.txt" | tail -15
log "Scalability完成 → output/scalability_bench.json"

# 3d. Cache有效性
log "实验4/5: Cache Effectiveness..."
LYNCEUS_DBG=0 python3 scripts/cache_effectiveness.py \
    2>&1 | tee "$LOG_DIR/cache_${TIMESTAMP}.txt" | tail -10
log "Cache完成 → output/cache_effectiveness.json"

# 3e. Cost Model准确度
log "实验5/5: Cost Model Accuracy..."
LYNCEUS_DBG=0 python3 scripts/cost_model_accuracy.py \
    2>&1 | tee "$LOG_DIR/accuracy_${TIMESTAMP}.txt" | tail -10
log "Accuracy完成 → output/costing_accuracy.json"

# ── 4. 生成LaTeX表格 ────────────────────────────────────────
log "生成LaTeX表格..."
python3 scripts/generate_paper_tables.py 2>&1 | tail -5
log "表格生成完成"

# ── 5. 完整debug状态快照 ────────────────────────────────────
log "生成完整状态快照..."
python3 -c "
import json, os, glob, sys
sys.path.insert(0, '.')

snapshot = {
    'timestamp': '${TIMESTAMP}',
    'experiments': {},
    'file_counts': {},
}

# 收集所有output JSON
for f in sorted(glob.glob('output/*.json')):
    name = os.path.basename(f).replace('.json', '')
    try:
        data = json.load(open(f))
        snapshot['experiments'][name] = {
            'keys': list(data.keys()),
            'size_bytes': os.path.getsize(f),
        }
        # 提取关键指标
        if 'our_methods' in data:
            for method, info in data['our_methods'].items():
                if isinstance(info, dict) and 'mean_latency_us' in info:
                    snapshot['experiments'][name].setdefault('results', {})[method] = {
                        'mean_us': info['mean_latency_us'],
                        'improvement_pct': info.get('improvement_pct', 0),
                    }
    except Exception as e:
        snapshot['experiments'][name] = {'error': str(e)}

# 文件统计
py_files = list(glob.glob('lynceus/**/*.py', recursive=True))
snapshot['file_counts'] = {
    'integration_py': len(py_files),
    'total_lines': sum(sum(1 for _ in open(f)) for f in py_files if os.path.isfile(f)),
    'scripts': len(glob.glob('scripts/*.py')),
    'tests': len(glob.glob('tests/*.py')),
}

json.dump(snapshot, open('output/experiment_snapshot.json', 'w'), indent=2)
print(f'Snapshot: {len(snapshot[\"experiments\"])} experiments, {snapshot[\"file_counts\"][\"integration_py\"]} py files')
"
log "状态快照 → output/experiment_snapshot.json"

# ── 6. Git commit + push ────────────────────────────────────
log "Git commit + push..."
git add -A
git diff --cached --stat

COMMIT_MSG="experiment: full run ${TIMESTAMP} on ags1

Hardware: 2x EPYC 9354, 2x A6000 + 1x H100 NVL
Experiments: SOTA comparison, ablation, scalability, cache, cost accuracy
Algorithm: UCB-Thompson hybrid, momentum FPL, query-type transfer penalty
Debug: full state snapshots in output/experiment_snapshot.json"

if git diff --cached --quiet; then
    log "无变化，跳过commit"
else
    git commit -m "$COMMIT_MSG" --author="dylanyunlon <dogechat@163.com>"
    if [ -n "${GH_PAT:-}" ]; then
        git push "https://dylanyunlon:${GH_PAT}@github.com/dylanyunlon/lynceus-CMD.git" main
        log "已push到GitHub"
    else
        warn "GH_PAT未设置，跳过push。手动: git push origin main"
    fi
fi

log "全部完成！耗时: ${SECONDS}s"
echo ""
echo -e "${C}=== 实验摘要 ===${R}"
python3 -c "
import json
data = json.load(open('output/sota_comparison.json'))
print('SOTA Comparison Results:')
for method, info in data.get('our_methods', {}).items():
    if isinstance(info, dict) and 'mean_latency_us' in info:
        imp = info.get('improvement_pct', 0)
        mark = '★' if imp > 10 else '  '
        print(f'  {mark} {method:20s}  mean={info[\"mean_latency_us\"]:8.1f}µs  improv={imp:+.1f}%')
"
