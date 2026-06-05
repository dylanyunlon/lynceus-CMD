#!/bin/bash
# ==============================================================================
#  Lynceus-CMD — Experiment Runner
#  Cost-model-driven heterogeneous (CPU/GPU) query routing
#  Repository: https://github.com/dylanyunlon/lynceus-CMD
#
#  Single entry point: env setup → verify → benchmark → package.
#  Lynceus core is pure-Python stdlib; C++ cost kernels compile as host C++17.
#  GPU is OPTIONAL — if present we record topology, but everything runs on CPU.
# ==============================================================================

set -e

echo "=========================================="
echo "   Lynceus-CMD  Experiment Runner"
echo "   Heterogeneous CPU/GPU query routing"
echo "=========================================="
echo ""

# ==============================================================================
# Configuration Section
# ==============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
UPSTREAM_DIR="${UPSTREAM_DIR:-$PROJECT_DIR/upstream}"

# Conda environment
CONDA_ENV_NAME="${CONDA_ENV_NAME:-walking3}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# Benchmark parameters
WORKLOAD_NAME="${WORKLOAD_NAME:-TPC-H_SF100}"
NUM_STEPS="${NUM_STEPS:-2000}"
NUM_SEEDS="${NUM_SEEDS:-3}"
CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--std=c++17 -O2 -Wall -Wextra}"

# Reproducibility
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

# Routing strategies compared
STRATEGIES=("GPU-Only" "CPU-Only" "Hybrid-Static" "CostModel-Routed" "PAR2QO-Enhanced" "Adaptive")

# ==============================================================================
# Utility Functions
# ==============================================================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: '$1' is not installed or not on PATH"
        return 1
    fi
    return 0
}

PYTHON_BIN="python"
resolve_python() {
    if command -v python &> /dev/null; then
        PYTHON_BIN="python"
    elif command -v python3 &> /dev/null; then
        PYTHON_BIN="python3"
    else
        echo "Error: no python interpreter found"
        exit 1
    fi
}

# ==============================================================================
# Check System Requirements
# ==============================================================================

check_system() {
    print_step "Checking System Requirements"

    echo "Host: $(hostname)"
    echo "Working dir: $SCRIPT_DIR"
    echo ""

    check_command conda || {
        echo "Error: Conda is required. Install Miniconda/Anaconda first."
        exit 1
    }
    echo "conda: $(conda --version)"

    check_command "$CXX" || {
        echo "Error: a C++17 compiler ('$CXX') is required for the cost kernels."
        exit 1
    }
    echo "compiler: $($CXX --version | head -1)"

    # GPU is OPTIONAL
    echo ""
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected (optional):"
        nvidia-smi --query-gpu=index,name,memory.total,driver_version \
                   --format=csv,noheader 2>/dev/null \
            || echo "  (nvidia-smi present but query failed; continuing)"
        echo ""
        echo "CUDA Version:"
        nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "  not found"
    else
        echo "No NVIDIA GPU detected — fine. Lynceus runs CPU-only."
    fi

    echo ""
    echo "OK  System check passed"
}

# ==============================================================================
# Environment Setup  (conda)
# ==============================================================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"

    eval "$(conda shell.bash hook)"

    if conda env list | grep -qE "^\s*${CONDA_ENV_NAME}\s"; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists — reusing."
    else
        echo "Creating conda environment '${CONDA_ENV_NAME}' (Python ${PYTHON_VERSION})..."
        conda create -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" -y
    fi

    conda activate "${CONDA_ENV_NAME}"
    resolve_python
    echo "Active python: $("$PYTHON_BIN" --version) at $(command -v "$PYTHON_BIN")"

    echo ""
    echo "Installing optional extras..."
    "$PYTHON_BIN" -m pip install --upgrade pip >/dev/null 2>&1 || true
    "$PYTHON_BIN" -m pip install numpy matplotlib rich >/dev/null 2>&1 \
        || echo "  (optional extras not installed — core benchmark still runs)"

    echo ""
    echo "OK  Environment ready: $CONDA_ENV_NAME"
}

# ==============================================================================
# Repository Preparation
# ==============================================================================

prepare_repo() {
    print_step "Preparing Repository"

    mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$LOG_DIR"

    if [ -f "$PROJECT_DIR/data.zip" ] && [ ! -d "$DATA_DIR/extracted" ]; then
        echo "Extracting bundled data.zip -> $DATA_DIR/extracted ..."
        mkdir -p "$DATA_DIR/extracted"
        ( cd "$DATA_DIR/extracted" && unzip -o -q "$PROJECT_DIR/data.zip" ) \
            || echo "  (data.zip extraction skipped/failed; not required)"
    fi

    if [ -d "$UPSTREAM_DIR" ]; then
        echo "Upstream references present: $(ls "$UPSTREAM_DIR" 2>/dev/null | tr '\n' ' ')"
    fi

    echo ""
    echo "OK  Repository prepared"
    echo "  PROJECT_DIR = $PROJECT_DIR"
    echo "  OUTPUT_DIR  = $OUTPUT_DIR"
    echo "  LOG_DIR     = $LOG_DIR"
}

# ==============================================================================
# Verify  (Python + C++ test suites)
# ==============================================================================

verify() {
    print_step "Verifying Build (Python + C++ test suites)"
    resolve_python
    mkdir -p "$LOG_DIR"

    local py_suites=(
        tests/test_m001_m002.py
        tests/test_m003_m004.py
        tests/test_m015_m016.py
        tests/test_m017_m018.py
        tests/test_m019_m020_py.py
    )

    echo "-- Python suites --"
    local failed=0
    for t in "${py_suites[@]}"; do
        if [ -f "$t" ]; then
            if "$PYTHON_BIN" "$t" > "$LOG_DIR/$(basename "$t").log" 2>&1; then
                echo "  PASS  $t"
            else
                echo "  FAIL  $t  (see $LOG_DIR/$(basename "$t").log)"
                failed=1
            fi
        fi
    done

    echo ""
    echo "-- C++ cost-kernel suite --"
    if [ -f tests/test_m019_m020.cpp ]; then
        if $CXX $CXXFLAGS -I. tests/test_m019_m020.cpp -o "$LOG_DIR/test_fp8_cpp" \
               > "$LOG_DIR/cpp_build.log" 2>&1; then
            if "$LOG_DIR/test_fp8_cpp" > "$LOG_DIR/cpp_run.log" 2>&1; then
                echo "  PASS  tests/test_m019_m020.cpp"
            else
                echo "  FAIL  C++ test run (see $LOG_DIR/cpp_run.log)"; failed=1
            fi
        else
            echo "  FAIL  C++ build (see $LOG_DIR/cpp_build.log)"; failed=1
        fi
    fi

    echo ""
    if [ "$failed" -eq 0 ]; then
        echo "OK  All test suites passed"
    else
        echo "WARN  Some suites failed — see $LOG_DIR/ (non-blocking, benchmark continues)"
    fi
}

# ==============================================================================
# Benchmark  (core experiment)
# ==============================================================================

run_benchmark() {
    print_step "Running Cost-Model Benchmark"
    resolve_python
    mkdir -p "$LOG_DIR" "$OUTPUT_DIR" output

    echo "Workload : $WORKLOAD_NAME"
    echo "Steps    : $NUM_STEPS per seed"
    echo "Seeds    : $NUM_SEEDS"
    echo "Strategies: ${STRATEGIES[*]}"
    echo ""

    local ts
    ts="$(date +%Y%m%d_%H%M%S)"

    WORKLOAD_NAME="$WORKLOAD_NAME" NUM_STEPS="$NUM_STEPS" NUM_SEEDS="$NUM_SEEDS" \
        "$PYTHON_BIN" -m lynceus.benchmark 2>&1 | tee "$LOG_DIR/benchmark_${ts}.log"

    mkdir -p "$OUTPUT_DIR/run_${ts}"
    for f in output/latency_vs_step.json output/cumulative_latency.json; do
        if [ -f "$f" ]; then
            cp "$f" "$OUTPUT_DIR/run_${ts}/"
            echo "  archived $(basename "$f") -> $OUTPUT_DIR/run_${ts}/"
        fi
    done

    echo ""
    echo "OK  Benchmark complete. Panels in $OUTPUT_DIR/run_${ts}/"
}

# ==============================================================================
# Evaluate  (topology + pipeline diagnostics)
# ==============================================================================

run_eval() {
    print_step "Running Evaluation Diagnostics"
    resolve_python

    "$PYTHON_BIN" -c "
import sys, json
sys.path.insert(0, '.')
from lynceus.topology import create_default_topology

print('=== Topology Diagnostics ===')
topo = create_default_topology(n_gpus=4, debug_print=True)

gpu_ids = [f'gpu{i}' for i in range(4)]
for size_mb in [10, 100, 1000]:
    t = topo.estimate_allreduce_us(gpu_ids, data_bytes=size_mb*1024*1024)
    print(f'  allreduce {size_mb}MB: {t:.1f} us')

# NUMA verification
c_same, _ = topo._dijkstra('gpu0', 'gpu1')
c_cross, _ = topo._dijkstra('gpu0', 'gpu2')
print(f'  NUMA same={c_same:.3f} cross={c_cross:.3f} ratio={c_cross/c_same:.2f}x')

snap = topo._dbg_snapshot()
print(f'  snapshot: {snap[\"n_nodes\"]} nodes, {snap[\"n_edges\"]} edges')
print('=== Topology OK ===')
" 2>&1 | tee "$LOG_DIR/eval_topo.log"

    echo ""
    echo "OK  Evaluation diagnostics complete"
}

# ==============================================================================
# Package Results
# ==============================================================================

package_results() {
    print_step "Packaging Results"
    mkdir -p "$OUTPUT_DIR"

    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local archive="$OUTPUT_DIR/lynceus_results_${ts}.tar.gz"

    if compgen -G "$OUTPUT_DIR/run_*" > /dev/null 2>&1; then
        local runs=()
        local d
        for d in "$OUTPUT_DIR"/run_*; do
            [ -d "$d" ] && runs+=("$(basename "$d")")
        done
        tar -czf "$archive" -C "$OUTPUT_DIR" "${runs[@]}" \
            && echo "OK  Packaged -> $archive"
    else
        echo "No benchmark runs found under $OUTPUT_DIR — run benchmark first."
    fi
}

# ==============================================================================
# Help
# ==============================================================================

show_help() {
    cat << 'EOF'
Lynceus-CMD Experiment Runner

Usage: ./run_lynceus.sh [command]

Commands:
  check        - Check system requirements (conda, g++, optional GPU)
  setup        - Create/activate conda env and install extras
  prepare      - Create data/output/log dirs, unpack bundled data.zip
  verify       - Run Python + C++ test suites
  benchmark    - Run the cost-model benchmark sweep
  eval         - Run topology + pipeline diagnostics
  package      - Tar up the produced result panels
  all          - check -> setup -> prepare -> verify -> benchmark -> package
  help         - Show this help

Environment variables (with defaults):
  PROJECT_DIR    Project root              (default: script directory)
  DATA_DIR       Data directory           (default: PROJECT_DIR/data)
  OUTPUT_DIR     Output directory         (default: PROJECT_DIR/output)
  LOG_DIR        Log directory            (default: PROJECT_DIR/logs)
  CONDA_ENV_NAME Conda environment        (default: walking3)
  PYTHON_VERSION Python for new env       (default: 3.10)
  WORKLOAD_NAME  Benchmark workload       (default: TPC-H_SF100)
  NUM_STEPS      Queries per seed         (default: 2000)
  NUM_SEEDS      Independent seeds        (default: 3)
  CXX            C++ compiler             (default: g++)

Examples:
  ./run_lynceus.sh all
  NUM_STEPS=500 NUM_SEEDS=2 ./run_lynceus.sh benchmark
  ./run_lynceus.sh eval

Repository: https://github.com/dylanyunlon/lynceus-CMD
EOF
}

# ==============================================================================
# Main
# ==============================================================================

main() {
    local COMMAND="${1:-help}"

    case "$COMMAND" in
        check)
            check_system
            ;;
        setup)
            check_system
            setup_environment
            ;;
        prepare)
            prepare_repo
            ;;
        verify)
            verify
            ;;
        benchmark|bench)
            run_benchmark
            ;;
        eval|evaluate)
            run_eval
            ;;
        package)
            package_results
            ;;
        all)
            check_system
            setup_environment
            prepare_repo
            verify
            run_benchmark
            run_eval
            package_results
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './run_lynceus.sh help' for usage information"
            exit 1
            ;;
    esac
}

main "$@"
