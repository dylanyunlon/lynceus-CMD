#!/bin/bash
# ==============================================================================
#  Lynceus-CMD — Experiment Runner
#  Cost-model-driven heterogeneous (CPU/GPU) query routing
#  Repository: https://github.com/dylanyunlon/lynceus-CMD
#
#  This is the single entry point for reproducing the Lynceus benchmark
#  experiments end to end: environment setup (conda), repository preparation,
#  test verification (Python + C++), the cost-model benchmark sweep, and
#  packaging of the JSON panels the paper figures are built from.
#
#  The Lynceus core is a pure-Python cost-model SIMULATOR (Python stdlib only);
#  it does not train models and does not require a GPU to run. The C++/CUDA-
#  style cost kernels compile as host C++17 with g++. A GPU is therefore
#  OPTIONAL — if present we record its topology for reference, but every step
#  below runs on a CPU-only box.
# ==============================================================================

set -euo pipefail

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

# --- Server paths (override via environment variables) ------------------------
# Default layout mirrors the lab server convention:
#   /data/<user>/system/.../lynceus
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
UPSTREAM_DIR="${UPSTREAM_DIR:-$PROJECT_DIR/upstream}"

# --- Conda environment --------------------------------------------------------
# Per request, the experiment environment is the shared 'walking3' env.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-walking3}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# --- Benchmark parameters (override via environment variables) ----------------
WORKLOAD_NAME="${WORKLOAD_NAME:-TPC-H_SF100}"
NUM_STEPS="${NUM_STEPS:-2000}"     # queries per seed
NUM_SEEDS="${NUM_SEEDS:-3}"        # independent seeds for mean/std
CXX="${CXX:-g++}"                  # C++ compiler for the cost kernels
CXXFLAGS="${CXXFLAGS:--std=c++17 -O2 -Wall -Wextra}"

# --- Reproducibility ----------------------------------------------------------
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

# Routing strategies compared in the benchmark (informational; the Python
# runner enumerates them itself).
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

# Resolve a python interpreter: prefer the active conda env's, else python3.
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
# System Check
# ==============================================================================

check_system() {
    print_step "Checking System Requirements"

    echo "Host: $(hostname)"
    echo "Working dir: $SCRIPT_DIR"
    echo ""

    # conda is required for the environment step.
    check_command conda || {
        echo "Error: Conda is required. Install Miniconda/Anaconda first."
        exit 1
    }
    echo "conda: $(conda --version)"

    # g++ is required to build and run the C++ cost-kernel tests.
    check_command "$CXX" || {
        echo "Error: a C++17 compiler ('$CXX') is required for the cost kernels."
        exit 1
    }
    echo "compiler: $($CXX --version | head -1)"

    # GPU is OPTIONAL. Record topology if a driver is present; never fail.
    echo ""
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected (optional — Lynceus runs CPU-only):"
        nvidia-smi --query-gpu=index,name,memory.total,driver_version \
                   --format=csv,noheader 2>/dev/null \
            || echo "  (nvidia-smi present but query failed; continuing)"
    else
        echo "No NVIDIA GPU detected — fine. Lynceus is a cost-model simulator"
        echo "and reproduces all benchmarks on CPU."
    fi

    echo ""
    echo "OK  System check passed"
}

# ==============================================================================
# Environment Setup  (conda: walking3)
# ==============================================================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"

    eval "$(conda shell.bash hook)"

    if conda env list | grep -qE "^\s*${CONDA_ENV_NAME}\s"; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists — reusing it."
    else
        echo "Creating conda environment '${CONDA_ENV_NAME}' (Python ${PYTHON_VERSION})..."
        conda create -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" -y
    fi

    conda activate "${CONDA_ENV_NAME}"
    resolve_python
    echo "Active python: $("$PYTHON_BIN" --version) at $(command -v "$PYTHON_BIN")"

    # The Lynceus core depends only on the Python standard library. The few
    # extras below are for optional plotting / nicer logs and are safe to skip
    # on a locked-down server (hence the '|| true').
    echo ""
    echo "Installing optional analysis extras (safe to skip if offline)..."
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

    # Unpack bundled sample data if present and not yet extracted.
    if [ -f "$PROJECT_DIR/data.zip" ] && [ ! -d "$DATA_DIR/extracted" ]; then
        echo "Extracting bundled data.zip -> $DATA_DIR/extracted ..."
        mkdir -p "$DATA_DIR/extracted"
        ( cd "$DATA_DIR/extracted" && unzip -o -q "$PROJECT_DIR/data.zip" ) \
            || echo "  (data.zip extraction skipped/failed; not required for the benchmark)"
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
# Verify  (Python test suites + C++ cost-kernel tests)
# ==============================================================================

verify() {
    print_step "Verifying Build (Python + C++ test suites)"
    resolve_python
    mkdir -p "$LOG_DIR"   # self-sufficient: do not depend on 'prepare' first

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
        echo "ERROR  One or more suites failed — see $LOG_DIR/"
        exit 1
    fi
}

# ==============================================================================
# Benchmark  (the core experiment)
# ==============================================================================

run_benchmark() {
    print_step "Running Cost-Model Benchmark"
    resolve_python
    mkdir -p "$LOG_DIR" "$OUTPUT_DIR" output  # self-sufficient dirs

    echo "Workload : $WORKLOAD_NAME"
    echo "Steps    : $NUM_STEPS per seed"
    echo "Seeds    : $NUM_SEEDS"
    echo "Strategies: ${STRATEGIES[*]}"
    echo ""

    # The benchmark runner writes JSON panels into ./output. We let it own the
    # paths it documents (output/latency_vs_step.json, output/cumulative_latency.json)
    # and then copy them under OUTPUT_DIR for archival with a timestamp.
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"

    WORKLOAD_NAME="$WORKLOAD_NAME" NUM_STEPS="$NUM_STEPS" NUM_SEEDS="$NUM_SEEDS" \
        "$PYTHON_BIN" -m lynceus.benchmark 2>&1 | tee "$LOG_DIR/benchmark_${ts}.log"

    # Archive the produced panels.
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
# Package Results
# ==============================================================================

package_results() {
    print_step "Packaging Results"
    mkdir -p "$OUTPUT_DIR"

    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local archive="$OUTPUT_DIR/lynceus_results_${ts}.tar.gz"

    if compgen -G "$OUTPUT_DIR/run_*" > /dev/null 2>&1; then
        # Collect run directories as a clean array (no word-splitting on ls).
        local runs=()
        local d
        for d in "$OUTPUT_DIR"/run_*; do
            [ -d "$d" ] && runs+=("$(basename "$d")")
        done
        tar -czf "$archive" -C "$OUTPUT_DIR" "${runs[@]}" \
            && echo "OK  Packaged -> $archive"
    else
        echo "No benchmark runs found under $OUTPUT_DIR — run './run_lynceus.sh benchmark' first."
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
  setup        - Create/activate conda env 'walking3' and install extras
  prepare      - Create data/output/log dirs, unpack bundled data.zip
  verify       - Run Python + C++ test suites
  benchmark    - Run the cost-model benchmark sweep, archive JSON panels
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
  WORKLOAD_NAME  Benchmark workload name  (default: TPC-H_SF100)
  NUM_STEPS      Queries per seed         (default: 2000)
  NUM_SEEDS      Independent seeds        (default: 3)
  CXX            C++ compiler             (default: g++)

Examples:
  # Full pipeline on a fresh server
  ./run_lynceus.sh all

  # Just (re)run the benchmark with a smaller sweep
  NUM_STEPS=500 NUM_SEEDS=2 ./run_lynceus.sh benchmark

  # Custom server paths
  PROJECT_DIR=/data/$USER/system/lynceus OUTPUT_DIR=/data/$USER/out \
      ./run_lynceus.sh all

Notes:
  - Lynceus is a cost-model SIMULATOR: pure Python stdlib, no GPU needed.
  - The C++/CUDA-style cost kernels compile as host C++17 (g++).
  - A GPU, if present, is only inspected for reference in 'check'.

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
        package)
            package_results
            ;;
        all)
            check_system
            setup_environment
            prepare_repo
            verify
            run_benchmark
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
