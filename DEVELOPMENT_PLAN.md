# Lynceus-CMD Development Plan — NeurIPS Review Grade

> **Data Target**: Match the X-axis granularity of `data_demo/` (2000–3000 steps,
> 3+ seeds, multi-method comparison panels) for **cost-model-driven query routing
> on heterogeneous GPU-CPU systems**.

---

## Architecture-to-Function Mapping (grep'd from real infra repos)

### NeurIPS Review Narrative

从 **NCCL `ncclTopoCompute()`** (nccl/src/graph/search.cc:1023) 这个好例子开始 —
它在拓扑图上搜索最优通信路径。然后，遵循该模式实现一个新的
**`HardwareTopologyGraph`**，让 Lynceus 可以描述 GPU/CPU 异构拓扑，并能像
`ncclGetBtree()` (nccl/src/graph/trees.cc:32) 那样构建二叉树通信拓扑。

接着 **Megatron `forward_backward_pipelining_with_interleaving()`**
(Megatron-LM/megatron/core/pipeline_parallel/schedules.py:896) 引入流水线调度思想，
使 Lynceus 的 QueryPipelineScheduler 能够将查询按成本模型分段路由到不同硬件，
同时 **DeepSpeed `DeepSpeedZeroOptimizer_Stage3`**
(DeepSpeed/deepspeed/runtime/zero/stage3.py:136) 优化内存分区策略 — 我们参考其
`partition_grads()` (stage3.py:1703) 来设计查询工作集的分片。

随后 **vLLM `PagedAttention`** (vllm/vllm/v1/attention/ops/paged_attn.py:15) 整合
分页式缓存管理思想，令 Lynceus 的 IndexCacheManager 支持类似 vLLM
`KVCacheBlocks` (vllm/vllm/v1/core/kv_cache_manager.py:26) 的块级索引缓存，
进而 **CUTLASS `GemmUniversal`**
(cutlass/include/cutlass/gemm/kernel/gemm_universal.h:65) 增强 GPU 上矩阵运算的
cost estimation kernel。

**DeepSeek-V3 `act_quant_kernel()`** (DeepSeek-V3/inference/kernel.py) 引入 FP8
量化策略，使 Lynceus 能对大表统计信息做低精度量化存储，同时 **TransformerEngine
`fp8_gemm_enabled()`** 验证 FP8 路径可行性。

**BytePS `push_pull()`** (byteps/byteps/torch/ops.py:127) 提供参数服务器通信原语，
令分布式 cost model 更新支持异步推拉。**APEX `DistributedFusedAdam`**
(apex/apex/contrib/optimizers/distributed_fused_adam.py:270) 完善分布式优化器，
确保 cost model 参数更新在多 GPU 上融合执行。

**JAX `pjit()`** (jax/jax/_src/pjit.py:671) 整合自动分片能力，令 Lynceus 在 TPU/GPU
混合环境下支持 `with_sharding_constraint()` (pjit.py:2027) 级别的负载分配。

**PyTorch `all_reduce()`** (pytorch/torch/distributed/distributed_c10d.py:3156) 兼容
标准 NCCL backend，全面升级 Lynceus 的分布式统计收集以达成生产级可用性。

最终 **FairScale `FullyShardedDataParallel`**
(fairscale/fairscale/nn/data_parallel/fully_sharded_data_parallel.py:99) 完善
端到端全分片训练兼容层，确保 Lynceus 兼容主流训练框架。

同时整合 upstream: **PAR2QO `get_plan_cost()`** (par2qo/code/postgres.py:110)
提供真实查询代价估计，**VIDEX `VidexModelBase.cardinality()`**
(videx/src/sub_platforms/sql_opt/videx/model/videx_strategy.py) 提供虚拟索引基数估计，
**Tabular `Index::Insert()`** (tabular/src/index/hash_table_common.h:82) 提供
高效索引构建能力。

---

## Milestone Plan (M001–M038)

| Claude # | Milestone | Scope | Key Deliverable |
|----------|-----------|-------|-----------------|
| **#1** | **M001–M002** | Core schema + cost model engine | `lynceus/schema.py`, `lynceus/cost_model.py`, `lynceus/topology.py` |
| #2 | M003–M004 | Query routing strategies (GPU/CPU/Hybrid/CostDriven) | `lynceus/router.py`, `lynceus/strategies/` |
| #3 | M005–M006 | Benchmark runner + data output in demo format | `lynceus/benchmark.py`, `lynceus/data_writer.py` |
| #4 | M007–M008 | PAR2QO integration (robust plan selection) | `lynceus/integrations/par2qo_bridge.py` |
| #5 | M009–M010 | VIDEX integration (virtual index cost) | `lynceus/integrations/videx_bridge.py` |
| #6 | M011–M012 | Tabular integration (index build cost) | `lynceus/integrations/tabular_bridge.py` |
| #7 | M013–M014 | Hardware topology graph (NCCL-inspired) | `lynceus/topology.py` enhancement |
| #8 | M015–M016 | Pipeline scheduler (Megatron-inspired) | `lynceus/pipeline_scheduler.py` |
| #9 | M017–M018 | Index cache manager (vLLM-inspired paging) | `lynceus/cache_manager.py` |
| #10 | M019–M020 | FP8 statistics quantization (DeepSeek-inspired) | `lynceus/fp8_stats.py` |
| #11 | M021–M022 | GPU kernel cost estimation (CUTLASS-informed) | `lynceus/gpu_cost_kernel.py` |
| #12 | M023–M024 | Distributed cost model sync (BytePS push-pull) | `lynceus/distributed/sync.py` |
| #13 | M025–M026 | Mixed-precision optimizer (APEX-inspired) | `lynceus/distributed/optimizer.py` |
| #14 | M027–M028 | Auto-sharding (JAX pjit-inspired) | `lynceus/sharding.py` |
| #15 | M029–M030 | NCCL-backend all_reduce collector | `lynceus/distributed/collector.py` |
| #16 | M031–M032 | FSDP compatibility layer | `lynceus/distributed/fsdp_compat.py` |
| #17 | M033–M034 | End-to-end experiment configs | `configs/`, `scripts/run_benchmark.py` |
| #18 | M035–M036 | Visualization + figure generation | `lynceus/viz/plot_panels.py` |
| #19 | M037 | Integration tests | `tests/` |
| #20–#38 | M038+ | Per-workload experiments, ablation panels, paper figures | `experiments/` |

---

## Data Output Schema (Target)

```json
{
  "metadata": {
    "panel": "Query Latency vs Workload Step — TPC-H SF100",
    "source": "lynceus_benchmark_run_20260528",
    "n_per_seed": 2000,
    "n_seeds": 3,
    "n_methods": 5,
    "total_data_points": 30000
  },
  "methods": {
    "GPU-Only": {
      "x_steps": [0, 1, ..., 1999],
      "total_cost": 847.3,
      "reported_final": 42.5,
      "seed_0": [latency_0, latency_1, ...],
      "seed_1": [...],
      "seed_2": [...],
      "mean": [...],
      "std": [...]
    },
    "CPU-Only": { ... },
    "Hybrid-Static": { ... },
    "CostModel-Routed": { ... },
    "PAR2QO-Enhanced": { ... }
  }
}
```

---

## Design Invariants & Audit Corrections (binding for all future milestones)

> An end-to-end audit of M015–M020 (run in-environment, no new test code) found
> 6 production-grade defects. The corrected semantics below are now **binding
> contracts**. Future Claudes MUST preserve them; violating any one reintroduces
> a known data-correctness or routing bug. See `AUDIT_REPORT.md` for the full
> critique (user-perspective + system-perspective) of each.

### INV-1 — Transfer cost never disappears from a query's latency
A single query's operator stages form a **strict data-dependency chain**
(SCAN→FILTER→JOIN→AGG→SORT) and cannot overlap. Its latency is the **full
critical path**: `latency_us = Σ(stage compute) + Σ(stage transfer)`, where the
transfer term includes the initial load into the first stage's device. Never
subtract a stage's transfer from its compute and treat it as "hidden".
*(Bug it prevents: a 15 ms PCIe load vanished, yielding a fake 246× speedup.)*

### INV-2 — Pipeline speedup is a property of BATCHES, not single queries
Intra-query speedup does not exist. Cross-query pipelining uses the Megatron-LM
bubble fraction `(p−1)/(m+p−1)` (Narayanan et al. 2021) for m independent
queries over p stages: `t_pipe = t_serial · (m+p−1)/(m·p)`. At m=1 this is
exactly `t_serial` (speedup 1.0). Use `schedule_pipeline(queries)` for the batch
case; `schedule(query)` reports single-query critical-path latency only.

### INV-3 — Logical table identity is an explicit field, never string-guessed
Cache-block keying and per-table row counts use `QueryDescriptor.table_name`.
Never infer a table from `query_id` via `str.rstrip(<digits>)` or similar — that
collapses unrelated ids (`q_00000`→`q`) and fabricates cache hit rates. The
benchmark generator stamps a real table (TPC-H catalog) on every query.

### INV-4 — Topology transfer cost is multi-hop shortest path
`HardwareTopology.get_transfer_cost` runs Dijkstra over the edge graph
(non-negative hop costs), so data on a second NUMA node reaches a GPU via
`cpu1→cpu0→gpu0`. Never assume a direct edge must exist; absence of a direct
link is not unreachability. Contracts: `src==dst → 0`, genuinely disconnected
→ `inf`.

### INV-5 — Quantization error metrics have zero/NaN guards
`measure_error` must (a) normalise the relative error of a zero-valued original
by the column's max magnitude (so FP8 polluting a 0 is not silently ignored),
and (b) flag `has_nonfinite` on any NaN/Inf input so `acceptable()` rejects it
rather than swallowing it through NaN comparisons. The C++ core and the Python
port must stay bit-identical on these paths.

### INV-6 — Under block-wise fp32 scaling, E4M3 dominates E5M2 on accuracy
Adoption order is E4M3 → fp32 (kept), never E4M3 → E5M2 "for accuracy": the
per-block scale absorbs dynamic range, so mantissa bits decide and E4M3 has more.
E5M2 is an explicit opt-in only for unscaled wide-range use. Adoption is gated on
**both** an SNR floor and a per-element max-relative-error ceiling.

---

## Experiment Runner (`run_lynceus.sh`)

A single, self-contained entry point reproduces the benchmark end to end on the
lab server. It is the operational front door for the `#17 (M033–M034)` experiment
milestone and should be kept in sync as new panels are added.

**Conventions (binding):**
- Conda environment is `walking3` (override with `CONDA_ENV_NAME`). The runner
  reuses it if present and only creates it (Python 3.10) when missing.
- Server paths are environment-overridable: `PROJECT_DIR`, `DATA_DIR`,
  `OUTPUT_DIR`, `LOG_DIR`, `UPSTREAM_DIR`. Defaults are relative to the script.
- Lynceus is a **pure-Python cost-model simulator** (stdlib only) — no GPU is
  required. A GPU, if present, is only *inspected* (topology print) in `check`;
  it is never a hard dependency. The C++/CUDA-style cost kernels build as host
  C++17 with `g++` (`CXX`/`CXXFLAGS` overridable).
- Sweep size is driven by `NUM_STEPS` / `NUM_SEEDS` / `WORKLOAD_NAME`. These are
  honoured by `lynceus.benchmark.main()` (it reads them from the environment),
  so the shell knobs are real — they change the produced metadata, not just the
  log text. Any future entry point MUST keep advertised knobs functional
  (no silently-ignored parameters).

**Commands:** `check`, `setup`, `prepare`, `verify`, `benchmark`, `package`,
`all` (= the six in order), `help`.

**Verified:** `bash -n` clean; `shellcheck -S warning` clean (0 warnings); the
full `prepare → verify → benchmark → package` chain runs on a CPU-only box and
`NUM_STEPS`/`NUM_SEEDS` provably propagate into the output JSON metadata.
