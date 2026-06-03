"""
lynceus/gpu_cost_kernel.py — GPU kernel cost estimation (CUTLASS-informed).

Architecture references (ported/adapted from):
  - tabular inline_btree.h (tabular/src/tabular/inline_btree.h:1-861)
    → InlineBTree node structure: fanout, height, split/merge
    → template<typename Key, typename Value> with configurable node size
    → insert/search path length estimation
    → node_capacity(), tree_height(), search_cost() computation
  - tabular inline_btree_wrapper.h (tabular/src/index/wrappers/inline_btree_wrapper.h:1-143)
    → BTree wrapper for index building: Insert(), Scan(), PointQuery()
    → cost tracking per operation type
  - CUTLASS (nvidia/cutlass/include/cutlass/gemm/kernel)
    → GEMM kernel launch configuration: tile sizes, warp counts, pipeline stages
    → memory access pattern cost model (global/shared/register)
  - tabular hash_table_common.h (tabular/src/index/hash_table_common.h:1-130)
    → hash bucket sizing, collision chain estimation

Modifications from upstream references (~20% original):
  - Removed: actual CUDA kernel code, GPU memory allocations
  - Removed: template metaprogramming, C++ node structs
  - Removed: CUTLASS GEMM tile configuration and CTA scheduling
  - Added:   Cost estimation for GPU kernels without actually running them
  - Added:   Memory hierarchy model (L1/L2/HBM bandwidth tiers)
  - Added:   Warp occupancy and SM utilisation modelling
  - Added:   Comprehensive debug dump at each estimation stage
  - Changed: BTree operations → GPU scan/probe cost model
  - Changed: CUTLASS tile sizes → cost model configuration parameters

Design:
  Estimates the cost of running database operations as GPU kernels,
  using architectural parameters from CUTLASS (warp sizes, SM counts,
  memory bandwidth tiers) combined with operation-specific models
  from tabular (BTree traversal, hash probing, sequential scan).
  This drives the CPU-vs-GPU routing decision in the cost model.
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto

from . import _dbg, _dump_obj, _snapshot, _Timer, LYNCEUS_DEBUG
_T = "GPU"


logger = logging.getLogger(__name__)


# ─── GPU Architecture Parameters ────────────────────────────────────────────
# Informed by CUTLASS kernel launch configs and CUDA architecture specs.

class GPUArch(Enum):
    """GPU architecture — affects kernel cost parameters."""
    SM80_A100 = auto()       # Ampere A100
    SM89_4090 = auto()       # Ada Lovelace RTX 4090
    SM90_H100 = auto()       # Hopper H100


@dataclass
class GPUArchConfig:
    """Architecture-specific performance parameters.

    Values from CUTLASS device properties and CUDA toolkit docs.
    bandwidth in GB/s, compute in TFLOPS.
    """
    arch: GPUArch
    n_sms: int                     # number of streaming multiprocessors
    warps_per_sm: int              # max concurrent warps per SM
    threads_per_warp: int = 32     # warp size (always 32 on NVIDIA)
    # Memory hierarchy bandwidth (GB/s)
    hbm_bandwidth_gbps: float = 0.0    # HBM / global memory
    l2_bandwidth_gbps: float = 0.0     # L2 cache
    l1_bandwidth_gbps: float = 0.0     # L1 / shared memory
    # Compute throughput
    fp32_tflops: float = 0.0
    fp16_tflops: float = 0.0
    int8_tops: float = 0.0             # INT8 tera-ops
    # Clock (GHz)
    clock_ghz: float = 1.5

    def dump_debug(self, prefix: str = "") -> str:
        _dbg(_T, f"dump_debug called")
        lines = [
            f"{prefix}╔══ GPUArchConfig ({self.arch.name}) ══════════════════",
            f"{prefix}║ SMs             = {self.n_sms}",
            f"{prefix}║ warps/SM        = {self.warps_per_sm}",
            f"{prefix}║ max_threads     = {self.n_sms * self.warps_per_sm * self.threads_per_warp:,}",
            f"{prefix}║ HBM bw          = {self.hbm_bandwidth_gbps:.0f} GB/s",
            f"{prefix}║ L2 bw           = {self.l2_bandwidth_gbps:.0f} GB/s",
            f"{prefix}║ L1/smem bw      = {self.l1_bandwidth_gbps:.0f} GB/s",
            f"{prefix}║ FP32            = {self.fp32_tflops:.1f} TFLOPS",
            f"{prefix}║ FP16            = {self.fp16_tflops:.1f} TFLOPS",
            f"{prefix}║ clock           = {self.clock_ghz:.2f} GHz",
            f"{prefix}╚═══════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


# Pre-configured architectures
GPU_CONFIGS = {
    GPUArch.SM80_A100: GPUArchConfig(
        arch=GPUArch.SM80_A100, n_sms=108, warps_per_sm=64,
        hbm_bandwidth_gbps=2039.0, l2_bandwidth_gbps=6000.0,
        l1_bandwidth_gbps=19000.0, fp32_tflops=19.5,
        fp16_tflops=312.0, int8_tops=624.0, clock_ghz=1.41,
    ),
    GPUArch.SM89_4090: GPUArchConfig(
        arch=GPUArch.SM89_4090, n_sms=128, warps_per_sm=48,
        hbm_bandwidth_gbps=1008.0, l2_bandwidth_gbps=5000.0,
        l1_bandwidth_gbps=16000.0, fp32_tflops=82.6,
        fp16_tflops=165.2, int8_tops=330.3, clock_ghz=2.52,
    ),
    GPUArch.SM90_H100: GPUArchConfig(
        arch=GPUArch.SM90_H100, n_sms=132, warps_per_sm=64,
        hbm_bandwidth_gbps=3350.0, l2_bandwidth_gbps=12000.0,
        l1_bandwidth_gbps=33000.0, fp32_tflops=67.0,
        fp16_tflops=989.0, int8_tops=1979.0, clock_ghz=1.83,
    ),
}


# ─── Kernel Operation Types ─────────────────────────────────────────────────

class KernelOp(Enum):
    """GPU kernel operation type for cost estimation."""
    SEQ_SCAN = auto()         # sequential scan of table data
    INDEX_SCAN = auto()       # B-tree index lookup on GPU
    HASH_PROBE = auto()       # hash table probe
    HASH_BUILD = auto()       # hash table construction
    SORT = auto()             # GPU sort (radix or merge)
    JOIN_NL = auto()          # nested-loop join kernel
    JOIN_HASH = auto()        # hash join probe kernel
    AGGREGATE = auto()        # group-by aggregation
    GEMM = auto()             # matrix multiply (for ML-in-DB)


# ─── BTree GPU Cost Model ───────────────────────────────────────────────────
# Adapted from tabular inline_btree.h:
#   - node_capacity = (node_size - header) / (key_size + value_size)
#   - height = ceil(log(N) / log(fanout))
#   - search_cost = height * (binary_search_in_node + cache_miss)
# And inline_btree_wrapper.h:
#   - Insert() tracks insert_count
#   - Scan() does range scan with iterator
#   - PointQuery() does single key lookup

@dataclass
class BTreeGPUConfig:
    """BTree configuration on GPU — adapted from tabular inline_btree.h.

    In tabular, the BTree is templated on Key/Value types with a
    configurable node size. Here we parameterise those as integers.
    """
    node_size_bytes: int = 256      # tabular: PAGE_SIZE
    key_size_bytes: int = 8         # sizeof(Key)
    value_size_bytes: int = 8       # sizeof(Value)
    # Derived from tabular inline_btree.h node structure
    header_bytes: int = 16          # node header (count, flags, pointers)

    @property
    def fanout(self) -> int:
        """Node capacity — from tabular inline_btree.h:
        capacity = (node_size - header) / (key + value)"""

        _dbg(_T, f"fanout called")
        return max(2, (self.node_size_bytes - self.header_bytes)
                   // (self.key_size_bytes + self.value_size_bytes))

    def tree_height(self, n_keys: int) -> int:
        """Tree height — from tabular: ceil(log_fanout(N))"""
        _dbg(_T, f"tree_height called")
        if n_keys <= 0:
            return 0

        return max(1, math.ceil(math.log(max(1, n_keys)) / math.log(self.fanout)))

    def dump_debug(self, prefix: str = "") -> str:

        _dbg(_T, f"dump_debug called")
        return (f"{prefix}BTreeGPU: node={self.node_size_bytes}B, "
                f"key={self.key_size_bytes}B, val={self.value_size_bytes}B, "
                f"fanout={self.fanout}")


# ─── Hash Table GPU Config ──────────────────────────────────────────────────
# From tabular hash_table_common.h:
#   - bucket_count = n_keys / load_factor
#   - collision chain length ~ 1 / (1 - load_factor)

@dataclass
class HashTableGPUConfig:
    """Hash table configuration — from tabular hash_table_common.h."""
    load_factor: float = 0.7
    bucket_size_bytes: int = 64     # cache-line aligned
    key_size_bytes: int = 8
    value_size_bytes: int = 8

    def bucket_count(self, n_keys: int) -> int:
        """From tabular: n_buckets = ceil(n_keys / load_factor)"""

        _dbg(_T, f"bucket_count called")
        return math.ceil(n_keys / max(0.1, self.load_factor))

    def avg_chain_length(self) -> float:
        """Expected collision chain: 1 / (1 - load_factor)"""

        _dbg(_T, f"avg_chain_length called")
        return 1.0 / max(0.01, 1.0 - self.load_factor)


# ─── GPU Kernel Cost Estimator ───────────────────────────────────────────────

@dataclass
class KernelCostEstimate:
    """Cost estimate for a GPU kernel execution."""
    op: KernelOp
    # Time breakdown (µs)
    compute_us: float = 0.0        # arithmetic operations
    memory_us: float = 0.0         # memory access (HBM/L2/L1)
    launch_overhead_us: float = 5.0  # kernel launch overhead (~5µs)
    # Resource usage
    threads_used: int = 0
    sm_occupancy: float = 0.0      # fraction of SMs utilised
    memory_bytes_accessed: int = 0
    # Bottleneck analysis
    bottleneck: str = "unknown"    # "compute" or "memory"
    total_us: float = 0.0

    def dump_debug(self, prefix: str = "") -> str:
        _dbg(_T, f"dump_debug called")
        lines = [
            f"{prefix}╔══ KernelCostEstimate ({self.op.name}) ═════════════════",
            f"{prefix}║ compute_us         = {self.compute_us:,.2f}",
            f"{prefix}║ memory_us          = {self.memory_us:,.2f}",
            f"{prefix}║ launch_overhead_us = {self.launch_overhead_us:.1f}",
            f"{prefix}║ total_us           = {self.total_us:,.2f} ({self.total_us/1000:.3f} ms)",
            f"{prefix}║ threads            = {self.threads_used:,}",
            f"{prefix}║ SM occupancy       = {self.sm_occupancy:.1%}",
            f"{prefix}║ memory accessed    = {self.memory_bytes_accessed:,} ({self.memory_bytes_accessed/(1024**2):.1f} MB)",
            f"{prefix}║ bottleneck         = {self.bottleneck}",
            f"{prefix}╚═══════════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


class GPUCostKernel:
    """Estimates GPU kernel execution costs for database operations.

    Uses GPU architecture parameters (from CUTLASS device models) combined
    with operation-specific cost models (from tabular BTree/hash structures)
    to predict whether a given operation is cheaper on GPU or CPU.
    """

    def __init__(self, arch: GPUArch = GPUArch.SM80_A100,
                 debug_print: bool = True):
        _dbg(_T, f"__init__ called")
        self._arch_config = GPU_CONFIGS.get(arch, GPU_CONFIGS[GPUArch.SM80_A100])  # typed
        self._btree_config = BTreeGPUConfig()
        self._hash_config = HashTableGPUConfig()
        self._debug = debug_print
        self._estimate_history: List[KernelCostEstimate] = []

        if debug_print:
            print(f"\n[gpu_cost_kernel] Initialized for {arch.name}")
            print(self._arch_config.dump_debug("  "))

    def estimate_seq_scan(self, n_rows: int, row_size_bytes: int = 128,
                          selectivity: float = 1.0,
                          debug_print: Optional[bool] = None) -> KernelCostEstimate:
        """Estimate cost of sequential scan on GPU.

        Each thread processes one row. Memory-bound: HBM bandwidth
        determines throughput.
        """
        _dbg(_T, f"estimate_seq_scan called")
        dp = debug_print if debug_print is not None else self._debug
        total_bytes = n_rows * row_size_bytes
        output_rows = int(n_rows * selectivity)

        # Memory time: read all rows from HBM
        memory_us = (total_bytes / (1024**3)) / self._arch_config.hbm_bandwidth_gbps * 1e6

        # Compute time: one comparison per row (trivial)
        # Each thread does ~10 ops (load, compare, conditional store)
        total_ops = n_rows * 10
        ops_per_us = self._arch_config.fp32_tflops * 1e6  # ops per µs
        compute_us = total_ops / max(1, ops_per_us)

        # Occupancy: threads = n_rows, up to max hardware threads
        max_threads = self._arch_config.n_sms * self._arch_config.warps_per_sm * 32
        threads = min(n_rows, max_threads)
        occupancy = threads / max_threads

        # 改写: warp divergence penalty — 低 selectivity 时一个 warp 内
        # 部分线程有输出、部分没有, 整个 warp 还是要跑完。
        # 有效利用率 ≈ selectivity + (1-selectivity) * idle_fraction
        # 当 selectivity=0.01 时, 31/32 线程在等, 效率极低。
        warp_efficiency = max(0.03125, selectivity + (1.0 - selectivity) / 32.0)
        # divergence 放大 compute time (memory 不受影响, 数据还是全读)
        effective_compute_us = compute_us / warp_efficiency

        # Bottleneck analysis
        bottleneck = "memory" if memory_us > effective_compute_us else "compute"
        # launch overhead 随 grid size 变: 小 grid ≈ 2µs, 大 grid 线性增长
        n_blocks = max(1, -(-threads // 256))
        launch_us = 2.0 + 0.001 * n_blocks  # 基础 + per-block 调度
        total_us = max(memory_us, effective_compute_us) + launch_us

        est = KernelCostEstimate(
            op=KernelOp.SEQ_SCAN,
            compute_us=compute_us,
            memory_us=memory_us,
            threads_used=threads,
            sm_occupancy=occupancy,
            memory_bytes_accessed=total_bytes,
            bottleneck=bottleneck,
            total_us=total_us,
        )
        self._estimate_history.append(est); self._estimate_history = self._estimate_history[-4096:]  # 改写: cap

        if dp:
            print(f"\n  [gpu_cost] SEQ_SCAN: {n_rows:,} rows × {row_size_bytes}B "
                  f"(sel={selectivity:.2f})")
            print(est.dump_debug("    "))

        return est

    def estimate_btree_lookup(self, n_keys: int, n_lookups: int,
                              debug_print: Optional[bool] = None) -> KernelCostEstimate:
        """Estimate cost of B-tree index lookups on GPU.

        Adapted from tabular inline_btree.h search path:
          height = ceil(log_fanout(N))
          per lookup: height × (binary search in node + L2 cache miss)

        On GPU, each thread handles one lookup. Memory accesses are
        mostly random (L2 cache misses), which is the bottleneck.
        """
        _dbg(_T, f"estimate_btree_lookup called")
        dp = debug_print if debug_print is not None else self._debug

        height = self._btree_config.tree_height(n_keys)
        fanout = self._btree_config.fanout

        # Per lookup: height levels × (binary search + node load)
        # Binary search in node: log2(fanout) comparisons
        comparisons_per_lookup = height * math.ceil(math.log2(max(2, fanout)))
        total_comparisons = n_lookups * comparisons_per_lookup

        # Memory: each level = one node read (mostly L2 misses for random access)
        bytes_per_lookup = height * self._btree_config.node_size_bytes
        total_bytes = n_lookups * bytes_per_lookup
        # 改写: L2 cache hit probability — 不全是 miss。
        # 顶层节点频繁访问, L2 命中率高; 叶子节点随机, 命中率低。
        # 模型: level k 的 L2 hit rate ≈ min(1, cache_size / working_set_at_k)
        # 简化: 前 2 层几乎全命中 L2, 之后逐层衰减。
        l2_size_bytes = 40 * (1024 ** 2)  # typical 40MB L2
        per_level_ws = n_lookups * self._btree_config.node_size_bytes
        l2_hit_bytes = 0
        l2_miss_bytes = 0
        for lv in range(height):
            level_bytes = n_lookups * self._btree_config.node_size_bytes
            # 顶层共享同一组节点, working set 小
            unique_nodes_at_level = min(n_lookups, max(1, n_keys // (self._btree_config.fanout ** (height - lv))))
            ws = unique_nodes_at_level * self._btree_config.node_size_bytes
            hit_rate = min(1.0, l2_size_bytes / max(1, ws))
            l2_hit_bytes += level_bytes * hit_rate
            l2_miss_bytes += level_bytes * (1.0 - hit_rate)
        # L2 hit 用 L2 bandwidth, miss 用 HBM bandwidth
        memory_us = (l2_hit_bytes / (1024**3)) / self._arch_config.l2_bandwidth_gbps * 1e6 +                      (l2_miss_bytes / (1024**3)) / self._arch_config.hbm_bandwidth_gbps * 1e6

        # Compute: comparisons
        ops_per_us = self._arch_config.fp32_tflops * 1e6
        compute_us = total_comparisons / max(1, ops_per_us)

        max_threads = self._arch_config.n_sms * self._arch_config.warps_per_sm * 32
        threads = min(n_lookups, max_threads)
        occupancy = threads / max_threads

        bottleneck = "memory" if memory_us > compute_us else "compute"
        total_us = max(memory_us, compute_us) + 5.0

        est = KernelCostEstimate(
            op=KernelOp.INDEX_SCAN,
            compute_us=compute_us,
            memory_us=memory_us,
            threads_used=threads,
            sm_occupancy=occupancy,
            memory_bytes_accessed=total_bytes,
            bottleneck=bottleneck,
            total_us=total_us,
        )
        self._estimate_history.append(est); self._estimate_history = self._estimate_history[-4096:]  # 改写: cap

        if dp:
            print(f"\n  [gpu_cost] BTREE_LOOKUP: {n_lookups:,} lookups in {n_keys:,} keys "
                  f"(height={height}, fanout={fanout})")
            print(f"    {self._btree_config.dump_debug()}")
            print(est.dump_debug("    "))

        return est

    def estimate_hash_probe(self, n_build: int, n_probe: int,
                            debug_print: Optional[bool] = None) -> KernelCostEstimate:
        """Estimate cost of hash probe on GPU.

        Adapted from tabular hash_table_common.h:
          n_buckets = n_build / load_factor
          avg_chain = 1 / (1 - load_factor)
          per probe: hash + chain walk
        """
        _dbg(_T, f"estimate_hash_probe called")
        dp = debug_print if debug_print is not None else self._debug

        n_buckets = self._hash_config.bucket_count(n_build)
        avg_chain = self._hash_config.avg_chain_length()

        # Per probe: 1 hash + avg_chain comparisons + avg_chain cache misses
        comparisons = n_probe * avg_chain
        bytes_per_probe = int(avg_chain) * self._hash_config.bucket_size_bytes
        total_bytes = n_probe * bytes_per_probe

        # 改写: warp bank conflict — 同一 warp 内 32 个线程 hash 到
        # 同一个 bucket 时, 内存访问要串行化。冲突概率 ≈ 32/n_buckets。
        # 当 n_buckets >> 32 时忽略不计, 但 build 表小时很显著。
        conflict_prob = min(1.0, 32.0 / max(1, n_buckets))
        # 冲突时一个 warp 的 32 次访问变成串行, 相当于 throughput 降到 1/32
        effective_parallelism = 32 * (1.0 - conflict_prob) + 1.0 * conflict_prob
        warp_slowdown = 32.0 / effective_parallelism

        memory_us = (total_bytes / (1024**3)) / self._arch_config.l2_bandwidth_gbps * 1e6
        memory_us *= warp_slowdown  # bank conflict 放大延迟
        ops_per_us = self._arch_config.fp32_tflops * 1e6
        compute_us = (comparisons * 10) / max(1, ops_per_us)

        max_threads = self._arch_config.n_sms * self._arch_config.warps_per_sm * 32
        threads = min(n_probe, max_threads)
        occupancy = threads / max_threads

        bottleneck = "memory" if memory_us > compute_us else "compute"
        total_us = max(memory_us, compute_us) + 5.0

        est = KernelCostEstimate(
            op=KernelOp.HASH_PROBE,
            compute_us=compute_us,
            memory_us=memory_us,
            threads_used=threads,
            sm_occupancy=occupancy,
            memory_bytes_accessed=total_bytes,
            bottleneck=bottleneck,
            total_us=total_us,
        )
        self._estimate_history.append(est); self._estimate_history = self._estimate_history[-4096:]  # 改写: cap

        if dp:
            print(f"\n  [gpu_cost] HASH_PROBE: {n_probe:,} probes into {n_build:,} "
                  f"(buckets={n_buckets:,}, chain={avg_chain:.1f})")
            print(est.dump_debug("    "))

        return est

    def estimate_hash_build(self, n_keys: int, key_size: int = 8, value_size: int = 8,
                            debug_print: Optional[bool] = None) -> KernelCostEstimate:
        """Estimate cost of building a hash table on GPU."""
        _dbg(_T, f"estimate_hash_build called")
        dp = debug_print if debug_print is not None else self._debug

        n_buckets = self._hash_config.bucket_count(n_keys)
        entry_size = key_size + value_size
        total_bytes = n_keys * entry_size + n_buckets * self._hash_config.bucket_size_bytes

        # Build: one write per key (hash + store)
        memory_us = (total_bytes / (1024**3)) / self._arch_config.hbm_bandwidth_gbps * 1e6
        compute_us = (n_keys * 20) / max(1, self._arch_config.fp32_tflops * 1e6)

        max_threads = self._arch_config.n_sms * self._arch_config.warps_per_sm * 32
        threads = min(n_keys, max_threads)
        occupancy = threads / max_threads

        bottleneck = "memory" if memory_us > compute_us else "compute"
        total_us = max(memory_us, compute_us) + 5.0

        est = KernelCostEstimate(
            op=KernelOp.HASH_BUILD,
            compute_us=compute_us,
            memory_us=memory_us,
            threads_used=threads,
            sm_occupancy=occupancy,
            memory_bytes_accessed=total_bytes,
            bottleneck=bottleneck,
            total_us=total_us,
        )
        self._estimate_history.append(est); self._estimate_history = self._estimate_history[-4096:]  # 改写: cap

        if dp:
            print(f"\n  [gpu_cost] HASH_BUILD: {n_keys:,} keys "
                  f"(buckets={n_buckets:,}, total_mem={total_bytes/(1024**2):.1f}MB)")
            print(est.dump_debug("    "))

        return est

    def compare_cpu_vs_gpu(self, op: KernelOp, n_rows: int,
                           cpu_time_us: float,
                           debug_print: bool = True) -> Dict[str, Any]:
        """Compare CPU vs GPU cost for an operation.

        This drives the core routing decision in Lynceus:
        should this operation run on CPU or GPU?

        Includes transfer cost (INV-1: transfer cost can't disappear).
        """
        # Estimate GPU cost based on operation type
        _dbg(_T, f"compare_cpu_vs_gpu called")
        if op == KernelOp.SEQ_SCAN:
            gpu_est = self.estimate_seq_scan(n_rows, debug_print=False)
        elif op == KernelOp.INDEX_SCAN:
            gpu_est = self.estimate_btree_lookup(n_rows, n_rows // 10, debug_print=False)
        elif op == KernelOp.HASH_PROBE:
            gpu_est = self.estimate_hash_probe(n_rows, n_rows // 2, debug_print=False)
        elif op == KernelOp.HASH_BUILD:
            gpu_est = self.estimate_hash_build(n_rows, debug_print=False)
        else:
            gpu_est = self.estimate_seq_scan(n_rows, debug_print=False)

        # INV-1: Add PCIe transfer cost (data must move to GPU)
        data_bytes = n_rows * 128  # assume 128B/row
        transfer_us = (data_bytes / (1024**3)) / 32.0 * 1e6  # PCIe 4.0 x16 ~32 GB/s

        gpu_total_with_transfer = gpu_est.total_us + transfer_us

        # 改写: Amdahl 修正 — GPU 只加速并行部分, 串行初始化
        # (malloc, 内核配置, 结果收集) 不可并行。
        # 串行比例随数据量递减: 大数据几乎全并行, 小数据开销占比高。
        serial_fraction = min(0.5, 100.0 / max(1, n_rows))  # 100行以下过半是开销
        # Amdahl: 实际加速 = 1 / (s + (1-s)/p), p = GPU对并行部分的加速比
        parallel_speedup = cpu_time_us / max(0.001, gpu_total_with_transfer)
        effective_speedup = 1.0 / (serial_fraction + (1.0 - serial_fraction) / max(0.001, parallel_speedup))
        # 用修正后的加速比决定winner
        winner = "GPU" if effective_speedup > 1.0 else "CPU"
        speedup = effective_speedup

        result = {
            "op": op.name,
            "n_rows": n_rows,
            "cpu_time_us": cpu_time_us,
            "gpu_compute_us": gpu_est.total_us,
            "gpu_transfer_us": transfer_us,
            "gpu_total_us": gpu_total_with_transfer,
            "speedup": speedup,
            "winner": winner,
            "bottleneck": gpu_est.bottleneck,
        }

        if debug_print:
            print(f"\n  [gpu_cost] CPU vs GPU comparison ({op.name}, {n_rows:,} rows):")
            print(f"    CPU:          {cpu_time_us:,.1f}µs")
            print(f"    GPU compute:  {gpu_est.total_us:,.1f}µs")
            print(f"    GPU transfer: {transfer_us:,.1f}µs (INV-1: included)")
            print(f"    GPU total:    {gpu_total_with_transfer:,.1f}µs")
            print(f"    Speedup:      {speedup:.2f}x")
            print(f"    → Winner:     {winner}")

        return result

    def dump_state(self) -> str:
        """Full state dump for breakpoint inspection."""
        _dbg(_T, f"dump_state called")
        lines = [
            "╔══ GPUCostKernel State ═══════════════════════════════",
            f"║ arch             = {self._arch_config.arch.name}",
            f"║ SMs              = {self._arch_config.n_sms}",
            f"║ HBM bw           = {self._arch_config.hbm_bandwidth_gbps:.0f} GB/s",
            f"║ FP16             = {self._arch_config.fp16_tflops:.1f} TFLOPS",
            f"║ estimates_done   = {len(self._estimate_history)}",
        ]
        if self._estimate_history:
            last = self._estimate_history[-1]
            lines.append(f"║ last_estimate    = {last.op.name}: {last.total_us:.1f}µs "
                       f"({last.bottleneck}-bound)")
        lines.append(f"║")
        lines.append(f"║ ── BTree Config ──")
        lines.append(f"║   {self._btree_config.dump_debug()}")
        lines.append(f"║ ── Hash Config ──")
        lines.append(f"║   load_factor={self._hash_config.load_factor}, "
                   f"bucket={self._hash_config.bucket_size_bytes}B")
        lines.append("╚════════════════════════════════════════════════════════")
        return "\n".join(lines)
