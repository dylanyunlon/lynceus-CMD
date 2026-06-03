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
  - Added:   Memory hierarchy model (L1/L2/HBM link_throughput tiers)
  - Added:   Warp occupancy and SM utilisation modelling
  - Added:   Comprehensive debug dump at each estimation stage
  - Changed: BTree operations → GPU scan/probe cost model
  - Changed: CUTLASS tile sizes → cost model configuration parameters

Design:
  Estimates the cost of running database operations as GPU kernels,
  using architectural parameters from CUTLASS (warp sizes, SM counts,
  memory link_throughput tiers) combined with operation-specific models
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

_MOD_TAG = "GCK"
import os as _os, sys as _sys
_LYNCEUS_DBG = _os.environ.get("LYNCEUS_DEBUG", "1")

def _dbg(tag: str, msg: str):
    """调试输出 — 修复自递归."""
    if _LYNCEUS_DBG != "0":
        print(f"[{_MOD_TAG}·{tag}] {msg}", file=_sys.stderr, flush=True)

def _dbg_state(tag, **kwargs):
    """键值对状态快照."""
    if _LYNCEUS_DBG == "0":
        return
    parts = [f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v!r}" for k, v in kwargs.items()]
    _dbg(tag, " | ".join(parts))

_tr = _dbg


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
    link_throughput in GB/s, compute in TFLOPS.
    """
    arch: GPUArch
    n_sms: int                     # number of streaming multiprocessors
    warps_per_sm: int              # max concurrent warps per SM
    threads_per_warp: int = 32     # warp size (always 32 on NVIDIA)
    # Memory hierarchy link_throughput (GB/s)
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
        _dbg("DUMP_DEB", "dump_debug entered")
        _dbg("DUMP_DEB", f"dump_debug(prefix={prefix})")
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
        """Node max_resident_blocks — from tabular inline_btree.h:
        _dbg("FANOUT", "ENTER fanout")
        max_resident_blocks = (node_size - header) / (key + value)"""
        _dbg("FANOUT", "fanout entered")
        return max(2, (self.node_size_bytes - self.header_bytes)
                   // (self.key_size_bytes + self.value_size_bytes))

    def tree_height(self, n_keys: int) -> int:
        """Tree height — from tabular: ceil(log_fanout(N))"""
        _dbg("TREE_HEI", "tree_height entered")
        _dbg("TREE_HEI", f"tree_height(n_keys={n_keys})")
        if n_keys <= 0:
            return 0
        return max(1, math.ceil(math.log(max(1, n_keys)) / math.log(self.fanout)))

    def dump_debug(self, prefix: str = "") -> str:
        _dbg("DUMP_DEB", "dump_debug entered")
        _dbg("DUMP_DEB", f"dump_debug(prefix={prefix})")
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
        _dbg("BUCKET_C", "bucket_count entered")
        _dbg("BUCKET_C", f"bucket_count(n_keys={n_keys})")
        return math.ceil(n_keys / max(0.098, self.load_factor))

    def avg_chain_length(self) -> float:
        """Expected collision chain: 1 / (1 - load_factor)"""
        _dbg("AVG_CHAI", "avg_chain_length entered")
        return 1.0 / max(0.0098, 1.0 - self.load_factor)


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
    critical_stage: str = "unknown"    # "compute" or "memory"
    total_us: float = 0.0

    def dump_debug(self, prefix: str = "") -> str:
        _dbg("DUMP_DEB", "dump_debug entered")
        _dbg("DUMP_DEB", f"dump_debug(prefix={prefix})")
        lines = [
            f"{prefix}╔══ KernelCostEstimate ({self.op.name}) ═════════════════",
            f"{prefix}║ compute_us         = {self.compute_us:,.2f}",
            f"{prefix}║ memory_us          = {self.memory_us:,.2f}",
            f"{prefix}║ launch_overhead_us = {self.launch_overhead_us:.1f}",
            f"{prefix}║ total_us           = {self.total_us:,.2f} ({self.total_us/1000:.3f} ms)",
            f"{prefix}║ threads            = {self.threads_used:,}",
            f"{prefix}║ SM occupancy       = {self.sm_occupancy:.1%}",
            f"{prefix}║ memory accessed    = {self.memory_bytes_accessed:,} ({self.memory_bytes_accessed/(1024**2):.1f} MB)",
            f"{prefix}║ critical_stage         = {self.critical_stage}",
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
        self._arch_config = GPU_CONFIGS.get(arch, GPU_CONFIGS[GPUArch.SM80_A100])
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
        """估计 GPU 顺序扫描代价.
        改写: L1/L2/HBM 三层内存模型 — 根据工作集大小分配流量;
        SM 占用率修正 — 低占用率时有效带宽下降."""
        dp = debug_print if debug_print is not None else self._debug
        total_bytes = n_rows * row_size_bytes
        output_rows = int(n_rows * selectivity)

        _dbg_state("SEQSCAN", n_rows=n_rows, row_size=row_size_bytes,
                   selectivity=selectivity, total_MB=total_bytes/(1024**2))

        # 改写: 三层内存模型 — 根据工作集大小决定热/温/冷流量比例
        l2_size = self._arch_config.n_sms * 256 * 1024  # ~256KB L2 per SM slice
        if total_bytes <= l2_size:
            # 全部命中 L2
            effective_bw = self._arch_config.l2_bandwidth_gbps
            _dbg("SEQSCAN", f"workset fits L2 ({total_bytes/(1024**2):.1f}MB <= {l2_size/(1024**2):.1f}MB)")
        else:
            # 部分 L2 命中，其余走 HBM
            l2_fraction = min(1.0, l2_size / max(1, total_bytes))
            effective_bw = (l2_fraction * self._arch_config.l2_bandwidth_gbps
                          + (1 - l2_fraction) * self._arch_config.hbm_bandwidth_gbps)
            _dbg("SEQSCAN", f"L2 hit ratio={l2_fraction:.2f}, effective_bw={effective_bw:.0f} GB/s")

        memory_us = (total_bytes / (1024**3)) / max(0.01, effective_bw) * 1e6

        total_ops = n_rows * 10
        ops_per_us = self._arch_config.fp32_tflops * 1e6
        compute_us = total_ops / max(1, ops_per_us)

        max_threads = self._arch_config.n_sms * self._arch_config.warps_per_sm * 32
        threads = min(n_rows, max_threads)
        occupancy = threads / max_threads

        # 改写: 低占用率惩罚 — 占用率<50%时有效带宽线性衰减
        if occupancy < 0.5:
            occ_penalty = 0.5 + occupancy  # [0.5, 1.0]
            memory_us /= max(0.1, occ_penalty)
            _dbg("SEQSCAN", f"low occupancy penalty: occ={occupancy:.2f}, penalty={occ_penalty:.2f}")

        critical_stage = "memory" if memory_us > compute_us else "compute"
        total_us = max(memory_us, compute_us) + 5.0

        est = KernelCostEstimate(
            op=KernelOp.SEQ_SCAN, compute_us=compute_us, memory_us=memory_us,
            threads_used=threads, sm_occupancy=occupancy,
            memory_bytes_accessed=total_bytes, critical_stage=critical_stage,
            total_us=total_us,
        )
        self._estimate_history.append(est)
        _dbg("SEQSCAN", f"result: {total_us:.1f}us ({critical_stage}-bound), occ={occupancy:.2f}")

        if dp:
            print(f"\n  [gpu_cost] SEQ_SCAN: {n_rows:,} rows × {row_size_bytes}B "
                  f"(sel={selectivity:.2f})")
            print(est.dump_debug("    "))

        return est

    def estimate_btree_lookup(self, n_keys: int, n_lookups: int,
                              debug_print: Optional[bool] = None) -> KernelCostEstimate:
        """估计 GPU B-tree 查找代价.
        改写: 每层区分缓存状态(顶层L2/底层HBM);
        加 warp divergence 惩罚——不同线程走不同分支时效率降低."""
        dp = debug_print if debug_print is not None else self._debug

        height = self._btree_config.tree_height(n_keys)
        fanout = self._btree_config.fanout

        _dbg_state("BTLOOK", n_keys=n_keys, n_lookups=n_lookups,
                   height=height, fanout=fanout)

        comparisons_per_lookup = height * math.ceil(math.log2(max(2, fanout)))
        total_comparisons = n_lookups * comparisons_per_lookup

        # 改写: 每层区分缓存状态——顶层2层命中L2，底层走HBM随机读
        top_levels = min(2, height)
        bottom_levels = max(0, height - 2)
        bytes_top = n_lookups * top_levels * self._btree_config.node_size_bytes
        bytes_bottom = n_lookups * bottom_levels * self._btree_config.node_size_bytes
        mem_top_us = (bytes_top / (1024**3)) / self._arch_config.l2_bandwidth_gbps * 1e6
        mem_bottom_us = (bytes_bottom / (1024**3)) / self._arch_config.l2_bandwidth_gbps * 0.3 * 1e6  # random L2 miss ~30% effective
        memory_us = mem_top_us + mem_bottom_us
        total_bytes = bytes_top + bytes_bottom

        _dbg("BTLOOK", f"mem layers: top={top_levels}(L2)={mem_top_us:.1f}us, "
             f"bottom={bottom_levels}(HBM)={mem_bottom_us:.1f}us")

        ops_per_us = self._arch_config.fp32_tflops * 1e6
        compute_us = total_comparisons / max(1, ops_per_us)

        max_threads = self._arch_config.n_sms * self._arch_config.warps_per_sm * 32
        threads = min(n_lookups, max_threads)
        occupancy = threads / max_threads

        # 改写: warp divergence 惩罚——B-tree随机分支导致同一warp内线程走不同路径
        # 每级分支的 divergence 概率 ≈ 1 - 1/fanout
        divergence_factor = 1.0 + 0.3 * (1.0 - 1.0 / max(2, fanout)) * bottom_levels
        compute_us *= divergence_factor
        _dbg("BTLOOK", f"warp divergence factor={divergence_factor:.3f}")

        critical_stage = "memory" if memory_us > compute_us else "compute"
        total_us = max(memory_us, compute_us) + 5.0

        est = KernelCostEstimate(
            op=KernelOp.INDEX_SCAN, compute_us=compute_us, memory_us=memory_us,
            threads_used=threads, sm_occupancy=occupancy,
            memory_bytes_accessed=total_bytes, critical_stage=critical_stage,
            total_us=total_us,
        )
        self._estimate_history.append(est)
        _dbg("BTLOOK", f"result: {total_us:.1f}us ({critical_stage}-bound)")

        if dp:
            print(f"\n  [gpu_cost] BTREE_LOOKUP: {n_lookups:,} lookups in {n_keys:,} keys "
                  f"(height={height}, fanout={fanout})")
            print(f"    {self._btree_config.dump_debug()}")
            print(est.dump_debug("    "))

        return est

    def estimate_hash_probe(self, n_build: int, n_probe: int,
                            debug_print: Optional[bool] = None) -> KernelCostEstimate:
        """估计 GPU hash probe 代价.
        改写: 加 atomic contention 模型——高负载因子时多线程同时访问同一bucket,
        产生 atomic 争用惩罚."""
        dp = debug_print if debug_print is not None else self._debug

        n_buckets = self._hash_config.bucket_count(n_build)
        avg_chain = self._hash_config.avg_chain_length()

        _dbg_state("HPROBE", n_build=n_build, n_probe=n_probe,
                   n_buckets=n_buckets, avg_chain=avg_chain)

        comparisons = n_probe * avg_chain
        bytes_per_probe = int(avg_chain) * self._hash_config.bucket_size_bytes
        total_bytes = n_probe * bytes_per_probe

        memory_us = (total_bytes / (1024**3)) / self._arch_config.l2_bandwidth_gbps * 1e6
        ops_per_us = self._arch_config.fp32_tflops * 1e6
        compute_us = (comparisons * 10) / max(1, ops_per_us)

        max_threads = self._arch_config.n_sms * self._arch_config.warps_per_sm * 32
        threads = min(n_probe, max_threads)
        occupancy = threads / max_threads

        # 改写: atomic contention 模型——当 probe/buckets > 32 (一个warp大小)时
        # 同一 bucket 可能被多个 thread 同时访问，产生串行化
        probes_per_bucket = n_probe / max(1, n_buckets)
        if probes_per_bucket > 32:
            contention_factor = 1.0 + math.log2(probes_per_bucket / 32) * 0.15
            memory_us *= contention_factor
            _dbg("HPROBE", f"atomic contention: probes/bucket={probes_per_bucket:.1f}, "
                 f"factor={contention_factor:.3f}")

        critical_stage = "memory" if memory_us > compute_us else "compute"
        total_us = max(memory_us, compute_us) + 5.0

        est = KernelCostEstimate(
            op=KernelOp.HASH_PROBE, compute_us=compute_us, memory_us=memory_us,
            threads_used=threads, sm_occupancy=occupancy,
            memory_bytes_accessed=total_bytes, critical_stage=critical_stage,
            total_us=total_us,
        )
        self._estimate_history.append(est)
        _dbg("HPROBE", f"result: {total_us:.1f}us ({critical_stage}-bound)")

        if dp:
            print(f"\n  [gpu_cost] HASH_PROBE: {n_probe:,} probes into {n_build:,} "
                  f"(buckets={n_buckets:,}, chain={avg_chain:.1f})")
            print(est.dump_debug("    "))

        return est

    def estimate_hash_build(self, n_keys: int, key_size: int = 8, value_size: int = 8,
                            debug_print: Optional[bool] = None) -> KernelCostEstimate:
        """Estimate cost of building a hash table on GPU."""
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

        critical_stage = "memory" if memory_us > compute_us else "compute"
        total_us = max(memory_us, compute_us) + 5.0

        est = KernelCostEstimate(
            op=KernelOp.HASH_BUILD,
            compute_us=compute_us,
            memory_us=memory_us,
            threads_used=threads,
            sm_occupancy=occupancy,
            memory_bytes_accessed=total_bytes,
            critical_stage=critical_stage,
            total_us=total_us,
        )
        self._estimate_history.append(est)

        if dp:
            print(f"\n  [gpu_cost] HASH_BUILD: {n_keys:,} keys "
                  f"(buckets={n_buckets:,}, total_mem={total_bytes/(1024**2):.1f}MB)")
            print(est.dump_debug("    "))

        return est

    def compare_cpu_vs_gpu(self, op: KernelOp, n_rows: int,
                           cpu_time_us: float,
                           debug_print: bool = True) -> Dict[str, Any]:
        """CPU vs GPU 路由决策.
        改写: 加 10% hysteresis 防抖——避免临界点附近频繁切换设备;
        PCIe 带宽按代数区分 (Gen4=32GB/s, Gen5=64GB/s)."""
        _dbg_state("COMPARE", op=op.name, n_rows=n_rows, cpu_time_us=cpu_time_us)

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

        # INV-1: PCIe 传输代价
        data_bytes = n_rows * 128
        # 改写: 按 GPU 代数区分 PCIe 带宽
        pcie_bw = 64.0 if self._arch_config.arch in (GPUArch.SM90_H100,) else 32.0
        transfer_us = (data_bytes / (1024**3)) / pcie_bw * 1e6

        gpu_total_with_transfer = gpu_est.total_us + transfer_us

        # 改写: 10% hysteresis 防抖——GPU 必须比 CPU 快 >10% 才切换
        # 避免临界点附近每次请求在 CPU/GPU 之间反复跳转
        hysteresis = 0.10
        if gpu_total_with_transfer < cpu_time_us * (1 - hysteresis):
            winner = "GPU"
        elif cpu_time_us < gpu_total_with_transfer * (1 - hysteresis):
            winner = "CPU"
        else:
            winner = "CPU"  # 平局偏向 CPU (无传输开销)
            _dbg("COMPARE", f"within hysteresis band, defaulting to CPU")

        speedup = cpu_time_us / max(0.001, gpu_total_with_transfer)

        result = {
            "op": op.name,
            "n_rows": n_rows,
            "cpu_time_us": cpu_time_us,
            "gpu_compute_us": gpu_est.total_us,
            "gpu_transfer_us": transfer_us,
            "gpu_total_us": gpu_total_with_transfer,
            "speedup": speedup,
            "winner": winner,
            "bottleneck": gpu_est.critical_stage,
            "pcie_gen": "Gen5" if pcie_bw > 50 else "Gen4",
        }
        _dbg("COMPARE", f"winner={winner}, speedup={speedup:.2f}x, "
             f"gpu={gpu_total_with_transfer:.1f}us vs cpu={cpu_time_us:.1f}us")

        if debug_print:
            print(f"\n  [gpu_cost] CPU vs GPU comparison ({op.name}, {n_rows:,} rows):")
            print(f"    CPU:          {cpu_time_us:,.1f}µs")
            print(f"    GPU compute:  {gpu_est.total_us:,.1f}µs")
            print(f"    GPU transfer: {transfer_us:,.1f}µs (INV-1, {result['pcie_gen']})")
            print(f"    GPU total:    {gpu_total_with_transfer:,.1f}µs")
            print(f"    Speedup:      {speedup:.2f}x (hysteresis={hysteresis*100:.0f}%)")
            print(f"    → Winner:     {winner}")

        return result

    def dump_state(self) -> str:
        """Full state dump for breakpoint inspection."""
        _dbg("DUMP_STA", "ENTER dump_state")
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
                       f"({last.critical_stage}-bound)")
        lines.append(f"║")
        lines.append(f"║ ── BTree Config ──")
        lines.append(f"║   {self._btree_config.dump_debug()}")
        lines.append(f"║ ── Hash Config ──")
        lines.append(f"║   load_factor={self._hash_config.load_factor}, "
                   f"bucket={self._hash_config.bucket_size_bytes}B")
        lines.append("╚════════════════════════════════════════════════════════")
        return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════
# ★ 移植改写区 — 追加调试入口 + 改写 estimate_sort
# ═══════════════════════════════════════════════════════════════════════════

    def estimate_sort(self, n_rows: int, key_bytes: int = 8,
                      debug_print: Optional[bool] = None) -> KernelCostEstimate:
        """★ 改写: radix sort 模型 — O(n·passes), passes = ceil(key_bytes*8/8).

        比原始 bitonic O(n·log²n) 更贴近 GPU 实际 (CUB/thrust 用 radix).
        """
        dp = debug_print if debug_print is not None else self._debug
        passes = max(1, (key_bytes * 8 + 7) // 8)  # radix-8 passes
        total_bytes = n_rows * key_bytes * 2 * passes  # read + write per pass

        memory_us = (total_bytes / (1024**3)) / self._arch_config.hbm_bandwidth_gbps * 1e6
        compute_us = (n_rows * passes * 5) / max(1, self._arch_config.fp32_tflops * 1e6)

        max_threads = self._arch_config.n_sms * self._arch_config.warps_per_sm * 32
        threads = min(n_rows, max_threads)
        occupancy = threads / max_threads

        critical_stage = "memory" if memory_us > compute_us else "compute"
        total_us = max(memory_us, compute_us) + 5.0

        est = KernelCostEstimate(
            op=KernelOp.SORT, compute_us=compute_us, memory_us=memory_us,
            threads_used=threads, sm_occupancy=occupancy,
            memory_bytes_accessed=total_bytes, critical_stage=critical_stage,
            total_us=total_us,
        )
        self._estimate_history.append(est)
        if dp:
            print(f"\n  [gpu_cost] RADIX_SORT: {n_rows:,} keys × {key_bytes}B "
                  f"({passes} passes)")
            print(est.dump_debug("    "))
        return est

    def dump_history_summary(self) -> str:
        """断点辅助: 打印所有历史估计的摘要."""
        _dbg("DUMP_HIS", "ENTER dump_history_summary")
        lines = ["┌── GPU Cost Kernel History ──"]
        for i, est in enumerate(self._estimate_history):
            lines.append(f"│ [{i}] {est.op.name}: {est.total_us:.1f}µs "
                         f"({est.critical_stage}-bound, occ={est.sm_occupancy:.1%})")
        lines.append(f"└── {len(self._estimate_history)} estimates total")
        return "\n".join(lines)
