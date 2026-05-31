"""
lynceus_port/gpu_cost_kernel.py — GPU kernel 代价估算（CUTLASS-informed）。

移植自 lynceus/gpu_cost_kernel.py，修改约20%:
  - 新增 L1 缓存命中率建模：小数据集可能完全驻留 L1
  - Roofline 分析：用算术强度判断 compute-bound vs memory-bound
  - debug 输出统一使用 _dbg
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum, auto

from .schema import _dbg


class GPUArch(Enum):
    SM80_A100 = auto()
    SM89_4090 = auto()
    SM90_H100 = auto()


@dataclass
class GPUArchConfig:
    arch: GPUArch
    n_sms: int
    warps_per_sm: int
    threads_per_warp: int = 32
    hbm_bandwidth_gbps: float = 0.0
    l2_bandwidth_gbps: float = 0.0
    l1_bandwidth_gbps: float = 0.0
    fp32_tflops: float = 0.0
    fp16_tflops: float = 0.0
    int8_tops: float = 0.0
    clock_ghz: float = 1.5
    # ── 新增: L1 cache 大小 (KB per SM) ──
    l1_cache_kb_per_sm: float = 192.0

    def dump_debug(self) -> str:
        return (f"GPUArch({self.arch.name}): {self.n_sms} SMs, "
                f"HBM={self.hbm_bandwidth_gbps:.0f}GB/s, "
                f"FP16={self.fp16_tflops:.0f}T")


GPU_CONFIGS = {
    GPUArch.SM80_A100: GPUArchConfig(
        arch=GPUArch.SM80_A100, n_sms=108, warps_per_sm=64,
        hbm_bandwidth_gbps=2039.0, l2_bandwidth_gbps=6000.0,
        l1_bandwidth_gbps=19000.0, fp32_tflops=19.5,
        fp16_tflops=312.0, int8_tops=624.0, clock_ghz=1.41,
        l1_cache_kb_per_sm=192.0,
    ),
    GPUArch.SM89_4090: GPUArchConfig(
        arch=GPUArch.SM89_4090, n_sms=128, warps_per_sm=48,
        hbm_bandwidth_gbps=1008.0, l2_bandwidth_gbps=5000.0,
        l1_bandwidth_gbps=16000.0, fp32_tflops=82.6,
        fp16_tflops=165.2, int8_tops=330.3, clock_ghz=2.52,
        l1_cache_kb_per_sm=128.0,
    ),
    GPUArch.SM90_H100: GPUArchConfig(
        arch=GPUArch.SM90_H100, n_sms=132, warps_per_sm=64,
        hbm_bandwidth_gbps=3350.0, l2_bandwidth_gbps=12000.0,
        l1_bandwidth_gbps=33000.0, fp32_tflops=67.0,
        fp16_tflops=989.0, int8_tops=1979.0, clock_ghz=1.83,
        l1_cache_kb_per_sm=256.0,
    ),
}


class KernelOp(Enum):
    SEQ_SCAN   = auto()
    INDEX_SCAN = auto()
    HASH_PROBE = auto()
    HASH_BUILD = auto()
    SORT       = auto()
    JOIN_NL    = auto()
    JOIN_HASH  = auto()
    AGGREGATE  = auto()
    GEMM       = auto()


@dataclass
class BTreeGPUConfig:
    node_size_bytes: int = 256
    key_size_bytes: int = 8
    value_size_bytes: int = 8
    header_bytes: int = 16

    @property
    def fanout(self) -> int:
        return max(2, (self.node_size_bytes - self.header_bytes)
                   // (self.key_size_bytes + self.value_size_bytes))

    def tree_height(self, n_keys: int) -> int:
        if n_keys <= 0:
            return 0
        return max(1, math.ceil(math.log(max(1, n_keys)) / math.log(self.fanout)))


@dataclass
class HashTableGPUConfig:
    load_factor: float = 0.7
    bucket_size_bytes: int = 64
    key_size_bytes: int = 8
    value_size_bytes: int = 8

    def bucket_count(self, n_keys: int) -> int:
        return math.ceil(n_keys / max(0.1, self.load_factor))

    def avg_chain_length(self) -> float:
        return 1.0 / max(0.01, 1.0 - self.load_factor)


@dataclass
class KernelCostEstimate:
    op: KernelOp
    compute_us: float = 0.0
    memory_us: float = 0.0
    launch_overhead_us: float = 5.0
    threads_used: int = 0
    sm_occupancy: float = 0.0
    memory_bytes_accessed: int = 0
    bottleneck: str = "unknown"
    total_us: float = 0.0
    # ── 新增: roofline 算术强度 ──
    arithmetic_intensity: float = 0.0

    def debug_snapshot(self) -> str:
        s = (f"Kernel({self.op.name}): total={self.total_us:.2f}us "
             f"comp={self.compute_us:.2f} mem={self.memory_us:.2f} "
             f"occ={self.sm_occupancy:.2f} AI={self.arithmetic_intensity:.3f} "
             f"[{self.bottleneck}]")
        _dbg("KernelEst", s)
        return s


class GPUCostKernel:
    def __init__(self, arch: GPUArch = GPUArch.SM80_A100,
                 debug_print: bool = True):
        self._arch = GPU_CONFIGS.get(arch, GPU_CONFIGS[GPUArch.SM80_A100])
        self._btree = BTreeGPUConfig()
        self._hash = HashTableGPUConfig()
        self._debug = debug_print
        self._history: List[KernelCostEstimate] = []
        if debug_print:
            _dbg("GPUKernel", f"init {arch.name}: {self._arch.dump_debug()}")

    def _effective_bandwidth(self, total_bytes: int) -> float:
        """根据数据量选择带宽层——新增 L1 命中模型"""
        l1_total_kb = self._arch.l1_cache_kb_per_sm * self._arch.n_sms
        l1_total_bytes = l1_total_kb * 1024
        if total_bytes <= l1_total_bytes * 0.5:
            bw = self._arch.l1_bandwidth_gbps
            _dbg("GPUKernel", f"L1-resident: {total_bytes/(1024**2):.1f}MB")
        elif total_bytes <= 40 * (1024 ** 2):  # ~40MB fits L2
            bw = self._arch.l2_bandwidth_gbps
        else:
            bw = self._arch.hbm_bandwidth_gbps
        return bw

    def _roofline(self, total_ops: float, total_bytes: int,
                  bw_gbps: float) -> str:
        """Roofline 瓶颈分析"""
        if total_bytes == 0:
            return "compute"
        ai = total_ops / total_bytes  # ops/byte
        peak_flops = self._arch.fp32_tflops * 1e12  # ops/s
        bw_bytes = bw_gbps * 1e9  # bytes/s
        ridge_point = peak_flops / bw_bytes if bw_bytes > 0 else 0
        return "compute" if ai > ridge_point else "memory"

    def estimate_seq_scan(self, n_rows: int, row_size_bytes: int = 128,
                          selectivity: float = 1.0) -> KernelCostEstimate:
        total_bytes = n_rows * row_size_bytes
        bw = self._effective_bandwidth(total_bytes)
        memory_us = (total_bytes / (1024**3)) / bw * 1e6

        total_ops = n_rows * 10.0
        ops_per_us = self._arch.fp32_tflops * 1e6
        compute_us = total_ops / max(1, ops_per_us)

        max_threads = self._arch.n_sms * self._arch.warps_per_sm * 32
        threads = min(n_rows, max_threads)
        occupancy = threads / max_threads

        bottleneck = self._roofline(total_ops, total_bytes, bw)
        ai = total_ops / max(1, total_bytes)
        total_us = max(memory_us, compute_us) + 5.0

        est = KernelCostEstimate(
            op=KernelOp.SEQ_SCAN, compute_us=compute_us, memory_us=memory_us,
            threads_used=threads, sm_occupancy=occupancy,
            memory_bytes_accessed=total_bytes, bottleneck=bottleneck,
            total_us=total_us, arithmetic_intensity=ai,
        )
        self._history.append(est)
        if self._debug:
            est.debug_snapshot()
        return est

    def estimate_btree_lookup(self, n_keys: int,
                              n_lookups: int) -> KernelCostEstimate:
        height = self._btree.tree_height(n_keys)
        fanout = self._btree.fanout
        comps_per = height * math.ceil(math.log2(max(2, fanout)))
        total_comps = n_lookups * comps_per

        bytes_per = height * self._btree.node_size_bytes
        total_bytes = n_lookups * bytes_per
        bw = self._effective_bandwidth(total_bytes)
        memory_us = (total_bytes / (1024**3)) / bw * 1e6

        ops_per_us = self._arch.fp32_tflops * 1e6
        compute_us = total_comps / max(1, ops_per_us)

        max_threads = self._arch.n_sms * self._arch.warps_per_sm * 32
        threads = min(n_lookups, max_threads)
        occupancy = threads / max_threads

        bottleneck = self._roofline(total_comps, total_bytes, bw)
        ai = total_comps / max(1, total_bytes)
        total_us = max(memory_us, compute_us) + 5.0

        est = KernelCostEstimate(
            op=KernelOp.INDEX_SCAN, compute_us=compute_us, memory_us=memory_us,
            threads_used=threads, sm_occupancy=occupancy,
            memory_bytes_accessed=total_bytes, bottleneck=bottleneck,
            total_us=total_us, arithmetic_intensity=ai,
        )
        self._history.append(est)
        if self._debug:
            _dbg("GPUKernel",
                 f"BTree: {n_lookups} lookups, h={height}, fanout={fanout}")
            est.debug_snapshot()
        return est

    def estimate_hash_probe(self, n_build: int,
                            n_probe: int) -> KernelCostEstimate:
        avg_chain = self._hash.avg_chain_length()
        comparisons = n_probe * avg_chain
        bytes_per = int(avg_chain) * self._hash.bucket_size_bytes
        total_bytes = n_probe * bytes_per
        bw = self._effective_bandwidth(total_bytes)
        memory_us = (total_bytes / (1024**3)) / bw * 1e6
        compute_us = (comparisons * 10) / max(1, self._arch.fp32_tflops * 1e6)

        max_threads = self._arch.n_sms * self._arch.warps_per_sm * 32
        threads = min(n_probe, max_threads)
        occupancy = threads / max_threads
        bottleneck = self._roofline(comparisons * 10, total_bytes, bw)
        ai = (comparisons * 10) / max(1, total_bytes)
        total_us = max(memory_us, compute_us) + 5.0

        est = KernelCostEstimate(
            op=KernelOp.HASH_PROBE, compute_us=compute_us, memory_us=memory_us,
            threads_used=threads, sm_occupancy=occupancy,
            memory_bytes_accessed=total_bytes, bottleneck=bottleneck,
            total_us=total_us, arithmetic_intensity=ai,
        )
        self._history.append(est)
        if self._debug:
            est.debug_snapshot()
        return est

    def estimate_hash_build(self, n_keys: int, key_size: int = 8,
                            value_size: int = 8) -> KernelCostEstimate:
        n_buckets = self._hash.bucket_count(n_keys)
        entry_size = key_size + value_size
        total_bytes = n_keys * entry_size + n_buckets * self._hash.bucket_size_bytes
        bw = self._effective_bandwidth(total_bytes)
        memory_us = (total_bytes / (1024**3)) / bw * 1e6
        total_ops = n_keys * 20.0
        compute_us = total_ops / max(1, self._arch.fp32_tflops * 1e6)

        max_threads = self._arch.n_sms * self._arch.warps_per_sm * 32
        threads = min(n_keys, max_threads)
        occupancy = threads / max_threads
        bottleneck = self._roofline(total_ops, total_bytes, bw)
        total_us = max(memory_us, compute_us) + 5.0

        est = KernelCostEstimate(
            op=KernelOp.HASH_BUILD, compute_us=compute_us, memory_us=memory_us,
            threads_used=threads, sm_occupancy=occupancy,
            memory_bytes_accessed=total_bytes, bottleneck=bottleneck,
            total_us=total_us,
        )
        self._history.append(est)
        if self._debug:
            est.debug_snapshot()
        return est

    def compare_cpu_vs_gpu(self, op: KernelOp, n_rows: int,
                           cpu_time_us: float) -> Dict[str, Any]:
        if op == KernelOp.SEQ_SCAN:
            gpu_est = self.estimate_seq_scan(n_rows)
        elif op == KernelOp.INDEX_SCAN:
            gpu_est = self.estimate_btree_lookup(n_rows, n_rows // 10)
        elif op == KernelOp.HASH_PROBE:
            gpu_est = self.estimate_hash_probe(n_rows, n_rows // 2)
        elif op == KernelOp.HASH_BUILD:
            gpu_est = self.estimate_hash_build(n_rows)
        else:
            gpu_est = self.estimate_seq_scan(n_rows)

        data_bytes = n_rows * 128
        transfer_us = (data_bytes / (1024**3)) / 32.0 * 1e6
        gpu_total = gpu_est.total_us + transfer_us
        speedup = cpu_time_us / max(0.001, gpu_total)
        winner = "GPU" if gpu_total < cpu_time_us else "CPU"

        _dbg("GPUKernel",
             f"CPU vs GPU ({op.name}, {n_rows} rows): "
             f"cpu={cpu_time_us:.1f} gpu={gpu_total:.1f} -> {winner}")

        return {
            "op": op.name, "n_rows": n_rows,
            "cpu_time_us": cpu_time_us,
            "gpu_compute_us": gpu_est.total_us,
            "gpu_transfer_us": transfer_us,
            "gpu_total_us": gpu_total,
            "speedup": speedup, "winner": winner,
            "bottleneck": gpu_est.bottleneck,
        }

    def dump_state(self) -> str:
        lines = [
            f"=== GPUCostKernel ({self._arch.arch.name}) ===",
            f"  SMs={self._arch.n_sms} HBM={self._arch.hbm_bandwidth_gbps:.0f}GB/s",
            f"  estimates_done={len(self._history)}",
        ]
        if self._history:
            last = self._history[-1]
            lines.append(f"  last: {last.op.name} {last.total_us:.1f}us "
                         f"[{last.bottleneck}]")
        s = "\n".join(lines)
        _dbg("GPUKernel", s)
        return s
