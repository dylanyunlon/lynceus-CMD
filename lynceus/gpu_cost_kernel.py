"""
lynceus/gpu_cost_kernel.py — GPU kernel cost estimation (CUTLASS-informed).

算法改动:
    1. coalescing efficiency: row_size 非 128B 对齐时, HBM 带宽打折
    2. BTree TLB miss penalty: 树高>4 时每层多加 ~100ns
    3. occupancy: Little's Law 模型代替固定 0.92
"""
from __future__ import annotations
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum, auto

logger = logging.getLogger(__name__)

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
    def dump_debug(self, prefix: str = "") -> str:
        return (f"{prefix}GPUArch({self.arch.name}): {self.n_sms}SMs, "
                f"HBM={self.hbm_bandwidth_gbps:.0f}GB/s, FP32={self.fp32_tflops:.1f}T")

GPU_CONFIGS = {
    GPUArch.SM80_A100: GPUArchConfig(
        arch=GPUArch.SM80_A100, n_sms=108, warps_per_sm=64,
        hbm_bandwidth_gbps=2100.0, l2_bandwidth_gbps=5800.0,
        l1_bandwidth_gbps=18500.0, fp32_tflops=18.8,
        fp16_tflops=305.0, int8_tops=624.0, clock_ghz=1.41),
    GPUArch.SM89_4090: GPUArchConfig(
        arch=GPUArch.SM89_4090, n_sms=128, warps_per_sm=48,
        hbm_bandwidth_gbps=1008.0, l2_bandwidth_gbps=5000.0,
        l1_bandwidth_gbps=16000.0, fp32_tflops=82.6,
        fp16_tflops=165.2, int8_tops=330.3, clock_ghz=2.52),
    GPUArch.SM90_H100: GPUArchConfig(
        arch=GPUArch.SM90_H100, n_sms=132, warps_per_sm=64,
        hbm_bandwidth_gbps=3350.0, l2_bandwidth_gbps=12000.0,
        l1_bandwidth_gbps=33000.0, fp32_tflops=67.0,
        fp16_tflops=989.0, int8_tops=1979.0, clock_ghz=1.83),
}

class KernelOp(Enum):
    SEQ_SCAN = auto()
    INDEX_SCAN = auto()
    HASH_PROBE = auto()
    HASH_BUILD = auto()
    SORT = auto()
    JOIN_NL = auto()
    JOIN_HASH = auto()
    AGGREGATE = auto()
    GEMM = auto()

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
    load_factor: float = 0.72
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
    def dump_debug(self, prefix: str = "") -> str:
        return (f"{prefix}Kernel({self.op.name}): {self.total_us:.1f}µs "
                f"(compute={self.compute_us:.1f}, mem={self.memory_us:.1f}, "
                f"occ={self.sm_occupancy:.1%}, {self.bottleneck}-bound)")


def _coalescing_efficiency(row_size_bytes: int, warp_size: int = 32) -> float:
    """改动: 建模 GPU memory coalescing。
    完美 coalescing 要求 warp 内连续线程访问连续 128B 地址。
    当 row_size 不是 128B 对齐时, 实际需要更多 transaction。
    返回 efficiency in (0, 1].
    """
    transaction_size = 128  # GPU L1 cache line
    row_bytes_per_warp = row_size_bytes * warp_size
    # 理想: 刚好用整数个 transaction 覆盖
    ideal_transactions = math.ceil(row_bytes_per_warp / transaction_size)
    # 实际: row 不对齐时多出 padding transactions
    actual_row_span = row_size_bytes
    if actual_row_span % transaction_size != 0:
        wasted_fraction = (transaction_size - (actual_row_span % transaction_size)) / transaction_size
        # 每个 warp 浪费一些 bandwidth
        efficiency = 1.0 - wasted_fraction * 0.4  # 经验系数
    else:
        efficiency = 1.0
    return max(0.3, min(1.0, efficiency))


def _littles_law_occupancy(threads_needed: int, max_threads: int,
                           memory_latency_cycles: int = 400,
                           clock_ghz: float = 1.5) -> float:
    """改动: 用 Little's Law 估算 SM occupancy。
    原版: 固定 0.92 折扣
    新版: L = λ * W, 需要足够多 in-flight warps 来隐藏 memory latency。
    occupancy = min(1.0, warps_needed_to_hide / warps_available)
    """
    if max_threads <= 0:
        return 0.0
    raw = threads_needed / max_threads
    # 隐藏 memory latency 需要的 concurrent warps
    cycles_per_us = clock_ghz * 1000
    warps_to_hide = memory_latency_cycles / max(1, 4)  # 每 warp 约 4 cycles 调度
    warps_available = max_threads / 32
    latency_hiding = min(1.0, warps_available / max(1, warps_to_hide))
    return min(1.0, raw * latency_hiding)


class GPUCostKernel:
    def __init__(self, arch: GPUArch = GPUArch.SM80_A100, debug_print: bool = True):
        self._arch_config = GPU_CONFIGS.get(arch, GPU_CONFIGS[GPUArch.SM80_A100])
        self._btree_config = BTreeGPUConfig()
        self._hash_config = HashTableGPUConfig()
        self._debug = debug_print
        self._estimate_history: List[KernelCostEstimate] = []
        if debug_print:
            print(f"\n[gpu_cost_kernel] Initialized for {arch.name}")
            print(f"  {self._arch_config.dump_debug()}")

    def estimate_seq_scan(self, n_rows: int, row_size_bytes: int = 128,
                          selectivity: float = 1.0,
                          debug_print: Optional[bool] = None) -> KernelCostEstimate:
        """改动: coalescing efficiency 影响有效带宽。"""
        dp = debug_print if debug_print is not None else self._debug
        total_bytes = n_rows * row_size_bytes

        # 改动: coalescing
        coal_eff = _coalescing_efficiency(row_size_bytes)
        effective_bw = self._arch_config.hbm_bandwidth_gbps * coal_eff
        memory_us = (total_bytes / (1024**3)) / max(0.1, effective_bw) * 1e6

        total_ops = n_rows * 10
        ops_per_us = self._arch_config.fp32_tflops * 1e6
        compute_us = total_ops / max(1, ops_per_us)

        max_threads = self._arch_config.n_sms * self._arch_config.warps_per_sm * 32
        threads = min(n_rows, max_threads)
        occupancy = _littles_law_occupancy(threads, max_threads,
                                           clock_ghz=self._arch_config.clock_ghz)

        bottleneck = "memory" if memory_us > compute_us else "compute"
        total_us = max(memory_us, compute_us) + 5.0

        est = KernelCostEstimate(
            op=KernelOp.SEQ_SCAN, compute_us=compute_us, memory_us=memory_us,
            threads_used=threads, sm_occupancy=occupancy,
            memory_bytes_accessed=total_bytes, bottleneck=bottleneck, total_us=total_us)
        self._estimate_history.append(est)
        if dp:
            print(f"\n  [gpu_cost] SEQ_SCAN: {n_rows:,} rows, coal_eff={coal_eff:.2f}")
            print(f"    {est.dump_debug()}")
        return est

    def estimate_btree_lookup(self, n_keys: int, n_lookups: int,
                              debug_print: Optional[bool] = None) -> KernelCostEstimate:
        """改动: 树高>4时加 TLB miss penalty。"""
        dp = debug_print if debug_print is not None else self._debug
        height = self._btree_config.tree_height(n_keys)
        fanout = self._btree_config.fanout

        comparisons_per_lookup = height * math.ceil(math.log2(max(2, fanout)))
        total_comparisons = n_lookups * comparisons_per_lookup

        bytes_per_lookup = height * self._btree_config.node_size_bytes
        total_bytes = n_lookups * bytes_per_lookup
        memory_us = (total_bytes / (1024**3)) / self._arch_config.l2_bandwidth_gbps * 1e6

        # 改动: TLB miss penalty for deep trees
        # GPU TLB 层级浅 (~128 entries L1 TLB), 深树跳转跨越更多 page
        if height > 4:
            extra_levels = height - 4
            # 每个 lookup 每多一层 ~100ns TLB miss (在大量 lookup 上摊销)
            tlb_penalty_per_lookup = extra_levels * 0.1  # 0.1 µs = 100ns
            memory_us += n_lookups * tlb_penalty_per_lookup

        ops_per_us = self._arch_config.fp32_tflops * 1e6
        compute_us = total_comparisons / max(1, ops_per_us)

        max_threads = self._arch_config.n_sms * self._arch_config.warps_per_sm * 32
        threads = min(n_lookups, max_threads)
        occupancy = _littles_law_occupancy(threads, max_threads,
                                           memory_latency_cycles=600,  # random access
                                           clock_ghz=self._arch_config.clock_ghz)

        bottleneck = "memory" if memory_us > compute_us else "compute"
        total_us = max(memory_us, compute_us) + 5.0

        est = KernelCostEstimate(
            op=KernelOp.INDEX_SCAN, compute_us=compute_us, memory_us=memory_us,
            threads_used=threads, sm_occupancy=occupancy,
            memory_bytes_accessed=total_bytes, bottleneck=bottleneck, total_us=total_us)
        self._estimate_history.append(est)
        if dp:
            print(f"\n  [gpu_cost] BTREE: {n_lookups:,} lookups, height={height}, fanout={fanout}")
            print(f"    {est.dump_debug()}")
        return est

    def estimate_hash_probe(self, n_build: int, n_probe: int,
                            debug_print: Optional[bool] = None) -> KernelCostEstimate:
        dp = debug_print if debug_print is not None else self._debug
        n_buckets = self._hash_config.bucket_count(n_build)
        avg_chain = self._hash_config.avg_chain_length()
        comparisons = n_probe * avg_chain
        bytes_per_probe = int(avg_chain) * self._hash_config.bucket_size_bytes
        total_bytes = n_probe * bytes_per_probe
        memory_us = (total_bytes / (1024**3)) / self._arch_config.l2_bandwidth_gbps * 1e6
        ops_per_us = self._arch_config.fp32_tflops * 1e6
        compute_us = (comparisons * 10) / max(1, ops_per_us)
        max_threads = self._arch_config.n_sms * self._arch_config.warps_per_sm * 32
        threads = min(n_probe, max_threads)
        occupancy = _littles_law_occupancy(threads, max_threads,
                                           clock_ghz=self._arch_config.clock_ghz)
        bottleneck = "memory" if memory_us > compute_us else "compute"
        total_us = max(memory_us, compute_us) + 5.0
        est = KernelCostEstimate(
            op=KernelOp.HASH_PROBE, compute_us=compute_us, memory_us=memory_us,
            threads_used=threads, sm_occupancy=occupancy,
            memory_bytes_accessed=total_bytes, bottleneck=bottleneck, total_us=total_us)
        self._estimate_history.append(est)
        if dp:
            print(f"\n  [gpu_cost] HASH_PROBE: {n_probe:,} probes, chain={avg_chain:.1f}")
            print(f"    {est.dump_debug()}")
        return est

    def estimate_hash_build(self, n_keys: int, key_size: int = 8, value_size: int = 8,
                            debug_print: Optional[bool] = None) -> KernelCostEstimate:
        dp = debug_print if debug_print is not None else self._debug
        n_buckets = self._hash_config.bucket_count(n_keys)
        entry_size = key_size + value_size
        total_bytes = n_keys * entry_size + n_buckets * self._hash_config.bucket_size_bytes
        memory_us = (total_bytes / (1024**3)) / self._arch_config.hbm_bandwidth_gbps * 1e6
        compute_us = (n_keys * 20) / max(1, self._arch_config.fp32_tflops * 1e6)
        max_threads = self._arch_config.n_sms * self._arch_config.warps_per_sm * 32
        threads = min(n_keys, max_threads)
        occupancy = _littles_law_occupancy(threads, max_threads,
                                           clock_ghz=self._arch_config.clock_ghz)
        bottleneck = "memory" if memory_us > compute_us else "compute"
        total_us = max(memory_us, compute_us) + 5.0
        est = KernelCostEstimate(
            op=KernelOp.HASH_BUILD, compute_us=compute_us, memory_us=memory_us,
            threads_used=threads, sm_occupancy=occupancy,
            memory_bytes_accessed=total_bytes, bottleneck=bottleneck, total_us=total_us)
        self._estimate_history.append(est)
        if dp:
            print(f"\n  [gpu_cost] HASH_BUILD: {n_keys:,} keys")
            print(f"    {est.dump_debug()}")
        return est

    def compare_cpu_vs_gpu(self, op: KernelOp, n_rows: int, cpu_time_us: float,
                           debug_print: bool = True) -> Dict[str, Any]:
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
        data_bytes = n_rows * 128
        transfer_us = (data_bytes / (1024**3)) / 32.0 * 1e6
        gpu_total = gpu_est.total_us + transfer_us
        speedup = cpu_time_us / max(0.001, gpu_total)
        winner = "GPU" if gpu_total < cpu_time_us else "CPU"
        result = {"op": op.name, "n_rows": n_rows, "cpu_time_us": cpu_time_us,
                  "gpu_compute_us": gpu_est.total_us, "gpu_transfer_us": transfer_us,
                  "gpu_total_us": gpu_total, "speedup": speedup, "winner": winner,
                  "bottleneck": gpu_est.bottleneck}
        if debug_print:
            print(f"\n  [gpu_cost] {op.name}: CPU={cpu_time_us:.1f}µs, "
                  f"GPU={gpu_total:.1f}µs (xfer={transfer_us:.1f}), → {winner}")
        return result

    def dump_state(self) -> str:
        lines = [
            f"GPUCostKernel: {self._arch_config.arch.name}",
            f"  estimates_done = {len(self._estimate_history)}",
        ]
        if self._estimate_history:
            last = self._estimate_history[-1]
            lines.append(f"  last = {last.dump_debug()}")
        return "\n".join(lines)
