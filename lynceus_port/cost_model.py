"""
lynceus/cost_model.py — Heterogeneous cost model for query routing.

This is the central decision engine. Given a query descriptor and hardware
topology, it estimates execution cost on each available device and recommends
the optimal routing.

Architecture references:
    - PAR2QO get_plan_cost() (par2qo/code/postgres.py:110)
      → foundation for plan-level cost estimation
    - VIDEX VidexModelBase.scan_time() (videx/src/.../videx_strategy.py)
      → virtual index cost model abstraction
    - CUTLASS GemmUniversal (cutlass/include/cutlass/gemm/kernel/gemm_universal.h:65)
      → GPU compute cost modeling (tile-level GEMM throughput)
    - DeepSeek act_quant_kernel (DeepSeek-V3/inference/kernel.py)
      → FP8 quantized statistics for compact cost tables
    - Megatron DistributedOptimizer (Megatron-LM/megatron/core/optimizer/distrib_optimizer.py:102)
      → distributed parameter update cost model
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from .schema import (
    HardwareNode,
    HardwareTopology,
    HardwareKind,
    RoutingStrategy,
    TopologyEdge,
)


# ---------------------------------------------------------------------------
# Query descriptor
# ---------------------------------------------------------------------------

class QueryType(Enum):
    POINT_LOOKUP = auto()
    RANGE_SCAN = auto()
    FULL_TABLE_SCAN = auto()
    INDEX_SCAN = auto()
    JOIN = auto()
    AGGREGATE = auto()
    SORT = auto()


@dataclass
class QueryDescriptor:
    """Describes a single query's characteristics for cost estimation.

    Inspired by PAR2QO's parametric representation of queries where each
    query is described by its cardinality estimates and plan structure.
    """
    query_id: str
    query_type: QueryType
    estimated_rows: int = 0
    estimated_width_bytes: int = 100
    num_predicates: int = 1
    selectivity: float = 1.0         # fraction of table accessed
    table_rows: int = 1_000_000
    index_available: bool = False
    index_depth: int = 3             # B-tree depth
    num_joins: int = 0
    sort_required: bool = False
    group_by_cardinality: int = 0
    table_name: str = ""             # logical table identity (for cache keying);
                                     # empty => fall back to query_id stem.

    def __post_init__(self):
        """Validate that estimated_rows and other fields are non-negative."""
        if self.estimated_rows < 0:
            raise ValueError(f"estimated_rows must be >= 0, got {self.estimated_rows}")
        if self.table_rows < 0:
            raise ValueError(f"table_rows must be >= 0, got {self.table_rows}")
        if not (0.0 <= self.selectivity <= 1.0):
            raise ValueError(f"selectivity must be in [0, 1], got {self.selectivity}")
        if self.estimated_width_bytes < 0:
            raise ValueError(f"estimated_width_bytes must be >= 0, got {self.estimated_width_bytes}")

    @property
    def estimated_data_bytes(self) -> int:
        return self.estimated_rows * self.estimated_width_bytes

    @property
    def full_table_bytes(self) -> int:
        # [PORT·COS] 断点: 返回值检查
        return self.table_rows * self.estimated_width_bytes


# ---------------------------------------------------------------------------
# Cost breakdown
# ---------------------------------------------------------------------------

@dataclass
class CostBreakdown:
    """Itemized cost estimate for a query on a specific device.

    Units: microseconds (to match NCCL latency_us convention).
    """
    device_id: str
    io_cost_us: float = 0.0
    compute_cost_us: float = 0.0
    transfer_cost_us: float = 0.0
    index_cost_us: float = 0.0
    sort_cost_us: float = 0.0

    @property
    def total_us(self) -> float:
        return (self.io_cost_us + self.compute_cost_us +
                self.transfer_cost_us + self.index_cost_us +
                self.sort_cost_us)

    @property
    def total_ms(self) -> float:
        return self.total_us / 1000.0


# ---------------------------------------------------------------------------
# Device-specific cost models
# ---------------------------------------------------------------------------

class CPUCostModel:
    """Cost model for CPU-side query execution.

    Inspired by PostgreSQL's cost model (seq_page_cost, random_page_cost,
    cpu_tuple_cost, cpu_operator_cost) as used in PAR2QO
    get_plan_cost_simple() (par2qo/code/postgres.py:81).

    IMPORTANT: All costs are in MICROSECONDS for dimensional consistency
    with GPUCostModel. PostgreSQL uses abstract "cost units"; we convert
    to µs using empirical measurements from modern server hardware:
      - DDR5 sequential bandwidth: ~50 GB/s → 8KB page ≈ 0.16 µs
      - L3 cache random access: ~30-100 ns
      - DRAM random access (NUMA): ~100-300 ns
      - NVMe SSD random 4K read: ~10-100 µs
    We model a warm buffer pool (data in DRAM/L3).
    """

    # Warm-cache microsecond costs per 8KB page
    SEQ_PAGE_COST: float = 0.024      # sequential page read (prefetched, ~20ns)
    RANDOM_PAGE_COST: float = 0.58    # random page read (~500ns DRAM access)
    CPU_TUPLE_COST: float = 0.044     # per-tuple processing (~50ns)
    CPU_OPERATOR_COST: float = 0.012  # per-predicate evaluation (~10ns)
    CPU_INDEX_TUPLE_COST: float = 0.018  # per-index-tuple fetch (~20ns)
    PAGE_SIZE: int = 8192

    # -- PORT: NUMA局部性惩罚因子 (远端socket多~40%延迟) --
    NUMA_REMOTE_PENALTY: float = 1.38

    def estimate(self, query: QueryDescriptor,
                 node: HardwareNode) -> CostBreakdown:
        from ._debug import dbg, tracer, runtime_stats
        tracer.enter("CPUCostModel.estimate", node=node.node_id, qtype=query.query_type.name)
        cb = CostBreakdown(device_id=node.node_id)

        pages = max(1, query.estimated_data_bytes // self.PAGE_SIZE)
        total_pages = max(1, query.full_table_bytes // self.PAGE_SIZE)

        # PORT: NUMA局部性建模 — 根据node_id推断socket位置
        # 偶数CPU本地访问, 奇数CPU交叉socket需要QPI惩罚
        numa_factor = 1.0
        if hasattr(node, 'node_id') and node.node_id.endswith('1'):
            numa_factor = self.NUMA_REMOTE_PENALTY

        # I/O cost — PORT: 引入硬件prefetcher流水线模型
        # 连续读>=8页时prefetcher生效, 有效带宽提升~2.5x
        if query.query_type == QueryType.FULL_TABLE_SCAN:
            if total_pages >= 8:
                # prefetch流水线: 前8页冷启动 + 剩余页受益于预取
                cold_pages = min(8, total_pages)
                warm_pages = total_pages - cold_pages
                prefetch_gain = 2.5  # HW prefetcher带来的有效加速
                cb.io_cost_us = (
                    cold_pages * self.SEQ_PAGE_COST * node.scan_cost_per_row
                    + warm_pages * self.SEQ_PAGE_COST * node.scan_cost_per_row / prefetch_gain
                ) * numa_factor
            else:
                cb.io_cost_us = total_pages * self.SEQ_PAGE_COST * max(1e-6, node.scan_cost_per_row)
                cb.io_cost_us *= numa_factor
        elif query.index_available and query.query_type in (
            QueryType.POINT_LOOKUP, QueryType.INDEX_SCAN, QueryType.RANGE_SCAN
        ):
            # PORT: B-tree节点在L3的驻留概率与树高/访问频率相关
            # 浅层节点(root, level-1)几乎必中L3, 深层随机
            index_pages = query.index_depth + max(1, int(pages * query.selectivity))
            l3_resident_levels = min(2, query.index_depth)  # 前2层驻留L3
            random_levels = max(0, query.index_depth - l3_resident_levels)
            index_io = (
                l3_resident_levels * 0.035  # L3 hit: ~35ns
                + random_levels * self.RANDOM_PAGE_COST  # DRAM random
                + max(1, int(pages * query.selectivity)) * self.RANDOM_PAGE_COST
            )
            cb.index_cost_us = (index_io + query.estimated_rows * self.CPU_INDEX_TUPLE_COST) * numa_factor
            cb.io_cost_us = pages * query.selectivity * self.SEQ_PAGE_COST * numa_factor
        else:
            cb.io_cost_us = pages * self.SEQ_PAGE_COST * numa_factor

        # CPU compute cost — PORT: 谓词求值加入短路概率
        # 多谓词AND: 第一个谓词过滤掉一部分行, 后续谓词的平均行数递减
        if query.num_predicates > 1:
            # 假设每个谓词独立过滤掉 selectivity^(1/n_pred) 的行
            per_pred_pass = query.selectivity ** (1.0 / query.num_predicates)
            effective_rows = float(query.estimated_rows)
            total_compute = 0.0
            for p_idx in range(query.num_predicates):
                total_compute += effective_rows * self.CPU_OPERATOR_COST
                effective_rows *= per_pred_pass
            cb.compute_cost_us = query.estimated_rows * self.CPU_TUPLE_COST + total_compute
        else:
            cb.compute_cost_us = (
                query.estimated_rows * self.CPU_TUPLE_COST
                + query.estimated_rows * self.CPU_OPERATOR_COST
            )

        # Sort — PORT: 自适应排序选择 (小集合用插入排序, 大集合用归并)
        if query.sort_required and query.estimated_rows > 1:
            n = query.estimated_rows
            if n <= 64:
                # 插入排序: O(n^2) 但缓存友好, 小集合实际更快
                cb.sort_cost_us = 0.3 * n * n * self.CPU_OPERATOR_COST
            else:
                # 归并排序: O(n log n), 加cache-line跨步惩罚
                cache_lines_touched = max(1, (n * query.estimated_width_bytes) // 64)
                cache_miss_penalty = 0.0
                if cache_lines_touched > 512:  # L1溢出
                    cache_miss_penalty = 0.004 * (cache_lines_touched - 512)
                cb.sort_cost_us = (
                    2.0 * n * math.log2(max(2, n)) * self.CPU_OPERATOR_COST
                    + cache_miss_penalty
                )

        # Scale by node capacity
        if node.compute_capacity > 0 and node.compute_capacity != 1.0:
            scale = 1.0 / node.compute_capacity
            cb.compute_cost_us *= scale
            cb.sort_cost_us *= scale

        # PORT: 诊断打印 — 打印完整cost breakdown供断点检查
        runtime_stats.incr("cpu_estimates")
        dbg("COS·cpu_estimate_done",
            device=node.node_id, io=cb.io_cost_us, compute=cb.compute_cost_us,
            index=cb.index_cost_us, sort=cb.sort_cost_us, total=cb.total_us,
            numa=numa_factor)
        tracer.exit(f"total={cb.total_us:.1f}µs")
        return cb


class GPUCostModel:
    """Cost model for GPU-accelerated query execution.

    Inspired by:
    - CUTLASS GemmUniversal tile scheduling: GPU throughput is modeled as
      number of tiles × cycles_per_tile, where tile dimensions come from
      the GEMM kernel configuration.
    - DeepSeek act_quant_kernel: FP8 quantization for statistics storage
      means we can keep per-column stats in GPU HBM cheaply.
    - vLLM PagedAttention block management: paged memory for index cache.

    Key insight: GPU excels at bulk-parallel operations (full scans,
    hash joins, sorts on large datasets) but suffers from kernel launch
    overhead and PCIe transfer latency for small queries.
    """

    KERNEL_LAUNCH_OVERHEAD_US: float = 8.5
    GPU_TUPLE_COST: float = 0.000085    # ~100x faster than CPU per tuple
    GPU_OPERATOR_COST: float = 0.000058
    HBM_BANDWIDTH_GB_S: float = 2150.0  # A100-class HBM bandwidth
    PCIE_BANDWIDTH_GB_S: float = 31.0    # PCIe Gen4 x16

    def estimate(self, query: QueryDescriptor,
                 node: HardwareNode,
                 data_resident_on_gpu: bool = False) -> CostBreakdown:
        from ._debug import dbg, tracer, runtime_stats
        tracer.enter("GPUCostModel.estimate", node=node.node_id,
                     rows=query.estimated_rows, resident=data_resident_on_gpu)
        cb = CostBreakdown(device_id=node.node_id)

        # PCIe transfer — PORT: 分段DMA模型, 小传输受latency主导
        if not data_resident_on_gpu:
            transfer_bytes = query.estimated_data_bytes
            # PCIe有固定握手开销 ~2µs, 小payload被latency主导
            pcie_latency_us = 2.0
            throughput_us = transfer_bytes / (self.PCIE_BANDWIDTH_GB_S * 1e3)  # bytes→µs
            cb.transfer_cost_us = pcie_latency_us + throughput_us

        # Kernel launch — PORT: 多操作融合时只付一次launch开销
        # 单谓词scan可以fuse成一个kernel; 多谓词需要多pass或者用shared mem
        n_kernels = 1
        if query.num_predicates > 3:
            n_kernels = 1 + (query.num_predicates - 3) // 4  # 每4谓词多一个kernel
        kernel_launch_us = self.KERNEL_LAUNCH_OVERHEAD_US * n_kernels

        # Memory throughput — PORT: 三级带宽模型 (L2 / HBM / L2+HBM混合)
        data_bytes = query.estimated_data_bytes
        l2_capacity = 40 * 1024**2  # A100: 40MB L2
        if data_bytes <= l2_capacity:
            # 全部命中L2, 带宽约 5TB/s (A100实测)
            eff_bw_gbs = 5000.0
        elif data_bytes <= l2_capacity * 4:
            # 部分命中, 混合带宽
            l2_frac = l2_capacity / data_bytes
            eff_bw_gbs = 5000.0 * l2_frac + self.HBM_BANDWIDTH_GB_S * (1.0 - l2_frac)
        else:
            # 纯HBM, 但大数据集有TLB miss惩罚
            tlb_penalty = 1.0 + 0.05 * math.log2(max(1, data_bytes / (256 * 1024**2)))
            eff_bw_gbs = self.HBM_BANDWIDTH_GB_S / tlb_penalty

        hbm_us = (data_bytes / (eff_bw_gbs * 1e9)) * 1e6

        # Compute — PORT: warp occupancy-aware模型
        # SM数量和warp调度效率影响实际吞吐
        num_sms = 108  # A100
        warps_per_sm = 64
        threads_per_warp = 32
        max_active_threads = num_sms * warps_per_sm * threads_per_warp
        actual_threads = min(query.estimated_rows, max_active_threads)
        occupancy = actual_threads / max_active_threads

        # 低占用率下SM利用不满, 实际吞吐打折
        occupancy_eff = occupancy ** 0.7  # 亚线性: 50%占用率给~62%吞吐

        compute_us = (
            query.estimated_rows * self.GPU_TUPLE_COST
            + query.estimated_rows * query.num_predicates * self.GPU_OPERATOR_COST
        )
        if occupancy_eff > 0:
            compute_us /= occupancy_eff

        # roofline: 取memory/compute的瓶颈
        scalable_compute_us = max(hbm_us, compute_us)

        # Sort — PORT: radix sort (O(nk)) vs bitonic (O(n log²n)), 按数据量选
        if query.sort_required and query.estimated_rows > 1:
            n = query.estimated_rows
            if n >= 100_000:
                # 大集合: GPU radix sort, 8-bit pass × ceil(key_bits/8)
                key_bits = min(64, query.estimated_width_bytes * 8)
                n_passes = max(1, (key_bits + 7) // 8)
                ops = n * n_passes * 4  # 每pass: histogram+scatter ≈ 4 ops/elem
                cb.sort_cost_us = ops * self.GPU_OPERATOR_COST / num_sms
            else:
                # 小集合: bitonic sort
                log_n = math.log2(max(2, n))
                ops = n * (log_n ** 2)
                warp_divergence = 1.0 + 0.12 * log_n  # bitonic的warp分化
                cb.sort_cost_us = ops * self.GPU_OPERATOR_COST * warp_divergence / num_sms

        # Scale compute (不scale kernel launch, 它是固定的)
        if node.compute_capacity > 0 and node.compute_capacity != 1.0:
            scale = 1.0 / node.compute_capacity
            scalable_compute_us *= scale
            cb.sort_cost_us *= scale

        cb.compute_cost_us = kernel_launch_us + scalable_compute_us

        # PORT: 全量诊断
        runtime_stats.incr("gpu_estimates")
        dbg("COS·gpu_estimate_done",
            device=node.node_id, transfer=cb.transfer_cost_us,
            compute=cb.compute_cost_us, sort=cb.sort_cost_us,
            total=cb.total_us, occupancy=occupancy, n_kernels=n_kernels)
        tracer.exit(f"total={cb.total_us:.1f}µs occ={occupancy:.2f}")
        return cb


# ---------------------------------------------------------------------------
# Unified cost model
# ---------------------------------------------------------------------------

class CostModelEngine:
    """Unified cost model that estimates query cost across all devices
    in the hardware topology and recommends routing.

    Inspired by:
    - Megatron's pipeline scheduler choosing forward/backward device placement
      (forward_backward_pipelining_with_interleaving, schedules.py:896)
    - NCCL's ncclTopoCompute choosing optimal communication topology
      (nccl/src/graph/search.cc:1023)
    - VIDEX's strategy-based cost model selection (VidexStrategy enum)
    """

    def __init__(self, topology: HardwareTopology):
        self.topology = topology
        self.cpu_model = CPUCostModel()
        self.gpu_model = GPUCostModel()
        self._cache: Dict[str, CostBreakdown] = {}

    def estimate_on_device(self, query: QueryDescriptor,
                           device_id: str,
                           data_location: Optional[str] = None
                           ) -> CostBreakdown:
        """Estimate cost of executing query on a specific device."""
        from ._debug import dbg
        dbg('CostModel.estimate', query_id=query.query_id, device=device_id, qtype=query.query_type.name)
        node = self.topology.get_node(device_id)
        if node is None:
            raise ValueError(f"Unknown device: {device_id}")

        if node.kind == HardwareKind.GPU:
            data_resident = (data_location == device_id)
            cb = self.gpu_model.estimate(query, node, data_resident)
            # Add transfer cost from data location
            if data_location and not data_resident:
                cb.transfer_cost_us = self.topology.get_transfer_cost(
                    data_location, device_id, query.estimated_data_bytes
                )
        elif node.kind == HardwareKind.CPU:
            cb = self.cpu_model.estimate(query, node)
            if data_location and data_location != device_id:
                cb.transfer_cost_us = self.topology.get_transfer_cost(
                    data_location, device_id, query.estimated_data_bytes
                )
        else:
            raise ValueError(f"Unsupported device kind: {node.kind}")

        # [PORT·COS] 断点: 返回值检查
        return cb

    def estimate_all_devices(self, query: QueryDescriptor,
                             data_location: Optional[str] = None
                             ) -> Dict[str, CostBreakdown]:
        """Estimate cost on every device in the topology."""
        results = {}
        for node_id, node in self.topology.nodes.items():
            if node.kind in (HardwareKind.GPU, HardwareKind.CPU):
                try:
                    results[node_id] = self.estimate_on_device(
                        query, node_id, data_location
                    )
                except ValueError:
                    continue
        # [PORT·COS] 断点: 返回值检查
        return results

    def recommend(self, query: QueryDescriptor,
                  data_location: Optional[str] = None
                  ) -> Tuple[str, CostBreakdown]:
        """Recommend the optimal device for this query.

        Returns (device_id, cost_breakdown) with the lowest total cost.

        This is the core routing decision — analogous to NCCL's
        ncclTopoCompute choosing the best algorithm/protocol path.
        """
        estimates = self.estimate_all_devices(query, data_location)
        if not estimates:
            raise RuntimeError("No devices available for estimation")

        from ._debug import dbg
        dbg('CostModel.recommend_result', query_id=query.query_id, n_candidates=len(estimates),
            costs={k: v.total_ms for k, v in estimates.items()})
        best_id = min(estimates, key=lambda k: estimates[k].total_us)
        return best_id, estimates[best_id]

    def route_batch(self, queries: List[QueryDescriptor],
                    data_location: Optional[str] = None
                    ) -> List[Tuple[str, CostBreakdown]]:
        """Route a batch of queries. Returns per-query (device_id, cost)."""
        return [self.recommend(q, data_location) for q in queries]


# ---------------------------------------------------------------------------
# Default topology factory
# ---------------------------------------------------------------------------

def create_default_topology() -> HardwareTopology:
    """Create a typical single-node heterogeneous topology:
    2x CPU sockets + 4x GPUs connected via PCIe/NVLink.

    Inspired by NCCL's ncclTopoFillGpu/ncclTopoFillNet pattern.
    """
    topo = HardwareTopology()

    # CPU nodes
    cpu0 = HardwareNode(
        node_id="cpu0", kind=HardwareKind.CPU,
        compute_capacity=1.0,
        memory_bytes=256 * (1 << 30),  # 256 GB
        scan_cost_per_row=1.0,
        seek_cost=4.0,
        compute_cost_per_op=0.01,
    )
    cpu1 = HardwareNode(
        node_id="cpu1", kind=HardwareKind.CPU,
        compute_capacity=1.0,
        memory_bytes=256 * (1 << 30),
        scan_cost_per_row=1.0,
        seek_cost=4.0,
        compute_cost_per_op=0.01,
    )

    # GPU nodes (A100-class)
    gpus = []
    for i in range(4):
        gpu = HardwareNode(
            node_id=f"gpu{i}", kind=HardwareKind.GPU,
            compute_capacity=110.0,  # ~100x CPU FLOPS
            memory_bytes=80 * (1 << 30),  # 80 GB HBM
            bandwidth_gbps=2000.0,   # HBM bandwidth
            scan_cost_per_row=0.001,
            seek_cost=0.01,
            compute_cost_per_op=0.0001,
        )
        gpus.append(gpu)

    for n in [cpu0, cpu1] + gpus:
        topo.add_node(n)

    # PCIe edges: CPU ↔ GPU
    for gpu in gpus:
        topo.add_edge(TopologyEdge(
            src="cpu0", dst=gpu.node_id,
            bandwidth_gbps=32.0, latency_us=1.0,
            link_type=HardwareKind.PCIE,
        ))
        topo.add_edge(TopologyEdge(
            src=gpu.node_id, dst="cpu0",
            bandwidth_gbps=32.0, latency_us=1.0,
            link_type=HardwareKind.PCIE,
        ))

    # NVLink edges: GPU ↔ GPU (mesh)
    for i, g1 in enumerate(gpus):
        for j, g2 in enumerate(gpus):
            # [PORT·COS] 循环迭代: j
            if i != j:
                topo.add_edge(TopologyEdge(
                    src=g1.node_id, dst=g2.node_id,
                    bandwidth_gbps=600.0, latency_us=0.5,
                    link_type=HardwareKind.NVLINK,
                ))

    # CPU ↔ CPU (QPI/UPI)
    topo.add_edge(TopologyEdge(
        src="cpu0", dst="cpu1",
        bandwidth_gbps=50.0, latency_us=0.3,
        link_type=HardwareKind.NETWORK,
    ))
    topo.add_edge(TopologyEdge(
        src="cpu1", dst="cpu0",
        bandwidth_gbps=50.0, latency_us=0.3,
        link_type=HardwareKind.NETWORK,
    ))

    return topo

