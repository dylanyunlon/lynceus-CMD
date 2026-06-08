"""
lynceus/costing.py — Heterogeneous cost model for query routing.

Architecture references:
    - PAR2QO get_plan_cost(), VIDEX scan_time(), CUTLASS GemmUniversal,
      DeepSeek act_quant_kernel, Megatron DistributedOptimizer
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from .schema import (
    HardwareNode, HardwareTopology, HardwareKind,
    RoutingStrategy, TopologyEdge,
)


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
    query_id: str
    query_type: QueryType
    estimated_rows: int = 0
    estimated_width_bytes: int = 100
    num_predicates: int = 1
    selectivity: float = 1.0
    table_rows: int = 1_000_000
    index_available: bool = False
    index_depth: int = 3
    num_joins: int = 0
    sort_required: bool = False
    group_by_cardinality: int = 0
    table_name: str = ""

    def __post_init__(self):
        if self.estimated_rows < 0:
            raise ValueError(f"estimated_rows must be >= 0, got {self.estimated_rows}")
        if self.table_rows < 0:
            raise ValueError(f"table_rows must be >= 0, got {self.table_rows}")
        if not (0.0 <= self.selectivity <= 1.0):
            raise ValueError(f"selectivity must be in [0, 1], got {self.selectivity}")
        if self.estimated_width_bytes < 0:
            raise ValueError(f"estimated_width_bytes must be >= 0")

    @property
    def estimated_data_bytes(self) -> int:
        return self.estimated_rows * self.estimated_width_bytes

    @property
    def full_table_bytes(self) -> int:
        return self.table_rows * self.estimated_width_bytes


@dataclass
class CostBreakdown:
    """Itemized cost estimate. Units: microseconds."""
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


class CPUCostModel:
    """Cost model for CPU-side query execution.

    改动: IO cost 引入 NUMA-locality 衰减因子,
    对多谓词评估用 short-circuit 概率递减模型代替线性叠加。
    """
    SEQ_PAGE_COST: float = 0.024
    RANDOM_PAGE_COST: float = 0.58
    CPU_TUPLE_COST: float = 0.044
    CPU_OPERATOR_COST: float = 0.012
    CPU_INDEX_TUPLE_COST: float = 0.018
    PAGE_SIZE: int = 8192
    # NUMA locality decay: 远端内存访问多 ~40% 延迟
    NUMA_PENALTY: float = 1.38

    def estimate(self, query: QueryDescriptor,
                 node: HardwareNode) -> CostBreakdown:
        from ._debug import dbg, checkpoint
        cb = CostBreakdown(device_id=node.node_id)

        pages = max(1, query.estimated_data_bytes // self.PAGE_SIZE)
        total_pages = max(1, query.full_table_bytes // self.PAGE_SIZE)

        if query.query_type == QueryType.FULL_TABLE_SCAN:
            base_io = total_pages * self.SEQ_PAGE_COST
            # NUMA衰减: 大表扫描可能跨NUMA, 用sqrt衰减模拟部分命中
            numa_factor = 1.0 + (self.NUMA_PENALTY - 1.0) * min(1.0, total_pages / 50000)
            cb.io_cost_us = base_io * node.scan_cost_per_row * numa_factor
            if node.scan_cost_per_row == 0:
                cb.io_cost_us = base_io * numa_factor
        elif query.index_available and query.query_type in (
            QueryType.POINT_LOOKUP, QueryType.INDEX_SCAN, QueryType.RANGE_SCAN
        ):
            index_pages = query.index_depth + max(1, int(
                pages * query.selectivity
            ))
            cb.index_cost_us = (index_pages * self.RANDOM_PAGE_COST +
                                query.estimated_rows * self.CPU_INDEX_TUPLE_COST)
            cb.io_cost_us = pages * query.selectivity * self.SEQ_PAGE_COST
        else:
            cb.io_cost_us = pages * self.SEQ_PAGE_COST

        # 多谓词短路概率递减: 每个后续谓词只在前面通过时才评估
        # 期望评估次数 = rows * (1 + sel + sel^2 + ... + sel^(p-1))
        # 用几何级数代替原版的 rows * num_predicates 线性叠加
        p = query.num_predicates
        sel = query.selectivity
        if sel < 1.0 and p > 1:
            geom_sum = (1.0 - sel ** p) / (1.0 - sel)
        else:
            geom_sum = float(p)
        cb.compute_cost_us = (
            query.estimated_rows * self.CPU_TUPLE_COST +
            query.estimated_rows * geom_sum * self.CPU_OPERATOR_COST
        )

        if query.sort_required and query.estimated_rows > 1:
            n = query.estimated_rows
            # Knuth TAOCP Vol.3: optimal 3-way mergesort = 1.39·n·ln(n) comparisons
            # Plus cache-miss penalty: when data > L3 (30MB), each merge pass
            # touches O(n) pages, incurring ~0.1µs TLB miss per 4KB page
            ln_n = math.log(max(2, n))
            comparisons = 1.39 * n * ln_n
            data_bytes = n * max(query.estimated_width_bytes, 8)
            l3_bytes = 30 * 1024 * 1024  # 30MB L3
            cache_miss_factor = 1.0 + 0.25 * max(0, data_bytes - l3_bytes) / l3_bytes
            cb.sort_cost_us = (
                comparisons * self.CPU_OPERATOR_COST * cache_miss_factor
                + 0.002 * n  # memory write-back
            )

        if node.compute_capacity > 0 and node.compute_capacity != 1.0:
            scale = 1.0 / node.compute_capacity
            cb.compute_cost_us *= scale
            cb.sort_cost_us *= scale

        checkpoint("cpu_cost_done", query_id=query.query_id,
                   io=cb.io_cost_us, compute=cb.compute_cost_us,
                   index=cb.index_cost_us, sort=cb.sort_cost_us,
                   total_us=cb.total_us)
        return cb


class GPUCostModel:
    """Cost model for GPU-accelerated query execution.

    改动: L2 cache 命中率用 sigmoid 曲线代替线性截断,
    kernel launch overhead 区分 cold/warm launch。
    """
    KERNEL_LAUNCH_COLD_US: float = 12.0   # 首次 launch 或 cache miss
    KERNEL_LAUNCH_WARM_US: float = 5.5    # 热路径 launch
    GPU_TUPLE_COST: float = 0.000085
    GPU_OPERATOR_COST: float = 0.000058
    HBM_BANDWIDTH_GB_S: float = 2150.0
    PCIE_BANDWIDTH_GB_S: float = 31.0
    _launch_count: int = 0  # 跟踪 warm/cold

    def estimate(self, query: QueryDescriptor,
                 node: HardwareNode,
                 data_resident_on_gpu: bool = False) -> CostBreakdown:
        from ._debug import checkpoint
        cb = CostBreakdown(device_id=node.node_id)

        if not data_resident_on_gpu:
            transfer_bytes = query.estimated_data_bytes
            transfer_seconds = transfer_bytes / (self.PCIE_BANDWIDTH_GB_S * 1e9)
            cb.transfer_cost_us = transfer_seconds * 1e6

        # cold/warm kernel launch
        self._launch_count += 1
        if self._launch_count <= 3:
            kernel_launch_us = self.KERNEL_LAUNCH_COLD_US
        else:
            kernel_launch_us = self.KERNEL_LAUNCH_WARM_US

        data_bytes = query.estimated_data_bytes
        # L2命中率用 sigmoid: 40MB L2, 中心点=20MB, 陡度k=0.15
        l2_size = 40 * 1024**2
        sigmoid_x = (l2_size / 2 - data_bytes) / (l2_size / 6)
        l2_hit = 1.0 / (1.0 + math.exp(-sigmoid_x)) if abs(sigmoid_x) < 20 else (1.0 if sigmoid_x > 0 else 0.0)

        eff_bw = self.HBM_BANDWIDTH_GB_S * (1.0 + 1.8 * l2_hit)
        hbm_us = (data_bytes / (eff_bw * 1e9)) * 1e6

        compute_us = (
            query.estimated_rows * self.GPU_TUPLE_COST +
            query.estimated_rows * query.num_predicates * self.GPU_OPERATOR_COST
        )

        scalable_compute_us = max(hbm_us, compute_us)

        if query.sort_required and query.estimated_rows > 1:
            n = query.estimated_rows
            num_sms = 108
            # CUB DeviceRadixSort: O(n · passes / P) where passes = key_bits / radix_bits
            # Typical: 32-bit keys, radix=8 → 4 passes; 64-bit → 8 passes
            key_bits = 64  # assume 64-bit sort keys
            radix_bits = 8  # CUB uses 8-bit radix
            passes = key_bits // radix_bits
            # Each pass: scatter n elements across 2^radix_bits bins
            scatter_cost = n * passes * self.GPU_OPERATOR_COST / num_sms
            # Histogram phase: each pass builds 256-bin histogram per SM tile
            hist_cost = passes * num_sms * 256 * self.GPU_OPERATOR_COST
            cb.sort_cost_us = scatter_cost + hist_cost

        if node.compute_capacity > 0 and node.compute_capacity != 1.0:
            scale = 1.0 / node.compute_capacity
            scalable_compute_us *= scale
            cb.sort_cost_us *= scale

        cb.compute_cost_us = kernel_launch_us + scalable_compute_us

        checkpoint("gpu_cost_done", query_id=query.query_id,
                   l2_hit=l2_hit, transfer=cb.transfer_cost_us,
                   compute=cb.compute_cost_us, sort=cb.sort_cost_us,
                   kernel_type="warm" if self._launch_count > 3 else "cold")
        return cb


class CostModelEngine:
    def __init__(self, topology: HardwareTopology):
        self.topology = topology
        self.cpu_model = CPUCostModel()
        self.gpu_model = GPUCostModel()
        self._cache: Dict[str, CostBreakdown] = {}

    def estimate_on_device(self, query: QueryDescriptor,
                           device_id: str,
                           data_location: Optional[str] = None
                           ) -> CostBreakdown:
        from ._debug import dbg
        dbg('CostModel.estimate', query_id=query.query_id,
            device=device_id, qtype=query.query_type.name)
        node = self.topology.get_node(device_id)
        if node is None:
            raise ValueError(f"Unknown device: {device_id}")

        if node.kind == HardwareKind.GPU:
            data_resident = (data_location == device_id)
            cb = self.gpu_model.estimate(query, node, data_resident)
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

        return cb

    def estimate_all_devices(self, query: QueryDescriptor,
                             data_location: Optional[str] = None
                             ) -> Dict[str, CostBreakdown]:
        results = {}
        for node_id, node in self.topology.nodes.items():
            if node.kind in (HardwareKind.GPU, HardwareKind.CPU):
                try:
                    results[node_id] = self.estimate_on_device(
                        query, node_id, data_location
                    )
                except ValueError:
                    continue
        return results

    def recommend(self, query: QueryDescriptor,
                  data_location: Optional[str] = None
                  ) -> Tuple[str, CostBreakdown]:
        estimates = self.estimate_all_devices(query, data_location)
        if not estimates:
            raise RuntimeError("No devices available for estimation")
        from ._debug import dbg
        dbg('CostModel.recommend_result', query_id=query.query_id,
            n_candidates=len(estimates),
            costs={k: v.total_ms for k, v in estimates.items()})
        best_id = min(estimates, key=lambda k: estimates[k].total_us)
        return best_id, estimates[best_id]

    def route_batch(self, queries: List[QueryDescriptor],
                    data_location: Optional[str] = None
                    ) -> List[Tuple[str, CostBreakdown]]:
        return [self.recommend(q, data_location) for q in queries]


def create_default_topology() -> HardwareTopology:
    topo = HardwareTopology()
    cpu0 = HardwareNode(
        node_id="cpu0", kind=HardwareKind.CPU,
        compute_capacity=1.0, memory_bytes=256 * (1 << 30),
        scan_cost_per_row=1.0, seek_cost=4.0, compute_cost_per_op=0.01,
    )
    cpu1 = HardwareNode(
        node_id="cpu1", kind=HardwareKind.CPU,
        compute_capacity=1.0, memory_bytes=256 * (1 << 30),
        scan_cost_per_row=1.0, seek_cost=4.0, compute_cost_per_op=0.01,
    )
    gpus = []
    for i in range(4):
        gpu = HardwareNode(
            node_id=f"gpu{i}", kind=HardwareKind.GPU,
            compute_capacity=110.0, memory_bytes=80 * (1 << 30),
            bandwidth_gbps=2000.0,
            scan_cost_per_row=0.001, seek_cost=0.01, compute_cost_per_op=0.0001,
        )
        gpus.append(gpu)
    for n in [cpu0, cpu1] + gpus:
        topo.add_node(n)
    # --- FIX(INV-4): 双socket拓扑 — cpu0/cpu1 各自直连本侧GPU ---
    # 原版只连 cpu0→GPU, cpu1→GPU 不可达(cost=inf), 砍掉一半NUMA加速。
    # 真实服务器: gpu0,gpu1 在 NUMA-0 (挂cpu0), gpu2,gpu3 在 NUMA-1 (挂cpu1)。
    # 跨NUMA走 UPI 互联, 带宽低于本地 PCIe, 但绝不是 inf。
    half = len(gpus) // 2
    for idx, gpu in enumerate(gpus):
        local_cpu = "cpu0" if idx < half else "cpu1"
        remote_cpu = "cpu1" if idx < half else "cpu0"
        # 本地 PCIe 直连 (32 GB/s, ~1µs)
        topo.add_edge(TopologyEdge(
            src=local_cpu, dst=gpu.node_id,
            bandwidth_gbps=32.0, latency_us=1.0,
            link_type=HardwareKind.PCIE,
        ))
        topo.add_edge(TopologyEdge(
            src=gpu.node_id, dst=local_cpu,
            bandwidth_gbps=32.0, latency_us=1.0,
            link_type=HardwareKind.PCIE,
        ))
        # 跨NUMA: remote_cpu → UPI → local_cpu → PCIe → GPU
        # 建模为直达边但带宽受 UPI 瓶颈限制 (~22 GB/s), 延迟 +0.3µs
        topo.add_edge(TopologyEdge(
            src=remote_cpu, dst=gpu.node_id,
            bandwidth_gbps=22.0, latency_us=1.3,
            link_type=HardwareKind.PCIE,
        ))
        topo.add_edge(TopologyEdge(
            src=gpu.node_id, dst=remote_cpu,
            bandwidth_gbps=22.0, latency_us=1.3,
            link_type=HardwareKind.PCIE,
        ))
    # NVLink 全互联 (GPU ↔ GPU)
    for i, g1 in enumerate(gpus):
        for j, g2 in enumerate(gpus):
            if i != j:
                topo.add_edge(TopologyEdge(
                    src=g1.node_id, dst=g2.node_id,
                    bandwidth_gbps=600.0, latency_us=0.5,
                    link_type=HardwareKind.NVLINK,
                ))
    # UPI 双向互联 (cpu0 ↔ cpu1)
    topo.add_edge(TopologyEdge(
        src="cpu0", dst="cpu1", bandwidth_gbps=50.0, latency_us=0.3,
        link_type=HardwareKind.NETWORK,
    ))
    topo.add_edge(TopologyEdge(
        src="cpu1", dst="cpu0", bandwidth_gbps=50.0, latency_us=0.3,
        link_type=HardwareKind.NETWORK,
    ))

    # --- 调试探针: 打印拓扑可达性矩阵 ---
    from ._debug import dbg
    all_nodes = sorted(topo.nodes.keys())
    reachability = {}
    for s in all_nodes:
        row = {}
        for d in all_nodes:
            c = topo.get_transfer_cost(s, d, 1_000_000)  # 1MB probe
            row[d] = f"{c:.1f}" if c < 1e9 else "INF"
        reachability[s] = row
    dbg("TOPO·reachability_matrix", nodes=all_nodes, matrix=reachability)
    return topo
