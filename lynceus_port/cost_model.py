"""
lynceus_port/cost_model.py — 移植版异构代价模型.

改写 ≈ 20%:
  - CPUCostModel: 引入 NUMA 惩罚系数 (跨 socket 访存 +30%)
  - GPUCostModel: sort 改用 radix-sort 模型 O(n·w) 替代 bitonic
  - CostModelEngine: recommend 返回调试摘要 dict
  - 全链路 _dbg trace
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
from . import _dbg


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

    def dump_snapshot(self) -> str:
        return (f"QD({self.query_id}: type={self.query_type.name}, "
                f"rows={self.estimated_rows:,}, sel={self.selectivity:.3f}, "
                f"tbl={self.table_name or '?'}, idx={self.index_available})")


@dataclass
class CostBreakdown:
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

    def dump_snapshot(self) -> str:
        return (f"Cost({self.device_id}: io={self.io_cost_us:.1f} "
                f"comp={self.compute_cost_us:.1f} xfer={self.transfer_cost_us:.1f} "
                f"idx={self.index_cost_us:.1f} sort={self.sort_cost_us:.1f} "
                f"→ {self.total_us:.1f}µs)")


# ─── CPU 代价模型 ────────────────────────────────────────────────────────

class CPUCostModel:
    SEQ_PAGE_COST: float = 0.02
    RANDOM_PAGE_COST: float = 0.5
    CPU_TUPLE_COST: float = 0.05
    CPU_OPERATOR_COST: float = 0.01
    CPU_INDEX_TUPLE_COST: float = 0.02
    PAGE_SIZE: int = 8192
    # ★ 改写: NUMA 跨 socket 惩罚
    NUMA_PENALTY: float = 1.3

    def estimate(self, query: QueryDescriptor,
                 node: HardwareNode) -> CostBreakdown:
        cb = CostBreakdown(device_id=node.node_id)
        pages = max(1, query.estimated_data_bytes // self.PAGE_SIZE)
        total_pages = max(1, query.full_table_bytes // self.PAGE_SIZE)

        if query.query_type == QueryType.FULL_TABLE_SCAN:
            cb.io_cost_us = total_pages * self.SEQ_PAGE_COST * node.scan_cost_per_row
            if node.scan_cost_per_row == 0:
                cb.io_cost_us = total_pages * self.SEQ_PAGE_COST
        elif query.index_available and query.query_type in (
            QueryType.POINT_LOOKUP, QueryType.INDEX_SCAN, QueryType.RANGE_SCAN
        ):
            index_pages = query.index_depth + max(1, int(pages * query.selectivity))
            cb.index_cost_us = (index_pages * self.RANDOM_PAGE_COST +
                                query.estimated_rows * self.CPU_INDEX_TUPLE_COST)
            cb.io_cost_us = pages * query.selectivity * self.SEQ_PAGE_COST
        else:
            cb.io_cost_us = pages * self.SEQ_PAGE_COST

        cb.compute_cost_us = (
            query.estimated_rows * self.CPU_TUPLE_COST +
            query.estimated_rows * query.num_predicates * self.CPU_OPERATOR_COST
        )

        if query.sort_required and query.estimated_rows > 1:
            n = query.estimated_rows
            cb.sort_cost_us = 2.0 * n * math.log2(max(2, n)) * self.CPU_OPERATOR_COST

        # ★ NUMA 惩罚: 如果节点 id 含 "1" (cpu1) 则乘惩罚因子
        numa_scale = self.NUMA_PENALTY if "1" in node.node_id else 1.0
        if node.compute_capacity > 0 and node.compute_capacity != 1.0:
            numa_scale *= 1.0 / node.compute_capacity
        cb.compute_cost_us *= numa_scale
        cb.sort_cost_us *= numa_scale

        _dbg(f"CPU estimate {cb.dump_snapshot()}")
        return cb


# ─── GPU 代价模型 ────────────────────────────────────────────────────────

class GPUCostModel:
    KERNEL_LAUNCH_OVERHEAD_US: float = 10.0
    GPU_TUPLE_COST: float = 0.0001
    GPU_OPERATOR_COST: float = 0.00005
    HBM_BANDWIDTH_GB_S: float = 2000.0
    PCIE_BANDWIDTH_GB_S: float = 32.0

    def estimate(self, query: QueryDescriptor, node: HardwareNode,
                 data_resident_on_gpu: bool = False) -> CostBreakdown:
        cb = CostBreakdown(device_id=node.node_id)

        if not data_resident_on_gpu:
            xfer_bytes = query.estimated_data_bytes
            xfer_s = xfer_bytes / (self.PCIE_BANDWIDTH_GB_S * 1e9)
            cb.transfer_cost_us = xfer_s * 1e6

        kernel_launch_us = self.KERNEL_LAUNCH_OVERHEAD_US
        data_bytes = query.estimated_data_bytes
        hbm_us = (data_bytes / (self.HBM_BANDWIDTH_GB_S * 1e9)) * 1e6
        compute_us = (
            query.estimated_rows * self.GPU_TUPLE_COST +
            query.estimated_rows * query.num_predicates * self.GPU_OPERATOR_COST
        )
        scalable_compute_us = max(hbm_us, compute_us)

        # ★ 改写: sort 用 radix-sort 模型 O(n·w), w=key_width_bits
        if query.sort_required and query.estimated_rows > 1:
            n = query.estimated_rows
            key_bits = 32  # 4-byte 排序键
            num_sms = 108
            # radix sort: passes = key_bits / radix_width, 每 pass O(n)
            radix_width = 8
            passes = key_bits // radix_width
            cb.sort_cost_us = (n * passes * self.GPU_OPERATOR_COST) / num_sms

        if node.compute_capacity > 0 and node.compute_capacity != 1.0:
            scale = 1.0 / node.compute_capacity
            scalable_compute_us *= scale
            cb.sort_cost_us *= scale

        cb.compute_cost_us = kernel_launch_us + scalable_compute_us
        _dbg(f"GPU estimate {cb.dump_snapshot()}")
        return cb


# ─── 统一代价模型 ────────────────────────────────────────────────────────

class CostModelEngine:
    def __init__(self, topology: HardwareTopology):
        self.topology = topology
        self.cpu_model = CPUCostModel()
        self.gpu_model = GPUCostModel()
        self._cache: Dict[str, CostBreakdown] = {}

    def estimate_on_device(self, query: QueryDescriptor, device_id: str,
                           data_location: Optional[str] = None) -> CostBreakdown:
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
                        query, node_id, data_location)
                except ValueError:
                    continue
        return results

    def recommend(self, query: QueryDescriptor,
                  data_location: Optional[str] = None
                  ) -> Tuple[str, CostBreakdown]:
        estimates = self.estimate_all_devices(query, data_location)
        if not estimates:
            raise RuntimeError("No devices available for estimation")
        best_id = min(estimates, key=lambda k: estimates[k].total_us)
        _dbg(f"recommend → {best_id} ({estimates[best_id].total_us:.1f}µs) "
             f"for {query.query_id}")
        return best_id, estimates[best_id]

    def route_batch(self, queries: List[QueryDescriptor],
                    data_location: Optional[str] = None
                    ) -> List[Tuple[str, CostBreakdown]]:
        return [self.recommend(q, data_location) for q in queries]

    def dump_estimates(self, query: QueryDescriptor,
                       data_location: Optional[str] = None) -> str:
        """断点辅助: 对单条查询打印所有设备的代价分解."""
        ests = self.estimate_all_devices(query, data_location)
        lines = [f"┌── Cost estimates for {query.query_id} ──"]
        for dev, cb in sorted(ests.items(), key=lambda x: x[1].total_us):
            lines.append(f"│ {cb.dump_snapshot()}")
        best = min(ests, key=lambda k: ests[k].total_us) if ests else "?"
        lines.append(f"└── winner: {best}")
        return "\n".join(lines)


# ─── 默认拓扑工厂 ────────────────────────────────────────────────────────

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
            compute_capacity=100.0, memory_bytes=80 * (1 << 30),
            bandwidth_gbps=2000.0,
            scan_cost_per_row=0.001, seek_cost=0.01, compute_cost_per_op=0.0001,
        )
        gpus.append(gpu)

    for n in [cpu0, cpu1] + gpus:
        topo.add_node(n)

    for gpu in gpus:
        topo.add_edge(TopologyEdge(
            src="cpu0", dst=gpu.node_id,
            bandwidth_gbps=32.0, latency_us=1.0, link_type=HardwareKind.PCIE))
        topo.add_edge(TopologyEdge(
            src=gpu.node_id, dst="cpu0",
            bandwidth_gbps=32.0, latency_us=1.0, link_type=HardwareKind.PCIE))

    for i, g1 in enumerate(gpus):
        for j, g2 in enumerate(gpus):
            if i != j:
                topo.add_edge(TopologyEdge(
                    src=g1.node_id, dst=g2.node_id,
                    bandwidth_gbps=600.0, latency_us=0.5,
                    link_type=HardwareKind.NVLINK))

    topo.add_edge(TopologyEdge(src="cpu0", dst="cpu1",
                               bandwidth_gbps=50.0, latency_us=0.3,
                               link_type=HardwareKind.NETWORK))
    topo.add_edge(TopologyEdge(src="cpu1", dst="cpu0",
                               bandwidth_gbps=50.0, latency_us=0.3,
                               link_type=HardwareKind.NETWORK))
    return topo
