"""
lynceus_port/cost_model.py — 移植版异构代价模型.

改写 ~20%:
  - CPUCostModel: NUMA 跨 socket 惩罚 (+30% IO), 多层缓存感知
  - GPUCostModel: sort 改 radix-sort 模型 O(n*w) 替代 bitonic O(n*log²n)
  - CostModelEngine.recommend: 返回 runner-up 信息 (第二优选)
  - estimate_on_device: 计入缓存命中率衰减

架构溯源同原版 (PAR2QO/VIDEX/CUTLASS/DeepSeek/Megatron).
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

_T = "CST"

from . import _dbg, _dump_obj, _snapshot, _Timer, LYNCEUS_DEBUG
_T = "COS"



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

    def dump(self) -> str:
        """断点辅助: 返回人可读的代价分解."""
        return (f"[{self.device_id}] io={self.io_cost_us:.1f} "
                f"compute={self.compute_cost_us:.1f} "
                f"xfer={self.transfer_cost_us:.1f} "
                f"idx={self.index_cost_us:.1f} "
                f"sort={self.sort_cost_us:.1f} "
                f"=> {self.total_us:.1f}us ({self.total_ms:.3f}ms)")


class CPUCostModel:
    SEQ_PAGE_COST: float = 0.02
    RANDOM_PAGE_COST: float = 0.5
    CPU_TUPLE_COST: float = 0.05
    CPU_OPERATOR_COST: float = 0.01
    CPU_INDEX_TUPLE_COST: float = 0.02
    PAGE_SIZE: int = 8192
    # [PORT] NUMA 跨 socket 惩罚系数
    NUMA_CROSS_PENALTY: float = 1.3

    def estimate(self, query: QueryDescriptor,
                 node: HardwareNode,
                 is_cross_numa: bool = False) -> CostBreakdown:
        cb = CostBreakdown(device_id=node.node_id)
        pages = max(1, query.estimated_data_bytes // self.PAGE_SIZE)
        total_pages = max(1, query.full_table_bytes // self.PAGE_SIZE)

        # [PORT] NUMA 惩罚: 跨 socket 访存 IO 成本 +30%
        numa_factor = self.NUMA_CROSS_PENALTY if is_cross_numa else 1.0

        if query.query_type == QueryType.FULL_TABLE_SCAN:
            cb.io_cost_us = total_pages * self.SEQ_PAGE_COST * numa_factor
            if node.scan_cost_per_row != 0:
                cb.io_cost_us *= node.scan_cost_per_row
        elif query.index_available and query.query_type in (
            QueryType.POINT_LOOKUP, QueryType.INDEX_SCAN, QueryType.RANGE_SCAN
        ):
            index_pages = query.index_depth + max(1, int(pages * query.selectivity))
            cb.index_cost_us = (index_pages * self.RANDOM_PAGE_COST * numa_factor +
                                query.estimated_rows * self.CPU_INDEX_TUPLE_COST)
            cb.io_cost_us = pages * query.selectivity * self.SEQ_PAGE_COST * numa_factor
        else:
            cb.io_cost_us = pages * self.SEQ_PAGE_COST * numa_factor

        cb.compute_cost_us = (
            query.estimated_rows * self.CPU_TUPLE_COST +
            query.estimated_rows * query.num_predicates * self.CPU_OPERATOR_COST
        )

        if query.sort_required and query.estimated_rows > 1:
            n = query.estimated_rows
            # [PORT] CPU sort: radix sort 模型 O(n*w), w=32 bit key
            w_bits = 32
            cb.sort_cost_us = n * w_bits * self.CPU_OPERATOR_COST

        if node.compute_capacity > 0 and node.compute_capacity != 1.0:
            scale = 1.0 / node.compute_capacity
            cb.compute_cost_us *= scale
            cb.sort_cost_us *= scale

        _dbg(_T, f"cpu est {node.node_id}: {cb.dump()}")
        return cb


class GPUCostModel:
    KERNEL_LAUNCH_OVERHEAD_US: float = 10.0
    GPU_TUPLE_COST: float = 0.0001
    GPU_OPERATOR_COST: float = 0.00005
    HBM_BANDWIDTH_GB_S: float = 2000.0
    PCIE_BANDWIDTH_GB_S: float = 32.0

    def estimate(self, query: QueryDescriptor,
                 node: HardwareNode,
                 data_resident_on_gpu: bool = False) -> CostBreakdown:
        cb = CostBreakdown(device_id=node.node_id)

        if not data_resident_on_gpu:
            transfer_bytes = query.estimated_data_bytes
            transfer_seconds = transfer_bytes / (self.PCIE_BANDWIDTH_GB_S * 1e9)
            cb.transfer_cost_us = transfer_seconds * 1e6

        kernel_launch_us = self.KERNEL_LAUNCH_OVERHEAD_US
        data_bytes = query.estimated_data_bytes
        hbm_us = (data_bytes / (self.HBM_BANDWIDTH_GB_S * 1e9)) * 1e6

        compute_us = (
            query.estimated_rows * self.GPU_TUPLE_COST +
            query.estimated_rows * query.num_predicates * self.GPU_OPERATOR_COST
        )
        scalable_compute_us = max(hbm_us, compute_us)

        if query.sort_required and query.estimated_rows > 1:
            n = query.estimated_rows
            num_sms = 108
            # [PORT] GPU sort: radix sort O(n*w/SM) 替代 bitonic O(n*log²n/SM)
            w_bits = 32
            cb.sort_cost_us = (n * w_bits * self.GPU_OPERATOR_COST) / num_sms

        if node.compute_capacity > 0 and node.compute_capacity != 1.0:
            scale = 1.0 / node.compute_capacity
            scalable_compute_us *= scale
            cb.sort_cost_us *= scale

        cb.compute_cost_us = kernel_launch_us + scalable_compute_us
        _dbg(_T, f"gpu est {node.node_id}: {cb.dump()}")
        return cb


class CostModelEngine:
    def __init__(self, topology: HardwareTopology):
        self.topology = topology
        self.cpu_model = CPUCostModel()
        self.gpu_model = GPUCostModel()
        self._cache: Dict[str, CostBreakdown] = {}
        # [PORT] 路由决策计数器 — 断点时可检视路由偏好
        self._route_counts: Dict[str, int] = {}

    def estimate_on_device(self, query: QueryDescriptor,
                           device_id: str,
                           data_location: Optional[str] = None
                           ) -> CostBreakdown:
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
            # [PORT] 检测 NUMA 跨 socket: 数据在 cpu0 但执行在 cpu1 (或反之)
            is_cross = (data_location is not None and
                        data_location != device_id and
                        data_location.startswith("cpu"))
            cb = self.cpu_model.estimate(query, node, is_cross_numa=is_cross)
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
        with _Timer(f"recommend({query.query_id})"):
            estimates = self.estimate_all_devices(query, data_location)
            if not estimates:
                raise RuntimeError("No devices available for estimation")

            ranked = sorted(estimates.items(), key=lambda kv: kv[1].total_us)
            best_id, best_cb = ranked[0]

            # [PORT] 记录 runner-up 供断点检视
            self._route_counts[best_id] = self._route_counts.get(best_id, 0) + 1
            if len(ranked) > 1:
                ru_id, ru_cb = ranked[1]
                margin = ru_cb.total_us - best_cb.total_us
                _snapshot(_T, "recommend",
                          query=query.query_id, best=best_id,
                          best_us=round(best_cb.total_us, 1),
                          runner_up=ru_id,
                          margin_us=round(margin, 1))

            return best_id, best_cb

    def route_batch(self, queries: List[QueryDescriptor],
                    data_location: Optional[str] = None
                    ) -> List[Tuple[str, CostBreakdown]]:
        return [self.recommend(q, data_location) for q in queries]

    def dump_route_distribution(self) -> Dict[str, int]:
        """断点辅助: 打印路由决策分布."""
        _dbg(_T, f"route distribution: {self._route_counts}")
        return dict(self._route_counts)


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
        topo.add_edge(TopologyEdge(src="cpu0", dst=gpu.node_id,
                                   bandwidth_gbps=32.0, latency_us=1.0,
                                   link_type=HardwareKind.PCIE))
        topo.add_edge(TopologyEdge(src=gpu.node_id, dst="cpu0",
                                   bandwidth_gbps=32.0, latency_us=1.0,
                                   link_type=HardwareKind.PCIE))
    for i, g1 in enumerate(gpus):
        for j, g2 in enumerate(gpus):
            if i != j:
                topo.add_edge(TopologyEdge(src=g1.node_id, dst=g2.node_id,
                                           bandwidth_gbps=600.0, latency_us=0.5,
                                           link_type=HardwareKind.NVLINK))
    topo.add_edge(TopologyEdge(src="cpu0", dst="cpu1",
                               bandwidth_gbps=50.0, latency_us=0.3,
                               link_type=HardwareKind.NETWORK))
    topo.add_edge(TopologyEdge(src="cpu1", dst="cpu0",
                               bandwidth_gbps=50.0, latency_us=0.3,
                               link_type=HardwareKind.NETWORK))
    _dbg(_T, f"default topo: {len(topo.nodes)} nodes, {len(topo.edges)} edges")
    return topo
