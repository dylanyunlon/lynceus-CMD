"""
lynceus_port/cost_model.py — 异构代价模型，核心路由决策引擎。

移植自 lynceus/cost_model.py，修改约20%:
  - GPU模型: 新增 SM 占用率 (occupancy) 建模，影响实际吞吐
  - CPU模型: 引入 NUMA 亲和性惩罚因子
  - CostBreakdown: 新增 confidence 字段，表示估算置信度
  - recommend(): 返回带置信度排序的候选列表
  - 全部 estimate 路径加入 debug_snapshot / trace
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from .schema import (
    HardwareNode,
    HardwareTopology,
    HardwareKind,
    RoutingStrategy,
    TopologyEdge,
    _dbg,
)


# ---------------------------------------------------------------------------
# 查询描述符
# ---------------------------------------------------------------------------

class QueryType(Enum):
    POINT_LOOKUP   = auto()
    RANGE_SCAN     = auto()
    FULL_TABLE_SCAN = auto()
    INDEX_SCAN     = auto()
    JOIN           = auto()
    AGGREGATE      = auto()
    SORT           = auto()


@dataclass
class QueryDescriptor:
    """单条查询的特征描述——代价估算的输入"""
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
            raise ValueError(f"estimated_rows >= 0 required, got {self.estimated_rows}")
        if self.table_rows < 0:
            raise ValueError(f"table_rows >= 0 required, got {self.table_rows}")
        if not (0.0 <= self.selectivity <= 1.0):
            raise ValueError(f"selectivity in [0,1] required, got {self.selectivity}")
        if self.estimated_width_bytes < 0:
            raise ValueError(f"estimated_width_bytes >= 0, got {self.estimated_width_bytes}")

    @property
    def estimated_data_bytes(self) -> int:
        return self.estimated_rows * self.estimated_width_bytes

    @property
    def full_table_bytes(self) -> int:
        return self.table_rows * self.estimated_width_bytes

    def debug_snapshot(self) -> str:
        s = (f"Query({self.query_id}, {self.query_type.name}, "
             f"rows={self.estimated_rows}, sel={self.selectivity:.3f}, "
             f"idx={'Y' if self.index_available else 'N'}, "
             f"joins={self.num_joins}, sort={'Y' if self.sort_required else 'N'})")
        _dbg("QueryDesc", s)
        return s


# ---------------------------------------------------------------------------
# 代价分解
# ---------------------------------------------------------------------------

@dataclass
class CostBreakdown:
    """某条查询在特定设备上的逐项代价估算（微秒）"""
    device_id: str
    io_cost_us: float = 0.0
    compute_cost_us: float = 0.0
    transfer_cost_us: float = 0.0
    index_cost_us: float = 0.0
    sort_cost_us: float = 0.0
    # ── 新增：估算置信度 [0,1]，1=完全确定 ──
    confidence: float = 1.0

    @property
    def total_us(self) -> float:
        return (self.io_cost_us + self.compute_cost_us +
                self.transfer_cost_us + self.index_cost_us +
                self.sort_cost_us)

    @property
    def total_ms(self) -> float:
        return self.total_us / 1000.0

    def debug_snapshot(self) -> str:
        s = (f"Cost({self.device_id}): "
             f"io={self.io_cost_us:.2f} compute={self.compute_cost_us:.2f} "
             f"xfer={self.transfer_cost_us:.2f} idx={self.index_cost_us:.2f} "
             f"sort={self.sort_cost_us:.2f} => total={self.total_us:.2f}us "
             f"(conf={self.confidence:.2f})")
        _dbg("CostBreakdown", s)
        return s


# ---------------------------------------------------------------------------
# CPU 代价模型
# ---------------------------------------------------------------------------

class CPUCostModel:
    """CPU 侧查询执行代价模型。

    参考 PostgreSQL 代价参数（seq_page_cost, random_page_cost, ...）
    和 PAR2QO get_plan_cost_simple()。

    修改点：新增 NUMA 亲和性惩罚——跨 NUMA 节点访问内存时
    代价乘以 numa_penalty_factor。
    """

    SEQ_PAGE_COST: float = 0.02
    RANDOM_PAGE_COST: float = 0.5
    CPU_TUPLE_COST: float = 0.05
    CPU_OPERATOR_COST: float = 0.01
    CPU_INDEX_TUPLE_COST: float = 0.02
    PAGE_SIZE: int = 8192
    # ── 新增：NUMA 惩罚 ──
    NUMA_PENALTY_FACTOR: float = 1.35  # 跨 NUMA ~35% 额外延迟

    def estimate(self, query: QueryDescriptor,
                 node: HardwareNode,
                 is_numa_local: bool = True) -> CostBreakdown:
        cb = CostBreakdown(device_id=node.node_id)

        pages = max(1, query.estimated_data_bytes // self.PAGE_SIZE)
        total_pages = max(1, query.full_table_bytes // self.PAGE_SIZE)

        # I/O 代价
        if query.query_type == QueryType.FULL_TABLE_SCAN:
            cb.io_cost_us = total_pages * self.SEQ_PAGE_COST * node.scan_cost_per_row
            if node.scan_cost_per_row == 0:
                cb.io_cost_us = total_pages * self.SEQ_PAGE_COST
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

        # 计算代价
        cb.compute_cost_us = (
            query.estimated_rows * self.CPU_TUPLE_COST +
            query.estimated_rows * query.num_predicates * self.CPU_OPERATOR_COST
        )

        # 排序代价
        if query.sort_required and query.estimated_rows > 1:
            n = query.estimated_rows
            cb.sort_cost_us = (
                2.0 * n * math.log2(max(2, n)) * self.CPU_OPERATOR_COST
            )

        # 容量缩放
        if node.compute_capacity > 0 and node.compute_capacity != 1.0:
            scale = 1.0 / node.compute_capacity
            cb.compute_cost_us *= scale
            cb.sort_cost_us *= scale

        # ── NUMA 亲和性惩罚 ──
        if not is_numa_local:
            cb.io_cost_us *= self.NUMA_PENALTY_FACTOR
            cb.index_cost_us *= self.NUMA_PENALTY_FACTOR
            cb.confidence *= 0.85  # 跨 NUMA 时估算不确定性增加
            _dbg("CPUCost", f"NUMA penalty applied on {node.node_id}")

        _dbg("CPUCost",
             f"{node.node_id}: io={cb.io_cost_us:.2f} "
             f"comp={cb.compute_cost_us:.2f} sort={cb.sort_cost_us:.2f}")
        return cb


# ---------------------------------------------------------------------------
# GPU 代价模型
# ---------------------------------------------------------------------------

class GPUCostModel:
    """GPU 加速查询执行代价模型。

    修改点：新增 SM 占用率建模——当 warp 数不足以填满所有 SM 时，
    实际吞吐按 occupancy 比例折减。小查询因 occupancy 低而吃亏。
    """

    KERNEL_LAUNCH_OVERHEAD_US: float = 10.0
    GPU_TUPLE_COST: float = 0.0001
    GPU_OPERATOR_COST: float = 0.00005
    HBM_BANDWIDTH_GB_S: float = 2000.0
    PCIE_BANDWIDTH_GB_S: float = 32.0
    # ── 新增：SM 占用率参数 ──
    NUM_SMS: int = 108           # A100
    WARPS_PER_SM: int = 64       # 每个 SM 最大活跃 warp 数
    ROWS_PER_WARP: int = 32      # warp 大小（线程数）

    def _compute_occupancy(self, num_rows: int) -> float:
        """计算 SM 占用率。行数少→warp 少→SM 空闲→实际吞吐打折"""
        total_warps_needed = max(1, num_rows // self.ROWS_PER_WARP)
        max_concurrent = self.NUM_SMS * self.WARPS_PER_SM
        occupancy = min(1.0, total_warps_needed / max_concurrent)
        # 至少 5% 占用率（kernel 总有调度开销）
        return max(0.05, occupancy)

    def estimate(self, query: QueryDescriptor,
                 node: HardwareNode,
                 data_resident_on_gpu: bool = False) -> CostBreakdown:
        cb = CostBreakdown(device_id=node.node_id)

        # PCIe 传输代价
        if not data_resident_on_gpu:
            transfer_bytes = query.estimated_data_bytes
            transfer_seconds = transfer_bytes / (self.PCIE_BANDWIDTH_GB_S * 1e9)
            cb.transfer_cost_us = transfer_seconds * 1e6

        # kernel 启动开销
        kernel_launch_us = self.KERNEL_LAUNCH_OVERHEAD_US

        # HBM 带宽受限分量
        data_bytes = query.estimated_data_bytes
        hbm_seconds = data_bytes / (self.HBM_BANDWIDTH_GB_S * 1e9)
        hbm_us = hbm_seconds * 1e6

        # 计算受限分量
        compute_us = (
            query.estimated_rows * self.GPU_TUPLE_COST +
            query.estimated_rows * query.num_predicates * self.GPU_OPERATOR_COST
        )

        scalable_compute_us = max(hbm_us, compute_us)

        # ── SM 占用率折减 ──
        occupancy = self._compute_occupancy(query.estimated_rows)
        if occupancy < 1.0:
            scalable_compute_us /= occupancy
            cb.confidence *= occupancy  # 低占用率→估算信心降低
            _dbg("GPUCost",
                 f"occupancy={occupancy:.3f} on {node.node_id}, "
                 f"compute inflated to {scalable_compute_us:.2f}us")

        # GPU 排序（bitonic sort）
        if query.sort_required and query.estimated_rows > 1:
            n = query.estimated_rows
            ops = n * (math.log2(max(2, n)) ** 2)
            cb.sort_cost_us = ops * self.GPU_OPERATOR_COST / self.NUM_SMS

        # 容量缩放
        if node.compute_capacity > 0 and node.compute_capacity != 1.0:
            scale = 1.0 / node.compute_capacity
            scalable_compute_us *= scale
            cb.sort_cost_us *= scale

        cb.compute_cost_us = kernel_launch_us + scalable_compute_us

        _dbg("GPUCost",
             f"{node.node_id}: launch={kernel_launch_us:.1f} "
             f"compute={scalable_compute_us:.2f} xfer={cb.transfer_cost_us:.2f} "
             f"occ={occupancy:.3f}")
        return cb


# ---------------------------------------------------------------------------
# 统一代价模型引擎
# ---------------------------------------------------------------------------

class CostModelEngine:
    """统一代价模型——在拓扑中所有设备上估算查询代价并推荐路由"""

    def __init__(self, topology: HardwareTopology):
        self.topology = topology
        self.cpu_model = CPUCostModel()
        self.gpu_model = GPUCostModel()
        self._estimate_count = 0  # 调试计数器

    def estimate_on_device(self, query: QueryDescriptor,
                           device_id: str,
                           data_location: Optional[str] = None
                           ) -> CostBreakdown:
        """估算查询在指定设备上的执行代价"""
        node = self.topology.get_node(device_id)
        if node is None:
            raise ValueError(f"未知设备: {device_id}")

        self._estimate_count += 1

        if node.kind == HardwareKind.GPU:
            data_resident = (data_location == device_id)
            cb = self.gpu_model.estimate(query, node, data_resident)
            if data_location and not data_resident:
                cb.transfer_cost_us = self.topology.get_transfer_cost(
                    data_location, device_id, query.estimated_data_bytes
                )
        elif node.kind == HardwareKind.CPU:
            # ── NUMA 亲和性判定 ──
            is_local = (data_location is None or
                        data_location == device_id or
                        data_location.startswith("cpu"))
            cb = self.cpu_model.estimate(query, node, is_numa_local=is_local)
            if data_location and data_location != device_id:
                cb.transfer_cost_us = self.topology.get_transfer_cost(
                    data_location, device_id, query.estimated_data_bytes
                )
        else:
            raise ValueError(f"不支持的设备类型: {node.kind}")

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
        """推荐最优设备。

        修改点：当最优和次优代价差距 <15% 时，优先选置信度更高的那个。
        这避免了在不确定场景下盲目追求理论最优。
        """
        estimates = self.estimate_all_devices(query, data_location)
        if not estimates:
            raise RuntimeError("没有可用设备进行估算")

        # 按代价排序
        ranked = sorted(estimates.items(), key=lambda kv: kv[1].total_us)

        best_id, best_cb = ranked[0]

        # ── 置信度修正：如果差距小，选信心更高的 ──
        if len(ranked) > 1:
            runner_id, runner_cb = ranked[1]
            gap_ratio = (runner_cb.total_us - best_cb.total_us) / max(
                best_cb.total_us, 1e-9)
            if gap_ratio < 0.15 and runner_cb.confidence > best_cb.confidence:
                _dbg("Engine",
                     f"confidence override: {best_id}({best_cb.total_us:.1f}us, "
                     f"conf={best_cb.confidence:.2f}) -> "
                     f"{runner_id}({runner_cb.total_us:.1f}us, "
                     f"conf={runner_cb.confidence:.2f})")
                best_id, best_cb = runner_id, runner_cb

        _dbg("Engine",
             f"recommend: q={query.query_id} -> {best_id} "
             f"({best_cb.total_us:.1f}us, conf={best_cb.confidence:.2f})")
        return best_id, best_cb

    def route_batch(self, queries: List[QueryDescriptor],
                    data_location: Optional[str] = None
                    ) -> List[Tuple[str, CostBreakdown]]:
        results = [self.recommend(q, data_location) for q in queries]
        _dbg("Engine",
             f"batch routed: {len(queries)} queries, "
             f"estimate_calls={self._estimate_count}")
        return results

    def debug_snapshot(self) -> str:
        lines = [
            "══════ CostModelEngine Snapshot ══════",
            f"  estimate_count = {self._estimate_count}",
            f"  topology nodes = {len(self.topology.nodes)}",
            f"  CPU model: SEQ_PAGE={self.cpu_model.SEQ_PAGE_COST}, "
            f"NUMA_PEN={self.cpu_model.NUMA_PENALTY_FACTOR}",
            f"  GPU model: LAUNCH={self.gpu_model.KERNEL_LAUNCH_OVERHEAD_US}us, "
            f"SMS={self.gpu_model.NUM_SMS}",
        ]
        s = "\n".join(lines)
        _dbg("Engine", s)
        return s


# ---------------------------------------------------------------------------
# 默认拓扑工厂
# ---------------------------------------------------------------------------

def create_default_topology() -> HardwareTopology:
    """创建典型单节点异构拓扑：2×CPU + 4×GPU (PCIe/NVLink)"""
    topo = HardwareTopology()

    cpu0 = HardwareNode(
        node_id="cpu0", kind=HardwareKind.CPU,
        compute_capacity=1.0,
        memory_bytes=256 * (1 << 30),
        scan_cost_per_row=1.0, seek_cost=4.0,
        compute_cost_per_op=0.01,
    )
    cpu1 = HardwareNode(
        node_id="cpu1", kind=HardwareKind.CPU,
        compute_capacity=1.0,
        memory_bytes=256 * (1 << 30),
        scan_cost_per_row=1.0, seek_cost=4.0,
        compute_cost_per_op=0.01,
    )

    gpus = []
    for i in range(4):
        gpu = HardwareNode(
            node_id=f"gpu{i}", kind=HardwareKind.GPU,
            compute_capacity=100.0,
            memory_bytes=80 * (1 << 30),
            bandwidth_gbps=2000.0,
            scan_cost_per_row=0.001, seek_cost=0.01,
            compute_cost_per_op=0.0001,
        )
        gpus.append(gpu)

    for n in [cpu0, cpu1] + gpus:
        topo.add_node(n)

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

    for i, g1 in enumerate(gpus):
        for j, g2 in enumerate(gpus):
            if i != j:
                topo.add_edge(TopologyEdge(
                    src=g1.node_id, dst=g2.node_id,
                    bandwidth_gbps=600.0, latency_us=0.5,
                    link_type=HardwareKind.NVLINK,
                ))

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

    _dbg("Factory", f"default topology created: {len(topo.nodes)} nodes, "
         f"{len(topo.edges)} edges")
    topo.debug_snapshot()
    return topo
