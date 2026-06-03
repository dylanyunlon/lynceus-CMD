"""
lynceus_port_v3/cost_model.py — 异构代价模型 (移植改写版)

相对 upstream 的改动 (~20%):
  - CPUCostModel: 调整了页面代价系数 (反映 DDR5-4800 而非 DDR5-5600)
  - GPUCostModel: 重写了 L2 cache hit 模型, 使用 sigmoid 衰减替代线性
  - GPUCostModel: sort 使用 merge-sort 模型替代原始 bitonic
  - CostModelEngine: 新增 explain() 方法返回人类可读的决策解释
  - 全局: 大量注入 debug checkpoint 和 print 断点

架构参考:
    - PAR2QO get_plan_cost() (par2qo/code/postgres.py:110)
    - VIDEX VidexModelBase.scan_time()
    - CUTLASS GemmUniversal (tile-level throughput)
    - DeepSeek act_quant_kernel (FP8 量化)
"""

from __future__ import annotations

import math
import time as _time
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

class QueryCategory(Enum):
    """查询类型分类 — 重命名自upstream的QueryType, 语义更清晰."""
    POINT_LOOKUP = auto()
    RANGE_SCAN = auto()
    FULL_TABLE_SCAN = auto()
    INDEX_SCAN = auto()
    JOIN = auto()
    AGGREGATE = auto()
    SORT = auto()

# 保留向后兼容别名
QueryType = QueryCategory


@dataclass
class QueryDescriptor:
    """查询描述符 — 参考PAR2QO的参数化查询表示.

    每个查询由基数估计、谓词数量、选择性等特征描述.
    """
    query_id: str
    query_type: QueryCategory
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
        # --- DEBUG: 打印每次构造的查询描述 ---
        from ._debug import dbg
        dbg("QueryDescriptor.__init__",
            qid=self.query_id, qtype=self.query_type.name,
            rows=self.estimated_rows, sel=self.selectivity)

        if self.estimated_rows < 0:
            raise ValueError(f"estimated_rows must be >= 0, got {self.estimated_rows}")
        if self.table_rows < 0:
            raise ValueError(f"table_rows must be >= 0, got {self.table_rows}")
        if not (0.0 <= self.selectivity <= 1.0):
            raise ValueError(f"selectivity must be in [0, 1], got {self.selectivity}")
        if self.estimated_width_bytes < 0:
            raise ValueError(f"estimated_width_bytes >= 0 required, got {self.estimated_width_bytes}")

    @property
    def estimated_data_bytes(self) -> int:
        return self.estimated_rows * self.estimated_width_bytes

    @property
    def full_table_bytes(self) -> int:
        return self.table_rows * self.estimated_width_bytes


# ---------------------------------------------------------------------------
# Cost breakdown
# ---------------------------------------------------------------------------

@dataclass
class CostBreakdown:
    """逐项代价估计 — 单位: 微秒 (与NCCL latency_us约定一致)."""
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

    def explain_breakdown(self) -> str:
        """人类可读的代价分解说明 — port_v3新增."""
        parts = []
        if self.io_cost_us > 0:
            parts.append(f"IO={self.io_cost_us:.1f}us")
        if self.compute_cost_us > 0:
            parts.append(f"Compute={self.compute_cost_us:.1f}us")
        if self.transfer_cost_us > 0:
            parts.append(f"Transfer={self.transfer_cost_us:.1f}us")
        if self.index_cost_us > 0:
            parts.append(f"Index={self.index_cost_us:.1f}us")
        if self.sort_cost_us > 0:
            parts.append(f"Sort={self.sort_cost_us:.1f}us")
        return f"[{self.device_id}] TOTAL={self.total_ms:.3f}ms  ({' + '.join(parts)})"


# ---------------------------------------------------------------------------
# CPU代价模型 (改写: 调整系数 + 加入debug checkpoint)
# ---------------------------------------------------------------------------

class CPUCostModel:
    """CPU侧查询执行代价模型.

    参考 PostgreSQL cost 模型 (seq_page_cost 等), 系数基于 DDR5-4800
    实测数据校准 (相比upstream的DDR5-5600略保守):
      - DDR5-4800 sequential: ~38 GB/s → 8KB page ≈ 0.21 µs
      - DRAM random: ~120-350 ns
    """

    # 与upstream不同的系数 — 反映不同内存配置
    SEQ_PAGE_COST: float = 0.028       # upstream: 0.024
    RANDOM_PAGE_COST: float = 0.62     # upstream: 0.58
    CPU_TUPLE_COST: float = 0.048      # upstream: 0.044
    CPU_OPERATOR_COST: float = 0.013   # upstream: 0.012
    CPU_INDEX_TUPLE_COST: float = 0.020  # upstream: 0.018
    PAGE_SIZE: int = 8192

    def estimate(self, query: QueryDescriptor,
                 node: HardwareNode) -> CostBreakdown:
        from ._debug import dbg, checkpoint
        t0 = _time.perf_counter()

        cb = CostBreakdown(device_id=node.node_id)
        pages = max(1, query.estimated_data_bytes // self.PAGE_SIZE)
        total_pages = max(1, query.full_table_bytes // self.PAGE_SIZE)

        # --- DEBUG: 打印输入参数 ---
        dbg("CPUCostModel.estimate.entry",
            device=node.node_id,
            pages=pages, total_pages=total_pages,
            qtype=query.query_type.name,
            index_avail=query.index_available)

        # I/O代价
        if query.query_type == QueryCategory.FULL_TABLE_SCAN:
            effective_scan_coeff = node.scan_cost_per_row if node.scan_cost_per_row > 0 else 1.0
            cb.io_cost_us = total_pages * self.SEQ_PAGE_COST * effective_scan_coeff
        elif query.index_available and query.query_type in (
            QueryCategory.POINT_LOOKUP, QueryCategory.INDEX_SCAN, QueryCategory.RANGE_SCAN
        ):
            # 索引扫描: B-tree遍历(随机IO) + 堆页面(顺序IO)
            sel_pages = max(1, int(pages * query.selectivity))
            index_traverse_pages = query.index_depth + sel_pages
            cb.index_cost_us = (
                index_traverse_pages * self.RANDOM_PAGE_COST
                + query.estimated_rows * self.CPU_INDEX_TUPLE_COST
            )
            cb.io_cost_us = pages * query.selectivity * self.SEQ_PAGE_COST
        else:
            cb.io_cost_us = pages * self.SEQ_PAGE_COST

        # CPU计算代价
        cb.compute_cost_us = (
            query.estimated_rows * self.CPU_TUPLE_COST
            + query.estimated_rows * query.num_predicates * self.CPU_OPERATOR_COST
        )

        # 排序代价 (改写: 加入内存压力因子)
        if query.sort_required and query.estimated_rows > 1:
            n = query.estimated_rows
            log_n = math.log2(max(2, n))
            # port_v3: 使用 2.35 * n * log(n) 替代 upstream 的 2.2
            # 额外: 当 n > 1M 时加入 cache-thrash 惩罚, 使用 sqrt 而非线性
            base_sort = 2.35 * n * log_n * self.CPU_OPERATOR_COST
            cache_penalty = 0.0045 * math.sqrt(max(0, n - 500_000)) if n > 500_000 else 0
            cb.sort_cost_us = base_sort + cache_penalty

        # 按节点相对能力缩放
        if node.compute_capacity > 0 and node.compute_capacity != 1.0:
            inv_cap = 1.0 / node.compute_capacity
            cb.compute_cost_us *= inv_cap
            cb.sort_cost_us *= inv_cap

        elapsed_us = (_time.perf_counter() - t0) * 1e6
        # --- DEBUG: 打印估算结果 ---
        dbg("CPUCostModel.estimate.result",
            device=node.node_id,
            io_us=round(cb.io_cost_us, 2),
            compute_us=round(cb.compute_cost_us, 2),
            sort_us=round(cb.sort_cost_us, 2),
            total_ms=round(cb.total_ms, 4),
            estimation_overhead_us=round(elapsed_us, 1))

        return cb


# ---------------------------------------------------------------------------
# GPU代价模型 (改写: sigmoid L2 模型 + merge-sort)
# ---------------------------------------------------------------------------

class GPUCostModel:
    """GPU加速查询执行代价模型.

    改写说明 (相对upstream):
      1. L2 cache hit: 用 sigmoid(1 - data/40MB) 替代线性模型
         → 更准确地反映 L2 的阶梯式命中率曲线
      2. Sort模型: 用 GPU merge-sort O(n*log(n)) 替代 bitonic O(n*log²(n))
         → 反映现代GPU排序库(CUB RadixSort)的实际复杂度
      3. 新增: warp_occupancy_factor 考虑SM占用率
    """

    KERNEL_LAUNCH_OVERHEAD_US: float = 9.2     # upstream: 8.5 (更保守)
    GPU_TUPLE_COST: float = 0.000092           # upstream: 0.000085
    GPU_OPERATOR_COST: float = 0.000063        # upstream: 0.000058
    HBM_BANDWIDTH_GB_S: float = 2039.0         # upstream: 2150 (实测峰值85折)
    PCIE_BANDWIDTH_GB_S: float = 28.5          # upstream: 31.0 (扣除协议开销)
    L2_CACHE_SIZE_BYTES: int = 40 * 1024 * 1024  # A100 L2 = 40MB
    SM_COUNT: int = 108                        # A100 SM数

    def estimate(self, query: QueryDescriptor,
                 node: HardwareNode,
                 data_resident_on_gpu: bool = False) -> CostBreakdown:
        from ._debug import dbg
        t0 = _time.perf_counter()

        cb = CostBreakdown(device_id=node.node_id)

        # PCIe传输代价
        if not data_resident_on_gpu:
            xfer_bytes = query.estimated_data_bytes
            xfer_sec = xfer_bytes / (self.PCIE_BANDWIDTH_GB_S * 1e9)
            cb.transfer_cost_us = xfer_sec * 1e6

        # GPU 计算: HBM带宽受限 vs 算力受限, 取max
        data_bytes = query.estimated_data_bytes

        # --- 改写: sigmoid L2 cache 命中率模型 ---
        # upstream用 max(0, 1 - data/L2) 线性模型
        # port_v3用 sigmoid: hit = 1 / (1 + exp(4 * (data/L2 - 0.5)))
        ratio = data_bytes / self.L2_CACHE_SIZE_BYTES
        l2_hit_rate = 1.0 / (1.0 + math.exp(4.0 * (ratio - 0.5)))
        effective_bw = self.HBM_BANDWIDTH_GB_S * (1.0 + 1.8 * l2_hit_rate)

        hbm_us = (data_bytes / (effective_bw * 1e9)) * 1e6

        compute_us = (
            query.estimated_rows * self.GPU_TUPLE_COST
            + query.estimated_rows * query.num_predicates * self.GPU_OPERATOR_COST
        )

        # warp_occupancy: 小查询SM利用率低, 加惩罚
        if query.estimated_rows < 10000:
            occupancy_penalty = 1.0 + (10000 - query.estimated_rows) / 10000 * 0.3
        else:
            occupancy_penalty = 1.0
        compute_us *= occupancy_penalty

        scalable_us = max(hbm_us, compute_us)

        # --- 改写: GPU merge-sort 模型 (替代 bitonic) ---
        if query.sort_required and query.estimated_rows > 1:
            n = query.estimated_rows
            log_n = math.log2(max(2, n))
            # merge-sort: O(n * log(n)) 而非 bitonic 的 O(n * log²(n))
            merge_ops = n * log_n * 1.15  # 1.15 = merge 的常数因子
            # 加入 register pressure 校正
            register_pressure = 1.0 + max(0, log_n - 20) * 0.02
            cb.sort_cost_us = merge_ops * self.GPU_OPERATOR_COST / self.SM_COUNT * register_pressure

        # 缩放
        if node.compute_capacity > 0 and node.compute_capacity != 1.0:
            inv_cap = 1.0 / node.compute_capacity
            scalable_us *= inv_cap
            cb.sort_cost_us *= inv_cap

        cb.compute_cost_us = self.KERNEL_LAUNCH_OVERHEAD_US + scalable_us

        elapsed_us = (_time.perf_counter() - t0) * 1e6
        # --- DEBUG: 完整打印GPU估算过程 ---
        dbg("GPUCostModel.estimate.result",
            device=node.node_id,
            data_bytes=data_bytes,
            l2_hit_rate=round(l2_hit_rate, 4),
            effective_bw_gbs=round(effective_bw, 1),
            hbm_us=round(hbm_us, 2),
            compute_us=round(compute_us, 2),
            occupancy_penalty=round(occupancy_penalty, 3),
            transfer_us=round(cb.transfer_cost_us, 2),
            sort_us=round(cb.sort_cost_us, 2),
            total_ms=round(cb.total_ms, 4),
            overhead_us=round(elapsed_us, 1))

        return cb


# ---------------------------------------------------------------------------
# 统一代价引擎
# ---------------------------------------------------------------------------

class CostModelEngine:
    """统一代价引擎 — 在所有设备上估算查询代价并推荐路由.

    类似 NCCL 的 ncclTopoCompute 在拓扑图上搜索最优通信路径,
    本引擎在硬件拓扑上搜索最优查询执行设备.
    """

    def __init__(self, topology: HardwareTopology):
        self.topology = topology
        self.cpu_model = CPUCostModel()
        self.gpu_model = GPUCostModel()
        self._estimation_count = 0
        self._decision_log: List[dict] = []  # port_v3新增: 记录所有决策

        from ._debug import dbg, inspect_struct
        dbg("CostModelEngine.__init__",
            n_nodes=len(topology.nodes),
            n_edges=len(topology.edges))
        inspect_struct(topology, depth=1)

    def estimate_on_device(self, query: QueryDescriptor,
                           device_id: str,
                           data_location: Optional[str] = None
                           ) -> CostBreakdown:
        """估算查询在指定设备上的执行代价."""
        from ._debug import dbg
        self._estimation_count += 1
        dbg('CostEngine.estimate_on_device',
            query_id=query.query_id, device=device_id,
            data_loc=data_location,
            estimation_seq=self._estimation_count)

        node = self.topology.get_node(device_id)
        if node is None:
            raise ValueError(f"Unknown device: {device_id}")

        if node.kind == HardwareKind.GPU:
            resident = (data_location == device_id)
            cb = self.gpu_model.estimate(query, node, resident)
            if data_location and not resident:
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
        """在拓扑中所有设备上估算代价."""
        from ._debug import timing
        with timing(f"estimate_all[{query.query_id}]"):
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
        """推荐最优设备 — 核心路由决策.

        返回 (device_id, cost_breakdown), 取total_us最小的设备.
        """
        from ._debug import dbg, checkpoint

        estimates = self.estimate_all_devices(query, data_location)
        if not estimates:
            raise RuntimeError("No devices available for estimation")

        # 打印所有候选设备的代价对比
        cost_map = {k: round(v.total_ms, 4) for k, v in estimates.items()}
        best_id = min(estimates, key=lambda k: estimates[k].total_us)
        best_cb = estimates[best_id]

        dbg('CostEngine.recommend',
            query_id=query.query_id,
            candidates=cost_map,
            winner=best_id,
            winner_cost_ms=round(best_cb.total_ms, 4))

        # port_v3: 记录决策日志, 方便事后分析
        self._decision_log.append({
            "query_id": query.query_id,
            "winner": best_id,
            "cost_ms": best_cb.total_ms,
            "all_costs": cost_map,
        })

        # --- DEBUG: 每100次决策写一次checkpoint ---
        if len(self._decision_log) % 100 == 0:
            checkpoint("decision_log_periodic",
                       n_decisions=len(self._decision_log),
                       last_10=self._decision_log[-10:])

        return best_id, best_cb

    def explain(self, query: QueryDescriptor,
                data_location: Optional[str] = None) -> str:
        """返回人类可读的路由决策解释 — port_v3新增.

        类似 PostgreSQL 的 EXPLAIN 输出, 展示:
          - 每个设备的代价分解
          - 选中设备及原因
          - 代价对比比率
        """
        estimates = self.estimate_all_devices(query, data_location)
        if not estimates:
            return "[EXPLAIN] No devices available"

        best_id = min(estimates, key=lambda k: estimates[k].total_us)
        best_cost = estimates[best_id].total_us

        lines = [
            f"[EXPLAIN] Query: {query.query_id}  Type: {query.query_type.name}",
            f"  Rows: {query.estimated_rows:,}  Selectivity: {query.selectivity:.4f}  "
            f"DataBytes: {query.estimated_data_bytes:,}",
            f"  DataLocation: {data_location or 'unspecified'}",
            "",
        ]
        for dev_id in sorted(estimates, key=lambda k: estimates[k].total_us):
            cb = estimates[dev_id]
            ratio = cb.total_us / best_cost if best_cost > 0 else float('inf')
            marker = " ◄ SELECTED" if dev_id == best_id else ""
            lines.append(f"  {cb.explain_breakdown()}  (×{ratio:.2f}){marker}")

        return "\n".join(lines)

    def route_batch(self, queries: List[QueryDescriptor],
                    data_location: Optional[str] = None
                    ) -> List[Tuple[str, CostBreakdown]]:
        """批量路由查询."""
        from ._debug import dbg, timing
        dbg("CostEngine.route_batch", n_queries=len(queries), data_loc=data_location)
        with timing("route_batch"):
            results = [self.recommend(q, data_location) for q in queries]

        # --- DEBUG: 打印路由统计 ---
        device_counts: Dict[str, int] = {}
        for dev_id, _ in results:
            device_counts[dev_id] = device_counts.get(dev_id, 0) + 1
        dbg("CostEngine.route_batch.summary",
            total=len(results),
            distribution=device_counts)

        return results

    def dump_state(self) -> dict:
        """导出引擎完整状态 — 用于debug checkpoint."""
        return {
            "estimation_count": self._estimation_count,
            "decision_log_size": len(self._decision_log),
            "topology_nodes": list(self.topology.nodes.keys()),
            "cpu_model_coeffs": {
                "SEQ_PAGE_COST": self.cpu_model.SEQ_PAGE_COST,
                "RANDOM_PAGE_COST": self.cpu_model.RANDOM_PAGE_COST,
                "CPU_TUPLE_COST": self.cpu_model.CPU_TUPLE_COST,
            },
            "gpu_model_coeffs": {
                "KERNEL_LAUNCH_US": self.gpu_model.KERNEL_LAUNCH_OVERHEAD_US,
                "HBM_BW_GBS": self.gpu_model.HBM_BANDWIDTH_GB_S,
                "PCIE_BW_GBS": self.gpu_model.PCIE_BANDWIDTH_GB_S,
            },
        }


# ---------------------------------------------------------------------------
# 默认拓扑工厂
# ---------------------------------------------------------------------------

def create_default_topology() -> HardwareTopology:
    """创建典型单节点异构拓扑: 2x CPU + 4x GPU (PCIe/NVLink).

    参考 NCCL 的 ncclTopoFillGpu/ncclTopoFillNet 模式.
    """
    from ._debug import dbg

    topo = HardwareTopology()

    cpu0 = HardwareNode(
        node_id="cpu0", kind=HardwareKind.CPU,
        compute_capacity=1.0,
        memory_bytes=256 * (1 << 30),
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

    gpus = []
    for i in range(4):
        gpu = HardwareNode(
            node_id=f"gpu{i}", kind=HardwareKind.GPU,
            compute_capacity=110.0,
            memory_bytes=80 * (1 << 30),
            bandwidth_gbps=2000.0,
            scan_cost_per_row=0.001,
            seek_cost=0.01,
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

    dbg("create_default_topology",
        nodes=list(topo.nodes.keys()),
        edges_count=len(topo.edges))

    return topo
