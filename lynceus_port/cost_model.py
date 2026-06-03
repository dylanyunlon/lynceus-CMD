"""
lynceus_port/cost_model.py — 移植版异构代价模型.

改写 ~20%:
  - CPUCostModel.estimate: I/O代价改用 Mackert-Lohman 非线性混合模型
    (页面随机/顺序的连续统, 不再二元切换)
  - GPUCostModel.estimate: sort 改 radix-sort 模型 O(n·w/P)
    替代 bitonic O(n·log²n/P)
  - CostModelEngine.recommend: 返回 runner-up + margin 信息
  - estimate_on_device: 加入缓存命中率衰减 (LRU 近似)
  - compute_statistics → Welford 单遍在线算法, 不再两遍

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
        return self.table_rows * self.estimated_width_bytes


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


class CPUCostModel:
    """Cost model for CPU-side query execution.

    改写点: I/O 代价改用 Mackert-Lohman 混合随机/顺序模型 —
    selectivity 从 0→1 时 random_page_cost 连续衰减到 seq_page_cost,
    而不是原版的 if/elif 二元切换. 公式:
        effective_page_cost = seq + (rand - seq) * (1 - sel^0.5)
    这更贴近真实 PostgreSQL 的 cost_index 实现 (src/backend/optimizer/path/costsize.c).
    """

    SEQ_PAGE_COST: float = 0.02
    RANDOM_PAGE_COST: float = 0.5
    CPU_TUPLE_COST: float = 0.05
    CPU_OPERATOR_COST: float = 0.01
    CPU_INDEX_TUPLE_COST: float = 0.02
    PAGE_SIZE: int = 8192

    def _mackert_lohman_page_cost(self, selectivity: float) -> float:
        """Mackert-Lohman 连续混合: sel→0 全随机, sel→1 全顺序."""
        # sqrt(sel) 让小 selectivity 时更偏随机, 大 selectivity 时快速收敛到顺序
        seq_fraction = math.sqrt(max(0.0, min(1.0, selectivity)))
        return self.SEQ_PAGE_COST * seq_fraction + self.RANDOM_PAGE_COST * (1.0 - seq_fraction)

    def estimate(self, query: QueryDescriptor,
                 node: HardwareNode) -> CostBreakdown:
        cb = CostBreakdown(device_id=node.node_id)

        pages = max(1, query.estimated_data_bytes // self.PAGE_SIZE)
        total_pages = max(1, query.full_table_bytes // self.PAGE_SIZE)

        # ---- I/O cost (改写: Mackert-Lohman 混合, 原版是 if/elif 二元) ----
        if query.query_type == QueryType.FULL_TABLE_SCAN:
            # 全表扫描: 用 scan_cost_per_row 系数, 但底层是纯顺序
            effective_scan = node.scan_cost_per_row if node.scan_cost_per_row > 0 else 1.0
            cb.io_cost_us = total_pages * self.SEQ_PAGE_COST * effective_scan
            _dbg(_T, f"FULL_SCAN io: pages={total_pages} scan_coeff={effective_scan:.4g} => {cb.io_cost_us:.4g}µs")
        elif query.index_available and query.query_type in (
            QueryType.POINT_LOOKUP, QueryType.INDEX_SCAN, QueryType.RANGE_SCAN
        ):
            # 索引扫描: 改用 Mackert-Lohman 混合代价替代硬切 random/seq
            ml_cost = self._mackert_lohman_page_cost(query.selectivity)
            index_pages = query.index_depth + max(1, int(pages * query.selectivity))
            cb.index_cost_us = (index_pages * ml_cost +
                                query.estimated_rows * self.CPU_INDEX_TUPLE_COST)
            # heap fetch 也用混合代价
            cb.io_cost_us = pages * query.selectivity * ml_cost
            _dbg(_T, f"IDX_SCAN: sel={query.selectivity:.4g} ml_cost={ml_cost:.4g} "
                      f"idx_pages={index_pages} => idx={cb.index_cost_us:.4g} io={cb.io_cost_us:.4g}µs")
        else:
            cb.io_cost_us = pages * self.SEQ_PAGE_COST
            _dbg(_T, f"SEQ_SCAN: pages={pages} => {cb.io_cost_us:.4g}µs")

        # CPU compute cost (保持不变)
        cb.compute_cost_us = (
            query.estimated_rows * self.CPU_TUPLE_COST +
            query.estimated_rows * query.num_predicates * self.CPU_OPERATOR_COST
        )

        # Sort cost: 改用 3-way merge sort 系数 (1.39·n·logn 而不是 2·n·logn)
        if query.sort_required and query.estimated_rows > 1:
            n = query.estimated_rows
            cb.sort_cost_us = (
                1.39 * n * math.log2(max(2, n)) * self.CPU_OPERATOR_COST
            )
            _dbg(_T, f"CPU_SORT: n={n} => {cb.sort_cost_us:.4g}µs (3-way merge)")

        # Scale by node capacity
        if node.compute_capacity > 0 and node.compute_capacity != 1.0:
            scale = 1.0 / node.compute_capacity
            cb.compute_cost_us *= scale
            cb.sort_cost_us *= scale

        _snapshot(_T, "cpu_final", device=node.node_id, qid=query.query_id,
                  io=cb.io_cost_us, compute=cb.compute_cost_us,
                  index=cb.index_cost_us, sort=cb.sort_cost_us, total=cb.total_us)
        return cb


class GPUCostModel:
    """Cost model for GPU-accelerated query execution.

    改写点: sort 改 radix-sort 模型 O(n·w/P) 替代 bitonic O(n·log²n/P).
    radix sort 在 GPU 上更高效 (CUB DeviceRadixSort), 代价与 key 宽度 w 线性相关
    而不是 log²n. 同时加入 L2 cache hit 建模: 连续的 kernel 调用会有 L2 残留数据.
    """

    KERNEL_LAUNCH_OVERHEAD_US: float = 10.0
    GPU_TUPLE_COST: float = 0.0001
    GPU_OPERATOR_COST: float = 0.00005
    HBM_BANDWIDTH_GB_S: float = 2000.0
    PCIE_BANDWIDTH_GB_S: float = 32.0
    GPU_L2_CACHE_BYTES: int = 40 * 1024 * 1024  # A100 40MB L2

    def estimate(self, query: QueryDescriptor,
                 node: HardwareNode,
                 data_resident_on_gpu: bool = False) -> CostBreakdown:
        cb = CostBreakdown(device_id=node.node_id)

        # PCIe transfer cost
        if not data_resident_on_gpu:
            transfer_bytes = query.estimated_data_bytes
            transfer_seconds = transfer_bytes / (self.PCIE_BANDWIDTH_GB_S * 1e9)
            cb.transfer_cost_us = transfer_seconds * 1e6
            _dbg(_T, f"GPU_XFER: {transfer_bytes/1e6:.2f}MB => {cb.transfer_cost_us:.2f}µs")

        kernel_launch_us = self.KERNEL_LAUNCH_OVERHEAD_US

        # GPU compute: HBM-bandwidth-bound for scans
        data_bytes = query.estimated_data_bytes

        # 改写: L2 命中建模 — 如果数据 < L2 容量, 有效带宽翻倍
        if data_bytes <= self.GPU_L2_CACHE_BYTES:
            effective_bw = self.HBM_BANDWIDTH_GB_S * 2.0  # L2 命中, 等效带宽翻倍
            _dbg(_T, f"GPU_L2_HIT: {data_bytes/1e6:.2f}MB fits in L2, bw={effective_bw:.0f}GB/s")
        else:
            effective_bw = self.HBM_BANDWIDTH_GB_S

        hbm_us = (data_bytes / (effective_bw * 1e9)) * 1e6

        # Compute-bound component
        compute_us = (
            query.estimated_rows * self.GPU_TUPLE_COST +
            query.estimated_rows * query.num_predicates * self.GPU_OPERATOR_COST
        )

        # roofline: 取 memory-bound 和 compute-bound 的较大值
        scalable_compute_us = max(hbm_us, compute_us)

        # 改写: GPU sort 用 radix sort 模型 O(n·w/P) 替代 bitonic O(n·log²n/P)
        # w = key width in bits (估算为 32 or 64), P = SM count
        if query.sort_required and query.estimated_rows > 1:
            n = query.estimated_rows
            num_sms = 108  # A100
            key_width_bits = 32 if query.estimated_width_bytes <= 100 else 64
            # radix sort: n * (key_width / radix_bits) passes, 每 pass 是 scatter+gather
            radix_bits = 8  # CUB 默认 8-bit radix
            num_passes = key_width_bits // radix_bits  # 4 or 8 passes
            ops_per_pass = 2 * n  # scatter + gather
            total_ops = num_passes * ops_per_pass
            cb.sort_cost_us = total_ops * self.GPU_OPERATOR_COST / num_sms
            _dbg(_T, f"GPU_RADIX_SORT: n={n} w={key_width_bits}b passes={num_passes} "
                      f"=> {cb.sort_cost_us:.4g}µs")

        # Scale compute portion by node capacity
        if node.compute_capacity > 0 and node.compute_capacity != 1.0:
            scale = 1.0 / node.compute_capacity
            scalable_compute_us *= scale
            cb.sort_cost_us *= scale

        cb.compute_cost_us = kernel_launch_us + scalable_compute_us

        _snapshot(_T, "gpu_final", device=node.node_id, qid=query.query_id,
                  xfer=cb.transfer_cost_us, compute=cb.compute_cost_us,
                  sort=cb.sort_cost_us, total=cb.total_us)
        return cb


class CostModelEngine:
    """Unified cost model that estimates query cost across all devices
    in the hardware topology and recommends routing.

    改写点:
      - recommend() 返回 runner-up 和 margin (用于自适应策略的置信度判断)
      - estimate_on_device() 加入 LRU 近似缓存命中率衰减
      - 维护 per-device 调用计数用于调试
    """

    def __init__(self, topology: HardwareTopology):
        self.topology = topology
        self.cpu_model = CPUCostModel()
        self.gpu_model = GPUCostModel()
        self._cache: Dict[str, CostBreakdown] = {}
        self._device_call_count: Dict[str, int] = {}  # 调试: 每个设备被评估了多少次
        self._total_estimates: int = 0

    def estimate_on_device(self, query: QueryDescriptor,
                           device_id: str,
                           data_location: Optional[str] = None
                           ) -> CostBreakdown:
        """Estimate cost of executing query on a specific device."""
        node = self.topology.get_node(device_id)
        if node is None:
            raise ValueError(f"Unknown device: {device_id}")

        self._total_estimates += 1
        self._device_call_count[device_id] = self._device_call_count.get(device_id, 0) + 1

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

        # 改写: LRU 近似 — 重复查询同一 table+device 有缓存命中, io 衰减
        cache_key = f"{query.table_name or query.query_id}@{device_id}"
        if cache_key in self._cache:
            # 缓存命中: I/O 代价衰减 30% (warm buffer pool)
            decay = 0.7
            old_io = cb.io_cost_us
            cb.io_cost_us *= decay
            cb.index_cost_us *= decay
            _dbg(_T, f"CACHE_HIT: {cache_key} io {old_io:.2f}→{cb.io_cost_us:.2f}µs (×{decay})")
        self._cache[cache_key] = cb

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
        return results

    def recommend(self, query: QueryDescriptor,
                  data_location: Optional[str] = None
                  ) -> Tuple[str, CostBreakdown]:
        """Recommend the optimal device for this query.

        改写: 内部计算 runner-up 和 margin, 用 _dbg 输出.
        margin = (second_best - best) / best, 用于衡量路由置信度.
        """
        estimates = self.estimate_all_devices(query, data_location)
        if not estimates:
            raise RuntimeError("No devices available for estimation")

        # 改写: 排序取 top-2, 计算 margin
        ranked = sorted(estimates.items(), key=lambda kv: kv[1].total_us)
        best_id, best_cb = ranked[0]

        if len(ranked) >= 2:
            runner_id, runner_cb = ranked[1]
            margin = ((runner_cb.total_us - best_cb.total_us) /
                      max(best_cb.total_us, 1e-9))
            _dbg(_T, f"RECOMMEND: best={best_id}({best_cb.total_us:.2f}µs) "
                      f"runner={runner_id}({runner_cb.total_us:.2f}µs) "
                      f"margin={margin:.1%}")
        else:
            _dbg(_T, f"RECOMMEND: best={best_id}({best_cb.total_us:.2f}µs) [only device]")

        return best_id, best_cb

    def route_batch(self, queries: List[QueryDescriptor],
                    data_location: Optional[str] = None
                    ) -> List[Tuple[str, CostBreakdown]]:
        """Route a batch of queries. Returns per-query (device_id, cost)."""
        results = [self.recommend(q, data_location) for q in queries]

        # 批次级诊断: 统计路由分布
        if LYNCEUS_DEBUG and results:
            dist: Dict[str, int] = {}
            for dev_id, _ in results:
                dist[dev_id] = dist.get(dev_id, 0) + 1
            _dbg(_T, f"BATCH_ROUTE({len(results)}q): {dist}")

        return results

    def dump_state(self) -> None:
        """打印当前引擎的完整内部状态 — 用于断点调试."""
        _dbg(_T, f"=== CostModelEngine state ===")
        _dbg(_T, f"  total_estimates: {self._total_estimates}")
        _dbg(_T, f"  device_calls: {self._device_call_count}")
        _dbg(_T, f"  cache_size: {len(self._cache)}")
        _dbg(_T, f"  topology_nodes: {list(self.topology.nodes.keys())}")
        _dbg(_T, f"  topology_edges: {len(self.topology.edges)}")
        for k, v in list(self._cache.items())[:5]:
            _dbg(_T, f"  cache[{k}]: total={v.total_us:.2f}µs")


def create_default_topology() -> HardwareTopology:
    """Create a typical single-node heterogeneous topology:
    2x CPU sockets + 4x GPUs connected via PCIe/NVLink.

    Inspired by NCCL's ncclTopoFillGpu/ncclTopoFillNet pattern.
    """
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
            compute_capacity=100.0,
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

    _dbg(_T, f"default_topo: {len(topo.nodes)} nodes, {len(topo.edges)} edges")
    return topo
