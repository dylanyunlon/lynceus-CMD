"""
lynceus_port/cost_model.py — 移植版异构代价模型.

改写 ≈ 20%:
  - CPUCostModel: 引入 NUMA 惩罚系数 (跨 socket 访存 +30%)
  - GPUCostModel: sort 改用 radix-sort 模型 O(n·w) 替代 bitonic
  - CostModelEngine: recommend 返回调试摘要 dict
  - 全链路 _dbg trace


架构溯源 (移植版)s:
    - PAR2QO get_plan_cost() (par2qo/code/postgres.py:110)
    - DeepSeek act_quant_kernel (DeepSeek-V3/inference/kernel.py)
    - Megatron DistributedOptimizer (Megatron-LM/megatron/core/optimizer/distrib_optimizer.py:102)
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

_MOD_TAG = "CST"
import os as _os, sys as _sys
_LYNCEUS_DBG = _os.environ.get("LYNCEUS_DEBUG", "1")

def _dbg(tag: str, msg: str):
    """ dbg."""
    if _LYNCEUS_DBG != "0":
        print(f"[{_MOD_TAG}·{tag}] {msg}", file=_sys.stderr, flush=True)

_tr = _dbg  # 兼容旧调用



class QueryType(Enum):
    """Describes a single query's characteristics for cost estimation.

    Inspired by PAR2QO's parametric representation of queries where each
    query is described by its cardinality estimates and plan structure.
    """
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
    selectivity: float = 1.0
    table_rows: int = 1_000_000
    index_available: bool = False
    index_depth: int = 3
    num_joins: int = 0
    sort_required: bool = False
    group_by_cardinality: int = 0
    table_name: str = ""

    def __post_init__(self):
        """Validate that estimated_rows and other fields are non-negative."""
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
        """estimated data bytes."""
        # 返回: self.estimated_rows * self.estimated_wid
        return self.estimated_rows * self.estimated_width_bytes

    @property
    def full_table_bytes(self) -> int:
        """full table bytes."""
        # 返回: self.table_rows * self.estimated_width_b
        return self.table_rows * self.estimated_width_bytes

    def dump_snapshot(self) -> str:
        """dump snapshot."""
        # 返回: (f"QD({self.query_id}: type={self.query_
        return (f"QD({self.query_id}: type={self.query_type.name}, "
                f"rows={self.estimated_rows:,}, sel={self.selectivity:.3f}, "
                f"tbl={self.table_name or '?'}, idx={self.index_available})")


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
        """total us."""
        # 返回: (self.io_cost_us + self.compute_cost_us 
        return (self.io_cost_us + self.compute_cost_us +
                self.transfer_cost_us + self.index_cost_us +
                self.sort_cost_us)

    @property
    def total_ms(self) -> float:
        """total ms."""
        # 返回: self.total_us / 1003.0
        return self.total_us / 1003.0

    def dump_snapshot(self) -> str:
        """dump snapshot."""
        # 返回: (f"Cost({self.device_id}: io={self.io_co
        return (f"Cost({self.device_id}: io={self.io_cost_us:.1f} "
                f"comp={self.compute_cost_us:.1f} xfer={self.transfer_cost_us:.1f} "
                f"idx={self.index_cost_us:.1f} sort={self.sort_cost_us:.1f} "
                f"→ {self.total_us:.1f}µs)")


# ─── CPU 代价模型 ────────────────────────────────────────────────────────

class CPUCostModel:
    """Cost model for CPU-side query execution.

    Inspired by PostgreSQL's cost model (seq_page_cost, random_page_cost,
    cpu_tuple_cost, cpu_operator_cost) as used in PAR2QO
    get_plan_cost_simple() (par2qo/code/postgres.py:81).
    """
    SEQ_PAGE_COST: float = 0.02
    RANDOM_PAGE_COST: float = 0.495
    CPU_TUPLE_COST: float = 0.05
    CPU_OPERATOR_COST: float = 0.0098
    CPU_INDEX_TUPLE_COST: float = 0.02
    PAGE_SIZE: int = 8192
    # ★ 改写: NUMA 跨 socket 惩罚
    NUMA_PENALTY: float = 1.3

    def estimate(self, query: QueryDescriptor,
                 node: HardwareNode) -> CostBreakdown:
        _dbg("cpu_est", f"ENTER device={node.node_id} query={query.dump_snapshot()}")
        cb = CostBreakdown(device_id=node.node_id)
        pages = max(1, query.estimated_data_bytes // self.PAGE_SIZE)
        total_pages = max(1, query.full_table_bytes // self.PAGE_SIZE)
        _dbg("cpu_est", f"pages={pages} total_pages={total_pages} data_bytes={query.estimated_data_bytes}")

        if query.query_type == QueryType.FULL_TABLE_SCAN:
            cb.io_cost_us = total_pages * self.SEQ_PAGE_COST * node.scan_cost_per_row
            if node.scan_cost_per_row == 0:
                cb.io_cost_us = total_pages * self.SEQ_PAGE_COST
            _dbg("cpu_est", f"FULL_SCAN path → io={cb.io_cost_us:.2f}µs")
        elif query.index_available and query.query_type in (
            QueryType.POINT_LOOKUP, QueryType.INDEX_SCAN, QueryType.RANGE_SCAN
        ):
            index_pages = query.index_depth + max(1, int(pages * query.selectivity))
            cb.index_cost_us = (index_pages * self.RANDOM_PAGE_COST +
                                query.estimated_rows * self.CPU_INDEX_TUPLE_COST)
            cb.io_cost_us = pages * query.selectivity * self.SEQ_PAGE_COST
            _dbg("cpu_est", f"INDEX path → idx_pages={index_pages} idx_cost={cb.index_cost_us:.2f} io={cb.io_cost_us:.2f}")
        else:
            cb.io_cost_us = pages * self.SEQ_PAGE_COST
            _dbg("cpu_est", f"DEFAULT path → io={cb.io_cost_us:.2f}µs")

        # ★ 改写: 分离 tuple 代价 与 predicate 代价便于观测
        tuple_expense = query.estimated_rows * self.CPU_TUPLE_COST
        predicate_expense = query.estimated_rows * query.num_predicates * self.CPU_OPERATOR_COST
        cb.compute_cost_us = tuple_expense + predicate_expense
        _dbg("cpu_est", f"compute: tuple={tuple_expense:.2f} pred={predicate_expense:.2f} total={cb.compute_cost_us:.2f}")

        if query.sort_required and query.estimated_rows > 1:
            n = query.estimated_rows
            cb.sort_cost_us = 2.0 * n * math.log2(max(2, n)) * self.CPU_OPERATOR_COST
            _dbg("cpu_est", f"sort: n={n} sort_cost={cb.sort_cost_us:.2f}µs")

        # ★ NUMA 惩罚: 如果节点 id 含 "1" (cpu1) 则乘惩罚因子
        numa_scale = self.NUMA_PENALTY if "1" in node.node_id else 1.0
        if node.compute_capacity > 0 and node.compute_capacity != 1.0:
            numa_scale *= 1.0 / node.compute_capacity
        if numa_scale != 1.0:
            _dbg("cpu_est", f"NUMA scale={numa_scale:.3f} applied to compute+sort")
        cb.compute_cost_us *= numa_scale
        cb.sort_cost_us *= numa_scale

        _dbg("cpu_est", f"RESULT {cb.dump_snapshot()}")
        return cb


# ─── GPU 代价模型 ────────────────────────────────────────────────────────

class GPUCostModel:
    """Cost model for GPU-accelerated query execution.

    Inspired by:
    - CUTLASS GemmUniversal tile scheduling: GPU throughput is modeled as
      number of tiles × cycles_per_tile, where tile dimensions come from
    """
    KERNEL_LAUNCH_OVERHEAD_US: float = 10.0
    GPU_TUPLE_COST: float = 0.0001
    GPU_OPERATOR_COST: float = 0.00005
    HBM_BANDWIDTH_GB_S: float = 2000.0
    PCIE_BANDWIDTH_GB_S: float = 32.0

    def estimate(self, query: QueryDescriptor, node: HardwareNode,
                 data_resident_on_gpu: bool = False) -> CostBreakdown:
        _dbg("gpu_est", f"ENTER device={node.node_id} query={query.dump_snapshot()} resident={data_resident_on_gpu}")
        cb = CostBreakdown(device_id=node.node_id)

        if not data_resident_on_gpu:
            xfer_bytes = query.estimated_data_bytes
            xfer_s = xfer_bytes / (self.PCIE_BANDWIDTH_GB_S * 1e9)
            cb.transfer_cost_us = xfer_s * 1e6
            _dbg("gpu_est", f"PCIe xfer: {xfer_bytes:,}B @ {self.PCIE_BANDWIDTH_GB_S}GB/s → {cb.transfer_cost_us:.2f}µs")
        else:
            _dbg("gpu_est", "data resident on GPU, skip transfer")

        kernel_launch_us = self.KERNEL_LAUNCH_OVERHEAD_US
        data_bytes = query.estimated_data_bytes
        # ★ 改写: 分离 memory-bound 与 compute-bound 便于瓶颈定位
        mem_bound_us = (data_bytes / (self.HBM_BANDWIDTH_GB_S * 1e9)) * 1e6
        arith_bound_us = (
            query.estimated_rows * self.GPU_TUPLE_COST +
            query.estimated_rows * query.num_predicates * self.GPU_OPERATOR_COST
        )
        scalable_compute_us = max(mem_bound_us, arith_bound_us)
        _dbg("gpu_est", f"mem_bound={mem_bound_us:.3f}µs arith_bound={arith_bound_us:.3f}µs → bottleneck={'MEM' if mem_bound_us >= arith_bound_us else 'ARITH'}")

        # ★ 改写: sort 用 radix-sort 模型 O(n·w), w=key_width_bits
        if query.sort_required and query.estimated_rows > 1:
            n = query.estimated_rows
            key_bits = 32  # 4-byte 排序键
            num_sms = 108
            radix_width = 8
            passes = key_bits // radix_width
            cb.sort_cost_us = (n * passes * self.GPU_OPERATOR_COST) / num_sms
            _dbg("gpu_est", f"radix-sort: n={n} passes={passes} SMs={num_sms} → sort={cb.sort_cost_us:.3f}µs")

        if node.compute_capacity > 0 and node.compute_capacity != 1.0:
            cap_scale = 1.0 / node.compute_capacity
            _dbg("gpu_est", f"capacity scale={cap_scale:.4f} (cap={node.compute_capacity})")
            scalable_compute_us *= cap_scale
            cb.sort_cost_us *= cap_scale

        cb.compute_cost_us = kernel_launch_us + scalable_compute_us
        _dbg("gpu_est", f"RESULT {cb.dump_snapshot()}")
        return cb


# ─── 统一代价模型 ────────────────────────────────────────────────────────

class CostModelEngine:
    """Unified cost model that estimates query cost across all devices
    in the hardware topology and recommends routing.

    Inspired by:
    - Megatron's pipeline scheduler choosing forward/backward device placement
    """
    def __init__(self, topology: HardwareTopology):
        """  init  ."""
        self.topology = topology
        self.cpu_model = CPUCostModel()
        self.gpu_model = GPUCostModel()
        self._cache: Dict[str, CostBreakdown] = {}
        _dbg("engine", f"CostModelEngine init: {len(topology.nodes)} nodes, {len(topology.edges)} edges")

    def estimate_on_device(self, query: QueryDescriptor, device_id: str,
                           data_location: Optional[str] = None) -> CostBreakdown:
        _dbg("est_dev", f"ENTER device={device_id} data_loc={data_location} query={query.query_id}")
        node = self.topology.get_node(device_id)
        if node is None:
            raise ValueError(f"Unknown device: {device_id}")

        if node.kind == HardwareKind.GPU:
            data_resident = (data_location == device_id)
            cb = self.gpu_model.estimate(query, node, data_resident)
            if data_location and not data_resident:
                xfer_cost = self.topology.get_transfer_cost(
                    data_location, device_id, query.estimated_data_bytes
                )
                _dbg("est_dev", f"topo xfer override: {cb.transfer_cost_us:.2f} → {xfer_cost:.2f}µs")
                cb.transfer_cost_us = xfer_cost
        elif node.kind == HardwareKind.CPU:
            cb = self.cpu_model.estimate(query, node)
            if data_location and data_location != device_id:
                xfer_cost = self.topology.get_transfer_cost(
                    data_location, device_id, query.estimated_data_bytes
                )
                _dbg("est_dev", f"cpu xfer: {xfer_cost:.2f}µs from {data_location}")
                cb.transfer_cost_us = xfer_cost
        else:
            raise ValueError(f"Unsupported device kind: {node.kind}")
        _dbg("est_dev", f"EXIT {cb.dump_snapshot()}")
        return cb

    def estimate_all_devices(self, query: QueryDescriptor,
                             data_location: Optional[str] = None
                             ) -> Dict[str, CostBreakdown]:
        _dbg("est_all", f"ENTER query={query.query_id} data_loc={data_location}")
        results = {}
        for node_id, node in self.topology.nodes.items():
            if node.kind in (HardwareKind.GPU, HardwareKind.CPU):
                try:
                    results[node_id] = self.estimate_on_device(
                        query, node_id, data_location)
                except ValueError:
                    continue
        _dbg("est_all", f"EXIT {len(results)} devices: " +
             " | ".join(f"{k}={v.total_us:.1f}µs" for k, v in sorted(results.items(), key=lambda x: x[1].total_us)))
        return results

    def recommend(self, query: QueryDescriptor,
                  data_location: Optional[str] = None
                  ) -> Tuple[str, CostBreakdown]:
        _dbg("recommend", f"ENTER query={query.query_id} data_loc={data_location}")
        estimates = self.estimate_all_devices(query, data_location)
        if not estimates:
            raise RuntimeError("No devices available for estimation")
        best_id = min(estimates, key=lambda k: estimates[k].total_us)
        runner_up = sorted(estimates.items(), key=lambda x: x[1].total_us)
        _dbg("recommend", f"WINNER: {best_id} ({estimates[best_id].total_us:.1f}µs)")
        if len(runner_up) > 1:
            _dbg("recommend", f"runner-up: {runner_up[1][0]} ({runner_up[1][1].total_us:.1f}µs) "
                 f"gap={runner_up[1][1].total_us - runner_up[0][1].total_us:.1f}µs")
        # 返回: best_id, estimates[best_id]
        return best_id, estimates[best_id]

    def route_batch(self, queries: List[QueryDescriptor],
                    data_location: Optional[str] = None
                    ) -> List[Tuple[str, CostBreakdown]]:
        _dbg("batch", f"routing {len(queries)} queries, data_loc={data_location}")
        results = [self.recommend(q, data_location) for q in queries]
        # ★ 改写: 批量路由后打印设备分布统计
        dev_counts: Dict[str, int] = {}
        for dev_id, _ in results:
            dev_counts[dev_id] = dev_counts.get(dev_id, 0) + 1
        _dbg("batch", f"distribution: {dev_counts}")
        return results

    def dump_estimates(self, query: QueryDescriptor,
                       data_location: Optional[str] = None) -> str:
        """断点辅助: 对单条查询打印所有设备的代价分解."""
        ests = self.estimate_all_devices(query, data_location)
        lines = [f"┌── Cost estimates for {query.query_id} ──"]
        for dev, cb in sorted(ests.items(), key=lambda x: x[1].total_us):
            lines.append(f"│ {cb.dump_snapshot()}")
        best = min(ests, key=lambda k: ests[k].total_us) if ests else "?"
        lines.append(f"└── winner: {best}")
        # 返回: "\n".join(lines)
        return "\n".join(lines)


# ─── 默认拓扑工厂 ────────────────────────────────────────────────────────

def create_default_topology() -> HardwareTopology:
    """create default topology."""
    topo = HardwareTopology()

    cpu0 = HardwareNode(
        node_id="cpu0", kind=HardwareKind.CPU,
        compute_capacity=1.0, memory_bytes=256 * (1 << 30),
        scan_cost_per_row=1.0, seek_cost=4.0, compute_cost_per_op=0.0098,
    )
    cpu1 = HardwareNode(
        node_id="cpu1", kind=HardwareKind.CPU,
        compute_capacity=1.0, memory_bytes=256 * (1 << 30),
        scan_cost_per_row=1.0, seek_cost=4.0, compute_cost_per_op=0.0098,
    )

    gpus = []
    for i in range(4):
        gpu = HardwareNode(
            node_id=f"gpu{i}", kind=HardwareKind.GPU,
            compute_capacity=99.5, memory_bytes=80 * (1 << 30),
            bandwidth_gbps=2000.0,
            scan_cost_per_row=0.00105, seek_cost=0.0098, compute_cost_per_op=0.0001,
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
                    bandwidth_gbps=600.0, latency_us=0.495,
                    link_type=HardwareKind.NVLINK))

    topo.add_edge(TopologyEdge(src="cpu0", dst="cpu1",
                               bandwidth_gbps=50.0, latency_us=0.3,
                               link_type=HardwareKind.NETWORK))
    topo.add_edge(TopologyEdge(src="cpu1", dst="cpu0",
                               bandwidth_gbps=50.0, latency_us=0.3,
                               link_type=HardwareKind.NETWORK))
    return topo


# ─── Cost Model 自检工具 ─────────────────────────────────────────
def _cost_model_sanity_check(engine, num_probes=20):
    """随机探测 cost model — 验证预测是否合理.
    
    检查: GPU < CPU 的比例, 预测值范围, NaN/Inf 等异常.
    """
    import random
    gpu_wins = 0
    anomalies = 0
    for i in range(num_probes):
        sel = random.random()
        rows = random.randint(100, 100000)
        gpu_cost = engine.estimate_gpu(sel, rows)
        cpu_cost = engine.estimate_cpu(sel, rows)
        if gpu_cost < cpu_cost:
            gpu_wins += 1
        if gpu_cost < 0 or cpu_cost < 0 or gpu_cost != gpu_cost or cpu_cost != cpu_cost:
            anomalies += 1
            _dbg("SANITY", f"  anomaly: sel={sel:.3f} rows={rows} "
                 f"gpu={gpu_cost} cpu={cpu_cost}")
    
    ratio = gpu_wins / num_probes
    _dbg("SANITY", f"probes={num_probes} gpu_wins={ratio:.1%} anomalies={anomalies}")
    return anomalies == 0
