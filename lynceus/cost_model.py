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
    SEQ_PAGE_COST: float = 0.02      # sequential page read (prefetched, ~20ns)
    RANDOM_PAGE_COST: float = 0.5    # random page read (~500ns DRAM access)
    CPU_TUPLE_COST: float = 0.05     # per-tuple processing (~50ns)
    CPU_OPERATOR_COST: float = 0.01  # per-predicate evaluation (~10ns)
    CPU_INDEX_TUPLE_COST: float = 0.02  # per-index-tuple fetch (~20ns)
    PAGE_SIZE: int = 8192

    def estimate(self, query: QueryDescriptor,
                 node: HardwareNode) -> CostBreakdown:
        cb = CostBreakdown(device_id=node.node_id)

        pages = max(1, query.estimated_data_bytes // self.PAGE_SIZE)
        total_pages = max(1, query.full_table_bytes // self.PAGE_SIZE)

        # I/O cost
        if query.query_type == QueryType.FULL_TABLE_SCAN:
            cb.io_cost_us = total_pages * self.SEQ_PAGE_COST * node.scan_cost_per_row
            if node.scan_cost_per_row == 0:
                cb.io_cost_us = total_pages * self.SEQ_PAGE_COST
        elif query.index_available and query.query_type in (
            QueryType.POINT_LOOKUP, QueryType.INDEX_SCAN, QueryType.RANGE_SCAN
        ):
            # Index scan: random I/O for index traversal + sequential for heap
            index_pages = query.index_depth + max(1, int(
                pages * query.selectivity
            ))
            cb.index_cost_us = (index_pages * self.RANDOM_PAGE_COST +
                                query.estimated_rows * self.CPU_INDEX_TUPLE_COST)
            cb.io_cost_us = pages * query.selectivity * self.SEQ_PAGE_COST
        else:
            # Range/table scan without index
            cb.io_cost_us = pages * self.SEQ_PAGE_COST

        # CPU compute cost
        cb.compute_cost_us = (
            query.estimated_rows * self.CPU_TUPLE_COST +
            query.estimated_rows * query.num_predicates * self.CPU_OPERATOR_COST
        )

        # Sort cost (if required)
        if query.sort_required and query.estimated_rows > 1:
            n = query.estimated_rows
            cb.sort_cost_us = (
                2.0 * n * math.log2(max(2, n)) * self.CPU_OPERATOR_COST
            )

        # Scale by node's relative capacity (inverse: slower node → higher cost)
        if node.compute_capacity > 0 and node.compute_capacity != 1.0:
            scale = 1.0 / node.compute_capacity
            cb.compute_cost_us *= scale
            cb.sort_cost_us *= scale

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

    KERNEL_LAUNCH_OVERHEAD_US: float = 10.0
    GPU_TUPLE_COST: float = 0.0001    # ~100x faster than CPU per tuple
    GPU_OPERATOR_COST: float = 0.00005
    HBM_BANDWIDTH_GB_S: float = 2000.0  # A100-class HBM bandwidth
    PCIE_BANDWIDTH_GB_S: float = 32.0    # PCIe Gen4 x16

    def estimate(self, query: QueryDescriptor,
                 node: HardwareNode,
                 data_resident_on_gpu: bool = False) -> CostBreakdown:
        cb = CostBreakdown(device_id=node.node_id)

        # PCIe transfer cost (if data not already on GPU)
        if not data_resident_on_gpu:
            transfer_bytes = query.estimated_data_bytes
            transfer_seconds = transfer_bytes / (self.PCIE_BANDWIDTH_GB_S * 1e9)
            cb.transfer_cost_us = transfer_seconds * 1e6

        # Kernel launch overhead (FIXED — not scalable by compute_capacity)
        kernel_launch_us = self.KERNEL_LAUNCH_OVERHEAD_US

        # GPU compute: massively parallel
        # Model as HBM-bandwidth-bound for scans
        data_bytes = query.estimated_data_bytes
        hbm_seconds = data_bytes / (self.HBM_BANDWIDTH_GB_S * 1e9)
        hbm_us = hbm_seconds * 1e6

        # Compute-bound component (predicate evaluation etc.)
        compute_us = (
            query.estimated_rows * self.GPU_TUPLE_COST +
            query.estimated_rows * query.num_predicates * self.GPU_OPERATOR_COST
        )

        # GPU is either compute-bound or memory-bound
        scalable_compute_us = max(hbm_us, compute_us)

        # GPU sort: bitonic sort or radix sort — O(n log²n) but massively parallel
        if query.sort_required and query.estimated_rows > 1:
            n = query.estimated_rows
            # GPU sort: ~n * log²(n) / num_SMs operations
            num_sms = 108  # A100
            ops = n * (math.log2(max(2, n)) ** 2)
            cb.sort_cost_us = ops * self.GPU_OPERATOR_COST / num_sms

        # Scale ONLY the compute-bound portion by node capacity
        # (kernel launch overhead is fixed regardless of GPU FLOPS)
        if node.compute_capacity > 0 and node.compute_capacity != 1.0:
            scale = 1.0 / node.compute_capacity
            scalable_compute_us *= scale
            cb.sort_cost_us *= scale

        cb.compute_cost_us = kernel_launch_us + scalable_compute_us

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

        Returns (device_id, cost_breakdown) with the lowest total cost.

        This is the core routing decision — analogous to NCCL's
        ncclTopoCompute choosing the best algorithm/protocol path.
        """
        estimates = self.estimate_all_devices(query, data_location)
        if not estimates:
            raise RuntimeError("No devices available for estimation")

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
            compute_capacity=100.0,  # ~100x CPU FLOPS
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

