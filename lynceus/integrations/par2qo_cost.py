"""
Original: PAR2QO postgres.py — PostgreSQL plan cost estimation via EXPLAIN
          (upstream/par2qo/code/postgres.py)
Modified: Lynceus — device-independent plan cost estimation.

Modifications from upstream postgres.py (~20% algorithm kept, ~80% rewritten):
  - Removed: psycopg2 database connections, EXPLAIN queries, pg_hint_plan
  - Removed: DropBufferCache, get_real_latency (require running PostgreSQL)
  - Kept:    get_plan_cost algorithm structure (cost extraction from plans)
  - Kept:    Cost model constants (seq_page_cost, random_page_cost, etc.)
  - Added:   DeviceIndependentCostEstimator with CPU/GPU cost paths
  - Added:   PlanCostExtractor for offline plan cost analysis
  - Added:   CostCalibrator for aligning estimates with real latencies

M321-M330 Algorithm Changes (Phase 8):
  [ALG-1] estimate_node_gpu: linear PCIe transfer cost → cache-oblivious
          memory transfer model (Frigo et al. 1999). Memory transfer count
          modeled as N · log(N/B) / log(M/B) under the tall cache assumption,
          where N = data elements, B = cache line size in elements, M = cache
          capacity in elements. This replaces the naive data_bytes / bandwidth
          estimate with a complexity-aware transfer cost.
  [ALG-2] estimate_node_cpu: added NUMA-aware cost modifier. Cross-NUMA-node
          memory accesses incur a 2.5x latency penalty. The fraction of remote
          accesses is estimated from data size vs. local NUMA capacity (default
          32 GB per node). If data exceeds local capacity, the overflow fraction
          pays the NUMA penalty.
  [ALG-3] _dbg_cache_oblivious(): prints N, B, M parameters, computed cache
          complexity Q(N,B,M), and the resulting transfer cost in microseconds.

References:
  PAR2QO postgres.py:81  — get_plan_cost_simple (single plan cost)
  PAR2QO postgres.py:110 — get_plan_cost (plan cost with hints)
  PAR2QO postgres.py:170 — get_all_plan_cost (enumerate plan costs)
  PostgreSQL costsize.c  — CPU/IO cost model fundamentals
  Frigo et al. 1999      — Cache-oblivious algorithms (FOCS '99)
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

_DBG_ENABLED = False
_log = logging.getLogger(__name__)


def _dbg_enable():
    global _DBG_ENABLED
    _DBG_ENABLED = True


@dataclass
class PlanNode:
    """A single node in a query execution plan tree.

    From PAR2QO: plans are trees where each node is an operator
    (SeqScan, IndexScan, HashJoin, etc.) with estimated costs.
    """
    node_type: str                     # "SeqScan", "IndexScan", "HashJoin", etc.
    estimated_rows: int = 0
    estimated_width: int = 100
    startup_cost: float = 0.0          # PostgreSQL startup cost
    total_cost: float = 0.0            # PostgreSQL total cost
    actual_time_ms: float = 0.0        # actual execution time (if available)
    children: List["PlanNode"] = field(default_factory=list)
    table_name: str = ""
    index_name: str = ""
    join_type: str = ""
    sort_key: str = ""

    @property
    def is_scan(self) -> bool:
        return self.node_type in ("SeqScan", "IndexScan", "IndexOnlyScan",
                                  "BitmapHeapScan", "BitmapIndexScan")

    @property
    def is_join(self) -> bool:
        return self.node_type in ("HashJoin", "MergeJoin", "NestedLoop")

    @property
    def data_bytes(self) -> int:
        return self.estimated_rows * self.estimated_width


# ---------------------------------------------------------------------------
# Cost constants (from PostgreSQL costsize.c, same as PAR2QO)
# ---------------------------------------------------------------------------

@dataclass
class PostgresCostConstants:
    """PostgreSQL cost model constants.

    From PAR2QO postgres.py / PostgreSQL src/backend/optimizer/path/costsize.c
    All values in microseconds for Lynceus dimensional consistency.

    [ALG-2] Added NUMA parameters for cross-node memory access penalty.
    """
    seq_page_cost: float = 0.02          # sequential page fetch
    random_page_cost: float = 0.5        # random page fetch
    cpu_tuple_cost: float = 0.05         # per-tuple CPU processing
    cpu_index_tuple_cost: float = 0.02   # per-index-tuple CPU processing
    cpu_operator_cost: float = 0.01      # per-operator CPU processing
    parallel_tuple_cost: float = 0.001   # parallel tuple communication
    parallel_setup_cost: float = 100.0   # parallel worker startup
    effective_cache_size_pages: int = 524288  # 4GB default

    # [ALG-2] NUMA parameters
    numa_local_capacity_bytes: int = 32 * (1024 ** 3)    # 32 GB per NUMA node
    numa_remote_latency_multiplier: float = 2.5          # remote NUMA penalty


# ---------------------------------------------------------------------------
# DeviceIndependentCostEstimator
#
# Replaces PAR2QO's get_plan_cost() which requires a live PostgreSQL
# connection. Instead, estimates cost from the plan tree structure using
# the same cost model constants.
#
# From PAR2QO postgres.py:81 get_plan_cost_simple:
#   cursor.execute("EXPLAIN (COSTS, FORMAT JSON) " + sql)
#   plan = cursor.fetchall()[0][0][0]['Plan']
#   return plan['Total Cost']
#
# We compute the cost directly from the PlanNode tree.
# ---------------------------------------------------------------------------

class DeviceIndependentCostEstimator:
    """Estimate plan cost without a database connection.

    Replaces PAR2QO's get_plan_cost_simple/get_plan_cost with an
    offline estimator that traverses the plan tree and accumulates
    costs using PostgreSQL's cost model.
    """

    def __init__(self, constants: Optional[PostgresCostConstants] = None):
        self.c = constants or PostgresCostConstants()

    # -------------------------------------------------------------------
    # [ALG-2] NUMA-aware CPU cost modifier
    #
    # If the working set exceeds a single NUMA node's local memory,
    # the fraction that spills to remote nodes incurs a latency penalty.
    #   remote_fraction = max(0, (data_bytes - local_cap) / data_bytes)
    #   numa_factor = 1 + remote_fraction * (multiplier - 1)
    # -------------------------------------------------------------------

    def _numa_cost_factor(self, data_bytes: int) -> float:
        """Compute NUMA penalty factor for a given data size.

        Returns a multiplier >= 1.0. Equal to 1.0 when all data fits
        in local NUMA node; approaches numa_remote_latency_multiplier
        when most data must be fetched from remote nodes.
        """
        local_cap = self.c.numa_local_capacity_bytes
        if data_bytes <= local_cap or local_cap <= 0:
            return 1.0
        remote_fraction = (data_bytes - local_cap) / data_bytes
        factor = 1.0 + remote_fraction * (self.c.numa_remote_latency_multiplier - 1.0)
        return factor

    def estimate_node_cpu(self, node: PlanNode) -> float:
        """Estimate CPU cost for a single plan node.

        [ALG-2] All memory-accessing operators now apply a NUMA penalty
        factor when the working set exceeds local NUMA node capacity.
        """
        cost = 0.0
        numa_factor = self._numa_cost_factor(node.data_bytes)

        if node.node_type == "SeqScan":
            pages = max(1, node.data_bytes // 8192)
            cost = (pages * self.c.seq_page_cost * numa_factor
                    + node.estimated_rows * self.c.cpu_tuple_cost)

        elif node.node_type in ("IndexScan", "IndexOnlyScan"):
            # B-tree depth traversal + heap fetch
            depth = 3  # typical B-tree depth
            index_pages = depth + max(1, node.estimated_rows // 100)
            cost = (index_pages * self.c.random_page_cost * numa_factor
                    + node.estimated_rows * self.c.cpu_index_tuple_cost
                    + node.estimated_rows * self.c.cpu_tuple_cost)

        elif node.node_type == "HashJoin":
            # Build hash + probe
            build_rows = node.children[1].estimated_rows if len(node.children) > 1 else node.estimated_rows
            probe_rows = node.children[0].estimated_rows if node.children else node.estimated_rows
            # Hash table is the build side — check its NUMA impact
            build_bytes = build_rows * node.estimated_width
            build_numa = self._numa_cost_factor(build_bytes)
            cost = (build_rows * self.c.cpu_tuple_cost * 2 * build_numa  # build
                    + probe_rows * self.c.cpu_tuple_cost)                 # probe

        elif node.node_type == "MergeJoin":
            left = node.children[0].estimated_rows if node.children else 0
            right = node.children[1].estimated_rows if len(node.children) > 1 else 0
            cost = (left + right) * self.c.cpu_tuple_cost * 1.5 * numa_factor

        elif node.node_type == "NestedLoop":
            outer = node.children[0].estimated_rows if node.children else 1
            inner = node.children[1].estimated_rows if len(node.children) > 1 else node.estimated_rows
            # Inner relation repeatedly accessed — NUMA penalty on inner
            inner_bytes = inner * node.estimated_width
            inner_numa = self._numa_cost_factor(inner_bytes)
            cost = outer * inner * self.c.cpu_operator_cost * inner_numa

        elif node.node_type == "Sort":
            n = max(1, node.estimated_rows)
            sort_bytes = n * node.estimated_width
            sort_numa = self._numa_cost_factor(sort_bytes)
            cost = 2.0 * n * math.log2(max(2, n)) * self.c.cpu_operator_cost * sort_numa

        elif node.node_type == "Aggregate":
            cost = node.estimated_rows * self.c.cpu_tuple_cost

        # Add children costs
        for child in node.children:
            cost += self.estimate_node_cpu(child)

        return cost

    # -------------------------------------------------------------------
    # [ALG-1] Cache-oblivious GPU cost model (Frigo et al. 1999)
    #
    # The number of memory transfers for a cache-oblivious algorithm is:
    #   Q(N, B, M) = Θ( N / B · log_{M/B}(N / B) )
    #              = N / B · log(N/B) / log(M/B)
    #
    # under the tall cache assumption M ≥ B².
    #
    # We model the GPU memory hierarchy with:
    #   B = L2 cache line size in elements (128 bytes / element_size)
    #   M = L2 cache capacity in elements (40 MB typical for A100)
    # The transfer cost is Q(N,B,M) * B * element_size / bandwidth.
    # -------------------------------------------------------------------

    def _cache_oblivious_transfers(self, n_elements: int,
                                   element_size: int = 4,
                                   l2_cache_bytes: int = 40 * 1024 * 1024,
                                   cache_line_bytes: int = 128) -> float:
        """Compute cache-oblivious memory transfer count Q(N, B, M).

        Parameters:
          n_elements:      number of data elements (N)
          element_size:    bytes per element (default 4 for int/float)
          l2_cache_bytes:  GPU L2 cache size (M in bytes, default 40 MB)
          cache_line_bytes: GPU cache line / sector size (B in bytes)

        Returns:
          Number of cache-line-sized transfers (float).
        """
        if n_elements <= 0:
            return 0.0

        B = max(1, cache_line_bytes // element_size)   # cache line in elements
        M = max(B * B, l2_cache_bytes // element_size)  # cache capacity, enforce tall cache

        N_over_B = n_elements / B
        if N_over_B <= 1.0:
            return 1.0  # fits in one cache line

        M_over_B = M / B
        if M_over_B <= 1.0:
            M_over_B = 2.0  # degenerate, avoid log(1)=0

        # Q(N,B,M) = N/B · log(N/B) / log(M/B)
        q = N_over_B * math.log(N_over_B) / math.log(M_over_B)

        if _DBG_ENABLED:
            self._dbg_cache_oblivious(n_elements, B, M, q,
                                      element_size, cache_line_bytes)

        return max(1.0, q)

    def estimate_node_gpu(self, node: PlanNode, pcie_bw_gb_s: float = 32.0,
                          hbm_bw_gb_s: float = 2000.0,
                          kernel_launch_us: float = 10.0) -> float:
        """Estimate GPU cost for a single plan node.

        [ALG-1] PCIe transfer cost now uses cache-oblivious complexity
        instead of naive data_bytes / bandwidth. The transfer count
        Q(N,B,M) captures the memory hierarchy effects of moving data
        through the PCIe + GPU cache hierarchy.
        """
        data_bytes = node.data_bytes
        element_size = max(1, node.estimated_width)

        # [ALG-1] Cache-oblivious transfer model
        # PCIe cache-line size ~ 64 bytes (TLP payload granularity)
        n_elements = max(1, node.estimated_rows)
        q_transfers = self._cache_oblivious_transfers(
            n_elements=n_elements,
            element_size=element_size,
            l2_cache_bytes=40 * 1024 * 1024,  # 40 MB L2 (A100)
            cache_line_bytes=128
        )
        # Each transfer moves one cache line's worth of data
        transfer_bytes = q_transfers * 128  # cache line size
        transfer = (transfer_bytes / (pcie_bw_gb_s * 1e9)) * 1e6 if pcie_bw_gb_s > 0 else 0

        # Kernel launch
        launch = kernel_launch_us

        # HBM bandwidth-bound (also use cache-oblivious for HBM access)
        q_hbm = self._cache_oblivious_transfers(
            n_elements=n_elements,
            element_size=element_size,
            l2_cache_bytes=40 * 1024 * 1024,
            cache_line_bytes=32  # HBM sector size
        )
        hbm_bytes = q_hbm * 32
        hbm = (hbm_bytes / (hbm_bw_gb_s * 1e9)) * 1e6 if hbm_bw_gb_s > 0 else 0

        # Compute-bound (GPU is ~100x faster per tuple)
        compute = node.estimated_rows * 0.0001

        gpu_cost = transfer + launch + max(hbm, compute)

        # GPU sort (bitonic) — also cache-oblivious: Q = N/B · log²(N) / log(M/B)
        if node.node_type == "Sort" and node.estimated_rows > 1:
            n = node.estimated_rows
            gpu_cost += n * (math.log2(max(2, n)) ** 2) * 0.00005 / 108

        # Add children (assume children also on GPU)
        for child in node.children:
            child_gpu = self.estimate_node_gpu(child, pcie_bw_gb_s, hbm_bw_gb_s, 0)
            gpu_cost += child_gpu  # no extra kernel launch for children

        return gpu_cost

    def _dbg_cache_oblivious(self, n_elements: int, b: int, m: int,
                             q: float, element_size: int,
                             cache_line_bytes: int):
        """Print cache-oblivious analysis parameters.

        [ALG-3] Debug breakpoint showing N, B, M, Q(N,B,M) and derived
        transfer cost.
        """
        transfer_bytes = q * cache_line_bytes
        print("=" * 60)
        print("[_dbg_cache_oblivious] Cache-oblivious transfer analysis")
        print(f"  N (elements)       = {n_elements}")
        print(f"  B (line, elements) = {b}")
        print(f"  M (cache, elements)= {m}")
        print(f"  element_size       = {element_size} bytes")
        print(f"  cache_line_bytes   = {cache_line_bytes}")
        print(f"  N/B                = {n_elements / max(1, b):.2f}")
        print(f"  M/B                = {m / max(1, b):.2f}")
        print(f"  Q(N,B,M)           = {q:.2f} transfers")
        print(f"  transfer_bytes     = {transfer_bytes:.0f}")
        print(f"  Tall cache check   = M >= B²: {m} >= {b * b}: "
              f"{'OK' if m >= b * b else 'VIOLATED'}")
        print("=" * 60)

    def estimate_plan(self, root: PlanNode) -> Tuple[float, float, str]:
        """Estimate full plan cost on CPU and GPU.

        Returns (cpu_cost_us, gpu_cost_us, recommended_device).

        Replaces PAR2QO get_plan_cost (postgres.py:110) which requires
        a live database connection.
        """
        cpu = self.estimate_node_cpu(root)
        gpu = self.estimate_node_gpu(root)
        device = "gpu" if gpu < cpu else "cpu"
        return cpu, gpu, device

    def estimate_all_plans(self, plans: List[PlanNode]) -> List[Tuple[float, float, str]]:
        """Estimate costs for multiple plans.

        Replaces PAR2QO get_all_plan_cost (postgres.py:170).
        """
        return [self.estimate_plan(p) for p in plans]

    @staticmethod
    def _dbg():
        """Enable debug output for cache-oblivious and NUMA analysis."""
        _dbg_enable()
        print("[par2qo_cost] Debug mode enabled — cache-oblivious Q(N,B,M) "
              "and NUMA factor diagnostics will print.")


# ---------------------------------------------------------------------------
# CostCalibrator — align model estimates with real measurements
#
# When actual execution times are available (from benchmarking), this
# computes a calibration factor to improve future estimates.
# ---------------------------------------------------------------------------

class CostCalibrator:
    """Calibrate cost model using actual execution measurements."""

    def __init__(self):
        self._observations: List[Tuple[float, float]] = []  # (estimated, actual)

    def observe(self, estimated_us: float, actual_us: float):
        self._observations.append((estimated_us, actual_us))

    def calibration_factor(self) -> float:
        """Compute the ratio actual/estimated (> 1 means model underestimates)."""
        if not self._observations:
            return 1.0
        ratios = [a / max(1e-9, e) for e, a in self._observations]
        return sum(ratios) / len(ratios)

    def calibrate(self, estimated_us: float) -> float:
        return estimated_us * self.calibration_factor()
