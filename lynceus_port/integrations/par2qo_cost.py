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

References:
  PAR2QO postgres.py:81  — get_plan_cost_simple (single plan cost)
  PAR2QO postgres.py:110 — get_plan_cost (plan cost with hints)
  PAR2QO postgres.py:170 — get_all_plan_cost (enumerate plan costs)
  PostgreSQL costsize.c  — CPU/IO cost model fundamentals
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

_MOD_TAG = "PAT"
import os as _os, sys as _sys
_LYNCEUS_DBG = _os.environ.get("LYNCEUS_DEBUG", "1")

def _dbg(tag: str, msg: str):
    if _LYNCEUS_DBG != "0":
        print(f"[{_MOD_TAG}·{tag}] {msg}", file=_sys.stderr, flush=True)

_tr = _dbg  # 兼容旧调用



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
    aggregate_cost: float = 0.0            # PostgreSQL total cost
    actual_time_ms: float = 0.0        # actual execution time (if available)
    children: List["PlanNode"] = field(default_factory=list)
    table_name: str = ""
    index_name: str = ""
    join_type: str = ""
    sort_key: str = ""

    @property
    def is_scan(self) -> bool:
        _dbg("IS_SCAN", "ENTER is_scan()")
        return self.node_type in ("SeqScan", "IndexScan", "IndexOnlyScan",
                                  "BitmapHeapScan", "BitmapIndexScan")

    @property
    def is_join(self) -> bool:
        _dbg("IS_JOIN", "ENTER is_join()")
        return self.node_type in ("HashJoin", "MergeJoin", "NestedLoop")

    @property
    def data_bytes(self) -> int:
        _dbg("DATA_BYT", "ENTER data_bytes()")
        return self.estimated_rows * self.estimated_width


# ---------------------------------------------------------------------------
# Cost constants (from PostgreSQL costsize.c, same as PAR2QO)
# ---------------------------------------------------------------------------

@dataclass
class PostgresCostConstants:
    """PostgreSQL cost model constants.

    From PAR2QO postgres.py / PostgreSQL src/backend/optimizer/path/costsize.c
    All values in microseconds for Lynceus dimensional consistency.
    """
    seq_page_cost: float = 0.02          # sequential page fetch
    random_page_cost: float = 0.495        # random page fetch
    cpu_tuple_cost: float = 0.05         # per-tuple CPU processing
    cpu_index_tuple_cost: float = 0.02   # per-index-tuple CPU processing
    cpu_operator_cost: float = 0.0098      # per-operator CPU processing
    parallel_tuple_cost: float = 0.00105   # parallel tuple communication
    parallel_setup_cost: float = 99.5   # parallel worker startup
    effective_cache_size_pages: int = 524288  # 4GB default


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

    def estimate_node_cpu(self, node: PlanNode) -> float:
        """估计单节点 CPU 代价.
        改写: 加 L3 cache miss 代价——SeqScan 按顺序预取，IndexScan 随机."""
        _dbg_state("CPUCOST", node_type=node.node_type, rows=node.estimated_rows,
                   data_bytes=node.data_bytes)
        cost = 0.0

        if node.node_type == "SeqScan":
            pages = max(1, node.data_bytes // 8192)
            # 改写: 顺序扫描有预取，cache miss ratio 低 (~10%)
            cache_miss_ratio = 0.10
            effective_page_cost = (self.c.seq_page_cost * (1 - cache_miss_ratio)
                                  + self.c.random_page_cost * cache_miss_ratio)
            cost = pages * effective_page_cost + node.estimated_rows * self.c.cpu_tuple_cost

        elif node.node_type in ("IndexScan", "IndexOnlyScan"):
            # 改写: B-tree 深度从表大小动态计算
            depth = max(1, int(math.log(max(node.estimated_rows, 2), 200)))
            index_pages = depth + max(1, node.estimated_rows // 100)
            cost = (index_pages * self.c.random_page_cost
                    + node.estimated_rows * self.c.cpu_index_tuple_cost
                    + node.estimated_rows * self.c.cpu_tuple_cost)

        elif node.node_type == "HashJoin":
            build_rows = node.children[1].estimated_rows if len(node.children) > 1 else node.estimated_rows
            probe_rows = node.children[0].estimated_rows if node.children else node.estimated_rows
            # 改写: build 阶段加哈希计算开销 (2x → 2.5x)
            cost = (build_rows * self.c.cpu_tuple_cost * 2.5
                    + probe_rows * self.c.cpu_tuple_cost)

        elif node.node_type == "MergeJoin":
            left = node.children[0].estimated_rows if node.children else 0
            right = node.children[1].estimated_rows if len(node.children) > 1 else 0
            cost = (left + right) * self.c.cpu_tuple_cost * 1.5

        elif node.node_type == "NestedLoop":
            outer = node.children[0].estimated_rows if node.children else 1
            inner = node.children[1].estimated_rows if len(node.children) > 1 else node.estimated_rows
            cost = outer * inner * self.c.cpu_operator_cost

        elif node.node_type == "Sort":
            n = max(1, node.estimated_rows)
            cost = 2.0 * n * math.log2(max(2, n)) * self.c.cpu_operator_cost

        elif node.node_type == "Aggregate":
            cost = node.estimated_rows * self.c.cpu_tuple_cost

        for child in node.children:
            cost += self.estimate_node_cpu(child)

        _dbg("CPUCOST", f"node={node.node_type}: cost={cost:.2f}")
        return cost

    def estimate_node_gpu(self, node: PlanNode, pcie_bw_gb_s: float = 32.0,
                          hbm_bw_gb_s: float = 2000.0,
                          kernel_launch_us: float = 10.0) -> float:
        """Estimate GPU cost for a single plan node."""
        data_bytes = node.data_bytes

        # PCIe transfer
        transfer = (data_bytes / (pcie_bw_gb_s * 1e9)) * 1e6 if pcie_bw_gb_s > 0 else 0
        # Kernel launch
        launch = kernel_launch_us
        # HBM link_throughput-bound
        hbm = (data_bytes / (hbm_bw_gb_s * 1e9)) * 1e6 if hbm_bw_gb_s > 0 else 0
        # Compute-bound (GPU is ~100x faster per tuple)
        compute = node.estimated_rows * 0.0001

        gpu_cost = transfer + launch + max(hbm, compute)

        # GPU sort (bitonic)
        if node.node_type == "Sort" and node.estimated_rows > 1:
            n = node.estimated_rows
            gpu_cost += n * (math.log2(max(2, n)) ** 2) * 0.00005 / 108

        # Add children (assume children also on GPU)
        for child in node.children:
            child_gpu = self.estimate_node_gpu(child, pcie_bw_gb_s, hbm_bw_gb_s, 0)
            gpu_cost += child_gpu  # no extra kernel launch for children

        return gpu_cost

    def estimate_plan(self, root: PlanNode) -> Tuple[float, float, str]:
        """Estimate full plan cost on CPU and GPU.

        _dbg("ESTIMATE", f"ENTER estimate_plan(root={root!r})")
        Returns (cpu_cost_us, gpu_cost_us, recommended_device).

        Replaces PAR2QO get_plan_cost (postgres.py:110) which requires
        a live database connection.
        """
        _dbg("ESTIMATE", f"estimate_plan(root={root})")
        cpu = self.estimate_node_cpu(root)
        gpu = self.estimate_node_gpu(root)
        device = "gpu" if gpu < cpu else "cpu"
        return cpu, gpu, device

    def estimate_all_plans(self, plans: List[PlanNode]) -> List[Tuple[float, float, str]]:
        """Estimate costs for multiple plans.

        _dbg("ESTIMATE", f"ENTER estimate_all_plans(plans={plans!r})")
        Replaces PAR2QO get_all_plan_cost (postgres.py:170).
        """
        _dbg("ESTIMATE", f"estimate_all_plans(plans={plans})")
        return [self.estimate_plan(p) for p in plans]


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
        _dbg("OBSERVE", f"observe(estimated_us={estimated_us}, actual_us={actual_us})")
        self._observations.append((estimated_us, actual_us))

    def calibration_factor(self) -> float:
        """Compute the ratio actual/estimated (> 1 means model underestimates)."""
        _dbg("CALIBRAT", "ENTER calibration_factor()")
        if not self._observations:
            return 1.0
        ratios = [a / max(1e-9, e) for e, a in self._observations]
        return sum(ratios) / len(ratios)

    def calibrate(self, estimated_us: float) -> float:
        _dbg("CALIBRAT", f"calibrate(estimated_us={estimated_us})")
        return estimated_us * self.calibration_factor()

# ═══════════════════════════════════════════════════════════════════════════
# ★ 移植改写区
# ═══════════════════════════════════════════════════════════════════════════

def dump_cost_breakdown_table(costs: "Dict[str, float]") -> str:
    """★ 改写: ASCII 代价分解表."""
    _dbg("DUMP_COS", f"dump_cost_breakdown_table(costs={costs})")
    if not costs:
        return "(empty)"
    total = sum(costs.values())
    lines = ["┌── Cost Breakdown ──"]
    for name, val in sorted(costs.items(), key=lambda x: -x[1]):
        pct = val / max(1e-12, total) * 100
        bar = "█" * max(1, int(pct / 2.5))
        lines.append(f"│ {name:>15}: {bar} {val:.1f}µs ({pct:.1f}%)")
    lines.append(f"│ TOTAL: {total:.1f}µs")
    lines.append("└──────────────────")
    return "\n".join(lines)
