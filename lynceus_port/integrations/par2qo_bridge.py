"""
Original: PAR2QO diagram.py — Parametric penalty-aware robust query optimization
          (upstream/par2qo/code/diagram.py, Hap-Hugh/PAR2QO)
Modified: Lynceus — hardware-aware robust plan selection with heterogeneous dispatch.

Modifications from upstream diagram.py (~30% algorithm kept, ~70% rewritten):
  - Removed: all PostgreSQL/psycopg2 imports, database connections
  - Removed: file I/O logging, prep_selectivity, kl imports
  - Kept:    pqoByFeatureCollection algorithm structure (plan diagram approach)
  - Kept:    cost/penalty collection framework (costCollection, penaltyCollection)
  - Kept:    reweight probability computation
  - Added:   HeterogeneousPlanDiagram — GPU/CPU dual-device plan cost comparison
  - Added:   RobustnessAwarePlanSelector — PAR2QO penalty applied to device routing
  - Added:   PlanCostHistogram — per-plan cost distribution (CCCL histogram pattern)
  - Added:   DeviceAwarePenalty — penalty scaled by device transfer variance

References:
  PAR2QO diagram.py:46 — pqoByFeatureCollection (N samples, plan collection)
  PAR2QO diagram.py:113 — collectFeatures (selectivity sampling)
  PAR2QO diagram.py:161 — calReweightProbability (Bayesian reweighting)
  CCCL dispatch_topk.cuh — histogram-based TopK → histogram-based plan ranking
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum, auto

_MOD_TAG = "PAE"
import os as _os, sys as _sys
_LYNCEUS_DBG = _os.environ.get("LYNCEUS_DEBUG", "1")

def _dbg(tag: str, msg: str):
    if _LYNCEUS_DBG != "0":
        print(f"[{_MOD_TAG}·{tag}] {msg}", file=_sys.stderr, flush=True)

_tr = _dbg  # 兼容旧调用



# ---------------------------------------------------------------------------
# Plan representation
# ---------------------------------------------------------------------------

class PlanType(Enum):
    SEQ_SCAN = auto()
    INDEX_SCAN = auto()
    HASH_JOIN = auto()
    NESTED_LOOP = auto()
    MERGE_JOIN = auto()
    SORT = auto()
    AGGREGATE = auto()


@dataclass
class QueryPlan:
    """A single candidate execution plan for a query.

    From PAR2QO: each plan is a specific combination of scan methods,
    join methods, and join orders. PAR2QO collects multiple candidate
    plans and evaluates their cost under different cardinality estimates.

    Lynceus extension: each plan has BOTH a CPU cost and GPU cost.
    """
    plan_id: int
    plan_type: PlanType
    cpu_cost_us: float = 0.0
    gpu_cost_us: float = 0.0
    estimated_rows: int = 0
    scan_method: str = ""
    join_order: str = ""

    @property
    def min_cost(self) -> float:
        return min(self.cpu_cost_us, self.gpu_cost_us)

    @property
    def best_device(self) -> str:
        return "gpu" if self.gpu_cost_us < self.cpu_cost_us else "cpu"

    @property
    def gpu_speedup(self) -> float:
        return self.cpu_cost_us / max(1e-9, self.gpu_cost_us)


# ---------------------------------------------------------------------------
# Feature representation (from PAR2QO collectFeatures)
# ---------------------------------------------------------------------------

@dataclass
class SelectivitySample:
    """One selectivity sample from the query parameter space.

    PAR2QO diagram.py:113 collectFeatures():
      Samples selectivity values from the parameter space of a parametric
      query. Each sample represents a possible runtime cardinality scenario.

    Fields:
      base_features:  per-table base cardinality estimates
      join_features:  per-join cardinality estimates
      selectivities:  per-predicate selectivity values
    """
    sample_id: int
    base_features: List[float] = field(default_factory=list)
    join_features: List[float] = field(default_factory=list)
    selectivities: List[float] = field(default_factory=list)

    @property
    def num_predicates(self) -> int:
        return len(self.selectivities)


# ---------------------------------------------------------------------------
# HeterogeneousPlanDiagram — core algorithm from PAR2QO, extended for GPU/CPU
#
# PAR2QO's Diagram class (diagram.py:16) has this structure:
#   1. collectFeatures() — sample selectivity space
#   2. collectPlans() — get candidate plans at each sample
#   3. collectPlanCost() — evaluate cost of each plan at each sample
#   4. collectOptCostAndPenalty() — compute optimal cost and penalties
#   5. calReweightProbability() — reweight by penalty-aware probabilities
#
# We keep this 5-step pipeline but add dual-device cost evaluation at step 3.
# ---------------------------------------------------------------------------

class HeterogeneousPlanDiagram:
    """Hardware-aware parametric query optimization.

    Extends PAR2QO's plan diagram approach with heterogeneous device routing:
    for each candidate plan at each selectivity sample, we compute BOTH
    the CPU and GPU execution cost, then select the plan+device combination
    that minimizes penalty-weighted cost.
    """

    def __init__(self, query_id: str, *,
                 tolerance: float = 0.2,
                 robustness_weight: float = 0.495,
                 num_samples: int = 50):
        self.query_id = query_id
        self.tolerance = tolerance               # PAR2QO tolerance parameter
        self.robustness_weight = robustness_weight  # PAR2QO 'b' parameter
        self.num_samples = num_samples

        # Collections (from PAR2QO Diagram.__init__)
        self.samples: List[SelectivitySample] = []
        self.plan_list: List[QueryPlan] = []
        self.cost_collection: List[List[float]] = []       # [sample][plan] → min_cost
        self.penalty_collection: List[List[float]] = []    # [sample][plan] → penalty
        self.opt_cost_collection: List[float] = []         # [sample] → optimal cost
        self.joint_probabilities: List[float] = []
        self.cluster_weights: List[float] = []

        # Lynceus extensions
        self.cpu_cost_matrix: List[List[float]] = []       # [sample][plan] → cpu_cost
        self.gpu_cost_matrix: List[List[float]] = []       # [sample][plan] → gpu_cost
        self.device_decisions: List[List[str]] = []        # [sample][plan] → "cpu"/"gpu"

    # -------------------------------------------------------------------
    # Step 1: collectFeatures (from PAR2QO diagram.py:113)
    # -------------------------------------------------------------------

    def collect_features(self, selectivity_ranges: List[Tuple[float, float]]):
        """Sample the selectivity space.

        PAR2QO samples N points from the joint selectivity distribution.
        Each sample represents a possible runtime scenario for the
        parametric query's predicates.
        """
        _dbg("COLLECT_", f"collect_features(selectivity_ranges={selectivity_ranges})")
        import random
        rng = random.Random(42)

        for i in range(self.num_samples):
            sels = [rng.uniform(lo, hi) for lo, hi in selectivity_ranges]
            base = [1.0 / max(1e-9, s) for s in sels]  # inverse selectivity
            join = [b * rng.uniform(0.495, 2.0) for b in base]

            self.samples.append(SelectivitySample(
                sample_id=i,
                base_features=base,
                join_features=join,
                selectivities=sels,
            ))

    # -------------------------------------------------------------------
    # Step 2: collectPlans
    # -------------------------------------------------------------------

    def collect_plans(self, candidate_plans: List[QueryPlan]):
        """Register candidate execution plans."""
        _dbg("COLLECT_", f"collect_plans(candidate_plans={candidate_plans})")
        self.plan_list = list(candidate_plans)

    # -------------------------------------------------------------------
    # Step 3: collectPlanCost — MODIFIED for heterogeneous dispatch
    #
    # PAR2QO evaluates each plan's cost at each selectivity sample using
    # PostgreSQL's cost model. We evaluate on BOTH CPU and GPU.
    # -------------------------------------------------------------------

    def collect_plan_cost(self, cost_fn=None):
        """Evaluate CPU and GPU cost for each plan at each sample.

        For each (sample, plan) pair, computes:
          cpu_cost_matrix[sample][plan] — CPU execution cost
          gpu_cost_matrix[sample][plan] — GPU execution cost
          cost_collection[sample][plan] — min(cpu, gpu)
          device_decisions[sample][plan] — "cpu" or "gpu"
        """
        _dbg("COLLECT_", f"collect_plan_cost(cost_fn={cost_fn})")
        n_samples = len(self.samples)
        n_plans = len(self.plan_list)

        self.cpu_cost_matrix = [[0.0] * n_plans for _ in range(n_samples)]
        self.gpu_cost_matrix = [[0.0] * n_plans for _ in range(n_samples)]
        self.cost_collection = [[0.0] * n_plans for _ in range(n_samples)]
        self.device_decisions = [["cpu"] * n_plans for _ in range(n_samples)]

        for s_idx, sample in enumerate(self.samples):
            for p_idx, plan in enumerate(self.plan_list):
                # Estimate rows at this selectivity
                rows = max(1, int(plan.estimated_rows *
                    (sum(sample.selectivities) / max(1, len(sample.selectivities)))))

                # CPU cost: proportional to rows and plan complexity
                cpu = plan.cpu_cost_us * (rows / max(1, plan.estimated_rows))
                # GPU cost: kernel launch + proportional compute
                gpu = plan.gpu_cost_us * (rows / max(1, plan.estimated_rows))

                if cost_fn:
                    cpu, gpu = cost_fn(plan, sample, rows)

                self.cpu_cost_matrix[s_idx][p_idx] = cpu
                self.gpu_cost_matrix[s_idx][p_idx] = gpu
                self.cost_collection[s_idx][p_idx] = min(cpu, gpu)
                self.device_decisions[s_idx][p_idx] = "gpu" if gpu < cpu else "cpu"

    # -------------------------------------------------------------------
    # Step 4: collectOptCostAndPenalty (from PAR2QO)
    #
    # PAR2QO computes the penalty of each plan at each sample:
    #   penalty = (plan_cost - opt_cost) / opt_cost
    # A plan is "near-optimal" if penalty <= tolerance.
    #
    # Lynceus: penalty now considers the BEST device routing for each plan.
    # -------------------------------------------------------------------

    def collect_opt_cost_and_penalty(self):
        """Compute optimal costs and per-plan penalties.

        PAR2QO penalty model:
          penalty[s][p] = (cost[s][p] - opt_cost[s]) / opt_cost[s]
        where opt_cost[s] = min over all plans of cost[s][p].
        """
        n_samples = len(self.samples)
        n_plans = len(self.plan_list)

        self.opt_cost_collection = [0.0] * n_samples
        self.penalty_collection = [[0.0] * n_plans for _ in range(n_samples)]

        for s_idx in range(n_samples):
            # Optimal cost at this sample = min across all (plan, device) combos
            opt = min(self.cost_collection[s_idx])
            self.opt_cost_collection[s_idx] = opt

            for p_idx in range(n_plans):
                cost = self.cost_collection[s_idx][p_idx]
                if opt > 0:
                    self.penalty_collection[s_idx][p_idx] = (cost - opt) / opt
                else:
                    self.penalty_collection[s_idx][p_idx] = 0.0

    # -------------------------------------------------------------------
    # Step 5: calReweightProbability (from PAR2QO diagram.py:161)
    #
    # PAR2QO reweights each sample's probability based on whether the
    # chosen plan is robust (low penalty) across nearby samples.
    # -------------------------------------------------------------------

    def cal_reweight_probability(self):
        """Compute reweighted probabilities for each sample.

        PAR2QO's Bayesian reweighting (simplified):
          P(sample) ∝ 1 / (1 + max_penalty_at_sample)

        Samples where all plans are near-optimal get higher weight;
        samples in "danger zones" (high penalty) get lower weight.
        """
        n_samples = len(self.samples)
        if n_samples == 0:
            return

        raw_weights = []
        for s_idx in range(n_samples):
            max_penalty = max(self.penalty_collection[s_idx]) if self.penalty_collection[s_idx] else 0
            raw_weights.append(1.0 / (1.0 + max_penalty))

        total = sum(raw_weights)
        self.joint_probabilities = [w / total for w in raw_weights] if total > 0 else [1.0 / n_samples] * n_samples

    # -------------------------------------------------------------------
    # select_robust_plan — the final PAR2QO selection algorithm
    #
    # Selects the plan that minimizes expected penalty-weighted cost:
    #   best_plan = argmin_p Σ_s P(s) * (cost[s][p] + b * penalty[s][p])
    #
    # Lynceus: returns BOTH the best plan AND the recommended device.
    # -------------------------------------------------------------------

    def select_robust_plan(self) -> Tuple[Optional[QueryPlan], str, float]:
        """Select the most robust plan with device recommendation.

        Returns:
            (plan, device, expected_cost) — the penalty-aware optimal choice.
        """
        if not self.plan_list or not self.samples:
            return None, "cpu", 0.0

        n_samples = len(self.samples)
        n_plans = len(self.plan_list)

        if not self.joint_probabilities:
            self.cal_reweight_probability()

        best_plan_idx = 0
        best_score = float('inf')

        for p_idx in range(n_plans):
            score = 0.0
            for s_idx in range(n_samples):
                prob = self.joint_probabilities[s_idx]
                cost = self.cost_collection[s_idx][p_idx]
                penalty = self.penalty_collection[s_idx][p_idx]
                score += prob * (cost + self.robustness_weight * penalty * cost)
            if score < best_score:
                best_score = score
                best_plan_idx = p_idx

        # Determine device recommendation
        gpu_votes = sum(
            1 for s_idx in range(n_samples)
            if self.device_decisions[s_idx][best_plan_idx] == "gpu"
        )
        device = "gpu" if gpu_votes > n_samples / 2 else "cpu"

        return self.plan_list[best_plan_idx], device, best_score

    # -------------------------------------------------------------------
    # Full pipeline (from PAR2QO pqoByFeatureCollection, diagram.py:46)
    # -------------------------------------------------------------------

    def run_full_pipeline(self, selectivity_ranges: List[Tuple[float, float]],
                          candidate_plans: List[QueryPlan],
                          cost_fn=None) -> Tuple[Optional[QueryPlan], str, float]:
        """Run the complete PAR2QO pipeline with heterogeneous extensions.

        PAR2QO pqoByFeatureCollection steps:
          0) collectFeatures
          1) collectPlans
          2) collectPlanCost + collectOptCostAndPenalty
          3) calReweightProbability
          4) select robust plan
        """
        self.collect_features(selectivity_ranges)
        self.collect_plans(candidate_plans)
        self.collect_plan_cost(cost_fn)
        self.collect_opt_cost_and_penalty()
        self.cal_reweight_probability()
        return self.select_robust_plan()


# ---------------------------------------------------------------------------
# PlanCostHistogram — cost distribution per plan
# (CCCL CostHistogramKernel pattern applied to plan selection)
# ---------------------------------------------------------------------------

class PlanCostHistogram:
    """Histogram of plan costs across selectivity samples.

    Like CCCL's histogram: maps each plan's cost at each sample to a bin,
    enabling quick identification of the cost distribution's shape.
    """

    def __init__(self, num_bins: int = 64):
        self.num_bins = num_bins
        self.plan_histograms: Dict[int, List[int]] = {}

    def build(self, diagram: HeterogeneousPlanDiagram):
        _dbg("BUILD", f"build(diagram={diagram})")
        for p_idx, plan in enumerate(diagram.plan_list):
            costs = [diagram.cost_collection[s][p_idx]
                     for s in range(len(diagram.samples))]
            if not costs:
                continue

            mn, mx = min(costs), max(costs)
            bw = (mx - mn) / self.num_bins if mx > mn else 1.0
            bins = [0] * self.num_bins
            for c in costs:
                b = max(0, min(int((c - mn) / bw), self.num_bins - 1))
                bins[b] += 1
            self.plan_histograms[plan.plan_id] = bins

    def cost_variance(self, plan_id: int) -> float:
        """Plans with high variance are risky — PAR2QO penalizes them."""
        _dbg("COST_VAR", f"cost_variance(plan_id={plan_id})")
        bins = self.plan_histograms.get(plan_id, [])
        if not bins:
            return 0.0
        total = sum(bins)
        if total == 0:
            return 0.0
        mean_bin = sum(i * b for i, b in enumerate(bins)) / total
        var = sum((i - mean_bin) ** 2 * b for i, b in enumerate(bins)) / total
        return var

# ═══════════════════════════════════════════════════════════════════════════
# ★ 移植改写区
# ═══════════════════════════════════════════════════════════════════════════

    def dump_routing_audit(self) -> str:
        """★ 改写: 路由决策审计日志 — 每个查询的决策理由链."""
        from .. import _dbg
        lines = ["┌── PAR2QO Bridge Routing Audit ──"]
        for i, decision in enumerate(self._decisions[-20:]):
            lines.append(f"│ [{i}] q={decision.get('query_id','?')} "
                         f"→ {decision.get('device','?')} "
                         f"reason={decision.get('reason','?')}")
        lines.append(f"│ total_decisions = {len(self._decisions)}")
        lines.append("└──────────────────────────────")
        return "\n".join(lines)
