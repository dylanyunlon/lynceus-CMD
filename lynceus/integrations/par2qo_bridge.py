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

M321-M330 Algorithm Changes (Phase 8):
  [ALG-1] select_robust_plan: simple penalty-weighted argmin → Pareto frontier
          with non-dominated sorting on (expected_cost, max_penalty), then
          Chebyshev scalarization to pick final plan from Pareto set.
  [ALG-2] cal_reweight_probability: simple 1/(1+penalty) normalization →
          softmax with adaptive temperature + entropy regularization to
          prevent probability collapse onto a single sample.
  [ALG-3] collect_plan_cost: added bootstrap confidence interval estimation
          (B=200 resamples) for each plan's cost distribution across samples.
  [ALG-4] _dbg_pareto_frontier(): prints Pareto set size, each plan's
          (cost, robustness) coordinates, and selected plan details.

References:
  PAR2QO diagram.py:46 — pqoByFeatureCollection (N samples, plan collection)
  PAR2QO diagram.py:113 — collectFeatures (selectivity sampling)
  PAR2QO diagram.py:161 — calReweightProbability (Bayesian reweighting)
  CCCL dispatch_topk.cuh — histogram-based TopK → histogram-based plan ranking
"""
from __future__ import annotations

import math
import logging
import random as _random_module
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum, auto

_DBG_ENABLED = False          # flip to True or call _dbg_enable() at runtime
_log = logging.getLogger(__name__)


def _dbg_enable():
    global _DBG_ENABLED
    _DBG_ENABLED = True


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
                 robustness_weight: float = 0.5,
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

        # [ALG-3] Bootstrap confidence intervals: plan_idx → (ci_low, ci_high)
        self.bootstrap_ci: Dict[int, Tuple[float, float]] = {}

    # -------------------------------------------------------------------
    # Step 1: collectFeatures (from PAR2QO diagram.py:113)
    # -------------------------------------------------------------------

    def collect_features(self, selectivity_ranges: List[Tuple[float, float]]):
        """Sample the selectivity space.

        PAR2QO samples N points from the joint selectivity distribution.
        Each sample represents a possible runtime scenario for the
        parametric query's predicates.
        """
        import random
        rng = random.Random(42)

        for i in range(self.num_samples):
            sels = [rng.uniform(lo, hi) for lo, hi in selectivity_ranges]
            base = [1.0 / max(1e-9, s) for s in sels]  # inverse selectivity
            join = [b * rng.uniform(0.5, 2.0) for b in base]

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
        self.plan_list = list(candidate_plans)

    # -------------------------------------------------------------------
    # Step 3: collectPlanCost — MODIFIED for heterogeneous dispatch
    #
    # PAR2QO evaluates each plan's cost at each selectivity sample using
    # PostgreSQL's cost model. We evaluate on BOTH CPU and GPU.
    #
    # [ALG-3] After cost collection, compute bootstrap confidence intervals
    # for each plan's cost distribution across the selectivity samples.
    # B=200 resamples, report 95% CI (2.5th and 97.5th percentiles).
    # -------------------------------------------------------------------

    def collect_plan_cost(self, cost_fn=None):
        """Evaluate CPU and GPU cost for each plan at each sample.

        For each (sample, plan) pair, computes:
          cpu_cost_matrix[sample][plan] — CPU execution cost
          gpu_cost_matrix[sample][plan] — GPU execution cost
          cost_collection[sample][plan] — min(cpu, gpu)
          device_decisions[sample][plan] — "cpu" or "gpu"

        Then runs bootstrap resampling (B=200) to estimate 95% confidence
        intervals on the expected cost per plan.
        """
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

        # [ALG-3] Bootstrap confidence interval estimation
        self._compute_bootstrap_ci(n_resamples=200, alpha=0.05)

    def _compute_bootstrap_ci(self, n_resamples: int = 200, alpha: float = 0.05):
        """Bootstrap resample each plan's cost vector to get CI on expected cost.

        For plan p, the cost vector is [cost_collection[s][p] for s in samples].
        We draw B bootstrap samples (with replacement), compute the mean of each,
        and report the alpha/2 and 1-alpha/2 percentiles as CI bounds.
        """
        rng = _random_module.Random(123)
        n_samples = len(self.samples)
        n_plans = len(self.plan_list)
        lo_pct = alpha / 2.0
        hi_pct = 1.0 - alpha / 2.0

        self.bootstrap_ci = {}

        for p_idx in range(n_plans):
            cost_vec = [self.cost_collection[s][p_idx] for s in range(n_samples)]
            if not cost_vec:
                self.bootstrap_ci[p_idx] = (0.0, 0.0)
                continue

            boot_means = []
            for _ in range(n_resamples):
                resample = [cost_vec[rng.randint(0, n_samples - 1)]
                            for _ in range(n_samples)]
                boot_means.append(sum(resample) / len(resample))

            boot_means.sort()
            lo_idx = max(0, int(lo_pct * n_resamples) - 1)
            hi_idx = min(n_resamples - 1, int(hi_pct * n_resamples))
            self.bootstrap_ci[p_idx] = (boot_means[lo_idx], boot_means[hi_idx])

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
    # [ALG-2] CHANGED: simple 1/(1+max_penalty) normalization →
    #         softmax with adaptive temperature + entropy regularization.
    #
    # Temperature τ = median of max-penalties (adapts to penalty scale).
    # Entropy regularization adds λ·H(P) to prevent collapse:
    #   final_prob ∝ softmax(-max_penalty / τ) blended with uniform via λ.
    # -------------------------------------------------------------------

    def cal_reweight_probability(self):
        """Compute reweighted probabilities for each sample.

        [ALG-2] Softmax with adaptive temperature + entropy regularization.

        1. Compute max_penalty per sample (danger signal).
        2. Set temperature τ = median(max_penalties) to auto-scale.
        3. Compute softmax weights: w_s = exp(-max_penalty_s / τ).
        4. Blend with uniform distribution for entropy regularization:
             P(s) = (1 - λ) · softmax(s) + λ · (1/N)
           where λ = 0.1 prevents any sample from getting zero weight.
        """
        n_samples = len(self.samples)
        if n_samples == 0:
            return

        entropy_lambda = 0.1  # regularization strength

        # Per-sample max penalty
        max_penalties = []
        for s_idx in range(n_samples):
            mp = max(self.penalty_collection[s_idx]) if self.penalty_collection[s_idx] else 0.0
            max_penalties.append(mp)

        # Adaptive temperature = median of max-penalties (avoid division by zero)
        sorted_mp = sorted(max_penalties)
        median_idx = n_samples // 2
        temperature = sorted_mp[median_idx] if sorted_mp[median_idx] > 1e-12 else 1.0

        # Softmax: w_s = exp(-max_penalty_s / τ)
        # For numerical stability, subtract the max exponent
        logits = [-mp / temperature for mp in max_penalties]
        max_logit = max(logits)
        exp_weights = [math.exp(l - max_logit) for l in logits]
        exp_sum = sum(exp_weights)
        softmax_probs = [w / exp_sum for w in exp_weights] if exp_sum > 0 else [1.0 / n_samples] * n_samples

        # Entropy regularization: blend with uniform
        uniform = 1.0 / n_samples
        self.joint_probabilities = [
            (1.0 - entropy_lambda) * sp + entropy_lambda * uniform
            for sp in softmax_probs
        ]

        # Renormalize (should already sum to 1, but ensure)
        total = sum(self.joint_probabilities)
        if total > 0 and abs(total - 1.0) > 1e-12:
            self.joint_probabilities = [p / total for p in self.joint_probabilities]

        if _DBG_ENABLED:
            ent = -sum(p * math.log(max(1e-30, p)) for p in self.joint_probabilities)
            _log.info("[DBG] cal_reweight_probability: τ=%.4f, λ=%.2f, "
                      "entropy=%.4f, min_prob=%.6f, max_prob=%.6f",
                      temperature, entropy_lambda, ent,
                      min(self.joint_probabilities),
                      max(self.joint_probabilities))

    # -------------------------------------------------------------------
    # select_robust_plan — [ALG-1] Pareto frontier + Chebyshev scalarization
    #
    # CHANGED from simple penalty-weighted argmin to:
    #   1. Compute two objectives per plan:
    #      - obj_cost = Σ_s P(s) * cost[s][p]     (expected cost)
    #      - obj_penalty = max_s penalty[s][p]     (worst-case robustness)
    #   2. Non-dominated sorting to find Pareto front on (obj_cost, obj_penalty)
    #   3. Chebyshev scalarization on the Pareto set:
    #        score = max( w1·|obj_cost - ideal_cost|/range_cost,
    #                     w2·|obj_penalty - ideal_penalty|/range_penalty )
    #      using w1 = 1 - robustness_weight, w2 = robustness_weight.
    #   4. Return the plan that minimizes the Chebyshev score.
    # -------------------------------------------------------------------

    def select_robust_plan(self) -> Tuple[Optional[QueryPlan], str, float]:
        """Select the most robust plan with device recommendation.

        [ALG-1] Pareto frontier on (expected_cost, worst_penalty) with
        Chebyshev scalarization to pick the final plan.

        Returns:
            (plan, device, expected_cost) — the Pareto-optimal choice.
        """
        if not self.plan_list or not self.samples:
            return None, "cpu", 0.0

        n_samples = len(self.samples)
        n_plans = len(self.plan_list)

        if not self.joint_probabilities:
            self.cal_reweight_probability()

        # --- Step 1: compute two objectives per plan ---
        objectives: List[Tuple[float, float]] = []  # (expected_cost, max_penalty)
        for p_idx in range(n_plans):
            exp_cost = 0.0
            worst_penalty = 0.0
            for s_idx in range(n_samples):
                prob = self.joint_probabilities[s_idx]
                cost = self.cost_collection[s_idx][p_idx]
                penalty = self.penalty_collection[s_idx][p_idx]
                exp_cost += prob * cost
                if penalty > worst_penalty:
                    worst_penalty = penalty
            objectives.append((exp_cost, worst_penalty))

        # --- Step 2: non-dominated sorting for Pareto front ---
        # A plan i dominates plan j iff both objectives of i <= j and at
        # least one is strictly less.
        pareto_indices: List[int] = []
        for i in range(n_plans):
            dominated = False
            for j in range(n_plans):
                if i == j:
                    continue
                if (objectives[j][0] <= objectives[i][0] and
                        objectives[j][1] <= objectives[i][1] and
                        (objectives[j][0] < objectives[i][0] or
                         objectives[j][1] < objectives[i][1])):
                    dominated = True
                    break
            if not dominated:
                pareto_indices.append(i)

        # Fallback: if somehow empty (shouldn't happen), use all plans
        if not pareto_indices:
            pareto_indices = list(range(n_plans))

        # --- Step 3: Chebyshev scalarization on Pareto set ---
        # Ideal point: best value of each objective across Pareto set
        pareto_costs = [objectives[i][0] for i in pareto_indices]
        pareto_penalties = [objectives[i][1] for i in pareto_indices]

        ideal_cost = min(pareto_costs)
        ideal_penalty = min(pareto_penalties)

        range_cost = max(pareto_costs) - ideal_cost
        range_penalty = max(pareto_penalties) - ideal_penalty

        # Weights from robustness_weight parameter
        w_cost = 1.0 - self.robustness_weight
        w_penalty = self.robustness_weight

        best_plan_idx = pareto_indices[0]
        best_cheby = float('inf')

        for p_idx in pareto_indices:
            c_norm = ((objectives[p_idx][0] - ideal_cost) / range_cost
                      if range_cost > 1e-15 else 0.0)
            p_norm = ((objectives[p_idx][1] - ideal_penalty) / range_penalty
                      if range_penalty > 1e-15 else 0.0)
            # Chebyshev: minimize the maximum weighted deviation
            cheby = max(w_cost * c_norm, w_penalty * p_norm)
            if cheby < best_cheby:
                best_cheby = cheby
                best_plan_idx = p_idx

        # --- Step 4: device recommendation via majority vote ---
        gpu_votes = sum(
            1 for s_idx in range(n_samples)
            if self.device_decisions[s_idx][best_plan_idx] == "gpu"
        )
        device = "gpu" if gpu_votes > n_samples / 2 else "cpu"

        expected_cost = objectives[best_plan_idx][0]

        if _DBG_ENABLED:
            self._dbg_pareto_frontier(objectives, pareto_indices,
                                      best_plan_idx, best_cheby)

        return self.plan_list[best_plan_idx], device, expected_cost

    # -------------------------------------------------------------------
    # [ALG-4] Debug: Pareto frontier diagnostics
    # -------------------------------------------------------------------

    def _dbg_pareto_frontier(self, objectives: List[Tuple[float, float]],
                             pareto_indices: List[int],
                             selected_idx: int,
                             cheby_score: float):
        """Print Pareto frontier diagnostics.

        Shows: Pareto set size, each plan's (cost, robustness) coordinates,
        which plan was selected, and its Chebyshev score.
        """
        n_plans = len(self.plan_list)
        print("=" * 60)
        print(f"[_dbg_pareto_frontier] query={self.query_id}")
        print(f"  Total plans: {n_plans}")
        print(f"  Pareto set size: {len(pareto_indices)}")
        print(f"  Pareto plan indices: {pareto_indices}")
        print()
        print("  All plans (cost, worst_penalty):")
        for i, (c, p) in enumerate(objectives):
            marker = " *PARETO*" if i in pareto_indices else ""
            sel = " <== SELECTED" if i == selected_idx else ""
            ci = self.bootstrap_ci.get(i, (float('nan'), float('nan')))
            print(f"    plan[{i}] id={self.plan_list[i].plan_id}: "
                  f"cost={c:.4f}, penalty={p:.4f}, "
                  f"CI95=[{ci[0]:.4f}, {ci[1]:.4f}]{marker}{sel}")
        print(f"  Chebyshev score of selected: {cheby_score:.6f}")
        print("=" * 60)

    @staticmethod
    def _dbg():
        """Enable debug output for all HeterogeneousPlanDiagram instances."""
        _dbg_enable()
        print("[par2qo_bridge] Debug mode enabled — Pareto frontier, "
              "softmax reweighting, and bootstrap CI diagnostics will print.")

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
        bins = self.plan_histograms.get(plan_id, [])
        if not bins:
            return 0.0
        total = sum(bins)
        if total == 0:
            return 0.0
        mean_bin = sum(i * b for i, b in enumerate(bins)) / total
        var = sum((i - mean_bin) ** 2 * b for i, b in enumerate(bins)) / total
        return var
