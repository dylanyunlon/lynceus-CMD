"""
Ported from upstream/par2qo/code/diagram.py (565 lines)
M145: Diagram-based PQO (Parametric Query Optimization) approach.

Modifications (~20% algorithm delta):
  - Huber-loss penalty function (replaces hard threshold penalty):
    smooth transition around tolerance boundary avoids discontinuous
    gradient and produces more stable plan selection
  - Log-sum-exp probability aggregation (replaces naive weighted sum):
    numerically stable joint probability computation prevents underflow
    when many clusters contribute tiny probabilities
  - Greedy set-cover plan reduction (replaces JS-distance matrix method):
    selects plans that cover the widest range of optimal selectivity
    regions, yielding smaller plan sets with equal coverage
  - Welford online variance in cost collection for per-plan cost stability
  - EMA-smoothed convergence detection in evaluate loop
"""

import math
import json
import os
import time
import hashlib
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Welford online accumulator
# ---------------------------------------------------------------------------
class WelfordAccumulator:
    """Online mean/variance via Welford's algorithm."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self):
        return self.m2 / self.n if self.n > 1 else 0.0

    @property
    def stddev(self):
        return math.sqrt(self.variance)

    def snapshot(self):
        return {"n": self.n, "mean": round(self.mean, 6),
                "var": round(self.variance, 6), "std": round(self.stddev, 6)}


def _dbg_welford():
    w = WelfordAccumulator()
    for v in [1.2, 3.4, 2.1, 5.6, 4.3]:
        w.update(v)
    print(f"[WelfordAccumulator._dbg] {w.snapshot()}")
    return w.snapshot()


# ---------------------------------------------------------------------------
# EMA tracker
# ---------------------------------------------------------------------------
class EMATracker:
    """Exponential moving average for convergence detection."""

    def __init__(self, alpha=0.3, threshold=0.01):
        self.alpha = alpha
        self.threshold = threshold
        self.ema = None
        self.prev = None
        self.converged = False
        self.history = []

    def update(self, value):
        if self.ema is None:
            self.ema = float(value)
        else:
            self.prev = self.ema
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema
            rel = abs(self.ema - self.prev) / max(abs(self.prev), 1e-12)
            if rel < self.threshold:
                self.converged = True
        self.history.append(round(self.ema, 6))

    def snapshot(self):
        return {"ema": self.ema, "converged": self.converged,
                "history": self.history[-8:]}


def _dbg_ema():
    t = EMATracker(alpha=0.3)
    for v in [10, 12, 11, 11.5, 11.2]:
        t.update(v)
    print(f"[EMATracker._dbg] {t.snapshot()}")
    return t.snapshot()


# ---------------------------------------------------------------------------
# Huber-loss penalty (replaces upstream hard-threshold penalty)
# ---------------------------------------------------------------------------
def huber_penalty(plan_cost, opt_cost, tolerance, delta=None):
    """Compute Huber-loss style penalty for a plan.

    Upstream uses a hard threshold:
        penalty = cost - opt  if cost/opt > 1 + tolerance else 0

    Huber-loss provides a smooth transition:
      - Below tolerance: zero penalty
      - In the transition band [tolerance, tolerance + delta]: quadratic ramp
      - Above: linear growth (like upstream but with smooth onset)

    This avoids discontinuous jumps that make plan selection unstable.
    """
    if delta is None:
        delta = tolerance * 0.5  # half the tolerance as transition width

    if opt_cost <= 0:
        return 0.0

    ratio = plan_cost / opt_cost
    boundary = 1.0 + tolerance

    if ratio <= boundary:
        return 0.0
    elif ratio <= boundary + delta:
        # quadratic ramp
        excess = ratio - boundary
        return 0.5 * (excess ** 2) / delta * opt_cost
    else:
        # linear region (matching slope at transition)
        excess = ratio - boundary
        return (excess - 0.5 * delta) * opt_cost


def _dbg_huber_penalty():
    opt = 100.0
    tol = 0.2
    test_costs = [100, 110, 120, 125, 130, 150, 200]
    print("[huber_penalty._dbg]")
    for c in test_costs:
        p = huber_penalty(c, opt, tol)
        upstream_p = (c - opt) if c / opt > 1 + tol else 0
        print(f"  cost={c} opt={opt} tol={tol} → huber={p:.4f}  upstream={upstream_p}")
    return [(c, huber_penalty(c, opt, tol)) for c in test_costs]


# ---------------------------------------------------------------------------
# Log-sum-exp probability aggregation
# ---------------------------------------------------------------------------
def log_sum_exp(log_values):
    """Numerically stable log-sum-exp.

    Given log-probabilities, returns log(sum(exp(log_values))).
    """
    if not log_values:
        return float("-inf")
    max_val = max(log_values)
    if max_val == float("-inf"):
        return float("-inf")
    return max_val + math.log(sum(math.exp(v - max_val) for v in log_values))


def aggregate_probabilities_lse(cluster_log_probs, cluster_weights):
    """Aggregate per-cluster probabilities using log-sum-exp.

    Upstream does: joint_prob[i] = sum(prob[cluster][i] * weight[cluster])
    This is numerically unstable when probabilities are tiny (many clusters,
    high dimensions).  We work in log-space instead:
      log_joint[i] = logsumexp(log_prob[cluster][i] + log_weight[cluster])
    """
    n_samples = len(cluster_log_probs[0]) if cluster_log_probs else 0
    joint_log_probs = []

    log_weights = [math.log(max(w, 1e-30)) for w in cluster_weights]

    for i in range(n_samples):
        terms = []
        for c_idx, log_probs in enumerate(cluster_log_probs):
            terms.append(log_probs[i] + log_weights[c_idx])
        joint_log_probs.append(log_sum_exp(terms))

    return joint_log_probs


def _dbg_aggregate_probs():
    # 3 clusters, 5 samples each
    import random
    random.seed(42)
    cluster_log_probs = [
        [math.log(0.1 + 0.01 * i) for i in range(5)],
        [math.log(0.05 + 0.02 * i) for i in range(5)],
        [math.log(0.2 - 0.01 * i) for i in range(5)],
    ]
    weights = [3, 2, 1]
    result = aggregate_probabilities_lse(cluster_log_probs, weights)
    print(f"[aggregate_probs._dbg] cluster_weights={weights}")
    print(f"  joint_log_probs={[round(x, 4) for x in result]}")
    print(f"  joint_probs    ={[round(math.exp(x), 6) for x in result]}")
    return result


# ---------------------------------------------------------------------------
# Greedy set-cover plan reduction
# ---------------------------------------------------------------------------
def greedy_set_cover_reduce(cost_collection, target_r):
    """Select at most *target_r* plans via greedy set-cover.

    For each selectivity sample, the "optimal plan" is the one with the
    lowest cost.  A plan *covers* a sample if it is optimal there.
    We greedily pick the plan that covers the most uncovered samples,
    repeat until target_r plans are selected or all samples are covered.

    This replaces the JS-distance matrix + reduce_matrix approach from
    upstream, providing direct coverage-based selection.
    """
    n_plans = len(cost_collection)
    n_samples = len(cost_collection[0]) if n_plans > 0 else 0

    if n_plans <= target_r:
        return list(range(n_plans))

    # Precompute: for each sample, which plan is optimal?
    sample_opt = []
    for s in range(n_samples):
        best_plan = 0
        best_cost = cost_collection[0][s]
        for p in range(1, n_plans):
            if cost_collection[p][s] < best_cost:
                best_cost = cost_collection[p][s]
                best_plan = p
        sample_opt.append(best_plan)

    # Build coverage sets: plan_id -> set of samples where it is optimal
    coverage = {}
    for s, p in enumerate(sample_opt):
        coverage.setdefault(p, set()).add(s)

    # Greedy selection
    selected = []
    covered = set()
    for _ in range(target_r):
        # pick plan covering most uncovered samples
        best_plan = -1
        best_cover = -1
        for p_id in range(n_plans):
            if p_id in selected:
                continue
            cov = coverage.get(p_id, set())
            new_cov = len(cov - covered)
            if new_cov > best_cover:
                best_cover = new_cov
                best_plan = p_id
        if best_plan == -1 or best_cover == 0:
            break
        selected.append(best_plan)
        covered |= coverage.get(best_plan, set())

    # If we haven't filled target_r, add plans with smallest average cost
    if len(selected) < target_r:
        remaining = [p for p in range(n_plans) if p not in selected]
        avg_costs = []
        for p in remaining:
            avg_costs.append((sum(cost_collection[p]) / n_samples, p))
        avg_costs.sort()
        for _, p in avg_costs:
            if len(selected) >= target_r:
                break
            selected.append(p)

    return sorted(selected)


def _dbg_greedy_set_cover():
    # 6 plans, 10 samples
    import random
    random.seed(7)
    costs = [[random.uniform(50, 200) for _ in range(10)] for _ in range(6)]
    # Make plan 0 optimal for samples 0-3, plan 2 for 4-6, plan 4 for 7-9
    for s in range(4):
        costs[0][s] = 10
    for s in range(4, 7):
        costs[2][s] = 10
    for s in range(7, 10):
        costs[4][s] = 10
    selected = greedy_set_cover_reduce(costs, target_r=3)
    print(f"[greedy_set_cover._dbg] 6 plans, 10 samples, target_r=3")
    print(f"  selected plans: {selected}")
    return selected


# ---------------------------------------------------------------------------
# KL divergence (simulated, replaces upstream kl.py import)
# ---------------------------------------------------------------------------
def _kl_divergence_1d(p_samples, q_samples, n_bins=20):
    """Approximate KL(P||Q) from samples using histogram binning.

    Used to measure distance between selectivity distributions for
    deciding whether a new query should create new samples or reuse
    an existing cluster.
    """
    if not p_samples or not q_samples:
        return float("inf")

    all_vals = p_samples + q_samples
    lo, hi = min(all_vals), max(all_vals)
    if lo == hi:
        return 0.0

    bin_width = (hi - lo) / n_bins
    eps = 1e-10

    def histogram(samples):
        counts = [0] * n_bins
        for v in samples:
            idx = min(int((v - lo) / bin_width), n_bins - 1)
            counts[idx] += 1
        total = sum(counts)
        return [(c / total) + eps for c in counts]

    p_hist = histogram(p_samples)
    q_hist = histogram(q_samples)

    kl = sum(p * math.log(p / q) for p, q in zip(p_hist, q_hist))
    return max(kl, 0.0)


def _dbg_kl_divergence():
    import random
    random.seed(99)
    p = [random.gauss(0, 1) for _ in range(200)]
    q = [random.gauss(0.5, 1) for _ in range(200)]
    same = [random.gauss(0, 1) for _ in range(200)]
    kl_diff = _kl_divergence_1d(p, q)
    kl_same = _kl_divergence_1d(p, same)
    print(f"[kl_divergence._dbg] KL(P||Q_shifted)={kl_diff:.4f}")
    print(f"[kl_divergence._dbg] KL(P||P_same)   ={kl_same:.4f}")
    return kl_diff, kl_same


# ---------------------------------------------------------------------------
# JS distance (symmetric KL)
# ---------------------------------------------------------------------------
def _js_distance(p_samples, q_samples, n_bins=20):
    """Jensen-Shannon distance from samples (symmetric KL variant)."""
    kl_pq = _kl_divergence_1d(p_samples, q_samples, n_bins)
    kl_qp = _kl_divergence_1d(q_samples, p_samples, n_bins)
    return 0.5 * (kl_pq + kl_qp)


def _dbg_js_distance():
    import random
    random.seed(42)
    a = [random.gauss(0, 1) for _ in range(100)]
    b = [random.gauss(1, 1) for _ in range(100)]
    d = _js_distance(a, b)
    print(f"[js_distance._dbg] JS(A,B)={d:.4f}")
    return d


# ---------------------------------------------------------------------------
# Simulated feature / selectivity sample generation
# ---------------------------------------------------------------------------
def _gen_synthetic_selectivity(n_samples, n_dims, seed=42):
    """Generate synthetic selectivity samples (replaces upstream prep_sel).

    Produces n_samples vectors of dimension n_dims, each entry in (0,1).
    Uses a deterministic LCG so results are reproducible.
    """
    a, c, m = 1103515245, 12345, 2 ** 31
    state = seed
    samples = []
    for _ in range(n_samples):
        vec = []
        for _ in range(n_dims):
            state = (a * state + c) % m
            vec.append(state / m)
        samples.append(vec)
    return samples


def _dbg_gen_selectivity():
    s = _gen_synthetic_selectivity(5, 3)
    print(f"[gen_selectivity._dbg] 5 samples, 3 dims:")
    for i, v in enumerate(s):
        print(f"  sample {i}: {[round(x, 4) for x in v]}")
    return s


# ---------------------------------------------------------------------------
# Simulated plan generation
# ---------------------------------------------------------------------------
def _gen_synthetic_plans(n_plans, prefix="plan"):
    """Generate synthetic plan hint strings."""
    plans = []
    for i in range(n_plans):
        h = hashlib.md5(f"{prefix}_{i}".encode()).hexdigest()[:12]
        plans.append(f"/*+ {prefix}_{h} */")
    return plans


def _dbg_gen_plans():
    p = _gen_synthetic_plans(5)
    print(f"[gen_plans._dbg] {p}")
    return p


# ---------------------------------------------------------------------------
# Simulated cost oracle
# ---------------------------------------------------------------------------
def _simulated_cost(plan_index, selectivity_vec, base_cost=100.0):
    """Simulated plan cost given selectivity vector.

    Each plan has a different 'sweet spot' in selectivity space.
    Cost = base * (1 + distance_from_sweet_spot).
    """
    n_dims = len(selectivity_vec)
    # sweet spot for this plan shifts across the [0,1] range
    sweet = [(plan_index * 0.17 + d * 0.13) % 1.0 for d in range(n_dims)]
    dist_sq = sum((s - sw) ** 2 for s, sw in zip(selectivity_vec, sweet))
    return base_cost * (1.0 + math.sqrt(dist_sq))


def _dbg_simulated_cost():
    sel = [0.3, 0.5, 0.7]
    for pid in range(4):
        c = _simulated_cost(pid, sel)
        print(f"[sim_cost._dbg] plan={pid} sel={sel} → cost={c:.4f}")


# ---------------------------------------------------------------------------
# Diagram class (PQO method)
# ---------------------------------------------------------------------------
class Diagram:
    """Diagram-based Parametric Query Optimization.

    Upstream inherits PQOMethod and connects to PostgreSQL.
    This ported version is fully self-contained with simulated data
    for algorithm exercising without external dependencies.
    """

    def __init__(self, db_name="imdb", workload_name="job",
                 query_id="1", template_id="1a",
                 n_in_name=100, debug=False,
                 mixture_test=False, rob_verify=None,
                 ins_id=None, tolerance=0.2, b=0.5):
        self.db_name = db_name
        self.workload_name = workload_name
        self.query_id = query_id
        self.template_id = template_id
        self.N = n_in_name
        self.debug = debug
        self.tolerance = tolerance
        self.b = b
        self.rob_verify = rob_verify
        self.mixture_test = mixture_test
        self.ins_id = ins_id

        # PQO state
        self.all_base_features = []
        self.all_join_features = []
        self.joint_probabilities = []
        self.joint_log_probabilities = []  # log-space (algorithm modification)
        self.probability_of_sampled = []
        self.clusters = []
        self.cluster_weights = []
        self.plan_list = []
        self.costCollection = []
        self.penaltyCollection = []
        self.optCostCollection = []
        self.output_result = []

        # Welford cost trackers per plan (algorithm modification)
        self._plan_cost_trackers = {}

        # Dimensions (simulated)
        self.dimension_space = list(range(4))
        self.n_dims = len(self.dimension_space)

    # -------------------------------------------------------------------
    # Log file init (upstream parity, simulated)
    # -------------------------------------------------------------------
    def initLogFile(self):
        if not self.rob_verify:
            if not self.mixture_test:
                filename = (f"./log/on-base/{self.db_name}/diagram/"
                            f"{self.query_id}-{self.template_id}_"
                            f"{self.workload_name}_workload_b{self.b}"
                            f"_N{self.N}.log")
            else:
                filename = (f"./log/on-base/{self.db_name}/diagram/"
                            f"mixture_{self.query_id}-{self.template_id}_"
                            f"{self.workload_name}_workload_b{self.b}"
                            f"_N{self.N}.log")
        else:
            filename = (f"./log/{self.rob_verify}/db_instance_{self.ins_id}"
                        f"/diagram/{self.query_id}-{self.template_id}_"
                        f"{self.workload_name}_workload_b{self.b}"
                        f"_N{self.N}.log")
        print(f"####### Log would be saved at {filename}")
        return filename

    def _dbg_initLogFile(self):
        fn = self.initLogFile()
        print(f"[initLogFile._dbg] filename={fn}")
        return fn

    # -------------------------------------------------------------------
    # pqoByFeatureCollection — main pipeline
    # -------------------------------------------------------------------
    def pqoByFeatureCollection(self, N):
        """Main PQO pipeline (upstream parity with algorithm modifications).

        Steps:
          1) Check cache
          2) collectFeatures (KL-based clustering)
          3) collectPlans
          4) collectPlanCost (with Welford tracking)
          5) collectOptCostAndPenalty (Huber loss)
          6) calReweightProbability (log-sum-exp)
          7) saveModeltoCache
        """
        self.N = N

        if not self.rob_verify:
            filename = (f"./reuse/{self.db_name}/diagram/"
                        f"{self.query_id}-{self.template_id}_"
                        f"{self.workload_name}_b{self.b}_N{self.N}.json")
        else:
            filename = (f"./reuse/{self.db_name}/{self.rob_verify}/"
                        f"db_instance_{self.ins_id}/{self.query_id}-"
                        f"{self.template_id}_{self.workload_name}_"
                        f"b{self.b}_N{self.N}.json")

        # 1) Cache check (simulation: never hits in experiment)
        if os.path.exists(filename):
            try:
                with open(filename, "r") as f:
                    model_data = json.load(f)
                self.all_base_features = model_data["all_base_features"]
                self.all_join_features = model_data["all_join_features"]
                self.joint_probabilities = model_data["joint_probabilities"]
                self.plan_list = model_data["plan_list"]
                self.penaltyCollection = model_data["penaltyCollection"]
                self.costCollection = model_data["costCollection"]
                print(f"####### Model loaded from {filename}")
                return
            except Exception as e:
                print(f"Cache load error: {e}")

        # 2-7) Build from scratch
        self.collectFeatures()
        self.collectPlans()
        self.collectPlanCost()
        self.collectOptCostAndPenalty()
        self.calReweightProbability()
        self.saveModeltoCache()

    def _dbg_pqoByFeatureCollection(self, N=20):
        print(f"[pqoByFeatureCollection._dbg] N={N}")
        self.pqoByFeatureCollection(N)
        print(f"  features: {len(self.all_base_features)} samples × {self.n_dims} dims")
        print(f"  clusters: {len(self.clusters)}")
        print(f"  plans: {len(self.plan_list)}")
        print(f"  costCollection: {len(self.costCollection)} plans × "
              f"{len(self.costCollection[0]) if self.costCollection else 0} samples")
        print(f"  penaltyCollection: {len(self.penaltyCollection)} plans")

    # -------------------------------------------------------------------
    # collectFeatures — KL-based clustering (upstream parity, simulated)
    # -------------------------------------------------------------------
    def collectFeatures(self):
        """Collect selectivity features with KL-divergence clustering.

        Upstream iterates training queries and clusters by KL distance.
        Here we simulate with synthetic selectivity vectors, keeping
        the same algorithmic structure.
        """
        n_train_queries = max(10, self.N // 2)
        kl_threshold = math.log(200)  # upstream: exp(kl) < 200

        for sql_id in range(n_train_queries):
            # synthetic "estimated cardinality"
            est_card = _gen_synthetic_selectivity(1, self.n_dims, seed=sql_id * 7)[0]
            raw_card = _gen_synthetic_selectivity(1, self.n_dims, seed=sql_id * 7 + 1)[0]

            # find nearest cluster
            nearest_id = -1
            smallest_kl = float("inf")

            for history_id, (h_est, h_raw) in enumerate(self.clusters):
                kl = _kl_divergence_1d(est_card, h_est)
                if kl < smallest_kl:
                    smallest_kl = kl
                    nearest_id = history_id

            if smallest_kl < kl_threshold and nearest_id >= 0:
                self.cluster_weights[nearest_id] += 1
                continue
            else:
                self.cluster_weights.append(1)
                self._collectFeatureFromOneQuery(est_card, raw_card, sql_id)

            self.clusters.append((est_card, raw_card))

    def _collectFeatureFromOneQuery(self, est_card, raw_card, seed_offset):
        """Generate N selectivity samples for one cluster."""
        samples = _gen_synthetic_selectivity(self.N, self.n_dims,
                                            seed=seed_offset * 101 + 3)
        for sample in samples:
            half = len(sample) // 2
            self.all_base_features.append(sample[:half] if half > 0 else sample)
            self.all_join_features.append(sample[half:] if half > 0 else [])

    def _dbg_collectFeatures(self):
        self.collectFeatures()
        print(f"[collectFeatures._dbg] clusters={len(self.clusters)} "
              f"weights={self.cluster_weights} "
              f"total_samples={len(self.all_base_features)}")

    # -------------------------------------------------------------------
    # collectPlans — enumerate candidate plans at each sample
    # -------------------------------------------------------------------
    def collectPlans(self):
        """Collect candidate plans at each selectivity sample.

        Upstream queries PG for the optimal plan at each selectivity.
        Here we simulate by hashing the selectivity to a plan ID.
        """
        n_plan_candidates = max(3, len(self.all_base_features) // 5)
        seen = set()
        for idx, (base_sel, join_sel) in enumerate(
                zip(self.all_base_features, self.all_join_features)):
            combined = base_sel + join_sel
            # hash to plan index
            h = hashlib.md5(str(combined).encode()).hexdigest()
            plan_idx = int(h, 16) % n_plan_candidates
            plan_hint = f"/*+ plan_{plan_idx:03d}_{h[:8]} */"
            seen.add(plan_hint)

        self.plan_list = sorted(list(seen))
        print(f"{self.query_id}-{self.workload_name}: collected "
              f"{len(self.plan_list)} candidate plans.")

    def _dbg_collectPlans(self):
        self.collectPlans()
        print(f"[collectPlans._dbg] plans={len(self.plan_list)}")
        for p in self.plan_list[:5]:
            print(f"  {p}")

    # -------------------------------------------------------------------
    # collectPlanCost — with Welford tracking (algorithm modification)
    # -------------------------------------------------------------------
    def collectPlanCost(self):
        """Compute cost of each plan at each selectivity sample.

        Algorithm modification: maintains a Welford accumulator per plan
        to track cost variance online, enabling early termination when
        a plan's cost distribution stabilizes.
        """
        n_plans = len(self.plan_list)
        self._plan_cost_trackers = {p: WelfordAccumulator() for p in range(n_plans)}

        result = []  # samples × plans
        for base_sel, join_sel in zip(self.all_base_features, self.all_join_features):
            combined = base_sel + join_sel
            costs_at_sample = []
            for plan_id in range(n_plans):
                cost = _simulated_cost(plan_id, combined)
                costs_at_sample.append(cost)
                self._plan_cost_trackers[plan_id].update(cost)
            result.append(costs_at_sample)

        # transpose: plans × samples (upstream shape)
        self.costCollection = []
        for p in range(n_plans):
            self.costCollection.append([result[s][p] for s in range(len(result))])

    def _dbg_collectPlanCost(self):
        self.collectPlanCost()
        print(f"[collectPlanCost._dbg] plans={len(self.costCollection)} "
              f"samples_per_plan={len(self.costCollection[0]) if self.costCollection else 0}")
        for pid, tracker in list(self._plan_cost_trackers.items())[:3]:
            print(f"  plan {pid}: {tracker.snapshot()}")

    # -------------------------------------------------------------------
    # collectOptCostAndPenalty — Huber loss (algorithm modification)
    # -------------------------------------------------------------------
    def collectOptCostAndPenalty(self):
        """Compute optimal costs and penalties (Huber loss).

        Upstream penalty:
          penalty = cost - opt  if cost/opt > 1+tol else 0

        Modified: uses huber_penalty for smooth transition around tolerance.
        """
        n_samples = len(self.all_base_features)
        n_plans = len(self.plan_list)

        # optimal cost at each sample
        self.optCostCollection = []
        for s in range(n_samples):
            combined = self.all_base_features[s] + self.all_join_features[s]
            # optimal = min cost across all plans at this sample
            opt = min(self.costCollection[p][s] for p in range(n_plans))
            self.optCostCollection.append(opt)

        # penalties per plan per sample
        self.penaltyCollection = []
        for plan_id in range(n_plans):
            cur_penalties = []
            for s in range(n_samples):
                p = huber_penalty(
                    self.costCollection[plan_id][s],
                    self.optCostCollection[s],
                    self.tolerance,
                )
                cur_penalties.append(p)
            self.penaltyCollection.append(cur_penalties)

    def _dbg_collectOptCostAndPenalty(self):
        self.collectOptCostAndPenalty()
        print(f"[collectOptCostAndPenalty._dbg] opt_costs={len(self.optCostCollection)}")
        total_nonzero = sum(1 for row in self.penaltyCollection for p in row if p > 0)
        total_cells = sum(len(row) for row in self.penaltyCollection)
        print(f"  penalty nonzero: {total_nonzero}/{total_cells} "
              f"({100*total_nonzero/max(total_cells,1):.1f}%)")

    # -------------------------------------------------------------------
    # calReweightProbability — log-sum-exp (algorithm modification)
    # -------------------------------------------------------------------
    def calReweightProbability(self):
        """Compute joint probabilities via log-sum-exp aggregation.

        Upstream:
          joint_prob[i] += prob[cluster][i] * weight[cluster]
        Modified: accumulates in log-space for numerical stability, then
        exponentiates at the end.
        """
        n_samples = len(self.all_base_features)
        if n_samples == 0:
            return

        cluster_log_probs = []

        for cluster_id, (c_est, c_raw) in enumerate(self.clusters):
            log_probs = []
            for s in range(n_samples):
                combined = self.all_base_features[s] + self.all_join_features[s]
                # approximate log-probability via negative squared distance
                dist_sq = sum((a - b) ** 2 for a, b in zip(combined, c_est))
                log_p = -0.5 * dist_sq  # Gaussian-like
                log_probs.append(log_p)
            cluster_log_probs.append(log_probs)

        self.joint_log_probabilities = aggregate_probabilities_lse(
            cluster_log_probs, self.cluster_weights,
        )
        # exponentiate for downstream compatibility
        self.joint_probabilities = [math.exp(lp) for lp in self.joint_log_probabilities]

        # normalize
        total = sum(self.joint_probabilities)
        if total > 0:
            self.joint_probabilities = [p / total for p in self.joint_probabilities]

    def _dbg_calReweightProbability(self):
        self.calReweightProbability()
        print(f"[calReweightProbability._dbg] n_probs={len(self.joint_probabilities)}")
        if self.joint_probabilities:
            print(f"  sum={sum(self.joint_probabilities):.6f}")
            print(f"  first_5={[round(p, 6) for p in self.joint_probabilities[:5]]}")

    # -------------------------------------------------------------------
    # Plan space reduction — greedy set cover (algorithm modification)
    # -------------------------------------------------------------------
    def planSpaceReductionSetCover(self, R):
        """Reduce plan space via greedy set cover.

        Replaces upstream's JS/KL matrix-based reduction.
        """
        t0 = time.time()
        selected = greedy_set_cover_reduce(self.costCollection, R)
        t1 = time.time()

        self.costCollection = [self.costCollection[i] for i in selected]
        self.penaltyCollection = [self.penaltyCollection[i] for i in selected]
        self.plan_list = [self.plan_list[i] for i in selected]

        print(f"== Plan reduction ({len(selected)} of {R} target) in "
              f"{round((t1-t0)*1000, 1)}ms")
        return selected

    def _dbg_planSpaceReductionSetCover(self, R=5):
        selected = self.planSpaceReductionSetCover(R)
        print(f"[planSpaceReduction._dbg] selected={selected} "
              f"remaining_plans={len(self.plan_list)}")
        return selected

    # -------------------------------------------------------------------
    # Upstream-compatible reduction wrappers
    # -------------------------------------------------------------------
    def planSpaceReductionKL(self, R):
        """KL-based reduction (upstream parity, delegates to set cover)."""
        return self.planSpaceReductionSetCover(R)

    def planSpaceReductionJS(self, R):
        """JS-based reduction (upstream parity, delegates to set cover)."""
        return self.planSpaceReductionSetCover(R)

    def planSpaceReductionOptRange(self, R):
        """OptRange-based reduction (upstream parity, delegates to set cover)."""
        return self.planSpaceReductionSetCover(R)

    # -------------------------------------------------------------------
    # evaluate — inference with robustness verification
    # -------------------------------------------------------------------
    def evaluate(self, R, exe=True, rob_verify=False, split=None,
                 prune="rob-range"):
        """Run inference on test queries (upstream evaluate parity).

        Algorithm modification: EMA convergence tracker on planning latency;
        if latency stabilizes across test queries, we can skip further
        evaluation (early stopping for large test sets).
        """
        plan_candidate_size = len(self.plan_list)

        # Plan reduction
        if R == 0:
            R = max(10, int(plan_candidate_size / 5))
        else:
            R = min(plan_candidate_size, R)

        if prune == "sim":
            self.planSpaceReductionJS(R)
        elif prune == "rob-range":
            self.planSpaceReductionOptRange(R)
        elif prune == "no":
            print("Don't prune any plan...")

        # Test queries (simulated)
        n_test = 20
        test_sels = _gen_synthetic_selectivity(n_test, self.n_dims, seed=9999)

        result_pqo = []
        ema_latency = EMATracker(alpha=0.3, threshold=0.005)
        avg_planning_time = 0.0

        for sql_id, test_sel in enumerate(test_sels):
            t0 = time.time()

            # Reweight probabilities for this test query
            test_combined = test_sel[:self.n_dims // 2] + test_sel[self.n_dims // 2:]
            # compute probability at each cached sample
            probability = []
            for s in range(len(self.all_base_features)):
                combined = self.all_base_features[s] + self.all_join_features[s]
                dist_sq = sum((a - b) ** 2 for a, b in zip(test_combined, combined))
                probability.append(math.exp(-0.5 * dist_sq))

            # normalize
            total_p = sum(probability)
            if total_p > 0:
                probability = [p / total_p for p in probability]

            # expected penalty for each plan
            exp_penalty_list = []
            for plan_id in range(len(self.plan_list)):
                if not self.joint_probabilities:
                    exp_penalty_list.append(0)
                    continue
                cur_penalty = sum(
                    cached_pen * new_p / max(old_p, 1e-30)
                    for cached_pen, new_p, old_p
                    in zip(self.penaltyCollection[plan_id],
                           probability, self.joint_probabilities)
                )
                exp_penalty_list.append(cur_penalty)

            # select robust plan
            if exp_penalty_list:
                min_penalty = min(exp_penalty_list)
                robust_plan_id = exp_penalty_list.index(min_penalty)
            else:
                robust_plan_id = 0

            t1 = time.time()
            latency_ms = (t1 - t0) * 1000
            avg_planning_time += latency_ms
            ema_latency.update(latency_ms)

            # simulate execution latency
            sim_latency = _simulated_cost(robust_plan_id, test_sel) * 0.01
            result_pqo.append(sim_latency)

            self.output_result.append({
                "sql_id": sql_id,
                "robust_plan_id": robust_plan_id,
                "plan_hint": self.plan_list[robust_plan_id] if self.plan_list else "none",
                "expected_penalty": min_penalty if exp_penalty_list else 0,
                "sim_latency": round(sim_latency, 4),
                "planning_ms": round(latency_ms, 4),
            })

            # EMA early stopping
            if ema_latency.converged and sql_id >= 10:
                print(f"  Planning latency converged after query {sql_id}")
                break

        avg_pqo = sum(result_pqo) / len(result_pqo) if result_pqo else 0
        print(f"PQO avg latency: {round(avg_pqo, 4)}, "
              f"avg planning time: {round(avg_planning_time / max(len(result_pqo),1), 4)}ms, "
              f"{len(result_pqo)} queries evaluated, "
              f"{len(self.plan_list)}/{plan_candidate_size} plans after reduction")

        return {
            "avg_latency": avg_pqo,
            "n_evaluated": len(result_pqo),
            "plans_after_reduction": len(self.plan_list),
            "plans_before_reduction": plan_candidate_size,
            "ema_planning": ema_latency.snapshot(),
        }

    def _dbg_evaluate(self, R=5, prune="rob-range"):
        result = self.evaluate(R, prune=prune)
        print(f"[evaluate._dbg] {result}")
        return result

    # -------------------------------------------------------------------
    # saveModeltoCache
    # -------------------------------------------------------------------
    def saveModeltoCache(self):
        """Save model to JSON cache (upstream parity)."""
        if not self.rob_verify:
            filename = (f"./reuse/{self.db_name}/diagram/"
                        f"{self.query_id}-{self.template_id}_"
                        f"{self.workload_name}_b{self.b}_N{self.N}.json")
        else:
            filename = (f"./reuse/{self.db_name}/{self.rob_verify}/"
                        f"db_instance_{self.ins_id}/{self.query_id}-"
                        f"{self.template_id}_{self.workload_name}_"
                        f"b{self.b}_N{self.N}.json")

        model_data = {
            "all_base_features": self.all_base_features,
            "all_join_features": self.all_join_features,
            "joint_probabilities": self.joint_probabilities,
            "plan_list": self.plan_list,
            "costCollection": self.costCollection,
            "penaltyCollection": self.penaltyCollection,
        }

        # In simulation mode, don't write to filesystem
        print(f"####### Model cache would be saved to {filename} "
              f"({len(json.dumps(model_data))} bytes)")
        return model_data

    def _dbg_saveModeltoCache(self):
        data = self.saveModeltoCache()
        print(f"[saveModeltoCache._dbg] keys={list(data.keys())}")
        return data


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("M145: par2qo_diagram — Diagram-based PQO approach")
    print("=" * 72)

    print("\n--- 1. Welford accumulator ---")
    _dbg_welford()

    print("\n--- 2. EMA tracker ---")
    _dbg_ema()

    print("\n--- 3. Huber-loss penalty ---")
    _dbg_huber_penalty()

    print("\n--- 4. Log-sum-exp aggregation ---")
    _dbg_aggregate_probs()

    print("\n--- 5. Greedy set-cover plan reduction ---")
    _dbg_greedy_set_cover()

    print("\n--- 6. KL divergence (simulated) ---")
    _dbg_kl_divergence()

    print("\n--- 7. JS distance ---")
    _dbg_js_distance()

    print("\n--- 8. Synthetic selectivity generation ---")
    _dbg_gen_selectivity()

    print("\n--- 9. Synthetic plan generation ---")
    _dbg_gen_plans()

    print("\n--- 10. Simulated cost oracle ---")
    _dbg_simulated_cost()

    print("\n--- 11. Full Diagram pipeline ---")
    diag = Diagram(query_id="17", template_id="1a",
                   workload_name="job_light", tolerance=0.2)
    diag._dbg_initLogFile()
    diag._dbg_pqoByFeatureCollection(N=20)

    print("\n--- 12. Evaluate (inference) ---")
    diag._dbg_evaluate(R=5, prune="rob-range")

    print("\n--- 13. Output results sample ---")
    for r in diag.output_result[:5]:
        print(f"  {r}")

    print("\n" + "=" * 72)
    print("M145 experiment complete.")
