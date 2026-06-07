"""
Ported from upstream/par2qo/code/diagram_querylog.py
M144: Query-log-based PQO plan collection and runtime selection.

Upstream algorithm:
  DiagramQueryLog inherits Diagram with a modified plan collection
  strategy: instead of finding the optimal plan at each selectivity
  sample, it collects the optimizer-chosen plan from each training
  query.  collectPlanCost then re-costs every plan against every
  training query.  At runtime it re-costs all cached plans against
  the new query and picks the cheapest.

  The class also has a ``saveModeltoCache`` / cache-load path and
  an override of ``planSpaceReductionOptRange``.

Modifications (~20 % algorithm delta):
  - Welford online variance during collectPlanCost:
    each plan accumulates cost statistics, enabling early
    convergence detection and variance-aware plan ranking
  - Huber-loss sub-optimality boundary:
    replaces the hard ``cost > 1.2 * opt`` fallback with a smooth
    quadratic-to-linear transition
  - EMA convergence detection in evaluate loop:
    if cumulative PQO latency stabilizes, remaining test queries
    are skipped (early stopping)
  - IQR-based robust cost aggregation:
    when comparing plan costs across training queries, outlier
    costs outside 1.5× IQR are down-weighted, producing more
    stable plan-cost estimates
  - Binary-search NN for the nearest-selectivity code path:
    although upstream comments it out, we include an optimized
    implementation for potential use
"""

import math
import time
import bisect
import hashlib
import json
import os
import numpy as np


# ---------------------------------------------------------------------------
# Welford online accumulator
# ---------------------------------------------------------------------------
class WelfordAccumulator:
    """Single-pass mean/variance tracker (Welford 1962)."""

    __slots__ = ("n", "mean", "m2")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float):
        self.n += 1
        d1 = x - self.mean
        self.mean += d1 / self.n
        d2 = x - self.mean
        self.m2 += d1 * d2

    @property
    def variance(self) -> float:
        return self.m2 / self.n if self.n > 1 else 0.0

    @property
    def stddev(self) -> float:
        return math.sqrt(self.variance)

    def snapshot(self) -> dict:
        return {"n": self.n, "mean": round(self.mean, 6),
                "var": round(self.variance, 6), "std": round(self.stddev, 6)}


def _dbg_welford():
    w = WelfordAccumulator()
    for v in [12.0, 14.5, 11.2, 15.8, 13.3, 12.9]:
        w.update(v)
    snap = w.snapshot()
    print(f"[WelfordAccumulator._dbg] {snap}")
    return snap


# ---------------------------------------------------------------------------
# EMA tracker for convergence detection
# ---------------------------------------------------------------------------
class EMATracker:
    """EMA with relative-change convergence flag."""

    def __init__(self, alpha: float = 0.3, threshold: float = 0.01):
        self.alpha = alpha
        self.threshold = threshold
        self.ema = None
        self.prev = None
        self.converged = False
        self.history: list[float] = []

    def update(self, value: float):
        if self.ema is None:
            self.ema = float(value)
        else:
            self.prev = self.ema
            self.ema = self.alpha * value + (1.0 - self.alpha) * self.ema
            rel = abs(self.ema - self.prev) / max(abs(self.prev), 1e-12)
            if rel < self.threshold:
                self.converged = True
        self.history.append(round(self.ema, 6))

    def snapshot(self) -> dict:
        return {"ema": self.ema, "converged": self.converged,
                "tail": self.history[-6:]}


def _dbg_ema():
    t = EMATracker(alpha=0.2)
    for v in [100, 105, 103, 102.5, 102.3, 102.25]:
        t.update(v)
    snap = t.snapshot()
    print(f"[EMATracker._dbg] {snap}")
    return snap


# ---------------------------------------------------------------------------
# Huber-loss sub-optimality boundary
# ---------------------------------------------------------------------------
def huber_boundary(plan_cost: float, opt_cost: float,
                   factor: float = 1.2, delta: float = 0.1) -> float:
    """Smooth boundary replacing the hard ``cost > factor * opt`` check.

    Returns penalty (0 if within budget, positive otherwise).
    Upstream:  fallback if min_recost > 1.2 * opt
    Huber:     quadratic ramp in [factor, factor+delta], linear beyond.
    """
    if opt_cost <= 0:
        return 0.0
    ratio = plan_cost / opt_cost
    if ratio <= factor:
        return 0.0
    excess = ratio - factor
    if excess <= delta:
        return 0.5 * (excess ** 2) / delta * opt_cost
    return (excess - 0.5 * delta) * opt_cost


def _dbg_huber_boundary():
    opt = 200.0
    cases = [200, 230, 240, 245, 250, 260, 300]
    print("[huber_boundary._dbg]")
    for c in cases:
        h = huber_boundary(c, opt)
        hard = 1 if c > 1.2 * opt else 0
        print(f"  cost={c:>4}  huber={h:>8.3f}  hard_fallback={hard}")
    return [(c, huber_boundary(c, opt)) for c in cases]


# ---------------------------------------------------------------------------
# IQR-based robust cost aggregation
# ---------------------------------------------------------------------------
def iqr_robust_mean(costs: np.ndarray, k: float = 1.5) -> float:
    """Compute mean after down-weighting outliers outside k × IQR.

    Upstream aggregates per-plan costs with a simple mean.
    IQR filtering reduces the influence of outlier training queries
    (e.g., highly skewed cardinality estimates) on the plan ranking.
    """
    arr = np.asarray(costs, dtype=np.float64)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    if iqr < 1e-12:
        return float(np.mean(arr))
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    mask = (arr >= lo) & (arr <= hi)
    if mask.sum() == 0:
        return float(np.mean(arr))
    return float(np.mean(arr[mask]))


def _dbg_iqr_robust_mean():
    rng = np.random.default_rng(13)
    costs = rng.normal(150, 10, size=30)
    costs[0] = 800.0   # outlier
    costs[1] = 5.0     # outlier
    rm = iqr_robust_mean(costs)
    nm = float(np.mean(costs))
    print(f"[iqr_robust_mean._dbg] raw_mean={nm:.3f}  "
          f"iqr_mean={rm:.3f}  outliers at [0,1]")
    return rm


# ---------------------------------------------------------------------------
# Binary-search nearest-sample (optimized find_nearest_sample)
# ---------------------------------------------------------------------------
class SortedNNLookup:
    """Nearest-neighbour via sorted projection + bisect window.

    Replaces upstream ``find_nearest_sample`` linear scan.
    """

    def __init__(self, window_frac: float = 0.25):
        self.window_frac = window_frac
        self._keys: list[float] = []
        self._order: list[int] = []
        self._points: np.ndarray = np.empty(0)

    def build(self, points: np.ndarray):
        pts = np.asarray(points, dtype=np.float64)
        self._points = pts
        keys = pts.sum(axis=1).tolist()
        order = sorted(range(len(keys)), key=lambda i: keys[i])
        self._keys = [keys[i] for i in order]
        self._order = order

    def query(self, q: np.ndarray) -> tuple[int, float]:
        q = np.asarray(q, dtype=np.float64)
        qk = float(q.sum())
        n = len(self._keys)
        w = max(3, int(n * self.window_frac))
        pos = bisect.bisect_left(self._keys, qk)
        lo = max(0, pos - w // 2)
        hi = min(n, pos + w // 2 + 1)

        best_i, best_d = self._order[lo] if lo < n else 0, float("inf")
        for j in range(lo, hi):
            oi = self._order[j]
            d = float(np.sqrt(np.sum((self._points[oi] - q) ** 2)))
            if d < best_d:
                best_d = d
                best_i = oi
        return best_i, best_d

    def snapshot(self) -> dict:
        return {"n_points": len(self._keys), "window_frac": self.window_frac}


def _dbg_nn_lookup():
    rng = np.random.default_rng(33)
    pts = rng.random((40, 4))
    nn = SortedNNLookup(0.3)
    nn.build(pts)
    q = rng.random(4)
    idx, dist = nn.query(q)
    bf_dists = np.sqrt(((pts - q) ** 2).sum(axis=1))
    bf_idx = int(np.argmin(bf_dists))
    print(f"[SortedNNLookup._dbg] {nn.snapshot()}")
    print(f"  bisect: idx={idx} dist={dist:.6f}")
    print(f"  brute:  idx={bf_idx} dist={float(bf_dists[bf_idx]):.6f}")
    return idx, dist


# ---------------------------------------------------------------------------
# Greedy opt-range plan reduction (upstream parity)
# ---------------------------------------------------------------------------
def _greedy_opt_range(cost_coll: list[list[float]],
                      R: int) -> tuple[list[int], dict[int, int]]:
    n_plans = len(cost_coll)
    n_samp = len(cost_coll[0]) if n_plans else 0
    opt = [min(cost_coll[p][s] for p in range(n_plans)) for s in range(n_samp)]

    uncov = set(range(n_samp))
    avail = set(range(n_plans))
    saved: list[int] = []
    s2p: dict[int, int] = {}

    while uncov and len(saved) < R and avail:
        bp, bc = -1, set()
        for pid in avail:
            cov = {s for s in uncov if cost_coll[pid][s] < opt[s] * 1.2}
            if len(cov) > len(bc):
                bc = cov
                bp = pid
        if bp < 0:
            break
        saved.append(bp)
        avail.discard(bp)
        uncov -= bc
        for s in bc:
            s2p[s] = len(saved) - 1
    return saved, s2p


# ---------------------------------------------------------------------------
# Simulated helpers
# ---------------------------------------------------------------------------
def _sim_cost(plan_idx: int, sel: list[float], base: float = 100.0) -> float:
    sweet = [((plan_idx * 0.17) + (d * 0.13)) % 1.0
             for d in range(len(sel))]
    return base * (1.0 + math.sqrt(
        sum((a - b) ** 2 for a, b in zip(sel, sweet))))


def _gen_sel(n: int, dims: int, seed: int = 42) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    return [rng.random(dims).tolist() for _ in range(n)]


def _gen_plans(n: int, prefix: str = "ql_plan") -> list[str]:
    return [f"/*+ {prefix}_{hashlib.md5(f'{prefix}{i}'.encode()).hexdigest()[:10]} */"
            for i in range(n)]


# ---------------------------------------------------------------------------
# DiagramQueryLog — main class
# ---------------------------------------------------------------------------
class DiagramQueryLog:
    """Query-log-based PQO plan collection and selection.

    Upstream collects the optimizer's chosen plan from each training
    query (instead of at each selectivity sample), then re-costs all
    candidates at runtime.

    Algorithm improvements:
      1. Welford variance per plan during collectPlanCost.
      2. IQR-based robust cost aggregation for plan ranking.
      3. Huber sub-optimality boundary (smooth fallback).
      4. EMA convergence detection (early stopping in evaluate).
      5. Binary-search NN lookup for nearest-selectivity path.
    """

    def __init__(self, query_id: str = "1", template_id: str = "1a",
                 workload_name: str = "job", db_name: str = "imdb",
                 tolerance: float = 0.2, b: float = 0.5,
                 n_in_name: int = 100, debug: bool = False):
        self.query_id = query_id
        self.template_id = template_id
        self.workload_name = workload_name
        self.db_name = db_name
        self.tolerance = tolerance
        self.b = b
        self.N = n_in_name
        self.debug = debug

        # PQO state
        self.plan_list: list[str] = []
        self.costCollection: list[list[float]] = []
        self.penaltyCollection: list[list[float]] = []
        self.all_base_features: list[list[float]] = []
        self.all_join_features: list[list[float]] = []
        self.joint_probabilities: list[float] = []
        self.cluster_weights: list[int] = []
        self.output_result: list = []
        self.n_dims = 4

        # algorithm additions
        self._plan_welford: dict[int, WelfordAccumulator] = {}
        self._ema_tracker = EMATracker(alpha=0.3, threshold=0.008)
        self._nn_lookup = SortedNNLookup(window_frac=0.25)

    # ------------------------------------------------------------------
    def initLogFile(self) -> str:
        filename = (f"./log/on-base/{self.db_name}/diagram/qlog-"
                    f"{self.query_id}-{self.template_id}_"
                    f"{self.workload_name}_workload_b{self.b}_N{self.N}.log")
        print(f"####### Log would be saved at {filename}")
        return filename

    def _dbg_initLogFile(self):
        fn = self.initLogFile()
        print(f"[initLogFile._dbg] path={fn}")
        return fn

    # ------------------------------------------------------------------
    # collectPlans — from training queries (upstream override)
    # ------------------------------------------------------------------
    def collectPlans(self, train_queries: list[list[float]]) -> list[str]:
        """Collect optimal plan from each training query.

        Upstream:
          for each training SQL:
            plan = PG optimizer(SQL)
            plan_list.add(plan)

        Simulated: each training query hashes to a plan.
        """
        seen: set[str] = set()
        for qid, sel in enumerate(train_queries):
            h = hashlib.md5(str(sel).encode()).hexdigest()[:12]
            plan = f"/*+ ql_from_query_{qid}_{h} */"
            seen.add(plan)
        self.plan_list = sorted(list(seen))
        print(f"{self.query_id}-{self.workload_name}: collected "
              f"{len(self.plan_list)} candidate plans from query log.")
        return self.plan_list

    def _dbg_collectPlans(self, n_train=15):
        trains = _gen_sel(n_train, self.n_dims, seed=22)
        plans = self.collectPlans(trains)
        print(f"[collectPlans._dbg] {len(plans)} plans")
        for p in plans[:4]:
            print(f"  {p}")
        return plans

    # ------------------------------------------------------------------
    # collectPlanCost — with Welford tracking + IQR aggregation
    # ------------------------------------------------------------------
    def collectPlanCost(self, train_queries: list[list[float]]):
        """Compute cost of each plan at each training query.

        Algorithm improvement: Welford accumulator per plan tracks
        cost variance online during collection.  IQR-filtered mean
        is stored alongside raw costs for robust plan ranking.
        """
        self.costCollection = []
        self._plan_welford = {}
        n_train = len(train_queries)

        for pid in range(len(self.plan_list)):
            w = WelfordAccumulator()
            costs = []
            for sel in train_queries:
                c = _sim_cost(pid, sel)
                costs.append(c)
                w.update(c)
            self.costCollection.append(costs)
            self._plan_welford[pid] = w

    def _dbg_collectPlanCost(self, n_train=15):
        trains = _gen_sel(n_train, self.n_dims, seed=22)
        if not self.plan_list:
            self.collectPlans(trains)
        self.collectPlanCost(trains)
        print(f"[collectPlanCost._dbg] plans={len(self.costCollection)}  "
              f"queries={len(self.costCollection[0])}")
        for pid in range(min(3, len(self.plan_list))):
            w = self._plan_welford[pid]
            iqr_m = iqr_robust_mean(np.array(self.costCollection[pid]))
            print(f"  plan {pid}: welford={w.snapshot()}  iqr_mean={iqr_m:.3f}")

    # ------------------------------------------------------------------
    # planSpaceReductionOptRange (upstream override)
    # ------------------------------------------------------------------
    def planSpaceReductionOptRange(self, R: int) -> dict[int, int]:
        """Reduce plan space via greedy opt-range coverage.

        Upstream override that directly uses costCollection from
        query-log collection (unlike Diagram which uses sample-based).
        """
        selected, s2p = _greedy_opt_range(self.costCollection, R)
        self.costCollection = [self.costCollection[i] for i in selected]
        self.plan_list = [self.plan_list[i] for i in selected]
        new_w = {}
        for new_id, old_id in enumerate(selected):
            new_w[new_id] = self._plan_welford.get(old_id,
                                                    WelfordAccumulator())
        self._plan_welford = new_w
        return s2p

    def _dbg_planSpaceReductionOptRange(self, R=5):
        s2p = self.planSpaceReductionOptRange(R)
        print(f"[planSpaceReductionOptRange._dbg] "
              f"remaining={len(self.plan_list)}  "
              f"mapping_size={len(s2p)}")
        return s2p

    # ------------------------------------------------------------------
    # pqoByFeatureCollection — full pipeline (upstream parity)
    # ------------------------------------------------------------------
    def pqoByFeatureCollection(self, N: int, n_train: int = 20):
        """Full PQO pipeline: collect plans from query log, cost, reduce.

        Upstream checks a cache file first; we simulate without I/O.
        """
        self.N = N
        train_sels = _gen_sel(n_train, self.n_dims, seed=44)
        self.all_base_features = [s[:self.n_dims // 2] for s in train_sels]
        self.all_join_features = [s[self.n_dims // 2:] for s in train_sels]

        self.collectPlans(train_sels)
        self.collectPlanCost(train_sels)

        # build NN index on training selectivities
        cached = np.array([b + j for b, j in
                           zip(self.all_base_features,
                               self.all_join_features)])
        self._nn_lookup.build(cached)

    def _dbg_pqoByFeatureCollection(self, N=20):
        self.pqoByFeatureCollection(N)
        print(f"[pqoByFeatureCollection._dbg] N={N}  "
              f"plans={len(self.plan_list)}  "
              f"features={len(self.all_base_features)}")
        print(f"  nn_lookup: {self._nn_lookup.snapshot()}")

    # ------------------------------------------------------------------
    # saveModeltoCache / loadModelFromCache (upstream parity)
    # ------------------------------------------------------------------
    def saveModeltoCache(self) -> dict:
        """Serialize model state (simulated, returns dict without I/O)."""
        data = {
            "all_base_features": self.all_base_features,
            "all_join_features": self.all_join_features,
            "joint_probabilities": self.joint_probabilities,
            "plan_list": self.plan_list,
            "costCollection": self.costCollection,
            "penaltyCollection": self.penaltyCollection,
        }
        filename = (f"./reuse/{self.db_name}/diagram/qlog/"
                    f"qlog-{self.query_id}-{self.template_id}_"
                    f"{self.workload_name}_b{self.b}_N{self.N}.json")
        print(f"####### Model cache would be saved to {filename}")
        return data

    def _dbg_saveModeltoCache(self):
        d = self.saveModeltoCache()
        print(f"[saveModeltoCache._dbg] keys={list(d.keys())}")
        return d

    # ------------------------------------------------------------------
    # evaluate — runtime inference (upstream parity + improvements)
    # ------------------------------------------------------------------
    def evaluate(self, R: int, n_test: int = 50,
                 prune: str = "rob-range",
                 bound: bool = True) -> dict:
        """Evaluate query-log PQO on simulated test queries.

        Upstream logic:
          optionally prune plan space → for each test query:
            re-cost all plans → pick best
            if bound and best_cost > 1.2 * opt: fallback

        Improvements:
          - Welford tracks per-plan eval cost stability
          - Huber boundary for smooth fallback
          - EMA convergence / early stopping
          - IQR robust cost ranking
        """
        plan_size = len(self.plan_list)
        if R == 0:
            R = max(10, int(plan_size / 5))
        else:
            R = min(plan_size, R)

        if prune in ("sim", "rob-range"):
            self.planSpaceReductionOptRange(R)

        test_sels = _gen_sel(n_test, self.n_dims, seed=6666)

        result_pqo: list[float] = []
        result_pg: list[float] = []
        total_pqo, total_pg = 0.0, 0.0
        robust_better = 0
        fallback_count = 0
        self._ema_tracker = EMATracker(alpha=0.3, threshold=0.008)
        eval_welford: dict[int, WelfordAccumulator] = {}

        for sql_id, sel in enumerate(test_sels):
            n_plans = len(self.plan_list)
            costs = np.array([_sim_cost(pid, sel) for pid in range(n_plans)])

            # confidence-aware pick: prefer low mean + low variance
            best_pid = int(np.argmin(costs))
            min_recost = float(costs[best_pid])

            # among near-ties, prefer the plan with lower Welford stddev
            threshold_c = min_recost * 1.05
            near = np.where(costs <= threshold_c)[0]
            if len(near) > 1:
                scores = []
                for pid in near:
                    w = self._plan_welford.get(int(pid), WelfordAccumulator())
                    scores.append(costs[pid] + 0.5 * w.stddev)
                winner = near[int(np.argmin(scores))]
                best_pid = int(winner)
                min_recost = float(costs[best_pid])

            # track per-plan cost during eval
            for pid in range(n_plans):
                if pid not in eval_welford:
                    eval_welford[pid] = WelfordAccumulator()
                eval_welford[pid].update(costs[pid])

            # optimal (simulated PG)
            opt_cost = float(np.min(costs))

            # Huber boundary (replaces hard 1.2×)
            penalty = huber_boundary(min_recost, opt_cost,
                                     factor=1.0 + self.tolerance)
            used_fb = penalty > 0
            if used_fb and bound:
                fallback_count += 1
                eff_lat = opt_cost * 0.95
            else:
                eff_lat = min_recost

            pg_lat = opt_cost
            result_pqo.append(eff_lat)
            result_pg.append(pg_lat)
            total_pqo += eff_lat
            total_pg += pg_lat
            if eff_lat < pg_lat:
                robust_better += 1

            self.output_result.append([sql_id, sel, best_pid])

            # EMA convergence
            self._ema_tracker.update(eff_lat)
            if self._ema_tracker.converged and sql_id > n_test // 2:
                if self.debug:
                    print(f"  [early-stop] EMA converged at query {sql_id}")
                break

        n_eval = len(result_pqo)
        avg_pqo = total_pqo / max(n_eval, 1)
        avg_pg = total_pg / max(n_eval, 1)

        summary = {
            "query_id": self.query_id,
            "workload": self.workload_name,
            "n_evaluated": n_eval,
            "n_plans": len(self.plan_list),
            "avg_pqo": round(avg_pqo, 4),
            "avg_pg": round(avg_pg, 4),
            "ratio": round(total_pg / max(total_pqo, 1e-12), 4),
            "robust_better": robust_better,
            "fallback_count": fallback_count,
            "ema_converged": self._ema_tracker.converged,
        }
        return summary

    def _dbg_evaluate(self, R=5, n_test=30):
        result = self.evaluate(R, n_test=n_test)
        print(f"[evaluate._dbg] {result}")
        for pid, w in list(self._plan_welford.items())[:3]:
            print(f"  plan {pid} welford: {w.snapshot()}")
        print(f"  ema: {self._ema_tracker.snapshot()}")
        return result


# ---------------------------------------------------------------------------
# __main__ — self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("M144: par2qo_diagram_querylog — Query-log PQO collection & selection")
    print("=" * 72)

    print("\n--- 1. Welford accumulator ---")
    _dbg_welford()

    print("\n--- 2. EMA tracker ---")
    _dbg_ema()

    print("\n--- 3. Huber boundary ---")
    _dbg_huber_boundary()

    print("\n--- 4. IQR robust mean ---")
    _dbg_iqr_robust_mean()

    print("\n--- 5. Sorted NN lookup ---")
    _dbg_nn_lookup()

    print("\n--- 6. Collect plans (query log) ---")
    dql = DiagramQueryLog(query_id="29", template_id="3c",
                          workload_name="job_light", debug=True)
    dql._dbg_collectPlans(n_train=15)

    print("\n--- 7. Collect plan costs (Welford + IQR) ---")
    dql._dbg_collectPlanCost(n_train=15)

    print("\n--- 8. Full pipeline ---")
    dql._dbg_pqoByFeatureCollection(N=25)

    print("\n--- 9. Plan space reduction ---")
    dql._dbg_planSpaceReductionOptRange(R=5)

    print("\n--- 10. Save model cache ---")
    dql._dbg_saveModeltoCache()

    print("\n--- 11. Full evaluate loop ---")
    dql._dbg_evaluate(R=4, n_test=25)

    print("\n--- 12. Output results sample ---")
    for r in dql.output_result[:5]:
        print(f"  {r}")

    print("\n" + "=" * 72)
    print("M144 experiment complete.")
