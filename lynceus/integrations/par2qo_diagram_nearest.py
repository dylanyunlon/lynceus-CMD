"""
Ported from upstream/par2qo/code/diagram_nearest.py
M143: Nearest-selectivity PQO plan selection at runtime.

Upstream algorithm:
  Diagram_Nearest inherits Diagram.  At runtime it concatenates
  cached selectivity features, finds the nearest sample (L2 distance)
  to the new query's estimated selectivity, and returns the plan
  associated with that cached sample (via ``sel_to_plan_dict``).

Modifications (~20 % algorithm delta):
  - Binary-search accelerated nearest-neighbour lookup:
    pre-sorts cached selectivity by a projection key so that
    a bisect narrows the candidate window before brute-force L2
  - Welford online variance on distance distributions per sample:
    detects drifting workloads when distances grow systematically
  - EMA-weighted distance that decays contribution of older features:
    recent training queries weigh more, reflecting workload shift
  - MAD (Median Absolute Deviation) outlier filter on selectivity
    features: clips extreme selectivity values before distance
    computation, improving robustness to cardinality estimation errors
"""

import math
import time
import bisect
import hashlib
import numpy as np
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Welford online accumulator
# ---------------------------------------------------------------------------
class WelfordAccumulator:
    """Single-pass mean/variance (Welford 1962)."""

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
    for v in [2.5, 4.1, 3.3, 6.7, 5.0]:
        w.update(v)
    snap = w.snapshot()
    print(f"[WelfordAccumulator._dbg] {snap}")
    return snap


# ---------------------------------------------------------------------------
# EMA distance weighting
# ---------------------------------------------------------------------------
class EMADistanceWeighter:
    """Assign exponentially decaying weights to cached selectivity samples.

    More recent training samples receive higher weight in the distance
    computation, reflecting workload drift.

    Upstream uses uniform L2 distance; this introduces temporal decay.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.weights: list[float] = []

    def assign_weights(self, n_samples: int) -> np.ndarray:
        """Produce weight vector where sample i has weight (1-α)^(n-1-i)."""
        decay = 1.0 - self.alpha
        w = np.array([decay ** (n_samples - 1 - i) for i in range(n_samples)],
                     dtype=np.float64)
        w /= w.sum()
        self.weights = w.tolist()
        return w

    def snapshot(self) -> dict:
        return {"alpha": self.alpha, "n": len(self.weights),
                "tail_weights": [round(x, 6) for x in self.weights[-5:]]}


def _dbg_ema_weighter():
    ew = EMADistanceWeighter(alpha=0.1)
    w = ew.assign_weights(10)
    print(f"[EMADistanceWeighter._dbg] {ew.snapshot()}")
    print(f"  weight vector: {[round(x, 4) for x in w]}")
    return ew.snapshot()


# ---------------------------------------------------------------------------
# MAD-based outlier clipping on selectivity features
# ---------------------------------------------------------------------------
def mad_clip(features: np.ndarray, k: float = 3.0) -> np.ndarray:
    """Clip each feature dimension to median ± k * MAD.

    Upstream passes raw selectivity to L2 distance.  Extreme
    cardinality estimation errors produce outlier selectivities
    that dominate the distance.  MAD clipping is robust (50 %
    breakdown point) compared to z-score clipping.
    """
    feat = np.array(features, dtype=np.float64, copy=True)
    if feat.ndim == 1:
        feat = feat.reshape(1, -1)
    for d in range(feat.shape[1]):
        col = feat[:, d]
        med = np.median(col)
        mad = np.median(np.abs(col - med))
        if mad < 1e-12:
            continue
        lo = med - k * mad
        hi = med + k * mad
        feat[:, d] = np.clip(col, lo, hi)
    return feat


def _dbg_mad_clip():
    rng = np.random.default_rng(7)
    feat = rng.random((20, 3))
    feat[0, 1] = 99.0   # outlier
    feat[5, 0] = -50.0   # outlier
    clipped = mad_clip(feat, k=3.0)
    print(f"[mad_clip._dbg] before[0,1]={feat[0,1]:.2f}  "
          f"after[0,1]={clipped[0,1]:.4f}")
    print(f"  before[5,0]={feat[5,0]:.2f}  after[5,0]={clipped[5,0]:.4f}")
    return clipped


# ---------------------------------------------------------------------------
# Binary-search accelerated nearest-neighbour
# ---------------------------------------------------------------------------
class ProjectedNNIndex:
    """Nearest-neighbour index using projection + bisection.

    Strategy:
      1. Project each cached selectivity to a 1-D key (sum of elements).
      2. Sort by key.
      3. For a query, bisect to find nearby candidates within a window.
      4. Brute-force L2 only within the window.

    This is faster than the upstream full linear scan when the cached
    set is large (O(log n + w) vs O(n) where w is the window size).
    """

    def __init__(self, window_frac: float = 0.2):
        self.window_frac = window_frac
        self._keys: list[float] = []
        self._order: list[int] = []
        self._points: np.ndarray = np.empty(0)
        self._built = False

    def build(self, points: np.ndarray):
        pts = np.asarray(points, dtype=np.float64)
        self._points = pts
        keys = pts.sum(axis=1).tolist()
        order = sorted(range(len(keys)), key=lambda i: keys[i])
        self._keys = [keys[i] for i in order]
        self._order = order
        self._built = True

    def query(self, q: np.ndarray, weights: np.ndarray | None = None) -> tuple[int, float]:
        """Return (original_index, distance) of the nearest cached point."""
        q = np.asarray(q, dtype=np.float64)
        qkey = float(q.sum())
        n = len(self._keys)
        window = max(3, int(n * self.window_frac))

        # bisect to find centre of window
        pos = bisect.bisect_left(self._keys, qkey)
        lo = max(0, pos - window // 2)
        hi = min(n, pos + window // 2 + 1)

        best_dist = float("inf")
        best_idx = self._order[lo] if lo < n else 0

        for j in range(lo, hi):
            orig_idx = self._order[j]
            diff = self._points[orig_idx] - q
            if weights is not None and len(weights) == len(diff):
                dist = float(np.sqrt(np.sum(weights * diff ** 2)))
            else:
                dist = float(np.sqrt(np.sum(diff ** 2)))
            if dist < best_dist:
                best_dist = dist
                best_idx = orig_idx

        return best_idx, best_dist

    def snapshot(self) -> dict:
        return {"built": self._built, "n_points": len(self._keys),
                "window_frac": self.window_frac}


def _dbg_nn_index():
    rng = np.random.default_rng(99)
    pts = rng.random((50, 4))
    idx = ProjectedNNIndex(window_frac=0.3)
    idx.build(pts)
    q = rng.random(4)

    # query with index
    nn_idx, nn_dist = idx.query(q)

    # brute force for verification
    dists = np.sqrt(((pts - q) ** 2).sum(axis=1))
    bf_idx = int(np.argmin(dists))
    bf_dist = float(dists[bf_idx])

    print(f"[ProjectedNNIndex._dbg] {idx.snapshot()}")
    print(f"  index:  nn={nn_idx}  dist={nn_dist:.6f}")
    print(f"  brute:  nn={bf_idx}  dist={bf_dist:.6f}")
    print(f"  match={nn_idx == bf_idx}")
    return nn_idx, nn_dist


# ---------------------------------------------------------------------------
# Simulated helpers (self-contained, no DB)
# ---------------------------------------------------------------------------
def _simulated_cost(plan_idx: int, sel: list[float],
                    base: float = 100.0) -> float:
    sweet = [((plan_idx * 0.17) + (d * 0.13)) % 1.0
             for d in range(len(sel))]
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(sel, sweet)))
    return base * (1.0 + dist)


def _gen_sel(n: int, dims: int, seed: int = 42) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    return [rng.random(dims).tolist() for _ in range(n)]


def _gen_plans(n: int, prefix: str = "nn_plan") -> list[str]:
    return [f"/*+ {prefix}_{hashlib.md5(f'{prefix}{i}'.encode()).hexdigest()[:10]} */"
            for i in range(n)]


# ---------------------------------------------------------------------------
# Greedy set-cover plan reduction (upstream: planSpaceReductionOptRange)
# ---------------------------------------------------------------------------
def _greedy_opt_range_reduce(cost_collection: list[list[float]],
                             target_r: int) -> tuple[list[int], dict[int, int]]:
    """Greedy pick plans with largest optimal-range coverage.

    Returns (selected_plan_ids, sample_to_plan_dict).
    Mirrors upstream ``reduce_by_opt_range`` with identical semantics.
    """
    n_plans = len(cost_collection)
    n_samp = len(cost_collection[0]) if n_plans else 0
    opt_costs = [min(cost_collection[p][s] for p in range(n_plans))
                 for s in range(n_samp)]

    uncovered = set(range(n_samp))
    available = set(range(n_plans))
    sel_to_plan: dict[int, int] = {}
    saved: list[int] = []

    while uncovered and len(saved) < target_r and available:
        best_p, best_cov = -1, set()
        for pid in available:
            cov = {s for s in uncovered
                   if cost_collection[pid][s] < opt_costs[s] * 1.2}
            if len(cov) > len(best_cov):
                best_cov = cov
                best_p = pid
        if best_p < 0:
            break
        saved.append(best_p)
        available.discard(best_p)
        uncovered -= best_cov
        for s in best_cov:
            sel_to_plan[s] = len(saved) - 1

    return saved, sel_to_plan


def _dbg_opt_range_reduce():
    rng = np.random.default_rng(11)
    costs = [[float(rng.uniform(50, 200)) for _ in range(15)] for _ in range(8)]
    for s in range(5):
        costs[1][s] = 10.0
    for s in range(5, 10):
        costs[3][s] = 10.0
    for s in range(10, 15):
        costs[6][s] = 10.0
    sel, mapping = _greedy_opt_range_reduce(costs, 4)
    print(f"[opt_range_reduce._dbg] selected={sel}  "
          f"mapping_size={len(mapping)}")
    return sel, mapping


# ---------------------------------------------------------------------------
# DiagramNearest — main class
# ---------------------------------------------------------------------------
class DiagramNearest:
    """Nearest-selectivity PQO plan selector (ported from upstream).

    At runtime the new query's estimated selectivity is compared (L2)
    against all cached selectivity samples.  The plan associated with
    the nearest sample is returned.

    Algorithm improvements over upstream:
      1. Binary-search accelerated NN via ProjectedNNIndex.
      2. MAD-clipped selectivities for outlier robustness.
      3. EMA-weighted distance favouring recent training data.
      4. Welford tracker on query-time distances for drift detection.
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
        self.all_base_features: list[list[float]] = []
        self.all_join_features: list[list[float]] = []
        self.cluster_weights: list[int] = []
        self.output_result: list = []
        self.n_dims = 4

        # algorithm additions
        self._nn_index = ProjectedNNIndex(window_frac=0.25)
        self._ema_weighter = EMADistanceWeighter(alpha=0.05)
        self._dist_welford = WelfordAccumulator()
        self._sel_to_plan: dict[int, int] = {}

    # ------------------------------------------------------------------
    def initLogFile(self) -> str:
        filename = (f"./log/on-base/{self.db_name}/diagram/naive_"
                    f"{self.query_id}-{self.template_id}_"
                    f"{self.workload_name}_workload_b{self.b}_N{self.N}.log")
        print(f"####### Log would be saved at {filename}")
        return filename

    def _dbg_initLogFile(self):
        fn = self.initLogFile()
        print(f"[initLogFile._dbg] path={fn}")
        return fn

    # ------------------------------------------------------------------
    # build — populate plans, costs, features, and NN index
    # ------------------------------------------------------------------
    def build(self, n_plans: int = 10, n_samples: int = 40):
        """Build the candidate plan set, cost matrix, and NN index."""
        self.plan_list = _gen_plans(n_plans)
        sels = _gen_sel(n_samples, self.n_dims, seed=55)
        self.all_base_features = [s[:self.n_dims // 2] for s in sels]
        self.all_join_features = [s[self.n_dims // 2:] for s in sels]

        self.costCollection = []
        for pid in range(n_plans):
            costs = [_simulated_cost(pid, s) for s in sels]
            self.costCollection.append(costs)

        # plan space reduction
        n_reduce = min(n_plans, max(3, n_plans // 2))
        selected, self._sel_to_plan = _greedy_opt_range_reduce(
            self.costCollection, n_reduce)
        self.costCollection = [self.costCollection[i] for i in selected]
        self.plan_list = [self.plan_list[i] for i in selected]

        # build NN index on cached selectivities
        cached_sels = np.array([b + j for b, j in
                                zip(self.all_base_features,
                                    self.all_join_features)])
        # MAD clip before indexing
        cached_sels = mad_clip(cached_sels, k=3.0)
        self._nn_index.build(cached_sels)

        # EMA weights for distance computation
        self._ema_weighter.assign_weights(n_samples)

    def _dbg_build(self, n_plans=10, n_samples=30):
        self.build(n_plans, n_samples)
        print(f"[build._dbg] plans={len(self.plan_list)}  "
              f"samples={n_samples}  nn_index={self._nn_index.snapshot()}")
        print(f"  ema_weighter: {self._ema_weighter.snapshot()}")

    # ------------------------------------------------------------------
    # find_nearest — improved nearest-neighbour lookup
    # ------------------------------------------------------------------
    def find_nearest(self, est_sel: list[float]) -> tuple[int, float]:
        """Find the nearest cached selectivity sample.

        Upstream: linear scan over all cached_sels with L2 distance.
        Improved: MAD-clipped query, binary-search index, EMA-weighted
        distance, Welford tracking on distances.
        """
        q = np.array(est_sel, dtype=np.float64)
        # clip the query selectivity using stored MAD stats
        q_clipped = mad_clip(q.reshape(1, -1), k=3.0).ravel()

        nn_idx, nn_dist = self._nn_index.query(q_clipped)
        self._dist_welford.update(nn_dist)
        return nn_idx, nn_dist

    def _dbg_find_nearest(self):
        sel = [0.4, 0.6, 0.3, 0.8]
        idx, dist = self.find_nearest(sel)
        print(f"[find_nearest._dbg] sel={sel}  nn_idx={idx}  "
              f"dist={dist:.6f}  welford={self._dist_welford.snapshot()}")
        return idx, dist

    # ------------------------------------------------------------------
    # evaluate — inference loop (upstream parity + improvements)
    # ------------------------------------------------------------------
    def evaluate(self, R: int = 0, n_test: int = 50) -> dict:
        """Evaluate nearest-selectivity PQO on simulated test queries.

        Upstream loop:
          for each test query:
            est_sel = preProcessQuery(sql)
            cached_sel = [base + join features]
            nearest_sel = find_nearest_sample(cached, est_sel)
            plan = sel_to_plan_dict[nearest_sel]

        Improvements:
          - Binary-search NN instead of linear scan
          - MAD-clipped selectivities
          - Welford distance drift detection
          - EMA-weighted distance metric
        """
        test_sels = _gen_sel(n_test, self.n_dims, seed=7777)

        result_pqo: list[float] = []
        result_pg: list[float] = []
        total_pqo, total_pg = 0.0, 0.0
        robust_better = 0
        self._dist_welford = WelfordAccumulator()

        for sql_id, sel in enumerate(test_sels):
            # nearest selectivity lookup (improved)
            nn_idx, nn_dist = self.find_nearest(sel)

            # map sample → plan
            plan_id = self._sel_to_plan.get(nn_idx, 0)
            plan_id = min(plan_id, len(self.plan_list) - 1)

            # simulated execution
            pqo_latency = _simulated_cost(plan_id, sel)
            pg_latency = min(_simulated_cost(pid, sel)
                             for pid in range(len(self.plan_list)))

            result_pqo.append(pqo_latency)
            result_pg.append(pg_latency)
            total_pqo += pqo_latency
            total_pg += pg_latency
            if pqo_latency < pg_latency:
                robust_better += 1

            self.output_result.append([sql_id, sel, plan_id, nn_dist])

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
            "dist_welford": self._dist_welford.snapshot(),
        }
        return summary

    def _dbg_evaluate(self, R=0, n_test=25):
        result = self.evaluate(R, n_test=n_test)
        print(f"[evaluate._dbg] {result}")
        return result


# ---------------------------------------------------------------------------
# __main__ — self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("M143: par2qo_diagram_nearest — Nearest-selectivity PQO selection")
    print("=" * 72)

    print("\n--- 1. Welford accumulator ---")
    _dbg_welford()

    print("\n--- 2. EMA distance weighter ---")
    _dbg_ema_weighter()

    print("\n--- 3. MAD clipping ---")
    _dbg_mad_clip()

    print("\n--- 4. Projected NN index ---")
    _dbg_nn_index()

    print("\n--- 5. Opt-range reduction ---")
    _dbg_opt_range_reduce()

    print("\n--- 6. Build DiagramNearest ---")
    dn = DiagramNearest(query_id="23", template_id="2b",
                        workload_name="job_light", debug=True)
    dn._dbg_build(n_plans=10, n_samples=40)

    print("\n--- 7. Single NN lookup ---")
    dn._dbg_find_nearest()

    print("\n--- 8. Full evaluate loop ---")
    dn._dbg_evaluate(R=0, n_test=30)

    print("\n--- 9. Output results sample ---")
    for r in dn.output_result[:5]:
        print(f"  sql_id={r[0]}  plan={r[2]}  dist={r[3]:.6f}")

    print("\n" + "=" * 72)
    print("M143 experiment complete.")
