"""
par2qo_evaluate_pqo.py
----------------------
Port of upstream/par2qo/code/carver/4_evaluate_PQO.py

Evaluates the *non-parametric* PQO model in isolation.  The original script
shell-invoked ``kepler.training_data_collection_pipeline.evaluate_pqo``; this
port replaces all DB / file-system / subprocess calls with in-memory dict
simulation and uses only numpy (no sklearn / TF).

Numerical improvements
~~~~~~~~~~~~~~~~~~~~~~
* Welford online variance for cost-ratio streams.
* Kahan compensated summation for harmonic-mean reciprocal accumulation.
* Shannon entropy over plan-selection frequency buckets.

Metrics
~~~~~~~
* Harmonic mean of cost ratios (predicted / optimal) — more sensitive to
  outliers than the arithmetic mean, appropriate for ratio data.
* Mean / std / max plan-cost ratio (PQO metric).
* Per-training-size breakdown.

Public API
~~~~~~~~~~
    run_pqo_evaluation(config) -> PQOEvaluationReport
    _debug_snapshot(state)     -> None
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PQOEvalConfig:
    """Mirrors CLI parameters of the original evaluate_pqo script."""
    query_ids: List[str] = field(default_factory=lambda: ["q7-t0"])
    methods: List[str] = field(default_factory=lambda: ["kepler"])
    training_sizes: List[int] = field(default_factory=lambda: [50, 400])
    n_test_params: int = 40
    rng_seed: int = 13


# ─────────────────────────────────────────────────────────────────────────────
# Query-ID helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kepler_sql_features_id(pqo_qid: str) -> str:
    """Convert PQO style 'q7-t0' -> kepler style '7-0'."""
    # pqo_qid: q{A}-t{B}
    parts = pqo_qid.split("-")
    a = parts[0][1:]   # strip leading 'q'
    b = parts[1][1:]   # strip leading 't'
    return f"{a}-{b}"


# ─────────────────────────────────────────────────────────────────────────────
# In-memory data simulation
# ─────────────────────────────────────────────────────────────────────────────

def _build_mock_db(cfg: PQOEvalConfig) -> Dict[str, Any]:
    """
    Simulate the on-disk PQO result CSV and reference cost data.

    db["pqo_costs"][(qid, method, ts)]    -> ndarray (n_test,)  predicted costs
    db["optimal_costs"][(qid, method)]    -> ndarray (n_test,)  ground-truth optimal
    db["plan_selections"][(qid, method, ts)] -> ndarray[int] shape (n_test,) plan indices
    """
    rng = np.random.default_rng(cfg.rng_seed)
    db: Dict[str, Any] = {
        "pqo_costs": {},
        "optimal_costs": {},
        "plan_selections": {},
    }

    for qid in cfg.query_ids:
        for method in cfg.methods:
            n = cfg.n_test_params
            optimal = rng.uniform(30.0, 800.0, size=n)
            db["optimal_costs"][(qid, method)] = optimal

            for ts in cfg.training_sizes:
                # PQO costs: worse with smaller training size
                noise_scale = 1.0 + 200.0 / (ts + 1)
                db["pqo_costs"][(qid, method, ts)] = (
                    optimal * rng.uniform(1.0, noise_scale, size=n)
                )
                # Simulate discrete plan selection (0-based plan index)
                n_plans = max(3, ts // 20)
                db["plan_selections"][(qid, method, ts)] = rng.integers(
                    0, n_plans, size=n
                )

    return db


# ─────────────────────────────────────────────────────────────────────────────
# Welford online variance
# ─────────────────────────────────────────────────────────────────────────────

class _WelfordAcc:
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self._M2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self._M2 += delta * (x - self.mean)

    @property
    def variance(self) -> float:
        return self._M2 / self.n if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


# ─────────────────────────────────────────────────────────────────────────────
# Kahan compensated sum
# ─────────────────────────────────────────────────────────────────────────────

def _kahan_sum(values: List[float]) -> float:
    total = 0.0
    c = 0.0
    for v in values:
        y = v - c
        t = total + y
        c = (t - total) - y
        total = t
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Shannon entropy over plan-selection frequencies
# ─────────────────────────────────────────────────────────────────────────────

def _selection_entropy(plan_indices: np.ndarray) -> float:
    """Shannon entropy of the discrete plan-selection distribution."""
    if len(plan_indices) == 0:
        return 0.0
    values, counts = np.unique(plan_indices, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


# ─────────────────────────────────────────────────────────────────────────────
# Harmonic mean of cost ratios
# ─────────────────────────────────────────────────────────────────────────────

def _harmonic_mean_ratio(predicted: np.ndarray, optimal: np.ndarray) -> float:
    """
    Harmonic mean of (predicted / optimal) ratios.
    HM = n / Σ(1 / ratio_i)
    Uses Kahan summation for the reciprocal accumulation.
    """
    ratios = predicted / (optimal + 1e-9)
    reciprocals = (1.0 / (ratios + 1e-9)).tolist()
    sum_recip = _kahan_sum(reciprocals)
    return len(ratios) / sum_recip if sum_recip > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PQOResult:
    query_id: str
    method: str
    training_size: int
    mean_cost_ratio: float          # arithmetic mean of predicted/optimal
    harmonic_mean_ratio: float      # harmonic mean (PQO metric)
    std_cost_ratio: float
    max_cost_ratio: float
    plan_selection_entropy: float   # Shannon entropy of selected plans
    n_params: int


@dataclass
class PQOEvaluationReport:
    results: List[PQOResult] = field(default_factory=list)
    global_harmonic_mean: float = 0.0
    best_training_size: int = 0
    summary: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Debug snapshot
# ─────────────────────────────────────────────────────────────────────────────

def _debug_snapshot(state: Dict[str, Any]) -> None:
    """Pretty-print internal PQO evaluation state for debugging."""
    print("=" * 60)
    print("[_debug_snapshot] par2qo_evaluate_pqo internal state")
    print("=" * 60)
    for k, v in state.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: ndarray shape={v.shape} dtype={v.dtype} "
                  f"min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f}")
        elif isinstance(v, list):
            print(f"  {k}: list len={len(v)}")
        elif isinstance(v, dict):
            print(f"  {k}: dict keys(first 6)={list(v.keys())[:6]}")
        else:
            print(f"  {k}: {v!r}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_pqo_single(
    qid: str,
    method: str,
    ts: int,
    db: Dict[str, Any],
) -> PQOResult:
    predicted = db["pqo_costs"][(qid, method, ts)]
    optimal = db["optimal_costs"][(qid, method)]
    selections = db["plan_selections"][(qid, method, ts)]

    ratios = predicted / (optimal + 1e-9)

    acc = _WelfordAcc()
    for r in ratios:
        acc.update(float(r))

    hmr = _harmonic_mean_ratio(predicted, optimal)
    entropy = _selection_entropy(selections)

    return PQOResult(
        query_id=qid,
        method=method,
        training_size=ts,
        mean_cost_ratio=acc.mean,
        harmonic_mean_ratio=hmr,
        std_cost_ratio=acc.std,
        max_cost_ratio=float(ratios.max()),
        plan_selection_entropy=entropy,
        n_params=len(predicted),
    )


def run_pqo_evaluation(
    cfg: Optional[PQOEvalConfig] = None,
) -> PQOEvaluationReport:
    """
    Evaluate the PQO model across all (query, method, training_size) combos.

    Mirrors the original nested loop:
        for method in methods:
            for query_id in query_ids:
                for training_size in training_sizes:
                    <evaluate pqo>
    """
    if cfg is None:
        cfg = PQOEvalConfig()

    db = _build_mock_db(cfg)

    _debug_snapshot({
        "query_ids": cfg.query_ids,
        "methods": cfg.methods,
        "training_sizes": cfg.training_sizes,
        "n_test_params": cfg.n_test_params,
        "sample_optimal": db["optimal_costs"].get(
            (cfg.query_ids[0], cfg.methods[0])
        ),
        "sample_pqo_costs": db["pqo_costs"].get(
            (cfg.query_ids[0], cfg.methods[0], cfg.training_sizes[0])
        ),
    })

    report = PQOEvaluationReport()
    global_acc = _WelfordAcc()

    for method in cfg.methods:
        for qid in cfg.query_ids:
            for ts in cfg.training_sizes:
                res = _evaluate_pqo_single(qid, method, ts, db)
                report.results.append(res)
                global_acc.update(res.harmonic_mean_ratio)

    report.global_harmonic_mean = global_acc.mean

    # best training size: lowest harmonic mean ratio
    ts_scores: Dict[int, List[float]] = {ts: [] for ts in cfg.training_sizes}
    for r in report.results:
        ts_scores[r.training_size].append(r.harmonic_mean_ratio)
    report.best_training_size = min(
        cfg.training_sizes,
        key=lambda ts: (
            sum(ts_scores[ts]) / len(ts_scores[ts]) if ts_scores[ts] else float("inf")
        ),
    )

    report.summary = {
        "total_evaluations": len(report.results),
        "global_harmonic_mean_ratio": report.global_harmonic_mean,
        "best_training_size": report.best_training_size,
        "query_ids": cfg.query_ids,
        "methods": cfg.methods,
    }

    _debug_snapshot({
        "total_results": len(report.results),
        "global_harmonic_mean": report.global_harmonic_mean,
        "best_training_size": report.best_training_size,
    })

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    try:
        cfg = PQOEvalConfig(
            query_ids=["q7-t0"],
            methods=["kepler"],
            training_sizes=[50, 400],
            n_test_params=40,
            rng_seed=13,
        )
        report = run_pqo_evaluation(cfg)

        expected = len(cfg.methods) * len(cfg.query_ids) * len(cfg.training_sizes)
        assert len(report.results) == expected, \
            f"Expected {expected} results, got {len(report.results)}"

        for r in report.results:
            assert r.mean_cost_ratio >= 1.0, \
                f"Cost ratio should be ≥1 (predicted ≥ optimal): {r.mean_cost_ratio}"
            assert r.harmonic_mean_ratio > 0.0
            assert r.std_cost_ratio >= 0.0
            assert r.plan_selection_entropy >= 0.0

        assert report.best_training_size in cfg.training_sizes

        # Harmonic mean of [1,1,1,...] = 1
        ones = np.ones(20)
        hm = _harmonic_mean_ratio(ones, ones)
        assert abs(hm - 1.0) < 1e-6, f"Harmonic mean of ratios=1 should be 1, got {hm}"

        # Harmonic mean < arithmetic mean for varied positive data
        rng = np.random.default_rng(99)
        a = rng.uniform(1.0, 3.0, 100)
        b = np.ones(100)
        hm2 = _harmonic_mean_ratio(a, b)
        am2 = float(a.mean())
        assert hm2 <= am2 + 1e-9, "Harmonic mean should not exceed arithmetic mean"

        # Query ID conversion
        assert _kepler_sql_features_id("q7-t0") == "7-0"
        assert _kepler_sql_features_id("q16-t0") == "16-0"

        # Welford stability
        acc = _WelfordAcc()
        for _ in range(200):
            acc.update(5.0)
        assert abs(acc.variance) < 1e-10

        # Kahan sum
        s = _kahan_sum([0.1] * 1000)
        assert abs(s - 100.0) < 1e-9, f"Kahan sum={s}"

        # Shannon entropy: uniform distribution should have maximal entropy
        uniform_plans = np.arange(10).repeat(10)  # 10 plans, 10 times each
        e1 = _selection_entropy(uniform_plans)
        skewed_plans = np.zeros(100, dtype=int)    # always plan 0
        e2 = _selection_entropy(skewed_plans)
        assert e1 > e2, "Uniform selection should have higher entropy than skewed"

        print("PASS")
        sys.exit(0)
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
