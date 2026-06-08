"""
par2qo_evaluate.py
------------------
Port of upstream/par2qo/code/carver/4_evaluate.py

Evaluates plan-cost predictions against ground-truth optimal costs for
parametric query optimization (PAR2QO).  The original script shell-invoked
``kepler.training_data_collection_pipeline.evaluate``; this port replaces
every DB / file-system / subprocess call with in-memory dict simulation and
uses only numpy (no sklearn / TF).

Numerical improvements over upstream
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Welford online variance (numerically stable, single-pass).
* Kahan compensated summation for regret accumulation.
* Shannon entropy over cost-distribution buckets.

Public API
~~~~~~~~~~
    run_evaluation(config)  -> EvaluationReport
    _debug_snapshot(state)  -> None   # prints internal data-structure state
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
class EvalConfig:
    """Mirrors the CLI parameters consumed by the original evaluate script."""
    query_ids: List[str] = field(default_factory=lambda: ["16-0"])
    methods: List[str] = field(default_factory=lambda: ["cardinality", "kepler", "csv"])
    training_sizes: List[int] = field(default_factory=lambda: [50])
    confidence_thresholds: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8]
    )
    n_test_params: int = 30          # synthetic test-parameter count
    rng_seed: int = 42


# ─────────────────────────────────────────────────────────────────────────────
# In-memory data simulation  (replaces DB + JSON file I/O)
# ─────────────────────────────────────────────────────────────────────────────

def _build_mock_db(cfg: EvalConfig) -> Dict[str, Any]:
    """
    Returns a nested dict that mimics the on-disk JSON training-param files
    and the CSV prediction outputs produced by the kepler pipeline.

    Structure
    ---------
    db["params"][query_id]          -> list of param-value dicts
    db["predictions"][key]          -> np.ndarray  shape (n_test,)
    db["optimal_costs"][query_id]   -> np.ndarray  shape (n_test,)
    db["default_costs"][query_id]   -> np.ndarray  shape (n_test,)
    """
    rng = np.random.default_rng(cfg.rng_seed)
    db: Dict[str, Any] = {
        "params": {},
        "predictions": {},
        "optimal_costs": {},
        "default_costs": {},
    }

    for qid in cfg.query_ids:
        n_params = cfg.n_test_params
        db["params"][qid] = [{"p": rng.uniform(0, 100)} for _ in range(n_params + 1)]

        optimal = rng.uniform(50.0, 500.0, size=n_params)
        db["optimal_costs"][qid] = optimal
        db["default_costs"][qid] = optimal * rng.uniform(1.0, 3.0, size=n_params)

        for method in cfg.methods:
            for ts in cfg.training_sizes:
                for ct in cfg.confidence_thresholds:
                    key = (qid, method, ts, ct)
                    noise = rng.uniform(0.9, 1.5, size=n_params)
                    db["predictions"][key] = optimal * noise

    return db


# ─────────────────────────────────────────────────────────────────────────────
# Welford online variance accumulator
# ─────────────────────────────────────────────────────────────────────────────

class _WelfordAcc:
    """Numerically stable single-pass mean + variance (Welford 1962)."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self._M2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._M2 += delta * delta2

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
    """Kahan compensated summation to reduce floating-point error."""
    total = 0.0
    c = 0.0
    for v in values:
        y = v - c
        t = total + y
        c = (t - total) - y
        total = t
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Shannon entropy of a cost distribution
# ─────────────────────────────────────────────────────────────────────────────

def _shannon_entropy(costs: np.ndarray, n_bins: int = 20) -> float:
    """Discretise *costs* into histogram bins and compute Shannon entropy."""
    counts, _ = np.histogram(costs, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MethodResult:
    query_id: str
    method: str
    training_size: int
    confidence_threshold: float
    mean_regret: float
    std_regret: float
    max_regret: float
    mean_plan_cost_ratio: float   # predicted / optimal
    cost_entropy: float
    n_params: int


@dataclass
class EvaluationReport:
    results: List[MethodResult] = field(default_factory=list)
    global_mean_regret: float = 0.0
    global_std_regret: float = 0.0
    best_method: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Debug snapshot
# ─────────────────────────────────────────────────────────────────────────────

def _debug_snapshot(state: Dict[str, Any]) -> None:
    """
    Pretty-print the internal evaluation state.  Call at any stage to
    inspect data-structure contents without a debugger.
    """
    print("=" * 60)
    print("[_debug_snapshot] par2qo_evaluate internal state")
    print("=" * 60)
    for k, v in state.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: ndarray shape={v.shape} dtype={v.dtype} "
                  f"min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f}")
        elif isinstance(v, list):
            print(f"  {k}: list len={len(v)} first={v[0] if v else 'empty'}")
        elif isinstance(v, dict):
            print(f"  {k}: dict keys={list(v.keys())[:6]}")
        else:
            print(f"  {k}: {v!r}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation logic
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_single(
    qid: str,
    method: str,
    training_size: int,
    confidence_threshold: float,
    db: Dict[str, Any],
) -> MethodResult:
    """
    Compute regret and plan-cost metrics for one (query, method, ts, ct) combo.

    Regret is defined as  (predicted_cost - optimal_cost) / optimal_cost,
    clipped to [0, ∞) so we never reward over-estimation.
    """
    key = (qid, method, training_size, confidence_threshold)
    predicted: np.ndarray = db["predictions"][key]
    optimal: np.ndarray = db["optimal_costs"][qid]

    regrets = np.maximum(0.0, (predicted - optimal) / (optimal + 1e-9))

    # Welford accumulator over regret values
    acc = _WelfordAcc()
    regret_list: List[float] = []
    for r in regrets:
        acc.update(float(r))
        regret_list.append(float(r))

    kahan_total = _kahan_sum(regret_list)
    mean_regret_kahan = kahan_total / len(regret_list) if regret_list else 0.0

    plan_cost_ratios = predicted / (optimal + 1e-9)
    entropy = _shannon_entropy(predicted)

    return MethodResult(
        query_id=qid,
        method=method,
        training_size=training_size,
        confidence_threshold=confidence_threshold,
        mean_regret=mean_regret_kahan,
        std_regret=acc.std,
        max_regret=float(regrets.max()) if len(regrets) else 0.0,
        mean_plan_cost_ratio=float(plan_cost_ratios.mean()),
        cost_entropy=entropy,
        n_params=len(predicted),
    )


def run_evaluation(cfg: Optional[EvalConfig] = None) -> EvaluationReport:
    """
    Main entry-point.  Mirrors the nested loop in the original script:

        for query_id in query_ids:
            for method in methods:
                for training_size in training_sizes:
                    for confidence_threshold in confidence_thresholds:
                        <evaluate>

    Returns an EvaluationReport instead of shell-printing commands.
    """
    if cfg is None:
        cfg = EvalConfig()

    db = _build_mock_db(cfg)

    # snapshot of raw DB structure
    _debug_snapshot({
        "query_ids": cfg.query_ids,
        "methods": cfg.methods,
        "training_sizes": cfg.training_sizes,
        "confidence_thresholds": cfg.confidence_thresholds,
        "sample_optimal_costs": db["optimal_costs"][cfg.query_ids[0]],
        "sample_default_costs": db["default_costs"][cfg.query_ids[0]],
    })

    report = EvaluationReport()
    global_acc = _WelfordAcc()

    for qid in cfg.query_ids:
        for method in cfg.methods:
            for ts in cfg.training_sizes:
                for ct in cfg.confidence_thresholds:
                    res = _evaluate_single(qid, method, ts, ct, db)
                    report.results.append(res)
                    global_acc.update(res.mean_regret)

    report.global_mean_regret = global_acc.mean
    report.global_std_regret = global_acc.std

    # best method: lowest mean regret averaged across all configs
    method_regrets: Dict[str, List[float]] = {m: [] for m in cfg.methods}
    for r in report.results:
        method_regrets[r.method].append(r.mean_regret)
    report.best_method = min(
        cfg.methods,
        key=lambda m: (sum(method_regrets[m]) / len(method_regrets[m]))
        if method_regrets[m]
        else float("inf"),
    )

    report.summary = {
        "total_evaluations": len(report.results),
        "global_mean_regret": report.global_mean_regret,
        "global_std_regret": report.global_std_regret,
        "best_method": report.best_method,
    }

    # final snapshot
    _debug_snapshot({
        "total_results": len(report.results),
        "global_mean_regret": report.global_mean_regret,
        "global_std_regret": report.global_std_regret,
        "best_method": report.best_method,
    })

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    try:
        cfg = EvalConfig(
            query_ids=["16-0"],
            methods=["cardinality", "kepler", "csv"],
            training_sizes=[50],
            confidence_thresholds=[0.0, 0.2, 0.4, 0.6, 0.8],
            n_test_params=30,
            rng_seed=0,
        )
        report = run_evaluation(cfg)

        assert len(report.results) == 3 * 1 * 5, \
            f"Expected 15 results, got {len(report.results)}"
        assert report.global_mean_regret >= 0.0, "Regret must be non-negative"
        assert report.best_method in cfg.methods, "best_method must be a known method"

        for r in report.results:
            assert r.mean_regret >= 0.0
            assert r.std_regret >= 0.0
            assert r.mean_plan_cost_ratio >= 0.0
            assert r.cost_entropy >= 0.0

        # Welford stability check: variance of constant array must be 0
        acc = _WelfordAcc()
        for _ in range(1000):
            acc.update(42.0)
        assert abs(acc.variance) < 1e-10, "Welford variance of constant should be ~0"

        # Kahan sum check
        vals = [0.1] * 1000
        k_sum = _kahan_sum(vals)
        assert abs(k_sum - 100.0) < 1e-9, f"Kahan sum error: {k_sum}"

        # Shannon entropy positive check
        costs = np.random.default_rng(1).uniform(1, 100, 50)
        h = _shannon_entropy(costs)
        assert h > 0.0, "Entropy of varied distribution must be positive"

        print("PASS")
        sys.exit(0)
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
