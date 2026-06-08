"""
par2qo_evaluate_both.py
-----------------------
Port of upstream/par2qo/code/carver/4_evaluate_both.py

Joint evaluation of the *parametric* Kepler model and the *non-parametric*
PQO (Parametric Query Optimization) model on the same test workload.
The original script shell-invoked
``kepler.training_data_collection_pipeline.evaluate_both``; this port replaces
all DB / file-system / subprocess calls with in-memory dict simulation and
uses only numpy (no sklearn / TF).

Numerical improvements
~~~~~~~~~~~~~~~~~~~~~~
* Welford online variance for both parametric and PQO cost streams.
* Kahan compensated summation for SMAPE accumulation.
* Shannon entropy over ranked-plan distributions.

Metrics
~~~~~~~
* SMAPE  (Symmetric Mean Absolute Percentage Error)  per model
* NDCG   (Normalized Discounted Cumulative Gain) on cost ranking
* Joint regret (parametric and PQO side by side)

Public API
~~~~~~~~~~
    run_evaluation_both(config)  -> JointEvaluationReport
    _debug_snapshot(state)       -> None
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BothEvalConfig:
    """Parameters mirroring the CLI of the original evaluate_both script."""
    query_ids: List[str] = field(default_factory=lambda: ["16-0", "18-0"])
    methods: List[str] = field(default_factory=lambda: ["cardinality", "kepler", "csv"])
    training_sizes: List[int] = field(default_factory=lambda: [50])
    confidence_thresholds: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8]
    )
    n_test_params: int = 30
    rng_seed: int = 7


# ─────────────────────────────────────────────────────────────────────────────
# In-memory data simulation
# ─────────────────────────────────────────────────────────────────────────────

def _pqo_query_id(kepler_qid: str) -> str:
    """Convert kepler style '16-0' -> PQO style 'q16-t0'."""
    parts = kepler_qid.split("-")
    return f"q{parts[0]}-t{parts[1]}"


def _build_mock_db(cfg: BothEvalConfig) -> Dict[str, Any]:
    """
    In-memory simulation of the dual-source data the original script reads
    from on-disk JSON/CSV files.

    Keys
    ----
    db["kepler_predictions"][(qid, method, ts, ct)]  -> ndarray (n_test,)
    db["pqo_predictions"][(pqo_qid, method, ts)]     -> ndarray (n_test,)
    db["optimal_costs"][qid]                          -> ndarray (n_test,)
    db["params"][qid]                                 -> list of dicts
    """
    rng = np.random.default_rng(cfg.rng_seed)
    db: Dict[str, Any] = {
        "kepler_predictions": {},
        "pqo_predictions": {},
        "optimal_costs": {},
        "params": {},
    }

    for qid in cfg.query_ids:
        n = cfg.n_test_params
        optimal = rng.uniform(40.0, 600.0, size=n)
        db["optimal_costs"][qid] = optimal
        db["params"][qid] = [{"p": rng.uniform(0, 100)} for _ in range(n + 1)]

        pqo_qid = _pqo_query_id(qid)
        for method in cfg.methods:
            for ts in cfg.training_sizes:
                for ct in cfg.confidence_thresholds:
                    # kepler: moderate noise
                    db["kepler_predictions"][(qid, method, ts, ct)] = (
                        optimal * rng.uniform(0.95, 1.8, size=n)
                    )
                # PQO: non-parametric, no confidence threshold axis
                db["pqo_predictions"][(pqo_qid, method, ts)] = (
                    optimal * rng.uniform(1.0, 2.5, size=n)
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
# Shannon entropy
# ─────────────────────────────────────────────────────────────────────────────

def _shannon_entropy(arr: np.ndarray, n_bins: int = 20) -> float:
    counts, _ = np.histogram(arr, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


# ─────────────────────────────────────────────────────────────────────────────
# SMAPE
# ─────────────────────────────────────────────────────────────────────────────

def _smape(predicted: np.ndarray, actual: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error.
    SMAPE = (2/n) * Σ |pred - actual| / (|pred| + |actual|)
    Uses Kahan summation for numerical stability.
    """
    denom = np.abs(predicted) + np.abs(actual) + 1e-9
    terms = 2.0 * np.abs(predicted - actual) / denom
    return _kahan_sum(terms.tolist()) / len(terms)


# ─────────────────────────────────────────────────────────────────────────────
# NDCG on cost ranking
# ─────────────────────────────────────────────────────────────────────────────

def _ndcg_cost_ranking(predicted: np.ndarray, optimal: np.ndarray, k: int = 10) -> float:
    """
    Treat lower cost as higher relevance.
    rel_i = 1 / (rank_in_optimal + 1)  (so rank-1 has rel=1.0)
    DCG uses predicted ranking; IDCG uses optimal ranking.
    """
    n = min(k, len(predicted))
    # relevance = inverse of optimal-cost rank (lower cost = higher relevance)
    opt_ranks = np.argsort(np.argsort(optimal))          # 0-based rank in optimal order
    relevance = 1.0 / (opt_ranks + 1.0)                  # shape (n_test,)

    pred_order = np.argsort(predicted)[:n]               # indices sorted by predicted cost
    ideal_order = np.argsort(optimal)[:n]                # indices sorted by true cost

    dcg = _kahan_sum(
        [relevance[pred_order[i]] / math.log2(i + 2) for i in range(n)]
    )
    idcg = _kahan_sum(
        [relevance[ideal_order[i]] / math.log2(i + 2) for i in range(n)]
    )
    return dcg / idcg if idcg > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SingleModelResult:
    model: str                    # "kepler" or "pqo"
    query_id: str
    method: str
    training_size: int
    confidence_threshold: Optional[float]   # None for PQO
    smape: float
    ndcg: float
    mean_regret: float
    std_regret: float
    cost_entropy: float
    n_params: int


@dataclass
class JointQueryResult:
    query_id: str
    method: str
    training_size: int
    confidence_threshold: float
    kepler: SingleModelResult
    pqo: SingleModelResult
    joint_smape_delta: float      # pqo_smape - kepler_smape (+ means kepler better)
    joint_ndcg_delta: float       # kepler_ndcg - pqo_ndcg


@dataclass
class JointEvaluationReport:
    joint_results: List[JointQueryResult] = field(default_factory=list)
    kepler_global_smape: float = 0.0
    pqo_global_smape: float = 0.0
    kepler_global_ndcg: float = 0.0
    pqo_global_ndcg: float = 0.0
    summary: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Debug snapshot
# ─────────────────────────────────────────────────────────────────────────────

def _debug_snapshot(state: Dict[str, Any]) -> None:
    """Print internal data-structure state for debugging."""
    print("=" * 60)
    print("[_debug_snapshot] par2qo_evaluate_both internal state")
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
# Core evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _eval_model(
    predicted: np.ndarray,
    optimal: np.ndarray,
    model_name: str,
    qid: str,
    method: str,
    ts: int,
    ct: Optional[float],
) -> SingleModelResult:
    regrets = np.maximum(0.0, (predicted - optimal) / (optimal + 1e-9))
    acc = _WelfordAcc()
    for r in regrets:
        acc.update(float(r))

    return SingleModelResult(
        model=model_name,
        query_id=qid,
        method=method,
        training_size=ts,
        confidence_threshold=ct,
        smape=_smape(predicted, optimal),
        ndcg=_ndcg_cost_ranking(predicted, optimal),
        mean_regret=acc.mean,
        std_regret=acc.std,
        cost_entropy=_shannon_entropy(predicted),
        n_params=len(predicted),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation_both(
    cfg: Optional[BothEvalConfig] = None,
) -> JointEvaluationReport:
    """
    Joint parametric (Kepler) + non-parametric (PQO) evaluation.

    Mirrors the original nested loop:
        for query_id in query_ids:
            pqo_query_id = ...
            for method in methods:
                for training_size in training_sizes:
                    for confidence_threshold in confidence_thresholds:
                        <evaluate both models>
    """
    if cfg is None:
        cfg = BothEvalConfig()

    db = _build_mock_db(cfg)

    _debug_snapshot({
        "query_ids": cfg.query_ids,
        "methods": cfg.methods,
        "n_test_params": cfg.n_test_params,
        "sample_optimal_16-0": db["optimal_costs"].get(cfg.query_ids[0]),
        "kepler_pred_count": len(db["kepler_predictions"]),
        "pqo_pred_count": len(db["pqo_predictions"]),
    })

    report = JointEvaluationReport()
    kepler_smape_acc = _WelfordAcc()
    pqo_smape_acc = _WelfordAcc()
    kepler_ndcg_acc = _WelfordAcc()
    pqo_ndcg_acc = _WelfordAcc()

    for qid in cfg.query_ids:
        pqo_qid = _pqo_query_id(qid)
        optimal = db["optimal_costs"][qid]

        for method in cfg.methods:
            for ts in cfg.training_sizes:
                pqo_pred = db["pqo_predictions"][(pqo_qid, method, ts)]
                pqo_res = _eval_model(pqo_pred, optimal, "pqo", qid, method, ts, None)

                for ct in cfg.confidence_thresholds:
                    kepler_pred = db["kepler_predictions"][(qid, method, ts, ct)]
                    kepler_res = _eval_model(
                        kepler_pred, optimal, "kepler", qid, method, ts, ct
                    )

                    jres = JointQueryResult(
                        query_id=qid,
                        method=method,
                        training_size=ts,
                        confidence_threshold=ct,
                        kepler=kepler_res,
                        pqo=pqo_res,
                        joint_smape_delta=pqo_res.smape - kepler_res.smape,
                        joint_ndcg_delta=kepler_res.ndcg - pqo_res.ndcg,
                    )
                    report.joint_results.append(jres)

                    kepler_smape_acc.update(kepler_res.smape)
                    pqo_smape_acc.update(pqo_res.smape)
                    kepler_ndcg_acc.update(kepler_res.ndcg)
                    pqo_ndcg_acc.update(pqo_res.ndcg)

    report.kepler_global_smape = kepler_smape_acc.mean
    report.pqo_global_smape = pqo_smape_acc.mean
    report.kepler_global_ndcg = kepler_ndcg_acc.mean
    report.pqo_global_ndcg = pqo_ndcg_acc.mean

    report.summary = {
        "total_joint_results": len(report.joint_results),
        "kepler_global_smape": report.kepler_global_smape,
        "pqo_global_smape": report.pqo_global_smape,
        "kepler_global_ndcg": report.kepler_global_ndcg,
        "pqo_global_ndcg": report.pqo_global_ndcg,
        "kepler_wins_smape": sum(
            1 for r in report.joint_results if r.joint_smape_delta > 0
        ),
    }

    _debug_snapshot({
        "total_joint_results": len(report.joint_results),
        "kepler_global_smape": report.kepler_global_smape,
        "pqo_global_smape": report.pqo_global_smape,
        "kepler_global_ndcg": report.kepler_global_ndcg,
        "pqo_global_ndcg": report.pqo_global_ndcg,
    })

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    try:
        cfg = BothEvalConfig(
            query_ids=["16-0", "18-0"],
            methods=["cardinality", "kepler", "csv"],
            training_sizes=[50],
            confidence_thresholds=[0.0, 0.2, 0.4, 0.6, 0.8],
            n_test_params=30,
            rng_seed=7,
        )
        report = run_evaluation_both(cfg)

        expected = len(cfg.query_ids) * len(cfg.methods) * len(cfg.training_sizes) * len(cfg.confidence_thresholds)
        assert len(report.joint_results) == expected, \
            f"Expected {expected} joint results, got {len(report.joint_results)}"

        for r in report.joint_results:
            assert 0.0 <= r.kepler.smape, "SMAPE must be non-negative"
            assert 0.0 <= r.kepler.ndcg <= 1.0, f"NDCG out of range: {r.kepler.ndcg}"
            assert 0.0 <= r.pqo.ndcg <= 1.0, f"PQO NDCG out of range: {r.pqo.ndcg}"

        assert report.kepler_global_smape >= 0.0
        assert report.pqo_global_smape >= 0.0

        # SMAPE of identical arrays should be 0
        ones = np.ones(50)
        assert _smape(ones, ones) < 1e-9, "SMAPE of identical arrays must be 0"

        # NDCG of perfect prediction should be 1.0
        opt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ndcg_perfect = _ndcg_cost_ranking(opt, opt, k=5)
        assert abs(ndcg_perfect - 1.0) < 1e-9, f"NDCG perfect={ndcg_perfect}"

        # PQO query id conversion
        assert _pqo_query_id("16-0") == "q16-t0"
        assert _pqo_query_id("18-0") == "q18-t0"

        # Welford stability
        acc = _WelfordAcc()
        for _ in range(500):
            acc.update(3.14)
        assert abs(acc.variance) < 1e-10

        # Kahan compensated sum
        s = _kahan_sum([0.1] * 1000)
        assert abs(s - 100.0) < 1e-9, f"Kahan sum={s}"

        print("PASS")
        sys.exit(0)
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
