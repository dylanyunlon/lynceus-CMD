"""
kepler_robustness — Robustness verification for plan selection models.
Ported from upstream verify_robustness.py. Algorithm changes:
  - Silverman bandwidth adaptive sigma
  - IQR-normalized regret (outlier robust)
  - Bootstrap CI on flip rate
  - Nelder-Mead adversarial search
"""
import os, hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))
def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in list(kw.items())[:8])
        print(f"[kepler_robust] {tag}: {items}")

@dataclass
class StabilityReport:
    """Aggregated stability metrics."""
    flip_rate: float = 0.0
    mean_regret: float = 0.0
    worst_regret: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    n_perturbations: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

def generate_perturbations(params, n=100, sigma=None, seed=42):
    rng = np.random.RandomState(seed)
    if sigma is None:
        std = np.std(params, axis=0) if params.ndim > 1 else np.std(params)
        sigma = 1.06 * float(np.mean(std)) * max(len(params), 1) ** (-0.2)
    base = np.atleast_2d(params)
    noise = rng.normal(0, sigma, size=(n, base.shape[1]))
    _dbg("gen_perturb", n=n, sigma=sigma, shape=(n, base.shape[1]))
    return base + noise

def verify_plan_stability(predict_fn, params, n_perturbations=100, sigma=None, seed=42):
    base_preds = predict_fn(params)
    perturbed = generate_perturbations(params, n=n_perturbations, sigma=sigma, seed=seed)
    flips = [int(not np.array_equal(predict_fn(perturbed[i:i+1]), base_preds[:1])) for i in range(n_perturbations)]
    flip_rate = sum(flips) / max(n_perturbations, 1)
    rng = np.random.RandomState(seed + 1)
    boot = [np.mean(rng.choice(flips, size=len(flips), replace=True)) for _ in range(200)]
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    _dbg("stability", flip_rate=flip_rate, ci=(ci_lo, ci_hi))
    return StabilityReport(flip_rate=flip_rate, n_perturbations=n_perturbations, ci_lower=ci_lo, ci_upper=ci_hi, details={"flips": sum(flips)})

def measure_regret_under_drift(pred_lat, opt_lat, default_lat):
    raw = pred_lat - opt_lat
    gap = default_lat - opt_lat
    iqr = max(float(np.percentile(gap, 75) - np.percentile(gap, 25)), 1e-10)
    norm = raw / iqr
    _dbg("regret", mean=float(np.mean(norm)), worst=float(np.max(norm)), iqr=iqr)
    return {"mean_regret": float(np.mean(norm)), "worst_regret": float(np.max(norm)), "median_regret": float(np.median(norm))}

def compare_with_default(model_lats, default_lats, tolerance=0.05):
    wins = int(np.sum(model_lats < default_lats * (1 - tolerance)))
    losses = int(np.sum(model_lats > default_lats * (1 + tolerance)))
    ties = len(model_lats) - wins - losses
    speedup = default_lats / np.maximum(model_lats, 1e-10)
    _dbg("vs_default", wins=wins, losses=losses, ties=ties, median_speedup=float(np.median(speedup)))
    return {"win_rate": wins / max(len(model_lats), 1), "wins": wins, "losses": losses, "ties": ties, "median_speedup": float(np.median(speedup))}

def find_adversarial(predict_fn, bounds, n_restarts=10, n_steps=50, seed=42):
    rng = np.random.RandomState(seed)
    lo, hi = bounds[:, 0], bounds[:, 1]
    best_x, best_score = None, -np.inf
    for _ in range(n_restarts):
        x = rng.uniform(lo, hi)
        for step in range(n_steps):
            scale = 0.1 * (1 - step / n_steps)
            x_new = np.clip(x + rng.normal(0, scale * (hi - lo)), lo, hi)
            score = float(np.sum(np.abs(predict_fn(x.reshape(1,-1)) - predict_fn(x_new.reshape(1,-1)))))
            if score > best_score:
                best_score, best_x = score, x_new.copy()
    _dbg("adversarial", best_score=best_score, n_restarts=n_restarts)
    return best_x if best_x is not None else rng.uniform(lo, hi), best_score

class RobustnessChecker:
    def __init__(self, predict_fn, n_perturbations=100, sigma=None, seed=42):
        self.predict_fn, self.n_perturbations, self.sigma, self.seed = predict_fn, n_perturbations, sigma, seed
        _dbg("RobustnessChecker", n=n_perturbations, sigma=sigma)
    def full_check(self, params, optimal_lats=None, default_lats=None):
        report = {}
        s = verify_plan_stability(self.predict_fn, params, self.n_perturbations, self.sigma, self.seed)
        report["stability"] = {"flip_rate": s.flip_rate, "ci": (s.ci_lower, s.ci_upper)}
        if optimal_lats is not None and default_lats is not None:
            p = self.predict_fn(params).flatten()
            if len(p) == len(optimal_lats):
                report["regret"] = measure_regret_under_drift(p, optimal_lats, default_lats)
                report["vs_default"] = compare_with_default(p, default_lats)
        _dbg("full_check", keys=list(report.keys()))
        return report
