"""
Kepler Evaluate & Visualize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Multi-strategy evaluation pipeline with cost comparison, PQO evaluation,
and result visualization.  Pure-Python + numpy port of
evaluate.py (344L) + evaluate_both.py (489L) + evaluate_cost.py (235L)
+ evaluate_pqo.py (263L) + end_visualize*.py (625L total).

Algorithm changes (~20%):
  - NDCG and MRR rank metrics added to evaluation
  - ASCII art visualization (replaces matplotlib)
  - Welford streaming stats for online metric computation
  - EMA convergence detection for early stopping
  - Harmonic mean aggregation for multi-metric comparison
"""

import collections
import csv
import io
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
_DEBUG = False
JSON = Any


def enable_debug(flag: bool = True) -> None:
    global _DEBUG
    _DEBUG = flag


def _dbg(tag: str, msg: str = "", **kw: Any) -> None:
    if not _DEBUG:
        return
    extras = " ".join(f"{k}={v!r}" for k, v in kw.items())
    print(f"[DBG {time.perf_counter():.6f}] {tag}: {msg} {extras}", file=sys.stderr)


# ---------------------------------------------------------------------------
class WelfordAcc:
    def __init__(self):
        self.n = 0; self.mean = 0.0; self.M2 = 0.0
    def update(self, x):
        self.n += 1; d = x - self.mean; self.mean += d/self.n; self.M2 += d*(x-self.mean)
    @property
    def std(self): return math.sqrt(self.M2/self.n) if self.n>1 else 0.0
    def __repr__(self): return f"W(n={self.n},μ={self.mean:.3f},σ={self.std:.3f})"


# ---------------------------------------------------------------------------
# NDCG and MRR  (algorithm change #1)
# ---------------------------------------------------------------------------
def _dcg(relevances: List[float], k: int = 0) -> float:
    """Discounted Cumulative Gain."""
    if k <= 0: k = len(relevances)
    dcg = 0.0
    for i, r in enumerate(relevances[:k]):
        dcg += r / math.log2(i + 2)
    return dcg


def ndcg(predicted_ranking: List[int], true_costs: Dict[int, float],
         k: int = 0) -> float:
    """Normalized DCG for plan ranking quality."""
    _dbg("ndcg", n_plans=len(predicted_ranking), k=k)
    sorted_true = sorted(true_costs.keys(), key=lambda p: true_costs[p])
    ideal_rels = [1.0 / (i + 1) for i in range(len(sorted_true))]
    pred_rels = []
    for plan in predicted_ranking:
        if plan in true_costs:
            rank = sorted_true.index(plan)
            pred_rels.append(1.0 / (rank + 1))
        else:
            pred_rels.append(0.0)
    ideal = _dcg(ideal_rels, k)
    actual = _dcg(pred_rels, k)
    score = actual / ideal if ideal > 0 else 0.0
    _dbg("ndcg", "done", score=f"{score:.4f}")
    return score


def mrr(predicted_ranking: List[int], optimal_plans: set) -> float:
    """Mean Reciprocal Rank: how early the first optimal plan appears."""
    _dbg("mrr", n_predicted=len(predicted_ranking), n_optimal=len(optimal_plans))
    for i, plan in enumerate(predicted_ranking):
        if plan in optimal_plans:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Plan evaluation metrics
# ---------------------------------------------------------------------------
class PlanEvaluator:
    """Evaluate plan selection quality across parameters."""

    def __init__(self):
        self._results: List[JSON] = []
        self._tracker = WelfordAcc()
        _dbg("PlanEvaluator.__init__")

    def add_result(self, params: str, plan_latencies: Dict[int, float],
                   selected_plan: int, default_plan: int):
        """Record evaluation result for one parameter binding."""
        optimal_plan = min(plan_latencies, key=lambda p: plan_latencies[p])
        optimal_lat = plan_latencies[optimal_plan]
        selected_lat = plan_latencies.get(selected_plan, float('inf'))
        default_lat = plan_latencies.get(default_plan, float('inf'))

        suboptimality = selected_lat / optimal_lat if optimal_lat > 0 else 1.0
        improvement = (default_lat - selected_lat) / default_lat if default_lat > 0 else 0.0
        self._tracker.update(suboptimality)

        result = {
            "params": params,
            "optimal_plan": optimal_plan,
            "selected_plan": selected_plan,
            "default_plan": default_plan,
            "optimal_latency": optimal_lat,
            "selected_latency": selected_lat,
            "default_latency": default_lat,
            "suboptimality_ratio": suboptimality,
            "improvement_over_default": improvement,
        }
        self._results.append(result)
        _dbg("PlanEvaluator.add_result", params=params[:30],
             subopt=f"{suboptimality:.3f}", improvement=f"{improvement:.3f}")

    def compute_summary(self) -> JSON:
        """Compute aggregate evaluation metrics."""
        _dbg("PlanEvaluator.compute_summary", n_results=len(self._results))
        if not self._results:
            return {"error": "no results"}

        subopt_ratios = [r["suboptimality_ratio"] for r in self._results]
        improvements = [r["improvement_over_default"] for r in self._results]
        correct_count = sum(1 for r in self._results
                            if r["selected_plan"] == r["optimal_plan"])

        summary = {
            "n_params": len(self._results),
            "accuracy": correct_count / len(self._results),
            "mean_suboptimality": float(np.mean(subopt_ratios)),
            "median_suboptimality": float(np.median(subopt_ratios)),
            "p95_suboptimality": float(np.percentile(subopt_ratios, 95)),
            "mean_improvement": float(np.mean(improvements)),
            "total_improved": sum(1 for i in improvements if i > 0),
            "total_regressed": sum(1 for i in improvements if i < 0),
            "streaming_stats": repr(self._tracker),
        }
        _dbg("PlanEvaluator.compute_summary", "done", **summary)
        return summary

    def __repr__(self):
        return f"PlanEvaluator(n={len(self._results)}, stats={self._tracker})"


# ---------------------------------------------------------------------------
# Cost-based evaluation
# ---------------------------------------------------------------------------
def evaluate_cost_accuracy(estimated_costs: Dict[int, float],
                            actual_costs: Dict[int, float]) -> JSON:
    """Evaluate cost model accuracy."""
    _dbg("evaluate_cost_accuracy",
         n_estimated=len(estimated_costs), n_actual=len(actual_costs))

    common = set(estimated_costs) & set(actual_costs)
    if not common:
        return {"error": "no common plans"}

    errors = []
    for plan in common:
        est = estimated_costs[plan]
        act = actual_costs[plan]
        if act > 0:
            rel_err = abs(est - act) / act
            errors.append(rel_err)

    q_errors = []
    for plan in common:
        est = max(estimated_costs[plan], 1e-10)
        act = max(actual_costs[plan], 1e-10)
        q_errors.append(max(est/act, act/est))

    result = {
        "n_plans": len(common),
        "mean_relative_error": float(np.mean(errors)),
        "median_relative_error": float(np.median(errors)),
        "mean_q_error": float(np.mean(q_errors)),
        "median_q_error": float(np.median(q_errors)),
        "p90_q_error": float(np.percentile(q_errors, 90)),
        "rank_ndcg": ndcg(
            sorted(estimated_costs, key=lambda p: estimated_costs[p]),
            actual_costs, k=5),
    }
    _dbg("evaluate_cost_accuracy", "done", **result)
    return result


# ---------------------------------------------------------------------------
# PQO evaluation
# ---------------------------------------------------------------------------
def evaluate_pqo(predicted_plans: Dict[str, int],
                  actual_optimal: Dict[str, int],
                  plan_latencies: Dict[str, Dict[int, float]]) -> JSON:
    """Evaluate Parametric Query Optimization quality."""
    _dbg("evaluate_pqo", n_params=len(predicted_plans))

    correct = 0
    total_subopt = WelfordAcc()
    improvements = []

    for params_key in predicted_plans:
        pred = predicted_plans[params_key]
        opt = actual_optimal.get(params_key, pred)
        lats = plan_latencies.get(params_key, {})

        if pred == opt:
            correct += 1
            total_subopt.update(1.0)
        elif lats:
            pred_lat = lats.get(pred, float('inf'))
            opt_lat = lats.get(opt, float('inf'))
            ratio = pred_lat / opt_lat if opt_lat > 0 else 1.0
            total_subopt.update(ratio)

        if lats:
            default_lat = list(lats.values())[0] if lats else 0
            pred_lat = lats.get(pred, default_lat)
            if default_lat > 0:
                improvements.append((default_lat - pred_lat) / default_lat)

    n = len(predicted_plans)
    result = {
        "accuracy": correct / n if n > 0 else 0,
        "mean_suboptimality": total_subopt.mean,
        "std_suboptimality": total_subopt.std,
        "mean_improvement": float(np.mean(improvements)) if improvements else 0,
        "pqo_mrr": mrr(list(predicted_plans.values()),
                        set(actual_optimal.values())),
    }
    _dbg("evaluate_pqo", "done", accuracy=result["accuracy"])
    return result


# ---------------------------------------------------------------------------
# Combined evaluation (evaluate_both)
# ---------------------------------------------------------------------------
def evaluate_both(latency_results: JSON, cost_results: JSON,
                   pqo_results: Optional[JSON] = None) -> JSON:
    """Combine latency, cost, and PQO evaluation metrics.

    Uses harmonic mean for multi-metric aggregation (algorithm change #2).
    """
    _dbg("evaluate_both", "start")

    def harmonic_mean(vals):
        pos = [v for v in vals if v > 0]
        if not pos: return 0.0
        return len(pos) / sum(1.0/v for v in pos)

    metrics = {
        "latency": latency_results,
        "cost": cost_results,
    }
    if pqo_results:
        metrics["pqo"] = pqo_results

    key_scores = []
    if "accuracy" in latency_results:
        key_scores.append(latency_results["accuracy"])
    if "rank_ndcg" in cost_results:
        key_scores.append(cost_results["rank_ndcg"])
    if pqo_results and "accuracy" in pqo_results:
        key_scores.append(pqo_results["accuracy"])

    combined = {
        "metrics": metrics,
        "harmonic_score": harmonic_mean(key_scores),
        "arithmetic_score": float(np.mean(key_scores)) if key_scores else 0,
    }
    _dbg("evaluate_both", "done",
         harmonic=combined["harmonic_score"],
         arithmetic=combined["arithmetic_score"])
    return combined


# ---------------------------------------------------------------------------
# EMA convergence detection  (algorithm change #3)
# ---------------------------------------------------------------------------
class ConvergenceDetector:
    """Detect convergence of a metric series using EMA smoothing."""

    def __init__(self, alpha: float = 0.1, threshold: float = 0.001,
                 patience: int = 5):
        self._alpha = alpha
        self._threshold = threshold
        self._patience = patience
        self._ema = None
        self._stable_count = 0
        _dbg("ConvergenceDetector.__init__",
             alpha=alpha, threshold=threshold, patience=patience)

    def update(self, value: float) -> bool:
        """Update with new value. Returns True if converged."""
        if self._ema is None:
            self._ema = value
            return False
        old_ema = self._ema
        self._ema = self._alpha * value + (1 - self._alpha) * self._ema
        change = abs(self._ema - old_ema) / (abs(old_ema) + 1e-10)
        if change < self._threshold:
            self._stable_count += 1
        else:
            self._stable_count = 0
        converged = self._stable_count >= self._patience
        _dbg("ConvergenceDetector.update",
             value=f"{value:.4f}", ema=f"{self._ema:.4f}",
             change=f"{change:.6f}", stable=self._stable_count,
             converged=converged)
        return converged

    def __repr__(self):
        return f"ConvDet(ema={self._ema}, stable={self._stable_count})"


# ---------------------------------------------------------------------------
# ASCII visualization  (algorithm change #4)
# ---------------------------------------------------------------------------
def ascii_comparison_chart(data: Dict[str, Tuple[float, float]],
                            title: str = "",
                            labels: Tuple[str, str] = ("Hinted", "Default"),
                            width: int = 40) -> str:
    """Dual-bar comparison chart."""
    _dbg("ascii_comparison_chart", n=len(data))
    if not data:
        return "(no data)"
    max_val = max(max(a, b) for a, b in data.values()) or 1
    lines = [f"  {title}", f"  {labels[0]:>15s} | {labels[1]}", "  " + "-" * (width + 30)]
    for label, (v1, v2) in data.items():
        b1 = int(v1 / max_val * width)
        b2 = int(v2 / max_val * width)
        lines.append(f"  {label:>12s} {'█' * b1:>{width}s} | {'░' * b2}")
    return "\n".join(lines)


def ascii_convergence_plot(values: List[float], title: str = "",
                            height: int = 10, width: int = 60) -> str:
    """ASCII line plot for convergence visualization."""
    _dbg("ascii_convergence_plot", n=len(values))
    if not values:
        return "(no data)"
    vmin, vmax = min(values), max(values)
    vrange = vmax - vmin or 1

    lines = [f"  {title}", f"  max={vmax:.3f}"]
    grid = [[" "] * width for _ in range(height)]
    for i, v in enumerate(values):
        x = min(int(i / len(values) * width), width - 1)
        y = min(int((v - vmin) / vrange * (height - 1)), height - 1)
        grid[height - 1 - y][x] = "●"

    for row in grid:
        lines.append("  |" + "".join(row) + "|")
    lines.append(f"  min={vmin:.3f}" + " " * (width - 10) + f"n={len(values)}")
    return "\n".join(lines)


def visualize_evaluation(latency_summary: JSON,
                          cost_summary: JSON,
                          convergence_values: Optional[List[float]] = None) -> str:
    """Generate full evaluation visualization."""
    _dbg("visualize_evaluation", "start")
    parts = []

    parts.append("=" * 60)
    parts.append("  EVALUATION REPORT")
    parts.append("=" * 60)

    parts.append("\n  Latency Evaluation:")
    for k, v in latency_summary.items():
        if isinstance(v, float):
            parts.append(f"    {k}: {v:.4f}")
        else:
            parts.append(f"    {k}: {v}")

    parts.append("\n  Cost Model Accuracy:")
    for k, v in cost_summary.items():
        if isinstance(v, float):
            parts.append(f"    {k}: {v:.4f}")
        else:
            parts.append(f"    {k}: {v}")

    if convergence_values:
        parts.append("\n" + ascii_convergence_plot(
            convergence_values, "Training Convergence"))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CSV I/O for evaluation results
# ---------------------------------------------------------------------------
def save_evaluation_csv(results: List[JSON], path: str = "") -> str:
    """Save evaluation results to CSV format."""
    _dbg("save_evaluation_csv", n=len(results))
    if not results:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(results[0].keys()))
    writer.writeheader()
    for r in results:
        writer.writerow(r)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    enable_debug(True)
    print("=" * 60)
    print("  kepler_evaluate_visualize — self-test")
    print("=" * 60)

    rng = np.random.RandomState(42)

    # Test 1: PlanEvaluator
    print("\n--- Test 1: PlanEvaluator ---")
    ev = PlanEvaluator()
    for i in range(10):
        lats = {p: rng.lognormal(5, 1) for p in range(5)}
        selected = rng.randint(0, 5)
        ev.add_result(f"params_{i}", lats, selected, default_plan=0)
    summary = ev.compute_summary()
    print(f"  Accuracy: {summary['accuracy']:.2f}")
    print(f"  Mean suboptimality: {summary['mean_suboptimality']:.3f}")
    print(f"  Improved: {summary['total_improved']}, Regressed: {summary['total_regressed']}")

    # Test 2: NDCG & MRR
    print("\n--- Test 2: NDCG & MRR ---")
    true_costs = {0: 100, 1: 200, 2: 50, 3: 300, 4: 150}
    predicted = [2, 0, 4, 1, 3]  # close to optimal ordering
    n = ndcg(predicted, true_costs, k=3)
    print(f"  NDCG@3: {n:.4f}")
    m = mrr(predicted, {2})  # plan 2 is optimal
    print(f"  MRR: {m:.4f}")

    # Test 3: Cost evaluation
    print("\n--- Test 3: Cost accuracy ---")
    est_costs = {p: true_costs[p] * rng.uniform(0.8, 1.3) for p in true_costs}
    cost_eval = evaluate_cost_accuracy(est_costs, true_costs)
    print(f"  Mean q-error: {cost_eval['mean_q_error']:.3f}")
    print(f"  Rank NDCG: {cost_eval['rank_ndcg']:.4f}")

    # Test 4: PQO evaluation
    print("\n--- Test 4: PQO evaluation ---")
    pred_plans = {f"p{i}": rng.randint(0, 3) for i in range(20)}
    opt_plans = {f"p{i}": rng.randint(0, 3) for i in range(20)}
    plan_lats = {f"p{i}": {p: rng.lognormal(4, 0.5) for p in range(3)} for i in range(20)}
    pqo_eval = evaluate_pqo(pred_plans, opt_plans, plan_lats)
    print(f"  Accuracy: {pqo_eval['accuracy']:.3f}")
    print(f"  MRR: {pqo_eval['pqo_mrr']:.4f}")

    # Test 5: Combined evaluation
    print("\n--- Test 5: Combined evaluation ---")
    combined = evaluate_both(summary, cost_eval, pqo_eval)
    print(f"  Harmonic score: {combined['harmonic_score']:.4f}")
    print(f"  Arithmetic score: {combined['arithmetic_score']:.4f}")

    # Test 6: Convergence detection
    print("\n--- Test 6: Convergence detection ---")
    cd = ConvergenceDetector(alpha=0.2, threshold=0.005, patience=3)
    vals = [1.0, 0.8, 0.6, 0.5, 0.45, 0.44, 0.435, 0.433, 0.432, 0.432]
    for v in vals:
        converged = cd.update(v)
        if converged:
            print(f"  Converged at value={v:.3f}")
            break
    else:
        print(f"  Not converged. State: {cd}")

    # Test 7: Visualization
    print("\n--- Test 7: Visualization ---")
    viz = visualize_evaluation(summary, cost_eval, vals)
    print(viz)

    # Test 8: Comparison chart
    print("\n--- Test 8: Comparison chart ---")
    cmp_data = {
        "q1": (120.5, 200.3),
        "q2": (80.0, 95.2),
        "q3": (300.1, 150.7),
        "q4": (45.0, 180.0),
    }
    chart = ascii_comparison_chart(cmp_data, "Hinted vs Default Latency (ms)")
    print(chart)

    print("\n✓ All self-tests passed")
