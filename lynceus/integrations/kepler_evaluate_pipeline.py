"""
Kepler Evaluation Pipeline
--------------------------
Evaluate query optimizer plan quality and cost model accuracy.
Pure numpy implementation. Every function has a _dbg() variant.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# Data classes
# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

@dataclass
class PlanQualityReport:
    """Aggregated plan-quality metrics."""
    suboptimality_ratios: np.ndarray
    mean_suboptimality: float
    median_suboptimality: float
    p90_suboptimality: float
    p99_suboptimality: float
    speedup_ratios: np.ndarray
    mean_speedup: float
    near_optimal_coverage: float  # fraction within threshold
    total_queries: int
    degraded_count: int  # queries where predicted plan is worse

    def _dbg(self) -> Dict[str, Any]:
        return {
            "mean_suboptimality": self.mean_suboptimality,
            "median_suboptimality": self.median_suboptimality,
            "p90_suboptimality": self.p90_suboptimality,
            "p99_suboptimality": self.p99_suboptimality,
            "mean_speedup": self.mean_speedup,
            "near_optimal_coverage": self.near_optimal_coverage,
            "total_queries": self.total_queries,
            "degraded_count": self.degraded_count,
        }


@dataclass
class CostAccuracyReport:
    """Cost-model accuracy metrics."""
    mae: float
    mape: float
    rmse: float
    spearman_rho: float
    pearson_r: float
    log_error_mean: float
    log_error_std: float
    total_queries: int
    residuals: np.ndarray

    def _dbg(self) -> Dict[str, Any]:
        return {
            "mae": self.mae,
            "mape": self.mape,
            "rmse": self.rmse,
            "spearman_rho": self.spearman_rho,
            "pearson_r": self.pearson_r,
            "log_error_mean": self.log_error_mean,
            "log_error_std": self.log_error_std,
            "total_queries": self.total_queries,
        }


@dataclass
class EvalReport:
    """Combined evaluation report."""
    plan_quality: Optional[PlanQualityReport] = None
    cost_accuracy: Optional[CostAccuracyReport] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def _dbg(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"metadata": self.metadata}
        if self.plan_quality is not None:
            out["plan_quality"] = self.plan_quality._dbg()
        if self.cost_accuracy is not None:
            out["cost_accuracy"] = self.cost_accuracy._dbg()
        return out


# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# Helper: rank-based correlation (pure numpy)
# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def _rankdata(arr: np.ndarray) -> np.ndarray:
    """Average-rank implementation without scipy."""
    sorter = np.argsort(arr)
    ranks = np.empty_like(sorter, dtype=np.float64)
    ranks[sorter] = np.arange(1, len(arr) + 1, dtype=np.float64)
    # handle ties â average ranks for equal values
    sorted_vals = arr[sorter]
    i = 0
    n = len(sorted_vals)
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j > i + 1:
            avg_rank = np.mean(ranks[sorter[i:j]])
            ranks[sorter[i:j]] = avg_rank
        i = j
    return ranks


def _rankdata_dbg(arr: np.ndarray) -> Dict[str, Any]:
    ranks = _rankdata(arr)
    return {"input_len": len(arr), "rank_min": float(ranks.min()), "rank_max": float(ranks.max())}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank-correlation coefficient."""
    rx = _rankdata(x)
    ry = _rankdata(y)
    mx, my = rx.mean(), ry.mean()
    dx, dy = rx - mx, ry - my
    denom = np.sqrt((dx ** 2).sum() * (dy ** 2).sum())
    if denom == 0.0:
        return 0.0
    return float(np.dot(dx, dy) / denom)


def _spearman_dbg(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    rho = _spearman(x, y)
    return {"rho": rho, "n": len(x)}


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    mx, my = x.mean(), y.mean()
    dx, dy = x - mx, y - my
    denom = np.sqrt((dx ** 2).sum() * (dy ** 2).sum())
    if denom == 0.0:
        return 0.0
    return float(np.dot(dx, dy) / denom)


def _pearson_dbg(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    r = _pearson(x, y)
    return {"r": r, "n": len(x)}


# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# PQO (Plan Quality Optimality) metrics
# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def compute_pqo_metrics(
    pred_costs: np.ndarray,
    optimal_costs: np.ndarray,
    near_optimal_threshold: float = 1.10,
) -> PlanQualityReport:
    """
    Compute plan quality / optimality metrics.

    Parameters
    ----------
    pred_costs : predicted plan execution costs (latencies).
    optimal_costs : true optimal plan execution costs.
    near_optimal_threshold : ratio ceiling to count as "near-optimal" (default 1.10 = 10%).

    Returns
    -------
    PlanQualityReport
    """
    pred_costs = np.asarray(pred_costs, dtype=np.float64)
    optimal_costs = np.asarray(optimal_costs, dtype=np.float64)
    assert pred_costs.shape == optimal_costs.shape, "Shape mismatch"

    safe_opt = np.where(optimal_costs > 0, optimal_costs, 1e-12)
    subopt = pred_costs / safe_opt
    speedup = safe_opt / np.where(pred_costs > 0, pred_costs, 1e-12)

    near_mask = subopt <= near_optimal_threshold
    degraded = int(np.sum(subopt > 1.0))

    return PlanQualityReport(
        suboptimality_ratios=subopt,
        mean_suboptimality=float(np.mean(subopt)),
        median_suboptimality=float(np.median(subopt)),
        p90_suboptimality=float(np.percentile(subopt, 90)),
        p99_suboptimality=float(np.percentile(subopt, 99)),
        speedup_ratios=speedup,
        mean_speedup=float(np.mean(speedup)),
        near_optimal_coverage=float(np.mean(near_mask)),
        total_queries=len(pred_costs),
        degraded_count=degraded,
    )


def compute_pqo_metrics_dbg(
    pred_costs: np.ndarray,
    optimal_costs: np.ndarray,
    near_optimal_threshold: float = 1.10,
) -> Dict[str, Any]:
    report = compute_pqo_metrics(pred_costs, optimal_costs, near_optimal_threshold)
    return {
        "report": report._dbg(),
        "threshold": near_optimal_threshold,
        "input_len": len(pred_costs),
    }


# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# Cost-model accuracy metrics
# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def compute_cost_metrics(
    predicted: np.ndarray,
    actual: np.ndarray,
) -> CostAccuracyReport:
    """
    Compute cost-model accuracy: MAE, MAPE, RMSE, Spearman, Pearson,
    log-space error statistics.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    assert predicted.shape == actual.shape, "Shape mismatch"

    residuals = predicted - actual
    mae = float(np.mean(np.abs(residuals)))

    safe_actual = np.where(actual != 0, actual, 1e-12)
    mape = float(np.mean(np.abs(residuals / safe_actual)) * 100.0)

    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    rho = _spearman(predicted, actual)
    r = _pearson(predicted, actual)

    log_pred = np.log1p(np.clip(predicted, 0, None))
    log_act = np.log1p(np.clip(actual, 0, None))
    log_err = log_pred - log_act
    log_mean = float(np.mean(log_err))
    log_std = float(np.std(log_err))

    return CostAccuracyReport(
        mae=mae,
        mape=mape,
        rmse=rmse,
        spearman_rho=rho,
        pearson_r=r,
        log_error_mean=log_mean,
        log_error_std=log_std,
        total_queries=len(predicted),
        residuals=residuals,
    )


def compute_cost_metrics_dbg(
    predicted: np.ndarray,
    actual: np.ndarray,
) -> Dict[str, Any]:
    report = compute_cost_metrics(predicted, actual)
    return {
        "report": report._dbg(),
        "residuals_abs_max": float(np.max(np.abs(report.residuals))),
    }


# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# LaTeX table generation
# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def generate_latex_table(report: EvalReport, caption: str = "Evaluation Results") -> str:
    """Render an EvalReport as a LaTeX table string."""
    lines: List[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\begin{tabular}{l r}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Metric} & \textbf{Value} \\")
    lines.append(r"\midrule")

    if report.plan_quality is not None:
        pq = report.plan_quality
        lines.append(rf"Mean Sub-optimality & {pq.mean_suboptimality:.4f} \\")
        lines.append(rf"Median Sub-optimality & {pq.median_suboptimality:.4f} \\")
        lines.append(rf"P90 Sub-optimality & {pq.p90_suboptimality:.4f} \\")
        lines.append(rf"P99 Sub-optimality & {pq.p99_suboptimality:.4f} \\")
        lines.append(rf"Mean Speedup & {pq.mean_speedup:.4f} \\")
        lines.append(rf"Near-optimal Coverage & {pq.near_optimal_coverage:.2%} \\")
        lines.append(rf"Degraded Queries & {pq.degraded_count} \\")
        lines.append(r"\midrule")

    if report.cost_accuracy is not None:
        ca = report.cost_accuracy
        lines.append(rf"MAE & {ca.mae:.4f} \\")
        lines.append(rf"MAPE (\%) & {ca.mape:.2f} \\")
        lines.append(rf"RMSE & {ca.rmse:.4f} \\")
        lines.append(rf"Spearman $\rho$ & {ca.spearman_rho:.4f} \\")
        lines.append(rf"Pearson $r$ & {ca.pearson_r:.4f} \\")
        lines.append(rf"Log-error $\mu$ & {ca.log_error_mean:.4f} \\")
        lines.append(rf"Log-error $\sigma$ & {ca.log_error_std:.4f} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_latex_table_dbg(report: EvalReport, caption: str = "Evaluation Results") -> Dict[str, Any]:
    tex = generate_latex_table(report, caption)
    return {"char_count": len(tex), "line_count": tex.count("\n") + 1, "preview": tex[:200]}


# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# Main pipeline
# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

_DEFAULTS: Dict[str, Any] = {
    "near_optimal_threshold": 1.10,
    "skip_plan_quality": False,
    "skip_cost_accuracy": False,
}


class EvaluationPipeline:
    """End-to-end evaluation pipeline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = {**_DEFAULTS, **(config or {})}

    # ---- public API ---------------------------------------------------

    def run(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> EvalReport:
        """
        Run the full evaluation pipeline.

        Parameters
        ----------
        predicted : predicted costs / latencies.
        actual : ground-truth costs / latencies.
        defaults : per-call overrides merged on top of pipeline config.
        """
        cfg = {**self.config, **(defaults or {})}
        predicted = np.asarray(predicted, dtype=np.float64).ravel()
        actual = np.asarray(actual, dtype=np.float64).ravel()
        assert predicted.shape == actual.shape, "predicted/actual length mismatch"

        pq: Optional[PlanQualityReport] = None
        ca: Optional[CostAccuracyReport] = None

        if not cfg["skip_plan_quality"]:
            pq = compute_pqo_metrics(predicted, actual, cfg["near_optimal_threshold"])

        if not cfg["skip_cost_accuracy"]:
            ca = compute_cost_metrics(predicted, actual)

        return EvalReport(
            plan_quality=pq,
            cost_accuracy=ca,
            metadata={"config": cfg, "n": len(predicted)},
        )

    def run_dbg(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        report = self.run(predicted, actual, defaults)
        return {"report": report._dbg(), "pipeline_config": self.config}

    # ---- convenience --------------------------------------------------

    def latex(self, report: EvalReport, caption: str = "Kepler Eval") -> str:
        return generate_latex_table(report, caption)

    def latex_dbg(self, report: EvalReport, caption: str = "Kepler Eval") -> Dict[str, Any]:
        return generate_latex_table_dbg(report, caption)
