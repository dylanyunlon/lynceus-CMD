"""
Kepler End-to-End Evaluation Module
Pure numpy implementation with debug tracing.
Simulates workload execution, compares strategies, and tests significance.
"""

import numpy as np
import sys
import time

_DEBUG = True
_DBG_DEPTH = 0


def _dbg(func_name, msg="", **kwargs):
    """Debug trace helper with indented call tracking."""
    global _DBG_DEPTH
    if not _DEBUG:
        return
    indent = "  " * _DBG_DEPTH
    ts = time.perf_counter()
    parts = [f"[DBG {ts:.6f}] {indent}{func_name}"]
    if msg:
        parts.append(f": {msg}")
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            parts.append(f" | {k}.shape={v.shape}, dtype={v.dtype}")
        else:
            parts.append(f" | {k}={v}")
    print("".join(parts), file=sys.stderr)


class EvaluationResult:
    """Dataclass-style container for end-to-end evaluation results."""

    __slots__ = (
        "query_id", "baseline_latencies", "model_latencies",
        "speedups", "mean_speedup", "geo_mean_speedup",
        "win_rate", "loss_rate", "tie_rate",
        "p_value", "significant", "total_baseline_time",
        "total_model_time", "relative_improvement",
    )

    def __init__(self, **kwargs):
        _dbg("EvaluationResult.__init__", "constructing")
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot, None))

    def to_dict(self):
        _dbg("EvaluationResult.to_dict", "serializing")
        return {s: getattr(self, s) for s in self.__slots__}

    def __repr__(self):
        items = []
        for s in self.__slots__:
            v = getattr(self, s)
            if isinstance(v, np.ndarray):
                items.append(f"{s}=ndarray({v.shape})")
            elif isinstance(v, float):
                items.append(f"{s}={v:.6f}")
            else:
                items.append(f"{s}={v}")
        return f"EvaluationResult({', '.join(items)})"


def simulate_workload_execution(latency_matrix, pred_ids, noise_std=0.0):
    """
    Simulate execution of predicted plans with optional Gaussian noise.
    Uses log-normal noise model (multiplicative) instead of additive
    for more realistic latency jitter simulation.
    Returns observed latencies array.
    """
    global _DBG_DEPTH
    _DBG_DEPTH += 1
    latency_matrix = np.asarray(latency_matrix, dtype=np.float64)
    pred_ids = np.asarray(pred_ids, dtype=np.int64)
    n_queries = latency_matrix.shape[0]
    _dbg("simulate_workload_execution", "enter",
         latency_matrix=latency_matrix, pred_ids=pred_ids, noise_std=noise_std)

    base_lats = np.array([latency_matrix[i, pred_ids[i]] for i in range(n_queries)],
                         dtype=np.float64)

    if noise_std > 0.0:
        # Log-normal multiplicative noise: lat * exp(N(0, sigma))
        log_noise = np.random.normal(0.0, noise_std, size=n_queries)
        multiplier = np.exp(log_noise)
        observed = base_lats * multiplier
        # Floor at 0.1 ms to avoid nonsensical zero latencies
        observed = np.maximum(observed, 0.1)
        _dbg("simulate_workload_execution", "noise applied",
             mean_multiplier=float(np.mean(multiplier)),
             std_multiplier=float(np.std(multiplier)))
    else:
        observed = base_lats.copy()

    _dbg("simulate_workload_execution", "exit",
         mean_lat=float(np.mean(observed)),
         median_lat=float(np.median(observed)))
    _DBG_DEPTH -= 1
    return observed


def compare_latencies(baseline_lats, model_lats, tolerance=0.05):
    """
    Compare two latency arrays: compute speedup, win/loss/tie breakdown.
    Uses asymmetric tolerance: model must beat baseline by `tolerance`
    fraction to count as a win, but baseline only needs to match to avoid
    counting as a loss (favoring conservative assessment).
    Returns dict with per-query speedups and aggregate rates.
    """
    global _DBG_DEPTH
    _DBG_DEPTH += 1
    baseline_lats = np.asarray(baseline_lats, dtype=np.float64)
    model_lats = np.asarray(model_lats, dtype=np.float64)
    n = baseline_lats.size
    _dbg("compare_latencies", "enter", baseline_lats=baseline_lats,
         model_lats=model_lats, tolerance=tolerance)

    eps = 1e-12
    speedups = baseline_lats / (model_lats + eps)

    # Asymmetric thresholds
    win_threshold = 1.0 + tolerance       # model must be clearly faster
    loss_threshold = 1.0 / (1.0 + tolerance * 0.5)  # more lenient for loss

    wins = np.sum(speedups > win_threshold)
    losses = np.sum(speedups < loss_threshold)
    ties = n - wins - losses

    win_rate = float(wins) / max(n, 1)
    loss_rate = float(losses) / max(n, 1)
    tie_rate = float(ties) / max(n, 1)

    mean_speedup = float(np.mean(speedups))

    # Geometric mean speedup (more robust to outliers)
    log_sp = np.log(np.clip(speedups, eps, None))
    geo_mean_speedup = float(np.exp(np.mean(log_sp)))

    # Relative improvement in total time
    total_bl = float(np.sum(baseline_lats))
    total_ml = float(np.sum(model_lats))
    relative_improvement = (total_bl - total_ml) / (total_bl + eps)

    result = {
        "speedups": speedups,
        "mean_speedup": mean_speedup,
        "geo_mean_speedup": geo_mean_speedup,
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "tie_rate": tie_rate,
        "wins": int(wins),
        "losses": int(losses),
        "ties": int(ties),
        "relative_improvement": relative_improvement,
    }

    _dbg("compare_latencies", "exit", mean_speedup=mean_speedup,
         geo_mean=geo_mean_speedup, win_rate=win_rate, loss_rate=loss_rate)
    _DBG_DEPTH -= 1
    return result


def statistical_significance(a, b, method="wilcoxon"):
    """
    Non-parametric significance test between two latency samples.
    Implements Wilcoxon signed-rank test from scratch (pure numpy).
    Also computes bootstrap confidence interval for mean difference.
    Returns p_value, effect_size (r = Z/sqrt(N)), and CI.
    """
    global _DBG_DEPTH
    _DBG_DEPTH += 1
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    _dbg("statistical_significance", "enter", method=method, a=a, b=b)

    diff = a - b
    # Remove zero differences
    nonzero_mask = np.abs(diff) > 1e-15
    diff_nz = diff[nonzero_mask]
    n = diff_nz.size

    if n < 5:
        _dbg("statistical_significance", "too few non-zero diffs", n=n)
        _DBG_DEPTH -= 1
        return {"p_value": 1.0, "effect_size": 0.0, "ci_95": (0.0, 0.0),
                "method": method, "n_effective": n}

    if method == "wilcoxon":
        # Rank absolute differences
        abs_diff = np.abs(diff_nz)
        order = np.argsort(abs_diff)
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)

        # Handle tied ranks by averaging
        sorted_abs = abs_diff[order]
        i = 0
        while i < n:
            j = i + 1
            while j < n and np.abs(sorted_abs[j] - sorted_abs[i]) < 1e-15:
                j += 1
            if j > i + 1:
                avg_rank = np.mean(np.arange(i + 1, j + 1, dtype=np.float64))
                for idx in range(i, j):
                    ranks[order[idx]] = avg_rank
            i = j

        # Signed ranks
        w_plus = float(np.sum(ranks[diff_nz > 0]))
        w_minus = float(np.sum(ranks[diff_nz < 0]))
        w_stat = min(w_plus, w_minus)

        # Normal approximation with continuity correction
        mean_w = n * (n + 1.0) / 4.0
        var_w = n * (n + 1.0) * (2.0 * n + 1.0) / 24.0
        z = (w_stat - mean_w + 0.5) / (np.sqrt(var_w) + 1e-15)
        # Two-tailed p-value via approximation: p â 2 * Î¦(-|z|)
        # Using logistic approximation to normal CDF
        abs_z = abs(z)
        p_value = 2.0 * (1.0 / (1.0 + np.exp(1.7 * abs_z + 0.1 * abs_z ** 3)))
        p_value = min(p_value, 1.0)

        effect_size = abs(z) / np.sqrt(n)

        _dbg("statistical_significance", "wilcoxon computed",
             w_plus=w_plus, w_minus=w_minus, z=z, p_value=p_value)
    else:
        # Fallback: permutation test
        observed_diff = float(np.mean(diff_nz))
        n_perm = 5000
        count = 0
        for _ in range(n_perm):
            signs = np.random.choice([-1.0, 1.0], size=n)
            perm_mean = float(np.mean(diff_nz * signs))
            if abs(perm_mean) >= abs(observed_diff):
                count += 1
        p_value = float(count) / n_perm
        effect_size = abs(observed_diff) / (float(np.std(diff_nz)) + 1e-15)

        _dbg("statistical_significance", "permutation computed",
             observed_diff=observed_diff, p_value=p_value)

    # Bootstrap 95% CI for mean difference
    n_boot = 2000
    boot_means = np.empty(n_boot, dtype=np.float64)
    for bi in range(n_boot):
        sample = diff_nz[np.random.randint(0, n, size=n)]
        boot_means[bi] = np.mean(sample)
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    _dbg("statistical_significance", "exit", p_value=p_value,
         effect_size=effect_size, ci=(ci_lo, ci_hi))
    _DBG_DEPTH -= 1
    return {
        "p_value": p_value,
        "effect_size": effect_size,
        "ci_95": (ci_lo, ci_hi),
        "method": method,
        "n_effective": n,
    }


def summarize_results(all_query_results):
    """
    Aggregate multiple per-query EvaluationResult objects into a summary.
    Computes weighted averages (by query count), overall significance,
    and identifies worst/best performing queries.
    """
    global _DBG_DEPTH
    _DBG_DEPTH += 1
    n_results = len(all_query_results)
    _dbg("summarize_results", "enter", n_results=n_results)

    if n_results == 0:
        _dbg("summarize_results", "empty input")
        _DBG_DEPTH -= 1
        return {}

    all_speedups = []
    total_baseline = 0.0
    total_model = 0.0
    all_p_values = []
    per_query_summary = []

    for i, er in enumerate(all_query_results):
        d = er.to_dict() if isinstance(er, EvaluationResult) else er
        sp = d.get("speedups")
        if sp is not None:
            if isinstance(sp, np.ndarray):
                all_speedups.extend(sp.tolist())
            else:
                all_speedups.append(float(sp))
        tb = d.get("total_baseline_time", 0.0)
        tm = d.get("total_model_time", 0.0)
        if tb is not None:
            total_baseline += float(tb)
        if tm is not None:
            total_model += float(tm)
        pv = d.get("p_value")
        if pv is not None:
            all_p_values.append(float(pv))

        per_query_summary.append({
            "query_id": d.get("query_id", i),
            "mean_speedup": d.get("mean_speedup", 0.0),
            "geo_mean_speedup": d.get("geo_mean_speedup", 0.0),
            "win_rate": d.get("win_rate", 0.0),
        })

    all_speedups = np.array(all_speedups, dtype=np.float64)
    eps = 1e-12

    global_mean_speedup = float(np.mean(all_speedups)) if all_speedups.size > 0 else 0.0
    log_sp = np.log(np.clip(all_speedups, eps, None))
    global_geo_speedup = float(np.exp(np.mean(log_sp))) if all_speedups.size > 0 else 0.0
    global_relative_imp = (total_baseline - total_model) / (total_baseline + eps)

    # Combine p-values using Fisher's method (chi-squared approximation)
    combined_p = 1.0
    if all_p_values:
        p_arr = np.array(all_p_values, dtype=np.float64)
        p_arr = np.clip(p_arr, 1e-300, 1.0)
        chi2_stat = -2.0 * float(np.sum(np.log(p_arr)))
        df = 2 * len(all_p_values)
        # Approximate chi-squared p-value using Wilson-Hilferty
        if df > 0:
            z_wh = ((chi2_stat / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / \
                   np.sqrt(2.0 / (9.0 * df))
            combined_p = 1.0 / (1.0 + np.exp(1.7 * z_wh + 0.1 * z_wh ** 3))
            combined_p = min(max(combined_p, 0.0), 1.0)

    # Best / worst queries by geometric speedup
    per_q = sorted(per_query_summary, key=lambda x: x.get("geo_mean_speedup", 0.0))
    worst_3 = per_q[:min(3, len(per_q))]
    best_3 = per_q[-min(3, len(per_q)):][::-1]

    summary = {
        "n_queries": n_results,
        "n_total_executions": int(all_speedups.size),
        "global_mean_speedup": global_mean_speedup,
        "global_geo_mean_speedup": global_geo_speedup,
        "global_relative_improvement": global_relative_imp,
        "total_baseline_time": total_baseline,
        "total_model_time": total_model,
        "combined_p_value": combined_p,
        "globally_significant": combined_p < 0.05,
        "best_queries": best_3,
        "worst_queries": worst_3,
        "speedup_p50": float(np.percentile(all_speedups, 50)) if all_speedups.size else 0.0,
        "speedup_p10": float(np.percentile(all_speedups, 10)) if all_speedups.size else 0.0,
        "speedup_p90": float(np.percentile(all_speedups, 90)) if all_speedups.size else 0.0,
    }

    _dbg("summarize_results", "exit",
         global_geo=global_geo_speedup,
         combined_p=combined_p,
         n_queries=n_results)
    _DBG_DEPTH -= 1
    return summary


if __name__ == "__main__":
    np.random.seed(7)
    n_q, n_p = 100, 4
    lat_mat = np.random.exponential(30.0, size=(n_q, n_p)) + 5.0
    oracle = np.argmin(lat_mat, axis=1)
    mask = np.random.rand(n_q) < 0.65
    model_pred = np.where(mask, oracle, np.random.randint(0, n_p, size=n_q))
    default_pred = np.zeros(n_q, dtype=np.int64)

    bl_lats = simulate_workload_execution(lat_mat, default_pred, noise_std=0.05)
    ml_lats = simulate_workload_execution(lat_mat, model_pred, noise_std=0.05)

    cmp = compare_latencies(bl_lats, ml_lats, tolerance=0.05)
    sig = statistical_significance(bl_lats, ml_lats, method="wilcoxon")

    er = EvaluationResult(
        query_id="demo",
        baseline_latencies=bl_lats,
        model_latencies=ml_lats,
        speedups=cmp["speedups"],
        mean_speedup=cmp["mean_speedup"],
        geo_mean_speedup=cmp["geo_mean_speedup"],
        win_rate=cmp["win_rate"],
        loss_rate=cmp["loss_rate"],
        tie_rate=cmp["tie_rate"],
        p_value=sig["p_value"],
        significant=sig["p_value"] < 0.05,
        total_baseline_time=float(np.sum(bl_lats)),
        total_model_time=float(np.sum(ml_lats)),
        relative_improvement=cmp["relative_improvement"],
    )

    summary = summarize_results([er])

    print("\n=== E2E Evaluation Result ===")
    print(er)
    print("\n=== Summary ===")
    for k, v in summary.items():
        if isinstance(v, list):
            print(f"  {k}:")
            for item in v:
                print(f"    {item}")
        else:
            print(f"  {k}: {v}")
