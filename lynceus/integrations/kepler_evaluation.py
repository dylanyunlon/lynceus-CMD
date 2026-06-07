"""
Kepler Model Evaluation Module
Pure numpy implementation with debug tracing.
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


def kahan_sum(arr):
    """
    Kahan compensated summation for improved floating-point accuracy.
    Uses Neumaier variant for better handling of large intermediate sums.
    """
    global _DBG_DEPTH
    _DBG_DEPTH += 1
    arr = np.asarray(arr, dtype=np.float64).ravel()
    _dbg("kahan_sum", "enter", n=arr.size)

    s = 0.0
    c = 0.0  # compensation
    for i in range(arr.size):
        t = s + arr[i]
        if abs(s) >= abs(arr[i]):
            c += (s - t) + arr[i]  # Neumaier correction
        else:
            c += (arr[i] - t) + s
        s = t
    result = s + c

    _dbg("kahan_sum", "exit", result=result, compensation=c)
    _DBG_DEPTH -= 1
    return result


def accuracy(y_true, y_pred):
    """
    Classification accuracy with Laplace-smoothed confidence interval.
    Returns (acc, lower_bound, upper_bound) using Wilson score interval.
    """
    global _DBG_DEPTH
    _DBG_DEPTH += 1
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    _dbg("accuracy", "enter", y_true=y_true, y_pred=y_pred)

    n = y_true.size
    if n == 0:
        _dbg("accuracy", "empty input, returning 0")
        _DBG_DEPTH -= 1
        return 0.0, 0.0, 0.0

    correct = np.sum(y_true == y_pred)
    acc = float(correct) / n

    # Wilson score interval (z=1.96 for 95%)
    z = 1.96
    denom = 1.0 + z * z / n
    centre = (acc + z * z / (2.0 * n)) / denom
    spread = z * np.sqrt((acc * (1.0 - acc) + z * z / (4.0 * n)) / n) / denom
    lo = max(0.0, centre - spread)
    hi = min(1.0, centre + spread)

    _dbg("accuracy", "exit", acc=acc, ci_lo=lo, ci_hi=hi, n=n)
    _DBG_DEPTH -= 1
    return acc, lo, hi


def near_optimal_ratio(latencies, predictions, threshold=1.1):
    """
    Fraction of predictions within `threshold` factor of optimal latency.
    Uses exponential moving average (EMA) smoothing over the sequence
    to provide a temporally-weighted estimate.
    """
    global _DBG_DEPTH
    _DBG_DEPTH += 1
    latencies = np.asarray(latencies, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    _dbg("near_optimal_ratio", "enter", latencies=latencies, predictions=predictions,
         threshold=threshold)

    n = latencies.size
    if n == 0:
        _DBG_DEPTH -= 1
        return 0.0, np.array([])

    optimal = np.min(latencies) if latencies.ndim == 1 else np.min(latencies, axis=1)

    if latencies.ndim == 1:
        hits = (predictions <= threshold * optimal).astype(np.float64)
    else:
        hits = (predictions <= threshold * optimal).astype(np.float64)

    # EMA smoothing with alpha decay
    alpha = 2.0 / (n + 1) if n > 1 else 1.0
    ema = np.empty(n, dtype=np.float64)
    ema[0] = hits[0]
    for i in range(1, n):
        ema[i] = alpha * hits[i] + (1.0 - alpha) * ema[i - 1]

    raw_ratio = float(kahan_sum(hits)) / n
    smoothed_ratio = float(ema[-1])
    combined = 0.6 * raw_ratio + 0.4 * smoothed_ratio  # blended estimate

    _dbg("near_optimal_ratio", "exit", raw=raw_ratio, smoothed=smoothed_ratio,
         combined=combined)
    _DBG_DEPTH -= 1
    return combined, ema


def normalized_regret(pred_lat, opt_lat, default_lat):
    """
    Normalized regret: 0 = oracle, 1 = default strategy.
    Uses robust normalization with epsilon floor to avoid division-by-zero.
    Applies Huber-like clipping for outlier regret values.
    """
    global _DBG_DEPTH
    _DBG_DEPTH += 1
    pred_lat = np.asarray(pred_lat, dtype=np.float64)
    opt_lat = np.asarray(opt_lat, dtype=np.float64)
    default_lat = np.asarray(default_lat, dtype=np.float64)
    _dbg("normalized_regret", "enter", pred_lat=pred_lat, opt_lat=opt_lat,
         default_lat=default_lat)

    eps = 1e-12
    gap = default_lat - opt_lat
    gap_safe = np.where(np.abs(gap) < eps, eps, gap)

    raw_regret = (pred_lat - opt_lat) / gap_safe

    # Huber-style soft clipping at boundaries [-0.5, 2.0]
    delta_lo, delta_hi = -0.5, 2.0
    clipped = np.clip(raw_regret, delta_lo, delta_hi)

    mean_regret = float(kahan_sum(clipped)) / max(clipped.size, 1)
    median_regret = float(np.median(clipped))
    # Trimmed mean (10% from each tail)
    sorted_r = np.sort(clipped)
    trim = max(1, clipped.size // 10)
    trimmed_mean = float(kahan_sum(sorted_r[trim:-trim])) / max(sorted_r[trim:-trim].size, 1)

    _dbg("normalized_regret", "exit", mean=mean_regret, median=median_regret,
         trimmed_mean=trimmed_mean)
    _DBG_DEPTH -= 1
    return {
        "per_query": clipped,
        "mean": mean_regret,
        "median": median_regret,
        "trimmed_mean": trimmed_mean,
    }


def top_k_accuracy(scores, labels, k=3):
    """
    Top-k hit rate: fraction where true label is among top-k scored items.
    Uses partial argsort for efficiency on large score matrices.
    Includes mean reciprocal rank (MRR) as secondary metric.
    """
    global _DBG_DEPTH
    _DBG_DEPTH += 1
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    _dbg("top_k_accuracy", "enter", scores=scores, labels=labels, k=k)

    n = scores.shape[0]
    if n == 0:
        _DBG_DEPTH -= 1
        return 0.0, 0.0

    n_classes = scores.shape[1] if scores.ndim > 1 else 1
    k = min(k, n_classes)

    hits = 0
    rr_sum = 0.0  # reciprocal ranks

    for i in range(n):
        row = scores[i] if scores.ndim > 1 else scores
        # descending partial sort via negation
        top_idx = np.argpartition(-row, k)[:k]
        # full sort within top-k for rank computation
        top_sorted = top_idx[np.argsort(-row[top_idx])]

        if labels[i] in top_sorted:
            hits += 1
            rank_pos = int(np.where(top_sorted == labels[i])[0][0]) + 1
            rr_sum += 1.0 / rank_pos
        else:
            # check full ranking for MRR even if outside top-k
            full_rank = np.argsort(-row)
            rank_pos = int(np.where(full_rank == labels[i])[0][0]) + 1
            rr_sum += 1.0 / rank_pos

    top_k_acc = float(hits) / n
    mrr = rr_sum / n

    _dbg("top_k_accuracy", "exit", top_k_acc=top_k_acc, mrr=mrr)
    _DBG_DEPTH -= 1
    return top_k_acc, mrr


def compute_confusion_matrix(y_true, y_pred, n_classes):
    """
    Confusion matrix with per-class precision, recall, F1 (macro & weighted).
    Uses Matthews Correlation Coefficient (MCC) for multi-class as additional metric.
    """
    global _DBG_DEPTH
    _DBG_DEPTH += 1
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    _dbg("compute_confusion_matrix", "enter", n_classes=n_classes,
         y_true=y_true, y_pred=y_pred)

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1

    eps = 1e-12
    precision = np.zeros(n_classes, dtype=np.float64)
    recall = np.zeros(n_classes, dtype=np.float64)
    f1 = np.zeros(n_classes, dtype=np.float64)
    support = np.zeros(n_classes, dtype=np.int64)

    for c in range(n_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        support[c] = tp + fn

        precision[c] = tp / (tp + fp + eps)
        recall[c] = tp / (tp + fn + eps)
        f1[c] = 2.0 * precision[c] * recall[c] / (precision[c] + recall[c] + eps)

    total_support = np.sum(support)
    weights = support.astype(np.float64) / max(float(total_support), 1.0)
    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(kahan_sum(f1 * weights))

    # MCC for multi-class (generalized formula)
    c_sum = np.sum(cm)
    t_k = np.sum(cm, axis=1).astype(np.float64)  # true class counts
    p_k = np.sum(cm, axis=0).astype(np.float64)  # pred class counts
    correct = float(np.trace(cm))
    cov_yy = float(np.dot(p_k, p_k))
    cov_xx = float(np.dot(t_k, t_k))
    numerator = correct * c_sum - float(np.dot(t_k, p_k))
    denominator = np.sqrt((c_sum * c_sum - cov_xx) * (c_sum * c_sum - cov_yy) + eps)
    mcc = numerator / denominator

    _dbg("compute_confusion_matrix", "exit", macro_f1=macro_f1,
         weighted_f1=weighted_f1, mcc=mcc)
    _DBG_DEPTH -= 1
    return {
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mcc": mcc,
    }


def evaluate(pred_plan_ids, latency_matrix, default_idx=0):
    """
    Comprehensive evaluation of plan predictions against latency matrix.
    latency_matrix: shape (n_queries, n_plans)
    pred_plan_ids: shape (n_queries,) â selected plan index per query
    Returns a dict with all metrics aggregated.
    Includes geometric mean of speedup and percentile latency analysis.
    """
    global _DBG_DEPTH
    _DBG_DEPTH += 1
    latency_matrix = np.asarray(latency_matrix, dtype=np.float64)
    pred_plan_ids = np.asarray(pred_plan_ids, dtype=np.int64)
    n_queries, n_plans = latency_matrix.shape
    _dbg("evaluate", "enter", n_queries=n_queries, n_plans=n_plans,
         default_idx=default_idx)

    # Gather predicted and oracle latencies
    pred_lats = np.array([latency_matrix[i, pred_plan_ids[i]] for i in range(n_queries)],
                         dtype=np.float64)
    opt_lats = np.min(latency_matrix, axis=1)
    default_lats = latency_matrix[:, default_idx]
    oracle_ids = np.argmin(latency_matrix, axis=1)

    # Accuracy
    acc_val, acc_lo, acc_hi = accuracy(oracle_ids, pred_plan_ids)

    # Near-optimal at several thresholds
    nor_1_05, _ = near_optimal_ratio(opt_lats, pred_lats, threshold=1.05)
    nor_1_10, _ = near_optimal_ratio(opt_lats, pred_lats, threshold=1.10)
    nor_1_20, _ = near_optimal_ratio(opt_lats, pred_lats, threshold=1.20)

    # Normalized regret
    regret = normalized_regret(pred_lats, opt_lats, default_lats)

    # Top-k (build pseudo-scores as negative latency)
    neg_lat = -latency_matrix
    top3_acc, mrr = top_k_accuracy(neg_lat, oracle_ids, k=3)

    # Confusion matrix
    cm_result = compute_confusion_matrix(oracle_ids, pred_plan_ids, n_classes=n_plans)

    # Geometric mean speedup vs default
    eps = 1e-12
    speedups = default_lats / (pred_lats + eps)
    log_speedups = np.log(np.clip(speedups, eps, None))
    geo_mean_speedup = float(np.exp(kahan_sum(log_speedups) / max(n_queries, 1)))

    # Percentile analysis
    p50 = float(np.percentile(pred_lats, 50))
    p90 = float(np.percentile(pred_lats, 90))
    p99 = float(np.percentile(pred_lats, 99))

    result = {
        "accuracy": acc_val,
        "accuracy_ci": (acc_lo, acc_hi),
        "near_optimal_1.05": nor_1_05,
        "near_optimal_1.10": nor_1_10,
        "near_optimal_1.20": nor_1_20,
        "regret_mean": regret["mean"],
        "regret_median": regret["median"],
        "regret_trimmed_mean": regret["trimmed_mean"],
        "regret_per_query": regret["per_query"],
        "top3_accuracy": top3_acc,
        "mrr": mrr,
        "confusion": cm_result,
        "geo_mean_speedup_vs_default": geo_mean_speedup,
        "latency_p50": p50,
        "latency_p90": p90,
        "latency_p99": p99,
        "n_queries": n_queries,
        "n_plans": n_plans,
    }

    _dbg("evaluate", "exit", accuracy=acc_val, regret_mean=regret["mean"],
         geo_speedup=geo_mean_speedup)
    _DBG_DEPTH -= 1
    return result


if __name__ == "__main__":
    np.random.seed(42)
    n_q, n_p = 200, 5
    lat = np.random.exponential(scale=50.0, size=(n_q, n_p)) + 10.0
    oracle = np.argmin(lat, axis=1)
    # simulate a decent predictor: 70% oracle, 30% random
    mask = np.random.rand(n_q) < 0.7
    preds = np.where(mask, oracle, np.random.randint(0, n_p, size=n_q))

    res = evaluate(preds, lat, default_idx=0)
    print("\n=== Kepler Evaluation Summary ===")
    for k in ["accuracy", "accuracy_ci", "near_optimal_1.10", "regret_mean",
              "regret_trimmed_mean", "top3_accuracy", "mrr",
              "geo_mean_speedup_vs_default", "latency_p50", "latency_p90",
              "latency_p99"]:
        print(f"  {k}: {res[k]}")
    print(f"  macro_f1: {res['confusion']['macro_f1']:.4f}")
    print(f"  mcc: {res['confusion']['mcc']:.4f}")
