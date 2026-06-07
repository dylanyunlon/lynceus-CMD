"""
Evaluation metrics and reporting for D2STGNN Aurora traffic forecasting.
Pure NumPy implementation.
"""

import numpy as np
import os

_DEBUG = os.environ.get("AURORA_DEBUG", "0") == "1"


def _dbg(*args, **kwargs):
    if _DEBUG:
        print("[DEBUG evaluator]", *args, **kwargs)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def mae(pred, target):
    """
    Mean Absolute Error.

    Parameters
    ----------
    pred, target : np.ndarray of same shape

    Returns
    -------
    float
    """
    return float(np.mean(np.abs(pred - target)))


def rmse(pred, target):
    """
    Root Mean Squared Error.

    Parameters
    ----------
    pred, target : np.ndarray of same shape

    Returns
    -------
    float
    """
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mape(pred, target, eps=1e-5):
    """
    Mean Absolute Percentage Error (%).

    Parameters
    ----------
    pred, target : np.ndarray of same shape
    eps : float â threshold to mask near-zero targets

    Returns
    -------
    float â percentage value (e.g. 5.2 means 5.2%)
    """
    mask = np.abs(target) > eps
    if not np.any(mask):
        _dbg("MAPE: all targets below eps, returning 0.0")
        return 0.0
    return float(np.mean(np.abs((pred[mask] - target[mask]) / target[mask]))) * 100.0


def compute_all_metrics(pred, target, eps=1e-5):
    """
    Compute MAE, RMSE, MAPE over the full prediction tensor.

    Parameters
    ----------
    pred : np.ndarray, shape (B, T_out, N, C) or (T_out, N, C)
    target : np.ndarray, same shape

    Returns
    -------
    dict with keys 'mae', 'rmse', 'mape'
    """
    result = {
        "mae": mae(pred, target),
        "rmse": rmse(pred, target),
        "mape": mape(pred, target, eps=eps),
    }
    _dbg(f"compute_all_metrics: {result}")
    return result


# ---------------------------------------------------------------------------
# Per-dimension metrics
# ---------------------------------------------------------------------------

def per_node_metrics(pred, target, eps=1e-5):
    """
    Compute metrics per spatial node.

    Parameters
    ----------
    pred : np.ndarray, shape (B, T, N, C) or (T, N, C)
    target : np.ndarray, same shape

    Returns
    -------
    dict mapping node_index -> {'mae', 'rmse', 'mape'}
    """
    if pred.ndim == 3:
        pred = pred[np.newaxis]
        target = target[np.newaxis]

    B, T, N, C = pred.shape
    results = {}
    for n in range(N):
        p = pred[:, :, n, :]
        t = target[:, :, n, :]
        results[n] = {
            "mae": mae(p, t),
            "rmse": rmse(p, t),
            "mape": mape(p, t, eps=eps),
        }
    _dbg(f"per_node_metrics: computed for {N} nodes")
    return results


def per_horizon_metrics(pred, target, eps=1e-5):
    """
    Compute metrics per prediction horizon step.

    Parameters
    ----------
    pred : np.ndarray, shape (B, T_out, N, C) or (T_out, N, C)
    target : np.ndarray, same shape

    Returns
    -------
    dict mapping horizon_step (0-indexed) -> {'mae', 'rmse', 'mape'}
    """
    if pred.ndim == 3:
        pred = pred[np.newaxis]
        target = target[np.newaxis]

    B, T_out, N, C = pred.shape
    results = {}
    for h in range(T_out):
        p = pred[:, h, :, :]
        t = target[:, h, :, :]
        results[h] = {
            "mae": mae(p, t),
            "rmse": rmse(p, t),
            "mape": mape(p, t, eps=eps),
        }
    _dbg(f"per_horizon_metrics: computed for {T_out} horizons")
    return results


# ---------------------------------------------------------------------------
# EvalReport
# ---------------------------------------------------------------------------

class EvalReport:
    """
    Structured evaluation report containing overall, per-horizon,
    and per-node metrics.

    Attributes
    ----------
    overall : dict â {'mae', 'rmse', 'mape'}
    horizons : dict â {step: {'mae', 'rmse', 'mape'}}
    nodes : dict â {node_id: {'mae', 'rmse', 'mape'}}
    pred_len : int
    n_nodes : int
    """

    def __init__(self, pred, target, eps=1e-5):
        """
        Build a full report from predictions and targets.

        Parameters
        ----------
        pred : np.ndarray, shape (B, T_out, N, C) or (T_out, N, C)
        target : np.ndarray, same shape
        eps : float
        """
        self.overall = compute_all_metrics(pred, target, eps=eps)
        self.horizons = per_horizon_metrics(pred, target, eps=eps)
        self.nodes = per_node_metrics(pred, target, eps=eps)

        if pred.ndim == 3:
            self.pred_len = pred.shape[0]
            self.n_nodes = pred.shape[1]
        else:
            self.pred_len = pred.shape[1]
            self.n_nodes = pred.shape[2]

        _dbg(
            f"EvalReport created: pred_len={self.pred_len}, "
            f"n_nodes={self.n_nodes}, overall={self.overall}"
        )

    def summary(self):
        """Return a concise summary dict."""
        return {
            "overall": self.overall,
            "horizon_mae": {h: v["mae"] for h, v in self.horizons.items()},
            "best_node": min(self.nodes, key=lambda k: self.nodes[k]["mae"]),
            "worst_node": max(self.nodes, key=lambda k: self.nodes[k]["mae"]),
        }


# ---------------------------------------------------------------------------
# ASCII table formatting
# ---------------------------------------------------------------------------

def format_eval_table(report):
    """
    Format an EvalReport as a readable ASCII table.

    Parameters
    ----------
    report : EvalReport

    Returns
    -------
    str â multi-section ASCII table
    """
    lines = []
    sep = "=" * 62

    # --- Overall ---
    lines.append(sep)
    lines.append("  D2STGNN Aurora â Evaluation Report")
    lines.append(sep)
    lines.append("")
    lines.append("  Overall Metrics:")
    lines.append(f"    MAE  : {report.overall['mae']:.6f}")
    lines.append(f"    RMSE : {report.overall['rmse']:.6f}")
    lines.append(f"    MAPE : {report.overall['mape']:.4f}%")
    lines.append("")

    # --- Per-horizon table ---
    lines.append("-" * 62)
    header = f"  {'Horizon':>8s} | {'MAE':>10s} | {'RMSE':>10s} | {'MAPE (%)':>10s}"
    lines.append(header)
    lines.append("  " + "-" * 56)

    for h in sorted(report.horizons.keys()):
        m = report.horizons[h]
        lines.append(
            f"  {h + 1:>8d} | {m['mae']:>10.5f} | "
            f"{m['rmse']:>10.5f} | {m['mape']:>10.4f}"
        )
    lines.append("")

    # --- Per-node summary (top/bottom 5) ---
    lines.append("-" * 62)
    lines.append("  Per-Node Summary (sorted by MAE):")
    lines.append("")

    node_maes = sorted(report.nodes.items(), key=lambda kv: kv[1]["mae"])
    n_show = min(5, len(node_maes))

    lines.append(f"  {'Node':>6s} | {'MAE':>10s} | {'RMSE':>10s} | {'MAPE (%)':>10s}")
    lines.append("  " + "-" * 50)

    if n_show > 0:
        lines.append("  --- Best nodes ---")
        for node_id, m in node_maes[:n_show]:
            lines.append(
                f"  {node_id:>6d} | {m['mae']:>10.5f} | "
                f"{m['rmse']:>10.5f} | {m['mape']:>10.4f}"
            )

    if len(node_maes) > n_show:
        lines.append("  ...")
        lines.append("  --- Worst nodes ---")
        for node_id, m in node_maes[-n_show:]:
            lines.append(
                f"  {node_id:>6d} | {m['mae']:>10.5f} | "
                f"{m['rmse']:>10.5f} | {m['mape']:>10.4f}"
            )

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)


def print_eval_report(report):
    """Print a formatted evaluation report to stdout."""
    print(format_eval_table(report))


# ---------------------------------------------------------------------------
# Quick-run helper
# ---------------------------------------------------------------------------

def evaluate_model(model, test_iter, adj=None, scaler=None):
    """
    Run a model over a test iterator and produce a full EvalReport.

    Parameters
    ----------
    model : object with forward(x, adj)
    test_iter : iterable of (x_batch, y_batch)
    adj : np.ndarray or None
    scaler : StandardScaler or None â if provided, inverse-transforms
             predictions and targets before computing metrics

    Returns
    -------
    EvalReport
    """
    all_preds = []
    all_targets = []

    for x_batch, y_batch in test_iter:
        pred = model.forward(x_batch, adj)
        all_preds.append(pred)
        all_targets.append(y_batch)

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    if scaler is not None:
        _dbg("Inverse-transforming predictions and targets for evaluation")
        preds = scaler.inverse_transform(preds)
        targets = scaler.inverse_transform(targets)

    report = EvalReport(preds, targets)
    return report
