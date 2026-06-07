"""
kepler_visualization — Text-based visualization utilities (no matplotlib).
Ported from upstream end_visualize.py. Algorithm changes:
  - Pure text/ASCII output (no GUI dependency)
  - Streaming CSV export
  - JSON metrics export
  - Progress bar with ETA estimation
"""
import os, time, json, math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))
def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in list(kw.items())[:8])
        print(f"[kepler_viz] {tag}: {items}")

def text_bar_chart(values, labels=None, width=40, title=""):
    """Render an ASCII horizontal bar chart."""
    vals = np.asarray(values, dtype=float)
    mx = max(abs(vals.max()), abs(vals.min()), 1e-10)
    if labels is None:
        labels = [f"[{i}]" for i in range(len(vals))]
    max_label = max(len(str(l)) for l in labels)
    lines = []
    if title:
        lines.append(f"  {title}")
        lines.append("  " + "-" * (max_label + width + 12))
    for i, (v, l) in enumerate(zip(vals, labels)):
        bar_len = int(abs(v) / mx * width)
        bar = "#" * bar_len
        lines.append(f"  {str(l):>{max_label}} | {bar:<{width}} {v:>10.2f}")
    _dbg("bar_chart", n=len(vals), max=float(mx), width=width)
    return "\n".join(lines)

def text_table(headers, rows, col_width=12):
    """Render a plain-text table."""
    hdr = " | ".join(f"{h:>{col_width}}" for h in headers)
    sep = "-+-".join("-" * col_width for _ in headers)
    lines = [f"  {hdr}", f"  {sep}"]
    for row in rows:
        cells = []
        for c in row:
            if isinstance(c, float):
                cells.append(f"{c:>{col_width}.4f}")
            else:
                cells.append(f"{str(c):>{col_width}}")
        lines.append("  " + " | ".join(cells))
    _dbg("table", headers=headers, n_rows=len(rows))
    return "\n".join(lines)

def export_latency_csv(results, path, delimiter=","):
    """Export latency arrays to CSV file."""
    keys = sorted(results.keys())
    arrays = [np.asarray(results[k]) for k in keys]
    n = max(len(a) for a in arrays) if arrays else 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(delimiter.join(keys) + "\n")
        for i in range(n):
            row = []
            for a in arrays:
                row.append(f"{a[i]:.6f}" if i < len(a) and np.issubdtype(a.dtype, np.floating) else str(a[i]) if i < len(a) else "")
            f.write(delimiter.join(row) + "\n")
    _dbg("export_csv", path=path, n_rows=n, n_cols=len(keys))

def export_summary_json(metrics, path):
    """Export summary metrics to JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    def default_ser(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.int64, np.int32)): return int(o)
        if isinstance(o, (np.float64, np.float32)): return float(o)
        return str(o)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=default_ser)
    _dbg("export_json", path=path, n_keys=len(metrics))

def format_comparison_report(baseline, model, names=("Default", "Model")):
    """Format a comparison report between baseline and model latencies."""
    bl, ml = np.asarray(baseline), np.asarray(model)
    speedup = bl / np.maximum(ml, 1e-10)
    wins = int(np.sum(ml < bl * 0.95))
    losses = int(np.sum(ml > bl * 1.05))
    ties = len(bl) - wins - losses
    lines = [
        f"  Comparison: {names[0]} vs {names[1]}",
        f"  {'='*50}",
        f"  {names[0]} median: {np.median(bl):.2f}ms",
        f"  {names[1]} median: {np.median(ml):.2f}ms",
        f"  Speedup: {np.median(speedup):.2f}x (median), {np.mean(speedup):.2f}x (mean)",
        f"  Wins/Losses/Ties: {wins}/{losses}/{ties}",
        f"  Win rate: {wins/max(len(bl),1)*100:.1f}%",
    ]
    _dbg("comparison", speedup_median=float(np.median(speedup)), wins=wins, losses=losses)
    return "\n".join(lines)

class ProgressTracker:
    """Track training progress with ETA estimation."""
    def __init__(self, total_epochs, bar_width=30):
        self.total = total_epochs
        self.bar_width = bar_width
        self.losses = []
        self.start_time = time.time()
        _dbg("ProgressTracker", total=total_epochs, bar_width=bar_width)

    def update(self, epoch, loss):
        self.losses.append(float(loss))

    def render(self):
        if not self.losses:
            return "(no data)"
        epoch = len(self.losses)
        pct = epoch / max(self.total, 1)
        filled = int(pct * self.bar_width)
        bar = "#" * filled + "." * (self.bar_width - filled)
        elapsed = time.time() - self.start_time
        eta = elapsed / max(epoch, 1) * (self.total - epoch)
        loss = self.losses[-1]
        return f"  [{bar}] {epoch}/{self.total} loss={loss:.4f} eta={eta:.0f}s"

    def render_loss_curve(self, height=10, width=50):
        """ASCII loss curve."""
        if len(self.losses) < 2:
            return "(not enough data)"
        vals = np.array(self.losses)
        mn, mx = vals.min(), vals.max()
        rng = max(mx - mn, 1e-10)
        n = len(vals)
        step = max(1, n // width)
        sampled = vals[::step][:width]
        lines = []
        for row in range(height):
            threshold = mx - (row / (height - 1)) * rng
            line = ""
            for v in sampled:
                line += "*" if v >= threshold else " "
            lines.append(f"  {threshold:8.3f} |{line}")
        _dbg("loss_curve", n_points=len(sampled), min_loss=float(mn), max_loss=float(mx))
        return "\n".join(lines)
