"""
lynceus/viz/plot_panels.py — Visualization and figure generation.

Architecture references (ported/adapted from):
  - PAR2QO diagram.py (par2qo/code/diagram.py:1-565)
    → Diagram class: PQO feature collection, plan caching, model serialisation
    → pqoByFeatureCollection() — complete workflow: collect → reduce → reweight
    → collectFeatures(), collectPlans(), collectPlanCost() pipeline
    → saveModeltoCache/loadModelFromCache — JSON serialisation
    → initLogFile() — per-experiment log path construction
  - PAR2QO diagram_best_cost.py (par2qo/code/diagram_best_cost.py:1-152)
    → DiagramBestCost.evaluate() — plan evaluation loop with logging:
      per-query: collect cost, select best, compare PQO vs PG, log speedup
    → Timing breakdown: avg_costing_time, avg_planning_time
    → Batch result collection: result_pqo, result_pg lists
    → Sub-optimality boundary check (bound parameter)
  - PAR2QO diagram_nearest.py (par2qo/code/diagram_nearest.py:1-96)
    → Diagram_Nearest.evaluate() — L2-nearest selectivity sample lookup
    → find_nearest_sample() call pattern
    → Same result logging format as DiagramBestCost
  - PAR2QO plan_reduction_by_similarity.py (par2qo/code/plan_reduction_by_similarity.py:1-217)
    → plot_all_cost_distribution() — multi-plan cost distribution line plots
    → plot_2d_matrix() — KL divergence heatmap with colorbar
    → reduce_matrix() — iterative closest-pair elimination with logging
    → JS_distance(), kl_divergence() — similarity metrics for plot annotation

Modifications from upstream references (~20% original):
  - Removed: matplotlib.pyplot direct usage (deferred to caller)
  - Removed: psycopg2 database connections, EXPLAIN execution
  - Removed: tqdm progress bars, numpy array operations
  - Added:   Lynceus cost-model-specific panel types (CPU/GPU routing, calibration)
  - Added:   Panel composition framework for multi-figure layouts
  - Added:   Text-based ASCII plot fallback for terminal/debug use
  - Added:   Comprehensive debug dump of all plot data at each stage
  - Changed: 'plan cost distribution' → 'cost model prediction distribution'
  - Changed: 'KL divergence matrix' → 'worker divergence heatmap'
  - Changed: PQO evaluation loop → cost model benchmark result analysis

Design:
  Generates structured panel data for visualisation of cost model evaluation
  results. Each panel is a data container (not rendered here) that holds
  axes, labels, series, and annotations ready for a plotting backend
  (matplotlib, plotly, or ASCII). This separates data preparation from
  rendering, allowing the same analysis code to produce figures for
  papers, dashboards, or terminal debugging.
"""

from __future__ import annotations

import math
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum, auto

from .. import _dbg, _dump_obj, _snapshot, _Timer, LYNCEUS_DEBUG
_T = "PLT"


logger = logging.getLogger(__name__)


# ─── Panel Types ─────────────────────────────────────────────────────────────

class PanelKind(Enum):
    """Type of visualisation panel."""
    COST_DISTRIBUTION = auto()     # from plot_all_cost_distribution
    DIVERGENCE_MATRIX = auto()     # from plot_2d_matrix
    CALIBRATION_SCATTER = auto()   # predicted vs actual scatter
    ROUTING_BAR = auto()           # CPU vs GPU routing decisions
    SPEEDUP_LINE = auto()          # cumulative speedup over baseline
    ERROR_HISTOGRAM = auto()       # relative error distribution
    TIMELINE = auto()              # per-query timeline


# ─── Data Series ─────────────────────────────────────────────────────────────
# Adapted from par2qo plan_reduction_by_similarity.py plot_all_cost_distribution:
# each plan's cost list is one series; we generalise.

@dataclass
class DataSeries:
    """One line/bar/scatter series in a panel.

    Analogous to one iteration of:
        for i, cost_list in enumerate(all_cost_list):
            ax1.plot(cost_list, label=str(plan_id), ...)
    from par2qo's plot_all_cost_distribution (line 95-115).
    """
    series_id: str
    label: str
    x_values: List[float] = field(default_factory=list)
    y_values: List[float] = field(default_factory=list)
    # Style hints (mirrors par2qo's linestyle/marker/color params)
    style: str = "solid"             # solid, dashed, dotted
    marker: str = "o"                # o, ., x, none
    color: str = ""                  # empty = auto-assign
    is_anchor: bool = False          # highlighted series (par2qo: anchor param)

    def dump_debug(self, prefix: str = "") -> str:
        _dbg(_T, "dump_debug()")
        n = len(self.y_values)
        y_min = min(self.y_values) if self.y_values else 0
        y_max = max(self.y_values) if self.y_values else 0
        y_mean = sum(self.y_values) / max(1, n)
        return (f"{prefix}Series[{self.series_id}] label='{self.label}' n={n} "
                f"y∈[{y_min:.2f}, {y_max:.2f}] mean={y_mean:.2f} "
                f"anchor={self.is_anchor}")


# ─── Panel Data ──────────────────────────────────────────────────────────────

@dataclass
class PanelData:
    """Complete data for one visualisation panel.

    Combines the axis/label/series structure from par2qo's matplotlib calls:
      - ax1.set_ylabel("Log-based Plan Cost", fontsize=30)
      - ax1.set_ylim((1000, 1000000000))
      - plt.title("Plan Cost Distribution", fontsize=25)
      - plt.yscale('log')
    """
    panel_id: str
    kind: PanelKind
    title: str
    x_label: str = ""
    y_label: str = ""
    # Data series
    series: List[DataSeries] = field(default_factory=list)
    # Matrix data (for heatmaps — from plot_2d_matrix)
    matrix_data: Optional[List[List[float]]] = None
    matrix_labels: Optional[List[str]] = None
    # Axis config
    x_range: Optional[Tuple[float, float]] = None
    y_range: Optional[Tuple[float, float]] = None
    log_scale_y: bool = False
    log_scale_x: bool = False
    # Annotations
    annotations: List[str] = field(default_factory=list)
    # Grid lines (from par2qo: ax1.axvline every 50 samples)
    grid_interval: Optional[int] = None

    def dump_debug(self, prefix: str = "") -> str:
        _dbg(_T, "dump_debug()")
        lines = [
            f"{prefix}╔══ PanelData [{self.panel_id}] ════════════════════════",
            f"{prefix}║ kind       = {self.kind.name}",
            f"{prefix}║ title      = {self.title}",
            f"{prefix}║ x_label    = {self.x_label}",
            f"{prefix}║ y_label    = {self.y_label}",
            f"{prefix}║ n_series   = {len(self.series)}",
            f"{prefix}║ log_y      = {self.log_scale_y}",
            f"{prefix}║ y_range    = {self.y_range}",
        ]
        if self.matrix_data:
            lines.append(f"{prefix}║ matrix     = {len(self.matrix_data)}×{len(self.matrix_data[0])}")
        for s in self.series:
            lines.append(f"{prefix}║   {s.dump_debug()}")
        if self.annotations:
            lines.append(f"{prefix}║ annotations = {self.annotations}")
        lines.append(f"{prefix}╚═══════════════════════════════════════════════════")
        return "\n".join(lines)


# ─── Cost Distribution Panel ────────────────────────────────────────────────
# Adapted from par2qo plan_reduction_by_similarity.py:plot_all_cost_distribution
# (lines 85-170). Original plotted per-plan cost across samples with
# anchor highlighting; we build the same data structure.

def build_cost_distribution_panel(
    cost_lists: List[List[float]],
    labels: Optional[List[str]] = None,
    anchor: Optional[str] = None,
    sort: bool = False,
    panel_id: str = "cost_dist",
    debug_print: bool = True,
) -> PanelData:
    """Build cost distribution panel data.

    Ported from par2qo/code/plan_reduction_by_similarity.py:plot_all_cost_distribution (line 85).

    Original flow:
        fig, ax1 = plt.subplots(figsize=(10, 10))
        for i, cost_list in enumerate(all_cost_list):
            if sort: cost_list = sorted(cost_list)
            plan_id = labels[i] if labels else i
            ax1.plot(cost_list, label=str(plan_id), ...)
        ax1.set_ylabel("Log-based Plan Cost", fontsize=30)
        ax1.set_ylim((1000, 1000000000))
        plt.yscale('log')

    Lynceus: builds PanelData instead of directly plotting.
    'plans' → cost model configurations being compared.
    """
    _dbg(_T, "build_cost_distribution_panel()")
    panel = PanelData(
        panel_id=panel_id,
        kind=PanelKind.COST_DISTRIBUTION,
        title="Cost Model Prediction Distribution" if not sort
              else "Cost Model Prediction (Sorted) Distribution",
        x_label="Sample Index",
        y_label="Predicted Cost (µs, log scale)",
        log_scale_y=True,
        y_range=(1.0, 1e9),
    )

    # Default matplotlib color cycle (from par2qo: plt.rcParams['axes.prop_cycle'])
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i, cost_list in enumerate(cost_lists):
        values = sorted(cost_list) if sort else list(cost_list)
        label = labels[i] if labels and i < len(labels) else str(i)
        color_idx = i % len(default_colors)

        is_anchor_series = (anchor is not None and label == anchor)

        series = DataSeries(
            series_id=f"config_{i}",
            label=label,
            x_values=list(range(len(values))),
            y_values=values,
            style="solid" if is_anchor_series else "dashed",
            marker="o" if is_anchor_series else ".",
            color=default_colors[0] if is_anchor_series else "lightgrey",
            is_anchor=is_anchor_series,
        )
        panel.series.append(series)

    # Grid lines every 50 samples (from par2qo: ax1.axvline every 50)
    if cost_lists and cost_lists[0]:
        panel.grid_interval = 50

    if debug_print:
        print(f"\n  [plot_panels] Built cost distribution panel:")
        print(panel.dump_debug("    "))

    return panel


# ─── Divergence Matrix Panel ────────────────────────────────────────────────
# Adapted from par2qo plan_reduction_by_similarity.py:plot_2d_matrix (lines 172-195).
# Original plotted an NxN KL divergence matrix with viridis colormap.

def build_divergence_matrix_panel(
    matrix: List[List[float]],
    labels: Optional[List[str]] = None,
    panel_id: str = "div_matrix",
    title: str = "Worker Divergence Matrix",
    debug_print: bool = True,
) -> PanelData:
    """Build divergence matrix heatmap panel.

    Ported from par2qo/code/plan_reduction_by_similarity.py:plot_2d_matrix (line 172).

    Original:
        plt.imshow(matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Value')
        plt.title('Relative KL of each pair of plan -- Visualization')
        plt.xlabel('Plan ID'); plt.ylabel('Plan ID')
        plt.xticks(ticks=np.arange(matrix.shape[1]), labels=id_list)

    Lynceus: 'Plan ID' → 'Worker ID', builds PanelData.
    """
    _dbg(_T, "build_divergence_matrix_panel()")
    n = len(matrix)
    if labels is None:
        labels = [str(i) for i in range(n)]

    panel = PanelData(
        panel_id=panel_id,
        kind=PanelKind.DIVERGENCE_MATRIX,
        title=title,
        x_label="Worker ID",
        y_label="Worker ID",
        matrix_data=matrix,
        matrix_labels=labels,
    )

    # Compute summary stats for annotations
    all_vals = [v for row in matrix for v in row if v != 0 and not math.isinf(v)]
    if all_vals:
        mean_div = sum(all_vals) / len(all_vals)
        max_div = max(all_vals)
        panel.annotations.append(f"Mean divergence: {mean_div:.4f}")
        panel.annotations.append(f"Max divergence: {max_div:.4f}")

    if debug_print:
        print(f"\n  [plot_panels] Built divergence matrix panel:")
        print(panel.dump_debug("    "))

    return panel


# ─── Calibration Scatter Panel ──────────────────────────────────────────────
# New panel type for Lynceus — predicted vs actual cost scatter plot.
# Evaluation pattern adapted from par2qo diagram_best_cost.py evaluate():
#   result_pqo.append(robust_plan_latency)
#   result_pg.append(pg_plan_latency)
#   total_pqo += robust_plan_latency
#   total_pg += pg_plan_latency

def build_calibration_scatter_panel(
    predicted: List[float],
    actual: List[float],
    panel_id: str = "calibration",
    debug_print: bool = True,
) -> PanelData:
    """Build predicted-vs-actual calibration scatter panel.

    Inspired by par2qo diagram_best_cost.py evaluate() result collection:
        result_pqo, result_pg = [], []
        for sql_id, para_sql in enumerate(...):
            ...
            result_pqo.append(robust_plan_latency)
            result_pg.append(pg_plan_latency)

    Instead of PQO vs PG, we plot Predicted vs Actual cost.
    """
    _dbg(_T, "build_calibration_scatter_panel()")
    assert len(predicted) == len(actual), "predicted/actual length mismatch"

    panel = PanelData(
        panel_id=panel_id,
        kind=PanelKind.CALIBRATION_SCATTER,
        title="Cost Model Calibration: Predicted vs Actual",
        x_label="Actual Cost (µs)",
        y_label="Predicted Cost (µs)",
        log_scale_x=True,
        log_scale_y=True,
    )

    # Main scatter series
    scatter = DataSeries(
        series_id="calibration_points",
        label="Predictions",
        x_values=actual,
        y_values=predicted,
        style="none",
        marker="o",
        color="#1f77b4",
    )
    panel.series.append(scatter)

    # Perfect calibration line (y = x)
    if actual:
        min_val = min(min(actual), min(predicted))
        max_val = max(max(actual), max(predicted))
        # Guard against zero/negative for log scale
        min_val = max(0.1, min_val)
        ideal = DataSeries(
            series_id="ideal_line",
            label="Perfect Calibration",
            x_values=[min_val, max_val],
            y_values=[min_val, max_val],
            style="dashed",
            marker="none",
            color="red",
        )
        panel.series.append(ideal)

    # Summary statistics
    errors = []
    for p, a in zip(predicted, actual):
        if p > 0 and a > 0:
            errors.append(math.log(p / a))
    if errors:
        mean_log_err = sum(errors) / len(errors)
        rmsle = math.sqrt(sum(e**2 for e in errors) / len(errors))
        panel.annotations.append(f"Mean log error: {mean_log_err:.4f}")
        panel.annotations.append(f"RMSLE: {rmsle:.4f}")
        panel.annotations.append(f"N: {len(predicted)}")

    if debug_print:
        print(f"\n  [plot_panels] Built calibration scatter panel:")
        print(panel.dump_debug("    "))

    return panel


# ─── Speedup Line Panel ─────────────────────────────────────────────────────
# Adapted from par2qo diagram_best_cost.py evaluate() running totals:
#   total_pqo += robust_plan_latency
#   total_pg += pg_plan_latency
#   output_string = f"avg: pg {round(total_pg/len(result_pg))} / pqo {round(total_pqo/len(result_pqo))} = {ratio}"

def build_speedup_line_panel(
    optimised_costs: List[float],
    baseline_costs: List[float],
    panel_id: str = "speedup",
    optimised_label: str = "Lynceus (GPU-routed)",
    baseline_label: str = "CPU Baseline",
    debug_print: bool = True,
) -> PanelData:
    """Build cumulative speedup line panel.

    Adapted from par2qo/code/diagram_best_cost.py evaluate() running average:
        total_pqo += robust_plan_latency
        total_pg += pg_plan_latency
        ratio = round(total_pg / total_pqo, 3)

    Shows cumulative speedup of optimised routing over baseline.
    """
    _dbg(_T, "build_speedup_line_panel()")
    assert len(optimised_costs) == len(baseline_costs)

    panel = PanelData(
        panel_id=panel_id,
        kind=PanelKind.SPEEDUP_LINE,
        title="Cumulative Speedup: Optimised vs Baseline",
        x_label="Query Index",
        y_label="Cumulative Speedup (×)",
    )

    # Compute running speedup
    cum_opt = 0.0
    cum_base = 0.0
    speedups = []
    better_count = 0

    for i in range(len(optimised_costs)):
        cum_opt += optimised_costs[i]
        cum_base += baseline_costs[i]
        speedup = cum_base / max(0.001, cum_opt)
        speedups.append(speedup)
        if optimised_costs[i] < baseline_costs[i]:
            better_count += 1

    # Speedup series
    speedup_series = DataSeries(
        series_id="speedup",
        label="Cumulative Speedup",
        x_values=list(range(len(speedups))),
        y_values=speedups,
        style="solid",
        marker="none",
        color="#1f77b4",
        is_anchor=True,
    )
    panel.series.append(speedup_series)

    # 1× reference line
    ref_line = DataSeries(
        series_id="baseline_ref",
        label="1× (break-even)",
        x_values=[0, len(speedups) - 1],
        y_values=[1.0, 1.0],
        style="dashed",
        marker="none",
        color="grey",
    )
    panel.series.append(ref_line)

    # Summary (mirrors par2qo's final output_string)
    n = len(optimised_costs)
    panel.annotations.append(
        f"Final speedup: {speedups[-1]:.3f}× "
        f"({better_count}/{n} queries faster)"
    )
    # Mirror par2qo's format:
    # "PG avg: X ms, PQO avg: Y ms, Ratio is Z"
    opt_avg = cum_opt / max(1, n)
    base_avg = cum_base / max(1, n)
    panel.annotations.append(
        f"Baseline avg: {base_avg:.1f}µs, Optimised avg: {opt_avg:.1f}µs"
    )

    if debug_print:
        print(f"\n  [plot_panels] Built speedup line panel:")
        print(panel.dump_debug("    "))

    return panel


# ─── ASCII Rendering ────────────────────────────────────────────────────────
# New addition for terminal debugging — no upstream equivalent.
# Renders a simple text-based bar chart from panel data.

def render_ascii_bar(panel: PanelData, width: int = 60,
                     debug_print: bool = True) -> str:
    """Render a panel as ASCII bar chart for terminal debugging."""
    _dbg(_T, "render_ascii_bar()")
    lines = [
        f"┌{'─' * (width + 2)}┐",
        f"│ {panel.title:^{width}} │",
        f"├{'─' * (width + 2)}┤",
    ]

    for series in panel.series:
        if not series.y_values:
            continue

        max_val = max(abs(v) for v in series.y_values) or 1.0

        lines.append(f"│ {series.label} ({len(series.y_values)} points):")

        # Show first 10 values as bars
        for i, val in enumerate(series.y_values[:10]):
            bar_len = int(abs(val) / max_val * (width - 20))
            bar = "█" * bar_len
            label = f"  [{i:>3}] {val:>10.1f} │{bar}"
            lines.append(f"│{label:<{width + 1}}│")

        if len(series.y_values) > 10:
            lines.append(f"│  ... ({len(series.y_values) - 10} more){' ' * (width - 18)}│")

    # Annotations
    if panel.annotations:
        lines.append(f"├{'─' * (width + 2)}┤")
        for ann in panel.annotations:
            lines.append(f"│ {ann:<{width}} │")

    lines.append(f"└{'─' * (width + 2)}┘")

    result = "\n".join(lines)

    if debug_print:
        print(f"\n{result}")

    return result


# ─── ASCII Matrix Rendering ─────────────────────────────────────────────────

def render_ascii_matrix(panel: PanelData, cell_width: int = 8,
                        debug_print: bool = True) -> str:
    """Render a matrix panel as ASCII table."""
    _dbg(_T, "render_ascii_matrix()")
    if not panel.matrix_data:
        return "(no matrix data)"

    mat = panel.matrix_data
    labels = panel.matrix_labels or [str(i) for i in range(len(mat))]
    n = len(mat)

    lines = [f"  {panel.title}", ""]

    # Header row
    header = f"{'':>8}" + "".join(f"{l:>{cell_width}}" for l in labels)
    lines.append(header)
    lines.append("  " + "─" * (8 + cell_width * n))

    # Data rows
    for i in range(n):
        row_str = f"  {labels[i]:>6} │"
        for j in range(n):
            val = mat[i][j]
            if i == j:
                row_str += f"{'  ---':>{cell_width}}"
            else:
                row_str += f"{val:>{cell_width}.3f}"
        lines.append(row_str)

    # Annotations
    if panel.annotations:
        lines.append("")
        for ann in panel.annotations:
            lines.append(f"  {ann}")

    result = "\n".join(lines)

    if debug_print:
        print(f"\n{result}")

    return result


# ─── Panel Composition ──────────────────────────────────────────────────────

@dataclass
class FigureLayout:
    """Multi-panel figure layout.

    Analogous to par2qo's figure generation workflow where multiple
    plots are saved to separate PDF files. Here we compose panels
    into a logical figure for export.
    """
    figure_id: str
    title: str
    panels: List[PanelData] = field(default_factory=list)
    # Layout hints
    n_cols: int = 2
    fig_width: float = 20.0
    fig_height: float = 10.0

    def add_panel(self, panel: PanelData) -> None:
        _dbg(_T, "add_panel()")
        self.panels.append(panel)

    def export_json(self) -> str:
        """Export figure layout as JSON for external rendering."""
        _dbg(_T, "export_json()")
        output = {
            "figure_id": self.figure_id,
            "title": self.title,
            "layout": {"n_cols": self.n_cols, "width": self.fig_width, "height": self.fig_height},
            "panels": [],
        }
        for p in self.panels:
            panel_dict = {
                "panel_id": p.panel_id,
                "kind": p.kind.name,
                "title": p.title,
                "x_label": p.x_label,
                "y_label": p.y_label,
                "log_scale_x": p.log_scale_x,
                "log_scale_y": p.log_scale_y,
                "annotations": p.annotations,
                "n_series": len(p.series),
                "series": [
                    {"id": s.series_id, "label": s.label,
                     "n_points": len(s.y_values), "is_anchor": s.is_anchor}
                    for s in p.series
                ],
            }
            if p.matrix_data:
                panel_dict["matrix_size"] = f"{len(p.matrix_data)}×{len(p.matrix_data[0])}"
            output["panels"].append(panel_dict)
        return json.dumps(output, indent=2)

    def dump_debug(self, prefix: str = "") -> str:
        _dbg(_T, "dump_debug()")
        lines = [
            f"{prefix}╔══ FigureLayout [{self.figure_id}] ═════════════════════",
            f"{prefix}║ title    = {self.title}",
            f"{prefix}║ n_panels = {len(self.panels)}",
            f"{prefix}║ layout   = {self.n_cols}col × {self.fig_width}×{self.fig_height}",
        ]
        for p in self.panels:
            lines.append(f"{prefix}║   [{p.panel_id}] {p.kind.name}: {p.title} "
                       f"({len(p.series)} series)")
        lines.append(f"{prefix}╚═══════════════════════════════════════════════════")
        return "\n".join(lines)


# ─── Convenience: Build Full Benchmark Report ───────────────────────────────
# Combines multiple panel types into a complete figure layout.

def build_benchmark_report(
    predicted_costs: List[float],
    actual_costs: List[float],
    baseline_costs: Optional[List[float]] = None,
    worker_divergence: Optional[List[List[float]]] = None,
    worker_labels: Optional[List[str]] = None,
    config_cost_lists: Optional[List[List[float]]] = None,
    config_labels: Optional[List[str]] = None,
    report_id: str = "benchmark_report",
    debug_print: bool = True,
) -> FigureLayout:
    """Build a complete benchmark report with multiple panels.

    Combines all panel types into a single figure layout, analogous
    to par2qo's workflow of generating multiple PDF figures per experiment.
    """
    _dbg(_T, "build_benchmark_report()")
    figure = FigureLayout(
        figure_id=report_id,
        title="Lynceus Cost Model Benchmark Report",
        n_cols=2,
    )

    # Panel 1: Calibration scatter
    figure.add_panel(build_calibration_scatter_panel(
        predicted_costs, actual_costs, debug_print=debug_print))

    # Panel 2: Speedup line (if baseline provided)
    if baseline_costs and len(baseline_costs) == len(actual_costs):
        figure.add_panel(build_speedup_line_panel(
            actual_costs, baseline_costs, debug_print=debug_print))

    # Panel 3: Worker divergence matrix
    if worker_divergence:
        figure.add_panel(build_divergence_matrix_panel(
            worker_divergence, worker_labels, debug_print=debug_print))

    # Panel 4: Config cost distribution
    if config_cost_lists:
        figure.add_panel(build_cost_distribution_panel(
            config_cost_lists, config_labels, debug_print=debug_print))

    if debug_print:
        print(f"\n  [plot_panels] Complete benchmark report:")
        print(figure.dump_debug("    "))

    return figure
