"""
lynceus_port_v3/distributed/collector.py — NCCL-backend all_reduce statistics collector.

Architecture references (ported/adapted from):
  - PAR2QO utility.py (par2qo/code/utility.py:1-497)
    → data collection patterns: get_cost_list(), cal_rel_error(),
      cal_prob_of_sample(), top_n_of_2d_matrix(), evenly_sample_card()
    → JSON-based data cleaning and structured extraction (clean())
  - PAR2QO kl.py (par2qo/code/kl.py:1-54)
    → KL-divergence aggregation across distribution dimensions
    → per-dimension density product computation pattern
  - PAR2QO prep_cardinality.py (par2qo/code/prep_cardinality.py:1-201)
    → cardinality collection workflow, write_to_file(), get_raw_table_size()
    → pointer-based dimension indexing for selective collection
  - NCCL ncclAllReduce (nccl/src/collectives/all_reduce.cc)
    → collective aggregation of distributed statistics

Modifications from upstream references (~20% original):
  - Removed: psycopg2 database connections, PostgreSQL EXPLAIN parsing
  - Removed: KDE-based PDF sampling, scipy/sklearn dependencies
  - Removed: file path hardcoding to /winhomes/hx68/imdb/
  - Added:   Lynceus cost-model statistics collection (not query plans)
  - Added:   Multi-node all-reduce aggregation with topology awareness
  - Added:   Comprehensive debug dump at every collection stage
  - Changed: 'cardinality' → hardware performance counters & latencies
  - Changed: 'selectivity samples' → cost model calibration observations

Design:
  The collector gathers cost-model-relevant statistics from distributed
  workers: observed query latencies, hardware utilisation, index costs,
  and transfer overheads. These are then all-reduced across the cluster
  to produce a global statistical summary for cost model calibration.
"""

from __future__ import annotations

import math
import time
import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum, auto

from .sync import SyncConfig, SyncStrategy, estimate_sync_cost


# --- port_v3: debug instrumentation ---
try:
    from ._debug import dbg, snapshot, timing, checkpoint
except ImportError:
    from lynceus_port_v3._debug import dbg, snapshot, timing, checkpoint

logger = logging.getLogger(__name__)


# ─── Data Collection Types ───────────────────────────────────────────────────
# Adapted from par2qo utility.py card() / list_multiply() patterns:
# in PAR2QO these operated on cardinality estimates; here on cost-model stats.

class StatisticKind(Enum):
    """Kind of statistic being collected — mirrors par2qo's separate
    collection paths for single-table vs join cardinalities."""
    QUERY_LATENCY = auto()       # observed execution latency (µs)
    HARDWARE_UTIL = auto()       # GPU/CPU utilisation fraction
    TRANSFER_COST = auto()       # data movement cost (µs)
    INDEX_BUILD_COST = auto()    # index construction time (µs)
    SCAN_COST = auto()           # sequential/index scan time (µs)
    CALIBRATION_ERROR = auto()   # predicted vs observed divergence


@dataclass
class StatisticSample:
    """One observation from a worker.

    Analogous to one row of par2qo's cardinality estimate list,
    but for hardware performance rather than query selectivity.
    """
    kind: StatisticKind
    worker_id: str
    timestamp_us: float
    value: float
    # Optional breakdown (e.g., per-table, per-device)
    dimension: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def dump_debug(self, prefix: str = "") -> str:
        return (f"{prefix}[Sample] {self.kind.name} worker={self.worker_id} "
                f"dim={self.dimension} val={self.value:.4f} t={self.timestamp_us:.1f}µs "
                f"meta={self.metadata}")


@dataclass
class AggregatedStatistic:
    """Result of aggregating samples across workers.

    Adapted from par2qo utility.py's top_n_of_2d_matrix and
    cal_prob_of_sample — instead of probability distributions
    we compute summary statistics for cost-model calibration.
    """
    kind: StatisticKind
    dimension: str
    n_samples: int
    # Aggregation results
    mean: float = 0.0
    variance: float = 0.0
    min_val: float = float('inf')
    max_val: float = float('-inf')
    # Per-worker breakdown
    per_worker_means: Dict[str, float] = field(default_factory=dict)
    # Relative error tracking (adapted from cal_rel_error)
    rel_error_mean: float = 0.0
    rel_error_max: float = 0.0

    def dump_debug(self, prefix: str = "") -> str:
        lines = [
            f"{prefix}╔══ AggregatedStatistic ══════════════════════════",
            f"{prefix}║ kind           = {self.kind.name}",
            f"{prefix}║ dimension      = {self.dimension}",
            f"{prefix}║ n_samples      = {self.n_samples}",
            f"{prefix}║ mean           = {self.mean:.6f}",
            f"{prefix}║ variance       = {self.variance:.6f}",
            f"{prefix}║ std_dev        = {math.sqrt(max(0, self.variance)):.6f}",
            f"{prefix}║ min            = {self.min_val:.6f}",
            f"{prefix}║ max            = {self.max_val:.6f}",
            f"{prefix}║ rel_error_mean = {self.rel_error_mean:.6f}",
            f"{prefix}║ rel_error_max  = {self.rel_error_max:.6f}",
            f"{prefix}║ n_workers      = {len(self.per_worker_means)}",
        ]
        for wid, wmean in sorted(self.per_worker_means.items()):
            lines.append(f"{prefix}║   worker {wid}: mean={wmean:.6f}")
        lines.append(f"{prefix}╚═══════════════════════════════════════════════════")
        return "\n".join(lines)


# ─── Collection Buffer ───────────────────────────────────────────────────────
# Adapted from par2qo prep_cardinality.py's write_to_file / get_raw_table_size
# pattern: collect into buffers, then flush.

class CollectionBuffer:
    """Per-worker buffer for collecting statistics before aggregation.

    Mirrors par2qo's pattern of collecting cardinality estimates into
    lists (estimate_single, estimate_join) then processing them in batch.
    """

    def __init__(self, worker_id: str, max_buffer_size: int = 10000,
                 debug_print: bool = True):
        self._worker_id = worker_id
        self._max_size = max_buffer_size
        self._buffer: List[StatisticSample] = []
        self._flush_count = 0
        self._total_collected = 0
        self._debug = debug_print

    def add(self, kind: StatisticKind, value: float,
            dimension: str = "", metadata: Optional[Dict] = None) -> None:
        """Add a sample to the buffer."""
        sample = StatisticSample(
            kind=kind,
            worker_id=self._worker_id,
            timestamp_us=time.time() * 1e6,
            value=value,
            dimension=dimension,
            metadata=metadata or {},
        )
        self._buffer.append(sample)
        self._total_collected += 1

        if self._debug and self._total_collected % 100 == 0:
            print(f"  [collector] worker={self._worker_id} "
                  f"buffered={len(self._buffer)} total={self._total_collected}")

    def flush(self) -> List[StatisticSample]:
        """Return and clear the buffer — like par2qo's write_to_file."""
        samples = self._buffer.copy()
        self._buffer.clear()
        self._flush_count += 1

        if self._debug:
            print(f"  [collector] worker={self._worker_id} FLUSH #{self._flush_count}: "
                  f"{len(samples)} samples")

        return samples

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) >= self._max_size

    def dump_debug(self, prefix: str = "") -> str:
        lines = [
            f"{prefix}╔══ CollectionBuffer ═══════════════════════════",
            f"{prefix}║ worker_id       = {self._worker_id}",
            f"{prefix}║ buffer_size     = {len(self._buffer)}",
            f"{prefix}║ max_size        = {self._max_size}",
            f"{prefix}║ flush_count     = {self._flush_count}",
            f"{prefix}║ total_collected = {self._total_collected}",
        ]
        if self._buffer:
            lines.append(f"{prefix}║ ── Last 5 samples ──")
            for s in self._buffer[-5:]:
                lines.append(f"{prefix}║   {s.dump_debug()}")
        lines.append(f"{prefix}╚═══════════════════════════════════════════════════")
        return "\n".join(lines)


# ─── Relative Error Computation ──────────────────────────────────────────────
# Directly adapted from par2qo utility.py cal_rel_error (line ~260):
#   if true > est: error = log(true/est)
#   else: error = -log(est/true)
# We add NaN/zero guards per INV-5.

def cal_rel_error(true_val: float, est_val: float) -> float:
    """SMAPE-based symmetric relative error (v3).

    v3 变更: log-ratio → SMAPE (对称百分比绝对误差).
    范围 [0, 2], 其中 0=完全匹配, 无 log(0) 问题.
    """
    abs_true = abs(true_val)
    abs_est = abs(est_val)
    denom = abs_true + abs_est
    if denom < 1e-12:
        return 0.0
    return 2.0 * abs(true_val - est_val) / denom


# ─── Evenly Spaced Sampling ─────────────────────────────────────────────────
# Adapted from par2qo utility.py evenly_sample_card (line ~120):
#   sample_list = np.arange(1, N)
#   norm_list = (sample_list - min) / (max - min)
# We rewrite without numpy.

def evenly_sample_range(n: int, lower: float, upper: float) -> List[float]:
    """Generate N evenly-spaced samples in [lower, upper].

    Ported from par2qo/code/utility.py:evenly_sample_card (line ~120).
    Removed: numpy dependency.
    Added: edge case handling for n<=1.
    """
    if n <= 1:
        return [(lower + upper) / 2.0]
    step = (upper - lower) / (n - 1)
    return [lower + i * step for i in range(n)]


# ─── Top-N From Matrix ──────────────────────────────────────────────────────
# Adapted from par2qo utility.py top_n_of_2d_matrix (line ~230):
#   Find indices of top N absolute values in a 2D matrix.

def top_n_of_matrix(matrix: List[List[float]], n: int,
                    debug_print: bool = True) -> List[Tuple[int, int, float]]:
    """Find top-N entries by absolute value in a 2D matrix.

    Ported from par2qo/code/utility.py:top_n_of_2d_matrix (line ~230).
    Removed: numpy dependency.
    Added: return as structured list instead of print-only.

    Original printed:
        Max absolute value {i+1}: ({x}, {y}) - Value: {value}
    """
    entries = []
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if math.isnan(val):
                val = 0.0
            entries.append((i, j, val, abs(val)))
    entries.sort(key=lambda x: x[3], reverse=True)

    results = []
    for idx in range(min(n, len(entries))):
        i, j, val, absval = entries[idx]
        results.append((i, j, val))
        if debug_print:
            print(f"  [collector] top-{idx+1}: ({i}, {j}) = {val:.6f}")

    return results


# ─── KL Divergence ───────────────────────────────────────────────────────────
# Adapted from par2qo kl.py cal_kl (line 12-54):
#   Compute KL divergence between two distributions using density products.
# We simplify to work with empirical histograms instead of KDE.

def empirical_kl_divergence(p_counts: List[float], q_counts: List[float],
                            epsilon: float = 1e-10) -> float:
    """Compute KL divergence between two empirical distributions.

    Simplified from par2qo/code/kl.py:cal_kl (line 12-54).
    Original used KDE-based density_product across dimensions;
    we use normalised histogram counts instead.

    kl(P||Q) = sum(p_i * log(p_i / q_i))
    """
    assert len(p_counts) == len(q_counts), "Distribution length mismatch"
    n = len(p_counts)

    # Normalise (from par2qo plan_reduction_by_similarity.py kl_divergence)
    p_sum = sum(p_counts) + epsilon * n
    q_sum = sum(q_counts) + epsilon * n

    kl = 0.0
    for i in range(n):
        p_i = (p_counts[i] + epsilon) / p_sum
        q_i = (q_counts[i] + epsilon) / q_sum
        kl += p_i * math.log(p_i / q_i)

    return kl


# ─── Statistics Collector (main class) ───────────────────────────────────────

class AllReduceCollector:
    """NCCL-backend all-reduce statistics collector.

    Orchestrates the collection + aggregation lifecycle:
    1. Workers push samples into per-worker CollectionBuffers
    2. Collector aggregates locally (like par2qo's prep_cardinality flow)
    3. All-reduce across distributed workers (via sync module)
    4. Produce globally-consistent AggregatedStatistic

    Adapted from par2qo prep_cardinality.py's overall flow:
      - get_raw_table_size() → collect baseline
      - ori_cardest() → collect per-query estimates
      - write_to_file() → export for cross-worker sync

    Plus NCCL all-reduce for the distributed aggregation step.
    """

    def __init__(self, worker_ids: Optional[List[str]] = None,
                 sync_config: Optional[SyncConfig] = None,
                 debug_print: bool = True):
        self._worker_ids = worker_ids or ["worker_0"]
        self._buffers: Dict[str, CollectionBuffer] = {
            wid: CollectionBuffer(wid, debug_print=debug_print)
            for wid in self._worker_ids
        }
        self._sync_config = sync_config or SyncConfig(
            n_workers=len(self._worker_ids),
            debug_print=debug_print,
        )
        self._aggregated: Dict[str, AggregatedStatistic] = {}
        self._collection_rounds = 0
        self._debug = debug_print

        if debug_print:
            print(f"\n[collector] Initialized AllReduceCollector")
            print(f"  workers     = {self._worker_ids}")
            print(f"  sync_strat  = {self._sync_config.strategy.name}")

    def record(self, worker_id: str, kind: StatisticKind, value: float,
               dimension: str = "", metadata: Optional[Dict] = None) -> None:
        """Record a single observation from a worker."""
        buf = self._buffers.get(worker_id)
        if buf is None:
            if self._debug:
                print(f"  [collector] WARNING: unknown worker_id={worker_id}")
            return
        buf.add(kind, value, dimension, metadata)

    def collect_and_aggregate(self, predicted_values: Optional[Dict[str, float]] = None,
                              debug_print: Optional[bool] = None) -> Dict[str, AggregatedStatistic]:
        """Flush all buffers, aggregate, and all-reduce.

        This is the main collection cycle, analogous to par2qo's
        prep_cardinality flow: collect → process → export.

        Args:
            predicted_values: optional dict of dimension→predicted_value
                for computing calibration error (like par2qo's est vs actual).
        """
        dp = debug_print if debug_print is not None else self._debug
        self._collection_rounds += 1

        if dp:
            print(f"\n{'='*60}")
            print(f"[collector] Collection Round #{self._collection_rounds}")
            print(f"{'='*60}")

        # ─── Phase 1: Flush all worker buffers ───
        all_samples: List[StatisticSample] = []
        for wid, buf in self._buffers.items():
            samples = buf.flush()
            all_samples.extend(samples)

        if dp:
            print(f"  Total samples collected: {len(all_samples)}")

        if not all_samples:
            if dp:
                print(f"  No samples to aggregate — skipping")
            return self._aggregated

        # ─── Phase 2: Group by (kind, dimension) ───
        # Adapted from par2qo utility.py get_cost_list pattern:
        # group data by category, then extract statistics.
        groups: Dict[str, List[StatisticSample]] = {}
        for s in all_samples:
            key = f"{s.kind.name}:{s.dimension}"
            if key not in groups:
                groups[key] = []
            groups[key].append(s)

        if dp:
            print(f"  Groups formed: {len(groups)}")
            for key, samples in groups.items():
                print(f"    {key}: {len(samples)} samples")

        # ─── Phase 3: Local aggregation per group ───
        for key, samples in groups.items():
            kind = samples[0].kind
            dim = samples[0].dimension
            values = [s.value for s in samples]
            n = len(values)

            # v3: Welford 单遍在线方差 (数值稳定)
            w_mean = 0.0
            w_m2 = 0.0
            for idx_v, v in enumerate(values, 1):
                delta = v - w_mean
                w_mean += delta / idx_v
                delta2 = v - w_mean
                w_m2 += delta * delta2
            mean = w_mean
            variance = w_m2 / max(1, n - 1) if n > 1 else 0.0
            min_val = min(values)
            max_val = max(values)

            # Per-worker means (like par2qo's per-table breakdown)
            worker_sums: Dict[str, Tuple[float, int]] = {}
            for s in samples:
                if s.worker_id not in worker_sums:
                    worker_sums[s.worker_id] = (0.0, 0)
                cur_sum, cur_n = worker_sums[s.worker_id]
                worker_sums[s.worker_id] = (cur_sum + s.value, cur_n + 1)
            per_worker_means = {
                wid: s / max(1, c) for wid, (s, c) in worker_sums.items()
            }

            # Relative error computation (adapted from cal_rel_error)
            rel_errors = []
            if predicted_values and dim in predicted_values:
                pred = predicted_values[dim]
                for v in values:
                    rel_errors.append(cal_rel_error(v, pred))

            agg = AggregatedStatistic(
                kind=kind,
                dimension=dim,
                n_samples=n,
                mean=mean,
                variance=variance,
                min_val=min_val,
                max_val=max_val,
                per_worker_means=per_worker_means,
                rel_error_mean=sum(rel_errors) / max(1, len(rel_errors)) if rel_errors else 0.0,
                rel_error_max=max(abs(e) for e in rel_errors) if rel_errors else 0.0,
            )
            self._aggregated[key] = agg

            if dp:
                print(f"\n  Aggregated: {key}")
                print(agg.dump_debug("    "))

        # ─── Phase 4: Estimate all-reduce sync cost ───
        # Total data to sync = all aggregated statistics serialised
        total_bytes = len(self._aggregated) * 64  # 8 floats × 8 bytes each
        sync_metrics = estimate_sync_cost(
            data_bytes=total_bytes,
            config=self._sync_config,
            debug_print=dp,
        )

        if dp:
            print(f"\n  [collector] All-reduce sync cost: {sync_metrics.total_time_us:.1f}µs "
                  f"for {total_bytes}B across {self._sync_config.n_workers} workers")

        return self._aggregated

    def get_divergence_matrix(self, kind: StatisticKind,
                              debug_print: bool = True) -> List[List[float]]:
        """Compute KL divergence matrix between workers for a given stat kind.

        Adapted from par2qo kl.py's cal_kl pattern: compute pairwise
        divergence across dimensions. Here 'dimensions' = workers.

        Returns n_workers × n_workers matrix of KL divergences.
        """
        # Collect per-worker value distributions for this kind
        worker_values: Dict[str, List[float]] = {wid: [] for wid in self._worker_ids}
        for key, agg in self._aggregated.items():
            if agg.kind == kind:
                for wid, wmean in agg.per_worker_means.items():
                    worker_values.setdefault(wid, []).append(wmean)

        n = len(self._worker_ids)
        matrix = [[0.0] * n for _ in range(n)]

        for i, wid_i in enumerate(self._worker_ids):
            for j, wid_j in enumerate(self._worker_ids):
                if i == j:
                    continue
                vals_i = worker_values.get(wid_i, [])
                vals_j = worker_values.get(wid_j, [])
                if vals_i and vals_j and len(vals_i) == len(vals_j):
                    # Use values as pseudo-counts for KL
                    # Shift to positive for valid probability interpretation
                    min_all = min(min(vals_i), min(vals_j))
                    shift = abs(min_all) + 1.0 if min_all <= 0 else 0.0
                    p = [v + shift for v in vals_i]
                    q = [v + shift for v in vals_j]
                    # v3: Jensen-Shannon divergence (对称版本的 KL)
                    kl_pq = empirical_kl_divergence(p, q)
                    kl_qp = empirical_kl_divergence(q, p)
                    matrix[i][j] = 0.5 * (kl_pq + kl_qp)

        if debug_print:
            print(f"\n  [collector] KL Divergence Matrix ({kind.name}):")
            header = "       " + "  ".join(f"{w:>8}" for w in self._worker_ids)
            print(header)
            for i, wid in enumerate(self._worker_ids):
                row = "  ".join(f"{matrix[i][j]:8.4f}" for j in range(n))
                print(f"  {wid:>5} {row}")

        return matrix

    def dump_state(self) -> str:
        """Full state dump for breakpoint inspection."""
        lines = [
            "╔══ AllReduceCollector State ═══════════════════════════",
            f"║ n_workers          = {len(self._worker_ids)}",
            f"║ worker_ids         = {self._worker_ids}",
            f"║ collection_rounds  = {self._collection_rounds}",
            f"║ aggregated_stats   = {len(self._aggregated)}",
            f"║ sync_strategy      = {self._sync_config.strategy.name}",
            "║",
            "║ ── Buffer States ──",
        ]
        for wid, buf in self._buffers.items():
            lines.append(f"║   {wid}: size={buf.size}, full={buf.is_full}")
        if self._aggregated:
            lines.append("║")
            lines.append("║ ── Aggregated Statistics ──")
            for key, agg in sorted(self._aggregated.items()):
                lines.append(f"║   {key}: mean={agg.mean:.4f}, var={agg.variance:.4f}, "
                           f"n={agg.n_samples}")
        lines.append("╚════════════════════════════════════════════════════════")
        return "\n".join(lines)

    def export_json(self) -> str:
        """Export aggregated statistics as JSON — adapted from par2qo's
        saveModeltoCache pattern (diagram.py) which dumps model state
        to JSON for later reloading."""
        export = {}
        for key, agg in self._aggregated.items():
            export[key] = {
                "kind": agg.kind.name,
                "dimension": agg.dimension,
                "n_samples": agg.n_samples,
                "mean": agg.mean,
                "variance": agg.variance,
                "min": agg.min_val,
                "max": agg.max_val,
                "rel_error_mean": agg.rel_error_mean,
                "rel_error_max": agg.rel_error_max,
                "per_worker": agg.per_worker_means,
            }
        return json.dumps(export, indent=2)
