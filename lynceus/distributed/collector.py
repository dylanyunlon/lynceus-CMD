"""
lynceus/distributed/collector.py — All-reduce statistics collector.

算法改动:
    1. Welford 在线 variance (单遍, 数值稳定)
    2. empirical_kl → 默认 JSD (对称)
    3. divergence_matrix → 离群 worker 检测
"""
from __future__ import annotations
import math
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum, auto
from .sync import SyncConfig, SyncStrategy, estimate_sync_cost

logger = logging.getLogger(__name__)

class StatisticKind(Enum):
    QUERY_LATENCY = auto()
    HARDWARE_UTIL = auto()
    TRANSFER_COST = auto()
    INDEX_BUILD_COST = auto()
    SCAN_COST = auto()
    CALIBRATION_ERROR = auto()

@dataclass
class StatisticSample:
    kind: StatisticKind
    worker_id: str
    timestamp_us: float
    value: float
    dimension: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    def dump_debug(self, prefix: str = "") -> str:
        return (f"{prefix}[Sample] {self.kind.name} w={self.worker_id} "
                f"dim={self.dimension} val={self.value:.4f}")

@dataclass
class AggregatedStatistic:
    kind: StatisticKind
    dimension: str
    n_samples: int
    mean: float = 0.0
    variance: float = 0.0
    min_val: float = float('inf')
    max_val: float = float('-inf')
    per_worker_means: Dict[str, float] = field(default_factory=dict)
    rel_error_mean: float = 0.0
    rel_error_max: float = 0.0
    def dump_debug(self, prefix: str = "") -> str:
        std = math.sqrt(max(0, self.variance))
        return (f"{prefix}Agg({self.kind.name}:{self.dimension}): "
                f"n={self.n_samples} mean={self.mean:.4f}±{std:.4f} "
                f"[{self.min_val:.4f}, {self.max_val:.4f}]")


class CollectionBuffer:
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
        sample = StatisticSample(
            kind=kind, worker_id=self._worker_id,
            timestamp_us=time.time() * 1e6, value=value,
            dimension=dimension, metadata=metadata or {})
        self._buffer.append(sample)
        self._total_collected += 1

    def flush(self) -> List[StatisticSample]:
        samples = self._buffer.copy()
        self._buffer.clear()
        self._flush_count += 1
        if self._debug:
            print(f"  [collector] w={self._worker_id} FLUSH #{self._flush_count}: "
                  f"{len(samples)} samples")
        return samples

    @property
    def size(self) -> int:
        return len(self._buffer)
    @property
    def is_full(self) -> bool:
        return len(self._buffer) >= self._max_size


def cal_rel_error(true_val: float, est_val: float) -> float:
    if true_val <= 0 or est_val <= 0:
        return 0.0 if (true_val <= 0 and est_val <= 0) else float('inf')
    if math.isnan(true_val) or math.isnan(est_val):
        return 0.0
    if true_val > est_val:
        return math.log(true_val / est_val)
    return -math.log(est_val / true_val)


def evenly_sample_range(n: int, lower: float, upper: float) -> List[float]:
    if n <= 1:
        return [(lower + upper) / 2.0]
    step = (upper - lower) / (n - 1)
    return [lower + i * step for i in range(n)]


def top_n_of_matrix(matrix: List[List[float]], n: int,
                    debug_print: bool = True) -> List[Tuple[int, int, float]]:
    entries = []
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            entries.append((i, j, val if not math.isnan(val) else 0.0, abs(val if not math.isnan(val) else 0.0)))
    entries.sort(key=lambda x: x[3], reverse=True)
    results = [(i, j, val) for i, j, val, _ in entries[:n]]
    if debug_print:
        for idx, (i, j, val) in enumerate(results):
            print(f"  [collector] top-{idx+1}: ({i}, {j}) = {val:.6f}")
    return results


def empirical_kl_divergence(p_counts: List[float], q_counts: List[float],
                            epsilon: float = 1e-10) -> float:
    assert len(p_counts) == len(q_counts)
    n = len(p_counts)
    p_sum = sum(p_counts) + epsilon * n
    q_sum = sum(q_counts) + epsilon * n
    kl = 0.0
    for i in range(n):
        p_i = (p_counts[i] + epsilon) / p_sum
        q_i = (q_counts[i] + epsilon) / q_sum
        kl += p_i * math.log(p_i / q_i)
    return kl


def jensen_shannon_divergence(p: List[float], q: List[float],
                              epsilon: float = 1e-10) -> float:
    """改动: 对称版 — JSD = 0.5*KL(P||M) + 0.5*KL(Q||M)。
    原版 collector 只有单向 KL(P||Q), 不对称, 方向选择影响结果。
    JSD 对称且 bounded [0, log2], 更适合比较 worker 间差异。
    """
    n = len(p)
    m = [(p[i] + q[i]) / 2.0 for i in range(n)]
    return 0.5 * empirical_kl_divergence(p, m, epsilon) + \
           0.5 * empirical_kl_divergence(q, m, epsilon)


def _welford_stats(values: List[float]) -> Tuple[float, float, float, float]:
    """改动: Welford 在线算法, 单遍计算 mean 和 variance。
    原版: mean = sum/n, var = sum((v-mean)^2)/(n-1) — 两遍
    新版: 单遍, 数值更稳定 (大 mean 小 variance 场景不会 catastrophic cancellation)
    """
    n = 0
    mean = 0.0
    m2 = 0.0
    min_v = float('inf')
    max_v = float('-inf')
    for x in values:
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        m2 += delta * delta2
        min_v = min(min_v, x)
        max_v = max(max_v, x)
    variance = m2 / max(1, n - 1) if n > 1 else 0.0
    return mean, variance, min_v, max_v


class AllReduceCollector:
    def __init__(self, worker_ids: Optional[List[str]] = None,
                 sync_config: Optional[SyncConfig] = None,
                 debug_print: bool = True):
        self._worker_ids = worker_ids or ["worker_0"]
        self._buffers: Dict[str, CollectionBuffer] = {
            wid: CollectionBuffer(wid, debug_print=debug_print)
            for wid in self._worker_ids}
        self._sync_config = sync_config or SyncConfig(
            n_workers=len(self._worker_ids), debug_print=debug_print)
        self._aggregated: Dict[str, AggregatedStatistic] = {}
        self._collection_rounds = 0
        self._debug = debug_print
        if debug_print:
            print(f"\n[collector] Initialized: {self._worker_ids}")

    def record(self, worker_id: str, kind: StatisticKind, value: float,
               dimension: str = "", metadata: Optional[Dict] = None) -> None:
        buf = self._buffers.get(worker_id)
        if buf is None:
            return
        buf.add(kind, value, dimension, metadata)

    def collect_and_aggregate(self, predicted_values: Optional[Dict[str, float]] = None,
                              debug_print: Optional[bool] = None) -> Dict[str, AggregatedStatistic]:
        dp = debug_print if debug_print is not None else self._debug
        self._collection_rounds += 1
        all_samples: List[StatisticSample] = []
        for wid, buf in self._buffers.items():
            all_samples.extend(buf.flush())
        if not all_samples:
            return self._aggregated

        groups: Dict[str, List[StatisticSample]] = {}
        for s in all_samples:
            key = f"{s.kind.name}:{s.dimension}"
            groups.setdefault(key, []).append(s)

        for key, samples in groups.items():
            kind = samples[0].kind
            dim = samples[0].dimension
            values = [s.value for s in samples]

            # 改动: Welford 单遍
            mean, variance, min_val, max_val = _welford_stats(values)

            # per-worker means (也用 Welford)
            worker_accum: Dict[str, Tuple[int, float, float]] = {}  # n, mean, m2
            for s in samples:
                if s.worker_id not in worker_accum:
                    worker_accum[s.worker_id] = (0, 0.0, 0.0)
                n_w, mean_w, m2_w = worker_accum[s.worker_id]
                n_w += 1
                d1 = s.value - mean_w
                mean_w += d1 / n_w
                d2 = s.value - mean_w
                m2_w += d1 * d2
                worker_accum[s.worker_id] = (n_w, mean_w, m2_w)
            per_worker_means = {wid: acc[1] for wid, acc in worker_accum.items()}

            rel_errors = []
            if predicted_values and dim in predicted_values:
                pred = predicted_values[dim]
                rel_errors = [cal_rel_error(v, pred) for v in values]

            agg = AggregatedStatistic(
                kind=kind, dimension=dim, n_samples=len(values),
                mean=mean, variance=variance, min_val=min_val, max_val=max_val,
                per_worker_means=per_worker_means,
                rel_error_mean=sum(rel_errors) / max(1, len(rel_errors)) if rel_errors else 0.0,
                rel_error_max=max(abs(e) for e in rel_errors) if rel_errors else 0.0)
            self._aggregated[key] = agg

            if dp:
                print(f"  {agg.dump_debug()}")

        total_bytes = len(self._aggregated) * 64
        estimate_sync_cost(data_bytes=total_bytes, config=self._sync_config,
                          debug_print=dp)
        return self._aggregated

    def get_divergence_matrix(self, kind: StatisticKind,
                              debug_print: bool = True) -> List[List[float]]:
        """改动: 用 JSD 代替单向 KL, 加离群 worker 检测。"""
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
                    min_all = min(min(vals_i), min(vals_j))
                    shift = abs(min_all) + 1.0 if min_all <= 0 else 0.0
                    p = [v + shift for v in vals_i]
                    q = [v + shift for v in vals_j]
                    # 改动: JSD 代替 KL
                    matrix[i][j] = jensen_shannon_divergence(p, q)

        if debug_print:
            # 改动: 离群 worker 检测
            row_sums = [sum(matrix[i][j] for j in range(n) if j != i) / max(1, n-1)
                       for i in range(n)]
            sorted_sums = sorted(row_sums)
            median_div = sorted_sums[len(sorted_sums) // 2] if sorted_sums else 0.0
            outlier_threshold = 2.0 * median_div if median_div > 0 else float('inf')

            print(f"\n  [collector] JSD Matrix ({kind.name}):")
            for i, wid in enumerate(self._worker_ids):
                is_outlier = row_sums[i] > outlier_threshold
                marker = " ⚠ OUTLIER" if is_outlier else ""
                row = " ".join(f"{matrix[i][j]:.4f}" for j in range(n))
                print(f"    {wid}: {row}  avg={row_sums[i]:.4f}{marker}")

        return matrix

    def dump_state(self) -> str:
        lines = [f"AllReduceCollector: {len(self._worker_ids)} workers, "
                 f"{self._collection_rounds} rounds, "
                 f"{len(self._aggregated)} stats"]
        return "\n".join(lines)

    def export_json(self) -> str:
        export = {}
        for key, agg in self._aggregated.items():
            export[key] = {"kind": agg.kind.name, "dim": agg.dimension,
                          "n": agg.n_samples, "mean": agg.mean,
                          "var": agg.variance, "per_worker": agg.per_worker_means}
        return json.dumps(export, indent=2)
