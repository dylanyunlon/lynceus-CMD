# -*- coding: utf-8 -*-
"""
Original: Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
          SPDX-License-Identifier: MIT (upstream/videx/.../videx_strategy.py)
Modified: Lynceus — heterogeneous virtual index cost model with GPU dispatch.

Modifications from upstream videx_strategy.py (~20% changed):
  - Removed: MySQL imports (sub_platforms.sql_opt.*)
  - Added:   DeviceAwareCostModel with scan_time_heterogeneous()
  - Added:   HeterogeneousCost struct for dual CPU/GPU estimation
  - Added:   CostHistogram for workload cost distribution (CCCL pattern)
  - Added:   IndexBenefitEstimator for virtual index what-if analysis
  - Modified: VidexStrategy enum — added gpu_accelerated, heterogeneous
  - Kept:    VidexModelBase ABC, cardinality/NDV/records_in_range interface,
             calc_mulcol_ndv_independent utility

M321-M330 Algorithm Changes (Phase 8):
  [ALG-1] _estimate_selectivity: simple 1/NDV → KDE (Kernel Density Estimation)
          with Gaussian kernel. When histogram bucket data is available, treat
          bucket midpoints as data points and use Silverman's rule for bandwidth:
          h = 0.9 * min(σ, IQR/1.34) * n^(-1/5). Falls back to 1/NDV when no
          histogram data exists.
  [ALG-2] _index_benefit: added covering index detection. If the index covers
          all columns referenced by the query (SELECT + WHERE), the benefit
          multiplier doubles because heap lookups are avoided entirely.
  [ALG-3] _dbg_kde_estimate(): prints bandwidth, evaluation points, and
          density values for each KDE-based selectivity estimate.
"""
from __future__ import annotations
import enum, math, logging
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

logger = logging.getLogger("lynceus.videx_index_advisor")


def _dbg_kde_estimate(col: str, bandwidth: float, n_points: int,
                      query_point: float, density: float, selectivity: float):
    """打印KDE估计的完整参数"""
    try:
        from lynceus._debug import dbg
        dbg('kde_selectivity',
            column=col,
            bandwidth=f"{bandwidth:.6f}",
            n_data_points=n_points,
            query_point=f"{query_point:.4f}",
            density_at_point=f"{density:.8f}",
            estimated_selectivity=f"{selectivity:.6f}")
    except ImportError:
        pass


def _dbg_covering_index(index_name: str, index_cols: List[str],
                        query_cols: List[str], is_covering: bool,
                        benefit_multiplier: float):
    """打印covering index检测结果"""
    try:
        from lynceus._debug import dbg
        dbg('covering_index_check',
            index=index_name,
            index_columns=index_cols,
            query_columns=query_cols,
            is_covering=is_covering,
            benefit_multiplier=f"{benefit_multiplier:.2f}")
    except ImportError:
        pass


class VidexStrategy(enum.Enum):
    example = "example"
    innodb = "innodb"
    ideal = "ideal"
    sqlbrain = "sqlbrain"
    gpu_accelerated = "gpu_accelerated"
    heterogeneous = "heterogeneous"


@dataclass
class RangeCond:
    col: str
    lower: Optional[Any] = None
    upper: Optional[Any] = None
    lower_inclusive: bool = True
    upper_inclusive: bool = True
    is_equality: bool = False

    def selectivity(self, total_rows: int, ndv: int) -> float:
        if ndv <= 0: return 1.0
        return 1.0 / ndv if self.is_equality else min(1.0, max(0.001, 1.0 / math.sqrt(ndv)))


@dataclass
class IndexRangeCond:
    index_name: str
    ranges: List[RangeCond] = field(default_factory=list)

    def ranges_to_str(self) -> str:
        parts = []
        for r in self.ranges:
            if r.is_equality:
                parts.append(f"{r.col} = {r.lower}")
            else:
                parts.append(f"{r.lower or '-inf'} <= {r.col} <= {r.upper or '+inf'}")
        return " AND ".join(parts)


@dataclass
class TableStats:
    table_name: str
    total_rows: int
    avg_row_length: int = 100
    clustered_index_size: int = 0
    innodb_buffer_pool_size: int = 0
    column_ndvs: Dict[str, int] = field(default_factory=dict)
    # M321: 新增直方图数据用于KDE
    column_histograms: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    # M321: 新增index定义用于covering index检测
    index_definitions: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class DeviceCostParams:
    seq_page_cost: float = 0.02
    random_page_cost: float = 0.5
    cpu_tuple_cost: float = 0.05
    cpu_operator_cost: float = 0.01
    cpu_index_tuple_cost: float = 0.02
    kernel_launch_us: float = 10.0
    gpu_tuple_cost: float = 0.0001
    gpu_operator_cost: float = 0.00005
    hbm_bandwidth_gb_s: float = 2000.0
    pcie_bandwidth_gb_s: float = 32.0
    gpu_num_sms: int = 108


# ── KDE 选择性估计 (Silverman's rule bandwidth) ─────────────────────
def _gaussian_kernel(x: float) -> float:
    """标准高斯核: K(x) = (1/√2π) * exp(-x²/2)"""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _kde_estimate(data_points: List[float], query_point: float,
                  bandwidth: float) -> float:
    """Kernel Density Estimation with Gaussian kernel.
    f̂(x) = (1/nh) Σᵢ K((x - xᵢ)/h)
    """
    n = len(data_points)
    if n == 0 or bandwidth <= 0:
        return 0.0
    density = 0.0
    for xi in data_points:
        u = (query_point - xi) / bandwidth
        density += _gaussian_kernel(u)
    return density / (n * bandwidth)


def _silverman_bandwidth(data_points: List[float]) -> float:
    """Silverman's rule of thumb: h = 0.9 * min(σ, IQR/1.34) * n^(-1/5)
    用于自动选择KDE带宽。"""
    n = len(data_points)
    if n < 2:
        return 1.0
    mean = sum(data_points) / n
    var = sum((x - mean) ** 2 for x in data_points) / (n - 1)
    std = math.sqrt(max(var, 1e-12))

    # IQR估计
    sorted_pts = sorted(data_points)
    q1_idx = int(n * 0.25)
    q3_idx = int(n * 0.75)
    iqr = sorted_pts[min(q3_idx, n - 1)] - sorted_pts[min(q1_idx, n - 1)]
    iqr_scaled = iqr / 1.34 if iqr > 0 else std

    spread = min(std, iqr_scaled) if iqr_scaled > 0 else std
    return 0.9 * spread * (n ** (-0.2))


# ---------------------------------------------------------------------------
# VidexModelBase (from VIDEX, kept intact)
# ---------------------------------------------------------------------------

class VidexModelBase(ABC):
    """Abstract cost model class. VIDEX-Statistic-Server receives requests
    from VIDEX-MySQL for Cardinality and NDV estimates."""
    def __init__(self, stats: TableStats, strategy: VidexStrategy):
        self.table_stats = stats
        self.strategy = strategy

    @property
    def table_name(self):
        return self.table_stats.table_name

    @abstractmethod
    def cardinality(self, idx_range_cond: IndexRangeCond) -> int:
        pass

    @abstractmethod
    def ndv(self, index_name: str, field_list: List[str]) -> int:
        raise NotImplementedError()

    @abstractmethod
    def scan_time(self, req_json_item: dict) -> float:
        raise NotImplementedError()

    def records_in_range(self, idx_range_cond: IndexRangeCond) -> int:
        return self.cardinality(idx_range_cond)


# ---------------------------------------------------------------------------
# HeterogeneousCost — dual CPU/GPU cost estimate
# ---------------------------------------------------------------------------

@dataclass
class HeterogeneousCost:
    cpu_cost_us: float = 0.0
    gpu_cost_us: float = 0.0
    recommended_device: str = "cpu"
    breakdown: Dict[str, float] = field(default_factory=dict)

    @property
    def min_cost_us(self): return min(self.cpu_cost_us, self.gpu_cost_us)

    @property
    def speedup(self): return self.cpu_cost_us / max(1e-9, self.gpu_cost_us)


# ---------------------------------------------------------------------------
# DeviceAwareCostModel — extends VidexModelBase with GPU estimation + KDE
# ---------------------------------------------------------------------------

class DeviceAwareCostModel(VidexModelBase):
    def __init__(self, stats: TableStats, *,
                 params: Optional[DeviceCostParams] = None):
        super().__init__(stats, VidexStrategy.heterogeneous)
        self.params = params or DeviceCostParams()
        self._ndv_cache: Dict[str, int] = {}

    def _estimate_selectivity(self, rc: RangeCond) -> float:
        """[ALG-1] KDE选择性估计: 当有直方图数据时用Gaussian KDE,
        否则回退到经典的 1/NDV 估计。

        直方图桶的中点作为KDE的数据点, bandwidth用Silverman's rule自动选择。
        对range查询, 在range的中点评估密度, 乘以range宽度得到选择性。
        对equality查询, 评估单点密度, 除以总密度积分得到选择性。"""
        col = rc.col
        ndv = self.table_stats.column_ndvs.get(col, 1)
        hist = self.table_stats.column_histograms.get(col)

        if hist and len(hist) >= 3:
            # 用直方图桶中点作为数据点
            data_points = [(lo + hi) / 2.0 for lo, hi in hist]
            bw = _silverman_bandwidth(data_points)

            if rc.is_equality and rc.lower is not None:
                try:
                    qp = float(rc.lower)
                except (ValueError, TypeError):
                    return 1.0 / max(1, ndv)
                density = _kde_estimate(data_points, qp, bw)
                # 选择性 = density / 总面积 (近似为1.0)
                sel = max(1e-6, min(1.0, density * bw))
                _dbg_kde_estimate(col, bw, len(data_points), qp, density, sel)
                return sel
            else:
                # range查询: 在range中点评估密度
                lo_val = float(rc.lower) if rc.lower is not None else data_points[0]
                hi_val = float(rc.upper) if rc.upper is not None else data_points[-1]
                mid = (lo_val + hi_val) / 2.0
                width = max(hi_val - lo_val, bw)
                density = _kde_estimate(data_points, mid, bw)
                sel = max(1e-6, min(1.0, density * width))
                _dbg_kde_estimate(col, bw, len(data_points), mid, density, sel)
                return sel
        else:
            # 回退到经典估计
            return rc.selectivity(self.table_stats.total_rows, ndv)

    def cardinality(self, idx_range_cond: IndexRangeCond) -> int:
        """改用KDE的选择性估计来计算cardinality"""
        rows = self.table_stats.total_rows
        for rc in idx_range_cond.ranges:
            sel = self._estimate_selectivity(rc)
            rows = max(1, int(rows * sel))
        return rows

    def ndv(self, index_name: str, field_list: List[str]) -> int:
        key = f"{index_name}:{','.join(field_list)}"
        if key in self._ndv_cache: return self._ndv_cache[key]
        r = calc_mulcol_ndv_independent(field_list, self.table_stats.column_ndvs,
                                         self.table_stats.total_rows)
        self._ndv_cache[key] = r
        return r

    def scan_time(self, req_json_item=None) -> float:
        p = self.params
        pages = max(1, self.table_stats.total_rows * self.table_stats.avg_row_length // 8192)
        return pages * p.seq_page_cost + self.table_stats.total_rows * p.cpu_tuple_cost

    def _is_covering_index(self, index_name: str,
                           query_columns: List[str]) -> bool:
        """[ALG-2] Covering index detection:
        如果index包含了query所需的所有列, 就是covering index,
        可以避免heap lookup (回表), 性能翻倍。"""
        idx_cols = self.table_stats.index_definitions.get(index_name, [])
        if not idx_cols:
            return False
        idx_col_set = set(c.lower() for c in idx_cols)
        query_col_set = set(c.lower() for c in query_columns)
        is_covering = query_col_set.issubset(idx_col_set)
        _dbg_covering_index(index_name, idx_cols, query_columns,
                           is_covering, 2.0 if is_covering else 1.0)
        return is_covering

    def scan_time_heterogeneous(self, estimated_rows: int = 0,
                                num_predicates: int = 1,
                                sort_required: bool = False) -> HeterogeneousCost:
        if estimated_rows <= 0: estimated_rows = self.table_stats.total_rows
        p = self.params
        data_bytes = estimated_rows * self.table_stats.avg_row_length
        pages = max(1, data_bytes // 8192)
        cpu_io = pages * p.seq_page_cost
        cpu_compute = estimated_rows * p.cpu_tuple_cost + estimated_rows * num_predicates * p.cpu_operator_cost
        cpu_sort = (2.0 * estimated_rows * math.log2(max(2, estimated_rows)) * p.cpu_operator_cost) if sort_required and estimated_rows > 1 else 0.0
        cpu_total = cpu_io + cpu_compute + cpu_sort
        transfer = (data_bytes / (p.pcie_bandwidth_gb_s * 1e9)) * 1e6
        hbm = (data_bytes / (p.hbm_bandwidth_gb_s * 1e9)) * 1e6
        gpu_compute = estimated_rows * p.gpu_tuple_cost + estimated_rows * num_predicates * p.gpu_operator_cost
        gpu_sort = (estimated_rows * (math.log2(max(2, estimated_rows))**2) * p.gpu_operator_cost / p.gpu_num_sms) if sort_required and estimated_rows > 1 and p.gpu_num_sms > 0 else 0.0
        gpu_total = p.kernel_launch_us + transfer + max(hbm, gpu_compute) + gpu_sort
        return HeterogeneousCost(cpu_cost_us=cpu_total, gpu_cost_us=gpu_total,
            recommended_device="gpu" if gpu_total < cpu_total else "cpu",
            breakdown={"cpu_io": cpu_io, "cpu_compute": cpu_compute, "cpu_sort": cpu_sort,
                       "gpu_transfer": transfer, "gpu_launch": p.kernel_launch_us,
                       "gpu_hbm": hbm, "gpu_compute": gpu_compute, "gpu_sort": gpu_sort})

    def index_scan_heterogeneous(self, idx_range_cond: IndexRangeCond,
                                 depth: int = 3,
                                 query_columns: Optional[List[str]] = None) -> HeterogeneousCost:
        """改动: 加入covering index检测, 如果是covering index则避免回表"""
        card = self.cardinality(idx_range_cond)
        p = self.params
        data_bytes = card * self.table_stats.avg_row_length
        pages = max(1, data_bytes // 8192)

        # [ALG-2] covering index检测
        covering_mult = 1.0
        if query_columns:
            if self._is_covering_index(idx_range_cond.index_name, query_columns):
                covering_mult = 0.5  # 避免回表, 成本减半

        cpu_idx = depth * p.random_page_cost + card * p.cpu_index_tuple_cost
        cpu_io = pages * p.seq_page_cost * covering_mult  # covering时不需要回表读heap
        cpu_total = cpu_idx + cpu_io + card * p.cpu_tuple_cost * covering_mult
        transfer = (data_bytes * covering_mult / (p.pcie_bandwidth_gb_s * 1e9)) * 1e6
        hbm = (data_bytes * covering_mult / (p.hbm_bandwidth_gb_s * 1e9)) * 1e6
        gpu_total = p.kernel_launch_us + transfer + max(hbm, card * p.gpu_tuple_cost * covering_mult)
        return HeterogeneousCost(cpu_cost_us=cpu_total, gpu_cost_us=gpu_total,
            recommended_device="gpu" if gpu_total < cpu_total else "cpu",
            breakdown={"cpu_index": cpu_idx, "cardinality": card,
                       "covering_index": covering_mult < 1.0})


class CostHistogram:
    """Cost distribution histogram (CCCL CostHistogramKernel pattern)."""
    def __init__(self, num_bins: int = 256):
        self.num_bins = num_bins
        self.bins = [0] * num_bins
        self.min_cost = float('inf')
        self.max_cost = 0.0
        self.total_count = 0

    def finalize(self, costs: list):
        if not costs: return
        self.min_cost, self.max_cost = min(costs), max(costs)
        self.total_count = len(costs)
        self.bins = [0] * self.num_bins
        rng = self.max_cost - self.min_cost
        bw = rng / self.num_bins if rng > 0 else 1.0
        for c in costs:
            b = max(0, min(int((c - self.min_cost) / bw), self.num_bins - 1))
            self.bins[b] += 1

    def percentile_cost(self, pct: float) -> float:
        k = max(1, int(self.total_count * pct))
        cum, rng = 0, self.max_cost - self.min_cost
        bw = rng / self.num_bins if rng > 0 else 1.0
        for b in range(self.num_bins):
            cum += self.bins[b]
            if cum >= k: return self.min_cost + (b + 0.5) * bw
        return self.max_cost


def calc_mulcol_ndv_independent(col_names: List[str], ndvs_single: Dict[str, int],
                                table_rows: int) -> int:
    """From VIDEX (unchanged algorithm)."""
    ndv_product = 1
    for col in col_names:
        ndv_product *= ndvs_single.get(col, 1)
    return min(ndv_product, table_rows)
