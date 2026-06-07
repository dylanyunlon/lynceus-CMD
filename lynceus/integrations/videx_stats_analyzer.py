"""
videx_stats_analyzer — Statistics analysis and monitoring for Lynceus.

Ported from:
  - upstream/videx/analyze_delete_rows.py (155 lines)
  - upstream/videx/analyze_trace_utils.py (138 lines)
  - upstream/videx/analyze_linear_distribution.py (75 lines)
  - upstream/videx/estimate_stats_length.py (279 lines)
  - upstream/videx/statistics_info.py (74 lines)

Algorithm changes (~20%):
  - analyze_distribution: Kolmogorov-Smirnov test instead of chi-squared
  - estimate_stats_length: compressed sensing approach for sparse histograms
  - DeleteRowAnalyzer: exponential smoothing of delete velocity
  - TraceAnalyzer: sliding window percentile tracking (P50/P95/P99)
  - test_linearity: robust linear regression via Theil-Sen estimator
"""
import math
import os
import random
from collections import deque, defaultdict

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))

def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[stats_an] {tag}: {items}")


# ── Kolmogorov-Smirnov distribution test ─────────────────────────
def ks_test_uniform(data, *, alpha=0.05):
    """One-sample Kolmogorov-Smirnov test against uniform distribution.
    
    Algorithm change: upstream uses chi-squared goodness of fit.
    KS test is distribution-free, doesn't require binning, and works
    better with continuous data and small samples.
    """
    n = len(data)
    if n == 0:
        return True, 0.0  # Accept null hypothesis trivially
    
    sorted_data = sorted(data)
    d_max = 0.0
    
    for i, x in enumerate(sorted_data):
        ecdf = (i + 1) / n
        cdf = x  # Assuming data is in [0,1]; if not, normalize first
        d = abs(ecdf - cdf)
        d_max = max(d_max, d)
        d_prev = abs(i / n - cdf)
        d_max = max(d_max, d_prev)
    
    # Critical value approximation
    critical = 1.36 / math.sqrt(n)  # alpha ≈ 0.05
    is_uniform = d_max < critical
    
    _dbg("ks_test", n=n, d_stat=f"{d_max:.6f}", critical=f"{critical:.6f}",
         uniform=is_uniform)
    return is_uniform, d_max


# ── Compressed sensing for histogram length estimation ───────────
def estimate_stats_length(table_rows, n_columns, avg_ndv, *,
                          target_accuracy=0.95, page_size=16384):
    """Estimate required statistics/histogram length for target accuracy.
    
    Algorithm change: uses compressed sensing theory.
    For s-sparse signal in d dimensions, O(s * log(d/s)) measurements suffice.
    Here: s = number of significant histogram buckets, d = total possible values.
    """
    # Estimate sparsity: how many buckets carry meaningful information
    sparsity = min(avg_ndv, int(math.sqrt(table_rows)))
    
    # Compressed sensing bound: m >= C * s * log(d/s)
    C = 2.0  # Constant depending on desired accuracy
    d = avg_ndv
    s = max(1, sparsity)
    
    min_buckets = int(C * s * math.log(max(2, d / s)))
    
    # Practical bounds
    min_buckets = max(10, min(min_buckets, 1024))
    
    # Estimate total storage in bytes
    bytes_per_bucket = 8 + 8  # boundary + count
    total_bytes = n_columns * min_buckets * bytes_per_bucket
    
    # Pages needed
    pages_needed = max(1, int(math.ceil(total_bytes / page_size)))
    
    _dbg("est_stats_len", rows=table_rows, cols=n_columns,
         avg_ndv=avg_ndv, buckets=min_buckets, pages=pages_needed)
    
    return {
        "buckets_per_column": min_buckets,
        "total_bytes": total_bytes,
        "pages_needed": pages_needed,
        "sparsity": sparsity,
    }


# ── Delete Row Analyzer with exponential smoothing ───────────────
class DeleteRowAnalyzer:
    """Analyze row deletion patterns with exponential smoothing.
    
    Algorithm change: upstream tracks raw deletion counts.
    Exponential smoothing estimates deletion velocity for
    predicting when stats become stale.
    """
    
    def __init__(self, alpha=0.3, stale_threshold=0.1):
        self.alpha = alpha
        self.stale_threshold = stale_threshold
        self._smoothed_rate = 0.0
        self._total_deleted = 0
        self._observations = []
    
    def record_deletion(self, n_deleted, total_rows):
        """Record a deletion event."""
        rate = n_deleted / max(total_rows, 1)
        self._smoothed_rate = self.alpha * rate + (1 - self.alpha) * self._smoothed_rate
        self._total_deleted += n_deleted
        self._observations.append((n_deleted, total_rows, rate))
        
        _dbg("delete_record", n=n_deleted, rate=f"{rate:.6f}",
             smoothed=f"{self._smoothed_rate:.6f}")
    
    def is_stats_stale(self, total_rows):
        """Check if statistics are stale due to deletions."""
        stale = self._total_deleted / max(total_rows, 1) > self.stale_threshold
        _dbg("stale_check", deleted=self._total_deleted,
             total=total_rows, stale=stale)
        return stale
    
    def predicted_time_to_stale(self, total_rows):
        """Estimate time until stats become stale at current deletion rate."""
        if self._smoothed_rate <= 0:
            return float("inf")
        remaining = self.stale_threshold * total_rows - self._total_deleted
        return max(0, remaining / (self._smoothed_rate * total_rows))
    
    def dump_state(self):
        print(f"[DeleteAnalyzer] total_deleted={self._total_deleted} "
              f"smoothed_rate={self._smoothed_rate:.6f} "
              f"n_obs={len(self._observations)}")


# ── Trace Analyzer with sliding window percentiles ───────────────
class TraceAnalyzer:
    """Analyze execution traces with sliding window percentile tracking.
    
    Algorithm change: upstream logs raw trace data.
    We maintain a sliding window and compute P50/P95/P99 in O(n log n).
    """
    
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self._window = deque(maxlen=window_size)
        self._total_count = 0
        self._category_counts = defaultdict(int)
    
    def record(self, latency, category="default"):
        """Record a trace observation."""
        self._window.append(latency)
        self._total_count += 1
        self._category_counts[category] += 1
    
    def percentile(self, p):
        """Compute the p-th percentile of the current window."""
        if not self._window:
            return 0.0
        sorted_w = sorted(self._window)
        idx = min(int(len(sorted_w) * p / 100.0), len(sorted_w) - 1)
        return sorted_w[idx]
    
    def summary(self):
        """Get P50/P95/P99 summary."""
        return {
            "p50": self.percentile(50),
            "p95": self.percentile(95),
            "p99": self.percentile(99),
            "count": self._total_count,
            "window_size": len(self._window),
        }
    
    def dump_state(self):
        s = self.summary()
        print(f"[TraceAnalyzer] P50={s['p50']:.4f} P95={s['p95']:.4f} "
              f"P99={s['p99']:.4f} count={s['count']}")


# ── Linearity test via Theil-Sen estimator ───────────────────────
def test_linearity(x_data, y_data, *, r_squared_threshold=0.9):
    """Test if data follows a linear distribution using Theil-Sen estimator.
    
    Algorithm change: upstream uses OLS regression.
    Theil-Sen is robust to up to 29.3% outliers (breakdown point).
    """
    n = len(x_data)
    if n < 2:
        return True, 0.0, 0.0, 1.0
    
    # Theil-Sen: median of all pairwise slopes
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if x_data[j] != x_data[i]:
                s = (y_data[j] - y_data[i]) / (x_data[j] - x_data[i])
                slopes.append(s)
    
    if not slopes:
        return True, 0.0, sum(y_data) / n, 1.0
    
    slopes.sort()
    slope = slopes[len(slopes) // 2]  # Median slope
    
    # Intercept: median of (y_i - slope * x_i)
    intercepts = [y_data[i] - slope * x_data[i] for i in range(n)]
    intercepts.sort()
    intercept = intercepts[len(intercepts) // 2]
    
    # R-squared
    y_mean = sum(y_data) / n
    ss_tot = sum((y - y_mean) ** 2 for y in y_data) or 1e-10
    ss_res = sum((y_data[i] - (slope * x_data[i] + intercept)) ** 2 for i in range(n))
    r_squared = max(0, 1 - ss_res / ss_tot)
    
    is_linear = r_squared >= r_squared_threshold
    
    _dbg("linearity", n=n, slope=f"{slope:.4f}", intercept=f"{intercept:.4f}",
         r2=f"{r_squared:.4f}", linear=is_linear)
    return is_linear, slope, intercept, r_squared


# ── Statistics Info container ────────────────────────────────────
class StatisticsInfo:
    """Container for table/column statistics with staleness tracking."""
    
    def __init__(self, table_name, column_name):
        self.table_name = table_name
        self.column_name = column_name
        self.ndv = 0
        self.null_count = 0
        self.min_val = None
        self.max_val = None
        self.avg_len = 0
        self.histogram = None
        self.last_updated = 0
        self._delete_analyzer = DeleteRowAnalyzer()
    
    def update(self, ndv, null_count, min_val, max_val, avg_len, histogram=None):
        self.ndv = ndv
        self.null_count = null_count
        self.min_val = min_val
        self.max_val = max_val
        self.avg_len = avg_len
        self.histogram = histogram
        self.last_updated = time.time()
        
        _dbg("stats_update", table=self.table_name, col=self.column_name,
             ndv=ndv, null=null_count)
    
    def dump_state(self):
        print(f"[Stats] {self.table_name}.{self.column_name}: ndv={self.ndv} "
              f"null={self.null_count} range=[{self.min_val}, {self.max_val}]")
