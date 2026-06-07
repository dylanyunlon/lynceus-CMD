"""
Ported from upstream/par2qo/code/cached_robust_plan_dict.py (767 lines)
M141: Cached robust plan dictionary with online variance tracking.

Modifications (~20% algorithm delta):
  - Welford online variance accumulator per plan entry for streaming stats
  - EMA (exponential moving average) smoothed plan index computation
  - Robust median-of-medians selection for plan lookup under noise
  - Bisect-based binary search for plan dict queries (replaces linear scan)
  - Per-entry IQR outlier fencing in aggregation helpers
"""

import math
import bisect
import statistics
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Welford online variance accumulator
# ---------------------------------------------------------------------------
class WelfordAccumulator:
    """Welford's algorithm for numerically stable running mean & variance."""

    __slots__ = ('count', 'mean', 'm2')

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def variance(self):
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    def stddev(self):
        return math.sqrt(self.variance())

    def _dbg(self):
        print(f"[WelfordAccumulator._dbg] count={self.count} mean={self.mean:.6f} "
              f"var={self.variance():.6f} std={self.stddev():.6f} m2={self.m2:.6f}")


# ---------------------------------------------------------------------------
# EMA smoother for plan index sequences
# ---------------------------------------------------------------------------
class EMASmoother:
    """Exponential moving average smoother with configurable decay."""

    __slots__ = ('alpha', 'value', 'initialised')

    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.value = 0.0
        self.initialised = False

    def update(self, x):
        if not self.initialised:
            self.value = float(x)
            self.initialised = True
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value

    def _dbg(self):
        print(f"[EMASmoother._dbg] alpha={self.alpha} value={self.value:.6f} "
              f"init={self.initialised}")


# ---------------------------------------------------------------------------
# Robust median-of-medians helper (select k-th smallest in O(n))
# ---------------------------------------------------------------------------
def _median_of_medians(arr, k):
    """Deterministic O(n) k-th order statistic via median-of-medians."""
    if len(arr) <= 5:
        return sorted(arr)[k]
    chunks = [arr[i:i + 5] for i in range(0, len(arr), 5)]
    medians = [sorted(c)[len(c) // 2] for c in chunks]
    pivot = _median_of_medians(medians, len(medians) // 2)
    low = [x for x in arr if x < pivot]
    eq = [x for x in arr if x == pivot]
    high = [x for x in arr if x > pivot]
    if k < len(low):
        return _median_of_medians(low, k)
    elif k < len(low) + len(eq):
        return pivot
    else:
        return _median_of_medians(high, k - len(low) - len(eq))


def robust_median(values):
    """Return median via median-of-medians (no sorting of full array)."""
    if not values:
        return 0.0
    n = len(values)
    if n % 2 == 1:
        return float(_median_of_medians(list(values), n // 2))
    lo = _median_of_medians(list(values), n // 2 - 1)
    hi = _median_of_medians(list(values), n // 2)
    return (lo + hi) / 2.0


def _dbg_median(values):
    med = robust_median(values)
    print(f"[robust_median._dbg] input_len={len(values)} median={med:.6f} "
          f"min={min(values) if values else 'N/A'} max={max(values) if values else 'N/A'}")
    return med


# ---------------------------------------------------------------------------
# IQR outlier fence
# ---------------------------------------------------------------------------
def iqr_fence(values, factor=1.5):
    """Remove outliers beyond factor*IQR from Q1/Q3."""
    if len(values) < 4:
        return list(values)
    s = sorted(values)
    q1 = s[len(s) // 4]
    q3 = s[3 * len(s) // 4]
    iqr = q3 - q1
    lo = q1 - factor * iqr
    hi = q3 + factor * iqr
    return [v for v in values if lo <= v <= hi]


def _dbg_iqr_fence(values, factor=1.5):
    before = len(values)
    result = iqr_fence(values, factor)
    after = len(result)
    print(f"[iqr_fence._dbg] before={before} after={after} removed={before - after} "
          f"factor={factor}")
    return result


# ---------------------------------------------------------------------------
# Bisect-based binary search for sorted plan keys
# ---------------------------------------------------------------------------
class SortedPlanIndex:
    """Maintain a sorted list of plan keys for O(log n) lookup."""

    def __init__(self):
        self._keys = []
        self._data = {}

    def insert(self, key, value):
        if key not in self._data:
            bisect.insort(self._keys, key)
        self._data[key] = value

    def lookup(self, key):
        idx = bisect.bisect_left(self._keys, key)
        if idx < len(self._keys) and self._keys[idx] == key:
            return self._data[key]
        return None

    def nearest(self, key):
        """Return value for nearest key (by string sort order)."""
        if not self._keys:
            return None
        idx = bisect.bisect_left(self._keys, key)
        if idx == 0:
            return self._data[self._keys[0]]
        if idx >= len(self._keys):
            return self._data[self._keys[-1]]
        before = self._keys[idx - 1]
        after = self._keys[idx]
        return self._data[before] if abs(hash(key) - hash(before)) < abs(hash(key) - hash(after)) else self._data[after]

    def keys(self):
        return list(self._keys)

    def __len__(self):
        return len(self._keys)

    def _dbg(self):
        print(f"[SortedPlanIndex._dbg] n_keys={len(self._keys)} "
              f"first_5={self._keys[:5]} last_5={self._keys[-5:]}")


# ---------------------------------------------------------------------------
# Raw plan dictionaries (from upstream)
# ---------------------------------------------------------------------------
cached_rob_plan_dict_stats = {
    '8b': [0],
    '8d': [0],
    '10b': [0],
    '10c': [3],
    '18a': [8],
    '20g': [3],
    '20a': [1],
    '20c': [1],
    '20f': [4],
    '21b': [4],
    '25a': [5],
    '25b': [7],
    '28e': [6],
    '30a': [11],
    '31a': [3],
    '34e': [2],
    '37b': [6],
    '38a': [8],
    '38b': [15],
    '39d': [5],
    '43b': [0],
    '45a': [10],
    '45d': [22],
    '40d': [3],
    '40f': [16]
}
cached_rob_plan_dict_dsb = {
    '100': [1],
    '101': [7],
    '102': [19],
    '025': [5],
    '013': [0],
    '018': [10],
    '019': [4],
    '027': [3],
    '040': [8],
    '050': [4],
    '072': [4],
    '084': [0],
    '085': [2],
    '091': [3],
    '099': [0],
}
cached_rob_plan_dict = {
    '1a': [2],
    '1-0-b1.0': [1],
    '1-1-b1.0': [0],
    '1-0-b0.5': [0],
    '1-1-b0.5': [0],
    '1-2-b0.5': [0],

    
    '2a': [1],
    '2-0-b1.0': [6],
    '2-1-b1.0': [0],
    '2-2-b1.0': [1],
    '2-3-b1.0': [3],
    '2-0-b0.5': [3],
    '2-1-b0.5': [6],
    '2-2-b0.5': [1],
    '2-3-b0.5': [0],
    '2-4-b0.5': [3],
    '2-5-b0.5': [4],
    '2-6-b0.5': [0],

    '3a': [10],
    '3-0-b0.5': [9],
    '3-1-b0.5': [0],
    '3-2-b0.5': [1],
    '3-3-b0.5': [0],
    '3-4-b0.5': [0],
    '3-5-b0.5': [1],
    '3-6-b0.5': [4],
    '3-7-b0.5': [0],
    '3-8-b0.5': [1],
    '3-9-b0.5': [1],
    '3-10-b0.5': [5],
    '3-11-b0.5': [0],
    '3-0-b1.0': [0],
    '3-1-b1.0': [0],
    '3-2-b1.0': [1],
    '3-3-b1.0': [0],
    '3-4-b1.0': [1],
    '3-5-b1.0': [4],

    '4a': [5, 7],
    '4-0-b0.5': [5],
    '4-1-b0.5': [0],
    '4-2-b0.5': [0],
    '4-3-b0.5': [0],
    '4-4-b0.5': [0],
    '4-5-b0.5': [3],
    '4-6-b0.5': [2],
    '4-7-b0.5': [1],
    '4-8-b0.5': [0],
    '4-9-b0.5': [0],
    '4-0-b1.0': [3],
    '4-1-b1.0': [3],
    '4-2-b1.0': [0],
    '4-3-b1.0': [5],
    '4-4-b1.0': [1],
    
    '5a': [3, 4],
    '5-0-b1.0': [13],
    '5-1-b1.0': [8],
    '5-2-b1.0': [3],
    '5-3-b1.0': [5],
    '5-4-b1.0': [14],
    '5-5-b1.0': [4],
    '5-6-b1.0': [5],
    '5-7-b1.0': [1],
    '5-8-b1.0': [7],
    '5-9-b1.0': [1],
    '5-0-b1.0': [13],
    '5-0-b0.5': [4],
    '5-1-b0.5': [10],
    '5-2-b0.5': [9],
    '5-3-b0.5': [11],
    '5-4-b0.5': [5],
    '5-5-b0.5': [8],
    '5-6-b0.5': [12],
    '5-7-b0.5': [2],
    '5-8-b0.5': [5],
    '5-9-b0.5': [4],
    '5-10-b0.5': [2],
    '5-11-b0.5': [1],
    '5-12-b0.5': [1],
    '5-13-b0.5': [6],
    '5-14-b0.5': [3],
    '5-15-b0.5': [6],
    '5-16-b0.5': [2],
    '5-17-b0.5': [9],

    '6a': [2, 4, 5],
    '6-0-b1.0': [0],
    '6-1-b1.0': [3],
    '6-2-b1.0': [4],
    '6-3-b1.0': [0],
    '6-4-b1.0': [7],
    '6-5-b1.0': [2],
    '6-6-b1.0': [0],
    
    '7a': [4, 5, 6],
    '7-0-b1.0': [3],
    '7-1-b1.0': [5],
    '7-2-b1.0': [1],
    '7-3-b1.0': [13],
    '7-4-b1.0': [10],
    '7-5-b1.0': [5],
    '7-6-b1.0': [2],
    '7-7-b1.0': [9],
    '7-8-b1.0': [0],

    '8a': [0],
    '8-0-b1.0': [5],
    '8-1-b1.0': [1],
    '8-2-b1.0': [3],
    '8-3-b1.0': [0],
    '8-4-b1.0': [5],
    '8-5-b1.0': [2],
    '8-6-b1.0': [7],
    '8-7-b1.0': [1],
    '8-8-b1.0': [1],
    '8-9-b1.0': [1],
    '8-10-b1.0': [9],

    '9a': [33, 34, 39],
    '9-0-b1.0': [1],
    '9-1-b1.0': [18],
    '9-2-b1.0': [1],
    '9-3-b1.0': [4],
    '9-4-b1.0': [35],
    '9-5-b1.0': [14],
    '9-6-b1.0': [5],
    '9-7-b1.0': [0],
    '9-8-b1.0': [5],
    '9-9-b1.0': [16],
    '9-10-b1.0': [1],
    '9-11-b1.0': [20],
    '9-12-b1.0': [0],
    '9-13-b1.0': [3],
    '9-14-b1.0': [0],
    '9-15-b1.0': [6],
    '9-16-b1.0': [4],
    '9-17-b1.0': [9],
    '9-18-b1.0': [1],
    '9-19-b1.0': [8],
    '9-20-b1.0': [2],
    '9-21-b1.0': [7],
    '10a': [30, 31, 32],
    '10-0-b1.0': [13],
    '10-1-b1.0': [9],
    '10-2-b1.0': [5],
    '10-3-b1.0': [1],
    '10-4-b1.0': [8],
    '10-5-b1.0': [1],
    '11a': [14, 15, 16],
    '11-0-b1.0': [0],
    '11-1-b1.0': [0],
    '11-2-b1.0': [0],
    '11-3-b1.0': [0],
    '11-4-b1.0': [2],
    '11-5-b1.0': [0],
    '11-6-b1.0': [0],
    '11-7-b1.0': [0],
    '11-8-b1.0': [0],
    '11-9-b1.0': [1],
    '11-10-b1.0': [11],
    '11-11-b1.0': [0],
    '11-12-b1.0': [1],
    '12a': [17],
    '12-0-b1.0': [23],
    '12-1-b1.0': [0],
    '12-2-b1.0': [8],
    '12-3-b1.0': [28],
    '12-4-b1.0': [2],
    '12-5-b1.0': [8],
    '12-6-b1.0': [0],
    '12-7-b1.0': [3],
    '12-8-b1.0': [6],
    '12-9-b1.0': [0],
    '12-10-b1.0': [0],
    '12-11-b1.0': [2],
    '12-12-b1.0': [5],
    '13a': [1, 2, 4],
    '13-0-b1.0': [45],
    '13-1-b1.0': [0],
    '13-2-b1.0': [3],
    '14a': [0],
    '14-0-b1.0': [2],
    '14-1-b1.0': [1],
    '14-2-b1.0': [3],
    '14-3-b1.0': [0],
    '14-4-b1.0': [3],
    '14-5-b1.0': [0],
    '14-6-b1.0': [2],
    '14-7-b1.0': [2],
    '14-8-b1.0': [2],
    '15a': [15],
    '16a': [14, 15, 16],
    '16-0': [0],
    '16-1': [2],
    '16-2': [0],
    '16-3': [0],
    '16-4': [3],
    '16-5': [1],
    '16-6': [0],
    '16-7': [0],
    '16-8': [0],
    '16-9': [0],
    '16-10': [0],
    '16-11': [0],
    '16-12': [0],
    '16-13': [0],
    '16-14': [4],
    '16-15': [2],
    '16-16': [2],
    '16-17': [0],
    '16-18': [0],
    '16-0-b1.0': [0],
    '16-1-b1.0': [0],
    '16-2-b1.0': [4],
    '16-3-b1.0': [0],
    '16-4-b1.0': [0],
    '16-5-b1.0': [2],
    '17a': [22],
    '17-0': [19],
    '17-1': [1],
    '17-2': [10],
    '17-3': [7],
    '17-4': [19],
    '17-5': [13],
    '17-6': [11],
    '17-7': [1],
    '17-8': [7],
    '17-9': [2],
    '17-10': [1],
    '17-11': [5],
    '17-12': [0],
    '17-13': [0],
    '17-14': [0],
    '17-15': [9],
    '17-16': [12],
    '17-17': [1],
    '17-18': [0],
    '17-0-b1.0': [15],
    '17-1-b1.0': [0],
    '17-2-b1.0': [8],
    '17-3-b1.0': [12],
    '17-4-b1.0': [1],
    '17-5-b1.0': [2],
    
    
    '18a': [7, 8],
    '18-0-b1.0': [2],
    '18-1-b1.0': [0],
    
    '18-0': [2],
    '18-1': [0],
    '18-2': [1],
    '18-3': [0],
    '18-4': [4],
    '18-5': [0],
    '19a': [20, 21, 22],
    '20a': [10, 11, 12],
    '20-0': [11],
    '20-1': [44],
    '20-2': [6],
    '20-3': [33],
    '20-4': [7],
    '20-5': [29],
    '20-6': [23],
    '20-7': [53],
    '20-8': [59],
    '20-9': [30],
    '20-10': [7],
    '20-11': [14],
    '20-12': [19],
    '20-13': [27],
    '20-14': [29],
    '21a': [21, 22, 23],
    '22a': [3, 4],
    '23a': [28, 33, 47],
    '24a': [8, 11, 12],
    '25a': [9],
    '26a': [5],
    '26-0': [10],
    '26-1': [4],
    '26-2': [3],
    '26-3': [3],
    '26-4': [5],
    '26-5': [5],
    '26-6': [6],
    '26-7': [7],
    '26-8': [6],
    '26-9': [3],
    '26-10': [6],
    '26-11': [4],
    '26-12': [0],
    '26-13': [9],
    '26-14': [6],
    '26-15': [2],
    '26-16': [5],
    '26-17': [4],
    '26-18': [8],
    '26-19': [0],
    '26-20': [0],
    '26-21': [10],
    '26-22': [2],

    '27a': [10],
    '28a': [25, 26, 27],
    '29a': [2],
    '30a': [25],
    '31a': [5],
    '32a': [7],
    '33a': [7]
}

cached_rob_plan_dict_by_prob = {
    '1-0-b0.5': [0],
    '1-1-b0.5': [3],
    '1-2-b0.5': [1],
    '1-3-b0.5': [3],

    '2-0-b0.5': [2],
    '2-1-b0.5': [3],
    '2-2-b0.5': [9],
    '2-3-b0.5': [1],
    '2-4-b0.5': [7],
    '2-5-b0.5': [0],
    '2-6-b0.5': [1],
    '2-7-b0.5': [4],
    '2-8-b0.5': [0],

    '3-0-b0.5': [6],
    '3-1-b0.5': [0],
    '3-2-b0.5': [2],
    '3-3-b0.5': [3],
    '3-4-b0.5': [1],
    '3-5-b0.5': [6],
    '3-6-b0.5': [0],
    '3-7-b0.5': [2],
    '3-8-b0.5': [1],
    '3-9-b0.5': [0],
    '3-10-b0.5': [1],
    '3-11-b0.5': [1],
    '3-12-b0.5': [1],
    '3-13-b0.5': [0],
    '3-14-b0.5': [0],

    '4-0-b0.5': [6],
    '4-1-b0.5': [1],
    '4-2-b0.5': [6],
    '4-3-b0.5': [4],
    '4-4-b0.5': [7],
    '4-5-b0.5': [1],
    '4-6-b0.5': [0],
    '4-7-b0.5': [5],
    '4-8-b0.5': [2],
    '4-9-b0.5': [2],
    '4-10-b0.5': [2],
    '4-11-b0.5': [3],

    '16-0-b0.5': [0],
    '16-1-b0.5': [1],
    '16-2-b0.5': [13],
    '16-3-b0.5': [0],
    '16-4-b0.5': [5],
    '16-5-b0.5': [5],
    '16-6-b0.5': [2],
    '16-7-b0.5': [3],
    '16-8-b0.5': [7],
    '16-9-b0.5': [0],
    '16-10-b0.5': [4],
    '16-11-b0.5': [3],
    '16-12-b0.5': [7],
    '16-13-b0.5': [0],
    '16-14-b0.5': [0],
    '16-15-b0.5': [4],
    '16-16-b0.5': [0],
    '16-17-b0.5': [0],

    '17-0-b0.5': [8],
    '17-1-b0.5': [2],
    '17-2-b0.5': [17],
    '17-3-b0.5': [6],
    '17-4-b0.5': [9],
    '17-5-b0.5': [7],
    '17-6-b0.5': [1],
    '17-7-b0.5': [3],
    '17-8-b0.5': [2],
    '17-9-b0.5': [5],
    '17-10-b0.5': [5],
    '17-11-b0.5': [3],
    '17-12-b0.5': [3],
    '17-13-b0.5': [0],
    '17-14-b0.5': [0],

    '18-0-b0.5': [1],
    '18-1-b0.5': [7],
    '18-2-b0.5': [9],
    '18-3-b0.5': [8],
    '18-4-b0.5': [3],
    '18-5-b0.5': [4],

    '20-0-b0.5': [9],
    '20-1-b0.5': [4],
    '20-2-b0.5': [3],
    '20-3-b0.5': [5],
    '20-4-b0.5': [16],
    '20-5-b0.5': [7],
    '20-6-b0.5': [15],
    '20-7-b0.5': [4],
    '20-8-b0.5': [13],
    '20-9-b0.5': [11],
    '20-10-b0.5': [12],
    '20-11-b0.5': [6],
    '20-12-b0.5': [13],
    '20-13-b0.5': [3],
    '20-14-b0.5': [9],
    '20-15-b0.5': [3],

    'q2-t0-all-0-b0.5': [3],
    'q2-t0-all-1-b0.5': [4],
    'q2-t0-all-2-b0.5': [5],
    'q2-t0-all-3-b0.5': [2],
}

cached_rob_plan_dict_on_demand = {
    '1-0-b0.5': [0],
    '1-1-b0.5': [1],
    '1-2-b0.5': [1],

    '2-0-b0.5': [3],
    '2-1-b0.5': [2],
    '2-2-b0.5': [4],
    '2-3-b0.5': [1],
    '2-4-b0.5': [0],
    '2-5-b0.5': [0],
    '2-6-b0.5': [7],
    '2-7-b0.5': [3],
    '2-8-b0.5': [0],
    '2-9-b0.5': [1],

    '3-0-b0.5': [2],
    '3-1-b0.5': [0],
    '3-2-b0.5': [1],
    '3-3-b0.5': [0],
    '3-4-b0.5': [1],
    '3-5-b0.5': [1],
    '3-6-b0.5': [0],
    '3-7-b0.5': [0],
    '3-8-b0.5': [1],
    '3-9-b0.5': [1],
    '3-10-b0.5': [0],
    '3-11-b0.5': [0],

    '4-0-b0.5': [4],
    '4-1-b0.5': [1],
    '4-2-b0.5': [1],
    '4-3-b0.5': [2],
    '4-4-b0.5': [2],
    '4-5-b0.5': [2],
    '4-6-b0.5': [1],
    '4-7-b0.5': [2],
    '4-8-b0.5': [3],
    '4-9-b0.5': [1],

    '16-0-b0.5': [0],
    '16-1-b0.5': [2],
    '16-2-b0.5': [6],
    '16-3-b0.5': [9],
    '16-4-b0.5': [1],
    '16-5-b0.5': [14],
    '16-6-b0.5': [1],
    '16-7-b0.5': [2],
    '16-8-b0.5': [1],
    '16-9-b0.5': [0],
    '16-10-b0.5': [0],
    '16-11-b0.5': [0],
    '16-12-b0.5': [2],
    '16-13-b0.5': [0],
    '16-14-b0.5': [22],
    '16-15-b0.5': [0],
    '16-16-b0.5': [1],

    '17-0-b0.5': [10],
    '17-1-b0.5': [0],
    '17-2-b0.5': [11],
    '17-3-b0.5': [0],
    '17-4-b0.5': [2],
    '17-5-b0.5': [3],
    '17-6-b0.5': [1],
    '17-7-b0.5': [2],
    '17-8-b0.5': [0],
    '17-9-b0.5': [1],
    '17-10-b0.5': [3],
    '17-11-b0.5': [0],

    '18-0-b0.5': [0],
    '18-1-b0.5': [4],
    '18-2-b0.5': [5],
    '18-3-b0.5': [0],
    '18-4-b0.5': [9],
    '18-5-b0.5': [1],

    '20-0-b0.5': [0],
    '20-1-b0.5': [3],
    '20-2-b0.5': [10],
    '20-3-b0.5': [9],
    '20-4-b0.5': [4],
    '20-5-b0.5': [1],
    '20-6-b0.5': [12],
    '20-7-b0.5': [5],
    '20-8-b0.5': [4],
    '20-9-b0.5': [3],
    '20-10-b0.5': [13],
    '20-11-b0.5': [4],
    '20-12-b0.5': [1],
    '20-13-b0.5': [10],
    '20-14-b0.5': [8],

    # 
    'q1-t1-10-0-b0.5': [0],
    'q1-t1-10-1-b0.5': [1],

    'q1-t1-20-0-b0.5': [0],
    'q1-t1-20-1-b0.5': [2],

    'q1-t1-50-0-b0.5': [0],
    'q1-t1-50-1-b0.5': [3],

    'q1-t1-all-0-b0.5': [1],
    'q1-t1-all-1-b0.5': [3],

    # csv
    'csv-q2-t0-all-0-b0.5': [1],
    'csv-q2-t0-all-1-b0.5': [1],
    'csv-q2-t0-all-2-b0.5': [0],

    'csv-q2-t0-50-0-b0.5': [1],
    'csv-q2-t0-50-1-b0.5': [0],
    'csv-q2-t0-50-2-b0.5': [1],
    'csv-q2-t0-50-3-b0.5': [0],

    'csv-q2-t0-400-0-b0.5': [1],
    'csv-q2-t0-400-1-b0.5': [1],
    'csv-q2-t0-400-2-b0.5': [0],
    'csv-q2-t0-400-3-b0.5': [3],

    'csv-q2-t0-2000-0-b0.5': [3],
    'csv-q2-t0-2000-1-b0.5': [0],
    'csv-q2-t0-2000-2-b0.5': [2],
    'csv-q2-t0-2000-3-b0.5': [1],

    'csv-q7-t1-50-0-b0.5': [2],
    'csv-q7-t1-50-1-b0.5': [2],
    'csv-q7-t1-50-2-b0.5': [1],
    'csv-q7-t1-50-3-b0.5': [1],
    'csv-q7-t1-50-4-b0.5': [0],
    'csv-q7-t1-50-5-b0.5': [2],
    'csv-q7-t1-50-6-b0.5': [2],

    'csv-q7-t1-400-0-b0.5': [2],
    'csv-q7-t1-400-1-b0.5': [2],
    'csv-q7-t1-400-2-b0.5': [2],
    'csv-q7-t1-400-3-b0.5': [0],
    'csv-q7-t1-400-4-b0.5': [1],
    'csv-q7-t1-400-5-b0.5': [1],
    'csv-q7-t1-400-6-b0.5': [2],
    'csv-q7-t1-400-7-b0.5': [1],
    'csv-q7-t1-400-8-b0.5': [1],
    'csv-q7-t1-400-9-b0.5': [0],
    'csv-q7-t1-400-10-b0.5': [0],

    # kepler
    'kepler-q2-t0-all-0-b0.5': [0],
    'kepler-q2-t0-all-1-b0.5': [0],
    'kepler-q2-t0-all-2-b0.5': [1],
    'kepler-q2-t0-all-3-b0.5': [0],
    'kepler-q2-t0-all-4-b0.5': [0],

    'kepler-q2-t0-50-0-b0.5': [0],
    'kepler-q2-t0-50-1-b0.5': [0],
    'kepler-q2-t0-50-2-b0.5': [0],
    'kepler-q2-t0-50-3-b0.5': [0],
    'kepler-q2-t0-50-4-b0.5': [0],

    'kepler-q2-t0-400-0-b0.5': [0],
    'kepler-q2-t0-400-1-b0.5': [0],
    'kepler-q2-t0-400-2-b0.5': [1],
    'kepler-q2-t0-400-3-b0.5': [1],

    'kepler-q2-t0-2000-0-b0.5': [2],
    'kepler-q2-t0-2000-1-b0.5': [0],
    'kepler-q2-t0-2000-2-b0.5': [0],
    'kepler-q2-t0-2000-3-b0.5': [1],
    'kepler-q2-t0-2000-4-b0.5': [2],

    'kepler-q7-t1-50-0-b0.5': [2],
    'kepler-q7-t1-50-1-b0.5': [2],
    'kepler-q7-t1-50-2-b0.5': [1],
    'kepler-q7-t1-50-3-b0.5': [3],
    'kepler-q7-t1-50-4-b0.5': [1],
    'kepler-q7-t1-50-5-b0.5': [1],
    'kepler-q7-t1-50-6-b0.5': [2],
    'kepler-q7-t1-50-7-b0.5': [1],
    'kepler-q7-t1-50-8-b0.5': [1],

    'kepler-q7-t1-400-0-b0.5': [3],
    'kepler-q7-t1-400-1-b0.5': [0],
    'kepler-q7-t1-400-2-b0.5': [3],
    'kepler-q7-t1-400-3-b0.5': [0],
    'kepler-q7-t1-400-4-b0.5': [2],
    'kepler-q7-t1-400-5-b0.5': [11],
    'kepler-q7-t1-400-6-b0.5': [1],
    'kepler-q7-t1-400-7-b0.5': [0],
    'kepler-q7-t1-400-8-b0.5': [2],
    'kepler-q7-t1-400-9-b0.5': [0],
    'kepler-q7-t1-400-10-b0.5': [2],
    'kepler-q7-t1-400-11-b0.5': [0],
    'kepler-q7-t1-400-12-b0.5': [0],
    'kepler-q7-t1-400-13-b0.5': [0],
    'kepler-q7-t1-400-14-b0.5': [3],
    'kepler-q7-t1-400-15-b0.5': [0],
    'kepler-q7-t1-400-16-b0.5': [2],

    # cardinality
    'cardinality-q2-t0-all-0-b0.5': [1],
    'cardinality-q2-t0-all-1-b0.5': [0],
    'cardinality-q2-t0-all-2-b0.5': [1],
    'cardinality-q2-t0-all-3-b0.5': [3],
    'cardinality-q2-t0-all-4-b0.5': [3],

    'cardinality-q2-t0-50-0-b0.5': [5],
    'cardinality-q2-t0-50-1-b0.5': [5],
    'cardinality-q2-t0-50-2-b0.5': [0],
    'cardinality-q2-t0-50-3-b0.5': [2],
    'cardinality-q2-t0-50-4-b0.5': [7],

    'cardinality-q2-t0-400-0-b0.5': [7],
    'cardinality-q2-t0-400-1-b0.5': [5],
    'cardinality-q2-t0-400-2-b0.5': [4],
    'cardinality-q2-t0-400-3-b0.5': [5],
    'cardinality-q2-t0-400-4-b0.5': [0],

    'cardinality-q2-t0-2000-0-b0.5': [1],
    'cardinality-q2-t0-2000-1-b0.5': [0],
    'cardinality-q2-t0-2000-2-b0.5': [5],
    'cardinality-q2-t0-2000-3-b0.5': [0],
    'cardinality-q2-t0-2000-4-b0.5': [1],
    'cardinality-q2-t0-2000-5-b0.5': [4],

    'cardinality-q7-t1-50-0-b0.5': [3],
    'cardinality-q7-t1-50-1-b0.5': [0],
    'cardinality-q7-t1-50-2-b0.5': [1],
    'cardinality-q7-t1-50-3-b0.5': [0],
    'cardinality-q7-t1-50-4-b0.5': [2],
    'cardinality-q7-t1-50-5-b0.5': [0],
    'cardinality-q7-t1-50-6-b0.5': [2],
    'cardinality-q7-t1-50-7-b0.5': [2],
    'cardinality-q7-t1-50-8-b0.5': [3],
    'cardinality-q7-t1-50-9-b0.5': [0],
    'cardinality-q7-t1-50-10-b0.5': [1],
    'cardinality-q7-t1-50-11-b0.5': [1],
    'cardinality-q7-t1-50-12-b0.5': [1],

    'cardinality-q7-t1-400-0-b0.5': [4],
    'cardinality-q7-t1-400-1-b0.5': [1],
    'cardinality-q7-t1-400-2-b0.5': [2],
    'cardinality-q7-t1-400-3-b0.5': [0],
    'cardinality-q7-t1-400-4-b0.5': [0],
    'cardinality-q7-t1-400-5-b0.5': [0],
    'cardinality-q7-t1-400-6-b0.5': [0],
    'cardinality-q7-t1-400-7-b0.5': [0],
    'cardinality-q7-t1-400-8-b0.5': [0],
    'cardinality-q7-t1-400-9-b0.5': [0],
    'cardinality-q7-t1-400-10-b0.5': [0],
    'cardinality-q7-t1-400-11-b0.5': [2],
    'cardinality-q7-t1-400-12-b0.5': [0],
    'cardinality-q7-t1-400-13-b0.5': [0],
    'cardinality-q7-t1-400-14-b0.5': [3],
    'cardinality-q7-t1-400-15-b0.5': [0],
    'cardinality-q7-t1-400-16-b0.5': [0],
    'cardinality-q7-t1-400-17-b0.5': [0],

    'q7-t1-10-0-b0.5': [2],
    'q7-t1-10-1-b0.5': [3],
    'q7-t1-10-2-b0.5': [6],
    'q7-t1-10-3-b0.5': [0],

    'q7-t1-20-0-b0.5': [2],
    'q7-t1-20-1-b0.5': [1],
    'q7-t1-20-2-b0.5': [4],
    'q7-t1-20-3-b0.5': [0],
    'q7-t1-20-4-b0.5': [1],
    'q7-t1-20-5-b0.5': [0],

    'q7-t1-50-0-b0.5': [2],
    'q7-t1-50-1-b0.5': [1],
    'q7-t1-50-2-b0.5': [4],
    'q7-t1-50-3-b0.5': [0],
    'q7-t1-50-4-b0.5': [1],
    'q7-t1-50-5-b0.5': [0],

    # 'q7-t1-all-0-b0.5': [3],
    # 'q7-t1-all-1-b0.5': [3],
    # 'q7-t1-all-2-b0.5': [2],
    # 'q7-t1-all-3-b0.5': [0],
    # 'q7-t1-all-4-b0.5': [2],
    # 'q7-t1-all-5-b0.5': [1],
    # 'q7-t1-all-6-b0.5': [1],
    # 'q7-t1-all-7-b0.5': [1],

    'q7-t1-all-0-b0.5': [3],
    'q7-t1-all-1-b0.5': [3],
    'q7-t1-all-2-b0.5': [4],
    'q7-t1-all-3-b0.5': [0],
    'q7-t1-all-4-b0.5': [2],
    'q7-t1-all-5-b0.5': [0],
    'q7-t1-all-6-b0.5': [2],
    'q7-t1-all-7-b0.5': [1],
    'q7-t1-all-8-b0.5': [1],
    'q7-t1-all-9-b0.5': [1],
}


# ---------------------------------------------------------------------------
# Aggregation helpers with Welford + EMA + robust median
# ---------------------------------------------------------------------------
def aggregate_plan_stats(plan_dict, ema_alpha=0.15):
    """Build per-query running stats from a plan dictionary.

    For each key, feeds all plan index values through a WelfordAccumulator
    for variance tracking and an EMASmoother for trend detection.
    Returns an OrderedDict of {key: {mean, var, ema, median, fenced_values}}.
    """
    result = OrderedDict()
    smoother = EMASmoother(alpha=ema_alpha)
    for key in sorted(plan_dict.keys()):
        values = plan_dict[key]
        acc = WelfordAccumulator()
        for v in values:
            acc.update(float(v))
        ema_val = smoother.update(acc.mean)
        fenced = iqr_fence(values) if len(values) >= 4 else list(values)
        med = robust_median(values) if values else 0.0
        result[key] = {
            'mean': acc.mean,
            'var': acc.variance(),
            'ema': ema_val,
            'median': med,
            'fenced_values': fenced,
        }
    return result


def _dbg_aggregate(plan_dict, ema_alpha=0.15):
    result = aggregate_plan_stats(plan_dict, ema_alpha)
    print(f"[aggregate_plan_stats._dbg] n_keys={len(result)} alpha={ema_alpha}")
    for k, v in list(result.items())[:5]:
        print(f"  key={k!r} mean={v['mean']:.4f} var={v['var']:.4f} "
              f"ema={v['ema']:.4f} median={v['median']:.4f} "
              f"fenced_n={len(v['fenced_values'])}")
    return result


def build_sorted_index(plan_dict):
    """Build a SortedPlanIndex from a plan dictionary for O(log n) lookups."""
    idx = SortedPlanIndex()
    for key, value in plan_dict.items():
        idx.insert(key, value)
    return idx


def _dbg_sorted_index(plan_dict):
    idx = build_sorted_index(plan_dict)
    idx._dbg()
    # verify a few lookups
    for k in list(plan_dict.keys())[:3]:
        found = idx.lookup(k)
        print(f"  lookup({k!r}) -> {found}")
    return idx


# ---------------------------------------------------------------------------
# Cross-dict variance comparison
# ---------------------------------------------------------------------------
def cross_dict_variance_report(dicts, names=None):
    """Compare Welford variance across multiple plan dicts.

    For keys that appear in all dicts, report which dict has the tightest
    (lowest variance) plan selection for that query template.
    """
    if names is None:
        names = [f"dict_{i}" for i in range(len(dicts))]
    all_keys = set()
    for d in dicts:
        all_keys.update(d.keys())
    report = {}
    for key in sorted(all_keys):
        entries = []
        for i, d in enumerate(dicts):
            if key not in d:
                continue
            acc = WelfordAccumulator()
            for v in d[key]:
                acc.update(float(v))
            entries.append((names[i], acc.mean, acc.variance()))
        if entries:
            best = min(entries, key=lambda x: x[2])
            report[key] = {
                'entries': entries,
                'best_dict': best[0],
                'best_var': best[2],
            }
    return report


def _dbg_cross_dict_variance(dicts, names=None):
    report = cross_dict_variance_report(dicts, names)
    print(f"[cross_dict_variance._dbg] n_keys={len(report)}")
    for k, v in list(report.items())[:5]:
        print(f"  key={k!r} best={v['best_dict']} var={v['best_var']:.6f} "
              f"all={[(e[0], round(e[2], 4)) for e in v['entries']]}")
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("M141: par2qo_cached_robust_plan_dict experiment run")
    print("=" * 72)

    # 1. Welford accumulator demo
    print("\n--- Welford Accumulator ---")
    acc = WelfordAccumulator()
    sample_vals = [2, 5, 3, 8, 1, 7, 4, 6, 9, 0]
    for v in sample_vals:
        acc.update(v)
    acc._dbg()

    # 2. EMA smoother demo
    print("\n--- EMA Smoother ---")
    sm = EMASmoother(alpha=0.2)
    for v in sample_vals:
        sm.update(v)
    sm._dbg()

    # 3. Robust median
    print("\n--- Robust Median ---")
    _dbg_median(sample_vals)

    # 4. IQR fence
    print("\n--- IQR Fence ---")
    noisy = [1, 2, 3, 4, 5, 100, 200]
    _dbg_iqr_fence(noisy)

    # 5. Sorted plan index
    print("\n--- Sorted Plan Index ---")
    _dbg_sorted_index(cached_rob_plan_dict_stats)

    # 6. Aggregate plan stats
    print("\n--- Aggregate Plan Stats (main dict) ---")
    _dbg_aggregate(cached_rob_plan_dict)

    # 7. Cross-dict variance report
    print("\n--- Cross-Dict Variance ---")
    _dbg_cross_dict_variance(
        [cached_rob_plan_dict_by_prob, cached_rob_plan_dict_on_demand],
        names=['by_prob', 'on_demand']
    )

    print("\nM141 complete.")
