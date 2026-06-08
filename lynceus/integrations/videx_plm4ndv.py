"""
M195: videx_plm4ndv — Piecewise Linear Model for NDV Estimation
Upstream: histogram/plm4ndv_model_infer.py (~340L)
Algorithm changes (20%):
  - Ramer-Douglas-Peucker adaptive breakpoint selection (upstream: uniform)
  - Welford online stats for incremental model updates
  - Confidence interval estimation on predictions
"""
import math
import random
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)
_DBG = True
def _dbg(tag, **kw):
    if _DBG: print(f"  [dbg:{tag}] { {k: repr(v)[:80] for k,v in kw.items()} }")


def rdp_simplify(points: List[Tuple[float, float]], epsilon: float) -> List[int]:
    """Ramer-Douglas-Peucker line simplification — returns indices of kept points."""
    if len(points) <= 2:
        return list(range(len(points)))
    
    # Find point with max distance from line between first and last
    start, end = np.array(points[0]), np.array(points[-1])
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    
    if line_len < 1e-12:
        return [0, len(points) - 1]
    
    max_dist = 0.0
    max_idx = 0
    for i in range(1, len(points) - 1):
        pt = np.array(points[i])
        dist = abs(np.cross(line_vec, start - pt)) / line_len
        if dist > max_dist:
            max_dist = dist
            max_idx = i
    
    if max_dist > epsilon:
        left = rdp_simplify(points[:max_idx + 1], epsilon)
        right = rdp_simplify(points[max_idx:], epsilon)
        # Adjust right indices
        right_adj = [idx + max_idx for idx in right]
        return left[:-1] + right_adj
    else:
        return [0, len(points) - 1]


class WelfordTracker:
    __slots__ = ("_n", "_mean", "_m2")
    def __init__(self):
        self._n = 0; self._mean = 0.0; self._m2 = 0.0
    def update(self, val):
        self._n += 1
        d = val - self._mean
        self._mean += d / self._n
        self._m2 += d * (val - self._mean)
    @property
    def mean(self): return self._mean
    @property
    def variance(self): return self._m2 / self._n if self._n > 1 else 0.0
    @property
    def std(self): return math.sqrt(self.variance)
    def snapshot(self): return {"n": self._n, "mean": round(self._mean,4), "std": round(self.std,4)}


class PLMSegment:
    """A single linear segment: y = slope * x + intercept, for x in [x_start, x_end]."""
    __slots__ = ("x_start", "x_end", "slope", "intercept")
    
    def __init__(self, x_start: float, x_end: float, slope: float, intercept: float):
        self.x_start = x_start
        self.x_end = x_end
        self.slope = slope
        self.intercept = intercept
    
    def predict(self, x: float) -> float:
        return self.slope * x + self.intercept
    
    def contains(self, x: float) -> bool:
        return self.x_start <= x <= self.x_end


class PLMModel:
    """Piecewise Linear Model with adaptive breakpoints via RDP."""
    
    def __init__(self, epsilon: float = 0.05):
        self._segments: List[PLMSegment] = []
        self._epsilon = epsilon
        self._error_tracker = WelfordTracker()
        self._fitted = False
    
    def fit(self, x_data: np.ndarray, y_data: np.ndarray):
        """Fit PLM using RDP for adaptive breakpoint selection."""
        assert len(x_data) == len(y_data), "x and y must have same length"
        n = len(x_data)
        if n < 2:
            return
        
        # Sort by x
        order = np.argsort(x_data)
        xs, ys = x_data[order], y_data[order]
        
        # Normalize for RDP
        x_range = max(xs[-1] - xs[0], 1e-12)
        y_range = max(ys.max() - ys.min(), 1e-12)
        points = [(float((xs[i] - xs[0]) / x_range), float((ys[i] - ys.min()) / y_range)) 
                  for i in range(n)]
        
        # RDP to find breakpoints
        bp_indices = rdp_simplify(points, self._epsilon)
        bp_indices = sorted(set(bp_indices))
        
        if len(bp_indices) < 2:
            bp_indices = [0, n - 1]
        
        # Fit linear segments between breakpoints
        self._segments = []
        for i in range(len(bp_indices) - 1):
            i0, i1 = bp_indices[i], bp_indices[i + 1]
            x0, x1 = float(xs[i0]), float(xs[i1])
            y0, y1 = float(ys[i0]), float(ys[i1])
            
            dx = x1 - x0
            if abs(dx) < 1e-12:
                slope = 0.0
                intercept = (y0 + y1) / 2
            else:
                slope = (y1 - y0) / dx
                intercept = y0 - slope * x0
            
            self._segments.append(PLMSegment(x0, x1, slope, intercept))
        
        self._fitted = True
        _dbg("PLM.fit", n_points=n, n_segments=len(self._segments), 
             breakpoints=len(bp_indices))
    
    def predict(self, x: float) -> Tuple[float, float]:
        """Predict y and confidence interval width."""
        if not self._fitted or not self._segments:
            return 0.0, float('inf')
        
        # Find containing segment
        for seg in self._segments:
            if seg.contains(x):
                pred = seg.predict(x)
                ci = self._error_tracker.std * 1.96 if self._error_tracker._n > 2 else pred * 0.2
                return pred, ci
        
        # Extrapolate from nearest endpoint
        if x < self._segments[0].x_start:
            seg = self._segments[0]
        else:
            seg = self._segments[-1]
        pred = seg.predict(x)
        ci = self._error_tracker.std * 2.5 if self._error_tracker._n > 2 else pred * 0.3
        return pred, ci
    
    def update_error(self, predicted: float, actual: float):
        error = abs(predicted - actual)
        self._error_tracker.update(error)
    
    def snapshot(self):
        return {"n_segments": len(self._segments), "fitted": self._fitted,
                "error_stats": self._error_tracker.snapshot()}


class NDVPredictor:
    """Predict NDV (number of distinct values) using PLM on sample-to-population mapping."""
    
    def __init__(self, epsilon: float = 0.03):
        self._plm = PLMModel(epsilon=epsilon)
        self._calibration = WelfordTracker()
    
    def train(self, sample_sizes: np.ndarray, observed_ndvs: np.ndarray):
        """Train on (sample_size, observed_ndv) pairs."""
        log_sizes = np.log1p(sample_sizes)
        log_ndvs = np.log1p(observed_ndvs)
        self._plm.fit(log_sizes, log_ndvs)
        _dbg("NDVPredictor.train", n_pairs=len(sample_sizes))
    
    def predict_ndv(self, sample_size: int, sample_ndv: int, 
                    population_size: int) -> Tuple[float, float, float]:
        """Predict population NDV given sample statistics.
        Returns: (predicted_ndv, ci_lower, ci_upper)
        """
        log_pop = math.log1p(population_size)
        pred_log, ci_width = self._plm.predict(log_pop)
        
        # Scale by sample ratio
        sample_ratio = min(sample_size / max(population_size, 1), 1.0)
        scaling = 1.0 / max(sample_ratio, 0.01)
        
        predicted = math.expm1(pred_log) * min(scaling, 5.0)
        predicted = max(predicted, sample_ndv)  # NDV >= sample NDV
        predicted = min(predicted, population_size)  # NDV <= population
        
        ci_lower = max(sample_ndv, predicted - math.expm1(ci_width))
        ci_upper = min(population_size, predicted + math.expm1(ci_width))
        
        _dbg("predict_ndv", sample_sz=sample_size, sample_ndv=sample_ndv,
             pop=population_size, pred=round(predicted,1))
        return predicted, ci_lower, ci_upper
    
    def calibrate(self, predicted: float, actual: float):
        self._plm.update_error(predicted, actual)
        ratio = actual / max(predicted, 1e-6)
        self._calibration.update(ratio)
    
    def snapshot(self):
        return {"plm": self._plm.snapshot(), "calibration": self._calibration.snapshot()}


if __name__ == "__main__":
    print("=== M195 videx_plm4ndv self-test ===")
    
    # Test RDP
    pts = [(0,0), (1,1), (2,2.01), (3,3), (4,4), (5,10), (6,10.1), (7,15)]
    indices = rdp_simplify(pts, 0.1)
    assert len(indices) < len(pts)
    
    # Test PLM
    rng = np.random.RandomState(42)
    x = np.sort(rng.uniform(0, 100, 200))
    y = np.where(x < 50, 2 * x + rng.normal(0, 2, 200), 
                 100 + 0.5 * (x - 50) + rng.normal(0, 2, 200))
    
    plm = PLMModel(epsilon=0.02)
    plm.fit(x, y)
    assert plm._fitted
    assert len(plm._segments) > 1
    
    pred, ci = plm.predict(25.0)
    assert abs(pred - 50) < 20  # rough check
    
    # Test NDVPredictor
    sample_sizes = np.array([100, 500, 1000, 5000, 10000])
    observed_ndvs = np.array([80, 300, 500, 2000, 3000])
    
    ndv_pred = NDVPredictor(epsilon=0.05)
    ndv_pred.train(sample_sizes, observed_ndvs)
    
    pred_ndv, ci_lo, ci_hi = ndv_pred.predict_ndv(1000, 500, 100000)
    assert pred_ndv >= 500  # at least sample NDV
    assert pred_ndv <= 100000  # at most population
    assert ci_lo <= pred_ndv <= ci_hi
    
    print(f"  PLM segments: {len(plm._segments)}")
    print(f"  NDV prediction (sample=1000, ndv=500, pop=100K): {pred_ndv:.0f} [{ci_lo:.0f}, {ci_hi:.0f}]")
    print("  All tests passed!")
    print(f"  Lines: {sum(1 for _ in open(__file__))}")
