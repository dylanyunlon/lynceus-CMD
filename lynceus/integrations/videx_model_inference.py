"""
videx_model_inference — NDV and cardinality model inference for Lynceus.

Ported from:
  - upstream/videx/histogram/plm4ndv_model_infer.py (291 lines)
  - upstream/videx/histogram/adandv_model_infer.py (121 lines)
  - upstream/videx/videx_model_example.py (46 lines)

Algorithm changes (~20%):
  - PLM4NDV: piecewise linear model with L1 regularization (Lasso-style)
  - AdaNDV: adaptive NDV estimation with exponential backoff on sample expansion
  - ExampleModel: Good-Turing smoothing for unseen value estimation
  - Feature normalization: robust scaler (IQR-based) instead of min-max
  - Prediction aggregation: median of ensemble instead of single model
"""
import math
import os
import random
from collections import Counter

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))

def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[model_inf] {tag}: {items}")


# ── Robust feature scaler (IQR-based) ───────────────────────────
class RobustScaler:
    """IQR-based feature normalization.
    
    Algorithm change: upstream uses min-max normalization.
    IQR-based scaling is robust to outliers: x' = (x - median) / IQR
    """
    
    def __init__(self):
        self.median = 0.0
        self.iqr = 1.0
        self._fitted = False
    
    def fit(self, data):
        if not data:
            return self
        sorted_d = sorted(data)
        n = len(sorted_d)
        self.median = sorted_d[n // 2]
        q1 = sorted_d[n // 4]
        q3 = sorted_d[3 * n // 4]
        self.iqr = max(q3 - q1, 1e-10)
        self._fitted = True
        _dbg("scaler_fit", median=self.median, iqr=self.iqr, n=n)
        return self
    
    def transform(self, value):
        return (value - self.median) / self.iqr
    
    def inverse_transform(self, value):
        return value * self.iqr + self.median


# ── PLM4NDV: Piecewise Linear Model for NDV estimation ──────────
class PLM4NDVModel:
    """Piecewise linear model for NDV estimation with L1 regularization.
    
    Algorithm change: upstream uses standard linear regression segments.
    We add L1 (Lasso) regularization to prevent overfitting on sparse
    sample data, promoting simpler piecewise models.
    """
    
    def __init__(self, n_pieces=5, lambda_l1=0.01):
        self.n_pieces = n_pieces
        self.lambda_l1 = lambda_l1
        self._breakpoints = []
        self._slopes = []
        self._intercepts = []
        self._scaler = RobustScaler()
        self._trained = False
    
    def train(self, sample_sizes, observed_ndvs):
        """Train piecewise linear model on (sample_size, observed_ndv) pairs."""
        if len(sample_sizes) < 2:
            _dbg("plm4ndv_skip", reason="insufficient data")
            return
        
        self._scaler.fit(sample_sizes)
        
        # Determine breakpoints evenly in scaled space
        scaled = [self._scaler.transform(x) for x in sample_sizes]
        mn, mx = min(scaled), max(scaled)
        step = (mx - mn) / self.n_pieces
        self._breakpoints = [mn + i * step for i in range(self.n_pieces + 1)]
        
        # Fit each piece using L1-regularized slope
        self._slopes = []
        self._intercepts = []
        
        for p in range(self.n_pieces):
            bp_lo = self._breakpoints[p]
            bp_hi = self._breakpoints[p + 1]
            
            # Get points in this segment
            seg_x, seg_y = [], []
            for i, sx in enumerate(scaled):
                if bp_lo <= sx <= bp_hi:
                    seg_x.append(sx)
                    seg_y.append(observed_ndvs[i])
            
            if len(seg_x) < 2:
                self._slopes.append(0.0)
                self._intercepts.append(sum(seg_y) / max(len(seg_y), 1) if seg_y else 0)
                continue
            
            # OLS + L1 soft thresholding
            x_mean = sum(seg_x) / len(seg_x)
            y_mean = sum(seg_y) / len(seg_y)
            
            num = sum((seg_x[i] - x_mean) * (seg_y[i] - y_mean) for i in range(len(seg_x)))
            den = sum((seg_x[i] - x_mean) ** 2 for i in range(len(seg_x))) or 1e-10
            slope = num / den
            
            # L1 soft thresholding
            if abs(slope) < self.lambda_l1:
                slope = 0.0
            elif slope > 0:
                slope -= self.lambda_l1
            else:
                slope += self.lambda_l1
            
            intercept = y_mean - slope * x_mean
            self._slopes.append(slope)
            self._intercepts.append(intercept)
        
        self._trained = True
        _dbg("plm4ndv_train", n_pieces=self.n_pieces,
             n_data=len(sample_sizes), slopes=self._slopes[:5])
    
    def predict(self, sample_size):
        """Predict NDV for a given sample size."""
        if not self._trained:
            return max(1, int(sample_size * 0.8))
        
        sx = self._scaler.transform(sample_size)
        
        # Find the right piece
        for p in range(self.n_pieces):
            if p == self.n_pieces - 1 or sx <= self._breakpoints[p + 1]:
                pred = self._slopes[p] * sx + self._intercepts[p]
                return max(1, int(pred))
        
        return max(1, int(self._intercepts[-1]))


# ── AdaNDV: Adaptive NDV with exponential backoff ────────────────
class AdaNDVEstimator:
    """Adaptive NDV estimation with exponential backoff on sample expansion.
    
    Algorithm change: upstream doubles sample size on each iteration.
    Exponential backoff with decay factor increases sample size more
    gradually, reducing waste when NDV converges early.
    """
    
    def __init__(self, initial_sample=100, backoff_factor=1.5, max_iterations=10,
                 convergence_threshold=0.05):
        self.initial_sample = initial_sample
        self.backoff_factor = backoff_factor
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self._history = []
    
    def estimate(self, data_source, total_rows):
        """Estimate NDV with adaptive sampling.
        
        data_source: callable that returns a random sample of given size.
        """
        sample_size = min(self.initial_sample, total_rows)
        prev_ndv = 0
        
        for iteration in range(self.max_iterations):
            sample = data_source(sample_size)
            unique_count = len(set(sample))
            
            # Good-Turing correction for unseen values
            freq_counts = Counter(Counter(sample).values())
            singletons = freq_counts.get(1, 0)
            gt_correction = singletons / max(sample_size, 1)
            
            estimated_ndv = int(unique_count / (1 - gt_correction) if gt_correction < 1 else unique_count * 2)
            estimated_ndv = min(estimated_ndv, total_rows)
            
            self._history.append({
                "iteration": iteration,
                "sample_size": sample_size,
                "unique": unique_count,
                "estimated_ndv": estimated_ndv,
            })
            
            # Convergence check
            if prev_ndv > 0:
                delta = abs(estimated_ndv - prev_ndv) / max(prev_ndv, 1)
                if delta < self.convergence_threshold:
                    _dbg("ada_converge", iter=iteration, ndv=estimated_ndv,
                         delta=f"{delta:.4f}")
                    break
            
            prev_ndv = estimated_ndv
            
            # Exponential backoff: increase sample size gradually
            sample_size = min(int(sample_size * self.backoff_factor), total_rows)
        
        _dbg("ada_estimate", final_ndv=estimated_ndv,
             iterations=iteration + 1, final_sample=sample_size)
        return estimated_ndv
    
    def dump_history(self):
        print(f"[AdaNDV] {len(self._history)} iterations:")
        for h in self._history[-5:]:
            print(f"  iter {h['iteration']}: sample={h['sample_size']} "
                  f"unique={h['unique']} est_ndv={h['estimated_ndv']}")


# ── Good-Turing smoothed model ───────────────────────────────────
class GoodTuringModel:
    """Frequency estimation with Good-Turing smoothing.
    
    Algorithm change: upstream uses raw frequency counts.
    Good-Turing re-estimates frequencies by using frequency-of-frequencies
    to handle unseen and rare values.
    """
    
    def __init__(self):
        self._freq_counts = Counter()
        self._freq_of_freq = Counter()
        self._total = 0
        self._ndv = 0
    
    def fit(self, values):
        """Fit model on observed values."""
        self._freq_counts = Counter(values)
        self._freq_of_freq = Counter(self._freq_counts.values())
        self._total = len(values)
        self._ndv = len(self._freq_counts)
        
        _dbg("gt_fit", total=self._total, ndv=self._ndv,
             singletons=self._freq_of_freq.get(1, 0))
    
    def smoothed_probability(self, value):
        """Get Good-Turing smoothed probability for a value."""
        freq = self._freq_counts.get(value, 0)
        
        if freq == 0:
            # Unseen value: P0 = N1 / N
            n1 = self._freq_of_freq.get(1, 1)
            return n1 / max(self._total, 1)
        
        # Good-Turing: r* = (r+1) * N_{r+1} / N_r
        n_r = self._freq_of_freq.get(freq, 1)
        n_r1 = self._freq_of_freq.get(freq + 1, 0)
        
        if n_r > 0 and n_r1 > 0:
            r_star = (freq + 1) * n_r1 / n_r
        else:
            r_star = freq
        
        return r_star / max(self._total, 1)
    
    def estimate_ndv_total(self, total_population):
        """Estimate total NDV in the population from sample."""
        # Chao1 estimator
        f1 = self._freq_of_freq.get(1, 0)
        f2 = max(self._freq_of_freq.get(2, 1), 1)
        chao1 = self._ndv + f1 * (f1 - 1) / (2 * f2)
        
        # Cap at total population
        return min(int(chao1), total_population)


# ── Ensemble prediction ─────────────────────────────────────────
def ensemble_ndv_predict(models, sample_size):
    """Aggregate NDV predictions using median of ensemble.
    
    Algorithm change: upstream uses single model prediction.
    Median of ensemble is more robust to individual model errors.
    """
    predictions = []
    for model in models:
        if hasattr(model, 'predict'):
            pred = model.predict(sample_size)
        elif hasattr(model, 'estimate_ndv_total'):
            pred = model.estimate_ndv_total(sample_size)
        else:
            continue
        predictions.append(pred)
    
    if not predictions:
        return max(1, int(sample_size * 0.8))
    
    predictions.sort()
    median = predictions[len(predictions) // 2]
    
    _dbg("ensemble", n_models=len(models), predictions=predictions[:5],
         median=median)
    return median
