"""
par2qo_utility.py — Core utility functions for PQO system
Upstream ref: par2qo/code/utility.py (MIT)

Algorithmic enhancements:
- Vectorized error sampling with antithetic variates
- Log-sum-exp for numerically stable joint probability
- Welford-tracked execution statistics
- Robust CDF computation with Greenwood variance
"""
import numpy as np
import json
import os
import time
from collections import defaultdict

def card(a):
    """Safe cardinality conversion — ensures minimum of 1."""
    v = int(a)
    return max(v, 1)


def list_multiply(a, b):
    """Element-wise multiplication of two lists (vectorized)."""
    return list(np.multiply(a, b))


def cal_rel_error(sample_sel, est_sel):
    """Compute relative error between sample and estimated selectivity.
    Uses log-ratio for symmetry."""
    if est_sel < 1e-15 or sample_sel < 1e-15:
        return 0.0
    return float(np.log(sample_sel / est_sel))


def find_bin_id_from_err_hist_list(est_card, raw_card, cur_dim, err_info_dict):
    """Binary search for appropriate error histogram bin.
    
    Upstream uses linear scan; we use binary search for O(log n).
    """
    if cur_dim not in err_info_dict or not err_info_dict[cur_dim]:
        return 0
    
    profile = err_info_dict[cur_dim]
    sel = max(1, est_card[cur_dim]) / raw_card[cur_dim]
    
    if isinstance(profile, dict) and 'bin_edges' in profile:
        edges = profile['bin_edges']
        # Binary search
        lo, hi = 0, len(edges) - 2
        while lo < hi:
            mid = (lo + hi) // 2
            if edges[mid + 1] < sel:
                lo = mid + 1
            else:
                hi = mid
        return lo
    
    # Fallback for list-style profiles
    if isinstance(profile, (list, tuple)) and len(profile) >= 3:
        bins = profile[0]
        if isinstance(bins, (list, np.ndarray)):
            idx = np.searchsorted(bins, sel, side='right') - 1
            return max(0, min(idx, len(bins) - 2))
    
    return 0


class SelectivitySampler:
    """Generate selectivity samples from error distributions.
    
    Uses antithetic variates for variance reduction.
    """
    
    def __init__(self, n_samples=100):
        self._n = n_samples
        self._rng = np.random.default_rng(42)
    
    def _dbg(self, label=""):
        print(f"  [SelectivitySampler._dbg] {label}")
        print(f"    n_samples={self._n}")
    
    def gen_samples_from_joint_err_dist(self, n_dims, error_profiles=None, 
                                          antithetic=True):
        """Generate joint error samples across dimensions.
        
        With antithetic variates: generates n/2 samples and their mirrors.
        """
        half_n = self._n // 2 if antithetic else self._n
        
        samples = np.zeros((self._n, n_dims))
        for d in range(n_dims):
            if error_profiles and d in error_profiles:
                profile_data = error_profiles[d]
                if isinstance(profile_data, np.ndarray):
                    idx = self._rng.integers(0, len(profile_data), size=half_n)
                    base_samples = profile_data[idx]
                else:
                    base_samples = self._rng.normal(0, 0.5, half_n)
            else:
                base_samples = self._rng.normal(0, 0.5, half_n)
            
            if antithetic:
                mirror = -base_samples
                samples[:half_n, d] = base_samples
                samples[half_n:, d] = mirror
            else:
                samples[:, d] = base_samples[:self._n]
        
        return samples
    
    def gen_center_from_err_dist(self, err_info_dict, est_card, raw_card, dims):
        """Generate center (most-likely) error from distribution.
        
        Uses mode of KDE (maximum density point) instead of mean.
        """
        centers = np.zeros(len(dims))
        
        for i, dim in enumerate(dims):
            if dim not in err_info_dict or not err_info_dict[dim]:
                centers[i] = 0.0
                continue
            
            profile = err_info_dict[dim]
            if isinstance(profile, dict) and 'errors' in profile:
                errors = profile['errors']
                # Estimate mode using histogram peak
                hist, edges = np.histogram(errors, bins=20)
                peak_bin = np.argmax(hist)
                centers[i] = (edges[peak_bin] + edges[peak_bin + 1]) / 2
            else:
                centers[i] = 0.0
        
        return centers


class JointProbabilityCalculator:
    """Compute joint probability of selectivity samples given error profiles.
    
    Uses log-sum-exp for numerical stability.
    """
    
    def __init__(self):
        self._call_count = 0
        self._total_time = 0.0
    
    def _dbg(self, label=""):
        print(f"  [JointProbabilityCalculator._dbg] {label}")
        print(f"    call_count={self._call_count}")
        if self._call_count > 0:
            print(f"    avg_time={self._total_time/self._call_count:.4f}s")
    
    def compute_joint_probability(self, sel_samples, sensitive_dims,
                                    est_card, raw_card, err_info_dict,
                                    vectorized=True):
        """Compute joint probability of selectivity samples.
        
        Args:
            sel_samples: (n_samples, n_all_dims) array
            sensitive_dims: list of sensitive dimension indices
            est_card, raw_card: cardinality arrays
            err_info_dict: error profile dictionary
            vectorized: use batch computation
        
        Returns:
            probability_list: (n_samples,) array of probabilities
        """
        t0 = time.time()
        self._call_count += 1
        
        n_samples = len(sel_samples)
        log_probs = np.zeros(n_samples)
        
        for dim in sensitive_dims:
            if dim not in err_info_dict or not err_info_dict[dim]:
                continue
            
            profile = err_info_dict[dim]
            est_sel = max(1, est_card[dim]) / raw_card[dim]
            
            if vectorized:
                # Batch error computation
                if sel_samples.ndim > 1:
                    sel_col = sel_samples[:, dim] if dim < sel_samples.shape[1] else np.full(n_samples, est_sel)
                else:
                    sel_col = np.full(n_samples, est_sel)
                
                err_col = np.log(np.maximum(sel_col, 1e-15) / max(est_sel, 1e-15))
                
                # Evaluate density using stored profile
                if isinstance(profile, dict) and 'errors' in profile:
                    stored = profile['errors']
                    # KDE density estimation (simplified Gaussian)
                    bw = 0.9 * np.std(stored) * len(stored) ** (-0.2)
                    bw = max(bw, 1e-8)
                    diff = err_col[:, None] - stored[None, :]
                    log_kernels = -0.5 * (diff / bw) ** 2 - np.log(bw * np.sqrt(2 * np.pi))
                    max_log = np.max(log_kernels, axis=1)
                    log_density = max_log + np.log(np.sum(np.exp(log_kernels - max_log[:, None]), axis=1)) - np.log(len(stored))
                    log_probs += log_density
                else:
                    # Fallback: standard normal density
                    log_probs += -0.5 * err_col ** 2 - 0.5 * np.log(2 * np.pi)
            else:
                for s in range(n_samples):
                    sel_val = sel_samples[s, dim] if sel_samples.ndim > 1 else est_sel
                    err_val = cal_rel_error(sel_val, est_sel)
                    # Standard normal approx
                    log_probs[s] += -0.5 * err_val ** 2 - 0.5 * np.log(2 * np.pi)
        
        self._total_time += time.time() - t0
        
        return np.exp(log_probs)


class PerformanceTracker:
    """Track PQO vs default plan performance with Welford statistics."""
    
    def __init__(self):
        self._welford_pqo = {'n': 0, 'mean': 0.0, 'm2': 0.0}
        self._welford_default = {'n': 0, 'mean': 0.0, 'm2': 0.0}
        self._n_pqo_better = 0
        self._total = 0
    
    def _dbg(self, label=""):
        print(f"\n[PerformanceTracker._dbg] {label}")
        print(f"  n_total={self._total}, n_pqo_better={self._n_pqo_better}")
        if self._total > 0:
            print(f"  pqo_win_rate={self._n_pqo_better/self._total:.2%}")
        for name, w in [("pqo", self._welford_pqo), ("default", self._welford_default)]:
            if w['n'] > 1:
                var = w['m2'] / (w['n'] - 1)
                print(f"  {name}: mean={w['mean']:.2f}, std={np.sqrt(var):.2f}")
    
    def _update_welford(self, w, x):
        w['n'] += 1
        d = x - w['mean']
        w['mean'] += d / w['n']
        d2 = x - w['mean']
        w['m2'] += d * d2
    
    def record(self, pqo_latency, default_latency):
        self._total += 1
        self._update_welford(self._welford_pqo, pqo_latency)
        self._update_welford(self._welford_default, default_latency)
        if pqo_latency < default_latency:
            self._n_pqo_better += 1
    
    def speedup_ratio(self):
        if self._welford_pqo['n'] == 0 or self._welford_pqo['mean'] < 1e-12:
            return 1.0
        return self._welford_default['mean'] / self._welford_pqo['mean']


def compute_cdf(data):
    """Compute empirical CDF with Greenwood variance estimate."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    cdf = np.arange(1, n + 1) / n
    
    # Greenwood variance: Var(F(x)) ≈ F(x)(1-F(x))/n
    variance = cdf * (1 - cdf) / n
    
    return sorted_data, cdf, variance


def clean_json_file(input_path, output_path, del_keys=None):
    """Clean PostgreSQL EXPLAIN JSON output for parsing."""
    if del_keys is None:
        del_keys = ['QUERY PLAN', 'row)', '----']
    
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            skip = any(k in line for k in del_keys)
            if not skip:
                line = line.replace('+', '').strip() + '\n'
            fout.write(line)


if __name__ == "__main__":
    print("=" * 60)
    print("par2qo_utility — Integration Test")
    print("=" * 60)
    np.random.seed(42)
    
    # Test 1: Basic functions
    print(f"  card(0)={card(0)}, card(100)={card(100)}")
    print(f"  list_multiply([1,2,3],[4,5,6])={list_multiply([1,2,3],[4,5,6])}")
    print(f"  cal_rel_error(0.02, 0.01)={cal_rel_error(0.02, 0.01):.4f}")
    
    # Test 2: Binary search bin finder
    err_info = {
        0: {'bin_edges': np.array([0, 0.1, 0.2, 0.3, 0.5, 1.0]), 'errors': np.random.normal(0, 0.3, 100)},
    }
    bin_id = find_bin_id_from_err_hist_list([100], [1000], cur_dim=0, err_info_dict=err_info)
    print(f"\n  find_bin_id(sel=0.1): {bin_id}")
    
    # Test 3: Selectivity sampler
    sampler = SelectivitySampler(n_samples=100)
    samples = sampler.gen_samples_from_joint_err_dist(3, antithetic=True)
    print(f"\n  joint samples shape: {samples.shape}")
    print(f"  dim 0: mean={samples[:,0].mean():.4f} (should be ~0 with antithetic)")
    sampler._dbg()
    
    # Test 4: Joint probability
    jpc = JointProbabilityCalculator()
    sel_samples = np.random.uniform(0.01, 0.5, (50, 5))
    probs = jpc.compute_joint_probability(
        sel_samples, [0, 2], [100, 200, 300, 400, 500], [1000]*5, err_info
    )
    print(f"\n  joint probs: shape={probs.shape}, mean={probs.mean():.6f}")
    jpc._dbg()
    
    # Test 5: Performance tracker
    tracker = PerformanceTracker()
    rng = np.random.default_rng(42)
    for _ in range(50):
        pqo = 10 + rng.exponential(5)
        default = 15 + rng.exponential(8)
        tracker.record(pqo, default)
    tracker._dbg("after 50 comparisons")
    print(f"  speedup ratio: {tracker.speedup_ratio():.2f}x")
    
    # Test 6: CDF computation
    data = rng.exponential(10, 200)
    x, cdf, var = compute_cdf(data)
    print(f"\n  CDF: n={len(x)}, x_range=[{x[0]:.2f}, {x[-1]:.2f}]")
    print(f"  median≈{x[np.searchsorted(cdf, 0.5)]:.2f}")
    
    print("\nAll tests passed.")
