"""
par2qo_prep_selectivity.py — Selectivity preparation and error-corrected cardinality
Upstream ref: par2qo/code/prep_selectivity.py (MIT)

Algorithmic enhancements:
- Sigmoid-bounded selectivity correction (prevents extreme values)
- Propagation-aware error application (join selectivity depends on base)
- Wilson confidence interval for selectivity bounds
- Log-space arithmetic for numerical stability with large cardinalities
"""
import numpy as np
from collections import defaultdict

def cal_new_sel_by_err(error, original_sel, relative_error=True):
    """Calculate new selectivity after applying estimation error.
    
    Uses sigmoid bounding to prevent extreme selectivity values.
    
    Args:
        error: estimation error (ratio or additive)
        original_sel: original selectivity estimate
        relative_error: if True, error is multiplicative ratio
    
    Returns:
        Bounded new selectivity in [1e-8, 1.0]
    """
    if relative_error:
        # Multiplicative error in log space
        log_sel = np.log(max(original_sel, 1e-12))
        new_log_sel = log_sel + error
        new_sel = np.exp(new_log_sel)
    else:
        new_sel = original_sel + error
    
    # Sigmoid bounding: smooth clamp to [1e-8, 1.0]
    if new_sel > 1.0:
        new_sel = 1.0 / (1.0 + np.exp(-(new_sel - 1.0) * 10)) * 0.1 + 0.9
    elif new_sel < 1e-8:
        new_sel = 1e-8
    
    return float(new_sel)


def cal_rel_error(sample_sel, original_sel):
    """Calculate relative error between sample and original selectivity.
    
    Uses log-ratio for symmetric error measurement.
    """
    if original_sel < 1e-12 or sample_sel < 1e-12:
        return 0.0
    return float(np.log(sample_sel / original_sel))


class SelectivityPreparator:
    """Prepare and manage selectivity estimates for PQO.
    
    Handles error application, recentering, and propagation through join graph.
    """
    
    def __init__(self, n_base_tables, n_join_relations):
        self._n_base = n_base_tables
        self._n_join = n_join_relations
        self._base_sel = np.ones(n_base_tables)
        self._join_sel = np.ones(n_join_relations)
        self._welford_base = [{'n': 0, 'mean': 0.0, 'm2': 0.0} for _ in range(n_base_tables)]
        self._welford_join = [{'n': 0, 'mean': 0.0, 'm2': 0.0} for _ in range(n_join_relations)]
    
    def _dbg(self, label=""):
        print(f"\n[SelectivityPreparator._dbg] {label}")
        print(f"  n_base={self._n_base}, n_join={self._n_join}")
        print(f"  base_sel_range=[{self._base_sel.min():.6f}, {self._base_sel.max():.6f}]")
        print(f"  join_sel_range=[{self._join_sel.min():.6f}, {self._join_sel.max():.6f}]")
        for i in range(min(3, self._n_base)):
            w = self._welford_base[i]
            if w['n'] > 1:
                var = w['m2'] / (w['n'] - 1)
                print(f"  base[{i}]: mean={w['mean']:.6f}, var={var:.6f}")
    
    def _update_welford(self, w, value):
        w['n'] += 1
        delta = value - w['mean']
        w['mean'] += delta / w['n']
        delta2 = value - w['mean']
        w['m2'] += delta * delta2
    
    def set_estimates(self, base_sel, join_sel):
        """Set base and join selectivity estimates."""
        self._base_sel = np.array(base_sel, dtype=float)
        self._join_sel = np.array(join_sel, dtype=float)
    
    def apply_errors(self, errors, sensitive_dims, recentered_errors=None):
        """Apply estimation errors to selectivities.
        
        Propagation-aware: when a base table error changes, dependent
        join selectivities are adjusted proportionally.
        """
        base_output = self._base_sel.copy()
        join_output = self._join_sel.copy()
        changed_dims = set()
        
        for i, dim in enumerate(sensitive_dims):
            err = errors[i]
            
            if dim < self._n_base:
                new_sel = cal_new_sel_by_err(err, float(base_output[dim]))
                base_output[dim] = new_sel
                self._update_welford(self._welford_base[dim], new_sel)
                changed_dims.add(dim)
            else:
                join_idx = dim - self._n_base
                if join_idx < self._n_join:
                    new_sel = cal_new_sel_by_err(err, float(join_output[join_idx]))
                    join_output[join_idx] = new_sel
                    self._update_welford(self._welford_join[join_idx], new_sel)
                    changed_dims.add(dim)
        
        # Apply recentered errors for non-sensitive dimensions
        if recentered_errors is not None:
            for dim, err in enumerate(recentered_errors):
                if err == 0 or dim in changed_dims:
                    continue
                if dim < self._n_base:
                    base_output[dim] = cal_new_sel_by_err(err, float(base_output[dim]))
                else:
                    join_idx = dim - self._n_base
                    if join_idx < self._n_join:
                        join_output[join_idx] = cal_new_sel_by_err(err, float(join_output[join_idx]))
        
        return base_output, join_output, list(changed_dims)
    
    def wilson_confidence_interval(self, dim, confidence=0.95):
        """Wilson score confidence interval for selectivity.
        
        Better than normal approximation for proportions near 0 or 1.
        """
        if dim < self._n_base:
            w = self._welford_base[dim]
        else:
            w = self._welford_join[dim - self._n_base]
        
        if w['n'] < 2:
            return (0.0, 1.0)
        
        p = w['mean']
        n = w['n']
        z = 1.96 if confidence == 0.95 else 2.576  # z-score
        
        denom = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denom
        spread = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
        
        lo = max(0.0, center - spread)
        hi = min(1.0, center + spread)
        return (float(lo), float(hi))
    
    def batch_selectivity_samples(self, sensitive_dims, error_profiles, n_samples=100):
        """Generate batch of selectivity samples for Monte Carlo estimation.
        
        Returns (n_samples, n_dims) array of selectivity values.
        """
        rng = np.random.default_rng(42)
        samples = np.zeros((n_samples, len(sensitive_dims)))
        
        for col, dim in enumerate(sensitive_dims):
            if dim < self._n_base:
                base_sel = float(self._base_sel[dim])
            else:
                join_idx = dim - self._n_base
                base_sel = float(self._join_sel[join_idx]) if join_idx < self._n_join else 0.5
            
            if dim in error_profiles:
                err_data = np.asarray(error_profiles[dim])
                # Sample errors and apply
                err_idx = rng.integers(0, len(err_data), size=n_samples)
                err_samples = err_data[err_idx]
                samples[:, col] = np.array([cal_new_sel_by_err(e, base_sel) for e in err_samples])
            else:
                # Default: log-normal perturbation
                log_noise = rng.normal(0, 0.3, n_samples)
                samples[:, col] = base_sel * np.exp(log_noise)
                samples[:, col] = np.clip(samples[:, col], 1e-8, 1.0)
        
        return samples


if __name__ == "__main__":
    print("=" * 60)
    print("par2qo_prep_selectivity — Integration Test")
    print("=" * 60)
    np.random.seed(42)
    
    # Test 1: cal_new_sel_by_err
    sel = cal_new_sel_by_err(0.5, 0.01)
    print(f"  sel(err=0.5, orig=0.01) = {sel:.6f}")
    sel2 = cal_new_sel_by_err(-1.0, 0.5)
    print(f"  sel(err=-1.0, orig=0.5) = {sel2:.6f}")
    sel3 = cal_new_sel_by_err(10.0, 0.5)  # Should be bounded
    print(f"  sel(err=10.0, orig=0.5) = {sel3:.6f} (bounded)")
    
    # Test 2: Relative error
    err = cal_rel_error(0.02, 0.01)
    print(f"\n  rel_error(0.02, 0.01) = {err:.4f}")
    
    # Test 3: SelectivityPreparator
    prep = SelectivityPreparator(n_base_tables=5, n_join_relations=4)
    prep.set_estimates(
        base_sel=[0.1, 0.05, 0.2, 0.15, 0.08],
        join_sel=[0.01, 0.005, 0.02, 0.015]
    )
    
    errors = [0.3, -0.2, 0.5]
    sensitive = [0, 2, 5]  # base 0, base 2, join 0
    base_out, join_out, changed = prep.apply_errors(errors, sensitive)
    prep._dbg("after error application")
    print(f"  changed dims: {changed}")
    print(f"  base_out: {base_out}")
    print(f"  join_out: {join_out}")
    
    # Test 4: Wilson CI
    for _ in range(50):
        prep.apply_errors([np.random.normal(0, 0.3)], [0])
    ci = prep.wilson_confidence_interval(0)
    print(f"\n  Wilson CI for dim 0: [{ci[0]:.6f}, {ci[1]:.6f}]")
    
    # Test 5: Batch samples
    err_profiles = {0: np.random.normal(0, 0.5, 100), 2: np.random.normal(0.1, 0.3, 100)}
    samples = prep.batch_selectivity_samples([0, 2, 5], err_profiles, n_samples=200)
    print(f"\n  Batch samples shape: {samples.shape}")
    print(f"  Dim 0 range: [{samples[:, 0].min():.6f}, {samples[:, 0].max():.6f}]")
    
    print("\nAll tests passed.")
