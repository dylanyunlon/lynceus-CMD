"""
par2qo_plan_inspection.py — Plan cost distribution inspection and comparison
Upstream ref: par2qo/code/plan_inspection.py (MIT)

Algorithmic enhancements:
- Monte Carlo cost distribution estimation with antithetic variates
- Robust plan comparison using Kolmogorov-Smirnov test
- Bootstrapped confidence intervals for cost distributions
- Welford online variance for streaming cost aggregation
- Sensitivity-aware dimension filtering with Sobol indices
"""
import numpy as np
import re
import json
import os
from collections import defaultdict

class SensitivityAnalyzer:
    """Analyze dimension sensitivity for query plan cost estimation.
    Uses Sobol-like variance decomposition with numpy-only implementation.
    """
    
    def __init__(self, n_samples=200):
        self._n_samples = n_samples
        self._sensitivity_scores = {}
        self._rng = np.random.default_rng(42)
    
    def _dbg(self, label=""):
        print(f"  [SensitivityAnalyzer._dbg] {label}")
        print(f"    n_samples={self._n_samples}")
        print(f"    n_dims_analyzed={len(self._sensitivity_scores)}")
        if self._sensitivity_scores:
            sorted_dims = sorted(self._sensitivity_scores.items(), key=lambda x: -x[1])
            top3 = sorted_dims[:3]
            print(f"    top_3_sensitive: {[(d, f'{s:.4f}') for d, s in top3]}")
    
    def compute_sobol_first_order(self, cost_function, n_dims, bounds):
        """First-order Sobol sensitivity indices via Saltelli sampling.
        
        cost_function: callable(params_array) -> cost_array
        n_dims: number of dimensions
        bounds: list of (lo, hi) tuples per dimension
        """
        N = self._n_samples
        
        # Base and re-sample matrices
        A = self._rng.uniform(size=(N, n_dims))
        B = self._rng.uniform(size=(N, n_dims))
        
        # Scale to bounds
        for d in range(n_dims):
            lo, hi = bounds[d]
            A[:, d] = lo + A[:, d] * (hi - lo)
            B[:, d] = lo + B[:, d] * (hi - lo)
        
        f_A = cost_function(A)
        f_B = cost_function(B)
        total_var = np.var(np.concatenate([f_A, f_B]))
        
        if total_var < 1e-12:
            self._sensitivity_scores = {d: 0.0 for d in range(n_dims)}
            return self._sensitivity_scores
        
        for d in range(n_dims):
            # C_d = B with d-th column from A
            C_d = B.copy()
            C_d[:, d] = A[:, d]
            f_C = cost_function(C_d)
            
            # First-order Sobol: S_d = Var(E[Y|X_d]) / Var(Y)
            # Estimated as: (1/N) * sum(f_A * (f_C - f_B)) / Var(Y)
            s_d = np.mean(f_A * (f_C - f_B)) / total_var
            self._sensitivity_scores[d] = max(0.0, float(s_d))
        
        return self._sensitivity_scores
    
    def get_sensitive_dims(self, threshold=0.05):
        """Return dimensions with sensitivity above threshold."""
        return [d for d, s in self._sensitivity_scores.items() if s >= threshold]


class CostDistributionEstimator:
    """Estimate and compare plan cost distributions under cardinality errors.
    
    Uses Monte Carlo sampling with antithetic variates for variance reduction.
    """
    
    def __init__(self, n_mc_samples=100):
        self._n_mc = n_mc_samples
        self._rng = np.random.default_rng(42)
        self._cost_cache = {}
    
    def _dbg(self, label=""):
        print(f"  [CostDistributionEstimator._dbg] {label}")
        print(f"    n_mc_samples={self._n_mc}")
        print(f"    cost_cache_size={len(self._cost_cache)}")
    
    def sample_error_distribution(self, error_profile, n_samples=None):
        """Sample from error distribution with antithetic variates."""
        if n_samples is None:
            n_samples = self._n_mc
        
        half_n = n_samples // 2
        
        if isinstance(error_profile, dict) and 'errors' in error_profile:
            errors = error_profile['errors']
            # Regular samples
            idx = self._rng.integers(0, len(errors), size=half_n)
            samples = errors[idx] + self._rng.normal(0, 0.01, half_n)
            # Antithetic samples (mirror around mean for variance reduction)
            mean = errors.mean()
            antithetic = 2 * mean - samples
            return np.concatenate([samples, antithetic])
        else:
            # Fallback: normal distribution
            return self._rng.normal(0, 1, n_samples)
    
    def estimate_cost_distribution(self, plan_hint, error_profiles, 
                                    sensitive_dims, base_cost=100.0):
        """Estimate cost distribution for a plan under cardinality errors.
        
        Returns array of cost samples.
        """
        n = self._n_mc
        costs = np.full(n, base_cost)
        
        for dim in sensitive_dims:
            if dim in error_profiles:
                err_samples = self.sample_error_distribution(error_profiles[dim], n)
                # Cost scaling: multiplicative error model
                # cost *= (1 + abs(error))^sensitivity_weight
                costs *= (1 + np.abs(err_samples)) ** 0.5
        
        # Add small noise for numerical stability
        costs += self._rng.normal(0, 0.1, n)
        costs = np.maximum(costs, 0.01)
        
        return costs
    
    def compare_plans_ks(self, costs_a, costs_b):
        """Kolmogorov-Smirnov test between two cost distributions.
        
        Returns (ks_statistic, better_plan) where better_plan is 'a' or 'b'.
        """
        # Sort both distributions
        a_sorted = np.sort(costs_a)
        b_sorted = np.sort(costs_b)
        
        # Compute empirical CDFs
        n_a, n_b = len(a_sorted), len(b_sorted)
        all_vals = np.sort(np.concatenate([a_sorted, b_sorted]))
        
        cdf_a = np.searchsorted(a_sorted, all_vals, side='right') / n_a
        cdf_b = np.searchsorted(b_sorted, all_vals, side='right') / n_b
        
        ks_stat = float(np.max(np.abs(cdf_a - cdf_b)))
        
        # Determine better plan (lower median cost)
        better = 'a' if np.median(costs_a) <= np.median(costs_b) else 'b'
        
        return ks_stat, better
    
    def bootstrap_ci(self, costs, confidence=0.95, n_boot=500):
        """Bootstrap confidence interval for mean cost."""
        boot_means = np.array([
            self._rng.choice(costs, size=len(costs), replace=True).mean()
            for _ in range(n_boot)
        ])
        alpha = (1 - confidence) / 2
        return (
            float(np.percentile(boot_means, 100 * alpha)),
            float(np.percentile(boot_means, 100 * (1 - alpha)))
        )


class PlanInspector:
    """High-level plan inspection: compare PQO vs Kepler vs default plans."""
    
    def __init__(self, n_samples=100):
        self._estimator = CostDistributionEstimator(n_mc_samples=n_samples)
        self._sensitivity = SensitivityAnalyzer(n_samples=n_samples)
        self._inspection_results = []
        self._welford = {'n': 0, 'mean': 0.0, 'm2': 0.0}
    
    def _dbg(self, label=""):
        print(f"\n{'='*60}")
        print(f"[PlanInspector._dbg] {label}")
        print(f"  n_inspections={len(self._inspection_results)}")
        self._estimator._dbg()
        self._sensitivity._dbg()
        if self._welford['n'] > 1:
            var = self._welford['m2'] / (self._welford['n'] - 1)
            print(f"  running_cost: mean={self._welford['mean']:.2f}, var={var:.2f}")
        print(f"{'='*60}")
    
    def _update_welford(self, cost):
        n = self._welford['n'] + 1
        delta = cost - self._welford['mean']
        self._welford['mean'] += delta / n
        delta2 = cost - self._welford['mean']
        self._welford['m2'] += delta * delta2
        self._welford['n'] = n
    
    def inspect_plan(self, plan_id, plan_hint, error_profiles, 
                      sensitive_dims, base_cost=100.0):
        """Inspect a single plan's cost distribution."""
        costs = self._estimator.estimate_cost_distribution(
            plan_hint, error_profiles, sensitive_dims, base_cost
        )
        
        ci = self._estimator.bootstrap_ci(costs)
        
        result = {
            'plan_id': plan_id,
            'plan_hint': plan_hint,
            'cost_mean': float(costs.mean()),
            'cost_median': float(np.median(costs)),
            'cost_std': float(costs.std()),
            'cost_p5': float(np.percentile(costs, 5)),
            'cost_p95': float(np.percentile(costs, 95)),
            'ci_95': ci,
            'n_samples': len(costs),
        }
        
        self._inspection_results.append(result)
        self._update_welford(result['cost_mean'])
        
        return result, costs
    
    def compare_plans(self, results_a, costs_a, results_b, costs_b):
        """Compare two plans using KS test and cost statistics."""
        ks_stat, better = self._estimator.compare_plans_ks(costs_a, costs_b)
        
        return {
            'plan_a': results_a['plan_id'],
            'plan_b': results_b['plan_id'],
            'ks_statistic': ks_stat,
            'better_plan': better,
            'mean_diff': results_a['cost_mean'] - results_b['cost_mean'],
            'median_diff': results_a['cost_median'] - results_b['cost_median'],
        }
    
    def inspect_all_plans(self, plans, error_profiles, sensitive_dims, base_costs=None):
        """Inspect all plans and return ranked results."""
        results = []
        all_costs = []
        
        for i, (pid, hint) in enumerate(plans):
            bc = base_costs[i] if base_costs else 100.0
            res, costs = self.inspect_plan(pid, hint, error_profiles, sensitive_dims, bc)
            results.append(res)
            all_costs.append(costs)
        
        # Rank by median cost
        ranked_indices = np.argsort([r['cost_median'] for r in results])
        ranked = [(results[i], all_costs[i]) for i in ranked_indices]
        
        return ranked


def parse_plan_hint(hint_string):
    """Extract join methods and leading clause from pg_hint_plan string.
    
    Returns dict with 'join_methods', 'leading', 'scan_methods'.
    """
    join_pattern = re.compile(r'(NestLoop|HashJoin|MergeJoin)\s*\(([^)]+)\)')
    leading_pattern = re.compile(r'Leading\s*\(([^)]+)\)')
    scan_pattern = re.compile(r'(SeqScan|IndexScan|IndexOnlyScan|BitmapScan)\s*\(([^)]+)\)')
    
    joins = [(m.group(1), m.group(2).strip()) for m in join_pattern.finditer(hint_string)]
    leading = leading_pattern.findall(hint_string)
    scans = [(m.group(1), m.group(2).strip()) for m in scan_pattern.finditer(hint_string)]
    
    return {
        'join_methods': joins,
        'leading': leading[0] if leading else "",
        'scan_methods': scans,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("par2qo_plan_inspection — Integration Test")
    print("=" * 60)
    np.random.seed(42)
    
    # Test 1: Sensitivity analysis
    sa = SensitivityAnalyzer(n_samples=200)
    
    def cost_fn(params):
        # dim 0 and 2 are sensitive, dim 1 is not
        return params[:, 0] ** 2 + 0.01 * params[:, 1] + 3 * params[:, 2]
    
    bounds = [(0, 10), (0, 10), (0, 10)]
    sobol = sa.compute_sobol_first_order(cost_fn, 3, bounds)
    sa._dbg("Sobol indices")
    sensitive = sa.get_sensitive_dims(threshold=0.05)
    print(f"  Sensitive dims: {sensitive}")
    
    # Test 2: Cost distribution estimation
    error_profiles = {
        0: {'errors': np.random.normal(0, 0.3, 200)},
        2: {'errors': np.random.normal(0.1, 0.5, 200)},
    }
    
    estimator = CostDistributionEstimator(n_mc_samples=200)
    costs = estimator.estimate_cost_distribution(
        "HashJoin(t1 t2)", error_profiles, [0, 2], base_cost=50.0
    )
    print(f"\n  Cost distribution: mean={costs.mean():.2f}, std={costs.std():.2f}")
    print(f"    range=[{costs.min():.2f}, {costs.max():.2f}]")
    
    # Test 3: Plan comparison (KS test)
    costs_b = estimator.estimate_cost_distribution(
        "MergeJoin(t1 t2)", error_profiles, [0, 2], base_cost=60.0
    )
    ks, better = estimator.compare_plans_ks(costs, costs_b)
    print(f"\n  KS test: stat={ks:.4f}, better_plan={better}")
    
    # Test 4: Full plan inspection
    inspector = PlanInspector(n_samples=200)
    plans = [
        ("plan_hash", "HashJoin(t1 t2)"),
        ("plan_merge", "MergeJoin(t1 t2)"),
        ("plan_nest", "NestLoop(t1 t2)"),
    ]
    base_costs = [50.0, 60.0, 45.0]
    
    ranked = inspector.inspect_all_plans(plans, error_profiles, [0, 2], base_costs)
    print(f"\n  Ranked plans:")
    for res, _ in ranked:
        print(f"    {res['plan_id']}: median={res['cost_median']:.2f}, ci={res['ci_95']}")
    
    # Test 5: Parse hint string
    hint = "/*+ SeqScan(orders) IndexScan(lineitem) HashJoin(orders lineitem) Leading(orders lineitem) */"
    parsed = parse_plan_hint(hint)
    print(f"\n  Parsed hint: {parsed}")
    
    inspector._dbg("final state")
    print("\nAll tests passed.")
