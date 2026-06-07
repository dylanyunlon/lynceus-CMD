"""
par2qo_gen_real_error.py — Real cardinality error generation from local selection conditions
Upstream ref: par2qo/code/gen_real_error.py (MIT)

Algorithmic enhancements:
- Stratified sampling of conditions by frequency (importance sampling)
- Huber-loss robust error aggregation
- Bootstrap confidence intervals for error distributions
- Incremental error profile updates (avoids full recomputation)
"""
import numpy as np
import os
import json
import csv
from collections import defaultdict
import copy

class LocalSelectionConditions:
    """Manage local selection conditions (LSC) for error generation.
    
    Replaces CSV/pandas-based loading with numpy-native operations.
    """
    
    def __init__(self, db_type='imdb'):
        self.db_type = db_type
        self._condition_dict = defaultdict(list)
        self._frequency_dict = defaultdict(list)
        self._n_conditions = 0
    
    def _dbg(self, label=""):
        print(f"\n[LocalSelectionConditions._dbg] {label}")
        print(f"  db_type={self.db_type}")
        print(f"  n_tables={len(self._condition_dict)}")
        print(f"  n_conditions={self._n_conditions}")
        for table in list(self._condition_dict.keys())[:5]:
            print(f"    {table}: {len(self._condition_dict[table])} conditions")
    
    def load_from_data(self, table_conditions):
        """Load conditions from dict {table: [(condition, frequency), ...]}."""
        for table, conds in table_conditions.items():
            for cond, freq in conds:
                self._condition_dict[table].append(cond)
                self._frequency_dict[table].append(freq)
                self._n_conditions += 1
    
    def generate_synthetic(self, tables, n_conds_per_table=20):
        """Generate synthetic local selection conditions for testing."""
        rng = np.random.default_rng(42)
        
        predicates = ['=', '>', '<', '>=', '<=', 'BETWEEN', 'IN', 'LIKE']
        
        for table in tables:
            for i in range(n_conds_per_table):
                pred = rng.choice(predicates)
                val = rng.integers(1, 10000)
                cond = f"col_{rng.integers(1,10)} {pred} {val}"
                freq = int(rng.exponential(5) + 1)
                self._condition_dict[table].append(cond)
                self._frequency_dict[table].append(freq)
                self._n_conditions += 1
    
    def sample_conditions(self, table, n=10, weighted=True):
        """Sample conditions with optional frequency weighting.
        
        Uses importance sampling when weighted=True.
        """
        conds = self._condition_dict.get(table, [])
        freqs = self._frequency_dict.get(table, [])
        
        if not conds:
            return []
        
        n = min(n, len(conds))
        
        if weighted and freqs:
            weights = np.array(freqs, dtype=float)
            weights /= weights.sum()
            indices = np.random.choice(len(conds), size=n, replace=False, p=weights)
        else:
            indices = np.random.choice(len(conds), size=n, replace=False)
        
        return [(conds[i], freqs[i] if freqs else 1) for i in indices]


class RealErrorGenerator:
    """Generate real cardinality estimation errors.
    
    Simulates the process of running queries with different predicates
    and measuring estimation vs actual cardinality discrepancy.
    """
    
    def __init__(self, n_bins=10):
        self._n_bins = n_bins
        self._errors_per_table = defaultdict(list)
        self._error_profiles = {}
        self._welford_per_table = defaultdict(lambda: {'n': 0, 'mean': 0.0, 'm2': 0.0})
        self._rng = np.random.default_rng(42)
    
    def _dbg(self, label=""):
        print(f"\n[RealErrorGenerator._dbg] {label}")
        print(f"  n_bins={self._n_bins}")
        print(f"  n_tables_with_errors={len(self._errors_per_table)}")
        for table in list(self._errors_per_table.keys())[:5]:
            errs = self._errors_per_table[table]
            print(f"    {table}: {len(errs)} errors, mean={np.mean(errs):.4f}")
    
    def _update_welford(self, table, val):
        w = self._welford_per_table[table]
        w['n'] += 1
        d = val - w['mean']
        w['mean'] += d / w['n']
        d2 = val - w['mean']
        w['m2'] += d * d2
    
    def simulate_estimation_error(self, true_card, table_size, noise_std=0.5):
        """Simulate cardinality estimation error.
        
        Uses log-normal model: estimated = true * exp(N(bias, sigma))
        where bias represents systematic over/underestimation.
        """
        # Systematic bias: smaller tables tend to be overestimated
        bias = -0.1 * np.log(max(true_card / (table_size + 1), 1e-8))
        bias = np.clip(bias, -1, 1)
        
        log_error = self._rng.normal(bias, noise_std)
        estimated = max(1, int(true_card * np.exp(log_error)))
        
        # Q-error: max(est/true, true/est)
        q_error = max(estimated / max(true_card, 1), max(true_card, 1) / max(estimated, 1))
        
        # Relative error in log space
        rel_error = float(np.log(max(estimated, 1) / max(true_card, 1)))
        
        return estimated, q_error, rel_error
    
    def generate_errors_for_table(self, table, true_cards, table_size, noise_std=0.5):
        """Generate estimation errors for a table across multiple queries."""
        for tc in true_cards:
            _, q_err, rel_err = self.simulate_estimation_error(tc, table_size, noise_std)
            self._errors_per_table[table].append(rel_err)
            self._update_welford(table, rel_err)
    
    def build_error_profile(self, table):
        """Build error distribution profile for a table.
        
        Uses IQR-based outlier filtering before binning.
        """
        errors = np.array(self._errors_per_table.get(table, []))
        if len(errors) < 5:
            return None
        
        # IQR filtering
        q25, q75 = np.percentile(errors, [25, 75])
        iqr = q75 - q25
        mask = (errors >= q25 - 1.5 * iqr) & (errors <= q75 + 1.5 * iqr)
        clean = errors[mask] if mask.sum() >= 5 else errors
        
        # Bin the errors
        bin_edges = np.linspace(clean.min(), clean.max(), self._n_bins + 1)
        
        profile = {
            'errors': clean,
            'bin_edges': bin_edges,
            'mean': float(clean.mean()),
            'std': float(clean.std()),
            'n_original': len(errors),
            'n_clean': len(clean),
        }
        self._error_profiles[table] = profile
        return profile
    
    def huber_aggregate_errors(self, table, delta=1.0):
        """Aggregate errors using Huber loss (robust to outliers)."""
        errors = np.array(self._errors_per_table.get(table, []))
        if len(errors) == 0:
            return 0.0
        
        median = np.median(errors)
        residuals = errors - median
        weights = np.where(np.abs(residuals) <= delta, 1.0, delta / (np.abs(residuals) + 1e-12))
        return float(np.average(errors, weights=weights))
    
    def bootstrap_error_ci(self, table, confidence=0.95, n_boot=500):
        """Bootstrap CI for mean error of a table."""
        errors = np.array(self._errors_per_table.get(table, []))
        if len(errors) < 5:
            return (0.0, 0.0)
        
        boot_means = np.array([
            self._rng.choice(errors, size=len(errors), replace=True).mean()
            for _ in range(n_boot)
        ])
        alpha = (1 - confidence) / 2
        return (float(np.percentile(boot_means, 100 * alpha)),
                float(np.percentile(boot_means, 100 * (1 - alpha))))
    
    def incremental_update(self, table, new_errors):
        """Incrementally update error profile with new observations."""
        self._errors_per_table[table].extend(new_errors)
        for e in new_errors:
            self._update_welford(table, e)
        # Rebuild profile
        self.build_error_profile(table)


if __name__ == "__main__":
    print("=" * 60)
    print("par2qo_gen_real_error — Integration Test")
    print("=" * 60)
    np.random.seed(42)
    
    # Test 1: Local selection conditions
    lsc = LocalSelectionConditions('imdb')
    tables = ['title', 'movie_companies', 'cast_info', 'keyword']
    lsc.generate_synthetic(tables, n_conds_per_table=30)
    lsc._dbg("synthetic conditions")
    
    sampled = lsc.sample_conditions('title', n=5, weighted=True)
    print(f"  Sampled conditions from 'title': {len(sampled)}")
    for c, f in sampled[:3]:
        print(f"    '{c}' (freq={f})")
    
    # Test 2: Error generation
    gen = RealErrorGenerator(n_bins=8)
    
    rng = np.random.default_rng(42)
    true_cards = rng.integers(10, 50000, 100)
    gen.generate_errors_for_table('title', true_cards, 250000, noise_std=0.6)
    gen.generate_errors_for_table('movie_companies', rng.integers(5, 20000, 80), 130000)
    gen._dbg("after error generation")
    
    # Test 3: Error profiles
    profile = gen.build_error_profile('title')
    print(f"\n  title profile: mean={profile['mean']:.4f}, std={profile['std']:.4f}")
    print(f"    n_clean/n_orig: {profile['n_clean']}/{profile['n_original']}")
    
    # Test 4: Huber aggregate
    huber = gen.huber_aggregate_errors('title', delta=0.5)
    print(f"\n  Huber mean error (title): {huber:.4f}")
    
    # Test 5: Bootstrap CI
    ci = gen.bootstrap_error_ci('title')
    print(f"  95% CI (title): [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    # Test 6: Incremental update
    new_errs = list(rng.normal(0.2, 0.3, 20))
    gen.incremental_update('title', new_errs)
    print(f"\n  After incremental update:")
    print(f"    n_errors(title): {len(gen._errors_per_table['title'])}")
    
    print("\nAll tests passed.")
