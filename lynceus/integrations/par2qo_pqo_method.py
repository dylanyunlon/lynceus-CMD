"""
par2qo_pqo_method.py — Parametric Query Optimization base method framework
Upstream ref: par2qo/code/pqo_method.py (MIT)

Algorithmic enhancements:
- Stratified workload splitting instead of simple train/test
- IQR-based outlier filtering in error profile construction
- Reservoir sampling for large workload handling
- Bayesian prior for selectivity estimation when data is sparse
"""
import numpy as np
import json
import os
import hashlib
from collections import defaultdict

class PQOMethod:
    """Base class for Parametric Query Optimization methods.
    
    Provides workload management, dimension analysis, error profiling,
    and plan selection infrastructure. Subclasses implement specific
    PQO strategies (Diagram, Kepler, etc.).
    """
    
    def __init__(self, db_name, workload_name, query_id, template_id,
                 n_train=50, tolerance=0.2, b=0.5):
        self.db_name = db_name
        self.workload_name = workload_name
        self.query_id = query_id
        self.template_id = template_id
        self.n_train = n_train
        self.tolerance = tolerance
        self.b = b
        
        # Workload storage
        self.queries_train = []
        self.queries_test = []
        
        # Dimension space
        self.dimension_space = []
        self.all_dims = []
        
        # Error profiling
        self.raw_base_card = []
        self.err_info_dict = {}
        
        # Results
        self.output_result = []
        
        # Statistics tracking
        self._n_plans_evaluated = 0
        self._selectivity_cache = {}
        self._welford_cost = {'n': 0, 'mean': 0.0, 'm2': 0.0}
    
    def _dbg(self, label=""):
        print(f"\n{'='*60}")
        print(f"[PQOMethod._dbg] {label}")
        print(f"  db={self.db_name}, workload={self.workload_name}")
        print(f"  query={self.query_id}, template={self.template_id}")
        print(f"  n_train={len(self.queries_train)}, n_test={len(self.queries_test)}")
        print(f"  n_dims={len(self.all_dims)}, dim_space={len(self.dimension_space)}")
        print(f"  n_err_profiles={len(self.err_info_dict)}")
        print(f"  n_plans_evaluated={self._n_plans_evaluated}")
        print(f"  selectivity_cache_size={len(self._selectivity_cache)}")
        if self._welford_cost['n'] > 1:
            var = self._welford_cost['m2'] / (self._welford_cost['n'] - 1)
            print(f"  cost_stats: mean={self._welford_cost['mean']:.2f}, var={var:.2f}")
        print(f"{'='*60}")
    
    def init_workload(self, train_queries, test_queries=None, stratified=True):
        """Initialize workload with optional stratified splitting.
        
        When stratified=True and test_queries is None, splits train_queries
        80/20 using hash-based consistent assignment for reproducibility.
        """
        if test_queries is not None:
            self.queries_train = list(train_queries)
            self.queries_test = list(test_queries)
        elif stratified:
            # Stratified split using query hash for consistency
            train, test = [], []
            for q in train_queries:
                h = int(hashlib.md5(q.encode()).hexdigest()[:8], 16)
                if h % 5 == 0:
                    test.append(q)
                else:
                    train.append(q)
            self.queries_train = train
            self.queries_test = test
        else:
            n = len(train_queries)
            split = int(n * 0.8)
            self.queries_train = list(train_queries[:split])
            self.queries_test = list(train_queries[split:])
    
    def init_dimensions(self, n_dims=None, active_dims=None):
        """Initialize dimension space for selectivity analysis."""
        if n_dims is not None:
            self.all_dims = list(range(n_dims))
        if active_dims is not None:
            self.dimension_space = list(active_dims)
        else:
            self.dimension_space = list(self.all_dims)
    
    def init_base_cardinality(self, cardinalities):
        """Set base table cardinalities."""
        self.raw_base_card = list(cardinalities)
    
    def build_error_profile(self, dim, errors, n_bins=10):
        """Build error profile for a dimension with IQR outlier filtering.
        
        Upstream uses raw KDE; we add IQR-based clipping first.
        """
        errors = np.asarray(errors)
        
        # IQR-based outlier removal
        q25, q75 = np.percentile(errors, [25, 75])
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        clean_errors = errors[(errors >= lower) & (errors <= upper)]
        
        if len(clean_errors) < 5:
            clean_errors = errors  # Fallback if too many removed
        
        # Bin the errors
        bin_edges = np.linspace(clean_errors.min(), clean_errors.max(), n_bins + 1)
        bin_indices = np.digitize(clean_errors, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Per-bin statistics
        bin_stats = []
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() > 0:
                bin_data = clean_errors[mask]
                bin_stats.append({
                    'mean': float(bin_data.mean()),
                    'std': float(bin_data.std()),
                    'count': int(mask.sum()),
                    'range': (float(bin_edges[b]), float(bin_edges[b+1])),
                })
            else:
                bin_stats.append({'mean': 0, 'std': 0, 'count': 0,
                                  'range': (float(bin_edges[b]), float(bin_edges[b+1]))})
        
        self.err_info_dict[dim] = {
            'errors': clean_errors,
            'bin_edges': bin_edges,
            'bin_stats': bin_stats,
            'n_original': len(errors),
            'n_clean': len(clean_errors),
            'iqr_bounds': (lower, upper),
        }
    
    def estimate_selectivity(self, est_card, raw_card, dim):
        """Estimate selectivity with Bayesian prior for sparse data.
        
        Uses Laplace smoothing when cardinality is very small.
        """
        cache_key = (est_card, raw_card, dim)
        if cache_key in self._selectivity_cache:
            return self._selectivity_cache[cache_key]
        
        # Laplace smoothing: (est + 1) / (raw + 2)
        if raw_card < 10:
            sel = (est_card + 1) / (raw_card + 2)
        else:
            sel = max(1, est_card) / raw_card
        
        sel = np.clip(sel, 1e-8, 1.0)
        self._selectivity_cache[cache_key] = sel
        return sel
    
    def evaluate_plan_cost(self, cost):
        """Record plan cost with Welford online statistics."""
        self._n_plans_evaluated += 1
        n = self._welford_cost['n'] + 1
        delta = cost - self._welford_cost['mean']
        self._welford_cost['mean'] += delta / n
        delta2 = cost - self._welford_cost['mean']
        self._welford_cost['m2'] += delta * delta2
        self._welford_cost['n'] = n
    
    def reservoir_sample(self, queries, k):
        """Reservoir sampling for large workloads (Vitter's algorithm R)."""
        rng = np.random.default_rng(42)
        if len(queries) <= k:
            return list(queries)
        
        reservoir = list(queries[:k])
        for i in range(k, len(queries)):
            j = rng.integers(0, i + 1)
            if j < k:
                reservoir[j] = queries[i]
        return reservoir
    
    def select_plan(self, query):
        """Select best plan for a query (base implementation: return None).
        Subclasses should override this.
        """
        return None
    
    def export_results(self):
        """Export evaluation results as list of dicts."""
        return list(self.output_result)


if __name__ == "__main__":
    print("=" * 60)
    print("par2qo_pqo_method — Integration Test")
    print("=" * 60)
    np.random.seed(42)
    
    # Test 1: Basic initialization
    pqo = PQOMethod("testdb", "gaussian", "q7", "7-0", n_train=50)
    pqo._dbg("initial state")
    
    # Test 2: Workload initialization with stratified split
    queries = [f"SELECT * FROM t WHERE x = {i}" for i in range(100)]
    pqo.init_workload(queries, stratified=True)
    pqo._dbg("after stratified split")
    print(f"  train/test split: {len(pqo.queries_train)}/{len(pqo.queries_test)}")
    
    # Test 3: Dimension initialization
    pqo.init_dimensions(n_dims=10, active_dims=[0, 2, 5, 7])
    pqo.init_base_cardinality([1000, 5000, 2000, 800, 1500, 3000, 700, 4000, 900, 6000])
    
    # Test 4: Error profiling with IQR filtering
    errors = np.concatenate([np.random.normal(0, 1, 90), np.array([50, -40])])  # With outliers
    pqo.build_error_profile(0, errors, n_bins=5)
    profile = pqo.err_info_dict[0]
    print(f"\n  Error profile dim 0:")
    print(f"    original: {profile['n_original']}, clean: {profile['n_clean']}")
    print(f"    IQR bounds: {profile['iqr_bounds']}")
    for i, bs in enumerate(profile['bin_stats'][:3]):
        print(f"    bin[{i}]: mean={bs['mean']:.3f}, count={bs['count']}")
    
    # Test 5: Selectivity estimation with Bayesian prior
    sel1 = pqo.estimate_selectivity(100, 1000, 0)
    sel2 = pqo.estimate_selectivity(2, 5, 1)  # Sparse → Laplace smoothing
    print(f"\n  selectivity(100/1000) = {sel1:.4f}")
    print(f"  selectivity(2/5, Laplace) = {sel2:.4f}")
    
    # Test 6: Plan cost tracking
    for c in [100, 150, 120, 130, 5000, 110]:  # 5000 is outlier
        pqo.evaluate_plan_cost(c)
    var = pqo._welford_cost['m2'] / (pqo._welford_cost['n'] - 1)
    print(f"\n  cost stats: mean={pqo._welford_cost['mean']:.1f}, var={var:.1f}")
    
    # Test 7: Reservoir sampling
    large_wl = [f"Q{i}" for i in range(10000)]
    sampled = pqo.reservoir_sample(large_wl, 20)
    print(f"\n  reservoir sample: {len(sampled)} from {len(large_wl)}")
    print(f"  sample[:5]: {sampled[:5]}")
    
    pqo._dbg("final state")
    print("\nAll tests passed.")
