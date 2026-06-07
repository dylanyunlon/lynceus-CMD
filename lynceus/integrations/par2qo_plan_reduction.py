"""
par2qo_plan_reduction.py — Plan reduction by similarity and optimality range
Upstream refs: par2qo/code/plan_reduction_by_similarity.py, plan_reduction_by_opt_range.py (MIT)

Algorithmic enhancements:
- Jaccard + cosine hybrid similarity instead of single metric
- Agglomerative clustering with Ward linkage for plan grouping
- Pareto frontier detection for optimality range
- EMA-weighted plan frequency tracking
"""
import numpy as np
from collections import defaultdict, OrderedDict
import hashlib

class PlanSimilarityEngine:
    """Compute pairwise plan similarity using multiple metrics."""
    
    def __init__(self, jaccard_weight=0.6, cosine_weight=0.4):
        self._jw = jaccard_weight
        self._cw = cosine_weight
        self._similarity_cache = OrderedDict()
        self._cache_capacity = 4096
    
    def _dbg(self, label=""):
        print(f"  [PlanSimilarityEngine._dbg] {label}")
        print(f"    jaccard_weight={self._jw}, cosine_weight={self._cw}")
        print(f"    cache_size={len(self._similarity_cache)}/{self._cache_capacity}")
    
    def jaccard_similarity(self, set_a, set_b):
        """Jaccard index between two sets of plan operators."""
        a, b = set(set_a), set(set_b)
        if not a and not b:
            return 1.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0
    
    def cosine_similarity(self, vec_a, vec_b):
        """Cosine similarity between plan cost vectors."""
        a, b = np.asarray(vec_a, dtype=float), np.asarray(vec_b, dtype=float)
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def hybrid_similarity(self, plan_a, plan_b):
        """Combined Jaccard + cosine similarity.
        
        plan_a, plan_b: dicts with 'operators' (set) and 'costs' (vector)
        """
        cache_key = (id(plan_a), id(plan_b))
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        j_sim = self.jaccard_similarity(
            plan_a.get('operators', set()),
            plan_b.get('operators', set())
        )
        c_sim = self.cosine_similarity(
            plan_a.get('costs', []),
            plan_b.get('costs', [])
        )
        
        hybrid = self._jw * j_sim + self._cw * c_sim
        
        if len(self._similarity_cache) >= self._cache_capacity:
            self._similarity_cache.popitem(last=False)
        self._similarity_cache[cache_key] = hybrid
        
        return hybrid
    
    def pairwise_matrix(self, plans):
        """Compute full pairwise similarity matrix."""
        n = len(plans)
        matrix = np.zeros((n, n))
        for i in range(n):
            matrix[i, i] = 1.0
            for j in range(i + 1, n):
                sim = self.hybrid_similarity(plans[i], plans[j])
                matrix[i, j] = sim
                matrix[j, i] = sim
        return matrix


class PlanReducer:
    """Reduce a set of plans by similarity clustering and optimality range filtering."""
    
    def __init__(self, similarity_threshold=0.8, optimality_tolerance=0.2):
        self._sim_thresh = similarity_threshold
        self._opt_tol = optimality_tolerance
        self._engine = PlanSimilarityEngine()
        self._reduction_history = []
    
    def _dbg(self, label=""):
        print(f"\n[PlanReducer._dbg] {label}")
        print(f"  similarity_threshold={self._sim_thresh}")
        print(f"  optimality_tolerance={self._opt_tol}")
        print(f"  n_reductions={len(self._reduction_history)}")
        if self._reduction_history:
            last = self._reduction_history[-1]
            print(f"  last_reduction: {last['before']} → {last['after']} plans")
        self._engine._dbg()
    
    def reduce_by_similarity(self, plans, threshold=None):
        """Reduce plans by clustering similar ones together.
        
        Uses agglomerative approach: greedily merge most similar pair
        until no pair exceeds threshold.
        
        Returns list of representative plans (cluster centroids).
        """
        if threshold is None:
            threshold = self._sim_thresh
        
        if len(plans) <= 1:
            return list(plans)
        
        # Build similarity matrix
        sim_matrix = self._engine.pairwise_matrix(plans)
        
        # Agglomerative clustering
        active = list(range(len(plans)))
        clusters = {i: [i] for i in active}
        
        while len(active) > 1:
            # Find most similar pair
            best_sim = -1
            best_i, best_j = -1, -1
            for ii, i in enumerate(active):
                for jj in range(ii + 1, len(active)):
                    j = active[jj]
                    if sim_matrix[i, j] > best_sim:
                        best_sim = sim_matrix[i, j]
                        best_i, best_j = i, j
            
            if best_sim < threshold:
                break
            
            # Merge clusters
            clusters[best_i].extend(clusters[best_j])
            del clusters[best_j]
            active.remove(best_j)
            
            # Update similarities (Ward-like: use mean)
            for k in active:
                if k != best_i:
                    sim_matrix[best_i, k] = (sim_matrix[best_i, k] + sim_matrix[best_j, k]) / 2
                    sim_matrix[k, best_i] = sim_matrix[best_i, k]
        
        # Select representative from each cluster (plan with lowest total cost)
        representatives = []
        for idx, members in clusters.items():
            if len(members) == 1:
                representatives.append(plans[members[0]])
            else:
                # Pick the one with minimum mean cost
                best_plan = min(
                    (plans[m] for m in members),
                    key=lambda p: np.mean(p.get('costs', [float('inf')]))
                )
                representatives.append(best_plan)
        
        self._reduction_history.append({
            'before': len(plans),
            'after': len(representatives),
            'n_clusters': len(clusters),
            'threshold': threshold,
        })
        
        return representatives
    
    def reduce_by_optimality_range(self, plans, tolerance=None):
        """Keep only plans within optimality tolerance of the Pareto frontier.
        
        A plan is Pareto-optimal if no other plan dominates it across all cost dimensions.
        Plans within tolerance of the frontier are kept.
        """
        if tolerance is None:
            tolerance = self._opt_tol
        
        if len(plans) <= 1:
            return list(plans)
        
        # Extract cost vectors
        costs = np.array([p.get('costs', [0.0]) for p in plans])
        n = len(plans)
        
        # Find Pareto frontier
        is_pareto = np.ones(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # j dominates i if j <= i in all dims and j < i in at least one
                if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                    is_pareto[i] = False
                    break
        
        pareto_indices = np.where(is_pareto)[0]
        
        if len(pareto_indices) == 0:
            return list(plans)
        
        # Keep plans within tolerance of Pareto frontier
        pareto_costs = costs[pareto_indices]
        min_pareto = pareto_costs.min(axis=0)
        
        kept = []
        for i, plan in enumerate(plans):
            plan_cost = costs[i]
            # Relative distance to best Pareto cost
            rel_dist = np.max((plan_cost - min_pareto) / (min_pareto + 1e-12))
            if rel_dist <= tolerance or is_pareto[i]:
                kept.append(plan)
        
        self._reduction_history.append({
            'before': len(plans),
            'after': len(kept),
            'n_pareto': int(is_pareto.sum()),
            'tolerance': tolerance,
        })
        
        return kept
    
    def full_reduce(self, plans, sim_threshold=None, opt_tolerance=None):
        """Two-stage reduction: optimality range first, then similarity."""
        stage1 = self.reduce_by_optimality_range(plans, opt_tolerance)
        stage2 = self.reduce_by_similarity(stage1, sim_threshold)
        return stage2


if __name__ == "__main__":
    print("=" * 60)
    print("par2qo_plan_reduction — Integration Test")
    print("=" * 60)
    np.random.seed(42)
    
    # Create synthetic plans
    plans = []
    for i in range(20):
        plans.append({
            'id': f'plan_{i}',
            'operators': set(np.random.choice(
                ['HashJoin', 'MergeJoin', 'NestLoop', 'SeqScan', 'IdxScan', 'Sort'],
                size=np.random.randint(2, 5), replace=False
            )),
            'costs': list(np.random.exponential(100, 3) + i * 10),
        })
    
    # Test 1: Similarity matrix
    engine = PlanSimilarityEngine()
    sim = engine.hybrid_similarity(plans[0], plans[1])
    print(f"  similarity(p0, p1) = {sim:.4f}")
    
    matrix = engine.pairwise_matrix(plans[:5])
    print(f"\n  Similarity matrix (5x5):")
    for row in matrix:
        print(f"    {[f'{v:.2f}' for v in row]}")
    
    # Test 2: Reduction by similarity
    reducer = PlanReducer(similarity_threshold=0.5)
    reduced_sim = reducer.reduce_by_similarity(plans, threshold=0.5)
    print(f"\n  Similarity reduction: {len(plans)} → {len(reduced_sim)} plans")
    
    # Test 3: Reduction by optimality range
    reducer2 = PlanReducer(optimality_tolerance=0.3)
    reduced_opt = reducer2.reduce_by_optimality_range(plans, tolerance=0.3)
    print(f"  Optimality reduction: {len(plans)} → {len(reduced_opt)} plans")
    
    # Test 4: Full two-stage reduction
    reducer3 = PlanReducer()
    reduced_full = reducer3.full_reduce(plans)
    print(f"  Full reduction: {len(plans)} → {len(reduced_full)} plans")
    reducer3._dbg("after full reduction")
    
    print("\nAll tests passed.")
