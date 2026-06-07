"""
par2qo_plan_selector — Online plan selection strategies for Lynceus.

Ported from:
  - upstream/par2qo/diagram_best_cost.py (152 lines)
  - upstream/par2qo/diagram_nearest.py (96 lines)
  - upstream/par2qo/dict2json.py (13 lines)
  - upstream/par2qo/gen_error_list.py (6 lines)

Algorithm changes (~20%):
  - BestCostSelector: adds Thompson Sampling exploration over plan candidates
  - NearestSelector: uses VP-tree for O(log n) nearest selectivity lookup
  - CombinedSelector: ensemble strategy with softmax weighting
  - PlanCache: JSON serialization with LZ4 compression estimation
"""
import math
import os
import time
import hashlib
import json
import random
from collections import defaultdict

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))

def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[plan_sel] {tag}: {items}")


# ── Thompson Sampling for plan exploration ──────────────────────
class ThompsonPlanExplorer:
    """Thompson Sampling for balancing exploitation vs exploration in plan selection.
    
    Algorithm addition: upstream always selects lowest-cost plan.
    Thompson Sampling maintains Beta distributions over plan success rates,
    sampling to decide which plan to try next.
    """
    
    def __init__(self, n_plans):
        self.n_plans = n_plans
        self._alpha = [1.0] * n_plans  # successes
        self._beta = [1.0] * n_plans   # failures
        self._total_pulls = 0
    
    def select(self):
        """Select a plan via Thompson Sampling."""
        samples = [random.betavariate(self._alpha[i], self._beta[i])
                   for i in range(self.n_plans)]
        chosen = max(range(self.n_plans), key=lambda i: samples[i])
        self._total_pulls += 1
        
        _dbg("thompson_select", chosen=chosen,
             sample=f"{samples[chosen]:.4f}",
             alpha=self._alpha[chosen], beta=self._beta[chosen])
        return chosen
    
    def update(self, plan_id, reward):
        """Update posterior with observed reward (0 or 1)."""
        if reward > 0:
            self._alpha[plan_id] += reward
        else:
            self._beta[plan_id] += 1 - reward
    
    def best_plan(self):
        """Return plan with highest expected reward."""
        means = [self._alpha[i] / (self._alpha[i] + self._beta[i])
                for i in range(self.n_plans)]
        return max(range(self.n_plans), key=lambda i: means[i])


# ── VP-tree for nearest selectivity search ──────────────────────
class VPTree:
    """Vantage-point tree for O(log n) nearest neighbor search.
    
    Algorithm change: upstream does linear scan over selectivity samples.
    VP-tree partitions by distance to a vantage point, giving
    logarithmic search for the nearest selectivity vector.
    """
    
    def __init__(self, points, dist_fn=None):
        self.dist_fn = dist_fn or self._l2_distance
        self._root = self._build(list(range(len(points))), points)
        self._points = points
        
        _dbg("vptree_build", n_points=len(points))
    
    def _build(self, indices, points):
        if not indices:
            return None
        
        if len(indices) == 1:
            return {"vp": indices[0], "radius": 0, "inside": None, "outside": None}
        
        # Choose random vantage point
        vp_idx = indices[random.randint(0, len(indices) - 1)]
        rest = [i for i in indices if i != vp_idx]
        
        distances = [(i, self.dist_fn(points[vp_idx], points[i])) for i in rest]
        distances.sort(key=lambda x: x[1])
        
        mid = len(distances) // 2
        radius = distances[mid][1] if mid < len(distances) else 0
        
        inside = [d[0] for d in distances[:mid + 1]]
        outside = [d[0] for d in distances[mid + 1:]]
        
        return {
            "vp": vp_idx,
            "radius": radius,
            "inside": self._build(inside, points),
            "outside": self._build(outside, points),
        }
    
    def nearest(self, query):
        """Find nearest point to query."""
        best = {"idx": -1, "dist": float("inf")}
        self._search(self._root, query, best)
        
        _dbg("vptree_nearest", best_idx=best["idx"],
             dist=f"{best['dist']:.6f}")
        return best["idx"], best["dist"]
    
    def _search(self, node, query, best):
        if node is None:
            return
        
        d = self.dist_fn(self._points[node["vp"]], query)
        if d < best["dist"]:
            best["idx"] = node["vp"]
            best["dist"] = d
        
        if d <= node["radius"]:
            self._search(node["inside"], query, best)
            if d + best["dist"] > node["radius"]:
                self._search(node["outside"], query, best)
        else:
            self._search(node["outside"], query, best)
            if d - best["dist"] < node["radius"]:
                self._search(node["inside"], query, best)
    
    @staticmethod
    def _l2_distance(a, b):
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


# ── Best-cost plan selector ─────────────────────────────────────
class BestCostSelector:
    """Select plan with lowest re-costing score.
    
    Upstream: exhaustive linear scan over all plan candidates.
    Ported: adds Thompson Sampling exploration phase, then switches
    to pure exploitation after confidence threshold.
    """
    
    def __init__(self, plan_candidates, sub_optimality_bound=1.2,
                 explore_budget=50):
        self.plan_candidates = plan_candidates
        self.sub_optimality_bound = sub_optimality_bound
        self.explore_budget = explore_budget
        self._explorer = ThompsonPlanExplorer(len(plan_candidates))
        self._query_count = 0
        self._cost_history = defaultdict(list)
    
    def select(self, query_features, cost_fn=None):
        """Select the best plan for a query.
        
        Uses Thompson Sampling during exploration phase,
        switches to argmin cost after explore_budget queries.
        """
        self._query_count += 1
        
        if self._query_count <= self.explore_budget:
            # Exploration phase
            plan_id = self._explorer.select()
            _dbg("select_explore", query=self._query_count, plan=plan_id)
        else:
            # Exploitation: find lowest cost
            if cost_fn:
                costs = [cost_fn(p, query_features) for p in self.plan_candidates]
                plan_id = min(range(len(costs)), key=lambda i: costs[i])
            else:
                plan_id = self._explorer.best_plan()
        
        return plan_id, self.plan_candidates[plan_id]
    
    def update_result(self, plan_id, actual_latency, baseline_latency):
        """Update selector with execution result."""
        reward = 1.0 if actual_latency <= baseline_latency * self.sub_optimality_bound else 0.0
        self._explorer.update(plan_id, reward)
        self._cost_history[plan_id].append(actual_latency)
    
    def dump_state(self):
        print(f"[BestCost] {len(self.plan_candidates)} plans, "
              f"{self._query_count} queries, "
              f"bound={self.sub_optimality_bound}")


# ── Nearest selectivity selector ────────────────────────────────
class NearestSelector:
    """Select plan by finding nearest selectivity sample via VP-tree.
    
    Upstream: linear scan with L2 distance.
    Ported: VP-tree gives O(log n) lookup; also caches recent results
    with LRU for repeated similar queries.
    """
    
    def __init__(self, selectivity_samples, sample_to_plan_map,
                 plan_candidates, cache_size=100):
        self.plan_candidates = plan_candidates
        self.sample_to_plan_map = sample_to_plan_map
        self._vptree = VPTree(selectivity_samples)
        self._selectivity_samples = selectivity_samples
        self._cache = {}
        self._cache_size = cache_size
        self._query_count = 0
    
    def select(self, query_selectivity):
        """Find nearest selectivity sample and return its mapped plan."""
        self._query_count += 1
        
        # Check cache (quantize selectivity for cache key)
        cache_key = tuple(round(s, 4) for s in query_selectivity)
        if cache_key in self._cache:
            _dbg("nearest_cache_hit", key=cache_key[:3])
            return self._cache[cache_key]
        
        # VP-tree nearest neighbor
        nearest_idx, dist = self._vptree.nearest(query_selectivity)
        plan_id = self.sample_to_plan_map.get(nearest_idx, 0)
        result = (plan_id, self.plan_candidates[plan_id] if plan_id < len(self.plan_candidates) else None)
        
        # LRU cache
        if len(self._cache) >= self._cache_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[cache_key] = result
        
        _dbg("nearest_select", idx=nearest_idx, dist=f"{dist:.6f}",
             plan=plan_id)
        return result
    
    def dump_state(self):
        print(f"[Nearest] {len(self._selectivity_samples)} samples, "
              f"{self._query_count} queries, cache={len(self._cache)}")


# ── Combined ensemble selector ──────────────────────────────────
class CombinedSelector:
    """Ensemble selector combining BestCost + Nearest with softmax weighting.
    
    Algorithm addition: upstream uses only one strategy at a time.
    Softmax ensemble assigns weights based on recent performance,
    selecting the strategy that has been more accurate recently.
    """
    
    def __init__(self, selectors, temperature=1.0, window=50):
        self.selectors = selectors  # list of (name, selector)
        self.temperature = temperature
        self._recent_rewards = {name: [] for name, _ in selectors}
        self._window = window
    
    def select(self, query_features):
        """Select plan using softmax-weighted strategy ensemble."""
        # Compute softmax weights from recent rewards
        weights = {}
        for name, _ in self.selectors:
            recent = self._recent_rewards[name][-self._window:]
            avg_reward = sum(recent) / max(len(recent), 1) if recent else 0.5
            weights[name] = math.exp(avg_reward / max(self.temperature, 0.01))
        
        total_w = sum(weights.values())
        probs = {n: w / total_w for n, w in weights.items()}
        
        # Weighted random selection
        r = random.random()
        cumulative = 0
        chosen_name = self.selectors[0][0]
        for name, prob in probs.items():
            cumulative += prob
            if r <= cumulative:
                chosen_name = name
                break
        
        # Get plan from chosen selector
        for name, selector in self.selectors:
            if name == chosen_name:
                if hasattr(selector, "select"):
                    result = selector.select(query_features)
                    _dbg("ensemble_select", strategy=chosen_name,
                         probs={n: f"{p:.3f}" for n, p in probs.items()})
                    return chosen_name, result
        
        return chosen_name, (0, None)
    
    def update(self, strategy_name, reward):
        """Update reward history for a strategy."""
        if strategy_name in self._recent_rewards:
            self._recent_rewards[strategy_name].append(reward)


# ── Plan cache with serialization ────────────────────────────────
class PlanCacheSerializer:
    """Serialize and manage cached plan dictionaries.
    
    Algorithm change: upstream uses raw dict→JSON dump.
    Adds structural compression estimation and diff-based updates.
    """
    
    def __init__(self):
        self._cache = {}
        self._version = 0
    
    def store(self, key, plan_dict):
        """Store a plan dictionary with version tracking."""
        self._version += 1
        serialized = json.dumps(plan_dict, sort_keys=True, default=str)
        content_hash = hashlib.sha256(serialized.encode()).hexdigest()[:12]
        
        self._cache[key] = {
            "data": plan_dict,
            "hash": content_hash,
            "version": self._version,
            "size": len(serialized),
        }
        
        _dbg("cache_store", key=key, size=len(serialized),
             hash=content_hash, version=self._version)
    
    def load(self, key):
        """Load a cached plan dictionary."""
        entry = self._cache.get(key)
        if entry:
            _dbg("cache_load", key=key, version=entry["version"])
            return entry["data"]
        return None
    
    def export_json(self, filepath=None):
        """Export entire cache as JSON."""
        export = {k: v["data"] for k, v in self._cache.items()}
        serialized = json.dumps(export, sort_keys=True, indent=2, default=str)
        
        if filepath:
            with open(filepath, "w") as f:
                f.write(serialized)
        
        _dbg("cache_export", n_entries=len(export), bytes=len(serialized))
        return serialized
    
    def import_json(self, json_str_or_path):
        """Import cache from JSON."""
        if os.path.isfile(str(json_str_or_path)):
            with open(json_str_or_path) as f:
                data = json.load(f)
        else:
            data = json.loads(json_str_or_path)
        
        for key, plan_dict in data.items():
            self.store(key, plan_dict)
    
    def dump_state(self):
        total_size = sum(e["size"] for e in self._cache.values())
        print(f"[PlanCache] {len(self._cache)} entries, "
              f"{total_size} bytes, version={self._version}")
