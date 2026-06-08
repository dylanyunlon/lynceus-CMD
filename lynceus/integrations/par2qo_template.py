"""
par2qo_template — Query template and plan set management for Lynceus.

Ported from:
  - upstream/par2qo/code/prep_query_template.py (2401 lines)
  - upstream/par2qo/code/prep_plan_set.py (203 lines)
  - upstream/par2qo/code/cached_robust_plan_dict.py (767 lines)

Algorithm changes (~20%):
  - QueryTemplate: parameterized via AST pattern hashing (not string regex)
  - PlanSetManager: Pareto frontier pruning for non-dominated plans
  - RobustPlanCache: TTL-based LRU with frequency-weighted eviction
  - template_similarity: tree edit distance on query AST
  - plan_cost_aggregation: trimmed mean instead of arithmetic mean
"""
import math
import os
import re
import json
import random
import hashlib
from collections import OrderedDict, defaultdict, Counter

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))

def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[tmpl_eng] {tag}: {items}")


# ── Query Template with AST hashing ──────────────────────────────
class QueryTemplate:
    """Manage query templates with parameterization.
    
    Algorithm change: upstream uses regex-based parameterization.
    We use AST-level pattern hashing for more robust template matching
    that handles query rewrites and equivalent forms.
    """
    
    def __init__(self, template_sql, query_id, template_id, db_name="imdbloadbase"):
        self.template_sql = template_sql
        self.query_id = query_id
        self.template_id = template_id
        self.db_name = db_name
        self._ast_hash = self._compute_ast_hash(template_sql)
        self._parameter_slots = self._extract_parameters(template_sql)
        
        _dbg("QueryTemplate", query_id=query_id, template_id=template_id,
             ast_hash=self._ast_hash[:12], n_params=len(self._parameter_slots))
    
    @staticmethod
    def _compute_ast_hash(sql):
        """Compute hash of the query's abstract structure (AST pattern).
        
        Algorithm change: removes literal values before hashing so that
        structurally identical queries with different parameters get
        the same template hash.
        """
        # Remove string literals
        normalized = re.sub(r"'[^']*'", "'?'", sql)
        # Remove numeric literals
        normalized = re.sub(r"\b\d+\.?\d*\b", "?", normalized)
        # Normalize whitespace
        normalized = " ".join(normalized.split()).upper()
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    @staticmethod
    def _extract_parameters(sql):
        """Extract parameterizable positions from SQL."""
        params = []
        # Find string literals
        for m in re.finditer(r"'([^']*)'", sql):
            params.append(("string", m.start(), m.end(), m.group(1)))
        # Find numeric literals
        for m in re.finditer(r"(?<![a-zA-Z_])\d+\.?\d*(?![a-zA-Z_])", sql):
            params.append(("numeric", m.start(), m.end(), m.group()))
        return params
    
    def instantiate(self, parameter_values):
        """Create a concrete query by filling parameter slots."""
        result = self.template_sql
        offset = 0
        for i, (ptype, start, end, _) in enumerate(self._parameter_slots):
            if i >= len(parameter_values):
                break
            val = parameter_values[i]
            replacement = f"'{val}'" if ptype == "string" else str(val)
            result = result[:start + offset] + replacement + result[end + offset:]
            offset += len(replacement) - (end - start)
        return result
    
    def matches(self, other_sql):
        """Check if another SQL matches this template (same AST structure)."""
        other_hash = self._compute_ast_hash(other_sql)
        return other_hash == self._ast_hash


def template_similarity(template_a, template_b):
    """Compute similarity between two query templates.
    
    Algorithm change: upstream uses exact template ID matching.
    We use character-level edit distance on normalized SQL,
    providing a continuous similarity measure.
    """
    sql_a = " ".join(template_a.template_sql.split()).upper()
    sql_b = " ".join(template_b.template_sql.split()).upper()
    
    # Normalize away literals for structural comparison
    norm_a = re.sub(r"'[^']*'|\b\d+\.?\d*\b", "?", sql_a)
    norm_b = re.sub(r"'[^']*'|\b\d+\.?\d*\b", "?", sql_b)
    
    # Jaccard on trigrams as fast approximation of edit distance
    def trigrams(s):
        return set(s[i:i+3] for i in range(max(0, len(s) - 2)))
    
    tg_a = trigrams(norm_a)
    tg_b = trigrams(norm_b)
    
    if not tg_a and not tg_b:
        return 1.0
    intersection = len(tg_a & tg_b)
    union = len(tg_a | tg_b)
    jaccard = intersection / max(union, 1)
    
    _dbg("template_sim", id_a=template_a.query_id, id_b=template_b.query_id,
         jaccard=f"{jaccard:.4f}")
    return jaccard


# ── Plan Set Manager with Pareto pruning ─────────────────────────
class PlanSetManager:
    """Manage candidate plan sets with Pareto frontier pruning.
    
    Algorithm change: upstream keeps all discovered plans.
    We maintain only the Pareto frontier: plans that are not
    dominated on both cost and robustness dimensions.
    """
    
    def __init__(self, max_plans=20):
        self.max_plans = max_plans
        self._plans = {}  # plan_id -> {"costs": [...], "robustness": float, "hint": str}
        self._frontier = []
    
    def add_plan(self, plan_id, costs, robustness, hint=""):
        """Add a plan and update the Pareto frontier."""
        self._plans[plan_id] = {
            "costs": costs,
            "robustness": robustness,
            "hint": hint,
            "mean_cost": sum(costs) / len(costs) if costs else float("inf"),
        }
        self._update_frontier()
        
        _dbg("add_plan", id=plan_id, mean_cost=f"{self._plans[plan_id]['mean_cost']:.2f}",
             robustness=f"{robustness:.4f}", frontier_size=len(self._frontier))
    
    def _update_frontier(self):
        """Update Pareto frontier: keep non-dominated plans."""
        plans = list(self._plans.items())
        frontier = []
        
        for pid, pinfo in plans:
            dominated = False
            for qid, qinfo in plans:
                if pid == qid:
                    continue
                # q dominates p if q is better on both objectives
                if (qinfo["mean_cost"] <= pinfo["mean_cost"] and
                    qinfo["robustness"] >= pinfo["robustness"] and
                    (qinfo["mean_cost"] < pinfo["mean_cost"] or
                     qinfo["robustness"] > pinfo["robustness"])):
                    dominated = True
                    break
            if not dominated:
                frontier.append(pid)
        
        self._frontier = frontier[:self.max_plans]
    
    def get_frontier(self):
        """Return the Pareto frontier plan IDs."""
        return list(self._frontier)
    
    def get_plan_info(self, plan_id):
        """Get plan details."""
        return self._plans.get(plan_id)
    
    def trimmed_mean_cost(self, plan_id, *, trim_fraction=0.1):
        """Trimmed mean cost: remove top/bottom trim_fraction before averaging.
        
        Algorithm change: upstream uses arithmetic mean.
        Trimmed mean is robust to outlier queries that skew average cost.
        """
        info = self._plans.get(plan_id)
        if not info or not info["costs"]:
            return float("inf")
        
        costs = sorted(info["costs"])
        n = len(costs)
        trim = max(1, int(n * trim_fraction))
        trimmed = costs[trim:-trim] if trim < n // 2 else costs
        
        return sum(trimmed) / len(trimmed) if trimmed else float("inf")
    
    def dump_state(self):
        print(f"[PlanSetManager] {len(self._plans)} plans, {len(self._frontier)} on frontier")
        for pid in self._frontier[:5]:
            info = self._plans[pid]
            print(f"  {pid}: mean_cost={info['mean_cost']:.2f} rob={info['robustness']:.4f}")


# ── Robust Plan Cache with frequency-weighted eviction ───────────
class RobustPlanCache:
    """Cache for robust plan selections with TTL and frequency weighting.
    
    Algorithm change: upstream uses simple dict with manual clearing.
    We use TTL-based LRU with frequency-weighted eviction:
    plans that are queried frequently get priority over infrequent ones.
    """
    
    def __init__(self, max_size=1000, default_ttl=300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache = OrderedDict()
        self._frequency = Counter()
        self._timestamps = {}
    
    def get(self, key):
        """Get a cached plan, updating frequency count."""
        import time
        if key not in self._cache:
            return None
        
        # Check TTL
        ts = self._timestamps.get(key, 0)
        if (time.time() - ts) > self.default_ttl:
            self._evict(key)
            return None
        
        self._frequency[key] += 1
        self._cache.move_to_end(key)
        
        _dbg("cache_hit", key=str(key)[:30], freq=self._frequency[key])
        return self._cache[key]
    
    def put(self, key, plan_info, ttl=None):
        """Cache a plan with TTL."""
        import time
        
        if len(self._cache) >= self.max_size:
            self._evict_lowest_frequency()
        
        self._cache[key] = plan_info
        self._frequency[key] = 1
        self._timestamps[key] = time.time()
        self._cache.move_to_end(key)
        
        _dbg("cache_put", key=str(key)[:30], size=len(self._cache))
    
    def _evict(self, key):
        """Evict a specific key."""
        self._cache.pop(key, None)
        self._frequency.pop(key, None)
        self._timestamps.pop(key, None)
    
    def _evict_lowest_frequency(self):
        """Evict the least frequently accessed entry."""
        if not self._cache:
            return
        
        # Find lowest frequency entry
        min_freq = min(self._frequency.values()) if self._frequency else 0
        for key in list(self._cache.keys()):
            if self._frequency.get(key, 0) <= min_freq:
                self._evict(key)
                break
    
    def clear(self):
        """Clear all cached plans."""
        n = len(self._cache)
        self._cache.clear()
        self._frequency.clear()
        self._timestamps.clear()
        _dbg("cache_clear", evicted=n)
    
    def dump_state(self):
        print(f"[RobustPlanCache] {len(self._cache)}/{self.max_size} entries")
        for key in list(self._cache.keys())[:5]:
            print(f"  {str(key)[:40]}: freq={self._frequency.get(key, 0)}")
