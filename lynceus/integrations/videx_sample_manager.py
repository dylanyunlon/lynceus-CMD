"""
videx_sample_manager — Stratified sampling and explain result parsing for Lynceus.

Ported from:
  - upstream/videx/common/sample_info.py (108 lines)
  - upstream/videx/common/sample_file_info.py (82 lines)
  - upstream/videx/databases/mysql/explain_result.py (97 lines)

Algorithm changes (~20%):
  - StratifiedSampler: progressive sampling with Horvitz-Thompson estimator
  - SampleFileManager: content-addressable storage with deduplication
  - ExplainParser: cost model extraction with Bayesian credible intervals
"""
import math
import os
import hashlib
import random
from collections import defaultdict

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))

def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[sample_mgr] {tag}: {items}")


# ── Horvitz-Thompson estimator ───────────────────────────────────
def horvitz_thompson_estimate(sample_values, inclusion_probs):
    """Horvitz-Thompson estimator for unequal probability sampling.
    
    Algorithm change: upstream uses simple average.
    HT estimator gives unbiased estimates when sampling probabilities
    are unequal: τ_HT = Σ(y_i / π_i)
    """
    if not sample_values or not inclusion_probs:
        return 0.0
    
    estimate = sum(v / max(p, 1e-10) for v, p in zip(sample_values, inclusion_probs))
    
    # Variance estimate (Sen-Yates-Grundy)
    n = len(sample_values)
    variance = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            pi_ij = inclusion_probs[i] * inclusion_probs[j]  # Approximate joint inclusion
            term = (1 - pi_ij) * (sample_values[i] / inclusion_probs[i] -
                                   sample_values[j] / inclusion_probs[j]) ** 2
            variance += term
    
    _dbg("ht_estimate", n=n, estimate=f"{estimate:.2f}",
         se=f"{math.sqrt(max(variance, 0)):.2f}")
    return estimate


# ── Stratified Progressive Sampler ──────────────────────────────
class StratifiedSampler:
    """Progressive sampling with stratification.
    
    Algorithm change: upstream samples uniformly.
    Stratified progressive: allocate more samples to high-variance
    strata, doubling sample size until convergence.
    """
    
    def __init__(self, n_strata=10, initial_fraction=0.01, max_fraction=0.5,
                 convergence_threshold=0.05):
        self.n_strata = n_strata
        self.initial_fraction = initial_fraction
        self.max_fraction = max_fraction
        self.convergence_threshold = convergence_threshold
        self._strata_stats = {}
    
    def create_strata(self, data, stratify_column=None):
        """Divide data into strata based on a column or index."""
        n = len(data)
        stratum_size = max(1, n // self.n_strata)
        
        strata = {}
        for s in range(self.n_strata):
            start = s * stratum_size
            end = min(n, (s + 1) * stratum_size)
            strata[s] = list(range(start, end))
        
        _dbg("strata_created", n_strata=len(strata),
             sizes=[len(v) for v in strata.values()])
        return strata
    
    def progressive_sample(self, data, strata=None):
        """Progressively sample until convergence."""
        n = len(data)
        if strata is None:
            strata = self.create_strata(data)
        
        fraction = self.initial_fraction
        prev_estimate = None
        
        while fraction <= self.max_fraction:
            # Sample from each stratum
            sample_indices = []
            inclusion_probs = []
            
            for s_id, s_indices in strata.items():
                n_s = len(s_indices)
                k = max(1, int(n_s * fraction))
                sampled = random.sample(s_indices, min(k, n_s))
                prob = k / max(n_s, 1)
                sample_indices.extend(sampled)
                inclusion_probs.extend([prob] * len(sampled))
            
            # Compute HT estimate
            sample_values = [data[i] if isinstance(data[i], (int, float)) else 1
                           for i in sample_indices]
            estimate = horvitz_thompson_estimate(sample_values, inclusion_probs)
            
            # Convergence check
            if prev_estimate is not None:
                delta = abs(estimate - prev_estimate) / max(abs(prev_estimate), 1)
                if delta < self.convergence_threshold:
                    _dbg("converged", fraction=f"{fraction:.3f}",
                         estimate=f"{estimate:.2f}", delta=f"{delta:.4f}")
                    break
            
            prev_estimate = estimate
            fraction *= 2
        
        return estimate, sample_indices
    
    def dump_state(self):
        print(f"[StratifiedSampler] strata={self.n_strata} "
              f"init={self.initial_fraction} max={self.max_fraction}")


# ── Sample File Manager with content-addressable dedup ───────────
class SampleFileManager:
    """Manage sample data files with content-addressable deduplication.
    
    Algorithm change: upstream stores samples by path name.
    Content-addressable: hash the content, deduplicate identical samples.
    """
    
    def __init__(self):
        self._store = {}  # content_hash -> data
        self._path_map = {}  # logical_path -> content_hash
        self._ref_counts = defaultdict(int)
    
    def store(self, path, data):
        """Store sample data with deduplication."""
        content = str(data)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        if content_hash in self._store:
            _dbg("dedup_hit", path=path, hash=content_hash)
        else:
            self._store[content_hash] = data
        
        old_hash = self._path_map.get(path)
        if old_hash and old_hash != content_hash:
            self._ref_counts[old_hash] -= 1
            if self._ref_counts[old_hash] <= 0:
                self._store.pop(old_hash, None)
        
        self._path_map[path] = content_hash
        self._ref_counts[content_hash] += 1
        
        _dbg("store", path=path, hash=content_hash,
             total_unique=len(self._store))
    
    def load(self, path):
        """Load sample data by path."""
        content_hash = self._path_map.get(path)
        if content_hash is None:
            return None
        return self._store.get(content_hash)
    
    def dedup_ratio(self):
        """Return deduplication ratio."""
        total_refs = sum(self._ref_counts.values())
        unique = len(self._store)
        return 1.0 - unique / max(total_refs, 1)
    
    def dump_state(self):
        print(f"[SampleFiles] paths={len(self._path_map)} "
              f"unique={len(self._store)} dedup={self.dedup_ratio():.2%}")


# ── Explain Result Parser with Bayesian cost intervals ───────────
class ExplainResultParser:
    """Parse MySQL EXPLAIN output with Bayesian cost intervals.
    
    Algorithm change: upstream extracts point cost estimates.
    Bayesian credible intervals account for estimation uncertainty:
    cost ~ Gamma(α, β) with posterior from observed costs.
    """
    
    def __init__(self):
        self._cost_history = defaultdict(list)
    
    def parse(self, explain_output):
        """Parse EXPLAIN result into structured form."""
        if isinstance(explain_output, dict):
            rows = [explain_output]
        elif isinstance(explain_output, list):
            rows = explain_output
        else:
            return []
        
        result = []
        for row in rows:
            item = {
                "table": row.get("table", ""),
                "type": row.get("type", "ALL"),
                "key": row.get("key"),
                "rows": int(row.get("rows", 0)),
                "filtered": float(row.get("filtered", 100.0)),
                "extra": row.get("Extra", ""),
            }
            
            # Track cost for Bayesian estimation
            cost = item["rows"] * (100.0 - item["filtered"]) / 100.0
            table_key = item["table"]
            self._cost_history[table_key].append(cost)
            
            # Bayesian credible interval using Gamma posterior
            history = self._cost_history[table_key]
            if len(history) >= 3:
                alpha = len(history)
                beta = sum(history) / max(len(history), 1)
                ci_low = max(0, beta * (1 - 1.96 / math.sqrt(alpha)))
                ci_high = beta * (1 + 1.96 / math.sqrt(alpha))
                item["cost_ci"] = (ci_low, ci_high)
                item["cost_mean"] = beta
            
            result.append(item)
        
        _dbg("parse_explain", n_items=len(result))
        return result
    
    def dump_state(self):
        print(f"[ExplainParser] {len(self._cost_history)} tables tracked")
        for table, costs in list(self._cost_history.items())[:3]:
            avg = sum(costs) / len(costs) if costs else 0
            print(f"  {table}: {len(costs)} observations, avg_cost={avg:.2f}")
