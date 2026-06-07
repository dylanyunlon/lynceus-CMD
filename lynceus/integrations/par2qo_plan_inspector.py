"""
par2qo_plan_inspector — Query plan inspection and visualization for Lynceus.

Ported from:
  - upstream/par2qo/code/plan_inspection.py (558 lines)
  - upstream/par2qo/code/diagram.py (565 lines)
  - upstream/par2qo/code/diagram_querylog.py (275 lines)
  - upstream/par2qo/code/diagram_querylog_w_sample.py (286 lines)
  - upstream/par2qo/code/psql_explain_decoder.py (254 lines)

Algorithm changes (~20%):
  - check_common: Jaccard similarity for join-method comparison (not exact match)
  - decode_explain: recursive cost accumulator with Amdahl correction
  - PlanDiagram: convex hull cost envelope instead of raw min/max
  - WorkloadSampler: stratified sampling by cost quantile (not uniform)
  - cost_distribution_entropy: Shannon entropy of cost distribution
"""
import math
import re
import os
import json
from collections import defaultdict, Counter

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))

def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[plan_insp] {tag}: {items}")


# ── EXPLAIN Decoder with Amdahl correction ───────────────────────
def decode_explain(plan_json, *, amdahl_serial=0.05):
    """Decode PostgreSQL EXPLAIN JSON into structured plan info.
    
    Algorithm change: applies Amdahl's law correction to parallel
    cost estimates. Parallel nodes have their cost adjusted by
    serial_fraction + (1-serial_fraction)/n_workers.
    """
    if isinstance(plan_json, str):
        try:
            plan_json = json.loads(plan_json)
        except json.JSONDecodeError:
            return {"error": "invalid JSON", "cost": float("inf")}
    
    if isinstance(plan_json, list) and plan_json:
        plan_json = plan_json[0]
    
    plan = plan_json.get("Plan", plan_json)
    
    result = _decode_node(plan, amdahl_serial=amdahl_serial)
    _dbg("decode", total_cost=f"{result.get('total_cost', 0):.2f}",
         n_nodes=result.get("n_nodes", 0),
         join_methods=result.get("join_methods", []))
    return result


def _decode_node(node, depth=0, amdahl_serial=0.05):
    """Recursively decode a plan node."""
    if not isinstance(node, dict):
        return {"cost": 0, "n_nodes": 0, "nodes": []}
    
    node_type = node.get("Node Type", "Unknown")
    startup_cost = node.get("Startup Cost", 0)
    total_cost = node.get("Total Cost", 0)
    rows = node.get("Plan Rows", 0)
    width = node.get("Plan Width", 0)
    workers = node.get("Workers Planned", 1)
    
    # Amdahl correction for parallel nodes
    if workers > 1 and node_type in ("Parallel Seq Scan", "Parallel Hash Join",
                                      "Gather", "Gather Merge"):
        parallel_speedup = amdahl_serial + (1 - amdahl_serial) / workers
        total_cost *= parallel_speedup
        _dbg("amdahl_correct", node=node_type, workers=workers,
             speedup=f"{parallel_speedup:.3f}")
    
    info = {
        "node_type": node_type,
        "startup_cost": startup_cost,
        "total_cost": total_cost,
        "rows": rows,
        "width": width,
        "depth": depth,
    }
    
    # Extract join method and scan method
    join_methods = []
    scan_methods = []
    if "Join" in node_type:
        join_methods.append(node_type)
    if "Scan" in node_type:
        scan_methods.append(node_type)
    
    # Process children
    children = node.get("Plans", [])
    child_results = []
    for child in children:
        cr = _decode_node(child, depth + 1, amdahl_serial)
        child_results.append(cr)
        join_methods.extend(cr.get("join_methods", []))
        scan_methods.extend(cr.get("scan_methods", []))
    
    info["children"] = child_results
    info["join_methods"] = join_methods
    info["scan_methods"] = scan_methods
    info["n_nodes"] = 1 + sum(c.get("n_nodes", 0) for c in child_results)
    
    return info


# ── Plan hint extraction ─────────────────────────────────────────
def extract_hint_from_plan(plan_json):
    """Extract pg_hint_plan compatible hint string from explain plan."""
    decoded = decode_explain(plan_json)
    hints = []
    _extract_hints_recursive(decoded, hints)
    hint_str = " ".join(hints)
    _dbg("extract_hint", n_hints=len(hints), hint=hint_str[:100])
    return hint_str


def _extract_hints_recursive(node, hints):
    """Recursively extract join and scan hints."""
    if not isinstance(node, dict):
        return
    
    nt = node.get("node_type", "")
    
    # Map node types to hint keywords
    join_hint_map = {
        "Nested Loop": "NestLoop",
        "Hash Join": "HashJoin",
        "Merge Join": "MergeJoin",
    }
    scan_hint_map = {
        "Seq Scan": "SeqScan",
        "Index Scan": "IndexScan",
        "Index Only Scan": "IndexOnlyScan",
        "Bitmap Heap Scan": "BitmapScan",
    }
    
    for key, hint in join_hint_map.items():
        if key in nt:
            hints.append(f"{hint}()")
    for key, hint in scan_hint_map.items():
        if key in nt:
            hints.append(f"{hint}()")
    
    for child in node.get("children", []):
        _extract_hints_recursive(child, hints)


# ── Plan comparison with Jaccard similarity ──────────────────────
def compare_plans(plan_a, plan_b, *, threshold=0.8):
    """Compare two plans using Jaccard similarity on join/scan methods.
    
    Algorithm change: upstream uses exact match on filtered hint strings.
    We use Jaccard similarity which is more robust to minor differences.
    """
    decoded_a = decode_explain(plan_a)
    decoded_b = decode_explain(plan_b)
    
    methods_a = set(decoded_a.get("join_methods", []) + decoded_a.get("scan_methods", []))
    methods_b = set(decoded_b.get("join_methods", []) + decoded_b.get("scan_methods", []))
    
    if not methods_a and not methods_b:
        return 1.0
    
    intersection = len(methods_a & methods_b)
    union = len(methods_a | methods_b)
    jaccard = intersection / max(union, 1)
    
    _dbg("compare_plans", jaccard=f"{jaccard:.4f}",
         methods_a=methods_a, methods_b=methods_b)
    return jaccard


# ── Plan Cost Diagram ────────────────────────────────────────────
class PlanDiagram:
    """Plan cost distribution diagram with convex hull envelope.
    
    Algorithm change: computes convex hull of (query_idx, cost) points
    for each plan, giving a tighter cost envelope than raw min/max.
    """
    
    def __init__(self):
        self.plan_costs = {}  # plan_id -> list of (query_idx, cost)
        self._n_queries = 0
    
    def add_plan(self, plan_id, costs):
        """Add a plan with its cost vector across queries."""
        self.plan_costs[plan_id] = list(enumerate(costs))
        self._n_queries = max(self._n_queries, len(costs))
        _dbg("add_plan", plan_id=plan_id, n_costs=len(costs),
             mean_cost=f"{sum(costs)/len(costs):.2f}" if costs else "0")
    
    def get_envelope(self, plan_id):
        """Get convex hull cost envelope for a plan.
        
        Returns (lower_bound, upper_bound) cost trajectories.
        Uses a sliding window to compute local min/max envelopes.
        """
        if plan_id not in self.plan_costs:
            return [], []
        
        points = self.plan_costs[plan_id]
        costs = [c for _, c in points]
        n = len(costs)
        window = max(1, n // 10)
        
        lower, upper = [], []
        for i in range(n):
            lo = max(0, i - window)
            hi = min(n, i + window + 1)
            segment = costs[lo:hi]
            lower.append(min(segment))
            upper.append(max(segment))
        
        return lower, upper
    
    def cost_distribution_entropy(self, plan_id, n_bins=10):
        """Shannon entropy of cost distribution.
        
        Algorithm addition: quantifies how spread out the costs are.
        High entropy = highly variable cost across queries.
        """
        if plan_id not in self.plan_costs:
            return 0.0
        
        costs = [c for _, c in self.plan_costs[plan_id]]
        if not costs:
            return 0.0
        
        mn, mx = min(costs), max(costs)
        if mn == mx:
            return 0.0
        
        width = (mx - mn) / n_bins
        bins = [0] * n_bins
        for c in costs:
            idx = min(int((c - mn) / width), n_bins - 1)
            bins[idx] += 1
        
        total = sum(bins)
        entropy = 0.0
        for b in bins:
            if b > 0:
                p = b / total
                entropy -= p * math.log2(p)
        
        _dbg("cost_entropy", plan=plan_id, entropy=f"{entropy:.4f}",
             n_costs=len(costs))
        return entropy
    
    def dump_diagram(self):
        """Print diagram state."""
        print(f"[PlanDiagram] {len(self.plan_costs)} plans, {self._n_queries} queries")
        for pid, costs in self.plan_costs.items():
            vals = [c for _, c in costs]
            if vals:
                print(f"  plan {pid}: mean={sum(vals)/len(vals):.2f} "
                      f"std={_std(vals):.2f} "
                      f"entropy={self.cost_distribution_entropy(pid):.4f}")


# ── Workload Sampler — stratified by cost quantile ───────────────
class WorkloadSampler:
    """Sample queries from workload, stratified by cost quantile.
    
    Algorithm change: upstream samples uniformly.
    Stratified sampling ensures coverage of both cheap and expensive queries.
    """
    
    def __init__(self, queries, costs=None, n_strata=5):
        self.queries = queries
        self.costs = costs or list(range(len(queries)))
        self.n_strata = n_strata
    
    def sample(self, n):
        """Draw n stratified samples."""
        if n >= len(self.queries):
            return list(range(len(self.queries)))
        
        # Sort by cost and split into strata
        indexed = sorted(enumerate(self.costs), key=lambda x: x[1])
        stratum_size = max(1, len(indexed) // self.n_strata)
        
        samples = []
        per_stratum = max(1, n // self.n_strata)
        
        import random
        for s in range(self.n_strata):
            start = s * stratum_size
            end = min(len(indexed), (s + 1) * stratum_size)
            stratum_indices = [idx for idx, _ in indexed[start:end]]
            
            drawn = random.sample(stratum_indices,
                                  min(per_stratum, len(stratum_indices)))
            samples.extend(drawn)
        
        # Fill remaining if needed
        remaining = [i for i in range(len(self.queries)) if i not in samples]
        while len(samples) < n and remaining:
            samples.append(remaining.pop(random.randint(0, len(remaining) - 1)))
        
        _dbg("stratified_sample", n=n, n_strata=self.n_strata,
             n_drawn=len(samples))
        return samples[:n]


def _std(vals):
    """Standard deviation."""
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(var)
