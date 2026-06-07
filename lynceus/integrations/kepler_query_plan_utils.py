"""
Kepler Query Plan Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pure-numpy helpers for manipulating, fingerprinting, and comparing
query execution plans.  Every public function carries a *_dbg()* trace.
"""

import hashlib
import json
import sys
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------
_DEBUG = False


def enable_debug(flag: bool = True) -> None:
    global _DEBUG
    _DEBUG = flag


def _dbg(fn_name: str, msg: str, **kw: Any) -> None:
    if not _DEBUG:
        return
    extras = " ".join(f"{k}={v!r}" for k, v in kw.items())
    ts = time.perf_counter()
    print(f"[DBG {ts:.6f}] {fn_name}: {msg} {extras}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Hint extraction
# ---------------------------------------------------------------------------
_KNOWN_HINT_KEYS = frozenset({
    "Leading", "HashJoin", "MergeJoin", "NestLoop",
    "SeqScan", "IndexScan", "IndexOnlyScan", "BitmapScan",
    "Parallel", "Materialize", "Sort", "Aggregate",
    "Set", "Rows", "Width", "Cost",
})


def extract_plan_hints(plan_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Walk a plan dict and pull out recognised optimiser hints.

    Returns a flat dictionary of ``{hint_key: value}`` pairs found at
    any nesting depth.
    """
    _dbg("extract_plan_hints", "enter", keys=list(plan_dict.keys())[:8])

    hints: Dict[str, Any] = {}

    def _walk(node: Any, depth: int = 0) -> None:
        _dbg("extract_plan_hints._walk", f"depth={depth}", type=type(node).__name__)
        if isinstance(node, dict):
            for k, v in node.items():
                if k in _KNOWN_HINT_KEYS:
                    hints[k] = v
                _walk(v, depth + 1)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _walk(item, depth + 1)

    _walk(plan_dict)
    _dbg("extract_plan_hints", "done", hint_count=len(hints))
    return hints


# ---------------------------------------------------------------------------
# Plan tree normalisation
# ---------------------------------------------------------------------------
def normalize_plan_tree(node: Dict[str, Any]) -> Dict[str, Any]:
    """Return a *canonical* copy of the plan tree.

    Normalisation rules:
      1. Keys are sorted alphabetically.
      2. ``"Actual â¦"`` timing/row keys are stripped (execution-dependent).
      3. Child ``Plans`` lists are recursively normalised and then sorted by
         their ``Node Type`` so that logically equivalent trees with
         differently-ordered children compare equal.
      4. Floating-point costs are rounded to 2 decimal places.
    """
    _dbg("normalize_plan_tree", "enter", node_type=node.get("Node Type"))

    skip_prefixes = ("Actual ", "Execution ", "Triggers", "Planning Time",
                     "Execution Time", "Peak Memory")

    out: Dict[str, Any] = {}

    for key in sorted(node.keys()):
        if any(key.startswith(p) for p in skip_prefixes):
            _dbg("normalize_plan_tree", f"skip key={key}")
            continue

        val = node[key]

        if key == "Plans" and isinstance(val, list):
            children = [normalize_plan_tree(c) for c in val]
            children.sort(key=lambda c: c.get("Node Type", ""))
            out[key] = children
        elif isinstance(val, float):
            out[key] = round(val, 2)
        elif isinstance(val, dict):
            out[key] = normalize_plan_tree(val)
        else:
            out[key] = val

    _dbg("normalize_plan_tree", "done", keys=len(out))
    return out


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------
def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def plan_to_fingerprint(tree: Dict[str, Any]) -> str:
    """SHA-256 fingerprint of a normalised plan tree.

    The tree is first normalised, then serialised to canonical JSON and
    hashed.  Identical logical plans yield the same fingerprint regardless
    of cosmetic differences.
    """
    _dbg("plan_to_fingerprint", "enter")

    normed = normalize_plan_tree(tree)
    payload = _canonical_json(normed).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()

    _dbg("plan_to_fingerprint", "done", digest=digest[:16])
    return digest


# ---------------------------------------------------------------------------
# Plan comparison  (Jaccard similarity over node-type multisets)
# ---------------------------------------------------------------------------
def _collect_node_types(node: Dict[str, Any]) -> List[str]:
    """DFS collection of all Node Type values."""
    types: List[str] = []

    def _dfs(n: Dict[str, Any]) -> None:
        nt = n.get("Node Type")
        if nt:
            types.append(nt)
        for child in n.get("Plans", []):
            _dfs(child)

    _dfs(node)
    return types


def _multiset_jaccard(a: List[str], b: List[str]) -> float:
    """Jaccard similarity on multisets (bags) of strings.

    J(A, B) = |A â© B| / |A âª B|  where intersection and union use
    *min* and *max* counts respectively.
    """
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    all_keys = set(ca) | set(cb)
    if not all_keys:
        return 1.0
    inter = sum(min(ca[k], cb[k]) for k in all_keys)
    union = sum(max(ca[k], cb[k]) for k in all_keys)
    return inter / union if union else 1.0


def compare_plans(
    a: Dict[str, Any],
    b: Dict[str, Any],
) -> float:
    """Return Jaccard similarity â [0, 1] between two plan trees.

    Comparison is based on the multiset of ``Node Type`` values after
    normalisation so that ordering and runtime stats do not matter.
    """
    _dbg("compare_plans", "enter")

    na = normalize_plan_tree(a)
    nb = normalize_plan_tree(b)

    ta = _collect_node_types(na)
    tb = _collect_node_types(nb)

    sim = _multiset_jaccard(ta, tb)
    _dbg("compare_plans", "done", similarity=sim)
    return sim


# ---------------------------------------------------------------------------
# PlanCandidate dataclass
# ---------------------------------------------------------------------------
@dataclass
class PlanCandidate:
    """Lightweight container for a candidate execution plan."""

    plan_tree: Dict[str, Any]
    fingerprint: str = ""
    estimated_cost: float = 0.0
    hints: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    score: float = 0.0

    def __post_init__(self) -> None:
        _dbg("PlanCandidate.__post_init__", "create")
        if not self.fingerprint:
            self.fingerprint = plan_to_fingerprint(self.plan_tree)
        if not self.hints:
            self.hints = extract_plan_hints(self.plan_tree)
        if self.estimated_cost == 0.0:
            self.estimated_cost = estimate_plan_cost(self.plan_tree)

    def similarity_to(self, other: "PlanCandidate") -> float:
        _dbg("PlanCandidate.similarity_to", "compare")
        return compare_plans(self.plan_tree, other.plan_tree)

    def __repr__(self) -> str:
        return (
            f"PlanCandidate(fp={self.fingerprint[:12]}â¦, "
            f"cost={self.estimated_cost:.2f}, "
            f"tags={self.tags})"
        )


# ---------------------------------------------------------------------------
# Cost estimation (bottom-up recursive)
# ---------------------------------------------------------------------------
# Default per-operator unit costs (numpy array for vectorised lookup)
_OP_COST_TABLE: Dict[str, float] = {
    "Seq Scan": 1.0,
    "Index Scan": 0.5,
    "Index Only Scan": 0.4,
    "Bitmap Heap Scan": 0.7,
    "Bitmap Index Scan": 0.3,
    "Nested Loop": 2.0,
    "Hash Join": 1.5,
    "Merge Join": 1.3,
    "Sort": 1.8,
    "Aggregate": 1.2,
    "Hash": 0.8,
    "Materialize": 0.6,
    "Gather": 1.0,
    "Gather Merge": 1.1,
    "Append": 0.4,
    "Subquery Scan": 0.5,
    "Limit": 0.1,
    "Result": 0.05,
}


def estimate_plan_cost(
    tree: Dict[str, Any],
    op_costs: Optional[Dict[str, float]] = None,
) -> float:
    """Bottom-up recursive cost estimation.

    For each node the cost equals::

        local_cost  =  op_weight Ã estimated_rows
        total_cost  =  local_cost + Î£ child_costs

    ``op_weight`` comes from *op_costs* (or the built-in table).
    ``estimated_rows`` is taken from ``Plan Rows`` or defaults to 1000.

    20 % algorithm tweak vs. a naÃ¯ve sum: a *discount factor* of 0.85 is
    applied when a node has â¥ 2 children, modelling the fact that parallel
    subtrees share I/O.
    """
    _dbg("estimate_plan_cost", "enter")
    costs = op_costs if op_costs is not None else _OP_COST_TABLE
    _parallel_discount = 0.85

    def _recurse(node: Dict[str, Any], depth: int = 0) -> float:
        node_type = node.get("Node Type", "Unknown")
        rows = float(node.get("Plan Rows", node.get("Actual Rows", 1000)))
        width = float(node.get("Plan Width", 1))

        unit = costs.get(node_type, 1.0)
        local = unit * rows * (1.0 + np.log1p(width) * 0.01)

        _dbg("estimate_plan_cost._recurse",
             f"depth={depth}", node_type=node_type, local=round(local, 2))

        children = node.get("Plans", [])
        if not children:
            return local

        child_costs_arr = np.array(
            [_recurse(c, depth + 1) for c in children], dtype=np.float64
        )

        if len(child_costs_arr) >= 2:
            child_total = float(np.sum(child_costs_arr)) * _parallel_discount
        else:
            child_total = float(np.sum(child_costs_arr))

        return local + child_total

    total = _recurse(tree)
    _dbg("estimate_plan_cost", "done", total=round(total, 2))
    return total


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    enable_debug(True)

    sample_plan: Dict[str, Any] = {
        "Node Type": "Hash Join",
        "Plan Rows": 500,
        "Plan Width": 40,
        "Hash Cond": "(a.id = b.id)",
        "Plans": [
            {
                "Node Type": "Seq Scan",
                "Plan Rows": 10000,
                "Plan Width": 20,
                "Relation Name": "table_a",
            },
            {
                "Node Type": "Hash",
                "Plan Rows": 200,
                "Plan Width": 20,
                "Plans": [
                    {
                        "Node Type": "Index Scan",
                        "Plan Rows": 200,
                        "Plan Width": 20,
                        "Index Name": "idx_b_id",
                        "Relation Name": "table_b",
                    }
                ],
            },
        ],
    }

    hints = extract_plan_hints(sample_plan)
    print("hints:", hints)

    normed = normalize_plan_tree(sample_plan)
    print("normalised keys:", list(normed.keys()))

    fp = plan_to_fingerprint(sample_plan)
    print("fingerprint:", fp)

    sim = compare_plans(sample_plan, sample_plan)
    print("self-similarity:", sim)

    cost = estimate_plan_cost(sample_plan)
    print("estimated cost:", round(cost, 2))

    candidate = PlanCandidate(plan_tree=sample_plan, tags=["baseline"])
    print(candidate)
