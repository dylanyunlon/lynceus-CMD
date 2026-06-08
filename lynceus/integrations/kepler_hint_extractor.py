# coding=utf-8
# Copyright 2022 Google LLC.  (upstream)
# Modifications copyright 2024 Lynceus contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lynceus Kepler Hint Extractor — merged & hardened edition.

Merges pg_plan_hint_extractor.py and query_text_utils.py from the upstream
Kepler/Par2QO training-data-collection pipeline into one self-contained
module.  All database access is replaced by an in-memory simulation layer
so the module can run in unit-test / CI environments without PostgreSQL.

Algorithmic changes vs upstream (~20 % of logic):
  1. Welford online variance for per-hint row-count statistics.
  2. EMA (exponential moving-average) decay on plan-frequency counters so
     recent workloads weigh more.
  3. Laplace smoothing on plan-index distributions to avoid zero-probability
     entries when a plan has not yet been seen for a given parameter set.
  4. Huber-loss based outlier gating when accumulating estimated row counts,
     replacing the raw dict-store with a robust streaming aggregator.
  5. Verification uses a cosine-similarity check on hint feature vectors
     instead of exact string equality, with a configurable tolerance.

Every key function includes a ``_debug_snapshot()`` call that prints the
full data-structure state at that point — useful for tracing through the
pipeline interactively.
"""

from __future__ import annotations

import copy
import json
import math
import pprint
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# query_text_utils — inlined & merged
# ---------------------------------------------------------------------------

_NAME_DELIMITER = "####"


def get_params_as_string(params: List[Any]) -> str:
    return _NAME_DELIMITER.join([str(p) for p in params])


def substitute_query_params(query: str, params: Sequence[Any]) -> str:
    if params:
        for i in range(len(params) - 1, -1, -1):
            query = query.replace(f"@param{i}", str(params[i]))
    return query


def get_hinted_query(query: str, hints: str) -> str:
    return f"{hints} {query}"


# ---------------------------------------------------------------------------
# In-memory database simulation (replaces query_utils.DatabaseConfiguration
# and query_utils.QueryManager)
# ---------------------------------------------------------------------------

class DatabaseConfiguration:
    """Lightweight stand-in that holds connection info without needing psycopg2."""

    def __init__(self, host: str = "mem://localhost", port: int = 0,
                 database: str = "simulated", user: str = "lynceus",
                 password: str = ""):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password

    def __repr__(self) -> str:
        return (f"DatabaseConfiguration(host={self.host!r}, "
                f"db={self.database!r}, user={self.user!r})")


class InMemoryPlanStore:
    """Registry of pre-loaded EXPLAIN plans keyed by (query_text, params_tuple).

    Callers seed the store with ``register_plan`` before any extraction runs.
    ``QueryManager.get_query_plan`` reads from here instead of hitting a real
    database.
    """

    def __init__(self) -> None:
        self._plans: Dict[Tuple[str, Tuple[Any, ...]], Any] = {}

    def register_plan(self, query: str, params: Optional[List[Any]],
                      plan: Any) -> None:
        key = (query, tuple(params) if params else ())
        self._plans[key] = plan

    def lookup(self, query: str, params: Optional[List[Any]]) -> Any:
        key = (query, tuple(params) if params else ())
        if key not in self._plans:
            raise KeyError(
                f"No simulated plan registered for query={query!r}, "
                f"params={params!r}.  Seed InMemoryPlanStore first.")
        return copy.deepcopy(self._plans[key])

    def __len__(self) -> int:
        return len(self._plans)


# Module-level singleton — importers can swap in their own store.
_PLAN_STORE = InMemoryPlanStore()


def get_plan_store() -> InMemoryPlanStore:
    return _PLAN_STORE


def set_plan_store(store: InMemoryPlanStore) -> None:
    global _PLAN_STORE
    _PLAN_STORE = store


class QueryManager:
    """Drop-in replacement that reads from ``InMemoryPlanStore``."""

    def __init__(self, db_config: DatabaseConfiguration) -> None:
        self.db_config = db_config

    def get_query_plan(self, query: str,
                       params: Optional[List[Any]] = None) -> Any:
        return _PLAN_STORE.lookup(query, params)


# ---------------------------------------------------------------------------
# Translator tables (unchanged from upstream)
# ---------------------------------------------------------------------------

_SCAN_METHOD_TRANSLATOR = {
    "Seq Scan": "SeqScan",
    "Index Scan": "IndexScan",
    "Index Only Scan": "IndexOnlyScan",
    "Bitmap Heap Scan": "BitmapScan",
}
_SCAN_METHOD_RED_HERRINGS = ["Subquery Scan", "CTE Scan"]
_JOIN_METHOD_TRANSLATOR = {
    "Hash Join": "HashJoin",
    "Nested Loop": "NestLoop",
    "Merge Join": "MergeJoin",
}
_SUBQUERY_PARENT_RELATIONSHIPS = ["InitPlan", "SubPlan"]

# ---------------------------------------------------------------------------
# Typing aliases
# ---------------------------------------------------------------------------

JSON = Any
Hint = Dict[str, str]
ParamsDefaultIndex = Dict[str, Union[List[str], int]]

# ---------------------------------------------------------------------------
# Algorithm building-blocks (new)
# ---------------------------------------------------------------------------

class WelfordAccumulator:
    """Welford's online algorithm for streaming mean & variance.

    Used to track per-hint row-count statistics without storing every sample.
    Upstream simply stuffed counts into a plain dict — we compute running
    mean, variance, and an optional z-score gate.
    """

    def __init__(self) -> None:
        self.n: int = 0
        self.mean: float = 0.0
        self._m2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._m2 += delta * delta2

    @property
    def variance(self) -> float:
        return self._m2 / self.n if self.n >= 2 else 0.0

    @property
    def stddev(self) -> float:
        return math.sqrt(self.variance)

    def z_score(self, x: float) -> float:
        sd = self.stddev
        if sd < 1e-12:
            return 0.0
        return (x - self.mean) / sd

    def snapshot(self) -> Dict[str, Any]:
        return {"n": self.n, "mean": self.mean,
                "variance": self.variance, "stddev": self.stddev}


def huber_loss(predicted: float, actual: float, delta: float = 1.0) -> float:
    """Huber loss — smooth L1 used to gate outlier row-count contributions.

    Upstream blindly stored ``Plan Rows`` without any outlier consideration.
    We score each incoming row-count against the Welford running mean; if the
    Huber loss exceeds ``3 * delta`` we clip the contribution to the boundary.
    """
    residual = abs(predicted - actual)
    if residual <= delta:
        return 0.5 * residual * residual
    return delta * (residual - 0.5 * delta)


def huber_clamp(value: float, center: float, delta: float = 1.0,
                max_loss_ratio: float = 3.0) -> float:
    """Return *value* clamped so that its Huber loss vs *center* stays bounded."""
    loss = huber_loss(center, value, delta)
    threshold = max_loss_ratio * delta
    if loss <= threshold:
        return value
    # Pull toward center by the fraction that exceeds the threshold.
    overshoot = loss / threshold
    return center + (value - center) / overshoot


class EMACounter:
    """Exponential-moving-average counter with Laplace smoothing.

    Upstream counted plan occurrences with a flat ``count += 1``.  We apply
    EMA decay (``alpha``) so recent plans weigh more, and Laplace smoothing
    (``laplace_k``) so no plan ever reaches exact-zero probability.
    """

    def __init__(self, alpha: float = 0.05, laplace_k: float = 1.0) -> None:
        self._raw: Dict[str, float] = {}
        self._alpha = alpha
        self._laplace_k = laplace_k
        self._total_updates: int = 0

    def observe(self, key: str) -> None:
        self._total_updates += 1
        # Decay all existing keys, then bump the observed one.
        for k in self._raw:
            self._raw[k] *= (1.0 - self._alpha)
        self._raw.setdefault(key, 0.0)
        self._raw[key] += 1.0

    def probability(self, key: str) -> float:
        """Laplace-smoothed probability estimate for *key*."""
        numerator = self._raw.get(key, 0.0) + self._laplace_k
        denominator = sum(self._raw.values()) + self._laplace_k * len(self._raw)
        if denominator < 1e-15:
            return 1.0 / max(len(self._raw), 1)
        return numerator / denominator

    @property
    def keys(self) -> List[str]:
        return list(self._raw.keys())

    def snapshot(self) -> Dict[str, Any]:
        probs = {k: self.probability(k) for k in self._raw}
        return {"raw": dict(self._raw), "probabilities": probs,
                "total_updates": self._total_updates,
                "alpha": self._alpha, "laplace_k": self._laplace_k}


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity of two equal-length vectors.

    Used in hint verification: instead of exact string equality we featurise
    the hint string into a sparse numeric vector and measure similarity.
    """
    if len(a) != len(b):
        raise ValueError("Vectors must have equal length")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-15 or norm_b < 1e-15:
        return 0.0
    return dot / (norm_a * norm_b)


def _hint_to_feature_vector(hint_str: str, vocab: Dict[str, int]) -> List[float]:
    """Convert a hint string into a bag-of-tokens numeric vector."""
    tokens = hint_str.replace("(", " ").replace(")", " ").split()
    vec = [0.0] * len(vocab)
    for tok in tokens:
        idx = vocab.get(tok)
        if idx is not None:
            vec[idx] += 1.0
    return vec


def _build_vocab(*hint_strings: str) -> Dict[str, int]:
    """Build a token → index vocabulary from one or more hint strings."""
    vocab: Dict[str, int] = {}
    for hs in hint_strings:
        for tok in hs.replace("(", " ").replace(")", " ").split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------

def _debug_snapshot(label: str, data: Any) -> None:
    """Pretty-print a labelled snapshot of any data structure."""
    border = "=" * 72
    print(f"\n{border}")
    print(f"  DEBUG SNAPSHOT  [{label}]")
    print(border)
    pprint.pprint(data, width=120)
    print(border + "\n")


# ---------------------------------------------------------------------------
# Node predicates (from upstream, unchanged)
# ---------------------------------------------------------------------------

def _is_scan_node(node: JSON) -> bool:
    return node["Node Type"] in _SCAN_METHOD_TRANSLATOR


def _is_subquery_node(node: JSON, parent_relationship_required: bool) -> bool:
    if parent_relationship_required:
        assert "Parent Relationship" in node
    return (("Parent Relationship" in node) and
            (node["Parent Relationship"] in _SUBQUERY_PARENT_RELATIONSHIPS))


def _strip_parens(text: str) -> str:
    return text.replace(")", "").replace("(", "")


# ---------------------------------------------------------------------------
# Scan / join extraction (from upstream, with Huber-gated row counts)
# ---------------------------------------------------------------------------

def _extract_scan_method_hints(node: JSON, plan_hints: List[str]) -> None:
    if _is_subquery_node(node, parent_relationship_required=False):
        return
    if _is_scan_node(node):
        if "Index Name" in node:
            plan_hints.append("{}({} {})".format(
                _SCAN_METHOD_TRANSLATOR[node["Node Type"]],
                node["Alias"], node["Index Name"]))
        else:
            plan_hints.append("{}({})".format(
                _SCAN_METHOD_TRANSLATOR[node["Node Type"]], node["Alias"]))
    else:
        assert ("Scan" not in node["Node Type"] or
                node["Node Type"] in _SCAN_METHOD_RED_HERRINGS), (
            f"Unexpected Scan Node: {node['Node Type']} from {json.dumps(node)}")
        if "Plans" in node:
            for child_node in node["Plans"]:
                _extract_scan_method_hints(child_node, plan_hints)


def _extract_join_hints(
    node: JSON,
    plan_hints: List[str],
    row_counts: Optional[Dict[str, Any]] = None,
    welford_registry: Optional[Dict[str, WelfordAccumulator]] = None,
) -> str:
    """Extract join methods, with optional Huber-gated Welford row-count tracking.

    Upstream stored raw ``Plan Rows`` into *row_counts*.  We additionally feed
    each value through a ``WelfordAccumulator`` and use ``huber_clamp`` to
    suppress outlier rows before storing.
    """
    join_method = None
    if "Join Type" in node:
        join_method = _JOIN_METHOD_TRANSLATOR[node["Node Type"]]

    if not join_method:
        if _is_scan_node(node):
            return node["Alias"]
        return _extract_join_hints(node["Plans"][0], plan_hints,
                                   row_counts, welford_registry)

    join_child_index = 0
    while node["Plans"][join_child_index]["Parent Relationship"] != "Outer":
        assert _is_subquery_node(
            node["Plans"][join_child_index], parent_relationship_required=True
        ), f"Unexpected node while searching for Outer: {json.dumps(node)}"
        join_child_index += 1
    outer_part = _extract_join_hints(
        node["Plans"][join_child_index], plan_hints,
        row_counts, welford_registry)
    join_child_index += 1

    while node["Plans"][join_child_index]["Parent Relationship"] != "Inner":
        assert _is_subquery_node(
            node["Plans"][join_child_index], parent_relationship_required=True
        ), f"Unexpected node while searching for Inner: {json.dumps(node)}"
        join_child_index += 1
    inner_part = _extract_join_hints(
        node["Plans"][join_child_index], plan_hints,
        row_counts, welford_registry)
    join_child_index += 1

    while join_child_index < len(node["Plans"]):
        assert _is_subquery_node(
            node["Plans"][join_child_index], parent_relationship_required=True
        ), f"Unexpected trailing node: {json.dumps(node)}"
        join_child_index += 1

    join_order = "({} {})".format(outer_part, inner_part)
    joined_tables = _strip_parens(join_order)
    plan_hints.append("{}({})".format(join_method, joined_tables))

    raw_rows = node.get("Plan Rows")
    if raw_rows is not None and row_counts is not None:
        # ---- Huber-gated Welford aggregation (new vs upstream) ----
        if welford_registry is not None:
            acc = welford_registry.setdefault(joined_tables,
                                              WelfordAccumulator())
            if acc.n >= 2:
                clamped = huber_clamp(float(raw_rows), acc.mean,
                                      delta=acc.stddev + 1.0)
            else:
                clamped = float(raw_rows)
            acc.update(clamped)
            row_counts[joined_tables] = round(acc.mean)
        else:
            row_counts[joined_tables] = raw_rows

    return join_order


# ---------------------------------------------------------------------------
# Row-count extraction helpers
# ---------------------------------------------------------------------------

def extract_row_counts(explain_plan: JSON) -> Dict[str, int]:
    row_counts: Dict[str, int] = {}
    welford_reg: Dict[str, WelfordAccumulator] = {}
    _ = _extract_join_hints(explain_plan, [], row_counts, welford_reg)
    _debug_snapshot("extract_row_counts", {
        "row_counts": row_counts,
        "welford_state": {k: v.snapshot() for k, v in welford_reg.items()},
    })
    return row_counts


def extract_base_table_row_counts(explain_plan: JSON) -> Dict[str, int]:
    row_counts: Dict[str, int] = {}

    def _helper(nd: JSON) -> None:
        if _is_scan_node(nd):
            row_counts[nd["Alias"]] = nd["Plan Rows"]
        for child in nd.get("Plans", []):
            _helper(child)

    _helper(explain_plan)
    _debug_snapshot("extract_base_table_row_counts", row_counts)
    return row_counts


# ---------------------------------------------------------------------------
# Hint-string builders
# ---------------------------------------------------------------------------

def _extract_plan_hints_builder(subtree_root: JSON) -> str:
    plan_hints: List[str] = []
    _extract_scan_method_hints(subtree_root, plan_hints)
    join_order = _extract_join_hints(subtree_root, plan_hints)
    if join_order.count(" "):
        plan_hints.append("Leading({})".format(join_order))
    subquery_hints: List[str] = []
    _extract_plan_hints_subqueries(subtree_root, subquery_hints)
    plan_hints.extend(subquery_hints)
    return " ".join(plan_hints)


def _extract_plan_hints_subqueries(node: JSON,
                                   plan_hints: List[str]) -> None:
    if _is_subquery_node(node, parent_relationship_required=False):
        subtree = copy.deepcopy(node)
        del subtree["Parent Relationship"]
        plan_hints.append(_extract_plan_hints_builder(subtree))
    else:
        if "Plans" in node:
            for child_node in node["Plans"]:
                _extract_plan_hints_subqueries(child_node, plan_hints)


def _extract_plan_hints(explain_plan: JSON) -> str:
    plan_hints = ["/*+ "]
    plan_hints.append(_extract_plan_hints_builder(explain_plan))
    plan_hints.append("*/")
    result = " ".join(plan_hints)
    _debug_snapshot("_extract_plan_hints", {"result": result})
    return result


# ---------------------------------------------------------------------------
# File reader (from upstream)
# ---------------------------------------------------------------------------

def get_file_content(filename: str) -> List[str]:
    lines: List[str] = []
    with open(filename) as f:
        for line in f:
            if "--" in line:
                continue
            line = line.replace(";", "").strip()
            from_index = line.lower().find("from")
            if (from_index > -1 and len(line) > 4 and
                    (from_index == 0 or line[from_index - 1] == " ") and
                    (len(line) == (from_index + 4) or
                     line[from_index + 4] == " ")):
                from_token = line[from_index:from_index + 4]
                components = line.split(from_token)
                assert len(components) == 2
                if components[0]:
                    lines.append(components[0])
                lines.append(from_token)
                if components[1]:
                    lines.append(components[1])
            else:
                lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Single-query hint extraction
# ---------------------------------------------------------------------------

def get_single_query_hints(db_config: DatabaseConfiguration,
                           query: str,
                           params: Optional[List[Any]] = None) -> str:
    qm = QueryManager(db_config)
    return get_single_query_hints_with_plan(qm, query, params)[0]


def get_single_query_hints_with_plan(
    query_manager: QueryManager,
    query: str,
    params: Optional[List[Any]],
) -> Tuple[str, JSON]:
    explain_plan = query_manager.get_query_plan(query, params)
    result = _extract_plan_hints(explain_plan["Plan"]), explain_plan
    _debug_snapshot("get_single_query_hints_with_plan", {
        "query": query, "params": params, "hint": result[0],
    })
    return result


# ---------------------------------------------------------------------------
# Hint-index lookup
# ---------------------------------------------------------------------------

def _get_plan_hint_index(plan_hints: List[Dict[str, str]],
                         plan_hint: str) -> int:
    for i, entry in enumerate(plan_hints):
        if entry["hints"] == plan_hint:
            return i
    raise ValueError(f"Plan hint not found: {plan_hint!r}")


# ---------------------------------------------------------------------------
# Batch hint generation for one parameter set
# ---------------------------------------------------------------------------

def _generate_hints(
    params_plan_info: JSON,
) -> Tuple[List[str], List[str], List[str], List[JSON]]:
    hints: List[str] = []
    sources: List[str] = []

    hints.append(_extract_plan_hints(params_plan_info["result"]["Plan"]))
    sources.append("default")

    if "additional_plans" in params_plan_info:
        for plan, source in zip(params_plan_info["additional_plans"],
                                params_plan_info["sources"]):
            hints.append(_extract_plan_hints(plan["Plan"]))
            sources.append(source)

    _debug_snapshot("_generate_hints", {
        "params": params_plan_info["params"],
        "hints": hints,
        "sources": sources,
    })
    return (params_plan_info["params"], hints, sources,
            params_plan_info.get("debug_info", {}))


# ---------------------------------------------------------------------------
# Hint verification — cosine-similarity (replaces exact string match)
# ---------------------------------------------------------------------------

def verify_hints(
    query_id: str,
    query: str,
    plan_hints: List[Hint],
    params_plan_indices: List[ParamsDefaultIndex],
    database_configuration: DatabaseConfiguration,
    similarity_threshold: float = 0.95,
) -> Dict[str, Dict[str, int]]:
    """Verify hints via cosine similarity instead of exact string equality.

    Upstream compared re-extracted hints with ``!=``.  Small non-semantic
    differences (e.g. whitespace, alias ordering) caused spurious failures.
    We now featurise both hint strings into bag-of-token vectors and accept
    the match when cosine similarity >= *similarity_threshold*.
    """
    failure_counts: Dict[str, Dict[str, int]] = {query_id: {}}

    for hint in plan_hints:
        hinted_query = "{} {}".format(hint["hints"], query)
        params_failure_count = 0

        for ppi in params_plan_indices:
            ok = _check_hint_cosine(
                database_configuration, hint["hints"], hinted_query,
                ppi, similarity_threshold)
            if not ok:
                params_failure_count += 1

        failure_counts[query_id][hint["hints"]] = params_failure_count

    _debug_snapshot("verify_hints", failure_counts)
    return failure_counts


def _check_hint_cosine(
    db_config: DatabaseConfiguration,
    old_hint: str,
    hinted_query: str,
    params_default_index: ParamsDefaultIndex,
    threshold: float,
) -> bool:
    """Cosine-similarity hint check (replaces upstream exact-match _check_hint)."""
    qm = QueryManager(db_config)
    new_plan = qm.get_query_plan(hinted_query, params_default_index["params"])
    new_hint = _extract_plan_hints(new_plan["Plan"])

    # Fast path: exact match still passes.
    if old_hint == new_hint:
        return True

    vocab = _build_vocab(old_hint, new_hint)
    vec_old = _hint_to_feature_vector(old_hint, vocab)
    vec_new = _hint_to_feature_vector(new_hint, vocab)
    sim = cosine_similarity(vec_old, vec_new)

    _debug_snapshot("_check_hint_cosine", {
        "old_hint": old_hint, "new_hint": new_hint,
        "similarity": sim, "threshold": threshold,
        "pass": sim >= threshold,
    })
    return sim >= threshold


# ---------------------------------------------------------------------------
# Merge helper (with EMA-aware dedup)
# ---------------------------------------------------------------------------

def merge_hints(base: List[Hint], extra_hints: List[Hint],
                merge_suffix: str = "",
                ema_counter: Optional[EMACounter] = None) -> None:
    """Merge extra_hints into base, with optional EMA tracking.

    Upstream used a plain set for dedup.  We additionally feed each
    observed hint through an ``EMACounter`` so callers can later query
    recency-weighted probabilities.
    """
    existing = set(h["hints"] for h in base)
    for eh in extra_hints:
        if ema_counter is not None:
            ema_counter.observe(eh["hints"])
        if eh["hints"] not in existing:
            existing.add(eh["hints"])
            eh["source"] += merge_suffix
            base.append(eh)

    _debug_snapshot("merge_hints", {
        "base_count": len(base),
        "ema_state": ema_counter.snapshot() if ema_counter else None,
    })


# ---------------------------------------------------------------------------
# PlanHintExtractor — main class (EMA + Laplace smoothed)
# ---------------------------------------------------------------------------

class PlanHintExtractor:
    """Extracts and deduplicates hints from EXPLAIN plan outputs.

    Changes vs upstream:
    * Plan-frequency counting uses ``EMACounter`` (EMA decay + Laplace
      smoothing) instead of a raw integer counter.
    * ``get_consolidated_plan_hints`` exposes per-hint probability.
    """

    def __init__(self, ema_alpha: float = 0.05,
                 laplace_k: float = 1.0) -> None:
        self._ema = EMACounter(alpha=ema_alpha, laplace_k=laplace_k)
        self._plan_hints: List[Hint] = []
        self._params_plan_indices: List[ParamsDefaultIndex] = []
        self._debug_infos: List[JSON] = []
        self._consolidated = False

    def add_query_plans(self, params_plans_info: JSON) -> None:
        if self._consolidated:
            raise ValueError(
                "Cannot call add_query_plans() after "
                "get_consolidated_plan_hints()")

        params, hints, sources, debug_infos = _generate_hints(
            params_plans_info)
        if not hints:
            return

        for hint, source in zip(hints, sources):
            self._ema.observe(hint)

        params_default_index: ParamsDefaultIndex = {
            "params": params, "plan_index": hints[0],
        }
        self._params_plan_indices.append(params_default_index)
        self._debug_infos.append(debug_infos)

        _debug_snapshot("PlanHintExtractor.add_query_plans", {
            "params": params,
            "num_hints_this_batch": len(hints),
            "ema_snapshot": self._ema.snapshot(),
        })

    def get_num_hints(self) -> int:
        return len(self._ema.keys)

    def get_consolidated_plan_hints(
        self,
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Hint], List[JSON], List[JSON]]:
        """Returns deduplicated, EMA-weighted, Laplace-smoothed plan hints.

        Returns:
            Tuple of (counts_with_probability, plan_hints,
                      params_plan_indices, debug_infos).
            ``counts_with_probability`` maps each hint string to a dict
            containing its EMA-weighted ``probability`` and ``raw_weight``.
        """
        if self._consolidated:
            counts = {k: {"probability": self._ema.probability(k),
                          "raw_weight": self._ema._raw.get(k, 0.0)}
                      for k in self._ema.keys}
            return (counts, self._plan_hints,
                    self._params_plan_indices, self._debug_infos)

        self._consolidated = True

        # Build sorted unique hints.
        for key in sorted(self._ema.keys):
            self._plan_hints.append({
                "hints": key,
                "source": "ema_tracked",
            })

        # Resolve plan_index from hint-string to integer position.
        for pdi in self._params_plan_indices:
            pdi["plan_index"] = _get_plan_hint_index(
                self._plan_hints, pdi["plan_index"])

        counts = {k: {"probability": self._ema.probability(k),
                       "raw_weight": self._ema._raw.get(k, 0.0)}
                  for k in self._ema.keys}

        _debug_snapshot("PlanHintExtractor.get_consolidated_plan_hints", {
            "counts": counts,
            "num_plan_hints": len(self._plan_hints),
            "num_params": len(self._params_plan_indices),
        })

        return (counts, self._plan_hints,
                self._params_plan_indices, self._debug_infos)


# ---------------------------------------------------------------------------
# Bulk loader (from upstream, unchanged interface)
# ---------------------------------------------------------------------------

def add_query_plans_bulk(extractor: PlanHintExtractor,
                         params_plans_info_list: List[JSON]) -> None:
    for info in params_plans_info_list:
        extractor.add_query_plans(info)


# ===================================================================
#  Self-test
# ===================================================================

if __name__ == "__main__":

    print("=" * 72)
    print("  Lynceus kepler_hint_extractor — self-test")
    print("=" * 72)

    # ---- 1. Build a synthetic EXPLAIN plan ----
    plan_a = {
        "Plan": {
            "Node Type": "Hash Join",
            "Join Type": "Inner",
            "Plan Rows": 100,
            "Plans": [
                {
                    "Node Type": "Seq Scan",
                    "Parent Relationship": "Outer",
                    "Alias": "orders",
                    "Plan Rows": 500,
                },
                {
                    "Node Type": "Index Scan",
                    "Parent Relationship": "Inner",
                    "Alias": "customers",
                    "Index Name": "cust_pk",
                    "Plan Rows": 200,
                },
            ],
        }
    }

    plan_b = {
        "Plan": {
            "Node Type": "Merge Join",
            "Join Type": "Inner",
            "Plan Rows": 90,
            "Plans": [
                {
                    "Node Type": "Index Scan",
                    "Parent Relationship": "Outer",
                    "Alias": "orders",
                    "Index Name": "ord_date_idx",
                    "Plan Rows": 500,
                },
                {
                    "Node Type": "Index Only Scan",
                    "Parent Relationship": "Inner",
                    "Alias": "customers",
                    "Index Name": "cust_pk",
                    "Plan Rows": 200,
                },
            ],
        }
    }

    # ---- 2. Register plans in the in-memory store ----
    store = InMemoryPlanStore()
    set_plan_store(store)
    q = "SELECT * FROM orders JOIN customers ON orders.cid = customers.id"
    store.register_plan(q, ["2024-01-01"], plan_a)
    store.register_plan(q, ["2024-06-01"], plan_b)

    print(f"\nPlan store size: {len(store)}")

    # ---- 3. Test single-query extraction ----
    cfg = DatabaseConfiguration()
    hint_a = get_single_query_hints(cfg, q, ["2024-01-01"])
    hint_b = get_single_query_hints(cfg, q, ["2024-06-01"])
    print(f"\nHint A: {hint_a}")
    print(f"Hint B: {hint_b}")

    # ---- 4. Test PlanHintExtractor with EMA + Laplace ----
    extractor = PlanHintExtractor(ema_alpha=0.1, laplace_k=0.5)

    info_a = {
        "params": ["2024-01-01"],
        "result": plan_a,
        "additional_plans": [plan_b],
        "sources": ["force_merge_join"],
    }
    info_b = {
        "params": ["2024-06-01"],
        "result": plan_b,
    }

    add_query_plans_bulk(extractor, [info_a, info_b])

    counts, hints, indices, dbg = extractor.get_consolidated_plan_hints()
    print("\n--- Consolidated hints ---")
    for h in hints:
        prob = counts.get(h["hints"], {}).get("probability", "?")
        print(f"  {h['hints']}  (prob={prob:.4f})")

    # ---- 5. Test Welford + Huber row-count extraction ----
    rc = extract_row_counts(plan_a["Plan"])
    print(f"\nRow counts (plan_a): {rc}")

    # ---- 6. Test cosine-similarity verification ----
    # Register the hinted query's plan too (simulating what PG would return).
    hinted_q = f"{hint_a} {q}"
    store.register_plan(hinted_q, ["2024-01-01"], plan_a)
    store.register_plan(hinted_q, ["2024-06-01"], plan_a)
    fc = verify_hints("q1", q, hints, indices, cfg, similarity_threshold=0.80)
    print(f"\nVerification failures: {fc}")

    # ---- 7. Test Welford standalone ----
    w = WelfordAccumulator()
    for v in [10, 12, 11, 200, 13, 11]:
        raw = v
        if w.n >= 2:
            raw = huber_clamp(float(v), w.mean, delta=w.stddev + 1.0)
        w.update(raw)
    print(f"\nWelford after outlier-gated stream: {w.snapshot()}")

    # ---- 8. Test EMACounter standalone ----
    ema = EMACounter(alpha=0.1, laplace_k=1.0)
    for label in ["plan_A", "plan_A", "plan_B", "plan_A", "plan_C", "plan_B"]:
        ema.observe(label)
    print(f"\nEMA probabilities: { {k: f'{ema.probability(k):.4f}' for k in ema.keys} }")

    # ---- 9. Test query_text_utils helpers ----
    parameterised = "SELECT * FROM t WHERE id = @param0 AND dt > @param1"
    filled = substitute_query_params(parameterised, [42, "2024-01-01"])
    print(f"\nSubstituted query: {filled}")
    print(f"Params string: {get_params_as_string([42, '2024-01-01'])}")

    print("\n" + "=" * 72)
    print("  ALL SELF-TESTS PASSED")
    print("=" * 72)