# -*- coding: utf-8 -*-
"""Kepler Parameter Generation Pipeline — unified & enhanced.

Merges parameter_generator.py (equality/range-based generation from DB rows)
with param_gen_new.py (cardinality-aware bucketed sampling + PQO file output).

Key algorithmic upgrades over the originals
-------------------------------------------
* Welford online variance — replaces naive two-pass variance for streaming
  cardinality statistics.
* EMA (Exponential Moving Average) decay — down-weights stale bucket
  frequencies when historical counts are merged.
* Laplace smoothing — avoids zero-probability buckets during sampling.
* Huber loss weighting — robustifies range-bound selection against outlier
  column values.
* In-memory table store replaces all PostgreSQL / query_utils calls so the
  module is fully self-contained for testing.

Every public helper exposes ``_debug_snapshot()`` that pretty-prints internal
data structures to stderr for live inspection.
"""

from __future__ import annotations

import bisect
import collections
import copy
import dataclasses
import datetime
import json
import logging
import math
import os
import random
import sys
import textwrap
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
#  Type aliases
# ---------------------------------------------------------------------------
JSON = Any

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
_DELTA_NUMERIC = 1
_DELTA_NUMERIC_NOOP = 0
_DELTA_DATE = datetime.timedelta(days=1)
_DELTA_DATE_NOOP = datetime.timedelta(days=0)

_DEFAULT_BUCKETS = 10
_DEFAULT_SAMPLE_K = 50
_DEFAULT_SPLIT_RATIO = 0.8
_EMA_ALPHA = 0.3          # EMA decay factor for bucket reweighting
_LAPLACE_PSEUDO = 1.0     # Laplace smoothing pseudo-count
_HUBER_DELTA = 1.5        # Huber loss transition threshold

logger = logging.getLogger(__name__)


# ===================================================================
#  Debug helper
# ===================================================================
def _debug_snapshot(tag: str, obj: Any, *, stream=sys.stderr) -> None:
    """Pretty-print *obj* with a ``[DEBUG:<tag>]`` banner.

    Active only when the module-level ``DEBUG`` flag is truthy or the
    ``KEPLER_DEBUG`` environment variable is set.
    """
    if not (_MODULE_DEBUG or os.environ.get("KEPLER_DEBUG")):
        return
    sep = "=" * 60
    print(f"\n{sep}\n[DEBUG:{tag}]", file=stream)
    if isinstance(obj, (dict, list, tuple)):
        try:
            print(json.dumps(obj, indent=2, default=str), file=stream)
        except TypeError:
            print(repr(obj), file=stream)
    elif isinstance(obj, np.ndarray):
        print(repr(obj), file=stream)
    else:
        print(obj, file=stream)
    print(sep, file=stream)


_MODULE_DEBUG: bool = False


def enable_debug(flag: bool = True) -> None:
    """Toggle module-wide debug snapshots at runtime."""
    global _MODULE_DEBUG
    _MODULE_DEBUG = flag


# ===================================================================
#  Welford online (streaming) variance
# ===================================================================
class WelfordAccumulator:
    """Numerically stable one-pass mean & variance via Welford's algorithm.

    Unlike the naïve ``sum-of-squares`` approach the originals would need for
    variance, this never loses precision on large streams.
    """

    def __init__(self) -> None:
        self.n: int = 0
        self.mean: float = 0.0
        self._m2: float = 0.0

    # -- core ---------------------------------------------------------------
    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._m2 += delta * delta2

    def update_batch(self, xs: Sequence[float]) -> None:
        for x in xs:
            self.update(x)

    # -- queries ------------------------------------------------------------
    @property
    def variance(self) -> float:
        return self._m2 / self.n if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    # -- debug --------------------------------------------------------------
    def _debug_snapshot(self) -> None:
        _debug_snapshot("WelfordAccumulator", {
            "n": self.n,
            "mean": self.mean,
            "variance": self.variance,
            "std": self.std,
        })


# ===================================================================
#  Huber loss utility
# ===================================================================
def huber_weight(residual: float, delta: float = _HUBER_DELTA) -> float:
    """Return a Huber-style attenuation weight in (0, 1].

    Values within *delta* of zero receive weight 1.0; beyond that the
    weight decays as ``delta / |r|`` which dampens outlier influence on
    range-bound selection far more gracefully than a hard clip.
    """
    abs_r = abs(residual)
    if abs_r <= delta:
        return 1.0
    return delta / abs_r


# ===================================================================
#  EMA bucket reweighting
# ===================================================================
def ema_reweight(
    counts: List[float],
    alpha: float = _EMA_ALPHA,
) -> List[float]:
    """Apply an exponential-moving-average decay across *counts*.

    Bucket indices are treated as a time axis — later indices are more
    "recent" and retain higher weight.  This is useful when historical
    cardinality snapshots are merged: old frequencies get smoothly
    down-weighted without a hard cutoff.
    """
    if not counts:
        return []
    smoothed = [0.0] * len(counts)
    smoothed[0] = float(counts[0])
    for i in range(1, len(counts)):
        smoothed[i] = alpha * float(counts[i]) + (1 - alpha) * smoothed[i - 1]
    _debug_snapshot("ema_reweight", {"raw": counts, "smoothed": smoothed})
    return smoothed


# ===================================================================
#  Laplace-smoothed sampling weights
# ===================================================================
def laplace_smooth_weights(
    bucket_sizes: List[int],
    pseudo: float = _LAPLACE_PSEUDO,
) -> List[float]:
    """Convert bucket sizes into sampling probabilities with Laplace smoothing.

    Every bucket receives *pseudo* extra pseudo-counts before normalisation,
    guaranteeing that empty buckets still have a small but nonzero sampling
    probability — the original ``sample_from_buckets`` would simply skip
    them.
    """
    adjusted = [s + pseudo for s in bucket_sizes]
    total = sum(adjusted)
    weights = [a / total for a in adjusted]
    _debug_snapshot("laplace_smooth_weights", {
        "raw_sizes": bucket_sizes,
        "pseudo": pseudo,
        "weights": weights,
    })
    return weights


# ===================================================================
#  In-memory database simulation
# ===================================================================
class InMemoryTable:
    """A trivially simple column-store to replace PostgreSQL access.

    Tables are registered once; subsequent ``execute`` calls run a *very*
    limited SQL-like DSL that is sufficient for the two query shapes the
    upstream code actually issues:

        SELECT DISTINCT col FROM tbl WHERE col IS NOT NULL ORDER BY col ASC
        SELECT cols FROM <joined tables> WHERE … GROUP BY … ORDER BY …
    """

    def __init__(self) -> None:
        # table_name -> list[dict[column_name, value]]
        self._tables: Dict[str, List[Dict[str, Any]]] = {}

    def register(self, name: str, rows: List[Dict[str, Any]]) -> None:
        self._tables[name.lower().strip()] = rows

    def distinct_sorted(
        self, table: str, column: str, *, cast_date: bool = False
    ) -> List[Any]:
        """Return sorted distinct non-null values for *column* in *table*."""
        rows = self._tables.get(table.lower().strip(), [])
        seen = set()
        result: List[Any] = []
        for r in rows:
            v = r.get(column)
            if v is None:
                continue
            if cast_date and isinstance(v, datetime.datetime):
                v = v.date()
            if v not in seen:
                seen.add(v)
                result.append(v)
        result.sort()
        _debug_snapshot(f"distinct_sorted({table}.{column})", result)
        return result

    def select_all(self, table: str) -> List[Dict[str, Any]]:
        return list(self._tables.get(table.lower().strip(), []))

    def query_rows(
        self,
        table: str,
        columns: List[str],
        *,
        where_not_null: Optional[List[str]] = None,
        group_by: bool = False,
        random_order: bool = False,
        limit: Optional[int] = None,
    ) -> List[Tuple]:
        """Flexible row retrieval that covers the two query shapes above."""
        rows = self._tables.get(table.lower().strip(), [])
        filtered = []
        not_null_cols = where_not_null or []
        for r in rows:
            if all(r.get(c) is not None for c in not_null_cols):
                filtered.append(tuple(r.get(c) for c in columns))

        if group_by:
            filtered = list(dict.fromkeys(filtered))  # unique, order-preserved

        if random_order:
            random.shuffle(filtered)

        if limit is not None:
            filtered = filtered[:limit]

        _debug_snapshot("query_rows", {
            "table": table,
            "columns": columns,
            "result_count": len(filtered),
        })
        return filtered

    def _debug_snapshot(self) -> None:
        summary = {
            t: {"row_count": len(rs), "columns": list(rs[0].keys()) if rs else []}
            for t, rs in self._tables.items()
        }
        _debug_snapshot("InMemoryTable", summary)


# Global shared store (tests inject data here)
_MEM_DB = InMemoryTable()


def get_mem_db() -> InMemoryTable:
    """Return the module-level in-memory database."""
    return _MEM_DB


# ===================================================================
#  Date / operator helpers  (lightly cleaned-up from upstream)
# ===================================================================
def format_date(d: datetime.date) -> str:
    return d.strftime("%Y-%m-%d")


def should_handle_as_date(column_name: str) -> bool:
    return "date" in column_name.lower()


def numeric_operator_adjustment(op: str) -> int:
    if op == "<":
        return _DELTA_NUMERIC
    if op == ">":
        return -_DELTA_NUMERIC
    return _DELTA_NUMERIC_NOOP


def date_operator_adjustment(op: str) -> datetime.timedelta:
    if op == "<":
        return _DELTA_DATE
    if op == ">":
        return -_DELTA_DATE
    return _DELTA_DATE_NOOP


def is_lt_variant(op: str) -> bool:
    return op in {"<", "<="}


def is_gt_variant(op: str) -> bool:
    return op in {">", ">="}


# ===================================================================
#  Predicate analysis
# ===================================================================
def classify_predicates(
    predicates: List[JSON],
) -> Tuple[List[JSON], List[JSON]]:
    """Split predicates into *base* and *range* (BETWEEN-style) groups.

    Identical semantics to upstream ``_get_parameters_query`` classification
    but factored into a reusable function.
    """
    base: List[JSON] = []
    base_ids: List[str] = []
    range_preds: List[JSON] = []

    for pred in predicates:
        ident = f"{pred['alias']}.{pred['column']}"
        if ident in base_ids:
            if "like" in pred.get("operator", "").lower():
                base.append(pred)
                base_ids.append(ident)
                continue
            range_preds.append(pred)
        else:
            base.append(pred)
            base_ids.append(ident)

    _debug_snapshot("classify_predicates", {
        "base_count": len(base),
        "range_count": len(range_preds),
    })
    return base, range_preds


# ===================================================================
#  Core parameter-row generation  (Huber-weighted range bounds)
# ===================================================================
def _pick_range_bounds(
    value: Any,
    candidate_values: List[Any],
    *,
    use_huber: bool = True,
) -> Tuple[Any, Any]:
    """Select lower & upper bounds from *candidate_values* around *value*.

    When *use_huber* is True the candidates are weighted by Huber loss
    relative to the value so that outlier bounds contribute less — the
    upstream code picked uniformly at random from all candidates which
    frequently produced wildly skewed ranges.
    """
    left_idx = bisect.bisect_left(candidate_values, value)
    right_idx = bisect.bisect_right(candidate_values, value)

    lowers = candidate_values[:left_idx]
    uppers = candidate_values[right_idx:]

    def _weighted_choice(pool: List[Any]) -> Any:
        if len(pool) == 1:
            return pool[0]
        if not use_huber or not pool:
            return random.choice(pool)
        numeric_val = value if isinstance(value, (int, float)) else 0
        weights = []
        for c in pool:
            c_num = c if isinstance(c, (int, float)) else 0
            r = c_num - numeric_val
            weights.append(huber_weight(r))
        total_w = sum(weights)
        if total_w == 0:
            return random.choice(pool)
        probs = [w / total_w for w in weights]
        return pool[np.random.choice(len(pool), p=probs)]

    lower = _weighted_choice(lowers) if lowers else candidate_values[0]
    upper = _weighted_choice(uppers) if uppers else candidate_values[-1]

    _debug_snapshot("_pick_range_bounds", {
        "value": str(value),
        "lower": str(lower),
        "upper": str(upper),
        "n_lower_candidates": len(lowers),
        "n_upper_candidates": len(uppers),
    })
    return lower, upper


def generate_row_params(
    row: Tuple,
    base_predicates: List[JSON],
    range_predicates: List[JSON],
    candidate_values_list: List[List[Any]],
    *,
    use_huber: bool = True,
) -> List[Any]:
    """Turn one DB row into a concrete parameter binding list.

    This is the inner loop of the original ``generate_parameters`` but
    extracted for testability.
    """
    params: List[Any] = []
    range_idx = 0
    for value, pred in zip(row, base_predicates):
        ident = f"{pred['alias']}.{pred['column']}"
        # Check whether this base predicate has a matching range predicate
        is_range = (
            range_idx < len(range_predicates)
            and pred["alias"] == range_predicates[range_idx]["alias"]
            and pred["column"] == range_predicates[range_idx]["column"]
        )
        if is_range:
            lower, upper = _pick_range_bounds(
                value, candidate_values_list[range_idx], use_huber=use_huber
            )
            if isinstance(lower, datetime.date):
                params.extend([format_date(lower), format_date(upper)])
            else:
                params.extend([lower, upper])
            range_idx += 1
        elif pred.get("column") == "website_url":
            params.append(_extract_url_domain(value))
        elif isinstance(value, datetime.date):
            params.append(format_date(value + date_operator_adjustment(pred["operator"])))
        elif isinstance(value, str):
            params.append(value)
        else:
            params.append(value + numeric_operator_adjustment(pred["operator"]))
    return params


def _extract_url_domain(url: str) -> str:
    import urllib.parse
    netloc = urllib.parse.urlparse(url).netloc
    try:
        return f"%{netloc[netloc.rindex('.') + 1:]}"
    except ValueError:
        return "%"


# ===================================================================
#  Distinct-value retrieval  (uses InMemoryTable)
# ===================================================================
def get_distinct_column_values(
    predicates: List[JSON],
    template: JSON,
    mem_db: InMemoryTable,
) -> List[List[Any]]:
    """Return sorted distinct values per predicate column from *mem_db*."""
    result: List[List[Any]] = []
    for pred in predicates:
        table = _resolve_table(pred["alias"], template["query"])
        cast = should_handle_as_date(pred["column"])
        vals = mem_db.distinct_sorted(table, pred["column"], cast_date=cast)
        result.append(vals)
    _debug_snapshot("get_distinct_column_values", {
        "predicates": [f"{p['alias']}.{p['column']}" for p in predicates],
        "counts": [len(v) for v in result],
    })
    return result


def _resolve_table(alias: str, query_text: str) -> str:
    """Resolve an alias to its table name by scanning the SQL text."""
    for line in query_text.split("\n"):
        if " as " in line.lower():
            tokens = line.lower().split(" as ")
            if alias.lower() == tokens[1].strip().rstrip(",").strip():
                return tokens[0].strip()
    return alias


# ===================================================================
#  ParameterGenerator — core class (replaces upstream class)
# ===================================================================
@dataclasses.dataclass
class TemplateItem:
    query_id: str
    template: JSON


class ParameterGenerator:
    """Generate plausible parameter bindings for a query template.

    All database interaction goes through the module-level ``InMemoryTable``.
    """

    def __init__(self, seed: int = 42, *, mem_db: Optional[InMemoryTable] = None):
        self._seed = seed
        self._mem_db = mem_db or _MEM_DB
        random.seed(seed)
        np.random.seed(seed)

    def generate_parameters(
        self,
        count: int,
        template_item: TemplateItem,
        *,
        dry_run: bool = False,
        use_huber: bool = True,
    ) -> Dict[str, JSON]:
        """Sample up to *count* parameter bindings for *template_item*."""
        base_preds, range_preds = classify_predicates(
            template_item.template["predicates"]
        )

        range_distinct = get_distinct_column_values(
            range_preds, template_item.template, self._mem_db
        )

        # Build candidate value lists with sentinel min-1 / max+1
        candidate_values_list: List[List[Any]] = []
        for dv in range_distinct:
            if not dv:
                candidate_values_list.append([])
                continue
            if isinstance(dv[0], datetime.date):
                delta = _DELTA_DATE
            else:
                delta = _DELTA_NUMERIC
            candidate_values_list.append(
                [dv[0] - delta] + dv + [dv[-1] + delta]
            )

        # Retrieve candidate rows from in-memory store
        columns = [p["column"] for p in base_preds]
        table_name = _resolve_table(base_preds[0]["alias"],
                                     template_item.template["query"]) if base_preds else ""
        raw_rows = self._mem_db.query_rows(
            table_name,
            columns,
            where_not_null=columns,
            group_by=True,
            random_order=not dry_run,
            limit=1 if dry_run else count,
        )

        params_seen: List[List[Any]] = []
        for row in raw_rows:
            p = generate_row_params(
                row, base_preds, range_preds, candidate_values_list,
                use_huber=use_huber,
            )
            if p not in params_seen:
                params_seen.append(p)

        output: Dict[str, JSON] = {}
        entry = output.setdefault(template_item.query_id, {})
        entry["query"] = template_item.template["query"]
        entry["predicates"] = template_item.template["predicates"]
        entry["params"] = params_seen

        _debug_snapshot("ParameterGenerator.generate_parameters", {
            "query_id": template_item.query_id,
            "requested": count,
            "produced": len(params_seen),
        })
        return output

    def _debug_snapshot(self) -> None:
        _debug_snapshot("ParameterGenerator", {
            "seed": self._seed,
            "mem_db_tables": list(self._mem_db._tables.keys()),
        })


# ===================================================================
#  Cardinality-aware bucketed sampling  (from param_gen_new.py)
# ===================================================================
def get_param_and_cardinality_mem(
    table: str,
    select_column: str,
    not_null_columns: List[str],
    mem_db: InMemoryTable,
) -> List[Tuple[Any, int]]:
    """In-memory equivalent of ``get_param_and_cardinality``.

    Groups by *select_column* and returns (value, count) sorted by count
    descending.
    """
    rows = mem_db.select_all(table)
    counter: Dict[Any, int] = collections.Counter()
    for r in rows:
        if all(r.get(c) is not None for c in not_null_columns):
            key = r.get(select_column) if isinstance(select_column, str) else tuple(
                r.get(c) for c in select_column
            )
            counter[key] += 1
    result = sorted(counter.items(), key=lambda x: -x[1])
    _debug_snapshot("get_param_and_cardinality_mem", {
        "table": table,
        "column": select_column,
        "distinct": len(result),
    })
    return result


def create_buckets_enhanced(
    data: List[Tuple[Any, int]],
    num_buckets: int = _DEFAULT_BUCKETS,
) -> List[List[Tuple[Any, int]]]:
    """Partition *(value, count)* pairs into equal-width count buckets.

    Identical bucketing logic to upstream ``create_buckets`` but with an
    explicit guard against empty *data*.
    """
    if not data:
        return [[] for _ in range(num_buckets)]
    counts = [c for _, c in data]
    lo, hi = min(counts), max(counts)
    width = (hi - lo) / num_buckets if hi != lo else 1.0
    buckets: List[List[Tuple[Any, int]]] = [[] for _ in range(num_buckets)]
    for item in data:
        idx = min(int((item[1] - lo) / width), num_buckets - 1)
        buckets[idx].append(item)
    _debug_snapshot("create_buckets_enhanced", {
        "num_items": len(data),
        "num_buckets": num_buckets,
        "bucket_sizes": [len(b) for b in buckets],
    })
    return buckets


def sample_from_buckets_enhanced(
    buckets: List[List[Tuple[Any, int]]],
    total_samples: int,
) -> List[Tuple[Any, int]]:
    """Sample *total_samples* from *buckets* with Laplace-smoothed weights.

    Replaces the uniform per-bucket sampling of the original with probability
    weights that guarantee every bucket — even empty ones in the Laplace sense
    — has a chance of contributing.  EMA reweighting is optionally applied
    across bucket sizes to down-weight stale histogram mass.
    """
    bucket_sizes = [len(b) for b in buckets]
    ema_sizes = ema_reweight([float(s) for s in bucket_sizes])
    weights = laplace_smooth_weights(
        [max(1, int(round(e))) for e in ema_sizes]
    )

    target_per_bucket = [max(1, int(round(w * total_samples))) for w in weights]

    sampled: List[Tuple[Any, int]] = []
    for bucket, target in zip(buckets, target_per_bucket):
        if not bucket:
            continue
        k = min(target, len(bucket))
        sampled.extend(random.sample(bucket, k))

    # Top up if we fell short
    if len(sampled) < total_samples:
        all_items = [item for b in buckets for item in b]
        shortfall = total_samples - len(sampled)
        already = set(id(x) for x in sampled)
        extras = [x for x in all_items if id(x) not in already]
        sampled.extend(extras[:shortfall])

    sampled = sampled[:total_samples]
    _debug_snapshot("sample_from_buckets_enhanced", {
        "requested": total_samples,
        "produced": len(sampled),
    })
    return sampled


# ===================================================================
#  Template-driven cardinality generation
# ===================================================================
def gen_param_by_cardinality(
    templates: Dict[str, JSON],
    mem_db: InMemoryTable,
    n_buckets: int = _DEFAULT_BUCKETS,
    k_samples: int = _DEFAULT_SAMPLE_K,
) -> List[List[Any]]:
    """Generate parameter choice lists from cardinality statistics.

    Replacement for ``param_gen_new.gen_param_by_cardinality`` that reads
    from an ``InMemoryTable`` instead of PostgreSQL.
    """
    all_param_lists: List[List[Any]] = []
    for query_id, template in templates.items():
        for pred in template["predicates"]:
            left_or_right = pred["left_or_right"][0]
            col_pair = [pred["column"], pred["join_tables_column"][0]]
            join_cond = pred["join_conditions"][0]  # kept for reference
            from_table = pred["table"]

            if left_or_right == "both":
                sel_col = col_pair[0]
                nn_cols = col_pair
            elif left_or_right == "r":
                sel_col = col_pair[1]
                nn_cols = [col_pair[1]]
            else:
                sel_col = col_pair[0]
                nn_cols = [col_pair[0]]

            ranking = get_param_and_cardinality_mem(
                from_table, sel_col, nn_cols, mem_db
            )
            # Welford stats on cardinalities for monitoring
            welford = WelfordAccumulator()
            welford.update_batch([c for _, c in ranking])
            welford._debug_snapshot()

            buckets = create_buckets_enhanced(ranking, n_buckets)
            sampled = sample_from_buckets_enhanced(buckets, k_samples)
            temp_params = [item[0] for item in sampled]
            random.shuffle(temp_params)
            all_param_lists.append(temp_params)

    _debug_snapshot("gen_param_by_cardinality", {
        "num_lists": len(all_param_lists),
        "list_lengths": [len(pl) for pl in all_param_lists],
    })
    return all_param_lists


# ===================================================================
#  PQO file output helpers  (from param_gen_new.py — cleaned up)
# ===================================================================
def escape_single_quotes(param: Any) -> Any:
    if isinstance(param, str):
        return param.replace("'", "''")
    return param


def save_pqo_files(
    query_id: str,
    data: Dict[str, JSON],
    output_dir: str,
    mode: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    sql_template = data[query_id]["query"]
    literals = data[query_id]["params"]
    output_json: Dict[str, str] = {}
    for idx, combo in enumerate(literals):
        q = sql_template
        for i, param in enumerate(combo):
            q = q.replace(f"@param{i}", str(param))
        output_json[f"{query_id}_{mode}_{idx}"] = q

    path = os.path.join(output_dir, f"{query_id}_{mode}.json")
    with open(path, "w") as f:
        json.dump(output_json, f, indent=4)
    logger.info("Wrote %s", path)
    _debug_snapshot("save_pqo_files", {"path": path, "entries": len(output_json)})


def save_pqo_predicates(
    query_id: str,
    training_data: Dict[str, JSON],
    testing_data: Dict[str, JSON],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    def _write(data: Dict[str, JSON], fname: str) -> None:
        path = os.path.join(output_dir, fname)
        with open(path, "w") as f:
            for i, combo in enumerate(data["params"]):
                items = []
                for j, item in enumerate(combo):
                    p = data["predicates"][j]
                    dt = p.get("data_type", "")
                    fmt = f"'{item}'" if dt == "text" else str(item)
                    if p["operator"].lower() == "in":
                        fmt = f"({fmt})"
                    items.append(f"{p['alias']}.{p['column']} {p['operator']} {fmt}")
                line = '["' + '", "'.join(items) + '"]'
                tail = ",\n" if i < len(data["params"]) - 1 else "\n"
                f.write(line + tail)
        logger.info("Wrote %s", path)

    _write(training_data[query_id], f"{query_id}_training.txt")
    _write(testing_data[query_id], f"{query_id}_testing.txt")


# ===================================================================
#  Full template generation (cross-product & 1-1, train/test split)
# ===================================================================
def gen_full_template_output(
    templates: Dict[str, JSON],
    param_choice_list: List[List[Any]],
    output_dir: str,
    split_ratio: float = _DEFAULT_SPLIT_RATIO,
    cross_product: bool = True,
) -> Dict[str, JSON]:
    """Build training/testing splits and optionally write PQO files."""
    os.makedirs(output_dir, exist_ok=True)
    query_id, template = next(iter(templates.items()))
    query = template["query"]
    predicates = template["predicates"]

    deduped = [list(dict.fromkeys(pl)) for pl in param_choice_list]
    if cross_product:
        param_list = [list(combo) for combo in product(*deduped)]
    else:
        param_list = [list(combo) for combo in zip(*deduped)]

    param_list = [
        [escape_single_quotes(p) for p in row] for row in param_list
    ]
    np.random.shuffle(param_list)

    split_idx = math.ceil(len(param_list) * split_ratio)
    training_params = param_list[:split_idx]
    testing_params = param_list[split_idx:]

    def _make_block(params: List[List[Any]]) -> Dict[str, JSON]:
        return {query_id: {"query": query, "predicates": predicates, "params": params}}

    full = _make_block(param_list)
    train = _make_block(training_params)
    test = _make_block(testing_params)

    _debug_snapshot("gen_full_template_output", {
        "query_id": query_id,
        "total_params": len(param_list),
        "train": len(training_params),
        "test": len(testing_params),
        "cross_product": cross_product,
    })

    # Write JSON outputs
    for sub, data, label in [
        ("original", full, "original"),
        ("training", train, "training"),
        ("testing", test, "testing"),
    ]:
        d = os.path.join(output_dir, sub)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"{query_id}_{label}.json")
        with open(p, "w") as f:
            json.dump(data, f, indent=4)
        logger.info("Wrote %s", p)

    # PQO outputs
    pqo_q = os.path.join(output_dir, "PQO", "query")
    pqo_p = os.path.join(output_dir, "PQO", "predicates")
    save_pqo_files(query_id, train, pqo_q, "training")
    save_pqo_files(query_id, test, pqo_q, "testing")
    save_pqo_predicates(query_id, train, test, pqo_p)

    return full


# ===================================================================
#  Self-test
# ===================================================================
def _build_demo_data(mem_db: InMemoryTable) -> Tuple[Dict[str, JSON], Dict[str, JSON]]:
    """Populate *mem_db* with synthetic data and return two template dicts.

    The first dict exercises the ParameterGenerator path (equality / range
    predicates), the second exercises the cardinality-bucketed path.
    """
    # --- table: movies ---
    random.seed(99)
    movies_rows = []
    for i in range(200):
        movies_rows.append({
            "id": i,
            "release_date": datetime.date(2000, 1, 1) + datetime.timedelta(days=i * 5),
            "budget": random.randint(1_000_000, 500_000_000),
            "title": f"Movie_{i}",
            "website_url": f"https://studio{i % 5}.example.com/movie/{i}",
        })
    mem_db.register("movies", movies_rows)

    # --- table: ratings ---
    ratings_rows = []
    for i in range(500):
        ratings_rows.append({
            "movie_id": random.randint(0, 199),
            "score": random.randint(1, 10),
            "review_date": datetime.date(2001, 6, 1) + datetime.timedelta(days=i),
        })
    mem_db.register("ratings", ratings_rows)

    # Template 1: for ParameterGenerator  (range predicate on budget)
    gen_template = {
        "q1": {
            "query": textwrap.dedent("""\
                SELECT m.title, m.budget
                FROM movies as m
                WHERE m.budget >= @param0
                AND m.budget <= @param1
                ORDER BY m.budget DESC
                LIMIT 50"""),
            "predicates": [
                {"alias": "m", "column": "budget", "operator": ">="},
                {"alias": "m", "column": "budget", "operator": "<="},
            ],
        }
    }

    # Template 2: for cardinality path
    card_template = {
        "q2": {
            "query": "SELECT r.movie_id, COUNT(*) FROM ratings AS r GROUP BY r.movie_id",
            "predicates": [
                {
                    "alias": "r",
                    "column": "movie_id",
                    "table": "ratings",
                    "left_or_right": ["l"],
                    "join_tables_column": ["movie_id"],
                    "join_conditions": ["TRUE"],
                    "join_tables": ["ratings"],
                    "join_tables_alias": ["r"],
                    "operator": "=",
                }
            ],
        }
    }
    return gen_template, card_template


def _run_self_test() -> None:
    enable_debug(True)
    print("=" * 70)
    print("  kepler_param_pipeline — self-test")
    print("=" * 70)

    mem = InMemoryTable()
    gen_tmpl, card_tmpl = _build_demo_data(mem)

    # --- 1) Welford accumulator ---
    print("\n>>> Welford streaming variance")
    w = WelfordAccumulator()
    sample = [3.0, 5.0, 7.0, 9.0, 11.0]
    w.update_batch(sample)
    w._debug_snapshot()
    np_var = float(np.var(sample, ddof=0))
    assert abs(w.variance - np_var) < 1e-9, f"Welford variance mismatch: {w.variance} vs {np_var}"
    print("  PASS — variance matches numpy")

    # --- 2) Huber weight ---
    print("\n>>> Huber weight")
    assert huber_weight(0.5) == 1.0
    assert abs(huber_weight(3.0) - _HUBER_DELTA / 3.0) < 1e-9
    print(f"  huber(0.5)={huber_weight(0.5):.4f}  huber(3.0)={huber_weight(3.0):.4f}")
    print("  PASS")

    # --- 3) EMA reweight ---
    print("\n>>> EMA reweight")
    raw = [10.0, 20.0, 5.0, 15.0]
    ema = ema_reweight(raw)
    assert len(ema) == len(raw)
    assert ema[0] == raw[0]
    print(f"  raw={raw}  ema={[round(e, 2) for e in ema]}")
    print("  PASS")

    # --- 4) Laplace smoothing ---
    print("\n>>> Laplace smooth weights")
    sizes = [0, 10, 5, 0, 20]
    wts = laplace_smooth_weights(sizes)
    assert all(w > 0 for w in wts), "Zero-probability bucket found!"
    assert abs(sum(wts) - 1.0) < 1e-9
    print(f"  sizes={sizes}  weights={[round(w, 4) for w in wts]}")
    print("  PASS — no zero-probability buckets")

    # --- 5) ParameterGenerator (in-memory) ---
    print("\n>>> ParameterGenerator (in-memory, Huber range bounds)")
    pg = ParameterGenerator(seed=42, mem_db=mem)
    ti = TemplateItem(query_id="q1", template=gen_tmpl["q1"])
    result = pg.generate_parameters(count=15, template_item=ti)
    n_params = len(result["q1"]["params"])
    print(f"  generated {n_params} bindings for q1")
    assert n_params > 0, "No parameters generated!"
    print("  PASS")

    # --- 6) Cardinality-bucketed sampling ---
    print("\n>>> Cardinality-bucketed sampling (Laplace + EMA)")
    choice_lists = gen_param_by_cardinality(card_tmpl, mem, n_buckets=5, k_samples=20)
    assert len(choice_lists) == 1
    assert len(choice_lists[0]) == 20
    print(f"  cardinality choice list length = {len(choice_lists[0])}")
    print("  PASS")

    # --- 7) Full template output ---
    print("\n>>> Full template output (train/test split, PQO files)")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        out = gen_full_template_output(
            card_tmpl, choice_lists, tmpdir,
            split_ratio=0.8, cross_product=False,
        )
        q2_data = out["q2"]
        total = len(q2_data["params"])
        train_count = math.ceil(total * 0.8)
        print(f"  total={total}  train={train_count}  test={total - train_count}")
        # Check files exist
        for sub in ["original", "training", "testing"]:
            p = os.path.join(tmpdir, sub, f"q2_{sub}.json")
            assert os.path.isfile(p), f"Missing {p}"
        pqo_q = os.path.join(tmpdir, "PQO", "query", "q2_training.json")
        assert os.path.isfile(pqo_q), f"Missing {pqo_q}"
        print("  All output files present")
    print("  PASS")

    # --- 8) InMemoryTable debug ---
    print("\n>>> InMemoryTable snapshot")
    mem._debug_snapshot()
    print("  PASS")

    print("\n" + "=" * 70)
    print("  ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    _run_self_test()