"""
Kepler Query Utilities
----------------------
Parse, normalize, fingerprint, and manipulate SQL query templates.
Pure numpy implementation. Every function has a _dbg() variant.
"""

import hashlib
import json
import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# Data classes
# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

@dataclass
class Predicate:
    """Single predicate extracted from a SQL WHERE clause."""
    column: str
    operator: str          # =, <, >, <=, >=, !=, LIKE, IN, BETWEEN, IS
    value: Optional[str]   # literal value or placeholder ($1, ?)
    table_alias: str = ""  # alias / table prefix if present

    def _dbg(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "operator": self.operator,
            "value": self.value,
            "table_alias": self.table_alias,
        }


@dataclass
class QueryTemplate:
    """Parsed query template with slots for parameter binding."""
    raw_sql: str
    normalized: str
    param_slots: List[str]            # ordered list of placeholder names
    predicates: List[Predicate] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    fingerprint: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def _dbg(self) -> Dict[str, Any]:
        return {
            "normalized_preview": self.normalized[:120],
            "param_count": len(self.param_slots),
            "predicate_count": len(self.predicates),
            "tables": self.tables,
            "fingerprint": self.fingerprint,
        }


# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# SQL normalization
# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

_WS_RE = re.compile(r"\s+")
_COMMENT_LINE_RE = re.compile(r"--[^\n]*")
_COMMENT_BLOCK_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LITERAL_STR_RE = re.compile(r"'[^']*'")
_LITERAL_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def normalize_sql(sql: str) -> str:
    """
    Normalize a SQL string: strip comments, collapse whitespace,
    lower-case keywords, replace literals with placeholders.
    """
    s = _COMMENT_BLOCK_RE.sub(" ", sql)
    s = _COMMENT_LINE_RE.sub(" ", s)
    s = s.strip().rstrip(";")
    s = _LITERAL_STR_RE.sub("?", s)
    s = _LITERAL_NUM_RE.sub("?", s)
    s = _WS_RE.sub(" ", s).strip()
    s = s.lower()
    return s


def normalize_sql_dbg(sql: str) -> Dict[str, Any]:
    normed = normalize_sql(sql)
    return {
        "original_len": len(sql),
        "normalized_len": len(normed),
        "normalized": normed,
    }


# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# Fingerprint
# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def compute_query_fingerprint(sql: str) -> str:
    """SHA-256 hex digest of the normalized SQL."""
    normed = normalize_sql(sql)
    return hashlib.sha256(normed.encode("utf-8")).hexdigest()


def compute_query_fingerprint_dbg(sql: str) -> Dict[str, Any]:
    fp = compute_query_fingerprint(sql)
    normed = normalize_sql(sql)
    return {"fingerprint": fp, "normalized_preview": normed[:80]}


# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# Predicate extraction
# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

_PRED_RE = re.compile(
    r"(?:(\w+)\.)?(\w+)"               # optional alias.column
    r"\s*(=|<>|!=|<=|>=|<|>|LIKE|IN|IS(?:\s+NOT)?|BETWEEN)"  # operator
    r"\s*"
    r"([^,\)]+?)"                       # value / placeholder
    r"(?=\s+AND\b|\s+OR\b|\s*\)|\s*$)",
    re.IGNORECASE,
)


def extract_predicates(sql: str) -> List[Predicate]:
    """
    Extract WHERE-clause predicates from a SQL statement.

    Returns a list of Predicate dataclasses.  Works on typical
    SELECT â¦ WHERE â¦ AND/OR â¦ patterns; not a full SQL parser.
    """
    # isolate the WHERE clause (up to GROUP BY / ORDER BY / LIMIT / HAVING / ;)
    where_match = re.search(
        r"\bWHERE\b(.+?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|\bHAVING\b|;|$)",
        sql,
        re.IGNORECASE | re.DOTALL,
    )
    if where_match is None:
        return []

    where_text = where_match.group(1)
    preds: List[Predicate] = []
    for m in _PRED_RE.finditer(where_text):
        alias = m.group(1) or ""
        col = m.group(2)
        op = m.group(3).upper().strip()
        val = m.group(4).strip().rstrip(";")
        preds.append(Predicate(column=col, operator=op, value=val, table_alias=alias))
    return preds


def extract_predicates_dbg(sql: str) -> Dict[str, Any]:
    preds = extract_predicates(sql)
    return {
        "count": len(preds),
        "predicates": [p._dbg() for p in preds],
    }


# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# Table extraction (simple)
# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

_FROM_TABLE_RE = re.compile(
    r"(?:\bFROM\b|\bJOIN\b)\s+(\w+)", re.IGNORECASE
)


def _extract_tables(sql: str) -> List[str]:
    return list(dict.fromkeys(m.group(1).lower() for m in _FROM_TABLE_RE.finditer(sql)))


def _extract_tables_dbg(sql: str) -> Dict[str, Any]:
    tables = _extract_tables(sql)
    return {"tables": tables, "count": len(tables)}


# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# Template parsing
# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

_PARAM_SLOT_RE = re.compile(r"\$(\d+)|\?")


def parse_query_template(json_input: str) -> QueryTemplate:
    """
    Parse a JSON object describing a query template.

    Expected JSON keys
    ------------------
    sql : str          â the parameterized SQL text
    metadata : dict    â optional extra info (label, hints, â¦)

    Returns
    -------
    QueryTemplate with normalized SQL, extracted predicates, tables, fingerprint.
    """
    obj = json.loads(json_input)
    raw_sql: str = obj["sql"]
    meta: Dict[str, Any] = obj.get("metadata", {})

    normed = normalize_sql(raw_sql)
    slots = _PARAM_SLOT_RE.findall(raw_sql)
    param_names = [s if s else "?" for s in slots]
    preds = extract_predicates(raw_sql)
    tables = _extract_tables(raw_sql)
    fp = compute_query_fingerprint(raw_sql)

    return QueryTemplate(
        raw_sql=raw_sql,
        normalized=normed,
        param_slots=param_names,
        predicates=preds,
        tables=tables,
        fingerprint=fp,
        metadata=meta,
    )


def parse_query_template_dbg(json_input: str) -> Dict[str, Any]:
    tpl = parse_query_template(json_input)
    return {"template": tpl._dbg(), "raw_len": len(tpl.raw_sql)}


# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# Parameter binding
# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def bind_parameters(template: QueryTemplate, params: Dict[str, str]) -> str:
    """
    Bind parameters to a QueryTemplate, returning executable SQL.

    Parameters
    ----------
    template : parsed QueryTemplate
    params : mapping from slot name ('1','2',â¦ or '?') to literal value.
             Values are **not** escaped â caller is responsible for safety.

    Returns
    -------
    SQL string with placeholders replaced.
    """
    sql = template.raw_sql

    # handle numbered placeholders $1, $2, â¦
    for slot_name, value in sorted(params.items(), key=lambda kv: -len(kv[0])):
        if slot_name == "?":
            sql = sql.replace("?", value, 1)
        else:
            sql = sql.replace(f"${slot_name}", value)
    return sql


def bind_parameters_dbg(template: QueryTemplate, params: Dict[str, str]) -> Dict[str, Any]:
    bound = bind_parameters(template, params)
    return {
        "bound_sql_preview": bound[:160],
        "params_used": list(params.keys()),
        "fingerprint": template.fingerprint,
    }


# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# Selectivity estimation
# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def estimate_selectivity(
    predicate: Predicate,
    column_stats: Dict[str, Any],
) -> float:
    """
    Estimate predicate selectivity from column statistics (pure numpy).

    column_stats keys
    -----------------
    n_distinct : int     â number of distinct values
    min_val    : float   â column minimum
    max_val    : float   â column maximum
    histogram  : list    â equi-width histogram bucket counts
    null_frac  : float   â fraction of NULLs

    Returns
    -------
    Estimated selectivity in [0, 1].
    """
    n_distinct = int(column_stats.get("n_distinct", 100))
    min_val = float(column_stats.get("min_val", 0.0))
    max_val = float(column_stats.get("max_val", 1.0))
    histogram = column_stats.get("histogram", None)
    null_frac = float(column_stats.get("null_frac", 0.0))

    op = predicate.operator

    # IS NULL / IS NOT NULL
    if op == "IS":
        if predicate.value and "NOT" in predicate.value.upper():
            return max(0.0, 1.0 - null_frac)
        return null_frac

    # equality
    if op == "=":
        if n_distinct <= 0:
            return 1.0
        return (1.0 - null_frac) / n_distinct

    # inequality
    if op in ("!=", "<>"):
        if n_distinct <= 0:
            return 1.0
        return (1.0 - null_frac) * (1.0 - 1.0 / n_distinct)

    # range operators â use histogram if available
    val = _safe_float(predicate.value)
    if val is None:
        return 0.33  # default heuristic

    col_range = max_val - min_val
    if col_range <= 0:
        return 0.5

    if histogram is not None:
        hist = np.asarray(histogram, dtype=np.float64)
        total = hist.sum()
        if total == 0:
            return 0.5
        n_buckets = len(hist)
        bucket_width = col_range / n_buckets
        bucket_idx = int((val - min_val) / bucket_width)
        bucket_idx = np.clip(bucket_idx, 0, n_buckets - 1)

        if op in ("<", "<="):
            sel = float(hist[:bucket_idx + 1].sum() / total)
        elif op in (">", ">="):
            sel = float(hist[bucket_idx:].sum() / total)
        else:
            sel = 0.33
        return np.clip(sel * (1.0 - null_frac), 0.0, 1.0).item()

    # fallback: uniform assumption
    if op in ("<", "<="):
        sel = (val - min_val) / col_range
    elif op in (">", ">="):
        sel = (max_val - val) / col_range
    else:
        sel = 0.33

    return float(np.clip(sel * (1.0 - null_frac), 0.0, 1.0))


def estimate_selectivity_dbg(
    predicate: Predicate,
    column_stats: Dict[str, Any],
) -> Dict[str, Any]:
    sel = estimate_selectivity(predicate, column_stats)
    return {
        "selectivity": sel,
        "predicate": predicate._dbg(),
        "n_distinct": column_stats.get("n_distinct"),
        "has_histogram": column_stats.get("histogram") is not None,
    }


# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# Internal helpers
# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def _safe_float(val: Optional[str]) -> Optional[float]:
    """Try to parse a string as float; return None on failure."""
    if val is None:
        return None
    try:
        return float(val.strip().strip("'\""))
    except (ValueError, AttributeError):
        return None


def _safe_float_dbg(val: Optional[str]) -> Dict[str, Any]:
    result = _safe_float(val)
    return {"input": val, "parsed": result, "ok": result is not None}
