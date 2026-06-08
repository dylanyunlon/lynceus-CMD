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

# === M198 extensions from query_utils.py (558L upstream) ===
# ──────────────────────────────────────────────────────────────────────────────
# Ported from upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/query_utils.py
# Algorithm changes (~20%):
#   1. SQL AST hashing: MD5 → SHA-256
#   2. Parameterised-query fingerprint: FNV-1a → Rabin fingerprint (64-bit)
#   3. Query similarity: cosine/edit-distance → Jaccard on token n-grams
#
# Author: dylanyunlon <dogechat@163.com>
# ──────────────────────────────────────────────────────────────────────────────

import dataclasses
import enum
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Optional heavy deps – gracefully absent in unit-test environments
# ---------------------------------------------------------------------------
try:
    import psycopg2
    import psycopg2.errorcodes
    _PSYCOPG2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PSYCOPG2_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Internal debug helper
# ──────────────────────────────────────────────────────────────────────────────

def _dbg(fn_name: str, **kwargs) -> None:
    """Print all named arguments to stdout for debugging."""
    parts = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    print(f"[DBG] {fn_name}({parts})")


# ──────────────────────────────────────────────────────────────────────────────
# Postgres catalogue queries (verbatim from upstream)
# ──────────────────────────────────────────────────────────────────────────────

_GET_INDEXES_QUERY = """
SELECT
    tablename,
    indexname,
    indexdef
FROM
    pg_indexes
WHERE
    schemaname = 'public'
ORDER BY
    tablename,
    indexname;
"""

_POSTGRES_COST_CONSTANTS: List[str] = [
    "seq_page_cost", "random_page_cost", "cpu_tuple_cost",
    "cpu_index_tuple_cost", "cpu_operator_cost", "parallel_setup_cost",
    "parallel_tuple_cost", "min_parallel_table_scan_size",
    "min_parallel_index_scan_size", "effective_cache_size", "jit_above_cost",
    "jit_inline_above_cost", "jit_optimize_above_cost",
]

_POSTGRES_RESOURCE_CONFIGS: List[str] = [
    "shared_buffers", "huge_pages", "temp_buffers", "max_prepared_transactions",
    "work_mem", "hash_mem_multiplier", "maintenance_work_mem",
    "autovacuum_work_mem", "max_stack_depth", "shared_memory_type",
    "dynamic_shared_memory_type", "temp_file_limit", "max_files_per_process",
]

JSON = Any


# ──────────────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────────────

class KeplerPostgresDataType(enum.Enum):
    """Postgres column types relevant to Kepler training pipelines."""
    INTEGER   = "integer"
    VARCHAR   = "character varying"
    DATE      = "date"
    TIMESTAMP = "timestamp without time zone"


@dataclasses.dataclass
class KeplerDatabaseConfiguration:
    """Connection parameters for a Kepler-managed Postgres instance."""
    dbname:   str
    user:     Optional[str]   = None
    password: Optional[str]   = None
    host:     Optional[str]   = "localhost"
    seed:     Optional[float] = 0


# ──────────────────────────────────────────────────────────────────────────────
# Algorithm change 1: SHA-256 SQL AST hash
# ──────────────────────────────────────────────────────────────────────────────

def _sql_ast_hash_sha256(sql: str) -> str:
    """Return a SHA-256 hex digest of the structurally normalised SQL.

    Unlike the original MD5-based approach, SHA-256 is collision-resistant
    enough to serve as a stable cache key across large query corpora.

    Args:
        sql: Raw or parameterised SQL string.

    Returns:
        64-character lowercase hex string.
    """
    _dbg("_sql_ast_hash_sha256", sql=sql)
    normalised = normalize_sql(sql)          # reuse existing helper
    digest = hashlib.sha256(normalised.encode("utf-8")).hexdigest()
    _dbg("_sql_ast_hash_sha256", normalised_preview=normalised[:60], digest=digest)
    return digest


# ──────────────────────────────────────────────────────────────────────────────
# Algorithm change 2: Rabin fingerprint for parameterised queries
# ──────────────────────────────────────────────────────────────────────────────

# Rabin polynomial constants (64-bit Galois-field polynomial)
_RABIN_MOD  = (1 << 61) - 1          # Mersenne prime M61
_RABIN_BASE = 131                     # chosen prime base


def _rabin_fingerprint(text: str) -> int:
    """Compute a 64-bit Rabin fingerprint of *text*.

    Uses Rabin–Karp rolling-hash arithmetic over M61 (2^61 − 1), a Mersenne
    prime.  Replaces the FNV-1a variant used upstream.

    Args:
        text: Arbitrary string (SQL template after literal stripping).

    Returns:
        Non-negative integer fingerprint < 2^61.
    """
    _dbg("_rabin_fingerprint", text_len=len(text), text_preview=text[:60])
    h = 0
    for ch in text:
        h = (h * _RABIN_BASE + ord(ch)) % _RABIN_MOD
    _dbg("_rabin_fingerprint", fingerprint=h)
    return h


def compute_parameterised_fingerprint(sql: str) -> int:
    """Produce a Rabin fingerprint for a parameterised query template.

    Literals and numeric constants are first stripped so that two queries
    that differ only in bound values share the same fingerprint.

    Args:
        sql: SQL string, possibly containing @param0 / $1 / ? placeholders
             or raw literal values.

    Returns:
        64-bit Rabin fingerprint as a Python int.
    """
    _dbg("compute_parameterised_fingerprint", sql=sql)
    stripped = _strip_literals(sql)
    fp = _rabin_fingerprint(stripped)
    _dbg("compute_parameterised_fingerprint", stripped_preview=stripped[:60], fingerprint=fp)
    return fp


def _strip_literals(sql: str) -> str:
    """Remove string literals and numeric constants from SQL, preserving structure.

    Args:
        sql: Raw SQL string.

    Returns:
        SQL string with literals replaced by the token ``<LIT>``.
    """
    _dbg("_strip_literals", sql_len=len(sql))
    s = re.sub(r"'[^']*'", "<LIT>", sql)
    s = re.sub(r"\b\d+(?:\.\d+)?\b", "<LIT>", s)
    _dbg("_strip_literals", result_preview=s[:80])
    return s


# ──────────────────────────────────────────────────────────────────────────────
# Algorithm change 3: Jaccard similarity on token n-grams
# ──────────────────────────────────────────────────────────────────────────────

def _token_ngrams(sql: str, n: int = 2) -> Set[Tuple[str, ...]]:
    """Build a set of token-level n-grams from a normalised SQL string.

    Args:
        sql: SQL string (will be normalised internally).
        n:   N-gram width (default 2 = bigrams).

    Returns:
        Set of n-gram tuples.
    """
    _dbg("_token_ngrams", sql_len=len(sql), n=n)
    tokens = normalize_sql(sql).split()
    grams: Set[Tuple[str, ...]] = set()
    for i in range(len(tokens) - n + 1):
        grams.add(tuple(tokens[i : i + n]))
    _dbg("_token_ngrams", gram_count=len(grams))
    return grams


def jaccard_sql_similarity(sql_a: str, sql_b: str, n: int = 2) -> float:
    """Compute Jaccard similarity between two SQL queries using token n-grams.

    Replaces the cosine / edit-distance approach used upstream.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Args:
        sql_a: First SQL string.
        sql_b: Second SQL string.
        n:     N-gram width (default 2).

    Returns:
        Float in [0.0, 1.0]; 1.0 means structurally identical after
        normalisation.
    """
    _dbg("jaccard_sql_similarity", sql_a_len=len(sql_a), sql_b_len=len(sql_b), n=n)
    grams_a = _token_ngrams(sql_a, n)
    grams_b = _token_ngrams(sql_b, n)
    union_size = len(grams_a | grams_b)
    if union_size == 0:
        _dbg("jaccard_sql_similarity", result=1.0, reason="both_empty")
        return 1.0
    score = len(grams_a & grams_b) / union_size
    _dbg("jaccard_sql_similarity",
         intersection=len(grams_a & grams_b),
         union=union_size,
         score=score)
    return score


# ──────────────────────────────────────────────────────────────────────────────
# KeplerQueryManager  (ported from upstream QueryManager)
# ──────────────────────────────────────────────────────────────────────────────

class KeplerQueryManager:
    """Wraps query, DDL, and DML access to Postgres for Kepler pipelines.

    Ported from upstream QueryManager; uses SHA-256 AST hashing and Rabin
    fingerprints internally.  Requires psycopg2 at runtime.

    Attributes:
        database_configuration: Connection parameters.
    """

    def __init__(self, database_configuration: KeplerDatabaseConfiguration) -> None:
        """Connect to Postgres via psycopg2.

        Args:
            database_configuration: Connection parameters for the target DB.
        """
        _dbg("KeplerQueryManager.__init__",
             dbname=database_configuration.dbname,
             host=database_configuration.host,
             user=database_configuration.user,
             seed=database_configuration.seed)

        if not _PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 is required for KeplerQueryManager")

        self.database_configuration = database_configuration

        self._conn = psycopg2.connect(
            dbname=self.database_configuration.dbname,
            user=self.database_configuration.user,
            password=self.database_configuration.password,
            host=self.database_configuration.host,
            port=5431,
        )
        self._cursor = self._conn.cursor()
        self._cursor.execute(f"SELECT SETSEED({self.database_configuration.seed})")
        self._cursor.execute("LOAD 'pg_hint_plan'")

    # ------------------------------------------------------------------
    # Cursor management
    # ------------------------------------------------------------------

    def refresh_cursor(self) -> None:
        """Close and reopen the internal cursor."""
        _dbg("refresh_cursor")
        self._cursor.close()
        self._cursor = self._conn.cursor()

    def enable_pg_hint_plan_debug(self) -> None:
        """Enable detailed pg_hint_plan debug output."""
        _dbg("enable_pg_hint_plan_debug")
        self._cursor.execute("set pg_hint_plan.debug_print to detailed;")

    def disable_pg_hint_plan_debug(self) -> None:
        """Disable pg_hint_plan debug output."""
        _dbg("disable_pg_hint_plan_debug")
        self._cursor.execute("set pg_hint_plan.debug_print to off;")

    def run_analyze(self) -> None:
        """Run ANALYZE on the database."""
        _dbg("run_analyze")
        self._cursor.execute("analyze;")

    # ------------------------------------------------------------------
    # Configuration introspection
    # ------------------------------------------------------------------

    def get_postgres_config_info(self) -> JSON:
        """Return indexes, cost constants, and resource configs as a dict.

        Returns:
            Dict with keys 'indexes', 'cost_constants', 'resource_configs'.
        """
        _dbg("get_postgres_config_info")
        return {
            "indexes": self.get_index_info(),
            "cost_constants": self.get_cost_constants(),
            "resource_configs": self.get_resource_configs(),
        }

    def get_index_info(self) -> JSON:
        """Fetch all public-schema index definitions.

        Returns:
            List of (tablename, indexname, indexdef) tuples.
        """
        _dbg("get_index_info")
        self._cursor.execute(_GET_INDEXES_QUERY)
        return self._cursor.fetchall()

    def get_cost_constants(self) -> JSON:
        """Return current values of Postgres query cost constants.

        Returns:
            Dict mapping constant name → value string.
        """
        _dbg("get_cost_constants")
        return {
            cfg: self.execute(f"SHOW {cfg};")
            for cfg in _POSTGRES_COST_CONSTANTS
        }

    def get_resource_configs(self) -> JSON:
        """Return current values of Postgres resource configuration parameters.

        Returns:
            Dict mapping parameter name → value string.
        """
        _dbg("get_resource_configs")
        return {
            cfg: self.execute(f"SHOW {cfg};")
            for cfg in _POSTGRES_RESOURCE_CONFIGS
        }

    # ------------------------------------------------------------------
    # DML / DDL
    # ------------------------------------------------------------------

    def execute_and_commit(self, sql: str) -> None:
        """Execute *sql* and immediately commit the transaction.

        Args:
            sql: DDL or DML statement.
        """
        _dbg("execute_and_commit", sql_preview=sql[:120])
        self._cursor.execute(sql)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Timed execution (server-side timing via pg_stat_statements)
    # ------------------------------------------------------------------

    def execute_timed(
        self,
        query: str,
        params: Optional[Sequence[Any]] = None,
        timeout_ms: Optional[float] = None,
    ) -> Tuple[Optional[float], Optional[int]]:
        """Execute a query and return server-side wall-clock time.

        Uses ``pg_stat_statements`` to measure execution time rather than
        client-side timing, giving per-query accuracy.

        Args:
            query:      SQL template with @param0 … @paramN placeholders.
            params:     Sequence of values to bind (all cast to str).
            timeout_ms: Optional statement timeout in milliseconds.

        Returns:
            ``(exec_time_ms, row_count)`` on success, or
            ``(None, None)`` if the query times out.
        """
        _dbg("execute_timed",
             query_preview=query[:80],
             params=params,
             timeout_ms=timeout_ms)

        self.refresh_cursor()
        executable_query = _substitute_query_params(query, params)

        if timeout_ms:
            self._cursor.execute(f"SET statement_timeout TO '{timeout_ms}'")

        self._cursor.execute("select pg_stat_statements_reset();")

        result: Optional[float] = None
        rows = 0
        try:
            self._cursor.execute(executable_query)
            while True:
                batch = self._cursor.fetchmany(10000)
                if not batch:
                    break
                rows += len(batch)

            if timeout_ms:
                self._cursor.execute("SET statement_timeout TO 0")

            self._cursor.execute(
                "SELECT total_exec_time from pg_stat_statements "
                "where query != 'BEGIN' "
                "and query not like '%pg_stat_statements_reset%' "
                "and query not like '%SET statement_timeout TO%' and calls = 1;"
            )
            row = self._cursor.fetchone()
            assert row and len(row) == 1
            result = row[0]
            assert isinstance(result, float)

        except psycopg2.OperationalError as e:
            assert e.pgcode == psycopg2.errorcodes.QUERY_CANCELED
            self._cursor.execute("END")
            if timeout_ms:
                self._cursor.execute("SET statement_timeout TO 0")

        _dbg("execute_timed", result_ms=result, rows=rows)
        return (result, rows) if result is not None else (None, None)

    # ------------------------------------------------------------------
    # Timed execution (client-side timing)
    # ------------------------------------------------------------------

    def execute_timed_local(
        self,
        query: str,
        params: Optional[Sequence[Any]] = None,
        timeout_ms: Optional[float] = None,
    ) -> Tuple[Optional[float], Optional[int]]:
        """Execute a query and measure elapsed time on the client side.

        Useful when running many queries concurrently where
        ``pg_stat_statements`` isolation is impractical.

        Args:
            query:      SQL template with @param0 … @paramN placeholders.
            params:     Sequence of values to bind.
            timeout_ms: Optional per-query timeout in milliseconds.

        Returns:
            ``(elapsed_ms, row_count)`` on success, or
            ``(None, None)`` on timeout.
        """
        _dbg("execute_timed_local",
             query_preview=query[:80],
             params=params,
             timeout_ms=timeout_ms)

        executable_query = _substitute_query_params(query, params)
        self._cursor.execute("BEGIN")
        if timeout_ms:
            self._cursor.execute(f"SET LOCAL statement_timeout TO '{timeout_ms}'")

        result_ms: Optional[float] = None
        rows = 0
        try:
            start = time.time()
            self._cursor.execute(executable_query)
            result_ms = (time.time() - start) * 1000

            while True:
                batch = self._cursor.fetchmany(10000)
                if not batch:
                    break
                rows += len(batch)
            self._cursor.execute("COMMIT")

        except psycopg2.OperationalError as e:
            assert e.pgcode == psycopg2.errorcodes.QUERY_CANCELED
            self._cursor.execute("END")

        _dbg("execute_timed_local", result_ms=result_ms, rows=rows)
        return (result_ms, rows) if result_ms is not None else (None, None)

    # ------------------------------------------------------------------
    # Plain SELECT execution
    # ------------------------------------------------------------------

    def execute(
        self,
        query: str,
        params: Optional[Sequence[Any]] = None,
    ) -> List[Tuple[Any, ...]]:
        """Execute a SELECT and return all result rows.

        Args:
            query:  SQL template with @param0 … @paramN placeholders.
            params: Sequence of values to bind.

        Returns:
            List of row tuples.
        """
        _dbg("execute", query_preview=query[:80], params=params)
        self.refresh_cursor()
        executable_query = _substitute_query_params(query, params)
        self._cursor.execute(executable_query)
        rows = self._cursor.fetchall()
        _dbg("execute", row_count=len(rows))
        return rows

    # ------------------------------------------------------------------
    # EXPLAIN ANALYZE
    # ------------------------------------------------------------------

    def get_query_plan_and_execute(
        self,
        query: str,
        params: Optional[Sequence[Any]] = None,
        configuration_parameters: Optional[Sequence[str]] = None,
    ) -> Any:
        """Run EXPLAIN (FORMAT JSON, ANALYZE, BUFFERS) and return the plan.

        Args:
            query:                    SQL template with placeholders.
            params:                   Bind values.
            configuration_parameters: Postgres optimizer knobs to disable
                                      (e.g. ``['enable_nestloop']``).

        Returns:
            EXPLAIN ANALYZE output as a parsed JSON object.
        """
        _dbg("get_query_plan_and_execute",
             query_preview=query[:80],
             params=params,
             configuration_parameters=configuration_parameters)

        executable_query = _substitute_query_params(query, params)
        query_string = f"EXPLAIN (FORMAT JSON, ANALYZE, BUFFERS) {executable_query}"
        return self._execute_query_with_configs(query_string, configuration_parameters)

    # ------------------------------------------------------------------
    # EXPLAIN (no execute)
    # ------------------------------------------------------------------

    def get_query_plan(
        self,
        query: str,
        params: Optional[Sequence[Any]] = None,
        configuration_parameters: Optional[List[str]] = None,
    ) -> Any:
        """Retrieve the EXPLAIN plan (no execution) for a SELECT query.

        Args:
            query:                    SQL template with placeholders.
            params:                   Bind values.
            configuration_parameters: Optimizer knobs to disable.

        Returns:
            EXPLAIN plan as a parsed JSON object.
        """
        _dbg("get_query_plan",
             query_preview=query[:80],
             params=params,
             configuration_parameters=configuration_parameters)

        executable_query = _substitute_query_params(query, params)
        query_string = f"EXPLAIN (FORMAT JSON) {executable_query}"
        return self._execute_query_with_configs(query_string, configuration_parameters)

    # ------------------------------------------------------------------
    # Internal helper for config-toggling execution
    # ------------------------------------------------------------------

