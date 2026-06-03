# -*- coding: utf-8 -*-
"""
Original: PAR2QO querylets.py — SQL template fragments for parametric workloads
          (upstream/par2qo/code/querylets.py, Hap-Hugh/PAR2QO)
Modified: Lynceus — heterogeneous query template generator with TPC-H catalog
          and GPU-aware predicate annotations.

Modifications from upstream querylets.py (~80% structure kept, ~20% changed):
  - Removed: psycopg2, hardcoded IMDB/Stats database schemas
  - Kept:    querylet() template dictionary structure (key → SQL fragment)
  - Kept:    gen_one_table_query/stats_join_querylet helper pattern
  - Modified: templates now use TPC-H schema (lineitem, orders, customer, etc.)
  - Modified: added device hint annotations (/*+GPU*/, /*+CPU*/) in templates
  - Added:   QueryletGenerator class for programmatic template construction
  - Added:   workload_from_querylets() for benchmark integration
  - Added:   debug_dump_querylet() for runtime inspection

References:
  PAR2QO querylets.py:12   — querylet() template dict
  PAR2QO querylets.py:1165 — stats_join_querylet (2-table join fragments)
  PAR2QO querylets.py:1204 — stats_single_querylet / stats_complex_querylet
"""
from __future__ import annotations

import hashlib
import itertools
import logging
import time
from dataclasses import dataclass, field
import math
from typing import List, Dict, Optional, Tuple, Any, Set
from enum import Enum, auto

logger = logging.getLogger("lynceus.querylets")


# ── TPC-H Schema Definition ────────────────────────────────────────────
# PAR2QO: hardcoded IMDB tables (cast_info, keyword, title, etc.)
# Lynceus: TPC-H catalog for cost-model benchmark reproducibility

TPCH_TABLES = {
    "lineitem":  {"rows": 6001215, "pk": "l_orderkey,l_linenumber",
                  "cols": ["l_orderkey","l_partkey","l_suppkey","l_linenumber",
                           "l_quantity","l_extendedprice","l_discount","l_tax",
                           "l_returnflag","l_linestatus","l_shipdate",
                           "l_commitdate","l_receiptdate","l_shipinstruct",
                           "l_shipmode","l_comment"]},
    "orders":    {"rows": 1500000, "pk": "o_orderkey",
                  "cols": ["o_orderkey","o_custkey","o_orderstatus",
                           "o_totalprice","o_orderdate","o_orderpriority",
                           "o_clerk","o_shippriority","o_comment"]},
    "customer":  {"rows": 150000, "pk": "c_custkey",
                  "cols": ["c_custkey","c_name","c_address","c_nationkey",
                           "c_phone","c_acctbal","c_mktsegment","c_comment"]},
    "part":      {"rows": 200000, "pk": "p_partkey",
                  "cols": ["p_partkey","p_name","p_mfgr","p_brand","p_type",
                           "p_size","p_container","p_retailprice","p_comment"]},
    "supplier":  {"rows": 10000, "pk": "s_suppkey",
                  "cols": ["s_suppkey","s_name","s_address","s_nationkey",
                           "s_phone","s_acctbal","s_comment"]},
    "partsupp":  {"rows": 800000, "pk": "ps_partkey,ps_suppkey",
                  "cols": ["ps_partkey","ps_suppkey","ps_availqty",
                           "ps_supplycost","ps_comment"]},
    "nation":    {"rows": 25, "pk": "n_nationkey",
                  "cols": ["n_nationkey","n_name","n_regionkey","n_comment"]},
    "region":    {"rows": 5, "pk": "r_regionkey",
                  "cols": ["r_regionkey","r_name","r_comment"]},
}

# Join graph: (left_table, right_table, left_col, right_col)
TPCH_JOINS = [
    ("lineitem", "orders",   "l_orderkey",  "o_orderkey"),
    ("lineitem", "partsupp", "l_partkey,l_suppkey", "ps_partkey,ps_suppkey"),
    ("lineitem", "part",     "l_partkey",   "p_partkey"),
    ("lineitem", "supplier", "l_suppkey",   "s_suppkey"),
    ("orders",   "customer", "o_custkey",   "c_custkey"),
    ("customer", "nation",   "c_nationkey", "n_nationkey"),
    ("supplier", "nation",   "s_nationkey", "n_nationkey"),
    ("nation",   "region",   "n_regionkey", "r_regionkey"),
    ("partsupp", "part",     "ps_partkey",  "p_partkey"),
    ("partsupp", "supplier", "ps_suppkey",  "s_suppkey"),
]


# ── Predicate types ────────────────────────────────────────────────────
class PredicateType(Enum):
    RANGE = auto()       # col BETWEEN ? AND ?
    EQUALITY = auto()    # col = ?
    LIKE = auto()        # col LIKE ?
    IN_LIST = auto()     # col IN (?, ?, ...)
    COMPARISON = auto()  # col > ?


@dataclass
class Predicate:
    """A single filter predicate — Lynceus analog of PAR2QO's {cc}/{kk}."""
    table: str
    column: str
    pred_type: PredicateType
    alias: str = ""
    selectivity_hint: float = 0.1   # estimated selectivity for cost model

    def to_sql(self, param_id: int = 0) -> str:
        col = f"{self.alias}.{self.column}" if self.alias else self.column
        if self.pred_type == PredicateType.RANGE:
            return f"{col} BETWEEN :p{param_id}_lo AND :p{param_id}_hi"
        elif self.pred_type == PredicateType.EQUALITY:
            return f"{col} = :p{param_id}"
        elif self.pred_type == PredicateType.LIKE:
            return f"{col} LIKE :p{param_id}"
        elif self.pred_type == PredicateType.IN_LIST:
            return f"{col} IN (:p{param_id}_list)"
        else:
            return f"{col} > :p{param_id}"

    def __repr__(self) -> str:
        return f"Pred({self.table}.{self.column} {self.pred_type.name} sel≈{self.selectivity_hint:.3f})"


# ── Querylet: a parameterised SQL fragment ─────────────────────────────
@dataclass
class Querylet:
    """A single querylet — parametric SQL fragment.

    PAR2QO: querylet_imdb_dict[template_name] → f-string SQL.
    Lynceus: structured template with table/predicate metadata.
    """
    template_id: str
    tables: List[str]
    aliases: Dict[str, str]   # table → alias
    joins: List[Tuple[str, str, str, str]]  # (left_alias, right_alias, l_col, r_col)
    predicates: List[Predicate]
    sql_template: str = ""
    device_hint: str = ""     # Lynceus: "gpu" | "cpu" | "" for auto

    def fingerprint(self) -> str:
        raw = f"{self.template_id}:{','.join(sorted(self.tables))}"
        return hashlib.md5(raw.encode()).hexdigest()[:10]

    def estimated_rows(self, table_rows: Dict[str, int]) -> float:
        """v3: cardinality with join-order damping (diminishing selectivity per join)."""
        total = 1.0
        for t in self.tables:
            total *= table_rows.get(t, 10000)
        for p in self.predicates:
            total *= p.selectivity_hint
        # v3: diminishing join selectivity — each additional join is less selective
        for j_idx in range(len(self.joins)):
            damped_sel = 0.001 * math.pow(0.5, j_idx)  # halves each subsequent join
            total *= damped_sel
        return max(total, 1.0)

    def to_sql(self) -> str:
        if self.sql_template:
            return self.sql_template
        # Build SQL from components
        select = "SELECT COUNT(*)"
        frm = "FROM " + ", ".join(
            f"{t} AS {self.aliases.get(t, t)}" for t in self.tables
        )
        where_parts = []
        for j_idx, (la, ra, lc, rc) in enumerate(self.joins):
            where_parts.append(f"{la}.{lc} = {ra}.{rc}")
        for p_idx, p in enumerate(self.predicates):
            where_parts.append(p.to_sql(p_idx))
        where = "WHERE " + " AND ".join(where_parts) if where_parts else ""
        hint = f"/*+ {self.device_hint.upper()} */" if self.device_hint else ""
        return f"{hint} {select}\n{frm}\n{where}"


# ── Querylet dictionary (from PAR2QO querylets.py:12) ──────────────────
# PAR2QO: 100+ IMDB-specific templates.
# Lynceus: TPC-H-based templates covering key join patterns.

def querylet_tpch(cc: str = "1=1", kk: str = "1=1") -> Dict[str, Querylet]:
    """Generate TPC-H querylet dictionary.

    PAR2QO: querylet(db, kk, cc, template_name) → dict of SQL strings.
    Lynceus: returns dict of structured Querylet objects.
    """
    templates = {}

    # ── Q1-style: lineitem scan with date range ──
    templates["tpch_q1_lineitem_scan"] = Querylet(
        template_id="tpch_q1_lineitem_scan",
        tables=["lineitem"],
        aliases={"lineitem": "l"},
        joins=[],
        predicates=[
            Predicate("lineitem", "l_shipdate", PredicateType.RANGE,
                      alias="l", selectivity_hint=0.05),
        ],
        sql_template=f"""
            SELECT l_returnflag, l_linestatus,
                   SUM(l_quantity), SUM(l_extendedprice)
            FROM lineitem AS l
            WHERE {cc} AND l.l_shipdate <= DATE '1998-12-01'
            GROUP BY l_returnflag, l_linestatus
        """,
    )

    # ── Q3-style: lineitem-orders-customer 3-way join ──
    templates["tpch_q3_lo_c"] = Querylet(
        template_id="tpch_q3_lo_c",
        tables=["lineitem", "orders", "customer"],
        aliases={"lineitem": "l", "orders": "o", "customer": "c"},
        joins=[
            ("l", "o", "l_orderkey", "o_orderkey"),
            ("o", "c", "o_custkey", "c_custkey"),
        ],
        predicates=[
            Predicate("customer", "c_mktsegment", PredicateType.EQUALITY,
                      alias="c", selectivity_hint=0.2),
            Predicate("orders", "o_orderdate", PredicateType.COMPARISON,
                      alias="o", selectivity_hint=0.5),
            Predicate("lineitem", "l_shipdate", PredicateType.COMPARISON,
                      alias="l", selectivity_hint=0.5),
        ],
        sql_template=f"""
            SELECT l.l_orderkey, SUM(l.l_extendedprice * (1 - l.l_discount))
            FROM customer AS c, orders AS o, lineitem AS l
            WHERE {cc}
              AND c.c_mktsegment = :p0
              AND c.c_custkey = o.o_custkey
              AND l.l_orderkey = o.o_orderkey
              AND o.o_orderdate < :p1
              AND l.l_shipdate > :p2
            GROUP BY l.l_orderkey
        """,
    )

    # ── Q5-style: 6-way join through nation/region ──
    templates["tpch_q5_6way"] = Querylet(
        template_id="tpch_q5_6way",
        tables=["customer", "orders", "lineitem", "supplier", "nation", "region"],
        aliases={"customer": "c", "orders": "o", "lineitem": "l",
                 "supplier": "s", "nation": "n", "region": "r"},
        joins=[
            ("c", "o", "c_custkey", "o_custkey"),
            ("l", "o", "l_orderkey", "o_orderkey"),
            ("l", "s", "l_suppkey", "s_suppkey"),
            ("c", "n", "c_nationkey", "n_nationkey"),
            ("s", "n", "s_nationkey", "n_nationkey"),
            ("n", "r", "n_regionkey", "r_regionkey"),
        ],
        predicates=[
            Predicate("region", "r_name", PredicateType.EQUALITY,
                      alias="r", selectivity_hint=0.2),
            Predicate("orders", "o_orderdate", PredicateType.RANGE,
                      alias="o", selectivity_hint=0.15),
        ],
        device_hint="gpu",  # big joins benefit from GPU
    )

    # ── Q9-style: partsupp-lineitem-part-supplier-orders-nation ──
    templates["tpch_q9_profit"] = Querylet(
        template_id="tpch_q9_profit",
        tables=["part", "supplier", "lineitem", "partsupp", "orders", "nation"],
        aliases={"part": "p", "supplier": "s", "lineitem": "l",
                 "partsupp": "ps", "orders": "o", "nation": "n"},
        joins=[
            ("l", "ps", "l_partkey,l_suppkey", "ps_partkey,ps_suppkey"),
            ("l", "o",  "l_orderkey", "o_orderkey"),
            ("ps", "s", "ps_suppkey", "s_suppkey"),
            ("ps", "p", "ps_partkey", "p_partkey"),
            ("s", "n",  "s_nationkey", "n_nationkey"),
        ],
        predicates=[
            Predicate("part", "p_name", PredicateType.LIKE,
                      alias="p", selectivity_hint=0.02),
        ],
        device_hint="gpu",
    )

    # ── Q12-style: lineitem-orders 2-way with predicates ──
    templates["tpch_q12_shipmode"] = Querylet(
        template_id="tpch_q12_shipmode",
        tables=["orders", "lineitem"],
        aliases={"orders": "o", "lineitem": "l"},
        joins=[("o", "l", "o_orderkey", "l_orderkey")],
        predicates=[
            Predicate("lineitem", "l_shipmode", PredicateType.IN_LIST,
                      alias="l", selectivity_hint=0.28),
            Predicate("lineitem", "l_commitdate", PredicateType.COMPARISON,
                      alias="l", selectivity_hint=0.5),
            Predicate("lineitem", "l_shipdate", PredicateType.RANGE,
                      alias="l", selectivity_hint=0.15),
            Predicate("lineitem", "l_receiptdate", PredicateType.COMPARISON,
                      alias="l", selectivity_hint=0.5),
        ],
    )

    # ── Q14-style: lineitem-part ──
    templates["tpch_q14_promo"] = Querylet(
        template_id="tpch_q14_promo",
        tables=["lineitem", "part"],
        aliases={"lineitem": "l", "part": "p"},
        joins=[("l", "p", "l_partkey", "p_partkey")],
        predicates=[
            Predicate("lineitem", "l_shipdate", PredicateType.RANGE,
                      alias="l", selectivity_hint=0.08),
        ],
    )

    # ── Q18-style: large-volume customer-orders-lineitem ──
    templates["tpch_q18_large_volume"] = Querylet(
        template_id="tpch_q18_large_volume",
        tables=["customer", "orders", "lineitem"],
        aliases={"customer": "c", "orders": "o", "lineitem": "l"},
        joins=[
            ("c", "o", "c_custkey", "o_custkey"),
            ("o", "l", "o_orderkey", "l_orderkey"),
        ],
        predicates=[
            Predicate("lineitem", "l_quantity", PredicateType.COMPARISON,
                      alias="l", selectivity_hint=0.01),
        ],
        device_hint="gpu",  # aggregate-heavy → GPU
    )

    # ── Q21-style: complex multi-way with EXISTS/NOT EXISTS ──
    templates["tpch_q21_supplier_wait"] = Querylet(
        template_id="tpch_q21_supplier_wait",
        tables=["supplier", "lineitem", "orders", "nation"],
        aliases={"supplier": "s", "lineitem": "l", "orders": "o", "nation": "n"},
        joins=[
            ("s", "l", "s_suppkey", "l_suppkey"),
            ("o", "l", "o_orderkey", "l_orderkey"),
            ("s", "n", "s_nationkey", "n_nationkey"),
        ],
        predicates=[
            Predicate("orders", "o_orderstatus", PredicateType.EQUALITY,
                      alias="o", selectivity_hint=0.5),
            Predicate("nation", "n_name", PredicateType.EQUALITY,
                      alias="n", selectivity_hint=0.04),
            Predicate("lineitem", "l_receiptdate", PredicateType.COMPARISON,
                      alias="l", selectivity_hint=0.5),
        ],
    )

    return templates


# ── Join querylet helpers (from PAR2QO querylets.py:1165) ──────────────
def make_join_querylet(
    left_table: str,
    right_table: str,
    left_alias: str,
    right_alias: str,
    join_col_left: str,
    join_col_right: str,
    predicates: Optional[List[Predicate]] = None,
    device_hint: str = "",
) -> Querylet:
    """Construct a 2-table join querylet.

    PAR2QO: stats_join_querylet(left_alias, right_alias, l_r_b, cc, kk)
    Lynceus: generic for any schema with optional device hint.
    """
    template_id = f"join_{left_table}_{right_table}"
    return Querylet(
        template_id=template_id,
        tables=[left_table, right_table],
        aliases={left_table: left_alias, right_table: right_alias},
        joins=[(left_alias, right_alias, join_col_left, join_col_right)],
        predicates=predicates or [],
        device_hint=device_hint,
    )


def make_single_table_querylet(
    table: str,
    alias: str,
    predicates: Optional[List[Predicate]] = None,
) -> Querylet:
    """Construct a single-table scan querylet.

    PAR2QO: gen_one_table_query(table_name, condition).
    """
    return Querylet(
        template_id=f"scan_{table}",
        tables=[table],
        aliases={table: alias},
        joins=[],
        predicates=predicates or [],
    )


# ── QueryletGenerator: programmatic template construction ──────────────
class QueryletGenerator:
    """Generate querylets for workload simulation.

    PAR2QO: querylet() is a flat function returning a dict.
    Lynceus: class-based generator with TPC-H catalog awareness.
    """

    def __init__(
        self,
        schema: Optional[Dict[str, Dict]] = None,
        join_graph: Optional[List[Tuple[str, str, str, str]]] = None,
        debug: bool = True,
    ):
        self.schema = schema or TPCH_TABLES
        self.join_graph = join_graph or TPCH_JOINS
        self.debug = debug
        self._templates: Dict[str, Querylet] = {}

        if debug:
            print(f"  ├─ QueryletGenerator: {len(self.schema)} tables, "
                  f"{len(self.join_graph)} join edges")

    def generate_all(self) -> Dict[str, Querylet]:
        """Generate the full querylet dictionary.

        PAR2QO: querylet(db, kk, cc, template_name).
        Lynceus: generates all TPC-H templates.
        """
        t0 = time.time()
        self._templates = querylet_tpch()

        # Also generate all possible 2-table join querylets from join graph
        for left, right, lcol, rcol in self.join_graph:
            la = left[0]
            ra = right[0]
            if la == ra:
                ra = right[:2]
            qlet = make_join_querylet(left, right, la, ra, lcol, rcol)
            key = qlet.template_id
            if key not in self._templates:
                self._templates[key] = qlet

        elapsed = time.time() - t0
        if self.debug:
            print(f"  ├─ generated {len(self._templates)} querylets in {elapsed:.3f}s")
            for tid, qlet in list(self._templates.items())[:5]:
                est = qlet.estimated_rows(
                    {t: self.schema[t]["rows"] for t in self.schema}
                )
                print(f"  │   {tid}: {len(qlet.tables)} tables, "
                      f"~{est:.0f} rows, device={qlet.device_hint or 'auto'}")

        return self._templates

    def get_template(self, template_id: str) -> Optional[Querylet]:
        if not self._templates:
            self.generate_all()
        return self._templates.get(template_id)

    def list_templates(self) -> List[str]:
        if not self._templates:
            self.generate_all()
        return list(self._templates.keys())

    def templates_for_tables(self, tables: Set[str]) -> List[Querylet]:
        """Find all querylets involving the given tables."""
        if not self._templates:
            self.generate_all()
        return [q for q in self._templates.values()
                if set(q.tables) & tables]


# ── Workload generation from querylets ─────────────────────────────────
@dataclass
class WorkloadQuery:
    """A single query instance — a querylet with bound parameters."""
    query_id: str
    querylet: Querylet
    parameters: Dict[str, Any] = field(default_factory=dict)
    table_name: str = ""   # explicit logical table (INV-3 compliance)

    def describe(self) -> str:
        return (f"WorkloadQuery({self.query_id}, "
                f"template={self.querylet.template_id}, "
                f"tables={self.querylet.tables})")


def workload_from_querylets(
    templates: Dict[str, Querylet],
    n_queries: int = 2000,
    seed: int = 42,
    debug: bool = True,
) -> List[WorkloadQuery]:
    """Generate a benchmark workload from querylet templates.

    PAR2QO: not directly present; workload was generated externally.
    Lynceus: produces WorkloadQuery instances for benchmark.main().
    """
    import random
    rng = random.Random(seed)
    template_list = list(templates.values())
    workload = []

    if debug:
        print(f"\n  ┌─ Workload generation: {n_queries} queries ──────────────")

    for i in range(n_queries):
        qlet = rng.choice(template_list)
        # Assign a TPC-H table for INV-3 compliance (explicit table identity)
        table = qlet.tables[0] if qlet.tables else "unknown"
        wq = WorkloadQuery(
            query_id=f"q_{i:05d}",
            querylet=qlet,
            table_name=table,
        )
        workload.append(wq)

    if debug:
        # Distribution summary
        from collections import Counter
        dist = Counter(wq.querylet.template_id for wq in workload)
        print(f"  │  {len(workload)} queries across {len(dist)} templates")
        for tid, cnt in dist.most_common(5):
            print(f"  │    {tid}: {cnt} queries ({cnt/len(workload)*100:.1f}%)")
        print(f"  └────────────────────────────────────────────────────")

    return workload


# ── Debug utilities ────────────────────────────────────────────────────
def debug_dump_querylet(qlet: Querylet):
    """Print full querylet structure for debugging."""
    print(f"\n  ┌─ QUERYLET: {qlet.template_id} ──────────────────────")
    print(f"  │  tables: {qlet.tables}")
    print(f"  │  aliases: {qlet.aliases}")
    print(f"  │  joins ({len(qlet.joins)}):")
    for la, ra, lc, rc in qlet.joins:
        print(f"  │    {la}.{lc} = {ra}.{rc}")
    print(f"  │  predicates ({len(qlet.predicates)}):")
    for p in qlet.predicates:
        print(f"  │    {p}")
    print(f"  │  device_hint: {qlet.device_hint or 'auto'}")
    print(f"  │  fingerprint: {qlet.fingerprint()}")
    print(f"  │  SQL:\n{qlet.to_sql()}")
    print(f"  └────────────────────────────────────────────────────")
