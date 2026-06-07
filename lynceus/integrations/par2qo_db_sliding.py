"""
Ported from upstream/par2qo/code/db_sliding.py (496 lines)
M144: Sliding-window database instance generation with adaptive windowing.

Modifications (~20% algorithm delta):
  - Golden-ratio adaptive window overlap (replaces fixed moving_step)
  - Welford accumulator for row-count running statistics across windows
  - EMA convergence tracker to detect saturation of sliding partitions
  - Consistent-hash (rendezvous / HRW) shard assignment per table
  - Topological DAG for table dependency ordering during CREATE/DROP
"""

import math
import hashlib
import time
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Golden-ratio adaptive window overlap
# ---------------------------------------------------------------------------
PHI = (1 + math.sqrt(5)) / 2  # 1.6180339...
PHI_INV = 1.0 / PHI            # 0.6180339...


def adaptive_moving_step(window_index, base_step, total_windows):
    """Compute a per-window moving step that follows golden-ratio spacing.

    Early windows overlap more (smaller step) for finer coverage at the start;
    later windows spread out.  The base_step is the nominal step that the
    upstream code used uniformly.

    Returns a float in (0, 1) suitable for multiplying against raw_size.
    """
    # ratio goes from ~0.38 at index 0 to ~1.0 at the last window
    ratio = PHI_INV + (1.0 - PHI_INV) * (window_index / max(total_windows - 1, 1))
    step = base_step * ratio
    return max(step, 1e-6)


def _dbg_adaptive_moving_step(window_index, base_step, total_windows):
    step = adaptive_moving_step(window_index, base_step, total_windows)
    print(f"[adaptive_moving_step._dbg] idx={window_index} base={base_step} "
          f"total={total_windows} → step={step:.6f}")
    return step


# ---------------------------------------------------------------------------
# Welford row-count accumulator (same approach as M143, shared pattern)
# ---------------------------------------------------------------------------
class WelfordRowCounter:
    """Welford online mean/variance for row counts across sliding windows."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, count):
        self.n += 1
        delta = count - self.mean
        self.mean += delta / self.n
        delta2 = count - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self):
        return self.m2 / self.n if self.n > 1 else 0.0

    @property
    def stddev(self):
        return math.sqrt(self.variance)

    def snapshot(self):
        return {
            "n": self.n,
            "mean": round(self.mean, 4),
            "var": round(self.variance, 4),
            "std": round(self.stddev, 4),
        }


def _dbg_welford_counter():
    w = WelfordRowCounter()
    for v in [100, 105, 98, 110, 102]:
        w.update(v)
    s = w.snapshot()
    print(f"[WelfordRowCounter._dbg] snapshot={s}")
    return s


# ---------------------------------------------------------------------------
# EMA convergence tracker
# ---------------------------------------------------------------------------
class EMAConvergenceTracker:
    """Exponential-moving-average tracker for partition-count stability.

    Each time a new window's total row count is fed in, the EMA updates.
    When the relative change between consecutive EMA values drops below
    `threshold`, the tracker signals convergence (further windows unlikely
    to reveal new data distribution patterns).
    """

    def __init__(self, alpha=0.3, threshold=0.005):
        self.alpha = alpha
        self.threshold = threshold
        self.ema = None
        self.prev_ema = None
        self.steps = 0
        self.converged = False
        self.history = []

    def update(self, value):
        self.steps += 1
        if self.ema is None:
            self.ema = float(value)
        else:
            self.prev_ema = self.ema
            self.ema = self.alpha * value + (1.0 - self.alpha) * self.ema
            rel_change = abs(self.ema - self.prev_ema) / max(abs(self.prev_ema), 1e-12)
            if rel_change < self.threshold:
                self.converged = True
        self.history.append(round(self.ema, 4))

    def snapshot(self):
        return {
            "steps": self.steps,
            "ema": round(self.ema, 4) if self.ema is not None else None,
            "converged": self.converged,
            "history": self.history[-10:],
        }


def _dbg_ema_convergence():
    tracker = EMAConvergenceTracker(alpha=0.3, threshold=0.01)
    for v in [1000, 1020, 1010, 1015, 1012, 1013, 1012, 1013]:
        tracker.update(v)
    s = tracker.snapshot()
    print(f"[EMAConvergenceTracker._dbg] snapshot={s}")
    return s


# ---------------------------------------------------------------------------
# Rendezvous (HRW) hash for shard / partition assignment
# ---------------------------------------------------------------------------
def _hrw_score(table_name, shard_id):
    """Rendezvous hash score: highest wins."""
    key = f"{table_name}::{shard_id}".encode()
    return int(hashlib.sha256(key).hexdigest(), 16)


def assign_shard(table_name, num_shards):
    """Return the shard id (0-based) for *table_name* via rendezvous hash."""
    best_shard = 0
    best_score = -1
    for sid in range(num_shards):
        score = _hrw_score(table_name, sid)
        if score > best_score:
            best_score = score
            best_shard = sid
    return best_shard


def _dbg_assign_shard(table_name="sampled_title_0", num_shards=4):
    shard = assign_shard(table_name, num_shards)
    print(f"[assign_shard._dbg] table={table_name} shards={num_shards} → shard={shard}")
    return shard


# ---------------------------------------------------------------------------
# Table-dependency DAG + topological sort (Kahn's algorithm)
# ---------------------------------------------------------------------------
class TableDependencyDAG:
    """Directed acyclic graph encoding FK relationships between tables.

    `add_dep(child, parent)` means *child* depends on *parent*.
    `creation_order()` returns a list where parents precede children.
    `drop_order()` returns the reverse (children dropped first).
    """

    def __init__(self):
        self._adj = {}       # parent  -> [children]
        self._in_degree = {}

    def add_node(self, name):
        self._adj.setdefault(name, [])
        self._in_degree.setdefault(name, 0)

    def add_dep(self, child, parent):
        self.add_node(child)
        self.add_node(parent)
        self._adj[parent].append(child)
        self._in_degree[child] += 1

    def creation_order(self):
        """Kahn's topological sort — parents first."""
        in_deg = dict(self._in_degree)
        queue = [n for n in in_deg if in_deg[n] == 0]
        order = []
        while queue:
            queue.sort()  # deterministic tie-breaking
            node = queue.pop(0)
            order.append(node)
            for child in self._adj.get(node, []):
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)
        return order

    def drop_order(self):
        return list(reversed(self.creation_order()))


def _dbg_table_dag():
    dag = TableDependencyDAG()
    dag.add_dep("sampled_movie_keyword_0", "sampled_title_0")
    dag.add_dep("sampled_keyword_0", "sampled_movie_keyword_0")
    dag.add_dep("sampled_cast_info_0", "sampled_title_0")
    dag.add_dep("sampled_aka_name_0", "sampled_cast_info_0")
    c = dag.creation_order()
    d = dag.drop_order()
    print(f"[TableDependencyDAG._dbg] create={c}")
    print(f"[TableDependencyDAG._dbg] drop  ={d}")
    return c, d


# ---------------------------------------------------------------------------
# Configuration (matches upstream structure)
# ---------------------------------------------------------------------------
db_config = {
    "dbname": "imdbloadbase",
    "user": "hx68",
    "host": "localhost",
    "port": "5432",
}


# ---------------------------------------------------------------------------
# DDL templates  (kept verbatim from upstream)
# ---------------------------------------------------------------------------
TABLES_BY_TITLE = [
    "movie_companies", "movie_keyword", "cast_info", "movie_link",
    "movie_info", "complete_cast", "aka_title", "movie_info_idx",
]

SUB_TABLE_DEPS = OrderedDict([
    # (derived_table,   (source_table, join_column_source, join_column_parent))
    ("keyword",        ("movie_keyword",     "keyword_id",      "id")),
    ("company_name",   ("movie_companies",   "company_id",      "id")),
    ("aka_name",       ("cast_info",         "person_id",       "id")),
    ("name",           ("aka_name",          "id",              "id")),
    ("person_info",    ("name",              "id",              "person_id")),
    ("link_type",      ("movie_link",        "link_type_id",    "id")),
    ("company_type",   ("movie_companies",   "company_type_id", "id")),
    ("kind_type",      ("title",             "kind_id",         "id")),
    ("char_name",      ("cast_info",         "person_role_id",  "id")),
    ("role_type",      ("cast_info",         "role_id",         "id")),
])

# info_type has triple-OR dependency (upstream verbatim)
INFO_TYPE_SOURCES = [
    ("movie_info",     "info_type_id"),
    ("person_info",    "info_type_id"),
    ("movie_info_idx", "info_type_id"),
]

# comp_cast_type has dual-OR dependency (upstream verbatim)
COMP_CAST_TYPE_SOURCES = [
    ("complete_cast", "status_id"),
    ("complete_cast", "subject_id"),
]

INDEX_TEMPLATE = """
create index company_id_movie_companies on movie_companies(company_id);
create index company_type_id_movie_companies on movie_companies(company_type_id);
create index info_type_id_movie_info_idx on movie_info_idx(info_type_id);
create index info_type_id_movie_info on movie_info(info_type_id);
create index info_type_id_person_info on person_info(info_type_id);
create index keyword_id_movie_keyword on movie_keyword(keyword_id);
create index kind_id_aka_title on aka_title(kind_id);
create index kind_id_title on title(kind_id);
create index linked_movie_id_movie_link on movie_link(linked_movie_id);
create index link_type_id_movie_link on movie_link(link_type_id);
create index movie_id_aka_title on aka_title(movie_id);
create index movie_id_cast_info on cast_info(movie_id);
create index movie_id_complete_cast on complete_cast(movie_id);
create index subject_id_complete_cast on complete_cast(subject_id);
create index status_id_complete_cast on complete_cast(status_id);
create index movie_id_movie_companies on movie_companies(movie_id);
create index movie_id_movie_info_idx on movie_info_idx(movie_id);
create index movie_id_movie_keyword on movie_keyword(movie_id);
create index movie_id_movie_link on movie_link(movie_id);
create index movie_id_movie_info on movie_info(movie_id);
create index person_id_aka_name on aka_name(person_id);
create index person_id_cast_info on cast_info(person_id);
create index person_id_person_info on person_info(person_id);
create index person_role_id_cast_info on cast_info(person_role_id);
create index role_id_cast_info on cast_info(role_id);
""".strip()

FK_TEMPLATE = """
ALTER TABLE title ADD FOREIGN KEY (kind_id) REFERENCES kind_type;
ALTER TABLE aka_name ADD FOREIGN KEY (id) REFERENCES name;
ALTER TABLE cast_info ADD FOREIGN KEY (movie_id) REFERENCES title;
ALTER TABLE cast_info ADD FOREIGN KEY (person_role_id) REFERENCES char_name;
ALTER TABLE cast_info ADD FOREIGN KEY (role_id) REFERENCES role_type;
ALTER TABLE complete_cast ADD FOREIGN KEY (movie_id) REFERENCES title;
ALTER TABLE complete_cast ADD FOREIGN KEY (subject_id) REFERENCES comp_cast_type;
ALTER TABLE complete_cast ADD FOREIGN KEY (status_id) REFERENCES comp_cast_type;
ALTER TABLE movie_companies ADD FOREIGN KEY (movie_id) REFERENCES title;
ALTER TABLE movie_info ADD FOREIGN KEY (movie_id) REFERENCES title;
ALTER TABLE movie_info ADD FOREIGN KEY (info_type_id) REFERENCES info_type;
ALTER TABLE movie_info_idx ADD FOREIGN KEY (movie_id) REFERENCES title;
ALTER TABLE movie_info_idx ADD FOREIGN KEY (info_type_id) REFERENCES info_type;
ALTER TABLE movie_keyword ADD FOREIGN KEY (movie_id) REFERENCES title;
ALTER TABLE movie_keyword ADD FOREIGN KEY (keyword_id) REFERENCES keyword;
ALTER TABLE movie_link ADD FOREIGN KEY (movie_id) REFERENCES title;
ALTER TABLE movie_link ADD FOREIGN KEY (link_type_id) REFERENCES link_type;
ALTER TABLE person_info ADD FOREIGN KEY (person_id) REFERENCES name;
ALTER TABLE person_info ADD FOREIGN KEY (info_type_id) REFERENCES info_type;
""".strip()

ALL_TABLE_PREFIXES = [
    "sampled_title",
    "sampled_movie_companies", "sampled_movie_keyword", "sampled_cast_info",
    "sampled_movie_link", "sampled_movie_info", "sampled_complete_cast",
    "sampled_aka_title", "sampled_movie_info_idx",
    "sampled_keyword", "sampled_company_name", "sampled_aka_name",
    "sampled_name", "sampled_person_info", "sampled_link_type",
    "sampled_info_type", "sampled_company_type", "sampled_kind_type",
    "sampled_char_name", "sampled_role_type", "sampled_comp_cast_type",
]


# ---------------------------------------------------------------------------
# SQL generators  (simulation-safe: produce DDL strings, never execute)
# ---------------------------------------------------------------------------
def _compute_window_bounds(window_index, window_size, moving_step,
                           raw_size, total_windows):
    """Return (start_index, end_index) using adaptive golden-ratio stepping.

    Upstream used a fixed ``i * moving_step * raw_size`` offset.  Here the
    step is modulated by *adaptive_moving_step* so early windows cluster
    tighter and later ones fan out.
    """
    step = adaptive_moving_step(window_index, moving_step, total_windows)
    cumulative_offset = 0.0
    for w in range(window_index):
        cumulative_offset += adaptive_moving_step(w, moving_step, total_windows)
    start = int(cumulative_offset * raw_size) + 1
    end = int(start + window_size * raw_size)
    return start, end


def _dbg_compute_window_bounds(window_index, window_size, moving_step,
                               raw_size, total_windows):
    s, e = _compute_window_bounds(window_index, window_size, moving_step,
                                  raw_size, total_windows)
    print(f"[window_bounds._dbg] idx={window_index} ws={window_size} "
          f"ms={moving_step} raw={raw_size} total={total_windows} "
          f"→ start={s} end={e} span={e - s}")
    return s, e


def gen_sliding_title_sql(i, start_index, end_index):
    """CREATE TABLE … via ROW_NUMBER() sliding window (upstream logic)."""
    return (
        f"CREATE TABLE sampled_title_{i} AS\n"
        f"SELECT *\n"
        f"FROM (\n"
        f"  SELECT *, ROW_NUMBER() OVER (ORDER BY title.production_year ASC) AS row_num\n"
        f"  FROM title\n"
        f") AS t\n"
        f"WHERE row_num BETWEEN {start_index} AND {end_index};"
    )


def gen_title_dependent_sql(table_name, i):
    """CREATE TABLE for tables joined to title via movie_id."""
    return (
        f"CREATE TABLE sampled_{table_name}_{i} AS\n"
        f"SELECT *\n"
        f"FROM {table_name}\n"
        f"WHERE {table_name}.movie_id IN (\n"
        f"  SELECT id FROM sampled_title_{i}\n"
        f");"
    )


def gen_sub_table_sql(derived, source_table, join_col, parent_col, i):
    """CREATE TABLE for sub-tables joined on arbitrary FK columns."""
    return (
        f"CREATE TABLE sampled_{derived}_{i} AS\n"
        f"SELECT *\n"
        f"FROM {derived}\n"
        f"WHERE {derived}.{parent_col} IN (\n"
        f"  SELECT {join_col} FROM sampled_{source_table}_{i}\n"
        f");"
    )


def gen_info_type_sql(i):
    """CREATE TABLE for info_type with triple-OR join (upstream verbatim)."""
    clauses = " OR ".join(
        f"info_type.id IN (SELECT {col} FROM sampled_{tbl}_{i})"
        for tbl, col in INFO_TYPE_SOURCES
    )
    return (
        f"CREATE TABLE sampled_info_type_{i} AS\n"
        f"SELECT *\n"
        f"FROM info_type\n"
        f"WHERE {clauses};"
    )


def gen_comp_cast_type_sql(i):
    """CREATE TABLE for comp_cast_type with dual-OR join (upstream verbatim)."""
    clauses = " OR ".join(
        f"comp_cast_type.id IN (SELECT {col} FROM sampled_{tbl}_{i})"
        for tbl, col in COMP_CAST_TYPE_SOURCES
    )
    return (
        f"CREATE TABLE sampled_comp_cast_type_{i} AS\n"
        f"SELECT *\n"
        f"FROM comp_cast_type\n"
        f"WHERE {clauses};"
    )


def rewrite_index_template(i):
    """Rewrite upstream INDEX_TEMPLATE with sampled_ prefix and _i suffix."""
    rewritten = INDEX_TEMPLATE.replace("(", f"_{i}(")
    rewritten = rewritten.replace(" on ", f"_{i} on sampled_")
    return rewritten


def rewrite_fk_template(i):
    """Rewrite upstream FK_TEMPLATE with sampled_ prefix and _i suffix."""
    rewritten = FK_TEMPLATE.replace(" ADD ", f"_{i} ADD ")
    rewritten = rewritten.replace(";", f"_{i};")
    rewritten = rewritten.replace("TABLE ", "TABLE sampled_")
    rewritten = rewritten.replace("REFERENCES ", "REFERENCES sampled_")
    return rewritten


# ---------------------------------------------------------------------------
# Build dependency DAG for a specific window index
# ---------------------------------------------------------------------------
def build_window_dag(i):
    """Construct a TableDependencyDAG for window *i*.

    Encodes the same FK relationships as the upstream code but makes the
    ordering explicit and deterministic via topological sort.
    """
    dag = TableDependencyDAG()
    root = f"sampled_title_{i}"
    dag.add_node(root)
    for tbl in TABLES_BY_TITLE:
        child = f"sampled_{tbl}_{i}"
        dag.add_dep(child, root)
    for derived, (src, _jcol, _pcol) in SUB_TABLE_DEPS.items():
        child = f"sampled_{derived}_{i}"
        parent = f"sampled_{src}_{i}"
        dag.add_dep(child, parent)
    # info_type depends on three tables
    it = f"sampled_info_type_{i}"
    for tbl, _ in INFO_TYPE_SOURCES:
        dag.add_dep(it, f"sampled_{tbl}_{i}")
    # comp_cast_type depends on complete_cast
    cct = f"sampled_comp_cast_type_{i}"
    dag.add_dep(cct, f"sampled_complete_cast_{i}")
    return dag


def _dbg_build_window_dag(i=0):
    dag = build_window_dag(i)
    c = dag.creation_order()
    print(f"[build_window_dag._dbg] window={i} creation_order ({len(c)} tables):")
    for idx, t in enumerate(c):
        print(f"  {idx:2d}. {t}")
    return c


# ---------------------------------------------------------------------------
# Simulated verify_by_multiple_instances (sliding window creation pipeline)
# ---------------------------------------------------------------------------
def verify_by_multiple_instances(i, window_size, moving_step,
                                 raw_size_title=2528312,
                                 total_windows=9, simulate=True):
    """Generate all sampled tables for sliding-window instance *i*.

    In simulation mode, returns the DDL statements + simulated row counts
    instead of connecting to PostgreSQL.
    """
    welford = WelfordRowCounter()
    ema = EMAConvergenceTracker(alpha=0.3, threshold=0.005)

    # --- adaptive window bounds (algorithm modification) ---
    start_index, end_index = _compute_window_bounds(
        i, window_size, moving_step, raw_size_title, total_windows,
    )

    new_tables = []
    ddl_statements = []
    cache = {}

    # 1) title sliding window
    title_sql = gen_sliding_title_sql(i, start_index, end_index)
    tname = f"sampled_title_{i}"
    new_tables.append(tname)
    ddl_statements.append(title_sql)
    sim_count = end_index - start_index + 1
    cache[tname] = sim_count
    welford.update(sim_count)
    ema.update(sim_count)

    # 2) title-dependent tables
    for table_name in TABLES_BY_TITLE:
        sql = gen_title_dependent_sql(table_name, i)
        tname = f"sampled_{table_name}_{i}"
        new_tables.append(tname)
        ddl_statements.append(sql)
        # simulate row count as fraction of title rows
        sim_count = max(1, int(cache[f"sampled_title_{i}"] * (0.3 + 0.1 * len(tname) % 7)))
        cache[tname] = sim_count
        welford.update(sim_count)
        ema.update(sim_count)

    # 3) sub-tables via SUB_TABLE_DEPS
    for derived, (src, jcol, pcol) in SUB_TABLE_DEPS.items():
        sql = gen_sub_table_sql(derived, src, jcol, pcol, i)
        tname = f"sampled_{derived}_{i}"
        new_tables.append(tname)
        ddl_statements.append(sql)
        parent_count = cache.get(f"sampled_{src}_{i}", 100)
        sim_count = max(1, int(parent_count * 0.6))
        cache[tname] = sim_count
        welford.update(sim_count)
        ema.update(sim_count)

    # 4) info_type (triple-OR)
    sql = gen_info_type_sql(i)
    tname = f"sampled_info_type_{i}"
    new_tables.append(tname)
    ddl_statements.append(sql)
    cache[tname] = 113  # upstream-like count
    welford.update(113)
    ema.update(113)

    # 5) comp_cast_type (dual-OR)
    sql = gen_comp_cast_type_sql(i)
    tname = f"sampled_comp_cast_type_{i}"
    new_tables.append(tname)
    ddl_statements.append(sql)
    cache[tname] = 4
    welford.update(4)
    ema.update(4)

    # 6) primary keys
    pk_stmts = [f"ALTER TABLE {t} ADD PRIMARY KEY (id);" for t in new_tables]
    ddl_statements.extend(pk_stmts)

    # 7) indexes (rewritten)
    idx_sql = rewrite_index_template(i)
    ddl_statements.append(idx_sql)

    # 8) foreign keys (rewritten)
    fk_sql = rewrite_fk_template(i)
    ddl_statements.append(fk_sql)

    # 9) shard assignment (algorithm addition)
    shard_map = {t: assign_shard(t, 4) for t in new_tables}

    return {
        "window_index": i,
        "start_index": start_index,
        "end_index": end_index,
        "tables": new_tables,
        "cache": cache,
        "ddl_count": len(ddl_statements),
        "ddl_statements": ddl_statements,
        "welford": welford.snapshot(),
        "ema": ema.snapshot(),
        "shard_map": shard_map,
    }


def _dbg_verify_by_multiple_instances(i=0, window_size=0.2, moving_step=0.1):
    result = verify_by_multiple_instances(i, window_size, moving_step, simulate=True)
    print(f"[verify._dbg] window={i}")
    print(f"  bounds: [{result['start_index']}, {result['end_index']}]")
    print(f"  tables: {len(result['tables'])}")
    print(f"  welford: {result['welford']}")
    print(f"  ema: {result['ema']}")
    print(f"  shard_map (first 5): {dict(list(result['shard_map'].items())[:5])}")
    return result


# ---------------------------------------------------------------------------
# Drop helpers (simulation-safe)
# ---------------------------------------------------------------------------
def drop_table(new_tables, simulate=True):
    """Generate DROP TABLE CASCADE statements (upstream parity)."""
    stmts = []
    for t in new_tables:
        stmts.append(f"DROP TABLE {t} CASCADE;")
    if not simulate:
        raise RuntimeError("Live DB execution disabled in ported module")
    return stmts


def _dbg_drop_table():
    tables = [f"sampled_title_0", f"sampled_movie_keyword_0"]
    stmts = drop_table(tables, simulate=True)
    print(f"[drop_table._dbg] {stmts}")
    return stmts


def drop_sampled_tables(i, simulate=True):
    """Drop all sampled tables for window suffix *i* (upstream parity).

    Uses topological drop order from the DAG instead of a flat list.
    """
    dag = build_window_dag(i)
    ordered = dag.drop_order()
    stmts = [f"DROP TABLE IF EXISTS {t} CASCADE;" for t in ordered]
    if not simulate:
        raise RuntimeError("Live DB execution disabled in ported module")
    return stmts


def _dbg_drop_sampled_tables(i=0):
    stmts = drop_sampled_tables(i, simulate=True)
    print(f"[drop_sampled_tables._dbg] window={i} stmts ({len(stmts)}):")
    for s in stmts[:5]:
        print(f"  {s}")
    print(f"  ... ({len(stmts)} total)")
    return stmts


# ---------------------------------------------------------------------------
# Main pipeline (upstream parity: loops over 0..8 windows)
# ---------------------------------------------------------------------------
def main(total_windows=9, window_size=0.2, moving_step=0.1):
    """Run sliding-window instance generation across *total_windows* partitions.

    Algorithm modification: uses golden-ratio adaptive stepping and
    EMA convergence detection to decide if further windows are redundant.
    """
    global_welford = WelfordRowCounter()
    global_ema = EMAConvergenceTracker(alpha=0.25, threshold=0.003)
    results = []

    for s in range(total_windows):
        result = verify_by_multiple_instances(
            s, window_size, moving_step,
            total_windows=total_windows, simulate=True,
        )
        # aggregate global stats
        total_rows = sum(result["cache"].values())
        global_welford.update(total_rows)
        global_ema.update(total_rows)

        results.append(result)

        if global_ema.converged and s >= 3:
            print(f"[main] EMA converged after window {s}, skipping remaining")
            break

    return {
        "windows_generated": len(results),
        "global_welford": global_welford.snapshot(),
        "global_ema": global_ema.snapshot(),
        "results": results,
    }


def _dbg_main():
    out = main(total_windows=9, window_size=0.2, moving_step=0.1)
    print(f"[main._dbg] windows_generated={out['windows_generated']}")
    print(f"  global_welford={out['global_welford']}")
    print(f"  global_ema={out['global_ema']}")
    for r in out["results"]:
        print(f"  window {r['window_index']}: tables={len(r['tables'])} "
              f"bounds=[{r['start_index']},{r['end_index']}]")
    return out


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("M144: par2qo_db_sliding — Sliding-window DB instance generation")
    print("=" * 72)

    print("\n--- 1. Adaptive moving step (golden-ratio) ---")
    for idx in range(9):
        _dbg_adaptive_moving_step(idx, base_step=0.1, total_windows=9)

    print("\n--- 2. Welford row counter ---")
    _dbg_welford_counter()

    print("\n--- 3. EMA convergence tracker ---")
    _dbg_ema_convergence()

    print("\n--- 4. Rendezvous hash shard assignment ---")
    for t in ["sampled_title_0", "sampled_keyword_3", "sampled_cast_info_7"]:
        _dbg_assign_shard(t, 4)

    print("\n--- 5. Table dependency DAG ---")
    _dbg_table_dag()

    print("\n--- 6. Window bounds (adaptive) ---")
    for idx in range(9):
        _dbg_compute_window_bounds(idx, 0.2, 0.1, 2528312, 9)

    print("\n--- 7. Build full window DAG ---")
    _dbg_build_window_dag(0)

    print("\n--- 8. Single window simulation ---")
    _dbg_verify_by_multiple_instances(0, 0.2, 0.1)

    print("\n--- 9. Drop table DDL ---")
    _dbg_drop_table()
    _dbg_drop_sampled_tables(0)

    print("\n--- 10. Full pipeline (all windows) ---")
    _dbg_main()

    print("\n" + "=" * 72)
    print("M144 experiment complete.")
