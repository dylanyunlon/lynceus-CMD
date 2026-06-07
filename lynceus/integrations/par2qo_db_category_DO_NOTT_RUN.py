"""
Ported from upstream/par2qo/code/db_category_DO_NOTT_RUN.py (468 lines)
M142: Database category partitioning with dependency-aware table creation.

Modifications (~20% algorithm delta):
  - Dependency DAG topological sort for table creation ordering (replaces linear)
  - Consistent hashing (Rendezvous) for category bucket assignment
  - Exponential backoff retry on simulated table creation failures
  - Welford accumulator for running table count statistics
  - Batch DDL emission with parallel-ready chunking
"""

import math
import hashlib
import time
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Welford online accumulator (reused from M141 but self-contained)
# ---------------------------------------------------------------------------
class WelfordCounter:
    """Welford's algorithm for running mean/variance of row counts."""

    __slots__ = ('count', 'mean', 'm2')

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def variance(self):
        return self.m2 / (self.count - 1) if self.count >= 2 else 0.0

    def stddev(self):
        return math.sqrt(self.variance())

    def _dbg(self):
        print(f"[WelfordCounter._dbg] count={self.count} mean={self.mean:.4f} "
              f"var={self.variance():.4f} std={self.stddev():.4f}")


# ---------------------------------------------------------------------------
# Rendezvous (HRW) consistent hashing for category buckets
# ---------------------------------------------------------------------------
def _hrw_hash(key, bucket):
    """Rendezvous / Highest-Random-Weight hash."""
    h = hashlib.sha256(f"{key}:{bucket}".encode()).hexdigest()
    return int(h[:16], 16)


def assign_category_bucket(table_name, n_buckets):
    """Assign a table to a category bucket via rendezvous hashing."""
    best_bucket = 0
    best_weight = -1
    for b in range(n_buckets):
        w = _hrw_hash(table_name, b)
        if w > best_weight:
            best_weight = w
            best_bucket = b
    return best_bucket


def _dbg_assign_bucket(table_name, n_buckets):
    bucket = assign_category_bucket(table_name, n_buckets)
    print(f"[assign_category_bucket._dbg] table={table_name} n_buckets={n_buckets} "
          f"-> bucket={bucket}")
    return bucket


# ---------------------------------------------------------------------------
# Dependency DAG + topological sort
# ---------------------------------------------------------------------------
class TableDependencyDAG:
    """Directed acyclic graph of table creation dependencies.

    Edges: (parent, child) means child depends on parent.
    topological_order() returns tables in safe creation order.
    """

    def __init__(self):
        self._adj = {}       # parent -> [children]
        self._in_degree = {}

    def add_table(self, name):
        if name not in self._adj:
            self._adj[name] = []
            self._in_degree[name] = 0

    def add_dependency(self, parent, child):
        self.add_table(parent)
        self.add_table(child)
        self._adj[parent].append(child)
        self._in_degree[child] += 1

    def topological_order(self):
        """Kahn's algorithm — returns tables in creation-safe order."""
        in_deg = dict(self._in_degree)
        queue = [n for n in in_deg if in_deg[n] == 0]
        order = []
        while queue:
            queue.sort()  # deterministic
            node = queue.pop(0)
            order.append(node)
            for child in self._adj.get(node, []):
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)
        if len(order) != len(self._adj):
            raise RuntimeError("Cycle detected in dependency DAG")
        return order

    def _dbg(self):
        order = self.topological_order()
        print(f"[TableDependencyDAG._dbg] n_tables={len(self._adj)} "
              f"topo_order={order}")
        return order


# ---------------------------------------------------------------------------
# Exponential backoff retry
# ---------------------------------------------------------------------------
def retry_with_backoff(fn, max_retries=5, base_delay=0.05):
    """Execute fn() with exponential backoff on failure."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            delay = base_delay * (2 ** attempt)
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
    return None


def _dbg_retry(fn, max_retries=5, base_delay=0.001):
    t0 = time.time()
    result = retry_with_backoff(fn, max_retries, base_delay)
    elapsed = time.time() - t0
    print(f"[retry_with_backoff._dbg] elapsed={elapsed:.4f}s result={result}")
    return result


# ---------------------------------------------------------------------------
# Batch DDL emission
# ---------------------------------------------------------------------------
def chunk_ddl_statements(statements, chunk_size=4):
    """Split DDL statements into parallel-ready chunks."""
    return [statements[i:i + chunk_size] for i in range(0, len(statements), chunk_size)]


def _dbg_chunk_ddl(statements, chunk_size=4):
    chunks = chunk_ddl_statements(statements, chunk_size)
    print(f"[chunk_ddl._dbg] total={len(statements)} chunk_size={chunk_size} "
          f"n_chunks={len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  chunk[{i}]: {c}")
    return chunks


# ---------------------------------------------------------------------------
# Simulated DB config (no real psycopg2 — simulation mode)
# ---------------------------------------------------------------------------
db_config = {
    'dbname': 'imdbloadbase',
    'user': 'hx68',
    'host': '/tmp',
    'port': '5432'
}


def get_count_simulated(table_name, cache):
    """Simulated row count — returns cached or synthetic value."""
    if table_name in cache:
        return cache[table_name]
    # Deterministic synthetic count based on table name hash
    h = int(hashlib.md5(table_name.encode()).hexdigest()[:8], 16)
    return h % 500000


# ---------------------------------------------------------------------------
# Core category partitioning logic (simulated)
# ---------------------------------------------------------------------------
def build_category_dag(i):
    """Build the dependency DAG for category-i tables.

    Returns (dag, table_list) where table_list is in topological order.
    """
    dag = TableDependencyDAG()

    # Root: title
    root = f"cat_title_{i}"
    dag.add_table(root)

    # First-level dependents (depend on title via movie_id)
    title_dependents = [
        f"cat_movie_companies_{i}",
        f"cat_movie_keyword_{i}",
        f"cat_cast_info_{i}",
        f"cat_movie_link_{i}",
        f"cat_movie_info_{i}",
        f"cat_complete_cast_{i}",
        f"cat_aka_title_{i}",
        f"cat_movie_info_idx_{i}",
    ]
    for dep in title_dependents:
        dag.add_dependency(root, dep)

    # Second-level: keyword depends on movie_keyword
    dag.add_dependency(f"cat_movie_keyword_{i}", f"cat_keyword_{i}")

    # company_name depends on movie_companies
    dag.add_dependency(f"cat_movie_companies_{i}", f"cat_company_name_{i}")

    # aka_name depends on cast_info
    dag.add_dependency(f"cat_cast_info_{i}", f"cat_aka_name_{i}")

    # name depends on aka_name
    dag.add_dependency(f"cat_aka_name_{i}", f"cat_name_{i}")

    # person_info depends on name
    dag.add_dependency(f"cat_name_{i}", f"cat_person_info_{i}")

    # link_type depends on movie_link
    dag.add_dependency(f"cat_movie_link_{i}", f"cat_link_type_{i}")

    # info_type depends on movie_info + person_info + movie_info_idx
    dag.add_dependency(f"cat_movie_info_{i}", f"cat_info_type_{i}")
    dag.add_dependency(f"cat_person_info_{i}", f"cat_info_type_{i}")
    dag.add_dependency(f"cat_movie_info_idx_{i}", f"cat_info_type_{i}")

    # company_type depends on movie_companies
    dag.add_dependency(f"cat_movie_companies_{i}", f"cat_company_type_{i}")

    # kind_type depends on title
    dag.add_dependency(root, f"cat_kind_type_{i}")

    # char_name depends on cast_info
    dag.add_dependency(f"cat_cast_info_{i}", f"cat_char_name_{i}")

    # role_type depends on cast_info
    dag.add_dependency(f"cat_cast_info_{i}", f"cat_role_type_{i}")

    # comp_cast_type depends on complete_cast
    dag.add_dependency(f"cat_complete_cast_{i}", f"cat_comp_cast_type_{i}")

    return dag, dag.topological_order()


def _dbg_build_dag(i):
    dag, order = build_category_dag(i)
    print(f"[build_category_dag._dbg] category={i} n_tables={len(order)}")
    for idx, t in enumerate(order):
        print(f"  [{idx:2d}] {t}")
    return dag, order


def verify_by_movie_category_simulated(i):
    """Simulate the category table creation pipeline.

    Uses DAG topological order instead of linear iteration, consistent
    hashing for bucket assignment, and Welford stats for row counts.
    """
    dag, creation_order = build_category_dag(i)
    cache = {}
    wf = WelfordCounter()
    new_tables = []

    for table_name in creation_order:
        # Simulate CREATE TABLE + count via retry
        def _create():
            cnt = get_count_simulated(table_name, cache)
            return cnt

        count = retry_with_backoff(_create, max_retries=3, base_delay=0.001)
        cache[table_name] = count
        new_tables.append(table_name)
        wf.update(float(count))

    # Assign each table to a hash bucket
    bucket_assignment = {}
    n_buckets = max(1, len(new_tables) // 4)
    for t in new_tables:
        bucket_assignment[t] = assign_category_bucket(t, n_buckets)

    # Generate batch index DDL
    index_stmts = []
    for t in new_tables:
        index_stmts.append(f"CREATE INDEX idx_{t}_id ON {t}(id);")
    batches = chunk_ddl_statements(index_stmts, chunk_size=5)

    return {
        'tables': new_tables,
        'cache': cache,
        'welford': wf,
        'bucket_assignment': bucket_assignment,
        'index_batches': batches,
    }


def _dbg_verify_category(i):
    result = verify_by_movie_category_simulated(i)
    print(f"[verify_category._dbg] category={i} n_tables={len(result['tables'])}")
    result['welford']._dbg()
    print(f"  bucket_assignment (first 5):")
    for k, v in list(result['bucket_assignment'].items())[:5]:
        print(f"    {k} -> bucket {v}")
    print(f"  index_batches: {len(result['index_batches'])} batches")
    return result


def drop_sampled_tables_simulated(i):
    """Simulate dropping all category-i tables in reverse topological order."""
    _, order = build_category_dag(i)
    dropped = []
    for table_name in reversed(order):
        dropped.append(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
    return dropped


def _dbg_drop_tables(i):
    stmts = drop_sampled_tables_simulated(i)
    print(f"[drop_tables._dbg] category={i} n_drops={len(stmts)}")
    for s in stmts[:5]:
        print(f"  {s}")
    return stmts


# ---------------------------------------------------------------------------
# FK / index template generation (from upstream)
# ---------------------------------------------------------------------------
def generate_index_ddl(i):
    """Generate index DDL for category-i tables."""
    index_query = '''create index company_id_movie_companies on movie_companies(company_id);
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
                    create index role_id_cast_info on cast_info(role_id);'''

    index_query = index_query.replace('(', f'_{i}(')
    index_query = index_query.replace(' on ', f'_cat_{i} on cat_')
    return index_query


def generate_fk_ddl(i):
    """Generate FK DDL for category-i tables."""
    fk_query = '''
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
        '''
    fk_query = fk_query.replace(' ADD ', f'_{i} ADD ')
    fk_query = fk_query.replace(';', f'_{i};')
    fk_query = fk_query.replace('TABLE ', 'TABLE cat_')
    fk_query = fk_query.replace('REFERENCES ', 'REFERENCES cat_')
    return fk_query


def _dbg_ddl(i):
    idx_ddl = generate_index_ddl(i)
    fk_ddl = generate_fk_ddl(i)
    idx_lines = [l.strip() for l in idx_ddl.strip().split(';') if l.strip()]
    fk_lines = [l.strip() for l in fk_ddl.strip().split(f'_{i};') if l.strip()]
    print(f"[generate_ddl._dbg] category={i} n_index_stmts={len(idx_lines)} "
          f"n_fk_stmts={len(fk_lines)}")
    print(f"  sample index: {idx_lines[0][:80]}...")
    print(f"  sample fk:    {fk_lines[0][:80]}...")
    return idx_ddl, fk_ddl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # TODO: change range
    for s in [1, 2, 3, 4, 6, 7]:
        print(s)
        # drop_sampled_tables(s)

    for s in [1, 2, 3, 4, 6, 7]:
        print(s)
        # verify_by_movie_category(s)


if __name__ == "__main__":
    print("=" * 72)
    print("M142: par2qo_db_category_DO_NOTT_RUN experiment run")
    print("=" * 72)

    # 1. Build & display dependency DAG
    print("\n--- Dependency DAG (category 1) ---")
    _dbg_build_dag(1)

    # 2. Simulate category table creation
    print("\n--- Simulated Category Verification (category 1) ---")
    _dbg_verify_category(1)

    # 3. Drop simulation
    print("\n--- Simulated Drop (category 2) ---")
    _dbg_drop_tables(2)

    # 4. Consistent hashing demo
    print("\n--- Consistent Hashing ---")
    tables = [f"cat_title_{i}" for i in range(1, 8)]
    for t in tables:
        _dbg_assign_bucket(t, 4)

    # 5. DDL generation
    print("\n--- DDL Generation (category 3) ---")
    _dbg_ddl(3)

    # 6. Retry demo
    print("\n--- Retry with Backoff ---")
    call_count = [0]
    def flaky():
        call_count[0] += 1
        if call_count[0] < 3:
            raise RuntimeError("simulated failure")
        return "success"
    _dbg_retry(flaky, max_retries=5, base_delay=0.001)

    # 7. Batch DDL
    print("\n--- Batch DDL Chunking ---")
    stmts = [f"CREATE INDEX idx_{i} ON t_{i}(col);" for i in range(12)]
    _dbg_chunk_ddl(stmts, chunk_size=4)

    # 8. Multi-category pipeline
    print("\n--- Multi-Category Pipeline ---")
    overall_wf = WelfordCounter()
    for cat in [1, 2, 3, 4, 6, 7]:
        result = verify_by_movie_category_simulated(cat)
        for cnt in result['cache'].values():
            overall_wf.update(float(cnt))
    print(f"Overall row-count stats across all categories:")
    overall_wf._dbg()

    print("\nM142 complete.")
