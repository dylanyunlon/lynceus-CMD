"""
Ported from upstream/par2qo/code/db_random.py (491 lines)
M143: Random database instance generation with quasi-random sampling.

Modifications (~20% algorithm delta):
  - Halton quasi-random sequence for deterministic low-discrepancy seeds
    (replaces simple integer seed list)
  - Reservoir sampling fallback for bounded-memory streaming
  - Welford accumulator for row-count running statistics
  - Consistent-hash shard assignment per generated table
  - EMA-smoothed progress tracking across instance generation
"""

import math
import hashlib
import time
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Halton quasi-random sequence (replaces random_seeds = [1..10])
# ---------------------------------------------------------------------------
def _halton_seq(index, base):
    """Return the index-th element of the Halton sequence in given base."""
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i = i // base
        f /= base
    return result


def generate_halton_seeds(n, base=2, scale=1000000):
    """Generate n quasi-random seeds via Halton sequence.

    Produces more uniformly distributed seeds than a simple range(1, n+1).
    """
    seeds = []
    for i in range(1, n + 1):
        h = _halton_seq(i, base)
        seed = int(h * scale) + 1  # ensure >= 1
        seeds.append(seed)
    return seeds


def _dbg_halton_seeds(n, base=2):
    seeds = generate_halton_seeds(n, base)
    print(f"[halton_seeds._dbg] n={n} base={base} seeds={seeds}")
    # Check uniformity: coefficient of variation of gaps
    sorted_s = sorted(seeds)
    gaps = [sorted_s[i + 1] - sorted_s[i] for i in range(len(sorted_s) - 1)]
    if gaps:
        mean_gap = sum(gaps) / len(gaps)
        var_gap = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
        cv = math.sqrt(var_gap) / mean_gap if mean_gap > 0 else 0
        print(f"  gap_mean={mean_gap:.2f} gap_cv={cv:.4f} (lower=more uniform)")
    return seeds


# Original seeds (kept for backward compat)
random_seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Quasi-random replacement
halton_seeds = generate_halton_seeds(10, base=2)


# ---------------------------------------------------------------------------
# Welford online accumulator for row counts
# ---------------------------------------------------------------------------
class WelfordRowCounter:
    """Running mean/variance of table row counts."""

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
        print(f"[WelfordRowCounter._dbg] count={self.count} mean={self.mean:.2f} "
              f"var={self.variance():.2f} std={self.stddev():.2f}")


# ---------------------------------------------------------------------------
# Reservoir sampling (bounded-memory random subset)
# ---------------------------------------------------------------------------
class ReservoirSampler:
    """Reservoir sampling — maintain k random items from a stream."""

    def __init__(self, k):
        self.k = k
        self.reservoir = []
        self.n_seen = 0

    def add(self, item):
        self.n_seen += 1
        if len(self.reservoir) < self.k:
            self.reservoir.append(item)
        else:
            # Replace with probability k/n_seen using deterministic hash
            h = int(hashlib.md5(f"{item}:{self.n_seen}".encode()).hexdigest()[:8], 16)
            j = h % self.n_seen
            if j < self.k:
                self.reservoir[j] = item

    def sample(self):
        return list(self.reservoir)

    def _dbg(self):
        print(f"[ReservoirSampler._dbg] k={self.k} n_seen={self.n_seen} "
              f"reservoir_size={len(self.reservoir)} "
              f"sample={self.reservoir[:5]}{'...' if len(self.reservoir) > 5 else ''}")


# ---------------------------------------------------------------------------
# EMA progress tracker
# ---------------------------------------------------------------------------
class EMAProgressTracker:
    """Track table creation progress with EMA-smoothed rate."""

    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.ema_rate = 0.0
        self.last_time = None
        self.total = 0
        self.initialised = False

    def tick(self):
        now = time.time()
        self.total += 1
        if self.last_time is not None:
            dt = now - self.last_time
            rate = 1.0 / dt if dt > 0 else 0.0
            if not self.initialised:
                self.ema_rate = rate
                self.initialised = True
            else:
                self.ema_rate = self.alpha * rate + (1 - self.alpha) * self.ema_rate
        self.last_time = now

    def _dbg(self):
        print(f"[EMAProgressTracker._dbg] total={self.total} "
              f"ema_rate={self.ema_rate:.2f} tables/sec")


# ---------------------------------------------------------------------------
# Consistent-hash shard assignment
# ---------------------------------------------------------------------------
def _shard_hash(table_name, shard_id):
    h = hashlib.sha256(f"{table_name}||{shard_id}".encode()).hexdigest()
    return int(h[:16], 16)


def assign_shard(table_name, n_shards):
    """Assign table to shard via rendezvous hashing."""
    best_shard, best_w = 0, -1
    for s in range(n_shards):
        w = _shard_hash(table_name, s)
        if w > best_w:
            best_w = w
            best_shard = s
    return best_shard


def _dbg_assign_shard(table_name, n_shards):
    shard = assign_shard(table_name, n_shards)
    print(f"[assign_shard._dbg] table={table_name} n_shards={n_shards} -> shard={shard}")
    return shard


# ---------------------------------------------------------------------------
# Simulated DB config
# ---------------------------------------------------------------------------
db_config = {
    'dbname': 'imdbloadbase',
    'user': 'hx68',
    'host': '/tmp',
    'port': '5432'
}


def get_count_simulated(table_name, cache=None):
    """Simulated row count from deterministic hash."""
    if cache and table_name in cache:
        return cache[table_name]
    h = int(hashlib.md5(table_name.encode()).hexdigest()[:8], 16)
    return h % 500000


# ---------------------------------------------------------------------------
# Table creation DAG (same structure as db_random upstream)
# ---------------------------------------------------------------------------
def build_random_table_list(i):
    """Return ordered list of table names for random instance i."""
    tables_to_sample_by_title = [
        "movie_companies", "movie_keyword", "cast_info", "movie_link",
        "movie_info", "complete_cast", "aka_title", "movie_info_idx"
    ]
    tables = [f"random_title_{i}"]
    for t in tables_to_sample_by_title:
        tables.append(f"random_{t}_{i}")
    # Second-level dependencies
    tables.append(f"random_keyword_{i}")
    tables.append(f"random_company_name_{i}")
    tables.append(f"random_aka_name_{i}")
    tables.append(f"random_name_{i}")
    tables.append(f"random_person_info_{i}")
    tables.append(f"random_link_type_{i}")
    tables.append(f"random_info_type_{i}")
    tables.append(f"random_company_type_{i}")
    tables.append(f"random_kind_type_{i}")
    tables.append(f"random_char_name_{i}")
    tables.append(f"random_role_type_{i}")
    tables.append(f"random_comp_cast_type_{i}")
    return tables


# ---------------------------------------------------------------------------
# Core random instance generation (simulated)
# ---------------------------------------------------------------------------
def verify_by_multiple_random_instances_simulated(i, window_size, use_halton=True):
    """Simulate random instance generation with Halton seeds.

    Uses quasi-random seeds instead of simple integers, tracks row
    counts via Welford, and assigns tables to shards.
    """
    seeds = halton_seeds if use_halton else random_seeds
    seed = seeds[i] if i < len(seeds) else i + 1

    tables = build_random_table_list(i)
    cache = OrderedDict()
    wf = WelfordRowCounter()
    tracker = EMAProgressTracker(alpha=0.25)
    reservoir = ReservoirSampler(k=5)

    raw_size_title = 2528312

    for table_name in tables:
        # Simulate CREATE TABLE with Halton-derived sampling
        count = get_count_simulated(table_name)
        # Apply quasi-random scaling based on seed
        scaled_count = int(count * (seed / 1000000.0) * (window_size / 100.0))
        cache[table_name] = scaled_count
        wf.update(float(scaled_count))
        tracker.tick()
        reservoir.add(table_name)

    # Shard assignment
    n_shards = max(1, len(tables) // 5)
    shard_map = {}
    for t in tables:
        shard_map[t] = assign_shard(t, n_shards)

    return {
        'seed': seed,
        'tables': tables,
        'cache': cache,
        'welford': wf,
        'tracker': tracker,
        'reservoir': reservoir,
        'shard_map': shard_map,
    }


def _dbg_random_instances(i, window_size=20, use_halton=True):
    result = verify_by_multiple_random_instances_simulated(i, window_size, use_halton)
    print(f"[random_instances._dbg] i={i} seed={result['seed']} "
          f"n_tables={len(result['tables'])}")
    result['welford']._dbg()
    result['tracker']._dbg()
    result['reservoir']._dbg()
    # Show first 3 shard assignments
    for k, v in list(result['shard_map'].items())[:3]:
        print(f"  {k} -> shard {v}")
    return result


# ---------------------------------------------------------------------------
# Drop + index/FK generation (same structure as upstream)
# ---------------------------------------------------------------------------
def drop_sampled_tables_simulated(i):
    """Simulate dropping all random-i tables."""
    table_prefixes = [
        "random_title_base",
        "random_movie_companies_base",
        "random_movie_keyword_base",
        "random_cast_info_base",
        "random_movie_link_base",
        "random_movie_info_base",
        "random_complete_cast_base",
        "random_aka_title_base",
        "random_movie_info_idx_base",
        "random_keyword_base",
        "random_company_name_base",
        "random_aka_name_base",
        "random_name_base",
        "random_person_info_base",
        "random_link_type_base",
        "random_info_type_base",
        "random_company_type_base",
        "random_kind_type_base",
        "random_char_name_base",
        "random_role_type_base",
        "random_comp_cast_type_base"
    ]
    stmts = []
    for prefix in table_prefixes:
        table_name = prefix.replace("_base", f"_{i}")
        stmts.append(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
    return stmts


def generate_index_ddl(i):
    """Generate index DDL for random instance i."""
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
    index_query = index_query.replace(' on ', f'_random_{i} on random_')
    return index_query


def generate_fk_ddl(i):
    """Generate FK DDL for random instance i."""
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
    fk_query = fk_query.replace('TABLE ', 'TABLE random_')
    fk_query = fk_query.replace('REFERENCES ', 'REFERENCES random_')
    return fk_query


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # TODO: change range
    for s in range(0, 9):
        print(s)
        # drop_sampled_tables(s)

    for s in range(0, 9):
        print(s)
        # verify_by_multiple_random_instances(s, 20)


if __name__ == "__main__":
    print("=" * 72)
    print("M143: par2qo_db_random experiment run")
    print("=" * 72)

    # 1. Halton seeds demo
    print("\n--- Halton Quasi-Random Seeds ---")
    _dbg_halton_seeds(10, base=2)
    print(f"Original seeds: {random_seeds}")

    # 2. Simulate random instance generation
    print("\n--- Random Instance i=0, window=20 ---")
    _dbg_random_instances(0, window_size=20, use_halton=True)

    # 3. Compare Halton vs original for instance i=3
    print("\n--- Compare: Halton vs Original (i=3) ---")
    r_halton = verify_by_multiple_random_instances_simulated(3, 20, use_halton=True)
    r_orig = verify_by_multiple_random_instances_simulated(3, 20, use_halton=False)
    print(f"  Halton seed={r_halton['seed']}, mean_count={r_halton['welford'].mean:.2f}")
    print(f"  Original seed={r_orig['seed']}, mean_count={r_orig['welford'].mean:.2f}")

    # 4. Reservoir sampler demo
    print("\n--- Reservoir Sampler (k=3) ---")
    rs = ReservoirSampler(k=3)
    for item in [f"table_{i}" for i in range(20)]:
        rs.add(item)
    rs._dbg()

    # 5. Drop simulation
    print("\n--- Drop Simulation (i=5) ---")
    drops = drop_sampled_tables_simulated(5)
    for d in drops[:5]:
        print(f"  {d}")
    print(f"  ... total {len(drops)} DROP statements")

    # 6. Multi-instance pipeline
    print("\n--- Multi-Instance Pipeline (9 instances) ---")
    global_wf = WelfordRowCounter()
    for s in range(0, 9):
        result = verify_by_multiple_random_instances_simulated(s, 20)
        for cnt in result['cache'].values():
            global_wf.update(float(cnt))
    print(f"Global row-count stats across 9 instances:")
    global_wf._dbg()

    print("\nM143 complete.")
