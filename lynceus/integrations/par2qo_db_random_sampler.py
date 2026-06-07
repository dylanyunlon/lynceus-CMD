"""
Ported from upstream/par2qo/code/db_random.py (490 lines)
M141: Database random sampling simulator with Bernoulli sampling and
      advanced cardinality / variance / smoothing estimators.

Modifications (~25% algorithm delta vs. upstream):
  - Bernoulli sampling simulator with seed-controlled deterministic RNG
    (replaces live PostgreSQL TABLESAMPLE BERNOULLI queries)
  - HyperLogLog cardinality estimator for approximate distinct-count
    tracking across sampled tables
  - Welford online variance accumulator for row-count statistics
    (numerically stable single-pass mean / variance / stddev)
  - EMA (Exponential Moving Average) smoother for rate / progress
    tracking with configurable decay
  - Every public function and class has a companion _dbg() diagnostic
    that prints internal state for debugging

Pure numpy/stdlib implementation — no external DB connection required.
"""

import math
import hashlib
import struct
import time
from collections import OrderedDict

import numpy as np


# ---------------------------------------------------------------------------
# Bernoulli sampling simulator (replaces live TABLESAMPLE BERNOULLI)
# ---------------------------------------------------------------------------
class BernoulliSampler:
    """Deterministic Bernoulli sampling over a simulated row population.

    Each row is accepted with probability *rate* (0..100 interpreted as
    percent, matching PostgreSQL's TABLESAMPLE BERNOULLI syntax).  The
    seed pins the numpy RNG state so that repeated runs produce identical
    samples — the same role REPEATABLE(seed) plays in the upstream SQL.
    """

    __slots__ = ('rate', 'seed', '_rng', 'accepted', 'rejected')

    def __init__(self, rate, seed):
        if not (0.0 <= rate <= 100.0):
            raise ValueError(f"rate must be in [0, 100], got {rate}")
        self.rate = rate
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.accepted = 0
        self.rejected = 0

    def sample(self, population_size):
        """Return the array of accepted row indices from range(population_size).

        Uses numpy vectorised Bernoulli draw for the entire population in one
        call, which is significantly faster than a Python-level per-row loop
        for large populations.
        """
        threshold = self.rate / 100.0
        # Reset RNG so that same seed + same population always gives same result
        self._rng = np.random.default_rng(self.seed)
        draws = self._rng.random(population_size)
        mask = draws < threshold
        result = np.flatnonzero(mask)
        self.accepted += int(result.size)
        self.rejected += int(population_size - result.size)
        return result

    def acceptance_ratio(self):
        total = self.accepted + self.rejected
        return self.accepted / total if total > 0 else 0.0

    def _dbg(self):
        total = self.accepted + self.rejected
        print(
            f"[BernoulliSampler._dbg] rate={self.rate}% seed={self.seed} "
            f"accepted={self.accepted} rejected={self.rejected} "
            f"total={total} actual_ratio={self.acceptance_ratio():.6f}"
        )


def _dbg_bernoulli(population, rate, seed):
    """Quick diagnostic: sample *population* rows and print stats."""
    bs = BernoulliSampler(rate=rate, seed=seed)
    rows = bs.sample(population)
    bs._dbg()
    print(f"  first_5_rows={rows[:5].tolist()} last_5_rows={rows[-5:].tolist()}")
    return rows


# ---------------------------------------------------------------------------
# HyperLogLog cardinality estimator
# ---------------------------------------------------------------------------
class HyperLogLog:
    """HyperLogLog probabilistic distinct-count estimator.

    Provides O(1) per-insert, O(m) memory cardinality estimates for
    streaming row-id populations, useful for tracking approximate
    distinct counts across sampled tables without keeping full sets.

    Uses the 64-bit hash variant with bias correction constants from
    the original Flajolet et al. paper.  Registers are stored as a
    numpy uint8 array for compact, vectorised access.
    """

    def __init__(self, precision=10):
        if not (4 <= precision <= 16):
            raise ValueError(f"precision must be in [4, 16], got {precision}")
        self.p = precision
        self.m = 1 << precision          # number of registers
        self.registers = np.zeros(self.m, dtype=np.uint8)
        self._alpha = self._compute_alpha()
        self.n_added = 0

    def _compute_alpha(self):
        """Bias correction constant alpha_m."""
        m = self.m
        if m == 16:
            return 0.673
        elif m == 32:
            return 0.697
        elif m == 64:
            return 0.709
        else:
            return 0.7213 / (1.0 + 1.079 / m)

    @staticmethod
    def _hash64(value):
        """Deterministic 64-bit hash via SHA-256 truncation."""
        h = hashlib.sha256(str(value).encode('utf-8')).digest()
        return struct.unpack('<Q', h[:8])[0]

    def _leading_zeros(self, w):
        """Count leading zeros in the 64-p least-significant bits."""
        if w == 0:
            return 64 - self.p
        count = 0
        bits = 64 - self.p
        for i in range(bits - 1, -1, -1):
            if w & (1 << i):
                break
            count += 1
        return count + 1  # rho in HLL is 1-indexed

    def add(self, value):
        """Insert a value into the sketch."""
        x = self._hash64(value)
        j = x & (self.m - 1)             # register index (low p bits)
        w = x >> self.p                   # remaining bits
        rho = self._leading_zeros(w)
        if rho > self.registers[j]:
            self.registers[j] = rho
        self.n_added += 1

    def add_batch(self, values):
        """Insert multiple values in one call (convenience wrapper)."""
        for v in values:
            self.add(v)

    def cardinality(self):
        """Return the estimated number of distinct values.

        Uses numpy vectorised operations for the harmonic-mean sum across
        all registers.
        """
        m = self.m
        z = np.sum(np.power(2.0, -self.registers.astype(np.float64)))
        raw = float(self._alpha * m * m / z)

        # Small-range correction
        if raw <= 2.5 * m:
            zeros = int(np.count_nonzero(self.registers == 0))
            if zeros > 0:
                return m * math.log(m / zeros)
            return raw

        # Large-range correction (64-bit)
        upper = 2.0 ** 64
        if raw > upper / 30.0:
            return -upper * math.log(1.0 - raw / upper)

        return raw

    def merge(self, other):
        """Merge another HyperLogLog sketch into this one (element-wise max)."""
        if self.p != other.p:
            raise ValueError("Cannot merge HLLs with different precisions")
        np.maximum(self.registers, other.registers, out=self.registers)
        self.n_added += other.n_added

    def _dbg(self):
        est = self.cardinality()
        non_zero = int(np.count_nonzero(self.registers))
        max_reg = int(np.max(self.registers)) if self.registers.size else 0
        print(
            f"[HyperLogLog._dbg] p={self.p} m={self.m} n_added={self.n_added} "
            f"est_cardinality={est:.1f} non_zero_regs={non_zero} "
            f"max_register={max_reg} alpha={self._alpha:.4f}"
        )


def _dbg_hyperloglog(values, precision=10):
    """Quick diagnostic: insert *values* and print cardinality estimate."""
    hll = HyperLogLog(precision=precision)
    for v in values:
        hll.add(v)
    hll._dbg()
    actual = len(set(values))
    est = hll.cardinality()
    rel_err = abs(est - actual) / actual if actual > 0 else 0.0
    print(f"  actual_distinct={actual} relative_error={rel_err:.4%}")
    return hll


# ---------------------------------------------------------------------------
# Welford online variance accumulator
# ---------------------------------------------------------------------------
class WelfordVariance:
    """Welford's algorithm for numerically stable online mean / variance.

    Tracks row counts per table in a single pass; no need to buffer all
    values.  Supports both scalar update and bulk numpy-array update.
    """

    __slots__ = ('count', 'mean', '_m2')

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self._m2 = 0.0

    def update(self, x):
        """Incorporate a new observation (scalar)."""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self._m2 += delta * delta2

    def update_batch(self, arr):
        """Incorporate an array of observations via numpy.

        Processes each element through the Welford recurrence.  For very
        large arrays, call this instead of a Python loop.
        """
        values = np.asarray(arr, dtype=np.float64)
        for x in values:
            self.update(float(x))

    def variance(self):
        """Sample variance (Bessel-corrected)."""
        return self._m2 / (self.count - 1) if self.count >= 2 else 0.0

    def stddev(self):
        return math.sqrt(self.variance())

    def cv(self):
        """Coefficient of variation (stddev / mean)."""
        return self.stddev() / self.mean if self.mean != 0 else 0.0

    def snapshot(self):
        """Return a dict summary of the current state."""
        return {
            'count': self.count,
            'mean': self.mean,
            'variance': self.variance(),
            'stddev': self.stddev(),
            'cv': self.cv(),
        }

    def _dbg(self):
        s = self.snapshot()
        print(
            f"[WelfordVariance._dbg] count={s['count']} mean={s['mean']:.4f} "
            f"var={s['variance']:.4f} std={s['stddev']:.4f} cv={s['cv']:.4f}"
        )


# ---------------------------------------------------------------------------
# EMA smoother for rate / progress tracking
# ---------------------------------------------------------------------------
class EMASmoother:
    """Exponential Moving Average smoother.

    Configurable decay (alpha).  Used to smooth instantaneous table-
    creation rates so that progress estimates don't jitter.
    """

    __slots__ = ('alpha', '_ema', '_last_ts', 'ticks', '_initialized')

    def __init__(self, alpha=0.3):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._ema = 0.0
        self._last_ts = None
        self.ticks = 0
        self._initialized = False

    def tick(self, ts=None):
        """Record a completion event; update the smoothed rate."""
        now = ts if ts is not None else time.time()
        self.ticks += 1
        if self._last_ts is not None:
            dt = now - self._last_ts
            rate = 1.0 / dt if dt > 0 else 0.0
            if not self._initialized:
                self._ema = rate
                self._initialized = True
            else:
                self._ema = self.alpha * rate + (1.0 - self.alpha) * self._ema
        self._last_ts = now

    def tick_batch(self, timestamps):
        """Process multiple timestamps in order (numpy array or list)."""
        ts_arr = np.asarray(timestamps, dtype=np.float64)
        for t in ts_arr:
            self.tick(ts=float(t))

    def smoothed_rate(self):
        return self._ema

    def eta(self, remaining):
        """Estimate time to complete *remaining* items at current rate."""
        if self._ema > 0:
            return remaining / self._ema
        return float('inf')

    def _dbg(self):
        print(
            f"[EMASmoother._dbg] alpha={self.alpha} ticks={self.ticks} "
            f"smoothed_rate={self._ema:.4f} items/sec"
        )


# ---------------------------------------------------------------------------
# DB configuration (mirrored from upstream db_random.py)
# ---------------------------------------------------------------------------
db_config = {
    'dbname': 'imdbloadbase',
    'user': 'hx68',
    'host': '/tmp',
    'port': '5432',
}

random_seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# IMDb JOB schema: tables sampled by title FK
_TABLES_BY_TITLE = [
    "movie_companies", "movie_keyword", "cast_info", "movie_link",
    "movie_info", "complete_cast", "aka_title", "movie_info_idx",
]

# Second-level derived tables and their join sources
_DERIVED_TABLES = OrderedDict([
    ("keyword",         ("movie_keyword",   "keyword_id")),
    ("company_name",    ("movie_companies", "company_id")),
    ("aka_name",        ("cast_info",       "person_id")),
    ("name",            ("aka_name",        "id")),
    ("person_info",     ("name",            "id")),
    ("link_type",       ("movie_link",      "link_type_id")),
    ("info_type",       None),  # multi-source, handled specially
    ("company_type",    ("movie_companies", "company_type_id")),
    ("kind_type",       ("title",           "kind_id")),
    ("char_name",       ("cast_info",       "person_role_id")),
    ("role_type",       ("cast_info",       "role_id")),
    ("comp_cast_type",  None),  # multi-source
])

RAW_SIZE_TITLE = 2528312


# ---------------------------------------------------------------------------
# Simulated row-count generation (deterministic hash, no live DB)
# ---------------------------------------------------------------------------
def _simulated_row_count(table_name, seed, window_pct):
    """Deterministic pseudo row-count for *table_name*.

    Mimics the size that a BERNOULLI(window_pct) sample with REPEATABLE(seed)
    would produce, without touching any database.
    """
    h = int(hashlib.sha256(f"{table_name}:{seed}".encode()).hexdigest()[:8], 16)
    base = h % 500_000
    scaled = int(base * (window_pct / 100.0))
    return max(1, scaled)


def _dbg_simulated_row_count(table_name, seed, window_pct):
    cnt = _simulated_row_count(table_name, seed, window_pct)
    print(
        f"[_simulated_row_count._dbg] table={table_name} seed={seed} "
        f"window_pct={window_pct} -> count={cnt}"
    )
    return cnt


# ---------------------------------------------------------------------------
# Build table list for a random instance (mirrors upstream DAG)
# ---------------------------------------------------------------------------
def build_table_manifest(instance_id):
    """Return the ordered list of table names for random instance *instance_id*.

    Follows the same dependency DAG as upstream db_random.py:
      1. random_title_{i}  (root, Bernoulli-sampled from title)
      2. random_{t}_{i}    for each table joined via title.id
      3. Second-level derived tables
    """
    tables = [f"random_title_{instance_id}"]
    for t in _TABLES_BY_TITLE:
        tables.append(f"random_{t}_{instance_id}")
    for derived in _DERIVED_TABLES:
        tables.append(f"random_{derived}_{instance_id}")
    return tables


def _dbg_build_table_manifest(instance_id):
    tables = build_table_manifest(instance_id)
    print(f"[build_table_manifest._dbg] instance={instance_id} n_tables={len(tables)}")
    for j, t in enumerate(tables):
        print(f"  [{j:2d}] {t}")
    return tables


# ---------------------------------------------------------------------------
# Core simulator: Bernoulli-sampled random instance with analytics
# ---------------------------------------------------------------------------
def simulate_random_instance(instance_id, window_pct=20, seed=None):
    """Simulate creation of a full random database instance.

    Produces the same set of tables as upstream
    verify_by_multiple_random_instances, but without a live PostgreSQL
    connection.  Each table gets a deterministic pseudo row-count driven
    by the seed and window percentage.

    Returns
    -------
    dict with keys:
        seed         – the RNG seed used
        tables       – ordered list of table names
        cache        – OrderedDict mapping table → simulated row count
        bernoulli    – BernoulliSampler used on the root table
        hll          – HyperLogLog sketch over all row IDs
        welford      – WelfordVariance accumulator over row counts
        ema          – EMASmoother tracking table-creation rate
    """
    if seed is None:
        seed = random_seeds[instance_id] if instance_id < len(random_seeds) else instance_id + 1

    tables = build_table_manifest(instance_id)
    cache = OrderedDict()

    # Bernoulli sample the root table
    bernoulli = BernoulliSampler(rate=window_pct, seed=seed)
    root_sample = bernoulli.sample(RAW_SIZE_TITLE)
    cache[tables[0]] = int(root_sample.size)

    # HLL sketch: track distinct row IDs across all tables
    hll = HyperLogLog(precision=12)
    cap = min(10000, root_sample.size)
    for rid in root_sample[:cap]:
        hll.add(f"{tables[0]}:{int(rid)}")

    # Welford + EMA
    wf = WelfordVariance()
    ema = EMASmoother(alpha=0.25)

    wf.update(float(root_sample.size))
    ema.tick()

    # Simulate dependent tables
    for tbl in tables[1:]:
        cnt = _simulated_row_count(tbl, seed, window_pct)
        cache[tbl] = cnt
        wf.update(float(cnt))
        ema.tick()
        # Feed a sample of IDs into HLL
        for rid in range(min(cnt, 5000)):
            hll.add(f"{tbl}:{rid}")

    return {
        'seed': seed,
        'tables': tables,
        'cache': cache,
        'bernoulli': bernoulli,
        'hll': hll,
        'welford': wf,
        'ema': ema,
    }


def _dbg_simulate_random_instance(instance_id, window_pct=20, seed=None):
    result = simulate_random_instance(instance_id, window_pct, seed)
    print(
        f"[simulate_random_instance._dbg] instance={instance_id} "
        f"seed={result['seed']} n_tables={len(result['tables'])}"
    )
    result['bernoulli']._dbg()
    result['hll']._dbg()
    result['welford']._dbg()
    result['ema']._dbg()
    print("  table row-counts (first 5):")
    for tbl, cnt in list(result['cache'].items())[:5]:
        print(f"    {tbl}: {cnt}")
    return result


# ---------------------------------------------------------------------------
# Batch pipeline: multiple instances with global statistics
# ---------------------------------------------------------------------------
def run_multi_instance_pipeline(n_instances=9, window_pct=20):
    """Run n_instances simulated random-DB creations and collect global stats.

    Returns
    -------
    dict with keys:
        instances     – list of per-instance result dicts
        global_wf     – WelfordVariance over *all* row counts across instances
        global_hll    – HyperLogLog over *all* row IDs across instances
        total_tables  – total number of tables created
    """
    global_wf = WelfordVariance()
    global_hll = HyperLogLog(precision=14)
    instances = []

    for i in range(n_instances):
        res = simulate_random_instance(i, window_pct)
        instances.append(res)
        for tbl, cnt in res['cache'].items():
            global_wf.update(float(cnt))
            for rid in range(min(cnt, 1000)):
                global_hll.add(f"{tbl}:{rid}")

    total_tables = sum(len(r['tables']) for r in instances)
    return {
        'instances': instances,
        'global_wf': global_wf,
        'global_hll': global_hll,
        'total_tables': total_tables,
    }


def _dbg_run_multi_instance_pipeline(n_instances=9, window_pct=20):
    result = run_multi_instance_pipeline(n_instances, window_pct)
    print(
        f"[run_multi_instance_pipeline._dbg] n_instances={n_instances} "
        f"total_tables={result['total_tables']}"
    )
    result['global_wf']._dbg()
    result['global_hll']._dbg()
    # Per-instance summary
    for i, inst in enumerate(result['instances']):
        wf = inst['welford']
        print(
            f"  instance[{i}] seed={inst['seed']} "
            f"mean_rows={wf.mean:.1f} cv={wf.cv():.4f}"
        )
    return result


# ---------------------------------------------------------------------------
# Drop simulation (mirrors upstream drop_sampled_tables)
# ---------------------------------------------------------------------------
def generate_drop_statements(instance_id):
    """Return DROP TABLE IF EXISTS … CASCADE statements for instance_id."""
    tables = build_table_manifest(instance_id)
    return [f"DROP TABLE IF EXISTS {t} CASCADE;" for t in tables]


def _dbg_generate_drop_statements(instance_id):
    stmts = generate_drop_statements(instance_id)
    print(f"[generate_drop_statements._dbg] instance={instance_id} n_stmts={len(stmts)}")
    for s in stmts[:5]:
        print(f"  {s}")
    if len(stmts) > 5:
        print(f"  ... ({len(stmts) - 5} more)")
    return stmts


# ---------------------------------------------------------------------------
# Index DDL generation (mirrors upstream index_query construction)
# ---------------------------------------------------------------------------
def generate_index_ddl(instance_id):
    """Generate CREATE INDEX statements for random instance *instance_id*."""
    index_template = """\
CREATE INDEX company_id_movie_companies ON movie_companies(company_id);
CREATE INDEX company_type_id_movie_companies ON movie_companies(company_type_id);
CREATE INDEX info_type_id_movie_info_idx ON movie_info_idx(info_type_id);
CREATE INDEX info_type_id_movie_info ON movie_info(info_type_id);
CREATE INDEX info_type_id_person_info ON person_info(info_type_id);
CREATE INDEX keyword_id_movie_keyword ON movie_keyword(keyword_id);
CREATE INDEX kind_id_aka_title ON aka_title(kind_id);
CREATE INDEX kind_id_title ON title(kind_id);
CREATE INDEX linked_movie_id_movie_link ON movie_link(linked_movie_id);
CREATE INDEX link_type_id_movie_link ON movie_link(link_type_id);
CREATE INDEX movie_id_aka_title ON aka_title(movie_id);
CREATE INDEX movie_id_cast_info ON cast_info(movie_id);
CREATE INDEX movie_id_complete_cast ON complete_cast(movie_id);
CREATE INDEX subject_id_complete_cast ON complete_cast(subject_id);
CREATE INDEX status_id_complete_cast ON complete_cast(status_id);
CREATE INDEX movie_id_movie_companies ON movie_companies(movie_id);
CREATE INDEX movie_id_movie_info_idx ON movie_info_idx(movie_id);
CREATE INDEX movie_id_movie_keyword ON movie_keyword(movie_id);
CREATE INDEX movie_id_movie_link ON movie_link(movie_id);
CREATE INDEX movie_id_movie_info ON movie_info(movie_id);
CREATE INDEX person_id_aka_name ON aka_name(person_id);
CREATE INDEX person_id_cast_info ON cast_info(person_id);
CREATE INDEX person_id_person_info ON person_info(person_id);
CREATE INDEX person_role_id_cast_info ON cast_info(person_role_id);
CREATE INDEX role_id_cast_info ON cast_info(role_id);"""

    i = instance_id
    ddl = index_template.replace('(', f'_{i}(')
    ddl = ddl.replace(' ON ', f'_random_{i} ON random_')
    return ddl


def _dbg_generate_index_ddl(instance_id):
    ddl = generate_index_ddl(instance_id)
    lines = [l for l in ddl.splitlines() if l.strip()]
    print(f"[generate_index_ddl._dbg] instance={instance_id} n_indexes={len(lines)}")
    for l in lines[:3]:
        print(f"  {l}")
    if len(lines) > 3:
        print(f"  ... ({len(lines) - 3} more)")
    return ddl


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("M141: par2qo_db_random_sampler experiment run")
    print("=" * 72)

    # 1. Bernoulli sampler demo
    print("\n--- Bernoulli Sampler (rate=20%, seed=42, pop=10000) ---")
    _dbg_bernoulli(population=10_000, rate=20.0, seed=42)

    # 2. Reproducibility check: same seed → same result
    print("\n--- Reproducibility Check ---")
    bs1 = BernoulliSampler(rate=15.0, seed=7)
    bs2 = BernoulliSampler(rate=15.0, seed=7)
    s1 = bs1.sample(5000)
    s2 = bs2.sample(5000)
    print(f"  same_seed_same_result: {np.array_equal(s1, s2)}")
    print(f"  sample_size: {s1.size}")

    # 3. HyperLogLog accuracy test
    print("\n--- HyperLogLog Accuracy (50k distinct, 80k total) ---")
    values = list(range(50_000)) + list(range(30_000))  # 50k distinct
    _dbg_hyperloglog(values, precision=12)

    # 4. Welford variance demo
    print("\n--- Welford Variance (100 observations) ---")
    wf = WelfordVariance()
    rng = np.random.default_rng(99)
    observations = rng.normal(1000, 200, size=100)
    wf.update_batch(observations)
    wf._dbg()

    # 5. EMA smoother demo
    print("\n--- EMA Smoother (20 ticks, alpha=0.3) ---")
    ema = EMASmoother(alpha=0.3)
    t0 = 1000.0
    timestamps = np.array([t0 + j * 0.05 + 0.01 * (j % 3) for j in range(20)])
    ema.tick_batch(timestamps)
    ema._dbg()

    # 6. Single instance simulation
    print("\n--- Single Instance Simulation (i=0, window=20%) ---")
    _dbg_simulate_random_instance(0, window_pct=20)

    # 7. Multi-instance pipeline
    print("\n--- Multi-Instance Pipeline (9 instances) ---")
    _dbg_run_multi_instance_pipeline(n_instances=9, window_pct=20)

    # 8. Drop statements demo
    print("\n--- Drop Statements (instance=5) ---")
    _dbg_generate_drop_statements(5)

    # 9. Index DDL demo
    print("\n--- Index DDL (instance=2) ---")
    _dbg_generate_index_ddl(2)

    print("\nM141 complete.")


if __name__ == "__main__":
    main()
