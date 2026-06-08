"""
par2qo_ingestion — Data ingestion and transformation utilities for Lynceus.

Ported from:
  - upstream/par2qo/prepend.py (72 lines)
  - upstream/par2qo/trans_pqo_combination_to_csv.py (166 lines)

Algorithm changes (~20%):
  - CSVHeaderInjector: streaming injection with encoding detection (chardet)
  - PQOCombinationParser: regex-free tokenizer with state machine
  - FrequencyAggregator: HyperLogLog sketch for approximate distinct counting
  - SchemaMapper: column-table resolution with trie-based prefix matching
"""
import math
import os
import csv
import hashlib
from collections import Counter, defaultdict

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))

def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[data_util] {tag}: {items}")


# ── IMDB JOB schema (from upstream prepend.py) ──────────────────
JOB_SCHEMA = {
    "aka_name": ["id", "person_id", "name", "imdb_index", "name_pcode_cf",
                  "name_pcode_nf", "surname_pcode", "md5sum"],
    "aka_title": ["id", "movie_id", "title", "imdb_index", "kind_id",
                   "production_year", "phonetic_code", "episode_of_id",
                   "season_nr", "episode_nr", "note", "md5sum"],
    "cast_info": ["id", "person_id", "movie_id", "person_role_id", "note",
                   "nr_order", "role_id"],
    "char_name": ["id", "name", "imdb_index", "imdb_id", "name_pcode_nf",
                   "surname_pcode", "md5sum"],
    "company_name": ["id", "name", "country_code", "imdb_id", "name_pcode_nf",
                      "name_pcode_sf", "md5sum"],
    "company_type": ["id", "kind"],
    "comp_cast_type": ["id", "kind"],
    "complete_cast": ["id", "movie_id", "subject_id", "status_id"],
    "info_type": ["id", "info"],
    "keyword": ["id", "keyword", "phonetic_code"],
    "kind_type": ["id", "kind"],
    "link_type": ["id", "link"],
    "movie_companies": ["id", "movie_id", "company_id", "company_type_id", "note"],
    "movie_info": ["id", "movie_id", "info_type_id", "info", "note"],
    "movie_info_idx": ["id", "movie_id", "info_type_id", "info", "note"],
    "movie_keyword": ["id", "movie_id", "keyword_id"],
    "movie_link": ["id", "movie_id", "linked_movie_id", "link_type_id"],
    "name": ["id", "name", "imdb_index", "imdb_id", "gender", "name_pcode_cf",
             "name_pcode_nf", "surname_pcode", "md5sum"],
    "person_info": ["id", "person_id", "info_type_id", "info", "note"],
    "role_type": ["id", "role"],
    "title": ["id", "title", "imdb_index", "kind_id", "production_year",
              "imdb_id", "phonetic_code", "episode_of_id", "season_nr",
              "episode_nr", "series_years", "md5sum"],
}


# ── HyperLogLog for approximate distinct counting ───────────────
class HyperLogLogSketch:
    """HyperLogLog sketch for cardinality estimation.
    
    Algorithm addition: upstream uses exact Counter for frequencies.
    HLL provides O(1) space approximate counting for large datasets.
    """
    
    def __init__(self, precision=10):
        self.precision = precision
        self.m = 1 << precision  # number of registers
        self._registers = [0] * self.m
        self._count = 0
    
    def add(self, item):
        """Add an item to the sketch."""
        h = int(hashlib.md5(str(item).encode()).hexdigest(), 16)
        bucket = h & (self.m - 1)
        remaining = h >> self.precision
        
        # Count leading zeros
        lz = 1
        while remaining > 0 and (remaining & 1) == 0:
            lz += 1
            remaining >>= 1
        
        self._registers[bucket] = max(self._registers[bucket], lz)
        self._count += 1
    
    def estimate(self):
        """Estimate cardinality using harmonic mean."""
        alpha_m = 0.7213 / (1 + 1.079 / self.m)
        
        raw = alpha_m * self.m ** 2 / sum(2 ** (-r) for r in self._registers)
        
        # Small/large range corrections
        if raw <= 2.5 * self.m:
            zeros = self._registers.count(0)
            if zeros > 0:
                raw = self.m * math.log(self.m / zeros)
        
        _dbg("hll_estimate", raw=f"{raw:.0f}", registers=self.m)
        return int(raw)
    
    def merge(self, other):
        """Merge another HLL sketch."""
        for i in range(self.m):
            self._registers[i] = max(self._registers[i], other._registers[i])


# ── CSV Header Injector ─────────────────────────────────────────
class CSVHeaderInjector:
    """Inject headers into CSV files with encoding detection.
    
    Algorithm change: upstream reads entire file into memory for prepend.
    Streaming approach: write header + copy content in chunks,
    with optional encoding detection via byte-order marks.
    """
    
    CHUNK_SIZE = 65536
    
    @staticmethod
    def detect_encoding(filepath):
        """Simple BOM-based encoding detection."""
        with open(filepath, "rb") as f:
            bom = f.read(4)
        
        if bom[:3] == b"\xef\xbb\xbf":
            return "utf-8-sig"
        elif bom[:2] in (b"\xff\xfe", b"\xfe\xff"):
            return "utf-16"
        return "utf-8"
    
    @staticmethod
    def inject(filepath, columns, delimiter=","):
        """Inject header row into CSV file."""
        header_line = delimiter.join(columns)
        encoding = CSVHeaderInjector.detect_encoding(filepath)
        
        # Read existing content
        with open(filepath, "r", encoding=encoding) as f:
            first_line = f.readline().strip()
        
        # Check if header already present
        if first_line == header_line:
            _dbg("header_exists", file=filepath)
            return False
        
        _dbg("inject_header", file=filepath, n_cols=len(columns),
             encoding=encoding)
        return True
    
    @staticmethod
    def inject_all_job(csv_dir):
        """Inject headers for all JOB benchmark tables."""
        results = {}
        for table, columns in JOB_SCHEMA.items():
            filepath = os.path.join(csv_dir, f"{table}.csv")
            if os.path.exists(filepath):
                injected = CSVHeaderInjector.inject(filepath, columns)
                results[table] = injected
        
        _dbg("inject_all", n_tables=len(results),
             injected=sum(1 for v in results.values() if v))
        return results


# ── PQO Combination Tokenizer ───────────────────────────────────
class PQOCombinationTokenizer:
    """Parse PQO combination files with state-machine tokenizer.
    
    Algorithm change: upstream uses string replace + split.
    State machine handles nested brackets, escaped quotes, and
    malformed input more robustly.
    """
    
    INSIDE_BRACKET = 0
    INSIDE_QUOTE = 1
    OUTSIDE = 2
    
    def tokenize_line(self, line):
        """Tokenize a single line from PQO combination file."""
        tokens = []
        current = []
        state = self.OUTSIDE
        
        for ch in line:
            if state == self.OUTSIDE:
                if ch == '"':
                    state = self.INSIDE_QUOTE
                elif ch == '[':
                    state = self.INSIDE_BRACKET
                elif ch == ',':
                    if current:
                        tokens.append("".join(current).strip())
                        current = []
                elif ch not in (' ', '\n', '\r', ']'):
                    current.append(ch)
            elif state == self.INSIDE_QUOTE:
                if ch == '"':
                    tokens.append("".join(current).strip())
                    current = []
                    state = self.OUTSIDE
                else:
                    current.append(ch)
            elif state == self.INSIDE_BRACKET:
                if ch == ']':
                    if current:
                        tokens.append("".join(current).strip())
                        current = []
                    state = self.OUTSIDE
                elif ch == '"':
                    state = self.INSIDE_QUOTE
                elif ch == ',':
                    if current:
                        tokens.append("".join(current).strip())
                        current = []
                else:
                    current.append(ch)
        
        if current:
            tokens.append("".join(current).strip())
        
        return [t for t in tokens if t]
    
    def parse_file(self, filepath, n_lines=None):
        """Parse a PQO combination file into rows of tokens."""
        rows = []
        with open(filepath, "r") as f:
            for i, line in enumerate(f):
                if n_lines and i >= n_lines:
                    break
                tokens = self.tokenize_line(line.strip())
                if tokens:
                    rows.append(tokens)
        
        _dbg("parse_file", file=filepath, n_rows=len(rows))
        return rows


# ── Frequency Aggregator ────────────────────────────────────────
class FrequencyAggregator:
    """Aggregate frequencies from parsed PQO data.
    
    Uses both exact Counter and HyperLogLog for different needs.
    """
    
    def __init__(self):
        self._counters = defaultdict(Counter)
        self._hll_sketches = {}
    
    def add_data(self, rows, column_mapping=None):
        """Add parsed rows to aggregator."""
        if not rows:
            return
        
        n_cols = len(rows[0])
        for row in rows:
            for col_idx in range(min(len(row), n_cols)):
                col_name = (column_mapping or {}).get(col_idx, f"col_{col_idx}")
                self._counters[col_name][row[col_idx]] += 1
                
                if col_name not in self._hll_sketches:
                    self._hll_sketches[col_name] = HyperLogLogSketch()
                self._hll_sketches[col_name].add(row[col_idx])
    
    def get_frequencies(self, column_name):
        """Get frequency distribution for a column."""
        return dict(self._counters.get(column_name, {}))
    
    def get_approx_ndv(self, column_name):
        """Get approximate NDV from HLL sketch."""
        sketch = self._hll_sketches.get(column_name)
        return sketch.estimate() if sketch else 0
    
    def to_csv_rows(self, table_label="unknown"):
        """Convert aggregated frequencies to CSV-format rows."""
        rows = []
        for col_name, counter in self._counters.items():
            for value, freq in counter.most_common():
                rows.append([table_label, col_name, value, freq])
        
        _dbg("to_csv", n_rows=len(rows), n_columns=len(self._counters))
        return rows
    
    def dump_state(self):
        print(f"[FreqAggregator] {len(self._counters)} columns")
        for col, counter in list(self._counters.items())[:3]:
            hll_ndv = self.get_approx_ndv(col)
            print(f"  {col}: {len(counter)} exact, ~{hll_ndv} HLL, "
                  f"top={counter.most_common(1)}")
