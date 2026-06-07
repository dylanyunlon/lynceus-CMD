"""
videx_metadata_extended — Extended metadata management with versioning for Lynceus.

Ported from:
  - upstream/videx/meta.py (363 lines)
  - upstream/videx/common/db_variable.py (237 lines)
  - upstream/videx/common/common_operation.py (123 lines)

Algorithm changes (~20%):
  - TableMeta: vector clock versioning for distributed metadata consistency
  - ColumnMeta: Bloom filter for fast column existence checking
  - IndexMeta: cost-based index ordering with Zipf selectivity model
  - VariableTracker: diff-based incremental sync (only send changes)
"""
import math
import os
import time
import hashlib
from collections import OrderedDict, defaultdict

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))

def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[meta_ext] {tag}: {items}")


# ── Vector Clock for metadata versioning ─────────────────────────
class VectorClock:
    """Vector clock for distributed metadata consistency.
    
    Algorithm addition: upstream has no versioning.
    Tracks causal ordering of metadata updates across nodes.
    """
    
    def __init__(self, node_id="local"):
        self.node_id = node_id
        self._clock = defaultdict(int)
        self._clock[node_id] = 0
    
    def tick(self):
        """Increment local clock."""
        self._clock[self.node_id] += 1
        _dbg("vc_tick", node=self.node_id, value=self._clock[self.node_id])
        return dict(self._clock)
    
    def merge(self, other_clock):
        """Merge with another vector clock (element-wise max)."""
        for node, ts in other_clock.items():
            self._clock[node] = max(self._clock[node], ts)
        self._clock[self.node_id] += 1
    
    def happens_before(self, other_clock):
        """Check if this clock happens-before another."""
        for node, ts in self._clock.items():
            if ts > other_clock.get(node, 0):
                return False
        return any(self._clock.get(n, 0) < v for n, v in other_clock.items())
    
    def concurrent_with(self, other_clock):
        """Check if two clocks are concurrent (neither happens-before)."""
        return not self.happens_before(other_clock) and not all(
            other_clock.get(n, 0) <= v for n, v in self._clock.items()
        )
    
    @property
    def version(self):
        return dict(self._clock)


# ── Bloom Filter for column existence ────────────────────────────
class BloomFilter:
    """Simple Bloom filter for fast column existence checking.
    
    Algorithm addition: upstream does linear scan over column lists.
    Bloom filter provides O(1) membership test with tunable false positive rate.
    """
    
    def __init__(self, expected_items=100, fp_rate=0.01):
        self.size = max(8, int(-expected_items * math.log(fp_rate) / (math.log(2) ** 2)))
        self.n_hashes = max(1, int(self.size / expected_items * math.log(2)))
        self._bits = [False] * self.size
        self._count = 0
        
        _dbg("bloom_init", size=self.size, hashes=self.n_hashes)
    
    def add(self, item):
        """Add an item to the filter."""
        for i in range(self.n_hashes):
            idx = self._hash(item, i) % self.size
            self._bits[idx] = True
        self._count += 1
    
    def contains(self, item):
        """Check if item might be in the filter (may give false positives)."""
        return all(self._bits[self._hash(item, i) % self.size]
                   for i in range(self.n_hashes))
    
    @staticmethod
    def _hash(item, seed):
        h = hashlib.md5(f"{seed}:{item}".encode()).hexdigest()
        return int(h[:8], 16)


# ── Column metadata ─────────────────────────────────────────────
class ColumnMeta:
    """Extended column metadata."""
    
    def __init__(self, name, data_type, ordinal_position=0, is_nullable=True,
                 character_max_length=None, numeric_precision=None,
                 numeric_scale=None, column_key="", is_pk=False):
        self.name = name
        self.data_type = data_type
        self.ordinal_position = ordinal_position
        self.is_nullable = is_nullable
        self.character_max_length = character_max_length
        self.numeric_precision = numeric_precision
        self.numeric_scale = numeric_scale
        self.column_key = column_key
        self.is_pk = is_pk
    
    def estimated_size_bytes(self):
        """Estimate byte size of column values."""
        type_sizes = {
            "int": 4, "bigint": 8, "smallint": 2, "tinyint": 1,
            "float": 4, "double": 8, "decimal": 8,
            "varchar": min(self.character_max_length or 255, 255),
            "char": self.character_max_length or 1,
            "text": 256, "blob": 256,
            "datetime": 8, "timestamp": 4, "date": 3,
        }
        base_type = self.data_type.lower().split("(")[0]
        return type_sizes.get(base_type, 8)
    
    def to_dict(self):
        return {"name": self.name, "type": self.data_type,
                "position": self.ordinal_position, "pk": self.is_pk}


# ── Index metadata with Zipf selectivity model ──────────────────
class IndexMeta:
    """Index metadata with cost-based ordering.
    
    Algorithm change: upstream stores indexes in declaration order.
    We order by estimated selectivity using Zipf's law on column NDV,
    giving the optimizer better candidates first.
    """
    
    def __init__(self, name, columns, index_type="BTREE", is_unique=False):
        self.name = name
        self.columns = columns  # list of column names
        self.index_type = index_type
        self.is_unique = is_unique
        self._selectivity = None
    
    def estimate_selectivity(self, ndv_dict, total_rows):
        """Estimate index selectivity using Zipf model.
        
        For multi-column index, selectivity combines individual column
        selectivities assuming power-law correlation (not independence).
        """
        if total_rows <= 0:
            return 1.0
        
        combined_sel = 1.0
        for i, col in enumerate(self.columns):
            ndv = ndv_dict.get(col, max(1, total_rows // 10))
            col_sel = 1.0 / max(ndv, 1)
            
            # Zipf correction: later columns are correlated with prefix
            # Selectivity reduction follows power law: s_i ∝ s_1^(1/(1+α*i))
            alpha = 0.5
            zipf_exponent = 1.0 / (1.0 + alpha * i)
            adjusted_sel = col_sel ** zipf_exponent
            combined_sel *= adjusted_sel
        
        self._selectivity = max(1.0 / total_rows, min(combined_sel, 1.0))
        
        _dbg("idx_sel", name=self.name, cols=self.columns,
             sel=f"{self._selectivity:.6f}")
        return self._selectivity
    
    @property
    def selectivity(self):
        return self._selectivity


# ── Table metadata with vector clock versioning ──────────────────
class TableMeta:
    """Table metadata container with distributed versioning."""
    
    def __init__(self, db_name, table_name, ddl="", engine="InnoDB",
                 row_count=0, avg_row_len=0, node_id="local"):
        self.db_name = db_name
        self.table_name = table_name
        self.ddl = ddl
        self.engine = engine
        self.row_count = row_count
        self.avg_row_len = avg_row_len
        self.columns = {}  # name -> ColumnMeta
        self.indexes = {}  # name -> IndexMeta
        self._vclock = VectorClock(node_id)
        self._bloom = BloomFilter(expected_items=200)
        self._last_modified = time.time()
        
        _dbg("table_init", db=db_name, table=table_name)
    
    def add_column(self, col):
        """Add a column to the table."""
        self.columns[col.name] = col
        self._bloom.add(col.name)
        self._vclock.tick()
    
    def add_index(self, idx):
        """Add an index to the table."""
        self.indexes[idx.name] = idx
        self._vclock.tick()
    
    def has_column(self, name):
        """Fast column existence check via Bloom filter."""
        if not self._bloom.contains(name):
            return False  # Definitely not present
        return name in self.columns  # Confirm (handles false positives)
    
    def order_indexes_by_selectivity(self, ndv_dict):
        """Order indexes by estimated selectivity (most selective first).
        
        Algorithm change: returns indexes ordered by cost-effectiveness.
        """
        for idx in self.indexes.values():
            idx.estimate_selectivity(ndv_dict, self.row_count)
        
        ordered = sorted(self.indexes.values(),
                        key=lambda x: x.selectivity or 1.0)
        return ordered
    
    @property
    def version(self):
        return self._vclock.version
    
    def dump_state(self):
        print(f"[TableMeta] {self.db_name}.{self.table_name}")
        print(f"  columns: {len(self.columns)}, indexes: {len(self.indexes)}")
        print(f"  rows: {self.row_count}, avg_len: {self.avg_row_len}")
        print(f"  version: {self._vclock.version}")


# ── Variable tracker with diff-based sync ────────────────────────
class VariableTracker:
    """Track MySQL variables with diff-based incremental sync.
    
    Algorithm change: upstream sends full variable state on every sync.
    Diff-based: only transmit changed variables, reducing bandwidth.
    """
    
    def __init__(self):
        self._current = {}
        self._previous = {}
        self._change_count = defaultdict(int)
    
    def update(self, variables):
        """Update variables and compute diff."""
        self._previous = dict(self._current)
        self._current = dict(variables)
        
        diff = {}
        for key, value in self._current.items():
            if key not in self._previous or self._previous[key] != value:
                diff[key] = {"old": self._previous.get(key), "new": value}
                self._change_count[key] += 1
        
        _dbg("var_update", total=len(self._current), changed=len(diff))
        return diff
    
    def get_diff(self):
        """Get current diff from previous state."""
        diff = {}
        for key in set(list(self._current.keys()) + list(self._previous.keys())):
            if self._current.get(key) != self._previous.get(key):
                diff[key] = {"old": self._previous.get(key),
                            "new": self._current.get(key)}
        return diff
    
    def most_volatile(self, top_n=5):
        """Return the most frequently changed variables."""
        return sorted(self._change_count.items(), key=lambda x: -x[1])[:top_n]
    
    def dump_state(self):
        print(f"[VarTracker] {len(self._current)} vars, "
              f"{len(self.get_diff())} changed")
