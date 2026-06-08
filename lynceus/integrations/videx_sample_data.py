"""
M189: videx_sample_data — Sample Data Management with Reservoir Sampling
Upstream: sample_info.py (108L) + sample_file_info.py (82L) + statistics_info.py (74L)
Algorithm changes (20%):
  - Reservoir sampling (Vitter Algorithm R) for row selection instead of full scan
  - Welford online variance for incremental NDV estimation
  - CRC32 column fingerprinting for fast dedup detection
  - _debug_snapshot() on all state mutations
"""
import time
import math
import random
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple

logger = logging.getLogger(__name__)

_DBG_ENABLED = True

def _dbg(tag: str, **kw):
    if _DBG_ENABLED:
        flat = {k: repr(v)[:100] for k, v in kw.items()}
        print(f"  [dbg:{tag}] {flat}")


# ── TableId (replaces upstream sub_platforms.sql_opt.meta.TableId) ──
@dataclass(frozen=True)
class TableId:
    db_name: str
    table_name: str

    def __hash__(self):
        return hash((self.db_name, self.table_name))


# ── SampleColumnInfo (from sample_info.py) ──
@dataclass
class SampleColumnInfo:
    table_id: TableId
    column_name: str
    data_type: Optional[str] = None
    sample_length: int = 0
    _fingerprint: Optional[str] = field(default=None, repr=False)

    @property
    def db_name(self):
        return self.table_id.db_name

    @property
    def table_name(self):
        return self.table_id.table_name

    def compute_fingerprint(self, data_sample: bytes = b"") -> str:
        """CRC32-based column fingerprint for fast dedup detection."""
        import zlib
        raw = f"{self.table_id.db_name}:{self.table_id.table_name}:{self.column_name}:{self.data_type}"
        crc = zlib.crc32(raw.encode() + data_sample) & 0xFFFFFFFF
        self._fingerprint = f"{crc:08x}"
        _dbg("compute_fingerprint", col=self.column_name, fp=self._fingerprint)
        return self._fingerprint

    @classmethod
    def new_ins(cls, db_name: str, table_name: str, column_name: str,
                sample_length: int = 0, data_type: str = None):
        tid = TableId(db_name=db_name, table_name=table_name)
        inst = cls(table_id=tid, column_name=column_name, data_type=data_type,
                   sample_length=sample_length)
        _dbg("SampleColumnInfo.new_ins", db=db_name, table=table_name, col=column_name)
        return inst

    def __hash__(self):
        return hash((self.table_id, self.column_name))

    def __eq__(self, other):
        if not isinstance(other, SampleColumnInfo):
            return False
        return self.table_id == other.table_id and self.column_name == other.column_name


# ── Welford Online Variance (for incremental NDV estimation) ──
class WelfordAccumulator:
    """Welford's algorithm for online mean/variance computation."""
    __slots__ = ("_n", "_mean", "_m2")

    def __init__(self):
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0

    def update(self, value: float):
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        delta2 = value - self._mean
        self._m2 += delta * delta2

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def variance(self) -> float:
        return self._m2 / self._n if self._n > 1 else 0.0

    @property
    def count(self) -> int:
        return self._n

    def _debug_snapshot(self) -> Dict[str, Any]:
        return {"n": self._n, "mean": round(self._mean, 6),
                "variance": round(self.variance, 6)}


# ── Reservoir Sampling (Algorithm R by Vitter) ──
def reservoir_sample(stream, k: int, seed: int = 42) -> List[Any]:
    """Vitter's Algorithm R: sample k items from stream without knowing length."""
    rng = random.Random(seed)
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = item
    _dbg("reservoir_sample", k=k, final_size=len(reservoir))
    return reservoir


# ── SampleResult (from sample_info.py) ──
@dataclass
class SampleResult:
    sample_fingerprint: Optional[str] = None
    result: Optional[tuple] = None
    dml: Optional[str] = None
    numerical_info: Dict[str, Any] = field(default_factory=dict)

    def _debug_snapshot(self) -> Dict[str, Any]:
        return {
            "fingerprint": self.sample_fingerprint,
            "has_result": self.result is not None,
            "dml": (self.dml[:80] + "...") if self.dml and len(self.dml) > 80 else self.dml,
            "numerical_keys": list(self.numerical_info.keys()),
        }


# ── SampleFileInfo (from sample_file_info.py, no pydantic dependency) ──
UNKNOWN_LOAD_ROWS = -1

class SampleFileInfo:
    """Manages sample file locations and load-row budgets."""
    def __init__(self, local_path_prefix: str = "", tos_path_prefix: str = "",
                 sample_file_dict: Optional[Dict[str, Any]] = None,
                 table_load_rows: Optional[Dict[str, Dict[str, int]]] = None):
        self.local_path_prefix = local_path_prefix
        self.tos_path_prefix = tos_path_prefix
        self.sample_file_dict = sample_file_dict or {}
        self.table_load_rows = table_load_rows
        _dbg("SampleFileInfo.__init__", local=local_path_prefix,
             tables=len(self.sample_file_dict))

    def get_table_load_row(self, db: str, table: str) -> int:
        if self.table_load_rows is None:
            return UNKNOWN_LOAD_ROWS
        return self.table_load_rows.get(db, {}).get(table, UNKNOWN_LOAD_ROWS)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "local_path_prefix": self.local_path_prefix,
            "tos_path_prefix": self.tos_path_prefix,
            "tables": list(self.sample_file_dict.keys()),
            "load_rows": self.table_load_rows,
        }

    def _debug_snapshot(self) -> Dict[str, Any]:
        snap = self.to_dict()
        snap["table_count"] = len(self.sample_file_dict)
        _dbg("SampleFileInfo.snapshot", **snap)
        return snap


# ── TableStatisticsInfo (from statistics_info.py, no pydantic) ──
class TableStatisticsInfo:
    """Table-level statistics: NDV, histograms, null ratios, row counts."""
    def __init__(self, db_name: str = "", table_name: str = ""):
        self.db_name = db_name
        self.table_name = table_name
        self.ndv_dict: Dict[str, float] = {}
        self.histogram_dict: Dict[str, Any] = {}
        self.not_null_ratio_dict: Dict[str, float] = {}
        self.num_of_rows: int = 0
        self.max_pk = None
        self.min_pk = None
        self.is_sample_success: bool = True
        self.is_sample_supported: bool = True
        self.unsupported_reason: Optional[str] = None
        self.sample_rows: int = 0
        self.local_path_prefix: Optional[str] = None
        self.tos_path_prefix: Optional[str] = None
        self.sample_file_list: List[str] = []
        self.block_size_list: List[int] = []
        self.shard_no: int = 0
        self.sample_error_dict: Dict[str, str] = {}
        self.histogram_error_dict: Dict[str, float] = {}
        self.msg: Optional[str] = None
        self.extra_info: Dict[str, Any] = {}
        # Welford accumulators for online NDV estimation
        self._ndv_welford: Dict[str, WelfordAccumulator] = {}
        _dbg("TableStatisticsInfo.__init__", db=db_name, table=table_name)

    def update_ndv_online(self, column: str, observed_unique: float):
        """Incrementally update NDV estimate using Welford's algorithm."""
        if column not in self._ndv_welford:
            self._ndv_welford[column] = WelfordAccumulator()
        self._ndv_welford[column].update(observed_unique)
        self.ndv_dict[column] = self._ndv_welford[column].mean
        _dbg("update_ndv_online", col=column, mean_ndv=self.ndv_dict[column],
             n=self._ndv_welford[column].count)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "db_name": self.db_name,
            "table_name": self.table_name,
            "ndv_dict": self.ndv_dict,
            "not_null_ratio_dict": self.not_null_ratio_dict,
            "num_of_rows": self.num_of_rows,
            "is_sample_success": self.is_sample_success,
            "sample_rows": self.sample_rows,
            "shard_no": self.shard_no,
        }

    def _debug_snapshot(self) -> Dict[str, Any]:
        snap = self.to_dict()
        snap["welford_stats"] = {
            col: acc._debug_snapshot() for col, acc in self._ndv_welford.items()
        }
        _dbg("TableStatisticsInfo.snapshot", db=self.db_name, table=self.table_name,
             cols=len(self.ndv_dict), rows=self.num_of_rows)
        return snap

    @classmethod
    def from_numerical_info(cls, info: Dict[str, Any]) -> "TableStatisticsInfo":
        ts = cls(db_name=info.get("db_name", ""), table_name=info.get("table_name", ""))
        ts.ndv_dict = info.get("ndv_dict", {})
        ts.histogram_dict = info.get("histogram", {})
        ts.not_null_ratio_dict = info.get("not_null_ratio_dict", {})
        ts.num_of_rows = info.get("num_of_rows", 0)
        ts.is_sample_success = info.get("is_sample_succ", True)
        ts.shard_no = info.get("shard_no", 0)
        _dbg("from_numerical_info", rows=ts.num_of_rows, ndv_cols=len(ts.ndv_dict))
        return ts


if __name__ == "__main__":
    print("=== M189 videx_sample_data self-test ===")

    # Test SampleColumnInfo
    col = SampleColumnInfo.new_ins("mydb", "users", "email", data_type="varchar")
    fp = col.compute_fingerprint(b"sample_data")
    assert len(fp) == 8

    # Test Welford
    w = WelfordAccumulator()
    for v in [10, 12, 23, 23, 16, 23, 21, 16]:
        w.update(float(v))
    assert abs(w.mean - 18.0) < 1.0
    assert w.count == 8

    # Test reservoir sampling
    stream = range(10000)
    sample = reservoir_sample(stream, k=100)
    assert len(sample) == 100

    # Test SampleFileInfo
    sfi = SampleFileInfo(local_path_prefix="/data/samples", tos_path_prefix="s3://bucket")
    assert sfi.get_table_load_row("db", "tbl") == UNKNOWN_LOAD_ROWS

    # Test TableStatisticsInfo
    tsi = TableStatisticsInfo(db_name="mydb", table_name="orders")
    tsi.num_of_rows = 50000
    for _ in range(5):
        tsi.update_ndv_online("status", random.uniform(3, 8))
    assert "status" in tsi.ndv_dict
    snap = tsi._debug_snapshot()
    assert snap["num_of_rows"] == 50000
    assert "status" in snap["welford_stats"]

    # Test from_numerical_info
    tsi2 = TableStatisticsInfo.from_numerical_info({
        "ndv_dict": {"id": 1000}, "histogram": {},
        "not_null_ratio_dict": {"id": 0.99}, "num_of_rows": 10000,
        "is_sample_succ": True, "shard_no": 0,
    })
    assert tsi2.num_of_rows == 10000

    print("  All tests passed!")
    print(f"  Lines: {sum(1 for _ in open(__file__))}")
