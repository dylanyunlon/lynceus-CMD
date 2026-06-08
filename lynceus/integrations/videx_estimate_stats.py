"""
M193: videx_estimate_stats — Column Statistics Length Estimation
Upstream: estimate_stats_length.py (~300L)
Algorithm changes (20%):
  - Zipf distribution for string length estimation (upstream: uniform)
  - Good-Turing smoothing for rare value frequency estimation
  - HyperLogLog cardinality approximation for NDV
"""
import math
import hashlib
import random
import logging
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)
_DBG = True

def _dbg(tag, **kw):
    if _DBG:
        print(f"  [dbg:{tag}] { {k: repr(v)[:80] for k,v in kw.items()} }")


class HyperLogLog:
    """HyperLogLog cardinality estimator (NDV approximation)."""
    def __init__(self, precision: int = 10):
        self._p = precision
        self._m = 1 << precision  # number of registers
        self._regs = bytearray(self._m)
        self._alpha = self._compute_alpha(self._m)
    
    @staticmethod
    def _compute_alpha(m: int) -> float:
        if m == 16: return 0.673
        if m == 32: return 0.697
        if m == 64: return 0.709
        return 0.7213 / (1 + 1.079 / m)
    
    def add(self, value: str):
        h = int(hashlib.md5(value.encode()).hexdigest(), 16)
        idx = h & (self._m - 1)
        remaining = h >> self._p
        rank = 1
        while remaining & 1 == 0 and rank <= 64 - self._p:
            rank += 1
            remaining >>= 1
        self._regs[idx] = max(self._regs[idx], rank)
    
    def estimate(self) -> float:
        indicator = sum(2.0 ** (-r) for r in self._regs)
        raw = self._alpha * self._m * self._m / indicator
        # Small range correction
        zeros = self._regs.count(0)
        if raw <= 2.5 * self._m and zeros > 0:
            raw = self._m * math.log(self._m / zeros)
        _dbg("hll_estimate", raw=round(raw, 1), zeros=zeros)
        return raw
    
    def snapshot(self):
        return {"precision": self._p, "registers": self._m, "estimate": round(self.estimate(), 1)}


class ZipfDistribution:
    """Zipf distribution for string length modeling."""
    def __init__(self, n: int, s: float = 1.0, seed: int = 42):
        self._n = n
        self._s = s
        self._rng = random.Random(seed)
        self._harmonic = sum(1.0 / (k ** s) for k in range(1, n + 1))
    
    def sample(self) -> int:
        u = self._rng.random()
        cumulative = 0.0
        for k in range(1, self._n + 1):
            cumulative += (1.0 / (k ** self._s)) / self._harmonic
            if u <= cumulative:
                return k
        return self._n
    
    def expected_value(self) -> float:
        return sum(k * (1.0 / (k ** self._s)) / self._harmonic for k in range(1, self._n + 1))


class GoodTuringSmoothing:
    """Good-Turing frequency estimation for rare values."""
    def __init__(self):
        self._freq_of_freq: Dict[int, int] = {}
        self._total = 0
    
    def observe(self, frequency: int):
        self._freq_of_freq[frequency] = self._freq_of_freq.get(frequency, 0) + 1
        self._total += 1
    
    def smoothed_frequency(self, r: int) -> float:
        """Smoothed frequency estimate for items seen r times."""
        if r == 0:
            n1 = self._freq_of_freq.get(1, 1)
            return n1 / max(self._total, 1)
        nr = self._freq_of_freq.get(r, 0)
        nr1 = self._freq_of_freq.get(r + 1, 0)
        if nr == 0:
            return float(r)
        return (r + 1) * nr1 / nr
    
    def missing_mass(self) -> float:
        """Fraction of population unseen (N1/N)."""
        n1 = self._freq_of_freq.get(1, 0)
        return n1 / max(self._total, 1)
    
    def snapshot(self):
        return {"freq_of_freq": dict(self._freq_of_freq), "total": self._total,
                "missing_mass": round(self.missing_mass(), 4)}


class ColumnStatsEstimator:
    """Estimate column statistics (length, NDV, selectivity) for query optimization."""
    
    # MySQL data type average lengths
    TYPE_LENGTHS = {
        "tinyint": 1, "smallint": 2, "mediumint": 3, "int": 4, "bigint": 8,
        "float": 4, "double": 8, "decimal": 8,
        "date": 3, "datetime": 8, "timestamp": 4, "time": 3, "year": 1,
        "char": 0, "varchar": 0,  # variable, estimated separately
        "tinytext": 20, "text": 200, "mediumtext": 2000, "longtext": 10000,
        "tinyblob": 20, "blob": 200, "mediumblob": 2000, "longblob": 10000,
        "enum": 2, "set": 8, "bit": 1, "json": 200, "binary": 0, "varbinary": 0,
    }
    
    def __init__(self):
        self._hll_cache: Dict[str, HyperLogLog] = {}
        self._gt_cache: Dict[str, GoodTuringSmoothing] = {}
        self._zipf = ZipfDistribution(n=256, s=1.1)
    
    def estimate_column_length(self, data_type: str, char_max_len: int = 0,
                                sample_values: Optional[List[str]] = None) -> float:
        """Estimate average stored length for a column."""
        base_type = data_type.lower().split("(")[0].strip()
        
        if base_type in ("varchar", "char", "varbinary", "binary"):
            if sample_values:
                avg_len = sum(len(v) for v in sample_values) / max(len(sample_values), 1)
            else:
                # Use Zipf distribution to estimate average string length
                zipf_est = self._zipf.expected_value()
                avg_len = min(zipf_est * (char_max_len / 256.0), char_max_len * 0.6)
            overhead = 2 if base_type.startswith("var") else 0
            result = avg_len + overhead
        else:
            result = float(self.TYPE_LENGTHS.get(base_type, 8))
        
        _dbg("estimate_column_length", type=data_type, result=round(result, 2))
        return result
    
    def estimate_ndv(self, column_key: str, sample_values: Optional[List[str]] = None,
                     num_rows: int = 0) -> float:
        """Estimate number of distinct values using HyperLogLog."""
        if column_key not in self._hll_cache:
            self._hll_cache[column_key] = HyperLogLog(precision=10)
        hll = self._hll_cache[column_key]
        
        if sample_values:
            for v in sample_values:
                hll.add(v)
            estimate = hll.estimate()
        else:
            estimate = max(1.0, num_rows * 0.3)  # fallback heuristic
        
        _dbg("estimate_ndv", col=column_key, ndv=round(estimate, 1))
        return estimate
    
    def estimate_selectivity(self, ndv: float, num_rows: int, 
                              is_equality: bool = True) -> float:
        """Estimate filter selectivity using Good-Turing for rare values."""
        if ndv <= 0 or num_rows <= 0:
            return 1.0
        
        col_key = f"sel_{ndv}_{num_rows}"
        if col_key not in self._gt_cache:
            gt = GoodTuringSmoothing()
            # Simulate frequency distribution
            freq_per_value = num_rows / ndv
            for i in range(int(min(ndv, 1000))):
                gt.observe(max(1, int(freq_per_value * (1.0 + 0.1 * (i % 10)))))
            self._gt_cache[col_key] = gt
        
        gt = self._gt_cache[col_key]
        
        if is_equality:
            base_sel = 1.0 / ndv
            # Adjust for unseen values using Good-Turing missing mass
            missing = gt.missing_mass()
            adjusted_sel = base_sel * (1 - missing) + missing * gt.smoothed_frequency(0)
            result = min(max(adjusted_sel, 1e-6), 1.0)
        else:
            result = min(max(1.0 / 3.0, 1e-6), 1.0)  # default range selectivity
        
        _dbg("estimate_selectivity", ndv=ndv, rows=num_rows, eq=is_equality, 
             sel=round(result, 6))
        return result
    
    def estimate_row_length(self, columns: List[Tuple[str, int]]) -> float:
        """Estimate average row length from column types."""
        total = 0.0
        for data_type, char_max_len in columns:
            total += self.estimate_column_length(data_type, char_max_len)
        # Add InnoDB record overhead
        total += 7  # record header
        total += 6  # transaction ID
        total += 7  # rollback pointer
        _dbg("estimate_row_length", n_cols=len(columns), total=round(total, 1))
        return total
    
    def _debug_snapshot(self) -> Dict[str, Any]:
        return {
            "hll_caches": len(self._hll_cache),
            "gt_caches": len(self._gt_cache),
            "hll_details": {k: v.snapshot() for k, v in self._hll_cache.items()},
        }


if __name__ == "__main__":
    print("=== M193 videx_estimate_stats self-test ===")
    
    est = ColumnStatsEstimator()
    
    # Test column length
    int_len = est.estimate_column_length("int")
    assert int_len == 4.0
    varchar_len = est.estimate_column_length("varchar", char_max_len=255)
    assert 0 < varchar_len < 255
    
    # Test NDV with HLL
    samples = [f"val_{i % 50}" for i in range(1000)]
    ndv = est.estimate_ndv("test_col", sample_values=samples)
    assert 30 < ndv < 80  # should be ~50
    
    # Test selectivity
    sel = est.estimate_selectivity(100, 10000, is_equality=True)
    assert 0 < sel < 0.1
    
    # Test row length
    cols = [("int", 0), ("varchar", 128), ("datetime", 0), ("decimal", 0)]
    row_len = est.estimate_row_length(cols)
    assert row_len > 20
    
    # Test Good-Turing
    gt = GoodTuringSmoothing()
    for freq in [1, 1, 1, 2, 2, 3, 5, 10]:
        gt.observe(freq)
    mm = gt.missing_mass()
    assert 0 <= mm <= 1
    
    # Test HyperLogLog
    hll = HyperLogLog(precision=12)
    for i in range(10000):
        hll.add(f"item_{i}")
    hll_est = hll.estimate()
    assert 8000 < hll_est < 12000
    
    print(f"  HLL estimate for 10K distinct: {hll_est:.0f}")
    print(f"  NDV estimate (true=50): {ndv:.0f}")
    print("  All tests passed!")
    print(f"  Lines: {sum(1 for _ in open(__file__))}")
