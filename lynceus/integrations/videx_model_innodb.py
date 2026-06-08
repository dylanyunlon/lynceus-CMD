"""
M192: videx_model_innodb — InnoDB Storage Engine Cost Model
Upstream: videx/src/sub_platforms/sql_opt/videx/model/ (multiple files ~600L total)
Algorithm changes (20%):
  - B-tree page split estimation using Fibonacci thresholds instead of fixed
  - IO cost via Amdahl parallel model: cost = serial + parallel/nthreads
  - Buffer pool hit-ratio using LRU-K(2) approximation
  - Welford online stats for page access latency
"""
import math
import time
import random
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict

logger = logging.getLogger(__name__)
_DBG = True

def _dbg(tag, **kw):
    if _DBG:
        print(f"  [dbg:{tag}] { {k: repr(v)[:80] for k,v in kw.items()} }")


class PageConfig:
    PAGE_SIZE = 16384       # 16KB InnoDB page
    FILL_FACTOR = 0.875     # default 15/16
    KEY_OVERHEAD = 14       # internal pointer overhead per key
    RECORD_HEADER = 7       # per-record header bytes


class FibonacciThresholds:
    """Page split thresholds based on Fibonacci sequence instead of fixed 50%."""
    _cache = [1, 1]
    
    @classmethod
    def fib(cls, n: int) -> int:
        while len(cls._cache) <= n:
            cls._cache.append(cls._cache[-1] + cls._cache[-2])
        return cls._cache[n]
    
    @classmethod
    def split_point(cls, page_fill_pct: float, level: int) -> float:
        """Fibonacci-weighted split point: deeper levels split more aggressively."""
        fib_weight = cls.fib(min(level + 2, 20))
        golden = 0.618
        threshold = golden * (1.0 + 0.01 * fib_weight)
        result = min(max(threshold, 0.5), 0.95)
        _dbg("fib_split", level=level, fill=page_fill_pct, threshold=result)
        return result


class WelfordLatency:
    __slots__ = ("_n", "_mean", "_m2")
    def __init__(self):
        self._n = 0; self._mean = 0.0; self._m2 = 0.0
    def record(self, val):
        self._n += 1
        d = val - self._mean
        self._mean += d / self._n
        self._m2 += d * (val - self._mean)
    @property
    def mean(self): return self._mean
    @property
    def std(self): return math.sqrt(self._m2/self._n) if self._n > 1 else 0.0
    def snapshot(self): return {"n": self._n, "mean": round(self._mean,4), "std": round(self.std,4)}


class BufferPoolLRUK:
    """LRU-K(2) buffer pool simulation for hit-ratio estimation."""
    def __init__(self, capacity_pages: int = 8192):
        self._capacity = capacity_pages
        self._hist1: OrderedDict[int, float] = OrderedDict()  # first access
        self._hist2: OrderedDict[int, float] = OrderedDict()  # second access (correlated)
        self._hits = 0
        self._misses = 0
    
    def access(self, page_id: int) -> bool:
        now = time.monotonic()
        if page_id in self._hist2:
            self._hist2.move_to_end(page_id)
            self._hist2[page_id] = now
            self._hits += 1
            return True
        
        if page_id in self._hist1:
            # Second access → promote to hist2
            del self._hist1[page_id]
            self._hist2[page_id] = now
            while len(self._hist2) > self._capacity:
                self._hist2.popitem(last=False)
            self._hits += 1
            return True
        
        # First access
        self._hist1[page_id] = now
        while len(self._hist1) > self._capacity * 2:
            self._hist1.popitem(last=False)
        self._misses += 1
        return False
    
    @property
    def hit_ratio(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    def snapshot(self):
        return {"capacity": self._capacity, "hits": self._hits, "misses": self._misses,
                "hit_ratio": round(self.hit_ratio, 4), "h1_size": len(self._hist1),
                "h2_size": len(self._hist2)}


class InnoDBCostModel:
    """InnoDB cost model with Amdahl-parallel IO and Fibonacci page splits."""
    
    # Cost constants (microseconds)
    RANDOM_IO_COST = 200.0    # SSD random read
    SEQ_IO_COST = 10.0        # sequential read per page
    CPU_TUPLE_COST = 0.5      # per-tuple CPU cost
    CPU_INDEX_COST = 0.3      # per-index-entry CPU cost
    
    def __init__(self, buffer_pool_pages: int = 8192, io_threads: int = 4):
        self._bp = BufferPoolLRUK(buffer_pool_pages)
        self._io_threads = io_threads
        self._latency = WelfordLatency()
        _dbg("InnoDBCostModel.__init__", bp_pages=buffer_pool_pages, io_threads=io_threads)
    
    def estimate_btree_height(self, num_rows: int, avg_key_len: int) -> int:
        """Estimate B-tree height from row count and key length."""
        keys_per_page = max(1, int(
            (PageConfig.PAGE_SIZE * PageConfig.FILL_FACTOR) /
            (avg_key_len + PageConfig.KEY_OVERHEAD)
        ))
        if num_rows <= 0:
            return 1
        height = max(1, math.ceil(math.log(num_rows) / math.log(keys_per_page)))
        _dbg("btree_height", rows=num_rows, key_len=avg_key_len, 
             keys_per_page=keys_per_page, height=height)
        return height
    
    def estimate_pages_in_range(self, num_rows: int, selectivity: float,
                                 avg_row_len: int) -> int:
        """Estimate pages to read for a range scan."""
        rows_in_range = max(1, int(num_rows * selectivity))
        rows_per_page = max(1, int(
            (PageConfig.PAGE_SIZE * PageConfig.FILL_FACTOR) /
            (avg_row_len + PageConfig.RECORD_HEADER)
        ))
        pages = max(1, math.ceil(rows_in_range / rows_per_page))
        _dbg("pages_in_range", rows=num_rows, sel=selectivity, 
             rows_in_range=rows_in_range, pages=pages)
        return pages
    
    def io_cost_amdahl(self, pages: int, is_sequential: bool = False) -> float:
        """
        Amdahl's law IO cost: C = serial_fraction * T + (1-serial_fraction) * T / threads.
        Sequential reads have higher parallelism benefit.
        """
        serial_frac = 0.1 if is_sequential else 0.4
        base_cost = self.SEQ_IO_COST if is_sequential else self.RANDOM_IO_COST
        total_cost = pages * base_cost
        
        parallel_cost = (serial_frac * total_cost + 
                        (1 - serial_frac) * total_cost / self._io_threads)
        
        # Adjust for buffer pool hits
        bp_miss_ratio = 1.0 - self._bp.hit_ratio
        adjusted = parallel_cost * bp_miss_ratio + pages * self.CPU_INDEX_COST
        
        _dbg("io_cost_amdahl", pages=pages, seq=is_sequential, 
             raw=round(total_cost,1), adjusted=round(adjusted,1),
             bp_hit=round(self._bp.hit_ratio, 3))
        self._latency.record(adjusted)
        return adjusted
    
    def full_table_scan_cost(self, num_rows: int, avg_row_len: int) -> float:
        pages = self.estimate_pages_in_range(num_rows, 1.0, avg_row_len)
        io = self.io_cost_amdahl(pages, is_sequential=True)
        cpu = num_rows * self.CPU_TUPLE_COST
        total = io + cpu
        _dbg("full_scan_cost", rows=num_rows, pages=pages, 
             io=round(io,1), cpu=round(cpu,1), total=round(total,1))
        return total
    
    def index_range_scan_cost(self, num_rows: int, selectivity: float,
                               avg_key_len: int, avg_row_len: int) -> float:
        height = self.estimate_btree_height(num_rows, avg_key_len)
        leaf_pages = self.estimate_pages_in_range(num_rows, selectivity, avg_row_len)
        
        # Index traversal: random IO for height + sequential for leaf pages
        traverse_io = self.io_cost_amdahl(height, is_sequential=False)
        leaf_io = self.io_cost_amdahl(leaf_pages, is_sequential=True)
        
        rows_in_range = max(1, int(num_rows * selectivity))
        cpu = rows_in_range * self.CPU_TUPLE_COST + height * self.CPU_INDEX_COST
        
        total = traverse_io + leaf_io + cpu
        _dbg("index_range_cost", rows=num_rows, sel=selectivity, height=height,
             leaf_pages=leaf_pages, total=round(total,1))
        return total
    
    def page_split_cost(self, current_fill: float, tree_level: int) -> float:
        """Estimate cost of a page split using Fibonacci thresholds."""
        threshold = FibonacciThresholds.split_point(current_fill, tree_level)
        if current_fill < threshold:
            return 0.0
        # Split cost: 2 random IOs (read old, write new + old)
        split_cost = 2 * self.RANDOM_IO_COST + PageConfig.PAGE_SIZE * 0.001  # memcpy cost
        _dbg("page_split_cost", fill=current_fill, threshold=threshold, cost=round(split_cost,1))
        return split_cost
    
    def simulate_workload(self, n_queries: int, num_rows: int = 100000,
                           avg_row_len: int = 120, avg_key_len: int = 20) -> Dict[str, Any]:
        """Simulate a mixed workload and return statistics."""
        rng = random.Random(42)
        costs = []
        for _ in range(n_queries):
            sel = rng.uniform(0.001, 0.3)
            # Simulate buffer pool access
            page_id = rng.randint(0, num_rows // 100)
            self._bp.access(page_id)
            
            if sel > 0.1:
                c = self.full_table_scan_cost(num_rows, avg_row_len)
            else:
                c = self.index_range_scan_cost(num_rows, sel, avg_key_len, avg_row_len)
            costs.append(c)
        
        result = {
            "n_queries": n_queries,
            "mean_cost": round(sum(costs)/len(costs), 2),
            "min_cost": round(min(costs), 2),
            "max_cost": round(max(costs), 2),
            "buffer_pool": self._bp.snapshot(),
            "latency_stats": self._latency.snapshot(),
        }
        _dbg("simulate_workload", n=n_queries, mean=result["mean_cost"])
        return result

    def _debug_snapshot(self) -> Dict[str, Any]:
        return {
            "io_threads": self._io_threads,
            "buffer_pool": self._bp.snapshot(),
            "latency": self._latency.snapshot(),
        }


if __name__ == "__main__":
    print("=== M192 videx_model_innodb self-test ===")
    
    model = InnoDBCostModel(buffer_pool_pages=4096, io_threads=4)
    
    # Test B-tree height
    h = model.estimate_btree_height(1000000, 20)
    assert 2 <= h <= 5
    
    # Test range scan
    cost_range = model.index_range_scan_cost(100000, 0.01, 20, 120)
    cost_full = model.full_table_scan_cost(100000, 120)
    assert cost_range < cost_full, "Range scan should be cheaper than full scan"
    
    # Test Fibonacci split
    split_cost = model.page_split_cost(0.95, 3)
    assert split_cost > 0
    no_split = model.page_split_cost(0.3, 3)
    assert no_split == 0.0
    
    # Test workload simulation
    result = model.simulate_workload(200, num_rows=50000)
    assert result["n_queries"] == 200
    assert result["mean_cost"] > 0
    assert result["buffer_pool"]["hit_ratio"] > 0
    
    snap = model._debug_snapshot()
    print(f"  BP hit ratio: {snap['buffer_pool']['hit_ratio']}")
    print(f"  Latency: {snap['latency']}")
    print("  All tests passed!")
    print(f"  Lines: {sum(1 for _ in open(__file__))}")
