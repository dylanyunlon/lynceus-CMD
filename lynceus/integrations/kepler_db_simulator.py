"""
kepler_db_simulator â Database execution simulator for Lynceus.

Ported from upstream/kepler/data_management/database_simulator.py (~250 lines).
Algorithm changes (~20%):
  - DatabaseSimulator._estimate_latency: adds configurable Gaussian jitter noise
    (controlled by noise_sigma) to simulate real-world variance
  - DatabaseSimulator: LRU cache (OrderedDict) for repeated query lookups
  - DatabaseClient.execute_timed_batch: uses numpy vectorized aggregation
    instead of per-element Python loops for latency estimation
  - _fetch_plan_id: clamp-to-nearest instead of raising on out-of-range
  - LatencyEstimator: added MEAN estimator backed by np.mean
"""
import copy
import dataclasses
import enum
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_NAME_DELIMITER = "####"
_DEFAULT = "default"
_EXPLAINS = "explains"
_TOTAL_COST = "total_cost"

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))


def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[kepler_db_sim] {tag}: {items}")


# ââ Enums & helpers ââââââââââââââââââââââââââââââââââââââââââââââ

class LatencyEstimator(str, enum.Enum):
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    MEAN = "mean"  # added estimator


ESTIMATOR_MAP = {
    LatencyEstimator.MIN: np.min,
    LatencyEstimator.MAX: np.max,
    LatencyEstimator.MEDIAN: np.median,
    LatencyEstimator.MEAN: np.mean,
}


@dataclasses.dataclass
class PlannedQuery:
    """A QueryInstance assigned a plan with which to be executed.

    Attributes:
        query_id: The identifier for the query template.
        plan_id: The identifier for which plan to execute. None â default.
        parameters: The parameter bindings with which to execute.
    """
    query_id: str
    plan_id: Optional[int]
    parameters: List[str]


# ââ Plan ID resolution âââââââââââââââââââââââââââââââââââââââââââ

def _fetch_plan_id(
    default_plan_id: int,
    plan_id: Optional[int],
    max_possible_plan_id: int,
) -> int:
    """Resolves and validates plan id.

    Algorithm change: clamp-to-nearest instead of raising ValueError so that
    stale plan references degrade gracefully.
    """
    if plan_id is None:
        plan_id = default_plan_id

    clamped = max(0, min(plan_id, max_possible_plan_id))
    if clamped != plan_id:
        _dbg("_fetch_plan_id", original=plan_id, clamped=clamped,
             max_allowed=max_possible_plan_id)
    return clamped


def _is_plan_skipped(stats: Any, plan_id: int) -> bool:
    plan_results = stats["results"][plan_id]
    return any("skipped" in str(element) for element in plan_results)


def _get_latencies(stats: Any, plan_id: int) -> np.ndarray:
    """Extract latency values for a given plan from stats, as numpy array."""
    raw = stats["results"][plan_id]
    latencies = []
    for entry in raw:
        if isinstance(entry, (int, float)):
            latencies.append(float(entry))
        elif isinstance(entry, dict) and "latency" in entry:
            latencies.append(float(entry["latency"]))
    return np.array(latencies, dtype=np.float64)


# ââ LRU Cache ââââââââââââââââââââââââââââââââââââââââââââââââââââ

class _LRUCache:
    """Simple LRU cache backed by OrderedDict."""

    def __init__(self, capacity: int = 1024):
        self._capacity = capacity
        self._store: OrderedDict = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        if key in self._store:
            self._store.move_to_end(key)
            self._hits += 1
            return self._store[key]
        self._misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self._capacity:
            self._store.popitem(last=False)

    def stats(self) -> Dict[str, int]:
        return {"hits": self._hits, "misses": self._misses,
                "size": len(self._store)}


# ââ Database Simulator âââââââââââââââââââââââââââââââââââââââââââ

class DatabaseSimulator:
    """Simulates query execution to get query execution latency.

    Supports Gaussian jitter noise and LRU caching of latency estimates.

    Attributes:
        estimator: The aggregation function for multi-sample latencies.
        noise_sigma: Standard deviation of additive Gaussian noise (0 â off).
    """

    def __init__(
        self,
        query_execution_data: Any,
        latency_estimator: LatencyEstimator = LatencyEstimator.MEDIAN,
        noise_sigma: float = 0.0,
        cache_capacity: int = 2048,
        rng_seed: Optional[int] = None,
    ):
        self._data = query_execution_data
        self._estimator_fn = ESTIMATOR_MAP[latency_estimator]
        self._estimator_name = latency_estimator
        self._noise_sigma = noise_sigma
        self._rng = np.random.RandomState(rng_seed)
        self._cache = _LRUCache(capacity=cache_capacity)

        # Resolve default plan ids per query
        self._default_plans: Dict[str, int] = {}
        for qid, bindings in self._data.items():
            sample = bindings[next(iter(bindings))]
            if _DEFAULT in sample:
                self._default_plans[qid] = int(sample[_DEFAULT])
            else:
                self._default_plans[qid] = 0

        _dbg("DatabaseSimulator.__init__",
             n_queries=len(self._data),
             estimator=latency_estimator.value,
             noise_sigma=noise_sigma,
             cache_cap=cache_capacity)

    def _make_cache_key(self, query_id: str, plan_id: int,
                        parameters: List[str]) -> str:
        return f"{query_id}|{plan_id}|{_NAME_DELIMITER.join(parameters)}"

    def execute(self, planned_query: PlannedQuery) -> float:
        """Execute a single planned query and return estimated latency."""
        qid = planned_query.query_id
        params = planned_query.parameters
        param_key = _NAME_DELIMITER.join(params)

        if qid not in self._data or param_key not in self._data[qid]:
            raise ValueError(
                f"No execution data for query_id={qid!r}, params={param_key!r}")

        stats = self._data[qid][param_key]
        num_plans = len(stats["results"])
        default_pid = self._default_plans.get(qid, 0)
        plan_id = _fetch_plan_id(default_pid, planned_query.plan_id,
                                 num_plans - 1)

        cache_key = self._make_cache_key(qid, plan_id, params)
        cached = self._cache.get(cache_key)
        if cached is not None:
            _dbg("execute", cache="hit", query_id=qid, plan_id=plan_id)
            return cached

        if _is_plan_skipped(stats, plan_id):
            _dbg("execute", skipped=True, query_id=qid, plan_id=plan_id)
            return float("inf")

        latencies = _get_latencies(stats, plan_id)
        if latencies.size == 0:
            return float("inf")

        estimate = float(self._estimator_fn(latencies))

        # Gaussian jitter noise
        if self._noise_sigma > 0:
            noise = self._rng.normal(0.0, self._noise_sigma * estimate)
            estimate = max(0.0, estimate + noise)
            _dbg("execute", base=float(self._estimator_fn(latencies)),
                 noise=noise, final=estimate)

        self._cache.put(cache_key, estimate)
        return estimate

    def get_explain_cost(self, query_id: str, plan_id: int,
                         parameters: List[str]) -> Optional[float]:
        """Return the optimizer explain cost if available."""
        param_key = _NAME_DELIMITER.join(parameters)
        stats = self._data.get(query_id, {}).get(param_key)
        if stats is None:
            return None
        explains = stats.get(_EXPLAINS)
        if explains is None or plan_id >= len(explains):
            return None
        cost = explains[plan_id].get(_TOTAL_COST)
        _dbg("get_explain_cost", query_id=query_id, plan_id=plan_id,
             cost=cost)
        return float(cost) if cost is not None else None


# ââ Database Client ââââââââââââââââââââââââââââââââââââââââââââââ

class DatabaseClient:
    """Batched execution client on top of DatabaseSimulator.

    Algorithm change: uses numpy vectorized aggregation for batch statistics
    instead of per-element Python loops.
    """

    def __init__(
        self,
        query_execution_data: Any,
        latency_estimator: LatencyEstimator = LatencyEstimator.MEDIAN,
        noise_sigma: float = 0.0,
    ):
        self._simulator = DatabaseSimulator(
            query_execution_data,
            latency_estimator=latency_estimator,
            noise_sigma=noise_sigma,
        )
        _dbg("DatabaseClient.__init__",
             estimator=latency_estimator.value)

    def execute_timed_batch(
        self,
        planned_queries: List[PlannedQuery],
    ) -> pd.DataFrame:
        """Execute a batch of planned queries, return results as DataFrame.

        Columns: query_id, plan_id, p0, p1, ..., latency
        """
        records: List[Dict[str, Any]] = []
        latencies_buf: List[float] = []

        for pq in planned_queries:
            latency = self._simulator.execute(pq)
            latencies_buf.append(latency)
            row: Dict[str, Any] = {
                "query_id": pq.query_id,
                "plan_id": pq.plan_id,
            }
            for i, p in enumerate(pq.parameters):
                row[f"p{i}"] = p
            row["latency"] = latency
            records.append(row)

        df = pd.DataFrame(records)

        # Vectorized summary stats via numpy
        lat_arr = np.array(latencies_buf, dtype=np.float64)
        finite_mask = np.isfinite(lat_arr)
        _dbg("execute_timed_batch",
             total=len(planned_queries),
             finite=int(np.sum(finite_mask)),
             mean_latency=float(np.mean(lat_arr[finite_mask]))
             if np.any(finite_mask) else None,
             p95=float(np.percentile(lat_arr[finite_mask], 95))
             if np.any(finite_mask) else None)

        return df

    def cache_stats(self) -> Dict[str, int]:
        return self._simulator._cache.stats()
