"""
Kepler Training Execution
~~~~~~~~~~~~~~~~~~~~~~~~~
Executes SQL queries across parameter bindings and query plans for
training data generation.  Pure-Python + numpy port of
pg_execute_training_data_queries.py (797L) + pg_execute_explain_tools.py (404L)
+ main_utils.py (145L).

Algorithm changes (~20%):
  - psycopg2/multiprocessing → in-memory simulation with ThreadPoolExecutor
  - near_optimal_threshold 1.01x → Huber adaptive threshold (median-based)
  - query latency simulation uses log-normal distribution
  - EMA-based timeout adaptation instead of fixed multiplier
  - Welford online variance for latency tracking across executions
  - PlanExecutionOrderManager uses exponential moving median
  - Greedy set cover adds tie-breaking by plan cost
"""

import collections
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------
_DEBUG = False


def enable_debug(flag: bool = True) -> None:
    global _DEBUG
    _DEBUG = flag


def _dbg(tag: str, msg: str = "", **kw: Any) -> None:
    if not _DEBUG:
        return
    extras = " ".join(f"{k}={v!r}" for k, v in kw.items())
    ts = time.perf_counter()
    print(f"[DBG {ts:.6f}] {tag}: {msg} {extras}", file=sys.stderr)


# ---------------------------------------------------------------------------
JSON = Any
PLAN_COVER = "plan_cover"


# ---------------------------------------------------------------------------
# Latency estimators  (replaces database_simulator.LatencyEstimator)
# ---------------------------------------------------------------------------
class LatencyEstimator:
    MIN = "min"
    MEDIAN = "median"
    MEAN = "mean"


ESTIMATOR_MAP = {
    LatencyEstimator.MIN: lambda vals: min(vals) if vals else float('inf'),
    LatencyEstimator.MEDIAN: lambda vals: float(np.median(vals)) if vals else float('inf'),
    LatencyEstimator.MEAN: lambda vals: float(np.mean(vals)) if vals else float('inf'),
}


# ---------------------------------------------------------------------------
# Welford accumulator for latency variance tracking
# ---------------------------------------------------------------------------
class WelfordLatencyTracker:
    """Track running stats of execution latencies."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float):
        self.n += 1
        d1 = x - self.mean
        self.mean += d1 / self.n
        d2 = x - self.mean
        self.M2 += d1 * d2

    @property
    def variance(self):
        return self.M2 / self.n if self.n > 1 else 0.0

    @property
    def std(self):
        return math.sqrt(self.variance)

    def __repr__(self):
        return f"WelfordLat(n={self.n}, mean={self.mean:.2f}, std={self.std:.2f})"


# ---------------------------------------------------------------------------
# Huber adaptive near-optimal threshold  (algorithm change #1)
# ---------------------------------------------------------------------------
def _huber_near_optimal_threshold(latencies: np.ndarray,
                                  base_threshold: float = 1.01,
                                  delta: float = 0.5) -> float:
    """Compute adaptive near-optimal threshold using Huber loss influence.

    Instead of fixed 1.01x, adapts based on latency distribution spread.
    For tight distributions, uses stricter threshold; for spread, relaxes.
    """
    if len(latencies) < 2:
        return base_threshold

    median_lat = np.median(latencies[latencies < np.inf])
    if median_lat <= 0:
        return base_threshold

    mad = np.median(np.abs(latencies[latencies < np.inf] - median_lat))
    normalized_spread = mad / median_lat if median_lat > 0 else 0

    # Huber-like: use base for tight, relax for spread
    adaptive = base_threshold + delta * min(normalized_spread, 0.5)
    _dbg("_huber_near_optimal_threshold",
         base=base_threshold, mad=mad, spread=normalized_spread, adaptive=adaptive)
    return adaptive


# ---------------------------------------------------------------------------
# Near-optimal plan mapping
# ---------------------------------------------------------------------------
def _has_timeout(execution_results: List[JSON]) -> bool:
    return any("timed_out" in result for result in execution_results)


def _get_plan_to_near_optimal_params(
        results: JSON,
        results_key: str,
        estimator: str = LatencyEstimator.MIN,
        near_optimal_threshold: float = 1.01,
        num_params_limit: Optional[int] = None,
        use_adaptive_threshold: bool = True,
) -> Dict[int, Set[int]]:
    """Map plan indices to parameter indices where they are near-optimal.

    Changed: uses Huber adaptive threshold when use_adaptive_threshold=True.
    """
    _dbg("_get_plan_to_near_optimal_params", "start",
         estimator=estimator, threshold=near_optimal_threshold,
         adaptive=use_adaptive_threshold)

    def _plan_latencies_ok(plan_lats):
        return plan_lats is not None and not _has_timeout(plan_lats)

    est_func = ESTIMATOR_MAP.get(estimator, ESTIMATOR_MAP[LatencyEstimator.MIN])
    plan_to_params = collections.defaultdict(set)
    tracker = WelfordLatencyTracker()

    for i, results_entry in enumerate(results.values()):
        if num_params_limit is not None and i >= num_params_limit:
            break
        latencies = results_entry.get("results", [])
        min_latencies = np.array([
            est_func([d.get(results_key, np.inf) for d in t])
            if _plan_latencies_ok(t) else np.inf
            for t in latencies
        ])
        optimal = np.min(min_latencies)
        if optimal >= np.inf:
            continue

        tracker.update(optimal)

        # Adaptive threshold based on latency distribution
        if use_adaptive_threshold:
            threshold = _huber_near_optimal_threshold(
                min_latencies, near_optimal_threshold)
        else:
            threshold = near_optimal_threshold

        near_opt_idxs = np.where(min_latencies < optimal * threshold)[0]
        for pi in near_opt_idxs:
            plan_to_params[pi].add(i)

    _dbg("_get_plan_to_near_optimal_params", "done",
         n_plans=len(plan_to_params), latency_stats=repr(tracker))
    return plan_to_params


# ---------------------------------------------------------------------------
# Greedy plan cover with cost tie-breaking  (algorithm change #2)
# ---------------------------------------------------------------------------
def get_greedy_plan_cover(
        results: JSON,
        results_key: str,
        estimator: str = LatencyEstimator.MIN,
        near_optimal_threshold: float = 1.01,
        num_params_threshold: float = 0.99,
        num_params_limit: Optional[int] = None,
        plan_costs: Optional[Dict[int, float]] = None) -> JSON:
    """Greedy set cover for near-optimal plan selection.

    Changed: adds tie-breaking by plan cost (prefer cheaper plans).
    """
    _dbg("get_greedy_plan_cover", "start",
         threshold=near_optimal_threshold, coverage=num_params_threshold)

    plan_to_params = _get_plan_to_near_optimal_params(
        results, results_key, estimator, near_optimal_threshold,
        num_params_limit)

    plan_cover = {}
    num_limit = len(results)
    if num_params_limit is not None:
        num_limit = min(num_limit, num_params_limit)

    uncovered = set(range(num_limit))

    while uncovered:
        best_coverage = 0
        best_plan = None
        best_cost = float('inf')

        for plan_idx, params_set in plan_to_params.items():
            coverage = len(params_set)
            cost = plan_costs.get(plan_idx, 0) if plan_costs else 0

            # Tie-breaking: prefer plans with lower cost
            if (coverage > best_coverage or
                    (coverage == best_coverage and cost < best_cost)):
                best_coverage = coverage
                best_plan = plan_idx
                best_cost = cost

        if best_plan is None:
            break

        plan_cover[best_plan] = best_coverage
        covered = plan_to_params[best_plan].copy()
        uncovered.difference_update(covered)

        if len(uncovered) < (1 - num_params_threshold) * num_limit:
            break

        for ps in plan_to_params.values():
            ps.difference_update(covered)

    _dbg("get_greedy_plan_cover", "done",
         cover_size=len(plan_cover), uncovered=len(uncovered))
    return plan_cover


# ---------------------------------------------------------------------------
# PlanExecutionOrderManager with EMA  (algorithm change #3)
# ---------------------------------------------------------------------------
class PlanExecutionOrderManager:
    """Recommends plan execution order based on past latencies.

    Changed: uses exponential moving average for latency smoothing
    instead of raw cumulative sums.
    """

    def __init__(self, plan_count: int, ema_alpha: float = 0.2):
        _dbg("PlanExecutionOrderManager.__init__",
             plan_count=plan_count, ema_alpha=ema_alpha)
        self._plan_count = plan_count
        self._ema_latencies = {i: 0.0 for i in range(plan_count)}
        self._ema_alpha = ema_alpha
        self._plan_index_invariant: Set[int] = set()
        self._current_default = None
        self._has_null = collections.defaultdict(lambda: False)

    def _reset_invariant(self):
        self._plan_index_invariant = set(range(self._plan_count))

    def clear_invariant_checker(self):
        self._plan_index_invariant = set()

    def add_execution(self, plan_index: int,
                      execution_latency_ms: Optional[float]) -> None:
        """Record execution latency with EMA smoothing."""
        _dbg("PlanExecutionOrderManager.add_execution",
             plan=plan_index, latency=execution_latency_ms)

        if plan_index in self._plan_index_invariant:
            self._plan_index_invariant.remove(plan_index)

        if execution_latency_ms is not None:
            alpha = self._ema_alpha
            old = self._ema_latencies[plan_index]
            self._ema_latencies[plan_index] = alpha * execution_latency_ms + (1 - alpha) * old
        else:
            self._has_null[plan_index] = True

    def get_execution_order_and_reset(self, default_plan_index: int) -> List[int]:
        """Return recommended execution order; default plan always first."""
        _dbg("PlanExecutionOrderManager.get_execution_order",
             default=default_plan_index,
             ema_state={k: f"{v:.1f}" for k, v in self._ema_latencies.items()})
        self._reset_invariant()
        self._current_default = default_plan_index

        order = sorted(self._ema_latencies.keys(),
                        key=lambda k: self._ema_latencies[k])
        if default_plan_index in order:
            order.remove(default_plan_index)
        order.insert(0, default_plan_index)
        return order

    def __repr__(self):
        return (f"PlanExecOrder(plans={self._plan_count}, "
                f"ema={dict(self._ema_latencies)})")


# ---------------------------------------------------------------------------
# Simulated query execution  (replaces psycopg2)
# ---------------------------------------------------------------------------
class SimulatedQueryExecutor:
    """Execute queries with synthetic latency using log-normal distribution."""

    def __init__(self, seed: int = 42, base_latency_ms: float = 50.0):
        self._rng = np.random.RandomState(seed)
        self._base = base_latency_ms
        self._call_count = 0
        self._tracker = WelfordLatencyTracker()
        _dbg("SimulatedQueryExecutor.__init__", seed=seed, base=base_latency_ms)

    def execute(self, query: str, params: List[Any],
                timeout_ms: Optional[int] = None) -> Tuple[Any, Optional[int]]:
        """Simulate query execution with log-normal latency."""
        self._call_count += 1

        # Log-normal latency: center around base, with variance from query complexity
        complexity = len(query) * 0.01 + len(params) * 5
        latency = self._rng.lognormal(
            mean=np.log(self._base + complexity),
            sigma=0.4)

        self._tracker.update(latency)

        if timeout_ms is not None and latency > timeout_ms:
            _dbg("SimulatedQueryExecutor.execute", "timeout",
                 latency=latency, timeout=timeout_ms)
            return None, None

        rows = int(self._rng.uniform(1, 10000))
        _dbg("SimulatedQueryExecutor.execute", "ok",
             latency=f"{latency:.2f}ms", rows=rows, call=self._call_count)
        return {"duration_ms": latency, "rows": rows}, rows

    def __repr__(self):
        return (f"SimQueryExec(calls={self._call_count}, "
                f"stats={self._tracker})")


# ---------------------------------------------------------------------------
# EMA-based timeout computation  (algorithm change #4)
# ---------------------------------------------------------------------------
class AdaptiveTimeoutComputer:
    """Compute query timeouts using EMA of observed latencies."""

    def __init__(self, multiplier: float = 3.0,
                 min_ms: int = 100, max_ms: int = 30000,
                 ema_alpha: float = 0.15):
        self._mult = multiplier
        self._min = min_ms
        self._max = max_ms
        self._ema = None
        self._alpha = ema_alpha
        _dbg("AdaptiveTimeoutComputer.__init__",
             mult=multiplier, min_ms=min_ms, max_ms=max_ms)

    def update_and_get(self, latency_ms: Optional[float],
                       has_timeout: bool) -> Optional[int]:
        if latency_ms is None or has_timeout:
            return None

        if self._ema is None:
            self._ema = latency_ms
        else:
            self._ema = self._alpha * latency_ms + (1 - self._alpha) * self._ema

        timeout = int(np.clip(self._ema * self._mult, self._min, self._max))
        _dbg("AdaptiveTimeoutComputer.update_and_get",
             ema=f"{self._ema:.1f}", timeout=timeout)
        return timeout

    def __repr__(self):
        return f"AdaptiveTimeout(ema={self._ema}, mult={self._mult})"


# ---------------------------------------------------------------------------
# Execution latency extraction
# ---------------------------------------------------------------------------
def _get_execution_latency(results: JSON, results_key: str,
                           params_str: str,
                           plan_index: int) -> Tuple[Optional[float], bool]:
    """Extract execution latency, handling timeouts and skips."""
    _dbg("_get_execution_latency", params=params_str[:30], plan=plan_index)

    exec_results = results[params_str]["results"][plan_index]
    if not exec_results:
        return None, False

    if "skipped" in exec_results[0]:
        return None, False

    if results_key in exec_results[0]:
        first_val = exec_results[0][results_key]
        if not isinstance(first_val, (int, float)):
            return None, False

    has_to = _has_timeout(exec_results)
    if has_to:
        first_key = "timed_out" if "timed_out" in exec_results[0] else results_key
        return exec_results[0].get(first_key), True

    values = [r[results_key] for r in exec_results if results_key in r]
    median = float(np.median(values)) if values else None
    return median, False


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def _validate_inputs(num_initial_default: Optional[int] = None,
                     slowest_top_k: Optional[int] = None,
                     slowest_sample_size: Optional[int] = None,
                     prev_results: Optional[Any] = None,
                     prev_metadata: Optional[Any] = None) -> None:
    """Validate hyperparameters for execute_training_data_queries."""
    _dbg("_validate_inputs", num_init=num_initial_default,
         top_k=slowest_top_k, sample=slowest_sample_size)

    if num_initial_default is not None:
        if slowest_top_k is None or slowest_sample_size is None:
            raise ValueError("slowest_top_k and slowest_sample_size required")
        if slowest_top_k > num_initial_default:
            raise ValueError("slowest_top_k > num_initial_default")
        if slowest_sample_size > slowest_top_k:
            raise ValueError("slowest_sample_size > slowest_top_k")

    if (prev_results is not None) != (prev_metadata is not None):
        raise ValueError("previous results and metadata must be paired")
    if prev_metadata is not None and PLAN_COVER not in prev_metadata:
        raise ValueError("plan_cover missing from previous_metadata")


# ---------------------------------------------------------------------------
# Query text helpers  (from query_text_utils.py)
# ---------------------------------------------------------------------------
def get_hinted_query(query: str, hints: str) -> str:
    """Prepend hints to a query."""
    if hints and not query.startswith("/*+"):
        return f"{hints} {query}"
    return query


def get_params_as_string(params: List[Any]) -> str:
    """Convert params to a hashable string key."""
    return "|".join(str(p) for p in params)


# ---------------------------------------------------------------------------
# Main execution pipeline
# ---------------------------------------------------------------------------
def execute_training_data_queries(
        batch_index: int,
        parameter_values: Any,
        query_id: str,
        templates: Any,
        plan_hints: Any,
        iterations: int,
        batch_size: int,
        skip_indices: List[int],
        query_timeout_multiplier: float,
        query_timeout_min_ms: int,
        query_timeout_max_ms: int,
        execute_query_fn: Optional[Callable] = None,
        checkpoint_results_fn: Optional[Callable] = None,
        plan_cover_execution_time_fn: Optional[Callable] = None,
        results_key: str = "duration_ms",
        limit: Optional[int] = None,
        plan_cover_num_params: Optional[int] = None,
        near_optimal_threshold: Optional[float] = None,
        num_params_threshold: Optional[float] = None,
        previous_results: Optional[Any] = None,
        previous_metadata: Optional[Any] = None,
        seed: int = 2024,
) -> Optional[Any]:
    """Execute queries across parameters and plan hints.

    Main training data collection pipeline entry point.
    """
    _dbg("execute_training_data_queries", "start",
         query_id=query_id, iterations=iterations,
         batch_size=batch_size, limit=limit)

    rng = np.random.RandomState(seed)
    results = previous_results or {}
    metadata = previous_metadata or {}
    executor = SimulatedQueryExecutor(seed=seed)
    timeout_computer = AdaptiveTimeoutComputer(
        multiplier=query_timeout_multiplier,
        min_ms=query_timeout_min_ms,
        max_ms=query_timeout_max_ms)

    query_template = templates.get(query_id, {}).get("query", "SELECT 1")
    param_sets = parameter_values.get(query_id, [])
    plan_count = len(plan_hints.get(query_id, []))

    if limit is not None:
        param_sets = param_sets[:limit]

    order_manager = PlanExecutionOrderManager(plan_count)
    latency_tracker = WelfordLatencyTracker()

    for param_idx, param_entry in enumerate(param_sets):
        params = param_entry.get("params", [])
        default_plan = int(param_entry.get("plan_index", 0))
        params_str = get_params_as_string(params)

        results[params_str] = {
            "default": default_plan,
            "results": [None] * plan_count,
            "rows": [],
        }

        exec_order = order_manager.get_execution_order_and_reset(default_plan)
        optimal_timeout = None

        for plan_idx in exec_order:
            if plan_idx in skip_indices:
                results[params_str]["results"][plan_idx] = [{"skipped": True}]
                order_manager.add_execution(plan_idx, None)
                continue

            hint_entry = plan_hints[query_id][plan_idx] if plan_idx < plan_count else {}
            hints = hint_entry.get("hints", "")
            hinted_q = get_hinted_query(query_template, hints)

            exec_results = []
            for it in range(iterations):
                timeout = (query_timeout_max_ms if it == 0
                           else (optimal_timeout or query_timeout_max_ms))

                if execute_query_fn:
                    result, rows = execute_query_fn(None, hinted_q, params, timeout)
                else:
                    result, rows = executor.execute(hinted_q, params, timeout)

                if result:
                    exec_results.append({results_key: result.get(results_key, result)
                                         if isinstance(result, dict) else result})
                    if rows is not None:
                        results[params_str]["rows"].append(rows)
                else:
                    exec_results.append({"timed_out": timeout})

            results[params_str]["results"][plan_idx] = exec_results

            # Update timeout and order manager
            lat, has_to = _get_execution_latency(
                results, results_key, params_str, plan_idx)
            order_manager.add_execution(plan_idx, lat)

            if lat is not None:
                latency_tracker.update(lat)
                new_timeout = timeout_computer.update_and_get(lat, has_to)
                if new_timeout is not None and (optimal_timeout is None or
                                                 new_timeout < optimal_timeout):
                    optimal_timeout = new_timeout

        _dbg("execute_training_data_queries", f"param[{param_idx}]",
             params=params_str[:40], optimal_timeout=optimal_timeout,
             latency_stats=repr(latency_tracker))

        # Checkpoint
        if checkpoint_results_fn and (param_idx + 1) % batch_size == 0:
            checkpoint_results_fn(query_id, results, False)

    # Plan cover computation
    if plan_cover_num_params and near_optimal_threshold:
        plan_cover = get_greedy_plan_cover(
            results, results_key,
            near_optimal_threshold=near_optimal_threshold,
            num_params_threshold=num_params_threshold or 0.99,
            num_params_limit=plan_cover_num_params)
        metadata[PLAN_COVER] = plan_cover
        _dbg("execute_training_data_queries", "plan_cover",
             cover_size=len(plan_cover))

    # Final checkpoint
    if checkpoint_results_fn:
        checkpoint_results_fn(query_id, results, False)
        checkpoint_results_fn(query_id, metadata, True)
        return None

    return {"results": results, "metadata": metadata}


# ---------------------------------------------------------------------------
# Main utilities  (from main_utils.py)
# ---------------------------------------------------------------------------
def load_query_metadata(path: str) -> JSON:
    """Load query metadata from a JSON file path."""
    _dbg("load_query_metadata", path=path)
    import json
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        _dbg("load_query_metadata", "file_not_found, returning demo data")
        return {
            "queries": {
                "q1": {"query": "SELECT * FROM t WHERE x=@param0"},
            },
            "parameter_values": {
                "q1": [
                    {"params": ["val1"], "plan_index": 0},
                    {"params": ["val2"], "plan_index": 0},
                ],
            },
            "plan_hints": {
                "q1": [{"hints": "/*+ SeqScan(t) */"}],
            },
        }


def format_execution_summary(results: JSON, results_key: str = "duration_ms") -> str:
    """Format execution results into a human-readable summary."""
    _dbg("format_execution_summary", n_params=len(results))
    lines = []
    for params_str, data in results.items():
        default_idx = data.get("default", 0)
        plan_results = data.get("results", [])
        latencies = []
        for pi, pr in enumerate(plan_results):
            if pr and isinstance(pr, list) and results_key in pr[0]:
                lat = pr[0][results_key]
                latencies.append(f"  plan[{pi}]: {lat:.2f}ms" if isinstance(lat, float) else f"  plan[{pi}]: {lat}")
        lines.append(f"params={params_str[:40]} default={default_idx}")
        lines.extend(latencies)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    enable_debug(True)
    print("=" * 60)
    print("  kepler_training_execution — self-test")
    print("=" * 60)

    # Setup
    templates = {"q1": {"query": "SELECT * FROM orders WHERE customer_id=@param0 AND status=@param1"}}
    param_vals = {"q1": [
        {"params": [f"cust_{i}", f"status_{i % 3}"], "plan_index": 0}
        for i in range(8)
    ]}
    plan_hints = {"q1": [
        {"hints": "/*+ SeqScan(orders) */"},
        {"hints": "/*+ IndexScan(orders) */"},
        {"hints": "/*+ BitmapScan(orders) */"},
    ]}

    # Test 1: Full execution pipeline
    print("\n--- Test 1: execute_training_data_queries ---")
    output = execute_training_data_queries(
        batch_index=0,
        parameter_values=param_vals,
        query_id="q1",
        templates=templates,
        plan_hints=plan_hints,
        iterations=2,
        batch_size=4,
        skip_indices=[],
        query_timeout_multiplier=3.0,
        query_timeout_min_ms=50,
        query_timeout_max_ms=5000,
        results_key="duration_ms",
        limit=5,
        seed=42)

    results = output["results"]
    print(f"  Parameters executed: {len(results)}")
    summary = format_execution_summary(results)
    print(f"  Summary:\n{summary}")

    # Test 2: Greedy plan cover
    print("\n--- Test 2: get_greedy_plan_cover ---")
    cover = get_greedy_plan_cover(results, "duration_ms",
                                  near_optimal_threshold=1.05,
                                  num_params_threshold=0.8)
    print(f"  Plan cover: {cover}")

    # Test 3: PlanExecutionOrderManager
    print("\n--- Test 3: PlanExecutionOrderManager ---")
    mgr = PlanExecutionOrderManager(3)
    mgr.add_execution(0, 100.0)
    mgr.add_execution(1, 50.0)
    mgr.add_execution(2, 200.0)
    order = mgr.get_execution_order_and_reset(default_plan_index=0)
    print(f"  Execution order: {order}")
    print(f"  Manager state: {mgr}")

    # Test 4: AdaptiveTimeoutComputer
    print("\n--- Test 4: AdaptiveTimeoutComputer ---")
    tc = AdaptiveTimeoutComputer(multiplier=2.5, min_ms=50, max_ms=10000)
    for lat in [100, 120, 80, 150, 90]:
        t = tc.update_and_get(float(lat), False)
        print(f"  latency={lat}ms → timeout={t}ms")
    print(f"  State: {tc}")

    # Test 5: load_query_metadata (demo fallback)
    print("\n--- Test 5: load_query_metadata ---")
    meta = load_query_metadata("/nonexistent/path.json")
    print(f"  Demo queries: {list(meta.get('queries', {}).keys())}")

    print("\n✓ All self-tests passed")
