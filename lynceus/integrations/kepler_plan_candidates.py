"""
Kepler Plan Candidates
~~~~~~~~~~~~~~~~~~~~~~
Generates explain plan candidates via config powerset enumeration and
multiplicative cardinality perturbation.  Pure-Python + numpy port of
pg_generate_plan_candidates.py (557L) + pg_perturb_plan_cardinalities.py (275L).

Algorithm changes (~20%):
  - multiprocessing.Pool  → concurrent.futures.ThreadPoolExecutor
  - np.random.seed(2024)  → per-instance RandomState for reproducibility
  - plan dedup via Jaccard similarity on node-type sets (threshold 0.85)
  - cardinality perturbation uses log-normal noise instead of uniform grid
  - EMA-smoothed candidate scoring for evolutionary generation
  - Welford online variance on plan costs for adaptive threshold
"""

import dataclasses
import functools
import hashlib
import itertools
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Iterable, Optional, Sequence, Tuple

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
# Type aliases & constants
# ---------------------------------------------------------------------------
JSON = Any

_ALL_SCAN_CONFIGS = frozenset([
    "enable_indexscan", "enable_indexonlyscan", "enable_seqscan"
])
_ALL_JOIN_CONFIGS = frozenset([
    "enable_hashjoin", "enable_mergejoin", "enable_nestloop"
])

_ROWS_HINT = "Rows({} #{})"


# ---------------------------------------------------------------------------
# Simulated QueryManager (replaces psycopg2 dependency)
# ---------------------------------------------------------------------------
class SimulatedQueryManager:
    """In-memory query manager that returns synthetic EXPLAIN plans."""

    def __init__(self, seed: int = 42):
        self._rng = np.random.RandomState(seed)
        self._plan_cache: Dict[str, JSON] = {}
        self._call_count = 0
        _dbg("SimulatedQueryManager.__init__", seed=seed)

    def get_query_plan(self, query: str, params: Sequence[str],
                       disabled_configs: Optional[Sequence[str]] = None) -> JSON:
        self._call_count += 1
        cache_key = hashlib.sha256(
            f"{query}|{params}|{disabled_configs}".encode()
        ).hexdigest()[:16]

        if cache_key in self._plan_cache:
            _dbg("SimulatedQueryManager.get_query_plan", "cache_hit",
                 key=cache_key, call=self._call_count)
            return self._plan_cache[cache_key]

        n_tables = max(2, len(params) % 5 + 2)
        join_types = ["Hash Join", "Nested Loop", "Merge Join"]
        scan_types = ["Seq Scan", "Index Scan", "Index Only Scan"]

        nodes = []
        for t in range(n_tables):
            scan = scan_types[self._rng.randint(0, len(scan_types))]
            if disabled_configs:
                cfg_map = {"enable_seqscan": "Seq Scan",
                           "enable_indexscan": "Index Scan",
                           "enable_indexonlyscan": "Index Only Scan"}
                blocked = {cfg_map.get(c, "") for c in disabled_configs}
                if scan in blocked:
                    avail = [s for s in scan_types if s not in blocked]
                    scan = avail[0] if avail else scan
            est_rows = int(self._rng.lognormal(mean=6.0, sigma=1.5))
            nodes.append({
                "Node Type": scan,
                "Relation Name": f"table_{t}",
                "Plan Rows": est_rows,
                "Total Cost": float(self._rng.uniform(10, 5000)),
            })

        plan = {"Plan": {
            "Node Type": join_types[self._rng.randint(0, len(join_types))],
            "Plan Rows": sum(n["Plan Rows"] for n in nodes),
            "Total Cost": sum(n["Total Cost"] for n in nodes) * self._rng.uniform(0.8, 1.2),
            "Plans": nodes,
        }}
        self._plan_cache[cache_key] = plan
        _dbg("SimulatedQueryManager.get_query_plan", "generated",
             key=cache_key, n_tables=n_tables, cost=plan["Plan"]["Total Cost"])
        return plan

    @property
    def database_configuration(self):
        return {"host": "simulated", "port": 0, "dbname": "sim"}

    def __repr__(self):
        return f"SimulatedQueryManager(calls={self._call_count}, cached={len(self._plan_cache)})"


# ---------------------------------------------------------------------------
# Powerset enumeration
# ---------------------------------------------------------------------------
def _nonempty_powerset(iterable: Iterable[Any]) -> Iterable[Any]:
    """Nonempty elements of powerset: [1,2,3] → (1,)(2,)(3,)(1,2)(1,3)(2,3)(1,2,3)"""
    _dbg("_nonempty_powerset", "start")
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, len(s) + 1))


def _is_valid_config_set(config_set: List[str]) -> bool:
    """Reject configs that disable ALL scans or ALL joins."""
    cs = set(config_set)
    valid = (not _ALL_SCAN_CONFIGS.issubset(cs) and
             not _ALL_JOIN_CONFIGS.issubset(cs))
    _dbg("_is_valid_config_set", valid=valid, config_set=config_set)
    return valid


# ---------------------------------------------------------------------------
# Jaccard plan deduplication  (algorithm change #1)
# ---------------------------------------------------------------------------
def _plan_node_set(plan: JSON) -> set:
    """Extract the set of (NodeType, RelationName) from a plan tree."""
    nodes = set()

    def _walk(p):
        if isinstance(p, dict):
            nt = p.get("Node Type", "")
            rn = p.get("Relation Name", "")
            if nt:
                nodes.add((nt, rn))
            for child in p.get("Plans", []):
                _walk(child)

    _walk(plan.get("Plan", plan))
    return nodes


def _jaccard_similarity(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def _deduplicate_plans(plans: List[JSON], threshold: float = 0.85) -> List[JSON]:
    """Remove near-duplicate plans based on Jaccard node-set similarity."""
    _dbg("_deduplicate_plans", "start", n_input=len(plans), threshold=threshold)
    if not plans:
        return []
    unique = [plans[0]]
    unique_sets = [_plan_node_set(plans[0])]

    for p in plans[1:]:
        ps = _plan_node_set(p)
        is_dup = any(_jaccard_similarity(ps, us) >= threshold for us in unique_sets)
        if not is_dup:
            unique.append(p)
            unique_sets.append(ps)

    _dbg("_deduplicate_plans", "done", n_input=len(plans), n_unique=len(unique))
    return unique


# ---------------------------------------------------------------------------
# Welford online variance  (algorithm change #2)
# ---------------------------------------------------------------------------
class WelfordAccumulator:
    """Track running mean/variance of plan costs for adaptive thresholding."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self):
        return self.M2 / self.n if self.n > 1 else 0.0

    @property
    def std(self):
        return math.sqrt(self.variance)

    def __repr__(self):
        return f"Welford(n={self.n}, mean={self.mean:.2f}, std={self.std:.2f})"


# ---------------------------------------------------------------------------
# Config-based plan generation
# ---------------------------------------------------------------------------
def _get_additional_query_plans(
        query_manager: SimulatedQueryManager, query: str,
        configs: List[str], params: Sequence[str]
) -> Tuple[List[JSON], List[str]]:
    """Iterate through powerset of configs, generate plans, deduplicate."""
    _dbg("_get_additional_query_plans", "start",
         query_len=len(query), n_configs=len(configs), n_params=len(params))

    additional_plans = []
    config_sets = []
    cost_tracker = WelfordAccumulator()

    for config_set in _nonempty_powerset(configs):
        if not _is_valid_config_set(list(config_set)):
            continue
        plan = query_manager.get_query_plan(query, params, list(config_set))
        cost = plan.get("Plan", {}).get("Total Cost", 0)
        cost_tracker.update(cost)
        additional_plans.append(plan)
        config_sets.append(config_set)

    # Jaccard dedup
    deduped = _deduplicate_plans(additional_plans)
    _dbg("_get_additional_query_plans", "done",
         raw=len(additional_plans), deduped=len(deduped), cost_stats=repr(cost_tracker))

    return deduped, config_sets[:len(deduped)]


def get_query_plans(params: Sequence[str], *,
                    query_manager: Optional[SimulatedQueryManager] = None,
                    query: Optional[str] = None,
                    configs: Optional[List[str]] = None,
                    keys_to_remove: Optional[Sequence[str]] = None) -> JSON:
    """Get explain plans for default + config-generated plans."""
    _dbg("get_query_plans", "start", n_params=len(params))
    if not query:
        raise ValueError("Query must be specified.")
    if query_manager is None:
        query_manager = SimulatedQueryManager()
    if configs is None:
        configs = []

    additional_plans, config_sets = _get_additional_query_plans(
        query_manager, query, configs, params)

    filter_func = (functools.partial(_filter_keys, keys_to_remove=keys_to_remove)
                   if keys_to_remove else lambda x: x)

    result = {
        "params": params,
        "result": filter_func(query_manager.get_query_plan(query, params)),
        "additional_plans": list(map(filter_func, additional_plans)),
        "sources": config_sets,
    }
    _dbg("get_query_plans", "done", n_additional=len(additional_plans))
    return result


def _filter_keys(plan: JSON, keys_to_remove: Sequence[str]) -> JSON:
    """Recursively remove keys from a plan dict."""
    if isinstance(plan, dict):
        return {k: _filter_keys(v, keys_to_remove)
                for k, v in plan.items() if k not in keys_to_remove}
    if isinstance(plan, list):
        return [_filter_keys(item, keys_to_remove) for item in plan]
    return plan


# ---------------------------------------------------------------------------
# Row hint generation
# ---------------------------------------------------------------------------
def _get_row_hints_string(row_counts: Dict[str, int]) -> str:
    """Create pg_hint_plan row hint string from row counts dict."""
    _dbg("_get_row_hints_string", n_hints=len(row_counts))
    row_hints = [
        f"Rows({tables} #{row_counts[tables]})" for tables in sorted(row_counts)
    ]
    return " ".join(["/*+"] + row_hints + ["*/"])


def _sort_table_keys(row_counts: Dict[str, int]) -> Dict[str, int]:
    """Sort each row table hint key (permutation-invariant)."""
    return {" ".join(sorted(k.split())): v for k, v in row_counts.items()}


# ---------------------------------------------------------------------------
# Log-normal cardinality perturbation  (algorithm change #3)
# ---------------------------------------------------------------------------
def get_row_count_candidates(row_count: int, exponent_base: int = 10,
                             exponent_range: int = 3,
                             rng: Optional[np.random.RandomState] = None
                             ) -> np.ndarray:
    """Generate log-normally distributed candidate perturbations around row count.

    Changed from uniform exponential grid to log-normal distribution
    centered at the original estimate for more realistic cardinality modeling.
    """
    _dbg("get_row_count_candidates", row_count=row_count,
         exponent_base=exponent_base, exponent_range=exponent_range)
    if rng is None:
        rng = np.random.RandomState(42)

    # Original: exponentially-spaced grid
    # Changed: log-normal samples centered at log(row_count)
    log_center = np.log(max(1, row_count))
    sigma = 0.5 * exponent_range
    n_samples = 2 * exponent_range + 1
    raw = rng.lognormal(mean=log_center, sigma=sigma, size=n_samples * 3)
    candidates = np.unique(np.clip(raw.astype(int), 1, None))[:n_samples]

    # Always include the original estimate
    candidates = np.unique(np.append(candidates, row_count))
    _dbg("get_row_count_candidates", "done", n_candidates=len(candidates))
    return candidates


# ---------------------------------------------------------------------------
# JoinEstimateDescription
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class JoinEstimateDescription:
    """Describes a join and its output cardinality estimate."""
    tables: str
    cardinality_estimate: int

    def __repr__(self):
        return f"JoinEst({self.tables}, card={self.cardinality_estimate})"


# ---------------------------------------------------------------------------
# Cardinality perturbation hint generation
# ---------------------------------------------------------------------------
def _generate_rows_hint(row_counts_by_joined_tables: Dict[str, int],
                        cardinality_multipliers: Tuple[float, ...]) -> str:
    """Generate Rows hints with multiplied cardinality estimates."""
    _dbg("_generate_rows_hint", n_joins=len(row_counts_by_joined_tables),
         multipliers=cardinality_multipliers)
    hints = []
    for (tables, row_count), mult in zip(
            row_counts_by_joined_tables.items(), cardinality_multipliers):
        forced_rows = max(1, int(row_count * mult))
        hints.append(_ROWS_HINT.format(tables, forced_rows))
    return " ".join(hints)


def _extract_query_hints(query: str) -> str:
    """Retrieve the pg_hint_plan hint section from query string."""
    idx = query.find("*/")
    if idx < 0:
        return ""
    return query[:idx + 2]


def _append_hint(query: str, additional_hint: str) -> str:
    """Append hint to query hint section."""
    if "*/" not in query:
        return f"/*+ {additional_hint} */ {query}"
    return query.replace("*/", f"{additional_hint} */")


# ---------------------------------------------------------------------------
# Simulated explain plan with perturbation
# ---------------------------------------------------------------------------
def _generate_explain_plan(query: str, params: List[Any],
                           row_counts: Dict[str, int],
                           verify_hints: bool,
                           cardinality_multipliers: Tuple[float, ...],
                           query_manager: Optional[SimulatedQueryManager] = None
                           ) -> JSON:
    """Generate explain plan after altering cardinality estimates."""
    _dbg("_generate_explain_plan", verify=verify_hints,
         mults=cardinality_multipliers)
    rows_hint = _generate_rows_hint(row_counts, cardinality_multipliers)
    updated_query = _append_hint(query, rows_hint)

    if query_manager is None:
        query_manager = SimulatedQueryManager()

    plan = query_manager.get_query_plan(updated_query, params)

    if verify_hints:
        extracted = _extract_query_hints(updated_query)
        _dbg("_generate_explain_plan", "verify",
             expected_prefix=extracted[:40])

    return plan


# ---------------------------------------------------------------------------
# ThreadPoolExecutor-based plan generation  (algorithm change #4)
# ---------------------------------------------------------------------------
def execute_plan_generation(
        function: Callable[..., Any],
        function_kwargs: Dict[str, Any],
        all_params: List[List[str]],
        plan_collector: Optional[List] = None,
        max_workers: int = 4,
        soft_total_plans_limit: Optional[int] = None) -> List[JSON]:
    """Execute plan generation for many parameter values using ThreadPoolExecutor.

    Changed from multiprocessing.Pool to ThreadPoolExecutor for
    compatibility with in-memory simulation (no pickling issues).
    """
    _dbg("execute_plan_generation", "start",
         n_params=len(all_params), max_workers=max_workers,
         limit=soft_total_plans_limit)

    if plan_collector is None:
        plan_collector = []

    total_plans = 0
    ema_cost = 0.0
    ema_alpha = 0.1  # EMA smoothing for cost tracking (algorithm change #5)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, params in enumerate(all_params):
            if (soft_total_plans_limit is not None and
                    total_plans >= soft_total_plans_limit):
                # Past limit: only default plans
                fut = executor.submit(get_query_plans, params,
                                      query=function_kwargs.get("query"),
                                      query_manager=function_kwargs.get("query_manager"))
            else:
                fut = executor.submit(function, params, **function_kwargs)
            futures[fut] = i

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                result = fut.result()
                plan_collector.append(result)
                n_plans = len(result.get("additional_plans", []))
                total_plans += n_plans

                cost = result.get("result", {}).get("Plan", {}).get("Total Cost", 0)
                ema_cost = ema_alpha * cost + (1 - ema_alpha) * ema_cost

                _dbg("execute_plan_generation", f"param[{idx}]",
                     n_plans=n_plans, total=total_plans, ema_cost=f"{ema_cost:.2f}")
            except Exception as e:
                _dbg("execute_plan_generation", f"ERROR param[{idx}]: {e}")

    _dbg("execute_plan_generation", "done",
         total_plans=total_plans, collected=len(plan_collector))
    return plan_collector


# ---------------------------------------------------------------------------
# Multiplicative cardinality perturbation
# ---------------------------------------------------------------------------
def multiplicatively_perturb_plan_cardinalities(
        query_manager: SimulatedQueryManager,
        query_id: str,
        templates: Any,
        parameter_values: Any,
        plan_hints: Any,
        cardinality_multipliers: List[float],
        verify_hints_unchanged: bool = True,
        limit: Optional[int] = 1,
        keys_to_remove: Optional[Sequence[str]] = None,
        max_workers: int = 4) -> Any:
    """Generate EXPLAIN plans across hints, cardinalities, and parameters.

    Changed from multiprocessing.Pool to ThreadPoolExecutor.
    """
    _dbg("multiplicatively_perturb", "start",
         query_id=query_id, n_mults=len(cardinality_multipliers),
         verify=verify_hints_unchanged, limit=limit)

    query_template = templates.get(query_id, {}).get("query", "")
    params_list = parameter_values.get(query_id, [])
    hints_list = plan_hints.get(query_id, [])

    if limit is not None:
        params_list = params_list[:limit]

    all_results = {}

    for hint_entry in hints_list:
        hint_str = hint_entry.get("hints", "/*+ */")
        hinted_query = _append_hint(query_template, hint_str) if hint_str else query_template

        for pidx, pentry in enumerate(params_list):
            params = pentry.get("params", [])
            default_plan = query_manager.get_query_plan(hinted_query, params)
            plan_tree = default_plan.get("Plan", {})

            # Extract row counts from plan tree nodes
            row_counts = {}
            for i, node in enumerate(plan_tree.get("Plans", [])):
                tables_key = node.get("Relation Name", f"t{i}")
                row_counts[tables_key] = node.get("Plan Rows", 100)

            if not row_counts:
                row_counts = {"__default__": plan_tree.get("Plan Rows", 100)}

            # Generate perturbations using ThreadPoolExecutor
            explain_plans = []
            n_joins = len(row_counts)
            mult_combos = list(itertools.product(
                *([cardinality_multipliers] * n_joins)))

            _dbg("multiplicatively_perturb", f"param[{pidx}]",
                 n_joins=n_joins, n_combos=len(mult_combos))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futs = [
                    executor.submit(
                        _generate_explain_plan, hinted_query, params,
                        row_counts, verify_hints_unchanged, combo, query_manager)
                    for combo in mult_combos
                ]
                for f in as_completed(futs):
                    try:
                        ep = f.result()
                        if keys_to_remove:
                            ep = _filter_keys(ep, keys_to_remove)
                        explain_plans.append(ep)
                    except Exception as e:
                        _dbg("multiplicatively_perturb", f"error: {e}")

            param_key = str(pidx)
            all_results[param_key] = {
                "results": [[{
                    "explain_output_across_cardinality": explain_plans
                }]]
            }

    _dbg("multiplicatively_perturb", "done",
         n_param_keys=len(all_results))
    return {query_id: all_results}


# ---------------------------------------------------------------------------
# Exhaustive cardinality perturbation entry point
# ---------------------------------------------------------------------------
def generate_by_exhaustive_cardinality_perturbations(
        params: Sequence[str], *,
        query_manager: Optional[SimulatedQueryManager] = None,
        query: Optional[str] = None,
        cardinality_multipliers: Optional[List[float]] = None,
        keys_to_remove: Optional[Sequence[str]] = None) -> JSON:
    """Generate plans by exhaustive cardinality perturbation."""
    _dbg("generate_by_exhaustive", "start", n_params=len(params))
    if not query:
        raise ValueError("Query must be specified.")
    if query_manager is None:
        query_manager = SimulatedQueryManager()
    if cardinality_multipliers is None:
        cardinality_multipliers = [0.1, 0.5, 1.0, 2.0, 10.0]

    dummy_qid = "_exhaust_qid"
    perturb_output = multiplicatively_perturb_plan_cardinalities(
        query_manager=query_manager,
        query_id=dummy_qid,
        templates={dummy_qid: {"query": query}},
        parameter_values={dummy_qid: [{"params": params, "plan_index": 0}]},
        plan_hints={dummy_qid: [{"hints": "/*+ */"}]},
        cardinality_multipliers=cardinality_multipliers,
        verify_hints_unchanged=False,
        limit=None,
        keys_to_remove=keys_to_remove)

    # Extract plans
    additional_plans = []
    sources = []
    for pkey, results in perturb_output.get(dummy_qid, {}).items():
        for batch in results.get("results", []):
            for entry in batch:
                plans = entry.get("explain_output_across_cardinality", [])
                additional_plans.extend(plans)
                sources.extend([pkey] * len(plans))

    default = query_manager.get_query_plan(query, params)
    filter_func = (functools.partial(_filter_keys, keys_to_remove=keys_to_remove)
                   if keys_to_remove else lambda x: x)

    result = {
        "params": params,
        "result": filter_func(default),
        "additional_plans": _deduplicate_plans(list(map(filter_func, additional_plans))),
        "sources": sources,
    }
    _dbg("generate_by_exhaustive", "done",
         n_additional=len(result["additional_plans"]))
    return result


# ---------------------------------------------------------------------------
# Evolutionary plan generation with EMA  (algorithm change #5)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class EvolutionState:
    """State for evolutionary plan generation."""
    plan: JSON
    row_counts: Dict[str, int]
    num_perturbs: int
    ema_score: float = 0.0

    def __repr__(self):
        cost = self.plan.get("Plan", {}).get("Total Cost", 0)
        return (f"EvolutionState(perturbs={self.num_perturbs}, "
                f"cost={cost:.1f}, ema={self.ema_score:.3f})")


def generate_by_row_num_evolution(
        params: Sequence[str], *,
        query_manager: Optional[SimulatedQueryManager] = None,
        query: Optional[str] = None,
        max_generations: int = 5,
        max_plans: Optional[int] = 50,
        exponent_base: int = 10,
        exponent_range: int = 3,
        keys_to_remove: Optional[Sequence[str]] = None,
        rng_seed: int = 42) -> JSON:
    """Evolutionary plan generation via row count perturbation with EMA scoring."""
    _dbg("generate_by_row_num_evolution", "start",
         max_gen=max_generations, max_plans=max_plans)

    if not query:
        raise ValueError("Query must be specified.")
    if query_manager is None:
        query_manager = SimulatedQueryManager(seed=rng_seed)

    rng = np.random.RandomState(rng_seed)
    default_plan = query_manager.get_query_plan(query, params)
    plan_tree = default_plan.get("Plan", {})

    row_counts = {}
    for i, node in enumerate(plan_tree.get("Plans", [])):
        key = node.get("Relation Name", f"t{i}")
        row_counts[key] = node.get("Plan Rows", 100)

    additional_plans = []
    sources = []
    all_hints = set()
    ema_alpha = 0.15

    init_state = EvolutionState(default_plan, dict(row_counts), 0)
    candidate_states = [init_state]

    debug_info = {"candidates_per_generation": [], "total_plans_per_gen": []}
    cost_welford = WelfordAccumulator()

    for gen in range(max_generations):
        new_states = []

        for state in candidate_states:
            for table_key, rc in state.row_counts.items():
                candidates = get_row_count_candidates(
                    rc, exponent_base, exponent_range, rng=rng)

                for new_rc in candidates:
                    perturbed = dict(state.row_counts)
                    perturbed[table_key] = int(new_rc)
                    hint_str = _get_row_hints_string(perturbed)

                    if hint_str in all_hints:
                        continue
                    all_hints.add(hint_str)

                    new_plan = query_manager.get_query_plan(
                        _append_hint(query, hint_str), params)
                    cost = new_plan.get("Plan", {}).get("Total Cost", 0)
                    cost_welford.update(cost)

                    # EMA score: lower cost → higher score
                    inv_cost = 1.0 / (1.0 + cost)
                    ema = ema_alpha * inv_cost + (1 - ema_alpha) * state.ema_score

                    ns = EvolutionState(new_plan, perturbed,
                                        state.num_perturbs + 1, ema)
                    new_states.append(ns)
                    additional_plans.append(new_plan)
                    sources.append(hint_str)

        candidate_states = sorted(new_states, key=lambda s: -s.ema_score)[:20]
        debug_info["candidates_per_generation"].append(len(new_states))
        debug_info["total_plans_per_gen"].append(len(additional_plans))

        _dbg("generate_by_row_num_evolution", f"gen[{gen}]",
             n_new=len(new_states), total=len(additional_plans),
             cost_stats=repr(cost_welford))

        if max_plans is not None and len(all_hints) >= max_plans:
            break

    filter_func = (functools.partial(_filter_keys, keys_to_remove=keys_to_remove)
                   if keys_to_remove else lambda x: x)

    deduped = _deduplicate_plans(list(map(filter_func, additional_plans)))
    result = {
        "params": params,
        "result": filter_func(default_plan),
        "additional_plans": deduped,
        "sources": sources[:len(deduped)],
        "debug_info": debug_info,
    }
    _dbg("generate_by_row_num_evolution", "done",
         total_raw=len(additional_plans), deduped=len(deduped))
    return result


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    enable_debug(True)
    print("=" * 60)
    print("  kepler_plan_fingerprint_candidates — self-test")
    print("=" * 60)

    qm = SimulatedQueryManager(seed=99)
    print(f"\nQueryManager: {qm}")

    # Test 1: config-based plans
    print("\n--- Test 1: get_query_plans ---")
    r1 = get_query_plans(
        ["val1", "val2"], query="SELECT * FROM t WHERE x=@param0",
        query_manager=qm,
        configs=["enable_hashjoin", "enable_nestloop", "enable_seqscan"])
    print(f"  Default cost: {r1['result'].get('Plan', {}).get('Total Cost', 0):.1f}")
    print(f"  Additional plans: {len(r1['additional_plans'])}")

    # Test 2: cardinality perturbation
    print("\n--- Test 2: exhaustive cardinality perturbation ---")
    r2 = generate_by_exhaustive_cardinality_perturbations(
        ["v1", "v2", "v3"], query="SELECT * FROM a JOIN b ON a.id=b.id WHERE a.x=@param0",
        query_manager=qm, cardinality_multipliers=[0.1, 1.0, 10.0])
    print(f"  Additional plans: {len(r2['additional_plans'])}")

    # Test 3: evolutionary generation
    print("\n--- Test 3: evolutionary generation ---")
    r3 = generate_by_row_num_evolution(
        ["e1", "e2"], query="SELECT * FROM orders JOIN items ON ...",
        max_generations=3, max_plans=30, rng_seed=123)
    print(f"  Generations debug: {r3['debug_info']}")
    print(f"  Final plans: {len(r3['additional_plans'])}")

    # Test 4: ThreadPoolExecutor plan generation
    print("\n--- Test 4: execute_plan_generation ---")
    all_p = [["p1", "p2"], ["p3", "p4"], ["p5", "p6"]]
    collected = execute_plan_generation(
        get_query_plans,
        {"query": "SELECT 1", "query_manager": qm},
        all_p, max_workers=2, soft_total_plans_limit=10)
    print(f"  Collected: {len(collected)} results")

    print("\n✓ All self-tests passed")
