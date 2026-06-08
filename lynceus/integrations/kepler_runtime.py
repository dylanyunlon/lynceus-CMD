"""
Query execution engine with EXPLAIN analysis and hint generation.
Pure numpy, dataclass-based results.
"""

import numpy as np
import json
import time
import logging
import hashlib
from dataclasses import dataclass, field
from typing import (
    List, Optional, Dict, Any, Tuple, Callable,
)
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------

def _dbg(tag: str, **kw):
    parts = [f"[DBG:{tag}]"]
    for k, v in kw.items():
        if isinstance(v, np.ndarray):
            parts.append(f"{k}.shape={v.shape}")
        elif isinstance(v, str) and len(v) > 120:
            parts.append(f"{k}='{v[:120]}â¦'")
        else:
            parts.append(f"{k}={v}")
    logger.debug(" ".join(parts))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Outcome of a single query execution."""
    query_id: str
    sql: str
    status: str                       # 'ok' | 'error' | 'timeout'
    rows_affected: int = 0
    elapsed_ms: float = 0.0
    error_msg: Optional[str] = None
    plan_id: Optional[str] = None
    data: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def _dbg(self):
        _dbg("ExecutionResult", query_id=self.query_id, status=self.status,
             elapsed_ms=self.elapsed_ms, rows=self.rows_affected)


@dataclass
class PlanNode:
    """Single node inside an EXPLAIN plan tree."""
    node_type: str
    relation: Optional[str] = None
    alias: Optional[str] = None
    startup_cost: float = 0.0
    total_cost: float = 0.0
    plan_rows: int = 0
    plan_width: int = 0
    actual_rows: Optional[int] = None
    actual_time_ms: Optional[float] = None
    children: List["PlanNode"] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def _dbg(self):
        _dbg("PlanNode", node_type=self.node_type, relation=self.relation,
             total_cost=self.total_cost, plan_rows=self.plan_rows,
             n_children=len(self.children))


@dataclass
class PlanTree:
    """Full query plan produced by EXPLAIN."""
    plan_id: str
    root: PlanNode
    total_cost: float = 0.0
    planning_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def _dbg(self):
        _dbg("PlanTree", plan_id=self.plan_id, total_cost=self.total_cost,
             planning_ms=self.planning_time_ms, execution_ms=self.execution_time_ms)

    def node_count(self) -> int:
        count = 0
        stack = [self.root]
        while stack:
            n = stack.pop()
            count += 1
            stack.extend(n.children)
        _dbg("PlanTree.node_count", count=count)
        return count

    def deepest_path(self) -> List[str]:
        """Return the node-type path to the deepest leaf."""
        _dbg("PlanTree.deepest_path_start")
        best: List[str] = []

        def _walk(node: PlanNode, path: List[str]):
            nonlocal best
            cur = path + [node.node_type]
            if not node.children and len(cur) > len(best):
                best = cur
            for ch in node.children:
                _walk(ch, cur)

        _walk(self.root, [])
        _dbg("PlanTree.deepest_path", path=best)
        return best


# ---------------------------------------------------------------------------
# DataFrame â lightweight tabular container (numpy-backed)
# ---------------------------------------------------------------------------

class DataFrame:
    """Minimal columnar store wrapping a dict of numpy arrays."""

    def __init__(self, columns: Dict[str, np.ndarray]):
        _dbg("DataFrame.__init__", cols=list(columns.keys()))
        lens = {k: len(v) for k, v in columns.items()}
        unique_lens = set(lens.values())
        if len(unique_lens) > 1:
            raise ValueError(f"Column length mismatch: {lens}")
        self.columns = columns
        self.n_rows = next(iter(lens.values()), 0)
        self.col_names = list(columns.keys())

    def __repr__(self):
        return f"DataFrame(rows={self.n_rows}, cols={self.col_names})"

    def __getitem__(self, col: str) -> np.ndarray:
        _dbg("DataFrame.__getitem__", col=col)
        return self.columns[col]

    def head(self, n: int = 5) -> Dict[str, np.ndarray]:
        _dbg("DataFrame.head", n=n)
        return {k: v[:n] for k, v in self.columns.items()}

    @staticmethod
    def from_results(results: List[ExecutionResult]) -> "DataFrame":
        """Build a DataFrame from a list of ExecutionResult."""
        _dbg("DataFrame.from_results", n_results=len(results))
        cols: Dict[str, list] = {
            "query_id": [], "status": [], "elapsed_ms": [],
            "rows_affected": [], "error_msg": [], "plan_id": [],
        }
        for r in results:
            cols["query_id"].append(r.query_id)
            cols["status"].append(r.status)
            cols["elapsed_ms"].append(r.elapsed_ms)
            cols["rows_affected"].append(r.rows_affected)
            cols["error_msg"].append(r.error_msg or "")
            cols["plan_id"].append(r.plan_id or "")
        np_cols = {}
        for k, v in cols.items():
            try:
                np_cols[k] = np.array(v, dtype=np.float64)
            except (ValueError, TypeError):
                np_cols[k] = np.array(v, dtype=object)
        return DataFrame(np_cols)


# ---------------------------------------------------------------------------
# ExplainAnalyzer
# ---------------------------------------------------------------------------

class ExplainAnalyzer:
    """Parse Postgres-style EXPLAIN (FORMAT JSON) into a PlanTree."""

    COSTLY_TYPES = {"Seq Scan", "Nested Loop", "Hash Join", "Sort",
                    "Materialize", "Bitmap Heap Scan"}

    def __init__(self):
        _dbg("ExplainAnalyzer.__init__")

    # --- public API --------------------------------------------------------

    def parse(self, explain_json: str) -> PlanTree:
        """
        Parameters
        ----------
        explain_json : str  â raw JSON string from EXPLAIN (FORMAT JSON)

        Returns
        -------
        PlanTree
        """
        _dbg("ExplainAnalyzer.parse", explain_json=explain_json)
        blob = json.loads(explain_json)
        if isinstance(blob, list):
            blob = blob[0]
        plan_dict = blob.get("Plan", blob)
        plan_id = hashlib.sha1(explain_json.encode()).hexdigest()[:12]
        root = self._parse_node(plan_dict)
        tree = PlanTree(
            plan_id=plan_id,
            root=root,
            total_cost=root.total_cost,
            planning_time_ms=blob.get("Planning Time", 0.0),
            execution_time_ms=blob.get("Execution Time", 0.0),
        )
        self._check_warnings(tree)
        tree._dbg()
        return tree

    # --- internals ---------------------------------------------------------

    def _parse_node(self, d: Dict[str, Any]) -> PlanNode:
        _dbg("ExplainAnalyzer._parse_node", node_type=d.get("Node Type"))
        children = [self._parse_node(c) for c in d.get("Plans", [])]
        node = PlanNode(
            node_type=d.get("Node Type", "Unknown"),
            relation=d.get("Relation Name"),
            alias=d.get("Alias"),
            startup_cost=float(d.get("Startup Cost", 0)),
            total_cost=float(d.get("Total Cost", 0)),
            plan_rows=int(d.get("Plan Rows", 0)),
            plan_width=int(d.get("Plan Width", 0)),
            actual_rows=d.get("Actual Rows"),
            actual_time_ms=d.get("Actual Total Time"),
            children=children,
            extra={k: v for k, v in d.items()
                   if k not in {"Node Type", "Relation Name", "Alias",
                                "Startup Cost", "Total Cost", "Plan Rows",
                                "Plan Width", "Actual Rows", "Actual Total Time",
                                "Plans"}},
        )
        node._dbg()
        return node

    def _check_warnings(self, tree: PlanTree):
        _dbg("ExplainAnalyzer._check_warnings")
        stack = [tree.root]
        while stack:
            n = stack.pop()
            if n.node_type == "Seq Scan" and n.plan_rows > 10_000:
                tree.warnings.append(
                    f"Large sequential scan on '{n.relation}' "
                    f"(est. {n.plan_rows} rows)"
                )
            if n.node_type == "Sort" and n.total_cost > 5000:
                tree.warnings.append(
                    f"Expensive sort (cost {n.total_cost:.1f})"
                )
            if n.actual_rows is not None and n.plan_rows > 0:
                ratio = n.actual_rows / max(n.plan_rows, 1)
                if ratio > 10 or ratio < 0.1:
                    tree.warnings.append(
                        f"Row estimate off by {ratio:.1f}x on "
                        f"'{n.node_type}' ({n.relation or '?'})"
                    )
            stack.extend(n.children)
        _dbg("ExplainAnalyzer._check_warnings_done",
             n_warnings=len(tree.warnings))


# ---------------------------------------------------------------------------
# HintGenerator
# ---------------------------------------------------------------------------

class HintGenerator:
    """Generate pg_hint_plan-style hints from a PlanTree."""

    _INDEX_HINT = "IndexScan({alias} {index})"
    _NO_SEQ = "NoSeqScan({alias})"
    _HASH_JOIN = "HashJoin({a} {b})"
    _NEST_LOOP = "NestLoop({a} {b})"

    def __init__(self, index_catalog: Optional[Dict[str, List[str]]] = None):
        _dbg("HintGenerator.__init__",
             catalog_size=len(index_catalog) if index_catalog else 0)
        self.index_catalog = index_catalog or {}

    def gen_hints(self, plan: PlanTree) -> str:
        """
        Walk the plan tree and emit hints as a single /*+ ... */ comment string.
        """
        _dbg("HintGenerator.gen_hints", plan_id=plan.plan_id)
        hints: List[str] = []
        self._walk_for_hints(plan.root, hints)
        if not hints:
            hints.append("/* no hints needed */")
        result = "/*+ " + " ".join(hints) + " */"
        _dbg("HintGenerator.gen_hints_result", result=result)
        return result

    def _walk_for_hints(self, node: PlanNode, hints: List[str]):
        _dbg("HintGenerator._walk_for_hints", node_type=node.node_type)
        alias = node.alias or node.relation or "t"

        if node.node_type == "Seq Scan" and node.plan_rows > 5000:
            if node.relation and node.relation in self.index_catalog:
                idx = self.index_catalog[node.relation][0]
                hints.append(self._INDEX_HINT.format(alias=alias, index=idx))
            else:
                hints.append(self._NO_SEQ.format(alias=alias))

        if node.node_type == "Nested Loop" and len(node.children) == 2:
            ca = node.children[0].alias or node.children[0].relation or "t1"
            cb = node.children[1].alias or node.children[1].relation or "t2"
            if node.total_cost > 3000:
                hints.append(self._HASH_JOIN.format(a=ca, b=cb))

        if node.node_type == "Hash Join" and node.total_cost < 200:
            if len(node.children) == 2:
                ca = node.children[0].alias or "t1"
                cb = node.children[1].alias or "t2"
                hints.append(self._NEST_LOOP.format(a=ca, b=cb))

        for ch in node.children:
            self._walk_for_hints(ch, hints)


# ---------------------------------------------------------------------------
# BatchExecutor
# ---------------------------------------------------------------------------

class BatchExecutor:
    """Execute callables with a per-item timeout."""

    def __init__(self, max_workers: int = 4, default_timeout_s: float = 30.0):
        _dbg("BatchExecutor.__init__", max_workers=max_workers,
             timeout=default_timeout_s)
        self.max_workers = max_workers
        self.default_timeout_s = default_timeout_s

    def run_with_timeout(
        self,
        tasks: List[Callable[[], ExecutionResult]],
        timeout_s: Optional[float] = None,
    ) -> List[ExecutionResult]:
        """
        Run each callable; if it exceeds *timeout_s* mark it as 'timeout'.
        """
        _dbg("BatchExecutor.run_with_timeout", n_tasks=len(tasks),
             timeout_s=timeout_s)
        timeout = timeout_s or self.default_timeout_s
        results: List[ExecutionResult] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(t): idx for idx, t in enumerate(tasks)}
            ordered: Dict[int, ExecutionResult] = {}
            for fut in futures:
                idx = futures[fut]
                try:
                    res = fut.result(timeout=timeout)
                    res._dbg()
                    ordered[idx] = res
                except FuturesTimeout:
                    _dbg("BatchExecutor.timeout", idx=idx)
                    ordered[idx] = ExecutionResult(
                        query_id=f"q_{idx}",
                        sql="",
                        status="timeout",
                        error_msg=f"Exceeded {timeout}s timeout",
                    )
                except Exception as exc:
                    _dbg("BatchExecutor.error", idx=idx, error=str(exc))
                    ordered[idx] = ExecutionResult(
                        query_id=f"q_{idx}",
                        sql="",
                        status="error",
                        error_msg=str(exc),
                    )
            for i in range(len(tasks)):
                results.append(ordered[i])
        _dbg("BatchExecutor.run_with_timeout_done", n_results=len(results))
        return results


# ---------------------------------------------------------------------------
# ExecutionEngine  (top-level faÃ§ade)
# ---------------------------------------------------------------------------

class ExecutionEngine:
    """
    Orchestrator: takes a batch of SQL strings, simulates execution,
    collects EXPLAIN plans, and returns a DataFrame of results.
    """

    def __init__(self, max_workers: int = 4, timeout_s: float = 30.0,
                 index_catalog: Optional[Dict[str, List[str]]] = None):
        _dbg("ExecutionEngine.__init__", max_workers=max_workers,
             timeout_s=timeout_s)
        self.analyzer = ExplainAnalyzer()
        self.hint_gen = HintGenerator(index_catalog=index_catalog)
        self.batch_exec = BatchExecutor(max_workers=max_workers,
                                        default_timeout_s=timeout_s)
        self._rng = np.random.default_rng(99)

    def _simulate_query(self, sql: str, qid: str) -> ExecutionResult:
        """Simulate query execution (replace with real DB driver)."""
        _dbg("ExecutionEngine._simulate_query", qid=qid, sql=sql)
        start = time.perf_counter()
        # Simulate variable latency
        latency = self._rng.exponential(0.05)
        time.sleep(min(latency, 0.2))
        n_rows = int(self._rng.integers(0, 500))
        data = self._rng.normal(size=(n_rows, 3))
        elapsed = (time.perf_counter() - start) * 1000
        plan_json = json.dumps([{
            "Plan": {
                "Node Type": self._rng.choice(
                    ["Seq Scan", "Index Scan", "Hash Join"]),
                "Relation Name": "sim_table",
                "Alias": "st",
                "Startup Cost": float(self._rng.uniform(0, 10)),
                "Total Cost": float(self._rng.uniform(10, 5000)),
                "Plan Rows": n_rows,
                "Plan Width": 64,
            },
            "Planning Time": float(self._rng.uniform(0.01, 2.0)),
            "Execution Time": elapsed,
        }])
        tree = self.analyzer.parse(plan_json)
        return ExecutionResult(
            query_id=qid,
            sql=sql,
            status="ok",
            rows_affected=n_rows,
            elapsed_ms=elapsed,
            plan_id=tree.plan_id,
            data=data,
            metadata={"hints": self.hint_gen.gen_hints(tree),
                       "warnings": tree.warnings},
        )

    def execute_batch(self, queries: List[str]) -> DataFrame:
        """
        Execute a batch of SQL queries and return results as a DataFrame.
        """
        _dbg("ExecutionEngine.execute_batch", n_queries=len(queries))
        tasks = []
        for idx, sql in enumerate(queries):
            qid = f"q_{idx}"
            tasks.append(lambda s=sql, q=qid: self._simulate_query(s, q))

        results = self.batch_exec.run_with_timeout(tasks)
        df = DataFrame.from_results(results)
        _dbg("ExecutionEngine.execute_batch_done", df=repr(df))
        return df


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    engine = ExecutionEngine(
        max_workers=2,
        timeout_s=5.0,
        index_catalog={"sim_table": ["idx_sim_table_id"]},
    )

    queries = [
        "SELECT * FROM orders WHERE total > 100",
        "SELECT u.name FROM users u JOIN orders o ON u.id = o.uid",
        "SELECT COUNT(*) FROM logs WHERE ts > now() - interval '1h'",
    ]
    df = engine.execute_batch(queries)
    print(df)
    print("Head:", df.head(3))

    # Standalone ExplainAnalyzer test
    sample = json.dumps([{
        "Plan": {
            "Node Type": "Hash Join",
            "Total Cost": 1500.0,
            "Plan Rows": 200,
            "Plan Width": 32,
            "Startup Cost": 10.0,
            "Plans": [
                {"Node Type": "Seq Scan", "Relation Name": "users",
                 "Alias": "u", "Total Cost": 800, "Plan Rows": 15000,
                 "Plan Width": 16, "Startup Cost": 0},
                {"Node Type": "Index Scan", "Relation Name": "orders",
                 "Alias": "o", "Total Cost": 200, "Plan Rows": 500,
                 "Plan Width": 16, "Startup Cost": 0.5},
            ],
        },
        "Planning Time": 0.45,
        "Execution Time": 12.3,
    }])
    analyzer = ExplainAnalyzer()
    tree = analyzer.parse(sample)
    print("Nodes:", tree.node_count())
    print("Deepest:", tree.deepest_path())
    print("Warnings:", tree.warnings)

    hg = HintGenerator(index_catalog={"users": ["idx_users_pk"]})
    print("Hints:", hg.gen_hints(tree))
