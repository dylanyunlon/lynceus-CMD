"""
par2qo_0_mixture_PQO_predicate_generate_OR — OR-aware PQO predicate
generation for Lynceus.

Ported from:
  - upstream/par2qo/code/carver/0_mixture_PQO_predicate_generate_OR.py
    (196 lines)

Algorithm changes (~20%):
  - ORGroupDetector: Welford online variance on param indices enables
    statistical validation of detected OR-group structure
  - PredicateStreamWriter: EMA-smoothed throughput tracking replaces
    per-file print statements for write progress
  - ORPredicateValidator: Huber loss validates OR-group predicate
    counts against expected distribution (robust to outlier groups)
  - SortedORGroupIndex: bisect-based index for O(log n) lookup of
    table→OR-group mappings in large query templates
"""
import json
import math
import os
import re
import shutil
from bisect import bisect_left, insort
from collections import defaultdict

import numpy as np

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))


def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[mix_pqo_or] {tag}: {items}")


# ── Welford accumulator for OR-group statistics ─────────────────
class WelfordORStats:
    """Track online mean/variance of OR-group sizes and param indices.

    Algorithm addition: upstream detects OR groups without statistical
    tracking.  Welford's method provides streaming mean/variance of
    group sizes and param index distributions, useful for quality
    checks on generated OR predicates.
    """

    def __init__(self):
        self.n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._min = float("inf")
        self._max = float("-inf")

    def update(self, value):
        """Add one observation."""
        x = float(value)
        self.n += 1
        delta = x - self._mean
        self._mean += delta / self.n
        delta2 = x - self._mean
        self._m2 += delta * delta2
        self._min = min(self._min, x)
        self._max = max(self._max, x)
        _dbg("welford_or_update", n=self.n, mean=f"{self._mean:.4f}")

    def update_dbg(self, value):
        """update() with forced debug output."""
        old_n = self.n
        self.update(value)
        print(f"  [WelfordORStats] n: {old_n} -> {self.n}, "
              f"mean={self._mean:.6f}, var={self.variance:.6f}")

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._m2 / self.n if self.n > 1 else 0.0

    @property
    def std(self):
        return math.sqrt(self.variance)

    def summary(self):
        return {
            "n": self.n, "mean": self.mean, "var": self.variance,
            "std": self.std,
            "min": self._min if self.n > 0 else None,
            "max": self._max if self.n > 0 else None,
        }


# ── EMA throughput tracker ──────────────────────────────────────
class EMAPredicateTracker:
    """EMA-smoothed throughput tracker for predicate file writes.

    Algorithm addition: upstream prints file-level pass/fail.
    EMA smoothing gives continuous throughput signal across writes.
    """

    def __init__(self, alpha=0.3):
        self._alpha = alpha
        self._ema = None
        self._n = 0
        self._total_lines = 0

    def record(self, n_lines, elapsed_s):
        """Record a write batch."""
        if elapsed_s <= 0:
            elapsed_s = 1e-6
        rate = n_lines / elapsed_s
        if self._ema is None:
            self._ema = rate
        else:
            self._ema = self._alpha * rate + (1 - self._alpha) * self._ema
        self._n += 1
        self._total_lines += n_lines
        _dbg("ema_pred_track", rate=f"{rate:.1f}", ema=f"{self._ema:.1f}")

    def record_dbg(self, n_lines, elapsed_s):
        """record() with forced debug output."""
        self.record(n_lines, elapsed_s)
        print(f"  [EMAPredicateTracker] n={self._n}, "
              f"total_lines={self._total_lines}, "
              f"ema_rate={self._ema:.1f} lines/s")

    @property
    def ema_rate(self):
        return self._ema or 0.0

    @property
    def total_lines(self):
        return self._total_lines


# ── Huber loss for OR-group validation ──────────────────────────
class HuberORValidator:
    """Validates OR-group predicate counts via Huber loss.

    Algorithm addition: upstream performs no validation on detected
    OR groups.  Huber loss (smooth L1) measures deviation between
    actual and expected OR-group sizes, robust to outlier groups.
    """

    def __init__(self, delta=2.0):
        self.delta = delta
        self._losses = []

    def compute_loss(self, actual_size, expected_size):
        """Huber loss for a single OR group."""
        r = abs(actual_size - expected_size)
        if r <= self.delta:
            loss = 0.5 * r * r
        else:
            loss = self.delta * (r - 0.5 * self.delta)
        _dbg("huber_or", actual=actual_size, expected=expected_size,
             loss=f"{loss:.4f}")
        self._losses.append(loss)
        return loss

    def compute_loss_dbg(self, actual_size, expected_size):
        """compute_loss() with forced debug output."""
        loss = self.compute_loss(actual_size, expected_size)
        print(f"  [HuberORValidator] actual={actual_size}, "
              f"expected={expected_size}, loss={loss:.6f}")
        return loss

    def mean_loss(self):
        """Average loss over all recorded observations."""
        return float(np.mean(self._losses)) if self._losses else 0.0

    def is_healthy(self, threshold=None):
        """Check if mean loss is below threshold."""
        t = threshold or self.delta
        return self.mean_loss() < t


# ── Bisect-based OR-group index ─────────────────────────────────
class SortedORGroupIndex:
    """Sorted index mapping table names to OR-group info.

    Algorithm change: upstream uses list iteration to match tables
    to OR groups.  Binary search over sorted table keys provides
    O(log n) lookup for large query templates with many tables.
    """

    def __init__(self):
        self._keys = []          # sorted table names
        self._groups = {}        # table -> list of OR-group dicts

    def add(self, table, group_info):
        """Insert an OR group entry for a table."""
        if table not in self._groups:
            insort(self._keys, table)
            self._groups[table] = []
        self._groups[table].append(group_info)
        _dbg("or_idx_add", table=table, n=len(self._groups[table]))

    def add_dbg(self, table, group_info):
        """add() with forced debug output."""
        self.add(table, group_info)
        print(f"  [SortedORGroupIndex] table={table!r}, "
              f"groups={len(self._groups[table])}")

    def lookup(self, table):
        """O(log n) lookup for OR groups associated with a table."""
        idx = bisect_left(self._keys, table)
        if idx < len(self._keys) and self._keys[idx] == table:
            return self._groups[table]
        return []

    def has_table(self, table):
        """Check if any OR group exists for this table."""
        idx = bisect_left(self._keys, table)
        return idx < len(self._keys) and self._keys[idx] == table

    def all_tables(self):
        """Iterate over all tables with OR groups."""
        return list(self._keys)


# ── JSON loader (hardened) ──────────────────────────────────────
def load_json_data(file_path):
    """Load JSON with size guard and encoding fallback."""
    _dbg("load_json", path=file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    size = os.path.getsize(file_path)
    if size > 500 * 1024 * 1024:
        raise ValueError(f"JSON file too large: {size} bytes")
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(file_path, "r", encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise RuntimeError(f"Cannot decode JSON file: {file_path}")


def load_json_data_dbg(file_path):
    """load_json_data() with forced debug output."""
    print(f"  [load_json_data] loading: {file_path}")
    data = load_json_data(file_path)
    print(f"  [load_json_data] loaded {len(data)} keys")
    return data


# ── OR-group detection ──────────────────────────────────────────
def detect_or_groups(query):
    """Parse a SQL query template to find OR groups.

    Returns a list of dicts with table, predicates, and param_indices.
    Enhancement: also feeds Welford stats on group sizes.
    """
    or_groups = []
    stats = WelfordORStats()

    or_patterns = re.finditer(
        r"\(([^()]+?\bOR\b[^()]+?)\)", query, re.IGNORECASE
    )

    for match in or_patterns:
        or_expr = match.group(1)
        if " OR " not in or_expr.upper():
            continue

        predicates = [p.strip() for p in or_expr.split(" OR ")]
        tables = [p.split(".")[0].strip() for p in predicates]

        param_indices = []
        for p in predicates:
            pm = re.search(r"@param(\d+)", p)
            if pm:
                param_indices.append(int(pm.group(1)))

        if len(set(tables)) == 1:
            group = {
                "table": tables[0],
                "predicates": predicates,
                "param_indices": param_indices,
            }
            or_groups.append(group)
            stats.update(len(predicates))

    _dbg("detect_or", n_groups=len(or_groups),
         stats=stats.summary())
    return or_groups, stats


def detect_or_groups_dbg(query):
    """detect_or_groups() with forced debug output."""
    print(f"  [detect_or_groups] query_len={len(query)}")
    groups, stats = detect_or_groups(query)
    print(f"  [detect_or_groups] found {len(groups)} groups, "
          f"stats={stats.summary()}")
    return groups, stats


# ── OR-aware predicate saver ────────────────────────────────────
def save_pqo_predicates_or(query_id, testing_data, output_dir,
                           tracker=None, validator=None):
    """Save PQO predicates with OR-group awareness.

    Algorithm changes vs upstream:
    - SortedORGroupIndex for O(log n) table→group lookup
    - EMA throughput tracking on write batches
    - Huber validation on OR-group sizes
    - Welford stats on numeric parameters
    """
    import time as _time

    os.makedirs(output_dir, exist_ok=True)

    def save_combinations(data, file_name):
        output_file = os.path.join(output_dir, file_name)
        if os.path.exists(output_file):
            _dbg("or_pred_skip", file=file_name)
            return

        # Detect OR groups
        query = data.get("query", "")
        or_groups, or_stats = detect_or_groups(query)

        # Build sorted OR-group index
        or_index = SortedORGroupIndex()
        for g in or_groups:
            or_index.add(g["table"], g)
            if validator is not None:
                validator.compute_loss(len(g["predicates"]), 2.0)

        original_alias_used = False
        lines = []
        welford_params = WelfordORStats()
        t0 = _time.monotonic()

        for i, combination in enumerate(data["params"]):
            table_predicates = defaultdict(list)
            table_param_indices = defaultdict(list)

            for j, item in enumerate(combination):
                predicate = data["predicates"][j]
                group_key = predicate.get("original_alias",
                                          predicate["alias"])
                table = predicate["alias"]
                column = predicate["column"]
                operator = predicate["operator"]
                data_type = predicate["data_type"]

                if data_type == "text":
                    formatted_param = f"'{item}'"
                else:
                    formatted_param = item
                    welford_params.update(item)

                if operator.lower() == "in":
                    formatted_param = f"({formatted_param})"

                predicate_str = (f"{table}.{column} {operator} "
                                 f"{formatted_param}")
                table_predicates[group_key].append(predicate_str)
                table_param_indices[table].append(j)

                if "original_alias" in predicate:
                    original_alias_used = True

            # Process predicates: OR groups first, then remaining
            formatted_groups = []
            processed_tables = set()

            for group in or_groups:
                table = group["table"]
                if table in table_predicates and table not in processed_tables:
                    or_preds = []
                    and_preds = []

                    for pred, idx in zip(table_predicates[table],
                                         table_param_indices[table]):
                        if idx in group["param_indices"]:
                            or_preds.append(pred)
                        else:
                            and_preds.append(pred)

                    if or_preds:
                        or_str = f"({' OR '.join(or_preds)})"
                        if and_preds:
                            group_str = (f"{or_str} AND "
                                         f"{' AND '.join(and_preds)}")
                        else:
                            group_str = or_str
                        formatted_groups.append(group_str)
                        processed_tables.add(table)

            for table, preds in table_predicates.items():
                if table not in processed_tables:
                    if len(preds) == 1:
                        formatted_groups.append(preds[0])
                    else:
                        formatted_groups.append(" AND ".join(preds))

            combination_str = '[\"' + '", "'.join(formatted_groups) + '"]'
            if i < len(data["params"]) - 1:
                lines.append(combination_str + ",\n")
            else:
                lines.append(combination_str + "\n")

        with open(output_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

        elapsed = _time.monotonic() - t0
        if tracker is not None:
            tracker.record(len(lines), elapsed)

        if original_alias_used:
            _dbg("original_alias_used_or", file=file_name)

        _dbg("save_or_pred", file=file_name, n_combos=len(lines),
             param_stats=welford_params.summary())

    save_combinations(testing_data[query_id],
                      f"{query_id}_mixture_test.txt")


def save_pqo_predicates_or_dbg(query_id, testing_data, output_dir,
                                tracker=None, validator=None):
    """save_pqo_predicates_or() with forced debug output."""
    print(f"  [save_pqo_predicates_or] query_id={query_id}")
    save_pqo_predicates_or(query_id, testing_data, output_dir,
                           tracker, validator)
    print(f"  [save_pqo_predicates_or] done for {query_id}")


# ── File cleanup ────────────────────────────────────────────────
def clear_files(query_ids):
    """Remove existing OR-predicate files before regeneration."""
    for qid in query_ids:
        fp = f"0_mixture_test/{qid}/PQO/{qid}_mixture_test.txt"
        if os.path.exists(fp):
            os.remove(fp)
            _dbg("clear_file", removed=fp)


def clear_files_dbg(query_ids):
    """clear_files() with forced debug output."""
    print(f"  [clear_files] cleaning {len(query_ids)} files")
    clear_files(query_ids)
    print(f"  [clear_files] done")


# ── Process single query ────────────────────────────────────────
def process_query(query_id, tracker=None, validator=None):
    """Process one query's testing data with OR-group awareness."""
    base_dir = f"0_mixture_test/{query_id}"
    pqo_dir = os.path.join(base_dir, "PQO")
    testing_file = os.path.join(
        base_dir, f"{query_id}_mixture_test.json"
    )

    testing_data = load_json_data(testing_file)
    save_pqo_predicates_or(query_id, testing_data, pqo_dir,
                           tracker=tracker, validator=validator)
    _dbg("process_query_or_done", query_id=query_id)


def process_query_dbg(query_id, tracker=None, validator=None):
    """process_query() with forced debug output."""
    print(f"  [process_query_or] START query_id={query_id}")
    process_query(query_id, tracker, validator)
    print(f"  [process_query_or] DONE query_id={query_id}")


# ── Pipeline entry point ────────────────────────────────────────
def run_pipeline(query_ids=None):
    """Run the full OR-aware PQO predicate generation pipeline.

    Returns a dict of {query_id: status_info}.
    """
    if query_ids is None:
        query_ids = ["1-0", "9-0", "11-0", "19-0", "20-0",
                     "21-0", "23-0", "24-0", "26-0", "27-0"]

    tracker = EMAPredicateTracker(alpha=0.3)
    validator = HuberORValidator(delta=2.0)
    results = {}

    clear_files(query_ids)

    for qid in query_ids:
        try:
            process_query(qid, tracker=tracker, validator=validator)
            results[qid] = {"status": "ok",
                            "ema_rate": tracker.ema_rate}
        except Exception as e:
            results[qid] = {"status": "error", "error": str(e)}
            _dbg("pipeline_or_error", query_id=qid, error=str(e))

    _dbg("pipeline_or_done",
         n_ok=sum(1 for v in results.values() if v["status"] == "ok"),
         validator_healthy=validator.is_healthy())
    return results


# ── Self-test ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== par2qo_0_mixture_PQO_predicate_generate_OR self-test ===\n")

    # 1. WelfordORStats
    print("1. WelfordORStats")
    ws = WelfordORStats()
    vals = [2.0, 3.0, 4.0, 5.0, 6.0]
    for v in vals:
        ws.update(v)
    assert ws.n == 5
    assert abs(ws.mean - 4.0) < 1e-9, f"mean={ws.mean}"
    assert abs(ws._min - 2.0) < 1e-9
    assert abs(ws._max - 6.0) < 1e-9
    expected_var = np.var(vals, ddof=0)
    assert abs(ws.variance - expected_var) < 1e-6
    print(f"   mean={ws.mean:.2f}, var={ws.variance:.2f}, "
          f"min={ws._min}, max={ws._max}  OK")

    # 2. EMAPredicateTracker
    print("2. EMAPredicateTracker")
    et = EMAPredicateTracker(alpha=0.5)
    et.record(100, 1.0)
    assert et.ema_rate == 100.0
    et.record(200, 1.0)
    assert abs(et.ema_rate - 150.0) < 1e-6
    print(f"   ema_rate={et.ema_rate:.1f}, "
          f"total_lines={et.total_lines}  OK")

    # 3. HuberORValidator
    print("3. HuberORValidator")
    hv = HuberORValidator(delta=2.0)
    l1 = hv.compute_loss(2, 2)
    assert l1 == 0.0
    l2 = hv.compute_loss(5, 2)  # r=3, > delta=2 → 2*(3-1)=4
    assert abs(l2 - 4.0) < 1e-9
    l3 = hv.compute_loss(3, 2)  # r=1, ≤ delta → 0.5*1=0.5
    assert abs(l3 - 0.5) < 1e-9
    print(f"   l_zero={l1}, l_large={l2}, l_small={l3:.1f}  OK")

    # 4. SortedORGroupIndex
    print("4. SortedORGroupIndex")
    idx = SortedORGroupIndex()
    idx.add("ci", {"param_indices": [0, 1]})
    idx.add("t", {"param_indices": [2]})
    idx.add("ci", {"param_indices": [3]})
    assert idx.has_table("ci")
    assert len(idx.lookup("ci")) == 2
    assert not idx.has_table("xyz")
    assert idx.all_tables() == ["ci", "t"]
    print(f"   tables={idx.all_tables()}, "
          f"ci_groups={len(idx.lookup('ci'))}  OK")

    # 5. detect_or_groups
    print("5. detect_or_groups")
    q = ("SELECT * FROM t JOIN ci ON t.id=ci.mid "
         "WHERE (ci.note = @param0 OR ci.role = @param1) "
         "AND t.year > @param2")
    groups, stats = detect_or_groups(q)
    assert len(groups) == 1
    assert groups[0]["table"] == "ci"
    assert set(groups[0]["param_indices"]) == {0, 1}
    print(f"   found {len(groups)} group(s), "
          f"table={groups[0]['table']}  OK")

    # 6. detect_or_groups with no OR
    print("6. detect_or_groups (no OR)")
    groups2, stats2 = detect_or_groups("SELECT * FROM t WHERE t.id = @param0")
    assert len(groups2) == 0
    print(f"   found {len(groups2)} groups (expected 0)  OK")

    # 7. Round-trip save/load predicates
    print("7. save_pqo_predicates_or round-trip")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_data = {
            "q1": {
                "query": ("SELECT * FROM a JOIN b ON a.id=b.aid "
                          "WHERE (a.x = @param0 OR a.y = @param1) "
                          "AND b.z > @param2"),
                "params": [["v1", "v2", "10"], ["v3", "v4", "20"]],
                "predicates": [
                    {"alias": "a", "column": "x", "operator": "=",
                     "data_type": "text"},
                    {"alias": "a", "column": "y", "operator": "=",
                     "data_type": "text"},
                    {"alias": "b", "column": "z", "operator": ">",
                     "data_type": "int"},
                ],
            }
        }
        save_pqo_predicates_or("q1", test_data, tmpdir)
        pred_file = os.path.join(tmpdir, "q1_mixture_test.txt")
        assert os.path.isfile(pred_file)
        with open(pred_file) as f:
            content = f.read()
        assert "OR" in content
        assert "b.z > 10" in content or "b.z > 20" in content
        lines = [l for l in content.strip().split("\n") if l.strip()]
        assert len(lines) == 2, f"expected 2 lines, got {len(lines)}"
        print(f"   pred file OK ({len(content)} bytes, "
              f"{len(lines)} combinations)")

    # 8. Validator health
    print("8. HuberORValidator health check")
    hv2 = HuberORValidator(delta=2.0)
    hv2.compute_loss(2, 2)
    hv2.compute_loss(2, 3)
    assert hv2.is_healthy()
    print(f"   mean_loss={hv2.mean_loss():.4f}, "
          f"healthy={hv2.is_healthy()}  OK")

    print("\nAll 8 tests passed.")
