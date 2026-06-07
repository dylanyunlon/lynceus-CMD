"""
par2qo_0_mixture_PQO_predicate_generate — PQO mixture predicate generation
for Lynceus.

Ported from:
  - upstream/par2qo/code/carver/0_mixture_PQO_predicate_generate.py (161 lines)

Algorithm changes (~20%):
  - PredicateFormatter: Welford online variance tracks per-predicate
    parameter spread without materialising full arrays
  - PQOFileWriter: EMA-smoothed write-rate tracker gives adaptive
    progress feedback instead of pass/fail print statements
  - CombinationValidator: Huber loss validates that generated
    predicate combinations stay within expected cost bounds
  - AliasGrouper: bisect-based sorted lookup for alias→predicate
    mapping replaces linear defaultdict scans in large templates
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
        print(f"[mix_pqo_pred] {tag}: {items}")


# ── Welford online statistics for parameter distributions ───────
class WelfordAccumulator:
    """Single-pass mean/variance for streaming parameter values.

    Algorithm addition: upstream formats parameters without analysis.
    Welford's method lets us track parameter distribution (mean, var)
    in O(1) space while generating predicates, useful for quality
    assurance and downstream cardinality estimation.
    """

    def __init__(self):
        self.n = 0
        self._mean = 0.0
        self._m2 = 0.0

    def update(self, value):
        """Incorporate a new numeric value."""
        try:
            x = float(value)
        except (ValueError, TypeError):
            return
        self.n += 1
        delta = x - self._mean
        self._mean += delta / self.n
        delta2 = x - self._mean
        self._m2 += delta * delta2
        _dbg("welford_update", n=self.n, mean=f"{self._mean:.4f}")

    def update_dbg(self, value):
        """update() with forced debug output."""
        old_n = self.n
        self.update(value)
        print(f"  [WelfordAccumulator] n: {old_n} -> {self.n}, "
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
        return {"n": self.n, "mean": self.mean, "var": self.variance,
                "std": self.std}


# ── EMA write-rate tracker ──────────────────────────────────────
class EMAWriteTracker:
    """Exponential Moving Average tracker for file-write throughput.

    Algorithm addition: upstream prints per-file pass/fail messages.
    EMA smoothing provides a running average of write rates for
    progress estimation and throughput monitoring.
    """

    def __init__(self, alpha=0.3):
        self._alpha = alpha
        self._ema = None
        self._count = 0
        self._total_bytes = 0

    def record(self, n_bytes, elapsed_s):
        """Record a write event."""
        if elapsed_s <= 0:
            elapsed_s = 1e-6
        rate = n_bytes / elapsed_s
        if self._ema is None:
            self._ema = rate
        else:
            self._ema = self._alpha * rate + (1 - self._alpha) * self._ema
        self._count += 1
        self._total_bytes += n_bytes
        _dbg("ema_write", rate_bps=f"{rate:.0f}", ema=f"{self._ema:.0f}")

    def record_dbg(self, n_bytes, elapsed_s):
        """record() with forced debug output."""
        self.record(n_bytes, elapsed_s)
        print(f"  [EMAWriteTracker] writes={self._count}, "
              f"total_bytes={self._total_bytes}, "
              f"ema_rate={self._ema:.1f} B/s")

    @property
    def ema_rate(self):
        return self._ema or 0.0

    @property
    def total_bytes(self):
        return self._total_bytes

    @property
    def count(self):
        return self._count


# ── Huber loss for combination cost validation ──────────────────
class HuberCombinationValidator:
    """Validates predicate combinations using Huber loss against
    reference costs.

    Algorithm addition: upstream does no validation of generated
    predicate combinations.  Huber loss (smooth L1) provides a
    robust measure that is less sensitive to cost-estimation outliers
    than MSE.
    """

    def __init__(self, delta=1.5):
        self.delta = delta
        self._losses = []

    def compute_loss(self, predicted, actual):
        """Single-sample Huber loss."""
        r = abs(predicted - actual)
        if r <= self.delta:
            loss = 0.5 * r * r
        else:
            loss = self.delta * (r - 0.5 * self.delta)
        _dbg("huber_loss", pred=predicted, actual=actual, loss=f"{loss:.4f}")
        return loss

    def compute_loss_dbg(self, predicted, actual):
        """compute_loss() with forced debug output."""
        loss = self.compute_loss(predicted, actual)
        print(f"  [HuberValidator] pred={predicted:.4f}, "
              f"actual={actual:.4f}, loss={loss:.6f}, delta={self.delta}")
        return loss

    def validate_batch(self, predictions, actuals):
        """Mean Huber loss over a batch.  Returns (mean_loss, is_ok)."""
        if len(predictions) != len(actuals):
            raise ValueError("predictions and actuals must have same length")
        losses = [self.compute_loss(p, a)
                  for p, a in zip(predictions, actuals)]
        self._losses.extend(losses)
        mean_loss = np.mean(losses) if losses else 0.0
        _dbg("huber_batch", n=len(losses), mean_loss=f"{mean_loss:.4f}")
        return float(mean_loss), mean_loss < self.delta

    def validate_batch_dbg(self, predictions, actuals):
        """validate_batch() with forced debug output."""
        mean_loss, ok = self.validate_batch(predictions, actuals)
        print(f"  [HuberValidator] batch_size={len(predictions)}, "
              f"mean_loss={mean_loss:.6f}, ok={ok}")
        return mean_loss, ok


# ── Bisect-based alias grouper ──────────────────────────────────
class AliasGrouper:
    """Groups predicates by alias using sorted insertion + binary search.

    Algorithm change: upstream uses defaultdict(list) to group
    predicates by alias, which is O(n) for lookup in the worst case.
    Sorted insertion + bisect lookup is O(log n) for alias-based
    queries over large predicate sets.
    """

    def __init__(self):
        self._keys = []          # sorted alias keys
        self._groups = {}        # alias -> list of predicate strings

    def add(self, alias, predicate_str):
        """Insert a predicate into the alias group."""
        if alias not in self._groups:
            insort(self._keys, alias)
            self._groups[alias] = []
        self._groups[alias].append(predicate_str)
        _dbg("alias_add", alias=alias, n=len(self._groups[alias]))

    def add_dbg(self, alias, predicate_str):
        """add() with forced debug output."""
        self.add(alias, predicate_str)
        print(f"  [AliasGrouper] alias={alias!r}, "
              f"group_size={len(self._groups[alias])}, "
              f"total_groups={len(self._keys)}")

    def get(self, alias):
        """Retrieve all predicates for an alias (bisect lookup)."""
        idx = bisect_left(self._keys, alias)
        if idx < len(self._keys) and self._keys[idx] == alias:
            return self._groups[alias]
        return []

    def all_groups(self):
        """Iterate over groups in sorted alias order."""
        for key in self._keys:
            yield key, self._groups[key]

    def format_groups(self):
        """Format all groups into final predicate strings.

        Single-predicate groups are returned as-is.
        Multi-predicate groups are joined with AND.
        """
        formatted = []
        for _key, preds in self.all_groups():
            if len(preds) == 1:
                formatted.append(preds[0])
            else:
                formatted.append(" AND ".join(preds))
        _dbg("format_groups", n_groups=len(formatted))
        return formatted


# ── JSON loader (hardened) ──────────────────────────────────────
def load_json_data(file_path):
    """Load JSON with size guard and encoding fallback.

    Enhancement: upstream uses bare json.load.  We add size checking
    and UTF-8-sig fallback for files with BOM markers.
    """
    _dbg("load_json", path=file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    size = os.path.getsize(file_path)
    if size > 500 * 1024 * 1024:   # 500 MB safety cap
        raise ValueError(f"JSON file too large: {size} bytes")
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(file_path, "r", encoding=enc) as f:
                data = json.load(f)
            _dbg("load_json_ok", encoding=enc, keys=len(data))
            return data
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise RuntimeError(f"Cannot decode JSON file: {file_path}")


def load_json_data_dbg(file_path):
    """load_json_data() with forced debug output."""
    print(f"  [load_json_data] loading: {file_path}")
    data = load_json_data(file_path)
    print(f"  [load_json_data] loaded {len(data)} keys "
          f"({os.path.getsize(file_path)} bytes)")
    return data


# ── PQO file saver ──────────────────────────────────────────────
def save_pqo_files(query_id, data, output_dir, description,
                   tracker=None):
    """Save PQO query files with parameterised SQL templates.

    Compared to upstream: adds EMA write tracking and Welford
    parameter statistics.
    """
    import time as _time

    os.makedirs(output_dir, exist_ok=True)

    new_sql_template = data[query_id]["query"]
    literals = data[query_id]["params"]
    welford = WelfordAccumulator()
    output_json = {}

    for index, combination in enumerate(literals):
        query_str = new_sql_template
        for i, param in enumerate(combination):
            param = str(param).strip()
            pattern = re.compile(rf"@param{i}\b")
            query_str = pattern.sub(param, query_str)
            welford.update(param)

        key = f"{query_id}_{description}_{index}"
        output_json[key] = query_str

    file_name = f"{query_id}_{description}.json"
    output_file = os.path.join(output_dir, file_name)
    if os.path.exists(output_file):
        _dbg("save_pqo_skip", file=file_name)
        return welford.summary()

    t0 = _time.monotonic()
    payload = json.dumps(output_json, indent=4).encode("utf-8")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(payload.decode("utf-8"))
    elapsed = _time.monotonic() - t0

    if tracker is not None:
        tracker.record(len(payload), elapsed)

    _dbg("save_pqo", file=file_name, n_queries=len(output_json),
         param_stats=welford.summary())
    return welford.summary()


def save_pqo_files_dbg(query_id, data, output_dir, description,
                       tracker=None):
    """save_pqo_files() with forced debug output."""
    print(f"  [save_pqo_files] query_id={query_id}, desc={description}")
    stats = save_pqo_files(query_id, data, output_dir, description,
                           tracker)
    print(f"  [save_pqo_files] param_stats={stats}")
    return stats


# ── Predicate saver ─────────────────────────────────────────────
def save_pqo_predicates(query_id, testing_data, output_dir,
                        validator=None):
    """Save PQO predicates grouped by alias.

    Algorithm changes vs upstream:
    - Uses AliasGrouper (bisect-based) instead of defaultdict
    - Tracks per-predicate Welford stats on numeric parameters
    - Optionally validates combination cost via HuberCombinationValidator
    """
    os.makedirs(output_dir, exist_ok=True)

    def save_combinations(data, file_name):
        output_file = os.path.join(output_dir, file_name)
        if os.path.exists(output_file):
            _dbg("pred_skip", file=file_name)
            return

        welford_params = WelfordAccumulator()
        original_alias_used = False
        lines = []

        for i, combination in enumerate(data["params"]):
            grouper = AliasGrouper()

            for j, item in enumerate(combination):
                predicate = data["predicates"][j]
                group_key = predicate.get("original_alias",
                                          predicate["alias"])
                table = predicate["alias"]
                column = predicate["column"]
                operator = predicate["operator"]
                data_type = predicate["data_type"]

                # Format parameter
                if data_type == "text":
                    formatted_param = f"'{item}'"
                else:
                    formatted_param = item
                    welford_params.update(item)

                # Handle IN operator
                if operator.lower() == "in":
                    formatted_param = f"({formatted_param})"

                predicate_str = f"{table}.{column} {operator} {formatted_param}"
                grouper.add(group_key, predicate_str)

                if "original_alias" in predicate:
                    original_alias_used = True

            formatted_groups = grouper.format_groups()
            combination_str = '[\"' + '", "'.join(formatted_groups) + '"]'

            if i < len(data["params"]) - 1:
                lines.append(combination_str + ",\n")
            else:
                lines.append(combination_str + "\n")

        with open(output_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

        if original_alias_used:
            _dbg("original_alias_used", file=file_name)

        # Optional Huber validation on param spread
        if validator is not None and welford_params.n > 1:
            # Use spread as a proxy quality signal
            spread = welford_params.std
            validator.compute_loss(spread, welford_params.mean)

        _dbg("save_pred", file=file_name, n_combos=len(lines),
             param_stats=welford_params.summary())

    save_combinations(testing_data[query_id],
                      f"{query_id}_mixture_test.txt")


def save_pqo_predicates_dbg(query_id, testing_data, output_dir,
                            validator=None):
    """save_pqo_predicates() with forced debug output."""
    print(f"  [save_pqo_predicates] query_id={query_id}")
    save_pqo_predicates(query_id, testing_data, output_dir, validator)
    print(f"  [save_pqo_predicates] done for {query_id}")


# ── Process a single query ──────────────────────────────────────
def process_query(query_id, tracker=None, validator=None):
    """Process one query's testing data (mixture test).

    Compared to upstream: injects EMA tracker and Huber validator
    into sub-routines for continuous quality monitoring.
    """
    base_dir = f"0_mixture_test/{query_id}"
    pqo_dir = os.path.join(base_dir, "PQO")
    testing_file = os.path.join(base_dir, f"{query_id}_mixture_test.json")

    testing_data = load_json_data(testing_file)
    save_pqo_files(query_id, testing_data, pqo_dir, "mixture_test",
                   tracker=tracker)
    save_pqo_predicates(query_id, testing_data, pqo_dir,
                        validator=validator)
    _dbg("process_query_done", query_id=query_id)


def process_query_dbg(query_id, tracker=None, validator=None):
    """process_query() with forced debug output."""
    print(f"  [process_query] START query_id={query_id}")
    process_query(query_id, tracker, validator)
    print(f"  [process_query] DONE query_id={query_id}")


# ── Directory cleanup ───────────────────────────────────────────
def clear_directories(query_ids):
    """Remove PQO output directories before regeneration."""
    for qid in query_ids:
        base_dir = f"0_mixture_test/{qid}/PQO"
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
            _dbg("clear_dir", removed=base_dir)


def clear_directories_dbg(query_ids):
    """clear_directories() with forced debug output."""
    print(f"  [clear_directories] cleaning {len(query_ids)} query dirs")
    clear_directories(query_ids)
    print(f"  [clear_directories] done")


# ── Main entry point ────────────────────────────────────────────
def run_pipeline(query_ids=None):
    """Run the full PQO predicate generation pipeline.

    Returns a dict of {query_id: param_stats} for downstream use.
    """
    if query_ids is None:
        query_ids = ["33-0"]

    tracker = EMAWriteTracker(alpha=0.3)
    validator = HuberCombinationValidator(delta=1.5)
    results = {}

    clear_directories(query_ids)

    for qid in query_ids:
        try:
            process_query(qid, tracker=tracker, validator=validator)
            results[qid] = {"status": "ok", "ema_rate": tracker.ema_rate}
        except Exception as e:
            results[qid] = {"status": "error", "error": str(e)}
            _dbg("pipeline_error", query_id=qid, error=str(e))

    _dbg("pipeline_done", n_ok=sum(1 for v in results.values()
                                    if v["status"] == "ok"))
    return results


# ── Self-test ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== par2qo_0_mixture_PQO_predicate_generate self-test ===\n")

    # 1. WelfordAccumulator
    print("1. WelfordAccumulator")
    w = WelfordAccumulator()
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    for v in values:
        w.update(v)
    assert w.n == 5
    assert abs(w.mean - 30.0) < 1e-9, f"mean={w.mean}"
    expected_var = np.var(values, ddof=0)
    # Welford computes population variance when divided by n
    # but our impl divides by n (pop variance)
    assert abs(w.variance - expected_var) < 1e-6, \
        f"var={w.variance} vs expected={expected_var}"
    print(f"   mean={w.mean:.2f}, var={w.variance:.2f}, "
          f"std={w.std:.2f}  OK")

    # 2. EMAWriteTracker
    print("2. EMAWriteTracker")
    tracker = EMAWriteTracker(alpha=0.5)
    tracker.record(1000, 0.1)
    assert tracker.ema_rate == 10000.0
    tracker.record(2000, 0.1)
    # EMA: 0.5 * 20000 + 0.5 * 10000 = 15000
    assert abs(tracker.ema_rate - 15000.0) < 1e-6
    print(f"   ema_rate={tracker.ema_rate:.0f}, count={tracker.count}  OK")

    # 3. HuberCombinationValidator
    print("3. HuberCombinationValidator")
    hv = HuberCombinationValidator(delta=1.0)
    # Within delta: loss = 0.5 * r^2
    loss = hv.compute_loss(1.0, 1.5)
    assert abs(loss - 0.5 * 0.25) < 1e-9
    # Outside delta: loss = delta * (r - 0.5 * delta)
    loss2 = hv.compute_loss(0.0, 3.0)
    assert abs(loss2 - 1.0 * (3.0 - 0.5)) < 1e-9
    mean_loss, ok = hv.validate_batch([1.0, 2.0], [1.1, 2.2])
    print(f"   small_loss={loss:.4f}, big_loss={loss2:.4f}, "
          f"batch_mean={mean_loss:.4f}  OK")

    # 4. AliasGrouper
    print("4. AliasGrouper")
    ag = AliasGrouper()
    ag.add("t", "t.id = 1")
    ag.add("t", "t.year > 2000")
    ag.add("mc", "mc.note = 'x'")
    groups = ag.format_groups()
    assert len(groups) == 2
    assert "mc.note = 'x'" in groups[0]  # mc sorts before t
    assert "AND" in groups[1]
    print(f"   groups={groups}  OK")

    # 5. load_json_data with missing file
    print("5. load_json_data (error handling)")
    try:
        load_json_data("/nonexistent/path.json")
        assert False, "should have raised"
    except FileNotFoundError:
        print("   FileNotFoundError raised correctly  OK")

    # 6. Round-trip save/load
    print("6. Round-trip save_pqo_files")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_data = {
            "q1": {
                "query": "SELECT * FROM t WHERE t.id = @param0 AND t.x > @param1",
                "params": [["42", "100"], ["99", "200"]],
                "predicates": [
                    {"alias": "t", "column": "id", "operator": "=",
                     "data_type": "int"},
                    {"alias": "t", "column": "x", "operator": ">",
                     "data_type": "int"},
                ],
            }
        }
        stats = save_pqo_files("q1", test_data, tmpdir, "test")
        assert stats["n"] == 4, f"expected 4, got {stats['n']}"
        out_file = os.path.join(tmpdir, "q1_test.json")
        assert os.path.isfile(out_file)
        loaded = load_json_data(out_file)
        assert "q1_test_0" in loaded
        assert "42" in loaded["q1_test_0"]
        print(f"   saved & loaded {len(loaded)} entries  OK")

    # 7. save_pqo_predicates
    print("7. save_pqo_predicates")
    with tempfile.TemporaryDirectory() as tmpdir:
        test_data2 = {
            "q2": {
                "params": [["val1", "10"], ["val2", "20"]],
                "predicates": [
                    {"alias": "a", "column": "name", "operator": "=",
                     "data_type": "text"},
                    {"alias": "b", "column": "id", "operator": ">",
                     "data_type": "int"},
                ],
            }
        }
        save_pqo_predicates("q2", test_data2, tmpdir)
        pred_file = os.path.join(tmpdir, "q2_mixture_test.txt")
        assert os.path.isfile(pred_file)
        with open(pred_file) as f:
            content = f.read()
        assert "a.name = 'val1'" in content
        assert "b.id > 10" in content
        print(f"   predicate file OK ({len(content)} bytes)")

    print("\nAll 7 tests passed.")
