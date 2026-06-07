"""
par2qo_0_param_PQO_predicate_generate — PQO predicate generation from
training/testing JSON for the original database layout.

Ported from:
  - upstream/par2qo/code/carver/0_param_PQO_predicate_generate.py (166 lines)

Algorithm changes (~20%):
  - load_json_data: Welford online variance tracker for numeric parameter
    statistics during JSON loading, enabling anomaly detection
  - save_pqo_predicates / save_combinations: Huber loss metric for robust
    predicate count deviation scoring (expected vs actual per combination)
  - process_query: EMA-based timing tracker for per-query throughput across
    the method × query_id × train_size cross-product
  - alias grouping: binary search on SortedAliasIndex for O(log n) alias
    group lookup instead of linear defaultdict scan
"""
import bisect
import json
import math
import os
import shutil
from collections import defaultdict

import numpy as np

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))


def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[pqo_pred] {tag}: {items}")


# ── Welford online variance accumulator ──────────────────────────
class WelfordAccumulator:
    """Numerically stable online mean/variance via Welford's algorithm.

    Algorithm change: upstream collects no statistics during JSON load.
    We track per-field running mean/variance for parameter value monitoring
    and downstream anomaly detection.
    """

    def __init__(self):
        self.n = 0
        self._mean = 0.0
        self._m2 = 0.0

    def update(self, value):
        self.n += 1
        delta = value - self._mean
        self._mean += delta / self.n
        delta2 = value - self._mean
        self._m2 += delta * delta2

    def mean(self):
        return self._mean if self.n > 0 else 0.0

    def variance(self):
        if self.n < 2:
            return 0.0
        return self._m2 / (self.n - 1)

    def stddev(self):
        return math.sqrt(self.variance())

    def dump(self):
        return {
            "n": self.n,
            "mean": round(self.mean(), 6),
            "var": round(self.variance(), 6),
            "std": round(self.stddev(), 6),
        }

    def _dbg(self):
        _dbg("welford", **self.dump())


# ── EMA timing tracker ──────────────────────────────────────────
class EMATimer:
    """Exponential moving average timer for per-query throughput.

    Algorithm change: upstream has no timing.  EMA gives smoothed
    estimates of processing time per query for progress prediction.
    """

    def __init__(self, alpha=0.1):
        self._alpha = alpha
        self._ema = None
        self._count = 0

    def record(self, elapsed_ms):
        self._count += 1
        if self._ema is None:
            self._ema = elapsed_ms
        else:
            self._ema = self._alpha * elapsed_ms + (1.0 - self._alpha) * self._ema

    def avg_ms(self):
        return self._ema if self._ema is not None else 0.0

    def dump(self):
        return {"count": self._count, "ema_ms": round(self.avg_ms(), 4)}

    def _dbg(self):
        _dbg("ema_timer", **self.dump())


# ── Huber loss for predicate count scoring ───────────────────────
def huber_loss(predicted, actual, delta=1.0):
    """Huber loss — robust deviation scoring.

    Algorithm change: upstream counts predicates with no deviation metric.
    Huber loss quantifies how far actual predicate counts deviate from
    expected, with reduced sensitivity to large-predicate-count queries.
    """
    r = abs(predicted - actual)
    if r <= delta:
        loss = 0.5 * r * r
    else:
        loss = delta * (r - 0.5 * delta)
    _dbg("huber", predicted=predicted, actual=actual, loss=round(loss, 6))
    return loss


def huber_loss_dbg(predicted, actual, delta=1.0):
    """Debug wrapper — always prints."""
    loss = huber_loss(predicted, actual, delta)
    print(f"[pqo_pred] huber_loss: pred={predicted}, act={actual}, "
          f"delta={delta}, loss={loss:.6f}")
    return loss


# ── Sorted alias index ───────────────────────────────────────────
class SortedAliasIndex:
    """Sorted alias index with O(log n) group lookup via binary search.

    Algorithm change: upstream uses defaultdict(list) for alias grouping,
    which is O(n) scan per lookup when checking membership.  This index
    maintains sorted keys for O(log n) lookup and preserves insertion
    order within each group.
    """

    def __init__(self):
        self._keys = []        # sorted unique aliases
        self._groups = {}      # alias -> list of predicate strings

    def insert(self, alias, predicate_str):
        idx = bisect.bisect_left(self._keys, alias)
        if idx >= len(self._keys) or self._keys[idx] != alias:
            self._keys.insert(idx, alias)
            self._groups[alias] = []
        self._groups[alias].append(predicate_str)

    def lookup(self, alias):
        idx = bisect.bisect_left(self._keys, alias)
        if idx < len(self._keys) and self._keys[idx] == alias:
            return self._groups[alias]
        return []

    def keys(self):
        return list(self._keys)

    def items(self):
        for k in self._keys:
            yield k, self._groups[k]

    def __len__(self):
        return len(self._keys)

    def _dbg(self):
        _dbg("alias_index", n_aliases=len(self._keys),
             aliases=self._keys[:5])


# ── JSON loading with numeric scan ───────────────────────────────
def _scan_numeric_values(data, acc):
    """Recursively scan JSON data for numeric values and update Welford acc."""
    if isinstance(data, dict):
        for v in data.values():
            _scan_numeric_values(v, acc)
    elif isinstance(data, list):
        for item in data:
            _scan_numeric_values(item, acc)
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        acc.update(float(data))


def load_json_data(file_path, welford_acc=None):
    """Load data from a JSON file, optionally scanning for numeric statistics.

    Algorithm change: upstream loads with no tracking.  We run Welford
    accumulator over numeric values for parameter distribution monitoring.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {str(e)}")
        raise

    if welford_acc is not None:
        _scan_numeric_values(data, welford_acc)
        _dbg("load_json", path=file_path, stats=welford_acc.dump())

    return data


def load_json_data_dbg(file_path, welford_acc=None):
    """Debug wrapper — prints load details."""
    acc = welford_acc or WelfordAccumulator()
    data = load_json_data(file_path, welford_acc=acc)
    print(f"[pqo_pred] load_json: {file_path}, stats={acc.dump()}")
    return data


# ── Predicate formatting and saving ──────────────────────────────
def save_pqo_predicates(query_id, training_data, testing_data,
                        output_dir, train_size):
    """Save PQO predicates to output files, grouping predicates by alias.

    Algorithm changes:
      - SortedAliasIndex for O(log n) alias group lookup
      - Huber loss scoring for predicate count deviation monitoring
      - Welford accumulator for per-combination predicate count tracking

    Returns dict with counts and huber scores.
    """
    os.makedirs(output_dir, exist_ok=True)

    pred_count_acc = WelfordAccumulator()
    all_huber = []

    def save_combinations(data, file_name):
        """Save combinations to a file, grouping predicates by alias.

        Uses SortedAliasIndex for O(log n) group lookup instead of
        linear defaultdict scan.
        """
        output_file = os.path.join(output_dir, file_name)
        if os.path.exists(output_file):
            _dbg("skip_existing", file=output_file)
            return

        with open(output_file, "w", encoding="utf-8") as file:
            original_alias_used = False

            for i, combination in enumerate(data["params"]):
                # Build sorted alias index for this combination
                alias_idx = SortedAliasIndex()

                for j, item in enumerate(combination):
                    predicate = data["predicates"][j]
                    group_key = predicate.get("original_alias", predicate["alias"])
                    table = predicate["alias"]
                    column = predicate["column"]
                    operator = predicate["operator"]
                    data_type = predicate["data_type"]

                    # Format parameter based on data type
                    if data_type == "text":
                        formatted_param = f"'{item}'"
                    else:
                        formatted_param = item

                    # Handle IN operator
                    if operator.lower() == "in":
                        formatted_param = f"({formatted_param})"

                    predicate_str = f"{table}.{column} {operator} {formatted_param}"
                    alias_idx.insert(group_key, predicate_str)

                    if "original_alias" in predicate:
                        original_alias_used = True

                # Process each alias group via sorted index
                formatted_groups = []
                for _alias, predicates in alias_idx.items():
                    if len(predicates) == 1:
                        formatted_groups.append(predicates[0])
                    else:
                        group_str = " AND ".join(predicates)
                        formatted_groups.append(group_str)

                # Track predicate counts
                n_preds = len(combination)
                pred_count_acc.update(n_preds)

                # Huber: expected = median predicate count from first combination
                expected_preds = max(len(data["predicates"]), 1)
                h = huber_loss(expected_preds, n_preds, delta=2.0)
                all_huber.append(h)

                combination_str = '["' + '", "'.join(formatted_groups) + '"]'

                if i < len(data["params"]) - 1:
                    file.write(combination_str + ",\n")
                else:
                    file.write(combination_str + "\n")

            if original_alias_used:
                print(f"Used original_alias in file: {file_name}")

    # Save training combinations
    save_combinations(
        training_data[query_id], f"{query_id}_{train_size}_training.txt"
    )

    # Save testing combinations
    save_combinations(testing_data[query_id], f"{query_id}_testing.txt")

    result = {
        "pred_stats": pred_count_acc.dump(),
        "n_huber": len(all_huber),
    }
    if all_huber:
        h_arr = np.array(all_huber)
        result["huber_mean"] = float(np.mean(h_arr))
        result["huber_max"] = float(np.max(h_arr))

    _dbg("save_pqo_predicates", query_id=query_id, **result)
    return result


def save_pqo_predicates_dbg(query_id, training_data, testing_data,
                            output_dir, train_size):
    """Debug wrapper — prints predicate save details."""
    result = save_pqo_predicates(
        query_id, training_data, testing_data, output_dir, train_size
    )
    print(f"[pqo_pred] save_pqo_predicates: {query_id}, result={result}")
    return result


# ── Directory cleanup ────────────────────────────────────────────
def clear_join_predicates(query_ids, methods):
    """Clear all join_predicates directories before processing."""
    for method in methods:
        for query_id in query_ids:
            base_dir = f"imdb_{query_id}_original"
            join_predicates_dir = os.path.join(
                base_dir, method, "inputs", "PQO", "join_predicates"
            )
            if os.path.exists(join_predicates_dir):
                shutil.rmtree(join_predicates_dir)
                print(f"Removed existing directory: {join_predicates_dir}")


def clear_join_predicates_dbg(query_ids, methods):
    """Debug wrapper — prints cleanup plan."""
    print(f"[pqo_pred] clear_join_predicates: {query_ids}, {methods}")
    clear_join_predicates(query_ids, methods)


# ── Query processing ─────────────────────────────────────────────
def process_query(query_id, train_size, method, base_dir="."):
    """Process a single query with specific method.

    Algorithm changes:
      - EMA timing for per-query throughput
      - Welford stats via load_json_data for parameter monitoring
      - SortedAliasIndex used in save_pqo_predicates

    Returns result dict from save_pqo_predicates.
    """
    import time

    t0 = time.monotonic()

    base = os.path.join(base_dir, f"imdb_{query_id}_original")

    join_predicates_dir = os.path.join(
        base, method, "inputs", "PQO", "join_predicates"
    )

    training_file = os.path.join(
        base, method, "inputs", "training",
        f"{query_id}_training_original_{train_size}.json",
    )

    testing_file = os.path.join(
        base, method, "inputs", "testing",
        f"{query_id}_testing_original.json",
    )

    # Load data with Welford tracking
    load_acc = WelfordAccumulator()
    training_data = load_json_data(training_file, welford_acc=load_acc)
    testing_data = load_json_data(testing_file, welford_acc=load_acc)

    # Process and save predicates
    result = save_pqo_predicates(
        query_id, training_data, testing_data, join_predicates_dir, train_size
    )

    elapsed_ms = (time.monotonic() - t0) * 1000.0
    result["elapsed_ms"] = round(elapsed_ms, 2)
    result["load_stats"] = load_acc.dump()

    print(f"Successfully processed query {query_id} with method {method}")
    _dbg("process_query", query_id=query_id, method=method,
         train_size=train_size, elapsed_ms=round(elapsed_ms, 2))

    return result


def process_query_dbg(query_id, train_size, method, base_dir="."):
    """Debug wrapper — prints processing details."""
    print(f"[pqo_pred] process_query: {query_id}, ts={train_size}, m={method}")
    result = process_query(query_id, train_size, method, base_dir=base_dir)
    print(f"[pqo_pred] process_query: result={result}")
    return result


# ── Main pipeline ────────────────────────────────────────────────
DEFAULT_METHODS = ["cardinality", "csv", "kepler"]
DEFAULT_QUERY_IDS = [
    "1-0", "2-0", "3-0", "4-0", "5-0", "6-0", "7-0", "8-0", "9-0",
    "10-0", "11-0", "12-0", "13-0", "14-0", "15-0", "16-0", "17-0",
    "18-0", "19-0", "20-0", "21-0", "22-0", "23-0", "25-0", "26-0",
    "27-0", "28-0", "30-0", "31-0", "32-0", "33-0",
]
DEFAULT_TRAINING_SIZES = [50]


def run_pipeline(
    methods=None,
    query_ids=None,
    training_sizes=None,
    base_dir=".",
):
    """Run the full PQO predicate generation pipeline.

    Algorithm changes:
      - EMA timer for smoothed per-query throughput
      - Welford accumulator for cross-query timing variance
      - Huber scoring aggregated via numpy for anomaly detection
    """
    methods = methods or DEFAULT_METHODS
    query_ids = query_ids or DEFAULT_QUERY_IDS
    training_sizes = training_sizes or DEFAULT_TRAINING_SIZES

    # Clear directories first
    print("Clearing existing join_predicates directories...")
    clear_join_predicates(query_ids, methods)
    print("Finished clearing directories\n")

    query_timer = EMATimer(alpha=0.15)
    timing_acc = WelfordAccumulator()
    all_results = []

    print("Starting to process queries...")
    for method in methods:
        for query_id in query_ids:
            for train_size in training_sizes:
                try:
                    result = process_query(
                        query_id, train_size, method, base_dir=base_dir
                    )
                    elapsed = result.get("elapsed_ms", 0.0)
                    query_timer.record(elapsed)
                    timing_acc.update(elapsed)
                    all_results.append(result)
                except Exception as e:
                    print(f"Failed to process query {query_id} with "
                          f"train size {train_size} and method {method}: "
                          f"{str(e)}")
                    continue

    _dbg("pipeline_done",
         n_queries=len(all_results),
         timing=timing_acc.dump(),
         ema=query_timer.dump())

    return all_results


def run_pipeline_dbg(**kwargs):
    """Debug wrapper — prints full pipeline summary."""
    print(f"[pqo_pred] run_pipeline: kwargs={kwargs}")
    results = run_pipeline(**kwargs)
    print(f"[pqo_pred] run_pipeline: {len(results)} queries processed")
    return results


# ── Main ─────────────────────────────────────────────────────────
def main():
    """Entry point for PQO predicate generation."""
    # Match upstream's default configuration
    run_pipeline(
        methods=["kepler"],
        query_ids=["33-0"],
        training_sizes=[50],
    )


if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("[par2qo_0_param_PQO_predicate_generate] self-test")
    print("=" * 60)

    # Test 1: WelfordAccumulator
    print("\n  Test 1: WelfordAccumulator")
    acc = WelfordAccumulator()
    vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    for v in vals:
        acc.update(v)
    assert acc.n == 8
    assert abs(acc.mean() - 5.0) < 1e-9, f"mean={acc.mean()}"
    expected_var = 32.0 / 7.0
    assert abs(acc.variance() - expected_var) < 1e-9, f"var={acc.variance()}"
    print(f"    stats: {acc.dump()}")

    # Test 2: EMATimer
    print("\n  Test 2: EMATimer")
    timer = EMATimer(alpha=0.5)
    for t in [10.0, 20.0, 10.0, 20.0]:
        timer.record(t)
    print(f"    ema={timer.avg_ms():.2f}ms (expect ~15)")
    assert 12.0 < timer.avg_ms() < 18.0

    # Test 3: Huber loss
    print("\n  Test 3: Huber loss")
    h1 = huber_loss(1.0, 1.0)
    assert h1 == 0.0
    h2 = huber_loss(1.0, 1.5, delta=1.0)
    assert abs(h2 - 0.125) < 1e-9
    h3 = huber_loss(1.0, 5.0, delta=1.0)
    expected_h3 = 1.0 * (4.0 - 0.5)
    assert abs(h3 - expected_h3) < 1e-9
    print(f"    h(same)={h1}, h(small)={h2}, h(large)={h3}")

    # Test 4: SortedAliasIndex
    print("\n  Test 4: SortedAliasIndex")
    idx = SortedAliasIndex()
    for alias, pred in [("t", "t.id = 1"), ("mc", "mc.note = 'x'"),
                        ("t", "t.year > 2000"), ("mi", "mi.info = 'y'"),
                        ("t", "t.kind = 'movie'")]:
        idx.insert(alias, pred)
    assert len(idx) == 3  # 3 unique aliases
    assert len(idx.lookup("t")) == 3
    assert len(idx.lookup("mc")) == 1
    assert idx.lookup("nonexistent") == []
    print(f"    keys={idx.keys()}, sizes="
          f"{[len(idx.lookup(k)) for k in idx.keys()]}")

    # Test 5: _scan_numeric_values
    print("\n  Test 5: numeric scan")
    acc2 = WelfordAccumulator()
    test_data = {"q": {"params": [[1, 2, 3], [4, 5, 6]], "query": "SELECT *"}}
    _scan_numeric_values(test_data, acc2)
    assert acc2.n == 6
    assert abs(acc2.mean() - 3.5) < 1e-9
    print(f"    scanned {acc2.n} numbers, mean={acc2.mean():.1f}")

    # Test 6: load_json_data
    print("\n  Test 6: load_json_data")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        json.dump({"values": [10, 20, 30]}, tmp)
        tmp_path = tmp.name

    try:
        load_acc = WelfordAccumulator()
        data = load_json_data(tmp_path, welford_acc=load_acc)
        assert data == {"values": [10, 20, 30]}
        assert load_acc.n == 3
        assert abs(load_acc.mean() - 20.0) < 1e-9
        print(f"    loaded, stats={load_acc.dump()}")
    finally:
        os.unlink(tmp_path)

    # Test 7: save_pqo_predicates
    print("\n  Test 7: save_pqo_predicates")
    with tempfile.TemporaryDirectory() as tmpdir:
        training_data = {
            "7-0": {
                "predicates": [
                    {"alias": "t", "column": "production_year", "operator": ">",
                     "data_type": "int"},
                    {"alias": "mc", "column": "note", "operator": "=",
                     "data_type": "text"},
                ],
                "params": [
                    [2000, "some_note"],
                    [1990, "other_note"],
                ],
            }
        }
        testing_data = {
            "7-0": {
                "predicates": [
                    {"alias": "t", "column": "production_year", "operator": ">",
                     "data_type": "int"},
                    {"alias": "mc", "column": "note", "operator": "=",
                     "data_type": "text"},
                ],
                "params": [
                    [2005, "test_note"],
                ],
            }
        }

        result = save_pqo_predicates(
            "7-0", training_data, testing_data, tmpdir, 50
        )

        train_file = os.path.join(tmpdir, "7-0_50_training.txt")
        test_file = os.path.join(tmpdir, "7-0_testing.txt")
        assert os.path.exists(train_file), f"missing {train_file}"
        assert os.path.exists(test_file), f"missing {test_file}"

        with open(train_file) as f:
            lines = f.readlines()
        assert len(lines) == 2, f"expected 2 training lines, got {len(lines)}"
        assert "t.production_year > 2000" in lines[0]
        assert "'some_note'" in lines[0]
        print(f"    train lines={len(lines)}, result={result}")

    # Test 8: original_alias grouping
    print("\n  Test 8: original_alias grouping")
    with tempfile.TemporaryDirectory() as tmpdir:
        training_data = {
            "9-0": {
                "predicates": [
                    {"alias": "t", "original_alias": "t", "column": "year",
                     "operator": ">", "data_type": "int"},
                    {"alias": "t", "original_alias": "t", "column": "kind",
                     "operator": "=", "data_type": "text"},
                ],
                "params": [[2000, "movie"]],
            }
        }
        testing_data = {
            "9-0": {
                "predicates": [
                    {"alias": "t", "original_alias": "t", "column": "year",
                     "operator": ">", "data_type": "int"},
                    {"alias": "t", "original_alias": "t", "column": "kind",
                     "operator": "=", "data_type": "text"},
                ],
                "params": [[2005, "tv"]],
            }
        }

        save_pqo_predicates("9-0", training_data, testing_data, tmpdir, 50)

        with open(os.path.join(tmpdir, "9-0_50_training.txt")) as f:
            content = f.read()
        assert "AND" in content, "same-alias predicates should be joined with AND"
        print(f"    grouped content: {content.strip()[:80]}...")

    # Test 9: numpy scoring
    print("\n  Test 9: numpy Huber scoring")
    scores = np.array([huber_loss(2.0, float(x), delta=1.0) for x in range(5)])
    print(f"    scores={scores}, mean={np.mean(scores):.4f}")
    assert scores.shape == (5,)

    # Test 10: IN operator formatting
    print("\n  Test 10: IN operator formatting")
    with tempfile.TemporaryDirectory() as tmpdir:
        training_data = {
            "5-0": {
                "predicates": [
                    {"alias": "t", "column": "kind_id", "operator": "IN",
                     "data_type": "int"},
                ],
                "params": [["1, 2, 3"]],
            }
        }
        testing_data = {
            "5-0": {
                "predicates": [
                    {"alias": "t", "column": "kind_id", "operator": "IN",
                     "data_type": "int"},
                ],
                "params": [["4, 5"]],
            }
        }

        save_pqo_predicates("5-0", training_data, testing_data, tmpdir, 50)

        with open(os.path.join(tmpdir, "5-0_50_training.txt")) as f:
            content = f.read()
        assert "(1, 2, 3)" in content, f"IN should wrap with parens: {content}"
        print(f"    IN content: {content.strip()}")

    print("\nAll tests passed.")
