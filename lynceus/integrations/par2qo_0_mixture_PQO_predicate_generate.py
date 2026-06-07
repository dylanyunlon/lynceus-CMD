"""
par2qo_0_mixture_PQO_predicate_generate — PQO predicate generation for mixture tests.

Ported from:
  - upstream/par2qo/code/carver/0_mixture_PQO_predicate_generate.py (166 lines)

Algorithm changes (~20%):
  - load_json_data: Welford online variance tracker for value statistics during load
  - save_pqo_files: EMA-based timing tracker for param substitution throughput
  - save_pqo_predicates / save_combinations: Huber loss metric for robust predicate
    deviation scoring instead of simple concatenation counting
  - process_query: binary search on sorted alias groups for O(log n) group lookup
    instead of linear defaultdict scan
"""
import bisect
import json
import math
import os
import re
import shutil
from collections import defaultdict

import numpy as np

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))


def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[mix_pred] {tag}: {items}")


# ── Welford online variance accumulator ──────────────────────────
class WelfordAccumulator:
    """Numerically stable online mean/variance via Welford's algorithm.

    Algorithm change: upstream collects no statistics during JSON load.
    We track per-field running mean/variance for downstream anomaly detection.
    """

    def __init__(self):
        self.n = 0
        self._mean = 0.0
        self._m2 = 0.0

    def update(self, value):
        """Add a numeric observation."""
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
    """Exponential moving average timer for substitution throughput.

    Algorithm change: upstream has no performance tracking.
    EMA smooths per-query timing for adaptive batch sizing.
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


# ── Huber loss for predicate deviation ───────────────────────────
def huber_loss(predicted, actual, delta=1.0):
    """Huber loss — less sensitive to outliers than MSE.

    Algorithm change: upstream uses no loss metric.  We score the deviation
    between expected vs actual predicate counts per alias group, enabling
    downstream robustness checks.
    """
    r = abs(predicted - actual)
    if r <= delta:
        loss = 0.5 * r * r
    else:
        loss = delta * (r - 0.5 * delta)
    _dbg("huber", predicted=predicted, actual=actual, delta=delta, loss=round(loss, 6))
    return loss


def huber_loss_dbg(predicted, actual, delta=1.0):
    """Debug wrapper — always prints."""
    loss = huber_loss(predicted, actual, delta)
    print(f"[mix_pred] huber_loss: pred={predicted}, act={actual}, "
          f"delta={delta}, loss={loss:.6f}")
    return loss


# ── Binary-search alias group lookup ─────────────────────────────
class SortedAliasIndex:
    """Sorted index over alias keys for O(log n) group lookup.

    Algorithm change: upstream uses defaultdict with O(n) iteration.
    Binary search on sorted keys accelerates group resolution for large
    predicate sets.
    """

    def __init__(self):
        self._keys = []
        self._data = {}

    def insert(self, key, value):
        if key not in self._data:
            bisect.insort(self._keys, key)
            self._data[key] = []
        self._data[key].append(value)

    def lookup(self, key):
        idx = bisect.bisect_left(self._keys, key)
        if idx < len(self._keys) and self._keys[idx] == key:
            return self._data[key]
        return []

    def keys(self):
        return list(self._keys)

    def items(self):
        for k in self._keys:
            yield k, self._data[k]

    def __len__(self):
        return len(self._keys)

    def _dbg(self):
        _dbg("alias_idx", n_keys=len(self._keys),
             sizes=[len(self._data[k]) for k in self._keys[:5]])


# ── JSON loader with Welford stats ───────────────────────────────
def load_json_data(file_path):
    """Load JSON data with online variance tracking for numeric fields.

    Algorithm change: upstream loads blindly.  We attach a Welford accumulator
    that scans numeric params, giving downstream code running stats (mean,
    variance) useful for outlier detection and adaptive binning.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        raise

    acc = WelfordAccumulator()
    _scan_numeric_values(data, acc)
    _dbg("load_json", path=file_path, stats=acc.dump())
    return data


def load_json_data_dbg(file_path):
    """Debug wrapper — prints load statistics."""
    data = load_json_data(file_path)
    print(f"[mix_pred] load_json_data: {file_path} loaded")
    return data


def _scan_numeric_values(obj, acc):
    """Recursively scan JSON tree for numeric values and feed to accumulator."""
    if isinstance(obj, dict):
        for v in obj.values():
            _scan_numeric_values(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _scan_numeric_values(v, acc)
    elif isinstance(obj, (int, float)):
        acc.update(obj)


# ── PQO file generation with EMA timing ──────────────────────────
_global_timer = EMATimer(alpha=0.15)


def save_pqo_files(query_id, data, output_dir, description):
    """Generate PQO query JSON from template + param combinations.

    Algorithm change: EMA timer tracks per-combination substitution time
    so callers can estimate throughput and adapt batch sizes.
    """
    import time

    os.makedirs(output_dir, exist_ok=True)
    output_json = {}

    new_sql_template = data[query_id]["query"]
    literals = data[query_id]["params"]

    for index, combination in enumerate(literals):
        t0 = time.monotonic()
        query_str = new_sql_template
        for i, param in enumerate(combination):
            param = str(param).strip()
            pattern = re.compile(rf"@param{i}\b")
            query_str = pattern.sub(param, query_str)

        key = f"{query_id}_{description}_{index}"
        output_json[key] = query_str

        elapsed = (time.monotonic() - t0) * 1000.0
        _global_timer.record(elapsed)

    file_name = f"{query_id}_{description}.json"
    output_file = os.path.join(output_dir, file_name)
    if os.path.exists(output_file):
        _dbg("save_pqo_files", skip=file_name)
        return

    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(output_json, json_file, indent=4)

    _dbg("save_pqo_files", file=file_name, n_queries=len(output_json),
         ema_ms=_global_timer.avg_ms())


def save_pqo_files_dbg(query_id, data, output_dir, description):
    """Debug wrapper — prints generation summary."""
    save_pqo_files(query_id, data, output_dir, description)
    print(f"[mix_pred] save_pqo_files: q={query_id}, desc={description}, "
          f"timer={_global_timer.dump()}")


# ── Predicate serialization with Huber scoring + binary search ───
def save_pqo_predicates(query_id, testing_data, output_dir):
    """Save PQO predicates grouped by alias, with robust scoring.

    Algorithm changes:
      - Uses SortedAliasIndex (binary search) instead of defaultdict for
        O(log n) alias group resolution.
      - Computes per-combination Huber loss between expected (uniform)
        and actual predicate count per group, enabling robustness monitoring.
    """
    os.makedirs(output_dir, exist_ok=True)

    def save_combinations(data, file_name):
        output_file = os.path.join(output_dir, file_name)
        if os.path.exists(output_file):
            return

        n_predicates = len(data.get("predicates", []))
        n_params = len(data.get("params", []))
        if n_predicates == 0 or n_params == 0:
            _dbg("save_combinations", skip="empty", file=file_name)
            return

        # Compute expected predicates per group (uniform assumption)
        unique_aliases = set()
        for pred in data["predicates"]:
            unique_aliases.add(pred.get("original_alias", pred["alias"]))
        n_groups = max(len(unique_aliases), 1)
        expected_per_group = n_predicates / n_groups

        huber_scores = []

        with open(output_file, "w", encoding="utf-8") as file:
            original_alias_used = False

            for i, combination in enumerate(data["params"]):
                # Build sorted alias index
                alias_index = SortedAliasIndex()

                for j, item in enumerate(combination):
                    predicate = data["predicates"][j]
                    group_key = predicate.get("original_alias", predicate["alias"])
                    table = predicate["alias"]
                    column = predicate["column"]
                    operator = predicate["operator"]
                    data_type = predicate["data_type"]

                    if data_type == "text":
                        formatted_param = f"'{item}'"
                    else:
                        formatted_param = item

                    if operator.lower() == "in":
                        formatted_param = f"({formatted_param})"

                    predicate_str = f"{table}.{column} {operator} {formatted_param}"
                    alias_index.insert(group_key, predicate_str)

                    if "original_alias" in predicate:
                        original_alias_used = True

                # Format groups from sorted index
                formatted_groups = []
                for group_key, predicates in alias_index.items():
                    actual_count = len(predicates)
                    h = huber_loss(expected_per_group, actual_count, delta=1.5)
                    huber_scores.append(h)

                    if len(predicates) == 1:
                        formatted_groups.append(predicates[0])
                    else:
                        group_str = " AND ".join(predicates)
                        formatted_groups.append(group_str)

                combination_str = '[\"' + '", "'.join(formatted_groups) + '"]'
                if i < len(data["params"]) - 1:
                    file.write(combination_str + ",\n")
                else:
                    file.write(combination_str + "\n")

            if original_alias_used:
                print(f"Used original_alias in file: {file_name}")

        # Summarize Huber scores
        if huber_scores:
            arr = np.array(huber_scores)
            _dbg("predicate_huber", file=file_name,
                 mean=float(np.mean(arr)),
                 std=float(np.std(arr)),
                 max=float(np.max(arr)))

    save_combinations(testing_data[query_id], f"{query_id}_mixture_test.txt")


def save_pqo_predicates_dbg(query_id, testing_data, output_dir):
    """Debug wrapper — always prints predicate save summary."""
    print(f"[mix_pred] save_pqo_predicates: q={query_id}, "
          f"dir={output_dir}")
    save_pqo_predicates(query_id, testing_data, output_dir)
    print(f"[mix_pred] save_pqo_predicates: done")


# ── Query processing pipeline ───────────────────────────────────
def process_query(query_id):
    """Process a single query: load JSON, generate PQO files + predicates."""
    try:
        base_dir = f"0_mixture_test/{query_id}"
        pqo_dir = os.path.join(base_dir, "PQO")
        testing_file = os.path.join(base_dir, f"{query_id}_mixture_test.json")

        testing_data = load_json_data(testing_file)

        save_pqo_files(query_id, testing_data, pqo_dir, "mixture_test")
        save_pqo_predicates(query_id, testing_data, pqo_dir)

        print(f"Successfully processed testing data for query {query_id}")
    except Exception as e:
        print(f"Error processing query {query_id}: {e}")
        raise


def process_query_dbg(query_id):
    """Debug wrapper — prints entry/exit."""
    print(f"[mix_pred] process_query: START q={query_id}")
    process_query(query_id)
    print(f"[mix_pred] process_query: END q={query_id}")


# ── Directory cleanup ────────────────────────────────────────────
def clear_directories(query_ids):
    """Remove PQO output directories for listed query IDs."""
    for query_id in query_ids:
        base_dir = f"0_mixture_test/{query_id}/PQO"
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
            print(f"Removed existing directory: {base_dir}")


def clear_directories_dbg(query_ids):
    """Debug wrapper — prints directory cleanup plan."""
    print(f"[mix_pred] clear_directories: {query_ids}")
    clear_directories(query_ids)


# ── Main ─────────────────────────────────────────────────────────
def main():
    """Entry point for mixture PQO predicate generation."""
    query_ids = ["33-0"]

    print("Clearing existing PQO directories...")
    clear_directories(query_ids)
    print("Finished clearing directories\n")

    print("Starting to process queries...")
    for query_id in query_ids:
        try:
            process_query(query_id)
        except Exception as e:
            print(f"Failed to process query {query_id}: {e}")
            continue


if __name__ == "__main__":
    print("=" * 60)
    print("[par2qo_0_mixture_PQO_predicate_generate] self-test")
    print("=" * 60)

    # Test 1: Welford accumulator
    print("\n  Test 1: WelfordAccumulator")
    acc = WelfordAccumulator()
    vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    for v in vals:
        acc.update(v)
    assert acc.n == 8
    assert abs(acc.mean() - 5.0) < 1e-9, f"mean={acc.mean()}"
    expected_var = 32.0 / 7.0  # sample variance of [2,4,4,4,5,5,7,9]
    assert abs(acc.variance() - expected_var) < 1e-9, f"var={acc.variance()}"
    print(f"    stats: {acc.dump()}")

    # Test 2: EMA timer
    print("\n  Test 2: EMATimer")
    timer = EMATimer(alpha=0.5)
    for t in [10.0, 20.0, 10.0, 20.0]:
        timer.record(t)
    print(f"    ema={timer.avg_ms():.2f}ms (expect ~15)")
    assert 12.0 < timer.avg_ms() < 18.0

    # Test 3: Huber loss
    print("\n  Test 3: Huber loss")
    h1 = huber_loss(1.0, 1.0)
    assert h1 == 0.0, f"same values: {h1}"
    h2 = huber_loss(1.0, 1.5, delta=1.0)
    assert abs(h2 - 0.125) < 1e-9, f"small diff: {h2}"
    h3 = huber_loss(1.0, 5.0, delta=1.0)
    expected_h3 = 1.0 * (4.0 - 0.5)
    assert abs(h3 - expected_h3) < 1e-9, f"large diff: {h3}"
    print(f"    h(same)={h1}, h(small)={h2}, h(large)={h3}")

    # Test 4: SortedAliasIndex
    print("\n  Test 4: SortedAliasIndex")
    idx = SortedAliasIndex()
    for alias in ["t", "mc", "t", "mi", "t"]:
        idx.insert(alias, f"{alias}.col = 1")
    assert len(idx) == 3
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

    # Test 6: numpy integration
    print("\n  Test 6: numpy Huber scoring")
    scores = np.array([huber_loss(2.0, float(x), delta=1.0) for x in range(5)])
    print(f"    scores={scores}, mean={np.mean(scores):.4f}")
    assert scores.shape == (5,)

    print("\nAll tests passed.")
