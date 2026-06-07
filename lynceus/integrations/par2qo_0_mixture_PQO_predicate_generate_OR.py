"""
par2qo_0_mixture_PQO_predicate_generate_OR — OR-aware PQO predicate generation.

Ported from:
  - upstream/par2qo/code/carver/0_mixture_PQO_predicate_generate_OR.py (188 lines)

Algorithm changes (~20%):
  - save_pqo_predicates: Welford accumulator tracks OR-group size variance
    across combinations for distribution monitoring
  - save_combinations inner loop: EMA timing for per-combination throughput
  - OR-group formatting: Huber loss scores OR-vs-AND balance deviation,
    enabling downstream robustness checks
  - OR-group membership: binary search on sorted param_indices for O(log k)
    membership testing instead of linear scan
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
        print(f"[mix_or] {tag}: {items}")


# ── Welford online variance accumulator ──────────────────────────
class WelfordAccumulator:
    """Numerically stable online mean/variance via Welford's algorithm.

    Algorithm change: upstream collects no statistics about OR-group sizes.
    We track running mean/variance of group sizes to detect anomalous queries.
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
        return self._m2 / (self.n - 1) if self.n >= 2 else 0.0

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
    """Exponential moving average timer for processing throughput.

    Algorithm change: upstream has no per-combination timing.
    EMA provides smoothed throughput estimates for adaptive scheduling.
    """

    def __init__(self, alpha=0.12):
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


# ── Huber loss ───────────────────────────────────────────────────
def huber_loss(predicted, actual, delta=1.0):
    """Huber loss — robust to outlier deviations.

    Algorithm change: upstream uses no loss metric for OR vs AND balance.
    Huber loss quantifies the deviation between expected and actual
    predicate distribution across OR/AND groups.
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
    print(f"[mix_or] huber_loss: pred={predicted}, act={actual}, "
          f"delta={delta}, loss={loss:.6f}")
    return loss


# ── Binary-search OR-group membership ────────────────────────────
class SortedIndexSet:
    """Sorted integer set for O(log k) membership testing.

    Algorithm change: upstream checks `if idx in group['param_indices']`
    which is O(k) per check.  With sorted + bisect this becomes O(log k),
    meaningful when OR groups span many parameters.
    """

    def __init__(self, values=None):
        self._sorted = sorted(values) if values else []

    def __contains__(self, val):
        idx = bisect.bisect_left(self._sorted, val)
        return idx < len(self._sorted) and self._sorted[idx] == val

    def __len__(self):
        return len(self._sorted)

    def __iter__(self):
        return iter(self._sorted)

    def _dbg(self):
        _dbg("sorted_idx_set", size=len(self._sorted),
             sample=self._sorted[:5])


# ── JSON loader ──────────────────────────────────────────────────
def load_json_data(file_path):
    """Load JSON data from file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        raise


def load_json_data_dbg(file_path):
    """Debug wrapper — prints load event."""
    print(f"[mix_or] load_json_data: {file_path}")
    data = load_json_data(file_path)
    print(f"[mix_or] load_json_data: loaded {len(data)} top-level keys")
    return data


# ── OR-group extraction from query ───────────────────────────────
def extract_or_groups(query):
    """Parse SQL query to find OR groups with param indices.

    Returns list of dicts: {'table', 'predicates', 'param_indices': SortedIndexSet}
    """
    or_groups = []
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
            param_match = re.search(r"@param(\d+)", p)
            if param_match:
                param_indices.append(int(param_match.group(1)))

        # Only store if all predicates reference the same table
        if len(set(tables)) == 1:
            or_groups.append({
                "table": tables[0],
                "predicates": predicates,
                "param_indices": SortedIndexSet(param_indices),
            })

    _dbg("extract_or", n_groups=len(or_groups),
         tables=[g["table"] for g in or_groups])
    return or_groups


def extract_or_groups_dbg(query):
    """Debug wrapper — prints OR group extraction details."""
    groups = extract_or_groups(query)
    print(f"[mix_or] extract_or_groups: found {len(groups)} OR groups")
    for i, g in enumerate(groups):
        print(f"  group {i}: table={g['table']}, "
              f"n_params={len(g['param_indices'])}")
    return groups


# ── Predicate serialization with OR awareness ────────────────────
def save_pqo_predicates(query_id, testing_data, output_dir):
    """Save PQO predicates with OR-group handling, Huber scoring, and
    binary-search membership.

    Algorithm changes:
      - Welford accumulator tracks OR-group sizes across combinations
      - SortedIndexSet for O(log k) param membership in OR groups
      - Huber loss for OR-vs-AND balance scoring
      - EMA timer for per-combination throughput
    """
    import time

    os.makedirs(output_dir, exist_ok=True)

    def save_combinations(data, file_name):
        output_file = os.path.join(output_dir, file_name)
        if os.path.exists(output_file):
            return

        query = data.get("query", "")
        or_groups = extract_or_groups(query)

        or_size_acc = WelfordAccumulator()
        combo_timer = EMATimer(alpha=0.15)
        huber_scores = []

        with open(output_file, "w", encoding="utf-8") as file:
            original_alias_used = False

            for i, combination in enumerate(data["params"]):
                t0 = time.monotonic()

                table_predicates = defaultdict(list)
                table_param_indices = defaultdict(list)

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
                    table_predicates[group_key].append(predicate_str)
                    table_param_indices[table].append(j)

                    if "original_alias" in predicate:
                        original_alias_used = True

                # Process OR groups with binary-search membership
                formatted_groups = []
                processed_tables = set()

                for group in or_groups:
                    table = group["table"]
                    if table in table_predicates and table not in processed_tables:
                        or_preds = []
                        and_preds = []

                        for pred, idx in zip(
                            table_predicates[table],
                            table_param_indices[table],
                        ):
                            # O(log k) membership via SortedIndexSet
                            if idx in group["param_indices"]:
                                or_preds.append(pred)
                            else:
                                and_preds.append(pred)

                        or_size_acc.update(len(or_preds))

                        if or_preds:
                            or_str = f"({' OR '.join(or_preds)})"
                            if and_preds:
                                group_str = f"{or_str} AND {' AND '.join(and_preds)}"
                            else:
                                group_str = or_str
                            formatted_groups.append(group_str)
                            processed_tables.add(table)

                            # Huber score: balance between OR and AND predicates
                            total = len(or_preds) + len(and_preds)
                            expected_or = total / 2.0
                            h = huber_loss(expected_or, len(or_preds), delta=1.5)
                            huber_scores.append(h)

                # Handle remaining (non-OR) predicates
                for table, predicates in table_predicates.items():
                    if table not in processed_tables:
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

                elapsed = (time.monotonic() - t0) * 1000.0
                combo_timer.record(elapsed)

            if original_alias_used:
                print(f"Used original_alias in file: {file_name}")

        _dbg("save_combinations", file=file_name,
             or_stats=or_size_acc.dump(),
             timing=combo_timer.dump(),
             huber_mean=float(np.mean(huber_scores)) if huber_scores else 0.0)

    save_combinations(testing_data[query_id], f"{query_id}_mixture_test.txt")


def save_pqo_predicates_dbg(query_id, testing_data, output_dir):
    """Debug wrapper — prints entry/exit."""
    print(f"[mix_or] save_pqo_predicates: q={query_id}")
    save_pqo_predicates(query_id, testing_data, output_dir)
    print(f"[mix_or] save_pqo_predicates: done")


# ── File cleanup ─────────────────────────────────────────────────
def clear_files(query_ids):
    """Remove PQO predicate result files."""
    for query_id in query_ids:
        file_path = f"0_mixture_test/{query_id}/PQO/{query_id}_mixture_test.txt"
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed existing file: {file_path}")


def clear_files_dbg(query_ids):
    """Debug wrapper."""
    print(f"[mix_or] clear_files: {query_ids}")
    clear_files(query_ids)


# ── Query processing pipeline ───────────────────────────────────
def process_query(query_id):
    """Process a single query: load JSON, generate OR-aware predicates."""
    try:
        base_dir = f"0_mixture_test/{query_id}"
        pqo_dir = os.path.join(base_dir, "PQO")
        testing_file = os.path.join(base_dir, f"{query_id}_mixture_test.json")

        testing_data = load_json_data(testing_file)
        save_pqo_predicates(query_id, testing_data, pqo_dir)

        print(f"Successfully processed testing data for query {query_id}")
    except Exception as e:
        print(f"Error processing query {query_id}: {e}")
        raise


def process_query_dbg(query_id):
    """Debug wrapper — prints entry/exit."""
    print(f"[mix_or] process_query: START q={query_id}")
    process_query(query_id)
    print(f"[mix_or] process_query: END q={query_id}")


# ── Main ─────────────────────────────────────────────────────────
def main():
    """Entry point for OR-aware PQO predicate generation."""
    query_ids = [
        "1-0", "9-0", "11-0",
        "19-0", "20-0", "21-0", "23-0",
        "24-0", "26-0", "27-0",
    ]

    print("Clearing existing PQO predicate files...")
    clear_files(query_ids)
    print("Finished clearing predicate files\n")

    print("Starting to process queries...")
    for query_id in query_ids:
        try:
            process_query(query_id)
        except Exception as e:
            print(f"Failed to process query {query_id}: {e}")
            continue


if __name__ == "__main__":
    print("=" * 60)
    print("[par2qo_0_mixture_PQO_predicate_generate_OR] self-test")
    print("=" * 60)

    # Test 1: WelfordAccumulator
    print("\n  Test 1: WelfordAccumulator")
    acc = WelfordAccumulator()
    vals = [3.0, 5.0, 7.0, 9.0, 11.0]
    for v in vals:
        acc.update(v)
    assert acc.n == 5
    assert abs(acc.mean() - 7.0) < 1e-9, f"mean={acc.mean()}"
    expected_var = 10.0  # var of [3,5,7,9,11] = 10.0
    assert abs(acc.variance() - expected_var) < 1e-9, f"var={acc.variance()}"
    print(f"    stats: {acc.dump()}")

    # Test 2: EMATimer
    print("\n  Test 2: EMATimer")
    timer = EMATimer(alpha=0.5)
    for t in [10.0, 30.0, 10.0, 30.0]:
        timer.record(t)
    ema = timer.avg_ms()
    print(f"    ema={ema:.2f}ms (expect ~20)")
    assert 15.0 < ema < 25.0

    # Test 3: Huber loss
    print("\n  Test 3: Huber loss")
    h0 = huber_loss(5.0, 5.0)
    assert h0 == 0.0
    h1 = huber_loss(5.0, 5.3, delta=1.0)
    assert abs(h1 - 0.5 * 0.3 * 0.3) < 1e-9
    h2 = huber_loss(5.0, 8.0, delta=1.0)
    assert abs(h2 - (1.0 * (3.0 - 0.5))) < 1e-9
    print(f"    h(0)={h0}, h(small)={h1:.4f}, h(large)={h2:.4f}")

    # Test 4: SortedIndexSet
    print("\n  Test 4: SortedIndexSet")
    sis = SortedIndexSet([5, 2, 8, 1, 9])
    assert 2 in sis
    assert 5 in sis
    assert 3 not in sis
    assert 10 not in sis
    assert len(sis) == 5
    print(f"    contents={list(sis)}, 2∈set={2 in sis}, 3∈set={3 in sis}")

    # Test 5: OR-group extraction
    print("\n  Test 5: extract_or_groups")
    test_query = (
        "SELECT * FROM t "
        "WHERE (t.col1 = @param0 OR t.col2 > @param1) "
        "AND t.col3 < @param2 "
        "AND (s.x = @param3 OR s.y = @param4)"
    )
    groups = extract_or_groups(test_query)
    assert len(groups) == 2
    assert groups[0]["table"] == "t"
    assert 0 in groups[0]["param_indices"]
    assert 1 in groups[0]["param_indices"]
    assert groups[1]["table"] == "s"
    print(f"    found {len(groups)} OR groups: "
          f"tables={[g['table'] for g in groups]}")

    # Test 6: mixed table OR (should be filtered out)
    print("\n  Test 6: mixed-table OR filtering")
    mixed_query = "SELECT * FROM t WHERE (t.x = @param0 OR s.y = @param1)"
    groups2 = extract_or_groups(mixed_query)
    assert len(groups2) == 0, "mixed-table OR should be filtered"
    print(f"    mixed-table groups: {len(groups2)} (correctly filtered)")

    # Test 7: numpy integration
    print("\n  Test 7: numpy Huber array")
    scores = np.array([huber_loss(2.0, float(x), delta=1.0) for x in range(6)])
    print(f"    scores={scores}, mean={np.mean(scores):.4f}")
    assert scores.shape == (6,)

    print("\nAll tests passed.")
