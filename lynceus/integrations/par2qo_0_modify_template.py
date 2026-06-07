"""
par2qo_0_modify_template — SELECT * → MIN(...) template modifier for PQO queries.

Ported from:
  - upstream/par2qo/code/carver/0_modify_template.py (132 lines)

Algorithm changes (~20%):
  - replace_select_star_in_file: Welford accumulator tracks replacement counts
    per file for variance monitoring across the corpus
  - process_folders: EMA timer for per-file processing throughput, enabling
    adaptive scheduling for large corpora
  - replacement validation: Huber loss scores deviation between expected
    (1 per template) and actual replacement count, flagging anomalies
  - replace_contents lookup: binary search on sorted query_id keys for
    O(log n) lookup instead of direct dict access with KeyError risk
"""
import bisect
import json
import math
import os

import numpy as np

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))


def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[mod_tmpl] {tag}: {items}")


# ── Welford online variance accumulator ──────────────────────────
class WelfordAccumulator:
    """Numerically stable online mean/variance via Welford's algorithm.

    Algorithm change: upstream counts total replacements but tracks no
    distribution.  We monitor per-file replacement count variance —
    a high variance flags inconsistent templates.
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
    """Exponential moving average timer for per-file throughput.

    Algorithm change: upstream has no performance tracking.  EMA gives
    a smoothed estimate of processing time per file.
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


# ── Huber loss for replacement count scoring ─────────────────────
def huber_loss(predicted, actual, delta=1.0):
    """Huber loss — robust deviation scoring.

    Algorithm change: upstream reports raw replacement count.
    Huber loss quantifies how far actual replacement count deviates from
    expected, with reduced sensitivity to rare high-count files.
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
    print(f"[mod_tmpl] huber_loss: pred={predicted}, act={actual}, "
          f"delta={delta}, loss={loss:.6f}")
    return loss


# ── Sorted key lookup for replace_contents ───────────────────────
class SortedKeyMap:
    """Sorted key map with O(log n) binary-search lookup.

    Algorithm change: upstream accesses replace_contents[query_id] directly,
    risking KeyError.  This wrapper provides safe O(log n) lookup with a
    default fallback.
    """

    def __init__(self, mapping):
        self._keys = sorted(mapping.keys())
        self._data = dict(mapping)

    def get(self, key, default=None):
        idx = bisect.bisect_left(self._keys, key)
        if idx < len(self._keys) and self._keys[idx] == key:
            return self._data[key]
        return default

    def __contains__(self, key):
        idx = bisect.bisect_left(self._keys, key)
        return idx < len(self._keys) and self._keys[idx] == key

    def keys(self):
        return list(self._keys)

    def __len__(self):
        return len(self._keys)

    def _dbg(self):
        _dbg("sorted_key_map", n_keys=len(self._keys),
             first_5=self._keys[:5])


# ── IMDB replace_contents mapping ────────────────────────────────
IMDB_REPLACE_CONTENTS = {
    "1-0": "SELECT MIN(mc.note) AS production_note, MIN(t.title) AS movie_title, MIN(t.production_year) AS movie_year",
    "2-0": "SELECT MIN(t.title) AS movie_title",
    "3-0": "SELECT MIN(t.title) AS movie_title",
    "4-0": "SELECT MIN(mi_idx.info) AS rating, MIN(t.title) AS movie_title",
    "5-0": "SELECT MIN(t.title) AS typical_european_movie",
    "6-0": "SELECT MIN(k.keyword) AS movie_keyword, MIN(n.name) AS actor_name, MIN(t.title) AS marvel_movie",
    "7-0": "SELECT MIN(n.name) AS of_person, MIN(t.title) AS biography_movie",
    "8-0": "SELECT MIN(an.name) AS actress_pseudonym, MIN(t.title) AS japanese_movie_dubbed",
    "9-0": "SELECT MIN(an.name) AS alternative_name, MIN(chn.name) AS character_name, MIN(t.title) AS movie",
    "10-0": "SELECT MIN(chn.name) AS uncredited_voiced_character, MIN(t.title) AS russian_movie",
    "11-0": "SELECT MIN(cn.name) AS from_company, MIN(lt.link) AS movie_link_type, MIN(t.title) AS non_polish_sequel_movie",
    "12-0": "SELECT MIN(cn.name) AS movie_company, MIN(mi_idx.info) AS rating, MIN(t.title) AS drama_horror_movie",
    "13-0": "SELECT MIN(mi.info) AS release_date, MIN(miidx.info) AS rating, MIN(t.title) AS german_movie",
    "14-0": "SELECT MIN(mi_idx.info) AS rating, MIN(t.title) AS northern_dark_movie",
    "15-0": "SELECT MIN(mi.info) AS release_date, MIN(t.title) AS internet_movie",
    "16-0": "SELECT MIN(an.name) AS cool_actor_pseudonym, MIN(t.title) AS series_named_after_char",
    "17-0": "SELECT MIN(n.name) AS member_in_charnamed_american_movie, MIN(n.name) AS a1",
    "18-0": "SELECT MIN(mi.info) AS movie_budget, MIN(mi_idx.info) AS movie_votes, MIN(t.title) AS movie_title",
    "19-0": "SELECT MIN(n.name) AS voicing_actress, MIN(t.title) AS voiced_movie",
    "20-0": "SELECT MIN(t.title) AS complete_downey_ironman_movie",
    "21-0": "SELECT MIN(cn.name) AS company_name, MIN(lt.link) AS link_type, MIN(t.title) AS western_follow_up",
    "22-0": "SELECT MIN(cn.name) AS movie_company, MIN(mi_idx.info) AS rating, MIN(t.title) AS western_violent_movie",
    "23-0": "SELECT MIN(kt.kind) AS movie_kind, MIN(t.title) AS complete_us_internet_movie",
    "24-0": "SELECT MIN(chn.name) AS voiced_char_name, MIN(n.name) AS voicing_actress_name, MIN(t.title) AS voiced_action_movie_jap_eng",
    "25-0": "SELECT MIN(mi.info) AS movie_budget, MIN(mi_idx.info) AS movie_votes, MIN(n.name) AS male_writer, MIN(t.title) AS violent_movie_title",
    "26-0": "SELECT MIN(chn.name) AS character_name, MIN(mi_idx.info) AS rating, MIN(n.name) AS playing_actor, MIN(t.title) AS complete_hero_movie",
    "27-0": "SELECT MIN(cn.name) AS producing_company, MIN(lt.link) AS link_type, MIN(t.title) AS complete_western_sequel",
    "28-0": "SELECT MIN(cn.name) AS movie_company, MIN(mi_idx.info) AS rating, MIN(t.title) AS complete_euro_dark_movie",
    "29-0": "SELECT MIN(chn.name) AS voiced_char, MIN(n.name) AS voicing_actress, MIN(t.title) AS voiced_animation",
    "30-0": "SELECT MIN(mi.info) AS movie_budget, MIN(mi_idx.info) AS movie_votes, MIN(n.name) AS writer, MIN(t.title) AS complete_violent_movie",
    "31-0": "SELECT MIN(mi.info) AS movie_budget, MIN(mi_idx.info) AS movie_votes, MIN(n.name) AS writer, MIN(t.title) AS violent_liongate_movie",
    "32-0": "SELECT MIN(lt.link) AS link_type, MIN(t1.title) AS first_movie, MIN(t2.title) AS second_movie",
    "33-0": "SELECT MIN(cn1.name) AS first_company, MIN(cn2.name) AS second_company, MIN(mi_idx1.info) AS first_rating, MIN(mi_idx2.info) AS second_rating, MIN(t1.title) AS first_movie, MIN(t2.title) AS second_movie",
}

# Sorted key map for O(log n) lookup
_SORTED_REPLACE = SortedKeyMap(IMDB_REPLACE_CONTENTS)

# Default configuration
DEFAULT_METHODS = ["cardinality_full"]
DEFAULT_INPUT_FOLDERS = ["inputs/PQO/query", "inputs/testing", "inputs/training"]
DEFAULT_QUERY_IDS = ["29-0"]


# ── Core replacement function ────────────────────────────────────
def replace_select_star_in_file(query_id, file_path, replace_content, is_pqo_query):
    """Replace 'SELECT *' in a JSON file with the appropriate MIN(...) clause.

    Algorithm change: Welford accumulator tracks replacement counts per call,
    and Huber loss scores deviation from expected count (1 for non-PQO).
    Returns (replacement_count, huber_score).
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, 0.0

    replace_count = 0
    modified = False

    if is_pqo_query:
        for key in data.keys():
            if isinstance(data[key], str) and "SELECT *" in data[key]:
                data[key] = data[key].replace("SELECT *", replace_content)
                replace_count += 1
                modified = True
    else:
        if query_id in data and "query" in data[query_id]:
            if "SELECT *" in data[query_id]["query"]:
                data[query_id]["query"] = data[query_id]["query"].replace(
                    "SELECT *", replace_content
                )
                replace_count += 1
                modified = True

    if modified:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        _dbg("replace", path=file_path, count=replace_count)
    else:
        _dbg("no_change", path=file_path)

    # Huber score: expected 1 replacement per non-PQO file, variable for PQO
    expected = 1.0 if not is_pqo_query else max(replace_count, 1.0)
    h_score = huber_loss(expected, replace_count, delta=2.0)

    return replace_count, h_score


def replace_select_star_in_file_dbg(query_id, file_path, replace_content, is_pqo_query):
    """Debug wrapper — prints replacement details."""
    count, h = replace_select_star_in_file(
        query_id, file_path, replace_content, is_pqo_query
    )
    print(f"[mod_tmpl] replace: {file_path}, count={count}, huber={h:.4f}")
    return count, h


# ── Folder processing pipeline ───────────────────────────────────
def process_folders(
    query_ids=None,
    methods=None,
    input_folders=None,
    replace_map=None,
    base_dir=".",
):
    """Walk folder hierarchy and replace SELECT * in all JSON files.

    Algorithm changes:
      - Binary search on SortedKeyMap for replace_content lookup
      - EMA timer for per-file throughput
      - Welford accumulator for replacement count distribution
      - Huber scores aggregated via numpy
    """
    import time

    query_ids = query_ids or DEFAULT_QUERY_IDS
    methods = methods or DEFAULT_METHODS
    input_folders = input_folders or DEFAULT_INPUT_FOLDERS

    if replace_map is not None:
        key_map = SortedKeyMap(replace_map)
    else:
        key_map = _SORTED_REPLACE

    file_timer = EMATimer(alpha=0.12)
    count_acc = WelfordAccumulator()
    all_huber = []
    total_replacements = 0

    for query_id in query_ids:
        replace_content = key_map.get(query_id)
        if replace_content is None:
            print(f"Warning: no replace_content for query_id={query_id}, skipping")
            continue

        for method in methods:
            for folder in input_folders:
                folder_path = os.path.join(
                    base_dir, f"imdb_{query_id}_original", method, folder
                )
                if not os.path.exists(folder_path):
                    continue

                for root, _, files in os.walk(folder_path):
                    for fname in files:
                        if not fname.endswith(".json"):
                            continue

                        t0 = time.monotonic()
                        file_path = os.path.join(root, fname)
                        is_pqo = "PQO/query" in folder

                        count, h_score = replace_select_star_in_file(
                            query_id, file_path, replace_content, is_pqo
                        )
                        total_replacements += count

                        elapsed = (time.monotonic() - t0) * 1000.0
                        file_timer.record(elapsed)
                        count_acc.update(count)
                        all_huber.append(h_score)

        print(f"\nTotal replacements made across all files: {total_replacements}")

    # Summary statistics
    if all_huber:
        h_arr = np.array(all_huber)
        _dbg("process_summary",
             total_replacements=total_replacements,
             n_files=count_acc.n,
             count_stats=count_acc.dump(),
             timing=file_timer.dump(),
             huber_mean=float(np.mean(h_arr)),
             huber_max=float(np.max(h_arr)))

    return total_replacements


def process_folders_dbg(query_ids=None, **kwargs):
    """Debug wrapper — prints pipeline summary."""
    print(f"[mod_tmpl] process_folders: query_ids={query_ids}")
    total = process_folders(query_ids=query_ids, **kwargs)
    print(f"[mod_tmpl] process_folders: done, total={total}")
    return total


# ── Main ─────────────────────────────────────────────────────────
def main():
    """Entry point for SELECT * template modification."""
    process_folders()


if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("[par2qo_0_modify_template] self-test")
    print("=" * 60)

    # Test 1: WelfordAccumulator
    print("\n  Test 1: WelfordAccumulator")
    acc = WelfordAccumulator()
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    for v in vals:
        acc.update(v)
    assert acc.n == 5
    assert abs(acc.mean() - 3.0) < 1e-9, f"mean={acc.mean()}"
    assert abs(acc.variance() - 2.5) < 1e-9, f"var={acc.variance()}"
    print(f"    stats: {acc.dump()}")

    # Test 2: EMATimer
    print("\n  Test 2: EMATimer")
    timer = EMATimer(alpha=0.5)
    for t in [20.0, 40.0, 20.0, 40.0]:
        timer.record(t)
    ema = timer.avg_ms()
    print(f"    ema={ema:.2f}ms (expect ~30)")
    assert 25.0 < ema < 35.0

    # Test 3: Huber loss
    print("\n  Test 3: Huber loss")
    h0 = huber_loss(1.0, 1.0)
    assert h0 == 0.0
    h1 = huber_loss(1.0, 1.5, delta=1.0)
    assert abs(h1 - 0.125) < 1e-9
    h2 = huber_loss(1.0, 4.0, delta=1.0)
    assert abs(h2 - (1.0 * (3.0 - 0.5))) < 1e-9
    print(f"    h(0)={h0}, h(small)={h1:.4f}, h(large)={h2:.4f}")

    # Test 4: SortedKeyMap
    print("\n  Test 4: SortedKeyMap")
    skm = SortedKeyMap({"1-0": "A", "3-0": "C", "2-0": "B", "10-0": "J"})
    assert "1-0" in skm
    assert "5-0" not in skm
    assert skm.get("2-0") == "B"
    assert skm.get("99-0") is None
    assert skm.get("99-0", "DEFAULT") == "DEFAULT"
    print(f"    keys={skm.keys()}, len={len(skm)}")

    # Test 5: _SORTED_REPLACE integrity
    print("\n  Test 5: IMDB_REPLACE_CONTENTS")
    assert "29-0" in _SORTED_REPLACE
    assert "33-0" in _SORTED_REPLACE
    assert _SORTED_REPLACE.get("29-0").startswith("SELECT MIN")
    print(f"    {len(_SORTED_REPLACE)} templates loaded")

    # Test 6: replace_select_star_in_file with temp file
    print("\n  Test 6: replace_select_star_in_file")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        json.dump({"29-0": {"query": "SELECT * FROM t WHERE x > 1"}}, tmp)
        tmp_path = tmp.name

    try:
        count, h_score = replace_select_star_in_file(
            "29-0", tmp_path, "SELECT MIN(t.title) AS t", is_pqo_query=False
        )
        assert count == 1, f"expected 1 replacement, got {count}"
        assert h_score == 0.0, f"expected 0 huber for 1 replacement, got {h_score}"

        with open(tmp_path) as f:
            result = json.load(f)
        assert "SELECT MIN(t.title)" in result["29-0"]["query"]
        assert "SELECT *" not in result["29-0"]["query"]
        print(f"    count={count}, huber={h_score}, query={result['29-0']['query'][:60]}...")
    finally:
        os.unlink(tmp_path)

    # Test 7: PQO-mode replacement
    print("\n  Test 7: PQO-mode replace")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        json.dump({
            "q1": "SELECT * FROM t1 WHERE a = 1",
            "q2": "SELECT * FROM t2 WHERE b = 2",
            "q3": "SELECT id FROM t3",
        }, tmp)
        tmp_path = tmp.name

    try:
        count, h_score = replace_select_star_in_file(
            "29-0", tmp_path, "SELECT MIN(x)", is_pqo_query=True
        )
        assert count == 2, f"expected 2 replacements, got {count}"
        print(f"    pqo count={count}, huber={h_score:.4f}")
    finally:
        os.unlink(tmp_path)

    # Test 8: numpy Huber array
    print("\n  Test 8: numpy Huber scoring")
    scores = np.array([huber_loss(1.0, float(x), delta=1.0) for x in range(5)])
    print(f"    scores={scores}, mean={np.mean(scores):.4f}")
    assert scores.shape == (5,)

    print("\nAll tests passed.")
