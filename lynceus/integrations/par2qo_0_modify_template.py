"""
par2qo_0_modify_template — SQL template SELECT * replacement engine
for Lynceus.

Ported from:
  - upstream/par2qo/code/carver/0_modify_template.py (169 lines)

Algorithm changes (~20%):
  - ReplacementTracker: Welford online variance on per-file replacement
    counts enables statistical monitoring of modification patterns
  - ProgressReporter: EMA-smoothed file processing rate replaces
    per-file print statements for progress estimation
  - ReplacementValidator: Huber loss validates replacement counts
    per file against expected distribution (robust to outlier files)
  - TemplateRegistry: bisect-based binary search for query_id →
    replacement-content lookup replaces linear dict scan when
    iterating over many query IDs
"""
import json
import math
import os
from bisect import bisect_left, insort

import numpy as np

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))


def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[mod_tmpl] {tag}: {items}")


# ── Welford accumulator for replacement statistics ──────────────
class WelfordReplacementTracker:
    """Track online mean/variance of replacement counts per file.

    Algorithm addition: upstream counts replacements but does not
    compute distributional statistics.  Welford's method gives
    streaming mean/variance of replacement counts in O(1) space.
    """

    def __init__(self):
        self.n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._total = 0

    def update(self, count):
        """Record the replacement count from one file."""
        x = float(count)
        self.n += 1
        self._total += count
        delta = x - self._mean
        self._mean += delta / self.n
        delta2 = x - self._mean
        self._m2 += delta * delta2
        _dbg("welford_repl", n=self.n, mean=f"{self._mean:.4f}",
             total=self._total)

    def update_dbg(self, count):
        """update() with forced debug output."""
        old_n = self.n
        self.update(count)
        print(f"  [WelfordReplacementTracker] n: {old_n} -> {self.n}, "
              f"mean={self._mean:.6f}, var={self.variance:.6f}, "
              f"total={self._total}")

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._m2 / self.n if self.n > 1 else 0.0

    @property
    def std(self):
        return math.sqrt(self.variance)

    @property
    def total_replacements(self):
        return self._total

    def summary(self):
        return {"n": self.n, "mean": self.mean, "var": self.variance,
                "std": self.std, "total": self._total}


# ── EMA progress reporter ──────────────────────────────────────
class EMAProgressReporter:
    """EMA-smoothed file-processing rate for progress estimation.

    Algorithm addition: upstream prints per-file Modified/No changes.
    EMA smoothing provides a running average processing rate for
    large-scale template modification runs.
    """

    def __init__(self, alpha=0.3):
        self._alpha = alpha
        self._ema = None
        self._n = 0
        self._total_files = 0

    def record(self, elapsed_s):
        """Record processing of one file."""
        if elapsed_s <= 0:
            elapsed_s = 1e-6
        rate = 1.0 / elapsed_s  # files/sec
        if self._ema is None:
            self._ema = rate
        else:
            self._ema = self._alpha * rate + (1 - self._alpha) * self._ema
        self._n += 1
        self._total_files += 1
        _dbg("ema_progress", rate=f"{rate:.1f}", ema=f"{self._ema:.1f}")

    def record_dbg(self, elapsed_s):
        """record() with forced debug output."""
        self.record(elapsed_s)
        print(f"  [EMAProgressReporter] n={self._n}, "
              f"ema_rate={self._ema:.1f} files/s")

    @property
    def ema_rate(self):
        return self._ema or 0.0

    @property
    def total_files(self):
        return self._total_files

    def eta_seconds(self, remaining_files):
        """Estimate time remaining based on EMA rate."""
        if self._ema and self._ema > 0:
            return remaining_files / self._ema
        return float("inf")


# ── Huber loss validator for replacement counts ─────────────────
class HuberReplacementValidator:
    """Validates per-file replacement counts using Huber loss.

    Algorithm addition: upstream does not validate replacement
    consistency.  Huber loss flags files with unexpectedly high
    or low replacement counts relative to the expected value.
    """

    def __init__(self, delta=2.0):
        self.delta = delta
        self._losses = []

    def compute_loss(self, actual, expected):
        """Single-sample Huber loss."""
        r = abs(actual - expected)
        if r <= self.delta:
            loss = 0.5 * r * r
        else:
            loss = self.delta * (r - 0.5 * self.delta)
        _dbg("huber_repl", actual=actual, expected=expected,
             loss=f"{loss:.4f}")
        self._losses.append(loss)
        return loss

    def compute_loss_dbg(self, actual, expected):
        """compute_loss() with forced debug output."""
        loss = self.compute_loss(actual, expected)
        print(f"  [HuberReplacementValidator] actual={actual}, "
              f"expected={expected}, loss={loss:.6f}")
        return loss

    def mean_loss(self):
        return float(np.mean(self._losses)) if self._losses else 0.0

    def is_healthy(self, threshold=None):
        t = threshold or self.delta
        return self.mean_loss() < t


# ── Bisect-based template registry ──────────────────────────────
class TemplateRegistry:
    """Sorted registry mapping query_id → SELECT replacement content.

    Algorithm change: upstream uses a plain dict and iterates over
    all query_ids.  Binary search provides O(log n) lookup by
    query_id, and sorted iteration gives deterministic processing
    order.
    """

    def __init__(self):
        self._keys = []          # sorted query IDs
        self._contents = {}      # query_id -> replacement string

    def register(self, query_id, content):
        """Add a query_id → replacement mapping."""
        if query_id not in self._contents:
            insort(self._keys, query_id)
        self._contents[query_id] = content
        _dbg("registry_add", query_id=query_id, content_len=len(content))

    def register_dbg(self, query_id, content):
        """register() with forced debug output."""
        self.register(query_id, content)
        print(f"  [TemplateRegistry] registered {query_id!r}, "
              f"total={len(self._keys)}")

    def lookup(self, query_id):
        """O(log n) lookup for replacement content."""
        idx = bisect_left(self._keys, query_id)
        if idx < len(self._keys) and self._keys[idx] == query_id:
            return self._contents[query_id]
        return None

    def has(self, query_id):
        """Check if query_id is registered."""
        idx = bisect_left(self._keys, query_id)
        return idx < len(self._keys) and self._keys[idx] == query_id

    def all_ids(self):
        """All registered query IDs in sorted order."""
        return list(self._keys)

    def __len__(self):
        return len(self._keys)

    @classmethod
    def from_dict(cls, d):
        """Build a registry from a plain dict."""
        reg = cls()
        for k, v in d.items():
            reg.register(k, v)
        return reg


# ── IMDB JOB replacement contents ──────────────────────────────
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


# ── SELECT * replacement engine ─────────────────────────────────
def replace_select_star_in_file(query_id, file_path, replace_content,
                                is_pqo_query, welford=None,
                                reporter=None):
    """Replace SELECT * in a JSON file with query-specific columns.

    Algorithm enhancements vs upstream:
    - Welford tracking of per-file replacement counts
    - EMA progress reporting for throughput
    - Encoding fallback (UTF-8/UTF-8-sig/latin-1)
    """
    import time as _time

    t0 = _time.monotonic()
    _dbg("replace_start", file=file_path, is_pqo=is_pqo_query)

    if not os.path.isfile(file_path):
        _dbg("replace_missing", file=file_path)
        return 0

    # Load with encoding fallback
    data = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(file_path, "r", encoding=enc) as f:
                data = json.load(f)
            break
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue

    if data is None:
        _dbg("replace_decode_fail", file=file_path)
        return 0

    replace_count = 0
    modified = False

    if is_pqo_query:
        for key in data:
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
        _dbg("replace_done", file=file_path, count=replace_count)

    elapsed = _time.monotonic() - t0
    if welford is not None:
        welford.update(replace_count)
    if reporter is not None:
        reporter.record(elapsed)

    return replace_count


def replace_select_star_in_file_dbg(query_id, file_path,
                                     replace_content, is_pqo_query,
                                     welford=None, reporter=None):
    """replace_select_star_in_file() with forced debug output."""
    print(f"  [replace_select_star] file={file_path}, "
          f"pqo={is_pqo_query}")
    count = replace_select_star_in_file(
        query_id, file_path, replace_content, is_pqo_query,
        welford, reporter
    )
    print(f"  [replace_select_star] replacements={count}")
    return count


# ── Folder processor ────────────────────────────────────────────
def process_folders(query_ids=None, methods=None, input_folders=None,
                    base_dir=".", registry=None, welford=None,
                    reporter=None, validator=None):
    """Walk folder hierarchy and apply SELECT * replacements.

    Algorithm enhancements vs upstream:
    - TemplateRegistry (bisect-based) for query_id lookup
    - Welford tracking of replacement counts across all files
    - EMA progress reporting
    - Huber validation on replacement-count distribution
    """
    if query_ids is None:
        query_ids = ["29-0"]
    if methods is None:
        methods = ["cardinality_full"]
    if input_folders is None:
        input_folders = [
            "inputs/PQO/query",
            "inputs/testing",
            "inputs/training",
        ]

    if registry is None:
        registry = TemplateRegistry.from_dict(IMDB_REPLACE_CONTENTS)
    if welford is None:
        welford = WelfordReplacementTracker()
    if reporter is None:
        reporter = EMAProgressReporter(alpha=0.3)
    if validator is None:
        validator = HuberReplacementValidator(delta=2.0)

    total_replacements = 0

    for qid in query_ids:
        replace_content = registry.lookup(qid)
        if replace_content is None:
            _dbg("skip_unknown_qid", query_id=qid)
            continue

        for method in methods:
            for folder in input_folders:
                folder_path = os.path.join(
                    base_dir, f"imdb_{qid}_original", method, folder
                )
                if not os.path.exists(folder_path):
                    continue

                for root, _, files in os.walk(folder_path):
                    for fname in files:
                        if not fname.endswith(".json"):
                            continue
                        fpath = os.path.join(root, fname)
                        is_pqo = "PQO/query" in folder
                        count = replace_select_star_in_file(
                            qid, fpath, replace_content, is_pqo,
                            welford=welford, reporter=reporter
                        )
                        total_replacements += count

                        if validator is not None and count > 0:
                            validator.compute_loss(count, 1.0)

        _dbg("process_qid_done", query_id=qid,
             running_total=total_replacements)

    _dbg("process_all_done", total=total_replacements,
         welford=welford.summary(),
         ema_rate=reporter.ema_rate,
         validator_healthy=validator.is_healthy())

    return {
        "total_replacements": total_replacements,
        "welford": welford.summary(),
        "ema_rate": reporter.ema_rate,
        "validator_healthy": validator.is_healthy(),
    }


def process_folders_dbg(query_ids=None, **kwargs):
    """process_folders() with forced debug output."""
    print(f"  [process_folders] query_ids={query_ids}")
    result = process_folders(query_ids=query_ids, **kwargs)
    print(f"  [process_folders] result={result}")
    return result


# ── Self-test ───────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    print("=== par2qo_0_modify_template self-test ===\n")

    # 1. WelfordReplacementTracker
    print("1. WelfordReplacementTracker")
    wt = WelfordReplacementTracker()
    counts = [1, 2, 3, 4, 5]
    for c in counts:
        wt.update(c)
    assert wt.n == 5
    assert abs(wt.mean - 3.0) < 1e-9
    assert wt.total_replacements == 15
    expected_var = np.var(counts, ddof=0)
    assert abs(wt.variance - expected_var) < 1e-6
    print(f"   mean={wt.mean:.2f}, var={wt.variance:.2f}, "
          f"total={wt.total_replacements}  OK")

    # 2. EMAProgressReporter
    print("2. EMAProgressReporter")
    pr = EMAProgressReporter(alpha=0.5)
    pr.record(0.1)  # 10 files/s
    assert abs(pr.ema_rate - 10.0) < 1e-6
    pr.record(0.2)  # 5 files/s → EMA = 0.5*5 + 0.5*10 = 7.5
    assert abs(pr.ema_rate - 7.5) < 1e-6
    eta = pr.eta_seconds(75)
    assert abs(eta - 10.0) < 1e-6
    print(f"   ema_rate={pr.ema_rate:.1f}, "
          f"eta_75={eta:.1f}s  OK")

    # 3. HuberReplacementValidator
    print("3. HuberReplacementValidator")
    hv = HuberReplacementValidator(delta=2.0)
    l1 = hv.compute_loss(1, 1)
    assert l1 == 0.0
    l2 = hv.compute_loss(5, 1)  # r=4, > delta=2 → 2*(4-1)=6
    assert abs(l2 - 6.0) < 1e-9
    l3 = hv.compute_loss(2, 1)  # r=1, ≤ delta → 0.5*1=0.5
    assert abs(l3 - 0.5) < 1e-9
    print(f"   l_zero={l1}, l_large={l2}, l_small={l3:.1f}  OK")

    # 4. TemplateRegistry
    print("4. TemplateRegistry")
    reg = TemplateRegistry()
    reg.register("1-0", "SELECT MIN(mc.note)")
    reg.register("33-0", "SELECT MIN(cn1.name)")
    reg.register("5-0", "SELECT MIN(t.title)")
    assert reg.has("1-0")
    assert reg.lookup("1-0") == "SELECT MIN(mc.note)"
    assert not reg.has("999-0")
    assert reg.all_ids() == ["1-0", "33-0", "5-0"]  # sorted: 1-0, 33-0, 5-0
    print(f"   ids={reg.all_ids()}, len={len(reg)}  OK")

    # 5. TemplateRegistry.from_dict
    print("5. TemplateRegistry.from_dict")
    reg2 = TemplateRegistry.from_dict(IMDB_REPLACE_CONTENTS)
    assert len(reg2) == 33
    assert reg2.has("29-0")
    assert "voiced_animation" in reg2.lookup("29-0")
    print(f"   loaded {len(reg2)} templates from IMDB dict  OK")

    # 6. replace_select_star_in_file (PQO mode)
    print("6. replace_select_star_in_file (PQO)")
    with tempfile.TemporaryDirectory() as tmpdir:
        pqo_data = {
            "q1_test_0": "SELECT * FROM t WHERE t.id = 1",
            "q1_test_1": "SELECT * FROM t WHERE t.id = 2",
        }
        fp = os.path.join(tmpdir, "test.json")
        with open(fp, "w") as f:
            json.dump(pqo_data, f)

        wt2 = WelfordReplacementTracker()
        count = replace_select_star_in_file(
            "q1", fp, "SELECT MIN(t.title)", True, welford=wt2
        )
        assert count == 2
        with open(fp) as f:
            result = json.load(f)
        assert "SELECT MIN(t.title)" in result["q1_test_0"]
        assert "SELECT *" not in result["q1_test_0"]
        print(f"   replaced {count} entries  OK")

    # 7. replace_select_star_in_file (non-PQO mode)
    print("7. replace_select_star_in_file (non-PQO)")
    with tempfile.TemporaryDirectory() as tmpdir:
        non_pqo = {
            "q2": {"query": "SELECT * FROM t JOIN ci ON t.id=ci.mid"},
        }
        fp = os.path.join(tmpdir, "test2.json")
        with open(fp, "w") as f:
            json.dump(non_pqo, f)

        count = replace_select_star_in_file(
            "q2", fp, "SELECT MIN(t.title)", False
        )
        assert count == 1
        with open(fp) as f:
            result = json.load(f)
        assert "SELECT MIN(t.title)" in result["q2"]["query"]
        print(f"   replaced {count} entry  OK")

    # 8. process_folders (no matching directory)
    print("8. process_folders (empty run)")
    with tempfile.TemporaryDirectory() as tmpdir:
        result = process_folders(
            query_ids=["29-0"], base_dir=tmpdir
        )
        assert result["total_replacements"] == 0
        assert result["validator_healthy"]
        print(f"   total={result['total_replacements']}, "
              f"healthy={result['validator_healthy']}  OK")

    # 9. process_folders (full integration)
    print("9. process_folders (integration with temp files)")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directory structure
        folder_path = os.path.join(
            tmpdir, "imdb_29-0_original", "cardinality_full",
            "inputs", "PQO", "query"
        )
        os.makedirs(folder_path)
        test_pqo = {
            "29-0_test_0": "SELECT * FROM t WHERE t.id = 1"
        }
        fp = os.path.join(folder_path, "29-0_test.json")
        with open(fp, "w") as f:
            json.dump(test_pqo, f)

        result = process_folders(
            query_ids=["29-0"],
            methods=["cardinality_full"],
            input_folders=["inputs/PQO/query"],
            base_dir=tmpdir,
        )
        assert result["total_replacements"] == 1
        with open(fp) as f:
            data = json.load(f)
        assert "voiced_animation" in data["29-0_test_0"]
        print(f"   total={result['total_replacements']}, "
              f"content verified  OK")

    print("\nAll 9 tests passed.")
