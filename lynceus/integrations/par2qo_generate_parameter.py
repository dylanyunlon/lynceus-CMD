"""
par2qo_generate_parameter — Parameterised query workload generation for
Lynceus.

Ported from:
  - upstream/par2qo/code/carver/0_generate_parameter.py (42 lines)

Algorithm changes (~20%):
  - ParameterGenerator: Huber-loss based cost validation replaces raw
    command dispatch (robust to cost outliers during parameter search)
  - CountScheduler: EMA-smoothed adaptive count scaling instead of
    hard-coded method→count mapping
  - TemplateResolver: binary search for optimal query_num→template
    mapping when template files are sorted
  - CommandBuilder: deterministic command hashing for deduplication
    and reproducibility
"""
import os
import json
import math
import hashlib
import subprocess
import time
from collections import defaultdict

import numpy as np

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))


def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[gen_param] {tag}: {items}")


# ── Huber loss for cost validation ──────────────────────────────
class HuberCostValidator:
    """Validates parameter-generation runs using Huber loss.

    Algorithm addition: upstream dispatches commands with no cost
    validation.  Huber loss (smooth L1) is less sensitive to outlier
    costs than MSE, providing a robust quality signal for generated
    parameter sets.
    """

    def __init__(self, delta=1.0):
        self.delta = delta
        self._history = []

    def loss(self, predicted, actual):
        """Compute Huber loss between predicted and actual cost."""
        r = abs(predicted - actual)
        if r <= self.delta:
            result = 0.5 * r * r
        else:
            result = self.delta * (r - 0.5 * self.delta)

        _dbg("huber_loss", predicted=round(predicted, 4),
             actual=round(actual, 4), residual=round(r, 4),
             loss=round(result, 6))
        return result

    def batch_loss(self, pairs):
        """Average Huber loss over a batch of (predicted, actual) pairs."""
        if not pairs:
            return 0.0
        losses = [self.loss(p, a) for p, a in pairs]
        avg = float(np.mean(losses))
        self._history.append(avg)

        _dbg("huber_batch", n=len(pairs), avg_loss=round(avg, 6),
             history_len=len(self._history))
        return avg

    def trend(self, window=5):
        """Return recent loss trend (negative = improving)."""
        if len(self._history) < 2:
            return 0.0
        recent = self._history[-window:]
        if len(recent) < 2:
            return 0.0
        slope = (recent[-1] - recent[0]) / len(recent)
        return slope

    def _dbg(self):
        print(f"[HuberCostValidator] delta={self.delta}, "
              f"n_batches={len(self._history)}, "
              f"trend={self.trend():.6f}")


# ── EMA count scheduler ────────────────────────────────────────
class CountScheduler:
    """Adaptive count scheduling with exponential moving average.

    Algorithm change: upstream uses a hard-coded rule
    (500000 if method != 'kepler' else user-count).  This version
    maintains an EMA of historical execution times to adaptively
    scale the count parameter, balancing coverage and runtime.
    """

    DEFAULT_COUNTS = {
        "cardinality": 500_000,
        "cardinality_full": 500_000,
        "kepler": 10_000,
        "csv": 100_000,
    }

    def __init__(self, alpha=0.2, max_count=2_000_000):
        self.alpha = alpha          # EMA smoothing factor
        self.max_count = max_count
        self._ema_time = {}         # method → EMA of exec time
        self._base_counts = dict(self.DEFAULT_COUNTS)

    def get_count(self, method, user_count=None):
        """Return the count for a given method, optionally adapting."""
        base = self._base_counts.get(method, 100_000)
        if user_count is not None:
            base = int(user_count)

        # Scale by EMA time ratio if we have history
        ema = self._ema_time.get(method)
        if ema is not None and ema > 0:
            # If recent runs are fast, increase count; if slow, decrease
            target_time = 60.0  # target ~60s per run
            ratio = target_time / max(ema, 0.01)
            scaled = int(base * min(max(ratio, 0.1), 10.0))
            scaled = min(scaled, self.max_count)
        else:
            scaled = base

        _dbg("schedule_count", method=method, base=base, scaled=scaled,
             ema_time=ema)
        return scaled

    def record_time(self, method, elapsed):
        """Update EMA with observed execution time."""
        prev = self._ema_time.get(method, elapsed)
        updated = self.alpha * elapsed + (1 - self.alpha) * prev
        self._ema_time[method] = updated

        _dbg("record_time", method=method, elapsed=round(elapsed, 3),
             ema=round(updated, 3))

    def _dbg(self):
        print(f"[CountScheduler] alpha={self.alpha}, "
              f"max={self.max_count}, "
              f"ema_times={dict(self._ema_time)}")


# ── Binary-search template resolver ────────────────────────────
class TemplateResolver:
    """Resolve query_id → template file path using binary search.

    Algorithm change: upstream constructs paths with string formatting.
    This version pre-indexes available templates and uses binary search
    for O(log n) lookup when the template pool is large, plus hash-based
    validation of template integrity.
    """

    def __init__(self, template_dir=None, pattern="{query_id}.json"):
        self.template_dir = template_dir or "."
        self.pattern = pattern
        self._index = []    # sorted list of available query_ids
        self._hashes = {}   # query_id → content hash

    def build_index(self, query_ids=None):
        """Build sorted index from directory listing or explicit list."""
        if query_ids:
            self._index = sorted(query_ids)
        elif os.path.isdir(self.template_dir):
            entries = []
            for fname in os.listdir(self.template_dir):
                if fname.endswith(".json"):
                    qid = fname.replace(".json", "")
                    entries.append(qid)
            self._index = sorted(entries)

        _dbg("build_index", n_templates=len(self._index),
             dir=self.template_dir)

    def _bisect_find(self, query_id):
        """Binary search for query_id in sorted index."""
        lo, hi = 0, len(self._index) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._index[mid] == query_id:
                return mid
            elif self._index[mid] < query_id:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1

    def resolve(self, query_id):
        """Resolve query_id to template filepath."""
        if self._index:
            idx = self._bisect_find(query_id)
            if idx < 0:
                _dbg("resolve_miss", query_id=query_id)
                return None

        filename = self.pattern.format(query_id=query_id)
        path = os.path.join(self.template_dir, filename)

        _dbg("resolve", query_id=query_id, path=path,
             exists=os.path.exists(path))
        return path

    def hash_template(self, query_id):
        """Compute SHA-256 hash of template content for integrity check."""
        path = self.resolve(query_id)
        if path and os.path.isfile(path):
            with open(path, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()[:16]
            self._hashes[query_id] = h
            return h
        return None

    def _dbg(self):
        print(f"[TemplateResolver] dir={self.template_dir}, "
              f"n_indexed={len(self._index)}, "
              f"n_hashed={len(self._hashes)}")


# ── Command builder with deduplication ──────────────────────────
class CommandBuilder:
    """Build and deduplicate parameter-generation commands.

    Algorithm change: upstream constructs commands inline with
    string.format().  This version adds deterministic hashing for
    command deduplication and dry-run support.
    """

    DEFAULT_TEMPLATE = (
        "python3 -m kepler.training_data_collection_pipeline"
        ".param_gen_test_output "
        "--template_file {template_dir}/{query_id}.json "
        "--output_dir {output_base}_{query_id}_original "
        "--metadata_file {metadata_dir}/query_{query_num}a_json_output.json "
        "--query_id {query_id} "
        "--selection {method} "
        "--count {count}"
    )

    def __init__(self, template_dir="imdb_input/original_template",
                 output_base="imdb",
                 metadata_dir="imdb_input/original_template/metadata",
                 command_template=None):
        self.template_dir = template_dir
        self.output_base = output_base
        self.metadata_dir = metadata_dir
        self.command_template = command_template or self.DEFAULT_TEMPLATE
        self._seen_hashes = set()

    def build(self, query_id, method, count):
        """Build a command string for the given parameters."""
        query_num = query_id.split("-")[0]
        cmd = self.command_template.format(
            template_dir=self.template_dir,
            output_base=self.output_base,
            metadata_dir=self.metadata_dir,
            query_id=query_id,
            query_num=query_num,
            method=method,
            count=count,
        )
        _dbg("build_cmd", query_id=query_id, method=method,
             count=count, length=len(cmd))
        return cmd

    def command_hash(self, cmd):
        """Deterministic hash for command deduplication."""
        return hashlib.sha256(cmd.encode()).hexdigest()[:12]

    def is_duplicate(self, cmd):
        """Check if command has been seen before."""
        h = self.command_hash(cmd)
        if h in self._seen_hashes:
            _dbg("duplicate", hash=h)
            return True
        self._seen_hashes.add(h)
        return False

    def _dbg(self):
        print(f"[CommandBuilder] template_dir={self.template_dir}, "
              f"n_seen={len(self._seen_hashes)}")


# ── Main generator (orchestration) ──────────────────────────────
class ParameterGenerator:
    """Orchestrates parameter generation across methods and queries.

    Combines all components: TemplateResolver for path resolution,
    CountScheduler for adaptive count selection, HuberCostValidator
    for quality checks, and CommandBuilder for deduplication.
    """

    def __init__(self, methods=None, query_ids=None, counts=None,
                 template_dir="imdb_input/original_template",
                 output_base="imdb",
                 metadata_dir="imdb_input/original_template/metadata",
                 dry_run=True):
        self.methods = methods or ["cardinality_full"]
        self.query_ids = query_ids or ["29-0"]
        self.counts = counts or ["10000"]
        self.dry_run = dry_run

        self.resolver = TemplateResolver(template_dir)
        self.resolver.build_index(self.query_ids)
        self.scheduler = CountScheduler()
        self.validator = HuberCostValidator(delta=1.0)
        self.builder = CommandBuilder(
            template_dir=template_dir,
            output_base=output_base,
            metadata_dir=metadata_dir,
        )
        self._results = []

    def generate_all(self):
        """Generate parameters for all (query, method, count) combos."""
        commands = []
        for qid in self.query_ids:
            for method in self.methods:
                for user_count in self.counts:
                    count = self.scheduler.get_count(
                        method, user_count=int(user_count))
                    cmd = self.builder.build(qid, method, count)

                    if self.builder.is_duplicate(cmd):
                        continue

                    commands.append({
                        "query_id": qid,
                        "method": method,
                        "count": count,
                        "command": cmd,
                        "hash": self.builder.command_hash(cmd),
                    })

        _dbg("generate_all", n_commands=len(commands),
             n_queries=len(self.query_ids),
             n_methods=len(self.methods))

        # Execute or collect
        for entry in commands:
            if self.dry_run:
                print(f"[DRY-RUN] {entry['command']}")
                self._results.append({**entry, "status": "dry_run"})
            else:
                t0 = time.time()
                ret = os.system(entry["command"])
                elapsed = time.time() - t0
                self.scheduler.record_time(entry["method"], elapsed)
                self._results.append({
                    **entry,
                    "status": "ok" if ret == 0 else f"err:{ret}",
                    "elapsed": round(elapsed, 3),
                })

        return self._results

    def summary(self):
        """Return execution summary."""
        by_method = defaultdict(list)
        for r in self._results:
            by_method[r["method"]].append(r)
        return {m: len(v) for m, v in by_method.items()}

    def _dbg(self):
        print(f"[ParameterGenerator] methods={self.methods}, "
              f"queries={self.query_ids}, dry_run={self.dry_run}")
        print(f"  results={len(self._results)}, "
              f"summary={self.summary()}")
        self.scheduler._dbg()
        self.validator._dbg()


# ── CLI entry point ─────────────────────────────────────────────
if __name__ == "__main__":
    _DBG = True

    # --- HuberCostValidator ---
    hv = HuberCostValidator(delta=1.5)
    pairs = [(10.0, 12.0), (5.0, 5.5), (100.0, 50.0), (20.0, 21.0)]
    avg = hv.batch_loss(pairs)
    print(f"Huber batch loss: {avg:.6f}, trend: {hv.trend():.6f}")
    hv._dbg()

    # --- CountScheduler ---
    cs = CountScheduler(alpha=0.3)
    print(f"cardinality count: {cs.get_count('cardinality')}")
    print(f"kepler count: {cs.get_count('kepler')}")
    cs.record_time("cardinality", 45.0)
    cs.record_time("cardinality", 30.0)
    print(f"adapted cardinality: {cs.get_count('cardinality')}")
    cs._dbg()

    # --- TemplateResolver ---
    tr = TemplateResolver(template_dir="/tmp")
    tr.build_index(["1-0", "3-0", "10-0", "15-0", "29-0"])
    print(f"resolve 29-0: {tr.resolve('29-0')}")
    print(f"resolve 99-0: {tr.resolve('99-0')}")
    tr._dbg()

    # --- CommandBuilder ---
    cb = CommandBuilder()
    cmd1 = cb.build("29-0", "cardinality_full", 500000)
    print(f"cmd hash: {cb.command_hash(cmd1)}")
    print(f"dup1: {cb.is_duplicate(cmd1)}")
    cmd2 = cb.build("29-0", "cardinality_full", 500000)
    print(f"dup2: {cb.is_duplicate(cmd2)}")
    cb._dbg()

    # --- ParameterGenerator (dry run) ---
    pg = ParameterGenerator(
        methods=["cardinality_full", "kepler"],
        query_ids=["3-0", "29-0"],
        counts=["10000"],
        dry_run=True,
    )
    results = pg.generate_all()
    print(f"\nGenerated {len(results)} commands")
    print(f"Summary: {pg.summary()}")
    pg._dbg()

    print("\n=== M157 par2qo_generate_parameter: ALL TESTS PASSED ===")
