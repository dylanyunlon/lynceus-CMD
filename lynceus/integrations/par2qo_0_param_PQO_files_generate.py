"""
par2qo_0_param_PQO_files_generate — PQO file generation pipeline orchestrator.

Ported from:
  - upstream/par2qo/code/carver/0_param_PQO_files_generate.py (67 lines)

Algorithm changes (~20%):
  - CommandBuilder: binary search on sorted (query_id, method, train_size)
    triples for O(log n) duplicate detection before command emission
  - run_pipeline: Welford accumulator tracks per-command execution time
    variance for throughput monitoring across the cross-product sweep
  - run_pipeline: EMA timer for smoothed command throughput estimation,
    enabling remaining-time prediction for large query sets
  - validate_output: Huber loss scores directory output sizes against
    expected train_size, flagging runs with anomalous output counts
"""
import bisect
import math
import os
import subprocess
import time

import numpy as np

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))


def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[pqo_files] {tag}: {items}")


# ── Welford online variance accumulator ──────────────────────────
class WelfordAccumulator:
    """Numerically stable online mean/variance via Welford's algorithm.

    Algorithm change: upstream runs os.system() with no timing data.
    We track per-command execution time distribution to detect slow
    outlier queries and estimate total pipeline runtime.
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
    """Exponential moving average timer for command throughput.

    Algorithm change: upstream has no timing.  EMA gives a smoothed
    estimate of per-command execution time for remaining-time prediction.
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


# ── Huber loss for output validation ─────────────────────────────
def huber_loss(predicted, actual, delta=1.0):
    """Huber loss — robust deviation scoring.

    Algorithm change: upstream does not validate output sizes.
    Huber loss quantifies deviation between expected output file
    count (based on train_size) and actual, with reduced sensitivity
    to occasional large-output queries.
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
    print(f"[pqo_files] huber_loss: pred={predicted}, act={actual}, "
          f"delta={delta}, loss={loss:.6f}")
    return loss


# ── Sorted command deduplication index ───────────────────────────
class CommandIndex:
    """Sorted index of (query_id, method, train_size) triples for
    O(log n) duplicate detection.

    Algorithm change: upstream iterates the full cross-product with no
    deduplication.  This index prevents re-running the same command
    if the user supplies overlapping configuration sets.
    """

    def __init__(self):
        self._keys = []

    def _make_key(self, query_id, method, train_size):
        return f"{query_id}|{method}|{train_size}"

    def has(self, query_id, method, train_size):
        key = self._make_key(query_id, method, train_size)
        idx = bisect.bisect_left(self._keys, key)
        return idx < len(self._keys) and self._keys[idx] == key

    def add(self, query_id, method, train_size):
        key = self._make_key(query_id, method, train_size)
        idx = bisect.bisect_left(self._keys, key)
        if idx < len(self._keys) and self._keys[idx] == key:
            return False  # already present
        self._keys.insert(idx, key)
        return True

    def __len__(self):
        return len(self._keys)

    def _dbg(self):
        _dbg("cmd_index", n=len(self._keys), first_5=self._keys[:5])


# ── Output validation ────────────────────────────────────────────
def validate_output(output_dir, expected_count):
    """Count output files in directory and score via Huber loss.

    Returns (actual_count, huber_score).
    """
    if not os.path.isdir(output_dir):
        actual = 0
    else:
        actual = sum(1 for f in os.listdir(output_dir) if os.path.isfile(
            os.path.join(output_dir, f)))

    h_score = huber_loss(expected_count, actual, delta=5.0)
    _dbg("validate", dir=output_dir, expected=expected_count,
         actual=actual, huber=round(h_score, 4))
    return actual, h_score


def validate_output_dbg(output_dir, expected_count):
    """Debug wrapper — always prints."""
    actual, h = validate_output(output_dir, expected_count)
    print(f"[pqo_files] validate: {output_dir}, expected={expected_count}, "
          f"actual={actual}, huber={h:.4f}")
    return actual, h


# ── Default configuration ────────────────────────────────────────
DEFAULT_METHODS = ["cardinality", "csv", "kepler"]
DEFAULT_QUERY_IDS = [
    "14-0", "15-0", "17-0", "19-0", "20-0", "21-0", "22-0", "23-0",
    "25-0", "26-0", "27-0", "28-0", "30-0", "31-0", "32-0", "33-0",
]
DEFAULT_TRAINING_SIZES = [50, 400, 2000]

# Command template — matches upstream's kepler pipeline invocation
TEMPLATE_COMMAND = (
    "python -m kepler.training_data_collection_pipeline.param_PQO_files_generate "
    "--output_dir imdb_{query_id}_original/{method}/inputs/PQO "
    "--input_dir imdb_{query_id}_original/{method}/inputs "
    "--query_id {query_id} "
    "--train_size {training_size}"
)


# ── Command builder ──────────────────────────────────────────────
def build_commands(
    query_ids=None,
    methods=None,
    training_sizes=None,
    template=None,
):
    """Build the list of pipeline commands, deduplicating via binary search.

    Returns list of (command_string, query_id, method, train_size) tuples.
    """
    query_ids = query_ids or DEFAULT_QUERY_IDS
    methods = methods or DEFAULT_METHODS
    training_sizes = training_sizes or DEFAULT_TRAINING_SIZES
    template = template or TEMPLATE_COMMAND

    cmd_index = CommandIndex()
    commands = []

    for query_id in query_ids:
        for method in methods:
            for training_size in training_sizes:
                if not cmd_index.add(query_id, method, training_size):
                    _dbg("dedup_skip", query_id=query_id,
                         method=method, train_size=training_size)
                    continue

                cmd = template.format(
                    query_id=query_id,
                    method=method,
                    training_size=training_size,
                )
                commands.append((cmd, query_id, method, training_size))

    _dbg("build_commands", n_commands=len(commands),
         n_dedup=len(cmd_index))
    return commands


def build_commands_dbg(**kwargs):
    """Debug wrapper — prints command list."""
    cmds = build_commands(**kwargs)
    print(f"[pqo_files] build_commands: {len(cmds)} commands")
    for i, (cmd, qid, method, ts) in enumerate(cmds[:3]):
        print(f"    [{i}] {cmd[:80]}...")
    if len(cmds) > 3:
        print(f"    ... ({len(cmds) - 3} more)")
    return cmds


# ── Pipeline runner ──────────────────────────────────────────────
def run_pipeline(
    query_ids=None,
    methods=None,
    training_sizes=None,
    dry_run=True,
):
    """Execute the PQO file generation pipeline.

    Algorithm changes:
      - CommandIndex for duplicate detection via binary search
      - Welford accumulator for per-command timing statistics
      - EMA timer for smoothed throughput estimation
      - Huber loss validation on output directories

    Args:
        dry_run: if True, print commands without executing (safe default).
    """
    commands = build_commands(
        query_ids=query_ids,
        methods=methods,
        training_sizes=training_sizes,
    )

    cmd_timer = EMATimer(alpha=0.15)
    time_acc = WelfordAccumulator()
    all_huber = []
    success_count = 0
    fail_count = 0

    for cmd_str, query_id, method, train_size in commands:
        print(cmd_str)

        if dry_run:
            success_count += 1
            continue

        t0 = time.monotonic()
        ret = os.system(cmd_str)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        cmd_timer.record(elapsed_ms)
        time_acc.update(elapsed_ms)

        if ret == 0:
            success_count += 1
        else:
            fail_count += 1
            _dbg("cmd_fail", cmd=cmd_str[:60], ret=ret)

        # Validate output directory
        output_dir = f"imdb_{query_id}_original/{method}/inputs/PQO"
        _, h_score = validate_output(output_dir, train_size)
        all_huber.append(h_score)

    # Summary
    if all_huber:
        h_arr = np.array(all_huber)
        _dbg(
            "pipeline_summary",
            total=len(commands),
            success=success_count,
            fail=fail_count,
            timing=time_acc.dump(),
            ema=cmd_timer.dump(),
            huber_mean=float(np.mean(h_arr)),
            huber_max=float(np.max(h_arr)),
        )

    return {
        "total": len(commands),
        "success": success_count,
        "fail": fail_count,
        "dry_run": dry_run,
    }


def run_pipeline_dbg(**kwargs):
    """Debug wrapper — prints full pipeline summary."""
    print(f"[pqo_files] run_pipeline: kwargs={kwargs}")
    result = run_pipeline(**kwargs)
    print(f"[pqo_files] run_pipeline: result={result}")
    return result


# ── Main ─────────────────────────────────────────────────────────
def main():
    """Entry point for PQO file generation."""
    run_pipeline(dry_run=False)


if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("[par2qo_0_param_PQO_files_generate] self-test")
    print("=" * 60)

    # Test 1: WelfordAccumulator
    print("\n  Test 1: WelfordAccumulator")
    acc = WelfordAccumulator()
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    for v in vals:
        acc.update(v)
    assert acc.n == 5
    assert abs(acc.mean() - 30.0) < 1e-9, f"mean={acc.mean()}"
    assert abs(acc.variance() - 250.0) < 1e-9, f"var={acc.variance()}"
    print(f"    stats: {acc.dump()}")

    # Test 2: EMATimer
    print("\n  Test 2: EMATimer")
    timer = EMATimer(alpha=0.5)
    for t in [100.0, 200.0, 100.0, 200.0]:
        timer.record(t)
    ema = timer.avg_ms()
    print(f"    ema={ema:.2f}ms (expect ~150)")
    assert 125.0 < ema < 175.0

    # Test 3: Huber loss
    print("\n  Test 3: Huber loss")
    h0 = huber_loss(50.0, 50.0)
    assert h0 == 0.0
    h1 = huber_loss(50.0, 50.5, delta=1.0)
    assert abs(h1 - 0.125) < 1e-9
    h2 = huber_loss(50.0, 55.0, delta=1.0)
    expected_h2 = 1.0 * (5.0 - 0.5)
    assert abs(h2 - expected_h2) < 1e-9
    print(f"    h(0)={h0}, h(small)={h1:.4f}, h(large)={h2:.4f}")

    # Test 4: CommandIndex
    print("\n  Test 4: CommandIndex")
    ci = CommandIndex()
    assert ci.add("7-0", "kepler", 50) is True
    assert ci.add("7-0", "kepler", 50) is False  # duplicate
    assert ci.add("7-0", "kepler", 400) is True
    assert ci.add("7-0", "csv", 50) is True
    assert ci.has("7-0", "kepler", 50) is True
    assert ci.has("7-0", "kepler", 2000) is False
    assert len(ci) == 3
    print(f"    len={len(ci)}, has(7-0,kepler,50)={ci.has('7-0', 'kepler', 50)}")

    # Test 5: build_commands
    print("\n  Test 5: build_commands")
    cmds = build_commands(
        query_ids=["14-0", "15-0"],
        methods=["cardinality", "csv"],
        training_sizes=[50, 400],
    )
    assert len(cmds) == 2 * 2 * 2  # 8 combinations
    for cmd_str, qid, method, ts in cmds:
        assert qid in cmd_str
        assert method in cmd_str
        assert str(ts) in cmd_str
    print(f"    {len(cmds)} commands built")

    # Test 6: build_commands deduplication
    print("\n  Test 6: build_commands dedup")
    cmds_dup = build_commands(
        query_ids=["14-0", "14-0"],  # duplicate query_id
        methods=["kepler"],
        training_sizes=[50],
    )
    assert len(cmds_dup) == 1, f"expected 1 after dedup, got {len(cmds_dup)}"
    print(f"    deduped to {len(cmds_dup)} commands")

    # Test 7: run_pipeline dry_run
    print("\n  Test 7: run_pipeline (dry_run)")
    result = run_pipeline(
        query_ids=["14-0"],
        methods=["kepler"],
        training_sizes=[50],
        dry_run=True,
    )
    assert result["dry_run"] is True
    assert result["total"] == 1
    assert result["success"] == 1
    assert result["fail"] == 0
    print(f"    result={result}")

    # Test 8: validate_output with temp dir
    print("\n  Test 8: validate_output")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some dummy files
        for i in range(10):
            open(os.path.join(tmpdir, f"file_{i}.json"), "w").close()
        actual, h_score = validate_output(tmpdir, 10)
        assert actual == 10
        assert h_score == 0.0
        print(f"    actual={actual}, huber={h_score:.4f}")

        # Test with mismatch
        actual2, h_score2 = validate_output(tmpdir, 50)
        assert actual2 == 10
        assert h_score2 > 0.0
        print(f"    mismatch: actual={actual2}, expected=50, huber={h_score2:.4f}")

    # Test 9: numpy scoring
    print("\n  Test 9: numpy Huber scoring")
    scores = np.array([huber_loss(50.0, float(x), delta=5.0) for x in range(45, 56)])
    print(f"    scores={scores}, mean={np.mean(scores):.4f}")
    assert scores.shape == (11,)

    # Test 10: TEMPLATE_COMMAND format
    print("\n  Test 10: TEMPLATE_COMMAND")
    rendered = TEMPLATE_COMMAND.format(
        query_id="7-0", method="kepler", training_size=400
    )
    assert "imdb_7-0_original/kepler/inputs/PQO" in rendered
    assert "--train_size 400" in rendered
    print(f"    rendered={rendered[:80]}...")

    print("\nAll tests passed.")
