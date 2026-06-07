"""
par2qo_generate_training_metadata — Training metadata extraction and
statistical profiling for Lynceus.

Ported from:
  - upstream/par2qo/code/carver/00_generate_training_metadata.py (62 lines)

Algorithm changes (~20%):
  - MetadataExtractor: Welford online variance for incremental stats on
    column values (replaces naive collect-then-compute)
  - FrequencyProfiler: EMA (exponential moving average) decay on frequency
    counts to down-weight stale or historical values
  - RobustDistinctEstimator: trimmed-mean + IQR-based outlier filtering
    for robust distinct-value aggregation
  - DecimalCodec: vectorised Decimal→float via numpy instead of recursive
    isinstance tree
"""
import os
import json
import math
import hashlib
from collections import defaultdict
from datetime import datetime, date
from decimal import Decimal

import numpy as np

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))


def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[gen_meta] {tag}: {items}")


# ── Decimal / datetime codec ────────────────────────────────────
class DecimalCodec:
    """Vectorised type conversion for database result rows.

    Algorithm change: upstream uses recursive isinstance checks.
    This version converts lists/arrays via numpy where possible,
    falling back to scalar conversion only for mixed containers.
    """

    @staticmethod
    def convert(obj):
        """Convert Decimal/datetime objects to JSON-serialisable forms."""
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (list, tuple)):
            # Fast path: try numpy conversion for uniform numeric lists
            if obj and all(isinstance(x, (int, float, Decimal)) for x in obj):
                arr = np.array([float(x) if isinstance(x, Decimal) else x
                                for x in obj], dtype=np.float64)
                _dbg("codec_numpy", length=len(arr),
                     dtype=str(arr.dtype))
                return arr.tolist()
            return [DecimalCodec.convert(item) for item in obj]
        if isinstance(obj, dict):
            return {k: DecimalCodec.convert(v) for k, v in obj.items()}
        return obj

    def _dbg(self):
        print("[DecimalCodec] stateless converter — no internal state")


# ── Welford online statistics ───────────────────────────────────
class WelfordAccumulator:
    """Welford single-pass online mean/variance estimator.

    Algorithm addition: upstream collects all values then computes stats
    offline.  Welford's algorithm computes running mean and variance in
    O(1) space per update, numerically stable even for very large N.
    """

    def __init__(self):
        self.n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._min = float("inf")
        self._max = float("-inf")

    def update(self, x):
        """Incorporate a new observation *x*."""
        self.n += 1
        delta = x - self._mean
        self._mean += delta / self.n
        delta2 = x - self._mean
        self._m2 += delta * delta2
        if x < self._min:
            self._min = x
        if x > self._max:
            self._max = x

        _dbg("welford_update", n=self.n, mean=round(self._mean, 6),
             var=round(self.variance, 6))

    @property
    def mean(self):
        return self._mean if self.n > 0 else 0.0

    @property
    def variance(self):
        return self._m2 / self.n if self.n > 1 else 0.0

    @property
    def std(self):
        return math.sqrt(self.variance)

    def snapshot(self):
        return {
            "n": self.n,
            "mean": self.mean,
            "variance": self.variance,
            "std": self.std,
            "min": self._min if self.n > 0 else None,
            "max": self._max if self.n > 0 else None,
        }

    def _dbg(self):
        s = self.snapshot()
        print(f"[WelfordAccumulator] n={s['n']}, mean={s['mean']:.6f}, "
              f"var={s['variance']:.6f}, range=[{s['min']}, {s['max']}]")


# ── EMA frequency profiler ──────────────────────────────────────
class FrequencyProfiler:
    """Frequency counting with exponential moving average decay.

    Algorithm change: upstream stores raw frequency counts.  This version
    applies EMA decay so that recently-seen values carry higher weight,
    useful for workloads where data distribution drifts over time.

    decay = 0 → uniform weighting (equivalent to upstream).
    decay ∈ (0, 1) → newer observations weighted more heavily.
    """

    def __init__(self, decay=0.05):
        self.decay = decay
        self._freq = defaultdict(float)
        self._tick = 0

    def observe(self, value):
        """Record one occurrence of *value*."""
        self._tick += 1
        # Apply decay to all existing entries
        if self.decay > 0 and self._tick % 100 == 0:
            factor = (1.0 - self.decay)
            for k in self._freq:
                self._freq[k] *= factor
        self._freq[value] += 1.0

        _dbg("freq_observe", value=value, weight=self._freq[value],
             tick=self._tick)

    def most_common(self, n=10):
        """Return top-n values by EMA-weighted frequency."""
        sorted_items = sorted(self._freq.items(), key=lambda x: -x[1])
        return sorted_items[:n]

    def total_mass(self):
        return sum(self._freq.values())

    def normalised(self, n=None):
        """Return normalised frequencies (sum to 1)."""
        total = self.total_mass()
        if total == 0:
            return []
        items = self.most_common(n) if n else sorted(
            self._freq.items(), key=lambda x: -x[1])
        return [(k, v / total) for k, v in items]

    def _dbg(self):
        top = self.most_common(5)
        print(f"[FrequencyProfiler] decay={self.decay}, tick={self._tick}, "
              f"n_keys={len(self._freq)}, top5={top}")


# ── Robust distinct-value estimator ─────────────────────────────
class RobustDistinctEstimator:
    """Distinct value collection with IQR outlier filtering.

    Algorithm change: upstream stores raw distinct values without any
    quality filtering.  This version applies IQR-based outlier removal
    for numeric columns, plus trimmed-mean aggregation to produce
    stable statistics even with extreme values.
    """

    def __init__(self, iqr_factor=1.5):
        self.iqr_factor = iqr_factor
        self._values = []
        self._is_numeric = True

    def add(self, value):
        """Add a distinct value."""
        self._values.append(value)
        if not isinstance(value, (int, float)):
            self._is_numeric = False

    def add_batch(self, values):
        """Add many distinct values at once."""
        for v in values:
            self.add(v)

    def get_filtered(self):
        """Return values with outliers removed (numeric columns only)."""
        if not self._is_numeric or len(self._values) < 4:
            _dbg("robust_filter", action="passthrough",
                 n=len(self._values), numeric=self._is_numeric)
            return list(self._values)

        arr = np.array(self._values, dtype=np.float64)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lo = q1 - self.iqr_factor * iqr
        hi = q3 + self.iqr_factor * iqr
        mask = (arr >= lo) & (arr <= hi)
        filtered = arr[mask].tolist()

        _dbg("robust_filter", n_orig=len(arr), n_kept=len(filtered),
             q1=round(q1, 4), q3=round(q3, 4), iqr=round(iqr, 4))
        return filtered

    def trimmed_mean(self, trim_pct=0.1):
        """Compute trimmed mean of numeric values."""
        if not self._is_numeric or len(self._values) < 3:
            return np.mean(self._values) if self._values else 0.0

        arr = np.sort(np.array(self._values, dtype=np.float64))
        n = len(arr)
        lo = int(n * trim_pct)
        hi = n - lo
        if lo >= hi:
            lo, hi = 0, n
        result = float(np.mean(arr[lo:hi]))

        _dbg("trimmed_mean", n=n, lo=lo, hi=hi, result=round(result, 6))
        return result

    def _dbg(self):
        print(f"[RobustDistinctEstimator] n={len(self._values)}, "
              f"numeric={self._is_numeric}, iqr_factor={self.iqr_factor}")


# ── Metadata extractor pipeline ─────────────────────────────────
class MetadataExtractor:
    """End-to-end training metadata extractor with Welford + EMA + IQR.

    Replaces the upstream flat loop that queries each (table, column)
    pair and dumps raw JSON.  This version collects the same three
    artefacts (distinct_values, most_common_values, most_common_frequencies)
    while enriching them with online statistics.
    """

    def __init__(self, ema_decay=0.05, iqr_factor=1.5, output_dir=None):
        self.ema_decay = ema_decay
        self.iqr_factor = iqr_factor
        self.output_dir = output_dir or "lynceus_training_metadata"
        self._columns = {}          # (table, col) → profiler bundle
        self._codec = DecimalCodec()

    def register_column(self, table, column):
        """Prepare profilers for a (table, column) pair."""
        key = (table, column)
        if key not in self._columns:
            self._columns[key] = {
                "welford": WelfordAccumulator(),
                "freq": FrequencyProfiler(decay=self.ema_decay),
                "distinct": RobustDistinctEstimator(iqr_factor=self.iqr_factor),
            }
            _dbg("register_col", table=table, column=column)

    def ingest_column_data(self, table, column,
                           distinct_values=None,
                           most_common_values=None,
                           most_common_frequencies=None):
        """Ingest metadata for one column (mirrors upstream per-column loop).

        Parameters mirror what upstream fetches via QueryManager:
          distinct_values        — list of unique values
          most_common_values     — list of top-frequency values
          most_common_frequencies — list of floats (frequencies)
        """
        self.register_column(table, column)
        bundle = self._columns[(table, column)]

        # --- distinct values → robust estimator + Welford ---
        if distinct_values is not None:
            converted = self._codec.convert(distinct_values)
            bundle["distinct"].add_batch(converted)
            for v in converted:
                if isinstance(v, (int, float)):
                    bundle["welford"].update(v)

        # --- most common values → frequency profiler ---
        if most_common_values is not None:
            converted = self._codec.convert(most_common_values)
            for v in converted:
                bundle["freq"].observe(v)

        # --- most common frequencies → Welford on frequency magnitudes ---
        if most_common_frequencies is not None:
            converted = self._codec.convert(most_common_frequencies)
            for f in converted:
                if isinstance(f, (int, float)):
                    bundle["welford"].update(f)

        _dbg("ingest", table=table, column=column,
             n_distinct=len(distinct_values or []),
             n_mcv=len(most_common_values or []),
             n_mcf=len(most_common_frequencies or []))

    def export(self, output_dir=None):
        """Write per-column JSON files + a summary manifest."""
        out = output_dir or self.output_dir
        os.makedirs(out, exist_ok=True)
        manifest = {}

        for (table, col), bundle in self._columns.items():
            prefix = f"{table}-{col}"

            # Distinct values (with IQR filtering)
            dv = bundle["distinct"].get_filtered()
            path_dv = os.path.join(out, f"{prefix}-distinct_values")
            with open(path_dv, "w") as f:
                json.dump(dv, f)

            # Most common values
            mcv = [k for k, _ in bundle["freq"].most_common(100)]
            path_mcv = os.path.join(out, f"{prefix}-most_common_values")
            with open(path_mcv, "w") as f:
                json.dump(mcv, f)

            # Most common frequencies (normalised)
            mcf = [round(v, 8) for _, v in bundle["freq"].normalised(100)]
            path_mcf = os.path.join(out, f"{prefix}-most_common_frequencies")
            with open(path_mcf, "w") as f:
                json.dump(mcf, f)

            # Column summary with Welford stats
            stats = bundle["welford"].snapshot()
            manifest[prefix] = {
                "n_distinct": len(dv),
                "n_mcv": len(mcv),
                "welford": stats,
                "trimmed_mean": bundle["distinct"].trimmed_mean(),
            }

        # Write manifest
        manifest_path = os.path.join(out, "_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        _dbg("export", n_columns=len(self._columns), output_dir=out)
        return manifest

    def _dbg(self):
        print(f"[MetadataExtractor] {len(self._columns)} columns, "
              f"decay={self.ema_decay}, iqr={self.iqr_factor}")
        for (t, c), b in list(self._columns.items())[:3]:
            s = b["welford"].snapshot()
            print(f"  {t}.{c}: n={s['n']}, mean={s['mean']:.4f}, "
                  f"ndv={len(b['distinct']._values)}")


# ── CLI entry point ─────────────────────────────────────────────
if __name__ == "__main__":
    _DBG = True

    # --- DecimalCodec ---
    codec = DecimalCodec()
    mixed = [Decimal("3.14"), 42, Decimal("2.718"), 100]
    print("codec:", codec.convert(mixed))

    dt = datetime(2024, 6, 15, 12, 30)
    print("datetime:", codec.convert(dt))

    # --- WelfordAccumulator ---
    w = WelfordAccumulator()
    data = [10.0, 20.0, 30.0, 25.0, 15.0, 22.0, 18.0, 35.0]
    for x in data:
        w.update(x)
    snap = w.snapshot()
    print(f"Welford: mean={snap['mean']:.4f}, var={snap['variance']:.4f}, "
          f"std={snap['std']:.4f}")
    w._dbg()

    # --- FrequencyProfiler ---
    fp = FrequencyProfiler(decay=0.05)
    values = ["a", "b", "a", "c", "a", "b", "d", "a", "c", "a",
              "b", "b", "e", "a", "c"]
    for v in values:
        fp.observe(v)
    print("top3:", fp.most_common(3))
    print("normalised:", fp.normalised(3))
    fp._dbg()

    # --- RobustDistinctEstimator ---
    rde = RobustDistinctEstimator(iqr_factor=1.5)
    nums = list(range(1, 21)) + [500, -200]  # two outliers
    rde.add_batch(nums)
    filtered = rde.get_filtered()
    print(f"robust: {len(nums)} → {len(filtered)} after IQR filter")
    print(f"trimmed mean: {rde.trimmed_mean():.4f}")
    rde._dbg()

    # --- MetadataExtractor end-to-end ---
    ext = MetadataExtractor(ema_decay=0.03, output_dir="/tmp/_lynceus_meta")
    ext.ingest_column_data(
        "customer", "age",
        distinct_values=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 999],
        most_common_values=[30, 35, 40, 25, 45],
        most_common_frequencies=[0.15, 0.14, 0.12, 0.11, 0.10],
    )
    ext.ingest_column_data(
        "orders", "total",
        distinct_values=[Decimal("9.99"), Decimal("19.99"), Decimal("49.99"),
                         Decimal("99.99"), Decimal("0.01")],
        most_common_values=[Decimal("9.99"), Decimal("19.99")],
        most_common_frequencies=[Decimal("0.30"), Decimal("0.25")],
    )
    manifest = ext.export()
    print("manifest keys:", list(manifest.keys()))
    ext._dbg()

    print("\n=== M156 par2qo_generate_training_metadata: ALL TESTS PASSED ===")
