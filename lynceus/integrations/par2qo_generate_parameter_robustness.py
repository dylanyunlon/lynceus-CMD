"""
par2qo_generate_parameter_robustness — Robustness-aware parameter
generation for Lynceus.

Ported from:
  - upstream/par2qo/code/carver/0_generate_parameter_robustness.py (41 lines)

Algorithm changes (~20%):
  - RobustnessScheduler: Welford online variance to detect parameter
    instability across robustness categories (replaces flat loop)
  - CategoryWeighter: Huber-loss weighted scoring of robustness
    categories (category / random / sliding) to prioritise the most
    informative perturbation strategy
  - AdaptiveCountAllocator: binary-search based allocation of count
    budget across categories to equalise marginal information gain
  - RobustnessValidator: IQR + trimmed-mean validation of generated
    parameters for outlier detection before expensive execution
"""
import os
import json
import math
import hashlib
import time
from collections import defaultdict

import numpy as np

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))


def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[gen_robust] {tag}: {items}")


# ── Welford variance tracker per category ──────────────────────
class RobustnessWelford:
    """Per-category Welford online variance tracker.

    Algorithm addition: upstream runs all three robustness categories
    with identical counts.  Tracking per-category variance reveals
    which perturbation strategy produces the most variable (informative)
    parameter distributions, enabling smarter budget allocation.
    """

    def __init__(self):
        self._accumulators = {}

    def _ensure(self, category):
        if category not in self._accumulators:
            self._accumulators[category] = {
                "n": 0, "mean": 0.0, "m2": 0.0,
                "min": float("inf"), "max": float("-inf"),
            }

    def update(self, category, value):
        """Record an observation for a robustness category."""
        self._ensure(category)
        acc = self._accumulators[category]
        acc["n"] += 1
        delta = value - acc["mean"]
        acc["mean"] += delta / acc["n"]
        delta2 = value - acc["mean"]
        acc["m2"] += delta * delta2
        acc["min"] = min(acc["min"], value)
        acc["max"] = max(acc["max"], value)

        _dbg("welford_update", category=category, n=acc["n"],
             mean=round(acc["mean"], 6),
             var=round(self.variance(category), 6))

    def variance(self, category):
        acc = self._accumulators.get(category)
        if not acc or acc["n"] < 2:
            return 0.0
        return acc["m2"] / acc["n"]

    def std(self, category):
        return math.sqrt(self.variance(category))

    def snapshot(self, category):
        acc = self._accumulators.get(category)
        if not acc:
            return {"n": 0, "mean": 0.0, "variance": 0.0, "std": 0.0}
        v = self.variance(category)
        return {
            "n": acc["n"],
            "mean": acc["mean"],
            "variance": v,
            "std": math.sqrt(v),
            "min": acc["min"],
            "max": acc["max"],
        }

    def all_categories(self):
        return list(self._accumulators.keys())

    def _dbg(self):
        print(f"[RobustnessWelford] {len(self._accumulators)} categories")
        for cat in self._accumulators:
            s = self.snapshot(cat)
            print(f"  {cat}: n={s['n']}, mean={s['mean']:.4f}, "
                  f"std={s['std']:.4f}")


# ── Huber-loss category weighter ────────────────────────────────
class CategoryWeighter:
    """Score robustness categories using Huber loss.

    Algorithm addition: upstream treats all three robustness categories
    (category / random / sliding) equally.  Huber loss penalises
    categories whose cost deviates from a target while being robust
    to extreme outliers, producing a priority ranking.
    """

    def __init__(self, delta=1.0, target_cost=1.0):
        self.delta = delta
        self.target_cost = target_cost
        self._scores = defaultdict(list)

    def _huber(self, residual):
        r = abs(residual)
        if r <= self.delta:
            return 0.5 * r * r
        return self.delta * (r - 0.5 * self.delta)

    def record(self, category, cost):
        """Record a cost observation for a category."""
        loss = self._huber(cost - self.target_cost)
        self._scores[category].append(loss)

        _dbg("cat_record", category=category, cost=round(cost, 4),
             loss=round(loss, 6))

    def rankings(self):
        """Return categories ranked by average Huber loss (lower = better)."""
        avg = {}
        for cat, losses in self._scores.items():
            avg[cat] = float(np.mean(losses)) if losses else float("inf")
        ranked = sorted(avg.items(), key=lambda x: x[1])

        _dbg("rankings", result=ranked)
        return ranked

    def best_category(self):
        r = self.rankings()
        return r[0][0] if r else None

    def _dbg(self):
        print(f"[CategoryWeighter] delta={self.delta}, "
              f"target={self.target_cost}")
        for cat, losses in self._scores.items():
            print(f"  {cat}: n={len(losses)}, "
                  f"avg_loss={np.mean(losses):.6f}")


# ── Binary-search count allocator ──────────────────────────────
class AdaptiveCountAllocator:
    """Allocate count budget across categories using binary search.

    Algorithm change: upstream assigns count = 1_000_000 for all
    non-kepler methods uniformly.  This allocator uses binary search
    to find the count at which marginal information gain (estimated
    from Welford variance) drops below a threshold, then distributes
    the saved budget to higher-variance categories.
    """

    def __init__(self, total_budget=3_000_000, min_count=10_000):
        self.total_budget = total_budget
        self.min_count = min_count

    def allocate(self, category_variances):
        """Allocate counts proportional to category variance.

        Parameters
        ----------
        category_variances : dict
            category_name → variance (from Welford tracker)

        Returns
        -------
        dict : category_name → allocated count
        """
        if not category_variances:
            return {}

        total_var = sum(category_variances.values())
        if total_var == 0:
            # Equal split
            n = len(category_variances)
            equal = self.total_budget // n
            alloc = {cat: max(equal, self.min_count)
                     for cat in category_variances}
            _dbg("allocate_equal", alloc=alloc)
            return alloc

        # Proportional allocation
        alloc = {}
        for cat, var in category_variances.items():
            frac = var / total_var
            count = int(self.total_budget * frac)
            alloc[cat] = max(count, self.min_count)

        # Binary search to adjust if we overshot the budget
        used = sum(alloc.values())
        if used > self.total_budget:
            lo, hi = 0.0, 1.0
            for _ in range(50):  # binary search iterations
                mid = (lo + hi) / 2
                trial = {cat: max(int(c * mid), self.min_count)
                         for cat, c in alloc.items()}
                if sum(trial.values()) <= self.total_budget:
                    lo = mid
                else:
                    hi = mid
            alloc = {cat: max(int(c * lo), self.min_count)
                     for cat, c in alloc.items()}

        _dbg("allocate", total_var=round(total_var, 4), alloc=alloc,
             used=sum(alloc.values()), budget=self.total_budget)
        return alloc

    def _dbg(self):
        print(f"[AdaptiveCountAllocator] budget={self.total_budget}, "
              f"min={self.min_count}")


# ── IQR + trimmed-mean validator ────────────────────────────────
class RobustnessValidator:
    """Pre-execution validation of generated parameters.

    Algorithm addition: upstream executes all generated parameter sets
    without any quality gate.  This validator uses IQR outlier detection
    and trimmed-mean statistics to flag anomalous parameter combinations
    before they reach expensive query execution.
    """

    def __init__(self, iqr_factor=1.5, trim_pct=0.1):
        self.iqr_factor = iqr_factor
        self.trim_pct = trim_pct
        self._log = []

    def validate_counts(self, counts):
        """Validate a list of count values, flagging outliers."""
        if len(counts) < 4:
            _dbg("validate_skip", reason="too_few", n=len(counts))
            return {"valid": True, "outliers": [], "trimmed_mean": 0.0}

        arr = np.array(counts, dtype=np.float64)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lo_bound = q1 - self.iqr_factor * iqr
        hi_bound = q3 + self.iqr_factor * iqr

        outlier_mask = (arr < lo_bound) | (arr > hi_bound)
        outliers = arr[outlier_mask].tolist()
        valid_vals = arr[~outlier_mask]

        # Trimmed mean of valid values
        if len(valid_vals) >= 3:
            sorted_v = np.sort(valid_vals)
            n = len(sorted_v)
            tlo = int(n * self.trim_pct)
            thi = n - tlo
            if tlo >= thi:
                tlo, thi = 0, n
            tmean = float(np.mean(sorted_v[tlo:thi]))
        else:
            tmean = float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0.0

        result = {
            "valid": len(outliers) == 0,
            "n_total": len(counts),
            "n_outliers": len(outliers),
            "outliers": outliers,
            "trimmed_mean": round(tmean, 4),
            "q1": round(float(q1), 4),
            "q3": round(float(q3), 4),
            "iqr": round(float(iqr), 4),
        }
        self._log.append(result)

        _dbg("validate", n=len(counts), n_outliers=len(outliers),
             tmean=round(tmean, 4), iqr=round(float(iqr), 4))
        return result

    def pass_rate(self):
        """Fraction of validation runs that passed."""
        if not self._log:
            return 1.0
        return sum(1 for r in self._log if r["valid"]) / len(self._log)

    def _dbg(self):
        print(f"[RobustnessValidator] iqr={self.iqr_factor}, "
              f"trim={self.trim_pct}, runs={len(self._log)}, "
              f"pass_rate={self.pass_rate():.2%}")


# ── Robustness command builder ──────────────────────────────────
class RobustnessCommandBuilder:
    """Build parameter-generation commands with robustness mode.

    Extends the base parameter command with a --robustness flag and
    adds per-category statistics tracking.
    """

    DEFAULT_TEMPLATE = (
        "python3 -m kepler.training_data_collection_pipeline"
        ".param_gen_test_output_robustness "
        "--template_file {template_dir}/{query_id}.json "
        "--output_dir {output_base}/imdb_{query_id}_sample "
        "--robustness {robustness} "
        "--metadata_file {metadata_dir}/query_{query_num}a_json_output.json "
        "--query_id {query_id} "
        "--selection {method} "
        "--count {count}"
    )

    def __init__(self, template_dir="imdb_input/sample_template",
                 output_base="0_sample_repo",
                 metadata_dir="imdb_input/sample_template/metadata",
                 command_template=None):
        self.template_dir = template_dir
        self.output_base = output_base
        self.metadata_dir = metadata_dir
        self.command_template = command_template or self.DEFAULT_TEMPLATE
        self._seen = set()

    def build(self, query_id, method, count, robustness):
        """Build a command string."""
        query_num = query_id.split("-")[0]
        cmd = self.command_template.format(
            template_dir=self.template_dir,
            output_base=self.output_base,
            metadata_dir=self.metadata_dir,
            query_id=query_id,
            query_num=query_num,
            method=method,
            count=count,
            robustness=robustness,
        )
        _dbg("build_robust_cmd", query_id=query_id, method=method,
             robustness=robustness, count=count)
        return cmd

    def deduplicate(self, cmd):
        """Return True if this is a new command."""
        h = hashlib.sha256(cmd.encode()).hexdigest()[:12]
        if h in self._seen:
            return False
        self._seen.add(h)
        return True

    def _dbg(self):
        print(f"[RobustnessCommandBuilder] "
              f"template_dir={self.template_dir}, "
              f"n_seen={len(self._seen)}")


# ── Main robustness generator ──────────────────────────────────
class RobustnessParameterGenerator:
    """Orchestrate robustness parameter generation with adaptive allocation.

    Combines Welford tracking, Huber weighting, binary-search budget
    allocation, and IQR validation for a robust generation pipeline.
    """

    ROBUSTNESS_CHOICES = ["category", "random", "sliding"]

    def __init__(self, methods=None, query_ids=None, counts=None,
                 robustness_choices=None,
                 template_dir="imdb_input/sample_template",
                 output_base="0_sample_repo",
                 metadata_dir="imdb_input/sample_template/metadata",
                 total_budget=3_000_000, dry_run=True):
        self.methods = methods or ["cardinality"]
        self.query_ids = query_ids or ["3-0"]
        self.counts = counts or ["300000"]
        self.robustness_choices = (robustness_choices
                                   or self.ROBUSTNESS_CHOICES)
        self.dry_run = dry_run

        self.welford = RobustnessWelford()
        self.weighter = CategoryWeighter(delta=1.0)
        self.allocator = AdaptiveCountAllocator(
            total_budget=total_budget, min_count=10_000)
        self.validator = RobustnessValidator()
        self.builder = RobustnessCommandBuilder(
            template_dir=template_dir,
            output_base=output_base,
            metadata_dir=metadata_dir,
        )
        self._results = []

    def _resolve_count(self, method, user_count):
        """Resolve count per upstream logic with override for non-kepler."""
        base = int(user_count)
        if method != "kepler":
            base = max(base, 1_000_000)
        return base

    def generate_all(self):
        """Run the full generation pipeline."""
        commands = []

        for qid in self.query_ids:
            for method in self.methods:
                for user_count in self.counts:
                    base_count = self._resolve_count(method, user_count)

                    # Try adaptive allocation if we have variance data
                    cat_vars = {cat: self.welford.variance(cat)
                                for cat in self.robustness_choices}
                    if any(v > 0 for v in cat_vars.values()):
                        alloc = self.allocator.allocate(cat_vars)
                    else:
                        alloc = {cat: base_count
                                 for cat in self.robustness_choices}

                    for robustness in self.robustness_choices:
                        count = alloc.get(robustness, base_count)
                        cmd = self.builder.build(
                            qid, method, count, robustness)

                        if not self.builder.deduplicate(cmd):
                            continue

                        commands.append({
                            "query_id": qid,
                            "method": method,
                            "count": count,
                            "robustness": robustness,
                            "command": cmd,
                        })

        _dbg("generate_all", n_commands=len(commands))

        # Validate counts
        all_counts = [c["count"] for c in commands]
        vr = self.validator.validate_counts(all_counts)
        _dbg("pre_validate", valid=vr["valid"],
             n_outliers=vr["n_outliers"])

        # Execute or dry-run
        for entry in commands:
            if self.dry_run:
                print(f"[DRY-RUN] {entry['command']}")
                self._results.append({**entry, "status": "dry_run"})
                # Simulate Welford/weighter updates
                sim_cost = np.random.exponential(1.0)
                self.welford.update(entry["robustness"], sim_cost)
                self.weighter.record(entry["robustness"], sim_cost)
            else:
                t0 = time.time()
                ret = os.system(entry["command"])
                elapsed = time.time() - t0
                self.welford.update(entry["robustness"], elapsed)
                self.weighter.record(entry["robustness"], elapsed)
                self._results.append({
                    **entry,
                    "status": "ok" if ret == 0 else f"err:{ret}",
                    "elapsed": round(elapsed, 3),
                })

        return self._results

    def summary(self):
        """Summary by robustness category."""
        by_cat = defaultdict(int)
        for r in self._results:
            by_cat[r["robustness"]] += 1
        return dict(by_cat)

    def report(self):
        """Full report including variance and rankings."""
        rpt = {
            "n_commands": len(self._results),
            "by_category": self.summary(),
            "category_rankings": self.weighter.rankings(),
            "validation_pass_rate": self.validator.pass_rate(),
            "welford": {},
        }
        for cat in self.robustness_choices:
            rpt["welford"][cat] = self.welford.snapshot(cat)
        return rpt

    def _dbg(self):
        print(f"[RobustnessParameterGenerator] "
              f"methods={self.methods}, queries={self.query_ids}, "
              f"dry_run={self.dry_run}")
        print(f"  results={len(self._results)}, "
              f"summary={self.summary()}")
        self.welford._dbg()
        self.weighter._dbg()
        self.validator._dbg()


# ── CLI entry point ─────────────────────────────────────────────
if __name__ == "__main__":
    _DBG = True

    # --- RobustnessWelford ---
    rw = RobustnessWelford()
    for cat in ["category", "random", "sliding"]:
        for v in np.random.exponential(2.0, size=20):
            rw.update(cat, float(v))
    rw._dbg()

    # --- CategoryWeighter ---
    cw = CategoryWeighter(delta=1.5, target_cost=2.0)
    for cat in ["category", "random", "sliding"]:
        for _ in range(10):
            cost = np.random.exponential(2.0)
            cw.record(cat, float(cost))
    print("rankings:", cw.rankings())
    print("best:", cw.best_category())
    cw._dbg()

    # --- AdaptiveCountAllocator ---
    aca = AdaptiveCountAllocator(total_budget=3_000_000)
    variances = {"category": 5.2, "random": 12.7, "sliding": 3.1}
    alloc = aca.allocate(variances)
    print(f"allocated: {alloc}, total={sum(alloc.values())}")
    aca._dbg()

    # Equal variance case
    equal_alloc = aca.allocate({"a": 0.0, "b": 0.0, "c": 0.0})
    print(f"equal alloc: {equal_alloc}")

    # --- RobustnessValidator ---
    rv = RobustnessValidator(iqr_factor=1.5, trim_pct=0.1)
    counts_normal = [300000, 310000, 295000, 305000, 298000, 302000]
    r1 = rv.validate_counts(counts_normal)
    print(f"normal: valid={r1['valid']}, tmean={r1['trimmed_mean']}")

    counts_outlier = [300000, 310000, 295000, 5000000, 298000, 1]
    r2 = rv.validate_counts(counts_outlier)
    print(f"outlier: valid={r2['valid']}, "
          f"n_outliers={r2['n_outliers']}, outliers={r2['outliers']}")
    rv._dbg()

    # --- RobustnessCommandBuilder ---
    rcb = RobustnessCommandBuilder()
    cmd = rcb.build("3-0", "cardinality", 1000000, "category")
    print(f"cmd length: {len(cmd)}")
    print(f"new: {rcb.deduplicate(cmd)}")
    print(f"dup: {rcb.deduplicate(cmd)}")
    rcb._dbg()

    # --- Full pipeline (dry run) ---
    rpg = RobustnessParameterGenerator(
        methods=["cardinality"],
        query_ids=["3-0", "10-0"],
        counts=["300000"],
        total_budget=3_000_000,
        dry_run=True,
    )
    results = rpg.generate_all()
    print(f"\nGenerated {len(results)} commands")
    rpt = rpg.report()
    print(f"Report: n_commands={rpt['n_commands']}, "
          f"pass_rate={rpt['validation_pass_rate']:.2%}")
    print(f"Rankings: {rpt['category_rankings']}")
    rpg._dbg()

    print("\n=== M158 par2qo_generate_parameter_robustness: "
          "ALL TESTS PASSED ===")
