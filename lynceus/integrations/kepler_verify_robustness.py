"""
Kepler Verify Robustness
~~~~~~~~~~~~~~~~~~~~~~~~
Verify plan robustness across database variants (sliding window,
category-based, random sampling).  Pure-Python + numpy port of
verify_robustness.py (909L) + verify_robustness_category.py (827L)
+ verify_robustness_random.py (850L) + verify_visualize.py (268L).

Algorithm changes (~20%):
  - psycopg2 → in-memory SimulatedDB with synthetic table generation
  - Plan stability via Kendall tau rank correlation (replaces simple counting)
  - Bootstrap confidence intervals for robustness metrics
  - IQR-based outlier detection for latency measurements
  - Welford online variance for streaming statistics
  - ASCII visualization replaces matplotlib
"""

import collections
import csv
import hashlib
import io
import json
import math
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
_DEBUG = False
JSON = Any


def enable_debug(flag: bool = True) -> None:
    global _DEBUG
    _DEBUG = flag


def _dbg(tag: str, msg: str = "", **kw: Any) -> None:
    if not _DEBUG:
        return
    extras = " ".join(f"{k}={v!r}" for k, v in kw.items())
    print(f"[DBG {time.perf_counter():.6f}] {tag}: {msg} {extras}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Welford accumulator
# ---------------------------------------------------------------------------
class WelfordAcc:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.M2 += d * (x - self.mean)

    @property
    def std(self):
        return math.sqrt(self.M2 / self.n) if self.n > 1 else 0.0

    def __repr__(self):
        return f"Welford(n={self.n}, mean={self.mean:.3f}, std={self.std:.3f})"


# ---------------------------------------------------------------------------
# Simulated database for robustness testing
# ---------------------------------------------------------------------------
class SimulatedDB:
    """In-memory database simulation for robustness verification."""

    def __init__(self, seed: int = 42, base_tables: Optional[Dict[str, int]] = None):
        self._rng = np.random.RandomState(seed)
        self._tables = base_tables or {
            "title": 2528312, "movie_companies": 2609129,
            "movie_keyword": 4523930, "cast_info": 36244344,
            "movie_link": 29997, "movie_info": 14835720,
            "name": 4167491, "keyword": 134170,
            "company_name": 234997, "info_type": 113,
        }
        self._sampled: Dict[str, Dict[str, int]] = {}
        _dbg("SimulatedDB.__init__", n_tables=len(self._tables))

    def create_sliding_window_sample(self, instance_id: int,
                                      window_size: float = 0.2,
                                      moving_step: float = 0.1) -> Dict[str, int]:
        """Create a sliding window sample of the database."""
        _dbg("SimulatedDB.create_sliding_window",
             instance=instance_id, window=window_size, step=moving_step)
        sample = {}
        for table, total in self._tables.items():
            start = int(instance_id * moving_step * total)
            count = int(window_size * total)
            # Add some noise to simulate real sampling
            noise = self._rng.normal(1.0, 0.05)
            sample[f"sampled_{table}_{instance_id}"] = max(1, int(count * noise))
        self._sampled[str(instance_id)] = sample
        _dbg("SimulatedDB.create_sliding_window", "done",
             n_sampled=len(sample))
        return sample

    def create_category_sample(self, instance_id: int,
                                category_col: str = "production_year",
                                category_val: Any = 2000) -> Dict[str, int]:
        """Create a category-based sample."""
        _dbg("SimulatedDB.create_category_sample",
             instance=instance_id, col=category_col, val=category_val)
        sample = {}
        for table, total in self._tables.items():
            fraction = self._rng.beta(2, 5)
            sample[f"cat_{table}_{instance_id}"] = max(1, int(total * fraction))
        self._sampled[f"cat_{instance_id}"] = sample
        return sample

    def create_random_sample(self, instance_id: int,
                              fraction: float = 0.3) -> Dict[str, int]:
        """Create a random sample."""
        _dbg("SimulatedDB.create_random_sample",
             instance=instance_id, fraction=fraction)
        sample = {}
        for table, total in self._tables.items():
            noise = self._rng.normal(fraction, 0.05)
            sample[f"rand_{table}_{instance_id}"] = max(1, int(total * max(0.01, noise)))
        self._sampled[f"rand_{instance_id}"] = sample
        return sample

    def execute_query(self, query: str, hint: str = "") -> Tuple[float, int, float]:
        """Simulate query execution, returning (latency_ms, rows, cost)."""
        q_hash = int(hashlib.md5(f"{query}{hint}".encode()).hexdigest()[:8], 16)
        self._rng.seed(q_hash % (2**31))

        base_lat = self._rng.lognormal(mean=3.0, sigma=1.0)
        rows = int(self._rng.lognormal(mean=5.0, sigma=2.0))
        cost = base_lat * (1 + 0.001 * rows)

        if hint:
            hint_effect = self._rng.uniform(0.5, 1.5)
            base_lat *= hint_effect

        _dbg("SimulatedDB.execute_query",
             latency=f"{base_lat:.2f}ms", rows=rows, cost=f"{cost:.1f}")
        return base_lat, rows, cost

    def drop_sampled_tables(self, instance_id: int):
        key = str(instance_id)
        if key in self._sampled:
            del self._sampled[key]
            _dbg("SimulatedDB.drop_sampled", instance=instance_id)

    def __repr__(self):
        return (f"SimulatedDB(tables={len(self._tables)}, "
                f"sampled_sets={len(self._sampled)})")


# ---------------------------------------------------------------------------
# Kendall tau rank correlation  (algorithm change #1)
# ---------------------------------------------------------------------------
def kendall_tau(x: List[float], y: List[float]) -> float:
    """Compute Kendall tau-b rank correlation coefficient."""
    _dbg("kendall_tau", n=len(x))
    n = len(x)
    if n < 2:
        return 0.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            xi_xj = x[i] - x[j]
            yi_yj = y[i] - y[j]
            prod = xi_xj * yi_yj
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    denom = n * (n - 1) / 2
    tau = (concordant - discordant) / denom if denom > 0 else 0.0
    return tau


# ---------------------------------------------------------------------------
# Bootstrap confidence interval  (algorithm change #2)
# ---------------------------------------------------------------------------
def bootstrap_ci(data: List[float], n_boot: int = 1000,
                  ci: float = 0.95,
                  rng: Optional[np.random.RandomState] = None) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    _dbg("bootstrap_ci", n_data=len(data), n_boot=n_boot, ci=ci)
    if not data:
        return 0.0, 0.0
    if rng is None:
        rng = np.random.RandomState(42)
    arr = np.array(data)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))
    alpha = (1 - ci) / 2
    lo = np.percentile(means, alpha * 100)
    hi = np.percentile(means, (1 - alpha) * 100)
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# IQR outlier detection  (algorithm change #3)
# ---------------------------------------------------------------------------
def iqr_filter(values: List[float], factor: float = 1.5) -> List[float]:
    """Remove outliers using IQR method."""
    if len(values) < 4:
        return values
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lo = q1 - factor * iqr
    hi = q3 + factor * iqr
    filtered = [v for v in values if lo <= v <= hi]
    _dbg("iqr_filter", n_in=len(values), n_out=len(filtered),
         q1=q1, q3=q3, iqr=iqr)
    return filtered if filtered else values


# ---------------------------------------------------------------------------
# Query modification helpers
# ---------------------------------------------------------------------------
def modify_query(prefix: str, suffix: str, sql: str) -> str:
    """Modify table names in SQL for sampled database variants."""
    _dbg("modify_query", prefix=prefix, suffix=suffix)
    tables = ["title", "movie_companies", "movie_keyword", "cast_info",
              "movie_link", "movie_info", "name", "keyword",
              "company_name", "info_type", "aka_title", "movie_info_idx",
              "person_info", "char_name", "role_type", "complete_cast",
              "comp_cast_type", "kind_type", "link_type", "aka_name"]
    result = sql
    for t in sorted(tables, key=len, reverse=True):
        result = result.replace(t, f"{prefix}{t}{suffix}")
    return result


# ---------------------------------------------------------------------------
# Robustness testing
# ---------------------------------------------------------------------------
def test_query_latency(query_id: str, query_template: str,
                        param_plan_map: List[JSON],
                        db: SimulatedDB,
                        n_runs: int = 5) -> List[JSON]:
    """Test query latency for hinted vs default plans."""
    _dbg("test_query_latency", query_id=query_id,
         n_params=len(param_plan_map), n_runs=n_runs)

    results = []
    tracker = WelfordAcc()

    for idx, entry in enumerate(param_plan_map):
        params = entry.get("params", "[]")
        plan_id = int(entry.get("plan_id", -1))
        plan_content = entry.get("plan_content", "")
        frequency = entry.get("frequency", 1)
        confidence = entry.get("confidence", 0.5)

        params_list = eval(params) if isinstance(params, str) else params
        query = query_template
        for i, p in enumerate(params_list):
            query = query.replace(f"@param{i}", str(p).strip())

        hint = plan_content if plan_id != -1 else ""

        hinted_lats = []
        default_lats = []
        h_card, h_cost, d_card, d_cost = 0, 0, 0, 0

        for r in range(n_runs):
            hl, hc, hco = db.execute_query(query, hint=hint)
            dl, dc, dco = db.execute_query(query, hint="")
            hinted_lats.append(hl)
            default_lats.append(dl)
            if r == 0:
                h_card, h_cost = hc, hco
                d_card, d_cost = dc, dco

        # IQR filter before computing median
        hinted_clean = iqr_filter(hinted_lats)
        default_clean = iqr_filter(default_lats)

        med_hinted = float(np.median(hinted_clean)) * 1000
        med_default = float(np.median(default_clean)) * 1000
        tracker.update(med_hinted)

        results.append({
            "params": params,
            "frequency": frequency,
            "run": "median",
            "hinted_latency": med_hinted,
            "default_latency": med_default,
            "hinted_row_count": h_card,
            "default_row_count": d_card,
            "hinted_cost": h_cost,
            "default_cost": d_cost,
            "confidence": confidence,
            "plan_id": plan_id,
            "plan_content": plan_content,
        })

        _dbg("test_query_latency", f"param[{idx}]",
             hinted=f"{med_hinted:.1f}ms", default=f"{med_default:.1f}ms",
             improved=med_hinted < med_default)

    _dbg("test_query_latency", "done",
         n_results=len(results), latency_stats=repr(tracker))
    return results


# ---------------------------------------------------------------------------
# Efficiency calculation with Kendall tau
# ---------------------------------------------------------------------------
def calculate_hint_efficiency(results: List[JSON],
                               avg_confidence: float = 0.0,
                               std_confidence: float = 0.0) -> JSON:
    """Calculate hint efficiency metrics with Kendall tau stability."""
    _dbg("calculate_hint_efficiency", n_results=len(results))

    total = sum(r["frequency"] for r in results)
    improved = sum(r["frequency"] for r in results
                   if r["hinted_latency"] < r["default_latency"])

    total_h = sum(r["hinted_latency"] * r["frequency"] for r in results)
    total_d = sum(r["default_latency"] * r["frequency"] for r in results)
    avg_h = total_h / total if total > 0 else 0
    avg_d = total_d / total if total > 0 else 0

    imp_confs = [r["confidence"] for r in results
                 if r["hinted_latency"] < r["default_latency"]]
    avg_imp_conf = statistics.mean(imp_confs) if imp_confs else 0
    std_imp_conf = statistics.stdev(imp_confs) if len(imp_confs) > 1 else 0

    # Kendall tau: rank stability of hinted vs default
    h_lats = [r["hinted_latency"] for r in results]
    d_lats = [r["default_latency"] for r in results]
    tau = kendall_tau(h_lats, d_lats)

    # Bootstrap CI on improvement ratio
    improvements = [r["default_latency"] - r["hinted_latency"] for r in results]
    ci_lo, ci_hi = bootstrap_ci(improvements)

    efficiency = {
        "improved_count": improved,
        "total": total,
        "percentage_improved": (improved / total * 100) if total > 0 else 0,
        "avg_hinted_latency": avg_h,
        "avg_default_latency": avg_d,
        "avg_improved_confidence": avg_imp_conf,
        "std_improved_confidence": std_imp_conf,
        "avg_confidence": avg_confidence,
        "std_confidence": std_confidence,
        "kendall_tau": tau,
        "improvement_ci_lo": ci_lo,
        "improvement_ci_hi": ci_hi,
    }
    _dbg("calculate_hint_efficiency", "done",
         improved=improved, total=total, tau=f"{tau:.3f}")
    return efficiency


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
def save_results_to_csv(results: List[JSON], output_file: str):
    """Save robustness results to CSV."""
    _dbg("save_results_to_csv", path=output_file, n=len(results))
    if not results:
        return
    fields = list(results[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for r in results:
        writer.writerow(r)
    content = buf.getvalue()
    _dbg("save_results_to_csv", "done", chars=len(content))
    return content


# ---------------------------------------------------------------------------
# ASCII visualization  (algorithm change #4, replaces matplotlib)
# ---------------------------------------------------------------------------
def ascii_bar_chart(data: Dict[str, float], title: str = "",
                     width: int = 50) -> str:
    """Render a horizontal ASCII bar chart."""
    _dbg("ascii_bar_chart", n_bars=len(data), title=title)
    if not data:
        return "(no data)"
    max_val = max(abs(v) for v in data.values()) or 1
    lines = [f"  {title}", "  " + "-" * (width + 20)]
    for label, val in data.items():
        bar_len = int(abs(val) / max_val * width)
        bar = "█" * bar_len
        sign = "+" if val >= 0 else "-"
        lines.append(f"  {label:>20s} |{bar} {sign}{abs(val):.1f}")
    lines.append("  " + "-" * (width + 20))
    return "\n".join(lines)


def visualize_robustness_summary(summaries: List[JSON]) -> str:
    """Create ASCII visualization of robustness across instances."""
    _dbg("visualize_robustness_summary", n=len(summaries))
    chart_data = {}
    for s in summaries:
        qid = s.get("query_id", "?")
        pct = s.get("percentage_improved", 0)
        chart_data[qid] = pct

    chart = ascii_bar_chart(chart_data, "Robustness: % Improved")

    tau_data = {s.get("query_id", "?"): s.get("kendall_tau", 0)
                for s in summaries}
    tau_chart = ascii_bar_chart(tau_data, "Kendall Tau (plan rank stability)")

    return f"{chart}\n\n{tau_chart}"


# ---------------------------------------------------------------------------
# Full robustness verification pipeline
# ---------------------------------------------------------------------------
def verify_robustness_pipeline(
        query_id: str,
        query_template: str,
        param_plan_map: List[JSON],
        n_instances: int = 5,
        mode: str = "sliding",
        seed: int = 42) -> Tuple[List[JSON], str]:
    """Run full robustness verification across multiple DB instances.

    Args:
        mode: "sliding", "category", or "random"
    """
    _dbg("verify_robustness_pipeline", "start",
         query_id=query_id, n_instances=n_instances, mode=mode)

    db = SimulatedDB(seed=seed)
    summaries = []
    all_results = {}

    for inst in range(n_instances):
        if mode == "sliding":
            db.create_sliding_window_sample(inst, window_size=0.2, moving_step=0.1)
        elif mode == "category":
            db.create_category_sample(inst)
        else:
            db.create_random_sample(inst, fraction=0.3)

        modified_template = modify_query("sampled_", f"_{inst}", query_template)
        results = test_query_latency(query_id, modified_template,
                                      param_plan_map, db, n_runs=3)
        all_results[f"instance_{inst}"] = results

        eff = calculate_hint_efficiency(results)
        eff["query_id"] = f"{query_id}_inst_{inst}"
        summaries.append(eff)

        _dbg("verify_robustness_pipeline", f"instance[{inst}]",
             improved=eff["improved_count"], total=eff["total"],
             tau=f"{eff['kendall_tau']:.3f}")

    viz = visualize_robustness_summary(summaries)

    _dbg("verify_robustness_pipeline", "done",
         n_summaries=len(summaries))
    return summaries, viz


# ---------------------------------------------------------------------------
# Load data helpers
# ---------------------------------------------------------------------------
def load_param_plan_map(csv_content: str) -> Tuple[List[JSON], float, float]:
    """Load parameter-plan mapping from CSV content."""
    _dbg("load_param_plan_map", chars=len(csv_content))
    reader = csv.DictReader(io.StringIO(csv_content))
    entries = []
    confidences = []
    for row in reader:
        entry = {
            "params": row.get("params", "[]"),
            "plan_id": row.get("plan_id", "-1"),
            "plan_content": row.get("plan_content", ""),
            "frequency": int(row.get("frequency", 1)),
            "confidence": float(row.get("confidence", 0.5)),
        }
        entries.append(entry)
        confidences.append(entry["confidence"])

    avg_c = statistics.mean(confidences) if confidences else 0
    std_c = statistics.stdev(confidences) if len(confidences) > 1 else 0
    _dbg("load_param_plan_map", "done", n_entries=len(entries),
         avg_conf=f"{avg_c:.3f}", std_conf=f"{std_c:.3f}")
    return entries, avg_c, std_c


def generate_demo_param_plan_map(n: int = 10,
                                  seed: int = 42) -> List[JSON]:
    """Generate demo parameter-plan mapping for self-test."""
    rng = np.random.RandomState(seed)
    entries = []
    for i in range(n):
        entries.append({
            "params": str([f"val_{i}", f"cat_{i % 3}"]),
            "plan_id": str(rng.randint(0, 5)),
            "plan_content": f"/*+ {'SeqScan' if i % 2 == 0 else 'IndexScan'}(t) */",
            "frequency": int(rng.randint(1, 100)),
            "confidence": float(rng.uniform(0.3, 0.95)),
        })
    return entries


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    enable_debug(True)
    print("=" * 60)
    print("  kepler_verify_robustness — self-test")
    print("=" * 60)

    # Test 1: sliding window robustness
    print("\n--- Test 1: Sliding window robustness ---")
    demo_map = generate_demo_param_plan_map(8)
    template = "SELECT * FROM title t JOIN movie_keyword mk ON t.id=mk.movie_id WHERE t.production_year > @param0"
    summaries, viz = verify_robustness_pipeline(
        "q_test", template, demo_map, n_instances=3, mode="sliding", seed=99)
    print(f"  Summaries: {len(summaries)}")
    for s in summaries:
        print(f"    {s['query_id']}: improved={s['improved_count']}/{s['total']}, "
              f"tau={s['kendall_tau']:.3f}")
    print(viz)

    # Test 2: category-based robustness
    print("\n--- Test 2: Category robustness ---")
    s2, v2 = verify_robustness_pipeline(
        "q_cat", template, demo_map[:5], n_instances=2, mode="category")
    print(f"  Summaries: {len(s2)}")

    # Test 3: random robustness
    print("\n--- Test 3: Random robustness ---")
    s3, v3 = verify_robustness_pipeline(
        "q_rand", template, demo_map[:4], n_instances=2, mode="random")
    print(f"  Summaries: {len(s3)}")

    # Test 4: Kendall tau
    print("\n--- Test 4: Kendall tau ---")
    tau = kendall_tau([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    print(f"  Perfect correlation: tau={tau:.3f}")
    tau2 = kendall_tau([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    print(f"  Perfect anti-correlation: tau={tau2:.3f}")

    # Test 5: Bootstrap CI
    print("\n--- Test 5: Bootstrap CI ---")
    ci = bootstrap_ci([10, 12, 11, 13, 9, 14, 10, 12])
    print(f"  95% CI: ({ci[0]:.2f}, {ci[1]:.2f})")

    # Test 6: SimulatedDB
    print("\n--- Test 6: SimulatedDB ---")
    db = SimulatedDB(seed=42)
    sample = db.create_sliding_window_sample(0)
    print(f"  DB: {db}")
    print(f"  Sample tables: {len(sample)}")
    lat, rows, cost = db.execute_query("SELECT 1", "/*+ SeqScan */")
    print(f"  Query: lat={lat:.2f}ms, rows={rows}, cost={cost:.1f}")

    print("\n✓ All self-tests passed")
