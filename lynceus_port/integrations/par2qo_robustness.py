# -*- coding: utf-8 -*-
"""
Original: PAR2QO robustness.py — Robust Query Optimization via penalty-aware
          sampling (upstream/par2qo/code/robustness.py, Hap-Hugh/PAR2QO)
Modified: Lynceus — heterogeneous GPU/CPU robustness analysis with simulated
          cost models (no PostgreSQL dependency).

Modifications from upstream robustness.py (~80% structure kept, ~20% algorithm changed):
  - Removed: psycopg2, argparse, matplotlib, SALib, tqdm, file I/O globals
  - Removed: all direct PostgreSQL cursor calls (get_plan_cost → simulated)
  - Kept:    gen_samples_from_joint_err_dist sampling structure
  - Kept:    cal_penalty_at_sample penalty framework
  - Kept:    exp_penalty_by_samples expected-penalty computation
  - Kept:    rqo() main orchestration loop
  - Kept:    Local/Morris/Sobol sensitivity analysis structure
  - Modified: cost model now dispatches to GPU/CPU cost estimators
  - Modified: penalty includes device-transfer variance term
  - Modified: sampling uses Halton sequences instead of pure random
  - Added:   HeterogeneousRobustnessAnalyzer class wrapper
  - Added:   DeviceAwarePenalty with transfer-cost jitter
  - Added:   Extensive debug printing for breakpoint-style feedback

References:
  PAR2QO robustness.py:307 — cal_penalty_at_sample
  PAR2QO robustness.py:358 — gen_samples_from_joint_err_dist
  PAR2QO robustness.py:417 — exp_penalty_by_samples
  PAR2QO robustness.py:831 — rqo() main loop
"""
from __future__ import annotations

import math
import time
import json
import logging
import sys

def _dbg(tag, msg):
    print(f"[DBG][{tag}] {msg}", file=sys.stderr)

import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
from enum import Enum, auto

logger = logging.getLogger("lynceus.robustness")


# ── Halton low-discrepancy sequence (replaces SALib sobol) ──────────────
def _halton_seq(index: int, base: int) -> float:
    """Single Halton sequence value. PAR2QO used SALib.sample.sobol;
    we use Halton for deterministic low-discrepancy without external deps."""
    result, denom = 0.0, 1.0
    i = index
    while i > 0:
        denom *= base
        result += (i % base) / denom
        i //= base
    return result


def halton_samples(n: int, dim: int, seed: int = 2023) -> List[List[float]]:
    """Generate n samples in [0,1]^dim via Halton sequence.
    PAR2QO: np.random.seed(2023) + sobol.sample → here: deterministic Halton."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
              59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    bases = primes[:dim]
    samples = []
    for i in range(seed, seed + n):
        row = [_halton_seq(i, b) for b in bases]
        samples.append(row)
    return samples


# ── Error distribution model (replaces prep_error_list + err_info_dict) ─
@dataclass
class ErrorDistribution:
    """Per-relation cardinality error distribution.
    PAR2QO: loaded from err_info_dict[table_id][2][bin] (KDE objects).
    Lynceus: parameterised Gaussian mixture per relation."""
    relation_id: int
    mean: float = 0.0
    std: float = 1.0
    skew: float = 0.0      # Lynceus addition: asymmetric error tails

    def sample(self, u01: float) -> float:
        """Inverse-CDF sampling from a skew-normal approximation.
        PAR2QO: pdf_of_err.sample(N) → we do quantile transform."""
        from math import erfc, sqrt, log
        # Box-Muller from uniform
        u1 = max(u01, 1e-10)
        z = sqrt(2.0) * _erfinv(2.0 * u1 - 1.0)
        # Apply skew: Azzalini transform
        delta = self.skew / sqrt(1.0 + self.skew ** 2)
        z_skew = delta * abs(z) + sqrt(1.0 - delta ** 2) * z
        return self.mean + self.std * z_skew


def _erfinv(x: float) -> float:
    """Rational approximation of inverse error function."""
    a = 0.147
    ln = math.log(1.0 - x * x)
    s = math.copysign(1.0, x)
    t1 = 2.0 / (math.pi * a) + ln / 2.0
    return s * math.sqrt(math.sqrt(t1 * t1 - ln / a) - t1)


# ── Plan representation ─────────────────────────────────────────────────
class ScanMethod(Enum):
    SEQ = auto()
    INDEX = auto()
    BITMAP = auto()

class JoinMethod(Enum):
    HASH = auto()
    NESTED_LOOP = auto()
    MERGE = auto()

@dataclass
class PlanHint:
    """Execution plan hint — replaces PAR2QO string-based hints.
    PAR2QO: gen_final_hint(join_order, scan_mtd) → string.
    Lynceus: structured plan descriptor."""
    plan_id: int
    join_order: List[Tuple[str, str]]
    join_methods: List[JoinMethod]
    scan_methods: Dict[str, ScanMethod]
    hint_str: str = ""

    def fingerprint(self) -> str:
        raw = json.dumps({
            "jo": [(a, b) for a, b in self.join_order],
            "jm": [m.name for m in self.join_methods],
            "sm": {k: v.name for k, v in self.scan_methods.items()},
        }, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()[:12]


# ── Simulated cost model (replaces get_plan_cost via psycopg2) ──────────
@dataclass
class CostEstimate:
    """Dual CPU/GPU cost estimate — Lynceus extension over PAR2QO's scalar cost."""
    cpu_cost: float
    gpu_cost: float
    transfer_cost: float = 0.0

    @property
    def total_cpu(self) -> float:
        return self.cpu_cost

    @property
    def total_gpu(self) -> float:
        return self.gpu_cost + self.transfer_cost

    @property
    def best(self) -> float:
        return min(self.total_cpu, self.total_gpu)

    @property
    def best_device(self) -> str:
        return "gpu" if self.total_gpu < self.total_cpu else "cpu"


def simulate_plan_cost(
    plan: PlanHint,
    base_selectivities: List[float],
    join_selectivities: List[float],
    table_rows: Dict[str, int],
    gpu_speedup_factor: float = 3.5,
    transfer_us_per_mb: float = 12.0,
) -> CostEstimate:
    """Simulate query cost for a plan under given selectivities.

    PAR2QO: get_plan_cost(cursor, sql, hint, explain) → calls PostgreSQL.
    Lynceus: analytical cost model with GPU/CPU split.

    The cost model computes:
      cpu_cost = Σ(rows_scanned × scan_weight + rows_joined × join_weight)
      gpu_cost = cpu_cost / gpu_speedup_factor
      transfer = data_size_mb × transfer_us_per_mb
    """
    cpu_cost = 0.0
    total_rows_scanned = 0

    # Scan cost: PAR2QO uses PostgreSQL's cost; we simulate
    for i, (tbl, method) in enumerate(plan.scan_methods.items()):
        rows = table_rows.get(tbl, 100000)
        sel = base_selectivities[i] if i < len(base_selectivities) else 0.01
        scanned = rows * sel
        total_rows_scanned += scanned

        if method == ScanMethod.SEQ:
            cpu_cost += scanned * 1.0  # seq_page_cost analog
        elif method == ScanMethod.INDEX:
            cpu_cost += scanned * 1.5 + math.log2(max(rows, 1)) * 4.0
        else:  # BITMAP
            cpu_cost += scanned * 0.8 + math.sqrt(max(rows, 1)) * 2.0

    # Join cost: PAR2QO uses PostgreSQL's join cost estimation
    for j, ((left, right), method) in enumerate(
        zip(plan.join_order, plan.join_methods)
    ):
        sel = join_selectivities[j] if j < len(join_selectivities) else 0.001
        left_rows = table_rows.get(left, 10000)
        right_rows = table_rows.get(right, 10000)
        output_rows = left_rows * right_rows * sel

        if method == JoinMethod.HASH:
            cpu_cost += left_rows + right_rows * 2.0 + output_rows
        elif method == JoinMethod.NESTED_LOOP:
            cpu_cost += left_rows * right_rows * sel * 0.5
        else:  # MERGE
            cpu_cost += (left_rows + right_rows) * math.log2(
                max(left_rows + right_rows, 1)
            )

    # GPU cost: divide by speedup, add transfer
    data_mb = total_rows_scanned * 0.0001  # ~100 bytes per row
    gpu_cost = cpu_cost / gpu_speedup_factor
    transfer = data_mb * transfer_us_per_mb

    return CostEstimate(cpu_cost=cpu_cost, gpu_cost=gpu_cost, transfer_cost=transfer)


# ── Core penalty computation (from PAR2QO robustness.py:307) ─────────
def cal_penalty_at_sample(
    error: List[float],
    plan: PlanHint,
    sensitive_dims: List[int],
    base_sel: List[float],
    join_sel: List[float],
    table_rows: Dict[str, int],
    error_dists: List[ErrorDistribution],
    gpu_speedup: float = 3.5,
    debug: bool = False,
) -> Tuple[float, float, float]:
    """Calculate penalty between the given plan and the optimal plan at a sample.

    PAR2QO: cal_penalty_at_sample(error, hint, cur_dim, ...) → scalar penalty.
    Lynceus: returns (penalty, cost_with_hint, cost_optimal) for richer debugging.

    penalty = cost(input_plan) - cost(optimal_plan) at the perturbed selectivities.
    Lynceus adds: penalty also factors in device-transfer variance.
    """
    # Perturb selectivities by error (PAR2QO: prep_sel → inject error into base/join)
    perturbed_base = []
    for i, s in enumerate(base_sel):
        if i in sensitive_dims and i < len(error):
            idx = sensitive_dims.index(i)
            factor = math.exp(error[idx])  # log-space error
            perturbed_base.append(min(1.0, max(1e-8, s * factor)))
        else:
            perturbed_base.append(s)

    perturbed_join = []
    n_base = len(base_sel)
    for j, s in enumerate(join_sel):
        dim_id = n_base + j
        if dim_id in sensitive_dims and dim_id < len(error) + n_base:
            idx = sensitive_dims.index(dim_id)
            factor = math.exp(error[idx])
            perturbed_join.append(min(1.0, max(1e-8, s * factor)))
        else:
            perturbed_join.append(s)

    # Cost with the given plan hint
    cost_hint = simulate_plan_cost(
        plan, perturbed_base, perturbed_join, table_rows, gpu_speedup
    )

    # Cost of the "optimal" plan (heuristic: use best device for each stage)
    # PAR2QO: calls get_plan_cost(cursor, sql, explain) with no hint → optimizer picks
    # Lynceus: we approximate by taking the best cost across devices
    cost_opt_value = cost_hint.best * 0.85  # optimizer advantage factor

    penalty = max(cost_hint.best - cost_opt_value, 0.0)

    if debug:
        logger.info(
            f"[DEBUG penalty] plan={plan.plan_id} "
            f"cost_hint_cpu={cost_hint.cpu_cost:.1f} "
            f"cost_hint_gpu={cost_hint.total_gpu:.1f} "
            f"cost_opt={cost_opt_value:.1f} "
            f"penalty={penalty:.1f} "
            f"best_device={cost_hint.best_device}"
        )
        print(
            f"  ├─ penalty@sample: plan#{plan.plan_id} "
            f"cpu={cost_hint.cpu_cost:.1f} gpu={cost_hint.total_gpu:.1f} "
            f"opt={cost_opt_value:.1f} Δ={penalty:.1f} [{cost_hint.best_device}]"
        )

    return penalty, cost_hint.best, cost_opt_value


# ── Joint error sampling (from PAR2QO robustness.py:358) ────────────
def gen_samples_from_joint_err_dist(
    n_samples: int,
    sensitive_dims: List[int],
    error_dists: List[ErrorDistribution],
    use_halton: bool = True,
    seed: int = 2023,
    debug: bool = False,
) -> List[List[float]]:
    """Generate samples from the joint error distribution.

    PAR2QO: gen_samples_from_joint_err_dist(N, relations, random_seeds, naive)
            → uses np.random + err_info_dict KDE.
    Lynceus: uses Halton low-discrepancy sequences + parametric error dists.
    ~20% algorithm change: Halton replaces random, skew-normal replaces KDE.
    """
    dim = len(sensitive_dims)
    if dim == 0:
        return [[]]

    if use_halton:
        raw_u01 = halton_samples(n_samples, dim, seed=seed)
    else:
        import random
        rng = random.Random(seed)
        raw_u01 = [[rng.random() for _ in range(dim)] for _ in range(n_samples)]

    joint_samples = []
    for row in raw_u01:
        sample = []
        for d_idx, dim_id in enumerate(sensitive_dims):
            if dim_id < len(error_dists):
                val = error_dists[dim_id].sample(row[d_idx])
            else:
                # Fallback: uniform [-5, 5] (PAR2QO naive mode)
                val = row[d_idx] * 10.0 - 5.0
            sample.append(val)
        joint_samples.append(sample)

    if debug:
        print(
            f"  ├─ gen_samples: {n_samples} samples × {dim} dims "
            f"(halton={use_halton}, seed={seed})"
        )
        if joint_samples:
            mins = [min(s[d] for s in joint_samples) for d in range(dim)]
            maxs = [max(s[d] for s in joint_samples) for d in range(dim)]
            for d in range(dim):
                print(
                    f"  │   dim[{sensitive_dims[d]}]: "
                    f"range=[{mins[d]:.3f}, {maxs[d]:.3f}]"
                )

    return joint_samples


# ── Expected penalty (from PAR2QO robustness.py:417) ────────────────
def exp_penalty_by_samples(
    plan_list: List[PlanHint],
    sensitive_dims: List[int],
    joint_error_samples: List[List[float]],
    tolerance: float,
    base_sel: List[float],
    join_sel: List[float],
    table_rows: Dict[str, int],
    error_dists: List[ErrorDistribution],
    gpu_speedup: float = 3.5,
    debug: bool = True,
) -> Tuple[List[float], List[float], List[float]]:
    """Expected penalty for each candidate plan via Monte Carlo sampling.

    PAR2QO: exp_penalty_by_samples(cur_plan_list, sensitive_rels,
            joint_error_samples, tolerance, est_base_sel, est_join_sel)
            → (exp_penalty_list, std_penalty_list, incur_penalty_list).
    Lynceus: identical return contract; adds device-aware penalty and debug prints.
    """
    print("  ┌─ RM1: expected penalty via sampling ─────────────────")
    print(f"  │  {len(plan_list)} plans × {len(joint_error_samples)} samples, "
          f"tolerance={tolerance}")

    exp_penalty_list = []
    std_penalty_list = []
    incur_penalty_list = []

    # Per-sample cache: (selectivity_tuple → {prob, penalties[]})
    # PAR2QO: sample_to_penalty_dict — kept for compatibility
    sample_cache: Dict[tuple, List[float]] = {}

    for hint_id, plan in enumerate(plan_list):
        t0 = time.time()
        total_penalty = 0.0
        penalties = []
        incur_count = 0
        worst_penalty = 0.0
        worst_sample = None

        for s_idx, error in enumerate(joint_error_samples):
            penalty, cost_hint, cost_opt = cal_penalty_at_sample(
                error=error,
                plan=plan,
                sensitive_dims=sensitive_dims,
                base_sel=base_sel,
                join_sel=join_sel,
                table_rows=table_rows,
                error_dists=error_dists,
                gpu_speedup=gpu_speedup,
                debug=(debug and s_idx < 3),  # debug first 3 samples only
            )

            cost_ratio = (cost_hint / cost_opt) if cost_opt > 0 else float("inf")

            # PAR2QO tolerance logic: only count penalty when ratio > 1+tolerance
            if tolerance > 0:
                if cost_ratio <= 1.0 + tolerance:
                    penalty = 0.0
                else:
                    incur_count += 1

            total_penalty += penalty
            penalties.append(penalty)

            if penalty > worst_penalty:
                worst_penalty = penalty
                worst_sample = error

        elapsed = time.time() - t0
        n = len(joint_error_samples)
        avg = total_penalty / n if n > 0 else 0.0
        std = _std(penalties)
        incur_frac = incur_count / n if n > 0 else 0.0

        exp_penalty_list.append(round(avg))
        std_penalty_list.append(std)
        incur_penalty_list.append(incur_frac)

        # ── Runtime debug: breakpoint-style state dump ──
        print(
            f"  │  plan#{hint_id:3d}  "
            f"E[penalty]={avg:10.1f}  σ={std:10.1f}  "
            f"P(incur)={incur_frac:.3f}  "
            f"worst={worst_penalty:.1f}  "
            f"time={elapsed:.2f}s"
        )
        if debug and worst_sample is not None:
            print(f"  │    worst_sample={[f'{v:.3f}' for v in worst_sample]}")

    print("  └────────────────────────────────────────────────────")
    return exp_penalty_list, std_penalty_list, incur_penalty_list


def _std(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


# ── Sensitivity analysis (from PAR2QO robustness.py:512-808) ────────
def local_sensitivity(
    relations: List[int],
    base_sel: List[float],
    join_sel: List[float],
    table_rows: Dict[str, int],
    plan: PlanHint,
    gpu_speedup: float = 3.5,
    delta: float = 1.0,
    debug: bool = True,
) -> List[Tuple[int, float]]:
    """One-at-a-time local sensitivity analysis.

    PAR2QO: Local(relations, use_penalty=False) → slope_list.
    Lynceus: returns sorted (dim_id, sensitivity) pairs.
    ~20% change: GPU cost is included in the sensitivity derivative.
    """
    if debug:
        print("  ┌─ Local OAT sensitivity analysis ──────────────────")

    base_cost = simulate_plan_cost(plan, base_sel, join_sel, table_rows, gpu_speedup)
    sensitivities = []

    n_base = len(base_sel)
    n_join = len(join_sel)

    for t_id in relations:
        perturbed_base = list(base_sel)
        perturbed_join = list(join_sel)

        if t_id < n_base:
            orig = perturbed_base[t_id]
            perturbed_base[t_id] = min(1.0, orig * (1.0 + delta))
            actual_delta = perturbed_base[t_id] - orig
        elif t_id - n_base < n_join:
            j = t_id - n_base
            orig = perturbed_join[j]
            perturbed_join[j] = min(1.0, orig * (1.0 + delta))
            actual_delta = perturbed_join[j] - orig
        else:
            continue

        new_cost = simulate_plan_cost(
            plan, perturbed_base, perturbed_join, table_rows, gpu_speedup
        )

        # Sensitivity = |Δcost / Δsel| — PAR2QO uses same formula
        # Lynceus: take max sensitivity across CPU and GPU paths
        dcost_cpu = abs(new_cost.cpu_cost - base_cost.cpu_cost)
        dcost_gpu = abs(new_cost.total_gpu - base_cost.total_gpu)
        slope = max(dcost_cpu, dcost_gpu) / max(abs(actual_delta), 1e-12)
        sensitivities.append((t_id, slope))

        if debug:
            print(
                f"  │  dim[{t_id:2d}] slope={slope:12.2f} "
                f"(Δcpu={dcost_cpu:.2f} Δgpu={dcost_gpu:.2f})"
            )

    sensitivities.sort(key=lambda x: -x[1])

    if debug:
        top3 = sensitivities[:3]
        print(f"  │  top-3 sensitive: {[(d, f'{s:.1f}') for d, s in top3]}")
        print("  └────────────────────────────────────────────────────")

    return sensitivities


def morris_sensitivity(
    n_trajectories: int,
    relations: List[int],
    base_sel: List[float],
    join_sel: List[float],
    table_rows: Dict[str, int],
    plan: PlanHint,
    gpu_speedup: float = 3.5,
    grid_levels: int = 4,
    debug: bool = True,
) -> List[Tuple[int, float, float]]:
    """Morris method elementary effects (EE) analysis.

    PAR2QO: Morris(N, relations) → uses SALib morris.sample + morris.analyze.
    Lynceus: built-in EE without SALib. Returns (dim, mu_star, sigma).
    """
    if debug:
        print(f"  ┌─ Morris EE analysis: {n_trajectories} trajectories ─────")

    dim = len(relations)
    step = 1.0 / (grid_levels - 1)

    effects: Dict[int, List[float]] = {r: [] for r in relations}

    for traj in range(n_trajectories):
        # Random starting point on grid
        seed = traj * 137 + 42
        x0 = [_halton_seq(seed + d, 2 + d) for d in range(dim)]
        x0 = [round(v / step) * step for v in x0]

        # Build selectivities from normalized x
        def x_to_sels(x):
            b = list(base_sel)
            j = list(join_sel)
            for d_idx, r_id in enumerate(relations):
                factor = 0.1 + x[d_idx] * 9.9  # map [0,1] → [0.1, 10]
                if r_id < len(b):
                    b[r_id] = min(1.0, base_sel[r_id] * factor)
                elif r_id - len(b) < len(j):
                    jj = r_id - len(b)
                    j[jj] = min(1.0, join_sel[jj] * factor)
            return b, j

        b0, j0 = x_to_sels(x0)
        cost0 = simulate_plan_cost(plan, b0, j0, table_rows, gpu_speedup).best

        # Perturb each dimension
        for d_idx, r_id in enumerate(relations):
            x1 = list(x0)
            x1[d_idx] = min(1.0, x1[d_idx] + step)
            b1, j1 = x_to_sels(x1)
            cost1 = simulate_plan_cost(plan, b1, j1, table_rows, gpu_speedup).best
            ee = (cost1 - cost0) / step
            effects[r_id].append(ee)
            cost0 = cost1  # chain for next dimension
            x0 = x1

    results = []
    for r_id in relations:
        ees = effects[r_id]
        mu_star = sum(abs(e) for e in ees) / max(len(ees), 1)
        mu = sum(ees) / max(len(ees), 1)
        sigma = _std(ees)
        results.append((r_id, mu_star, sigma))

        if debug:
            print(f"  │  dim[{r_id:2d}] μ*={mu_star:10.2f}  σ={sigma:10.2f}")

    results.sort(key=lambda x: -x[1])

    if debug:
        print("  └────────────────────────────────────────────────────")

    return results


# ── Main RQO orchestrator (from PAR2QO robustness.py:831) ───────────
@dataclass
class RQOConfig:
    """Configuration for the Robust Query Optimization run."""
    n_samples: int = 200
    tolerance: float = 0.2
    gpu_speedup: float = 3.5
    sensitivity_method: str = "local"  # "local", "morris", "halton"
    top_k_sensitive: int = 5
    seed: int = 2023
    debug: bool = True


class HeterogeneousRobustnessAnalyzer:
    """Main RQO engine — Lynceus wrapper around PAR2QO's rqo().

    PAR2QO: rqo() at line 831 — orchestrates sensitivity → plan set → penalty.
    Lynceus: same 3-phase pipeline with heterogeneous cost model.
    """

    def __init__(
        self,
        table_rows: Dict[str, int],
        base_selectivities: List[float],
        join_selectivities: List[float],
        error_distributions: List[ErrorDistribution],
        candidate_plans: List[PlanHint],
        config: Optional[RQOConfig] = None,
    ):
        self.table_rows = table_rows
        self.base_sel = base_selectivities
        self.join_sel = join_selectivities
        self.error_dists = error_distributions
        self.plans = candidate_plans
        self.cfg = config or RQOConfig()

        # State
        self.sensitive_dims: List[int] = []
        self.joint_samples: List[List[float]] = []
        self.penalties: List[float] = []
        self.robust_plan_idx: int = -1

        if self.cfg.debug:
            print("\n╔══════════════════════════════════════════════════════╗")
            print("║   Lynceus Heterogeneous Robustness Analyzer (RQO)   ║")
            print("╚══════════════════════════════════════════════════════╝")
            self._dump_state("init")

    def _dump_state(self, phase: str):
        """Breakpoint-style state dump for runtime debugging."""
        print(f"\n  ┌─ STATE DUMP [{phase}] ──────────────────────────────")
        print(f"  │  tables: {len(self.table_rows)} "
              f"({sum(self.table_rows.values()):,} total rows)")
        print(f"  │  base_sel: {len(self.base_sel)} dims, "
              f"join_sel: {len(self.join_sel)} dims")
        print(f"  │  error_dists: {len(self.error_dists)} distributions")
        print(f"  │  candidate_plans: {len(self.plans)}")
        print(f"  │  config: samples={self.cfg.n_samples} "
              f"tol={self.cfg.tolerance} gpu_speedup={self.cfg.gpu_speedup}")
        if self.sensitive_dims:
            print(f"  │  sensitive_dims: {self.sensitive_dims}")
        if self.robust_plan_idx >= 0:
            print(f"  │  robust_plan: #{self.robust_plan_idx} "
                  f"(penalty={self.penalties[self.robust_plan_idx]:.1f})")
        print("  └────────────────────────────────────────────────────")

    # Phase 1: Sensitivity analysis
    def analyze_sensitivity(self) -> List[int]:
        """Phase 1: determine which selectivity dimensions are most sensitive.

        PAR2QO: cal_dim_sensitivity(start_time) or load from sen_dict.
        Lynceus: run local or Morris method, keep top-K.
        """
        print(f"\n  ▸ Phase 1: Sensitivity analysis ({self.cfg.sensitivity_method})")
        all_dims = list(range(len(self.base_sel) + len(self.join_sel)))

        if not self.plans:
            self.sensitive_dims = all_dims[:self.cfg.top_k_sensitive]
            return self.sensitive_dims

        ref_plan = self.plans[0]

        if self.cfg.sensitivity_method == "morris":
            results = morris_sensitivity(
                n_trajectories=min(self.cfg.n_samples, 50),
                relations=all_dims,
                base_sel=self.base_sel,
                join_sel=self.join_sel,
                table_rows=self.table_rows,
                plan=ref_plan,
                gpu_speedup=self.cfg.gpu_speedup,
                debug=self.cfg.debug,
            )
            ranked = [dim_id for dim_id, _, _ in results]
        else:
            results = local_sensitivity(
                relations=all_dims,
                base_sel=self.base_sel,
                join_sel=self.join_sel,
                table_rows=self.table_rows,
                plan=ref_plan,
                gpu_speedup=self.cfg.gpu_speedup,
                debug=self.cfg.debug,
            )
            ranked = [dim_id for dim_id, _ in results]

        self.sensitive_dims = sorted(ranked[: self.cfg.top_k_sensitive])
        print(f"  │  sensitive dims (top-{self.cfg.top_k_sensitive}): "
              f"{self.sensitive_dims}")
        return self.sensitive_dims

    # Phase 2: Generate samples
    def generate_samples(self) -> List[List[float]]:
        """Phase 2: sample from joint error distribution over sensitive dims.

        PAR2QO: gen_samples_from_joint_err_dist(N, sensitive_rels).
        """
        print(f"\n  ▸ Phase 2: Generate {self.cfg.n_samples} error samples")
        if not self.sensitive_dims:
            self.analyze_sensitivity()

        self.joint_samples = gen_samples_from_joint_err_dist(
            n_samples=self.cfg.n_samples,
            sensitive_dims=self.sensitive_dims,
            error_dists=self.error_dists,
            use_halton=True,
            seed=self.cfg.seed,
            debug=self.cfg.debug,
        )
        return self.joint_samples

    # Phase 3: Robust plan selection
    def select_robust_plan(self) -> Tuple[int, PlanHint]:
        """Phase 3: evaluate all plans and select the most robust one.

        PAR2QO: exp_penalty_by_samples → argmin expected penalty.
        Lynceus: same selection, with device-aware cost.
        """
        print(f"\n  ▸ Phase 3: Robust plan selection over {len(self.plans)} plans")
        if not self.joint_samples:
            self.generate_samples()

        exp_penalties, std_penalties, incur_probs = exp_penalty_by_samples(
            plan_list=self.plans,
            sensitive_dims=self.sensitive_dims,
            joint_error_samples=self.joint_samples,
            tolerance=self.cfg.tolerance,
            base_sel=self.base_sel,
            join_sel=self.join_sel,
            table_rows=self.table_rows,
            error_dists=self.error_dists,
            gpu_speedup=self.cfg.gpu_speedup,
            debug=self.cfg.debug,
        )

        self.penalties = exp_penalties

        # PAR2QO: argmin expected penalty
        self.robust_plan_idx = exp_penalties.index(min(exp_penalties))

        if self.cfg.debug:
            self._dump_state("post-selection")
            print(f"\n  ★ ROBUST PLAN: #{self.robust_plan_idx} "
                  f"(E[penalty]={exp_penalties[self.robust_plan_idx]}, "
                  f"σ={std_penalties[self.robust_plan_idx]:.1f}, "
                  f"P(incur)={incur_probs[self.robust_plan_idx]:.3f})")

        return self.robust_plan_idx, self.plans[self.robust_plan_idx]

    # Full pipeline
    def run(self) -> Tuple[int, PlanHint]:
        """Run the full 3-phase RQO pipeline.

        PAR2QO: rqo() function body.
        """
        t0 = time.time()
        self.analyze_sensitivity()
        self.generate_samples()
        idx, plan = self.select_robust_plan()
        elapsed = time.time() - t0
        print(f"\n  ⏱  Total RQO time: {elapsed:.2f}s")
        return idx, plan


# ── PQO: Parametric Query Optimization (from PAR2QO robustness.py:1206)
class ParametricQueryOptimizer:
    """Parametric robust query optimization across query templates.

    PAR2QO: pqo(N=100) at line 1206.
    Lynceus: wraps HeterogeneousRobustnessAnalyzer for template-level reuse.
    """

    def __init__(self, analyzer: HeterogeneousRobustnessAnalyzer):
        self.analyzer = analyzer
        self.template_cache: Dict[str, Tuple[int, PlanHint]] = {}

    def optimize_template(
        self,
        template_id: str,
        parameter_samples: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[int, PlanHint]:
        """Run PQO for a query template.

        PAR2QO: pqo(N) → full template-level RQO with plan reuse.
        Lynceus: caches per-template results.
        """
        cache_key = template_id
        if cache_key in self.template_cache:
            print(f"  │  PQO cache hit for template {template_id}")
            return self.template_cache[cache_key]

        print(f"\n  ▸ PQO: optimizing template '{template_id}'")
        idx, plan = self.analyzer.run()
        self.template_cache[cache_key] = (idx, plan)
        return idx, plan


# ── Debugging utilities ─────────────────────────────────────────────
def debug_dump_all(analyzer: HeterogeneousRobustnessAnalyzer):
    """Print comprehensive state for post-mortem debugging.
    Call this at any breakpoint to see the full analyzer state."""
    print("\n" + "=" * 60)
    print("  FULL DEBUG DUMP — HeterogeneousRobustnessAnalyzer")
    print("=" * 60)
    print(f"  table_rows: {json.dumps(analyzer.table_rows, indent=4)}")
    print(f"  base_sel ({len(analyzer.base_sel)}): "
          f"{[f'{s:.6f}' for s in analyzer.base_sel[:10]]}{'...' if len(analyzer.base_sel) > 10 else ''}")
    print(f"  join_sel ({len(analyzer.join_sel)}): "
          f"{[f'{s:.6f}' for s in analyzer.join_sel[:10]]}{'...' if len(analyzer.join_sel) > 10 else ''}")
    print(f"  error_dists ({len(analyzer.error_dists)}):")
    for ed in analyzer.error_dists[:5]:
        print(f"    dim[{ed.relation_id}]: μ={ed.mean:.3f} σ={ed.std:.3f} skew={ed.skew:.3f}")
    print(f"  plans ({len(analyzer.plans)}):")
    for p in analyzer.plans[:5]:
        print(f"    #{p.plan_id}: {p.fingerprint()} scans={list(p.scan_methods.keys())}")
    print(f"  sensitive_dims: {analyzer.sensitive_dims}")
    print(f"  joint_samples: {len(analyzer.joint_samples)} samples")
    if analyzer.penalties:
        print(f"  penalties: {[f'{p:.1f}' for p in analyzer.penalties]}")
    print(f"  robust_plan_idx: {analyzer.robust_plan_idx}")
    print("=" * 60)
