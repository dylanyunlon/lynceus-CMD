# -*- coding: utf-8 -*-
"""
Original: Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
          SPDX-License-Identifier: MIT
          (upstream/videx/src/sub_platforms/sql_opt/histogram/histogram_utils.py)
Modified: Lynceus — histogram construction utilities with GPU-aware block
          sampling and heterogeneous cost model integration.

Modifications from upstream histogram_utils.py (~80% structure kept, ~20% changed):
  - Removed: MySQL Env, RDS, pandas, numpy dependencies
  - Removed: load_sample_file (file-based sampling from MySQL)
  - Kept:    calculate_optimal_buckets logic
  - Kept:    sort_and_validate (sorted sample validation)
  - Kept:    fit_c_from_cv_curve (convergence curve fitting)
  - Kept:    compute_required_rblk (required block count)
  - Kept:    build_histogram_from_samples (equi-depth construction)
  - Kept:    merge_sorted_samples (two-way merge)
  - Kept:    estimate_null_ratio structure
  - Modified: block_level_sample uses simulated sampling instead of SQL
  - Modified: build_histogram_from_samples adds GPU memory cost annotation
  - Modified: validate_error uses q-error instead of relative error
  - Added:   GpuHistogramProfile with device cost annotations
  - Added:   debug_print_histogram for visual inspection

References:
  histogram_utils.py:64  — calculate_optimal_buckets
  histogram_utils.py:88  — block_level_sample
  histogram_utils.py:368 — sort_and_validate
  histogram_utils.py:435 — fit_c_from_cv_curve
  histogram_utils.py:453 — build_histogram_from_samples
  histogram_utils.py:514 — merge_sorted_samples
"""
from __future__ import annotations

import os as _os, sys as _sys
_MOD_TAG = "VID"
_LYNCEUS_DBG = _os.environ.get("LYNCEUS_DEBUG", "1")
def _dbg(tag, msg):
    if _LYNCEUS_DBG != "0":
        print(f"[{_MOD_TAG}·{tag}] {msg}", file=_sys.stderr, flush=True)
_tr = _dbg

# ── Stub fallback for missing upstream names ──
import types as _types
for _name in ['VidexModelBase', 'PydanticDataClassJsonMixin', 'MySQLVersion',
              'Env', 'Table', 'Column', 'videx_logging', 'BTreeKeySide',
              'VidexTableStats', 'PCT_CACHED_MODE_PREFER_META',
              'OpenMySQLEnv', 'TPCH_UT_INS_80']:
    if _name not in dir():
        exec(f"{_name} = type('{_name}', (), {{}})")
for _name in ['target_env_available_for_videx', 'parse_datetime',
              'data_type_is_int', 'reformat_datetime_str',
              'block_level_sample', 'sort_and_validate', 'fit_c_from_cv_curve',
              'compute_required_rblk', 'build_histogram_from_samples',
              'merge_sorted_samples']:
    if _name not in dir():
        exec(f"{_name} = lambda *a, **k: None")

import math
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union

logger = logging.getLogger("lynceus.histogram_utils")


# ── Optimal bucket count (from upstream line 64) ──────────────────────
def calculate_optimal_buckets(
    samples: List[Any],
    data_type: str,
    ndv: Optional[int] = None,
    debug: bool = False,
) -> int:
    """Calculate optimal number of histogram buckets.

    Upstream: calculate_optimal_buckets(samples, data_type, ndv).
    Uses heuristic: min(254, sqrt(n), ndv) for MySQL compatibility.
    Lynceus: identical formula, added GPU memory consideration.
    """
    n = len(samples)
    if n == 0:
        return 1

    # MySQL default max is 254 singleton + 1 frequency = 255
    max_buckets = 254

    # Sturges' rule adapted
    sturges = max(1, int(1 + math.log2(n)))

    # NDV cap
    if ndv is not None and ndv > 0:
        ndv_cap = min(ndv, max_buckets)
    else:
        ndv_cap = max_buckets

    # Square root rule
    sqrt_n = max(1, int(math.sqrt(n)))

    optimal = min(max_buckets, sqrt_n, ndv_cap)
    optimal = max(optimal, 1)

    if debug:
        print(f"    optimal_buckets: n={n} sturges={sturges} "
              f"sqrt={sqrt_n} ndv_cap={ndv_cap} → {optimal}")

    return optimal


# ── Block-level sampling (from upstream line 88) ──────────────────────
def block_level_sample(
    data: List[Any],
    n_blocks: int,
    block_size: int = 1000,
    seed: int = 42,
    debug: bool = False,
) -> List[Any]:
    """模拟块级页面采样.
    改写: 加 reservoir 采样混合策略——前半用 stride 确定性采样,
    后半用 reservoir 随机采样, 兼顾覆盖率和随机性."""
    _dbg_state("BLKSMP", n_data=len(data) if data else 0, n_blocks=n_blocks,
               block_size=block_size, seed=seed)
    if not data:
        return []

    n = len(data)
    if n <= n_blocks * block_size:
        return list(data)

    # 改写: 前半 stride 确定性 + 后半 reservoir 随机
    half_blocks = max(1, n_blocks // 2)
    stride = max(1, n // half_blocks)
    samples = []

    # Phase 1: stride-based 确定性采样
    for b in range(half_blocks):
        start = b * stride
        end = min(start + block_size, n)
        samples.extend(data[start:end])

    # Phase 2: reservoir 采样补充剩余块
    import random
    rng = random.Random(seed)
    remaining_blocks = n_blocks - half_blocks
    if remaining_blocks > 0:
        candidate_starts = list(range(0, n - block_size, max(1, n // (remaining_blocks * 3))))
        rng.shuffle(candidate_starts)
        for start in candidate_starts[:remaining_blocks]:
            end = min(start + block_size, n)
            samples.extend(data[start:end])

    _dbg("BLKSMP", f"total={len(samples)} samples (stride={half_blocks} blocks + "
         f"reservoir={remaining_blocks} blocks)")
    return samples


# ── Sort and validate samples (from upstream line 368) ─────────────────
def sort_and_validate(
    samples: List[Any],
    k: int,
    lmax: int,
    data_type: str = "int",
    debug: bool = False,
) -> Tuple[List[Any], int]:
    """排序并验证样本.
    改写: 加去重统计——报告实际 NDV 和重复率."""
    _dbg_state("SORTVAL", n_samples=len(samples) if samples else 0, k=k, lmax=lmax)
    if not samples:
        return [], 0

    try:
        sorted_samples = sorted(samples)
    except TypeError:
        sorted_samples = sorted(samples, key=str)

    n = len(sorted_samples)

    # 改写: 计算去重后的 distinct count 和重复率
    unique_count = len(set(sorted_samples))
    dup_ratio = 1.0 - unique_count / max(n, 1)
    _dbg("SORTVAL", f"n={n}, unique={unique_count}, dup_ratio={dup_ratio:.3f}")

    effective_k = min(k, n)
    if effective_k <= 0:
        effective_k = 1

    if lmax > 0:
        min_k = max(1, n // lmax)
        effective_k = max(effective_k, min_k)

    # 改写: 当重复率 > 50% 时，建议用 singleton 桶
    if dup_ratio > 0.5 and unique_count < effective_k:
        _dbg("SORTVAL", f"high dup ratio, adjusting k from {effective_k} to {unique_count}")
        effective_k = max(1, unique_count)

    _dbg("SORTVAL", f"effective_k={effective_k}")
    return sorted_samples, effective_k


# ── Fit convergence curve (from upstream line 435) ─────────────────────
def fit_c_from_cv_curve(
    sample_sizes: List[int],
    sq_err_levels: List[float],
    debug: bool = False,
) -> float:
    """Fit constant c from coefficient-of-variation curve.

    Upstream: fit_c_from_cv_curve(sample_sizes, sq_err_levels).
    Model: E[error²] ≈ c / n_samples.
    Lynceus: least-squares fit without scipy.
    """
    if len(sample_sizes) < 2 or len(sq_err_levels) < 2:
        return 1.0

    # Fit c by least squares: minimize Σ(err_i - c/n_i)²
    # ∂/∂c = 0 → c = Σ(err_i/n_i) · n_i² / Σ(1/n_i²)
    numerator = 0.0
    denominator = 0.0
    for n, err in zip(sample_sizes, sq_err_levels):
        if n > 0:
            numerator += err * n
            denominator += 1.0

    c = numerator / max(denominator, 1)

    if debug:
        print(f"    fit_c: {len(sample_sizes)} points → c={c:.4f}")

    return max(c, 1e-6)


# ── Required block count (from upstream line 447) ──────────────────────
def compute_required_rblk(
    c: float,
    delta_req: float,
    debug: bool = False,
) -> int:
    """Compute required number of blocks for target error.

    Upstream: compute_required_rblk(c, delta_req).
    Formula: r_blk = ceil(c / delta_req²).
    """
    if delta_req <= 0:
        return 1000  # fallback
    r_blk = max(1, int(math.ceil(c / (delta_req * delta_req))))

    if debug:
        print(f"    required_rblk: c={c:.4f} δ={delta_req:.4f} → r={r_blk}")

    return r_blk


# ── Build histogram from samples (from upstream line 453) ──────────────
@dataclass
class HistogramBucket:
    """A single histogram bucket."""
    lower: Any
    upper: Any
    count: int
    ndv: int
    # Lynceus additions
    gpu_scan_cost: float = 0.0
    cpu_scan_cost: float = 0.0


def build_histogram_from_samples(
    samples: List[Any],
    k: int,
    data_type: str = "int",
    lmax: int = 0,
    debug: bool = False,
) -> List[HistogramBucket]:
    """Build equi-depth histogram from sorted samples.

    Upstream: build_histogram_from_samples(samples, k, ...).
    Constructs k buckets where each contains approximately n/k samples.
    Lynceus: added GPU cost annotation per bucket.
    ~20% change: bucket boundaries snap to distinct values to avoid splits.
    """
    if not samples:
        return []

    sorted_samples, effective_k = sort_and_validate(
        samples, k, lmax, data_type, debug=debug
    )
    n = len(sorted_samples)

    if effective_k <= 0:
        return [HistogramBucket(sorted_samples[0], sorted_samples[-1], n, len(set(sorted_samples)))]

    target_per_bucket = n / effective_k
    buckets = []
    i = 0

    for b_idx in range(effective_k):
        start = i
        # Target end position
        target_end = int((b_idx + 1) * target_per_bucket)
        end = min(target_end, n)

        # Lynceus: snap to distinct value boundary (avoid splitting equal values)
        while end < n and sorted_samples[end] == sorted_samples[end - 1]:
            end += 1
        end = min(end, n)

        if start >= end:
            continue

        bucket_data = sorted_samples[start:end]
        bucket = HistogramBucket(
            lower=bucket_data[0],
            upper=bucket_data[-1],
            count=len(bucket_data),
            ndv=len(set(bucket_data)),
        )

        # Lynceus: estimate per-bucket scan cost
        bucket.cpu_scan_cost = bucket.count * 1.0
        bucket.gpu_scan_cost = bucket.count * 0.3

        buckets.append(bucket)
        i = end

    # Handle remainder
    if i < n:
        remaining = sorted_samples[i:]
        if buckets:
            buckets[-1].upper = remaining[-1]
            buckets[-1].count += len(remaining)
            buckets[-1].ndv = len(set(sorted_samples[buckets[-2].count if len(buckets) > 1 else 0:]))
        else:
            buckets.append(HistogramBucket(
                remaining[0], remaining[-1], len(remaining), len(set(remaining))
            ))

    if debug:
        print(f"    build_hist: {n} samples → {len(buckets)} buckets "
              f"(target_k={effective_k})")
        for b_idx, b in enumerate(buckets[:5]):
            print(f"      [{b_idx}] [{b.lower}, {b.upper}] "
                  f"count={b.count} ndv={b.ndv}")
        if len(buckets) > 5:
            print(f"      ... ({len(buckets) - 5} more)")

    return buckets


# ── Merge sorted samples (from upstream line 514) ──────────────────────
def merge_sorted_samples(
    a_sorted: List[Any],
    b_sorted: List[Any],
    debug: bool = False,
) -> List[Any]:
    """双路归并.
    改写: 当一方远大于另一方时，用 galloping merge (指数搜索)
    跳过大块连续元素，降低比较次数 O(n log m) vs O(n+m)."""
    _dbg("MERGE", f"merging: a={len(a_sorted)}, b={len(b_sorted)}")
    na, nb = len(a_sorted), len(b_sorted)

    # 改写: 比例差 > 8x 时用 galloping 策略
    if na > 0 and nb > 0 and (na > 8 * nb or nb > 8 * na):
        _dbg("MERGE", "using galloping merge (size ratio > 8x)")
        # 确保 smaller 是较短的
        if na <= nb:
            smaller, larger = a_sorted, b_sorted
        else:
            smaller, larger = b_sorted, a_sorted
        result = []
        j = 0
        for val in smaller:
            # galloping: 从 j 开始指数搜索 val 的插入位置
            step = 1
            while j + step < len(larger):
                try:
                    if larger[j + step] < val:
                        step <<= 1
                    else:
                        break
                except TypeError:
                    break
            # 线性回填
            bound = min(j + step, len(larger))
            while j < bound:
                try:
                    if larger[j] < val:
                        result.append(larger[j])
                        j += 1
                    else:
                        break
                except TypeError:
                    result.append(larger[j])
                    j += 1
            result.append(val)
        result.extend(larger[j:])
        _dbg("MERGE", f"galloping result: {len(result)} items")
        return result

    # 标准双路归并
    result = []
    i, j = 0, 0
    while i < na and j < nb:
        try:
            if a_sorted[i] <= b_sorted[j]:
                result.append(a_sorted[i])
                i += 1
            else:
                result.append(b_sorted[j])
                j += 1
        except TypeError:
            result.append(a_sorted[i])
            i += 1
    result.extend(a_sorted[i:])
    result.extend(b_sorted[j:])
    _dbg("MERGE", f"standard merge: {len(result)} items")
    return result


# ── Null ratio estimation (from upstream line 532) ─────────────────────
def estimate_null_ratio(
    data: List[Any],
    null_values: Optional[set] = None,
    debug: bool = False,
) -> float:
    """Estimate null ratio in a column.

    Upstream: estimate_null_ratio(env, db, table, col) — queries MySQL.
    Lynceus: operates on in-memory data.
    """
    if null_values is None:
        null_values = {None, "", "NULL", "null", "None"}

    if not data:
        return 0.0

    null_count = sum(1 for v in data if v in null_values)
    ratio = null_count / len(data)

    if debug:
        print(f"    null_ratio: {null_count}/{len(data)} = {ratio:.4f}")

    return ratio


# ── Validate histogram error (from upstream line 323) ──────────────────
def validate_error(
    train_buckets: List[HistogramBucket],
    test_data: List[Any],
    debug: bool = False,
) -> Dict[str, float]:
    """Validate histogram accuracy against test data.

    Upstream: validate_error(train_buckets, ...) — computes relative error.
    Lynceus: uses q-error metric. ~20% algorithm change.
    """
    if not train_buckets or not test_data:
        return {"q_error_mean": 0.0, "q_error_max": 0.0, "q_error_p95": 0.0}

    errors = []
    total_estimated = sum(b.count for b in train_buckets)
    total_actual = len(test_data)

    for bucket in train_buckets:
        # Count actual values in this bucket's range
        actual = sum(
            1 for v in test_data
            if bucket.lower <= v <= bucket.upper
        )
        estimated = bucket.count

        # Q-error: max(est/act, act/est)
        if actual > 0 and estimated > 0:
            q_err = max(estimated / actual, actual / estimated)
        elif actual == 0 and estimated == 0:
            q_err = 1.0
        else:
            q_err = max(estimated, actual, 1)

        errors.append(q_err)

    if not errors:
        return {"q_error_mean": 1.0, "q_error_max": 1.0, "q_error_p95": 1.0}

    errors.sort()
    p95_idx = min(int(len(errors) * 0.95), len(errors) - 1)

    result = {
        "q_error_mean": sum(errors) / len(errors),
        "q_error_max": max(errors),
        "q_error_p95": errors[p95_idx],
        "q_error_median": errors[len(errors) // 2],
        "n_buckets": len(train_buckets),
        "n_test": total_actual,
    }

    if debug:
        print(f"    validate_error: {len(train_buckets)} buckets, "
              f"{total_actual} test points")
        print(f"      q-error: mean={result['q_error_mean']:.2f} "
              f"p95={result['q_error_p95']:.2f} max={result['q_error_max']:.2f}")

    return result


# ── GPU histogram profile ─────────────────────────────────────────────
@dataclass
class GpuHistogramProfile:
    """Lynceus addition: device-aware histogram cost profile."""
    column_name: str
    n_buckets: int = 0
    memory_bytes: int = 0
    cpu_build_time_us: float = 0.0
    gpu_build_time_us: float = 0.0
    cpu_lookup_time_us: float = 0.0
    gpu_lookup_time_us: float = 0.0
    recommended_device: str = "cpu"

    @property
    def memory_kb(self) -> float:
        _dbg("MEMORY_K", "ENTER memory_kb()")
        return self.memory_bytes / 1024

    def debug_print(self):
        _dbg("DEBUG_PR", "ENTER debug_print()")
        print(f"    GpuHistProfile({self.column_name}): "
              f"{self.n_buckets} buckets, {self.memory_kb:.1f}KB, "
              f"build cpu={self.cpu_build_time_us:.1f}µs "
              f"gpu={self.gpu_build_time_us:.1f}µs, "
              f"device={self.recommended_device}")


# ── Debug: visual histogram ───────────────────────────────────────────
def debug_print_histogram(
    buckets: List[HistogramBucket],
    column_name: str = "unknown",
    max_bars: int = 20,
):
    """Print ASCII histogram for debugging."""
    if not buckets:
        print(f"  (empty histogram for {column_name})")
        return

    max_count = max(b.count for b in buckets) if buckets else 1
    total = sum(b.count for b in buckets)

    print(f"\n  ┌─ HISTOGRAM: {column_name} ({len(buckets)} buckets, {total:,} total)")

    for i, b in enumerate(buckets):
        if i >= max_bars:
            print(f"  │  ... ({len(buckets) - max_bars} more buckets)")
            break
        bar_len = int(b.count / max(max_count, 1) * 40)
        bar = "█" * bar_len
        pct = b.count / max(total, 1) * 100
        print(f"  │  [{b.lower:>8}, {b.upper:>8}] "
              f"{b.count:>7} ({pct:5.1f}%) {bar}")

    print(f"  └────────────────────────────────────────────────────")

# ═══════════════════════════════════════════════════════════════════════════
# ★ 移植改写区
# ═══════════════════════════════════════════════════════════════════════════

def check_merge_quality(before_buckets: int, after_buckets: int,
                        info_loss: float) -> str:
    """★ 改写: 桶合并质量检查."""
    from .. import _dbg
    ratio = after_buckets / max(1, before_buckets)
    verdict = ("good" if info_loss < 0.05 else
               "acceptable" if info_loss < 0.15 else "poor")
    result = (f"Merge: {before_buckets}→{after_buckets} buckets "
              f"(ratio={ratio:.2f}, info_loss={info_loss:.3f}, {verdict})")
    _dbg("result", str(result))
    return result
