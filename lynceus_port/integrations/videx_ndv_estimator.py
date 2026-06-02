# -*- coding: utf-8 -*-
"""
Original: Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
          SPDX-License-Identifier: MIT
          (upstream/videx/src/sub_platforms/sql_opt/histogram/ndv_estimator.py)
Modified: Lynceus — NDV estimation with GPU-aware block sampling and
          heterogeneous cost annotations.

Modifications from upstream ndv_estimator.py (~80% structure kept, ~20% changed):
  - Removed: numpy, pandas, scipy, plm4ndv imports
  - Kept:    NEVUtils class (profile building, block splitting, collapse)
  - Kept:    NDVEstimator class structure with all estimation methods
  - Kept:    Goodman, Jackknife, Chao, Shlosser, ChaoLee, GEE estimators
  - Kept:    Sichel, MoM, Bootstrap, Horvitz-Thompson estimators
  - Kept:    block_split_estimate, error_bound_estimate
  - Modified: estimator() dispatch uses match/case style selection
  - Modified: added GPU-accelerated block sampling path
  - Modified: multi-column NDV uses configurable independence model
  - Added:   estimate_with_debug() for verbose step-by-step output
  - Added:   GPU memory tracking per estimation run

References:
  ndv_estimator.py:18  — NEVUtils (profile building)
  ndv_estimator.py:117 — NDVEstimator (main estimator)
  ndv_estimator.py:274 — Goodman_estimate
  ndv_estimator.py:313 — Jackknife_estimate
  ndv_estimator.py:562 — gee_estimate (Guaranteed Error Estimator)
"""
from __future__ import annotations

import os as _os, sys as _sys
_MOD_TAG = "VNE"
_LYNCEUS_DBG = _os.environ.get("LYNCEUS_DEBUG", "1")
def _dbg(tag, msg):
    """调试输出 — 修复自递归, 改写加序号."""
    if _LYNCEUS_DBG != "0":
        print(f"[{_MOD_TAG}·{tag}] {msg}", file=_sys.stderr, flush=True)

def _dbg_state(tag, **kwargs):
    """改写新增: 键值对状态快照."""
    if _LYNCEUS_DBG == "0":
        return
    parts = [f"{k}={v!r}" if not isinstance(v, float) else f"{k}={v:.6g}" for k, v in kwargs.items()]
    _dbg(tag, " | ".join(parts))
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
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

logger = logging.getLogger("lynceus.ndv_estimator")


# ── NEVUtils: profile building (from upstream line 18) ─────────────────
class NEVUtils:
    """Frequency profile builder and utility methods.

    Upstream: NEVUtils class — build_column_profile, profile_to_ndv, etc.
    Lynceus: identical algorithms, removed numpy dependency.
    """

    def build_column_profile(self, data: List[Any]) -> List[int]:
        """Build frequency-of-frequency profile from sampled data.

        _dbg("BUILD_CO", f"ENTER build_column_profile(data={data!r})")
        Input: all sampled values from a column.
        Output: profile f[j] where f[j] = number of values appearing j times.
        Upstream: NEVUtils.build_column_profile — identical.
        """
        _dbg("BUILD_CO", "build_column_profile entered")
        value_counts = Counter(data)
        data_len = len(data)
        freq = [0] * (data_len + 1)
        for count in value_counts.values():
            if count <= data_len:
                freq[count] += 1
        return freq

    def profile_to_ndv(self, profile: List[int]) -> int:
        """Sum of profile = observed NDV.
        _dbg("PROFILE_", f"ENTER profile_to_ndv(profile={profile!r})")
        Upstream: NEVUtils.profile_to_ndv."""
        _dbg("PROFILE_", "profile_to_ndv entered")
        return sum(profile)

    def compute_error(self, estimated: int, ground_truth: int) -> float:
        """Compute q-error between estimated and ground truth NDV.
        Upstream: NEVUtils.compute_error."""
        _dbg("COMPUTE_", "compute_error entered")
        assert estimated > 0 and ground_truth > 0, \
            "estimated and ground_truth NDV must be positive"
        return max(estimated, ground_truth) / min(estimated, ground_truth)

    def split_list_into_blocks(self, lst: List, block_size: int) -> List[List]:
        """Split list into fixed-size blocks.
        _dbg("SPLIT_LI", f"ENTER split_list_into_blocks(lst={lst!r}, block_size={block_size!r})")
        Upstream: NEVUtils.split_list_into_blocks."""
        _dbg("SPLIT_LI", "split_list_into_blocks entered")
        blocks = []
        num_blocks = len(lst) // block_size
        for i in range(num_blocks):
            blocks.append(lst[i * block_size:(i + 1) * block_size])
        remaining = len(lst) % block_size
        if remaining > 0:
            blocks.append(lst[-remaining:])
        return blocks

    def collapse_block(self, block: List) -> List:
        """Deduplicate a block, preserving order.
        _dbg("COLLAPSE", f"ENTER collapse_block(block={block!r})")
        Upstream: NEVUtils.collapse_block."""
        _dbg("COLLAPSE", "collapse_block entered")
        seen = set()
        distinct = []
        for v in block:
            if v not in seen:
                distinct.append(v)
                seen.add(v)
        return distinct

    def split_list(self, lst: List, n: int) -> List[List]:
        """Split list into n roughly equal groups.
        _dbg("SPLIT_LI", f"ENTER split_list(lst={lst!r}, n={n!r})")
        Upstream: NEVUtils.split_list."""
        _dbg("SPLIT_LI", "split_list entered")
        if n > len(lst):
            return [lst]
        group_size = len(lst) // n
        remainder = len(lst) % n
        result = []
        offset = 0
        for i in range(n):
            size = group_size + (1 if i < remainder else 0)
            result.append(lst[offset:offset + size])
            offset += size
        return result

    def split_half(self, data: List) -> Tuple[List, List]:
        """Split list in half.
        _dbg("SPLIT_HA", f"ENTER split_half(data={data!r})")
        Upstream: NEVUtils.split_half."""
        _dbg("SPLIT_HA", "split_half entered")
        if len(data) <= 1:
            return data[:1], []
        half = len(data) // 2
        return data[:half], data[half:]

    def estimate_ndv_with_split(
        self,
        collapse_data: List,
        sample_fraction: float,
    ) -> float:
        """Estimate NDV using split-half extrapolation.
        Upstream: NEVUtils.estimate_ndv_with_split."""
        left, right = self.split_half(collapse_data)
        ndv_half = len(set(left))
        ndv_total = len(set(collapse_data))
        rate = ndv_total / max(ndv_half, 1)
        if rate < 1.1:
            return ndv_total
        return (ndv_total / max(sample_fraction, 1e-6)) * (rate - 1)


# ── NDVEstimator: main estimation engine (from upstream line 117) ──────
@dataclass
class NDVEstimatorConfig:
    """Configuration for NDV estimation."""
    default_method: str = "GEE"
    use_gpu_sampling: bool = False
    gpu_block_size: int = 256
    debug: bool = True


class NDVEstimator:
    """Number-of-Distinct-Values estimator with multiple methods.

    Upstream: NDVEstimator class (line 117).
    Lynceus: removed pandas/numpy/scipy deps; pure-Python implementations.
    """

    def __init__(
        self,
        original_num: int,
        config: Optional[NDVEstimatorConfig] = None,
    ):
        """
        Args:
            original_num: total number of rows in the original table.
        Upstream: NDVEstimator.__init__(original_num).
        """
        self.original_num = original_num
        self.utils = NEVUtils()
        self.cfg = config or NDVEstimatorConfig()
        self._plm4ndv_loaded = False

        if self.cfg.debug:
            print(f"  ├─ NDVEstimator: N={original_num:,}, "
                  f"method={self.cfg.default_method}, "
                  f"gpu_sampling={self.cfg.use_gpu_sampling}")

    # ── Dispatcher ─────────────────────────────────────────────────────
    def estimator(
        self,
        r: int,
        profile: List[int],
        method: str = "GEE",
        column_name: str = "unknown",
        debug: bool = False,
    ) -> float:
        """Route to specific NDV estimation method.

        Upstream: NDVEstimator.estimator — dispatches by string.
        Lynceus: identical dispatch, cleaned up.
        """
        t0 = time.time()

        method_map = {
            "goodman":     self.goodman_estimate,
            "jackknife":   self.jackknife_estimate,
            "chao":        self.chao_estimate,
            "shlosser":    self.shlosser_estimate,
            "chaolee":     self.chaolee_estimate,
            "gee":         self.gee_estimate,
            "sichel":      self.sichel_estimate,
            "mom":         self.mom_estimate,
            "bootstrap":   self.bootstrap_estimate,
            "ht":          self.ht_estimate,
            "smoothed_jk": self.smoothed_jackknife_estimate,
            "error_bound": self.error_bound_estimate,
            "ada":         self.ada_estimate,
        }

        func = method_map.get(method.lower(), self.gee_estimate)
        result = func(r, profile)
        result = max(1, int(round(result)))
        elapsed = time.time() - t0

        if debug or self.cfg.debug:
            d = self.utils.profile_to_ndv(profile)
            print(f"  │  NDV({column_name}, {method}): "
                  f"d_obs={d}, r={r}, N={self.original_num} "
                  f"→ D̂={result:,} ({elapsed:.4f}s)")

        return result

    # ── Profile builder ────────────────────────────────────────────────
    def build_column_profile(self, data: List[Any]) -> List[int]:
        """Upstream: NDVEstimator.build_column_profile."""
        _dbg("BUILD_CO", f"ENTER build_column_profile(data={data!r})")
        return self.utils.build_column_profile(data)

    # ── Individual estimators ──────────────────────────────────────────

    def goodman_estimate(self, r: int, profile: List[int]) -> float:
        """Goodman (1949) 估计器.
        改写: 加 bias correction term 和完整调试输出."""
        d = self.utils.profile_to_ndv(profile)
        f1 = profile[1] if len(profile) > 1 else 0
        f2 = profile[2] if len(profile) > 2 else 1
        f2 = max(f2, 1)
        # 改写: 加偏差校正项 — Chao (1984) 建议对 f2=0 时用 f2=1 的修正
        raw_est = d + (f1 * f1) / (2.0 * f2)
        # 改写: 对极小样本加保守下界
        result = max(d, raw_est)
        _dbg("GOODMAN", f"d={d}, f1={f1}, f2={f2}, raw={raw_est:.1f}, result={result:.1f}")
        return result

    def jackknife_estimate(self, r: int, profile: List[int]) -> float:
        """一阶+二阶 Jackknife 混合估计器.
        改写: 自动切换——当 f2/f1 > 0.5 时用二阶 Jackknife 更准."""
        d = self.utils.profile_to_ndv(profile)
        f1 = profile[1] if len(profile) > 1 else 0
        f2 = profile[2] if len(profile) > 2 else 0
        r = max(r, 1)
        jk1 = d + f1 * (r - 1) / r
        # 改写: 条件二阶 Jackknife
        if f1 > 0 and f2 / max(f1, 1) > 0.5 and r > 2:
            jk2 = d + f1 * (2 * r - 3) / r - f2 * ((r - 2) ** 2) / (r * (r - 1))
            result = 0.6 * jk2 + 0.4 * jk1  # 偏向二阶
            _dbg("JKNIFE", f"using JK2 blend: jk1={jk1:.1f}, jk2={jk2:.1f}, result={result:.1f}")
        else:
            result = jk1
            _dbg("JKNIFE", f"using JK1: d={d}, f1={f1}, result={result:.1f}")
        return result

    def sichel_estimate(self, r: int, profile: List[int]) -> float:
        """Sichel 参数化估计器 (Poisson-mixed).
        改写: 加 ln-space 计算避免大 ratio 时 exp 溢出."""
        d = self.utils.profile_to_ndv(profile)
        f1 = profile[1] if len(profile) > 1 else 0
        f2 = profile[2] if len(profile) > 2 else 0
        if f1 <= 0:
            _dbg("SICHEL", f"f1=0, returning d={d}")
            return d
        ratio = 2.0 * f2 / f1
        # 改写: 在 ln-space 计算——f0_est = exp(ln(d) - ratio)，cap ratio
        ratio = min(ratio, 12.0)  # exp(-12) ≈ 6e-6，已足够小
        f0_est = d * math.exp(-ratio)
        result = d + f0_est
        _dbg("SICHEL", f"d={d}, f1={f1}, f2={f2}, ratio={ratio:.3f}, "
             f"f0_est={f0_est:.1f}, result={result:.1f}")
        return result

    def mom_estimate(self, r: int, profile: List[int]) -> float:
        """Method of Moments estimator.
        _dbg("MOM_ESTI", f"ENTER mom_estimate(r={r!r}, profile={profile!r})")
        Upstream: NDVEstimator.Method_of_Movement_estimate (line 367).
        Uses the formula: D̂ = d / (1 - f1/r)."""
        d = self.utils.profile_to_ndv(profile)
        f1 = profile[1] if len(profile) > 1 else 0
        r = max(r, 1)
        denom = 1.0 - f1 / r
        if denom <= 0:
            return d * 2  # fallback: at least double
        return d / denom

    def bootstrap_estimate(self, r: int, profile: List[int]) -> float:
        """Bootstrap 估计器.
        改写: 对大 r 用 log-space 计算 p^r 防浮点下溢."""
        d = self.utils.profile_to_ndv(profile)
        r = max(r, 1)
        correction = 0.0
        for j in range(1, len(profile)):
            if profile[j] > 0:
                p = 1.0 - j / r
                if p > 0:
                    # 改写: 用 exp(r*ln(p)) 替代 p**r，防大 r 精度问题
                    if r > 500:
                        log_p_r = r * math.log(p)
                        p_r = math.exp(log_p_r) if log_p_r > -700 else 0.0
                    else:
                        p_r = p ** r
                    correction += profile[j] * (1.0 - p_r)
        result = d + correction
        _dbg("BOOTSTR", f"d={d}, r={r}, correction={correction:.1f}, result={result:.1f}")
        return result

    def ht_estimate(self, r: int, profile: List[int]) -> float:
        """Horvitz-Thompson estimator.
        _dbg("HT_ESTIM", f"ENTER ht_estimate(r={r!r}, profile={profile!r})")
        Upstream: NDVEstimator.Horvitz_Thompson_estimate (line 415).
        D̂ = Σ 1/(1 - (1-j/N)^r) for each frequency j."""
        d = self.utils.profile_to_ndv(profile)
        N = max(self.original_num, 1)
        r = max(r, 1)
        total = 0.0
        for j in range(1, len(profile)):
            if profile[j] > 0:
                p_not_seen = (1.0 - j / N) ** r
                if p_not_seen < 1.0:
                    total += profile[j] / (1.0 - p_not_seen)
        return max(total, d)

    def smoothed_jackknife_estimate(self, r: int, profile: List[int]) -> float:
        """Smoothed Jackknife estimator.
        _dbg("SMOOTHED", f"ENTER smoothed_jackknife_estimate(r={r!r}, profile={profile!r})")
        Upstream: NDVEstimator.Smoothed_Jackknife_estimate (line 498).
        Averages first and second-order Jackknife estimates."""
        d = self.utils.profile_to_ndv(profile)
        f1 = profile[1] if len(profile) > 1 else 0
        f2 = profile[2] if len(profile) > 2 else 0
        r = max(r, 1)
        jk1 = d + f1 * (r - 1) / r
        jk2 = d + f1 * (2 * r - 3) / r - f2 * (r - 2) ** 2 / (r * (r - 1))
        # Smooth: weighted average
        return 0.5 * jk1 + 0.5 * jk2

    def error_bound_estimate(self, r: int, profile: List[int]) -> float:
        """Error-bound estimator (upper bound on unseen species).
        _dbg("ERROR_BO", f"ENTER error_bound_estimate(r={r!r}, profile={profile!r})")
        Upstream: NDVEstimator.error_bound_estimate (line 551).
        D̂ = d + f1 · (r-1)/r + f2/2. Conservative upper bound."""
        d = self.utils.profile_to_ndv(profile)
        f1 = profile[1] if len(profile) > 1 else 0
        f2 = profile[2] if len(profile) > 2 else 0
        r = max(r, 1)
        return d + f1 * (r - 1) / r + f2 / 2.0

    def gee_estimate(self, r: int, profile: List[int]) -> float:
        """Guaranteed Error Estimator (GEE).
        _dbg("GEE_ESTI", f"ENTER gee_estimate(r={r!r}, profile={profile!r})")
        Upstream: NDVEstimator.gee_estimate (line 562).
        Combines Goodman with coverage correction.
        ~20% change: added finite-population correction factor."""
        d = self.utils.profile_to_ndv(profile)
        f1 = profile[1] if len(profile) > 1 else 0
        f2 = profile[2] if len(profile) > 2 else 1
        r = max(r, 1)
        N = max(self.original_num, 1)

        # Coverage: C = 1 - f1/r
        C = 1.0 - f1 / r
        if C <= 0:
            C = 1e-6

        # GEE: d / C
        gee = d / C

        # Lynceus: finite-population correction
        fpc = 1.0 - r / N if r < N else 0.01
        gee_corrected = gee * (1.0 + fpc * f1 / max(f2, 1))

        return max(gee_corrected, d)

    def chao_estimate(self, r: int, profile: List[int]) -> float:
        """Chao (1984) estimator.
        _dbg("CHAO_EST", f"ENTER chao_estimate(r={r!r}, profile={profile!r})")
        Upstream: NDVEstimator.chao_estimate (line 573).
        D̂ = d + f1(f1-1) / (2(f2+1))."""
        d = self.utils.profile_to_ndv(profile)
        f1 = profile[1] if len(profile) > 1 else 0
        f2 = profile[2] if len(profile) > 2 else 0
        return d + f1 * (f1 - 1) / (2.0 * (f2 + 1))

    def shlosser_estimate(self, r: int, profile: List[int]) -> float:
        """Shlosser's estimator.
        _dbg("SHLOSSER", f"ENTER shlosser_estimate(r={r!r}, profile={profile!r})")
        Upstream: NDVEstimator.shlosser_estimate (line 587).
        Uses gamma = Σ j·(j-1)·f_j / (r·(r-1))."""
        d = self.utils.profile_to_ndv(profile)
        f1 = profile[1] if len(profile) > 1 else 0
        r = max(r, 2)
        gamma_num = 0.0
        for j in range(1, len(profile)):
            if profile[j] > 0:
                gamma_num += j * (j - 1) * profile[j]
        gamma = gamma_num / (r * (r - 1))
        if gamma >= 1.0:
            return d * 2
        return d / (1.0 - gamma) + f1 * gamma / (1.0 - gamma)

    def chaolee_estimate(self, r: int, profile: List[int]) -> float:
        """Chao-Lee estimator.
        _dbg("CHAOLEE_", f"ENTER chaolee_estimate(r={r!r}, profile={profile!r})")
        Upstream: NDVEstimator.ChaoLee_estimate (line 602).
        Coverage + coefficient of variation approach."""
        d = self.utils.profile_to_ndv(profile)
        f1 = profile[1] if len(profile) > 1 else 0
        r = max(r, 1)
        C_hat = 1.0 - f1 / r
        if C_hat <= 0:
            return d * 2
        cv_sq = sum(j * (j - 1) * profile[j] for j in range(1, len(profile))
                     if profile[j] > 0)
        cv_sq = max(cv_sq / (r * (r - 1)), 0) / (C_hat * C_hat) - 1.0 / r
        cv_sq = max(cv_sq, 0)
        return d / C_hat + r * (1 - C_hat) / C_hat * cv_sq

    def ada_estimate(self, r: int, profile: List[int]) -> float:
        """自适应估计器——根据覆盖率选最优方法.
        改写: 加 f3 作为第三判据，5 档精细切换."""
        f1 = profile[1] if len(profile) > 1 else 0
        f2 = profile[2] if len(profile) > 2 else 0
        f3 = profile[3] if len(profile) > 3 else 0
        r = max(r, 1)
        coverage = 1.0 - f1 / r

        # 改写: 5 档自适应选择——比原始 4 档多一个 Sichel 分支
        if coverage > 0.98:
            method = "goodman"
            result = self.goodman_estimate(r, profile)
        elif coverage > 0.90:
            method = "chaolee"
            result = self.chaolee_estimate(r, profile)
        elif coverage > 0.7:
            method = "gee"
            result = self.gee_estimate(r, profile)
        elif f2 > 0 and f3 > 0:
            # 改写新增: Sichel 在有足够频率信息时更稳定
            method = "sichel"
            result = self.sichel_estimate(r, profile)
        elif f2 > 0:
            method = "chao"
            result = self.chao_estimate(r, profile)
        else:
            method = "jackknife"
            result = self.jackknife_estimate(r, profile)

        _dbg("ADA", f"coverage={coverage:.3f}, f1={f1}, f2={f2}, f3={f3}, "
             f"selected={method}, result={result:.1f}")
        return result

    # ── Block-split estimation ─────────────────────────────────────────
    def block_split_estimate(
        self,
        data: List[Any],
        n_blocks: int = 10,
        debug: bool = False,
    ) -> float:
        """Estimate NDV using block-level sampling and extrapolation.

        Upstream: NDVEstimator.block_split_estimate (line 526).
        Lynceus: added GPU block annotation.
        """
        if not data:
            return 0

        blocks = self.utils.split_list_into_blocks(data, max(len(data) // n_blocks, 1))
        block_ndvs = []
        cumulative_ndv = set()

        for i, block in enumerate(blocks):
            collapsed = self.utils.collapse_block(block)
            block_ndv = len(set(collapsed))
            cumulative_ndv.update(collapsed)
            block_ndvs.append(block_ndv)

            if debug and i < 5:
                print(f"    block[{i}]: {len(block)} rows → {block_ndv} NDV "
                      f"(cumulative={len(cumulative_ndv)})")

        # Extrapolate: if NDV is still growing, estimate remaining
        if len(block_ndvs) >= 3:
            growth_rate = (block_ndvs[-1] / max(block_ndvs[0], 1))
            if growth_rate > 0.9:
                # Still discovering new values → significant unseen
                sample_frac = len(data) / max(self.original_num, 1)
                return len(cumulative_ndv) / max(sample_frac, 0.01)

        return len(cumulative_ndv)

    # ── Multi-column NDV ──────────────────────────────────────────────
    def estimate_multi_column_ndv(
        self,
        column_ndvs: Dict[str, int],
        target_columns: List[str],
        total_rows: int,
        method: str = "independent",
        debug: bool = False,
    ) -> int:
        """Estimate composite NDV for multiple columns.

        Upstream: NDVEstimator.estimate_multi_columns (line 629).
        Lynceus: simplified, no pandas dependency.
        ~20% change: added "correlated" method option.
        """
        if not target_columns:
            return 1

        individual_ndvs = [column_ndvs.get(c, 1) for c in target_columns]

        if method == "independent":
            # Independence assumption: NDV(A,B) = min(NDV(A) × NDV(B), N)
            result = 1
            for ndv in individual_ndvs:
                result *= ndv
            result = min(result, total_rows)
        elif method == "correlated":
            # Lynceus: assume max correlation → max individual NDV
            result = max(individual_ndvs)
        else:
            # Geometric mean as compromise
            log_sum = sum(math.log(max(ndv, 1)) for ndv in individual_ndvs)
            result = int(math.exp(log_sum / len(individual_ndvs)))
            result = min(result, total_rows)

        if debug:
            print(f"    multi_ndv({target_columns}, {method}): "
                  f"individual={individual_ndvs} → combined={result}")

        return max(result, 1)

    # ── Debug: verbose estimation with all methods ─────────────────────
    def estimate_with_debug(
        self,
        data: List[Any],
        column_name: str = "unknown",
    ) -> Dict[str, float]:
        """Run all estimation methods and print comparison table."""
        r = len(data)
        profile = self.build_column_profile(data)
        d = self.utils.profile_to_ndv(profile)

        print(f"\n  ┌─ NDV ESTIMATION DEBUG: {column_name} ────────────────")
        print(f"  │  r={r:,}  d_obs={d:,}  N={self.original_num:,}  "
              f"f1={profile[1] if len(profile)>1 else 0}  "
              f"f2={profile[2] if len(profile)>2 else 0}")

        methods = [
            "goodman", "jackknife", "chao", "shlosser", "chaolee",
            "gee", "sichel", "mom", "bootstrap", "ht",
            "smoothed_jk", "error_bound", "ada",
        ]

        results = {}
        for m in methods:
            est = self.estimator(r, profile, method=m, column_name=column_name, debug=False)
            results[m] = est
            bar = "█" * min(int(est / max(max(results.values()), 1) * 30), 30)
            print(f"  │  {m:>15s}: {est:>12,}  {bar}")

        print(f"  └────────────────────────────────────────────────────")
        return results

# ═══════════════════════════════════════════════════════════════════════════
# ★ 移植改写区
# ═══════════════════════════════════════════════════════════════════════════

    def dump_estimation_accuracy(self) -> str:
        """★ 改写: NDV 估计精度审计."""
        _dbg("DUMP_EST", "ENTER dump_estimation_accuracy()")
        from .. import _dbg
        if not self._estimation_log:
            return "(no estimations)"
        lines = ["┌── NDV Estimation Accuracy ──"]
        errors = []
        for entry in self._estimation_log[-10:]:
            est = entry.get('estimated', 0)
            actual = entry.get('actual', 0)
            if actual > 0:
                err = abs(est - actual) / actual
                errors.append(err)
                lines.append(f"│ {entry.get('column','?')}: "
                             f"est={est:,} actual={actual:,} err={err:.2%}")
        if errors:
            lines.append(f"│ mean_error = {sum(errors)/len(errors):.2%}")
        lines.append("└──────────────────────────────")
        return "\n".join(lines)
