"""
lynceus/fp8_stats.py — FP8 statistics quantization.

算法改动:
    1. Fp8Codec.quantize: stochastic rounding 模式
       原版: round-to-nearest-ties-to-even (确定性)
       新版: 新增 stochastic=True 参数, 用确定性种子做概率 rounding,
       使均值偏差从 O(1/2^man) 降到 O(1/n * 1/2^man)
    2. measure_error: Kahan compensated summation
       原版: 直接 sig += o*o, noise += e*e
       新版: 用 Kahan 公式, 大列下数值更稳定
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple

class Fp8Format(Enum):
    E4M3 = "E4M3"
    E5M2 = "E5M2"

_TRAITS = {
    Fp8Format.E4M3: dict(exp=4, man=3, bias=7, has_inf=False, max_normal=448.0),
    Fp8Format.E5M2: dict(exp=5, man=2, bias=15, has_inf=True, max_normal=57344.0),
}
DEFAULT_BLOCK_SIZE = 128


class Fp8Codec:
    def __init__(self, fmt: Fp8Format, stochastic: bool = False):
        t = _TRAITS[fmt]
        self.fmt = fmt
        self.exp_bits = t["exp"]
        self.man_bits = t["man"]
        self.bias = t["bias"]
        self.has_inf = t["has_inf"]
        self.max_normal = t["max_normal"]
        self.sign_shift = self.exp_bits + self.man_bits
        self.exp_mask = (1 << self.exp_bits) - 1
        self.man_mask = (1 << self.man_bits) - 1
        self.max_biased_exp = (1 << self.exp_bits) - 1
        # 改动: stochastic rounding
        self._stochastic = stochastic
        self._sr_counter = 0

    def _sr_rand(self) -> float:
        """确定性伪随机数, 用于 stochastic rounding。
        用简单的 LCG 避免依赖 random 模块。
        """
        self._sr_counter = (self._sr_counter * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
        return (self._sr_counter >> 33) / 0x7FFFFFFF

    def dequantize(self, code: int) -> float:
        sign = (code >> self.sign_shift) & 0x1
        exp = (code >> self.man_bits) & self.exp_mask
        man = code & self.man_mask
        s = -1.0 if sign else 1.0
        if exp == 0:
            if man == 0:
                return s * 0.0
            frac = man / (1 << self.man_bits)
            return s * frac * (2.0 ** (1 - self.bias))
        if self.has_inf and exp == self.max_biased_exp:
            return s * math.inf if man == 0 else math.nan
        if (not self.has_inf) and exp == self.max_biased_exp and man == self.man_mask:
            return math.nan
        frac = 1.0 + man / (1 << self.man_bits)
        return s * frac * (2.0 ** (exp - self.bias))

    def quantize(self, x: float) -> int:
        if math.isnan(x):
            if self.has_inf:
                return (self.max_biased_exp << self.man_bits) | 1
            return (self.max_biased_exp << self.man_bits) | self.man_mask
        sign = 1 if math.copysign(1.0, x) < 0 else 0
        ax = abs(x)
        if ax == 0.0:
            return sign << self.sign_shift
        if ax >= self.max_normal:
            top = self.max_biased_exp - (1 if self.has_inf else 0)
            man = self.man_mask if self.has_inf else (self.man_mask - 1)
            return (sign << self.sign_shift) | (top << self.man_bits) | man
        m, e2 = math.frexp(ax)
        m *= 2.0
        e2 -= 1
        biased = e2 + self.bias
        if biased >= 1:
            return self._round_normal(sign, biased, m)
        return self._round_subnormal(sign, ax)

    def _round_normal(self, sign: int, biased: int, m: float) -> int:
        scale = 1 << self.man_bits
        scaled = (m - 1.0) * scale
        man = math.floor(scaled)
        rem = scaled - man
        # 改动: stochastic rounding
        if self._stochastic:
            if self._sr_rand() < rem:
                man += 1
        else:
            if rem > 0.5 or (rem == 0.5 and (man & 1)):
                man += 1
        if man == scale:
            man = 0
            biased += 1
        top = self.max_biased_exp - (1 if self.has_inf else 0)
        if biased > top or (biased == self.max_biased_exp and not self.has_inf
                            and man > self.man_mask - 1):
            mm = self.man_mask if self.has_inf else (self.man_mask - 1)
            return (sign << self.sign_shift) | (top << self.man_bits) | mm
        return ((sign << self.sign_shift) |
                ((biased & self.exp_mask) << self.man_bits) |
                (man & self.man_mask))

    def _round_subnormal(self, sign: int, ax: float) -> int:
        step = (2.0 ** (1 - self.bias)) / (1 << self.man_bits)
        q = ax / step
        man = math.floor(q)
        rem = q - man
        # 改动: stochastic rounding
        if self._stochastic:
            if self._sr_rand() < rem:
                man += 1
        else:
            if rem > 0.5 or (rem == 0.5 and (man & 1)):
                man += 1
        if man >= (1 << self.man_bits):
            return (sign << self.sign_shift) | (1 << self.man_bits)
        return (sign << self.sign_shift) | (man & self.man_mask)


@dataclass
class QuantError:
    snr_db: float = 0.0
    rel_l2: float = 0.0
    max_abs: float = 0.0
    max_rel: float = 0.0
    cosine: float = 1.0
    has_nonfinite: bool = False
    def acceptable(self, snr_floor_db: float, max_rel_ceil: float = 0.45) -> bool:
        if self.has_nonfinite:
            return False
        return self.snr_db >= snr_floor_db and self.max_rel <= max_rel_ceil


def measure_error(orig: List[float], recon: List[float]) -> QuantError:
    """改动: Kahan compensated summation for sig/noise/dot/no/nr.
    FIX(INV-5): orig==0 用绝对误差兜底; NaN 输入显式标记到 has_nonfinite+计数。
    FIX(INV-6): 零值污染检测 — 原值为0但量化后非零的误差不再被遗漏。
    """
    from ._debug import dbg

    # Kahan accumulators: (sum, compensation)
    sig_s, sig_c = 0.0, 0.0
    noise_s, noise_c = 0.0, 0.0
    dot_s, dot_c = 0.0, 0.0
    no_s, no_c = 0.0, 0.0
    nr_s, nr_c = 0.0, 0.0
    mx = mrel = 0.0
    nonfinite = False
    nonfinite_count = 0       # FIX: 计数而非仅 bool
    zero_polluted_count = 0   # FIX: orig==0 被量化为非零的计数

    amax = 0.0
    for o in orig:
        if not math.isfinite(o):
            nonfinite = True
            nonfinite_count += 1
        else:
            amax = max(amax, abs(o))
    scale = amax if amax > 0.0 else 1.0

    def kahan_add(s: float, c: float, val: float) -> Tuple[float, float]:
        y = val - c
        t = s + y
        c = (t - s) - y
        return t, c

    for o, r in zip(orig, recon):
        if not (math.isfinite(o) and math.isfinite(r)):
            nonfinite = True
            nonfinite_count += 1
            continue
        e = o - r
        sig_s, sig_c = kahan_add(sig_s, sig_c, o * o)
        noise_s, noise_c = kahan_add(noise_s, noise_c, e * e)
        dot_s, dot_c = kahan_add(dot_s, dot_c, o * r)
        no_s, no_c = kahan_add(no_s, no_c, o * o)
        nr_s, nr_c = kahan_add(nr_s, nr_c, r * r)
        mx = max(mx, abs(e))
        # FIX(INV-5): orig==0 时用绝对误差 / scale 做 relative 兜底
        # 这保证了零值被量化成 ±subnormal×scale 时误差不被遗漏
        if abs(o) > 0.0:
            mrel = max(mrel, abs(e) / abs(o))
        else:
            # orig==0: 绝对误差除以全局 scale 做归一化 relative
            mrel = max(mrel, abs(e) / scale)
            if abs(r) > 0.0:
                zero_polluted_count += 1

    # FIX: NaN 输入时在调试输出里明确警告
    if nonfinite_count > 0:
        dbg("FP8·nonfinite_input_WARNING",
            nonfinite_count=nonfinite_count,
            total=len(orig),
            pct=f"{100.0 * nonfinite_count / max(1, len(orig)):.1f}%")
    if zero_polluted_count > 0:
        dbg("FP8·zero_pollution_WARNING",
            zero_polluted=zero_polluted_count,
            msg="orig==0 quantized to nonzero — sparse column may be corrupted")

    q = QuantError(max_abs=mx, max_rel=mrel, has_nonfinite=nonfinite)
    q.rel_l2 = math.sqrt(noise_s / sig_s) if sig_s > 0 else 0.0
    q.snr_db = 10.0 * math.log10(sig_s / noise_s) if noise_s > 0 else math.inf
    q.cosine = dot_s / (math.sqrt(no_s) * math.sqrt(nr_s)) if no_s > 0 and nr_s > 0 else 1.0

    dbg("FP8·measure_error_done",
        max_abs=f"{q.max_abs:.6e}", max_rel=f"{q.max_rel:.6e}",
        snr_db=f"{q.snr_db:.1f}", cosine=f"{q.cosine:.6f}",
        zero_polluted=zero_polluted_count, nonfinite=nonfinite_count)
    return q


@dataclass
class QuantizedColumn:
    codes: List[int] = field(default_factory=list)
    scales: List[float] = field(default_factory=list)
    block_size: int = DEFAULT_BLOCK_SIZE
    n: int = 0
    fmt: Fp8Format = Fp8Format.E4M3


class BlockQuantizer:
    def __init__(self, fmt: Fp8Format = Fp8Format.E4M3,
                 block_size: int = DEFAULT_BLOCK_SIZE,
                 stochastic: bool = False):
        self.codec = Fp8Codec(fmt, stochastic=stochastic)
        self.fmt = fmt
        self.block_size = block_size if block_size > 0 else DEFAULT_BLOCK_SIZE

    def _block_scale(self, x: List[float]) -> float:
        amax = max((abs(v) for v in x), default=0.0)
        if amax == 0.0:
            return 1.0
        return amax / self.codec.max_normal

    def quantize(self, x: List[float]) -> QuantizedColumn:
        n = len(x)
        q = QuantizedColumn(codes=[0]*n, block_size=self.block_size, n=n, fmt=self.fmt)
        nblocks = (n + self.block_size - 1) // self.block_size
        q.scales = [1.0] * nblocks
        for b in range(nblocks):
            start = b * self.block_size
            chunk = x[start:start + self.block_size]
            s = self._block_scale(chunk)
            q.scales[b] = s
            inv = 1.0 / s
            for i, v in enumerate(chunk):
                q.codes[start + i] = self.codec.quantize(v * inv)
        return q

    def dequantize(self, q: QuantizedColumn) -> List[float]:
        out = [0.0] * q.n
        for i in range(q.n):
            b = i // q.block_size
            out[i] = self.codec.dequantize(q.codes[i]) * q.scales[b]
        return out


@dataclass
class ColumnQuantResult:
    fmt: Fp8Format
    quantized: QuantizedColumn
    error: QuantError
    used_fp8: bool
    compression: float


class StatColumnQuantizer:
    """改动: 先试 stochastic rounding, 失败再试确定性, 再失败才 fallback fp32。"""
    def __init__(self, block_size: int = DEFAULT_BLOCK_SIZE,
                 snr_floor_db: float = 28.0, max_rel_ceil: float = 0.5):
        self.block_size = block_size
        self.snr_floor_db = snr_floor_db
        self.max_rel_ceil = max_rel_ceil

    @staticmethod
    def _compression(n: int, nblocks: int) -> float:
        fp32_bytes = 8.0 * n
        fp8_bytes = 1.0 * n + 8.0 * nblocks
        return fp32_bytes / fp8_bytes if fp8_bytes > 0 else 1.0

    def quantize_column(self, col: List[float]) -> ColumnQuantResult:
        from ._debug import checkpoint
        # 改动: 先试 stochastic rounding (均值偏差更小)
        bq_sr = BlockQuantizer(Fp8Format.E4M3, self.block_size, stochastic=True)
        q_sr = bq_sr.quantize(col)
        recon_sr = bq_sr.dequantize(q_sr)
        err_sr = measure_error(col, recon_sr)
        if err_sr.acceptable(self.snr_floor_db, self.max_rel_ceil):
            checkpoint("fp8_accepted", mode="stochastic", snr=err_sr.snr_db, max_rel=err_sr.max_rel)
            return ColumnQuantResult(
                Fp8Format.E4M3, q_sr, err_sr, True,
                self._compression(q_sr.n, len(q_sr.scales)))

        # fallback: 确定性 rounding
        bq_det = BlockQuantizer(Fp8Format.E4M3, self.block_size, stochastic=False)
        q_det = bq_det.quantize(col)
        recon_det = bq_det.dequantize(q_det)
        err_det = measure_error(col, recon_det)
        if err_det.acceptable(self.snr_floor_db, self.max_rel_ceil):
            checkpoint("fp8_accepted", mode="deterministic", snr=err_det.snr_db, max_rel=err_det.max_rel)
            return ColumnQuantResult(
                Fp8Format.E4M3, q_det, err_det, True,
                self._compression(q_det.n, len(q_det.scales)))

        # 都不行 → fp32
        checkpoint("fp8_rejected", snr_sr=err_sr.snr_db, snr_det=err_det.snr_db)
        return ColumnQuantResult(Fp8Format.E4M3, q_det, err_det, False, 1.0)
