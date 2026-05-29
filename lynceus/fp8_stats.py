"""
lynceus/fp8_stats.py — M019-M020: FP8 statistics quantization (Python layer).

A faithful, dependency-free port of lynceus/core/fp8_stats_quant.cuh so the
cost model and benchmark can decide, per statistics column, whether to store it
in block-scaled FP8 (4x smaller HBM footprint) without building the C++ core.

The bit-level codec, block scaling, and acceptance policy match the .cuh
exactly:
  * E4M3 (bias 7, max 448, no inf, single NaN) and E5M2 (bias 15, max 57344,
    IEEE-like) encode/decode with round-to-nearest-ties-to-even, subnormals,
    and saturating overflow.
  * Block-wise fp32 scale = block_absmax / FP8_MAX (DeepSeek act_quant pattern).
  * Adoption requires BOTH an SNR floor and a per-element max-relative-error
    ceiling — because SNR is energy-weighted and can hide crushed small values.
  * Measured fact respected: under block scaling, E4M3 dominates E5M2 on SNR,
    so the accuracy fallback from E4M3 is fp32, never E5M2.

Architecture references:
    DeepSeek-V3 act_quant_kernel / weight_dequant   (block-wise FP8 scaling)
    NVIDIA "FP8 Formats for Deep Learning" (2209.05433)
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
    """Bit-exact scalar quantize/dequantize for one FP8 format."""

    def __init__(self, fmt: Fp8Format):
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
        m, e2 = math.frexp(ax)      # ax = m * 2^e2, m in [0.5, 1)
        m *= 2.0
        e2 -= 1                     # m in [1, 2)
        biased = e2 + self.bias
        if biased >= 1:
            return self._round_normal(sign, biased, m)
        return self._round_subnormal(sign, ax)

    def _round_normal(self, sign: int, biased: int, m: float) -> int:
        scale = 1 << self.man_bits
        scaled = (m - 1.0) * scale
        man = math.floor(scaled)
        rem = scaled - man
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
        return ((sign << self.sign_shift)
                | ((biased & self.exp_mask) << self.man_bits)
                | (man & self.man_mask))

    def _round_subnormal(self, sign: int, ax: float) -> int:
        step = (2.0 ** (1 - self.bias)) / (1 << self.man_bits)
        q = ax / step
        man = math.floor(q)
        rem = q - man
        if rem > 0.5 or (rem == 0.5 and (man & 1)):
            man += 1
        if man >= (1 << self.man_bits):
            return (sign << self.sign_shift) | (1 << self.man_bits)
        return (sign << self.sign_shift) | (man & self.man_mask)


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

@dataclass
class QuantError:
    snr_db: float = 0.0
    rel_l2: float = 0.0
    max_abs: float = 0.0
    max_rel: float = 0.0
    cosine: float = 1.0
    has_nonfinite: bool = False   # True if orig/recon contained NaN or Inf

    def acceptable(self, snr_floor_db: float, max_rel_ceil: float = 0.5) -> bool:
        # Non-finite input is never silently accepted: the caller must handle
        # NaN/Inf explicitly rather than have it swallowed by NaN comparisons.
        if self.has_nonfinite:
            return False
        return self.snr_db >= snr_floor_db and self.max_rel <= max_rel_ceil


def measure_error(orig: List[float], recon: List[float]) -> QuantError:
    sig = noise = dot = no = nr = mx = mrel = 0.0
    nonfinite = False
    # Reference scale for relative error of zero-valued originals: when the
    # true value is 0 but the reconstruction is not, a ratio is undefined, so
    # we normalise that element's error by the column's max magnitude instead.
    # This catches FP8 polluting zeros (common in sparse histogram columns),
    # which the old `if abs(o) > 0` guard silently ignored.
    amax = 0.0
    for o in orig:
        if not math.isfinite(o):
            nonfinite = True
        else:
            amax = max(amax, abs(o))
    scale = amax if amax > 0.0 else 1.0

    for o, r in zip(orig, recon):
        if not (math.isfinite(o) and math.isfinite(r)):
            nonfinite = True
            continue
        e = o - r
        sig += o * o
        noise += e * e
        dot += o * r
        no += o * o
        nr += r * r
        mx = max(mx, abs(e))
        denom = abs(o) if abs(o) > 0.0 else scale
        mrel = max(mrel, abs(e) / denom)

    q = QuantError(max_abs=mx, max_rel=mrel, has_nonfinite=nonfinite)
    q.rel_l2 = math.sqrt(noise / sig) if sig > 0 else 0.0
    q.snr_db = 10.0 * math.log10(sig / noise) if noise > 0 else math.inf
    q.cosine = dot / (math.sqrt(no) * math.sqrt(nr)) if no > 0 and nr > 0 else 1.0
    return q


# ---------------------------------------------------------------------------
# Block quantizer
# ---------------------------------------------------------------------------

@dataclass
class QuantizedColumn:
    codes: List[int] = field(default_factory=list)
    scales: List[float] = field(default_factory=list)
    block_size: int = DEFAULT_BLOCK_SIZE
    n: int = 0
    fmt: Fp8Format = Fp8Format.E4M3


class BlockQuantizer:
    def __init__(self, fmt: Fp8Format = Fp8Format.E4M3,
                 block_size: int = DEFAULT_BLOCK_SIZE):
        self.codec = Fp8Codec(fmt)
        self.fmt = fmt
        self.block_size = block_size if block_size > 0 else DEFAULT_BLOCK_SIZE

    def _block_scale(self, x: List[float]) -> float:
        amax = max((abs(v) for v in x), default=0.0)
        if amax == 0.0:
            return 1.0
        return amax / self.codec.max_normal

    def quantize(self, x: List[float]) -> QuantizedColumn:
        n = len(x)
        q = QuantizedColumn(codes=[0] * n, block_size=self.block_size,
                            n=n, fmt=self.fmt)
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


# ---------------------------------------------------------------------------
# Column-level policy
# ---------------------------------------------------------------------------

@dataclass
class ColumnQuantResult:
    fmt: Fp8Format
    quantized: QuantizedColumn
    error: QuantError
    used_fp8: bool
    compression: float


class StatColumnQuantizer:
    """E4M3-first, fp32 fallback, gated on SNR floor AND max-relative ceiling."""

    def __init__(self, block_size: int = DEFAULT_BLOCK_SIZE,
                 snr_floor_db: float = 30.0, max_rel_ceil: float = 0.5):
        self.block_size = block_size
        self.snr_floor_db = snr_floor_db
        self.max_rel_ceil = max_rel_ceil

    @staticmethod
    def _compression(n: int, nblocks: int) -> float:
        fp32_bytes = 8.0 * n
        fp8_bytes = 1.0 * n + 8.0 * nblocks
        return fp32_bytes / fp8_bytes if fp8_bytes > 0 else 1.0

    def quantize_column(self, col: List[float]) -> ColumnQuantResult:
        bq = BlockQuantizer(Fp8Format.E4M3, self.block_size)
        q = bq.quantize(col)
        recon = bq.dequantize(q)
        err = measure_error(col, recon)
        if err.acceptable(self.snr_floor_db, self.max_rel_ceil):
            return ColumnQuantResult(
                Fp8Format.E4M3, q, err, True,
                self._compression(q.n, len(q.scales)))
        # E4M3 missed the floor: E5M2 would do worse under block scaling, so
        # keep fp32.
        return ColumnQuantResult(Fp8Format.E4M3, q, err, False, 1.0)
