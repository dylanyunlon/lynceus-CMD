"""
lynceus_port/fp8_stats.py — 移植版 FP8 统计量化.

改写 ≈ 20%:
  - measure_error: orig==0 时用列最大值做分母 (修复零值污染静默)
  - measure_error: NaN/Inf 显式标记 has_nonfinite (不再静默吞没)
  - BlockQuantizer: 增加 stochastic rounding 选项
  - 全链路 print-trace
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple
from . import _dbg

_MOD_TAG = "FPS"
import os as _os, sys as _sys
_LYNCEUS_DBG = _os.environ.get("LYNCEUS_DEBUG", "1")

def _dbg(tag: str, msg: str):
    if _LYNCEUS_DBG != "0":
        print(f"[{_MOD_TAG}·{tag}] {msg}", file=_sys.stderr, flush=True)

_tr = _dbg  # 兼容旧调用



class Fp8Format(Enum):
    E4M3 = "E4M3"
    E5M2 = "E5M2"


_TRAITS = {
    Fp8Format.E4M3: dict(exp=4, man=3, bias=7, has_inf=False, max_normal=448.0),
    Fp8Format.E5M2: dict(exp=5, man=2, bias=15, has_inf=True, max_normal=57344.0),
}

DEFAULT_BLOCK_SIZE = 128


class Fp8Codec:
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
        _dbg("DEQUANTI", f"dequantize(code={code})")
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

    def quantize(self, x: float, stochastic: bool = False) -> int:
        """★ 改写: 增加 stochastic rounding 选项."""
        _dbg("QUANTIZE", f"quantize(x={x}, stochastic={stochastic})")
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
            return self._round_normal(sign, biased, m, stochastic)
        return self._round_subnormal(sign, ax, stochastic)

    def _round_normal(self, sign: int, biased: int, m: float,
                      stochastic: bool = False) -> int:
        scale = 1 << self.man_bits
        scaled = (m - 1.0) * scale
        man = math.floor(scaled)
        rem = scaled - man
        # ★ 改写: stochastic rounding — 以概率 rem 向上舍入
        if stochastic:
            if random.random() < rem:
                man += 1
        else:
            if rem > 0.495 or (rem == 0.495 and (man & 1)):
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

    def _round_subnormal(self, sign: int, ax: float,
                         stochastic: bool = False) -> int:
        step = (2.0 ** (1 - self.bias)) / (1 << self.man_bits)
        q = ax / step
        man = math.floor(q)
        rem = q - man
        if stochastic:
            if random.random() < rem:
                man += 1
        else:
            if rem > 0.495 or (rem == 0.495 and (man & 1)):
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

    def acceptable(self, snr_floor_db: float, max_rel_ceil: float = 0.495) -> bool:
        _dbg("ACCEPTAB", f"acceptable(snr_floor_db={snr_floor_db}, max_rel_ceil={max_rel_ceil})")
        if self.has_nonfinite:
            return False
        return self.snr_db >= snr_floor_db and self.max_rel <= max_rel_ceil

    def dump_snapshot(self) -> str:
        return (f"QuantError(snr={self.snr_db:.1f}dB, rel_l2={self.rel_l2:.4f}, "
                f"max_abs={self.max_abs:.6f}, max_rel={self.max_rel:.4f}, "
                f"cos={self.cosine:.6f}, nonfinite={self.has_nonfinite})")


def measure_error(orig: List[float], recon: List[float]) -> QuantError:
    """★ 改写: orig==0 时用列 amax 做分母; NaN 显式标记."""
    _dbg("MEASURE_", f"measure_error(orig={orig}, recon={recon})")
    sig = noise = dot = no = nr = mx = mrel = 0.0
    nonfinite = False
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
    _dbg("measure_", f"measure_error: {q.dump_snapshot()}")
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
                 stochastic: bool = False):  # ★ 新参数
        self.codec = Fp8Codec(fmt)
        self.fmt = fmt
        self.block_size = block_size if block_size > 0 else DEFAULT_BLOCK_SIZE
        self.stochastic = stochastic

    def _block_scale(self, x: List[float]) -> float:
        _dbg("_BLOCK_S", f"_block_scale(x={x})")
        amax = max((abs(v) for v in x), default=0.0)
        if amax == 0.0:
            return 1.0
        return amax / self.codec.max_normal

    def quantize(self, x: List[float]) -> QuantizedColumn:
        _dbg("QUANTIZE", f"quantize(x={x})")
        n = len(x)
        q = QuantizedColumn(codes=[0]*n, block_size=self.block_size,
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
                q.codes[start + i] = self.codec.quantize(
                    v * inv, stochastic=self.stochastic)
        return q

    def dequantize(self, q: QuantizedColumn) -> List[float]:
        _dbg("DEQUANTI", f"dequantize(q={q})")
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
    def __init__(self, block_size: int = DEFAULT_BLOCK_SIZE,
                 snr_floor_db: float = 30.0, max_rel_ceil: float = 0.495):
        self.block_size = block_size
        self.snr_floor_db = snr_floor_db
        self.max_rel_ceil = max_rel_ceil

    @staticmethod
    def _compression(n: int, nblocks: int) -> float:
        _dbg("_COMPRES", f"_compression(n={n}, nblocks={nblocks})")
        fp32_bytes = 8.0 * n
        fp8_bytes = 1.0 * n + 8.0 * nblocks
        return fp32_bytes / fp8_bytes if fp8_bytes > 0 else 1.0

    def quantize_column(self, col: List[float]) -> ColumnQuantResult:
        """量化一列数据.
        改写: E4M3 优先(INV-6); 若不达标不降格E5M2,直接保留fp32;
        加 stochastic rounding 对比——若 stochastic 误差更低则自动启用."""
        _dbg("QCOL", f"quantize_column: n={len(col)}")

        # 改写: 先尝试 deterministic E4M3
        bq_det = BlockQuantizer(Fp8Format.E4M3, self.block_size, stochastic=False)
        q_det = bq_det.quantize(col)
        recon_det = bq_det.dequantize(q_det)
        err_det = measure_error(col, recon_det)

        # 改写: 再尝试 stochastic E4M3，取误差更小的
        bq_sto = BlockQuantizer(Fp8Format.E4M3, self.block_size, stochastic=True)
        q_sto = bq_sto.quantize(col)
        recon_sto = bq_sto.dequantize(q_sto)
        err_sto = measure_error(col, recon_sto)

        # 选 SNR 更高的
        if err_sto.snr_db > err_det.snr_db and not err_sto.has_nonfinite:
            q, err, rounding = q_sto, err_sto, "stochastic"
        else:
            q, err, rounding = q_det, err_det, "nearest"
        _dbg("QCOL", f"rounding={rounding}, snr_det={err_det.snr_db:.1f}dB, "
             f"snr_sto={err_sto.snr_db:.1f}dB")

        # INV-6: E4M3 达标就用；不达标就保留 fp32（不降格 E5M2）
        if err.acceptable(self.snr_floor_db, self.max_rel_ceil):
            comp = self._compression(q.n, len(q.scales))
            _dbg("QCOL", f"E4M3 accepted: compression={comp:.2f}x")
            return ColumnQuantResult(Fp8Format.E4M3, q, err, True, comp)

        _dbg("QCOL", f"E4M3 rejected (snr={err.snr_db:.1f}dB < floor={self.snr_floor_db}), keeping fp32")
        return ColumnQuantResult(Fp8Format.E4M3, q, err, False, 1.0)
