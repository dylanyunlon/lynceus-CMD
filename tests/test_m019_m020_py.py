"""
tests/test_m019_m020_py.py — Python-layer validation for M019-M020.

Mirrors the C++ suite (tests/test_m019_m020.cpp) and additionally asserts the
Python codec is bit-identical to the documented FP8 grid. Run:
    python tests/test_m019_m020_py.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lynceus.fp8_stats import (
    Fp8Codec, Fp8Format, BlockQuantizer, StatColumnQuantizer, measure_error,
)


def _approx(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_01_e4m3_known_codes():
    e4 = Fp8Codec(Fp8Format.E4M3)
    assert _approx(e4.dequantize(0x38), 1.0)
    assert _approx(e4.dequantize(0x40), 2.0)
    assert _approx(e4.dequantize(0x7E), 448.0)


def test_02_e4m3_roundtrip_exact():
    e4 = Fp8Codec(Fp8Format.E4M3)
    for v in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 0.25, 0.125, 256.0]:
        assert _approx(e4.dequantize(e4.quantize(v)), v)
        assert _approx(e4.dequantize(e4.quantize(-v)), -v)


def test_03_e4m3_saturates():
    e4 = Fp8Codec(Fp8Format.E4M3)
    r = e4.dequantize(e4.quantize(1e6))
    assert _approx(r, 448.0) and math.isfinite(r)


def test_04_inf_semantics():
    e4 = Fp8Codec(Fp8Format.E4M3)
    e5 = Fp8Codec(Fp8Format.E5M2)
    assert math.isfinite(e4.dequantize(e4.quantize(1e9)))
    assert math.isinf(e5.dequantize(0x7C))


def test_05_e4m3_min_subnormal():
    e4 = Fp8Codec(Fp8Format.E4M3)
    assert _approx(e4.dequantize(0x01), 2.0 ** -9)


def test_06_round_ties_to_even():
    e4 = Fp8Codec(Fp8Format.E4M3)
    assert _approx(e4.dequantize(e4.quantize(1.0625)), 1.0)


def test_07_e5m2_wider_range():
    e4 = Fp8Codec(Fp8Format.E4M3)
    e5 = Fp8Codec(Fp8Format.E5M2)
    assert _approx(e5.dequantize(e5.quantize(1024.0)), 1024.0)
    assert _approx(e4.dequantize(e4.quantize(1024.0)), 448.0)


def test_08_sign_preserved():
    e4 = Fp8Codec(Fp8Format.E4M3)
    assert math.copysign(1.0, e4.dequantize(e4.quantize(-3.0))) < 0
    nz = e4.quantize(-0.0)
    assert (nz >> 7) == 1 and e4.dequantize(nz) == 0.0


def test_09_block_scale():
    bq = BlockQuantizer(Fp8Format.E4M3, 4)
    assert _approx(bq._block_scale([10.0, -200.0, 50.0, 7.0]), 200.0 / 448.0)


def test_10_block_roundtrip_snr():
    blk = [math.sin(i * 0.1) * 1000.0 for i in range(128)]
    bq = BlockQuantizer(Fp8Format.E4M3, 128)
    err = measure_error(blk, bq.dequantize(bq.quantize(blk)))
    assert err.snr_db > 25.0


def test_11_zero_error_inf_snr():
    a = [1.0, 2.0, 3.0]
    e = measure_error(a, a)
    assert math.isinf(e.snr_db) and _approx(e.cosine, 1.0) and e.rel_l2 == 0.0


def test_12_max_rel_detects_crushed_small():
    easy = [100.0 + 0.05 * (i % 64) for i in range(256)]
    hard = [10.0 ** ((i % 64) / 10.0) for i in range(256)]
    bq = BlockQuantizer(Fp8Format.E4M3, 64)
    mr_easy = measure_error(easy, bq.dequantize(bq.quantize(easy))).max_rel
    mr_hard = measure_error(hard, bq.dequantize(bq.quantize(hard))).max_rel
    assert mr_easy < mr_hard and mr_easy < 0.1


def test_13_picks_e4m3():
    col = [0.5 + 0.4 * math.sin(i * 0.05) for i in range(512)]
    r = StatColumnQuantizer(128, 30.0).quantize_column(col)
    assert r.used_fp8 and r.fmt == Fp8Format.E4M3


def test_14_compression_8x():
    col = [1.0] * 4096
    r = StatColumnQuantizer(128, 30.0).quantize_column(col)
    assert r.used_fp8 and r.compression > 7.5


def test_15_keeps_fp32_on_crushed_small():
    col = [4.0e2 if i % 4 == 0 else 1.0e-3 for i in range(128)]
    r = StatColumnQuantizer(128, 30.0).quantize_column(col)
    assert (not r.used_fp8) and _approx(r.compression, 1.0) and r.error.max_rel > 0.5


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
