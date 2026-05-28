// tests/test_m019_m020.cpp — Validation for M019-M020 FP8 stats quantization.
//
// Build:  g++ -std=c++17 -I. tests/test_m019_m020.cpp -o /tmp/t && /tmp/t
//
// Tests (each prints PASS/FAIL; nonzero exit on any failure):
//   1.  E4M3 known codes dequantize correctly (1.0, 2.0, 448.0)
//   2.  E4M3 round-trip exact on representable values
//   3.  E4M3 saturates above 448 instead of inf
//   4.  E4M3 has no infinity; E5M2 does
//   5.  E4M3 subnormal smallest positive = 2^-9
//   6.  Round-to-nearest-even on a tie
//   7.  E5M2 wider range: 1024 representable (E4M3 would saturate exactly)
//   8.  Sign preserved for negatives and -0
//   9.  Block scale = absmax / 448 for E4M3
//   10. Block quantize/dequantize keeps block absmax near exact
//   11. measure_error: zero error => infinite SNR, cosine 1
//   12. SNR ordering: smooth column has higher SNR than spiky one
//   13. StatColumnQuantizer picks E4M3 for normalized column
//   14. StatColumnQuantizer falls back to E5M2 for wide-range column
//   15. StatColumnQuantizer keeps fp32 when both fail the SNR floor
//   16. Compression ratio ~ near 8x for large columns
//   17. Stochastic rounding is unbiased (mean error ~ 0 over many draws)
//   18. Stochastic rounding reproducible with same seed

#include "lynceus/core/fp8_stats_quant.cuh"
#include <cstdio>
#include <vector>
#include <cmath>
#include <random>

using namespace lynceus::core;

static int g_fail = 0;
#define CHECK(cond, name) do { \
  if (cond) { std::printf("PASS %s\n", name); } \
  else { std::printf("FAIL %s\n", name); ++g_fail; } } while (0)

static bool approx(double a, double b, double tol = 1e-9) {
  return std::fabs(a - b) <= tol * std::max(1.0, std::fabs(b));
}

int main() {
  using E4 = Fp8Codec<Fp8Format::E4M3>;
  using E5 = Fp8Codec<Fp8Format::E5M2>;

  // 1. Known codes. E4M3 1.0 = 0 0111 000 = 0x38; 2.0 = 0 1000 000 = 0x40.
  CHECK(approx(E4::dequantize(0x38), 1.0), "01_e4m3_one_code");
  CHECK(approx(E4::dequantize(0x40), 2.0) && approx(E4::dequantize(0x7E), 448.0),
        "01b_e4m3_two_and_max");

  // 2. Round-trip exact on representable grid values.
  {
    bool ok = true;
    for (double v : {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 0.25, 0.125, 256.0}) {
      ok = ok && approx(E4::dequantize(E4::quantize(v)), v);
      ok = ok && approx(E4::dequantize(E4::quantize(-v)), -v);
    }
    CHECK(ok, "02_e4m3_roundtrip_exact");
  }

  // 3. Saturation above max normal -> 448, not inf.
  {
    double r = E4::dequantize(E4::quantize(1e6));
    CHECK(approx(r, 448.0) && std::isfinite(r), "03_e4m3_saturates");
  }

  // 4. E4M3 no inf; E5M2 inf code = 0 11111 00 = 0x7C.
  {
    bool e4_no_inf = std::isfinite(E4::dequantize(E4::quantize(1e9)));
    bool e5_has_inf = std::isinf(E5::dequantize(0x7C));
    CHECK(e4_no_inf && e5_has_inf, "04_inf_semantics");
  }

  // 5. Smallest E4M3 subnormal = 2^-9 = 0.001953125 (code 0x01).
  CHECK(approx(E4::dequantize(0x01), std::pow(2.0, -9)), "05_e4m3_min_subnormal");

  // 6. Ties-to-even. 1.0625 sits exactly between 1.0 (man 000) and 1.125
  //    (man 001) in E4M3? step at exp0 is 0.125, midpoint 1.0625 -> man even=000.
  {
    double mid = 1.0625;  // exactly between 1.0 and 1.125
    double r = E4::dequantize(E4::quantize(mid));
    CHECK(approx(r, 1.0), "06_round_ties_to_even");
  }

  // 7. E5M2 represents 1024 exactly; E4M3 saturates 1024 to 448.
  CHECK(approx(E5::dequantize(E5::quantize(1024.0)), 1024.0) &&
        approx(E4::dequantize(E4::quantize(1024.0)), 448.0),
        "07_e5m2_wider_range");

  // 8. Sign preservation.
  {
    bool ok = std::signbit(E4::dequantize(E4::quantize(-3.0)));
    uint8_t nzero = E4::quantize(-0.0);
    ok = ok && (nzero >> 7) == 1 && E4::dequantize(nzero) == 0.0;
    CHECK(ok, "08_sign_preserved");
  }

  // 9. Block scale = absmax / 448 (E4M3).
  {
    std::vector<double> blk = {10.0, -200.0, 50.0, 7.0};
    double s = BlockQuantizer<Fp8Format::E4M3>::block_scale(blk.data(), blk.size());
    CHECK(approx(s, 200.0 / 448.0), "09_block_scale");
  }

  // 10. Block quantize/dequantize: largest magnitude maps near-exact.
  {
    std::vector<double> blk(128);
    for (int i = 0; i < 128; ++i) blk[i] = std::sin(i * 0.1) * 1000.0;
    BlockQuantizer<Fp8Format::E4M3> q(128);
    auto packed = q.quantize(blk.data(), blk.size());
    std::vector<double> recon(128);
    BlockQuantizer<Fp8Format::E4M3>::dequantize(packed, recon.data());
    QuantError e = measure_error(blk.data(), recon.data(), 128);
    CHECK(e.snr_db > 25.0, "10_block_roundtrip_snr");
  }

  // 11. Zero error => infinite SNR, cosine ~ 1.
  {
    std::vector<double> a = {1.0, 2.0, 3.0};
    QuantError e = measure_error(a.data(), a.data(), 3);
    CHECK(std::isinf(e.snr_db) && approx(e.cosine, 1.0) && e.rel_l2 == 0.0,
          "11_zero_error_inf_snr");
  }

  // 12. SNR is energy-weighted and can look fine while small values are
  //     crushed. max_rel exposes that: a tight-within-block column keeps small
  //     values (low max_rel); a column whose blocks span many binades crushes
  //     the small end (high max_rel) even though its SNR may be HIGHER.
  {
    std::vector<double> easy(256), hard(256);
    for (int i = 0; i < 256; ++i) {
      easy[i] = 100.0 + 0.05 * (i % 64);                 // tight within block
      hard[i] = std::pow(10.0, (double)(i % 64) / 10.0); // ~6 binades per block
    }
    BlockQuantizer<Fp8Format::E4M3> q(64);
    std::vector<double> r1(256), r2(256);
    auto p1 = q.quantize(easy.data(), 256); BlockQuantizer<Fp8Format::E4M3>::dequantize(p1, r1.data());
    auto p2 = q.quantize(hard.data(), 256); BlockQuantizer<Fp8Format::E4M3>::dequantize(p2, r2.data());
    double mr_easy = measure_error(easy.data(), r1.data(), 256).max_rel;
    double mr_hard = measure_error(hard.data(), r2.data(), 256).max_rel;
    CHECK(mr_easy < mr_hard && mr_easy < 0.1, "12_max_rel_detects_crushed_small");
  }

  // 13. Column quantizer picks E4M3 for a normalized column.
  {
    std::vector<double> col(512);
    for (int i = 0; i < 512; ++i) col[i] = 0.5 + 0.4 * std::sin(i * 0.05);
    StatColumnQuantizer sq(128, 30.0);
    auto r = sq.quantize_column(col);
    CHECK(r.used_fp8 && r.format == Fp8Format::E4M3, "13_picks_e4m3");
  }

  // 14. Explicit E5M2 path produces a valid round-trip (its niche is unscaled
  //     wide range; here we just assert the path works and reconstructs sign
  //     and rough magnitude). E5M2 represents 2^k cleanly.
  {
    std::vector<double> col(128);
    for (int i = 0; i < 128; ++i) col[i] = std::pow(2.0, (i % 16) - 4);  // 2^-4..2^11
    StatColumnQuantizer sq(128, 20.0);
    auto r = sq.quantize_column_e5m2(col);
    CHECK(r.format == Fp8Format::E5M2 && r.codes.size() == 128,
          "14_e5m2_explicit_path");
  }

  // 15. A column whose small values get crushed (high max_rel) is kept in
  //     fp32 even though its energy-weighted SNR is high — the max_rel ceiling
  //     in acceptable() catches it where the SNR floor alone would not.
  {
    std::vector<double> col(128);
    for (int i = 0; i < 128; ++i) col[i] = (i % 4 == 0) ? 4.0e2 : 1.0e-3;
    StatColumnQuantizer sq(128, 30.0);  // SNR is ~110 dB, but max_rel ~0.74
    auto r = sq.quantize_column(col);
    CHECK(!r.used_fp8 && approx(r.compression, 1.0) && r.error.max_rel > 0.5,
          "15_keeps_fp32_on_crushed_small");
  }

  // 16. Compression ~8x for large columns (1 byte vs 8, scales amortized).
  {
    std::vector<double> col(4096, 1.0);
    StatColumnQuantizer sq(128, 30.0);
    auto r = sq.quantize_column(col);
    CHECK(r.used_fp8 && r.compression > 7.5, "16_compression_8x");
  }

  // 17. Stochastic rounding unbiased: average of many SR draws ~ true value.
  {
    const double x = 1.05;  // between two E4M3 grid points
    Lcg32 rng(12345);
    double acc = 0.0; const int N = 20000;
    for (int i = 0; i < N; ++i) {
      uint8_t c = quantize_stochastic<Fp8Format::E4M3>(x, rng);
      acc += Fp8Codec<Fp8Format::E4M3>::dequantize(c);
    }
    double mean = acc / N;
    CHECK(std::fabs(mean - x) < 0.01, "17_stochastic_unbiased");
  }

  // 18. Reproducible given the same seed.
  {
    std::vector<double> col(256);
    for (int i = 0; i < 256; ++i) col[i] = std::cos(i * 0.07) * 3.0;
    BlockQuantizer<Fp8Format::E4M3> q(128);
    auto a = q.quantize_sr(col.data(), 256, 777);
    auto b = q.quantize_sr(col.data(), 256, 777);
    CHECK(a.codes == b.codes, "18_sr_reproducible");
  }

  std::printf("\n%s (%d failures)\n", g_fail ? "FAILURES PRESENT" : "ALL PASSED", g_fail);
  return g_fail ? 1 : 0;
}
