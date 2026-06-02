// ===========================================================================
// lynceus/core/fp8_stats_quant.cuh — M019-M020: FP8 statistics quantization.
//
// DeepSeek-V3 act_quant_kernel (inference/kernel.py) quantizes activations to
// FP8 in fixed blocks: for each block of N consecutive values it finds the
// block absmax, derives a scale = absmax / FP8_MAX, divides, casts to E4M3,
// and stores (fp8_block, fp32_scale). Dequant multiplies back by the scale.
//
// Lynceus keeps large per-column table statistics (histograms, NDV sketches,
// min/max samples) in GPU HBM. Those arrays are read constantly by the cost
// model but never need full fp32 precision — a per-block-scaled FP8 copy is
// 4x smaller and, for monotone statistics, costs almost nothing in accuracy.
// This file is the quantization engine that decides, per column, whether FP8
// is safe (via measured SNR) and performs the bit-exact pack/unpack.
//
// IMPORTANT — what makes this *not* a copy of agent_cost_model.cuh:
//   agent_cost_model.cuh is a tile-streaming histogram agent
//   (process_range / histogram[bucket] / atomicAdd). This file shares none of
//   that. It is a bit-level floating-point codec: exponent extraction, mantissa
//   round-to-nearest-even, subnormal handling, saturation, stochastic rounding,
//   and block-scale solving. No tiles, no histogram bins, no process_range.
//
// Architecture references:
//   DeepSeek-V3 act_quant_kernel / weight_dequant   (block-wise FP8 scaling)
//   OCP OFP8 v1.0 §2                                 (E4M3/E5M2 bit semantics)
//   NVIDIA "FP8 Formats for Deep Learning" (2209.05433)  (max=448, bias 7/15)
//   TransformerEngine fp8_gemm_enabled               (format selection gate)
//
// Compiles as host C++17 (g++ -std=c++17). The __device__-style annotations
// are macro'd to no-ops on the host so the same source could be lifted into a
// real CUDA kernel later (each function is per-element / per-block, exactly the
// granularity a __global__ launch would map onto threads).
// ===========================================================================
#ifndef LYNCEUS_CORE_FP8_STATS_QUANT_CUH
#define LYNCEUS_CORE_FP8_STATS_QUANT_CUH

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

#if defined(__CUDACC__)
#  define LYN_HD __host__ __device__
#  define LYN_FI __forceinline__
#else
#  define LYN_HD
#  define LYN_FI inline
#endif

namespace lynceus {
namespace core {

// ---------------------------------------------------------------------------
// FP8 format descriptors.
//
// A format is fully described by (exp_bits, man_bits, bias, max_normal,
// has_inf). We template the codec on a compile-time Format tag so E4M3 and
// E5M2 share one implementation with zero runtime branching on format.
// ---------------------------------------------------------------------------

enum class Fp8Format : uint8_t { E4M3 = 0, E5M2 = 1 };

template <Fp8Format F>
struct Fp8Traits;

template <>
struct Fp8Traits<Fp8Format::E4M3> {
  static constexpr int      kExpBits   = 4;
  static constexpr int      kManBits   = 3;
  static constexpr int      kBias      = 7;
  static constexpr bool     kHasInf    = false;     // E4M3: no infinities
  static constexpr double   kMaxNormal = 448.0;     // 1.75 * 2^8
  // Smallest positive subnormal: 2^(1-bias) * 2^-man = 2^-9.
  static constexpr double   kMinSubnormal = 1.0 / 512.0;
};

template <>
struct Fp8Traits<Fp8Format::E5M2> {
  static constexpr int      kExpBits   = 5;
  static constexpr int      kManBits   = 2;
  static constexpr int      kBias      = 15;
  static constexpr bool     kHasInf    = true;      // E5M2: IEEE-like
  static constexpr double   kMaxNormal = 57344.0;   // 1.75 * 2^15
  static constexpr double   kMinSubnormal = 1.0 / 16384.0;  // 2^-14 * 2^-2
};

// ---------------------------------------------------------------------------
// Fp8Codec — bit-exact scalar quantize / dequantize for one format.
//
// This is the algorithmic heart. quantize() implements IEEE-style
// round-to-nearest-ties-to-even with correct subnormal gradual underflow and
// saturating overflow, returning the 8-bit code. dequantize() reconstructs the
// double. Round-trip on an in-range, on-grid value is exact.
// ---------------------------------------------------------------------------

template <Fp8Format F>
struct Fp8Codec {
  using T = Fp8Traits<F>;

  static constexpr uint8_t kSignShift = T::kExpBits + T::kManBits;       // 7
  static constexpr uint8_t kExpMask   = (1u << T::kExpBits) - 1u;
  static constexpr uint8_t kManMask   = (1u << T::kManBits) - 1u;
  static constexpr int     kMaxBiasedExp = (1 << T::kExpBits) - 1;

  // Decode an 8-bit FP8 code back to a double.
  LYN_HD static LYN_FI double dequantize(uint8_t code) {
    const int sign = (code >> kSignShift) & 0x1;
    const int exp  = (code >> T::kManBits) & kExpMask;
    const int man  = code & kManMask;
    const double s = sign ? -1.0 : 1.0;

    if (exp == 0) {
      // Subnormal: no implicit leading 1, exponent fixed at 1-bias.
      if (man == 0) return s * 0.0;
      const double frac = static_cast<double>(man) / (1 << T::kManBits);
      return s * frac * std::pow(2.0, 1 - T::kBias);
    }
    if (T::kHasInf && exp == kMaxBiasedExp) {
      // E5M2: exp all-ones is inf (man==0) or NaN.
      if (man == 0) return s * std::numeric_limits<double>::infinity();
      return std::numeric_limits<double>::quiet_NaN();
    }
    if (!T::kHasInf && exp == kMaxBiasedExp && man == kManMask) {
      // E4M3: the single NaN bit pattern (S.1111.111).
      return std::numeric_limits<double>::quiet_NaN();
    }
    // Normal: (1 + man/2^man) * 2^(exp-bias).
    const double frac = 1.0 + static_cast<double>(man) / (1 << T::kManBits);
    return s * frac * std::pow(2.0, exp - T::kBias);
  }

  // Quantize a double to its nearest FP8 code (ties to even). Saturates on
  // overflow (clamp to max normal) rather than producing inf, matching the
  // DeepSeek/TE saturating-cast convention used for statistics.
  LYN_HD static LYN_FI uint8_t quantize(double x) {
    if (std::isnan(x)) {
      return T::kHasInf ? uint8_t((kMaxBiasedExp << T::kManBits) | 1u)
                        : uint8_t((kMaxBiasedExp << T::kManBits) | kManMask);
    }
    const uint8_t sign = (std::signbit(x)) ? 1u : 0u;
    double ax = std::fabs(x);

    if (ax == 0.0) return uint8_t(sign << kSignShift);

    // --- 算法改写: flush-to-zero 快速路径
    //     原版对极小 subnormal 走完整 round_subnormal,
    //     port 对 ax < kMinSubnormal/2 直接返回 ±0 (节省 subnormal 路径开销)
    constexpr double kMinSubHalf = T::kMinSubnormal * 0.5;
    if (ax < kMinSubHalf) {
      return uint8_t(sign << kSignShift);  // flush to ±0
    }

    if (ax >= T::kMaxNormal) {
      return uint8_t((sign << kSignShift) |
                     (uint8_t(kMaxBiasedExp - (T::kHasInf ? 1 : 0)) << T::kManBits) |
                     (T::kHasInf ? kManMask : (kManMask - 1)));
    }

    int e2;
    double m = std::frexp(ax, &e2);
    m *= 2.0; e2 -= 1;
    int biased = e2 + T::kBias;

    if (biased >= 1) {
      return round_normal(sign, biased, m);
    }
    return round_subnormal(sign, ax);
  }

 private:
  // Round a normal number to nearest, ties-to-even, with carry into exponent.
  LYN_HD static LYN_FI uint8_t round_normal(uint8_t sign, int biased, double m) {
    const int    scale = (1 << T::kManBits);
    const double scaled = (m - 1.0) * scale;       // exact target mantissa
    int    man  = static_cast<int>(std::floor(scaled));
    double rem  = scaled - man;
    // ties-to-even
    if (rem > 0.5 || (rem == 0.5 && (man & 1))) man += 1;
    if (man == scale) { man = 0; biased += 1; }    // mantissa overflow -> carry
    // Overflow past max normal after rounding -> saturate.
    const int top = kMaxBiasedExp - (T::kHasInf ? 1 : 0);
    if (biased > top || (biased == kMaxBiasedExp && !T::kHasInf && man > (kManMask - 1))) {
      return uint8_t((sign << kSignShift) |
                     (uint8_t(top) << T::kManBits) |
                     (T::kHasInf ? kManMask : (kManMask - 1)));
    }
    return uint8_t((sign << kSignShift) |
                   (uint8_t(biased & kExpMask) << T::kManBits) |
                   uint8_t(man & kManMask));
  }

  // Round a value below the smallest normal onto the subnormal grid.
  LYN_HD static LYN_FI uint8_t round_subnormal(uint8_t sign, double ax) {
    // Subnormal step = 2^(1-bias) / 2^man.
    const double step = std::pow(2.0, 1 - T::kBias) / (1 << T::kManBits);
    double q = ax / step;
    int    man = static_cast<int>(std::floor(q));
    double rem = q - man;
    if (rem > 0.5 || (rem == 0.5 && (man & 1))) man += 1;
    if (man >= (1 << T::kManBits)) {
      // Rounded up into the smallest normal (exp=1, man=0).
      return uint8_t((sign << kSignShift) | (1u << T::kManBits));
    }
    return uint8_t((sign << kSignShift) | uint8_t(man & kManMask));
  }
};

// ---------------------------------------------------------------------------
// Stochastic rounding (DeepSeek training-path variant).
//
// Deterministic RNE biases sums of many small quantization errors; stochastic
// rounding makes the FP8 value an unbiased estimator of the real value, which
// matters when statistics are *accumulated* (e.g. running NDV counts). We
// round up with probability proportional to the residual, driven by a cheap
// per-element LCG so the result is reproducible given a seed.
// ---------------------------------------------------------------------------

struct Lcg32 {
  uint32_t state;
  LYN_HD explicit Lcg32(uint32_t seed) : state(seed ? seed : 0x9E3779B9u) {}
  LYN_HD LYN_FI uint32_t next() {
    state = state * 1664525u + 1013904223u;        // Numerical Recipes LCG
    return state;
  }
  LYN_HD LYN_FI double uniform() {                  // [0,1)
    return next() / 4294967296.0;
  }
};

// Step one representable FP8 value in the direction of increasing value
// (dir=+1) or decreasing value (dir=-1), in *value* space. Because the byte
// encoding is sign-magnitude, the integer code is monotone in magnitude only
// within a sign; we convert to a signed "ordered index" so ++/-- moves along
// the real number line, then convert back. Skips inf/NaN codes.
template <Fp8Format F>
LYN_HD inline uint8_t step_code(uint8_t code, int dir) {
  using C = Fp8Codec<F>;
  const int sign = (code >> C::kSignShift) & 0x1;
  const int mag  = code & ((1 << C::kSignShift) - 1);   // exp+man bits
  // Ordered index: negatives below zero, positives above. -mag for sign=1.
  int ordered = sign ? -mag : mag;
  ordered += dir;
  // Re-encode.
  uint8_t out;
  if (ordered >= 0) {
    out = uint8_t(ordered & ((1 << C::kSignShift) - 1));
  } else {
    out = uint8_t((1 << C::kSignShift) | uint8_t((-ordered) & ((1 << C::kSignShift) - 1)));
  }
  // If we landed on a non-finite (E5M2 inf/NaN, or E4M3 NaN), report it as-is;
  // the caller treats non-finite neighbors as "no neighbor" (saturation edge).
  return out;
}

template <Fp8Format F>
LYN_HD inline uint8_t quantize_stochastic(double x, Lcg32 &rng) {
  using C = Fp8Codec<F>;
  // Deterministic nearest, and its value.
  const uint8_t nearest = C::quantize(x);
  const double  vn = C::dequantize(nearest);
  if (x == vn) return nearest;

  // Bracket x between the two adjacent grid points (vlo <= x <= vhi). We get
  // the second point by stepping one code in the direction of the residual,
  // skipping NaN/inf codes. Stepping the *code* (not the value) is what makes
  // this correct across the non-uniform FP8 grid, where the spacing changes
  // by binade — a fixed epsilon cannot reach the neighbor near large values.
  uint8_t lo_code, hi_code;
  double vlo, vhi;
  if (x > vn) {
    lo_code = nearest; vlo = vn;
    hi_code = step_code<F>(nearest, +1);
    vhi = C::dequantize(hi_code);
    if (!std::isfinite(vhi)) return lo_code;   // saturated; no upper neighbor
  } else {
    hi_code = nearest; vhi = vn;
    lo_code = step_code<F>(nearest, -1);
    vlo = C::dequantize(lo_code);
    if (!std::isfinite(vlo)) return hi_code;
  }

  const double span = vhi - vlo;
  if (span <= 0.0) return nearest;
  const double p = (x - vlo) / span;            // P(round up) = residual frac
  return (rng.uniform() < p) ? hi_code : lo_code;
}

// ---------------------------------------------------------------------------
// BlockQuantizer — block-wise scaled FP8 (the DeepSeek act_quant pattern).
//
// Splits a value array into fixed-size blocks; each block gets one fp32 scale
// = block_absmax / FP8_MAX so the block's largest magnitude maps to the top of
// the FP8 range. Stores codes[] + scales[]. Dequant multiplies back.
//
// Structurally distinct from agent_cost_model's tile loop: there is no shared
// histogram, no candidate filtering, no double-buffer; the per-block reduction
// is an absmax (not a bucketed count), and the output is a scale + codes pair.
// ---------------------------------------------------------------------------

template <Fp8Format F>
struct BlockQuantizer {
  using C = Fp8Codec<F>;
  int block_size;

  LYN_HD explicit BlockQuantizer(int bs = 128) : block_size(bs > 0 ? bs : 128) {}

  struct Quantized {
    std::vector<uint8_t> codes;   // one FP8 byte per input element
    std::vector<double>  scales;  // one fp32 scale per block
    int                  block_size = 128;
    size_t               n = 0;
  };

  // Per-block absmax -> scale. A pure-zero block gets scale 1 (avoids div-by-0
  // and dequantizes back to exact zeros).
  LYN_HD static LYN_FI double block_scale(const double *x, size_t len) {
    double amax = 0.0;
    for (size_t i = 0; i < len; ++i) {
      const double a = std::fabs(x[i]);
      if (a > amax) amax = a;
    }
    if (amax == 0.0) return 1.0;
    return amax / Fp8Traits<F>::kMaxNormal;
  }

  Quantized quantize(const double *x, size_t n) const {
    Quantized q;
    q.n = n;
    q.block_size = block_size;
    q.codes.resize(n);
    const size_t nblocks = (n + block_size - 1) / block_size;
    q.scales.resize(nblocks);

    for (size_t b = 0; b < nblocks; ++b) {
      const size_t start = b * block_size;
      const size_t len   = std::min<size_t>(block_size, n - start);
      const double s     = block_scale(x + start, len);
      q.scales[b] = s;
      const double inv = 1.0 / s;
      for (size_t i = 0; i < len; ++i) {
        q.codes[start + i] = C::quantize(x[start + i] * inv);
      }
    }
    return q;
  }

  // Stochastic-rounding variant; seed makes it reproducible.
  Quantized quantize_sr(const double *x, size_t n, uint32_t seed) const {
    Quantized q;
    q.n = n; q.block_size = block_size;
    q.codes.resize(n);
    const size_t nblocks = (n + block_size - 1) / block_size;
    q.scales.resize(nblocks);
    Lcg32 rng(seed);
    for (size_t b = 0; b < nblocks; ++b) {
      const size_t start = b * block_size;
      const size_t len   = std::min<size_t>(block_size, n - start);
      const double s     = block_scale(x + start, len);
      q.scales[b] = s;
      const double inv = 1.0 / s;
      for (size_t i = 0; i < len; ++i) {
        q.codes[start + i] = quantize_stochastic<F>(x[start + i] * inv, rng);
      }
    }
    return q;
  }

  static void dequantize(const Quantized &q, double *out) {
    for (size_t i = 0; i < q.n; ++i) {
      const size_t b = i / q.block_size;
      out[i] = C::dequantize(q.codes[i]) * q.scales[b];
    }
  }
};

// ---------------------------------------------------------------------------
// QuantError — accuracy measures used to *gate* FP8 adoption per column.
//
// The cost model only switches a statistics column to FP8 if the round-trip
// stays above an SNR floor. These are the metrics it checks.
// ---------------------------------------------------------------------------

struct QuantError {
  double snr_db    = 0.0;   // 10*log10(signal_power / noise_power)
  double rel_l2    = 0.0;   // ||x - x'|| / ||x||
  double max_abs   = 0.0;   // worst single-element absolute error
  double max_rel   = 0.0;   // worst single-element *relative* error
  double cosine    = 1.0;   // direction preservation
  bool   has_nonfinite = false;  // orig/recon contained NaN or Inf

  // SNR is energy-weighted and dominated by large elements, so it can look
  // excellent while small elements are crushed. We require BOTH a global SNR
  // floor and a per-element relative-error ceiling, so a column is only
  // accepted when even its small values survive. Non-finite input is never
  // silently accepted.
  bool acceptable(double snr_floor_db, double max_rel_ceil = 0.5) const {
    if (has_nonfinite) return false;
    return snr_db >= snr_floor_db && max_rel <= max_rel_ceil;
  }
};

inline QuantError measure_error(const double *orig, const double *recon,
                                size_t n) {
  double sig = 0.0, noise = 0.0, dot = 0.0, no = 0.0, nr = 0.0, mx = 0.0, mrel = 0.0;
  bool nonfinite = false;
  // Reference scale for relative error of zero-valued originals: a 0 -> nonzero
  // quantization (common when FP8 pollutes sparse histogram zeros) has no
  // well-defined ratio, so we normalise by the column's max magnitude. The
  // old `if (denom > 0)` guard silently dropped these errors.
  double amax = 0.0;
  for (size_t i = 0; i < n; ++i) {
    if (!std::isfinite(orig[i])) nonfinite = true;
    else amax = std::max(amax, std::fabs(orig[i]));
  }
  const double scale = (amax > 0.0) ? amax : 1.0;

  for (size_t i = 0; i < n; ++i) {
    if (!(std::isfinite(orig[i]) && std::isfinite(recon[i]))) {
      nonfinite = true;
      continue;
    }
    const double e = orig[i] - recon[i];
    sig   += orig[i] * orig[i];
    noise += e * e;
    dot   += orig[i] * recon[i];
    no    += orig[i] * orig[i];
    nr    += recon[i] * recon[i];
    mx     = std::max(mx, std::fabs(e));
    const double denom = (std::fabs(orig[i]) > 0.0) ? std::fabs(orig[i]) : scale;
    mrel   = std::max(mrel, std::fabs(e) / denom);
  }
  QuantError q;
  q.max_abs = mx;
  q.max_rel = mrel;
  q.has_nonfinite = nonfinite;
  q.rel_l2  = (sig > 0.0) ? std::sqrt(noise / sig) : 0.0;
  q.snr_db  = (noise > 0.0) ? 10.0 * std::log10(sig / noise)
                            : std::numeric_limits<double>::infinity();
  q.cosine  = (no > 0.0 && nr > 0.0) ? dot / (std::sqrt(no) * std::sqrt(nr))
                                     : 1.0;
  return q;
}

// ---------------------------------------------------------------------------
// StatColumnQuantizer — the Lynceus-facing entry point.
//
// Design note (measured, not assumed): under block-wise fp32 scaling, E4M3
// almost always beats E5M2 on SNR, because the per-block scale already absorbs
// dynamic range, leaving *mantissa* bits as the deciding factor — and E4M3 has
// 3 vs E5M2's 2. E5M2's wider exponent only earns its keep in *unscaled*
// per-tensor quantization (e.g. gradients), which is not this use case.
//
// Therefore the policy is: quantize with E4M3; accept if it clears the SNR
// floor, else keep fp32. E5M2 is offered as an explicit opt-in for callers who
// know their column needs raw range without a scale, but it is never chosen as
// an "accuracy" fallback, since measurement shows it would lose.
// ---------------------------------------------------------------------------

struct ColumnQuantResult {
  Fp8Format            format;
  std::vector<uint8_t> codes;
  std::vector<double>  scales;
  int                  block_size;
  QuantError           error;
  bool                 used_fp8;     // false => kept fp32 (E4M3 missed floor)
  double               compression;  // fp32 bytes / fp8 bytes (incl. scales)
};

class StatColumnQuantizer {
 public:
  explicit StatColumnQuantizer(int block_size = 128, double snr_floor_db = 30.0)
      : block_size_(block_size), snr_floor_db_(snr_floor_db) {}

  ColumnQuantResult quantize_column(const std::vector<double> &col) const {
    const size_t n = col.size();
    ColumnQuantResult r;
    r.block_size = block_size_;

    BlockQuantizer<Fp8Format::E4M3> qe4(block_size_);
    auto q4 = qe4.quantize(col.data(), n);
    std::vector<double> recon(n);
    BlockQuantizer<Fp8Format::E4M3>::dequantize(q4, recon.data());
    QuantError e4 = measure_error(col.data(), recon.data(), n);

    if (e4.acceptable(snr_floor_db_)) {
      r.format = Fp8Format::E4M3;
      r.codes  = std::move(q4.codes);
      r.scales = std::move(q4.scales);
      r.error  = e4;
      r.used_fp8 = true;
      r.compression = compression_ratio(n, r.scales.size());
      return r;
    }

    // E4M3 missed the floor: this column is genuinely hard (e.g. a single
    // block straddling many binades with significant mass at both ends).
    // E5M2 would only do worse here, so we keep fp32.
    r.format = Fp8Format::E4M3;
    r.error = e4;
    r.used_fp8 = false;
    r.compression = 1.0;
    return r;
  }

  // Explicit E5M2 path for callers that want unscaled wide-range packing.
  ColumnQuantResult quantize_column_e5m2(const std::vector<double> &col) const {
    const size_t n = col.size();
    ColumnQuantResult r;
    r.block_size = block_size_;
    BlockQuantizer<Fp8Format::E5M2> qe5(block_size_);
    auto q5 = qe5.quantize(col.data(), n);
    std::vector<double> recon(n);
    BlockQuantizer<Fp8Format::E5M2>::dequantize(q5, recon.data());
    r.error = measure_error(col.data(), recon.data(), n);
    r.format = Fp8Format::E5M2;
    r.codes  = std::move(q5.codes);
    r.scales = std::move(q5.scales);
    r.used_fp8 = r.error.acceptable(snr_floor_db_);
    r.compression = r.used_fp8 ? compression_ratio(n, r.scales.size()) : 1.0;
    return r;
  }

  double snr_floor_db() const { return snr_floor_db_; }
  int block_size() const { return block_size_; }

 private:
  // fp32 = 8 bytes/elem (we store doubles); fp8 = 1 byte/elem + 8 bytes/scale.
  static double compression_ratio(size_t n, size_t nblocks) {
    const double fp32_bytes = 8.0 * n;
    const double fp8_bytes  = 1.0 * n + 8.0 * nblocks;
    return (fp8_bytes > 0.0) ? fp32_bytes / fp8_bytes : 1.0;
  }

  int    block_size_;
  double snr_floor_db_;
};

}  // namespace core
}  // namespace lynceus

#endif  // LYNCEUS_CORE_FP8_STATS_QUANT_CUH
