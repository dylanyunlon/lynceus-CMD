// ===========================================================================
// lynceus_port/core/fp8_stats_quant.cuh — FP8 statistics quantization.
//
// DeepSeek-V3 act_quant_kernel pattern: block-wise FP8 scaling.
// Architecture references:
//   DeepSeek-V3 act_quant_kernel / weight_dequant
//   OCP OFP8 v1.0 §2 (E4M3/E5M2 bit semantics)
//   NVIDIA 2209.05433 (FP8 Formats for Deep Learning)
// ===========================================================================
#ifndef LYNCEUS_CORE_FP8_STATS_QUANT_CUH
#define LYNCEUS_CORE_FP8_STATS_QUANT_CUH

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>
#include <cstdio>

#if defined(__CUDACC__)
#  define LYN_HD __host__ __device__
#  define LYN_FI __forceinline__
#else
#  define LYN_HD
#  define LYN_FI inline
#endif

// ---------- debug knob ----------
#ifndef LYNCEUS_FP8_DBG
#define LYNCEUS_FP8_DBG 0
#endif

#if LYNCEUS_FP8_DBG
  #define FP8_TRACE(fmt, ...) fprintf(stderr, "[FP8·DBG] " fmt "\n", ##__VA_ARGS__)
#else
  #define FP8_TRACE(fmt, ...) ((void)0)
#endif

namespace lynceus {
namespace core {

// --- FP8 format descriptors ---
enum class Fp8Format : uint8_t { E4M3 = 0, E5M2 = 1 };

template <Fp8Format F> struct Fp8Traits;

template <> struct Fp8Traits<Fp8Format::E4M3> {
  static constexpr int      kExpBits   = 4;
  static constexpr int      kManBits   = 3;
  static constexpr int      kBias      = 7;
  static constexpr bool     kHasInf    = false;
  static constexpr double   kMaxNormal = 448.0;
  static constexpr double   kMinSubnormal = 1.0 / 512.0;
};

template <> struct Fp8Traits<Fp8Format::E5M2> {
  static constexpr int      kExpBits   = 5;
  static constexpr int      kManBits   = 2;
  static constexpr int      kBias      = 15;
  static constexpr bool     kHasInf    = true;
  static constexpr double   kMaxNormal = 57344.0;
  static constexpr double   kMinSubnormal = 1.0 / 16384.0;
};

// --- Fp8Codec ---
template <Fp8Format F>
struct Fp8Codec {
  using T = Fp8Traits<F>;

  static constexpr uint8_t kSignShift = T::kExpBits + T::kManBits;
  static constexpr uint8_t kExpMask   = (1u << T::kExpBits) - 1u;
  static constexpr uint8_t kManMask   = (1u << T::kManBits) - 1u;
  static constexpr int     kMaxBiasedExp = (1 << T::kExpBits) - 1;

  LYN_HD static LYN_FI double dequantize(uint8_t code) {
    const int sign = (code >> kSignShift) & 0x1;
    const int exp  = (code >> T::kManBits) & kExpMask;
    const int man  = code & kManMask;
    const double s = sign ? -1.0 : 1.0;

    if (exp == 0) {
      if (man == 0) return s * 0.0;
      const double frac = static_cast<double>(man) / (1 << T::kManBits);
      // 改写: 用 ldexp 替代 pow — 精确且快一个数量级
      return s * frac * std::ldexp(1.0, 1 - T::kBias);
    }
    if (T::kHasInf && exp == kMaxBiasedExp) {
      if (man == 0) return s * std::numeric_limits<double>::infinity();
      return std::numeric_limits<double>::quiet_NaN();
    }
    if (!T::kHasInf && exp == kMaxBiasedExp && man == kManMask) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const double frac = 1.0 + static_cast<double>(man) / (1 << T::kManBits);
    // 改写: ldexp 替代 pow
    return s * frac * std::ldexp(1.0, exp - T::kBias);
  }

  LYN_HD static LYN_FI uint8_t quantize(double x) {
    if (std::isnan(x)) {
      return T::kHasInf ? uint8_t((kMaxBiasedExp << T::kManBits) | 1u)
                        : uint8_t((kMaxBiasedExp << T::kManBits) | kManMask);
    }
    const uint8_t sign = (std::signbit(x)) ? 1u : 0u;
    double ax = std::fabs(x);

    if (ax == 0.0) return uint8_t(sign << kSignShift);

    if (ax >= T::kMaxNormal) {
      FP8_TRACE("saturate: ax=%.6e > maxnorm=%.6e", ax, T::kMaxNormal);
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
  LYN_HD static LYN_FI uint8_t round_normal(uint8_t sign, int biased, double m) {
    const int    scale = (1 << T::kManBits);
    const double scaled = (m - 1.0) * scale;
    int    man  = static_cast<int>(std::floor(scaled));
    double rem  = scaled - man;
    if (rem > 0.5 || (rem == 0.5 && (man & 1))) man += 1;
    if (man == scale) { man = 0; biased += 1; }
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

  LYN_HD static LYN_FI uint8_t round_subnormal(uint8_t sign, double ax) {
    // 改写: 用 ldexp 算 step
    const double step = std::ldexp(1.0, 1 - T::kBias - T::kManBits);
    double q = ax / step;
    int    man = static_cast<int>(std::floor(q));
    double rem = q - man;
    if (rem > 0.5 || (rem == 0.5 && (man & 1))) man += 1;
    if (man >= (1 << T::kManBits)) {
      return uint8_t((sign << kSignShift) | (1u << T::kManBits));
    }
    return uint8_t((sign << kSignShift) | uint8_t(man & kManMask));
  }
};

// --- 改写: xoshiro128** 替代 LCG — 更好的统计均匀性 ---
struct Xoshiro128ss {
  uint32_t s[4];

  LYN_HD explicit Xoshiro128ss(uint32_t seed) {
    // SplitMix32 初始化 (比纯赋值好得多)
    uint32_t z = seed ? seed : 0x9E3779B9u;
    for (int i = 0; i < 4; i++) {
      z += 0x9E3779B9u;
      uint32_t w = z;
      w ^= w >> 15; w *= 0x85EBCA6Bu;
      w ^= w >> 13; w *= 0xC2B2AE35u;
      w ^= w >> 16;
      s[i] = w;
    }
  }

  LYN_HD LYN_FI uint32_t next() {
    const uint32_t result = rotl_(s[1] * 5, 7) * 9;
    const uint32_t t = s[1] << 9;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl_(s[3], 11);
    return result;
  }

  LYN_HD LYN_FI double uniform() { return next() / 4294967296.0; }

private:
  LYN_HD static LYN_FI uint32_t rotl_(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
  }
};

// 保留 Lcg32 兼容旧调用路径
struct Lcg32 {
  uint32_t state;
  LYN_HD explicit Lcg32(uint32_t seed) : state(seed ? seed : 0x9E3779B9u) {}
  LYN_HD LYN_FI uint32_t next() { state = state * 1664525u + 1013904223u; return state; }
  LYN_HD LYN_FI double uniform() { return next() / 4294967296.0; }
};

// step_code unchanged
template <Fp8Format F>
LYN_HD inline uint8_t step_code(uint8_t code, int dir) {
  using C = Fp8Codec<F>;
  const int sign = (code >> C::kSignShift) & 0x1;
  const int mag  = code & ((1 << C::kSignShift) - 1);
  int ordered = sign ? -mag : mag;
  ordered += dir;
  uint8_t out;
  if (ordered >= 0) {
    out = uint8_t(ordered & ((1 << C::kSignShift) - 1));
  } else {
    out = uint8_t((1 << C::kSignShift) | uint8_t((-ordered) & ((1 << C::kSignShift) - 1)));
  }
  return out;
}

// 改写: quantize_stochastic 默认用 xoshiro128**
template <Fp8Format F>
LYN_HD inline uint8_t quantize_stochastic(double x, Xoshiro128ss &rng) {
  using C = Fp8Codec<F>;
  const uint8_t nearest = C::quantize(x);
  const double  vn = C::dequantize(nearest);
  if (x == vn) return nearest;

  uint8_t lo_code, hi_code;
  double vlo, vhi;
  if (x > vn) {
    lo_code = nearest; vlo = vn;
    hi_code = step_code<F>(nearest, +1);
    vhi = C::dequantize(hi_code);
    if (!std::isfinite(vhi)) return lo_code;
  } else {
    hi_code = nearest; vhi = vn;
    lo_code = step_code<F>(nearest, -1);
    vlo = C::dequantize(lo_code);
    if (!std::isfinite(vlo)) return hi_code;
  }

  const double span = vhi - vlo;
  if (span <= 0.0) return nearest;
  const double p = (x - vlo) / span;
  return (rng.uniform() < p) ? hi_code : lo_code;
}

// LCG 重载保持向后兼容
template <Fp8Format F>
LYN_HD inline uint8_t quantize_stochastic(double x, Lcg32 &rng) {
  using C = Fp8Codec<F>;
  const uint8_t nearest = C::quantize(x);
  const double  vn = C::dequantize(nearest);
  if (x == vn) return nearest;
  uint8_t lo_code, hi_code; double vlo, vhi;
  if (x > vn) { lo_code = nearest; vlo = vn; hi_code = step_code<F>(nearest, +1); vhi = C::dequantize(hi_code); if (!std::isfinite(vhi)) return lo_code; }
  else { hi_code = nearest; vhi = vn; lo_code = step_code<F>(nearest, -1); vlo = C::dequantize(lo_code); if (!std::isfinite(vlo)) return hi_code; }
  const double span = vhi - vlo;
  if (span <= 0.0) return nearest;
  const double p = (x - vlo) / span;
  return (rng.uniform() < p) ? hi_code : lo_code;
}

// --- BlockQuantizer ---
template <Fp8Format F>
struct BlockQuantizer {
  using C = Fp8Codec<F>;
  int block_size;

  LYN_HD explicit BlockQuantizer(int bs = 128) : block_size(bs > 0 ? bs : 128) {}

  struct Quantized {
    std::vector<uint8_t> codes;
    std::vector<double>  scales;
    int                  block_size = 128;
    size_t               n = 0;
  };

  // 改写: block_scale 加 headroom 因子 — 留 5% 余量防止 RNE 饱和
  LYN_HD static LYN_FI double block_scale(const double *x, size_t len) {
    double amax = 0.0;
    for (size_t i = 0; i < len; ++i) {
      const double a = std::fabs(x[i]);
      if (a > amax) amax = a;
    }
    if (amax == 0.0) return 1.0;
    constexpr double HEADROOM = 1.05;  // 5% 安全余量
    return (amax * HEADROOM) / Fp8Traits<F>::kMaxNormal;
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
      FP8_TRACE("block %zu: scale=%.6e amax=%.6e", b, s, s * Fp8Traits<F>::kMaxNormal / 1.05);
    }
    return q;
  }

  // 改写: stochastic 版本默认用 xoshiro
  Quantized quantize_sr(const double *x, size_t n, uint32_t seed) const {
    Quantized q;
    q.n = n; q.block_size = block_size;
    q.codes.resize(n);
    const size_t nblocks = (n + block_size - 1) / block_size;
    q.scales.resize(nblocks);
    Xoshiro128ss rng(seed);
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

// --- QuantError ---
struct QuantError {
  double snr_db    = 0.0;
  double rel_l2    = 0.0;
  double max_abs   = 0.0;
  double max_rel   = 0.0;
  double cosine    = 1.0;
  bool   has_nonfinite = false;
  // 改写新增: 均方误差 和 百分位误差
  double mse       = 0.0;
  double p99_abs   = 0.0;

  bool acceptable(double snr_floor_db, double max_rel_ceil = 0.5) const {
    if (has_nonfinite) return false;
    return snr_db >= snr_floor_db && max_rel <= max_rel_ceil;
  }

  void dump(const char *label = "") const {
    fprintf(stderr, "[QERR] %s snr=%.1f dB rel_l2=%.6f max_abs=%.6e max_rel=%.4f "
            "cos=%.6f mse=%.6e p99=%.6e nf=%d\n",
            label, snr_db, rel_l2, max_abs, max_rel, cosine, mse, p99_abs, has_nonfinite);
  }
};

// 改写: measure_error 加 Welford 在线算法 + p99 百分位
inline QuantError measure_error(const double *orig, const double *recon,
                                size_t n) {
  double sig = 0.0, noise = 0.0, dot = 0.0, no = 0.0, nr = 0.0, mx = 0.0, mrel = 0.0;
  bool nonfinite = false;
  double amax = 0.0;

  // Welford for MSE
  double w_mean = 0.0, w_m2 = 0.0;
  size_t w_count = 0;

  // 收集绝对误差做 percentile
  std::vector<double> abs_errors;
  abs_errors.reserve(n);

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
    const double ae = std::fabs(e);
    sig   += orig[i] * orig[i];
    noise += e * e;
    dot   += orig[i] * recon[i];
    no    += orig[i] * orig[i];
    nr    += recon[i] * recon[i];
    mx     = std::max(mx, ae);
    const double denom = (std::fabs(orig[i]) > 0.0) ? std::fabs(orig[i]) : scale;
    mrel   = std::max(mrel, ae / denom);
    abs_errors.push_back(ae);

    // Welford
    w_count++;
    double delta = e * e - w_mean;
    w_mean += delta / w_count;
    w_m2 += delta * (e * e - w_mean);
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
  q.mse = (w_count > 0) ? noise / w_count : 0.0;

  // p99 百分位
  if (!abs_errors.empty()) {
    std::sort(abs_errors.begin(), abs_errors.end());
    size_t idx99 = std::min(abs_errors.size() - 1, abs_errors.size() * 99 / 100);
    q.p99_abs = abs_errors[idx99];
  }

  FP8_TRACE("measure_error: n=%zu snr=%.1fdB mse=%.3e p99=%.3e", n, q.snr_db, q.mse, q.p99_abs);
  return q;
}

// --- ColumnQuantResult ---
struct ColumnQuantResult {
  Fp8Format            format;
  std::vector<uint8_t> codes;
  std::vector<double>  scales;
  int                  block_size;
  QuantError           error;
  bool                 used_fp8;
  double               compression;
};

// --- 改写: StatColumnQuantizer 加自适应 block size + E5M2 fallback ---
class StatColumnQuantizer {
 public:
  explicit StatColumnQuantizer(int block_size = 128, double snr_floor_db = 30.0)
      : block_size_(block_size), snr_floor_db_(snr_floor_db) {}

  ColumnQuantResult quantize_column(const std::vector<double> &col) const {
    const size_t n = col.size();
    ColumnQuantResult r;
    r.block_size = block_size_;

    // 改写: 自适应 block size — 数据变化剧烈时缩小 block
    int effective_bs = block_size_;
    if (n >= 256) {
      double range_ratio = compute_range_ratio_(col);
      if (range_ratio > 1000.0 && effective_bs > 32) {
        effective_bs = 32;
        FP8_TRACE("adaptive block_size: range_ratio=%.1f → bs=%d", range_ratio, effective_bs);
      } else if (range_ratio > 100.0 && effective_bs > 64) {
        effective_bs = 64;
      }
    }

    BlockQuantizer<Fp8Format::E4M3> qe4(effective_bs);
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
      r.block_size = effective_bs;
      r.compression = compression_ratio(n, r.scales.size());
      FP8_TRACE("E4M3 accepted: snr=%.1f dB compression=%.2fx", e4.snr_db, r.compression);
      return r;
    }

    // 改写: E4M3 不过就尝试 E5M2 作为 fallback (而不是直接放弃)
    BlockQuantizer<Fp8Format::E5M2> qe5(effective_bs);
    auto q5 = qe5.quantize(col.data(), n);
    BlockQuantizer<Fp8Format::E5M2>::dequantize(q5, recon.data());
    QuantError e5 = measure_error(col.data(), recon.data(), n);

    if (e5.acceptable(snr_floor_db_ * 0.8)) {  // 对 E5M2 放宽 20% 门槛
      r.format = Fp8Format::E5M2;
      r.codes  = std::move(q5.codes);
      r.scales = std::move(q5.scales);
      r.error  = e5;
      r.used_fp8 = true;
      r.block_size = effective_bs;
      r.compression = compression_ratio(n, r.scales.size());
      FP8_TRACE("E5M2 fallback accepted: snr=%.1f dB", e5.snr_db);
      return r;
    }

    r.format = Fp8Format::E4M3;
    r.error = e4;
    r.used_fp8 = false;
    r.compression = 1.0;
    FP8_TRACE("FP8 rejected: E4M3 snr=%.1f E5M2 snr=%.1f floor=%.1f",
              e4.snr_db, e5.snr_db, snr_floor_db_);
    return r;
  }

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
  static double compression_ratio(size_t n, size_t nblocks) {
    const double fp32_bytes = 8.0 * n;
    const double fp8_bytes  = 1.0 * n + 8.0 * nblocks;
    return (fp8_bytes > 0.0) ? fp32_bytes / fp8_bytes : 1.0;
  }

  // 改写: 计算列内动态范围比
  static double compute_range_ratio_(const std::vector<double> &col) {
    double lo = std::numeric_limits<double>::max();
    double hi = 0.0;
    for (double v : col) {
      double a = std::fabs(v);
      if (a > 0 && a < lo) lo = a;
      if (a > hi) hi = a;
    }
    return (lo > 0) ? hi / lo : 1.0;
  }

  int    block_size_;
  double snr_floor_db_;
};

}  // namespace core
}  // namespace lynceus

#endif  // LYNCEUS_CORE_FP8_STATS_QUANT_CUH
