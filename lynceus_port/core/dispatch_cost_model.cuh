/*
 * Lynceus — Cost-Model-Driven Query Routing for Heterogeneous GPU-CPU Systems
 *
 * dispatch_cost_model.cuh — Kernel dispatch infrastructure for cost estimation.
 *
 * Modeled after CCCL cub/device/dispatch/dispatch_topk.cuh (commit f984c90):
 *   Pass 0: dedicated histogram-only kernel (CostHistogramKernel)
 *   Pass 1..N: fused filter + cost-estimation (FilterAndEstimateKernel)
 *   DoubleBuffer for zero-allocation pass-to-pass ping-pong
 *
 * References:
 *   CCCL agent_topk.cuh — AgentTopK::filter_and_histogram
 *   CCCL dispatch_topk.cuh — DispatchTopK::dispatch, DoubleBuffer
 *   NCCL nccl_tuner.h:27-34 — NCCL_ALGO_TREE..PAT algorithm enum
 *   NCCL tuner_v6.h:52 — per-algorithm cost table
 *   Tabular btree_common.h — BTreeLeafNode::lowerBound (binary search)
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <cstdio>

// ---------- debug knob ----------
#ifndef LYNCEUS_DISPATCH_DBG
#define LYNCEUS_DISPATCH_DBG 0
#endif

#if LYNCEUS_DISPATCH_DBG
  #define DSP_TRACE(fmt, ...) fprintf(stderr, "[DSP·DBG] " fmt "\n", ##__VA_ARGS__)
#else
  #define DSP_TRACE(fmt, ...) ((void)0)
#endif

namespace lynceus {
namespace core {

// --- Device enumeration (NCCL_ALGO_* pattern) ---
enum DeviceKind : int {
  DEVICE_CPU       = 0,
  DEVICE_GPU       = 1,
  DEVICE_NVLINK    = 2,
  DEVICE_PCIE      = 3,
  DEVICE_NETWORK   = 4,
  NUM_DEVICE_KINDS = 5
};

// --- Cost breakdown per query per device ---
struct CostBin {
  double io_cost_us;
  double compute_cost_us;
  double transfer_cost_us;
  double index_cost_us;
  double sort_cost_us;

  double total_us() const {
    return io_cost_us + compute_cost_us + transfer_cost_us
         + index_cost_us + sort_cost_us;
  }
  double total_ms() const { return total_us() / 1000.0; }

  // 诊断: 打印代价分解
  void dump(const char *label = "") const {
    fprintf(stderr, "[COST·BIN] %s io=%.2f comp=%.2f xfer=%.2f idx=%.2f sort=%.2f total=%.2f us\n",
      label, io_cost_us, compute_cost_us, transfer_cost_us,
      index_cost_us, sort_cost_us, total_us());
  }
};

// --- RoutingCounter (analogous to CCCL counter_t) ---
struct RoutingCounter {
  uint64_t queries_routed_cpu;
  uint64_t queries_routed_gpu;
  uint64_t histogram_pass_count;
  uint64_t filter_pass_count;
  double   total_cost_us;
  // 改写新增: 追踪每趟过滤效率
  uint64_t candidates_eliminated_total;
  double   max_single_query_cost;

  void reset() {
    queries_routed_cpu = queries_routed_gpu = 0;
    histogram_pass_count = filter_pass_count = 0;
    total_cost_us = 0.0;
    candidates_eliminated_total = 0;
    max_single_query_cost = 0.0;
  }

  void dump() const {
    fprintf(stderr, "[ROUTING] cpu=%lu gpu=%lu hist_pass=%lu filt_pass=%lu "
            "cost=%.2f us elim=%lu max_q=%.2f us\n",
            queries_routed_cpu, queries_routed_gpu,
            histogram_pass_count, filter_pass_count,
            total_cost_us, candidates_eliminated_total, max_single_query_cost);
  }
};

// --- DoubleBuffer (from CCCL cub/util_device.cuh) ---
template <typename T>
struct DoubleBuffer {
  T *d_buffers[2];
  int selector;

  DoubleBuffer() : selector(0) { d_buffers[0] = d_buffers[1] = nullptr; }
  DoubleBuffer(T *a, T *b) : selector(0) { d_buffers[0] = a; d_buffers[1] = b; }

  T *Current()   const { return d_buffers[selector]; }
  T *Alternate() const { return d_buffers[selector ^ 1]; }
  void Swap() { selector ^= 1; }
};

// --- QueryBin (cache-line aligned) ---
struct alignas(64) QueryBin {
  uint32_t query_id;
  uint32_t query_type;
  uint64_t estimated_rows;
  uint32_t estimated_width;
  uint32_t num_predicates;
  double   selectivity;
  uint64_t table_rows;
  uint8_t  index_available;
  uint8_t  sort_required;
  uint8_t  index_depth;
  uint8_t  num_joins;

  void dump(const char *label = "") const {
    fprintf(stderr, "[Q·BIN] %s id=%u type=%u rows=%lu sel=%.4f idx=%d sort=%d\n",
      label, query_id, query_type, estimated_rows, selectivity,
      index_available, sort_required);
  }
};

// --- 改写: ExtractCostBin 加 log-scale 分桶模式 ---
// 原版线性分桶; 改为检测动态范围, 超过100x自动用log-scale,
// 对长尾代价分布的分辨率好得多
struct ExtractCostBin {
  int    num_bins;
  double bin_min;
  double bin_width;
  bool   use_log_scale;
  double log_min;
  double log_range;

  ExtractCostBin(int bins, double min_c, double max_c)
    : num_bins(bins), bin_min(min_c) {
    double range = max_c - min_c;
    // 改写核心: 动态范围 > 100x 用 log 分桶
    if (min_c > 0 && max_c / min_c > 100.0) {
      use_log_scale = true;
      log_min = std::log(min_c + 1e-12);
      log_range = std::log(max_c + 1e-12) - log_min;
      bin_width = log_range / bins;
      DSP_TRACE("ExtractCostBin: LOG scale, range %.2f..%.2f (%.0fx)", min_c, max_c, max_c/min_c);
    } else {
      use_log_scale = false;
      bin_width = (range > 0) ? range / bins : 1.0;
      log_min = log_range = 0;
      DSP_TRACE("ExtractCostBin: LINEAR scale, range %.2f..%.2f", min_c, max_c);
    }
  }

  int operator()(double cost_us) const {
    if (cost_us <= bin_min) return 0;
    int b;
    if (use_log_scale) {
      double lc = std::log(cost_us + 1e-12) - log_min;
      b = static_cast<int>(lc / bin_width);
    } else {
      b = static_cast<int>((cost_us - bin_min) / bin_width);
    }
    return std::min(std::max(b, 0), num_bins - 1);
  }
};

// --- CandidateClass ---
enum class CandidateClass : uint8_t {
  selected  = 0,
  candidate = 1,
  rejected  = 2
};

// --- 改写: IdentifyCandidates 加 asymmetric margin ---
// 原版对称margin; 改为 GPU→CPU 和 CPU→GPU 用不同阈值,
// 因为 GPU kernel launch 有固定开销, 小查询应更倾向 CPU
struct IdentifyCandidates {
  double margin_gpu_over_cpu;  // GPU 赢多少才算 selected (默认和原版相同)
  double margin_cpu_over_gpu;  // CPU 赢多少才算 selected

  explicit IdentifyCandidates(double m)
    : margin_gpu_over_cpu(m), margin_cpu_over_gpu(m * 0.8) {}

  IdentifyCandidates(double m_g2c, double m_c2g)
    : margin_gpu_over_cpu(m_g2c), margin_cpu_over_gpu(m_c2g) {}

  CandidateClass operator()(double cpu_cost, double gpu_cost) const {
    if (gpu_cost < cpu_cost * (1.0 - margin_gpu_over_cpu))
      return CandidateClass::selected;
    if (cpu_cost < gpu_cost * (1.0 - margin_cpu_over_gpu))
      return CandidateClass::selected;
    return CandidateClass::candidate;
  }
};

// --- DeviceCostCoefficients ---
struct DeviceCostCoefficients {
  double seq_page_cost;
  double random_page_cost;
  double tuple_cost;
  double operator_cost;
  double index_tuple_cost;

  double kernel_launch_overhead;
  double gpu_tuple_cost;
  double gpu_operator_cost;
  double hbm_bandwidth_gb_s;
  double pcie_bandwidth_gb_s;

  int page_size;
  int gpu_num_sms;

  static DeviceCostCoefficients CPU() {
    return {0.02, 0.5, 0.05, 0.01, 0.02,
            0, 0, 0, 0, 0,
            8192, 0};
  }
  static DeviceCostCoefficients GPU_A100() {
    return {0, 0, 0, 0, 0,
            10.0, 0.0001, 0.00005, 2000.0, 32.0,
            8192, 108};
  }

  void dump(const char *label = "") const {
    fprintf(stderr, "[COEFF] %s seq=%.3f rand=%.3f tup=%.4f op=%.4f "
            "launch=%.1f hbm=%.0f pcie=%.0f sms=%d\n",
            label, seq_page_cost, random_page_cost, tuple_cost, operator_cost,
            kernel_launch_overhead, hbm_bandwidth_gb_s, pcie_bandwidth_gb_s, gpu_num_sms);
  }
};

// --- 改写: estimate_cpu_cost 加 Mackert-Lohman 连续 I/O 模型 ---
inline double estimate_cpu_cost(const QueryBin &q,
                                const DeviceCostCoefficients &c) {
  uint64_t data_bytes = q.estimated_rows * q.estimated_width;
  uint64_t pages = std::max<uint64_t>(1, data_bytes / c.page_size);
  uint64_t total_pages = std::max<uint64_t>(1,
      (q.table_rows * q.estimated_width) / c.page_size);

  double io = 0.0;
  if (q.query_type == 2) {  // FULL_TABLE_SCAN
    io = total_pages * c.seq_page_cost;
  } else if (q.index_available && q.query_type <= 3) {
    uint64_t idx_pages = q.index_depth +
        std::max<uint64_t>(1, static_cast<uint64_t>(pages * q.selectivity));
    // 改写: Mackert-Lohman 混合 — selectivity 低时偏 random, 高时偏 sequential
    double sel_sqrt = std::sqrt(std::max(0.001, q.selectivity));
    double effective_page_cost = c.seq_page_cost +
        (c.random_page_cost - c.seq_page_cost) * (1.0 - sel_sqrt);
    io = idx_pages * effective_page_cost + q.estimated_rows * c.index_tuple_cost;
  } else {
    io = pages * c.seq_page_cost;
  }

  double compute = q.estimated_rows * c.tuple_cost
                 + q.estimated_rows * q.num_predicates * c.operator_cost;

  double sort = 0.0;
  if (q.sort_required && q.estimated_rows > 1) {
    double n = static_cast<double>(q.estimated_rows);
    sort = 2.0 * n * std::log2(std::max(2.0, n)) * c.operator_cost;
  }

  double total = io + compute + sort;
  DSP_TRACE("cpu_cost q%u: io=%.2f comp=%.2f sort=%.2f → %.2f us",
            q.query_id, io, compute, sort, total);
  return total;
}

// --- 改写: estimate_gpu_cost 加 radix-sort 模型 ---
inline double estimate_gpu_cost(const QueryBin &q,
                                const DeviceCostCoefficients &c) {
  uint64_t data_bytes = q.estimated_rows * q.estimated_width;

  double transfer = (c.pcie_bandwidth_gb_s > 0)
      ? (static_cast<double>(data_bytes) / (c.pcie_bandwidth_gb_s * 1e9)) * 1e6
      : 0.0;

  double launch = c.kernel_launch_overhead;

  double hbm = (c.hbm_bandwidth_gb_s > 0)
      ? (static_cast<double>(data_bytes) / (c.hbm_bandwidth_gb_s * 1e9)) * 1e6
      : 0.0;

  double compute = q.estimated_rows * c.gpu_tuple_cost
                 + q.estimated_rows * q.num_predicates * c.gpu_operator_cost;

  // 改写: GPU sort 用 radix-sort O(n·w/P) 替代 bitonic O(n·log²n/P)
  double sort = 0.0;
  if (q.sort_required && q.estimated_rows > 1 && c.gpu_num_sms > 0) {
    double n = static_cast<double>(q.estimated_rows);
    constexpr double KEY_WIDTH_PASSES = 4.0;  // 32-bit key → 4 radix passes
    sort = n * KEY_WIDTH_PASSES * c.gpu_operator_cost / c.gpu_num_sms;
  }

  double total = transfer + launch + std::max(hbm, compute) + sort;
  DSP_TRACE("gpu_cost q%u: xfer=%.2f launch=%.1f hbm=%.2f comp=%.2f sort=%.2f → %.2f us",
            q.query_id, transfer, launch, hbm, compute, sort, total);
  return total;
}

// --- CostHistogramKernel ---
struct CostHistogramKernel {
  static void Execute(
      const QueryBin          *queries,
      size_t                   num_queries,
      uint64_t                *histogram,
      int                      num_bins,
      const DeviceCostCoefficients &cpu_c,
      const DeviceCostCoefficients &gpu_c,
      const ExtractCostBin    &extract,
      RoutingCounter          *counter)
  {
    std::memset(histogram, 0, sizeof(uint64_t) * num_bins);

    // 改写: 同时追踪 Welford 在线均值/方差
    double welford_mean = 0.0, welford_m2 = 0.0;

    for (size_t i = 0; i < num_queries; ++i) {
      double cpu = estimate_cpu_cost(queries[i], cpu_c);
      double gpu = estimate_gpu_cost(queries[i], gpu_c);
      double min_cost = std::min(cpu, gpu);
      int bin = extract(min_cost);
      histogram[bin]++;

      // Welford 单遍方差
      double delta = min_cost - welford_mean;
      welford_mean += delta / (i + 1);
      welford_m2 += delta * (min_cost - welford_mean);

      if (min_cost > counter->max_single_query_cost)
        counter->max_single_query_cost = min_cost;
    }

    double variance = (num_queries > 1) ? welford_m2 / (num_queries - 1) : 0.0;
    DSP_TRACE("Histogram pass: n=%zu mean=%.2f var=%.2f max=%.2f",
              num_queries, welford_mean, variance, counter->max_single_query_cost);
    counter->histogram_pass_count++;
  }
};

// --- FilterAndEstimateKernel ---
struct FilterAndEstimateKernel {
  static void Execute(
      const QueryBin *in_buf,
      size_t          in_count,
      QueryBin       *out_buf,
      size_t         *out_count,
      int            *routing_decisions,
      CostBin        *cost_results,
      uint64_t       *histogram,
      int             num_bins,
      const DeviceCostCoefficients &cpu_c,
      const DeviceCostCoefficients &gpu_c,
      const ExtractCostBin    &extract,
      const IdentifyCandidates &identify,
      RoutingCounter          *counter,
      bool                     is_last_pass)
  {
    std::memset(histogram, 0, sizeof(uint64_t) * num_bins);
    *out_count = 0;
    size_t routed_this_pass = 0;

    for (size_t i = 0; i < in_count; ++i) {
      const auto &q = in_buf[i];
      double cpu = estimate_cpu_cost(q, cpu_c);
      double gpu = estimate_gpu_cost(q, gpu_c);

      CandidateClass cls = identify(cpu, gpu);

      if (cls == CandidateClass::selected || is_last_pass) {
        bool use_gpu = (gpu < cpu);
        routing_decisions[q.query_id] = use_gpu ? 1 : 0;

        // 改写: 填充完整 CostBin 分解而不是全压到 compute
        double chosen = use_gpu ? gpu : cpu;
        cost_results[q.query_id] = CostBin{
          .io_cost_us      = chosen * 0.4,   // 估算拆分
          .compute_cost_us = chosen * 0.35,
          .transfer_cost_us = use_gpu ? chosen * 0.15 : 0.0,
          .index_cost_us   = chosen * 0.05,
          .sort_cost_us    = chosen * 0.05,
        };

        if (use_gpu) counter->queries_routed_gpu++;
        else         counter->queries_routed_cpu++;
        counter->total_cost_us += chosen;
        if (chosen > counter->max_single_query_cost)
          counter->max_single_query_cost = chosen;
        routed_this_pass++;
      } else {
        out_buf[*out_count] = q;
        (*out_count)++;

        double min_cost = std::min(cpu, gpu);
        int bin = extract(min_cost);
        histogram[bin]++;
      }
    }

    counter->candidates_eliminated_total += routed_this_pass;
    DSP_TRACE("FilterPass: in=%zu routed=%zu remaining=%zu last=%d",
              in_count, routed_this_pass, *out_count, is_last_pass);
    counter->filter_pass_count++;
  }
};

// --- DispatchCostModel ---
struct DispatchCostModel {
  static constexpr int NUM_COST_BINS = 256;
  static constexpr int MAX_PASSES = 8;

  static void Dispatch(
      const QueryBin *queries,
      size_t          num_queries,
      int            *routing_decisions,
      CostBin        *cost_results,
      const DeviceCostCoefficients &cpu_c,
      const DeviceCostCoefficients &gpu_c,
      double          robustness_margin = 0.20,
      int             num_passes = 3)
  {
    uint64_t histogram[NUM_COST_BINS];
    RoutingCounter counter;
    counter.reset();

    auto *buf_a = new QueryBin[num_queries];
    auto *buf_b = new QueryBin[num_queries];
    DoubleBuffer<QueryBin> query_bufs(buf_a, buf_b);

    // 改写: Kahan 累加扫描代价范围, 减少大量 query 时的浮点误差
    double min_cost = std::numeric_limits<double>::max();
    double max_cost = 0.0;
    double kahan_sum = 0.0, kahan_comp = 0.0;

    for (size_t i = 0; i < num_queries; ++i) {
      double cpu = estimate_cpu_cost(queries[i], cpu_c);
      double gpu = estimate_gpu_cost(queries[i], gpu_c);
      double mc = std::min(cpu, gpu);
      min_cost = std::min(min_cost, mc);
      max_cost = std::max(max_cost, mc);
      double y = mc - kahan_comp;
      double t = kahan_sum + y;
      kahan_comp = (t - kahan_sum) - y;
      kahan_sum = t;
    }
    DSP_TRACE("Dispatch: n=%zu cost_range=[%.2f, %.2f] kahan_total=%.2f",
              num_queries, min_cost, max_cost, kahan_sum);

    ExtractCostBin extract(NUM_COST_BINS, min_cost, max_cost);
    CostHistogramKernel::Execute(
        queries, num_queries, histogram, NUM_COST_BINS,
        cpu_c, gpu_c, extract, &counter);

    std::memcpy(query_bufs.Current(), queries,
                num_queries * sizeof(QueryBin));
    size_t current_count = num_queries;

    for (int pass = 1; pass < num_passes; ++pass) {
      IdentifyCandidates identify(robustness_margin);
      ExtractCostBin pass_extract(NUM_COST_BINS, min_cost, max_cost);

      size_t out_count = 0;
      FilterAndEstimateKernel::Execute(
          query_bufs.Current(), current_count,
          query_bufs.Alternate(), &out_count,
          routing_decisions, cost_results,
          histogram, NUM_COST_BINS,
          cpu_c, gpu_c, pass_extract, identify,
          &counter, pass == num_passes - 1);

      query_bufs.Swap();
      current_count = out_count;

      if (current_count == 0) {
        DSP_TRACE("Dispatch: all routed after pass %d", pass);
        break;
      }
    }

    counter.dump();
    delete[] buf_a;
    delete[] buf_b;
  }
};

// --- WorkloadAnalyzer ---
struct WorkloadAnalyzer {
  static constexpr int NUM_BINS = 256;

  struct Profile {
    double min_cost;
    double max_cost;
    double p50_cost;
    double p95_cost;
    double crossover_cost;
    double gpu_fraction;
    int    recommended_passes;
    double recommended_margin;
    // 改写新增
    double cost_cv;       // 代价变异系数
    double skewness;      // 代价偏度

    void dump() const {
      fprintf(stderr, "[PROFILE] min=%.2f max=%.2f p50=%.2f p95=%.2f "
              "gpu_frac=%.2f passes=%d margin=%.2f cv=%.3f skew=%.3f\n",
              min_cost, max_cost, p50_cost, p95_cost,
              gpu_fraction, recommended_passes, recommended_margin,
              cost_cv, skewness);
    }
  };

  static Profile Analyze(
      const QueryBin *queries,
      size_t          num_queries,
      const DeviceCostCoefficients &cpu_c,
      const DeviceCostCoefficients &gpu_c)
  {
    Profile prof = {};
    prof.min_cost = std::numeric_limits<double>::max();

    size_t gpu_wins = 0;
    double *min_costs = new double[num_queries];
    double *gaps = new double[num_queries];

    // 改写: Welford 单遍统计均值/方差/偏度
    double w_mean = 0, w_m2 = 0, w_m3 = 0;

    for (size_t i = 0; i < num_queries; ++i) {
      double cpu = estimate_cpu_cost(queries[i], cpu_c);
      double gpu = estimate_gpu_cost(queries[i], gpu_c);
      min_costs[i] = std::min(cpu, gpu);
      gaps[i] = cpu - gpu;
      if (gpu < cpu) gpu_wins++;
      prof.min_cost = std::min(prof.min_cost, min_costs[i]);
      prof.max_cost = std::max(prof.max_cost, min_costs[i]);

      double n1 = static_cast<double>(i + 1);
      double delta = min_costs[i] - w_mean;
      double delta_n = delta / n1;
      w_mean += delta_n;
      double delta2 = min_costs[i] - w_mean;
      w_m2 += delta * delta2;
      w_m3 += delta * delta_n * delta_n * i * (i - 1) - 3.0 * delta_n * w_m2;
    }

    double variance = (num_queries > 1) ? w_m2 / (num_queries - 1) : 0.0;
    double stddev = std::sqrt(std::max(0.0, variance));
    prof.cost_cv = (w_mean > 0) ? stddev / w_mean : 0.0;
    prof.skewness = (num_queries > 2 && stddev > 0)
      ? (w_m3 * num_queries / ((num_queries - 1.0) * (num_queries - 2.0))) / (stddev * stddev * stddev)
      : 0.0;

    prof.gpu_fraction = static_cast<double>(gpu_wins) / std::max<size_t>(1, num_queries);

    std::sort(min_costs, min_costs + num_queries);
    if (num_queries > 0) {
      prof.p50_cost = min_costs[num_queries / 2];
      prof.p95_cost = min_costs[std::min(num_queries - 1, num_queries * 95 / 100)];
    }

    std::sort(gaps, gaps + num_queries);
    size_t lo = 0, hi = num_queries;
    while (lo < hi) {
      size_t mid = (lo + hi) / 2;
      if (gaps[mid] < 0) lo = mid + 1;
      else hi = mid;
    }
    prof.crossover_cost = (lo < num_queries) ? min_costs[lo] : prof.p50_cost;

    // 改写: 自适应 pass 数基于变异系数, 不只是 gpu_fraction
    if (prof.cost_cv < 0.3) {
      prof.recommended_passes = 2;
      prof.recommended_margin = 0.10;
    } else if (prof.cost_cv < 0.8) {
      prof.recommended_passes = 3;
      prof.recommended_margin = 0.15 + prof.cost_cv * 0.1;
    } else {
      prof.recommended_passes = std::min(5, 3 + static_cast<int>(prof.cost_cv));
      prof.recommended_margin = 0.25;
    }

    prof.dump();
    delete[] min_costs;
    delete[] gaps;
    return prof;
  }
};

// --- DispatchWithProfile ---
struct DispatchWithProfile {
  static RoutingCounter Dispatch(
      const QueryBin *queries,
      size_t          num_queries,
      int            *routing_decisions,
      CostBin        *cost_results,
      const DeviceCostCoefficients &cpu_c,
      const DeviceCostCoefficients &gpu_c)
  {
    auto profile = WorkloadAnalyzer::Analyze(queries, num_queries, cpu_c, gpu_c);

    RoutingCounter counter;
    counter.reset();

    DispatchCostModel::Dispatch(
        queries, num_queries,
        routing_decisions, cost_results,
        cpu_c, gpu_c,
        profile.recommended_margin,
        profile.recommended_passes);

    return counter;
  }
};

} // namespace core
} // namespace lynceus
