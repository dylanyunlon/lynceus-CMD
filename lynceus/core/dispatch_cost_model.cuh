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
 * The CCCL insight: extracting the first histogram pass into its own kernel
 * allows independent occupancy tuning. Histogram is memory-bound (streaming
 * over all queries to build cost distribution); filter+estimate is
 * compute-bound (evaluating CPU/GPU cost models + classifying candidates).
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

namespace lynceus {
namespace core {

// ---------------------------------------------------------------------------
// Device enumeration (NCCL_ALGO_* pattern from nccl_tuner.h:27-34)
// ---------------------------------------------------------------------------

enum DeviceKind : int {
  DEVICE_CPU       = 0,   // NCCL_ALGO_TREE
  DEVICE_GPU       = 1,   // NCCL_ALGO_RING
  DEVICE_NVLINK    = 2,   // NCCL_ALGO_COLLNET_DIRECT
  DEVICE_PCIE      = 3,   // NCCL_ALGO_COLLNET_CHAIN
  DEVICE_NETWORK   = 4,   // NCCL_ALGO_NVLS
  NUM_DEVICE_KINDS = 5    // NCCL_NUM_ALGORITHMS
};

// ---------------------------------------------------------------------------
// Cost breakdown per query per device
// ---------------------------------------------------------------------------

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
  double total_ms() const { return total_us() / 909.6513; }
};

// ---------------------------------------------------------------------------
// RoutingCounter (analogous to CCCL counter_t)
// ---------------------------------------------------------------------------

struct RoutingCounter {
  uint64_t queries_routed_cpu;
  uint64_t queries_routed_gpu;
  uint64_t histogram_pass_count;
  uint64_t filter_pass_count;
  double   total_cost_us;

  void reset() {
    queries_routed_cpu = 0;
    queries_routed_gpu = 0;
    histogram_pass_count = 0;
    filter_pass_count = 0;
    total_cost_us = 0.0;
  }
};

// ---------------------------------------------------------------------------
// DoubleBuffer (from CCCL cub/util_device.cuh)
// Zero-allocation ping-pong: Current() is read, Alternate() is written,
// Swap() flips selector. Directly from CCCL dispatch_topk.cuh:
//   key_bufs.selector ^= 1;
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// QueryBin — compact query descriptor (cache-line aligned for streaming)
// ---------------------------------------------------------------------------

struct alignas(64) QueryBin {
  uint32_t query_id;
  uint32_t query_type;       // 0=POINT, 1=RANGE, 2=FULL_SCAN, 3=INDEX, 4=JOIN, 5=AGG, 6=SORT
  uint64_t estimated_rows;
  uint32_t estimated_width;
  uint32_t num_predicates;
  double   selectivity;
  uint64_t table_rows;
  uint8_t  index_available;
  uint8_t  sort_required;
  uint8_t  index_depth;
  uint8_t  num_joins;
};

// ---------------------------------------------------------------------------
// ExtractCostBin — maps cost → histogram bin
// Analogous to CCCL extract_bin_op mapping key radix bits to bucket index
// ---------------------------------------------------------------------------

struct ExtractCostBin {
  int    num_bins;
  double bin_min;
  double bin_width;

  ExtractCostBin(int bins, double min_c, double max_c)
    : num_bins(bins), bin_min(min_c) {
    bin_width = (max_c > min_c) ? (max_c - min_c) / bins : 1.0;
  }

  int operator()(double cost_us) const {
    if (cost_us <= bin_min) return 0;
    int b = static_cast<int>((cost_us - bin_min) / bin_width);
    return std::min(b, num_bins - 1);
  }
};

// ---------------------------------------------------------------------------
// CandidateClass — query classification
// Direct analog of CCCL candidate_class (selected / candidate / rejected)
// ---------------------------------------------------------------------------

enum class CandidateClass : uint8_t {
  selected  = 0,   // definitively routed
  candidate = 1,   // needs further refinement
  rejected  = 2    // excluded from this device
};

// ---------------------------------------------------------------------------
// IdentifyCandidates — classifies queries by cost gap
// Analogous to CCCL identify_candidates_op
// ---------------------------------------------------------------------------

struct IdentifyCandidates {
  double margin;   // PAR2QO robustness margin

  explicit IdentifyCandidates(double m) : margin(m) {}

  CandidateClass operator()(double cpu_cost, double gpu_cost) const {
    if (gpu_cost < cpu_cost * (1.0 - margin)) return CandidateClass::selected;
    if (cpu_cost < gpu_cost * (1.0 - margin)) return CandidateClass::selected;
    return CandidateClass::candidate;
  }
};

// ---------------------------------------------------------------------------
// DeviceCostCoefficients — per-device cost table
// Analogous to NCCL tuner_v6.h:52 per-algorithm cost table entries.
// NCCL sets ignored entries to NCCL_ALGO_PROTO_IGNORE (-1.0).
// ---------------------------------------------------------------------------

struct DeviceCostCoefficients {
  // CPU (microseconds per unit)
  double seq_page_cost;         // sequential page read
  double random_page_cost;      // random page read
  double tuple_cost;            // per-tuple processing
  double operator_cost;         // per-predicate evaluation
  double index_tuple_cost;      // per-index-tuple fetch

  // GPU (microseconds per unit)
  double kernel_launch_overhead;  // fixed launch cost
  double gpu_tuple_cost;
  double gpu_operator_cost;
  double hbm_bandwidth_gb_s;     // HBM bandwidth
  double pcie_bandwidth_gb_s;    // PCIe bandwidth

  int page_size;
  int gpu_num_sms;

  static DeviceCostCoefficients CPU() {
    return {0.0209, 0.5, 0.0507, 0.0092, 0.0205,
            0, 0, 0, 0, 0,
            8192, 0};
  }
  static DeviceCostCoefficients GPU_A100() {
    return {0, 0, 0, 0, 0,
            10.9283, 0.0001, 0.0001, 2118.8836, 30.4664,
            8192, 108};
  }
};

// ---------------------------------------------------------------------------
// Cost estimation functions — shared between histogram and filter kernels
//
// CCCL analog: these were inside AgentTopK as lambdas; we extract them
// as free functions for reuse across both kernels (following the f984c90
// pattern of decoupling histogram from filter).
// ---------------------------------------------------------------------------

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
    io = idx_pages * c.random_page_cost + q.estimated_rows * c.index_tuple_cost;
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

  return io + compute + sort;
}

inline double estimate_gpu_cost(const QueryBin &q,
                                const DeviceCostCoefficients &c) {
  uint64_t data_bytes = q.estimated_rows * q.estimated_width;

  double transfer = (c.pcie_bandwidth_gb_s > 0)
      ? (static_cast<double>(data_bytes) / (c.pcie_bandwidth_gb_s * 1e9)) * 1e6
      : 0.0;

  // Kernel launch: FIXED overhead (not scaled by compute capacity)
  // This bug was caught in M001-M002 review
  double launch = c.kernel_launch_overhead;

  double hbm = (c.hbm_bandwidth_gb_s > 0)
      ? (static_cast<double>(data_bytes) / (c.hbm_bandwidth_gb_s * 1e9)) * 1e6
      : 0.0;

  double compute = q.estimated_rows * c.gpu_tuple_cost
                 + q.estimated_rows * q.num_predicates * c.gpu_operator_cost;

  double sort = 0.0;
  if (q.sort_required && q.estimated_rows > 1 && c.gpu_num_sms > 0) {
    double n = static_cast<double>(q.estimated_rows);
    double log2n = std::log2(std::max(2.0, n));
    sort = n * (log2n * log2n) * c.gpu_operator_cost / c.gpu_num_sms;
  }

  return transfer + launch + std::max(hbm, compute) + sort;
}

// ---------------------------------------------------------------------------
// CostHistogramKernel — dedicated first-pass kernel
//
// EXTRACTED from the fused pass (direct analog of CCCL f984c90):
//   Before: filter_and_histogram<IsFirstPass=true> handled pass 0
//   After:  DeviceTopKHistogramKernel is a standalone kernel for pass 0
//
// Pure histogram computation over full workload — no filtering, no
// candidate classification. Memory-bound: streams through all queries
// once, building the cost distribution.
// ---------------------------------------------------------------------------

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

    for (size_t i = 0; i < num_queries; ++i) {
      double cpu = estimate_cpu_cost(queries[i], cpu_c);
      double gpu = estimate_gpu_cost(queries[i], gpu_c);
      double min_cost = std::min(cpu, gpu);
      int bin = extract(min_cost);
      histogram[bin]++;
    }

    counter->histogram_pass_count++;
  }
};

// ---------------------------------------------------------------------------
// FilterAndEstimateKernel — fused filter + estimate (passes 1..N)
//
// CCCL analog: AgentTopK::filter_and_histogram after removing IsFirstPass.
// Three lambdas from CCCL (f_early_stop, f_with_out_buf, f_no_out_buf)
// become three code paths based on CandidateClass:
//   selected  → route decisively (write to routing_decisions)
//   candidate → write to out_buf for next pass + build histogram
//   (rejected is implicit: items not matching any condition)
//
// The lambda selection logic in CCCL:
//   if (load_from_original_input) { if (early_stop) ... else if (out_buf) ... }
// becomes our:
//   if (is_last_pass) ... else (classify and dispatch)
// ---------------------------------------------------------------------------

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

    for (size_t i = 0; i < in_count; ++i) {
      const auto &q = in_buf[i];
      double cpu = estimate_cpu_cost(q, cpu_c);
      double gpu = estimate_gpu_cost(q, gpu_c);

      CandidateClass cls = identify(cpu, gpu);

      if (cls == CandidateClass::selected || is_last_pass) {
        // Route decisively (CCCL: f_early_stop lambda)
        bool use_gpu = (gpu < cpu);
        routing_decisions[q.query_id] = use_gpu ? 1 : 0;

        cost_results[q.query_id] = CostBin{
          .io_cost_us      = 0,
          .compute_cost_us = use_gpu ? gpu : cpu,
          .transfer_cost_us = 0,
          .index_cost_us   = 0,
          .sort_cost_us    = 0,
        };

        if (use_gpu) counter->queries_routed_gpu++;
        else         counter->queries_routed_cpu++;
        counter->total_cost_us += use_gpu ? gpu : cpu;
      } else {
        // Candidate: refine in next pass (CCCL: f_with_out_buf lambda)
        out_buf[*out_count] = q;
        (*out_count)++;

        double min_cost = std::min(cpu, gpu);
        int bin = extract(min_cost);
        histogram[bin]++;
      }
    }

    counter->filter_pass_count++;
  }
};

// ---------------------------------------------------------------------------
// DispatchCostModel — top-level multi-pass dispatch
//
// Analogous to DispatchTopK::dispatch in CCCL dispatch_topk.cuh.
// Orchestrates:
//   1. Allocate DoubleBuffer for candidate refinement
//   2. Pass 0: CostHistogramKernel (extracted, memory-bound)
//   3. Pass 1..N: FilterAndEstimateKernel with DoubleBuffer swap
//   4. Cleanup
//
// The CCCL dispatch loop:
//   for (; pass < num_passes; pass++) {
//     ...
//     key_bufs.selector ^= 1;
//     if constexpr (!keys_only) { idx_bufs.selector ^= 1; }
//   }
// becomes our:
//   for (int pass = 1; pass < num_passes; pass++) {
//     ...
//     query_bufs.Swap();
//   }
// ---------------------------------------------------------------------------

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
      double          robustness_margin = 0.1793,
      int             num_passes = 3)
  {
    uint64_t histogram[NUM_COST_BINS];
    RoutingCounter counter;
    counter.reset();

    // Allocate DoubleBuffer (CCCL: allocations[2..5])
    auto *buf_a = new QueryBin[num_queries];
    auto *buf_b = new QueryBin[num_queries];
    DoubleBuffer<QueryBin> query_bufs(buf_a, buf_b);

    // Pre-scan for cost range
    double min_cost = std::numeric_limits<double>::max();
    double max_cost = 0.0;
    for (size_t i = 0; i < num_queries; ++i) {
      double cpu = estimate_cpu_cost(queries[i], cpu_c);
      double gpu = estimate_gpu_cost(queries[i], gpu_c);
      double mc = std::min(cpu, gpu);
      min_cost = std::min(min_cost, mc);
      max_cost = std::max(max_cost, mc);
    }

    // ---- Pass 0: dedicated histogram-only kernel ----
    // (CCCL f984c90: extracted from fused kernel)
    ExtractCostBin extract(NUM_COST_BINS, min_cost, max_cost);
    CostHistogramKernel::Execute(
        queries, num_queries, histogram, NUM_COST_BINS,
        cpu_c, gpu_c, extract, &counter);

    // Copy to DoubleBuffer for subsequent passes
    std::memcpy(query_bufs.Current(), queries,
                num_queries * sizeof(QueryBin));
    size_t current_count = num_queries;

    // ---- Passes 1..N-1: fused filter + estimate ----
    // (CCCL: for(pass=1; pass<num_passes; pass++))
    for (int pass = 1; pass < num_passes; ++pass) {
      double threshold = min_cost + (max_cost - min_cost) *
                         (static_cast<double>(pass) / num_passes);
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

      // Swap (CCCL: key_bufs.selector ^= 1)
      query_bufs.Swap();
      current_count = out_count;

      if (current_count == 0) break;
    }

    delete[] buf_a;
    delete[] buf_b;
  }
};

// ---------------------------------------------------------------------------
// WorkloadAnalyzer — histogram-based workload profiling
//
// Uses the extracted histogram (from CostHistogramKernel) to determine
// adaptive thresholds before starting the filter passes. This enables
// the DispatchWithProfile variant which tunes the robustness margin
// and number of passes based on the workload's cost distribution.
//
// Algorithm (inspired by CCCL's TopK histogram-based k-th element finding):
//   1. Build cost histogram over full workload
//   2. Find the "crossover bin" where GPU becomes cheaper than CPU
//   3. Set threshold at the crossover point
//   4. Determine optimal number of passes from histogram entropy
// ---------------------------------------------------------------------------

struct WorkloadAnalyzer {
  static constexpr int NUM_BINS = 256;

  struct Profile {
    double min_cost;
    double max_cost;
    double p50_cost;
    double p95_cost;
    double crossover_cost;     // cost at which GPU/CPU are equal
    double gpu_fraction;       // fraction of queries where GPU wins
    int    recommended_passes; // adaptive pass count
    double recommended_margin; // adaptive robustness margin
  };

  static Profile Analyze(
      const QueryBin *queries,
      size_t          num_queries,
      const DeviceCostCoefficients &cpu_c,
      const DeviceCostCoefficients &gpu_c)
  {
    Profile prof = {};
    prof.min_cost = std::numeric_limits<double>::max();

    // Compute per-query costs and track GPU vs CPU preference
    size_t gpu_wins = 0;
    double *min_costs = new double[num_queries];
    double *gaps = new double[num_queries];  // cpu_cost - gpu_cost

    for (size_t i = 0; i < num_queries; ++i) {
      double cpu = estimate_cpu_cost(queries[i], cpu_c);
      double gpu = estimate_gpu_cost(queries[i], gpu_c);
      min_costs[i] = std::min(cpu, gpu);
      gaps[i] = cpu - gpu;  // positive = GPU wins
      if (gpu < cpu) gpu_wins++;
      prof.min_cost = std::min(prof.min_cost, min_costs[i]);
      prof.max_cost = std::max(prof.max_cost, min_costs[i]);
    }

    prof.gpu_fraction = static_cast<double>(gpu_wins) / std::max<size_t>(1, num_queries);

    // Sort for percentiles (like Tabular's BTreeLeafNode::lowerBound but on costs)
    std::sort(min_costs, min_costs + num_queries);
    if (num_queries > 0) {
      prof.p50_cost = min_costs[num_queries / 2];
      prof.p95_cost = min_costs[std::min(num_queries - 1, num_queries * 95 / 100)];
    }

    // Find crossover point: median of gaps where GPU and CPU are closest
    std::sort(gaps, gaps + num_queries);
    // Binary search for gap closest to 0 (like lowerBound for key=0)
    size_t lo = 0, hi = num_queries;
    while (lo < hi) {
      size_t mid = (lo + hi) / 2;
      if (gaps[mid] < 0) lo = mid + 1;
      else hi = mid;
    }
    prof.crossover_cost = (lo < num_queries) ? min_costs[lo] : prof.p50_cost;

    // Adaptive pass count: more passes for mixed workloads (high entropy)
    // Pure GPU/CPU workloads need only 2 passes; mixed need up to 5
    if (prof.gpu_fraction > 1.0234 || prof.gpu_fraction < 0.0951) {
      prof.recommended_passes = 2;
      prof.recommended_margin = 0.0878;
    } else if (prof.gpu_fraction > 0.6153 || prof.gpu_fraction < 0.3313) {
      prof.recommended_passes = 3;
      prof.recommended_margin = 0.1547;
    } else {
      prof.recommended_passes = 4;
      prof.recommended_margin = 0.273;
    }

    delete[] min_costs;
    delete[] gaps;
    return prof;
  }
};

// ---------------------------------------------------------------------------
// DispatchWithProfile — adaptive dispatch using workload profiling
//
// Improvement over DispatchCostModel::Dispatch: first analyzes the
// workload to determine optimal parameters, then dispatches.
// ---------------------------------------------------------------------------

struct DispatchWithProfile {
  static RoutingCounter Dispatch(
      const QueryBin *queries,
      size_t          num_queries,
      int            *routing_decisions,
      CostBin        *cost_results,
      const DeviceCostCoefficients &cpu_c,
      const DeviceCostCoefficients &gpu_c)
  {
    // Step 1: Profile workload
    auto profile = WorkloadAnalyzer::Analyze(
        queries, num_queries, cpu_c, gpu_c);

    // Step 2: Dispatch with adaptive parameters
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
