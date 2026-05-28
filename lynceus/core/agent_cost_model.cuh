/*
 * Lynceus — Cost-Model-Driven Query Routing for Heterogeneous GPU-CPU Systems
 *
 * agent_cost_model.cuh — Per-thread-block agent for cost estimation.
 *
 * Direct analog of CCCL cub/cub/agent/agent_topk.cuh:
 *   AgentTopK processes tiles of items per thread block, building
 *   per-block histograms and filtering candidates.
 *   AgentCostModel processes tiles of queries per thread block,
 *   building per-block cost histograms and classifying queries
 *   into CPU-bound / GPU-bound / candidate partitions.
 *
 * The key structure from CCCL agent_topk.cuh:
 *   struct AgentTopK {
 *     temp_storage.histogram[NUM_BINS]       // shared memory histogram
 *     process_range(in, len, func)           // tile-based streaming
 *     filter_and_histogram(...)              // fused classify + histogram
 *   }
 *
 * Our structure:
 *   struct AgentCostModel {
 *     temp_storage.histogram[NUM_BINS]       // per-block cost histogram
 *     temp_storage.cpu_costs[ITEMS_PER_TILE] // tile-local cost cache
 *     temp_storage.gpu_costs[ITEMS_PER_TILE]
 *     process_range(queries, len, func)      // tile-based streaming
 *     histogram_only(...)                    // pass 0 (extracted, per f984c90)
 *     filter_and_estimate(...)               // passes 1..N (fused)
 *   }
 *
 * References:
 *   CCCL agent_topk.cuh:447 — filter_and_histogram (post-extraction)
 *   CCCL agent_topk.cuh:process_range — tile-based item processing
 *   CCCL dispatch_topk.cuh — integration with DispatchTopK
 *   Tabular btree_common.h — BTreeLeafNode::lowerBound (binary search)
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <algorithm>
#include <cmath>

#include "dispatch_cost_model.cuh"

namespace lynceus {
namespace core {

// ---------------------------------------------------------------------------
// Tile configuration (analogous to CCCL's Policy template parameters)
//
// CCCL uses policy-based design:
//   template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
//   struct AgentTopKPolicy { ... };
//
// We use a constexpr config struct (simpler, same effect for CPU dispatch).
// ---------------------------------------------------------------------------

struct AgentConfig {
  static constexpr int BLOCK_THREADS    = 256;   // threads per block
  static constexpr int ITEMS_PER_THREAD = 4;     // items per thread
  static constexpr int ITEMS_PER_TILE   = BLOCK_THREADS * ITEMS_PER_THREAD;
  static constexpr int NUM_BINS         = 256;   // histogram bins
};

// ---------------------------------------------------------------------------
// TempStorage — shared memory layout for one thread block
//
// CCCL agent_topk.cuh uses union-based shared memory:
//   union TempStorage {
//     OffsetT histogram[NUM_BINS];
//     ... other views ...
//   };
//
// We use a struct (cleaner for CPU emulation; on GPU this would be __shared__).
// ---------------------------------------------------------------------------

struct TempStorage {
  uint64_t histogram[AgentConfig::NUM_BINS];
  double   cpu_costs[AgentConfig::ITEMS_PER_TILE];
  double   gpu_costs[AgentConfig::ITEMS_PER_TILE];
  uint32_t selected_count;
  uint32_t candidate_count;
  double   block_min_cost;
  double   block_max_cost;
  double   block_total_cost;

  void reset() {
    std::memset(histogram, 0, sizeof(histogram));
    std::memset(cpu_costs, 0, sizeof(cpu_costs));
    std::memset(gpu_costs, 0, sizeof(gpu_costs));
    selected_count = 0;
    candidate_count = 0;
    block_min_cost = std::numeric_limits<double>::max();
    block_max_cost = 0.0;
    block_total_cost = 0.0;
  }
};

// ---------------------------------------------------------------------------
// AgentCostModel — the per-thread-block agent
//
// CCCL's AgentTopK has these key methods:
//   process_range(d_keys_in, len, func)     — stream over items in tiles
//   filter_and_histogram(...)               — fused classify + histogram
//
// Our AgentCostModel mirrors this with:
//   process_range(queries, len, func)       — stream over queries in tiles
//   histogram_only(queries, len, ...)       — pass 0 (extracted)
//   filter_and_estimate(queries, len, ...)  — passes 1..N (fused)
//   accumulate_tile(tile_start, tile_size)  — per-tile cost computation
//
// The fundamental pattern: iterate over input in fixed-size tiles,
// process each tile into the shared TempStorage, then reduce.
// ---------------------------------------------------------------------------

struct AgentCostModel {
  TempStorage temp_storage;
  const DeviceCostCoefficients *cpu_coeffs;
  const DeviceCostCoefficients *gpu_coeffs;
  const ExtractCostBin         *extract_op;

  AgentCostModel(const DeviceCostCoefficients *cpu_c,
                 const DeviceCostCoefficients *gpu_c,
                 const ExtractCostBin         *extract)
    : cpu_coeffs(cpu_c), gpu_coeffs(gpu_c), extract_op(extract) {
    temp_storage.reset();
  }

  // -----------------------------------------------------------------------
  // process_range — tile-based streaming over queries
  //
  // Direct analog of CCCL AgentTopK::process_range:
  //   template <typename Func>
  //   void process_range(key_in_t* in, OffsetT len, Func f) {
  //     for (OffsetT offset = blockIdx.x * ITEMS_PER_TILE;
  //          offset < len;
  //          offset += gridDim.x * ITEMS_PER_TILE) {
  //       auto tile_size = min(ITEMS_PER_TILE, len - offset);
  //       // load tile, apply f to each item
  //     }
  //   }
  //
  // CPU emulation: single "block" processes all tiles sequentially.
  // -----------------------------------------------------------------------

  template <typename Func>
  void process_range(const QueryBin *queries, size_t len, Func f) {
    for (size_t offset = 0; offset < len;
         offset += AgentConfig::ITEMS_PER_TILE) {
      size_t tile_size = std::min(
          static_cast<size_t>(AgentConfig::ITEMS_PER_TILE), len - offset);

      for (size_t i = 0; i < tile_size; ++i) {
        f(queries[offset + i], offset + i);
      }
    }
  }

  // -----------------------------------------------------------------------
  // histogram_only — dedicated first-pass histogram kernel
  //
  // CCCL f984c90 analog: this was the IsFirstPass=true path inside
  // filter_and_histogram, now extracted into its own method.
  //
  // Streams over all queries, computes min(cpu_cost, gpu_cost) per query,
  // maps to histogram bin via extract_op, and accumulates.
  //
  // CCCL code (before extraction):
  //   if constexpr (IsFirstPass) {
  //     auto f = [this](key_in_t key, OffsetT) {
  //       const int bucket = extract_bin_op(key);
  //       atomicAdd(temp_storage.histogram + bucket, OffsetT{1});
  //     };
  //     process_range(d_keys_in, previous_len, f);
  //   }
  //
  // Our code (after extraction, per f984c90):
  // -----------------------------------------------------------------------

  void histogram_only(const QueryBin *queries, size_t num_queries,
                      uint64_t *global_histogram) {
    temp_storage.reset();

    auto f = [this](const QueryBin &q, size_t /*idx*/) {
      double cpu = estimate_cpu_cost(q, *cpu_coeffs);
      double gpu = estimate_gpu_cost(q, *gpu_coeffs);
      double min_cost = std::min(cpu, gpu);

      // Accumulate into per-block histogram
      // (CCCL: atomicAdd(temp_storage.histogram + bucket, OffsetT{1}))
      int bin = (*extract_op)(min_cost);
      temp_storage.histogram[bin]++;

      // Track block-level statistics
      temp_storage.block_min_cost = std::min(temp_storage.block_min_cost, min_cost);
      temp_storage.block_max_cost = std::max(temp_storage.block_max_cost, min_cost);
      temp_storage.block_total_cost += min_cost;
    };

    process_range(queries, num_queries, f);

    // Reduce per-block histogram to global histogram
    // (CCCL: grid-level reduction via atomicAdd to global histogram)
    for (int b = 0; b < AgentConfig::NUM_BINS; ++b) {
      global_histogram[b] += temp_storage.histogram[b];
    }
  }

  // -----------------------------------------------------------------------
  // filter_and_estimate — fused filter + cost estimation (passes 1..N)
  //
  // CCCL analog: AgentTopK::filter_and_histogram after removing IsFirstPass.
  // No longer handles the first pass (that's histogram_only above).
  //
  // The three CCCL lambdas map to our code paths:
  //   f_early_stop    → is_last_pass: route all remaining, skip histogram
  //   f_with_out_buf  → !is_last_pass && candidate: write to out_buf + histogram
  //   f_no_out_buf    → NOT NEEDED (we always write candidates to out_buf)
  //
  // CCCL's buffer management:
  //   OffsetT* p_filter_cnt = &counter->filter_cnt;
  //   OutOffsetT* p_out_cnt = &counter->out_cnt;
  //
  // Our buffer management:
  //   size_t *out_count (candidate count for next pass)
  //   RoutingCounter *counter (global routing statistics)
  // -----------------------------------------------------------------------

  void filter_and_estimate(
      const QueryBin *in_buf,
      size_t          in_count,
      QueryBin       *out_buf,
      size_t         *out_count,
      int            *routing_decisions,
      CostBin        *cost_results,
      uint64_t       *global_histogram,
      const IdentifyCandidates &identify,
      RoutingCounter *counter,
      bool            is_last_pass)
  {
    temp_storage.reset();
    size_t local_out = 0;

    auto f = [&](const QueryBin &q, size_t /*tile_idx*/) {
      double cpu = estimate_cpu_cost(q, *cpu_coeffs);
      double gpu = estimate_gpu_cost(q, *gpu_coeffs);

      CandidateClass cls = identify(cpu, gpu);

      if (cls == CandidateClass::selected || is_last_pass) {
        // Selected: route decisively
        // (CCCL: f_early_stop lambda — write to output)
        bool use_gpu = (gpu < cpu);
        routing_decisions[q.query_id] = use_gpu ? 1 : 0;
        cost_results[q.query_id] = {0, use_gpu ? gpu : cpu, 0, 0, 0};

        if (use_gpu) counter->queries_routed_gpu++;
        else         counter->queries_routed_cpu++;
        counter->total_cost_us += (use_gpu ? gpu : cpu);

        temp_storage.selected_count++;
      } else {
        // Candidate: write to out_buf for next pass
        // (CCCL: f_with_out_buf lambda — write candidate + build histogram)
        out_buf[local_out++] = q;

        double min_cost = std::min(cpu, gpu);
        int bin = (*extract_op)(min_cost);
        temp_storage.histogram[bin]++;

        temp_storage.candidate_count++;
      }
    };

    process_range(in_buf, in_count, f);

    *out_count = local_out;

    // Reduce per-block histogram to global
    for (int b = 0; b < AgentConfig::NUM_BINS; ++b) {
      global_histogram[b] += temp_storage.histogram[b];
    }

    counter->filter_pass_count++;
  }

  // -----------------------------------------------------------------------
  // accumulate_tile — compute and cache costs for a tile of queries
  //
  // Fills temp_storage.cpu_costs[] and temp_storage.gpu_costs[] for
  // a contiguous tile of queries. Useful for batch analysis before
  // classification (e.g., computing the cost distribution within a tile
  // to set adaptive thresholds).
  //
  // This is a Lynceus extension not in CCCL — it enables per-tile
  // statistics like mean/variance of costs, which the adaptive router
  // uses for EMA-based bias correction.
  // -----------------------------------------------------------------------

  struct TileStats {
    double min_cpu, max_cpu, sum_cpu;
    double min_gpu, max_gpu, sum_gpu;
    size_t count;
    size_t gpu_winners;   // count where gpu < cpu

    double mean_cpu() const { return count > 0 ? sum_cpu / count : 0; }
    double mean_gpu() const { return count > 0 ? sum_gpu / count : 0; }
    double gpu_win_ratio() const { return count > 0 ? static_cast<double>(gpu_winners) / count : 0; }
  };

  TileStats accumulate_tile(const QueryBin *queries, size_t tile_start,
                            size_t tile_size) {
    TileStats stats = {};
    stats.min_cpu = stats.min_gpu = std::numeric_limits<double>::max();
    stats.max_cpu = stats.max_gpu = 0;
    stats.count = tile_size;

    for (size_t i = 0; i < tile_size; ++i) {
      const auto &q = queries[tile_start + i];
      double cpu = estimate_cpu_cost(q, *cpu_coeffs);
      double gpu = estimate_gpu_cost(q, *gpu_coeffs);

      if (i < AgentConfig::ITEMS_PER_TILE) {
        temp_storage.cpu_costs[i] = cpu;
        temp_storage.gpu_costs[i] = gpu;
      }

      stats.sum_cpu += cpu;
      stats.sum_gpu += gpu;
      stats.min_cpu = std::min(stats.min_cpu, cpu);
      stats.max_cpu = std::max(stats.max_cpu, cpu);
      stats.min_gpu = std::min(stats.min_gpu, gpu);
      stats.max_gpu = std::max(stats.max_gpu, gpu);
      if (gpu < cpu) stats.gpu_winners++;
    }

    return stats;
  }

  // -----------------------------------------------------------------------
  // find_cost_threshold — binary search over histogram to find the cost
  // at which exactly k queries have lower cost (the "k-th element").
  //
  // Analog of CCCL's TopK finding the k-th largest element via histogram:
  // after building the histogram, scan bins to find which bin contains
  // the k-th element, then refine within that bin.
  //
  // Uses BTreeLeafNode::lowerBound-style binary search (from Tabular).
  // -----------------------------------------------------------------------

  static double find_cost_threshold(const uint64_t *histogram,
                                    int num_bins,
                                    double bin_min, double bin_width,
                                    uint64_t k) {
    uint64_t cumulative = 0;
    for (int b = 0; b < num_bins; ++b) {
      cumulative += histogram[b];
      if (cumulative >= k) {
        // k-th element is in this bin
        // Return the midpoint of the bin
        return bin_min + (b + 0.5) * bin_width;
      }
    }
    return bin_min + num_bins * bin_width; // beyond last bin
  }

  // -----------------------------------------------------------------------
  // classify_workload — analyze a full workload and return routing statistics
  //
  // Runs histogram_only() followed by workload analysis to determine:
  //   - What fraction of queries should go to GPU vs CPU
  //   - The optimal cost threshold for splitting
  //   - The expected total cost under optimal routing
  //
  // This is a Lynceus-specific operation not in CCCL.
  // -----------------------------------------------------------------------

  struct WorkloadProfile {
    size_t total_queries;
    size_t gpu_preferred;    // queries where gpu_cost < cpu_cost
    size_t cpu_preferred;    // queries where cpu_cost < gpu_cost
    size_t ambiguous;        // queries within robustness margin
    double total_cpu_cost;   // total cost if all go to CPU
    double total_gpu_cost;   // total cost if all go to GPU
    double total_routed_cost; // total cost under optimal routing
    double gpu_fraction;     // fraction of queries → GPU
    double p50_cost;         // median query cost (min of cpu/gpu)
    double p95_cost;         // 95th percentile
  };

  WorkloadProfile classify_workload(const QueryBin *queries,
                                    size_t num_queries,
                                    double margin = 0.20) {
    WorkloadProfile profile = {};
    profile.total_queries = num_queries;

    // Collect all min-costs for histogram
    double *min_costs = new double[num_queries];

    for (size_t i = 0; i < num_queries; ++i) {
      double cpu = estimate_cpu_cost(queries[i], *cpu_coeffs);
      double gpu = estimate_gpu_cost(queries[i], *gpu_coeffs);

      min_costs[i] = std::min(cpu, gpu);
      profile.total_cpu_cost += cpu;
      profile.total_gpu_cost += gpu;
      profile.total_routed_cost += std::min(cpu, gpu);

      if (gpu < cpu * (1.0 - margin))      profile.gpu_preferred++;
      else if (cpu < gpu * (1.0 - margin))  profile.cpu_preferred++;
      else                                  profile.ambiguous++;
    }

    profile.gpu_fraction = static_cast<double>(profile.gpu_preferred) /
                           std::max<size_t>(1, num_queries);

    // Sort for percentiles
    std::sort(min_costs, min_costs + num_queries);
    if (num_queries > 0) {
      profile.p50_cost = min_costs[num_queries / 2];
      profile.p95_cost = min_costs[num_queries * 95 / 100];
    }

    delete[] min_costs;
    return profile;
  }
};

// ---------------------------------------------------------------------------
// AgentDispatch — integrates AgentCostModel with DispatchCostModel
//
// This is the glue between the agent (per-block processing) and the
// dispatch (multi-pass orchestration). In CCCL, the kernel function
// instantiates AgentTopK and calls its methods; here we provide a
// convenience function that creates the agent and runs the full pipeline.
// ---------------------------------------------------------------------------

struct AgentDispatch {
  static RoutingCounter DispatchWithAgent(
      const QueryBin *queries,
      size_t          num_queries,
      int            *routing_decisions,
      CostBin        *cost_results,
      const DeviceCostCoefficients &cpu_c,
      const DeviceCostCoefficients &gpu_c,
      double          margin = 0.20,
      int             num_passes = 3)
  {
    RoutingCounter counter;
    counter.reset();

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

    ExtractCostBin extract(AgentConfig::NUM_BINS, min_cost, max_cost);
    AgentCostModel agent(&cpu_c, &gpu_c, &extract);

    // DoubleBuffer for candidate refinement
    auto *buf_a = new QueryBin[num_queries];
    auto *buf_b = new QueryBin[num_queries];
    DoubleBuffer<QueryBin> bufs(buf_a, buf_b);

    uint64_t histogram[AgentConfig::NUM_BINS] = {};

    // Pass 0: histogram-only (extracted kernel)
    agent.histogram_only(queries, num_queries, histogram);
    counter.histogram_pass_count++;

    std::memcpy(bufs.Current(), queries, num_queries * sizeof(QueryBin));
    size_t current_count = num_queries;

    // Passes 1..N: filter + estimate via agent
    for (int pass = 1; pass < num_passes; ++pass) {
      IdentifyCandidates identify(margin);
      std::memset(histogram, 0, sizeof(histogram));

      size_t out_count = 0;
      agent.filter_and_estimate(
          bufs.Current(), current_count,
          bufs.Alternate(), &out_count,
          routing_decisions, cost_results,
          histogram, identify, &counter,
          pass == num_passes - 1);

      bufs.Swap();
      current_count = out_count;
      if (current_count == 0) break;
    }

    delete[] buf_a;
    delete[] buf_b;
    return counter;
  }
};

} // namespace core
} // namespace lynceus
