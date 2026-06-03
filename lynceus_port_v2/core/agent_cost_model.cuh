/*
 * Lynceus — agent_cost_model.cuh — Per-thread-block agent for cost estimation.
 *
 * Direct analog of CCCL cub/cub/agent/agent_topk.cuh.
 * References:
 *   CCCL agent_topk.cuh:447 — filter_and_histogram
 *   CCCL agent_topk.cuh:process_range — tile-based item processing
 *   Tabular btree_common.h — BTreeLeafNode::lowerBound (binary search)
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdio>

#include "dispatch_cost_model.cuh"

#ifndef LYNCEUS_AGENT_DBG
#define LYNCEUS_AGENT_DBG 0
#endif

#if LYNCEUS_AGENT_DBG
  #define AGT_TRACE(fmt, ...) fprintf(stderr, "[AGT·DBG] " fmt "\n", ##__VA_ARGS__)
#else
  #define AGT_TRACE(fmt, ...) ((void)0)
#endif

namespace lynceus {
namespace core {

// --- Tile configuration ---
struct AgentConfig {
  static constexpr int BLOCK_THREADS    = 256;
  static constexpr int ITEMS_PER_THREAD = 4;
  static constexpr int ITEMS_PER_TILE   = BLOCK_THREADS * ITEMS_PER_THREAD;
  static constexpr int NUM_BINS         = 256;
};

// --- TempStorage: 改写加 EMA 跟踪 ---
struct TempStorage {
  uint64_t histogram[AgentConfig::NUM_BINS];
  double   cpu_costs[AgentConfig::ITEMS_PER_TILE];
  double   gpu_costs[AgentConfig::ITEMS_PER_TILE];
  uint32_t selected_count;
  uint32_t candidate_count;
  double   block_min_cost;
  double   block_max_cost;
  double   block_total_cost;
  // 改写新增: EMA 跟踪代价漂移
  double   ema_cost;
  double   ema_alpha;
  uint32_t tiles_processed;

  void reset() {
    std::memset(histogram, 0, sizeof(histogram));
    std::memset(cpu_costs, 0, sizeof(cpu_costs));
    std::memset(gpu_costs, 0, sizeof(gpu_costs));
    selected_count = candidate_count = 0;
    block_min_cost = std::numeric_limits<double>::max();
    block_max_cost = block_total_cost = 0.0;
    ema_cost = 0.0;
    ema_alpha = 0.05;   // 20-tile 窗口
    tiles_processed = 0;
  }

  void dump(const char *label = "") const {
    fprintf(stderr, "[TEMP·STORAGE] %s sel=%u cand=%u min=%.2f max=%.2f "
            "total=%.2f ema=%.2f tiles=%u\n",
            label, selected_count, candidate_count,
            block_min_cost, block_max_cost, block_total_cost,
            ema_cost, tiles_processed);
  }
};

// --- AgentCostModel ---
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

  // 改写: process_range 加 tile 级 EMA 更新 + 诊断
  template <typename Func>
  void process_range(const QueryBin *queries, size_t len, Func f) {
    for (size_t offset = 0; offset < len;
         offset += AgentConfig::ITEMS_PER_TILE) {
      size_t tile_size = std::min(
          static_cast<size_t>(AgentConfig::ITEMS_PER_TILE), len - offset);

      double tile_cost_sum = 0.0;
      for (size_t i = 0; i < tile_size; ++i) {
        f(queries[offset + i], offset + i);
        // 估算本 tile 代价用于 EMA
        double cpu = estimate_cpu_cost(queries[offset + i], *cpu_coeffs);
        double gpu = estimate_gpu_cost(queries[offset + i], *gpu_coeffs);
        tile_cost_sum += std::min(cpu, gpu);
      }

      // EMA 更新
      double tile_avg = tile_cost_sum / std::max<size_t>(1, tile_size);
      if (temp_storage.tiles_processed == 0) {
        temp_storage.ema_cost = tile_avg;
      } else {
        temp_storage.ema_cost = temp_storage.ema_alpha * tile_avg
            + (1.0 - temp_storage.ema_alpha) * temp_storage.ema_cost;
      }
      temp_storage.tiles_processed++;
      AGT_TRACE("tile @%zu: avg=%.2f ema=%.2f", offset, tile_avg, temp_storage.ema_cost);
    }
  }

  // --- histogram_only ---
  void histogram_only(const QueryBin *queries, size_t num_queries,
                      uint64_t *global_histogram) {
    temp_storage.reset();

    auto f = [this](const QueryBin &q, size_t /*idx*/) {
      double cpu = estimate_cpu_cost(q, *cpu_coeffs);
      double gpu = estimate_gpu_cost(q, *gpu_coeffs);
      double min_cost = std::min(cpu, gpu);

      int bin = (*extract_op)(min_cost);
      temp_storage.histogram[bin]++;

      temp_storage.block_min_cost = std::min(temp_storage.block_min_cost, min_cost);
      temp_storage.block_max_cost = std::max(temp_storage.block_max_cost, min_cost);
      temp_storage.block_total_cost += min_cost;
    };

    process_range(queries, num_queries, f);

    for (int b = 0; b < AgentConfig::NUM_BINS; ++b) {
      global_histogram[b] += temp_storage.histogram[b];
    }
    AGT_TRACE("histogram_only: %zu queries, range [%.2f, %.2f]",
              num_queries, temp_storage.block_min_cost, temp_storage.block_max_cost);
  }

  // --- filter_and_estimate ---
  void filter_and_estimate(
      const QueryBin *in_buf, size_t in_count,
      QueryBin *out_buf, size_t *out_count,
      int *routing_decisions, CostBin *cost_results,
      uint64_t *global_histogram,
      const IdentifyCandidates &identify,
      RoutingCounter *counter, bool is_last_pass)
  {
    temp_storage.reset();
    size_t local_out = 0;

    auto f = [&](const QueryBin &q, size_t /*tile_idx*/) {
      double cpu = estimate_cpu_cost(q, *cpu_coeffs);
      double gpu = estimate_gpu_cost(q, *gpu_coeffs);

      CandidateClass cls = identify(cpu, gpu);

      if (cls == CandidateClass::selected || is_last_pass) {
        bool use_gpu = (gpu < cpu);
        routing_decisions[q.query_id] = use_gpu ? 1 : 0;

        // 改写: 分解代价而非全压 compute
        double chosen = use_gpu ? gpu : cpu;
        cost_results[q.query_id] = {
          chosen * 0.40, chosen * 0.35,
          use_gpu ? chosen * 0.15 : 0.0,
          chosen * 0.05, chosen * 0.05
        };

        if (use_gpu) counter->queries_routed_gpu++;
        else         counter->queries_routed_cpu++;
        counter->total_cost_us += chosen;
        temp_storage.selected_count++;
      } else {
        out_buf[local_out++] = q;
        double min_cost = std::min(cpu, gpu);
        int bin = (*extract_op)(min_cost);
        temp_storage.histogram[bin]++;
        temp_storage.candidate_count++;
      }
    };

    process_range(in_buf, in_count, f);

    *out_count = local_out;
    for (int b = 0; b < AgentConfig::NUM_BINS; ++b) {
      global_histogram[b] += temp_storage.histogram[b];
    }
    counter->filter_pass_count++;

    AGT_TRACE("filter: in=%zu sel=%u cand=%u last=%d ema=%.2f",
              in_count, temp_storage.selected_count,
              temp_storage.candidate_count, is_last_pass, temp_storage.ema_cost);
    temp_storage.dump("post-filter");
  }

  // --- accumulate_tile: 改写加 Welford 方差 ---
  struct TileStats {
    double min_cpu, max_cpu, sum_cpu;
    double min_gpu, max_gpu, sum_gpu;
    size_t count;
    size_t gpu_winners;
    // 改写新增
    double variance_cpu, variance_gpu;

    double mean_cpu() const { return count > 0 ? sum_cpu / count : 0; }
    double mean_gpu() const { return count > 0 ? sum_gpu / count : 0; }
    double gpu_win_ratio() const { return count > 0 ? static_cast<double>(gpu_winners) / count : 0; }

    void dump() const {
      fprintf(stderr, "[TILE] n=%zu gpu_win=%.2f cpu_mean=%.2f(var=%.2f) gpu_mean=%.2f(var=%.2f)\n",
              count, gpu_win_ratio(), mean_cpu(), variance_cpu, mean_gpu(), variance_gpu);
    }
  };

  TileStats accumulate_tile(const QueryBin *queries, size_t tile_start,
                            size_t tile_size) {
    TileStats stats = {};
    stats.min_cpu = stats.min_gpu = std::numeric_limits<double>::max();
    stats.max_cpu = stats.max_gpu = 0;
    stats.count = tile_size;

    // Welford 在线方差
    double w_mean_c = 0, w_m2_c = 0;
    double w_mean_g = 0, w_m2_g = 0;

    for (size_t i = 0; i < tile_size; ++i) {
      const auto &q = queries[tile_start + i];
      double cpu = estimate_cpu_cost(q, *cpu_coeffs);
      double gpu = estimate_gpu_cost(q, *gpu_coeffs);

      if (i < static_cast<size_t>(AgentConfig::ITEMS_PER_TILE)) {
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

      double n1 = static_cast<double>(i + 1);
      double dc = cpu - w_mean_c; w_mean_c += dc / n1; w_m2_c += dc * (cpu - w_mean_c);
      double dg = gpu - w_mean_g; w_mean_g += dg / n1; w_m2_g += dg * (gpu - w_mean_g);
    }

    stats.variance_cpu = (tile_size > 1) ? w_m2_c / (tile_size - 1) : 0.0;
    stats.variance_gpu = (tile_size > 1) ? w_m2_g / (tile_size - 1) : 0.0;

    AGT_TRACE("accumulate_tile @%zu: %zu items", tile_start, tile_size);
    stats.dump();
    return stats;
  }

  // --- 改写: find_cost_threshold 用插值替代 midpoint ---
  static double find_cost_threshold(const uint64_t *histogram,
                                    int num_bins,
                                    double bin_min, double bin_width,
                                    uint64_t k) {
    uint64_t cumulative = 0;
    for (int b = 0; b < num_bins; ++b) {
      uint64_t prev_cum = cumulative;
      cumulative += histogram[b];
      if (cumulative >= k) {
        // 改写: 线性插值到 bin 内精确位置, 不用 midpoint
        double frac = (histogram[b] > 0)
          ? static_cast<double>(k - prev_cum) / histogram[b]
          : 0.5;
        double threshold = bin_min + (b + frac) * bin_width;
        AGT_TRACE("threshold: k=%lu bin=%d frac=%.3f → %.2f", k, b, frac, threshold);
        return threshold;
      }
    }
    return bin_min + num_bins * bin_width;
  }

  // --- classify_workload ---
  struct WorkloadProfile {
    size_t total_queries;
    size_t gpu_preferred;
    size_t cpu_preferred;
    size_t ambiguous;
    double total_cpu_cost;
    double total_gpu_cost;
    double total_routed_cost;
    double gpu_fraction;
    double p50_cost;
    double p95_cost;
    // 改写新增
    double iqr;  // 四分位距

    void dump() const {
      fprintf(stderr, "[WKLD] n=%zu gpu=%zu cpu=%zu amb=%zu frac=%.2f "
              "p50=%.2f p95=%.2f iqr=%.2f\n",
              total_queries, gpu_preferred, cpu_preferred, ambiguous,
              gpu_fraction, p50_cost, p95_cost, iqr);
    }
  };

  WorkloadProfile classify_workload(const QueryBin *queries,
                                    size_t num_queries,
                                    double margin = 0.20) {
    WorkloadProfile profile = {};
    profile.total_queries = num_queries;

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

    std::sort(min_costs, min_costs + num_queries);
    if (num_queries > 0) {
      profile.p50_cost = min_costs[num_queries / 2];
      profile.p95_cost = min_costs[num_queries * 95 / 100];
      // 改写: 计算 IQR
      double q1 = min_costs[num_queries / 4];
      double q3 = min_costs[num_queries * 3 / 4];
      profile.iqr = q3 - q1;
    }

    profile.dump();
    delete[] min_costs;
    return profile;
  }
};

// --- AgentDispatch ---
struct AgentDispatch {
  static RoutingCounter DispatchWithAgent(
      const QueryBin *queries, size_t num_queries,
      int *routing_decisions, CostBin *cost_results,
      const DeviceCostCoefficients &cpu_c,
      const DeviceCostCoefficients &gpu_c,
      double margin = 0.20, int num_passes = 3)
  {
    RoutingCounter counter;
    counter.reset();

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

    auto *buf_a = new QueryBin[num_queries];
    auto *buf_b = new QueryBin[num_queries];
    DoubleBuffer<QueryBin> bufs(buf_a, buf_b);

    uint64_t histogram[AgentConfig::NUM_BINS] = {};

    agent.histogram_only(queries, num_queries, histogram);
    counter.histogram_pass_count++;

    std::memcpy(bufs.Current(), queries, num_queries * sizeof(QueryBin));
    size_t current_count = num_queries;

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

    AGT_TRACE("DispatchWithAgent done: ema_cost=%.2f tiles=%u",
              agent.temp_storage.ema_cost, agent.temp_storage.tiles_processed);
    delete[] buf_a;
    delete[] buf_b;
    return counter;
  }
};

} // namespace core
} // namespace lynceus
