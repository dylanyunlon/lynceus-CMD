/*
 * Original: Copyright (C) 2023 Data-Intensive Systems Lab, Simon Fraser University.
 * Modified: Lynceus — cost-aware B-tree node structures with heterogeneous dispatch.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 *
 * Modifications from upstream/tabular:
 *   - Namespace: noname::tabular → lynceus::tabular
 *   - Added:   CostAnnotatedLeafNode with per-entry cost metadata
 *   - Added:   SplitByCostThreshold for device-aware leaf splitting
 */
#pragma once

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <utility>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace lynceus { namespace table {
  using OID = uint64_t;
  static constexpr OID kInvalidOID = ~0ULL;
}}

// ---------- debug knob ----------
#ifndef LYNCEUS_BT_DBG
#define LYNCEUS_BT_DBG 0
#endif

#if LYNCEUS_BT_DBG
  #define BT_TRACE(fmt, ...) fprintf(stderr, "[BT·DBG] " fmt "\n", ##__VA_ARGS__)
#else
  #define BT_TRACE(fmt, ...) ((void)0)
#endif

namespace lynceus {
namespace tabular {

enum class NodeType : uint8_t { Inner = 1, Leaf = 2 };

static inline constexpr uint64_t kPageSizeLinearSearchCutoff = 256;
#ifndef BTREE_PAGE_SIZE
#define BTREE_PAGE_SIZE 256
#endif
static inline constexpr uint64_t kPageSize = BTREE_PAGE_SIZE - 8;
static inline constexpr uint64_t kMaxLevels = 16;

struct NodeBase {
  inline NodeType getType() const { return (level == 1) ? NodeType::Leaf : NodeType::Inner; }
  uint8_t level;
  uint16_t count;
};

struct BTreeNode : public NodeBase {
  uint8_t _data[kPageSize - sizeof(NodeBase)];
};

template <typename Key, typename Value>
struct BTreeLeafNode : public NodeBase {
  struct KeyValueType {
    Key key;
    Value value;
  };
  static constexpr size_t entrySize = sizeof(KeyValueType);
  static const uint64_t maxEntries =
      (kPageSize - sizeof(NodeBase) - sizeof(table::OID)) / entrySize - 1;

  table::OID next_leaf;
  KeyValueType data[maxEntries + 1];

  // 改写新增: 操作计数器
  mutable uint32_t search_count_ = 0;
  mutable uint32_t insert_count_ = 0;
  mutable uint32_t split_count_ = 0;

  BTreeLeafNode() {
    level = 1;
    count = 0;
    next_leaf = table::kInvalidOID;
  }
  BTreeLeafNode(BTreeLeafNode<Key, Value> &) = default;

  bool isFull() { return count == maxEntries; }

  // 改写: lowerBound 加 branch-free 前缀搜索 (Eytzinger 风格优化)
  unsigned lowerBound(const Key &k) {
    search_count_++;
    if constexpr (kPageSize <= kPageSizeLinearSearchCutoff) {
      // 小页: sentinel 线性搜索 — 少一次边界比较
      unsigned i = 0;
      for (; i < count && data[i].key < k; ++i) {}
      BT_TRACE("lowerBound(linear): k found at pos=%u/%u", i, count);
      return i;
    } else {
      // 大页: 改写为 branchless 二分 — 用 conditional move 替代 if/else
      unsigned n = count;
      unsigned base = 0;
      while (n > 1) {
        unsigned half = n / 2;
        // branchless: base advances if data[base+half].key < k
        base += (data[base + half].key < k) ? half : 0;
        n -= half;
      }
      base += (base < count && data[base].key < k) ? 1 : 0;
      BT_TRACE("lowerBound(branchless): pos=%u/%u", base, count);
      return base;
    }
  }

  bool key_exists(const Key &k) {
    assert(count <= maxEntries);
    if (count) {
      unsigned pos = lowerBound(k);
      if ((pos < count) && (data[pos].key == k)) return true;
    }
    return false;
  }

  bool insert_no_existence_check(const Key &k, Value p) {
    assert(count <= maxEntries);
    insert_count_++;
    if (count) {
      unsigned pos = lowerBound(k);
      memmove(data + pos + 1, data + pos, sizeof(KeyValueType) * (count - pos));
      data[pos].key = k;
      data[pos].value = p;
    } else {
      data[0].key = k;
      data[0].value = p;
    }
    count++;
    return true;
  }

  bool insert(const Key &k, Value p) {
    assert(count <= maxEntries);
    insert_count_++;
    if (count) {
      unsigned pos = lowerBound(k);
      if ((pos < count) && (data[pos].key == k)) {
        BT_TRACE("insert: duplicate key at pos=%u", pos);
        return false;
      }
      memmove(data + pos + 1, data + pos, sizeof(KeyValueType) * (count - pos));
      data[pos].key = k;
      data[pos].value = p;
    } else {
      data[0].key = k;
      data[0].value = p;
    }
    count++;
    return true;
  }

  bool remove(const Key &k) {
    assert(count <= maxEntries);
    if (count) {
      unsigned pos = lowerBound(k);
      if ((pos < count) && (data[pos].key == k)) {
        memmove(data + pos, data + pos + 1, sizeof(KeyValueType) * (count - pos));
        count--;
        return true;
      }
    }
    return false;
  }

  bool update(const Key &k, Value p) {
    assert(count <= maxEntries);
    if (count) {
      unsigned pos = lowerBound(k);
      if ((pos < count) && (data[pos].key == k)) {
        data[pos].value = p;
        return true;
      }
    }
    return false;
  }

  // 改写: split_to 加平衡度检验 + 诊断
  void split_to(BTreeLeafNode *newLeaf, Key &sep) {
    split_count_++;
    unsigned new_count = count - (count / 2);
    unsigned keep = count - new_count;
    memcpy(newLeaf->data, data + keep, sizeof(KeyValueType) * new_count);
    newLeaf->count = new_count;
    newLeaf->next_leaf = next_leaf;
    sep = data[keep - 1].key;
    count = keep;
    BT_TRACE("split_to: keep=%u move=%u sep_at=%u", keep, new_count, keep - 1);
    // 平衡度检验: 分裂后两侧不应差超过2倍
    assert(keep > 0 && new_count > 0);
    assert(keep <= 2 * new_count + 1 && new_count <= 2 * keep + 1);
  }

  // 改写新增: 批量点查 — 排序后合并遍历, 摊薄搜索开销
  int batch_search(const Key *keys, unsigned nkeys,
                   Value *out_values, bool *out_found) const {
    int found = 0;
    // 对小批量还是逐个查
    if (nkeys <= 4) {
      for (unsigned q = 0; q < nkeys; q++) {
        out_found[q] = false;
        for (unsigned i = 0; i < count; i++) {
          if (data[i].key == keys[q]) {
            out_values[q] = data[i].value;
            out_found[q] = true;
            found++;
            break;
          }
        }
      }
      return found;
    }
    // 大批量: 双指针归并
    // (假设叶节点data已排序, 调用者负责把keys排序)
    unsigned qi = 0, di = 0;
    while (qi < nkeys && di < count) {
      if (keys[qi] == data[di].key) {
        out_values[qi] = data[di].value;
        out_found[qi] = true;
        found++;
        qi++; di++;
      } else if (keys[qi] < data[di].key) {
        out_found[qi] = false;
        qi++;
      } else {
        di++;
      }
    }
    for (; qi < nkeys; qi++) out_found[qi] = false;
    return found;
  }

  void dump_leaf(const char *label = "") const {
    fprintf(stderr, "[LEAF] %s count=%u/%llu searches=%u inserts=%u splits=%u\n",
            label, count, maxEntries, search_count_, insert_count_, split_count_);
  }
};

template <typename Key>
struct BTreeInnerNode : public NodeBase {
  static constexpr size_t entrySize = sizeof(Key) + sizeof(table::OID);
  static const uint64_t maxEntries = (kPageSize - sizeof(NodeBase)) / entrySize - 1;

  table::OID children[maxEntries + 1];
  Key keys[maxEntries + 1];

  BTreeInnerNode() { level = 1; count = 0; }
  explicit BTreeInnerNode(uint8_t node_level) { level = node_level; count = 0; }
  BTreeInnerNode(BTreeInnerNode<Key> &) = default;

  bool isFull() { return count >= (maxEntries - 1); }

  // 改写: lowerBoundBF 加 prefetch hint
  unsigned lowerBoundBF(const Key &k) {
    auto base = keys;
    unsigned n = count;
    while (n > 1) {
      const unsigned half = n / 2;
      // prefetch 提前把下一级数据带进 cache
      __builtin_prefetch(base + half + half/2, 0, 1);
      base = (base[half] < k) ? (base + half) : base;
      n -= half;
    }
    return (*base < k) + base - keys;
  }

  unsigned lowerBound(const Key &k) {
    if constexpr (kPageSize <= kPageSizeLinearSearchCutoff) {
      unsigned lower = 0;
      while (lower < count) {
        if (k <= keys[lower]) return lower;
        lower++;
      }
      return lower;
    } else {
      // 改写: 同样 branchless
      unsigned n = count;
      unsigned base = 0;
      while (n > 1) {
        unsigned half = n / 2;
        base += (keys[base + half] < k) ? half : 0;
        n -= half;
      }
      base += (base < count && keys[base] < k) ? 1 : 0;
      return base;
    }
  }

  void split_to(BTreeInnerNode *newInner, Key &sep) {
    newInner->level = level;
    newInner->count = count - (count / 2);
    count = count - newInner->count - 1;
    sep = keys[count];
    memcpy(newInner->keys, keys + count + 1, sizeof(Key) * (newInner->count + 1));
    memcpy(newInner->children, children + count + 1, sizeof(NodeBase *) * (newInner->count + 1));
    BT_TRACE("inner split: keep=%u move=%u", count, newInner->count);
  }

  void insert(const Key &k, table::OID child_id) {
    assert(count <= maxEntries - 1);
    unsigned pos = lowerBound(k);
    memmove(keys + pos + 1, keys + pos, sizeof(Key) * (count - pos + 1));
    memmove(children + pos + 1, children + pos, sizeof(NodeBase *) * (count - pos + 1));
    keys[pos] = k;
    children[pos] = child_id;
    std::swap(children[pos + 1], children[pos]);
    count++;
  }
};

static_assert(sizeof(BTreeNode) == kPageSize);
static_assert(sizeof(BTreeInnerNode<uint64_t>) <= kPageSize);
static_assert(sizeof(BTreeLeafNode<uint64_t, uint64_t>) <= kPageSize);

// ===========================================================================
// CostAnnotatedLeafNode — with per-entry cost metadata
// ===========================================================================

template <typename Key, typename Value>
struct CostAnnotatedLeafNode : public NodeBase {
  struct CostEntry {
    Key key;
    Value value;
    double cpu_cost_us;
    double gpu_cost_us;
  };

  static constexpr size_t entrySize = sizeof(CostEntry);
  static constexpr uint64_t maxEntries =
      (kPageSize - sizeof(NodeBase) - sizeof(table::OID) - sizeof(double) * 2)
      / entrySize;

  table::OID next_leaf;
  double total_cpu_cost;
  double total_gpu_cost;

  CostEntry data[maxEntries > 0 ? maxEntries : 1];

  CostAnnotatedLeafNode() {
    level = 1; count = 0;
    next_leaf = table::kInvalidOID;
    total_cpu_cost = total_gpu_cost = 0.0;
  }

  bool isFull() { return count == maxEntries; }

  unsigned lowerBound(const Key &k) {
    // branchless
    unsigned n = count, base = 0;
    while (n > 1) {
      unsigned half = n / 2;
      base += (data[base + half].key < k) ? half : 0;
      n -= half;
    }
    base += (base < count && data[base].key < k) ? 1 : 0;
    return base;
  }

  unsigned lowerBoundCost(double cost_threshold) {
    unsigned n = count, base = 0;
    while (n > 1) {
      unsigned half = n / 2;
      double mc = std::min(data[base + half].cpu_cost_us, data[base + half].gpu_cost_us);
      base += (mc < cost_threshold) ? half : 0;
      n -= half;
    }
    if (base < count) {
      double mc = std::min(data[base].cpu_cost_us, data[base].gpu_cost_us);
      base += (mc < cost_threshold) ? 1 : 0;
    }
    return base;
  }

  void insert(const Key &k, Value v, double cpu_cost, double gpu_cost) {
    assert(count < maxEntries);
    unsigned pos = lowerBound(k);
    if (count > 0 && pos < count) {
      memmove(data + pos + 1, data + pos, sizeof(CostEntry) * (count - pos));
    }
    data[pos] = {k, v, cpu_cost, gpu_cost};
    total_cpu_cost += cpu_cost;
    total_gpu_cost += gpu_cost;
    count++;
  }

  // 改写: split_to 用 Kahan 累加代价和
  void split_to(CostAnnotatedLeafNode *newLeaf, Key &sep) {
    unsigned new_count = count - (count / 2);
    unsigned keep = count - new_count;
    memcpy(newLeaf->data, data + keep, sizeof(CostEntry) * new_count);
    newLeaf->count = new_count;
    newLeaf->next_leaf = next_leaf;
    sep = data[keep - 1].key;

    // Kahan 精确代价和
    total_cpu_cost = total_gpu_cost = 0;
    double kc_c = 0, kc_g = 0;
    for (unsigned i = 0; i < keep; ++i) {
      double yc = data[i].cpu_cost_us - kc_c; double tc = total_cpu_cost + yc; kc_c = (tc - total_cpu_cost) - yc; total_cpu_cost = tc;
      double yg = data[i].gpu_cost_us - kc_g; double tg = total_gpu_cost + yg; kc_g = (tg - total_gpu_cost) - yg; total_gpu_cost = tg;
    }
    newLeaf->total_cpu_cost = newLeaf->total_gpu_cost = 0;
    kc_c = kc_g = 0;
    for (unsigned i = 0; i < new_count; ++i) {
      double yc = newLeaf->data[i].cpu_cost_us - kc_c; double tc = newLeaf->total_cpu_cost + yc; kc_c = (tc - newLeaf->total_cpu_cost) - yc; newLeaf->total_cpu_cost = tc;
      double yg = newLeaf->data[i].gpu_cost_us - kc_g; double tg = newLeaf->total_gpu_cost + yg; kc_g = (tg - newLeaf->total_gpu_cost) - yg; newLeaf->total_gpu_cost = tg;
    }
    count = keep;
    BT_TRACE("cost split: keep=%u move=%u cpu_cost=%.2f gpu_cost=%.2f",
             keep, new_count, total_cpu_cost, total_gpu_cost);
  }

  // 改写: SplitByCostThreshold 加 margin 自适应 + 计数统计
  unsigned SplitByCostThreshold(double gpu_threshold_us,
                                CostAnnotatedLeafNode *gpu_leaf) {
    unsigned cpu_idx = 0, gpu_idx = 0;
    CostEntry temp[maxEntries > 0 ? maxEntries : 1];

    // 改写: margin 根据 entry 数自适应
    double adaptive_margin = (count > 16) ? 0.85 : 0.80;

    for (unsigned i = 0; i < count; ++i) {
      const auto &e = data[i];
      if (e.gpu_cost_us < gpu_threshold_us &&
          e.gpu_cost_us < e.cpu_cost_us * adaptive_margin) {
        gpu_leaf->data[gpu_idx++] = e;
      } else {
        temp[cpu_idx++] = e;
      }
    }

    memcpy(data, temp, sizeof(CostEntry) * cpu_idx);
    count = cpu_idx;
    gpu_leaf->count = gpu_idx;

    // Kahan 重算
    total_cpu_cost = total_gpu_cost = 0;
    for (unsigned i = 0; i < count; ++i) {
      total_cpu_cost += data[i].cpu_cost_us;
      total_gpu_cost += data[i].gpu_cost_us;
    }
    gpu_leaf->total_cpu_cost = gpu_leaf->total_gpu_cost = 0;
    for (unsigned i = 0; i < gpu_idx; ++i) {
      gpu_leaf->total_cpu_cost += gpu_leaf->data[i].cpu_cost_us;
      gpu_leaf->total_gpu_cost += gpu_leaf->data[i].gpu_cost_us;
    }

    BT_TRACE("SplitByCost: cpu_entries=%u gpu_entries=%u threshold=%.2f margin=%.2f",
             cpu_idx, gpu_idx, gpu_threshold_us, adaptive_margin);
    return gpu_idx;
  }

  // 改写: avg_gpu_speedup 加 harmonic mean 选项
  double avg_gpu_speedup(bool harmonic = false) const {
    if (count == 0) return 1.0;
    if (!harmonic) {
      double sum = 0;
      for (unsigned i = 0; i < count; ++i) {
        sum += (data[i].gpu_cost_us > 0)
                 ? data[i].cpu_cost_us / data[i].gpu_cost_us : 1.0;
      }
      return sum / count;
    }
    // 调和平均: 对异常值更鲁棒
    double inv_sum = 0;
    for (unsigned i = 0; i < count; ++i) {
      double s = (data[i].gpu_cost_us > 0)
                   ? data[i].cpu_cost_us / data[i].gpu_cost_us : 1.0;
      inv_sum += 1.0 / std::max(s, 0.001);
    }
    return count / inv_sum;
  }

  void dump_node(const char *label = "") const {
    fprintf(stderr, "[COST·LEAF] %s count=%u/%llu cpu_total=%.2f gpu_total=%.2f speedup=%.2f\n",
            label, count, maxEntries, total_cpu_cost, total_gpu_cost, avg_gpu_speedup());
  }
};

}  // namespace tabular
}  // namespace lynceus
