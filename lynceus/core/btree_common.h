/*
 * Original: Copyright (C) 2023 Data-Intensive Systems Lab, Simon Fraser University.
 * Modified: Lynceus — cost-aware B-tree node structures with heterogeneous dispatch.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 *
 * Modifications from upstream/tabular/src/tabular/btree_common.h:
 *   - Namespace: noname::tabular → lynceus::tabular
 *   - Removed: dependency on table/object.h (use local OID type)
 *   - Added:   CostAnnotatedLeafNode with per-entry cost metadata
 *   - Added:   SplitByCostThreshold for device-aware leaf splitting
 *   - Added:   lowerBoundCost for cost-ordered search
 *   - Kept:    BTreeLeafNode, BTreeInnerNode core (lowerBound, insert, split_to)
 */
#pragma once

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <utility>
#include <algorithm>
#include <cmath>

// Local OID type (replaces table/object.h dependency)
namespace lynceus { namespace table {
  using OID = uint64_t;
  static constexpr OID kInvalidOID = ~0ULL;
}}

namespace lynceus {
namespace tabular {

enum class NodeType : uint8_t { Inner = 1, Leaf = 2 };

static inline constexpr uint64_t kPageSizeLinearSearchCutoff = 256;  // Use binary search if larger
#ifndef BTREE_PAGE_SIZE
#define BTREE_PAGE_SIZE 256
#endif
// make BTree page size configurable through compiling macros
static inline constexpr uint64_t kPageSize = BTREE_PAGE_SIZE - 8;  // Match memory layout of OptiQL BTree
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
  // XXX(shiges): one spot less to accept the new key-val pair when splitting
  static const uint64_t maxEntries =
      (kPageSize - sizeof(NodeBase) - sizeof(table::OID)) / entrySize - 1;

  // Singly linked list pointer to my sibling
  table::OID next_leaf;

  // This is the array that we perform search on
  KeyValueType data[maxEntries + 1];

  BTreeLeafNode() {
    level = 1;
    count = 0;
    next_leaf = table::kInvalidOID;
  }
  // generate copy ctor for in-memory copying
  BTreeLeafNode(BTreeLeafNode<Key, Value> &) = default;

  bool isFull() { return count == maxEntries; }

  unsigned lowerBound(const Key &k) {
    if constexpr (kPageSize <= kPageSizeLinearSearchCutoff) {
      unsigned lower = 0;
      while (lower < count) {
        const Key &next_key = data[lower].key;

        if (k <= next_key) {
          return lower;
        } else {
          lower++;
        }
      }
      return lower;
    } else {
      unsigned lower = 0;
      unsigned upper = count;
      do {
        unsigned mid = ((upper - lower) / 2) + lower;
        // This is the key at the pivot position
        const Key &middle_key = data[mid].key;

        if (k < middle_key) {
          upper = mid;
        } else if (k > middle_key) {
          lower = mid + 1;
        } else {
          return mid;
        }
      } while (lower < upper);
      return lower;
    }
  }

  bool key_exists(const Key &k) {
    assert(count <= maxEntries);
    if (count) {
      unsigned pos = lowerBound(k);
      if ((pos < count) && (data[pos].key == k)) {
        // key already exists
        return true;
      }
    }
    return false;
  }

  bool insert_no_existence_check(const Key &k, Value p) {
    assert(count <= maxEntries);
    if (count) {
      unsigned pos = lowerBound(k);
      memmove(data + pos + 1, data + pos, sizeof(KeyValueType) * (count - pos));
      // memmove(payloads+pos+1,payloads+pos,sizeof(Value)*(count-pos));
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
    if (count) {
      unsigned pos = lowerBound(k);
      if ((pos < count) && (data[pos].key == k)) {
        // key already exists
        return false;
      }
      memmove(data + pos + 1, data + pos, sizeof(KeyValueType) * (count - pos));
      // memmove(payloads+pos+1,payloads+pos,sizeof(Value)*(count-pos));
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
        // key found
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
        // Update
        data[pos].value = p;
        return true;
      }
    }
    return false;
  }

  void split_to(BTreeLeafNode *newLeaf, Key &sep) {
    newLeaf->count = count - (count / 2);
    count = count - newLeaf->count;
    memcpy(newLeaf->data, data + count, sizeof(KeyValueType) * newLeaf->count);
    newLeaf->next_leaf = next_leaf;
    sep = data[count - 1].key;
  }

};

template <typename Key>
struct BTreeInnerNode : public NodeBase {
  static constexpr size_t entrySize = sizeof(Key) + sizeof(table::OID);
  // XXX(shiges): one spot less to accept the new key-val pair when splitting
  static const uint64_t maxEntries = (kPageSize - sizeof(NodeBase)) / entrySize - 1;

  table::OID children[maxEntries + 1];
  Key keys[maxEntries + 1];

  BTreeInnerNode() {
    level = 1;
    count = 0;
  }
  explicit BTreeInnerNode(uint8_t node_level) {
    level = node_level;
    count = 0;
  }
  // generate copy ctor for in-memory copying
  BTreeInnerNode(BTreeInnerNode<Key> &) = default;

  bool isFull() { return count >= (maxEntries - 1); }

  unsigned lowerBoundBF(const Key &k) {
    auto base = keys;
    unsigned n = count;
    while (n > 1) {
      const unsigned half = n / 2;
      base = (base[half] < k) ? (base + half) : base;
      n -= half;
    }
    return (*base < k) + base - keys;
  }

  unsigned lowerBound(const Key &k) {
    if constexpr (kPageSize <= kPageSizeLinearSearchCutoff) {
      unsigned lower = 0;
      while (lower < count) {
        const Key &next_key = keys[lower];

        if (k <= next_key) {
          return lower;
        } else {
          lower++;
        }
      }
      return lower;
    } else {
      unsigned lower = 0;
      unsigned upper = count;
      do {
        unsigned mid = ((upper - lower) / 2) + lower;
        if (k < keys[mid]) {
          upper = mid;
        } else if (k > keys[mid]) {
          lower = mid + 1;
        } else {
          return mid;
        }
      } while (lower < upper);
      return lower;
    }
  }

  void split_to(BTreeInnerNode *newInner, Key &sep) {
    newInner->level = level;
    newInner->count = count - (count / 2);
    count = count - newInner->count - 1;
    sep = keys[count];
    memcpy(newInner->keys, keys + count + 1, sizeof(Key) * (newInner->count + 1));
    memcpy(newInner->children, children + count + 1, sizeof(NodeBase *) * (newInner->count + 1));
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
// Lynceus extensions: CostAnnotatedLeafNode
//
// Extends BTreeLeafNode with per-entry cost metadata for heterogeneous
// device dispatch. Each entry carries CPU and GPU cost estimates,
// enabling the router to split the leaf into GPU-bound and CPU-bound
// partitions without re-estimating costs.
//
// Architecture analogy:
//   CCCL f984c90 extracts histogram from filter_and_histogram
//   → Here we extract cost annotation from the scan+route path
//   The cost metadata is computed once (annotation pass) and consumed
//   multiple times (routing decisions, histogram building, load balancing)
//
//   Tabular BTreeLeafNode::split_to splits by key median
//   → CostAnnotatedLeafNode::SplitByCostThreshold splits by cost threshold
// ===========================================================================

template <typename Key, typename Value>
struct CostAnnotatedLeafNode : public NodeBase {
  struct CostEntry {
    Key key;
    Value value;
    double cpu_cost_us;   // estimated CPU execution cost
    double gpu_cost_us;   // estimated GPU execution cost
  };

  static constexpr size_t entrySize = sizeof(CostEntry);
  static constexpr uint64_t maxEntries =
      (kPageSize - sizeof(NodeBase) - sizeof(table::OID) - sizeof(double) * 2)
      / entrySize;

  table::OID next_leaf;
  double total_cpu_cost;   // sum of cpu_cost_us across all entries
  double total_gpu_cost;   // sum of gpu_cost_us across all entries

  CostEntry data[maxEntries > 0 ? maxEntries : 1];

  CostAnnotatedLeafNode() {
    level = 1;
    count = 0;
    next_leaf = table::kInvalidOID;
    total_cpu_cost = 0.0;
    total_gpu_cost = 0.0;
  }

  bool isFull() { return count == maxEntries; }

  // Standard lowerBound (by key) — unchanged from Tabular
  unsigned lowerBound(const Key &k) {
    unsigned lower = 0;
    unsigned upper = count;
    while (lower < upper) {
      unsigned mid = (upper + lower) / 2;
      if (k <= data[mid].key) upper = mid;
      else lower = mid + 1;
    }
    return lower;
  }

  // NEW: lowerBoundCost — find position of first entry with min_cost >= threshold
  // Analogous to CCCL ExtractCostBin mapping cost to histogram bin:
  // used to identify the "split point" between GPU-bound and CPU-bound entries.
  //
  // Requires entries sorted by min_cost (see SortByCost below).
  unsigned lowerBoundCost(double cost_threshold) {
    unsigned lower = 0;
    unsigned upper = count;
    while (lower < upper) {
      unsigned mid = (upper + lower) / 2;
      double min_cost = std::min(data[mid].cpu_cost_us, data[mid].gpu_cost_us);
      if (cost_threshold <= min_cost) upper = mid;
      else lower = mid + 1;
    }
    return lower;
  }

  void insert(const Key &k, Value v, double cpu_cost, double gpu_cost) {
    assert(count < maxEntries);
    unsigned pos = lowerBound(k);
    if (count > 0 && pos < count) {
      memmove(data + pos + 1, data + pos,
              sizeof(CostEntry) * (count - pos));
    }
    data[pos].key = k;
    data[pos].value = v;
    data[pos].cpu_cost_us = cpu_cost;
    data[pos].gpu_cost_us = gpu_cost;
    total_cpu_cost += cpu_cost;
    total_gpu_cost += gpu_cost;
    count++;
  }

  // Standard split_to (by key median) — unchanged from Tabular
  void split_to(CostAnnotatedLeafNode *newLeaf, Key &sep) {
    unsigned new_count = count - (count / 2);
    unsigned keep_count = count - new_count;
    memcpy(newLeaf->data, data + keep_count,
           sizeof(CostEntry) * new_count);
    newLeaf->count = new_count;
    newLeaf->next_leaf = next_leaf;
    sep = data[keep_count - 1].key;

    // Recompute cost sums
    total_cpu_cost = 0;
    total_gpu_cost = 0;
    for (unsigned i = 0; i < keep_count; ++i) {
      total_cpu_cost += data[i].cpu_cost_us;
      total_gpu_cost += data[i].gpu_cost_us;
    }
    newLeaf->total_cpu_cost = 0;
    newLeaf->total_gpu_cost = 0;
    for (unsigned i = 0; i < new_count; ++i) {
      newLeaf->total_cpu_cost += newLeaf->data[i].cpu_cost_us;
      newLeaf->total_gpu_cost += newLeaf->data[i].gpu_cost_us;
    }
    count = keep_count;
  }

  // NEW: SplitByCostThreshold — heterogeneous device-aware split
  //
  // Unlike standard split_to (which splits by key median), this splits
  // by cost: entries whose GPU cost is below threshold go to gpu_leaf,
  // the rest stay (CPU-bound).
  //
  // Analogous to CCCL's filter_and_histogram classifying items as
  // selected (clear winner) vs candidate (needs refinement):
  //   selected (GPU wins clearly) → move to gpu_leaf
  //   candidate (ambiguous)       → stay in this leaf
  //
  // Returns the count of entries moved to gpu_leaf.
  unsigned SplitByCostThreshold(double gpu_threshold_us,
                                CostAnnotatedLeafNode *gpu_leaf) {
    unsigned cpu_idx = 0;
    unsigned gpu_idx = 0;
    CostEntry temp[maxEntries > 0 ? maxEntries : 1];

    // Classify and partition (like CCCL's f_with_out_buf lambda)
    for (unsigned i = 0; i < count; ++i) {
      const auto &e = data[i];
      if (e.gpu_cost_us < gpu_threshold_us &&
          e.gpu_cost_us < e.cpu_cost_us * 0.8551) {
        // Clear GPU winner → move to gpu_leaf
        gpu_leaf->data[gpu_idx] = e;
        gpu_idx++;
      } else {
        // CPU-bound or ambiguous → stay
        temp[cpu_idx] = e;
        cpu_idx++;
      }
    }

    // Compact this leaf
    memcpy(data, temp, sizeof(CostEntry) * cpu_idx);
    count = cpu_idx;
    gpu_leaf->count = gpu_idx;

    // Recompute cost sums
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

    return gpu_idx;
  }

  // Utility: average GPU speedup for entries in this leaf
  double avg_gpu_speedup() const {
    if (count == 0) return 1.0;
    double sum = 0;
    for (unsigned i = 0; i < count; ++i) {
      sum += (data[i].gpu_cost_us > 0)
               ? data[i].cpu_cost_us / data[i].gpu_cost_us
               : 1.0;
    }
    return sum / count;
  }
};

}  // namespace tabular
}  // namespace lynceus
