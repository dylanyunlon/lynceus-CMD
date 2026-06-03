/*
 * Copyright (C) 2024 Data-Intensive Systems Lab, Simon Fraser University.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <atomic>

// ---------- debug knob ----------
#ifndef LYNCEUS_HT_DBG
#define LYNCEUS_HT_DBG 0
#endif

#if LYNCEUS_HT_DBG
  static std::atomic<uint64_t> g_ht_insert_count{0};
  static std::atomic<uint64_t> g_ht_lookup_count{0};
  static std::atomic<uint64_t> g_ht_split_count{0};
  static std::atomic<uint64_t> g_ht_collision_count{0};
  #define HT_TRACE(fmt, ...) fprintf(stderr, "[HT·DBG] " fmt "\n", ##__VA_ARGS__)
  #define HT_STAT_INC(ctr) ctr.fetch_add(1, std::memory_order_relaxed)
#else
  #define HT_TRACE(fmt, ...) ((void)0)
  #define HT_STAT_INC(ctr) ((void)0)
#endif

// 诊断: 打印全局累计统计
static inline void ht_dump_stats() {
#if LYNCEUS_HT_DBG
  fprintf(stderr, "[HT·STATS] inserts=%lu lookups=%lu splits=%lu collisions=%lu\n",
    g_ht_insert_count.load(), g_ht_lookup_count.load(),
    g_ht_split_count.load(), g_ht_collision_count.load());
#endif
}


#ifndef HASHTABLE_PAGE_SIZE
#define HASHTABLE_PAGE_SIZE 256
#endif

namespace lynceus {
namespace index {

static inline constexpr uint64_t kHashTablePageSize = HASHTABLE_PAGE_SIZE;

// ---- 改写: 双轮 FNV-1a + rotate 混合哈希 ----
// 原版单轮 FNV-1a; 改为先 FNV-1a 再做 finalizer 混合,
// 分布质量更好 (avalanche 更均匀, 减少 extendible bucket 分裂次数)
template <typename Key>
inline static uint64_t Hash(Key key) {
  uint64_t h = 14695981039346656037ULL;
  const auto *b = reinterpret_cast<const uint8_t*>(&key);
  for (size_t i = 0; i < sizeof(Key); i++) {
    h ^= b[i];
    h *= 1099511628211ULL;
  }
  // --- finalizer (改写核心): splitmix64 式混合 ---
  h ^= h >> 30;
  h *= 0xbf58476d1ce4e5b9ULL;
  h ^= h >> 27;
  h *= 0x94d049bb133111ebULL;
  h ^= h >> 31;
  HT_TRACE("Hash(key, sz=%zu) → 0x%016lx", sizeof(Key), h);
  return h;
}

template <typename Key, typename Value>
struct HashTableBucket {
  struct Header {
    size_t size;
    Header() : size(0) {}
  };
  struct Entry {
    Key key;
    Value value;
  };

  static constexpr size_t kEntrySize = sizeof(Entry);
  static constexpr uint64_t kMaxEntries =
      (kHashTablePageSize - sizeof(Header)) / kEntrySize;

  Header header;
  Entry entries[kMaxEntries];

  bool Find(Key key, Value *out_value) {
    HT_TRACE("Find(with-out): scanning %zu entries", header.size);
    // 改写: 反向扫描 — 最近插入的 key 往往是热点, 放后面先找到
    for (auto i = static_cast<int64_t>(header.size) - 1; i >= 0; i--) {
      auto &entry = entries[i];
      if (entry.key == key) {
        *out_value = entry.value;
        HT_TRACE("  hit at slot %ld", i);
        return true;
      }
    }
    return false;
  }

  bool Find(Key key) {
    for (auto i = static_cast<int64_t>(header.size) - 1; i >= 0; i--) {
      if (entries[i].key == key) return true;
    }
    return false;
  }

  bool IsFull() {
    return header.size == kMaxEntries;
  }

  void Insert(Key key, const Value &value) {
    assert(header.size < kMaxEntries);
    auto &entry = entries[header.size++];
    entry.key = key;
    entry.value = value;
    HT_TRACE("Insert: bucket now %zu/%llu", header.size, kMaxEntries);
  }

  bool Update(Key key, const Value &value) {
    // 改写: 同样反向扫描 — 和 Find 保持一致的热点倾向
    for (auto i = static_cast<int64_t>(header.size) - 1; i >= 0; i--) {
      auto &entry = entries[i];
      if (entry.key == key) {
        entry.value = value;
        HT_TRACE("Update: hit at slot %ld", i);
        return true;
      }
    }
    return false;
  }

  // 改写: 分裂时就地压缩 + 计数校验断言
  void SplitTo(uint64_t local_depth, HashTableBucket *new_bucket) {
    HT_STAT_INC(g_ht_split_count);
    const auto mask = 1ULL << local_depth;
    size_t stay_count = 0, move_count = 0;

    for (size_t i = 0; i < header.size; i++) {
      auto &entry = entries[i];
      auto hash = Hash(entry.key);
      if (hash & mask) {
        new_bucket->entries[move_count].key = entry.key;
        new_bucket->entries[move_count].value = entry.value;
        move_count++;
      } else {
        if (i > stay_count) {
          entries[stay_count].key = entry.key;
          entries[stay_count].value = entry.value;
        }
        stay_count++;
      }
    }

    // 守恒断言: 分裂前后 entry 总数不变
    assert(stay_count + move_count == header.size);
    HT_TRACE("SplitTo: depth=%lu stay=%zu move=%zu (was %zu)",
             local_depth, stay_count, move_count, header.size);

    header.size = stay_count;
    new_bucket->header.size = move_count;
  }

  // 诊断: dump bucket 内容
  void dump_bucket(const char *label = "") const {
    fprintf(stderr, "[HT·BUCKET] %s size=%zu/%llu\n", label, header.size, kMaxEntries);
  }
};

} // namespace index
} // namespace lynceus
