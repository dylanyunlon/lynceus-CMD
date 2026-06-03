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

#include "hash_table_common.h"

namespace lynceus {
namespace index {

// Optimistic Lock — 改写: 加 backoff 指数退避 + 竞争统计
struct OptiLock {
  inline static constexpr uint64_t LOCK_BIT = 1;

  uint64_t lock_word;
#if LYNCEUS_HT_DBG
  std::atomic<uint64_t> contention_spins{0};
#endif

  OptiLock() : lock_word(0) {}

  uint64_t SetLockedBit(uint64_t version) { return version | LOCK_BIT; }
  bool IsLocked(uint64_t version) { return version & LOCK_BIT; }

  // 改写: 指数退避读锁, 减少 cache-line bouncing
  uint64_t RLock() {
    uint64_t version;
    uint32_t backoff = 1;
    do {
      version = __atomic_load_n(&lock_word, __ATOMIC_ACQUIRE);
      if (!IsLocked(version)) return version;
      // spin-wait with exponential backoff (cap 1024)
      for (uint32_t i = 0; i < backoff; i++) {
        __builtin_ia32_pause();
      }
      if (backoff < 1024) backoff <<= 1;
#if LYNCEUS_HT_DBG
      contention_spins.fetch_add(1, std::memory_order_relaxed);
#endif
    } while (true);
  }

  bool Check(uint64_t version) {
    return version == __atomic_load_n(&lock_word, __ATOMIC_ACQUIRE);
  }

  bool RUnlock(uint64_t version) { return Check(version); }

  bool Upgrade(uint64_t version) {
    return __atomic_compare_exchange_n(&lock_word, &version,
                                       SetLockedBit(version), false,
                                       __ATOMIC_RELEASE, __ATOMIC_ACQUIRE);
  }

  // 改写: 写锁也用退避
  void Lock() {
    uint64_t version;
    uint32_t backoff = 1;
    do {
      version = RLock();
      if (Upgrade(version)) return;
      for (uint32_t i = 0; i < backoff; i++) __builtin_ia32_pause();
      if (backoff < 1024) backoff <<= 1;
    } while (true);
  }

  void Unlock() {
    __atomic_store_n(&lock_word, lock_word + LOCK_BIT, __ATOMIC_RELEASE);
  }

  void dump_lock_state(const char *label = "") const {
    fprintf(stderr, "[LOCK·DBG] %s lock_word=%lu locked=%d\n",
            label, lock_word, (int)(lock_word & LOCK_BIT));
#if LYNCEUS_HT_DBG
    fprintf(stderr, "[LOCK·DBG] %s contention_spins=%lu\n",
            label, contention_spins.load());
#endif
  }
};

// Extendible Hash Table
template <typename Key, typename Value>
struct HashTable {
  struct ConcurrentBucket {
    OptiLock lock;
    HashTableBucket<Key, Value> bucket;
  };

  struct Entry {
    uint8_t local_depth;
    ConcurrentBucket *concurrent_bucket;
  };

  // 改写: 追踪运行时统计
  struct RuntimeStats {
    uint64_t total_inserts = 0;
    uint64_t total_lookups = 0;
    uint64_t total_updates = 0;
    uint64_t directory_doublings = 0;
    uint64_t bucket_splits = 0;
    uint8_t  peak_depth = 0;

    void dump() const {
      fprintf(stderr, "[HT·RT] inserts=%lu lookups=%lu updates=%lu "
              "doublings=%lu splits=%lu peak_depth=%u\n",
              total_inserts, total_lookups, total_updates,
              directory_doublings, bucket_splits, peak_depth);
    }
  };

  RuntimeStats stats;

  HashTable() : global_depth(0) {
    entries = new Entry[1];
    entries[0].concurrent_bucket = new ConcurrentBucket();
    entries[0].local_depth = 0;
    HT_TRACE("HashTable ctor: initial capacity=1");
  }

  Entry GetEntry(uint64_t hash) {
    bool ok = false;
    Entry entry;
    do {
      auto version = directory_lock.RLock();
      auto offset = hash & ~((~0ULL) << global_depth);
      entry = entries[offset];
      ok = directory_lock.Check(version);
    } while (!ok);
    return entry;
  }

  template <bool IsDirectoryLocked>
  void UpdateEntries(uint8_t local_depth, uint64_t hash,
                     ConcurrentBucket *new_bucket) {
    auto first_idx = hash & ~(~0ULL << local_depth);
    auto step = 1ULL << local_depth;
    if constexpr (!IsDirectoryLocked) {
      directory_lock.Lock();
    }
    auto dir_size = 1ULL << global_depth;
    uint64_t updated_count = 0;
    for (auto i = first_idx; i < dir_size; i += step) {
      entries[i].local_depth += 1;
      if (i & (1ULL << local_depth)) {
        entries[i].concurrent_bucket = new_bucket;
      }
      updated_count++;
    }
    HT_TRACE("UpdateEntries: touched %lu directory slots", updated_count);
    if constexpr (!IsDirectoryLocked) {
      directory_lock.Unlock();
    }
  }

  bool insert(Key key, Value value) {
    auto hash = Hash(key);
    stats.total_inserts++;
    HT_STAT_INC(g_ht_insert_count);
  restart:
    while (true) {
      auto [local_depth, concurrent_bucket] = GetEntry(hash);
      concurrent_bucket->lock.Lock();
      auto [curr_depth, curr_bucket] = GetEntry(hash);
      if (curr_depth != local_depth) {
        concurrent_bucket->lock.Unlock();
        goto restart;
      }

      if (concurrent_bucket->bucket.Find(key)) {
        concurrent_bucket->lock.Unlock();
        HT_TRACE("insert: duplicate key, rejected");
        return false;
      }

      if (!concurrent_bucket->bucket.IsFull()) {
        concurrent_bucket->bucket.Insert(key, value);
        concurrent_bucket->lock.Unlock();
        return true;
      }

      // --- bucket split ---
      stats.bucket_splits++;
      auto new_bucket = new ConcurrentBucket();
      concurrent_bucket->bucket.SplitTo(local_depth, &(new_bucket->bucket));

      if (local_depth < global_depth) {
        UpdateEntries<false>(local_depth, hash, new_bucket);
      } else {
        directory_lock.Lock();
        if (local_depth < global_depth) {
          UpdateEntries<true>(local_depth, hash, new_bucket);
        } else {
          // --- 目录翻倍: 改写为 realloc 风格 + 统计 ---
          auto dir_size = 1ULL << global_depth;
          auto offset = hash & ~((~0ULL) << global_depth);
          auto new_entries = new Entry[2 * dir_size];
          std::memcpy(&new_entries[0], entries, dir_size * sizeof(Entry));
          std::memcpy(&new_entries[dir_size], entries, dir_size * sizeof(Entry));

          new_entries[offset].local_depth += 1;
          new_entries[offset + dir_size].local_depth += 1;
          new_entries[offset + dir_size].concurrent_bucket = new_bucket;

          delete[] entries;
          entries = new_entries;
          global_depth += 1;
          stats.directory_doublings++;
          if (global_depth > stats.peak_depth)
            stats.peak_depth = global_depth;
          HT_TRACE("directory doubled → depth=%u, capacity=%lu",
                   global_depth, 1ULL << global_depth);
        }
        directory_lock.Unlock();
      }
      concurrent_bucket->lock.Unlock();
    }
  }

  // 改写: lookup 加 retry 计数诊断
  bool lookup(Key key, Value &out_value) {
    auto hash = Hash(key);
    stats.total_lookups++;
    HT_STAT_INC(g_ht_lookup_count);
    bool ok;
    bool directory_ok;
    bool bucket_ok;
    uint32_t retries = 0;
    do {
      auto lock_word = directory_lock.RLock();
      auto offset = hash & ~(~0ULL << global_depth);
      auto [local_depth, concurrent_bucket] = entries[offset];

      auto bucket_lock_word = concurrent_bucket->lock.RLock();
      ok = concurrent_bucket->bucket.Find(key, &out_value);
      bucket_ok = concurrent_bucket->lock.RUnlock(bucket_lock_word);
      directory_ok = directory_lock.RUnlock(lock_word);
      retries++;
    } while (!(bucket_ok && directory_ok));

    if (retries > 1) {
      HT_TRACE("lookup: required %u retries (contention)", retries);
    }
    return ok;
  }

  bool update(Key key, Value value) {
    auto hash = Hash(key);
    stats.total_updates++;
    bool ok;
    uint64_t lock_word;
    do {
      lock_word = directory_lock.RLock();
      auto offset = hash & ~(~0ULL << global_depth);
      auto [local_depth, concurrent_bucket] = entries[offset];

      concurrent_bucket->lock.Lock();
      if (directory_lock.RUnlock(lock_word)) {
        ok = concurrent_bucket->bucket.Update(key, value);
        concurrent_bucket->lock.Unlock();
        return ok;
      }
      concurrent_bucket->lock.Unlock();
    } while (true);
  }

  uint64_t scan(Key k, int64_t range, Value *output) {
    fprintf(stderr, "[HT·FATAL] Hash table doesn't support scan\n");
    assert(false && "Hash table doesn't support scan");
    return 0;
  }

  // 诊断: dump 整个 hash table 状态
  void dump_state() const {
    auto dir_size = 1ULL << global_depth;
    fprintf(stderr, "[HT·STATE] global_depth=%u dir_size=%lu\n",
            global_depth, dir_size);
    stats.dump();
    directory_lock.dump_lock_state("dir");
  }

  OptiLock directory_lock;
  uint8_t global_depth;
  Entry *entries;
};

} // namespace index
} // namespace lynceus
