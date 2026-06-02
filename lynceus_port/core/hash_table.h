#pragma once

#include "hash_table_common.h"
#include <cstdio>
#include <cstring>
#include <atomic>

namespace lynceus {
namespace index {

struct OptiLock {
  inline static constexpr uint64_t LOCK_BIT = 1;
  uint64_t lock_word;

  // 调试: 争用计数
  std::atomic<uint64_t> _contention_spins{0};

  OptiLock() : lock_word(0) {}

  uint64_t SetLockedBit(uint64_t version) { return version | LOCK_BIT; }
  bool IsLocked(uint64_t version) { return version & LOCK_BIT; }

  uint64_t RLock() {
    uint64_t version;
    uint64_t spins = 0;
    do {
      version = __atomic_load_n(&lock_word, __ATOMIC_ACQUIRE);
      if (IsLocked(version)) spins++;
    } while (IsLocked(version));
    _contention_spins.fetch_add(spins, std::memory_order_relaxed);
    return version;
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

  void Lock() {
    uint64_t version;
    do {
      version = RLock();
    } while (!Upgrade(version));
  }

  // --- 算法改写: unlock 时用 fetch_add 替代 load+store,
  //     原版是 __atomic_store_n(lock_word + LOCK_BIT)
  //     这里改成 FAA(+2) 保证偶数version (bit0==0 → unlocked)
  void Unlock() {
    __atomic_fetch_add(&lock_word, LOCK_BIT + 1, __ATOMIC_RELEASE);
  }
};

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

  // 调试: 全局统计
  uint64_t _total_inserts = 0;
  uint64_t _total_splits  = 0;
  uint64_t _total_lookups = 0;

  HashTable() : global_depth(0) {
    entries = new Entry[1];
    entries[0].concurrent_bucket = new ConcurrentBucket();
    entries[0].local_depth = 0;
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
    for (auto i = first_idx; i < dir_size; i += step) {
      entries[i].local_depth += 1;
      if (i & (1ULL << local_depth)) {
        entries[i].concurrent_bucket = new_bucket;
      }
    }
    if constexpr (!IsDirectoryLocked) {
      directory_lock.Unlock();
    }
  }

  bool insert(Key key, Value value) {
    auto hash = Hash(key);
    _total_inserts++;
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
        return false;
      }

      if (!concurrent_bucket->bucket.IsFull()) {
        concurrent_bucket->bucket.Insert(key, value);
        concurrent_bucket->lock.Unlock();
        return true;
      }

      // split
      _total_splits++;
      auto new_bucket = new ConcurrentBucket();
      concurrent_bucket->bucket.SplitTo(local_depth, &(new_bucket->bucket));

      if (local_depth < global_depth) {
        UpdateEntries<false>(local_depth, hash, new_bucket);
      } else {
        directory_lock.Lock();
        if (local_depth < global_depth) {
          UpdateEntries<true>(local_depth, hash, new_bucket);
        } else {
          auto dir_size = 1ULL << global_depth;
          auto offset = hash & ~((~0ULL) << global_depth);
          auto new_entries = new Entry[2 * dir_size];
          memcpy(&new_entries[0], entries, dir_size * sizeof(Entry));
          memcpy(&new_entries[dir_size], entries, dir_size * sizeof(Entry));
          new_entries[offset].local_depth += 1;
          new_entries[offset + dir_size].local_depth += 1;
          new_entries[offset + dir_size].concurrent_bucket = new_bucket;
          delete[] entries;
          entries = new_entries;
          global_depth += 1;

          // 调试: directory doubling 事件
          fprintf(stderr, "[HT·GROW] global_depth=%u dir_size=%lu->%lu\n",
                  global_depth, (unsigned long)dir_size, (unsigned long)(2*dir_size));
        }
        directory_lock.Unlock();
      }
      concurrent_bucket->lock.Unlock();
    }
  }

  bool lookup(Key key, Value &out_value) {
    _total_lookups++;
    auto hash = Hash(key);
    bool ok;
    bool bucket_ok, directory_ok;
    do {
      auto lock_word = directory_lock.RLock();
      auto offset = hash & ~(~0ULL << global_depth);
      auto [local_depth, concurrent_bucket] = entries[offset];
      auto bucket_lock_word = concurrent_bucket->lock.RLock();
      ok = concurrent_bucket->bucket.Find(key, &out_value);
      bucket_ok = concurrent_bucket->lock.RUnlock(bucket_lock_word);
      directory_ok = directory_lock.RUnlock(lock_word);
    } while (!(bucket_ok && directory_ok));
    return ok;
  }

  bool update(Key key, Value value) {
    auto hash = Hash(key);
    bool ok;
    do {
      auto lock_word = directory_lock.RLock();
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
    fprintf(stderr, "[HT·FATAL] scan() not supported on hash table\n");
    assert(0 && "hash table scan unsupported");
    return 0;
  }

  // 调试: 打印全局状态快照
  void dump_state(const char *tag = "?") const {
    fprintf(stderr, "[HT·STATE] %s depth=%u inserts=%lu splits=%lu lookups=%lu lock_spins=%lu\n",
            tag, global_depth, _total_inserts, _total_splits, _total_lookups,
            directory_lock._contention_spins.load());
  }

  OptiLock directory_lock;
  uint8_t global_depth;
  Entry *entries;
};

} // namespace index
} // namespace lynceus
