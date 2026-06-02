#pragma once
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdio>

#ifndef HASHTABLE_PAGE_SIZE
#define HASHTABLE_PAGE_SIZE 256
#endif

namespace lynceus {
namespace index {

static inline constexpr uint64_t kHashTablePageSize = HASHTABLE_PAGE_SIZE;

// --- 算法改写: 原版用 FNV-1a (h^=b[i]; h*=prime)
//     port 换 murmur3-finalizer 风格的 avalanche mixing
//     减少低位碰撞, 对 extendible hashing 的 directory mask 更友好
template <typename Key>
inline static uint64_t Hash(Key key) {
  uint64_t h = 0;
  const auto *b = reinterpret_cast<const uint8_t*>(&key);
  for (size_t i = 0; i < sizeof(Key); i++) {
    h = (h << 8) | b[i];           // 先把字节拼进去
  }
  // murmur3 finalizer (原版是 FNV 的逐字节 XOR-mul)
  h ^= h >> 33;
  h *= 0xFF51AFD7ED558CCDULL;
  h ^= h >> 33;
  h *= 0xC4CEB9FE1A85EC53ULL;
  h ^= h >> 33;
  return h;
}

template <typename Key, typename Value>
struct HashTableBucket {
  struct Header {
    size_t size;
    size_t _probe_total;  // 调试: 累计线性探测步数
    Header() : size(0), _probe_total(0) {}
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
    for (size_t i = 0; i < header.size; i++) {
      header._probe_total++;
      auto &entry = entries[i];
      if (entry.key == key) {
        *out_value = entry.value;
        return true;
      }
    }
    return false;
  }

  bool Find(Key key) {
    for (size_t i = 0; i < header.size; i++) {
      header._probe_total++;
      if (entries[i].key == key) return true;
    }
    return false;
  }

  bool IsFull() { return header.size == kMaxEntries; }

  void Insert(Key key, const Value &value) {
    assert(header.size < kMaxEntries);
    auto &entry = entries[header.size++];
    entry.key = key;
    entry.value = value;
  }

  bool Update(Key key, const Value &value) {
    for (size_t i = 0; i < header.size; i++) {
      if (entries[i].key == key) {
        entries[i].value = value;
        return true;
      }
    }
    return false;
  }

  // --- 算法改写: split 时统计迁移量, 方便判断 hash 分布是否倾斜
  void SplitTo(uint64_t local_depth, HashTableBucket *new_bucket) {
    size_t stay = 0, move = 0;
    auto mask = 1ULL << local_depth;
    for (size_t i = 0; i < header.size; i++) {
      auto &e = entries[i];
      auto hash = Hash(e.key);
      if (hash & mask) {
        new_bucket->entries[move].key = e.key;
        new_bucket->entries[move].value = e.value;
        move++;
      } else {
        if (i > stay) {
          entries[stay].key = e.key;
          entries[stay].value = e.value;
        }
        stay++;
      }
    }
    header.size = stay;
    new_bucket->header.size = move;
    // 调试: 倾斜检测
    if (stay > 0 && move > 0) {
      double ratio = (double)stay / (double)move;
      if (ratio > 4.0 || ratio < 0.25) {
        fprintf(stderr, "[HT·SPLIT] skew detected depth=%lu stay=%zu move=%zu ratio=%.2f\n",
                (unsigned long)local_depth, stay, move, ratio);
      }
    }
  }

  // 调试: 输出桶的统计摘要
  void dump_bucket(const char *tag) const {
    fprintf(stderr, "[HT·BUCKET] %s size=%zu/%lu probes=%zu\n",
            tag, header.size, (unsigned long)kMaxEntries, header._probe_total);
  }
};

} // namespace index
} // namespace lynceus
