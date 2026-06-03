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
#include <cstdio>
#include <cstdint>
#include <atomic>

namespace lynceus {
namespace index {

// 改写: 加入操作计数器 + dump 诊断接口
struct GenericIndex {
  // --- 运行时统计 (改写新增) ---
  struct OpCounters {
    std::atomic<uint64_t> inserts{0};
    std::atomic<uint64_t> searches{0};
    std::atomic<uint64_t> scans{0};
    std::atomic<uint64_t> updates{0};
    std::atomic<uint64_t> deletes{0};
    std::atomic<uint64_t> scan_rows_returned{0};  // scan 实际返回行数

    void dump(const char *label = "") const {
      fprintf(stderr, "[IDX·STATS] %s ins=%lu srch=%lu scan=%lu upd=%lu del=%lu scan_rows=%lu\n",
        label, inserts.load(), searches.load(), scans.load(),
        updates.load(), deletes.load(), scan_rows_returned.load());
    }
    void reset() {
      inserts = searches = scans = updates = deletes = scan_rows_returned = 0;
    }
  };

  mutable OpCounters counters;

  virtual bool Insert(const char *key, size_t key_sz, const char *value, size_t value_sz) = 0;
  virtual bool Search(const char *key, size_t key_sz, char *&value_out) = 0;
  virtual int Scan(const char *start_key, bool start_key_inclusive,
                  const char *end_key, bool end_key_inclusive,
                  size_t key_sz, int scan_sz, char *&value_out) = 0;
  virtual int ReverseScan(const char *start_key, bool start_key_inclusive,
                         const char *end_key, bool end_key_inclusive,
                         size_t key_sz, int scan_sz, char *&value_out) = 0;
  virtual bool Update(const char *key, size_t key_sz, const char *value, size_t value_sz) = 0;
  virtual bool Delete(const char *key, size_t key_sz) = 0;

  // 改写: 批量点查接口 — 摊薄虚函数开销
  virtual int BatchSearch(const char **keys, size_t key_sz, size_t n,
                          char **values_out) {
    int found = 0;
    for (size_t i = 0; i < n; i++) {
      found += Search(keys[i], key_sz, values_out[i]) ? 1 : 0;
    }
    return found;
  }

  GenericIndex(size_t key_size, size_t value_size)
    : key_size(key_size), value_size(value_size) {}

  virtual ~GenericIndex() = default;

  void dump_counters(const char *label = "") const {
    counters.dump(label);
  }

  size_t key_size;
  size_t value_size;
};

} //namespace index
} // namespace lynceus
