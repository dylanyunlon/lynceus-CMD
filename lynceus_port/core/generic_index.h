#pragma once
#include <cstddef>
#include <cstdio>
#include <cstdint>

namespace lynceus {
namespace index {

struct GenericIndex {
  // 调试: 操作计数, 不用 atomic — 单线程调试场景够用, 避免引入原版没有的头文件
  uint64_t _cnt_ins = 0;
  uint64_t _cnt_search = 0;
  uint64_t _cnt_scan = 0;
  uint64_t _cnt_del = 0;

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

  // --- 算法改写: 原版没有虚析构, port加上防止基类指针 delete 时内存泄漏
  virtual ~GenericIndex() = default;

  GenericIndex(size_t key_size, size_t value_size)
    : key_size(key_size), value_size(value_size) {}

  // 调试: 打印计数快照
  void dump_counters(const char *who = "?") const {
    fprintf(stderr, "[IDX·STAT] %s ins=%lu srch=%lu scan=%lu del=%lu\n",
            who, _cnt_ins, _cnt_search, _cnt_scan, _cnt_del);
  }

  size_t key_size;
  size_t value_size;
};

} // namespace index
} // namespace lynceus
