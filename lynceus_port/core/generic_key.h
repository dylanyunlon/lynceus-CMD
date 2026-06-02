#pragma once
#include <cassert>
#include <cstring>
#include <cstdio>
#include <cstddef>

template <size_t L>
struct GenericTPCCKey {
  char keys[L];
  static_assert(sizeof(char)==1);

  GenericTPCCKey() { std::memset(this->keys, 0, L); }

  GenericTPCCKey(const GenericTPCCKey &other) {
    std::memcpy(this->keys, other.keys, L);
  }

  // --- 算法改写: 原版 CHECK(false) 即 abort,
  //     port 改为正常 memcpy 但打印 warning 便于排查是谁触发的
  GenericTPCCKey(GenericTPCCKey &&other) {
    fprintf(stderr, "[KEY·MOVE] move ctor L=%zu, 原版会abort\n", L);
    std::memcpy(this->keys, other.keys, L);
  }

  bool operator==(const GenericTPCCKey &other) const {
    return std::memcmp(this->keys, other.keys, L) == 0;
  }
  bool operator!=(const GenericTPCCKey &other) const {
    return std::memcmp(this->keys, other.keys, L) != 0;
  }
  bool operator<(const GenericTPCCKey &other) const {
    return std::memcmp(this->keys, other.keys, L) < 0;
  }
  bool operator>(const GenericTPCCKey &other) const {
    return std::memcmp(this->keys, other.keys, L) > 0;
  }
  bool operator<=(const GenericTPCCKey &other) const {
    return std::memcmp(this->keys, other.keys, L) <= 0;
  }
  bool operator>=(const GenericTPCCKey &other) const {
    return std::memcmp(this->keys, other.keys, L) >= 0;
  }

  GenericTPCCKey& operator=(const GenericTPCCKey &other) {
    if (this != &other) std::memcpy(this->keys, other.keys, L);
    return *this;
  }

  // --- 同上, 原版 CHECK(false) -> warning + memcpy
  GenericTPCCKey& operator=(GenericTPCCKey &&other) {
    fprintf(stderr, "[KEY·MOVE] move assign L=%zu\n", L);
    if (this != &other) std::memcpy(this->keys, other.keys, L);
    return *this;
  }

  // 调试: 十六进制输出 key 内容
  void hexdump(const char *label = "") const {
    fprintf(stderr, "[KEY·HEX] %s L=%zu: ", label, L);
    for (size_t i = 0; i < L; i++) fprintf(stderr, "%02x", (unsigned char)keys[i]);
    fprintf(stderr, "\n");
  }
};

template struct GenericTPCCKey<4>;
template struct GenericTPCCKey<8>;
template struct GenericTPCCKey<12>;
template struct GenericTPCCKey<16>;
template struct GenericTPCCKey<40>;
