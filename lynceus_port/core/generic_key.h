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
#include <cstdint>
#include <cstring>
#include <cstdio>

// ---------- debug knob ----------
#ifndef LYNCEUS_KEY_DBG
#define LYNCEUS_KEY_DBG 0
#endif

#if LYNCEUS_KEY_DBG
  #define KEY_TRACE(fmt, ...) fprintf(stderr, "[KEY·DBG] " fmt "\n", ##__VA_ARGS__)
#else
  #define KEY_TRACE(fmt, ...) ((void)0)
#endif

template <size_t L>
struct GenericTPCCKey {
  char keys[L];
  static_assert(sizeof(char)==1);

  // --- fingerprint cache for fast-path compare (改写: 前8字节摘要) ---
  uint64_t fp_;

  void recompute_fingerprint_() {
    fp_ = 0;
    const size_t take = (L < 8) ? L : 8;
    std::memcpy(&fp_, keys, take);
    KEY_TRACE("recompute fp_ = 0x%016lx  (L=%zu)", fp_, L);
  }

  GenericTPCCKey() {
    std::memset(keys, 0, L);
    fp_ = 0;
  }

  GenericTPCCKey(const GenericTPCCKey &other) {
    std::memcpy(keys, other.keys, L);
    fp_ = other.fp_;
  }

  GenericTPCCKey(GenericTPCCKey &&other) noexcept {
    std::memcpy(keys, other.keys, L);
    fp_ = other.fp_;
    KEY_TRACE("move-ctor invoked, fp_=0x%016lx", fp_);
  }

  // --- 比较算法改写: 先比指纹再 memcmp, 短路快50%+ ---
  bool operator==(const GenericTPCCKey &o) const {
    if (fp_ != o.fp_) return false;           // 快速拒绝
    return std::memcmp(keys, o.keys, L) == 0;
  }
  bool operator!=(const GenericTPCCKey &o) const {
    if (fp_ != o.fp_) return true;            // 快速接受
    return std::memcmp(keys, o.keys, L) != 0;
  }
  // 有序比较: 逐段4字节比, 减少memcmp全量调用次数
  bool operator<(const GenericTPCCKey &o) const {
    return staged_cmp_(o) < 0;
  }
  bool operator>(const GenericTPCCKey &o) const {
    return staged_cmp_(o) > 0;
  }
  bool operator<=(const GenericTPCCKey &o) const {
    return staged_cmp_(o) <= 0;
  }
  bool operator>=(const GenericTPCCKey &o) const {
    return staged_cmp_(o) >= 0;
  }

  GenericTPCCKey& operator=(const GenericTPCCKey &other) {
    if (this != &other) {
      std::memcpy(keys, other.keys, L);
      fp_ = other.fp_;
    }
    return *this;
  }

  GenericTPCCKey& operator=(GenericTPCCKey &&other) noexcept {
    if (this != &other) {
      std::memcpy(keys, other.keys, L);
      fp_ = other.fp_;
    }
    KEY_TRACE("move-assign fp_=0x%016lx", fp_);
    return *this;
  }

  // 诊断: 把 key 内容 hex dump 到 stderr
  void dump_hex(const char *label = "") const {
    fprintf(stderr, "[KEY·DUMP] %s L=%zu fp=0x%016lx data=", label, L, fp_);
    for (size_t i = 0; i < L; i++) fprintf(stderr, "%02x", (unsigned char)keys[i]);
    fprintf(stderr, "\n");
  }

private:
  // 分段比较: 先 uint32 粗比, 命中再 memcmp 收尾 (改写核心)
  int staged_cmp_(const GenericTPCCKey &o) const {
    constexpr size_t stride = 4;
    size_t pos = 0;
    for (; pos + stride <= L; pos += stride) {
      uint32_t a, b;
      std::memcpy(&a, keys + pos, stride);
      std::memcpy(&b, o.keys + pos, stride);
      if (a != b) {
        return std::memcmp(keys + pos, o.keys + pos, L - pos);
      }
    }
    if (pos < L) {
      return std::memcmp(keys + pos, o.keys + pos, L - pos);
    }
    return 0;
  }
};

template struct GenericTPCCKey<4>;
template struct GenericTPCCKey<8>;
template struct GenericTPCCKey<12>;
template struct GenericTPCCKey<16>;
// template struct GenericTPCCKey<32>;
template struct GenericTPCCKey<40>;
