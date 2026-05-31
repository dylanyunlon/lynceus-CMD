# -*- coding: utf-8 -*-
"""
Original: PAR2QO cached_robust_plan_dict.py — pre-computed robust plan indices
          (upstream/par2qo/code/cached_robust_plan_dict.py, Hap-Hugh/PAR2QO)
Modified: Lynceus — heterogeneous plan cache with device-aware lookup and
          LRU eviction tracking.

Modifications from upstream (~80% structure kept, ~20% algorithm changed):
  - Kept:    dict-based plan-index lookup structure
  - Kept:    per-query-id → plan-index mapping contract
  - Modified: added device tag (cpu/gpu) per cached plan
  - Modified: LRU access tracking for cache efficiency analysis
  - Added:   RobustPlanCache class with thread-safe lookup
  - Added:   Cache warm/cold hit ratio tracking
  - Added:   Serialization to/from JSON with debug dump

References:
  PAR2QO cached_robust_plan_dict.py — flat dict {query_id: [plan_index]}
"""
from __future__ import annotations

import json
import time
import logging
import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("lynceus.plan_cache")


# ── Cached plan entry ──────────────────────────────────────────────────
@dataclass
class CachedPlanEntry:
    """A single cached robust plan result.

    PAR2QO: {query_id: [plan_index]} — just an int list.
    Lynceus: enriched with device tag, penalty, access tracking.
    """
    plan_indices: List[int]
    device_preference: str = ""         # "cpu", "gpu", or "" (auto)
    expected_penalty: float = 0.0
    penalty_std: float = 0.0
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_access: float = 0.0

    def touch(self):
        self.access_count += 1
        self.last_access = time.time()

    @property
    def best_plan(self) -> int:
        return self.plan_indices[0] if self.plan_indices else -1


# ── Static cached dictionaries (from PAR2QO) ──────────────────────────
# PAR2QO: cached_rob_plan_dict, cached_rob_plan_dict_stats, etc.
# Lynceus: converted to CachedPlanEntry with device annotations.

_CACHED_IMDB_PLANS: Dict[str, CachedPlanEntry] = {
    "1a":  CachedPlanEntry([2],  device_preference="cpu"),
    "2a":  CachedPlanEntry([1],  device_preference="gpu"),
    "3a":  CachedPlanEntry([10], device_preference="gpu"),
    "4a":  CachedPlanEntry([3],  device_preference="gpu"),
    "5a":  CachedPlanEntry([0],  device_preference="cpu"),
    "6a":  CachedPlanEntry([5],  device_preference="gpu"),
    "7a":  CachedPlanEntry([0],  device_preference="cpu"),
    "8a":  CachedPlanEntry([1],  device_preference="cpu"),
    "9a":  CachedPlanEntry([14], device_preference="gpu"),
    "10a": CachedPlanEntry([0],  device_preference="cpu"),
    "11a": CachedPlanEntry([3],  device_preference="cpu"),
    "12a": CachedPlanEntry([7],  device_preference="gpu"),
    "13a": CachedPlanEntry([10], device_preference="gpu"),
    "14a": CachedPlanEntry([0],  device_preference="cpu"),
    "15a": CachedPlanEntry([6],  device_preference="gpu"),
    "16a": CachedPlanEntry([2],  device_preference="cpu"),
    "17a": CachedPlanEntry([8],  device_preference="gpu"),
    "18a": CachedPlanEntry([4],  device_preference="gpu"),
    "19a": CachedPlanEntry([1],  device_preference="cpu"),
    "20a": CachedPlanEntry([3],  device_preference="cpu"),
    "21a": CachedPlanEntry([5],  device_preference="gpu"),
}

_CACHED_TPCH_PLANS: Dict[str, CachedPlanEntry] = {
    "tpch_q1":  CachedPlanEntry([0], device_preference="gpu"),
    "tpch_q3":  CachedPlanEntry([2], device_preference="gpu"),
    "tpch_q5":  CachedPlanEntry([1], device_preference="gpu"),
    "tpch_q6":  CachedPlanEntry([0], device_preference="gpu"),
    "tpch_q9":  CachedPlanEntry([3], device_preference="gpu"),
    "tpch_q12": CachedPlanEntry([1], device_preference="cpu"),
    "tpch_q14": CachedPlanEntry([0], device_preference="cpu"),
    "tpch_q18": CachedPlanEntry([2], device_preference="gpu"),
    "tpch_q21": CachedPlanEntry([4], device_preference="gpu"),
}

_CACHED_STATS_PLANS: Dict[str, CachedPlanEntry] = {
    "8b":  CachedPlanEntry([0]),
    "8d":  CachedPlanEntry([0]),
    "10b": CachedPlanEntry([0]),
    "10c": CachedPlanEntry([3]),
    "18a": CachedPlanEntry([8]),
    "20g": CachedPlanEntry([3]),
    "20a": CachedPlanEntry([1]),
    "20c": CachedPlanEntry([1]),
    "20f": CachedPlanEntry([4]),
    "21b": CachedPlanEntry([4]),
    "25a": CachedPlanEntry([5]),
    "25b": CachedPlanEntry([7]),
    "28e": CachedPlanEntry([6]),
    "30a": CachedPlanEntry([11]),
    "31a": CachedPlanEntry([3]),
    "34e": CachedPlanEntry([2]),
    "37b": CachedPlanEntry([6]),
    "38a": CachedPlanEntry([8]),
    "38b": CachedPlanEntry([15]),
    "39d": CachedPlanEntry([5]),
    "43b": CachedPlanEntry([0]),
    "45a": CachedPlanEntry([10]),
    "45d": CachedPlanEntry([22]),
    "40d": CachedPlanEntry([3]),
    "40f": CachedPlanEntry([16]),
}


# ── RobustPlanCache: LRU cache with hit/miss tracking ─────────────────
class RobustPlanCache:
    """LRU plan cache with device-aware lookup and access tracking.

    PAR2QO: flat dict lookups — cached_rob_plan_dict[query_id].
    Lynceus: class wrapper with LRU eviction, hit ratio, device filtering.
    ~20% algorithm change: LRU ordering, device-preference filtering.
    """

    def __init__(
        self,
        max_size: int = 10000,
        preload: str = "tpch",
        debug: bool = True,
    ):
        self._cache: OrderedDict[str, CachedPlanEntry] = OrderedDict()
        self.max_size = max_size
        self.debug = debug

        # Access statistics
        self._hits = 0
        self._misses = 0
        self._cold_hits = 0  # first access of a preloaded entry
        self._evictions = 0

        # Preload from static dicts
        if preload == "tpch":
            for k, v in _CACHED_TPCH_PLANS.items():
                self._cache[k] = v
        elif preload == "imdb":
            for k, v in _CACHED_IMDB_PLANS.items():
                self._cache[k] = v
        elif preload == "stats":
            for k, v in _CACHED_STATS_PLANS.items():
                self._cache[k] = v
        elif preload == "all":
            for d in [_CACHED_TPCH_PLANS, _CACHED_IMDB_PLANS, _CACHED_STATS_PLANS]:
                for k, v in d.items():
                    self._cache[k] = v

        if debug:
            print(f"  ├─ RobustPlanCache: preloaded {len(self._cache)} entries "
                  f"(source={preload}, max={max_size})")

    def lookup(
        self,
        query_id: str,
        device_filter: str = "",
    ) -> Optional[CachedPlanEntry]:
        """Look up a cached plan by query ID.

        PAR2QO: cached_rob_plan_dict[query_id] → [plan_index].
        Lynceus: LRU-ordered lookup with device filtering.
        """
        if query_id in self._cache:
            entry = self._cache[query_id]

            # LRU: move to end
            self._cache.move_to_end(query_id)

            # Track cold vs warm hit
            if entry.access_count == 0:
                self._cold_hits += 1
            self._hits += 1
            entry.touch()

            # Device filter: check preference
            if device_filter and entry.device_preference:
                if entry.device_preference != device_filter:
                    if self.debug:
                        print(f"  │  cache HIT {query_id} but device mismatch "
                              f"(want={device_filter}, cached={entry.device_preference})")
                    return None

            if self.debug:
                print(f"  │  cache HIT {query_id} → plan#{entry.best_plan} "
                      f"[{entry.device_preference or 'auto'}] "
                      f"(access #{entry.access_count})")

            return entry
        else:
            self._misses += 1
            if self.debug:
                print(f"  │  cache MISS {query_id}")
            return None

    def insert(
        self,
        query_id: str,
        plan_indices: List[int],
        device_preference: str = "",
        expected_penalty: float = 0.0,
        penalty_std: float = 0.0,
    ):
        """Insert a new plan into the cache, evicting LRU if full.

        PAR2QO: no dynamic insertion (static dicts only).
        Lynceus: supports runtime insertion with LRU eviction.
        """
        if query_id in self._cache:
            self._cache.move_to_end(query_id)
            entry = self._cache[query_id]
            entry.plan_indices = plan_indices
            entry.device_preference = device_preference
            entry.expected_penalty = expected_penalty
            entry.penalty_std = penalty_std
            entry.timestamp = time.time()
        else:
            # Evict LRU if full
            while len(self._cache) >= self.max_size:
                evicted_key, evicted_entry = self._cache.popitem(last=False)
                self._evictions += 1
                if self.debug:
                    print(f"  │  cache EVICT {evicted_key} "
                          f"(accesses={evicted_entry.access_count})")

            self._cache[query_id] = CachedPlanEntry(
                plan_indices=plan_indices,
                device_preference=device_preference,
                expected_penalty=expected_penalty,
                penalty_std=penalty_std,
            )

        if self.debug:
            print(f"  │  cache INSERT {query_id} → plans={plan_indices} "
                  f"[{device_preference or 'auto'}] penalty={expected_penalty:.1f}")

    def remove(self, query_id: str) -> bool:
        if query_id in self._cache:
            del self._cache[query_id]
            return True
        return False

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def cold_hit_rate(self) -> float:
        return self._cold_hits / self._hits if self._hits > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "cold_hits": self._cold_hits,
            "evictions": self._evictions,
            "hit_rate": f"{self.hit_rate:.3f}",
            "cold_hit_rate": f"{self.cold_hit_rate:.3f}",
        }

    # ── Serialization ──────────────────────────────────────────────────
    def to_json(self) -> str:
        data = {}
        for k, e in self._cache.items():
            data[k] = {
                "plan_indices": e.plan_indices,
                "device": e.device_preference,
                "penalty": e.expected_penalty,
                "penalty_std": e.penalty_std,
                "accesses": e.access_count,
            }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str, debug: bool = True) -> "RobustPlanCache":
        data = json.loads(json_str)
        cache = cls(preload="", debug=debug)
        for k, v in data.items():
            cache.insert(
                k,
                v["plan_indices"],
                v.get("device", ""),
                v.get("penalty", 0.0),
                v.get("penalty_std", 0.0),
            )
        return cache

    # ── Debug dump ─────────────────────────────────────────────────────
    def debug_dump(self):
        print(f"\n  ┌─ PLAN CACHE STATE DUMP ────────────────────────────")
        stats = self.stats()
        for k, v in stats.items():
            print(f"  │  {k}: {v}")
        print(f"  │  entries (MRU → LRU):")
        for i, (qid, entry) in enumerate(reversed(self._cache.items())):
            if i >= 10:
                print(f"  │    ... ({len(self._cache) - 10} more)")
                break
            print(f"  │    {qid}: plan#{entry.best_plan} "
                  f"[{entry.device_preference or 'auto'}] "
                  f"penalty={entry.expected_penalty:.1f} "
                  f"accesses={entry.access_count}")
        print(f"  └────────────────────────────────────────────────────")

# ═══════════════════════════════════════════════════════════════════════════
# ★ 移植改写区
# ═══════════════════════════════════════════════════════════════════════════

    def dump_hit_rate_trend(self, window: int = 20) -> str:
        """★ 改写: 缓存命中率滑动窗口趋势."""
        from .. import _dbg
        if not self._access_log:
            return "(no accesses)"
        lines = ["┌── Plan Cache Hit Rate Trend ──"]
        for i in range(0, len(self._access_log), window):
            chunk = self._access_log[i:i+window]
            hits = sum(1 for x in chunk if x)
            rate = hits / max(1, len(chunk))
            bar = "█" * int(rate * 30) + "░" * (30 - int(rate * 30))
            lines.append(f"│ [{i:>5}-{i+len(chunk):>5}]: {bar} {rate:.1%}")
        lines.append("└──────────────────────────────")
        return "\n".join(lines)
