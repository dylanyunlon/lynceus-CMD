"""
lynceus/distributed/fsdp_compat.py — FSDP compatibility layer.

算法改动:
    1. k_center_greedy: lazy distance update
    2. reduce_shard_matrix: dendrogram height tracking
    3. estimate_forward_cost: allgather/forward overlap
"""
from __future__ import annotations
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum, auto
from .sync import SyncConfig, SyncStrategy, estimate_sync_cost, SyncMetrics

logger = logging.getLogger(__name__)

class FSDPShardingStrategy(Enum):
    FULL_SHARD = auto()
    SHARD_GRAD_OP = auto()
    NO_SHARD = auto()
    HYBRID_SHARD = auto()

class MixedPrecisionPolicy(Enum):
    FP32 = auto()
    FP16 = auto()
    BF16 = auto()
    FP8_E4M3 = auto()


def kl_divergence(p: List[float], q: List[float]) -> float:
    epsilon = 1e-10
    n = len(p)
    p_sum = sum(p) + epsilon * n
    q_sum = sum(q) + epsilon * n
    result = 0.0
    for i in range(n):
        p_i = (p[i] + epsilon) / p_sum
        q_i = (q[i] + epsilon) / q_sum
        result += p_i * math.log(p_i / q_i)
    return result


def js_distance(p: List[float], q: List[float]) -> float:
    n = len(p)
    m = [(p[i] + q[i]) / 2.0 for i in range(n)]
    js_div = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return math.sqrt(max(0.0, js_div))


def k_center_greedy_shards(
    shard_profiles: List[List[float]], k: int,
    first_shard: Optional[int] = None,
    debug_print: bool = True,
) -> Tuple[List[int], Dict[int, List[int]]]:
    """改动: lazy distance update — 每轮只计算到新 center 的距离。

    原版: 每轮 for i in range(n): distances[i] = min(distances[i], dist(i, new_center))
    这已经是 O(n) per iteration, 但原版的 assignment 阶段对每个 point 遍历所有 center。
    新版: assignment 也用 cached nearest, O(n) 而非 O(nk)。
    """
    n = len(shard_profiles)
    if k >= n:
        centers = list(range(n))
        return centers, {c: [c] for c in centers}

    centers = [first_shard if first_shard is not None else 0]
    distances = [float('inf')] * n
    nearest_center = [centers[0]] * n

    # 初始化: 到第一个 center 的距离
    for i in range(n):
        distances[i] = js_distance(shard_profiles[i], shard_profiles[centers[0]])
        nearest_center[i] = centers[0]

    for iteration in range(1, k):
        new_center = max(range(n), key=lambda i: distances[i])
        centers.append(new_center)

        # 改动: lazy update — 只更新到新 center 的距离
        for i in range(n):
            d_new = js_distance(shard_profiles[i], shard_profiles[new_center])
            if d_new < distances[i]:
                distances[i] = d_new
                nearest_center[i] = new_center

        if debug_print:
            print(f"  [fsdp] k-center iter {iteration}: shard {new_center} "
                  f"(max_dist={distances[new_center]:.4f})")

    # Assignment 直接用 cached nearest_center (O(n) 而非 O(nk))
    assignments: Dict[int, List[int]] = {c: [] for c in centers}
    for i in range(n):
        assignments[nearest_center[i]].append(i)

    return centers, assignments


def reduce_shard_matrix(
    similarity_matrix: List[List[float]], target_shards: int = 4,
    debug_print: bool = True,
) -> Tuple[List[List[float]], List[int], List[float]]:
    """改动: 返回 dendrogram merge distances (第三个返回值)。

    原版: 只返回 (reduced_matrix, surviving_indices)
    新版: 额外返回每步 merge 的距离值, 用于评估 clustering quality
    """
    n = len(similarity_matrix)
    mat = [row[:] for row in similarity_matrix]
    surviving = list(range(n))
    for i in range(n):
        mat[i][i] = float('inf')

    merge_distances: List[float] = []

    while len(mat) > target_shards:
        min_val = float('inf')
        min_i, min_j = 0, 0
        for i in range(len(mat)):
            for j in range(len(mat)):
                if mat[i][j] < min_val:
                    min_val = mat[i][j]
                    min_i, min_j = i, j

        merge_distances.append(min_val)
        remove_idx = min_i
        removed_shard = surviving[remove_idx]

        if debug_print:
            print(f"  [fsdp] merge: remove shard {removed_shard} "
                  f"(sim to {surviving[min_j]}, dist={min_val:.4f})")

        mat = [row[:remove_idx] + row[remove_idx + 1:]
               for idx, row in enumerate(mat) if idx != remove_idx]
        surviving.pop(remove_idx)

    if debug_print and merge_distances:
        print(f"  [fsdp] merge quality: min_dist={min(merge_distances):.4f}, "
              f"max_dist={max(merge_distances):.4f}, "
              f"mean_dist={sum(merge_distances)/len(merge_distances):.4f}")

    return mat, surviving, merge_distances


@dataclass
class FSDPConfig:
    sharding_strategy: FSDPShardingStrategy = FSDPShardingStrategy.FULL_SHARD
    mixed_precision: MixedPrecisionPolicy = MixedPrecisionPolicy.FP32
    n_workers: int = 4
    total_params: int = 64
    sync_config: SyncConfig = field(default_factory=SyncConfig)
    prefetch_shards: int = 2
    debug_print: bool = True

@dataclass
class ShardInfo:
    shard_id: int
    owner_worker: str
    n_params: int
    size_bytes: int
    access_profile: List[float] = field(default_factory=list)

@dataclass
class FSDPCostEstimate:
    strategy: FSDPShardingStrategy
    allgather_us: float = 0.0
    forward_us: float = 0.0
    reduce_scatter_us: float = 0.0
    memory_bytes: int = 0
    comm_bytes: int = 0
    total_us: float = 0.0
    overlap_savings_us: float = 0.0  # 改动: overlap 节省量
    def dump_debug(self, prefix: str = "") -> str:
        return (f"{prefix}FSDP({self.strategy.name}): {self.total_us:.1f}µs "
                f"(ag={self.allgather_us:.1f}, fwd={self.forward_us:.1f}, "
                f"rs={self.reduce_scatter_us:.1f}, overlap_saved={self.overlap_savings_us:.1f})")


class FSDPCompatLayer:
    def __init__(self, config: Optional[FSDPConfig] = None):
        self._config = config or FSDPConfig()
        self._shards: List[ShardInfo] = []
        self._worker_ids = [f"worker_{i}" for i in range(self._config.n_workers)]
        self._initialised = False
        if self._config.debug_print:
            print(f"\n[fsdp] Init: {self._config.sharding_strategy.name}, "
                  f"{self._config.n_workers} workers, {self._config.total_params} params")

    def _bytes_per_param(self) -> int:
        return {MixedPrecisionPolicy.FP32: 4, MixedPrecisionPolicy.FP16: 2,
                MixedPrecisionPolicy.BF16: 2, MixedPrecisionPolicy.FP8_E4M3: 1
                }.get(self._config.mixed_precision, 4)

    def initialise_shards(self, debug_print: Optional[bool] = None) -> List[ShardInfo]:
        bpp = self._bytes_per_param()
        total_params = self._config.total_params
        n_workers = self._config.n_workers
        self._shards = []
        if self._config.sharding_strategy == FSDPShardingStrategy.NO_SHARD:
            for i in range(n_workers):
                self._shards.append(ShardInfo(
                    shard_id=i, owner_worker=self._worker_ids[i],
                    n_params=total_params, size_bytes=total_params * bpp,
                    access_profile=[1.0] * n_workers))
        else:
            params_per_shard = total_params // n_workers
            remainder = total_params % n_workers
            for i in range(n_workers):
                n_p = params_per_shard + (1 if i < remainder else 0)
                access_prof = [0.2] * n_workers
                access_prof[i] = 1.0
                self._shards.append(ShardInfo(
                    shard_id=i, owner_worker=self._worker_ids[i],
                    n_params=n_p, size_bytes=n_p * bpp,
                    access_profile=access_prof))
        self._initialised = True
        return self._shards

    def estimate_forward_cost(self, debug_print: Optional[bool] = None) -> FSDPCostEstimate:
        """改动: allgather 与 forward 的 overlap。

        原版: total = allgather + forward + reduce_scatter (全串行)
        新版: FSDP 实际上 pipeline allgather 第 i+1 层与 forward 第 i 层:
            overlap_fraction = min(0.8, 1 - 1/n_shards)
            → allgather 的一部分被 forward 隐藏掉了
        """
        if not self._initialised:
            self.initialise_shards()
        dp = debug_print if debug_print is not None else self._config.debug_print
        bpp = self._bytes_per_param()
        strategy = self._config.sharding_strategy
        n_workers = self._config.n_workers
        total_bytes = self._config.total_params * bpp

        if strategy == FSDPShardingStrategy.NO_SHARD:
            allgather_us = 0.0
            allgather_bytes = 0
        else:
            allgather_bytes = int(total_bytes * (n_workers - 1) / n_workers)
            ag_sync = estimate_sync_cost(allgather_bytes, self._config.sync_config,
                                         debug_print=False)
            allgather_us = ag_sync.total_time_us

        forward_us = self._config.total_params * 0.0001

        if strategy in (FSDPShardingStrategy.NO_SHARD, FSDPShardingStrategy.SHARD_GRAD_OP):
            reduce_scatter_us = 0.0
            rs_bytes = 0
        else:
            rs_bytes = allgather_bytes
            rs_sync = estimate_sync_cost(rs_bytes, self._config.sync_config,
                                         debug_print=False)
            reduce_scatter_us = rs_sync.total_time_us

        # 改动: overlap estimation
        n_shards = len(self._shards)
        if n_shards > 1 and strategy != FSDPShardingStrategy.NO_SHARD:
            overlap_fraction = min(0.8, 1.0 - 1.0 / n_shards)
            overlap_savings = allgather_us * overlap_fraction
        else:
            overlap_fraction = 0.0
            overlap_savings = 0.0

        if strategy == FSDPShardingStrategy.FULL_SHARD:
            memory = total_bytes // n_workers + total_bytes
        elif strategy == FSDPShardingStrategy.NO_SHARD:
            memory = total_bytes
        else:
            memory = total_bytes // n_workers + total_bytes

        total_comm = allgather_bytes + rs_bytes
        # 改动: 减去 overlap 节省
        total_us = allgather_us + forward_us + reduce_scatter_us - overlap_savings

        estimate = FSDPCostEstimate(
            strategy=strategy, allgather_us=allgather_us,
            forward_us=forward_us, reduce_scatter_us=reduce_scatter_us,
            memory_bytes=memory, comm_bytes=total_comm,
            total_us=max(0.0, total_us), overlap_savings_us=overlap_savings)

        if dp:
            print(f"  {estimate.dump_debug()}")
        return estimate

    def optimise_prefetch(self, debug_print: Optional[bool] = None):
        if not self._initialised:
            self.initialise_shards()
        profiles = [s.access_profile for s in self._shards]
        k = min(self._config.prefetch_shards, len(profiles))
        dp = debug_print if debug_print is not None else self._config.debug_print
        return k_center_greedy_shards(profiles, k, debug_print=dp)

    def compare_strategies(self, debug_print: bool = True) -> Dict[str, FSDPCostEstimate]:
        results = {}
        for strategy in FSDPShardingStrategy:
            cfg = FSDPConfig(
                sharding_strategy=strategy,
                mixed_precision=self._config.mixed_precision,
                n_workers=self._config.n_workers,
                total_params=self._config.total_params,
                sync_config=self._config.sync_config, debug_print=False)
            layer = FSDPCompatLayer(cfg)
            estimate = layer.estimate_forward_cost(debug_print=False)
            results[strategy.name] = estimate
            if debug_print:
                print(f"  {strategy.name}: {estimate.dump_debug()}")
        return results

    def dump_state(self) -> str:
        return (f"FSDPCompat: {self._config.sharding_strategy.name}, "
                f"{self._config.n_workers} workers, "
                f"{len(self._shards)} shards, init={self._initialised}")
