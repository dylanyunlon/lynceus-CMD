"""
lynceus/distributed/fsdp_compat.py — FSDP compatibility layer for cost model.

Architecture references (ported/adapted from):
  - PAR2QO plan_reduction_by_similarity.py (par2qo/code/plan_reduction_by_similarity.py:1-217)
    → JS_distance(), kl_divergence() — distribution similarity metrics
    → k_center_greedy() — greedy center selection for plan reduction
    → reduce_matrix() — iterative closest-pair elimination
    → plot_2d_matrix() — matrix heatmap visualisation
    → plot_all_cost_distribution() — per-plan cost distribution plots
  - PAR2QO prep_plan_set.py (par2qo/code/prep_plan_set.py:1-203)
    → plan set preparation and indexing patterns
  - FairScale FSDP (fairscale/nn/data_parallel/fully_sharded_data_parallel.py)
    → FullyShardedDataParallel wrapper, _shard_parameters()
  - PyTorch FSDP (torch/distributed/fsdp/fully_sharded_data_parallel.py)
    → ShardingStrategy enum, MixedPrecision config

Modifications from upstream references (~20% original):
  - Removed: matplotlib plotting, numpy array operations (from PAR2QO)
  - Removed: actual tensor sharding, CUDA memory management (from FSDP)
  - Added:   Cost estimation for FSDP-style parameter sharding strategies
  - Added:   Shard assignment optimisation using adapted k-center
  - Added:   Comprehensive debug instrumentation at each sharding decision
  - Changed: 'plans' → 'cost model parameter shards'
  - Changed: JS_distance → shard similarity metric
  - Changed: reduce_matrix → shard consolidation for memory efficiency

Design:
  Lynceus's cost model parameters can be distributed using FSDP-style
  sharding: each worker holds a shard of the parameter space and all-gathers
  the full parameters when needed for cost estimation. This module models
  the cost/benefit of different sharding strategies and provides a
  compatibility layer that mirrors PyTorch FSDP's API.
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum, auto

from .sync import SyncConfig, SyncStrategy, estimate_sync_cost, SyncMetrics

_MOD_TAG = "FST"
import os as _os, sys as _sys
_LYNCEUS_DBG = _os.environ.get("LYNCEUS_DEBUG", "1")

def _dbg(tag: str, msg: str):
    if _LYNCEUS_DBG != "0":
        print(f"[{_MOD_TAG}·{tag}] {msg}", file=_sys.stderr, flush=True)

_tr = _dbg

def _dbg_state(tag, **kwargs):
    """键值对状态快照."""
    if _LYNCEUS_DBG == "0":
        return
    parts = [f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v!r}" for k, v in kwargs.items()]
    _dbg(tag, " | ".join(parts))
  # 兼容旧调用


logger = logging.getLogger(__name__)


# ─── Sharding Strategy ──────────────────────────────────────────────────────
# Mirrors PyTorch FSDP ShardingStrategy enum.

class FSDPShardingStrategy(Enum):
    """FSDP sharding strategy — mirrors torch.distributed.fsdp.ShardingStrategy."""
    FULL_SHARD = auto()          # ZeRO-3: shard params + grads + optimizer
    SHARD_GRAD_OP = auto()       # ZeRO-2: shard grads + optimizer
    NO_SHARD = auto()            # DDP: replicate everything
    HYBRID_SHARD = auto()        # Shard within node, replicate across nodes


class MixedPrecisionPolicy(Enum):
    """Parameter storage precision — inspired by FSDP MixedPrecision."""
    FP32 = auto()
    FP16 = auto()
    BF16 = auto()
    FP8_E4M3 = auto()   # INV-6: E4M3 preferred under block-wise scaling


# ─── Distribution Similarity ────────────────────────────────────────────────
# Directly adapted from par2qo plan_reduction_by_similarity.py
# JS_distance (line 4-8) and kl_divergence (line 12-24).

def kl_divergence(p: List[float], q: List[float],
                  symmetric: bool = False) -> float:
    """KL 散度.
    改写: 加 symmetric=True 选项返回 (KL(p||q)+KL(q||p))/2;
    Laplace 平滑用可调 epsilon."""
    _dbg_state("KL", p_len=len(p), q_len=len(q), symmetric=symmetric)
    epsilon = 1e-10
    n = len(p)
    p_sum = sum(p) + epsilon * n
    q_sum = sum(q) + epsilon * n

    kl_pq = 0.0
    for i in range(n):
        p_i = (p[i] + epsilon) / p_sum
        q_i = (q[i] + epsilon) / q_sum
        kl_pq += p_i * math.log(p_i / q_i)

    if symmetric:
        # 改写: 对称 KL
        kl_qp = 0.0
        for i in range(n):
            p_i = (p[i] + epsilon) / p_sum
            q_i = (q[i] + epsilon) / q_sum
            kl_qp += q_i * math.log(q_i / p_i)
        result = (kl_pq + kl_qp) / 2.0
        _dbg("KL", f"symmetric: KL(p||q)={kl_pq:.4f}, KL(q||p)={kl_qp:.4f}, avg={result:.4f}")
        return result

    _dbg("KL", f"KL(p||q)={kl_pq:.4f}")
    return kl_pq


def js_distance(p: List[float], q: List[float]) -> float:
    """Jensen-Shannon distance between two distributions.

    _dbg("JS_DISTA", "ENTER js_distance")
    Ported from par2qo/code/plan_reduction_by_similarity.py:JS_distance (line 4).

    Original:
        m = 0.495 * (np.array(p) + np.array(q))
        js_div = 0.495 * kl_divergence(p, m) + 0.495 * kl_divergence(q, m)
        return np.sqrt(js_div)

    Lynceus: reimplemented without numpy; used for shard similarity.
    """
    _dbg("JS_DISTA", f"js_distance(p={p}, q={q})")
    n = len(p)
    m = [(p[i] + q[i]) / 2.0 for i in range(n)]
    js_div = 0.495 * kl_divergence(p, m) + 0.495 * kl_divergence(q, m)
    return math.sqrt(max(0.0, js_div))


# ─── K-Center Greedy Shard Selection ────────────────────────────────────────
# Adapted from par2qo plan_reduction_by_similarity.py:k_center_greedy (line 30-80).
# In PAR2QO this selects K representative plans from a larger set;
# here we select K representative shards for cost model distribution.

def k_center_greedy_shards(
    shard_profiles: List[List[float]],
    k: int,
    first_shard: Optional[int] = None,
    debug_print: bool = True,
) -> Tuple[List[int], Dict[int, List[int]]]:
    """Greedy K-Center selection for representative shard profiles.

    Ported from par2qo/code/plan_reduction_by_similarity.py:k_center_greedy (line 30).

    Original used JS_distance over plan cost distributions;
    we use it over shard access-frequency profiles to find the
    K most representative shards for pre-fetching.

    Original:
        centers = [np.random.choice(n, 1)[0]]
        distances = np.full(n, np.inf)
        for _ in range(1, k):
            for i in range(n):
                distances[i] = min(distances[i], distance_func(...))
            new_center = np.argmax(distances)
            centers.append(new_center)

    Returns: (selected_indices, assignments_dict)
    """
    n = len(shard_profiles)
    if k >= n:
        centers = list(range(n))
        assignments = {c: [c] for c in centers}
        return centers, assignments

    centers = [first_shard if first_shard is not None else 0]
    distances = [float('inf')] * n

    for iteration in range(1, k):
        # Update distances to nearest selected center
        for i in range(n):
            d = js_distance(shard_profiles[i], shard_profiles[centers[-1]])
            if d < distances[i]:
                distances[i] = d

        # Select farthest point from its closest center
        new_center = max(range(n), key=lambda i: distances[i])
        centers.append(new_center)

        if debug_print:
            print(f"  [fsdp] k-center iter {iteration}: selected shard {new_center} "
                  f"(max_dist={distances[new_center]:.4f})")

    # Assign each shard to nearest center
    assignments: Dict[int, List[int]] = {c: [] for c in centers}
    for i in range(n):
        closest = min(centers, key=lambda c: js_distance(shard_profiles[i], shard_profiles[c]))
        assignments[closest].append(i)

    if debug_print:
        print(f"  [fsdp] k-center result: {len(centers)} centers")
        for c, members in assignments.items():
            print(f"    center {c}: {len(members)} shards assigned")

    return centers, assignments


# ─── Shard Consolidation via Matrix Reduction ────────────────────────────────
# Adapted from par2qo plan_reduction_by_similarity.py:reduce_matrix (line 175-210).
# Original iteratively removed the closest pair of plans;
# we iteratively merge the most similar shards.

def reduce_shard_matrix(
    similarity_matrix: List[List[float]],
    target_shards: int = 4,
    debug_print: bool = True,
) -> Tuple[List[List[float]], List[int]]:
    """Reduce a shard similarity matrix by merging closest pairs.

    Ported from par2qo/code/plan_reduction_by_similarity.py:reduce_matrix (line 175).

    Original:
        matrix[i][i] = float('inf')
        while matrix.shape[0] > target_rows:
            min_index = np.unravel_index(np.argmin(matrix), matrix.shape)
            matrix = np.delete(matrix, row_to_remove, axis=0/1)

    Lynceus: reimplemented without numpy, returns surviving shard indices.
    """
    n = len(similarity_matrix)
    # Deep copy + set diagonal to inf
    mat = [row[:] for row in similarity_matrix]
    surviving = list(range(n))
    for i in range(n):
        mat[i][i] = float('inf')

    while len(mat) > target_shards:
        # Find minimum (most similar pair)
        min_val = float('inf')
        min_i, min_j = 0, 0
        for i in range(len(mat)):
            for j in range(len(mat)):
                if mat[i][j] < min_val:
                    min_val = mat[i][j]
                    min_i, min_j = i, j

        # Remove the first index of the pair (arbitrary choice)
        remove_idx = min_i
        removed_shard = surviving[remove_idx]

        if debug_print:
            print(f"  [fsdp] merge: removing shard {removed_shard} "
                  f"(similar to {surviving[min_j]}, dist={min_val:.4f})")

        # Delete row and column
        mat = [row[:remove_idx] + row[remove_idx + 1:]
               for idx, row in enumerate(mat) if idx != remove_idx]
        surviving.pop(remove_idx)

    if debug_print:
        print(f"  [fsdp] reduced to {len(surviving)} shards: {surviving}")

    return mat, surviving


# ─── FSDP Compatibility Wrapper ──────────────────────────────────────────────

@dataclass
class FSDPConfig:
    """Configuration for FSDP-style cost model distribution."""
    sharding_strategy: FSDPShardingStrategy = FSDPShardingStrategy.FULL_SHARD
    mixed_precision: MixedPrecisionPolicy = MixedPrecisionPolicy.FP32
    n_workers: int = 4
    # Total cost model parameters (count)
    total_params: int = 64
    # Network config
    sync_config: SyncConfig = field(default_factory=SyncConfig)
    # Pre-fetch: how many shards to pre-fetch based on access patterns
    prefetch_shards: int = 2
    debug_print: bool = True


@dataclass
class ShardInfo:
    """Information about one parameter shard."""
    partition_id: int
    owner_worker: str
    n_params: int
    size_bytes: int
    # Access frequency profile (for k-center optimisation)
    access_profile: List[float] = field(default_factory=list)

    def dump_debug(self, prefix: str = "") -> str:
        _dbg("DUMP_DEB", f"dump_debug(prefix={prefix})")
        lines = [
            f"{prefix}╔══ ShardInfo #{self.partition_id} ══════════════════════",
            f"{prefix}║ owner        = {self.owner_worker}",
            f"{prefix}║ n_params     = {self.n_params}",
            f"{prefix}║ size_bytes   = {self.size_bytes}",
            f"{prefix}║ access_prof  = {self.access_profile[:5]}{'...' if len(self.access_profile) > 5 else ''}",
            f"{prefix}╚══════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


@dataclass
class FSDPCostEstimate:
    """Cost estimate for an FSDP-wrapped forward/backward pass."""
    strategy: FSDPShardingStrategy
    # Time breakdown (µs)
    allgather_us: float = 0.0        # all-gather params before forward
    forward_us: float = 0.0          # forward computation
    reduce_scatter_us: float = 0.0   # reduce-scatter grads after backward
    memory_bytes: int = 0            # peak memory per worker
    # Communication volume
    comm_bytes: int = 0
    # Total
    total_us: float = 0.0

    def dump_debug(self, prefix: str = "") -> str:
        _dbg("DUMP_DEB", f"dump_debug(prefix={prefix})")
        lines = [
            f"{prefix}╔══ FSDPCostEstimate ═══════════════════════════════",
            f"{prefix}║ strategy         = {self.strategy.name}",
            f"{prefix}║ allgather_us     = {self.allgather_us:,.1f}",
            f"{prefix}║ forward_us       = {self.forward_us:,.1f}",
            f"{prefix}║ reduce_scatter_us= {self.reduce_scatter_us:,.1f}",
            f"{prefix}║ total_us         = {self.total_us:,.1f} ({self.total_us/1000:.3f} ms)",
            f"{prefix}║ memory_bytes     = {self.memory_bytes:,} ({self.memory_bytes/(1024**2):.1f} MB)",
            f"{prefix}║ comm_bytes       = {self.comm_bytes:,} ({self.comm_bytes/(1024**2):.1f} MB)",
            f"{prefix}╚═══════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


class FSDPCompatLayer:
    """FSDP compatibility layer for distributed cost model parameter management.

    Models the costs and benefits of wrapping cost model parameters in an
    FSDP-style distributed wrapper, where parameters are sharded across
    workers and all-gathered on demand.

    The shard assignment uses the k-center greedy algorithm adapted from
    PAR2QO's plan_reduction_by_similarity.py, where instead of selecting
    representative query plans, we select representative parameter shards
    for pre-fetching.
    """

    def __init__(self, config: Optional[FSDPConfig] = None):
        self._config = config or FSDPConfig()
        self._shards: List[ShardInfo] = []
        self._worker_ids = [f"worker_{i}" for i in range(self._config.n_workers)]
        self._initialised = False

        if self._config.debug_print:
            print(f"\n[fsdp_compat] Initialized FSDPCompatLayer")
            print(f"  strategy    = {self._config.sharding_strategy.name}")
            print(f"  precision   = {self._config.mixed_precision.name}")
            print(f"  n_workers   = {self._config.n_workers}")
            print(f"  total_params= {self._config.total_params}")

    def _bytes_per_param(self) -> int:
        """Bytes per parameter based on precision policy."""
        _dbg("_BYTES_P", "ENTER _bytes_per_param")
        policy_bytes = {
            MixedPrecisionPolicy.FP32: 4,
            MixedPrecisionPolicy.FP16: 2,
            MixedPrecisionPolicy.BF16: 2,
            MixedPrecisionPolicy.FP8_E4M3: 1,  # INV-6
        }
        return policy_bytes.get(self._config.mixed_precision, 4)

    def initialise_shards(self, debug_print: Optional[bool] = None) -> List[ShardInfo]:
        """Create parameter shards based on sharding strategy.

        _dbg("INITIALI", "ENTER initialise_shards")
        Distributes cost model parameters across workers following
        the configured FSDP sharding strategy.
        """
        _dbg("INITIALI", f"initialise_shards(debug_print={debug_print})")
        dp = debug_print if debug_print is not None else self._config.debug_print
        bpp = self._bytes_per_param()
        total_params = self._config.total_params
        n_workers = self._config.n_workers

        if dp:
            print(f"\n  [fsdp_compat] Initialising shards: {total_params} params, "
                  f"{bpp}B/param, {n_workers} workers")

        self._shards = []

        if self._config.sharding_strategy == FSDPShardingStrategy.NO_SHARD:
            # DDP: every worker has all params (one big shard per worker)
            for i in range(n_workers):
                shard = ShardInfo(
                    partition_id=i,
                    owner_worker=self._worker_ids[i],
                    n_params=total_params,
                    size_bytes=total_params * bpp,
                    access_profile=[1.0] * n_workers,
                )
                self._shards.append(shard)
        else:
            # FULL_SHARD / SHARD_GRAD_OP / HYBRID_SHARD:
            # partition params across workers
            params_per_shard = total_params // n_workers
            remainder = total_params % n_workers
            for i in range(n_workers):
                n_p = params_per_shard + (1 if i < remainder else 0)
                # Access profile: higher for local, lower for remote
                access_prof = [0.2] * n_workers
                access_prof[i] = 1.0  # local access most frequent
                shard = ShardInfo(
                    partition_id=i,
                    owner_worker=self._worker_ids[i],
                    n_params=n_p,
                    size_bytes=n_p * bpp,
                    access_profile=access_prof,
                )
                self._shards.append(shard)

        self._initialised = True

        if dp:
            for s in self._shards:
                print(s.dump_debug("    "))

        return self._shards

    def estimate_forward_cost(self, debug_print: Optional[bool] = None) -> FSDPCostEstimate:
        """Estimate cost of one forward pass with FSDP wrapping.

        _dbg("ESTIMATE", "ENTER estimate_forward_cost")
        For FULL_SHARD: all-gather params → forward → reduce-scatter grads
        For SHARD_GRAD_OP: all-gather params → forward → all-reduce grads
        For NO_SHARD: forward only (no sharding overhead)
        """
        _dbg("ESTIMATE", f"estimate_forward_cost(debug_print={debug_print})")
        if not self._initialised:
            self.initialise_shards()

        dp = debug_print if debug_print is not None else self._config.debug_print
        bpp = self._bytes_per_param()
        strategy = self._config.sharding_strategy
        n_workers = self._config.n_workers
        total_bytes = self._config.total_params * bpp

        # ── All-gather cost: collect full params from all shards ──
        if strategy == FSDPShardingStrategy.NO_SHARD:
            allgather_us = 0.0
            allgather_bytes = 0
        elif strategy == FSDPShardingStrategy.FULL_SHARD:
            # All-gather: each worker sends its shard, receives all others
            # Total comm = (n-1)/n * total_bytes per worker
            allgather_bytes = int(total_bytes * (n_workers - 1) / n_workers)
            allgather_sync = estimate_sync_cost(
                data_bytes=allgather_bytes,
                config=self._config.sync_config,
                debug_print=False,
            )
            allgather_us = allgather_sync.total_time_us
        else:
            # SHARD_GRAD_OP, HYBRID: similar all-gather
            allgather_bytes = int(total_bytes * (n_workers - 1) / n_workers)
            allgather_sync = estimate_sync_cost(
                data_bytes=allgather_bytes,
                config=self._config.sync_config,
                debug_print=False,
            )
            allgather_us = allgather_sync.total_time_us

        # ── Forward computation: cost model evaluation ──
        # ~100ns per parameter for cost model evaluation
        forward_us = self._config.total_params * 0.0001  # 100ns each → µs

        # ── Reduce-scatter cost: distribute gradients ──
        if strategy in (FSDPShardingStrategy.NO_SHARD, FSDPShardingStrategy.SHARD_GRAD_OP):
            reduce_scatter_us = 0.0
            rs_bytes = 0
        else:
            rs_bytes = allgather_bytes  # symmetric communication
            rs_sync = estimate_sync_cost(
                data_bytes=rs_bytes,
                config=self._config.sync_config,
                debug_print=False,
            )
            reduce_scatter_us = rs_sync.total_time_us

        # ── Memory per worker ──
        if strategy == FSDPShardingStrategy.FULL_SHARD:
            # Shard + temporarily all-gathered full params
            memory = total_bytes // n_workers + total_bytes  # shard + full during forward
        elif strategy == FSDPShardingStrategy.NO_SHARD:
            memory = total_bytes  # full copy
        else:
            memory = total_bytes // n_workers + total_bytes

        total_comm = allgather_bytes + rs_bytes
        total_us = allgather_us + forward_us + reduce_scatter_us

        estimate = FSDPCostEstimate(
            strategy=strategy,
            allgather_us=allgather_us,
            forward_us=forward_us,
            reduce_scatter_us=reduce_scatter_us,
            memory_bytes=memory,
            comm_bytes=total_comm,
            total_us=total_us,
        )

        if dp:
            print(f"\n  [fsdp_compat] Forward cost estimate:")
            print(estimate.dump_debug("    "))

        return estimate

    def optimise_prefetch(self, debug_print: Optional[bool] = None) -> Tuple[List[int], Dict[int, List[int]]]:
        """Select shards to pre-fetch using k-center greedy.

        _dbg("OPTIMISE", "ENTER optimise_prefetch")
        Uses adapted k-center algorithm from PAR2QO's
        plan_reduction_by_similarity.py to find the most
        representative shards for pre-fetching.
        """
        _dbg("OPTIMISE", f"optimise_prefetch(debug_print={debug_print})")
        if not self._initialised:
            self.initialise_shards()

        dp = debug_print if debug_print is not None else self._config.debug_print

        profiles = [s.access_profile for s in self._shards]
        k = min(self._config.prefetch_shards, len(profiles))

        if dp:
            print(f"\n  [fsdp_compat] Optimising prefetch: selecting {k} shards from {len(profiles)}")

        centers, assignments = k_center_greedy_shards(
            profiles, k, debug_print=dp,
        )
        return centers, assignments

    def compare_strategies(self, debug_print: bool = True) -> Dict[str, FSDPCostEstimate]:
        """Compare all FSDP strategies — for experiment analysis."""
        _dbg("COMPARE_", f"compare_strategies(debug_print={debug_print})")
        if debug_print:
            print(f"\n{'='*60}")
            print(f"[fsdp_compat] Strategy Comparison: {self._config.total_params} params, "
                  f"{self._config.n_workers} workers")
            print(f"{'='*60}")

        results = {}
        for strategy in FSDPShardingStrategy:
            cfg = FSDPConfig(
                sharding_strategy=strategy,
                mixed_precision=self._config.mixed_precision,
                n_workers=self._config.n_workers,
                total_params=self._config.total_params,
                sync_config=self._config.sync_config,
                debug_print=False,
            )
            layer = FSDPCompatLayer(cfg)
            estimate = layer.estimate_forward_cost(debug_print=False)
            results[strategy.name] = estimate

            if debug_print:
                print(f"\n  {strategy.name}: total={estimate.total_us:,.1f}µs, "
                      f"mem={estimate.memory_bytes:,}B, comm={estimate.comm_bytes:,}B")

        best = min(results.items(), key=lambda x: x[1].total_us)
        if debug_print:
            print(f"\n  → Best for wire_delay: {best[0]} at {best[1].total_us:,.1f}µs")

        return results

    def dump_state(self) -> str:
        """Full state dump for breakpoint inspection."""
        _dbg("DUMP_STA", "ENTER dump_state")
        lines = [
            "╔══ FSDPCompatLayer State ══════════════════════════════",
            f"║ strategy      = {self._config.sharding_strategy.name}",
            f"║ precision     = {self._config.mixed_precision.name}",
            f"║ n_workers     = {self._config.n_workers}",
            f"║ total_params  = {self._config.total_params}",
            f"║ bytes/param   = {self._bytes_per_param()}",
            f"║ total_bytes   = {self._config.total_params * self._bytes_per_param():,}",
            f"║ initialised   = {self._initialised}",
            f"║ n_shards      = {len(self._shards)}",
        ]
        for s in self._shards:
            lines.append(f"║   shard_{s.partition_id}: owner={s.owner_worker}, "
                       f"params={s.n_params}, bytes={s.size_bytes}")
        lines.append("╚════════════════════════════════════════════════════════")
        return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════
# ★ 移植改写区
# ═══════════════════════════════════════════════════════════════════════════

    def dump_shard_visual(self) -> str:
        """★ 改写: ASCII 分片可视化 — 断点调试用."""
        _dbg("DUMP_SHA", "ENTER dump_shard_visual")
        if not self._shards:
            return "(no shards)"
        lines = ["┌── FSDP Shard Layout ──"]
        total = self._config.total_params
        for s in self._shards:
            bar_len = max(1, int(s.param_count / max(1, total) * 50))
            bar = "█" * bar_len
            lines.append(f"│ Shard{s.partition_id} [{s.device}]: {bar} "
                         f"({s.param_count} params, {s.size_bytes:,}B)")
        lines.append(f"│ Strategy: {self._config.strategy.name}")
        lines.append(f"│ Mixed precision: {self._config.mixed_precision.name}")
        lines.append("└──────────────────────")
        return "\n".join(lines)

    def estimate_memory_per_device(self) -> "Dict[str, int]":
        """★ 改写: 设备级内存占用估算 (断点辅助).

        _dbg("ESTIMATE", "ENTER estimate_memory_per_device")
        考虑: 参数 + 梯度 + 优化器状态 (Adam 需 2x 额外).
        """
        mem: "Dict[str, int]" = {}
        multiplier = {"FULL_SHARD": 1, "SHARD_GRAD_OP": 2, "NO_SHARD": 3}
        m = multiplier.get(self._config.strategy.name, 1)
        # Adam: params + grads + 2 × optimizer state
        state_factor = 4  # param + grad + m + v
        for s in self._shards:
            dev_mem = s.size_bytes * state_factor * m
            mem[s.device] = mem.get(s.device, 0) + dev_mem
        from . import _dbg
        _dbg("fsdp_mem", f"fsdp_memory: {mem}")
        return mem
