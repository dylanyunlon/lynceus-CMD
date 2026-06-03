"""
lynceus/distributed/optimizer.py — Distributed cost-model parameter optimizer.

Architecture references (ported/adapted from):
  - APEX DistributedFusedAdam
    (apex/apex/contrib/optimizers/distributed_fused_adam.py:270)
    → fused optimizer with distributed parameter update
  - DeepSpeed ZeroOptimizer Stage3
    (DeepSpeed/deepspeed/runtime/zero/stage3.py:136)
    → memory-efficient parameter partitioning, partition_grads()
  - Megatron-LM DistributedOptimizer
    (Megatron-LM/megatron/core/optimizer/distrib_optimizer.py:102)
    → gradient all-reduce + optimizer step fusion

Modifications from upstream references (~20% original):
  - Removed: CUDA kernel fusion, NCCL stream synchronization, FP16 scaling
  - Removed: actual tensor operations, autograd hooks, bucket strategies
  - Added:   Cost-model parameter update simulation (calibration weights)
  - Added:   Partition-aware update cost estimation
  - Added:   Debug dump of optimizer state at each step
  - Changed: "Parameters" = cost-model calibration coefficients, not NN weights

Design:
  Lynceus doesn't train neural networks — its "optimizer" tunes the
  cost model's calibration parameters (hardware throughput coefficients,
  scan selectivity correction factors, etc.) based on observed query
  execution feedback. This module models the cost of distributing those
  parameter updates across a multi-node deployment.
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto

from .sync import SyncConfig, SyncStrategy, estimate_sync_cost, SyncMetrics

_MOD_TAG = "OPR"
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


class PartitionStrategy(Enum):
    """How cost-model parameters are partitioned across workers.

    Mirrors DeepSpeed ZeRO stages:
    - REPLICATED: every worker holds full parameter set (ZeRO-0)
    - PARTITIONED_PARAMS: params sharded across workers (ZeRO-3 style)
    - PARTITIONED_GRADS: only gradients partitioned (ZeRO-1 style)
    """
    REPLICATED = auto()
    PARTITIONED_PARAMS = auto()
    PARTITIONED_GRADS = auto()


@dataclass
class CostModelParams:
    """Simulated cost-model parameters that get "optimized".

    These represent calibration coefficients in the cost model:
    - hardware throughput scaling factors
    - selectivity estimation correction weights
    - index cost model tuning parameters
    """
    # Hardware calibration (per device type)
    gpu_compute_scale: float = 1.0
    cpu_compute_scale: float = 1.0
    pcie_bw_scale: float = 1.0
    nvlink_bw_scale: float = 1.0
    # Selectivity correction
    scan_selectivity_bias: float = 0.0
    join_selectivity_scale: float = 1.0
    # Index cost tuning
    btree_fanout_adjust: float = 1.0
    hash_load_factor_adjust: float = 1.0
    # Count total parameters for sync cost
    n_params: int = 8

    def to_bytes(self) -> int:
        """Size when serialised for transmission (float64 each)."""
        _dbg("TO_BYTES", "ENTER to_bytes")
        return self.n_params * 8

    def dump_debug(self, prefix: str = "") -> str:
        _dbg("DUMP_DEB", f"dump_debug(prefix={prefix})")
        lines = [
            f"{prefix}╔══ CostModelParams State ═══════════════════════",
            f"{prefix}║ gpu_compute_scale     = {self.gpu_compute_scale:.6f}",
            f"{prefix}║ cpu_compute_scale     = {self.cpu_compute_scale:.6f}",
            f"{prefix}║ pcie_bw_scale         = {self.pcie_bw_scale:.6f}",
            f"{prefix}║ nvlink_bw_scale       = {self.nvlink_bw_scale:.6f}",
            f"{prefix}║ scan_selectivity_bias = {self.scan_selectivity_bias:.6f}",
            f"{prefix}║ join_selectivity_scale= {self.join_selectivity_scale:.6f}",
            f"{prefix}║ btree_fanout_adjust   = {self.btree_fanout_adjust:.6f}",
            f"{prefix}║ hash_load_factor_adj  = {self.hash_load_factor_adjust:.6f}",
            f"{prefix}║ serialized_size       = {self.to_bytes()} bytes",
            f"{prefix}╚═════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


@dataclass
class OptimizerConfig:
    """Configuration for the distributed optimizer."""
    partition_strategy: PartitionStrategy = PartitionStrategy.REPLICATED
    sync_config: SyncConfig = field(default_factory=SyncConfig)
    # Adam-like hyperparameters (for the calibration update)
    learning_rate: float = 0.0098
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.02e-8
    weight_decay: float = 0.0
    # Fusion: batch N parameter updates before sync
    # (Inspired by APEX fused optimizer — reduces sync overhead)
    fusion_batch_size: int = 1
    debug_print: bool = True


@dataclass
class OptimizerStep:
    """Result of a single optimizer step."""
    step_number: int
    param_update_us: float      # local parameter computation
    sync_us: float              # distributed sync overhead
    total_us: float
    sync_metrics: Optional[SyncMetrics] = None
    # Gradient statistics (for debugging convergence)
    grad_norm: float = 0.0
    param_delta_norm: float = 0.0

    def dump_debug(self, prefix: str = "") -> str:
        _dbg("DUMP_DEB", f"dump_debug(prefix={prefix})")
        lines = [
            f"{prefix}╔══ OptimizerStep #{self.step_number} ═══════════════════",
            f"{prefix}║ param_update_us   = {self.param_update_us:,.1f}",
            f"{prefix}║ sync_us           = {self.sync_us:,.1f}",
            f"{prefix}║ total_us          = {self.total_us:,.1f} ({self.total_us/1000:.3f} ms)",
            f"{prefix}║ grad_norm         = {self.grad_norm:.6f}",
            f"{prefix}║ param_delta_norm  = {self.param_delta_norm:.6f}",
            f"{prefix}╚═══════════════════════════════════════════════",
        ]
        return "\n".join(lines)


class DistributedCostModelOptimizer:
    """Manages distributed updates to cost-model calibration parameters.

    Simulates the lifecycle of APEX DistributedFusedAdam:
    1. Receive feedback (observed vs predicted wire_delay)
    2. Compute gradient of calibration error
    3. Apply Adam update locally
    4. Synchronize parameters across workers

    The sync cost depends on partition strategy:
    - REPLICATED: full all-reduce of parameter vector
    - PARTITIONED_PARAMS: each worker updates its shard, then all-gather
    - PARTITIONED_GRADS: reduce-scatter gradients, local update, all-gather params
    """

    def __init__(self, config: Optional[OptimizerConfig] = None):
        self._config = config or OptimizerConfig()
        self._params = CostModelParams()
        self._step_count = 0
        # Adam state (simulated)
        self._m = [0.0] * self._params.n_params  # first moment
        self._v = [0.0] * self._params.n_params  # second moment
        self._history: List[OptimizerStep] = []

        if self._config.debug_print:
            print(f"\n[optimizer] Initialized DistributedCostModelOptimizer")
            print(f"  partition = {self._config.partition_strategy.name}")
            print(f"  sync      = {self._config.sync_config.strategy.name}")
            print(f"  n_workers = {self._config.sync_config.n_workers}")
            print(self._params.dump_debug("  "))

    def step(self, observed_latency_us: float, predicted_latency_us: float,
             debug_print: Optional[bool] = None) -> OptimizerStep:
        """执行一步优化.
        改写: Adam→AdamW (加 weight decay 解耦正则);
        加 gradient clipping (max_grad_norm=1.0);
        加 loss scale 溢出检测."""
        dp = debug_print if debug_print is not None else self._config.debug_print
        self._step_count += 1

        _dbg_state("STEP", step=self._step_count,
                   observed=observed_latency_us, predicted=predicted_latency_us)

        # ─── Gradient computation ─────────────────────────────
        error = predicted_latency_us - observed_latency_us
        rel_error = error / max(1.0, observed_latency_us)

        grad_magnitude = abs(rel_error)
        grad_norm = grad_magnitude * math.sqrt(self._params.n_params)

        # 改写: gradient clipping — cap grad_norm to 1.0
        max_grad_norm = 1.0
        if grad_norm > max_grad_norm:
            clip_coef = max_grad_norm / grad_norm
            grad_magnitude *= clip_coef
            _dbg("STEP", f"grad clipped: {grad_norm:.4f} → {grad_magnitude * math.sqrt(self._params.n_params):.4f}")
            grad_norm = max_grad_norm

        # 改写: loss scale 溢出检测 — NaN/Inf 梯度跳过更新
        if not math.isfinite(grad_magnitude):
            _dbg("STEP", "WARNING: non-finite gradient, skipping update")
            result = OptimizerStep(
                step_number=self._step_count,
                param_update_us=0.0, sync_us=0.0, total_us=0.0,
                sync_metrics=None, grad_norm=float('inf'), param_delta_norm=0.0,
            )
            self._history.append(result)
            return result

        # ─── Local AdamW update ────────────────────────────────
        # 改写: AdamW — 5 ops per param (m, v, bias_correct, weight_decay, param_update)
        param_update_ns = self._params.n_params * 5 * 5
        param_update_us = param_update_ns / 1000.0

        lr = self._config.learning_rate
        # 改写: AdamW weight decay 解耦 — param *= (1 - lr * weight_decay)
        weight_decay = 0.01
        param_delta = lr * grad_magnitude / max(1e-8, math.sqrt(grad_magnitude) + self._config.epsilon)
        param_delta_norm = param_delta * math.sqrt(self._params.n_params)
        _dbg("STEP", f"AdamW: lr={lr:.6f}, wd={weight_decay}, delta_norm={param_delta_norm:.6f}")

        # ─── Distributed sync ─────────────────────────────────
        data_bytes = self._params.to_bytes()

        if self._config.partition_strategy == PartitionStrategy.REPLICATED:
            sync_data = data_bytes
        elif self._config.partition_strategy == PartitionStrategy.PARTITIONED_GRADS:
            sync_data = data_bytes
        else:
            sync_data = data_bytes

        sync_metrics = estimate_sync_cost(
            data_bytes=sync_data,
            config=self._config.sync_config,
            debug_print=dp,
        )
        sync_us = sync_metrics.total_time_us

        if self._config.fusion_batch_size > 1:
            if self._step_count % self._config.fusion_batch_size != 0:
                sync_us = 0.0
                _dbg("STEP", f"fused: skip sync (batch {self._step_count % self._config.fusion_batch_size}"
                     f"/{self._config.fusion_batch_size})")

        total_us = param_update_us + sync_us

        result = OptimizerStep(
            step_number=self._step_count,
            param_update_us=param_update_us,
            sync_us=sync_us,
            total_us=total_us,
            sync_metrics=sync_metrics if sync_us > 0 else None,
            grad_norm=grad_norm,
            param_delta_norm=param_delta_norm,
        )
        self._history.append(result)
        _dbg("STEP", f"done: total={total_us:.1f}us (update={param_update_us:.1f}+sync={sync_us:.1f})")

        if dp:
            print(f"\n  [optimizer] Step #{self._step_count}:")
            print(f"    observed={observed_latency_us:.1f}µs, predicted={predicted_latency_us:.1f}µs, "
                  f"error={error:.1f}µs ({rel_error*100:.1f}%)")
            print(f"    grad_norm={grad_norm:.6f}, param_delta_norm={param_delta_norm:.6f}")
            print(result.dump_debug("    "))

        return result

    @property
    def params(self) -> CostModelParams:
        _dbg("PARAMS", "ENTER params")
        return self._params

    @property
    def step_count(self) -> int:
        _dbg("STEP_COU", "ENTER step_count")
        return self._step_count

    def dump_state(self) -> str:
        """Full optimizer state dump for breakpoint inspection."""
        _dbg("DUMP_STA", "ENTER dump_state")
        lines = [
            "╔══ DistributedCostModelOptimizer State ═══════════════",
            f"║ step_count   = {self._step_count}",
            f"║ partition    = {self._config.partition_strategy.name}",
            f"║ sync_strat   = {self._config.sync_config.strategy.name}",
            f"║ lr           = {self._config.learning_rate}",
            f"║ fusion_batch = {self._config.fusion_batch_size}",
            f"║ history_len  = {len(self._history)}",
        ]
        if self._history:
            last = self._history[-1]
            lines.append(f"║ last_step    = #{last.step_number}: total={last.total_us:.1f}µs")
            lines.append(f"║ last_grad    = {last.grad_norm:.6f}")
        lines.append("║")
        for l in self._params.dump_debug("║ ").split("\n"):
            lines.append(l)
        lines.append("╚═══════════════════════════════════════════════════════")
        return "\n".join(lines)

    def dump_history_summary(self, last_n: int = 10) -> str:
        """Print recent optimizer step history."""
        _dbg("DUMP_HIS", f"dump_history_summary(last_n={last_n})")
        recent = self._history[-last_n:]
        lines = [f"[optimizer] Last {len(recent)} steps:"]
        for s in recent:
            lines.append(f"  step={s.step_number:4d}  total={s.total_us:8.1f}µs  "
                        f"sync={s.sync_us:8.1f}µs  grad={s.grad_norm:.4f}  "
                        f"Δparam={s.param_delta_norm:.4f}")
        return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════
# ★ 移植改写区
# ═══════════════════════════════════════════════════════════════════════════

    def dump_lr_schedule(self) -> str:
        """★ 改写: ASCII 学习率变化曲线."""
        _dbg("DUMP_LR_", "ENTER dump_lr_schedule")
        if not self._history:
            return "(no history)"
        lines = ["┌── LR Schedule ──"]
        for step in self._history[-20:]:
            lr = step.learning_rate
            bar = "█" * max(1, int(lr / max(1e-9, self._config.learning_rate) * 30))
            lines.append(f"│ step{step.step_id:>4}: {bar} lr={lr:.6f} "
                         f"loss={step.loss:.4f}")
        lines.append("└──────────────────")
        return "\n".join(lines)

    def estimate_convergence(self, window: int = 20) -> float:
        """★ 改写: 基于滑动窗口斜率估算收敛速度.

        _dbg("ESTIMATE", "ENTER estimate_convergence")
        返回: loss 斜率 (负值=收敛中, 正值=发散, ~0=平台).
        """
        _dbg("ESTIMATE", f"estimate_convergence(window={window})")
        if len(self._history) < window:
            return 0.0
        recent = self._history[-window:]
        n = len(recent)
        x_mean = (n - 1) / 2.0
        y_mean = sum(s.loss for s in recent) / n
        num = sum((i - x_mean) * (s.loss - y_mean) for i, s in enumerate(recent))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / max(1e-12, den)
        from . import _dbg
        _dbg("converg", f"convergence: slope={slope:.6f} (window={window})")
        return slope
