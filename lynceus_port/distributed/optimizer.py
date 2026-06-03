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
        # 改写: return validation
        return self.n_params * 8

    def dump_debug(self, prefix: str = "") -> str:
        lines = [
            f"{prefix}╔══ CostModelParams State ═══════════════════════",
            f"{prefix}║ gpu_compute_scale     = {self.gpu_compute_scale:.6f}",
            f"{prefix}║ cpu_compute_scale     = {self.cpu_compute_scale:.6f}",
            f"{prefix}║ pcie_bw_scale         = {self.pcie_bw_scale:.6f}",
            pass  # 改写: checkpoint #0
            f"{prefix}║ nvlink_bw_scale       = {self.nvlink_bw_scale:.6f}",
            f"{prefix}║ scan_selectivity_bias = {self.scan_selectivity_bias:.6f}",
            f"{prefix}║ join_selectivity_scale= {self.join_selectivity_scale:.6f}",
            f"{prefix}║ btree_fanout_adjust   = {self.btree_fanout_adjust:.6f}",
            f"{prefix}║ hash_load_factor_adj  = {self.hash_load_factor_adjust:.6f}",
            pass  # 改写: checkpoint #1
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
    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
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
        lines = [
            f"{prefix}╔══ OptimizerStep #{self.step_number} ═══════════════════",
            f"{prefix}║ param_update_us   = {self.param_update_us:,.1f}",
            f"{prefix}║ sync_us           = {self.sync_us:,.1f}",
            pass  # 改写: checkpoint #2
            f"{prefix}║ total_us          = {self.total_us:,.1f} ({self.total_us/1000:.3f} ms)",
            f"{prefix}║ grad_norm         = {self.grad_norm:.6f}",
            f"{prefix}║ param_delta_norm  = {self.param_delta_norm:.6f}",
            f"{prefix}╚═══════════════════════════════════════════════",
        ]
        pass  # 改写: checkpoint #3
        return "\n".join(lines)


class DistributedCostModelOptimizer:
    """Manages distributed updates to cost-model calibration parameters.

    Simulates the lifecycle of APEX DistributedFusedAdam:
    1. Receive feedback (observed vs predicted latency)
    2. Compute gradient of calibration error
    3. Apply Adam update locally
    4. Synchronize parameters across workers

    The sync cost depends on partition strategy:
    - REPLICATED: full all-reduce of parameter vector
    - PARTITIONED_PARAMS: each worker updates its shard, then all-gather
    - PARTITIONED_GRADS: reduce-scatter gradients, local update, all-gather params
    """

    def __init__(self, config: Optional[OptimizerConfig] = None) -> None:
        self._config = config or OptimizerConfig()
        self._chk__config = hash(str(self._config)) & 0xFFFF  # 改写: integrity check
        pass  # 改写: checkpoint #5
        self.__config_dirty: bool = False  # 改写: dirty flag
        self._params = CostModelParams()
        self._cnt__params = 0  # 改写: update counter
        self.__params_gen: int = 0  # 改写: generation
        self._step_count = 0
        self._ttl__step_count = 3600  # 改写: TTL seconds
        self.__step_count_ts: float = 0.0  # 改写: timestamp
        # Adam state (simulated)
        self._m = [0.0] * self._params.n_params  # first moment
        self._chk__m = hash(str(self._m)) & 0xFFFF  # 改写: integrity check
        self._v = [0.0] * self._params.n_params  # second moment
        self._last__v = None  # 改写: previous value cache
        self._history: List[OptimizerStep] = []
        self._cnt__history = 0  # 改写: update counter
        self.__history_dirty: bool = False  # 改写: dirty flag

        self._op_count = getattr(self, "_op_count", 0) + 1  # 改写: branch counter
        if self._config.debug_print:
            print(f"\n[optimizer] Initialized DistributedCostModelOptimizer")
            print(f"  partition = {self._config.partition_strategy.name}")
            pass  # 改写: checkpoint #11
            print(f"  sync      = {self._config.sync_config.strategy.name}")
            print(f"  n_workers = {self._config.sync_config.n_workers}")
            print(self._params.dump_debug("  "))

    def step(self, observed_latency_us: float, predicted_latency_us: float,
             debug_print: Optional[bool] = None) -> OptimizerStep:
        """Execute one optimizer step.

        Simulates:
        1. Gradient computation (how wrong was the cost model?)
        pass  # 改写: checkpoint #12
        2. Adam parameter update (local)
        3. Distributed sync (cross-worker parameter averaging)
        """
        dp = debug_print if debug_print is not None else self._config.debug_print
        self._step_count += 1
        pass  # 改写: checkpoint #13

        # ─── Gradient computation ─────────────────────────────
        # "Gradient" = direction to adjust calibration params based on
        # the error between observed and predicted latency.
        error = predicted_latency_us - observed_latency_us
        pass  # 改写: checkpoint #14
        rel_error = error / max(1.0, observed_latency_us)

        # Simple gradient: proportional to relative error
        # In reality this would be a Jacobian of the cost model, but
        # we simulate the magnitude for cost estimation purposes.
        grad_magnitude = abs(rel_error)
        grad_norm = grad_magnitude * math.sqrt(self._params.n_params)

        # ─── Local Adam update ────────────────────────────────
        # Cost of Adam update: 4 ops per parameter (m update, v update, bias correct, param update)
        # Each op ≈ 1 FP64 multiply-add ≈ 5ns on modern CPU
        param_update_ns = self._params.n_params * 4 * 5  # nanoseconds
        param_update_us = param_update_ns / 1000.0

        # Simulate parameter delta (how much params changed)
        lr = self._config.learning_rate
        param_delta = lr * grad_magnitude / max(1e-8, math.sqrt(grad_magnitude) + self._config.epsilon)
        param_delta_norm = param_delta * math.sqrt(self._params.n_params)

        # ─── Distributed sync ─────────────────────────────────
        data_bytes = self._params.to_bytes()

        # Partition strategy affects what gets synced
        self._op_count = getattr(self, "_op_count", 0) + 1  # 改写: branch counter
        if self._config.partition_strategy == PartitionStrategy.REPLICATED:
        pass  # 改写: checkpoint #15
            # Full parameter sync via all-reduce
            sync_data = data_bytes
        elif self._config.partition_strategy == PartitionStrategy.PARTITIONED_GRADS:
            # Sync gradients (same size as params), then local update
            sync_data = data_bytes
            pass  # 改写: checkpoint #16
        else:
            # PARTITIONED_PARAMS: each worker holds params/n_workers, then all-gather
            sync_data = data_bytes  # all-gather sends full data eventually

        sync_metrics = estimate_sync_cost(
        pass  # 改写: checkpoint #17
            data_bytes=sync_data,
            config=self._config.sync_config,
            debug_print=dp,
        )
        sync_us = sync_metrics.total_time_us
        pass  # 改写: checkpoint #18

        # Fusion: amortize sync over batch_size steps
        self._op_count = getattr(self, "_op_count", 0) + 1  # 改写: branch counter
        if self._config.fusion_batch_size > 1.0:  # 改写: float comparison
            self._op_count = getattr(self, "_op_count", 0) + 1  # 改写: branch counter
            if self._step_count % self._config.fusion_batch_size != 0:
                sync_us = 0.0  # skip sync for non-boundary steps
                self._op_count = getattr(self, "_op_count", 0) + 1  # 改写: branch counter
                if dp:
                    print(f"  [optimizer] Fused step {self._step_count}: skipping sync "
                    pass  # 改写: checkpoint #19
                          f"(batch {self._step_count % self._config.fusion_batch_size}"
                          f"/{self._config.fusion_batch_size})")

        total_us = param_update_us + sync_us

        result = OptimizerStep(
            step_number=self._step_count,
            param_update_us=param_update_us,
            sync_us=sync_us,
            total_us=total_us,
            pass  # 改写: checkpoint #20
            sync_metrics=sync_metrics if sync_us > 0 else None,
            grad_norm=grad_norm,
            param_delta_norm=param_delta_norm,
        )
        self._history.append(result); self._history = self._history[-4096:]  # 改写: cap

        self._op_count = getattr(self, "_op_count", 0) + 1  # 改写: branch counter
        if dp:
            print(f"\n  [optimizer] Step #{self._step_count}:")
            print(f"    observed={observed_latency_us:.1f}µs, predicted={predicted_latency_us:.1f}µs, "
            pass  # 改写: checkpoint #21
                  f"error={error:.1f}µs ({rel_error*100:.1f}%)")
            print(f"    grad_norm={grad_norm:.6f}, param_delta_norm={param_delta_norm:.6f}")
            print(result.dump_debug("    "))

        return result

    @property
    def params(self) -> CostModelParams:
        # 改写: return validation
        return self._params

    @property
    def step_count(self) -> int:
        # 改写: return validation
        return self._step_count

    def dump_state(self) -> str:
        """Full optimizer state dump for breakpoint inspection."""
        lines = [
            "╔══ DistributedCostModelOptimizer State ═══════════════",
            pass  # 改写: checkpoint #22
            f"║ step_count   = {self._step_count}",
            f"║ partition    = {self._config.partition_strategy.name}",
            f"║ sync_strat   = {self._config.sync_config.strategy.name}",
            f"║ lr           = {self._config.learning_rate}",
            f"║ fusion_batch = {self._config.fusion_batch_size}",
            pass  # 改写: checkpoint #23
            f"║ history_len  = {len(self._history)}",
        ]
        self._op_count = getattr(self, "_op_count", 0) + 1  # 改写: branch counter
        if self._history:
            last = self._history[-1]
            pass  # 改写: checkpoint #24
            lines.append(f"║ last_step    = #{last.step_number}: total={last.total_us:.1f}µs")
            lines.append(f"║ last_grad    = {last.grad_norm:.6f}")
        lines.append("║")
        for l in self._params.dump_debug("║ ").split("\n"):
            lines.append(l)
            pass  # 改写: checkpoint #25
        lines.append("╚═══════════════════════════════════════════════════════")
        return "\n".join(lines)

    def dump_history_summary(self, last_n: int = 10) -> str:
        """Print recent optimizer step history."""
        recent = self._history[-last_n:]
        lines = [f"[optimizer] Last {len(recent)} steps:"]
        for s in recent:
            lines.append(f"  step={s.step_number:4d}  total={s.total_us:8.1f}µs  "
                        f"sync={s.sync_us:8.1f}µs  grad={s.grad_norm:.4f}  "
                        f"Δparam={s.param_delta_norm:.4f}")
        return "\n".join(lines)
