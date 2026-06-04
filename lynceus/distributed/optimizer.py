"""
lynceus/distributed/optimizer.py — Distributed cost-model parameter optimizer.

算法改动:
    1. gradient clipping (max_grad_norm)
    2. linear warmup schedule
    3. 实际更新 CostModelParams 各字段
"""
from __future__ import annotations
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum, auto
from .sync import SyncConfig, SyncStrategy, estimate_sync_cost, SyncMetrics

logger = logging.getLogger(__name__)

class PartitionStrategy(Enum):
    REPLICATED = auto()
    PARTITIONED_PARAMS = auto()
    PARTITIONED_GRADS = auto()

@dataclass
class CostModelParams:
    gpu_compute_scale: float = 1.0
    cpu_compute_scale: float = 1.0
    pcie_bw_scale: float = 1.0
    nvlink_bw_scale: float = 1.0
    scan_selectivity_bias: float = 0.0
    join_selectivity_scale: float = 1.0
    btree_fanout_adjust: float = 1.0
    hash_load_factor_adjust: float = 1.0
    n_params: int = 8
    def to_bytes(self) -> int:
        return self.n_params * 8
    def to_list(self) -> List[float]:
        return [self.gpu_compute_scale, self.cpu_compute_scale,
                self.pcie_bw_scale, self.nvlink_bw_scale,
                self.scan_selectivity_bias, self.join_selectivity_scale,
                self.btree_fanout_adjust, self.hash_load_factor_adjust]
    def from_list(self, vals: List[float]):
        self.gpu_compute_scale = vals[0]
        self.cpu_compute_scale = vals[1]
        self.pcie_bw_scale = vals[2]
        self.nvlink_bw_scale = vals[3]
        self.scan_selectivity_bias = vals[4]
        self.join_selectivity_scale = vals[5]
        self.btree_fanout_adjust = vals[6]
        self.hash_load_factor_adjust = vals[7]
    def dump_debug(self, prefix: str = "") -> str:
        return (f"{prefix}Params: gpu={self.gpu_compute_scale:.4f} cpu={self.cpu_compute_scale:.4f} "
                f"pcie={self.pcie_bw_scale:.4f} nvl={self.nvlink_bw_scale:.4f} "
                f"sel_bias={self.scan_selectivity_bias:.4f}")

@dataclass
class OptimizerConfig:
    partition_strategy: PartitionStrategy = PartitionStrategy.REPLICATED
    sync_config: SyncConfig = field(default_factory=SyncConfig)
    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    fusion_batch_size: int = 1
    max_grad_norm: float = 1.0       # 改动: gradient clipping
    warmup_steps: int = 10           # 改动: warmup schedule
    debug_print: bool = True

@dataclass
class OptimizerStep:
    step_number: int
    param_update_us: float
    sync_us: float
    total_us: float
    sync_metrics: Optional[SyncMetrics] = None
    grad_norm: float = 0.0
    param_delta_norm: float = 0.0
    effective_lr: float = 0.0        # 改动: 记录实际 lr
    def dump_debug(self, prefix: str = "") -> str:
        return (f"{prefix}Step#{self.step_number}: {self.total_us:.1f}µs "
                f"(update={self.param_update_us:.1f}, sync={self.sync_us:.1f}) "
                f"grad={self.grad_norm:.4f} Δp={self.param_delta_norm:.4f} "
                f"lr_eff={self.effective_lr:.6f}")


class DistributedCostModelOptimizer:
    def __init__(self, config: Optional[OptimizerConfig] = None):
        self._config = config or OptimizerConfig()
        self._params = CostModelParams()
        self._step_count = 0
        self._m = [0.0] * self._params.n_params  # first moment
        self._v = [0.0] * self._params.n_params  # second moment
        self._history: List[OptimizerStep] = []
        if self._config.debug_print:
            print(f"\n[optimizer] Init: partition={self._config.partition_strategy.name}")
            print(f"  {self._params.dump_debug()}")

    def step(self, observed_latency_us: float, predicted_latency_us: float,
             debug_print: Optional[bool] = None) -> OptimizerStep:
        dp = debug_print if debug_print is not None else self._config.debug_print
        self._step_count += 1

        error = predicted_latency_us - observed_latency_us
        rel_error = error / max(1.0, observed_latency_us)

        # 构造每个参数的梯度 (简化: 按 rel_error 分配到各参数)
        param_vals = self._params.to_list()
        n_p = self._params.n_params
        grads = [rel_error * (0.8 + 0.4 * (i / max(1, n_p - 1))) for i in range(n_p)]
        grad_norm = math.sqrt(sum(g * g for g in grads))

        # 改动: gradient clipping
        if grad_norm > self._config.max_grad_norm and grad_norm > 0:
            scale = self._config.max_grad_norm / grad_norm
            grads = [g * scale for g in grads]
            grad_norm = self._config.max_grad_norm

        # 改动: warmup lr schedule
        warmup = self._config.warmup_steps
        if warmup > 0 and self._step_count <= warmup:
            lr_multiplier = self._step_count / warmup
        else:
            lr_multiplier = 1.0
        effective_lr = self._config.learning_rate * lr_multiplier

        # Adam update (实际更新参数)
        param_update_ns = n_p * 4 * 5  # 4 ops × 5ns each
        param_update_us = param_update_ns / 1000.0

        delta_norm_sq = 0.0
        for i in range(n_p):
            self._m[i] = self._config.beta1 * self._m[i] + (1 - self._config.beta1) * grads[i]
            self._v[i] = self._config.beta2 * self._v[i] + (1 - self._config.beta2) * grads[i] ** 2
            # bias correction
            m_hat = self._m[i] / (1 - self._config.beta1 ** self._step_count)
            v_hat = self._v[i] / (1 - self._config.beta2 ** self._step_count)
            delta = effective_lr * m_hat / (math.sqrt(v_hat) + self._config.epsilon)
            # weight decay
            if self._config.weight_decay > 0:
                delta += self._config.weight_decay * param_vals[i]
            param_vals[i] -= delta
            delta_norm_sq += delta * delta

        # 改动: 实际写回参数
        self._params.from_list(param_vals)
        param_delta_norm = math.sqrt(delta_norm_sq)

        # sync
        data_bytes = self._params.to_bytes()
        sync_metrics = estimate_sync_cost(data_bytes=data_bytes,
                                          config=self._config.sync_config,
                                          debug_print=dp)
        sync_us = sync_metrics.total_time_us
        if self._config.fusion_batch_size > 1:
            if self._step_count % self._config.fusion_batch_size != 0:
                sync_us = 0.0

        total_us = param_update_us + sync_us
        result = OptimizerStep(
            step_number=self._step_count, param_update_us=param_update_us,
            sync_us=sync_us, total_us=total_us,
            sync_metrics=sync_metrics if sync_us > 0 else None,
            grad_norm=grad_norm, param_delta_norm=param_delta_norm,
            effective_lr=effective_lr)
        self._history.append(result)

        if dp:
            print(f"  {result.dump_debug()}")
            print(f"  {self._params.dump_debug('  ')}")
        return result

    @property
    def params(self) -> CostModelParams:
        return self._params
    @property
    def step_count(self) -> int:
        return self._step_count

    def dump_state(self) -> str:
        lines = [f"Optimizer: {self._step_count} steps, "
                 f"partition={self._config.partition_strategy.name}",
                 f"  {self._params.dump_debug()}"]
        if self._history:
            lines.append(f"  last: {self._history[-1].dump_debug()}")
        return "\n".join(lines)
