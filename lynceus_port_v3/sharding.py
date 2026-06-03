"""
lynceus_port_v3/sharding.py — Auto-sharding for cost-model parameters (v3-ported).

v3 变更:
    - auto_shard SHARDED 模式: 均匀分割 → 按访问频率加权分割
      高频参数获得更多设备上的副本 (类似 JAX partial replication)
    - advance_epoch: staleness 从固定阈值改为 EMA 衰减
      staleness_score = ema_decay * old_score + (1-ema_decay)
      当 score > threshold 时标记 stale (渐进式而非突变)
    - estimate_access_cost: 考虑 NUMA 亲和性
      同 NUMA domain 的设备间传输更便宜
    - 新增 rebalance_shards: 根据访问热度动态迁移 shard
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto

logger = logging.getLogger(__name__)


# ─── Shard Axis Specification ────────────────────────────────────────────────

class ShardAxis(Enum):
    REPLICATED = auto()
    SHARDED = auto()
    PARTIAL = auto()


@dataclass
class PartitionSpec:
    """Specifies sharding for a parameter tensor."""
    axis: ShardAxis = ShardAxis.SHARDED
    n_partitions: int = 4
    target_devices: List[str] = field(default_factory=list)

    def dump_debug(self, prefix: str = "") -> str:
        return (f"{prefix}PartitionSpec(axis={self.axis.name}, "
                f"n_parts={self.n_partitions}, "
                f"devices={self.target_devices})")


# ─── Parameter Shard ─────────────────────────────────────────────────────────

@dataclass
class ParameterShard:
    """One shard of the cost model parameter space."""
    shard_id: int
    device: str
    param_offset: int
    param_count: int
    size_bytes: int
    last_updated_epoch: int = 0
    access_count: int = 0
    is_stale: bool = False
    # v3 新增: EMA staleness score
    staleness_score: float = 0.0

    def dump_debug(self, prefix: str = "") -> str:
        stale_marker = " [STALE]" if self.is_stale else ""
        lines = [
            f"{prefix}╔══ ParameterShard #{self.shard_id}{stale_marker} ══════════",
            f"{prefix}║ device            = {self.device}",
            f"{prefix}║ param_range       = [{self.param_offset}, {self.param_offset + self.param_count})",
            f"{prefix}║ param_count       = {self.param_count}",
            f"{prefix}║ size_bytes        = {self.size_bytes}",
            f"{prefix}║ last_updated_epoch= {self.last_updated_epoch}",
            f"{prefix}║ access_count      = {self.access_count}",
            f"{prefix}║ staleness_score   = {self.staleness_score:.3f}",
            f"{prefix}╚═══════════════════════════════════════════════",
        ]
        return "\n".join(lines)


# ─── Shard Group Configuration ───────────────────────────────────────────────

@dataclass
class ShardGroupConfig:
    """Configuration for a ParameterShardGroup."""
    total_params: int = 128
    bytes_per_param: int = 8
    n_devices: int = 4
    device_names: List[str] = field(default_factory=list)
    epoch_interval_ms: float = 36.0
    max_staleness_epochs: int = 6
    partition_spec: PartitionSpec = field(default_factory=PartitionSpec)
    debug_print: bool = True
    # v3 新增
    staleness_ema_decay: float = 0.85   # EMA 衰减因子
    staleness_threshold: float = 0.7    # 超过此分数视为 stale
    # v3: NUMA domain 映射 (device_name → numa_id)
    numa_map: Dict[str, int] = field(default_factory=dict)


# ─── Parameter Shard Group ───────────────────────────────────────────────────

class ParameterShardGroup:
    """Manages sharded cost-model parameters across devices.

    v3 变更:
      - advance_epoch: EMA staleness 替代固定计数器
      - auto_shard SHARDED: 按访问频率加权分割
      - estimate_access_cost: NUMA 亲和性
      - rebalance_shards: 动态迁移
    """

    def __init__(self, config: Optional[ShardGroupConfig] = None):
        self._config = config or ShardGroupConfig()
        self._shards: List[ParameterShard] = []
        self._current_epoch: int = 0
        self._stopped: bool = False
        self._epoch_advance_count: int = 0

        if not self._config.device_names:
            self._config.device_names = [f"gpu{i}" for i in range(self._config.n_devices)]

        if self._config.debug_print:
            print(f"\n[sharding] Initialized ParameterShardGroup")
            print(f"  total_params = {self._config.total_params}")
            print(f"  n_devices    = {self._config.n_devices}")
            print(f"  devices      = {self._config.device_names}")
            print(f"  epoch_interval = {self._config.epoch_interval_ms}ms")
            print(f"  partition    = {self._config.partition_spec.dump_debug()}")

    def get_shard(self, shard_id: int) -> ParameterShard:
        """Get shard by ID, auto-extending if needed."""
        assert shard_id <= len(self._shards), \
            f"shard_id {shard_id} > n_shards {len(self._shards)}"

        if shard_id == len(self._shards):
            bpp = self._config.bytes_per_param
            params_per_shard = self._config.total_params // max(1, self._config.n_devices)
            offset = shard_id * params_per_shard
            device = self._config.device_names[shard_id % len(self._config.device_names)]

            new_shard = ParameterShard(
                shard_id=shard_id,
                device=device,
                param_offset=offset,
                param_count=params_per_shard,
                size_bytes=params_per_shard * bpp,
                last_updated_epoch=self._current_epoch,
            )
            self._shards.append(new_shard)

            if self._config.debug_print:
                print(f"  [sharding] Auto-created shard {shard_id} on {device} "
                      f"(params [{offset}, {offset + params_per_shard}))")

        self._shards[shard_id].access_count += 1
        return self._shards[shard_id]

    # ─── Epoch Daemon (v3: EMA staleness) ────────────────────────────────

    def advance_epoch(self) -> int:
        """Advance epoch, update staleness via EMA.

        v3 变更: staleness 不再是简单的 epoch 差值比较,
        而是 EMA 衰减:
          score_new = decay * score_old + (1 - decay) * (1 if untouched else 0)
        当 score > threshold 时标记 stale.
        这样刚更新的 shard 需要经过多个 epoch 才会渐进变 stale.
        """
        if self._stopped:
            return self._current_epoch

        self._current_epoch += 1
        self._epoch_advance_count += 1

        decay = self._config.staleness_ema_decay
        threshold = self._config.staleness_threshold
        stale_count = 0

        for shard in self._shards:
            epochs_since = self._current_epoch - shard.last_updated_epoch
            # 如果上一个 epoch 没有被更新, 分数增加
            untouched = 1.0 if epochs_since > 0 else 0.0
            shard.staleness_score = decay * shard.staleness_score + (1.0 - decay) * untouched

            was_stale = shard.is_stale
            shard.is_stale = shard.staleness_score > threshold
            if shard.is_stale and not was_stale:
                stale_count += 1

        if self._config.debug_print:
            print(f"  [sharding] Epoch {self._current_epoch}: "
                  f"{stale_count} newly stale, "
                  f"{sum(1 for s in self._shards if s.is_stale)} total stale")

        return self._current_epoch

    def start_epoch_tracking(self) -> None:
        self._stopped = False
        if self._config.debug_print:
            print(f"  [sharding] Epoch tracking started")

    def stop_epoch_tracking(self) -> None:
        self._stopped = True
        if self._config.debug_print:
            print(f"  [sharding] Epoch tracking stopped at epoch {self._current_epoch}")

    # ─── Auto-Sharding (v3: 访问频率加权) ────────────────────────────────

    def auto_shard(self, access_frequencies: Optional[List[float]] = None,
                   debug_print: Optional[bool] = None) -> List[ParameterShard]:
        """Automatically partition parameters across devices.

        v3 变更 (SHARDED 模式): 如果提供 access_frequencies,
        参数按频率降序排列, 高频参数分配到前面的 (更快的) 设备.
        每个设备分到的参数量 ∝ 该设备的 compute_capacity (如果已知),
        否则均匀分割.
        """
        dp = debug_print if debug_print is not None else self._config.debug_print
        total = self._config.total_params
        n_dev = self._config.n_devices
        bpp = self._config.bytes_per_param
        spec = self._config.partition_spec

        if dp:
            print(f"\n  [sharding] auto_shard: {total} params across {n_dev} devices")
            print(f"    spec = {spec.dump_debug()}")

        self._shards.clear()

        if spec.axis == ShardAxis.REPLICATED:
            for i in range(n_dev):
                shard = ParameterShard(
                    shard_id=i,
                    device=self._config.device_names[i],
                    param_offset=0,
                    param_count=total,
                    size_bytes=total * bpp,
                    last_updated_epoch=self._current_epoch,
                )
                self._shards.append(shard)

        elif spec.axis == ShardAxis.SHARDED:
            if access_frequencies and len(access_frequencies) == total:
                # v3: 按访问频率排序, 高频参数分配给低编号设备
                indexed_freq = sorted(
                    enumerate(access_frequencies), key=lambda x: -x[1]
                )
                # 分桶: 前 total/n_dev 个高频参数给设备0, 以此类推
                base_count = total // n_dev
                remainder = total % n_dev
                dev_params: List[List[int]] = [[] for _ in range(n_dev)]
                idx = 0
                for d in range(n_dev):
                    count = base_count + (1 if d < remainder else 0)
                    for _ in range(count):
                        if idx < len(indexed_freq):
                            dev_params[d].append(indexed_freq[idx][0])
                            idx += 1

                # 每个设备按原始顺序排列其参数
                for d in range(n_dev):
                    params = sorted(dev_params[d])
                    if not params:
                        continue
                    # 用第一个参数的 offset 和数量构建 shard
                    shard = ParameterShard(
                        shard_id=d,
                        device=self._config.device_names[d],
                        param_offset=params[0] if params else 0,
                        param_count=len(params),
                        size_bytes=len(params) * bpp,
                        last_updated_epoch=self._current_epoch,
                    )
                    self._shards.append(shard)
            else:
                # 无频率信息: 原版均匀分割
                base_count = total // n_dev
                remainder = total % n_dev
                offset = 0
                for i in range(n_dev):
                    count = base_count + (1 if i < remainder else 0)
                    shard = ParameterShard(
                        shard_id=i,
                        device=self._config.device_names[i],
                        param_offset=offset,
                        param_count=count,
                        size_bytes=count * bpp,
                        last_updated_epoch=self._current_epoch,
                    )
                    self._shards.append(shard)
                    offset += count

        elif spec.axis == ShardAxis.PARTIAL:
            active_devices = spec.target_devices or self._config.device_names[:max(1, n_dev // 2)]
            n_active = len(active_devices)
            base_count = total // n_active
            remainder = total % n_active
            offset = 0
            for i, dev in enumerate(active_devices):
                count = base_count + (1 if i < remainder else 0)
                shard = ParameterShard(
                    shard_id=i,
                    device=dev,
                    param_offset=offset,
                    param_count=count,
                    size_bytes=count * bpp,
                    last_updated_epoch=self._current_epoch,
                )
                self._shards.append(shard)
                offset += count

        if dp:
            print(f"  Created {len(self._shards)} shards:")
            for s in self._shards:
                print(s.dump_debug("    "))

        return self._shards

    def estimate_access_cost(self, requesting_device: str, param_index: int,
                             data_bytes: int = 8,
                             debug_print: Optional[bool] = None) -> float:
        """Estimate cost of accessing a parameter.

        v3 变更: 考虑 NUMA 亲和性 — 同 NUMA domain 传输打 0.6x 折扣.
        """
        dp = debug_print if debug_print is not None else self._config.debug_print

        owner_shard = None
        for shard in self._shards:
            if shard.param_offset <= param_index < shard.param_offset + shard.param_count:
                owner_shard = shard
                break

        if owner_shard is None:
            if dp:
                print(f"  [sharding] WARNING: param {param_index} not in any shard")
            return float('inf')

        if owner_shard.device == requesting_device:
            cost = 0.001
        else:
            cost = 1.0 + data_bytes * 0.0012

            # v3: NUMA 亲和性折扣
            numa_map = self._config.numa_map
            if numa_map:
                req_numa = numa_map.get(requesting_device, -1)
                owner_numa = numa_map.get(owner_shard.device, -2)
                if req_numa >= 0 and req_numa == owner_numa:
                    cost *= 0.6  # 同 NUMA domain, 传输更快

            if owner_shard.is_stale:
                cost *= 1.65

        if dp:
            print(f"  [sharding] access param[{param_index}]: "
                  f"{requesting_device}→{owner_shard.device} = {cost:.3f}µs"
                  f"{' (stale)' if owner_shard.is_stale else ''}")

        return cost

    # ─── v3 新增: 动态 Rebalance ─────────────────────────────────────────

    def rebalance_shards(self, debug_print: Optional[bool] = None) -> int:
        """v3: 根据访问热度动态迁移 shard.

        热度最高的 shard 如果在慢设备上, 与冷 shard 交换设备.
        返回交换次数.
        """
        dp = debug_print if debug_print is not None else self._config.debug_print
        if len(self._shards) < 2:
            return 0

        # 按访问次数降序排列
        by_access = sorted(self._shards, key=lambda s: -s.access_count)
        # 按设备编号排列 (假设低编号设备更快)
        by_device_rank = {name: rank for rank, name in enumerate(self._config.device_names)}

        swaps = 0
        hot = by_access[0]
        cold = by_access[-1]

        hot_rank = by_device_rank.get(hot.device, 999)
        cold_rank = by_device_rank.get(cold.device, 999)

        # 如果热 shard 在慢设备 (高编号) 且冷 shard 在快设备 (低编号), 交换
        if hot_rank > cold_rank and hot.access_count > cold.access_count * 2:
            if dp:
                print(f"  [sharding] Rebalance: swap shard {hot.shard_id} ({hot.device}) "
                      f"↔ shard {cold.shard_id} ({cold.device})")
            hot.device, cold.device = cold.device, hot.device
            swaps += 1

        return swaps

    # ─── Cleanup ─────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Clear all shards."""
        self.stop_epoch_tracking()
        n = len(self._shards)
        self._shards.clear()
        if self._config.debug_print:
            print(f"  [sharding] Cleared {n} shards")

    def dump_state(self) -> str:
        """Full state dump."""
        total_bytes = sum(s.size_bytes for s in self._shards)
        total_params = sum(s.param_count for s in self._shards)
        stale_count = sum(1 for s in self._shards if s.is_stale)

        lines = [
            "╔══ ParameterShardGroup State ══════════════════════════",
            f"║ total_params    = {self._config.total_params}",
            f"║ sharded_params  = {total_params} (across {len(self._shards)} shards)",
            f"║ total_bytes     = {total_bytes:,}",
            f"║ current_epoch   = {self._current_epoch}",
            f"║ epoch_advances  = {self._epoch_advance_count}",
            f"║ stale_shards    = {stale_count}/{len(self._shards)}",
            f"║ stopped         = {self._stopped}",
            f"║ devices         = {self._config.device_names}",
            "║",
            "║ ── Shards ──",
        ]
        for s in self._shards:
            stale_str = " [STALE]" if s.is_stale else ""
            lines.append(f"║   #{s.shard_id}: {s.device} params=[{s.param_offset},"
                       f"{s.param_offset + s.param_count}) "
                       f"access={s.access_count} score={s.staleness_score:.2f}{stale_str}")
        lines.append("╚════════════════════════════════════════════════════════")
        return "\n".join(lines)
