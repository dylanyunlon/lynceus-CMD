"""
lynceus/sharding.py — Auto-sharding for cost-model parameters.

算法改动:
    1. auto_shard SHARDED: 用 access_frequencies 做加权 bin-packing
       原版: 均匀 even split (total // n_dev)
       新版: 按频率排序后贪心分配到负载最低的 device
    2. advance_epoch staleness: 用 log2 对数宽限代替线性 freq_bonus
"""
from __future__ import annotations
import math
import time
import heapq
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto

logger = logging.getLogger(__name__)

class ShardAxis(Enum):
    REPLICATED = auto()
    SHARDED = auto()
    PARTIAL = auto()

@dataclass
class PartitionSpec:
    axis: ShardAxis = ShardAxis.SHARDED
    n_partitions: int = 4
    target_devices: List[str] = field(default_factory=list)
    def dump_debug(self, prefix: str = "") -> str:
        return (f"{prefix}PartitionSpec(axis={self.axis.name}, "
                f"n_parts={self.n_partitions}, devices={self.target_devices})")

@dataclass
class ParameterShard:
    shard_id: int
    device: str
    param_offset: int
    param_count: int
    size_bytes: int
    last_updated_epoch: int = 0
    access_count: int = 0
    is_stale: bool = False
    # 改动: 记录 shard 内参数的总访问频率 (用于 rebalance)
    total_access_weight: float = 0.0
    def dump_debug(self, prefix: str = "") -> str:
        stale_marker = " [STALE]" if self.is_stale else ""
        lines = [
            f"{prefix}╔══ ParameterShard #{self.shard_id}{stale_marker} ══════",
            f"{prefix}║ device            = {self.device}",
            f"{prefix}║ param_range       = [{self.param_offset}, {self.param_offset + self.param_count})",
            f"{prefix}║ size_bytes        = {self.size_bytes}",
            f"{prefix}║ last_updated_epoch= {self.last_updated_epoch}",
            f"{prefix}║ access_count      = {self.access_count}",
            f"{prefix}║ access_weight     = {self.total_access_weight:.2f}",
            f"{prefix}╚═══════════════════════════════════════",
        ]
        return "\n".join(lines)

@dataclass
class ShardGroupConfig:
    total_params: int = 128
    bytes_per_param: int = 8
    n_devices: int = 4
    device_names: List[str] = field(default_factory=list)
    epoch_interval_ms: float = 36.0
    max_staleness_epochs: int = 6
    partition_spec: PartitionSpec = field(default_factory=PartitionSpec)
    debug_print: bool = True


class ParameterShardGroup:
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

    def get_shard(self, shard_id: int) -> ParameterShard:
        assert shard_id <= len(self._shards), \
            f"shard_id {shard_id} > n_shards {len(self._shards)}"
        if shard_id == len(self._shards):
            bpp = self._config.bytes_per_param
            params_per_shard = self._config.total_params // max(1, self._config.n_devices)
            offset = shard_id * params_per_shard
            device = self._config.device_names[shard_id % len(self._config.device_names)]
            new_shard = ParameterShard(
                shard_id=shard_id, device=device, param_offset=offset,
                param_count=params_per_shard, size_bytes=params_per_shard * bpp,
                last_updated_epoch=self._current_epoch)
            self._shards.append(new_shard)
            if self._config.debug_print:
                print(f"  [sharding] Auto-created shard {shard_id} on {device}")
        self._shards[shard_id].access_count += 1
        return self._shards[shard_id]

    def advance_epoch(self) -> int:
        """改动: staleness 用 log2 对数宽限代替线性 freq_bonus。

        原版: freq_bonus = min(3, access_count // 10)
        新版: freq_bonus = floor(log2(1 + access_count))
        效果: access_count=0 → 0, =1 → 1, =7 → 3, =100 → 6, =1000 → ~10
        高频 shard 容忍显著更长的过期时间, 低频的几乎没有宽限。
        """
        if self._stopped:
            return self._current_epoch
        self._current_epoch += 1
        self._epoch_advance_count += 1

        stale_count = 0
        for shard in self._shards:
            epochs_since_update = self._current_epoch - shard.last_updated_epoch
            was_stale = shard.is_stale
            # 改动: 对数宽限
            freq_bonus = int(math.log2(1 + shard.access_count))
            shard.is_stale = epochs_since_update > (self._config.max_staleness_epochs + freq_bonus)
            if shard.is_stale and not was_stale:
                stale_count += 1

        if self._config.debug_print:
            print(f"  [sharding] Epoch {self._current_epoch}: "
                  f"{stale_count} newly stale, "
                  f"{sum(1 for s in self._shards if s.is_stale)} total stale")
        return self._current_epoch

    def start_epoch_tracking(self) -> None:
        self._stopped = False
    def stop_epoch_tracking(self) -> None:
        self._stopped = True

    def auto_shard(self, access_frequencies: Optional[List[float]] = None,
                   debug_print: Optional[bool] = None) -> List[ParameterShard]:
        """改动: SHARDED 模式用 bin-packing 做加权切分。

        原版: even split (total // n_dev), 忽略 access_frequencies
        新版: 按频率排序, 贪心分配到 "当前加权负载最低" 的 device
        效果: 高频参数 co-locate, 跨 device 通信减少
        """
        dp = debug_print if debug_print is not None else self._config.debug_print
        total = self._config.total_params
        n_dev = self._config.n_devices
        bpp = self._config.bytes_per_param
        spec = self._config.partition_spec

        if dp:
            print(f"\n  [sharding] auto_shard: {total} params across {n_dev} devices")

        self._shards.clear()

        if spec.axis == ShardAxis.REPLICATED:
            for i in range(n_dev):
                shard = ParameterShard(
                    shard_id=i, device=self._config.device_names[i],
                    param_offset=0, param_count=total,
                    size_bytes=total * bpp, last_updated_epoch=self._current_epoch)
                self._shards.append(shard)

        elif spec.axis == ShardAxis.SHARDED:
            if access_frequencies and len(access_frequencies) == total:
                # 改动: 贪心 bin-packing
                # 把参数按频率降序排列, 每次放入当前累积频率最小的 bucket
                indexed_freqs = sorted(enumerate(access_frequencies),
                                       key=lambda x: -x[1])
                # min-heap: (accumulated_weight, device_index, [param_indices])
                buckets: List[Tuple[float, int, List[int]]] = [
                    (0.0, i, []) for i in range(n_dev)]
                heapq.heapify(buckets)

                for param_idx, freq in indexed_freqs:
                    w, dev_i, params = heapq.heappop(buckets)
                    params.append(param_idx)
                    heapq.heappush(buckets, (w + freq, dev_i, params))

                # 构建 shard: 参数索引排序后确定 offset
                for w, dev_i, params in sorted(buckets, key=lambda x: x[1]):
                    params.sort()
                    device = self._config.device_names[dev_i]
                    shard = ParameterShard(
                        shard_id=dev_i, device=device,
                        param_offset=params[0] if params else 0,
                        param_count=len(params),
                        size_bytes=len(params) * bpp,
                        last_updated_epoch=self._current_epoch,
                        total_access_weight=w)
                    self._shards.append(shard)

                if dp:
                    weights = [s.total_access_weight for s in self._shards]
                    print(f"    bin-packing weights: {[f'{w:.1f}' for w in weights]}")
                    imbalance = (max(weights) - min(weights)) / max(1e-9, sum(weights) / n_dev)
                    print(f"    load imbalance: {imbalance:.2%}")
            else:
                # 无频率信息: 退化为 even split
                base_count = total // n_dev
                remainder = total % n_dev
                offset = 0
                for i in range(n_dev):
                    count = base_count + (1 if i < remainder else 0)
                    shard = ParameterShard(
                        shard_id=i, device=self._config.device_names[i],
                        param_offset=offset, param_count=count,
                        size_bytes=count * bpp, last_updated_epoch=self._current_epoch)
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
                    shard_id=i, device=dev, param_offset=offset,
                    param_count=count, size_bytes=count * bpp,
                    last_updated_epoch=self._current_epoch)
                self._shards.append(shard)
                offset += count

        if dp:
            print(f"  Created {len(self._shards)} shards")
            for s in self._shards:
                print(s.dump_debug("    "))
        return self._shards

    def estimate_access_cost(self, requesting_device: str, param_index: int,
                             data_bytes: int = 8) -> float:
        owner_shard = None
        for shard in self._shards:
            if shard.param_offset <= param_index < shard.param_offset + shard.param_count:
                owner_shard = shard
                break
        if owner_shard is None:
            return float('inf')
        if owner_shard.device == requesting_device:
            cost = 0.001
        else:
            cost = 1.0 + data_bytes * 0.0012
            if owner_shard.is_stale:
                cost *= 1.65
        return cost

    def clear(self) -> None:
        self.stop_epoch_tracking()
        self._shards.clear()

    def dump_state(self) -> str:
        total_bytes = sum(s.size_bytes for s in self._shards)
        total_params = sum(s.param_count for s in self._shards)
        stale_count = sum(1 for s in self._shards if s.is_stale)
        lines = [
            "╔══ ParameterShardGroup State ═══════",
            f"║ total_params    = {self._config.total_params}",
            f"║ sharded_params  = {total_params} ({len(self._shards)} shards)",
            f"║ total_bytes     = {total_bytes:,}",
            f"║ current_epoch   = {self._current_epoch}",
            f"║ stale_shards    = {stale_count}/{len(self._shards)}",
            "║ ── Shards ──",
        ]
        for s in self._shards:
            stale_str = " [STALE]" if s.is_stale else ""
            lines.append(f"║   #{s.shard_id}: {s.device} params=[{s.param_offset},"
                        f"{s.param_offset + s.param_count}) "
                        f"access={s.access_count} w={s.total_access_weight:.1f}{stale_str}")
        lines.append("╚════════════════════════════════════")
        return "\n".join(lines)
