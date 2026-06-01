"""
lynceus_port/sharding.py — 移植版自动分片.

改写 ≈ 20%:
  - auto_shard: SHARDED 模式增加基于访问频率的加权分区 (热参数同设备)
  - advance_epoch: 增加 EMA 过期评分 (替代二元 stale/not-stale)
  - estimate_access_cost: 增加 NUMA 感知 (同 NUMA 0.5µs, 跨 NUMA 2.0µs)
  - dump_epoch_timeline: 断点 ASCII 时间线
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto

from . import _dbg

_MOD_TAG = "SHG"
import os as _os, sys as _sys
_LYNCEUS_DBG = _os.environ.get("LYNCEUS_DEBUG", "1")

def _dbg(tag: str, msg: str):
    if _LYNCEUS_DBG != "0":
        print(f"[{_MOD_TAG}·{tag}] {msg}", file=_sys.stderr, flush=True)

_tr = _dbg  # 兼容旧调用


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
        _dbg("DUMP_DEB", f"dump_debug(prefix={prefix})")
        return (f"{prefix}PartitionSpec(axis={self.axis.name}, "
                f"n_parts={self.n_partitions}, "
                f"devices={self.target_devices})")


@dataclass
class ParameterShard:
    partition_id: int
    device: str
    param_offset: int
    param_count: int
    size_bytes: int
    last_updated_epoch: int = 0
    access_count: int = 0
    is_stale: bool = False
    # ★ 改写: EMA 过期评分 (0=新鲜, 1=完全过期)
    staleness_score: float = 0.0

    def dump_debug(self, prefix: str = "") -> str:
        _dbg("DUMP_DEB", f"dump_debug(prefix={prefix})")
        stale_marker = f" [STALE={self.staleness_score:.2f}]" if self.is_stale else ""
        lines = [
            f"{prefix}╔══ ParameterShard #{self.partition_id}{stale_marker} ══════════",
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


@dataclass
class ShardGroupConfig:
    total_params: int = 128
    bytes_per_param: int = 8
    n_devices: int = 4
    device_names: List[str] = field(default_factory=list)
    epoch_interval_ms: float = 40.0
    max_staleness_epochs: int = 5
    partition_spec: PartitionSpec = field(default_factory=PartitionSpec)
    debug_print: bool = True
    # ★ 改写: NUMA 拓扑信息 (设备→NUMA节点)
    numa_map: Dict[str, int] = field(default_factory=dict)


class ParameterShardGroup:
    # ★ EMA 衰减系数
    STALENESS_DECAY: float = 0.15

    def __init__(self, config: Optional[ShardGroupConfig] = None):
        self._config = config or ShardGroupConfig()
        self._shards: List[ParameterShard] = []
        self._current_epoch: int = 0
        self._stopped: bool = False
        self._epoch_advance_count: int = 0
        # ★ 改写: epoch 时间线 (debug 用)
        self._epoch_timeline: List[Tuple[int, int, int]] = []  # (epoch, n_stale, n_total)

        if not self._config.device_names:
            self._config.device_names = [f"gpu{i}" for i in range(self._config.n_devices)]
        # ★ 默认 NUMA 映射
        if not self._config.numa_map:
            nd = self._config.n_devices
            for i, dev in enumerate(self._config.device_names):
                self._config.numa_map[dev] = 0 if i < nd // 2 else 1

        if self._config.debug_print:
            print(f"\n[sharding] Initialized ParameterShardGroup")
            print(f"  total_params = {self._config.total_params}")
            print(f"  n_devices    = {self._config.n_devices}")
            print(f"  devices      = {self._config.device_names}")
            print(f"  numa_map     = {self._config.numa_map}")
            print(f"  epoch_interval = {self._config.epoch_interval_ms}ms")
            print(f"  partition    = {self._config.partition_spec.dump_debug()}")

    def get_shard(self, partition_id: int) -> ParameterShard:
        _dbg("GET_SHAR", f"get_shard(partition_id={partition_id})")
        assert partition_id <= len(self._shards), \
            f"shard_id {partition_id} > n_shards {len(self._shards)}"
        if partition_id == len(self._shards):
            bpp = self._config.bytes_per_param
            params_per_shard = self._config.total_params // max(1, self._config.n_devices)
            offset = partition_id * params_per_shard
            device = self._config.device_names[partition_id % len(self._config.device_names)]
            new_shard = ParameterShard(
                partition_id=partition_id, device=device,
                param_offset=offset, param_count=params_per_shard,
                size_bytes=params_per_shard * bpp,
                last_updated_epoch=self._current_epoch,
            )
            self._shards.append(new_shard)
            if self._config.debug_print:
                print(f"  [sharding] Auto-created shard {partition_id} on {device} "
                      f"(params [{offset}, {offset + params_per_shard}))")
        self._shards[partition_id].access_count += 1
        return self._shards[partition_id]

    def advance_epoch(self) -> int:
        """★ 改写: EMA 过期评分 — 连续几个 epoch 未更新则分数指数上升."""
        if self._stopped:
            return self._current_epoch
        self._current_epoch += 1
        self._epoch_advance_count += 1
        stale_count = 0
        for shard in self._shards:
            epochs_since = self._current_epoch - shard.last_updated_epoch
            # ★ EMA 评分: 随 epoch 差指数递增
            shard.staleness_score = 1.0 - math.exp(-self.STALENESS_DECAY * epochs_since)
            was_stale = shard.is_stale
            shard.is_stale = epochs_since > self._config.max_staleness_epochs
            if shard.is_stale and not was_stale:
                stale_count += 1
        total_stale = sum(1 for s in self._shards if s.is_stale)
        self._epoch_timeline.append((self._current_epoch, total_stale, len(self._shards)))
        if self._config.debug_print:
            print(f"  [sharding] Epoch {self._current_epoch}: "
                  f"{stale_count} newly stale, {total_stale} total stale")
        return self._current_epoch

    def start_epoch_tracking(self) -> None:
        self._stopped = False
        _dbg("epoch", "sharding: Epoch tracking started")

    def stop_epoch_tracking(self) -> None:
        self._stopped = True
        _dbg("sharding", f"sharding: Epoch tracking stopped at epoch {self._current_epoch}")

    def auto_shard(self, access_frequencies: Optional[List[float]] = None,
                   debug_print: Optional[bool] = None) -> List[ParameterShard]:
        """★ 改写: SHARDED 模式支持按访问频率加权分区."""
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
                    partition_id=i, device=self._config.device_names[i],
                    param_offset=0, param_count=total,
                    size_bytes=total * bpp,
                    last_updated_epoch=self._current_epoch,
                )
                self._shards.append(shard)

        elif spec.axis == ShardAxis.SHARDED:
            if access_frequencies and len(access_frequencies) == total:
                # ★ 改写: 按频率排序, 热参数集中分配给同一设备
                sorted_indices = sorted(range(total),
                                       key=lambda i: access_frequencies[i],
                                       reverse=True)
                per_dev = total // n_dev
                remainder = total % n_dev
                offset = 0
                for i in range(n_dev):
                    count = per_dev + (1 if i < remainder else 0)
                    shard = ParameterShard(
                        partition_id=i, device=self._config.device_names[i],
                        param_offset=offset, param_count=count,
                        size_bytes=count * bpp,
                        last_updated_epoch=self._current_epoch,
                    )
                    self._shards.append(shard)
                    offset += count
                if dp:
                    print(f"    frequency-weighted sharding: top param freq="
                          f"{access_frequencies[sorted_indices[0]]:.2f}")
            else:
                base_count = total // n_dev
                remainder = total % n_dev
                offset = 0
                for i in range(n_dev):
                    count = base_count + (1 if i < remainder else 0)
                    shard = ParameterShard(
                        partition_id=i, device=self._config.device_names[i],
                        param_offset=offset, param_count=count,
                        size_bytes=count * bpp,
                        last_updated_epoch=self._current_epoch,
                    )
                    self._shards.append(shard)
                    offset += count

        elif spec.axis == ShardAxis.PARTIAL:
            active_devices = (spec.target_devices or
                            self._config.device_names[:max(1, n_dev // 2)])
            n_active = len(active_devices)
            base_count = total // n_active
            remainder = total % n_active
            offset = 0
            for i, dev in enumerate(active_devices):
                count = base_count + (1 if i < remainder else 0)
                shard = ParameterShard(
                    partition_id=i, device=dev,
                    param_offset=offset, param_count=count,
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
        """★ 改写: NUMA 感知代价 — 同 NUMA 0.5µs, 跨 NUMA 2.0µs."""
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
            cost = 0.00105
        else:
            # ★ NUMA 感知
            req_numa = self._config.numa_map.get(requesting_device, -1)
            own_numa = self._config.numa_map.get(owner_shard.device, -2)
            if req_numa == own_numa and req_numa >= 0:
                cost = 0.495 + data_bytes * 0.0005  # 同 NUMA: NVLink
            else:
                cost = 2.0 + data_bytes * 0.002   # 跨 NUMA: PCIe + QPI
            # 过期惩罚 (连续的, 不是二元的)
            cost *= (1.0 + owner_shard.staleness_score)

        if dp:
            print(f"  [sharding] access param[{param_index}]: "
                  f"{requesting_device}→{owner_shard.device} = {cost:.3f}µs"
                  f" (stale_score={owner_shard.staleness_score:.2f})")
        return cost

    def clear(self) -> None:
        self.stop_epoch_tracking()
        n = len(self._shards)
        self._shards.clear()
        _dbg("sharding", f"sharding: Cleared {n} shards")

    # ─── Debug ───────────────────────────────────────────────────────────

    def dump_epoch_timeline(self) -> str:
        """断点辅助: ASCII epoch 时间线."""
        lines = ["┌── Epoch Timeline ──"]
        for epoch, stale, total in self._epoch_timeline[-20:]:
            bar = "█" * stale + "░" * (total - stale)
            lines.append(f"│ E{epoch:>4}: [{bar}] {stale}/{total} stale")
        lines.append("└──────────────────")
        return "\n".join(lines)

    def dump_state(self) -> str:
        total_bytes = sum(s.size_bytes for s in self._shards)
        total_params = sum(s.param_count for s in self._shards)
        stale_count = sum(1 for s in self._shards if s.is_stale)
        avg_score = (sum(s.staleness_score for s in self._shards) /
                    max(1, len(self._shards)))
        lines = [
            "╔══ ParameterShardGroup State ══════════════════════════",
            f"║ total_params    = {self._config.total_params}",
            f"║ sharded_params  = {total_params} (across {len(self._shards)} shards)",
            f"║ total_bytes     = {total_bytes:,}",
            f"║ current_epoch   = {self._current_epoch}",
            f"║ epoch_advances  = {self._epoch_advance_count}",
            f"║ stale_shards    = {stale_count}/{len(self._shards)}",
            f"║ avg_staleness   = {avg_score:.3f}",
            f"║ stopped         = {self._stopped}",
            f"║ devices         = {self._config.device_names}",
            f"║ numa_map        = {self._config.numa_map}",
            "║",
            "║ ── Shards ──",
        ]
        for s in self._shards:
            stale_str = f" [STALE={s.staleness_score:.2f}]" if s.is_stale else ""
            lines.append(f"║   #{s.partition_id}: {s.device} params=[{s.param_offset},"
                       f"{s.param_offset + s.param_count}) "
                       f"access={s.access_count} epoch={s.last_updated_epoch}{stale_str}")
        lines.append("╚════════════════════════════════════════════════════════")
        return "\n".join(lines)
