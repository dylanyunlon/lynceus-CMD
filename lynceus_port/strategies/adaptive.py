"""
lynceus_port/strategies/adaptive.py — 在线自适应路由策略。

移植自 lynceus/strategies/adaptive.py，修改约20%:
  - EMA 改为带滑动窗口的衰减加权平均（窗口内保留原始观测值）
  - observe_with_estimate: 打印逐次偏差更新
  - debug_snapshot: 完整内部状态输出
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

from ..cost_model import CostBreakdown, CostModelEngine, QueryDescriptor
from ..schema import HardwareKind, _dbg
from .base import RoutingDecision, RoutingStrategyBase


class AdaptiveStrategy(RoutingStrategyBase):
    """在线自适应路由：带窗口的 EMA 偏差修正 + 负载均衡。

    修改点：保留最近 window_size 个观测值，在窗口内计算加权平均偏差，
    权重随距离指数衰减。比纯 EMA 更能应对突变（burst）。
    """

    def __init__(self, engine: CostModelEngine, *,
                 ema_alpha: float = 0.1,
                 warmup_steps: int = 50,
                 load_balance_margin: float = 0.05,
                 window_size: int = 100,
                 **kwargs):
        super().__init__(engine, **kwargs)
        self._ema_alpha = ema_alpha
        self._warmup_steps = warmup_steps
        self._lb_margin = load_balance_margin
        self._window_size = window_size

        # ── 修改：带窗口的偏差追踪 ──
        self._bias_window: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=window_size))
        self._bias_ema: Dict[str, float] = defaultdict(lambda: 1.0)

        self._device_load: Dict[str, int] = defaultdict(int)
        self._rr_index = 0

    @property
    def name(self) -> str:
        return "online_adaptive"

    def _windowed_bias(self, device_id: str) -> float:
        """从窗口内的观测值计算衰减加权平均偏差"""
        window = self._bias_window[device_id]
        if not window:
            return self._bias_ema[device_id]

        total_weight = 0.0
        weighted_sum = 0.0
        alpha = self._ema_alpha
        for i, ratio in enumerate(reversed(window)):
            w = (1.0 - alpha) ** i  # 越近的权重越大
            weighted_sum += w * ratio
            total_weight += w

        return weighted_sum / total_weight if total_weight > 0 else 1.0

    def _adjusted_cost(self, device_id: str, raw_cost_us: float) -> float:
        bias = self._windowed_bias(device_id)
        return raw_cost_us * bias

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        self._query_count += 1
        estimates = self._engine.estimate_all_devices(query, data_location)

        if not estimates:
            raise RuntimeError("没有可用设备")

        # warmup 阶段：纯最小代价
        if self._query_count <= self._warmup_steps:
            best_id = min(estimates, key=lambda k: estimates[k].total_us)
            self._device_load[best_id] += 1
            _dbg("Adaptive",
                 f"warmup #{self._query_count}: {query.query_id} -> {best_id}")
            return RoutingDecision(
                query_id=query.query_id, device_id=best_id,
                cost=estimates[best_id], confidence=0.5,
                metadata={"reason": "warmup", "step": self._query_count},
            )

        # 后 warmup：应用偏差修正
        adjusted: Dict[str, float] = {}
        for dev_id, cb in estimates.items():
            adjusted[dev_id] = self._adjusted_cost(dev_id, cb.total_us)

        best_id = min(adjusted, key=adjusted.get)  # type: ignore
        best_cost = adjusted[best_id]

        # 负载均衡
        eligible = [
            dev_id for dev_id, cost in adjusted.items()
            if cost <= best_cost * (1.0 + self._lb_margin)
        ]

        if len(eligible) > 1:
            chosen = eligible[self._rr_index % len(eligible)]
            self._rr_index += 1
            reason = "load_balanced"
        else:
            chosen = best_id
            reason = "min_adjusted_cost"

        self._device_load[chosen] += 1

        _dbg("Adaptive",
             f"#{self._query_count} {query.query_id} -> {chosen} "
             f"(adj={adjusted[chosen]:.1f}us, raw={estimates[chosen].total_us:.1f}us, "
             f"bias={self._windowed_bias(chosen):.3f}, {reason})")

        return RoutingDecision(
            query_id=query.query_id, device_id=chosen,
            cost=estimates[chosen],
            confidence=min(1.0, self._query_count / max(1, self._warmup_steps * 2)),
            metadata={
                "reason": reason,
                "adjusted_cost_us": adjusted[chosen],
                "eligible_devices": eligible,
                "device_load": dict(self._device_load),
            },
        )

    def observe(self, query_id: str, device_id: str,
                actual_latency_us: float) -> None:
        current_bias = self._bias_ema[device_id]
        self._bias_ema[device_id] = (
            self._ema_alpha * actual_latency_us /
            max(1e-9, actual_latency_us / current_bias) +
            (1.0 - self._ema_alpha) * current_bias
        )

    def observe_with_estimate(self, device_id: str,
                              estimated_us: float,
                              actual_us: float) -> None:
        if estimated_us <= 0:
            return
        ratio = actual_us / estimated_us

        # ── 窗口记录 ──
        self._bias_window[device_id].append(ratio)

        # 同时更新 EMA（作为备选）
        current_bias = self._bias_ema[device_id]
        self._bias_ema[device_id] = (
            self._ema_alpha * ratio +
            (1.0 - self._ema_alpha) * current_bias
        )

        _dbg("Adaptive",
             f"observe {device_id}: est={estimated_us:.1f} act={actual_us:.1f} "
             f"ratio={ratio:.3f} window_len={len(self._bias_window[device_id])} "
             f"ema_bias={self._bias_ema[device_id]:.3f}")

    def reset(self) -> None:
        super().reset()
        self._bias_ema.clear()
        self._bias_window.clear()
        self._device_load.clear()
        self._rr_index = 0
        _dbg("Adaptive", "full reset")

    def debug_snapshot(self) -> str:
        lines = [
            f"=== AdaptiveStrategy Snapshot ===",
            f"  queries_seen = {self._query_count}",
            f"  warmup_steps = {self._warmup_steps}",
            f"  window_size  = {self._window_size}",
            f"  device_load  = {dict(self._device_load)}",
        ]
        for dev_id in sorted(set(list(self._bias_ema.keys()) +
                                 list(self._bias_window.keys()))):
            ema = self._bias_ema.get(dev_id, 1.0)
            wlen = len(self._bias_window.get(dev_id, []))
            wbias = self._windowed_bias(dev_id)
            lines.append(f"  {dev_id}: ema={ema:.3f}, "
                         f"window_bias={wbias:.3f} (n={wlen})")
        s = "\n".join(lines)
        _dbg("Adaptive", s)
        return s
