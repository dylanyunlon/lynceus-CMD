"""lynceus/strategies/adaptive.py — Online adaptive routing with softmax balancing."""
from __future__ import annotations
import math
import random as _random
from collections import defaultdict
from typing import Dict, List, Optional
from ..cost_model import CostBreakdown, CostModelEngine, QueryDescriptor
from ..schema import HardwareKind
from .base import RoutingDecision, RoutingStrategyBase

class AdaptiveStrategy(RoutingStrategyBase):
    """改动: softmax 概率采样代替 round-robin, 温度退火。"""
    def __init__(self, engine: CostModelEngine, *, ema_alpha: float = 0.12,
                 warmup_steps: int = 40, initial_temperature: float = 50.0,
                 min_temperature: float = 2.0, anneal_rate: float = 0.97, **kwargs):
        super().__init__(engine, **kwargs)
        self._ema_alpha = ema_alpha
        self._warmup_steps = warmup_steps
        self._temperature = initial_temperature
        self._min_temp = min_temperature
        self._anneal_rate = anneal_rate
        self._bias_ema: Dict[str, float] = defaultdict(lambda: 1.0)
        self._device_load: Dict[str, int] = defaultdict(int)
    @property
    def name(self) -> str:
        return "Adaptive"
    def _adjusted_cost(self, device_id: str, raw_cost_us: float) -> float:
        return raw_cost_us * self._bias_ema[device_id]
    def _softmax_select(self, costs: Dict[str, float]) -> str:
        items = list(costs.items())
        if not items:
            raise RuntimeError("No devices")
        max_neg = max(-c for _, c in items)
        weights = [math.exp((-c - max_neg) / max(0.01, self._temperature)) for _, c in items]
        total = sum(weights)
        if total <= 0:
            return items[0][0]
        r = _random.random() * total
        cumulative = 0.0
        for i, (dev_id, _) in enumerate(items):
            cumulative += weights[i]
            if r <= cumulative:
                return dev_id
        return items[-1][0]
    def route_one(self, query: QueryDescriptor, data_location: Optional[str] = None) -> RoutingDecision:
        from .._debug import dbg, checkpoint
        dbg('Adaptive.route', query_id=query.query_id,
            query_count=self._query_count, temperature=self._temperature)
        estimates = self._engine.estimate_all_devices(query, data_location)
        if not estimates:
            raise RuntimeError("No devices available for routing")
        if self._query_count <= self._warmup_steps:
            best_id = min(estimates, key=lambda k: estimates[k].total_us)
            self._device_load[best_id] += 1
            self._query_count += 1
            return RoutingDecision(query_id=query.query_id, device_id=best_id,
                cost=estimates[best_id], confidence=0.5,
                metadata={"reason": "warmup", "step": self._query_count})
        adjusted = {dev_id: self._adjusted_cost(dev_id, cb.total_us)
                    for dev_id, cb in estimates.items()}
        chosen = self._softmax_select(adjusted)
        self._device_load[chosen] += 1
        self._query_count += 1
        self._temperature = max(self._min_temp, self._temperature * self._anneal_rate)
        checkpoint("adaptive_select", chosen=chosen, temperature=self._temperature,
                   adjusted_costs={k: f"{v:.1f}" for k, v in adjusted.items()})
        return RoutingDecision(query_id=query.query_id, device_id=chosen,
            cost=estimates[chosen],
            confidence=min(1.0, self._query_count / max(1, self._warmup_steps * 2)),
            metadata={"reason": "softmax_select", "temperature": self._temperature,
                      "bias_ema": dict(self._bias_ema), "adjusted_cost_us": adjusted[chosen],
                      "device_load": dict(self._device_load)})
    def observe(self, query_id: str, device_id: str, actual_latency_us: float) -> None:
        current_bias = self._bias_ema[device_id]
        self._bias_ema[device_id] = (
            self._ema_alpha * actual_latency_us / max(1e-9, actual_latency_us / current_bias) +
            (1.0 - self._ema_alpha) * current_bias)
    def observe_with_estimate(self, device_id: str, estimated_us: float, actual_us: float) -> None:
        if estimated_us <= 0:
            return
        ratio = actual_us / estimated_us
        current_bias = self._bias_ema[device_id]
        warmup_decay = min(1.0, self._query_count / max(1, self._warmup_steps * 3))
        eff_alpha = self._ema_alpha * warmup_decay
        self._bias_ema[device_id] = eff_alpha * ratio + (1.0 - eff_alpha) * current_bias
    def reset(self) -> None:
        super().reset()
        self._bias_ema.clear()
        self._device_load.clear()
        self._temperature = 50.0
