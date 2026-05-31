"""
lynceus_port/strategies/adaptive.py — 在线自适应路由.

改写: 增加 UCB (Upper Confidence Bound) 探索项 —
      访问次数少的设备获得探索奖励, 避免过早收敛到次优设备.
"""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Dict, List, Optional
from ..cost_model import CostBreakdown, CostModelEngine, QueryDescriptor
from ..schema import HardwareKind
from .base import RoutingDecision, RoutingStrategyBase
from .. import _dbg


class AdaptiveStrategy(RoutingStrategyBase):
    def __init__(self, engine: CostModelEngine, *,
                 ema_alpha: float = 0.1,
                 warmup_steps: int = 50,
                 load_balance_margin: float = 0.05,
                 ucb_coeff: float = 2.0,  # ★ UCB 探索系数
                 **kw):
        super().__init__(engine, **kw)
        self._ema_alpha = ema_alpha
        self._warmup_steps = warmup_steps
        self._lb_margin = load_balance_margin
        self._ucb_c = ucb_coeff
        self._bias_ema: Dict[str, float] = defaultdict(lambda: 1.0)
        self._device_load: Dict[str, int] = defaultdict(int)
        self._rr_index = 0

    @property
    def name(self) -> str:
        return "Adaptive"

    def _adjusted_cost(self, device_id: str, raw_cost_us: float) -> float:
        bias = self._bias_ema[device_id]
        return raw_cost_us * bias

    def _ucb_bonus(self, device_id: str) -> float:
        """UCB 探索项 — 访问少的设备获得负成本奖励."""
        total = max(1, sum(self._device_load.values()))
        visits = max(1, self._device_load.get(device_id, 0))
        return -self._ucb_c * math.sqrt(math.log(total) / visits)

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        self._query_count += 1
        estimates = self._engine.estimate_all_devices(query, data_location)
        if not estimates:
            raise RuntimeError("No devices available")

        if self._query_count <= self._warmup_steps:
            best_id = min(estimates, key=lambda k: estimates[k].total_us)
            self._device_load[best_id] += 1
            return RoutingDecision(
                query_id=query.query_id, device_id=best_id,
                cost=estimates[best_id], confidence=0.5,
                metadata={"reason": "warmup", "step": self._query_count},
                trace_log=[f"warmup step {self._query_count}"],
            )

        # ★ 改写: adjusted cost + UCB exploration bonus
        adjusted: Dict[str, float] = {}
        for dev_id, cb in estimates.items():
            adj = self._adjusted_cost(dev_id, cb.total_us)
            ucb = self._ucb_bonus(dev_id)
            adjusted[dev_id] = adj + ucb

        best_id = min(adjusted, key=adjusted.get)
        best_cost = adjusted[best_id]

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

        trace = [
            f"adj_costs: {', '.join(f'{k}={v:.1f}' for k, v in sorted(adjusted.items()))}",
            f"eligible: {eligible}, chosen={chosen} ({reason})",
        ]
        return RoutingDecision(
            query_id=query.query_id, device_id=chosen,
            cost=estimates[chosen],
            confidence=min(1.0, self._query_count / max(1, self._warmup_steps * 2)),
            metadata={"reason": reason, "bias_ema": dict(self._bias_ema),
                      "adjusted_cost_us": adjusted[chosen],
                      "eligible_devices": eligible,
                      "device_load": dict(self._device_load)},
            trace_log=trace,
        )

    def observe(self, query_id: str, device_id: str,
                actual_latency_us: float) -> None:
        current_bias = self._bias_ema[device_id]
        self._bias_ema[device_id] = (
            self._ema_alpha * actual_latency_us / max(1e-9, actual_latency_us / current_bias) +
            (1.0 - self._ema_alpha) * current_bias
        )

    def observe_with_estimate(self, device_id: str,
                              estimated_us: float, actual_us: float) -> None:
        if estimated_us <= 0:
            return
        ratio = actual_us / estimated_us
        current_bias = self._bias_ema[device_id]
        self._bias_ema[device_id] = (
            self._ema_alpha * ratio + (1.0 - self._ema_alpha) * current_bias
        )

    def reset(self) -> None:
        super().reset()
        self._bias_ema.clear()
        self._device_load.clear()
        self._rr_index = 0

    def dump_snapshot(self) -> str:
        return (f"Adaptive(queries={self._query_count}, "
                f"biases={dict(self._bias_ema)}, "
                f"loads={dict(self._device_load)})")
