"""
lynceus_port_v3/strategies/adaptive.py — Online adaptive routing strategy.

NEW in M003-M004: a strategy that learns from past routing decisions
to improve future ones. This is the key differentiator over static
cost-model routing.

Two key innovations:
    1. EMA cost tracking: maintain per-device exponential moving average
       of actual vs estimated costs, adjusting future estimates by the
       observed bias. This corrects systematic errors in the cost model.

    2. Multi-GPU load balancing: when multiple GPUs have similar costs,
       distribute queries across them to maximize throughput (avoid
       queueing on a single GPU).

Architecture references:
    - DeepSpeed TopKGate (deepspeed/moe/sharded_moe.py:452)
      → capacity_factor controls load balance across experts
      → drop_tokens prevents expert overload
    - DeepSeek Gate.forward (DeepSeek-V3/inference/model.py:581)
      → top-k routing with group-aware scoring
      → route_scale adjusts routing weights
    - vLLM Scheduler.schedule (vllm/v1/core/sched/scheduler.py:334)
      → workload-aware scheduling considering KV cache state
    - Megatron DistributedOptimizer (distrib_optimizer.py:102)
      → EMA-based gradient statistics for adaptive optimization
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional

from ..cost_model import CostBreakdown, CostModelEngine, QueryDescriptor
from ..schema import HardwareKind
from .base import RoutingDecision, RoutingStrategyBase



# --- port_v3: debug instrumentation ---
try:
    from ._debug import dbg, snapshot, timing, checkpoint
except ImportError:
    from lynceus_port_v3._debug import dbg, snapshot, timing, checkpoint

class AdaptiveStrategy(RoutingStrategyBase):
    """Online adaptive routing with EMA bias correction and load balancing.

    Lifecycle:
        1. For the first `warmup_steps` queries, behaves like
           CostModelRoutedStrategy (pure min-cost).
        2. After warmup, adjusts cost estimates using EMA of the
           ratio (actual / estimated) for each device.
        3. When multiple devices are within `load_balance_margin` of
           the best cost, distributes queries round-robin.

    Parameters:
        ema_alpha: EMA decay factor (0.0 = no memory, 1.0 = full memory).
                   Default 0.1 — quickly adapts to recent observations.
        warmup_steps: Number of queries before EMA kicks in.
        load_balance_margin: If a device's adjusted cost is within this
                             fraction of the best, it's eligible for
                             load balancing. Default 0.05 (5%).
    """

    def __init__(self, engine: CostModelEngine, *,
                 ema_alpha: float = 0.12,
                 warmup_steps: int = 40,
                 load_balance_margin: float = 0.06,
                 **kwargs):
        super().__init__(engine, **kwargs)
        self._ema_alpha = ema_alpha
        self._warmup_steps = warmup_steps
        self._lb_margin = load_balance_margin

        # v3: Holt double-exponential smoothing (level + trend)
        # Captures both current bias AND drift direction
        self._bias_level: Dict[str, float] = defaultdict(lambda: 1.0)
        self._bias_trend: Dict[str, float] = defaultdict(lambda: 0.0)
        self._beta_trend = 0.05  # trend smoothing factor

        # Per-device query count for load balancing
        self._device_load: Dict[str, int] = defaultdict(int)

        # Round-robin index for tie-breaking
        self._rr_index = 0

    @property
    def name(self) -> str:
        return "Adaptive"

    def _adjusted_cost(self, device_id: str,
                       raw_cost_us: float) -> float:
        """v3: Holt forecast = level + trend (one-step-ahead prediction)."""
        level = self._bias_level[device_id]
        trend = self._bias_trend[device_id]
        forecast = level + trend  # one-step-ahead
        return raw_cost_us * max(0.1, forecast)

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        from ._debug import dbg
        dbg('Adaptive.route', query_id=query.query_id, query_count=self._query_count)
        estimates = self._engine.estimate_all_devices(query, data_location)

        if not estimates:
            raise RuntimeError("No devices available for routing")

        # During warmup: pure min-cost (no adaptation)
        if self._query_count <= self._warmup_steps:
            best_id = min(estimates, key=lambda k: estimates[k].total_us)
            self._device_load[best_id] += 1
            return RoutingDecision(
                query_id=query.query_id,
                device_id=best_id,
                cost=estimates[best_id],
                confidence=0.5,  # low confidence during warmup
                metadata={"reason": "warmup", "step": self._query_count},
            )

        # Post-warmup: apply EMA bias correction
        adjusted: Dict[str, float] = {}
        for dev_id, cb in estimates.items():
            adjusted[dev_id] = self._adjusted_cost(dev_id, cb.total_us)

        best_id = min(adjusted, key=adjusted.get)
        best_cost = adjusted[best_id]

        # Load balancing: find all devices within margin of best
        # (like DeepSpeed TopKGate selecting top-k experts with
        #  capacity_factor controlling distribution)
        eligible = [
            dev_id for dev_id, cost in adjusted.items()
            if cost <= best_cost * (1.0 + self._lb_margin)
        ]

        if len(eligible) > 1:
            # v3: weighted probability sampling (inverse-cost weighting)
            import random
            inv_costs = [1.0 / max(1e-6, adjusted[d]) for d in eligible]
            total_inv = sum(inv_costs)
            probs = [ic / total_inv for ic in inv_costs]
            cumulative = 0.0
            r = random.random()
            chosen = eligible[-1]
            for idx, p in enumerate(probs):
                cumulative += p
                if r <= cumulative:
                    chosen = eligible[idx]
                    break
            reason = "weighted_sample"
        else:
            chosen = best_id
            reason = "min_adjusted_cost"

        self._device_load[chosen] += 1

        return RoutingDecision(
            query_id=query.query_id,
            device_id=chosen,
            cost=estimates[chosen],
            confidence=min(1.0, self._query_count / max(1, self._warmup_steps * 2)),
            metadata={
                "reason": reason,
                "bias_level": dict(self._bias_level),
                "bias_trend": dict(self._bias_trend),
                "adjusted_cost_us": adjusted[chosen],
                "eligible_devices": eligible,
                "device_load": dict(self._device_load),
            },
        )

    def observe(self, query_id: str, device_id: str,
                actual_latency_us: float) -> None:
        """Update EMA bias from actual execution feedback.

        EMA update rule (same as Megatron's grad norm tracking):
            bias_new = alpha * (actual / estimated) + (1 - alpha) * bias_old

        This corrects systematic over/under-estimation per device.
        """
        # We need the last estimated cost; for simplicity, update
        # the bias directly. In production, we'd store the estimate
        # alongside the decision.
        current_bias = self._bias_ema[device_id]
        # Assume observed ratio; if actual > estimated bias > 1, inflate
        # If we don't have the estimate handy, use the bias adjustment
        # relative to "expected" (current_bias * some_base)
        # For now, treat actual as the ground truth and adjust:
        # v3: delegate to Holt update (use current level as estimate proxy)
        proxy_estimated = actual_latency_us / max(0.1, self._bias_level[device_id])
        self.observe_with_estimate(device_id, proxy_estimated, actual_latency_us)

    def observe_with_estimate(self, device_id: str,
                              estimated_us: float,
                              actual_us: float) -> None:
        """v3: Holt double-exponential update with outlier damping.

        Level update:  L_t = α * ratio + (1-α) * (L_{t-1} + T_{t-1})
        Trend update:  T_t = β * (L_t - L_{t-1}) + (1-β) * T_{t-1}
        Outlier: if |ratio - level| > 4*|trend|, damp α to α/4.
        """
        if estimated_us <= 0:
            return
        ratio = actual_us / estimated_us
        old_level = self._bias_level[device_id]
        old_trend = self._bias_trend[device_id]

        warmup_decay = min(1.0, self._query_count / max(1, self._warmup_steps * 3))
        eff_alpha = self._ema_alpha * warmup_decay

        # v3: outlier damping — if observation deviates wildly, reduce alpha
        deviation = abs(ratio - old_level)
        trend_magnitude = max(abs(old_trend), 0.01)
        if deviation > 4.0 * trend_magnitude:
            eff_alpha *= 0.25  # damp outliers

        # Holt level update
        new_level = eff_alpha * ratio + (1.0 - eff_alpha) * (old_level + old_trend)
        # Holt trend update
        new_trend = self._beta_trend * (new_level - old_level) + (1.0 - self._beta_trend) * old_trend

        self._bias_level[device_id] = new_level
        self._bias_trend[device_id] = new_trend

    def reset(self) -> None:
        # v3: reset Holt state
        super().reset()
        self._bias_ema.clear()
        self._device_load.clear()
        self._rr_index = 0
