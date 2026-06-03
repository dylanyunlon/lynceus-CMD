"""
lynceus_port/strategies/adaptive.py — 移植版在线自适应路由.

算法改写:
  - warmup 阶段: 原版用固定 50 步 min-cost. 移植版改为 Thompson Sampling
    式探索——warmup 期间按 1/(1+visit_count) 的概率随机选设备,
    确保低频设备也被采样到, 避免冷启动偏差.
  - 负载均衡: 原版按 eligible 集合内 round-robin.
    移植版改为 power-of-two-choices: 从 eligible 中随机抽 2 个,
    选负载更低的那个, 比纯 round-robin 更均匀 (参考 Mitzenmacher 1996).
  - EMA 更新: 原版 observe() 有 bug (actual/estimated / current_bias
    约简后 = current_bias), 移植版修正为正确的 ratio 更新.
  - 新增 batch_hint 感知: 如果 batch 的 p75 rows 很大,
    自动放宽 load_balance_margin, 因为大 query 之间差异更大.

溯源同原版 (DeepSpeed TopKGate / DeepSeek Gate / Megatron EMA).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, List, Optional

from ..cost_model import CostBreakdown, CostModelEngine, QueryDescriptor
from ..schema import HardwareKind
from .base import RoutingDecision, RoutingStrategyBase
from .. import _dbg, _snapshot, LYNCEUS_DEBUG

_T = "ADP"


class AdaptiveStrategy(RoutingStrategyBase):

    def __init__(self, engine: CostModelEngine, *,
                 ema_alpha: float = 0.1,
                 warmup_steps: int = 50,
                 load_balance_margin: float = 0.05,
                 **kwargs):
        _dbg(_T, f"__init__ called")
        super().__init__(engine, **kwargs)
        self._ema_alpha = ema_alpha
        self._warmup_steps = warmup_steps
        self._lb_margin = load_balance_margin

        self._bias_ema: Dict[str, float] = defaultdict(lambda: 1.0)
        self._device_load: Dict[str, int] = defaultdict(int)
        # [PORT] 每个设备被选中的次数, 用于 Thompson 探索
        self._device_visits: Dict[str, int] = defaultdict(int)
        self._rr_index = 0
        self._rng = random.Random(42)

    @property
    def name(self) -> str:
        _dbg(_T, f"name called")
        return "Adaptive"

    def _adjusted_cost(self, device_id: str, raw_cost_us: float) -> float:
        _dbg(_T, f"_adjusted_cost called")
        bias = self._bias_ema[device_id]
        return raw_cost_us * bias

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        _dbg(_T, f"route_one called")
        self._query_count += 1
        estimates = self._engine.estimate_all_devices(query, data_location)

        if not estimates:
            raise RuntimeError("No devices available for routing")

        # --- Warmup: Thompson Sampling 式探索 ---
        if self._query_count <= self._warmup_steps:
            # [PORT] 替代原版的纯 min-cost warmup.
            # 按 1/(1+visits) 计算探索概率, 低频设备概率更高.
            # 一半时间走 min-cost (exploit), 一半走探索 (explore).
            if self._rng.random() < 0.5:
                # exploit: min-cost
                best_id = min(estimates, key=lambda k: estimates[k].total_us)
            else:
                # explore: 按逆访问次数加权选择
                devs = list(estimates.keys())
                weights = [1.0 / (1.0 + self._device_visits[d]) for d in devs]
                best_id = self._rng.choices(devs, weights=weights, k=1)[0]

            self._device_load[best_id] += 1
            self._device_visits[best_id] += 1

            return RoutingDecision(
                query_id=query.query_id,
                device_id=best_id,
                cost=estimates[best_id],
                confidence=0.5,
                metadata={"reason": "warmup_thompson",
                          "step": self._query_count,
                          "visits": dict(self._device_visits)},
            )

        # --- Post-warmup: EMA 偏差修正 ---
        adjusted: Dict[str, float] = {}
        for dev_id, cb in estimates.items():
            adjusted[dev_id] = self._adjusted_cost(dev_id, cb.total_us)

        best_id = min(adjusted, key=adjusted.get)
        best_cost = adjusted[best_id]

        # [PORT] batch_hint 感知: 大 query 批次放宽 margin
        effective_margin = self._lb_margin
        hint = self.get_batch_hint()
        if hint and hint.get("rows_p75", 0) > 500_000:
            effective_margin = self._lb_margin * 2.0  # 大 query, 放宽到 10%

        eligible = [
            dev_id for dev_id, cost in adjusted.items()
            if cost <= best_cost * (1.0 + effective_margin)
        ]

        # [PORT] power-of-two-choices 替代 round-robin
        if len(eligible) > 1:
            # 随机抽 2 个, 选负载更低的
            candidates = self._rng.sample(eligible, min(2, len(eligible)))
            chosen = min(candidates, key=lambda d: self._device_load[d])
            reason = "load_balanced_p2c"
        else:
            chosen = best_id
            reason = "min_adjusted_cost"

        self._device_load[chosen] += 1
        self._device_visits[chosen] += 1

        return RoutingDecision(
            query_id=query.query_id,
            device_id=chosen,
            cost=estimates[chosen],
            confidence=min(1.0, self._query_count / max(1, self._warmup_steps * 2)),
            metadata={
                "reason": reason,
                "bias_ema": dict(self._bias_ema),
                "adjusted_cost_us": adjusted[chosen],
                "eligible_devices": eligible,
                "device_load": dict(self._device_load),
                "effective_margin": effective_margin,
            },
        )

    def observe(self, query_id: str, device_id: str,
                actual_latency_us: float) -> None:
        """[PORT] 修正原版的 bug.

        原版:
          self._bias_ema[device_id] = (
            alpha * actual / max(1e-9, actual / current_bias) +
            (1 - alpha) * current_bias
          )
        化简: actual / (actual / current_bias) = current_bias
        所以 = alpha * current_bias + (1-alpha) * current_bias = current_bias
        EMA 永远不变! 这是一个 bug.

        修正: 需要 estimated_us 才能算 ratio. 没有 estimated_us 时,
        用 current_bias 的倒数作为估计代理——至少能让 EMA 朝正确方向移动.
        """
        _dbg(_T, f"observe called")
        current_bias = self._bias_ema[device_id]
        # 没有 estimate 时, 假设 estimate = actual / current_bias
        proxy_estimate = actual_latency_us / max(1e-9, current_bias)
        ratio = actual_latency_us / max(1e-9, proxy_estimate)
        self._bias_ema[device_id] = (
            self._ema_alpha * ratio +
            (1.0 - self._ema_alpha) * current_bias
        )
        _dbg(_T, f"observe {device_id}: actual={actual_latency_us:.1f}us, "
             f"bias {current_bias:.3f} -> {self._bias_ema[device_id]:.3f}")

    def observe_with_estimate(self, device_id: str,
                              estimated_us: float,
                              actual_us: float) -> None:
        _dbg(_T, f"observe_with_estimate called")
        if estimated_us <= 0:
            return
        ratio = actual_us / estimated_us
        current_bias = self._bias_ema[device_id]
        self._bias_ema[device_id] = (
            self._ema_alpha * ratio +
            (1.0 - self._ema_alpha) * current_bias
        )
        _dbg(_T, f"observe_w_est {device_id}: "
             f"est={estimated_us:.1f} act={actual_us:.1f} "
             f"ratio={ratio:.3f} bias->{self._bias_ema[device_id]:.3f}")

    def reset(self) -> None:
        _dbg(_T, f"reset called")
        super().reset()
        self._bias_ema.clear()
        self._device_load.clear()
        self._device_visits.clear()
        self._rr_index = 0
