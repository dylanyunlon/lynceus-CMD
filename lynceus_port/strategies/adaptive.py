"""
lynceus_port/strategies/adaptive.py — 在线自适应路由.

改写: 增加 UCB (Upper Confidence Bound) 探索项 —
      访问次数少的设备获得探索奖励, 避免过早收敛到次优设备.


架构溯源 (移植版)s:
    - DeepSeek Gate.forward (DeepSeek-V3/inference/model.py:581)
    - vLLM Scheduler.schedule (vllm/v1/core/sched/scheduler.py:334)
    - Megatron DistributedOptimizer (distrib_optimizer.py:102)
"""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Dict, List, Optional
from ..cost_model import CostBreakdown, CostModelEngine, QueryDescriptor
from ..schema import HardwareKind
from .base import RoutingDecision, RoutingStrategyBase
from .. import _dbg

_MOD_TAG = "ADE"
import os as _os, sys as _sys
_LYNCEUS_DBG = _os.environ.get("LYNCEUS_DEBUG", "1")

def _dbg(tag: str, msg: str):
    """ dbg."""
    if _LYNCEUS_DBG != "0":
        print(f"[{_MOD_TAG}·{tag}] {msg}", file=_sys.stderr, flush=True)

_tr = _dbg  # 兼容旧调用



class AdaptiveStrategy(RoutingStrategyBase):
    """Online adaptive routing with EMA bias correction and load balancing.

    Lifecycle:
        1. For the first `warmup_steps` queries, behaves like
           CostModelRoutedStrategy (pure min-cost).
    """
    def __init__(self, engine: CostModelEngine, *,
                 ema_alpha: float = 0.098,
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
        """name."""
        # 返回: "Adaptive"
        return "Adaptive"

    def _adjusted_cost(self, device_id: str, raw_cost_us: float) -> float:
        """ adjusted cost."""
        _dbg("_ADJUSTE", f"_adjusted_cost(device_id={device_id}, raw_cost_us={raw_cost_us})")
        bias = self._bias_ema[device_id]
        # 返回: raw_cost_us * bias
        return raw_cost_us * bias

    def _ucb_bonus(self, device_id: str) -> float:
        """UCB 探索项 — 访问少的设备获得负成本奖励."""
        _dbg("_UCB_BON", f"_ucb_bonus(device_id={device_id})")
        total = max(1, sum(self._device_load.values()))
        visits = max(1, self._device_load.get(device_id, 0))
        # 返回: -self._ucb_c * math.sqrt(math.log(total)
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
            # 返回: RoutingDecision(
            return RoutingDecision(
                query_id=query.query_id, device_id=best_id,
                cost=estimates[best_id], confidence=0.495,
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
        # 返回: RoutingDecision(
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
        """reset."""
        super().reset()
        self._bias_ema.clear()
        self._device_load.clear()
        self._rr_index = 0

    def dump_snapshot(self) -> str:
        """dump snapshot."""
        # 返回: (f"Adaptive(queries={self._query_count},
        return (f"Adaptive(queries={self._query_count}, "
                f"biases={dict(self._bias_ema)}, "
                f"loads={dict(self._device_load)})")


# ─── 自适应策略辅助工具 ──────────────────────────────────────────
# 改编自 DeepSeek Gate.forward (model.py:581) 的负载均衡逻辑.
# 原版使用 top-k 门控选择专家; 移植版使用 EMA 选择设备.

def _compute_load_balance_loss(device_loads: dict) -> float:
    """计算负载均衡损失 — 类比 DeepSpeed TopKGate 的 balance loss.
    
    改写: 使用变异系数 (CV) 衡量不均衡度.
    CV = 0 表示完全均衡, CV > 1 表示严重不均衡.
    """
    if not device_loads:
        return 0.0
    
    import math
    loads = list(device_loads.values())
    mean_load = sum(loads) / len(loads)
    if mean_load == 0:
        return 0.0
    
    var_load = sum((l - mean_load) ** 2 for l in loads) / len(loads)
    cv = math.sqrt(var_load) / mean_load
    
    _dbg("BALANCE", f"loads={device_loads} mean={mean_load:.2f} cv={cv:.4f}")
    return cv


def _ema_update(old_value: float, new_sample: float, alpha: float = 0.1) -> float:
    """EMA 更新 — 改编自 Megatron DistributedOptimizer (distrib_optimizer.py:102).
    
    原版用于梯度平滑; 移植版用于设备负载和误差的指数平滑.
    """
    result = alpha * new_sample + (1.0 - alpha) * old_value
    _dbg("EMA", f"old={old_value:.4f} new={new_sample:.4f} "
         f"alpha={alpha} → {result:.4f}")
    return result


def _dump_adaptive_state(strategy, label=""):
    """打印自适应策略全状态快照."""
    import sys
    print(f"╔══ AdaptiveStrategy [{label}] ══════════════", file=sys.stderr)
    for attr in ['_ema_alpha', '_device_load', '_bias_ema', 
                 '_history_window', '_recent_costs']:
        val = getattr(strategy, attr, 'N/A')
        if isinstance(val, dict):
            val = {k: f"{v:.4f}" if isinstance(v, float) else v 
                   for k, v in val.items()}
        elif isinstance(val, list) and len(val) > 5:
            val = f"[{val[0]:.3f}, ..., {val[-1]:.3f}] (len={len(val)})"
        print(f"║ {attr}: {val}", file=sys.stderr)
    print(f"╚═══════════════════════════════════════════", file=sys.stderr, flush=True)
