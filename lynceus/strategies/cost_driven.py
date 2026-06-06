"""lynceus/strategies/cost_driven.py — Cost-model-driven routing with regret minimization.

算法改动 (Claude #7, M311-M320):
    1. PAR2QO route_one: 简单阈值比较 → Follow the Perturbed Leader (FPL)
       — 对每个device维护累积loss, 加指数噪声扰动后选loss最小的。
       — 理论: Kalai & Vempala 2005, O(√T)regret bound
    2. observe_gpu_bias: 简单EMA → Hedge算法 (Multiplicative Weights Update)
       — 指数权重更新 w_i *= exp(-η·loss_i), 自动集中在低loss device上。
       — 理论: Freund & Schapire 1997, regret ≤ (ln N)/η + η·T
    3. _dbg_regret_trace: 打印累积regret和各device权重变化
"""
from __future__ import annotations
import math
import random as _random
from collections import defaultdict
from typing import Dict, List, Optional
from ..cost_model import CostBreakdown, CostModelEngine, QueryDescriptor
from ..schema import HardwareKind
from .base import RoutingDecision, RoutingStrategyBase


def _dbg_regret_trace(step: int, cum_loss: Dict[str, float],
                      hedge_weights: Dict[str, float],
                      best_fixed_loss: float, actual_cum_loss: float):
    """打印FPL/Hedge的regret追踪"""
    from .._debug import dbg
    regret = actual_cum_loss - best_fixed_loss
    dbg('regret_trace',
        step=step,
        cumulative_losses={k: f"{v:.1f}" for k, v in cum_loss.items()},
        hedge_weights={k: f"{v:.4f}" for k, v in hedge_weights.items()},
        best_fixed_loss=f"{best_fixed_loss:.1f}",
        actual_cum_loss=f"{actual_cum_loss:.1f}",
        regret=f"{regret:.1f}")


class CostModelRoutedStrategy(RoutingStrategyBase):
    @property
    def name(self) -> str:
        return "CostModel-Routed"

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        device_id, cb = self._engine.recommend(query, data_location)
        return RoutingDecision(query_id=query.query_id, device_id=device_id,
            cost=cb, confidence=1.0, metadata={"reason": "min_cost"})

    def route_batch(self, queries: List[QueryDescriptor],
                    data_location: Optional[str] = None) -> List[RoutingDecision]:
        results = self._engine.route_batch(queries, data_location)
        return [RoutingDecision(query_id=q.query_id, device_id=dev,
            cost=cb, confidence=1.0, metadata={"reason": "min_cost"})
            for q, (dev, cb) in zip(queries, results)]


class PAR2QOEnhancedStrategy(RoutingStrategyBase):
    """改动: FPL (Follow the Perturbed Leader) + Hedge算法。
    FPL: 对每个device的累积loss加指数扰动, 选扰动后loss最小的。
    Hedge: 维护multiplicative weights, 指数衰减高loss的device。"""

    def __init__(self, engine: CostModelEngine, *, base_margin: float = 0.18,
                 fpl_eta: float = 0.01, hedge_eta: float = 0.05, **kwargs):
        super().__init__(engine, **kwargs)
        self._base_margin = base_margin
        self._fpl_eta = fpl_eta        # FPL扰动尺度
        self._hedge_eta = hedge_eta    # Hedge学习率
        # FPL状态
        self._cum_loss: Dict[str, float] = defaultdict(float)
        self._fpl_step = 0
        # Hedge状态
        self._hedge_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        self._actual_cum_loss = 0.0  # 实际选择的device的累积loss

    @property
    def name(self) -> str:
        return "PAR2QO-Enhanced"

    def _fpl_select(self, estimates: Dict[str, CostBreakdown]) -> str:
        """Follow the Perturbed Leader: 对累积loss加指数噪声, 选最小。
        扰动分布: Exp(1/η), 使得在线后悔 O(√(T·ln N))"""
        perturbed_loss: Dict[str, float] = {}
        for dev_id, cb in estimates.items():
            noise = _random.expovariate(1.0 / max(1e-6, self._fpl_eta))
            perturbed_loss[dev_id] = self._cum_loss[dev_id] + cb.total_us - noise
        return min(perturbed_loss, key=perturbed_loss.get)

    def _hedge_weight_for(self, device_id: str) -> float:
        """返回Hedge算法下device的归一化权重"""
        total = sum(self._hedge_weights.values())
        if total <= 0:
            return 1.0 / max(1, len(self._hedge_weights))
        return self._hedge_weights[device_id] / total

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        from .._debug import checkpoint
        estimates = self._engine.estimate_all_devices(query, data_location)

        gpu_ids = [k for k, n in self._engine.topology.nodes.items()
                   if n.kind == HardwareKind.GPU and k in estimates]
        cpu_ids = [k for k, n in self._engine.topology.nodes.items()
                   if n.kind == HardwareKind.CPU and k in estimates]

        if not gpu_ids or not cpu_ids:
            device_id, cb = self._engine.recommend(query, data_location)
            return RoutingDecision(query_id=query.query_id, device_id=device_id,
                cost=cb, confidence=0.5,
                metadata={"reason": "fallback_incomplete_topology"})

        # FPL选择: 在所有device中选扰动后loss最小的
        fpl_chosen = self._fpl_select(estimates)

        # Hedge修正: 用hedge权重调整FPL选择
        # 如果hedge权重表示FPL选的device权重太低, 转向权重最高的
        hedge_dev = max(self._hedge_weights, key=self._hedge_weights.get,
                        default=fpl_chosen)

        # 最终决策: FPL主导, Hedge作为安全阀
        hedge_w = self._hedge_weight_for(fpl_chosen) if fpl_chosen in self._hedge_weights else 0.5
        if hedge_w < 0.1 and hedge_dev in estimates:
            # FPL选的device在Hedge中权重过低, 用Hedge的推荐
            chosen = hedge_dev
            reason = "hedge_safety_valve"
        else:
            chosen = fpl_chosen
            reason = "fpl_leader"

        self._fpl_step += 1

        checkpoint("fpl_decision",
                   fpl_chosen=fpl_chosen, hedge_recommended=hedge_dev,
                   final=chosen, fpl_step=self._fpl_step,
                   hedge_weight=f"{hedge_w:.4f}")

        return RoutingDecision(
            query_id=query.query_id, device_id=chosen,
            cost=estimates[chosen],
            confidence=min(1.0, hedge_w * 2.0),
            metadata={"reason": reason,
                      "fpl_step": self._fpl_step,
                      "hedge_weight": hedge_w,
                      "cum_loss": dict(self._cum_loss)})

    def observe(self, query_id: str, device_id: str,
                actual_latency_us: float) -> None:
        """Hedge更新: w_i *= exp(-η·loss_i), 然后归一化。
        loss用actual latency作为代理。"""
        # 更新累积loss
        self._cum_loss[device_id] += actual_latency_us
        self._actual_cum_loss += actual_latency_us

        # Hedge multiplicative weights update
        # 所有device都更新: 被选中的用actual loss, 未选中的用0
        for dev in list(self._hedge_weights.keys()):
            if dev == device_id:
                loss = actual_latency_us / 1e6  # 归一化到秒级
                self._hedge_weights[dev] *= math.exp(-self._hedge_eta * loss)
            # 防止权重下溢
            self._hedge_weights[dev] = max(1e-10, self._hedge_weights[dev])

        # 确保选中的device在权重表中
        if device_id not in self._hedge_weights:
            self._hedge_weights[device_id] = 1.0

        # regret trace
        best_fixed = min(self._cum_loss.values()) if self._cum_loss else 0.0
        _dbg_regret_trace(
            self._fpl_step,
            dict(self._cum_loss),
            dict(self._hedge_weights),
            best_fixed,
            self._actual_cum_loss)

    def observe_gpu_bias(self, estimated_us: float, actual_us: float):
        """用Hedge思想更新GPU相关device的权重。
        如果GPU预估偏差大, 降低所有GPU device的Hedge权重。"""
        if estimated_us <= 0:
            return
        bias_ratio = actual_us / estimated_us
        # bias > 1 说明GPU实际比预估慢 → 惩罚GPU devices
        if bias_ratio > 1.2:
            penalty = self._hedge_eta * (bias_ratio - 1.0)
            for dev in list(self._hedge_weights.keys()):
                if 'gpu' in dev.lower():
                    self._hedge_weights[dev] *= math.exp(-penalty)
                    self._hedge_weights[dev] = max(1e-10, self._hedge_weights[dev])
        elif bias_ratio < 0.8:
            # GPU实际比预估快 → 奖励
            bonus = self._hedge_eta * (1.0 - bias_ratio)
            for dev in list(self._hedge_weights.keys()):
                if 'gpu' in dev.lower():
                    self._hedge_weights[dev] *= math.exp(bonus)
