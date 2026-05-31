"""
lynceus_port/strategies/cost_driven.py — 代价模型驱动的路由策略。

移植自 lynceus/strategies/cost_driven.py，修改约20%:
  - PAR2QOEnhancedStrategy -> PAR2QORobustStrategy
  - route_one: 打印完整候选设备列表及代价
  - CostModelRoutedStrategy: route_batch 记录批次摘要
"""

from __future__ import annotations
from typing import Dict, List, Optional

from ..cost_model import CostBreakdown, CostModelEngine, QueryDescriptor
from ..schema import HardwareKind, _dbg
from .base import RoutingDecision, RoutingStrategyBase


class CostModelRoutedStrategy(RoutingStrategyBase):
    """纯最小代价路由"""

    @property
    def name(self) -> str:
        return "cost_model_dispatch"

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        self._query_count += 1
        device_id, cb = self._engine.recommend(query, data_location)
        _dbg("CostRouted",
             f"#{self._query_count} {query.query_id} -> {device_id} "
             f"({cb.total_us:.1f}us)")
        return RoutingDecision(
            query_id=query.query_id, device_id=device_id,
            cost=cb, confidence=cb.confidence,
            metadata={"reason": "min_cost"},
        )

    def route_batch(self, queries: List[QueryDescriptor],
                    data_location: Optional[str] = None
                    ) -> List[RoutingDecision]:
        results = self._engine.route_batch(queries, data_location)
        decisions = [
            RoutingDecision(
                query_id=q.query_id, device_id=dev,
                cost=cb, confidence=cb.confidence,
                metadata={"reason": "min_cost"},
            )
            for q, (dev, cb) in zip(queries, results)
        ]
        # ── 批次摘要 ──
        gpu_count = sum(1 for d in decisions if "gpu" in d.device_id)
        cpu_count = len(decisions) - gpu_count
        avg_cost = sum(d.cost.total_us for d in decisions) / max(1, len(decisions))
        _dbg("CostRouted",
             f"batch: {len(decisions)} queries, "
             f"gpu={gpu_count} cpu={cpu_count} avg_cost={avg_cost:.1f}us")
        return decisions


class PAR2QORobustStrategy(RoutingStrategyBase):
    """代价路由 + PAR2QO 鲁棒性惩罚。

    修改点：打印完整候选设备对比表，方便调试时看清决策过程。
    """

    def __init__(self, engine: CostModelEngine, *,
                 robustness_margin: float = 0.20, **kwargs):
        super().__init__(engine, **kwargs)
        self._margin = robustness_margin

    @property
    def name(self) -> str:
        return "par2qo_penalty_aware"

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        self._query_count += 1
        estimates = self._engine.estimate_all_devices(query, data_location)

        gpu_ids = [k for k, n in self._engine.topology.nodes.items()
                   if n.kind == HardwareKind.GPU and k in estimates]
        cpu_ids = [k for k, n in self._engine.topology.nodes.items()
                   if n.kind == HardwareKind.CPU and k in estimates]

        # ── 调试：打印所有候选 ──
        for dev_id, cb in sorted(estimates.items(),
                                  key=lambda kv: kv[1].total_us):
            _dbg("PAR2QO",
                 f"  candidate {dev_id}: {cb.total_us:.1f}us "
                 f"(io={cb.io_cost_us:.1f} comp={cb.compute_cost_us:.1f} "
                 f"xfer={cb.transfer_cost_us:.1f})")

        if not gpu_ids or not cpu_ids:
            device_id, cb = self._engine.recommend(query, data_location)
            return RoutingDecision(
                query_id=query.query_id, device_id=device_id,
                cost=cb, confidence=0.5,
                metadata={"reason": "fallback_incomplete_topology"},
            )

        best_gpu = min(gpu_ids, key=lambda k: estimates[k].total_us)
        best_cpu = min(cpu_ids, key=lambda k: estimates[k].total_us)
        gpu_cost = estimates[best_gpu].total_us
        cpu_cost = estimates[best_cpu].total_us

        if gpu_cost < cpu_cost * (1.0 - self._margin):
            chosen = best_gpu
            reason = "gpu_definite_winner"
        else:
            chosen = best_cpu
            reason = "cpu_robust_fallback"

        adv_pct = (1.0 - gpu_cost / cpu_cost) * 100 if cpu_cost > 0 else 0.0
        _dbg("PAR2QO",
             f"#{self._query_count} {query.query_id}: "
             f"gpu={gpu_cost:.1f} cpu={cpu_cost:.1f} "
             f"adv={adv_pct:.1f}% margin={self._margin*100:.0f}% "
             f"-> {chosen} ({reason})")

        return RoutingDecision(
            query_id=query.query_id, device_id=chosen,
            cost=estimates[chosen],
            confidence=1.0 if reason == "gpu_definite_winner" else 0.8,
            metadata={
                "reason": reason,
                "gpu_cost_us": gpu_cost,
                "cpu_cost_us": cpu_cost,
                "margin": self._margin,
                "gpu_advantage_pct": adv_pct,
            },
        )
