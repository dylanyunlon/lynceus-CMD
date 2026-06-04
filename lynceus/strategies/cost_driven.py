"""lynceus/strategies/cost_driven.py — Cost-model-driven routing."""
from __future__ import annotations
import math
from typing import Dict, List, Optional
from ..cost_model import CostBreakdown, CostModelEngine, QueryDescriptor
from ..schema import HardwareKind
from .base import RoutingDecision, RoutingStrategyBase

class CostModelRoutedStrategy(RoutingStrategyBase):
    @property
    def name(self) -> str:
        return "CostModel-Routed"
    def route_one(self, query: QueryDescriptor, data_location: Optional[str] = None) -> RoutingDecision:
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
    """改动: margin 根据累积 GPU 预估偏差在线调节。"""
    def __init__(self, engine: CostModelEngine, *, base_margin: float = 0.18,
                 margin_adapt_rate: float = 0.08, **kwargs):
        super().__init__(engine, **kwargs)
        self._base_margin = base_margin
        self._margin = base_margin
        self._adapt_rate = margin_adapt_rate
        self._gpu_bias_ema = 0.0
    @property
    def name(self) -> str:
        return "PAR2QO-Enhanced"
    def route_one(self, query: QueryDescriptor, data_location: Optional[str] = None) -> RoutingDecision:
        from .._debug import checkpoint
        estimates = self._engine.estimate_all_devices(query, data_location)
        gpu_ids = [k for k, n in self._engine.topology.nodes.items()
                   if n.kind == HardwareKind.GPU and k in estimates]
        cpu_ids = [k for k, n in self._engine.topology.nodes.items()
                   if n.kind == HardwareKind.CPU and k in estimates]
        if not gpu_ids or not cpu_ids:
            device_id, cb = self._engine.recommend(query, data_location)
            return RoutingDecision(query_id=query.query_id, device_id=device_id,
                cost=cb, confidence=0.5, metadata={"reason": "fallback_incomplete_topology"})
        best_gpu = min(gpu_ids, key=lambda k: estimates[k].total_us)
        best_cpu = min(cpu_ids, key=lambda k: estimates[k].total_us)
        gpu_cost = estimates[best_gpu].total_us
        cpu_cost = estimates[best_cpu].total_us
        effective_margin = max(0.02, min(0.5, self._base_margin + self._gpu_bias_ema * 0.3))
        if gpu_cost < cpu_cost * (1.0 - effective_margin):
            chosen, reason = best_gpu, "gpu_clear_winner"
        else:
            chosen, reason = best_cpu, "cpu_robust_choice"
        checkpoint("par2qo_decision", gpu_cost=gpu_cost, cpu_cost=cpu_cost,
                   effective_margin=effective_margin, chosen=chosen)
        return RoutingDecision(query_id=query.query_id, device_id=chosen,
            cost=estimates[chosen],
            confidence=1.0 if reason == "gpu_clear_winner" else 0.8,
            metadata={"reason": reason, "gpu_cost_us": gpu_cost, "cpu_cost_us": cpu_cost,
                      "effective_margin": effective_margin,
                      "gpu_advantage_pct": (1.0 - gpu_cost / cpu_cost) * 100 if cpu_cost > 0 else 0.0})
    def observe(self, query_id: str, device_id: str, actual_latency_us: float) -> None:
        pass
    def observe_gpu_bias(self, estimated_us: float, actual_us: float):
        if estimated_us <= 0:
            return
        bias = (actual_us - estimated_us) / estimated_us
        self._gpu_bias_ema = self._adapt_rate * bias + (1.0 - self._adapt_rate) * self._gpu_bias_ema
        self._gpu_bias_ema = max(-0.5, min(0.5, self._gpu_bias_ema))
