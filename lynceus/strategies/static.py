"""lynceus/strategies/static.py — Static routing strategies."""
from __future__ import annotations
import math
from typing import Optional
from ..costing import CostModelEngine, QueryDescriptor
from ..schema import HardwareKind
from .foundation import RoutingDecision, RoutingStrategyBase

class GPUOnlyStrategy(RoutingStrategyBase):
    def __init__(self, engine: CostModelEngine, *, gpu_id: str = "gpu0", **kwargs):
        super().__init__(engine, **kwargs)
        self._gpu_id = gpu_id
    @property
    def name(self) -> str:
        return "GPU-Only"
    def route_one(self, query: QueryDescriptor, data_location: Optional[str] = None) -> RoutingDecision:
        cb = self._engine.estimate_on_device(query, self._gpu_id, data_location)
        return RoutingDecision(query_id=query.query_id, device_id=self._gpu_id,
            cost=cb, confidence=1.0, metadata={"reason": "fixed_gpu"})

class CPUOnlyStrategy(RoutingStrategyBase):
    def __init__(self, engine: CostModelEngine, *, cpu_id: str = "cpu0", **kwargs):
        super().__init__(engine, **kwargs)
        self._cpu_id = cpu_id
    @property
    def name(self) -> str:
        return "CPU-Only"
    def route_one(self, query: QueryDescriptor, data_location: Optional[str] = None) -> RoutingDecision:
        cb = self._engine.estimate_on_device(query, self._cpu_id, data_location)
        return RoutingDecision(query_id=query.query_id, device_id=self._cpu_id,
            cost=cb, confidence=1.0, metadata={"reason": "fixed_cpu"})

class HybridStaticStrategy(RoutingStrategyBase):
    """改动: sigmoid soft threshold 代替硬阈值, 阈值附近平滑过渡。"""
    def __init__(self, engine: CostModelEngine, *, gpu_threshold_rows: int = 90_000,
                 gpu_id: str = "gpu0", cpu_id: str = "cpu0", steepness: float = 2.5, **kwargs):
        super().__init__(engine, **kwargs)
        self._threshold = gpu_threshold_rows
        self._gpu_id = gpu_id
        self._cpu_id = cpu_id
        self._steepness = steepness
        self._log_threshold = math.log(max(1, gpu_threshold_rows))
    @property
    def name(self) -> str:
        return "Hybrid-Static"
    def route_one(self, query: QueryDescriptor, data_location: Optional[str] = None) -> RoutingDecision:
        from .._debug import checkpoint
        log_rows = math.log(max(1, query.estimated_rows))
        z = self._steepness * (log_rows - self._log_threshold)
        gpu_prob = 1.0 / (1.0 + math.exp(-z)) if abs(z) < 20 else (1.0 if z > 0 else 0.0)
        if gpu_prob > 0.5:
            device = self._gpu_id
            reason = "rows_above_soft_threshold"
        else:
            device = self._cpu_id
            reason = "rows_below_soft_threshold"
        cb = self._engine.estimate_on_device(query, device, data_location)
        checkpoint("hybrid_decision", rows=query.estimated_rows, gpu_prob=gpu_prob, device=device)
        return RoutingDecision(query_id=query.query_id, device_id=device,
            cost=cb, confidence=abs(gpu_prob - 0.5) * 2,
            metadata={"reason": reason, "threshold": self._threshold,
                      "gpu_probability": gpu_prob, "estimated_rows": query.estimated_rows})
