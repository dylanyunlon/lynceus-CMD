"""
lynceus/strategies/static.py — Static (non-adaptive) routing strategies.

These are the baseline strategies that don't learn from past queries:
    - GPUOnlyStrategy: all queries to gpu0
    - CPUOnlyStrategy: all queries to cpu0
    - HybridStaticStrategy: threshold-based GPU/CPU split

Architecture references:
    - NCCL NCCL_ALGO_RING (nccl_tuner.h:29) — a fixed algorithm choice
    - Megatron get_tensor_model_parallel_group (parallel_state.py:1449)
      → static group assignment, no runtime adaptation
"""

from __future__ import annotations

from typing import Optional

from ..cost_model import CostModelEngine, QueryDescriptor
from ..schema import HardwareKind
from .base import RoutingDecision, RoutingStrategyBase


class GPUOnlyStrategy(RoutingStrategyBase):
    """Route every query to gpu0, regardless of cost.

    Baseline for measuring GPU overhead on small queries.
    """

    def __init__(self, engine: CostModelEngine, *,
                 gpu_id: str = "gpu0", **kwargs):
        super().__init__(engine, **kwargs)
        self._gpu_id = gpu_id

    @property
    def name(self) -> str:
        return "GPU-Only"

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        cb = self._engine.estimate_on_device(query, self._gpu_id, data_location)
        return RoutingDecision(
            query_id=query.query_id,
            device_id=self._gpu_id,
            cost=cb,
            confidence=1.0,
            metadata={"reason": "fixed_gpu"},
        )


class CPUOnlyStrategy(RoutingStrategyBase):
    """Route every query to cpu0.

    Baseline for measuring CPU performance without GPU acceleration.
    """

    def __init__(self, engine: CostModelEngine, *,
                 cpu_id: str = "cpu0", **kwargs):
        super().__init__(engine, **kwargs)
        self._cpu_id = cpu_id

    @property
    def name(self) -> str:
        return "CPU-Only"

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        cb = self._engine.estimate_on_device(query, self._cpu_id, data_location)
        return RoutingDecision(
            query_id=query.query_id,
            device_id=self._cpu_id,
            cost=cb,
            confidence=1.0,
            metadata={"reason": "fixed_cpu"},
        )


class HybridStaticStrategy(RoutingStrategyBase):
    """Threshold-based routing: large queries → GPU, small → CPU.

    The threshold (estimated_rows > gpu_threshold_rows) is a static
    configuration parameter, not learned. This models a common production
    heuristic: "anything touching more than 100K rows goes to the GPU."
    """

    def __init__(self, engine: CostModelEngine, *,
                 gpu_threshold_rows: int = 100_000,
                 gpu_id: str = "gpu0",
                 cpu_id: str = "cpu0", **kwargs):
        super().__init__(engine, **kwargs)
        self._threshold = gpu_threshold_rows
        self._gpu_id = gpu_id
        self._cpu_id = cpu_id

    @property
    def name(self) -> str:
        return "Hybrid-Static"

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        if query.estimated_rows > self._threshold:
            device = self._gpu_id
            reason = "rows_above_threshold"
        else:
            device = self._cpu_id
            reason = "rows_below_threshold"

        cb = self._engine.estimate_on_device(query, device, data_location)
        return RoutingDecision(
            query_id=query.query_id,
            device_id=device,
            cost=cb,
            confidence=1.0,
            metadata={
                "reason": reason,
                "threshold": self._threshold,
                "estimated_rows": query.estimated_rows,
            },
        )
