"""
lynceus_port/strategies/static.py — 静态路由策略（基线）。

移植自 lynceus/strategies/static.py，修改约20%:
  - 每次 route_one 自动打印调试信息
  - HybridStaticStrategy 的阈值判定加入 selectivity 二次校验
"""

from __future__ import annotations
from typing import Optional

from ..cost_model import CostModelEngine, QueryDescriptor
from ..schema import HardwareKind, _dbg
from .base import RoutingDecision, RoutingStrategyBase


class GPUOnlyStrategy(RoutingStrategyBase):
    """所有查询强制发往 GPU"""

    def __init__(self, engine: CostModelEngine, *,
                 gpu_id: str = "gpu0", **kwargs):
        super().__init__(engine, **kwargs)
        self._gpu_id = gpu_id

    @property
    def name(self) -> str:
        return "gpu_exclusive"

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        self._query_count += 1
        cb = self._engine.estimate_on_device(query, self._gpu_id, data_location)
        _dbg("GPUOnly", f"#{self._query_count} {query.query_id} -> {self._gpu_id}")
        return RoutingDecision(
            query_id=query.query_id, device_id=self._gpu_id,
            cost=cb, confidence=1.0,
            metadata={"reason": "forced_gpu"},
        )


class CPUOnlyStrategy(RoutingStrategyBase):
    """所有查询强制发往 CPU"""

    def __init__(self, engine: CostModelEngine, *,
                 cpu_id: str = "cpu0", **kwargs):
        super().__init__(engine, **kwargs)
        self._cpu_id = cpu_id

    @property
    def name(self) -> str:
        return "cpu_exclusive"

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        self._query_count += 1
        cb = self._engine.estimate_on_device(query, self._cpu_id, data_location)
        _dbg("CPUOnly", f"#{self._query_count} {query.query_id} -> {self._cpu_id}")
        return RoutingDecision(
            query_id=query.query_id, device_id=self._cpu_id,
            cost=cb, confidence=1.0,
            metadata={"reason": "forced_cpu"},
        )


class HybridStaticStrategy(RoutingStrategyBase):
    """基于阈值的混合路由：大查询→GPU，小查询→CPU

    修改点：当行数超过阈值但 selectivity 极低时（<0.01），
    仍保留在 CPU——因为高选择性意味着实际扫描量很小。
    """

    def __init__(self, engine: CostModelEngine, *,
                 gpu_threshold_rows: int = 100_000,
                 selectivity_override: float = 0.01,
                 gpu_id: str = "gpu0",
                 cpu_id: str = "cpu0", **kwargs):
        super().__init__(engine, **kwargs)
        self._threshold = gpu_threshold_rows
        self._sel_override = selectivity_override
        self._gpu_id = gpu_id
        self._cpu_id = cpu_id

    @property
    def name(self) -> str:
        return "hybrid_threshold"

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        self._query_count += 1

        # ── 修改：selectivity 二次校验 ──
        effective_rows = query.estimated_rows * query.selectivity
        if (query.estimated_rows > self._threshold and
                query.selectivity >= self._sel_override):
            device = self._gpu_id
            reason = "rows_above_threshold"
        else:
            device = self._cpu_id
            reason = ("selectivity_override"
                      if query.estimated_rows > self._threshold
                      else "rows_below_threshold")

        cb = self._engine.estimate_on_device(query, device, data_location)
        _dbg("Hybrid",
             f"#{self._query_count} rows={query.estimated_rows} "
             f"sel={query.selectivity:.3f} eff={effective_rows:.0f} -> {device} "
             f"({reason})")
        return RoutingDecision(
            query_id=query.query_id, device_id=device,
            cost=cb, confidence=1.0,
            metadata={
                "reason": reason,
                "threshold": self._threshold,
                "estimated_rows": query.estimated_rows,
                "effective_rows": effective_rows,
            },
        )
