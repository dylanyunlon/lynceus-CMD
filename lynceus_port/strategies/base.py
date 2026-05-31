"""
lynceus_port/strategies/base.py — 路由策略抽象基类。

移植自 lynceus/strategies/base.py，修改约20%:
  - RoutingDecision: 新增 wall_time_us 字段
  - RoutingDecision: 新增 to_dict() 序列化
  - RoutingStrategyBase: 新增 debug_snapshot()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from ..cost_model import CostBreakdown, CostModelEngine, QueryDescriptor
from ..schema import HardwareKind, _dbg


@dataclass
class RoutingDecision:
    """路由决策——策略的结构化输出"""
    query_id: str
    device_id: str
    cost: CostBreakdown
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    # ── 新增：实际路由决策耗时 ──
    wall_time_us: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "device_id": self.device_id,
            "total_us": self.cost.total_us,
            "confidence": self.confidence,
            "wall_time_us": self.wall_time_us,
            "metadata": self.metadata,
        }

    def debug_snapshot(self) -> str:
        s = (f"Decision({self.query_id} -> {self.device_id}, "
             f"cost={self.cost.total_us:.1f}us, conf={self.confidence:.2f}, "
             f"reason={self.metadata.get('reason', '?')})")
        _dbg("Decision", s)
        return s


class RoutingStrategyBase(ABC):
    """路由策略抽象基类"""

    def __init__(self, engine: CostModelEngine, **kwargs):
        self._engine = engine
        self._query_count = 0

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision: ...

    def route_batch(self, queries: List[QueryDescriptor],
                    data_location: Optional[str] = None
                    ) -> List[RoutingDecision]:
        return [self.route_one(q, data_location) for q in queries]

    def observe(self, query_id: str, device_id: str,
                actual_latency_us: float) -> None:
        pass

    def reset(self) -> None:
        self._query_count = 0
        _dbg("Strategy", f"{self.name}: reset")

    @staticmethod
    def decisions_to_latencies(decisions: List[RoutingDecision]) -> List[float]:
        return [d.cost.total_ms for d in decisions]

    def debug_snapshot(self) -> str:
        s = f"Strategy({self.name}, queries_seen={self._query_count})"
        _dbg("Strategy", s)
        return s
