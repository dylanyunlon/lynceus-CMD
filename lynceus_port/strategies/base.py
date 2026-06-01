"""
lynceus_port/strategies/base.py — 路由策略抽象基类.

改写: RoutingDecision 增加 trace_log 字段, 记录决策推理链.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from ..cost_model import CostBreakdown, CostModelEngine, QueryDescriptor
from ..schema import HardwareKind
from .. import _dbg

_MOD_TAG = "BAE"
import os as _os, sys as _sys
_LYNCEUS_DBG = _os.environ.get("LYNCEUS_DEBUG", "1")

def _dbg(tag: str, msg: str):
    if _LYNCEUS_DBG != "0":
        print(f"[{_MOD_TAG}·{tag}] {msg}", file=_sys.stderr, flush=True)

_tr = _dbg  # 兼容旧调用



@dataclass
class RoutingDecision:
    query_id: str
    device_id: str
    cost: CostBreakdown
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    # ★ 改写: 推理链 trace — 记录每一步决策理由
    trace_log: List[str] = field(default_factory=list)

    def dump_snapshot(self) -> str:
        return (f"Decision({self.query_id} → {self.device_id}, "
                f"{self.cost.total_us:.1f}µs, conf={self.confidence:.2f})")


class RoutingStrategyBase(ABC):
    def __init__(self, engine: CostModelEngine, **kwargs):
        self._engine = engine
        self._query_count = 0
        self._decision_log: List[str] = []  # ★ 改写: 策略级日志

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
        self._decision_log.clear()

    @staticmethod
    def decisions_to_latencies(decisions: List[RoutingDecision]) -> List[float]:
        _dbg("DECISION", f"decisions_to_latencies(decisions={decisions})")
        return [d.cost.total_ms for d in decisions]

    def dump_decision_stats(self) -> str:
        """断点辅助: 输出策略的决策统计."""
        return (f"Strategy({self.name}): {self._query_count} queries routed, "
                f"log_entries={len(self._decision_log)}")
