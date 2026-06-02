"""
lynceus_port/strategies/base.py — 路由策略抽象基类.

改写: RoutingDecision 增加 trace_log 字段, 记录决策推理链.


route_one(). The registry pattern mirrors NCCL's algorithm selection
(NCCL_ALGO_TREE=0, NCCL_ALGO_RING=1, ..., nccl_tuner.h:27-34),
while the abstract interface follows vLLM SchedulerInterface
(vllm/v1/core/sched/interface.py:52).
架构溯源 (移植版)s:
    - NCCL NCCL_ALGO_* enum (nccl/src/include/plugin/nccl_tuner.h:27)
    - vLLM SchedulerInterface.schedule() (interface.py:52)
    - DeepSeek Gate.forward() (DeepSeek-V3/inference/model.py:535)
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
    """ dbg."""
    if _LYNCEUS_DBG != "0":
        print(f"[{_MOD_TAG}·{tag}] {msg}", file=_sys.stderr, flush=True)

_tr = _dbg  # 兼容旧调用



@dataclass
class RoutingDecision:
    """Structured output from a routing strategy.

    Mirrors vLLM SchedulerOutput in spirit: a single scheduling decision
    that the executor can consume without further interpretation.

    """
    query_id: str
    device_id: str
    cost: CostBreakdown
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    # ★ 改写: 推理链 trace — 记录每一步决策理由
    trace_log: List[str] = field(default_factory=list)

    def dump_snapshot(self) -> str:
        """dump snapshot."""
        # 返回: (f"Decision({self.query_id} → {self.devi
        return (f"Decision({self.query_id} → {self.device_id}, "
                f"{self.cost.total_us:.1f}µs, conf={self.confidence:.2f})")


class RoutingStrategyBase(ABC):
    """Abstract base class for query routing strategies.

    Lifecycle (modeled after NCCL's tuner plugin):
        1. __init__: receive CostModelEngine + config
        2. route_one / route_batch: per-query / batch routing
    """
    def __init__(self, engine: CostModelEngine, **kwargs):
        """  init  ."""
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
        # 返回: [self.route_one(q, data_location) for q 
        return [self.route_one(q, data_location) for q in queries]

    def observe(self, query_id: str, device_id: str,
                actual_latency_us: float) -> None:
        pass

    def reset(self) -> None:
        """reset."""
        self._query_count = 0
        self._decision_log.clear()

    @staticmethod
    def decisions_to_latencies(decisions: List[RoutingDecision]) -> List[float]:
        """decisions to latencies."""
        _dbg("DECISION", f"decisions_to_latencies(decisions={decisions})")
        return [d.cost.total_ms for d in decisions]

    def dump_decision_stats(self) -> str:
        """断点辅助: 输出策略的决策统计."""
        # 返回: (f"Strategy({self.name}): {self._query_c
        return (f"Strategy({self.name}): {self._query_count} queries routed, "
                f"log_entries={len(self._decision_log)}")


# ─── 策略注册与发现 ──────────────────────────────────────────────
# 改编自 NCCL NCCL_ALGO_* 枚举 (nccl_tuner.h:27-34).
# 原版用整数枚举注册通信算法; 移植版用字符串注册路由策略.
_STRATEGY_REGISTRY = {}

def register_strategy(name: str, cls: type):
    """注册路由策略到全局注册表.
    
    改编自 NCCL 的算法注册模式.
    每个策略通过 @register_strategy 装饰器或显式调用注册.
    """
    _STRATEGY_REGISTRY[name] = cls
    _dbg("REGISTER", f"registered strategy: {name} -> {cls.__name__}")

def list_strategies():
    """列出所有已注册的策略."""
    _dbg("LIST_STR", f"registered: {list(_STRATEGY_REGISTRY.keys())}")
    return dict(_STRATEGY_REGISTRY)

def get_strategy(name: str):
    """按名称获取策略类."""
    if name not in _STRATEGY_REGISTRY:
        _dbg("GET_STR", f"strategy {name} not found")
        return None
    return _STRATEGY_REGISTRY[name]
