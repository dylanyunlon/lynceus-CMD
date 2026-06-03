"""
lynceus/strategies/base.py — Abstract base for routing strategies.

Every routing strategy inherits from RoutingStrategyBase and implements
route_one(). The registry pattern mirrors NCCL's algorithm selection
(NCCL_ALGO_TREE=0, NCCL_ALGO_RING=1, ..., nccl_tuner.h:27-34),
while the abstract interface follows vLLM SchedulerInterface
(vllm/v1/core/sched/interface.py:52).

Architecture references:
    - NCCL NCCL_ALGO_* enum (nccl/src/include/plugin/nccl_tuner.h:27)
      → each algorithm is an independent implementation of a common API
    - vLLM SchedulerInterface.schedule() (interface.py:52)
      → returns SchedulerOutput; we return RoutingDecision
    - DeepSeek Gate.forward() (DeepSeek-V3/inference/model.py:535)
      → returns (weights, indices); we return (device_id, cost)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from ..cost_model import (
    CostBreakdown,
    CostModelEngine,
    QueryDescriptor,
)
from ..schema import HardwareKind


# ---------------------------------------------------------------------------
# Routing decision (analogous to vLLM SchedulerOutput)
# ---------------------------------------------------------------------------

@dataclass
class RoutingDecision:
    """Structured output from a routing strategy.

    Mirrors vLLM SchedulerOutput in spirit: a single scheduling decision
    that the executor can consume without further interpretation.

    Attributes:
        query_id:     Which query this decision is for.
        device_id:    Where to execute it.
        cost:         Estimated cost breakdown.
        confidence:   Routing confidence in [0, 1]. Adaptive strategies
                      may have lower confidence during warm-up.
        metadata:     Strategy-specific metadata (e.g. reason, alternatives).
    """
    query_id: str
    device_id: str
    cost: CostBreakdown
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract strategy
# ---------------------------------------------------------------------------

class RoutingStrategyBase(ABC):
    """Abstract base class for query routing strategies.

    Lifecycle (modeled after NCCL's tuner plugin):
        1. __init__: receive CostModelEngine + config
        2. route_one / route_batch: per-query / batch routing
        3. observe: optional feedback from actual execution
        4. reset: clear internal state

    Subclasses MUST implement:
        - name (property): unique strategy identifier
        - route_one: single-query routing decision

    Subclasses MAY override:
        - route_batch: batch routing (default: loop over route_one)
        - observe: feedback from actual execution
        - reset: clear internal state
    """

    def __init__(self, engine: CostModelEngine, **kwargs):
        self._engine = engine
        self._query_count = 0

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this strategy (e.g. 'GPU-Only').

        Must match a RoutingStrategy enum value for benchmark compatibility.
        """
        ...

    @abstractmethod
    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        """Route a single query to a device.

        This is the core method — analogous to DeepSeek Gate.forward()
        returning (weights, indices) for one token.
        """
        ...

    def route_batch(self, queries: List[QueryDescriptor],
                    data_location: Optional[str] = None
                    ) -> List[RoutingDecision]:
        """Route a batch of queries. Default: sequential route_one.

        Override for batch-level optimizations (e.g. load balancing
        across GPUs, like DeepSpeed TopKGate's capacity_factor).
        """
        return [self.route_one(q, data_location) for q in queries]

    def observe(self, query_id: str, device_id: str,
                actual_latency_us: float) -> None:
        """Feedback from actual execution.

        Adaptive strategies use this to update their internal model.
        Default: no-op.
        """
        pass

    def reset(self) -> None:
        """Clear internal state (e.g. EMA accumulators, counters)."""
        self._query_count = 0

    # Convenience: extract latencies from decisions
    @staticmethod
    def decisions_to_latencies(decisions: List[RoutingDecision]) -> List[float]:
        """Extract per-decision latency in ms."""
        return [d.cost.total_ms for d in decisions]
