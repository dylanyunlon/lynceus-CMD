"""
lynceus/router.py — Strategy registry and query router.

The Router is the single entry point for query routing. It holds a
registry of RoutingStrategyBase implementations (like NCCL's algorithm
table: NCCL_ALGO_TREE, NCCL_ALGO_RING, ..., NCCL_NUM_ALGORITHMS) and
dispatches queries to the active strategy.

Architecture references:
    - NCCL algorithm registry (nccl/src/include/plugin/nccl_tuner.h:27-35)
      → NCCL_ALGO_TREE=0 ... NCCL_ALGO_PAT=6, NCCL_NUM_ALGORITHMS
      → each algorithm is independently registered
    - NCCL tuner_v6 getAlgo (tuner/tuner_v6.h:73)
      → tuner selects algo based on cost table
    - vLLM SchedulerInterface (interface.py:38)
      → single schedule() method as the public API
    - PyTorch c10d all_reduce (distributed_c10d.py:3156)
      → dispatches to backend (NCCL/Gloo) based on tensor placement

Usage:
    router = Router(engine)
    router.register(GPUOnlyStrategy(engine))
    router.register(CostModelRoutedStrategy(engine))
    router.set_active("CostModel-Routed")
    decisions = router.route_batch(queries, data_location="cpu0")
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Type

from .cost_model import CostModelEngine, QueryDescriptor
from .schema import RoutingStrategy
from .strategies.base import RoutingDecision, RoutingStrategyBase
from .strategies.static import (
    CPUOnlyStrategy,
    GPUOnlyStrategy,
    HybridStaticStrategy,
)
from .strategies.cost_driven import (
    CostModelRoutedStrategy,
    PAR2QOEnhancedStrategy,
)
from .strategies.adaptive import AdaptiveStrategy


class Router:
    """Strategy registry and query dispatcher.

    Models NCCL's algorithm selection pattern: all algorithms are
    registered at init time, and the tuner (or user) selects which
    one to use at runtime.

    The registry maps strategy.name → strategy instance. This is
    analogous to NCCL's cost table where each NCCL_ALGO_* has its
    own cost entry (tuner_v6.h:52).
    """

    def __init__(self, engine: CostModelEngine):
        self._engine = engine
        self._registry: Dict[str, RoutingStrategyBase] = {}
        self._active: Optional[RoutingStrategyBase] = None

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    def register(self, strategy: RoutingStrategyBase) -> None:
        """Register a strategy by its name.

        Raises ValueError if a strategy with the same name already exists.
        """
        if strategy.name in self._registry:
            raise ValueError(
                f"Strategy '{strategy.name}' already registered. "
                f"Use replace() to override."
            )
        self._registry[strategy.name] = strategy

    def replace(self, strategy: RoutingStrategyBase) -> None:
        """Register or replace a strategy."""
        self._registry[strategy.name] = strategy

    def unregister(self, name: str) -> None:
        """Remove a strategy from the registry."""
        if name not in self._registry:
            raise KeyError(f"Strategy '{name}' not found in registry")
        if self._active is not None and self._active.name == name:
            self._active = None
        del self._registry[name]

    @property
    def registered_names(self) -> List[str]:
        """List all registered strategy names."""
        return list(self._registry.keys())

    def get(self, name: str) -> RoutingStrategyBase:
        """Get a strategy by name."""
        if name not in self._registry:
            raise KeyError(
                f"Strategy '{name}' not found. "
                f"Available: {self.registered_names}"
            )
        return self._registry[name]

    # ------------------------------------------------------------------
    # Active strategy selection
    # ------------------------------------------------------------------

    def set_active(self, name: str) -> None:
        """Set the active routing strategy.

        Analogous to NCCL tuner selecting algo/proto combination
        (tuner_v6.h:73: "algo: selected algorithm (NCCL_ALGO_*)").
        """
        self._active = self.get(name)

    @property
    def active(self) -> RoutingStrategyBase:
        if self._active is None:
            raise RuntimeError(
                "No active strategy set. Call set_active() first."
            )
        return self._active

    @property
    def active_name(self) -> Optional[str]:
        return self._active.name if self._active else None

    # ------------------------------------------------------------------
    # Routing (delegates to active strategy)
    # ------------------------------------------------------------------

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        """Route a single query using the active strategy."""
        from ._debug import dbg
        dbg('Router.route_one', query_id=query.query_id, strategy=self.active_name)
        return self.active.route_one(query, data_location)

    def route_batch(self, queries: List[QueryDescriptor],
                    data_location: Optional[str] = None
                    ) -> List[RoutingDecision]:
        """Route a batch of queries using the active strategy."""
        return self.active.route_batch(queries, data_location)

    # ------------------------------------------------------------------
    # Factory: create router with all default strategies registered
    # ------------------------------------------------------------------

    @classmethod
    def create_default(cls, engine: CostModelEngine, **kwargs) -> "Router":
        """Create a Router with all built-in strategies registered.

        Like NCCL initializing all NCCL_NUM_ALGORITHMS at startup.
        """
        router = cls(engine)
        router.register(GPUOnlyStrategy(engine, **kwargs))
        router.register(CPUOnlyStrategy(engine, **kwargs))
        router.register(HybridStaticStrategy(engine, **kwargs))
        router.register(CostModelRoutedStrategy(engine, **kwargs))
        router.register(PAR2QOEnhancedStrategy(engine, **kwargs))
        router.register(AdaptiveStrategy(engine, **kwargs))
        return router

    # ------------------------------------------------------------------
    # Benchmark helper: run all strategies on same workload
    # ------------------------------------------------------------------

    # v3: strategies execute in parallel via ThreadPoolExecutor
    def run_all_strategies(
        self,
        queries: List[QueryDescriptor],
        data_location: Optional[str] = None,
        strategy_names: Optional[List[str]] = None,
    ) -> Dict[str, List[RoutingDecision]]:
        """Run every registered strategy on the same query sequence.

        Returns a dict of {strategy_name: [RoutingDecision, ...]}.
        Useful for benchmark comparisons.
        """
        names = strategy_names or self.registered_names
        results: Dict[str, List[RoutingDecision]] = {}
        for name in names:
            strategy = self.get(name)
            strategy.reset()
            results[name] = strategy.route_batch(queries, data_location)
        return results
