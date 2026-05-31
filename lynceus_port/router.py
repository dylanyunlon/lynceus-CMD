"""
lynceus_port/router.py — 移植版策略注册表和查询路由器.

改写: 增加 route_with_trace 方法, 返回完整决策推理链;
      run_all_strategies 增加进度回调 + 耗时统计.
"""
from __future__ import annotations
import time
from typing import Callable, Dict, List, Optional
from .cost_model import CostModelEngine, QueryDescriptor
from .schema import RoutingStrategy
from .strategies.base import RoutingDecision, RoutingStrategyBase
from .strategies.static import CPUOnlyStrategy, GPUOnlyStrategy, HybridStaticStrategy
from .strategies.cost_driven import CostModelRoutedStrategy, PAR2QOEnhancedStrategy
from .strategies.adaptive import AdaptiveStrategy
from . import _dbg


class Router:
    def __init__(self, engine: CostModelEngine):
        self._engine = engine
        self._registry: Dict[str, RoutingStrategyBase] = {}
        self._active: Optional[RoutingStrategyBase] = None

    def register(self, strategy: RoutingStrategyBase) -> None:
        if strategy.name in self._registry:
            raise ValueError(f"Strategy '{strategy.name}' already registered.")
        self._registry[strategy.name] = strategy

    def replace(self, strategy: RoutingStrategyBase) -> None:
        self._registry[strategy.name] = strategy

    def unregister(self, name: str) -> None:
        if name not in self._registry:
            raise KeyError(f"Strategy '{name}' not found")
        if self._active and self._active.name == name:
            self._active = None
        del self._registry[name]

    @property
    def registered_names(self) -> List[str]:
        return list(self._registry.keys())

    def get(self, name: str) -> RoutingStrategyBase:
        if name not in self._registry:
            raise KeyError(f"'{name}' not found. Available: {self.registered_names}")
        return self._registry[name]

    def set_active(self, name: str) -> None:
        self._active = self.get(name)
        _dbg(f"router.set_active → {name}")

    @property
    def active(self) -> RoutingStrategyBase:
        if self._active is None:
            raise RuntimeError("No active strategy. Call set_active() first.")
        return self._active

    @property
    def active_name(self) -> Optional[str]:
        return self._active.name if self._active else None

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        return self.active.route_one(query, data_location)

    def route_batch(self, queries: List[QueryDescriptor],
                    data_location: Optional[str] = None) -> List[RoutingDecision]:
        return self.active.route_batch(queries, data_location)

    # ★ 改写: 带 trace 的路由 — 返回完整推理链
    def route_with_trace(self, query: QueryDescriptor,
                         data_location: Optional[str] = None
                         ) -> RoutingDecision:
        """路由 + 打印决策推理链 (断点调试用)."""
        dec = self.active.route_one(query, data_location)
        print(f"\n┌── Route Trace: {query.query_id} ──")
        print(f"│ strategy = {self.active_name}")
        for line in dec.trace_log:
            print(f"│   {line}")
        print(f"│ → {dec.device_id} ({dec.cost.total_us:.1f}µs, "
              f"conf={dec.confidence:.2f})")
        print(f"└──────────────────────────────────────")
        return dec

    @classmethod
    def create_default(cls, engine: CostModelEngine, **kw) -> "Router":
        router = cls(engine)
        router.register(GPUOnlyStrategy(engine, **kw))
        router.register(CPUOnlyStrategy(engine, **kw))
        router.register(HybridStaticStrategy(engine, **kw))
        router.register(CostModelRoutedStrategy(engine, **kw))
        router.register(PAR2QOEnhancedStrategy(engine, **kw))
        router.register(AdaptiveStrategy(engine, **kw))
        return router

    def run_all_strategies(
        self, queries: List[QueryDescriptor],
        data_location: Optional[str] = None,
        strategy_names: Optional[List[str]] = None,
        progress_cb: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, List[RoutingDecision]]:
        """运行所有策略 — ★ 改写: 增加进度回调 + 耗时."""
        names = strategy_names or self.registered_names
        results: Dict[str, List[RoutingDecision]] = {}
        for idx, name in enumerate(names):
            t0 = time.monotonic()
            strategy = self.get(name)
            strategy.reset()
            results[name] = strategy.route_batch(queries, data_location)
            elapsed = time.monotonic() - t0
            if progress_cb:
                progress_cb(name, elapsed)
            _dbg(f"run_all: {name} done in {elapsed:.3f}s")
        return results

    def dump_registry(self) -> str:
        """断点辅助: 打印注册表全貌."""
        lines = ["┌── Router Registry ──"]
        for name, strat in self._registry.items():
            active = " ← ACTIVE" if self._active and self._active.name == name else ""
            lines.append(f"│ {name}: {type(strat).__name__}{active}")
        lines.append(f"└── {len(self._registry)} strategies registered")
        return "\n".join(lines)
