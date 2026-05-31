"""
lynceus_port/router.py — 策略注册中心与查询路由器。

移植自 lynceus/router.py，修改约20%:
  - 新增路由历史记录 (history)，保留最近 N 条决策
  - set_active 增加切换钩子回调
  - route_one/route_batch 自动记录 wall-clock 耗时
  - debug_snapshot 打印注册表 + 活跃策略 + 历史摘要
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional

from .cost_model import CostModelEngine, QueryDescriptor
from .schema import RoutingStrategy, _dbg
from .strategies.base import RoutingDecision, RoutingStrategyBase
from .strategies.static import (
    CPUOnlyStrategy,
    GPUOnlyStrategy,
    HybridStaticStrategy,
)
from .strategies.cost_driven import (
    CostModelRoutedStrategy,
    PAR2QORobustStrategy,
)
from .strategies.adaptive import AdaptiveStrategy


class Router:
    """策略注册中心与查询分发器。

    维护一个策略注册表，类似 NCCL 的算法表。
    新增：路由历史缓冲区 + 策略切换钩子。
    """

    HISTORY_CAPACITY = 500  # 保留最近 500 条路由决策

    def __init__(self, engine: CostModelEngine):
        self._engine = engine
        self._registry: Dict[str, RoutingStrategyBase] = {}
        self._active: Optional[RoutingStrategyBase] = None
        # ── 新增：路由历史 ──
        self._history: Deque[RoutingDecision] = deque(
            maxlen=self.HISTORY_CAPACITY)
        self._switch_hooks: List[Callable[[str, str], None]] = []
        self._total_route_time_us = 0.0

    # ── 注册 ──

    def register(self, strategy: RoutingStrategyBase) -> None:
        if strategy.name in self._registry:
            raise ValueError(
                f"策略 '{strategy.name}' 已注册，用 replace() 覆盖")
        self._registry[strategy.name] = strategy
        _dbg("Router", f"registered: {strategy.name}")

    def replace(self, strategy: RoutingStrategyBase) -> None:
        self._registry[strategy.name] = strategy

    def unregister(self, name: str) -> None:
        if name not in self._registry:
            raise KeyError(f"策略 '{name}' 不存在")
        if self._active is not None and self._active.name == name:
            self._active = None
        del self._registry[name]

    @property
    def registered_names(self) -> List[str]:
        return list(self._registry.keys())

    def get(self, name: str) -> RoutingStrategyBase:
        if name not in self._registry:
            raise KeyError(
                f"策略 '{name}' 不存在。可用: {self.registered_names}")
        return self._registry[name]

    # ── 活跃策略选择 ──

    def add_switch_hook(self, hook: Callable[[str, str], None]) -> None:
        """添加策略切换钩子: hook(old_name, new_name)"""
        self._switch_hooks.append(hook)

    def set_active(self, name: str) -> None:
        old_name = self._active.name if self._active else "(none)"
        self._active = self.get(name)
        _dbg("Router", f"switch: {old_name} -> {name}")
        for hook in self._switch_hooks:
            hook(old_name, name)

    @property
    def active(self) -> RoutingStrategyBase:
        if self._active is None:
            raise RuntimeError("未设置活跃策略，先调用 set_active()")
        return self._active

    @property
    def active_name(self) -> Optional[str]:
        return self._active.name if self._active else None

    # ── 路由 ──

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        t0 = time.perf_counter()
        decision = self.active.route_one(query, data_location)
        elapsed_us = (time.perf_counter() - t0) * 1e6
        self._total_route_time_us += elapsed_us
        self._history.append(decision)
        _dbg("Router",
             f"routed {query.query_id} -> {decision.device_id} "
             f"in {elapsed_us:.1f}us")
        return decision

    def route_batch(self, queries: List[QueryDescriptor],
                    data_location: Optional[str] = None
                    ) -> List[RoutingDecision]:
        t0 = time.perf_counter()
        decisions = self.active.route_batch(queries, data_location)
        elapsed_us = (time.perf_counter() - t0) * 1e6
        self._total_route_time_us += elapsed_us
        for d in decisions:
            self._history.append(d)
        _dbg("Router",
             f"batch {len(queries)} queries in {elapsed_us:.1f}us, "
             f"history_len={len(self._history)}")
        return decisions

    # ── 工厂 ──

    @classmethod
    def create_default(cls, engine: CostModelEngine, **kwargs) -> "Router":
        router = cls(engine)
        router.register(GPUOnlyStrategy(engine, **kwargs))
        router.register(CPUOnlyStrategy(engine, **kwargs))
        router.register(HybridStaticStrategy(engine, **kwargs))
        router.register(CostModelRoutedStrategy(engine, **kwargs))
        router.register(PAR2QORobustStrategy(engine, **kwargs))
        router.register(AdaptiveStrategy(engine, **kwargs))
        _dbg("Router", f"default router: {router.registered_names}")
        return router

    # ── 基准测试辅助 ──

    def run_all_strategies(
        self,
        queries: List[QueryDescriptor],
        data_location: Optional[str] = None,
        strategy_names: Optional[List[str]] = None,
    ) -> Dict[str, List[RoutingDecision]]:
        names = strategy_names or self.registered_names
        results: Dict[str, List[RoutingDecision]] = {}
        for name in names:
            strategy = self.get(name)
            strategy.reset()
            results[name] = strategy.route_batch(queries, data_location)
            _dbg("Router",
                 f"strategy '{name}': {len(results[name])} decisions")
        return results

    # ── 调试 ──

    def debug_snapshot(self) -> str:
        # 历史统计
        device_counts: Dict[str, int] = {}
        for d in self._history:
            device_counts[d.device_id] = device_counts.get(
                d.device_id, 0) + 1

        lines = [
            "══════ Router Snapshot ══════",
            f"  active     = {self.active_name}",
            f"  registered = {self.registered_names}",
            f"  history    = {len(self._history)}/{self.HISTORY_CAPACITY}",
            f"  total_route_time = {self._total_route_time_us:.1f}us",
            f"  device_dist = {device_counts}",
        ]
        s = "\n".join(lines)
        _dbg("Router", s)
        return s
