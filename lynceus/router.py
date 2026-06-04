"""
lynceus/router.py — Strategy registry and query router.

改动:
    run_all_strategies → 附带 tournament 对决: 两两比较每条 query 的 latency,
    输出 win_matrix (A在多少条query上beat B) 和 Elo rating。
    便于 benchmark 不仅看平均值, 还能看"谁对谁有优势"。
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
from .cost_model import CostModelEngine, QueryDescriptor
from .schema import RoutingStrategy
from .strategies.base import RoutingDecision, RoutingStrategyBase
from .strategies.static import CPUOnlyStrategy, GPUOnlyStrategy, HybridStaticStrategy
from .strategies.cost_driven import CostModelRoutedStrategy, PAR2QOEnhancedStrategy
from .strategies.adaptive import AdaptiveStrategy


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
            raise KeyError(f"Strategy '{name}' not found in registry")
        if self._active is not None and self._active.name == name:
            self._active = None
        del self._registry[name]

    @property
    def registered_names(self) -> List[str]:
        return list(self._registry.keys())

    def get(self, name: str) -> RoutingStrategyBase:
        if name not in self._registry:
            raise KeyError(f"Strategy '{name}' not found. Available: {self.registered_names}")
        return self._registry[name]

    def set_active(self, name: str) -> None:
        self._active = self.get(name)

    @property
    def active(self) -> RoutingStrategyBase:
        if self._active is None:
            raise RuntimeError("No active strategy set. Call set_active() first.")
        return self._active

    @property
    def active_name(self) -> Optional[str]:
        return self._active.name if self._active else None

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        from ._debug import dbg
        dbg('Router.route_one', query_id=query.query_id, strategy=self.active_name)
        return self.active.route_one(query, data_location)

    def route_batch(self, queries: List[QueryDescriptor],
                    data_location: Optional[str] = None) -> List[RoutingDecision]:
        return self.active.route_batch(queries, data_location)

    @classmethod
    def create_default(cls, engine: CostModelEngine, **kwargs) -> "Router":
        router = cls(engine)
        router.register(GPUOnlyStrategy(engine, **kwargs))
        router.register(CPUOnlyStrategy(engine, **kwargs))
        router.register(HybridStaticStrategy(engine, **kwargs))
        router.register(CostModelRoutedStrategy(engine, **kwargs))
        router.register(PAR2QOEnhancedStrategy(engine, **kwargs))
        router.register(AdaptiveStrategy(engine, **kwargs))
        return router

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
        return results

    # ─── 改动: tournament bracket ────────────────────────────────
    def tournament(
        self,
        queries: List[QueryDescriptor],
        data_location: Optional[str] = None,
        strategy_names: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """两两对决 + Elo rating, 比 run_all_strategies 多出竞选维度。

        返回:
            {
                "results": {name: [decisions]},
                "win_matrix": {A: {B: A对B赢了多少条query}},
                "elo": {name: Elo分},
                "per_query_winner": [每条query的赢家name],
            }
        """
        from ._debug import dbg, checkpoint
        results = self.run_all_strategies(queries, data_location, strategy_names)
        names = list(results.keys())
        n_queries = len(queries)

        # 抽取每个策略每条 query 的 latency
        latencies: Dict[str, List[float]] = {}
        for name, decs in results.items():
            latencies[name] = [d.cost.total_ms for d in decs]

        # win_matrix[A][B] = A beat B 的 query 数
        win_matrix: Dict[str, Dict[str, int]] = {n: {m: 0 for m in names} for n in names}
        per_query_winner: List[str] = []

        for qi in range(n_queries):
            best_name = min(names, key=lambda n: latencies[n][qi])
            per_query_winner.append(best_name)
            for a in names:
                for b in names:
                    if a == b:
                        continue
                    if latencies[a][qi] < latencies[b][qi]:
                        win_matrix[a][b] += 1

        # Elo rating: K=32, 从1500起步, 按 query 顺序更新
        elo: Dict[str, float] = {n: 1500.0 for n in names}
        K = 32.0
        for qi in range(n_queries):
            # 找这条 query 的最优和最差
            sorted_names = sorted(names, key=lambda n: latencies[n][qi])
            winner = sorted_names[0]
            loser = sorted_names[-1]
            if winner == loser:
                continue
            # 标准 Elo 公式
            expected_w = 1.0 / (1.0 + 10.0 ** ((elo[loser] - elo[winner]) / 400.0))
            elo[winner] += K * (1.0 - expected_w)
            elo[loser] -= K * (1.0 - expected_w)

        checkpoint("tournament_done",
                   n_strategies=len(names), n_queries=n_queries,
                   elo={k: f"{v:.0f}" for k, v in sorted(elo.items(), key=lambda x: -x[1])})

        dbg('Router.tournament', elo=elo, win_matrix_diagonal={
            n: sum(win_matrix[n].values()) for n in names})

        return {
            "results": results,
            "win_matrix": win_matrix,
            "elo": elo,
            "per_query_winner": per_query_winner,
        }
