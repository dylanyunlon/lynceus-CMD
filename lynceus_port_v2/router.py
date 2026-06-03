"""
lynceus_port/router.py — 移植版策略注册与查询路由器.

算法改写:
  - route_batch: 增加批量前瞻(lookahead)——先统计整批的查询分布,
    再给 strategy 一个 hint dict, 让 adaptive 策略能做全局负载均衡
    而不是逐条贪心.
  - run_all_strategies: 增加 Borda 排名聚合, 对每个 query 跨策略
    的延迟做排名, 输出哪个策略在全局上最优.

溯源同原版 (NCCL registry / vLLM scheduler / PyTorch c10d).
"""

from __future__ import annotations

from collections import defaultdict
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

from . import _dbg, _dump_obj, _snapshot, _Timer, LYNCEUS_DEBUG
_T = "ROU"


class Router:
    """策略注册表 + 查询分发器."""

    def __init__(self, engine: CostModelEngine):
        _dbg(_T, f"__init__ called")
        self._engine = engine
        self._registry: Dict[str, RoutingStrategyBase] = {}
        self._active: Optional[RoutingStrategyBase] = None
        # [PORT] 路由历史, 供事后分析
        self._history: List[RoutingDecision] = []
        _dbg(_T, "Router.__init__")

    # --- Registry ---

    def register(self, strategy: RoutingStrategyBase) -> None:
        _dbg(_T, f"register called")
        if strategy.name in self._registry:
            raise ValueError(
                f"Strategy '{strategy.name}' already registered. "
                f"Use replace() to override."
            )
        self._registry[strategy.name] = strategy
        _dbg(_T, f"register({strategy.name})")

    def replace(self, strategy: RoutingStrategyBase) -> None:
        _dbg(_T, f"replace called")
        self._registry[strategy.name] = strategy

    def unregister(self, name: str) -> None:
        _dbg(_T, f"unregister called")
        if name not in self._registry:
            raise KeyError(f"Strategy '{name}' not found in registry")
        if self._active is not None and self._active.name == name:
            self._active = None
        del self._registry[name]

    @property
    def registered_names(self) -> List[str]:
        _dbg(_T, f"registered_names called")
        return list(self._registry.keys())

    def get(self, name: str) -> RoutingStrategyBase:
        _dbg(_T, f"get called")
        if name not in self._registry:
            raise KeyError(
                f"Strategy '{name}' not found. "
                f"Available: {self.registered_names}"
            )
        return self._registry[name]

    # --- Active strategy ---

    def set_active(self, name: str) -> None:
        self._active = self.get(name)
        _dbg(_T, f"set_active({name})")

    @property
    def active(self) -> RoutingStrategyBase:
        _dbg(_T, f"active called")
        if self._active is None:
            raise RuntimeError("No active strategy set. Call set_active() first.")
        return self._active

    @property
    def active_name(self) -> Optional[str]:
        _dbg(_T, f"active_name called")
        return self._active.name if self._active else None

    # --- Routing ---

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        _dbg(_T, f"route_one called")
        dec = self.active.route_one(query, data_location)
        self._history.append(dec)
        return dec

    def route_batch(self, queries: List[QueryDescriptor],
                    data_location: Optional[str] = None
                    ) -> List[RoutingDecision]:
        """[PORT] 批量路由 — 增加前瞻 hint.

        原版逐条调用 route_one. 移植版先扫一遍整批 queries,
        统计 query_type 分布和 estimated_rows 的分位数,
        作为 batch_hint 传给 strategy (如果它支持的话).
        这让 adaptive 策略能做全局决策而非逐条贪心.
        """
        _dbg(_T, f"route_batch called")
        strategy = self.active

        # [PORT] 构建批量前瞻 hint
        type_counts: Dict[str, int] = defaultdict(int)
        all_rows = []
        for q in queries:
            type_counts[q.query_type.name] += 1
            all_rows.append(q.estimated_rows)
        all_rows.sort()
        n = len(all_rows)
        batch_hint = {
            "batch_size": n,
            "type_distribution": dict(type_counts),
            "rows_p25": all_rows[n // 4] if n >= 4 else (all_rows[0] if n else 0),
            "rows_p50": all_rows[n // 2] if n >= 2 else (all_rows[0] if n else 0),
            "rows_p75": all_rows[3 * n // 4] if n >= 4 else (all_rows[-1] if n else 0),
        }
        _dbg(_T, f"route_batch: n={n}, hint_p50={batch_hint['rows_p50']}")

        # 如果 strategy 有 set_batch_hint, 传入
        if hasattr(strategy, 'set_batch_hint'):
            strategy.set_batch_hint(batch_hint)

        decisions = strategy.route_batch(queries, data_location)
        self._history.extend(decisions)
        return decisions

    # --- Factory ---

    @classmethod
    def create_default(cls, engine: CostModelEngine, **kwargs) -> "Router":
        _dbg(_T, f"create_default called")
        router = cls(engine)
        router.register(GPUOnlyStrategy(engine, **kwargs))
        router.register(CPUOnlyStrategy(engine, **kwargs))
        router.register(HybridStaticStrategy(engine, **kwargs))
        router.register(CostModelRoutedStrategy(engine, **kwargs))
        router.register(PAR2QOEnhancedStrategy(engine, **kwargs))
        router.register(AdaptiveStrategy(engine, **kwargs))
        return router

    # --- Benchmark helper ---

    def run_all_strategies(
        self,
        queries: List[QueryDescriptor],
        data_location: Optional[str] = None,
        strategy_names: Optional[List[str]] = None,
    ) -> Dict[str, List[RoutingDecision]]:
        """[PORT] 增加 Borda 排名聚合.

        对每个 query, 各策略产生一个延迟. 按延迟排名,
        最低得 0 分, 最高得 len(strategies)-1 分.
        最终 Borda 得分最低的策略全局最优.
        """
        _dbg(_T, f"run_all_strategies called")
        names = strategy_names or self.registered_names
        results: Dict[str, List[RoutingDecision]] = {}
        for name in names:
            strategy = self.get(name)
            strategy.reset()
            results[name] = strategy.route_batch(queries, data_location)

        # [PORT] Borda 排名
        borda: Dict[str, int] = defaultdict(int)
        for i in range(len(queries)):
            # 收集各策略在第 i 个 query 上的延迟
            costs = []
            for name in names:
                if i < len(results[name]):
                    costs.append((name, results[name][i].cost.total_us))
            costs.sort(key=lambda x: x[1])
            for rank, (name, _) in enumerate(costs):
                borda[name] += rank

        if LYNCEUS_DEBUG:
            _dbg(_T, f"Borda ranks: {dict(borda)}")
            winner = min(borda, key=borda.get) if borda else "?"
            _dbg(_T, f"Borda winner: {winner}")

        return results

    # [PORT] 路由历史分析
    def dump_history_summary(self) -> Dict[str, int]:
        """断点辅助: 按 device_id 统计路由历史."""
        _dbg(_T, f"dump_history_summary called")
        counts: Dict[str, int] = defaultdict(int)
        for dec in self._history:
            counts[dec.device_id] += 1
        _dbg(_T, f"history summary: {dict(counts)}")
        return dict(counts)
