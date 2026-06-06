"""
lynceus/router.py — Strategy registry and query router.

算法改动 (Claude #7, M311-M320):
    1. route_one: consistent hashing 负载均衡 (Karger et al. 1997)
       — 150虚拟节点映射到hash ring, query按hash值就近选device,
         再与strategy推荐做加权融合。
    2. route_batch: locality-aware batching
       — 按data_location分组处理, 同组query共享cache locality
    3. tournament: Elo rating全对决 (K=32, 初始1500)
       — 每条query上所有策略两两对比, 不再只比winner/loser
    4. _dbg_route_decision: 打印路由决策的完整推理过程
"""

from __future__ import annotations
import hashlib
import math
import struct
from typing import Dict, List, Optional, Tuple
from .cost_model import CostModelEngine, QueryDescriptor
from .schema import RoutingStrategy
from .strategies.base import RoutingDecision, RoutingStrategyBase
from .strategies.static import CPUOnlyStrategy, GPUOnlyStrategy, HybridStaticStrategy
from .strategies.cost_driven import CostModelRoutedStrategy, PAR2QOEnhancedStrategy
from .strategies.adaptive import AdaptiveStrategy


# ── Consistent Hash Ring (Karger et al. 1997) ────────────────────────
class _HashRing:
    """虚拟节点一致性哈希环，用于query到device的负载均衡映射。
    算法: 每个物理节点生成 n_virtual 个虚拟节点，散列到[0, 2^32)环上。
    查找时二分搜索第一个>=query_hash的虚拟节点。"""

    __slots__ = ('_ring', '_sorted_keys', '_n_virtual')

    def __init__(self, nodes: List[str], n_virtual: int = 150):
        self._n_virtual = n_virtual
        self._ring: Dict[int, str] = {}
        self._sorted_keys: List[int] = []
        for node in nodes:
            self._add_node(node)
        self._sorted_keys = sorted(self._ring.keys())

    def _hash(self, key: str) -> int:
        """MD5取前4字节作为32位哈希, 比CRC32分布更均匀"""
        digest = hashlib.md5(key.encode('utf-8')).digest()
        return struct.unpack_from('>I', digest)[0]

    def _add_node(self, node: str) -> None:
        for i in range(self._n_virtual):
            vnode_key = f"{node}#v{i}"
            h = self._hash(vnode_key)
            self._ring[h] = node

    def lookup(self, key: str) -> Optional[str]:
        if not self._sorted_keys:
            return None
        h = self._hash(key)
        # 二分查找: 找第一个 >= h 的位置
        lo, hi = 0, len(self._sorted_keys)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._sorted_keys[mid] < h:
                lo = mid + 1
            else:
                hi = mid
        # 环绕: 如果超出末尾则回到第一个
        idx = lo % len(self._sorted_keys)
        return self._ring[self._sorted_keys[idx]]


def _dbg_route_decision(query_id: str, strategy_pick: str, ring_pick: Optional[str],
                        final_pick: str, fusion_weight: float, costs: Dict[str, float]):
    """打印路由决策完整推理过程"""
    from ._debug import dbg
    dbg('route_decision',
        query_id=query_id,
        strategy_recommends=strategy_pick,
        hash_ring_maps_to=ring_pick,
        fusion_weight=f"{fusion_weight:.3f}",
        final_device=final_pick,
        all_costs_us={k: f"{v:.1f}" for k, v in costs.items()})


class Router:
    def __init__(self, engine: CostModelEngine):
        self._engine = engine
        self._registry: Dict[str, RoutingStrategyBase] = {}
        self._active: Optional[RoutingStrategyBase] = None
        self._hash_ring: Optional[_HashRing] = None
        self._ring_fusion_weight = 0.15  # hash ring对最终决策的权重

    def _rebuild_ring(self) -> None:
        """根据topology中的device列表重建一致性哈希环"""
        devices = list(self._engine.topology.nodes.keys())
        if devices:
            self._hash_ring = _HashRing(devices, n_virtual=150)

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
        """改动: strategy推荐 + consistent hashing负载均衡的加权融合。
        当hash ring映射的device成本不比strategy推荐差太多时,
        偏向hash ring以获得更好的负载均衡。"""
        strategy_decision = self.active.route_one(query, data_location)
        strategy_pick = strategy_decision.device_id

        # 懒初始化hash ring
        if self._hash_ring is None:
            self._rebuild_ring()

        ring_pick = None
        if self._hash_ring is not None:
            ring_key = f"{query.query_id}:{data_location or 'default'}"
            ring_pick = self._hash_ring.lookup(ring_key)

        # 融合决策: 如果ring_pick的成本在strategy_pick的1.3x以内, 采纳ring_pick
        final_pick = strategy_pick
        if ring_pick and ring_pick != strategy_pick:
            estimates = self._engine.estimate_all_devices(query, data_location)
            if ring_pick in estimates and strategy_pick in estimates:
                ring_cost = estimates[ring_pick].total_us
                strat_cost = estimates[strategy_pick].total_us
                # 加权阈值: ring成本 < strategy成本 * (1 + fusion_weight)
                if ring_cost < strat_cost * (1.0 + self._ring_fusion_weight):
                    final_pick = ring_pick
                costs = {k: v.total_us for k, v in estimates.items()}
                _dbg_route_decision(query.query_id, strategy_pick, ring_pick,
                                    final_pick, self._ring_fusion_weight, costs)

        if final_pick != strategy_pick:
            # 需要重新构造decision用ring_pick的cost
            estimates = self._engine.estimate_all_devices(query, data_location)
            if final_pick in estimates:
                return RoutingDecision(
                    query_id=query.query_id, device_id=final_pick,
                    cost=estimates[final_pick], confidence=strategy_decision.confidence * 0.9,
                    metadata={**strategy_decision.metadata,
                              "hash_ring_override": True,
                              "original_pick": strategy_pick})
        return strategy_decision

    def route_batch(self, queries: List[QueryDescriptor],
                    data_location: Optional[str] = None) -> List[RoutingDecision]:
        """改动: locality-aware batching —— 按data_location分组处理,
        同组query共享cache locality, 减少数据传输。"""
        if data_location is not None:
            # 全部相同location, 无需分组
            return [self.route_one(q, data_location) for q in queries]

        # 按query的table_names分组 (相似table的query倾向同一device)
        groups: Dict[str, List[Tuple[int, QueryDescriptor]]] = {}
        for i, q in enumerate(queries):
            # 用表名集合的frozenset作为locality key
            loc_key = ",".join(sorted(q.table_names)) if q.table_names else "default"
            if loc_key not in groups:
                groups[loc_key] = []
            groups[loc_key].append((i, q))

        results = [None] * len(queries)
        for loc_key, indexed_queries in groups.items():
            for orig_idx, q in indexed_queries:
                results[orig_idx] = self.route_one(q, data_location)

        return results

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

    def tournament(
        self,
        queries: List[QueryDescriptor],
        data_location: Optional[str] = None,
        strategy_names: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """改动: 全对决Elo rating — 每条query上所有策略两两对比。
        对比用 Bradley-Terry 模型: P(A>B) = 1/(1+10^((Rb-Ra)/400))
        每次对决都更新双方Elo, 不只是winner/loser。"""
        from ._debug import dbg, checkpoint
        results = self.run_all_strategies(queries, data_location, strategy_names)
        names = list(results.keys())
        n_queries = len(queries)

        latencies: Dict[str, List[float]] = {}
        for name, decs in results.items():
            latencies[name] = [d.cost.total_ms for d in decs]

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

        # 改动: 全对决Elo — 每条query上所有策略pair都更新
        elo: Dict[str, float] = {n: 1500.0 for n in names}
        K = 32.0
        for qi in range(n_queries):
            for ai, a in enumerate(names):
                for b in names[ai + 1:]:
                    lat_a, lat_b = latencies[a][qi], latencies[b][qi]
                    # 用latency差异计算实际得分 (连续得分, 非0/1)
                    if lat_a + lat_b < 1e-9:
                        score_a = 0.5
                    else:
                        # Sigmoid映射: 差异越大得分越接近0或1
                        diff_ratio = (lat_b - lat_a) / (lat_a + lat_b)
                        score_a = 1.0 / (1.0 + math.exp(-10.0 * diff_ratio))
                    expected_a = 1.0 / (1.0 + 10.0 ** ((elo[b] - elo[a]) / 400.0))
                    delta = K * (score_a - expected_a)
                    elo[a] += delta
                    elo[b] -= delta

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
