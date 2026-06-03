"""
lynceus_port/strategies/cost_driven.py — 移植版代价驱动路由.

算法改写:
  - CostModelRoutedStrategy.route_batch: 原版逐条 engine.recommend.
    移植版增加 batch 级别的代价缓存——相同 (query_type, table_name,
    selectivity_bucket) 的 query 复用上一次的路由决策, 省掉重复计算.
    selectivity 分桶 (8 个 bucket) 做近似匹配.

  - PAR2QOEnhancedStrategy: 原版用固定 margin=20%.
    移植版改为 SMAPE 对称百分比误差计算 GPU 优势度:
    advantage = |cpu - gpu| / (|cpu| + |gpu|) * 2
    只有当 advantage > margin 且 gpu < cpu 时才选 GPU.
    SMAPE 对称, 不会因为绝对值差异大而偏向一方.

溯源同原版 (NCCL cost table / PAR2QO parametric penalty).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..cost_model import CostBreakdown, CostModelEngine, QueryDescriptor
from ..schema import HardwareKind
from .base import RoutingDecision, RoutingStrategyBase
from .. import _dbg, _snapshot, LYNCEUS_DEBUG

_T = "CDR"

# [PORT] selectivity 分桶数
_NUM_SEL_BUCKETS = 8


def _sel_bucket(selectivity: float) -> int:
    """将 selectivity 映射到 0~_NUM_SEL_BUCKETS-1 的桶."""
    return min(_NUM_SEL_BUCKETS - 1, int(selectivity * _NUM_SEL_BUCKETS))


class CostModelRoutedStrategy(RoutingStrategyBase):

    def __init__(self, engine: CostModelEngine, **kwargs):
        super().__init__(engine, **kwargs)
        # [PORT] batch 级代价缓存: (query_type, table, sel_bucket) -> (dev, cb)
        self._batch_cache: Dict[tuple, tuple] = {}

    @property
    def name(self) -> str:
        return "CostModel-Routed"

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        device_id, cb = self._engine.recommend(query, data_location)
        return RoutingDecision(
            query_id=query.query_id,
            device_id=device_id,
            cost=cb,
            confidence=1.0,
            metadata={"reason": "min_cost"},
        )

    def route_batch(self, queries: List[QueryDescriptor],
                    data_location: Optional[str] = None
                    ) -> List[RoutingDecision]:
        """[PORT] 带缓存的批量路由.

        相同 signature 的 query 复用决策, 减少重复代价计算.
        cache hit 时直接返回上次结果(用新 query_id 替换),
        cache miss 时调 engine.recommend 并缓存.
        """
        self._batch_cache.clear()
        results = []
        hits = 0

        for q in queries:
            cache_key = (q.query_type.value, q.table_name,
                         _sel_bucket(q.selectivity))

            if cache_key in self._batch_cache:
                cached_dev, cached_cb = self._batch_cache[cache_key]
                # 用缓存的 device, 但重新估算代价(因为 rows 可能不同)
                cb = self._engine.estimate_on_device(q, cached_dev, data_location)
                results.append(RoutingDecision(
                    query_id=q.query_id,
                    device_id=cached_dev,
                    cost=cb,
                    confidence=0.95,  # 缓存决策稍低置信度
                    metadata={"reason": "cached_route"},
                ))
                hits += 1
            else:
                dev, cb = self._engine.recommend(q, data_location)
                self._batch_cache[cache_key] = (dev, cb)
                results.append(RoutingDecision(
                    query_id=q.query_id,
                    device_id=dev,
                    cost=cb,
                    confidence=1.0,
                    metadata={"reason": "min_cost"},
                ))

        if LYNCEUS_DEBUG:
            _dbg(_T, f"batch cache: {hits}/{len(queries)} hits "
                 f"({len(self._batch_cache)} unique signatures)")

        return results


class PAR2QOEnhancedStrategy(RoutingStrategyBase):
    """[PORT] SMAPE 对称误差替代固定 margin.

    原版: gpu_cost < cpu_cost * (1 - margin) → GPU.
    移植版: SMAPE advantage = |cpu - gpu| / ((|cpu| + |gpu|) / 2)
    只有 SMAPE > margin 且 gpu < cpu 时选 GPU.

    SMAPE 的好处: 对称, 不受绝对量级影响.
    当 cpu=1000, gpu=800 时, 原版 margin=20% 刚好不选 GPU (800 < 800),
    但 SMAPE = 200/900 = 22.2%, 会选 GPU. 更合理.
    """

    def __init__(self, engine: CostModelEngine, *,
                 robustness_margin: float = 0.20, **kwargs):
        super().__init__(engine, **kwargs)
        self._margin = robustness_margin

    @property
    def name(self) -> str:
        return "PAR2QO-Enhanced"

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        estimates = self._engine.estimate_all_devices(query, data_location)

        gpu_ids = [k for k, n in self._engine.topology.nodes.items()
                   if n.kind == HardwareKind.GPU and k in estimates]
        cpu_ids = [k for k, n in self._engine.topology.nodes.items()
                   if n.kind == HardwareKind.CPU and k in estimates]

        if not gpu_ids or not cpu_ids:
            device_id, cb = self._engine.recommend(query, data_location)
            return RoutingDecision(
                query_id=query.query_id, device_id=device_id,
                cost=cb, confidence=0.5,
                metadata={"reason": "fallback_incomplete_topology"},
            )

        best_gpu = min(gpu_ids, key=lambda k: estimates[k].total_us)
        best_cpu = min(cpu_ids, key=lambda k: estimates[k].total_us)
        gpu_cost = estimates[best_gpu].total_us
        cpu_cost = estimates[best_cpu].total_us

        # [PORT] SMAPE 对称百分比误差
        denom = (abs(cpu_cost) + abs(gpu_cost)) / 2.0
        if denom > 0:
            smape_advantage = abs(cpu_cost - gpu_cost) / denom
        else:
            smape_advantage = 0.0

        # GPU 选择条件: SMAPE 超过 margin 且 GPU 确实更快
        if gpu_cost < cpu_cost and smape_advantage > self._margin:
            chosen = best_gpu
            reason = "gpu_smape_winner"
        else:
            chosen = best_cpu
            reason = "cpu_robust_choice"

        _snapshot(_T, "par2qo_decision",
                  query=query.query_id,
                  gpu_cost_us=round(gpu_cost, 1),
                  cpu_cost_us=round(cpu_cost, 1),
                  smape=round(smape_advantage, 4),
                  chosen=chosen, reason=reason)

        return RoutingDecision(
            query_id=query.query_id,
            device_id=chosen,
            cost=estimates[chosen],
            confidence=1.0 if reason == "gpu_smape_winner" else 0.8,
            metadata={
                "reason": reason,
                "gpu_cost_us": gpu_cost,
                "cpu_cost_us": cpu_cost,
                "margin": self._margin,
                "smape_advantage": smape_advantage,
            },
        )
