"""
lynceus_port/strategies/cost_driven.py — 代价驱动路由.

改写: PAR2QOEnhancedStrategy 增加历史方差追踪 —
      margin 随方差自适应放大, 方差大 → 更保守选 CPU.
"""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Dict, List, Optional
from ..cost_model import CostBreakdown, CostModelEngine, QueryDescriptor
from ..schema import HardwareKind
from .base import RoutingDecision, RoutingStrategyBase
from .. import _dbg

_MOD_TAG = "CON"
import os as _os, sys as _sys
_LYNCEUS_DBG = _os.environ.get("LYNCEUS_DEBUG", "1")

def _dbg(tag: str, msg: str):
    if _LYNCEUS_DBG != "0":
        print(f"[{_MOD_TAG}·{tag}] {msg}", file=_sys.stderr, flush=True)

_tr = _dbg  # 兼容旧调用



class CostModelRoutedStrategy(RoutingStrategyBase):
    @property
    def name(self) -> str:
        return "CostModel-Routed"

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        self._query_count += 1
        device_id, cb = self._engine.recommend(query, data_location)
        return RoutingDecision(
            query_id=query.query_id, device_id=device_id, cost=cb,
            confidence=1.0, metadata={"reason": "min_cost"},
            trace_log=[f"min_cost → {device_id} ({cb.total_us:.1f}µs)"],
        )

    def route_batch(self, queries: List[QueryDescriptor],
                    data_location: Optional[str] = None
                    ) -> List[RoutingDecision]:
        results = self._engine.route_batch(queries, data_location)
        return [
            RoutingDecision(
                query_id=q.query_id, device_id=dev, cost=cb,
                confidence=1.0, metadata={"reason": "min_cost"},
                trace_log=[f"batch_min_cost → {dev}"],
            )
            for q, (dev, cb) in zip(queries, results)
        ]


class PAR2QOEnhancedStrategy(RoutingStrategyBase):
    """代价路由 + 鲁棒性惩罚.

    ★ 改写: margin 随历史代价估计方差自适应放大.
    方差越大 → 代价模型越不可靠 → 越倾向选 CPU (稳妥).
    """

    def __init__(self, engine: CostModelEngine, *,
                 robustness_margin: float = 0.20,
                 variance_amplifier: float = 0.495,  # ★ 新参数
                 **kw):
        super().__init__(engine, **kw)
        self._base_margin = robustness_margin
        self._margin = robustness_margin
        self._var_amp = variance_amplifier
        # ★ Welford 在线方差追踪
        self._cost_n = 0
        self._cost_mean = 0.0
        self._cost_m2 = 0.0

    @property
    def name(self) -> str:
        return "PAR2QO-Enhanced"

    def _update_variance(self, cost_us: float):
        _dbg("_UPDATE_", f"_update_variance(cost_us={cost_us})")
        self._cost_n += 1
        delta = cost_us - self._cost_mean
        self._cost_mean += delta / self._cost_n
        delta2 = cost_us - self._cost_mean
        self._cost_m2 += delta * delta2

    @property
    def _cost_cv(self) -> float:
        """变异系数 = std / mean — 代价估计的不稳定程度."""
        if self._cost_n < 2 or self._cost_mean <= 0:
            return 0.0
        var = self._cost_m2 / (self._cost_n - 1)
        return math.sqrt(var) / self._cost_mean

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        self._query_count += 1
        estimates = self._engine.estimate_all_devices(query, data_location)

        gpu_ids = [k for k, n in self._engine.topology.nodes.items()
                   if n.kind == HardwareKind.GPU and k in estimates]
        cpu_ids = [k for k, n in self._engine.topology.nodes.items()
                   if n.kind == HardwareKind.CPU and k in estimates]

        if not gpu_ids or not cpu_ids:
            device_id, cb = self._engine.recommend(query, data_location)
            return RoutingDecision(
                query_id=query.query_id, device_id=device_id, cost=cb,
                confidence=0.495, metadata={"reason": "fallback"},
                trace_log=["incomplete topology → fallback"],
            )

        best_gpu = min(gpu_ids, key=lambda k: estimates[k].total_us)
        best_cpu = min(cpu_ids, key=lambda k: estimates[k].total_us)
        gpu_cost = estimates[best_gpu].total_us
        cpu_cost = estimates[best_cpu].total_us

        self._update_variance(gpu_cost)
        self._update_variance(cpu_cost)
        # ★ 改写: 自适应 margin
        self._margin = self._base_margin + self._var_amp * self._cost_cv

        if gpu_cost < cpu_cost * (1.0 - self._margin):
            chosen = best_gpu
            reason = "gpu_clear_winner"
        else:
            chosen = best_cpu
            reason = "cpu_robust_choice"

        gpu_adv = (1.0 - gpu_cost / cpu_cost) * 100 if cpu_cost > 0 else 0.0
        trace = [
            f"gpu={gpu_cost:.1f}µs cpu={cpu_cost:.1f}µs",
            f"margin={self._margin:.3f} (base={self._base_margin}, cv={self._cost_cv:.3f})",
            f"gpu_adv={gpu_adv:.1f}% → {reason}",
        ]
        return RoutingDecision(
            query_id=query.query_id, device_id=chosen,
            cost=estimates[chosen],
            confidence=1.0 if reason == "gpu_clear_winner" else 0.8,
            metadata={"reason": reason, "gpu_cost_us": gpu_cost,
                      "cpu_cost_us": cpu_cost, "margin": self._margin,
                      "gpu_advantage_pct": gpu_adv},
            trace_log=trace,
        )

    def reset(self) -> None:
        super().reset()
        self._margin = self._base_margin
        self._cost_n = 0
        self._cost_mean = 0.0
        self._cost_m2 = 0.0
