"""
lynceus_port/strategies/static.py — 静态路由策略.

改写: HybridStaticStrategy 增加自适应阈值衰减 —
      阈值随查询数缓慢降低, 越来越倾向 GPU.
"""
from __future__ import annotations
from typing import Optional
from ..cost_model import CostModelEngine, QueryDescriptor
from ..schema import HardwareKind
from .base import RoutingDecision, RoutingStrategyBase
from .. import _dbg

_MOD_TAG = "STC"
import os as _os, sys as _sys
_LYNCEUS_DBG = _os.environ.get("LYNCEUS_DEBUG", "1")

def _dbg(tag: str, msg: str):
    if _LYNCEUS_DBG != "0":
        print(f"[{_MOD_TAG}·{tag}] {msg}", file=_sys.stderr, flush=True)

_tr = _dbg  # 兼容旧调用



class GPUOnlyStrategy(RoutingStrategyBase):
    def __init__(self, engine: CostModelEngine, *, gpu_id: str = "gpu0", **kw):
        super().__init__(engine, **kw)
        self._gpu_id = gpu_id

    @property
    def name(self) -> str:
        return "GPU-Only"

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        self._query_count += 1
        cb = self._engine.estimate_on_device(query, self._gpu_id, data_location)
        return RoutingDecision(
            query_id=query.query_id, device_id=self._gpu_id, cost=cb,
            confidence=1.0, metadata={"reason": "fixed_gpu"},
            trace_log=[f"GPU-Only: always {self._gpu_id}"],
        )


class CPUOnlyStrategy(RoutingStrategyBase):
    def __init__(self, engine: CostModelEngine, *, cpu_id: str = "cpu0", **kw):
        super().__init__(engine, **kw)
        self._cpu_id = cpu_id

    @property
    def name(self) -> str:
        return "CPU-Only"

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        self._query_count += 1
        cb = self._engine.estimate_on_device(query, self._cpu_id, data_location)
        return RoutingDecision(
            query_id=query.query_id, device_id=self._cpu_id, cost=cb,
            confidence=1.0, metadata={"reason": "fixed_cpu"},
            trace_log=[f"CPU-Only: always {self._cpu_id}"],
        )


class HybridStaticStrategy(RoutingStrategyBase):
    """阈值路由 — ★ 改写: 阈值随查询数衰减 (模拟预热学习)."""

    def __init__(self, engine: CostModelEngine, *,
                 gpu_threshold_rows: int = 100_000,
                 gpu_id: str = "gpu0", cpu_id: str = "cpu0",
                 threshold_decay: float = 0.9995,  # ★ 新参数
                 **kw):
        super().__init__(engine, **kw)
        self._initial_threshold = gpu_threshold_rows
        self._threshold = float(gpu_threshold_rows)
        self._gpu_id = gpu_id
        self._cpu_id = cpu_id
        self._decay = threshold_decay

    @property
    def name(self) -> str:
        return "Hybrid-Static"

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        self._query_count += 1
        # ★ 改写: 阈值衰减
        self._threshold *= self._decay
        effective_threshold = max(1000, self._threshold)  # 下限 1000 行

        if query.estimated_rows > effective_threshold:
            device = self._gpu_id
            reason = "rows_above_threshold"
        else:
            device = self._cpu_id
            reason = "rows_below_threshold"

        cb = self._engine.estimate_on_device(query, device, data_location)
        trace = [
            f"threshold={effective_threshold:.0f} (initial={self._initial_threshold})",
            f"rows={query.estimated_rows} → {reason} → {device}",
        ]
        return RoutingDecision(
            query_id=query.query_id, device_id=device, cost=cb,
            confidence=1.0,
            metadata={"reason": reason, "threshold": effective_threshold,
                      "estimated_rows": query.estimated_rows},
            trace_log=trace,
        )

    def reset(self) -> None:
        super().reset()
        self._threshold = float(self._initial_threshold)


# ─── 静态策略调试 ────────────────────────────────────────────────
def _dump_static_strategy(strategy, label=""):
    """打印静态策略快照."""
    import sys
    print(f"[STR·STATIC] {label}: target={getattr(strategy, 'target_device', '?')}", 
          file=sys.stderr, flush=True)
