"""
lynceus_port/strategies/base.py — 移植版路由策略基类.

算法改写:
  - route_batch: 增加 batch-level costing — 记录每条决策的耗时,
    在批次结束后输出 throughput (decisions/sec) 和 P99 延迟.
  - decisions_to_latencies: 增加 IQR 异常值标记 (不剔除,只标记),
    让 benchmark 能识别哪些 query 是 outlier.
  - 新增 set_batch_hint / get_batch_hint 钩子, 供 router 的
    前瞻 hint 机制使用.

溯源同原版 (NCCL algo registry / vLLM scheduler / DeepSeek gate).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from ..cost_model import (
    CostBreakdown,
    CostModelEngine,
    QueryDescriptor,
)
from ..schema import HardwareKind
from .. import _dbg, _dump_obj, _snapshot, _Timer, LYNCEUS_DEBUG

_T = "BAS"


@dataclass
class RoutingDecision:
    query_id: str
    device_id: str
    cost: CostBreakdown
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class RoutingStrategyBase(ABC):

    def __init__(self, engine: CostModelEngine, **kwargs):
        _dbg(_T, f"__init__ called")
        self._engine = engine
        self._query_count = 0
        # [PORT] batch hint 钩子
        self._batch_hint: Optional[Dict[str, Any]] = None
        _dbg(_T, f"RoutingStrategyBase.__init__({self.__class__.__name__})")

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        ...

    def route_batch(self, queries: List[QueryDescriptor],
                    data_location: Optional[str] = None
                    ) -> List[RoutingDecision]:
        """[PORT] batch 路由 + 吞吐量/P99 统计.

        原版是纯 for 循环. 移植版记录每条决策耗时,
        批次结束后计算 throughput 和 P99, 输出到调试.
        """
        _dbg(_T, f"route_batch called")
        t0 = time.monotonic()
        per_query_ns = []
        results = []
        for q in queries:
            q_start = time.monotonic()
            results.append(self.route_one(q, data_location))
            per_query_ns.append(time.monotonic() - q_start)

        elapsed = time.monotonic() - t0
        if LYNCEUS_DEBUG and per_query_ns:
            per_query_ns.sort()
            n = len(per_query_ns)
            p99_idx = min(n - 1, int(n * 0.99))
            throughput = n / max(1e-9, elapsed)
            _dbg(_T, f"batch done: n={n}, "
                 f"throughput={throughput:.0f} q/s, "
                 f"p50={per_query_ns[n//2]*1e6:.1f}us, "
                 f"p99={per_query_ns[p99_idx]*1e6:.1f}us")

        return results

    def observe(self, query_id: str, device_id: str,
                actual_latency_us: float) -> None:
        _dbg(_T, f"observe called")
        pass

    def reset(self) -> None:
        _dbg(_T, f"reset called")
        self._query_count = 0
        self._batch_hint = None

    # [PORT] batch hint 接口
    def set_batch_hint(self, hint: Dict[str, Any]) -> None:
        """接收来自 Router 的前瞻 hint."""
        self._batch_hint = hint
        _dbg(_T, f"set_batch_hint: {hint.get('batch_size', '?')} queries")

    def get_batch_hint(self) -> Optional[Dict[str, Any]]:
        _dbg(_T, f"get_batch_hint called")
        return self._batch_hint

    @staticmethod
    def decisions_to_latencies(decisions: List[RoutingDecision]) -> List[float]:
        """提取每条决策的延迟(ms), 并标记 IQR 异常值.

        [PORT] IQR 异常值检测: 不剔除, 只在 metadata 里标记
        is_outlier=True, 供 benchmark 分析. 阈值 = Q3 + 1.5*IQR.
        """
        _dbg(_T, f"decisions_to_latencies called")
        latencies = [d.cost.total_ms for d in decisions]

        if len(latencies) >= 4:
            s = sorted(latencies)
            n = len(s)
            q1 = s[n // 4]
            q3 = s[3 * n // 4]
            iqr = q3 - q1
            upper_fence = q3 + 1.5 * iqr

            outlier_count = 0
            for d in decisions:
                if d.cost.total_ms > upper_fence:
                    d.metadata["is_outlier"] = True
                    outlier_count += 1
            if outlier_count > 0 and LYNCEUS_DEBUG:
                _dbg(_T, f"outliers: {outlier_count}/{len(decisions)} "
                     f"(fence={upper_fence:.3f}ms)")

        return latencies
