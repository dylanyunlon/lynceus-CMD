"""
lynceus_port/strategies/static.py — 移植版静态路由策略.

算法改写:
  - HybridStaticStrategy: 原版用固定行数阈值 (estimated_rows > threshold).
    移植版改为数据量阈值 (estimated_data_bytes > byte_threshold),
    因为宽表和窄表行数相同但数据量差 10x, 按字节更合理.
  - HybridStaticStrategy: 增加 query_type 感知——index_scan 即使数据量大
    也倾向 CPU (GPU 的 B-tree 索引遍历效率低于 CPU).

溯源同原版 (NCCL algo ring / Megatron static group).
"""

from __future__ import annotations

from typing import Optional

from ..cost_model import CostModelEngine, QueryDescriptor, QueryType
from ..schema import HardwareKind
from .base import RoutingDecision, RoutingStrategyBase
from .. import _dbg, _snapshot, LYNCEUS_DEBUG

_T = "STC"


class GPUOnlyStrategy(RoutingStrategyBase):

    def __init__(self, engine: CostModelEngine, *,
                 gpu_id: str = "gpu0", **kwargs):
        super().__init__(engine, **kwargs)
        self._gpu_id = gpu_id

    @property
    def name(self) -> str:
        return "GPU-Only"

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        cb = self._engine.estimate_on_device(query, self._gpu_id, data_location)
        return RoutingDecision(
            query_id=query.query_id,
            device_id=self._gpu_id,
            cost=cb,
            confidence=1.0,
            metadata={"reason": "fixed_gpu"},
        )


class CPUOnlyStrategy(RoutingStrategyBase):

    def __init__(self, engine: CostModelEngine, *,
                 cpu_id: str = "cpu0", **kwargs):
        super().__init__(engine, **kwargs)
        self._cpu_id = cpu_id

    @property
    def name(self) -> str:
        return "CPU-Only"

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        cb = self._engine.estimate_on_device(query, self._cpu_id, data_location)
        return RoutingDecision(
            query_id=query.query_id,
            device_id=self._cpu_id,
            cost=cb,
            confidence=1.0,
            metadata={"reason": "fixed_cpu"},
        )


class HybridStaticStrategy(RoutingStrategyBase):
    """[PORT] 数据量阈值 + query_type 感知.

    原版: estimated_rows > gpu_threshold_rows → GPU.
    移植版:
      1. 改按 estimated_data_bytes > gpu_threshold_bytes (字节阈值),
         因为 10M 行 x 50B/row = 500MB, 而 100K 行 x 5000B/row 也是 500MB,
         原版会把前者发 GPU、后者发 CPU, 但它们的数据量一样大.
      2. INDEX_SCAN 强制 CPU: GPU 的 B-tree 遍历需要频繁分支,
         在 SIMD 架构上效率很差, 不如让 CPU 的分支预测器处理.
    """

    def __init__(self, engine: CostModelEngine, *,
                 gpu_threshold_rows: int = 100_000,
                 gpu_threshold_bytes: int = 50_000_000,  # 50MB
                 gpu_id: str = "gpu0",
                 cpu_id: str = "cpu0", **kwargs):
        super().__init__(engine, **kwargs)
        self._threshold_rows = gpu_threshold_rows
        self._threshold_bytes = gpu_threshold_bytes
        self._gpu_id = gpu_id
        self._cpu_id = cpu_id

    @property
    def name(self) -> str:
        return "Hybrid-Static"

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:

        # [PORT] 强制 CPU: index scan 类型 (GPU B-tree 效率低)
        if query.query_type == QueryType.INDEX_SCAN and query.index_available:
            device = self._cpu_id
            reason = "index_scan_prefers_cpu"
        # [PORT] 按数据字节量判断而非行数
        elif query.estimated_data_bytes > self._threshold_bytes:
            device = self._gpu_id
            reason = "bytes_above_threshold"
        elif query.estimated_rows > self._threshold_rows:
            # 保留行数阈值作为后备
            device = self._gpu_id
            reason = "rows_above_threshold"
        else:
            device = self._cpu_id
            reason = "below_thresholds"

        cb = self._engine.estimate_on_device(query, device, data_location)

        _snapshot(_T, "hybrid_route",
                  query=query.query_id,
                  device=device,
                  reason=reason,
                  est_bytes=query.estimated_data_bytes,
                  est_rows=query.estimated_rows)

        return RoutingDecision(
            query_id=query.query_id,
            device_id=device,
            cost=cb,
            confidence=1.0,
            metadata={
                "reason": reason,
                "threshold_rows": self._threshold_rows,
                "threshold_bytes": self._threshold_bytes,
                "estimated_rows": query.estimated_rows,
                "estimated_bytes": query.estimated_data_bytes,
            },
        )
