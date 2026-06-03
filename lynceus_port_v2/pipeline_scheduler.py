"""
lynceus_port/pipeline_scheduler.py — M015-M016: Query pipeline scheduler.

Architecture references:
    - Megatron forward_backward_pipelining_with_interleaving (schedules.py:896)
    - NCCL ncclTopoCompute (nccl/src/graph/search.cc:1023)
    - vLLM SchedulerOutput (vllm/v1/core/sched/output.py)

改写 ~20%:
  - decompose_query: join 阶段行流从 parity 改为 selectivity 衰减
    (每个 join 按 join_selectivity 缩减行数, 更贴近真实基数估计)
  - _critical_path: 设备切换时加 context-switch 罚分
    (DMA 重映射 + TLB flush 开销, 不只是带宽延迟)
  - schedule_pipeline: bubble 公式从均匀 stage 推广到非均匀 stage
    (用最长 stage 的比例代替 1/p, 因为瓶颈 stage 决定吞吐)
  - _compute_bubble_ratio: 异构阶段时间修正
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from .cost_model import (
    CostBreakdown,
    CostModelEngine,
    QueryDescriptor,
    QueryType,
)
from .schema import HardwareKind
from . import _dbg, _dump_obj, _snapshot, _Timer, LYNCEUS_DEBUG

_T = "PIP"


# ---------------------------------------------------------------------------
# Stage model
# ---------------------------------------------------------------------------

class StageKind(Enum):
    SCAN = auto()
    FILTER = auto()
    JOIN = auto()
    AGGREGATE = auto()
    SORT = auto()


@dataclass
class QueryStage:
    stage_id: str
    kind: StageKind
    descriptor: QueryDescriptor
    produces_rows: int


@dataclass
class StageAssignment:
    stage_id: str
    kind: StageKind
    device_id: str
    cost: CostBreakdown


@dataclass
class PipelineSchedule:
    query_id: str
    assignments: List[StageAssignment]
    compute_cost_us: float
    transfer_cost_us: float
    latency_us: float

    @property
    def devices_used(self) -> List[str]:
        seen: List[str] = []
        for a in self.assignments:
            if a.device_id not in seen:
                seen.append(a.device_id)
        return seen

    def dump_state(self) -> str:
        lines = [f"PipelineSchedule({self.query_id}): "
                 f"latency={self.latency_us:.2f}us "
                 f"(compute={self.compute_cost_us:.2f} + "
                 f"transfer={self.transfer_cost_us:.2f})"]
        for a in self.assignments:
            lines.append(f"  {a.kind.name:>10s} → {a.device_id} "
                         f"total={a.cost.total_us:.2f}us "
                         f"xfer={a.cost.transfer_cost_us:.2f}us")
        return "\n".join(lines)


@dataclass
class PipelineBatchSchedule:
    num_queries: int
    num_stages: int
    serial_makespan_us: float
    pipelined_makespan_us: float
    bubble_fraction: float

    @property
    def speedup(self) -> float:
        if self.pipelined_makespan_us <= 0:
            return 1.0
        return self.serial_makespan_us / self.pipelined_makespan_us


# ---------------------------------------------------------------------------
# Stage decomposition
# ---------------------------------------------------------------------------

# 改写: join selectivity 衰减系数, 每个 join 阶段行数乘以此系数
_JOIN_SELECTIVITY: float = 0.4


def decompose_query(query: QueryDescriptor) -> List[QueryStage]:
    """Break a query into operator stages.

    改写: join 阶段的行流模型从 parity (1:1) 改为 selectivity 衰减。
    原版每个 join 输出行数 = 输入行数 (假设 PK join), 对 multi-way join
    会让后续 stage 的 cost 虚高。改为每个 join 按 _JOIN_SELECTIVITY
    衰减, 更接近 TPC-H 里 lineitem JOIN orders 这种实际比例。
    """
    stages: List[QueryStage] = []

    scan_rows = max(1, int(query.selectivity * query.table_rows))
    base = dict(
        estimated_width_bytes=query.estimated_width_bytes,
        table_rows=query.table_rows,
        index_available=query.index_available,
        index_depth=query.index_depth,
        table_name=query.table_name,
    )

    def mk(kind: StageKind, qtype: QueryType, in_rows: int,
           selectivity: float, **extra) -> QueryStage:
        desc = QueryDescriptor(
            query_id=f"{query.query_id}::{kind.name.lower()}",
            query_type=qtype,
            estimated_rows=in_rows,
            num_predicates=query.num_predicates,
            selectivity=min(1.0, max(0.0, selectivity)),
            **base,
            **extra,
        )
        return QueryStage(
            stage_id=desc.query_id,
            kind=kind,
            descriptor=desc,
            produces_rows=in_rows,
        )

    # 1. SCAN
    scan_type = (QueryType.INDEX_SCAN if query.index_available
                 else QueryType.FULL_TABLE_SCAN)
    stages.append(mk(StageKind.SCAN, scan_type, scan_rows, query.selectivity))

    # 2. FILTER
    filtered = max(1, query.estimated_rows or scan_rows)
    if query.num_predicates > 0 and filtered < scan_rows:
        st = mk(StageKind.FILTER, QueryType.RANGE_SCAN, scan_rows,
                filtered / max(1, scan_rows))
        st.produces_rows = filtered
        stages.append(st)
    else:
        filtered = scan_rows

    # 3. JOIN — 改写: 每个 join 按 selectivity 衰减
    join_rows = filtered
    for j in range(int(max(0, query.num_joins))):
        st = mk(StageKind.JOIN, QueryType.JOIN, join_rows, _JOIN_SELECTIVITY,
                num_joins=1)
        # 改写: 输出行数按 selectivity 衰减, 不再 parity
        out_rows = max(1, int(join_rows * _JOIN_SELECTIVITY))
        st.produces_rows = out_rows
        stages.append(st)
        join_rows = out_rows

    # 4. AGGREGATE
    if query.group_by_cardinality > 0:
        gb = max(1, query.group_by_cardinality)
        st = mk(StageKind.AGGREGATE, QueryType.AGGREGATE, join_rows,
                min(1.0, gb / max(1, join_rows)),
                group_by_cardinality=gb)
        st.produces_rows = gb
        stages.append(st)
        join_rows = gb

    # 5. SORT
    if query.sort_required:
        st = mk(StageKind.SORT, QueryType.SORT, join_rows, 1.0,
                sort_required=True)
        st.produces_rows = join_rows
        stages.append(st)

    _dbg(_T, f"decompose({query.query_id}): {len(stages)} stages, "
         f"row_flow={[s.produces_rows for s in stages]}")
    return stages


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

# 改写: 设备切换的 context-switch 固定罚分 (DMA 重映射 + TLB flush)
_DEVICE_SWITCH_PENALTY_US: float = 5.0


class QueryPipelineScheduler:
    """Segment a query into stages and pipeline across devices."""

    def __init__(self, engine: CostModelEngine,
                 max_pipeline_depth: Optional[int] = None):
        self.engine = engine
        n_devices = sum(
            1 for n in engine.topology.nodes.values()
            if n.kind in (HardwareKind.GPU, HardwareKind.CPU)
        )
        self.max_pipeline_depth = max_pipeline_depth or max(1, n_devices)
        _dbg(_T, f"PipelineScheduler: max_depth={self.max_pipeline_depth}, "
             f"n_devices={n_devices}")

    def assign_stages(self, stages: List[QueryStage],
                      data_location: str = "cpu0"
                      ) -> List[StageAssignment]:
        assignments: List[StageAssignment] = []
        current_location = data_location
        for st in stages:
            with _Timer(f"assign:{st.kind.name}") as t:
                device_id, cost = self.engine.recommend(
                    st.descriptor, data_location=current_location
                )
            assignments.append(StageAssignment(
                stage_id=st.stage_id,
                kind=st.kind,
                device_id=device_id,
                cost=cost,
            ))
            _dbg(_T, f"  {st.kind.name}: {current_location} → {device_id}, "
                 f"cost={cost.total_us:.2f}us")
            current_location = device_id
        return assignments

    @staticmethod
    def _critical_path(assignments: List[StageAssignment]
                       ) -> Tuple[float, float, float]:
        """改写: 加 context-switch 罚分.

        原版只算 transfer_cost_us + compute, 但真实系统里设备切换
        还有 DMA context 重建、TLB flush、PCIe BAR 重映射等固定开销,
        不纯粹是带宽×数据量。每次设备切换加一个固定罚分。
        """
        compute_us = 0.0
        transfer_us = 0.0
        prev_device = None
        switch_count = 0
        for a in assignments:
            transfer_us += a.cost.transfer_cost_us
            compute_us += (a.cost.total_us - a.cost.transfer_cost_us)
            # 改写: 设备切换罚分
            if prev_device is not None and a.device_id != prev_device:
                transfer_us += _DEVICE_SWITCH_PENALTY_US
                switch_count += 1
            prev_device = a.device_id

        total = compute_us + transfer_us
        _dbg(_T, f"critical_path: compute={compute_us:.2f} "
             f"transfer={transfer_us:.2f} switches={switch_count} "
             f"total={total:.2f}us")
        return compute_us, transfer_us, total

    def schedule(self, query: QueryDescriptor,
                 data_location: str = "cpu0") -> PipelineSchedule:
        with _Timer(f"schedule:{query.query_id}") as t:
            stages = decompose_query(query)
            assignments = self.assign_stages(stages, data_location)
            compute, transfer, latency = self._critical_path(assignments)

        sched = PipelineSchedule(
            query_id=query.query_id,
            assignments=assignments,
            compute_cost_us=compute,
            transfer_cost_us=transfer,
            latency_us=latency,
        )
        _dbg(_T, f"schedule result: {sched.dump_state()}")
        return sched

    def schedule_batch(self, queries: List[QueryDescriptor],
                       data_location: str = "cpu0"
                       ) -> List[PipelineSchedule]:
        return [self.schedule(q, data_location) for q in queries]

    def schedule_pipeline(self, queries: List[QueryDescriptor],
                          data_location: str = "cpu0"
                          ) -> PipelineBatchSchedule:
        """改写: 非均匀 stage 的加权 bubble 公式.

        原版 Megatron 公式 bubble = (p-1)/(m+p-1) 假设所有 stage 耗时
        相等。实际上异构设备 + 不同 operator 导致 stage 时间差异很大,
        瓶颈 stage 决定吞吐。

        修正: 设 t_max 为所有 query 中最慢 stage 的时间,
        t_avg 为平均 stage 时间, imbalance = t_max / t_avg。
        t_pipe = t_serial * (m + p - 1) / (m * p) * imbalance
        当所有 stage 等长时 imbalance=1 退化为原公式。
        """
        if not queries:
            return PipelineBatchSchedule(0, 0, 0.0, 0.0, 0.0)

        schedules = [self.schedule(q, data_location) for q in queries]
        t_serial = sum(s.latency_us for s in schedules)

        m = len(queries)
        max_stages = max(len(s.assignments) for s in schedules)
        p = min(max_stages, self.max_pipeline_depth)
        p = max(1, p)

        bubble = (p - 1) / (m + p - 1)

        # 改写: 计算 stage imbalance factor
        all_stage_times = []
        for s in schedules:
            for a in s.assignments:
                all_stage_times.append(a.cost.total_us)
        if all_stage_times:
            t_max = max(all_stage_times)
            t_avg = sum(all_stage_times) / len(all_stage_times)
            imbalance = t_max / t_avg if t_avg > 0 else 1.0
        else:
            imbalance = 1.0

        t_pipe = t_serial * (m + p - 1) / (m * p) * imbalance

        _dbg(_T, f"pipeline: m={m} p={p} bubble={bubble:.4f} "
             f"imbalance={imbalance:.3f} serial={t_serial:.1f}us "
             f"pipelined={t_pipe:.1f}us "
             f"speedup={t_serial/t_pipe if t_pipe > 0 else 1:.2f}x")

        return PipelineBatchSchedule(
            num_queries=m,
            num_stages=p,
            serial_makespan_us=t_serial,
            pipelined_makespan_us=t_pipe,
            bubble_fraction=bubble,
        )
