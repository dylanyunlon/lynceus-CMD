"""
lynceus_port/pipeline_scheduler.py — 查询流水线调度器。

移植自 lynceus/pipeline_scheduler.py，修改约20%:
  - schedule(): 自动打印阶段分解和设备分配的完整 debug dump
  - schedule_pipeline: bubble fraction 公式加入通信开销修正项
  - PipelineSchedule: 新增 stage_summary() 方法
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from .cost_model import (
    CostBreakdown, CostModelEngine, QueryDescriptor, QueryType,
)
from .schema import HardwareKind, _dbg


class StageKind(Enum):
    SCAN      = auto()
    FILTER    = auto()
    JOIN      = auto()
    AGGREGATE = auto()
    SORT      = auto()


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

    def stage_summary(self) -> str:
        """逐阶段摘要——适合断点调试时打印"""
        lines = [f"Pipeline({self.query_id}): {self.latency_us:.1f}us total"]
        for a in self.assignments:
            lines.append(
                f"  {a.kind.name:>10} -> {a.device_id}: "
                f"comp={a.cost.compute_cost_us:.1f} "
                f"xfer={a.cost.transfer_cost_us:.1f} "
                f"total={a.cost.total_us:.1f}us")
        lines.append(f"  devices: {self.devices_used}")
        s = "\n".join(lines)
        _dbg("Pipeline", s)
        return s


@dataclass
class PipelineBatchSchedule:
    num_queries: int
    num_stages: int
    serial_makespan_us: float
    pipelined_makespan_us: float
    bubble_fraction: float
    # ── 新增：通信开销占比 ──
    comm_overhead_fraction: float = 0.0

    @property
    def speedup(self) -> float:
        if self.pipelined_makespan_us <= 0:
            return 1.0
        return self.serial_makespan_us / self.pipelined_makespan_us

    def debug_snapshot(self) -> str:
        s = (f"BatchPipeline: m={self.num_queries} p={self.num_stages} "
             f"serial={self.serial_makespan_us:.0f}us "
             f"piped={self.pipelined_makespan_us:.0f}us "
             f"speedup={self.speedup:.2f}x "
             f"bubble={self.bubble_fraction:.3f} "
             f"comm_overhead={self.comm_overhead_fraction:.3f}")
        _dbg("BatchPipe", s)
        return s


# ---------------------------------------------------------------------------
# 阶段分解
# ---------------------------------------------------------------------------

def decompose_query(query: QueryDescriptor) -> List[QueryStage]:
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
            query_type=qtype, estimated_rows=in_rows,
            num_predicates=query.num_predicates,
            selectivity=min(1.0, max(0.0, selectivity)),
            **base, **extra,
        )
        return QueryStage(stage_id=desc.query_id, kind=kind,
                          descriptor=desc, produces_rows=in_rows)

    scan_type = (QueryType.INDEX_SCAN if query.index_available
                 else QueryType.FULL_TABLE_SCAN)
    stages.append(mk(StageKind.SCAN, scan_type, scan_rows, query.selectivity))

    filtered = max(1, query.estimated_rows or scan_rows)
    if query.num_predicates > 0 and filtered < scan_rows:
        st = mk(StageKind.FILTER, QueryType.RANGE_SCAN, scan_rows,
                filtered / max(1, scan_rows))
        st.produces_rows = filtered
        stages.append(st)
    else:
        filtered = scan_rows

    join_rows = filtered
    for _ in range(max(0, query.num_joins)):
        st = mk(StageKind.JOIN, QueryType.JOIN, join_rows, 1.0, num_joins=1)
        st.produces_rows = join_rows
        stages.append(st)

    if query.group_by_cardinality > 0:
        gb = max(1, query.group_by_cardinality)
        st = mk(StageKind.AGGREGATE, QueryType.AGGREGATE, join_rows,
                min(1.0, gb / max(1, join_rows)),
                group_by_cardinality=gb)
        st.produces_rows = gb
        stages.append(st)
        join_rows = gb

    if query.sort_required:
        st = mk(StageKind.SORT, QueryType.SORT, join_rows, 1.0,
                sort_required=True)
        st.produces_rows = join_rows
        stages.append(st)

    _dbg("Decompose",
         f"{query.query_id}: {len(stages)} stages "
         f"[{', '.join(s.kind.name for s in stages)}]")
    return stages


# ---------------------------------------------------------------------------
# 调度器
# ---------------------------------------------------------------------------

class QueryPipelineScheduler:
    def __init__(self, engine: CostModelEngine,
                 max_pipeline_depth: Optional[int] = None):
        self.engine = engine
        n_devices = sum(
            1 for n in engine.topology.nodes.values()
            if n.kind in (HardwareKind.GPU, HardwareKind.CPU)
        )
        self.max_pipeline_depth = max_pipeline_depth or max(1, n_devices)

    def assign_stages(self, stages: List[QueryStage],
                      data_location: str = "cpu0"
                      ) -> List[StageAssignment]:
        assignments: List[StageAssignment] = []
        current_location = data_location
        for st in stages:
            device_id, cost = self.engine.recommend(
                st.descriptor, data_location=current_location)
            assignments.append(StageAssignment(
                stage_id=st.stage_id, kind=st.kind,
                device_id=device_id, cost=cost,
            ))
            _dbg("Assign",
                 f"  {st.kind.name} -> {device_id} "
                 f"(from {current_location}, cost={cost.total_us:.1f}us)")
            current_location = device_id
        return assignments

    @staticmethod
    def _critical_path(assignments: List[StageAssignment]
                       ) -> Tuple[float, float, float]:
        compute_us = 0.0
        transfer_us = 0.0
        for a in assignments:
            transfer_us += a.cost.transfer_cost_us
            compute_us += (a.cost.total_us - a.cost.transfer_cost_us)
        return compute_us, transfer_us, compute_us + transfer_us

    def schedule(self, query: QueryDescriptor,
                 data_location: str = "cpu0") -> PipelineSchedule:
        stages = decompose_query(query)
        assignments = self.assign_stages(stages, data_location)
        compute, transfer, latency = self._critical_path(assignments)
        sched = PipelineSchedule(
            query_id=query.query_id, assignments=assignments,
            compute_cost_us=compute, transfer_cost_us=transfer,
            latency_us=latency,
        )
        sched.stage_summary()  # 自动 debug dump
        return sched

    def schedule_batch(self, queries: List[QueryDescriptor],
                       data_location: str = "cpu0"
                       ) -> List[PipelineSchedule]:
        return [self.schedule(q, data_location) for q in queries]

    def schedule_pipeline(self, queries: List[QueryDescriptor],
                          data_location: str = "cpu0"
                          ) -> PipelineBatchSchedule:
        if not queries:
            return PipelineBatchSchedule(0, 0, 0.0, 0.0, 0.0)

        schedules = [self.schedule(q, data_location) for q in queries]
        t_serial = sum(s.latency_us for s in schedules)
        total_xfer = sum(s.transfer_cost_us for s in schedules)

        m = len(queries)
        max_stages = max(len(s.assignments) for s in schedules)
        p = min(max_stages, self.max_pipeline_depth)
        p = max(1, p)

        bubble = (p - 1) / (m + p - 1)

        # ── 修改：通信开销修正——设备切换越多，bubble 越大 ──
        comm_fraction = total_xfer / max(t_serial, 1e-9)
        adjusted_bubble = bubble + comm_fraction * 0.1  # 通信增加10%等效bubble
        adjusted_bubble = min(adjusted_bubble, 0.95)  # 上界

        t_pipe = t_serial * (m + p - 1) / (m * p)
        # 修正后的流水线时间
        t_pipe_adj = t_pipe * (1.0 + comm_fraction * 0.1)

        result = PipelineBatchSchedule(
            num_queries=m, num_stages=p,
            serial_makespan_us=t_serial,
            pipelined_makespan_us=t_pipe_adj,
            bubble_fraction=adjusted_bubble,
            comm_overhead_fraction=comm_fraction,
        )
        result.debug_snapshot()
        return result
