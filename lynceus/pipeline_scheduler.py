"""
lynceus/pipeline_scheduler.py — Query pipeline scheduler.

算法改动:
    1. decompose_query 的 row-flow 注入 cardinality estimation 误差
       (log-normal 噪声, σ=0.15), 模拟真实查询优化器的 CE 偏差
    2. schedule_pipeline 的 bubble fraction 加 memory-contention 修正:
       当 Σ(stage_data) / total_hbm > 0.7 时, 按超出比例增加 bubble
"""
from __future__ import annotations
import math
import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
from .cost_model import CostBreakdown, CostModelEngine, QueryDescriptor, QueryType
from .schema import HardwareKind


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


@dataclass
class PipelineBatchSchedule:
    num_queries: int
    num_stages: int
    serial_makespan_us: float
    pipelined_makespan_us: float
    bubble_fraction: float
    memory_pressure: float = 0.0  # 改动: 新增
    @property
    def speedup(self) -> float:
        if self.pipelined_makespan_us <= 0:
            return 1.0
        return self.serial_makespan_us / self.pipelined_makespan_us


def _ce_noise(query_id: str, stage_name: str, sigma: float = 0.15) -> float:
    """确定性 cardinality estimation 噪声 (同一 query+stage 总得到同一噪声)。
    用 query_id+stage_name 的 hash 做种子, 生成 log-normal 乘数。
    σ=0.15 → 大约 68% 的 CE 误差在 0.86x ~ 1.16x 之间。
    """
    h = hashlib.md5(f"{query_id}:{stage_name}".encode()).hexdigest()
    # 用前8字节做确定性 uniform [0,1)
    u1 = int(h[:8], 16) / 0xFFFFFFFF
    u2 = int(h[8:16], 16) / 0xFFFFFFFF
    # Box-Muller
    u1 = max(1e-10, min(1 - 1e-10, u1))
    u2 = max(1e-10, min(1 - 1e-10, u2))
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return math.exp(sigma * z)


def decompose_query(query: QueryDescriptor,
                    ce_sigma: float = 0.15) -> List[QueryStage]:
    """改动: row-flow 注入 CE 噪声, 每个 stage 的 in_rows 乘以 log-normal 因子。"""
    stages: List[QueryStage] = []
    scan_rows = max(1, int(query.selectivity * query.table_rows))
    # 改动: CE噪声
    scan_rows = max(1, int(scan_rows * _ce_noise(query.query_id, "scan", ce_sigma)))

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
    filtered = max(1, int(filtered * _ce_noise(query.query_id, "filter", ce_sigma)))
    if query.num_predicates > 0 and filtered < scan_rows:
        st = mk(StageKind.FILTER, QueryType.RANGE_SCAN, scan_rows,
                filtered / max(1, scan_rows))
        st.produces_rows = filtered
        stages.append(st)
    else:
        filtered = scan_rows

    join_rows = filtered
    for j in range(max(0, query.num_joins)):
        noise = _ce_noise(query.query_id, f"join_{j}", ce_sigma)
        join_rows_noisy = max(1, int(join_rows * noise))
        st = mk(StageKind.JOIN, QueryType.JOIN, join_rows_noisy, 1.0, num_joins=1)
        st.produces_rows = join_rows_noisy
        stages.append(st)
        join_rows = join_rows_noisy

    if query.group_by_cardinality > 0:
        gb = max(1, query.group_by_cardinality)
        gb = max(1, int(gb * _ce_noise(query.query_id, "agg", ce_sigma)))
        st = mk(StageKind.AGGREGATE, QueryType.AGGREGATE, join_rows,
                min(1.0, gb / max(1, join_rows)), group_by_cardinality=gb)
        st.produces_rows = gb
        stages.append(st)
        join_rows = gb

    if query.sort_required:
        st = mk(StageKind.SORT, QueryType.SORT, join_rows, 1.0, sort_required=True)
        st.produces_rows = join_rows
        stages.append(st)

    return stages


class QueryPipelineScheduler:
    def __init__(self, engine: CostModelEngine,
                 max_pipeline_depth: Optional[int] = None):
        self.engine = engine
        n_devices = sum(1 for n in engine.topology.nodes.values()
                        if n.kind in (HardwareKind.GPU, HardwareKind.CPU))
        self.max_pipeline_depth = max_pipeline_depth or max(1, n_devices)
        # 收集 total HBM 用于 memory pressure 计算
        self._total_hbm = sum(
            n.memory_bytes for n in engine.topology.nodes.values()
            if n.kind == HardwareKind.GPU
        )

    def assign_stages(self, stages: List[QueryStage],
                      data_location: str = "cpu0") -> List[StageAssignment]:
        assignments: List[StageAssignment] = []
        current_location = data_location
        for st in stages:
            device_id, cost = self.engine.recommend(
                st.descriptor, data_location=current_location)
            assignments.append(StageAssignment(
                stage_id=st.stage_id, kind=st.kind,
                device_id=device_id, cost=cost))
            current_location = device_id
        return assignments

    @staticmethod
    def _critical_path(assignments: List[StageAssignment]) -> Tuple[float, float, float]:
        compute_us = 0.0
        transfer_us = 0.0
        for a in assignments:
            transfer_us += a.cost.transfer_cost_us
            compute_us += (a.cost.total_us - a.cost.transfer_cost_us)
        return compute_us, transfer_us, compute_us + transfer_us

    def schedule(self, query: QueryDescriptor,
                 data_location: str = "cpu0") -> PipelineSchedule:
        from ._debug import dbg
        dbg('Pipeline.schedule_start', query_id=query.query_id)
        stages = decompose_query(query)
        assignments = self.assign_stages(stages, data_location)
        compute, transfer, latency = self._critical_path(assignments)
        dbg('Pipeline.schedule_done', query_id=query.query_id,
            n_stages=len(assignments), total_us=latency)
        return PipelineSchedule(
            query_id=query.query_id, assignments=assignments,
            compute_cost_us=compute, transfer_cost_us=transfer,
            latency_us=latency)

    def schedule_batch(self, queries: List[QueryDescriptor],
                       data_location: str = "cpu0") -> List[PipelineSchedule]:
        return [self.schedule(q, data_location) for q in queries]

    def schedule_pipeline(self, queries: List[QueryDescriptor],
                          data_location: str = "cpu0") -> PipelineBatchSchedule:
        """改动: bubble fraction 加 memory-contention 修正。"""
        from ._debug import checkpoint
        if not queries:
            return PipelineBatchSchedule(0, 0, 0.0, 0.0, 0.0)

        schedules = [self.schedule(q, data_location) for q in queries]
        t_serial = sum(s.latency_us for s in schedules)

        m = len(queries)
        max_stages = max(len(s.assignments) for s in schedules)
        p = min(max_stages, self.max_pipeline_depth)
        p = max(1, p)

        raw_bubble = (p - 1) / (m + p - 1)
        sync_tax = 0.02 * (p - 1)

        # 改动: memory contention — 所有 stage 的数据量 / total HBM
        total_data_bytes = sum(
            a.cost.transfer_cost_us  # 近似: transfer_cost ∝ data size
            for s in schedules for a in s.assignments
        )
        # 换算回 bytes (粗略: 1µs transfer ≈ 32KB @ 32GB/s PCIe)
        approx_bytes = total_data_bytes * 32 * 1024
        mem_pressure = approx_bytes / max(1, self._total_hbm) if self._total_hbm > 0 else 0.0
        mem_penalty = max(0.0, (mem_pressure - 0.7) * 0.3) if mem_pressure > 0.7 else 0.0

        bubble = min(raw_bubble + sync_tax + mem_penalty, 0.95)
        t_pipe = t_serial * (m + p - 1) / (m * p)
        # 把 memory penalty 加到实际 makespan 上
        t_pipe *= (1.0 + mem_penalty)

        checkpoint("pipeline_batch", m=m, p=p, raw_bubble=raw_bubble,
                   mem_pressure=mem_pressure, mem_penalty=mem_penalty,
                   t_serial_us=t_serial, t_pipe_us=t_pipe)

        return PipelineBatchSchedule(
            num_queries=m, num_stages=p,
            serial_makespan_us=t_serial,
            pipelined_makespan_us=t_pipe,
            bubble_fraction=bubble,
            memory_pressure=mem_pressure)
