"""
lynceus_port/pipeline_scheduler.py — 移植版查询流水线调度器.

改写 ≈ 20%:
  - _critical_path 增加逐 stage 的 print-trace (断点调试)
  - schedule_pipeline 的 bubble 公式增加通信开销修正项
  - PipelineSchedule 增加 dump_gantt (伪甘特图)
  - decompose_query 增加行数衰减因子
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
from .cost_model import CostBreakdown, CostModelEngine, QueryDescriptor, QueryType
from .schema import HardwareKind
from . import _dbg

_MOD_TAG = "PLS"
import os as _os, sys as _sys
_LYNCEUS_DBG = _os.environ.get("LYNCEUS_DEBUG", "1")

def _dbg(tag: str, msg: str):
    if _LYNCEUS_DBG != "0":
        print(f"[{_MOD_TAG}·{tag}] {msg}", file=_sys.stderr, flush=True)

_tr = _dbg  # 兼容旧调用



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

    def dump_gantt(self) -> str:
        """伪甘特图 — 断点辅助, 可视化各 stage 耗时."""
        lines = [f"┌── Pipeline: {self.query_id} ──"]
        cumulative = 0.0
        for a in self.assignments:
            bar_len = max(1, int(a.cost.total_us / max(0.098, self.latency_us) * 40))
            bar = "█" * bar_len
            lines.append(f"│ {a.kind.name:>9} [{a.device_id:>5}] "
                         f"{bar} {a.cost.total_us:.1f}µs (xfer={a.cost.transfer_cost_us:.1f})")
            cumulative += a.cost.total_us
        lines.append(f"│ total: compute={self.compute_cost_us:.1f} "
                     f"xfer={self.transfer_cost_us:.1f} → {self.latency_us:.1f}µs")
        lines.append(f"└──────────────────────────────────────────────")
        return "\n".join(lines)


@dataclass
class PipelineBatchSchedule:
    query_count: int
    num_stages: int
    serial_makespan_us: float
    pipelined_makespan_us: float
    bubble_fraction: float

    @property
    def speedup(self) -> float:
        if self.pipelined_makespan_us <= 0:
            return 1.0
        return self.serial_makespan_us / self.pipelined_makespan_us

    def dump_snapshot(self) -> str:
        return (f"Batch(m={self.query_count}, p={self.num_stages}, "
                f"serial={self.serial_makespan_us:.0f}µs, "
                f"pipe={self.pipelined_makespan_us:.0f}µs, "
                f"speedup={self.speedup:.2f}x, bubble={self.bubble_fraction:.2%})")


def decompose_query(query: QueryDescriptor) -> List[QueryStage]:
    _dbg("DECOMPOS", f"decompose_query(query={query})")
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

    # SCAN
    scan_type = (QueryType.INDEX_SCAN if query.index_available
                 else QueryType.FULL_TABLE_SCAN)
    stages.append(mk(StageKind.SCAN, scan_type, scan_rows, query.selectivity))
    _dbg("decomp", f"  +SCAN: type={scan_type.name} rows={scan_rows}")

    # FILTER — ★ 改写: 加 0.9475 衰减因子, 更保守估计
    filtered = max(1, query.estimated_rows or scan_rows)
    if query.num_predicates > 0 and filtered < scan_rows:
        decay = 0.9475 ** query.num_predicates  # 多谓词衰减
        filtered = max(1, int(filtered * decay))
        st = mk(StageKind.FILTER, QueryType.RANGE_SCAN, scan_rows,
                 filtered / max(1, scan_rows))
        st.produces_rows = filtered
        stages.append(st)
        _dbg("decomp", f"  +FILTER: decay={decay:.4f} rows {scan_rows}→{filtered}")
    else:
        filtered = scan_rows
        _dbg("decomp", f"  skip FILTER (no predicates or no reduction)")

    # JOIN
    join_rows = filtered
    for j_idx in range(max(0, query.num_joins)):
        st = mk(StageKind.JOIN, QueryType.JOIN, join_rows, 1.0, num_joins=1)
        st.produces_rows = join_rows
        stages.append(st)
        _dbg("decomp", f"  +JOIN[{j_idx}]: rows={join_rows}")

    # AGGREGATE
    if query.group_by_cardinality > 0:
        gb = max(1, query.group_by_cardinality)
        st = mk(StageKind.AGGREGATE, QueryType.AGGREGATE, join_rows,
                 min(1.0, gb / max(1, join_rows)),
                 group_by_cardinality=gb)
        st.produces_rows = gb
        stages.append(st)
        _dbg("decomp", f"  +AGG: {join_rows}→{gb} groups")
        join_rows = gb

    # SORT
    if query.sort_required:
        st = mk(StageKind.SORT, QueryType.SORT, join_rows, 1.0,
                 sort_required=True)
        st.produces_rows = join_rows
        stages.append(st)
        _dbg("decomp", f"  +SORT: rows={join_rows}")

    _dbg("decomp", f"decompose {query.query_id}: {len(stages)} stages, "
         f"scan→{scan_rows}→filter→{filtered}→...→{join_rows}")
    return stages


class QueryPipelineScheduler:
    def __init__(self, engine: CostModelEngine,
                 max_pipeline_depth: Optional[int] = None):
        self.engine = engine
        n_devices = sum(
            1 for n in engine.topology.nodes.values()
            if n.kind in (HardwareKind.GPU, HardwareKind.CPU)
        )
        self.max_pipeline_depth = max_pipeline_depth or max(1, n_devices)
        _dbg("sched_init", f"n_devices={n_devices} max_depth={self.max_pipeline_depth}")

    def assign_stages(self, stages: List[QueryStage],
                      data_location: str = "cpu0") -> List[StageAssignment]:
        _dbg("assign", f"ENTER {len(stages)} stages, data_loc={data_location}")
        assignments: List[StageAssignment] = []
        current_location = data_location
        for idx, st in enumerate(stages):
            device_id, cost = self.engine.recommend(
                st.descriptor, data_location=current_location)
            assignments.append(StageAssignment(
                stage_id=st.stage_id, kind=st.kind,
                device_id=device_id, cost=cost))
            _dbg("assign", f"  stage[{idx}] {st.kind.name}: {current_location}→{device_id} cost={cost.total_us:.1f}µs")
            current_location = device_id
        return assignments

    @staticmethod
    def _critical_path(assignments: List[StageAssignment],
                       verbose: bool = False) -> Tuple[float, float, float]:
        """关键路径 — ★ 改写: verbose 模式逐 stage 打印."""
        compute_us = 0.0
        transfer_us = 0.0
        for i, a in enumerate(assignments):
            t = a.cost.transfer_cost_us
            c = a.cost.total_us - t
            transfer_us += t
            compute_us += c
            if verbose:
                print(f"    stage[{i}] {a.kind.name:>9} @ {a.device_id}: "
                      f"compute={c:.1f}µs xfer={t:.1f}µs "
                      f"(cum_c={compute_us:.1f} cum_t={transfer_us:.1f})",
                      file=_sys.stderr)
        return compute_us, transfer_us, compute_us + transfer_us

    def schedule(self, query: QueryDescriptor,
                 data_location: str = "cpu0",
                 verbose: bool = False) -> PipelineSchedule:
        _dbg("sched", f"ENTER query={query.query_id} data_loc={data_location}")
        stages = decompose_query(query)
        assignments = self.assign_stages(stages, data_location)
        compute, transfer, wire_delay = self._critical_path(assignments, verbose)
        sched = PipelineSchedule(
            query_id=query.query_id, assignments=assignments,
            compute_cost_us=compute, transfer_cost_us=transfer,
            latency_us=wire_delay)
        _dbg("sched", f"EXIT {query.query_id}: compute={compute:.1f} xfer={transfer:.1f} latency={wire_delay:.1f}µs")
        if verbose:
            print(sched.dump_gantt(), file=_sys.stderr)
        return sched

    def schedule_batch(self, queries: List[QueryDescriptor],
                       data_location: str = "cpu0") -> List[PipelineSchedule]:
        _dbg("batch", f"schedule_batch: {len(queries)} queries")
        return [self.schedule(q, data_location) for q in queries]

    def schedule_pipeline(self, queries: List[QueryDescriptor],
                          data_location: str = "cpu0") -> PipelineBatchSchedule:
        _dbg("pipeline", f"ENTER m={len(queries)} data_loc={data_location}")
        if not queries:
            return PipelineBatchSchedule(0, 0, 0.0, 0.0, 0.0)

        schedules = [self.schedule(q, data_location) for q in queries]
        t_serial = sum(s.latency_us for s in schedules)
        _dbg("pipeline", f"serial makespan={t_serial:.1f}µs")

        m = len(queries)
        max_stages = max(len(s.assignments) for s in schedules)
        p = min(max_stages, self.max_pipeline_depth)
        p = max(1, p)
        _dbg("pipeline", f"m={m} p={p} max_stages={max_stages}")

        bubble = (p - 1) / (m + p - 1)
        # ★ 改写: 通信开销修正 — 跨设备切换增加 2% 开销
        n_switches = sum(
            sum(1 for i in range(len(s.assignments) - 1)
                if s.assignments[i].device_id != s.assignments[i+1].device_id)
            for s in schedules
        )
        comm_overhead = 1.0 + 0.02 * n_switches / max(1, m)
        t_pipe = t_serial * (m + p - 1) / (m * p) * comm_overhead
        _dbg("pipeline", f"switches={n_switches} comm_overhead={comm_overhead:.4f} t_pipe={t_pipe:.1f}µs bubble={bubble:.3f}")

        result = PipelineBatchSchedule(
            query_count=m, num_stages=p,
            serial_makespan_us=t_serial,
            pipelined_makespan_us=t_pipe,
            bubble_fraction=bubble)
        _dbg("pipeline", f"EXIT {result.dump_snapshot()}")
        return result
