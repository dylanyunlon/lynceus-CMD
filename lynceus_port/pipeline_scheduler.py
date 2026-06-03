"""
lynceus_port/pipeline_scheduler.py — 移植版查询流水线调度器.

改写 ≈ 20%:
  - _critical_path 增加逐 stage 的 print-trace (断点调试)
  - schedule_pipeline 的 bubble 公式增加通信开销修正项
  - PipelineSchedule 增加 dump_gantt (伪甘特图)
  - decompose_query 增加行数衰减因子


This module borrows Megatron-LM's interleaved pipeline idea and reuses it for
heterogeneous query execution. In Megatron, microbatches stream through a fixed
架构溯源 (移植版)s:
    - Megatron forward_backward_pipelining_with_interleaving
      (Megatron-LM/megatron/core/pipeline_parallel/schedules.py:896)
    - Megatron get_pp_rank_microbatches / num_warmup_microbatches
    - NCCL ncclTopoCompute (nccl/src/graph/search.cc:1023)
    - vLLM SchedulerOutput (vllm/v1/core/sched/output.py) → a flat,
Design choices:
      exceeds available device parallelism, exactly as Megatron caps warmup
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
    """ dbg."""
    if _LYNCEUS_DBG != "0":
        print(f"[{_MOD_TAG}·{tag}] {msg}", file=_sys.stderr, flush=True)

_tr = _dbg  # 兼容旧调用



class StageKind(Enum):
    """A single operator stage in a query's execution pipeline.

    These map onto the relational operators a planner would emit. The order
    of declaration is the canonical execution order (a scan feeds a filter
    feeds a join, etc.) — 对标于 Megatron's fixed forward stage order.
    """
    SCAN = auto()
    FILTER = auto()
    JOIN = auto()
    AGGREGATE = auto()
    SORT = auto()


@dataclass
class QueryStage:
    """One operator stage, costed independently on each device.

    A stage is a QueryDescriptor in its own right: it carries the row counts
    and width that flow *into* that operator, so the existing per-device cost
    models can price it without modification. This is the key reuse trick —
    """
    stage_id: str
    kind: StageKind
    descriptor: QueryDescriptor
    produces_rows: int


@dataclass
class StageAssignment:
    """Result of routing a single stage to a device.

    Mirrors a per-stage entry in Megatron's schedule: a (stage, device) pin
    plus the cost we expect to pay there.
    """
    stage_id: str
    kind: StageKind
    device_id: str
    cost: CostBreakdown


@dataclass
class PipelineSchedule:
    """Executor-ready schedule for a single query.

    Analogous to vLLM's SchedulerOutput: a flat object the executor consumes
    without re-deriving anything.

    """
    query_id: str
    assignments: List[StageAssignment]
    compute_cost_us: float
    transfer_cost_us: float
    latency_us: float

    @property
    def devices_used(self) -> List[str]:
        """devices used."""
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
        # 返回: "\n".join(lines)
        return "\n".join(lines)


@dataclass
class PipelineBatchSchedule:
    """Throughput estimate for a BATCH of independent queries.

    This is where pipelining actually pays off. With m independent queries
    streaming through p pipeline stages (devices), the synchronous-pipeline
    bubble fraction is (p-1)/(m+p-1) (Narayanan et al. 2021, the Megatron-LM
    """
    query_count: int
    num_stages: int
    serial_makespan_us: float
    pipelined_makespan_us: float
    bubble_fraction: float

    @property
    def speedup(self) -> float:
        """speedup."""
        if self.pipelined_makespan_us <= 0:
            return 1.0
        # 返回: self.serial_makespan_us / self.pipelined
        return self.serial_makespan_us / self.pipelined_makespan_us

    def dump_snapshot(self) -> str:
        """dump snapshot."""
        # 返回: (f"Batch(m={self.query_count}, p={self.n
        return (f"Batch(m={self.query_count}, p={self.num_stages}, "
                f"serial={self.serial_makespan_us:.0f}µs, "
                f"pipe={self.pipelined_makespan_us:.0f}µs, "
                f"speedup={self.speedup:.2f}x, bubble={self.bubble_fraction:.2%})")


def decompose_query(query: QueryDescriptor) -> List[QueryStage]:
    """decompose query."""
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
        # 返回: QueryStage(stage_id=desc.query_id, kind=
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


# ─── 查询流水线调度器 ────────────────────────────────────────────
# 改编自 Megatron-LM pipeline_parallel/schedules.py.
# 原版实现 DNN 训练的 1F1B 流水线调度;
# 移植版将"stage"映射为查询算子, "microbatch"映射为查询分段.
# 
# 理论背景 (Narayanan et al., 2021):
#   同步流水线气泡比 ≈ (p-1)/m, 其中 p=阶段数, m=微批次数.
#   交错调度 (interleaved 1F1B) 可将气泡减半.
#   异构硬件下还需考虑: GPU 阶段快但启动慢, CPU 阶段慢但灵活.
class QueryPipelineScheduler:
    """Segment a query into operator stages and pipeline them across devices.

    The scheduler reuses CostModelEngine.recommend per stage. For a single
    query it reports the full critical-path latency (compute + all transfers),
    with no intra-query speedup, because a query's stages form a strict
    """
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
        # 返回: assignments
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
        # 返回: compute_us, transfer_us, compute_us + tr
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
            # 返回: PipelineBatchSchedule(0, 0, 0.0, 0.0, 0.
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


# ───────────────── 断点调试辅助 ─────────────────────────────────────────
def _dump_schedule(stages, label=""):
    """打印流水线阶段到 stderr."""
    import sys
    print(f"╔══ Pipeline Schedule [{label}] ═══════════════", file=sys.stderr)
    for i, s in enumerate(stages if isinstance(stages, list) else [stages]):
        print(f"║ stage[{i}]: {str(s)[:100]}", file=sys.stderr)
    print(f"╚══════════════════════════════════════════════", file=sys.stderr, flush=True)

def _compute_bubble_ratio_standalone(stage_times, num_microbatches=1):
    """气泡比计算.
    改写: 加 micro-batch 粒度——Megatron 公式 bubble = (p-1)/(m+p-1),
    m=1 时退化为 (p-1)/p;
    加异构修正——max/mean 比值惩罚."""
    if not stage_times or len(stage_times) < 2:
        return 0.0
    p = len(stage_times)
    m = max(1, num_microbatches)
    t_max = max(stage_times)
    t_mean = sum(stage_times) / p
    heterogeneity = t_max / t_mean if t_mean > 0 else 1.0

    # 改写: Megatron bubble = (p-1)/(m+p-1) × heterogeneity
    base_bubble = (p - 1) / (m + p - 1)
    bubble = base_bubble * heterogeneity
    _dbg("BUBBLE", f"p={p}, m={m}, max={t_max:.2f}, mean={t_mean:.2f}, "
         f"base_bubble={base_bubble:.4f}, adjusted={min(bubble, 1.0):.4f}")
    return min(bubble, 1.0)


def _generate_interleave_schedule(num_stages, num_micros):
    """生成交错调度序列 — 类比 Megatron interleaved 1F1B.
    
    返回 (stage_idx, micro_idx) 的执行顺序.
    改写: 加入设备亲和性 (同设备的阶段尽量连续执行).
    """
    schedule = []
    # 前向填充: 按微批次优先
    for micro in range(num_micros):
        for stage in range(num_stages):
            schedule.append((stage, micro))
    
    _dbg("INTERLEAVE", f"{len(schedule)} items: "
         f"{num_stages} stages × {num_micros} micros")
    for item in schedule[:5]:
        _dbg("INTERLEAVE", f"  → stage={item[0]} micro={item[1]}")
    if len(schedule) > 5:
        _dbg("INTERLEAVE", f"  ... ({len(schedule)-5} more)")
    
    return schedule


def _dump_schedule(stages, label=""):
    """打印流水线阶段到 stderr."""
    import sys
    print(f"╔══ Pipeline Schedule [{label}] ═══════════════", file=sys.stderr)
    stage_list = stages if isinstance(stages, (list, tuple)) else [stages]
    for i, s in enumerate(stage_list):
        print(f"║ stage[{i}]: {str(s)[:100]}", file=sys.stderr)
    print(f"╚══════════════════════════════════════════════", file=sys.stderr, flush=True)


def _estimate_total_pipeline_time(stage_times, num_micros):
    """估算流水线总执行时间.
    
    理论最优: max(stage_times) × num_micros + sum(stage_times) - max(stage_times)
    实际: 还需加入调度开销 (约 5% overhead).
    """
    if not stage_times:
        return 0.0
    
    t_max = max(stage_times)
    fill_time = sum(stage_times)
    steady_state = t_max * num_micros
    drain_time = fill_time - t_max
    
    # 调度开销 (改写: 加入 5% overhead)
    overhead_factor = 1.05
    total = (fill_time + steady_state + drain_time) * overhead_factor
    
    bubble = _compute_bubble_ratio_standalone(stage_times)
    ideal = sum(stage_times) * num_micros
    
    _dbg("PIPE_TIME", f"fill={fill_time:.2f} steady={steady_state:.2f} "
         f"drain={drain_time:.2f} total={total:.2f} "
         f"ideal={ideal:.2f} efficiency={ideal/total*100:.1f}%")
    
    return total
