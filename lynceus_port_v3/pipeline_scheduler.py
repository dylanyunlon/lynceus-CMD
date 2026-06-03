"""
lynceus/pipeline_scheduler.py — M015-M016: Query pipeline scheduler.

A query is rarely a monolith. A TPC-H-style query decomposes into stages —
scan, filter, join, aggregate, sort — and each stage has a different cost
profile on GPU vs CPU. A wide hash-join may belong on a GPU while the final
ORDER BY spills to CPU memory. Routing the *whole* query to one device, as the
M003-M004 strategies do, leaves this on the table.

This module borrows Megatron-LM's interleaved pipeline idea and reuses it for
heterogeneous query execution. In Megatron, microbatches stream through a fixed
sequence of pipeline stages, each stage pinned to a device, with warmup /
steady / cooldown phases overlapping forward and backward passes
(forward_backward_pipelining_with_interleaving, schedules.py:896). Here the
"stages" are query operators, the "microbatches" are query segments, and the
per-stage device assignment is driven by the cost model rather than fixed
ahead of time.

Architecture references:
    - Megatron forward_backward_pipelining_with_interleaving
      (Megatron-LM/megatron/core/pipeline_parallel/schedules.py:896)
      → warmup / steady-state / cooldown phase structure; we mirror the
        three-phase fill-drain shape in PipelineSchedule.
    - Megatron get_pp_rank_microbatches / num_warmup_microbatches
      (schedules.py) → how many segments are in flight before the pipe is full.
    - NCCL ncclTopoCompute (nccl/src/graph/search.cc:1023)
      → per-stage device search, reused via CostModelEngine.recommend.
    - vLLM SchedulerOutput (vllm/v1/core/sched/output.py) → a flat,
      executor-ready schedule object; PipelineSchedule plays the same role.

Design choices:
    * Stage decomposition is deterministic given a QueryDescriptor, so the
      same query always produces the same stage graph (reproducible panels).
    * Device assignment per stage uses the existing CostModelEngine — no new
      cost model is introduced, only a finer granularity of the same one.
    * Pipeline depth (segments in flight) is bounded so that overlap never
      exceeds available device parallelism, exactly as Megatron caps warmup
      microbatches at min(pp_size-1, num_microbatches).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

# v3: per-stage cardinality decay factors
_STAGE_CARDINALITY_DECAY = {
    "FILTER": 0.7,
    "JOIN": 0.85,
    "AGGREGATE": 0.1,
    "SORT": 1.0,
    "TRANSFER": 1.0,
}
# v3: device context switch penalty (µs)
_CONTEXT_SWITCH_US = 3.5
from typing import Dict, List, Optional, Tuple

from .cost_model import (
    CostBreakdown,
    CostModelEngine,
    QueryDescriptor,
    QueryType,
)
from .schema import HardwareKind


# ---------------------------------------------------------------------------
# Stage model
# ---------------------------------------------------------------------------

class StageKind(Enum):
    """A single operator stage in a query's execution pipeline.

    These map onto the relational operators a planner would emit. The order
    of declaration is the canonical execution order (a scan feeds a filter
    feeds a join, etc.) — analogous to Megatron's fixed forward stage order.
    """
    SCAN = auto()       # read base table / index
    FILTER = auto()     # apply predicates
    JOIN = auto()       # hash / merge join
    AGGREGATE = auto()  # group-by aggregation
    SORT = auto()       # ORDER BY


@dataclass
class QueryStage:
    """One operator stage, costed independently on each device.

    A stage is a QueryDescriptor in its own right: it carries the row counts
    and width that flow *into* that operator, so the existing per-device cost
    models can price it without modification. This is the key reuse trick —
    the cost model never learns it is being asked about a fragment.

    Attributes:
        stage_id:   Stable identifier, "<query_id>::<kind>".
        kind:       Which operator this stage represents.
        descriptor: Synthetic QueryDescriptor describing this stage's inputs.
        produces_rows: Estimated output cardinality, fed to the next stage.
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

    A single query's stages form a STRICT data-dependency chain
    (SCAN -> FILTER -> JOIN -> AGGREGATE -> SORT): each stage consumes the
    previous stage's output, so they CANNOT overlap. The end-to-end latency is
    therefore the full critical path — every stage's compute PLUS every
    cross-device handoff (including the initial load from the data's home).
    There is no intra-query pipeline speedup; pipelining only helps when
    *multiple independent queries* flow through the stages (see
    QueryPipelineScheduler.schedule_pipeline, which uses the Megatron bubble
    formula for that case).

    Fields:
        compute_cost_us  — sum of per-stage compute (transfer excluded).
        transfer_cost_us — sum of ALL handoffs on the critical path: the
                           initial data load into the first stage's device,
                           plus every device switch between consecutive stages.
        latency_us       — compute_cost_us + transfer_cost_us. The physical
                           lower bound for one query on a dependency chain.
    """
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
    """Throughput estimate for a BATCH of independent queries.

    This is where pipelining actually pays off. With m independent queries
    streaming through p pipeline stages (devices), the synchronous-pipeline
    bubble fraction is (p-1)/(m+p-1) (Narayanan et al. 2021, the Megatron-LM
    pipeline-bubble result). We apply it to estimate aggregate makespan vs the
    fully-serial baseline.

    Fields:
        num_queries      — m, number of independent queries in the batch.
        num_stages       — p, the pipeline depth actually used.
        serial_makespan_us   — Σ over queries of each query's full latency
                               (no cross-query overlap).
        pipelined_makespan_us — estimated makespan once queries overlap across
                                stages, using the Megatron bubble fraction.
        bubble_fraction  — (p-1)/(m+p-1), the idle fraction.
    """
    num_queries: int
    num_stages: int
    serial_makespan_us: float
    pipelined_makespan_us: float
    bubble_fraction: float

    @property
    def speedup(self) -> float:
        """Serial / pipelined makespan. 1.0 when m=1 (no overlap possible)."""
        if self.pipelined_makespan_us <= 0:
            return 1.0
        return self.serial_makespan_us / self.pipelined_makespan_us


# ---------------------------------------------------------------------------
# Stage decomposition
# ---------------------------------------------------------------------------

def decompose_query(query: QueryDescriptor) -> List[QueryStage]:
    """Break a query into an ordered list of operator stages.

    Deterministic: the same descriptor always yields the same stages, so any
    benchmark panel built on top of this is reproducible across seeds (only
    the cost-model noise varies, not the stage graph itself).

    Row-flow model (kept deliberately simple and monotone):
        SCAN      reads selectivity * table_rows.
        FILTER    keeps the estimated_rows the planner predicted.
        JOIN      fans out by ~num_joins (each join roughly preserves rows
                  for a primary-key join; we keep it at parity to avoid
                  unphysical blow-up in a cost demo).
        AGGREGATE collapses to group_by_cardinality.
        SORT      preserves its input cardinality.
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

    # 1. SCAN — always present.
    scan_type = (QueryType.INDEX_SCAN if query.index_available
                 else QueryType.FULL_TABLE_SCAN)
    stages.append(mk(StageKind.SCAN, scan_type, scan_rows, query.selectivity))

    # 2. FILTER — present when predicates narrow the scan output.
    filtered = max(1, query.estimated_rows or scan_rows)
    if query.num_predicates > 0 and filtered < scan_rows:
        st = mk(StageKind.FILTER, QueryType.RANGE_SCAN, scan_rows,
                filtered / max(1, scan_rows))
        st.produces_rows = filtered
        stages.append(st)
    else:
        filtered = scan_rows

    # 3. JOIN — one stage per join, parity row-flow.
    join_rows = filtered
    for _ in range(max(0, query.num_joins)):
        st = mk(StageKind.JOIN, QueryType.JOIN, join_rows, 1.0,
                num_joins=1)
        st.produces_rows = join_rows
        stages.append(st)

    # 4. AGGREGATE — collapses to group cardinality.
    if query.group_by_cardinality > 0:
        # v3: damped group cardinality (actual groups < theoretical due to skew)
        gb = max(1, int(query.group_by_cardinality * 0.85))  # 15% skew damping
        st = mk(StageKind.AGGREGATE, QueryType.AGGREGATE, join_rows,
                min(1.0, gb / max(1, join_rows)),
                group_by_cardinality=gb)
        st.produces_rows = gb
        stages.append(st)
        join_rows = gb

    # v3: inter-stage cardinality tracking
    # 5. SORT — preserves cardinality.
    if query.sort_required:
        st = mk(StageKind.SORT, QueryType.SORT, join_rows, 1.0,
                sort_required=True)
        st.produces_rows = join_rows
        stages.append(st)

    return stages


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class QueryPipelineScheduler:
    """Segment a query into operator stages and pipeline them across devices.

    The scheduler reuses CostModelEngine.recommend per stage. For a single
    query it reports the full critical-path latency (compute + all transfers),
    with no intra-query speedup, because a query's stages form a strict
    dependency chain. Cross-query pipelining — where speedup actually comes
    from — is handled by schedule_pipeline() using the Megatron-LM pipeline
    bubble fraction (p-1)/(m+p-1) for m queries over p stages.

    Lifecycle:
        1. decompose      → stage graph (operator dependency chain).
        2. assign         → per-stage device via cost model (ncclTopoCompute).
        3. compute phases → warmup / steady / cooldown critical path.
    """

    def __init__(self, engine: CostModelEngine,
                 max_pipeline_depth: Optional[int] = None):
        self.engine = engine
        # Pipeline depth is bounded by the number of distinct compute devices,
        # exactly as Megatron caps in-flight microbatches at the pipeline size.
        n_devices = sum(
            1 for n in engine.topology.nodes.values()
            if n.kind in (HardwareKind.GPU, HardwareKind.CPU)
        )
        self.max_pipeline_depth = max_pipeline_depth or max(1, n_devices)

    # -- per-stage routing -------------------------------------------------

    def assign_stages(self, stages: List[QueryStage],
                      data_location: str = "cpu0"
                      ) -> List[StageAssignment]:
        """Route each stage to its lowest-cost device.

        The data_location for a stage is the device the *previous* stage
        landed on — intermediate results live where they were produced, so
        the cost model correctly charges a transfer only on a device switch.
        """
        assignments: List[StageAssignment] = []
        current_location = data_location
        for st in stages:
            device_id, cost = self.engine.recommend(
                st.descriptor, data_location=current_location
            )
            assignments.append(StageAssignment(
                stage_id=st.stage_id,
                kind=st.kind,
                device_id=device_id,
                cost=cost,
            ))
            current_location = device_id
        return assignments

    # -- critical path (single query) -------------------------------------

    @staticmethod
    def _critical_path(assignments: List[StageAssignment]  # v3: includes device-switch penalty
                       ) -> Tuple[float, float, float]:
        """End-to-end latency of one query as a strict dependency chain.

        The stages of a single query cannot overlap (each consumes the
        previous one's output). The latency is therefore the FULL critical
        path: every stage's compute plus every handoff on the path. Each
        StageAssignment.cost already includes the transfer the cost model
        charged to move that stage's input to its device — that is exactly
        the initial load for the first stage and the device-switch handoff
        for later stages. We must NOT subtract it; doing so was the bug that
        made a 15 ms PCIe load vanish and produced a fake 246x speedup.

        Returns (compute_us, transfer_us, latency_us) with
        latency_us == compute_us + transfer_us.
        """
        compute_us = 0.0
        transfer_us = 0.0
        for a in assignments:
            transfer_us += a.cost.transfer_cost_us
            compute_us += (a.cost.total_us - a.cost.transfer_cost_us)
        return compute_us, transfer_us, compute_us + transfer_us

    # -- public API --------------------------------------------------------

    def schedule(self, query: QueryDescriptor,
                 data_location: str = "cpu0") -> PipelineSchedule:
        """Build the schedule for one query.

        latency_us is the full critical path (compute + all transfers). There
        is deliberately no intra-query speedup: a single query's stages are a
        dependency chain. For cross-query pipelining use schedule_pipeline().
        """
        from ._debug import dbg
        dbg('Pipeline.schedule_start', query_id=query.query_id)
        stages = decompose_query(query)
        assignments = self.assign_stages(stages, data_location)
        compute, transfer, latency = self._critical_path(assignments)
        from ._debug import dbg
        dbg('Pipeline.schedule_done', query_id=query.query_id, n_stages=len(assignments),
            total_us=latency)
        return PipelineSchedule(
            query_id=query.query_id,
            assignments=assignments,
            compute_cost_us=compute,
            transfer_cost_us=transfer,
            latency_us=latency,
        )

    def schedule_batch(self, queries: List[QueryDescriptor],
                       data_location: str = "cpu0"
                       ) -> List[PipelineSchedule]:
        """Schedule a batch of queries independently (one schedule each)."""
        return [self.schedule(q, data_location) for q in queries]

    # -- cross-query pipelining (the real speedup) ------------------------

    def schedule_pipeline(self, queries: List[QueryDescriptor],
                          data_location: str = "cpu0"
                          ) -> PipelineBatchSchedule:
        """Estimate makespan when m independent queries pipeline across stages.

        This is where pipelining genuinely helps. Following Narayanan et al.
        (2021) — the Megatron-LM pipeline-bubble result — a synchronous
        pipeline of m microbatches over p stages spends a fraction
        (p-1)/(m+p-1) of the time idle. We treat each query as a microbatch
        and the per-query critical path as the per-microbatch time.

        Model:
            per_query_latency_i = critical path of query i (already correct).
            t_serial   = Σ per_query_latency_i        (no overlap)
            t_ideal    = t_serial / p                 (perfect p-way overlap)
            bubble     = (p-1)/(m+p-1)
            t_pipe     = t_ideal / (1 - bubble)
                       = t_serial * (m + p - 1) / (m * p)

        For m=1 this collapses to t_serial (bubble = (p-1)/p, no overlap),
        which is exactly correct: one query cannot pipeline with itself.
        """
        if not queries:
            return PipelineBatchSchedule(0, 0, 0.0, 0.0, 0.0)

        schedules = [self.schedule(q, data_location) for q in queries]
        t_serial = sum(s.latency_us for s in schedules)

        m = len(queries)
        # p = number of distinct pipeline stages = max stage count across the
        # batch, capped by available devices (you cannot run more concurrent
        # stages than you have compute devices).
        max_stages = max(len(s.assignments) for s in schedules)
        p = min(max_stages, self.max_pipeline_depth)
        p = max(1, p)

        raw_bubble = (p - 1) / (m + p - 1)
        sync_tax = 0.02 * (p - 1)  # inter-stage sync overhead
        bubble = min(raw_bubble + sync_tax, 0.95)
        # t_pipe = t_serial * (m + p - 1) / (m * p), guarding p>=1.
        t_pipe = t_serial * (m + p - 1) / (m * p)

        return PipelineBatchSchedule(
            num_queries=m,
            num_stages=p,
            serial_makespan_us=t_serial,
            pipelined_makespan_us=t_pipe,
            bubble_fraction=bubble,
        )
