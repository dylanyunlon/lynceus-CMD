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
    """Executor-ready schedule for one query.

    Analogous to vLLM's SchedulerOutput: a flat object the executor consumes
    without re-deriving anything. The three timing aggregates capture the
    Megatron fill-drain intuition:

        serial_cost_us    — sum of all stages (no overlap; the CPU/GPU-only
                            baselines effectively pay this).
        pipelined_cost_us — critical-path cost once independent stages on
                            distinct devices overlap (warmup + steady + drain).
        transfer_cost_us  — cross-device handoff cost incurred between stages
                            assigned to different devices (the "bubble" tax).
    """
    query_id: str
    assignments: List[StageAssignment]
    serial_cost_us: float
    pipelined_cost_us: float
    transfer_cost_us: float
    warmup_segments: int

    @property
    def speedup(self) -> float:
        """Serial / pipelined. >= 1.0; 1.0 means pipelining bought nothing."""
        if self.pipelined_cost_us <= 0:
            return 1.0
        return self.serial_cost_us / self.pipelined_cost_us

    @property
    def devices_used(self) -> List[str]:
        seen: List[str] = []
        for a in self.assignments:
            if a.device_id not in seen:
                seen.append(a.device_id)
        return seen


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
        gb = max(1, query.group_by_cardinality)
        st = mk(StageKind.AGGREGATE, QueryType.AGGREGATE, join_rows,
                min(1.0, gb / max(1, join_rows)),
                group_by_cardinality=gb)
        st.produces_rows = gb
        stages.append(st)
        join_rows = gb

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

    The scheduler reuses CostModelEngine.recommend per stage, then estimates
    end-to-end latency under a Megatron-style fill-drain overlap model rather
    than summing stage costs. Stages pinned to *different* devices can overlap
    once the pipe is full; stages on the *same* device serialize.

    Lifecycle (mirrors Megatron schedule construction):
        1. decompose      → stage graph (forward stage order).
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

    # -- fill-drain timing -------------------------------------------------

    def _pipelined_cost(self, assignments: List[StageAssignment]
                        ) -> Tuple[float, float, int]:
        """Estimate overlapped latency under a Megatron fill-drain model.

        Intuition: collapse adjacent stages on the same device into "runs"
        (one device cannot overlap with itself). A single run is a plain
        serial chain — no speedup. With two or more runs on distinct devices
        the pipe can fill and drain: the bottleneck run sets the floor, and
        non-bottleneck work is hidden behind it up to the bottleneck's own
        length, scaled by how full the pipe gets (warmup_segments / runs).
        Cross-device handoffs add a transfer tax — the pipeline bubble.

        Returns (pipelined_us, transfer_us, warmup_segments).
        """
        if not assignments:
            return 0.0, 0.0, 0

        # Transfer tax: charged whenever two adjacent stages differ in device.
        transfer_us = 0.0
        for prev, cur in zip(assignments, assignments[1:]):
            transfer_us += cur.cost.transfer_cost_us

        # Group into maximal runs on the same device. Within a run, stages
        # serialize (one device, no overlap). Across runs on distinct devices,
        # the runs overlap up to max_pipeline_depth, so the critical path is
        # the heaviest run plus the fill/drain of the rest.
        runs: List[float] = []
        run_cost = 0.0
        run_device: Optional[str] = None
        for a in assignments:
            compute = a.cost.total_us - a.cost.transfer_cost_us
            if a.device_id == run_device:
                run_cost += compute
            else:
                if run_device is not None:
                    runs.append(run_cost)
                run_cost = compute
                run_device = a.device_id
        if run_device is not None:
            runs.append(run_cost)

        total = sum(runs)

        # Single run = single device: nothing overlaps, the pipeline is a
        # plain serial chain. This is the degenerate (and common) case and
        # must NOT report any speedup.
        if len(runs) <= 1:
            return total + transfer_us, transfer_us, 0

        bottleneck = max(runs)
        non_bottleneck = total - bottleneck
        # Warmup/drain segments: runs that fill and drain the pipe, capped by
        # pipeline depth (Megatron num_warmup_microbatches = min(pp-1, mb)).
        warmup_segments = min(self.max_pipeline_depth, len(runs) - 1)

        # Overlap can hide non-bottleneck work behind the bottleneck run, but
        # never more than the bottleneck itself (you cannot hide 100µs of work
        # behind a 10µs stage), and the effect scales with how full the pipe
        # gets relative to its depth. This keeps pipelined in
        # [bottleneck + transfer, serial].
        fill_fraction = warmup_segments / len(runs)
        hidden = min(non_bottleneck, bottleneck) * fill_fraction
        pipelined = total - hidden + transfer_us
        return pipelined, transfer_us, warmup_segments

    # -- public API --------------------------------------------------------

    def schedule(self, query: QueryDescriptor,
                 data_location: str = "cpu0") -> PipelineSchedule:
        """Build a full pipeline schedule for one query."""
        stages = decompose_query(query)
        assignments = self.assign_stages(stages, data_location)

        serial = sum(a.cost.total_us for a in assignments)
        pipelined, transfer, warmup = self._pipelined_cost(assignments)

        return PipelineSchedule(
            query_id=query.query_id,
            assignments=assignments,
            serial_cost_us=serial,
            pipelined_cost_us=pipelined,
            transfer_cost_us=transfer,
            warmup_segments=warmup,
        )

    def schedule_batch(self, queries: List[QueryDescriptor],
                       data_location: str = "cpu0"
                       ) -> List[PipelineSchedule]:
        """Schedule a batch of queries independently."""
        return [self.schedule(q, data_location) for q in queries]
