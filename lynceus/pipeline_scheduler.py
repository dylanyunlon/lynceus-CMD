"""
lynceus/pipeline_scheduler.py — Query pipeline scheduler.

FIX(致命-1): transfer cost 不再从 critical path 蒸发。
    原版把每个 stage 的 transfer_cost_us 从 total_us 里减掉，
    导致首 stage 的 PCIe 搬运（可达 15ms）凭空消失，制造 246× 假加速。
    修复: 单查询 latency = Σ(total_us)，即 compute + transfer 全在 critical path 上。
    这是数据依赖链 (SCAN→FILTER→JOIN→AGG→SORT) 的物理下界。

FIX(概念-2): 流水线加速只对批量查询生效。
    Megatron 流水线的加速来自 m 个 microbatch 流过 p 个 stage，
    bubble = (p-1)/(m+p-1)。单查询 stage 间有严格数据依赖，等价 m=1，
    bubble = (p-1)/p，几乎无加速。
    修复: schedule() 单查询 latency = critical path 全长，speedup 恒为 1.0。
    schedule_pipeline() 用批量 Megatron 公式，只有 m≥2 才有真实加速。

FIX(健壮-6): assign_stages 的中间结果 data_location 传播 和 transfer 口径统一。
    所有 transfer 都留在 critical path，不做减法。
"""
from __future__ import annotations
import math
import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
from .costing import CostBreakdown, CostModelEngine, QueryDescriptor, QueryType
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
    memory_pressure: float = 0.0
    per_query_latencies_us: List[float] = field(default_factory=list)
    @property
    def speedup(self) -> float:
        if self.pipelined_makespan_us <= 0:
            return 1.0
        return self.serial_makespan_us / self.pipelined_makespan_us


# ---------------------------------------------------------------------------
# CE noise: 确定性 cardinality estimation 误差注入
# ---------------------------------------------------------------------------
# Marsaglia polar method 替换 Box-Muller:
#   Box-Muller 需要 cos(2πu) 调用三角函数，Polar 用拒绝采样避免它。
#   对确定性 hash 输入，Polar 的拒绝率稳定在 ~21.5% (π/4 圆面积比)，
#   均摊下来每次 ~1.27 次 hash 查表，比 cos() 的 ~50 cycle 快。
#   这里的 "拒绝" 是用 hash 不同 suffix 做重试，仍然是确定性的。

def _ce_noise(query_id: str, stage_name: str, sigma: float = 0.15) -> float:
    seed_str = f"{query_id}:{stage_name}"
    attempt = 0
    while True:
        h = hashlib.md5(f"{seed_str}:{attempt}".encode()).hexdigest()
        # 两个 uniform(-1,1)
        v1 = (int(h[:8], 16) / 0x7FFFFFFF) - 1.0
        v2 = (int(h[8:16], 16) / 0x7FFFFFFF) - 1.0
        s = v1 * v1 + v2 * v2
        if 0 < s < 1.0:
            # Polar transform
            z = v1 * math.sqrt(-2.0 * math.log(s) / s)
            return math.exp(sigma * z)
        attempt += 1
        if attempt > 20:
            # fallback: 无噪声 (不应该发生, 概率 < 1e-12)
            return 1.0


# ---------------------------------------------------------------------------
# decompose_query: 把一个 query 拆成 stage 链
# ---------------------------------------------------------------------------
# 改写: join 基数不再独立乘噪声，而是用递减衰减链模型:
#   join_k 的输出 = join_{k-1} 的输出 × selectivity_k × noise_k
#   其中 selectivity_k = base_sel × decay^k，decay=0.7
#   这比原版每个 join 独立噪声更符合实际: 后续 join 的选择率
#   受前面 join 条件的约束（条件概率而非独立概率）。

def decompose_query(query: QueryDescriptor,
                    ce_sigma: float = 0.15) -> List[QueryStage]:
    from ._debug import dbg

    stages: List[QueryStage] = []
    scan_rows = max(1, int(query.selectivity * query.table_rows))
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

    # 递减衰减链 join 模型
    join_rows = filtered
    join_decay = 0.7
    for j in range(max(0, query.num_joins)):
        noise = _ce_noise(query.query_id, f"join_{j}", ce_sigma)
        # 每层 join 的有效选择率递减: 后续 join 的条件概率更小
        eff_sel = noise * (join_decay ** j)
        join_rows_out = max(1, int(join_rows * min(eff_sel, 2.0)))
        st = mk(StageKind.JOIN, QueryType.JOIN, join_rows_out, 1.0, num_joins=1)
        st.produces_rows = join_rows_out
        stages.append(st)
        join_rows = join_rows_out

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

    dbg("PIP·decompose", query_id=query.query_id,
        n_stages=len(stages),
        row_flow=[s.produces_rows for s in stages],
        kinds=[s.kind.name for s in stages])
    return stages


# ---------------------------------------------------------------------------
# QueryPipelineScheduler
# ---------------------------------------------------------------------------

class QueryPipelineScheduler:
    def __init__(self, engine: CostModelEngine,
                 max_pipeline_depth: Optional[int] = None):
        self.engine = engine
        n_devices = sum(1 for n in engine.topology.nodes.values()
                        if n.kind in (HardwareKind.GPU, HardwareKind.CPU))
        self.max_pipeline_depth = max_pipeline_depth or max(1, n_devices)
        self._total_hbm = sum(
            n.memory_bytes for n in engine.topology.nodes.values()
            if n.kind == HardwareKind.GPU
        )
        # 在线 Welford 跟踪 per-stage latency 方差，用于检测 straggler
        self._stage_welford: Dict[str, Tuple[int, float, float]] = {}  # kind -> (n, mean, M2)

    def _welford_push(self, kind: str, val: float) -> None:
        n, mean, m2 = self._stage_welford.get(kind, (0, 0.0, 0.0))
        n += 1
        delta = val - mean
        mean += delta / n
        delta2 = val - mean
        m2 += delta * delta2
        self._stage_welford[kind] = (n, mean, m2)

    def _welford_summary(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for kind, (n, mean, m2) in self._stage_welford.items():
            var = m2 / (n - 1) if n > 1 else 0.0
            cv = math.sqrt(var) / mean if mean > 0 else 0.0
            out[kind] = {"n": n, "mean_us": round(mean, 2),
                         "std_us": round(math.sqrt(var), 2), "cv": round(cv, 3)}
        return out

    def assign_stages(self, stages: List[QueryStage],
                      data_location: str = "cpu0") -> List[StageAssignment]:
        from ._debug import dbg
        assignments: List[StageAssignment] = []
        current_location = data_location
        for idx, st in enumerate(stages):
            device_id, cost = self.engine.recommend(
                st.descriptor, data_location=current_location)
            assignments.append(StageAssignment(
                stage_id=st.stage_id, kind=st.kind,
                device_id=device_id, cost=cost))
            # Welford 跟踪每类 stage 的 latency
            self._welford_push(st.kind.name, cost.total_us)
            dbg("PIP·assign_stage",
                idx=idx, kind=st.kind.name, device=device_id,
                total_us=f"{cost.total_us:.1f}",
                compute_us=f"{cost.total_us - cost.transfer_cost_us:.1f}",
                transfer_us=f"{cost.transfer_cost_us:.1f}",
                from_loc=current_location)
            current_location = device_id
        return assignments

    @staticmethod
    def _critical_path(assignments: List[StageAssignment]) -> Tuple[float, float, float]:
        """FIX(致命-1): transfer 永不蒸发。
        latency = Σ(total_us)。total_us 已经包含 compute + transfer。
        我们仍然分拆 compute/transfer 做诊断，但 latency 取的是全长。
        """
        compute_us = 0.0
        transfer_us = 0.0
        for a in assignments:
            transfer_us += a.cost.transfer_cost_us
            compute_us += (a.cost.total_us - a.cost.transfer_cost_us)
        # latency = 全长 critical path（compute + transfer 都在依赖链上）
        latency = compute_us + transfer_us
        return compute_us, transfer_us, latency

    def schedule(self, query: QueryDescriptor,
                 data_location: str = "cpu0") -> PipelineSchedule:
        """单查询调度: latency = critical path 全长, speedup 恒为 1.0。"""
        from ._debug import dbg
        dbg('PIP·schedule_start', query_id=query.query_id, data_loc=data_location)
        stages = decompose_query(query)
        assignments = self.assign_stages(stages, data_location)
        compute, transfer, latency = self._critical_path(assignments)

        # 检测异常: 如果 transfer 占比 > 90%，说明数据搬运主导延迟
        if latency > 0 and transfer / latency > 0.9:
            dbg("PIP·transfer_dominated_WARNING",
                query_id=query.query_id,
                transfer_pct=f"{100 * transfer / latency:.1f}%",
                transfer_us=f"{transfer:.1f}",
                compute_us=f"{compute:.1f}",
                devices=[a.device_id for a in assignments])

        dbg('PIP·schedule_done', query_id=query.query_id,
            n_stages=len(assignments), latency_us=f"{latency:.1f}",
            compute_us=f"{compute:.1f}", transfer_us=f"{transfer:.1f}",
            devices=[a.device_id for a in assignments])
        return PipelineSchedule(
            query_id=query.query_id, assignments=assignments,
            compute_cost_us=compute, transfer_cost_us=transfer,
            latency_us=latency)

    def schedule_batch(self, queries: List[QueryDescriptor],
                       data_location: str = "cpu0") -> List[PipelineSchedule]:
        return [self.schedule(q, data_location) for q in queries]

    def schedule_pipeline(self, queries: List[QueryDescriptor],
                          data_location: str = "cpu0") -> PipelineBatchSchedule:
        """FIX(概念-2): 流水线加速只对批量查询生效。

        Megatron 真实公式 (Narayanan et al., 2021):
            T_pipe = max_stage × (m + p - 1)
            T_serial = Σ(per_query_latency)
            bubble = (p - 1) / (m + p - 1)

        其中:
            m = 查询数 (microbatch count)
            p = 流水线深度 (stage count on distinct devices)

        m=1 时 bubble = (p-1)/p ≈ 1，几乎无加速。
        m≥2 时才有真实的流水线并行。

        改写: 用 per-stage max latency 做 bottleneck stage 估算,
        而非原版对 t_serial 做除法（那是假设所有 stage 等长）。
        """
        from ._debug import dbg, checkpoint

        if not queries:
            return PipelineBatchSchedule(0, 0, 0.0, 0.0, 0.0)

        schedules = [self.schedule(q, data_location) for q in queries]
        per_query_lats = [s.latency_us for s in schedules]
        t_serial = sum(per_query_lats)

        m = len(queries)
        max_stages = max(len(s.assignments) for s in schedules)
        p = min(max_stages, self.max_pipeline_depth)
        p = max(1, p)

        # 找 bottleneck stage: 用所有查询在每个 stage position 上的 max latency
        # 这比原版 "t_serial / (m*p)" 精确，因为 stage 之间 latency 可能差 10x
        stage_maxes: List[float] = []
        for si in range(max_stages):
            worst = 0.0
            for sched in schedules:
                if si < len(sched.assignments):
                    worst = max(worst, sched.assignments[si].cost.total_us)
            stage_maxes.append(worst)
        bottleneck_us = max(stage_maxes) if stage_maxes else 0.0

        # Megatron 公式: 先 fill p-1 个 stage，然后 steady state m 个 microbatch
        # T_pipe = bottleneck × (m + p - 1)
        t_pipe_ideal = bottleneck_us * (m + p - 1)

        # memory contention 修正: 活跃数据量 / 总 HBM
        total_transfer_us = sum(s.transfer_cost_us for s in schedules)
        approx_active_bytes = total_transfer_us * 32 * 1024  # 1µs ≈ 32KB @ 32GB/s
        mem_pressure = (approx_active_bytes / max(1, self._total_hbm)
                        if self._total_hbm > 0 else 0.0)
        # 超过 70% HBM 占用时，每超 10% 增加 5% makespan（内存带宽争用）
        mem_slowdown = max(0.0, (mem_pressure - 0.7) * 0.5)

        # sync overhead: 每个 stage 边界有 barrier 开销
        sync_overhead_us = 2.0 * (p - 1)  # ~2µs per barrier

        t_pipe = t_pipe_ideal * (1.0 + mem_slowdown) + sync_overhead_us * m
        t_pipe = max(t_pipe, max(per_query_lats))  # 不能比最慢的单查询还快

        bubble = (p - 1) / (m + p - 1)

        # straggler 检测: 如果 bottleneck stage 比平均 stage 慢 3x 以上, 警告
        avg_stage = sum(stage_maxes) / max(1, len(stage_maxes))
        if avg_stage > 0 and bottleneck_us / avg_stage > 3.0:
            dbg("PIP·straggler_WARNING",
                bottleneck_us=f"{bottleneck_us:.1f}",
                avg_stage_us=f"{avg_stage:.1f}",
                ratio=f"{bottleneck_us / avg_stage:.1f}x",
                stage_maxes=[f"{x:.1f}" for x in stage_maxes])

        checkpoint("PIP·pipeline_batch",
                   m=m, p=p, bubble=f"{bubble:.3f}",
                   bottleneck_us=f"{bottleneck_us:.1f}",
                   t_serial_us=f"{t_serial:.1f}",
                   t_pipe_us=f"{t_pipe:.1f}",
                   speedup=f"{t_serial / t_pipe:.2f}" if t_pipe > 0 else "inf",
                   mem_pressure=f"{mem_pressure:.3f}",
                   mem_slowdown=f"{mem_slowdown:.3f}",
                   sync_overhead_us=f"{sync_overhead_us * m:.1f}",
                   stage_welford=self._welford_summary())

        return PipelineBatchSchedule(
            num_queries=m, num_stages=p,
            serial_makespan_us=t_serial,
            pipelined_makespan_us=t_pipe,
            bubble_fraction=bubble,
            memory_pressure=mem_pressure,
            per_query_latencies_us=per_query_lats)

    def dump_state(self) -> Dict[str, Any]:
        """当前调度器全状态快照，用于断点调试。"""
        return {
            "max_pipeline_depth": self.max_pipeline_depth,
            "total_hbm_bytes": self._total_hbm,
            "total_hbm_gb": round(self._total_hbm / (1 << 30), 1),
            "stage_welford": self._welford_summary(),
            "n_devices": sum(1 for n in self.engine.topology.nodes.values()
                             if n.kind in (HardwareKind.GPU, HardwareKind.CPU)),
            "device_ids": sorted(self.engine.topology.nodes.keys()),
        }

    # ===================================================================
    # M015 新增: 1F1B Interleaved Virtual Pipeline Schedule
    # ===================================================================
    # Megatron-LM 的 1F1B 调度 (Narayanan et al. 2021):
    #   - Warmup phase: 前 p-1 个 microbatch 只做 forward
    #   - Steady state: 每做一个 forward 就紧跟一个 backward
    #   - Cooldown phase: 最后 p-1 个 microbatch 只做 backward
    #
    # Interleaved (virtual pipeline):
    #   - 每个设备持有 v 个 virtual chunks (v=num_model_chunks)
    #   - 有效 pipeline 深度 = p×v
    #   - bubble 从 (p-1)/(m+p-1) 降到 (p-1)/(m×v+p-1)
    #   - 代价: 更多通信次数 (每个 chunk 边界要交换 activation)

    def schedule_1f1b(self, queries: List[QueryDescriptor],
                      data_location: str = "cpu0",
                      virtual_chunks: int = 1) -> PipelineBatchSchedule:
        """1F1B interleaved pipeline scheduling.

        Args:
            queries: microbatch 列表
            data_location: 数据起始位置
            virtual_chunks: 虚拟流水线分块数 (v≥1, v>1 为 interleaved)
        """
        from ._debug import dbg, checkpoint

        if not queries:
            return PipelineBatchSchedule(0, 0, 0.0, 0.0, 0.0)

        v = max(1, virtual_chunks)
        m = len(queries)
        schedules = [self.schedule(q, data_location) for q in queries]
        per_query_lats = [s.latency_us for s in schedules]
        t_serial = sum(per_query_lats)

        max_stages = max(len(s.assignments) for s in schedules)
        p = min(max_stages, self.max_pipeline_depth)
        p = max(1, p)

        # Virtual pipeline: 有效深度 p_eff = p * v
        # 每个 stage 的 latency 被分成 v 个 chunk, 每个 chunk 更短
        stage_maxes: List[float] = []
        for si in range(max_stages):
            worst = 0.0
            for sched in schedules:
                if si < len(sched.assignments):
                    worst = max(worst, sched.assignments[si].cost.total_us)
            stage_maxes.append(worst)
        bottleneck_us = max(stage_maxes) if stage_maxes else 0.0

        # Virtual pipeline 拆分: 每个 chunk 的 bottleneck ≈ bottleneck / v
        chunk_bottleneck = bottleneck_us / v

        # 1F1B bubble 公式 (interleaved):
        #   bubble = (p - 1) / (m × v + p - 1)
        # 标准1F1B (v=1): bubble = (p-1)/(m+p-1)
        eff_m = m * v
        bubble = (p - 1) / (eff_m + p - 1)

        # 1F1B 三阶段 makespan:
        #   warmup:  (p-1) × chunk_bottleneck  (只 forward, 填满 pipeline)
        #   steady:  m × chunk_bottleneck      (1F1B steady state)
        #   cooldown: (p-1) × chunk_bottleneck  (只 backward, 排空 pipeline)
        # 但 steady state 的 forward+backward 是重叠的, 所以:
        #   T = chunk_bottleneck × (m × v + p - 1)
        t_1f1b = chunk_bottleneck * (eff_m + p - 1)

        # Virtual pipeline 的额外通信: 每个 chunk 边界要交换 activation
        # 每次交换 ≈ 5µs (PCIe P2P) 或 1µs (NVLink)
        extra_comm_per_chunk = 3.0  # µs, 保守估计
        n_chunk_boundaries = (v - 1) * m  # 每个 microbatch 经过 v-1 个 chunk 边界
        extra_comm_us = extra_comm_per_chunk * n_chunk_boundaries

        t_pipe = t_1f1b + extra_comm_us
        t_pipe = max(t_pipe, max(per_query_lats))  # 下界

        # Memory pressure: 1F1B 的优势是只需保存 p-1 个 microbatch 的 activation
        # (而非 m 个), 大幅降低 memory 需求
        peak_activations = p - 1  # 1F1B 的 activation 峰值
        est_activation_bytes = bottleneck_us * 32 * 1024  # 粗估
        mem_pressure = (peak_activations * est_activation_bytes
                        / max(1, self._total_hbm))

        checkpoint("PIP·1f1b_schedule",
                   m=m, p=p, v=v, eff_m=eff_m,
                   bubble=f"{bubble:.4f}",
                   bottleneck_us=f"{bottleneck_us:.1f}",
                   chunk_bottleneck_us=f"{chunk_bottleneck:.1f}",
                   t_serial_us=f"{t_serial:.1f}",
                   t_1f1b_us=f"{t_1f1b:.1f}",
                   extra_comm_us=f"{extra_comm_us:.1f}",
                   t_pipe_us=f"{t_pipe:.1f}",
                   speedup=f"{t_serial / t_pipe:.2f}" if t_pipe > 0 else "inf",
                   mem_pressure=f"{mem_pressure:.4f}",
                   peak_activations=peak_activations,
                   stage_welford=self._welford_summary())

        return PipelineBatchSchedule(
            num_queries=m, num_stages=p,
            serial_makespan_us=t_serial,
            pipelined_makespan_us=t_pipe,
            bubble_fraction=bubble,
            memory_pressure=mem_pressure,
            per_query_latencies_us=per_query_lats)

    def _dbg_pipeline_gantt(self, schedules: List[PipelineSchedule]) -> str:
        """调试用: 生成 ASCII Gantt 图 展示 pipeline 时间线."""
        if not schedules:
            return "(empty)"
        lines = ["== Pipeline Gantt Chart =="]
        max_lat = max(s.latency_us for s in schedules)
        width = 60  # ASCII chart width
        for i, s in enumerate(schedules):
            bar_len = int(s.latency_us / max_lat * width) if max_lat > 0 else 0
            devices = ",".join(s.devices_used)
            bar = "█" * bar_len + "░" * (width - bar_len)
            lines.append(f"  Q{i:03d} [{bar}] {s.latency_us:8.1f}µs  {devices}")
        lines.append(f"  Scale: |{'─'*width}| = {max_lat:.1f}µs")
        return "\n".join(lines)
