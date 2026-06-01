"""
lynceus_port/schema.py — 移植版数据 schema.

基于 lynceus/schema.py 框架, 改写 ≈ 20%:
  - compute_statistics 使用 Welford 在线算法替代双遍扫描
  - HardwareTopology.get_transfer_cost 增加带宽衰减因子 (拥塞模拟)
  - 全链路结构体 dump 断点
  - metadata 增加 wall_clock_s 字段

架构参考:
    ncclResult_t / DistributedDataParallelConfig / KVCacheSpec
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


# ── 调试工具 ──────────────────────────────────────────────────────────────
import os as _os
_LYNCEUS_DBG = _os.environ.get("LYNCEUS_DEBUG", "1")  # 默认开启

def _dbg(tag: str, msg: str):
    """运行时断点输出 — 打印当前数据结构和状态到 stderr 以便管道分离."""
    if _LYNCEUS_DBG != "0":
        import sys
        print(f"[SCH·{tag}] {msg}", file=sys.stderr, flush=True)

# 保留旧名兼容
_tr = _dbg

# ───────────────────────── Enums ─────────────────────────────────────────

class RoutingStrategy(Enum):
    GPU_ONLY = "GPU-Only"
    CPU_ONLY = "CPU-Only"
    HYBRID_STATIC = "Hybrid-Static"
    COST_MODEL_ROUTED = "CostModel-Routed"
    PAR2QO_ENHANCED = "PAR2QO-Enhanced"
    VIDEX_ENHANCED = "VIDEX-Enhanced"
    ADAPTIVE = "Adaptive"


class MetricKind(Enum):
    LATENCY_MS = auto()
    THROUGHPUT_QPS = auto()
    COST_UNITS = auto()
    GRADIENT_NORM = auto()
    PERPLEXITY = auto()
    CACHE_HIT_RATE = auto()
    INDEX_BUILD_TIME_S = auto()


class HardwareKind(Enum):
    GPU = auto()
    CPU = auto()
    NVLINK = auto()
    PCIE = auto()
    NETWORK = auto()


# ───────────────────────── Core data structures ─────────────────────────

@dataclass
class SeedCurve:
    """单种子的 Y 值序列. len(values) == 父 MethodResult.num_steps."""
    seed_id: int
    values: List[float] = field(default_factory=list)

    def append(self, v: float) -> None:
        _dbg("SEED", f"seed#{self.seed_id} += {v:.6f} (pos={len(self.values)})")
        self.values.append(v)

    def dump_snapshot(self, max_show: int = 8) -> str:
        """断点检查: 打印前/后几个值 + Welford 增量统计."""
        n = len(self.values)
        if n == 0:
            return f"SeedCurve(id={self.seed_id}, EMPTY)"
        half = max_show >> 1
        head = self.values[:half]
        tail = self.values[-half:] if n > max_show else []
        # Kahan 求和减少浮点漂移
        kahan_sum = 0.0
        kahan_c = 0.0
        for v in self.values:
            y = v - kahan_c
            t = kahan_sum + y
            kahan_c = (t - kahan_sum) - y
            kahan_sum = t
        mu = kahan_sum / n
        return (f"SeedCurve(id={self.seed_id}, n={n}, "
                f"mean={mu:.6f}, head={head}, tail={tail})")


@dataclass
class MethodResult:
    """聚合结果 — 一个路由策略在所有种子上的表现."""
    strategy: RoutingStrategy
    num_steps: int
    num_seeds: int
    x_values: List[float] = field(default_factory=list)
    seed_curves: List[SeedCurve] = field(default_factory=list)
    reported_final: Optional[float] = None
    total_cost: Optional[float] = None

    _mean: Optional[List[float]] = field(default=None, repr=False)
    _std: Optional[List[float]] = field(default=None, repr=False)

    def add_seed(self) -> SeedCurve:
        cur_count = len(self.seed_curves)
        _dbg("METHOD", f"{self.strategy.value}: add_seed#{cur_count} "
             f"(limit={self.num_seeds}, steps={self.num_steps})")
        if cur_count >= self.num_seeds:
            raise ValueError(
                f"Cannot add seed #{cur_count}: "
                f"num_seeds is {self.num_seeds}"
            )
        sc = SeedCurve(seed_id=cur_count)
        self.seed_curves.append(sc)
        return sc

    def compute_statistics(self) -> None:
        """Welford 在线算法 — 单遍完成 mean/std, 增加 min/max 追踪用于断点."""
        total_steps = self.num_steps
        seed_count = len(self.seed_curves)
        _dbg("STAT", f"compute_statistics: strategy={self.strategy.value}, "
             f"steps={total_steps}, seeds={seed_count}")
        if seed_count == 0 or total_steps == 0:
            _dbg("STAT", "SKIP: no data")
            return

        for sc in self.seed_curves:
            actual_len = len(sc.values)
            if actual_len != total_steps:
                raise ValueError(
                    f"Seed {sc.seed_id} has {actual_len} values, "
                    f"expected {total_steps} (num_steps)"
                )

        mean_accum: List[float] = []
        std_accum: List[float] = []
        for step_idx in range(total_steps):
            # Welford 单遍 — 用 running_mean / running_m2
            running_mean = 0.0
            running_m2 = 0.0
            for j in range(seed_count):
                val = self.seed_curves[j].values[step_idx]
                prev_mean = running_mean
                running_mean = prev_mean + (val - prev_mean) / (j + 1)
                running_m2 += (val - prev_mean) * (val - running_mean)
            mean_accum.append(running_mean)
            variance = running_m2 / (seed_count - 1) if seed_count > 1 else 0.0
            std_accum.append(math.sqrt(max(variance, 0.0)))

        self._mean = mean_accum
        self._std = std_accum
        if mean_accum:
            self.reported_final = mean_accum[-1]

        _dbg("STAT", f"{self.strategy.value}: final_mean={self.reported_final:.6f}, "
             f"final_std={std_accum[-1]:.6f}, "
             f"range=[{min(mean_accum):.4f}, {max(mean_accum):.4f}]")

    @property
    def mean(self) -> List[float]:
        if self._mean is None:
            self.compute_statistics()
        return self._mean or []

    @property
    def std(self) -> List[float]:
        if self._std is None:
            self.compute_statistics()
        return self._std or []

    def to_dict(self) -> Dict[str, Any]:
        self.compute_statistics()
        d: Dict[str, Any] = {}
        if self.x_values:
            d["x_steps"] = self.x_values
        d["total_cost"] = self.total_cost
        d["reported_final"] = self.reported_final
        for sc in self.seed_curves:
            d[f"seed_{sc.seed_id}"] = sc.values
        d["mean"] = self.mean
        d["std"] = self.std
        return d

    def dump_snapshot(self) -> str:
        """断点辅助: 全状态概览."""
        lines = [f"MethodResult({self.strategy.value}, steps={self.num_steps}, "
                 f"seeds={self.num_seeds}, final={self.reported_final})"]
        for sc in self.seed_curves:
            lines.append(f"  {sc.dump_snapshot()}")
        return "\n".join(lines)


@dataclass
class PanelResult:
    panel_name: str
    metric: MetricKind
    x_label: str = "step"
    y_label: str = "latency_ms"
    methods: Dict[str, MethodResult] = field(default_factory=dict)

    def add_method(self, strategy: RoutingStrategy, num_steps: int,
                   num_seeds: int) -> MethodResult:
        mr = MethodResult(strategy=strategy, num_steps=num_steps,
                          num_seeds=num_seeds)
        self.methods[strategy.value] = mr
        return mr

    def to_dict(self) -> Dict[str, Any]:
        return {name: mr.to_dict() for name, mr in self.methods.items()}


@dataclass
class BenchmarkOutput:
    description: str = ""
    source: str = ""
    timestamp: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))
    panels: Dict[str, PanelResult] = field(default_factory=dict)
    _start_wall: float = field(default_factory=time.monotonic, repr=False)

    def add_panel(self, name: str, metric: MetricKind,
                  x_label: str = "step", y_label: str = "value") -> PanelResult:
        pr = PanelResult(panel_name=name, metric=metric,
                         x_label=x_label, y_label=y_label)
        self.panels[name] = pr
        return pr

    @property
    def metadata(self) -> Dict[str, Any]:
        total_points = 0
        n_methods = 0
        n_seeds = 0
        n_per_seed = 0
        for panel in self.panels.values():
            for mr in panel.methods.values():
                n_methods += 1
                n_seeds = max(n_seeds, mr.num_seeds)
                n_per_seed = max(n_per_seed, mr.num_steps)
                total_points += mr.num_steps * mr.num_seeds
        return {
            "description": self.description,
            "source": self.source,
            "timestamp": self.timestamp,
            "n_per_seed": n_per_seed,
            "n_seeds": n_seeds,
            "n_methods": n_methods,
            "total_data_points": total_points,
            "wall_clock_s": round(time.monotonic() - self._start_wall, 3),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "panels": {name: pr.to_dict() for name, pr in self.panels.items()},
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        _dbg("IO", f"save → {p}, panels={list(payload.get('panels',{}).keys())}, "
             f"meta={payload.get('metadata',{})}")
        with open(p, "w") as f:
            json.dump(payload, f, indent=2)
        byte_size = p.stat().st_size
        _dbg("IO", f"written {byte_size:,}B to {p}")
        return p


# ─────────────────── Hardware topology node ──────────────────────────────

@dataclass
class HardwareNode:
    node_id: str
    kind: HardwareKind
    compute_capacity: float = 1.0
    memory_bytes: int = 0
    bandwidth_gbps: float = 0.0
    latency_us: float = 0.0
    scan_cost_per_row: float = 0.02
    seek_cost: float = 0.5
    compute_cost_per_op: float = 0.01


@dataclass
class TopologyEdge:
    src: str
    dst: str
    bandwidth_gbps: float = 0.0
    latency_us: float = 0.0
    link_type: HardwareKind = HardwareKind.PCIE


@dataclass
class HardwareTopology:
    """完整硬件拓扑图 — Dijkstra 多跳最短路."""
    nodes: Dict[str, HardwareNode] = field(default_factory=dict)
    edges: List[TopologyEdge] = field(default_factory=list)

    def add_node(self, node: HardwareNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: TopologyEdge) -> None:
        self.edges.append(edge)

    def get_transfer_cost(self, src: str, dst: str, data_bytes: int) -> float:
        """Dijkstra 多跳传输成本 (µs).

        改写: 拥塞衰减因子从固定 0.95 改为 sigmoid 递减模型 —
        hop_penalty = 1 / (1 + 0.05 * hop_idx), 首跳无衰减,
        远跳渐近 0, 更贴合交换矩阵实测曲线.
        """
        _dbg("XFER", f"get_transfer_cost({src}→{dst}, {data_bytes}B)")
        if src == dst:
            return 0.0
        if src not in self.nodes or dst not in self.nodes:
            _dbg("XFER", f"UNREACHABLE: {src} or {dst} not in topology")
            return float("inf")

        DECAY_COEFF = 0.05  # sigmoid 衰减系数

        def hop_cost(edge: TopologyEdge, hop_idx: int) -> float:
            if edge.bandwidth_gbps <= 0:
                return float("inf")
            penalty = 1.0 / (1.0 + DECAY_COEFF * hop_idx)
            eff_bw = edge.bandwidth_gbps * penalty
            transfer_us = (data_bytes / (eff_bw * 1e9 / 8)) * 1e6
            total = edge.latency_us + transfer_us
            return total

        import heapq
        dist: Dict[str, float] = {src: 0.0}
        hops: Dict[str, int] = {src: 0}
        pq: List[Tuple[float, str]] = [(0.0, src)]
        visited = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                _dbg("XFER", f"  result={d:.2f}µs via {hops.get(u,0)} hops")
                return d
            if u in visited:
                continue
            visited.add(u)
            h = hops.get(u, 0)
            for e in self.edges:
                if e.src != u:
                    continue
                v = e.dst
                w = hop_cost(e, h)
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    hops[v] = h + 1
                    heapq.heappush(pq, (nd, v))
        final = dist.get(dst, float("inf"))
        _dbg("XFER", f"  result={final:.2f}µs (Dijkstra converged)")
        return final

    def get_node(self, node_id: str) -> Optional[HardwareNode]:
        return self.nodes.get(node_id)

    def dump_reachability(self) -> str:
        """断点辅助: 打印全对可达矩阵."""
        ids = sorted(self.nodes.keys())
        lines = ["Reachability (1MB payload):"]
        hdr = "       " + "  ".join(f"{n:>6}" for n in ids)
        lines.append(hdr)
        for s in ids:
            row = []
            for d in ids:
                if s == d:
                    row.append("     0")
                else:
                    c = self.get_transfer_cost(s, d, 1 << 20)
                    row.append(f"{c:6.1f}" if c < float("inf") else "   inf")
            lines.append(f" {s:>5} " + "  ".join(row))
        return "\n".join(lines)
