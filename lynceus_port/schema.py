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
def _tr(tag: str, msg: str):
    """轻量 trace, 始终可用 (无需环境变量); 生产环境可全局替换为 pass."""
    pass  # 改为 print(f"[SCH·{tag}] {msg}") 即可启用

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
        self.values.append(v)

    def dump_snapshot(self, max_show: int = 8) -> str:
        """断点检查: 打印前/后几个值 + 基本统计."""
        n = len(self.values)
        if n == 0:
            return f"SeedCurve(id={self.seed_id}, EMPTY)"
        head = self.values[:max_show // 2]
        tail = self.values[-(max_show // 2):] if n > max_show else []
        mu = sum(self.values) / n
        return (f"SeedCurve(id={self.seed_id}, n={n}, "
                f"mean={mu:.4f}, head={head}, tail={tail})")


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
        if len(self.seed_curves) >= self.num_seeds:
            raise ValueError(
                f"Cannot add seed #{len(self.seed_curves)}: "
                f"num_seeds is {self.num_seeds}"
            )
        sc = SeedCurve(seed_id=len(self.seed_curves))
        self.seed_curves.append(sc)
        return sc

    def compute_statistics(self) -> None:
        """Welford 在线算法 — 单遍完成 mean/std, 数值更稳定."""
        n = self.num_steps
        k = len(self.seed_curves)
        if k == 0 or n == 0:
            return

        for sc in self.seed_curves:
            if len(sc.values) != n:
                raise ValueError(
                    f"Seed {sc.seed_id} has {len(sc.values)} values, "
                    f"expected {n} (num_steps)"
                )

        mean_out: List[float] = []
        std_out: List[float] = []
        for i in range(n):
            # Welford 单遍
            w_mean = 0.0
            w_m2 = 0.0
            for j, sc in enumerate(self.seed_curves):
                v = sc.values[i]
                delta = v - w_mean
                w_mean += delta / (j + 1)
                delta2 = v - w_mean
                w_m2 += delta * delta2
            mean_out.append(w_mean)
            std_out.append(math.sqrt(w_m2 / (k - 1)) if k > 1 else 0.0)

        self._mean = mean_out
        self._std = std_out
        if mean_out:
            self.reported_final = mean_out[-1]

        _tr("STAT", f"{self.strategy.value}: final_mean={self.reported_final:.4f}")

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
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        _tr("IO", f"saved {p} ({p.stat().st_size:,}B)")
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

        改写点: 引入拥塞衰减因子 — 带宽随跳数递减 5%, 模拟
        真实交换矩阵的竞争效应 (NCCL ncclTopoComputePaths 概念).
        """
        if src == dst:
            return 0.0
        if src not in self.nodes or dst not in self.nodes:
            return float("inf")

        CONGESTION_DECAY = 0.95  # 每跳带宽衰减

        def hop_cost(e: TopologyEdge, hop_idx: int) -> float:
            if e.bandwidth_gbps <= 0:
                return float("inf")
            effective_bw = e.bandwidth_gbps * (CONGESTION_DECAY ** hop_idx)
            return e.latency_us + (data_bytes / (effective_bw * 1e9 / 8)) * 1e6

        # 邻接表
        adj: Dict[str, List[Tuple[str, float]]] = {}
        hop_counter: Dict[Tuple[str, str], int] = {}
        for idx, e in enumerate(self.edges):
            w = hop_cost(e, 0)  # 初始 hop_idx=0
            if w != float("inf"):
                adj.setdefault(e.src, []).append((e.dst, w))

        import heapq
        dist: Dict[str, float] = {src: 0.0}
        hops: Dict[str, int] = {src: 0}
        pq: List[Tuple[float, str]] = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                return d
            if d > dist.get(u, float("inf")):
                continue
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
        return dist.get(dst, float("inf"))

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
