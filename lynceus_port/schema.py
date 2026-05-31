"""
lynceus_port/schema.py — 规范数据模式，用于基准测试输出。

移植自 lynceus/schema.py，修改约20%:
  - 统计计算改为 Welford 在线算法（单遍扫描，数值更稳定）
  - 新增 debug_snapshot() 方法：随时打印全部内部状态
  - HardwareTopology.get_transfer_cost 改用 A* 启发式
  - 新增 __repr__ 深度格式化，方便断点调试时阅读
"""

from __future__ import annotations

import json
import math
import time
import heapq
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# ── 调试开关：设为 True 时所有 debug_snapshot 自动打印到 stderr ──
_DEBUG_TRACE = True

import sys
def _dbg(tag: str, msg: str) -> None:
    """统一的调试输出，方便用 grep 过滤"""
    if _DEBUG_TRACE:
        print(f"[DBG][{tag}] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------

class RoutingStrategy(Enum):
    """查询路由策略"""
    GPU_ONLY       = "gpu_exclusive"
    CPU_ONLY       = "cpu_exclusive"
    HYBRID_STATIC  = "hybrid_threshold"
    COST_ROUTED    = "cost_model_dispatch"
    PAR2QO_ROBUST  = "par2qo_penalty_aware"
    VIDEX_VIRTUAL  = "videx_virtual_idx"
    ADAPTIVE_ONLINE = "online_adaptive"


class MetricKind(Enum):
    """Y轴度量类型"""
    LATENCY_US    = auto()   # 微秒（非毫秒——更贴近实际内核粒度）
    THROUGHPUT_QPS = auto()
    COST_UNITS    = auto()
    GRAD_NORM     = auto()
    PERPLEXITY    = auto()
    CACHE_HIT_PCT = auto()
    IDX_BUILD_S   = auto()


class HardwareKind(Enum):
    GPU     = auto()
    CPU     = auto()
    NVLINK  = auto()
    PCIE    = auto()
    NETWORK = auto()


# ---------------------------------------------------------------------------
# 核心数据结构
# ---------------------------------------------------------------------------

@dataclass
class SeedCurve:
    """单个随机种子在所有 step 上的 Y 值序列"""
    seed_id: int
    values: List[float] = field(default_factory=list)

    def append(self, v: float) -> None:
        self.values.append(v)

    def debug_snapshot(self) -> str:
        n = len(self.values)
        tail = self.values[-5:] if n >= 5 else self.values
        s = (f"SeedCurve(id={self.seed_id}, len={n}, "
             f"min={min(self.values):.4f}, max={max(self.values):.4f}, "
             f"tail_5={[round(v,4) for v in tail]})")
        _dbg("SeedCurve", s)
        return s

    def __repr__(self) -> str:
        return f"SeedCurve(id={self.seed_id}, pts={len(self.values)})"


@dataclass
class MethodResult:
    """单个路由策略在所有种子上的聚合结果"""
    strategy: RoutingStrategy
    num_steps: int
    num_seeds: int
    x_values: List[float] = field(default_factory=list)
    seed_curves: List[SeedCurve] = field(default_factory=list)
    reported_final: Optional[float] = None
    total_cost: Optional[float] = None

    # Welford 在线统计——惰性计算
    _mean: Optional[List[float]] = field(default=None, repr=False)
    _std: Optional[List[float]] = field(default=None, repr=False)

    def add_seed(self) -> SeedCurve:
        if len(self.seed_curves) >= self.num_seeds:
            raise ValueError(
                f"种子数已满: 已有 {len(self.seed_curves)}, 上限 {self.num_seeds}"
            )
        sc = SeedCurve(seed_id=len(self.seed_curves))
        self.seed_curves.append(sc)
        _dbg("MethodResult", f"add_seed #{sc.seed_id} for {self.strategy.value}")
        return sc

    def compute_statistics(self) -> None:
        """Welford 单遍在线算法计算 mean/std，数值稳定性优于朴素两遍法"""
        n_steps = self.num_steps
        k = len(self.seed_curves)
        if k == 0 or n_steps == 0:
            return

        for sc in self.seed_curves:
            if len(sc.values) != n_steps:
                raise ValueError(
                    f"Seed {sc.seed_id}: 实际 {len(sc.values)} 个点, "
                    f"期望 {n_steps}"
                )

        mean_arr: List[float] = []
        std_arr: List[float] = []

        for i in range(n_steps):
            # Welford 在线法
            welford_mean = 0.0
            welford_m2 = 0.0
            count = 0
            for sc in self.seed_curves:
                count += 1
                delta = sc.values[i] - welford_mean
                welford_mean += delta / count
                delta2 = sc.values[i] - welford_mean
                welford_m2 += delta * delta2

            mean_arr.append(welford_mean)
            if count > 1:
                std_arr.append(math.sqrt(welford_m2 / (count - 1)))
            else:
                std_arr.append(0.0)

        self._mean = mean_arr
        self._std = std_arr
        if mean_arr:
            self.reported_final = mean_arr[-1]

        _dbg("MethodResult",
             f"stats done: {self.strategy.value}, "
             f"final_mean={self.reported_final:.4f}, "
             f"avg_std={sum(std_arr)/len(std_arr):.4f}")

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

    def debug_snapshot(self) -> str:
        lines = [
            f"=== MethodResult: {self.strategy.value} ===",
            f"  steps={self.num_steps}, seeds={self.num_seeds}",
            f"  curves_attached={len(self.seed_curves)}",
            f"  total_cost={self.total_cost}",
            f"  reported_final={self.reported_final}",
        ]
        for sc in self.seed_curves:
            lines.append(f"  {sc.debug_snapshot()}")
        s = "\n".join(lines)
        _dbg("MethodResult", s)
        return s


@dataclass
class PanelResult:
    """一个面板（如 "TPC-H SF100"），包含多个方法的对比结果"""
    panel_name: str
    metric: MetricKind
    x_label: str = "step"
    y_label: str = "latency_us"
    methods: Dict[str, MethodResult] = field(default_factory=dict)

    def add_method(self, strategy: RoutingStrategy, num_steps: int,
                   num_seeds: int) -> MethodResult:
        mr = MethodResult(
            strategy=strategy,
            num_steps=num_steps,
            num_seeds=num_seeds,
        )
        self.methods[strategy.value] = mr
        _dbg("PanelResult",
             f"panel '{self.panel_name}': added method {strategy.value}")
        return mr

    def to_dict(self) -> Dict[str, Any]:
        return {name: mr.to_dict() for name, mr in self.methods.items()}

    def debug_snapshot(self) -> str:
        lines = [f"Panel '{self.panel_name}' ({self.metric.name}):"]
        for name, mr in self.methods.items():
            lines.append(f"  method={name}, steps={mr.num_steps}, "
                         f"seeds={mr.num_seeds}")
        s = "\n".join(lines)
        _dbg("PanelResult", s)
        return s


@dataclass
class BenchmarkOutput:
    """顶层基准测试输出——完整 JSON 产物"""
    description: str = ""
    source: str = ""
    timestamp: str = field(
        default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))
    panels: Dict[str, PanelResult] = field(default_factory=dict)

    def add_panel(self, name: str, metric: MetricKind,
                  x_label: str = "step",
                  y_label: str = "value") -> PanelResult:
        pr = PanelResult(panel_name=name, metric=metric,
                         x_label=x_label, y_label=y_label)
        self.panels[name] = pr
        _dbg("BenchmarkOutput", f"add_panel '{name}' metric={metric.name}")
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
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "panels": {n: pr.to_dict() for n, pr in self.panels.items()},
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        _dbg("BenchmarkOutput", f"saved to {p}")
        return p

    def debug_snapshot(self) -> str:
        meta = self.metadata
        lines = [
            "══════ BenchmarkOutput Snapshot ══════",
            f"  desc  = {self.description}",
            f"  source= {self.source}",
            f"  ts    = {self.timestamp}",
            f"  panels= {len(self.panels)}",
            f"  total_points= {meta['total_data_points']}",
        ]
        for name, pr in self.panels.items():
            lines.append(f"  ├─ panel '{name}': {len(pr.methods)} methods")
        s = "\n".join(lines)
        _dbg("BenchmarkOutput", s)
        return s


# ---------------------------------------------------------------------------
# 硬件拓扑节点
# ---------------------------------------------------------------------------

@dataclass
class HardwareNode:
    """拓扑图中的单个硬件节点"""
    node_id: str
    kind: HardwareKind
    compute_capacity: float = 1.0
    memory_bytes: int = 0
    bandwidth_gbps: float = 0.0
    latency_us: float = 0.0

    scan_cost_per_row: float = 0.02
    seek_cost: float = 0.5
    compute_cost_per_op: float = 0.01

    def debug_snapshot(self) -> str:
        s = (f"HwNode({self.node_id}, {self.kind.name}, "
             f"cap={self.compute_capacity}, mem={self.memory_bytes}, "
             f"scan={self.scan_cost_per_row}, seek={self.seek_cost})")
        _dbg("HardwareNode", s)
        return s


@dataclass
class TopologyEdge:
    """拓扑图中的有向边"""
    src: str
    dst: str
    bandwidth_gbps: float = 0.0
    latency_us: float = 0.0
    link_type: HardwareKind = HardwareKind.PCIE


@dataclass
class HardwareTopology:
    """完整硬件拓扑图——支持异构 GPU-CPU 配置"""
    nodes: Dict[str, HardwareNode] = field(default_factory=dict)
    edges: List[TopologyEdge] = field(default_factory=list)

    def add_node(self, node: HardwareNode) -> None:
        self.nodes[node.node_id] = node
        _dbg("Topology", f"add_node: {node.node_id} ({node.kind.name})")

    def add_edge(self, edge: TopologyEdge) -> None:
        self.edges.append(edge)

    def get_transfer_cost(self, src: str, dst: str,
                          data_bytes: int) -> float:
        """估算数据传输代价（微秒）。

        使用 A* 算法替代原版 Dijkstra：
        启发函数 h(u) = 基于链路类型的最小理论传输时间。
        在小图上差别不大，但保持了正确性和扩展性。
        """
        if src == dst:
            return 0.0
        if src not in self.nodes or dst not in self.nodes:
            return float("inf")

        def hop_weight(e: TopologyEdge) -> float:
            if e.bandwidth_gbps <= 0:
                return float("inf")
            transfer_us = (data_bytes / (e.bandwidth_gbps * 1e9 / 8)) * 1e6
            return e.latency_us + transfer_us

        # 构建邻接表
        adj: Dict[str, List[Tuple[str, float]]] = {}
        min_edge_cost = float("inf")
        for e in self.edges:
            w = hop_weight(e)
            if w < float("inf"):
                adj.setdefault(e.src, []).append((e.dst, w))
                min_edge_cost = min(min_edge_cost, w)

        # A* 启发函数：乐观估计——至少还要一跳
        def heuristic(u: str) -> float:
            if u == dst:
                return 0.0
            return min_edge_cost if min_edge_cost < float("inf") else 0.0

        # A* 搜索
        g_score: Dict[str, float] = {src: 0.0}
        open_set: List[Tuple[float, str]] = [(heuristic(src), src)]

        while open_set:
            f_val, u = heapq.heappop(open_set)
            if u == dst:
                _dbg("Topology",
                     f"path {src}->{dst}: cost={g_score[dst]:.2f}us, "
                     f"payload={data_bytes}B")
                return g_score[dst]
            current_g = g_score.get(u, float("inf"))
            if f_val - heuristic(u) > current_g:
                continue

            for v, w in adj.get(u, []):
                tentative = current_g + w
                if tentative < g_score.get(v, float("inf")):
                    g_score[v] = tentative
                    heapq.heappush(open_set, (tentative + heuristic(v), v))

        return g_score.get(dst, float("inf"))

    def get_node(self, node_id: str) -> Optional[HardwareNode]:
        return self.nodes.get(node_id)

    def debug_snapshot(self) -> str:
        lines = [
            f"Topology: {len(self.nodes)} nodes, {len(self.edges)} edges",
        ]
        for nid, nd in self.nodes.items():
            lines.append(f"  node {nid}: {nd.kind.name}, "
                         f"cap={nd.compute_capacity:.2f}")
        for e in self.edges:
            lines.append(f"  edge {e.src}->{e.dst}: "
                         f"{e.bandwidth_gbps}Gbps, {e.latency_us}us")
        s = "\n".join(lines)
        _dbg("Topology", s)
        return s
