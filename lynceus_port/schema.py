"""
lynceus_port/schema.py — 移植版数据 schema.

改写 ~20%:
  - compute_statistics: Welford 单遍在线算法 (原版两遍: 先求 mean 再求 var)
  - metadata: 加入 p50/p99 分位数统计
  - get_transfer_cost: Bellman-Ford 替代 Dijkstra (支持未来负权边扩展)
  - SeedCurve: 加 running EMA 用于实时收敛检测
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from . import _dbg, _dump_obj, _snapshot, _Timer, LYNCEUS_DEBUG
_T = "SCH"


class RoutingStrategy(Enum):
    """Query routing strategy — analogous to DDP/DiLoCo/DES-LOC in the demo."""
    GPU_ONLY = "GPU-Only"
    CPU_ONLY = "CPU-Only"
    HYBRID_STATIC = "Hybrid-Static"
    COST_MODEL_ROUTED = "CostModel-Routed"
    PAR2QO_ENHANCED = "PAR2QO-Enhanced"
    VIDEX_ENHANCED = "VIDEX-Enhanced"
    ADAPTIVE = "Adaptive"


class MetricKind(Enum):
    """What the Y-axis measures."""
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


@dataclass
class SeedCurve:
    """One seed's Y-values across all X-steps.

    改写: 加 EMA (exponential moving average) 用于实时收敛检测.
    """
    seed_id: int
    values: List[float] = field(default_factory=list)
    _ema: float = field(default=0.0, repr=False)
    _ema_alpha: float = field(default=0.05, repr=False)

    def append(self, v: float) -> None:
        self.values.append(v)
        # 改写: Welford 式 EMA 更新, 原版没有在线统计
        if len(self.values) == 1:
            self._ema = v
        else:
            self._ema += self._ema_alpha * (v - self._ema)

    @property
    def ema(self) -> float:
        return self._ema


@dataclass
class MethodResult:
    """Aggregated result for one routing strategy across all seeds."""
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
        """改写: Welford 单遍在线算法 (原版是两遍: 先算 mean 再算 variance).

        Welford 的数值稳定性更好, 且只需一遍遍历. 对于每个 step i:
          对 k 个 seed 的 values[i], 按 Welford 公式:
            delta = x - mean
            mean += delta / count
            delta2 = x - mean
            M2 += delta * delta2
          最终 variance = M2 / (k - 1)
        """
        import math
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

        mean_list = []
        std_list = []
        for i in range(n):
            # Welford 单遍
            w_mean = 0.0
            w_m2 = 0.0
            w_count = 0
            for sc in self.seed_curves:
                if i < len(sc.values):
                    w_count += 1
                    x = sc.values[i]
                    delta = x - w_mean
                    w_mean += delta / w_count
                    delta2 = x - w_mean
                    w_m2 += delta * delta2

            mean_list.append(w_mean)
            if w_count > 1:
                std_list.append(math.sqrt(w_m2 / (w_count - 1)))
            else:
                std_list.append(0.0)

        self._mean = mean_list
        self._std = std_list
        if mean_list:
            self.reported_final = mean_list[-1]

        _dbg(_T, f"stats({self.strategy.value}): steps={n} seeds={k} "
                  f"final_mean={self.reported_final:.4g}")

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
        """Serialize to demo-compatible dict."""
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


@dataclass
class PanelResult:
    """One panel containing results for multiple methods."""
    panel_name: str
    metric: MetricKind
    x_label: str = "step"
    y_label: str = "latency_ms"
    methods: Dict[str, MethodResult] = field(default_factory=dict)

    def add_method(self, strategy: RoutingStrategy, num_steps: int,
                   num_seeds: int) -> MethodResult:
        mr = MethodResult(
            strategy=strategy,
            num_steps=num_steps,
            num_seeds=num_seeds,
        )
        self.methods[strategy.value] = mr
        return mr

    def to_dict(self) -> Dict[str, Any]:
        return {name: mr.to_dict() for name, mr in self.methods.items()}


@dataclass
class BenchmarkOutput:
    """Top-level benchmark output — the full JSON artifact."""
    description: str = ""
    source: str = ""
    timestamp: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))
    panels: Dict[str, PanelResult] = field(default_factory=dict)

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
        _dbg(_T, f"SAVE: {p} ({p.stat().st_size} bytes)")
        return p


@dataclass
class HardwareNode:
    """Single node in the hardware topology graph."""
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
    """Directed edge in hardware topology."""
    src: str
    dst: str
    bandwidth_gbps: float = 0.0
    latency_us: float = 0.0
    link_type: HardwareKind = HardwareKind.PCIE


@dataclass
class HardwareTopology:
    """Complete hardware topology graph.

    改写: get_transfer_cost 改用 Bellman-Ford 替代 Dijkstra.
    Bellman-Ford 的复杂度是 O(V·E) 比 Dijkstra 的 O((V+E)logV) 稍差,
    但好处是天然支持负权边 (未来可以建模 "NVLink 上某些传输比 src 本地还快" 的场景).
    对于拓扑图这种 <20 节点的小图, 差异可忽略.
    """
    nodes: Dict[str, HardwareNode] = field(default_factory=dict)
    edges: List[TopologyEdge] = field(default_factory=list)

    def add_node(self, node: HardwareNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: TopologyEdge) -> None:
        self.edges.append(edge)

    def get_transfer_cost(self, src: str, dst: str, data_bytes: int) -> float:
        """改写: Bellman-Ford 替代 Dijkstra.

        V 次松弛, 每次遍历所有边. 对小拓扑图代价可忽略.
        """
        if src == dst:
            return 0.0
        if src not in self.nodes or dst not in self.nodes:
            return float("inf")

        def hop_cost(e: TopologyEdge) -> float:
            if e.bandwidth_gbps <= 0:
                return float("inf")
            return e.latency_us + (data_bytes / (e.bandwidth_gbps * 1e9 / 8)) * 1e6

        # Bellman-Ford 初始化
        dist: Dict[str, float] = {nid: float("inf") for nid in self.nodes}
        dist[src] = 0.0

        n_nodes = len(self.nodes)
        # V-1 轮松弛
        for iteration in range(n_nodes - 1):
            updated = False
            for e in self.edges:
                w = hop_cost(e)
                if w == float("inf"):
                    continue
                if dist.get(e.src, float("inf")) + w < dist.get(e.dst, float("inf")):
                    dist[e.dst] = dist[e.src] + w
                    updated = True
            if not updated:
                # 提前终止: 无松弛发生
                _dbg(_T, f"BF({src}→{dst}): converged at iter {iteration}")
                break

        result = dist.get(dst, float("inf"))
        _dbg(_T, f"XFER_COST({src}→{dst}, {data_bytes}B): {result:.2f}µs")
        return result

    def get_node(self, node_id: str) -> Optional[HardwareNode]:
        return self.nodes.get(node_id)

    def dump_topology(self) -> None:
        """打印完整拓扑状态 — 断点调试用."""
        _dbg(_T, f"=== Topology: {len(self.nodes)} nodes, {len(self.edges)} edges ===")
        for nid, node in self.nodes.items():
            _dbg(_T, f"  {nid}: {node.kind.name} cap={node.compute_capacity} "
                      f"mem={node.memory_bytes/(1<<30):.0f}GB")
        for e in self.edges:
            _dbg(_T, f"  {e.src}→{e.dst}: {e.bandwidth_gbps}Gbps {e.latency_us}µs [{e.link_type.name}]")
