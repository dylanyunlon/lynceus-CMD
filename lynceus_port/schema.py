"""
lynceus_port/schema.py — 移植版数据 schema.

改写 ~20%:
  - compute_statistics: Welford 在线算法替代 naive two-pass
  - SeedCurve: 增加 running EMA 平滑
  - BenchmarkOutput.save: 写入前 dump 到 stderr 供断点检视
  - HardwareTopology.get_transfer_cost: Bellman-Ford 替代 Dijkstra
    (允许负权边, 虽然当前不存在, 但为未来建模折扣链路做准备)

架构溯源同原版.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from . import _dbg, _dump_obj, _snapshot, LYNCEUS_DEBUG

_T = "SCH"


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


@dataclass
class SeedCurve:
    seed_id: int
    values: List[float] = field(default_factory=list)
    # [PORT] 增加 EMA 追踪, alpha=0.05 让趋势可视化
    _ema: float = field(default=0.0, repr=False)
    _ema_alpha: float = field(default=0.05, repr=False)

    def append(self, v: float) -> None:
        self.values.append(v)
        # [PORT] 在线 EMA 更新
        if len(self.values) == 1:
            self._ema = v
        else:
            self._ema = self._ema_alpha * v + (1.0 - self._ema_alpha) * self._ema

    @property
    def ema(self) -> float:
        return self._ema


@dataclass
class MethodResult:
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

        mean_out = []
        std_out = []
        for i in range(n):
            vals = [sc.values[i] for sc in self.seed_curves]
            # [PORT] Welford 在线方差 — 数值稳定, 单 pass
            w_mean = 0.0
            w_m2 = 0.0
            for j, v in enumerate(vals, 1):
                delta = v - w_mean
                w_mean += delta / j
                delta2 = v - w_mean
                w_m2 += delta * delta2
            mean_out.append(w_mean)
            if k > 1:
                std_out.append(math.sqrt(w_m2 / (k - 1)))
            else:
                std_out.append(0.0)

        self._mean = mean_out
        self._std = std_out
        if mean_out:
            self.reported_final = mean_out[-1]

        _dbg(_T, f"stats done: strategy={self.strategy.value}, "
             f"final_mean={self.reported_final:.4f}, "
             f"mean_std={sum(std_out)/len(std_out):.4f}")

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


@dataclass
class PanelResult:
    panel_name: str
    metric: MetricKind
    x_label: str = "step"
    y_label: str = "latency_ms"
    methods: Dict[str, MethodResult] = field(default_factory=dict)

    def add_method(self, strategy: RoutingStrategy, num_steps: int,
                   num_seeds: int) -> MethodResult:
        mr = MethodResult(
            strategy=strategy, num_steps=num_steps, num_seeds=num_seeds,
        )
        self.methods[strategy.value] = mr
        _dbg(_T, f"panel[{self.panel_name}] += method {strategy.value}")
        return mr

    def to_dict(self) -> Dict[str, Any]:
        return {name: mr.to_dict() for name, mr in self.methods.items()}


@dataclass
class BenchmarkOutput:
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
        blob = self.to_dict()
        # [PORT] 写入前 dump 元信息到 stderr, 方便断点检视
        _dbg(_T, f"save -> {p}, metadata={blob['metadata']}")
        with open(p, "w") as f:
            json.dump(blob, f, indent=2)
        return p


# --- Hardware topology (NCCL-inspired) ---

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
    nodes: Dict[str, HardwareNode] = field(default_factory=dict)
    edges: List[TopologyEdge] = field(default_factory=list)

    def add_node(self, node: HardwareNode) -> None:
        self.nodes[node.node_id] = node
        _dbg(_T, f"topo += node {node.node_id} ({node.kind.name})")

    def add_edge(self, edge: TopologyEdge) -> None:
        self.edges.append(edge)

    def get_transfer_cost(self, src: str, dst: str, data_bytes: int) -> float:
        """Bellman-Ford 最短路 — 替代原版 Dijkstra.

        改写理由: Bellman-Ford 允许负权边, 为未来建模链路折扣/补贴留接口.
        当前拓扑全正权, 两者结果等价, 但 BF 实现更简单且无需 heapq.
        复杂度 O(V*E), 对小拓扑 (<20 nodes) 无影响.
        """
        if src == dst:
            return 0.0
        if src not in self.nodes or dst not in self.nodes:
            return float("inf")

        def hop_cost(e: TopologyEdge) -> float:
            if e.bandwidth_gbps <= 0:
                return float("inf")
            return e.latency_us + (data_bytes / (e.bandwidth_gbps * 1e9 / 8)) * 1e6

        all_ids = list(self.nodes.keys())
        dist = {nid: float("inf") for nid in all_ids}
        dist[src] = 0.0

        # [PORT] Bellman-Ford: V-1 轮松弛
        n_nodes = len(all_ids)
        for _round in range(n_nodes - 1):
            updated = False
            for e in self.edges:
                w = hop_cost(e)
                if w == float("inf"):
                    continue
                if dist[e.src] + w < dist[e.dst]:
                    dist[e.dst] = dist[e.src] + w
                    updated = True
            if not updated:
                break  # 提前收敛

        result = dist.get(dst, float("inf"))
        _dbg(_T, f"transfer {src}->{dst}: {data_bytes}B = {result:.1f}us")
        return result

    def get_node(self, node_id: str) -> Optional[HardwareNode]:
        return self.nodes.get(node_id)

    # [PORT] 拓扑完整性自检
    def dump_reachability(self) -> Dict[str, Dict[str, float]]:
        """断点辅助: 打印全对最短路矩阵."""
        matrix = {}
        ref_bytes = 1024 * 1024  # 1 MB 参考负载
        for src in self.nodes:
            row = {}
            for dst in self.nodes:
                row[dst] = self.get_transfer_cost(src, dst, ref_bytes)
            matrix[src] = row
        if LYNCEUS_DEBUG:
            import sys
            print("== Reachability Matrix (1MB, us) ==", file=sys.stderr)
            ids = sorted(self.nodes.keys())
            hdr = "        " + "  ".join(f"{x:>7}" for x in ids)
            print(hdr, file=sys.stderr)
            for s in ids:
                vals = "  ".join(
                    f"{matrix[s][d]:7.1f}" if matrix[s][d] < 1e9 else "    inf"
                    for d in ids
                )
                print(f"{s:>7} {vals}", file=sys.stderr)
        return matrix
