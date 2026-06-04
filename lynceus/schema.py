"""
lynceus/schema.py — Canonical data schema for benchmark outputs.

Architecture reference:
    - ncclResult_t, DistributedDataParallelConfig, KVCacheSpec
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


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


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class SeedCurve:
    seed_id: int
    values: List[float] = field(default_factory=list)

    def append(self, v: float) -> None:
        self.values.append(v)


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
        """Compute per-step mean/std across seeds.
        改动: 用 Welford 在线算法代替两遍扫描, 数值更稳定。
        """
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
            # Welford online: single pass, numerically stable
            w_mean = 0.0
            w_m2 = 0.0
            w_count = 0
            for sc in self.seed_curves:
                if i >= len(sc.values):
                    continue
                w_count += 1
                delta = sc.values[i] - w_mean
                w_mean += delta / w_count
                delta2 = sc.values[i] - w_mean
                w_m2 += delta * delta2

            mean_out.append(w_mean)
            if w_count > 1:
                std_out.append(math.sqrt(w_m2 / (w_count - 1)))
            else:
                std_out.append(0.0)

        self._mean = mean_out
        self._std = std_out
        if mean_out:
            self.reported_final = mean_out[-1]

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
        return p


# ---------------------------------------------------------------------------
# Hardware topology node
# ---------------------------------------------------------------------------

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

    def add_edge(self, edge: TopologyEdge) -> None:
        self.edges.append(edge)

    def get_transfer_cost(self, src: str, dst: str, data_bytes: int) -> float:
        """Multi-hop shortest path via Dijkstra (INV-4).
        改动: 用 Fibonacci-heap 风格的懒删除替代原版简单堆,
        加 visited set 提前终止, 减少松弛次数。
        """
        from ._debug import checkpoint
        if src == dst:
            return 0.0
        if src not in self.nodes or dst not in self.nodes:
            return float("inf")

        def hop_cost(e: "TopologyEdge") -> float:
            if e.bandwidth_gbps <= 0:
                return float("inf")
            return e.latency_us + (data_bytes / (e.bandwidth_gbps * 1e9 / 8)) * 1e6

        adj: Dict[str, List[Tuple[str, float]]] = {}
        for e in self.edges:
            w = hop_cost(e)
            if w != float("inf"):
                adj.setdefault(e.src, []).append((e.dst, w))

        import heapq
        dist: Dict[str, float] = {src: 0.0}
        visited: set = set()
        pq: List[Tuple[float, str]] = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                checkpoint("dijkstra_found", src=src, dst=dst, cost=d)
                return d
            if u in visited:
                continue
            visited.add(u)
            for v, w in adj.get(u, ()):
                nd = d + w
                if v not in visited and nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist.get(dst, float("inf"))

    def get_node(self, node_id: str) -> Optional[HardwareNode]:
        return self.nodes.get(node_id)
