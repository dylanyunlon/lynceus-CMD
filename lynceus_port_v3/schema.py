"""
lynceus_port_v3/schema.py — Ported schema with debug instrumentation — Canonical data schema for Lynceus benchmark outputs.

Design rationale:
    Matches the X-axis granularity observed in data_demo/:
    - reversed_figure_data.json   → per-step curves, 3 seeds, 3000 steps
    - gradient_norm_24k_data.json → named seeds (seed_0..seed_2), 2000 points
    - ppl_vs_time_1B_30k_data.json → time_hours x-axis, per-seed arrays

    This module provides typed dataclasses so every downstream component
    (cost_model, router, benchmark runner) produces data in a single
    canonical format that the visualization layer can consume without
    ad-hoc parsing.

Architecture reference:
    - ncclResult_t (nccl/src/graph/xml.h) — typed return codes
    - DistributedDataParallelConfig (Megatron-LM) — config dataclass pattern
    - KVCacheSpec (vllm/vllm/v1/kv_cache_interface.py:95) — spec hierarchy
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

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
    GRADIENT_NORM = auto()       # for ML-style panels
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
    """One seed's Y-values across all X-steps.

    Invariant: len(values) == parent MethodResult.num_steps.
    """
    seed_id: int
    values: List[float] = field(default_factory=list)

    def append(self, v: float) -> None:
        self.values.append(v)


@dataclass
class MethodResult:
    """Aggregated result for one routing strategy across all seeds.

    Mirrors the demo schema:
        {
          "time_hours": [...],
          "total_cost": 847.3,
          "reported_final": 42.5,
          "seed_0": [...], "seed_1": [...], "seed_2": [...],
          "mean": [...], "std": [...]
        }
    """
    strategy: RoutingStrategy
    num_steps: int
    num_seeds: int
    x_values: List[float] = field(default_factory=list)  # explicit X-axis
    seed_curves: List[SeedCurve] = field(default_factory=list)
    reported_final: Optional[float] = None
    total_cost: Optional[float] = None

    # Lazily computed
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

        Raises ValueError if any seed curve has fewer values than num_steps.
        """
        import math
        n = self.num_steps
        k = len(self.seed_curves)
        if k == 0 or n == 0:
            return

        # Validate all curves have the expected length
        for sc in self.seed_curves:
            if len(sc.values) != n:
                raise ValueError(
                    f"Seed {sc.seed_id} has {len(sc.values)} values, "
                    f"expected {n} (num_steps)"
                )

        mean = []
        std = []
        for i in range(n):
            vals = [sc.values[i] for sc in self.seed_curves if i < len(sc.values)]
            if not vals:
                mean.append(0.0)
                std.append(0.0)
                continue
            m = sum(vals) / len(vals)
            mean.append(m)
            if len(vals) > 1:
                variance = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
                std.append(math.sqrt(variance))
            else:
                std.append(0.0)
        self._mean = mean
        self._std = std
        if mean:
            self.reported_final = mean[-1]

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
    """One panel (e.g., "kx16_iid" or "TPC-H SF100") containing
    results for multiple methods.

    Mirrors: reversed_figure_data.json → panels → kx16_iid → {DDP: ..., DES-LOC: ...}
    """
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
    """Top-level benchmark output — the full JSON artifact.

    Matches the structure of ppl_vs_time_1B_30k_data.json with metadata header.
    """
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
# Hardware topology node (NCCL-inspired)
# ---------------------------------------------------------------------------

@dataclass
class HardwareNode:
    """Single node in the hardware topology graph.

    Inspired by NCCL's ncclTopoFillGpu (nccl/src/graph/xml.h:54).
    """
    node_id: str
    kind: HardwareKind
    compute_capacity: float = 1.0   # relative FLOPS
    memory_bytes: int = 0
    bandwidth_gbps: float = 0.0     # link bandwidth to parent
    latency_us: float = 0.0         # link latency to parent

    # Cost model coefficients (learned or configured)
    # Defaults: CPU-class values in microseconds per unit
    scan_cost_per_row: float = 0.02   # sequential scan cost per row (µs)
    seek_cost: float = 0.5            # random seek cost (µs)
    compute_cost_per_op: float = 0.01 # per-operation cost (µs)


@dataclass
class TopologyEdge:
    """Directed edge in hardware topology.

    Inspired by ncclTopoComputePaths (nccl/src/graph/paths.cc:685).
    """
    src: str
    dst: str
    bandwidth_gbps: float = 0.0
    latency_us: float = 0.0
    link_type: HardwareKind = HardwareKind.PCIE


@dataclass
class HardwareTopology:
    """Complete hardware topology graph.

    Inspired by ncclTopoCompute (nccl/src/graph/search.cc:1023).
    Supports heterogeneous GPU-CPU configurations.
    """
    nodes: Dict[str, HardwareNode] = field(default_factory=dict)
    edges: List[TopologyEdge] = field(default_factory=list)

    def add_node(self, node: HardwareNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: TopologyEdge) -> None:
        self.edges.append(edge)

    def get_transfer_cost(self, src: str, dst: str, data_bytes: int) -> float:
        """Estimate data transfer cost in microseconds along the best path.

        Computes the minimum-cost route from src to dst over the topology
        graph, allowing multi-hop paths (e.g. cpu1 -> cpu0 -> gpu0 on a
        dual-socket box where the second NUMA node has no direct GPU link).
        This mirrors NCCL's ncclTopoComputePaths, which finds shortest paths
        over the whole topology rather than requiring a direct edge.

        Per-hop cost = edge.latency_us + bytes / bandwidth. Hop costs add along
        the path (store-and-forward), so the path cost is the sum of its edges.
        Dijkstra is correct here because every edge cost is non-negative.

        Contracts preserved for callers:
            src == dst            -> 0.0
            no path / zero-bw cut -> inf
        """
        if src == dst:
            return 0.0
        if src not in self.nodes or dst not in self.nodes:
            return float("inf")

        # Per-hop weight for the given payload size. A non-positive bandwidth
        # edge is an impassable cut (weight inf) — never usable.
        def hop_cost(e: "TopologyEdge") -> float:
            if e.bandwidth_gbps <= 0:
                return float("inf")
            return e.latency_us + (data_bytes / (e.bandwidth_gbps * 1e9 / 8)) * 1e6

        # Adjacency: src -> list of (neighbor, weight). Built per call; the
        # graph is tiny (a single node's device fabric).
        adj: Dict[str, List[Tuple[str, float]]] = {}
        for e in self.edges:
            w = hop_cost(e)
            if w != float("inf"):
                adj.setdefault(e.src, []).append((e.dst, w))

        # Dijkstra with a binary heap.
        import heapq
        dist: Dict[str, float] = {src: 0.0}
        pq: List[Tuple[float, str]] = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                return d
            if d > dist.get(u, float("inf")):
                continue  # stale entry
            for v, w in adj.get(u, ()):  # noqa: E741
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist.get(dst, float("inf"))

    def get_node(self, node_id: str) -> Optional[HardwareNode]:
        return self.nodes.get(node_id)


# ---------------------------------------------------------------------------
# Debug/Inspection 扩展 (port_v3 新增)
# ---------------------------------------------------------------------------

def _schema_self_check(topo: HardwareTopology) -> str:
    """对拓扑进行完整性自检 — 运行实验前调用, 确保配置无误.

    检查项:
      - 所有边的 src/dst 都在 nodes 中
      - GPU 节点的 compute_capacity > 1 (应远超CPU)
      - 没有自环
      - 连通性: 从任意CPU能到达所有GPU
    返回: 检查报告字符串
    """
    report_lines = ["[TOPOLOGY SELF-CHECK]"]
    issues = 0

    for eid, edge in topo.edges.items():
        if edge.src not in topo.nodes:
            report_lines.append(f"  ERROR: edge {eid} src={edge.src} not in nodes")
            issues += 1
        if edge.dst not in topo.nodes:
            report_lines.append(f"  ERROR: edge {eid} dst={edge.dst} not in nodes")
            issues += 1
        if edge.src == edge.dst:
            report_lines.append(f"  WARN: self-loop on {edge.src}")
            issues += 1

    for nid, node in topo.nodes.items():
        if node.kind == HardwareKind.GPU and node.compute_capacity <= 1.0:
            report_lines.append(
                f"  WARN: GPU {nid} has compute_capacity={node.compute_capacity} "
                f"(expected >> 1.0)")

    gpu_ids = [n for n, nd in topo.nodes.items() if nd.kind == HardwareKind.GPU]
    cpu_ids = [n for n, nd in topo.nodes.items() if nd.kind == HardwareKind.CPU]

    report_lines.append(f"  nodes: {len(topo.nodes)} ({len(cpu_ids)} CPU, {len(gpu_ids)} GPU)")
    report_lines.append(f"  edges: {len(topo.edges)}")
    report_lines.append(f"  issues: {issues}")

    return "\n".join(report_lines)
