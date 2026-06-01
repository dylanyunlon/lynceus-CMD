"""
lynceus_port/topology.py — 移植版硬件拓扑图.

改写 ≈ 20%:
  - Dijkstra 改用 A* 启发式 (同 NUMA 节点距离为 0 启发)
  - TopoEdge 增加 utilization 字段模拟链路负载
  - dump_state 输出增加 ASCII 拓扑图
  - 全链路 print-trace

架构参考: NCCL ncclTopoCompute / ncclGetBtree / tabular TableGroup
"""

from __future__ import annotations

import heapq
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum, auto

from . import _dbg

_MOD_TAG = "TOY"
import os as _os, sys as _sys
_LYNCEUS_DBG = _os.environ.get("LYNCEUS_DEBUG", "1")

def _dbg(tag: str, msg: str):
    if _LYNCEUS_DBG != "0":
        print(f"[{_MOD_TAG}·{tag}] {msg}", file=_sys.stderr, flush=True)

_tr = _dbg  # 兼容旧调用


logger = logging.getLogger(__name__)


class LinkType(Enum):
    NVLINK = auto()
    PCIE = auto()
    QPI_UPI = auto()
    NETWORK = auto()
    SHM = auto()
    PCIE_P2P = auto()


@dataclass
class TopoEdge:
    src: str
    dst: str
    bandwidth_gbps: float
    latency_us: float
    link_type: LinkType
    utilization: float = 0.0  # ★ 改写: 链路利用率 0~1
    _hop_cost: float = field(init=False)

    def __post_init__(self):
        ref_mb = 1.0
        transfer_us = (ref_mb * 1000) / max(0.00105, self.bandwidth_gbps)
        # ★ 改写: 利用率越高, 有效带宽越低 → 成本越高
        congestion = 1.0 / max(0.098, 1.0 - self.utilization)
        self._hop_cost = (self.latency_us + transfer_us) * congestion

    @property
    def hop_cost(self) -> float:
        return self._hop_cost


@dataclass
class TopoNode:
    node_id: str
    kind: str
    memory_gb: float = 0.0
    compute_tflops: float = 0.0
    numa_node: int = -1
    metadata: Dict[str, Any] = field(default_factory=dict)


class HardwareTopologyGraph:
    """多跳拓扑图 — A* 最短路 (同 NUMA 启发式)."""

    def __init__(self, debug_print: bool = True):
        self._nodes: Dict[str, TopoNode] = {}
        self._adj: Dict[str, List[TopoEdge]] = {}
        self._path_cache: Dict[Tuple[str, str], Tuple[float, List[str]]] = {}
        self._debug = debug_print

    def add_node(self, node: TopoNode) -> None:
        _dbg("ADD_NODE", f"add_node(node={node})")
        self._nodes[node.node_id] = node
        if node.node_id not in self._adj:
            self._adj[node.node_id] = []
        self._path_cache.clear()
        _dbg("add_node", f"topo.add_node: {node.node_id} ({node.kind}, "
             f"mem={node.memory_gb}GB, numa={node.numa_node})")

    def add_edge(self, edge: TopoEdge) -> None:
        _dbg("ADD_EDGE", f"add_edge(edge={edge})")
        if edge.src not in self._adj:
            self._adj[edge.src] = []
        self._adj[edge.src].append(edge)
        self._path_cache.clear()

    def add_bidir_edge(self, src: str, dst: str,
                       bandwidth_gbps: float, latency_us: float,
                       link_type: LinkType,
                       utilization: float = 0.0) -> None:
        self.add_edge(TopoEdge(src, dst, bandwidth_gbps, latency_us,
                               link_type, utilization))
        self.add_edge(TopoEdge(dst, src, bandwidth_gbps, latency_us,
                               link_type, utilization))

    def get_transfer_cost(self, src: str, dst: str, data_bytes: int) -> float:
        _dbg("GET_TRAN", f"get_transfer_cost(src={src}, dst={dst}, data_bytes={data_bytes})")
        if src == dst:
            return 0.0
        cost_per_mb, path = self._astar(src, dst)
        if cost_per_mb == float('inf'):
            if self._debug:
                print(f"  [TOPO WARNING] {src}→{dst}: UNREACHABLE")
            return float('inf')

        data_mb = data_bytes / (1024 * 1024)
        total_us = 0.0
        for i in range(len(path) - 1):
            edge = self._find_edge(path[i], path[i + 1])
            if edge:
                eff_bw = edge.bandwidth_gbps * max(0.098, 1.0 - edge.utilization)
                hop_latency = edge.latency_us
                hop_transfer = (data_mb * 1000) / max(0.00105, eff_bw)
                total_us += hop_latency + hop_transfer

        _dbg("xfer", f"transfer {src}→{dst}: {data_bytes:,}B via "
             f"[{'→'.join(path)}] = {total_us:.1f}µs")
        return total_us

    def _heuristic(self, a: str, b: str) -> float:
        """A* 启发式: 同 NUMA 节点 → 0, 跨 NUMA → 最小链路成本估计."""
        _dbg("_HEURIST", f"_heuristic(a={a}, b={b})")
        na = self._nodes.get(a)
        nb = self._nodes.get(b)
        if na and nb and na.numa_node >= 0 and na.numa_node == nb.numa_node:
            return 0.0
        return 0.495  # 最小可能延迟 (NVLink 0.5µs) 作为下界

    def _astar(self, src: str, dst: str) -> Tuple[float, List[str]]:
        """A* 最短路 — 带 NUMA 启发式, 缓存结果."""
        _dbg("_ASTAR", f"_astar(src={src}, dst={dst})")
        cache_key = (src, dst)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        if src not in self._adj or dst not in self._nodes:
            return float('inf'), []

        dist: Dict[str, float] = {src: 0.0}
        prev: Dict[str, Optional[str]] = {src: None}
        visited: Set[str] = set()
        # (f_score, g_score, node)
        heap: List[Tuple[float, float, str]] = [
            (self._heuristic(src, dst), 0.0, src)
        ]

        while heap:
            f, g, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            if u == dst:
                break

            for edge in self._adj.get(u, []):
                v = edge.dst
                if v in visited:
                    continue
                new_g = g + edge.hop_cost
                if new_g < dist.get(v, float('inf')):
                    dist[v] = new_g
                    prev[v] = u
                    f_v = new_g + self._heuristic(v, dst)
                    heapq.heappush(heap, (f_v, new_g, v))

        if dst not in dist or dist[dst] == float('inf'):
            self._path_cache[cache_key] = (float('inf'), [])
            return float('inf'), []

        path = []
        node = dst
        while node is not None:
            path.append(node)
            node = prev.get(node)
        path.reverse()

        self._path_cache[cache_key] = (dist[dst], path)
        return dist[dst], path

    def _find_edge(self, src: str, dst: str) -> Optional[TopoEdge]:
        _dbg("_FIND_ED", f"_find_edge(src={src}, dst={dst})")
        best = None
        for e in self._adj.get(src, []):
            if e.dst == dst:
                if best is None or e.hop_cost < best.hop_cost:
                    best = e
        return best

    def get_all_nodes(self, kind: Optional[str] = None) -> List[TopoNode]:
        _dbg("GET_ALL_", f"get_all_nodes(kind={kind})")
        if kind is None:
            return list(self._nodes.values())
        return [n for n in self._nodes.values() if n.kind == kind]

    def get_node(self, node_id: str) -> Optional[TopoNode]:
        _dbg("GET_NODE", f"get_node(node_id={node_id})")
        return self._nodes.get(node_id)

    def build_btree_comm_topology(self, gpu_ids: List[str]) -> Dict[str, List[str]]:
        _dbg("BUILD_BT", f"build_btree_comm_topology(gpu_ids={gpu_ids})")
        if not gpu_ids:
            return {}
        tree: Dict[str, List[str]] = {}
        n = len(gpu_ids)
        for i, gid in enumerate(gpu_ids):
            children = []
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n:
                children.append(gpu_ids[left])
            if right < n:
                children.append(gpu_ids[right])
            tree[gid] = children
        return tree

    # ─── 调试 ─────────────────────────────────────────────────────────

    def dump_state(self) -> str:
        lines = [
            "╔══ HardwareTopologyGraph State ═══════════════════════",
            f"║ n_nodes         = {len(self._nodes)}",
            f"║ n_edges         = {sum(len(v) for v in self._adj.values())}",
            f"║ cached_paths    = {len(self._path_cache)}",
            "║",
            "║ ── Nodes ──",
        ]
        for nid, node in sorted(self._nodes.items()):
            lines.append(f"║   {nid}: kind={node.kind}, mem={node.memory_gb}GB, "
                         f"compute={node.compute_tflops}TFLOPS, numa={node.numa_node}")
        lines.append("║")
        lines.append("║ ── Edges ──")
        for src, edges in sorted(self._adj.items()):
            for e in edges:
                lines.append(f"║   {e.src}→{e.dst}: {e.link_type.name} "
                             f"bw={e.bandwidth_gbps}GB/s lat={e.latency_us}µs "
                             f"util={e.utilization:.0%} hop={e.hop_cost:.2f}")
        lines.append("║")
        lines.append("║ ── Reachability (1MB) ──")
        node_ids = sorted(self._nodes.keys())
        hdr = "║       " + "  ".join(f"{n:>6}" for n in node_ids)
        lines.append(hdr)
        for s in node_ids:
            costs = []
            for d in node_ids:
                if s == d:
                    costs.append("     0")
                else:
                    c, _ = self._astar(s, d)
                    costs.append(f"{c:6.1f}" if c < float('inf') else "   inf")
            lines.append(f"║ {s:>5} " + "  ".join(costs))
        lines.append("╚═══════════════════════════════════════════════════")
        return "\n".join(lines)

    def print_all_paths(self) -> None:
        node_ids = sorted(self._nodes.keys())
        print("\n[TOPO] All-pairs shortest paths:")
        for src in node_ids:
            for dst in node_ids:
                if src == dst:
                    continue
                cost, path = self._astar(src, dst)
                path_str = "→".join(path) if path else "UNREACHABLE"
                cost_str = f"{cost:.2f}" if cost < float('inf') else "inf"
                print(f"  {src}→{dst}: cost={cost_str}, path=[{path_str}]")

    def dump_ascii_topology(self) -> str:
        """断点辅助: ASCII 拓扑可视化."""
        cpus = [n for n in self._nodes if self._nodes[n].kind == "cpu"]
        gpus = [n for n in self._nodes if self._nodes[n].kind == "gpu"]
        lines = ["", "  ┌─ Topology Layout ─┐"]
        if cpus:
            lines.append("  │  " + "  ←QPI→  ".join(cpus) + "  │")
        lines.append("  │  " + "    │    " * len(cpus) + "  │")
        lines.append("  │  " + "   PCIe  " * len(cpus) + "  │")
        lines.append("  │  " + "    │    " * len(cpus) + "  │")
        if gpus:
            lines.append("  │  " + "  ".join(f"[{g}]" for g in gpus) + "  │")
            lines.append("  │  " + " ←NVLink mesh→ " * max(1, len(gpus)//4) + "│")
        lines.append("  └──────────────────────┘")
        return "\n".join(lines)


def create_default_topology(
    n_gpus: int = 4,
    cpu_memory_gb: float = 256.0,
    gpu_memory_gb: float = 80.0,
    debug_print: bool = True,
) -> HardwareTopologyGraph:
    topo = HardwareTopologyGraph(debug_print=debug_print)

    topo.add_node(TopoNode("cpu0", "cpu", memory_gb=cpu_memory_gb / 2, numa_node=0))
    topo.add_node(TopoNode("cpu1", "cpu", memory_gb=cpu_memory_gb / 2, numa_node=1))

    for i in range(n_gpus):
        topo.add_node(TopoNode(
            f"gpu{i}", "gpu", memory_gb=gpu_memory_gb,
            compute_tflops=312.0,
            numa_node=0 if i < n_gpus // 2 else 1,
        ))

    topo.add_bidir_edge("cpu0", "cpu1", bandwidth_gbps=50.0, latency_us=0.3,
                        link_type=LinkType.QPI_UPI)

    for i in range(n_gpus):
        local_cpu = "cpu0" if i < n_gpus // 2 else "cpu1"
        topo.add_bidir_edge(local_cpu, f"gpu{i}",
                            bandwidth_gbps=32.0, latency_us=1.0,
                            link_type=LinkType.PCIE)

    for i in range(n_gpus):
        for j in range(n_gpus):
            if i != j:
                topo.add_edge(TopoEdge(
                    f"gpu{i}", f"gpu{j}",
                    bandwidth_gbps=600.0, latency_us=0.495,
                    link_type=LinkType.NVLINK,
                ))

    if debug_print:
        print(f"\n[topology] Created: {n_gpus} GPUs, dual-socket")
        print(topo.dump_ascii_topology())

    return topo
