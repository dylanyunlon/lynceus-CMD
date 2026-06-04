"""
lynceus/topology.py — Hardware topology graph with multi-hop shortest path.

Architecture references:
  - NCCL ncclTopoCompute, ncclGetBtree, tabular TableGroup,
    Megatron DistributedDataParallelConfig
"""

from __future__ import annotations

import heapq
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum, auto

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
    _hop_cost: float = field(init=False)

    def __post_init__(self):
        # 改动: 拥塞修正用饱和对数, 高带宽链路拥塞概率更低
        ref_mb = 1.0
        transfer_us = (ref_mb * 1000) / max(0.001, self.bandwidth_gbps)
        # 饱和对数拥塞: 低带宽链路惩罚更重
        cong = 1.0 + 0.12 / (1.0 + math.exp(self.bandwidth_gbps / 200.0 - 2.0))
        self._hop_cost = (self.latency_us + transfer_us) * cong

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
    """Multi-hop hardware topology with Dijkstra + BFS reachability pre-check."""

    def __init__(self, debug_print: bool = True):
        self._nodes: Dict[str, TopoNode] = {}
        self._adj: Dict[str, List[TopoEdge]] = {}
        self._path_cache: Dict[Tuple[str, str], Tuple[float, List[str]]] = {}
        self._reachable: Optional[Dict[str, Set[str]]] = None  # BFS预计算
        self._debug = debug_print

    def add_node(self, node: TopoNode) -> None:
        self._nodes[node.node_id] = node
        if node.node_id not in self._adj:
            self._adj[node.node_id] = []
        self._invalidate_cache()
        if self._debug:
            logger.debug(f"[topo] add_node: {node.node_id} ({node.kind}, "
                        f"mem={node.memory_gb}GB, compute={node.compute_tflops}TFLOPS)")

    def add_edge(self, edge: TopoEdge) -> None:
        if edge.src not in self._adj:
            self._adj[edge.src] = []
        self._adj[edge.src].append(edge)
        self._invalidate_cache()
        if self._debug:
            logger.debug(f"[topo] add_edge: {edge.src}->{edge.dst} "
                        f"({edge.link_type.name}, {edge.bandwidth_gbps}GB/s, "
                        f"{edge.latency_us}us, hop_cost={edge.hop_cost:.2f})")

    def add_bidir_edge(self, src: str, dst: str,
                       bandwidth_gbps: float, latency_us: float,
                       link_type: LinkType) -> None:
        self.add_edge(TopoEdge(src, dst, bandwidth_gbps, latency_us, link_type))
        self.add_edge(TopoEdge(dst, src, bandwidth_gbps, latency_us, link_type))

    def _invalidate_cache(self):
        self._path_cache.clear()
        self._reachable = None

    def _build_reachability(self):
        """BFS预计算所有节点的可达集合, 避免Dijkstra在不可达时白跑。"""
        self._reachable = {}
        for start in self._adj:
            visited: Set[str] = set()
            queue = [start]
            visited.add(start)
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for edge in self._adj.get(u, []):
                    if edge.dst not in visited:
                        visited.add(edge.dst)
                        queue.append(edge.dst)
            self._reachable[start] = visited

    def get_transfer_cost(self, src: str, dst: str, data_bytes: int) -> float:
        """INV-4 compliant: multi-hop Dijkstra, BFS预检不可达。"""
        from ._debug import dbg, checkpoint
        dbg('Topo.transfer', src=src, dst=dst, data_bytes=data_bytes)
        if src == dst:
            return 0.0

        # BFS预检
        if self._reachable is None:
            self._build_reachability()
        if src not in self._reachable or dst not in self._reachable.get(src, set()):
            checkpoint("topo_unreachable", src=src, dst=dst)
            return float('inf')

        cost_per_mb, path = self._dijkstra(src, dst)
        if cost_per_mb == float('inf'):
            return float('inf')

        total_us = 0.0
        for i in range(len(path) - 1):
            edge = self._find_edge(path[i], path[i + 1])
            if edge:
                hop_latency = edge.latency_us
                data_mb = data_bytes / (1024 * 1024)
                hop_transfer = (data_mb * 1000) / max(0.001, edge.bandwidth_gbps)
                total_us += hop_latency + hop_transfer

        checkpoint("topo_transfer_done", src=src, dst=dst,
                   path="->".join(path), total_us=total_us)
        return total_us

    def _dijkstra(self, src: str, dst: str) -> Tuple[float, List[str]]:
        cache_key = (src, dst)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        if src not in self._adj or dst not in self._adj:
            return float('inf'), []

        dist: Dict[str, float] = {src: 0.0}
        prev: Dict[str, Optional[str]] = {src: None}
        visited: Set[str] = set()
        heap = [(0.0, src)]

        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            if u == dst:
                break
            for edge in self._adj.get(u, []):
                v = edge.dst
                if v in visited:
                    continue
                new_dist = d + edge.hop_cost
                if new_dist < dist.get(v, float('inf')):
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(heap, (new_dist, v))

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
        best = None
        for e in self._adj.get(src, []):
            if e.dst == dst:
                if best is None or e.hop_cost < best.hop_cost:
                    best = e
        return best

    def get_all_nodes(self, kind: Optional[str] = None) -> List[TopoNode]:
        if kind is None:
            return list(self._nodes.values())
        return [n for n in self._nodes.values() if n.kind == kind]

    def get_node(self, node_id: str) -> Optional[TopoNode]:
        return self._nodes.get(node_id)

    def build_btree_comm_topology(self, gpu_ids: List[str]) -> Dict[str, List[str]]:
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
        if self._debug:
            print(f"  [TOPO] BTree comm topology for {n} GPUs:")
            for parent, children in tree.items():
                child_str = ", ".join(children) if children else "(leaf)"
                print(f"    {parent} -> {child_str}")
        return tree

    def dump_state(self) -> str:
        lines = [
            "== HardwareTopologyGraph State Dump ==",
            f"  n_nodes         = {len(self._nodes)}",
            f"  n_edges         = {sum(len(v) for v in self._adj.values())}",
            f"  cached_paths    = {len(self._path_cache)}",
            f"  reachability    = {'computed' if self._reachable else 'pending'}",
            "",
            "  -- Nodes --",
        ]
        for nid, node in sorted(self._nodes.items()):
            lines.append(f"    {nid}: kind={node.kind}, mem={node.memory_gb}GB, "
                        f"compute={node.compute_tflops}TFLOPS, numa={node.numa_node}")
        lines.append("")
        lines.append("  -- Edges --")
        for src, edges in sorted(self._adj.items()):
            for e in edges:
                lines.append(f"    {e.src}->{e.dst}: {e.link_type.name} "
                           f"bw={e.bandwidth_gbps}GB/s lat={e.latency_us}us "
                           f"hop_cost={e.hop_cost:.2f}")
        lines.append("")
        lines.append("  -- Reachability Matrix --")
        node_ids = sorted(self._nodes.keys())
        header = "         " + "  ".join(f"{n:>6}" for n in node_ids)
        lines.append(header)
        for src in node_ids:
            costs = []
            for dst in node_ids:
                if src == dst:
                    costs.append("     0")
                else:
                    c, _ = self._dijkstra(src, dst)
                    costs.append(f"{c:6.1f}" if c < float('inf') else "   inf")
            lines.append(f"  {src:>5} " + "  ".join(costs))
        return "\n".join(lines)

    def print_all_paths(self) -> None:
        node_ids = sorted(self._nodes.keys())
        print("\n[TOPO] All-pairs shortest paths:")
        for src in node_ids:
            for dst in node_ids:
                if src == dst:
                    continue
                cost, path = self._dijkstra(src, dst)
                path_str = "->".join(path) if path else "UNREACHABLE"
                cost_str = f"{cost:.2f}" if cost < float('inf') else "inf"
                print(f"  {src}->{dst}: cost={cost_str}, path=[{path_str}]")


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
            compute_tflops=290.0,
            numa_node=0 if i < n_gpus // 2 else 1,
        ))
    topo.add_bidir_edge("cpu0", "cpu1", bandwidth_gbps=48.0, latency_us=0.3,
                        link_type=LinkType.QPI_UPI)
    for i in range(n_gpus):
        local_cpu = "cpu0" if i < n_gpus // 2 else "cpu1"
        topo.add_bidir_edge(local_cpu, f"gpu{i}",
                           bandwidth_gbps=30.5, latency_us=1.0,
                           link_type=LinkType.PCIE)
    for i in range(n_gpus):
        for j in range(n_gpus):
            if i != j:
                topo.add_edge(TopoEdge(
                    f"gpu{i}", f"gpu{j}",
                    bandwidth_gbps=580.0, latency_us=0.5,
                    link_type=LinkType.NVLINK,
                ))
    if debug_print:
        print(f"\n[topology] Created default topology: {n_gpus} GPUs, dual-socket")
        print(topo.dump_state())
    return topo
