"""
lynceus_port/topology.py — 硬件拓扑图与多跳最短路径。

移植自 lynceus/topology.py，修改约20%:
  - Dijkstra 改为带路径重建的双向搜索（bidirectional Dijkstra）
  - 路径缓存增加 TTL（每 N 次查询刷新）
  - 新增 check_connectivity()：检测图中孤立节点
  - dump_state: 增加连通分量信息
"""

from __future__ import annotations

import heapq
import time
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum, auto

from .schema import _dbg


class LinkType(Enum):
    NVLINK   = auto()
    PCIE     = auto()
    QPI_UPI  = auto()
    NETWORK  = auto()
    SHM      = auto()
    PCIE_P2P = auto()


@dataclass
class TopoEdge:
    """拓扑图中的有向边"""
    src: str
    dst: str
    bandwidth_gbps: float
    latency_us: float
    link_type: LinkType
    _hop_cost: float = field(init=False)

    def __post_init__(self):
        ref_mb = 1.0
        transfer_us = (ref_mb * 1000) / max(0.001, self.bandwidth_gbps)
        self._hop_cost = self.latency_us + transfer_us

    @property
    def hop_cost(self) -> float:
        return self._hop_cost


@dataclass
class TopoNode:
    """拓扑图中的硬件节点"""
    node_id: str
    kind: str
    memory_gb: float = 0.0
    compute_tflops: float = 0.0
    numa_node: int = -1
    metadata: Dict[str, Any] = field(default_factory=dict)


class HardwareTopologyGraph:
    """多跳硬件拓扑——带双向 Dijkstra 和缓存 TTL"""

    CACHE_TTL_QUERIES = 1000  # 每 1000 次查询刷新路径缓存

    def __init__(self, debug_print: bool = True):
        self._nodes: Dict[str, TopoNode] = {}
        self._adj: Dict[str, List[TopoEdge]] = {}
        self._rev_adj: Dict[str, List[TopoEdge]] = {}  # 反向邻接表
        self._path_cache: Dict[Tuple[str, str], Tuple[float, List[str]]] = {}
        self._cache_age = 0
        self._debug = debug_print

    def add_node(self, node: TopoNode) -> None:
        self._nodes[node.node_id] = node
        if node.node_id not in self._adj:
            self._adj[node.node_id] = []
        if node.node_id not in self._rev_adj:
            self._rev_adj[node.node_id] = []
        self._path_cache.clear()
        if self._debug:
            _dbg("Topo", f"add_node: {node.node_id} ({node.kind}, "
                 f"mem={node.memory_gb}GB, compute={node.compute_tflops}TFLOPS)")

    def add_edge(self, edge: TopoEdge) -> None:
        self._adj.setdefault(edge.src, []).append(edge)
        # ── 反向邻接 ──
        rev_edge = TopoEdge(edge.dst, edge.src, edge.bandwidth_gbps,
                            edge.latency_us, edge.link_type)
        self._rev_adj.setdefault(edge.dst, []).append(rev_edge)
        self._path_cache.clear()
        if self._debug:
            _dbg("Topo", f"add_edge: {edge.src}->{edge.dst} "
                 f"({edge.link_type.name}, {edge.bandwidth_gbps}GB/s, "
                 f"hop_cost={edge.hop_cost:.2f})")

    def add_bidir_edge(self, src: str, dst: str,
                       bandwidth_gbps: float, latency_us: float,
                       link_type: LinkType) -> None:
        self.add_edge(TopoEdge(src, dst, bandwidth_gbps, latency_us, link_type))
        self.add_edge(TopoEdge(dst, src, bandwidth_gbps, latency_us, link_type))

    def get_transfer_cost(self, src: str, dst: str, data_bytes: int) -> float:
        if src == dst:
            return 0.0

        cost_per_mb, path = self._bidir_dijkstra(src, dst)
        if cost_per_mb == float('inf'):
            if self._debug:
                _dbg("Topo", f"UNREACHABLE: {src}->{dst}")
            return float('inf')

        data_mb = data_bytes / (1024 * 1024)
        total_us = 0.0
        for i in range(len(path) - 1):
            edge = self._find_edge(path[i], path[i + 1])
            if edge:
                hop_transfer = (data_mb * 1000) / max(0.001, edge.bandwidth_gbps)
                total_us += edge.latency_us + hop_transfer

        if self._debug:
            path_str = "->".join(path)
            _dbg("Topo", f"xfer {src}->{dst}: {data_bytes:,}B "
                 f"via [{path_str}] = {total_us:.1f}us")
        return total_us

    def _bidir_dijkstra(self, src: str, dst: str
                        ) -> Tuple[float, List[str]]:
        """双向 Dijkstra：从 src 正向 + 从 dst 反向同时搜索，相遇即停。

        小图上性能差别不大，但保留了正确的最短路径语义。
        """
        self._cache_age += 1
        if self._cache_age > self.CACHE_TTL_QUERIES:
            self._path_cache.clear()
            self._cache_age = 0

        cache_key = (src, dst)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        if src not in self._adj or dst not in self._adj:
            return float('inf'), []

        # 正向 Dijkstra 状态
        dist_fwd: Dict[str, float] = {src: 0.0}
        prev_fwd: Dict[str, Optional[str]] = {src: None}
        visited_fwd: Set[str] = set()
        heap_fwd = [(0.0, src)]

        # 反向 Dijkstra 状态
        dist_rev: Dict[str, float] = {dst: 0.0}
        prev_rev: Dict[str, Optional[str]] = {dst: None}
        visited_rev: Set[str] = set()
        heap_rev = [(0.0, dst)]

        best_cost = float('inf')
        meeting_node: Optional[str] = None

        while heap_fwd or heap_rev:
            # 正向一步
            if heap_fwd:
                d, u = heapq.heappop(heap_fwd)
                if u not in visited_fwd:
                    visited_fwd.add(u)
                    if u in visited_rev:
                        total = dist_fwd.get(u, float('inf')) + dist_rev.get(u, float('inf'))
                        if total < best_cost:
                            best_cost = total
                            meeting_node = u
                    for edge in self._adj.get(u, []):
                        v = edge.dst
                        nd = d + edge.hop_cost
                        if nd < dist_fwd.get(v, float('inf')):
                            dist_fwd[v] = nd
                            prev_fwd[v] = u
                            heapq.heappush(heap_fwd, (nd, v))

            # 反向一步
            if heap_rev:
                d, u = heapq.heappop(heap_rev)
                if u not in visited_rev:
                    visited_rev.add(u)
                    if u in visited_fwd:
                        total = dist_fwd.get(u, float('inf')) + dist_rev.get(u, float('inf'))
                        if total < best_cost:
                            best_cost = total
                            meeting_node = u
                    for edge in self._rev_adj.get(u, []):
                        v = edge.dst  # 在反向图中 dst 是 "上游"
                        nd = d + edge.hop_cost
                        if nd < dist_rev.get(v, float('inf')):
                            dist_rev[v] = nd
                            prev_rev[v] = u
                            heapq.heappush(heap_rev, (nd, v))

            # 终止条件
            min_fwd = heap_fwd[0][0] if heap_fwd else float('inf')
            min_rev = heap_rev[0][0] if heap_rev else float('inf')
            if min_fwd + min_rev >= best_cost:
                break

        if meeting_node is None:
            self._path_cache[cache_key] = (float('inf'), [])
            return float('inf'), []

        # 重建路径
        path_fwd = []
        node: Optional[str] = meeting_node
        while node is not None:
            path_fwd.append(node)
            node = prev_fwd.get(node)
        path_fwd.reverse()

        path_rev = []
        node = prev_rev.get(meeting_node)
        while node is not None:
            path_rev.append(node)
            node = prev_rev.get(node)

        full_path = path_fwd + path_rev
        self._path_cache[cache_key] = (best_cost, full_path)
        return best_cost, full_path

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

    # ── 新增：连通性检查 ──

    def check_connectivity(self) -> List[Set[str]]:
        """检测连通分量，返回分量列表。多于1个分量说明存在孤立节点"""
        visited: Set[str] = set()
        components: List[Set[str]] = []
        for node_id in self._nodes:
            if node_id not in visited:
                comp: Set[str] = set()
                stack = [node_id]
                while stack:
                    u = stack.pop()
                    if u in visited:
                        continue
                    visited.add(u)
                    comp.add(u)
                    for e in self._adj.get(u, []):
                        if e.dst not in visited:
                            stack.append(e.dst)
                components.append(comp)
        if self._debug and len(components) > 1:
            _dbg("Topo", f"WARNING: {len(components)} connected components!")
            for i, c in enumerate(components):
                _dbg("Topo", f"  component {i}: {c}")
        return components

    # ── B-Tree 通信拓扑 ──

    def build_btree_comm_topology(self, gpu_ids: List[str]
                                   ) -> Dict[str, List[str]]:
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
            _dbg("Topo", f"BTree comm for {n} GPUs: root={gpu_ids[0]}")
        return tree

    # ── 调试 ──

    def dump_state(self) -> str:
        n_edges = sum(len(v) for v in self._adj.values())
        components = self.check_connectivity()
        lines = [
            "=== HardwareTopologyGraph State ===",
            f"  nodes={len(self._nodes)}, edges={n_edges}, "
            f"components={len(components)}, cache={len(self._path_cache)}",
            "  -- Nodes --",
        ]
        for nid, node in sorted(self._nodes.items()):
            lines.append(f"    {nid}: {node.kind}, mem={node.memory_gb}GB, "
                         f"compute={node.compute_tflops}T, numa={node.numa_node}")
        lines.append("  -- Edges --")
        for src, edges in sorted(self._adj.items()):
            for e in edges:
                lines.append(f"    {e.src}->{e.dst}: {e.link_type.name} "
                             f"bw={e.bandwidth_gbps} lat={e.latency_us} "
                             f"hop={e.hop_cost:.2f}")
        s = "\n".join(lines)
        _dbg("Topo", s)
        return s


# ── 默认拓扑工厂 ──

def create_default_topology(
    n_gpus: int = 4,
    cpu_memory_gb: float = 256.0,
    gpu_memory_gb: float = 80.0,
    debug_print: bool = True,
) -> HardwareTopologyGraph:
    """创建典型双路服务器拓扑"""
    topo = HardwareTopologyGraph(debug_print=debug_print)

    topo.add_node(TopoNode("cpu0", "cpu", memory_gb=cpu_memory_gb / 2, numa_node=0))
    topo.add_node(TopoNode("cpu1", "cpu", memory_gb=cpu_memory_gb / 2, numa_node=1))

    for i in range(n_gpus):
        topo.add_node(TopoNode(
            f"gpu{i}", "gpu",
            memory_gb=gpu_memory_gb,
            compute_tflops=312.0,
            numa_node=0 if i < n_gpus // 2 else 1,
        ))

    topo.add_bidir_edge("cpu0", "cpu1", bandwidth_gbps=50.0,
                        latency_us=0.3, link_type=LinkType.QPI_UPI)

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
                    bandwidth_gbps=600.0, latency_us=0.5,
                    link_type=LinkType.NVLINK,
                ))

    if debug_print:
        _dbg("Factory", f"topology: {n_gpus} GPUs, dual-socket")
        topo.dump_state()
        topo.check_connectivity()

    return topo
