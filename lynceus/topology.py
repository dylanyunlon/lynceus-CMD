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
        # 改写(M013): NCCL alpha-beta transport model替代饱和对数
        # NCCL ncclTopoCompute uses alpha(latency) + n/beta(bandwidth) per hop;
        # 我们用 1MB 参考包大小, 带宽利用率 eta 按链路类型分级:
        #   NVLink eta=0.92, PCIe eta=0.78, QPI eta=0.65, NET eta=0.55
        _eta_table = {
            LinkType.NVLINK:  0.92,
            LinkType.PCIE:    0.78,
            LinkType.PCIE_P2P: 0.82,
            LinkType.QPI_UPI: 0.65,
            LinkType.SHM:     0.88,
            LinkType.NETWORK: 0.55,
        }
        eta = _eta_table.get(self.link_type, 0.70)
        ref_mb = 1.0
        effective_bw = max(0.001, self.bandwidth_gbps * eta)
        transfer_us = (ref_mb * 1000.0) / effective_bw
        # alpha-beta: hop_cost = alpha + n/beta, 这里alpha=latency, n/beta=transfer
        self._hop_cost = self.latency_us + transfer_us
        self._effective_bw = effective_bw  # 保存供调试输出

    @property
    def hop_cost(self) -> float:
        return self._hop_cost

    def effective_bandwidth(self) -> float:
        """调试用: 返回考虑利用率后的有效带宽 (GB/s)."""
        return self._effective_bw


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

        # M013改写: NUMA亲和性折扣——同NUMA域内转移代价×0.70
        # 参考NCCL ncclTopoGetIntraNode对同socket通信的优先路径选择
        src_numa = self._nodes[src].numa_node if src in self._nodes else -1
        dst_numa = self._nodes[dst].numa_node if dst in self._nodes else -1

        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            if u == dst:
                break
            u_numa = self._nodes[u].numa_node if u in self._nodes else -1
            for edge in self._adj.get(u, []):
                v = edge.dst
                if v in visited:
                    continue
                v_numa = self._nodes[v].numa_node if v in self._nodes else -1
                cost = edge.hop_cost
                # NUMA亲和折扣: 两端在同一NUMA域且非默认(-1)
                if u_numa >= 0 and u_numa == v_numa:
                    cost *= 0.70
                new_dist = d + cost
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

        result = (dist[dst], path)
        self._path_cache[cache_key] = result
        if self._debug and len(path) > 2:
            logger.debug(f"[topo:dijkstra] {src}->{dst}: cost={dist[dst]:.2f}, "
                        f"hops={len(path)-1}, numa_affinity="
                        f"{'yes' if src_numa==dst_numa and src_numa>=0 else 'no'}")
        return result

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

    # =======================================================================
    # M013 新增: NCCL Ring Allreduce 拓扑推导
    # =======================================================================

    def build_ring_topology(self, gpu_ids: List[str]) -> List[Tuple[str, str]]:
        """构造最优ring顺序, 参考NCCL ncclTopoCompute ring search.

        贪心策略: 从任意GPU出发, 每次选hop_cost最小的未访问邻居,
        类似TSP最近邻启发(NCCL用的也是贪心+局部搜索).
        返回 [(gpu0,gpu1), (gpu1,gpu2), ..., (gpuN-1,gpu0)] 的环.
        """
        if len(gpu_ids) <= 1:
            return []

        # 贪心构造: 从第一个GPU出发, 始终选最近的
        ring_order: List[str] = [gpu_ids[0]]
        remaining = set(gpu_ids[1:])
        while remaining:
            cur = ring_order[-1]
            best_next, best_cost = None, float('inf')
            for cand in remaining:
                c, _ = self._dijkstra(cur, cand)
                if c < best_cost:
                    best_cost = c
                    best_next = cand
            if best_next is None:
                break
            ring_order.append(best_next)
            remaining.discard(best_next)

        # 闭合环
        ring_edges = []
        for i in range(len(ring_order)):
            s = ring_order[i]
            d = ring_order[(i + 1) % len(ring_order)]
            ring_edges.append((s, d))

        if self._debug:
            ring_str = "->".join(ring_order + [ring_order[0]])
            total_ring_cost = sum(
                self._dijkstra(s, d)[0] for s, d in ring_edges
            )
            print(f"  [TOPO:ring] Ring order: {ring_str}")
            print(f"  [TOPO:ring] Total ring cost: {total_ring_cost:.2f}")

        return ring_edges

    def estimate_allreduce_us(self, gpu_ids: List[str],
                               data_bytes: int) -> float:
        """Ring allreduce 时间估计: 2·(n-1)/n · size/bw.

        NCCL ring allreduce的理论下界: 对n个GPU, 数据量S, 环上最小带宽bw:
          T = 2·(n-1)/n · S/bw
        2倍因为 reduce-scatter + allgather 各一趟.
        """
        n = len(gpu_ids)
        if n <= 1:
            return 0.0

        ring_edges = self.build_ring_topology(gpu_ids)
        if not ring_edges:
            return float('inf')

        # 找环上的瓶颈带宽 (最慢的那条边)
        min_bw_gbps = float('inf')
        total_latency_us = 0.0
        for s, d in ring_edges:
            edge = self._find_edge(s, d)
            if edge:
                min_bw_gbps = min(min_bw_gbps, edge.effective_bandwidth())
                total_latency_us += edge.latency_us
            else:
                # fallback: 用dijkstra路径的transfer
                _, path = self._dijkstra(s, d)
                for i in range(len(path) - 1):
                    e2 = self._find_edge(path[i], path[i + 1])
                    if e2:
                        min_bw_gbps = min(min_bw_gbps, e2.effective_bandwidth())
                        total_latency_us += e2.latency_us

        if min_bw_gbps <= 0 or min_bw_gbps == float('inf'):
            return float('inf')

        data_mb = data_bytes / (1024 * 1024)
        # ring allreduce公式: 2·(n-1)/n · data / bw + latency
        coeff = 2.0 * (n - 1) / n
        transfer_us = coeff * (data_mb * 1000.0) / min_bw_gbps
        # latency: 每个step需要2(n-1)次hop
        ring_latency_us = total_latency_us * 2.0 * (n - 1) / n

        result = transfer_us + ring_latency_us

        if self._debug:
            print(f"  [TOPO:allreduce] n={n}, data={data_mb:.2f}MB, "
                  f"bottleneck_bw={min_bw_gbps:.1f}GB/s, "
                  f"transfer={transfer_us:.1f}us, lat={ring_latency_us:.1f}us, "
                  f"total={result:.1f}us")
        return result

    def _dbg_snapshot(self) -> Dict[str, Any]:
        """调试快照: 返回完整拓扑状态字典, 用于断点检查."""
        return {
            'n_nodes': len(self._nodes),
            'n_edges': sum(len(v) for v in self._adj.values()),
            'cached_paths': len(self._path_cache),
            'reachability_computed': self._reachable is not None,
            'nodes': {
                nid: {
                    'kind': n.kind,
                    'mem_gb': n.memory_gb,
                    'compute_tflops': n.compute_tflops,
                    'numa': n.numa_node,
                }
                for nid, n in self._nodes.items()
            },
            'edges': [
                {
                    'src': e.src, 'dst': e.dst,
                    'bw': e.bandwidth_gbps,
                    'lat': e.latency_us,
                    'link': e.link_type.name,
                    'hop_cost': round(e.hop_cost, 3),
                    'eff_bw': round(e.effective_bandwidth(), 3),
                }
                for edges in self._adj.values()
                for e in edges
            ],
        }


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
                # M013改写: 同NUMA内NVLink带宽更高 (全速600GB/s vs 跨域540GB/s)
                # 参考DGX A100/H100真实拓扑: 同socket GPU对走直连NVLink,
                # 跨socket GPU对经NVSwitch中转, 有效带宽略低
                same_numa = (i < n_gpus // 2) == (j < n_gpus // 2)
                bw = 600.0 if same_numa else 540.0
                lat = 0.4 if same_numa else 0.7
                topo.add_edge(TopoEdge(
                    f"gpu{i}", f"gpu{j}",
                    bandwidth_gbps=bw, latency_us=lat,
                    link_type=LinkType.NVLINK,
                ))
    if debug_print:
        print(f"\n[topology] Created default topology: {n_gpus} GPUs, dual-socket")
        print(topo.dump_state())
        # 自动构造并打印ring拓扑
        gpu_ids = [f"gpu{i}" for i in range(n_gpus)]
        topo.build_ring_topology(gpu_ids)
        # 估算1GB allreduce时间
        topo.estimate_allreduce_us(gpu_ids, data_bytes=1024*1024*1024)
    return topo
