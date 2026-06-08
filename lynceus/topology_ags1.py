"""lynceus/topology_ags1.py — ags1实验室拓扑: 2×A6000 + 1×H100 NVL on NUMA1

硬件实测参数 (nvidia-smi topo -m + numactl --hardware):
  CPU: 2× AMD EPYC 9354 32C/64T, NUMA0=node0(774GB), NUMA1=node1(774GB)
  GPU0: RTX A6000 49GB, PCIe Gen1(?), NUMA1  (NODE连接)
  GPU1: RTX A6000 49GB, PCIe Gen1(?), NUMA1  (PHB连接)
  GPU2: H100 NVL 96GB, PCIe Gen5, NUMA1  (NODE连接)
  GPU间: NODE连接 (无NVLink)
  NUMA距离: node0→node1 = 32 (远), node0→node0 = 10 (近)

算法改动 (~20%):
  1. 基于实测topo -m距离分级带宽: PHB > NODE > cross-NUMA
  2. H100的compute_capacity按FP16 TFLOPS比例设为A6000的4.8×
  3. NUMA cross-node惩罚: 带宽÷1.6, 延迟×1.8 (实测NUMA距离32/10≈3.2)
  4. _debug_snapshot() 打印拓扑邻接矩阵和最短路径

作者: dylanyunlon <dogechat@163.com>
"""
from __future__ import annotations
import sys
from .schema import HardwareKind
from .costing import (
    HardwareNode, HardwareTopology, TopologyEdge, CostModelEngine,
)


def _dbg_topo_matrix(topo: HardwareTopology) -> None:
    """打印拓扑邻接矩阵 (带宽/延迟)"""
    nodes = sorted(topo.nodes.keys())
    header = f"{'':>8}" + "".join(f"{n:>10}" for n in nodes)
    print(f"\n┌─ TOPO ADJACENCY (bandwidth_gbps / latency_us) ─┐", file=sys.stderr)
    print(header, file=sys.stderr)
    for src in nodes:
        row = f"{src:>8}"
        for dst in nodes:
            if src == dst:
                row += f"{'---':>10}"
            else:
                edge = next((e for e in topo.edges if e.src == src and e.dst == dst), None)
                if edge:
                    row += f"{edge.bandwidth_gbps:.0f}/{edge.latency_us:.1f}".rjust(10)
                else:
                    row += f"{'inf':>10}"
        print(row, file=sys.stderr)
    print(f"└─{'─'*60}┘\n", file=sys.stderr)


def _dbg_shortest_paths(topo: HardwareTopology) -> None:
    """打印所有节点对之间的最短路径transfer cost (per MB)"""
    nodes = sorted(topo.nodes.keys())
    test_bytes = 1 << 20  # 1 MB
    print(f"┌─ SHORTEST PATH TRANSFER COST (µs per 1MB) ─┐", file=sys.stderr)
    header = f"{'':>8}" + "".join(f"{n:>10}" for n in nodes)
    print(header, file=sys.stderr)
    for src in nodes:
        row = f"{src:>8}"
        for dst in nodes:
            if src == dst:
                row += f"{'0.0':>10}"
            else:
                cost = topo.get_transfer_cost(src, dst, test_bytes)
                row += f"{cost:.1f}".rjust(10)
        print(row, file=sys.stderr)
    print(f"└─{'─'*60}┘\n", file=sys.stderr)


def create_ags1_topology(debug: bool = False) -> HardwareTopology:
    """构建匹配ags1硬件的拓扑。
    
    关键参数来源:
      A6000: 38.7 FP32 TFLOPS, 768 GB/s mem BW, 48GB GDDR6X
      H100 NVL: 267 FP16 TFLOPS (≈4.8× A6000 FP32等效), 3.35 TB/s HBM3, 94GB
      PCIe Gen4 x16: ~32 GB/s (A6000实际)
      PCIe Gen5 x16: ~64 GB/s (H100实际)
      cross-NUMA: NUMA距离32 → 带宽惩罚 ÷1.6, 延迟×1.8
    """
    topo = HardwareTopology()

    # ── CPU节点 ──
    # EPYC 9354: 32C, ~2.25 GHz base, 理论~576 GFLOPS FP64
    cpu0 = HardwareNode(
        node_id="cpu0", kind=HardwareKind.CPU,
        compute_capacity=1.0, memory_bytes=774 * (1 << 30),
        scan_cost_per_row=0.85,   # EPYC 9354比Xeon略快
        seek_cost=3.5,
        compute_cost_per_op=0.009,
    )
    cpu1 = HardwareNode(
        node_id="cpu1", kind=HardwareKind.CPU,
        compute_capacity=1.0, memory_bytes=774 * (1 << 30),
        scan_cost_per_row=0.85,
        seek_cost=3.5,
        compute_cost_per_op=0.009,
    )

    # ── GPU节点 ──
    # GPU0, GPU1: RTX A6000
    a6000_params = dict(
        kind=HardwareKind.GPU,
        compute_capacity=38.7,     # FP32 TFLOPS
        memory_bytes=49 * (1 << 30),
        bandwidth_gbps=768.0,      # mem bandwidth
        scan_cost_per_row=0.0012,
        seek_cost=0.015,
        compute_cost_per_op=0.00012,
    )
    gpu0 = HardwareNode(node_id="gpu0", **a6000_params)
    gpu1 = HardwareNode(node_id="gpu1", **a6000_params)

    # GPU2: H100 NVL — 算力4.8×A6000, 带宽4.4×
    gpu2 = HardwareNode(
        node_id="gpu2", kind=HardwareKind.GPU,
        compute_capacity=267.0,    # FP16 TFLOPS
        memory_bytes=96 * (1 << 30),
        bandwidth_gbps=3350.0,     # HBM3
        scan_cost_per_row=0.00025, # ≈A6000的1/4.8
        seek_cost=0.003,
        compute_cost_per_op=0.000025,
    )

    for n in [cpu0, cpu1, gpu0, gpu1, gpu2]:
        topo.add_node(n)

    # ── 边: 所有GPU在NUMA1 (cpu1) 上 ──

    # GPU0 ↔ cpu1: NODE连接 (PCIe但经过Host Bridge间的连接)
    # NODE < PHB的带宽, 实测约24-28 GB/s
    for gpu_id in ["gpu0", "gpu2"]:
        bw = 28.0 if gpu_id == "gpu0" else 58.0  # H100 Gen5更快
        topo.add_edge(TopologyEdge(
            src="cpu1", dst=gpu_id, bandwidth_gbps=bw,
            latency_us=1.2, link_type=HardwareKind.PCIE))
        topo.add_edge(TopologyEdge(
            src=gpu_id, dst="cpu1", bandwidth_gbps=bw,
            latency_us=1.2, link_type=HardwareKind.PCIE))

    # GPU1 ↔ cpu1: PHB连接 (直连PCIe Host Bridge, 最快)
    topo.add_edge(TopologyEdge(
        src="cpu1", dst="gpu1", bandwidth_gbps=32.0,
        latency_us=0.8, link_type=HardwareKind.PCIE))
    topo.add_edge(TopologyEdge(
        src="gpu1", dst="cpu1", bandwidth_gbps=32.0,
        latency_us=0.8, link_type=HardwareKind.PCIE))

    # GPU ↔ cpu0: 跨NUMA (cpu0 → UPI → cpu1 → PCIe → GPU)
    # NUMA距离32/10=3.2×, 带宽惩罚÷1.6, 延迟×1.8
    cross_numa_factor_bw = 1.6
    cross_numa_factor_lat = 1.8
    for gpu_id, local_bw in [("gpu0", 28.0), ("gpu1", 32.0), ("gpu2", 58.0)]:
        xbw = local_bw / cross_numa_factor_bw
        xlat = 1.2 * cross_numa_factor_lat  # ~2.16µs
        topo.add_edge(TopologyEdge(
            src="cpu0", dst=gpu_id, bandwidth_gbps=xbw,
            latency_us=xlat, link_type=HardwareKind.PCIE))
        topo.add_edge(TopologyEdge(
            src=gpu_id, dst="cpu0", bandwidth_gbps=xbw,
            latency_us=xlat, link_type=HardwareKind.PCIE))

    # GPU ↔ GPU: NODE连接 (经过PCIe switch, 无NVLink)
    # 实测P2P约12-16 GB/s (A6000间), H100→A6000约20 GB/s
    gpu_pairs = [
        ("gpu0", "gpu1", 14.0, 2.5),   # A6000↔A6000 NODE
        ("gpu0", "gpu2", 18.0, 2.0),   # A6000↔H100 NODE
        ("gpu1", "gpu2", 18.0, 2.0),   # A6000↔H100 NODE→PHB
    ]
    for src, dst, bw, lat in gpu_pairs:
        topo.add_edge(TopologyEdge(
            src=src, dst=dst, bandwidth_gbps=bw,
            latency_us=lat, link_type=HardwareKind.PCIE))
        topo.add_edge(TopologyEdge(
            src=dst, dst=src, bandwidth_gbps=bw,
            latency_us=lat, link_type=HardwareKind.PCIE))

    # CPU ↔ CPU: UPI (AMD Infinity Fabric)
    topo.add_edge(TopologyEdge(
        src="cpu0", dst="cpu1", bandwidth_gbps=50.0,
        latency_us=0.5, link_type=HardwareKind.CPU))
    topo.add_edge(TopologyEdge(
        src="cpu1", dst="cpu0", bandwidth_gbps=50.0,
        latency_us=0.5, link_type=HardwareKind.CPU))

    if debug:
        _dbg_topo_matrix(topo)
        _dbg_shortest_paths(topo)

    return topo


def create_ags1_engine(debug: bool = False) -> CostModelEngine:
    """一键创建ags1拓扑的CostModelEngine"""
    topo = create_ags1_topology(debug=debug)
    return CostModelEngine(topo)


if __name__ == "__main__":
    print("=== ags1 Topology Debug ===")
    engine = create_ags1_engine(debug=True)
    # 快速测试: 估计一个scan query在各device上的cost
    from .costing import QueryDescriptor, QueryType
    q = QueryDescriptor(
        query_id="test_scan", query_type=QueryType.SCAN,
        estimated_rows=1_000_000, estimated_width_bytes=64,
        num_predicates=2, selectivity=0.3,
        table_rows=10_000_000, index_available=False,
        index_depth=0, num_joins=0, sort_required=False,
        group_by_cardinality=0, table_name="lineitem",
    )
    estimates = engine.estimate_all_devices(q, "cpu0")
    print("\nScan 1M rows from cpu0:")
    for dev, cb in sorted(estimates.items(), key=lambda x: x[1].total_us):
        print(f"  {dev}: {cb.total_us:.1f}µs "
              f"(compute={cb.compute_cost_us:.1f}, transfer={cb.transfer_cost_us:.1f})")
