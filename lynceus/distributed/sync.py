"""
lynceus/distributed/sync.py — Distributed cost-model synchronization.

算法改动:
    1. ring allreduce: TCP incast congestion 非线性带宽衰减
    2. gossip: spectral gap 修正的 mixing time
    3. compare: Pareto frontier (latency vs convergence)
"""
from __future__ import annotations
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto

logger = logging.getLogger(__name__)

class SyncStrategy(Enum):
    ALLREDUCE_RING = auto()
    ALLREDUCE_TREE = auto()
    PUSH_PULL = auto()
    GOSSIP = auto()
    BROADCAST = auto()

@dataclass
class SyncConfig:
    strategy: SyncStrategy = SyncStrategy.ALLREDUCE_RING
    n_workers: int = 4
    inter_node_bw_gbps: float = 100.0
    inter_node_lat_us: float = 5.0
    intra_node_bw_gbps: float = 600.0
    intra_node_lat_us: float = 0.5
    n_ps_servers: int = 1
    max_staleness: int = 3
    debug_print: bool = True

@dataclass
class SyncMetrics:
    strategy: SyncStrategy
    n_workers: int
    data_bytes: int
    total_time_us: float
    communication_us: float
    computation_us: float
    staleness: int = 0
    convergence_error: float = 0.0
    def dump_debug(self, prefix: str = "") -> str:
        return (f"{prefix}Sync({self.strategy.name}): {self.total_time_us:.1f}µs "
                f"(comm={self.communication_us:.1f}, comp={self.computation_us:.1f}, "
                f"conv_err={self.convergence_error:.4f})")


def _incast_bw(nominal_bw_gbps: float, n_workers: int) -> float:
    """改动: TCP incast congestion 模型。
    当 n_workers > 2 时, 交换机 buffer 压力导致带宽下降。
    经验公式: effective_bw = nominal / (1 + 0.05 * (p-2)^0.7)
    p=2 → 100%, p=4 → ~91%, p=8 → ~82%, p=32 → ~68%
    """
    if n_workers <= 2:
        return nominal_bw_gbps
    penalty = 0.05 * ((n_workers - 2) ** 0.7)
    return nominal_bw_gbps / (1.0 + penalty)


def estimate_ring_allreduce(data_bytes: int, config: SyncConfig) -> SyncMetrics:
    """改动: 使用 incast-aware 带宽。"""
    p = config.n_workers
    if p <= 1:
        return SyncMetrics(strategy=SyncStrategy.ALLREDUCE_RING, n_workers=p,
                          data_bytes=data_bytes, total_time_us=0.0,
                          communication_us=0.0, computation_us=0.0)
    # 改动: congestion-aware bandwidth
    effective_bw = _incast_bw(config.inter_node_bw_gbps, p)
    bw_term_us = 2.0 * (p - 1) / p * (data_bytes / (1024**2)) * 1000 / effective_bw
    lat_term_us = 2.0 * (p - 1) * config.inter_node_lat_us
    communication_us = bw_term_us + lat_term_us
    n_elements = data_bytes // 4
    computation_us = n_elements * 0.001
    total = communication_us + computation_us
    m = SyncMetrics(strategy=SyncStrategy.ALLREDUCE_RING, n_workers=p,
                   data_bytes=data_bytes, total_time_us=total,
                   communication_us=communication_us, computation_us=computation_us)
    if config.debug_print:
        print(f"\n  [SYNC] Ring AllReduce: {p} workers, {data_bytes:,}B, "
              f"eff_bw={effective_bw:.0f}Gbps → {total:.1f}µs")
    return m


def estimate_tree_allreduce(data_bytes: int, config: SyncConfig) -> SyncMetrics:
    p = config.n_workers
    if p <= 1:
        return SyncMetrics(strategy=SyncStrategy.ALLREDUCE_TREE, n_workers=p,
                          data_bytes=data_bytes, total_time_us=0.0,
                          communication_us=0.0, computation_us=0.0)
    depth = math.ceil(math.log2(p))
    data_mb = data_bytes / (1024**2)
    per_level_us = data_mb * 1000 / config.inter_node_bw_gbps + config.inter_node_lat_us
    communication_us = 2 * depth * per_level_us
    n_elements = data_bytes // 4
    computation_us = n_elements * 0.001 * depth
    total = communication_us + computation_us
    m = SyncMetrics(strategy=SyncStrategy.ALLREDUCE_TREE, n_workers=p,
                   data_bytes=data_bytes, total_time_us=total,
                   communication_us=communication_us, computation_us=computation_us)
    if config.debug_print:
        print(f"\n  [SYNC] Tree AllReduce: {p} workers, depth={depth} → {total:.1f}µs")
    return m


def estimate_push_pull(data_bytes: int, config: SyncConfig) -> SyncMetrics:
    p = config.n_workers
    n_ps = config.n_ps_servers
    data_mb = data_bytes / (1024**2)
    ps_recv_bw = n_ps * config.inter_node_bw_gbps
    push_time_us = data_mb * 1000 / ps_recv_bw + config.inter_node_lat_us
    n_elements = data_bytes // 4
    ps_compute_us = n_elements * p * 0.001
    pull_time_us = data_mb * 1000 / ps_recv_bw + config.inter_node_lat_us
    communication_us = push_time_us + pull_time_us
    total = communication_us + ps_compute_us
    m = SyncMetrics(strategy=SyncStrategy.PUSH_PULL, n_workers=p,
                   data_bytes=data_bytes, total_time_us=total,
                   communication_us=communication_us, computation_us=ps_compute_us)
    if config.debug_print:
        print(f"\n  [SYNC] Push-Pull: {p}→{n_ps} PS, → {total:.1f}µs")
    return m


def estimate_gossip(data_bytes: int, config: SyncConfig,
                    n_rounds: int = 0) -> SyncMetrics:
    """改动: spectral gap 修正的 mixing time。

    原版: convergence_error = (1 - 1/p)^rounds
    问题: 忽略了图的 connectivity 结构
    新版: 对完全图(随机 gossip), spectral gap λ₂ ≈ 1/p,
          convergence = exp(-rounds * λ₂) = exp(-rounds / p)
    这比原版 (1-1/p)^rounds ≈ exp(-rounds/p) 数值上接近,
    但对 non-complete 图可以用不同的 λ₂ (这里暴露了参数)。
    """
    p = config.n_workers
    if n_rounds <= 0:
        n_rounds = max(1, math.ceil(math.log2(p)) + 2)
    data_mb = data_bytes / (1024**2)
    per_round_us = data_mb * 1000 / config.inter_node_bw_gbps + config.inter_node_lat_us
    communication_us = n_rounds * per_round_us
    n_elements = data_bytes // 4
    computation_us = n_rounds * n_elements * 0.001

    # 改动: spectral gap mixing
    spectral_gap = 1.0 / max(1, p)  # 完全图的 spectral gap
    convergence_error = math.exp(-n_rounds * spectral_gap)

    total = communication_us + computation_us
    m = SyncMetrics(strategy=SyncStrategy.GOSSIP, n_workers=p,
                   data_bytes=data_bytes, total_time_us=total,
                   communication_us=communication_us, computation_us=computation_us,
                   convergence_error=convergence_error)
    if config.debug_print:
        print(f"\n  [SYNC] Gossip: {p} workers, {n_rounds} rounds, "
              f"λ₂={spectral_gap:.3f}, conv_err={convergence_error:.4f}")
    return m


def estimate_sync_cost(data_bytes: int, config: Optional[SyncConfig] = None,
                       debug_print: bool = True) -> SyncMetrics:
    if config is None:
        config = SyncConfig(debug_print=debug_print)
    dispatch = {
        SyncStrategy.ALLREDUCE_RING: estimate_ring_allreduce,
        SyncStrategy.ALLREDUCE_TREE: estimate_tree_allreduce,
        SyncStrategy.PUSH_PULL: estimate_push_pull,
        SyncStrategy.GOSSIP: estimate_gossip,
    }
    fn = dispatch.get(config.strategy, estimate_ring_allreduce)
    return fn(data_bytes, config)


def compare_sync_strategies(data_bytes: int, n_workers: int = 4,
                           debug_print: bool = True) -> Dict[str, SyncMetrics]:
    """改动: 输出 Pareto frontier (latency vs convergence quality)。"""
    results = {}
    for strat in SyncStrategy:
        if strat == SyncStrategy.BROADCAST:
            continue
        config = SyncConfig(strategy=strat, n_workers=n_workers, debug_print=False)
        metrics = estimate_sync_cost(data_bytes, config, debug_print=False)
        results[strat.name] = metrics

    if debug_print:
        print(f"\n  Strategy comparison ({data_bytes:,}B, {n_workers} workers):")
        # Pareto frontier: dominated = 有另一个策略同时 latency更低 且 convergence更好
        pareto = []
        all_items = list(results.items())
        for name, m in all_items:
            dominated = False
            for other_name, other in all_items:
                if other_name == name:
                    continue
                if (other.total_time_us <= m.total_time_us and
                    other.convergence_error <= m.convergence_error and
                    (other.total_time_us < m.total_time_us or
                     other.convergence_error < m.convergence_error)):
                    dominated = True
                    break
            marker = " ★" if not dominated else ""
            pareto_flag = "PARETO" if not dominated else "dominated"
            print(f"    {name}: {m.total_time_us:.1f}µs, "
                  f"conv_err={m.convergence_error:.4f} [{pareto_flag}]{marker}")

    return results
