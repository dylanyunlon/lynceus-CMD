"""
lynceus/strategies/cost_driven.py — Cost-model-driven routing strategies.

These strategies use the CostModelEngine to make per-query routing
decisions based on estimated execution cost:
    - CostModelRoutedStrategy: pure min-cost routing
    - PAR2QOEnhancedStrategy: min-cost + robustness penalty margin

Architecture references:
    - NCCL ncclTopoCompute (nccl/src/graph/search.cc:1023)
      → search over all algorithms, pick lowest cost
    - NCCL tuner_v6 cost table (tuner/tuner_v6.h:52)
      → per-algorithm cost estimation with NCCL_ALGO_PROTO_IGNORE=-1.0
    - PAR2QO Diagram.pqoByFeatureCollection (diagram.py:46)
      → parametric penalty-aware robust plan selection
    - PAR2QO get_plan_cost (postgres.py:110)
      → postgres-level plan cost estimation
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..cost_model import CostBreakdown, CostModelEngine, QueryDescriptor
from ..schema import HardwareKind
from .base import RoutingDecision, RoutingStrategyBase


class CostModelRoutedStrategy(RoutingStrategyBase):
    """Route each query to the device with the lowest estimated cost.

    This is the core Lynceus strategy — analogous to NCCL's
    ncclTopoCompute choosing the best ring/tree/collnet algorithm
    by evaluating cost tables for all options and picking the minimum.

    Unlike NCCL (which has a fixed set of algorithms), we search over
    all hardware devices in the topology. The cost model considers:
      - I/O cost (sequential vs random page access)
      - Compute cost (CPU single-thread vs GPU massively-parallel)
      - Transfer cost (PCIe latency + bandwidth)
      - Index access cost (B-tree traversal)
      - Sort cost (CPU merge-sort vs GPU bitonic sort)
    """

    @property
    def name(self) -> str:
        return "CostModel-Routed"

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        device_id, cb = self._engine.recommend(query, data_location)
        return RoutingDecision(
            query_id=query.query_id,
            device_id=device_id,
            cost=cb,
            confidence=1.0,
            metadata={"reason": "min_cost"},
        )

    def route_batch(self, queries: List[QueryDescriptor],
                    data_location: Optional[str] = None
                    ) -> List[RoutingDecision]:
        """Batch routing — could be parallelized in production."""
        results = self._engine.route_batch(queries, data_location)
        return [
            RoutingDecision(
                query_id=q.query_id,
                device_id=dev,
                cost=cb,
                confidence=1.0,
                metadata={"reason": "min_cost"},
            )
            for q, (dev, cb) in zip(queries, results)
        ]


class PAR2QOEnhancedStrategy(RoutingStrategyBase):
    """Cost-model routing with PAR2QO-inspired robustness penalty.

    Adds a margin: if the GPU advantage is marginal (within
    robustness_margin of CPU cost), we prefer CPU to avoid the
    variance introduced by PCIe transfers.

    This models PAR2QO's parametric penalty approach:
    - PAR2QO Diagram.pqoByFeatureCollection (diagram.py:46)
      collects plan features and evaluates robustness
    - PAR2QO Diagram.calReweightProbability (diagram.py:161)
      reweights plan selection based on penalty
    - The "penalty" is: GPU wins by < margin → treat as tie → prefer CPU

    Why: In a real system, the cost model has estimation errors (just
    like PostgreSQL's cardinality estimator). PAR2QO's insight is that
    plans with marginal cost differences should be selected based on
    robustness, not raw estimated cost.
    """

    def __init__(self, engine: CostModelEngine, *,
                 robustness_margin: float = 0.20, **kwargs):
        super().__init__(engine, **kwargs)
        self._margin = robustness_margin

    @property
    def name(self) -> str:
        return "PAR2QO-Enhanced"

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        estimates = self._engine.estimate_all_devices(query, data_location)

        gpu_ids = [k for k, n in self._engine.topology.nodes.items()
                   if n.kind == HardwareKind.GPU and k in estimates]
        cpu_ids = [k for k, n in self._engine.topology.nodes.items()
                   if n.kind == HardwareKind.CPU and k in estimates]

        # Fallback if topology lacks GPU or CPU
        if not gpu_ids or not cpu_ids:
            device_id, cb = self._engine.recommend(query, data_location)
            return RoutingDecision(
                query_id=query.query_id, device_id=device_id,
                cost=cb, confidence=0.5,
                metadata={"reason": "fallback_incomplete_topology"},
            )

        best_gpu = min(gpu_ids, key=lambda k: estimates[k].total_us)
        best_cpu = min(cpu_ids, key=lambda k: estimates[k].total_us)
        gpu_cost = estimates[best_gpu].total_us
        cpu_cost = estimates[best_cpu].total_us

        # PAR2QO penalty: prefer CPU when GPU advantage is marginal
        if gpu_cost < cpu_cost * (1.0 - self._margin):
            chosen = best_gpu
            reason = "gpu_clear_winner"
        else:
            chosen = best_cpu
            reason = "cpu_robust_choice"

        return RoutingDecision(
            query_id=query.query_id,
            device_id=chosen,
            cost=estimates[chosen],
            confidence=1.0 if reason == "gpu_clear_winner" else 0.8,
            metadata={
                "reason": reason,
                "gpu_cost_us": gpu_cost,
                "cpu_cost_us": cpu_cost,
                "margin": self._margin,
                "gpu_advantage_pct": (1.0 - gpu_cost / cpu_cost) * 100
                    if cpu_cost > 0 else 0.0,
            },
        )
