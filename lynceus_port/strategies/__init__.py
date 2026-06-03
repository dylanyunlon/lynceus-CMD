"""lynceus/strategies — Pluggable routing strategy implementations."""

from .base import RoutingDecision, RoutingStrategyBase
from .static import CPUOnlyStrategy, GPUOnlyStrategy, HybridStaticStrategy
from .cost_driven import CostModelRoutedStrategy, PAR2QOEnhancedStrategy
from .adaptive import AdaptiveStrategy

from .. import _dbg, _dump_obj, _snapshot, _Timer, LYNCEUS_DEBUG
_T = "__I"


__all__ = [
    "RoutingDecision",
    "RoutingStrategyBase",
    "CPUOnlyStrategy",
    "GPUOnlyStrategy",
    "HybridStaticStrategy",
    "CostModelRoutedStrategy",
    "PAR2QOEnhancedStrategy",
    "AdaptiveStrategy",
]
