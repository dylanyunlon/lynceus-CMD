"""lynceus_port/strategies — 可插拔路由策略实现 (移植版)."""

import sys as _sys, os as _os
_LYNCEUS_DBG = _os.environ.get("LYNCEUS_DEBUG", "1")

def _dbg(tag, msg):
    if _LYNCEUS_DBG != "0":
        print(f"[STR·{tag}] {msg}", file=_sys.stderr, flush=True)

from .base import RoutingDecision, RoutingStrategyBase
from .static import CPUOnlyStrategy, GPUOnlyStrategy, HybridStaticStrategy
from .cost_driven import CostModelRoutedStrategy, PAR2QOEnhancedStrategy
from .adaptive import AdaptiveStrategy

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

_dbg("INIT", f"strategies loaded: {len(__all__)} exports")
