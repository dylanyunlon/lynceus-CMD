"""lynceus_port/integrations — 上游系统集成桥接 (移植版)."""

import sys as _sys, os as _os
_LYNCEUS_DBG = _os.environ.get("LYNCEUS_DEBUG", "1")

def _dbg(tag, msg):
    if _LYNCEUS_DBG != "0":
        print(f"[INT·{tag}] {msg}", file=_sys.stderr, flush=True)

_AVAILABLE_BRIDGES = {
    "par2qo": ".par2qo_bridge",
    "videx": ".videx_bridge",
    "tabular": ".tabular_bridge",
}

def list_bridges():
    """返回所有可用桥接名称."""
    _dbg("LIST", f"available bridges: {list(_AVAILABLE_BRIDGES.keys())}")
    return list(_AVAILABLE_BRIDGES.keys())

from .par2qo_utils import (
    extract_join_predicates,
    parse_plan_tree,
    normalize_cost,
)

__all__ = [
    "list_bridges",
    "extract_join_predicates",
    "parse_plan_tree",
    "normalize_cost",
]

_dbg("INIT", f"integrations package ready, bridges={list(_AVAILABLE_BRIDGES.keys())}")
