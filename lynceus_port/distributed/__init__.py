"""lynceus_port/distributed — 分布式 cost model 通信层 (移植版)."""

import sys as _sys, os as _os
_LYNCEUS_DBG = _os.environ.get("LYNCEUS_DEBUG", "1")

def _dbg(tag, msg):
    if _LYNCEUS_DBG != "0":
        print(f"[DST·{tag}] {msg}", file=_sys.stderr, flush=True)

_dbg("INIT", "distributed package loaded")
