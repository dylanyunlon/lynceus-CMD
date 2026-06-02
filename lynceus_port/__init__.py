"""
lynceus_port — 移植版本, 算法改写 ≈ 20%, 密集调试打桩.

每个模块在原始 lynceus/ 基础上做了:
  1. 核心算法微调 (常数、分支、公式变体)
  2. 全链路 print-trace / dump_state 断点辅助
  3. 关键路径的 _DBG 前缀日志, 可通过 LYNCEUS_DEBUG=1 环境变量打开
"""
import os as _os

LYNCEUS_DEBUG: bool = _os.environ.get("LYNCEUS_DEBUG", "0") == "1"

def _dbg(*args, **kw):
    """条件调试输出 — 运行实验时 export LYNCEUS_DEBUG=1 即可打开."""
    _dbg("_DBG", "_dbg entered")
    if LYNCEUS_DEBUG:
        print("[DBG]", *args, **kw)
