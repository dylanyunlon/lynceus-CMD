"""
lynceus_port — 移植版异构查询路由代价模型.

调试基础设施:
    LYNCEUS_DEBUG 环境变量控制全局诊断输出 (默认关; 设 "1" 打开).
    _dbg(tag, msg) / _dump_obj / _snapshot / _Timer
"""
import os as _os, sys as _sys, time as _time

LYNCEUS_DEBUG: bool = _os.environ.get("LYNCEUS_DEBUG", "0") != "0"
_CALL_SEQ: int = 0

def _dbg(tag: str, msg: str, level: int = 1) -> None:
    global _CALL_SEQ
    if not LYNCEUS_DEBUG: return
    _CALL_SEQ += 1
    print(f"[{_CALL_SEQ:05d}|{tag}] {msg}", file=_sys.stderr, flush=True)

class _Timer:
    _depth = 0
    def __init__(self, label, warn_ms=100.0):
        self.label, self.warn_ms, self.t0, self.dt_ms = label, warn_ms, 0.0, 0.0
    def __enter__(self):
        _Timer._depth += 1; self.t0 = _time.perf_counter(); return self
    def __exit__(self, *_):
        self.dt_ms = (_time.perf_counter() - self.t0) * 1000.0
        w = " SLOW" if self.dt_ms > self.warn_ms else ""
        _dbg("TMR", f"{'  '*_Timer._depth}{self.label}: {self.dt_ms:.2f}ms{w}")
        _Timer._depth = max(0, _Timer._depth - 1)

def _dump_obj(tag, obj, fields=None):
    if not LYNCEUS_DEBUG: return
    if fields is None:
        fields = [k for k in dir(obj) if not k.startswith('_') and not callable(getattr(obj, k, None))]
    parts = []
    for f in fields:
        v = getattr(obj, f, '?')
        if isinstance(v, float): parts.append(f"{f}={v:.6g}")
        elif isinstance(v, list) and len(v) > 20:
            parts.append(f"{f}=[...{len(v)} items]")
        else: parts.append(f"{f}={v!r}")
    _dbg(tag, f"{type(obj).__name__}({', '.join(parts)})")

def _snapshot(tag, label, **kw):
    if not LYNCEUS_DEBUG: return
    _dbg(tag, f"[{label}] " + ", ".join(f"{k}={v!r}" for k, v in kw.items()))

def _dump_dict(tag, d, label=""):
    if not LYNCEUS_DEBUG: return
    prefix = f"{label}: " if label else ""
    _dbg(tag, prefix + str({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in list(d.items())[:10]}))
