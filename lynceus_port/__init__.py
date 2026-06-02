import os as _os, sys as _sys, time as _time

LYNCEUS_DEBUG: bool = _os.environ.get("LYNCEUS_DEBUG", "0") != "0"

def _dbg(tag: str, msg: str):
    if LYNCEUS_DEBUG:
        print(f"[{tag}] {msg}", file=_sys.stderr, flush=True)

class _Timer:
    """调试用计时器, with 语句块出入自动打印耗时."""
    def __init__(self, label):
        self.label = label
        self.t0 = 0.0
    def __enter__(self):
        self.t0 = _time.perf_counter()
        return self
    def __exit__(self, *_):
        dt = (_time.perf_counter() - self.t0) * 1000
        _dbg("TIMER", f"{self.label}: {dt:.2f}ms")

def _dump_obj(tag: str, obj, fields=None):
    """调试: 打印对象的指定字段(或全部非下划线字段)到 stderr."""
    if not LYNCEUS_DEBUG:
        return
    if fields is None:
        fields = [k for k in dir(obj) if not k.startswith('_') and not callable(getattr(obj, k, None))]
    parts = []
    for f in fields:
        v = getattr(obj, f, '?')
        if isinstance(v, float):
            parts.append(f"{f}={v:.4f}")
        else:
            parts.append(f"{f}={v}")
    print(f"[{tag}·DUMP] {' '.join(parts)}", file=_sys.stderr, flush=True)
