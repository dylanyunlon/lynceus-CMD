"""
_debug.py — 增强型断点调试 / 状态快照 / 运行时诊断基础设施

移植改写 (~40% 新增):
  - 原有: DebugPrinter, snapshot 基本功能
  - 新增: 计时器装饰器 @timed, 调用链追踪 CallTracer,
          结构体差分 diff_snapshot, 断言守卫 guard,
          运行时统计仪表盘 RuntimeStats, checkpoint/restore 机制

在任何模块中:
    from ._debug import dbg, snapshot, timed, tracer, guard

    dbg("tag", var1=x, var2=y)        # 打印到 stderr
    snapshot(obj)                       # dump 对象所有属性
    @timed("label")                    # 函数耗时统计
    guard(cond, "msg", ctx={...})      # 条件断言 + 上下文转储
    tracer.enter("func"); tracer.exit()# 调用链追踪

在运行实验时:
    LYNCEUS_DBG=0 python -m benchmark   # 环境变量控制
    LYNCEUS_DBG=2 ...                   # 级别2: 含计时+调用链
"""

import sys
import os
import time
import functools
import threading
from dataclasses import fields, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict


# ─── 调试级别 ────────────────────────────────────────────────────────────────
# 0=静默  1=基本dbg  2=含计时+调用链  3=全量(含diff)
_DBG_LEVEL = int(os.environ.get("LYNCEUS_DBG", "1"))


class CallTracer:
    """调用链追踪器 — 记录函数进入/退出, 构建实时调用栈.

    用法:
        tracer.enter("CostModel.estimate", query_id="q_001")
        ... 执行逻辑 ...
        tracer.exit(result_summary="cost=42.5µs")

    输出:
        ┌─ CostModel.estimate (query_id=q_001)
        │  ┌─ CPUCostModel.calc_io
        │  └─ CPUCostModel.calc_io → 12.3µs
        └─ CostModel.estimate → cost=42.5µs
    """

    def __init__(self):
        self._stack: List[Tuple[str, float, Dict]] = []  # (name, t0, kwargs)
        self._depth = 0
        self._log: List[str] = []
        self._lock = threading.Lock()

    def enter(self, name: str, **ctx) -> None:
        if _DBG_LEVEL < 2:
            return
        with self._lock:
            indent = "│  " * self._depth
            ctx_str = ", ".join(f"{k}={v}" for k, v in ctx.items()) if ctx else ""
            msg = f"{indent}┌─ {name}" + (f" ({ctx_str})" if ctx_str else "")
            print(msg, file=sys.stderr)
            self._log.append(msg)
            self._stack.append((name, time.perf_counter(), ctx))
            self._depth += 1

    def exit(self, result_summary: str = "") -> float:
        if _DBG_LEVEL < 2:
            return 0.0
        with self._lock:
            if not self._stack:
                return 0.0
            name, t0, _ = self._stack.pop()
            elapsed_us = (time.perf_counter() - t0) * 1e6
            self._depth = max(0, self._depth - 1)
            indent = "│  " * self._depth
            suffix = f" → {result_summary}" if result_summary else ""
            msg = f"{indent}└─ {name}{suffix} [{elapsed_us:.1f}µs]"
            print(msg, file=sys.stderr)
            self._log.append(msg)
            return elapsed_us

    @property
    def call_log(self) -> List[str]:
        return list(self._log)

    def clear(self) -> None:
        with self._lock:
            self._stack.clear()
            self._log.clear()
            self._depth = 0


class RuntimeStats:
    """运行时统计仪表盘 — 跨模块汇聚的计数器/直方图/计时."""

    def __init__(self):
        self.counters: Dict[str, int] = defaultdict(int)
        self.timers: Dict[str, List[float]] = defaultdict(list)  # µs
        self.gauges: Dict[str, float] = {}

    def incr(self, key: str, n: int = 1) -> None:
        self.counters[key] += n

    def record_time(self, key: str, us: float) -> None:
        self.timers[key].append(us)

    def set_gauge(self, key: str, val: float) -> None:
        self.gauges[key] = val

    def report(self) -> str:
        lines = ["╔══ RuntimeStats Dashboard ════════════════════════════"]
        if self.counters:
            lines.append("║ ── Counters ──")
            for k, v in sorted(self.counters.items()):
                lines.append(f"║   {k:30s} = {v:>10,}")
        if self.timers:
            lines.append("║ ── Timers (µs) ──")
            for k, vs in sorted(self.timers.items()):
                if not vs:
                    continue
                avg = sum(vs) / len(vs)
                mn, mx = min(vs), max(vs)
                lines.append(f"║   {k:30s}  n={len(vs):>5}  "
                             f"avg={avg:>10.1f}  min={mn:>10.1f}  max={mx:>10.1f}")
        if self.gauges:
            lines.append("║ ── Gauges ──")
            for k, v in sorted(self.gauges.items()):
                lines.append(f"║   {k:30s} = {v:.4f}")
        lines.append("╚══════════════════════════════════════════════════════")
        return "\n".join(lines)


class DebugPrinter:
    """可开关的调试打印器, 像现实世界开发中的 log/trace 系统.

    改写: 增加了级别控制、计数阈值报警、自动截断深层嵌套.
    """

    def __init__(self):
        self.enabled = _DBG_LEVEL >= 1
        self._counts: Dict[str, int] = defaultdict(int)
        self._file = sys.stderr
        self._alert_threshold = int(os.environ.get("LYNCEUS_ALERT_AFTER", "500"))

    def __call__(self, tag: str, **kwargs) -> None:
        """打印带 tag 的调试快照.

        用法:
            dbg("CostModel.estimate", query=q, device_id=dev, result=cb)

        输出到 stderr, 不干扰正常 stdout 数据流.
        可以在这里 set breakpoint() 进入 pdb.
        """
        if not self.enabled:
            return

        self._counts[tag] += 1
        n = self._counts[tag]
        runtime_stats.incr(f"dbg.{tag}")

        # 高频标签降采样: 超过阈值后每100次才打印一次
        if n > self._alert_threshold and n % 100 != 0:
            return

        print(f"\n{'─'*60}", file=self._file)
        hdr = f"[DBG #{n:04d}] {tag}"
        if n == self._alert_threshold:
            hdr += f"  ⚠ HIGH-FREQ: {n}+ calls"
        print(hdr, file=self._file)

        for k, v in kwargs.items():
            self._print_value(k, v, indent=2)

        print(f"{'─'*60}", file=self._file)

    def _print_value(self, name: str, val: Any, indent: int = 2) -> None:
        pad = " " * indent
        if val is None:
            print(f"{pad}{name} = None", file=self._file)
        elif hasattr(val, '__dataclass_fields__'):
            print(f"{pad}{name}: {type(val).__name__}", file=self._file)
            for f in fields(val):
                fval = getattr(val, f.name)
                s = repr(fval)
                if len(s) > 120:
                    s = s[:117] + "..."
                print(f"{pad}  .{f.name} = {s}", file=self._file)
        elif isinstance(val, dict):
            print(f"{pad}{name}: dict[{len(val)}]", file=self._file)
            for i, (dk, dv) in enumerate(val.items()):
                if i >= 8:
                    print(f"{pad}  ... +{len(val)-8} more", file=self._file)
                    break
                print(f"{pad}  [{dk}] = {repr(dv)[:80]}", file=self._file)
        elif isinstance(val, (list, tuple)):
            print(f"{pad}{name}: {type(val).__name__}[{len(val)}]", file=self._file)
            for item in val[:5]:
                print(f"{pad}  - {repr(item)[:80]}", file=self._file)
            if len(val) > 5:
                print(f"{pad}  ... +{len(val)-5} more", file=self._file)
        else:
            s = repr(val)
            if len(s) > 160:
                s = s[:157] + "..."
            print(f"{pad}{name} = {s}", file=self._file)


def snapshot(obj: Any, label: str = "") -> None:
    """打印对象完整状态 — 用于断点调试时快速查看.

    改写: 增加了时间戳和内存地址, 便于追踪对象生命周期.
    """
    if not dbg.enabled:
        return
    tag = label or type(obj).__name__
    ts = time.strftime("%H:%M:%S")
    addr = f"0x{id(obj):x}"
    print(f"\n[SNAPSHOT {ts}] {tag} @ {addr}", file=sys.stderr)

    if hasattr(obj, 'dump_state'):
        print(obj.dump_state(), file=sys.stderr)
    elif hasattr(obj, '__dataclass_fields__'):
        dbg(f"snapshot:{tag}", **{f.name: getattr(obj, f.name) for f in fields(obj)})
    elif hasattr(obj, '__dict__'):
        public = {k: v for k, v in vars(obj).items() if not k.startswith('_')}
        dbg(f"snapshot:{tag}", **public)
    else:
        print(f"  = {repr(obj)[:400]}", file=sys.stderr)


def diff_snapshot(label: str, before: Dict[str, Any], after: Dict[str, Any]) -> None:
    """对比两个状态字典的差异 — 用于追踪状态突变.

    新增函数: 在修改前后分别capture状态, 然后diff.
    """
    if _DBG_LEVEL < 3:
        return
    all_keys = set(before) | set(after)
    diffs = []
    for k in sorted(all_keys):
        bv = before.get(k, "<ABSENT>")
        av = after.get(k, "<ABSENT>")
        if bv != av:
            diffs.append(f"  Δ {k}: {repr(bv)[:60]} → {repr(av)[:60]}")
    if diffs:
        print(f"\n[DIFF] {label}: {len(diffs)} changes", file=sys.stderr)
        for d in diffs:
            print(d, file=sys.stderr)
    else:
        print(f"\n[DIFF] {label}: no changes", file=sys.stderr)


def guard(condition: bool, message: str, **context) -> None:
    """条件断言守卫 — 断言失败时转储上下文而非直接crash.

    新增函数: 比 assert 更友好, 总是打印诊断信息.
    用法:
        guard(cost > 0, "cost must be positive", cost=cost, device=dev)
    """
    if condition:
        return
    print(f"\n{'!'*60}", file=sys.stderr)
    print(f"[GUARD FAILED] {message}", file=sys.stderr)
    for k, v in context.items():
        print(f"  {k} = {repr(v)[:200]}", file=sys.stderr)
    print(f"{'!'*60}", file=sys.stderr)
    runtime_stats.incr("guard.failures")
    # 不抛异常, 但记录; 想崩溃的话在这里 breakpoint()


def timed(label: str) -> Callable:
    """函数计时装饰器 — 自动记录每次调用的耗时.

    新增函数.
    用法:
        @timed("CostModel.recommend")
        def recommend(self, query): ...

    输出:
        [TIMED] CostModel.recommend: 42.3µs
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_us = (time.perf_counter() - t0) * 1e6
            runtime_stats.record_time(label, elapsed_us)
            if _DBG_LEVEL >= 2:
                print(f"  [TIMED] {label}: {elapsed_us:.1f}µs", file=sys.stderr)
            return result
        return wrapper
    return decorator


# ─── Checkpoint / Restore ────────────────────────────────────────────────────

_checkpoints: Dict[str, Dict[str, Any]] = {}


def checkpoint(name: str, **state) -> None:
    """保存一个命名检查点 — 可在后续对比或恢复.

    新增函数.
    用法:
        checkpoint("before_routing", costs=costs, device=dev)
        ...
        prev = restore("before_routing")
    """
    _checkpoints[name] = {"_ts": time.time(), **state}
    if _DBG_LEVEL >= 2:
        print(f"  [CHECKPOINT] '{name}' saved ({len(state)} fields)", file=sys.stderr)


def restore(name: str) -> Optional[Dict[str, Any]]:
    """恢复一个命名检查点."""
    cp = _checkpoints.get(name)
    if cp is None and _DBG_LEVEL >= 1:
        print(f"  [CHECKPOINT] '{name}' not found", file=sys.stderr)
    return cp


# ─── 全局单例 ────────────────────────────────────────────────────────────────
dbg = DebugPrinter()
tracer = CallTracer()
runtime_stats = RuntimeStats()
