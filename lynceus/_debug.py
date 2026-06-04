"""
_debug.py — 断点调试 / 状态快照 / 运行时诊断 基础设施

扩展能力:
    - 计时器: 自动记录每个 tag 的累积耗时与调用次数
    - 调用栈追踪: 可选打印 caller 信息
    - 结构体 diff: 比较两次 snapshot 之间的变化
    - 断点陷阱: 条件触发 pdb (生产中用 LYNCEUS_TRAP=tag1,tag2)
    - 直方图: 自动统计数值型参数的分布
"""

import sys
import os
import time
import math
import traceback
from dataclasses import fields, asdict
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


class _Histogram:
    """在线直方图，用 Welford 算法跟踪均值/方差，不存原始值。"""
    __slots__ = ('n', '_m', '_s', '_min', '_max')

    def __init__(self):
        self.n = 0
        self._m = 0.0
        self._s = 0.0
        self._min = float('inf')
        self._max = float('-inf')

    def push(self, x: float):
        self.n += 1
        if self.n == 1:
            self._m = x
            self._s = 0.0
        else:
            prev_m = self._m
            self._m += (x - prev_m) / self.n
            self._s += (x - prev_m) * (x - self._m)
        if x < self._min:
            self._min = x
        if x > self._max:
            self._max = x

    @property
    def mean(self) -> float:
        return self._m if self.n > 0 else 0.0

    @property
    def var(self) -> float:
        return self._s / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.var) if self.n > 1 else 0.0

    def summary(self) -> str:
        if self.n == 0:
            return "empty"
        return (f"n={self.n} μ={self.mean:.4g} σ={self.std:.4g} "
                f"[{self._min:.4g}, {self._max:.4g}]")


class DebugPrinter:
    """可开关的调试打印器，带计时、直方图、条件断点。"""

    def __init__(self):
        self.enabled = os.environ.get("LYNCEUS_DBG", "1") != "0"
        self._counts: Dict[str, int] = {}
        self._timers: Dict[str, float] = {}
        self._histograms: Dict[str, _Histogram] = defaultdict(_Histogram)
        self._snapshots: Dict[str, Dict[str, Any]] = {}
        self._file = sys.stderr
        # 条件断点: LYNCEUS_TRAP="CostModel.estimate,Router.route" 触发pdb
        trap_env = os.environ.get("LYNCEUS_TRAP", "")
        self._traps = set(t.strip() for t in trap_env.split(",") if t.strip())
        self._show_caller = os.environ.get("LYNCEUS_CALLER", "0") != "0"

    def __call__(self, tag: str, **kwargs):
        if not self.enabled:
            return
        self._counts[tag] = self._counts.get(tag, 0) + 1
        n = self._counts[tag]

        # 条件断点陷阱
        if tag in self._traps:
            print(f"\n🔴 TRAP [{tag}] hit (call #{n}), dropping into pdb...",
                  file=self._file)
            import pdb; pdb.set_trace()

        print(f"\n{'─'*60}", file=self._file)
        ts = time.strftime("%H:%M:%S")
        print(f"[DBG #{n:04d} @{ts}] {tag}", file=self._file)

        if self._show_caller:
            frame = traceback.extract_stack(limit=3)
            if len(frame) >= 2:
                f = frame[-2]
                print(f"  caller: {f.filename}:{f.lineno} in {f.name}",
                      file=self._file)

        for k, v in kwargs.items():
            self._print_value(k, v, indent=2)
            # 数值型自动进直方图
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                hist_key = f"{tag}/{k}"
                self._histograms[hist_key].push(float(v))

        print(f"{'─'*60}", file=self._file)

    def _print_value(self, name: str, val: Any, indent: int = 2):
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
                print(f"{pad}  [{dk}] = {repr(dv)[:90]}", file=self._file)
        elif isinstance(val, (list, tuple)):
            tp = type(val).__name__
            print(f"{pad}{name}: {tp}[{len(val)}]", file=self._file)
            for item in val[:5]:
                print(f"{pad}  - {repr(item)[:90]}", file=self._file)
            if len(val) > 5:
                print(f"{pad}  ... +{len(val)-5} more", file=self._file)
        else:
            s = repr(val)
            if len(s) > 160:
                s = s[:157] + "..."
            print(f"{pad}{name} = {s}", file=self._file)

    # ─── 计时器 ───────────────────────────────────────────────
    def timer_start(self, tag: str):
        self._timers[tag] = time.perf_counter()

    def timer_stop(self, tag: str) -> float:
        """返回耗时(秒)，同时打印。"""
        start = self._timers.pop(tag, None)
        if start is None:
            return 0.0
        elapsed = time.perf_counter() - start
        if self.enabled:
            print(f"  [TIMER] {tag}: {elapsed*1e6:.1f}µs", file=self._file)
        return elapsed

    # ─── 结构体 diff ──────────────────────────────────────────
    def diff_snapshot(self, label: str, obj: Any) -> Optional[Dict[str, Tuple]]:
        """对比同一个 label 的两次 snapshot，返回变化的字段。"""
        current = self._extract_state(obj)
        prev = self._snapshots.get(label)
        self._snapshots[label] = current
        if prev is None:
            return None
        changes = {}
        all_keys = set(prev.keys()) | set(current.keys())
        for k in all_keys:
            old_v = prev.get(k, "<absent>")
            new_v = current.get(k, "<absent>")
            if old_v != new_v:
                changes[k] = (old_v, new_v)
        if changes and self.enabled:
            print(f"\n[DIFF] {label}: {len(changes)} field(s) changed",
                  file=self._file)
            for k, (ov, nv) in changes.items():
                print(f"  {k}: {repr(ov)[:60]} → {repr(nv)[:60]}",
                      file=self._file)
        return changes

    def _extract_state(self, obj: Any) -> Dict[str, Any]:
        if hasattr(obj, '__dataclass_fields__'):
            return {f.name: getattr(obj, f.name) for f in fields(obj)}
        elif hasattr(obj, '__dict__'):
            return {k: v for k, v in vars(obj).items() if not k.startswith('_')}
        return {"__value__": obj}

    # ─── 直方图报告 ───────────────────────────────────────────
    def report_histograms(self):
        if not self._histograms:
            return
        print(f"\n{'═'*60}", file=self._file)
        print("[HISTOGRAMS] Accumulated numeric distributions:", file=self._file)
        for key in sorted(self._histograms):
            h = self._histograms[key]
            if h.n > 0:
                print(f"  {key}: {h.summary()}", file=self._file)
        print(f"{'═'*60}", file=self._file)

    # ─── 调用频率报告 ─────────────────────────────────────────
    def report_call_counts(self):
        if not self._counts:
            return
        print(f"\n{'═'*60}", file=self._file)
        print("[CALL COUNTS]:", file=self._file)
        for tag, n in sorted(self._counts.items(), key=lambda x: -x[1]):
            print(f"  {tag}: {n}", file=self._file)
        print(f"{'═'*60}", file=self._file)


def snapshot(obj: Any, label: str = ""):
    """打印对象完整状态 — 用于断点调试时快速查看。"""
    if not dbg.enabled:
        return
    tag = label or type(obj).__name__
    if hasattr(obj, 'dump_state'):
        print(f"\n[SNAPSHOT] {tag}:", file=sys.stderr)
        print(obj.dump_state(), file=sys.stderr)
    elif hasattr(obj, '__dataclass_fields__'):
        dbg(f"snapshot:{tag}",
            **{f.name: getattr(obj, f.name) for f in fields(obj)})
    elif hasattr(obj, '__dict__'):
        dbg(f"snapshot:{tag}",
            **{k: v for k, v in vars(obj).items() if not k.startswith('_')})
    else:
        print(f"[SNAPSHOT] {tag} = {repr(obj)[:300]}", file=sys.stderr)


def checkpoint(label: str, **state):
    """轻量断点: 打印当前位置和关键变量，不中断执行。
    用法:
        checkpoint("after_dijkstra", dist=dist, path=path, cost=total_cost)
    """
    if not dbg.enabled:
        return
    ts = time.strftime("%H:%M:%S.") + f"{time.time()%1:.3f}"[2:]
    frame = traceback.extract_stack(limit=2)[-2]
    loc = f"{frame.filename.split('/')[-1]}:{frame.lineno}"
    print(f"\n  ◆ CHECKPOINT [{label}] @{ts} ({loc})", file=sys.stderr)
    for k, v in state.items():
        s = repr(v)
        if len(s) > 120:
            s = s[:117] + "..."
        print(f"    {k} = {s}", file=sys.stderr)


# 全局单例
dbg = DebugPrinter()
