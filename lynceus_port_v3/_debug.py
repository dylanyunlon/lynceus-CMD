"""
lynceus_port_v3/_debug.py — 增强型断点调试 / 状态快照 / 运行时检查点

与原始 _debug.py 相比, 本版本增加:
  1. checkpoint_to_file(): 把当前所有状态写入 JSON 文件,
     可在崩溃后从文件恢复查看
  2. timing_ctx(): 上下文管理器, 测量代码块耗时
  3. inspect_struct(): 递归打印结构体的每一层嵌套
  4. assertion_hook(): 带详细dump的assert,失败时自动snapshot

用法:
    from ._debug import dbg, snapshot, timing, checkpoint, inspect_struct

    with timing("route_batch"):
        decisions = router.route_batch(queries)

    checkpoint("after_routing", engine=engine, decisions=decisions)

    inspect_struct(cost_breakdown, depth=3)

环境变量:
    LYNCEUS_DBG=0        静默关闭所有调试输出
    LYNCEUS_DBG_FILE=path  调试输出写入文件而非stderr
    LYNCEUS_CHECKPOINT_DIR=path  checkpoint文件输出目录
"""

import sys
import os
import time
import json
import traceback
from contextlib import contextmanager
from dataclasses import fields, asdict
from pathlib import Path
from typing import Any, Optional


class DebugPrinter:
    """可开关的调试打印器, 支持文件输出和计数统计."""

    def __init__(self):
        self.enabled = os.environ.get("LYNCEUS_DBG", "1") != "0"
        self._call_counts = {}
        self._timing_stack = []
        self._accumulated_timings = {}

        # 输出目标: stderr 或指定文件
        dbg_file = os.environ.get("LYNCEUS_DBG_FILE")
        if dbg_file:
            self._file = open(dbg_file, "a", buffering=1)
        else:
            self._file = sys.stderr

        # checkpoint目录
        self._ckpt_dir = Path(
            os.environ.get("LYNCEUS_CHECKPOINT_DIR", "/tmp/lynceus_checkpoints")
        )
        if self.enabled:
            self._ckpt_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, tag: str, **kwargs):
        """带标签的调试快照打印.

        每次调用自动编号, 方便追踪调用序列.
        可以在这里设置 breakpoint() 进入 pdb 交互式调试.
        """
        if not self.enabled:
            return

        self._call_counts[tag] = self._call_counts.get(tag, 0) + 1
        seq = self._call_counts[tag]
        ts = time.strftime("%H:%M:%S")

        print(f"\n{'━'*60}", file=self._file)
        print(f"[DBG #{seq:05d}] {tag}  @ {ts}", file=self._file)

        for k, v in kwargs.items():
            self._render_value(k, v, indent=2)

        print(f"{'━'*60}", file=self._file)
        self._file.flush()

    def _render_value(self, name: str, val: Any, indent: int = 2):
        pad = " " * indent
        if val is None:
            print(f"{pad}{name} = None", file=self._file)
        elif hasattr(val, '__dataclass_fields__'):
            print(f"{pad}{name}: <{type(val).__name__}>", file=self._file)
            for fld in fields(val):
                fval = getattr(val, fld.name)
                truncated = repr(fval)
                if len(truncated) > 120:
                    truncated = truncated[:117] + "..."
                print(f"{pad}  .{fld.name} = {truncated}", file=self._file)
        elif isinstance(val, dict):
            print(f"{pad}{name}: dict[{len(val)} entries]", file=self._file)
            for i, (dk, dv) in enumerate(val.items()):
                if i >= 8:
                    print(f"{pad}  ... +{len(val)-8} more entries", file=self._file)
                    break
                print(f"{pad}  [{dk}] = {repr(dv)[:90]}", file=self._file)
        elif isinstance(val, (list, tuple)):
            container_type = "list" if isinstance(val, list) else "tuple"
            print(f"{pad}{name}: {container_type}[{len(val)}]", file=self._file)
            for i, item in enumerate(val):
                if i >= 5:
                    print(f"{pad}  ... +{len(val)-5} more items", file=self._file)
                    break
                print(f"{pad}  [{i}] {repr(item)[:90]}", file=self._file)
        else:
            s = repr(val)
            if len(s) > 160:
                s = s[:157] + "..."
            print(f"{pad}{name} = {s}", file=self._file)

    def dump_call_stats(self):
        """打印所有调试点的调用次数统计 — 用于确认代码路径覆盖."""
        if not self.enabled:
            return
        print(f"\n{'='*60}", file=self._file)
        print("[DBG CALL STATISTICS]", file=self._file)
        for tag in sorted(self._call_counts, key=self._call_counts.get, reverse=True):
            print(f"  {self._call_counts[tag]:6d}x  {tag}", file=self._file)
        print(f"{'='*60}", file=self._file)

    def dump_timing_stats(self):
        """打印所有计时区间的累积耗时 — 发现性能瓶颈."""
        if not self.enabled or not self._accumulated_timings:
            return
        print(f"\n{'='*60}", file=self._file)
        print("[DBG TIMING STATISTICS]", file=self._file)
        for label, (total_s, count) in sorted(
            self._accumulated_timings.items(),
            key=lambda kv: kv[1][0], reverse=True
        ):
            avg_ms = (total_s / count) * 1000.0 if count else 0
            print(f"  {label:40s}  total={total_s*1000:.1f}ms  "
                  f"calls={count}  avg={avg_ms:.2f}ms", file=self._file)
        print(f"{'='*60}", file=self._file)


def snapshot(obj: Any, label: str = ""):
    """完整状态快照 — 用于pdb中或关键路径上查看对象全貌."""
    if not dbg.enabled:
        return
    tag = label or type(obj).__name__
    if hasattr(obj, 'dump_state'):
        print(f"\n[SNAPSHOT] {tag}:", file=dbg._file)
        state_str = str(obj.dump_state())
        # 截断过长的状态
        if len(state_str) > 2000:
            state_str = state_str[:2000] + "\n  ... (truncated)"
        print(state_str, file=dbg._file)
    elif hasattr(obj, '__dataclass_fields__'):
        dbg(f"snapshot:{tag}",
            **{fld.name: getattr(obj, fld.name) for fld in fields(obj)})
    elif hasattr(obj, '__dict__'):
        public = {k: v for k, v in vars(obj).items() if not k.startswith('__')}
        dbg(f"snapshot:{tag}", **public)
    else:
        print(f"[SNAPSHOT] {tag} = {repr(obj)[:500]}", file=dbg._file)


@contextmanager
def timing(label: str):
    """计时上下文管理器 — 测量代码块执行时间.

    用法:
        with timing("cost_estimation"):
            result = engine.estimate_all_devices(query)

    输出:
        [TIMING] cost_estimation: 12.34 ms
    """
    if not dbg.enabled:
        yield
        return

    t0 = time.perf_counter()
    dbg._timing_stack.append(label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        dbg._timing_stack.pop()
        # 累积统计
        if label not in dbg._accumulated_timings:
            dbg._accumulated_timings[label] = (0.0, 0)
        prev_total, prev_count = dbg._accumulated_timings[label]
        dbg._accumulated_timings[label] = (prev_total + elapsed, prev_count + 1)

        depth_marker = "│ " * len(dbg._timing_stack)
        print(f"[TIMING] {depth_marker}{label}: {elapsed*1000:.2f} ms",
              file=dbg._file)


def checkpoint(tag: str, **kwargs):
    """把当前状态写入JSON checkpoint文件.

    用法:
        checkpoint("after_routing", engine_state=engine, decisions=decs)

    生成:
        /tmp/lynceus_checkpoints/after_routing_001.json

    可以在实验崩溃后用 json.load 恢复查看当时的状态.
    """
    if not dbg.enabled:
        return

    dbg._call_counts[f"ckpt:{tag}"] = dbg._call_counts.get(f"ckpt:{tag}", 0) + 1
    seq = dbg._call_counts[f"ckpt:{tag}"]

    # 序列化所有kwargs
    serializable = {}
    for k, v in kwargs.items():
        try:
            if hasattr(v, '__dataclass_fields__'):
                serializable[k] = asdict(v)
            elif isinstance(v, (dict, list, tuple, str, int, float, bool)):
                serializable[k] = v
            else:
                serializable[k] = repr(v)[:500]
        except Exception as e:
            serializable[k] = f"<unserializable: {e}>"

    serializable["_meta"] = {
        "tag": tag,
        "seq": seq,
        "timestamp": time.time(),
        "timestamp_human": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    ckpt_path = dbg._ckpt_dir / f"{tag}_{seq:04d}.json"
    try:
        with open(ckpt_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"[CHECKPOINT] wrote {ckpt_path}", file=dbg._file)
    except Exception as e:
        print(f"[CHECKPOINT] FAILED to write {ckpt_path}: {e}", file=dbg._file)


def inspect_struct(obj: Any, depth: int = 2, _current_depth: int = 0):
    """递归打印结构体的嵌套字段 — 比snapshot更深入.

    用法:
        inspect_struct(topology, depth=3)
    """
    if not dbg.enabled:
        return
    pad = "  " * _current_depth
    name = type(obj).__name__

    if _current_depth >= depth:
        print(f"{pad}<{name}> ...(max depth)", file=dbg._file)
        return

    if hasattr(obj, '__dataclass_fields__'):
        print(f"{pad}<{name}>", file=dbg._file)
        for fld in fields(obj):
            val = getattr(obj, fld.name)
            if hasattr(val, '__dataclass_fields__'):
                print(f"{pad}  .{fld.name}:", file=dbg._file)
                inspect_struct(val, depth, _current_depth + 2)
            elif isinstance(val, dict) and val:
                print(f"{pad}  .{fld.name}: dict[{len(val)}]", file=dbg._file)
                for dk, dv in list(val.items())[:3]:
                    if hasattr(dv, '__dataclass_fields__'):
                        print(f"{pad}    [{dk}]:", file=dbg._file)
                        inspect_struct(dv, depth, _current_depth + 3)
                    else:
                        print(f"{pad}    [{dk}] = {repr(dv)[:80]}", file=dbg._file)
            else:
                print(f"{pad}  .{fld.name} = {repr(val)[:100]}", file=dbg._file)
    elif isinstance(obj, dict):
        print(f"{pad}dict[{len(obj)}]", file=dbg._file)
        for dk, dv in list(obj.items())[:5]:
            print(f"{pad}  [{dk}]:", file=dbg._file)
            inspect_struct(dv, depth, _current_depth + 1)
    else:
        print(f"{pad}{repr(obj)[:200]}", file=dbg._file)


def assert_valid(condition: bool, msg: str, **context):
    """带详细上下文的断言 — 失败时自动dump所有上下文变量.

    用法:
        assert_valid(cost > 0, "cost must be positive",
                     query=query, device=dev, cost=cost)
    """
    if condition:
        return
    print(f"\n{'!'*60}", file=dbg._file)
    print(f"[ASSERTION FAILED] {msg}", file=dbg._file)
    for k, v in context.items():
        dbg._render_value(k, v, indent=2)
    print(f"Stack trace:", file=dbg._file)
    traceback.print_stack(file=dbg._file)
    print(f"{'!'*60}", file=dbg._file)
    raise AssertionError(msg)


# 全局单例
dbg = DebugPrinter()
