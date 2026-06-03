"""
_debug.py — 断点调试 / 状态快照 基础设施

在任何模块中:
    from ._debug import dbg, snapshot

    dbg("tag", var1=x, var2=y)        # 打印到 stderr
    snapshot(obj)                       # dump 对象所有属性
    dbg.enabled = False                 # 静默关闭

在运行实验时:
    LYNCEUS_DBG=0 python -m benchmark   # 环境变量控制
"""

import sys
import os
import time
from dataclasses import fields, asdict


class DebugPrinter:
    """可开关的调试打印器, 像现实世界开发中的 log/trace 系统."""

    def __init__(self):
        self.enabled = os.environ.get("LYNCEUS_DBG", "1") != "0"
        self._counts = {}
        self._file = sys.stderr

    def __call__(self, tag, **kwargs):
        """打印带 tag 的调试快照.

        用法:
            dbg("CostModel.estimate", query=q, device_id=dev, result=cb)

        输出到 stderr, 不干扰正常 stdout 数据流.
        可以在这里 set breakpoint() 进入 pdb.
        """
        if not self.enabled:
            return

        self._counts[tag] = self._counts.get(tag, 0) + 1
        n = self._counts[tag]

        print(f"\n{'─'*56}", file=self._file)
        print(f"[DBG #{n:04d}] {tag}", file=self._file)

        for k, v in kwargs.items():
            self._print_value(k, v, indent=2)

        print(f"{'─'*56}", file=self._file)

    def _print_value(self, name, val, indent=2):
        pad = " " * indent
        if val is None:
            print(f"{pad}{name} = None", file=self._file)
        elif hasattr(val, '__dataclass_fields__'):
            # dataclass: 打印所有字段
            print(f"{pad}{name}: {type(val).__name__}", file=self._file)
            for f in fields(val):
                fval = getattr(val, f.name)
                s = repr(fval)
                if len(s) > 100:
                    s = s[:100] + "…"
                print(f"{pad}  .{f.name} = {s}", file=self._file)
        elif isinstance(val, dict):
            print(f"{pad}{name}: dict[{len(val)}]", file=self._file)
            for dk, dv in list(val.items())[:6]:
                print(f"{pad}  [{dk}] = {repr(dv)[:80]}", file=self._file)
            if len(val) > 6:
                print(f"{pad}  ... +{len(val)-6} more", file=self._file)
        elif isinstance(val, (list, tuple)):
            print(f"{pad}{name}: {type(val).__name__}[{len(val)}]", file=self._file)
            for item in val[:4]:
                print(f"{pad}  - {repr(item)[:80]}", file=self._file)
            if len(val) > 4:
                print(f"{pad}  ... +{len(val)-4} more", file=self._file)
        else:
            s = repr(val)
            if len(s) > 150:
                s = s[:150] + "…"
            print(f"{pad}{name} = {s}", file=self._file)


def snapshot(obj, label=""):
    """打印对象完整状态 — 用于断点调试时快速查看."""
    if not dbg.enabled:
        return
    tag = label or type(obj).__name__
    if hasattr(obj, 'dump_state'):
        print(f"\n[SNAPSHOT] {tag}:", file=sys.stderr)
        print(obj.dump_state(), file=sys.stderr)
    elif hasattr(obj, '__dataclass_fields__'):
        dbg(f"snapshot:{tag}", **{f.name: getattr(obj, f.name) for f in fields(obj)})
    elif hasattr(obj, '__dict__'):
        dbg(f"snapshot:{tag}", **{k: v for k, v in vars(obj).items() if not k.startswith('_')})
    else:
        print(f"[SNAPSHOT] {tag} = {repr(obj)[:300]}", file=sys.stderr)


# 全局单例
dbg = DebugPrinter()
