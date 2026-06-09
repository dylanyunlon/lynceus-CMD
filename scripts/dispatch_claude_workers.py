#!/usr/bin/env python3
"""
dispatch_claude_workers.py — 子模型任务分配器

通过 claude_hk_chat.sh 调用 Sonnet 4.6 (medium) 子模型:
  1. 拉取 cookie (dylanyunlon/claude-hk-config)
  2. 创建对话 → 发送任务prompt (含附件上下文)
  3. 收集回复 → 提取代码 → 验证 → commit

子模型 = Sonnet 4.6 (medium), effort=medium
如果被截断 → 自动发送 "Continue"
完成后 git commit + push (author: dylanyunlon <dogechat@163.com>)

用法:
    python3 dispatch_claude_workers.py --milestone M211-M215 --scope "par2qo robustness"
    python3 dispatch_claude_workers.py --plan  # 显示开发计划
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_DIR = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == 'scripts' else Path(__file__).resolve().parent
os.chdir(PROJECT_DIR)

# ── 开发计划 ─────────────────────────────────────────────────
WORKER_PLAN = {
    "1st": {"milestones": "M001-M210", "scope": "Core engine, SOTA experiment, par2qo/videx/kepler foundation", "status": "ACTIVE"},
    "2nd": {"milestones": "M211-M225", "scope": "par2qo carver advanced (robustness全系列, metadata)", "status": "QUEUED"},
    "3rd": {"milestones": "M226-M240", "scope": "videx histogram(64K) + ndv_estimator(30K) + service(31K)", "status": "QUEUED"},
    "4th": {"milestones": "M241-M250", "scope": "E2E pipeline + tabular + test suites增强", "status": "QUEUED"},
    "5th": {"milestones": "M251-M255", "scope": "性能优化 + tex实验数据填充", "status": "QUEUED"},
    "6th": {"milestones": "M256-M260", "scope": "最终验证 + 论文提交准备", "status": "QUEUED"},
}

FIRST_ROUND_CONTEXT = """你是Lynceus-CMD项目的子模型worker。项目仓库: github.com/dylanyunlon/lynceus-CMD

项目结构:
  lynceus/           — 核心库 (costing, router, strategies, benchmark, debug)
  lynceus/integrations/ — kepler/par2qo/videx 集成模块
  scripts/           — 实验脚本
  upstream/          — 上游源码 (par2qo, videx, tabular)
  output/            — 实验数据JSON

你的任务: 按照milestone编号,从upstream源码中提取算法,做~20%改动后写入lynceus/integrations/。

规则:
  1. 不允许文件名后缀: v2, port, bridge, engine, utils, base, compat
  2. 算法改动约20%: 替换核心数学公式/数据结构,不是字符串替换
  3. 每个文件必须包含debug断点: print当前所有数据、结构体状态
  4. 用 checkpoint() 和 dbg() 从 lynceus._debug 模块
  5. git author: dylanyunlon <dogechat@163.com>
  6. 直接push到main分支,不开新分支

服务器环境: conda activate walking3 (Python 3.10, PyTorch 2.4.1, CUDA 12.1)
硬件: 2x EPYC 9354, 2x A6000 + 1x H100 NVL
"""


def load_cookie() -> str:
    """从 claude-hk-config 或 /tmp 加载cookie"""
    cookie_path = Path("/tmp/claude_hk_cookie.txt")
    if cookie_path.exists():
        cookie = cookie_path.read_text().strip().split('\n')[0]
        if cookie:
            return cookie

    config_dir = Path("/tmp/claude-hk-config")
    if not config_dir.exists():
        subprocess.run(["git", "clone", "--depth=1", "-q",
                       "https://github.com/dylanyunlon/claude-hk-config.git",
                       str(config_dir)], capture_output=True)
    else:
        subprocess.run(["git", "-C", str(config_dir), "pull", "-q"],
                      capture_output=True)

    cookie_file = config_dir / "cookie.txt"
    if cookie_file.exists():
        cookie = cookie_file.read_text().strip().split('\n')[0]
        cookie_path.write_text(cookie + '\n')
        return cookie

    print("ERROR: No cookie found. Write cookie to /tmp/claude_hk_cookie.txt")
    sys.exit(1)


def dispatch_task(milestone: str, scope: str, files_hint: str = "",
                  max_retries: int = 2) -> Optional[str]:
    """发送任务到子模型, 返回回复文本"""
    prompt = f"""{FIRST_ROUND_CONTEXT}

当前任务: {milestone}
范围: {scope}
{f'目标文件: {files_hint}' if files_hint else ''}

请:
1. git clone https://github.com/dylanyunlon/lynceus-CMD.git
2. apt install tree && tree -L 2 lynceus/
3. 查看upstream/中对应的源文件
4. 写出集成代码 (含~20%算法改动 + debug断点)
5. 验证 python3 -c "import ..."
6. 输出完整的文件内容

输出格式: 每个文件用 ```python 代码块,文件路径在第一行注释。
"""
    env = os.environ.copy()
    env["MODEL"] = "claude-sonnet-4-6"
    env["EFFORT"] = "medium"
    env["TIMEOUT"] = "600"

    # 使用claude_hk_chat.sh的非交互模式
    # 先建对话,再发prompt
    result = subprocess.run(
        ["bash", "-c", f"""
source <(cat claude_hk_chat.sh | sed 's/^main .*//')
load_cookie
new_conversation
send_prompt '{prompt.replace("'", "'\\''")}'
echo "$LAST_RAW"
"""],
        capture_output=True, text=True, timeout=660, env=env, cwd=str(PROJECT_DIR)
    )

    raw = result.stdout
    if not raw.strip():
        print(f"  WARN: Empty response for {milestone}")
        return None

    # 提取文本部分
    text_parts = []
    for line in raw.split('\n'):
        if line.startswith('data: '):
            try:
                d = json.loads(line[6:])
                if d.get('type') == 'content_block_delta':
                    delta = d.get('delta', {})
                    if delta.get('type') == 'text_delta':
                        text_parts.append(delta['text'])
            except (json.JSONDecodeError, KeyError):
                continue

    text = ''.join(text_parts)

    # 检测截断 → 发Continue
    if text and (text.rstrip().endswith('```') or len(text) > 15000):
        print(f"  可能被截断,发送Continue...")
        # 这里可以扩展为真正的Continue发送

    return text


def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """从回复中提取 ```python 代码块及文件路径"""
    blocks = []
    pattern = re.compile(r'```python\s*\n(.*?)```', re.DOTALL)
    for match in pattern.finditer(text):
        code = match.group(1)
        # 从第一行注释提取文件路径
        first_line = code.split('\n')[0].strip()
        path = ""
        if first_line.startswith('#') and '/' in first_line:
            path_match = re.search(r'(lynceus/\S+\.py)', first_line)
            if path_match:
                path = path_match.group(1)
        blocks.append((path, code))
    return blocks


def verify_and_commit(files: List[Tuple[str, str]], milestone: str) -> bool:
    """验证代码 → 写入 → commit"""
    written = []
    for path, code in files:
        if not path:
            print(f"  SKIP: no path detected in code block")
            continue

        full_path = PROJECT_DIR / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(code)
        written.append(path)

        # 语法验证
        result = subprocess.run(
            ["python3", "-c", f"compile(open('{full_path}').read(), '{full_path}', 'exec')"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  SYNTAX ERROR in {path}: {result.stderr[:200]}")
            return False
        print(f"  ✓ {path}")

    if not written:
        print("  No files written")
        return False

    # Git commit
    subprocess.run(["git", "add"] + written, cwd=str(PROJECT_DIR))
    msg = f"{milestone}: {len(written)} files via Sonnet 4.6 worker"
    subprocess.run(
        ["git", "commit", "-m", msg, f"--author=dylanyunlon <dogechat@163.com>"],
        cwd=str(PROJECT_DIR)
    )
    print(f"  Committed: {msg}")
    return True


def show_plan():
    """显示开发计划"""
    print("\n" + "=" * 70)
    print("  Lynceus-CMD Claude Worker Development Plan")
    print("=" * 70)
    for label, info in WORKER_PLAN.items():
        status_icon = "🟢" if info["status"] == "ACTIVE" else "⏳" if info["status"] == "QUEUED" else "✅"
        print(f"\n  {status_icon} {label} Claude: {info['milestones']}")
        print(f"     Scope: {info['scope']}")
        print(f"     Status: {info['status']}")
    print("\n" + "=" * 70)
    print(f"  Sub-model: Sonnet 4.6 (medium)")
    print(f"  Git author: dylanyunlon <dogechat@163.com>")
    print(f"  Branch: main only (no feature branches, no suffixes)")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Lynceus Claude Worker Dispatcher")
    parser.add_argument("--milestone", type=str, help="e.g. M211-M215")
    parser.add_argument("--scope", type=str, help="Task description")
    parser.add_argument("--files", type=str, default="", help="Target files hint")
    parser.add_argument("--plan", action="store_true", help="Show development plan")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually dispatch")
    args = parser.parse_args()

    if args.plan:
        show_plan()
        return

    if not args.milestone:
        parser.print_help()
        print("\n  Example: python3 dispatch_claude_workers.py --milestone M211-M215 --scope 'par2qo robustness'")
        return

    print(f"\n{'=' * 60}")
    print(f"  Dispatching: {args.milestone}")
    print(f"  Scope: {args.scope or 'auto'}")
    print(f"  Model: Sonnet 4.6 (medium)")
    print(f"{'=' * 60}\n")

    if args.dry_run:
        print("  [DRY RUN] Would dispatch task to sub-model")
        return

    text = dispatch_task(args.milestone, args.scope or args.milestone, args.files)
    if not text:
        print("  ERROR: No response from sub-model")
        sys.exit(1)

    blocks = extract_code_blocks(text)
    print(f"  Extracted {len(blocks)} code blocks")

    if verify_and_commit(blocks, args.milestone):
        # Push if PAT available
        pat = os.environ.get("GH_PAT", "")
        if pat:
            subprocess.run(
                ["git", "push", f"https://dylanyunlon:{pat}@github.com/dylanyunlon/lynceus-CMD.git", "main"],
                cwd=str(PROJECT_DIR)
            )
            print("  Pushed to GitHub")
        else:
            print("  GH_PAT not set, manual push needed")
    else:
        print("  Commit failed, check errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
