#!/usr/bin/env python3
"""dispatch_session12.py — Session 12: 子模型批量移植 (M181-M196)

第一位Claude (指挥官) 通过 claude_hk_chat.sh 派遣 Opus 4.6 子模型
处理未移植的 upstream 文件。

每个Worker收到:
  1. clone指令
  2. upstream源码路径
  3. 移植规则 (鲁迅式mv + 20%算法改动 + debug断点)
  4. git commit规范

作者: dylanyunlon <dogechat@163.com>
"""
import json
import os
import subprocess
import sys
import time
import uuid

REPO = "https://github.com/dylanyunlon/lynceus-CMD.git"
GIT_TOKEN = os.environ.get("GIT_TOKEN", "SET_VIA_ENV_VAR")
COOKIE_FILE = "/tmp/claude_hk_cookie.txt"

# Worker任务分配
WORKERS = [
    {
        "id": "W1-M181-M183",
        "milestone": "M181-M183",
        "files_to_port": [
            ("upstream/par2qo/code/carver/3_model_test.py", "par2qo_model_test.py",
             "carver model test: numpy MLP forward pass mock, confusion matrix, per-query accuracy breakdown"),
            ("upstream/par2qo/code/carver/3_model_test_mixture.py", "par2qo_model_test_mixture.py",
             "mixture model test: multi-distribution evaluation, KL divergence between predicted vs actual"),
            ("upstream/par2qo/code/carver/3_model_test_single.py", "par2qo_model_test_single.py",
             "single template model test: isolated query template accuracy, Welford online variance"),
        ],
    },
    {
        "id": "W2-M184-M186",
        "milestone": "M184-M186",
        "files_to_port": [
            ("upstream/par2qo/code/carver/4_evaluate.py", "par2qo_evaluate.py",
             "evaluation pipeline: plan cost comparison, regret computation, rank correlation"),
            ("upstream/par2qo/code/carver/4_evaluate_both.py", "par2qo_evaluate_both.py",
             "joint evaluation: parametric + non-parametric comparison, SMAPE, NDCG"),
            ("upstream/par2qo/code/carver/4_evaluate_PQO.py", "par2qo_evaluate_pqo.py",
             "PQO evaluation: parametric query optimization metric, harmonic mean aggregation"),
        ],
    },
    {
        "id": "W3-M187-M189",
        "milestone": "M187-M189",
        "files_to_port": [
            ("upstream/par2qo/code/carver/5_visualize.py", "par2qo_visualize.py",
             "visualization: ASCII/matplotlib plan cost distribution, CDF plots"),
            ("upstream/par2qo/code/carver/5_visualize_PQO.py", "par2qo_visualize_pqo.py",
             "PQO visualization: parametric performance heatmap, threshold sweep curves"),
            ("upstream/par2qo/code/carver/5_visualize_workload.py", "par2qo_visualize_workload.py",
             "workload visualization: per-template latency box plots, workload drift tracking"),
        ],
    },
    {
        "id": "W4-M190-M192",
        "milestone": "M190-M192",
        "files_to_port": [
            ("upstream/par2qo/code/carver/6_metadata.py", "par2qo_metadata.py",
             "metadata extraction: query template statistics, plan space cardinality"),
            ("upstream/par2qo/code/carver/6_metadata_PQO.py", "par2qo_metadata_pqo.py",
             "PQO metadata: parametric plan fingerprinting, selectivity distribution summary"),
            ("upstream/par2qo/code/carver/6_metadata_workload.py", "par2qo_metadata_workload.py",
             "workload metadata: workload fingerprint, query arrival rate estimation, EMA smoothing"),
        ],
    },
]


def build_prompt(worker):
    """构建发给子模型的prompt"""
    files_desc = "\n".join([
        f"  {i+1}. {src} → lynceus/integrations/{dst}\n     说明: {desc}"
        for i, (src, dst, desc) in enumerate(worker["files_to_port"])
    ])

    return f"""你是第一位Claude的子模型Worker {worker['id']}。任务: 移植upstream文件到lynceus/integrations/。

## 执行步骤 (严格按顺序):

1. git clone {REPO} /home/user/lynceus
2. cd /home/user/lynceus
3. 读取以下upstream源文件 (用cat):
{files_desc}

4. 对每个文件, 在 lynceus/integrations/ 下创建对应文件:
   - 鲁迅式拿法: mv基础上动态修改算法的20%内容
   - 不是字符串/docstring替换! 是真正的算法改动:
     * psycopg2/DB调用 → 内存模拟 (dict/list)
     * TensorFlow/PyTorch → numpy实现
     * 加入 _debug_snapshot() 函数, print当前所有数据结构状态
     * 加入 __main__ 自测代码 (python -c "from lynceus.integrations.xxx import *; ...")
     * 加入 Welford在线方差/Kahan补偿求和/Shannon熵等数值改进
   - 不要加 _port/_v2/_v10 等后缀!

5. 验证: python3 -c "from lynceus.integrations.{dst_name} import *" 无报错

6. git add + commit:
   git config user.name "dylanyunlon"
   git config user.email "dogechat@163.com"
   git remote set-url origin https://{GIT_TOKEN}@github.com/dylanyunlon/lynceus-CMD.git
   git add lynceus/integrations/
   git commit -m "{worker['milestone']}: {', '.join(dst for _, dst, _ in worker['files_to_port'])}" --author="dylanyunlon <dogechat@163.com>"
   git push origin main

7. 输出每个文件的行数和关键算法改动列表。
"""


def dispatch_worker(worker, model="claude-opus-4-6"):
    """通过claude_hk_chat.sh发送任务"""
    prompt = build_prompt(worker)

    print(f"\n{'='*60}")
    print(f"DISPATCHING {worker['id']} ({worker['milestone']})")
    print(f"{'='*60}")
    print(f"Files: {[dst for _, dst, _ in worker['files_to_port']]}")
    print(f"Model: {model}")

    # 写入临时prompt文件
    prompt_file = f"/tmp/dispatch_{worker['id']}.txt"
    with open(prompt_file, "w") as f:
        f.write(prompt)

    # 通过claude_hk_chat.sh发送
    env = os.environ.copy()
    env["MODEL"] = model
    env["EFFORT"] = "high"
    env["TIMEOUT"] = "600"

    try:
        result = subprocess.run(
            ["bash", "claude_hk_chat.sh"],
            input=prompt,
            capture_output=True, text=True, timeout=660,
            cwd="/home/claude/lynceus-CMD",
            env=env,
        )
        print(f"  stdout (last 500 chars): {result.stdout[-500:]}")
        if result.returncode != 0:
            print(f"  stderr: {result.stderr[-300:]}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (>660s)")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    print("Session 12: Dispatching Workers")
    print(f"Total workers: {len(WORKERS)}")
    print(f"Total files to port: {sum(len(w['files_to_port']) for w in WORKERS)}")

    results = {}
    for w in WORKERS:
        ok = dispatch_worker(w)
        results[w["id"]] = "OK" if ok else "FAIL"
        time.sleep(5)  # 避免rate limit

    print(f"\n{'='*60}")
    print("DISPATCH SUMMARY")
    for wid, status in results.items():
        print(f"  {wid}: {status}")


if __name__ == "__main__":
    main()
