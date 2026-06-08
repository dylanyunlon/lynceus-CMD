#!/usr/bin/env python3
"""
dispatch_session16.py — Session 16 指挥官调度脚本
Claude #1 (当前) 完成 M199-M204, 然后通过 claude_hk_chat.sh 分配子任务

调度协议:
  1. git pull claude-hk-config → 提取cookie
  2. 创建新对话 → 发送附件prompt → 收集回复
  3. 提取代码块 → 验证 → git commit + push
  4. 如果truncated → 发送 "Continue"

里程碑分配:
  Claude #1 (指挥官, 当前session): M199-M210 (SOTA实验 + 剩余kepler pipeline)
  Claude #2 (子模型): M211-M225 (par2qo carver高级系列)
  Claude #3 (子模型): M226-M240 (videx histogram + ndv全覆盖)
  Claude #4 (子模型): M241-M250 (E2E pipeline + benchmark对比)
  Claude #5 (子模型): M251-M255 (性能优化 + tex数据填充)
  Claude #6 (子模型): M256-M260 (最终验证 + 论文提交)
"""
import json
import os
import subprocess
import sys
import time
import uuid

# ── Config ──
CONFIG_REPO = "https://github.com/dylanyunlon/claude-hk-config.git"
CONFIG_DIR = "/tmp/claude-hk-config"
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = "claude-opus-4-6"
EFFORT = "medium"
TIMEOUT = 300

def sync_config():
    """拉取最新config"""
    if os.path.exists(CONFIG_DIR):
        subprocess.run(["git", "-C", CONFIG_DIR, "pull", "-q"],
                      timeout=15, capture_output=True)
    else:
        subprocess.run(["git", "clone", "--depth=1", "-q",
                       CONFIG_REPO, CONFIG_DIR],
                      timeout=20, capture_output=True)

def get_cookie():
    """从config提取cookie"""
    # Check cache first
    cache = "/tmp/claude_hk_cookie.txt"
    if os.path.exists(cache):
        with open(cache) as f:
            c = f.read().strip()
            if c: return c

    raw_curl = os.path.join(CONFIG_DIR, "raw_curl.txt")
    if not os.path.exists(raw_curl):
        raise RuntimeError("No raw_curl.txt found")

    import re
    with open(raw_curl) as f:
        text = f.read()
    m = re.search(r"-b '([^']+)'", text)
    if not m:
        raise RuntimeError("No cookie in raw_curl.txt")
    cookie = m.group(1)
    with open(cache, 'w') as f:
        f.write(cookie)
    return cookie

def get_org():
    """从config提取org_id"""
    raw_curl = os.path.join(CONFIG_DIR, "raw_curl.txt")
    import re
    with open(raw_curl) as f:
        text = f.read()
    m = re.search(r'organizations/([a-f0-9-]+)', text)
    return m.group(1) if m else None

def build_prompt_for_worker(worker_id, milestones, scope):
    """构建给子claude的prompt, 包含第一轮上下文"""
    return f"""你是Lynceus-CMD项目的第{worker_id}位子Claude开发者。

任务: 完成 {milestones} — {scope}

步骤:
1. git clone https://github.com/dylanyunlon/lynceus-CMD.git && cd lynceus-CMD
2. apt install -y tree && tree -L 2 lynceus/
3. cat RELAY_STATUS.md | head -80  # 了解当前进度
4. 查看 upstream/ 目录中对应的源文件

开发规则:
- 改动算法的20%内容, 不是全权复制
- 文件名不允许v10、port等后缀, 直接放在 lynceus/integrations/ 下
- 每个文件必须加断点调试print (当前数据/结构体状态)
- 用 python3 -c "import lynceus.integrations.xxx" 验证语法
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git remote set-url origin https://$GH_PAT@github.com/dylanyunlon/lynceus-CMD.git
- 完成后 git add/commit/push origin main

具体文件分配 ({milestones}):
{scope}

注意: 如果回复被截断, 发送者会发 "Continue" 让你继续。
"""

# ── Worker definitions ──
WORKERS = [
    {
        'id': 2,
        'milestones': 'M211-M225',
        'scope': '''par2qo carver高级系列:
- M211-M212: 8_verify_robustness_by_category.py + 8_verify_robustness_random.py → par2qo_robustness.py扩展
- M213-M214: 99_check_all_join_predicate_exist.py + 99_plan_content.py → par2qo_plan_inspection.py扩展
- M215-M216: 99_testing_query_valid.py + 99_testing_query_valid_robustness.py → par2qo_query_parser.py扩展
- M217-M218: db_robustness_setup/db_category.py + db_random.py → par2qo_db_random_sampler.py扩展
- M219-M220: db_robustness_setup/db_sliding.py + utility.py → par2qo_utility.py扩展
- M221-M222: db_robustness_setup/postgres.py + psql_explain_decoder.py → par2qo_explain_decoder.py扩展
- M223-M225: 0_query_metadata_analysis.py + 6_metadata*.py → par2qo_generate_training_metadata.py扩展'''
    },
    {
        'id': 3,
        'milestones': 'M226-M240',
        'scope': '''videx histogram + NDV全覆盖:
- M226-M228: videx_histogram.py(64K) → videx_histogram.py扩展 (分3轮)
- M229-M230: videx_metadata.py(57K) → videx_metadata_extended.py扩展
- M231-M232: videx_utils.py(36K) → videx_primitives.py扩展
- M233-M234: videx_service.py(31K) → videx_strategy.py扩展
- M235-M236: ndv_estimator.py(30K) → videx_ndv_estimator.py扩展
- M237-M238: histogram_utils.py(22K) → videx_histogram_builder.py扩展
- M239-M240: videx_mysql_utils.py + videx_logging.py → videx_mysql.py + videx_logging_telemetry.py扩展'''
    },
    {
        'id': 4,
        'milestones': 'M241-M250',
        'scope': '''E2E pipeline + benchmark:
- M241-M243: kepler e2e test + trainer_test → tests/test_e2e_pipeline.py扩展
- M244-M246: tabular config.py + benchmark perf → tabular_featurizer.py扩展
- M247-M248: test_records_in_range.py + test_videx_utils.py → tests增强
- M249-M250: scripts/run_full_experiment.py增强 — 自动push结果到git'''
    },
]

if __name__ == '__main__':
    sync_config()
    cookie = get_cookie()
    org = get_org()
    print(f"ORG: {org}")
    print(f"Cookie: {cookie[:20]}...")
    print(f"Model: {MODEL}")
    print(f"\nWorker plan:")
    for w in WORKERS:
        print(f"  Claude #{w['id']}: {w['milestones']} — {w['scope'][:60]}...")
    print(f"\nTo dispatch manually:")
    print(f"  MODEL={MODEL} EFFORT={EFFORT} bash claude_hk_chat.sh")
    print(f"  Then paste the prompt for each worker")
