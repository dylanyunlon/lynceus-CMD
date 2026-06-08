#!/usr/bin/env python3
"""
dispatch_loop.py — 指挥官Claude #1 Session 10
向Opus 4.6子模型循环分发移植任务
每次dispatch: clone仓库 → 读取upstream → 输出移植代码
"""
import json, uuid, sys, os, time, re, requests, subprocess

# ── 配置 ──
COOKIE_FILE = "/tmp/claude_hk_cookie.txt"
CONFIG_REPO = "/tmp/claude-hk-config"
PROJECT = "/home/claude/lynceus-CMD"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"

def sync_cookie():
    """从claude-hk-config同步最新cookie"""
    subprocess.run(["git", "-C", CONFIG_REPO, "pull", "-q"], capture_output=True)
    raw = open(f"{CONFIG_REPO}/raw_curl.txt").read()
    import re
    m = re.search(r"-b '([^']+)'", raw)
    cookie = m.group(1) if m else ""
    open(COOKIE_FILE, "w").write(cookie)
    org = re.search(r"organizations/([a-f0-9-]+)", raw).group(1)
    return cookie, org

def cookie_dict(cookie_str):
    d = {}
    for part in cookie_str.split("; "):
        if "=" in part:
            k, v = part.split("=", 1)
            d[k] = v
    return d

def new_conv(org, cookies):
    r = requests.post(
        f"https://claude.hk.cn/api/organizations/{org}/chat_conversations",
        json={"name":"","project_uuid":None,"model":None},
        headers={"content-type":"application/json","origin":"https://claude.hk.cn","user-agent":UA},
        cookies=cookie_dict(cookies), timeout=15)
    return r.json()["uuid"]

def send(org, conv_id, cookies, prompt, model="claude-opus-4-6", timeout=600):
    h = str(uuid.uuid4()); a = str(uuid.uuid4())
    payload = {
        "prompt": prompt, "timezone":"Asia/Shanghai",
        "personalized_styles":[{"type":"default","key":"Default","name":"Normal",
            "nameKey":"normal_style_name","prompt":"Normal\n",
            "summary":"Default responses from Claude",
            "summaryKey":"normal_style_summary","isDefault":True}],
        "locale":"en-US","model":model,"effort":"high","thinking_mode":"off",
        "tools":[],
        "turn_message_uuids":{"human_message_uuid":h,"assistant_message_uuid":a},
        "attachments":[],"files":[],"sync_sources":[],"rendering_mode":"messages",
        "create_conversation_params":{"name":"","model":model,
            "include_conversation_preferences":True,"paprika_mode":None,
            "compass_mode":None,"tool_search_mode":"auto",
            "is_temporary":False,"enabled_imagine":False}
    }
    t0 = time.time()
    r = requests.post(
        f"https://claude.hk.cn/api/organizations/{org}/chat_conversations/{conv_id}/completion",
        json=payload,
        headers={"accept":"text/event-stream","content-type":"application/json",
                 "anthropic-client-platform":"web_claude_ai",
                 "origin":"https://claude.hk.cn","referer":"https://claude.hk.cn/new",
                 "user-agent":UA},
        cookies=cookie_dict(cookies), timeout=timeout, stream=True)
    
    parts = []
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "): continue
        try: d = json.loads(line[6:])
        except: continue
        if d.get("type") == "content_block_delta":
            delta = d.get("delta", {})
            if delta.get("type") == "text_delta":
                parts.append(delta["text"])
    
    text = "".join(parts)
    print(f"  回复: {len(text)} chars, {time.time()-t0:.0f}s")
    return text

def extract_files(text):
    """提取```python ... ```代码块"""
    blocks = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
    files = {}
    for block in blocks:
        m = re.match(r'#\s*FILE:\s*(\S+)', block)
        if m:
            files[m.group(1)] = block[m.end():].lstrip('\n')
        else:
            files[f"block_{len(files)}.py"] = block
    return files

def verify(filepath):
    """验证语法+import"""
    import py_compile
    try:
        py_compile.compile(filepath, doraise=True)
    except py_compile.PyCompileError as e:
        return False, f"syntax: {e}"
    
    mod = os.path.splitext(os.path.basename(filepath))[0]
    r = subprocess.run(
        ["python3", "-c", f"import sys; sys.path.insert(0,'{os.path.dirname(filepath)}'); import {mod}"],
        capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return False, f"import: {r.stderr[:300]}"
    return True, "OK"

# ── 任务定义 ──
BATCHES = [
    {
        "id": "W1", "milestone": "M172-M173",
        "files": {
            "kepler_plan_fingerprint_candidates.py": "查询计划候选生成+基数扰动",
            "kepler_training_execution.py": "训练数据执行引擎+延迟收集",
        },
        "upstream": [
            "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/pg_generate_plan_candidates.py",
            "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/pg_perturb_plan_cardinalities.py",
            "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/pg_execute_training_data_queries.py",
            "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/pg_execute_explain_tools.py",
        ],
    },
    {
        "id": "W2", "milestone": "M174-M175",
        "files": {
            "kepler_hint_extractor.py": "PG plan hint提取+query text规范化",
            "kepler_param_gen_pipeline.py": "参数生成pipeline+PQO文件输出",
        },
        "upstream": [
            "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/pg_plan_hint_extractor.py",
            "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/query_text_utils.py",
            "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/query_utils.py",
            "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/parameter_generator.py",
            "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/param_gen_new.py",
        ],
    },
    {
        "id": "W3", "milestone": "M176-M177",
        "files": {
            "kepler_verify_robustness.py": "鲁棒性验证(跨DB+分类+随机)",
            "kepler_evaluate_visualize.py": "评估+可视化pipeline",
        },
        "upstream": [
            "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/verify_robustness.py",
            "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/verify_robustness_category.py",
            "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/verify_robustness_random.py",
            "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/evaluate.py",
            "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/evaluate_both.py",
            "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/end_visualize.py",
        ],
    },
    {
        "id": "W4", "milestone": "M178-M179",
        "files": {
            "kepler_sngp_model.py": "SNGP多头模型(Spectral-normalized GP)",
            "kepler_mlp_server.py": "模型服务器+查询解析",
        },
        "upstream": [
            "upstream/par2qo/code/carver/kepler/model_trainer/sngp_multihead_model.py",
            "upstream/par2qo/code/carver/kepler/database_integrations/model_serving/model_server.py",
            "upstream/par2qo/code/carver/kepler/database_integrations/model_serving/model_server_main.py",
            "upstream/par2qo/code/carver/kepler/database_integrations/model_serving/query_parsing_utils.py",
        ],
    },
    {
        "id": "W5", "milestone": "M180-M181",
        "files": {
            "videx_model_innodb.py": "InnoDB虚拟索引模型+策略选择",
            "videx_analyze_tools.py": "分析工具(delete_rows+linear分布+trace)",
        },
        "upstream": [
            "upstream/videx/src/sub_platforms/sql_opt/videx/model/videx_model_innodb.py",
            "upstream/videx/src/sub_platforms/sql_opt/videx/model/videx_strategy.py",
            "upstream/videx/src/sub_platforms/sql_opt/videx/model/videx_model_example.py",
            "upstream/videx/src/sub_platforms/sql_opt/videx/scripts/analyze/analyze_delete_rows.py",
            "upstream/videx/src/sub_platforms/sql_opt/videx/scripts/analyze/analyze_linear_distribution.py",
            "upstream/videx/src/sub_platforms/sql_opt/videx/scripts/analyze/analyze_trace_utils.py",
        ],
    },
    {
        "id": "W6", "milestone": "M182-M183",
        "files": {
            "kepler_carver_pipeline.py": "carver编号脚本统一pipeline(0_到10_)",
            "par2qo_prep_query_template.py": "查询模板准备(2402行大文件)",
        },
        "upstream": [
            "upstream/par2qo/code/carver/0_generate_parameter.py",
            "upstream/par2qo/code/carver/1_generate_plan_candidates.py",
            "upstream/par2qo/code/carver/2_execute_training_data.py",
            "upstream/par2qo/code/carver/4_evaluate.py",
            "upstream/par2qo/code/carver/4_evaluate_both.py",
            "upstream/par2qo/code/carver/5_visualize.py",
            "upstream/par2qo/code/carver/6_metadata.py",
            "upstream/par2qo/code/carver/7_best_performance.py",
            "upstream/par2qo/code/prep_query_template.py",
        ],
    },
]

def build_prompt(batch):
    """构建精简prompt: clone指令 + 第一轮prompt原文 + upstream文件路径"""
    
    # 读取upstream文件内容(控制长度)
    upstream_content = ""
    for path in batch["upstream"]:
        full = os.path.join(PROJECT, path)
        if os.path.exists(full):
            with open(full) as f:
                content = f.read()
            if len(content) > 4000:
                content = content[:4000] + f"\n# ... truncated ({content.count(chr(10))} lines total)\n"
            upstream_content += f"\n### {path}\n```python\n{content}\n```\n"
    
    # 读取样例
    example_file = os.path.join(PROJECT, "lynceus/integrations/kepler_parameter_generator.py")
    example = ""
    if os.path.exists(example_file):
        with open(example_file) as f:
            example = f.read()[:2000]
    
    output_spec = ""
    for fname, desc in batch["files"].items():
        output_spec += f"\n**# FILE: {fname}** — {desc}"

    prompt = f"""你是子模型Opus 4.6, 负责 {batch['milestone']} 移植任务。

## 第一轮prompt (来自用户):
"看看这个项目(upstream文件夹)关于代码移植的问题,我们需要每一个文件的每一行都用上。
鲁迅那样的拿法,别全权复制啊。我的意思是在mv的基础上,动态修改算法的20%的内容就行了。
注意多写一点关于断点调试(或者print当前所有数据、结构体状态)的内容,
让我们在运行实验的时候能像现实世界开发一样得到反馈。注意不是让你训练模型。"

## 仓库: git clone https://github.com/dylanyunlon/lynceus-CMD.git
## Cookie同步: git clone https://github.com/dylanyunlon/claude-hk-config.git

## 移植铁律
1. 算法改20%: 替换psycopg2/absl/TF为纯Python; multiprocessing→ThreadPoolExecutor; np.random.seed→RandomState; 加EMA/Welford/自适应阈值
2. 改的是算法,不是字符串/docstring/str_replace
3. 每个函数加 `_dbg(tag, **kw)` 断点调试
4. 文件名无v2/port后缀
5. 无外部DB依赖, 用内存dict模拟
6. 能 python3 -c "import xxx" 无报错
7. 每文件200-500行, 末尾加 if __name__=="__main__": 自测

## 已有风格样例
```python
{example}
```

## upstream源文件
{upstream_content}

## 输出文件 (每个```python块第一行写 # FILE: 文件名)
{output_spec}

开始输出代码:"""
    return prompt


def run_batch(batch):
    """执行一个batch的dispatch"""
    print(f"\n{'='*50}")
    print(f"  Worker {batch['id']}: {batch['milestone']}")
    print(f"{'='*50}")
    
    cookies, org = sync_cookie()
    conv_id = new_conv(org, cookies)
    print(f"  对话: {conv_id}")
    
    prompt = build_prompt(batch)
    print(f"  Prompt: {len(prompt)} chars")
    
    text = send(org, conv_id, cookies, prompt)
    files = extract_files(text)
    
    # Continue if truncated
    if len(files) < len(batch["files"]):
        print(f"  只得到{len(files)}/{len(batch['files'])}文件, 发Continue...")
        time.sleep(3)
        cont = send(org, conv_id, cookies, "Continue — 继续输出剩余的代码文件")
        more = extract_files(cont)
        files.update(more)
        text += "\n\n---CONTINUE---\n\n" + cont
    
    # 保存
    out_dir = os.path.join(PROJECT, "opus_output", batch["id"])
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "response.md"), "w").write(text)
    
    results = {}
    for fname, code in files.items():
        fpath = os.path.join(out_dir, os.path.basename(fname))
        open(fpath, "w").write(code)
        ok, msg = verify(fpath)
        status = "✓" if ok else "✗"
        print(f"  [{status}] {fname} ({len(code)} chars) — {msg}")
        results[fname] = {"path": fpath, "ok": ok, "msg": msg, "chars": len(code)}
    
    return results, conv_id

def main():
    print("=" * 60)
    print("  Claude #1 指挥官 — Session 10 Dispatch Loop")
    print("  M172-M183 via Opus 4.6 Workers")
    print("=" * 60)
    
    all_results = {}
    for batch in BATCHES:
        try:
            results, conv = run_batch(batch)
            all_results[batch["id"]] = {"milestone": batch["milestone"], "files": results, "conv": conv}
            print(f"  {batch['id']} 完成, 等待8秒避免rate limit...")
            time.sleep(8)
        except Exception as e:
            print(f"  [ERROR] {batch['id']}: {e}")
            import traceback; traceback.print_exc()
            time.sleep(10)
    
    # 汇总
    print(f"\n{'='*60}")
    print("  DISPATCH SUMMARY")
    print(f"{'='*60}")
    total_ok = 0
    total_files = 0
    for wid, info in all_results.items():
        print(f"\n  {wid} ({info['milestone']}):")
        for fname, finfo in info["files"].items():
            s = "✓" if finfo["ok"] else "✗"
            print(f"    [{s}] {fname} — {finfo['chars']} chars")
            total_files += 1
            if finfo["ok"]: total_ok += 1
    
    print(f"\n  总计: {total_ok}/{total_files} 文件验证通过")
    
    # 输出JSON summary
    summary_path = os.path.join(PROJECT, "opus_output", "dispatch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Summary: {summary_path}")

if __name__ == "__main__":
    main()
