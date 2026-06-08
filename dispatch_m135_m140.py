#!/usr/bin/env python3
"""
dispatch_m135_m140.py — M135-M140 里程碑: training_data_collection_pipeline 移植
向 Opus 4.6 子模型发送6个移植任务批次, 收集代码 → 验证 → 实验

本轮移植范围:
  M135: kepler_plan_fingerprint_candidates.py      ← pg_generate_plan_candidates + pg_perturb_plan_cardinalities
  M136: kepler_training_execution.py   ← pg_execute_training_data_queries + pg_execute_explain_tools
  M137: kepler_hint_extractor.py       ← pg_plan_hint_extractor + query_text_utils + query_plan_utils
  M138: kepler_param_gen_pipeline.py   ← parameter_generator + param_gen_new + param_PQO_files_generate
  M139: kepler_verify_robustness.py    ← verify_robustness + verify_robustness_category + verify_robustness_random
  M140: kepler_evaluate_visualize.py   ← evaluate + evaluate_both + evaluate_cost + evaluate_pqo + end_visualize*

每个文件: 算法20%改动 + _dbg()断点调试 + 纯numpy无TF
"""

import json, uuid, sys, os, time, re, requests, textwrap

ORG = "6bbaaedb-4337-470e-8353-6f208e788b73"
BASE = f"https://claude.hk.cn/api/organizations/{ORG}"
MODEL = "claude-opus-4-6"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"

with open("/tmp/claude_hk_cookie.txt") as f:
    COOKIES = f.read().strip()

PROJECT = "/home/claude/lynceus-CMD"
INTEGRATIONS = f"{PROJECT}/lynceus/integrations"

def cookie_dict():
    d = {}
    for part in COOKIES.split("; "):
        if "=" in part:
            k, v = part.split("=", 1)
            d[k] = v
    return d

def new_conversation():
    r = requests.post(
        f"{BASE}/chat_conversations",
        json={"name": "", "project_uuid": None, "model": None},
        headers={"content-type": "application/json", "origin": "https://claude.hk.cn", "user-agent": UA},
        cookies=cookie_dict(),
        timeout=15,
    )
    cid = r.json()["uuid"]
    print(f"[dispatch] 新对话: {cid}")
    return cid

def send_prompt(conv_id, prompt, timeout=600):
    h_uuid = str(uuid.uuid4())
    a_uuid = str(uuid.uuid4())
    payload = {
        "prompt": prompt,
        "timezone": "Asia/Shanghai",
        "personalized_styles": [{"type":"default","key":"Default","name":"Normal",
            "nameKey":"normal_style_name","prompt":"Normal\n",
            "summary":"Default responses from Claude",
            "summaryKey":"normal_style_summary","isDefault":True}],
        "locale": "en-US", "model": MODEL, "effort": "high",
        "thinking_mode": "off",
        "tools": [],
        "turn_message_uuids": {"human_message_uuid": h_uuid, "assistant_message_uuid": a_uuid},
        "attachments":[],"files":[],"sync_sources":[],"rendering_mode":"messages",
        "create_conversation_params":{"name":"","model":MODEL,
            "include_conversation_preferences":True,"paprika_mode":None,"compass_mode":None,
            "tool_search_mode":"auto","is_temporary":False,"enabled_imagine":False}
    }
    
    print(f"[dispatch] 发送prompt ({len(prompt)} chars)...")
    t0 = time.time()
    r = requests.post(
        f"{BASE}/chat_conversations/{conv_id}/completion",
        json=payload,
        headers={
            "accept": "text/event-stream",
            "content-type": "application/json",
            "anthropic-client-platform": "web_claude_ai",
            "origin": "https://claude.hk.cn",
            "referer": "https://claude.hk.cn/new",
            "user-agent": UA,
        },
        cookies=cookie_dict(),
        timeout=timeout,
        stream=True,
    )
    
    text_parts = []
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        try:
            d = json.loads(line[6:])
        except:
            continue
        t = d.get("type", "")
        if t == "content_block_delta":
            delta = d.get("delta", {})
            if delta.get("type") == "text_delta":
                text_parts.append(delta["text"])
    
    full = "".join(text_parts)
    elapsed = time.time() - t0
    print(f"[dispatch] 回复: {len(full)} chars, {elapsed:.1f}s")
    return full

def extract_python_blocks(text):
    """提取 ```python ... ``` 代码块, 按 # FILE: xxx 标记分割"""
    blocks = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
    files = {}
    for block in blocks:
        m = re.match(r'#\s*FILE:\s*(\S+)', block)
        if m:
            fname = m.group(1)
            code = block[m.end():].lstrip('\n')
            files[fname] = code
        else:
            # 尝试从模块docstring推断
            m2 = re.search(r'"""[^"]*?(kepler_\w+)', block)
            if m2:
                fname = m2.group(1) + ".py"
            else:
                fname = f"block_{len(files)}.py"
            files[fname] = block
    return files

def read_upstream(path, max_chars=3200):
    full = os.path.join(PROJECT, path)
    if not os.path.exists(full):
        return f"# File not found: {path}"
    with open(full) as f:
        content = f.read()
    if len(content) > max_chars:
        content = content[:max_chars] + f"\n# ... truncated (total {content.count(chr(10))} lines)\n"
    return content

def read_example_style():
    """读取已有的移植样例供子模型参考"""
    ex = os.path.join(INTEGRATIONS, "kepler_parameter_generator.py")
    if os.path.exists(ex):
        with open(ex) as f:
            return f.read()[:2500]
    return "# No example available"

def build_batch_prompt(milestone_label, output_files, upstream_map, extra_instructions=""):
    """构建发给子模型的移植prompt"""
    
    example = read_example_style()
    
    prompt = f"""你是 lynceus-CMD 项目的子模型(Opus 4.6), 负责 {milestone_label} 的代码移植。

这是第一位Claude(指挥官)给你的prompt原文背景:
"看看这个项目(upstream文件夹)关于代码移植的问题,我们需要每一个文件的每一行都用上。
鲁迅那样的拿法,别全权复制啊。我的意思是在mv的基础上,动态修改算法的20%的内容就行了。
注意多写一点关于断点调试(或者print当前所有数据、结构体状态)的内容,
让我们在运行实验的时候能像现实世界开发一样得到反馈。"

## 移植规则（铁律）
1. 算法改20%: 替换psycopg2/absl依赖为纯Python模拟; 把multiprocessing.Pool替换为ThreadPoolExecutor;
   把np.random.seed改为可复现的RandomState; 加入EMA平滑、Welford在线统计、自适应阈值等
2. 不改字符串/docstring: 改的是真正的算法逻辑
3. 断点调试: 每个函数都加 `_dbg(tag, **kw)` 打印所有输入参数和数据结构状态
4. 文件名: 不加v2/port后缀
5. 无外部DB依赖: 用内存dict模拟psycopg2连接, 用dict模拟SQL查询结果
6. 所有文件都必须能 `python3 -c "import xxx"` 无报错
7. 每个文件200-500行

## 已有移植风格样例
```python
{example}
```

## 需要移植的upstream源文件
"""
    for label, path in upstream_map.items():
        content = read_upstream(path)
        prompt += f"\n### {label} ({path})\n```python\n{content}\n```\n"
    
    prompt += f"""
## 你需要输出以下文件（每个文件用 ```python 代码块, 第一行必须写 # FILE: 文件名）
"""
    for fname, desc in output_files.items():
        prompt += f"\n**# FILE: {fname}** — {desc}"
    
    prompt += f"""

{extra_instructions}

## 额外要求
- debug模式: 全局 `_DEBUG = False`, `enable_debug(True)` 打开
- 每个类都要有 `__repr__` 方法显示关键状态
- 在模块末尾加一个 `if __name__ == "__main__":` 的自测demo
- 自测demo要能print出结构体状态, 跑通核心逻辑路径
- 确保每个upstream文件的每一行逻辑都体现在移植代码中

开始输出代码:"""
    
    return prompt

def dispatch_batch(milestone, output_files, upstream_map, extra=""):
    """执行一个批次的dispatch"""
    prompt = build_batch_prompt(milestone, output_files, upstream_map, extra)
    
    conv_id = new_conversation()
    text = send_prompt(conv_id, prompt, timeout=600)
    
    # 提取代码块
    files = extract_python_blocks(text)
    
    # 如果没提取到足够文件, 发Continue
    if len(files) < len(output_files):
        print(f"[dispatch] 只得到{len(files)}/{len(output_files)}文件, 发Continue...")
        cont = send_prompt(conv_id, "Continue — 继续输出剩余的文件代码", timeout=600)
        more = extract_python_blocks(cont)
        files.update(more)
    
    # 保存到opus_output
    out_dir = os.path.join(PROJECT, "opus_output", milestone.replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)
    
    # 保存完整回复
    with open(os.path.join(out_dir, "response.md"), "w") as f:
        f.write(text)
    
    saved = {}
    for fname, code in files.items():
        path = os.path.join(out_dir, os.path.basename(fname))
        with open(path, "w") as f:
            f.write(code)
        saved[fname] = path
        print(f"[dispatch] 保存: {fname} ({len(code)} chars)")
    
    return saved, conv_id

def verify_syntax(filepath):
    """验证Python语法"""
    import py_compile
    try:
        py_compile.compile(filepath, doraise=True)
        return True, ""
    except py_compile.PyCompileError as e:
        return False, str(e)

def verify_import(module_name, filepath):
    """验证import"""
    import subprocess
    dirname = os.path.dirname(filepath)
    r = subprocess.run(
        ["python3", "-c", f"import sys; sys.path.insert(0, '{dirname}'); import {module_name}"],
        capture_output=True, text=True, timeout=30
    )
    return r.returncode == 0, r.stderr[:500]

def main():
    print("=" * 60)
    print("  M135-M140 Dispatch to Opus 4.6")
    print("  training_data_collection_pipeline 移植")
    print("=" * 60)
    
    # 定义6个批次
    batches = [
        {
            "milestone": "M135",
            "output_files": {
                "kepler_plan_fingerprint_candidates.py": 
                    "查询计划候选生成: powerset配置组合 + 计划去重 + 基数扰动. "
                    "用ThreadPoolExecutor替代multiprocessing, 加入Jaccard plan similarity去重, "
                    "基数扰动用log-normal噪声替代均匀分布",
            },
            "upstream_map": {
                "plan_candidates": "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/pg_generate_plan_candidates.py",
                "perturb": "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/pg_perturb_plan_cardinalities.py",
            },
        },
        {
            "milestone": "M136",
            "output_files": {
                "kepler_training_execution.py": 
                    "训练数据执行引擎: 参数绑定 × 查询计划的全组合执行 + 延迟收集 + near-optimal判定. "
                    "用内存模拟替代psycopg2, 延迟用合成log-normal分布, "
                    "near-optimal用Huber阈值替代固定1.01x",
            },
            "upstream_map": {
                "execute_queries": "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/pg_execute_training_data_queries.py",
                "explain_tools": "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/pg_execute_explain_tools.py",
                "main_utils": "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/main_utils.py",
            },
        },
        {
            "milestone": "M137",
            "output_files": {
                "kepler_hint_extractor.py": 
                    "PG plan hint提取器: 从EXPLAIN JSON提取join/scan hints + 基数修正 + query text规范化. "
                    "hint匹配用Levenshtein distance做模糊匹配, 基数修正加Bayesian shrinkage",
            },
            "upstream_map": {
                "hint_extractor": "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/pg_plan_hint_extractor.py",
                "text_utils": "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/query_text_utils.py",
                "plan_utils": "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/query_plan_utils.py",
                "query_utils": "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/query_utils.py",
            },
        },
        {
            "milestone": "M138",
            "output_files": {
                "kepler_param_gen_pipeline.py": 
                    "参数生成pipeline: 从DB schema提取列值分布 + 生成参数绑定 + PQO文件输出. "
                    "参数采样用Sobol quasi-random替代纯随机, 日期参数用beta分布集中在常见范围",
            },
            "upstream_map": {
                "param_gen": "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/parameter_generator.py",
                "param_new": "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/param_gen_new.py",
                "pqo_files": "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/param_PQO_files_generate.py",
                "gen_metadata": "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/generate_training_metadata.py",
            },
        },
        {
            "milestone": "M139",
            "output_files": {
                "kepler_verify_robustness.py": 
                    "鲁棒性验证: 跨DB变体的计划稳定性检测 + 分类鲁棒性 + 随机鲁棒性. "
                    "稳定性度量用Kendall tau rank correlation替代简单计数, "
                    "加入bootstrap置信区间",
            },
            "upstream_map": {
                "verify_rob": "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/verify_robustness.py",
                "verify_cat": "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/verify_robustness_category.py",
                "verify_rand": "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/verify_robustness_random.py",
            },
        },
        {
            "milestone": "M140",
            "output_files": {
                "kepler_evaluate_visualize.py": 
                    "评估与可视化pipeline: 多策略评估 + cost比较 + PQO评估 + 结果可视化. "
                    "评估指标加入NDCG和MRR rank metrics, 可视化用ASCII art替代matplotlib",
            },
            "upstream_map": {
                "evaluate": "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/evaluate.py",
                "eval_both": "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/evaluate_both.py",
                "eval_cost": "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/evaluate_cost.py",
                "eval_pqo": "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/evaluate_pqo.py",
                "end_viz": "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/end_visualize.py",
                "end_viz_cost": "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/end_visualize_cost.py",
                "end_viz_pqo": "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/end_visualize_pqo.py",
                "verify_viz": "upstream/par2qo/code/carver/kepler/training_data_collection_pipeline/verify_visualize.py",
            },
        },
    ]
    
    all_saved = {}
    all_convs = {}
    
    for batch in batches:
        ms = batch["milestone"]
        print(f"\n{'='*60}")
        print(f"  Dispatching {ms}")
        print(f"{'='*60}")
        
        try:
            saved, conv = dispatch_batch(ms, batch["output_files"], batch["upstream_map"])
            all_saved[ms] = saved
            all_convs[ms] = conv
            
            # 验证每个文件
            for fname, fpath in saved.items():
                ok, err = verify_syntax(fpath)
                status = "✓" if ok else "✗"
                print(f"  [{status}] syntax: {fname}")
                if not ok:
                    print(f"      Error: {err[:200]}")
                else:
                    mod = os.path.splitext(os.path.basename(fname))[0]
                    ok2, err2 = verify_import(mod, fpath)
                    status2 = "✓" if ok2 else "✗"
                    print(f"  [{status2}] import: {mod}")
                    if not ok2:
                        print(f"      Error: {err2[:200]}")
            
            # 间隔避免rate limit
            print(f"[dispatch] {ms} 完成, 等待5秒...")
            time.sleep(5)
            
        except Exception as e:
            print(f"[ERROR] {ms} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 汇总
    print(f"\n{'='*60}")
    print(f"  汇总")
    print(f"{'='*60}")
    for ms, saved in all_saved.items():
        print(f"  {ms}: {len(saved)} files")
        for fname in saved:
            print(f"    - {fname}")
    
    print(f"\nConversation IDs:")
    for ms, cid in all_convs.items():
        print(f"  {ms}: {cid}")

if __name__ == "__main__":
    main()
