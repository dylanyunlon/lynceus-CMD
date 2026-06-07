#!/usr/bin/env python3
"""
dispatch_to_opus.py — 向 claude.hk.cn Opus 4.6 子模型分发移植任务
非交互式：发送prompt → 收集回复 → 提取代码块 → 写入文件
支持 Continue 续传
"""
import json, uuid, sys, os, time, re, requests

ORG = "518a2313-4665-47d1-bad5-803ab2700f7c"
BASE = f"https://claude.hk.cn/api/organizations/{ORG}"
MODEL = "claude-opus-4-6"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"

with open("/tmp/claude_hk_cookie.txt") as f:
    COOKIES = f.read().strip()

def new_conversation():
    r = requests.post(
        f"{BASE}/chat_conversations",
        json={"name": "", "project_uuid": None, "model": None},
        headers={"content-type": "application/json", "origin": "https://claude.hk.cn", "user-agent": UA},
        cookies={k.split("=", 1)[0]: k.split("=", 1)[1] for k in COOKIES.split("; ") if "=" in k},
        timeout=15,
    )
    conv_id = r.json()["uuid"]
    print(f"[dispatch] 新对话: {conv_id}")
    return conv_id

def send_prompt(conv_id, prompt, effort="high", thinking="off", timeout=600):
    h_uuid = str(uuid.uuid4())
    a_uuid = str(uuid.uuid4())
    payload = {
        "prompt": prompt,
        "timezone": "Asia/Shanghai",
        "personalized_styles": [{"type": "default", "key": "Default", "name": "Normal",
            "nameKey": "normal_style_name", "prompt": "Normal\n",
            "summary": "Default responses from Claude",
            "summaryKey": "normal_style_summary", "isDefault": True}],
        "locale": "en-US", "model": MODEL, "effort": effort,
        "thinking_mode": thinking,
        "tools": [
            {"type": "web_search_v0", "name": "web_search"},
            {"type": "artifacts_v0", "name": "artifacts"},
            {"type": "repl_v0", "name": "repl"},
        ],
        "turn_message_uuids": {"human_message_uuid": h_uuid, "assistant_message_uuid": a_uuid},
        "attachments": [], "files": [], "sync_sources": [], "rendering_mode": "messages",
        "create_conversation_params": {"name": "", "model": MODEL,
            "include_conversation_preferences": True, "paprika_mode": None, "compass_mode": None,
            "tool_search_mode": "auto", "is_temporary": False, "enabled_imagine": True}
    }
    cookie_dict = {}
    for part in COOKIES.split("; "):
        if "=" in part:
            k, v = part.split("=", 1)
            cookie_dict[k] = v
    
    print(f"[dispatch] 发送prompt ({len(prompt)} chars)...")
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
        cookies=cookie_dict,
        timeout=timeout,
        stream=True,
    )
    
    text_parts = []
    tool_code_parts = []
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
                sys.stdout.write(delta["text"][-1:] if len(delta["text"]) > 0 else "")
                sys.stdout.flush()
            elif delta.get("type") == "tool_use_block_update_delta":
                dc = delta.get("display_content", {})
                if dc and dc.get("type") == "json_block":
                    try:
                        info = json.loads(dc["json_block"])
                        if info.get("code"):
                            tool_code_parts.append(info["code"])
                    except:
                        pass
        elif t == "content_block_start":
            cb = d.get("content_block", {})
            if cb.get("type") == "tool_result":
                dc = cb.get("display_content", {})
                if dc and dc.get("type") == "json_block":
                    try:
                        rr = json.loads(dc["json_block"])
                        if rr.get("stdout"):
                            print(f"\n  [tool_result] {rr['stdout'][:200]}")
                    except:
                        pass
    
    full_text = "".join(text_parts)
    print(f"\n[dispatch] 回复长度: {len(full_text)} chars, {len(tool_code_parts)} tool blocks")
    return full_text, tool_code_parts

def extract_code_blocks(text):
    """从回复中提取 ```python ... ``` 代码块，按 # FILE: xxx 分割"""
    blocks = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
    files = {}
    for block in blocks:
        # 查找 FILE: 标记
        m = re.match(r'#\s*FILE:\s*(\S+)\n', block)
        if m:
            fname = m.group(1)
            files[fname] = block[m.end():]
        else:
            # 尝试从内容推断文件名
            m2 = re.search(r'"""[^"]*?(?:kepler_\w+|par2qo_\w+|videx_\w+)', block)
            files[f"block_{len(files)}.py"] = block
    return files, blocks

def main():
    # 构造给子模型的prompt
    prompt = build_porting_prompt()
    
    conv_id = new_conversation()
    full_text, tool_blocks = send_prompt(conv_id, prompt, effort="high", timeout=600)
    
    # 提取代码块
    files, raw_blocks = extract_code_blocks(full_text)
    
    # 如果有 tool_blocks (REPL code)，也收集
    for i, code in enumerate(tool_blocks):
        if len(code) > 100:
            files[f"repl_block_{i}.py"] = code
    
    # 检查是否截断（最后几个字符不是完整结尾）
    if full_text.rstrip().endswith("...") or full_text.rstrip().endswith("```") is False:
        print("[dispatch] 可能截断，发送 Continue...")
        cont_text, cont_tools = send_prompt(conv_id, "Continue", effort="high", timeout=600)
        cont_files, _ = extract_code_blocks(cont_text)
        files.update(cont_files)
        for i, code in enumerate(cont_tools):
            if len(code) > 100:
                files[f"repl_cont_{i}.py"] = code
    
    # 保存所有提取的代码
    out_dir = "/home/claude/lynceus-CMD/opus_output"
    os.makedirs(out_dir, exist_ok=True)
    
    # 保存完整回复
    with open(f"{out_dir}/full_response.md", "w") as f:
        f.write(full_text)
    
    for fname, code in files.items():
        path = os.path.join(out_dir, os.path.basename(fname))
        with open(path, "w") as f:
            f.write(code)
        print(f"[dispatch] 已保存: {path} ({len(code)} chars)")
    
    # 同时保存原始代码块
    for i, block in enumerate(raw_blocks):
        path = os.path.join(out_dir, f"raw_block_{i}.py")
        with open(path, "w") as f:
            f.write(block)
    
    print(f"\n[dispatch] 完成，共 {len(files)} 个文件保存到 {out_dir}/")
    return conv_id

def build_porting_prompt():
    """构造发送给子模型的移植任务prompt"""
    
    # 读取几个关键上游文件作为附件
    upstream_files = {}
    key_files = [
        "upstream/par2qo/code/carver/kepler/model_trainer/loss_functions.py",
        "upstream/par2qo/code/carver/kepler/model_trainer/model_base.py",
        "upstream/par2qo/code/carver/kepler/model_trainer/trainer.py",
        "upstream/par2qo/code/carver/kepler/model_trainer/trainer_util.py",
        "upstream/par2qo/code/carver/kepler/data_management/workload.py",
        "upstream/par2qo/code/carver/kepler/data_management/database_simulator.py",
    ]
    
    for fpath in key_files:
        full = f"/home/claude/lynceus-CMD/{fpath}"
        if os.path.exists(full):
            with open(full) as f:
                content = f.read()
            upstream_files[fpath] = content
    
    # 读取已有ported文件样例
    example_path = "/home/claude/lynceus-CMD/lynceus/integrations/par2qo_cardinality.py"
    if os.path.exists(example_path):
        with open(example_path) as f:
            example_code = f.read()[:3000]
    else:
        example_code = "# 无法读取样例"

    prompt = f"""你是 lynceus-CMD 项目的第二位Claude开发者(Claude #2)，负责 M121-M126 里程碑。

## 任务背景
我们在移植 upstream/par2qo/code/carver/kepler/ 下的模型训练代码到 lynceus/integrations/ 下。

## 移植规则（必须遵守）
1. **算法改20%**：把TensorFlow/Keras替换为纯numpy实现（MLP前向传播、损失函数、梯度更新全用numpy矩阵运算）
2. **不改字符串/docstring**：改的是真正的算法逻辑，比如用 Welford 在线方差替代 batch 统计，用 EMA 平滑替代 raw mean
3. **断点调试**：每个函数都加 `_dbg(tag, **kw)` 打印所有输入参数和结构体状态
4. **文件名**：不加 v2/port 后缀，直接叫 kepler_xxx.py
5. **无TF依赖**：全部用 numpy + scipy（可选）

## 已有移植样例（学习这个风格）
```python
{example_code}
```

## 需要移植的上游文件

"""
    for fpath, content in upstream_files.items():
        # 截取前面部分以控制长度
        truncated = content[:4000] if len(content) > 4000 else content
        prompt += f"\n### {fpath}\n```python\n{truncated}\n```\n"

    prompt += """

## 你需要输出以下6个文件（每个文件用 ```python 代码块，第一行写 # FILE: 文件名）

1. **# FILE: kepler_loss_functions.py** — 损失函数 (numpy实现 MSE/LogMSE，额外加 Huber loss 和 asymmetric loss)
2. **# FILE: kepler_model_base.py** — 模型基类 (numpy MLP前向传播，Xavier初始化，ReLU/GELU激活，dropout用mask)
3. **# FILE: kepler_trainer.py** — 训练器 (Classification/Regression/NearOptimal三种，numpy SGD+momentum+weight decay)
4. **# FILE: kepler_trainer_util.py** — 工具函数 (类型转换、预处理、sample_weight计算，无TF)
5. **# FILE: kepler_workload.py** — 工作负载管理 (QueryInstance/Workload/WorkloadGenerator，加reservoir采样)
6. **# FILE: kepler_db_simulator.py** — 数据库模拟器 (PlannedQuery/DatabaseSimulator，加延迟抖动模拟和缓存)

每个文件 200-400 行，总计 ~1800 行。关键：
- 每个函数必须有 `_dbg()` 调用
- 算法部分真正改动（不是改注释）
- 可以直接运行 `python3 -c "import kepler_loss_functions"` 无报错

开始输出代码："""

    return prompt

if __name__ == "__main__":
    main()
