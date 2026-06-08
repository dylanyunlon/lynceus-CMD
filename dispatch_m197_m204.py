#!/usr/bin/env python3
"""
dispatch_m197_m204.py — 调度Opus 4.6 sub-claude完成M197-M204
用法: python3 dispatch_m197_m204.py
"""
import json, uuid, time, sys, os
import urllib.request

ORG = "7b16d754-0edf-487d-8b5f-238d723c399c"
with open("/tmp/claude_hk_cookie.txt") as f:
    COOKIES = f.read().strip()
MODEL = "claude-opus-4-6"
BASE = f"https://claude.hk.cn/api/organizations/{ORG}"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
TIMEOUT = 280  # seconds

def make_headers(stream=False):
    h = {
        "Content-Type": "application/json",
        "Cookie": COOKIES,
        "Origin": "https://claude.hk.cn",
        "User-Agent": UA,
        "anthropic-client-platform": "web_claude_ai",
    }
    if stream:
        h["Accept"] = "text/event-stream"
        h["Referer"] = "https://claude.hk.cn/new"
    return h

def create_conv():
    req = urllib.request.Request(
        f"{BASE}/chat_conversations",
        data=json.dumps({"name":"","project_uuid":None,"model":None}).encode(),
        headers=make_headers(), method="POST"
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())["uuid"]

def send_msg(conv_id, prompt):
    h_uuid, a_uuid = str(uuid.uuid4()), str(uuid.uuid4())
    payload = {
        "prompt": prompt,
        "timezone": "Asia/Shanghai",
        "personalized_styles": [{"type":"default","key":"Default","name":"Normal",
            "nameKey":"normal_style_name","prompt":"Normal\n",
            "summary":"Default responses from Claude",
            "summaryKey":"normal_style_summary","isDefault":True}],
        "locale": "en-US", "model": MODEL, "effort": "medium",
        "thinking_mode": "off",
        "tools": [],
        "turn_message_uuids": {"human_message_uuid": h_uuid, "assistant_message_uuid": a_uuid},
        "attachments":[],"files":[],"sync_sources":[],"rendering_mode":"messages",
    }
    req = urllib.request.Request(
        f"{BASE}/chat_conversations/{conv_id}/completion",
        data=json.dumps(payload).encode(),
        headers=make_headers(stream=True), method="POST"
    )
    text_parts = []
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "): continue
                try:
                    d = json.loads(line[6:])
                except: continue
                if d.get("type") == "content_block_delta":
                    delta = d.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text_parts.append(delta["text"])
    except Exception as e:
        print(f"  Stream error: {e}")
    text = "".join(text_parts)
    elapsed = time.time() - t0
    return text, elapsed

# === 精简prompt: 给clone链接+milestone指令 ===
PROMPT = """git clone https://github.com/dylanyunlon/lynceus-CMD.git && cd lynceus-CMD && tree -L 2 lynceus/

你是lynceus-CMD第2位worker。任务: M197-M204, 移植8个upstream最大的未覆盖.py文件到 lynceus/integrations/。

规则:
- clone后看 upstream/ 目录和 lynceus/integrations/ 对比哪些没移植
- 每个文件算法改20%(不改字符串docstring), 加 _dbg() 断点
- 不许 v10/port/bridge/engine/utils/base/compat 后缀
- Author: dylanyunlon <dogechat@163.com>
- pure python+numpy, 能 import 通过

先输出M197: kepler_param_generation.py (从 upstream param_gen_test_output.py 2046行移植, Sobol替换random, Welford统计)。只输出代码。"""

def main():
    print(f"=== Dispatch M197-M204 to {MODEL} ===")
    
    conv_id = create_conv()
    print(f"Conv: {conv_id}")
    
    # Round 1: 发送主prompt
    print(f"Sending prompt ({len(PROMPT)} chars)...")
    text, elapsed = send_msg(conv_id, PROMPT)
    print(f"Got {len(text)} chars in {elapsed:.1f}s")
    
    out_path = "/tmp/subclaude_m197.txt"
    with open(out_path, "w") as f:
        f.write(text)
    print(f"Saved to {out_path}")
    
    # 如果截断了，发Continue
    if len(text) > 1000 and not text.rstrip().endswith("```"):
        print("Response may be truncated, sending Continue...")
        text2, elapsed2 = send_msg(conv_id, "Continue")
        print(f"Continue: {len(text2)} chars in {elapsed2:.1f}s")
        with open(out_path, "a") as f:
            f.write("\n" + text2)
    
    # 预览
    print("\n=== PREVIEW (first 300 chars) ===")
    print(text[:300])

if __name__ == "__main__":
    main()
