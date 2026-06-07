#!/usr/bin/env python3
"""loop_dispatch.py — 循环调度子模型Claude, 让它们自己clone+干活"""
import requests, json, uuid, sys, os, time, re

ORG = "518a2313-4665-47d1-bad5-803ab2700f7c"
BASE = f"https://claude.hk.cn/api/organizations/{ORG}"
MODEL = "claude-opus-4-6"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

with open("/tmp/claude_hk_cookie.txt") as f:
    raw_cookie = f.read().strip()
CK = {}
for part in raw_cookie.split("; "):
    if "=" in part: k,v = part.split("=",1); CK[k]=v

def new_conv():
    r = requests.post(f"{BASE}/chat_conversations",
        json={"name":"","project_uuid":None,"model":None},
        headers={"content-type":"application/json","origin":"https://claude.hk.cn","user-agent":UA},
        cookies=CK, timeout=15)
    return r.json()["uuid"]

def send(conv_id, prompt, timeout=480):
    h, a = str(uuid.uuid4()), str(uuid.uuid4())
    payload = {
        "prompt": prompt, "timezone": "Asia/Shanghai",
        "personalized_styles": [{"type":"default","key":"Default","name":"Normal",
            "nameKey":"normal_style_name","prompt":"Normal\n",
            "summary":"Default responses from Claude","summaryKey":"normal_style_summary","isDefault":True}],
        "locale": "en-US", "model": MODEL, "effort": "high", "thinking_mode": "off",
        "tools": [{"type":"repl_v0","name":"repl"}],
        "turn_message_uuids": {"human_message_uuid": h, "assistant_message_uuid": a},
        "attachments":[],"files":[],"sync_sources":[],"rendering_mode":"messages",
        "create_conversation_params":{"name":"","model":MODEL,
            "include_conversation_preferences":True,"paprika_mode":None,"compass_mode":None,
            "tool_search_mode":"auto","is_temporary":False,"enabled_imagine":True}
    }
    r = requests.post(f"{BASE}/chat_conversations/{conv_id}/completion",
        json=payload,
        headers={"accept":"text/event-stream","content-type":"application/json",
                 "anthropic-client-platform":"web_claude_ai",
                 "origin":"https://claude.hk.cn","referer":"https://claude.hk.cn/new","user-agent":UA},
        cookies=CK, timeout=timeout, stream=True)
    text_parts = []
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "): continue
        try: d = json.loads(line[6:])
        except: continue
        if d.get("type") == "content_block_delta":
            delta = d.get("delta", {})
            if delta.get("type") == "text_delta": text_parts.append(delta["text"])
    return "".join(text_parts)

def extract_files(text):
    blocks = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
    files = {}
    for block in blocks:
        m = re.match(r'#\s*FILE:\s*(\S+)', block)
        if m:
            files[m.group(1)] = block[m.end():].lstrip('\n')
    return files

def dispatch_worker(worker_id, prompt, out_dir):
    """一个worker = 一个对话, 可能多轮Continue"""
    os.makedirs(out_dir, exist_ok=True)
    conv_id = new_conv()
    print(f"  [Worker#{worker_id}] conv={conv_id}")
    
    t0 = time.time()
    text = send(conv_id, prompt)
    elapsed = time.time() - t0
    print(f"  [Worker#{worker_id}] Round1: {len(text)} chars in {elapsed:.0f}s")
    
    with open(f"{out_dir}/response.md", "w") as f: f.write(text)
    all_files = extract_files(text)
    
    # Continue如果截断
    rounds = 1
    while len(text) > 100 and not text.rstrip().endswith("```") and rounds < 4:
        print(f"  [Worker#{worker_id}] 可能截断, 发Continue (round {rounds+1})...")
        cont = send(conv_id, "Continue 继续输出剩余代码")
        with open(f"{out_dir}/response_r{rounds+1}.md", "w") as f: f.write(cont)
        all_files.update(extract_files(cont))
        text = cont
        rounds += 1
    
    for fname, code in all_files.items():
        path = f"{out_dir}/{os.path.basename(fname)}"
        with open(path, "w") as f: f.write(code)
        print(f"    → {path} ({len(code)} chars)")
    
    return all_files, conv_id

if __name__ == "__main__":
    worker_id = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    prompt = sys.argv[2] if len(sys.argv) > 2 else ""
    if not prompt:
        print("Usage: python3 loop_dispatch.py <worker_id> '<prompt>'")
        sys.exit(1)
    out_dir = f"/home/claude/lynceus-CMD/opus_output/worker{worker_id}"
    files, conv = dispatch_worker(worker_id, prompt, out_dir)
    print(f"\n[Worker#{worker_id}] 完成: {len(files)} files")
