#!/usr/bin/env bash
# scripts/lab_auto_push.sh — 实验室日志自动上传
# 在ags1上后台运行, 每次实验输出变化后自动commit+push
#
# 用法:
#   cd /data/jiacheng/system/cache/temp/nips2026
#   bash scripts/lab_auto_push.sh &
#
# Claude子模型可以通过 git pull 读取最新实验数据
set -euo pipefail

INTERVAL=${INTERVAL:-30}  # 检查间隔(秒)
BRANCH="main"

cd "$(dirname "$0")/.."
echo "[lab_auto_push] watching output/ every ${INTERVAL}s"

git config user.name "dylanyunlon"
git config user.email "dogechat@163.com"

last_hash=""
while true; do
    cur_hash=$(find output/ -name '*.json' -o -name '*.tex' 2>/dev/null | sort | xargs md5sum 2>/dev/null | md5sum | cut -d' ' -f1)
    if [ "$cur_hash" != "$last_hash" ] && [ -n "$cur_hash" ]; then
        git add output/ 2>/dev/null || true
        if ! git diff --cached --quiet 2>/dev/null; then
            ts=$(date +%Y%m%d_%H%M%S)
            git commit -m "exp: ags1 data update ${ts}" --author="dylanyunlon <dogechat@163.com>" 2>/dev/null
            git push origin "${BRANCH}" 2>/dev/null && echo "[${ts}] pushed" || echo "[${ts}] push failed"
            last_hash="$cur_hash"
        fi
    fi
    sleep "$INTERVAL"
done
