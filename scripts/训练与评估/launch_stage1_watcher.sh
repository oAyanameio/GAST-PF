#!/usr/bin/env bash
# Stage 1 · AnchorGeo watcher 脱离式启动（幂等：已在跑则先停旧的再起新的）。
#
# 用 setsid 使 watcher 成为独立会话首进程，避免随 agent 会话被回收。
set -u

REPO=/home/lbh/GAST-PF
PY=/home/lbh/.conda/envs/DeMo/bin/python
ROOT=outputs/stage1_anchor_geo_seed2024

cd "$REPO" || exit 1
mkdir -p "$ROOT"

WATCHER_PATTERN='[w]atch_stage_variant[.]py --variant GAST-PF-AnchorGeo --root outputs/stage1_anchor_geo_seed2024'
for p in $(pgrep -f "$WATCHER_PATTERN"); do
  kill "$p" 2>/dev/null && echo "  stopped old watcher $p"
done
sleep 2

setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH=. "$PY" -u \
  scripts/训练与评估/watch_stage_variant.py \
  --variant GAST-PF-AnchorGeo --root "$ROOT" \
  >> "$ROOT/watcher_stage1.log" 2>&1 < /dev/null &
echo "  launched watcher pid=$!"
