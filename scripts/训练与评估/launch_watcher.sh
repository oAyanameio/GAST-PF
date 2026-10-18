#!/usr/bin/env bash
# 通用 watcher 脱离式启动（幂等：已在跑则先停旧的再起新的）。
#
# 用 setsid 使 watcher 成为独立会话首进程，避免随 agent 会话被回收。
#
# 用法：
#   bash scripts/训练与评估/launch_watcher.sh                      # 默认 Stage 2
#   VARIANT=GAST-PF-AnchorGeo ROOT=outputs/stage1_anchor_geo_seed2024 \
#     bash scripts/训练与评估/launch_watcher.sh
set -u

REPO=/home/lbh/GAST-PF
PY=/home/lbh/.conda/envs/DeMo/bin/python
VARIANT=${VARIANT:-GAST-PF-FrameGeo}
ROOT=${ROOT:-outputs/stage2_frame_geo_seed2024}
LOG=${LOG:-$ROOT/watcher.log}

cd "$REPO" || exit 1
mkdir -p "$ROOT"

PATTERN="[w]atch_stage_variant[.]py --variant ${VARIANT} --root ${ROOT}"
for p in $(pgrep -f "$PATTERN"); do
  kill "$p" 2>/dev/null && echo "  stopped old watcher $p"
done
sleep 2

setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH=. "$PY" -u \
  scripts/训练与评估/watch_stage_variant.py \
  --variant "$VARIANT" --root "$ROOT" \
  >> "$LOG" 2>&1 < /dev/null &
echo "  launched watcher pid=$! variant=$VARIANT root=$ROOT"
