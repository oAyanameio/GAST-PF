#!/usr/bin/env bash
# Stage 1 · AnchorGeo 四卡训练启动器。
#
# 为什么用 setsid + nohup：早期用 Hermes 的 background 机制启动的链挂在会话进程树
# 下，会话被中断（压缩/重启）时整棵树被回收，训练静默停止。setsid 让每条链成为
# 独立会话首进程，与 agent 会话完全脱钩。
#
# 用法：bash scripts/训练与评估/launch_stage1_anchor_geo.sh
set -u

REPO=/home/lbh/GAST-PF
PY=/home/lbh/.conda/envs/DeMo/bin/python
ROOT=outputs/stage1_anchor_geo_seed2024
VARIANT=GAST-PF-AnchorGeo
COMMON="--protocol mixed-direct --variant $VARIANT --seed 2024 \
  --batch-size 64 --num-workers 4 --epochs 100 --K 20 --screening \
  --output-root $ROOT"

cd "$REPO" || exit 1
mkdir -p "$ROOT"

launch() {  # $1=gpu  $2..=scenes
  local gpu=$1; shift
  setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH=. "$PY" -u \
    scripts/训练与评估/run_trajimpute_experiments.py \
    $COMMON --scenes "$@" --gpu "$gpu" \
    >> "$ROOT/launcher_gpu${gpu}.log" 2>&1 < /dev/null &
  echo "  launched gpu$gpu scenes=[$*] pid=$!"
}

launch 0 ZARA2-M
launch 1 ETH-M UNIV-M
launch 2 HOTEL-M
launch 3 ZARA1-M

# watcher 同样脱离会话；已在跑则不重复启动。
if pgrep -f 'watch_stage1_anchor_geo[.]py' > /dev/null; then
  echo "  watcher 已在运行，跳过"
else
  setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH=. "$PY" -u \
    scripts/训练与评估/watch_stage1_anchor_geo.py \
    >> "$ROOT/watcher_stage1.log" 2>&1 < /dev/null &
  echo "  launched watcher pid=$!"
fi
echo "done; 检查: pgrep -af 'run_trajimpute_experiments[.]py'"
