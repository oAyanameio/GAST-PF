#!/usr/bin/env bash
# Stage 2 · FrameGeo 训练启动器（setsid 脱离会话，与 agent 会话解耦）。
#
# 卡分配按启动时实际空闲情况给出，通过 LANES 覆盖；默认四卡各一条链。
#
# 用法：
#   bash scripts/训练与评估/launch_stage2_frame_geo.sh
#   LANES="1:HOTEL-M,ZARA1-M 2:ETH-M,ZARA2-M,UNIV-M" \
#     bash scripts/训练与评估/launch_stage2_frame_geo.sh
#
# 启动前先看空闲卡：
#   nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
set -u

REPO=/home/lbh/GAST-PF
PY=/home/lbh/.conda/envs/DeMo/bin/python
ROOT=outputs/stage2_frame_geo_seed2024
VARIANT=GAST-PF-FrameGeo
LANES=${LANES:-"0:ZARA2-M 1:ETH-M,UNIV-M 2:HOTEL-M 3:ZARA1-M"}
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

echo "LANES=$LANES"
for lane in $LANES; do
  gpu=${lane%%:*}
  scenes=${lane#*:}
  # shellcheck disable=SC2086
  launch "$gpu" $(echo "$scenes" | tr ',' ' ')
done

# watcher 同样脱离会话；已在跑则不重复启动。
WATCHER_PATTERN="[w]atch_stage_variant[.]py --variant ${VARIANT} --root ${ROOT}"
if pgrep -f "$WATCHER_PATTERN" > /dev/null; then
  echo "  watcher 已在运行，跳过"
else
  setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH=. "$PY" -u \
    scripts/训练与评估/watch_stage_variant.py \
    --variant "$VARIANT" --root "$ROOT" \
    >> "$ROOT/watcher_stage2.log" 2>&1 < /dev/null &
  echo "  launched watcher pid=$!"
fi
echo "done; 检查: pgrep -af 'run_trajimpute_experiments[.]py'"
