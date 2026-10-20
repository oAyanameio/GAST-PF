#!/usr/bin/env bash
# GAST-PF + Directional Attention Pooling 五场景筛选训练。
#
# 用法：
#   bash scripts/训练与评估/launch_stage_dap.sh
#   LANES="1:ETH-M,UNIV-M,ZARA2-M 2:HOTEL-M,ZARA1-M" \
#     bash scripts/训练与评估/launch_stage_dap.sh

set -u

REPO=/home/lbh/GAST-PF
PY=/home/lbh/.conda/envs/DeMo/bin/python
ROOT=${ROOT:-outputs/gast_dap_seed2024}
VARIANT=GAST-PF-DAP
LANES=${LANES:-"1:ETH-M,UNIV-M,ZARA2-M 2:HOTEL-M,ZARA1-M"}
COMMON="--protocol mixed-direct --variant $VARIANT --seed 2024 \
  --batch-size 64 --num-workers 4 --epochs 100 --K 20 --screening \
  --allow-scene-partition --output-root $ROOT"

cd "$REPO" || exit 1
mkdir -p "$ROOT"

launch() {
  local gpu=$1
  shift
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
  launch "$gpu" $(echo "$scenes" | tr ',' ' ')
done

WATCHER_PATTERN="[w]atch_stage_variant[.]py --variant ${VARIANT} --root ${ROOT}"
if ! pgrep -f "$WATCHER_PATTERN" > /dev/null; then
  setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH=. "$PY" -u \
    scripts/训练与评估/watch_stage_variant.py \
    --variant "$VARIANT" --root "$ROOT" \
    >> "$ROOT/watcher_dap.log" 2>&1 < /dev/null &
  echo "  launched watcher pid=$!"
fi
