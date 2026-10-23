#!/usr/bin/env bash
# Historical Stage 8 launcher retained for exact ECSR run-directory compatibility.
# New experiments should use launch_stage8_oer.sh.
set -u

REPO=/home/lbh/GAST-PF
PY=/home/lbh/.conda/envs/DeMo/bin/python
ROOT=${ROOT:-outputs/stage8_ecsr_seed2024}
VARIANT=GAST-PF-ECSR
LANES=${LANES:-"0:ZARA2-M 1:ETH-M,UNIV-M 2:HOTEL-M 3:ZARA1-M"}
COMMON="--protocol mixed-direct --variant $VARIANT --seed 2024 \
  --batch-size 64 --num-workers 4 --epochs 100 --K 20 --screening \
  --output-root $ROOT"

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
  # shellcheck disable=SC2086
  launch "$gpu" $(echo "$scenes" | tr ',' ' ')
done

WATCHER_PATTERN="[w]atch_stage_variant[.]py --variant ${VARIANT} --root ${ROOT}"
if pgrep -f "$WATCHER_PATTERN" > /dev/null; then
  echo "  watcher 已在运行，跳过"
else
  setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH=. "$PY" -u \
    scripts/训练与评估/watch_stage_variant.py \
    --variant "$VARIANT" --root "$ROOT" \
    >> "$ROOT/watcher_stage8.log" 2>&1 < /dev/null &
  echo "  launched watcher pid=$!"
fi
echo "done; 检查: pgrep -af 'run_trajimpute_experiments[.]py'"
