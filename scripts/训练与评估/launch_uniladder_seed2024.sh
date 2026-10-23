#!/usr/bin/env bash
# Uni→UniTail→UniOER→UniOER-Tail→OEHSR 五臂递进消融（TrajGap-Bench Mixed-direct）
# 协议：seed 2024, K=20, 100 epochs, batch 64, monitor=val_new_minFDE20, screening（单种子）
# 每臂一条串行 lane（5 场景），输出 outputs/uniladder_seed2024/<Variant>_<SCENE>_..._uni/
# 场景顺序 UNIV-M 优先（最快出完整 cell，尽早拿到样本主导场景信号）。
# 挂卡：GPU0=Uni+UniOER, GPU3=UniTail+OEHSR, GPU2=UniOER-Tail（用户确认挂满载卡）。
set -u
cd /home/lbh/GAST-PF
ROOT=outputs/uniladder_seed2024
mkdir -p "$ROOT"
PY=/home/lbh/.conda/envs/DeMo/bin/python
ENVV="PYTHONNOUSERSITE=1 PYTHONPATH=. LD_LIBRARY_PATH=/home/lbh/.conda/envs/DeMo/lib"

run_lane () {
  local variant=$1 gpu=$2
  local log="$ROOT/launcher_${variant}.log"
  setsid env $ENVV "$PY" -u "scripts/训练与评估/run_trajimpute_experiments.py" \
    --protocol mixed-direct --variant "$variant" \
    --scenes UNIV-M ETH-M HOTEL-M ZARA1-M ZARA2-M \
    --seed 2024 --gpu "$gpu" --screening --output-root "$ROOT" \
    > "$log" 2>&1 < /dev/null &
  echo "[$(date '+%F %T')] launched $variant gpu$gpu pid=$! log=$log"
}

run_lane GAST-PF-Uni        0
run_lane GAST-PF-UniOER     0
run_lane GAST-PF-UniTail    3
run_lane GAST-PF-OEHSR      3
run_lane GAST-PF-UniOER-Tail 2
wait_not_needed=1
echo "all lanes detached"
