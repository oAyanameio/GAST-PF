#!/usr/bin/env bash
# 证据—置信度探针批跑：M0 与 GAST-PF 五场景，只读推理，串行跑在单卡。
# 注意：checkpoint 文件名含 '=' 字符，统一用 epo*.ckpt glob 解析，避免手写路径出错。
set -u
cd /home/lbh/GAST-PF
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate DeMo
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTHONPATH=.

OUT=outputs/grad_probe/evidence_confidence
mkdir -p "$OUT"

resolve() {  # $1 = run dir
  local hit
  hit=$(ls "$1"/train/checkpoints/epo*.ckpt 2>/dev/null | head -1)
  [ -n "$hit" ] && echo "$hit"
}

declare -A M0=(
  [ETH-M]=outputs/m0_baseline/M0_optimizerfix_seed2024_gpu0/M0_ETH-M_mixed-direct_seed2024_uni
  [HOTEL-M]=outputs/m0_baseline/M0_optimizerfix_seed2024_gpu0/M0_HOTEL-M_mixed-direct_seed2024_uni
  [UNIV-M]=outputs/m0_baseline/M0_optimizerfix_seed2024_gpu1/M0_UNIV-M_mixed-direct_seed2024_uni
  [ZARA1-M]=outputs/m0_baseline/M0_optimizerfix_seed2024_gpu1/M0_ZARA1-M_mixed-direct_seed2024_uni
  [ZARA2-M]=outputs/m0_baseline/M0_optimizerfix_seed2024_gpu1/M0_ZARA2-M_mixed-direct_seed2024_uni
)
declare -A GAST=(
  [ETH-M]=outputs/unitraj_encoder_single_scene_seed2024/UT-ENC_ETH-M_mixed-direct_seed2024_uni
  [HOTEL-M]=outputs/unitraj_encoder_bench_seed2024/lane_a/UT-ENC_HOTEL-M_mixed-direct_seed2024_uni
  [UNIV-M]=outputs/unitraj_encoder_bench_seed2024/lane_d/UT-ENC_UNIV-M_mixed-direct_seed2024_uni
  [ZARA1-M]=outputs/unitraj_encoder_bench_seed2024/lane_b/UT-ENC_ZARA1-M_mixed-direct_seed2024_uni
  [ZARA2-M]=outputs/unitraj_encoder_bench_seed2024/lane_b/UT-ENC_ZARA2-M_mixed-direct_seed2024_uni
)

SCENES="ETH-M HOTEL-M ZARA1-M ZARA2-M UNIV-M"

for variant in M0 GAST-PF; do
  for scene in $SCENES; do
    tag="${scene}_${variant}"
    if [ -f "$OUT/${tag}.json" ]; then echo "[skip] $tag exists"; continue; fi
    if [ "$variant" = "M0" ]; then dir=${M0[$scene]}; else dir=${GAST[$scene]}; fi
    ckpt=$(resolve "$dir")
    if [ -z "$ckpt" ]; then echo "[FAIL] no checkpoint under $dir"; continue; fi
    echo "=== [$variant] $scene -> $ckpt"
    python scripts/审计与校验/probe_evidence_confidence.py \
      --scene "$scene" --variant "$variant" --checkpoint "$ckpt" \
      --device cuda:0 --batch-size 256 || echo "[FAIL] $tag"
  done
done
echo "[done] evidence-confidence probe finished"
