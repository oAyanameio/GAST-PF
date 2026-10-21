#!/usr/bin/env bash
# NLL 对照臂（MaskedSocialLSTM）五场景证据—置信度探针；只读推理，串行单卡。
set -u
cd /home/lbh/GAST-PF
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate DeMo
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTHONPATH=.

ROOT=/home/lbh/SocialLSTM/trajgap_bench/artifacts/sociallstm_bench
OUT=outputs/grad_probe/evidence_confidence
mkdir -p "$OUT"

for scene in HOTEL-M ZARA1-M ZARA2-M UNIV-M; do
  if [ -f "$OUT/${scene}_SocialLSTM-NLL.json" ]; then echo "[skip] $scene"; continue; fi
  SNAP=$(ls -d "$ROOT"/SocialLSTM_"$scene"_mixed-direct_seed2024_uni/train 2>/dev/null | head -1)
  [ -z "$SNAP" ] && { echo "[FAIL] no snapshot for $scene"; continue; }
  CKPT=$(ls "$SNAP"/checkpoints/epo*.ckpt 2>/dev/null | head -1)
  [ -z "$CKPT" ] && { echo "[FAIL] no ckpt for $scene"; continue; }
  echo "=== SocialLSTM-NLL $scene -> $CKPT"
  python scripts/审计与校验/probe_evidence_confidence_nll.py \
    --scene "$scene" --checkpoint "$CKPT" --arch-root "$SNAP" \
    --device cuda:0 --batch-size 256 || echo "[FAIL] $scene"
done
echo "[done] NLL contrast probe finished"
