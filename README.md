# GAST-PF

GAST-PF 是面向**缺失历史直接未来预测**的研究代码库，包含本文方法 **GAST** 与 **TrajGap-Bench** 基准协议。仓库覆盖 ETH/UCY 完整数据、SDD 以及 TrajImpute 缺失历史数据，统一采用无地图/车道的 actor-only 设置。

| 项目 | 说明 |
|---|---|
| 方法 | `GAST`（Gap-Aware Spatio-Temporal Pedestrian Forecaster） |
| 数据集与基准 | `TrajGap-Bench`：基于 TrajImpute 官方 Easy/Hard，按 Mixed split 组织并直接预测 |
| 任务 | 8 帧历史（显式缺失掩码）→ 12 帧未来，`K=20` |
| 指标 | `minADE20` / `minFDE20` |

方法完整定义、协议细节与结果见 [TrajGap-Bench.md](TrajGap-Bench.md)。

## 方法概览

GAST 由观测证据感知时空编码器和终点引导解码器组成。编码器根据有效性掩码和缺失时长估计观测可靠度，并将其用于逐帧交互建模与历史信息汇总：

```text
GAST
├── 观测证据感知时空编码器
│   ├── 逐帧交互建模
│   └── 可靠度加权的时序聚合（可靠度调制的 Mamba 与历史状态汇总）
└── 终点引导解码器
    ├── 使用末次有效观测状态和实际预测间隔
    ├── 多模态终点预测与未来轨迹生成
    └── 双指派多模态训练目标（dual-assignment multimodal objective，`loss_mode=final_consistent`）+ mode diversity regularization
```

终点引导解码与轨迹级模式排序是 GAST 的内置设计。当前仓库的自有训练与评估入口只实现 GAST。

## 环境安装

```bash
conda create -n gast-pf python=3.10
conda activate gast-pf
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Mamba（VideoMamba fork）
git clone git@github.com:OpenGVLab/VideoMamba.git
cd VideoMamba
pip install -e causal-conv1d
pip install -e mamba
```

## 数据准备

### TrajImpute（缺失历史）

正式缺失历史数据位于 `/home/lbh/TrajImpute/dataset/TrajImpute`，包含 ETH-M、HOTEL-M、UNIV-M、ZARA1-M、ZARA2-M 的 Easy/Hard train/val/test pkl。

**TrajGap-Bench** 是数据集与评估基准的统一名称。每个 split 将 `Easy + Hard` 做 actor-centric 样本拼接，形成 `Mixed` 数据；`direct` 描述任务约束，即模型直接使用带缺失掩码的历史进行预测，不使用外部插补或未来真值填补历史。Easy/Hard 标签仅用于内部审计与诊断。

## 训练与评估

```bash
# 默认入口：TrajImpute 单场景（ETH-M / Easy）
python train.py

# 正式 TrajGap-Bench Mixed 单场景 smoke
PYTHONPATH=. python train.py --config-name=config_missing_aware_trajimpute \
    scene=ETH-M difficulty=Mixed train_difficulties=[Mixed] val_difficulties=[Mixed] \
    hydra.run.dir=outputs/trajgap_smoke/GAST_ETH-M

# 五场景 TrajGap-Bench 统一训练入口（仅注册 GAST）
PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/训练与评估/run_trajimpute_experiments.py \
    --protocol trajgap-bench --variant GAST \
    --scenes ETH-M HOTEL-M UNIV-M ZARA1-M ZARA2-M \
    --seed 2024 --gpu 0 \
    --output-root outputs/trajgap_bench_gast_seed2024

# 逐场景 direct 评估
python scripts/结果分析/evaluate_trajimpute_direct.py \
    --data-root /home/lbh/TrajImpute/dataset/TrajImpute \
    --scene ETH-M --difficulty Mixed --split test \
    --variant GAST --K 20 --seed 2024 \
    --checkpoint <path/to/checkpoint.ckpt> --output-root <output_dir>

# 通用 checkpoint 评估
python eval.py checkpoint='/path/to/ckpt' test=true
```

注意：checkpoint 文件名含 `=`（如 `epoch=73.ckpt`）时 Hydra override 需整体加引号；`eval.py` 输出目录已存在 `model.log` 会报 `FileExistsError`，重跑前先删除该目录。

## 目录结构

```text
conf/                 Hydra 配置（TrajImpute 主线与对照；模型见 conf/model/）
src/model/            TrajectoryForecaster、loss 与 LightningModule
src/model/layers/     gast_encoder.py（逐帧交互建模、可靠度加权的时序聚合）、
                      gast_goal_decoder.py（终点引导解码器）、
                      mamba
src/datamodule/       TrajImpute（TrajGapDataset）数据管线
src/metrics/          minADE/minFDE/MR/brierFDE
scripts/              训练与评估 / 结果分析 / 数据集构建 / 审计与校验
tests/                数据集、编码器、终点引导解码器、loss 与优化器契约单元测试
docs/                 研究方案、实验总汇总、数据审计与文献
paper/                论文 LaTeX 源（本地草稿，未纳入版本控制）
outputs/              训练与评估产物（逐折 results.json）
```

## 文档与图表

- 基准定义与复现：`TrajGap-Bench.md`
- 结果总索引：`docs/results/README.md`
- 唯一权威实验总汇总：`docs/results/实验总汇总.md`
- TrajGap-Bench 榜单：`docs/results/trajgap_bench_leaderboard.md`
- 数据审计：`docs/audits/trajimpute_release_审计.md`

## 当前结果

TrajGap-Bench（seed=2024；GAST 按最终预测头 `val_new_minFDE20` 选点；其他外部方法沿用各自选点流程）：

| 方法 | Macro `minADE20` | Macro `minFDE20` | 完成度 |
|---|---:|---:|---:|
| **GAST** | **0.3168** | **0.4831** | 5/5 |
| MoFlow | 0.3496 | 0.5227 | 5/5 |
| AgentFormer | 0.3911 | 0.6144 | 5/5 |
| SocialLSTM | 0.4576 | 0.6079 | 5/5 |
| Re | 0.6382 | 0.9775 | 5/5 |
| MS-TIP（插补感知） | 1.1838 | 1.3929 | 5/5 |
| Social-STGCNN | 1.3471 | 2.2861 | 5/5 |

逐场景数字、微平均与缺失分层见 `docs/results/trajgap_bench_leaderboard.md`；唯一汇总见 `docs/results/实验总汇总.md`。正式口径使用五场景 manifest 同一代码 revision 的完整轮次，不跨 revision 或逐场景拼接。

## 方法名称

```text
GAST = Gap-Aware Spatio-Temporal Pedestrian Forecaster
中文：缺失间隔感知时空行人轨迹预测模型
GAST 为完整方法唯一名称；终点引导解码与轨迹级模式排序属于其内置设计
TrajGap-Bench 是基准名称，不是方法名称
`GAST-PF` 是代码仓库名；本文方法统一称为 GAST。
```

致谢：[VideoMamba](https://github.com/OpenGVLab/VideoMamba)、[Forecast-MAE](https://github.com/jchengai/forecast-mae)、[StreamPETR](https://github.com/exiawsh/StreamPETR)
