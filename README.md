# GAST-PF

GAST-PF 是面向**缺失历史直接未来预测**的研究代码库，包含本文方法 **GAST** 与 **TrajGap-Bench** 基准协议。仓库覆盖 ETH/UCY 完整数据、SDD 以及 TrajImpute 缺失历史数据，统一采用无地图/车道的 actor-only 设置。

| 项目 | 说明 |
|---|---|
| 方法 | `GAST`（Gap-Aware Spatio-Temporal Pedestrian Forecaster） |
| 基准 | `TrajGap-Bench`：TrajImpute 官方 Easy/Hard 整理为 Mixed direct 协议 |
| 任务 | 8 帧历史（显式缺失掩码）→ 12 帧未来，`K=20` |
| 指标 | `minADE20` / `minFDE20` |
| 基线 | `M0`（同协议历史对照） |

方法完整定义、协议细节与结果见 [TrajGap-Bench.md](TrajGap-Bench.md)。

## 方法概览

GAST 由缺失证据感知编码器与 goal 条件化解码器组成：

```text
GAST
├── GAST gap-aware encoder
│   ├── 单向 Mamba 时序编码
│   ├── GERE  逐帧几何证据关系编码
│   └── ECTA  观测证据条件化因果时间聚合
│       ├── RMCM  可靠度调制的因果 Mamba
│       └── EWSA  证据加权状态聚合
└── Goal-conditioned future decoder
    ├── anchor state + 显式未来时间间隔
    ├── 终点偏移预测（相对最后有效观测位置）
    └── final-consistent loss + mode diversity regularization
```

goal 条件解码与轨迹级模式排序是 GAST 的内置设计，不单独注册为第二个模型；`gast_goal=false` / `M0` 路径仅用于消融与历史基线对照。

## 环境安装

```bash
conda create -n gast-pf python=3.10
conda activate gast-pf
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Mamba（VideoMamba fork，支持 bimamba=True）
git clone git@github.com:OpenGVLab/VideoMamba.git
cd VideoMamba
pip install -e causal-conv1d
pip install -e mamba
```

注意：RTX 5880 Ada / CUDA 12.8 下 Triton RMSNorm 内核不兼容，`vim_mamba.py` 已改用 PyTorch 原生回退，详见 `docs/audits/`。

## 数据准备

### ETH/UCY（完整数据）

预处理产物位于 `data/ETHUCY_processed/`（5 场景 .pt，经典留一 LOO 划分），由 `conf/datamodule/ethucy.yaml` 直接读取。

### SDD（Stanford Drone Dataset）

仓库内不包含 SDD 预处理脚本，需自行准备标准 pkl/划分格式源数据至 `data/sdd/`。

### TrajImpute（缺失历史）

正式缺失历史数据位于 `/home/lbh/TrajImpute/dataset/TrajImpute`，包含 ETH-M、HOTEL-M、UNIV-M、ZARA1-M、ZARA2-M 的 Easy/Hard train/val/test pkl。

正式协议为 **Mixed direct**：每个 split 下 `Easy + Hard` 做 actor-centric 样本拼接，缺失掩码显式输入，不使用外部插补，也不使用未来真值填补历史。Easy/Hard 标签仅用于内部审计与诊断。

## 训练与评估

```bash
# 默认入口：TrajImpute 单场景（ETH-M / Easy）
python train.py

# 正式 TrajGap-Bench Mixed 单场景 smoke
PYTHONPATH=. python train.py --config-name=config_missing_aware_trajimpute \
    scene=ETH-M difficulty=Mixed train_difficulties=[Mixed] val_difficulties=[Mixed] \
    hydra.run.dir=outputs/trajgap_smoke/GAST_ETH-M

# 五场景 Mixed direct 统一训练入口（仅注册 GAST / M0）
PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/训练与评估/run_trajimpute_experiments.py \
    --protocol mixed-direct --variant GAST \
    --scenes ETH-M HOTEL-M UNIV-M ZARA1-M ZARA2-M \
    --seed 2024 --gpu 0 \
    --output-root outputs/trajgap_bench_gast_seed2024

# 逐场景 direct 评估
python scripts/结果分析/evaluate_trajimpute_direct.py \
    --data-root /home/lbh/TrajImpute/dataset/TrajImpute \
    --scene ETH-M --difficulty Mixed --split test \
    --variant GAST --K 20 --seed 2024 \
    --checkpoint <path/to/checkpoint.ckpt> --output-root <output_dir>

# ETH/UCY 完整数据（按 LOO 换 test_scenes，5 折分别运行）
python train.py --config-name=config_ethucy
python train.py --config-name=config_ethucy \
    datamodule.train_scenes=[HOTEL,ZARA1,ZARA2] \
    datamodule.val_scenes=[ETH] \
    datamodule.test_scenes=[UNIV]

# 通用 checkpoint 评估
python eval.py checkpoint='/path/to/ckpt' test=true
```

注意：checkpoint 文件名含 `=`（如 `epoch=73.ckpt`）时 Hydra override 需整体加引号；`eval.py` 输出目录已存在 `model.log` 会报 `FileExistsError`，重跑前先删除该目录。

## 目录结构

```text
conf/                 Hydra 配置（ETH/UCY、SDD、TrajImpute；模型见 conf/model/）
src/model/            TrajectoryForecaster、loss 与 LightningModule
src/model/layers/     gast_encoder.py（GERE/ECTA/RMCM/EWSA）、gast_goal_decoder.py、
                      structured_future_decoder.py、mamba
src/datamodule/       ETH/UCY、SDD 与 TrajImpute（TrajGapDataset）数据管线
src/metrics/          minADE/minFDE/MR/brierFDE
scripts/              训练与评估 / 结果分析 / 数据集构建 / 审计与校验
tests/                数据集、编码器、goal 解码器、loss 与优化器契约单元测试
docs/                 研究方案、实验总汇总、数据审计、文献与框架图
paper/                论文 LaTeX 源（本地草稿，未纳入版本控制）
outputs/              训练与评估产物（逐折 results.json）
```

## 文档与图表

- 基准定义与复现：`TrajGap-Bench.md`
- 结果总索引：`docs/results/README.md`
- 唯一权威实验总汇总：`docs/results/实验总汇总.md`
- TrajGap-Bench 榜单：`docs/results/trajgap_bench_leaderboard.md`
- 数据审计：`docs/audits/trajimpute_release_审计.md`
- 框架图：根目录 `GAST-Goal-framework.drawio` / `.png`，历史版本见 `docs/figures/`

## 当前结果

TrajGap-Bench Mixed direct（单种子 2024，最终预测头 `val_new_minFDE20` 选点）：

| 方法 | Macro `minADE20` | Macro `minFDE20` | 完成度 |
|---|---:|---:|---:|
| **GAST** | **0.3168** | **0.4831** | 5/5 |
| GAST 消融（标准 decoder，不含 goal 条件解码） | 0.3264 | 0.4946 | 5/5 |
| MoFlow | 0.3496 | 0.5227 | 5/5 |

逐场景数字、micro average 与外部基线见 `TrajGap-Bench.md` §4。当前结果为单种子 screening，定稿前仍需第二种子与按原始轨迹聚类的统计检验。

## 方法名称

```text
GAST = Gap-Aware Spatio-Temporal Pedestrian Forecaster
中文：缺失间隔感知时空行人轨迹预测模型
GAST 为完整方法唯一名称；goal 条件解码与轨迹级模式排序属于其内置设计
TrajGap-Bench 是基准名称，不是方法名称
GAST-PF 为代码仓库名与历史产物前缀，仅作兼容保留
```

致谢：[VideoMamba](https://github.com/OpenGVLab/VideoMamba)、[Forecast-MAE](https://github.com/jchengai/forecast-mae)、[StreamPETR](https://github.com/exiawsh/StreamPETR)
