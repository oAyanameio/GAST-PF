# GAST-PF: 面向缺失历史轨迹的行人未来预测代码库

GAST-PF（Gap-Aware Spatio-Temporal Pedestrian Forecaster）是面向不完整历史观测的行人未来轨迹预测研究代码库，当前覆盖 ETH/UCY、SDD 和 TrajImpute 缺失历史数据协议，采用无地图/车道的 actor-only 预测设置。

- 模型：`GAST-PF` 由 `TrajectoryForecaster` 和 `FutureDistributionDecoder` 组成；Mode Query、State Query、Hybrid Coupling 以及 UniMamba/BiMamba 作为内部结构术语
- 数据：ETH/UCY 5 折留一（LOO）+ SDD 原始协议；缺失历史主线使用 TrajImpute 官方 Easy/Hard release
- 研究范围：缺失历史条件下的直接未来预测，不执行历史轨迹插补

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

预处理产物已在 `data/ETHUCY_processed/`（5 场景 .pt，经典 LOO 划分）；
原始 txt 预处理脚本已随 benchmark_v1 作废删除，正常使用无须重建。

### SDD（Stanford Drone Dataset）

仓库内无 SDD 预处理脚本。需自行准备 **标准 pkl/划分格式源数据**（`data/sdd/`）。

### TrajImpute 官方缺失历史数据

正式缺失历史数据位于 `/home/lbh/TrajImpute/dataset/TrajImpute`，包含
ETH-M、HOTEL-M、UNIV-M、ZARA1-M、ZARA2-M 的 Easy/Hard train/val/test pkl。
使用 `config_missing_aware_trajimpute` 和 `run_trajimpute_experiments.py`，不再生成仓库内自定义缺失数据。

## 训练与评估

```bash
# ETH/UCY 完整数据（原始协议）
python train.py
python train.py --config-name=config_ethucy fold=UNIV
python eval.py checkpoint='/path/to/ckpt' test=true

# TrajImpute Easy/Hard direct 缺失历史训练
PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/训练与评估/run_trajimpute_experiments.py \
    --protocol mixed-direct --variant GAST-PF \
    --scenes ETH-M HOTEL-M UNIV-M ZARA1-M ZARA2-M --gpu 0
```

## 目录结构

```
conf/                 Hydra 配置（ETH/UCY、SDD、TrajImpute）
src/datamodule/       ETH/UCY、SDD 与 TrajImpute 数据管线
src/model/            TrajectoryForecaster + FutureDistributionDecoder（旧名兼容入口保留）
src/metrics/          minADE/minFDE/MR/brierFDE
scripts/              数据集构建 / 训练与评估 / 审计与校验 / 结果分析
docs/                 研究文档、实验结果、审计报告、数据集说明
```

## 实验结果索引

- 结果总索引：`docs/results/README.md`（按实验性质分类）
- 总汇总：`docs/results/主线实验记录.md`
- 逐折数字：`outputs/*/results.json`
- 审计报告：`docs/audits/`
- 完整数据基线与公开 SOTA 参考：`docs/results/完整数据基线对比.md`

## 方法名称

```text
GAST-PF = Gap-Aware Spatio-Temporal Pedestrian Forecaster
中文：缺失间隔感知时空行人轨迹预测模型
```

致谢：[VideoMamba](https://github.com/OpenGVLab/VideoMamba)、[Forecast-MAE](https://github.com/jchengai/forecast-mae)、[StreamPETR](https://github.com/exiawsh/StreamPETR)
