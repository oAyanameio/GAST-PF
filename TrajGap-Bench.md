# TrajGap-Bench

[中文版](#中文版) ｜ [English Version](#english-version)

<details open>
<summary id="中文版">中文版</summary>

[English version](#english-version)

## 1. 基准简介

本文整理形成 **TrajGap 数据集**，并在此基础上建立 **TrajGap-Bench 评测协议**，用于评估缺失历史条件下的行人轨迹直接预测。模型直接读取带缺失掩码的历史观测并预测未来 12 帧，不以前置外部插补为条件。

TrajGap 数据集基于 TrajImpute 官方数据发布，沿用其场景和 train/validation/test 划分。TrajGap-Bench 明确任务约束、checkpoint 选择规则和评估指标，提供可复现且不使用测试集调参的比较口径；本文不重新主张上游数据资源贡献。

### 1.1 当前本文模型

当前本文完整方法统一命名为 **GAST（Gap-Aware Spatio-Temporal Pedestrian Forecaster）**：

```text
GAST
├── 观测状态感知时空编码器
│   ├── 逐帧交互建模
│   └── 状态强度加权的时序聚合（状态强度调制的 Mamba 与历史状态汇总）
└── 终点引导解码器
    ├── 使用末次有效观测状态和实际预测间隔
    ├── 多模态终点预测与未来轨迹生成
    ├── final-consistent loss
    └── mode diversity regularization
```

终点引导解码和轨迹级模式排序是 GAST 的内置设计，不单独注册为第二个模型。当前仓库的自有训练与评估入口只实现 GAST。

### 1.2 研究问题

完整历史预测假设所有观测均可用，但真实跟踪系统可能受到遮挡、漏检、数据关联失败和传感器丢帧影响。TrajGap-Bench 关注以下问题：

1. 模型能否直接利用不完整历史，而不是依赖外部插补？
2. 在相同缺失数量下，缺失位置和时间结构是否会改变预测难度？
3. 模型性能是否能在可观测证据减少时平稳退化？
4. GAST 的终点引导解码是否能改善缺失历史下的多模态未来覆盖？

## 2. 数据集与任务

**TrajGap** 指本文整理形成的数据集，**TrajGap-Bench** 指围绕该数据集定义的评测协议。`direct prediction` 表示模型直接依据不完整历史观测预测未来，不使用外部历史插补器。

### 2.1 数据来源

TrajGap 使用 TrajImpute 官方 release：

- 场景：`ETH-M`、`HOTEL-M`、`UNIV-M`、`ZARA1-M`、`ZARA2-M`；
- 划分：`train`、`val`、`test`；
- 历史：8 帧，时间间隔 0.4 s，共 3.2 s；
- 未来：12 帧，共 4.8 s；
- 预测：每个目标行人输出 `K=20` 条候选轨迹。

每个样本包含：

| 字段 | 形状 | 含义 |
|---|---:|---|
| `obs_traj` | `[N, 8, 2]` | 历史坐标，缺失位置为 `NaN` |
| `pred_traj` | `[N, 12, 2]` | 完整未来轨迹 |
| `missing_mask` | `[N, 8, 2]` | `True` 表示该坐标缺失 |
| `seq_start_end` | `[M, 2]` | 场景内行人分组边界 |

每个样本至少保留 1 个有效历史帧。缺失坐标由 `missing_mask` 显式定义，不允许使用未来真值填补历史输入。

### 2.2 数据划分与样本规模

TrajGap 沿用上游数据的场景划分，并对各 split 中的 focal 行人轨迹统一组织。适配过程不重新切窗、不重采样、不去重，也不生成新的未来目标。

| 场景 | TrajGap train | TrajGap val | TrajGap test |
|---|---:|---:|---:|
| `ETH-M` | 59,618 | 10,698 | 1,629 |
| `HOTEL-M` | 58,304 | 10,272 | 9,477 |
| `UNIV-M` | 18,462 | 5,416 | 219,006 |
| `ZARA1-M` | 56,020 | 10,236 | 20,277 |
| `ZARA2-M` | 51,014 | 8,346 | 52,497 |
| **合计** | **243,418** | **44,968** | **302,886** |

训练集的缺失帧数量覆盖 `0…7`。测试集保留官方的缺失程度扫描：同一潜在轨迹可能对应多个不同缺失掩码。因此，样本数不能直接解释为独立原始轨迹数；统计显著性分析应按原始轨迹聚类或采用 paired bootstrap。

## 3. 评估协议

### 3.1 训练、验证与测试

| 阶段 | 数据 | 用途 |
|---|---|---|
| Train | TrajGap train | 模型训练 |
| Validation | TrajGap validation | 只用于 checkpoint 选择 |
| Test | TrajGap test | 最终报告，不参与调参 |

所有方法必须遵守相同的：

- 8 帧历史 → 12 帧未来任务；
- TrajGap train/validation/test；
- direct 输入约束；
- `K=20` 候选轨迹；
- 五场景评估；
- 测试集不参与模型选择。

模型可以在内部学习隐式补全或 latent state，但不允许依赖一个外部插补模型作为预测前置步骤。先插补后预测方法应作为单独口径报告，不能直接进入 direct leaderboard。

### 3.2 Checkpoint 选择

Checkpoint 规则固定为：

```text
monitor = val_new_minFDE20
source  = TrajGap validation
head    = final prediction head
mode    = K=20
```

不允许：

- 使用测试结果选择 epoch；
- 根据单个场景结果挑选 checkpoint；
- 使用旧预测头或不同指标替代 `val_new_minFDE20`。

### 3.3 主指标

TrajGap-Bench 正式主榜只报告两个指标：

| 指标 | 定义 | 趋势 |
|---|---|---|
| `minADE20` | 20 条候选轨迹中最小的平均位移误差 | 越低越好 |
| `minFDE20` | 20 条候选轨迹中最小的终点位移误差 | 越低越好 |

两个指标必须来自同一批 `K=20` 候选轨迹、同一个 checkpoint 和同一个 TrajGap test 划分。不计算综合分数。

可以将缺失帧数量、末端缺失、`anchor_lag` 和 `forecast_gap` 作为辅助诊断，但它们不能替代主指标或定义新的总榜。

### 3.4 汇报方式

正式结果应同时报告：

1. 五个场景的逐场景结果；
2. 五场景 macro average；
3. 按测试样本数加权的 micro average；
4. worst-scene；
5. 随机种子和 checkpoint 选点规则。

不同方法若未完成五个场景，不得计算五场景 macro 或 micro，也不得把部分场景结果写成完整 benchmark 结论。

## 4. 对比实验结果

### 4.1 结果说明

完整逐场景、宏/微平均、排序诊断和缺失分层表由 [`docs/results/trajgap_bench_leaderboard.md`](docs/results/trajgap_bench_leaderboard.md) 自动生成并维护。正式结果采用五场景 manifest revision 一致的完整 GAST 轮次，宏平均 `minADE20/minFDE20` 为 **0.3168/0.4831**；MoFlow 为 **0.3496/0.5227**。六个外部对照均完成五场景。其他结果因场景间 revision 不一致，不合并为完整轮次。

### 4.2 当前结果

对比表以生成榜单为唯一数值来源，不在此重复保存静态表。协议文档保留模型配置、复现入口和评估限制。

### 4.3 本文模型配置

| 项目 | `GAST` |
|---|---|
| Encoder | 观测状态感知时空编码器（Mamba） |
| Interaction modeling | 逐帧交互建模（含交互偏置） |
| Temporal aggregation | 状态强度加权的时序聚合 |
| Decoder | 终点引导解码器 |
| Mode ranking | GAST 内置 trajectory-aware classification |
| Prediction modes | `K=20` |
| Loss | `final_consistent`（正文称双指派多模态训练目标，dual-assignment multimodal objective） |
| Checkpoint | TrajGap validation `val_new_minFDE20` |
| Seed | 2024 |
| Epochs | 100 |
| Batch size | 64 |

当前 GAST 的五场景结果为单种子 screening 结果。它可作为当前滚动基线，但定稿时仍应补充第二随机种子、置信区间以及按原始轨迹聚类的配对统计。

## 5. 代码与复现

### 5.1 训练

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. \
/home/lbh/.conda/envs/gast-pf/bin/python \
scripts/训练与评估/run_trajimpute_experiments.py \
  --protocol trajgap-bench \
  --variant GAST \
  --scenes ETH-M HOTEL-M UNIV-M ZARA1-M ZARA2-M \
  --seed 2024 \
  --gpu 0 \
  --output-root outputs/trajgap_bench_gast_seed2024
```

### 5.2 评估

```bash
/home/lbh/.conda/envs/gast-pf/bin/python \
scripts/结果分析/evaluate_trajimpute_direct.py \
  --data-root /home/lbh/TrajImpute/dataset/TrajImpute \
  --scene ETH-M \
  --split test \
  --variant GAST \
  --K 20 \
  --seed 2024 \
  --checkpoint <path/to/checkpoint.ckpt> \
  --output-root <output_dir>
```

### 5.3 关键实现

| 功能 | 实现 |
|---|---|
| TrajGap 数据集 | `src/datamodule/trajimpute_dataset.py::TrajGapDataset` |
| TrajGap 训练/验证 | `TrajImputeDataModule` |
| 统一训练入口 | `scripts/训练与评估/run_trajimpute_experiments.py` |
| 统一 direct 评估 | `scripts/结果分析/evaluate_trajimpute_direct.py` |
| 最新模型配置 | `conf/model/gast_model_forecast.yaml` |
| 主协议配置 | `conf/config_missing_aware_trajimpute.yaml` |

## 6. 限制与论文使用边界

1. TrajGap 是基于官方 release 的受控缺失协议，不应直接宣称等同于真实跟踪器的缺失分布。
2. 测试集保留同一潜在轨迹的多种缺失掩码，这是缺失程度扫描的设计，不应简单去重后再作为主结果。
3. 当前部分基线仍是历史 screening 结果；最终论文表应统一模型版本、checkpoint 规则、随机种子和评估脚本。
4. 先插补后预测与 direct prediction 输入信息不同，应分表报告，不能直接混入同一 leaderboard。
5. GAST 当前优势主要体现在 `minADE20` 与 `minFDE20`；其 mode 排序仍需通过第二种子和额外诊断确认，不能仅凭单次运行作强泛化结论。
6. 当前正式指标只包括 `minADE20` 与 `minFDE20`，不定义综合分数。

## 7. 一句话定位

> **本文整理形成 TrajGap 数据集，并建立 TrajGap-Bench 评测协议；在五个场景的 TrajGap validation 划分上按最终预测头 `val_new_minFDE20` 选择 checkpoint，并使用 `minADE20` 与 `minFDE20` 评估 GAST 直接利用不完整历史进行未来轨迹预测的能力。**

</details>

<details>
<summary id="english-version">English Version</summary>

[中文版](#中文版)

## 1. Overview

This work organizes the **TrajGap dataset** from the official TrajImpute release and defines **TrajGap-Bench**, an evaluation protocol for direct future trajectory forecasting from incomplete history. A method must consume the masked history directly and predict 12 future frames; external imputation followed by forecasting is reported separately.

TrajGap follows the upstream scene and train/validation/test partitions. TrajGap-Bench specifies the task, checkpoint rule, and primary metrics. It does not regenerate trajectories or claim that the missingness exactly matches real tracker failures.

### 1.1 Latest model

The complete method in this document is **GAST**:

- state strength derived from validity masks and missing duration;
- frame-wise actor interaction modeling;
- state-strength-weighted temporal aggregation with Mamba;
- endpoint-guided trajectory decoding based on the last valid observation;
- final-consistent loss and mode-diversity regularization.

Endpoint-guided decoding and trajectory-aware mode ranking are internal components of GAST, not separate model names. The repository's first-party training and evaluation entry points implement GAST only.

## 2. Dataset and task

TrajGap uses five scenes: `ETH-M`, `HOTEL-M`, `UNIV-M`, `ZARA1-M`, and `ZARA2-M`. Each sample contains 8 history frames at 0.4 s resolution (3.2 s total) and 12 complete future frames (4.8 s total). The model outputs `K=20` candidate futures. At least one history frame remains observed per sample, and `missing_mask=True` marks missing coordinates.

The dataset retains the upstream scene and train/validation/test partitions. Samples are organized around a focal actor; no re-windowing, re-sampling, de-duplication, or future-target generation is performed.

| Scene | TrajGap train | TrajGap validation | TrajGap test |
|---|---:|---:|---:|
| `ETH-M` | 59,618 | 10,698 | 1,629 |
| `HOTEL-M` | 58,304 | 10,272 | 9,477 |
| `UNIV-M` | 18,462 | 5,416 | 219,006 |
| `ZARA1-M` | 56,020 | 10,236 | 20,277 |
| `ZARA2-M` | 51,014 | 8,346 | 52,497 |
| **Total** | **243,418** | **44,968** | **302,886** |

The test set preserves the official missing-level sweep: one latent trajectory may appear with multiple masks. Counts therefore are not counts of unique source trajectories. Statistical inference should cluster by original trajectory or use paired bootstrap.

This is a controlled missing-history stress test. It should not be described as a measured real-world tracker-missingness distribution without external validation.

## 3. Evaluation protocol

- **Training:** use the TrajGap training partition.
- **Validation:** use the TrajGap validation partition for checkpoint selection.
- **Testing:** report once on the TrajGap test partition; test data must not affect tuning or model selection.
- **Checkpoint:** use the final prediction head and `val_new_minFDE20` on TrajGap validation.
- **Task:** keep the 8→12 setting, direct input, `K=20`, and five scenes consistent.
- **Primary metrics:** report `minADE20` and `minFDE20`, both lower-is-better; do not combine them into a composite score.
- **Aggregates:** report per-scene values, five-scene macro average, sample-weighted micro average, and worst-scene when all five scenes are complete.

Methods may use an internal latent completion mechanism, but external imputation before forecasting must be reported separately. Incomplete methods must not receive five-scene macro or micro averages.

## 4. Comparison experiments

The complete scene-wise comparison, macro/micro averages, ranking diagnostics, and missingness-stratified results are generated in [`docs/results/trajgap_bench_leaderboard.md`](docs/results/trajgap_bench_leaderboard.md), the single numerical source for this benchmark. The selected GAST run has matching code revisions across all five scenes, with macro `minADE20/minFDE20` of **0.3168/0.4831**. MoFlow is **0.3496/0.5227**. All six external baselines have completed all five scenes. Other results with mismatched scene revisions are not combined into a single run.

The protocol document does not duplicate static result tables. It retains model configuration, reproduction commands, and evaluation limitations.

## 5. Reproducibility and limitations

The official release files are read-only. The TrajGap data adapter, training runner, evaluator, and model configuration are:

| Function | Implementation |
|---|---|
| TrajGap dataset | `src/datamodule/trajimpute_dataset.py::TrajGapDataset` |
| Training and validation | `TrajImputeDataModule` |
| Training runner | `scripts/训练与评估/run_trajimpute_experiments.py` |
| Direct evaluator | `scripts/结果分析/evaluate_trajimpute_direct.py` |
| Latest model config | `conf/model/gast_model_forecast.yaml` |
| Benchmark config | `conf/config_missing_aware_trajimpute.yaml` |

Key limitations:

1. TrajGap is a controlled missing-history protocol, not a confirmed real-world tracker-missingness distribution.
2. The test set retains repeated latent trajectories with different masks; statistical analysis should account for this dependence.
3. Some comparison results are historical screening runs. Final publication should align model revisions, checkpoint selection, seeds, and evaluation scripts.
4. Impute-then-predict and direct prediction use different input information and should be reported separately.
5. Add multi-seed confirmation, confidence intervals, trajectory-clustered paired statistics, and external validation before making strong generalization claims.

</details>
