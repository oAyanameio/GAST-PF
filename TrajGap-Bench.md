# TrajGap-Bench

[中文版](#中文版) ｜ [English Version](#english-version)

<details open>
<summary id="中文版">中文版</summary>

[English version](#english-version)

## 1. 基准简介

**TrajGap-Bench** 是一个面向「缺失历史条件下直接未来预测」的行人轨迹预测基准。它不重新生成轨迹，也不把缺失历史先交给外部插补器，而是要求模型直接读取带缺失掩码的历史观测，并预测未来 12 帧轨迹。

本基准的底层数据来自 TrajImpute 官方 Easy/Hard release。TrajGap-Bench 的贡献不是重新发明数据，而是将官方数据整理为统一、可复现、无测试泄漏的 Mixed direct 评估协议，并冻结训练、选点和结果汇报规则。

### 1.1 当前本文模型

当前本文模型使用最新的 **GAST-Goal**，其底座为正式命名的 **GAST（Gap-Aware Spatio-Temporal Pedestrian Forecaster）编码器**：

```text
GAST-Goal
├── GAST gap-aware encoder
│   ├── 单向 Mamba 时序编码
│   ├── OER 空间证据路由
│   ├── EMC 时间证据调制
│   └── EWHR 证据加权读出
└── Goal-conditioned future decoder
    ├── 终点条件化
    ├── final-consistent loss
    └── mode diversity regularization
```

旧实验中的 `GAST-PF`、`GAST-PF-UniOER` 和 `UT-ENC` 是历史标签或产物目录名。本文统一将当前编码器称为 `GAST`，将最新完整模型称为 `GAST-Goal`，避免把旧标签误解为另一套方法。

### 1.2 研究问题

完整历史预测假设所有观测均可用，但真实跟踪系统可能受到遮挡、漏检、数据关联失败和传感器丢帧影响。TrajGap-Bench 关注以下问题：

1. 模型能否直接利用不完整历史，而不是依赖外部插补？
2. 在相同缺失数量下，缺失位置和时间结构是否会改变预测难度？
3. 模型性能是否能在可观测证据减少时平稳退化？
4. goal 条件化是否能改善缺失历史下的多模态未来覆盖？

## 2. 数据集与任务

### 2.1 数据来源

TrajGap 使用 TrajImpute 官方 release：

- 场景：`ETH-M`、`HOTEL-M`、`UNIV-M`、`ZARA1-M`、`ZARA2-M`；
- 底层文件：每个场景的 `Easy` 与 `Hard`；
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

### 2.2 Mixed 合并协议

对每个场景和每个 split，TrajGap 只进行样本级拼接：

```text
Mixed/train = Easy/train + Hard/train
Mixed/val   = Easy/val   + Hard/val
Mixed/test  = Easy/test  + Hard/test
```

这里的 `+` 表示 actor-centric 样本拼接，不表示重新切窗、重采样或去重。Easy/Hard 标签仅用于内部审计，对外统一称为 `Mixed`。

采用 Mixed 的理由是：

- 训练、验证和测试使用同一个公开协议；
- checkpoint 只由一个 Mixed-val 决定；
- 模型不能依赖 Easy/Hard 标签获得额外信息；
- 主榜不需要人为定义 Easy/Hard 权重；
- 缺失程度仍可作为测试后的诊断分组，而不改变主协议。

Mixed 的使用不等于声称数据完全服从真实跟踪器的缺失分布。它是一个**受控的缺失历史压力测试协议**；真实分布外推需要额外数据或外部验证。

### 2.3 样本规模

| 场景 | Mixed train | Mixed val | Mixed test |
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
| Train | `Mixed-train` | 模型训练 |
| Validation | `Mixed-val` | 只用于 checkpoint 选择 |
| Test | `Mixed-test` | 最终报告，不参与调参 |

所有方法必须遵守相同的：

- 8 帧历史 → 12 帧未来任务；
- `Mixed` train/val/test；
- direct 输入约束；
- `K=20` 候选轨迹；
- 五场景评估；
- 测试集不参与模型选择。

模型可以在内部学习隐式补全或 latent state，但不允许依赖一个外部插补模型作为预测前置步骤。先插补后预测方法应作为单独口径报告，不能直接进入 direct leaderboard。

### 3.2 Checkpoint 选择

Checkpoint 规则固定为：

```text
monitor = val_new_minFDE20
source  = Mixed-val
head    = final prediction head
mode    = K=20
```

不允许：

- 分别在 Easy-val 和 Hard-val 上选点；
- 使用测试结果选择 epoch；
- 根据单个场景结果挑选 checkpoint；
- 使用旧预测头或不同指标替代 `val_new_minFDE20`。

### 3.3 主指标

TrajGap-Bench 正式主榜只报告两个指标：

| 指标 | 定义 | 趋势 |
|---|---|---|
| `minADE20` | 20 条候选轨迹中最小的平均位移误差 | 越低越好 |
| `minFDE20` | 20 条候选轨迹中最小的终点位移误差 | 越低越好 |

两个指标必须来自同一批 `K=20` 候选轨迹、同一个 checkpoint 和同一个 Mixed-test。不计算综合分数。

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

下面的表格汇总当前仓库中已经完成或已有可审计产物的 direct 对比结果，单位为米，单元格格式为 `minADE20 / minFDE20`，所有指标均越低越好。

**本文最新模型为 `GAST-Goal`，使用 `GAST` 编码器作为底座。** 表中同时保留标准解码器版本 `GAST`（历史实验标签为 `GAST-PF`）作为上一代内部对照；`MoFlow`、`SocialLSTM`、`AgentFormer`、`Re` 和 `MS-TIP` 是外部对照或外部复现结果。

需要注意：`GAST-Goal` 五场景结果已经按最终预测头 `val_new_minFDE20` 完成；部分历史对照仍来自早期 screening 产物，正式投稿前应将所有方法统一重跑到同一 checkpoint 规则。表格因此用于当前研究对照，不应在重跑前称为最终 v1.1 leaderboard。

### 4.2 五场景对照表

| 方法 | `ETH-M` | `HOTEL-M` | `UNIV-M` | `ZARA1-M` | `ZARA2-M` | Macro Avg | 完成度 |
|---|---|---|---|---|---|---|---|
| **GAST-Goal（本文最新模型）** | **0.5852 / 0.8212** | 0.2541 / 0.3769 | **0.3200 / 0.5244** | 0.2510 / 0.4128 | 0.1738 / **0.2801** | **0.3168 / 0.4831** | 5/5 |
| GAST（标准解码器；历史标签 GAST-PF） | 0.6191 / 0.8544 | 0.2572 / 0.3877 | 0.3258 / 0.5292 | **0.2494 / 0.4014** | 0.1808 / 0.3004 | 0.3264 / 0.4946 | 5/5 |
| MoFlow | 0.6819 / 0.9389 | 0.2777 / 0.4247 | 0.3478 / 0.5572 | 0.2566 / 0.4057 | 0.1839 / 0.2867 | 0.3496 / 0.5227 | 5/5 |
| SocialLSTM | 0.8430 / 1.0456 | 0.2924 / 0.3871 | 0.4543 / 0.6512 | 0.4126 / 0.5650 | 0.2858 / 0.3907 | 0.4576 / 0.6079 | 5/5 |
| AgentFormer | 0.7098 / 1.0436 | **0.2083 / 0.3316** | 0.4132 / 0.6796 | — | — | — | 3/5 |
| Re | 0.9945 / 1.4262 | 0.7799 / 1.2808 | 0.5388 / 0.8408 | 0.4960 / 0.7470 | 0.3819 / 0.5928 | 0.6382 / 0.9775 | 5/5 |
| MS-TIP | — | — | 1.3240 / 1.1566 | 0.8842 / 1.1720 | — | — | 2/5 |

每个单元格按 `minADE20 / minFDE20` 排列，单位为米。**加粗规则：** 对每个场景分别在所有已报告结果中对两个指标独立取最优；macro average 只比较完成五场景的方法。部分完成的方法不计算 macro/micro，也不参与完整五场景排序。`—` 表示暂无可用于本表的结果，不表示性能为 0。

| 方法 | Macro `minADE20` | Macro `minFDE20` | Micro `minADE20` | Micro `minFDE20` |
|---|---:|---:|---:|---:|
| **GAST-Goal（本文最新模型）** | **0.3168** | **0.4831** | **0.2894** | **0.4716** |
| GAST（标准解码器；历史标签 GAST-PF） | 0.3264 | 0.4946 | 0.2950 | 0.4783 |
| MoFlow | 0.3496 | 0.5227 | 0.3129 | 0.4981 |
| SocialLSTM | 0.4576 | 0.6079 | 0.4193 | 0.5941 |
| AgentFormer | — | — | — | — |
| Re | 0.6382 | 0.9775 | 0.5187 | 0.8085 |
| MS-TIP | — | — | — | — |

Micro average 按测试样本数加权。结果来源：`GAST-Goal` 为 `outputs/trajgap_gast_goal_3gpu_retry/` 下五场景 Mixed-test 结果；其余模型数字来自仓库已归档的 `docs/results/trajgap_bench_leaderboard.md`。其中未完成五场景的方法不计算聚合结果。

### 4.3 本文模型配置

| 项目 | `GAST-Goal` |
|---|---|
| Encoder | `GAST`，单向 gap-aware Mamba |
| Evidence routing | OER + EMC + EWHR |
| Decoder | Goal-conditioned future decoder |
| Prediction modes | `K=20` |
| Loss | `final_consistent` |
| Checkpoint | Mixed-val `val_new_minFDE20` |
| Seed | 2024 |
| Epochs | 100 |
| Batch size | 64 |

当前 `GAST-Goal` 的五场景结果为单种子 screening 结果。它可作为最新滚动基线，但定稿时仍应补充第二随机种子、置信区间以及按原始轨迹聚类的配对统计。

## 5. 代码与复现

### 5.1 训练

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. \
/home/lbh/.conda/envs/gast-pf/bin/python \
scripts/训练与评估/run_trajimpute_experiments.py \
  --protocol mixed-direct \
  --variant GAST-Goal \
  --scenes ETH-M HOTEL-M UNIV-M ZARA1-M ZARA2-M \
  --seed 2024 \
  --gpu 0 \
  --output-root outputs/trajgap_bench_gast_goal_seed2024
```

### 5.2 评估

```bash
/home/lbh/.conda/envs/gast-pf/bin/python \
scripts/结果分析/evaluate_trajimpute_direct.py \
  --data-root /home/lbh/TrajImpute/dataset/TrajImpute \
  --scene ETH-M \
  --difficulty Mixed \
  --split test \
  --variant GAST-Goal \
  --K 20 \
  --seed 2024 \
  --checkpoint <path/to/checkpoint.ckpt> \
  --output-root <output_dir>
```

### 5.3 关键实现

| 功能 | 实现 |
|---|---|
| Mixed 数据集 | `src/datamodule/trajimpute_dataset.py::TrajGapDataset` |
| Mixed 训练/验证 | `TrajImputeDataModule` |
| 统一训练入口 | `scripts/训练与评估/run_trajimpute_experiments.py` |
| 统一 direct 评估 | `scripts/结果分析/evaluate_trajimpute_direct.py` |
| 最新模型配置 | `conf/model/gast_goal_model_forecast.yaml` |
| 主协议配置 | `conf/config_missing_aware_trajimpute.yaml` |

## 6. 限制与论文使用边界

1. TrajGap 是基于官方 release 的受控缺失协议，不应直接宣称等同于真实跟踪器的缺失分布。
2. 测试集保留同一潜在轨迹的多种缺失掩码，这是缺失程度扫描的设计，不应简单去重后再作为主结果。
3. 当前部分基线仍是历史 screening 结果；最终论文表应统一模型版本、checkpoint 规则、随机种子和评估脚本。
4. 先插补后预测与 direct prediction 输入信息不同，应分表报告，不能直接混入同一 leaderboard。
5. `GAST-Goal` 当前优势主要体现在 `minADE20` 与 `minFDE20`；其 mode 排序仍需通过第二种子和额外诊断确认，不能仅凭单次运行作强泛化结论。
6. 当前正式指标只包括 `minADE20` 与 `minFDE20`，不定义综合分数。

## 7. 一句话定位

> **TrajGap-Bench 将 TrajImpute 官方 Easy/Hard 的同一 train/val/test split 统一为 Mixed direct 协议，并以最终预测头的 `val_new_minFDE20` 选择 checkpoint，在五个场景上使用 `minADE20` 与 `minFDE20` 评估模型直接利用不完整历史进行未来轨迹预测的能力；本文最新模型为以 GAST 为底座的 GAST-Goal。**

</details>

<details>
<summary id="english-version">English Version</summary>

[中文版](#中文版)

## 1. Overview

**TrajGap-Bench** is a benchmark for direct future trajectory forecasting from incomplete history. A method must consume the masked history directly and predict 12 future frames. External imputation followed by forecasting is reported separately and is not part of the direct leaderboard.

The benchmark reuses the official TrajImpute Easy/Hard release. Its contribution is a fixed Mixed protocol with unified train/validation/test construction, a single checkpoint rule, and two transparent primary metrics. It does not regenerate trajectories or claim that the synthetic missingness exactly matches real tracker failures.

### 1.1 Latest model

The latest model in this document is **GAST-Goal**, built on the current **GAST** gap-aware encoder:

- unidirectional Mamba temporal encoding;
- OER spatial evidence routing;
- EMC temporal evidence modulation;
- EWHR evidence-weighted readout;
- goal-conditioned future decoding;
- final-consistent loss and mode-diversity regularization.

`GAST-PF`, `GAST-PF-UniOER`, and `UT-ENC` are legacy names used by earlier experiments or output directories. The current terminology is `GAST` for the encoder and `GAST-Goal` for the complete latest model.

## 2. Dataset and task

TrajGap uses five scenes: `ETH-M`, `HOTEL-M`, `UNIV-M`, `ZARA1-M`, and `ZARA2-M`. Each sample contains 8 history frames at 0.4 s resolution (3.2 s total) and 12 complete future frames (4.8 s total). The model outputs `K=20` candidate futures. At least one history frame remains observed per sample, and `missing_mask=True` marks missing coordinates.

For every scene and split:

```text
Mixed/train = Easy/train + Hard/train
Mixed/val   = Easy/val   + Hard/val
Mixed/test  = Easy/test  + Hard/test
```

The merge is actor-centric sample concatenation. There is no re-windowing, re-sampling, or de-duplication. Easy/Hard labels are retained for audit only; the public protocol is `Mixed`.

| Scene | Mixed train | Mixed val | Mixed test |
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

- **Training:** use `Mixed-train`.
- **Validation:** use one `Mixed-val` set for checkpoint selection.
- **Testing:** report once on `Mixed-test`; test data must not affect tuning or model selection.
- **Checkpoint:** use the final prediction head and `val_new_minFDE20` on `Mixed-val`.
- **Task:** keep the 8→12 setting, direct input, `K=20`, and five scenes consistent.
- **Primary metrics:** report `minADE20` and `minFDE20`, both lower-is-better; do not combine them into a composite score.
- **Aggregates:** report per-scene values, five-scene macro average, sample-weighted micro average, and worst-scene when all five scenes are complete.

Methods may use an internal latent completion mechanism, but external imputation before forecasting must be reported separately. Incomplete methods must not receive five-scene macro or micro averages.

## 4. Comparison experiments

The latest model is `GAST-Goal`, with `GAST` as its base encoder. Each table cell is `minADE20 / minFDE20` in meters. Lower is better; bold indicates the best available value per scene and metric. Macro averages compare only methods with all five scenes completed.

| Method | `ETH-M` | `HOTEL-M` | `UNIV-M` | `ZARA1-M` | `ZARA2-M` | Macro Avg | Coverage |
|---|---|---|---|---|---|---|---|
| **GAST-Goal (latest model)** | **0.5852 / 0.8212** | 0.2541 / 0.3769 | **0.3200 / 0.5244** | 0.2510 / 0.4128 | 0.1738 / **0.2801** | **0.3168 / 0.4831** | 5/5 |
| GAST (standard decoder; historical label: GAST-PF) | 0.6191 / 0.8544 | 0.2572 / 0.3877 | 0.3258 / 0.5292 | **0.2494 / 0.4014** | 0.1808 / 0.3004 | 0.3264 / 0.4946 | 5/5 |
| MoFlow | 0.6819 / 0.9389 | 0.2777 / 0.4247 | 0.3478 / 0.5572 | 0.2566 / 0.4057 | 0.1839 / 0.2867 | 0.3496 / 0.5227 | 5/5 |
| SocialLSTM | 0.8430 / 1.0456 | 0.2924 / 0.3871 | 0.4543 / 0.6512 | 0.4126 / 0.5650 | 0.2858 / 0.3907 | 0.4576 / 0.6079 | 5/5 |
| AgentFormer | 0.7098 / 1.0436 | **0.2083 / 0.3316** | 0.4132 / 0.6796 | — | — | — | 3/5 |
| Re | 0.9945 / 1.4262 | 0.7799 / 1.2808 | 0.5388 / 0.8408 | 0.4960 / 0.7470 | 0.3819 / 0.5928 | 0.6382 / 0.9775 | 5/5 |
| MS-TIP | — | — | 1.3240 / 1.1566 | 0.8842 / 1.1720 | — | — | 2/5 |

| Method | Macro `minADE20` | Macro `minFDE20` | Micro `minADE20` | Micro `minFDE20` |
|---|---:|---:|---:|---:|
| **GAST-Goal (latest model)** | **0.3168** | **0.4831** | **0.2894** | **0.4716** |
| GAST (standard decoder; historical label: GAST-PF) | 0.3264 | 0.4946 | 0.2950 | 0.4783 |
| MoFlow | 0.3496 | 0.5227 | 0.3129 | 0.4981 |
| SocialLSTM | 0.4576 | 0.6079 | 0.4193 | 0.5941 |
| AgentFormer | — | — | — | — |
| Re | 0.6382 | 0.9775 | 0.5187 | 0.8085 |
| MS-TIP | — | — | — | — |

`GAST-Goal` uses seed 2024, 100 epochs, batch size 64, and the final-head checkpoint rule. Its five-scene numbers are single-seed screening results. The `GAST-Goal` results come from `outputs/trajgap_gast_goal_3gpu_retry/`; the other results are archived comparison runs. They should not be presented as a final v1.1 leaderboard until historical baselines are regenerated under the frozen checkpoint rule.

## 5. Reproducibility and limitations

The official release files are read-only. The Mixed adapter, training runner, evaluator, and model configuration are:

| Function | Implementation |
|---|---|
| Mixed dataset | `src/datamodule/trajimpute_dataset.py::TrajGapDataset` |
| Training and validation | `TrajImputeDataModule` |
| Training runner | `scripts/训练与评估/run_trajimpute_experiments.py` |
| Direct evaluator | `scripts/结果分析/evaluate_trajimpute_direct.py` |
| Latest model config | `conf/model/gast_goal_model_forecast.yaml` |
| Benchmark config | `conf/config_missing_aware_trajimpute.yaml` |

Key limitations:

1. TrajGap is a controlled missing-history protocol, not a confirmed real-world tracker-missingness distribution.
2. The test set retains repeated latent trajectories with different masks; statistical analysis should account for this dependence.
3. Some comparison results are historical screening runs. Final publication should align model revisions, checkpoint selection, seeds, and evaluation scripts.
4. Impute-then-predict and direct prediction use different input information and should be reported separately.
5. Add multi-seed confirmation, confidence intervals, trajectory-clustered paired statistics, and external validation before making strong generalization claims.

</details>
