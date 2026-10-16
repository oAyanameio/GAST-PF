# TrajGap-Bench 对比实验表（Mixed-direct）

> 协议：TrajImpute 官方 Easy+Hard 同一 split 合并（Mixed），seed 2024，K=20，100 epochs，`val_minFDE20` 选点。
> 测试样本数（所有方法完全一致）：ETH-M 1,629 / HOTEL-M 9,477 / UNIV-M 219,006 / ZARA1-M 20,277 / ZARA2-M 52,497，合计 302,886。
> `Avg` = 五场景宏平均。所有指标**越低越好**。

## 表 1 · 精度（minADE20 / minFDE20，米）

| Method | ETH-M | HOTEL-M | UNIV-M | ZARA1-M | ZARA2-M | Avg |
|---|---|---|---|---|---|---|
| M0 (ours, reference) | 0.6145 / **0.8532** | **0.2396** / 0.3649 | **0.3473** / 0.5696 | **0.2398** / **0.3877** | **0.1700** / **0.2830** | **0.3222 / 0.4917** |
| MoFlow (CVPR'25) | 0.6819 / 0.9389 | 0.2777 / 0.4247 | 0.3478 / **0.5572** | 0.2566 / 0.4057 | 0.1839 / 0.2867 | 0.3496 / 0.5226 |
| SocialLSTM | 0.8430 / 1.0456 | 0.2924 / 0.3871 | 0.4543 / 0.6512 | 0.4126 / 0.5650 | 0.2858 / 0.3907 | 0.4576 / 0.6079 |

## 表 2 · 排序质量（ADE@1 / FDE@1，米，取最高概率模式）

| Method | ETH-M | HOTEL-M | UNIV-M | ZARA1-M | ZARA2-M | Avg |
|---|---|---|---|---|---|---|
| M0 (ours, reference) | 1.4034 / 2.4877 | **0.5278 / 0.8983** | 0.9880 / 1.8068 | **0.8368 / 1.5223** | **0.6687 / 1.2131** | **0.8849 / 1.5856** |
| MoFlow (CVPR'25) | 1.4669 / 2.5903 | 0.6818 / 1.1561 | 1.0569 / 1.8893 | 0.9863 / 1.7563 | 0.7152 / 1.2559 | 0.9814 / 1.7296 |
| SocialLSTM | 2.1039 / 3.2800 | 0.9876 / 1.5940 | 1.2964 / 2.2710 | 1.2546 / 2.1104 | 0.9964 / 1.6772 | 1.3278 / 2.1865 |

## 表 3 · 缺失率（MR，%，末帧误差全部 > 2.0 m 才计一次 miss）

| Method | ETH-M | HOTEL-M | UNIV-M | ZARA1-M | ZARA2-M | Avg |
|---|---|---|---|---|---|---|
| M0 (ours, reference) | **11.97** | 3.85 | 4.38 | 1.63 | 1.50 | 4.67 |
| MoFlow (CVPR'25) | 14.30 | 3.87 | **3.35** | **0.85** | **0.61** | **4.60** |
| SocialLSTM | 15.84 | **3.66** | 4.84 | 3.03 | 2.67 | 6.01 |

## 表 4 · 样本加权（micro）平均

| Method | minADE20 | minFDE20 | ADE@1 | FDE@1 | MR |
|---|---|---|---|---|---|
| M0 (ours, reference) | **0.3075** | **0.5029** | **0.9104** | **1.6601** | **3.72** |
| SocialLSTM | 0.4193 | 0.5941 | 1.2363 | 2.1416 | 4.36 |

## 表注

- 部分场景的平均值与五场景宏平均**不可直接比较**；`--` = 尚未评估。
- SocialLSTM 使用 missing-aware 变体（`config_missing_aware_sociallstm`）：零填充输入 + 缺失帧跳过 LSTM 状态更新 + 高斯混合头（K=20，按分量采样），social pooling 已向量化。
- **b-minFDE20 无法对齐**：DeMo 的 direct evaluator 不产出该项，仅 MoFlow 侧有 test 值（宏平均 1.2945），故未入表。
- 加粗 = 该列最优。

## 5 · UT-ENC screening（旧 monitor 选点，非最终定稿）

> 借用 UniTraj `Past_Encoder` 的完整 encoder replacement；实验标签仍为 `UT-ENC`，当前代码名为 GAST Encoder。协议与本表一致：Mixed-direct、seed=2024、K=20、100 epochs。**本节不作为最终主榜**：训练/runner 按初始头 `val_minFDE20` 选 checkpoint，但 test 使用最终头 `new_y_hat/new_pi`；五个场景的 `val_new_minFDE20` 最佳 epoch 均与实际 test checkpoint 不同。详见 `docs/results/实验总汇总.md` §13。

### UT-ENC 原始 test 指标

| Method | ETH-M | HOTEL-M | UNIV-M | ZARA1-M | ZARA2-M | Avg |
|---|---|---|---|---|---|---|
| UT-ENC（screening） minADE20 / minFDE20 | 0.6191 / 0.8544 | 0.2572 / 0.3877 | 0.3258 / 0.5292 | 0.2494 / 0.4014 | 0.1808 / 0.3004 | 0.3264 / 0.4946 |
| UT-ENC（screening） ADE@1 / FDE@1 | 1.4232 / 2.4693 | 0.5079 / 0.8615 | 0.8615 / 1.6233 | 0.9072 / 1.6382 | 0.7483 / 1.3425 | 0.8896 / 1.5870 |
| UT-ENC（screening） MR | 11.91% | 4.12% | 3.65% | 1.51% | 1.32% | 4.50% |

样本加权 micro：minADE20 0.2950，minFDE20 0.4783，ADE@1 0.8369，FDE@1 1.5564，MR 3.16%。与外部模型的宏平均相对差：相对 MoFlow 为 −6.62% / −5.37% / −9.36% / −8.25% / −2.07%；相对 SocialLSTM 为 −28.67% / −18.64% / −33.00% / −27.42% / −25.06%。

## 数据来源

| 方法 | 产物路径 |
|---|---|
| M0 | `outputs/m0_baseline/M0_optimizerfix_seed2024_gpu{0,1}/` |
| MoFlow | `MoFlow/results_trajimpute_mixed/<scene>/mixed/results.json` |
| SocialLSTM | `/home/lbh/SocialLSTM/trajgap_bench/artifacts/sociallstm_bench/` |

> 状态（2026-10-13）：SocialLSTM 五场景 100 epochs 训练 + Mixed-test 评估全部完成，`val_minFDE20` best epoch 为 ETH-M 63 / HOTEL-M 72 / UNIV-M 42 / ZARA1-M 75 / ZARA2-M 55。SocialLSTM 五场景口径与 M0 完全一致，可直接比较。
> 归档（2026-10-14）：SocialLSTM 相关代码（`src/model/social_lstm.py`、`conf/config_missing_aware_sociallstm.yaml`、`conf/model/social_lstm_trajimpute.yaml`）与全部产物已迁至 `/home/lbh/SocialLSTM/trajgap_bench/`，并从 DeMo 的 runner/evaluator 中移除该变体；下表数值仍为迁出前的正式结果，不受影响。