# TrajGap-Bench 对比实验表（Mixed-direct）

> 协议：TrajImpute 官方 Easy+Hard 同一 split 合并（Mixed），seed 2024，K=20，100 epochs，按最终预测头 `val_new_minFDE20` 选点。
> 测试样本数（所有方法完全一致）：ETH-M 1,629 / HOTEL-M 9,477 / UNIV-M 219,006 / ZARA1-M 20,277 / ZARA2-M 52,497，合计 302,886。
> `Avg` = 五场景宏平均；`micro` = 按样本数加权。所有指标**越低越好**。加粗 = 该列最优。
> **术语统一**：全文「本文方法」均指 **GAST-PF**（`encoder_type=gast`，`conf` 默认）；历史实验产物标签为 `UT-ENC`，2026-10-15 起代码更名为 `gast_encoder`，本文一律写作 GAST-PF。

> 本文件由 `docs/tools/gen_leaderboard_tables.py` 从各方法 `results.json` 现算生成，不手工转抄。
> 重新生成：`python docs/tools/gen_leaderboard_tables.py --write`

## 表 0 · 方法出处与机制引入

| Method | 论文 / 来源 | 定位 | 引入机制与本文改动 |
|---|---|---|---|
| **GAST-PF** | 本工作；将 Sports-Traj / UniTraj (Xu and Fu, ICLR 2025, arXiv:2405.17680) 的编码机制引入缺失历史直接预测任务并完成适配 | **本文方法**（`encoder_type=gast`，conf 默认） | 引入 UniTraj 的 GSM 逐帧空间注意力 + ghost token、Temporal Decay、双向 Mamba，适配为缺失历史直接预测编码器（GAST-PF）；decoder、loss、指标与 TrajGap 协议为本工作原有，保持不变 |
| **M0** | 本工作内部基线（`encoder_type=m0`） | 本文内部基线（历史对照） | 自有历史编码器；optimizer 分组策略 `m0_adamw_complete_module_local_v1` |
| **MoFlow** | Fu, Yan, Wang, Li, Liao. CVPR 2025. arXiv:2503.09950 | 外部基线（官方实现） | 完整外部方法，作为上界参照；未做任何改动 |
| **SocialLSTM** | Alahi, Goel, Ramanathan, Robicquet, Li, Savarese. CVPR 2016 | 外部基线（经典 social pooling） | 零填充 + 缺失帧跳过 LSTM 状态更新 + 高斯混合头（K=20）；social pooling 向量化；已移除原仓库会泄漏 GT 的未来粘贴补丁 |
| **AgentFormer** | Yuan, Weng, Ou, Kitani. ICCV 2021. arXiv:2103.14023 | 外部基线（**进行中，仅 1/5 场景**） | 缺失历史以 mask-aware 场景原点填充（在 `*_scene_norm` 参照系下为 0）+ 逐帧有效性显式输入通道；解码锚点取最后有效观测。无模态概率，故 ADE@1/FDE@1 不适用 |
| **M1** | 本工作内部（missingness-aware 输出校准） | **已判负撤回**，仅作负结果证据 | 校准头条件化调整 mode logits 温度与 Laplace scale；不修改候选轨迹坐标 |

## 表 1 · 精度（minADE20 / minFDE20，米）

| Method | ETH-M | HOTEL-M | UNIV-M | ZARA1-M | ZARA2-M | Avg |
|---|---|---|---|---|---|---|
| GAST-PF | 0.6191 / 0.8544 | 0.2572 / 0.3877 | **0.3258** / **0.5292** | 0.2494 / 0.4014 | 0.1808 / 0.3004 | 0.3264 / 0.4946 |
| M0 | **0.6145** / **0.8532** | **0.2396** / **0.3649** | 0.3473 / 0.5696 | **0.2398** / **0.3877** | **0.1700** / **0.2830** | **0.3223** / **0.4917** |
| MoFlow | 0.6819 / 0.9389 | 0.2777 / 0.4247 | 0.3478 / 0.5572 | 0.2566 / 0.4057 | 0.1839 / 0.2867 | 0.3496 / 0.5227 |
| SocialLSTM | 0.8430 / 1.0456 | 0.2924 / 0.3871 | 0.4543 / 0.6512 | 0.4126 / 0.5650 | 0.2858 / 0.3907 | 0.4576 / 0.6079 |
| AgentFormer | -- | -- | 0.4132 / 0.6796 | -- | -- | -- |

## 表 2 · 排序质量（ADE@1 / FDE@1，米，取最高概率模式）

| Method | ETH-M | HOTEL-M | UNIV-M | ZARA1-M | ZARA2-M | Avg |
|---|---|---|---|---|---|---|
| GAST-PF | 1.4232 / **2.4693** | **0.5079** / **0.8615** | **0.8615** / **1.6233** | 0.9072 / 1.6382 | 0.7483 / 1.3425 | 0.8896 / 1.5870 |
| M0 | **1.4034** / 2.4877 | 0.5278 / 0.8983 | 0.9880 / 1.8068 | **0.8368** / **1.5223** | **0.6687** / **1.2131** | **0.8849** / **1.5856** |
| MoFlow | 1.4669 / 2.5903 | 0.6818 / 1.1561 | 1.0569 / 1.8893 | 0.9863 / 1.7563 | 0.7152 / 1.2559 | 0.9814 / 1.7296 |
| SocialLSTM | 2.1039 / 3.2800 | 0.9876 / 1.5940 | 1.2964 / 2.2710 | 1.2546 / 2.1104 | 0.9964 / 1.6772 | 1.3278 / 2.1865 |
| AgentFormer | -- | -- | -- | -- | -- | -- |

## 表 3 · 样本加权（micro）平均

| Method | minADE20 | minFDE20 | ADE@1 | FDE@1 |
|---|---|---|---|---|
| GAST-PF | 0.2950 | 0.4783 | 0.8369 | 1.5564 |
| M0 | 0.3075 | 0.5029 | 0.9104 | 1.6601 |
| MoFlow | 0.3129 | 0.4981 | 0.9834 | 1.7515 |
| SocialLSTM | 0.4193 | 0.5941 | 1.2363 | 2.1416 |
| AgentFormer | -- | -- | -- | -- |

## 表 4 · 缺失程度分层（宏平均 minFDE20）

| missing | GAST-PF minFDE20 | M0 minFDE20 | MoFlow minFDE20 | SocialLSTM minFDE20 |
|---|---|---|---|---|
| 0 | 0.3584 | 0.3553 | 0.3793 | 0.4231 |
| 1 | 0.3632 | 0.3600 | 0.3889 | 0.4332 |
| 2 | 0.3773 | 0.3777 | 0.3968 | 0.4491 |
| 3 | 0.3920 | 0.3809 | 0.4254 | 0.4772 |
| 4 | 0.3943 | 0.3831 | 0.4341 | 0.5228 |
| 5 | 0.4157 | 0.4163 | 0.4695 | 0.5844 |
| 6 | 0.4809 | 0.4616 | 0.5534 | 0.6997 |
| 7 | 1.2755 | 1.3073 | 1.2224 | 1.3587 |

## 表注

- **`--` = 该场景/指标不可用**。AgentFormer 仅完成 UNIV-M 一场景，故其 `Avg` 与 `micro` 均为 `--`——**单场景值不得当宏平均报**；其 ADE@1/FDE@1 不适用（采样器无模态概率）。
- **本表当前数值仅为历史结果，不是 TrajGap-Bench v1.1 正式成绩**：GAST-PF 的 checkpoint 来自旧选点口径，尚须按最终预测头 `val_new_minFDE20` 重新训练、评估后才能定稿。
- 正式指标为 `minADE20` 和 `minFDE20`；ADE@1/FDE@1 仅作辅助诊断。不计算 MR 或综合分数。
- **M1 已判负撤回**（2026-10-11），仅作负结果证据，不进主表。
- **M0-pre-fix 不混入**：旧 revision `a4f1803` 的宏平均为 0.3249 / 0.5039，与本表 M0（0.3223 / 0.4917）不同定义，论文中只能有一个 M0。
- **b-minFDE20 无法对齐**：本工作 direct evaluator 不产出该项，仅 MoFlow 侧有 test 值（宏平均 1.2945），故未入表。
- 部分场景的平均值与五场景宏平均**不可直接比较**。
- SocialLSTM 使用 missing-aware 变体：零填充输入 + 缺失帧跳过 LSTM 状态更新 + 高斯混合头（K=20，按分量采样），social pooling 已向量化。

## 数据来源

| 方法 | 产物路径 |
|---|---|
| GAST-PF | `outputs/unitraj_encoder_single_scene_seed2024/UT-ENC_ETH-M_mixed-direct_seed2024_uni/`（ETH-M）；`outputs/unitraj_encoder_bench_seed2024/lane_{a,b,d}/`（其余四场景） |
| M0 | `outputs/m0_baseline/M0_optimizerfix_seed2024_gpu{0,1}/` |
| MoFlow | `MoFlow/results_trajimpute_mixed/<scene>/mixed/results.json` |
| SocialLSTM | `/home/lbh/SocialLSTM/trajgap_bench/artifacts/sociallstm_bench/` |
| AgentFormer | `/home/lbh/AgentFormer/results/trajgap_agentformer/`（**进行中，1/5**） |
| M1（撤回） | `outputs/trajgap_plan/M1_calibration_seed2024/`（负结果证据保留） |

> 归档（2026-10-14）：SocialLSTM 代码与全部产物已迁至 `/home/lbh/SocialLSTM/trajgap_bench/`，并从本仓库 runner/evaluator 中移除该变体。
> 更名（2026-10-15）：引入自 UniTraj `Past_Encoder` 的编码器由 `unitraj_encoder` 更名为 `gast_encoder`（`GASTEncoder`）；`conf` 中 `encoder_type` 默认已是 `gast`，`m0` 仅作历史对照入口。产物目录名沿用历史标签 `UT-ENC`（仅路径，非方法名），本文一律称 GAST-PF。
