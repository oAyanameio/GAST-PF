# TrajGap-Bench 对比实验表

> 数据集与基准：TrajGap-Bench；TrajImpute 官方 Easy+Hard 同一 split 合并为 Mixed，任务为无插补直接预测。seed 2024，K=20，100 epochs。GAST 按最终预测头 `val_new_minFDE20` 选点；外部方法沿用各自的训练与选点流程。
> 测试样本数（所有方法完全一致）：ETH-M 1,629 / HOTEL-M 9,477 / UNIV-M 219,006 / ZARA1-M 20,277 / ZARA2-M 52,497，合计 302,886。
> `Avg` = 五场景宏平均；`micro` = 按样本数加权。所有指标**越低越好**。加粗 = 该列最优。
> **术语统一**：全文「本文方法」均指 **GAST**；历史实验产物标签 `UT-ENC`、`GAST-PF` 和 `GAST-Goal` 仅用于路径溯源，不作为当前正式方法名称。

> 本文件由 `docs/tools/gen_leaderboard_tables.py` 从各方法 `results.json` 现算生成，不手工转抄。
> 重新生成：`python docs/tools/gen_leaderboard_tables.py --write`

## 表 0 · 方法出处与机制引入

| Method | 论文 / 来源 | 定位 | 引入机制与本文改动 |
|---|---|---|---|
| **GAST** | 本工作；编码机制引入自 Sports-Traj / UniTraj (Xu and Fu, ICLR 2025, arXiv:2405.17680)，goal 条件化机制为本文设计 | **本文方法（当前滚动基线）**（完整 GAST） | GAST 编码器（GERE + ECTA：RMCM/EWSA）+ decoder 侧 goal 端点条件化（`loss_mode=final_consistent`、goal endpoint loss + mode diversity loss）；GSM/Temporal Decay/Mamba 引自 UniTraj，不主张原创；goal 条件化与证据路由的适配为本文工作 |
| **MoFlow** | Fu, Yan, Wang, Li, Liao. CVPR 2025. arXiv:2503.09950 | 外部基线（官方实现） | 完整外部方法，作为上界参照；未做任何改动 |
| **SocialLSTM** | Alahi, Goel, Ramanathan, Robicquet, Li, Savarese. CVPR 2016 | 外部基线（经典 social pooling） | 零填充 + 缺失帧跳过 LSTM 状态更新 + 高斯混合头（K=20）；social pooling 向量化；已移除原仓库会泄漏 GT 的未来粘贴补丁 |
| **AgentFormer** | Yuan, Weng, Ou, Kitani. ICCV 2021. arXiv:2103.14023 | 外部基线（官方实现，5/5 完成） | 缺失历史以 mask-aware 场景原点填充（在 `*_scene_norm` 参照系下为 0）+ 逐帧有效性显式输入通道；解码锚点取最后有效观测。无模态概率，故 ADE@1/FDE@1 不适用 |
| **Re** | Mao et al. ICCV 2025. arXiv:2412.02447（Resonance: Learning to Predict Social-Aware Pedestrian Trajectories as Co-Vibrations） | 外部基线（官方实现，5/5 完成） | 完整外部方法；缺失历史按上游口径零填充（anchor 保持），K=20 随机采样；无概率排序头，ADE@1/FDE@1 不可引用（首候选非最优模式） |
| **MS-TIP** | Wang et al. ICML 2024（MS-TIP: Imputation Aware Pedestrian Trajectory Prediction） | 外部基线（官方实现，5/5 完成） | SAITS 插补感知联合训练（插补分支与预测分支端到端）；采样式输出，无模态概率，ADE@1/FDE@1 不适用；训练含 NaN 守卫（上游 GMM 数值不稳定，跳过脏 batch 回滚权重） |
| **Social-STGCNN** | Mohamed et al. CVPR 2020（Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction） | 外部基线（官方实现，5/5 完成） | 官方 pkl→逐帧图序列：有效帧以 obs_traj_rel 非 NaN 判定（差分口径）；缺失帧为图上幽灵节点（坐标置 0、邻接行列置 0，不收发消息），社交距离用最后有效位置；训练目标从 anchor（最后有效位置）差分；高斯头 K=20 采样，无概率排序，@1 不适用 |

## 表 1 · 精度（minADE20 / minFDE20，米）

| Method | ETH-M | HOTEL-M | UNIV-M | ZARA1-M | ZARA2-M | Avg |
|---|---|---|---|---|---|---|
| GAST | **0.5852** / **0.8212** | 0.2541 / 0.3769 | **0.3200** / **0.5244** | **0.2510** / 0.4128 | **0.1738** / **0.2801** | **0.3168** / **0.4831** |
| MoFlow | 0.6819 / 0.9389 | 0.2777 / 0.4247 | 0.3478 / 0.5572 | 0.2566 / **0.4057** | 0.1839 / 0.2867 | 0.3496 / 0.5227 |
| SocialLSTM | 0.8430 / 1.0456 | 0.2924 / 0.3871 | 0.4543 / 0.6512 | 0.4126 / 0.5650 | 0.2858 / 0.3907 | 0.4576 / 0.6079 |
| AgentFormer | 0.7098 / 1.0436 | **0.2083** / **0.3316** | 0.4132 / 0.6796 | 0.3551 / 0.5795 | 0.2690 / 0.4375 | 0.3911 / 0.6144 |
| Re | 0.9945 / 1.4262 | 0.7799 / 1.2808 | 0.5388 / 0.8408 | 0.4960 / 0.7470 | 0.3819 / 0.5928 | 0.6382 / 0.9775 |
| MS-TIP | 1.6345 / 2.0985 | 1.3102 / 1.5472 | 1.3240 / 1.1566 | 0.8842 / 1.1720 | 0.7663 / 0.9902 | 1.1838 / 1.3929 |
| Social-STGCNN | 2.1393 / 3.5048 | 1.3022 / 2.1624 | 0.9496 / 1.5336 | 1.4801 / 2.7038 | 0.8646 / 1.5258 | 1.3471 / 2.2861 |

## 表 2 · 排序质量（ADE@1 / FDE@1，米，取最高概率模式）

| Method | ETH-M | HOTEL-M | UNIV-M | ZARA1-M | ZARA2-M | Avg |
|---|---|---|---|---|---|---|
| GAST | **1.4404** / **2.5893** | **0.6112** / **1.1032** | 1.0680 / 2.0228 | **0.9554** / **1.7450** | 0.8284 / 1.4833 | **0.9807** / 1.7887 |
| MoFlow | 1.4669 / 2.5903 | 0.6818 / 1.1561 | **1.0569** / **1.8893** | 0.9863 / 1.7563 | **0.7152** / **1.2559** | 0.9814 / **1.7296** |
| SocialLSTM | 2.1039 / 3.2800 | 0.9876 / 1.5940 | 1.2964 / 2.2710 | 1.2546 / 2.1104 | 0.9964 / 1.6772 | 1.3278 / 2.1865 |
| AgentFormer | -- | -- | -- | -- | -- | -- |
| Re | -- | -- | -- | -- | -- | -- |
| MS-TIP | -- | -- | -- | -- | -- | -- |
| Social-STGCNN | -- | -- | -- | -- | -- | -- |

## 表 3 · 样本加权（micro）平均

| Method | minADE20 | minFDE20 | ADE@1 | FDE@1 |
|---|---|---|---|---|
| GAST | 0.2894 | 0.4716 | 1.0067 | 1.8850 |
| MoFlow | 0.3129 | 0.4981 | 0.9834 | 1.7515 |
| SocialLSTM | 0.4193 | 0.5941 | 1.2363 | 2.1416 |
| AgentFormer | 0.3795 | 0.6220 | -- | -- |
| Re | 0.5187 | 0.8085 | -- | -- |
| MS-TIP | 1.1991 | 1.1461 | -- | -- |
| Social-STGCNN | 0.9878 | 1.6409 | -- | -- |

## 表 4 · 缺失程度分层（宏平均 minFDE20）

| missing | GAST minFDE20 | MoFlow minFDE20 | SocialLSTM minFDE20 | AgentFormer minFDE20 | Re minFDE20 | MS-TIP minFDE20 | Social-STGCNN minFDE20 |
|---|---|---|---|---|---|---|---|
| 0 | 0.3590 | 0.3793 | 0.4231 | 0.4575 | 0.7113 | 1.1231 | 1.3903 |
| 1 | 0.3632 | 0.3889 | 0.4332 | 0.4811 | 0.6771 | 1.1810 | 1.4912 |
| 2 | 0.3756 | 0.3968 | 0.4491 | 0.4726 | 0.6804 | 1.2128 | 1.6938 |
| 3 | 0.3777 | 0.4254 | 0.4772 | 0.4856 | 0.8266 | 1.2016 | 1.9453 |
| 4 | 0.3854 | 0.4341 | 0.5228 | 0.5012 | 0.9533 | 1.2859 | 2.2993 |
| 5 | 0.4002 | 0.4695 | 0.5844 | 0.5395 | 1.0994 | 1.4487 | 2.6822 |
| 6 | 0.4281 | 0.5534 | 0.6997 | 0.6118 | 1.3166 | 1.7603 | 3.1150 |
| 7 | 1.2731 | 1.2224 | 1.3587 | 1.4787 | 1.5797 | 2.0369 | 3.6583 |

## 表注

- 七个方法均完成 ETH-M、HOTEL-M、UNIV-M、ZARA1-M、ZARA2-M 五场景；`--` 仅表示该方法没有可解释的概率 top-1 指标。
- **Re、AgentFormer、MS-TIP、Social-STGCNN 不提供可比的概率排序头**，因此 ADE@1/FDE@1 不适用；不使用其首个采样候选替代概率 top-1。
- 表中指标按各方法 Mixed-test `results.json` 计算；宏平均为五场景等权，micro 为测试样本加权。
- 主指标为 `minADE20` 与 `minFDE20`；`ADE@1/FDE@1` 仅作为有概率输出方法的排序诊断。
- 外部模型保持其方法属性：MS-TIP 为插补感知联合预测方法，其余依各自实现处理缺失历史；榜单比较的是统一测试集上的结果，不代表内部结构均为纯 direct。
- **b-minFDE20 不纳入对比**：本工作 evaluator 不产出该指标，无法对所有方法公平汇总。
- SocialLSTM 使用零填充、缺失帧跳过 LSTM 状态更新及 K=20 高斯混合采样头。

## 数据来源

| 方法 | 产物路径 |
|---|---|
| GAST | `outputs/trajgap_gast_goal_3gpu_retry/`（五场景同 revision 完整轮次） |
| MoFlow | `/home/lbh/MoFlow/results_trajimpute_mixed/` |
| SocialLSTM | `/home/lbh/SocialLSTM/trajgap_bench/artifacts/sociallstm_bench/` |
| AgentFormer | `/home/lbh/AgentFormer/results/trajgap_agentformer/`（5/5） |
| Re | `/home/lbh/Re/outputs/trajgap_resonance_seed2024_gpu1/`（5/5） |
| MS-TIP | `/home/lbh/MS-TIP-TrajGap/results/`（5/5，插补感知） |
| Social-STGCNN | `/home/lbh/Social-STGCNN/results_trajgap/`（5/5） |
