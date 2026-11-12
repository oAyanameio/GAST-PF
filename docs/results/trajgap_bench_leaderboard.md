# TrajGap-Bench 对比实验表

> 数据集与协议：TrajGap 数据集与 TrajGap-Bench；任务为无插补直接预测。seed 2024，K=20，100 epochs。GAST 按最终预测头 `val_new_minFDE20` 选点；外部方法沿用各自的训练与选点流程。
> 测试样本数（所有方法完全一致）：ETH-M 1,629 / HOTEL-M 9,477 / UNIV-M 219,006 / ZARA1-M 20,277 / ZARA2-M 52,497，合计 302,886。
> `Avg` = 五场景宏平均；`micro` = 汇总全部测试样本后的算术平均。所有指标**越低越好**。加粗 = 该列最优。
> 本文方法统一称为 **GAST**；终点引导解码与轨迹级模式排序属于完整方法设计。

> 本文件由 `docs/tools/gen_leaderboard_tables.py` 从各方法 `results.json` 现算生成，不手工转抄。
> 重新生成：`python docs/tools/gen_leaderboard_tables.py --write`

## 表 1 · 精度（minADE20 / minFDE20，米）

| Method | ETH-M | HOTEL-M | UNIV-M | ZARA1-M | ZARA2-M | Avg |
|---|---|---|---|---|---|---|
| GAST | **0.5852** / **0.8212** | 0.2541 / 0.3769 | **0.3200** / **0.5244** | **0.2510** / 0.4128 | **0.1738** / **0.2801** | **0.3168** / **0.4831** |
| MoFlow | 0.6819 / 0.9389 | 0.2777 / 0.4247 | 0.3478 / 0.5572 | 0.2566 / **0.4057** | 0.1839 / 0.2867 | 0.3496 / 0.5227 |
| AgentFormer | 0.7098 / 1.0436 | **0.2083** / **0.3316** | 0.4132 / 0.6796 | 0.3551 / 0.5795 | 0.2690 / 0.4375 | 0.3911 / 0.6144 |
| Re | 0.9945 / 1.4262 | 0.7799 / 1.2808 | 0.5388 / 0.8408 | 0.4960 / 0.7470 | 0.3819 / 0.5928 | 0.6382 / 0.9775 |
| MS-TIP | 1.6345 / 2.0985 | 1.3102 / 1.5472 | 1.3240 / 1.1566 | 0.8842 / 1.1720 | 0.7663 / 0.9902 | 1.1838 / 1.3929 |
| Social-STGCNN | 2.1393 / 3.5048 | 1.3022 / 2.1624 | 0.9496 / 1.5336 | 1.4801 / 2.7038 | 0.8646 / 1.5258 | 1.3471 / 2.2861 |

## 表 2 · 汇总全部样本的平均

| Method | minADE20 | minFDE20 |
|---|---|---|
| GAST | 0.2894 | 0.4716 |
| MoFlow | 0.3129 | 0.4981 |
| AgentFormer | 0.3795 | 0.6220 |
| Re | 0.5187 | 0.8085 |
| MS-TIP | 1.1991 | 1.1461 |
| Social-STGCNN | 0.9878 | 1.6409 |

## 表 3 · 缺失程度分层（宏平均 minFDE20）

| missing | GAST minFDE20 | MoFlow minFDE20 | AgentFormer minFDE20 | Re minFDE20 | MS-TIP minFDE20 | Social-STGCNN minFDE20 |
|---|---|---|---|---|---|---|
| 0 | 0.3590 | 0.3793 | 0.4575 | 0.7113 | 1.1231 | 1.3903 |
| 1 | 0.3632 | 0.3889 | 0.4811 | 0.6771 | 1.1810 | 1.4912 |
| 2 | 0.3756 | 0.3968 | 0.4726 | 0.6804 | 1.2128 | 1.6938 |
| 3 | 0.3777 | 0.4254 | 0.4856 | 0.8266 | 1.2016 | 1.9453 |
| 4 | 0.3854 | 0.4341 | 0.5012 | 0.9533 | 1.2859 | 2.2993 |
| 5 | 0.4002 | 0.4695 | 0.5395 | 1.0994 | 1.4487 | 2.6822 |
| 6 | 0.4281 | 0.5534 | 0.6118 | 1.3166 | 1.7603 | 3.1150 |
| 7 | 1.2731 | 1.2224 | 1.4787 | 1.5797 | 2.0369 | 3.6583 |

## 表注

- 六个方法均完成 ETH-M、HOTEL-M、UNIV-M、ZARA1-M、ZARA2-M 五场景。
- 只报告普通 `minADE20`、`minFDE20`；不使用模式分数加权，也不报告 top-1、MR、Brier 或综合分数。
- 表中指标按各方法 TrajGap test `results.json` 计算；宏平均为五场景等权，micro 为汇总全部测试样本后的算术平均。
- 外部模型保持其方法属性：MS-TIP 为插补感知联合预测方法，其余依各自实现处理缺失历史；榜单比较的是统一测试集上的结果，不代表内部结构均为纯 direct。
- 选点口径：GAST / AgentFormer / Re / MS-TIP / Social-STGCNN 均按各自验证集 minFDE20 选最优 checkpoint；MoFlow 沿用其官方实现原生的验证集 minADE（ADE_MIN）选点，未做改动。
- MS-TIP 五场景最优 checkpoint 均出现在 epoch 5–9（后续验证集 minFDE20 持续变差，为其联合插补目标在 direct 协议下的原生行为，非调参不足）；表中数值取各自验证最优 checkpoint。
- SocialLSTM 已移除：原实现为单模态输出，不参与本多模态榜单。

## 数据来源

| 方法 | 产物路径 |
|---|---|
| GAST | 五场景完整结果（manifest revision 一致） |
| MoFlow | `/home/lbh/MoFlow/results_trajgap/` |
| AgentFormer | `/home/lbh/AgentFormer/results/trajgap_agentformer/`（5/5） |
| Re | `/home/lbh/Re/outputs/trajgap_resonance_seed2024_gpu1/`（5/5） |
| MS-TIP | `/home/lbh/MS-TIP-TrajGap/results/`（5/5，插补感知） |
| Social-STGCNN | `/home/lbh/Social-STGCNN/results_trajgap/`（5/5） |
