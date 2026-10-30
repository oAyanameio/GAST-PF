# TrajGap-Bench：统一混合缺失历史直接预测 Benchmark 设计

> 文档性质：Benchmark（基准）设计与当前实施协议
>
> 当前状态：统一 Mixed 方案，暂不加入其他扩展轨道
>
> 数据集与 Benchmark 名称：**TrajGap-Bench**（基于 TrajImpute 官方数据整理的缺失历史直接预测数据集与评估基准）
>
> 底层数据：TrajImpute 官方 Easy/Hard release
>
> 研究任务：缺失历史条件下的直接未来轨迹预测（direct / imputation-free trajectory forecasting）

## 1. 当前协议定位

TrajGap-Bench 对外不再区分 Easy、Hard 或 Clean。TrajImpute 仍是底层数据来源，但 Easy/Hard 目录只作为文件来源；训练、验证和测试均由 TrajGap 适配器合并为一个 Mixed 数据集。

本阶段只落实以下设想：

1. 将官方 Easy/Hard 的同一 split 合并为 Mixed；
2. 训练、验证和测试均使用 Mixed 口径；
3. 验证集只使用一个 Mixed 验证集进行 checkpoint 选点；
4. 主评估只报告一个 Mixed 结果，使用 `minADE20` 和 `minFDE20` 两个主指标，不计算综合总榜分数；
5. 暂不加入 unseen-pattern、额外数据轨道、Clean 主分支或额外评价指标。

TrajGap-Bench 不修改官方 pkl，不重新切窗，也不重新生成未来目标；只在适配层进行样本级合并。官方 impute-then-predict 数字不进入 TrajGap-Bench direct 总榜。

## 2. 数据集组织：统一 Mixed split

对每个场景 `scene ∈ {ETH-M, HOTEL-M, UNIV-M, ZARA1-M, ZARA2-M}`，TrajGap-Bench 使用官方 release 的同一 split 合并：

```text
TrajGap-Bench/<scene>/train = Easy/data_train.pkl + Hard/data_train.pkl
TrajGap-Bench/<scene>/val   = Easy/data_val.pkl   + Hard/data_val.pkl
TrajGap-Bench/<scene>/test  = Easy/data_test.pkl  + Hard/data_test.pkl
```

这里的 `+` 是 actor-centric 样本级拼接，不是重新切窗，也不是去重后重采样。Easy/Hard 的来源标签保留在内部审计信息中，对外协议名称统一为 `Mixed`。

### 2.1 训练集

```text
Mixed-train = Easy/train + Hard/train
```

当前采用自然拼接（natural Mixed），不使用 severity-balanced 或 evidence-balanced 采样。两个底层数据源沿用官方场景训练边界，合并不改变场景划分。

### 2.2 验证集

```text
Mixed-val = Easy/val + Hard/val
```

验证集不再拆成 Easy-val 和 Hard-val，也不对两部分分别选 checkpoint。训练期间只计算一个最终预测头 `val_new_minFDE20`，根据合并后的 Mixed-val 选择最佳 checkpoint。

这样可以保证：

1. checkpoint 选择目标与 Mixed 主测试口径一致；
2. 不因分别查看 Easy/Hard 验证结果而产生多重选点；
3. 不使用测试集决定 Easy/Hard 权重或 checkpoint；
4. 缺失程度差异只作为测试结果的诊断分组，不改变主验证协议。

### 2.3 测试集

```text
Mixed-test = Easy/test + Hard/test
```

评估器只输出一套 Mixed 主结果。`missing_count`、`valid_count`、`anchor_lag`、`forecast_gap` 等字段仍保留用于内部诊断，但不再输出 Easy 与 Hard 两个主榜。

当前已审计的 focal 样本规模如下：

| 场景 | Mixed train | Mixed val | Mixed test |
|---|---:|---:|---:|
| ETH-M | 59,618 | 10,698 | 1,629 |
| HOTEL-M | 58,304 | 10,272 | 9,477 |
| UNIV-M | 18,462 | 5,416 | 219,006 |
| ZARA1-M | 56,020 | 10,236 | 20,277 |
| ZARA2-M | 51,014 | 8,346 | 52,497 |

这些数量是官方 Easy/Hard 文件的样本级相加；测试集中的缺失块重复结构保持不变，不将其误称为独立原始轨迹数量。

### 2.4 数据适配规则

- 读取官方 Easy/Hard pkl，不修改源文件；
- 历史缺失坐标仍由 `missing_mask` 定义；
- 未来 `pred_traj` 保持完整，只用于监督和评估；
- 对 Easy/Hard 文件分别修复已知的 `seq_start_end` block offset 问题，再进行样本合并；
- focal actor、邻居分组、局部坐标和跨缺口运动特征沿用原适配器；
- 对外 `scene_id` 使用 `scene-Mixed-split-*`；
- 不把 Easy 的零缺失副本定义为 Clean；本阶段不建立独立 Clean 主分支。

## 3. 训练、验证、测试流程

```text
Easy/data_train.pkl ─┐
                     ├─> TrajGap Mixed-train ─┐
Hard/data_train.pkl ─┘                       │
                                             ├─> 模型训练
Easy/data_val.pkl   ─┐                       │
                     ├─> TrajGap Mixed-val ──┘
Hard/data_val.pkl   ─┘                       │
                                             └─> val_new_minFDE20 选 checkpoint

Easy/data_test.pkl  ─┐
                     ├─> TrajGap Mixed-test ─> 单一主结果 / 单一总榜
Hard/data_test.pkl  ─┘
```

训练、验证和测试使用相同的 8 帧历史、12 帧未来、direct 输入和 `K=20` 设置。测试集不参与 checkpoint 选择。

## 4. 评估指标与结果汇报

当前主榜只纳入两个指标：

```text
minADE20
minFDE20
```

三者均为越低越好：

| 指标 | 含义 |
|---|---|
| `minFDE20` | 20 条候选轨迹中最小的终点位移误差 |
| `minADE20` | 20 条候选轨迹中最小的平均位移误差 |

minADE 与 minFDE 分别在各自误差上取最优候选，该候选可以不同。结果汇报规则：

1. 两个指标来自同一 Mixed-test、同一 checkpoint、同一场景协议和同一 `K=20`；
2. 报告五场景明细、macro average、按样本数加权的 micro average 和 worst-scene；
3. 不定义综合分数或单一总排名，避免引入未冻结的归一化边界与指标权重。

`ADE@1`、`FDE@1`、`b-minFDE20`、Energy Score、coverage、calibration 等不进入当前主榜，可按辅助分析另行报告。MR 不属于 TrajGap-Bench 正式指标。

## 5. 当前代码对应关系

| 功能 | 实现 |
|---|---|
| TrajGap-Bench 数据适配 | `src/datamodule/trajimpute_dataset.py::TrajGapDataset` |
| Mixed 训练/验证 | `TrajImputeDataModule` 接收 `train_difficulties=[Mixed]`、`val_difficulties=[Mixed]` |
| Mixed 测试 | `TrajGapDataset(..., split="test")` |
| 统一训练入口 | `scripts/训练与评估/run_trajimpute_experiments.py --protocol trajgap-bench` |
| 统一 direct 评估 | `scripts/结果分析/evaluate_trajimpute_direct.py --difficulty Mixed` |
| checkpoint 选点 | 合并后的 Mixed-val 上，按最终预测头的 `val_new_minFDE20` 选点 |

推荐入口：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. \
/home/lbh/.conda/envs/DeMo/bin/python \
scripts/训练与评估/run_trajimpute_experiments.py \
  --protocol trajgap-bench \
  --variant M0 \
  --scenes ETH-M HOTEL-M UNIV-M ZARA1-M ZARA2-M \
  --seed 2024 \
  --gpu 3 \
  --output-root outputs/trajgap_bench_m0_seed2024
```

## 6. 当前范围边界

本阶段只实现统一 Mixed 数据组织和单一 Mixed 验证/测试口径，暂不实现：

- Easy/Hard 独立主榜；
- Clean 主训练或主测试分支；
- unseen-pattern 数据扩展；
- severity-balanced 或 evidence-balanced 主训练；
- 额外 direct/impute 双轨排行榜；
- 除 `minFDE20`、`minADE20` 外的主榜指标。

当前最准确的定位是：

> **TrajGap-Bench 将 TrajImpute 官方 Easy/Hard 的同一 train/val/test split 在适配层统一合并为 Mixed，并以 Mixed-val 最终预测头的单一 `val_new_minFDE20` 选点，在 Mixed-test 上并列报告 `minFDE20` 与 `minADE20`，不计算 MR 或综合总分。**
