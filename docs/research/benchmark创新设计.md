# TrajGap 数据集与 TrajGap-Bench 评测协议设计

> 文档性质：数据集整理与评测协议说明
>
> 当前状态：面向缺失历史直接预测的统一数据与评测设置
>
> 数据集名称：**TrajGap**
>
> 评测协议名称：**TrajGap-Bench**
>
> 数据来源：TrajImpute 官方发布
>
> 研究任务：缺失历史条件下的直接未来轨迹预测（direct / imputation-free trajectory forecasting）

## 1. 当前协议定位

TrajGap 是基于 TrajImpute 官方发布整理形成的数据集；TrajGap-Bench 则规定该数据集上的任务定义、训练与选点规则、测试指标及结果报告方式。对外介绍以 TrajGap 作为数据集整体名称，不再使用内部文件标签命名数据子集。

评测协议包括：

1. 沿用 TrajGap 的官方场景及 train、validation、test 划分；
2. 直接以带缺失标记的历史观测预测完整未来，不以前置外部插补为条件；
3. 使用 TrajGap validation 划分进行 checkpoint 选点；
4. 在 TrajGap test 划分上报告普通 `minADE20` 和 `minFDE20`；
5. 按缺失数量、连续缺失长度和末端缺失等属性进行分层分析，不计算综合总榜分数；
6. 暂不加入额外数据轨道或额外主评估指标。

TrajGap 的数据适配不修改官方 pkl、不重新切窗，也不重新生成未来目标。官方 impute-then-predict 数字不进入 TrajGap-Bench 的直接预测结果表。

## 2. TrajGap 数据组织

TrajGap 包含五个场景。每个场景沿用官方 train、validation 和 test 划分，并由适配器组织成以焦点行人为中心的样本：

```text
TrajGap/<scene>/train
TrajGap/<scene>/validation
TrajGap/<scene>/test
```

每个样本包含 8 帧历史和 12 帧完整未来；历史缺失位置由掩码标明。当前已审计的 focal 轨迹样本数量如下：

| 场景 | TrajGap train | TrajGap validation | TrajGap test |
|---|---:|---:|---:|
| ETH-M | 59,618 | 10,698 | 1,629 |
| HOTEL-M | 58,304 | 10,272 | 9,477 |
| UNIV-M | 18,462 | 5,416 | 219,006 |
| ZARA1-M | 56,020 | 10,236 | 20,277 |
| ZARA2-M | 51,014 | 8,346 | 52,497 |

测试划分保留上游发布中的缺失程度扫描结构，因此同一潜在轨迹可能对应多个缺失掩码。样本数不等于独立原始轨迹数；统计推断应按原始轨迹聚类或采用配对 bootstrap。

### 2.1 数据适配规则

- 读取 TrajImpute 官方 pkl，不修改源文件；
- 历史缺失坐标仍由 `missing_mask` 定义；
- 未来 `pred_traj` 保持完整，只用于监督和评估；
- 对官方文件中的 `seq_start_end` block offset 问题进行确定性修复；
- focal actor、邻居分组、局部坐标和跨缺口运动特征沿用原适配器；
- 对外样本标识采用 TrajGap 场景与 split 信息；
- 不将缺失数量为零的样本另行命名为独立 Clean 数据集。

## 3. TrajGap-Bench 训练与评估流程

```text
TrajGap train ──────────┐
                        ├─> 模型训练
TrajGap validation ─────┘
           └──────────────> val_new_minFDE20 选择 checkpoint

TrajGap test ─────────────> minADE20 / minFDE20 与分层分析
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

1. 两个指标来自同一 TrajGap test 划分、同一 checkpoint、同一场景协议和同一 `K=20`；
2. 报告五场景明细、macro average、全部样本汇总平均和 worst-scene；
3. 不定义综合分数或单一总排名，避免引入未冻结的归一化边界与指标权重。

正式结果只报告普通 `minADE20` 和 `minFDE20`，不按模式分数加权；MR、Brier、top-1、Energy Score、coverage 和 calibration 均不进入当前报告。

## 5. 当前代码对应关系

| 功能 | 实现 |
|---|---|
| TrajGap 数据适配 | `src/datamodule/trajimpute_dataset.py::TrajGapDataset` |
| TrajGap 训练/验证 | `TrajImputeDataModule` |
| TrajGap 测试 | `TrajGapDataset(..., split="test")` |
| 统一训练入口 | `scripts/训练与评估/run_trajimpute_experiments.py --protocol trajgap-bench` |
| 统一 direct 评估 | `scripts/结果分析/evaluate_trajimpute_direct.py` |
| checkpoint 选点 | TrajGap validation 上按最终预测头的 `val_new_minFDE20` 选点 |

推荐入口：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. \
/home/lbh/.conda/envs/DeMo/bin/python \
scripts/训练与评估/run_trajimpute_experiments.py \
  --protocol trajgap-bench \
  --variant GAST \
  --scenes ETH-M HOTEL-M UNIV-M ZARA1-M ZARA2-M \
  --seed 2024 \
  --gpu 3 \
  --output-root outputs/trajgap_bench_gast_seed2024
```

## 6. 当前范围边界

本阶段使用 TrajGap 统一数据组织与单一验证/测试口径，暂不实现：

- 按底层来源标签拆分的独立主榜；
- Clean 主训练或主测试分支；
- unseen-pattern 数据扩展；
- severity-balanced 或 evidence-balanced 主训练；
- 额外 direct/impute 双轨排行榜；
- 除 `minFDE20`、`minADE20` 外的主榜指标。

当前最准确的定位是：

> **TrajGap 是基于 TrajImpute 官方发布整理形成的数据集；TrajGap-Bench 是其缺失历史直接预测评测协议。协议在 TrajGap validation 上按最终预测头的 `val_new_minFDE20` 选择 checkpoint，并在 TrajGap test 上报告 `minFDE20` 与 `minADE20`，不计算 MR 或综合总分。**
