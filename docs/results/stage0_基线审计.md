# Stage 0 · GAST-PF baseline audit

_执行日期 2026-10-15/16 · 依据 `docs/research/观测证据条件化几何时序编码器后续优化方向.md` §4 Stage 0_

> 目的：确认当前 `GAST-PF` 是可复现的比较起点，而不是在不稳定基线上堆模块。
> 本文件只记录 Stage 0 的协议、证据与验收结论。

## 1. 协议冻结（Stage 0 措施 1）

| 项 | 冻结值 | 证据 |
| --- | --- | --- |
| 数据协议 | TrajGap-Bench `mixed-direct`（Easy+Hard 同一 split 合并为 Mixed） | `run_trajimpute_experiments.py` `PROTOCOLS` |
| decoder / loss | `FutureDistributionDecoder` + 原 M0 loss 全套，未改动 | `trajectory_forecaster.py`、`forecasting_module.cal_loss` |
| optimizer | AdamW，模块本地参数分组（`named_parameters(recurse=False)`），白名单 weight 用 weight decay、自定义 mixer/RMSNorm weight 归 no-decay | `forecasting_module.configure_optimizers` |
| 数据 split | 训练/验证/测试均为 Mixed；五场景独立场景划分 | manifest `files` |
| seed | 2024（含 `train_sampling_seed=2024`） | manifest `seed` |
| **checkpoint 选点** | **`val_new_minFDE20`（最终输出头 `new_y_hat/new_pi`）** | `conf`、runner `checkpoint_monitor()`、manifest `monitor` |
| 训练预算 | 100 epochs、batch size 64、K=20、`lr=1e-3`、`weight_decay=1e-4`、warmup 5 | manifest |
| 代码 revision | `b60100ae18530667a848d781f7602de3a7511cce` | manifest `git_revision` |

**选点口径变更说明**：本轮之前所有历史产物（M0-optimizerfix、M0-trajgapbench、M1、GAST-PF/UT-ENC）均按初始头 `val_minFDE20` 选点，而 test 用最终头评估。Stage 0 起统一为 `val_new_minFDE20`，历史产物按旧口径归档，需重跑才能进入正式主榜。

**已知元数据差异**：历史 M0 manifest 记 `optimizer_policy = m0_adamw_complete_module_local_v1`，当前 runner 记 `adamw_complete_module_local_v1`。两者指向同一 `configure_optimizers` 实现（分组代码未变），仅为标签差异，不影响复现；如需跨版本配对比较应在文档中标注。

## 2. 复跑（Stage 0 措施 2）· 已停止

输出根：`outputs/stage0_baseline_audit_seed2024/`（新口径，与历史产物隔离）

| 臂 | 目录 | 状态 |
| --- | --- | --- |
| GAST-PF | — | **已按裁定停止**（2026-10-16，训练至 e5 后终止，产物已清理） |
| M0 | — | 未在新口径下重训 |

场景选择 ETH-M：五场景中 test 规模最小（1,629），且是路线文档 §1.5 初步结果所在场景。

**停止理由**：基础版本已冻结（路线文档 §0.1），当前优先事项转为在基础版本上做增量优化；新口径复跑不属于该阶段，留待需要正式主榜数字时再执行。

**实测速率（停止前记录，供后续估算）**：GAST-PF 112.5 step/min ≈ 8.3 min/epoch，100 epochs ≈ 13.7 h（ETH-M，GPU 1）。

**M0 同口径基线现状（影响 Stage 1 门槛）**

新口径下**没有**可配对的 M0 checkpoint：

| 项 | 历史 M0 ETH-M |
| --- | --- |
| 旧 monitor(`val_minFDE20`) 最佳 | **e85**（0.3644） |
| 新 monitor(`val_new_minFDE20`) 最佳 | **e76**（0.3598） |
| 现存 checkpoint | 仅 `epo.ckpt`(e85) 与 `last.ckpt` |

`save_top_k=1` 是按旧 monitor 保存的，新 monitor 最优的 e76 **从未落盘**，因此无法靠"重新评估历史 checkpoint"补救。历史 M0 结果（`0.3223/0.4917` 宏平均等）为旧口径，**不能与新版 GAST-PF 数字配对**。

因此 Stage 0 验收项「M0 与 GAST-PF 可在相同协议下复现」当前**未满足**（缺 M0 侧）；Stage 1 若以 M0 为主对照，需先补跑同口径 M0，或明确改用其他同口径参照。

## 3. 分层维度（Stage 0 措施 3）

评估器输出 `results.by_dimension`。Stage 0 要求的六个维度现状：

| 维度 | 状态 |
| --- | --- |
| `missing_count` | 已有 |
| `valid_count` | 已有 |
| `anchor_lag` | 已有 |
| `forecast_gap` | 已有 |
| `terminal_missing` | 已有（由 `anchor_lag > 0` 派生） |
| **连续缺失长度 `max_missing_run`** | **本轮补齐** |

补齐内容：`build_sample` 输出 focal 的 `max_missing_run`（邻居缺失不计入）→ `trajimpute_collate_fn` 透出 → `DirectEvaluator.update` 新增该维度 → evaluator CLI 传入。新增 `tests/test_stage0_stratification.py`（5 项）覆盖字段来源、focal-only 语义与六维度齐备。

ETH-M test 抽样分布（`max_missing_run`）：`0:181 1:409 2:405 3:271 4:146 5:105 6:67 7:45`。

## 4. GAST-PF 资源记录（Stage 0 措施 4）

| 项 | M0 | GAST-PF |
| --- | ---: | ---: |
| 总参数量 | 5,630,928 | 5,213,968（−417k，−7.4%） |
| 编码器参数量 | 1,466,560 | 1,067,136 |
| └ 空间（GSM） | — | 116,992 |
| └ 时序（双向 Mamba） | — | 949,888 |
| └ Temporal Decay | — | 256 |
| decoder（共用） | 4,099,192 | 4,099,192 |
| 训练显存（单进程） | 3,352 MiB | 5,020 MiB |
| 输出张量形状 | `[B, N, 128]` | `[B, N, 128]` |
| 每 epoch 墙钟 | 4.8 min | 9.3 min（含共卡拖慢） |

GAST-PF 编码器参数量低于 M0 编码器，但训练更慢——瓶颈在计算（双向 Mamba + 逐帧 agent 自注意力）而非参数规模。

## 5. 代码行为审计（Stage 0 措施 5）

逐条核对文档描述与 `src/model/layers/gast_encoder.py` 实际实现：

| 文档描述 | 代码事实 | 结论 |
| --- | --- | --- |
| Temporal Decay `x' = gamma ⊙ x`，`gamma = exp(-ReLU(Linear(delta)))` | `TemporalDecay.forward`：`torch.exp(-self.relu(self.mlp(delta)))` | 一致 |
| 前向/后向 Mamba 两条独立栈，输出直接相加 | `TemporalEncoder.forward`：`y_forw + torch.flip(y_back, dims=[1])`，`ssm_forw`/`ssm_back` 为两套独立参数 | 一致 |
| 固定读取最后时间步 `y[:, -1]` | `GASTEncoder.forward`：`last = y[:, -1].reshape(B, N, -1)` | 一致 |
| 输出 `[B, N, embed_dim]` | 实测 `(2, 5, 128)` | 一致 |
| padding actor 不参与 | 输出端 `last * (~kp)`；GSM 注意力 `key_padding_mask` 屏蔽 padding key；ghost 池化排除 padding 行 | 一致 |

**文档 §1.4.4 被代码证实**：全有效历史输入下，`delta` 为 `[0,1,1,...,1]`，`gamma` 实测落在 **0.26–1.00**，**并非恒为 1**。即当前 Temporal Decay 确实会衰减有效观测（帧 0 的 delta=0 处 gamma=1）。这为后续把 decay 改造成 evidence-age gate 提供了直接依据。

## 6. 已知负对照（Stage 0 措施 6）

历史 `last-valid` 读出 screening（旧 M0 Easy/Hard 口径）作为**已知负对照**记录，不作为新的主实验：

- Easy→Easy 宏平均 `minADE20/minFDE20 = 0.2538/0.4204`；Hard→Hard `0.4355/0.6538`。
- Hard 上有局部正向信号，但 Easy 回归；收益**不集中于** `valid_count=1` 或末端缺失样本。
- 裁定：`last-valid` 读出判负，读取位置本身未被证明是主要瓶颈。
- 边界：该结论不能反证固定末端读出无问题；后续读出实验只能围绕 `last + evidence` 自适应融合，不得把 `last-valid` 重新包装成新模块。

## 7. 验收核对（Stage 0 验收）

| 验收项 | 状态 | 证据 |
| --- | --- | --- |
| M0 与 GAST-PF 可在相同协议下复现 | **未满足（复跑已停止）** | 复跑按裁定终止，未产生新口径数字；历史 M0 的新 monitor 最优 checkpoint 未落盘，无法补齐配对 |
| 无 NaN | 通过 | 停止前运行 `runner.log` 中 `nan=0`；编码器输出 `isfinite` |
| 无空 actor | 通过 | 数据层 `load_trajimpute_pkl` 对「无任何有效历史帧」直接报错 |
| 无 padding 泄漏 | 通过 | 编码器输出端置零 + 注意力 key 屏蔽（实测 padding actor 输出全零） |
| 能得到 terminal missing 子组结果 | 通过 | `by_dimension.terminal_missing` 已有；连同 `max_missing_run` 六维度齐备 |

**结论**：协议冻结、代码行为审计、资源记录、分层维度、负对照五项已完成并落盘；复跑一项按裁定停止。基础版本已冻结（路线文档 §0.1），后续优化基于该版本增量进行；新口径正式主榜数字留待需要时再跑。
