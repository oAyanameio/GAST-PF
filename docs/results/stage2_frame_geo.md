# Stage 2 · FrameGeo

_依据 `docs/research/观测证据条件化几何时序编码器后续优化方向.md` §4 Stage 2 执行_

## 1. 目标

把逐帧位置几何显式送入 GSM，而不是只在最终 actor token 上补位置（Stage 1 的做法）。

## 2. 实现

**新增特征**

```text
relative_position_i,t = x_positions_i,t - x_centers_i
```

`x_centers_i` 是该 actor 最后有效历史位置（即 anchor，数据层已有）。历史输入由 4 维扩为 6 维：

```text
[x_positions_diff(2), x_velocity_diff(1), valid_mask(1), relative_position(2)]
```

**关键正确性点：缺失帧必须置零。** 数据层把缺失历史的 `x_positions` 填成 **0 占位**（`trajimpute_dataset.py:280` `nan_to_num`），若直接相减会得到 `-x_centers` 的**伪几何**——相当于给模型一个"该帧位于 anchor 反方向"的假信号。因此按 `valid_mask` 归零；有效性本身仍由第 4 通道表达。

**改动位置**

| 项 | 路径 |
| --- | --- |
| 特征构造 | `src/model/trajectory_forecaster.py`（`_frame_geo_features`） |
| 输入拼接 | 同上 `forward`，`hist_feat` 后追加 2 通道 |
| 输入宽度 | 同上 `__init__`：`gast_input_dim = 6 if gast_frame_geo else 4` |
| 开关 | `conf/model/missing_aware_ethucy_model_forecast.yaml`（`gast_frame_geo: false`） |
| 变体映射 | `scripts/训练与评估/run_trajimpute_experiments.py`、`scripts/结果分析/evaluate_trajimpute_direct.py` |
| 行为测试 | `tests/test_stage2_frame_geo.py` |

**保持不动**：GSM 内部结构（patch embed 只变输入宽度）、`Temporal Decay`、双向 Mamba、固定末端读出、decoder 输入维度、loss、优化器、TrajGap 协议。

**开关纪律**

- `gast_frame_geo: false`（默认）→ 输入仍 4 维，基础版本 `state_dict` 与行为不变；
- 只在 `encoder_type=gast` 生效；M0 对照路径的历史投影恒为 4 维（测试覆盖）；
- 开关写入 manifest `encoder_config` 与 checkpoint `hyper_parameters`；evaluator 对 `gast_anchor_geo` / `gast_frame_geo` **双向 fail-closed**（已由单开关扩展为循环校验）。

**参数量**

| 模型 | 总参数 | 相对基础版 |
| --- | ---: | ---: |
| M0 | 5,630,928 | — |
| GAST-PF | 5,213,968 | — |
| GAST-PF-AnchorGeo | 5,231,376 | +17,408 |
| GAST-PF-FrameGeo | 5,214,224 | **+256** |

增量仅来自 `patch_embed` 输入宽度 4→6（2×128）。**注意**：这也意味着 FrameGeo 与基础版**不是同参数初始化**（第一层形状不同），因此"增益"可能部分来自输入容量而非几何本身——见 §7 的限制说明。

## 3. 行为测试

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 PYTHONPATH=. \
  /home/lbh/.conda/envs/DeMo/bin/python -m pytest tests/test_stage2_frame_geo.py -q
# 13 passed, 1 skipped（skip = batch 内无 padding actor 时的隔离用例）
```

| 测试 | 覆盖 |
| --- | --- |
| `test_frame_geo_features_zero_at_missing_frames` | 缺失帧相对位置严格为 0；有效帧等于 `x_positions - x_centers` |
| `test_frame_geo_features_finite_for_all_invalid_actor` | 全无效 actor 不产生 NaN/Inf |
| `test_frame_geo_disabled_by_default` | 默认关闭时输入仍 4 维、无额外参数 |
| `test_frame_geo_enables_six_dim_input` | 开启后 `patch_embed.in_features == 6`、内部宽度仍 128 |
| `test_frame_geo_not_applied_to_m0_path` | M0 历史投影恒为 4 维 |
| `test_frame_geo_relative_position_shuffle_changes_output` | **判定条件一**：打乱逐帧位置必须改变输出 |
| `test_frame_geo_actor_permutation_invariant` | **判定条件二**：邻居置换不改变 focal 输出 |
| `test_frame_geo_missing_placeholder_is_ignored` | 改写缺失帧的 0 占位不影响输出（无伪几何泄漏） |
| `test_frame_geo_padding_actor_isolation` | padding actor 几何扰动不泄漏 |
| `test_frame_geo_ignores_future_targets` | 末端缺失不泄漏 future target |
| `test_frame_geo_finite_on_edge_fixtures` | 全有效 / 末端缺失 / 中间缺失 / 最小证据 |
| `test_frame_geo_decoder_contract_unchanged` | decoder 输入维度与输出契约不变 |
| `test_evaluator_frame_geo_contract` | checkpoint 开关契约双向 fail-closed |
| `test_frame_geo_parameters_covered_by_adamw_groups` | 优化器闭包纪律 |

全套回归 **405 passed, 3 skipped**；`compileall` 与 `git diff --check` 通过。

## 4. sanity run

`--smoke` 端到端 ETH-M：训练 → checkpoint（`monitor=val_new_minFDE20`）→ 评估 `status=complete rc=0`；manifest `encoder_config` 含 `gast_frame_geo: True`；评估输出六维分层齐备。产物已清理。

## 5. 正式训练（2026-10-17 18:20 → 2026-10-18 完成）

实际运行的队列（由 `LANES` 覆盖，脚本默认四卡布局未改）：

| 卡 | 队列 | 实测速率 |
| --- | --- | ---: |
| GPU 1 | ETH-M → HOTEL-M | 200.0 step/min（4.66 min/ep） |
| GPU 2 | ZARA1-M → ZARA2-M → UNIV-M | 221.4 step/min（3.95 min/ep） |

> 五场景全部完成，总墙钟约 14.7 h（7 分钟实测窗口，2026-10-17 18:55 采样）。全程 `Traceback=0 / CUDA错误=0 / OOM=0`。

协议：TrajGap-Bench Mixed-direct；seed=2024；K=20；100 epochs；batch size=64；选点 `val_new_minFDE20`；`experiment_class=screening`。输出根 `outputs/stage2_frame_geo_seed2024`。

启动方式（`setsid` 让每条链成为独立会话首进程，裸 background 启动会被会话清理静默回收）：

```bash
LANES="1:ETH-M,HOTEL-M 2:ZARA1-M,ZARA2-M,UNIV-M" \
  bash scripts/训练与评估/launch_stage2_frame_geo.sh
```

watcher 单独启动（幂等，可随时重挂）：

```bash
bash scripts/训练与评估/launch_watcher.sh    # 默认 Stage 2 / GAST-PF-FrameGeo
```

**本轮训练使用 `LAPLACE_LOSS_IMPL=vectorized-batch-v1`**（manifest 已记录）。该实现与原逐样本循环数学等价，但消除 64 份重复算子派发，使 step 速率从约 165–190 step/min 提升到 200–221 step/min。跨实现续训会被 resume contract 拒绝。

## 6. 结果（Mixed-test，K=20）

**协议**：TrajGap-Bench Mixed-direct；seed=2024；K=20；100 epochs；batch size=64；选点 `val_new_minFDE20`；`experiment_class=screening`（单种子）。样本合计 302,886。

| 场景 | n | best ep | minADE20 | minFDE20 | ADE@1 | FDE@1 | MR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH-M | 1,629 | 53 | 0.6441 | **0.9163** | 1.3921 | **2.3536** | 15.53% |
| HOTEL-M | 9,477 | 66 | 0.2363 | **0.3486** | **0.5173** | **0.8926** | 3.97% |
| UNIV-M | 219,006 | 65 | **0.3243** | **0.5220** | **0.8864** | **1.6547** | **3.31%** |
| ZARA1-M | 20,277 | 82 | 0.2481 | 0.4022 | 0.9075 | 1.6352 | 1.58% |
| ZARA2-M | 52,497 | 69 | 0.1754 | 0.2819 | 0.8366 | 1.4722 | 1.34% |
| **宏平均** | — | — | **0.3256** | **0.4942** | **0.9080** | **1.6016** | **5.15%** |

加粗 = 该指标优于 Stage 1。样本加权（micro）：minADE20 `0.2923`、minFDE20 `0.4691`、ADE@1 `0.8704`、FDE@1 `1.6017`、MR `2.94%`。

### 6.1 对 Stage 1 AnchorGeo（单变量对照）

| 指标 | Stage 2 | Stage 1 | 相对差 |
| --- | ---: | ---: | ---: |
| minADE20 | 0.3256 | 0.3241 | +0.47% |
| minFDE20 | 0.4942 | 0.4995 | **−1.06%** |
| ADE@1 | 0.9080 | 0.8980 | +1.11% |
| FDE@1 | 1.6016 | 1.5967 | +0.31% |
| MR | 5.15% | 4.86% | **+5.92%** |

逐场景相对差（负 = FrameGeo 更优）：

| 场景 | minADE20 | minFDE20 | ADE@1 | FDE@1 | MR |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETH-M | −0.78% | **−2.61%** | −1.56% | **−4.13%** | +7.20% |
| HOTEL-M | +1.57% | **−2.59%** | **−4.66%** | **−3.24%** | +10.91% |
| UNIV-M | **−1.04%** | **−2.84%** | **−8.47%** | **−7.48%** | **−6.38%** |
| ZARA1-M | +3.47% | +4.07% | +3.88% | +3.95% | −6.96% |
| ZARA2-M | +2.36% | +2.51% | **+21.04%** | **+18.29%** | **+34.80%** |

### 6.2 对外部模型（主对照）

| 宏平均相对差 | minADE20 | minFDE20 | ADE@1 | FDE@1 | MR |
| --- | ---: | ---: | ---: | ---: | ---: |
| vs MoFlow | −6.86% | −5.45% | −7.48% | −7.40% | **+11.87%** |
| vs SocialLSTM | −28.84% | −18.70% | −31.62% | −26.75% | −14.38% |

四项几何/排序指标优于 MoFlow；**MR 劣于 MoFlow（5.15% vs 4.60%）**，且相对 MoFlow 的 MR 劣势由 Stage 1 的 +5.7% 扩大到 +11.9%。

### 6.3 机制分层

按 `anchor_lag`（最后一次有效观测距当前多远），五场景样本加权：

| anchor_lag | n | S2 minFDE20 | S1 minFDE20 | Δ | S2 MR | S1 MR | ΔMR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 167,469 | 0.3698 | 0.3785 | −2.30% | 1.13% | 1.20% | −5.92% |
| 1 | 60,284 | 0.4422 | 0.4509 | −1.93% | 2.08% | 2.14% | −3.10% |
| 2 | 30,263 | 0.5247 | 0.5317 | −1.32% | 3.57% | 3.68% | −2.96% |
| 3 | 17,368 | 0.6398 | 0.6436 | −0.60% | 5.88% | 5.91% | −0.58% |
| 4 | 10,763 | 0.7748 | 0.7885 | −1.74% | 8.50% | 8.91% | −4.59% |
| 5 | 7,379 | 0.9167 | 0.9462 | **−3.12%** | 11.68% | 12.33% | −5.27% |
| 6 | 5,356 | 1.1244 | 1.1567 | **−2.79%** | 16.95% | 17.83% | −4.92% |
| 7 | 4,004 | 1.3445 | 1.3421 | **+0.17%** | 24.05% | 22.65% | **+6.17%** |

按 `missing_count`：

| 缺失帧数 | n | S2 minFDE20 | S1 minFDE20 | Δ | ΔMR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 33,654 | 0.3507 | 0.3557 | −1.42% | −20.57% |
| 1 | 33,654 | 0.3549 | 0.3604 | −1.54% | −15.85% |
| 2 | 33,654 | 0.3606 | 0.3685 | −2.14% | −11.24% |
| 3 | 33,654 | 0.3724 | 0.3795 | −1.87% | −6.47% |
| 4 | 67,308 | 0.3865 | 0.3931 | −1.67% | −5.82% |
| 5 | 33,654 | 0.4109 | 0.4185 | −1.82% | −6.13% |
| 6 | 33,654 | 0.4503 | 0.4708 | **−4.35%** | −7.73% |
| 7 | 33,654 | 1.1491 | 1.1653 | −1.39% | +0.52% |

按 `terminal_missing`：

| terminal_missing | n | S2 minFDE20 | S1 minFDE20 | Δ | ΔMR |
| --- | ---: | ---: | ---: | ---: | ---: |
| true | 135,417 | 0.5919 | 0.6018 | −1.64% | −2.26% |
| false | 167,469 | 0.3698 | 0.3785 | −2.30% | −5.92% |

**分层解读**：改善是**普遍小改善**，不是针对缺失的定向改善。`missing_count=0`（完全无缺失）也改善 −1.42%，与有缺失组同量级；`anchor_lag` 上没有 Stage 1 那种随缺失年龄单调放大的剂量—效应关系（lag 0 已 −2.30%，lag 7 反而 +0.17%）。该形态与"输入多 2 维带来的全局表示容量提升"一致，而非"逐帧几何帮助应对缺失"的机制性收益。

## 7. 判定

**Stage 2 判定：No-Go（不晋级 Stage 3）。**

| 依据 | 证据 |
| --- | --- |
| 宏平均无实质改善 | 仅 minFDE20 改善 −1.06%；minADE20 +0.47%、ADE@1 +1.11%、FDE@1 +0.31% 变差，落在单种子噪声带内 |
| MR 实质恶化 | 宏平均 MR 4.86% → 5.15%（+5.92%）；相对 MoFlow 由 +5.7% 扩大到 +11.9% |
| 单场景崩坏 | ZARA2-M ADE@1 +21.04%、FDE@1 +18.29%、MR +34.80%，非噪声量级 |
| 机制分层不支持几何假说 | 无缺失组与有缺失组改善幅度相当；`anchor_lag` 无单调剂量—效应 |
| 混淆未排除 | 无噪声通道对照臂，无法区分"几何"与"输入容量" |

**行为判定条件已满足但不构成 Go**：路线文档 §4 的两条行为测试（打乱相对位置改变输出、actor permutation 不变）均通过——模型确实在读这两个通道。问题在于读到之后没有产生机制性收益。

**必须记录的限制**

1. **训练损失实现在两批之间不同**：本轮用 `LAPLACE_LOSS_IMPL=vectorized-batch-v1`，Stage 1 用原逐样本循环。等价性门禁 `scripts/审计与校验/check_laplace_loss_equivalence.py` 结果 **38/38 PASS**（对 `traj/scale/probs` 梯度逐位相同，前向 loss 差 2.4e-7），数学上不构成混淆，但数值路径不同。
2. **无噪声通道对照臂**：`GAST-PF-FrameGeo` 与 `GAST-PF` 的 `patch_embed` 首层形状不同（4→6），非同参数初始化。分层证据指向容量解释，但没有直接实验证明。
3. **单种子 screening**：全部结论为 screening 级别，不作定稿 SOTA 表述。

## 8. 状态

- [x] 代码实现 + 开关 + 契约
- [x] 行为测试 14 项通过（全套 405 passed, 3 skipped）
- [x] sanity run（smoke 全链路）
- [x] 正式训练五场景（2026-10-17 18:20 → 10-18 完成，全程 `Traceback=0 / CUDA错误=0 / OOM=0`）
- [x] 结果回填 + 判定
- [ ] 噪声通道对照臂（`GAST-PF-FrameGeo-Noise`）——判定"几何 vs 容量"归属的唯一缺口，未运行
- [ ] `GAST-PF` 同口径配对基线（消融阶段补）

## 8. OER 实现 checkpoint（历史记录曾称 ECSR）

本节保留当时的实现审计记录。旧变体名 `GAST-PF-ECSR` 现对应 `GAST-PF-OER`，旧配置开关 `gast_ecsr` 由 `gast_oer_spatial`、`gast_emc_temporal` 和 `gast_ewhr_readout` 取代；旧 checkpoint 与 Stage 8 产物仍可通过兼容别名读取。本节中的验证状态是该记录形成时的状态，不代表当前训练进度。

### 8.1 统一证据状态

对每个 actor 和历史时间步维护同一组状态：

- `s`：直接观测资格；只有真实有效、非 padding 帧可以作为空间 attention 的 source/key；
- `r`：软可靠度；有效帧为 1，缺失帧按缺失年龄指数衰减；
- `a`：缺失年龄；用于构造 `r`，并保留前缀全缺的有限值语义。

该状态沿同一条信息流依次控制空间 source 资格、时序输入强度、前后向路径融合和历史读出，不能拆解为互不关联的五个模块。

### 8.2 已实现接口

| 层 | 实现 |
| --- | --- |
| 空间关系 | 逐帧 7 维相对位置/距离/速度/朝向差 attention bias；缺失 actor 不能作为 source，缺失 query 仍可接收有效邻居与 ghost token |
| 时序状态 | `r` 分别门控前向和反向 Mamba 输入；方向融合按累计缺失负担加权 |
| 读出 | 可靠度加权历史 evidence readout；末端有效时保留末端状态，末端缺失时融合历史证据 |
| 变体 | `TrajectoryForecaster(gast_oer_spatial=True, gast_emc_temporal=True, gast_ewhr_readout=True)` / `GAST-PF-OER` |
| 兼容性 | OER-GSM 与 `gast_anchor_geo`、`gast_frame_geo` 互斥；默认配置关闭，旧 GAST-PF 行为不变 |

### 8.3 验证状态

- `tests/test_ecsr.py`：CPU 行为、source/query 资格、几何屏蔽、padding 隔离、 permutation 等价性、checkpoint 契约、AdamW 参数闭包均通过；
- CUDA 编码器前向/反向冒烟通过；真实 ETH-M Mixed batch 的完整 forecaster 前向/反向冒烟通过；
- runner 闭环 smoke（ETH-M、1 epoch、训练/验证/评估各 1 batch）返回 `status=complete`、checkpoint 选择成功、evaluator `rc=0`；临时产物未纳入仓库；
- 旧回归：`tests/test_gast_encoder.py tests/test_stage2_frame_geo.py` 为 `22 passed, 2 skipped`；
- 代码提交 checkpoint：`564cc0d`；后续新增真实 batch 测试尚未提交；
- 该审计记录形成时尚未启动五场景正式训练；此处不据此判断当前 OER/OEHSR 的实验结论。

后续正式实验应比较 `GAST-PF`、最近一个通过验收的几何版本和 `GAST-PF-OER`，并检查完整历史、高缺失组、末端缺失组和全场景结果；若没有稳定收益，应收缩方法主张，而不是继续无条件叠加模块。

正式实验启动器：

```bash
bash scripts/训练与评估/launch_stage8_oer.sh
```

默认输出根为 `outputs/stage8_ecsr_seed2024`，与 Stage 1/2 结果隔离；可通过 `ROOT` 和 `LANES` 覆盖输出目录与 GPU/场景分配。

### 8.4 与 Sports-Traj BTS 的机制对照（本文差异=恒等性重参数化+统一证据状态）

BTS（Bidirectional Temporal Scaled）不是数据集，而是 Sports-Traj / UniTraj 的时间缩放模块；论文三个数据集为 Basketball-U / Football-U / Soccer-U。

**基础版 GAST-PF 已完整移植 BTS**，逐行对照官方代码 `~/UniTraj-pytorch/models/{unitraj,encoder_decoder,modules}.py`：

| BTS 构成 | 官方实现 | GAST-PF 实现 |
| --- | --- | --- |
| 前向距最近观测步数 `d_forw` | `unitraj.py::get_delta` 递归 `d[t]=1+(1-m[t])·d[t-1]` | `gast_encoder.py::missing_run_deltas._scan`（向量化，逐位一致，`tests/test_gast_encoder.py` 覆盖） |
| 反向 `d_back` | 对翻转 mask 同递推 | 同，保留翻转布局不翻回 |
| delta 升维 `delta_dim=2` | `repeat_interleave` | `unsqueeze(-1).expand(...,2)` |
| 缩放 `gamma = exp(-ReLU(Linear(delta)))` | `modules.py::Temporal_Decay`（`bias=False`） | `gast_encoder.py::TemporalDecay`（同式、同无 bias） |
| 正/反栈前分别乘 `gamma[0]`/`gamma[1]` | `Temporal_Encoder.forward` | `TemporalEncoder.forward` |
| 两条独立 Mamba 栈、反向 `flip` 后相加 | `ssm_forw`/`ssm_back` | 同 |

未引入的是 UniTraj 的**生成式框架**：`Future_Encoder`、CVAE 后验/先验（`qz_layer`/`pz_layer`）、WTA/diverse loss、MLP decoder、三类 agent one-hot、固定槽位 `pos_embed`。这些会改训练目标或 decoder，破坏单变量对照，故不引入。论文 Eq.(8) 所述"仅训练期使用的 GT 卷积核"对应 `Future_Encoder` 分支，不属于 BTS 本身。

**EMC-Mamba 的时间落点与 BTS 同族**：两者都是"按缺失年龄衰减特征后送入时间栈"；当前设计差异在参数化与恒等性，并将其纳入统一证据状态组织。早期记录将整体机制称为 ECSR：

| 项 | BTS | OER / EMC-Mamba 状态路由 |
| --- | --- | --- |
| 衰减形式 | 逐通道 MLP：`exp(-ReLU(Linear(delta)))` | 共享标量：`r = m + (1-m)·exp(-τa)`，`τ = softplus(θ)` |
| 有效帧取值 | `delta=1` → `gamma<1`，有效观测也被衰减 | 严格 `r=1`，全有效输入恒等 |
| 作用位置 | block 前缩放输入特征 | 同（`TemporalEncoder.forward` 内 `x_forw *= reliability`） |

OER 的设计由同一证据状态驱动：OER-GSM 控制逐帧空间 source 资格并注入几何 attention bias；EMC-Mamba 调制时序输入，并在双向模式下按累计缺失负担融合路径；EWHR 执行可靠度加权历史读出。若实验只有时间落点带来收益，则方法表述应收缩为证据新鲜度重参数化，不能据结构本身主张完整分层路由有效。
