# GAST-PF-HER：面向缺失历史的分层证据推理行人轨迹预测框架

_研究方法与实验执行指南 · 版本日期：2026-10-23 · 面向缺失历史条件下的行人未来轨迹多模态预测_

> **文档定位**：本文将当前模型统一命名为 GAST-PF-HER（Gap-Aware Spatio-Temporal Pedestrian Forecaster with Hierarchical Evidence Reasoning），并整理为可复现的方法章节草案。它不是实验结果论文，也不把尚未验证的机制假设写成结论。旧 OEHSR 仅作为兼容标签保留。

## 方法组件与术语

| 代码模块 | 英文名称 | 作用 |
|---|---|---|
| `OES` | Observation Evidence State | 从有效性、缺失年龄和 padding 构造贯穿各阶段的共享状态 |
| `GERE` | Geometry-Evidence Relational Encoding | 在逐帧关系交互中约束消息源资格并注入相对几何关系 |
| `RMCM` | Reliability-Modulated Causal Mamba | 用证据可靠度调制单向 Mamba 的时序状态输入 |
| `EPHR` | Evidence-Pooled History Readout | 按证据可靠度聚合历史状态并处理末端缺失 |
| `ECSI` | Evidence-Conditioned Scene Integration | 用末端证据调节 actor 级场景消息和残差融合 |
| `HER` | Hierarchical Evidence Reasoning | 将 OES、GERE、RMCM、EPHR 与 ECSI 组织为完整预测流程 |

旧名称 `OEHSR`、`ECSR`、`ECHSR`、`gast_m0_scene_tail` 仅作为历史 checkpoint、配置和实验产物的兼容标识；新实验和论文正文统一使用 `OES`、`GERE`、`RMCM`、`EPHR`、`ECSI` 与 `HER`。这些名称是对完整方法中功能阶段的统一抽象，不意味着基础 GSM、Mamba、Temporal Decay 或场景 Transformer 本身被重新发明。相关工作中仍需准确引用其来源；本文的方法贡献落在这些阶段针对缺失历史轨迹预测的接口设计、证据状态贯穿和整体组合，并由配对消融与分层实验验证。

## 摘要式概述

缺失历史下的行人轨迹预测同时面临输入不完整、运动连续性受损、相对几何关系变化、场景上下文不均衡以及末端缺失导致历史读出不稳等问题。本文围绕这一任务构建 **GAST-PF-HER** 完整预测框架：将历史运动特征、观测掩码与几何信息组织为统一输入；通过 GERE 收集逐帧关系证据；采用 RMCM 编码个体时间演化；在末端观测不完整时通过 EPHR 聚合历史状态；再利用 ECSI 整合经过证据条件化的邻居上下文，最终由既有多模态分布解码器生成未来轨迹候选。OES 为多个阶段提供一致的缺失信息接口，是整体方法的组织中枢；完整框架的有效性仍须由同口径对照、缺失分层和消融实验验证。

## 1. 研究问题与可检验假设

### 1.1 研究问题

给定多行人场景中不完整的历史轨迹，如何在不先重建完整历史的条件下，综合利用剩余运动观测、行人间几何关系、个体时间演化和场景上下文，直接预测目标行人的未来多模态轨迹？本文从完整预测流程处理缺失带来的影响：缺失如何改变逐帧空间关系，如何影响个体时序状态与历史读出，以及邻居信息如何参与编码后的场景推理。共享观测证据状态为这些设计环节提供统一的缺失信息描述，核心问题可概括为：

> **如何把可见的个体运动、行人间相对几何、不同时间位置的历史状态和邻居场景上下文，整合为对未来轨迹有用的预测表示？**

### 1.2 研究假设

- **H1：显式几何关系有助于缺失条件下的空间建模。** 在逐帧 actor 表示中纳入相对位置、相对速度和方向关系，预期比仅依据个体运动差分更好地表达邻居对当前运动的影响；该作用需通过匹配输入与训练设置的消融检验。
- **H2：因果时序编码与历史读出共同影响末端缺失预测。** 单向 Mamba 建模可用历史的因果演化，证据加权读出在末端缺失时整合此前状态；预期收益应结合末端缺失与不同缺失长度分层评估，而不能由模型结构直接推断。
- **H3：编码后场景交互补充逐帧局部关系建模。** 在个体时序编码后整合 actor 级上下文，预期能补充逐帧 GSM 的交互信息；尾部是否需要由观测状态调制，应通过普通 Scene Tail 与证据条件化 Scene Tail 的配对对照区分。
- **H4：整体框架的收益来自对空间、时间和场景信息的协同组织。** GAST-PF-HER 应在统一缺失协议下与基线及结构匹配变体比较，并报告总体性能、缺失分层、参数量和计算成本；若收益只出现在单一场景或单一子组，应相应收窄结论。

以上均为待验证假设。若完整历史样本明显退化，或收益不集中在预先定义的证据困难子组，则应收缩机制解释；若主指标未优于结构匹配对照，则不应仅凭可解释门控宣称方法有效。

### 1.3 整体方法贡献的表述

论文的模型贡献应从**缺失历史行人轨迹预测的完整方法设计**来阐述，而不是将创新压缩为单一的信息路由机制，也不是把每个已有算子拆成互不相关的贡献：

> **面向不完整历史下的行人未来轨迹多模态预测，构建 GAST-PF-HER 时空预测框架，将逐帧几何关系建模、因果时序编码、缺失感知历史读出与编码后场景交互组织为端到端预测流程，并以共享层级证据状态协调不同阶段对历史与邻居信息的使用。**

论文应完整说明各阶段如何回应任务需求、阶段间如何传递表示，以及该组合相对于直接套用完整历史预测器的具体差异。GSM、Mamba、几何编码、历史读出与场景交互的已有研究来源应通过规范引用交代；论文的贡献表述聚焦于缺失历史预测这一任务中的整体方法设计、各组件的具体适配及经实验验证的增益。结构命名本身不替代新颖性比较或实证证据。

## 2. 问题定义

对一个场景样本，设 actor 集合为 \(\mathcal{V}=\{1,\ldots,N\}\)，观测长度为 \(T\)，预测长度为 \(H\)。actor \(i\) 在时刻 \(t\) 的二维位置为 \(\mathbf{p}_{i,t}\in\mathbb{R}^2\)，其直接观测掩码为 \(m_{i,t}\in\{0,1\}\)。当 \(m_{i,t}=0\) 时，该帧不提供直接位置证据；输入张量中即使存在占位值，也不得将其解释成真实观测。

模型还可使用 actor 属性、有效历史锚点（最后有效位置）及其方向等数据集字段。目标是直接预测 focal actor 的未来轨迹分布：

\[
\hat{\mathcal{Y}}_1=\{(\hat{\mathbf{y}}_1^{(k)},\pi_k)\}_{k=1}^{K},\qquad
\sum_{k=1}^{K}\pi_k=1,
\]

其中 \(\hat{\mathbf{y}}_1^{(k)}\in\mathbb{R}^{H\times2}\) 是第 \(k\) 个未来模式，\(\pi_k\) 是其概率。HER 不先生成插补轨迹，也不改变既有多模态解码器。

**当前 TrajGap-Bench 配置**：ETH-M、HOTEL-M、UNIV-M、ZARA1-M、ZARA2-M 五个场景；Mixed-direct 训练、验证与测试；\(T=8\)、\(H=12\)、\(K=20\)。正式报告须明确数据 release、split、缺失协议和 checkpoint 选择规则，不能把 impute-then-predict 结果与 direct 结果混作同一评估口径。

## 3. 输入、证据状态与整体信息流

### 3.1 输入特征

当前 `TrajectoryForecaster` 为历史编码器构造 4 维逐帧特征：

\[
\mathbf{x}_{i,t}^{\mathrm{hist}}=
[\Delta\mathbf{p}_{i,t},\;\Delta v_{i,t},\;m_{i,t}]\in\mathbb{R}^{4},
\]

其中包含 2 维位置差分、1 维速度差分和 1 维有效性标记。该 4 维张量并不等于模型的全部信息：启用 OER 时，原始历史位置还被送入几何分支；Scene Tail 另接收最后有效位置 `x_centers`、最后有效方向 `x_last_valid_angle`（或兼容回退的末帧角度）及 actor 类型。padding actor 由 `x_key_valid_mask` 区分。

输入张量的主要形状为：历史特征 \([B,N,T,4]\)、逐帧掩码 \([B,N,T]\)、位置 \([B,N,T,2]\)、actor padding 掩码 \([B,N]\)。批内 actor 数量可变；padding actor 不参与有效的消息传播。

### 3.2 共享证据状态

为每个 actor、每个历史时刻定义证据状态：

\[
\mathbf{e}_{i,t}=(s_{i,t},a_{i,t},r_{i,t}),
\]

其中 \(s_{i,t}\) 为直接观测资格，\(a_{i,t}\) 为距离最近一次有效观测的年龄，\(r_{i,t}\) 为软可靠度。令

\[
\ell_{i,t}=\max\{u\le t:m_{i,u}=1\}.
\]

若 \(\ell_{i,t}\) 存在，则 \(a_{i,t}=t-\ell_{i,t}\)；若此前没有任何有效观测，则当前实现令 \(a_{i,t}=t+1\)。于是

\[
s_{i,t}=m_{i,t},\qquad
r_{i,t}=\begin{cases}
1,&m_{i,t}=1,\\
\exp(-\tau a_{i,t}),&m_{i,t}=0,
\end{cases}\quad \tau=\operatorname{softplus}(\theta_\tau)>0.
\]

\(\theta_\tau\) 是当前 OER 实现中的可学习标量。此定义使有效帧的可靠度严格为 1，缺失帧随年龄增加而衰减。**可靠度是基于缺失年龄的确定性代理量，不是经校准的概率置信度，也没有估计坐标噪声或遮挡成因。** 对 padding actor，证据状态被置为无效。

### 3.3 总体架构

```text
历史运动特征 + 有效性掩码 + 原始观测位置
                         │
             共享证据状态 (s, a, r)
                         │
       ┌─────────────────┴───────────────────┐
       │                                     │
逐帧几何关系建模
  GSM + ghost token + 相对几何偏置
  有效观测 source 与可靠度调节
       └─────────────────┬───────────────────┘
                         ↓
             可靠度缩放的单向 Mamba
                         ↓
        可靠度加权历史证据读出（actor token）
                         ↓
   证据条件化 Scene Tail（context + actor type）
                         ↓
          原有 FutureDistributionDecoder
                         ↓
                K 个未来轨迹模式
```

## 4. 方法细节

### 4.1 逐帧关系证据收集与几何条件化 GSM

先将逐帧历史特征投影到 \(d\) 维 actor 表示。GSM 与 ghost token 用于构建逐帧交互表示，相关结构见轨迹预测工作 [2]；ghost 表示当前帧聚合后的可用上下文。标准自注意力的 query、key、value 由归一化 token 生成。本文在缺失历史任务中进一步对消息来源施加直接观测资格约束：ghost 始终可用，actor \(j\) 只有在 \(s_{j,t}=1\) 且非 padding 时才能作为有效 source。

对同时存在直接观测的 actor 对，构造 7 维相对几何描述：

\[
\boldsymbol{\phi}_{ij,t}=
[\Delta x,\Delta y,\log(1+\|\Delta\mathbf{p}\|),
\Delta v_x,\Delta v_y,\cos\Delta\psi,\sin\Delta\psi],
\]

其中位置为同帧 actor 间相对位置，速度由该 actor 与其前一直接观测之间的位移和间隔计算，\(\Delta\psi\) 为运动方向差。每个注意力头用小型 MLP 将其映射为加性几何偏置 \(b^{geo}_{ij,t,h}\)，并加到 actor–actor 注意力 logits：

\[
\alpha_{ij,t,h}=\operatorname{softmax}_{j\in\mathcal{S}_{i,t}}
\left(\frac{q_{i,t,h}^{\top}k_{j,t,h}}{\sqrt{d_h}}+
b^{geo}_{ij,t,h}\right),
\]

\(\mathcal{S}_{i,t}\) 包含 ghost 与合格的直接观测 actor。几何偏置的最后一层零初始化，使训练起始阶段不强制施加随机几何先验。

对于 actor query，当前实现先完成注意力、FFN 与归一化，再乘以该 query 的可靠度：

\[
\tilde{\mathbf{z}}_{i,t}=r_{i,t}\,\mathbf{z}_{i,t}^{GSM}.
\]

因此，准确说法是**直接观测资格控制 source，可靠度在 GSM 输出后调节 query 状态强度**。当前代码没有把 \(r_{i,t}\) 加进 query logits，也不是严格意义上的 query routing 或 query attention-mask。ghost token 不作为最终 actor token 输出。

### 4.2 可靠度调制的单向 gap-aware Mamba

HER 使用单向因果 Mamba 栈。Mamba 的选择性状态空间序列建模来源见 [1]，并已用于轨迹预测编码 [2]。对 actor \(i\) 的逐帧表示先经共享线性投影，再按可靠度缩放：

\[
\mathbf{u}_{i,t}=r_{i,t}\,\operatorname{MLP}(\tilde{\mathbf{z}}_{i,t}),\qquad
\mathbf{h}_{i,1:T}=\operatorname{Mamba}_{\rightarrow}
(\mathbf{u}_{i,1:T}).
\]

这使高年龄缺失状态以较小幅度进入因果状态递推。它是**输入特征幅度调制**，不是修改 Mamba 内部状态转移、选择性扫描步长 \(\Delta\)，也不是跳过缺失时刻的状态更新。单向时序栈保证 \(\mathbf{h}_{i,t}\) 不依赖未来历史位置。

### 4.3 证据加权的 actor 历史读出

为避免末端缺失时无条件依赖最后一个时间索引，模型以可学习线性评分 \(q_{i,t}\) 与可靠度对数构造历史权重：

\[
w_{i,t}=\operatorname{softmax}_{t}
\left(q_{i,t}+\beta\log(r_{i,t}+\epsilon)\right),
\qquad
\bar{\mathbf{h}}_i=\sum_t w_{i,t}\mathbf{h}_{i,t}.
\]

最终 token 为

\[
\mathbf{a}_i=r_{i,T}\mathbf{h}_{i,T}+
(1-r_{i,T})\bar{\mathbf{h}}_i.
\]

当前实现中 \(\beta=1\) 为固定 buffer；末端可靠度为 1 时读出严格回到末帧状态，末端越不可靠，历史证据项占比越高。它不是简单的 last-valid 拷贝：历史权重仍由状态评分和可靠度共同决定。

### 4.4 证据条件化的编码后 Scene Tail

将 actor token、场景锚点上下文和 actor 类型嵌入组合为场景 token：

\[
\mathbf{v}_i=\mathbf{a}_i+
f_{ctx}([\mathbf{c}_i,\cos\psi_i,\sin\psi_i])+
\mathbf{e}^{type}_i,
\]

其中 \(\mathbf{c}_i\) 是最后有效位置，\(\psi_i\) 是最后有效方向。随后使用 M0 原有的 5 层 actor-level Transformer Scene Tail 建模场景交互。对 key actor \(j\) 加入可靠度偏置：

\[
b^{key}_{ij}=\log(\max(r_{j,T},10^{-4})),
\]

padding key 被屏蔽。该偏置降低低可靠度 actor 作为消息源的权重；低可靠度 query 仍可接收场景上下文。对每个 query actor 计算融合门：

\[
g_i=\sigma\left(f_g\left([r_{i,T},
\operatorname{clip}(a_{i,T}/(T-1),0,1)]\right)\right).
\]

Scene Tail 输出经归一化后，与输入 actor token 作证据调制的残差融合：

\[
\mathbf{a}^{out}_i=\mathbf{a}_i+
\alpha_{scene}g_i(\operatorname{SceneTail}(\mathbf{v})_i-\mathbf{a}_i).
\]

\(\alpha_{scene}\) 是标量可学习残差系数，初始化为 0，使训练初始时尾部不扰动已有 actor token；融合门由可靠度与末端年龄产生。此处的可靠度偏置是在 Scene Tail 的 key 维度上调节信息来源，不等于把低可靠度 actor 从场景中彻底删除。

### 4.5 预测头与训练目标

\(\mathbf{a}^{out}\) 输入原有 `FutureDistributionDecoder`，输出 K 个未来轨迹模式、模式概率及原预测头所需的不确定性参数。HER 不增加插补头、未来编码器、辅助重建目标或独立损失。训练损失、优化器和主指标与配对对照保持一致，以便将差异归因于历史编码与信息组织，而不是训练目标改写。

## 5. 实现与论文术语对照

| 论文概念 | 代码实现 | 精确表述边界 |
|---|---|---|
| OES：共享观测证据状态 \((s,a,r)\) | `construct_hierarchical_evidence_state`，`gast_encoder.py` | 可靠度由 mask 与年龄解析得到；不是不确定性估计器 |
| GERE：逐帧几何证据关系编码 | `GeometryEvidenceRelationalGSM` / `gast_oer_spatial` | ghost token + actor self-attention；padding 与无直接观测 source 被屏蔽 |
| 相对几何偏置 | `relative_geometry_features`、`geometry_bias` | 7 维 pairwise 特征生成每头加性 bias |
| GSM query 可靠度调节 | `query_reliability` | 注意力后的表示相乘；不要称作 query-logit routing |
| RMCM：单向证据调制时序编码 | `ReliabilityModulatedCausalMamba` / `gast_emc_temporal` | HER 路径中 Mamba 输入按 \(r\) 缩放，内部状态更新规则未改 |
| EPHR：证据池化历史读出 | `evidence_pooled_history_readout` / `gast_ewhr_readout` | score + \(\log r\) 加权，再与末帧状态按末端 \(r\) 混合 |
| Temporal Decay 的路径边界 | `TemporalDecay` | 基础 GAST-PF 的非证据时序路径使用；HER 打开证据调制时序编码后不实例化该模块 |
| 普通场景尾部 | `SceneReasoningTail` / `gast_scene_reasoning_tail` | actor context + 5 层 Scene Transformer；未恢复 M0 历史 Mamba |
| ECSI：证据条件化场景整合 | `EvidenceConditionedSceneTail` / `gast_egst_tail` | 末端可靠度控制 actor key bias 与 query actor 的残差融合门 |
| key 可靠度偏置与 actor 融合门 | `_build_evidence_attention_mask`、`_evidence_fusion_gate` | 分别控制 source actor key 权重和 query actor 的残差注入 |
| 整体变体 | `gast_her=True` / `GAST-PF-HER` | 自动开启 GERE、RMCM、EPHR、ECSI 与普通场景尾部，并要求单向 Mamba；`gast_oehsr` 为兼容别名 |

主要代码位置：`src/model/layers/gast_encoder.py`、`src/model/trajectory_forecaster.py`、`scripts/训练与评估/run_trajimpute_experiments.py`。研究路线与历史决策见 `docs/research/观测证据条件化几何时序编码器后续优化方向.md`。

## 6. 实验设计

### 6.1 对照组与最小递进消融

所有内部对照应固定数据 split、seed、预测头、损失、优化策略和 checkpoint 选择规则。至少报告 M0 与基础 GAST-PF 作为参考。主对照围绕完整预测流程安排，并将互不兼容的几何实现作为独立分支，而不是假设单一线性消融能解释全部机制：

| 变体 / 设置 | 相对比较 | 该对比回答的问题 |
|---|---|---|
| `GAST-PF-Uni` | `GAST-PF` → 单向 Mamba | 时序方向改变对因果历史编码有何影响？ |
| `GAST-PF-UniTail` | `Uni` → 加普通 Scene Tail | 编码后场景交互是否补充逐帧关系建模？ |
| `Uni + FrameGeo` | `Uni` → 打开 `gast_frame_geo` | 逐帧相对位置输入能否补足运动差分表示？ |
| `GAST-PF-UniOER` | `Uni` → OER 空间、时间与读出路径 | 整合几何关系和缺失状态处理后，编码路径整体如何变化？ |
| `GAST-PF-UniOER-Tail` | `UniOER` → 普通 Scene Tail | 普通场景尾部对 OER 编码表示的增量是什么？ |
| `GAST-PF-HER` | `UniOER-Tail` → 证据条件化 Scene Tail | 场景层是否需要根据观测状态调节消息源和残差融合？ |

该比较组用于评估时序方向、几何输入、历史编码、场景交互及其整合效果。`Uni` 到 `UniOER` 同时改变关系、时间和读出处理，只能解释为编码路径整体差异，不能拆作单个机制的因果效应。若整体结果通过预设门槛，再追加最小定向消融：

1. 固定 OER 与尾部结构，仅关闭/开启 GSM post-GSM query reliability gate。
2. 固定其余结构，仅关闭/开启 Scene Tail key reliability bias。
3. 固定其余结构，仅关闭/开启 actor-level evidence fusion gate。

不建议在主筛选未通过前运行完整全因子组合；若需声称交互效应，才扩展为设计完整的因子实验。

### 6.2 数据协议与训练控制

- **数据与任务**：TrajGap-Bench Mixed-direct，五个 ETH/UCY 场景；训练、验证和测试均使用对应 Mixed split，模型直接从不完整历史预测未来。
- **窗口与候选数**：当前实验配置为 8 帧历史、12 帧未来、20 个模式；所有变体一致。
- **优化设置**：按实验 manifest 固定 seed、epoch 数、batch size、学习率、weight decay、warmup、precision、数据采样、优化器策略和代码 revision。当前筛选配置使用 seed 2024、100 epochs、batch size 64、`lr=1e-3`、`weight_decay=1e-4`、`bf16-mixed`，按最终预测头 `val_new_minFDE20` 选 checkpoint；正式论文的复跑应由 manifest 逐项复核，而非仅复制此处文字。
- **配对原则**：模型变体之外的训练因素保持相同；新增模块使用相同的零初始化/配对初始化纪律，避免初始化随机数流改变共享部分参数。参数量与 FLOPs 若变化，必须与精度和耗时同时报告。
- **数据泄漏检查**：几何和证据状态只能从历史 mask 与历史观测位置构造；未来坐标仅进入训练损失与评估。padding actor 不得参与聚合、注意力 source 或统计分层。

### 6.3 指标与统计汇报

- **主指标**：五场景宏平均 `minADE@20` 与 `minFDE@20`，另给逐场景数值。宏平均是场景等权平均，不能与样本加权 micro 平均混称。
- **辅助指标**：`ADE@1`、`FDE@1` 诊断最高概率模式的排序质量；与 best-of-20 指标分开解释。
- **分层指标**：至少按 `missing_count`、`valid_count`、`anchor_lag`、`forecast_gap`、`terminal_missing` 和最长连续缺失长度分层；同时报告每组样本数。不得只挑收益最大的场景或缺失档位。
- **统计稳定性**：单 seed 只能作为 screening。若主线通过，应在关键对照上增加多 seed 复跑，并对测试样本做成对 bootstrap 区间；报告绝对差值、相对变化和置信区间，不仅报告最佳值。
- **机制诊断**：保存可聚合的 \(\tau\)、不同年龄下的 \(r\)、GSM source mask 覆盖率、Scene Tail key bias、\(g_i\) 分布、\(\alpha_{scene}\) 训练轨迹，以及全有效/末端缺失/长缺失组的门控统计。若评估代码尚不能导出这些量，应先增加只读诊断，不改变模型前向。

项目已裁定 `MR` 不作为 TrajGap-Bench 正式主榜指标；不要在新论文主结果表中将其与旧表混排。若另行报告校准、NLL 或覆盖率等指标，必须定义计算协议并增加校准实验，不能把当前 age-based reliability 直接解释为已校准置信度。

### 6.4 预先判定规则

1. **进入机制归因**：HER 在完整五场景同口径测试中，相对 `GAST-PF-UniOER-Tail` 的主指标表现有稳定改善，且该改善不是由少数单一场景主导。
2. **支持证据路由解释**：末端缺失或长连续缺失组有与假设一致的收益，门控/可靠度诊断方向合理；全有效组没有不可接受的系统性退化。
3. **收缩方法主张**：若只有 UniTail 有益，则结论限于编码后场景交互；若仅 OER 编码路径有益，则不主张 HER 尾部；若 HER 只改善 overall 而证据困难子组无改善，则不得将收益归因于缺失证据路由。
4. **停止条件**：完整测试不优于结构匹配对照，或关键收益不能在多 seed / 关键场景上复现时，停止叠加模块，记录负结果并回到主问题与机制实现审计。

## 7. 当前证据状态与论文结论边界

**历史实验快照（2026-10-22）**：当时 OEHSR 产物中有 4 个场景的 Mixed-test `results.json`；HOTEL-M 训练尚未完成。该状态仅用于解释历史产物，不代表当前实验状态。相关目录仍使用旧标签 `GAST-PF-OEHSR_seed2024`；正式结论须核验该目录下的 manifest、checkpoint 与评估产物。

即使五场景单 seed 完成，也只能支持 screening 层面的性能观察。论文投稿前还需要：确认所有变体使用匹配的 `val_new_minFDE20` 选点与一致代码/协议；核实 M0、Uni、UniTail、UniOER、UniOER-Tail 对照产物完整；完成至少关键对照的多 seed 稳定性评估；针对性能主张执行必要机制消融；补足与当前直接缺失历史预测和缺失感知社会交互工作的相关工作比较。当前不能据此宣称 HER 已达到顶会或顶刊的创新/性能门槛。

## 8. 论文写作结构建议

### 8.1 引言要回答的问题

1. 缺失历史直接预测与先插补再预测的任务差异是什么？为什么不把插补质量等同于未来预测质量？
2. 历史缺失会如何影响运动表示、逐帧行人关系、个体时间建模、末端读出和场景交互？
3. 如何把几何关系建模、因果时序编码、历史状态汇聚和多模态预测组织成适用于缺失历史的完整预测框架？
4. 共享观测证据状态在整体架构中承担什么作用，哪些设计带来实际增益，如何由对照实验检验？

### 8.2 方法章节顺序

建议依次撰写：任务定义与输入边界 → 整体架构 → 输入表示与观测证据状态 → 逐帧几何关系建模 → 单向 gap-aware Mamba 与历史读出 → 编码后场景交互 → 多模态预测头与训练目标 → 复杂度与实现细节。正文从完整预测流程阐述 HER；消融章节分析各阶段设计的作用及其组合效果。

### 8.3 结果章节顺序

先给完整五场景主表，再给缺失分层和末端缺失分析，然后给机制消融与门控诊断，最后报告参数量、速度、显存和失败案例。对每一张表明确标注 seed 数、checkpoint 规则、宏/微平均定义和样本数。

## 9. 局限与可证伪点

- 年龄衰减把“缺失时长”视为可靠度代理，无法区分随机丢帧、遮挡、检测失败、异步采样或系统性缺失；真实应用若缺失机制不同，\(r\) 未必对应真实置信度。
- GSM query 可靠度门控发生在注意力计算后，因此它不会阻止低可靠度 query 在当前层读取可靠 source；若论文要声称 query 本身按证据路由，需实现 logits/消息级控制并重新实验。
- Scene Tail 的 key bias 只在 actor 级末端可靠度上路由，不表示每条成对交互都得到因果解释；其 actor context 使用锚点位置和末端有效方向，不能扩写为完整场景地图理解。
- 单向 Mamba 的因果性不自动等于实时系统验证；本文仍需报告延迟、复杂度和在线缓存策略，若声称部署优势。
- 若收益依赖某个场景、某个 seed 或训练预算，说明证据不足以支持普遍机制结论；应报告不确定性并收缩结论。

## 10. 复现检查清单

- [ ] 论文代码版本、checkpoint、配置 manifest 与数据协议可对应。
- [ ] 五场景 train/val/test 结果完整，缺失窗口与 focal/neighbor 规则已核验。
- [ ] 所有核心内部变体使用同一最终预测头选点口径 `val_new_minFDE20`。
- [ ] 运行前确认 `gast_temporal_direction=unidirectional`、`gast_oehsr=true`、`gast_oer_spatial=true`、`gast_emc_temporal=true`、`gast_ewhr_readout=true`、`gast_egst_tail=true`。
- [ ] checkpoint 的模型开关、编码器方向、模式数与评估变体一致，且无缺失/额外参数键。
- [ ] 主指标、场景宏平均、辅助 top-1 指标和各缺失子组样本数均由同一评估器生成。
- [ ] 报告参数量、FLOPs/推理延迟、显存、训练时间、seed 和训练预算。
- [ ] 结果叙述严格区分代码机制、实验观察、机制解释与尚未验证的假设。

## 参考文献与引用边界

1. Gu, A. and Dao, T. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752, 2023。用于 Mamba 时序算子的来源说明。
2. Xu, Y. and Fu, Y. *Sports-Traj / UniTraj*. ICLR 2025, arXiv:2405.17680。用于 GSM、ghost token、Temporal Decay/BTS 与轨迹预测时序编码等相关设计的来源说明。
3. Chib, P. and Singh, P. *TrajImpute: Pedestrian Trajectory Prediction with Missing Data*. NeurIPS 2024 Datasets and Benchmarks, arXiv:2411.00174。用于描述缺失历史轨迹任务与数据基准背景；本文的 direct 设置不等同于其插补后预测协议。

投稿前应重新核验引用元数据，并补充与本文整体方法最相关的缺失感知轨迹预测、几何关系建模、时序编码、社会交互和直接预测研究。引用用于准确交代各方法组件及任务背景；本文的模型贡献围绕这些设计如何共同适配缺失历史行人轨迹预测任务，以及相应实验是否支持整体效果展开。
