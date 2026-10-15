# TrajGap-Bench: A Benchmark for Direct Trajectory Forecasting under Missing History

# TrajGap-Bench：面向缺失历史直接预测的行人轨迹预测基准

**Version 1.0 — 2026-10-12 · Bilingual 中英对照**

> **Dataset / 数据集：** TrajGap (TrajImpute-derived Missing-History Trajectory Dataset / 基于 TrajImpute 的缺失历史轨迹数据集)
> **Benchmark / 基准：** TrajGap-Bench (Missing-History Direct Forecasting Benchmark / 缺失历史直接预测基准)

---

## 1. Motivation 动机

Pedestrian trajectories observed in the wild frequently contain gaps: occlusions, tracker failures, or sensor dropouts. Two families of methods handle such cases:

真实场景中的行人轨迹常出现缺口：遮挡、跟踪器丢失或传感器采样中断。处理方法分两类：

- **Impute-then-predict 先插补后预测**：first reconstruct the missing coordinates, then forecast with a complete-history model (e.g., TrajImpute) / 先重建缺失坐标，再用完整历史模型预测（如 TrajImpute）；
- **Direct prediction 直接预测**：consume the masked history as-is and condition forecasting on the evidence that is actually available / 直接消费带掩码的历史输入，基于实际可用的证据进行预测。

Direct prediction matters because interpolation itself introduces error, and because a model that can reason over *the amount and arrangement of available evidence* is the only one that can degrade gracefully as observations vanish. Yet direct missing-history forecasting currently lacks a shared benchmark: existing protocols either separate conditions into isolated tracks (Easy/Hard with no unified validation or leaderboard), or do not define a common checkpoint-selection rule.

直接预测之所以重要，是因为插补本身会引入误差；只有能感知*可用证据的数量与排布*的模型，才能在观测逐渐减少时优雅退化。然而，直接缺失历史预测目前缺少共享基准：现有协议要么把条件拆成相互隔离的轨道（Easy/Hard 各自验证、无统一榜单），要么不定义统一的 checkpoint（检查点）选择规则。

TrajGap-Bench does not re-invent missing-history data. It re-uses the audited official TrajImpute release verbatim and removes the fragmentation:

TrajGap-Bench 不重新发明缺失历史数据，而是原样复用已审计的 TrajImpute 官方发布，消除碎片化：

1. **Unified protocol 统一协议**：Easy and Hard are merged sample-wise per split into one Mixed dataset; training, validation, and testing all follow the same Mixed calibre / Easy 与 Hard 按样本级合并为单一 Mixed 数据集，训练、验证、测试统一为 Mixed 口径；
2. **Single validation 单一验证**：one Mixed validation set with a single `val_minFDE20` selects the checkpoint — no per-difficulty selection, no test-set tuning / 一个 Mixed 验证集、单一 `val_minFDE20` 选择 checkpoint——不按难度分别选点，不用测试集调参；
3. **Single leaderboard 单一榜单**：one Mixed test set, one metric family (`minFDE20`, `minADE20`, `MR` at K=20), one composite ranking / 一个 Mixed 测试集、一套指标族（K=20 的 `minFDE20`/`minADE20`/`MR`）、一个综合排名。

## 2. Relationship to Prior Benchmarks 与既有基准的关系

**Dimension 维度** | **ETH/UCY (classic) 经典** | **TrajImpute (official) 官方** | **TrajGap-Bench 本基准**
--- | --- | --- | ---
Task 任务 | Complete-history prediction 完整历史预测 | Missing-coordinate imputation **and** imputed prediction 缺失坐标插补 **及** 插补后预测 | Direct (no-imputation) missing-history prediction 直接（无插补）缺失历史预测
Data 数据 | Raw trajectories, LOO/LOSO predefined splits 原始轨迹，LOO/LOSO 预定义划分 | 5 scenes × Easy/Hard × train/val/test | Same official release, Easy+Hard merged per split into Mixed 同一官方发布，每个 split 合并 Easy+Hard 为 Mixed
Missing levels 缺失程度 | None 无 | Easy: 0–4 missing points; Hard: 4–7 缺 0–4 点；缺 4–7 点 | 0–7, in a single Mixed distribution 0–7，单一 Mixed 缺失分布
Evaluation 评估 | ADE/FDE (K=20) | Impute metrics + prediction metrics, per difficulty 按难度分开 | `minFDE20`, `minADE20`, `MR`, single Mixed test 单一 Mixed 测试
Checkpoint selection 选点 | Full val set 完整验证集 | Per difficulty 按难度分别选择 | Single Mixed val, `val_minFDE20` 单一 Mixed 验证集

**Positioning statement 定位声明.** TrajImpute established the missing-history data resource and the impute-then-predict protocol. TrajGap-Bench contributes the missing piece for the direct-prediction community: a fixed, leakage-free protocol with a reproducible 8→12-frame task, a single validation rule, and a single composite leaderboard. Official impute-then-predict numbers from TrajImpute are **not** co-ranked with direct predictions — the two calibres differ in input information and are reported separately.

TrajImpute 建立了缺失历史数据资源和先插补后预测协议。TrajGap-Bench 为直接预测社区补上缺失的一环：固定且无泄漏的协议、可复现的 8→12 帧任务、单一验证规则、单一综合榜单。TrajImpute 官方的先插补后预测数字**不**与直接预测结果同榜比较——两种口径的输入信息不同，须分开报告。

**Clean setting Clean 设置.** Clean refers to the original ETH/UCY complete-history benchmark (8 observed frames, no missing injection, no imputation). It is not a subset of the TrajImpute Easy test; missing-0 copies there are only an artefact of the test construction. Clean results are quoted from the original papers (e.g., MoFlow CVPR 2025).

Clean 指原始 ETH/UCY 完整历史基准（8 帧观测、无缺失注入、无插补）。它不是 TrajImpute Easy 测试的子集，其中的 missing-0 副本只是测试集构造的产物。Clean 的结果引用原论文（如 MoFlow, CVPR 2025）。

## 3. Task Definition 任务定义

Every sample is an actor-centric clip; 每个样本是以目标行人为中心的片段：

- **Input 输入**：8 frames of history at 0.4 s resolution (3.2 s). For each frame and coordinate, `missing_mask` (bool) marks missing observations; missing coordinates are `NaN` in the official files and a masked placeholder downstream. Every sample retains **at least one** valid history frame. The input also carries the validity mask explicitly / 8 帧历史，分辨率 0.4 s（共 3.2 s）。每帧每坐标由 `missing_mask`（布尔）标记是否缺失；官方文件中缺失坐标为 `NaN`，下游以掩码占位。每个样本**至少保留 1 个有效历史帧**。输入同时显式携带有效性掩码。
- **Output 输出**：12 frames of future (4.8 s), always complete, used for supervision and evaluation / 12 帧未来（4.8 s），始终完整，用于监督与评估。
- **Prediction format 预测格式**：K = 20 candidate trajectories per focal actor, sampled from the model's output distribution (mode queries / sampling) / 每个目标行人 K=20 条候选轨迹，来自模型输出分布（mode queries 多模态查询 / 采样）。
- **Direct constraint 直接约束**：methods must consume the masked history directly. Imputation is allowed *inside* a model as an internal latent mechanism, but a method that requires an external off-the-shelf imputation module before forecasting is evaluated separately, not on this leaderboard / 方法必须直接消费带掩码历史。模型*内部*将插补作为隐式机制是允许的；但若方法在预测前需要外部现成插补模块，则单独评估，不进本榜单。

## 4. Dataset Construction 数据集构造

### 4.1 Source 数据来源

TrajGap uses the official TrajImpute release: 5 scenes (ETH-M, HOTEL-M, UNIV-M, ZARA1-M, ZARA2-M) × 2 difficulty tiers (Easy, Hard) × 3 splits (train/val/test), 30 pickles in total. The official pkl fields are:

TrajGap 使用 TrajImpute 官方发布：5 个场景（ETH-M、HOTEL-M、UNIV-M、ZARA1-M、ZARA2-M）× 2 难度（Easy、Hard）× 3 划分（train/val/test），共 30 个 pickle 文件。官方 pkl 字段：

**Field 字段** | **Shape 形状** | **Meaning 含义**
--- | --- | ---
`obs_traj` | [N, 8, 2] | History; `NaN` where missing (`missing_mask=True`) 历史轨迹；缺失处为 `NaN`
`pred_traj` | [N, 12, 2] | Future; always complete 未来轨迹；恒完整
`missing_mask` | [N, 8, 2] | `True` = coordinate missing 该坐标缺失
`seq_start_end` | [M, 2] | Scene/group boundaries (rebuilt deterministically, see §4.4) 场景/组边界（确定性重建，见 §4.4）

### 4.2 Merge rule 合并规则

For every scene `s ∈ {ETH-M, HOTEL-M, UNIV-M, ZARA1-M, ZARA2-M}` and every split `sp ∈ {train, val, test}`:

对每个场景 `s ∈ {ETH-M, HOTEL-M, UNIV-M, ZARA1-M, ZARA2-M}` 和每个划分 `sp ∈ {train, val, test}`：

```text
TrajGap(s, sp) = TrajImpute(s, Easy, sp)  +  TrajImpute(s, Hard, sp)
```

`+` is **sample-wise concatenation** (actor-centric, no re-windowing, no de-duplication, no re-sampling). Source labels are kept internally for audit; the public protocol name is **Mixed** for all splits.

`+` 表示**样本级拼接**（以行人为中心；不重切窗口、不去重、不重采样）。来源标签仅保留在内部供审计；对外协议名统一为 **Mixed**（混合）。

### 4.3 Statistics 统计信息

Focal-sample counts (audited from the official files); Focal（目标行人）样本数（已对官方文件审计）：

**Scene 场景** | **Mixed train** | **Mixed val** | **Mixed test**
--- | ---: | ---: | ---:
ETH-M | 59,618 | 10,698 | 1,629
HOTEL-M | 58,304 | 10,272 | 9,477
UNIV-M | 18,462 | 5,416 | 219,006
ZARA1-M | 56,020 | 10,236 | 20,277
ZARA2-M | 51,014 | 8,346 | 52,497
**Total 合计** | **243,418** | **44,968** | **302,886**

Missing-level distribution in the Mixed training set covers the full range m = 0…7 (Easy provides 0–4, Hard 4–7). Approximately 10% of Mixed training focal samples have complete history (m=0), so models are simultaneously exposed to complete, lightly missing, and heavily missing inputs.

Mixed 训练集的缺失程度覆盖完整区间 m = 0…7（Easy 提供 0–4，Hard 提供 4–7）。约 10% 的 Mixed 训练样本历史完整（m=0），因此模型同时暴露于完整、轻度缺失与重度缺失输入。

### 4.4 Data-integrity notes (audited) 数据完整性说明（已审计）

- `NaN` positions in `obs_traj` match `missing_mask` exactly; `pred_traj` contains no `NaN` / `obs_traj` 中 `NaN` 位置与 `missing_mask` 严格一致；`pred_traj` 无 `NaN`。
- The official test pickles contain repeated copies of each original trajectory (Easy: ×5 with 0–4 missing points; Hard: ×4 with 4–7). Repeated copies are **retained**: the test set is intentionally the full missing-level sweep, and sample counts above must not be read as unique original trajectories / 官方测试 pickle 包含每条原始轨迹的重复副本（Easy：×5，缺 0–4 点；Hard：×4，缺 4–7 点）。重复副本**保留**：测试集刻意构成完整缺失程度扫描，§4.3 的样本数不得解读为独立原始轨迹数。
- A known upstream bug multiply-repeats `seq_start_end` blocks without scene offsets, so block indices misalign for blocks 2+. The adapter deterministically rebuilds `seq_start_end`; official files are never modified / 上游已知 bug 会以列表乘法重复 `seq_start_end` 块且不加场景偏移，导致第 2 块起的行索引错位。适配器确定性重建 `seq_start_end`；官方文件绝不修改。
- TrajGap never re-windows the raw trajectories and never constructs a new train/test split; it consumes the predefined official splits verbatim / TrajGap 从不基于原始数据重切窗口，也不构造新的 train/test 划分；严格按官方预定义划分原样消费。

## 5. Evaluation Protocol 评估协议

### 5.1 Splits and checkpoint selection 划分与 checkpoint 选择

- **Train 训练**：Mixed-train (Easy+Hard train).
- **Validation 验证**：Mixed-val (Easy+Hard val). A single `val_minFDE20` on the merged set selects the checkpoint. Per-difficulty validation and per-difficulty checkpoint selection are **not** part of the protocol / 在合并集上以单一 `val_minFDE20` 选择 checkpoint。按难度分别验证、分别选点**不是**本协议的一部分。
- **Test 测试**：Mixed-test (Easy+Hard test). The test set never influences sampling, hyper-parameters, or checkpoint selection / 测试集不影响采样、超参数或 checkpoint 选择。
- All splits share the same 8-frame history, 12-frame future, direct input format, and K=20 / 所有划分共用相同的 8 帧历史、12 帧未来、直接输入格式与 K=20。

### 5.2 Metrics 指标

All metrics are computed on the K=20 candidate sets; **lower is better** for all three; 所有指标基于 K=20 候选集计算；三个指标**均越低越好**：

**Metric 指标** | **Definition 定义**
--- | ---
`minFDE20` | Final-displacement error of the candidate closest to the ground-truth endpoint 20 条候选中最接近真值终点者的终点位移误差
`minADE20` | Average-displacement error of the candidate closest to the ground truth 20 条候选中最接近真值者的平均位移误差
`MR` | Miss rate: fraction of samples where no candidate is within 2.0 m of the true endpoint 漏检率：20 条候选均未达到真值终点 2.0 m 阈值的样本比例

`ADE@1` / `FDE@1` (top-probability mode) and distribution-quality metrics (energy score, coverage, calibration) are **not** part of the current leaderboard and may be reported as auxiliary diagnostics.

`ADE@1`/`FDE@1`（最高概率模态）与分布质量指标（能量分数、覆盖率、校准度）**不**进入当前榜单，可作为辅助诊断另行报告。

### 5.3 Leaderboard 榜单

TrajGap-Bench uses a single composite leaderboard; TrajGap-Bench 采用单一综合榜单：

1. All three raw metrics come from the same Mixed test, the same checkpoint, the same scene protocol, and the same K=20 — mixing calibres or protocols in one ranking is disallowed / 三个原始指标必须来自同一 Mixed 测试、同一 checkpoint、同一场景协议、同一 K=20——不允许混口径或混协议排名；
2. Reports must include per-scene results, macro average, micro average, and worst-scene in addition to the composite score / 除综合分数外，必须报告各场景明细、macro 平均、micro 平均和 worst-scene（最差场景）；
3. Composite score (lower is better), with min–max normalization over **pre-frozen** reference bounds / 综合分数（越低越好），采用**预先冻结**参考范围的 min–max 归一化：

```text
z(m, j) = (x(m, j) − best(j)) / (worst(j) − best(j)),   j ∈ {minFDE20, minADE20, MR}
S(m)    = (z(m, minFDE20) + z(m, minADE20) + z(m, MR)) / 3
```

The normalization bounds must be frozen before the leaderboard is released and must not be re-chosen from test results. Near-tied composite scores must report the raw metrics; small margins are not interpreted as stable wins.

归一化上下界必须在榜单发布前冻结，不得根据测试结果事后调整。综合分数接近时须并列展示原始指标，微小差异不解释为稳定优势。

## 6. Reference Implementation (M0) 参考实现

M0 is the reference model of this project: a MoFlow-derived state-space trajectory forecaster (unidirectional Mamba history encoder, Scene Transformer multi-agent cross-attention, structured time decoder) with a mask-aware input (`x_valid_mask` as an explicit input channel). It is trained on Mixed-train exactly under the protocol above.

M0 是本项目的参考模型：MoFlow 派生的状态空间轨迹预测器（单向 Mamba 历史编码器、Scene Transformer 多智能体交叉注意力、结构化时间解码器），输入包含显式有效性掩码通道（`x_valid_mask`）。严格按上述协议在 Mixed-train 上训练。

Training setup 训练配置：seed 2024, 100 epochs, batch size 64, K=20 mode queries, AdamW with module-local parameter grouping, `val_minFDE20` checkpoint selection.

Reference results, Mixed-test (macro average over 5 scenes, seed 2024); Mixed-test 参考结果（5 场景 macro 平均，seed 2024）：

**Model 模型** | **minADE20** | **minFDE20** | **MR**
--- | ---: | ---: | ---:
M0 (this work 本工作) | 0.3249 | 0.5039 | 0.0461

These numbers define the current reference point of the leaderboard and are regenerated from a single fixed checkpoint per scene.

这些数字定义榜单当前参考点，每场景由单一固定 checkpoint 重算得到。

## 7. Code and Usage 代码与使用

Repository 代码仓库：`GAST-PF` (private 私有). All data loading, training, and evaluation go through the TrajGap adapter; official release files are read-only / 数据加载、训练、评估全部经由 TrajGap 适配器；官方发布文件只读。

```bash
# Train + evaluate all 5 scenes on Mixed protocol (example, single GPU)
# 五场景 Mixed 协议训练+评估（示例，单卡）
PYTHONNOUSERSITE=1 PYTHONPATH=. \
/home/lbh/.conda/envs/gast-pf/bin/python \
scripts/训练与评估/run_trajimpute_experiments.py \
  --protocol mixed-direct \
  --variant M0 \
  --scenes ETH-M HOTEL-M UNIV-M ZARA1-M ZARA2-M \
  --seed 2024 \
  --gpu 3 \
  --output-root outputs/trajgap_bench_m0_seed2024

# Evaluate an existing checkpoint on the Mixed test set
# 用已有 checkpoint 在 Mixed 测试集上评估
/home/lbh/.conda/envs/gast-pf/bin/python \
scripts/结果分析/evaluate_trajimpute_direct.py \
  --data-root /home/lbh/TrajImpute/dataset/TrajImpute \
  --scene ETH-M --difficulty Mixed --split test \
  --variant M0 --K 20 --seed 2024 \
  --checkpoint <path.ckpt> --output-root <out_dir>
```

Key interfaces 关键接口：`src/datamodule/trajimpute_dataset.py::TrajGapDataset` (dataset 数据集), `TrajImputeDataModule` with `train_difficulties=[Mixed], val_difficulties=[Mixed]` (data module), `scripts/结果分析/evaluate_trajimpute_direct.py` (evaluator 评估器, `--difficulty Mixed`).

**Submission checklist for new methods 新方法提交清单:**

1. Same Mixed train/val/test organization 使用相同的 Mixed train/val/test 组织；
2. Single `val_minFDE20` checkpoint selection 单一 `val_minFDE20` checkpoint 选择；
3. K=20, same 8→12 task, direct (no external imputation) K=20、相同的 8→12 任务、直接预测（无外部插补）；
4. Report per-scene + macro + micro + worst-scene raw metrics 报告各场景 + macro + micro + worst-scene 原始指标；
5. Report the seed(s); multi-seed with mean±std is encouraged 报告 seed；鼓励多种子并给出 mean±std。

## 8. Known Limitations & Notes 已知限制与说明

- The official Easy test reuses each trajectory 5 times with different missing levels and Hard 4 times; the Mixed test therefore contains repeated latent trajectories by construction. Metric results are still meaningful for the missing-level sweep, but sample counts in §4.3 are not unique original trajectories / 官方 Easy 测试每条轨迹按不同缺失程度复用 5 次、Hard 复用 4 次，因此 Mixed 测试按构造包含重复的潜在轨迹。指标对缺失程度扫描仍有意义，但 §4.3 的样本数不是独立原始轨迹数。
- UNIV-M test is large (~219k focal samples); evaluation cost should be planned accordingly / UNIV-M 测试集很大（约 21.9 万 focal 样本），评估开销需提前规划。
- Impute-then-predict pipelines are deliberately excluded from the leaderboard; compare them in a separate table with clearly separated calibre / 先插补后预测管线刻意排除在榜单之外；请在单独表中以明确区分的口径比较。
- Clean complete-history results are quoted from the original literature, not re-run on this benchmark / Clean 完整历史结果引用原文献，不在本基准上重跑。

## 9. Citation 引用

If you use TrajGap / TrajGap-Bench, please cite; 若使用 TrajGap / TrajGap-Bench，请引用：

```bibtex
@misc{trajgap2026,
  title  = {TrajGap-Bench: A Benchmark for Direct Trajectory Forecasting under Missing History},
  author = {GAST-PF project},
  year   = {2026},
  note   = {Built on TrajImpute (arXiv:2411.00174)}
}
```

And the underlying data source; 以及底层数据来源：

```bibtex
@article{trajimpute2024,
  title  = {TrajImpute: A Training-Free Approach for the Missing Trajectory Prediction},
  journal = {arXiv preprint arXiv:2411.00174},
  year   = {2024}
}
```
