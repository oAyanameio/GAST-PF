# GAST 统一主方法命名实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将当前完整模型的默认名称统一为 `GAST`，把 goal 条件解码和轨迹级模式排序作为 GAST 内部设计，并保留旧 `GAST-Goal` 产物的加载兼容性。

**架构：** `encoder_type=gast` 时直接使用完整 GAST 编码器、goal decoder、trajectory-aware classification 和 final-consistent loss；不再通过 `GAST-Goal` 注册为独立方法。训练与评估脚本只把 `GAST` 作为完整方法入口，旧模型名仅在兼容解析层识别并映射到 GAST 配置。

**技术栈：** PyTorch、PyTorch Lightning、Hydra、PyYAML、pytest、Markdown/LaTeX 结果生成器。

---

### 任务 1：先写统一命名的失败测试

**文件：**
- 修改：`tests/test_gast_encoder.py`
- 修改：`tests/test_gast_goal_integration.py`
- 修改：`tests/test_gast_goal_decoder.py`

- [ ] **步骤 1：增加 canonical GAST 注册断言**

测试应断言训练和评估入口都包含 `M0` 与 `GAST`，完整 GAST 配置使用 `gast_goal=True`、`classification_target_mode="trajectory"`，并断言 `GAST-Goal` 不再是新实验注册名。

- [ ] **步骤 2：增加模型内部名称断言**

将完整 GAST forward 测试改名为 `test_gast_forward_uses_goal_decoder_as_base_design`，构造完整 GAST 后断言 `model.gast_method_name == "GAST"`、存在 `goal_decoder`、不存在 `future_decoder`，并保持 endpoint 与最终轨迹一致。

- [ ] **步骤 3：增加旧 checkpoint 加载兼容断言**

构造旧 `GAST-Goal` checkpoint（含旧模型配置和 `goal_decoder.*` state dict），使用新 `GAST` evaluator 构建模型后加载，断言 `missing_keys=[]` 且 `unexpected_keys=[]`。

- [ ] **步骤 4：运行失败测试**

运行：

```bash
pytest -q tests/test_gast_encoder.py tests/test_gast_goal_integration.py tests/test_gast_goal_decoder.py
```

预期：新 canonical 命名断言失败，证明当前代码仍把 `GAST-Goal` 注册为独立方法或把完整模型元数据标为 `GAST-Goal`。

### 任务 2：将完整模型行为并入 GAST

**文件：**
- 修改：`src/model/trajectory_forecaster.py`
- 删除：`conf/model/gast_goal_model_forecast.yaml`
- 创建：`conf/model/gast_model_forecast.yaml`
- 修改：`conf/config_missing_aware_trajimpute.yaml`

- [ ] **步骤 1：把完整 GAST 配置迁移到 canonical 文件**

将旧完整配置迁移为 `conf/model/gast_model_forecast.yaml`，配置名改为 `gast`，默认启用完整 goal decoder 和 trajectory-aware classification；根配置指向此唯一 GAST 配置。

- [ ] **步骤 2：让 `encoder_type=gast` 默认构造完整解码器**

在 `TrajectoryForecaster` 中将完整 GAST 的 decoder 选择与 `encoder_type=="gast"` 绑定；M0 继续使用 legacy `FutureDistributionDecoder`。旧 `gast_goal` 参数只作为兼容输入，不再决定当前 GAST 的公开方法名称。

- [ ] **步骤 3：统一内部方法名**

完整 GAST 的 `gast_method_name` 固定为 `"GAST"`；错误信息、注释和 checkpoint 迁移说明不再把完整设计称为 `GAST-Goal`。

- [ ] **步骤 4：保留旧 state dict 加载**

旧 goal checkpoint 的参数路径继续映射到当前 `goal_decoder`；旧 metadata 中的 `gast_goal=True` 只用于兼容校验，不改变当前 canonical name。

- [ ] **步骤 5：运行模型测试确认通过**

运行：

```bash
pytest -q tests/test_gast_encoder.py tests/test_gast_goal_integration.py tests/test_gast_goal_decoder.py
```

预期：相关测试全部通过。

### 任务 3：统一训练、评估和配置一致性入口

**文件：**
- 修改：`scripts/训练与评估/run_trajimpute_experiments.py`
- 修改：`scripts/结果分析/evaluate_trajimpute_direct.py`
- 修改：`scripts/审计与校验/check_config_parity.py`
- 修改：`scripts/README.md`
- 修改：`tests/test_gast_encoder.py`

- [ ] **步骤 1：压缩 canonical variant registry**

训练和评估的新入口只注册 `M0`、`GAST`；完整 GAST 配置直接使用 goal decoder 与 trajectory classification。历史 encoder-only 实验只作为结果存档，不作为新方法选项。

- [ ] **步骤 2：限制新实验方法选项**

argparse 新实验 choices 只包含 `M0`、`GAST`。旧 `GAST-Goal` checkpoint 通过 `--variant GAST` 加载，canonical GAST 配置与其 state dict 保持兼容。

- [ ] **步骤 3：按 canonical variant 写 manifest**

新的训练/评估 manifest、输出 metadata 和结果目录统一写 `GAST`；旧目录与历史 JSON 保持原样，不覆盖或改写。

- [ ] **步骤 4：更新配置 parity 检查**

检查 active registry 只有 `M0`、`GAST`，并显式检查 `GAST` 使用完整 encoder、goal decoder 与 trajectory target。

- [ ] **步骤 5：运行入口和静态检查**

运行：

```bash
python scripts/审计与校验/check_config_parity.py
pytest -q tests/test_gast_encoder.py tests/test_resume_checkpoint_selection.py
```

预期：配置检查通过，相关测试全部通过。

### 任务 4：结果生成与研究文档统一为 GAST

**文件：**
- 修改：`docs/tools/gen_leaderboard_tables.py`
- 修改：`TrajGap-Bench.md`
- 修改：`README.md`
- 修改：`docs/results/README.md`
- 修改：`docs/results/实验总汇总.md`
- 修改：`docs/results/完整数据基线对比.md`
- 删除：`docs/results/主线实验记录.md`（合并至唯一汇总）
- 修改：`scripts/README.md`

- [ ] **步骤 1：合并榜单方法行**

榜单只保留一个 `GAST` 主方法行，使用当前 GAST-Goal 实验数值；旧标准 decoder 结果改为“GAST 消融（不含 goal 条件解码）”的内部消融说明，不再作为第二个正式方法名称。

- [ ] **步骤 2：统一文档叙述**

将“GAST-Goal 是最新模型、GAST 是底座”的叙述改为“GAST 是完整方法；goal 条件解码和轨迹级模式排序是 GAST 的内置设计”。

- [ ] **步骤 3：保留历史溯源但不制造新方法**

旧路径、旧 checkpoint 和历史实验标签只在脚注或复现说明中出现，明确其为 legacy artifact，不在主表和主贡献叙述中作为独立模型。

- [ ] **步骤 4：运行榜单生成器**

运行：

```bash
python docs/tools/gen_leaderboard_tables.py --check
```

预期：生成器检查通过，并且输出中的 canonical 方法名只有 `GAST`、M0 和外部基线。

### 任务 5：全量验证和命名残留审计

**文件：**
- 验证：上述代码、配置、测试和文档文件

- [ ] **步骤 1：运行全量测试**

运行：

```bash
pytest -q
```

预期：测试全部通过。

- [ ] **步骤 2：运行配置与榜单检查**

运行：

```bash
python scripts/审计与校验/check_config_parity.py
python docs/tools/gen_leaderboard_tables.py --check
```

预期：两项均返回退出码 0。

- [ ] **步骤 3：执行 canonical 命名残留审计**

运行：

```bash
rg -n --hidden -g '!outputs/**' -g '!*.pyc' \
  'GAST-Goal（最新模型|GAST-Goal.*本文方法|GAST-Goal.*latest|GAST-Goal.*主方法)' \
  README.md TrajGap-Bench.md docs scripts conf src tests
```

预期：不出现把 `GAST-Goal` 当作当前正式方法的叙述；允许旧 checkpoint 路径、兼容 alias 和历史结果脚注出现。

- [ ] **步骤 4：检查变更边界**

运行：

```bash
git diff --check
git status --short
```

确认没有修改训练产物、删除历史 checkpoint 或改写实验数值。
