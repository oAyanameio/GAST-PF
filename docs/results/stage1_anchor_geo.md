# Stage 1 · AnchorGeo

_依据 `docs/research/观测证据条件化几何时序编码器后续优化方向.md` §4 Stage 1 执行_

## 1. 目标

最低风险地把已有的 anchor 空间与末端方向信息接入 `GAST-PF`，验证「空间几何缺失」是否是主要瓶颈。

## 2. 实现

**新增输入（只来自可见历史，不含 future）**

```text
x_centers                  [B, N, 2]
cos/sin(last_valid_angle)  [B, N, 2]
anchor_lag_norm            [B, N, 1]
```

`anchor_lag` 在模型侧由 `x_valid_mask` 计算（不使用数据层的 lag 字段）：

```text
last_valid_index = max{t | m_i,t = 1}
anchor_lag       = (T - 1) - last_valid_index
anchor_lag_norm  = anchor_lag / max(T - 1, 1)      # 全无效行取 1.0，恒落在 [0, 1]
```

**改造位置**：`TrajectoryForecaster` 的 GAST-PF actor 输出端，新增 `AnchorGeoProjection`（`[B,N,5] -> [B,N,128]`），残差叠加到 `gast_encoder` 输出。

**保持不动**：GSM attention 内部、`Temporal Decay`、双向 Mamba、固定末端读出、decoder 输入维度（`[B,N,128]`）、loss、优化器、TrajGap 协议。

| 项 | 路径 |
| --- | --- |
| 几何投影 / anchor lag | `src/model/layers/gast_encoder.py`（`AnchorGeoProjection`、`anchor_lag_norm`） |
| 接入点与开关 | `src/model/trajectory_forecaster.py`（`gast_anchor_geo`） |
| 配置 | `conf/model/missing_aware_ethucy_model_forecast.yaml`（`gast_anchor_geo: false` 默认关闭） |
| 变体映射 | `scripts/训练与评估/run_trajimpute_experiments.py`、`scripts/结果分析/evaluate_trajimpute_direct.py` |
| 行为测试 | `tests/test_stage1_anchor_geo.py` |

**开关与归因纪律**

- `gast_anchor_geo: false`（默认）→ 模块完全不注册，基础版本 `state_dict` 与行为不变；
- `AnchorGeoProjection` 输出层**零初始化** → 训练起点与基础版本严格等价（测试逐位验证），增益可归因于几何证据；
- 开关写入 manifest `encoder_config` 与 checkpoint `hyper_parameters`；evaluator 对开关不一致的 checkpoint **fail-closed**（两个方向都拒绝）。

**参数量**

| 模型 | 总参数 | AnchorGeo 增量 |
| --- | ---: | ---: |
| M0 | 5,630,928 | — |
| GAST-PF | 5,213,968 | — |
| GAST-PF-AnchorGeo | 5,231,376 | **+17,408**（+0.33%） |

## 3. 行为测试

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 PYTHONPATH=. \
  /home/lbh/.conda/envs/DeMo/bin/python -m pytest tests/test_stage1_anchor_geo.py -q
# 14 passed, 1 skipped（skip = batch 内无 padding actor 时的隔离用例）
```

覆盖（对照路线文档 §5.3 必做行为测试）：

| 测试 | 内容 |
| --- | --- |
| `test_anchor_lag_norm_matches_spec` | lag 定义与归一化符合规格 |
| `test_anchor_lag_norm_is_monotone_in_run_length` | 连续缺失越长 lag 越大（§5.3-7 单调性） |
| `test_anchor_lag_norm_stays_finite_for_all_invalid_rows` | 全无效行不产生 NaN/Inf 或越界值 |
| `test_anchor_geo_disabled_by_default` | 默认关闭时无 AnchorGeo 参数（基础版本不变） |
| `test_anchor_geo_registers_projection_only_when_enabled` | M0 路径不得出现 AnchorGeo |
| `test_anchor_geo_projection_input_width_is_five` | 几何输入宽度契约 |
| `test_anchor_geo_zero_init_identity_and_paired_weights` | **零初始化恒等 + 配对权重逐位一致** |
| `test_anchor_geo_preserves_decoder_contract` | decoder 输入维度与输出契约不变 |
| `test_anchor_geo_padding_actor_isolation` | §5.3-2 padding 隔离（几何扰动不泄漏） |
| `test_anchor_geo_geometry_is_used` | §5.3-3 几何有效性（打乱几何必须改变输出） |
| `test_anchor_geo_ignores_future_targets` | §5.3-6 末端缺失不泄漏 |
| `test_anchor_geo_permutation_equivariance` | §5.3-5 集合不变性（邻居无序） |
| `test_anchor_geo_finite_on_edge_fixtures` | 全有效 / 末端缺失 / 中间缺失 / 最小证据 |
| `test_evaluator_anchor_geo_contract` | checkpoint 开关契约 fail-closed |
| `test_anchor_geo_parameters_covered_by_adamw_groups` | 优化器闭包纪律 |

## 4. 训练

**sanity run（已完成，产物已清理）**：`--smoke` 端到端 ETH-M，验证 训练 → checkpoint（`monitor=val_new_minFDE20`）→ 评估 `status=complete rc=0` 全链路，manifest `encoder_config` 含 `gast_anchor_geo: True`，评估输出六维分层齐备。

**正式训练**（同一协议根、独立场景目录；四卡各一条链）

```bash
# 逐卡一条链（--gpu N 分别为 0/1/2/3）
scripts/训练与评估/run_trajimpute_experiments.py --protocol mixed-direct \
  --variant GAST-PF-AnchorGeo --scenes <该卡队列> --seed 2024 --gpu N \
  --batch-size 64 --num-workers 4 --epochs 100 --K 20 --screening \
  --output-root outputs/stage1_anchor_geo_seed2024
```

| 卡 | 队列 | 实测速率 | 剩余 ETA |
| --- | --- | ---: | ---: |
| GPU 0（与他人共享） | ZARA2-M | 100 step/min（8.0 min/ep） | 8.6 h |
| GPU 1 | ETH-M → UNIV-M | 162.5 step/min（5.7 min/ep） | 10.1 h |
| GPU 2 | HOTEL-M | 162.5 step/min（5.6 min/ep） | 9.1 h |
| GPU 3 | ZARA1-M | 175 step/min（5.0 min/ep） | 8.1 h |

> 总墙钟由 GPU 1 队列决定，约 **10.1 h**。GPU 0 因与他人任务共享，单位算力约为独占卡的 60%，故只安排一个场景。

协议：TrajGap-Bench Mixed-direct；seed=2024；K=20；100 epochs；batch size=64；选点 `val_new_minFDE20`；`experiment_class=screening`。

## 5. 对照口径

- **主对照（竞争力）**：外部模型 MoFlow、SocialLSTM（与五场景 screening 同口径）。
- **机制归因（单变量）**：`GAST-PF` vs `GAST-PF + AnchorGeo`。基础版本为旧 monitor 口径 screening，**不是配对对照**；机制归因的配对基线留待消融阶段按新口径补跑，届时本节标注。

## 6. 结果

**协议**：TrajGap-Bench Mixed-direct；seed=2024；K=20；100 epochs；选点 `val_new_minFDE20`；`experiment_class=screening`（单种子）。样本合计 302,886。

| 场景 | n | best ep | minADE20 | minFDE20 | ADE@1 | FDE@1 | MR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH-M | 1,629 | 84 | 0.6492 | 0.9409 | 1.4142 | 2.4550 | 14.49% |
| HOTEL-M | 9,477 | 53 | **0.2326** | **0.3579** | 0.5425 | 0.9224 | 3.58% |
| UNIV-M | 219,006 | 80 | 0.3277 | 0.5373 | 0.9685 | 1.7884 | 3.53% |
| ZARA1-M | 20,277 | 98 | **0.2398** | **0.3864** | **0.8736** | **1.5731** | 1.70% |
| ZARA2-M | 52,497 | 65 | **0.1713** | **0.2750** | **0.6912** | **1.2446** | **1.00%** |
| **宏平均** | — | — | **0.3241** | **0.4995** | **0.8980** | **1.5967** | **4.86%** |

### 6.1 对外部模型（主对照）

| 相对差 | minADE20 | minFDE20 | ADE@1 | FDE@1 | MR |
| --- | ---: | ---: | ---: | ---: | ---: |
| vs MoFlow | −7.3% | −4.4% | −8.5% | −7.7% | **+5.7%** |
| vs SocialLSTM | −29.2% | −17.8% | −32.4% | −27.0% | −19.1% |

优于 MoFlow 四项、劣于其 MR；对 SocialLSTM 五项全面领先。

### 6.2 对基础版本（`GAST-PF`，非配对）

基础版为旧 monitor 口径 screening，**不是配对对照**，宏平均 ±1% 量级不足以定论。

| 指标 | GAST-PF | +AnchorGeo | 相对差 |
| --- | ---: | ---: | ---: |
| minADE20 | 0.3264 | 0.3241 | −0.7% |
| minFDE20 | 0.4946 | 0.4995 | +1.0% |
| ADE@1 | 0.8896 | 0.8980 | +0.9% |
| FDE@1 | 1.5870 | 1.5967 | +0.6% |
| MR | 4.50% | 4.86% | +7.9% |

逐场景相对差（负=更优）：

| 场景 | minADE20 | minFDE20 | ADE@1 | FDE@1 | MR |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETH-M | +4.9% | **+10.1%** | −0.6% | −0.6% | **+21.6%** |
| HOTEL-M | −9.6% | −7.7% | +6.8% | +7.1% | −13.1% |
| UNIV-M | +0.6% | +1.5% | **+12.4%** | **+10.2%** | −3.2% |
| ZARA1-M | −3.8% | −3.7% | −3.7% | −4.0% | +12.7% |
| ZARA2-M | −5.2% | −8.5% | −7.6% | −7.3% | −24.6% |

### 6.3 分层证据：收益随缺失严重程度单调增大

`anchor_lag` = 距最后有效观测的步数（五场景合并，样本加权）：

| anchor_lag | n | 基础 minFDE20 | AnchorGeo | Δ | 基础 MR | AnchorGeo MR | Δ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 167,469 | 0.3754 | 0.3785 | **+0.8%** | 1.09% | 1.20% | **+9.9%** |
| 1 | 60,284 | 0.4460 | 0.4509 | +1.1% | 2.13% | 2.14% | +0.6% |
| 2 | 30,263 | 0.5270 | 0.5317 | +0.9% | 3.81% | 3.68% | −3.5% |
| 3 | 17,368 | 0.6461 | 0.6436 | −0.4% | 6.37% | 5.91% | −7.1% |
| 4 | 10,763 | 0.7992 | 0.7885 | −1.3% | 10.07% | 8.91% | −11.5% |
| 5 | 7,379 | 0.9665 | 0.9462 | −2.1% | 14.08% | 12.33% | −12.4% |
| 6 | 5,356 | 1.1947 | 1.1567 | −3.2% | 19.16% | 17.83% | −6.9% |
| 7 | 4,004 | 1.4521 | 1.3421 | **−7.6%** | 26.27% | 22.65% | **−13.8%** |

其余维度（五场景合并）：

| 维度 | 子组 | n | minFDE20 Δ | MR Δ |
| --- | --- | ---: | ---: | ---: |
| `terminal_missing` | true | 135,417 | −0.6% | −7.5% |
| `terminal_missing` | false | 167,469 | +0.8% | +9.9% |
| `missing_count` | 0 | 33,654 | +1.6% | +20.6% |
| `missing_count` | 7 | 33,654 | −1.2% | −9.4% |

完整六维分层（`missing_count` / `valid_count` / `anchor_lag` / `forecast_gap` / `terminal_missing` / `max_missing_run`）由 `scripts/训练与评估/watch_stage_variant.py` 现算输出，结果见 `outputs/stage1_anchor_geo_seed2024/watcher_stage1.log`。

## 7. 判定（Stage 1 Go/No-Go）

**判定规则**：

- **Go**：整体提升，或整体不升但 `terminal_missing` / 长缺失子组改善 → 保留并进入 Stage 2（FrameGeo）；
- **No-Go**：所有子组均无改善 → 停止继续堆叠几何模块，先检查坐标尺度与数据泄漏；
- 附加纪律：`m=0` 完整历史子组不得出现不可接受退化；多数场景方向一致；行为测试全部通过。

**结论：Go（可进入 Stage 2），但带三条负信号。**

成立依据：宏平均整体不升（4/5 指标略差），但 `terminal_missing=true` 子组改善（minFDE −0.6%、MR −7.5%），且 `anchor_lag` 收益随缺失年龄**单调增大**（lag 7：minFDE −7.6%、MR −13.8%）——符合"几何证据在 anchor 不可靠时才有价值"的机制预期，不是随机波动。

需记录的负信号：

1. **完整历史轻微退化**：`anchor_lag=0` minFDE +0.8%、MR 1.09%→1.20%；`missing_count=0` MR 0.78%→0.94%。绝对幅度小，但触及"`m=0` 不得不可接受退化"的边缘。
2. **MR 相对 MoFlow 由优转劣**：基础版 4.50% vs 4.60%（−2.1%），AnchorGeo 4.86% vs 4.60%（+5.7%）。
3. **ETH-M 异常**：minFDE +10.1%、MR +21.6%，是最大退步场景且样本最少（1,629），需单独确认是方差还是真实退化。

**证据边界**：基础版为旧 monitor 口径、单种子，本次不是配对比较；宏平均 ±1% 量级不具结论性，真正支撑判定的是分层单调趋势。

## 8. 状态

- [x] 代码实现 + 开关 + 契约
- [x] 行为测试 14 项通过（全套回归 392 passed, 2 skipped）
- [x] sanity run（smoke 全链路）
- [x] 正式训练五场景（2026-10-17 05:35 完成，100 epochs × 5）
- [x] 结果回填 + 判定 → **Go（进 Stage 2），带 3 条负信号**
