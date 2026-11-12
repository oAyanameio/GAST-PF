# scripts/ 目录结构

所有脚本统一约定：**从仓库根目录（/home/lbh/GAST-PF）运行**，入口为仓库根的 `train.py` / `eval.py`。

| 子目录 | 内容 |
|---|---|
| `训练与评估/` | TrajGap-Bench 训练评估：`run_trajimpute_experiments.py`（GAST 与单因素结构消融） |
| `审计与校验/` | 当前模型与配置的静态检查 |
| `结果分析/` | TrajGap-Bench direct 评估与汇总：`evaluate_trajimpute_direct.py`（按 TrajGap 数据集划分评估）、`summarize_trajimpute.py`（重训结果汇总） |

注意事项：

- 正式缺失实验使用 TrajGap 数据集和 TrajGap-Bench 评测协议；训练、验证与测试沿用 TrajImpute 官方数据划分。

- checkpoint 文件名含 `=`（如 `epoch=73.ckpt`）时，Hydra override 必须整体加引号：`"checkpoint='outputs/.../epoch=73.ckpt'"`。
- eval 输出目录已存在 `model.log` 会 FileExistsError，重跑前先删整个 eval 目录。
- `--variant` 可选 `GAST`、`GAST-no-interaction-bias`、`GAST-no-source-filter`、`GAST-no-state-strength-modulation`、`GAST-content-only-readout`、`GAST-no-goal-guidance`；后五项每次只关闭一个结构开关。
- 正式结果只报告普通 `minADE20` 和 `minFDE20`：对 20 条预测轨迹取最小误差，再对样本做等权算术平均，不使用模式分数或额外指标加权。
