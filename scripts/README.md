# scripts/ 目录结构

所有脚本统一约定：**从仓库根目录（/home/lbh/GAST-PF）运行**，入口为仓库根的 `train.py` / `eval.py`。

| 子目录 | 内容 |
|---|---|
| `训练与评估/` | TrajGap-Bench 训练评估：`run_trajimpute_experiments.py`（当前仅注册 M0 和 GAST） |
| `审计与校验/` | 当前模型与配置的静态检查 |
| `结果分析/` | TrajGap-Bench direct 评估与汇总：`evaluate_trajimpute_direct.py`（按 Mixed split 分组评估）、`summarize_trajimpute.py`（重训结果汇总） |

注意事项：

- 正式缺失实验统一使用 TrajGap-Bench：TrajImpute 官方 Easy/Hard 只作为底层文件来源，训练、验证、测试统一使用 Mixed。

- checkpoint 文件名含 `=`（如 `epoch=73.ckpt`）时，Hydra override 必须整体加引号：`"checkpoint='outputs/.../epoch=73.ckpt'"`。
- eval 输出目录已存在 `model.log` 会 FileExistsError，重跑前先删整个 eval 目录。
- 当前完整方法唯一名称为 `GAST`，goal 条件解码和轨迹级模式排序均为其内置设计；`M0` 是同协议历史基线。新训练与评估只注册 `GAST`、`M0`，旧实验目录名只用于结果溯源。
