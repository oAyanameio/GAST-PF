# scripts/ 目录结构

所有脚本统一约定：**从仓库根目录（/home/lbh/GAST-PF）运行**，入口为仓库根的 `train.py` / `eval.py`。

| 子目录 | 内容 |
|---|---|
| `训练与评估/` | TrajGap-Bench Mixed direct 训练评估：`run_trajimpute_experiments.py`（正式 GAST-PF runner，M0 为对照） |
| `审计与校验/` | 当前模型与配置的静态检查 |
| `结果分析/` | TrajImpute direct 评估与汇总：`evaluate_trajimpute_direct.py`（Mixed direct 分组评估）、`summarize_trajimpute.py`（重训结果汇总） |

注意事项：

- 正式缺失实验统一使用 TrajGap-Bench：TrajImpute 官方 Easy/Hard 只作为底层文件来源，训练、验证、测试统一使用 Mixed。

- checkpoint 文件名含 `=`（如 `epoch=73.ckpt`）时，Hydra override 必须整体加引号：`"checkpoint='outputs/.../epoch=73.ckpt'"`。
- eval 输出目录已存在 `model.log` 会 FileExistsError，重跑前先删整个 eval 目录。
- 当前正式模型入口为 GAST-PF；M0 仅用于同协议基线对照，历史校准实验不属于当前代码路径。
