# 实验结果索引

结果按主题保留单一权威文档；原始运行产物仍在各实验目录。旧筛选路线、Easy/Hard 旧主榜和重复过程记录已清理。

## 文档清单

| 文档 | 内容 | 当前结果 |
|---|---|---|
| [实验总汇总](实验总汇总.md) | 当前 GAST 五场景结果、六个外部方法对比与缺失等级分层 | 同 revision 完整轮次宏平均 minADE20/minFDE20 = 0.3168/0.4831；共 7 种方法完整 |
| [TrajGap-Bench 榜单](trajgap_bench_leaderboard.md) | 从各方法 `results.json` 生成的逐场景、宏/微平均和分层对比 | GAST 对 MoFlow 宏平均 minADE20/minFDE20 改善 9.4%/7.6% |
| [完整数据基线对比](完整数据基线对比.md) | ETH/UCY Clean GAST 结果及公开文献参考；和 TrajGap-Bench 缺失历史主榜分开 | GAST Clean 宏平均 0.2260/0.3927 |

## 口径约束

- 缺失历史主结果统一采用 TrajGap-Bench Mixed；Easy/Hard 仅为底层文件来源或缺失程度诊断。
- direct 与插补感知方法保留各自方法属性；表格注明测试协议，不把方法类别混同。
- train-adapt 与 zero-shot 分开报告；协议定义见《[选题](../research/选题.md)》，数据来源见《[TrajImpute release 审计](../audits/trajimpute_release_审计.md)》。
