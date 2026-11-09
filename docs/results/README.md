# 实验结果索引

结果按主题保留单一权威文档；原始运行产物仍在各实验目录。旧筛选路线、Easy/Hard 旧主榜和重复过程记录已清理。

## 文档清单

| 文档 | 内容 | 当前结果 |
|---|---|---|
| [实验总汇总](实验总汇总.md) | 当前 GAST 五场景结果、六个外部方法对比与缺失等级分层 | 同 revision 完整轮次 Avg minADE20/minFDE20 = 0.3168/0.4831；共 7 种方法完整 |
| [TrajGap-Bench 榜单](trajgap_bench_leaderboard.md) | 从各方法 `results.json` 生成的逐场景、Avg 与分层对比 | GAST 对 MoFlow Avg minADE20/minFDE20 改善 9.4%/7.6% |
| [TrajGap-Bench 数据集分布](trajgap_dataset_distribution.md) | 五个场景的 Mixed train/test/validation focal 轨迹样本数 | 含可复现的 PNG/PDF 分组柱状图和逐场景计数 |
| [完整数据基线对比](完整数据基线对比.md) | ETH/UCY Clean GAST 结果及公开文献参考；和 TrajGap-Bench 缺失历史主榜分开 | GAST Clean Avg 0.2260/0.3927 |

## 结果图

- [TrajGap 数据集分布（PNG）](../figures/results/trajgap_dataset_distribution.png) · [PDF](../figures/results/trajgap_dataset_distribution.pdf)：五个场景中 Mixed 训练、测试与验证 focal 轨迹样本数。
- [五场景基准对比（PNG）](../figures/results/scene_benchmark.png) · [PDF](../figures/results/scene_benchmark.pdf)：逐场景指标与 Avg，低误差方法排在上方。
- [缺失程度退化曲线（PNG）](../figures/results/missing_degradation.png) · [PDF](../figures/results/missing_degradation.pdf)：相对 missing=0 的 minFDE20 比值。
- [GAST 定性轨迹汇总（PNG）](../figures/trajgap_qualitative_moflow/gast_moflow_style_samples.png) · [PDF](../figures/trajgap_qualitative_moflow/gast_moflow_style_samples.pdf)：参照 MoFlow 样式，五场景 × 三类缺失程度（0、1–4、≥5）；从固定种子候选池筛选预测、GT 和历史轨迹都清晰可见的样本，用于形态展示而非随机代表样本或性能证据。黄色为基于 GT 终点选出的 best-of-20 oracle 预测。
- [本次附件图像的轨迹可视化](../figures/trajgap_qualitative/UNIV-M_attached_redcap_m3.png)：UNIV-M 测试样本，锚点与红帽行人位置按像素匹配，缺失 3 帧。
- 新版单样本高清图共 15 张，位于 `docs/figures/trajgap_qualitative_moflow/`。背景使用原版静态场景图；本地没有逐时刻同步视频帧。

## 口径约束

- 缺失历史主结果统一采用 TrajGap-Bench Mixed；Easy/Hard 仅为底层文件来源或缺失程度诊断。
- direct 与插补感知方法保留各自方法属性；表格注明测试协议，不把方法类别混同。
- train-adapt 与 zero-shot 分开报告；协议定义见《[选题](../research/选题.md)》，数据来源见《[TrajImpute release 审计](../audits/trajimpute_release_审计.md)》。
