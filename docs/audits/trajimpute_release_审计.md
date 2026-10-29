# TrajImpute release 来源核验与数据审计（TrajGap-Bench 底层数据）

日期：2026-09-07（核验与审计完成）｜归档：2026-10-21
定位：本文 TrajGap-Bench 的底层数据来源证明记录。原位于 `docs/research/选题.md` §八.0.1–§八.0.2，因该文档改写为论文描述而迁出，内容未改动事实。
代码引用：`src/datamodule/trajimpute_dataset.py` 头部注释指向本文件。

## 一、官方来源核验（仅依据官方仓库 github.com/Pranav-chib/TrajImpute、README 与论文 arXiv:2411.00174）

1. 场景：ETH、HOTEL、UNIV、ZARA1、ZARA2（缺失子集以 `-M` 后缀标注）；历史 8 帧、未来 12 帧（`generate_data.py`：`total_values=8`、`obs_traj_rel` 取前 8 帧、`pred_traj` 12 帧）。
2. 官方原始数据组织：脚本依赖 `obs_traj`、`pred_traj`、`pred_traj_rel`、`loss_mask`、`non_linear_ped`、`seq_start_end` 等字段，train/test/val 为三个独立 pkl，划分由 pkl 直接提供。release 实际字段、shape 与文件组织由下载审计确认（见第二节）。
3. `generate_data.py` 行为：读三个原始 pkl → 每轨迹从 {0,1,2,3,4} 均匀采样缺失帧数、`np.random.choice(8, k, replace=False)` 散布缺失位置、缺失值置 `np.nan` → 输出 `data_train/val/test.pkl`（新增 `missing_mask` 与 `obs_traj_rel`）；test 按每个缺失数复制 5 份。
4. Easy/Hard 缺失帧数：README 明确 Easy = 0–4 帧、Hard = 4–7 帧。release 实测：Easy train/val 每轨迹缺失 0–4 帧（近似均匀，均值约 25%）；Hard 每轨迹恰为 4/5/6/7 帧之一（均值约 68.7%）。`generate_data.py` **不含 Hard 生成逻辑**，Hard 由官方生成后直接发布。
5. 官方预测参考协议名称：README 使用 `Clean`、`Easy-impute`、`Hard-impute`，预测基准表明确在**插补后的数据**上评估（论文 §5.2 “prediction methods on the imputed subsets”）。
6. 官方原生直接预测评估器：**不提供**。`test.py` 为外部模型（EigenTrajectory 仓库风格）代码拷贝，依赖本仓库不存在的 `dataset.py`/`model.py`；ADE/FDE 由外部预测模型代码完成。
7. 发布形态：官方直接发布生成好的缺失数据（Google Drive release，README 链接），无需自行运行 `generate_data.py`。
8. 脚本专项风险：`generate_data.py` 测试集循环内 L69 生成 `test_missing_indices`、L71 赋 NaN 使用外层残留变量 `missing_indices` 的不一致确实存在于脚本文本；**但对官方 release 实测未生效**——test 各缺失数块内缺失位置逐轨迹变化，且 k 帧缺失恰呈现 C(8,k) 种位置组合（k=1→8、k=2→28、k=3→56、k=4→70）。裁定：正式实验以发布 pkl 为准，不重跑该脚本；若需重跑生成逻辑，须先建本地修正补丁并分别报告"官方脚本输出"与"修正版脚本输出"。

仍待核验：官方预测实验的聚合口径与外部评估实现细节（官方仓库无原生 evaluator）。

## 二、release 下载与全量审计（30/30 文件，0 违规）

位置：`~/TrajImpute/dataset/TrajImpute/`，约 238 MB；结构 5 场景 × 2 协议（Easy/Hard）× 3 split = 30 个 pkl。

1. **字段与 shape**：8 键字典；`obs_traj` float32 `[N,8,2]`（缺失处为 NaN），`pred_traj` float32 `[N,12,2]`，`missing_mask` bool `[N,8,2]`。
2. **掩码语义**：`missing_mask=True` 表示该坐标缺失；30/30 文件 `obs_traj` 的 NaN 计数与 `missing_mask.sum()` 严格相等。
3. **未来完整**：30 文件 `pred_traj` NaN 计数均为 0。
4. **Easy 分布**：train/val 每轨迹缺失 0–4 帧近均匀（均值约 25%）。
5. **Hard 分布**：每轨迹缺失恰为 4–7 帧之一（直方图仅 4/5/6/7 有质量，均值约 68.7%）。
6. **test 块结构**：Easy test 为 5 个整块（缺失数恰为 0/1/2/3/4），Hard test 为 4 个整块（4/5/6/7）；各块含该场景全部测试轨迹，`pred_traj` 跨块完全相同（10/10 场景×协议验证）。
7. **缺失位置多样性**：块内缺失位置逐轨迹变化，k 帧缺失呈现全部 C(8,k) 种组合 → 第一节第 8 条的脚本变量残留未影响发布数据。
8. **样本保障**：逐样本最大缺失帧数 ≤7，即每样本至少保留 1 个有效历史观测（0 违规）。
9. **划分隔离**：train/val/test 的 `pred_traj` 精确坐标集合两两无交集（5 场景均验证，泄漏 = 0）。

唯一测试轨迹数：ETH 181 / HOTEL 1053 / UNIV 24334 / ZARA1 2253 / ZARA2 5833。
train/val 规模（ETH-M 例）：train 29,809 / val 5,349 / test 905（=181×5）；`seq_start_end` 保留帧序列内行人分组信息。

坐标单位、世界系语义与 Social-GAN 预处理一致性按官方论文描述采用；评测聚合口径见第一节"仍待核验"。
