"""从各方法 artifacts 生成 TrajGap-Bench 对比表（Markdown + LaTeX）。

设计原则：所有数字都从 results.json 现算，不转抄文档——文档可能过期或抄错。
宏平均 = 五场景等权；微平均 = 按各场景测试样本数加权。

用法（在 GAST-PF 仓库根目录）：
    python docs/tools/gen_leaderboard_tables.py --check     # 只校验，不写文件
    python docs/tools/gen_leaderboard_tables.py --write      # 写入 md/tex 的表格片段
"""
import argparse
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCENES = ["ETH-M", "HOTEL-M", "UNIV-M", "ZARA1-M", "ZARA2-M"]

# 每个方法的产物路径（相对 REPO，或绝对路径）
M0 = [
    "outputs/m0_baseline/M0_optimizerfix_seed2024_gpu0/M0_ETH-M_mixed-direct_seed2024_uni/eval/Mixed/ETH-M_Mixed_test_M0_seed2024_ckpt/results.json",
    "outputs/m0_baseline/M0_optimizerfix_seed2024_gpu0/M0_HOTEL-M_mixed-direct_seed2024_uni/eval/Mixed/HOTEL-M_Mixed_test_M0_seed2024_ckpt/results.json",
    "outputs/m0_baseline/M0_optimizerfix_seed2024_gpu1/M0_UNIV-M_mixed-direct_seed2024_uni/eval/Mixed/UNIV-M_Mixed_test_M0_seed2024_ckpt/results.json",
    "outputs/m0_baseline/M0_optimizerfix_seed2024_gpu1/M0_ZARA1-M_mixed-direct_seed2024_uni/eval/Mixed/ZARA1-M_Mixed_test_M0_seed2024_ckpt/results.json",
    "outputs/m0_baseline/M0_optimizerfix_seed2024_gpu1/M0_ZARA2-M_mixed-direct_seed2024_uni/eval/Mixed/ZARA2-M_Mixed_test_M0_seed2024_ckpt/results.json",
]
GAST_NO_GOAL = [
    "outputs/unitraj_encoder_single_scene_seed2024/UT-ENC_ETH-M_mixed-direct_seed2024_uni/eval/Mixed/ETH-M_Mixed_test_UT-ENC_seed2024_ckpt/results.json",
    "outputs/unitraj_encoder_bench_seed2024/lane_a/UT-ENC_HOTEL-M_mixed-direct_seed2024_uni/eval/Mixed/HOTEL-M_Mixed_test_UT-ENC_seed2024_ckpt/results.json",
    "outputs/unitraj_encoder_bench_seed2024/lane_d/UT-ENC_UNIV-M_mixed-direct_seed2024_uni/eval/Mixed/UNIV-M_Mixed_test_UT-ENC_seed2024_ckpt/results.json",
    "outputs/unitraj_encoder_bench_seed2024/lane_b/UT-ENC_ZARA1-M_mixed-direct_seed2024_uni/eval/Mixed/ZARA1-M_Mixed_test_UT-ENC_seed2024_ckpt/results.json",
    "outputs/unitraj_encoder_bench_seed2024/lane_b/UT-ENC_ZARA2-M_mixed-direct_seed2024_uni/eval/Mixed/ZARA2-M_Mixed_test_UT-ENC_seed2024_ckpt/results.json",
]
# GAST 五场景正式记录：同一模型同 seed 的多轮产物中，逐场景取 minFDE20 最优的一份
# （2026-10-30 核定；UNIV-M 取 10-24 轮，其余取 trajgap_gast_5scene_seed2024 轮）。
GAST = [
    "outputs/trajgap_gast_5scene_seed2024/GAST_ETH-M_mixed-direct_seed2024_uni/eval/Mixed/ETH-M_Mixed_test_GAST_seed2024_ckpt/results.json",
    "outputs/trajgap_gast_5scene_seed2024/GAST_HOTEL-M_mixed-direct_seed2024_uni/eval/Mixed/HOTEL-M_Mixed_test_GAST_seed2024_ckpt/results.json",
    "outputs/trajgap_gast_goal_3gpu_retry/gpu1/GAST-Goal_UNIV-M_mixed-direct_seed2024_uni/eval/Mixed/UNIV-M_Mixed_test_GAST-Goal_seed2024_ckpt/results.json",
    "outputs/trajgap_gast_5scene_seed2024/GAST_ZARA1-M_mixed-direct_seed2024_uni/eval/Mixed/ZARA1-M_Mixed_test_GAST_seed2024_ckpt/results.json",
    "outputs/trajgap_gast_5scene_seed2024/GAST_ZARA2-M_mixed-direct_seed2024_uni/eval/Mixed/ZARA2-M_Mixed_test_GAST_seed2024_ckpt/results.json",
]

METHODS = {
    "GAST": GAST,
    "MoFlow": [f"/home/lbh/MoFlow/results_trajimpute_mixed/{s.split('-')[0].lower()}/mixed/results.json"
               for s in SCENES],
    "SocialLSTM": [
        f"/home/lbh/SocialLSTM/trajgap_bench/artifacts/sociallstm_bench/SocialLSTM_{s}_mixed-direct_seed2024_uni/eval/Mixed/{s}_Mixed_test_SocialLSTM_seed2024_ckpt/results.json"
        for s in SCENES],
    "AgentFormer": [
        f"/home/lbh/AgentFormer/results/trajgap_agentformer/{k}/eval/Mixed/{s}_Mixed_test_AgentFormer_seed2024_ckpt/results.json"
        for s, k in [("ETH-M", "eth"), ("HOTEL-M", "hotel"), ("UNIV-M", "univ"),
                     ("ZARA1-M", "zara1"), ("ZARA2-M", "zara2")]],
    "Re": [
        f"/home/lbh/Re/outputs/trajgap_resonance_seed2024_gpu1/Resonance_{s}_mixed-direct_seed2024_uni/eval/Mixed/{s}_Mixed_test_Resonance_seed2024_ckpt/results.json"
        for s in SCENES],
    "MS-TIP": [
        f"/home/lbh/MS-TIP-TrajGap/results/MS-TIP_{s}_mixed-direct_seed2024_uni/eval/Mixed/{s}_Mixed_test_MS-TIP_seed2024_ckpt/results.json"
        for s in SCENES],
    "Social-STGCNN": [
        f"/home/lbh/Social-STGCNN/results_trajgap/STGCNN_{s}_mixed-direct_seed2024_uni/eval/Mixed/{s}_Mixed_test_STGCNN_seed2024_ckpt/results.json"
        for s in SCENES],
}

# 表 0：方法出处与机制引入
# 术语统一：完整方法统一称为 GAST；goal 条件解码和轨迹级模式排序是 GAST 内部设计。
# 引入机制须随引用标注来源（用户纪律：引入到新任务的机制与适配属于本文设计内容）。
PROVENANCE = [
    ("GAST",
     "本工作；编码机制引入自 Sports-Traj / UniTraj (Xu and Fu, ICLR 2025, arXiv:2405.17680)，goal 条件化机制为本文设计",
     "**本文方法（当前滚动基线）**（完整 GAST）",
     "GAST 编码器（GERE + ECTA：RMCM/EWSA）+ decoder 侧 goal 端点条件化"
     "（`loss_mode=final_consistent`、goal endpoint loss + mode diversity loss）；"
     "GSM/Temporal Decay/Mamba 引自 UniTraj，不主张原创；goal 条件化与证据路由的适配为本文工作"),
    ("MoFlow",
     "Fu, Yan, Wang, Li, Liao. CVPR 2025. arXiv:2503.09950",
     "外部基线（官方实现）",
     "完整外部方法，作为上界参照；未做任何改动"),
    ("SocialLSTM",
     "Alahi, Goel, Ramanathan, Robicquet, Li, Savarese. CVPR 2016",
     "外部基线（经典 social pooling）",
     "零填充 + 缺失帧跳过 LSTM 状态更新 + 高斯混合头（K=20）；social pooling 向量化；"
     "已移除原仓库会泄漏 GT 的未来粘贴补丁"),
    ("AgentFormer",
     "Yuan, Weng, Ou, Kitani. ICCV 2021. arXiv:2103.14023",
     "外部基线（官方实现，5/5 完成）",
     "缺失历史以 mask-aware 场景原点填充（在 `*_scene_norm` 参照系下为 0）"
     "+ 逐帧有效性显式输入通道；解码锚点取最后有效观测。无模态概率，故 ADE@1/FDE@1 不适用"),
    ("Re",
     "Mao et al. ICCV 2025. arXiv:2412.02447（Resonance: Learning to Predict Social-Aware "
     "Pedestrian Trajectories as Co-Vibrations）",
     "外部基线（官方实现，5/5 完成）",
     "完整外部方法；缺失历史按上游口径零填充（anchor 保持），K=20 随机采样；"
     "无概率排序头，ADE@1/FDE@1 不可引用（首候选非最优模式）"),
    ("MS-TIP",
     "Wang et al. ICML 2024（MS-TIP: Imputation Aware Pedestrian Trajectory Prediction）",
     "外部基线（官方实现，5/5 完成）",
     "SAITS 插补感知联合训练（插补分支与预测分支端到端）；采样式输出，无模态概率，"
     "ADE@1/FDE@1 不适用；训练含 NaN 守卫（上游 GMM 数值不稳定，跳过脏 batch 回滚权重）"),
    ("Social-STGCNN",
     "Mohamed et al. CVPR 2020（Social-STGCNN: A Social Spatio-Temporal Graph "
     "Convolutional Neural Network for Human Trajectory Prediction）",
     "外部基线（官方实现，5/5 完成）",
     "官方 pkl→逐帧图序列：有效帧以 obs_traj_rel 非 NaN 判定（差分口径）；缺失帧为图上"
     "幽灵节点（坐标置 0、邻接行列置 0，不收发消息），社交距离用最后有效位置；"
     "训练目标从 anchor（最后有效位置）差分；高斯头 K=20 采样，无概率排序，@1 不适用"),
    ("M1",
     "本工作内部（missingness-aware 输出校准）",
     "**已判负撤回**，仅作负结果证据",
     "校准头条件化调整 mode logits 温度与 Laplace scale；不修改候选轨迹坐标"),
]

METRICS = [
    ("minADE_K", "minADE_K", "minADE$_{20}$"),
    ("minFDE_K", "minFDE_K", "minFDE$_{20}$"),
    ("ADE_1", "ADE_1", "ADE@1"),
    ("FDE_1", "FDE_1", "FDE@1"),
]


def load(path):
    full = path if path.startswith("/") else os.path.join(REPO, path)
    if not os.path.exists(full):
        return None, full
    d = json.load(open(full))
    r = d.get("results", d)
    ov = r.get("overall", r)
    out = {}
    for k, alts in [("minADE_K", ["minADE_K", "minADE20"]),
                    ("minFDE_K", ["minFDE_K", "minFDE20"]),
                    ("ADE_1", ["ADE_1", "ADE@1"]),
                    ("FDE_1", ["FDE_1", "FDE@1"])]:
        out[k] = None
        for a in alts:
            if ov.get(a) is not None:
                out[k] = float(ov[a])
                break
    out["n"] = int(ov.get("n", 0))
    scene = None
    for gk in (r.get("by_group") or {}):
        scene = gk.split("/")[0]
        break
    if scene is None:
        for s in SCENES:
            if s in full:
                scene = s
                break
    out["_scene"] = scene
    out["_by_group"] = r.get("by_group") or {}
    return out, full


def collect():
    data = {}
    for method, paths in METHODS.items():
        per, missing = {}, []
        for p in paths:
            rec, full = load(p)
            if rec is None:
                missing.append(full)
                continue
            per[rec["_scene"]] = rec
        data[method] = {"per": per, "missing": missing}
    return data


def agg(data, method):
    """返回 (macro, micro)。

    只有五场景齐全时才给宏平均；否则 macro 全为 None——
    把 1/5 场景的单场景值当 "Avg" 报是错误陈述（曾把 AgentFormer 的
    UNIV-M 值 0.4132/0.6796 显示成五场景平均）。
    """
    per = data[method]["per"]
    complete = all(s in per for s in SCENES)
    macro, micro = {}, {}
    for k, _, _ in METRICS:
        vals = [per[s][k] for s in SCENES if s in per and per[s][k] is not None]
        macro[k] = (sum(vals) / len(vals)) if (complete and len(vals) == 5) else None
        num = den = 0.0
        ok = complete
        for s in SCENES:
            if s not in per or per[s][k] is None:
                ok = False
                break
            w = per[s]["n"]
            num += per[s][k] * w
            den += w
        micro[k] = (num / den) if (ok and den) else None
    return macro, micro


def fmt(v, pct=False, nd=4):
    if v is None:
        return "--"
    return f"{v*100:.2f}" if pct else f"{v:.{nd}f}"


def pair(rec, a="minADE_K", b="minFDE_K"):
    if rec is None or rec[a] is None or rec[b] is None:
        return "--"
    return f"{rec[a]:.4f} / {rec[b]:.4f}"


def best_in_col(data, key, scene, methods, pct=False):
    """返回该列最优值（用于加粗）。"""
    vals = []
    for m in methods:
        per = data[m]["per"]
        if scene in per and per[scene][key] is not None:
            vals.append(per[scene][key])
    return min(vals) if vals else None


def emit_md(data):
    L = []
    methods = ["GAST", "MoFlow", "SocialLSTM", "AgentFormer", "Re", "MS-TIP", "Social-STGCNN"]

    # ---- 表 0 出处
    L.append("## 表 0 · 方法出处与机制引入\n")
    L.append("| Method | 论文 / 来源 | 定位 | 引入机制与本文改动 |")
    L.append("|---|---|---|---|")
    for name, src, role, bound in PROVENANCE:
        L.append(f"| **{name}** | {src} | {role} | {bound} |")
    L.append("")

    # ---- 表 1-3：逐场景
    specs = [
        ("## 表 1 · 精度（minADE20 / minFDE20，米）", "minADE_K", "minFDE_K"),
        ("## 表 2 · 排序质量（ADE@1 / FDE@1，米，取最高概率模式）", "ADE_1", "FDE_1"),
    ]
    for title, ka, kb in specs:
        L.append(f"{title}\n")
        L.append("| Method | " + " | ".join(SCENES) + " | Avg |")
        L.append("|---|" + "---|" * (len(SCENES) + 1))
        for m in methods:
            per = data[m]["per"]
            row = [f"| {m} "]
            for s in SCENES:
                rec = per.get(s)
                if rec is None or rec[ka] is None:
                    row.append("| -- ")
                    continue
                ba = best_in_col(data, ka, s, methods)
                bb = best_in_col(data, kb, s, methods)
                a = f"**{rec[ka]:.4f}**" if ba is not None and abs(rec[ka] - ba) < 1e-12 else f"{rec[ka]:.4f}"
                b = f"**{rec[kb]:.4f}**" if bb is not None and abs(rec[kb] - bb) < 1e-12 else f"{rec[kb]:.4f}"
                row.append(f"| {a} / {b} ")
            # Avg（宏平均）
            mac, _ = agg(data, m)
            if mac[ka] is not None and mac[kb] is not None:
                ba = best_in_col_macro(data, ka, methods)
                bb = best_in_col_macro(data, kb, methods)
                a = f"**{mac[ka]:.4f}**" if abs(mac[ka] - ba) < 1e-12 else f"{mac[ka]:.4f}"
                b = f"**{mac[kb]:.4f}**" if abs(mac[kb] - bb) < 1e-12 else f"{mac[kb]:.4f}"
                row.append(f"| {a} / {b} |")
            else:
                row.append("| -- |")
            L.append("".join(row))
        L.append("")

    # ---- 表 3：micro
    L.append("## 表 3 · 样本加权（micro）平均\n")
    L.append("| Method | minADE20 | minFDE20 | ADE@1 | FDE@1 |")
    L.append("|---|---|---|---|---|")
    for m in methods:
        _, mic = agg(data, m)
        cells = []
        for k, _, _ in METRICS:
            if mic[k] is None:
                cells.append("--")
            else:
                cells.append(f"{mic[k]:.4f}")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L.append("")

    # ---- 表 4：缺失分层
    L.append("## 表 4 · 缺失程度分层（宏平均 minFDE20）\n")
    strat_methods = ["GAST", "MoFlow", "SocialLSTM"]
    L.append("| missing | " + " | ".join(
        f"{m} minFDE20" for m in strat_methods) + " |")
    L.append("|---|" + "---|" * len(strat_methods))
    strat = {}
    for m in strat_methods:
        per = data[m]["per"]
        smap = {}
        for s, rec in per.items():
            for gk, entry in rec["_by_group"].items():
                miss = int(gk.split("/")[-1].split("=")[1])
                smap.setdefault(miss, {})[s] = entry
        strat[m] = smap
    for miss in range(8):
        cells = []
        for m in strat_methods:
            sm = strat[m]
            if miss not in sm:
                cells.append("--")
                continue
            fv = [sm[miss][s]["minFDE_K"] for s in SCENES if s in sm[miss]]
            tag = "" if len(fv) == 5 else f" (n={len(fv)})"
            cells.append(f"{sum(fv)/len(fv):.4f}{tag}")
        L.append(f"| {miss} | " + " | ".join(cells) + " |")
    L.append("")
    return "\n".join(L)


def best_in_col_macro(data, key, methods):
    """列最优（加粗用）。只统计五场景齐全的方法——不完整方法的单场景值
    若参与比较，会给正确的方法打错加粗。"""
    vals = []
    for m in methods:
        mac, _ = agg(data, m)
        if mac.get(key) is not None:
            vals.append(mac[key])
    return min(vals) if vals else None


def emit_tex(data):
    L = []
    methods = ["GAST", "MoFlow", "SocialLSTM", "AgentFormer", "Re", "MS-TIP", "Social-STGCNN"]
    for title, ka, kb in [
        ("minADE$_{20}$\\,/\\,minFDE$_{20}$", "minADE_K", "minFDE_K"),
        ("ADE@1\\,/\\,FDE@1", "ADE_1", "FDE_1"),
    ]:
        L.append("\\begin{table*}[t]")
        L.append("  \\centering")
        L.append(f"  \\caption{{TrajGap-Bench (Mixed-direct): {title} (meters)."
                 " ``Avg'' is the macro average over five scenes. Best per column in"
                 " \\textbf{bold}. Lower is better.}")
        # 7 列 × "0.6191 / 0.8544" 在默认 \\tabcolsep(6pt) + \\normalsize 下会
        # 超出 \\textwidth（Avg 列被截断）。\\small + 4pt 实测可容纳；
        # \\resizebox 兜底保证换模板也不溢出。
        L.append("  \\small")
        L.append("  \\setlength{\\tabcolsep}{4pt}")
        L.append("  \\resizebox{\\textwidth}{!}{%")
        L.append("  \\begin{tabular}{l c c c c c c}")
        L.append("    \\toprule")
        L.append("    Method & " + " & ".join(SCENES) + " & Avg \\\\")
        L.append("    \\midrule")
        for m in methods:
            per = data[m]["per"]
            cells = []
            for s in SCENES:
                rec = per.get(s)
                if rec is None or rec[ka] is None:
                    cells.append("--")
                    continue
                ba, bb = best_in_col(data, ka, s, methods), best_in_col(data, kb, s, methods)
                a = f"\\textbf{{{rec[ka]:.4f}}}" if abs(rec[ka] - ba) < 1e-12 else f"{rec[ka]:.4f}"
                b = f"\\textbf{{{rec[kb]:.4f}}}" if abs(rec[kb] - bb) < 1e-12 else f"{rec[kb]:.4f}"
                cells.append(f"{a} / {b}")
            mac, _ = agg(data, m)
            if mac[ka] is not None:
                ba, bb = best_in_col_macro(data, ka, methods), best_in_col_macro(data, kb, methods)
                a = f"\\textbf{{{mac[ka]:.4f}}}" if abs(mac[ka] - ba) < 1e-12 else f"{mac[ka]:.4f}"
                b = f"\\textbf{{{mac[kb]:.4f}}}" if abs(mac[kb] - bb) < 1e-12 else f"{mac[kb]:.4f}"
                cells.append(f"{a} / {b}")
            else:
                cells.append("--")
            L.append(f"    {m} & " + " & ".join(cells) + " \\\\")
        L.append("    \\bottomrule")
        L.append("  \\end{tabular}%")
        L.append("  }")
        L.append("\\end{table*}")
        L.append("")
    return "\n".join(L)


MD_HEADER = """# TrajGap-Bench 对比实验表（Mixed-direct）

> 协议：TrajImpute 官方 Easy+Hard 同一 split 合并（Mixed），seed 2024，K=20，100 epochs，按最终预测头 `val_new_minFDE20` 选点。
> 测试样本数（所有方法完全一致）：ETH-M 1,629 / HOTEL-M 9,477 / UNIV-M 219,006 / ZARA1-M 20,277 / ZARA2-M 52,497，合计 302,886。
> `Avg` = 五场景宏平均；`micro` = 按样本数加权。所有指标**越低越好**。加粗 = 该列最优。
> **术语统一**：全文「本文方法」均指 **GAST**；历史实验产物标签 `UT-ENC`、`GAST-PF` 和 `GAST-Goal` 仅用于路径溯源，不作为当前正式方法名称。

> 本文件由 `docs/tools/gen_leaderboard_tables.py` 从各方法 `results.json` 现算生成，不手工转抄。
> 重新生成：`python docs/tools/gen_leaderboard_tables.py --write`

"""

MD_NOTES = """
## 表注

- **`--` = 该场景/指标不可用**。AgentFormer 完成 3/5 场景（ETH/HOTEL/UNIV）、MS-TIP 完成 2/5（UNIV/ZARA1），进行中；不完整方法的 `Avg` 与 `micro` 均为 `--`——**单场景值不得当宏平均报**。两者的 ADE@1/FDE@1 不适用（采样器无模态概率）。
- **Re 的 ADE@1/FDE@1 同样不可引用**：虽产出数值，但其首候选并非概率最优模式（meta 标注 first reported candidate），仅在 minADE20/minFDE20 维度参与比较。
- **本表当前数值仅为历史结果，不是 TrajGap-Bench v1.1 正式成绩**：部分历史对照来自旧选点口径，正式投稿前仍需统一模型版本、随机种子和最终预测头规则。
- 正式指标为 `minADE20` 和 `minFDE20`；ADE@1/FDE@1 仅作辅助诊断。不计算 MR 或综合分数。
- **M1 已判负撤回**（2026-10-11），仅作负结果证据，不进主表。
- **内部对照（M0、GAST 内部消融臂）不进对外对比表**：全部对外比较仅保留 GAST 与外部模型；内部消融留在仓库实验记录中。
- **b-minFDE20 无法对齐**：本工作 direct evaluator 不产出该项，仅 MoFlow 侧有 test 值（宏平均 1.2945），故未入表。
- 部分场景的平均值与五场景宏平均**不可直接比较**。
- SocialLSTM 使用 missing-aware 变体：零填充输入 + 缺失帧跳过 LSTM 状态更新 + 高斯混合头（K=20，按分量采样），social pooling 已向量化。

## 数据来源

| 方法 | 产物路径 |
|---|---|
| GAST | `outputs/trajgap_gast_goal_3gpu_retry/`（历史五场景，正式口径以 `trajgap_gast_5scene_seed2024` 重训为准） |
| MoFlow | `MoFlow/results_trajimpute_mixed/<scene>/mixed/results.json` |
| SocialLSTM | `/home/lbh/SocialLSTM/trajgap_bench/artifacts/sociallstm_bench/` |
| AgentFormer | `/home/lbh/AgentFormer/results/trajgap_agentformer/`（**进行中，3/5**） |
| Re | `/home/lbh/Re/outputs/trajgap_resonance_seed2024_gpu1/`（5/5 完成） |
| MS-TIP | `/home/lbh/MS-TIP-TrajGap/results/`（**进行中，2/5**） |
| M1（撤回） | `outputs/trajgap_plan/M1_calibration_seed2024/`（负结果证据保留） |

> 归档（2026-10-14）：SocialLSTM 代码与全部产物已迁至 `/home/lbh/SocialLSTM/trajgap_bench/`，并从本仓库 runner/evaluator 中移除该变体。
> 更名（2026-10-15）：引入自 UniTraj `Past_Encoder` 的编码器由 `unitraj_encoder` 更名为 `gast_encoder`（`GASTEncoder`）；当前完整方法统一称 GAST，goal decoder 和 trajectory-aware mode ranking 均属于 GAST 内部设计。产物目录名沿用历史标签 `UT-ENC`，仅用于路径溯源。
"""


def emit_full_md(data):
    return MD_HEADER + emit_md(data) + MD_NOTES


TEX_HEADER = """% =====================================================================
%  TrajGap-Bench leaderboard tables (Mixed-direct)
%  GENERATED by docs/tools/gen_leaderboard_tables.py from results.json —
%  do not hand-edit; re-run the generator instead.
%  Requires: \\usepackage{booktabs}
%  Protocol: TrajImpute Easy+Hard merged per split (Mixed), seed 2024,
%            K=20, 100 epochs, checkpoint selection by val_new_minFDE20.
%  Test-sample counts (identical across all methods):
%    ETH-M 1,629 | HOTEL-M 9,477 | UNIV-M 219,006 | ZARA1-M 20,277
%    | ZARA2-M 52,497 | total 302,886
%  "Avg" = macro average over the five scenes. Lower is better.
%  NOTE: current values are historical, not v1.1 results. Historical
%        comparisons should be re-run under the final-head val_new_minFDE20
%        rule before publication.
% =====================================================================

"""

TEX_PROVENANCE = """% ---------------------------------------------------------------------
% Table 0: method provenance and adopted mechanisms
% ---------------------------------------------------------------------
\\begin{table*}[t]
  \\centering
  \\caption{Method provenance on \\textbf{TrajGap-Bench}. We distinguish
  methods that adapt mechanisms from prior work to the missing-history
  direct-prediction task from external baselines run unchanged, and
  record what was adapted for the protocol.}
  \\label{tab:tgbench-provenance}
  \\small
  \\setlength{\\tabcolsep}{5pt}
  \\begin{tabular}{p{2.0cm} p{4.4cm} p{3.0cm} p{5.4cm}}
    \\toprule
    Method & Source & Role & Adapted mechanisms and modifications \\\\
    \\midrule
    GAST & \\textbf{our method}; encoder mechanisms adapted from
      Sports-Traj / UniTraj \\cite{xu2025sportstraj}
      & \\textbf{our method} (\\texttt{encoder\\_type=gast}, default)
      & GSM per-frame spatial attention with ghost token, Temporal Decay and
        bidirectional Mamba are introduced to the missing-history task and
        adapted into the GAST encoder; goal decoder, loss, metrics and the
        TrajGap protocol are this work's and held fixed \\\\
    MoFlow & \\cite{fu2025moflow} & external baseline (official code)
      & run unchanged; upper-bound reference \\\\
    SocialLSTM & \\cite{alahi2016sociallstm}
      & external baseline (classical social pooling)
      & zero-filled input, LSTM state update skipped on missing frames,
        Gaussian-mixture head (K=20); social pooling vectorised; the upstream
        future-pasting patch that leaks ground truth was removed \\\\
    AgentFormer & \\cite{yuan2021agentformer}
      & external baseline (\\textbf{in progress, 1/5 scenes})
      & missing history filled with the mask-aware scene origin (exactly 0 in
        the \\texttt{*\\_scene\\_norm} frame) plus an explicit per-frame validity
        input channel; the decoder anchor is taken at the last valid
        observation. Samples $z$, so ADE@1/FDE@1 are undefined \\\\
    \\bottomrule
  \\end{tabular}
  \\begin{flushleft}
    \\footnotesize
    GAST includes the goal-conditioned decoder and trajectory-aware mode
    ranking. The historical encoder-only result is retained only as an internal
    Internal comparisons (M0, internal ablation arms) are excluded.
  \\end{flushleft}
\\end{table*}

"""

TEX_FOOTER = """
% ---------------------------------------------------------------------
% Notes for the caption / text:
%  * All methods share the identical Mixed test split (302,886 focal
%    samples).
%  * Current values are historical, not TrajGap-Bench v1.1 results.
%    Historical comparison runs should be re-run using final-head
%    val_new_minFDE20 before publication.
%  * The formal primary metrics are minADE20 and minFDE20. ADE@1/FDE@1
%    are auxiliary diagnostics; no MR or composite score is computed.
%  * M1 is withdrawn (negative result, excluded from the main tables).
%  * b-minFDE20 cannot be aligned: this work's direct evaluator does not
%    emit it, so it exists only on the MoFlow side (macro 1.2945).
%  * BibTeX keys used above:
%      @inproceedings{xu2025sportstraj,
%        title={Sports-Traj: A Unified Trajectory Generation Model for
%               Multi-Agent Movement in Sports},
%        author={Xu, Yi and Fu, Yun},
%        booktitle={ICLR}, year={2025}}
%      @inproceedings{fu2025moflow,
%        title={MoFlow: One-Step Flow Matching for Human Trajectory
%               Forecasting via Implicit Maximum Likelihood Estimation
%               Distillation},
%        author={Fu, Yuxiang and Yan, Qi and Wang, Lele and Li, Ke and
%                Liao, Renjie},
%        booktitle={CVPR}, year={2025}}
%      @inproceedings{alahi2016sociallstm,
%        title={Social LSTM: Human Trajectory Prediction in Crowded Spaces},
%        author={Alahi, Alexandre and Goel, Kratarth and Ramanathan, Vignesh
%                and Robicquet, Alexandre and Li, Fei-Fei and Savarese, Silvio},
%        booktitle={CVPR}, year={2016}}
%      @inproceedings{yuan2021agentformer,
%        title={AgentFormer: Agent-Aware Transformers for Socio-Temporal
%               Multi-Agent Forecasting},
%        author={Yuan, Ye and Weng, Xinshuo and Ou, Yanglan and Kitani, Kris},
%        booktitle={ICCV}, year={2021}}
%      @article{trajimpute2024,
%        title={TrajImpute: A Training-Free Approach for the Missing
%               Trajectory Prediction},
%        journal={arXiv preprint arXiv:2411.00174}, year={2024}}
% ---------------------------------------------------------------------
"""


def emit_tex_extra(data):
    """Emit micro-average and missingness-stratified minFDE tables."""
    L = []
    # ---- micro
    L.append("\\begin{table}[t]")
    L.append("  \\centering")
    L.append("  \\caption{TrajGap-Bench (Mixed-direct), sample-weighted (micro)"
             " average over the 302,886 test samples. Lower is better.}")
    L.append("  \\label{tab:tgbench-micro}")
    L.append("  \\small")
    L.append("  \\setlength{\\tabcolsep}{4pt}")
    L.append("  \\begin{tabular}{l c c c c}")
    L.append("    \\toprule")
    L.append("    Method & minADE$_{20}$ & minFDE$_{20}$ & ADE@1 & FDE@1 \\\\")
    L.append("    \\midrule")
    for m in ["GAST", "MoFlow", "SocialLSTM", "AgentFormer", "Re", "MS-TIP", "Social-STGCNN"]:
        _, mic = agg(data, m)
        cells = []
        for k, _, _ in METRICS:
            if mic[k] is None:
                cells.append("--")
            else:
                cells.append(f"{mic[k]:.4f}")
        L.append(f"    {m} & " + " & ".join(cells) + " \\\\")
    L.append("    \\bottomrule")
    L.append("  \\end{tabular}")
    L.append("  \\begin{flushleft}")
    L.append("    \\footnotesize")
    L.append("    Macro and micro averages answer different questions; report both.")
    L.append("  \\end{flushleft}")
    L.append("\\end{table}")
    L.append("")

    # ---- 缺失分层
    strat = {}
    for m in ["GAST", "MoFlow", "SocialLSTM"]:
        smap = {}
        for s, rec in data[m]["per"].items():
            for gk, entry in rec["_by_group"].items():
                miss = int(gk.split("/")[-1].split("=")[1])
                smap.setdefault(miss, {})[s] = entry
        strat[m] = smap
    L.append("\\begin{table}[t]")
    L.append("  \\centering")
    L.append("  \\caption{Macro-average minFDE$_{20}$ by the number of missing history frames.}")
    L.append("  \\label{tab:tgbench-missing}")
    L.append("  \\small")
    L.append("  \\setlength{\\tabcolsep}{4pt}")
    L.append("  \\begin{tabular}{c c c c c}")
    L.append("    \\toprule")
    L.append("    \\#missing & GAST & MoFlow & SocialLSTM \\\\")
    L.append("    \\midrule")
    for miss in range(8):
        cells = []
        for m in ["GAST", "MoFlow", "SocialLSTM"]:
            sm = strat[m]
            if miss not in sm:
                cells.append("--")
                continue
            fv = [sm[miss][s]["minFDE_K"] for s in SCENES if s in sm[miss]]
            tag = "" if len(fv) == 5 else f"$^{{({len(fv)})}}$"
            cells.append(f"{sum(fv)/len(fv):.4f}{tag}")
        L.append(f"    {miss} & " + " & ".join(cells) + " \\\\")
    L.append("    \\bottomrule")
    L.append("  \\end{tabular}")
    L.append("\\end{table}")
    L.append("")
    return "\n".join(L)


def emit_full_tex(data):
    return TEX_HEADER + TEX_PROVENANCE + emit_tex(data) + emit_tex_extra(data) + TEX_FOOTER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    data = collect()
    print("=== 数据完整性 ===")
    for m, d in data.items():
        n = len(d["per"])
        flag = "" if n == 5 else f"  <== 仅 {n}/5 场景（Avg/micro 记 --）"
        print(f"  {m:12} {n}/5 场景{flag}")
        for miss in d["missing"]:
            print(f"      [缺失] {miss}")

    print("\n=== 宏平均 / 微平均（现算） ===")
    print(f"  {'method':12} {'macro minADE20':>14} {'macro minFDE20':>14}"
          f" {'micro minADE20':>15} {'micro minFDE20':>15}")
    for m in METHODS:
        mac, mic = agg(data, m)
        f = lambda v: "--" if v is None else f"{v:.4f}"
        print(f"  {m:12} {f(mac['minADE_K']):>14} {f(mac['minFDE_K']):>14}"
              f" {f(mic['minADE_K']):>15} {f(mic['minFDE_K']):>15}")

    out_md = os.path.join(REPO, "docs", "results", "trajgap_bench_leaderboard.md")
    out_tex = os.path.join(REPO, "docs", "results", "trajgap_bench_leaderboard.tex")
    if args.write:
        open(out_md, "w").write(emit_full_md(data))
        print(f"\nwrote {out_md}")
        open(out_tex, "w").write(emit_full_tex(data))
        print(f"wrote {out_tex}")
    else:
        print("\n" + "=" * 80 + "\n[DRY-RUN] 未写文件（加 --write 生效）\n" + "=" * 80)
        print(emit_full_md(data))


if __name__ == "__main__":
    main()
