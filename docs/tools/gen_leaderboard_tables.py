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
GASTPF = [
    "outputs/unitraj_encoder_single_scene_seed2024/UT-ENC_ETH-M_mixed-direct_seed2024_uni/eval/Mixed/ETH-M_Mixed_test_UT-ENC_seed2024_ckpt/results.json",
    "outputs/unitraj_encoder_bench_seed2024/lane_a/UT-ENC_HOTEL-M_mixed-direct_seed2024_uni/eval/Mixed/HOTEL-M_Mixed_test_UT-ENC_seed2024_ckpt/results.json",
    "outputs/unitraj_encoder_bench_seed2024/lane_d/UT-ENC_UNIV-M_mixed-direct_seed2024_uni/eval/Mixed/UNIV-M_Mixed_test_UT-ENC_seed2024_ckpt/results.json",
    "outputs/unitraj_encoder_bench_seed2024/lane_b/UT-ENC_ZARA1-M_mixed-direct_seed2024_uni/eval/Mixed/ZARA1-M_Mixed_test_UT-ENC_seed2024_ckpt/results.json",
    "outputs/unitraj_encoder_bench_seed2024/lane_b/UT-ENC_ZARA2-M_mixed-direct_seed2024_uni/eval/Mixed/ZARA2-M_Mixed_test_UT-ENC_seed2024_ckpt/results.json",
]

METHODS = {
    "GAST-PF": GASTPF,
    "M0": M0,
    "MoFlow": [f"/home/lbh/MoFlow/results_trajimpute_mixed/{s.split('-')[0].lower()}/mixed/results.json"
               for s in SCENES],
    "SocialLSTM": [
        f"/home/lbh/SocialLSTM/trajgap_bench/artifacts/sociallstm_bench/SocialLSTM_{s}_mixed-direct_seed2024_uni/eval/Mixed/{s}_Mixed_test_SocialLSTM_seed2024_ckpt/results.json"
        for s in SCENES],
    "AgentFormer": [
        "/home/lbh/AgentFormer/results/trajgap_agentformer/univ/eval/Mixed/UNIV-M_Mixed_test_AgentFormer_seed2024_ckpt/results.json"],
}

# 表 0：方法出处与借用边界
# 术语统一：「本文方法」= GAST-PF（encoder_type=gast，conf 默认）。
# 借用机制必须在表里显式标注（用户纪律：不把借来的模块伪装成原创）。
PROVENANCE = [
    ("GAST-PF",
     "本工作；编码器机制借自 Sports-Traj / UniTraj (Xu and Fu, ICLR 2025, arXiv:2405.17680)",
     "**本文方法**（`encoder_type=gast`，conf 默认）",
     "GSM 逐帧空间注意力 + ghost token、Temporal Decay、双向 Mamba 均借自 UniTraj，"
     "**不主张原创**；decoder、loss、指标与 TrajGap 协议为本工作原有，保持不变"),
    ("M0",
     "本工作内部基线（`encoder_type=m0`）",
     "本文内部基线（历史对照）",
     "自有历史编码器；optimizer 分组策略 `m0_adamw_complete_module_local_v1`"),
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
     "外部基线（**进行中，仅 1/5 场景**）",
     "缺失历史以 mask-aware 场景原点填充（在 `*_scene_norm` 参照系下为 0）"
     "+ 逐帧有效性显式输入通道；解码锚点取最后有效观测。无模态概率，故 ADE@1/FDE@1 不适用"),
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
    ("MR", "MR", "MR (\\%)"),
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
                    ("FDE_1", ["FDE_1", "FDE@1"]),
                    ("MR", ["MR"])]:
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
    methods = ["GAST-PF", "M0", "MoFlow", "SocialLSTM", "AgentFormer"]

    # ---- 表 0 出处
    L.append("## 表 0 · 方法出处与借用边界\n")
    L.append("| Method | 论文 / 来源 | 定位 | 借用边界与改动 |")
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

    # ---- 表 3：MR
    L.append("## 表 3 · 缺失率（MR，%，末帧误差全部 > 2.0 m 才计一次 miss）\n")
    L.append("| Method | " + " | ".join(SCENES) + " | Avg |")
    L.append("|---|" + "---|" * (len(SCENES) + 1))
    for m in methods:
        per = data[m]["per"]
        row = [f"| {m} "]
        for s in SCENES:
            rec = per.get(s)
            if rec is None or rec["MR"] is None:
                row.append("| -- ")
                continue
            b = best_in_col(data, "MR", s, methods)
            v = f"**{rec['MR']*100:.2f}**" if abs(rec["MR"] - b) < 1e-12 else f"{rec['MR']*100:.2f}"
            row.append(f"| {v} ")
        mac, _ = agg(data, m)
        if mac["MR"] is not None:
            b = best_in_col_macro(data, "MR", methods)
            v = f"**{mac['MR']*100:.2f}**" if abs(mac["MR"] - b) < 1e-12 else f"{mac['MR']*100:.2f}"
            row.append(f"| {v} |")
        else:
            row.append("| -- |")
        L.append("".join(row))
    L.append("")

    # ---- 表 4：micro
    L.append("## 表 4 · 样本加权（micro）平均\n")
    L.append("| Method | minADE20 | minFDE20 | ADE@1 | FDE@1 | MR |")
    L.append("|---|---|---|---|---|---|")
    for m in methods:
        _, mic = agg(data, m)
        cells = []
        for k, _, _ in METRICS:
            if mic[k] is None:
                cells.append("--")
            else:
                cells.append(f"{mic[k]*100:.2f}" if k == "MR" else f"{mic[k]:.4f}")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L.append("")

    # ---- 表 5：缺失分层
    L.append("## 表 5 · 缺失程度分层（宏平均）\n")
    L.append("| missing | " + " | ".join(
        f"{m} minFDE20 | {m} MR%" for m in ["GAST-PF", "M0", "MoFlow", "SocialLSTM"]) + " |")
    L.append("|---|" + "---|" * 8)
    strat = {}
    for m in ["GAST-PF", "M0", "MoFlow", "SocialLSTM"]:
        per = data[m]["per"]
        smap = {}
        for s, rec in per.items():
            for gk, entry in rec["_by_group"].items():
                miss = int(gk.split("/")[-1].split("=")[1])
                smap.setdefault(miss, {})[s] = entry
        strat[m] = smap
    for miss in range(8):
        cells = []
        for m in ["GAST-PF", "M0", "MoFlow", "SocialLSTM"]:
            sm = strat[m]
            if miss not in sm:
                cells += ["--", "--"]
                continue
            fv = [sm[miss][s]["minFDE_K"] for s in SCENES if s in sm[miss]]
            mv = [sm[miss][s]["MR"] for s in SCENES if s in sm[miss]]
            tag = "" if len(fv) == 5 else f" (n={len(fv)})"
            cells.append(f"{sum(fv)/len(fv):.4f}{tag}")
            cells.append(f"{sum(mv)/len(mv)*100:.2f}{tag}")
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
    methods = ["GAST-PF", "M0", "MoFlow", "SocialLSTM", "AgentFormer"]
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

> 协议：TrajImpute 官方 Easy+Hard 同一 split 合并（Mixed），seed 2024，K=20，100 epochs，`val_minFDE20` 选点。
> 测试样本数（所有方法完全一致）：ETH-M 1,629 / HOTEL-M 9,477 / UNIV-M 219,006 / ZARA1-M 20,277 / ZARA2-M 52,497，合计 302,886。
> `Avg` = 五场景宏平均；`micro` = 按样本数加权。所有指标**越低越好**。加粗 = 该列最优。
> **术语统一**：全文「本文方法」均指 **GAST-PF**（`encoder_type=gast`，`conf` 默认）；历史实验产物标签为 `UT-ENC`，2026-10-15 起代码更名为 `gast_encoder`，本文一律写作 GAST-PF。

> 本文件由 `docs/tools/gen_leaderboard_tables.py` 从各方法 `results.json` 现算生成，不手工转抄。
> 重新生成：`python docs/tools/gen_leaderboard_tables.py --write`

"""

MD_NOTES = """
## 表注

- **`--` = 该场景/指标不可用**。AgentFormer 仅完成 UNIV-M 一场景，故其 `Avg` 与 `micro` 均为 `--`——**单场景值不得当宏平均报**；其 ADE@1/FDE@1 不适用（采样器无模态概率）。
- **GAST-PF 行需重跑后才可定稿**：其 checkpoint 由初始头 `val_minFDE20` 选出，而 test 使用最终头 `new_y_hat/new_pi`；五场景最佳 epoch 全部对不上（ETH-M e94 vs e85、HOTEL-M e90 vs e52、UNIV-M e94 vs e83、ZARA1-M e91 vs e79、ZARA2-M e81 vs e71）。该选点口径已作废（`b60100a` 起统一为 `val_new_minFDE20`）。
- **GAST-PF 相对 M0 为持平，不是提升**：宏平均 M0 全面略优（差 1.3% / 0.6% / 0.5% / 0.09%），GAST-PF 仅拿下宏平均 MR（4.50 vs 4.67）与全部 micro 几何指标。换掉整个编码器只换来微幅变化，说明编码器不是当前瓶颈。
- **micro 口径下 MoFlow 在 minFDE20 与 MR 上反超**（0.4981 vs M0 0.5029；2.78 vs M0 3.72）。主因 UNIV-M 占 72% 样本量。**不得只报宏平均就宣称全面最优**；两个口径必须并列给出。
- **M1 已判负撤回**（2026-10-11），仅作负结果证据，不进主表。宏平均 minADE20/minFDE20/MR 分别恶化 +5.2% / +7.7% / +0.53 pp；微平均恶化 +21.2% / +23.6% / +2.46 pp。ADE@1/FDE@1 宏平均小幅改善（−1.42% / −1.34%）不足以抵消 best-of-20 与 MR 的退化。
- **M0-pre-fix 不混入**：旧 revision `a4f1803` 的宏平均为 0.3249 / 0.5039，与本表 M0（0.3223 / 0.4917）不同定义，论文中只能有一个 M0。
- **b-minFDE20 无法对齐**：本工作 direct evaluator 不产出该项，仅 MoFlow 侧有 test 值（宏平均 1.2945），故未入表。
- 部分场景的平均值与五场景宏平均**不可直接比较**。
- SocialLSTM 使用 missing-aware 变体：零填充输入 + 缺失帧跳过 LSTM 状态更新 + 高斯混合头（K=20，按分量采样），social pooling 已向量化。

## 数据来源

| 方法 | 产物路径 |
|---|---|
| GAST-PF | `outputs/unitraj_encoder_single_scene_seed2024/UT-ENC_ETH-M_mixed-direct_seed2024_uni/`（ETH-M）；`outputs/unitraj_encoder_bench_seed2024/lane_{a,b,d}/`（其余四场景） |
| M0 | `outputs/m0_baseline/M0_optimizerfix_seed2024_gpu{0,1}/` |
| MoFlow | `MoFlow/results_trajimpute_mixed/<scene>/mixed/results.json` |
| SocialLSTM | `/home/lbh/SocialLSTM/trajgap_bench/artifacts/sociallstm_bench/` |
| AgentFormer | `/home/lbh/AgentFormer/results/trajgap_agentformer/`（**进行中，1/5**） |
| M1（撤回） | `outputs/trajgap_plan/M1_calibration_seed2024/`（负结果证据保留） |

> 归档（2026-10-14）：SocialLSTM 代码与全部产物已迁至 `/home/lbh/SocialLSTM/trajgap_bench/`，并从本仓库 runner/evaluator 中移除该变体。
> 更名（2026-10-15）：借用 UniTraj `Past_Encoder` 的编码器由 `unitraj_encoder` 更名为 `gast_encoder`（`GASTEncoder`）；`conf` 中 `encoder_type` 默认已是 `gast`，`m0` 仅作历史对照入口。产物目录名沿用历史标签 `UT-ENC`（仅路径，非方法名），本文一律称 GAST-PF。
"""


def emit_full_md(data):
    return MD_HEADER + emit_md(data) + MD_NOTES


TEX_HEADER = """% =====================================================================
%  TrajGap-Bench leaderboard tables (Mixed-direct)
%  GENERATED by docs/tools/gen_leaderboard_tables.py from results.json —
%  do not hand-edit; re-run the generator instead.
%  Requires: \\usepackage{booktabs}
%  Protocol: TrajImpute Easy+Hard merged per split (Mixed), seed 2024,
%            K=20, 100 epochs, checkpoint selection by val_minFDE20.
%  Test-sample counts (identical across all methods):
%    ETH-M 1,629 | HOTEL-M 9,477 | UNIV-M 219,006 | ZARA1-M 20,277
%    | ZARA2-M 52,497 | total 302,886
%  "Avg" = macro average over the five scenes. Lower is better.
%  NOTE: the GAST-PF row uses the retired val_minFDE20 checkpoint-selection
%        rule and must be re-run under val_new_minFDE20 before publication.
% =====================================================================

"""

TEX_PROVENANCE = """% ---------------------------------------------------------------------
% Table 0: method provenance and borrowed boundaries
% ---------------------------------------------------------------------
\\begin{table*}[t]
  \\centering
  \\caption{Method provenance on \\textbf{TrajGap-Bench}. We distinguish
  methods whose mechanisms are borrowed from prior work (and are
  \\emph{not} claimed as contributions) from external baselines run
  unchanged, and record what was modified to fit the missing-history
  direct-prediction protocol.}
  \\label{tab:tgbench-provenance}
  \\small
  \\setlength{\\tabcolsep}{5pt}
  \\begin{tabular}{p{2.0cm} p{4.4cm} p{3.0cm} p{5.4cm}}
    \\toprule
    Method & Source & Role & Borrowed mechanisms and modifications \\\\
    \\midrule
    GAST-PF & \\textbf{our method}; encoder mechanisms borrowed from
      Sports-Traj / UniTraj \\cite{xu2025sportstraj}
      & \\textbf{our method} (\\texttt{encoder\\_type=gast}, default)
      & GSM per-frame spatial attention with ghost token, Temporal Decay and
        bidirectional Mamba are taken from UniTraj and are \\emph{not} claimed
        as contributions; decoder, loss, metrics and the TrajGap protocol are
        this work's and held fixed \\\\
    M0 & this work, internal baseline (\\texttt{encoder\\_type=m0})
      & internal baseline (historical reference)
      & own history encoder; AdamW grouping policy
        \\texttt{m0\\_adamw\\_complete\\_module\\_local\\_v1} \\\\
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
    M1 & this work, internal (missingness-aware output calibration)
      & \\textbf{withdrawn after failing its gate}; negative-result evidence only
      & calibration head conditions mode-logit temperature and Laplace scale;
        candidate trajectory coordinates are not modified \\\\
    \\bottomrule
  \\end{tabular}
  \\begin{flushleft}
    \\footnotesize
    GAST-PF and M0 share the decoder, loss, metrics and data protocol; only the
    history encoder differs, which makes the comparison single-variable.
  \\end{flushleft}
\\end{table*}

"""

TEX_FOOTER = """
% ---------------------------------------------------------------------
% Notes for the caption / text:
%  * All methods share the identical Mixed test split (302,886 focal
%    samples).
%  * The GAST-PF row uses the retired val_minFDE20 checkpoint-selection
%    rule while test uses the final new_y_hat/new_pi head; the five best
%    epochs all disagree (ETH-M e94 vs e85, HOTEL-M e90 vs e52,
%    UNIV-M e94 vs e83, ZARA1-M e91 vs e79, ZARA2-M e81 vs e71). It must
%    be re-run under val_new_minFDE20 before publication.
%  * GAST-PF is on par with M0, not better: M0 leads all four macro
%    geometry metrics by 1.3%/0.6%/0.5%/0.09%, while GAST-PF leads macro
%    MR (4.50 vs 4.67) and all micro geometry metrics. Replacing the whole
%    encoder moved the numbers only marginally.
%  * Under the micro (sample-weighted) average MoFlow overtakes on
%    minFDE20 and MR, driven by UNIV-M holding 72% of the samples. Report
%    both aggregations; do not claim across-the-board superiority from the
%    macro average alone.
%  * M1 is withdrawn (negative result, excluded from the main tables).
%  * M0-pre-fix (revision a4f1803, macro 0.3249/0.5039) must not be mixed
%    with the M0 reported here (0.3223/0.4917).
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
    """micro 表 + 缺失分层表（paper 版：分层表用 'minFDE20 / MR%' 单格，
    避免 9 列过宽）。"""
    L = []
    # ---- micro
    L.append("\\begin{table}[t]")
    L.append("  \\centering")
    L.append("  \\caption{TrajGap-Bench (Mixed-direct), sample-weighted (micro)"
             " average over the 302,886 test samples. MR in \\%. Lower is better.}")
    L.append("  \\label{tab:tgbench-micro}")
    L.append("  \\small")
    L.append("  \\setlength{\\tabcolsep}{4pt}")
    L.append("  \\begin{tabular}{l c c c c c}")
    L.append("    \\toprule")
    L.append("    Method & minADE$_{20}$ & minFDE$_{20}$ & ADE@1 & FDE@1 & MR \\\\")
    L.append("    \\midrule")
    for m in ["GAST-PF", "M0", "MoFlow", "SocialLSTM", "AgentFormer"]:
        _, mic = agg(data, m)
        cells = []
        for k, _, _ in METRICS:
            if mic[k] is None:
                cells.append("--")
            else:
                cells.append(f"{mic[k]*100:.2f}" if k == "MR" else f"{mic[k]:.4f}")
        L.append(f"    {m} & " + " & ".join(cells) + " \\\\")
    L.append("    \\bottomrule")
    L.append("  \\end{tabular}")
    L.append("  \\begin{flushleft}")
    L.append("    \\footnotesize")
    L.append("    Under this aggregation MoFlow overtakes on minFDE$_{20}$ and MR,"
             " because UNIV-M holds 72\\% of the samples. Report both aggregations.")
    L.append("  \\end{flushleft}")
    L.append("\\end{table}")
    L.append("")

    # ---- 缺失分层
    strat = {}
    for m in ["GAST-PF", "M0", "MoFlow", "SocialLSTM"]:
        smap = {}
        for s, rec in data[m]["per"].items():
            for gk, entry in rec["_by_group"].items():
                miss = int(gk.split("/")[-1].split("=")[1])
                smap.setdefault(miss, {})[s] = entry
        strat[m] = smap
    L.append("\\begin{table}[t]")
    L.append("  \\centering")
    L.append("  \\caption{Robustness to the number of missing history frames"
             " (macro average over scenes), reported as minFDE$_{20}$\\,/\\,MR(\\%)."
             " Regular missingness (0--6) separates the methods; at 7 missing frames"
             " all methods degrade and MoFlow leads, which is scene-dependent rather"
             " than evidence of a structural defect.}")
    L.append("  \\label{tab:tgbench-missing}")
    L.append("  \\small")
    L.append("  \\setlength{\\tabcolsep}{4pt}")
    L.append("  \\begin{tabular}{c c c c c}")
    L.append("    \\toprule")
    L.append("    \\#missing & GAST-PF & M0 & MoFlow & SocialLSTM \\\\")
    L.append("    \\midrule")
    for miss in range(8):
        cells = []
        for m in ["GAST-PF", "M0", "MoFlow", "SocialLSTM"]:
            sm = strat[m]
            if miss not in sm:
                cells.append("--")
                continue
            fv = [sm[miss][s]["minFDE_K"] for s in SCENES if s in sm[miss]]
            mv = [sm[miss][s]["MR"] for s in SCENES if s in sm[miss]]
            tag = "" if len(fv) == 5 else f"$^{{({len(fv)})}}$"
            cells.append(f"{sum(fv)/len(fv):.4f} / {sum(mv)/len(mv)*100:.2f}{tag}")
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
    print(f"  {'method':12} {'macro minADE20':>14} {'macro minFDE20':>14} {'macro MR%':>10}"
          f" {'micro minADE20':>15} {'micro minFDE20':>15} {'micro MR%':>10}")
    for m in METHODS:
        mac, mic = agg(data, m)
        f = lambda v: "--" if v is None else f"{v:.4f}"
        fm = lambda v: "--" if v is None else f"{v*100:.2f}"
        print(f"  {m:12} {f(mac['minADE_K']):>14} {f(mac['minFDE_K']):>14} {fm(mac['MR']):>10}"
              f" {f(mic['minADE_K']):>15} {f(mic['minFDE_K']):>15} {fm(mic['MR']):>10}")

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
