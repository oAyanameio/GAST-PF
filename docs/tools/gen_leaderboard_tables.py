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
from pathlib import Path

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCENES = ["ETH-M", "HOTEL-M", "UNIV-M", "ZARA1-M", "ZARA2-M"]

# External baseline paths; GAST results are selected from run metadata.
METHODS = {
    "GAST": None,
    "MoFlow": [f"/home/lbh/MoFlow/results_trajimpute_mixed/{s.split('-')[0].lower()}/mixed/results.json"
               for s in SCENES],
    # SocialLSTM 已移除：原实现为单模态输出，K=20 头为仓库侧改造，
    # 与其余多模态方法口径不可比（2026-10 用户裁定）。
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

METRICS = [
    ("minADE_K", "minADE_K", "minADE$_{20}$"),
    ("minFDE_K", "minFDE_K", "minFDE$_{20}$"),
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
                    ("minFDE_K", ["minFDE_K", "minFDE20"])]:
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


def discover_gast_results():
    """Select the unique confirmatory five-scene GAST result set."""
    output_root = Path(REPO) / "outputs"
    groups = {}
    for result_path in output_root.rglob("results.json"):
        try:
            payload = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        meta = payload.get("meta", {})
        scene = meta.get("scene")
        if not (
            scene in SCENES
            and meta.get("split") == "test"
            and meta.get("difficulty") == "Mixed"
            and meta.get("protocol") == "mixed-direct"
            and meta.get("K") == 20
            and meta.get("seed") == 2024
            and not meta.get("untrained_smoke", False)
        ):
            continue

        manifest_path = next(
            (
                parent / "manifest.json"
                for parent in result_path.parents
                if (parent / "manifest.json").is_file()
            ),
            None,
        )
        if manifest_path is None:
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        model_config = manifest.get("model_config", {})
        is_gast = (
            manifest.get("model_version") == "GAST"
            or model_config.get("encoder_type") == "gast"
        )
        revision = meta.get("git_revision")
        if (
            manifest.get("experiment_class") != "confirmatory"
            or not is_gast
            or not revision
            or manifest.get("git_revision") != revision
        ):
            continue
        groups.setdefault(revision, {}).setdefault(scene, []).append(
            str(result_path.relative_to(REPO))
        )

    complete = {
        revision: per_scene
        for revision, per_scene in groups.items()
        if all(len(per_scene.get(scene, [])) == 1 for scene in SCENES)
    }
    if len(complete) != 1:
        raise RuntimeError(
            "expected one confirmatory GAST result set with one test result "
            f"per scene and a shared revision; found {len(complete)}"
        )

    per_scene = next(iter(complete.values()))
    return [per_scene[scene][0] for scene in SCENES]


def collect():
    data = {}
    for method, paths in METHODS.items():
        if method == "GAST":
            paths = discover_gast_results()
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
    methods = ["GAST", "MoFlow", "AgentFormer", "Re", "MS-TIP", "Social-STGCNN"]

    # ---- 表 1：逐场景 minADE20/minFDE20
    specs = [("## 表 1 · 精度（minADE20 / minFDE20，米）", "minADE_K", "minFDE_K")]
    for title, ka, kb in specs:
        L.append(f"{title}\n")
        L.append("| Method | " + " | ".join(SCENES) + " | Avg |")
        L.append("|---|" + "---|" * (len(SCENES) + 1))
        comparison_methods = methods
        for m in methods:
            per = data[m]["per"]
            row = [f"| {m} "]
            for s in SCENES:
                rec = per.get(s)
                if rec is None or rec[ka] is None:
                    row.append("| -- ")
                    continue
                ba = best_in_col(data, ka, s, comparison_methods)
                bb = best_in_col(data, kb, s, comparison_methods)
                a = f"**{rec[ka]:.4f}**" if ba is not None and abs(rec[ka] - ba) < 1e-12 else f"{rec[ka]:.4f}"
                b = f"**{rec[kb]:.4f}**" if bb is not None and abs(rec[kb] - bb) < 1e-12 else f"{rec[kb]:.4f}"
                row.append(f"| {a} / {b} ")
            # Avg（宏平均）
            mac, _ = agg(data, m)
            if mac[ka] is not None and mac[kb] is not None:
                ba = best_in_col_macro(data, ka, comparison_methods)
                bb = best_in_col_macro(data, kb, comparison_methods)
                a = f"**{mac[ka]:.4f}**" if abs(mac[ka] - ba) < 1e-12 else f"{mac[ka]:.4f}"
                b = f"**{mac[kb]:.4f}**" if abs(mac[kb] - bb) < 1e-12 else f"{mac[kb]:.4f}"
                row.append(f"| {a} / {b} |")
            else:
                row.append("| -- |")
            L.append("".join(row))
        L.append("")

    # ---- 表 2：汇总全部样本的平均
    L.append("## 表 2 · 汇总全部样本的平均\n")
    L.append("| Method | minADE20 | minFDE20 |")
    L.append("|---|---|---|")
    for m in methods:
        _, micro = agg(data, m)
        cells = ["--" if micro[key] is None else f"{micro[key]:.4f}"
                 for key, _, _ in METRICS]
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L.append("")

    # ---- 表 3：缺失分层
    L.append("## 表 3 · 缺失程度分层（宏平均 minFDE20）\n")
    strat_methods = methods
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
            fv = [
                sm[miss][s].get("minFDE20", sm[miss][s].get("minFDE_K"))
                for s in SCENES if s in sm[miss]
            ]
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
    methods = ["GAST", "MoFlow", "AgentFormer", "Re", "MS-TIP", "Social-STGCNN"]
    for title, ka, kb in [
        ("minADE$_{20}$\\,/\\,minFDE$_{20}$", "minADE_K", "minFDE_K"),
    ]:
        L.append("\\begin{table*}[t]")
        L.append("  \\centering")
        L.append(f"  \\caption{{TrajGap-Bench (Mixed): {title} (meters)."
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
        comparison_methods = methods
        for m in methods:
            per = data[m]["per"]
            cells = []
            for s in SCENES:
                rec = per.get(s)
                if rec is None or rec[ka] is None:
                    cells.append("--")
                    continue
                ba, bb = best_in_col(data, ka, s, comparison_methods), best_in_col(data, kb, s, comparison_methods)
                a = f"\\textbf{{{rec[ka]:.4f}}}" if abs(rec[ka] - ba) < 1e-12 else f"{rec[ka]:.4f}"
                b = f"\\textbf{{{rec[kb]:.4f}}}" if abs(rec[kb] - bb) < 1e-12 else f"{rec[kb]:.4f}"
                cells.append(f"{a} / {b}")
            mac, _ = agg(data, m)
            if mac[ka] is not None:
                ba, bb = best_in_col_macro(data, ka, comparison_methods), best_in_col_macro(data, kb, comparison_methods)
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


MD_HEADER = """# TrajGap-Bench 对比实验表

> 数据集与基准：TrajGap-Bench；TrajImpute 官方 Easy+Hard 同一 split 合并为 Mixed，任务为无插补直接预测。seed 2024，K=20，100 epochs。GAST 按最终预测头 `val_new_minFDE20` 选点；外部方法沿用各自的训练与选点流程。
> 测试样本数（所有方法完全一致）：ETH-M 1,629 / HOTEL-M 9,477 / UNIV-M 219,006 / ZARA1-M 20,277 / ZARA2-M 52,497，合计 302,886。
> `Avg` = 五场景宏平均；`micro` = 汇总全部测试样本后的算术平均。所有指标**越低越好**。加粗 = 该列最优。
> 本文方法统一称为 **GAST**；终点引导解码与轨迹级模式排序属于完整方法设计。

> 本文件由 `docs/tools/gen_leaderboard_tables.py` 从各方法 `results.json` 现算生成，不手工转抄。
> 重新生成：`python docs/tools/gen_leaderboard_tables.py --write`

"""

MD_NOTES = """
## 表注

- 六个方法均完成 ETH-M、HOTEL-M、UNIV-M、ZARA1-M、ZARA2-M 五场景。
- 只报告普通 `minADE20`、`minFDE20`；不使用模式分数加权，也不报告 top-1、MR、Brier 或综合分数。
- 表中指标按各方法 Mixed-test `results.json` 计算；宏平均为五场景等权，micro 为汇总全部测试样本后的算术平均。
- 外部模型保持其方法属性：MS-TIP 为插补感知联合预测方法，其余依各自实现处理缺失历史；榜单比较的是统一测试集上的结果，不代表内部结构均为纯 direct。
- 选点口径：GAST / AgentFormer / Re / MS-TIP / Social-STGCNN 均按各自验证集 minFDE20 选最优 checkpoint；MoFlow 沿用其官方实现原生的验证集 minADE（ADE_MIN）选点，未做改动。
- MS-TIP 五场景最优 checkpoint 均出现在 epoch 5–9（后续验证集 minFDE20 持续变差，为其联合插补目标在 direct 协议下的原生行为，非调参不足）；表中数值取各自验证最优 checkpoint。
- SocialLSTM 已移除：原实现为单模态输出，不参与本多模态榜单。

## 数据来源

| 方法 | 产物路径 |
|---|---|
| GAST | 五场景完整结果（manifest revision 一致） |
| MoFlow | `/home/lbh/MoFlow/results_trajimpute_mixed/` |
| AgentFormer | `/home/lbh/AgentFormer/results/trajgap_agentformer/`（5/5） |
| Re | `/home/lbh/Re/outputs/trajgap_resonance_seed2024_gpu1/`（5/5） |
| MS-TIP | `/home/lbh/MS-TIP-TrajGap/results/`（5/5，插补感知） |
| Social-STGCNN | `/home/lbh/Social-STGCNN/results_trajgap/`（5/5） |
"""


def emit_full_md(data):
    return MD_HEADER + emit_md(data) + MD_NOTES


TEX_HEADER = """% =====================================================================
%  TrajGap-Bench leaderboard tables
%  GENERATED by docs/tools/gen_leaderboard_tables.py from results.json —
%  do not hand-edit; re-run the generator instead.
%  Requires: \\usepackage{booktabs}
%  Protocol: TrajImpute Easy+Hard merged per split (Mixed), seed 2024,
%  GAST checkpoint selection: val_new_minFDE20; external baselines retain
%  each implementation's native training and checkpoint selection procedure.
%  Test-sample counts (identical across all methods):
%    ETH-M 1,629 | HOTEL-M 9,477 | UNIV-M 219,006 | ZARA1-M 20,277
%    | ZARA2-M 52,497 | total 302,886
%  "Avg" = macro average over the five scenes. Lower is better.
%  Current formal comparison values are generated from complete five-scene
%  Mixed-test result artifacts using the fixed GAST selection rule.
% =====================================================================

"""

TEX_FOOTER = """
% ---------------------------------------------------------------------
% Notes for the caption / text:
%  * All methods share the identical Mixed test split (302,886 focal
%    samples).
%  * Current values are generated from complete five-scene result artifacts.
%  * GAST uses val_new_minFDE20 for checkpoint selection; external baselines
%    retain their native selection procedures.
%  * The report contains only ordinary minADE20 and minFDE20; mode scores
%    do not reweight either metric. No top-1, MR, Brier, or composite score.
% ---------------------------------------------------------------------
"""


def emit_tex_micro(data):
    """Emit only ordinary minADE20/minFDE20 pooled over test samples."""
    slash = chr(92)
    L = [
        f"{slash}begin{{table}}[t]",
        f"  {slash}centering",
        f"  {slash}caption{{TrajGap-Bench (Mixed), arithmetic mean over all test samples. Lower is better.}}",
        f"  {slash}label{{tab:tgbench-micro}}",
        f"  {slash}small",
        f"  {slash}setlength{{{slash}tabcolsep}}{{4pt}}",
        f"  {slash}begin{{tabular}}{{l c c}}",
        f"    {slash}toprule",
        "    Method & minADE$_{20}$ & minFDE$_{20}$ " + slash * 2,
        f"    {slash}midrule",
    ]
    methods = ["GAST", "MoFlow", "AgentFormer", "Re", "MS-TIP", "Social-STGCNN"]
    for method in methods:
        _, micro = agg(data, method)
        cells = ["--" if micro[key] is None else f"{micro[key]:.4f}"
                 for key, _, _ in METRICS]
        L.append(f"    {method} & " + " & ".join(cells) + " " + slash * 2)
    L.extend([
        f"    {slash}bottomrule",
        f"  {slash}end{{tabular}}",
        f"{slash}end{{table}}",
        "",
    ])
    return "\n".join(L)


def emit_tex_extra(data):
    """Emit the missingness-stratified minFDE table."""
    L = []
    # ---- 缺失分层
    strat_methods = list(METHODS)
    strat = {}
    for m in strat_methods:
        smap = {}
        for s, rec in data[m]["per"].items():
            for gk, entry in rec["_by_group"].items():
                miss = int(gk.split("/")[-1].split("=")[1])
                smap.setdefault(miss, {})[s] = entry
        strat[m] = smap
    L.append("\\begin{table}[t]")
    L.append("  \\centering")
    L.append("  \\caption{Five-scene macro-average minFDE$_{20}$ by missing history frames.}")
    L.append("  \\label{tab:tgbench-missing}")
    L.append("  \\small")
    L.append("  \\setlength{\\tabcolsep}{2pt}")
    L.append("  \\resizebox{\\textwidth}{!}{%")
    L.append("  \\begin{tabular}{c " + "c " * len(strat_methods) + "}")
    L.append("    \\toprule")
    L.append("    \\#missing & " + " & ".join(strat_methods) + " \\\\")
    L.append("    \\midrule")
    for miss in range(8):
        vals = []
        for m in strat_methods:
            sm = strat[m]
            fv = [
                sm[miss][s].get("minFDE20", sm[miss][s].get("minFDE_K"))
                for s in SCENES if miss in sm and s in sm[miss]
            ]
            vals.append(sum(fv) / len(fv) if fv else None)
        best = min((v for v in vals if v is not None), default=None)
        cells = ["--" if v is None else (f"\\textbf{{{v:.4f}}}" if v == best else f"{v:.4f}") for v in vals]
        L.append(f"    {miss} & " + " & ".join(cells) + " \\\\")
    L.append("    \\bottomrule")
    L.append("  \\end{tabular}%")
    L.append("  }")
    L.append("\\end{table}")
    L.append("")
    return "\n".join(L)


def emit_full_tex(data):
    return TEX_HEADER + emit_tex(data) + emit_tex_micro(data) + emit_tex_extra(data) + TEX_FOOTER


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
