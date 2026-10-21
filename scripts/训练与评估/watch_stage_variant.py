"""通用 Stage watcher：等待五场景跑完后输出指标、对照与分层表。

只读：轮询 results.json，不写入实验目录。
口径：新选点 `val_new_minFDE20`；主对照 MoFlow / SocialLSTM（外部模型）。

用法：
    python watch_stage_variant.py                       # 默认 Stage 1 · AnchorGeo
    python watch_stage_variant.py --variant GAST-PF-FrameGeo \
        --root outputs/stage2_frame_geo_seed2024
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "outputs/stage1_anchor_geo_seed2024"
VARIANT = "GAST-PF-AnchorGeo"
STAGE_LABELS = {
    "GAST-PF-AnchorGeo": "Stage 1 · AnchorGeo",
    "GAST-PF-FrameGeo": "Stage 2 · FrameGeo",
    "GAST-PF-DAP": "Directional Attention Pooling",
    "GAST-PF-Tail": "GAST-PF + M0 scene interaction tail",
}
SCENES = ["ETH-M", "HOTEL-M", "UNIV-M", "ZARA1-M", "ZARA2-M"]
# MR 已于 2026-10-20 废弃（见 docs/results/MR废弃决策.md）。
# 这里按 results.json 实际含有的键自适应：新结果只有 4 项，历史结果仍带 MR。
BASE_METRICS = ["minADE_K", "minFDE_K", "ADE@1", "FDE@1"]
LEGACY_METRICS = ["MR"]
METRICS = BASE_METRICS + LEGACY_METRICS


def metrics_of(rows):
    """按已有结果决定列：全部含 MR 才展示 MR（历史口径），否则只报四指标。"""
    if rows and all("MR" in r for r in rows.values()):
        return BASE_METRICS + LEGACY_METRICS
    return BASE_METRICS
DIMS = ["missing_count", "valid_count", "anchor_lag", "forecast_gap",
        "terminal_missing", "max_missing_run"]
MOFLOW = REPO.parent / "MoFlow/results_trajimpute_mixed"
SOCIAL = REPO.parent / "SocialLSTM/trajgap_bench/artifacts/sociallstm_bench"
MOFLOW_DIR = {"ETH-M": "eth", "HOTEL-M": "hotel", "UNIV-M": "univ",
              "ZARA1-M": "zara1", "ZARA2-M": "zara2"}
POLL_SECONDS = 600
MAX_HOURS = 40


def arm_dir(scene):
    return ROOT / f"{VARIANT}_{scene}_mixed-direct_seed2024_uni"


def load(arm, scene):
    hits = sorted((arm / "eval/Mixed").glob("*/results.json"))
    return json.loads(hits[0].read_text()) if hits else None


def load_arm(scene):
    return load(arm_dir(scene), scene)


def load_moflow(scene):
    p = MOFLOW / MOFLOW_DIR[scene] / "mixed/results.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())["results"]["overall"]


def load_social(scene):
    hits = sorted((SOCIAL / f"SocialLSTM_{scene}_mixed-direct_seed2024_uni"
                   / "eval/Mixed").glob("*/results.json"))
    return json.loads(hits[0].read_text())["results"]["overall"] if hits else None


def macro(rows, keys=None):
    keys = keys or METRICS
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


def report():
    stage_label = STAGE_LABELS.get(VARIANT, VARIANT)
    out = [f"=== {stage_label} 五场景（Mixed-test, K=20, val_new_minFDE20）==="]
    mine, mo, so = {}, {}, {}
    for s in SCENES:
        d = load_arm(s)
        if d:
            mine[s] = d["results"]["overall"]
        m = load_moflow(s)
        if m:
            mo[s] = m
        x = load_social(s)
        if x:
            so[s] = x

    cols = metrics_of(mine)
    has_mr = "MR" in cols
    out.append(f"\n{'场景':<9} {'n':>7} " + " ".join(f"{m:>9s}" for m in cols)
               + "  |  ΔminFDE20 vs MoFlow" + ("  ΔMR" if has_mr else ""))
    for s in SCENES:
        if s not in mine:
            out.append(f"{s:<9} {'待完成':>7}")
            continue
        r = mine[s]
        line = f"{s:<9} {r['n']:>7} " + " ".join(f"{r[k]:>9.4f}" for k in cols)
        if s in mo:
            line += f"  |  {100*(r['minFDE_K']-mo[s]['minFDE_K'])/mo[s]['minFDE_K']:+8.2f}%"
            if has_mr and "MR" in mo[s]:
                line += f"  {100*(r['MR']-mo[s]['MR'])/mo[s]['MR']:+8.2f}%"
        out.append(line)

    if len(mine) == len(SCENES):
        A, B, C = macro([mine[s] for s in SCENES], cols), None, None
        out.append("")
        out.append(f"{'宏平均':<9} {'—':>7} " + " ".join(f"{A[k]:>9.4f}" for k in cols))
        if len(mo) == len(SCENES):
            B = macro([mo[s] for s in SCENES])
            out.append(f"{'MoFlow':<9} {'—':>7} " + " ".join(f"{B[k]:>9.4f}" for k in cols))
            common = [k for k in cols if k in B]
            out.append("  相对 MoFlow: " + "  ".join(
                f"{k}={100*(A[k]-B[k])/B[k]:+.1f}%" for k in common))
        if len(so) == len(SCENES):
            C = macro([so[s] for s in SCENES])
            out.append(f"{'SocialLSTM':<9} {'-':>7} " + " ".join(f"{C[k]:>9.4f}" for k in cols))
            common = [k for k in cols if k in C]
            out.append("  相对 SocialLSTM: " + "  ".join(
                f"{k}={100*(A[k]-C[k])/C[k]:+.1f}%" for k in common))
        out.append("")
        out.append(f"=== 分层（by_dimension，{VARIANT} 样本加权）===")
        for dim in DIMS:
            agg = {}
            for s in SCENES:
                payload = load_arm(s)
                if payload is None:
                    continue
                d = payload["results"]["by_dimension"].get(dim, {})
                for v, e in d.items():
                    # MR 已废弃：只在分层数据确实含 MR 时才累计（历史口径）
                    want_mr = has_mr and "MR" in e
                    a = agg.setdefault(v, {"n": 0, "minFDE_K": 0.0, "MR": 0.0,
                                           "has_mr": want_mr})
                    a["n"] += e["n"]
                    a["minFDE_K"] += e["minFDE_K"] * e["n"]
                    if want_mr:
                        a["MR"] += e["MR"] * e["n"]
                    a["has_mr"] = a["has_mr"] and want_mr
            if not agg:
                out.append(f"[{dim}] 缺失")
                continue
            any_mr = any(e["has_mr"] for e in agg.values())
            out.append(f"\n[{dim}]  value |      n   minFDE20"
                       + ("        MR" if any_mr else "")
                       + "   （MR 已废弃，仅历史口径展示）" if any_mr else "")
            for v in sorted(agg, key=lambda z: (len(z), z)):
                e = agg[v]
                line = f"  {v:>6} | {e['n']:>7} {e['minFDE_K']/e['n']:>11.4f}"
                if any_mr and e["has_mr"]:
                    line += f" {100*e['MR']/e['n']:>7.2f}%"
                out.append(line)
    return "\n".join(out)


def main():
    global ROOT, VARIANT
    ap = argparse.ArgumentParser(description="通用 Stage watcher（只读轮询）")
    ap.add_argument("--variant", default=VARIANT,
                    help=f"模型变体名（默认 {VARIANT}）")
    ap.add_argument("--root", default=str(ROOT.relative_to(REPO)),
                    help="输出根目录（相对仓库根，默认 Stage 1 根）")
    args = ap.parse_args()
    VARIANT = args.variant
    ROOT = Path(args.root)
    if not ROOT.is_absolute():
        ROOT = REPO / ROOT
    print(f"[watch] variant={VARIANT} root={ROOT}", flush=True)

    deadline = time.time() + MAX_HOURS * 3600
    while time.time() < deadline:
        pending = [s for s in SCENES if load_arm(s) is None]
        if not pending:
            print(f"STAGE_DONE variant={VARIANT}")
            print(report())
            return 0
        print(f"[waiting] pending={pending}", flush=True)
        time.sleep(POLL_SECONDS)
    print("TIMEOUT")
    print(report())
    return 1


if __name__ == "__main__":
    sys.exit(main())
