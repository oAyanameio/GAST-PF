"""Stage 1 · AnchorGeo：等待五场景复跑完成后输出指标、对照与分层表。

只读：轮询 results.json，不写入实验目录。
口径：新选点 `val_new_minFDE20`；主对照 MoFlow / SocialLSTM（外部模型）。
"""

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "outputs/stage1_anchor_geo_seed2024"
VARIANT = "GAST-PF-AnchorGeo"
SCENES = ["ETH-M", "HOTEL-M", "UNIV-M", "ZARA1-M", "ZARA2-M"]
METRICS = ["minADE_K", "minFDE_K", "ADE@1", "FDE@1", "MR"]
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


def macro(rows):
    return {k: sum(r[k] for r in rows) / len(rows) for k in METRICS}


def report():
    out = ["=== Stage 1 · AnchorGeo 五场景（Mixed-test, K=20, val_new_minFDE20）==="]
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

    out.append(f"\n{'场景':<9} {'n':>7} " + " ".join(f"{m:>9s}" for m in METRICS)
               + "  |  ΔminFDE20 vs MoFlow  ΔMR")
    for s in SCENES:
        if s not in mine:
            out.append(f"{s:<9} {'待完成':>7}")
            continue
        r = mine[s]
        line = f"{s:<9} {r['n']:>7} " + " ".join(f"{r[k]:>9.4f}" for k in METRICS)
        if s in mo:
            line += (f"  |  {100*(r['minFDE_K']-mo[s]['minFDE_K'])/mo[s]['minFDE_K']:+8.2f}%"
                     f"  {100*(r['MR']-mo[s]['MR'])/mo[s]['MR']:+8.2f}%")
        out.append(line)

    if len(mine) == len(SCENES):
        A, B, C = macro([mine[s] for s in SCENES]), None, None
        out.append("")
        out.append(f"{'宏平均':<9} {'—':>7} " + " ".join(f"{A[k]:>9.4f}" for k in METRICS))
        if len(mo) == len(SCENES):
            B = macro([mo[s] for s in SCENES])
            out.append(f"{'MoFlow':<9} {'—':>7} " + " ".join(f"{B[k]:>9.4f}" for k in METRICS))
            out.append("  相对 MoFlow: " + "  ".join(
                f"{k}={100*(A[k]-B[k])/B[k]:+.1f}%" for k in METRICS))
        if len(so) == len(SCENES):
            C = macro([so[s] for s in SCENES])
            out.append(f"{'SocialLSTM':<9} {'-':>7} " + " ".join(f"{C[k]:>9.4f}" for k in METRICS))
            out.append("  相对 SocialLSTM: " + "  ".join(
                f"{k}={100*(A[k]-C[k])/C[k]:+.1f}%" for k in METRICS))
        out.append("")
        out.append("=== 分层（by_dimension，GAST-PF-AnchorGeo 宏平均）===")
        for dim in DIMS:
            agg = {}
            for s in SCENES:
                payload = load_arm(s)
                if payload is None:
                    continue
                d = payload["results"]["by_dimension"].get(dim, {})
                for v, e in d.items():
                    a = agg.setdefault(v, {"n": 0, "minFDE_K": 0.0, "MR": 0.0})
                    a["n"] += e["n"]
                    a["minFDE_K"] += e["minFDE_K"] * e["n"]
                    a["MR"] += e["MR"] * e["n"]
            if not agg:
                out.append(f"[{dim}] 缺失")
                continue
            out.append(f"\n[{dim}]  value |      n   minFDE20       MR")
            for v in sorted(agg, key=lambda z: (len(z), z)):
                e = agg[v]
                out.append(f"  {v:>6} | {e['n']:>7} {e['minFDE_K']/e['n']:>11.4f}"
                           f" {100*e['MR']/e['n']:>7.2f}%")
    return "\n".join(out)


def main():
    deadline = time.time() + MAX_HOURS * 3600
    while time.time() < deadline:
        pending = [s for s in SCENES if load_arm(s) is None]
        if not pending:
            print("STAGE1_ANCHORGEO_DONE")
            print(report())
            return 0
        print(f"[waiting] pending={pending}", flush=True)
        time.sleep(POLL_SECONDS)
    print("TIMEOUT")
    print(report())
    return 1


if __name__ == "__main__":
    sys.exit(main())
