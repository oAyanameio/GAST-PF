"""变体五场景对比报告：逐场景指标 + 宏平均 + 胜场统计 + 机制分层。

所有数字从 results.json 现算，不转抄文档。

用法（仓库根）：
    PYTHONPATH=. python scripts/结果分析/report_variant.py --variant GAST-PF-DAP \
        --root outputs/gast_dap_seed2024
    # 只看已有场景（未跑完时）
    PYTHONPATH=. python scripts/结果分析/report_variant.py --variant GAST-PF-DAP \
        --root outputs/gast_dap_seed2024 --partial
"""

import argparse
import glob
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCENES = ["ETH-M", "HOTEL-M", "UNIV-M", "ZARA1-M", "ZARA2-M"]
N = {"ETH-M": 1629, "HOTEL-M": 9477, "UNIV-M": 219006,
     "ZARA1-M": 20277, "ZARA2-M": 52497}
# MR 已于 2026-10-20 废弃（见 docs/results/MR废弃决策.md）。
# 新结果只有四项；历史结果仍含 MR，仅作存档展示。
BASE_KEYS = ["minADE_K", "minFDE_K", "ADE@1", "FDE@1"]
BASE_NAMES = ["minADE20", "minFDE20", "ADE@1", "FDE@1"]
KEYS = BASE_KEYS + ["MR"]
NAMES = BASE_NAMES + ["MR"]

# 外部/内部对照的逐场景数值（来源见 docs/results/trajgap_bench_leaderboard.md，
# 由 gen_leaderboard_tables.py 从各自 results.json 现算）
COMPARATORS = {
    "MoFlow": {
        "ETH-M": [0.6819, 0.9389, 1.4669, 2.5903, 0.1430],
        "HOTEL-M": [0.2777, 0.4247, 0.6818, 1.1561, 0.0387],
        "UNIV-M": [0.3478, 0.5572, 1.0569, 1.8893, 0.0335],
        "ZARA1-M": [0.2566, 0.4057, 0.9863, 1.7563, 0.0085],
        "ZARA2-M": [0.1839, 0.2867, 0.7152, 1.2559, 0.0061],
    },
    "SocialLSTM": {
        "ETH-M": [0.8430, 1.0456, 2.1039, 3.2800, 0.1584],
        "HOTEL-M": [0.2924, 0.3871, 0.9876, 1.5940, 0.0366],
        "UNIV-M": [0.4543, 0.6512, 1.2964, 2.2710, 0.0484],
        "ZARA1-M": [0.4126, 0.5650, 1.2546, 2.1104, 0.0303],
        "ZARA2-M": [0.2858, 0.3907, 0.9964, 1.6772, 0.0267],
    },
    "M0": {
        "ETH-M": [0.6145, 0.8532, 1.4034, 2.4877, 0.1197],
        "HOTEL-M": [0.2396, 0.3649, 0.5278, 0.8983, 0.0385],
        "UNIV-M": [0.3473, 0.5696, 0.9880, 1.8068, 0.0438],
        "ZARA1-M": [0.2398, 0.3877, 0.8368, 1.5223, 0.0163],
        "ZARA2-M": [0.1700, 0.2830, 0.6687, 1.2131, 0.0150],
    },
    "GAST-PF": {
        "ETH-M": [0.6191, 0.8544, 1.4232, 2.4693, 0.1191],
        "HOTEL-M": [0.2572, 0.3877, 0.5079, 0.8615, 0.0412],
        "UNIV-M": [0.3258, 0.5292, 0.8615, 1.6233, 0.0365],
        "ZARA1-M": [0.2494, 0.4014, 0.9072, 1.6382, 0.0151],
        "ZARA2-M": [0.1808, 0.3004, 0.7483, 1.3425, 0.0132],
    },
}

# 其它内部变体：直接从各自产物目录读取
BASELINE_ROOTS = [
    "outputs/unitraj_encoder_bench_seed2024",
    "outputs/unitraj_encoder_single_scene_seed2024",
]

VARIANT_ROOTS = {
    "MRT": ("outputs/gast_mrt_seed2024", "GAST-PF-MRT"),
    "FrameGeo": ("outputs/stage2_frame_geo_seed2024", "GAST-PF-FrameGeo"),
    "AnchorGeo": ("outputs/stage1_anchor_geo_seed2024", "GAST-PF-AnchorGeo"),
}


def load_variant(root, variant):
    out = {}
    for f in glob.glob(f"{root}/{variant}_*_mixed-direct_seed2024_uni/eval/Mixed/*/results.json"):
        scene = next((s for s in SCENES if f"_{s}_mixed" in f), None)
        if scene is None:
            continue
        overall = json.loads(Path(f).read_text())["results"]["overall"]
        out[scene] = [overall.get(k, float("nan")) for k in KEYS]
    return out


def load_dims(roots, variant):
    if isinstance(roots, str):
        roots = [roots]
    dims = {}
    files = []
    for root in roots:
        files += glob.glob(f"{root}/{variant}_*_mixed-direct_seed2024_uni/eval/Mixed/*/results.json")
    for f in files:
        scene = next((s for s in SCENES if f"_{s}_mixed" in f), None)
        if scene is None:
            continue
        dims[scene] = json.loads(Path(f).read_text())["results"].get("by_dimension", {})
    return dims


def macro(data, scenes):
    return [sum(data[s][i] for s in scenes) / len(scenes) for i in range(len(KEYS))]


def micro(data, scenes):
    total = sum(N[s] for s in scenes)
    return [sum(data[s][i] * N[s] for s in scenes) / total for i in range(len(KEYS))]


def fmt(values, mr_is_pct=True):
    cells = []
    for i, v in enumerate(values):
        cells.append(f"{100*v:.2f}%" if (mr_is_pct and i == 4) else f"{v:.4f}")
    return cells


def binomial_se(p, n):
    return math.sqrt(p * (1 - p) / n)


def main():
    global KEYS, NAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="产物里的 model_version，如 GAST-PF-DAP")
    ap.add_argument("--root", required=True)
    ap.add_argument("--partial", action="store_true", help="允许只报告已完成场景")
    ap.add_argument("--compare", default="GAST-PF",
                    help="分层对照变体：GAST-PF / MRT / FrameGeo / AnchorGeo / none")
    args = ap.parse_args()

    mine = load_variant(args.root, args.variant)
    scenes = [s for s in SCENES if s in mine]
    # 缺 MR（新口径）时整体降级为四指标
    keep = len(BASE_KEYS) if any(
        mine[s][KEYS.index("MR")] != mine[s][KEYS.index("MR")] for s in scenes
    ) else len(KEYS)
    KEYS, NAMES = KEYS[:keep], NAMES[:keep]
    if not scenes:
        raise SystemExit("没有可用结果")
    missing = [s for s in SCENES if s not in mine]
    if missing and not args.partial:
        raise SystemExit(f"场景不完整，缺 {missing}；加 --partial 可只报已完成场景")

    print(f"# {args.variant} · 场景 {len(scenes)}/5"
          + (f"（缺 {missing}，**非最终**）" if missing else ""))
    print("\n## 逐场景\n")
    print("| 场景 | n | " + " | ".join(NAMES) + " |")
    print("|---|---:|" + "---:|" * 5)
    for s in scenes:
        print(f"| {s} | {N[s]:,} | " + " | ".join(fmt(mine[s][:len(NAMES)])) + " |")
    mac = macro(mine, scenes)
    mic = micro(mine, scenes)
    print("| **宏平均** | — | " + " | ".join(fmt(mac)) + " |")
    print("| 样本加权 | " + f"{sum(N[s] for s in scenes):,} | " + " | ".join(fmt(mic)) + " |")

    print("\n## 宏平均对比\n")
    print("| 方法 | " + " | ".join(NAMES) + " |")
    print("|---|" + "---:|" * 5)
    print(f"| **{args.variant}** | " + " | ".join(fmt(mac)) + " |")
    others = {}
    for name, table in COMPARATORS.items():
        if not all(s in table for s in scenes):
            continue
        others[name] = table
        print(f"| {name} | " + " | ".join(fmt(macro(table, scenes))) + " |")
    for name, (root, variant) in VARIANT_ROOTS.items():
        data = load_variant(root, variant)
        if all(s in data for s in scenes):
            others[name] = data
            print(f"| {name} | " + " | ".join(fmt(macro(data, scenes))) + " |")

    print("\n## 相对差（负 = 本变体更优）\n")
    print("| 对比 | " + " | ".join(NAMES) + " | 胜场 |")
    print("|---|" + "---:|" * 5 + "---:|")
    for name, table in others.items():
        other_mac = macro(table, scenes)
        diffs = [f"{100*(mac[i]-other_mac[i])/other_mac[i]:+.2f}%" for i in range(5)]
        wins = sum(1 for s in scenes for i in range(5) if mine[s][i] < table[s][i])
        print(f"| vs {name} | " + " | ".join(diffs) + f" | {wins}/{5*len(scenes)} |")

    print("\n## MR 显著性（二项标准误）\n")
    print("| 对比 | Δ(pp) | 合并 SE(pp) | \\|Δ\\|/SE | 判定 |")
    print("|---|---:|---:|---:|---|")
    for name, table in others.items():
        d = sum(mine[s][4] - table[s][4] for s in scenes) / len(scenes)
        sd = math.sqrt(sum(binomial_se(mine[s][4], N[s]) ** 2
                           + binomial_se(table[s][4], N[s]) ** 2 for s in scenes)) / len(scenes)
        z = abs(d) / sd if sd else float("inf")
        verdict = "显著" if z >= 2 else ("边缘" if z >= 1.5 else "噪声")
        print(f"| vs {name} | {100*d:+.2f} | {100*sd:.2f} | {z:.2f} | {verdict} |")

    dims = load_dims(args.root, args.variant)
    # 对照分层只在**场景集合完全一致**时计算；否则样本集不同，聚合值不可比。
    base_dims = load_dims(BASELINE_ROOTS, "UT-ENC") if args.compare == "GAST-PF" else {}
    if args.compare and args.compare in VARIANT_ROOTS:
        croot, cvar = VARIANT_ROOTS[args.compare]
        base_dims = load_dims(croot, cvar)
    base_ok = bool(base_dims) and all(s in base_dims for s in scenes)
    if base_dims and not base_ok:
        got = sorted(base_dims)
        print(f"> ⚠ 分层对照跳过：{args.compare} 只覆盖 {got}，与本变体 {scenes} 不一致；"
              f"不同样本集的聚合值不可比。\n")
    if dims and all(s in dims for s in scenes):
        print(f"\n## 机制分层（样本加权，对照 {args.compare}）\n")
        for dim in ("anchor_lag", "missing_count", "terminal_missing"):
            agg, base = {}, {}
            for s in scenes:
                for bucket, v in (dims[s].get(dim) or {}).items():
                    a = agg.setdefault(bucket, {"n": 0, "f": 0.0, "m": 0.0})
                    a["n"] += v["n"]
                    a["f"] += v["minFDE_K"] * v["n"]
                    a["m"] += v["MR"] * v["n"]
                for bucket, v in ((base_dims.get(s) or {}).get(dim) or {}).items():
                    b = base.setdefault(bucket, {"n": 0, "f": 0.0, "m": 0.0})
                    b["n"] += v["n"]
                    b["f"] += v["minFDE_K"] * v["n"]
                    b["m"] += v["MR"] * v["n"]
            if not agg:
                continue
            print(f"### {dim}\n")
            if not base_ok:
                print("| 桶 | n | minFDE20 | MR |")
                print("|---:|---:|---:|---:|")
                for k in sorted(agg, key=lambda x: (len(str(x)), str(x))):
                    a = agg[k]
                    print(f"| {k} | {a['n']:,} | {a['f']/a['n']:.4f} | {100*a['m']/a['n']:.2f}% |")
                print()
                continue
            print("| 桶 | n | 本变体 minFDE20 | 对照 minFDE20 | Δ% | 本变体 MR | 对照 MR | ΔMR% |")
            print("|---:|---:|---:|---:|---:|---:|---:|---:|")
            for k in sorted(agg, key=lambda x: (len(str(x)), str(x))):
                a = agg[k]
                b = base.get(k)
                if b and b["n"]:
                    df = 100 * (a["f"] / a["n"] - b["f"] / b["n"]) / (b["f"] / b["n"])
                    dm = 100 * (a["m"] / a["n"] - b["m"] / b["n"]) / (b["m"] / b["n"])
                    print(f"| {k} | {a['n']:,} | {a['f']/a['n']:.4f} | {b['f']/b['n']:.4f} | "
                          f"{df:+.2f}% | {100*a['m']/a['n']:.2f}% | {100*b['m']/b['n']:.2f}% | {dm:+.2f}% |")
                else:
                    print(f"| {k} | {a['n']:,} | {a['f']/a['n']:.4f} | — | — | "
                          f"{100*a['m']/a['n']:.2f}% | — | — |")
            print()


if __name__ == "__main__":
    main()
