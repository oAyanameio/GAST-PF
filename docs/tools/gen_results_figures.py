"""从 results.json 现算生成结果图（不手工转抄数据）。

用法：python docs/tools/gen_results_figures.py [--outdir paper/figures]

- Fig 1  scene_benchmark：九方法五场景 minFDE20 / minADE20 dot plot（按宏平均排序）
- Fig 2  missing_degradation：minFDE20 相对 missing=0 的退化比值（宏平均，log 轴）

数据源复用 gen_leaderboard_tables.collect()，路径变更只需改该文件。
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "docs" / "tools"))
sys.path.insert(0, str(REPO))

from gen_leaderboard_tables import SCENES, collect  # noqa: E402

# Okabe-Ito 色盲安全配色
SCENE_COLOR = {
    "ETH-M": "#0072B2", "HOTEL-M": "#56B4E9", "UNIV-M": "#009E73",
    "ZARA1-M": "#E69F00", "ZARA2-M": "#D55E00",
}
SCENE_MARKER = {"ETH-M": "o", "HOTEL-M": "s", "UNIV-M": "^",
                "ZARA1-M": "D", "ZARA2-M": "v"}
LINE_COLOR = {
    "GAST": "#0072B2",
    "MoFlow": "#D55E00",
    "SocialLSTM": "#CC79A7",
}
LINE_STYLE = {
    "GAST": "-", "MoFlow": ":", "SocialLSTM": (0, (3, 1, 1, 1)),
}
DISPLAY = {
    "GAST": "GAST (ours)",
    "GAST ablation (standard decoder)": "GAST w/o goal decoder",
    "Social-STGCNN": "Social-STGCNN",
}
# 内部对照（M0 / 消融臂）不进图：图中只比较 GAST 与外部基线
INTERNAL_EXCLUDE = {"M0", "GAST ablation (standard decoder)"}
MISSING_METHODS = ["GAST", "MoFlow", "SocialLSTM"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})


def fig_benchmark(data, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), dpi=300,
                             constrained_layout=True)
    for ax, key, label in zip(axes, ("minFDE_K", "minADE_K"),
                              (r"$\mathrm{minFDE}_{20}$ (m)",
                               r"$\mathrm{minADE}_{20}$ (m)")):
        rows = []
        for m, d in data.items():
            if m in INTERNAL_EXCLUDE:
                continue
            if len(d["per"]) == 5:
                avg = float(np.mean([d["per"][s][key] for s in SCENES]))
                rows.append((avg, m))
        rows.sort(reverse=True)  # 最优(小)在上
        names = [m for _, m in rows]

        for i, m in enumerate(names):
            per = data[m]["per"]
            is_ours = m == "GAST"
            if is_ours:
                ax.axhspan(i - 0.42, i + 0.42, color="#0072B2", alpha=0.07,
                           zorder=0)
            for s in SCENES:
                ax.plot(per[s][key], i, marker=SCENE_MARKER[s],
                        color=SCENE_COLOR[s], ms=4.2 if not is_ours else 5.0,
                        mec="white", mew=0.4, ls="none", zorder=3,
                        alpha=1.0 if is_ours else 0.85)
            avg = float(np.mean([per[s][key] for s in SCENES]))
            ax.plot(avg, i, marker="D", color="black", ms=4.6, mec="white",
                    mew=0.5, ls="none", zorder=4)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([DISPLAY.get(n, n) for n in names])
        for tick, n in zip(ax.get_yticklabels(), names):
            if n == "GAST":
                tick.set_fontweight("bold")
        ax.set_xlabel(label)
        ax.grid(axis="x", lw=0.4, alpha=0.35)
        ax.invert_yaxis()

    handles = [plt.Line2D([], [], ls="none", marker=SCENE_MARKER[s],
                          color=SCENE_COLOR[s], label=s) for s in SCENES]
    handles.append(plt.Line2D([], [], ls="none", marker="D", color="black",
                              label="Macro avg."))
    axes[1].legend(handles=handles, loc="lower right", frameon=False,
                   handletextpad=0.2, borderaxespad=0.2)
    for tag, ax in zip("ab", axes):
        ax.text(-0.32 if tag == "a" else -0.14, 1.06, f"({tag})",
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="top")
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"scene_benchmark.{ext}")
    plt.close(fig)


def fig_degradation(data, outdir):
    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=300,
                           constrained_layout=True)
    for m in MISSING_METHODS:
        per = data[m]["per"]
        if len(per) != 5:
            continue
        curves = []
        for s in SCENES:
            bg = per[s]["_by_group"]
            c = np.array([bg[f"{s}/Mixed/test/missing={i}"]["minFDE_K"]
                          for i in range(8)])
            curves.append(c / c[0])
        ratio = np.mean(np.stack(curves), axis=0)
        ms = 3.4 if m == "GAST" else 2.8
        ax.plot(range(8), ratio, linestyle=LINE_STYLE[m], color=LINE_COLOR[m],
                marker="o", ms=ms, lw=1.6 if m == "GAST" else 1.1,
                label=DISPLAY.get(m, m), zorder=5 if m == "GAST" else 3,
                alpha=1.0 if m == "GAST" else 0.85)
    ax.axvline(6.5, color="gray", lw=0.6, ls="--", alpha=0.6)
    ax.text(6.42, 2.9, "single frame\nremains", fontsize=6.5, color="gray",
            va="top", ha="right")
    ax.set_yscale("log")
    ax.set_ylim(0.95, 5.0)
    ax.set_xticks(range(8))
    ax.set_xlabel("Number of missing history frames")
    ax.set_ylabel(r"$\mathrm{minFDE}_{20}$ ratio vs.\ missing$=$0")
    ax.grid(lw=0.4, alpha=0.35, which="both")
    ax.legend(frameon=False, loc="lower right", handlelength=1.8,
              bbox_to_anchor=(1.0, 0.02))
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"missing_degradation.{ext}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="paper/figures")
    args = ap.parse_args()
    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    data = collect()
    fig_benchmark(data, outdir)
    fig_degradation(data, outdir)
    print(f"written to {outdir}/scene_benchmark.{{pdf,png}} "
          f"missing_degradation.{{pdf,png}}")


if __name__ == "__main__":
    main()
