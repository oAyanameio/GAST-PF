"""从 results.json 现算生成结果图（不手工转抄数据）。

用法：python docs/tools/gen_results_figures.py [--outdir docs/figures/results]

- Fig 1  scene_benchmark：七方法五场景 minFDE20 / minADE20 dot plot（按宏平均排序）
- Fig 2  missing_severity_degradation：三级分层的 minFDE20 相对退化百分比
- Fig S1 missing_degradation：精确缺失帧数的 minFDE20 相对退化比值

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

from gen_leaderboard_tables import (  # noqa: E402
    MISSING_SEVERITY_BINS,
    MISSING_SEVERITY_LABELS_EN,
    MISSING_SEVERITY_PERCENT_LABELS_EN,
    SCENES,
    collect,
    exact_missing_records,
    severity_macro,
)

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
}
LINE_STYLE = {
    "GAST": "-", "MoFlow": ":",
}
DISPLAY = {
    "GAST": "GAST (ours)",
    "GAST ablation (standard decoder)": "GAST w/o endpoint-guided decoding",
    "Social-STGCNN": "Social-STGCNN",
}
# 图中只比较 GAST 与外部基线。
EXCLUDED_METHODS = {"GAST ablation (standard decoder)"}
MISSING_METHODS = ["GAST", "MoFlow"]

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
        rows = benchmark_rows(data, key)
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


def benchmark_rows(data, key):
    """Return complete methods ordered from lowest to highest macro error."""
    rows = []
    for method, record in data.items():
        if method in EXCLUDED_METHODS:
            continue
        per_scene = record["per"]
        if set(per_scene) == set(SCENES):
            avg = float(np.mean([per_scene[scene][key] for scene in SCENES]))
            rows.append((avg, method))
    return sorted(rows)


def fig_degradation(data, outdir):
    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=300,
                           constrained_layout=True)
    for m in MISSING_METHODS:
        per = data[m]["per"]
        if len(per) != 5:
            continue
        curves = []
        for s in SCENES:
            exact = exact_missing_records(per[s])
            values = []
            for i in range(8):
                entry = exact.get(i)
                values.append(
                    entry.get("minFDE20", entry.get("minFDE_K"))
                    if entry else np.nan
                )
            c = np.asarray(values)
            curves.append(c / c[0])
        ratio = np.mean(np.stack(curves), axis=0)
        ms = 3.4 if m == "GAST" else 2.8
        missing_percent = np.arange(8) * 12.5
        ax.plot(missing_percent, ratio, linestyle=LINE_STYLE[m],
                color=LINE_COLOR[m],
                marker="o", ms=ms, lw=1.6 if m == "GAST" else 1.1,
                label=DISPLAY.get(m, m), zorder=5 if m == "GAST" else 3,
                alpha=1.0 if m == "GAST" else 0.85)
    severity_bands = (
        (-6.25, 31.25, "Near-complete\n(≤25%)"),
        (31.25, 56.25, "Moderate\n(>25–50%)"),
        (56.25, 93.75, "Severe\n(>50–87.5%)"),
    )
    colors = ("#EAF2F8", "#E8F6F3", "#FEF5E7")
    for (left, right, label), color in zip(severity_bands, colors):
        ax.axvspan(left, right, color=color, alpha=0.55, zorder=0)
        ax.text(
            (left + right) / 2,
            0.98,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=5.8,
            color="#555555",
        )
    ax.set_yscale("log")
    ax.set_ylim(0.95, 5.0)
    ax.set_xticks(np.arange(8) * 12.5)
    ax.set_xticklabels([f"{value:g}%" for value in np.arange(8) * 12.5])
    ax.set_xlabel("Missing history (% of 8 frames)")
    ax.set_ylabel(r"$\mathrm{minFDE}_{20}$ ratio vs.\ missing$=$0")
    ax.grid(lw=0.4, alpha=0.35, which="both")
    ax.legend(frameon=False, loc="lower right", handlelength=1.8,
              bbox_to_anchor=(1.0, 0.02))
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"missing_degradation_exact.{ext}")
    plt.close(fig)


def fig_severity_degradation(data, outdir):
    fig, ax = plt.subplots(figsize=(3.8, 2.7), dpi=300,
                           constrained_layout=True)
    positions = np.arange(len(MISSING_SEVERITY_BINS))
    for method in MISSING_METHODS:
        values = [
            severity_macro(
                data, method, severity_key,
                "minFDE_K_relative_vs_missing0_pct",
            )
            for severity_key, _, _ in MISSING_SEVERITY_BINS
        ]
        if any(value is None for value in values):
            continue
        ax.plot(
            positions,
            values,
            linestyle=LINE_STYLE[method],
            color=LINE_COLOR[method],
            marker="o",
            ms=3.8 if method == "GAST" else 3.2,
            lw=1.6 if method == "GAST" else 1.1,
            label=DISPLAY.get(method, method),
            zorder=5 if method == "GAST" else 3,
        )
    ax.axhline(0.0, color="gray", lw=0.7, ls="--", alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels([
        f"{MISSING_SEVERITY_LABELS_EN[key]}\n"
        f"({MISSING_SEVERITY_PERCENT_LABELS_EN[key]})"
        for key, _, levels in MISSING_SEVERITY_BINS
    ])
    ax.set_xlabel("Missing-history severity (% of 8 frames)")
    ax.set_ylabel(r"$\mathrm{minFDE}_{20}$ change vs. missing$=0$ (%)")
    ax.grid(axis="y", lw=0.4, alpha=0.35)
    ax.legend(frameon=False, loc="upper left", handlelength=1.8)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"missing_severity_degradation.{ext}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="docs/figures/results")
    args = ap.parse_args()
    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    data = collect()
    fig_benchmark(data, outdir)
    fig_severity_degradation(data, outdir)
    fig_degradation(data, outdir)
    print(f"written to {outdir}/scene_benchmark.{{pdf,png}} "
          f"missing_severity_degradation.{{pdf,png}} "
          f"missing_degradation_exact.{{pdf,png}}")


if __name__ == "__main__":
    main()
