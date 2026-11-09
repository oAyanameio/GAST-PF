'''Generate the TrajGap-Bench Mixed split-count figure and table.

Run with:
    python docs/tools/gen_trajgap_dataset_distribution.py
'''
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.datamodule.trajimpute_dataset import (  # noqa: E402
    SCENES, TRAJIMPUTE_ROOT, TrajGapDataset,
)

SPLIT_ORDER = ('train', 'test', 'val')
SPLIT_LABELS = {'train': 'Training', 'test': 'Test', 'val': 'Validation'}
SPLIT_COLORS = {'train': '#0072B2', 'test': '#009E73', 'val': '#E69F00'}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.7,
})


def collect_counts(data_root):
    '''Count focal samples through the same Mixed adapter used in training.'''
    return {
        scene: {
            split: len(TrajGapDataset(data_root, scene, split))
            for split in SPLIT_ORDER
        }
        for scene in SCENES
    }


def split_totals(counts):
    return {
        split: sum(counts[scene][split] for scene in SCENES)
        for split in SPLIT_ORDER
    }


def plot_distribution(counts, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(SCENES))
    width = 0.23
    fig, ax = plt.subplots(figsize=(8.6, 4.1), dpi=300, constrained_layout=True)

    for index, split in enumerate(SPLIT_ORDER):
        values = [counts[scene][split] for scene in SCENES]
        bars = ax.bar(
            x + (index - 1) * width, values, width,
            label=SPLIT_LABELS[split], color=SPLIT_COLORS[split],
            edgecolor='white', linewidth=0.35, zorder=3,
        )
        ax.bar_label(
            bars, labels=[f'{value:,}' for value in values],
            padding=2, fontsize=7,
        )

    ax.set_title('TrajGap Dataset Distribution by Split', pad=9)
    ax.set_xlabel('Scene')
    ax.set_ylabel('Number of Focal Trajectory Samples')
    ax.set_xticks(x, SCENES)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda value, _: '0' if value == 0 else f'{value / 1000:.0f}k'
    ))
    max_value = max(
        counts[scene][split] for scene in SCENES for split in SPLIT_ORDER
    )
    ax.set_ylim(0, max_value * 1.16)
    ax.grid(axis='y', color='#D9DEE5', linewidth=0.55, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='upper right')

    outputs = []
    for extension in ('png', 'pdf'):
        path = outdir / f'trajgap_dataset_distribution.{extension}'
        fig.savefig(path, bbox_inches='tight')
        outputs.append(path)
    plt.close(fig)
    return outputs


def render_markdown(counts, outdir, report_path):
    totals = split_totals(counts)
    image_rel = os.path.relpath(
        outdir / 'trajgap_dataset_distribution.png', report_path.parent
    ).replace(os.sep, '/')
    pdf_rel = os.path.relpath(
        outdir / 'trajgap_dataset_distribution.pdf', report_path.parent
    ).replace(os.sep, '/')
    lines = [
        '# TrajGap-Bench 数据集分布',
        '',
        '> 统计口径：每个场景和 split 将官方 Easy、Hard 数据合并为 Mixed；样本数按 `TrajGapDataset` 实际产生的 focal 行人轨迹计数。',
        '',
        f'![TrajGap 各场景数据划分分布]({image_rel})',
        '',
        f'[PDF 矢量图]({pdf_rel})',
        '',
        '| 场景 | Training | Test | Validation | 合计 |',
        '|---|---:|---:|---:|---:|',
    ]
    for scene in SCENES:
        train = counts[scene]['train']
        test = counts[scene]['test']
        val = counts[scene]['val']
        row_total = train + test + val
        lines.append(
            f'| {scene} | {train:,} | {test:,} | {val:,} | {row_total:,} |'
        )
    grand_total = sum(totals.values())
    train_total = totals['train']
    test_total = totals['test']
    val_total = totals['val']
    lines.append(
        f'| **Total** | **{train_total:,}** | **{test_total:,}** | '
        f'**{val_total:,}** | **{grand_total:,}** |'
    )
    lines.extend([
        '',
        'Train、validation、test 均使用同一 Mixed 合并规则；每个样本以一个 focal 行人为中心，包含 8 帧历史输入和 12 帧未来目标。',
        '',
    ])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default=TRAJIMPUTE_ROOT)
    parser.add_argument('--outdir', default='docs/figures/results')
    parser.add_argument('--report', default='docs/results/trajgap_dataset_distribution.md')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = REPO / outdir
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = REPO / report_path

    counts = collect_counts(args.data_root)
    outputs = plot_distribution(counts, outdir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_markdown(counts, outdir, report_path), encoding='utf-8'
    )

    print(f'wrote {report_path}')
    for path in outputs:
        print(f'wrote {path}')
    totals = split_totals(counts)
    print('totals: ' + ', '.join(
        f'{SPLIT_LABELS[split]}={totals[split]:,}' for split in SPLIT_ORDER
    ))


if __name__ == '__main__':
    main()
