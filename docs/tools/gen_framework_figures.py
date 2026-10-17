#!/usr/bin/env python3
"""GAST-PF 框架图生成器（整体框架 / 编码器 / 解码器）。

只表示模型组成（结构、张量形状、模块连接），不含任何实验结果。

版式约定（三图统一风格，对齐参考论文图）：
  · 编码器：横向主链 + 侧向输入与跳连（mask → ghost token / δ → decay）；
  · 解码器：双分支 fork（mode / state）→ 汇合为联合 query → 横向融合链；
  · 算子块：注意力=灰、Mamba/读出=白、mask 驱动=橙、缺失间隔驱动=青、
    输入与预测头=浅蓝、query/token 图元=淡紫；
  · 模块分组：浅蓝虚线圆角框 + 左上角标题；箭头为细黑实线直角折线。

坐标约定：1 数据单位 = 1/100 inch；figsize 与画布等比例。
绘制纪律：每条箭头起点贴源图元、终点距目标边界 2 单位；脚本内置几何审计
（运行后打印 [audit] 行，端点悬空 > 3 单位即报缺陷）。

输出：docs/figures/*.png (300 dpi) 与同名 *.pdf（矢量，供论文排版）。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.textpath import TextPath

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False

FIGDIR = Path(__file__).resolve().parents[1] / "figures"

# ----------------------------------------------------------------------------
# 样式常量
# ----------------------------------------------------------------------------
INK = "#1A1A1A"
MUTED = "#5B6B7A"
GRID = "#AEBAC6"

C_BLUE, C_BLUE_S = "#D8E6F8", "#4A6B8A"      # 输入 / 预测头
C_ORANGE, C_ORANGE_S = "#EFA277", "#B96A38"  # mask / ghost token
C_TEAL, C_TEAL_S = "#2E9AA8", "#1E6B76"      # 缺失间隔驱动的衰减
C_GRAY, C_GRAY_S = "#D9D9D9", "#8C8C8C"      # 注意力算子
C_LAV, C_LAV_S = "#E3D3F0", "#B9A6D1"        # token / query 图元
C_HL = "#5CB85C"                             # 高亮行
C_WHITE, C_BLACK = "#FFFFFF", "#000000"
C_GROUP, C_GROUP_S = "#F4F8FC", "#8FB8D8"    # 模块分组（虚线框）
C_TRAJ = "#93A7BC"
C_CYAN = "#3FC8D8"

FS_LABEL = 14
FS_SUB = 12
FS_CAPTION = 13
FS_ANNOT = 11
FS_LEGEND = 12

ELEMS = []
ARROWS = []


def reg(name, x, y, w, h):
    if name:
        ELEMS.append((name, float(x), float(y), float(x + w), float(y + h)))


def _gap(p, elem):
    _, x0, y0, x1, y1 = elem
    dx = max(x0 - p[0], 0.0, p[0] - x1)
    dy = max(y0 - p[1], 0.0, p[1] - y1)
    return (dx * dx + dy * dy) ** 0.5


def audit(label):
    bad = []
    targets = [e for e in ELEMS if not e[0].startswith("panel")]
    for name, p0, p1 in ARROWS:
        for tag, p in (("start", p0), ("end", p1)):
            elem = min(targets, key=lambda e: _gap(p, e))
            gap = _gap(p, elem)
            if gap > 3.0:
                bad.append(f"    {name:26s} {tag:5s} {str(p):16s} "
                           f"nearest={elem[0]:18s} gap={gap:5.1f}")
    print(f"[audit] {label}: {len(ARROWS)} arrows, {len(bad)} dangling endpoint(s)")
    for line in bad:
        print(line)
    return len(bad)


# ----------------------------------------------------------------------------
# 文本宽度测量与自动缩号
# ----------------------------------------------------------------------------
def _text_width(text, fs, bold=True):
    fp = FontProperties(family="DejaVu Sans", size=fs,
                        weight="bold" if bold else "normal")
    widest = 0.0
    for line in text.split("\n"):
        if not line.strip():
            continue
        ext = TextPath((0, 0), line, prop=fp).get_extents()
        widest = max(widest, ext.width)
    return widest / 72.0 * 100.0


def _fit(text, fs, max_width, min_fs=7.0, bold=True):
    if not text or max_width <= 0:
        return fs
    w = _text_width(text, fs, bold)
    if w <= max_width:
        return fs
    return max(min_fs, fs * max_width / w)


# ----------------------------------------------------------------------------
# 基础绘制工具
# ----------------------------------------------------------------------------
def new_canvas(width, height):
    fig = plt.figure(figsize=(width / 100.0, height / 100.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, label, sub=None, fill=C_BLUE, stroke=C_BLUE_S, fs=FS_LABEL,
        sub_fs=FS_SUB, tc=INK, sub_tc=None, rounding=9, lw=1.4, ls="solid", z=2,
        name=None):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}",
        linewidth=lw, edgecolor=stroke, facecolor=fill, linestyle=ls, zorder=z))
    cx, cy = x + w / 2.0, y + h / 2.0
    if sub is None:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=_fit(label, fs, w - 22), color=tc, fontweight="bold",
                zorder=z + 1, linespacing=1.3)
    else:
        ax.text(cx, cy + h * 0.18, label, ha="center", va="center",
                fontsize=_fit(label, fs, w - 22), color=tc, fontweight="bold",
                zorder=z + 1, linespacing=1.25)
        ax.text(cx, cy - h * 0.22, sub, ha="center", va="center",
                fontsize=_fit(sub, sub_fs, w - 16, bold=False),
                color=sub_tc or tc, zorder=z + 1, linespacing=1.2)
    reg(name, x, y, w, h)


def panel(ax, x, y, w, h, title, fs=FS_LABEL, z=1, rounding=14, name=None,
          title_tc="#1F3B57", title_x=None):
    """模块分组框：浅蓝虚线圆角框 + 标题（默认左上角，可用 title_x 避让连线）。"""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}",
        linewidth=1.5, edgecolor=C_GROUP_S, facecolor=C_GROUP,
        linestyle=(0, (6, 4)), zorder=z))
    tx = x + 16 if title_x is None else title_x
    ax.text(tx, y + h - 21, title, ha="left", va="center",
            fontsize=_fit(title, fs, w - 32), color=title_tc, fontweight="bold",
            zorder=z + 1)
    reg(name, x, y, w, h)


def arrow(ax, p0, p1, color=INK, lw=1.5, ms=13, z=3, ls="solid", label=""):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
        shrinkA=0, shrinkB=0, zorder=z, linestyle=ls, joinstyle="miter"))
    ARROWS.append((label or "arrow", tuple(p0), tuple(p1)))


def poly_arrow(ax, pts, color=INK, lw=1.5, ms=13, z=3, ls="solid", label=""):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
                           zorder=z, linestyle=ls, solid_capstyle="round"))
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ms=ms, z=z, ls=ls,
          label=label)
    ARROWS[-1] = (label or "poly", tuple(pts[0]), tuple(pts[-1]))


def caption(ax, x, y, text, fs=FS_CAPTION, color=INK, bold=True, ha="center",
            max_width=None):
    fs_eff = _fit(text, fs, max_width, bold=bold) if max_width else fs
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs_eff, color=color,
            fontweight="bold" if bold else "normal", zorder=6)


def annot(ax, x, y, text, fs=FS_ANNOT, color=MUTED, ha="left", style="italic",
          max_width=None):
    fs_eff = _fit(text, fs, max_width, bold=False) if max_width else fs
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs_eff, color=color,
            fontstyle=style, zorder=6)


def mul_node(ax, cx, cy, r=18, z=4, name=None):
    ax.add_patch(Circle((cx, cy), r, facecolor=C_WHITE, edgecolor=INK, lw=1.5,
                        zorder=z))
    ax.text(cx, cy - 1, "⊙", ha="center", va="center", fontsize=16, zorder=z + 1)
    reg(name, cx - r, cy - r, 2 * r, 2 * r)


def stacked_block(ax, x, y, w, h, label, n=4, offset=9, fill=C_WHITE, stroke=INK,
                  fs=FS_SUB, z=2, name=None):
    for i in range(n - 1, -1, -1):
        ax.add_patch(FancyBboxPatch(
            (x + i * offset, y + i * offset), w, h,
            boxstyle="round,pad=0,rounding_size=8", linewidth=1.4,
            edgecolor=stroke, facecolor=fill, zorder=z))
    ax.text(x + w / 2.0, y + h / 2.0, label, ha="center", va="center",
            fontsize=_fit(label, fs, h - 18), color=INK, fontweight="bold",
            rotation=90, zorder=z + 1, linespacing=1.25)
    span = (n - 1) * offset
    reg(name, x, y, w + span, h + span)


def glyph_grid(ax, x, y, cell, mat, axis_labels=True, highlight=(), z=4,
               label_fs=12, name=None):
    rows, cols = len(mat), len(mat[0])
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle(
                (x + c * cell, y + (rows - 1 - r) * cell), cell, cell,
                facecolor=C_BLACK if mat[r][c] else C_WHITE, edgecolor=GRID,
                lw=0.9, zorder=z))
    if axis_labels:
        ax.text(x - cell * 0.7, y + rows * cell / 2.0, "N", rotation=90,
                ha="center", va="center", fontsize=label_fs, fontweight="bold")
        ax.text(x + cols * cell / 2.0, y + rows * cell + cell * 0.7, "T",
                ha="center", va="center", fontsize=label_fs, fontweight="bold")
    for (r0, c0, nr, nc, col) in highlight:
        y0 = y + (rows - r0 - nr) * cell
        ax.add_patch(FancyBboxPatch(
            (x + c0 * cell - 3, y0 - 3), nc * cell + 6, nr * cell + 6,
            boxstyle="round,pad=0,rounding_size=3", fill=False, edgecolor=col,
            lw=2.2, zorder=z + 2))
    reg(name, x, y, cols * cell, rows * cell)


def glyph_col(ax, x, y, cell, colors, z=4, edge=GRID, lw=0.9, name=None):
    for i, col in enumerate(colors):
        ax.add_patch(Rectangle((x, y + i * cell), cell, cell, facecolor=col,
                               edgecolor=edge, lw=lw, zorder=z))
    reg(name, x, y, cell, cell * len(colors))


def glyph_row(ax, x, y, cell, colors, z=4, edge=GRID, lw=0.9, name=None):
    for i, col in enumerate(colors):
        ax.add_patch(Rectangle((x + i * cell, y), cell, cell, facecolor=col,
                               edgecolor=edge, lw=lw, zorder=z))
    reg(name, x, y, cell * len(colors), cell)


def glyph_traj(ax, x, y, w, h, n=20, hl=(0,), z=5, base=C_TRAJ, hl_color=C_HL,
               lw=1.2):
    t = np.linspace(0, 1, 60)
    for i in range(n):
        s = (i - (n - 1) / 2.0) / max(n - 1, 1)
        px = x + w * t
        py = (y + h * 0.12 + h * 0.58 * t + h * 0.24 * s * t
              + h * 0.04 * np.sin(np.pi * t) * (s * 2))
        if i in hl:
            ax.plot(px, py, color=hl_color, lw=lw * 2.6, zorder=z + 2,
                    solid_capstyle="round")
        else:
            ax.plot(px, py, color=base, lw=lw, zorder=z, alpha=0.92)
    ax.add_patch(Circle((x, y + h * 0.12), h * 0.035, facecolor=INK,
                        edgecolor="none", zorder=z + 3))


def fan_box(ax, x, y, w, h, name=None, n=20, hl=(1,), z=4):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=8", facecolor=C_WHITE,
        edgecolor=GRID, lw=1.3, zorder=z))
    mx, my = w * 0.14, h * 0.16
    glyph_traj(ax, x + mx, y + my, w - 2 * mx, h - 2 * my, n=n, hl=hl, z=z + 1)
    reg(name, x, y, w, h)


def legend_strip(ax, x, y, entries, fs=FS_LEGEND, z=6, pad=28, gap=42):
    widths = [_text_width(s, fs, bold=False) for s in entries]
    inner = sum(widths) + gap * (len(entries) - 1)
    w = inner + 2 * pad
    h = fs * 1.9 + 18
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=8", facecolor=C_WHITE,
        edgecolor="#C9CFD6", lw=1.3, zorder=z))
    cursor = x + pad
    for s, tw in zip(entries, widths):
        ax.text(cursor, y + h / 2.0, s, ha="left", va="center", fontsize=fs,
                color=INK, zorder=z + 1)
        cursor += tw + gap
    return w


def save(fig, stem):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    png = FIGDIR / f"{stem}.png"
    pdf = FIGDIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, facecolor=C_WHITE)
    fig.savefig(pdf, facecolor=C_WHITE)
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")


LOSS_TEXT = ("Loss:  regression + mode classification\n"
             "neighbour regression + dense regression + Laplace NLL")


# ----------------------------------------------------------------------------
# 图 1：编码器（横向主链 + 侧向 mask 输入与跳连）
# ----------------------------------------------------------------------------
def figure_encoder():
    ELEMS.clear()
    ARROWS.clear()
    W, H = 1600, 620
    fig, ax = new_canvas(W, H)

    # ---------------- 主链（横向单行）----------------
    box(ax, 40, 265, 230, 70, "History features",
        "8 frames × 0.4 s\nfocal + neighbours", fill=C_BLUE, stroke=C_BLUE_S,
        fs=13, name="history")
    arrow(ax, (270, 300), (308, 300), label="history->agent_emb")

    mat_agent = [[0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0],
                 [0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0]]
    glyph_grid(ax, 310, 262, 13, mat_agent, name="agent_emb")
    caption(ax, 342, 240, "Agent embeddings\nx [N, T, C]", fs=12)
    arrow(ax, (375, 300), (408, 300), label="agent_emb->tok_a")

    glyph_col(ax, 410, 262, 13, [C_LAV] * 4 + [C_ORANGE], name="tok_a")

    box(ax, 460, 250, 170, 100, "GSM Spatial\nAttention", None,
        fill=C_GRAY, stroke=C_GRAY_S, fs=13.5, name="gsm_spatial")
    arrow(ax, (423, 300), (458, 300), label="tok_a->gsm_spatial")

    glyph_col(ax, 665, 262, 13, [C_LAV] * 4 + [C_ORANGE], name="tok_b")
    arrow(ax, (630, 300), (663, 300), label="gsm_spatial->tok_b")

    mat_spa = [[0, 0, 0, 0, 0],
               [0, 0, 1, 1, 0],
               [0, 0, 0, 0, 0],
               [0, 1, 1, 0, 0],
               [0, 0, 0, 0, 0]]
    glyph_grid(ax, 715, 262, 13, mat_spa, name="spa_emb")
    caption(ax, 747, 240, "Spatial embeddings\nz [N, T, C]", fs=12)
    arrow(ax, (678, 300), (713, 300), label="tok_b->spa_emb")

    mul_node(ax, 840, 300, r=18, name="mul")
    arrow(ax, (780, 300), (820, 300), label="spa_emb->mul")

    stacked_block(ax, 880, 225, 100, 150, "Bidirectional\nTemporal Mamba", n=4,
                  offset=9, fs=12, name="mamba_stack")
    arrow(ax, (858, 300), (878, 300), label="mul->mamba")

    box(ax, 1050, 260, 160, 80, "Readout", "y_fwd + flip(y_bwd)\n→ last time step",
        fill=C_WHITE, stroke=INK, fs=13, name="readout")
    arrow(ax, (1009, 300), (1048, 300), label="mamba->readout")

    mat_out = [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [1, 1, 1, 1, 1],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]
    glyph_grid(ax, 1240, 262, 13, mat_out, highlight=[(2, 0, 1, 5, C_HL)],
               name="out_emb")
    caption(ax, 1272, 242, "Actor embedding  h [N, C]", fs=12, max_width=260)
    arrow(ax, (1210, 300), (1238, 300), label="readout->out_emb")

    # ---------------- 侧向输入：validity mask ----------------
    mat_mask = [[0, 1, 0, 0, 1],
                [0, 1, 0, 0, 1],
                [0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0]]
    glyph_grid(ax, 100, 500, 13, mat_mask,
               highlight=[(0, 3, 2, 2, C_ORANGE_S)], name="mask")
    caption(ax, 132, 480, "Validity mask  m [N, T]", fs=12, max_width=250)

    glyph_col(ax, 190, 500, 13, [C_WHITE, C_BLACK, C_WHITE, C_WHITE, C_BLACK],
              name="mask_col")
    arrow(ax, (165, 532), (188, 532), label="mask->mask_col")

    box(ax, 230, 498, 190, 70, "GSM Module", "ghost token pooling",
        fill=C_ORANGE, stroke=C_ORANGE_S, fs=13, name="gsm_module")
    arrow(ax, (203, 532), (228, 532), label="mask_col->gsm_module")
    poly_arrow(ax, [(325, 498), (325, 478), (416, 478), (416, 329)],
               label="gsm_module->ghost")
    annot(ax, 402, 348, "ghost token", ha="right", fs=10.5)

    # mask -> δ 跳连（沿画布顶部）
    poly_arrow(ax, [(132, 565), (132, 590), (752, 590), (752, 515)],
               color=MUTED, lw=1.2, ms=11, ls=(0, (5, 3)), label="mask->delta")
    annot(ax, 420, 604, "δ derived from m", fs=10.5)

    glyph_row(ax, 700, 500, 13, [C_WHITE, C_BLACK, C_WHITE, C_WHITE, C_BLACK,
                                 C_WHITE, C_WHITE, C_BLACK], name="delta_row")
    annot(ax, 812, 506, "missing-run delta  δ [N, T]", fs=11)

    box(ax, 720, 405, 240, 56, "Temporal Decay", "γ = exp(−ReLU(W δ))",
        fill=C_TEAL, stroke=C_TEAL_S, tc=C_WHITE, sub_tc="#EAF7F9", fs=13,
        name="temporal_decay")
    arrow(ax, (752, 500), (752, 463), label="delta->decay")

    glyph_row(ax, 780, 380, 12, [C_TEAL] * 8, name="gamma_row")
    arrow(ax, (838, 405), (838, 394), label="decay->gamma")
    arrow(ax, (838, 380), (838, 320), label="gamma->mul")

    # ---------------- 分组框与说明 ----------------
    panel(ax, 300, 150, 1030, 320, "GAST-PF\nEncoder", name="panel_enc")
    annot(ax, 948, 205, "4 blocks per direction", ha="center", fs=11)
    annot(ax, 948, 186, "forward + flipped backward, summed", ha="center", fs=10.5)

    legend_strip(ax, 40, 26,
                 ["N agents", "T history", "C channels", "⊙ element-wise"])
    annot(ax, 700, 48, "encoder output h [N, C]  →  structured future decoder",
          fs=11)

    save(fig, "gast_pf_encoder_framework")
    return audit("encoder")


# ----------------------------------------------------------------------------
# 图 2：解码器（双分支 fork → 汇合 → 横向融合链）
# ----------------------------------------------------------------------------
def figure_decoder():
    ELEMS.clear()
    ARROWS.clear()
    W, H = 1900, 800
    fig, ax = new_canvas(W, H)

    # ---------------- 左列：场景上下文与两类 query ----------------
    box(ax, 40, 720, 460, 50, "Neighbour prediction head", None,
        fill=C_WHITE, stroke=C_LAV_S, fs=12.5, name="neighbour_head")
    annot(ax, 520, 745, "→ y_hat_others", fs=11)

    box(ax, 40, 620, 260, 66, "Actor embedding  h [N, C]", "focal actor at slot 0",
        fill=C_LAV, stroke=C_LAV_S, fs=13, name="actor_emb")
    arrow(ax, (170, 686), (170, 718), label="actor->neighbour")

    box(ax, 40, 520, 260, 60, "Mode query embedding",
        "K = 20 learnable + focal h_0", fill=C_BLUE, stroke=C_BLUE_S, fs=12.5,
        name="mode_q_emb")
    arrow(ax, (110, 620), (110, 582), label="actor->mode_q_emb")

    glyph_row(ax, 40, 480, 12, [C_LAV] * 10, name="mode_q_row")
    annot(ax, 40, 462, "mode query  Q_m [K, C]", fs=11)
    arrow(ax, (110, 520), (110, 494), label="mode_q_emb->mode_q_row")

    box(ax, 40, 380, 260, 60, "Future time embedding",
        "MLP over t = 0.4 … 4.8 s", fill=C_BLUE, stroke=C_BLUE_S, fs=12.5,
        name="time_emb")

    glyph_row(ax, 40, 330, 12, [C_LAV] * 10, name="state_q_row")
    annot(ax, 40, 312, "state query  Q_s [T_f, C]", fs=11)
    arrow(ax, (110, 380), (110, 344), label="time_emb->state_q_row")

    # ---------------- 上分支：Mode Localization ----------------
    panel(ax, 560, 430, 620, 240, "Mode Localization Module", name="panel_mode")
    glyph_row(ax, 590, 600, 11, [C_LAV] * 6, name="ctx_row_a")
    annot(ax, 668, 605, "scene context  h", fs=10.5)
    box(ax, 580, 500, 150, 60, "Cross-Attention", None, fill=C_GRAY,
        stroke=C_GRAY_S, fs=12, name="mode_cross")
    box(ax, 750, 500, 150, 60, "Self-Attention", None, fill=C_GRAY,
        stroke=C_GRAY_S, fs=12, name="mode_self")
    box(ax, 920, 500, 150, 60, "Mode head", None, fill=C_BLUE,
        stroke=C_BLUE_S, fs=12, name="mode_head")
    arrow(ax, (623, 600), (623, 562), label="ctx_a->mode_cross")
    arrow(ax, (730, 530), (748, 530), label="mode_cross->mode_self")
    arrow(ax, (900, 530), (918, 530), label="mode_self->mode_head")
    fan_box(ax, 920, 435, 130, 55, name="mode_out", n=8, hl=(1,))
    arrow(ax, (995, 500), (995, 492), label="mode_head->mode_out")
    annot(ax, 740, 462, "y_hat [K, T_f, 2]    π [K]    scale", ha="center", fs=11)
    poly_arrow(ax, [(160, 486), (540, 486), (540, 530), (578, 530)],
               label="mode_q_row->mode_cross")

    # ---------------- 下分支：State Consistency ----------------
    panel(ax, 560, 130, 620, 200, "State Consistency Module", name="panel_state")
    glyph_row(ax, 590, 280, 11, [C_LAV] * 6, name="ctx_row_b")
    annot(ax, 668, 285, "scene context  h", fs=10.5)
    box(ax, 580, 190, 150, 60, "Cross-Attention", None, fill=C_GRAY,
        stroke=C_GRAY_S, fs=12, name="state_cross")
    box(ax, 750, 190, 150, 60, "Bidirectional", "Mamba", fill=C_WHITE,
        stroke=INK, fs=12, name="state_mamba")
    box(ax, 920, 190, 150, 60, "State head", None, fill=C_BLUE,
        stroke=C_BLUE_S, fs=12, name="state_head")
    arrow(ax, (623, 280), (623, 252), label="ctx_b->state_cross")
    arrow(ax, (730, 220), (748, 220), label="state_cross->state_mamba")
    arrow(ax, (900, 220), (918, 220), label="state_mamba->state_head")
    glyph_row(ax, 962, 150, 11, [C_BLUE] * 6, name="dense_pred")
    arrow(ax, (995, 190), (995, 163), label="state_head->dense_pred")
    annot(ax, 760, 156, "dense prediction [T_f, 2]", ha="center", fs=11)
    poly_arrow(ax, [(160, 336), (540, 336), (540, 220), (578, 220)],
               label="state_q_row->state_cross")

    # ---------------- 汇合：联合 query ----------------
    glyph_grid(ax, 1200, 340, 11,
               [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]], axis_labels=False, name="q_h")
    annot(ax, 1240, 400, "Q_h = Q_m ⊕ Q_s", ha="center", fs=11)
    poly_arrow(ax, [(1050, 462), (1165, 462), (1165, 375), (1198, 375)],
               label="mode_out->q_h")
    poly_arrow(ax, [(1070, 220), (1180, 220), (1180, 362), (1198, 362)],
               label="state_head->q_h")

    # ---------------- 融合链（横向） ----------------
    panel(ax, 1340, 140, 520, 500, "Hybrid Coupling Module", name="panel_hybrid")
    box(ax, 1360, 470, 140, 60, "Cross-Attention", None, fill=C_GRAY,
        stroke=C_GRAY_S, fs=12, name="h_cross")
    box(ax, 1520, 470, 140, 60, "Self-Attention", None, fill=C_GRAY,
        stroke=C_GRAY_S, fs=12, name="h_self")
    box(ax, 1680, 470, 160, 60, "Mode\nSelf-Attention", None, fill=C_GRAY,
        stroke=C_GRAY_S, fs=12, name="h_modeself")
    arrow(ax, (1500, 500), (1518, 500), label="h_cross->h_self")
    arrow(ax, (1660, 500), (1678, 500), label="h_self->h_modeself")
    box(ax, 1360, 350, 200, 60, "Bidirectional Mamba", None, fill=C_WHITE,
        stroke=INK, fs=12, name="h_mamba")
    box(ax, 1600, 350, 240, 60, "Coupled trajectory head", None, fill=C_BLUE,
        stroke=C_BLUE_S, fs=12, name="h_head")
    arrow(ax, (1560, 380), (1598, 380), label="h_mamba->h_head")
    poly_arrow(ax, [(1760, 470), (1760, 430), (1460, 430), (1460, 412)],
               label="h_modeself->h_mamba")
    poly_arrow(ax, [(1257, 362), (1320, 362), (1320, 500), (1358, 500)],
               label="q_h->h_cross")
    fan_box(ax, 1600, 200, 200, 100, name="h_out", n=20, hl=(2,))
    arrow(ax, (1720, 350), (1720, 302), label="h_head->h_out")
    annot(ax, 1400, 175, "y_hat_new [K, T_f, 2] + π_new + scale_new", fs=11)
    annot(ax, 1400, 152, "K = 20 multimodal futures", fs=11)

    # ---------------- 损失与图例 ----------------
    box(ax, 840, 30, 700, 60, LOSS_TEXT, None, fill=C_WHITE, stroke=INK,
        fs=12.5, name="loss_box")
    legend_strip(ax, 40, 26,
                 ["N agents", "T_f future steps", "K modes", "C channels",
                  "⊕ element-wise add"])

    save(fig, "gast_pf_decoder_framework")
    return audit("decoder")


# ----------------------------------------------------------------------------
# 图 3：整体框架（编码器 + 解码器）
# ----------------------------------------------------------------------------
def figure_overall():
    ELEMS.clear()
    ARROWS.clear()
    W, H = 2320, 840
    fig, ax = new_canvas(W, H)

    # ================= 编码器：照参考图版式（mask 支路 + 横向主链）=================
    panel(ax, 60, 180, 1300, 560, "GAST-PF Encoder", name="panel_enc")

    # --- mask 支路：mask 网格 -> mask 列 -> GSM Module -> ghost token ---
    mat_mask = [[0, 1, 0, 0, 1],
                [0, 0, 0, 1, 1],
                [1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 0, 0, 0]]
    glyph_grid(ax, 100, 600, 15, mat_mask,
               highlight=[(0, 3, 2, 2, C_ORANGE_S)], name="mask")
    caption(ax, 137, 580, "validity mask  m [N, T]", fs=12, max_width=230)

    glyph_col(ax, 215, 600, 15, [C_WHITE, C_BLACK, C_WHITE, C_WHITE, C_BLACK],
              name="mask_col")
    ax.add_patch(FancyBboxPatch((211, 596), 23, 83,
                                boxstyle="round,pad=0,rounding_size=4",
                                fill=False, edgecolor=C_CYAN, lw=2.0, zorder=6))
    arrow(ax, (175, 637), (213, 637), label="mask->mask_col")
    box(ax, 265, 605, 175, 60, "GSM Module", "ghost token pooling",
        fill=C_ORANGE, stroke=C_ORANGE_S, fs=12.5, name="gsm_module")
    arrow(ax, (230, 637), (263, 637), label="mask_col->gsm_module")
    poly_arrow(ax, [(340, 605), (340, 500), (472, 500), (472, 478)],
               label="gsm_module->ghost")

    # --- 输入：不完整历史（叠帧）-> agent embeddings ---
    for dx in (16, 8):
        ax.add_patch(FancyBboxPatch((100 + dx, 400 + dx), 210, 62,
                                    boxstyle="round,pad=0,rounding_size=9",
                                    facecolor=C_WHITE, edgecolor=C_BLUE_S,
                                    lw=1.2, zorder=1))
    box(ax, 100, 400, 210, 62, "Incomplete history", "8 frames × 0.4 s",
        fill=C_BLUE, stroke=C_BLUE_S, fs=12, name="history")
    arrow(ax, (310, 431), (348, 431), label="history->agent_emb")

    mat_emb = [[0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0]]
    glyph_grid(ax, 350, 393, 15, mat_emb, name="agent_emb")
    caption(ax, 387, 375, "Agent embeddings", fs=12, max_width=180)
    arrow(ax, (425, 431), (463, 431), label="agent_emb->tok_a")

    # --- token 栈（ghost token 置顶）-> GSM 空间注意力 -> token 栈 ---
    glyph_col(ax, 465, 386, 15, [C_LAV] * 5 + [C_ORANGE], name="tok_a")
    annot(ax, 458, 490, "ghost token", ha="right", fs=10.5)
    arrow(ax, (480, 431), (513, 431), label="tok_a->gsm_spatial")
    box(ax, 515, 366, 130, 130, "GSM Spatial\nAttention", None,
        fill=C_GRAY, stroke=C_GRAY_S, fs=12.5, name="gsm_spatial")
    arrow(ax, (645, 431), (678, 431), label="gsm_spatial->tok_b")
    glyph_col(ax, 680, 386, 15, [C_LAV] * 5 + [C_ORANGE], name="tok_b")
    arrow(ax, (695, 431), (728, 431), label="tok_b->spa_emb")

    mat_spa = [[0, 0, 0, 0, 0],
               [0, 0, 1, 1, 0],
               [0, 0, 0, 0, 0],
               [0, 1, 1, 0, 0],
               [0, 0, 0, 0, 0]]
    glyph_grid(ax, 730, 393, 15, mat_spa, name="spa_emb")
    caption(ax, 767, 375, "Spatial embeddings", fs=12, max_width=200)

    # --- 缺失间隔 -> 衰减门 -> ⊙ ---
    glyph_row(ax, 807, 720, 15, [C_WHITE, C_WHITE, C_BLACK, C_WHITE, C_WHITE,
                                 C_WHITE, C_BLACK, C_WHITE], name="delta_row")
    annot(ax, 940, 727, "missing-run delta  δ", fs=11)
    box(ax, 753, 630, 220, 56, "Temporal Decay", "γ = exp(−ReLU(W δ))",
        fill=C_TEAL, stroke=C_TEAL_S, tc=C_WHITE, sub_tc="#EAF7F9", fs=12.5,
        name="temporal_decay")
    arrow(ax, (863, 720), (863, 688), label="delta->decay")
    glyph_row(ax, 807, 600, 14, [C_TEAL] * 8, name="gamma_row")
    arrow(ax, (863, 630), (863, 616), label="decay->gamma")

    mul_node(ax, 863, 431, r=20, name="mul")
    arrow(ax, (805, 431), (841, 431), label="spa_emb->mul")
    arrow(ax, (863, 600), (863, 453), label="gamma->mul")

    # --- 双向时序编码 + 末步读出 ---
    stacked_block(ax, 920, 350, 110, 165, "Bidirectional\nTemporal Mamba", n=4,
                  offset=10, fs=12, name="mamba_stack")
    annot(ax, 995, 330, "4 blocks per direction", ha="center", fs=11)
    arrow(ax, (883, 431), (918, 431), label="mul->mamba")
    box(ax, 1080, 400, 150, 62, "Last-step\nreadout", None, fill=C_WHITE,
        stroke=INK, fs=12, name="readout")
    arrow(ax, (1062, 431), (1078, 431), label="mamba->readout")

    mat_out = [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [1, 1, 1, 1, 1],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]
    glyph_grid(ax, 1250, 393, 15, mat_out, highlight=[(2, 0, 1, 5, C_HL)],
               name="h_out")
    caption(ax, 1287, 375, "Actor embedding", fs=12, max_width=200)
    arrow(ax, (1230, 431), (1248, 431), label="readout->h_out")

    # mask -> δ 跳连（沿编码器上方）
    poly_arrow(ax, [(222, 675), (290, 675), (290, 752), (863, 752), (863, 737)],
               color=MUTED, lw=1.2, ms=11, ls=(0, (5, 3)), label="mask->delta")
    annot(ax, 76, 200, "mechanisms borrowed from Sports-Traj", fs=10.5)

    # ================= 解码器：双分支 fork → 联合 query → 融合 =================
    panel(ax, 1450, 120, 660, 480, "Structured Future Decoder", name="panel_dec")
    box(ax, 1470, 470, 230, 54, "Mode localization", None, fill=C_GRAY,
        stroke=C_GRAY_S, fs=12, name="d_mode")
    box(ax, 1470, 230, 230, 54, "State consistency", "dense + Mamba",
        fill=C_GRAY, stroke=C_GRAY_S, fs=12, name="d_state")
    glyph_grid(ax, 1730, 342, 14,
               [[0] * 6 for _ in range(5)], axis_labels=False, name="q_h")
    annot(ax, 1730, 322, "Q_h = Q_m ⊕ Q_s", ha="left", fs=10)
    box(ax, 1850, 400, 230, 54, "Hybrid coupling", "mode ⊕ state",
        fill=C_GRAY, stroke=C_GRAY_S, fs=12, name="d_hybrid")
    box(ax, 1730, 230, 320, 54,
        "Predicted trajectories\n+ mode probabilities", None,
        fill=C_WHITE, stroke=INK, fs=11.5, name="d_out")

    # 编码器输出分叉：上入 Mode 支、下入 State 支
    poly_arrow(ax, [(1325, 431), (1420, 431), (1420, 497), (1468, 497)],
               label="h->mode_branch")
    poly_arrow(ax, [(1325, 431), (1420, 431), (1420, 257), (1468, 257)],
               label="h->state_branch")
    poly_arrow(ax, [(1700, 497), (1712, 497), (1712, 400), (1728, 400)],
               label="mode->q_h")
    poly_arrow(ax, [(1700, 257), (1720, 257), (1720, 355), (1728, 355)],
               label="state->q_h")
    poly_arrow(ax, [(1816, 377), (1832, 377), (1832, 427), (1848, 427)],
               label="q_h->hybrid")
    poly_arrow(ax, [(1965, 400), (1965, 286)], label="hybrid->out")
    arrow(ax, (2052, 257), (2148, 257), label="out->fan")
    annot(ax, 1466, 142, "structured query decoupling from DeMo", fs=10.5)

    fan_box(ax, 2150, 200, 120, 110, name="fan", n=20, hl=(1,))
    caption(ax, 2210, 180, "K = 20 futures", fs=11.5, max_width=200)
    caption(ax, 2210, 162, "π [K]", fs=11.5, max_width=200)

    box(ax, 60, 30, 1300, 56, LOSS_TEXT, None, fill=C_WHITE, stroke=INK,
        fs=12.5, name="loss_box")
    box(ax, 1450, 30, 660, 56, "TrajGap-Bench · Mixed · 8 → 12 frames · K = 20",
        "minADE20 / minFDE20 / MR", fill=C_WHITE, stroke=INK, fs=12,
        name="bench_box")
    legend_strip(ax, 60, 780,
                 ["N agents", "T history", "K modes",
                  "⊙ element-wise multiply", "⊕ element-wise add"])

    save(fig, "gast_pf_overall_framework")
    return audit("overall")


def figure_model_framework():
    ELEMS.clear()
    ARROWS.clear()
    W, H = 3400, 1320
    fig, ax = new_canvas(W, H)

    ax.text(W / 2, 1285,
            "GAST-PF: missing-aware spatiotemporal encoding with decoupled-query future decoding",
            ha="center", va="center", fontsize=23, fontweight="bold", color=INK)
    legend_strip(ax, 70, 1215,
                 ["N agents", "T = 8 history frames", "T_f = 12 future steps",
                  "K = 20 modes", "C = 128 channels", "⊙ multiply", "⊕ add"],
                 fs=12, gap=32)

    panel(ax, 40, 180, 350, 990, "Observed trajectories", name="panel_input")
    for dx in (18, 9):
        ax.add_patch(FancyBboxPatch((100 + dx, 885 + dx), 220, 78,
                                    boxstyle="round,pad=0,rounding_size=9",
                                    facecolor=C_WHITE, edgecolor=C_BLUE_S,
                                    lw=1.2, zorder=1))
    box(ax, 100, 885, 220, 78, "Complete trajectory",
        "focal + neighbours", fill=C_BLUE, stroke=C_BLUE_S, fs=13,
        sub_fs=11.5, name="input_traj")
    glyph_grid(ax, 110, 650, 20,
               [[0, 1, 0, 0, 1, 0, 0, 1],
                [1, 0, 0, 1, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 1],
                [1, 0, 0, 1, 0, 0, 1, 0]],
               highlight=[(0, 2, 2, 2, C_ORANGE_S)], name="input_mask")
    caption(ax, 190, 610, "Binary validity mask  m [B, N, T]",
            fs=12, max_width=300)
    box(ax, 80, 420, 270, 90, "Feature construction",
        "x = [Δp, Δv, m]  [B, N, T, 4]", fill=C_ORANGE,
        stroke=C_ORANGE_S, fs=13, sub_fs=10.5, name="input_features")
    box(ax, 80, 250, 270, 86, "Target convention",
        "actor slot 0 = focal; neighbours = 1:N", fill=C_WHITE,
        stroke=INK, fs=12, sub_fs=10.2, name="input_target")
    poly_arrow(ax, [(210, 885), (210, 850), (210, 850), (210, 512)],
               label="trajectory->features")
    poly_arrow(ax, [(210, 650), (210, 590), (210, 590), (210, 512)],
               label="mask->features")
    poly_arrow(ax, [(350, 465), (430, 465), (430, 826), (498, 826)],
               label="features->encoder")

    panel(ax, 430, 180, 1300, 990, "GAST-PF encoder", name="panel_encoder")
    box(ax, 500, 785, 215, 82, "History features",
        "8 × 0.4 s  [B, N, T, 4]", fill=C_BLUE, stroke=C_BLUE_S,
        fs=12.5, sub_fs=10.8, name="enc_history")
    glyph_grid(ax, 750, 790, 14,
               [[0, 0, 1, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]],
               axis_labels=False, highlight=[(1, 2, 1, 1, C_LAV)],
               name="enc_tokens")
    caption(ax, 806, 765, "Agent tokens", fs=11.5)
    box(ax, 870, 785, 220, 82, "Patch embedding",
        "Linear → LayerNorm → LeakyReLU", fill=C_GRAY, stroke=C_GRAY_S,
        fs=12, sub_fs=9.8, name="enc_patch")
    box(ax, 1140, 785, 260, 82, "GSM spatial attention",
        "ghost-token pooling + agent MHA", fill=C_ORANGE,
        stroke=C_ORANGE_S, fs=12.5, sub_fs=10.2, name="enc_gsm")
    glyph_grid(ax, 1450, 790, 14,
               [[0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]],
               axis_labels=False, name="enc_spatial")
    caption(ax, 1510, 758, "Spatial embeddings z",
            fs=11.5, max_width=150)
    annot(ax, 1510, 742, "[B, T, N, C]", fs=10.0, ha="center")
    arrow(ax, (715, 826), (748, 826), label="history->tokens")
    arrow(ax, (836, 826), (868, 826), label="tokens->patch")
    arrow(ax, (1092, 826), (1138, 826), label="patch->gsm")
    arrow(ax, (1402, 826), (1448, 826), label="gsm->spatial")

    box(ax, 520, 975, 225, 64, "Validity mask m", "[B, N, T]",
        fill=C_ORANGE, stroke=C_ORANGE_S, fs=12, sub_fs=10.5,
        name="enc_mask")
    box(ax, 820, 975, 245, 64, "Missing-run deltas",
        "δᶠ and δᵇ from m", fill=C_ORANGE, stroke=C_ORANGE_S,
        fs=12, sub_fs=10.5, name="enc_delta")
    box(ax, 1140, 975, 260, 64, "Temporal Decay",
        "γ = exp(−ReLU(Wδ))", fill=C_TEAL, stroke=C_TEAL_S,
        tc=C_WHITE, sub_tc="#EAF7F9", fs=12.5, sub_fs=10.5,
        name="enc_decay")
    glyph_row(ax, 860, 1080, 15,
              [C_WHITE, C_WHITE, C_BLACK, C_WHITE, C_WHITE, C_BLACK, C_WHITE, C_WHITE],
              name="enc_delta_row")
    annot(ax, 990, 1100, "forward / reversed missing-run deltas δ", fs=10.5,
          ha="center")
    glyph_row(ax, 1210, 900, 15, [C_TEAL] * 8, name="enc_gamma")
    annot(ax, 1270, 886, "decay gate γ", fs=10.5, ha="center")
    poly_arrow(ax, [(632, 975), (632, 930), (1270, 930), (1270, 900)],
               color=MUTED, lw=1.2, ms=11, ls=(0, (5, 3)),
               label="mask->delta")
    arrow(ax, (745, 1007), (818, 1007), label="mask->delta_box")
    arrow(ax, (1067, 1007), (1138, 1007), label="delta->decay")
    arrow(ax, (1270, 975), (1270, 900), label="decay->gamma")

    mul_node(ax, 1510, 690, r=20, name="enc_mul")
    arrow(ax, (1510, 790), (1510, 712), label="spatial->mul")
    arrow(ax, (1270, 900), (1510, 712), label="gamma->mul")
    stacked_block(ax, 1580, 585, 110, 180, "Bidirectional\nTemporal Mamba",
                  n=4, offset=10, fs=11.5, name="enc_mamba")
    annot(ax, 1640, 560, "4 blocks per direction; forward + reversed, summed",
          fs=10.5, ha="center", max_width=270)
    arrow(ax, (1532, 690), (1578, 690), label="mul->mamba")
    box(ax, 1490, 385, 200, 74, "Last-step readout",
        "take y[:, −1]", fill=C_WHITE, stroke=INK, fs=12.5,
        sub_fs=10.5, name="enc_readout")
    arrow(ax, (1640, 585), (1590, 461), label="mamba->readout")
    glyph_grid(ax, 1490, 245, 15,
               [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]],
               axis_labels=False, highlight=[(2, 0, 1, 5, C_HL)],
               name="enc_actor")
    caption(ax, 1548, 220, "Actor embeddings h [B, N, C]", fs=11.5,
            max_width=250)
    arrow(ax, (1590, 385), (1548, 320), label="readout->actor")
    annot(ax, 490, 220,
          "Default: gast_anchor_geo = false; padding actors are masked throughout.",
          fs=10.5)
    poly_arrow(ax, [(1565, 280), (1730, 280), (1730, 942), (1878, 942)],
               label="encoder->decoder")

    panel(ax, 1820, 180, 1480, 990,
          "Structured future decoder with decoupled queries", name="panel_decoder")
    box(ax, 1880, 905, 250, 74, "Actor embeddings h",
        "[B, N, C]", fill=C_LAV, stroke=C_LAV_S, fs=12.5,
        sub_fs=10.8, name="dec_h")
    box(ax, 1880, 775, 250, 74, "Mode query Qₘ",
        "h₀ + learnable mode embeddings", fill=C_BLUE, stroke=C_BLUE_S,
        fs=12.5, sub_fs=10.2, name="dec_qm")
    box(ax, 1880, 645, 250, 74, "Future time embedding",
        "MLP(t),  t = 0.4 … 4.8 s", fill=C_BLUE, stroke=C_BLUE_S,
        fs=12, sub_fs=10.0, name="dec_time")
    box(ax, 1880, 515, 250, 74, "State query Qₛ",
        "[B, T_f, C]", fill=C_BLUE, stroke=C_BLUE_S, fs=12.5,
        sub_fs=10.8, name="dec_qs")
    box(ax, 1880, 1015, 315, 64, "Neighbour prediction head",
        "x_encoder[:, 1:N] → ŷ_others", fill=C_WHITE, stroke=INK,
        fs=11.5, sub_fs=10.2, name="dec_neighbour")
    arrow(ax, (2005, 905), (2005, 850), label="h->mode_init")
    arrow(ax, (2005, 775), (2005, 719), label="mode_init->qm")
    arrow(ax, (2005, 645), (2005, 589), label="time->qs")
    poly_arrow(ax, [(2005, 905), (2005, 1095), (2195, 1095), (2195, 1079)],
               label="h->neighbour_head")

    panel(ax, 2200, 720, 510, 365, "Mode localization module", name="panel_mode")
    glyph_row(ax, 2240, 1015, 13, [C_LAV] * 8, name="mode_ctx")
    annot(ax, 2355, 1022, "scene context h", fs=10.5)
    box(ax, 2230, 900, 130, 62, "Cross-Attn", None, fill=C_GRAY,
        stroke=C_GRAY_S, fs=11.5, name="mode_cross")
    box(ax, 2385, 900, 130, 62, "Self-Attn", None, fill=C_GRAY,
        stroke=C_GRAY_S, fs=11.5, name="mode_self")
    box(ax, 2540, 900, 130, 62, "Mode head", None, fill=C_BLUE,
        stroke=C_BLUE_S, fs=11.5, name="mode_head")
    arrow(ax, (2362, 931), (2383, 931), label="mode_cross->self")
    arrow(ax, (2517, 931), (2538, 931), label="mode_self->head")
    glyph_row(ax, 2560, 800, 12, [C_LAV] * 8, name="mode_out")
    arrow(ax, (2605, 900), (2605, 814), label="mode_head->out")
    annot(ax, 2390, 770, "ŷ [B, K, T_f, 2]  +  π  +  scale", fs=10.5,
          ha="center", max_width=360)
    poly_arrow(ax, [(2005, 812), (2160, 812), (2160, 931), (2228, 931)],
               label="qm->mode_module")

    panel(ax, 2200, 340, 510, 320, "State consistency module", name="panel_state2")
    glyph_row(ax, 2240, 590, 13, [C_LAV] * 8, name="state_ctx")
    annot(ax, 2355, 597, "scene context h", fs=10.5)
    box(ax, 2230, 475, 130, 62, "Cross-Attn", None, fill=C_GRAY,
        stroke=C_GRAY_S, fs=11.5, name="state_cross")
    box(ax, 2385, 475, 130, 62, "Bi-Mamba", None, fill=C_WHITE,
        stroke=INK, fs=11.5, name="state_mamba")
    box(ax, 2540, 475, 130, 62, "State head", None, fill=C_BLUE,
        stroke=C_BLUE_S, fs=11.5, name="state_head")
    arrow(ax, (2362, 506), (2383, 506), label="state_cross->mamba")
    arrow(ax, (2517, 506), (2538, 506), label="state_mamba->head")
    glyph_row(ax, 2560, 385, 12, [C_BLUE] * 8, name="dense_out")
    arrow(ax, (2605, 475), (2605, 399), label="state_head->dense")
    annot(ax, 2390, 355, "dense prediction [B, T_f, 2]", fs=10.5,
          ha="center")
    poly_arrow(ax, [(2005, 515), (2160, 515), (2160, 506), (2228, 506)],
               label="qs->state_module")

    glyph_grid(ax, 2760, 620, 15,
               [[0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]], axis_labels=False, name="dec_qh")
    annot(ax, 2805, 595, "Qₕ = Qₘ ⊕ Qₛ", fs=11, ha="center")
    poly_arrow(ax, [(2605, 800), (2685, 800), (2685, 665), (2758, 665)],
               label="mode->joint_query")
    poly_arrow(ax, [(2605, 385), (2685, 385), (2685, 635), (2758, 635)],
               label="state->joint_query")

    panel(ax, 2940, 310, 285, 650, "Hybrid coupling", name="panel_hybrid2")
    box(ax, 2965, 820, 235, 58, "Cross-Attn", None, fill=C_GRAY,
        stroke=C_GRAY_S, fs=11.5, name="hyb_cross")
    box(ax, 2965, 735, 235, 58, "Self-Attn", None, fill=C_GRAY,
        stroke=C_GRAY_S, fs=11.5, name="hyb_self")
    box(ax, 2965, 650, 235, 58, "Mode Self-Attn", None, fill=C_GRAY,
        stroke=C_GRAY_S, fs=11.5, name="hyb_mode_self")
    box(ax, 2965, 525, 235, 58, "Bidirectional Mamba", None, fill=C_WHITE,
        stroke=INK, fs=11.5, name="hyb_mamba")
    box(ax, 2965, 410, 235, 58, "Coupled trajectory head", None,
        fill=C_BLUE, stroke=C_BLUE_S, fs=11.5, name="hyb_head")
    arrow(ax, (3082, 820), (3082, 795), label="hyb_cross->self")
    arrow(ax, (3082, 735), (3082, 710), label="hyb_self->mode_self")
    arrow(ax, (3082, 650), (3082, 585), label="hyb_mode_self->mamba")
    arrow(ax, (3082, 525), (3082, 470), label="hyb_mamba->head")
    poly_arrow(ax, [(2820, 650), (2880, 650), (2880, 849), (2963, 849)],
               label="joint_query->hybrid")
    fan_box(ax, 3060, 235, 170, 120, name="dec_final", n=20, hl=(2,))
    arrow(ax, (3082, 410), (3145, 357), label="head->final")
    caption(ax, 3145, 213, "ŷ_new [B, K, T_f, 2]", fs=11.5,
            max_width=220)
    caption(ax, 3145, 188, "+ π_new + scale_new", fs=11, max_width=220)
    annot(ax, 2960, 335, "mode × state interaction", fs=10.5)

    box(ax, 60, 48, 3160, 72,
        "Training objectives: neighbour regression + dense regression + multimodal regression / classification + Laplace NLL",
        None, fill=C_WHITE, stroke=INK, fs=13.5, name="model_loss")

    save(fig, "gast_pf_model_framework")
    return audit("model_framework")


if __name__ == "__main__":
    issues = figure_overall() + figure_encoder() + figure_decoder() + figure_model_framework()
    print(f"[audit] total dangling endpoints: {issues}")
