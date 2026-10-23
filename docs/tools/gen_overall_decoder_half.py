#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按编码器风格重绘解码器半图：白底、黑框圆角块、CM 字体、7px 主链箭头 + 4px 次链箭头。
去掉标题，内容放大并垂直居中；最后合成回整图右半部分。
"""
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

SRC = "/home/lbh/.hermes/cache/scratch/edited_overall.png"
OUT = "/home/lbh/GAST-PF/docs/figures/gast_pf_overall_reference_style.png"
FDIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
CMB, CMR, CMMI = FDIR + "/cmb10.ttf", FDIR + "/cmr10.ttf", FDIR + "/cmmi10.ttf"
STIX = FDIR + "/STIXGeneral.ttf"

INK = (26, 26, 26)
MUTED = (85, 100, 115)
WHITE = (255, 255, 255)
GREY = (191, 191, 191)
SALMON = (248, 203, 173)
GREEN = (197, 224, 180)
LEG_Y = (255, 230, 153)
LEG_B = (189, 215, 238)
TCOL = [(46, 117, 182), (110, 160, 215), (157, 195, 230), (244, 177, 131), (255, 217, 102), (255, 242, 204)]

S = 3
W, H = 3384, 670
DY = -62                      # 整体竖直偏移（使内容垂直居中；竖框版内容更高）

img = Image.new("RGB", (W * S, H * S), WHITE)
d = ImageDraw.Draw(img)
ELEMS, TEXTS = [], []


def F(p, s):
    return ImageFont.truetype(p, int(round(s * S)))


def tw(txt, f):
    b = d.textbbox((0, 0), txt, font=f)
    return (b[2] - b[0]) / S


def fit(txt, path, maxw, start=44, minsize=10):
    for s in range(int(start), int(minsize) - 1, -1):
        if tw(txt, F(path, s)) <= maxw:
            return F(path, s)
    return F(path, minsize)


def tc(txt, cx, cy, f, fill=INK, tag=None):
    b = d.textbbox((0, 0), txt, font=f)
    d.text(((cx - (b[0] + b[2]) / 2.0 / S) * S, (cy - (b[1] + b[3]) / 2.0 / S) * S), txt, font=f, fill=fill)
    if tag:
        TEXTS.append((tag, cx - (b[2] - b[0]) / 2.0 / S, cy - (b[3] - b[1]) / 2.0 / S,
                      cx + (b[2] - b[0]) / 2.0 / S, cy + (b[3] - b[1]) / 2.0 / S))


def tl(txt, x, cy, f, fill=INK):
    b = d.textbbox((0, 0), txt, font=f)
    d.text((x * S - b[0], (cy - (b[1] + b[3]) / 2.0 / S) * S), txt, font=f, fill=fill)


def head(p, ang, size=15, color=INK):
    a1, a2 = ang + math.radians(152), ang - math.radians(152)
    d.polygon([(p[0] * S, p[1] * S),
               ((p[0] + size * math.cos(a1)) * S, (p[1] + size * math.sin(a1)) * S),
               ((p[0] + size * math.cos(a2)) * S, (p[1] + size * math.sin(a2)) * S)], fill=color)


def arrow(p0, p1, color=INK, lw=4, size=15):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = max(1e-6, math.hypot(dx, dy))
    ux, uy = dx / L, dy / L
    d.line([(p0[0] * S, p0[1] * S), ((p1[0] - ux * size * 0.9) * S, (p1[1] - uy * size * 0.9) * S)],
           fill=color, width=int(lw * S))
    head(p1, math.atan2(dy, dx), size, color)


def poly_arrow(pts, color=INK, lw=4, size=15):
    for i in range(len(pts) - 2):
        d.line([(pts[i][0] * S, pts[i][1] * S), (pts[i + 1][0] * S, pts[i + 1][1] * S)],
               fill=color, width=int(lw * S))
    arrow(pts[-2], pts[-1], color, lw, size)


def box(x0, y0, x1, y1, lines, fill=GREY, name=None, fs=None):
    d.rounded_rectangle((x0 * S, y0 * S, x1 * S, y1 * S), radius=10 * S, fill=fill,
                        outline=INK, width=int(3 * S))
    ELEMS.append((name or "box", x0, y0, x1, y1))
    n = len(lines)
    fs = fs or min(30, (y1 - y0) / (n * 1.5))
    step = (y1 - y0) / (n + 1)
    for i, ln in enumerate(lines):
        f = fit(ln, CMB, (x1 - x0) - 18, start=fs)
        tc(ln, (x0 + x1) / 2.0, y0 + step * (i + 1), f, tag=name)


def cubes(x0, y0, x1, y1, n1=3, n2=3, name="cubes"):
    n = n1 + n2
    gap = 5
    cw = ((x1 - x0) - gap * (n - 1)) / n
    depth = min(6, cw * 0.3)
    for i in range(n):
        cx0 = x0 + i * (cw + gap)
        col = SALMON if i < n1 else GREEN
        d.polygon([(cx0 * S, (y0 + depth) * S), ((cx0 + depth) * S, y0 * S),
                   ((cx0 + cw + depth) * S, y0 * S), ((cx0 + cw) * S, (y0 + depth) * S)],
                  fill=col, outline=INK, width=int(2 * S))
        d.rectangle((cx0 * S, (y0 + depth) * S, (cx0 + cw) * S, y1 * S), fill=col, outline=INK, width=int(2 * S))
        d.polygon([((cx0 + cw) * S, (y0 + depth) * S), ((cx0 + cw + depth) * S, y0 * S),
                   ((cx0 + cw + depth) * S, (y1 - depth) * S), ((cx0 + cw) * S, y1 * S)],
                  fill=col, outline=INK, width=int(2 * S))
    ELEMS.append((name, x0, y0, x1 + depth, y1))


def heat(x0, y0, x1, y1, cols=6, rows=5, name="tensor"):
    cw, chh = (x1 - x0) / cols, (y1 - y0) / rows
    for i in range(cols):
        for j in range(rows):
            d.rectangle((x0 * S + i * cw * S, y0 * S + j * chh * S,
                         x0 * S + (i + 1) * cw * S, y0 * S + (j + 1) * chh * S),
                        fill=TCOL[i], outline=(210, 214, 220), width=int(1 * S))
    d.rectangle((x0 * S, y0 * S, x1 * S, y1 * S), outline=INK, width=int(2 * S))
    ELEMS.append((name, x0, y0, x1, y1))


def math_w(tokens, size):
    tot = 0
    for kind, s in tokens:
        f = F(CMMI if kind in ("it", "sub") else CMR, size if kind != "sub" else size * 0.72)
        b = d.textbbox((0, 0), s, font=f)
        tot += (b[2] - b[0]) / S
    return tot


def math_str(tokens, cx, cy, size, fill=INK, left=None):
    parts = []
    for kind, s in tokens:
        f = F(CMMI if kind in ("it", "sub") else CMR, size if kind != "sub" else size * 0.72)
        b = d.textbbox((0, 0), s, font=f)
        parts.append((b, f, kind, s))
    total = sum((b[2] - b[0]) / S for b, f, k, s in parts)
    x = (left if left is not None else cx - total / 2.0)
    bb = d.textbbox((0, 0), "B", font=F(CMMI, size))
    base = (bb[1] + bb[3]) / 2.0 / S
    for b, f, kind, s in parts:
        dy = size * 0.26 if kind == "sub" else 0
        d.text((x * S - b[0], (cy - base + dy) * S), s, font=f, fill=fill)
        x += (b[2] - b[0]) / S
    return total


def toks(spec):
    out, i = [], 0
    while i < len(spec):
        c = spec[i]
        if c == "_":
            j = i + 1
            while j < len(spec) and spec[j].isalpha():
                j += 1
            out.append(("sub", spec[i + 1:j])); i = j
        elif c.isalpha():
            out.append(("it", c)); i += 1
        else:
            j = i
            while j < len(spec) and not spec[j].isalpha() and spec[j] != "_":
                j += 1
            out.append(("rm", spec[i:j])); i = j
    return out


def loss_badge(cx, cy, sub, size=30, fill=INK, tag="loss"):
    """按示例图风格画损失标记：只有斜体 L + 下标，不带文字"""
    w = math_str(toks("L_" + sub), cx, cy, size, fill=fill)
    h = size * 1.0
    TEXTS.append((tag, cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0))


def mul_count(cx, cy, n, size=24, fill=MUTED):
    """用 STIX 画 ×，CM 画数字，两者按墨迹中心对齐"""
    fs = F(STIX, size)
    fn = F(CMR, size)
    bs = d.textbbox((0, 0), "\u00d7", font=fs)
    bn = d.textbbox((0, 0), n, font=fn)
    ws, wn = (bs[2] - bs[0]) / S, (bn[2] - bn[0]) / S
    gap = 6
    x0 = cx - (ws + gap + wn) / 2.0
    d.text((x0 * S - bs[0], (cy - (bs[1] + bs[3]) / 2.0 / S) * S), "\u00d7", font=fs, fill=fill)
    d.text(((x0 + ws + gap) * S - bn[0], (cy - (bn[1] + bn[3]) / 2.0 / S) * S), n, font=fn, fill=fill)


def Y(v):
    return v + DY


# ================= 布局 =================
TRUNK = 1880
AX, BX, BW, BH = 1945, 2175, 180, 75
A_C, B_C = 245, 500   # 两支中心拉开，给竖框留间隙

# 主干
d.line([(TRUNK * S, Y(A_C) * S), (TRUNK * S, Y(B_C) * S)], fill=INK, width=int(7 * S))
arrow((TRUNK, Y(A_C)), (AX - 2, Y(A_C)), lw=7, size=18)
arrow((TRUNK, Y(B_C)), (AX - 2, Y(B_C)), lw=7, size=18)

# ---- (a) 支：CrossAttn + SelfAttn 合并为一个模块，只暴露综合名；竖向长框（对齐 Mamba 模块风格）----
box(AX, Y(A_C - 110), AX + 150, Y(A_C + 110), ["Mode", "Localization", "Module"], name="a1")
arrow((2413, Y(A_C - 2)), (2413, Y(A_C - 34)), lw=4, size=13)
loss_badge(2413, Y(A_C - 56), "m", 32, tag="loss_a")

# ---- (b) 支：CrossAttn + BiMamba 合并为一个模块，只暴露综合名；竖向长框 ----
box(AX, Y(B_C - 110), AX + 150, Y(B_C + 110), ["State", "Consistency", "Module"], name="b1")
arrow((2413, Y(B_C + 2)), (2413, Y(B_C + 62)), lw=4, size=13)
loss_badge(2413, Y(B_C + 84), "s", 32, tag="loss_b")

# ---- 两支输出汇合（去掉中间大矩阵张量）----
SX0, SX1 = 2580, 2810          # (c) 模块位置（提前定义，供汇合箭头使用）
MERGE_X = 2470
d.line([((AX + 152) * S, Y(A_C) * S), (MERGE_X * S, Y(A_C) * S)], fill=INK, width=int(7 * S))          # (a) 支输出（框右缘 AX+150）
d.line([((AX + 152) * S, Y(B_C) * S), (MERGE_X * S, Y(B_C) * S)], fill=INK, width=int(7 * S))          # (b) 支输出（框右缘 AX+150）
d.line([(MERGE_X * S, Y(B_C) * S), (MERGE_X * S, Y(A_C) * S)], fill=INK, width=int(7 * S))          # 汇合竖干
arrow((MERGE_X, Y(362)), (SX0 - 2, Y(362)), lw=7, size=18)                 # 合并后进 (c)，落在框垂直中心

# ---- (c) 模块：四级合成一个模块，只暴露综合名；竖向长框（与两支分支、Mamba 同风格）----
SX0, SX1 = 2600, 2750          # 竖框宽 150，中心 x=2675 与汇合/输出线同轴
box(SX0, Y(212), SX1, Y(512), ["Hybrid", "Coupling", "Module"], name="c1")   # 高 300，中心 y=362

# ---- 扇面 + 输出 ----
FX, FY, FL = 2855, 362, 150   # FY = 合并框垂直中心，输出箭头与扇面原点都在此
for k in range(20):
    ang = math.radians(-40 + (k / 19.0) * 80)   # 上下对称扇面
    pts = []
    for s in range(11):
        r = FL * s / 10.0
        bend = math.radians(15) * (s / 10.0) ** 2 * (1 if k % 2 == 0 else -1)
        pts.append(((FX + r * math.cos(ang + bend)) * S, (Y(FY) + r * math.sin(ang + bend)) * S))
    if k == 4:
        d.line(pts, fill=(60, 60, 60), width=int(8 * S), joint="curve")
        d.line(pts, fill=(255, 226, 60), width=int(5 * S), joint="curve")
    else:
        d.line(pts, fill=(176, 190, 205), width=int(2 * S), joint="curve")
d.ellipse(((FX - 5) * S, (Y(FY) - 5) * S, (FX + 5) * S, (Y(FY) + 5) * S), fill=INK)
arrow((SX1 + 2, Y(FY)), (FX - 5, Y(FY)), lw=7, size=16)          # (c) 输出 -> 扇面起点（整框右缘即落点）
tc("K = 20 futures", FX + 75, Y(240), fit("K = 20 futures", CMB, 210), INK, tag="k20")

# ---- 图例 ----
def swatch(x0, y0, x1, y1, col, label, word, name):
    d.rounded_rectangle((x0 * S, Y(y0) * S, x1 * S, Y(y1) * S), radius=6 * S, fill=col,
                        outline=INK, width=int(2 * S))
    ELEMS.append((name, x0, Y(y0), x1, Y(y1)))
    cy = (y0 + y1) / 2.0
    arrow((x1 + 8, Y(cy)), (x1 + 30, Y(cy)), lw=3, size=11)
    math_str(toks(label), x1 + 66, Y(cy), 25)
    tc(word, (x0 + x1) / 2.0, Y(y1 + 18), fit(word, CMR, 130), MUTED, tag=name + "_w")


# ================= 合成 =================
GS = 1.20


def _bbox(arr, xmin=1880):
    m = (abs(arr - 255).max(2) > 12)
    m[:, :xmin] = False
    ys, xs = np.where(m)
    return int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1


# 第一步：只含主内容（图例未画）-> 按主内容垂直居中
a0 = np.asarray(img.resize((W, H), Image.LANCZOS)).astype(int)
bx0, bx1, by0, by1 = _bbox(a0)
px = 1885 + (3384 - 1885 - int((bx1 - bx0) * GS)) // 2
py = int(round(335 - (by1 - by0) / 2.0 * GS))
print("主内容 bbox:", bx0, by0, bx1, by1, " -> paste at", (px, py))

# 第二步：图例钉在最终坐标（外框左上 3145,514），反算回布局坐标
LEG_TARGET_X, LEG_TARGET_Y = 3145, 514
lx = int(round(bx0 + (LEG_TARGET_X - px) / GS))
ly = by0 + (LEG_TARGET_Y - py) / GS - DY   # swatch() 内部会再加 DY，这里先抵消
swatch(lx, ly, lx + 80, ly + 24, LEG_Y, "L_reg", "Trajectories", "leg1")
swatch(lx, ly + 60, lx + 80, ly + 84, LEG_B, "L_cls", "Probabilities", "leg2")
print("图例布局坐标: (%d, %.1f)" % (lx, ly))

# 第三步：正式合成（裁剪框要包住图例）
big = img.resize((W, H), Image.LANCZOS)
big.save("/home/lbh/.hermes/cache/scratch/_layout_debug.png")
bx0, bx1, by0, by1 = _bbox(np.asarray(big).astype(int))
print("含图例 bbox:", bx0, by0, bx1, by1)
crop = big.crop((bx0, by0, bx1, by1))
nw, nh = int(crop.width * GS), int(crop.height * GS)
crop = crop.resize((nw, nh), Image.LANCZOS)

fig = Image.open(SRC).convert("RGB")
d3 = ImageDraw.Draw(fig)
d3.rectangle((1862, 0, 3384, 670), fill=WHITE)
fig.paste(crop, (px, py))   # px/py 已在上面按主内容居中算好，勿覆盖

trunk_x = px + int((TRUNK - bx0) * GS)   # 用真实主干坐标（此前写死 1920 导致线越过主干 47px）
# 输入箭头：与编码器同一条线，统一 7px
# 编码器输出 -> 主干：这里是"分支"而不是终止箭头，所以不画箭头头，线直接接上主干
d3.line([(1795, 382), (trunk_x + 3, 382)], fill=INK, width=7)
fig.save(OUT)
print("saved", OUT, fig.size, "scale=%.2f  content=%dx%d  at (%d,%d)" % (GS, nw, nh, px, py))

# 内容 bbox（解码区）
a = np.asarray(fig).astype(int)
m = (abs(a[:, 1900:] - 255).max(2) > 12)
ys, xs = np.where(m)
print("解码区内容 bbox: x %d..%d  y %d..%d  (垂直中心 %.1f, 画布中心 335)"
      % (1900 + xs.min(), 1900 + xs.max(), ys.min(), ys.max(), (ys.min() + ys.max()) / 2.0))

# 重叠自检（图元 vs 图元、文字 vs 文字）
bad = 0
for i in range(len(ELEMS)):
    for j in range(i + 1, len(ELEMS)):
        n1, ax0, ay0, ax1, ay1 = ELEMS[i]
        n2, bx0, by0, bx1, by1 = ELEMS[j]
        if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
            print("  [overlap]", n1, n2); bad += 1
for i in range(len(TEXTS)):
    for j in range(i + 1, len(TEXTS)):
        n1, ax0, ay0, ax1, ay1 = TEXTS[i]
        n2, bx0, by0, bx1, by1 = TEXTS[j]
        if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
            print("  [text overlap]", n1, n2); bad += 1
print("overlaps:", bad)
