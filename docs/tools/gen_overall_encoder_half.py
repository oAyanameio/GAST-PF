#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在原图（UniTraj 编码器 + DeMo 解码器拼接图）上做像素级修正，使其表达 GAST-PF。
v2：修正缺字符（×/→/· 用 STIX 绘制）、放大输入框字号、损失名避让、补 readout 并接通解码器主干。
"""
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
from scipy import ndimage

SRC = "/home/lbh/GAST-PF/docs/figures/reference_input.png"
OUT = "/home/lbh/.hermes/cache/scratch/edited_overall.png"   # 中间产物（可重新生成）
FDIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
CMB, CMR, CMMI = FDIR + "/cmb10.ttf", FDIR + "/cmr10.ttf", FDIR + "/cmmi10.ttf"
STIX, STIXB = FDIR + "/STIXGeneral.ttf", FDIR + "/STIXGeneralBol.ttf"

INK = (26, 26, 26)
MUTED = (85, 100, 115)
WHITE = (255, 255, 255)
PANEL = (242, 247, 252)
CONT = (228, 235, 247)
INBOX = (218, 233, 253)
ORANGE = (228, 126, 56)
BAND = (222, 235, 247)

im = Image.open(SRC).convert("RGB")
d = ImageDraw.Draw(im)
A = np.asarray(im).astype(int)
log = []


def F(p, s):
    return ImageFont.truetype(p, s)


def tw(txt, f):
    b = d.textbbox((0, 0), txt, font=f)
    return b[2] - b[0]


def fit(txt, path, maxw, start=44, minsize=9):
    for s in range(start, minsize - 1, -1):
        if tw(txt, F(path, s)) <= maxw:
            return F(path, s)
    return F(path, minsize)


def tc(txt, cx, cy, f, fill=INK):
    b = d.textbbox((0, 0), txt, font=f)
    d.text((cx - (b[0] + b[2]) / 2.0, cy - (b[1] + b[3]) / 2.0), txt, font=f, fill=fill)


def tl(txt, x, cy, f, fill=INK):
    b = d.textbbox((0, 0), txt, font=f)
    d.text((x - b[0], cy - (b[1] + b[3]) / 2.0), txt, font=f, fill=fill)


def ink_left(txt, x, cy, f, fill=INK):
    """按墨迹左缘 x、墨迹竖直中心 cy 落字（避免空前缀时基线算成 0 而下沉）"""
    b = d.textbbox((0, 0), txt, font=f)
    d.text((x - b[0], cy - (b[1] + b[3]) / 2.0), txt, font=f, fill=fill)


def sym(ch, cx, cy, size, bold=False, fill=INK):
    """用 STIX 画符号（CM 系字体没有 Unicode 映射的 ×/→/·）"""
    f = F(STIXB if bold else STIX, size)
    b = d.textbbox((0, 0), ch, font=f)
    d.text((cx - (b[0] + b[2]) / 2.0, cy - (b[1] + b[3]) / 2.0), ch, font=f, fill=fill)


def label_sym(prefix, ch, suffix, cx, cy, path, size, fill=INK, bold_sym=False):
    """混合绘制：文字用 CM，符号用 STIX，整体按给定中心对齐"""
    fl, fs = F(path, size), F(STIXB if bold_sym else STIX, size)
    wl, ws = tw(prefix, fl), tw(ch, fs)
    w2 = tw(suffix, fl)
    total = wl + ws + w2
    x = cx - total / 2.0
    bl = d.textbbox((0, 0), prefix, font=fl)
    base = (bl[1] + bl[3]) / 2.0
    d.text((x - bl[0], cy - base), prefix, font=fl, fill=fill)
    x += wl
    bs = d.textbbox((0, 0), ch, font=fs)
    d.text((x - bs[0], cy - (bs[1] + bs[3]) / 2.0), ch, font=fs, fill=fill)
    x += ws
    if suffix:
        d.text((x - bl[0], cy - base), suffix, font=fl, fill=fill)


def clear(box, color=WHITE):
    d.rectangle(box, fill=color)


def head(p, ang, size=15, color=INK):
    a1, a2 = ang + math.radians(152), ang - math.radians(152)
    d.polygon([p, (p[0] + size * math.cos(a1), p[1] + size * math.sin(a1)),
               (p[0] + size * math.cos(a2), p[1] + size * math.sin(a2))], fill=color)


def arrow(p0, p1, color=INK, lw=4, size=15):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = max(1e-6, math.hypot(dx, dy))
    ux, uy = dx / L, dy / L
    d.line([p0, (p1[0] - ux * size * 0.9, p1[1] - uy * size * 0.9)], fill=color, width=lw)
    head(p1, math.atan2(dy, dx), size, color)


def poly_arrow(pts, color=INK, lw=4, size=15):
    for i in range(len(pts) - 2):
        d.line([pts[i], pts[i + 1]], fill=color, width=lw)
    arrow(pts[-2], pts[-1], color, lw, size)


def vtext(txt, cx, cy, path, size, fill=INK):
    """竖排文字（顺时针 90°），用于窄高块"""
    f = F(path, size)
    b = d.textbbox((0, 0), txt, font=f)
    w, h = b[2] - b[0] + 8, b[3] - b[1] + 8
    tmp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((4 - b[0], 4 - b[1]), txt, font=f, fill=fill + (255,))
    tmp = tmp.rotate(90, expand=True)
    im.paste(tmp, (int(cx - tmp.width / 2), int(cy - tmp.height / 2)), tmp)


# ---------- E1 输入：Complete Trajectory -> Incomplete history ----------
clear((16, 338, 282, 542), WHITE)
# 框与右侧黄色块箭头（y 407..431，中心 419）垂直对齐；模块名下不放小字
d.rounded_rectangle((22, 368, 278, 470), radius=14, fill=INBOX, outline=INK, width=3)
tc("Incomplete", 150, 402, fit("Incomplete", CMB, 200), INK)
tc("history", 150, 436, fit("history", CMB, 200), INK)
log.append("E1 输入框: 照片堆 -> Incomplete history")

# ---------- E2 Input Process -> 真实输入处理 ----------
clear((325, 380, 578, 462), INBOX)
tc("History features", 451, 418, fit("History features", CMB, 215), INK)   # 框 368..469，中心 418
log.append("E2 Input Process -> History features / patch embed 4 → 128")

# ---------- E3 损失名 ----------
clear((2540, 143, 2826, 200), PANEL)
tc("mode regression", 2660, 158, fit("mode regression", CMR, 240), INK)
tc("+ classification", 2660, 187, fit("+ classification", CMR, 240), INK)
clear((2540, 533, 2826, 590), PANEL)
d.line([(2653, 533), (2653, 546)], fill=INK, width=4)
tc("dense regression", 2692, 566, fit("dense regression", CMR, 232), INK)
log.append("E3 ℒ_ts -> mode regression + classification ; ℒ_m -> dense regression")

# ---------- E4 层数：× L -> × 4 ----------
clear((1795, 106, 1892, 175), WHITE)
ink_left("\u00d7", 1802, 141.5, F(STIX, 38))
ink_left("4", 1836, 141.5, F(CMR, 44))
log.append("E4 × L -> × 4")

# ---------- E5 形状标注 T_s -> T_f ----------
ANNS = [((2624, 442, 2802, 480), PANEL, "(B, K, T_f, C)"),
        ((3105, 249, 3218, 278), CONT, "(B, KT_f, C)"),
        ((3105, 317, 3218, 346), CONT, "(B, KT_f, C)"),
        ((3105, 386, 3218, 415), CONT, "(BT_f, K, C)"),
        ((3105, 456, 3218, 485), CONT, "(BK, T_f, C)")]


def tokens_of(spec):
    out, i = [], 0
    while i < len(spec):
        c = spec[i]
        if c == "_":
            out.append(("sub", spec[i + 1]))
            i += 2
        elif c.isalpha():
            out.append(("it", c))
            i += 1
        else:
            j = i
            while j < len(spec) and not spec[j].isalpha() and spec[j] != "_":
                j += 1
            out.append(("rm", spec[i:j]))
            i = j
    return out


def math_str(toks, cx, cy, size, fill=INK):
    widths = []
    for kind, s in toks:
        f = F(CMMI if kind in ("it", "sub") else CMR, size if kind != "sub" else int(size * 0.72))
        b = d.textbbox((0, 0), s, font=f)
        widths.append((b[2] - b[0], b, f, kind, s))
    x = cx - sum(w for w, *_ in widths) / 2.0
    bb = d.textbbox((0, 0), "B", font=F(CMMI, size))
    base = (bb[1] + bb[3]) / 2.0
    for w, b, f, kind, s in widths:
        d.text((x - b[0], cy - base + (size * 0.26 if kind == "sub" else 0)), s, font=f, fill=fill)
        x += w


for (bx0, by0, bx1, by1), bg, spec in ANNS:
    reg = A[by0:by1, bx0:bx1]
    ys, xs = np.where(reg.mean(2) < 130)
    x0, x1 = bx0 + xs.min(), bx0 + xs.max()
    y0, y1 = by0 + ys.min(), by0 + ys.max()
    w0 = x1 - x0
    clear((x0 - 3, y0 - 3, x1 + 4, y1 + 4), bg)
    toks = tokens_of(spec)
    best = None
    for s in range(16, 34):
        tot = sum(d.textbbox((0, 0), t, font=F(CMMI if k in ("it", "sub") else CMR,
                                               s if k != "sub" else int(s * 0.72)))[2]
                  for k, t in toks)
        if best is None or abs(tot - w0) < best[0]:
            best = (abs(tot - w0), s)
    math_str(toks, (x0 + x1) / 2.0, (y0 + y1) / 2.0, best[1])
    log.append(f"E5 {spec} @size{best[1]}")

# ---------- E6 输出图：道路交叉口 -> K=20 轨迹扇面 ----------
clear((3296, 281, 3502, 453), PANEL)
ox, oy = 3322, 434
for k in range(20):
    ang = math.radians(-86 + (k / 19.0) * 74)
    pts = []
    for s in range(11):
        r = 132 * s / 10.0
        bend = math.radians(16) * (s / 10.0) ** 2 * (1 if k % 2 == 0 else -1)
        pts.append((ox + r * math.cos(ang + bend), oy + r * math.sin(ang + bend)))
    if k == 4:
        d.line(pts, fill=(60, 60, 60), width=8, joint="curve")
        d.line(pts, fill=(255, 226, 60), width=5, joint="curve")
    else:
        d.line(pts, fill=(176, 190, 205), width=2, joint="curve")
d.ellipse((ox - 5, oy - 5, ox + 5, oy + 5), fill=INK)
tc("K = 20 futures", 3399, 248, fit("K = 20 futures", CMB, 215), INK)
tc("no map input", 3399, 284, fit("no map input", CMR, 215), MUTED)
log.append("E6 道路缩略图 -> K=20 轨迹扇面（无地图）")

# ---------- E7 补 Last-step readout 并接通解码器主干 ----------
clear((1748, 436, 1790, 455), (242, 242, 242))  # 仅清掉原箭头（1794 处即断，与面板不相连），保留背卡底色
clear((1796, 270, 1898, 494), WHITE)
d.rounded_rectangle((1798, 275, 1896, 490), radius=10, fill=WHITE, outline=INK, width=3)
vtext("Last-step readout", 1847, 382, CMB, 24)
arrow((1747, 382), (1796, 382))
d.line([(1898, 382), (1930, 382)], fill=INK, width=4)
d.line([(1930, 255), (1930, 498)], fill=INK, width=4)          # 解码器输入主干
d.line([(1930, 255), (1939, 255)], fill=INK, width=4)          # 接 (a) 支
d.line([(1930, 498), (1939, 498)], fill=INK, width=4)          # 接 (b) 支
log.append("E7 补 Last-step readout，并补上原图缺失的编码器→解码器连线与分叉主干")

# ---------- E8 (c) 具体层数 ----------
for cy, cnt in [(264, "3"), (332, "3"), (401, "3"), (471, "2")]:
    ink_left("\u00d7", 3226, cy, F(STIX, 26), MUTED)
    ink_left(cnt, 3248, cy, F(CMR, 26), MUTED)
log.append("E8 (c) 四级层数：× 3 / × 3 / × 3 / × 2")

# ---------- E9 来源标注：按用户要求删除（2026-10-18）----------

# ---------- 收尾：把扇面清理误伤的图例黄块顶边贴回原样 ----------
im.paste(Image.open(SRC).crop((3340, 440, 3460, 458)), (3340, 440))
d = ImageDraw.Draw(im)

# ---------- E10 去掉 Last-step readout ----------
clear((1795, 262, 1904, 500), WHITE)             # readout 方块（白底区）
clear((1748, 372, 1789, 394), (242, 242, 242))   # 只清卡内，保留背卡右缘（1790..1794）不被打断
log.append("E10 去掉 Last-step readout")

# ---------- E11 解码面板内容整体左移 200，面板收窄 ----------
SHIFT = 200
clear((1907, 246, 2107, 507), PANEL)             # 清掉旧分支连线/主干
block = im.crop((2107, 30, 3571, 640))           # 面板内容整块
im.paste(block, (2107 - SHIFT, 30))
clear((3571 - SHIFT + 1, 0, 3584, 670), WHITE)   # 面板右端收到新位置
ttl = im.crop((2270, 58, 2822, 107))             # 标题重新居中
im.paste(ttl, (2270 + 96, 58))
clear((2270, 58, 2270 + 96, 107), BAND)
clear((1907, 246, 1933, 262), PANEL)             # 分支起点收到主干
clear((1907, 491, 1933, 507), PANEL)
d.line([(1933, 255), (1933, 498)], fill=INK, width=4)
arrow((1747, 382), (1931, 382))                  # 编码器 -> 解码面板
im = im.crop((0, 0, 3384, 670))                  # 画布裁到内容右缘
d = ImageDraw.Draw(im)
log.append("E11 面板内容左移 %d px、面板收窄、画布裁到 3384" % SHIFT)

# ---------- E12 补 GSM Spatial Attention：历史分支原来在此断头 ----------
# 原图把 GSM Module 直接连到 Spatial 网格，agent tokens 没有任何输出箭头；
# 按 gast_encoder.py 的 GSMAttnBlock 补一级：agent tokens + ghost token -> 注意力 -> spatial
# 实测原肘形连线：水平段 y 199..205、竖段 x 984..990、进网格段 y 415..421，线宽 7
clear((966, 196, 996, 350), WHITE)      # 原肘形竖段与拐角
clear((980, 200, 996, 412), WHITE)      # 原肘形竖段下半
d.rounded_rectangle((880, 348, 1060, 488), radius=12, fill=(191, 191, 191),
                    outline=INK, width=3)
tc("GSM Spatial", 970, 402, fit("GSM Spatial", CMB, 158), INK)
tc("Attention", 970, 440, fit("Attention", CMB, 158), INK)
arrow((826, 418), (876, 418), lw=7, size=18)                    # agent tokens -> 注意力
d.line([(890, 202), (973, 202)], fill=INK, width=7)          # ghost token -> 注意力（先画水平段补齐拐角）
arrow((970, 202), (970, 346), lw=7, size=18)
arrow((1062, 418), (1117, 418), lw=7, size=18)                  # 注意力 -> spatial 网格
log.append("E12 补 GSM Spatial Attention 一级（修历史分支断头 + ghost token 注入）")

# ---------- E13 合并两个 GSM：代码里只有一个 GSMAttnBlock ----------
# 去掉橙色 "GSM Module" 框（它只是 mask->ghost token 那一步），
# mask 直接进注意力块，箭头旁标 "ghost token"，保留机制信息但不重复模块名。
clear((604, 145, 902, 258), WHITE)                 # 清掉橙色 GSM Module 框
d.line([(560, 201), (973, 201)], fill=INK, width=7)  # mask -> 注意力（原 mask 列右缘起）
arrow((970, 201), (970, 346), lw=7, size=18)       # 拐角向下注入
tl("ghost token", 690, 222, F(CMR, 24), MUTED)     # 标注这条支路的作用
log.append("E13 合并两个 GSM：删掉橙色 GSM Module 框，mask 直连注意力并标注 ghost token")

# ---------- E14 解码器改成编码器风格：去掉蓝色底 ----------
# 面板体 #F2F7FC / 顶部带 #DEEBF7 / (c) 容器底 #E4EBF7 / 虚线分组框 #8DA8DB 一律刷白，
# 只保留模块、标题、张量、扇面与图例（图例色块 #FFE699 / #BDD7EE 与张量格不受影响）。
a = np.asarray(im).astype(int).copy()
Y0, Y1, X0, X1 = 30, 645, 1900, 3384
sub = a[Y0:Y1, X0:X1]
for col, tol in [((242, 247, 252), 7), ((228, 235, 247), 7),
                 ((222, 235, 247), 7), ((141, 168, 219), 9)]:
    m = (abs(sub - np.array(col)).max(axis=2) <= tol)
    sub[m] = (255, 255, 255)
a[Y0:Y1, X0:X1] = sub
im = Image.fromarray(a.astype(np.uint8))
d = ImageDraw.Draw(im)
log.append("E14 解码器去蓝底：面板底/顶带/(c)容器/虚线分组框 -> 白")

# ---------- E15 去掉虚线分组框的描边（浅蓝 (154,179,223) 及其抗锯齿）----------
a = np.asarray(im).astype(int).copy()
Y0, Y1, X0, X1 = 30, 645, 1900, 3384
sub = a[Y0:Y1, X0:X1]
R, B = sub[:, :, 0], sub[:, :, 2]
m = (B - R >= 30) & (B - R <= 70) & (R >= 120) & (R <= 215)
protect = [((275, 450), (2415, 2595)),   # 张量热力图
           ((440, 485), (3130, 3250)),   # 图例黄块
           ((510, 570), (3130, 3250))]   # 图例蓝块
for (py0, py1), (px0, px1) in protect:
    m[max(0, py0 - Y0):py1 - Y0, max(0, px0 - X0):px1 - X0] = False
sub[m] = (255, 255, 255)
a[Y0:Y1, X0:X1] = sub
im = Image.fromarray(a.astype(np.uint8))
d = ImageDraw.Draw(im)
log.append("E15 去掉三个虚线分组框的描边")

# ---------- E16 去底后的收尾：补两处因去掉容器/缩略图而断开的连线 ----------
clear((3365, 0, 3384, 670), WHITE)                 # 面板旧右缘残留的浅灰竖线
arrow((2634, 363), (2682, 363), lw=7, size=18)     # 张量 -> (c) 第一级（原来停在容器边上）
arrow((2895, 357), (3094, 357), lw=7, size=18)     # (c) 组 -> 预测扇面（原来指向缩略图，被扇面清理吃掉）
log.append("E16 收尾：清面板旧右缘残留；补 张量->(c)、 (c)->扇面 两条连线")

# ---------- E17 去掉 (a)/(b) 组下方的蓝色强调线，并把两条连线接到具体模块上 ----------
clear((1960, 343, 2400, 354), WHITE)      # (a) 组下方蓝线残留
clear((1960, 577, 2400, 593), WHITE)      # (b) 组下方蓝线残留
clear((2578, 354, 2686, 374), WHITE)      # 旧张量->(c) 箭头（原来停在两级之间的空隙）
poly_arrow([(2578, 363), (2620, 363), (2620, 263), (2682, 263)], lw=7, size=18)  # 接到第一级左缘
clear((3092, 348, 3130, 368), WHITE)      # 旧 (c)->扇面 箭头末端（停在空白处）
arrow((2895, 357), (3124, 357), lw=7, size=18)   # 接到扇面轨迹起点
log.append("E17 清 (a)/(b) 蓝线；张量箭头改接第一级、(c) 输出箭头接到扇面")

# ---------- E18 二次清蓝：放宽阈值再扫一遍虚线残迹，并删掉指向已删除容器的死箭头 ----------
a = np.asarray(im).astype(int).copy()
Y0, Y1, X0, X1 = 30, 645, 1900, 3384
sub = a[Y0:Y1, X0:X1]
R, B = sub[:, :, 0], sub[:, :, 2]
m = (B - R >= 30) & (B - R <= 95) & (R >= 120) & (R <= 220)
for (py0, py1), (px0, px1) in [((275, 450), (2415, 2595)),    # 张量热力图
                               ((440, 485), (3130, 3250)),    # 图例黄块
                               ((510, 570), (3130, 3250)),    # 图例蓝块
                               ((225, 465), (3085, 3335))]:   # 预测扇面
    m[max(0, py0 - Y0):py1 - Y0, max(0, px0 - X0):px1 - X0] = False
sub[m] = (255, 255, 255)
a[Y0:Y1, X0:X1] = sub
im = Image.fromarray(a.astype(np.uint8))
d = ImageDraw.Draw(im)
clear((2772, 497, 2796, 540), WHITE)      # 指向已删容器的死箭头
clear((3118, 348, 3140, 368), WHITE)
arrow((2895, 357), (3132, 357), lw=7, size=18)   # 箭头伸到扇面轨迹上
log.append("E18 二次清蓝 + 删死箭头 + 扇面箭头接上")

# ---------- E19 三次清蓝：虚线抗锯齿残留（很淡的蓝，R≥200）----------
a = np.asarray(im).astype(int).copy()
Y0, Y1, X0, X1 = 30, 645, 1900, 3384
sub = a[Y0:Y1, X0:X1]
R, B = sub[:, :, 0], sub[:, :, 2]
m = (B - R >= 22) & (B - R <= 95) & (R >= 200) & (R <= 250)
for (py0, py1), (px0, px1) in [((275, 450), (2415, 2595)),
                               ((440, 485), (3130, 3250)),
                               ((510, 570), (3130, 3250)),
                               ((225, 465), (3085, 3335))]:
    m[max(0, py0 - Y0):py1 - Y0, max(0, px0 - X0):px1 - X0] = False
sub[m] = (255, 255, 255)
a[Y0:Y1, X0:X1] = sub
im = Image.fromarray(a.astype(np.uint8))
d = ImageDraw.Draw(im)
log.append("E19 三次清蓝：清掉虚线抗锯齿的极淡蓝")

# ---------- E20 修掉 (c) 输入箭头向上多出的 24px 空尾巴 ----------
clear((2772, 174, 2798, 201), WHITE)
log.append("E20 去掉 (c) 输入箭头的空尾巴")

# ---------- E21 把 (c) 顶部两条孤立竖箭头的尾巴接到 query 立方体块上 ----------
clear((2780, 176, 2852, 202), WHITE)      # 去掉两条竖箭头伸进空白的尾巴
d.line([(2790, 205), (2898, 205)], fill=INK, width=4)   # 与立方体块之间的横连接
log.append("E21 (c) 顶部两条竖箭头改为从 query 立方体块引出")

# ---------- E22 修 4 处：query 立方体接线、双箭头、缺输出、张量入线断口 ----------
# (1) query 立方体块搬到 Hybrid CrossAttn 正上方，并从块底接两条箭头进模块顶
cubes = im.crop((2872, 176, 3072, 232))
clear((2872, 176, 3072, 232), WHITE)
im.paste(cubes, (2694, 144))
d = ImageDraw.Draw(im)
clear((2775, 196, 2902, 243), WHITE)              # 原来悬空的两条竖箭头 + 后加的横线
arrow((2740, 197), (2740, 243), lw=4, size=13)
arrow((2846, 197), (2846, 243), lw=4, size=13)
# (2) 去掉 (c) 右侧那条起点在两级之间、且叠了三个箭头头的线
clear((2893, 346, 3142, 370), WHITE)
# (3) (c) 的输出改为从最后一级 State BiMamba 底部引出 -> 预测扇面
poly_arrow([(2788, 498), (2788, 520), (3122, 520), (3122, 442)], lw=4, size=15)
# (4) 补完 (b)->张量 的竖线；(a)->张量 补箭头头
arrow((2503, 496), (2503, 442), lw=4, size=15)
arrow((2499, 272), (2499, 283), lw=4, size=10)
log.append("E22 修 query 立方体接线/双箭头/(c) 缺输出/(b)->张量 断口")

# ---------- E23 补 (a)/(b) 两支的输出连线与损失引出线 ----------
d.line([(2369, 250), (2500, 250)], fill=INK, width=4)      # (a) 支 -> 张量/损失 的横连接
arrow((2448, 250), (2448, 205), lw=4, size=14)             # (a) -> mode regression + classification
clear((2372, 494, 2500, 502), WHITE)                       # 原灰色横线（不与模块相连）
d.line([(2370, 498), (2501, 498)], fill=INK, width=4)      # (b) 支 -> 张量 的横连接（改黑、接上模块）
clear((2445, 528, 2466, 552), WHITE)                       # 旧的错位短竖线
arrow((2453, 500), (2453, 548), lw=4, size=14)             # (b) -> dense regression
log.append("E23 补 (a)/(b) 输出横连接 + 两条损失引出箭头")

# ---------- E24 去掉 (a) 支那条 2px 的重复横线 ----------
clear((2372, 254, 2496, 260), WHITE)
log.append("E24 去掉 (a) 支重复的细横线")

# ---------- E25 去掉 ghost token 标注小字（文字在 y 213..231，其上 198..204 是连线，保留）----------
clear((686, 208, 815, 236), WHITE)
log.append("E25 去掉 ghost token 小字")

# ---------- E26 按当前 GAST 主线更新编码器尾段 ----------
# 当前配置 GERE/ECTA 全开：
# 1) GERE：GSM 变为 geometry-aware attention；
# 2) RMCM：可靠度调制单向 Mamba 输入；
# 3) EWSA：Mamba 后接 evidence-weighted state aggregation。
#
# GERE 标签：保留同一个 GSM 模块，不增设伪造的独立子模块。
clear((886, 354, 1054, 482), (191, 191, 191))
tc("Geometry-aware", 970, 382, fit("Geometry-aware", CMB, 154), INK)
tc("GSM Spatial", 970, 416, fit("GSM Spatial", CMB, 154), INK)
tc("Attention", 970, 450, fit("Attention", CMB, 154), INK)

# RMCM：删除旧 Temporal Decay/gamma 支路，改画 age -> reliability -> ⊙。
clear((1278, 84, 1540, 222), WHITE)
clear((1272, 216, 1548, 340), WHITE)
clear((1332, 412, 1622, 480), WHITE)
# age/evidence 输入条
# 由 mask 可观测性支路延伸到 age 输入，保留与 ghost/GSM 共用的证据来源。
d.line([(973, 201), (1407, 201)], fill=INK, width=4)
arrow((1407, 201), (1407, 154), lw=4, size=11)
d.rounded_rectangle((1312, 107, 1502, 153), radius=7, outline=(46, 154, 168), width=3)
for x0, x1 in [(1319, 1353), (1357, 1391), (1395, 1429), (1433, 1467), (1471, 1495)]:
    d.rectangle((x0, 114, x1, 146), fill=(242, 247, 252), outline=INK, width=2)
tc("age", 1407, 130, fit("age", CMR, 45, start=18), INK)
# RMCM reliability: reliability = exp(-tau * age), tau learnable; arrow feeds ⊙.
d.rounded_rectangle((1295, 234, 1524, 323), radius=12, fill=(46, 154, 168),
                    outline=INK, width=3)
tc("Reliability", 1409, 258, fit("Reliability", CMB, 200), WHITE)
tc("modulation", 1409, 291, fit("modulation", CMB, 200), WHITE)
arrow((1409, 325), (1409, 435), lw=4, size=13)
# ⊙ 输出为 evidence-modulated temporal features，进入 Mamba。
d.rounded_rectangle((1353, 437, 1505, 460), radius=4, fill=(76, 170, 179), outline=INK, width=2)
for x0, x1 in [(1358, 1380), (1385, 1408), (1413, 1436), (1441, 1464), (1469, 1500)]:
    d.rectangle((x0, 440, x1, 457), fill=(76, 170, 179), outline=INK, width=1)
arrow((1332, 448), (1350, 448), lw=4, size=11)

# 单向 causal Mamba ×4。清掉原双向字样和叠卡残段，重画单个清晰模块。
clear((1608, 100, 1861, 620), WHITE)
d.rounded_rectangle((1626, 140, 1738, 612), radius=12, fill=(242, 242, 242), outline=INK, width=3)
vtext("Causal Temporal Mamba Block", 1682, 376, CMB, 21)
ink_left("×", 1744, 143, F(STIX, 25), MUTED)
ink_left("4", 1768, 143, F(CMR, 27), MUTED)
arrow((1507, 448), (1624, 448), lw=7, size=18)

# EWSA：Mamba 后进行证据加权状态聚合，再接解码器分叉主干。
d.rounded_rectangle((1760, 294, 1858, 470), radius=10, fill=(242, 242, 242), outline=INK, width=3)
tc("Evidence", 1809, 338, F(CMB, 15), INK)
tc("weighted", 1809, 382, F(CMB, 15), INK)
tc("readout", 1809, 426, F(CMB, 15), INK)
# 输入/输出连接贴到框的中线，框内文字不被连线穿过
arrow((1740, 382), (1758, 382), lw=4, size=8)
d.line([(1858, 382), (1862, 382)], fill=INK, width=4)
log.append("E26 GAST 主线：GERE geometry-aware GSM；RMCM reliability exp(-tau*age) 调制；causal Mamba ×4；EWSA aggregation")

# ---------- E27 顶部观测证据状态支路 ----------
# 参考图风格：把 e(i,t)=(s,a,r) 统一画在主干最上方，
# s/a/r 分别以橙色虚线注入 GERE、Reliability modulation、Evidence-weighted state aggregation。
# 先清掉旧的顶部 age 小条及其旧水平连接，避免两套证据支路重复。
clear((965, 194, 1415, 210), WHITE)
clear((1278, 84, 1540, 222), WHITE)
# 顶部栏覆盖原图空白上带，不遮 Binary Mask 主网格（x<480）。
clear((480, 6, 1940, 98), WHITE)
d.rounded_rectangle((500, 8, 1935, 94), radius=12,
                    fill=(255, 245, 235), outline=ORANGE, width=3)
tc("Observation evidence state", 675, 29,
   fit("Observation evidence state", CMB, 300, start=24), ORANGE)
tc("e(i,t) = (s, a, r)", 675, 68, F(STIX, 20), INK)

# 三个状态分量：只保留定义，不再添加模块下方解释小字。
def evidence_box(x0, x1, text):
    d.rounded_rectangle((x0, 22, x1, 80), radius=7,
                        fill=WHITE, outline=ORANGE, width=2)
    tc(text, (x0 + x1) / 2.0, 51, fit(text, STIX, x1 - x0 - 16, start=20), INK)

evidence_box(900, 1050, "s = m")
evidence_box(1215, 1603, "a = t − ℓ")
evidence_box(1680, 1932, "r = exp(−τa)")

# 橙色虚线注入箭头：虚线杆 + 实心小箭头头，目标落在三个当前模块上。
def evidence_arrow(p0, p1, dash=10, gap=7, lw=3, size=10):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = max(1.0, math.hypot(dx, dy))
    ux, uy = dx / length, dy / length
    pos = 0.0
    while pos < length - size:
        end = min(pos + dash, length - size)
        d.line([(p0[0] + ux * pos, p0[1] + uy * pos),
                (p0[0] + ux * end, p0[1] + uy * end)],
               fill=ORANGE, width=lw)
        pos += dash + gap
    head(p1, math.atan2(dy, dx), size, ORANGE)

evidence_arrow((975, 96), (970, 348))    # s -> geometry-aware GSM
evidence_arrow((1409, 96), (1409, 228))  # a -> reliability modulation
evidence_arrow((1809, 96), (1809, 286))  # r -> evidence-weighted readout
log.append("E27 顶部观测证据状态：e(i,t)=(s,a,r)，三条虚线注入 GERE/RMCM/EWSA")

im.save(OUT)
print("saved", OUT, im.size)
for l in log:
    print("  ", l)
