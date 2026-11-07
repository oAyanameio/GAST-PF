"""源数据帧截图：从 sgan 系原始世界坐标直接渲染连续 4 帧，无任何标注。

用途：论文框架图的输入面板（干净背景图 + 行人位置点，不做缺失/预测标记）。

用法（任意 python3，无需 GPU/env）：
  python3 docs/tools/screenshot_raw_frames.py [--src crowds_zara02 --start 930]

选帧标准：连续 4 帧（间隔一致）、每帧 >=6 人、所有点投影落在背景图内。
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
MAPS_DIR = REPO / "docs/figures/scene_maps"
RAW_DIR = Path("/home/lbh/MoFlow/data/eth_ucy/original/raw/all_data")

# 世界(米)->像素，与 visualize_trajgap_scenes.py 相同
UCY_SX, UCY_SY, UCY_TY = 0.02104651, 0.02386598, 13.74680446

SOURCES = {
    "crowds_zara02": dict(img="zara02.png", kind="ucy"),
    "crowds_zara01": dict(img="zara01.png", kind="ucy"),
    "students003": dict(img="students03.png", kind="ucy"),
}
# HOTEL/ETH 换算不同，如需可加


def world2pix(scene, xy):
    u = xy[..., 0] / UCY_SX
    v = (UCY_TY - xy[..., 1]) / UCY_SY
    return np.stack([u, v], axis=-1)


def pick_frames(a, min_peds=6, span=4):
    frames = np.unique(a[:, 0])
    cnt = {f: int((a[:, 0] == f).sum()) for f in frames}
    fl = sorted(frames)
    for i in range(len(fl) - span + 1):
        q = fl[i:i + span]
        if not all(q[k + 1] - q[k] == q[1] - q[0] for k in range(span - 1)):
            continue
        if min(cnt[f] for f in q) < min_peds:
            continue
        ok = True
        for f in q:
            pts = a[a[:, 0] == f][:, 2:4]
            px = world2pix("ucy", pts)
            W, H = 720, 576
            # 至少 90% 的点在图内（人群大的场景允许少量边缘点）
            inside = ((px[:, 0] > 30) & (px[:, 0] < W - 30) &
                      (px[:, 1] > 30) & (px[:, 1] < H - 30)).mean()
            if inside < 0.9:
                ok = False
                break
        if ok:
            yield q, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="crowds_zara02")
    ap.add_argument("--start", type=float, default=None,
                    help="起始帧号；缺省=自动选第一个满足条件的段")
    ap.add_argument("--min-peds", type=int, default=6)
    ap.add_argument("--win-half", type=int, default=170,
                    help="方形窗口半边长（像素）")
    ap.add_argument("--no-dots", action="store_true",
                    help="完全不加任何标记，只输出场景图裁剪帧")
    ap.add_argument("--dot", type=int, default=14, help="行人点半径(px)")
    ap.add_argument("--out-dir",
                    default=str(REPO / "docs/figures/raw_scene_frames"))
    args = ap.parse_args()

    src = SOURCES[args.src]
    img = Image.open(MAPS_DIR / src["img"])
    W, H = img.size
    a = np.loadtxt(RAW_DIR / f"{args.src}.txt")

    if args.start is not None:
        step = float(np.diff(np.unique(a[:, 0]))[0])
        frames = [args.start + k * step for k in range(4)]
    else:
        gen = pick_frames(a, args.min_peds)
        frames, cnts = next(gen)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 共享窗口：4 帧所有人点的联合 bbox，居中方形化
    allpx = []
    for f in frames:
        pts = a[a[:, 0] == f][:, 2:4]
        allpx.append(world2pix("ucy", pts))
    allpx = np.concatenate(allpx)
    cu, cv = (allpx[:, 0].max() + allpx[:, 0].min()) / 2, \
             (allpx[:, 1].max() + allpx[:, 1].min()) / 2
    half = args.win_half
    u0, u1 = int(np.clip(cu - half, 0, W)), int(np.clip(cu + half, 0, W))
    v0, v1 = int(np.clip(cv - half, 0, W)), int(np.clip(cv + half, 0, H))

    for k, f in enumerate(frames):
        canvas = img.crop((u0, v0, u1, v1)).convert("RGB")
        if not args.no_dots:
            px = world2pix("ucy", a[a[:, 0] == f][:, 2:4]) - [u0, v0]
            # 行人位置：黑色实心圆点（源数据标注本身；无轨迹/缺失/标签）
            from PIL import ImageDraw
            d = ImageDraw.Draw(canvas)
            for x, y in px:
                r = args.dot / 2
                d.ellipse([x - r, y - r, x + r, y + r], fill=(24, 24, 24))
        out = out_dir / f"{args.src}_frame{k + 1}.png"
        canvas.save(out, dpi=(200, 200))
        print(f"frame{k+1} raw_frame_id={f:.0f} n_peds={len(px) if not args.no_dots else 0} -> {out.name} "
              f"win=({u0},{v0})-({u1},{v1})")


if __name__ == "__main__":
    main()
