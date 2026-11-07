"""TrajGap 输入样本截图：同一场景、同一观测窗的 4 个邻近帧，用于论文框架图的输入面板。

不做预测可视化：只画观测（focal 蓝色轨迹 + 邻居灰色轨迹）在原版场景图上，
缺失帧用红色空心圈标出。四个面板共享同一裁剪窗口、同一画布尺寸，可直接并排。

用法（DeMo env，仓库根目录，无需 GPU）：
  LD_LIBRARY_PATH=$CONDA_PREFIX/lib ~/.conda/envs/DeMo/bin/python \
      docs/tools/screenshot_trajgap_inputs.py --scene ZARA2-M

输出：docs/figures/trajgap_input_frames/<scene>_input_t-4.png 等 4 张。
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # 复用 visualize_trajgap_scenes 的场景图与投影

import numpy as np  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

import visualize_trajgap_scenes as V  # noqa: E402

COL_OBS = "#0072B2"
COL_MISSING = "#D55E00"
COL_NBR = "#8a8a8a"


def contiguous_runs(mask):
    """布尔序列 -> 连续 True 段列表 [(i0,i1) 含端点]。"""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    runs, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            runs.append((start, a))
            start = b
    runs.append((start, idx[-1]))
    return runs


def pick_sample(scene, frames=(4, 5, 6, 7), min_valid=5, min_actors=3,
                min_disp=1.2, seed=7):
    """选一个适合展示的样本：后段有缺失但锚点仍近、邻居够、focal 在移动。"""
    frames = list(frames)
    ds = V.TrajGapDataset("/home/lbh/TrajImpute/dataset/TrajImpute", scene, "test")
    rng = np.random.default_rng(seed)
    for i in rng.permutation(len(ds)):
        s = ds[int(i)]
        vm = s["x_valid_mask"].numpy()          # (A,8)
        fvm = vm[0]
        if fvm.sum() < min_valid:
            continue
        if not (~fvm[frames]).any():            # 展示帧里必须能看到缺失
            continue
        if not fvm[frames].any():               # 也不能全缺
            continue
        if fvm.nonzero()[0][-1] < 5:            # 最近有效锚点太早的不要
            continue
        if vm.shape[0] < min_actors:
            continue
        if not vm[1:][:, frames].any():         # 邻居在展示帧至少出现过
            continue
        origin = s["origin"].numpy()
        theta = float(s["theta"])
        w = V.local2world(s["x_positions"][0].numpy()[None], origin.reshape(1, 2), theta)[0]
        wv = w[fvm]
        if np.linalg.norm(wv[-1] - wv[0]) < min_disp:
            continue
        return ds, int(i), s
    raise RuntimeError("no sample matches criteria")


def render(scene, s, out_dir, frames=(4, 5, 6, 7), size_in=3.4, dpi=200):
    frames = list(frames)
    img = Image.open(V.MAPS_DIR / V.SCENE_CONF[scene]["img"])
    W, H = img.size
    origin = s["origin"].numpy()
    theta = float(s["theta"])

    world = V.local2world(s["x_positions"].numpy(), origin.reshape(1, 2), theta)  # (A,8,2)
    valid = s["x_valid_mask"].numpy()                                            # (A,8)
    pix = V.world2pix(scene, world.reshape(-1, 2)).reshape(world.shape)          # (A,8,2)

    # 共享裁剪窗：focal 全部有效点 + 邻居有效点，方形化
    pts = [pix[0][valid[0]]] + [pix[a][valid[a]] for a in range(1, len(pix))]
    pts = np.concatenate(pts)
    pad = 70
    u0, u1 = max(pts[:, 0].min() - pad, 0), min(pts[:, 0].max() + pad, W)
    v0, v1 = max(pts[:, 1].min() - pad, 0), min(pts[:, 1].max() + pad, H)
    span = max(u1 - u0, v1 - v0)
    cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    u0, u1 = np.clip([cu - span / 2, cu + span / 2], 0, W)
    v0, v1 = np.clip([cv - span / 2, cv + span / 2], 0, H)

    out = []
    for t in frames:
        fig, ax = plt.subplots(figsize=(size_in, size_in), dpi=dpi)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.imshow(img)
        ax.set_xlim(u0, u1)
        ax.set_ylim(v1, v0)
        ax.axis("off")

        # 邻居：灰色轨迹（截至当前帧）+ 当前位置点
        for a in range(1, len(pix)):
            m = valid[a].copy()
            m[t + 1:] = False
            for i0, i1 in contiguous_runs(m):
                ax.plot(pix[a, i0:i1 + 1, 0], pix[a, i0:i1 + 1, 1], "-",
                        color=COL_NBR, lw=1.0, zorder=2)
            if m.any():
                li = np.where(m)[0][-1]
                ax.plot(pix[a, li, 0], pix[a, li, 1], "o",
                        color=COL_NBR, ms=3, zorder=3)

        # focal：蓝色轨迹（断开缺口）+ 当前帧状态
        m = valid[0].copy()
        m[t + 1:] = False
        for i0, i1 in contiguous_runs(m):
            ax.plot(pix[0, i0:i1 + 1, 0], pix[0, i0:i1 + 1, 1], "-",
                    color=COL_OBS, lw=2.4, zorder=4)
            ax.plot(pix[0, i0:i1 + 1, 0], pix[0, i0:i1 + 1, 1], "o",
                    color=COL_OBS, ms=2.8, zorder=5)
        li = int(np.where(m)[0][-1])
        if valid[0][t]:
            ax.plot(pix[0, t, 0], pix[0, t, 1], "o",
                    color=COL_OBS, ms=8, zorder=6)
        else:
            ax.plot(pix[0, li, 0], pix[0, li, 1], "o", mfc="none",
                    mec=COL_MISSING, ms=11, mew=1.8, zorder=6)
            ax.annotate("missing", xy=(pix[0, li, 0], pix[0, li, 1]),
                        xytext=(0, 12), textcoords="offset points", ha="center",
                        color=COL_MISSING, fontsize=7.5, zorder=7)

        tag = f"t{t - 7:+d}".replace("+", "+") if t != 7 else "t"
        ax.text(0.03, 0.04, tag, transform=ax.transAxes, fontsize=9,
                color="white", va="bottom",
                bbox=dict(fc="black", alpha=0.55, pad=1.5, lw=0))

        path = out_dir / f"{scene}_input_t{t - 7:+d}.png".replace("+", "m") \
            if t != 7 else out_dir / f"{scene}_input_t.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        out.append(path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="ZARA2-M")
    ap.add_argument("--out-dir",
                    default=str(V.REPO / "docs/figures/trajgap_input_frames"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds, idx, s = pick_sample(args.scene, seed=args.seed)
    vm = s["x_valid_mask"][0].numpy()
    print(f"[{args.scene}] sample idx={idx} actors={s['x_positions'].shape[0]} "
          f"valid={[int(v) for v in vm]} missing_at="
          f"{[j for j in range(8) if not vm[j]]}")
    for p in render(args.scene, s, out_dir):
        print(" ->", p.name)


if __name__ == "__main__":
    main()
