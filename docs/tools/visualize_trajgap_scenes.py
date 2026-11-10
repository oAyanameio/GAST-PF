"""TrajGap 多模态定性可视化：将 GAST 的全部未来模式叠加到场景图。

用法（DeMo env，仓库根目录）：
  PYTHONPATH=outputs/trajgap_gast_goal_3gpu_retry/gpu0/GAST-Goal_ETH-M_mixed-direct_seed2024_uni/train \
  CUDA_VISIBLE_DEVICES=3 ~/.conda/envs/DeMo/bin/python docs/tools/visualize_trajgap_scenes.py \
      --scenes ETH-M --scenes ZARA2-M

数据口径：
  - TrajImpute 各场景 pkl 的 obs/pred 即 sgan 系世界坐标（米），与原始
    biwi_*/crowds_*/students* 完全同源（已在 2026-11-06 逐场景对齐验证）。
  - 世界->像素投影：
      ETH   : 官方 EWAP H（列已按 u,v 置换），验证落图 100%
      HOTEL : IMGCN H（pixel->world）求逆；背景图需 ROTATE_90（576x720）
      UCY 三景（UNIV/ZARA1/ZARA2）: u = x/0.02104651, v = (13.74680446-y)/0.02386598
              （官方 UCY 对角缩放，Habiba-Amroune CSV 拟合残差 0；落图 ~99.9%）

选样规则（防选择性展示）：
  每场景按 focal missing_count 分 3 层（0 / 1-4 / >=5），固定种子抽取最多 32 个候选
  前向；要求 best-of-20 和 GT 预测跨度 >=28px、路径长度 >=40px，历史至少有一段
  连续移动。合格候选中取联合可见度最接近第 75 百分位者；这是轨迹形态展示，
  不作为随机代表样本或性能证据。

输出：每场景/缺失层一张高清图，以及 3 缺失层 × 5 场景的 GAST PNG/PDF 汇总图；
图中绘制观测历史、真实未来和按最小 FDE 选出的 best-of-20 预测。背景为静态场景图，
不是逐时刻同步的原始视频帧（当前数据目录不含视频帧）。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]

# 必须在 import src 之前：使用正式五场景 GAST 轮次的快照代码，保证
# checkpoint 的参数名、decoder 和 encoder 实现与推理代码一致。
SNAPSHOT_ROOT = REPO / "outputs/trajgap_gast_5scene_seed2024/GAST_ETH-M_mixed-direct_seed2024_uni/train"
sys.path.insert(0, str(SNAPSHOT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

from src.datamodule.trajimpute_dataset import (  # noqa: E402
    TrajGapDataset, trajimpute_collate_fn,
)

# ---------------------------------------------------------------- 场景映射

MAPS_DIR = REPO / "docs/figures/scene_maps"
MISSING_LAYERS = (
    ("m0", "0 missing"),
    ("m14", "1-4 missing"),
    ("m5p", "5+ missing"),
)

H_ETH_OFFICIAL = np.array([
    [2.0091900e-03, 2.8128700e-02, -4.6693600e+00],
    [2.5195500e-02, 8.0625700e-04, -5.0608800e+00],
    [9.2512200e-05, 3.4555400e-04, 4.6255300e-01],
])  # world(x,y) -> pixel(u,v)，ECP-MPC 已置换列版

H_HOTEL_IMG = np.array([
    [1.1048200e-02, 6.6958900e-04, -3.3295300e+00],
    [-1.5966000e-03, 1.1632400e-02, -5.3951400e+00],
    [1.1190700e-04, 1.3617400e-05, 5.4276600e-01],
])  # pixel(u,v) -> world(x,y)，IMGCN hotel H

UCY_SX, UCY_SY, UCY_TY = 0.02104651, 0.02386598, 13.74680446

SCENE_CONF = {
    "ETH-M":   dict(img="eth.png",        map="world2pix_eth"),
    "HOTEL-M": dict(img="hotel_rot90.png", map="world2pix_hotel"),
    "UNIV-M":  dict(img="students03.png", map="world2pix_ucy"),
    "ZARA1-M": dict(img="zara01.png",     map="world2pix_ucy"),
    "ZARA2-M": dict(img="zara02.png",     map="world2pix_ucy"),
}
SCENE_LABELS = {
    "ETH-M": "ETH",
    "HOTEL-M": "HOTEL",
    "UNIV-M": "UNIV",
    "ZARA1-M": "ZARA1",
    "ZARA2-M": "ZARA2",
}

CKPTS = {
    "ETH-M":   "outputs/trajgap_gast_5scene_seed2024/GAST_ETH-M_mixed-direct_seed2024_uni/train/checkpoints/epoch=54.ckpt",
    "HOTEL-M": "outputs/trajgap_gast_5scene_seed2024/GAST_HOTEL-M_mixed-direct_seed2024_uni/train/checkpoints/epoch=51.ckpt",
    "UNIV-M":  "outputs/trajgap_gast_5scene_seed2024/GAST_UNIV-M_mixed-direct_seed2024_uni/train/checkpoints/epoch=52.ckpt",
    "ZARA1-M": "outputs/trajgap_gast_5scene_seed2024/GAST_ZARA1-M_mixed-direct_seed2024_uni/train/checkpoints/epoch=55.ckpt",
    "ZARA2-M": "outputs/trajgap_gast_5scene_seed2024/GAST_ZARA2-M_mixed-direct_seed2024_uni/train/checkpoints/epoch=36.ckpt",
}

H_ETH_INV = np.linalg.inv(H_ETH_OFFICIAL)
H_HOTEL_INV = np.linalg.inv(H_HOTEL_IMG)


def world2pix(scene: str, xy: np.ndarray) -> np.ndarray:
    """xy (..., 2) 世界坐标(米) -> 像素 (..., 2) [u, v]。"""
    kind = SCENE_CONF[scene]["map"]
    h = np.stack([xy[..., 0], xy[..., 1], np.ones(xy.shape[:-1])], axis=-1)
    if kind == "world2pix_eth":
        p = h @ H_ETH_INV.T
    elif kind == "world2pix_hotel":
        p = h @ H_HOTEL_INV.T
    else:
        u = xy[..., 0] / UCY_SX
        v = (UCY_TY - xy[..., 1]) / UCY_SY
        return np.stack([u, v], axis=-1)
    return p[..., :2] / p[..., 2:3]


def local2world(pred, origin, theta):
    """pred (K,T,2) 局部坐标 -> 世界坐标。

    dataset 正变换: local = (world - origin) @ R, R=[[c,-s],[s,c]]，
    故 world = local @ R^T + origin。与旧 submission_ethucy 的
    [c,s;-s,c] 矩阵互为转置，二者一致。
    """
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return pred @ rot.T + origin.reshape(1, 1, 2)


# ---------------------------------------------------------------- 模型

# 正式五场景 GAST manifest 记录的完整模型配置（git_revision 9752c169）
GAST_MODEL_CONFIG = dict(
    embed_dim=128, future_steps=12, num_heads=8, mlp_ratio=4.0,
    qkv_bias=False, drop_path=0.2, num_actor_types=1, num_modes=20,
    dt=0.4, obs_len=8, bimamba=False, encoder_type="gast",
    gast_depth=4, gast_gsm_operator="max",
    gast_temporal_direction="unidirectional",
    goal_endpoint_loss_weight=1.0, mode_diversity_loss_weight=0.05,
    mode_diversity_margin=0.5, classification_target_mode="trajectory",
    gast_gere_spatial=True, gast_ecta=True,
)


def build_model(scene: str):
    """用正式 GAST 快照代码构建并加载对应场景 checkpoint。"""
    from src.model.trajectory_forecaster import TrajectoryForecaster  # noqa: E402

    import src.model.trajectory_forecaster as tf_snap
    model = tf_snap.TrajectoryForecaster(**GAST_MODEL_CONFIG)
    ckpt = torch.load(CKPTS[scene], map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    cleaned = {k[len("net."):]: v for k, v in state.items() if k.startswith("net.")}
    model.load_state_dict(cleaned, strict=True)
    return model


# ---------------------------------------------------------------- 选样与前向

def stratify(dataset, per_layer=128, seed=2024):
    """Sample focal histories from three missingness strata."""
    rng = np.random.default_rng(seed)
    mc = dataset.missing_counts.numpy() if torch.is_tensor(dataset.missing_counts) \
        else np.asarray(dataset.missing_counts)
    layers = [
        ("m0", np.where(mc == 0)[0]),
        ("m14", np.where((mc >= 1) & (mc <= 4))[0]),
        ("m5p", np.where(mc >= 5)[0]),
    ]
    out = {}
    for name, idx in layers:
        if len(idx) == 0:
            continue
        take = rng.choice(idx, size=min(per_layer, len(idx)), replace=False)
        out[name] = np.sort(take)
    return out


@torch.no_grad()
def forward_indices(model, dataset, indices, device, batch_size=4):
    """对给定索引跑前向，返回 {idx: (pred_world(K,12,2), pi(K), minfde, top1)}。"""
    model = model.to(device).eval()
    results = {}
    for s in range(0, len(indices), batch_size):
        chunk = np.asarray(indices[s:s + batch_size])
        samples = [dataset[int(i)] for i in chunk]
        batch = trajimpute_collate_fn(samples)
        bd = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(bd)
        pred = out["new_y_hat"] if out.get("new_y_hat") is not None else out["y_hat"]
        pi = out["new_pi"] if out.get("new_pi") is not None else out["pi"]
        pred = pred[..., :2].float().cpu().numpy()          # (B,K,T,2) 局部
        pi = pi.float().cpu().numpy()                        # (B,K) logits -> softmax
        pi = np.exp(pi - pi.max(axis=-1, keepdims=True))
        pi = pi / pi.sum(axis=-1, keepdims=True)
        origin = batch["origin"].numpy()                     # (B,2)
        theta = batch["theta"].numpy()                       # (B,)
        target = batch["target"][:, 0].numpy()               # (B,T,2) 局部
        obs_local = batch["x_positions"][:, 0].float().cpu().numpy()
        obs_valid = batch["x_valid_mask"][:, 0].cpu().numpy()
        for j, idx in enumerate(chunk):
            pw = local2world(pred[j], origin[j], float(theta[j]))
            tw = local2world(target[j][None], origin[j], float(theta[j]))[0]
            ow = local2world(obs_local[j][None], origin[j], float(theta[j]))[0]
            fde = np.linalg.norm(pw[:, -1] - tw[-1], axis=-1)  # (K,)
            results[int(idx)] = dict(
                pred_world=pw, gt_world=tw, pi=pi[j],
                minfde=float(fde.min()), top1=int(np.argmax(pi[j])),
                obs_world=ow, obs_valid=obs_valid[j],
            )
    return results


def best_of_k_index(pred_world, gt_world):
    """Select the oracle best-of-K trajectory by final displacement error."""
    pred_world = np.asarray(pred_world)
    gt_world = np.asarray(gt_world)
    if pred_world.ndim != 3 or pred_world.shape[-1] != 2:
        raise ValueError(f"expected predictions (K,T,2), got {pred_world.shape}")
    if gt_world.ndim != 2 or gt_world.shape[-1] != 2:
        raise ValueError(f"expected ground truth (T,2), got {gt_world.shape}")
    if pred_world.shape[1] != len(gt_world):
        raise ValueError("prediction and ground-truth horizons do not match")
    final_errors = np.linalg.norm(pred_world[:, -1] - gt_world[-1], axis=-1)
    return int(np.argmin(final_errors))


def trajectory_visibility(scene, trajectory_world):
    """Measure a trajectory's pixel-space bounding-box span and arc length."""
    points = world2pix(scene, np.asarray(trajectory_world))
    span = float(np.linalg.norm(np.ptp(points, axis=0)))
    arc = float(np.linalg.norm(np.diff(points, axis=0), axis=-1).sum())
    return span, arc


def observed_visibility(scene, result):
    """Measure focal-history span and arc without connecting across missing frames."""
    points = world2pix(scene, result["obs_world"])
    valid = np.asarray(result["obs_valid"], dtype=bool)
    visible = points[valid]
    if len(visible) < 2:
        return 0.0, 0.0, 0
    span = float(np.linalg.norm(np.ptp(visible, axis=0)))
    adjacent = valid[:-1] & valid[1:]
    steps = np.linalg.norm(np.diff(points, axis=0), axis=-1)
    arc = float(steps[adjacent].sum())
    run = current = 0
    for is_valid in valid:
        current = current + 1 if is_valid else 0
        run = max(run, current)
    return span, arc, run


def pick_visible_sample(results, indices, scene, min_span_px=28, min_arc_px=40):
    """Pick a visible prediction/ground-truth pair near the upper quartile."""
    metrics = []
    for index in indices:
        result = results[int(index)]
        mode = best_of_k_index(result["pred_world"], result["gt_world"])
        pred_span, pred_arc = trajectory_visibility(
            scene, result["pred_world"][mode]
        )
        gt_span, gt_arc = trajectory_visibility(scene, result["gt_world"])
        obs_span, obs_arc, obs_run = observed_visibility(scene, result)
        score = min(
            pred_span / min_span_px, pred_arc / min_arc_px,
            gt_span / min_span_px, gt_arc / min_arc_px,
            obs_span / 8.0, obs_arc / 12.0,
        )
        metrics.append((int(index), pred_span, pred_arc, gt_span, gt_arc,
                        obs_span, obs_arc, obs_run, score))

    eligible = [item for item in metrics
                if item[1] >= min_span_px and item[2] >= min_arc_px
                and item[3] >= min_span_px and item[4] >= min_arc_px
                and item[6] >= 12 and item[7] >= 2]
    pool = eligible or metrics
    target = float(np.quantile([item[8] for item in pool], 0.75))
    selected = min(pool, key=lambda item: abs(item[8] - target))
    return (*selected[:8], len(eligible))


# ---------------------------------------------------------------- 绘图

def render_one(
    scene, sample, res, out_png, zoom_pad_px=120, ax=None, title=None,
    show_legend=True, view_aspect=None,
):
    """Draw one GAST history / ground-truth / best-of-K panel."""
    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=200)
    img = Image.open(MAPS_DIR / SCENE_CONF[scene]["img"])
    W, H = img.size

    origin = sample["origin"].numpy()                   # (1,2)
    theta = float(sample["theta"])
    # 观测世界坐标：复用预测轨迹的局部坐标逆变换。
    obs_local = sample["x_positions"].numpy()           # (A,8,2)
    obs_world = local2world(obs_local, origin, theta)
    valid = sample["x_valid_mask"].numpy()              # (A,8)
    gt_world = res["gt_world"]                          # (12,2) focal GT 世界坐标
    pred_world = res["pred_world"]                      # (K,12,2)
    best = best_of_k_index(pred_world, gt_world)

    # 全部像素坐标
    focal_obs_px = world2pix(scene, obs_world[0])
    focal_valid = valid[0]
    gt_px = world2pix(scene, gt_world)
    best_px = world2pix(scene, pred_world[best])

    # Display a fixed-aspect crop in summary grids without distorting the map.
    pts = np.concatenate([focal_obs_px[focal_valid], gt_px, best_px])
    u0, u1 = pts[:, 0].min() - zoom_pad_px, pts[:, 0].max() + zoom_pad_px
    v0, v1 = pts[:, 1].min() - zoom_pad_px, pts[:, 1].max() + zoom_pad_px
    if view_aspect is not None and view_aspect > 0:
        crop_h = max(v1 - v0, (u1 - u0) / view_aspect)
        crop_w = crop_h * view_aspect
        if crop_w > W:
            crop_w = float(W)
            crop_h = crop_w / view_aspect
        if crop_h > H:
            crop_h = float(H)
            crop_w = crop_h * view_aspect
        center_u = (u0 + u1) / 2
        center_v = (v0 + v1) / 2
        u0, u1 = center_u - crop_w / 2, center_u + crop_w / 2
        v0, v1 = center_v - crop_h / 2, center_v + crop_h / 2
        if u0 < 0:
            u1 -= u0
            u0 = 0
        if u1 > W:
            u0 -= u1 - W
            u1 = W
        if v0 < 0:
            v1 -= v0
            v0 = 0
        if v1 > H:
            v0 -= v1 - H
            v1 = H
    u0, v0 = max(u0, 0), max(v0, 0)
    u1, v1 = min(u1, W), min(v1, H)
    if u1 - u0 < 200: u1 = min(u0 + 200, W)
    if v1 - v0 < 200: v1 = min(v0 + 200, H)

    ax.imshow(img)
    ax.set_xlim(u0, u1)
    ax.set_ylim(v1, v0)
    ax.axis("off")

    ax.plot(
        gt_px[:, 0], gt_px[:, 1], "-o", color="#20A36A",
        lw=1.0, ms=1.6, mec="#20A36A", mew=0, zorder=5,
    )
    ax.plot(
        best_px[:, 0], best_px[:, 1], "-o", color="#F2C230",
        lw=1.2, ms=1.8, mec="#F2C230", mew=0, zorder=6,
    )
    # Keep the original time positions and break the line at missing frames.
    obs_line_px = mask_invalid_observations(focal_obs_px, focal_valid)
    ax.plot(
        obs_line_px[:, 0], obs_line_px[:, 1], "-x", color="#F28E2B",
        lw=1.0, ms=2.8, mew=0.9, zorder=7,
    )

    if title:
        ax.set_title(title, fontsize=8, pad=3)
    ax.axis("off")
    if show_legend:
        ax.legend(
            handles=gast_legend_handles(), loc="upper left", ncol=1,
            fontsize=7, framealpha=0.9, handlelength=1.5, borderpad=0.4,
        )
    if owns_figure:
        fig.tight_layout(pad=0.3)
        fig.savefig(out_png, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def gast_legend_handles():
    return [
        Line2D([], [], color="#F28E2B", marker="x", lw=1.0, ms=3,
               label="Past trajectory"),
        Line2D([], [], color="#20A36A", marker="o", lw=1.0, ms=2,
               label="Ground truth"),
        Line2D([], [], color="#F2C230", marker="o", lw=1.2, ms=2,
               label="Best-of-20 prediction (oracle minFDE)"),
    ]


def render_gast_grid(panels, scenes, out_dir):
    """Create a 3-by-5 GAST qualitative grid."""
    fig, axes = plt.subplots(
        len(MISSING_LAYERS), len(scenes),
        figsize=(2.55 * len(scenes), 1.9 * len(MISSING_LAYERS)),
        dpi=220, squeeze=False,
    )
    for row, (layer, layer_label) in enumerate(MISSING_LAYERS):
        for col, scene in enumerate(scenes):
            ax = axes[row, col]
            entry = panels.get((scene, layer))
            if entry is None:
                ax.axis("off")
                ax.text(0.5, 0.5, "No sample", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
                continue
            sample, res = entry
            missing = int((~sample["x_valid_mask"][0].numpy()).sum())
            render_one(
                scene, sample, res, out_png=None, ax=ax, show_legend=False,
                title=SCENE_LABELS.get(scene, scene) if row == 0 else None,
                zoom_pad_px=90, view_aspect=2.55 / 1.9,
            )
            ax.set_gid(f"{scene}:{layer_label}:missing={missing}")
    fig.subplots_adjust(top=0.94, bottom=0.12, left=0.055, right=0.995,
                        wspace=0.025, hspace=0.06)
    fig.canvas.draw()
    for row, (_, label) in enumerate(MISSING_LAYERS):
        bbox = axes[row, 0].get_position()
        fig.text(0.018, (bbox.y0 + bbox.y1) / 2, label, rotation=90,
                 ha="center", va="center", fontsize=8)
    fig.legend(
        handles=gast_legend_handles(), loc="lower center", ncol=3,
        bbox_to_anchor=(0.53, 0.012), frameon=False, fontsize=8.5,
        handlelength=1.8, columnspacing=2.2,
    )
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"gast_trajectory_samples.{ext}",
                    dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mask_invalid_observations(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Set missing observation positions to NaN so plotted lines break there."""
    points = np.asarray(points, dtype=float).copy()
    valid = np.asarray(valid, dtype=bool)
    if points.shape[:-1] != valid.shape:
        raise ValueError(
            f"point shape {points.shape} and validity shape {valid.shape} do not align"
        )
    points[~valid] = np.nan
    return points


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", action="append", default=None,
                    help="可多次指定；默认全部五场景")
    ap.add_argument("--data-root", default="/home/lbh/TrajImpute/dataset/TrajImpute")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--per-layer", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--out-dir", default=str(REPO / "docs/figures/trajgap_qualitative_gast"))
    args = ap.parse_args()
    scenes = args.scenes or list(SCENE_CONF)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = {}

    for scene in scenes:
        ckpt = CKPTS[scene]
        dataset = TrajGapDataset(args.data_root, scene, "test")
        layers = stratify(dataset, per_layer=args.per_layer)
        print(f"[{scene}] test samples={len(dataset)}, layers="
              f"{ {k: len(v) for k, v in layers.items()} }")

        model = build_model(scene)
        all_idx = np.concatenate(list(layers.values()))
        results = forward_indices(
            model, dataset, all_idx, args.device, batch_size=args.batch_size
        )

        for lname, idxs in layers.items():
            (pick, pred_span, pred_arc, gt_span, gt_arc,
             obs_span, obs_arc, obs_run, eligible) = pick_visible_sample(
                results, idxs, scene
            )
            sample = dataset[int(pick)]
            res = results[int(pick)]
            out_png = out_dir / f"{scene}_{lname}_minFDE{res['minfde']:.2f}.png"
            render_one(scene, sample, res, out_png)
            panels[(scene, lname)] = (sample, res)
            print(f"  {lname}: idx={pick} visible_candidates={eligible}/{len(idxs)} "
                  f"pred_span={pred_span:.1f}px pred_arc={pred_arc:.1f}px "
                  f"gt_span={gt_span:.1f}px gt_arc={gt_arc:.1f}px "
                  f"obs_span={obs_span:.1f}px obs_arc={obs_arc:.1f}px "
                  f"obs_run={obs_run} frames "
                  f"-> {out_png.name}")
        del model
        torch.cuda.empty_cache()
    render_gast_grid(panels, scenes, out_dir)
    print(f"  -> {out_dir / 'gast_trajectory_samples.{png,pdf}'}")


# Backward-compatible aliases for existing visualization tests and callers.
MOFLOW_LAYERS = MISSING_LAYERS
moflow_legend_handles = gast_legend_handles
render_moflow_grid = render_gast_grid


if __name__ == "__main__":
    main()
