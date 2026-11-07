"""TrajGap 定性可视化：把 GAST 模型输出轨迹投到 ETH/UCY 原版场景图上。

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
  每场景按 focal missing_count 分 4 层（0 / 1-2 / 3-4 / >=5），每层抽 128 样本
  前向，取 minFDE20 最接近该层中位数的样本。即"各缺失条件典型样本"，非最好看样本。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]

# 必须在 import src 之前：优先使用正式轮次快照代码（revision ad270824，
# 与 checkpoint 一致）；当前仓库 src 已重构分叉，strict 加载会失败。
SNAPSHOT_ROOT = REPO / "outputs/trajgap_gast_goal_3gpu_retry/gpu0/GAST-Goal_ETH-M_mixed-direct_seed2024_uni/train"
sys.path.insert(0, str(SNAPSHOT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.datamodule.trajimpute_dataset import (  # noqa: E402
    TrajGapDataset, trajimpute_collate_fn,
)
from src.model.trajectory_forecaster import TrajectoryForecaster  # noqa: E402

# ---------------------------------------------------------------- 场景映射

MAPS_DIR = REPO / "docs/figures/scene_maps"

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

CKPTS = {
    "ETH-M":   "outputs/trajgap_gast_goal_3gpu_retry/gpu0/GAST-Goal_ETH-M_mixed-direct_seed2024_uni/train/checkpoints/epoch=58.ckpt",
    "HOTEL-M": "outputs/trajgap_gast_goal_3gpu_retry/gpu0/GAST-Goal_HOTEL-M_mixed-direct_seed2024_uni/train/checkpoints/epoch=41.ckpt",
    "UNIV-M":  "outputs/trajgap_gast_goal_3gpu_retry/gpu1/GAST-Goal_UNIV-M_mixed-direct_seed2024_uni/train/checkpoints/epoch=42.ckpt",
    "ZARA1-M": "outputs/trajgap_gast_goal_3gpu_retry/gpu2/GAST-Goal_ZARA1-M_mixed-direct_seed2024_uni/train/checkpoints/epoch=33.ckpt",
    "ZARA2-M": "outputs/trajgap_gast_goal_3gpu_retry/gpu1_zara2/GAST-Goal_ZARA2-M_mixed-direct_seed2024_uni/train/checkpoints/epoch=49.ckpt",
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

# 正式轮次 manifest 记录的完整模型配置（git_revision ad270824）
GAST_MODEL_CONFIG = dict(
    embed_dim=128, future_steps=12, num_heads=8, mlp_ratio=4.0,
    qkv_bias=False, drop_path=0.2, num_actor_types=1, num_modes=20,
    dt=0.4, obs_len=8, bimamba=False, encoder_type="gast",
    gast_depth=4, gast_gsm_operator="max",
    gast_temporal_direction="unidirectional", gast_goal=True,
    goal_endpoint_loss_weight=1.0, mode_diversity_loss_weight=0.05,
    mode_diversity_margin=0.5, loss_mode="final_consistent",
    gast_oer_spatial=True, gast_emc_temporal=True, gast_ewhr_readout=True,
)


def build_model(scene: str):
    """用快照代码（revision ad270824，与 checkpoint 一致）构建并加载模型。"""
    import src.model.trajectory_forecaster as tf_snap
    model = tf_snap.TrajectoryForecaster(**GAST_MODEL_CONFIG)
    ckpt = torch.load(CKPTS[scene], map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    cleaned = {k[len("net."):]: v for k, v in state.items() if k.startswith("net.")}
    model.load_state_dict(cleaned, strict=True)
    return model


# ---------------------------------------------------------------- 选样与前向

def stratify(dataset, per_layer=128, seed=2024):
    """missing_count 分 4 层，每层随机抽 per_layer 个索引。"""
    rng = np.random.default_rng(seed)
    mc = dataset.missing_counts.numpy() if torch.is_tensor(dataset.missing_counts) \
        else np.asarray(dataset.missing_counts)
    layers = [
        ("m0", np.where(mc == 0)[0]),
        ("m12", np.where((mc >= 1) & (mc <= 2))[0]),
        ("m34", np.where((mc >= 3) & (mc <= 4))[0]),
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
def forward_indices(model, dataset, indices, device, batch_size=16):
    """对给定索引跑前向，返回 {idx: (pred_world(K,12,2), pi(K), minfde, top1)}。"""
    model = model.to(device).eval()
    from torch.utils.data import DataLoader
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
        for j, idx in enumerate(chunk):
            pw = local2world(pred[j], origin[j], float(theta[j]))
            tw = local2world(target[j][None], origin[j], float(theta[j]))[0]
            fde = np.linalg.norm(pw[:, -1] - tw[-1], axis=-1)  # (K,)
            results[int(idx)] = dict(
                pred_world=pw, gt_world=tw, pi=pi[j],
                minfde=float(fde.min()), top1=int(np.argmax(pi[j])),
            )
    return results


def pick_typical(results, indices):
    """取 minFDE20 最接近层内中位数的样本。"""
    vals = np.array([results[i]["minfde"] for i in indices])
    med = np.median(vals)
    return indices[int(np.argmin(np.abs(vals - med)))], med


# ---------------------------------------------------------------- 绘图

COL_GT = "#111111"
COL_OBS = "#0072B2"
COL_BEST = "#D55E00"
COL_TOP1 = "#009E73"
COL_CAND = "#B0B0B0"


def render_one(scene, sample, res, out_png, zoom_pad_px=120):
    """sample: dataset[i]（含 x_positions 局部 + origin/theta + target）。"""
    img = Image.open(MAPS_DIR / SCENE_CONF[scene]["img"])
    W, H = img.size

    origin = sample["origin"].numpy()                   # (1,2)
    theta = float(sample["theta"])
    # 观测世界坐标：局部 -> 世界（x_positions 已是局部）
    obs_local = sample["x_positions"].numpy()           # (A,8,2)
    rot = np.array([[np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)]])
    obs_world = obs_local @ rot.T + origin.reshape(1, 1, 2)  # origin (1,2)
    valid = sample["x_valid_mask"].numpy()              # (A,8)
    gt_world = res["gt_world"]                          # (12,2) focal GT 世界坐标
    pred_world = res["pred_world"]                      # (20,12,2)
    pi = res["pi"]
    best = int(np.argmin(np.linalg.norm(pred_world[:, -1] - gt_world[-1], axis=-1)))
    top1 = res["top1"]

    # 全部像素坐标
    focal_obs_px = world2pix(scene, obs_world[0])
    focal_valid = valid[0]
    gt_px = world2pix(scene, gt_world)
    pred_px = world2pix(scene, pred_world.reshape(-1, 2)).reshape(20, 12, 2)

    # 显示窗口
    pts = np.concatenate([focal_obs_px[focal_valid], gt_px, pred_px.reshape(-1, 2)])
    u0, u1 = pts[:, 0].min() - zoom_pad_px, pts[:, 0].max() + zoom_pad_px
    v0, v1 = pts[:, 1].min() - zoom_pad_px, pts[:, 1].max() + zoom_pad_px
    u0, v0 = max(u0, 0), max(v0, 0)
    u1, v1 = min(u1, W), min(v1, H)
    if u1 - u0 < 200: u1 = min(u0 + 200, W)
    if v1 - v0 < 200: v1 = min(v0 + 200, H)

    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=200)
    ax.imshow(img)
    ax.set_xlim(u0, u1)
    ax.set_ylim(v1, v0)
    ax.axis("off")

    # K=20 候选（细灰）
    for k in range(pred_px.shape[0]):
        if k in (best, top1):
            continue
        ax.plot(pred_px[k, :, 0], pred_px[k, :, 1], color=COL_CAND,
                lw=0.7, alpha=0.45, zorder=2)
    # top-1（概率最大）
    ax.plot(pred_px[top1, :, 0], pred_px[top1, :, 1], color=COL_TOP1,
            lw=1.7, alpha=0.9, zorder=4, label=f"top-1 (p={pi[top1]:.2f})")
    # best-of-K
    ax.plot(pred_px[best, :, 0], pred_px[best, :, 1], color=COL_BEST,
            lw=2.2, zorder=4, label=f"best-of-20 (FDE={res['minfde']:.2f}m)")
    # GT
    ax.plot(gt_px[:, 0], gt_px[:, 1], "--", color=COL_GT, lw=1.6,
            zorder=5, dashes=(5, 3), label="ground truth")
    ax.plot(gt_px[-1, 0], gt_px[-1, 1], "*", color=COL_GT, ms=9, zorder=6)
    # 观测（有效帧实线；缺口断开）
    op = focal_obs_px[focal_valid]
    ax.plot(op[:, 0], op[:, 1], "-", color=COL_OBS, lw=2.2, zorder=5,
            label="observed")
    ax.plot(op[:, 0], op[:, 1], "o", color=COL_OBS, ms=2.5, zorder=6)
    ax.plot(op[-1, 0], op[-1, 1], "s", color=COL_OBS, ms=7, zorder=6)

    ax.legend(loc="upper left", fontsize=6.5, framealpha=0.85,
              handlelength=1.6, borderpad=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", action="append", default=None,
                    help="可多次指定；默认全部五场景")
    ap.add_argument("--data-root", default="/home/lbh/TrajImpute/dataset/TrajImpute")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--per-layer", type=int, default=128)
    ap.add_argument("--out-dir", default=str(REPO / "docs/figures/trajgap_qualitative"))
    args = ap.parse_args()
    scenes = args.scenes or list(SCENE_CONF)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for scene in scenes:
        ckpt = CKPTS[scene]
        dataset = TrajGapDataset(args.data_root, scene, "test")
        layers = stratify(dataset, per_layer=args.per_layer)
        print(f"[{scene}] test samples={len(dataset)}, layers="
              f"{ {k: len(v) for k, v in layers.items()} }")

        model = build_model(scene)
        all_idx = np.concatenate(list(layers.values()))
        results = forward_indices(model, dataset, all_idx, args.device)

        for lname, idxs in layers.items():
            pick, med = pick_typical(results, idxs)
            sample = dataset[int(pick)]
            res = results[int(pick)]
            out_png = out_dir / f"{scene}_{lname}_minFDE{res['minfde']:.2f}.png"
            render_one(scene, sample, res, out_png)
            print(f"  {lname}: idx={pick} layer-median minFDE20={med:.3f} "
                  f"-> {out_png.name}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
