"""梯度探针：从训练好的 checkpoint 取真实 batch，统计分层梯度范数。

用途：比较不同变体的梯度流是否健康——某个新增模块是否真的在接收梯度、
是否存在梯度消失/爆炸、门控是否饱和。

用法（仓库根）：
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python scripts/审计与校验/probe_gradients.py
"""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.datamodule.trajimpute_dataset import (  # noqa: E402
    TRAJIMPUTE_ROOT, TrajGapDataset, trajimpute_collate_fn,
)
from src.model.forecasting_module import ForecastingLightningModule  # noqa: E402


def build_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    # checkpoint 里的 model 是 OmegaConf DictConfig，且不含构造所需的 type 字段
    from omegaconf import OmegaConf
    model_cfg = OmegaConf.to_container(hp["model"], resolve=True)
    model_cfg.setdefault("type", "TrajectoryForecaster")
    hp = dict(hp)
    hp["model"] = model_cfg
    module = ForecastingLightningModule(**hp)
    state = {k[len("net."):]: v for k, v in ckpt["state_dict"].items()
             if k.startswith("net.")}
    missing, unexpected = module.net.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state_dict 不匹配: missing={missing[:5]} unexpected={unexpected[:5]}")
    module.to(device).train()
    return module, hp["model"]


def disable_stochastic_depth(module):
    """关闭 dropout / drop-path：随机失活会让某些层在单个 batch 上梯度恰为 0，
    干扰跨模型比较。这里只关随机性，不动权重与反向图。"""
    import torch.nn as nn
    n_drop = n_dp = 0
    for m in module.net.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.p = 0.0
            n_drop += 1
        if hasattr(m, "drop_prob"):
            m.drop_prob = 0.0
            n_dp += 1
    return n_drop, n_dp


def grad_table(module):
    """按参数名收集梯度范数（只保留有梯度的）。"""
    rows = []
    for name, p in module.net.named_parameters():
        if p.grad is None:
            rows.append({"name": name, "grad_norm": 0.0, "param_norm": float(p.norm()), "has_grad": False})
        else:
            rows.append({"name": name, "grad_norm": float(p.grad.norm()),
                         "param_norm": float(p.norm()), "has_grad": True})
    return rows


def group_of(name):
    """把参数名归到可读的模块组。"""
    parts = name.split(".")
    for key in ("directional_readout", "middle_readout_token", "spatial", "temporal",
                "temporal_decay", "anchor_geo_projection", "history_input_projection"):
        if key in parts:
            return f"gast_encoder.{key}" if key != "history_input_projection" else key
    if parts[0] == "gast_encoder":
        return "gast_encoder.other"
    return parts[0]


def probe(ckpt, scene, variant, device, n_batch, no_drop=False):
    module, model_cfg = build_from_ckpt(ckpt, device)
    if no_drop:
        disable_stochastic_depth(module)
    ds = TrajGapDataset(TRAJIMPUTE_ROOT, scene, "train")
    module.zero_grad(set_to_none=True)
    losses = []
    for i in range(n_batch):
        batch = trajimpute_collate_fn([ds[i]])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        loss = module.training_step(batch, i)
        loss.backward()
        losses.append(float(loss))
    rows = grad_table(module)
    # 聚合到模块组
    groups = {}
    for r in rows:
        g = group_of(r["name"])
        cur = groups.setdefault(g, {"grad_norm_sq": 0.0, "param_norm_sq": 0.0, "n": 0, "n_no_grad": 0})
        cur["grad_norm_sq"] += r["grad_norm"] ** 2
        cur["param_norm_sq"] += r["param_norm"] ** 2
        cur["n"] += 1
        if not r["has_grad"]:
            cur["n_no_grad"] += 1
    for g, v in groups.items():
        v["grad_norm"] = v.pop("grad_norm_sq") ** 0.5
        v["param_norm"] = v.pop("param_norm_sq") ** 0.5
    total = sum(v["grad_norm"] ** 2 for v in groups.values()) ** 0.5
    return {
        "variant": variant, "scene": scene, "ckpt": str(ckpt),
        "model_cfg": model_cfg, "loss": losses,
        "total_grad_norm": total,
        "groups": dict(sorted(groups.items(), key=lambda kv: -kv[1]["grad_norm"])),
        "params": sorted(rows, key=lambda r: -r["grad_norm"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="ETH-M")
    ap.add_argument("--n-batch", type=int, default=1)
    ap.add_argument("--no-drop", action="store_true",
                    help="关闭 dropout / drop-path，消除单 batch 随机失活导致的零梯度")
    ap.add_argument("--out", default="outputs/grad_probe/gradients.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = {
        "GAST-PF-DAP": "outputs/gast_dap_seed2024/"
                       f"GAST-PF-DAP_{args.scene}_mixed-direct_seed2024_uni/train/checkpoints/last.ckpt",
        "GAST-PF": "outputs/unitraj_encoder_single_scene_seed2024/"
                   f"UT-ENC_{args.scene}_mixed-direct_seed2024_uni/train/checkpoints/last.ckpt",
    }
    results = {}
    for variant, path in ckpts.items():
        p = Path(path)
        if not p.is_file():
            print(f"[skip] {variant}: 无 checkpoint {p}", flush=True)
            continue
        print(f"[probe] {variant} <- {p}", flush=True)
        results[variant] = probe(p, args.scene, variant, device, args.n_batch, args.no_drop)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n写入 {out}", flush=True)
    for variant, r in results.items():
        print(f"\n=== {variant} · {r['scene']} · loss={r['loss'][0]:.4f} · 总梯度范数={r['total_grad_norm']:.4f}")
        for g, v in list(r["groups"].items())[:12]:
            print(f"  {g:34s} grad={v['grad_norm']:10.4f}  param={v['param_norm']:9.3f}  参数数={v['n']:3d} 无梯度={v['n_no_grad']}")


if __name__ == "__main__":
    main()
