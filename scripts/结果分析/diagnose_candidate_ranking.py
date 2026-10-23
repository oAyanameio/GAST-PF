"""零训练诊断：候选覆盖 vs mode 排序分解（M0 vs 对比臂）。

问题：某臂相对 M0 的 minFDE20 劣化，来自候选集质量（oracle 覆盖塌了）
还是 mode 排序（top-1 选错）？本脚本不训练，只加载既有 checkpoint 在
Mixed-test 上前向，按 missing_count 分组输出：

  minFDE20   oracle 覆盖（候选集质量）
  FDE@1      排序质量（最高 pi 的 mode）
  regret     FDE@1 − minFDE20（排序代价；越小越好）
  spread     K 条轨迹末点的平均两两距离（候选多样性）
  pi_entropy pi 分布熵（排序信心）

用法：
  PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/结果分析/diagnose_candidate_ranking.py \
      --scene ETH-M --base-variant M0 --base-ckpt <m0.ckpt> \
      --variant GAST-PF-OEHSR --ckpt <oehsr.ckpt> --device cuda:1
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.datamodule.trajimpute_dataset import (  # noqa: E402
    TrajGapDataset, trajimpute_collate_fn,
)
# 复用正式评估入口的模型构建与 checkpoint 契约校验
sys.path.insert(0, str(REPO / "scripts" / "结果分析"))
from evaluate_trajimpute_direct import build_model, load_checkpoint  # noqa: E402


@torch.no_grad()
def collect(variant, ckpt, scene, data_root, K, device, batch_size=64):
    from torch.utils.data import DataLoader
    model = build_model(variant, K)
    load_checkpoint(model, ckpt)
    model = model.to(device).eval()
    ds = TrajGapDataset(data_root, scene, "test")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, collate_fn=trajimpute_collate_fn)
    out = defaultdict(lambda: defaultdict(list))
    for batch in loader:
        dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        o = model(dev)
        pred = o["new_y_hat"] if o.get("new_y_hat") is not None else o["y_hat"]
        prob = o["new_pi"] if o.get("new_pi") is not None else o["pi"]
        pred, prob = pred[..., :2].float(), prob.float()
        tgt = dev["target"][:, 0]
        d = torch.norm(pred - tgt.unsqueeze(1), p=2, dim=-1)  # [B,K,T]
        ade, fde = d.mean(-1), d[..., -1]
        order = torch.argsort(prob, dim=-1, descending=True)
        top1 = order[:, 0]
        b = torch.arange(pred.shape[0], device=device)
        fde1 = fde[b, top1]
        minfde = fde.min(-1).values
        # spread: 末点两两距离均值
        end = pred[..., -1, :]  # [B,K,2]
        pd = torch.cdist(end, end)  # [B,K,K]
        K_ = end.shape[1]
        spread = pd.sum(dim=(1, 2)) / (K_ * (K_ - 1))
        # pi entropy (softmax over modes)
        p = torch.softmax(prob, dim=-1)
        ent = -(p * (p + 1e-12).log()).sum(-1)
        for i in range(pred.shape[0]):
            g = int(batch["missing_count"][i])
            rec = out[g]
            rec["minFDE20"].append(minfde[i].item())
            rec["FDE@1"].append(fde1[i].item())
            rec["spread"].append(spread[i].item())
            rec["entropy"].append(ent[i].item())
        out["_all"]["minFDE20"].extend(minfde.tolist())
        out["_all"]["FDE@1"].extend(fde1.tolist())
        out["_all"]["spread"].extend(spread.tolist())
        out["_all"]["entropy"].extend(ent.tolist())
    return dict(out)


def agg(vals):
    return sum(vals) / max(1, len(vals))


def report(name, data):
    print(f"\n=== {name} ===")
    print(f"{'missing':>7} {'n':>6} {'minFDE20':>9} {'FDE@1':>8} {'regret':>8} {'spread':>8} {'entropy':>8}")
    for g in sorted(k for k in data if k != "_all"):
        r = data[g]
        print(f"{g:>7} {len(r['minFDE20']):>6} {agg(r['minFDE20']):>9.4f} "
              f"{agg(r['FDE@1']):>8.4f} {agg(r['FDE@1'])-agg(r['minFDE20']):>8.4f} "
              f"{agg(r['spread']):>8.4f} {agg(r['entropy']):>8.4f}")
    r = data["_all"]
    print(f"{'ALL':>7} {len(r['minFDE20']):>6} {agg(r['minFDE20']):>9.4f} "
          f"{agg(r['FDE@1']):>8.4f} {agg(r['FDE@1'])-agg(r['minFDE20']):>8.4f} "
          f"{agg(r['spread']):>8.4f} {agg(r['entropy']):>8.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/lbh/TrajImpute/dataset/TrajImpute")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--base-variant", default="M0")
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    base = collect(args.base_variant, args.base_ckpt, args.scene,
                   args.data_root, args.K, args.device, args.batch_size)
    arm = collect(args.variant, args.ckpt, args.scene,
                  args.data_root, args.K, args.device, args.batch_size)
    report(f"M0 ({args.scene})", base)
    report(f"{args.variant} ({args.scene})", arm)

    print(f"\n=== Δ (arm − M0) {args.scene} ===")
    print(f"{'missing':>7} {'ΔminFDE20':>10} {'ΔFDE@1':>9} {'Δregret':>9} {'Δspread':>9} {'Δentropy':>10}")
    for g in sorted(k for k in base if k != "_all"):
        rb, ra = base[g], arm[g]
        if not rb or not ra:
            continue
        dm = agg(ra["minFDE20"]) - agg(rb["minFDE20"])
        df = agg(ra["FDE@1"]) - agg(rb["FDE@1"])
        dreg = (agg(ra["FDE@1"]) - agg(ra["minFDE20"])) - (agg(rb["FDE@1"]) - agg(rb["minFDE20"]))
        dsp = agg(ra["spread"]) - agg(rb["spread"])
        de = agg(ra["entropy"]) - agg(rb["entropy"])
        print(f"{g:>7} {dm:>+10.4f} {df:>+9.4f} {dreg:>+9.4f} {dsp:>+9.4f} {de:>+10.4f}")


if __name__ == "__main__":
    main()
