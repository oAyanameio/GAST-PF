"""证据—置信度探针（只读，不训练、不改模型）。

诊断问题：模型的 mode 置信度 pi 是否随「可用证据量」变化？

背景：`forecasting_module.cal_loss` 的 mode 分类目标是
`cross_entropy(pi, argmin_k ||y_hat_k - y||)`——同一个真实未来 y，在缺失
0 帧和缺失 7 帧时得到完全相同的监督标签。若模型因此学成「证据无关的
置信度」，则 pi 的熵、top-1 概率、mode 排序质量都应随缺失等级基本不变。

本脚本只读 checkpoint + Mixed-test，输出四组量：
  1. 按 missing_count 分组的 H(pi)、top-1 概率、top1-top2 margin；
  2. GT-best mode 在 pi 排序中的名次（排序质量）；
  3. 配对分析：TrajImpute 官方 test 把同一批行人按不同缺失数复制成多个
     块（Easy 5 块 / Hard 4 块），因此同一 track 在不同缺失等级下共享
     同一未来 y——直接测「同一个人、同一个未来，只改证据量时 pi 变了多少」；
  4. 熵—缺失等级的同人 Spearman 相关（单调性检验）。

用法（仓库根）：
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/审计与校验/probe_evidence_confidence.py \
        --scene ETH-M --checkpoint <path/to/epo.ckpt>

产物：outputs/grad_probe/evidence_confidence/<scene>_<variant>.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.datamodule.trajimpute_dataset import (  # noqa: E402
    TRAJIMPUTE_ROOT, TrajGapDataset, load_trajimpute_pkl, trajimpute_collate_fn,
)
from scripts.结果分析.evaluate_trajimpute_direct import (  # noqa: E402
    VARIANTS, build_model, load_checkpoint,
)


def block_size_of(scene, difficulty, data_root, split="test"):
    """官方 test 的复制块大小 = 块 0 的原始轨迹行数。

    官方 generate_data.py 用列表乘法复制 seq_start_end，块间逐条完全相同；
    这里按「n_orig 个条目为一组、逐块重复」检测 n_orig，块大小取其末行。
    """
    path = Path(data_root) / scene / difficulty / f"data_{split}.pkl"
    obs, _, _, sse = load_trajimpute_pkl(path, 8, 12)
    for n_orig in range(1, len(sse) + 1):
        if len(sse) % n_orig:
            continue
        base = sse[:n_orig]
        if all(sse[k:k + n_orig] == base
               for k in range(0, len(sse), n_orig)):
            block = int(base[-1][1])
            if block * (len(sse) // n_orig) == int(obs.shape[0]):
                return block
    return int(obs.shape[0])


def entropy(prob):
    p = prob.clamp_min(1e-12)
    return -(p * p.log()).sum(dim=-1)


@torch.no_grad()
def probe_scene(model, dataset, scene, variant, device, batch_size, data_root,
                max_batches=None):
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, collate_fn=trajimpute_collate_fn)
    n_easy = len(dataset.datasets[0])
    rows = []
    n_batches = 0
    for batch in loader:
        dev = {k: (v.to(device) if torch.is_tensor(v) else v)
               for k, v in batch.items()}
        out = model(dev)
        # 预测源：确定性分量均值优先（SocialLSTM 的 y_hat 是采样值，会引入随机性）
        if out.get("mu") is not None:
            pred = out["mu"]
        elif out.get("new_y_hat") is not None:
            pred = out["new_y_hat"]
        else:
            pred = out["y_hat"]
        # pi 语义随实现不同：MoFlow 族是未归一化 logits（evaluator 只用来排序），
        # SocialLSTM 的 GMM 已 softmax 成概率。按行和判定，避免二次 softmax。
        logits = out["new_pi"] if out.get("new_pi") is not None else out["pi"]
        pred = pred[..., :2].float()
        logits = logits.float()
        row_sum = logits.sum(dim=-1)
        if torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-3):
            prob = logits
            logits = torch.log(prob.clamp_min(1e-12))
        else:
            prob = torch.softmax(logits, dim=-1)
        target = dev["target"][:, 0]  # focal [B, T, 2]

        # 逐 mode 终点误差 -> best mode / 排序名次
        fde = torch.norm(pred[..., -1, :] - target[:, None, -1, :], dim=-1)  # [B,K]
        ade = torch.norm(pred - target[:, None], dim=-1).mean(-1)           # [B,K]
        best = fde.argmin(dim=-1)                                            # [B]
        top1 = prob.argmax(dim=-1)
        order = prob.argsort(dim=-1, descending=True)
        rank_of_best = (order == best[:, None]).float().argmax(dim=-1)       # [B]

        ent = entropy(prob)
        srt = prob.sort(dim=-1, descending=True).values
        top1_p = srt[:, 0]
        margin = srt[:, 0] - srt[:, 1]
        top1_ok = (top1 == best)
        logit_std = logits.std(dim=-1)
        logit_range = logits.max(dim=-1).values - logits.min(dim=-1).values
        # 随机抽 mode 的期望误差：若 pi 无信息，FDE@1 应接近这个值
        fde_uniform = fde.mean(dim=-1)

        focal_valid = batch["x_valid_mask"][:, 0]
        valid_count = focal_valid.sum(-1)
        anchor_lag = batch["x_anchor_lag_steps"][:, 0]

        for i in range(pred.shape[0]):
            idx = len(rows)
            rows.append({
                "source": "Easy" if idx < n_easy else "Hard",
                "track": int(batch["track_id"][i]) if "track_id" in batch else -1,
                "missing_count": int(batch["missing_count"][i]),
                "valid_count": int(valid_count[i]),
                "anchor_lag": int(anchor_lag[i]),
                "entropy": float(ent[i]),
                "top1_prob": float(top1_p[i]),
                "margin": float(margin[i]),
                "logit_std": float(logit_std[i]),
                "logit_range": float(logit_range[i]),
                "rank_of_best": int(rank_of_best[i]),
                "top1_correct": bool(top1_ok[i]),
                "minFDE20": float(fde[i].min()),
                "minADE20": float(ade[i].min()),
                "FDE@1": float(fde[i, top1[i]]),
                "ADE@1": float(ade[i, top1[i]]),
                "FDE_uniform": float(fde_uniform[i]),
                "FDE_best": float(fde[i, best[i]]),
            })
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    return rows


def summarize(rows, K):
    def agg(sub):
        if not sub:
            return None
        return {
            "n": len(sub),
            "entropy": float(np.mean([r["entropy"] for r in sub])),
            "entropy_norm": float(np.mean([r["entropy"] for r in sub]) / np.log(K)),
            "top1_prob": float(np.mean([r["top1_prob"] for r in sub])),
            "margin": float(np.mean([r["margin"] for r in sub])),
            "logit_std": float(np.mean([r["logit_std"] for r in sub])),
            "logit_range": float(np.mean([r["logit_range"] for r in sub])),
            "rank_of_best": float(np.mean([r["rank_of_best"] for r in sub])),
            "top1_acc": float(np.mean([r["top1_correct"] for r in sub])),
            "minFDE20": float(np.mean([r["minFDE20"] for r in sub])),
            "minADE20": float(np.mean([r["minADE20"] for r in sub])),
            "FDE@1": float(np.mean([r["FDE@1"] for r in sub])),
            "ADE@1": float(np.mean([r["ADE@1"] for r in sub])),
            "FDE_uniform": float(np.mean([r["FDE_uniform"] for r in sub])),
            "FDE_best": float(np.mean([r["FDE_best"] for r in sub])),
        }

    by_missing = {}
    for m in sorted({r["missing_count"] for r in rows}):
        by_missing[m] = agg([r for r in rows if r["missing_count"] == m])
    return {"overall": agg(rows), "by_missing": by_missing}


def paired_analysis(rows, block_size):
    """同一 track 在不同缺失等级下共享同一未来 y：配对测 pi 的变化。

    track_id 是 pkl 全局行号；官方 test 把块 0 的轨迹复制成 n_blocks 块，
    所以 person = track_id % block_size。
    """
    groups = {}
    for r in rows:
        key = r["track"] % block_size
        groups.setdefault(key, []).append(r)

    pairs = []
    within_corr = []
    for key, g in groups.items():
        if len(g) < 2:
            continue
        g = sorted(g, key=lambda x: x["missing_count"])
        base = g[0]
        for r in g[1:]:
            pairs.append({
                "delta_missing": r["missing_count"] - base["missing_count"],
                "delta_entropy": r["entropy"] - base["entropy"],
                "delta_top1_prob": r["top1_prob"] - base["top1_prob"],
                "delta_margin": r["margin"] - base["margin"],
                "delta_rank": r["rank_of_best"] - base["rank_of_best"],
                "delta_FDE1": r["FDE@1"] - base["FDE@1"],
                "delta_minFDE": r["minFDE20"] - base["minFDE20"],
            })
        ms = [x["missing_count"] for x in g]
        es = [x["entropy"] for x in g]
        if len(set(ms)) > 1 and np.std(es) > 0:
            # Spearman via ranks
            rm = np.argsort(np.argsort(ms)).astype(float)
            re = np.argsort(np.argsort(es)).astype(float)
            within_corr.append(float(np.corrcoef(rm, re)[0, 1]))

    def agg_pairs(sub):
        if not sub:
            return None
        return {k: float(np.mean([p[k] for p in sub])) for k in sub[0]}

    out = {
        "n_pairs": len(pairs),
        "n_groups": len(groups),
        "all": agg_pairs(pairs),
        "by_delta_missing": {},
        "within_person_spearman_missing_vs_entropy": {
            "mean": float(np.mean(within_corr)) if within_corr else None,
            "median": float(np.median(within_corr)) if within_corr else None,
            "frac_positive": float(np.mean([c > 0 for c in within_corr])) if within_corr else None,
            "n_groups": len(within_corr),
        },
    }
    for d in sorted({p["delta_missing"] for p in pairs}):
        out["by_delta_missing"][d] = agg_pairs(
            [p for p in pairs if p["delta_missing"] == d])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=TRAJIMPUTE_ROOT)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--variant", default="M0", choices=list(VARIANTS))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-root", default="outputs/grad_probe/evidence_confidence")
    args = ap.parse_args()

    dataset = TrajGapDataset(args.data_root, args.scene, args.split)
    model = build_model(args.variant, args.K)
    missing, unexpected = load_checkpoint(model, args.checkpoint)
    if missing or unexpected:
        raise SystemExit(f"checkpoint 不匹配: {missing} {unexpected}")

    block = block_size_of(args.scene, "Easy", args.data_root, args.split)
    rows = probe_scene(model, dataset, args.scene, args.variant, args.device,
                       args.batch_size, args.data_root, args.max_batches)

    result = {
        "scene": args.scene,
        "variant": args.variant,
        "split": args.split,
        "K": args.K,
        "checkpoint": str(args.checkpoint),
        "block_size": block,
        "summary": summarize(rows, args.K),
        "paired": paired_analysis(rows, block),
    }
    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.scene}_{args.variant}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\n=== {args.scene} / {args.variant} ===")
    print(f"{'missing':>7} {'n':>7} {'H(pi)':>7} {'H/lnK':>6} {'top1p':>6} "
          f"{'margin':>7} {'logitSD':>7} {'rank*':>6} {'acc':>6} {'minFDE':>7} {'FDE@1':>7}")
    for m, a in result["summary"]["by_missing"].items():
        print(f"{m:>7} {a['n']:>7} {a['entropy']:>7.3f} {a['entropy_norm']:>6.3f} "
              f"{a['top1_prob']:>6.3f} {a['margin']:>7.4f} {a['logit_std']:>7.4f} "
              f"{a['rank_of_best']:>6.2f} "
              f"{a['top1_acc']:>6.3f} {a['minFDE20']:>7.4f} {a['FDE@1']:>7.4f}")
    p = result["paired"]
    print(f"\npaired: {p['n_pairs']} pairs / {p['n_groups']} person-groups")
    if p["all"]:
        print("  Δ per missing step: "
              + " ".join(f"{k}={v:+.4f}" for k, v in p["all"].items()))
    ws = p["within_person_spearman_missing_vs_entropy"]
    print(f"  within-person Spearman(missing, entropy): mean={ws['mean']} "
          f"frac>0={ws['frac_positive']} n={ws['n_groups']}")
    print(f"\n[results] {out_path}")


if __name__ == "__main__":
    main()
