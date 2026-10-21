"""证据—置信度探针的对照臂：NLL 训练的 GMM（MaskedSocialLSTM）。

动机：主探针发现 MoFlow 族（WTA + 交叉熵训练 mode 分类）的 mode 后验
在 m=0..6 上几乎不变（H/lnK≈0.96，margin≈0.015），且与缺失等级反相关。
本脚本用**同为 K=20、同数据、同协议、但用混合密度 NLL 训练**的
MaskedSocialLSTM 做对照，检验该病理是否与 WTA 监督绑定。

用法（仓库根）：
    PYTHONPATH=. python scripts/审计与校验/probe_evidence_confidence_nll.py \
        --scene ETH-M --checkpoint <snapshot>/train/checkpoints/epo*.ckpt \
        --arch-root <snapshot>/train
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.datamodule.trajimpute_dataset import (  # noqa: E402
    TRAJIMPUTE_ROOT, TrajGapDataset, trajimpute_collate_fn,
)
from scripts.审计与校验.probe_evidence_confidence import (  # noqa: E402
    block_size_of, paired_analysis, probe_scene, summarize,
)


def load_social_lstm(arch_root):
    path = Path(arch_root) / "src" / "model" / "social_lstm.py"
    spec = importlib.util.spec_from_file_location("social_lstm_arch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MaskedSocialLSTM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=TRAJIMPUTE_ROOT)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--arch-root", required=True,
                    help="归档快照的 train/ 目录（含 src/model/social_lstm.py）")
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-root", default="outputs/grad_probe/evidence_confidence")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"]["model"])
    hp.pop("type", None)
    cls = load_social_lstm(args.arch_root)
    model = cls(**hp)
    state = {k[len("net."):]: v for k, v in ckpt["state_dict"].items()
             if k.startswith("net.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint 不匹配: {missing[:5]} {unexpected[:5]}")

    dataset = TrajGapDataset(args.data_root, args.scene, "test")
    block = block_size_of(args.scene, "Easy", args.data_root, "test")
    rows = probe_scene(model, dataset, args.scene, "SocialLSTM-NLL",
                       args.device, args.batch_size, args.data_root)

    result = {
        "scene": args.scene, "variant": "SocialLSTM-NLL", "split": "test",
        "K": args.K, "checkpoint": str(args.checkpoint), "block_size": block,
        "summary": summarize(rows, args.K),
        "paired": paired_analysis(rows, block),
    }
    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.scene}_SocialLSTM-NLL.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\n=== {args.scene} / SocialLSTM-NLL ===")
    print(f"{'missing':>7} {'n':>7} {'H/lnK':>6} {'top1p':>6} {'margin':>7} "
          f"{'rank*':>6} {'acc':>6} {'minFDE':>7} {'FDE@1':>7} {'FDEunif':>7}")
    for m, a in result["summary"]["by_missing"].items():
        print(f"{m:>7} {a['n']:>7} {a['entropy_norm']:>6.3f} {a['top1_prob']:>6.3f} "
              f"{a['margin']:>7.4f} {a['rank_of_best']:>6.2f} {a['top1_acc']:>6.3f} "
              f"{a['minFDE20']:>7.4f} {a['FDE@1']:>7.4f} {a['FDE_uniform']:>7.4f}")
    ws = result["paired"]["within_person_spearman_missing_vs_entropy"]
    print(f"  Spearman(miss,H)={ws['mean']:.3f} frac>0={ws['frac_positive']:.3f}")
    print(f"\n[results] {out_path}")


if __name__ == "__main__":
    main()
