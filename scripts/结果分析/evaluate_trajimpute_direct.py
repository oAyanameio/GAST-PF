"""TrajGap-Bench 直接缺失预测评估入口。

正式协议固定为 `--difficulty Mixed`：
  PYTHONPATH=. python scripts/结果分析/evaluate_trajimpute_direct.py \
      --checkpoint outputs/.../checkpoints/xxx.ckpt \
      --scene ETH-M --difficulty Mixed --split test --K 20 --variant GAST

输出：outputs/trajgap_bench/<scene>_Mixed_<split>_<variant>_seed<seed>/results.json
层级：scene/split/missing_count/variant/seed；K 显式写入 meta；
外部参考数字（TrajImpute 原论文）只能以 external_reference 单独存放，不混入。
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
BENCHMARK_NAME = "TrajGap-Bench"

from src.datamodule.trajimpute_dataset import (  # noqa: E402
    MIXED_DIFFICULTY, TrajGapDataset, trajimpute_collate_fn,
    inspect_clean_source,
)
from src.evaluation.trajimpute_direct import DirectEvaluator, save_results  # noqa: E402

VARIANTS = {
    "GAST": {
        "gast_depth": 4,
        "gast_gsm_operator": "max",
        "goal_endpoint_loss_weight": 1.0,
        "mode_diversity_loss_weight": 0.05,
        "mode_diversity_margin": 0.5,
        "use_relation_bias": True,
        "use_source_filter": True,
        "use_reliability_modulation": True,
        "use_reliability_readout": True,
        "use_goal_guidance": True,
    },
    "GAST-no-relation-bias": {
        "gast_depth": 4, "gast_gsm_operator": "max",
        "goal_endpoint_loss_weight": 1.0,
        "mode_diversity_loss_weight": 0.05, "mode_diversity_margin": 0.5,
        "use_relation_bias": False, "use_source_filter": True,
        "use_reliability_modulation": True, "use_reliability_readout": True,
        "use_goal_guidance": True,
    },
    "GAST-no-source-filter": {
        "gast_depth": 4, "gast_gsm_operator": "max",
        "goal_endpoint_loss_weight": 1.0,
        "mode_diversity_loss_weight": 0.05, "mode_diversity_margin": 0.5,
        "use_relation_bias": True, "use_source_filter": False,
        "use_reliability_modulation": True, "use_reliability_readout": True,
        "use_goal_guidance": True,
    },
    "GAST-no-reliability-modulation": {
        "gast_depth": 4, "gast_gsm_operator": "max",
        "goal_endpoint_loss_weight": 1.0,
        "mode_diversity_loss_weight": 0.05, "mode_diversity_margin": 0.5,
        "use_relation_bias": True, "use_source_filter": True,
        "use_reliability_modulation": False, "use_reliability_readout": True,
        "use_goal_guidance": True,
    },
    "GAST-content-only-readout": {
        "gast_depth": 4, "gast_gsm_operator": "max",
        "goal_endpoint_loss_weight": 1.0,
        "mode_diversity_loss_weight": 0.05, "mode_diversity_margin": 0.5,
        "use_relation_bias": True, "use_source_filter": True,
        "use_reliability_modulation": True, "use_reliability_readout": False,
        "use_goal_guidance": True,
    },
    "GAST-no-goal-guidance": {
        "gast_depth": 4, "gast_gsm_operator": "max",
        "goal_endpoint_loss_weight": 1.0,
        "mode_diversity_loss_weight": 0.05, "mode_diversity_margin": 0.5,
        "use_relation_bias": True, "use_source_filter": True,
        "use_reliability_modulation": True, "use_reliability_readout": True,
        "use_goal_guidance": False,
    },
}

def get_git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def build_model(variant: str, num_modes: int):
    from src.model.trajectory_forecaster import TrajectoryForecaster
    model_config = {
        "embed_dim": 128,
        "future_steps": 12,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "qkv_bias": False,
        "drop_path": 0.2,
        "num_modes": num_modes,
        "dt": 0.4,
        "obs_len": 8,
    }
    model_config.update(VARIANTS[variant])
    return TrajectoryForecaster(**model_config)


def load_checkpoint(model, ckpt_path, expected_variant=None):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_config = ckpt.get("hyper_parameters", {}).get("model", {})
    checkpoint_modes = model_config.get("num_modes") if hasattr(model_config, "get") else None
    expected_modes = model.num_modes
    if checkpoint_modes is not None and int(checkpoint_modes) != expected_modes:
        raise ValueError(
            f"checkpoint num_modes={checkpoint_modes} 与评估 K={expected_modes} 不一致"
        )
    run_contract = ckpt.get("trajgap_run_contract", {})
    checkpoint_variant = (
        ckpt.get("meta", {}).get("variant")
        or run_contract.get("model_version")
    )
    configured_variant = model_config.get("model_version")
    required_variant = expected_variant or configured_variant
    if (
        checkpoint_variant
        and required_variant
        and checkpoint_variant != required_variant
    ):
        raise ValueError(
            f"checkpoint variant metadata mismatch: {checkpoint_variant} vs "
            f"{required_variant}"
        )
    for key in (
        "goal_endpoint_loss_weight",
        "mode_diversity_loss_weight",
        "mode_diversity_margin",
        "use_relation_bias",
        "use_source_filter",
        "use_reliability_modulation",
        "use_reliability_readout",
        "use_goal_guidance",
    ):
        checkpoint_value = (
            model_config.get(key) if hasattr(model_config, "get") else None
        )
        if checkpoint_value is not None and checkpoint_value != getattr(model, key):
            raise ValueError(
                f"checkpoint {key}={checkpoint_value} does not match "
                f"variant {key}={getattr(model, key)}"
            )
    state = ckpt.get("state_dict", ckpt)
    cleaned = {k[len("net."):]: v for k, v in state.items() if k.startswith("net.")}
    incompatible = model.load_state_dict(cleaned, strict=False)
    missing_keys = set(incompatible.missing_keys)
    unexpected_keys = set(incompatible.unexpected_keys)
    if missing_keys or unexpected_keys:
        raise ValueError(
            "checkpoint/model mismatch: "
            f"missing={sorted(missing_keys)[:10]} "
            f"unexpected={sorted(unexpected_keys)[:10]}"
        )
    return incompatible.missing_keys, incompatible.unexpected_keys


@torch.no_grad()
def run_evaluation(model, dataset, K, scene, difficulty, split, variant, seed,
                   device="cuda:0", batch_size=64, num_workers=2, max_batches=None):
    from torch.utils.data import DataLoader
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=trajimpute_collate_fn)
    evaluator = DirectEvaluator()
    n_batches = 0
    for batch in loader:
        batch_dev = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        out = model(batch_dev)
        # 与 trainer.test_step 一致：最终输出 new_y_hat/new_pi 优先
        pred = out["new_y_hat"] if out.get("new_y_hat") is not None else out["y_hat"]
        prob = out["new_pi"] if out.get("new_pi") is not None else out["pi"]
        pred = pred[..., :2]
        if pred.shape[1] != K:
            raise RuntimeError(
                f"模型实际输出 K={pred.shape[1]}，但评估协议要求 K={K}"
            )
        target = batch_dev["target"][:, 0]  # focal [B, T, 2]
        prob = prob.float()
        focal_valid = batch["x_valid_mask"][:, 0]
        valid_count = focal_valid.sum(-1)
        anchor_lag = batch["x_anchor_lag_steps"][:, 0]
        forecast_gap = batch["x_forecast_gap_steps"][:, 0]
        # 逐样本分组（batch 内证据条件可能不同）
        for i in range(pred.shape[0]):
            evaluator.update(
                pred[i:i + 1].float().cpu(), prob[i:i + 1].cpu(),
                target[i:i + 1].cpu(),
                scene=scene, difficulty=difficulty, split=split,
                missing_count=int(batch["missing_count"][i]),
                valid_count=int(valid_count[i]),
                anchor_lag=int(anchor_lag[i]),
                forecast_gap=int(forecast_gap[i]),
                max_missing_run=int(batch["max_missing_run"][i]),
            )
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    return evaluator.compute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/lbh/TrajImpute/dataset/TrajImpute")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--difficulty", default="Mixed", choices=["Mixed"],
                    help="TrajGap-Bench 正式主协议固定为 Mixed")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--variant", default="GAST", choices=list(VARIANTS))
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--untrained", action="store_true",
                    help="未训练模型冒烟（结果无意义，仅验证管线）")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--output-root", default="outputs/trajgap_bench")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if args.K != 20:
        raise SystemExit("TrajGap-Bench 正式评估固定 K=20")

    torch.manual_seed(args.seed)
    if args.difficulty != MIXED_DIFFICULTY:
        raise SystemExit("TrajGap-Bench 正式评估只允许 Mixed")
    dataset = TrajGapDataset(args.data_root, args.scene, args.split)
    # Mixed 主协议不做子集过滤；保留总行数用于完整性审计。
    n_total_rows = int(dataset.missing_counts.shape[0])
    n_kept_focal = len(dataset)
    n_filtered_out = 0
    model = build_model(args.variant, args.K)

    ckpt_info = None
    if args.checkpoint:
        missing, unexpected = load_checkpoint(
            model, args.checkpoint, expected_variant=args.variant
        )
        ckpt_info = {
            "path": str(args.checkpoint),
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected)[:20],
        }
    elif not args.untrained:
        raise SystemExit("需要 --checkpoint 或 --untrained（冒烟）之一")

    results = run_evaluation(
        model, dataset, args.K, args.scene, args.difficulty, args.split,
        args.variant, args.seed, device=args.device,
        batch_size=args.batch_size, num_workers=args.num_workers,
        max_batches=args.max_batches,
    )

    meta = {
        "dataset": BENCHMARK_NAME,
        "data_root": args.data_root,
        "scene": args.scene,
        "difficulty": args.difficulty,
        "split": args.split,
        "variant": args.variant,
        "seed": args.seed,
        "K": args.K,
        "evaluator": "src/evaluation/trajimpute_direct.py (TrajGap-Bench direct, no imputation)",
        "git_revision": get_git_revision(),
        "untrained_smoke": bool(args.untrained),
        "checkpoint": ckpt_info,
        "metric_semantics": {
            f"minADE{args.K}": f"unweighted minimum ADE over all K={args.K} modes",
            f"minFDE{args.K}": f"unweighted minimum FDE over all K={args.K} modes",
        },
        "external_reference": None,
        "release_subset": inspect_clean_source(args.data_root),
        "protocol": BENCHMARK_NAME,
        "zero_missing_only": False,
        "n_samples_kept": n_kept_focal,
        "n_samples_total_rows": n_total_rows,
        "n_filtered_out": n_filtered_out,
    }
    tag = "untrained" if args.untrained else "ckpt"
    out_dir = Path(args.output_root) / (
        f"{args.scene}_{args.difficulty}_{args.split}_{args.variant}_seed{args.seed}_{tag}")
    if out_dir.exists():
        raise SystemExit(f"输出目录已存在，拒绝覆盖: {out_dir}")
    out_path = save_results(results, meta, out_dir / "results.json")
    print(f"\n[results] {out_path}")
    ov = results["overall"]
    print(
        f"overall n={ov['n']} minADE{args.K}={ov[f'minADE{args.K}']:.4f} "
        f"minFDE{args.K}={ov[f'minFDE{args.K}']:.4f}"
    )
    print("[groups]")
    for g, entry in sorted(results["by_group"].items()):
        print(
            f"  {g}: n={entry['n']} minADE{args.K}={entry[f'minADE{args.K}']:.4f} "
            f"minFDE{args.K}={entry[f'minFDE{args.K}']:.4f}"
        )


if __name__ == "__main__":
    main()
