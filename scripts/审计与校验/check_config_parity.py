#!/usr/bin/env python
"""Validate the active TrajGap configuration and model registry."""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.训练与评估.run_trajimpute_experiments import (  # noqa: E402
    DEFAULT_VARIANT,
    VARIANTS as TRAIN_VARIANTS,
)
from scripts.结果分析.evaluate_trajimpute_direct import (  # noqa: E402
    VARIANTS as EVAL_VARIANTS,
)


def main():
    root_config = yaml.safe_load(
        (ROOT / "conf/config_missing_aware_trajimpute.yaml").read_text()
    )
    model_config = yaml.safe_load(
        (ROOT / "conf/model/missing_aware_ethucy_model_forecast.yaml").read_text()
    )
    model = model_config["target"]["model"]
    expected = {
        "monitor": "val_new_minFDE20",
        "bimamba": False,
        "num_modes": 20,
        "encoder_type": "gast",
        "gast_temporal_direction": "unidirectional",
        "gast_oer_spatial": True,
        "gast_emc_temporal": True,
        "gast_ewhr_readout": True,
        "gast_goal": False,
        "loss_mode": "final_consistent",
        "goal_endpoint_loss_weight": 1.0,
        "mode_diversity_loss_weight": 0.05,
        "mode_diversity_margin": 0.5,
        "classification_target_mode": "joint",
    }
    actual = {
        "monitor": root_config.get("monitor"),
        "bimamba": root_config.get("bimamba"),
        "num_modes": model.get("num_modes"),
        **{key: model.get(key) for key in expected
           if key in model and key not in {"monitor", "bimamba", "num_modes"}},
    }
    problems = [f"{key}: expected {value!r}, found {actual.get(key)!r}"
                for key, value in expected.items() if actual.get(key) != value]
    if DEFAULT_VARIANT != "GAST":
        problems.append(f"default variant is {DEFAULT_VARIANT!r}")
    if TRAIN_VARIANTS.keys() != EVAL_VARIANTS.keys():
        problems.append("training and evaluation variant registries differ")
    if set(TRAIN_VARIANTS) != {
        "M0", "GAST-PF-Uni", "GAST", "GAST-Goal",
        "GAST-Goal-TrajCls", "GAST-PF-UniOER"
    }:
        problems.append(f"unexpected active variants: {sorted(TRAIN_VARIANTS)}")
    if TRAIN_VARIANTS.get("GAST") != EVAL_VARIANTS.get("GAST"):
        problems.append("GAST training and evaluation loss/model settings differ")
    if TRAIN_VARIANTS.get("GAST-Goal") != EVAL_VARIANTS.get("GAST-Goal"):
        problems.append("GAST-Goal training and evaluation loss/model settings differ")
    if TRAIN_VARIANTS.get("GAST-Goal-TrajCls") != EVAL_VARIANTS.get("GAST-Goal-TrajCls"):
        problems.append("GAST-Goal-TrajCls training and evaluation settings differ")
    if TRAIN_VARIANTS.get("M0") != EVAL_VARIANTS.get("M0"):
        problems.append("M0 training and evaluation loss/model settings differ")
    if TRAIN_VARIANTS.get("GAST-PF-Uni") != EVAL_VARIANTS.get("GAST-PF-Uni"):
        problems.append("GAST-PF-Uni training and evaluation loss/model settings differ")
    if TRAIN_VARIANTS.get("GAST-PF-UniOER") != EVAL_VARIANTS.get("GAST-PF-UniOER"):
        problems.append("GAST-PF-UniOER training and evaluation loss/model settings differ")

    if problems:
        raise SystemExit("CONFIG CHECK FAILED\n- " + "\n- ".join(problems))
    print("CONFIG CHECK PASSED: GAST defaults and active variants are aligned")


if __name__ == "__main__":
    main()
