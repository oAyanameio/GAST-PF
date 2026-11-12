import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.训练与评估.run_trajimpute_experiments import (
    DEFAULT_VARIANT,
    VARIANTS as TRAIN_VARIANTS,
)
from scripts.结果分析.evaluate_trajimpute_direct import VARIANTS as EVAL_VARIANTS
from docs.tools import gen_leaderboard_tables

SWITCHES = {
    "use_interaction_bias",
    "use_source_filter",
    "use_state_strength_modulation",
    "use_state_strength_readout",
    "use_goal_guidance",
}
EXPECTED_VARIANTS = {
    "GAST",
    "GAST-no-interaction-bias",
    "GAST-no-source-filter",
    "GAST-no-state-strength-modulation",
    "GAST-content-only-readout",
    "GAST-no-goal-guidance",
}


def test_training_and_evaluation_share_single_switch_ablation_configs():
    assert DEFAULT_VARIANT == "GAST"
    assert set(TRAIN_VARIANTS) == EXPECTED_VARIANTS
    assert TRAIN_VARIANTS == EVAL_VARIANTS

    assert all(TRAIN_VARIANTS["GAST"][switch] is True for switch in SWITCHES)
    expected_disabled = {
        "GAST-no-interaction-bias": "use_interaction_bias",
        "GAST-no-source-filter": "use_source_filter",
        "GAST-no-state-strength-modulation": "use_state_strength_modulation",
        "GAST-content-only-readout": "use_state_strength_readout",
        "GAST-no-goal-guidance": "use_goal_guidance",
    }
    for name, config in TRAIN_VARIANTS.items():
        disabled = {switch for switch in SWITCHES if config[switch] is False}
        expected = {expected_disabled[name]} if name in expected_disabled else set()
        assert disabled == expected


def test_hydra_model_defaults_enable_full_gast():
    config = yaml.safe_load(
        (REPO / "conf/model/gast_model_forecast.yaml").read_text()
    )
    model = config["target"]["model"]
    assert all(model[switch] is True for switch in SWITCHES)


def test_leaderboard_generator_emits_only_ordinary_min_metrics():
    assert [key for key, _, _ in gen_leaderboard_tables.METRICS] == [
        "minADE_K",
        "minFDE_K",
    ]
    data = {method: {"per": {}} for method in gen_leaderboard_tables.METHODS}
    output = (
        gen_leaderboard_tables.emit_md(data)
        + gen_leaderboard_tables.emit_tex_micro(data)
        + gen_leaderboard_tables.emit_tex_extra(data)
    )
    assert "minADE$_{20}$" in output
    assert "minFDE$_{20}$" in output
    assert "ADE@1" not in output
    assert "FDE@1" not in output
    assert "Brier" not in output
