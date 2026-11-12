import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.model.layers.observation_state_encoder import (  # noqa: E402
    FramewiseInteractionModel,
    ObservationStateEncoder,
    aggregate_history_by_state_strength,
    build_framewise_interaction_features,
    build_observation_state,
)


def test_observation_state_tracks_age_state_strength_and_padding():
    valid = torch.tensor([[[1, 0, 0], [1, 1, 1]]], dtype=torch.bool)
    padding = torch.tensor([[False, True]])
    observed, state_strength, age = build_observation_state(
        valid, tau=0.5, key_padding=padding
    )
    assert observed[0, 0].tolist() == [1.0, 0.0, 0.0]
    assert age[0, 0].tolist() == [0.0, 1.0, 2.0]
    assert torch.allclose(state_strength[0, 0], torch.tensor([1.0, torch.exp(torch.tensor(-0.5)), torch.exp(torch.tensor(-1.0))]))
    assert not observed[0, 1].any()
    assert not state_strength[0, 1].any()


def test_actor_relation_features_ignore_missing_placeholders():
    valid = torch.tensor([[[1, 0], [1, 1]]], dtype=torch.bool)
    positions = torch.tensor(
        [[[[1.0, 1.0], [1000.0, 1000.0]], [[2.0, 0.0], [3.0, 0.0]]]]
    )
    changed = positions.clone()
    changed[0, 0, 1] = torch.tensor([-9000.0, 8000.0])
    first = build_framewise_interaction_features(positions, valid)
    second = build_framewise_interaction_features(changed, valid)
    assert first.shape == (1, 2, 2, 2, 7)
    assert torch.equal(first, second)


def test_history_readout_uses_terminal_state_when_history_is_complete():
    hidden = torch.randn(2, 3, 4, 8)
    state_strength = torch.ones(2, 3, 4)
    scores = torch.randn(2, 3, 4, 1)
    output, weights = aggregate_history_by_state_strength(
        hidden, state_strength, scores
    )
    assert torch.equal(output, hidden[..., -1, :])
    assert torch.isfinite(weights).all()


def test_content_only_readout_uses_scores_without_state_strength_weights():
    hidden = torch.tensor([[[[0.0], [10.0]]]])
    state_strength = torch.tensor([[[1.0, 0.01]]])
    scores = torch.tensor([[[[0.0], [1.0]]]])
    output, weights = aggregate_history_by_state_strength(
        hidden, state_strength, scores, use_state_strength_weighting=False
    )
    expected_weights = torch.softmax(scores.squeeze(-1), dim=-1)
    expected = torch.sum(expected_weights.unsqueeze(-1) * hidden, dim=-2)

    assert torch.allclose(weights, expected_weights)
    assert torch.allclose(output, expected)


def test_gast_uses_complete_evidence_stages():
    gast = ObservationStateEncoder(embed_dim=16, num_heads=4, depth=1)
    assert isinstance(gast.framewise_interaction, FramewiseInteractionModel)
    assert hasattr(gast.framewise_interaction, "interaction_bias")
    assert hasattr(gast.state_strength_aggregation.history_readout, "score")
    blocks = gast.state_strength_aggregation.mamba.mamba_blocks
    assert len(blocks) == 1


def test_gast_structural_switches_disable_their_target_components():
    gast = ObservationStateEncoder(
        embed_dim=16,
        num_heads=4,
        depth=1,
        use_interaction_bias=False,
        use_source_filter=False,
        use_state_strength_modulation=False,
        use_state_strength_readout=False,
    )

    assert gast.framewise_interaction.use_source_filter is False
    assert gast.framewise_interaction.interaction_bias is None
    assert gast.state_strength_aggregation.mamba.use_state_strength_modulation is False
    assert gast.state_strength_aggregation.history_readout.use_state_strength_readout is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GAST Mamba forward requires CUDA")
def test_observation_encoder_outputs_are_finite_and_zero_padding_actors():
    device = torch.device("cuda")
    torch.manual_seed(4)
    encoder = ObservationStateEncoder(
        input_dim=4,
        embed_dim=16,
        num_heads=4,
        depth=1,
        p_dropout=0,
        drop_path=0,
    ).to(device).eval()
    valid = torch.tensor(
        [[[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]]], dtype=torch.bool
    )
    padding = torch.tensor([[False, False, True]], device=device)
    valid = valid.to(device)
    features = torch.randn(1, 3, 4, 4, device=device)
    positions = torch.randn(1, 3, 4, 2, device=device)
    positions = positions * valid.unsqueeze(-1)
    with torch.no_grad():
        output = encoder(features, valid, padding, positions)
    assert output.shape == (1, 3, 16)
    assert torch.isfinite(output).all()
    assert torch.equal(output[:, 2], torch.zeros_like(output[:, 2]))


def test_training_and_evaluation_expose_matching_structural_ablations():
    from scripts.训练与评估.run_trajimpute_experiments import (
        DEFAULT_VARIANT,
        VARIANTS as train_variants,
    )
    from scripts.结果分析.evaluate_trajimpute_direct import VARIANTS as eval_variants

    assert DEFAULT_VARIANT == "GAST"
    assert set(train_variants) == {
        "GAST",
        "GAST-no-interaction-bias",
        "GAST-no-source-filter",
        "GAST-no-state-strength-modulation",
        "GAST-content-only-readout",
        "GAST-no-goal-guidance",
    }
    assert train_variants.keys() == eval_variants.keys()
    assert train_variants["GAST"] == eval_variants["GAST"]
    assert train_variants["GAST"]["encoder_depth"] == 4
    for variant in train_variants.values():
        assert all(
            key in variant
            for key in (
                "use_source_filter",
                "use_interaction_bias",
                "use_state_strength_modulation",
                "use_state_strength_readout",
                "use_goal_guidance",
            )
        )
    assert train_variants["GAST-no-interaction-bias"]["use_interaction_bias"] is False
    assert train_variants["GAST-no-source-filter"]["use_source_filter"] is False
    assert train_variants["GAST-no-state-strength-modulation"]["use_state_strength_modulation"] is False
    assert train_variants["GAST-content-only-readout"]["use_state_strength_readout"] is False

    assert train_variants["GAST-no-goal-guidance"]["use_goal_guidance"] is False


def test_gast_yaml_uses_full_method_by_default():
    import yaml

    config = yaml.safe_load(
        (REPO / "conf/model/gast_model_forecast.yaml").read_text()
    )
    model = config["target"]["model"]
    assert model["encoder_depth"] == 4
    assert model["interaction_operator"] == "max"
    for switch in (
        "use_interaction_bias",
        "use_source_filter",
        "use_state_strength_modulation",
        "use_state_strength_readout",
        "use_goal_guidance",
    ):
        assert model[switch] is True


def test_forecaster_builds_only_the_gast_components():
    from src.model.trajectory_forecaster import TrajectoryForecaster

    model = TrajectoryForecaster(num_modes=4)
    assert hasattr(model, "observation_encoder")
    assert hasattr(model, "goal_guided_decoder")
    assert not hasattr(model, "gast_scene_tail")


def test_current_gast_checkpoint_loads_into_gast(tmp_path):
    from scripts.结果分析.evaluate_trajimpute_direct import (
        build_model,
        load_checkpoint,
    )
    source_model = build_model("GAST", num_modes=4)
    state_dict = {
        f"net.{key}": value for key, value in source_model.state_dict().items()
    }
    checkpoint = {
        "state_dict": state_dict,
        "hyper_parameters": {
            "model": {
                **{
                    "num_modes": 4,
                    "goal_endpoint_loss_weight": 1.0,
                    "mode_diversity_loss_weight": 0.05,
                    "mode_diversity_margin": 0.5,
                }
            }
        },
        "meta": {"variant": "GAST"},
    }
    checkpoint_path = tmp_path / "gast.ckpt"
    torch.save(checkpoint, checkpoint_path)

    current_model = build_model("GAST", num_modes=4)
    missing, unexpected = load_checkpoint(current_model, checkpoint_path)
    assert missing == []
    assert unexpected == []
