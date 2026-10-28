import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.model.layers.gast_encoder import (  # noqa: E402
    GASTEncoder,
    construct_observation_evidence_state,
    evidence_weighted_state_aggregation,
    missing_run_ages,
    relative_geometry_features,
)


def test_missing_run_ages_matches_forward_recursion():
    valid = torch.tensor([[[1, 1, 0, 0, 1, 0]]], dtype=torch.bool)
    assert missing_run_ages(valid).tolist() == [[[0, 1, 2, 3, 1, 2]]]


def test_observation_evidence_state_tracks_age_reliability_and_padding():
    valid = torch.tensor([[[1, 0, 0], [1, 1, 1]]], dtype=torch.bool)
    padding = torch.tensor([[False, True]])
    observed, reliability, age = construct_observation_evidence_state(
        valid, tau=0.5, key_padding=padding
    )
    assert observed[0, 0].tolist() == [1.0, 0.0, 0.0]
    assert age[0, 0].tolist() == [0.0, 1.0, 2.0]
    assert torch.allclose(reliability[0, 0], torch.tensor([1.0, torch.exp(torch.tensor(-0.5)), torch.exp(torch.tensor(-1.0))]))
    assert not observed[0, 1].any()
    assert not reliability[0, 1].any()


def test_relative_geometry_ignores_missing_placeholders():
    valid = torch.tensor([[[1, 0], [1, 1]]], dtype=torch.bool)
    positions = torch.tensor([[[[1.0, 1.0], [1000.0, 1000.0]], [[2.0, 0.0], [3.0, 0.0]]]])
    changed = positions.clone()
    changed[0, 0, 1] = torch.tensor([-9000.0, 8000.0])
    first = relative_geometry_features(positions, valid)
    second = relative_geometry_features(changed, valid)
    assert first.shape == (1, 2, 2, 2, 7)
    assert torch.equal(first, second)


def test_evidence_weighted_state_aggregation_uses_terminal_state_when_history_is_complete():
    hidden = torch.randn(2, 3, 4, 8)
    reliability = torch.ones(2, 3, 4)
    scores = torch.randn(2, 3, 4, 1)
    output, weights = evidence_weighted_state_aggregation(
        hidden, reliability, scores
    )
    assert torch.equal(output, hidden[..., -1, :])
    assert torch.isfinite(weights).all()


def test_gast_variants_have_only_forward_mamba_and_expected_modules():
    uni = GASTEncoder(embed_dim=16, num_heads=4, depth=1)
    gast = GASTEncoder(
        embed_dim=16,
        num_heads=4,
        depth=1,
        geometry_evidence_relational_encoding=True,
        evidence_conditioned_causal_temporal_aggregation=True,
    )
    assert not hasattr(uni.temporal, "ssm_back")
    assert not hasattr(gast.temporal_aggregation.temporal_mamba, "ssm_back")
    assert hasattr(gast.spatial, "geometry_bias")
    assert hasattr(gast.temporal_aggregation.state_aggregation, "score")
    assert not hasattr(gast, "directional_readout")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GAST Mamba forward requires CUDA")
def test_gast_encoder_outputs_are_finite_and_zero_padding_actors():
    device = torch.device("cuda")
    torch.manual_seed(4)
    encoder = GASTEncoder(
        input_dim=4,
        embed_dim=16,
        num_heads=4,
        depth=1,
        p_dropout=0,
        drop_path=0,
        geometry_evidence_relational_encoding=True,
        evidence_conditioned_causal_temporal_aggregation=True,
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


def test_gast_requires_all_evidence_stages_together():
    with pytest.raises(ValueError, match="GERE and ECTA together"):
        GASTEncoder(
            embed_dim=16,
            num_heads=4,
            geometry_evidence_relational_encoding=True,
        )


def test_training_and_evaluation_expose_only_selected_variants():
    from scripts.训练与评估.run_trajimpute_experiments import (
        DEFAULT_VARIANT,
        VARIANTS as train_variants,
    )
    from scripts.结果分析.evaluate_trajimpute_direct import VARIANTS as eval_variants

    assert DEFAULT_VARIANT == "GAST"
    assert set(train_variants) == {"M0", "GAST"}
    assert train_variants.keys() == eval_variants.keys()
    assert train_variants["GAST"] == eval_variants["GAST"]
    assert train_variants["GAST"]["gast_goal"] is True
    assert train_variants["GAST"]["classification_target_mode"] == "trajectory"


def test_gast_yaml_uses_trajectory_target_and_goal_decoder_by_default():
    import yaml

    config = yaml.safe_load(
        (REPO / "conf/model/gast_model_forecast.yaml").read_text()
    )
    model = config["target"]["model"]
    assert model["gast_goal"] is True
    assert model["classification_target_mode"] == "trajectory"


def test_forecaster_registers_gast_and_preserves_m0_state_dict():
    from src.model.trajectory_forecaster import TrajectoryForecaster

    torch.manual_seed(2024)
    m0_a = TrajectoryForecaster(num_modes=4)
    torch.manual_seed(2024)
    m0_b = TrajectoryForecaster(num_modes=4, encoder_type="m0")
    assert all(torch.equal(value, m0_b.state_dict()[key]) for key, value in m0_a.state_dict().items())

    model = TrajectoryForecaster(
        num_modes=4,
        encoder_type="gast",
        gast_gere_spatial=True,
        gast_ecta=True,
    )
    assert model.gast_method_name == "GAST"
    assert not hasattr(model, "gast_scene_tail")
    assert not any(key.startswith("history_input_projection.") for key in model.state_dict())


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
                    "encoder_type": "gast",
                    "gast_gere_spatial": True,
                    "gast_ecta": True,
                    "gast_goal": True,
                    "loss_mode": "final_consistent",
                    "classification_target_mode": "trajectory",
                    "goal_endpoint_loss_weight": 1.0,
                    "mode_diversity_loss_weight": 0.05,
                    "mode_diversity_margin": 0.5,
                    "gast_temporal_direction": "unidirectional",
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
