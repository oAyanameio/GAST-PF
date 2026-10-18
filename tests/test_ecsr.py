"""ECSR shared evidence state and routing behavior tests."""

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.model.layers.gast_encoder import (  # noqa: E402
    GASTEncoder,
    GSMAttnBlock,
    compute_evidence_state,
    evidence_aware_readout,
    relative_geometry_features,
    route_bidirectional_paths,
)


def test_evidence_state_uses_zero_age_for_observed_frames():
    valid = torch.tensor([[[True, False, False, True, False]]])

    direct, reliability, age = compute_evidence_state(valid, tau=1.0)

    assert direct.tolist() == [[[1.0, 0.0, 0.0, 1.0, 0.0]]]
    assert age.tolist() == [[[0.0, 1.0, 2.0, 0.0, 1.0]]]
    assert torch.allclose(reliability[..., [0, 3]], torch.ones(1, 1, 2))
    assert reliability[0, 0, 1] > reliability[0, 0, 2]
    assert torch.isfinite(reliability).all()


def test_evidence_state_disables_padding_sources_and_reliability():
    valid = torch.tensor([[[True, False], [True, True]]])
    padding = torch.tensor([[False, True]])

    direct, reliability, _ = compute_evidence_state(valid, tau=0.1, key_padding=padding)

    assert direct[0, 1].eq(0).all()
    assert reliability[0, 1].eq(0).all()


def test_geometry_features_ignore_missing_position_placeholders():
    positions = torch.tensor(
        [[[[0.0, 0.0], [0.0, 0.0]],
          [[1.0, 0.0], [0.0, 0.0]],
          [[0.0, 1.0], [2.0, 1.0]]]]
    )
    valid = torch.tensor([[[True, False], [True, False], [True, True]]])
    changed = positions.clone()
    changed[~valid] = float("nan")

    base_features, pair_valid = relative_geometry_features(positions, valid)
    changed_features, _ = relative_geometry_features(changed, valid)

    assert torch.isfinite(changed_features).all()
    assert not pair_valid[0, 1, 1].any()
    assert not pair_valid[0, 1, :, 1].any()
    assert torch.allclose(base_features, changed_features)


def test_bidirectional_route_recovers_original_sum_for_full_evidence():
    forward = torch.randn(2, 3, 4, 5)
    backward = torch.randn(2, 3, 4, 5)
    reliability = torch.ones(2, 3, 4)

    fused, alpha = route_bidirectional_paths(forward, backward, reliability)

    assert torch.allclose(fused, forward + backward)
    assert torch.allclose(alpha, torch.full_like(alpha, 0.5))


def test_bidirectional_route_prefers_less_stale_direction():
    forward = torch.ones(1, 1, 4, 1)
    backward = torch.zeros_like(forward)
    reliability = torch.tensor([[[1.0, 0.5, 0.25, 1.0]]])

    fused, alpha = route_bidirectional_paths(forward, backward, reliability)

    assert alpha[0, 0, 0, 0] > alpha[0, 0, 0, 1]
    assert fused[0, 0, 0, 0] > 1.0


def test_evidence_aware_readout_is_finite_and_uses_reliability():
    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    reliability = torch.tensor([[1.0, 0.5, 0.25]])
    score = torch.zeros(1, 3, 1)

    output, weights = evidence_aware_readout(hidden, reliability, score, beta=1.0)

    assert output.shape == (1, 4)
    assert weights.shape == (1, 3)
    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()
    assert weights[0, 0] > weights[0, 1] > weights[0, 2]
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1))


def test_evidence_aware_readout_returns_terminal_state_when_fully_observed():
    hidden = torch.randn(2, 4, 5)
    reliability = torch.ones(2, 4)
    scores = torch.randn(2, 4, 1)

    output, _ = evidence_aware_readout(hidden, reliability, scores)

    assert torch.equal(output, hidden[:, -1])


def test_missing_source_is_blocked_but_missing_query_can_receive_context():
    torch.manual_seed(7)
    block = GSMAttnBlock(
        input_dim=1,
        model_dim=8,
        num_heads=2,
        p_dropout=0.0,
        ecsr=True,
    ).eval()
    x = torch.tensor([[[[0.2], [0.4], [0.8]]]])
    valid = torch.tensor([[[True], [False], [True]]])
    mask = valid.permute(0, 2, 1)
    padding = torch.zeros(1, 3, dtype=torch.bool)
    positions = torch.tensor([[[[0.0, 0.0]], [[2.0, -1.0]], [[1.0, 1.0]]]])

    with torch.no_grad():
        base = block(x, mask, padding, positions, valid.float())
        perturbed = x.clone()
        perturbed[..., 1, :] = 1000.0
        source_perturbed = block(perturbed, mask, padding, positions, valid.float())

    assert torch.allclose(base[..., [0, 2], :], source_perturbed[..., [0, 2], :], atol=1e-5)

    neighbor_changed = x.clone()
    neighbor_changed[..., 0, :] += 3.0
    with torch.no_grad():
        query_context_changed = block(
            neighbor_changed, mask, padding, positions, valid.float()
        )
    assert not torch.allclose(base[..., 1, :], query_context_changed[..., 1, :])


def test_ecsr_geometry_bias_changes_attention_with_valid_geometry():
    torch.manual_seed(15)
    block = GSMAttnBlock(2, 8, 2, p_dropout=0.0, ecsr=True).eval()
    with torch.no_grad():
        block.geometry_bias[-1].weight[:, 0] = 0.8
    x = torch.tensor([[[[0.1, 0.8], [0.6, -0.2], [-0.4, 0.3]]]])
    valid = torch.ones(1, 3, 1, dtype=torch.bool)
    mask = valid.permute(0, 2, 1)
    padding = torch.zeros(1, 3, dtype=torch.bool)
    positions = torch.tensor(
        [[[[0.0, 0.0]], [[1.0, 0.0]], [[0.0, 2.0]]]]
    )
    moved = positions[:, [1, 0, 2]]

    with torch.no_grad():
        base = block(x, mask, padding, positions, valid.float())
        geometry_changed = block(x, mask, padding, moved, valid.float())

    assert not torch.allclose(base, geometry_changed)


def test_ecsr_spatial_attention_is_actor_permutation_equivariant():
    torch.manual_seed(16)
    block = GSMAttnBlock(2, 8, 2, p_dropout=0.0, ecsr=True).eval()
    x = torch.randn(1, 2, 3, 2)
    valid = torch.tensor([[[True, True], [True, False], [True, True]]])
    mask = valid.permute(0, 2, 1)
    padding = torch.zeros(1, 3, dtype=torch.bool)
    positions = torch.randn(1, 3, 2, 2)
    permutation = torch.tensor([2, 0, 1])

    with torch.no_grad():
        base = block(x, mask, padding, positions, valid.float())
        permuted = block(
            x[:, :, permutation],
            mask[:, :, permutation],
            padding[:, permutation],
            positions[:, permutation],
            valid[:, permutation].float(),
        )

    assert torch.allclose(base[:, :, permutation], permuted, atol=1e-5)


def test_ecsr_spatial_attention_isolates_padding_actor():
    torch.manual_seed(17)
    block = GSMAttnBlock(2, 8, 2, p_dropout=0.0, ecsr=True).eval()
    x = torch.randn(1, 2, 3, 2)
    valid = torch.tensor([[[True, True], [True, False], [False, False]]])
    mask = valid.permute(0, 2, 1)
    padding = torch.tensor([[False, False, True]])
    positions = torch.randn(1, 3, 2, 2)
    changed_x = x.clone()
    changed_x[:, :, 2] += 1000.0
    changed_positions = positions.clone()
    changed_positions[:, 2] -= 1000.0

    with torch.no_grad():
        base = block(x, mask, padding, positions, valid.float())
        changed = block(changed_x, mask, padding, changed_positions, valid.float())

    assert torch.allclose(base[:, :, :2], changed[:, :, :2], atol=1e-5)


def test_ecsr_model_variant_registers_shared_routing_modules():
    from src.model.trajectory_forecaster import TrajectoryForecaster
    from scripts.训练与评估.run_trajimpute_experiments import VARIANTS

    model = TrajectoryForecaster(num_modes=6, encoder_type="gast", gast_ecsr=True)

    assert VARIANTS["GAST-PF-ECSR"]["gast_ecsr"] is True
    assert model.gast_ecsr is True
    assert model.gast_encoder.ecsr is True
    assert hasattr(model.gast_encoder, "evidence_tau")
    assert hasattr(model.gast_encoder, "readout_score")


def test_ecsr_parameters_are_covered_by_adamw_groups():
    from src.model.forecasting_module import ForecastingLightningModule

    module = ForecastingLightningModule(
        model={
            "type": "TrajectoryForecaster",
            "embed_dim": 128,
            "future_steps": 12,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "qkv_bias": False,
            "drop_path": 0.2,
            "num_actor_types": 1,
            "num_modes": 20,
            "dt": 0.4,
            "obs_len": 8,
            "bimamba": False,
            "encoder_type": "gast",
            "gast_ecsr": True,
        },
        lr=1e-3,
        warmup_epochs=5,
        epochs=100,
        weight_decay=1e-4,
    )
    optimizer = module.configure_optimizers()[0][0]
    trainable = {id(parameter) for parameter in module.parameters() if parameter.requires_grad}
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]

    assert {id(parameter) for parameter in grouped} == trainable
    assert len(grouped) == len({id(parameter) for parameter in grouped})


def test_ecsr_rejects_legacy_geometry_switch_combination():
    from src.model.trajectory_forecaster import TrajectoryForecaster

    try:
        TrajectoryForecaster(
            num_modes=6,
            encoder_type="gast",
            gast_ecsr=True,
            gast_frame_geo=True,
        )
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("ECSR must not silently combine legacy geometry variants")


def test_ecsr_evaluator_checkpoint_contract(tmp_path):
    from scripts.结果分析.evaluate_trajimpute_direct import build_model, load_checkpoint

    ecsr_model = build_model("GAST-PF-ECSR", num_modes=20)
    baseline_model = build_model("GAST-PF", num_modes=20)

    def save_checkpoint(model, path):
        torch.save(
            {
                "state_dict": {
                    f"net.{key}": value for key, value in model.state_dict().items()
                },
                "hyper_parameters": {
                    "model": {
                        "num_modes": 20,
                        "encoder_type": "gast",
                        "gast_ecsr": model.gast_ecsr,
                    }
                },
            },
            path,
        )
        return path

    ecsr_checkpoint = save_checkpoint(ecsr_model, tmp_path / "ecsr.ckpt")
    baseline_checkpoint = save_checkpoint(baseline_model, tmp_path / "baseline.ckpt")

    missing, unexpected = load_checkpoint(
        build_model("GAST-PF-ECSR", num_modes=20), ecsr_checkpoint
    )
    assert not missing and not unexpected

    with pytest.raises(ValueError, match="gast_ecsr"):
        load_checkpoint(build_model("GAST-PF", num_modes=20), ecsr_checkpoint)

    with pytest.raises(ValueError, match="gast_ecsr"):
        load_checkpoint(
            build_model("GAST-PF-ECSR", num_modes=20), baseline_checkpoint
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU（Mamba CUDA）")
def test_ecsr_encoder_gpu_forward_backward_with_missing_and_padding():
    device = torch.device("cuda:0")
    encoder = GASTEncoder(
        input_dim=4,
        embed_dim=32,
        num_heads=4,
        depth=1,
        p_dropout=0.0,
        drop_path=0.0,
        ecsr=True,
    ).to(device)
    features = torch.randn(2, 3, 8, 4, device=device)
    valid = torch.ones(2, 3, 8, dtype=torch.bool, device=device)
    valid[0, 1, 3:5] = False
    valid[1, 2] = False
    key_padding = torch.tensor([[False, False, False], [False, False, True]], device=device)
    positions = torch.randn(2, 3, 8, 2, device=device)
    positions[~valid] = float("nan")
    features[~valid] = 1000.0

    output = encoder(features, valid, key_padding, positions=positions)
    loss = output.square().mean()
    loss.backward()

    assert output.shape == (2, 3, 32)
    assert torch.isfinite(output).all()
    assert torch.equal(output[1, 2], torch.zeros_like(output[1, 2]))
    gradients = [parameter.grad for parameter in encoder.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
