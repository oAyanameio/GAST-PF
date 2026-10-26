import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GAST Mamba forward requires CUDA",
)
def test_gast_goal_forward_uses_only_goal_decoder():
    try:
        import selective_scan_cuda
    except (ImportError, OSError) as error:
        pytest.skip(f"selective_scan_cuda cannot be loaded: {error}")

    from src.model.trajectory_forecaster import TrajectoryForecaster

    model = TrajectoryForecaster(
        embed_dim=128,
        future_steps=12,
        num_heads=8,
        num_actor_types=1,
        num_modes=6,
        dt=0.4,
        obs_len=8,
        encoder_type="gast",
        gast_oer_spatial=True,
        gast_emc_temporal=True,
        gast_ewhr_readout=True,
        gast_goal=True,
        loss_mode="final_consistent",
    ).cuda().eval()
    data = {
        "x_positions_diff": torch.randn(2, 3, 8, 2, device="cuda"),
        "x_velocity_diff": torch.randn(2, 3, 8, device="cuda"),
        "x_valid_mask": torch.ones(2, 3, 8, dtype=torch.bool, device="cuda"),
        "x_key_valid_mask": torch.ones(2, 3, dtype=torch.bool, device="cuda"),
        "x_positions": torch.randn(2, 3, 8, 2, device="cuda"),
        "x_attr": torch.zeros(2, 3, 3, dtype=torch.uint8, device="cuda"),
        "target": torch.randn(2, 3, 12, 2, device="cuda"),
        "target_mask": torch.ones(
            2, 3, 12, dtype=torch.bool, device="cuda"
        ),
    }

    with torch.no_grad():
        output = model(data)

    assert not hasattr(model, "future_decoder")
    assert output["new_y_hat"].shape == (2, 6, 12, 2)
    assert output["goal_endpoints"].shape == (2, 6, 2)
    assert torch.allclose(
        output["new_y_hat"][:, :, -1], output["goal_endpoints"], atol=1e-5
    )
