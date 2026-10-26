import torch

from src.model.layers.gast_goal_decoder import GapAwareGoalDecoder
from src.utils.LaplaceNLLLoss import LaplaceNLLLoss
from src.model.trajectory_loss import final_consistent_loss


def _goal_batch():
    positions = torch.zeros(2, 3, 8, 2)
    positions[0, 0, 0] = torch.tensor([1.0, -2.0])
    positions[1, 0, 0] = torch.tensor([0.0, 0.0])
    positions[1, 0, 5] = torch.tensor([5.0, 0.0])
    valid = torch.zeros(2, 3, 8, dtype=torch.bool)
    valid[0, 0, 0] = True
    valid[1, 0, [0, 5]] = True
    valid[:, 1:, :] = True
    return {"x_positions": positions, "x_valid_mask": valid}


def test_gap_aware_goal_decoder_handles_single_observation_and_exact_endpoint():
    torch.manual_seed(7)
    decoder = GapAwareGoalDecoder(
        future_steps=4, dim=16, num_modes=3, num_heads=4, dt=0.4
    ).eval()
    encoding = torch.randn(2, 3, 16)
    actor_padding_mask = torch.zeros(2, 3, dtype=torch.bool)

    output = decoder(encoding, _goal_batch(), actor_padding_mask)

    assert output["trajectories"].shape == (2, 3, 4, 2)
    assert output["endpoints"].shape == (2, 3, 2)
    assert torch.isfinite(output["trajectories"]).all()
    assert torch.isfinite(output["scores"]).all()
    assert torch.allclose(
        output["trajectories"][:, :, -1], output["endpoints"], atol=1e-6
    )
    assert torch.allclose(
        output["future_time_seconds"],
        torch.tensor([[3.2, 3.6, 4.0, 4.4], [1.2, 1.6, 2.0, 2.4]]),
    )


def test_goal_decoder_backpropagates_through_endpoint_and_trajectory_heads():
    decoder = GapAwareGoalDecoder(
        future_steps=3, dim=16, num_modes=2, num_heads=4, dt=0.4
    )
    encoding = torch.randn(2, 3, 16)
    output = decoder(
        encoding,
        _goal_batch(),
        actor_padding_mask=torch.zeros(2, 3, dtype=torch.bool),
    )
    target = torch.randn(2, 3, 2)
    loss = LaplaceNLLLoss().compute(
        {
            "traj": output["trajectories"],
            "scale": output["scales"],
            "probs": output["scores"],
        },
        target,
    )
    loss = loss + output["trajectories"].square().mean()
    loss.backward()

    assert decoder.endpoint_head[-1].weight.grad is not None
    assert decoder.trajectory_head[-1].weight.grad is not None
    assert decoder.trajectory_scale_head[-1].weight.grad is not None


def test_final_consistent_loss_matches_goal_and_non_goal_output_contracts():
    torch.manual_seed(11)
    batch_size, modes, steps = 2, 3, 4
    target = torch.randn(batch_size, 2, steps, 2)
    prediction = torch.randn(batch_size, modes, steps, 2, requires_grad=True)
    scale = (torch.rand(batch_size, modes, steps, 2) + 1.1).requires_grad_()
    logits = torch.randn(batch_size, modes, requires_grad=True)
    others = torch.randn(batch_size, 1, steps, 2, requires_grad=True)
    data = {
        "target": target,
        "target_mask": torch.ones(batch_size, 2, steps, dtype=torch.bool),
    }
    common = {
        "y_hat": prediction,
        "pi": logits,
        "scal": scale,
        "new_y_hat": prediction,
        "new_pi": logits,
        "scal_new": scale,
        "y_hat_others": others,
    }
    baseline_loss, baseline_logs = final_consistent_loss(
        common, data, LaplaceNLLLoss()
    )
    goal_out = {
        **common,
        "goal_endpoints": prediction[..., -1, :].detach().clone(),
    }
    goal_loss, goal_logs = final_consistent_loss(
        goal_out, data, LaplaceNLLLoss()
    )

    assert torch.allclose(baseline_loss, goal_loss)
    assert baseline_logs.keys() == goal_logs.keys()
    goal_loss.backward()
    assert scale.grad is not None
    assert torch.isfinite(scale.grad).all()


def test_goal_trajectory_classification_uses_full_trajectory_best_mode():
    target = torch.zeros(1, 1, 4, 2)
    prediction = torch.tensor(
        [[
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [2.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
        ]]
    )
    logits = torch.tensor([[2.0, -1.0]])
    scale = torch.ones(1, 2, 4, 2)
    data = {
        "target": target,
        "target_mask": torch.ones(1, 1, 4, dtype=torch.bool),
    }
    output = {
        "y_hat": prediction,
        "pi": logits,
        "scal": scale,
        "new_y_hat": prediction,
        "new_pi": logits,
        "scal_new": scale,
        "y_hat_others": torch.zeros(1, 0, 4, 2),
        "goal_endpoints": prediction[..., -1, :],
    }

    _, joint_logs = final_consistent_loss(output, data, LaplaceNLLLoss())
    _, trajectory_logs = final_consistent_loss(
        output,
        data,
        LaplaceNLLLoss(),
        classification_target_mode="trajectory",
    )

    assert torch.allclose(
        joint_logs["cls_loss"],
        torch.nn.functional.cross_entropy(logits, torch.tensor([1]), label_smoothing=0.2),
    )
    assert torch.allclose(
        trajectory_logs["cls_loss"],
        torch.nn.functional.cross_entropy(logits, torch.tensor([0]), label_smoothing=0.2),
    )
    assert not torch.allclose(joint_logs["cls_loss"], trajectory_logs["cls_loss"])
    for key in ("reg_loss", "endpoint_loss", "laplace_loss", "diversity_loss"):
        assert torch.allclose(joint_logs[key], trajectory_logs[key])
