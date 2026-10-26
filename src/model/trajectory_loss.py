import torch
import torch.nn.functional as F


def final_consistent_loss(
    out: dict,
    data: dict,
    laplace_loss,
    endpoint_loss_weight: float = 1.0,
    mode_diversity_loss_weight: float = 0.05,
    mode_diversity_margin: float = 0.5,
    classification_target_mode: str = "joint",
    tag: str = "",
):
    if classification_target_mode not in {"joint", "trajectory"}:
        raise ValueError(
            "classification_target_mode must be 'joint' or 'trajectory', "
            f"got {classification_target_mode!r}"
        )
    prediction = (
        out["new_y_hat"] if out.get("new_y_hat") is not None else out["y_hat"]
    )
    probability = (
        out["new_pi"] if out.get("new_pi") is not None else out["pi"]
    )
    scale = out["scal_new"] if out.get("scal_new") is not None else out["scal"]
    target = data["target"][:, 0]
    focal_mask = data["target_mask"][:, 0].bool()
    batch_indices = torch.arange(target.shape[0], device=target.device)

    point_error = torch.linalg.vector_norm(
        prediction[..., :2] - target[:, None], dim=-1
    )
    point_error = point_error * focal_mask[:, None].to(point_error.dtype)
    trajectory_error_by_mode = point_error.sum(dim=-1) / focal_mask.sum(
        dim=-1
    ).clamp(min=1)[:, None]

    endpoint_target = target[:, -1]
    endpoint_candidates = out.get("goal_endpoints")
    if endpoint_candidates is None:
        endpoint_candidates = prediction[..., -1, :2]
    endpoint_error_by_mode = torch.linalg.vector_norm(
        endpoint_candidates - endpoint_target[:, None], dim=-1
    )
    endpoint_valid = focal_mask[:, -1]
    endpoint_error_by_mode = endpoint_error_by_mode * endpoint_valid[
        :, None
    ].to(endpoint_error_by_mode.dtype)
    joint_mode_error = (
        trajectory_error_by_mode
        + endpoint_loss_weight * endpoint_error_by_mode
    )
    best_mode = joint_mode_error.argmin(dim=-1)
    classification_mode_error = (
        joint_mode_error
        if classification_target_mode == "joint"
        else trajectory_error_by_mode
    )
    classification_mode = classification_mode_error.argmin(dim=-1)

    best_prediction = prediction[batch_indices, best_mode, :, :2]
    trajectory_error = F.smooth_l1_loss(
        best_prediction, target, reduction="none"
    ).mean(dim=-1)
    valid_point_count = focal_mask.sum(dim=-1).clamp(min=1)
    agent_reg_loss = (
        trajectory_error * focal_mask.to(trajectory_error.dtype)
    ).sum() / valid_point_count.sum()
    agent_cls_loss = F.cross_entropy(
        probability, classification_mode.detach(), label_smoothing=0.2
    )

    endpoint_prediction = endpoint_candidates[batch_indices, best_mode]
    endpoint_error = F.smooth_l1_loss(
        endpoint_prediction, endpoint_target, reduction="none"
    ).mean(dim=-1)
    endpoint_loss = (
        endpoint_error * endpoint_valid.to(endpoint_error.dtype)
    ).sum() / endpoint_valid.sum().clamp(min=1)

    if prediction.shape[1] > 1:
        pairwise_distance = torch.cdist(endpoint_candidates, endpoint_candidates)
        pairwise_mask = ~torch.eye(
            prediction.shape[1], dtype=torch.bool, device=prediction.device
        )
        diversity_loss = F.relu(
            mode_diversity_margin - pairwise_distance
        )[:, pairwise_mask].mean()
    else:
        diversity_loss = prediction.new_zeros(())

    target_others = data["target"][:, 1:]
    prediction_others = out["y_hat_others"]
    others_mask = data["target_mask"][:, 1:].bool()
    if others_mask.any():
        others_reg_loss = F.smooth_l1_loss(
            prediction_others[others_mask], target_others[others_mask]
        )
    else:
        others_reg_loss = prediction_others.new_zeros(())

    laplace = laplace_loss.compute(
        {
            "traj": prediction,
            "scale": scale,
            "probs": probability,
        },
        target,
    )
    loss = (
        agent_reg_loss
        + agent_cls_loss
        + others_reg_loss
        + laplace
        + endpoint_loss_weight * endpoint_loss
        + mode_diversity_loss_weight * diversity_loss
    )
    disp_dict = {
        f"{tag}loss": loss.detach(),
        f"{tag}reg_loss": agent_reg_loss.detach(),
        f"{tag}cls_loss": agent_cls_loss.detach(),
        f"{tag}others_reg_loss": others_reg_loss.detach(),
        f"{tag}laplace_loss": laplace.detach(),
        f"{tag}endpoint_loss": endpoint_loss.detach(),
        f"{tag}diversity_loss": diversity_loss.detach(),
    }
    return loss, disp_dict
