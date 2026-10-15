"""Regression tests for the M0 AdamW parameter-group policy."""

from src.model.forecasting_module import ForecastingLightningModule


def _make_module():
    return ForecastingLightningModule(
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
        },
        lr=1e-3,
        warmup_epochs=5,
        epochs=100,
        weight_decay=1e-4,
    )


def test_m0_adamw_groups_cover_each_trainable_parameter_once():
    module = _make_module()
    optimizers, _ = module.configure_optimizers()
    optimizer = optimizers[0]

    trainable = {
        id(param): name
        for name, param in module.named_parameters()
        if param.requires_grad
    }
    grouped = [
        param
        for group in optimizer.param_groups
        for param in group["params"]
    ]
    grouped_ids = [id(param) for param in grouped]

    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == set(trainable)


def test_m0_custom_norm_weights_are_in_optimizer():
    module = _make_module()
    optimizers, _ = module.configure_optimizers()
    norm_params = [
        param
        for _, child in module.named_modules()
        if type(child).__name__ == "RMSNorm"
        for _, param in child.named_parameters(recurse=False)
        if param.requires_grad
    ]
    optimizer_ids = {
        id(param)
        for group in optimizers[0].param_groups
        for param in group["params"]
    }

    no_decay_ids = {
        id(param)
        for group in optimizers[0].param_groups
        if group["weight_decay"] == 0.0
        for param in group["params"]
    }
    assert norm_params
    assert all(id(param) in optimizer_ids for param in norm_params)
    assert all(id(param) in no_decay_ids for param in norm_params)
