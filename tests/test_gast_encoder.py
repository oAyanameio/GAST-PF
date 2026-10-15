"""GAST-PF 编码器回归测试。

覆盖：
1. missing_run_deltas 与 UniTraj get_delta 递归逐位一致；
2. M0 默认路径完全不变（encoder_type 默认 'm0'，state_dict 逐位一致）；
3. GAST-PF 真实 TrajGap Mixed batch 前向/反向（GPU）；
4. padding actor 隔离（固定 batch 内扰动 padding 数据不改有效 actor 输出）；
5. evaluator 编码器契约（GAST-PF ckpt 拒载 M0、M0 拒载 GAST-PF、缺键拒绝）；
6. GAST-PF 可训练参数全部进入 AdamW 参数组（优化器闭包纪律）。
"""

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.model.layers.gast_encoder import (  # noqa: E402
    GASTEncoder,
    missing_run_deltas,
)


def _reference_get_delta(m):
    """gap-delta 递推的独立参考实现（前向）。"""
    d = torch.zeros_like(m)
    for t in range(1, m.shape[-1]):
        d[..., t] = 1 + (1 - m[..., t]) * d[..., t - 1]
    return d


def test_missing_run_deltas_matches_reference_recursion():
    torch.manual_seed(0)
    m = (torch.rand(4, 3, 8) > 0.5).float()
    m[..., 0] = 1.0  # TrajGap 保证至少一个有效帧（前向首帧必有效）
    f, b = missing_run_deltas(m.bool())
    uf = _reference_get_delta(m)
    ub = torch.flip(_reference_get_delta(torch.flip(m, dims=[-1])), dims=[-1])
    assert torch.allclose(f, uf, atol=1e-6)
    # GAST keeps the backward delta in the flipped-time layout because it
    # is multiplied with the already flipped backward input sequence.
    assert torch.allclose(b, _reference_get_delta(torch.flip(m, dims=[-1])), atol=1e-6)
    # 全缺尾部与观测点语义
    row = torch.tensor([[[1.0, 0, 0, 0, 0, 0, 0, 0]]])
    f2, _ = missing_run_deltas(row.bool())
    assert f2[0, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def test_m0_default_state_dict_unchanged():
    from src.model.trajectory_forecaster import TrajectoryForecaster

    torch.manual_seed(2024)
    baseline = TrajectoryForecaster(num_modes=6)
    torch.manual_seed(2024)
    default = TrajectoryForecaster(num_modes=6, encoder_type="m0")
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, default.state_dict()[key]), key
    assert not any(key.startswith("gast_encoder.") for key in default.state_dict())


def test_gast_does_not_register_m0_encoder_modules():
    """完整替换臂不应留下 M0 编码器的死参数或重复场景编码器。"""
    from src.model.trajectory_forecaster import TrajectoryForecaster

    gast = TrajectoryForecaster(num_modes=6, encoder_type="gast")
    m0_prefixes = (
        "history_input_projection.", "hist_embed_mamba.", "norm_f.",
        "actor_context_projection.", "scene_interaction_blocks.",
        "norm.", "actor_type_embed",
    )
    assert not any(key.startswith(m0_prefixes) for key in gast.state_dict())
    assert any(key.startswith("gast_encoder.") for key in gast.state_dict())
    assert gast.gast_encoder.temporal_decay.mlp.in_features == 2


def test_gast_rejects_unknown_encoder_type():
    from src.model.trajectory_forecaster import TrajectoryForecaster

    with pytest.raises(ValueError, match="encoder_type"):
        TrajectoryForecaster(encoder_type="gsm")


def test_direct_checkpoint_load_rejects_partial_gast(tmp_path):
    from src.model.trajectory_forecaster import TrajectoryForecaster

    model = TrajectoryForecaster(num_modes=6, encoder_type="gast")
    state = {f"net.{k}": v for k, v in model.state_dict().items()}
    state.pop(next(iter(state)))
    path = tmp_path / "partial.ckpt"
    torch.save({"state_dict": state}, path)
    with pytest.raises(RuntimeError, match="checkpoint/model mismatch"):
        model.load_from_checkpoint(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU（Mamba CUDA）")
def test_gast_real_mixed_batch_forward_backward_smoke():
    from src.datamodule.trajimpute_dataset import (
        TRAJIMPUTE_ROOT, TrajGapDataset, trajimpute_collate_fn,
    )
    from src.model.trajectory_forecaster import TrajectoryForecaster

    device = "cuda:0"
    dataset = TrajGapDataset(TRAJIMPUTE_ROOT, "ETH-M", "train")
    batch = trajimpute_collate_fn([dataset[i] for i in (0, 100, 1000)])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    torch.manual_seed(2024)
    model = TrajectoryForecaster(num_modes=20, encoder_type="gast").to(device)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert out["new_y_hat"].shape == (3, 20, 12, 2)
    assert torch.isfinite(out["new_y_hat"]).all()
    assert torch.isfinite(out["new_pi"]).all()
    # 训练一步：gast_encoder 梯度有限且非零
    model.train()
    out = model(batch)
    loss = out["new_y_hat"].square().mean() + out["new_pi"].square().mean()
    loss.backward()
    grads = [p.grad for p in model.gast_encoder.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum().item() > 0 for g in grads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU（Mamba CUDA）")
def test_gast_padding_actor_isolation():
    """同一 batch 内扰动 padding actor 输入，不改有效 actor 输出。"""
    from src.datamodule.trajimpute_dataset import (
        TRAJIMPUTE_ROOT, TrajGapDataset, trajimpute_collate_fn,
    )
    from src.model.trajectory_forecaster import TrajectoryForecaster

    dataset = TrajGapDataset(TRAJIMPUTE_ROOT, "ETH-M", "train")
    items = [dataset[i] for i in (0, 1)]
    batch = trajimpute_collate_fn(items)
    n_pad = int((~batch["x_key_valid_mask"][0]).sum().item())
    if n_pad == 0:
        pytest.skip("batch 内无 padding actor")
    device = "cuda:0"
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    torch.manual_seed(2024)
    model = TrajectoryForecaster(num_modes=20, encoder_type="gast").to(device).eval()
    with torch.no_grad():
        base = model(batch)["new_y_hat"]
        perturbed = dict(batch)
        perturbed["x_positions_diff"] = batch["x_positions_diff"].clone()
        # padding actor 行（key_valid=False）注入大值
        pad_rows = ~batch["x_key_valid_mask"]
        perturbed["x_positions_diff"][pad_rows] += 100.0
        pert = model(perturbed)["new_y_hat"]
    assert torch.allclose(base, pert, atol=1e-4), "padding actor 泄漏进有效输出"


def test_evaluator_encoder_contract(tmp_path):
    from scripts.结果分析.evaluate_trajimpute_direct import build_model, load_checkpoint

    torch.manual_seed(2024)
    m0 = build_model("M0", num_modes=20)
    gast = build_model("GAST-PF", num_modes=20)

    # M0 checkpoint 拒载入 GAST-PF：缺 gast_encoder.* 键
    ckpt = tmp_path / "m0.ckpt"
    torch.save({
        "state_dict": {f"net.{k}": v for k, v in m0.state_dict().items()},
        "hyper_parameters": {"model": {"num_modes": 20, "encoder_type": "m0"}},
    }, ckpt)
    with pytest.raises(ValueError, match="encoder_type"):
        load_checkpoint(build_model("GAST-PF", num_modes=20), ckpt)

    # GAST-PF checkpoint 拒载入 M0（encoder_type 契约先行拒绝）
    ckpt_gast = tmp_path / "gast.ckpt"
    torch.save({
        "state_dict": {f"net.{k}": v for k, v in gast.state_dict().items()},
        "hyper_parameters": {"model": {"num_modes": 20, "encoder_type": "gast"}},
    }, ckpt_gast)
    with pytest.raises(ValueError, match="encoder_type"):
        load_checkpoint(build_model("M0", num_modes=20), ckpt_gast)

    # GAST-PF checkpoint 键完整时可正常加载。
    state = {f"net.{k}": v for k, v in gast.state_dict().items()}
    ckpt_gast2 = tmp_path / "gast_pure.ckpt"
    torch.save({
        "state_dict": state,
        "hyper_parameters": {"model": {"num_modes": 20, "encoder_type": "gast"}},
    }, ckpt_gast2)
    missing, unexpected = load_checkpoint(build_model("GAST-PF", num_modes=20),
                                          ckpt_gast2)
    assert not missing
    assert not unexpected

    # A partial GAST-PF checkpoint must not be evaluated under the full model.
    partial = dict(state)
    partial.pop(next(iter(partial)))
    ckpt_partial = tmp_path / "gast_partial.ckpt"
    torch.save({
        "state_dict": partial,
        "hyper_parameters": {"model": {"num_modes": 20, "encoder_type": "gast"}},
    }, ckpt_partial)
    with pytest.raises(ValueError, match="checkpoint/model mismatch"):
        load_checkpoint(build_model("GAST-PF", num_modes=20), ckpt_partial)


def test_evaluator_accepts_lightning_omegaconf_model_config(tmp_path):
    from omegaconf import OmegaConf
    from scripts.结果分析.evaluate_trajimpute_direct import build_model, load_checkpoint

    model = build_model("GAST-PF", num_modes=20)
    path = tmp_path / "gast_lightning.ckpt"
    torch.save({
        "state_dict": {f"net.{k}": v for k, v in model.state_dict().items()},
        "hyper_parameters": {
            "model": OmegaConf.create({
                "num_modes": 20,
                "encoder_type": "gast",
                "gast_depth": 4,
                "gast_gsm_operator": "max",
            })
        },
    }, path)
    missing, unexpected = load_checkpoint(build_model("GAST-PF", num_modes=20), path)
    assert not missing
    assert not unexpected


def test_gast_parameters_covered_by_adamw_groups():
    from src.model.forecasting_module import ForecastingLightningModule

    module = ForecastingLightningModule(
        model={
            "type": "TrajectoryForecaster",
            "embed_dim": 128, "future_steps": 12, "num_heads": 8,
            "mlp_ratio": 4.0, "qkv_bias": False, "drop_path": 0.2,
            "num_actor_types": 1, "num_modes": 20, "dt": 0.4, "obs_len": 8,
            "bimamba": False, "encoder_type": "gast",
        },
        lr=1e-3, warmup_epochs=5, epochs=100, weight_decay=1e-4,
    )
    optimizer = module.configure_optimizers()[0][0]
    trainable = {id(p) for p in module.parameters() if p.requires_grad}
    grouped = [p for group in optimizer.param_groups for p in group["params"]]
    assert {id(p) for p in grouped} == trainable
    gast_ids = {id(p) for name, p in module.named_parameters()
                if name.startswith("net.gast_encoder.") and p.requires_grad}
    assert gast_ids
    assert gast_ids.issubset({id(p) for p in grouped})
