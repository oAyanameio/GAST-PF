"""Stage 2 · FrameGeo 行为测试（先写测试，后实现）。

Stage 2 目标：把逐帧位置几何显式送入 GSM，而不是只在最终 actor token 上补位置。

新增特征（路线文档 §4 Stage 2）：
    relative_position_i,t = x_positions_i,t - x_centers_i
历史输入扩展为 `[x_positions_diff(2), x_velocity_diff(1), valid_mask(1), relative_position(2)]`
即 `input_dim=6`。

纪律：
1. 缺失帧的几何输入必须置零（`x_positions` 在缺失帧是 0 占位，直接相减会得到
   `-x_centers` 的伪几何），但仍保留 `valid_mask` 通道；
2. 不得使用未来 target 或插值位置；
3. 不改 decoder / loss / 协议；
4. 判定要求：打乱相对位置必须改变输出（说明几何被使用），而 actor 置换不得改变
   focal 输出（说明没有引入槽位依赖）。
"""

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU（Mamba CUDA）")


# ---------------------------------------------------------------- 特征构造（CPU）

def _fake_data(B=2, N=3, T=8):
    torch.manual_seed(0)
    valid = torch.ones(B, N, T, dtype=torch.bool)
    # 制造缺失：actor 1 的第 3-5 帧
    valid[:, 1, 3:6] = False
    positions = torch.randn(B, N, T, 2)
    # 缺失帧按数据层约定为 0 占位
    positions[~valid] = 0.0
    centers = positions[:, :, -1].clone()
    return {
        "x_positions": positions,
        "x_centers": centers,
        "x_valid_mask": valid,
    }


def test_frame_geo_features_zero_at_missing_frames():
    """缺失帧的相对位置必须为 0（不能是 -x_centers 的伪几何）。"""
    from src.model.trajectory_forecaster import TrajectoryForecaster

    data = _fake_data()
    rel = TrajectoryForecaster._frame_geo_features(data, data["x_valid_mask"])
    assert rel.shape == (2, 3, 8, 2)
    missing = ~data["x_valid_mask"]
    assert torch.equal(rel[missing], torch.zeros_like(rel[missing]))
    # 有效帧 = x_positions - x_centers
    expect = data["x_positions"] - data["x_centers"].unsqueeze(2)
    valid = data["x_valid_mask"]
    assert torch.allclose(rel[valid], expect[valid], atol=1e-6)


def test_frame_geo_features_finite_for_all_invalid_actor():
    """全无效 actor（padding 的极端情形）不得产生 NaN/Inf。"""
    from src.model.trajectory_forecaster import TrajectoryForecaster

    data = _fake_data()
    data["x_valid_mask"][:, 2, :] = False
    rel = TrajectoryForecaster._frame_geo_features(data, data["x_valid_mask"])
    assert torch.isfinite(rel).all()
    assert torch.equal(rel[:, 2], torch.zeros_like(rel[:, 2]))


# ---------------------------------------------------------------- 结构契约

def test_frame_geo_disabled_by_default():
    """默认关闭：不改变输入宽度，也不注册额外参数。"""
    from src.model.trajectory_forecaster import TrajectoryForecaster

    base = TrajectoryForecaster(num_modes=6, encoder_type="gast")
    assert getattr(base, "gast_frame_geo") is False
    first = next(m for m in base.gast_encoder.spatial.patch_embed
                 if isinstance(m, torch.nn.Linear))
    assert first.in_features == 4


def test_frame_geo_enables_six_dim_input():
    """开启后 encoder 输入宽度为 6，且不改变 GSM 内部宽度。"""
    from src.model.trajectory_forecaster import TrajectoryForecaster

    model = TrajectoryForecaster(num_modes=6, encoder_type="gast",
                                 gast_frame_geo=True)
    assert model.gast_frame_geo is True
    first = next(m for m in model.gast_encoder.spatial.patch_embed
                 if isinstance(m, torch.nn.Linear))
    assert first.in_features == 6
    assert first.out_features == 128


def test_frame_geo_not_applied_to_m0_path():
    """M0 对照路径不受 FrameGeo 影响（历史投影仍为 4 维）。"""
    from src.model.trajectory_forecaster import TrajectoryForecaster

    m0 = TrajectoryForecaster(num_modes=6, encoder_type="m0",
                              gast_frame_geo=True)
    first = next(m for m in m0.history_input_projection
                 if isinstance(m, torch.nn.Linear))
    assert first.in_features == 4


# ---------------------------------------------------------------- 行为测试（GPU）

def _load_batch(idx=(0, 100)):
    from src.datamodule.trajimpute_dataset import (
        TRAJIMPUTE_ROOT, TrajGapDataset, trajimpute_collate_fn,
    )
    ds = TrajGapDataset(TRAJIMPUTE_ROOT, "ETH-M", "train")
    batch = trajimpute_collate_fn([ds[i] for i in idx])
    device = "cuda:0"
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _model(seed=2024):
    from src.model.trajectory_forecaster import TrajectoryForecaster
    torch.manual_seed(seed)
    return TrajectoryForecaster(num_modes=20, encoder_type="gast",
                                gast_frame_geo=True).cuda().eval()


@GPU
def test_frame_geo_relative_position_shuffle_changes_output():
    """判定条件一：打乱相对位置必须改变输出（几何确实被使用）。"""
    batch = _load_batch()
    model = _model()
    with torch.no_grad():
        base = model(batch)["new_y_hat"]
        pert = dict(batch)
        # 逐帧位置在时间轴上轮转 -> 相对位置整体改变
        pert["x_positions"] = batch["x_positions"].roll(1, dims=2)
        got = model(pert)["new_y_hat"]
    assert not torch.allclose(base, got, atol=1e-4), "相对位置未被使用"


@GPU
def test_frame_geo_actor_permutation_invariant():
    """判定条件二：邻居置换不得改变 focal 输出（无槽位依赖）。"""
    batch = _load_batch(idx=(0,))
    n = batch["x_valid_mask"].shape[1]
    if n < 3:
        pytest.skip("actor 数不足")
    model = _model()
    perm = torch.arange(n)
    perm[1:] = perm[1:].roll(1)

    def shuffled(b):
        out = dict(b)
        for key in ("x_positions_diff", "x_velocity_diff", "x_valid_mask",
                    "x_key_valid_mask", "x_positions", "x_centers",
                    "x_last_valid_angle", "x_last_valid_idx", "x_angles",
                    "x_attr", "x_velocity", "x_anchor_lag_steps",
                    "x_forecast_gap_steps", "x_gap_steps"):
            if key in b and torch.is_tensor(b[key]):
                out[key] = b[key][:, perm]
        return out

    with torch.no_grad():
        base = model(batch)["new_y_hat"]
        got = model(shuffled(batch))["new_y_hat"]
    assert torch.allclose(base, got, atol=1e-3), "邻居置换改变了 focal 输出"


@GPU
def test_frame_geo_missing_placeholder_is_ignored():
    """缺失帧的 0 占位被改写不得影响输出（几何已按 mask 置零）。"""
    batch = _load_batch()
    model = _model()
    missing = ~batch["x_valid_mask"]
    if int(missing.sum()) == 0:
        pytest.skip("batch 内无缺失帧")
    with torch.no_grad():
        base = model(batch)["new_y_hat"]
        pert = dict(batch)
        pert["x_positions"] = batch["x_positions"].clone()
        pert["x_positions"][missing] += 37.0
        got = model(pert)["new_y_hat"]
    assert torch.allclose(base, got, atol=1e-5), "缺失帧占位泄漏进几何特征"


@GPU
def test_frame_geo_padding_actor_isolation():
    """扰动 padding actor 的逐帧位置，有效 actor 输出不变。"""
    batch = _load_batch(idx=(0, 1))
    pad = ~batch["x_key_valid_mask"]
    if int(pad.sum()) == 0:
        pytest.skip("batch 内无 padding actor")
    model = _model()
    with torch.no_grad():
        base = model(batch)["new_y_hat"]
        pert = dict(batch)
        pert["x_positions"] = batch["x_positions"].clone()
        pert["x_centers"] = batch["x_centers"].clone()
        pert["x_positions"][pad] += 100.0
        pert["x_centers"][pad] += 100.0
        got = model(pert)["new_y_hat"]
    assert torch.allclose(base, got, atol=1e-4), "padding actor 几何泄漏"


@GPU
def test_frame_geo_ignores_future_targets():
    """末端缺失不泄漏：改动 future target 不得影响输出。"""
    batch = _load_batch()
    model = _model()
    with torch.no_grad():
        base = model(batch)["new_y_hat"]
        pert = dict(batch)
        pert["target"] = batch["target"] + 50.0
        got = model(pert)["new_y_hat"]
    assert torch.allclose(base, got, atol=1e-5), "future target 泄漏"


@GPU
def test_frame_geo_finite_on_edge_fixtures():
    """全有效 / 末端缺失 / 中间缺失 / 最小证据 fixtures 输出必须有限。"""
    batch = _load_batch()
    model = _model()
    m0 = torch.ones_like(batch["x_valid_mask"])
    terminal = m0.clone(); terminal[:, :, -3:] = False; terminal[:, 0, 0] = True
    middle = m0.clone(); middle[:, :, 3:5] = False; middle[:, 0, 0] = True
    minimal = torch.zeros_like(batch["x_valid_mask"]); minimal[:, 0, 0] = True
    fixtures = {"all_valid": m0, "terminal": terminal, "middle": middle,
                "minimal": minimal}
    with torch.no_grad():
        for name, m in fixtures.items():
            fx = dict(batch)
            fx["x_valid_mask"] = m
            out = model(fx)
            assert torch.isfinite(out["new_y_hat"]).all(), name
            assert torch.isfinite(out["new_pi"]).all(), name


@GPU
def test_frame_geo_decoder_contract_unchanged():
    """decoder 输入维度与输出契约不变。"""
    batch = _load_batch()
    model = _model()
    with torch.no_grad():
        out = model(batch)
    assert out["new_y_hat"].shape == (2, 20, 12, 2)
    assert out["new_pi"].shape == (2, 20)


# ---------------------------------------------------------------- 契约与优化器

def test_evaluator_frame_geo_contract(tmp_path):
    """GAST-PF 与 GAST-PF-FrameGeo 的 checkpoint 互不通用。"""
    from scripts.结果分析.evaluate_trajimpute_direct import build_model, load_checkpoint

    base = build_model("GAST-PF", num_modes=20)
    geo = build_model("GAST-PF-FrameGeo", num_modes=20)
    assert getattr(base, "gast_frame_geo") is False
    assert getattr(geo, "gast_frame_geo") is True

    def save(model, name):
        path = tmp_path / name
        torch.save({
            "state_dict": {f"net.{k}": v for k, v in model.state_dict().items()},
            "hyper_parameters": {"model": {
                "num_modes": 20, "encoder_type": "gast",
                "gast_frame_geo": model.gast_frame_geo,
                "model_version": "GAST-PF-FrameGeo" if model.gast_frame_geo
                                 else "GAST-PF",
            }},
        }, path)
        return path

    geo_ckpt = save(geo, "frame_geo.ckpt")
    with pytest.raises(ValueError, match="gast_frame_geo"):
        load_checkpoint(build_model("GAST-PF", num_modes=20), geo_ckpt)
    base_ckpt = save(base, "base.ckpt")
    with pytest.raises(ValueError, match="gast_frame_geo"):
        load_checkpoint(build_model("GAST-PF-FrameGeo", num_modes=20), base_ckpt)
    missing, unexpected = load_checkpoint(
        build_model("GAST-PF-FrameGeo", num_modes=20), geo_ckpt)
    assert not missing and not unexpected


def test_frame_geo_parameters_covered_by_adamw_groups():
    """FrameGeo 变体的全部可训练参数进入 AdamW 参数组。"""
    from src.model.forecasting_module import ForecastingLightningModule

    module = ForecastingLightningModule(
        model={
            "type": "TrajectoryForecaster",
            "embed_dim": 128, "future_steps": 12, "num_heads": 8,
            "mlp_ratio": 4.0, "qkv_bias": False, "drop_path": 0.2,
            "num_actor_types": 1, "num_modes": 20, "dt": 0.4, "obs_len": 8,
            "bimamba": False, "encoder_type": "gast", "gast_frame_geo": True,
        },
        lr=1e-3, warmup_epochs=5, epochs=100, weight_decay=1e-4,
    )
    optimizer = module.configure_optimizers()[0][0]
    trainable = {id(p) for p in module.parameters() if p.requires_grad}
    grouped = [p for group in optimizer.param_groups for p in group["params"]]
    assert {id(p) for p in grouped} == trainable
