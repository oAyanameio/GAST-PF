"""Stage 1 · AnchorGeo 行为测试（先写测试，后实现）。

Stage 1 目标：最低风险地把 anchor 空间与末端方向信息接入 GAST-PF 的 actor 输出端，
验证「空间几何缺失」是否为主要瓶颈。

纪律（路线文档 §5.3 / Stage 1）：
1. 不改变 GSM attention；只在 actor 输出端加几何投影；
2. 保持 decoder 输入维度不变（[B, N, embed_dim]）；
3. padding actor 的几何输入与输出显式置零；
4. 几何量只来自可见历史，不得使用 future target / future validity；
5. 零初始化输出层 → 训练起点与 GAST-PF 严格等价（可归因）。
"""

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.model.layers.gast_encoder import anchor_lag_norm  # noqa: E402

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU（Mamba CUDA）")


# ---------------------------------------------------------------- anchor_lag

def test_anchor_lag_norm_matches_spec():
    """anchor_lag = (T-1) - last_valid_index，再除以 max(T-1, 1)。"""
    # 8 帧：actor0 末帧有效 -> lag 0；actor1 缺末 3 帧 -> lag 3
    mask = torch.ones(1, 2, 8, dtype=torch.bool)
    mask[0, 1, 5:] = False
    lag = anchor_lag_norm(mask)
    assert lag.shape == (1, 2, 1)
    assert lag[0, 0, 0].item() == pytest.approx(0.0)
    assert lag[0, 1, 0].item() == pytest.approx(3.0 / 7.0)


def test_anchor_lag_norm_is_monotone_in_run_length():
    """连续缺失越长，anchor_lag_norm 越大（单调性）。"""
    vals = []
    for run in range(0, 8):
        mask = torch.ones(1, 1, 8, dtype=torch.bool)
        if run:
            mask[0, 0, 8 - run:] = False
        vals.append(anchor_lag_norm(mask)[0, 0, 0].item())
    assert vals == sorted(vals)
    assert vals[0] == 0.0 and vals[-1] == pytest.approx(1.0)


def test_anchor_lag_norm_stays_finite_for_all_invalid_rows():
    """全无效行（padding actor 的极端情形）不得产生 NaN/Inf 或 >1 的值。"""
    mask = torch.zeros(1, 2, 8, dtype=torch.bool)
    mask[0, 0, 0] = True
    lag = anchor_lag_norm(mask)
    assert torch.isfinite(lag).all()
    assert (lag >= 0).all() and (lag <= 1).all()


# ---------------------------------------------------------------- 结构契约

def test_anchor_geo_disabled_by_default():
    """基础版本（Stage 0 冻结）默认不带 AnchorGeo 参数，state_dict 不变。"""
    from src.model.trajectory_forecaster import TrajectoryForecaster

    model = TrajectoryForecaster(num_modes=6, encoder_type="gast")
    assert not any("anchor_geo" in k for k in model.state_dict())


def test_anchor_geo_registers_projection_only_when_enabled():
    from src.model.trajectory_forecaster import TrajectoryForecaster

    model = TrajectoryForecaster(num_modes=6, encoder_type="gast",
                                 gast_anchor_geo=True)
    keys = [k for k in model.state_dict() if "anchor_geo" in k]
    assert keys, "开启后应注册 AnchorGeo 投影参数"
    # M0 对照路径不得出现 AnchorGeo。
    m0 = TrajectoryForecaster(num_modes=6, encoder_type="m0",
                              gast_anchor_geo=True)
    assert not any("anchor_geo" in k for k in m0.state_dict())


def test_anchor_geo_projection_input_width_is_five():
    """几何输入契约：[x_centers(2), cos/sin(angle)(2), anchor_lag_norm(1)]。"""
    from src.model.trajectory_forecaster import TrajectoryForecaster

    model = TrajectoryForecaster(num_modes=6, encoder_type="gast",
                                 gast_anchor_geo=True)
    proj = model.anchor_geo_projection
    first = next(m for m in proj.modules() if isinstance(m, torch.nn.Linear))
    assert first.in_features == 5


@GPU
def test_anchor_geo_zero_init_identity_and_paired_weights():
    """零初始化：训练起点输出与基础版本逐位一致，且共享参数逐位一致。"""
    from src.datamodule.trajimpute_dataset import (
        TRAJIMPUTE_ROOT, TrajGapDataset, trajimpute_collate_fn,
    )
    from src.model.trajectory_forecaster import TrajectoryForecaster

    dataset = TrajGapDataset(TRAJIMPUTE_ROOT, "ETH-M", "train")
    batch = trajimpute_collate_fn([dataset[i] for i in (0, 100)])
    device = "cuda:0"
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    torch.manual_seed(2024)
    base = TrajectoryForecaster(num_modes=20, encoder_type="gast").to(device).eval()
    torch.manual_seed(2024)
    geo = TrajectoryForecaster(num_modes=20, encoder_type="gast",
                               gast_anchor_geo=True).to(device).eval()

    base_sd = base.state_dict()
    for key, value in geo.state_dict().items():
        if "anchor_geo" in key:
            continue
        assert torch.equal(value, base_sd[key]), key

    with torch.no_grad():
        out_base = base(batch)["new_y_hat"]
        out_geo = geo(batch)["new_y_hat"]
    assert torch.equal(out_base, out_geo), "零初始化下 AnchorGeo 必须严格恒等"


@GPU
def test_anchor_geo_preserves_decoder_contract():
    """decoder 输入维度与输出契约不变。"""
    from src.datamodule.trajimpute_dataset import (
        TRAJIMPUTE_ROOT, TrajGapDataset, trajimpute_collate_fn,
    )
    from src.model.trajectory_forecaster import TrajectoryForecaster

    dataset = TrajGapDataset(TRAJIMPUTE_ROOT, "ETH-M", "train")
    batch = trajimpute_collate_fn([dataset[i] for i in (0, 1)])
    device = "cuda:0"
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    torch.manual_seed(2024)
    model = TrajectoryForecaster(num_modes=20, encoder_type="gast",
                                 gast_anchor_geo=True).to(device).eval()
    with torch.no_grad():
        out = model(batch)
    assert out["new_y_hat"].shape == (2, 20, 12, 2)
    assert out["new_pi"].shape == (2, 20)
    assert torch.isfinite(out["new_y_hat"]).all()


# ---------------------------------------------------------------- 行为测试

@GPU
def test_anchor_geo_padding_actor_isolation():
    """扰动 padding actor 的几何输入，有效 actor 输出不变。"""
    from src.datamodule.trajimpute_dataset import (
        TRAJIMPUTE_ROOT, TrajGapDataset, trajimpute_collate_fn,
    )
    from src.model.trajectory_forecaster import TrajectoryForecaster

    dataset = TrajGapDataset(TRAJIMPUTE_ROOT, "ETH-M", "train")
    batch = trajimpute_collate_fn([dataset[i] for i in (0, 1)])
    pad = ~batch["x_key_valid_mask"]
    if int(pad.sum()) == 0:
        pytest.skip("batch 内无 padding actor")
    device = "cuda:0"
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    torch.manual_seed(2024)
    model = TrajectoryForecaster(num_modes=20, encoder_type="gast",
                                 gast_anchor_geo=True).to(device).eval()
    # 打开几何分支的贡献：把输出层权重设为非零，否则恒等会掩盖泄漏。
    with torch.no_grad():
        for p in model.anchor_geo_projection.parameters():
            p.add_(0.05)

    with torch.no_grad():
        base = model(batch)["new_y_hat"]
        pert = dict(batch)
        pert["x_centers"] = batch["x_centers"].clone()
        pert["x_last_valid_angle"] = batch["x_last_valid_angle"].clone()
        pert["x_centers"][pad] += 100.0
        pert["x_last_valid_angle"][pad] += 3.0
        got = model(pert)["new_y_hat"]
    assert torch.allclose(base, got, atol=1e-4), "padding actor 几何泄漏进有效输出"


@GPU
def test_anchor_geo_geometry_is_used():
    """几何有效性：打乱真实几何后输出必须发生可解释变化（不能是死分支）。"""
    from src.datamodule.trajimpute_dataset import (
        TRAJIMPUTE_ROOT, TrajGapDataset, trajimpute_collate_fn,
    )
    from src.model.trajectory_forecaster import TrajectoryForecaster

    dataset = TrajGapDataset(TRAJIMPUTE_ROOT, "ETH-M", "train")
    batch = trajimpute_collate_fn([dataset[i] for i in (0, 100)])
    device = "cuda:0"
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    torch.manual_seed(2024)
    model = TrajectoryForecaster(num_modes=20, encoder_type="gast",
                                 gast_anchor_geo=True).to(device).eval()
    with torch.no_grad():
        for p in model.anchor_geo_projection.parameters():
            p.add_(0.05)

    with torch.no_grad():
        base = model(batch)["new_y_hat"]
        pert = dict(batch)
        pert["x_centers"] = batch["x_centers"].roll(1, dims=1) + 5.0
        got = model(pert)["new_y_hat"]
    assert not torch.allclose(base, got, atol=1e-4), "几何分支未被使用"


@GPU
def test_anchor_geo_ignores_future_targets():
    """末端缺失不泄漏：改动 future target 不得影响编码器输出。"""
    from src.datamodule.trajimpute_dataset import (
        TRAJIMPUTE_ROOT, TrajGapDataset, trajimpute_collate_fn,
    )
    from src.model.trajectory_forecaster import TrajectoryForecaster

    dataset = TrajGapDataset(TRAJIMPUTE_ROOT, "ETH-M", "train")
    batch = trajimpute_collate_fn([dataset[i] for i in (0, 100)])
    device = "cuda:0"
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    torch.manual_seed(2024)
    model = TrajectoryForecaster(num_modes=20, encoder_type="gast",
                                 gast_anchor_geo=True).to(device).eval()
    with torch.no_grad():
        for p in model.anchor_geo_projection.parameters():
            p.add_(0.05)

    with torch.no_grad():
        base = model(batch)["new_y_hat"]
        pert = dict(batch)
        pert["target"] = batch["target"] + 50.0
        got = model(pert)["new_y_hat"]
    assert torch.allclose(base, got, atol=1e-5), "future target 泄漏进编码器"


@GPU
def test_anchor_geo_permutation_equivariance():
    """集合不变性：邻居无序 —— 同步置换邻居输入、mask 与几何，focal 输出不变。"""
    from src.datamodule.trajimpute_dataset import (
        TRAJIMPUTE_ROOT, TrajGapDataset, trajimpute_collate_fn,
    )
    from src.model.trajectory_forecaster import TrajectoryForecaster

    dataset = TrajGapDataset(TRAJIMPUTE_ROOT, "ETH-M", "train")
    items = [dataset[0]]
    batch = trajimpute_collate_fn(items)
    n = batch["x_valid_mask"].shape[1]
    if n < 3:
        pytest.skip("actor 数不足")
    device = "cuda:0"
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    torch.manual_seed(2024)
    model = TrajectoryForecaster(num_modes=20, encoder_type="gast",
                                 gast_anchor_geo=True).to(device).eval()
    with torch.no_grad():
        for p in model.anchor_geo_projection.parameters():
            p.add_(0.05)

    # focal(0) 保持在 0 位；对 1..n-1 做轮转（邻居无序）
    perm = torch.arange(n)
    perm[1:] = perm[1:].roll(1)

    def shuffled(b):
        out = dict(b)
        for key in ("x_positions_diff", "x_velocity_diff", "x_valid_mask",
                    "x_key_valid_mask", "x_centers", "x_last_valid_angle",
                    "x_last_valid_idx", "x_angles", "x_attr", "x_velocity",
                    "x_anchor_lag_steps", "x_forecast_gap_steps", "x_gap_steps"):
            if key in b and torch.is_tensor(b[key]):
                out[key] = b[key][:, perm]
        return out

    with torch.no_grad():
        base = model(batch)["new_y_hat"]
        got = model(shuffled(batch))["new_y_hat"]
    assert torch.allclose(base, got, atol=1e-3), "邻居置换改变了 focal 输出"


@GPU
def test_anchor_geo_finite_on_edge_fixtures():
    """全有效 / 末端缺失 / 中间缺失 / 最小证据 fixtures 输出必须有限。"""
    from src.datamodule.trajimpute_dataset import (
        TRAJIMPUTE_ROOT, TrajGapDataset, trajimpute_collate_fn,
    )
    from src.model.trajectory_forecaster import TrajectoryForecaster

    dataset = TrajGapDataset(TRAJIMPUTE_ROOT, "ETH-M", "train")
    batch = trajimpute_collate_fn([dataset[i] for i in (0, 100)])
    device = "cuda:0"
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    torch.manual_seed(2024)
    model = TrajectoryForecaster(num_modes=20, encoder_type="gast",
                                 gast_anchor_geo=True).to(device).eval()

    fixtures = {}
    all_valid = dict(batch)
    all_valid["x_valid_mask"] = torch.ones_like(batch["x_valid_mask"])
    fixtures["all_valid"] = all_valid

    terminal = dict(batch)
    m = torch.ones_like(batch["x_valid_mask"])
    m[:, :, -3:] = False
    m[:, 0, 0] = True
    terminal["x_valid_mask"] = m
    fixtures["terminal_missing"] = terminal

    middle = dict(batch)
    m = torch.ones_like(batch["x_valid_mask"])
    m[:, :, 3:5] = False
    m[:, 0, 0] = True
    middle["x_valid_mask"] = m
    fixtures["middle_gap"] = middle

    minimal = dict(batch)
    m = torch.zeros_like(batch["x_valid_mask"])
    m[:, 0, 0] = True
    minimal["x_valid_mask"] = m
    fixtures["minimal_evidence"] = minimal

    with torch.no_grad():
        for name, fx in fixtures.items():
            out = model(fx)
            assert torch.isfinite(out["new_y_hat"]).all(), name
            assert torch.isfinite(out["new_pi"]).all(), name


def test_evaluator_anchor_geo_contract(tmp_path):
    """GAST-PF 与 GAST-PF-AnchorGeo 的 checkpoint 互不通用（契约 fail-closed）。"""
    from scripts.结果分析.evaluate_trajimpute_direct import build_model, load_checkpoint

    base = build_model("GAST-PF", num_modes=20)
    geo = build_model("GAST-PF-AnchorGeo", num_modes=20)
    assert getattr(base, "gast_anchor_geo") is False
    assert getattr(geo, "gast_anchor_geo") is True

    def save(model, name):
        path = tmp_path / name
        torch.save({
            "state_dict": {f"net.{k}": v for k, v in model.state_dict().items()},
            "hyper_parameters": {"model": {
                "num_modes": 20, "encoder_type": "gast",
                "gast_anchor_geo": model.gast_anchor_geo,
                "model_version": "GAST-PF-AnchorGeo" if model.gast_anchor_geo
                                 else "GAST-PF",
            }},
        }, path)
        return path

    geo_ckpt = save(geo, "geo.ckpt")
    # AnchorGeo ckpt 不得载入基础 GAST-PF 模型（几何参数丢失）。
    with pytest.raises(ValueError, match="gast_anchor_geo"):
        load_checkpoint(build_model("GAST-PF", num_modes=20), geo_ckpt)
    # 反向：基础 ckpt 不得载入 AnchorGeo 模型。
    base_ckpt = save(base, "base.ckpt")
    with pytest.raises(ValueError, match="gast_anchor_geo"):
        load_checkpoint(build_model("GAST-PF-AnchorGeo", num_modes=20), base_ckpt)
    # 同变体可正常加载。
    missing, unexpected = load_checkpoint(
        build_model("GAST-PF-AnchorGeo", num_modes=20), geo_ckpt)
    assert not missing and not unexpected


def test_anchor_geo_parameters_covered_by_adamw_groups():
    """AnchorGeo 参数必须全部进入 AdamW 参数组（优化器闭包纪律）。"""
    from src.model.forecasting_module import ForecastingLightningModule

    module = ForecastingLightningModule(
        model={
            "type": "TrajectoryForecaster",
            "embed_dim": 128, "future_steps": 12, "num_heads": 8,
            "mlp_ratio": 4.0, "qkv_bias": False, "drop_path": 0.2,
            "num_actor_types": 1, "num_modes": 20, "dt": 0.4, "obs_len": 8,
            "bimamba": False, "encoder_type": "gast", "gast_anchor_geo": True,
        },
        lr=1e-3, warmup_epochs=5, epochs=100, weight_decay=1e-4,
    )
    optimizer = module.configure_optimizers()[0][0]
    trainable = {id(p) for p in module.parameters() if p.requires_grad}
    grouped = [p for group in optimizer.param_groups for p in group["params"]]
    assert {id(p) for p in grouped} == trainable
    geo_ids = {id(p) for name, p in module.named_parameters()
               if "anchor_geo" in name and p.requires_grad}
    assert geo_ids
    assert geo_ids.issubset({id(p) for p in grouped})
