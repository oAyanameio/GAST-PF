"""GAST-PF 编码器回归测试。

覆盖：
1. missing_run_deltas 与 UniTraj get_delta 递归逐位一致；
2. M0 默认路径完全不变（encoder_type 默认 'm0'，state_dict 逐位一致）；
3. GAST-PF 真实 TrajGap Mixed batch 前向/反向（GPU）；
4. padding actor 隔离（固定 batch 内扰动 padding 数据不改有效 actor 输出）；
5. 双向注意力读出屏蔽缺失帧、零门控退化为基线且不扩展时间轴；
6. SceneReasoningTail 零初始化时严格退化为 DAP actor token；
7. 单向因果 Mamba 的末帧读出与 Tail 解耦；
8. evaluator 编码器契约（GAST-PF ckpt 拒载 M0、M0 拒载 GAST-PF、缺键拒绝）；
9. GAST-PF 可训练参数全部进入 AdamW 参数组（优化器闭包纪律）。
"""

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.model.layers.gast_encoder import (  # noqa: E402
    DirectionalAttentionReadout,
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


def test_directional_attention_readout_starts_at_baseline():
    torch.manual_seed(17)
    readout = DirectionalAttentionReadout(embed_dim=4, max_steps=4)
    forward_states = torch.randn(2, 3, 4, 4)
    backward_states = torch.randn(2, 3, 4, 4)
    valid_mask = torch.tensor(
        [[[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
         [[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]]],
        dtype=torch.bool,
    )
    key_padding = torch.tensor([[False, False, True], [False, False, False]])
    age_forward = torch.zeros_like(valid_mask, dtype=torch.float32)
    age_backward = torch.zeros_like(age_forward)

    output = readout(
        forward_states,
        backward_states,
        valid_mask,
        key_padding,
        age_forward,
        age_backward,
    )

    expected = (forward_states + backward_states)[:, :, -1]
    expected = expected * (~key_padding).unsqueeze(-1)
    assert output.shape == (2, 3, 4)
    assert torch.equal(output, expected)


def test_directional_attention_readout_masks_missing_frames():
    readout = DirectionalAttentionReadout(embed_dim=2, max_steps=4)
    with torch.no_grad():
        for parameter in readout.parameters():
            parameter.zero_()
        readout.gate[-1].bias.fill_(1000.0)
        readout.beta.fill_(1000.0)

    forward_states = torch.tensor([[[[1.0, 1.0], [1000.0, 1000.0],
                                    [3.0, 3.0], [1000.0, 1000.0]]]])
    backward_states = torch.tensor([[[[10.0, 10.0], [1000.0, 1000.0],
                                      [30.0, 30.0], [1000.0, 1000.0]]]])
    valid_mask = torch.tensor([[[True, False, True, False]]])
    key_padding = torch.tensor([[False]])
    ages = torch.zeros(1, 1, 4)

    output = readout(
        forward_states,
        backward_states,
        valid_mask,
        key_padding,
        ages,
        ages,
    )

    assert torch.equal(output, torch.full((1, 1, 2), 22.0))


def test_directional_attention_readout_is_finite_for_empty_and_padding_rows():
    readout = DirectionalAttentionReadout(embed_dim=4, max_steps=4)
    states = torch.randn(1, 2, 4, 4)
    valid_mask = torch.zeros(1, 2, 4, dtype=torch.bool)
    key_padding = torch.tensor([[False, True]])
    ages = torch.zeros(1, 2, 4)

    output = readout(states, states, valid_mask, key_padding, ages, ages)

    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[:, 1]) == 0


def test_directional_readout_keeps_original_temporal_length(monkeypatch):
    from src.model.layers import gast_encoder as gast_encoder_module

    class SpatialIdentity(torch.nn.Module):
        def forward(self, x, mask, key_padding):
            return x

    class TemporalCapture(torch.nn.Module):
        def forward(self, x, gamma_forw=None, gamma_back=None,
                    return_directions=False):
            self.sequence = x.detach().clone()
            self.gamma_forw = gamma_forw.detach().clone()
            self.gamma_back = gamma_back.detach().clone()
            batch, steps, actors, channels = x.shape
            forward = x.permute(0, 2, 1, 3)
            backward = forward * 2
            fused = (forward + backward).reshape(batch * actors, steps, channels)
            if return_directions:
                return fused, forward, backward
            return fused

    temporal = TemporalCapture()
    monkeypatch.setattr(
        gast_encoder_module, "TemporalEncoder", lambda *args, **kwargs: temporal
    )
    encoder = GASTEncoder(
        input_dim=4, embed_dim=4, num_heads=1, depth=1,
        p_dropout=0.0, drop_path=0.0, directional_readout=True,
    )
    encoder.spatial = SpatialIdentity()

    hist_feat = torch.arange(2 * 3 * 8 * 4, dtype=torch.float32).reshape(2, 3, 8, 4)
    valid_mask = torch.ones(2, 3, 8, dtype=torch.bool)
    key_padding = torch.tensor([[False, False, True], [False, False, False]])
    output = encoder(hist_feat, valid_mask, key_padding)

    assert temporal.sequence.shape == (2, 8, 3, 4)
    assert torch.equal(temporal.sequence, hist_feat.permute(0, 2, 1, 3))
    expected = temporal.sequence[:, -1] * 3
    expected = expected * (~key_padding).unsqueeze(-1)
    assert output.shape == (2, 3, 4)
    assert torch.equal(output, expected)
    assert torch.count_nonzero(output[0, 2]) == 0


def test_unidirectional_encoder_reads_forward_terminal_state(monkeypatch):
    from src.model.layers import gast_encoder as gast_encoder_module

    class SpatialIdentity(torch.nn.Module):
        def forward(self, x, mask, key_padding):
            return x

    class CausalTemporalCapture(torch.nn.Module):
        def forward(self, x, gamma_forw=None, gamma_back=None,
                    reliability=None, return_directions=False):
            self.gamma_back = gamma_back
            return x.permute(0, 2, 1, 3)

    temporal = CausalTemporalCapture()
    monkeypatch.setattr(
        gast_encoder_module, "TemporalEncoder", lambda *args, **kwargs: temporal
    )
    encoder = GASTEncoder(
        input_dim=4, embed_dim=4, num_heads=1, depth=1,
        p_dropout=0.0, drop_path=0.0,
        temporal_direction="unidirectional",
    )
    encoder.spatial = SpatialIdentity()

    hist_feat = torch.arange(2 * 3 * 8 * 4, dtype=torch.float32).reshape(2, 3, 8, 4)
    valid_mask = torch.ones(2, 3, 8, dtype=torch.bool)
    key_padding = torch.tensor([[False, False, True], [False, False, False]])
    output = encoder(hist_feat, valid_mask, key_padding)

    expected = hist_feat[:, :, -1]
    expected = expected * (~key_padding).unsqueeze(-1)
    assert output.shape == (2, 3, 4)
    assert torch.equal(output, expected)
    assert temporal.gamma_back is None
    assert torch.count_nonzero(output[0, 2]) == 0


def test_directional_readout_is_variant_and_checkpoint_contract(tmp_path):
    from scripts.训练与评估.run_trajimpute_experiments import VARIANTS as TRAIN_VARIANTS
    from scripts.结果分析.evaluate_trajimpute_direct import build_model, load_checkpoint

    assert TRAIN_VARIANTS["GAST-PF-DAP"]["gast_directional_readout"] is True
    model = build_model("GAST-PF-DAP", num_modes=20)
    assert model.gast_directional_readout is True
    assert model.gast_encoder.directional_readout.beta.item() == 0.0

    path = tmp_path / "gast_dap.ckpt"
    torch.save({
        "state_dict": {f"net.{key}": value for key, value in model.state_dict().items()},
        "hyper_parameters": {"model": {
            "num_modes": 20,
            "encoder_type": "gast",
            "gast_directional_readout": True,
        }},
    }, path)
    missing, unexpected = load_checkpoint(build_model("GAST-PF-DAP", 20), path)
    assert not missing
    assert not unexpected

    baseline_path = tmp_path / "gast_without_dap.ckpt"
    baseline = build_model("GAST-PF", num_modes=20)
    torch.save({
        "state_dict": {f"net.{key}": value for key, value in baseline.state_dict().items()},
        "hyper_parameters": {"model": {
            "num_modes": 20,
            "encoder_type": "gast",
            "gast_directional_readout": False,
        }},
    }, baseline_path)
    with pytest.raises(ValueError, match="gast_directional_readout"):
        load_checkpoint(build_model("GAST-PF-DAP", 20), baseline_path)


def test_scene_reasoning_tail_starts_as_exact_dap_identity():
    from src.model.trajectory_forecaster import TrajectoryForecaster

    model = TrajectoryForecaster(
        embed_dim=8,
        num_heads=2,
        future_steps=4,
        num_modes=3,
        obs_len=4,
        encoder_type="gast",
        gast_directional_readout=True,
        gast_scene_reasoning_tail=True,
    )
    model.gast_scene_tail.eval()
    actor_feat = torch.randn(1, 3, 8)
    data = {
        "x_centers": torch.randn(1, 3, 2),
        "x_last_valid_angle": torch.randn(1, 3),
        "x_attr": torch.zeros(1, 3, 3, dtype=torch.long),
    }
    key_padding = torch.tensor([[False, False, True]])

    output = model.gast_scene_tail(actor_feat, data, key_padding)

    assert model.gast_scene_tail.residual_scale.item() == 0.0
    assert torch.equal(output, actor_feat)
    assert not any(
        key.startswith((
            "history_input_projection.", "hist_embed_mamba.", "norm_f.",
        ))
        for key in model.state_dict()
    )


def test_scene_reasoning_tail_does_not_require_dap_readout():
    from src.model.trajectory_forecaster import TrajectoryForecaster

    model = TrajectoryForecaster(
        num_modes=6,
        encoder_type="gast",
        gast_scene_reasoning_tail=True,
    )
    assert model.gast_scene_reasoning_tail is True


def test_unidirectional_tail_variant_does_not_register_backward_mamba():
    from src.model.trajectory_forecaster import TrajectoryForecaster

    model = TrajectoryForecaster(
        embed_dim=8,
        num_heads=2,
        future_steps=4,
        num_modes=3,
        obs_len=4,
        encoder_type="gast",
        gast_temporal_direction="unidirectional",
        gast_scene_reasoning_tail=True,
    )

    assert model.gast_temporal_direction == "unidirectional"
    assert model.gast_scene_reasoning_tail is True
    assert not hasattr(model.gast_encoder.temporal, "ssm_back")
    assert not hasattr(model.gast_encoder.temporal, "norm_f_back")
    assert model.gast_encoder.temporal.temporal_direction == "unidirectional"


def test_unidirectional_tail_variant_is_registered():
    from scripts.训练与评估.run_trajimpute_experiments import VARIANTS as TRAIN_VARIANTS
    from scripts.结果分析.evaluate_trajimpute_direct import VARIANTS as EVAL_VARIANTS
    from scripts.结果分析.evaluate_trajimpute_direct import build_model

    assert TRAIN_VARIANTS["GAST-PF-UniTail"]["gast_temporal_direction"] == "unidirectional"
    assert EVAL_VARIANTS["GAST-PF-UniTail"]["gast_temporal_direction"] == "unidirectional"
    model = build_model("GAST-PF-UniTail", num_modes=6)
    assert model.gast_temporal_direction == "unidirectional"
    assert model.gast_scene_reasoning_tail is True


def test_evaluator_rejects_bidirectional_checkpoint_for_unidirectional_variant(tmp_path):
    from scripts.结果分析.evaluate_trajimpute_direct import build_model, load_checkpoint

    model = build_model("GAST-PF", num_modes=20)
    path = tmp_path / "bidirectional.ckpt"
    torch.save({
        "state_dict": {f"net.{key}": value for key, value in model.state_dict().items()},
        "hyper_parameters": {"model": {
            "num_modes": 20,
            "encoder_type": "gast",
            "gast_temporal_direction": "bidirectional",
        }},
    }, path)

    with pytest.raises(ValueError, match="gast_temporal_direction"):
        load_checkpoint(build_model("GAST-PF-Uni", num_modes=20), path)


def test_scene_reasoning_tail_variant_and_checkpoint_contract(tmp_path):
    from scripts.训练与评估.run_trajimpute_experiments import VARIANTS as TRAIN_VARIANTS
    from scripts.结果分析.evaluate_trajimpute_direct import build_model, load_checkpoint

    assert TRAIN_VARIANTS["GAST-PF-Tail"]["gast_scene_reasoning_tail"] is True
    model = build_model("GAST-PF-Tail", num_modes=20)
    assert model.gast_directional_readout is True
    assert model.gast_scene_reasoning_tail is True
    assert model.gast_scene_tail.residual_scale.item() == 0.0

    path = tmp_path / "gast_dap_scene_tail.ckpt"
    torch.save({
        "state_dict": {f"net.{key}": value for key, value in model.state_dict().items()},
        "hyper_parameters": {"model": {
            "num_modes": 20,
            "encoder_type": "gast",
            "gast_directional_readout": True,
            "gast_scene_reasoning_tail": True,
        }},
    }, path)
    missing, unexpected = load_checkpoint(
        build_model("GAST-PF-Tail", 20), path
    )
    assert not missing
    assert not unexpected
    with pytest.raises(ValueError, match="gast_scene_reasoning_tail"):
        load_checkpoint(build_model("GAST-PF-DAP", 20), path)


def test_ecsr_exposes_shared_evidence_state_for_scene_tail(monkeypatch):
    class SpatialIdentity(torch.nn.Module):
        def forward(self, x, mask, key_padding, query_reliability=None,
                    **kwargs):
            self.query_reliability = query_reliability
            return x

    class TemporalIdentity(torch.nn.Module):
        def forward(self, x, gamma_forw=None, gamma_back=None,
                    reliability=None, return_directions=False):
            return x.permute(0, 2, 1, 3)

    encoder = GASTEncoder(
        input_dim=4, embed_dim=4, num_heads=1, depth=1,
        p_dropout=0.0, drop_path=0.0, ecsr=True,
        hierarchical_evidence=True,
        temporal_direction="unidirectional",
    )
    spatial = SpatialIdentity()
    encoder.spatial = spatial
    encoder.temporal = TemporalIdentity()
    encoder.readout_score = torch.nn.Linear(4, 1)
    torch.nn.init.zeros_(encoder.readout_score.weight)
    torch.nn.init.zeros_(encoder.readout_score.bias)

    hist_feat = torch.zeros(1, 2, 4, 4)
    valid_mask = torch.tensor([
        [[True, True, False, False], [True, False, False, True]],
    ])
    key_padding = torch.zeros(1, 2, dtype=torch.bool)
    positions = torch.zeros(1, 2, 4, 2)
    _, evidence = encoder(
        hist_feat, valid_mask, key_padding, positions=positions,
        return_evidence=True,
    )

    assert set(evidence) == {"source_eligibility", "reliability", "age"}
    assert evidence["reliability"].shape == (1, 2, 4)
    assert evidence["source_eligibility"].shape == (1, 2, 4)
    assert evidence["reliability"][0, 0, 0].item() == 1.0
    assert evidence["source_eligibility"][0, 0, 2].item() == 0.0
    assert torch.equal(
        spatial.query_reliability,
        evidence["reliability"].permute(0, 2, 1),
    )


def test_evidence_scene_tail_builds_actor_key_bias():
    from src.model.trajectory_forecaster import SceneReasoningTail

    tail = SceneReasoningTail(
        embed_dim=4, num_heads=2, mlp_ratio=2.0, qkv_bias=False,
        num_actor_types=1, drop_path=0.0, max_steps=4,
        evidence_conditioned=True,
    )
    reliability = torch.tensor([[0.9, 0.1, 0.0]])
    key_padding = torch.tensor([[False, False, True]])
    bias = tail._build_evidence_attention_mask(reliability, key_padding)

    assert bias.shape == (2, 3, 3)
    assert torch.isfinite(bias[:, :, :2]).all()
    assert torch.isneginf(bias[:, :, 2]).all()
    assert bias[0, 0, 0] > bias[0, 0, 1]

    data = {
        "x_centers": torch.zeros(1, 3, 2),
        "x_last_valid_angle": torch.zeros(1, 3),
        "x_attr": torch.zeros(1, 3, 3),
    }
    actor_feat = torch.randn(1, 3, 4)
    output = tail(
        actor_feat,
        data,
        key_padding,
        evidence_state={
            "reliability": reliability.unsqueeze(-1).expand(-1, -1, 4),
            "age": torch.tensor([[
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 2.0, 2.0],
                [0.0, 1.0, 2.0, 4.0],
            ]]),
        },
    )
    assert output.shape == actor_feat.shape
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[:, 2]) == 0


def test_oehsr_variant_wires_named_evidence_modules():
    from scripts.训练与评估.run_trajimpute_experiments import VARIANTS as TRAIN_VARIANTS
    from scripts.结果分析.evaluate_trajimpute_direct import VARIANTS as EVAL_VARIANTS, build_model
    from src.model.layers.gast_encoder import (
        EvidenceModulatedCausalMamba,
        ObservationEvidenceGSM,
    )
    from src.model.trajectory_forecaster import EvidenceGatedSceneTail

    for variants in (TRAIN_VARIANTS, EVAL_VARIANTS):
        config = variants["GAST-PF-OEHSR"]
        assert config["gast_oer_spatial"] is True
        assert config["gast_emc_temporal"] is True
        assert config["gast_ewhr_readout"] is True
        assert config["gast_scene_reasoning_tail"] is True
        assert config["gast_egst_tail"] is True
        assert config["gast_oehsr"] is True

    model = build_model("GAST-PF-OEHSR", num_modes=6)
    assert model.gast_oer_spatial is True
    assert model.gast_emc_temporal is True
    assert model.gast_ewhr_readout is True
    assert model.gast_scene_reasoning_tail is True
    assert model.gast_egst_tail is True
    assert model.gast_oehsr is True
    assert isinstance(model.gast_encoder.spatial, ObservationEvidenceGSM)
    assert isinstance(model.gast_encoder.temporal, EvidenceModulatedCausalMamba)
    assert isinstance(model.gast_scene_tail, EvidenceGatedSceneTail)

    from src.model.trajectory_forecaster import TrajectoryForecaster
    with pytest.raises(ValueError, match="unidirectional"):
        TrajectoryForecaster(
            encoder_type="gast", gast_oer_spatial=True,
            gast_scene_reasoning_tail=True, gast_oehsr=True,
        )


def test_legacy_echsr_switches_still_construct_oehsr():
    from src.model.trajectory_forecaster import TrajectoryForecaster

    model = TrajectoryForecaster(
        encoder_type="gast",
        gast_temporal_direction="unidirectional",
        gast_ecsr=True,
        gast_m0_scene_tail=True,
        gast_hierarchical_evidence=True,
    )

    assert model.gast_oehsr is True
    assert model.gast_scene_reasoning_tail is True
    assert model.gast_scene_tail.evidence_conditioned is True


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


def test_directional_readout_parameters_are_covered_by_adamw_groups():
    from src.model.forecasting_module import ForecastingLightningModule

    module = ForecastingLightningModule(
        model={
            "type": "TrajectoryForecaster",
            "embed_dim": 128, "future_steps": 12, "num_heads": 8,
            "mlp_ratio": 4.0, "qkv_bias": False, "drop_path": 0.2,
            "num_actor_types": 1, "num_modes": 20, "dt": 0.4, "obs_len": 8,
            "bimamba": False, "encoder_type": "gast",
            "gast_directional_readout": True,
        },
        lr=1e-3, warmup_epochs=5, epochs=100, weight_decay=1e-4,
    )
    optimizer = module.configure_optimizers()[0][0]
    grouped = {id(parameter) for group in optimizer.param_groups
               for parameter in group["params"]}
    beta = module.net.gast_encoder.directional_readout.beta
    assert id(beta) in grouped


def test_scene_reasoning_tail_parameters_are_covered_by_adamw_groups():
    from src.model.forecasting_module import ForecastingLightningModule

    module = ForecastingLightningModule(
        model={
            "type": "TrajectoryForecaster",
            "embed_dim": 128, "future_steps": 12, "num_heads": 8,
            "mlp_ratio": 4.0, "qkv_bias": False, "drop_path": 0.2,
            "num_actor_types": 1, "num_modes": 20, "dt": 0.4, "obs_len": 8,
            "bimamba": False, "encoder_type": "gast",
            "gast_directional_readout": True,
            "gast_scene_reasoning_tail": True,
        },
        lr=1e-3, warmup_epochs=5, epochs=100, weight_decay=1e-4,
    )
    optimizer = module.configure_optimizers()[0][0]
    grouped = {id(parameter) for group in optimizer.param_groups
               for parameter in group["params"]}
    scale = module.net.gast_scene_tail.residual_scale
    assert id(scale) in grouped
