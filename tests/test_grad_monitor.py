"""GradientNormMonitor 行为测试。

要点：
1. `every_n_steps=0` 必须完全跳过（零开销，不产生日志）；
2. 采样间隔生效：每 N 步只记录一次；
3. 分组规则把参数名映射到稳定的组名；
4. 记录的是范数（非平方），且 total 与分组自洽。
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils.grad_monitor import GradientNormMonitor, group_of  # noqa: E402


class _FakeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gast_encoder = torch.nn.Module()
        self.gast_encoder.spatial = torch.nn.Linear(4, 4)
        self.gast_encoder.temporal = torch.nn.Linear(4, 4)
        self.future_decoder = torch.nn.Linear(4, 4)
        self.other_head = torch.nn.Linear(4, 4)


class _FakeModule:
    def __init__(self):
        self.net = _FakeNet()
        self.logged = {}

    def log(self, key, value, **kwargs):
        self.logged.setdefault(key, []).append(float(value))


def _seed_grads(module, scale=1.0):
    for p in module.net.parameters():
        p.grad = torch.full_like(p, scale)


def test_group_of_maps_modules():
    assert group_of("gast_encoder.spatial.patch_embed.0.weight") == "enc.spatial"
    assert group_of("gast_encoder.temporal.ssm_forw.0.mixer.A_log") == "enc.temporal"
    assert group_of("gast_encoder.temporal_decay.mlp.weight") == "enc.temporal_decay"
    assert group_of("gast_encoder.directional_readout.beta") == "enc.readout"
    assert group_of("gast_encoder.middle_readout_token") == "enc.readout"
    assert group_of("future_decoder.x.weight") == "decoder"
    assert group_of("neighbor_prediction_head.0.weight") == "aux.neighbor_head"
    assert group_of("future_time_embedding.0.weight") == "aux.time_embed"
    assert group_of("something_else") == "other"


def test_disabled_when_interval_zero():
    cb = GradientNormMonitor(every_n_steps=0)
    mod = _FakeModule()
    _seed_grads(mod)
    for _ in range(10):
        cb.on_after_backward(None, mod)
    assert mod.logged == {}, "every_n_steps=0 不应产生任何日志"


def test_samples_every_n_steps():
    cb = GradientNormMonitor(every_n_steps=3)
    mod = _FakeModule()
    _seed_grads(mod)
    for _ in range(6):
        cb.on_after_backward(None, mod)
    # 第 3、6 步各记录一次
    assert len(mod.logged["grad/total"]) == 2


def test_logs_norms_not_squares_and_total_is_consistent():
    cb = GradientNormMonitor(every_n_steps=1)
    mod = _FakeModule()
    _seed_grads(mod, scale=2.0)
    cb.on_after_backward(None, mod)

    groups = {k[len("grad/"):]: v[0] for k, v in mod.logged.items()
              if k.startswith("grad/") and k != "grad/total"}
    expect_total = sum(v * v for v in groups.values()) ** 0.5
    assert abs(mod.logged["grad/total"][0] - expect_total) < 1e-5

    # 单参数层：Linear(4,4) 的 weight+bias 梯度全为 2 → 范数 = 2*sqrt(20)
    assert abs(groups["decoder"] - 2.0 * (20 ** 0.5)) < 1e-5


def test_skips_params_without_grad():
    cb = GradientNormMonitor(every_n_steps=1)
    mod = _FakeModule()
    _seed_grads(mod)
    mod.net.other_head.weight.grad = None
    mod.net.other_head.bias.grad = None
    cb.on_after_backward(None, mod)
    assert "grad/other" not in mod.logged


def test_epoch_start_resets_counter():
    cb = GradientNormMonitor(every_n_steps=2)
    mod = _FakeModule()
    _seed_grads(mod)
    cb.on_after_backward(None, mod)          # 第 1 步
    cb.on_train_epoch_start(None, mod)       # 重置
    cb.on_after_backward(None, mod)          # 又是第 1 步 → 不记录
    assert "grad/total" not in mod.logged
    cb.on_after_backward(None, mod)          # 第 2 步 → 记录
    assert len(mod.logged["grad/total"]) == 1
