"""LaplaceNLLLoss 向量化等价门禁 / 回归测试。

`compute`（向量化）与 `compute_reference`（原逐样本循环）必须等价。
真实训练路径是 bf16 权重 + autocast 前向，loss 在 fp32 累加，因此验收口径是：

  * 对 traj / scale / probs 的梯度：**逐位相同**（bitwise），任何 dtype
  * 前向 loss 值：fp32 下差异 <= 1e-5（累加顺序导致的 1-2 ulp）
  * bf16 输入下的值差异放宽到 bf16 ulp（真实路径不会在 bf16 里做 batch 求和）

另有两条边界说明（不是回归，见 CLAUDE/提交说明）：
  * clamp(min=1e-6) 在原实现里是 no_grad 原地修改，会给出错误梯度；
    新实现用非原地 clamp，梯度正确。生产 scale 最小约 2.3e-4，不可达。
  * 因此 clamp 分支不要求两版一致，而是要求新实现给出正确的（0）梯度。

用法:
    PYTHONPATH=. python scripts/审计与校验/check_laplace_loss_equivalence.py
退出码 0 = 全部通过。
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.utils.LaplaceNLLLoss import LaplaceNLLLoss  # noqa: E402

failures = []
checks = 0


def report(name, ok, detail=""):
    global checks
    checks += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def grads(fn, tensors):
    g = torch.autograd.grad(fn(), tensors, retain_graph=True, allow_unused=True)
    return [x.detach() if x is not None else None for x in g]


def check_equivalence(label, traj, scale, probs, gt, value_atol=1e-5, value_rtol=1e-5,
                      require_bitwise_grad=True):
    """在同一份输入上分别跑两版实现，比对值与梯度。"""
    mod = LaplaceNLLLoss()

    def build():
        return [t.detach().clone().requires_grad_(True) for t in (traj, scale, probs)]

    t1, s1, p1 = build()
    l_ref = mod.compute_reference({"traj": t1, "scale": s1, "probs": p1}, gt)
    g_ref = grads(lambda: l_ref, [t1, s1, p1])

    t2, s2, p2 = build()
    l_new = mod.compute({"traj": t2, "scale": s2, "probs": p2}, gt)
    g_new = grads(lambda: l_new, [t2, s2, p2])

    a, b = l_ref.detach(), l_new.detach()
    diff = (a.float() - b.float()).abs().max().item()
    denom = max(a.float().abs().max().item(), 1e-12)
    ok = torch.allclose(a.float(), b.float(), atol=value_atol, rtol=value_rtol)
    report(f"{label} loss", ok,
           f"abs_diff={diff:.3e} rel={diff/denom:.3e} bitwise={torch.equal(a, b)}")

    for name, ga, gb in zip(("d_traj", "d_scale", "d_probs"), g_ref, g_new):
        if ga is None or gb is None:
            report(f"{label} {name}", ga is None and gb is None, "unused")
            continue
        gd = (ga.float() - gb.float()).abs().max().item()
        bit = torch.equal(ga, gb)
        ok_g = bit if require_bitwise_grad else torch.allclose(
            ga.float(), gb.float(), atol=value_atol, rtol=value_rtol)
        report(f"{label} {name}", ok_g, f"bitwise={bit} max_abs_diff={gd:.3e}")


def main():
    torch.manual_seed(2024)
    B, M, T = 64, 20, 12

    print("=" * 74)
    print("1. fp32 随机张量（严格口径：梯度逐位相同）")
    print("=" * 74)
    traj = torch.randn(B, M, T, 2) * 2.0
    scale = torch.rand(B, M, T, 2) * 3.0 + 1.0
    probs = torch.randn(B, M)
    gt = torch.randn(B, T, 2) * 2.0
    check_equivalence("rand", traj, scale, probs, gt)

    print()
    print("=" * 74)
    print("2. 边界情况")
    print("=" * 74)
    check_equivalence("B=1", traj[:1], scale[:1], probs[:1], gt[:1])
    check_equivalence("zero_traj", torch.zeros(B, M, T, 2), scale, probs,
                      torch.zeros(B, T, 2))
    check_equivalence("B=256",
                      torch.randn(256, M, T, 2), torch.rand(256, M, T, 2) * 3 + 1,
                      torch.randn(256, M), torch.randn(256, T, 2))
    # 极大 scale（数值压力）
    check_equivalence("large_scale", traj, torch.full((B, M, T, 2), 50.0), probs, gt)

    print()
    print("=" * 74)
    print("3. bf16 输入（梯度仍须逐位相同；值放宽到 bf16 ulp）")
    print("=" * 74)
    check_equivalence("bf16", traj.bfloat16(), scale.bfloat16(), probs.bfloat16(),
                      gt.bfloat16(), value_atol=1e-2, value_rtol=1e-2)

    print()
    print("=" * 74)
    print("4. clamp 分支：新实现必须给出正确的（饱和→0）梯度")
    print("=" * 74)
    mod = LaplaceNLLLoss()
    tiny = torch.full((B, M, T, 2), 1e-9)
    t = traj.clone().requires_grad_(True)
    s = tiny.clone().requires_grad_(True)
    p = probs.clone().requires_grad_(True)
    l = mod.compute({"traj": t, "scale": s, "probs": p}, gt)
    _, gs, _ = torch.autograd.grad(l, [t, s, p], retain_graph=True)
    # scale=1e-9 < 1e-6，clamp 饱和，对原始 scale 的梯度应为 0
    report("clamp_saturated_d_scale_is_zero",
           torch.equal(gs, torch.zeros_like(gs)),
           f"max_abs={gs.abs().max().item():.3e}")
    # 未饱和（scale 全 > 1e-6）时梯度必须与原实现一致
    check_equivalence("clamp_not_triggered", traj, scale.clamp(min=1e-3), probs, gt)

    print()
    print("=" * 74)
    print("5. 真实训练路径：bf16 权重 + autocast 前向（同一份输出上比较）")
    print("=" * 74)
    try:
        from src.datamodule.trajimpute_dataset import TrajGapDataset, trajimpute_collate_fn
        from src.model.forecasting_module import ForecastingLightningModule

        cfg = dict(type="TrajectoryForecaster", embed_dim=128, future_steps=12, num_heads=8,
                   mlp_ratio=4.0, qkv_bias=False, drop_path=0.2, num_actor_types=1,
                   num_modes=20, dt=0.4, obs_len=8, bimamba=False,
                   encoder_type="gast", gast_depth=4, gast_gsm_operator="max")
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        autocast_dev = "cuda" if torch.cuda.is_available() else "cpu"
        lm = ForecastingLightningModule(model=dict(cfg), lr=1e-3, weight_decay=1e-4,
                                        epochs=100, warmup_epochs=5).to(dev).eval()
        lm = lm.bfloat16()
        ds = TrajGapDataset("/home/lbh/TrajImpute/dataset/TrajImpute", "ETH-M", "val")
        batch = trajimpute_collate_fn([ds[i] for i in range(128)])
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            with torch.autocast(device_type=autocast_dev, dtype=torch.bfloat16):
                out = lm(batch)

        # 记录生产 scale 下界，证明 clamp 不可达
        smin = min(out["scal"].float().min().item(), out["scal_new"].float().min().item())
        report("production_scale_above_clamp", smin > 1e-6, f"min_scale={smin:.3e}")

        y = batch["target"][:, 0]
        for head, keys in (("mode_head", ("y_hat", "scal", "pi")),
                           ("final_head", ("new_y_hat", "scal_new", "new_pi"))):
            check_equivalence(head, out[keys[0]], out[keys[1]], out[keys[2]], y)
    except Exception as exc:  # noqa: BLE001
        report("real_path", False, f"{type(exc).__name__}: {exc}")

    print()
    print("=" * 74)
    if failures:
        print(f"FAILED: {len(failures)}/{checks}")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"ALL PASS: {checks}/{checks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
