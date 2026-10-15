"""把 pre-GAST-PF 改名前的 UT-ENC last.ckpt 迁移到 gast_encoder 命名空间。

背景：DeMo→GAST-PF 重构把属性 unitraj_encoder 改名为 gast_encoder。模型
state_dict 键有 _load_from_state_dict 迁移，但 checkpoint 里的 optimizer
param_groups 按 sorted(参数名) 展平索引；前缀改名改变字母序（gast < neighbor
< unitraj），Lightning resume 时 optimizer 状态张量与参数错位。

已验证（HOTEL-M last.ckpt）：旧 optimizer 展平顺序 ==
sorted(旧 decay 名) + sorted(旧 no_decay 名)。

本脚本（原地替换，先写 .tmp 再 rename）：
1. state_dict 键 unitraj_encoder.* -> gast_encoder.*；
2. 分别构造旧名/新名的排序展平序列，按【名字对应】（而非位置）搬运
   optimizer state，再按新顺序重编号 param_groups；
3. 校验每个新参数的 optimizer 状态形状 == 模型参数形状，不匹配即中止。

用法: python scripts/训练与评估/migrate_utenc_ckpt.py <last.ckpt> [...]
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OLD, NEW = "unitraj_encoder.", "gast_encoder."


def classify(named_modules_iter):
    """复刻 forecasting_module.configure_optimizers 的 decay/no_decay 分类。"""
    import torch.nn as nn
    whitelist = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
                 nn.MultiheadAttention, nn.LSTM, nn.GRU)
    decay, no_decay = set(), set()
    for module_name, module in named_modules_iter:
        for param_name, _ in module.named_parameters(recurse=False):
            full = f"{module_name}.{param_name}" if module_name else param_name
            if "bias" in param_name:
                no_decay.add(full)
            elif "weight" in param_name:
                if isinstance(module, whitelist):
                    decay.add(full)
                else:
                    no_decay.add(full)
            else:
                no_decay.add(full)
    return decay, no_decay


def migrate(path: Path) -> None:
    ckpt = torch.load(path, map_location="cpu")
    net = "net."

    # 1) 模型键迁移
    sd = ckpt["state_dict"]
    migrated = 0
    for k in list(sd):
        idx = k.find(OLD)
        if k.startswith(net) and idx != -1:
            nk = k[:idx] + NEW + k[idx + len(OLD):]
            sd[nk] = sd.pop(k)
            migrated += 1
    print(f"[{path.name}] model keys migrated: {migrated}")

    # 2) optimizer 索引重排（按名字配对）
    from src.model.trajectory_forecaster import TrajectoryForecaster
    model = TrajectoryForecaster(encoder_type="utenc", num_modes=20,
                                 bimamba=False, gast_depth=4)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    decay_new, no_decay_new = classify(model.named_modules())
    decay_new &= trainable
    no_decay_new &= trainable

    flat_new = sorted(decay_new) + sorted(no_decay_new)
    # 旧名序列：与新序列同集合、同分组，仅名字前缀不同
    flat_old = [n.replace(NEW, OLD) for n in flat_new]
    # 旧 optimizer 里这个名字序列的真实键序（param_groups 展平）
    old_seq_keys = [n.replace(NEW, OLD) for n in flat_new]
    # 但旧 checkpoint 的实际展平顺序 = sorted(旧名) 分组排序，需按名字查位置
    sorted_old = (sorted(n.replace(NEW, OLD) for n in decay_new)
                  + sorted(n.replace(NEW, OLD) for n in no_decay_new))
    name_to_sorted_pos = {n: i for i, n in enumerate(sorted_old)}

    opt_states = ckpt.get("optimizer_states")
    if opt_states:
        opt = opt_states[0]
        old_flat_idx = []
        for g in opt["param_groups"]:
            old_flat_idx.extend(g["params"])
        if len(old_flat_idx) != len(sorted_old):
            raise RuntimeError(
                f"{path}: optimizer params {len(old_flat_idx)} != "
                f"expected {len(sorted_old)}")

        param_shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}
        old_state = opt["state"]
        new_state = {}
        mismatches = 0
        for new_idx, new_name in enumerate(flat_new):
            old_name = new_name.replace(NEW, OLD)
            key = old_flat_idx[name_to_sorted_pos[old_name]]
            entry = old_state.get(key)
            if entry is None:
                raise RuntimeError(f"{path}: no state for {old_name} (key {key})")
            entry = dict(entry)
            for stat in ("exp_avg", "exp_avg_sq"):
                if stat in entry and tuple(entry[stat].shape) != param_shapes[new_name]:
                    mismatches += 1
                    if mismatches <= 3:
                        print(f"  MISMATCH {new_name}: "
                              f"{tuple(entry[stat].shape)} vs "
                              f"{param_shapes[new_name]}")
            new_state[new_idx] = entry
        if mismatches:
            raise RuntimeError(f"{path}: {mismatches} optimizer tensors "
                               f"mismatch after remap — abort")
        opt["state"] = new_state
        for gi, names in enumerate((sorted(decay_new), sorted(no_decay_new))):
            opt["param_groups"][gi]["params"] = [
                flat_new.index(n) for n in names]
        print(f"[{path.name}] optimizer state remapped: "
              f"{len(new_state)} tensors, 0 mismatches")

    tmp = path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)
    print(f"[{path.name}] saved in place")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        migrate(Path(arg))
