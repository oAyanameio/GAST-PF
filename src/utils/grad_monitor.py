"""训练期梯度范数监控（Lightning Callback）。

目的：把「编码器到底有没有在学」变成可观测的时间序列，而不是训练结束后
拿单个 checkpoint 拍快照。

设计约束：
1. **不写进 checkpoint 的 hyper_parameters**——用 Callback 而不是改模块构造函数，
   避免改动 resume contract 的字段。
2. **不参与梯度计算**：只读 `p.grad`，全部 detach。
3. **按模块组聚合**：单个模型有 400+ 参数张量，逐张量写 CSV 会让日志不可读。
4. **可关闭**：`every_n_steps=0` 时完全跳过，零开销。

输出列（CSVLogger 会写到 metrics.csv）：
    grad/total
    grad/<group>       例如 grad/enc.spatial、grad/decoder

注意：drop-path / dropout 会让某些层在单步上梯度恰为 0，因此这里做的是
**跨 step 的 epoch 平均**，不要拿单步值下结论。
"""

import pytorch_lightning as pl


# 模块组划分：顺序敏感（先匹配更具体的前缀）
_GROUP_RULES = (
    ("gast_encoder.spatial", "enc.spatial"),
    ("gast_encoder.temporal_decay", "enc.temporal_decay"),
    ("gast_encoder.evidence_tau_logit", "enc.readout"),
    ("gast_encoder.temporal_aggregation.state_aggregation", "enc.aggregation"),
    ("gast_encoder.temporal_aggregation.temporal_mamba", "enc.aggregation"),
    ("gast_encoder.temporal", "enc.temporal"),
    ("gast_encoder", "enc.other"),
    ("goal_decoder", "decoder"),
    ("neighbor_prediction_head", "aux.neighbor_head"),
    ("future_time_embedding", "aux.time_embed"),
)


def group_of(name: str) -> str:
    for prefix, group in _GROUP_RULES:
        if name.startswith(prefix):
            return group
    return "other"


class GradientNormMonitor(pl.Callback):
    """每 N 个训练 step 记录一次分层梯度 L2 范数，按 epoch 平均。"""

    def __init__(self, every_n_steps: int = 50, groups=None):
        super().__init__()
        self.every_n_steps = int(every_n_steps)
        self.groups = groups
        self._seen = 0

    def _enabled(self):
        return self.every_n_steps > 0

    def on_train_epoch_start(self, trainer, pl_module):
        self._seen = 0

    def on_after_backward(self, trainer, pl_module):
        if not self._enabled():
            return
        self._seen += 1
        if self._seen % self.every_n_steps != 0:
            return

        net = getattr(pl_module, "net", None)
        if net is None:
            return

        acc = {}
        for name, param in net.named_parameters():
            if param.grad is None:
                continue
            group = group_of(name)
            if self.groups is not None and group not in self.groups:
                continue
            value = float(param.grad.detach().norm())
            acc[group] = acc.get(group, 0.0) + value * value

        total = sum(acc.values()) ** 0.5
        pl_module.log("grad/total", total, on_step=False, on_epoch=True,
                      prog_bar=False, sync_dist=False)
        for group, sq in acc.items():
            pl_module.log(f"grad/{group}", sq ** 0.5, on_step=False, on_epoch=True,
                          prog_bar=False, sync_dist=False)
