import datetime
from pathlib import Path
import time
import pickle
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.metrics import MR, minADE, minFDE, brier_minFDE
from src.utils.optim import WarmupCosLR
from src.utils.LaplaceNLLLoss import LaplaceNLLLoss
from .trajectory_forecaster import TrajectoryForecaster
from .trajectory_loss import final_consistent_loss


def _log_detach(x):
    """日志标量统一 detach：tensor 去梯度，标量原样返回（避免对 0 调 .detach）。"""
    return x.detach() if torch.is_tensor(x) else x


class ForecastingLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: dict,
        pretrained_weights: str = None,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
    ) -> None:
        super(Trainer, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        from src.utils.submission_ethucy import SubmissionEthUcy
        self.submission_handler = SubmissionEthUcy()

        model_type = model.pop('type')

        self.net = self.get_model(model_type)(**model)

        if pretrained_weights is not None:
            self.net.load_from_checkpoint(pretrained_weights)
            print('Pretrained weights have been loaded.')

        num_modes = model.get('num_modes', 6)

        metrics = MetricCollection(
            {
                "minADE1": minADE(k=1),
                f"minADE{num_modes}": minADE(k=num_modes),
                "minFDE1": minFDE(k=1),
                f"minFDE{num_modes}": minFDE(k=num_modes),
                "MR": MR(),
                f"b-minFDE{num_modes}": brier_minFDE(k=num_modes),
            }
        )
        self.laplace_loss = LaplaceNLLLoss()
        self.val_metrics = metrics.clone(prefix="val_")
        self.val_metrics_new = metrics.clone(prefix="val_new_")

    def get_model(self, model_type):
        model_dict = {
            'TrajectoryForecaster': TrajectoryForecaster,
        }
        assert model_type in model_dict, f"Unknown model type: {model_type}"
        return model_dict[model_type]

    def forward(self, data):
        return self.net(data)

    def predict(self, data):
        predictions = []
        probs = []
        for i in range(len(data)):
            cur_data = data[i]
            out = self(cur_data)
            prediction, prob = self.submission_handler.format_data(
                cur_data, out["y_hat"], out["pi"], inference=True)
            predictions.append(prediction)
            probs.append(prob)

        return predictions, probs

    def _cal_loss_final_consistent(self, out, data, tag=''):
        return final_consistent_loss(
            out=out,
            data=data,
            laplace_loss=self.laplace_loss,
            endpoint_loss_weight=self.net.goal_endpoint_loss_weight,
            mode_diversity_loss_weight=self.net.mode_diversity_loss_weight,
            mode_diversity_margin=self.net.mode_diversity_margin,
            classification_target_mode=self.net.classification_target_mode,
            tag=tag,
        )

    def cal_loss(self, out, data, tag=''):
        return self._cal_loss_final_consistent(out, data, tag=tag)

    def training_step(self, data, batch_idx):
        out = self(data)
        loss, loss_dict = self.cal_loss(out, data)

        for k, v in loss_dict.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,  # 单卡无 all-reduce；多卡再改回 True
            )

        return loss

    def validation_step(self, data, batch_idx):
        out = self(data)
        _, loss_dict = self.cal_loss(out, data)
        metrics = self.val_metrics(out, data['target'][:, 0])
        if out['new_y_hat'] is not None:
            out['y_hat'] = out['new_y_hat']
            out['pi'] = out['new_pi']
            metrics_new = self.val_metrics_new(out, data['target'][:, 0])

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        if out['new_y_hat'] is not None:
            self.log_dict(
                metrics_new,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )

    def on_test_start(self) -> None:
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)
        from src.utils.submission_ethucy import SubmissionEthUcy
        self.submission_handler = SubmissionEthUcy(save_dir=str(save_dir))
        self.test_metrics = self.val_metrics.clone(prefix="test_")

    def test_step(self, data, batch_idx) -> None:
        out = self(data)
        if out['new_y_hat'] is not None:
            out['y_hat'] = out['new_y_hat']
            out['pi'] = out['new_pi']
        self.submission_handler.format_data(data, out["y_hat"], out["pi"])
        metrics = self.test_metrics(out, data['target'][:, 0])
        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()
        print("TEST METRICS:", self.test_metrics.compute())

    def configure_optimizers(self):
        # M0 baseline policy: enumerate module-local parameters and retain
        # custom mixer/RMSNorm weights in the no-decay group by default.
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    else:
                        # Custom mixers/RMSNorms are not all covered by the
                        # module lists; trainable weights must never be omitted.
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        if inter_params:
            raise RuntimeError(
                f"AdamW parameter groups overlap: {sorted(inter_params)}"
            )
        trainable_params = {
            name for name, param in self.named_parameters() if param.requires_grad
        }
        if union_params != trainable_params:
            raise RuntimeError(
                "AdamW parameter groups must cover every trainable parameter exactly once; "
                f"missing={sorted(trainable_params - union_params)}, "
                f"unexpected={sorted(union_params - trainable_params)}"
            )

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-5,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        return [optimizer], [scheduler]


# Legacy class name kept for old Hydra targets and scripts.
Trainer = ForecastingLightningModule
