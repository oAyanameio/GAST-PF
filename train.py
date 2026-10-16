import json
import os
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

import torch

# TF32（RTX 5880 Ada 张量核心，PyTorch warning 提示）。只影响新启动的进程；
# bf16 AMP 训练下数值影响局限于 autocast 外的 fp32 matmul。
torch.set_float32_matmul_precision("high")


def optimizer_signature(trainer, module):
    """Named optimizer signature: group order, decay policy and tensor shapes."""
    name_by_id = {id(parameter): name for name, parameter in module.named_parameters()}
    signature = []
    for group in trainer.optimizers[0].param_groups:
        signature.append({
            "weight_decay": float(group["weight_decay"]),
            "parameters": [
                {"name": name_by_id[id(parameter)], "shape": list(parameter.shape)}
                for parameter in group["params"]
            ],
        })
    return signature


class RunContractCallback(pl.Callback):
    """Stamp the run contract and optimizer signature into every checkpoint.

    The runner refuses to resume any checkpoint whose stamped contract differs
    from the current experiment, so a historical run can never be resumed as a
    different model, protocol, optimizer policy or checkpoint monitor.
    """

    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path)

    def _manifest(self):
        return json.loads(self.manifest_path.read_text())

    def on_fit_start(self, trainer, pl_module):
        # Lightning configures optimizers during setup; when they are already
        # available, fail fast on any drift between the live model and the
        # manifest this run was launched with.
        if not getattr(trainer, "optimizers", None):
            return
        self._verify_signature(trainer, pl_module)

    def _verify_signature(self, trainer, pl_module):
        expected = self._manifest().get("optimizer_signature")
        actual = optimizer_signature(trainer, pl_module)
        if actual != expected:
            raise RuntimeError(
                "run contract mismatch: live optimizer parameter names/order/shapes "
                "differ from the manifest this run was launched with"
            )

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        self._verify_signature(trainer, pl_module)
        manifest = self._manifest()
        checkpoint["trajgap_run_contract"] = manifest["resume_contract"]
        checkpoint["trajgap_optimizer_signature"] = manifest["optimizer_signature"]


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf):
    pl.seed_everything(conf.seed, workers=True)
    output_dir = HydraConfig.get().runtime.output_dir

    logger = CSVLogger(save_dir=output_dir, name="logs")

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="{epoch}",
            monitor=f"{conf.monitor}",
            mode="min",
            save_top_k=conf.save_top_k,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    run_manifest_path = conf.get("run_manifest_path", None) if hasattr(conf, "get") else None
    if run_manifest_path:
        callbacks.append(RunContractCallback(run_manifest_path))

    trainer = pl.Trainer(
        logger=logger,
        gradient_clip_val=conf.gradient_clip_val,
        gradient_clip_algorithm=conf.gradient_clip_algorithm,
        max_epochs=conf.epochs,
        accelerator="gpu",
        devices=conf.gpus,
        strategy="auto",
        callbacks=callbacks,
        limit_train_batches=conf.limit_train_batches,
        limit_val_batches=conf.limit_val_batches,
        sync_batchnorm=conf.sync_bn,
        precision=conf.precision,
    )

    model = instantiate(conf.model.target)
    os.system('cp -a %s %s' % ('conf', output_dir))
    os.system('cp -a %s %s' % ('src', output_dir))
    with open(f'{output_dir}/model.txt', 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        print(model)
        sys.stdout = original_stdout
    datamodule = instantiate(conf.datamodule.target)
    trainer.fit(model, datamodule, ckpt_path=conf.checkpoint)
    # 验证必须显式加载 runner 选中的 best checkpoint；直接验证内存权重会给出
    # 与最终选点不一致的“最后 epoch”结果，容易被误读为最佳模型指标。
    trainer.validate(model, datamodule.val_dataloader(), ckpt_path="best")


if __name__ == "__main__":
    main()
