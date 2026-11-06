from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from torch import Tensor

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm, rms_norm_fn


class CausalMambaBlock(nn.Module):
    def __init__(self, dim, mixer_cls, drop_path=0.0):
        super().__init__()
        self.residual_in_fp32 = True
        self.mixer = mixer_cls(dim)
        self.norm = RMSNorm(dim, eps=1e-5)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
    ):
        hidden_states, residual = rms_norm_fn(
            hidden_states if residual is None else self.drop_path(hidden_states),
            self.norm.weight,
            self.norm.bias,
            residual=residual,
            prenorm=True,
            residual_in_fp32=self.residual_in_fp32,
            eps=self.norm.eps,
        )
        hidden_states = self.mixer(
            hidden_states, inference_params=inference_params
        )
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )


def create_causal_mamba_block(d_model=128, layer_idx=None, drop_path=0.0):
    mixer_cls = partial(Mamba, layer_idx=layer_idx)
    block = CausalMambaBlock(d_model, mixer_cls, drop_path=drop_path)
    block.layer_idx = layer_idx
    return block
