"""GAST 观测状态编码器的实现。

公开名称与实现类对应关系：
- 观测状态感知时空编码器：ObservationStateEncoder
- 逐帧交互建模：FramewiseInteractionModel
- 状态强度加权的时序聚合：StateStrengthTemporalAggregation

以上类名是代码实现标识，公开说明使用左侧名称。
"""

import torch
import torch.nn as nn
from mamba_ssm.ops.triton.layernorm import RMSNorm, rms_norm_fn

from .mamba.mamba_block import create_mamba_block


def build_observation_state(
    valid_mask: torch.Tensor,
    tau: torch.Tensor | float,
    key_padding: torch.Tensor | None = None,
):
    """Construct observation eligibility, age, and state-strength tensors."""
    observed = valid_mask.bool()
    if key_padding is not None:
        observed = observed & ~key_padding.bool().unsqueeze(-1)
    steps = torch.arange(
        observed.shape[-1], device=observed.device, dtype=torch.long
    ).view(*([1] * (observed.ndim - 1)), -1)
    last_observed = torch.where(observed, steps, -1).cummax(dim=-1).values
    age = torch.where(last_observed >= 0, steps - last_observed, steps + 1)
    tau_tensor = torch.as_tensor(tau, dtype=torch.float32, device=observed.device)
    state_strength = torch.where(
        observed,
        torch.ones_like(age, dtype=torch.float32),
        torch.exp(-tau_tensor * age.float()),
    )
    if key_padding is not None:
        state_strength = state_strength * (~key_padding.bool()).unsqueeze(-1)
    return observed.to(torch.float32), state_strength, age.to(torch.float32)


def build_framewise_interaction_features(
    positions: torch.Tensor, valid_mask: torch.Tensor
):
    """Return pairwise actor interaction features [B,T,N,N,7]."""
    batch_size, actors, steps, _ = positions.shape
    observed = valid_mask.bool()
    positions = torch.where(
        observed.unsqueeze(-1), positions, torch.zeros_like(positions)
    )
    time = torch.arange(steps, device=positions.device).view(1, 1, steps)
    candidates = torch.where(observed, time, -1)
    previous = torch.cat(
        [
            torch.full_like(candidates[..., :1], -1),
            candidates[..., :-1].cummax(dim=-1).values,
        ],
        dim=-1,
    )
    safe_previous = previous.clamp(min=0)
    previous_position = positions.gather(
        2, safe_previous.unsqueeze(-1).expand(batch_size, actors, steps, 2)
    )
    gap = (time - safe_previous).clamp(min=1).to(positions.dtype)
    velocity = (positions - previous_position) / gap.unsqueeze(-1)
    has_previous = (previous >= 0) & observed
    velocity = velocity * has_previous.unsqueeze(-1).to(positions.dtype)

    position_t = positions.permute(0, 2, 1, 3)
    velocity_t = velocity.permute(0, 2, 1, 3)
    query_position = position_t.unsqueeze(-2)
    source_position = position_t.unsqueeze(-3)
    relative_position = source_position - query_position
    distance = torch.linalg.vector_norm(relative_position, dim=-1, keepdim=True)
    relative_velocity = velocity_t.unsqueeze(-3) - velocity_t.unsqueeze(-2)
    heading = torch.atan2(velocity_t[..., 1], velocity_t[..., 0])
    heading_difference = heading.unsqueeze(-1) - heading.unsqueeze(-2)
    pair_valid = observed.permute(0, 2, 1)
    pair_valid = pair_valid.unsqueeze(-1) & pair_valid.unsqueeze(-2)
    features = torch.cat(
        [
            relative_position,
            torch.log1p(distance),
            relative_velocity,
            torch.cos(heading_difference).unsqueeze(-1),
            torch.sin(heading_difference).unsqueeze(-1),
        ],
        dim=-1,
    )
    features = features * pair_valid.unsqueeze(-1).to(features.dtype)
    return features


def aggregate_history_by_state_strength(
    hidden: torch.Tensor,
    state_strength: torch.Tensor,
    content_scores: torch.Tensor,
    beta: torch.Tensor | float = 1.0,
    eps: float = 1e-8,
    use_state_strength_weighting: bool = True,
):
    """Aggregate temporal states using content scores and state strength."""
    logits = content_scores.squeeze(-1)
    if use_state_strength_weighting:
        logits = logits + torch.as_tensor(
            beta, dtype=hidden.dtype, device=hidden.device
        ) * torch.log(state_strength.to(hidden.dtype).clamp_min(eps))
    weights = torch.softmax(logits, dim=-1)
    weighted_history = torch.sum(weights.unsqueeze(-1) * hidden, dim=-2)
    if not use_state_strength_weighting:
        return weighted_history, weights
    terminal_state_strength = state_strength[..., -1].unsqueeze(-1).to(hidden.dtype)
    output = (
        terminal_state_strength * hidden[..., -1, :]
        + (1.0 - terminal_state_strength) * weighted_history
    )
    return output, weights


class FramewiseInteractionModel(nn.Module):
    """实现“逐帧交互建模”阶段。"""

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_heads: int,
        operator: str = "max",
        p_dropout: float = 0.2,
        use_interaction_bias: bool = True,
        use_source_filter: bool = True,
    ):
        super().__init__()
        if operator not in {"mean", "sum", "max"}:
            raise ValueError(f"unsupported interaction operator={operator!r}")
        self.operator = operator
        self.use_interaction_bias = bool(use_interaction_bias)
        self.use_source_filter = bool(use_source_filter)
        self.patch_embed = nn.Sequential(
            nn.Linear(input_dim, model_dim, bias=False),
            nn.LayerNorm(model_dim),
            nn.LeakyReLU(),
            nn.Dropout(p_dropout),
        )
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=p_dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim, bias=False),
            nn.LeakyReLU(),
            nn.Linear(model_dim, model_dim, bias=False),
            nn.Dropout(p_dropout),
        )
        self.mask_embed = nn.Sequential(
            nn.Linear(model_dim, model_dim, bias=False),
            nn.LayerNorm(model_dim),
            nn.LeakyReLU(),
            nn.Dropout(p_dropout),
        )
        self.interaction_bias = (
            nn.Sequential(
                nn.Linear(7, max(model_dim // 4, 16)),
                nn.GELU(),
                nn.Linear(max(model_dim // 4, 16), num_heads),
            )
            if self.use_interaction_bias
            else None
        )
        if self.interaction_bias is not None:
            nn.init.zeros_(self.interaction_bias[-1].weight)
            nn.init.zeros_(self.interaction_bias[-1].bias)
        self.apply(self._init_weights)
        if self.interaction_bias is not None:
            nn.init.zeros_(self.interaction_bias[-1].weight)
            nn.init.zeros_(self.interaction_bias[-1].bias)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _ghost(
        self, mask_bt_n: torch.Tensor, key_padding_bt: torch.Tensor
    ) -> torch.Tensor:
        """Pool valid-actor observation masks into one ghost token."""
        keep = (~key_padding_bt).float().unsqueeze(-1)  # [B*T,N,1]
        mask_bt_n = mask_bt_n.float() * keep.squeeze(-1)  # padding 行置 0
        c = self.mask_embed[0].in_features
        mask_c = mask_bt_n.unsqueeze(-1).expand(-1, -1, c)  # [B*T,N,C]
        mask_emb = self.mask_embed(mask_c) * keep  # [B*T,N,C]
        if self.operator == "mean":
            denom = keep.sum(dim=1, keepdim=True).clamp(min=1.0)
            return mask_emb.sum(dim=1, keepdim=True) / denom
        if self.operator == "sum":
            return mask_emb.sum(dim=1, keepdim=True)
        # max — exclude padding rather than multiplying it by zero: a zero
        # padding value must not win against a valid negative embedding.
        masked = mask_emb.masked_fill(~keep.bool(), torch.finfo(mask_emb.dtype).min)
        return masked.max(dim=1, keepdim=True).values

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        key_padding: torch.Tensor,
        positions: torch.Tensor,
        source_eligibility: torch.Tensor,
    ) -> torch.Tensor:
        """x [B,T,N,C_in]，mask [B,T,N]（有效=1），key_padding [B,N]（True=padding）。

        返回 [B,T,N,C]。
        """
        B, T, N, _ = x.shape
        h = self.patch_embed(x)  # [B,T,N,C]

        # ghost：padding actor 不参与池化
        mask_bt_n = mask.reshape(B * T, N)
        kp_bt = key_padding.unsqueeze(1).expand(B, T, N).reshape(B * T, N)
        ghost = self._ghost(mask_bt_n, kp_bt)  # [B*T,1,C]

        tokens = torch.cat([ghost, h.reshape(B * T, N, -1)], dim=1)  # [B*T, N+1, C]

        qkv = self.norm1(tokens)
        if self.use_interaction_bias:
            interactions = build_framewise_interaction_features(
                positions, source_eligibility.bool()
            )
            interaction_bias = self.interaction_bias(interactions).permute(
                0, 1, 4, 2, 3
            )
            interaction_bias = interaction_bias.reshape(
                B * T * self.attn_layer.num_heads, N, N
            )
            interaction_bias = interaction_bias.to(dtype=qkv.dtype)
        else:
            interaction_bias = qkv.new_zeros(
                B * T * self.attn_layer.num_heads, N, N
            )
        actor_sources = (
            source_eligibility.bool().permute(0, 2, 1)
            if self.use_source_filter
            else torch.ones(B, T, N, dtype=torch.bool, device=x.device)
        )
        source_allowed = torch.cat(
            [
                torch.ones(B, T, 1, dtype=torch.bool, device=x.device),
                actor_sources & ~key_padding.unsqueeze(1),
            ],
            dim=-1,
        ).reshape(B * T, N + 1)
        attn_mask = (
            qkv.new_zeros(B * T, N + 1, N + 1)
            .reshape(B * T, 1, N + 1, N + 1)
            .expand(B * T, self.attn_layer.num_heads, N + 1, N + 1)
            .clone()
        )
        attn_mask[..., 1:, 1:] = interaction_bias.reshape(
            B * T, self.attn_layer.num_heads, N, N
        )
        attn_mask = attn_mask.masked_fill(
            ~source_allowed[:, None, None, :], float("-inf")
        )
        attn_out, _ = self.attn_layer(
            qkv,
            qkv,
            qkv,
            attn_mask=attn_mask.reshape(
                B * T * self.attn_layer.num_heads, N + 1, N + 1
            ),
            need_weights=False,
        )
        x_attn = attn_out
        x_attn = self.ffn(self.norm2(x_attn)) + x_attn
        y = self.norm3(x_attn)[:, 1:, :].reshape(B, T, N, -1)
        return y


class StateStrengthMamba(nn.Module):
    """状态强度加权时序聚合内部的 Mamba 输入调制环节。"""

    def __init__(
        self,
        model_dim: int,
        depth: int = 4,
        p_dropout: float = 0.2,
        drop_path: float = 0.2,
        use_state_strength_modulation: bool = True,
    ):
        super().__init__()
        self.use_state_strength_modulation = bool(use_state_strength_modulation)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.LeakyReLU(),
            nn.Dropout(p_dropout),
        )
        self.mamba_blocks = self._stack(model_dim, depth, drop_path, 0)
        self.mamba_norm = RMSNorm(model_dim, eps=1e-5)

    @staticmethod
    def _stack(model_dim, depth, drop_path, offset):
        return nn.ModuleList(
            [
                create_mamba_block(
                    d_model=model_dim,
                    layer_idx=offset + i,
                    drop_path=drop_path,
                )
                for i in range(depth)
            ]
        )

    def _run(self, x):
        residual = None
        for blk in self.mamba_blocks:
            x, residual = blk(x, residual)
        return rms_norm_fn(
            x,
            self.mamba_norm.weight,
            self.mamba_norm.bias,
            eps=self.mamba_norm.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True,
        )

    def forward(
        self,
        x_b_t_n_c: torch.Tensor,
        state_strength: torch.Tensor,
    ):
        """Return forward-only temporal states with shape ``[B,N,T,C]``."""
        B, T, N, C = x_b_t_n_c.shape
        x_forw = self.mlp(x_b_t_n_c).permute(0, 2, 1, 3).reshape(B * N, T, C)
        if self.use_state_strength_modulation:
            strength_flat = state_strength.reshape(B * N, T, 1).to(x_forw.dtype)
            x_forw = x_forw * strength_flat
        hidden = self._run(x_forw)
        return hidden.reshape(B, N, T, C)

class StateStrengthHistoryReadout(nn.Module):
    """状态强度加权时序聚合内部的历史状态读出环节。"""

    def __init__(
        self,
        embed_dim: int,
        use_state_strength_readout: bool = True,
    ):
        super().__init__()
        self.use_state_strength_readout = bool(use_state_strength_readout)
        self.score = nn.Linear(embed_dim, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)
        self.register_buffer("beta", torch.tensor(1.0))

    def forward(
        self,
        hidden: torch.Tensor,
        state_strength: torch.Tensor,
    ):
        return aggregate_history_by_state_strength(
            hidden,
            state_strength,
            self.score(hidden),
            beta=self.beta,
            use_state_strength_weighting=self.use_state_strength_readout,
        )


class StateStrengthTemporalAggregation(nn.Module):
    """实现“状态强度加权的时序聚合”，组合 Mamba 输入调制与历史状态读出。"""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        p_dropout: float = 0.2,
        drop_path: float = 0.2,
        use_state_strength_modulation: bool = True,
        use_state_strength_readout: bool = True,
    ):
        super().__init__()
        self.mamba = StateStrengthMamba(
            embed_dim,
            depth=depth,
            p_dropout=p_dropout,
            drop_path=drop_path,
            use_state_strength_modulation=use_state_strength_modulation,
        )
        self.history_readout = StateStrengthHistoryReadout(
            embed_dim,
            use_state_strength_readout=use_state_strength_readout,
        )

    def forward(
        self,
        interaction_states: torch.Tensor,
        state_strength: torch.Tensor,
    ):
        hidden = self.mamba(
            interaction_states,
            state_strength=state_strength,
        )
        return self.history_readout(hidden, state_strength)


class ObservationStateEncoder(nn.Module):
    """实现“观测状态感知时空编码器”，包含逐帧交互建模和状态强度加权的时序聚合。"""

    def __init__(
        self,
        input_dim: int = 4,
        embed_dim: int = 128,
        num_heads: int = 8,
        depth: int = 4,
        operator: str = "max",
        p_dropout: float = 0.2,
        drop_path: float = 0.2,
        use_interaction_bias: bool = True,
        use_source_filter: bool = True,
        use_state_strength_modulation: bool = True,
        use_state_strength_readout: bool = True,
    ):
        super().__init__()
        self.framewise_interaction = FramewiseInteractionModel(
            input_dim,
            embed_dim,
            num_heads,
            operator=operator,
            p_dropout=p_dropout,
            use_interaction_bias=use_interaction_bias,
            use_source_filter=use_source_filter,
        )
        self.state_strength_decay_logit = nn.Parameter(torch.tensor(-2.2521685))
        self.state_strength_aggregation = StateStrengthTemporalAggregation(
            embed_dim,
            depth=depth,
            p_dropout=p_dropout,
            drop_path=drop_path,
            use_state_strength_modulation=use_state_strength_modulation,
            use_state_strength_readout=use_state_strength_readout,
        )

    @property
    def state_strength_decay(self):
        return torch.nn.functional.softplus(self.state_strength_decay_logit)

    def forward(
        self,
        hist_feat: torch.Tensor,
        valid_mask: torch.Tensor,
        key_padding: torch.Tensor,
        positions: torch.Tensor,
    ):
        x = hist_feat.permute(0, 2, 1, 3)
        mask = valid_mask.permute(0, 2, 1).float()
        padding = key_padding

        source_eligibility, state_strength, _ = build_observation_state(
            valid_mask, self.state_strength_decay, padding
        )
        interaction_states = self.framewise_interaction(
            x,
            mask,
            padding,
            positions=positions,
            source_eligibility=source_eligibility,
        )
        output, _ = self.state_strength_aggregation(
            interaction_states, state_strength
        )

        return output * (~padding).unsqueeze(-1).to(output.dtype)
