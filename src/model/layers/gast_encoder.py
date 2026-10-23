"""GAST Encoder for GAST-PF.

来源：colorfulfuture/UniTraj-pytorch（ICLR'25 Sports-Traj）
    models/encoder_decoder.py :: Past_Encoder / Spatial_Encoder / Temporal_Encoder
    models/modules.py        :: Attn_Block / Mamba_Block / Temporal_Decay

机制引入：GSM 空间注意力（mask ghost token + 逐帧 agent 自注意力）、
Temporal_Decay 缺失衰减、前向/反向 Mamba 时序状态引入自 Sports-Traj / UniTraj
（colorfulfuture/UniTraj-pytorch），适配为缺失历史直接预测任务的编码器组件；
GAST-PF 的 decoder、loss、指标与 TrajGap 协议保持不变。
本文件不含外部方法的 Future_Encoder、CVAE 潜变量或 diverse loss。

与源实现的适配差异（逐条为 TrajGap 契约所需，非机制改动）：
1. 布局：源 [B,T,N,C] -> 本文件入口 [B,N,T,C]（GAST-PF batch 字段口径），内部
   转置到源布局后复用原计算图。
2. 动态 N + padding：源 Attn_Block 假设固定球员槽位（pos_embed 长度固定、
   MHA 无 key_padding_mask）。TrajGap 批内 N 可变：去除固定槽位 pos_embed
   （focal 恒在 0 位，邻居无序，槽位身份编码无依据——skill 裁定 actor 文件
   顺序不是社会排序），并给 MHA 传 key_padding_mask 屏蔽 padding actor。
   ghost token 恒可参加注意力（有效 actor >= 1 由数据层保证）。
3. mask 极性：源 mask=1 表示 observed（set_input 中 `torch.mul(x, m` 掩出
   观测）。TrajGap 的 x_valid_mask=True 同为有效，语义一致，直接使用。
4. delta：源 get_delta 递归产生前/后向距最近观测步数（观测点=1，缺口气累积，
   全缺行仍递增）。本文件向量化复现同一数值约定（观测点 1 起算），喂给
   Temporal_Decay 的 exp(-ReLU(Linear(.))) 衰减头，宽度=embed_dim。
5. 宽度：model_dim 由源默认 64 提到 128，与 GAST-PF decoder 隐藏维对齐，省去接口
   投影；参数量变化须记录在实验 manifest。
6. 时序方向：默认保留源实现的前向/反向两条独立 Mamba 栈；新增
   ``temporal_direction=\"unidirectional\"``，用于严格因果的前向 Mamba + 末帧读出。
   两种模式共享 GSM、Temporal_Decay、decoder 和编码后场景尾部。
"""

import torch
import torch.nn as nn

from .mamba.vim_mamba import create_block

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def missing_run_deltas(mask: torch.Tensor) -> tuple:
    """前/后向距最近观测步数，与 UniTraj get_delta 递归逐位一致。

    Input : mask [B, N, T] bool/float，True=该帧有效观测（源 mask=1 同义）。
    Output: (d_forw, d_back) [B, N, T] float——源递归 d[0]=0；t>=1 时
            d[t] = 1 + (1 - m[t]) * d[t-1]：观测点=1，其后连续缺失
            2,3,4...；前缀全缺时 d[t]=t。反向为翻转后同递推。
    """
    obs = (mask.float() > 0).float()

    def _scan(o):
        d = torch.zeros_like(o)
        prev = torch.zeros(*o.shape[:-1], device=o.device, dtype=o.dtype)
        for t in range(1, o.shape[-1]):
            prev = 1.0 + (1.0 - o[..., t]) * prev
            d[..., t] = prev
        return d

    d_forw = _scan(obs)
    # Keep the backward delta in the flipped-time layout.  UniTraj first
    # flips the input sequence and then multiplies it by this tensor; flipping
    # the delta back here would silently change the decay alignment.
    d_back = _scan(torch.flip(obs, dims=[-1]))
    return d_forw, d_back


def build_observation_evidence_state(
    valid_mask: torch.Tensor,
    tau: torch.Tensor | float,
    key_padding: torch.Tensor | None = None,
):
    """Build the shared Observation Evidence State (OES)."""
    observed = valid_mask.bool()
    if key_padding is not None:
        observed = observed & ~key_padding.bool().unsqueeze(-1)
    steps = torch.arange(
        observed.shape[-1], device=observed.device, dtype=torch.long
    ).view(*([1] * (observed.ndim - 1)), -1)
    last_observed = torch.where(observed, steps, -1).cummax(dim=-1).values
    age = torch.where(last_observed >= 0, steps - last_observed, steps + 1)
    tau_tensor = torch.as_tensor(tau, dtype=torch.float32, device=observed.device)
    reliability = torch.where(
        observed,
        torch.ones_like(age, dtype=torch.float32),
        torch.exp(-tau_tensor * age.float()),
    )
    if key_padding is not None:
        reliability = reliability * (~key_padding.bool()).unsqueeze(-1)
    return observed.to(torch.float32), reliability, age.to(torch.float32)


def relative_geometry_features(positions: torch.Tensor, valid_mask: torch.Tensor):
    """Return pairwise relative geometry [B,T,N,N,7] and its direct-evidence mask."""
    batch_size, actors, steps, _ = positions.shape
    observed = valid_mask.bool()
    positions = torch.where(
        observed.unsqueeze(-1), positions, torch.zeros_like(positions)
    )
    time = torch.arange(steps, device=positions.device).view(1, 1, steps)
    candidates = torch.where(observed, time, -1)
    previous = torch.cat(
        [torch.full_like(candidates[..., :1], -1),
         candidates[..., :-1].cummax(dim=-1).values],
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
    return features, pair_valid


def route_bidirectional_paths(forward: torch.Tensor, backward: torch.Tensor,
                              reliability: torch.Tensor):
    """Fuse directional states using cumulative missing-evidence burden."""
    missing = 1.0 - reliability
    burden_forward = torch.cumsum(missing, dim=-1)
    burden_backward = torch.flip(
        torch.cumsum(torch.flip(missing, dims=[-1]), dim=-1), dims=[-1]
    )
    logits = torch.stack([-burden_forward, -burden_backward], dim=-1)
    alpha = torch.softmax(logits, dim=-1)
    fused = 2.0 * (
        alpha[..., 0, None] * forward + alpha[..., 1, None] * backward
    )
    return fused, alpha


def evidence_weighted_history_readout(
    hidden: torch.Tensor,
    reliability: torch.Tensor,
    scores: torch.Tensor,
    beta: torch.Tensor | float = 1.0,
    eps: float = 1e-8,
):
    """Evidence-Weighted History Readout (EWHR)."""
    logits = scores.squeeze(-1) + torch.as_tensor(
        beta, dtype=hidden.dtype, device=hidden.device
    ) * torch.log(reliability.to(hidden.dtype).clamp_min(eps))
    weights = torch.softmax(logits, dim=-1)
    evidence = torch.sum(weights.unsqueeze(-1) * hidden, dim=-2)
    terminal_reliability = reliability[..., -1].unsqueeze(-1).to(hidden.dtype)
    output = (
        terminal_reliability * hidden[..., -1, :]
        + (1.0 - terminal_reliability) * evidence
    )
    return output, weights


# Backward-compatible names for historical experiment scripts.
compute_evidence_state = build_observation_evidence_state
evidence_aware_readout = evidence_weighted_history_readout


class GSMAttnBlock(nn.Module):
    """UniTraj Attn_Block 移植：patch embed + mask ghost token + 逐帧自注意力。

    源结构：patch_embed -> ghost(mask 池化) 拼接 [B*T,N+1,C] -> (pos_embed)
    -> MHA -> FFN 残差 -> norm -> 去 ghost。本移植去除固定槽位 pos_embed
    （动态 N），MHA 增加 key_padding_mask 屏蔽 padding actor。
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_heads: int,
        operator: str = "max",
        p_dropout: float = 0.2,
        observation_evidence_spatial: bool = False,
        ecsr: bool | None = None,
    ):
        super().__init__()
        if operator not in {"mean", "sum", "max"}:
            raise ValueError(f"unsupported GSM operator={operator!r}")
        self.operator = operator
        if ecsr is not None:
            observation_evidence_spatial = (
                bool(observation_evidence_spatial) or bool(ecsr)
            )
        self.observation_evidence_spatial = bool(observation_evidence_spatial)
        # Compatibility attribute for historical ECSR code.
        self.ecsr = self.observation_evidence_spatial
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
        if self.observation_evidence_spatial:
            self.geometry_bias = nn.Sequential(
                nn.Linear(7, max(model_dim // 4, 16)),
                nn.GELU(),
                nn.Linear(max(model_dim // 4, 16), num_heads),
            )
            nn.init.zeros_(self.geometry_bias[-1].weight)
            nn.init.zeros_(self.geometry_bias[-1].bias)
        self.apply(self._init_weights)
        if self.observation_evidence_spatial:
            nn.init.zeros_(self.geometry_bias[-1].weight)
            nn.init.zeros_(self.geometry_bias[-1].bias)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _ghost(self, mask_bt_n: torch.Tensor, key_padding_bt: torch.Tensor) -> torch.Tensor:
        """mask [B*T, N] -> ghost token [B*T, 1, C]（源 get_mask_headpatch，padding 行除外）。

        源语义：mask 值先 repeat 到 C 维再过 mask_embed（Linear C->C）。
        """
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                key_padding: torch.Tensor, positions: torch.Tensor | None = None,
                source_eligibility: torch.Tensor | None = None,
                query_reliability: torch.Tensor | None = None) -> torch.Tensor:
        """x [B,T,N,C_in]，mask [B,T,N]（有效=1），key_padding [B,N]（True=padding）。

        返回 [B,T,N,C]。
        """
        B, T, N, _ = x.shape
        h = self.patch_embed(x)  # [B,T,N,C]

        # ghost：padding actor 不参与池化
        mask_bt_n = mask.reshape(B * T, N)
        kp_bt = key_padding.unsqueeze(1).expand(B, T, N).reshape(B * T, N)
        ghost = self._ghost(mask_bt_n, kp_bt)  # [B*T,1,C]

        tokens = torch.cat(
            [ghost, h.reshape(B * T, N, -1)], dim=1
        )  # [B*T, N+1, C]

        qkv = self.norm1(tokens)
        # ghost(槽位 0) 恒有效；agent 槽位按 key_padding 屏蔽
        attn_key_padding = torch.cat(
            [torch.zeros(B * T, 1, dtype=torch.bool, device=x.device),
             key_padding.unsqueeze(1).expand(B, T, N).reshape(B * T, N)], dim=1)
        if self.observation_evidence_spatial:
            if positions is None or source_eligibility is None:
                raise ValueError(
                    "OER-GSM requires positions and source eligibility"
                )
            geometry, _ = relative_geometry_features(positions, source_eligibility.bool())
            geometry_bias = self.geometry_bias(geometry).permute(0, 1, 4, 2, 3)
            geometry_bias = geometry_bias.reshape(
                B * T * self.attn_layer.num_heads, N, N
            )
            geometry_bias = geometry_bias.to(dtype=qkv.dtype)
            source_allowed = torch.cat(
                [
                    torch.ones(B, T, 1, dtype=torch.bool, device=x.device),
                    source_eligibility.bool().permute(0, 2, 1)
                    & ~key_padding.unsqueeze(1),
                ],
                dim=-1,
            ).reshape(B * T, N + 1)
            attn_mask = qkv.new_zeros(
                B * T, N + 1, N + 1
            ).reshape(B * T, 1, N + 1, N + 1).expand(
                B * T, self.attn_layer.num_heads, N + 1, N + 1
            ).clone()
            attn_mask[..., 1:, 1:] = geometry_bias.reshape(
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
        else:
            attn_out, _ = self.attn_layer(qkv, qkv, qkv,
                                          key_padding_mask=attn_key_padding)
        x_attn = attn_out  # 源实现：Try no Res
        x_attn = self.ffn(self.norm2(x_attn)) + x_attn
        y = self.norm3(x_attn)[:, 1:, :].reshape(B, T, N, -1)
        if query_reliability is not None:
            y = y * query_reliability.unsqueeze(-1).to(y.dtype)
        return y


class ObservationEvidenceGSM(GSMAttnBlock):
    """OER-GSM: Observation-Evidence Relational GSM."""

    def __init__(self, *args, **kwargs):
        kwargs["observation_evidence_spatial"] = True
        super().__init__(*args, **kwargs)


class TemporalEncoder(nn.Module):
    """Gap-aware temporal Mamba with bidirectional and causal modes."""

    def __init__(self, model_dim: int, depth: int = 4, p_dropout: float = 0.2,
                 drop_path: float = 0.2,
                 temporal_direction: str = "bidirectional"):
        super().__init__()
        direction_aliases = {
            "bi": "bidirectional",
            "bidir": "bidirectional",
            "uni": "unidirectional",
            "causal": "unidirectional",
        }
        temporal_direction = direction_aliases.get(
            temporal_direction, temporal_direction
        )
        if temporal_direction not in {"bidirectional", "unidirectional"}:
            raise ValueError(
                "temporal_direction must be 'bidirectional' or 'unidirectional', "
                f"got {temporal_direction!r}"
            )
        self.temporal_direction = temporal_direction
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.LeakyReLU(),
            nn.Dropout(p_dropout),
        )
        self.ssm_forw = self._stack(model_dim, depth, drop_path, 0)
        self.norm_f_forw = RMSNorm(model_dim, eps=1e-5)
        if self.temporal_direction == "bidirectional":
            self.ssm_back = self._stack(model_dim, depth, drop_path, depth)
            self.norm_f_back = RMSNorm(model_dim, eps=1e-5)

    @staticmethod
    def _stack(model_dim, depth, drop_path, offset):
        return nn.ModuleList([
            create_block(d_model=model_dim, layer_idx=offset + i,
                         drop_path=drop_path, bimamba=False, rms_norm=True)
            for i in range(depth)
        ])

    def _run(self, x, stack, norm):
        residual = None
        for blk in stack:
            x, residual = blk(x, residual)
        fused = rms_norm_fn if isinstance(norm, RMSNorm) else layer_norm_fn
        return fused(x, norm.weight, norm.bias, eps=norm.eps, residual=residual,
                     prenorm=False, residual_in_fp32=True)

    def forward(self, x_b_t_n_c: torch.Tensor, gamma_forw=None, gamma_back=None,
                reliability: torch.Tensor | None = None,
                return_directions: bool = False):
        """Encode ``x`` and return the configured temporal state sequence.

        ``bidirectional`` keeps the historical flattened return contract when
        evidence routing is disabled and returns ``[B,N,T,C]`` for routed
        states. ``unidirectional`` always returns causal states as
        ``[B,N,T,C]`` so the caller can explicitly read the terminal state or
        perform a causal evidence-aware readout.
        """
        B, T, N, C = x_b_t_n_c.shape
        x_forw = self.mlp(x_b_t_n_c).permute(0, 2, 1, 3).reshape(B * N, T, C)

        if reliability is None:
            x_forw = x_forw * gamma_forw.reshape(B * N, T, C)
        else:
            reliability_flat = reliability.reshape(B * N, T, 1).to(x_forw.dtype)
            x_forw = x_forw * reliability_flat

        y_forw = self._run(x_forw, self.ssm_forw, self.norm_f_forw)
        forward_states = y_forw.reshape(B, N, T, C)

        if self.temporal_direction == "unidirectional":
            if return_directions:
                raise ValueError(
                    "return_directions is only available in bidirectional mode"
                )
            return forward_states

        x_flip = torch.flip(x_b_t_n_c, dims=[1])
        x_back = self.mlp(x_flip).permute(0, 2, 1, 3).reshape(B * N, T, C)
        if reliability is None:
            x_back = x_back * gamma_back.reshape(B * N, T, C)
        else:
            x_back = x_back * torch.flip(reliability_flat, dims=[1])

        y_back = self._run(x_back, self.ssm_back, self.norm_f_back)
        y_back = torch.flip(y_back, dims=[1])
        if reliability is None:
            fused = y_forw + y_back
            if return_directions:
                backward_states = y_back.reshape(B, N, T, C)
                return fused, forward_states, backward_states
            return fused
        backward_states = y_back.reshape(B, N, T, C)
        return route_bidirectional_paths(
            forward_states, backward_states, reliability
        )[0]


class EvidenceModulatedCausalMamba(TemporalEncoder):
    """EMC-Mamba: Evidence-Modulated Causal Mamba temporal encoder."""


class DirectionalAttentionReadout(nn.Module):
    """Pool both temporal directions into one gated actor token."""

    def __init__(self, embed_dim: int, max_steps: int = 8,
                 hidden_dim: int = 32, gate_hidden_dim: int = 16):
        super().__init__()
        self.max_steps = int(max_steps)
        self.state_projection = nn.Linear(embed_dim, hidden_dim)
        self.reliability_projection = nn.Linear(2, hidden_dim, bias=False)
        self.time_embedding = nn.Parameter(torch.empty(1, 1, self.max_steps, hidden_dim))
        self.direction_embedding = nn.Parameter(torch.empty(2, hidden_dim))
        self.attention_score = nn.Linear(hidden_dim, 1)
        self.gate = nn.Sequential(
            nn.Linear(4, gate_hidden_dim),
            nn.SiLU(),
            nn.Linear(gate_hidden_dim, 1),
        )
        self.beta = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.time_embedding, std=0.02)
        nn.init.normal_(self.direction_embedding, std=0.02)

    @staticmethod
    def _gate_features(valid_mask: torch.Tensor) -> torch.Tensor:
        steps = valid_mask.shape[-1]
        split = steps // 2
        left_ratio = (
            valid_mask[..., :split].float().mean(dim=-1)
            if split > 0 else torch.zeros_like(valid_mask[..., 0], dtype=torch.float32)
        )
        right_ratio = (
            valid_mask[..., split:].float().mean(dim=-1)
            if steps - split > 0 else torch.zeros_like(left_ratio)
        )
        terminal_missing = (~valid_mask[..., -1]).float()

        missing = ~valid_mask
        run = torch.zeros_like(terminal_missing)
        max_run = torch.zeros_like(terminal_missing)
        for index in range(steps):
            run = torch.where(missing[..., index], run + 1.0, torch.zeros_like(run))
            max_run = torch.maximum(max_run, run)
        max_run = max_run / max(steps, 1)
        return torch.stack(
            [left_ratio, right_ratio, terminal_missing, max_run], dim=-1
        )

    def _pool_direction(self, states: torch.Tensor, valid_mask: torch.Tensor,
                        age: torch.Tensor, direction: int,
                        pool_mask: torch.Tensor) -> torch.Tensor:
        _, _, steps, _ = states.shape
        reliability = torch.stack(
            [valid_mask.float(), age.float().clamp(min=0) / max(steps, 1)], dim=-1
        )
        features = (
            self.state_projection(states)
            + self.reliability_projection(reliability)
            + self.time_embedding[:, :, :steps]
            + self.direction_embedding[direction].view(1, 1, 1, -1)
        )
        scores = self.attention_score(torch.tanh(features)).squeeze(-1)
        scores = scores.masked_fill(~pool_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(weights.unsqueeze(-1) * states, dim=2)

    def forward(self, forward_states: torch.Tensor, backward_states: torch.Tensor,
                valid_mask: torch.Tensor, key_padding: torch.Tensor,
                age_forward: torch.Tensor, age_backward: torch.Tensor) -> torch.Tensor:
        """Inputs are [B,N,T,C] states and [B,N,T] masks/ages."""
        steps = forward_states.shape[2]
        if steps > self.max_steps:
            raise ValueError(
                f"directional readout supports at most {self.max_steps} steps, got {steps}"
            )
        observed = valid_mask.bool() & ~key_padding.bool().unsqueeze(-1)
        has_observation = observed.any(dim=-1, keepdim=True)
        fallback = torch.zeros_like(observed)
        fallback[..., -1] = ~has_observation.squeeze(-1)
        pool_mask = observed | fallback

        pooled_forward = self._pool_direction(
            forward_states, observed, age_forward, 0, pool_mask
        )
        pooled_backward = self._pool_direction(
            backward_states, observed, age_backward, 1, pool_mask
        )
        evidence = pooled_forward + pooled_backward
        base = (forward_states + backward_states)[:, :, -1]

        gate_features = self._gate_features(observed)
        gate = torch.sigmoid(self.gate(gate_features))
        output = base + torch.tanh(self.beta) * gate * (evidence - base)
        return output * (~key_padding).unsqueeze(-1).to(output.dtype)


class TemporalDecay(nn.Module):
    """UniTraj Temporal_Decay 移植：exp(-ReLU(Linear(delta))) 衰减头。"""

    def __init__(self, model_dim: int, delta_dim: int = 2):
        super().__init__()
        self.delta_dim = delta_dim
        self.mlp = nn.Linear(delta_dim, model_dim, bias=False)
        self.relu = nn.ReLU()

    def forward(self, delta_b_n_t: torch.Tensor) -> torch.Tensor:
        """delta [B,N,T,delta_dim] -> [B,N,T,C]，元素落在 (0,1]。"""
        if delta_b_n_t.shape[-1] != self.delta_dim:
            raise ValueError(
                f"expected delta_dim={self.delta_dim}, got {delta_b_n_t.shape[-1]}"
            )
        x = self.relu(self.mlp(delta_b_n_t))
        return torch.exp(-x)


def anchor_lag_norm(valid_mask: torch.Tensor) -> torch.Tensor:
    """anchor lag（距最后有效观测的步数）归一化到 [0, 1]。

    Input : valid_mask [B, N, T] bool，True=该帧有效观测。
    Output: [B, N, 1] float。last_valid_index = max{t | m[t]=1}；
            anchor_lag = (T - 1) - last_valid_index；除以 max(T - 1, 1)。
            全无效行取 1.0（最大不确定），保证有限且恒落在 [0, 1]。
    """
    T = valid_mask.shape[-1]
    idx = torch.arange(T, device=valid_mask.device)
    last = (valid_mask.long() * idx).max(dim=-1).values
    last = torch.where(
        valid_mask.any(dim=-1), last, torch.full_like(last, -1)
    )
    denom = max(T - 1, 1)
    lag = ((T - 1) - last).clamp(min=0, max=denom)
    return (lag.float() / denom).unsqueeze(-1)


class AnchorGeoProjection(nn.Module):
    """Stage 1 AnchorGeo：把 anchor 空间与末端方向证据投影到 actor token 空间。

    输入几何 [B, N, 5] = [x_centers(2), cos(last_valid_angle), sin(...),
    anchor_lag_norm(1)]，只来自可见历史（不得使用 future）。

    纪律：
    - 输出层零初始化 → 训练起点与基础版本（无 AnchorGeo）严格等价，可归因；
    - padding actor 的几何输入与输出显式置零；
    - 不改变 GSM attention，也不改变 decoder 输入维度。
    """

    def __init__(self, embed_dim: int = 128, geo_dim: int = 5,
                 hidden_dim: int | None = None, p_dropout: float = 0.2):
        super().__init__()
        self.geo_dim = geo_dim
        hidden = hidden_dim or embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(geo_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden, embed_dim, bias=False),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        # 残差式零初始化：初始时刻本模块贡献恒为 0。
        nn.init.zeros_(self.mlp[-1].weight)

    def forward(self, geo: torch.Tensor, key_padding: torch.Tensor) -> torch.Tensor:
        """geo [B,N,5]，key_padding [B,N]（True=padding）-> [B,N,embed_dim]。"""
        if geo.shape[-1] != self.geo_dim:
            raise ValueError(
                f"expected geo_dim={self.geo_dim}, got {geo.shape[-1]}"
            )
        keep = (~key_padding).unsqueeze(-1).to(geo.dtype)
        y = self.mlp(geo * keep)
        return y * keep


class GASTEncoder(nn.Module):
    """GAST-PF encoder with modular observation-evidence reasoning.

    The named innovation modules are:
      - OER-GSM: Observation-Evidence Relational GSM;
      - EMC-Mamba: Evidence-Modulated Causal Mamba;
      - EWHR: Evidence-Weighted History Readout.

    ``ecsr`` and ``hierarchical_evidence`` remain accepted as compatibility
    aliases for historical checkpoints. New configurations should use the
    explicit module switches below.
    """

    def __init__(
        self,
        input_dim: int = 4,
        embed_dim: int = 128,
        num_heads: int = 8,
        depth: int = 4,
        operator: str = "max",
        p_dropout: float = 0.2,
        drop_path: float = 0.2,
        observation_evidence_spatial: bool = False,
        evidence_modulated_temporal: bool = False,
        evidence_weighted_readout: bool = False,
        directional_readout: bool = False,
        hierarchical_scene_reasoning: bool = False,
        max_steps: int = 8,
        temporal_direction: str = "bidirectional",
        ecsr: bool | None = None,
        hierarchical_evidence: bool | None = None,
    ):
        super().__init__()
        direction_aliases = {
            "bi": "bidirectional",
            "bidir": "bidirectional",
            "uni": "unidirectional",
            "causal": "unidirectional",
        }
        temporal_direction = direction_aliases.get(
            temporal_direction, temporal_direction
        )
        if temporal_direction not in {"bidirectional", "unidirectional"}:
            raise ValueError(
                "temporal_direction must be 'bidirectional' or 'unidirectional', "
                f"got {temporal_direction!r}"
            )

        legacy_evidence = bool(ecsr) if ecsr is not None else False
        legacy_hierarchical = (
            bool(hierarchical_evidence)
            if hierarchical_evidence is not None else False
        )
        self.observation_evidence_spatial = bool(
            observation_evidence_spatial or legacy_evidence
        )
        self.evidence_modulated_temporal = bool(
            evidence_modulated_temporal or legacy_evidence
        )
        self.evidence_weighted_readout = bool(
            evidence_weighted_readout or legacy_evidence
        )
        self.hierarchical_scene_reasoning = bool(
            hierarchical_scene_reasoning or legacy_hierarchical
        )
        self.evidence_enabled = any(
            (
                self.observation_evidence_spatial,
                self.evidence_modulated_temporal,
                self.evidence_weighted_readout,
                self.hierarchical_scene_reasoning,
            )
        )

        if self.evidence_weighted_readout and not self.evidence_modulated_temporal:
            raise ValueError(
                "EWHR requires evidence_modulated_temporal=True"
            )

        # Compatibility attributes used by historical checkpoint contracts.
        self.ecsr = self.evidence_enabled
        self.hierarchical_evidence = self.hierarchical_scene_reasoning
        self.use_directional_readout = bool(directional_readout)
        self.temporal_direction = temporal_direction
        if self.hierarchical_scene_reasoning and not self.evidence_enabled:
            raise ValueError(
                "hierarchical_scene_reasoning requires an evidence module"
            )
        if self.use_directional_readout and temporal_direction != "bidirectional":
            raise ValueError(
                "directional_readout requires temporal_direction='bidirectional'"
            )
        if self.observation_evidence_spatial and self.use_directional_readout:
            raise ValueError(
                "observation_evidence_spatial is mutually exclusive with "
                "directional_readout"
            )

        spatial_cls = (
            ObservationEvidenceGSM
            if self.observation_evidence_spatial else GSMAttnBlock
        )
        self.spatial = spatial_cls(
            input_dim,
            embed_dim,
            num_heads,
            operator=operator,
            p_dropout=p_dropout,
        )
        temporal_cls = (
            EvidenceModulatedCausalMamba
            if self.evidence_modulated_temporal else TemporalEncoder
        )
        self.temporal = temporal_cls(
            embed_dim,
            depth=depth,
            p_dropout=p_dropout,
            drop_path=drop_path,
            temporal_direction=temporal_direction,
        )
        if self.use_directional_readout:
            self.directional_readout = DirectionalAttentionReadout(
                embed_dim=embed_dim, max_steps=max_steps
            )
        if self.evidence_enabled:
            self.evidence_tau_logit = nn.Parameter(torch.tensor(-2.2521685))
        if self.evidence_weighted_readout:
            self.readout_score = nn.Linear(embed_dim, 1)
            nn.init.zeros_(self.readout_score.weight)
            nn.init.zeros_(self.readout_score.bias)
            self.register_buffer("readout_beta", torch.tensor(1.0))
        if not self.evidence_modulated_temporal:
            self.temporal_decay = TemporalDecay(embed_dim, delta_dim=2)

    @property
    def evidence_tau(self):
        if not self.evidence_enabled:
            raise AttributeError(
                "evidence_tau is only available when evidence modules are enabled"
            )
        return torch.nn.functional.softplus(self.evidence_tau_logit)

    def _regular_temporal_readout(self, x_spa, valid_mask, key_padding, batch_size):
        d_forw, d_back = missing_run_deltas(valid_mask)
        delta_forw = d_forw.unsqueeze(-1).expand(*d_forw.shape, 2)
        g_forw = self.temporal_decay(delta_forw)
        if self.temporal_direction == "unidirectional":
            y = self.temporal(x_spa, g_forw)
            return y[:, :, -1]
        delta_back = d_back.unsqueeze(-1).expand(*d_back.shape, 2)
        g_back = self.temporal_decay(delta_back)
        if self.use_directional_readout:
            _, forward_states, backward_states = self.temporal(
                x_spa,
                g_forw,
                g_back,
                return_directions=True,
            )
            age_back_aligned = torch.flip(d_back, dims=[-1])
            return self.directional_readout(
                forward_states,
                backward_states,
                valid_mask,
                key_padding,
                d_forw,
                age_back_aligned,
            )
        y = self.temporal(x_spa, g_forw, g_back)
        return y[:, -1].reshape(batch_size, valid_mask.shape[1], -1)

    def forward(
        self,
        hist_feat: torch.Tensor,
        valid_mask: torch.Tensor,
        key_padding: torch.Tensor,
        positions: torch.Tensor | None = None,
        return_evidence: bool = False,
    ):
        batch_size, actors, steps, _ = hist_feat.shape
        x = hist_feat.permute(0, 2, 1, 3)
        mask = valid_mask.permute(0, 2, 1).float()
        padding = key_padding

        evidence_state = None
        if self.evidence_enabled:
            source_eligibility, reliability, age = build_observation_evidence_state(
                valid_mask, self.evidence_tau, padding
            )
            evidence_state = {
                "source_eligibility": source_eligibility,
                "reliability": reliability,
                "age": age,
            }
        else:
            source_eligibility = reliability = None

        if self.observation_evidence_spatial:
            if positions is None:
                raise ValueError(
                    "OER-GSM requires observed positions for relational attention"
                )
            x_spa = self.spatial(
                x,
                mask,
                padding,
                positions=positions,
                source_eligibility=source_eligibility,
                query_reliability=(
                    reliability.permute(0, 2, 1)
                    if self.hierarchical_scene_reasoning else None
                ),
            )
        else:
            x_spa = self.spatial(x, mask, padding)

        if self.evidence_modulated_temporal:
            y = self.temporal(x_spa, reliability=reliability)
        else:
            y = None

        if self.evidence_weighted_readout:
            scores = self.readout_score(y)
            output, _ = evidence_weighted_history_readout(
                y, reliability, scores, beta=self.readout_beta
            )
        elif self.evidence_modulated_temporal:
            if self.temporal_direction == "unidirectional":
                output = y[:, :, -1]
            else:
                output = (
                    y[:, :, -1]
                    if y.shape[1] == actors
                    else y[:, -1].reshape(batch_size, actors, -1)
                )
        else:
            output = self._regular_temporal_readout(
                x_spa, valid_mask, padding, batch_size
            )

        output = output * (~padding).unsqueeze(-1).to(output.dtype)
        if return_evidence:
            return output, evidence_state
        return output


# Compatibility alias for checkpoints and external research scripts that still
# import the old class name. New code should use GASTEncoder.
UniTrajPastEncoder = GASTEncoder
