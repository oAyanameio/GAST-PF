"""GAST Encoder for GAST-PF.

来源：colorfulfuture/UniTraj-pytorch（ICLR'25 Sports-Traj）
    models/encoder_decoder.py :: Past_Encoder / Spatial_Encoder / Temporal_Encoder
    models/modules.py        :: Attn_Block / Mamba_Block / Temporal_Decay

借用边界：GSM 空间注意力（mask ghost token + 逐帧 agent 自注意力）、
Temporal_Decay 缺失衰减、前向+反向双 Mamba 相加均来自外部机制并在此处作为
编码器组件使用；GAST-PF 的 decoder、loss、指标与 TrajGap 协议保持不变。
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
6. 双向 Mamba：源为两条独立 Mamba_Block（各 depth 层，残差相加）。本文件用
   当前运行环境的 VideoMamba fork create_block 单向栈等价实现（fork 的 bimamba
   旗标仅作用于主链裁定，不在此处使用）。
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


class GSMAttnBlock(nn.Module):
    """UniTraj Attn_Block 移植：patch embed + mask ghost token + 逐帧自注意力。

    源结构：patch_embed -> ghost(mask 池化) 拼接 [B*T,N+1,C] -> (pos_embed)
    -> MHA -> FFN 残差 -> norm -> 去 ghost。本移植去除固定槽位 pos_embed
    （动态 N），MHA 增加 key_padding_mask 屏蔽 padding actor。
    """

    def __init__(self, input_dim: int, model_dim: int, num_heads: int,
                 operator: str = "max", p_dropout: float = 0.2):
        super().__init__()
        if operator not in {"mean", "sum", "max"}:
            raise ValueError(f"unsupported GSM operator={operator!r}")
        self.operator = operator
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
        self.apply(self._init_weights)

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
                key_padding: torch.Tensor) -> torch.Tensor:
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
        attn_out, _ = self.attn_layer(qkv, qkv, qkv,
                                      key_padding_mask=attn_key_padding)
        x_attn = attn_out  # 源实现：Try no Res
        x_attn = self.ffn(self.norm2(x_attn)) + x_attn
        y = self.norm3(x_attn)[:, 1:, :].reshape(B, T, N, -1)
        return y


class TemporalEncoder(nn.Module):
    """UniTraj Temporal_Encoder 移植：MLP + 前向/反向 Mamba + 衰减门 + 相加。"""

    def __init__(self, model_dim: int, depth: int = 4, p_dropout: float = 0.2,
                 drop_path: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.LeakyReLU(),
            nn.Dropout(p_dropout),
        )
        self.ssm_forw = self._stack(model_dim, depth, drop_path, 0)
        self.ssm_back = self._stack(model_dim, depth, drop_path, depth)
        self.norm_f_forw = RMSNorm(model_dim, eps=1e-5)
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

    def forward(self, x_b_t_n_c: torch.Tensor, gamma_forw, gamma_back):
        """x [B,T,N,C]；gamma_* [B,N,T,C]（衰减门）。返回 [B*N,T,C]。"""
        B, T, N, C = x_b_t_n_c.shape
        x_forw = self.mlp(x_b_t_n_c).permute(0, 2, 1, 3).reshape(B * N, T, C)
        x_flip = torch.flip(x_b_t_n_c, dims=[1])
        x_back = self.mlp(x_flip).permute(0, 2, 1, 3).reshape(B * N, T, C)

        x_forw = x_forw * gamma_forw.reshape(B * N, T, C)
        x_back = x_back * gamma_back.reshape(B * N, T, C)

        y_forw = self._run(x_forw, self.ssm_forw, self.norm_f_forw)
        y_back = self._run(x_back, self.ssm_back, self.norm_f_back)
        return y_forw + torch.flip(y_back, dims=[1])


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
    """GAST-PF 的缺失间隔感知时空编码器。

    入口契约（GAST-PF batch 字段 -> 时间优先）：
        hist_feat [B, N, T, D_in]（GAST-PF 当前 4 维历史特征）
        valid_mask [B, N, T] bool（True=有效观测）
        key_padding [B, N] bool（True=padding actor）
    输出：[B, N, C=embed_dim] —— 内部保留完整 T 序列，末尾按当前模型语义取固定
    最后时间步（非 last-valid；TrajGap 末端缺失语义由 decay 门承担）。
    """

    def __init__(self, input_dim: int = 4, embed_dim: int = 128, num_heads: int = 8,
                 depth: int = 4, operator: str = "max",
                 p_dropout: float = 0.2, drop_path: float = 0.2):
        super().__init__()
        self.spatial = GSMAttnBlock(input_dim, embed_dim, num_heads,
                                    operator=operator, p_dropout=p_dropout)
        self.temporal = TemporalEncoder(embed_dim, depth=depth,
                                        p_dropout=p_dropout, drop_path=drop_path)
        # UniTraj repeats each scalar delta over args.delta_dim=2 before the
        # shared Temporal_Decay linear layer.  Keep that two-channel contract.
        self.temporal_decay = TemporalDecay(embed_dim, delta_dim=2)

    def forward(self, hist_feat: torch.Tensor, valid_mask: torch.Tensor,
                key_padding: torch.Tensor) -> torch.Tensor:
        B, N, T, _ = hist_feat.shape
        # GAST-PF [B,N,T,*] -> internal [B,T,N,*]
        x = hist_feat.permute(0, 2, 1, 3)
        m = valid_mask.permute(0, 2, 1).float()
        kp = key_padding

        x_spa = self.spatial(x, m, kp)                    # [B,T,N,C]
        d_forw, d_back = missing_run_deltas(valid_mask)   # [B,N,T] 各 1 起算
        delta_forw = d_forw.unsqueeze(-1).expand(*d_forw.shape, 2)
        delta_back = d_back.unsqueeze(-1).expand(*d_back.shape, 2)
        g_forw = self.temporal_decay(delta_forw)          # [B,N,T,C]
        g_back = self.temporal_decay(delta_back)
        y = self.temporal(x_spa, g_forw, g_back)          # [B*N,T,C]

        # 当前 GAST-PF 读出语义：固定取最后时间步；padding actor 输出置零。
        last = y[:, -1].reshape(B, N, -1)
        last = last * (~kp).unsqueeze(-1).to(last.dtype)
        return last


# Compatibility alias for checkpoints and external research scripts that still
# import the old class name. New code should use GASTEncoder.
UniTrajPastEncoder = GASTEncoder
