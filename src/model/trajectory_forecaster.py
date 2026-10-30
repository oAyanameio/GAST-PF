import torch
import torch.nn as nn
from .layers.transformer_blocks import Block
from .layers.gast_goal_decoder import GapAwareGoalDecoder
from .layers.gast_encoder import GASTEncoder
from .layers.mamba.vim_mamba import create_block
from timm.models.layers import DropPath
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# Actor-centric trajectory forecaster.
class TrajectoryForecaster(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        future_steps: int = 12,
        num_actor_types: int = 1,
        num_modes: int = 20,
        bimamba: bool = False,
        dt: float = 0.4,
        obs_len: int = 8,
        encoder_type: str = "m0",
        gast_depth: int = 4,
        gast_gsm_operator: str = "max",
        gast_gere_spatial: bool = False,
        gast_ecta: bool = False,
        gast_temporal_direction: str = "unidirectional",
        goal_endpoint_loss_weight: float = 1.0,
        mode_diversity_loss_weight: float = 0.05,
        mode_diversity_margin: float = 0.5,
        classification_target_mode=None,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.dt = dt
        self.num_actor_types = num_actor_types
        self.obs_len = obs_len
        self.goal_endpoint_loss_weight = float(goal_endpoint_loss_weight)
        self.mode_diversity_loss_weight = float(mode_diversity_loss_weight)
        self.mode_diversity_margin = float(mode_diversity_margin)
        if classification_target_mode is None:
            classification_target_mode = (
                "trajectory" if encoder_type == "gast" else "joint"
            )
        if classification_target_mode not in {"joint", "trajectory"}:
            raise ValueError(
                "classification_target_mode must be 'joint' or 'trajectory', "
                f"got {classification_target_mode!r}"
            )
        self.classification_target_mode = classification_target_mode
        if encoder_type not in {"m0", "gast"}:
            raise ValueError(f"unsupported encoder_type={encoder_type!r}")
        self.encoder_type = encoder_type
        self.gast_temporal_direction = gast_temporal_direction
        if self.gast_temporal_direction != "unidirectional":
            raise ValueError("GAST supports only unidirectional temporal encoding")
        self.gast_gere_spatial = bool(gast_gere_spatial) and encoder_type == "gast"
        self.gast_ecta = bool(gast_ecta) and encoder_type == "gast"
        if self.gast_gere_spatial != self.gast_ecta:
            raise ValueError("GAST enables GERE and ECTA together")
        self.gast_method_name = "GAST" if encoder_type == "gast" else "M0"
        hist_input_dim = 4

        if self.encoder_type == "m0":
            # M0 history encoder. These modules are deliberately absent from
            # GAST so a full replacement has no dead parameters.
            self.history_input_projection = nn.Sequential(
                nn.Linear(hist_input_dim, 64),
                nn.GELU(),
                nn.Linear(64, embed_dim),
            )
            self.hist_embed_mamba = nn.ModuleList(
                [
                    create_block(
                        d_model=embed_dim,
                        layer_idx=i,
                        drop_path=0.2,
                        bimamba=bimamba,
                        rms_norm=True,
                    )
                    for i in range(4)
                ]
            )
            self.norm_f = RMSNorm(embed_dim, eps=1e-5)
            self.drop_path = DropPath(drop_path)
            self.actor_context_projection = nn.Sequential(
                nn.Linear(4, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.scene_interaction_blocks = nn.ModuleList(
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=0.2,
                )
                for _ in range(5)
            )
            self.norm = nn.LayerNorm(embed_dim)
            self.actor_type_embed = nn.Parameter(torch.Tensor(num_actor_types, embed_dim))

        self.neighbor_prediction_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, future_steps * 2)
        )
        gast_encoder_rng_state = (
            torch.random.get_rng_state() if self.encoder_type == "gast" else None
        )

        self.goal_decoder = GapAwareGoalDecoder(
            future_steps=future_steps,
            dim=embed_dim,
            num_modes=num_modes,
            num_heads=num_heads,
            dt=dt,
        )
        self.initialize_weights()

        if self.encoder_type == "gast":
            # Isolate encoder initialization from decoder architecture so GAST
            # Keep encoder initialization independent from the decoder choice.
            rng_state = torch.random.get_rng_state()
            torch.random.set_rng_state(gast_encoder_rng_state)
            self.gast_encoder = GASTEncoder(
                input_dim=hist_input_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
                depth=gast_depth,
                operator=gast_gsm_operator,
                p_dropout=0.2,
                drop_path=drop_path,
                geometry_evidence_relational_encoding=self.gast_gere_spatial,
                evidence_conditioned_causal_temporal_aggregation=self.gast_ecta,
            )
            torch.random.set_rng_state(rng_state)

    def initialize_weights(self):
        if hasattr(self, "actor_type_embed"):
            nn.init.normal_(self.actor_type_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("net.") :]: v for k, v in ckpt.items() if k.startswith("net.")
        }
        incompatible = self.load_state_dict(state_dict=state_dict, strict=False)
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if missing or unexpected:
            raise RuntimeError(
                "checkpoint/model mismatch: "
                f"missing={sorted(missing)[:10]} "
                f"unexpected={sorted(unexpected)[:10]}"
            )
        return incompatible


    def forward(self, data):
        hist_valid_mask = data["x_valid_mask"]
        hist_key_valid_mask = data["x_key_valid_mask"]

        hist_feat = torch.cat([
            data["x_positions_diff"],
            data["x_velocity_diff"][..., None],
            hist_valid_mask[..., None],
        ], dim=-1)

        B, N, L, D = hist_feat.shape
        hist_key_valid_flat = hist_key_valid_mask.view(B * N)

        if self.encoder_type == "gast":
            x_encoder = self.gast_encoder(
                hist_feat,
                hist_valid_mask,
                ~hist_key_valid_mask,
                positions=(
                    data["x_positions"]
                    if self.gast_gere_spatial
                    else None
                ),
            )
        else:
            # unidirectional mamba（M0 原版路径）
            hist_feat = hist_feat.view(B * N, L, D)
            actor_feat = self.history_input_projection(
                hist_feat[hist_key_valid_flat].contiguous()
            )
            residual = None
            for blk_mamba in self.hist_embed_mamba:
                actor_feat, residual = blk_mamba(actor_feat, residual)
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            actor_feat = fused_add_norm_fn(
                self.drop_path(actor_feat),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=True
            )

            actor_feat = actor_feat[:, -1]
            actor_feat_tmp = torch.zeros(
                B * N, actor_feat.shape[-1], dtype=actor_feat.dtype, device=actor_feat.device
            )
            actor_feat_tmp[hist_key_valid_flat] = actor_feat
            actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1])
            # M0-only context and scene interaction path.
            x_centers = data["x_centers"]
            if "x_last_valid_angle" in data:
                angles = data["x_last_valid_angle"]
            else:
                angles = data["x_angles"][:, :, -1]
            x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
            pos_feat = torch.cat([x_centers, x_angles], dim=-1)
            pos_embed = self.actor_context_projection(pos_feat)
            actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2].long()]
            actor_feat = actor_feat + actor_type_embed
            x_encoder = actor_feat + pos_embed
            for blk in self.scene_interaction_blocks:
                x_encoder = blk(x_encoder, key_padding_mask=~hist_key_valid_mask)
            x_encoder = self.norm(x_encoder)

        # Neighbor predictions also supervise the encoder representations
        # (handle N=1: no other agents).
        x_others = x_encoder[:, 1:N]
        if x_others.size(1) > 0:
            y_hat_others = self.neighbor_prediction_head(x_others).view(
                B, x_others.size(1), -1, 2
            )
        else:
            y_hat_others = x_encoder.new_zeros((B, 0, self.future_steps, 2))


        # GAST Gap Aware Goal Decoder conditions on the last valid observation
        # and the actual forecast gap.
        goal_out = self.goal_decoder(
            x_encoder,
            data,
            actor_padding_mask=~hist_key_valid_mask,
        )
        return {
            "y_hat": goal_out["trajectories"],
            "pi": goal_out["scores"],
            "scal": goal_out["scales"],
            "dense_predict": None,
            "y_hat_others": y_hat_others,
            "new_y_hat": goal_out["trajectories"],
            "new_pi": goal_out["scores"],
            "scal_new": goal_out["scales"],
            "goal_endpoints": goal_out["endpoints"],
            "goal_anchor": goal_out["anchor"],
            "goal_delta": goal_out["goal_delta"],
            "goal_future_time_seconds": goal_out["future_time_seconds"],
        }


