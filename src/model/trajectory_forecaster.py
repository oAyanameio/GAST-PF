import torch
import torch.nn as nn
from .layers.transformer_blocks import Block
from .layers.structured_future_decoder import FutureDistributionDecoder
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
        gast_oer_spatial: bool = False,
        gast_emc_temporal: bool = False,
        gast_ewhr_readout: bool = False,
        gast_temporal_direction: str = "unidirectional",
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.dt = dt
        self.num_actor_types = num_actor_types
        self.obs_len = obs_len
        if encoder_type not in {"m0", "gast"}:
            raise ValueError(f"unsupported encoder_type={encoder_type!r}")
        self.encoder_type = encoder_type
        self.gast_temporal_direction = gast_temporal_direction
        if self.gast_temporal_direction != "unidirectional":
            raise ValueError("GAST-PF supports only unidirectional temporal encoding")
        self.gast_oer_spatial = bool(gast_oer_spatial) and encoder_type == "gast"
        self.gast_emc_temporal = bool(gast_emc_temporal) and encoder_type == "gast"
        self.gast_ewhr_readout = bool(gast_ewhr_readout) and encoder_type == "gast"
        evidence_switches = (
            self.gast_oer_spatial,
            self.gast_emc_temporal,
            self.gast_ewhr_readout,
        )
        if any(evidence_switches) and not all(evidence_switches):
            raise ValueError("UniOER enables OER, EMC, and EWHR together")
        self.gast_method_name = (
            "GAST-PF-UniOER" if all(evidence_switches)
            else ("GAST-PF-Uni" if encoder_type == "gast" else "M0")
        )
        hist_input_dim = 4

        if self.encoder_type == "m0":
            # M0 history encoder. These modules are deliberately absent from
            # the GAST-PF model so a full replacement has no dead parameters.
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

        self.future_time_embedding = nn.Sequential(
            nn.Linear(1, 64), nn.GELU(), nn.Linear(64, embed_dim)
        )

        self.future_decoder = FutureDistributionDecoder(
            future_len=future_steps, dim=embed_dim, num_modes=num_modes
        )

        self.initialize_weights()

        if self.encoder_type == "gast":
            # GAST-PF 缺失间隔感知时空编码器：
            # 在 initialize_weights 之后、RNG save/restore 内构造——同 seed 下
            # 共享模块初始值与 M0 逐位一致（配对初始化纪律），借鉴臂从自身
            # RNG 流起步，不污染主链。GAST-PF 前向中不使用 M0 历史栈参数。
            rng_state = torch.random.get_rng_state()
            self.gast_encoder = GASTEncoder(
                input_dim=hist_input_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
                depth=gast_depth,
                operator=gast_gsm_operator,
                p_dropout=0.2,
                drop_path=drop_path,
                observation_evidence_spatial=self.gast_oer_spatial,
                evidence_modulated_temporal=self.gast_emc_temporal,
                evidence_weighted_readout=self.gast_ewhr_readout,
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

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        """Migrate decoder and head parameter paths from earlier checkpoints."""
        replacements = (
            ("hist_embed_mlp.", "history_input_projection."),
            ("pos_embed.", "actor_context_projection."),
            ("blocks.", "scene_interaction_blocks."),
            ("dense_predictor.", "neighbor_prediction_head."),
            ("time_embedding_mlp.", "future_time_embedding."),
            ("time_decoder.predictor_dense.", "future_decoder.coupled_trajectory_head."),
            ("time_decoder.predictor.", "future_decoder.mode_trajectory_head."),
            ("time_decoder.cross_block_time.", "future_decoder.state_context_layers."),
            ("time_decoder.dense_predict.", "future_decoder.state_forecast_head."),
            ("time_decoder.timequery_norm_f.", "future_decoder.state_sequence_norm."),
            ("time_decoder.timequery_drop_path.", "future_decoder.state_sequence_drop_path."),
            ("time_decoder.timequery_embed_mamba.", "future_decoder.timequery_embed_mamba."),
            ("time_decoder.self_block_mode.", "future_decoder.mode_self_attention_layers."),
            ("time_decoder.cross_block_mode.", "future_decoder.mode_context_layers."),
            ("time_decoder.multi_modal_query_embedding.", "future_decoder.mode_query_embedding."),
            ("time_decoder.self_block_dense.", "future_decoder.coupling_joint_layers."),
            ("time_decoder.cross_block_dense.", "future_decoder.coupling_context_layers."),
            ("time_decoder.self_block_different_mode.", "future_decoder.coupling_mode_interaction_layers."),
            ("time_decoder.dense_embed_mamba.", "future_decoder.coupled_temporal_mamba."),
            ("time_decoder.dense_norm_f.", "future_decoder.coupled_temporal_norm."),
            ("time_decoder.dense_drop_path.", "future_decoder.coupled_temporal_drop_path."),
            ("time_decoder.", "future_decoder."),
        )
        head_replacements = (
            (".gaussian.", ".trajectory_mean_head."),
            (".score.", ".mode_score_head."),
            (".scale.", ".trajectory_scale_head."),
        )
        for key in list(state_dict):
            if not key.startswith(prefix):
                continue
            relative = key[len(prefix):]
            migrated = relative
            for old, new in replacements:
                if migrated.startswith(old):
                    migrated = new + migrated[len(old):]
                    break
            for old, new in head_replacements:
                migrated = migrated.replace(old, new)
            if migrated != relative:
                state_dict[prefix + migrated] = state_dict.pop(key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

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
                positions=data["x_positions"] if self.gast_oer_spatial else None,
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

        ###### Trajectory decoding with decoupled queries ###### 
        new_y_hat = None
        new_pi = None
        dense_predict = None
        mode = None

        # outputs of other agents (handle N=1: no other agents)
        x_others = x_encoder[:, 1:N]
        if x_others.size(1) > 0:
            y_hat_others = self.neighbor_prediction_head(x_others).view(
                B, x_others.size(1), -1, 2
            )
        else:
            y_hat_others = x_encoder.new_zeros((B, 0, self.future_steps, 2))


        # state query initialization
        # dt 参数化：ETH/UCY 与 SDD 均为 0.4s/帧（frame stride=10 @ 2.5Hz）
        time = torch.arange(self.future_steps).long().to(x_encoder.device)
        time = time * self.dt + self.dt
        time = time.unsqueeze(-1)
        mode = self.future_time_embedding(time)
        mode = mode.repeat(x_encoder.size(0), 1, 1)

        # decoder module with decoupled queries
        dense_predict, y_hat, pi, x_mode, new_y_hat, new_pi, mode_dense, scal, scal_new = \
        self.future_decoder(mode, x_encoder, mask=~hist_key_valid_mask)

        ret_dict = {
            "y_hat": y_hat,  # trajectory output from mode query
            "pi": pi,  # probability output from mode query
            "scal": scal,  # output for Laplace loss from mode query

            "dense_predict": dense_predict,  # trajectory output from state query

            "y_hat_others": y_hat_others,  # trajectory of other agents

            "new_y_hat": new_y_hat,  # final trajectory output
            "new_pi": new_pi,  # final probability output
            "scal_new": scal_new,  # final output for Laplace loss
        }

        return ret_dict


# Legacy class name kept for old configs and external scripts.
ModelForecast = TrajectoryForecaster
