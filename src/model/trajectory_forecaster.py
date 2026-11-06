import torch
import torch.nn as nn
from .layers.gast_goal_decoder import GapAwareGoalDecoder
from .layers.gast_encoder import GASTEncoder


# Actor-centric trajectory forecaster.
class TrajectoryForecaster(nn.Module):
    """组装 GAST：GASTEncoder 编码历史，GapAwareGoalDecoder 执行终点引导解码。"""

    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        future_steps: int = 12,
        num_modes: int = 20,
        dt: float = 0.4,
        obs_len: int = 8,
        gast_depth: int = 4,
        gast_gsm_operator: str = "max",
        goal_endpoint_loss_weight: float = 1.0,
        mode_diversity_loss_weight: float = 0.05,
        mode_diversity_margin: float = 0.5,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.dt = dt
        self.obs_len = obs_len
        self.goal_endpoint_loss_weight = float(goal_endpoint_loss_weight)
        self.mode_diversity_loss_weight = float(mode_diversity_loss_weight)
        self.mode_diversity_margin = float(mode_diversity_margin)
        hist_input_dim = 4

        self.neighbor_prediction_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, future_steps * 2)
        )
        gast_encoder_rng_state = torch.random.get_rng_state()

        self.goal_decoder = GapAwareGoalDecoder(
            future_steps=future_steps,
            dim=embed_dim,
            num_modes=num_modes,
            num_heads=num_heads,
            dt=dt,
        )
        self.initialize_weights()

        # Keep encoder initialization independent from the decoder architecture.
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
        )
        torch.random.set_rng_state(rng_state)

    def initialize_weights(self):
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

        B, N = hist_feat.shape[:2]

        x_encoder = self.gast_encoder(
            hist_feat,
            hist_valid_mask,
            ~hist_key_valid_mask,
            positions=data["x_positions"],
        )

        # Neighbor predictions also supervise the encoder representations
        # (handle N=1: no other agents).
        x_others = x_encoder[:, 1:N]
        if x_others.size(1) > 0:
            y_hat_others = self.neighbor_prediction_head(x_others).view(
                B, x_others.size(1), -1, 2
            )
        else:
            y_hat_others = x_encoder.new_zeros((B, 0, self.future_steps, 2))


        # The endpoint-guided decoder uses the last valid observation and
        # the actual forecast gap.
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
