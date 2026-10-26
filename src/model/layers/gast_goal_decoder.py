import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_blocks import Cross_Block


class GapAwareGoalDecoder(nn.Module):
    def __init__(
        self,
        future_steps: int,
        dim: int,
        num_modes: int,
        num_heads: int = 8,
        dt: float = 0.4,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.dt = dt
        self.mode_embedding = nn.Embedding(num_modes, dim)
        self.anchor_state_encoder = nn.Sequential(
            nn.Linear(7, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.mode_context_layers = nn.ModuleList(
            Cross_Block(
                dim=dim,
                num_heads=num_heads,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.0,
            )
            for _ in range(2)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.endpoint_head = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2)
        )
        self.endpoint_embedding = nn.Sequential(
            nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.trajectory_head = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2)
        )
        self.trajectory_scale_head = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2)
        )
        self.mode_score_head = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 1)
        )

    def _anchor_state(self, data: dict):
        positions = data["x_positions"][:, 0]
        valid = data["x_valid_mask"][:, 0].bool()
        batch_size, history_steps = valid.shape
        indices = torch.arange(history_steps, device=valid.device).expand(
            batch_size, history_steps
        )

        last_index = torch.where(valid, indices, -1).amax(dim=-1).clamp(min=0)
        previous_index = torch.where(
            valid & (indices < last_index[:, None]), indices, -1
        ).amax(dim=-1)
        anchor = positions[
            torch.arange(batch_size, device=valid.device), last_index
        ]
        previous = positions[
            torch.arange(batch_size, device=valid.device), previous_index.clamp(min=0)
        ]
        elapsed = (last_index - previous_index).clamp(min=1).to(positions.dtype)
        velocity = (anchor - previous) / elapsed[:, None]
        velocity = velocity * (previous_index >= 0).to(positions.dtype)[:, None]

        gap_steps = (history_steps - last_index).clamp(min=1)
        valid_ratio = valid.to(positions.dtype).mean(dim=-1)
        terminal_missing = (~valid[:, -1]).to(positions.dtype)
        state = torch.cat(
            [
                anchor,
                velocity,
                gap_steps.to(positions.dtype).div(history_steps)[:, None],
                valid_ratio[:, None],
                terminal_missing[:, None],
            ],
            dim=-1,
        )
        return anchor, state, gap_steps

    def forward(
        self,
        encoding: torch.Tensor,
        data: dict,
        actor_padding_mask: torch.Tensor | None = None,
    ) -> dict:
        batch_size = encoding.shape[0]
        anchor, anchor_state, gap_steps = self._anchor_state(data)
        state_embedding = self.anchor_state_encoder(anchor_state)

        mode_ids = torch.arange(self.num_modes, device=encoding.device)
        modes = (
            encoding[:, 0, None, :]
            + state_embedding[:, None, :]
            + self.mode_embedding(mode_ids)[None, :, :]
        )
        for layer in self.mode_context_layers:
            modes = layer(
                modes,
                encoding,
                key_padding_mask=actor_padding_mask,
            )

        goal_delta = self.endpoint_head(modes)
        endpoints = anchor[:, None, :] + goal_delta

        future_index = torch.arange(self.future_steps, device=encoding.device)
        future_steps = gap_steps[:, None] + future_index[None, :]
        future_time_seconds = future_steps.to(encoding.dtype) * self.dt
        time_features = self.time_embedding(future_time_seconds.unsqueeze(-1))
        endpoint_features = self.endpoint_embedding(goal_delta)
        trajectory_queries = (
            modes[:, :, None, :]
            + endpoint_features[:, :, None, :]
            + time_features[:, None, :, :]
        )

        relative_time = future_steps.to(encoding.dtype)
        progress = relative_time / relative_time[:, -1:].clamp(min=1.0)
        progress = progress[:, None, :, None]
        linear_path = anchor[:, None, None, :] + progress * goal_delta[:, :, None, :]
        residual = self.trajectory_head(trajectory_queries)
        trajectories = linear_path + residual * (1.0 - progress)

        scale = F.elu(self.trajectory_scale_head(trajectory_queries)) + 1.0001
        trajectory_summary = trajectory_queries.mean(dim=2)
        score_features = torch.cat(
            [modes, endpoint_features, trajectory_summary], dim=-1
        )
        scores = self.mode_score_head(score_features).squeeze(-1)

        return {
            "trajectories": trajectories,
            "scores": scores,
            "scales": scale,
            "endpoints": endpoints,
            "anchor": anchor,
            "goal_delta": goal_delta,
            "future_time_seconds": future_time_seconds,
        }
