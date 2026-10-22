from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .encoders import EncoderConfig, build_encoder, encoder_config_from_dict, encoder_config_to_dict


@dataclass
class LatentWorldModelConfig:
    obs_shape: tuple[int, ...] = (128,)
    obs_history: int = 1
    latent_dim: int = 128
    action_dim: int = 2
    action_horizon: int = 8
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.0
    residual: bool = True
    predictor_kind: str = "transformer"
    encoder: EncoderConfig | None = None

    @classmethod
    def from_task(
        cls,
        *,
        obs_shape: Sequence[int],
        obs_history: int,
        latent_dim: int,
        action_dim: int,
        action_horizon: int,
        encoder_type: str = "auto",
        **kwargs,
    ) -> LatentWorldModelConfig:
        encoder = EncoderConfig.from_values(
            input_shape=obs_shape,
            obs_history=obs_history,
            latent_dim=latent_dim,
            encoder_type=encoder_type,
        )
        return cls(
            obs_shape=tuple(int(x) for x in obs_shape),
            obs_history=int(obs_history),
            latent_dim=int(latent_dim),
            action_dim=int(action_dim),
            action_horizon=int(action_horizon),
            encoder=encoder,
            **kwargs,
        )


class AdaptiveLayerNorm(nn.Module):
    """Layer normalization modulated by a single-step action embedding."""

    def __init__(self, hidden_dim: int, condition_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.modulation = nn.Linear(condition_dim, hidden_dim * 2)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(condition).chunk(2, dim=-1)
        while scale.ndim < x.ndim:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return self.norm(x) * (1.0 + scale) + shift


class ActionResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, dropout: float):
        super().__init__()
        self.norm = AdaptiveLayerNorm(hidden_dim, action_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor, action_embedding: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x, action_embedding))


class ActionConditionedMLPPredictor(nn.Module):
    """AdaLN residual MLP used with the DDPM temporal U-Net policy."""

    def __init__(self, cfg: LatentWorldModelConfig):
        super().__init__()
        self.latent_in = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        self.action_in = nn.Sequential(
            nn.Linear(cfg.action_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                ActionResidualBlock(cfg.hidden_dim, cfg.hidden_dim, cfg.dropout)
                for _ in range(cfg.num_layers)
            ]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim), nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        )
        self.residual = bool(cfg.residual)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action_embedding = self.action_in(action)
        hidden = self.latent_in(latent)
        for block in self.blocks:
            hidden = block(hidden, action_embedding)
        prediction = self.out(hidden)
        return latent + prediction if self.residual else prediction


class ActionTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm1 = AdaptiveLayerNorm(hidden_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = AdaptiveLayerNorm(hidden_dim, hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, action_embedding: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(tokens, action_embedding)
        update, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + update
        return tokens + self.ff(self.norm2(tokens, action_embedding))


class ActionConditionedTransformerPredictor(nn.Module):
    """State/action-token Transformer used with the ACT flow policy."""

    def __init__(self, cfg: LatentWorldModelConfig):
        super().__init__()
        self.state_in = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        self.action_in = nn.Sequential(
            nn.Linear(cfg.action_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.token_type = nn.Parameter(torch.zeros(1, 2, cfg.hidden_dim))
        nn.init.normal_(self.token_type, std=0.02)
        self.blocks = nn.ModuleList(
            [
                ActionTransformerBlock(cfg.hidden_dim, cfg.num_heads, cfg.dropout)
                for _ in range(cfg.num_layers)
            ]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim), nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        )
        self.residual = bool(cfg.residual)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action_embedding = self.action_in(action)
        tokens = torch.stack([self.state_in(latent), action_embedding], dim=1) + self.token_type
        for block in self.blocks:
            tokens = block(tokens, action_embedding)
        prediction = self.out(tokens[:, 0])
        return latent + prediction if self.residual else prediction


class ActionConditionedPredictor(nn.Module):
    """Paper-aligned one-step predictor ``pred_phi(z_t, a_t)``."""

    def __init__(self, cfg: LatentWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("action_mean", torch.zeros(cfg.action_dim))
        self.register_buffer("action_std", torch.ones(cfg.action_dim))
        kind = cfg.predictor_kind.lower()
        if kind in {"transformer", "act"}:
            self.model = ActionConditionedTransformerPredictor(cfg)
        elif kind in {"mlp", "unet"}:
            self.model = ActionConditionedMLPPredictor(cfg)
        else:
            raise ValueError(f"Unknown predictor_kind={cfg.predictor_kind!r}")

    def set_action_normalizer(self, actions: torch.Tensor) -> None:
        flat = actions.detach().float().reshape(-1, self.cfg.action_dim)
        self.action_mean.copy_(flat.mean(dim=0))
        self.action_std.copy_(flat.std(dim=0, unbiased=False).clamp_min(1e-6))

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.ndim == 3 and action.shape[-2] == 1:
            action = action[:, 0]
        if action.ndim != 2 or action.shape[-1] != self.cfg.action_dim:
            raise ValueError(
                "LeWM predicts one transition from a single action; expected "
                f"(B,{self.cfg.action_dim}), got {tuple(action.shape)}"
            )
        normalized_action = (action.float() - self.action_mean) / self.action_std.clamp_min(1e-6)
        return self.model(latent.float(), normalized_action)


class LatentWorldModel(nn.Module):
    """Shared encoder and one-step action-conditioned latent predictor."""

    def __init__(self, cfg: LatentWorldModelConfig):
        super().__init__()
        if cfg.encoder is None:
            cfg.encoder = EncoderConfig.from_values(
                input_shape=cfg.obs_shape,
                obs_history=cfg.obs_history,
                latent_dim=cfg.latent_dim,
            )
        self.cfg = cfg
        self.encoder = build_encoder(cfg.encoder)
        self.predictor = ActionConditionedPredictor(cfg)

    @property
    def latent_dim(self) -> int:
        return int(self.cfg.latent_dim)

    @property
    def action_dim(self) -> int:
        return int(self.cfg.action_dim)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def predict_next_latent(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predictor(latent, action)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predict_next_latent(self.encode(obs), action)

    def loss(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        target = self.encode(next_obs)
        prediction = self(obs, action)
        return F.mse_loss(prediction, target)

    @torch.no_grad()
    def rollout_from_obs(self, obs: torch.Tensor, action_candidates: torch.Tensor) -> torch.Tensor:
        return self.rollout_from_latent(self.encode(obs), action_candidates)

    def rollout_from_latent(
        self, latent: torch.Tensor, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Recursively imagine one latent per action in every candidate chunk."""
        if action_candidates.ndim == 3:
            action_candidates = action_candidates[:, None]
        if action_candidates.ndim != 4:
            raise ValueError(
                "action_candidates must be (B,S,H,Da) or (B,H,Da), got "
                f"{tuple(action_candidates.shape)}"
            )
        batch_size, num_samples, horizon, action_dim = action_candidates.shape
        if action_dim != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {action_dim}")
        if latent.ndim != 2 or latent.shape[0] != batch_size:
            raise ValueError(f"latent must be (B,D), got {tuple(latent.shape)}")

        state = (
            latent[:, None]
            .expand(batch_size, num_samples, latent.shape[-1])
            .reshape(batch_size * num_samples, latent.shape[-1])
        )
        actions = action_candidates.reshape(batch_size * num_samples, horizon, action_dim)
        predictions = []
        for step in range(horizon):
            state = self.predict_next_latent(state, actions[:, step])
            predictions.append(state.reshape(batch_size, num_samples, -1))
        return torch.stack(predictions, dim=2)

    def set_action_normalizer(self, actions: torch.Tensor) -> None:
        self.predictor.set_action_normalizer(actions)


def _config_to_dict(cfg: LatentWorldModelConfig) -> dict:
    data = asdict(cfg)
    data["obs_shape"] = list(cfg.obs_shape)
    if cfg.encoder is not None:
        data["encoder"] = encoder_config_to_dict(cfg.encoder)
    return data


def _config_from_dict(data: dict) -> LatentWorldModelConfig:
    data = dict(data)
    data["obs_shape"] = tuple(data.get("obs_shape", (128,)))
    if isinstance(data.get("encoder"), dict):
        data["encoder"] = encoder_config_from_dict(data["encoder"])
    return LatentWorldModelConfig(**data)


def save_world_model(
    model: LatentWorldModel, path: str | Path, metadata: dict | None = None
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "preact_latent_world_model_v2",
            "config": _config_to_dict(model.cfg),
            "state_dict": model.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )


def load_world_model(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> LatentWorldModel:
    checkpoint = torch.load(Path(path).expanduser(), map_location=map_location, weights_only=False)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format") != "preact_latent_world_model_v2"
    ):
        raise ValueError(
            f"{path} is not a v2 PreAct checkpoint. Retrain Stage I: v1 used chunk-conditioned "
            "prediction and is intentionally not loaded by the paper-aligned implementation."
        )
    model = LatentWorldModel(_config_from_dict(checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
