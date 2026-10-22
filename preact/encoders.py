from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class EncoderConfig:
    type: str = "auto"
    input_shape: tuple[int, ...] = (128,)
    obs_history: int = 1
    latent_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0

    @classmethod
    def from_values(
        cls,
        *,
        input_shape: Sequence[int],
        obs_history: int,
        latent_dim: int,
        encoder_type: str = "auto",
        **kwargs,
    ) -> EncoderConfig:
        return cls(
            type=encoder_type,
            input_shape=tuple(int(x) for x in input_shape),
            obs_history=int(obs_history),
            latent_dim=int(latent_dim),
            **kwargs,
        )


class IdentityEncoder(nn.Module):
    """Use precomputed embeddings as the latent state.

    If an observation history is provided as ``(B,T,D)``, the last embedding is
    used. This matches the release path where upstream visual policies export
    policy-aligned embeddings.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = int(latent_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float()
        if x.ndim == 1:
            x = x[None]
        if x.ndim == 3:
            x = x[:, -1]
        elif x.ndim > 3:
            x = x.reshape(x.shape[0], -1)
        if x.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {x.shape[-1]}")
        return x


class VectorEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        input_dim = int(np.prod(cfg.input_shape) * cfg.obs_history)
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(max(cfg.num_layers - 1, 1)):
            layers.extend(
                [
                    nn.Linear(dim, cfg.hidden_dim),
                    nn.LayerNorm(cfg.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                ]
            )
            dim = cfg.hidden_dim
        layers.append(nn.Linear(dim, cfg.latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float()
        if x.ndim == 1:
            x = x[None]
        return self.net(x.reshape(x.shape[0], -1))


class ImageEncoder(nn.Module):
    """Small release-safe CNN for image observations.

    This is not intended to replace the d3il or Diffusion Policy visual
    backbone. It provides a functional default for synthetic dry runs and
    small custom datasets.
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        if len(cfg.input_shape) != 3:
            raise ValueError(f"Image encoder expects input_shape=(C,H,W), got {cfg.input_shape}")
        channels = int(cfg.input_shape[0]) * int(cfg.obs_history)
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, cfg.latent_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float()
        if x.ndim == 4:
            x = x[:, None]
        if x.ndim != 5:
            raise ValueError(f"Image observations must be (B,T,C,H,W), got {tuple(x.shape)}")
        b, t, c, h, w = x.shape
        x = x.reshape(b, t * c, h, w)
        if x.max() > 2.0:
            x = x / 255.0
        return self.head(self.cnn(x))


def build_encoder(cfg: EncoderConfig) -> nn.Module:
    encoder_type = cfg.type.lower()
    if encoder_type == "auto":
        if (
            len(cfg.input_shape) == 1
            and int(cfg.input_shape[-1]) == int(cfg.latent_dim)
            and int(cfg.obs_history) == 1
        ):
            encoder_type = "identity"
        elif len(cfg.input_shape) == 3:
            encoder_type = "image"
        else:
            encoder_type = "vector"
    if encoder_type in {"identity", "embedding"}:
        return IdentityEncoder(cfg.latent_dim)
    if encoder_type in {"vector", "mlp"}:
        return VectorEncoder(cfg)
    if encoder_type in {"image", "cnn"}:
        return ImageEncoder(cfg)
    raise ValueError(f"Unknown encoder type: {cfg.type}")


def encoder_config_to_dict(cfg: EncoderConfig) -> dict:
    data = asdict(cfg)
    data["input_shape"] = list(cfg.input_shape)
    return data


def encoder_config_from_dict(data: dict) -> EncoderConfig:
    data = dict(data)
    if "input_shape" in data:
        data["input_shape"] = tuple(data["input_shape"])
    return EncoderConfig(**data)
