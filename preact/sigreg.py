from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SIGRegConfig:
    knots: int = 17
    num_projections: int = 1024


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian regularizer used in LeWM-style training."""

    def __init__(self, cfg: SIGRegConfig | None = None):
        super().__init__()
        cfg = cfg or SIGRegConfig()
        self.num_projections = int(cfg.num_projections)
        t = torch.linspace(0, 3, int(cfg.knots), dtype=torch.float32)
        dt = 3.0 / max(int(cfg.knots) - 1, 1)
        weights = torch.full((int(cfg.knots),), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim == 2:
            embeddings = embeddings[None]
        if embeddings.ndim != 3:
            raise ValueError(f"SIGReg expects (T,B,D) or (B,D), got {tuple(embeddings.shape)}")
        z = embeddings.float()
        directions = torch.randn(z.shape[-1], self.num_projections, device=z.device, dtype=z.dtype)
        directions = directions / directions.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8)
        projected = torch.matmul(z, directions).unsqueeze(-1) * self.t.to(z.dtype)
        err = (projected.cos().mean(dim=1) - self.phi.to(z.dtype)).square()
        err = err + projected.sin().mean(dim=1).square()
        statistic = torch.matmul(err, self.weights.to(z.dtype)) * z.shape[1]
        return statistic.mean()
