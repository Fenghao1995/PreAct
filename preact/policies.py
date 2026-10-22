from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class FlowMatchingPolicyConfig:
    latent_dim: int
    action_dim: int
    action_horizon: int
    hidden_dim: int = 256
    depth: int = 3
    num_heads: int = 4
    dropout: float = 0.0
    integration_steps: int = 16


@dataclass
class DiffusionPolicyConfig:
    latent_dim: int
    action_dim: int
    action_horizon: int
    hidden_dim: int = 256
    depth: int = 4
    dropout: float = 0.0
    diffusion_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().reshape(-1)
        half = max(self.dim // 2, 1)
        freqs = torch.exp(
            torch.arange(half, device=x.device, dtype=x.dtype)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        emb = x[:, None] * freqs[None]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb[:, : self.dim]


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, hidden_dim: int, condition_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.modulation = nn.Linear(condition_dim, hidden_dim * 2)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(condition).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale[:, None]) + shift[:, None]


class ActionTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.self_norm = AdaptiveLayerNorm(hidden_dim, hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = AdaptiveLayerNorm(hidden_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff_norm = AdaptiveLayerNorm(hidden_dim, hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, time_condition: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.self_norm(x, time_condition)
        y, _ = self.self_attn(normalized, normalized, normalized, need_weights=False)
        x = x + y
        q = self.cross_norm(x, time_condition)
        y, _ = self.cross_attn(q, cond, cond, need_weights=False)
        x = x + y
        return x + self.ff(self.ff_norm(x, time_condition))


class FlowMatchingPolicy(nn.Module):
    """Conditional flow-matching action-chunk policy."""

    def __init__(self, cfg: FlowMatchingPolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.action_proj = nn.Linear(cfg.action_dim, cfg.hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, cfg.action_horizon, cfg.hidden_dim))
        nn.init.normal_(self.pos, std=0.02)
        self.latent_proj = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.time_proj = nn.Sequential(
            SinusoidalEmbedding(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                ActionTransformerBlock(cfg.hidden_dim, cfg.num_heads, cfg.dropout)
                for _ in range(cfg.depth)
            ]
        )
        self.out_norm = nn.LayerNorm(cfg.hidden_dim)
        self.out = nn.Linear(cfg.hidden_dim, cfg.action_dim)

    @property
    def action_horizon(self) -> int:
        return int(self.cfg.action_horizon)

    @property
    def action_dim(self) -> int:
        return int(self.cfg.action_dim)

    def forward(
        self, action: torch.Tensor, latent: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        action = _fit_action_horizon(action.float(), self.action_horizon)
        cond = self.latent_proj(latent.float())
        time_condition = self.time_proj(time)
        cond_token = cond[:, None]
        x = self.action_proj(action) + self.pos[:, : action.shape[1]]
        for block in self.blocks:
            x = block(x, cond_token, time_condition)
        return self.out(self.out_norm(x))

    def loss(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action = _fit_action_horizon(action.float(), self.action_horizon)
        noise = torch.randn_like(action)
        time = torch.rand(action.shape[0], device=action.device, dtype=action.dtype)
        x_t = (1.0 - time[:, None, None]) * action + time[:, None, None] * noise
        target_velocity = noise - action
        pred_velocity = self.forward(x_t, latent, time)
        return F.mse_loss(pred_velocity, target_velocity)

    @torch.no_grad()
    def sample_action_chunks(self, latent: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        self.eval()
        b = latent.shape[0]
        samples = max(int(num_samples), 1)
        latent_flat = latent[:, None].expand(b, samples, latent.shape[-1]).reshape(b * samples, -1)
        action = torch.randn(
            b * samples,
            self.action_horizon,
            self.action_dim,
            device=latent.device,
            dtype=latent.dtype,
        )
        steps = max(int(self.cfg.integration_steps), 1)
        grid = torch.linspace(1.0, 0.0, steps + 1, device=latent.device, dtype=latent.dtype)
        for idx in range(steps):
            s = grid[idx].expand(action.shape[0])
            ds = grid[idx + 1] - grid[idx]
            action = action + ds * self.forward(action, latent_flat, s)
        return action.reshape(b, samples, self.action_horizon, self.action_dim)


class TemporalResidualBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, cond_dim: int, dropout: float):
        super().__init__()
        groups = 8 if output_dim % 8 == 0 else 1
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, output_dim, affine=False)
        self.norm2 = nn.GroupNorm(groups, output_dim, affine=False)
        self.cond1 = nn.Linear(cond_dim, output_dim * 2)
        self.cond2 = nn.Linear(cond_dim, output_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(input_dim, output_dim, kernel_size=1)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        scale, shift = self.cond1(cond).chunk(2, dim=-1)
        y = self.norm1(y) * (1.0 + scale[:, :, None]) + shift[:, :, None]
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.conv2(y)
        scale, shift = self.cond2(cond).chunk(2, dim=-1)
        y = self.norm2(y) * (1.0 + scale[:, :, None]) + shift[:, :, None]
        return self.residual(x) + F.gelu(y)


class DiffusionPolicy(nn.Module):
    """DDPM action-chunk policy with a conditional temporal U-Net."""

    def __init__(self, cfg: DiffusionPolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.latent_proj = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.in_proj = nn.Conv1d(cfg.action_dim, cfg.hidden_dim, kernel_size=1)
        wider = cfg.hidden_dim * 2
        self.down_block = TemporalResidualBlock(
            cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout
        )
        self.downsample = nn.Conv1d(cfg.hidden_dim, wider, kernel_size=4, stride=2, padding=1)
        self.mid_blocks = nn.ModuleList(
            [
                TemporalResidualBlock(wider, wider, cfg.hidden_dim, cfg.dropout)
                for _ in range(cfg.depth)
            ]
        )
        self.upsample = nn.ConvTranspose1d(
            wider, cfg.hidden_dim, kernel_size=4, stride=2, padding=1
        )
        self.up_block = TemporalResidualBlock(
            cfg.hidden_dim * 2, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout
        )
        self.out_norm = nn.GroupNorm(8 if cfg.hidden_dim % 8 == 0 else 1, cfg.hidden_dim)
        self.out = nn.Conv1d(cfg.hidden_dim, cfg.action_dim, kernel_size=1)

        betas = torch.linspace(
            cfg.beta_start, cfg.beta_end, cfg.diffusion_steps, dtype=torch.float32
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        previous = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - previous) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-20))
        self.register_buffer(
            "posterior_mean_coef1", betas * torch.sqrt(previous) / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - previous) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    @property
    def action_horizon(self) -> int:
        return int(self.cfg.action_horizon)

    @property
    def action_dim(self) -> int:
        return int(self.cfg.action_dim)

    def forward(
        self, noisy_action: torch.Tensor, latent: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        noisy_action = _fit_action_horizon(noisy_action.float(), self.action_horizon)
        time = timesteps.float() / max(float(self.cfg.diffusion_steps - 1), 1.0)
        cond = self.latent_proj(latent.float()) + self.time_embed(time)
        x = self.in_proj(noisy_action.transpose(1, 2))
        skip = self.down_block(x, cond)
        x = self.downsample(skip)
        for block in self.mid_blocks:
            x = block(x, cond)
        x = self.upsample(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = self.up_block(torch.cat([x, skip], dim=1), cond)
        x = F.gelu(self.out_norm(x))
        return self.out(x).transpose(1, 2)

    def q_sample(
        self, action: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        sqrt_alpha = _extract(self.sqrt_alphas_cumprod, timesteps, action.shape)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, action.shape)
        return sqrt_alpha * action + sqrt_one_minus * noise

    def loss(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action = _fit_action_horizon(action.float(), self.action_horizon)
        timesteps = torch.randint(
            low=0,
            high=self.cfg.diffusion_steps,
            size=(action.shape[0],),
            device=action.device,
        )
        noise = torch.randn_like(action)
        noisy = self.q_sample(action, timesteps, noise)
        pred = self.forward(noisy, latent, timesteps)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample_action_chunks(self, latent: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        self.eval()
        b = latent.shape[0]
        samples = max(int(num_samples), 1)
        latent_flat = latent[:, None].expand(b, samples, latent.shape[-1]).reshape(b * samples, -1)
        action = torch.randn(
            b * samples,
            self.action_horizon,
            self.action_dim,
            device=latent.device,
            dtype=latent.dtype,
        )
        for step in reversed(range(self.cfg.diffusion_steps)):
            timesteps = torch.full((action.shape[0],), step, device=latent.device, dtype=torch.long)
            eps = self.forward(action, latent_flat, timesteps)
            alpha_bar = _extract(self.alphas_cumprod, timesteps, action.shape)
            predicted_clean = (action - (1.0 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt()
            coef1 = _extract(self.posterior_mean_coef1, timesteps, action.shape)
            coef2 = _extract(self.posterior_mean_coef2, timesteps, action.shape)
            mean = coef1 * predicted_clean + coef2 * action
            if step > 0:
                variance = _extract(self.posterior_variance, timesteps, action.shape)
                action = mean + variance.sqrt() * torch.randn_like(action)
            else:
                action = mean
        return action.reshape(b, samples, self.action_horizon, self.action_dim)


def build_policy(kind: str, **kwargs) -> nn.Module:
    kind = str(kind).lower()
    if kind in {"flow", "fm", "flow_matching"}:
        return FlowMatchingPolicy(_build_dataclass(FlowMatchingPolicyConfig, kwargs))
    if kind in {"diffusion", "ddpm", "dp"}:
        return DiffusionPolicy(_build_dataclass(DiffusionPolicyConfig, kwargs))
    raise ValueError(f"Unknown policy kind: {kind}")


def save_policy(policy: nn.Module, path: str | Path, metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(policy, FlowMatchingPolicy):
        kind = "flow"
        cfg = asdict(policy.cfg)
    elif isinstance(policy, DiffusionPolicy):
        kind = "diffusion"
        cfg = asdict(policy.cfg)
    else:
        raise TypeError(f"Unsupported policy type: {type(policy)!r}")
    torch.save(
        {
            "format": "preact_policy_v2",
            "kind": kind,
            "config": cfg,
            "state_dict": policy.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )


def load_policy(path: str | Path, map_location: str | torch.device = "cpu") -> nn.Module:
    checkpoint = torch.load(Path(path).expanduser(), map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "preact_policy_v2":
        raise ValueError(f"{path} is not a paper-aligned PreAct policy v2 checkpoint.")
    policy = build_policy(checkpoint["kind"], **checkpoint["config"])
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy


def _build_dataclass(cls, values: dict[str, Any]):
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in allowed})


def _extract(
    buffer: torch.Tensor, timesteps: torch.Tensor, target_shape: torch.Size
) -> torch.Tensor:
    out = buffer.gather(0, timesteps.long())
    return out.reshape(-1, *([1] * (len(target_shape) - 1)))


def _fit_action_horizon(action: torch.Tensor, horizon: int) -> torch.Tensor:
    if action.ndim == 2:
        action = action[:, None, :]
    if action.shape[1] == horizon:
        return action
    if action.shape[1] > horizon:
        return action[:, :horizon]
    pad = action[:, -1:].expand(action.shape[0], horizon - action.shape[1], action.shape[-1])
    return torch.cat([action, pad], dim=1)
