from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class RNDConfig:
    input_dim: int
    output_dim: int = 128
    target_hidden_dims: tuple[int, ...] = (1024, 2048, 4096)
    predictor_tail_dims: tuple[int, ...] = (2048, 1024)
    negative_slope: float = 0.01
    score_mode: str = "squared_l2"
    normalize_input: bool = False

    @classmethod
    def from_dict(cls, values: dict) -> RNDConfig:
        values = dict(values)
        for key in ("target_hidden_dims", "predictor_tail_dims"):
            if key in values:
                values[key] = tuple(int(x) for x in values[key])
        return cls(**values)


def _target_network(cfg: RNDConfig) -> nn.Sequential:
    dims = (cfg.input_dim, *cfg.target_hidden_dims, cfg.output_dim)
    layers: list[nn.Module] = []
    for index, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        layers.append(nn.Linear(input_dim, output_dim))
        if index < len(dims) - 2:
            layers.append(nn.LeakyReLU(cfg.negative_slope))
    return nn.Sequential(*layers)


def _predictor_network(cfg: RNDConfig) -> nn.Sequential:
    dims = (cfg.input_dim, *cfg.target_hidden_dims, *cfg.predictor_tail_dims, cfg.output_dim)
    layers: list[nn.Module] = []
    target_depth = len(cfg.target_hidden_dims)
    for index, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        layers.append(nn.Linear(input_dim, output_dim))
        if index < target_depth:
            layers.append(nn.LeakyReLU(cfg.negative_slope))
        elif index < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class RandomNetworkDistillation(nn.Module):
    """DRS-O with the target/predictor MLPs specified in Appendix A.3."""

    def __init__(self, cfg: RNDConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("input_mean", torch.zeros(cfg.input_dim))
        self.register_buffer("input_std", torch.ones(cfg.input_dim))
        self.target = _target_network(cfg)
        self.predictor = _predictor_network(cfg)
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.target.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.target.eval()
        return self

    def set_normalizer(self, values: torch.Tensor) -> None:
        values = values.reshape(values.shape[0], -1).detach().float()
        self.input_mean.copy_(values.mean(dim=0))
        self.input_std.copy_(values.std(dim=0, unbiased=False).clamp_min(1e-6))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        values = latent.reshape(latent.shape[0], -1).float()
        if self.cfg.normalize_input:
            values = (values - self.input_mean) / self.input_std.clamp_min(1e-6)
        with torch.no_grad():
            target = self.target(values)
        prediction = self.predictor(values)
        squared_l2 = (prediction - target).square().sum(dim=-1)
        if self.cfg.score_mode in {"squared_l2", "l2_squared"}:
            return squared_l2
        if self.cfg.score_mode == "mse":
            return squared_l2 / max(self.cfg.output_dim, 1)
        if self.cfg.score_mode == "l2":
            return squared_l2.sqrt()
        raise ValueError(f"Unknown RND score_mode={self.cfg.score_mode!r}")

    def train_loss(self, latent: torch.Tensor) -> torch.Tensor:
        return self(latent.detach()).mean()


def save_rnd(
    model: RandomNetworkDistillation,
    path: str | Path,
    *,
    metadata: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "preact_rnd_v2",
            "config": asdict(model.cfg),
            "state_dict": model.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )


def load_rnd(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> RandomNetworkDistillation:
    checkpoint = torch.load(Path(path).expanduser(), map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "preact_rnd_v2":
        raise ValueError(f"{path} is not a paper-aligned PreAct RND v2 checkpoint.")
    model = RandomNetworkDistillation(RNDConfig.from_dict(checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
