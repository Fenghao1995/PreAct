from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .rnd import RandomNetworkDistillation


@dataclass
class ActionEntropyConfig:
    """Action-chunk entropy settings from Appendix A.2.

    ``cell_size_factor`` is the paper's alpha. The paper does not report its
    numerical value, so it remains an explicit experiment parameter.
    """

    mode: str = "ace"
    cell_size_factor: float = 0.1
    action_range: tuple[float, ...] | None = None
    latent_batch_size: int = 8
    eps: float = 1e-8

    @classmethod
    def from_dict(cls, values: dict | None) -> ActionEntropyConfig:
        values = dict(values or {})
        if values.get("action_range") is not None:
            values["action_range"] = tuple(float(x) for x in values["action_range"])
        return cls(**values)


@dataclass
class Thresholds:
    obs: list[float]
    action: list[float]
    obs_scale: list[float]
    action_scale: list[float]
    quantile: float = 0.9
    style: str = "time_varying"
    action_range: list[float] | None = None
    cell_size_factor: float | None = None

    def obs_at(self, step_index: int) -> float:
        return _value_at(self.obs, step_index)

    def action_at(self, step_index: int) -> float:
        return _value_at(self.action, step_index)

    def obs_scale_at(self, step_index: int) -> float:
        return max(_value_at(self.obs_scale, step_index), 1e-6)

    def action_scale_at(self, step_index: int) -> float:
        return max(_value_at(self.action_scale, step_index), 1e-6)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def scalar(
        cls,
        *,
        obs: float,
        action: float,
        obs_scale: float,
        action_scale: float,
        quantile: float = 0.9,
        action_range: Sequence[float] | None = None,
        cell_size_factor: float | None = None,
    ) -> Thresholds:
        return cls(
            obs=[float(obs)],
            action=[float(action)],
            obs_scale=[float(obs_scale)],
            action_scale=[float(action_scale)],
            quantile=float(quantile),
            style="scalar",
            action_range=None if action_range is None else [float(x) for x in action_range],
            cell_size_factor=cell_size_factor,
        )

    @classmethod
    def from_dict(cls, payload: dict) -> Thresholds:
        return cls(
            obs=[float(x) for x in payload["obs"]],
            action=[float(x) for x in payload["action"]],
            obs_scale=[float(x) for x in payload.get("obs_scale", [1.0])],
            action_scale=[float(x) for x in payload.get("action_scale", [1.0])],
            quantile=float(payload.get("quantile", 0.9)),
            style=str(payload.get("style", "time_varying")),
            action_range=(
                None
                if payload.get("action_range") is None
                else [float(x) for x in payload["action_range"]]
            ),
            cell_size_factor=(
                None
                if payload.get("cell_size_factor") is None
                else float(payload["cell_size_factor"])
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> Thresholds:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


class ObservationDeviationScorer:
    def __init__(self, rnd: RandomNetworkDistillation, reduce: str = "sum"):
        self.rnd = rnd
        self.reduce = reduce

    def to(self, device: torch.device | str) -> ObservationDeviationScorer:
        self.rnd.to(device)
        return self

    @torch.no_grad()
    def score(self, future_latents: torch.Tensor) -> torch.Tensor:
        if future_latents.ndim == 3:
            future_latents = future_latents[:, None]
        if future_latents.ndim != 4:
            raise ValueError(f"future_latents must be (B,S,T,D), got {tuple(future_latents.shape)}")
        batch, samples, horizon, latent_dim = future_latents.shape
        model_device = next(self.rnd.parameters()).device
        flat = future_latents.reshape(batch * samples * horizon, latent_dim).to(model_device)
        scores = self.rnd(flat).reshape(batch, samples, horizon).to(future_latents.device)
        return _reduce_horizon(scores, self.reduce)


class ActionEntropyScorer:
    """DRS-A: timestep-wise histogram entropy summed over an action chunk."""

    def __init__(self, cfg: ActionEntropyConfig | None = None):
        self.cfg = cfg or ActionEntropyConfig()

    @property
    def action_range(self) -> tuple[float, ...] | None:
        return self.cfg.action_range

    def fit(self, calibration_actions: torch.Tensor) -> ActionEntropyScorer:
        if calibration_actions.ndim == 2:
            calibration_actions = calibration_actions[:, None]
        if calibration_actions.ndim != 3:
            raise ValueError(
                "calibration_actions must be (N,H,Da) or (N,Da), got "
                f"{tuple(calibration_actions.shape)}"
            )
        values = calibration_actions.detach().float()
        flat = values.reshape(-1, values.shape[-1])
        ranges = (flat.max(dim=0).values - flat.min(dim=0).values).clamp_min(self.cfg.eps)
        self.cfg.action_range = tuple(float(x) for x in ranges.cpu().tolist())
        return self

    def configure_from_thresholds(self, thresholds: Thresholds) -> None:
        if thresholds.action_range is not None:
            self.cfg.action_range = tuple(thresholds.action_range)
        if thresholds.cell_size_factor is not None:
            configured = float(self.cfg.cell_size_factor)
            stored = float(thresholds.cell_size_factor)
            if abs(configured - stored) > 1e-12:
                raise ValueError(
                    "ACE cell_size_factor differs from the calibrated thresholds: "
                    f"config={configured}, thresholds={stored}"
                )

    @torch.no_grad()
    def score_action_set(self, action_samples: torch.Tensor) -> torch.Tensor:
        if action_samples.ndim == 3:
            action_samples = action_samples[:, None]
        if action_samples.ndim != 4:
            raise ValueError(
                f"action_samples must be (B,S,H,Da), got {tuple(action_samples.shape)}"
            )
        if action_samples.shape[1] <= 1:
            return torch.zeros(
                action_samples.shape[0], device=action_samples.device, dtype=action_samples.dtype
            )
        if self.cfg.mode == "logvar":
            variance = action_samples.float().var(dim=1, unbiased=False)
            return torch.log1p(variance.clamp_min(0.0)).sum(dim=(-1, -2))
        if self.cfg.mode != "ace":
            raise ValueError(f"Unknown action entropy mode={self.cfg.mode!r}")
        return self._ace(action_samples)

    def score_candidates_by_future_policy(
        self,
        *,
        policy,
        future_latents: torch.Tensor,
        num_samples: int,
        latent_index: int | None = None,
        reduce: str = "sum",
    ) -> torch.Tensor:
        if future_latents.ndim != 4:
            raise ValueError(f"future_latents must be (B,S,T,D), got {tuple(future_latents.shape)}")
        batch, candidates, horizon, latent_dim = future_latents.shape
        if latent_index is not None:
            index = min(max(int(latent_index), 0), horizon - 1)
            flat_latent = future_latents[:, :, index].reshape(batch * candidates, latent_dim)
            scores = self._score_policy_batches(policy, flat_latent, num_samples)
            return scores.reshape(batch, candidates).to(future_latents.device)

        flat_latent = future_latents.reshape(batch * candidates * horizon, latent_dim)
        scores = self._score_policy_batches(policy, flat_latent, num_samples).reshape(
            batch, candidates, horizon
        )
        return _reduce_horizon(scores, reduce).to(future_latents.device)

    def _score_policy_batches(
        self, policy, flat_latent: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        latent_batch_size = max(int(self.cfg.latent_batch_size), 1)
        scores = []
        for start in range(0, flat_latent.shape[0], latent_batch_size):
            latent_batch = flat_latent[start : start + latent_batch_size]
            samples = policy.sample_action_chunks(latent_batch, num_samples=num_samples)
            scores.append(self.score_action_set(samples))
        return torch.cat(scores, dim=0)

    def _ace(self, action_samples: torch.Tensor) -> torch.Tensor:
        if self.cfg.action_range is None:
            raise RuntimeError(
                "ACE needs offline action ranges R_d. Call scorer.fit(calibration_actions), "
                "or load them from a calibrated thresholds file."
            )
        action_range = torch.as_tensor(
            self.cfg.action_range,
            device=action_samples.device,
            dtype=action_samples.dtype,
        )
        if action_range.numel() != action_samples.shape[-1]:
            raise ValueError(
                "ACE action_range has "
                f"{action_range.numel()} dimensions, expected {action_samples.shape[-1]}"
            )
        widths = (float(self.cfg.cell_size_factor) * action_range).clamp_min(self.cfg.eps)
        output = []
        for batch_index in range(action_samples.shape[0]):
            total = action_samples.new_zeros(())
            for horizon_index in range(action_samples.shape[2]):
                values = action_samples[batch_index, :, horizon_index]
                minimum = values.min(dim=0).values
                maximum = values.max(dim=0).values
                bins_per_dim = torch.ceil((maximum - minimum) / widths).long().clamp_min(1)
                indices = torch.floor((values - minimum) / widths).long()
                indices = torch.minimum(indices, bins_per_dim - 1)
                counts = torch.unique(indices, dim=0, return_counts=True)[1].to(values.dtype)
                probabilities = counts / counts.sum()
                total = total - (probabilities * torch.log2(probabilities)).sum()
            output.append(total)
        return torch.stack(output)


def scalar_thresholds_from_scores(
    obs_scores: torch.Tensor,
    action_scores: torch.Tensor,
    *,
    quantile: float = 0.9,
    action_range: Sequence[float] | None = None,
    cell_size_factor: float | None = None,
) -> Thresholds:
    obs = obs_scores.detach().float().reshape(-1)
    action = action_scores.detach().float().reshape(-1)
    if obs.numel() == 0 or action.numel() == 0:
        raise ValueError("Cannot calibrate thresholds from empty score tensors.")
    return Thresholds.scalar(
        obs=float(torch.quantile(obs, quantile).item()),
        action=float(torch.quantile(action, quantile).item()),
        obs_scale=float(obs.std(unbiased=False).clamp_min(1e-6).item()),
        action_scale=float(action.std(unbiased=False).clamp_min(1e-6).item()),
        quantile=quantile,
        action_range=action_range,
        cell_size_factor=cell_size_factor,
    )


def time_varying_thresholds(
    obs_scores: Sequence[Sequence[float]],
    action_scores: Sequence[Sequence[float]],
    *,
    quantile: float = 0.9,
    action_range: Sequence[float] | None = None,
    cell_size_factor: float | None = None,
) -> Thresholds:
    obs_values, obs_scale = _timewise_quantile_and_scale(obs_scores, quantile)
    action_values, action_scale = _timewise_quantile_and_scale(action_scores, quantile)
    return Thresholds(
        obs=obs_values,
        action=action_values,
        obs_scale=obs_scale,
        action_scale=action_scale,
        quantile=float(quantile),
        style="time_varying",
        action_range=None if action_range is None else [float(x) for x in action_range],
        cell_size_factor=cell_size_factor,
    )


def normalize_score(
    raw: torch.Tensor, threshold: float, scale: float, clamp: bool = True
) -> torch.Tensor:
    normalized = (raw - float(threshold)) / max(float(scale), 1e-6)
    return normalized.clamp_min(0.0) if clamp else normalized


def _reduce_horizon(scores: torch.Tensor, reduce: str) -> torch.Tensor:
    if reduce == "sum":
        return scores.sum(dim=-1)
    if reduce == "mean":
        return scores.mean(dim=-1)
    if reduce == "max":
        return scores.max(dim=-1).values
    if reduce == "last":
        return scores[..., -1]
    if reduce == "first":
        return scores[..., 0]
    raise ValueError(f"Unknown horizon reduction={reduce!r}")


def _value_at(values: Sequence[float], step_index: int) -> float:
    if not values:
        return float("inf")
    index = min(max(int(step_index), 0), len(values) - 1)
    return float(values[index])


def _timewise_quantile_and_scale(
    scores: Sequence[Sequence[float]], quantile: float
) -> tuple[list[float], list[float]]:
    max_length = max((len(sequence) for sequence in scores), default=0)
    values, scales = [], []
    for step in range(max_length):
        items = [float(sequence[step]) for sequence in scores if step < len(sequence)]
        if not items:
            continue
        tensor = torch.tensor(items, dtype=torch.float32)
        values.append(float(torch.quantile(tensor, quantile).item()))
        scales.append(float(tensor.std(unbiased=False).clamp_min(1e-6).item()))
    if not values:
        raise ValueError("Cannot calibrate time-varying thresholds from empty score sequences.")
    return values, scales
