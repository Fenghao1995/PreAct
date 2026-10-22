from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

CostFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]]


@dataclass
class CEMConfig:
    num_samples: int = 64
    topk: int = 8
    iterations: int = 4
    init_std: float = 0.15
    min_std: float = 1e-3
    momentum: float = 0.1
    keep_policy_action: bool = True
    lower_bound: float | None = None
    upper_bound: float | None = None


@dataclass
class CEMResult:
    action_sequence: torch.Tensor
    obs_cost: torch.Tensor
    action_cost: torch.Tensor
    total_cost: torch.Tensor
    info: dict[str, torch.Tensor]


class CEMOptimizer:
    def __init__(self, cfg: CEMConfig | None = None):
        self.cfg = cfg or CEMConfig()

    @torch.no_grad()
    def optimize(
        self,
        init_action: torch.Tensor,
        cost_fn: CostFn,
        *,
        initial_samples: torch.Tensor | None = None,
        obs_weight: float = 1.0,
        action_weight: float = 1.0,
    ) -> CEMResult:
        if init_action.ndim != 3:
            raise ValueError(f"init_action must be (B,H,Da), got {tuple(init_action.shape)}")
        cfg = self.cfg
        batch_size = init_action.shape[0]
        num_samples = max(int(cfg.num_samples), 1)
        topk = min(max(int(cfg.topk), 1), num_samples)

        mean = init_action.detach().clone()
        std = torch.full_like(mean, float(cfg.init_std))
        if (
            initial_samples is not None
            and initial_samples.ndim == 4
            and initial_samples.shape[1] > 1
        ):
            samples = initial_samples.to(device=init_action.device, dtype=init_action.dtype)
            mean = samples.mean(dim=1)
            std = samples.std(dim=1, unbiased=False).clamp_min(float(cfg.min_std))

        best_action = init_action.detach().clone()
        best_obs = torch.full(
            (batch_size,), float("inf"), device=init_action.device, dtype=init_action.dtype
        )
        best_act = best_obs.clone()
        best_total = best_obs.clone()
        best_info: dict[str, torch.Tensor] = {}

        for _ in range(max(int(cfg.iterations), 1)):
            noise = torch.randn(
                (batch_size, num_samples, *mean.shape[1:]),
                device=mean.device,
                dtype=mean.dtype,
            )
            candidates = mean[:, None] + std[:, None] * noise
            if cfg.keep_policy_action:
                candidates[:, 0] = init_action
            if cfg.lower_bound is not None or cfg.upper_bound is not None:
                candidates = candidates.clamp(
                    min=-float("inf") if cfg.lower_bound is None else float(cfg.lower_bound),
                    max=float("inf") if cfg.upper_bound is None else float(cfg.upper_bound),
                )

            obs_cost, action_cost, info = cost_fn(candidates)
            total = float(obs_weight) * obs_cost + float(action_weight) * action_cost
            idx = total.argmin(dim=1)
            row = torch.arange(batch_size, device=init_action.device)
            iter_total = total[row, idx]
            update = iter_total < best_total
            best_total = torch.where(update, iter_total, best_total)
            best_obs = torch.where(update, obs_cost[row, idx], best_obs)
            best_act = torch.where(update, action_cost[row, idx], best_act)
            best_action = torch.where(update[:, None, None], candidates[row, idx], best_action)
            best_info = info

            elite_idx = total.topk(topk, dim=1, largest=False).indices
            elite = candidates[row[:, None], elite_idx]
            elite_mean = elite.mean(dim=1)
            elite_std = elite.std(dim=1, unbiased=False).clamp_min(float(cfg.min_std))
            mean = float(cfg.momentum) * mean + (1.0 - float(cfg.momentum)) * elite_mean
            std = float(cfg.momentum) * std + (1.0 - float(cfg.momentum)) * elite_std

        return CEMResult(
            action_sequence=best_action,
            obs_cost=best_obs,
            action_cost=best_act,
            total_cost=best_total,
            info=best_info,
        )
