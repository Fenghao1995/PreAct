from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import TransitionDataset
from .policies import build_policy, save_policy
from .rnd import RandomNetworkDistillation, RNDConfig, save_rnd
from .scoring import (
    ActionEntropyScorer,
    ObservationDeviationScorer,
    Thresholds,
    scalar_thresholds_from_scores,
    time_varying_thresholds,
)
from .sigreg import SIGReg, SIGRegConfig
from .utils import ensure_dir, write_json
from .world_model import LatentWorldModel, LatentWorldModelConfig, save_world_model


@dataclass
class Stage1TrainConfig:
    epochs: int = 20
    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 1e-5
    sigreg_weight: float = 0.1
    sigreg_projections: int = 1024


@dataclass
class Stage2TrainConfig:
    epochs: int = 20
    batch_size: int = 256
    lr: float = 5e-4
    rnd_lr: float = 5e-4
    weight_decay: float = 1e-5
    policy_kind: str = "flow"
    rnd_output_dim: int = 128
    rnd_target_hidden_dims: tuple[int, ...] = (1024, 2048, 4096)
    rnd_predictor_tail_dims: tuple[int, ...] = (2048, 1024)
    rnd_normalize_input: bool = False
    normalizer_batches: int = 16


def train_stage1_lewm(
    *,
    dataset: TransitionDataset,
    model: LatentWorldModel,
    cfg: Stage1TrainConfig,
    device: torch.device,
    out_dir: str | Path | None = None,
) -> tuple[LatentWorldModel, dict[str, Any]]:
    model.to(device)
    model.set_action_normalizer(dataset.arrays.action)
    loader = DataLoader(dataset, batch_size=min(cfg.batch_size, len(dataset)), shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sigreg = SIGReg(SIGRegConfig(num_projections=cfg.sigreg_projections)).to(device)
    history: list[dict[str, float]] = []

    for epoch in range(max(int(cfg.epochs), 1)):
        model.train()
        losses, pred_losses, reg_losses = [], [], []
        for batch in loader:
            obs = batch["obs"].to(device)
            action_chunk = batch["action"].to(device)
            action = action_chunk[:, 0]
            next_obs = batch["next_obs"].to(device)
            latent = model.encode(obs)
            next_latent = model.encode(next_obs)
            pred = model.predict_next_latent(latent, action)
            pred_loss = torch.nn.functional.mse_loss(pred, next_latent)
            reg_loss = sigreg(torch.stack([latent, next_latent], dim=0))
            loss = pred_loss + float(cfg.sigreg_weight) * reg_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            pred_losses.append(float(pred_loss.item()))
            reg_losses.append(float(reg_loss.item()))
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "prediction_loss": float(np.mean(pred_losses)),
                "sigreg": float(np.mean(reg_losses)),
            }
        )

    metrics = evaluate_world_model(dataset, model, cfg.batch_size, device)
    metrics["history"] = history
    if out_dir is not None:
        out_dir = ensure_dir(out_dir)
        save_world_model(
            model.cpu().eval(),
            out_dir / "lewm.pt",
            metadata={"metrics": metrics, "train_config": asdict(cfg)},
        )
        write_json(out_dir / "metrics.json", metrics)
        model.to(device)
    return model, metrics


@torch.no_grad()
def evaluate_world_model(
    dataset: TransitionDataset,
    model: LatentWorldModel,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)
    losses = []
    for batch in loader:
        obs = batch["obs"].to(device)
        action = batch["action"].to(device)[:, 0]
        next_obs = batch["next_obs"].to(device)
        losses.append(float(model.loss(obs, action, next_obs).item()))
    return {
        "num_transitions": int(len(dataset)),
        "normalized_mse": float(np.mean(losses)) if losses else float("nan"),
    }


def train_stage2_policy_and_rnd(
    *,
    dataset: TransitionDataset,
    world_model: LatentWorldModel,
    policy_kind: str,
    policy_kwargs: dict[str, Any],
    cfg: Stage2TrainConfig,
    device: torch.device,
    out_dir: str | Path | None = None,
) -> tuple[torch.nn.Module, RandomNetworkDistillation, dict[str, Any]]:
    world_model.to(device).eval()
    world_model.requires_grad_(False)
    policy = build_policy(policy_kind, **policy_kwargs).to(device)
    rnd = RandomNetworkDistillation(
        RNDConfig(
            input_dim=int(policy_kwargs["latent_dim"]),
            output_dim=int(cfg.rnd_output_dim),
            target_hidden_dims=tuple(int(x) for x in cfg.rnd_target_hidden_dims),
            predictor_tail_dims=tuple(int(x) for x in cfg.rnd_predictor_tail_dims),
            normalize_input=bool(cfg.rnd_normalize_input),
        )
    ).to(device)
    if cfg.rnd_normalize_input:
        _fit_rnd_normalizer(dataset, world_model, rnd, cfg, device)

    loader = DataLoader(dataset, batch_size=min(cfg.batch_size, len(dataset)), shuffle=True)
    policy_optimizer = torch.optim.AdamW(
        policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    rnd_optimizer = torch.optim.AdamW(
        rnd.predictor.parameters(), lr=cfg.rnd_lr, weight_decay=cfg.weight_decay
    )
    history: list[dict[str, float]] = []

    for epoch in range(max(int(cfg.epochs), 1)):
        policy.train()
        rnd.train()
        policy_losses, rnd_losses = [], []
        for batch in loader:
            obs = batch["obs"].to(device)
            action = batch["action"].to(device)
            next_obs = batch["next_obs"].to(device)
            with torch.no_grad():
                latent = world_model.encode(obs)
                real_future = world_model.encode(next_obs)
                imagined_future = world_model.predict_next_latent(latent, action[:, 0])
            policy_loss = policy.loss(latent, action)
            rnd_input = torch.cat([real_future, imagined_future], dim=0)
            rnd_loss = rnd.train_loss(rnd_input)

            policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            policy_optimizer.step()

            rnd_optimizer.zero_grad(set_to_none=True)
            rnd_loss.backward()
            rnd_optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            rnd_losses.append(float(rnd_loss.item()))
        history.append(
            {
                "epoch": epoch + 1,
                "policy_loss": float(np.mean(policy_losses)),
                "rnd_loss": float(np.mean(rnd_losses)),
            }
        )

    metrics = {"num_transitions": int(len(dataset)), "history": history}
    if out_dir is not None:
        out_dir = ensure_dir(out_dir)
        metadata = {
            "metrics": metrics,
            "train_config": asdict(cfg),
            "policy_kind": policy_kind,
        }
        save_policy(policy.cpu().eval(), out_dir / "policy.pt", metadata=metadata)
        save_rnd(rnd.cpu().eval(), out_dir / "drs_o.pt", metadata=metadata)
        write_json(out_dir / "metrics.json", metrics)
        policy.to(device)
        rnd.to(device)
    return policy, rnd, metrics


@torch.no_grad()
def calibrate_thresholds(
    *,
    dataset: TransitionDataset,
    world_model: LatentWorldModel,
    policy,
    rnd: RandomNetworkDistillation,
    action_scorer: ActionEntropyScorer,
    batch_size: int,
    device: torch.device,
    quantile: float = 0.9,
    policy_samples: int = 1,
    future_action_samples: int = 8,
    future_action_latent_index: int | None = None,
    future_action_reduce: str = "sum",
    observation_reduce: str = "sum",
    style: str = "time_varying",
) -> Thresholds:
    world_model.to(device).eval()
    policy.to(device).eval()
    rnd.to(device).eval()
    obs_scorer = ObservationDeviationScorer(rnd, reduce=observation_reduce)
    if action_scorer.cfg.mode == "ace":
        action_scorer.fit(dataset.arrays.action)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)
    obs_scores, action_scores = [], []
    episodes_obs: dict[int, list[tuple[int, float]]] = {}
    episodes_action: dict[int, list[tuple[int, float]]] = {}

    for batch in loader:
        obs = batch["obs"].to(device)
        latent = world_model.encode(obs)
        samples = policy.sample_action_chunks(latent, num_samples=policy_samples)
        base_action = samples[:, 0]
        future = world_model.rollout_from_latent(latent, base_action[:, None])
        batch_obs = obs_scorer.score(future)[:, 0].detach().cpu()
        batch_action = (
            action_scorer.score_candidates_by_future_policy(
                policy=policy,
                future_latents=future,
                num_samples=future_action_samples,
                latent_index=future_action_latent_index,
                reduce=future_action_reduce,
            )[:, 0]
            .detach()
            .cpu()
        )
        obs_scores.append(batch_obs)
        action_scores.append(batch_action)

        if "episode_id" in batch and "step_id" in batch:
            for ep, st, os, ac in zip(
                batch["episode_id"].tolist(),
                batch["step_id"].tolist(),
                batch_obs.tolist(),
                batch_action.tolist(),
                strict=True,
            ):
                episodes_obs.setdefault(int(ep), []).append((int(st), float(os)))
                episodes_action.setdefault(int(ep), []).append((int(st), float(ac)))

    obs_tensor = torch.cat(obs_scores) if obs_scores else torch.empty(0)
    action_tensor = torch.cat(action_scores) if action_scores else torch.empty(0)
    if style == "time_varying":
        if not episodes_obs:
            raise ValueError(
                "Time-varying threshold calibration requires episode_id and step_id in the "
                "calibration dataset. Add these fields or explicitly choose style='scalar'."
            )
        obs_sequences = [_ordered_scores(seq) for seq in episodes_obs.values()]
        action_sequences = [_ordered_scores(seq) for seq in episodes_action.values()]
        return time_varying_thresholds(
            obs_sequences,
            action_sequences,
            quantile=quantile,
            action_range=action_scorer.action_range,
            cell_size_factor=action_scorer.cfg.cell_size_factor,
        )
    return scalar_thresholds_from_scores(
        obs_tensor,
        action_tensor,
        quantile=quantile,
        action_range=action_scorer.action_range,
        cell_size_factor=action_scorer.cfg.cell_size_factor,
    )


def make_world_model_from_task(
    task: dict[str, Any],
    encoder_type: str = "auto",
    world_model_cfg: dict[str, Any] | None = None,
) -> LatentWorldModel:
    model_options = dict(world_model_cfg or {})
    cfg = LatentWorldModelConfig.from_task(
        obs_shape=tuple(task["obs_shape"]),
        obs_history=int(task.get("obs_history", 1)),
        latent_dim=int(task["latent_dim"]),
        action_dim=int(task["action_dim"]),
        action_horizon=int(task["action_horizon"]),
        encoder_type=encoder_type,
        **model_options,
    )
    return LatentWorldModel(cfg)


def policy_kwargs_from_task(
    task: dict[str, Any], policy_cfg: dict[str, Any] | None = None
) -> dict[str, Any]:
    policy_cfg = dict(policy_cfg or {})
    policy_cfg.pop("kind", None)
    return {
        "latent_dim": int(task["latent_dim"]),
        "action_dim": int(task["action_dim"]),
        "action_horizon": int(task["action_horizon"]),
        **policy_cfg,
    }


@torch.no_grad()
def _fit_rnd_normalizer(
    dataset: TransitionDataset,
    world_model: LatentWorldModel,
    rnd: RandomNetworkDistillation,
    cfg: Stage2TrainConfig,
    device: torch.device,
) -> None:
    loader = DataLoader(dataset, batch_size=min(cfg.batch_size, len(dataset)), shuffle=False)
    values = []
    for batch_idx, batch in enumerate(loader):
        obs = batch["obs"].to(device)
        action = batch["action"].to(device)
        next_obs = batch["next_obs"].to(device)
        latent = world_model.encode(obs)
        values.append(world_model.encode(next_obs).cpu())
        values.append(world_model.predict_next_latent(latent, action[:, 0]).cpu())
        if batch_idx + 1 >= int(cfg.normalizer_batches):
            break
    rnd.set_normalizer(torch.cat(values, dim=0))


def _ordered_scores(seq: list[tuple[int, float]]) -> list[float]:
    return [score for _, score in sorted(seq, key=lambda item: item[0])]
