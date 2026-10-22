from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from preact.data import TransitionArrays

from .dataset import build_transition_arrays


@dataclass(frozen=True)
class CollectedEpisode:
    arrays: TransitionArrays
    frame_count: int
    duration_s: float


def collect_embedding_episode(
    hardware: Any,
    action_source: Callable[[dict[str, Any]], np.ndarray | None],
    observation_encoder: Callable[[dict[str, Any]], torch.Tensor | np.ndarray],
    *,
    episode_id: int,
    maximum_steps: int,
    fps: int,
    obs_history: int,
    action_horizon: int,
) -> CollectedEpisode:
    """Collect one synchronized physical rollout at a bounded control rate.

    Returning ``None`` from ``action_source`` ends the episode. Hardware applies
    the final motion limits; the action actually returned by ``send_action`` is
    what is stored, so the dynamics tuple remains physically aligned.
    """

    if maximum_steps <= 0 or fps <= 0:
        raise ValueError("maximum_steps and fps must be positive")
    embeddings: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    period_s = 1.0 / float(fps)
    start = time.perf_counter()
    observation = hardware.get_observation()
    embeddings.append(_encode(observation_encoder, observation))

    for _ in range(maximum_steps):
        loop_start = time.perf_counter()
        proposed = action_source(observation)
        if proposed is None:
            break
        sent = hardware.send_action(np.asarray(proposed, dtype=np.float32))
        actions.append(torch.as_tensor(sent, dtype=torch.float32).flatten())
        remaining = period_s - (time.perf_counter() - loop_start)
        if remaining > 0:
            time.sleep(remaining)
        observation = hardware.get_observation()
        embeddings.append(_encode(observation_encoder, observation))

    if not actions:
        raise RuntimeError("Episode ended before any action was executed")
    # build_transition_arrays consumes equal frame counts and ignores the final
    # frame's placeholder action because it has no outgoing transition.
    actions.append(actions[-1].clone())
    arrays = build_transition_arrays(
        torch.stack(embeddings),
        torch.stack(actions),
        torch.full((len(embeddings),), int(episode_id), dtype=torch.long),
        obs_history=obs_history,
        action_horizon=action_horizon,
    )
    return CollectedEpisode(
        arrays=arrays,
        frame_count=len(embeddings),
        duration_s=time.perf_counter() - start,
    )


def _encode(
    encoder: Callable[[dict[str, Any]], torch.Tensor | np.ndarray],
    observation: dict[str, Any],
) -> torch.Tensor:
    value = torch.as_tensor(encoder(observation), dtype=torch.float32).detach().cpu()
    if value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 1:
        raise ValueError(f"observation encoder returned {tuple(value.shape)}, expected one vector")
    return value
