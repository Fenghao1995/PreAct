from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from preact.data import TransitionArrays


def build_transition_arrays(
    embeddings: torch.Tensor | np.ndarray,
    actions: torch.Tensor | np.ndarray,
    episode_ids: torch.Tensor | np.ndarray,
    *,
    obs_history: int,
    action_horizon: int,
) -> TransitionArrays:
    """Convert synchronized frame data into paper-aligned transition tuples.

    Histories are edge padded at the start of an episode; action chunks are
    edge padded at the end. The first action in every chunk is always the
    physically executed action between ``obs`` and ``next_obs``.
    """

    z = torch.as_tensor(embeddings, dtype=torch.float32)
    action = torch.as_tensor(actions, dtype=torch.float32)
    episode = torch.as_tensor(episode_ids, dtype=torch.long).flatten()
    if z.ndim != 2:
        raise ValueError(f"embeddings must have shape (frames, dim), got {tuple(z.shape)}")
    if action.ndim != 2:
        raise ValueError(f"actions must have shape (frames, dim), got {tuple(action.shape)}")
    if not (len(z) == len(action) == len(episode)):
        raise ValueError("embeddings, actions, and episode_ids must have equal frame counts")
    if obs_history <= 0 or action_horizon <= 0:
        raise ValueError("obs_history and action_horizon must be positive")

    obs_rows: list[torch.Tensor] = []
    next_rows: list[torch.Tensor] = []
    action_rows: list[torch.Tensor] = []
    out_episode: list[int] = []
    out_step: list[int] = []
    ordered_episode_ids = list(dict.fromkeys(int(value) for value in episode.tolist()))
    for episode_id in ordered_episode_ids:
        indices = torch.nonzero(episode == episode_id, as_tuple=False).flatten()
        if len(indices) < 2:
            continue
        if not torch.equal(indices, torch.arange(indices[0], indices[-1] + 1)):
            raise ValueError(f"episode {episode_id} is not contiguous in frame order")
        first, length = int(indices[0]), len(indices)
        for local_t in range(length - 1):
            obs_indices = _edge_padded_indices(local_t, obs_history, length)
            next_indices = _edge_padded_indices(local_t + 1, obs_history, length)
            action_indices = [min(local_t + offset, length - 1) for offset in range(action_horizon)]
            obs_rows.append(z[[first + value for value in obs_indices]])
            next_rows.append(z[[first + value for value in next_indices]])
            action_rows.append(action[[first + value for value in action_indices]])
            out_episode.append(episode_id)
            out_step.append(local_t)

    if not obs_rows:
        raise ValueError("No episode contains at least two synchronized frames")
    return TransitionArrays(
        obs=torch.stack(obs_rows),
        action=torch.stack(action_rows),
        next_obs=torch.stack(next_rows),
        episode_id=torch.tensor(out_episode, dtype=torch.long),
        step_id=torch.tensor(out_step, dtype=torch.long),
    ).validate()


def load_lerobot_transition_arrays(
    *,
    repo_id: str,
    root: str | Path,
    observation_encoder: Callable[[dict[str, Any]], torch.Tensor | np.ndarray],
    obs_history: int,
    action_horizon: int,
    max_frames: int | None = None,
) -> TransitionArrays:
    """Read a local LeRobot 0.3.x dataset without copying images or videos.

    ``observation_encoder`` must be the same policy-aligned encoder used during
    deployment and return one 1-D embedding per frame (64-D for all paper real
    tasks). Keeping this callback explicit prevents silent train/deploy feature
    mismatch.
    """

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install the 'real' extra to read a LeRobot dataset.") from exc

    dataset = LeRobotDataset(repo_id=repo_id, root=Path(root).expanduser())
    embeddings: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    episode_ids: list[int] = []
    frame_count = len(dataset) if max_frames is None else min(len(dataset), int(max_frames))
    for index in range(frame_count):
        frame = dataset[index]
        embedding = torch.as_tensor(observation_encoder(frame), dtype=torch.float32).detach().cpu()
        if embedding.ndim == 2 and embedding.shape[0] == 1:
            embedding = embedding[0]
        if embedding.ndim != 1:
            raise ValueError(
                f"observation_encoder must return a 1-D embedding, got {tuple(embedding.shape)}"
            )
        embeddings.append(embedding)
        actions.append(torch.as_tensor(frame["action"], dtype=torch.float32).flatten().cpu())
        episode_ids.append(_scalar_int(frame.get("episode_index"), name="episode_index"))

    return build_transition_arrays(
        torch.stack(embeddings),
        torch.stack(actions),
        torch.tensor(episode_ids),
        obs_history=obs_history,
        action_horizon=action_horizon,
    )


def save_transition_npz(path: str | Path, arrays: TransitionArrays) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "obs": arrays.obs.cpu().numpy(),
        "action": arrays.action.cpu().numpy(),
        "next_obs": arrays.next_obs.cpu().numpy(),
    }
    if arrays.episode_id is not None:
        payload["episode_id"] = arrays.episode_id.cpu().numpy()
    if arrays.step_id is not None:
        payload["step_id"] = arrays.step_id.cpu().numpy()
    np.savez_compressed(path, **payload)
    return path


def episode_count(episode_ids: Sequence[int] | torch.Tensor | np.ndarray) -> int:
    values = torch.as_tensor(episode_ids).flatten().tolist()
    return len(set(int(value) for value in values))


def _edge_padded_indices(end: int, history: int, episode_length: int) -> list[int]:
    start = end - history + 1
    return [min(max(index, 0), episode_length - 1) for index in range(start, end + 1)]


def _scalar_int(value: Any, *, name: str) -> int:
    if value is None:
        raise KeyError(f"LeRobot frame is missing {name!r}")
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        return int(value.item())
    return int(value)
