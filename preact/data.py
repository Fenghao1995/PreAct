from __future__ import annotations

import pickle
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import as_float_tensor

OBS_KEYS = ("obs", "observation", "obs_embedding", "obs_embeddings", "state")
NEXT_OBS_KEYS = ("next_obs", "next_observation", "next_obs_embedding", "future_obs_embedding")
ACTION_KEYS = ("action", "actions", "action_chunk", "action_pred", "action_preds")


@dataclass
class TransitionArrays:
    obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor
    episode_id: torch.Tensor | None = None
    step_id: torch.Tensor | None = None
    action_candidates: torch.Tensor | None = None

    def validate(self) -> TransitionArrays:
        if self.obs.shape[0] != self.action.shape[0] or self.obs.shape[0] != self.next_obs.shape[0]:
            raise ValueError(
                "obs, action, and next_obs must have the same leading dimension: "
                f"{self.obs.shape[0]}, {self.action.shape[0]}, {self.next_obs.shape[0]}"
            )
        if self.action.ndim == 2:
            self.action = self.action[:, None, :]
        if self.action.ndim != 3:
            raise ValueError(f"action must be (N,H,Da) or (N,Da), got {tuple(self.action.shape)}")
        for name in ("episode_id", "step_id", "action_candidates"):
            value = getattr(self, name)
            if value is not None and value.shape[0] != self.obs.shape[0]:
                raise ValueError(f"{name} must have the same leading dimension as obs")
        return self


class TransitionDataset(Dataset):
    def __init__(self, arrays: TransitionArrays):
        self.arrays = arrays.validate()

    def __len__(self) -> int:
        return int(self.arrays.obs.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "obs": self.arrays.obs[index].float(),
            "action": self.arrays.action[index].float(),
            "next_obs": self.arrays.next_obs[index].float(),
        }
        if self.arrays.episode_id is not None:
            item["episode_id"] = self.arrays.episode_id[index].long()
        if self.arrays.step_id is not None:
            item["step_id"] = self.arrays.step_id[index].long()
        if self.arrays.action_candidates is not None:
            item["action_candidates"] = self.arrays.action_candidates[index].float()
        return item


class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_transition_dataset(
    path: str | Path,
    *,
    obs_keys: Sequence[str] = OBS_KEYS,
    next_obs_keys: Sequence[str] = NEXT_OBS_KEYS,
    action_keys: Sequence[str] = ACTION_KEYS,
    max_samples: int | None = None,
) -> TransitionDataset:
    path = Path(path).expanduser()
    if str(path) == "." and not path.exists():
        raise FileNotFoundError("Dataset path is empty.")
    if not path.exists():
        raise FileNotFoundError(path)
    arrays = load_transition_arrays(
        path,
        obs_keys=obs_keys,
        next_obs_keys=next_obs_keys,
        action_keys=action_keys,
        max_samples=max_samples,
    )
    return TransitionDataset(arrays)


def load_transition_arrays(
    path: str | Path,
    *,
    obs_keys: Sequence[str] = OBS_KEYS,
    next_obs_keys: Sequence[str] = NEXT_OBS_KEYS,
    action_keys: Sequence[str] = ACTION_KEYS,
    max_samples: int | None = None,
) -> TransitionArrays:
    path = Path(path).expanduser()
    if path.is_dir():
        chunks = [
            load_transition_arrays(
                child,
                obs_keys=obs_keys,
                next_obs_keys=next_obs_keys,
                action_keys=action_keys,
                max_samples=None,
            )
            for child in _iter_supported_files(path)
        ]
        if not chunks:
            raise RuntimeError(f"No supported dataset files found under {path}")
        arrays = concatenate_arrays(chunks)
    elif path.suffix.lower() == ".npz":
        arrays = _load_npz(path, obs_keys, next_obs_keys, action_keys)
    elif path.suffix.lower() in {".h5", ".hdf5"}:
        arrays = _load_hdf5(path, obs_keys, next_obs_keys, action_keys)
    elif path.suffix.lower() == ".pkl":
        arrays = _load_rollout_pickle(path, obs_keys, action_keys)
    else:
        raise ValueError(f"Unsupported dataset format: {path}")
    if max_samples is not None:
        arrays = slice_arrays(arrays, max_samples)
    return arrays.validate()


def concatenate_arrays(items: Sequence[TransitionArrays]) -> TransitionArrays:
    if not items:
        raise ValueError("Cannot concatenate an empty array list.")

    def cat_optional(name: str) -> torch.Tensor | None:
        values = [getattr(item, name) for item in items]
        if any(value is None for value in values):
            return None
        return torch.cat([value for value in values if value is not None], dim=0)

    episode_ids = _concatenate_episode_ids(items)
    return TransitionArrays(
        obs=torch.cat([item.obs for item in items], dim=0),
        action=torch.cat([item.action for item in items], dim=0),
        next_obs=torch.cat([item.next_obs for item in items], dim=0),
        episode_id=episode_ids,
        step_id=cat_optional("step_id"),
        action_candidates=cat_optional("action_candidates"),
    ).validate()


def slice_arrays(arrays: TransitionArrays, max_samples: int) -> TransitionArrays:
    n = min(int(max_samples), arrays.obs.shape[0])
    return TransitionArrays(
        obs=arrays.obs[:n],
        action=arrays.action[:n],
        next_obs=arrays.next_obs[:n],
        episode_id=None if arrays.episode_id is None else arrays.episode_id[:n],
        step_id=None if arrays.step_id is None else arrays.step_id[:n],
        action_candidates=None
        if arrays.action_candidates is None
        else arrays.action_candidates[:n],
    )


def make_synthetic_dataset(
    *,
    num_samples: int = 512,
    obs_shape: Sequence[int] = (16,),
    action_horizon: int = 8,
    action_dim: int = 2,
    obs_history: int = 1,
    seed: int = 7,
) -> TransitionDataset:
    rng = torch.Generator().manual_seed(int(seed))
    flat_obs_dim = int(np.prod(obs_shape))
    obs = torch.randn(num_samples, obs_history, flat_obs_dim, generator=rng)
    action = torch.randn(num_samples, action_horizon, action_dim, generator=rng) * 0.4
    dynamics = torch.randn(action_dim, flat_obs_dim, generator=rng) * 0.08
    drift = torch.einsum("nd,df->nf", action[:, 0], dynamics)
    next_last = obs[:, -1] + drift + 0.03 * torch.randn(num_samples, flat_obs_dim, generator=rng)
    next_obs = (
        torch.cat([obs[:, 1:], next_last[:, None]], dim=1)
        if obs_history > 1
        else next_last[:, None]
    )
    episode_id = torch.arange(num_samples) // max(action_horizon, 1)
    step_id = torch.arange(num_samples) % max(action_horizon, 1)
    return TransitionDataset(
        TransitionArrays(
            obs=obs, action=action, next_obs=next_obs, episode_id=episode_id, step_id=step_id
        )
    )


def _iter_supported_files(path: Path) -> Iterable[Path]:
    for suffix in ("*.npz", "*.h5", "*.hdf5", "*.pkl"):
        yield from sorted(path.rglob(suffix))


def _load_npz(
    path: Path, obs_keys: Sequence[str], next_obs_keys: Sequence[str], action_keys: Sequence[str]
):
    with np.load(path, allow_pickle=True) as data:
        obs = _first_array(data, obs_keys)
        next_obs = _first_array(data, next_obs_keys)
        action = _first_array(data, action_keys)
        episode_id = _optional_array(data, ("episode_id", "episode", "traj_id"))
        step_id = _optional_array(data, ("step_id", "timestep", "t"))
        candidates = _optional_array(
            data, ("action_candidates", "action_pred_samples", "action_preds")
        )
    return TransitionArrays(
        obs=as_float_tensor(obs),
        action=as_float_tensor(action),
        next_obs=as_float_tensor(next_obs),
        episode_id=None if episode_id is None else torch.as_tensor(episode_id, dtype=torch.long),
        step_id=None if step_id is None else torch.as_tensor(step_id, dtype=torch.long),
        action_candidates=None if candidates is None else as_float_tensor(candidates),
    )


def _load_hdf5(
    path: Path, obs_keys: Sequence[str], next_obs_keys: Sequence[str], action_keys: Sequence[str]
):
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install h5py to read HDF5 datasets.") from exc
    with h5py.File(path, "r") as f:
        obs = _first_h5(f, obs_keys)
        next_obs = _first_h5(f, next_obs_keys)
        action = _first_h5(f, action_keys)
        episode_id = _optional_h5(f, ("episode_id", "episode", "traj_id"))
        step_id = _optional_h5(f, ("step_id", "timestep", "t"))
        candidates = _optional_h5(f, ("action_candidates", "action_pred_samples", "action_preds"))
    return TransitionArrays(
        obs=as_float_tensor(obs),
        action=as_float_tensor(action),
        next_obs=as_float_tensor(next_obs),
        episode_id=None if episode_id is None else torch.as_tensor(episode_id, dtype=torch.long),
        step_id=None if step_id is None else torch.as_tensor(step_id, dtype=torch.long),
        action_candidates=None if candidates is None else as_float_tensor(candidates),
    )


def _load_rollout_pickle(
    path: Path, obs_keys: Sequence[str], action_keys: Sequence[str]
) -> TransitionArrays:
    with path.open("rb") as f:
        payload = _NumpyCompatUnpickler(f).load()
    if isinstance(payload, dict):
        rollout = payload.get("rollout") or payload.get("steps") or payload.get("trajectory") or []
    else:
        rollout = payload
    if len(rollout) < 2:
        raise RuntimeError(f"Rollout {path} has fewer than two steps.")
    obs, action, next_obs, step_ids, candidates = [], [], [], [], []
    for index, (step, next_step) in enumerate(zip(rollout[:-1], rollout[1:], strict=True)):
        current = _first_mapping_value(step, obs_keys)
        future = _first_mapping_value(next_step, obs_keys)
        act = _first_mapping_value(step, action_keys)
        cand = _first_mapping_value(
            step, ("action_candidates", "action_pred_samples", "action_preds")
        )
        if current is None or future is None or act is None:
            continue
        obs.append(as_float_tensor(current))
        action.append(as_float_tensor(act))
        next_obs.append(as_float_tensor(future))
        step_ids.append(index)
        if cand is not None:
            candidates.append(as_float_tensor(cand))
    if not obs:
        raise RuntimeError(f"No transition triples found in {path}")
    candidate_tensor = None
    if len(candidates) == len(obs) and _same_shapes(candidates):
        candidate_tensor = torch.stack(candidates)
    episode_id = torch.zeros(len(obs), dtype=torch.long)
    return TransitionArrays(
        obs=torch.stack(obs),
        action=torch.stack(action),
        next_obs=torch.stack(next_obs),
        episode_id=episode_id,
        step_id=torch.as_tensor(step_ids, dtype=torch.long),
        action_candidates=candidate_tensor,
    )


def _first_array(data: Any, keys: Sequence[str]) -> np.ndarray:
    value = _optional_array(data, keys)
    if value is None:
        raise KeyError(f"None of the keys {tuple(keys)} were found. Available: {list(data.keys())}")
    return value


def _optional_array(data: Any, keys: Sequence[str]) -> np.ndarray | None:
    for key in keys:
        if key in data:
            return np.asarray(data[key])
    return None


def _first_h5(group: Any, keys: Sequence[str]) -> np.ndarray:
    value = _optional_h5(group, keys)
    if value is None:
        raise KeyError(f"None of the keys {tuple(keys)} were found in HDF5 file.")
    return value


def _optional_h5(group: Any, keys: Sequence[str]) -> np.ndarray | None:
    for key in keys:
        if key in group:
            return np.asarray(group[key])
    for key in keys:
        found = _find_h5_dataset(group, key)
        if found is not None:
            return np.asarray(found)
    return None


def _find_h5_dataset(group: Any, target_name: str):
    for key, value in group.items():
        if key == target_name:
            return value
        if hasattr(value, "items"):
            found = _find_h5_dataset(value, target_name)
            if found is not None:
                return found
    return None


def _first_mapping_value(step: dict[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in step and step[key] is not None:
            return step[key]
    return None


def _same_shapes(values: Sequence[torch.Tensor]) -> bool:
    if not values:
        return False
    shape = tuple(values[0].shape)
    return all(tuple(value.shape) == shape for value in values)


def _concatenate_episode_ids(items: Sequence[TransitionArrays]) -> torch.Tensor | None:
    if any(item.episode_id is None for item in items):
        return None
    offset = 0
    adjusted = []
    for item in items:
        episode_id = item.episode_id.long()
        if episode_id.numel() == 0:
            continue
        normalized = episode_id - episode_id.min()
        adjusted.append(normalized + offset)
        offset += int(normalized.max().item()) + 1
    return torch.cat(adjusted, dim=0) if adjusted else torch.empty(0, dtype=torch.long)
