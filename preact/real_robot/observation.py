from __future__ import annotations

from typing import Any

import torch

WRIST_IMAGE = "observation.images.wrist_image"
EXTERIOR_IMAGE = "observation.images.exterior_image"
OBSERVATION_STATE = "observation.state"


def canonicalize_observation(
    observation: dict[str, Any],
    *,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Normalize raw hardware and LeRobot frame layouts to one encoder input.

    Images become float ``(C,H,W)`` tensors in ``[0,1]`` and state becomes a
    float 15-vector. This function intentionally performs no dataset-statistics
    normalization; the shared policy encoder factory must apply the same saved
    normalization used during training.
    """

    missing = [
        key for key in (WRIST_IMAGE, EXTERIOR_IMAGE, OBSERVATION_STATE) if key not in observation
    ]
    if missing:
        raise KeyError(f"observation is missing required keys: {missing}")
    return {
        WRIST_IMAGE: _canonical_image(observation[WRIST_IMAGE], device=device),
        EXTERIOR_IMAGE: _canonical_image(observation[EXTERIOR_IMAGE], device=device),
        OBSERVATION_STATE: _canonical_state(observation[OBSERVATION_STATE], device=device),
    }


def _canonical_image(value: Any, *, device: torch.device | str | None) -> torch.Tensor:
    image = torch.as_tensor(value, device=device)
    if image.ndim != 3:
        raise ValueError(f"RGB image must have three dimensions, got {tuple(image.shape)}")
    if image.shape[-1] == 3:
        image = image.permute(2, 0, 1)
    elif image.shape[0] != 3:
        raise ValueError(f"RGB image must be HWC or CHW, got {tuple(image.shape)}")
    image = image.float()
    if image.numel() and image.max() > 1.5:
        image = image / 255.0
    return image.contiguous()


def _canonical_state(value: Any, *, device: torch.device | str | None) -> torch.Tensor:
    state = torch.as_tensor(value, dtype=torch.float32, device=device).flatten()
    if state.shape != (15,):
        raise ValueError(f"Franka state must have shape (15,), got {tuple(state.shape)}")
    return state
