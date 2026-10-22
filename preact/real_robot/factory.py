from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from .config import RealTaskConfig


def load_object(import_path: str) -> Any:
    if ":" not in import_path:
        raise ValueError("import path must have form package.module:object")
    module_name, object_name = import_path.split(":", 1)
    return getattr(importlib.import_module(module_name), object_name)


def make_observation_encoder(
    import_path: str,
    *,
    checkpoint: str | Path | None,
    device: torch.device,
    task: RealTaskConfig,
) -> Callable[[dict[str, Any]], torch.Tensor]:
    """Instantiate the shared policy-aligned observation encoder.

    The factory signature is ``factory(checkpoint=..., device=..., task=...)``.
    The returned callable must map one raw/LeRobot observation dictionary to
    the 64-D latent used for both dataset export and deployment.
    """

    factory = load_object(import_path)
    if not callable(factory):
        raise TypeError(f"{import_path} is not callable")
    encoder = factory(checkpoint=checkpoint, device=device, task=task)
    if not callable(encoder):
        raise TypeError(f"{import_path} did not return a callable encoder")
    return encoder
