from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import torch

from .config import RealTaskConfig


class FrankaHardwareProtocol(Protocol):
    def connect(self) -> None:
        ...

    def reset(self) -> None:
        ...

    def get_observation(self) -> dict[str, Any]:
        ...

    def send_action(self, action: np.ndarray) -> np.ndarray:
        ...

    def close(self) -> None:
        ...


class FrankaPreActEnv:
    """Gym-style adapter joining Franka hardware to ``OnlinePreActAgent``.

    The encoder is deliberately injected: it must be the frozen, shared
    policy-aligned encoder that produced the training embeddings. The adapter
    never substitutes raw pixels or proprioception when that encoder is absent.
    """

    def __init__(
        self,
        hardware: FrankaHardwareProtocol,
        task: RealTaskConfig,
        observation_encoder: Callable[[dict[str, Any]], torch.Tensor | np.ndarray],
        *,
        auto_connect: bool = True,
        reset_robot_on_reset: bool = False,
        success_fn: Callable[[dict[str, Any]], bool] | None = None,
    ):
        self.hardware = hardware
        self.task = task
        self.observation_encoder = observation_encoder
        self.reset_robot_on_reset = bool(reset_robot_on_reset)
        self.success_fn = success_fn
        self.period_s = 1.0 / float(task.collection.fps)
        self.step_count = 0
        self._connected = False
        if auto_connect:
            self.connect()

    def connect(self) -> None:
        if not self._connected:
            self.hardware.connect()
            self._connected = True

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        self.connect()
        if self.reset_robot_on_reset:
            self.hardware.reset()
        self.step_count = 0
        raw = self.hardware.get_observation()
        return self._encode(raw), {"task": self.task.name, "raw_observation": raw}

    def step(self, action: np.ndarray) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        start = time.perf_counter()
        sent = self.hardware.send_action(np.asarray(action, dtype=np.float32))
        raw = self.hardware.get_observation()
        self.step_count += 1
        success = bool(self.success_fn(raw)) if self.success_fn is not None else False
        terminated = success
        truncated = self.step_count >= self.task.maximum_episode_steps
        remaining = self.period_s - (time.perf_counter() - start)
        if remaining > 0:
            time.sleep(remaining)
        info: dict[str, Any] = {
            "task": self.task.name,
            "sent_action": sent,
            "raw_observation": raw,
        }
        if self.success_fn is not None:
            info["success"] = success
        return self._encode(raw), 1.0 if success else 0.0, terminated, truncated, info

    def close(self) -> None:
        if self._connected:
            self.hardware.close()
            self._connected = False

    def _encode(self, raw: dict[str, Any]) -> torch.Tensor:
        embedding = torch.as_tensor(self.observation_encoder(raw), dtype=torch.float32).detach()
        if embedding.ndim == 2 and embedding.shape[0] == 1:
            embedding = embedding[0]
        if embedding.ndim != 1:
            raise ValueError(
                f"observation encoder must return one 1-D embedding, got {tuple(embedding.shape)}"
            )
        paper_dim = 64
        if embedding.numel() != paper_dim:
            raise ValueError(
                f"{self.task.name} requires a {paper_dim}-D policy-aligned embedding; "
                f"encoder returned {embedding.numel()} values"
            )
        return embedding.cpu()

    def __enter__(self) -> FrankaPreActEnv:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
