from __future__ import annotations

from typing import Any

import numpy as np
from lerobot.robots.robot import Robot

from preact.real_robot.config import ACTION_NAMES, STATE_NAMES, load_real_task_config
from preact.real_robot.franka import FrankaHardware

from .config_preact_franka import PreActFrankaConfig


class PreActFranka(Robot):
    """LeRobot device interface preserving the original dataset field names."""

    config_class = PreActFrankaConfig
    name = "preact_franka"

    def __init__(self, config: PreActFrankaConfig):
        super().__init__(config)
        self.config = config
        self.task = load_real_task_config(
            config.task_config,
            hardware_path=config.hardware_config,
            require_runtime_values=True,
        )
        self.hardware = FrankaHardware(self.task.hardware)

    @property
    def is_connected(self) -> bool:
        return self.hardware.connected

    @property
    def is_calibrated(self) -> bool:
        return self.is_connected

    @property
    def action_features(self) -> dict[str, type]:
        return {name: float for name in ACTION_NAMES}

    @property
    def observation_features(self) -> dict[str, Any]:
        camera = self.task.hardware.cameras[0]
        state = {name: float for name in STATE_NAMES}
        return {
            **state,
            "wrist_image": (camera.height, camera.width, 3),
            "exterior_image": (camera.height, camera.width, 3),
        }

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self.hardware.connect()

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def reset(self) -> None:
        self.hardware.reset()

    def get_observation(self) -> dict[str, Any]:
        raw = self.hardware.get_observation()
        state = np.asarray(raw["observation.state"], dtype=np.float32)
        return {
            **{name: float(state[index]) for index, name in enumerate(STATE_NAMES)},
            "wrist_image": raw["observation.images.wrist_image"],
            "exterior_image": raw["observation.images.exterior_image"],
        }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        command = np.asarray([action[name] for name in ACTION_NAMES], dtype=np.float32)
        sent = self.hardware.send_action(command)
        # LeRobot 0.3.4 stores action_values after send_action. Mutating the
        # passed mapping keeps the stored transition aligned with clipped/EMA
        # commands even when the processor returned the same dictionary.
        for index, name in enumerate(ACTION_NAMES):
            action[name] = float(sent[index])
        return action

    def disconnect(self) -> None:
        self.hardware.close()
