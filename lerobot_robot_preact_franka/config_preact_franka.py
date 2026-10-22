from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("preact_franka")
@dataclass(kw_only=True)
class PreActFrankaConfig(RobotConfig):
    """Paths to the public task config and ignored machine-local overlay."""

    task_config: Path
    hardware_config: Path
