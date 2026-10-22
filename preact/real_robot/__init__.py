"""Real-world Franka/LeRobot integration for PreAct.

Hardware dependencies are imported lazily so the core package and its tests do
not require a robot, RealSense SDK, or LeRobot installation.
"""

from .config import (
    ACTION_NAMES,
    STATE_NAMES,
    CameraConfig,
    FrankaConfig,
    RealTaskConfig,
    load_real_task_config,
    validate_real_task_config,
)
from .dataset import build_transition_arrays
from .env import FrankaPreActEnv
from .observation import canonicalize_observation

__all__ = [
    "ACTION_NAMES",
    "STATE_NAMES",
    "CameraConfig",
    "FrankaConfig",
    "FrankaPreActEnv",
    "RealTaskConfig",
    "build_transition_arrays",
    "canonicalize_observation",
    "load_real_task_config",
    "validate_real_task_config",
]
