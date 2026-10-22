"""LeRobot 0.3.x device plugin backed by PreAct's FrankaHardware."""

from .config_preact_franka import PreActFrankaConfig
from .preactfranka import PreActFranka

__all__ = ["PreActFranka", "PreActFrankaConfig"]
