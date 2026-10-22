from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from preact.alignment import PAPER_TASKS
from preact.utils import deep_update, read_text_config

ACTION_NAMES = (
    "delta_ee_pose.x",
    "delta_ee_pose.y",
    "delta_ee_pose.z",
    "delta_ee_pose.rx",
    "delta_ee_pose.ry",
    "delta_ee_pose.rz",
    "gripper_cmd_bin",
)

STATE_NAMES = (
    "joint_1.pos",
    "joint_2.pos",
    "joint_3.pos",
    "joint_4.pos",
    "joint_5.pos",
    "joint_6.pos",
    "joint_7.pos",
    "gripper_state_norm",
    "gripper_cmd_bin",
    "ee_pose.x",
    "ee_pose.y",
    "ee_pose.z",
    "ee_pose.rx",
    "ee_pose.ry",
    "ee_pose.rz",
)

REAL_TASKS = {"pick_place", "insert", "push_tennis"}
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


@dataclass(frozen=True)
class CameraConfig:
    name: str
    serial: str
    width: int
    height: int
    fps: int = 15
    timeout_ms: int = 1000
    max_attempts: int = 3
    reconnect_attempts: int = 2
    reconnect_delay_s: float = 1.0
    warmup_frames: int = 45


@dataclass(frozen=True)
class SafetyConfig:
    max_translation_m: float = 0.02
    max_rotation_rad: float = 0.1
    ema_alpha: float = 0.4
    workspace_min_m: tuple[float, float, float] | None = None
    workspace_max_m: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class FrankaConfig:
    host: str
    port: int = 4242
    cameras: tuple[CameraConfig, ...] = ()
    home_joints_rad: tuple[float, ...] = (
        0.0424,
        -0.5245,
        -0.0951,
        -2.7361,
        0.0030,
        2.9713,
        0.8088,
    )
    gripper_max_open_m: float = 0.080
    gripper_speed_m_s: float = 0.1
    gripper_force_n: float = 20.0
    reset_gripper_open: bool = False
    debug: bool = False
    safety: SafetyConfig = field(default_factory=SafetyConfig)


@dataclass(frozen=True)
class CollectionConfig:
    expert_rollouts: int = 50
    random_play_rollouts: int = 10
    calibration_rollouts: int = 20
    fps: int = 15
    data_root: str = ""


@dataclass(frozen=True)
class RealTaskConfig:
    name: str
    description: str
    success: str
    observation_shape: tuple[int, ...]
    action_dim: int
    state_dim: int
    maximum_episode_steps: int
    evaluation: dict[str, Any]
    random_play: tuple[str, ...]
    hardware: FrankaConfig
    collection: CollectionConfig = field(default_factory=CollectionConfig)


def load_real_task_config(
    task_path: str | Path,
    *,
    hardware_path: str | Path | None = None,
    environ: dict[str, str] | None = None,
    require_runtime_values: bool = False,
) -> RealTaskConfig:
    """Load a paper task and optional machine-local hardware overlay.

    Values support ``${NAME}`` and ``${NAME:-fallback}``. Machine-specific
    addresses and camera serials therefore stay outside version control.
    """

    raw = read_text_config(task_path)
    if hardware_path is not None:
        raw = deep_update(raw, read_text_config(hardware_path))
    expanded = _expand_env(raw, environ if environ is not None else dict(os.environ))
    cfg = _from_mapping(expanded)
    validate_real_task_config(cfg, require_runtime_values=require_runtime_values)
    return cfg


def validate_real_task_config(
    cfg: RealTaskConfig,
    *,
    require_runtime_values: bool = False,
) -> None:
    errors: list[str] = []
    if cfg.name not in REAL_TASKS:
        errors.append(f"task.name must be one of {sorted(REAL_TASKS)}, got {cfg.name!r}")
    else:
        paper = PAPER_TASKS[cfg.name]
        expected = {
            "observation_shape": paper.observation_shape,
            "action_dim": paper.action_dim,
            "state_dim": paper.state_dim,
            "calibration_rollouts": paper.calibration_rollouts,
            "maximum_episode_steps": paper.maximum_episode_length,
        }
        actual = {
            "observation_shape": cfg.observation_shape,
            "action_dim": cfg.action_dim,
            "state_dim": cfg.state_dim,
            "calibration_rollouts": cfg.collection.calibration_rollouts,
            "maximum_episode_steps": cfg.maximum_episode_steps,
        }
        for key, value in expected.items():
            if actual[key] != value:
                errors.append(f"{key}={actual[key]!r}, paper={value!r}")

    if len(cfg.hardware.cameras) != 2:
        errors.append("exactly two cameras (wrist and exterior) are required")
    expected_hw = cfg.observation_shape
    if len(expected_hw) == 4:
        expected_views, expected_channels, expected_height, expected_width = expected_hw
        if expected_views != 2 or expected_channels != 3:
            errors.append(f"paper observation shape must encode two RGB views, got {expected_hw}")
        for camera in cfg.hardware.cameras:
            if (camera.height, camera.width) != (expected_height, expected_width):
                errors.append(
                    f"camera {camera.name} is {camera.height}x{camera.width}; "
                    f"paper requires {expected_height}x{expected_width}"
                )
            if camera.fps != 15:
                errors.append(f"camera {camera.name} must run at paper collection rate 15 FPS")
    if {camera.name for camera in cfg.hardware.cameras} != {"wrist", "exterior"}:
        errors.append("camera names must be exactly 'wrist' and 'exterior'")
    if cfg.collection.fps != 15:
        errors.append("collection.fps must be 15")
    if cfg.collection.expert_rollouts != 50:
        errors.append("collection.expert_rollouts must be 50")
    if cfg.collection.random_play_rollouts != 10:
        errors.append("collection.random_play_rollouts must be 10")
    if len(cfg.hardware.home_joints_rad) != 7:
        errors.append("hardware.home_joints_rad must contain seven joints")
    if not cfg.evaluation.get("id"):
        errors.append("task.evaluation.id protocol is required")
    if cfg.name == "push_tennis" and cfg.evaluation.get("ood") is not None:
        errors.append("PUSH_TENNIS must not define a separate OOD scene")
    if not cfg.random_play:
        errors.append("task.random_play must document bounded physical exploration")
    safety = cfg.hardware.safety
    if not 0.0 < safety.max_translation_m <= 0.05:
        errors.append("safety.max_translation_m must be in (0, 0.05]")
    if not 0.0 < safety.max_rotation_rad <= 0.25:
        errors.append("safety.max_rotation_rad must be in (0, 0.25]")
    if not 0.0 < safety.ema_alpha <= 1.0:
        errors.append("safety.ema_alpha must be in (0, 1]")
    if (safety.workspace_min_m is None) != (safety.workspace_max_m is None):
        errors.append("workspace_min_m and workspace_max_m must be set together")
    if safety.workspace_min_m is not None and safety.workspace_max_m is not None:
        if len(safety.workspace_min_m) != 3 or len(safety.workspace_max_m) != 3:
            errors.append("workspace bounds must have three coordinates")
        elif any(
            lo >= hi for lo, hi in zip(safety.workspace_min_m, safety.workspace_max_m, strict=True)
        ):
            errors.append("every workspace minimum must be below its maximum")
    if require_runtime_values:
        if not cfg.hardware.host or "${" in cfg.hardware.host:
            errors.append("Franka host is unresolved; set PREACT_FRANKA_HOST or a local overlay")
        for camera in cfg.hardware.cameras:
            if not camera.serial or "${" in camera.serial:
                errors.append(f"camera {camera.name} serial is unresolved")
        if not cfg.collection.data_root or "${" in cfg.collection.data_root:
            errors.append("collection.data_root is unresolved")
        if safety.workspace_min_m is None or safety.workspace_max_m is None:
            errors.append(
                "measured workspace bounds are required for hardware deployment; "
                "set them in hardware.local.yaml"
            )
    if errors:
        raise ValueError("Real-task configuration is invalid:\n- " + "\n- ".join(errors))


def _from_mapping(raw: dict[str, Any]) -> RealTaskConfig:
    task = dict(raw.get("task", {}))
    hardware = dict(raw.get("hardware", {}))
    collection = dict(raw.get("collection", {}))
    cameras = tuple(CameraConfig(**camera) for camera in hardware.pop("cameras", []))
    safety_raw = hardware.pop("safety", {})
    for key in ("workspace_min_m", "workspace_max_m"):
        if safety_raw.get(key) is not None:
            safety_raw[key] = tuple(float(value) for value in safety_raw[key])
    safety = SafetyConfig(**safety_raw)
    if "home_joints_rad" in hardware:
        hardware["home_joints_rad"] = tuple(float(value) for value in hardware["home_joints_rad"])
    franka = FrankaConfig(cameras=cameras, safety=safety, **hardware)
    return RealTaskConfig(
        name=str(task["name"]),
        description=str(task["description"]),
        success=str(task["success"]),
        observation_shape=tuple(int(value) for value in task["observation_shape"]),
        action_dim=int(task["action_dim"]),
        state_dim=int(task["state_dim"]),
        maximum_episode_steps=int(task["maximum_episode_steps"]),
        evaluation=dict(task["evaluation"]),
        random_play=tuple(str(value) for value in task["random_play"]),
        hardware=franka,
        collection=CollectionConfig(**collection),
    )


def _expand_env(value: Any, environ: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item, environ) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item, environ) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, fallback = match.group(1), match.group(2)
        if name in environ:
            return environ[name]
        if fallback is not None:
            return fallback
        return match.group(0)

    return _ENV_PATTERN.sub(replace, value)
