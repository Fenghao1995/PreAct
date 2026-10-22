from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

import numpy as np

from .config import ACTION_NAMES, STATE_NAMES, CameraConfig, FrankaConfig

logger = logging.getLogger(__name__)


class HardwareError(RuntimeError):
    """Raised when a real-robot operation cannot be completed safely."""


class CameraProtocol(Protocol):
    def connect(self) -> None:
        ...

    def read(self, timeout_ms: int) -> np.ndarray:
        ...

    def close(self) -> None:
        ...


class RPCProtocol(Protocol):
    def call(self, method: str, *args: Any) -> Any:
        ...

    def close(self) -> None:
        ...


class FrankaRPCClient:
    """Small explicit wrapper around the existing Franka zero-RPC server."""

    def __init__(self, host: str, port: int, *, heartbeat_s: int = 20):
        try:
            import zerorpc
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install zerorpc (the 'real' extra) for Franka deployment.") from exc
        self._client = zerorpc.Client(heartbeat=heartbeat_s)
        self._client.connect(f"tcp://{host}:{int(port)}")

    def call(self, method: str, *args: Any) -> Any:
        try:
            return getattr(self._client, method)(*args)
        except Exception as exc:
            raise HardwareError(f"Franka RPC call {method!r} failed") from exc

    def close(self) -> None:
        self._client.close()


class RealSenseCamera:
    """Color-only RealSense stream with bounded read/reconnect behavior."""

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self._pipeline: Any | None = None

    def connect(self) -> None:
        try:
            import pyrealsense2 as rs
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Install pyrealsense2 (the 'real' extra) for camera access."
            ) from exc
        if self._pipeline is not None:
            raise HardwareError(f"camera {self.cfg.name} is already connected")
        pipeline = rs.pipeline()
        stream = rs.config()
        stream.enable_device(self.cfg.serial)
        stream.enable_stream(
            rs.stream.color,
            self.cfg.width,
            self.cfg.height,
            rs.format.bgr8,
            self.cfg.fps,
        )
        pipeline.start(stream)
        self._pipeline = pipeline
        for _ in range(self.cfg.warmup_frames):
            self.read(self.cfg.timeout_ms)

    def read(self, timeout_ms: int) -> np.ndarray:
        if self._pipeline is None:
            raise HardwareError(f"camera {self.cfg.name} is not connected")
        try:
            frames = self._pipeline.wait_for_frames(int(timeout_ms))
            frame = frames.get_color_frame()
            if not frame:
                raise HardwareError(f"camera {self.cfg.name} returned no color frame")
            bgr = np.asanyarray(frame.get_data())
            return np.ascontiguousarray(bgr[..., ::-1])
        except HardwareError:
            raise
        except Exception as exc:
            raise HardwareError(f"camera {self.cfg.name} read failed") from exc

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None


class FrankaHardware:
    """Paper-compatible Franka, dual-camera, and Cartesian-action adapter.

    Action order is ``dx,dy,dz,dRx,dRy,dRz,gripper``. Observations contain two
    RGB arrays plus the 15-D state listed in ``STATE_NAMES``. Motion increments
    are norm-clipped and workspace checked before any RPC command is sent.
    """

    def __init__(
        self,
        cfg: FrankaConfig,
        *,
        rpc: RPCProtocol | None = None,
        cameras: dict[str, CameraProtocol] | None = None,
    ):
        self.cfg = cfg
        self.rpc = rpc
        self.cameras = cameras or {camera.name: RealSenseCamera(camera) for camera in cfg.cameras}
        self.connected = False
        self._smoothed_delta: np.ndarray | None = None
        self._gripper_command = 1.0 if cfg.reset_gripper_open else 0.0

    @property
    def action_names(self) -> tuple[str, ...]:
        return ACTION_NAMES

    @property
    def state_names(self) -> tuple[str, ...]:
        return STATE_NAMES

    def connect(self) -> None:
        if self.connected:
            raise HardwareError("Franka hardware is already connected")
        if self.rpc is None:
            self.rpc = FrankaRPCClient(self.cfg.host, self.cfg.port)
        connected_cameras: list[CameraProtocol] = []
        try:
            joints = np.asarray(self.rpc.call("robot_get_joint_positions"), dtype=np.float64)
            if joints.shape != (7,) or not np.isfinite(joints).all():
                raise HardwareError("Franka server returned an invalid seven-joint state")
            self.rpc.call("robot_start_joint_impedance_control", None, None, True)
            self.rpc.call("gripper_initialize")
            for camera in self.cameras.values():
                camera.connect()
                connected_cameras.append(camera)
            self.connected = True
        except Exception:
            for camera in reversed(connected_cameras):
                camera.close()
            if self.rpc is not None:
                self.rpc.close()
            self.rpc = None
            raise

    def reset(self) -> None:
        rpc = self._require_rpc()
        self._smoothed_delta = None
        if self.cfg.debug:
            return
        rpc.call(
            "robot_move_to_joint_positions",
            list(self.cfg.home_joints_rad),
            5.0,
            False,
            None,
            None,
        )
        target_width = self.cfg.gripper_max_open_m if self.cfg.reset_gripper_open else 0.0
        rpc.call(
            "gripper_goto",
            target_width,
            self.cfg.gripper_speed_m_s,
            self.cfg.gripper_force_n,
            -1.0,
            -1.0,
            True,
        )
        self._gripper_command = 1.0 if self.cfg.reset_gripper_open else 0.0
        rpc.call("robot_start_joint_impedance_control", None, None, True)

    def get_observation(self) -> dict[str, Any]:
        rpc = self._require_rpc()
        with ThreadPoolExecutor(max_workers=max(1, len(self.cameras))) as pool:
            futures = {
                name: pool.submit(self._read_camera_with_retry, name, camera)
                for name, camera in self.cameras.items()
            }
            images = {name: future.result() for name, future in futures.items()}
        joints = np.asarray(rpc.call("robot_get_joint_positions"), dtype=np.float32)
        ee_pose = np.asarray(rpc.call("robot_get_ee_pose"), dtype=np.float32)
        gripper = rpc.call("gripper_get_state")
        if joints.shape != (7,) or ee_pose.shape != (6,):
            raise HardwareError("Franka server returned malformed state")
        width = float(gripper["width"])
        gripper_state = float(np.clip(width / self.cfg.gripper_max_open_m, 0.0, 1.0))
        state = np.concatenate(
            [joints, [gripper_state, self._gripper_command], ee_pose], dtype=np.float32
        )
        if state.shape != (15,) or not np.isfinite(state).all():
            raise HardwareError("assembled Franka proprioceptive state is invalid")
        return {
            "observation.images.wrist_image": images["wrist"],
            "observation.images.exterior_image": images["exterior"],
            "observation.state": state,
            "state_names": STATE_NAMES,
        }

    def send_action(self, action: np.ndarray) -> np.ndarray:
        rpc = self._require_rpc()
        command = np.asarray(action, dtype=np.float64).reshape(-1)
        if command.shape != (7,) or not np.isfinite(command).all():
            raise ValueError("Franka action must contain seven finite values")
        delta = self._clip_delta(command[:6])
        alpha = self.cfg.safety.ema_alpha
        self._smoothed_delta = (
            delta.copy()
            if self._smoothed_delta is None
            else alpha * delta + (1.0 - alpha) * self._smoothed_delta
        )
        current_pose = np.asarray(rpc.call("robot_get_ee_pose"), dtype=np.float64)
        target_pose = self._target_pose(current_pose, self._smoothed_delta)
        self._check_workspace(target_pose[:3])
        gripper_command = float(command[6] > 0.5)
        sent = np.concatenate([self._smoothed_delta, [gripper_command]]).astype(np.float32)
        if not self.cfg.debug:
            rpc.call("robot_update_desired_ee_pose", target_pose.tolist())
            self._update_gripper(gripper_command)
        return sent

    def close(self) -> None:
        errors: list[Exception] = []
        for camera in self.cameras.values():
            try:
                camera.close()
            except Exception as exc:  # close every device before reporting
                errors.append(exc)
        if self.rpc is not None:
            try:
                if self.connected and not self.cfg.debug:
                    self.rpc.call("robot_terminate_current_policy")
                self.rpc.close()
            except Exception as exc:
                errors.append(exc)
        self.rpc = None
        self.connected = False
        if errors:
            raise HardwareError(f"hardware close reported {len(errors)} error(s)") from errors[0]

    def _read_camera_with_retry(self, name: str, camera: CameraProtocol) -> np.ndarray:
        cfg = next(item for item in self.cfg.cameras if item.name == name)
        last_error: Exception | None = None
        for reconnect_round in range(cfg.reconnect_attempts + 1):
            for attempt in range(cfg.max_attempts):
                try:
                    image = camera.read(cfg.timeout_ms)
                    expected = (cfg.height, cfg.width, 3)
                    if image.shape != expected:
                        raise HardwareError(
                            f"camera {name} returned {image.shape}, expected {expected}"
                        )
                    return image
                except Exception as exc:
                    last_error = exc
                    if attempt + 1 < cfg.max_attempts:
                        time.sleep(0.05)
            if reconnect_round < cfg.reconnect_attempts:
                logger.warning("restarting stalled camera %s", name)
                camera.close()
                time.sleep(cfg.reconnect_delay_s)
                camera.connect()
        raise HardwareError(f"camera {name} exhausted bounded retries") from last_error

    def _clip_delta(self, delta: np.ndarray) -> np.ndarray:
        clipped = delta.astype(np.float64, copy=True)
        limits = (
            (slice(0, 3), self.cfg.safety.max_translation_m),
            (slice(3, 6), self.cfg.safety.max_rotation_rad),
        )
        for section, limit in limits:
            norm = float(np.linalg.norm(clipped[section]))
            if norm > limit:
                clipped[section] *= limit / norm
        return clipped

    @staticmethod
    def _target_pose(current_pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
        if current_pose.shape != (6,) or not np.isfinite(current_pose).all():
            raise HardwareError("Franka server returned an invalid end-effector pose")
        try:
            from scipy.spatial.transform import Rotation
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Install scipy (the 'real' extra) for Cartesian rotation updates."
            ) from exc
        current_rotation = Rotation.from_rotvec(current_pose[3:])
        target_rotation = Rotation.from_rotvec(delta[3:]) * current_rotation
        return np.concatenate([current_pose[:3] + delta[:3], target_rotation.as_rotvec()])

    def _check_workspace(self, position: np.ndarray) -> None:
        lower = self.cfg.safety.workspace_min_m
        upper = self.cfg.safety.workspace_max_m
        if lower is None or upper is None:
            return
        if np.any(position < np.asarray(lower)) or np.any(position > np.asarray(upper)):
            raise HardwareError(
                f"target Cartesian position {position.tolist()} is outside workspace"
            )

    def _update_gripper(self, command: float) -> None:
        if command == self._gripper_command:
            return
        rpc = self._require_rpc()
        if command > 0.5:
            rpc.call(
                "gripper_goto",
                self.cfg.gripper_max_open_m,
                self.cfg.gripper_speed_m_s,
                self.cfg.gripper_force_n,
                -1.0,
                -1.0,
                False,
            )
        else:
            rpc.call(
                "gripper_grasp",
                self.cfg.gripper_speed_m_s,
                self.cfg.gripper_force_n,
                0.0,
                -1.0,
                0.1,
                False,
            )
        self._gripper_command = command

    def _require_rpc(self) -> RPCProtocol:
        if not self.connected or self.rpc is None:
            raise HardwareError("Franka hardware is not connected")
        return self.rpc
