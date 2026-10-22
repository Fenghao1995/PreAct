#!/usr/bin/env python
from __future__ import annotations

import argparse

import numpy as np
import torch


class FrankaInterfaceServer:
    """Zero-RPC facade for a Polymetis Franka controller on the robot NUC."""

    def __init__(self, *, robot_ip: str | None = None):
        try:
            from polymetis import GripperInterface, RobotInterface
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install Polymetis in the Franka NUC environment.") from exc
        kwargs = {"enforce_version": False}
        if robot_ip is not None:
            kwargs["ip_address"] = robot_ip
        self.robot = RobotInterface(**kwargs)
        self.gripper_class = GripperInterface
        self.gripper = None

    def gripper_initialize(self):
        self.gripper = self.gripper_class()

    def gripper_goto(
        self,
        width,
        speed,
        force,
        epsilon_inner=-1.0,
        epsilon_outer=-1.0,
        blocking=True,
    ):
        self._require_gripper().goto(
            width=width,
            speed=speed,
            force=force,
            epsilon_inner=epsilon_inner,
            epsilon_outer=epsilon_outer,
            blocking=blocking,
        )

    def gripper_grasp(
        self,
        speed,
        force,
        grasp_width=0.0,
        epsilon_inner=-1.0,
        epsilon_outer=-1.0,
        blocking=True,
    ):
        self._require_gripper().grasp(
            speed=speed,
            force=force,
            grasp_width=grasp_width,
            epsilon_inner=epsilon_inner,
            epsilon_outer=epsilon_outer,
            blocking=blocking,
        )

    def gripper_get_state(self):
        state = self._require_gripper().get_state()
        return {
            "width": state.width,
            "is_moving": state.is_moving,
            "is_grasped": state.is_grasped,
            "prev_command_successful": state.prev_command_successful,
            "error_code": state.error_code,
        }

    def robot_get_joint_positions(self):
        return self.robot.get_joint_positions().numpy().tolist()

    def robot_get_joint_velocities(self):
        return self.robot.get_joint_velocities().numpy().tolist()

    def robot_get_ee_pose(self):
        from scipy.spatial.transform import Rotation

        position, quaternion_xyzw = self.robot.get_ee_pose()
        rotation = Rotation.from_quat(quaternion_xyzw.numpy()).as_rotvec()
        return np.concatenate([position.numpy(), rotation]).tolist()

    def robot_move_to_joint_positions(
        self,
        positions,
        time_to_go=None,
        delta=False,
        Kq=None,
        Kqd=None,
    ):
        self.robot.move_to_joint_positions(
            positions=torch.tensor(positions),
            time_to_go=time_to_go,
            delta=delta,
            Kq=None if Kq is None else torch.tensor(Kq),
            Kqd=None if Kqd is None else torch.tensor(Kqd),
        )

    def robot_go_home(self):
        self.robot.go_home()

    def robot_move_to_ee_pose(
        self,
        pose,
        time_to_go=None,
        delta=False,
        Kx=None,
        Kxd=None,
        op_space_interp=True,
    ):
        from scipy.spatial.transform import Rotation

        pose = np.asarray(pose, dtype=np.float32)
        quaternion = Rotation.from_rotvec(pose[3:]).as_quat()
        self.robot.move_to_ee_pose(
            position=torch.tensor(pose[:3]),
            orientation=torch.tensor(quaternion, dtype=torch.float32),
            time_to_go=time_to_go,
            delta=delta,
            Kx=None if Kx is None else torch.tensor(Kx),
            Kxd=None if Kxd is None else torch.tensor(Kxd),
            op_space_interp=op_space_interp,
        )

    def robot_start_joint_impedance_control(self, Kq=None, Kqd=None, adaptive=True):
        self.robot.start_joint_impedance(
            Kq=None if Kq is None else torch.tensor(Kq),
            Kqd=None if Kqd is None else torch.tensor(Kqd),
            adaptive=adaptive,
        )

    def robot_start_cartesian_impedance_control(self, Kx=None, Kxd=None):
        self.robot.start_cartesian_impedance(
            Kx=None if Kx is None else torch.tensor(Kx),
            Kxd=None if Kxd is None else torch.tensor(Kxd),
        )

    def robot_update_desired_joint_positions(self, positions):
        self.robot.update_desired_joint_positions(positions=torch.tensor(positions))

    def robot_update_desired_ee_pose(self, pose):
        from scipy.spatial.transform import Rotation

        pose = np.asarray(pose, dtype=np.float32)
        quaternion = Rotation.from_rotvec(pose[3:]).as_quat()
        self.robot.update_desired_ee_pose(
            position=torch.tensor(pose[:3]),
            orientation=torch.tensor(quaternion, dtype=torch.float32),
        )

    def robot_terminate_current_policy(self):
        self.robot.terminate_current_policy()

    def _require_gripper(self):
        if self.gripper is None:
            raise RuntimeError("Call gripper_initialize before using the gripper.")
        return self.gripper


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PreAct Franka RPC bridge on the NUC.")
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4242)
    parser.add_argument("--robot-ip", default=None)
    parser.add_argument(
        "--allow-network-rpc",
        action="store_true",
        help="Required when binding beyond loopback; Zero-RPC provides no authentication.",
    )
    args = parser.parse_args()
    if args.bind_host not in {"127.0.0.1", "localhost"} and not args.allow_network_rpc:
        raise SystemExit("Refusing unauthenticated network bind without --allow-network-rpc.")
    try:
        import zerorpc
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install zerorpc in the Franka NUC environment.") from exc
    server = zerorpc.Server(FrankaInterfaceServer(robot_ip=args.robot_ip))
    server.bind(f"tcp://{args.bind_host}:{args.port}")
    server.run()


if __name__ == "__main__":
    main()
