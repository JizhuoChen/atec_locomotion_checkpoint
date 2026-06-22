"""Verify cuRobo IK by executing a solved Piper target in Task E."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINTS = ["joint7", "joint8"]
ACTION_SCALE = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="ATEC-TaskE-Piper")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--submission-dir", default="submissions/task_e_act_baseline_root_submission")
    parser.add_argument("--output", default="outputs/task_e_curobo_ik_execution/latest")
    parser.add_argument("--target-offset-root", type=float, nargs=3, default=(0.03, -0.02, 0.03))
    parser.add_argument("--num-seeds", type=int, default=64)
    parser.add_argument("--waypoint-max-step-rad", type=float, default=0.035)
    parser.add_argument("--waypoint-hold-steps", type=int, default=10)
    parser.add_argument("--final-hold-steps", type=int, default=160)
    parser.add_argument("--success-threshold-m", type=float, default=0.005)
    parser.add_argument("--ik-position-tolerance-m", type=float, default=0.005)
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


def quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.asarray([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)


def quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return np.asarray([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


def pose_in_robot_root(robot, ee_idx: int, rotation_cls) -> tuple[np.ndarray, np.ndarray]:
    root_pose = robot.data.root_pose_w[0].detach().cpu().numpy()
    ee_pose = robot.data.body_pose_w[0, ee_idx].detach().cpu().numpy()

    root_pos = root_pose[:3].astype(np.float64)
    root_rot = rotation_cls.from_quat(quat_wxyz_to_xyzw(root_pose[3:7]))
    ee_pos_w = ee_pose[:3].astype(np.float64)
    ee_rot_w = rotation_cls.from_quat(quat_wxyz_to_xyzw(ee_pose[3:7]))
    ee_pos_root = root_rot.inv().apply(ee_pos_w - root_pos)
    ee_quat_root = quat_xyzw_to_wxyz((root_rot.inv() * ee_rot_w).as_quat())
    return ee_pos_root.astype(np.float32), ee_quat_root.astype(np.float32)


def quat_abs_dot(q0: np.ndarray, q1: np.ndarray) -> float:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 /= max(np.linalg.norm(q0), 1e-12)
    q1 /= max(np.linalg.norm(q1), 1e-12)
    return float(abs(np.dot(q0, q1)))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym  # noqa: WPS433
    import torch  # noqa: WPS433
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: WPS433
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: WPS433
    from scipy.spatial.transform import Rotation as R  # noqa: WPS433

    import atec_rl_lab.tasks  # noqa: F401, WPS433

    repo_root = Path(__file__).resolve().parents[1]
    submission_dir = (repo_root / args.submission_dir).resolve()
    sys.path.insert(0, str(submission_dir))
    from task_e_curobo_planner import TaskECuRoboPlanner, TaskECuRoboPlannerProcess  # noqa: WPS433

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    env = gym.make(args.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]
    ee_idx = robot.body_names.index("gripper_base")
    arm_ids, _ = robot.find_joints(ARM_JOINTS)
    gripper_ids, _ = robot.find_joints(GRIPPER_JOINTS)

    planner = TaskECuRoboPlannerProcess(
        num_seeds=args.num_seeds,
        position_tolerance_m=args.ik_position_tolerance_m,
    )
    start_time = time.time()
    result: dict[str, object] = {}

    try:
        with torch.inference_mode():
            current_q = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().astype(np.float32)
            current_gripper_q = robot.data.joint_pos[0, gripper_ids].detach().cpu().numpy().astype(np.float32)
            default_q = robot.data.default_joint_pos[0].detach().cpu().numpy().astype(np.float32)

        current_fk_pos, current_fk_quat = planner.compute_fk(current_q)
        target_pos = current_fk_pos + np.asarray(args.target_offset_root, dtype=np.float32)
        target_quat = current_fk_quat

        ik_result = planner.solve_ik(current_q, target_pos, target_quat, return_seeds=1)
        waypoints = TaskECuRoboPlanner.interpolate_joint_waypoints(
            current_q,
            ik_result.joint_position,
            max_step_rad=args.waypoint_max_step_rad,
        )

        final_target = default_q.copy()
        final_target[arm_ids] = ik_result.joint_position
        final_target[gripper_ids] = current_gripper_q

        def step_to_joint_target(target_joint_pos: np.ndarray, steps: int) -> None:
            nonlocal obs
            action_np = ((target_joint_pos - default_q) / ACTION_SCALE).astype(np.float32)
            action = torch.as_tensor(action_np, dtype=torch.float32, device=args.device).view(1, -1)
            for _ in range(int(steps)):
                obs, _, terminated, truncated, _ = env.step(action)
                if bool(terminated.item() or truncated.item()):
                    break

        for waypoint in waypoints:
            target = default_q.copy()
            target[arm_ids] = waypoint
            target[gripper_ids] = current_gripper_q
            step_to_joint_target(target, args.waypoint_hold_steps)
        step_to_joint_target(final_target, args.final_hold_steps)

        final_pos, final_quat = pose_in_robot_root(robot, ee_idx, R)
        final_q = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().astype(np.float32)
        final_pos_error = float(np.linalg.norm(final_pos.astype(np.float64) - target_pos.astype(np.float64)))
        final_quat_dot = quat_abs_dot(final_quat, target_quat)

        result = {
            "success_threshold_m": float(args.success_threshold_m),
            "pass": bool(ik_result.success and final_pos_error <= args.success_threshold_m),
            "target_offset_root_m": [float(v) for v in args.target_offset_root],
            "current_arm_joint_pos": current_q.astype(float).tolist(),
            "target_arm_joint_pos": ik_result.joint_position.astype(float).tolist(),
            "final_arm_joint_pos": final_q.astype(float).tolist(),
            "current_fk_pos_root": current_fk_pos.astype(float).tolist(),
            "target_pos_root": target_pos.astype(float).tolist(),
            "target_quat_wxyz_root": target_quat.astype(float).tolist(),
            "ik_success": bool(ik_result.success),
            "ik_position_error_m": float(ik_result.position_error_m),
            "ik_rotation_error_rad": float(ik_result.rotation_error_rad),
            "ik_solve_time_s": float(ik_result.solve_time_s),
            "num_waypoints": len(waypoints),
            "final_pos_root": final_pos.astype(float).tolist(),
            "final_quat_wxyz_root": final_quat.astype(float).tolist(),
            "final_position_error_m": final_pos_error,
            "final_quat_abs_dot": final_quat_dot,
            "wall_time_s": float(time.time() - start_time),
        }
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    finally:
        planner.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
