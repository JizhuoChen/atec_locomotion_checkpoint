"""Compare generated cuRobo Piper FK against IsaacLab Task E body poses."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="ATEC-TaskE-Piper")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--submission-dir", default="submissions/task_e_act_baseline_root_submission")
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


def import_planner_module(submission_dir: Path):
    sys.path.insert(0, str(submission_dir))
    spec = importlib.util.spec_from_file_location(
        "task_e_curobo_planner", submission_dir / "task_e_curobo_planner.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import planner from {submission_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.asarray([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)


def quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return np.asarray([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


def main() -> None:
    args = parse_args()
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
    planner_module = import_planner_module(submission_dir)
    planner = planner_module.TaskECuRoboPlannerProcess(num_seeds=16)

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    env = gym.make(args.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    ee_idx = robot.body_names.index("gripper_base")
    arm_ids, _ = robot.find_joints(["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"])

    with torch.inference_mode():
        q = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy()
        root_pose = robot.data.root_pose_w[0].detach().cpu().numpy()
        ee_pose = robot.data.body_pose_w[0, ee_idx].detach().cpu().numpy()

    fk_pos, fk_quat = planner.compute_fk(q)

    root_pos = root_pose[:3].astype(np.float64)
    root_rot = R.from_quat(quat_wxyz_to_xyzw(root_pose[3:7]))
    ee_pos_w = ee_pose[:3].astype(np.float64)
    ee_rot_w = R.from_quat(quat_wxyz_to_xyzw(ee_pose[3:7]))
    ee_pos_root = root_rot.inv().apply(ee_pos_w - root_pos)
    ee_quat_root = quat_xyzw_to_wxyz((root_rot.inv() * ee_rot_w).as_quat())

    pos_error = float(np.linalg.norm(np.asarray(fk_pos, dtype=np.float64) - ee_pos_root))
    quat_dot = float(abs(np.dot(np.asarray(fk_quat, dtype=np.float64), ee_quat_root)))

    print(f"joint_pos={q.tolist()}", flush=True)
    print(f"curobo_fk_pos={fk_pos.tolist()}", flush=True)
    print(f"sim_root_frame_pos={ee_pos_root.tolist()}", flush=True)
    print(f"position_error_m={pos_error:.8f}", flush=True)
    print(f"curobo_fk_quat_wxyz={fk_quat.tolist()}", flush=True)
    print(f"sim_root_frame_quat_wxyz={ee_quat_root.tolist()}", flush=True)
    print(f"quat_abs_dot={quat_dot:.8f}", flush=True)

    planner.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
    repo_root = Path(__file__).resolve().parents[1]
    submission_dir = (repo_root / args.submission_dir).resolve()
    planner_module = import_planner_module(submission_dir)
    planner = planner_module.TaskECuRoboPlannerProcess(num_seeds=16)
