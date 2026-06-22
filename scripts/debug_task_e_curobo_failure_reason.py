#!/usr/bin/env python3
"""Replay saved CuRobo IK targets and separate convergence from scene feasibility."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", default="ATEC-TaskE-Piper")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num-seeds", type=int, default=64)
    parser.add_argument("--position-tolerance-m", type=float, default=0.025)
    parser.add_argument("--orientation-tolerance-rad", type=float, default=0.30)
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


def quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.asarray([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)


def quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return np.asarray([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


def world_pose_to_robot_root(robot, pos_w: np.ndarray, quat_wxyz: np.ndarray, rotation_cls) -> tuple[np.ndarray, np.ndarray]:
    root_pose = robot.data.root_pose_w[0].detach().cpu().numpy()
    root_pos = root_pose[:3].astype(np.float64)
    root_rot = rotation_cls.from_quat(quat_wxyz_to_xyzw(root_pose[3:7]))
    target_rot_w = rotation_cls.from_quat(quat_wxyz_to_xyzw(quat_wxyz))
    pos_root = root_rot.inv().apply(np.asarray(pos_w, dtype=np.float64) - root_pos)
    quat_root = quat_xyzw_to_wxyz((root_rot.inv() * target_rot_w).as_quat())
    return pos_root.astype(np.float32), quat_root.astype(np.float32)


def cuboid_world_to_robot_root(robot, rotation_cls, name: str, center_w: list[float], dims: list[float]) -> dict:
    pos_root, quat_root = world_pose_to_robot_root(
        robot,
        np.asarray(center_w, dtype=np.float32),
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        rotation_cls,
    )
    return {
        "name": name,
        "pose": pos_root.astype(float).tolist() + quat_root.astype(float).tolist(),
        "dims": [float(v) for v in dims],
    }


def load_candidate(selection_path: Path, rank: int) -> dict:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    for candidate in selection.get("candidates", []):
        if int(candidate.get("rank", -1)) == int(rank):
            return candidate
    raise ValueError(f"Rank {rank} not found in {selection_path}")


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym  # noqa: WPS433
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: WPS433
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: WPS433
    from scipy.spatial.transform import Rotation as R  # noqa: WPS433

    import atec_rl_lab.tasks  # noqa: F401, WPS433
    from task_e_full_baseline_request import (  # noqa: WPS433
        BASKET_CENTER_X,
        BASKET_CENTER_Y,
        TABLE_CENTER_X,
        TABLE_CENTER_Y,
        TABLE_DIMS,
        TABLE_TOP_Z,
    )

    repo_root = Path(__file__).resolve().parents[1]
    submission_dir = repo_root / "submissions/task_e_act_baseline_root_submission"
    if str(submission_dir) not in sys.path:
        sys.path.insert(0, str(submission_dir))
    from task_e_curobo_planner import TaskECuRoboPlannerProcess  # noqa: WPS433

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    planners = []
    try:
        env.reset()
        robot = env.unwrapped.scene["robot"]
        arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
        current_q0 = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().astype(np.float32)
        candidate = load_candidate(args.selection.expanduser().resolve(), args.rank)
        stages = candidate.get("stages") or []
        if not stages:
            raise ValueError(f"Candidate rank {args.rank} has no saved stages")

        table = cuboid_world_to_robot_root(
            robot,
            R,
            "table",
            [float(TABLE_CENTER_X), float(TABLE_CENTER_Y), float(TABLE_TOP_Z * 0.5)],
            [float(TABLE_DIMS[0] + 0.04), float(TABLE_DIMS[1] + 0.04), float(TABLE_TOP_Z)],
        )
        basket = cuboid_world_to_robot_root(
            robot,
            R,
            "basket_outer",
            [float(BASKET_CENTER_X), float(BASKET_CENTER_Y), float(TABLE_TOP_Z + 0.075)],
            [0.46, 0.30, 0.15],
        )

        variants = [
            ("no_scene_collision", False, []),
            ("scene_empty", True, []),
            ("scene_table", True, [table]),
            ("scene_table_basket", True, [table, basket]),
        ]
        summary = {
            "selection": str(args.selection.expanduser().resolve()),
            "rank": int(args.rank),
            "initial_arm_q": current_q0.astype(float).tolist(),
            "candidate_saved_summary": {
                key: candidate.get(key)
                for key in (
                    "ok",
                    "all_ik_success",
                    "position_ok",
                    "rotation_ok",
                    "accepted_by_policy",
                    "max_ik_position_error_m",
                    "max_ik_rotation_error_rad",
                )
            },
            "cuboids_root": {"table": table, "basket_outer": basket},
            "variants": {},
        }

        for name, scene_collision, cuboids in variants:
            planner = TaskECuRoboPlannerProcess(
                num_seeds=args.num_seeds,
                position_tolerance_m=args.position_tolerance_m,
                orientation_tolerance_rad=args.orientation_tolerance_rad,
                request_timeout_s=args.request_timeout_s,
                scene_collision_check=scene_collision,
                self_collision_check=False,
            )
            planners.append(planner)
            if scene_collision and cuboids:
                planner.update_world_cuboids(cuboids)
            q = current_q0.copy()
            records = []
            all_success = True
            all_feasible = True
            for stage in stages:
                result = planner.solve_ik(
                    q,
                    stage["target_pos_root"],
                    stage["target_quat_root_wxyz"],
                    return_seeds=1,
                )
                q = result.joint_position.astype(np.float32)
                all_success = all_success and bool(result.success)
                all_feasible = all_feasible and bool(result.feasible)
                records.append(
                    {
                        "label": stage.get("label"),
                        "success": bool(result.success),
                        "feasible": result.feasible,
                        "position_error_m": float(result.position_error_m),
                        "rotation_error_rad": float(result.rotation_error_rad),
                        "joint_position": result.joint_position.astype(float).tolist(),
                    }
                )
            summary["variants"][name] = {
                "scene_collision_check": bool(scene_collision),
                "cuboids": [cuboid["name"] for cuboid in cuboids],
                "all_success": bool(all_success),
                "all_feasible": bool(all_feasible),
                "stages": records,
            }

        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        for planner in planners:
            planner.close()
        env.close()
        try:
            simulation_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
