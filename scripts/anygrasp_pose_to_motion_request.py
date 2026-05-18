#!/usr/bin/env python3
"""Convert AnyGrasp final_grasp_pose.json into a Task E motion request."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRASP = REPO_ROOT / "outputs/task_e_banana_pipeline/latest/anygrasp/final_grasp_pose.json"
REQUEST_SCHEMA = "atec.task_e.motion_request.v1"
GRIPPER_OPEN = [0.035, -0.035]
GRIPPER_CLOSE = [-0.015, 0.015]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anygrasp-pose", type=Path, default=DEFAULT_GRASP)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON. Defaults to <anygrasp-pose parent>/motion_request.json.",
    )
    parser.add_argument("--target", default="banana")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pregrasp-distance", type=float, default=0.12)
    parser.add_argument("--lift-distance", type=float, default=0.18)
    parser.add_argument("--camera-steps", type=int, default=120)
    parser.add_argument("--pregrasp-steps", type=int, default=180)
    parser.add_argument("--descend-steps", type=int, default=140)
    parser.add_argument("--close-steps", type=int, default=70)
    parser.add_argument("--lift-steps", type=int, default=150)
    parser.add_argument(
        "--approach-axis-column",
        type=int,
        default=0,
        choices=(0, 1, 2),
        help="AnyGrasp rotation matrix column used as the approach axis. GraspNet geometry uses column 0.",
    )
    parser.add_argument(
        "--preferred-backend",
        choices=("moveit_py", "isaaclab_cartesian_controller"),
        default="moveit_py",
    )
    return parser.parse_args()


def quat_normalize(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def quat_wxyz_from_matrix(matrix: np.ndarray) -> list[float]:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 0.0)) * 2.0
        quat = np.array(
            [
                (m[2, 1] - m[1, 2]) / s,
                0.25 * s,
                (m[0, 1] + m[1, 0]) / s,
                (m[0, 2] + m[2, 0]) / s,
            ],
            dtype=np.float64,
        )
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 0.0)) * 2.0
        quat = np.array(
            [
                (m[0, 2] - m[2, 0]) / s,
                (m[0, 1] + m[1, 0]) / s,
                0.25 * s,
                (m[1, 2] + m[2, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        s = np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 0.0)) * 2.0
        quat = np.array(
            [
                (m[1, 0] - m[0, 1]) / s,
                (m[0, 2] + m[2, 0]) / s,
                (m[1, 2] + m[2, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )
    return quat_normalize(quat).astype(float).tolist()


def pose(position: np.ndarray, quat_wxyz: list[float]) -> dict:
    return {
        "position": np.asarray(position, dtype=np.float64).astype(float).tolist(),
        "quat_wxyz": [float(value) for value in quat_wxyz],
    }


def waypoint(name: str, pose_w: dict, gripper: list[float], steps: int, capture: bool = True) -> dict:
    return {
        "name": name,
        "pose_w": pose_w,
        "gripper_joint_pos": list(gripper),
        "steps": int(steps),
        "capture": bool(capture),
    }


def load_anygrasp_pose(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("frame") != "world":
        raise ValueError(f"Expected world-frame AnyGrasp pose, got {payload.get('frame')!r}")
    if "translation" not in payload or "rotation_matrix" not in payload:
        raise ValueError("AnyGrasp pose must contain translation and rotation_matrix.")
    return payload


def build_request(grasp: dict, grasp_path: Path, args: argparse.Namespace) -> dict:
    translation = np.asarray(grasp["translation"], dtype=np.float64)
    rotation = np.asarray(grasp["rotation_matrix"], dtype=np.float64)
    quat = quat_wxyz_from_matrix(rotation)

    approach_axis = rotation[:, args.approach_axis_column]
    approach_axis = approach_axis / max(np.linalg.norm(approach_axis), 1e-9)
    pregrasp_position = translation - approach_axis * float(args.pregrasp_distance)
    lift_position = translation - approach_axis * float(args.lift_distance)
    camera_position = pregrasp_position

    grasp_pose = pose(translation, quat)
    pregrasp_pose = pose(pregrasp_position, quat)
    lift_pose = pose(lift_position, quat)
    camera_pose = pose(camera_position, quat)

    return {
        "schema": REQUEST_SCHEMA,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": "ATEC-TaskE-Piper",
        "frame": "world",
        "units": {"position": "meter", "quaternion": "wxyz"},
        "source": {
            "type": "anygrasp_final_pose",
            "path": str(grasp_path),
            "target": args.target,
            "score": grasp.get("score"),
            "width": grasp.get("width"),
            "depth": grasp.get("depth"),
            "pose_type": grasp.get("pose_type", "anygrasp_gripper"),
            "approach_axis_column": args.approach_axis_column,
            "note": (
                "AnyGrasp outputs a GraspNet gripper-frame pose. This request uses it "
                "directly as the Piper gripper_base target for plumbing tests; calibrate "
                "the fixed AnyGrasp-to-Piper tool transform before relying on execution."
            ),
        },
        "backend": {
            "preferred": args.preferred_backend,
            "fallback": "isaaclab_cartesian_controller",
        },
        "robot": {
            "name": "piper",
            "planning_group": "piper_arm",
            "ee_link": "gripper_base",
            "arm_joints": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            "gripper_joints": ["joint7", "joint8"],
            "action_scale": 0.5,
        },
        "start_state": {"source": "task_reset", "seed": args.seed},
        "waypoints": [
            waypoint("00_camera_look", camera_pose, GRIPPER_OPEN, args.camera_steps),
            waypoint("01_pregrasp", pregrasp_pose, GRIPPER_OPEN, args.pregrasp_steps),
            waypoint("02_grasp", grasp_pose, GRIPPER_OPEN, args.descend_steps),
            waypoint("03_close", grasp_pose, GRIPPER_CLOSE, args.close_steps),
            waypoint("04_lift", lift_pose, GRIPPER_CLOSE, args.lift_steps),
        ],
    }


def main() -> None:
    args = parse_args()
    grasp_path = args.anygrasp_pose.expanduser().resolve()
    grasp = load_anygrasp_pose(grasp_path)
    output = (args.output or (grasp_path.parent / "motion_request.json")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    request = build_request(grasp, grasp_path, args)
    output.write_text(json.dumps(request, indent=2), encoding="utf-8")
    print(f"[INFO] Saved motion request: {output}")


if __name__ == "__main__":
    main()
