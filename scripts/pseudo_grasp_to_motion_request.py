#!/usr/bin/env python3
"""Convert a Task E grasp JSON into the unified motion-request format."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRASP = REPO_ROOT / "outputs/task_e_banana_pipeline/latest/pseudo_grasp/pseudo_grasp.json"

REQUEST_SCHEMA = "atec.task_e.motion_request.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo-grasp", type=Path, default=DEFAULT_GRASP)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON. Defaults to <pseudo-grasp parent>/motion_request.json.",
    )
    parser.add_argument("--camera-steps", type=int, default=160)
    parser.add_argument("--pregrasp-steps", type=int, default=180)
    parser.add_argument("--descend-steps", type=int, default=140)
    parser.add_argument("--close-steps", type=int, default=70)
    parser.add_argument("--lift-steps", type=int, default=150)
    parser.add_argument(
        "--preferred-backend",
        choices=("moveit_py", "isaaclab_cartesian_controller"),
        default="moveit_py",
        help="Preferred solver recorded in the request. Runner may fall back if unavailable.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Task E environment seed to reproduce the captured scene during execution.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.exists():
        return path.resolve()
    if "latest" in path.parts:
        latest_txt = REPO_ROOT / "outputs/task_e_banana_pipeline/latest.txt"
        if latest_txt.exists():
            latest = Path(latest_txt.read_text(encoding="utf-8").strip())
            candidate = latest / "pseudo_grasp/pseudo_grasp.json"
            if candidate.exists():
                return candidate.resolve()
    raise FileNotFoundError(f"Missing pseudo grasp JSON: {path}")


def pose_or_none(section: dict | None, fallback: dict) -> dict:
    section = section or {}
    position = section.get("position") or fallback["position"]
    quat = section.get("quat_wxyz") or fallback["quat_wxyz"]
    return {"position": position, "quat_wxyz": quat}


def waypoint(
    name: str,
    pose: dict,
    gripper: list[float],
    steps: int,
    capture: bool = True,
) -> dict:
    return {
        "name": name,
        "pose_w": {
            "position": pose["position"],
            "quat_wxyz": pose["quat_wxyz"],
        },
        "gripper_joint_pos": gripper,
        "steps": steps,
        "capture": capture,
    }


def build_request(pseudo: dict, pseudo_path: Path, args: argparse.Namespace) -> dict:
    open_gripper = pseudo["gripper"]["open_joint7_joint8"]
    closed_gripper = pseudo["gripper"]["closed_joint7_joint8"]
    grasp_pose = pseudo["grasp_pose_w"]
    pregrasp_pose = pseudo["pregrasp_pose_w"]
    lift_pose = pseudo["lift_pose_w"]
    camera_pose = pose_or_none(pseudo.get("camera_look_pose_w"), pregrasp_pose)

    return {
        "schema": REQUEST_SCHEMA,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": "ATEC-TaskE-Piper",
        "frame": "world",
        "units": {"position": "meter", "quaternion": "wxyz"},
        "source": {
            "type": "pseudo_grasp",
            "path": str(pseudo_path),
            "target": pseudo.get("target", "unknown"),
            "pose_source": pseudo.get("pose_source"),
            "target_center_w": pseudo.get("target_center_w"),
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
            waypoint("00_camera_look", camera_pose, open_gripper, args.camera_steps),
            waypoint("01_pregrasp", pregrasp_pose, open_gripper, args.pregrasp_steps),
            waypoint("02_grasp", grasp_pose, open_gripper, args.descend_steps),
            waypoint("03_close", grasp_pose, closed_gripper, args.close_steps),
            waypoint("04_lift", lift_pose, closed_gripper, args.lift_steps),
        ],
    }


def main() -> None:
    args = parse_args()
    pseudo_path = resolve_path(args.pseudo_grasp)
    pseudo = json.loads(pseudo_path.read_text(encoding="utf-8"))
    output = (args.output or (pseudo_path.parent / "motion_request.json")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    request = build_request(pseudo, pseudo_path, args)
    output.write_text(json.dumps(request, indent=2), encoding="utf-8")
    print(f"[INFO] Saved motion request: {output}")


if __name__ == "__main__":
    main()
