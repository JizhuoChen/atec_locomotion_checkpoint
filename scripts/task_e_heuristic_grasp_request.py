#!/usr/bin/env python3
"""Build a Task E motion request from prepared heuristic grasp candidates."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from task_e_full_anygrasp_request import (  # noqa: E402
    DEFAULT_PICK_ORDER,
    OBJECTS,
    TABLE_TOP_Z,
    build_request,
    deterministic_object_poses,
    parse_object_order,
    prepare_output_dir,
)


DEFAULT_CANDIDATE_ROOT = (
    REPO_ROOT
    / "outputs/task_e_ideal_ee_camera_debug/20260614_same_env_lookat_real_ee/"
    "grasp_candidates_mixed_ee_banana_all_others"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/task_e_heuristic_physics_request/latest",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--object-order",
        type=parse_object_order,
        default=list(DEFAULT_PICK_ORDER),
        help="Comma-separated pick order. Default: banana,mustard_bottle,box_object.",
    )
    parser.add_argument(
        "--object-pose-source",
        choices=("seed_reset", "heuristic_contact_xy", "debug_request"),
        default="seed_reset",
        help=(
            "seed_reset leaves source.objects empty so the runner uses the Task E seed reset. "
            "heuristic_contact_xy embeds approximate object centers from heuristic contact XY. "
            "debug_request embeds the target centers saved in the camera/grasp debug capture."
        ),
    )
    parser.add_argument("--object-transport-mode", choices=("physics", "kinematic_attach"), default="physics")
    parser.add_argument("--pregrasp-distance", type=float, default=0.12)
    parser.add_argument("--lift-distance", type=float, default=0.18)
    parser.add_argument("--init-steps", type=int, default=100)
    parser.add_argument("--staging-steps", type=int, default=140)
    parser.add_argument("--pregrasp-steps", type=int, default=140)
    parser.add_argument("--grasp-steps", type=int, default=120)
    parser.add_argument("--close-steps", type=int, default=70)
    parser.add_argument("--lift-steps", type=int, default=140)
    parser.add_argument("--transport-steps", type=int, default=230)
    parser.add_argument("--place-steps", type=int, default=110)
    parser.add_argument("--open-steps", type=int, default=80)
    parser.add_argument("--retract-steps", type=int, default=100)
    parser.add_argument(
        "--closed-gripper-joints",
        type=parse_joint_pair,
        default=None,
        help="Override closed gripper joint target, e.g. 0.005,-0.005 for a softer close probe.",
    )
    parser.add_argument(
        "--grasp-offset",
        type=parse_grasp_offset,
        action="append",
        default=[],
        metavar="OBJECT:DX,DY,DZ",
        help="Offset one object's executed heuristic grasp pose in metres, e.g. box_object:-0.02,0,0.",
    )
    parser.add_argument(
        "--truncate-after-waypoint",
        default=None,
        help="Drop waypoints after this name. Useful for short contact/IK diagnostics.",
    )
    return parser.parse_args()


def parse_joint_pair(value: str) -> list[float]:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected joint7,joint8")
    return parts


def parse_grasp_offset(value: str) -> tuple[str, list[float]]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("expected OBJECT:DX,DY,DZ")
    name, raw = value.split(":", 1)
    name = name.strip()
    if name not in OBJECTS:
        raise argparse.ArgumentTypeError(f"unknown object {name!r}")
    parts = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three offsets: DX,DY,DZ")
    return name, parts


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def postprocess_waypoints(request: dict, args: argparse.Namespace) -> None:
    if args.closed_gripper_joints is not None:
        for waypoint in request["waypoints"]:
            joints = waypoint.get("gripper_joint_pos")
            if joints == [-0.015, 0.015]:
                waypoint["gripper_joint_pos"] = list(args.closed_gripper_joints)
        request["controller"]["closed_gripper_joint_override"] = list(args.closed_gripper_joints)

    if args.truncate_after_waypoint:
        truncated = []
        found = False
        for waypoint in request["waypoints"]:
            truncated.append(waypoint)
            if waypoint["name"] == args.truncate_after_waypoint:
                found = True
                break
        if not found:
            raise ValueError(f"truncate waypoint not found: {args.truncate_after_waypoint}")
        request["waypoints"] = truncated
        request["source"]["truncated_after_waypoint"] = args.truncate_after_waypoint


def heuristic_execution_pose(candidate_root: Path, object_name: str) -> tuple[Path, dict]:
    pose_path = candidate_root / object_name / "heuristic" / "final_grasp_pose.json"
    payload = load_json(pose_path)
    execution = payload.get("execution_pose_world")
    if not isinstance(execution, dict) or "translation" not in execution:
        raise ValueError(f"{pose_path} does not contain execution_pose_world.translation")
    pose = {
        **payload,
        "translation": execution["translation"],
        "quat_wxyz": execution.get("quat_wxyz", payload.get("quat_wxyz")),
        "pose_type": "heuristic_top_down_execution_pose",
        "contact_pose_world": {
            "translation": payload["translation"],
            "quat_wxyz": payload.get("quat_wxyz"),
        },
        "request_translation_source": "execution_pose_world.translation",
        "pregrasp_distance_override_m": float(payload.get("pregrasp_distance_override_m", 0.12)),
        "approach_lift_distance_override_m": float(payload.get("approach_lift_distance_override_m", 0.18)),
    }
    return pose_path, pose


def apply_grasp_offset(final_pose: dict, offset: list[float]) -> None:
    delta = [float(v) for v in offset]
    for key in ("translation", "object_center_w"):
        value = final_pose.get(key)
        if isinstance(value, list) and len(value) >= 3:
            final_pose[key] = [float(value[i]) + delta[i] for i in range(3)]
    for nested_key in ("execution_pose_world", "contact_pose_world"):
        nested = final_pose.get(nested_key)
        if isinstance(nested, dict):
            value = nested.get("translation")
            if isinstance(value, list) and len(value) >= 3:
                nested["translation"] = [float(value[i]) + delta[i] for i in range(3)]
    final_pose["grasp_offset_override_m"] = delta


def candidate_source_root(candidate_root: Path) -> Path:
    summary_path = candidate_root / "summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        source_root = summary.get("source_root")
        if source_root:
            return Path(source_root).expanduser().resolve()
    return candidate_root.parent.resolve()


def debug_request_center(candidate_root: Path, object_name: str) -> list[float]:
    pose_path = candidate_source_root(candidate_root) / object_name / "intended_camera_pose.json"
    payload = load_json(pose_path)
    center = payload.get("request_object_center_w")
    if center is None:
        center = (payload.get("intended_camera_pose") or {}).get("target_object_center_w")
    if center is None or len(center) < 3:
        raise ValueError(f"{pose_path} does not contain request_object_center_w")
    return [float(v) for v in center[:3]]


def object_poses_for_request(
    args: argparse.Namespace,
    candidate_root: Path,
    final_poses: dict[str, dict],
) -> dict[str, dict]:
    poses = deterministic_object_poses(args.seed)
    if args.object_pose_source == "debug_request":
        for name in final_poses:
            poses[name]["center_w"] = debug_request_center(candidate_root, name)
            poses[name]["source"] = "debug_intended_camera_pose.request_object_center_w"
    elif args.object_pose_source == "heuristic_contact_xy":
        for name, final_pose in final_poses.items():
            contact = final_pose.get("contact_pose_world", {}).get("translation") or final_pose["translation"]
            poses[name]["center_w"] = [
                float(contact[0]),
                float(contact[1]),
                float(TABLE_TOP_Z + 0.05),
            ]
            poses[name]["source"] = "heuristic_contact_xy"
    return poses


def main() -> None:
    args = parse_args()
    candidate_root = args.candidate_root.expanduser().resolve()
    output_dir = prepare_output_dir(args.output)
    grasp_offsets = dict(args.grasp_offset)

    final_poses: dict[str, dict] = {}
    grasp_records: dict[str, dict] = {}
    for name in args.object_order:
        pose_path, final_pose = heuristic_execution_pose(candidate_root, name)
        if name in grasp_offsets:
            apply_grasp_offset(final_pose, grasp_offsets[name])
        final_pose["pregrasp_distance_override_m"] = args.pregrasp_distance
        final_pose["approach_lift_distance_override_m"] = args.lift_distance
        final_poses[name] = final_pose
        heuristic_dir = pose_path.parent
        result_path = heuristic_dir / "heuristic_result.json"
        grasp_records[name] = {
            "sam3": {
                "prompt": name,
                "source": "precomputed_mixed_fusion_debug",
                "mask_path": None,
            },
            "mask_center_estimate": {
                "source": "heuristic_final_grasp_pose.object_center_w",
                "center_world_median": final_pose.get("object_center_w"),
                "point_count": None,
            },
            "anygrasp_result_path": result_path,
            "final_grasp_pose_path": pose_path,
            "anygrasp_status": "ok",
            "final_grasp_pose": final_pose,
        }

    request_args = SimpleNamespace(
        input=candidate_root,
        object_order=list(args.object_order),
        object_center_source=args.object_pose_source,
        grasp_mode="raw_anygrasp",
        raw_tool_transform="identity",
        gripper_base_offset=0.0,
        gripper_base_offset_mode="towards_object_center",
        approach_axis_column=2,
        grasp_tuning={},
        pregrasp_distance=args.pregrasp_distance,
        lift_distance=args.lift_distance,
        object_transport_mode=args.object_transport_mode,
        actuator_mode="task_e_scripted_high_stiffness",
        preferred_backend="moveit_py",
        seed=args.seed,
        init_steps=args.init_steps,
        staging_steps=args.staging_steps,
        pregrasp_steps=args.pregrasp_steps,
        grasp_steps=args.grasp_steps,
        close_steps=args.close_steps,
        lift_steps=args.lift_steps,
        transport_steps=args.transport_steps,
        place_steps=args.place_steps,
        open_steps=args.open_steps,
        retract_steps=args.retract_steps,
    )
    object_poses = object_poses_for_request(args, candidate_root, final_poses)
    request = build_request(request_args, output_dir, object_poses, grasp_records)
    request["source"]["type"] = "heuristic_mixed_fusion_debug"
    request["source"]["input_dir"] = str(candidate_root)
    request["source"]["grasp_pose_source"] = "heuristic_top_down_execution_pose"
    request["source"]["note"] = (
        "Physics-mode request built from precomputed heuristic grasps. "
        "The waypoint grasp pose uses execution_pose_world.translation; "
        "contact_pose_world preserves the visual/contact target."
    )
    if args.object_pose_source == "seed_reset":
        request["source"]["objects_for_reference"] = request["source"].pop("objects", [])
        request["source"]["objects"] = []
        request["source"]["object_pose_source"] = "task_e_seed_reset_no_manual_object_pose_write"
    request["controller"]["object_transport_mode"] = args.object_transport_mode
    request["controller"]["grasp_quat_source"] = "heuristic_top_down_execution_pose"
    request["start_state"] = {"source": "task_reset", "seed": args.seed}
    postprocess_waypoints(request, args)

    request_path = output_dir / "motion_request.json"
    summary_path = output_dir / "pipeline_summary.json"
    write_json(request_path, request)
    write_json(
        summary_path,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "candidate_root": str(candidate_root),
            "request": str(request_path),
            "object_order": list(args.object_order),
            "object_pose_source": args.object_pose_source,
            "object_transport_mode": args.object_transport_mode,
            "grasp_mode": request_args.grasp_mode,
            "approach_axis_column": request_args.approach_axis_column,
            "objects": request.get("source", {}).get("objects_for_reference")
            or request.get("source", {}).get("objects"),
        },
    )
    print(f"[INFO] Saved heuristic motion request: {request_path}")
    print(f"[INFO] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
