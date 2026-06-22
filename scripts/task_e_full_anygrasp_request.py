#!/usr/bin/env python3
"""Create a full Task E motion request from video-camera SAM3 + AnyGrasp results."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from task_e_full_baseline_request import (  # noqa: E402
    BASKET_CENTER_X,
    BASKET_CENTER_Y,
    CARRY_Z,
    DEFAULT_PICK_ORDER,
    DEFAULT_QUAT_WXYZ,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    OBJECTS,
    PLACE_HEIGHT,
    REQUEST_SCHEMA,
    RETRACT_POS_X,
    RETRACT_POS_Y,
    TABLE_TOP_Z,
    deterministic_object_poses,
    parse_object_order,
    parse_grasp_tuning,
    pose,
    quat_wxyz_from_matrix,
    task_e_grasp_quat,
)


OBJECT_PROMPTS = {
    "banana": "banana",
    "mustard_bottle": "mustard bottle",
    "box_object": "yellow and white box",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "outputs/task_e_banana_pipeline/latest",
        help="Directory containing video_rgb.png, video_depth.npy, and video_camera.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/task_e_full_anygrasp/<timestamp>.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--object-order",
        type=parse_object_order,
        default=list(DEFAULT_PICK_ORDER),
        help="Comma-separated pick order. Default: banana,mustard_bottle,box_object.",
    )
    parser.add_argument(
        "--object-center-source",
        choices=("deterministic", "mask_median"),
        default="deterministic",
        help="Object centers embedded for runner reset. Grasp poses still come from AnyGrasp.",
    )
    parser.add_argument(
        "--grasp-mode",
        choices=("raw_anygrasp", "topdown_position"),
        default="raw_anygrasp",
        help=(
            "raw_anygrasp uses AnyGrasp translation and rotation directly. "
            "topdown_position uses AnyGrasp translation with the existing Task E top-down orientation."
        ),
    )
    parser.add_argument(
        "--raw-tool-transform",
        choices=("identity", "graspnet_to_piper_z"),
        default="identity",
        help=(
            "Frame mapping for raw_anygrasp. graspnet_to_piper_z maps AnyGrasp +X approach "
            "onto Piper gripper_base +Z, which is the approach axis used by the Task E top-down pose."
        ),
    )
    parser.add_argument(
        "--gripper-base-offset",
        type=float,
        default=0.0,
        help=(
            "Offset the Piper gripper_base from the AnyGrasp contact pose. "
            "Useful because AnyGrasp reports a gripper/contact frame, not necessarily Piper gripper_base."
        ),
    )
    parser.add_argument(
        "--gripper-base-offset-mode",
        choices=("approach_axis", "finger_centerline", "yellow_line", "towards_object_center"),
        default="towards_object_center",
        help=(
            "approach_axis preserves the old fixed-tool behavior. finger_centerline/yellow_line "
            "shifts along the Piper finger centerline direction. towards_object_center shifts "
            "from the generated grasp pose toward the object center."
        ),
    )
    parser.add_argument(
        "--grasp-tuning",
        type=parse_grasp_tuning,
        default={},
        help=(
            "Optional object:dx,dy,dz_override entries applied after AnyGrasp. "
            "dz_override is an offset above the embedded object center."
        ),
    )
    parser.add_argument(
        "--object-transport-mode",
        choices=("physics", "kinematic_attach"),
        default="physics",
        help="Written into request.controller.object_transport_mode.",
    )
    parser.add_argument(
        "--actuator-mode",
        choices=("default", "task_e_scripted_high_stiffness"),
        default="task_e_scripted_high_stiffness",
    )
    parser.add_argument("--preferred-backend", choices=("moveit_py", "isaaclab_cartesian_controller"), default="moveit_py")
    parser.add_argument("--sam3-env", default="sam3_full")
    parser.add_argument("--anygrasp-env", default="anygrasp")
    parser.add_argument("--sam3-device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--approach-axis-column", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--pregrasp-distance", type=float, default=0.12)
    parser.add_argument("--lift-distance", type=float, default=0.18)
    parser.add_argument("--force", action="store_true", help="Rerun SAM3 and AnyGrasp even when outputs already exist.")
    parser.add_argument("--no-sam3", action="store_true", help="Require existing masks under <output>/sam3.")
    parser.add_argument("--no-anygrasp", action="store_true", help="Require existing final_grasp_pose.json files.")
    parser.add_argument("--init-steps", type=int, default=100)
    parser.add_argument("--staging-steps", type=int, default=140)
    parser.add_argument("--pregrasp-steps", type=int, default=120)
    parser.add_argument("--grasp-steps", type=int, default=100)
    parser.add_argument("--close-steps", type=int, default=55)
    parser.add_argument("--lift-steps", type=int, default=120)
    parser.add_argument("--transport-steps", type=int, default=210)
    parser.add_argument("--place-steps", type=int, default=100)
    parser.add_argument("--open-steps", type=int, default=70)
    parser.add_argument("--retract-steps", type=int, default=90)
    return parser.parse_args()


def prepare_output_dir(path: Path | None) -> Path:
    if path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPO_ROOT / "outputs/task_e_full_anygrasp" / timestamp
    path.mkdir(parents=True, exist_ok=True)
    latest_txt = path.parent / "latest.txt"
    latest_txt.write_text(str(path.resolve()), encoding="utf-8")
    latest_link = path.parent / "latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(path.resolve(), target_is_directory=True)
    except OSError:
        pass
    return path.resolve()


def require_input(input_dir: Path) -> tuple[Path, Path, Path]:
    rgb = input_dir / "video_rgb.png"
    depth = input_dir / "video_depth.npy"
    camera = input_dir / "video_camera.json"
    missing = [path for path in (rgb, depth, camera) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input artifacts: {', '.join(str(path) for path in missing)}")
    return rgb.resolve(), depth.resolve(), camera.resolve()


def run_command(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    print(f"[INFO] Running: {' '.join(command)}", flush=True)
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit {proc.returncode}. See {log_path}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L")
    width, height = shape[1], shape[0]
    if mask.size != (width, height):
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(mask) > 127


def quat_wxyz_to_matrix(quat: list[float] | np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def estimate_mask_center(depth_path: Path, mask_path: Path, camera_path: Path, max_depth: float) -> dict:
    depth = np.load(depth_path).astype(np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[:, :, 0]
    camera_payload = load_json(camera_path)
    camera = camera_payload.get("camera", camera_payload)
    intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float64)
    mask = load_mask(mask_path, depth.shape)
    valid = mask & np.isfinite(depth) & (depth > 0.0) & (depth < float(max_depth))
    if not valid.any():
        raise ValueError(f"Mask has no valid depth pixels: {mask_path}")
    yy, xx = np.where(valid)
    z = depth[yy, xx].astype(np.float64)
    points_cam = np.stack(
        [
            (xx.astype(np.float64) - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (yy.astype(np.float64) - intrinsic[1, 2]) * z / intrinsic[1, 1],
            z,
        ],
        axis=1,
    )
    rot_wc = quat_wxyz_to_matrix(camera["quat_w_ros"])
    pos_w = np.asarray(camera["pos_w"], dtype=np.float64)
    points_w = points_cam @ rot_wc.T + pos_w
    return {
        "center_world_median": np.median(points_w, axis=0).astype(float).tolist(),
        "center_world_mean": np.mean(points_w, axis=0).astype(float).tolist(),
        "point_count": int(points_w.shape[0]),
        "bbox_xyxy": [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())],
    }


def run_sam3_for_object(
    args: argparse.Namespace,
    rgb: Path,
    output_dir: Path,
    name: str,
) -> tuple[Path, dict]:
    label = f"video_{name}"
    sam3_dir = output_dir / "sam3" / name
    mask_path = sam3_dir / f"{label}_mask.png"
    detection_path = sam3_dir / f"{label}_detections.json"
    if args.no_sam3:
        if not mask_path.exists() or not detection_path.exists():
            raise FileNotFoundError(f"Missing existing SAM3 output for {name}: {mask_path}")
        return mask_path, load_json(detection_path)
    if args.force or not mask_path.exists() or not detection_path.exists():
        sam3_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "conda",
            "run",
            "-n",
            args.sam3_env,
            "python",
            "scripts/sam3_single_image_mask.py",
            "--image",
            str(rgb),
            "--prompt",
            OBJECT_PROMPTS[name],
            "--label",
            label,
            "--view-label",
            "eye-to-hand / video_cam",
            "--output",
            str(sam3_dir),
            "--device",
            args.sam3_device,
        ]
        run_command(command, sam3_dir / "sam3.log")
    return mask_path, load_json(detection_path)


def run_anygrasp_for_object(
    args: argparse.Namespace,
    rgb: Path,
    depth: Path,
    camera: Path,
    mask: Path,
    output_dir: Path,
    name: str,
) -> tuple[Path, dict]:
    anygrasp_dir = output_dir / "anygrasp" / name
    pose_path = anygrasp_dir / "final_grasp_pose.json"
    result_path = anygrasp_dir / "anygrasp_result.json"
    if args.no_anygrasp:
        if not pose_path.exists() or not result_path.exists():
            raise FileNotFoundError(f"Missing existing AnyGrasp output for {name}: {pose_path}")
        return pose_path, load_json(result_path)
    if args.force or not pose_path.exists() or not result_path.exists():
        anygrasp_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        openssl_lib = REPO_ROOT / "third_party/openssl11/lib"
        current_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{openssl_lib}:{current_ld}" if current_ld else str(openssl_lib)
        command = [
            "conda",
            "run",
            "-n",
            args.anygrasp_env,
            "python",
            "scripts/anygrasp_from_rgbd_mask.py",
            "--rgb",
            str(rgb),
            "--depth-npy",
            str(depth),
            "--mask",
            str(mask),
            "--camera-json",
            str(camera),
            "--output",
            str(anygrasp_dir),
            "--max-depth",
            str(args.max_depth),
        ]
        run_command(command, anygrasp_dir / "anygrasp.log", env=env)
    return pose_path, load_json(result_path)


def waypoint(
    name: str,
    position: np.ndarray | list[float],
    quat: list[float],
    gripper: list[float],
    steps: int,
    object_transport: dict | None = None,
    hold_current_pose: bool = False,
) -> dict:
    item = {
        "name": name,
        "pose_w": pose(np.asarray(position, dtype=np.float64).astype(float).tolist(), quat),
        "gripper_joint_pos": list(gripper),
        "steps": int(steps),
        "capture": True,
    }
    if hold_current_pose:
        item["hold_current_pose"] = True
    if object_transport is not None:
        item["object_transport"] = object_transport
    return item


def normalize_vector(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        return fallback.astype(np.float64)
    return value.astype(np.float64) / norm


def grasp_pose_for_object(
    args: argparse.Namespace,
    name: str,
    anygrasp_pose: dict,
    object_pose: dict,
) -> tuple[np.ndarray, list[float], np.ndarray, dict]:
    cfg = OBJECTS[name]
    translation = np.asarray(anygrasp_pose["translation"], dtype=np.float64)
    rotation = np.asarray(anygrasp_pose["rotation_matrix"], dtype=np.float64)
    if args.grasp_mode == "raw_anygrasp":
        if args.raw_tool_transform == "graspnet_to_piper_z":
            piper_rotation = np.stack(
                [-rotation[:, 2], rotation[:, 1], rotation[:, 0]],
                axis=1,
            )
            approach_axis = normalize_vector(
                piper_rotation[:, 2],
                np.array([0.0, 0.0, -1.0], dtype=np.float64),
            )
        else:
            piper_rotation = rotation
            approach_axis = normalize_vector(
                rotation[:, args.approach_axis_column],
                np.array([0.0, 0.0, -1.0], dtype=np.float64),
            )
        offset_value = float(anygrasp_pose.get("gripper_base_offset_override_m", args.gripper_base_offset))
        offset_mode = anygrasp_pose.get(
            "gripper_base_offset_mode_override",
            getattr(args, "gripper_base_offset_mode", "towards_object_center"),
        )
        if offset_value:
            if offset_mode == "towards_object_center":
                center = np.asarray(object_pose["center_w"], dtype=np.float64)
                offset_axis = normalize_vector(center - translation, -approach_axis)
                translation = translation + offset_axis * offset_value
            elif offset_mode in {"finger_centerline", "yellow_line"}:
                offset_axis = approach_axis
                translation = translation + offset_axis * offset_value
            else:
                offset_axis = -approach_axis
                translation = translation + offset_axis * offset_value
        else:
            offset_axis = np.zeros(3, dtype=np.float64)
        quat = quat_wxyz_from_matrix(piper_rotation)
        note = {
            "grasp_mode": "raw_anygrasp",
            "approach_axis_column": args.approach_axis_column,
            "raw_tool_transform": args.raw_tool_transform,
            "gripper_base_offset": offset_value,
            "gripper_base_offset_mode": offset_mode,
            "gripper_base_offset_axis_w": offset_axis.astype(float).tolist(),
            "warning": (
                "AnyGrasp reports a gripper/contact frame. The executed Piper gripper_base "
                "position is offset from that generated grasp pose according to "
                "gripper_base_offset_mode."
            ),
        }
    else:
        center = np.asarray(object_pose["center_w"], dtype=np.float64)
        grasp_z = max(float(translation[2]), float(center[2] + cfg["grasp_z_offset"]))
        translation = np.array([translation[0], translation[1], grasp_z], dtype=np.float64)
        quat = task_e_grasp_quat(object_pose["quat_wxyz"])
        approach_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        note = {
            "grasp_mode": "topdown_position",
            "warning": "Uses AnyGrasp translation but constrains orientation to the existing Task E top-down grasp.",
        }
    tuning = args.grasp_tuning.get(name, {})
    if tuning:
        center = np.asarray(object_pose["center_w"], dtype=np.float64)
        translation[0] += float(tuning.get("dx", 0.0))
        translation[1] += float(tuning.get("dy", 0.0))
        if "z_offset" in tuning:
            translation[2] = float(center[2] + float(tuning["z_offset"]))
        note["grasp_tuning"] = {
            "dx": float(tuning.get("dx", 0.0)),
            "dy": float(tuning.get("dy", 0.0)),
            "z_offset": float(tuning["z_offset"]) if "z_offset" in tuning else None,
        }
    return translation, quat, approach_axis, note


def build_request(
    args: argparse.Namespace,
    output_dir: Path,
    object_poses: dict[str, dict],
    grasp_records: dict[str, dict],
) -> dict:
    waypoints = [
        waypoint(
            "00_initial_retract",
            [RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z],
            DEFAULT_QUAT_WXYZ,
            GRIPPER_OPEN,
            args.init_steps,
        )
    ]
    source_objects = []

    for obj_index, name in enumerate(args.object_order, start=1):
        cfg = OBJECTS[name]
        obj = object_poses[name]
        anygrasp_pose = grasp_records[name]["final_grasp_pose"]
        grasp_pos, grasp_quat, approach_axis, mode_note = grasp_pose_for_object(
            args,
            name,
            anygrasp_pose,
            obj,
        )
        pregrasp_distance = float(anygrasp_pose.get("pregrasp_distance_override_m", args.pregrasp_distance))
        approach_lift_distance = float(anygrasp_pose.get("approach_lift_distance_override_m", args.lift_distance))
        pregrasp_pos = grasp_pos - approach_axis * pregrasp_distance
        approach_lift_pos = grasp_pos - approach_axis * approach_lift_distance
        carry_from_grasp = np.array(
            [approach_lift_pos[0], approach_lift_pos[1], max(float(CARRY_Z), float(approach_lift_pos[2]))],
            dtype=np.float64,
        )
        stage_mode = anygrasp_pose.get("pregrasp_stage_mode_override")
        stage_distance = anygrasp_pose.get("pregrasp_stage_distance_override_m")
        if stage_mode == "straight_approach_axis" and stage_distance is not None:
            pregrasp_stage = grasp_pos - approach_axis * float(stage_distance)
        else:
            pregrasp_stage = np.array([pregrasp_pos[0], pregrasp_pos[1], max(float(CARRY_Z), float(pregrasp_pos[2]))])

        place_x = BASKET_CENTER_X + cfg["place_offset_xy"][0]
        place_y = BASKET_CENTER_Y + cfg["place_offset_xy"][1]
        place_pos = np.array([place_x, place_y, PLACE_HEIGHT], dtype=np.float64)
        transport_pos = np.array([place_x, place_y, CARRY_Z], dtype=np.float64)
        prefix = f"{obj_index:02d}_{name}"

        center = np.asarray(obj["center_w"], dtype=np.float64)
        ee_to_object_pos = (center - grasp_pos).astype(float).tolist()
        attach_payload = {
            "action": "attach",
            "object_key": cfg["object_key"],
            "object_name": name,
            "ee_to_object_pos_w": ee_to_object_pos,
            "object_quat_wxyz": obj["quat_wxyz"],
        }
        release_payload = {
            "action": "release",
            "object_key": cfg["object_key"],
            "object_name": name,
            "release_center_w": [place_x, place_y, TABLE_TOP_Z + 0.05],
            "object_quat_wxyz": obj["quat_wxyz"],
        }

        source_objects.append(
            {
                "name": name,
                "object_key": cfg["object_key"],
                "label": cfg["label"],
                "center_w": obj["center_w"],
                "object_quat_wxyz": obj["quat_wxyz"],
                "mask_center_estimate": grasp_records[name]["mask_center_estimate"],
                "sam3": grasp_records[name]["sam3"],
                "anygrasp": {
                    "result_path": str(grasp_records[name]["anygrasp_result_path"]),
                    "final_grasp_pose_path": str(grasp_records[name]["final_grasp_pose_path"]),
                    "status": grasp_records[name]["anygrasp_status"],
                    "score": anygrasp_pose.get("score"),
                    "width": anygrasp_pose.get("width"),
                    "depth": anygrasp_pose.get("depth"),
                },
                "grasp_pose_w": pose(grasp_pos.astype(float).tolist(), grasp_quat),
                "pregrasp_pose_w": pose(pregrasp_pos.astype(float).tolist(), grasp_quat),
                "pregrasp_distance_m": pregrasp_distance,
                "pregrasp_stage_pose_w": pose(pregrasp_stage.astype(float).tolist(), grasp_quat),
                "pregrasp_stage_mode": stage_mode or "vertical_above_pregrasp",
                "pregrasp_stage_distance_m": float(stage_distance) if stage_distance is not None else None,
                "approach_lift_distance_m": approach_lift_distance,
                "place_pose_w": pose(place_pos.astype(float).tolist(), grasp_quat),
                **mode_note,
            }
        )

        waypoints.extend(
            [
                waypoint(f"{prefix}_pregrasp_stage", pregrasp_stage, grasp_quat, GRIPPER_OPEN, args.staging_steps),
                waypoint(f"{prefix}_pregrasp", pregrasp_pos, grasp_quat, GRIPPER_OPEN, args.pregrasp_steps),
                waypoint(f"{prefix}_grasp", grasp_pos, grasp_quat, GRIPPER_OPEN, args.grasp_steps),
                waypoint(
                    f"{prefix}_close",
                    grasp_pos,
                    grasp_quat,
                    GRIPPER_CLOSE,
                    args.close_steps,
                    object_transport=attach_payload,
                    hold_current_pose=True,
                ),
                waypoint(f"{prefix}_approach_lift", approach_lift_pos, grasp_quat, GRIPPER_CLOSE, args.lift_steps),
                waypoint(f"{prefix}_lift", carry_from_grasp, grasp_quat, GRIPPER_CLOSE, args.lift_steps),
                waypoint(f"{prefix}_transport", transport_pos, grasp_quat, GRIPPER_CLOSE, args.transport_steps),
                waypoint(f"{prefix}_place", place_pos, grasp_quat, GRIPPER_CLOSE, args.place_steps),
                waypoint(
                    f"{prefix}_open",
                    place_pos,
                    grasp_quat,
                    GRIPPER_OPEN,
                    args.open_steps,
                    object_transport=release_payload,
                ),
                waypoint(f"{prefix}_lift_retract", transport_pos, grasp_quat, GRIPPER_OPEN, args.retract_steps),
            ]
        )

    waypoints.append(
        waypoint(
            "99_final_retract",
            [RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z],
            DEFAULT_QUAT_WXYZ,
            GRIPPER_OPEN,
            args.retract_steps,
        )
    )

    return {
        "schema": REQUEST_SCHEMA,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": "ATEC-TaskE-Piper",
        "frame": "world",
        "units": {"position": "meter", "quaternion": "wxyz"},
        "source": {
            "type": "full_task_anygrasp_video",
            "input_dir": str(args.input.resolve()),
            "output_dir": str(output_dir),
            "object_pose_source": args.object_center_source,
            "grasp_pose_source": "video_cam_sam3_anygrasp",
            "object_order": list(args.object_order),
            "objects": source_objects,
            "note": (
                "This is an experimental full-task request. It uses external video_cam RGB-D "
                "for segmentation and AnyGrasp, then executes through the existing Task E runner."
            ),
        },
        "backend": {
            "preferred": args.preferred_backend,
            "fallback": "isaaclab_cartesian_controller",
        },
        "controller": {
            "actuator_mode": args.actuator_mode,
            "object_transport_mode": args.object_transport_mode,
            "grasp_quat_source": args.grasp_mode,
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
        "waypoints": waypoints,
    }


def object_poses_for_request(
    args: argparse.Namespace,
    mask_centers: dict[str, dict],
) -> dict[str, dict]:
    poses = deterministic_object_poses(args.seed)
    if args.object_center_source == "deterministic":
        return poses
    for name, estimate in mask_centers.items():
        xy = estimate["center_world_median"][:2]
        poses[name]["center_w"] = [float(xy[0]), float(xy[1]), float(TABLE_TOP_Z + 0.05)]
        poses[name]["source"] = "video_mask_depth_median_xy"
    return poses


def main() -> None:
    args = parse_args()
    args.input = args.input.expanduser().resolve()
    output_dir = prepare_output_dir(args.output)
    rgb, depth, camera = require_input(args.input)

    mask_centers: dict[str, dict] = {}
    grasp_records: dict[str, dict] = {}
    for name in args.object_order:
        mask_path, detection = run_sam3_for_object(args, rgb, output_dir, name)
        mask_center = estimate_mask_center(depth, mask_path, camera, args.max_depth)
        pose_path, anygrasp_result = run_anygrasp_for_object(
            args,
            rgb,
            depth,
            camera,
            mask_path,
            output_dir,
            name,
        )
        anygrasp_payload = anygrasp_result.get("anygrasp", {})
        if anygrasp_payload.get("status") != "ok":
            raise RuntimeError(f"AnyGrasp failed for {name}: {anygrasp_payload}")
        final_pose = load_json(pose_path)
        mask_centers[name] = mask_center
        grasp_records[name] = {
            "sam3": {
                "prompt": detection.get("prompt"),
                "mask_count": detection.get("mask_count"),
                "best_index": detection.get("best_index"),
                "areas_px": detection.get("areas_px"),
                "scores": detection.get("scores"),
                "mask_path": str(mask_path),
            },
            "mask_center_estimate": mask_center,
            "anygrasp_result_path": output_dir / "anygrasp" / name / "anygrasp_result.json",
            "final_grasp_pose_path": pose_path,
            "anygrasp_status": anygrasp_payload.get("status"),
            "final_grasp_pose": final_pose,
        }
        print(
            "[INFO] "
            f"{name}: mask_points={mask_center['point_count']} "
            f"anygrasp_score={final_pose.get('score')} "
            f"grasp={np.round(final_pose['translation'], 4).tolist()}",
            flush=True,
        )

    object_poses = object_poses_for_request(args, mask_centers)
    request = build_request(args, output_dir, object_poses, grasp_records)
    request_path = output_dir / "motion_request.json"
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    summary_path = output_dir / "pipeline_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "input_dir": str(args.input),
                "request": str(request_path),
                "object_order": list(args.object_order),
                "grasp_mode": args.grasp_mode,
                "object_transport_mode": args.object_transport_mode,
                "objects": request["source"]["objects"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[INFO] Saved full AnyGrasp request: {request_path}")
    print(f"[INFO] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
