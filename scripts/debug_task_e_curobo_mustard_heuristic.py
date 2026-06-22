"""Mustard-only heuristic grasp test using cuRobo IK and submission-style actions."""

from __future__ import annotations

import argparse
import json
import sys
import time
import copy
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher
from PIL import Image


ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINTS = ["joint7", "joint8"]
ACTION_SCALE = 0.5
GRIPPER_OPEN = np.array([0.035, -0.035], dtype=np.float32)
GRIPPER_CLOSE = np.array([-0.015, 0.015], dtype=np.float32)
PIPER_FINGER_LENGTH_M = 0.0765
PIPER_FINGER_WIDTH_M = 0.0265
PIPER_FINGER_DEPTH_M = 0.056
PIPER_PALM_APPROACH_THICKNESS_M = 0.035
PIPER_MAX_JAW_WIDTH_M = 0.075

TABLE_CENTER_X = 1.00
TABLE_CENTER_Y = 0.00
TABLE_CENTER_Z = 0.00
TABLE_SCALE = 0.01
TABLE_DIMS_AT_0P008 = (0.6468062441005529, 0.9084968693231588, 0.6613141183247961)
TABLE_DIMS = tuple(dim * (TABLE_SCALE / 0.008) for dim in TABLE_DIMS_AT_0P008)
TABLE_TOP_Z = TABLE_CENTER_Z + TABLE_DIMS[2]
BASKET_CENTER_X = TABLE_CENTER_X + 0.08
BASKET_CENTER_Y = TABLE_CENTER_Y - 0.30

VIDEO_INTRINSIC = np.array(
    [[732.999267578125, 0.0, 320.0], [0.0, 732.999267578125, 240.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
VIDEO_POS_W = np.array([-0.20000000298023224, 0.0, 1.6266427040100098], dtype=np.float32)
VIDEO_QUAT_W_ROS = np.array(
    [-0.3335084915161133, 0.6235159635543823, -0.6235158443450928, 0.3335084915161133],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="ATEC-TaskE-Piper")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--submission-dir", default="submissions/task_e_act_baseline_root_submission")
    parser.add_argument("--output", default="outputs/task_e_curobo_mustard_heuristic/latest")
    parser.add_argument("--num-seeds", type=int, default=96)
    parser.add_argument("--grasp-z-offset", type=float, default=0.09)
    parser.add_argument("--grasp-height-mode", choices=("offset", "tip_inside"), default="tip_inside")
    parser.add_argument("--tool-tip-offset", type=float, default=0.0765)
    parser.add_argument("--tip-insert-depth", type=float, default=0.035)
    parser.add_argument("--palm-clearance", type=float, default=0.015)
    parser.add_argument("--pregrasp-height", type=float, default=0.04)
    parser.add_argument("--lift-height", type=float, default=0.22)
    parser.add_argument("--lift-mode", choices=("fixed", "relaxed"), default="relaxed")
    parser.add_argument("--lift-orientation-mode", choices=("fixed", "tilt_search"), default="fixed")
    parser.add_argument("--lift-min-height", type=float, default=0.09)
    parser.add_argument("--lift-xy-radius", type=float, default=0.06)
    parser.add_argument("--lift-tilt-degrees", type=float, nargs="*", default=[15.0, 30.0])
    parser.add_argument("--orientation-mode", choices=("yaw_search", "legacy"), default="yaw_search")
    parser.add_argument("--yaw-candidates", type=int, default=24)
    parser.add_argument("--position-search", action="store_true")
    parser.add_argument("--search-x-radius", type=float, default=0.045)
    parser.add_argument("--search-y-radius", type=float, default=0.025)
    parser.add_argument("--search-grid", type=int, default=5)
    parser.add_argument("--waypoint-max-step-rad", type=float, default=0.035)
    parser.add_argument("--waypoint-hold-steps", type=int, default=10)
    parser.add_argument("--final-hold-steps", type=int, default=60)
    parser.add_argument("--scene-collision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--basket-obstacle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grasp-source", choices=("heuristic", "anygrasp", "graspgen"), default="heuristic")
    parser.add_argument(
        "--candidate-base-dir",
        default=(
            "outputs/task_e_ideal_ee_camera_debug/20260614_same_env_lookat_real_ee/"
            "grasp_candidates_mixed_ee_banana_all_others/mustard_bottle"
        ),
    )
    parser.add_argument("--candidate-top-k", type=int, default=10)
    parser.add_argument("--candidate-pregrasp-distance", type=float, default=0.12)
    parser.add_argument("--candidate-stage-extra-distance", type=float, default=0.06)
    parser.add_argument("--candidate-lift-distance", type=float, default=0.18)
    parser.add_argument("--candidate-gripper-base-offset", type=float, default=0.09)
    parser.add_argument("--candidate-offset-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidate-offset-min", type=float, default=0.0)
    parser.add_argument("--candidate-offset-max", type=float, default=0.12)
    parser.add_argument("--candidate-offset-step", type=float, default=0.01)
    parser.add_argument(
        "--candidate-gripper-base-offset-mode",
        choices=("approach_axis", "finger_centerline", "yellow_line", "towards_object_center"),
        default="towards_object_center",
    )
    parser.add_argument("--candidate-align-to-current-cloud", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidate-approach-collision-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidate-approach-collision-samples", type=int, default=10)
    parser.add_argument("--candidate-approach-collision-final-fraction", type=float, default=0.85)
    parser.add_argument("--candidate-approach-collision-min-points", type=int, default=5)
    parser.add_argument("--candidate-approach-collision-clearance", type=float, default=0.002)
    parser.add_argument("--candidate-final-solid-max-points", type=int, default=0)
    parser.add_argument("--candidate-final-palm-max-points", type=int, default=8)
    parser.add_argument("--candidate-final-region-min-points", type=int, default=10)
    parser.add_argument("--candidate-max-jaw-width", type=float, default=PIPER_MAX_JAW_WIDTH_M)
    parser.add_argument("--candidate-ik-position-tol", type=float, default=0.025)
    parser.add_argument("--candidate-ik-rotation-tol", type=float, default=0.30)
    parser.add_argument("--candidate-save-collision-samples", action="store_true")
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


def to_rgb_array(value) -> np.ndarray:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        if array.size and np.nanmax(array) <= 1.0:
            array *= 255.0
        array = np.clip(np.nan_to_num(array), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def to_depth_array(value) -> np.ndarray:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3:
        if array.shape[0] == 1:
            array = array[0]
        elif array.shape[-1] == 1:
            array = array[..., 0]
    return array.astype(np.float32)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat /= max(np.linalg.norm(quat), 1e-12)
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.asarray([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)


def quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return np.asarray([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


def quat_wxyz_from_matrix(matrix: np.ndarray, rotation_cls) -> np.ndarray:
    return quat_xyzw_to_wxyz(rotation_cls.from_matrix(np.asarray(matrix, dtype=np.float64)).as_quat()).astype(np.float32)


def normalize_vector(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        return np.asarray(fallback, dtype=np.float64)
    return np.asarray(value, dtype=np.float64) / norm


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def camera_points_to_world(points_cam: np.ndarray, camera_payload: dict) -> np.ndarray:
    camera = camera_payload.get("camera", camera_payload)
    rot_wc = quat_wxyz_to_matrix(np.asarray(camera["quat_w_ros"], dtype=np.float64))
    pos_w = np.asarray(camera["pos_w"], dtype=np.float64)
    return np.asarray(points_cam, dtype=np.float64) @ rot_wc.T + pos_w


def perceive_mustard(obs: dict, output_dir: Path) -> dict:
    rgb = to_rgb_array(obs["image"]["video_rgb"])
    depth = to_depth_array(obs["image"]["video_depth"])
    r, g, b = rgb[..., 0].astype(np.int16), rgb[..., 1].astype(np.int16), rgb[..., 2].astype(np.int16)
    yellow = (r > 135) & (g > 105) & (b < 120) & (r > b + 45) & (g > b + 35)
    valid = yellow & np.isfinite(depth) & (depth > 0.05) & (depth < 2.2)

    ys, xs = np.nonzero(valid)
    if xs.size < 20:
        raise RuntimeError(f"Mustard mask too small: {xs.size} valid pixels")
    z = depth[ys, xs]
    points_cam = np.stack(
        [
            (xs.astype(np.float32) - VIDEO_INTRINSIC[0, 2]) * z / VIDEO_INTRINSIC[0, 0],
            (ys.astype(np.float32) - VIDEO_INTRINSIC[1, 2]) * z / VIDEO_INTRINSIC[1, 1],
            z,
        ],
        axis=-1,
    )
    rot_wc = quat_wxyz_to_matrix(VIDEO_QUAT_W_ROS)
    points_w = points_cam @ rot_wc.T + VIDEO_POS_W
    workspace = (
        (points_w[:, 0] >= 0.84)
        & (points_w[:, 0] <= 1.16)
        & (points_w[:, 1] >= 0.10)
        & (points_w[:, 1] <= 0.23)
        & (points_w[:, 2] >= TABLE_TOP_Z - 0.02)
        & (points_w[:, 2] <= TABLE_TOP_Z + 0.35)
    )
    points_w = points_w[workspace]
    if len(points_w) < 20:
        raise RuntimeError(f"Mustard workspace point cloud too small: {len(points_w)} points")

    q_low, q_high = np.quantile(points_w, [0.01, 0.99], axis=0)
    trimmed = points_w[np.all((points_w >= q_low) & (points_w <= q_high), axis=1)]
    if len(trimmed) < 20:
        trimmed = points_w
    bbox_min = trimmed.min(axis=0)
    bbox_max = trimmed.max(axis=0)
    center = (bbox_min + bbox_max) * 0.5
    center[2] = max(float(center[2]), TABLE_TOP_Z + 0.01)

    Image.fromarray(rgb).save(output_dir / "video_rgb.png")
    Image.fromarray(valid.astype(np.uint8) * 255).save(output_dir / "mustard_mask.png")
    np.save(output_dir / "mustard_points_world.npy", points_w.astype(np.float32))
    return {
        "mask_pixels": int(xs.size),
        "workspace_points": int(len(points_w)),
        "bbox_min_w": bbox_min.astype(float).tolist(),
        "bbox_max_w": bbox_max.astype(float).tolist(),
        "center_w": center.astype(float).tolist(),
    }


def pose_in_robot_root(robot, ee_idx: int, rotation_cls) -> tuple[np.ndarray, np.ndarray]:
    root_pose = robot.data.root_pose_w[0].detach().cpu().numpy()
    ee_pose = robot.data.body_pose_w[0, ee_idx].detach().cpu().numpy()
    root_pos = root_pose[:3].astype(np.float64)
    root_rot = rotation_cls.from_quat(quat_wxyz_to_xyzw(root_pose[3:7]))
    ee_pos_w = ee_pose[:3].astype(np.float64)
    ee_rot_w = rotation_cls.from_quat(quat_wxyz_to_xyzw(ee_pose[3:7]))
    return (
        root_rot.inv().apply(ee_pos_w - root_pos).astype(np.float32),
        quat_xyzw_to_wxyz((root_rot.inv() * ee_rot_w).as_quat()).astype(np.float32),
    )


def world_pose_to_root(robot, pos_w: np.ndarray, quat_wxyz: np.ndarray, rotation_cls) -> tuple[np.ndarray, np.ndarray]:
    root_pose = robot.data.root_pose_w[0].detach().cpu().numpy()
    root_pos = root_pose[:3].astype(np.float64)
    root_rot = rotation_cls.from_quat(quat_wxyz_to_xyzw(root_pose[3:7]))
    target_rot_w = rotation_cls.from_quat(quat_wxyz_to_xyzw(quat_wxyz))
    return (
        root_rot.inv().apply(np.asarray(pos_w, dtype=np.float64) - root_pos).astype(np.float32),
        quat_xyzw_to_wxyz((root_rot.inv() * target_rot_w).as_quat()).astype(np.float32),
    )


def quat_abs_dot(q0: np.ndarray, q1: np.ndarray) -> float:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 /= max(np.linalg.norm(q0), 1e-12)
    q1 /= max(np.linalg.norm(q1), 1e-12)
    return float(abs(np.dot(q0, q1)))


def topdown_quat_wxyz_from_yaw(yaw_rad: float, rotation_cls) -> np.ndarray:
    long_axis = np.asarray([np.cos(yaw_rad), np.sin(yaw_rad), 0.0], dtype=np.float64)
    grip_z = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    jaw_dir = np.cross(long_axis, grip_z)
    jaw_dir /= max(np.linalg.norm(jaw_dir), 1e-12)
    align_dir = np.cross(jaw_dir, grip_z)
    align_dir /= max(np.linalg.norm(align_dir), 1e-12)
    matrix = np.stack([align_dir, jaw_dir, grip_z], axis=1)
    return quat_xyzw_to_wxyz(rotation_cls.from_matrix(matrix).as_quat()).astype(np.float32)


def local_tilt_quat_candidates(quat_wxyz: np.ndarray, tilt_degrees: list[float], rotation_cls) -> list[dict]:
    base_rot = rotation_cls.from_quat(quat_wxyz_to_xyzw(quat_wxyz))
    candidates = [
        {
            "label": "fixed",
            "tilt_axis": "none",
            "tilt_degrees": 0.0,
            "tilt_l1_degrees": 0.0,
            "quat_wxyz": np.asarray(quat_wxyz, dtype=np.float32),
        }
    ]
    for degrees in tilt_degrees:
        for axis in ("x", "y"):
            for sign in (-1.0, 1.0):
                angle = float(sign * degrees)
                quat = quat_xyzw_to_wxyz((base_rot * rotation_cls.from_euler(axis, angle, degrees=True)).as_quat())
                candidates.append(
                    {
                        "label": f"local_{axis}_{angle:g}deg",
                        "tilt_axis": axis,
                        "tilt_degrees": angle,
                        "tilt_l1_degrees": abs(angle),
                        "quat_wxyz": quat.astype(np.float32),
                    }
                )
    return candidates


def gripper_tip_center_w(base_pos_w: np.ndarray, quat_wxyz: np.ndarray, tip_offset: float) -> np.ndarray:
    rot_w_tool = quat_wxyz_to_matrix(quat_wxyz)
    return np.asarray(base_pos_w, dtype=np.float32) + rot_w_tool[:, 2].astype(np.float32) * float(tip_offset)


def gripper_base_from_tip_w(tip_pos_w: np.ndarray, quat_wxyz: np.ndarray, tip_offset: float) -> np.ndarray:
    rot_w_tool = quat_wxyz_to_matrix(quat_wxyz)
    return np.asarray(tip_pos_w, dtype=np.float32) - rot_w_tool[:, 2].astype(np.float32) * float(tip_offset)


def load_ranked_generator_candidates(candidate_base_dir: Path, generator: str, top_k: int) -> tuple[list[dict], dict]:
    generator_dir = candidate_base_dir / generator
    top_grasps_path = generator_dir / "top_grasps.json"
    if not top_grasps_path.exists():
        raise FileNotFoundError(top_grasps_path)

    raw_candidates = json.loads(top_grasps_path.read_text(encoding="utf-8"))
    converted: list[dict] = []
    for idx, raw in enumerate(raw_candidates, start=1):
        world = raw.get("pose_world")
        if world is None and raw.get("translation_world") is not None:
            world = {
                "translation": raw.get("translation_world"),
                "rotation_matrix": raw.get("rotation_matrix_world"),
            }
        if world is None and raw.get("translation") is not None:
            world = {
                "translation": raw.get("translation"),
                "rotation_matrix": raw.get("rotation_matrix"),
            }
        if not world or world.get("translation") is None or world.get("rotation_matrix") is None:
            continue
        converted.append(
            {
                "rank": int(raw.get("rank", raw.get("source_rank", idx))),
                "source_rank": int(raw.get("source_rank", raw.get("rank", idx))),
                "score": float(raw.get("score", 0.0)),
                "width": float(raw.get("width", 0.065)),
                "depth": float(raw.get("depth", 0.0)),
                "translation": [float(v) for v in world["translation"]],
                "rotation_matrix": [[float(v) for v in row] for row in world["rotation_matrix"]],
                "generator": generator,
                "collision_free": bool(raw.get("collision_free", True)),
                "raw_candidate": raw,
            }
        )

    converted.sort(key=lambda item: (-float(item["score"]), int(item["source_rank"])))
    if top_k > 0:
        converted = converted[: int(top_k)]

    meta: dict = {
        "generator": generator,
        "generator_dir": str(generator_dir),
        "top_grasps_path": str(top_grasps_path),
        "loaded_count": len(raw_candidates),
        "used_count": len(converted),
    }
    result_path = generator_dir / "anygrasp_result.json"
    cloud_path = generator_dir / "masked_cloud.npy"
    if result_path.exists() and cloud_path.exists():
        result = load_json(result_path)
        camera_path = Path(result["camera_json"]).expanduser()
        if not camera_path.is_absolute():
            camera_path = (generator_dir / camera_path).resolve()
        points_w = camera_points_to_world(np.load(cloud_path), load_json(camera_path))
        q_low, q_high = np.quantile(points_w, [0.01, 0.99], axis=0)
        trimmed = points_w[np.all((points_w >= q_low) & (points_w <= q_high), axis=1)]
        if len(trimmed) < 20:
            trimmed = points_w
        bbox_min = trimmed.min(axis=0)
        bbox_max = trimmed.max(axis=0)
        meta.update(
            {
                "saved_cloud_point_count": int(points_w.shape[0]),
                "saved_cloud_center_w": ((bbox_min + bbox_max) * 0.5).astype(float).tolist(),
                "saved_cloud_bbox_min_w": bbox_min.astype(float).tolist(),
                "saved_cloud_bbox_max_w": bbox_max.astype(float).tolist(),
                "saved_cloud_camera_json": str(camera_path),
            }
        )
    return converted, meta


def align_candidates_to_current_cloud(
    candidates: list[dict],
    saved_center_w: list[float] | None,
    current_center_w: np.ndarray,
) -> tuple[list[dict], dict]:
    if saved_center_w is None:
        return candidates, {"enabled": False, "reason": "no_saved_cloud_center"}
    saved_center = np.asarray(saved_center_w, dtype=np.float64)
    current_center = np.asarray(current_center_w, dtype=np.float64)
    shift = current_center - saved_center
    aligned = []
    for candidate in candidates:
        item = copy.deepcopy(candidate)
        item["translation"] = (np.asarray(item["translation"], dtype=np.float64) + shift).astype(float).tolist()
        item["alignment_shift_w"] = shift.astype(float).tolist()
        aligned.append(item)
    return aligned, {
        "enabled": True,
        "saved_center_w": saved_center.astype(float).tolist(),
        "current_center_w": current_center.astype(float).tolist(),
        "shift_w": shift.astype(float).tolist(),
    }


def piper_pose_from_generator_candidate(
    candidate: dict,
    object_center_w: np.ndarray,
    *,
    offset_distance: float,
    offset_mode: str,
    rotation_cls,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    translation = np.asarray(candidate["translation"], dtype=np.float64)
    rotation = np.asarray(candidate["rotation_matrix"], dtype=np.float64)
    piper_rotation = np.stack([-rotation[:, 2], rotation[:, 1], rotation[:, 0]], axis=1)
    approach_axis = normalize_vector(piper_rotation[:, 2], np.array([0.0, 0.0, -1.0], dtype=np.float64))

    if offset_mode == "towards_object_center":
        offset_axis = normalize_vector(np.asarray(object_center_w, dtype=np.float64) - translation, -approach_axis)
    elif offset_mode in {"finger_centerline", "yellow_line"}:
        offset_axis = approach_axis
    else:
        offset_axis = -approach_axis

    gripper_base_pos = translation + offset_axis * float(offset_distance)
    quat_wxyz = quat_wxyz_from_matrix(piper_rotation, rotation_cls)
    note = {
        "raw_generator_translation_w": translation.astype(float).tolist(),
        "raw_generator_rotation_matrix_w": rotation.astype(float).tolist(),
        "piper_rotation_matrix_w": piper_rotation.astype(float).tolist(),
        "approach_axis_w": approach_axis.astype(float).tolist(),
        "offset_mode": offset_mode,
        "offset_distance_m": float(offset_distance),
        "offset_axis_w": offset_axis.astype(float).tolist(),
        "note": "Generator GraspNet-style +X approach is mapped onto Piper gripper_base +Z.",
    }
    return gripper_base_pos.astype(np.float32), quat_wxyz, approach_axis.astype(np.float32), piper_rotation.astype(np.float32), note


def piper_gripper_target_collision_stats(
    object_points_w: np.ndarray,
    piper_rotation: np.ndarray,
    gripper_pos_w: np.ndarray,
    jaw_width: float,
    clearance: float,
) -> dict:
    side_axis = normalize_vector(np.asarray(piper_rotation, dtype=np.float64)[:, 0], np.array([1.0, 0.0, 0.0]))
    jaw_axis = normalize_vector(np.asarray(piper_rotation, dtype=np.float64)[:, 1], np.array([0.0, 1.0, 0.0]))
    approach_axis = normalize_vector(np.asarray(piper_rotation, dtype=np.float64)[:, 2], np.array([0.0, 0.0, -1.0]))
    jaw_width = float(np.clip(jaw_width, 0.030, 0.095))
    clearance = max(0.0, float(clearance))
    gripper_pos = np.asarray(gripper_pos_w, dtype=np.float64)
    object_points = np.asarray(object_points_w, dtype=np.float64)

    finger_root = gripper_pos - approach_axis * PIPER_FINGER_LENGTH_M
    inside_solid = np.zeros(object_points.shape[0], dtype=bool)
    components: dict[str, int] = {}
    for side_sign, label in [(-1.0, "left_finger"), (1.0, "right_finger")]:
        center_offset = jaw_axis * side_sign * (jaw_width * 0.5 + PIPER_FINGER_WIDTH_M * 0.5)
        root_center = finger_root + center_offset
        rel = object_points - root_center[None, :]
        along = rel @ approach_axis
        jaw = rel @ jaw_axis
        side = rel @ side_axis
        inside = (
            (along >= -clearance)
            & (along <= PIPER_FINGER_LENGTH_M + clearance)
            & (np.abs(jaw) <= PIPER_FINGER_WIDTH_M * 0.5 + clearance)
            & (np.abs(side) <= PIPER_FINGER_DEPTH_M * 0.5 + clearance)
        )
        components[label] = int(np.count_nonzero(inside))
        inside_solid |= inside

    palm_center = finger_root - approach_axis * (PIPER_PALM_APPROACH_THICKNESS_M * 0.5)
    rel = object_points - palm_center[None, :]
    palm_along = rel @ approach_axis
    palm_jaw = rel @ jaw_axis
    palm_side = rel @ side_axis
    palm_inside = (
        (np.abs(palm_along) <= PIPER_PALM_APPROACH_THICKNESS_M * 0.5 + clearance)
        & (np.abs(palm_jaw) <= (jaw_width * 0.5 + PIPER_FINGER_WIDTH_M) + clearance)
        & (np.abs(palm_side) <= PIPER_FINGER_DEPTH_M * 0.5 + clearance)
    )
    components["palm_base"] = int(np.count_nonzero(palm_inside))
    inside_solid |= palm_inside

    rel = object_points - finger_root[None, :]
    region_along = rel @ approach_axis
    region_jaw = rel @ jaw_axis
    region_side = rel @ side_axis
    closing_region = (
        (region_along >= -clearance)
        & (region_along <= PIPER_FINGER_LENGTH_M + clearance)
        & (np.abs(region_jaw) <= jaw_width * 0.5 + clearance)
        & (np.abs(region_side) <= PIPER_FINGER_DEPTH_M * 0.5 + clearance)
    )

    return {
        "solid_collision_point_count": int(np.count_nonzero(inside_solid)),
        "component_collision_points": components,
        "closing_region_point_count": int(np.count_nonzero(closing_region)),
        "gripper_pos_w": gripper_pos.astype(float).tolist(),
        "jaw_width_m": jaw_width,
    }


def approach_collision_check_for_candidate(
    object_points_w: np.ndarray,
    piper_rotation: np.ndarray,
    grasp_pos_w: np.ndarray,
    approach_axis_w: np.ndarray,
    jaw_width: float,
    args: argparse.Namespace,
) -> tuple[bool, dict]:
    if not args.candidate_approach_collision_filter:
        return True, {"enabled": False, "safe": True}

    pregrasp_distance = float(args.candidate_pregrasp_distance)
    stage_distance = pregrasp_distance + float(args.candidate_stage_extra_distance)
    pregrasp_pos = np.asarray(grasp_pos_w, dtype=np.float64) - np.asarray(approach_axis_w, dtype=np.float64) * pregrasp_distance
    stage_pos = np.asarray(grasp_pos_w, dtype=np.float64) - np.asarray(approach_axis_w, dtype=np.float64) * stage_distance
    samples = max(2, int(args.candidate_approach_collision_samples))
    final_fraction = float(np.clip(args.candidate_approach_collision_final_fraction, 0.0, 1.0))
    min_points = max(1, int(args.candidate_approach_collision_min_points))
    clearance = float(args.candidate_approach_collision_clearance)

    tested: list[dict] = []
    first_illegal = None

    def add_segment(label: str, start: np.ndarray, end: np.ndarray, max_fraction: float) -> None:
        nonlocal first_illegal
        fractions = [float(v) for v in np.linspace(0.0, max_fraction, samples)]
        for fraction in fractions:
            pos = start * (1.0 - fraction) + end * fraction
            stats = piper_gripper_target_collision_stats(object_points_w, piper_rotation, pos, jaw_width, clearance)
            entry = {
                "segment": label,
                "fraction": float(fraction),
                "solid_collision_point_count": int(stats["solid_collision_point_count"]),
                "component_collision_points": stats["component_collision_points"],
                "closing_region_point_count": int(stats["closing_region_point_count"]),
                "gripper_pos_w": stats["gripper_pos_w"],
                "illegal": bool(stats["solid_collision_point_count"] >= min_points),
            }
            tested.append(entry)
            if first_illegal is None and entry["illegal"]:
                first_illegal = entry

    add_segment("stage_to_pregrasp", stage_pos, pregrasp_pos, 1.0)
    add_segment("pregrasp_to_grasp_early", pregrasp_pos, np.asarray(grasp_pos_w, dtype=np.float64), final_fraction)

    final_stats = piper_gripper_target_collision_stats(
        object_points_w,
        piper_rotation,
        grasp_pos_w,
        jaw_width,
        clearance,
    )
    final_solid_ok = int(final_stats["solid_collision_point_count"]) <= int(args.candidate_final_solid_max_points)
    final_palm_ok = int(final_stats["component_collision_points"]["palm_base"]) <= int(args.candidate_final_palm_max_points)
    final_region_ok = int(final_stats["closing_region_point_count"]) >= int(args.candidate_final_region_min_points)
    safe = first_illegal is None and final_solid_ok and final_palm_ok and final_region_ok
    return safe, {
        "enabled": True,
        "safe": bool(safe),
        "first_illegal": first_illegal,
        "min_points": int(min_points),
        "clearance_m": float(clearance),
        "pregrasp_distance_m": float(pregrasp_distance),
        "stage_distance_m": float(stage_distance),
        "checked_until_final_fraction": float(final_fraction),
        "final_stats": final_stats,
        "final_solid_ok": bool(final_solid_ok),
        "final_solid_max_points": int(args.candidate_final_solid_max_points),
        "final_palm_ok": bool(final_palm_ok),
        "final_region_ok": bool(final_region_ok),
        "tested_sample_count": len(tested),
        "tested_samples": tested if bool(args.candidate_save_collision_samples) else [],
        "note": (
            "The final open-gripper pose must not contain target points inside the physical "
            "finger or palm solids. Target points are only required inside the empty closing "
            "gap between fingers; contact should occur during gripper closing, not during "
            "trajectory generation."
        ),
    }


def candidate_offset_values(args: argparse.Namespace) -> list[float]:
    preferred = float(args.candidate_gripper_base_offset)
    if not args.candidate_offset_search:
        return [preferred]

    min_offset = float(args.candidate_offset_min)
    max_offset = max(min_offset, float(args.candidate_offset_max))
    step = max(1e-5, float(args.candidate_offset_step))
    values = [float(v) for v in np.arange(min_offset, max_offset + step * 0.5, step)]
    values.append(preferred)
    values = sorted({round(float(np.clip(v, min_offset, max_offset)), 6) for v in values})
    values.sort(key=lambda v: (abs(v - preferred), v))
    return values


def select_curobo_generator_candidate(
    planner,
    robot,
    arm_ids,
    object_points_w: np.ndarray,
    perception: dict,
    args: argparse.Namespace,
    rotation_cls,
) -> tuple[dict | None, list[dict], dict]:
    candidate_base_dir = Path(args.candidate_base_dir)
    candidates, meta = load_ranked_generator_candidates(candidate_base_dir, args.grasp_source, args.candidate_top_k)
    center_w = np.asarray(perception["center_w"], dtype=np.float32)
    if args.candidate_align_to_current_cloud:
        candidates, alignment = align_candidates_to_current_cloud(
            candidates,
            meta.get("saved_cloud_center_w"),
            center_w,
        )
    else:
        alignment = {"enabled": False, "reason": "disabled"}

    current_q_start = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().astype(np.float32)
    evaluations: list[dict] = []
    selected: dict | None = None
    for candidate in candidates:
        grasp_pos_w, grasp_quat_w, approach_axis_w, piper_rotation_w, conversion = piper_pose_from_generator_candidate(
            candidate,
            center_w,
            offset_distance=float(args.candidate_gripper_base_offset),
            offset_mode=args.candidate_gripper_base_offset_mode,
            rotation_cls=rotation_cls,
        )
        jaw_width = float(candidate.get("width", 0.065))
        max_jaw_width = float(args.candidate_max_jaw_width)
        width_ok = jaw_width <= max_jaw_width + 1e-6
        physical_jaw_width = min(jaw_width, max_jaw_width)
        if not width_ok:
            evaluations.append(
                {
                    "rank": int(candidate["source_rank"]),
                    "score": float(candidate["score"]),
                    "generator": args.grasp_source,
                    "collision_free_from_generator": bool(candidate.get("collision_free", True)),
                    "selected": False,
                    "ok": False,
                    "approach_safe": False,
                    "ik_ok": False,
                    "max_ik_position_error_m": None,
                    "max_ik_rotation_error_rad": None,
                    "physical_width_check": {
                        "ok": False,
                        "candidate_width_m": float(jaw_width),
                        "max_piper_open_width_m": float(max_jaw_width),
                        "reason": "candidate_width_exceeds_physical_piper_opening",
                    },
                    "candidate": {
                        "translation": candidate["translation"],
                        "rotation_matrix": candidate["rotation_matrix"],
                        "width": float(candidate.get("width", 0.065)),
                        "depth": float(candidate.get("depth", 0.0)),
                    },
                    "conversion": conversion,
                    "approach_collision_check": {
                        "enabled": False,
                        "safe": False,
                        "skipped": True,
                        "skip_reason": "candidate_width_exceeds_physical_piper_opening",
                    },
                    "stages": [],
                }
            )
            continue
        approach_safe, approach_report = approach_collision_check_for_candidate(
            object_points_w,
            piper_rotation_w,
            grasp_pos_w,
            approach_axis_w,
            physical_jaw_width,
            args,
        )

        pregrasp_distance = float(args.candidate_pregrasp_distance)
        stage_distance = pregrasp_distance + float(args.candidate_stage_extra_distance)
        stage_pos_w = grasp_pos_w - approach_axis_w * stage_distance
        pregrasp_pos_w = grasp_pos_w - approach_axis_w * pregrasp_distance
        lift_pos_w = grasp_pos_w - approach_axis_w * float(args.candidate_lift_distance)
        stages_w = [
            ("pregrasp_stage_open", stage_pos_w, grasp_quat_w),
            ("pregrasp_open", pregrasp_pos_w, grasp_quat_w),
            ("grasp_open", grasp_pos_w, grasp_quat_w),
            ("lift_closed", lift_pos_w, grasp_quat_w),
        ]

        q = current_q_start.copy()
        stage_results: list[dict] = []
        max_pos_error = 0.0
        max_rot_error = 0.0
        all_success = True
        for label, pos_w, quat_w in stages_w:
            pos_root, quat_root = world_pose_to_root(robot, pos_w, quat_w, rotation_cls)
            ik = planner.solve_ik(q, pos_root, quat_root)
            stage_results.append(
                {
                    "label": label,
                    "ik_success": bool(ik.success),
                    "ik_position_error_m": float(ik.position_error_m),
                    "ik_rotation_error_rad": float(ik.rotation_error_rad),
                    "target_pos_w": np.asarray(pos_w, dtype=float).tolist(),
                    "target_pos_root": pos_root.astype(float).tolist(),
                    "target_quat_wxyz": np.asarray(quat_w, dtype=float).tolist(),
                    "target_quat_root_wxyz": quat_root.astype(float).tolist(),
                    "joint_position": ik.joint_position.astype(float).tolist(),
                }
            )
            max_pos_error = max(max_pos_error, float(ik.position_error_m))
            max_rot_error = max(max_rot_error, float(ik.rotation_error_rad))
            all_success = all_success and bool(ik.success)
            q = ik.joint_position

        ik_ok = (
            all_success
            and max_pos_error <= float(args.candidate_ik_position_tol)
            and max_rot_error <= float(args.candidate_ik_rotation_tol)
        )
        evaluation = {
            "rank": int(candidate["source_rank"]),
            "score": float(candidate["score"]),
            "generator": args.grasp_source,
            "collision_free_from_generator": bool(candidate.get("collision_free", True)),
            "selected": False,
            "ok": bool(approach_safe and ik_ok),
            "approach_safe": bool(approach_safe),
            "ik_ok": bool(ik_ok),
            "max_ik_position_error_m": float(max_pos_error),
            "max_ik_rotation_error_rad": float(max_rot_error),
            "physical_width_check": {
                "ok": True,
                "candidate_width_m": float(jaw_width),
                "checked_jaw_width_m": float(physical_jaw_width),
                "max_piper_open_width_m": float(max_jaw_width),
            },
            "candidate": {
                "translation": candidate["translation"],
                "rotation_matrix": candidate["rotation_matrix"],
                "width": float(candidate.get("width", 0.065)),
                "depth": float(candidate.get("depth", 0.0)),
            },
            "conversion": conversion,
            "approach_collision_check": approach_report,
            "stages": stage_results,
        }
        evaluations.append(evaluation)
        if evaluation["ok"] and selected is None:
            selected = copy.deepcopy(evaluation)
            selected["selected"] = True
            evaluations[-1]["selected"] = True
            break

    meta["alignment"] = alignment
    meta["selection_backend"] = "curobo_ik_scene_collision_plus_target_point_cloud_filter"
    return selected, evaluations, meta


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
    from task_e_full_baseline_request import OBJECTS, task_e_grasp_quat  # noqa: WPS433

    repo_root = Path(__file__).resolve().parents[1]
    submission_dir = (repo_root / args.submission_dir).resolve()
    sys.path.insert(0, str(submission_dir))
    from task_e_curobo_planner import TaskECuRoboPlanner, TaskECuRoboPlannerProcess  # noqa: WPS433

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    obs, _ = env.reset()

    robot = env.unwrapped.scene["robot"]
    ee_idx = robot.body_names.index("gripper_base")
    arm_ids, _ = robot.find_joints(ARM_JOINTS)
    gripper_ids, _ = robot.find_joints(GRIPPER_JOINTS)
    default_q = robot.data.default_joint_pos[0].detach().cpu().numpy().astype(np.float32)

    planner = TaskECuRoboPlannerProcess(
        num_seeds=args.num_seeds,
        position_tolerance_m=0.006,
        request_timeout_s=120.0,
        scene_collision_check=args.scene_collision,
        self_collision_check=False,
    )
    total_score = 0.0
    stage_records: list[dict] = []
    start_time = time.time()

    def cuboid_world_to_root(name: str, center_w: list[float], dims: list[float]) -> dict:
        pos_root, quat_root = world_pose_to_root(
            robot,
            np.asarray(center_w, dtype=np.float32),
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            R,
        )
        return {
            "name": name,
            "pose": pos_root.astype(float).tolist() + quat_root.astype(float).tolist(),
            "dims": [float(v) for v in dims],
        }

    def build_scene_cuboids() -> list[dict]:
        cuboids = [
            cuboid_world_to_root(
                "table",
                [TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_TOP_Z * 0.5],
                [TABLE_DIMS[0] + 0.04, TABLE_DIMS[1] + 0.04, TABLE_TOP_Z],
            )
        ]
        if args.basket_obstacle:
            cuboids.append(
                cuboid_world_to_root(
                    "basket_outer",
                    [BASKET_CENTER_X, BASKET_CENTER_Y, TABLE_TOP_Z + 0.075],
                    [0.46, 0.30, 0.15],
                )
            )
        return cuboids

    scene_cuboids = build_scene_cuboids()
    if args.scene_collision:
        planner.update_world_cuboids(scene_cuboids)

    def step_joint_target(target_joint_pos: np.ndarray, steps: int) -> None:
        nonlocal obs, total_score
        action_np = ((target_joint_pos - default_q) / ACTION_SCALE).astype(np.float32)
        action = torch.as_tensor(action_np, dtype=torch.float32, device=args.device).view(1, -1)
        for _ in range(int(steps)):
            obs, reward, terminated, truncated, info = env.step(action)
            sim_dt = info.get("Step_dt", env.unwrapped.step_dt)
            sim_dt = sim_dt.item() if hasattr(sim_dt, "item") else float(sim_dt)
            total_score += float(reward.mean().item()) / sim_dt
            if bool(terminated.item() or truncated.item()):
                break

    def solve_and_execute(label: str, pos_root: np.ndarray, quat_root: np.ndarray, gripper: np.ndarray, hold_steps: int) -> dict:
        current_q = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().astype(np.float32)
        ik = planner.solve_ik(current_q, pos_root, quat_root)
        waypoints = TaskECuRoboPlanner.interpolate_joint_waypoints(
            current_q,
            ik.joint_position,
            max_step_rad=args.waypoint_max_step_rad,
        )
        for waypoint in waypoints:
            target = default_q.copy()
            target[arm_ids] = waypoint
            target[gripper_ids] = gripper
            step_joint_target(target, args.waypoint_hold_steps)
        target = default_q.copy()
        target[arm_ids] = ik.joint_position
        target[gripper_ids] = gripper
        step_joint_target(target, hold_steps)
        ee_pos, ee_quat = pose_in_robot_root(robot, ee_idx, R)
        record = {
            "label": label,
            "ik_success": bool(ik.success),
            "ik_position_error_m": float(ik.position_error_m),
            "ik_rotation_error_rad": float(ik.rotation_error_rad),
            "num_waypoints": len(waypoints),
            "target_pos_root": pos_root.astype(float).tolist(),
            "final_ee_pos_root": ee_pos.astype(float).tolist(),
            "final_ee_position_error_m": float(np.linalg.norm(ee_pos.astype(np.float64) - pos_root.astype(np.float64))),
            "final_quat_abs_dot": quat_abs_dot(ee_quat, quat_root),
        }
        stage_records.append(record)
        return record

    def make_grasp_base_position(
        candidate_center_w: np.ndarray,
        bbox_min_w: np.ndarray,
        bbox_max_w: np.ndarray,
        quat_w: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        if args.grasp_height_mode == "offset":
            base_pos = candidate_center_w + np.asarray([0.0, 0.0, args.grasp_z_offset], dtype=np.float32)
            tip_pos = gripper_tip_center_w(base_pos, quat_w, args.tool_tip_offset)
        else:
            min_base_z = float(bbox_max_w[2] + args.palm_clearance)
            tip_z = float(bbox_max_w[2] - args.tip_insert_depth)
            tip_z = min(max(tip_z, float(bbox_min_w[2] + 0.01)), float(bbox_max_w[2] - 0.004))
            tip_pos = np.asarray([candidate_center_w[0], candidate_center_w[1], tip_z], dtype=np.float32)
            base_pos = gripper_base_from_tip_w(tip_pos, quat_w, args.tool_tip_offset)
            if float(base_pos[2]) < min_base_z:
                base_pos[2] = min_base_z
                tip_pos = gripper_tip_center_w(base_pos, quat_w, args.tool_tip_offset)
            if float(tip_pos[2]) > float(bbox_max_w[2] - 0.004):
                tip_pos[2] = float(bbox_max_w[2] - 0.004)
                base_pos = gripper_base_from_tip_w(tip_pos, quat_w, args.tool_tip_offset)
                if float(base_pos[2]) < min_base_z:
                    base_pos[2] = min_base_z
                    tip_pos = gripper_tip_center_w(base_pos, quat_w, args.tool_tip_offset)
            if float(tip_pos[2]) < float(bbox_min_w[2] + 0.01):
                tip_pos[2] = float(bbox_min_w[2] + 0.01)
                base_pos = gripper_base_from_tip_w(tip_pos, quat_w, args.tool_tip_offset)
            tip_pos = gripper_tip_center_w(base_pos, quat_w, args.tool_tip_offset)

        tip_inside = bool(np.all(tip_pos >= bbox_min_w) and np.all(tip_pos <= bbox_max_w))
        clearance = float(base_pos[2] - bbox_max_w[2])
        return base_pos.astype(np.float32), {
            "tip_center_w": tip_pos.astype(float).tolist(),
            "tip_insert_depth_m": float(bbox_max_w[2] - tip_pos[2]),
            "tip_inside_bbox": tip_inside,
            "palm_clearance_m": clearance,
            "target_bbox_min_w": bbox_min_w.astype(float).tolist(),
            "target_bbox_max_w": bbox_max_w.astype(float).tolist(),
        }

    def select_topdown_grasp_quat(
        pre_pos_w: np.ndarray,
        grasp_pos_w: np.ndarray,
        lift_pos_w: np.ndarray,
    ) -> tuple[np.ndarray, list[dict]]:
        current_q = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().astype(np.float32)
        candidates: list[dict] = []
        yaw_count = max(1, int(args.yaw_candidates))
        for yaw in np.linspace(-np.pi, np.pi, yaw_count, endpoint=False, dtype=np.float32):
            quat_w = topdown_quat_wxyz_from_yaw(float(yaw), R)
            pre_root, pre_quat = world_pose_to_root(robot, pre_pos_w, quat_w, R)
            grasp_root, grasp_quat = world_pose_to_root(robot, grasp_pos_w, quat_w, R)
            lift_root, lift_quat = world_pose_to_root(robot, lift_pos_w, quat_w, R)

            q = current_q.copy()
            candidate_stages = []
            for label, pos_root, quat_root in (
                ("pregrasp_open", pre_root, pre_quat),
                ("grasp_open", grasp_root, grasp_quat),
                ("lift_closed", lift_root, lift_quat),
            ):
                ik = planner.solve_ik(q, pos_root, quat_root)
                candidate_stages.append(
                    {
                        "label": label,
                        "ik_success": bool(ik.success),
                        "ik_position_error_m": float(ik.position_error_m),
                        "ik_rotation_error_rad": float(ik.rotation_error_rad),
                    }
                )
                q = ik.joint_position

            pos_errors = [stage["ik_position_error_m"] for stage in candidate_stages]
            rot_errors = [stage["ik_rotation_error_rad"] for stage in candidate_stages]
            candidates.append(
                {
                    "yaw_rad": float(yaw),
                    "quat_wxyz": quat_w.astype(float).tolist(),
                    "all_success": bool(all(stage["ik_success"] for stage in candidate_stages)),
                    "max_position_error_m": float(max(pos_errors)),
                    "sum_position_error_m": float(sum(pos_errors)),
                    "max_rotation_error_rad": float(max(rot_errors)),
                    "stages": candidate_stages,
                }
            )

        candidates.sort(
            key=lambda item: (
                not item["all_success"],
                item["max_position_error_m"],
                item["sum_position_error_m"],
                item["max_rotation_error_rad"],
            )
        )
        return np.asarray(candidates[0]["quat_wxyz"], dtype=np.float32), candidates

    def select_reachable_center(
        center_w: np.ndarray,
        bbox_min_w: np.ndarray,
        bbox_max_w: np.ndarray,
        quat_w: np.ndarray,
    ) -> tuple[np.ndarray, list[dict]]:
        current_q = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().astype(np.float32)
        grid_count = max(1, int(args.search_grid))
        margin_xy = np.asarray([0.006, 0.006], dtype=np.float32)
        candidates: list[dict] = []
        for dx in np.linspace(-args.search_x_radius, args.search_x_radius, grid_count, dtype=np.float32):
            for dy in np.linspace(-args.search_y_radius, args.search_y_radius, grid_count, dtype=np.float32):
                candidate_center = center_w.copy()
                candidate_center[:2] += np.asarray([dx, dy], dtype=np.float32)
                if np.any(candidate_center[:2] < bbox_min_w[:2] + margin_xy):
                    continue
                if np.any(candidate_center[:2] > bbox_max_w[:2] - margin_xy):
                    continue

                grasp_pos, clearance = make_grasp_base_position(candidate_center, bbox_min_w, bbox_max_w, quat_w)
                pre_pos = grasp_pos + np.asarray([0.0, 0.0, args.pregrasp_height], dtype=np.float32)
                pre_root, pre_quat = world_pose_to_root(robot, pre_pos, quat_w, R)
                grasp_root, grasp_quat = world_pose_to_root(robot, grasp_pos, quat_w, R)

                q = current_q.copy()
                candidate_stages = []
                for label, pos_root, quat_root in (
                    ("pregrasp_open", pre_root, pre_quat),
                    ("grasp_open", grasp_root, grasp_quat),
                ):
                    ik = planner.solve_ik(q, pos_root, quat_root)
                    candidate_stages.append(
                        {
                            "label": label,
                            "ik_success": bool(ik.success),
                            "ik_position_error_m": float(ik.position_error_m),
                            "ik_rotation_error_rad": float(ik.rotation_error_rad),
                        }
                    )
                    q = ik.joint_position

                pos_errors = [stage["ik_position_error_m"] for stage in candidate_stages]
                rot_errors = [stage["ik_rotation_error_rad"] for stage in candidate_stages]
                candidates.append(
                    {
                        "center_w": candidate_center.astype(float).tolist(),
                        "offset_xy_w": [float(dx), float(dy)],
                        "offset_l1_m": float(abs(dx) + abs(dy)),
                        "inside_bbox_xy": True,
                        "gripper_clearance": clearance,
                        "all_success": bool(all(stage["ik_success"] for stage in candidate_stages)),
                        "max_position_error_m": float(max(pos_errors)),
                        "sum_position_error_m": float(sum(pos_errors)),
                        "grasp_position_error_m": float(candidate_stages[1]["ik_position_error_m"]),
                        "max_rotation_error_rad": float(max(rot_errors)),
                        "stages": candidate_stages,
                    }
                )

        if not candidates:
            return center_w, []
        candidates.sort(
            key=lambda item: (
                not item["gripper_clearance"]["tip_inside_bbox"],
                item["gripper_clearance"]["palm_clearance_m"] < args.palm_clearance,
                item["max_position_error_m"] > 0.035,
                item["grasp_position_error_m"] > 0.03,
                not item["all_success"],
                item["offset_l1_m"],
                item["grasp_position_error_m"],
                item["max_position_error_m"],
            )
        )
        return np.asarray(candidates[0]["center_w"], dtype=np.float32), candidates

    def select_relaxed_lift_pose(grasp_pos_w: np.ndarray, quat_w: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        current_q = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().astype(np.float32)
        xy_radius = float(args.lift_xy_radius)
        dx_values = [-xy_radius, -0.5 * xy_radius, 0.0, 0.5 * xy_radius]
        dy_values = [-0.5 * xy_radius, 0.0, 0.5 * xy_radius]
        lift_values = sorted({float(args.lift_min_height), 0.5 * (float(args.lift_min_height) + float(args.lift_height)), float(args.lift_height)})
        quat_candidates = (
            local_tilt_quat_candidates(quat_w, [float(v) for v in args.lift_tilt_degrees], R)
            if args.lift_orientation_mode == "tilt_search"
            else [
                {
                    "label": "fixed",
                    "tilt_axis": "none",
                    "tilt_degrees": 0.0,
                    "tilt_l1_degrees": 0.0,
                    "quat_wxyz": np.asarray(quat_w, dtype=np.float32),
                }
            ]
        )
        candidates: list[dict] = []
        for dz in lift_values:
            for dx in dx_values:
                for dy in dy_values:
                    lift_pos = grasp_pos_w + np.asarray([dx, dy, dz], dtype=np.float32)
                    for quat_candidate in quat_candidates:
                        lift_quat_w = np.asarray(quat_candidate["quat_wxyz"], dtype=np.float32)
                        lift_root, lift_quat = world_pose_to_root(robot, lift_pos, lift_quat_w, R)
                        ik = planner.solve_ik(current_q, lift_root, lift_quat)
                        xy_distance = float(abs(dx) + abs(dy))
                        candidates.append(
                            {
                                "lift_pos_w": lift_pos.astype(float).tolist(),
                                "lift_quat_wxyz": lift_quat_w.astype(float).tolist(),
                                "offset_w": [float(dx), float(dy), float(dz)],
                                "ik_success": bool(ik.success),
                                "ik_position_error_m": float(ik.position_error_m),
                                "ik_rotation_error_rad": float(ik.rotation_error_rad),
                                "xy_l1_m": xy_distance,
                                "height_m": float(dz),
                                "quat_label": quat_candidate["label"],
                                "tilt_axis": quat_candidate["tilt_axis"],
                                "tilt_degrees": float(quat_candidate["tilt_degrees"]),
                                "tilt_l1_degrees": float(quat_candidate["tilt_l1_degrees"]),
                            }
                        )

        candidates.sort(
            key=lambda item: (
                item["ik_position_error_m"] > 0.02,
                not item["ik_success"],
                item["ik_position_error_m"],
                item["tilt_l1_degrees"],
                item["xy_l1_m"],
                -item["height_m"],
            )
        )
        return (
            np.asarray(candidates[0]["lift_pos_w"], dtype=np.float32),
            np.asarray(candidates[0]["lift_quat_wxyz"], dtype=np.float32),
            candidates,
        )

    try:
        perception = perceive_mustard(obs, output_dir)
        object_points_w = np.load(output_dir / "mustard_points_world.npy").astype(np.float32)
        center_w = np.asarray(perception["center_w"], dtype=np.float32)
        bbox_min_w = np.asarray(perception["bbox_min_w"], dtype=np.float32)
        bbox_max_w = np.asarray(perception["bbox_max_w"], dtype=np.float32)
        legacy_grasp_quat_w = np.asarray(
            task_e_grasp_quat(OBJECTS["mustard_bottle"]["object_quat_wxyz"]),
            dtype=np.float32,
        )

        yaw_candidates: list[dict] = []
        position_candidates: list[dict] = []
        lift_candidates: list[dict] = []
        selected_candidate: dict | None = None
        candidate_evaluations: list[dict] = []
        candidate_meta: dict = {}
        grasp_clearance: dict | None = None

        if args.grasp_source in {"anygrasp", "graspgen"}:
            selected_candidate, candidate_evaluations, candidate_meta = select_curobo_generator_candidate(
                planner,
                robot,
                arm_ids,
                object_points_w,
                perception,
                args,
                R,
            )
            if selected_candidate is None:
                grasp_pos_w = center_w.copy()
                pre_pos_w = center_w.copy()
                lift_pos_w = center_w.copy()
                fixed_lift_pos_w = center_w.copy()
                grasp_quat_w = legacy_grasp_quat_w.copy()
                lift_quat_w = legacy_grasp_quat_w.copy()
            else:
                stages = {stage["label"]: stage for stage in selected_candidate["stages"]}
                for label in ("pregrasp_stage_open", "pregrasp_open", "grasp_open"):
                    stage = stages[label]
                    solve_and_execute(
                        label,
                        np.asarray(stage["target_pos_root"], dtype=np.float32),
                        np.asarray(stage["target_quat_root_wxyz"], dtype=np.float32),
                        GRIPPER_OPEN,
                        args.final_hold_steps,
                    )
                close_target = default_q.copy()
                close_target[arm_ids] = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().astype(np.float32)
                close_target[gripper_ids] = GRIPPER_CLOSE
                step_joint_target(close_target, 45)
                lift_stage = stages["lift_closed"]
                solve_and_execute(
                    "lift_closed",
                    np.asarray(lift_stage["target_pos_root"], dtype=np.float32),
                    np.asarray(lift_stage["target_quat_root_wxyz"], dtype=np.float32),
                    GRIPPER_CLOSE,
                    args.final_hold_steps,
                )
                pre_pos_w = np.asarray(stages["pregrasp_open"]["target_pos_w"], dtype=np.float32)
                grasp_pos_w = np.asarray(stages["grasp_open"]["target_pos_w"], dtype=np.float32)
                lift_pos_w = np.asarray(lift_stage["target_pos_w"], dtype=np.float32)
                fixed_lift_pos_w = lift_pos_w.copy()
                grasp_quat_w = np.asarray(stages["grasp_open"]["target_quat_wxyz"], dtype=np.float32)
                lift_quat_w = np.asarray(lift_stage["target_quat_wxyz"], dtype=np.float32)
                grasp_clearance = selected_candidate.get("approach_collision_check", {}).get("final_stats")
        elif args.orientation_mode == "legacy":
            grasp_quat_w = legacy_grasp_quat_w.copy()
            yaw_candidates = []
        else:
            rough_grasp_pos_w = center_w + np.asarray([0.0, 0.0, args.grasp_z_offset], dtype=np.float32)
            rough_pre_pos_w = rough_grasp_pos_w + np.asarray([0.0, 0.0, args.pregrasp_height], dtype=np.float32)
            rough_lift_pos_w = rough_grasp_pos_w + np.asarray([0.0, 0.0, args.lift_height], dtype=np.float32)
            grasp_quat_w, yaw_candidates = select_topdown_grasp_quat(
                rough_pre_pos_w,
                rough_grasp_pos_w,
                rough_lift_pos_w,
            )

        if args.grasp_source == "heuristic" and args.position_search and args.orientation_mode == "legacy":
            center_w, position_candidates = select_reachable_center(
                center_w,
                bbox_min_w,
                bbox_max_w,
                legacy_grasp_quat_w,
            )
        if args.grasp_source == "heuristic":
            grasp_pos_w, grasp_clearance = make_grasp_base_position(center_w, bbox_min_w, bbox_max_w, grasp_quat_w)
            pre_pos_w = grasp_pos_w + np.asarray([0.0, 0.0, args.pregrasp_height], dtype=np.float32)
            fixed_lift_pos_w = grasp_pos_w + np.asarray([0.0, 0.0, args.lift_height], dtype=np.float32)
            lift_quat_w = grasp_quat_w.copy()
            if args.lift_mode == "fixed":
                lift_pos_w = fixed_lift_pos_w
            else:
                lift_pos_w = fixed_lift_pos_w

            pre_pos_root, pre_quat_root = world_pose_to_root(robot, pre_pos_w, grasp_quat_w, R)
            grasp_pos_root, grasp_quat_root = world_pose_to_root(robot, grasp_pos_w, grasp_quat_w, R)

            solve_and_execute("pregrasp_open", pre_pos_root, pre_quat_root, GRIPPER_OPEN, args.final_hold_steps)
            solve_and_execute("grasp_open", grasp_pos_root, grasp_quat_root, GRIPPER_OPEN, args.final_hold_steps)
            close_target = default_q.copy()
            close_target[arm_ids] = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().astype(np.float32)
            close_target[gripper_ids] = GRIPPER_CLOSE
            step_joint_target(close_target, 45)
            if args.lift_mode == "relaxed":
                lift_pos_w, lift_quat_w, lift_candidates = select_relaxed_lift_pose(grasp_pos_w, grasp_quat_w)
            lift_pos_root, lift_quat_root = world_pose_to_root(robot, lift_pos_w, lift_quat_w, R)
            solve_and_execute("lift_closed", lift_pos_root, lift_quat_root, GRIPPER_CLOSE, args.final_hold_steps)

        object_pos = (env.unwrapped.scene["object_2"].data.root_pos_w[0, :3] - env.unwrapped.scene.env_origins[0]).detach().cpu().numpy()
        ee_pos_root, _ = pose_in_robot_root(robot, ee_idx, R)
        summary = {
            "seed": args.seed,
            "pass": bool(object_pos[2] > TABLE_TOP_Z + 0.08),
            "perception": perception,
            "grasp_pos_w": grasp_pos_w.astype(float).tolist(),
            "pre_pos_w": pre_pos_w.astype(float).tolist(),
            "lift_pos_w": lift_pos_w.astype(float).tolist(),
            "lift_quat_wxyz": lift_quat_w.astype(float).tolist(),
            "fixed_lift_pos_w": fixed_lift_pos_w.astype(float).tolist(),
            "grasp_clearance": grasp_clearance,
            "legacy_grasp_quat_wxyz": legacy_grasp_quat_w.astype(float).tolist(),
            "scene_collision": bool(args.scene_collision),
            "scene_cuboids": scene_cuboids,
            "grasp_source": args.grasp_source,
            "candidate_meta": candidate_meta,
            "selected_candidate": selected_candidate,
            "candidate_evaluations": candidate_evaluations,
            "grasp_height_mode": args.grasp_height_mode,
            "lift_mode": args.lift_mode,
            "lift_orientation_mode": args.lift_orientation_mode,
            "selected_lift_candidate": lift_candidates[0] if lift_candidates else None,
            "lift_candidates": lift_candidates,
            "orientation_mode": args.orientation_mode,
            "position_search": bool(args.position_search),
            "selected_position_candidate": position_candidates[0] if position_candidates else None,
            "position_candidates": position_candidates,
            "selected_grasp_quat_wxyz": grasp_quat_w.astype(float).tolist(),
            "selected_yaw_candidate": yaw_candidates[0] if yaw_candidates else None,
            "yaw_candidates": yaw_candidates,
            "stage_records": stage_records,
            "mustard_final_pos_local": object_pos.astype(float).tolist(),
            "mustard_lift_above_table_m": float(object_pos[2] - TABLE_TOP_Z),
            "final_ee_pos_root": ee_pos_root.astype(float).tolist(),
            "total_score_estimate": float(total_score),
            "wall_time_s": float(time.time() - start_time),
        }
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    finally:
        planner.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
