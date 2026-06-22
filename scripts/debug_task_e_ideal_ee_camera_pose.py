#!/usr/bin/env python3
"""Debug the intended EE-camera pose computed from Task E video-camera masks.

This script isolates the first perception-to-camera-target step of the real
Task E grasping pipeline:

1. Reset Task E and capture the external video camera.
2. Segment one or more objects in that video image.
3. Back-project the segmented mask with video depth to estimate object center.
4. Compute the intended EE-camera world pose using the same hover logic as the
   full EE-camera grasp pipeline.
5. Spawn fixed virtual cameras at those intended poses and draw frustum markers
   in the scene.
6. Save both an overview image from the external camera and the RGB output from
   each virtual camera.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from isaaclab.app import AppLauncher

from task_e_full_baseline_request import (
    DEFAULT_PICK_ORDER,
    GRIPPER_OPEN,
    OBJECTS,
    TABLE_TOP_Z,
    quat_wxyz_from_matrix,
)


OBJECT_PROMPTS = {
    "banana": "banana",
    "mustard_bottle": "mustard bottle",
    "box_object": "yellow and white box",
}
TOP_DOWN_GRIPPER_QUAT_WXYZ = [0.0, 1.0, 0.0, 0.0]
ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINT_NAMES = ["joint7", "joint8"]
ACTION_SCALE = 0.5


def parse_object_names(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(DEFAULT_PICK_ORDER)
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in OBJECTS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown objects: {unknown}. Valid: {sorted(OBJECTS)}")
    return names


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--objects",
        type=parse_object_names,
        default=list(DEFAULT_PICK_ORDER),
        help="Comma-separated object names or 'all'. Default: banana,mustard_bottle,box_object.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/task_e_ideal_ee_camera_debug/<timestamp>.",
    )
    parser.add_argument(
        "--reuse-computed",
        action="store_true",
        help=(
            "Skip video capture/SAM3 and read existing <object>/intended_camera_pose.json "
            "from --output, then only render debug cameras/frustums."
        ),
    )
    parser.add_argument(
        "--compute-only",
        action="store_true",
        help="Only compute video segmentation and intended poses; do not spawn virtual debug cameras.",
    )
    parser.add_argument("--sam3-env", default="sam3_full")
    parser.add_argument("--sam3-device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--no-sam3", action="store_true", help="Reuse existing masks in the output directory.")
    parser.add_argument("--force-sam3", action="store_true")
    parser.add_argument(
        "--hover-mode",
        choices=("gripper_offset", "camera_center", "look_at", "adaptive_lookat"),
        default="adaptive_lookat",
    )
    parser.add_argument(
        "--lookat-objects",
        type=parse_object_names,
        default=["mustard_bottle", "box_object"],
        help="Objects using look_at when --hover-mode adaptive_lookat.",
    )
    parser.add_argument(
        "--video-color-refine-objects",
        type=parse_object_names,
        default=["banana"],
        help="Objects whose video SAM3 pose is refined by color component overlap.",
    )
    parser.add_argument("--hover-height", type=float, default=0.23)
    parser.add_argument("--gripper-hover-offset", type=parse_float_list, default=[0.05, 0.0, 0.0])
    parser.add_argument("--lookat-camera-position", type=parse_float_list, default=[])
    parser.add_argument("--lookat-camera-offset", type=parse_float_list, default=[0.0, 0.0, 0.0])
    parser.add_argument(
        "--real-ee-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also move the real robot-mounted EE camera to each intended pose and save its image.",
    )
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Skip virtual debug cameras and only run the real mounted EE-camera move/capture stage.",
    )
    parser.add_argument("--real-hover-settle-steps", type=int, default=260)
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--debug-steps", type=int, default=8, help="Steps after spawning virtual cameras.")
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


args_cli = parse_args()


def make_output_dir(path: Path | None) -> Path:
    if path is None:
        path = REPO_ROOT / "outputs/task_e_ideal_ee_camera_debug" / datetime.now().strftime("%Y%m%d_%H%M%S")
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


OUTPUT_DIR = make_output_dir(args_cli.output)


def write_failure(name: str, exc: BaseException) -> None:
    (OUTPUT_DIR / name).write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "repr": repr(exc),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


try:
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
except BaseException as exc:
    write_failure("launch_error.json", exc)
    raise

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401,E402
from atec_rl_lab.utils import CartesianController  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def to_numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def to_rgb_array(value) -> np.ndarray:
    array = to_numpy(value)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        if array.size and float(np.nanmax(array)) <= 1.0:
            array *= 255.0
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def to_depth_array(value) -> np.ndarray:
    array = to_numpy(value).astype(np.float32)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError(f"Expected depth image, got {array.shape}")
    return np.ascontiguousarray(array)


def save_depth_preview(path: Path, depth: np.ndarray, max_depth: float) -> None:
    valid = np.isfinite(depth) & (depth > 0.0)
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        clipped = np.clip(depth, 0.0, max_depth)
        scaled = (255.0 * (1.0 - clipped / max_depth)).astype(np.uint8)
        scaled[~valid] = 0
    Image.fromarray(scaled, mode="L").save(path)


def tensor_to_list(value) -> list[float]:
    return value.detach().cpu().numpy().astype(float).tolist()


def camera_metadata(env, sensor_name: str) -> dict:
    camera = env.unwrapped.scene.sensors[sensor_name]
    data = camera.data
    return {
        "sensor": sensor_name,
        "image_shape": list(data.image_shape),
        "intrinsic_matrix": tensor_to_list(data.intrinsic_matrices[0]),
        "pos_w": tensor_to_list(data.pos_w[0]),
        "quat_w_world": tensor_to_list(data.quat_w_world[0]),
        "quat_w_ros": tensor_to_list(data.quat_w_ros[0]),
        "quat_w_opengl": tensor_to_list(data.quat_w_opengl[0]),
    }


def write_camera_json(path: Path, camera: dict) -> None:
    path.write_text(json.dumps({"camera": camera}, indent=2), encoding="utf-8")


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if mask.size != (shape[1], shape[0]):
        mask = mask.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(mask) > 127


def quat_normalize(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def quat_inverse(quat: np.ndarray) -> np.ndarray:
    quat = quat_normalize(quat)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = quat_normalize(left)
    w2, x2, y2, z2 = quat_normalize(right)
    return quat_normalize(
        np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )
    )


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def backproject(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    z = depth.astype(np.float32)
    x = (xx - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (yy - intrinsic[1, 2]) * z / intrinsic[1, 1]
    return np.stack([x, y, z], axis=-1)


def transform_points_to_world(points_cam: np.ndarray, camera: dict) -> np.ndarray:
    rot = quat_wxyz_to_matrix(np.asarray(camera["quat_w_ros"], dtype=np.float64))
    pos = np.asarray(camera["pos_w"], dtype=np.float64)
    return points_cam @ rot.T + pos


def estimate_pose_from_mask(depth: np.ndarray, mask: np.ndarray, camera: dict, max_depth: float) -> dict:
    intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float32)
    points_cam_all = backproject(depth, intrinsic)
    valid = mask & np.isfinite(depth) & (depth > 0.0) & (depth < max_depth)
    if not valid.any():
        raise ValueError("Mask has no valid depth pixels.")
    yy, xx = np.where(valid)
    points_world = transform_points_to_world(points_cam_all[valid], camera)
    center_median = np.median(points_world, axis=0)
    center_mean = np.mean(points_world, axis=0)
    return {
        "center_world": center_median.astype(float).tolist(),
        "center_world_median": center_median.astype(float).tolist(),
        "center_world_mean": center_mean.astype(float).tolist(),
        "point_count": int(points_world.shape[0]),
        "pixel_center": [float(np.median(xx)), float(np.median(yy))],
        "bbox_xyxy": [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())],
    }


def draw_target_overlay(image_path: Path, output_path: Path, pose: dict, title: str, mask_path: Path | None = None) -> None:
    image = Image.open(image_path).convert("RGBA")
    if mask_path is not None and mask_path.exists():
        mask = load_mask(mask_path, (image.height, image.width))
        layer = np.zeros((image.height, image.width, 4), dtype=np.uint8)
        layer[mask] = (255, 204, 0, 85)
        image = Image.alpha_composite(image, Image.fromarray(layer, mode="RGBA"))
    draw = ImageDraw.Draw(image)
    bbox = pose.get("bbox_xyxy")
    if bbox is not None:
        draw.rectangle(tuple(bbox), outline=(255, 60, 60, 255), width=3)
    center = pose.get("pixel_center")
    if center is not None:
        x, y = center
        draw.line((x - 14, y, x + 14, y), fill=(255, 255, 0, 255), width=3)
        draw.line((x, y - 14, x, y + 14), fill=(255, 255, 0, 255), width=3)
    text = title
    center_world = pose.get("center_world_median") or pose.get("center_world")
    if center_world is not None:
        text += f" | world {np.round(center_world, 3).tolist()}"
    draw.rectangle((0, 0, image.width, 34), fill=(0, 0, 0, 190))
    draw.text((8, 10), text, fill=(255, 255, 255, 255))
    image.convert("RGB").save(output_path)


def color_candidate_mask(rgb: np.ndarray, object_name: str) -> np.ndarray:
    rgb_i = rgb.astype(np.int16)
    r, g, b = rgb_i[..., 0], rgb_i[..., 1], rgb_i[..., 2]
    yellow = (r > 145) & (g > 105) & (b < 135) & ((r - b) > 55) & ((g - b) > 35)
    white = (r > 165) & (g > 165) & (b > 145) & (np.abs(r - g) < 45) & (np.abs(g - b) < 65)
    dark_table = (r < 95) & (g < 95) & (b < 95)
    if object_name == "box_object":
        return ((yellow | white) & ~dark_table).astype(bool)
    return (yellow & ~dark_table).astype(bool)


def select_color_component_overlapping_guide(color_mask: np.ndarray, guide_mask: np.ndarray) -> np.ndarray | None:
    try:
        import cv2

        kernel = np.ones((3, 3), dtype=np.uint8)
        cleaned = cv2.morphologyEx(color_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
        best_idx = None
        best_overlap = 0
        best_area = 0
        for idx in range(1, count):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < 50:
                continue
            component = labels == idx
            overlap = int(np.count_nonzero(component & guide_mask))
            if overlap > best_overlap or (overlap == best_overlap and overlap > 0 and area > best_area):
                best_idx = idx
                best_overlap = overlap
                best_area = area
        if best_idx is None or best_overlap < 20:
            return None
        return labels == best_idx
    except Exception:
        overlap = color_mask.astype(bool) & guide_mask
        return overlap if np.count_nonzero(overlap) >= 20 else None


def maybe_refine_pose_with_color(
    object_dir: Path,
    image_path: Path,
    depth: np.ndarray,
    camera: dict,
    guide_mask: np.ndarray,
    object_name: str,
) -> tuple[dict | None, Path | None]:
    if object_name not in args_cli.video_color_refine_objects:
        return None, None
    image = Image.open(image_path).convert("RGB")
    component = select_color_component_overlapping_guide(color_candidate_mask(np.asarray(image), object_name), guide_mask)
    if component is None:
        return None, None
    mask_path = object_dir / f"video_{object_name}_color_refine_mask.png"
    Image.fromarray(component.astype(np.uint8) * 255, mode="L").save(mask_path)
    pose = estimate_pose_from_mask(depth, component, camera, args_cli.max_depth)
    pose["source"] = "video_cam_color_component_overlapping_sam3_depth"
    draw_target_overlay(
        image_path,
        object_dir / "video_color_refined_target_overlay.png",
        pose,
        f"{object_name} color-refined target",
        mask_path,
    )
    return pose, mask_path


def run_sam3(image_path: Path, object_dir: Path, object_name: str) -> tuple[Path, dict]:
    object_dir.mkdir(parents=True, exist_ok=True)
    label = f"video_{object_name}"
    mask_path = object_dir / f"{label}_mask.png"
    detection_path = object_dir / f"{label}_detections.json"
    if args_cli.no_sam3:
        if not mask_path.exists() or not detection_path.exists():
            raise FileNotFoundError(f"Missing existing SAM3 output: {mask_path}")
        return mask_path, json.loads(detection_path.read_text(encoding="utf-8"))
    if args_cli.force_sam3 or not mask_path.exists() or not detection_path.exists():
        command = [
            "conda",
            "run",
            "-n",
            args_cli.sam3_env,
            "python",
            "scripts/sam3_single_image_mask.py",
            "--image",
            str(image_path),
            "--prompt",
            OBJECT_PROMPTS[object_name],
            "--label",
            label,
            "--view-label",
            "eye-to-hand / video_cam",
            "--output",
            str(object_dir),
            "--device",
            args_cli.sam3_device,
        ]
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (object_dir / f"{label}_sam3.log").write_text(proc.stdout or "", encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"SAM3 failed for {object_name}. See {object_dir / f'{label}_sam3.log'}")
    return mask_path, json.loads(detection_path.read_text(encoding="utf-8"))


def normalize_or_fallback(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        return np.asarray(fallback, dtype=np.float64)
    return np.asarray(value, dtype=np.float64) / norm


def lookat_camera_quat_wxyz(camera_pos_w: np.ndarray, target_pos_w: np.ndarray) -> np.ndarray:
    optical_axis = normalize_or_fallback(target_pos_w - camera_pos_w, np.array([-1.0, 0.0, -1.0]))
    image_down_hint = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    image_down = image_down_hint - optical_axis * float(np.dot(image_down_hint, optical_axis))
    if np.linalg.norm(image_down) <= 1e-6:
        image_down_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        image_down = image_down_hint - optical_axis * float(np.dot(image_down_hint, optical_axis))
    image_down = normalize_or_fallback(image_down, np.array([0.0, 1.0, 0.0]))
    image_right = normalize_or_fallback(np.cross(image_down, optical_axis), np.array([1.0, 0.0, 0.0]))
    image_down = normalize_or_fallback(np.cross(optical_axis, image_right), image_down)
    rot = np.stack([image_right, image_down, optical_axis], axis=1)
    return np.asarray(quat_wxyz_from_matrix(rot), dtype=np.float64)


def calibrate_ee_camera(env, robot, ee_idx: int) -> dict:
    sensor_camera = camera_metadata(env, "ee_camera")
    grip_pose_w = robot.data.body_pose_w[0, ee_idx]
    grip_pos_w = np.asarray(tensor_to_list(grip_pose_w[:3]), dtype=np.float64)
    grip_quat_w = np.asarray(tensor_to_list(grip_pose_w[3:]), dtype=np.float64)
    cam_pos_w = np.asarray(sensor_camera["pos_w"], dtype=np.float64)
    cam_quat_w_ros = np.asarray(sensor_camera["quat_w_ros"], dtype=np.float64)
    grip_to_cam_pos = quat_wxyz_to_matrix(grip_quat_w).T @ (cam_pos_w - grip_pos_w)
    grip_to_cam_quat = quat_multiply(quat_inverse(grip_quat_w), cam_quat_w_ros)
    return {
        "source": "reset_sensor_pose_and_gripper_body_pose",
        "ee_body_idx": ee_idx,
        "reset_camera_sensor": sensor_camera,
        "reset_gripper_pos_w": grip_pos_w.tolist(),
        "reset_gripper_quat_wxyz": grip_quat_w.tolist(),
        "gripper_to_camera_pos": grip_to_cam_pos.tolist(),
        "gripper_to_camera_quat_ros": grip_to_cam_quat.tolist(),
    }


def ee_camera_metadata_from_gripper(env, robot, ee_idx: int, calibration: dict) -> dict:
    metadata = camera_metadata(env, "ee_camera")
    grip_pose_w = robot.data.body_pose_w[0, ee_idx]
    grip_pos_w = np.asarray(tensor_to_list(grip_pose_w[:3]), dtype=np.float64)
    grip_quat_w = np.asarray(tensor_to_list(grip_pose_w[3:]), dtype=np.float64)
    grip_to_cam_pos = np.asarray(calibration["gripper_to_camera_pos"], dtype=np.float64)
    grip_to_cam_quat = np.asarray(calibration["gripper_to_camera_quat_ros"], dtype=np.float64)
    cam_pos_w = grip_pos_w + quat_wxyz_to_matrix(grip_quat_w) @ grip_to_cam_pos
    cam_quat_w_ros = quat_multiply(grip_quat_w, grip_to_cam_quat)
    metadata["pos_w_sensor_raw"] = metadata["pos_w"]
    metadata["quat_w_ros_sensor_raw"] = metadata["quat_w_ros"]
    metadata["pos_w"] = cam_pos_w.astype(float).tolist()
    metadata["quat_w_ros"] = cam_quat_w_ros.astype(float).tolist()
    metadata["pose_source"] = "gripper_body_pose_plus_reset_camera_calibration"
    return metadata


def gripper_pose_for_camera_pose(
    desired_camera_pos_w: np.ndarray,
    desired_camera_quat_w_ros: np.ndarray,
    ee_camera_calibration: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    grip_to_cam_pos = np.asarray(ee_camera_calibration["gripper_to_camera_pos"], dtype=np.float64)
    grip_to_cam_quat = np.asarray(ee_camera_calibration["gripper_to_camera_quat_ros"], dtype=np.float64)
    desired_grip_quat = quat_multiply(desired_camera_quat_w_ros, quat_inverse(grip_to_cam_quat))
    desired_grip_pos = desired_camera_pos_w - quat_wxyz_to_matrix(desired_grip_quat) @ grip_to_cam_pos
    return desired_grip_pos, desired_grip_quat, {
        "desired_camera_pos_w": desired_camera_pos_w.astype(float).tolist(),
        "desired_camera_quat_w_ros": desired_camera_quat_w_ros.astype(float).tolist(),
        "desired_gripper_pos_w": desired_grip_pos.astype(float).tolist(),
        "desired_gripper_quat_wxyz": desired_grip_quat.astype(float).tolist(),
        "gripper_to_camera_pos": grip_to_cam_pos.astype(float).tolist(),
        "gripper_to_camera_quat_ros": grip_to_cam_quat.astype(float).tolist(),
    }


def quat_abs_dot(left: list[float] | np.ndarray, right: list[float] | np.ndarray) -> float:
    left_q = quat_normalize(np.asarray(left, dtype=np.float64))
    right_q = quat_normalize(np.asarray(right, dtype=np.float64))
    return float(abs(np.dot(left_q, right_q)))


def move_real_ee_camera_to_pose(
    env,
    obs,
    robot,
    ee_camera_calibration: dict,
    desired_camera_pose: dict,
    steps: int,
) -> tuple[dict, dict]:
    arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
    gripper_ids, _ = robot.find_joints(GRIPPER_JOINT_NAMES)
    controller = CartesianController(
        robot=robot,
        ee_body_name="gripper_base",
        arm_joint_names=ARM_JOINT_NAMES,
        num_envs=1,
        device=env.unwrapped.device,
        command_type="pose",
        lambda_val=0.05,
        max_joint_delta=0.18,
    )
    controller.reset()
    desired_cam_pos = np.asarray(desired_camera_pose["desired_camera_pos_w"], dtype=np.float64)
    desired_cam_quat = np.asarray(desired_camera_pose["desired_camera_quat_w_ros"], dtype=np.float64)
    grip_pos_des, grip_quat_des, command_record = gripper_pose_for_camera_pose(
        desired_cam_pos,
        desired_cam_quat,
        ee_camera_calibration,
    )
    pos_t = torch.tensor([grip_pos_des], dtype=torch.float32, device=env.unwrapped.device)
    quat_t = torch.tensor([grip_quat_des], dtype=torch.float32, device=env.unwrapped.device)
    gripper_t = torch.tensor([GRIPPER_OPEN], dtype=torch.float32, device=env.unwrapped.device)
    default_jpos = robot.data.default_joint_pos.clone()
    stop_reason = None
    last_action = None
    for _ in range(max(1, int(steps))):
        arm_des = controller.compute(pos_t, quat_t)
        target = robot.data.joint_pos.clone()
        target[:, arm_ids] = arm_des
        target[:, gripper_ids] = gripper_t
        action = (target - default_jpos) / ACTION_SCALE
        last_action = action
        obs, _, terminated, truncated, _ = env.step(action)
        robot.update(dt=env.unwrapped.physics_dt)
        if terminated.any() or truncated.any():
            stop_reason = {
                "terminated": bool(terminated.any().item()),
                "truncated": bool(truncated.any().item()),
            }
            break

    ee_pose = robot.data.body_pose_w[0, controller.ee_idx]
    actual_camera = ee_camera_metadata_from_gripper(env, robot, controller.ee_idx, ee_camera_calibration)
    actual_cam_pos = np.asarray(actual_camera["pos_w"], dtype=np.float64)
    actual_cam_quat = np.asarray(actual_camera["quat_w_ros"], dtype=np.float64)
    command_record.update(
        {
            "final_ee_pos_w": tensor_to_list(ee_pose[:3]),
            "final_ee_quat_wxyz": tensor_to_list(ee_pose[3:]),
            "actual_camera": actual_camera,
            "camera_position_error_m": float(np.linalg.norm(actual_cam_pos - desired_cam_pos)),
            "camera_orientation_abs_dot": quat_abs_dot(actual_cam_quat, desired_cam_quat),
            "gripper_position_error_m": float(
                np.linalg.norm(np.asarray(tensor_to_list(ee_pose[:3]), dtype=np.float64) - grip_pos_des)
            ),
            "gripper_orientation_abs_dot": quat_abs_dot(tensor_to_list(ee_pose[3:]), grip_quat_des),
            "settle_steps": int(steps),
            "stop_reason": stop_reason,
            "last_action": tensor_to_list(last_action[0]) if last_action is not None else None,
        }
    )
    return obs, command_record


def intended_camera_pose(target_center_w: list[float], object_name: str, ee_camera_calibration: dict) -> dict:
    requested_mode = args_cli.hover_mode
    if requested_mode == "adaptive_lookat":
        effective_mode = "look_at" if object_name in set(args_cli.lookat_objects) else "gripper_offset"
    else:
        effective_mode = requested_mode

    target = np.asarray(target_center_w, dtype=np.float64)
    look_at_target = target
    if effective_mode == "look_at":
        if args_cli.lookat_camera_position:
            base_camera_pos = np.asarray(args_cli.lookat_camera_position[:3], dtype=np.float64)
        else:
            base_camera_pos = np.asarray(ee_camera_calibration["reset_camera_sensor"]["pos_w"], dtype=np.float64)
        offset = list(args_cli.lookat_camera_offset[:3])
        if len(offset) < 3:
            offset.extend([0.0] * (3 - len(offset)))
        desired_camera_pos = base_camera_pos + np.asarray(offset, dtype=np.float64)
        desired_camera_quat = lookat_camera_quat_wxyz(desired_camera_pos, look_at_target)
    elif effective_mode == "gripper_offset":
        offset = list(args_cli.gripper_hover_offset[:3])
        if len(offset) < 3:
            offset.extend([0.0] * (3 - len(offset)))
        desired_camera_pos = target + np.asarray(offset, dtype=np.float64)
        grip_quat = np.asarray(TOP_DOWN_GRIPPER_QUAT_WXYZ, dtype=np.float64)
        grip_to_cam_quat = np.asarray(ee_camera_calibration["gripper_to_camera_quat_ros"], dtype=np.float64)
        desired_camera_quat = quat_multiply(grip_quat, grip_to_cam_quat)
    else:
        desired_camera_pos = target
        desired_camera_quat = lookat_camera_quat_wxyz(desired_camera_pos, look_at_target)

    return {
        "requested_mode": requested_mode,
        "effective_mode": effective_mode,
        "target_object_center_w": target.astype(float).tolist(),
        "desired_camera_pos_w": desired_camera_pos.astype(float).tolist(),
        "desired_camera_quat_w_ros": desired_camera_quat.astype(float).tolist(),
        "look_at_target_w": look_at_target.astype(float).tolist(),
        "hover_height_argument": float(args_cli.hover_height),
        "gripper_hover_offset": list(args_cli.gripper_hover_offset),
        "lookat_camera_offset": list(args_cli.lookat_camera_offset),
    }


def camera_sensor_rgb(sensor) -> np.ndarray:
    output = sensor.data.output
    value = output.get("rgb")
    if value is None:
        value = output.get("rgba")
    if value is None:
        raise KeyError(f"Camera sensor {sensor} has no rgb/rgba output. Keys: {list(output.keys())}")
    return to_rgb_array(value)


def camera_sensor_depth(sensor) -> np.ndarray | None:
    output = sensor.data.output
    value = output.get("depth")
    if value is None:
        return None
    return to_depth_array(value)


def add_debug_camera_cfg(env_cfg, name: str, pose: dict) -> str:
    sensor_name = f"debug_{name}_cam"
    setattr(
        env_cfg.scene,
        sensor_name,
        CameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{sensor_name}",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=15.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 50.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=tuple(float(v) for v in pose["desired_camera_pos_w"]),
                rot=tuple(float(v) for v in pose["desired_camera_quat_w_ros"]),
                convention="ros",
            ),
        ),
    )
    return sensor_name


def draw_debug_frustums(records: dict[str, dict]) -> None:
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, Vt
    except Exception as exc:
        print(f"[WARN] Could not import USD drawing modules: {exc}", flush=True)
        return

    stage = omni.usd.get_context().get_stage()
    root = "/World/IdealEECameraDebug"
    UsdGeom.Xform.Define(stage, root)
    colors = {
        "banana": (1.0, 0.75, 0.05),
        "mustard_bottle": (0.1, 0.85, 1.0),
        "box_object": (1.0, 0.15, 0.15),
    }
    for name, record in records.items():
        pose = record["intended_camera_pose"]
        pos = np.asarray(pose["desired_camera_pos_w"], dtype=np.float64)
        quat = np.asarray(pose["desired_camera_quat_w_ros"], dtype=np.float64)
        rot = quat_wxyz_to_matrix(quat)
        x_axis = rot[:, 0]
        y_axis = rot[:, 1]
        z_axis = rot[:, 2]
        d = 0.18
        half_w = 0.09
        half_h = 0.065
        center = pos + z_axis * d
        corners = [
            center + x_axis * half_w + y_axis * half_h,
            center - x_axis * half_w + y_axis * half_h,
            center - x_axis * half_w - y_axis * half_h,
            center + x_axis * half_w - y_axis * half_h,
        ]
        segments = []
        for corner in corners:
            segments.append((pos, corner))
        for i in range(4):
            segments.append((corners[i], corners[(i + 1) % 4]))
        segments.extend(
            [
                (pos, pos + x_axis * 0.14),
                (pos, pos + y_axis * 0.14),
                (pos, pos + z_axis * 0.20),
            ]
        )
        points = []
        counts = []
        for start, end in segments:
            points.extend([Gf.Vec3f(*start.astype(float)), Gf.Vec3f(*end.astype(float))])
            counts.append(2)
        curve = UsdGeom.BasisCurves.Define(stage, f"{root}/{name}_frustum")
        curve.CreateTypeAttr("linear")
        curve.CreateCurveVertexCountsAttr(counts)
        curve.CreatePointsAttr(points)
        curve.CreateWidthsAttr(Vt.FloatArray([0.012] * len(points)))
        curve.CreateDisplayColorAttr([Gf.Vec3f(*colors.get(name, (1.0, 1.0, 1.0)))])


def save_labeled(path: Path, image: np.ndarray, label: str) -> None:
    pil = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(pil)
    draw.rectangle((0, 0, pil.width, 34), fill=(0, 0, 0))
    draw.text((8, 10), label, fill=(255, 255, 255))
    pil.save(path)


def create_env(with_debug_cameras: dict[str, dict] | None = None):
    env_cfg = parse_env_cfg(
        "ATEC-TaskE-Piper",
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    sensor_names = {}
    if with_debug_cameras:
        for name, record in with_debug_cameras.items():
            sensor_names[name] = add_debug_camera_cfg(env_cfg, name, record["intended_camera_pose"])
    env = gym.make("ATEC-TaskE-Piper", cfg=env_cfg)
    obs, _ = env.reset()
    return env, obs, sensor_names


def scene_object_pose(env, object_name: str) -> dict | None:
    object_key = OBJECTS[object_name]["object_key"]
    scene = env.unwrapped.scene
    if object_key not in scene.rigid_objects:
        return None
    obj = scene.rigid_objects[object_key]
    center = obj.data.root_pos_w[0, :3].detach().cpu().numpy().astype(float).tolist()
    payload = {
        "object_key": object_key,
        "center_world": center,
        "source": "scene.rigid_objects.root_pos_w",
    }
    root_quat_w = getattr(obj.data, "root_quat_w", None)
    if root_quat_w is not None:
        payload["quat_wxyz"] = root_quat_w[0].detach().cpu().numpy().astype(float).tolist()
    return payload


def run_real_ee_camera_captures(
    records: dict[str, dict],
    env=None,
    obs=None,
    robot=None,
    ee_camera_calibration: dict | None = None,
) -> None:
    owns_env = env is None
    if owns_env:
        env, obs, _ = create_env()
        robot = env.unwrapped.scene.articulations["robot"]
        ee_ids, _ = robot.find_bodies("gripper_base")
        if len(ee_ids) != 1:
            raise RuntimeError(f"Expected exactly one gripper_base body, got {ee_ids}")
        ee_camera_calibration = calibrate_ee_camera(env, robot, int(ee_ids[0]))
    elif obs is None or robot is None or ee_camera_calibration is None:
        raise ValueError("Reusing an env for real EE-camera captures requires obs, robot, and ee_camera_calibration.")
    for name in args_cli.objects:
        object_dir = OUTPUT_DIR / name
        obs, real_record = move_real_ee_camera_to_pose(
            env,
            obs,
            robot,
            ee_camera_calibration,
            records[name]["intended_camera_pose"],
            args_cli.real_hover_settle_steps,
        )
        ee_rgb = to_rgb_array(obs["image"]["ee_rgb"])
        save_labeled(
            object_dir / "actual_ee_camera_rgb.png",
            ee_rgb,
            f"Mounted EE camera after IK move to intended {name} pose",
        )
        ee_depth = to_depth_array(obs["image"]["ee_depth"])
        np.save(object_dir / "actual_ee_camera_depth.npy", ee_depth)
        save_depth_preview(object_dir / "actual_ee_camera_depth_preview.png", ee_depth, args_cli.max_depth)
        video_rgb_after = to_rgb_array(obs["image"]["video_rgb"])
        save_labeled(
            object_dir / "actual_ee_camera_video_overview.png",
            video_rgb_after,
            f"External view after real EE-camera move for {name}",
        )
        write_camera_json(object_dir / "actual_ee_camera.json", real_record["actual_camera"])
        (object_dir / "actual_ee_camera_move.json").write_text(
            json.dumps(real_record, indent=2),
            encoding="utf-8",
        )
        records[name]["actual_ee_camera_rgb"] = str(object_dir / "actual_ee_camera_rgb.png")
        records[name]["actual_ee_camera_video_overview"] = str(object_dir / "actual_ee_camera_video_overview.png")
        records[name]["actual_ee_camera_move"] = real_record
    if owns_env:
        env.close()


def main() -> None:
    records: dict[str, dict] = {}
    real_ee_captured_in_source_env = False

    ee_camera_calibration = None
    if args_cli.reuse_computed:
        for name in args_cli.objects:
            path = OUTPUT_DIR / name / "intended_camera_pose.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing computed intended pose: {path}")
            records[name] = json.loads(path.read_text(encoding="utf-8"))
        summary_path = OUTPUT_DIR / "summary.json"
        if summary_path.exists():
            previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            ee_camera_calibration = previous_summary.get("ee_camera_calibration")
    else:
        env, obs, _ = create_env()
        robot = env.unwrapped.scene.articulations["robot"]
        ee_ids, _ = robot.find_bodies("gripper_base")
        if len(ee_ids) != 1:
            raise RuntimeError(f"Expected exactly one gripper_base body, got {ee_ids}")
        ee_camera_calibration = calibrate_ee_camera(env, robot, int(ee_ids[0]))

        video_rgb = to_rgb_array(obs["image"]["video_rgb"])
        video_depth = to_depth_array(obs["image"]["video_depth"])
        Image.fromarray(video_rgb, mode="RGB").save(OUTPUT_DIR / "video_rgb.png")
        np.save(OUTPUT_DIR / "video_depth.npy", video_depth)
        save_depth_preview(OUTPUT_DIR / "video_depth_preview.png", video_depth, args_cli.max_depth)
        video_camera = camera_metadata(env, "video_cam")
        write_camera_json(OUTPUT_DIR / "video_camera.json", video_camera)

        for name in args_cli.objects:
            object_dir = OUTPUT_DIR / name
            object_dir.mkdir(parents=True, exist_ok=True)
            mask_path, detection = run_sam3(OUTPUT_DIR / "video_rgb.png", object_dir, name)
            mask = load_mask(mask_path, video_depth.shape)
            sam3_pose = estimate_pose_from_mask(video_depth, mask, video_camera, args_cli.max_depth)
            sam3_pose["source"] = "video_cam_sam3_depth"
            refined_pose, refined_mask_path = maybe_refine_pose_with_color(
                object_dir,
                OUTPUT_DIR / "video_rgb.png",
                video_depth,
                video_camera,
                mask,
                name,
            )
            selected_pose = refined_pose or sam3_pose
            selected_pose["selected_for_intended_camera"] = True
            selected_pose["selected_pose_source"] = selected_pose.get("source", "video_cam_sam3_depth")

            draw_target_overlay(
                OUTPUT_DIR / "video_rgb.png",
                object_dir / "video_segment_target_overlay.png",
                selected_pose,
                f"{name} video segment target",
                refined_mask_path or mask_path,
            )
            target_center = selected_pose["center_world_median"]
            target_center = [float(target_center[0]), float(target_center[1]), float(TABLE_TOP_Z + 0.05)]
            pose = intended_camera_pose(target_center, name, ee_camera_calibration)
            records[name] = {
                "object": name,
                "prompt": OBJECT_PROMPTS[name],
                "scene_object_pose": scene_object_pose(env, name),
                "sam3": {
                    "mask_path": str(mask_path),
                    "detection_path": str(object_dir / f"video_{name}_detections.json"),
                    "mask_count": detection.get("mask_count"),
                    "best_index": detection.get("best_index"),
                    "areas_px": detection.get("areas_px"),
                    "scores": detection.get("scores"),
                },
                "sam3_pose_estimate": sam3_pose,
                "color_refined_pose_estimate": refined_pose,
                "selected_video_pose_estimate": selected_pose,
                "request_object_center_w": target_center,
                "intended_camera_pose": pose,
            }
            (object_dir / "intended_camera_pose.json").write_text(json.dumps(records[name], indent=2), encoding="utf-8")

        if args_cli.real_ee_camera and not args_cli.compute_only:
            run_real_ee_camera_captures(
                records,
                env=env,
                obs=obs,
                robot=robot,
                ee_camera_calibration=ee_camera_calibration,
            )
            real_ee_captured_in_source_env = True

        env.close()

    if args_cli.compute_only:
        summary = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "task": "ATEC-TaskE-Piper",
            "seed": args_cli.seed,
            "objects": args_cli.objects,
            "output_dir": str(OUTPUT_DIR),
            "video_rgb": str(OUTPUT_DIR / "video_rgb.png"),
            "video_depth": str(OUTPUT_DIR / "video_depth.npy"),
            "video_camera_json": str(OUTPUT_DIR / "video_camera.json"),
            "ee_camera_calibration": ee_camera_calibration,
            "records": records,
            "compute_only": True,
        }
        (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote computed intended poses: {OUTPUT_DIR}", flush=True)
        simulation_app.close()
        return

    if args_cli.real_only:
        if not real_ee_captured_in_source_env:
            run_real_ee_camera_captures(records)
        summary = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "task": "ATEC-TaskE-Piper",
            "seed": args_cli.seed,
            "objects": args_cli.objects,
            "output_dir": str(OUTPUT_DIR),
            "video_rgb": str(OUTPUT_DIR / "video_rgb.png"),
            "video_depth": str(OUTPUT_DIR / "video_depth.npy"),
            "video_camera_json": str(OUTPUT_DIR / "video_camera.json"),
            "ee_camera_calibration": ee_camera_calibration,
            "real_ee_camera": True,
            "real_ee_camera_same_env_as_video": bool(real_ee_captured_in_source_env),
            "real_only": True,
            "records": records,
            "reuse_computed": bool(args_cli.reuse_computed),
        }
        (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote real EE-camera debug output: {OUTPUT_DIR}", flush=True)
        for name in args_cli.objects:
            print(f"[INFO] {name}: {OUTPUT_DIR / name / 'actual_ee_camera_rgb.png'}", flush=True)
        simulation_app.close()
        return

    debug_env, debug_obs, sensor_names = create_env(records)
    for _ in range(max(1, int(args_cli.debug_steps))):
        zero_action = torch.zeros_like(debug_env.unwrapped.scene.articulations["robot"].data.default_joint_pos)
        debug_obs, _, terminated, truncated, _ = debug_env.step(zero_action)
        if terminated.any() or truncated.any():
            break

    for name, sensor_name in sensor_names.items():
        sensor = debug_env.unwrapped.scene.sensors[sensor_name]
        object_dir = OUTPUT_DIR / name
        rgb = camera_sensor_rgb(sensor)
        save_labeled(object_dir / "virtual_camera_rgb.png", rgb, f"Virtual camera output at intended {name} pose")
        depth = camera_sensor_depth(sensor)
        if depth is not None:
            np.save(object_dir / "virtual_camera_depth.npy", depth)
            save_depth_preview(object_dir / "virtual_camera_depth_preview.png", depth, args_cli.max_depth)
        records[name]["virtual_camera_sensor"] = sensor_name
        records[name]["virtual_camera_rgb"] = str(object_dir / "virtual_camera_rgb.png")

    draw_debug_frustums(records)
    for _ in range(max(1, int(args_cli.debug_steps))):
        zero_action = torch.zeros_like(debug_env.unwrapped.scene.articulations["robot"].data.default_joint_pos)
        debug_obs, _, terminated, truncated, _ = debug_env.step(zero_action)
        if terminated.any() or truncated.any():
            break

    overview = to_rgb_array(debug_obs["image"]["video_rgb"])
    save_labeled(
        OUTPUT_DIR / "scene_debug_overview.png",
        overview,
        "External video camera view with intended EE-camera frustum markers",
    )
    debug_env.close()

    if args_cli.real_ee_camera and not real_ee_captured_in_source_env:
        run_real_ee_camera_captures(records)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": "ATEC-TaskE-Piper",
        "seed": args_cli.seed,
        "objects": args_cli.objects,
        "output_dir": str(OUTPUT_DIR),
        "video_rgb": str(OUTPUT_DIR / "video_rgb.png"),
        "video_depth": str(OUTPUT_DIR / "video_depth.npy"),
        "video_camera_json": str(OUTPUT_DIR / "video_camera.json"),
        "scene_debug_overview": str(OUTPUT_DIR / "scene_debug_overview.png"),
        "ee_camera_calibration": ee_camera_calibration,
        "real_ee_camera": bool(args_cli.real_ee_camera),
        "real_ee_camera_same_env_as_video": bool(real_ee_captured_in_source_env),
        "records": records,
        "reuse_computed": bool(args_cli.reuse_computed),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[INFO] Wrote debug output: {OUTPUT_DIR}", flush=True)
    print(f"[INFO] Overview: {OUTPUT_DIR / 'scene_debug_overview.png'}", flush=True)
    for name in args_cli.objects:
        print(f"[INFO] {name}: {OUTPUT_DIR / name / 'virtual_camera_rgb.png'}", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write_failure("debug_error.json", exc)
        raise
