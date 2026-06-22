#!/usr/bin/env python3
"""Back-project a masked RGB-D frame and run AnyGrasp when licensed."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANYGRASP_ROOT = Path(
    os.environ.get("ANYGRASP_ROOT", str(REPO_ROOT / "third_party/anygrasp_sdk"))
)
DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "ANYGRASP_CHECKPOINT",
        str(DEFAULT_ANYGRASP_ROOT / "checkpoint_detection.tar"),
    )
)
DEFAULT_LICENSE_DIR = Path(
    os.environ.get("ANYGRASP_LICENSE_DIR", str(DEFAULT_ANYGRASP_ROOT / "license"))
)
DEFAULT_OPENSSL_LIB_DIR = Path(
    os.environ.get("ANYGRASP_OPENSSL_LIB_DIR", str(REPO_ROOT / "third_party/openssl11/lib"))
)


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", required=True, type=Path, help="RGB PNG.")
    parser.add_argument("--depth-npy", required=True, type=Path, help="Depth array in metres.")
    parser.add_argument("--mask", required=True, type=Path, help="Binary mask PNG.")
    parser.add_argument(
        "--camera-json",
        required=True,
        type=Path,
        help="Camera metadata JSON with intrinsics and optional world pose.",
    )
    parser.add_argument(
        "--extra-view",
        action="append",
        default=[],
        metavar="RGB,DEPTH_NPY,MASK,CAMERA_JSON",
        help=(
            "Additional segmented RGB-D view to fuse into the primary camera frame before "
            "AnyGrasp. May be repeated. The four paths are comma-separated."
        ),
    )
    parser.add_argument(
        "--anygrasp-cloud-mode",
        choices=("target_mask", "full_scene_target_filter"),
        default="target_mask",
        help=(
            "target_mask runs AnyGrasp on the segmented target cloud. "
            "full_scene_target_filter runs AnyGrasp on the full RGB-D scene, then keeps only "
            "grasps whose contact pose lies on or near the target mask/cloud."
        ),
    )
    parser.add_argument(
        "--symmetric-cloud-mode",
        choices=("off", "mirror", "bottle_surface"),
        default="off",
        help=(
            "Complete the segmented target cloud before AnyGrasp inference. "
            "mirror reflects the observed target about the estimated center; "
            "bottle_surface additionally synthesizes an upright bottle-like surface."
        ),
    )
    parser.add_argument(
        "--symmetry-center-source",
        choices=("bbox_center", "object_center", "mean"),
        default="bbox_center",
    )
    parser.add_argument("--symmetric-surface-points", type=int, default=16000)
    parser.add_argument("--symmetric-body-rx", type=float, default=0.0)
    parser.add_argument("--symmetric-body-ry", type=float, default=0.0)
    parser.add_argument("--symmetric-neck-radius-scale", type=float, default=0.46)
    parser.add_argument("--symmetric-min-z-margin", type=float, default=0.012)
    parser.add_argument(
        "--scene-cloud-stride",
        type=int,
        default=1,
        help="Pixel stride for full-scene AnyGrasp input. 1 keeps all valid depth pixels.",
    )
    parser.add_argument(
        "--scene-cloud-max-points",
        type=int,
        default=180000,
        help="Deterministically downsample full-scene AnyGrasp input to at most this many points. <=0 disables.",
    )
    parser.add_argument(
        "--filter-target-outliers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only the largest connected voxel component of the fused target cloud.",
    )
    parser.add_argument("--target-filter-voxel-size", type=float, default=0.008)
    parser.add_argument("--target-filter-min-points", type=int, default=50)
    parser.add_argument(
        "--target-grasp-filter-distance",
        type=float,
        default=0.035,
        help="Keep full-scene grasps whose contact point is within this many metres of the target cloud.",
    )
    parser.add_argument(
        "--target-grasp-filter-pixel-radius",
        type=int,
        default=8,
        help="Also keep full-scene grasps whose projected contact point falls within this target-mask pixel radius.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument("--anygrasp-root", type=Path, default=DEFAULT_ANYGRASP_ROOT)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--license-dir",
        type=Path,
        default=DEFAULT_LICENSE_DIR,
        help=(
            "AnyGrasp license directory. If omitted, expects "
            "<anygrasp-root>/license and links it to <anygrasp-root>/grasp_detection/license."
        ),
    )
    parser.add_argument(
        "--openssl-lib-dir",
        type=Path,
        default=DEFAULT_OPENSSL_LIB_DIR,
        help="Directory containing libcrypto.so.1.1/libssl.so.1.1 for vendor license binaries.",
    )
    parser.add_argument("--max-gripper-width", type=float, default=0.1)
    parser.add_argument("--gripper-height", type=float, default=0.03)
    parser.add_argument("--top-down-grasp", dest="top_down_grasp", action="store_true")
    parser.add_argument("--no-top-down-grasp", dest="top_down_grasp", action="store_false")
    parser.add_argument("--dense-grasp", action="store_true")
    parser.add_argument("--apply-object-mask", dest="apply_object_mask", action="store_true")
    parser.add_argument("--no-apply-object-mask", dest="apply_object_mask", action="store_false")
    parser.add_argument("--no-collision-detection", action="store_true")
    parser.add_argument("--no-relaxed-retry", action="store_true")
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument(
        "--save-top-grasps",
        type=int,
        default=5,
        help="Save and visualize this many highest-scoring AnyGrasp candidates.",
    )
    parser.add_argument(
        "--overlay-tool-transform",
        choices=("identity", "graspnet_to_piper_z"),
        default="graspnet_to_piper_z",
        help="Frame mapping used only for drawing the Piper gripper in top_grasps_overlay.png.",
    )
    parser.add_argument(
        "--overlay-gripper-base-offset",
        type=float,
        default=0.09,
        help="Distance from AnyGrasp contact frame back to Piper gripper_base for overlay drawing.",
    )
    parser.add_argument(
        "--overlay-gripper-base-offset-mode",
        choices=("approach_axis", "finger_centerline", "yellow_line", "towards_object_center"),
        default="approach_axis",
        help=(
            "How to offset the overlay gripper_base from the AnyGrasp contact point. "
            "finger_centerline/yellow_line follows the drawn finger centerline. "
            "towards_object_center needs --overlay-object-center-world."
        ),
    )
    parser.add_argument(
        "--overlay-object-center-world",
        type=parse_float_list,
        default=[],
        help="Optional world XYZ object center used by towards_object_center overlay mode.",
    )
    parser.add_argument(
        "--overlay-finger-length",
        type=float,
        default=0.075,
        help="Approximate Piper finger length in metres for overlay drawing.",
    )
    parser.add_argument(
        "--overlay-palm-width",
        type=float,
        default=0.085,
        help="Approximate Piper palm width in metres for overlay drawing.",
    )
    parser.add_argument(
        "--overlay-show-full-tool",
        action="store_true",
        help="Also draw the projected gripper_base/palm/fingers, which may leave the frame in close EE-camera views.",
    )
    parser.set_defaults(top_down_grasp=True, apply_object_mask=True)
    args = parser.parse_args()
    args.anygrasp_root = args.anygrasp_root.expanduser().resolve()
    args.checkpoint_path = args.checkpoint_path.expanduser().resolve()
    if args.license_dir is not None:
        args.license_dir = args.license_dir.expanduser().resolve()
    if args.openssl_lib_dir is not None:
        args.openssl_lib_dir = args.openssl_lib_dir.expanduser().resolve()
    return args


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def to_depth_array(path: Path) -> np.ndarray:
    depth = np.load(path).astype(np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected depth shape (H,W) or (H,W,1), got {depth.shape}")
    return depth


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L")
    width, height = shape[1], shape[0]
    if mask.size != (width, height):
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(mask) > 127


def backproject(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    z = depth.astype(np.float32)
    x = (xx - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (yy - intrinsic[1, 2]) * z / intrinsic[1, 1]
    return np.stack([x, y, z], axis=-1)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = quat / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_pose_to_world(camera: dict, translation: np.ndarray, rotation: np.ndarray) -> dict | None:
    pos = camera.get("pos_w")
    quat = camera.get("quat_w_ros")
    if pos is None or quat is None:
        return None
    r_wc = quat_wxyz_to_matrix(np.asarray(quat, dtype=np.float64))
    t_wc = np.asarray(pos, dtype=np.float64)
    t_w = r_wc @ np.asarray(translation, dtype=np.float64) + t_wc
    r_w = r_wc @ np.asarray(rotation, dtype=np.float64)
    return {"translation": t_w.tolist(), "rotation_matrix": r_w.tolist()}


def transform_points_to_world(points_cam: np.ndarray, camera: dict) -> np.ndarray:
    r_wc = quat_wxyz_to_matrix(np.asarray(camera["quat_w_ros"], dtype=np.float64))
    t_wc = np.asarray(camera["pos_w"], dtype=np.float64)
    return np.asarray(points_cam, dtype=np.float64) @ r_wc.T + t_wc


def transform_points_from_world(points_w: np.ndarray, camera: dict) -> np.ndarray:
    r_wc = quat_wxyz_to_matrix(np.asarray(camera["quat_w_ros"], dtype=np.float64))
    t_wc = np.asarray(camera["pos_w"], dtype=np.float64)
    return (np.asarray(points_w, dtype=np.float64) - t_wc) @ r_wc


def symmetry_center_world(
    points_world: np.ndarray,
    bbox_center: np.ndarray,
    object_center_world: np.ndarray | None,
    args: argparse.Namespace,
) -> np.ndarray:
    if args.symmetry_center_source == "object_center" and object_center_world is not None:
        return np.asarray(object_center_world, dtype=np.float64).copy()
    if args.symmetry_center_source == "mean" and len(points_world):
        center = np.mean(points_world, axis=0)
        center[2] = bbox_center[2]
        return center.astype(np.float64)
    return np.asarray(bbox_center, dtype=np.float64).copy()


def bottle_radius_profile(z_norm: np.ndarray, rx: float, ry: float, neck_scale: float) -> tuple[np.ndarray, np.ndarray]:
    z_norm = np.clip(z_norm, 0.0, 1.0)
    shoulder = np.clip((z_norm - 0.72) / 0.18, 0.0, 1.0)
    neck = float(np.clip(neck_scale, 0.25, 0.85))
    scale = (1.0 - shoulder) + shoulder * neck
    cap = z_norm >= 0.90
    scale = np.where(cap, neck * 0.92, scale)
    return rx * scale, ry * scale


def complete_symmetric_cloud_camera(
    target_points: np.ndarray,
    target_colors: np.ndarray,
    camera: dict,
    object_center_world: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict]:
    mode = str(args.symmetric_cloud_mode)
    if mode == "off" or len(target_points) == 0:
        return target_points, target_colors, {"enabled": False, "mode": mode}

    observed_world = transform_points_to_world(target_points.astype(np.float64), camera).astype(np.float64)
    colors = target_colors.astype(np.float64)
    bbox_min = np.min(observed_world, axis=0)
    bbox_max = np.max(observed_world, axis=0)
    bbox_center = 0.5 * (bbox_min + bbox_max)
    center = symmetry_center_world(observed_world, bbox_center, object_center_world, args)
    bbox_size = np.maximum(bbox_max - bbox_min, 1e-4)
    rx = float(args.symmetric_body_rx) if float(args.symmetric_body_rx) > 0.0 else float(np.clip(bbox_size[0] * 0.55, 0.028, 0.060))
    ry = float(args.symmetric_body_ry) if float(args.symmetric_body_ry) > 0.0 else float(np.clip(bbox_size[1] * 0.58, 0.020, 0.043))
    z_min = float(bbox_min[2] + max(0.0, float(args.symmetric_min_z_margin)))
    z_max = float(bbox_max[2] - max(0.0, float(args.symmetric_min_z_margin) * 0.5))
    if z_max <= z_min + 0.04:
        z_min = float(bbox_min[2])
        z_max = float(bbox_max[2])

    point_batches = [observed_world]
    color_batches = [colors]
    if mode in {"mirror", "bottle_surface"}:
        for flip_x, flip_y in ((True, False), (False, True), (True, True)):
            reflected = observed_world.copy()
            if flip_x:
                reflected[:, 0] = 2.0 * center[0] - reflected[:, 0]
            if flip_y:
                reflected[:, 1] = 2.0 * center[1] - reflected[:, 1]
            point_batches.append(reflected)
            color_batches.append(colors)

    synthetic_count = max(0, int(args.symmetric_surface_points)) if mode == "bottle_surface" else 0
    if synthetic_count:
        z_count = max(24, int(np.sqrt(synthetic_count) * 0.85))
        theta_count = max(48, int(np.ceil(synthetic_count / z_count)))
        z_values = np.linspace(z_min, z_max, z_count)
        theta_values = np.linspace(-np.pi, np.pi, theta_count, endpoint=False)
        zz, tt = np.meshgrid(z_values, theta_values, indexing="ij")
        z_norm = (zz - z_min) / max(z_max - z_min, 1e-6)
        rxs, rys = bottle_radius_profile(z_norm, rx, ry, float(args.symmetric_neck_radius_scale))
        c = np.cos(tt)
        s = np.sin(tt)
        power = 0.55
        surface = np.stack(
            [
                center[0] + rxs * np.sign(c) * (np.abs(c) ** power),
                center[1] + rys * np.sign(s) * (np.abs(s) ** power),
                zz,
            ],
            axis=-1,
        ).reshape(-1, 3)
        top_rx, top_ry = bottle_radius_profile(
            np.ones_like(theta_values),
            rx,
            ry,
            float(args.symmetric_neck_radius_scale),
        )
        cap_points = []
        for scale in np.linspace(0.0, 1.0, max(4, theta_count // 12)):
            cap_points.append(
                np.stack(
                    [
                        center[0] + top_rx * scale * np.cos(theta_values),
                        center[1] + top_ry * scale * np.sin(theta_values),
                        np.full_like(theta_values, z_max),
                    ],
                    axis=1,
                )
            )
        synthetic = np.concatenate([surface, *cap_points], axis=0)
        synthetic_colors = np.tile(np.array([[0.95, 0.78, 0.12]], dtype=np.float64), (len(synthetic), 1))
        point_batches.append(synthetic)
        color_batches.append(synthetic_colors)

    completed_world = np.concatenate(point_batches, axis=0).astype(np.float64)
    completed_colors = np.concatenate(color_batches, axis=0).astype(np.float64)
    completed_camera = transform_points_from_world(completed_world, camera).astype(np.float32)
    keep = np.isfinite(completed_camera).all(axis=1) & (completed_camera[:, 2] > 0.0)
    completed_camera = completed_camera[keep]
    completed_colors = completed_colors[keep].astype(np.float32)
    return completed_camera, completed_colors, {
        "enabled": True,
        "mode": mode,
        "symmetry_center_source": str(args.symmetry_center_source),
        "center_world": center.astype(float).tolist(),
        "body_rx_m": rx,
        "body_ry_m": ry,
        "z_min_m": z_min,
        "z_max_m": z_max,
        "neck_radius_scale": float(args.symmetric_neck_radius_scale),
        "observed_point_count": int(len(target_points)),
        "completed_point_count": int(len(completed_camera)),
        "synthetic_surface_point_budget": int(args.symmetric_surface_points),
    }


def parse_extra_view(value: str) -> tuple[Path, Path, Path, Path]:
    parts = [Path(part.strip()).expanduser() for part in value.split(",")]
    if len(parts) != 4 or any(str(part) == "." for part in parts):
        raise ValueError(
            "--extra-view expects RGB,DEPTH_NPY,MASK,CAMERA_JSON with no empty fields"
        )
    return parts[0], parts[1], parts[2], parts[3]


def masked_points_for_view(
    rgb_path: Path,
    depth_path: Path,
    mask_path: Path,
    camera: dict,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float32)
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
    depth = to_depth_array(depth_path)
    mask = load_mask(mask_path, depth.shape)
    points_organized = backproject(depth, intrinsic)
    valid = mask & np.isfinite(depth) & (depth > 0.0) & (depth < max_depth)
    points = points_organized[valid].astype(np.float32)
    colors = rgb[valid].astype(np.float32)
    ys, xs = np.where(valid)
    stats = {
        "rgb": str(rgb_path.resolve()),
        "depth_npy": str(depth_path.resolve()),
        "mask": str(mask_path.resolve()),
        "camera_json": None,
        "point_count": int(points.shape[0]),
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if xs.size else None,
    }
    return points, colors, stats


def scene_points_for_view(
    rgb_path: Path,
    depth_path: Path,
    camera: dict,
    max_depth: float,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict]:
    intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float32)
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
    depth = to_depth_array(depth_path)
    points_organized = backproject(depth, intrinsic)
    valid = np.isfinite(depth) & (depth > 0.0) & (depth < max_depth)
    stride = max(1, int(stride))
    if stride > 1:
        stride_mask = np.zeros_like(valid, dtype=bool)
        stride_mask[::stride, ::stride] = True
        valid &= stride_mask
    points = points_organized[valid].astype(np.float32)
    colors = rgb[valid].astype(np.float32)
    ys, xs = np.where(valid)
    stats = {
        "rgb": str(rgb_path.resolve()),
        "depth_npy": str(depth_path.resolve()),
        "camera_json": None,
        "point_count": int(points.shape[0]),
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if xs.size else None,
        "stride": int(stride),
    }
    return points, colors, stats


def deterministic_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    max_points = int(max_points)
    if max_points <= 0 or len(points) <= max_points:
        return points, colors, {
            "enabled": max_points > 0,
            "input_point_count": int(len(points)),
            "output_point_count": int(len(points)),
            "max_points": max_points,
            "applied": False,
        }
    indices = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
    return points[indices], colors[indices], {
        "enabled": True,
        "input_point_count": int(len(points)),
        "output_point_count": int(max_points),
        "max_points": max_points,
        "applied": True,
        "method": "linspace_deterministic",
    }


def largest_voxel_component_filter(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
    min_points: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    stats = {
        "enabled": True,
        "method": "largest_connected_voxel_component",
        "voxel_size_m": float(voxel_size),
        "min_points": int(min_points),
        "input_point_count": int(len(points)),
        "output_point_count": int(len(points)),
        "applied": False,
    }
    if len(points) < int(min_points) or float(voxel_size) <= 0.0:
        stats["reason"] = "too_few_points_or_invalid_voxel_size"
        return points, colors, stats

    finite = np.isfinite(points).all(axis=1)
    if not finite.all():
        points = points[finite]
        colors = colors[finite]
        stats["finite_point_count"] = int(len(points))
    if len(points) < int(min_points):
        stats["reason"] = "too_few_finite_points"
        stats["output_point_count"] = int(len(points))
        return points, colors, stats

    origin = points.min(axis=0)
    voxels = np.floor((points - origin) / float(voxel_size)).astype(np.int32)
    unique_voxels, inverse, counts = np.unique(voxels, axis=0, return_inverse=True, return_counts=True)
    occupied = {tuple(v.tolist()): i for i, v in enumerate(unique_voxels)}
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if dx or dy or dz
    ]
    visited = np.zeros(len(unique_voxels), dtype=bool)
    best_component: list[int] = []
    best_points = -1
    component_count = 0
    for start in range(len(unique_voxels)):
        if visited[start]:
            continue
        component_count += 1
        stack = [start]
        visited[start] = True
        component = []
        point_count = 0
        while stack:
            idx = stack.pop()
            component.append(idx)
            point_count += int(counts[idx])
            vx, vy, vz = unique_voxels[idx]
            for dx, dy, dz in offsets:
                nidx = occupied.get((int(vx + dx), int(vy + dy), int(vz + dz)))
                if nidx is not None and not visited[nidx]:
                    visited[nidx] = True
                    stack.append(nidx)
        if point_count > best_points:
            best_component = component
            best_points = point_count

    if not best_component:
        stats["reason"] = "no_components"
        return points, colors, stats

    keep_voxel = np.zeros(len(unique_voxels), dtype=bool)
    keep_voxel[np.asarray(best_component, dtype=np.int64)] = True
    keep = keep_voxel[inverse]
    filtered_points = points[keep]
    filtered_colors = colors[keep]
    stats.update(
        {
            "applied": bool(len(filtered_points) < len(points)),
            "component_count": int(component_count),
            "occupied_voxel_count": int(len(unique_voxels)),
            "kept_voxel_count": int(len(best_component)),
            "output_point_count": int(len(filtered_points)),
            "removed_point_count": int(len(points) - len(filtered_points)),
        }
    )
    return filtered_points.astype(np.float32), filtered_colors.astype(np.float32), stats


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, rgb, strict=False):
            f.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def draw_projection(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    points: np.ndarray,
    colors: np.ndarray,
    axes: tuple[int, int],
    title: str,
    marker: np.ndarray | None = None,
) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline=(80, 80, 80), width=1)
    draw.text((left + 8, top + 6), title, fill=(255, 255, 255))
    if len(points) == 0:
        draw.text((left + 8, top + 28), "no points", fill=(255, 160, 160))
        return

    pts = points[:, list(axes)]
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    span = np.maximum(maxs - mins, 1e-4)
    pad = span * 0.08
    mins -= pad
    maxs += pad
    span = maxs - mins

    max_points = 9000
    if len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        pts = pts[idx]
        draw_colors = colors[idx]
    else:
        draw_colors = colors

    px = left + 8 + (pts[:, 0] - mins[0]) / span[0] * (right - left - 16)
    py = bottom - 8 - (pts[:, 1] - mins[1]) / span[1] * (bottom - top - 32)
    rgb = np.clip(draw_colors * 255.0, 0, 255).astype(np.uint8)
    for x, y, color in zip(px.astype(int), py.astype(int), rgb, strict=False):
        draw.point((int(x), int(y)), fill=tuple(int(c) for c in color))

    if marker is not None:
        m = marker[list(axes)]
        mx = left + 8 + (m[0] - mins[0]) / span[0] * (right - left - 16)
        my = bottom - 8 - (m[1] - mins[1]) / span[1] * (bottom - top - 32)
        r = 5
        draw.ellipse((mx - r, my - r, mx + r, my + r), outline=(255, 40, 40), width=2)


def save_cloud_views(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    status: str,
    marker: np.ndarray | None = None,
) -> None:
    canvas = Image.new("RGB", (1260, 520), (18, 22, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), f"Masked RGB-D point cloud | AnyGrasp: {status}", fill=(255, 255, 255))
    boxes = [(20, 44, 410, 500), (435, 44, 825, 500), (850, 44, 1240, 500)]
    draw_projection(draw, boxes[0], points, colors, (0, 1), "camera XY", marker)
    draw_projection(draw, boxes[1], points, colors, (0, 2), "camera XZ", marker)
    draw_projection(draw, boxes[2], points, colors, (1, 2), "camera YZ", marker)
    canvas.save(path)


@contextlib.contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def ensure_license_dir(args: argparse.Namespace) -> tuple[Path | None, list[str]]:
    detection_license = args.anygrasp_root / "grasp_detection/license"
    requested_license = args.license_dir
    reasons: list[str] = []

    if detection_license.exists():
        return detection_license, reasons

    if requested_license is None:
        return None, [f"missing license dir: {detection_license}"]

    if not requested_license.exists():
        return None, [f"missing requested license dir: {requested_license}"]

    try:
        detection_license.symlink_to(requested_license, target_is_directory=True)
        return detection_license, reasons
    except FileExistsError:
        if detection_license.exists():
            return detection_license, reasons
        reasons.append(f"license path exists but is invalid: {detection_license}")
    except OSError as exc:
        reasons.append(
            "could not link requested license dir into "
            f"{detection_license}: {exc!r}"
        )
    return requested_license, reasons


def anygrasp_missing_reasons(args: argparse.Namespace) -> list[str]:
    reasons = []
    detection_dir = args.anygrasp_root / "grasp_detection"
    if not detection_dir.exists():
        reasons.append(f"missing detection dir: {detection_dir}")
    _, license_reasons = ensure_license_dir(args)
    reasons.extend(license_reasons)
    if not args.checkpoint_path.exists():
        reasons.append(f"missing checkpoint: {args.checkpoint_path}")
    return reasons


def preload_openssl11(args: argparse.Namespace) -> list[str]:
    """Preload OpenSSL 1.1 libraries required by the closed-source SDK binaries."""
    messages = []
    lib_dir = args.openssl_lib_dir
    if lib_dir is None:
        return messages
    for name in ("libcrypto.so.1.1", "libssl.so.1.1"):
        path = lib_dir / name
        if not path.exists():
            messages.append(f"missing optional OpenSSL 1.1 library: {path}")
            continue
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            messages.append(f"preloaded {path}")
        except OSError as exc:
            messages.append(f"failed to preload {path}: {exc!r}")
    return messages


def grasp_attempts(args: argparse.Namespace) -> list[dict]:
    configured_object_mask = bool(args.apply_object_mask)
    if args.anygrasp_cloud_mode == "full_scene_target_filter":
        configured_object_mask = False
    attempts = [
        {
            "name": "configured",
            "apply_object_mask": configured_object_mask,
            "dense_grasp": bool(args.dense_grasp),
            "collision_detection": not bool(args.no_collision_detection),
        }
    ]
    if not args.no_relaxed_retry:
        attempts.extend(
            [
                {
                    "name": "relaxed_no_object_mask",
                    "apply_object_mask": False,
                    "dense_grasp": bool(args.dense_grasp),
                    "collision_detection": not bool(args.no_collision_detection),
                },
                {
                    "name": "relaxed_no_collision",
                    "apply_object_mask": False,
                    "dense_grasp": bool(args.dense_grasp),
                    "collision_detection": False,
                },
                {
                    "name": "dense_no_collision",
                    "apply_object_mask": False,
                    "dense_grasp": True,
                    "collision_detection": False,
                },
            ]
        )

    unique = []
    seen = set()
    for attempt in attempts:
        key = (
            attempt["apply_object_mask"],
            attempt["dense_grasp"],
            attempt["collision_detection"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(attempt)
    return unique


def serialize_grasp(grasp, camera: dict, rank: int | None = None) -> dict:
    translation = np.asarray(grasp.translation, dtype=np.float64)
    rotation = np.asarray(grasp.rotation_matrix, dtype=np.float64)
    world_pose = transform_pose_to_world(camera, translation, rotation)
    payload = {
        "score": float(grasp.score),
        "width": float(grasp.width),
        "depth": float(grasp.depth),
        "translation_camera": translation.tolist(),
        "rotation_matrix_camera": rotation.tolist(),
        "pose_world": world_pose,
        "final_ee_pose_world_candidate": world_pose,
        "note": (
            "Candidate is in the AnyGrasp gripper frame. Validate the fixed "
            "transform to Piper gripper_base before executing on the robot. "
            "For GraspNet geometry, the local +X axis points from the gripper "
            "tail toward the fingers and is normally used as the approach axis."
        ),
    }
    if rank is not None:
        payload["rank"] = int(rank)
    if world_pose is not None:
        payload["translation_world"] = world_pose["translation"]
        payload["rotation_matrix_world"] = world_pose["rotation_matrix"]
    return payload


def serialize_best_grasp(best, camera: dict) -> dict:
    return serialize_grasp(best, camera)


def mask_hit_near(mask: np.ndarray, uv: tuple[float, float] | None, radius: int) -> bool:
    if uv is None:
        return False
    u, v = uv
    if not np.isfinite(u) or not np.isfinite(v):
        return False
    cx = int(round(u))
    cy = int(round(v))
    radius = max(0, int(radius))
    x0 = max(0, cx - radius)
    x1 = min(mask.shape[1], cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(mask.shape[0], cy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return False
    return bool(np.any(mask[y0:y1, x0:x1]))


def nearest_distance(points: np.ndarray, query: np.ndarray) -> float | None:
    if len(points) == 0:
        return None
    diff = np.asarray(points, dtype=np.float64) - np.asarray(query, dtype=np.float64)[None, :]
    return float(np.sqrt(np.min(np.sum(diff * diff, axis=1))))


def annotate_target_filter(
    payload: dict,
    intrinsic: np.ndarray,
    target_mask: np.ndarray,
    target_points: np.ndarray,
    distance_threshold: float,
    pixel_radius: int,
) -> dict:
    translation = np.asarray(payload.get("translation_camera"), dtype=np.float64)
    uv = project_camera_point(intrinsic, translation)
    hit_mask = mask_hit_near(target_mask, uv, pixel_radius)
    distance = nearest_distance(target_points, translation)
    hit_distance = distance is not None and distance <= float(distance_threshold)
    accepted = bool(hit_mask or hit_distance)
    payload["target_filter"] = {
        "accepted": accepted,
        "projected_uv": [float(uv[0]), float(uv[1])] if uv is not None else None,
        "mask_hit_with_radius": bool(hit_mask),
        "pixel_radius": int(pixel_radius),
        "nearest_target_point_distance_m": distance,
        "distance_threshold_m": float(distance_threshold),
        "distance_hit": bool(hit_distance),
        "mode": "projected_mask_or_nearest_target_cloud",
    }
    return payload


def filter_grasp_payloads_to_target(
    payloads: list[dict],
    intrinsic: np.ndarray,
    target_mask: np.ndarray,
    target_points: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    annotated = [
        annotate_target_filter(
            payload,
            intrinsic,
            target_mask,
            target_points,
            args.target_grasp_filter_distance,
            args.target_grasp_filter_pixel_radius,
        )
        for payload in payloads
    ]
    kept = [payload for payload in annotated if payload["target_filter"]["accepted"]]
    for filtered_rank, payload in enumerate(kept, start=1):
        payload["target_filtered_rank"] = int(filtered_rank)
    distances = [
        payload["target_filter"]["nearest_target_point_distance_m"]
        for payload in annotated
        if payload["target_filter"]["nearest_target_point_distance_m"] is not None
    ]
    return kept, {
        "enabled": True,
        "mode": "projected_mask_or_nearest_target_cloud",
        "input_grasp_count": int(len(payloads)),
        "kept_grasp_count": int(len(kept)),
        "distance_threshold_m": float(args.target_grasp_filter_distance),
        "pixel_radius": int(args.target_grasp_filter_pixel_radius),
        "nearest_distance_min_m": float(min(distances)) if distances else None,
        "nearest_distance_median_m": float(np.median(distances)) if distances else None,
    }


def project_camera_point(intrinsic: np.ndarray, point: np.ndarray) -> tuple[float, float] | None:
    point = np.asarray(point, dtype=np.float64)
    if point.shape != (3,) or point[2] <= 1e-6:
        return None
    u = intrinsic[0, 0] * point[0] / point[2] + intrinsic[0, 2]
    v = intrinsic[1, 1] * point[1] / point[2] + intrinsic[1, 2]
    return float(u), float(v)


def draw_projected_axis(
    draw: ImageDraw.ImageDraw,
    intrinsic: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    color: tuple[int, int, int],
    length: float,
) -> None:
    start = project_camera_point(intrinsic, origin)
    end = project_camera_point(intrinsic, origin + axis * length)
    if start is None or end is None:
        return
    draw.line((*start, *end), fill=color, width=3)
    r = 3
    draw.ellipse((end[0] - r, end[1] - r, end[0] + r, end[1] + r), fill=color)


def normalize_vector(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        return fallback.astype(np.float64)
    return value.astype(np.float64) / norm


def piper_overlay_pose(
    translation: np.ndarray,
    rotation: np.ndarray,
    tool_transform: str,
    gripper_base_offset: float,
    offset_mode: str = "approach_axis",
    object_center_camera: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if tool_transform == "graspnet_to_piper_z":
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
            rotation[:, 0],
            np.array([0.0, 0.0, -1.0], dtype=np.float64),
        )
    if offset_mode == "towards_object_center" and object_center_camera is not None:
        offset_axis = normalize_vector(
            np.asarray(object_center_camera, dtype=np.float64) - translation,
            -approach_axis,
        )
    elif offset_mode in {"finger_centerline", "yellow_line"}:
        offset_axis = approach_axis
    else:
        offset_axis = -approach_axis
    gripper_base = translation + offset_axis * float(gripper_base_offset)
    return gripper_base, piper_rotation, approach_axis


def draw_projected_segment(
    draw: ImageDraw.ImageDraw,
    intrinsic: np.ndarray,
    start_point: np.ndarray,
    end_point: np.ndarray,
    color: tuple[int, int, int, int],
    width: int,
    outline: bool = True,
) -> None:
    start = project_camera_point(intrinsic, start_point)
    end = project_camera_point(intrinsic, end_point)
    if start is None or end is None:
        return
    if outline:
        draw.line((*start, *end), fill=(0, 0, 0, min(220, color[3])), width=width + 3)
    draw.line((*start, *end), fill=color, width=width)


def draw_projected_dot(
    draw: ImageDraw.ImageDraw,
    intrinsic: np.ndarray,
    point: np.ndarray,
    color: tuple[int, int, int, int],
    radius: int,
) -> None:
    uv = project_camera_point(intrinsic, point)
    if uv is None:
        return
    draw.ellipse(
        (uv[0] - radius, uv[1] - radius, uv[0] + radius, uv[1] + radius),
        fill=color,
        outline=(0, 0, 0, min(220, color[3])),
        width=2,
    )


def draw_piper_gripper_overlay(
    draw: ImageDraw.ImageDraw,
    intrinsic: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
    grasp: dict,
    color: tuple[int, int, int],
    args: argparse.Namespace,
    rank: int,
    object_center_camera: np.ndarray | None = None,
) -> None:
    offset = max(0.0, float(args.overlay_gripper_base_offset))
    gripper_base, piper_rotation, approach_axis = piper_overlay_pose(
        translation,
        rotation,
        args.overlay_tool_transform,
        offset,
        args.overlay_gripper_base_offset_mode,
        object_center_camera,
    )
    jaw_axis = normalize_vector(
        piper_rotation[:, 1],
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
    )
    side_axis = normalize_vector(
        piper_rotation[:, 0],
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )
    try:
        jaw_width = float(grasp.get("width", 0.065))
    except (TypeError, ValueError):
        jaw_width = 0.065
    jaw_width = float(np.clip(jaw_width, 0.030, 0.095))
    finger_length = min(max(float(args.overlay_finger_length), 0.020), max(offset, 0.020))
    alpha = 255 if rank == 1 else 210
    line_color = (color[0], color[1], color[2], alpha)
    width = 5 if rank == 1 else 3

    # Draw the jaw footprint at the grasp/contact plane. This stays readable in
    # the top-down EE camera even when the physical gripper_base projects out of frame.
    pad_length = min(max(finger_length * 0.75, 0.030), 0.065)
    left_center = translation - jaw_axis * (jaw_width * 0.5)
    right_center = translation + jaw_axis * (jaw_width * 0.5)
    left_pad_a = left_center - side_axis * (pad_length * 0.5)
    left_pad_b = left_center + side_axis * (pad_length * 0.5)
    right_pad_a = right_center - side_axis * (pad_length * 0.5)
    right_pad_b = right_center + side_axis * (pad_length * 0.5)
    draw_projected_segment(draw, intrinsic, left_pad_a, left_pad_b, line_color, width)
    draw_projected_segment(draw, intrinsic, right_pad_a, right_pad_b, line_color, width)
    draw_projected_segment(
        draw,
        intrinsic,
        left_center,
        right_center,
        (255, 255, 255, 145),
        1,
        outline=False,
    )
    draw_projected_dot(draw, intrinsic, translation, (255, 220, 40, 230), 4)

    if args.overlay_show_full_tool:
        palm_width = max(float(args.overlay_palm_width), jaw_width + 0.018)
        palm_left = gripper_base - jaw_axis * (palm_width * 0.5)
        palm_right = gripper_base + jaw_axis * (palm_width * 0.5)
        left_root = gripper_base - jaw_axis * (jaw_width * 0.5)
        right_root = gripper_base + jaw_axis * (jaw_width * 0.5)
        left_tip = left_root + approach_axis * finger_length
        right_tip = right_root + approach_axis * finger_length
        pad_half = 0.007

        full_tool_color = (color[0], color[1], color[2], 135 if rank == 1 else 95)
        full_tool_width = 3 if rank == 1 else 2
        draw_projected_segment(draw, intrinsic, palm_left, palm_right, full_tool_color, full_tool_width)
        draw_projected_segment(draw, intrinsic, left_root, left_tip, full_tool_color, full_tool_width)
        draw_projected_segment(draw, intrinsic, right_root, right_tip, full_tool_color, full_tool_width)
        draw_projected_segment(
            draw,
            intrinsic,
            left_tip - side_axis * pad_half,
            left_tip + side_axis * pad_half,
            full_tool_color,
            full_tool_width,
        )
        draw_projected_segment(
            draw,
            intrinsic,
            right_tip - side_axis * pad_half,
            right_tip + side_axis * pad_half,
            full_tool_color,
            full_tool_width,
        )
        draw_projected_segment(
            draw,
            intrinsic,
            gripper_base,
            translation,
            (255, 255, 255, 170),
            2 if rank == 1 else 1,
            outline=False,
        )
        draw_projected_dot(draw, intrinsic, gripper_base, (255, 255, 255, 235), 4 if rank == 1 else 3)


def save_grasp_overlay(
    path: Path,
    rgb: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    camera: dict,
    grasps: list[dict],
    status: str,
    args: argparse.Namespace,
) -> None:
    base = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB").convert("RGBA")
    if mask.shape[:2] == rgb.shape[:2]:
        layer = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        layer[mask] = (255, 204, 0, 78)
        base = Image.alpha_composite(base, Image.fromarray(layer, mode="RGBA"))

    draw = ImageDraw.Draw(base)
    draw.rectangle((0, 0, base.width, 36), fill=(0, 0, 0, 180))
    draw.text(
        (8, 10),
        f"top AnyGrasp candidates + Piper EE overlay | status={status}",
        fill=(255, 255, 255, 255),
    )

    palette = [
        (255, 64, 64),
        (64, 200, 255),
        (80, 220, 120),
        (255, 180, 64),
        (210, 120, 255),
        (255, 255, 80),
    ]
    object_center_camera = None
    if len(args.overlay_object_center_world) >= 3:
        center_w = np.asarray(args.overlay_object_center_world[:3], dtype=np.float64)[None, :]
        object_center_camera = transform_points_from_world(center_w, camera)[0]
    for idx, grasp in enumerate(grasps):
        translation = np.asarray(grasp.get("translation_camera"), dtype=np.float64)
        rotation = np.asarray(grasp.get("rotation_matrix_camera"), dtype=np.float64)
        if translation.shape != (3,) or rotation.shape != (3, 3):
            continue
        uv = project_camera_point(intrinsic, translation)
        if uv is None:
            continue
        color = palette[idx % len(palette)]
        r = 7 if idx == 0 else 5
        draw.ellipse((uv[0] - r, uv[1] - r, uv[0] + r, uv[1] + r), outline=color, width=3)
        draw.text(
            (uv[0] + 9, uv[1] - 9),
            f"{idx + 1}:{float(grasp.get('score', 0.0)):.3f}",
            fill=color,
        )
        if idx == 0:
            axis_len = 0.040
            draw_projected_axis(draw, intrinsic, translation, rotation[:, 0], (255, 70, 70), axis_len)
            draw_projected_axis(draw, intrinsic, translation, rotation[:, 1], (70, 230, 90), axis_len)
            draw_projected_axis(draw, intrinsic, translation, rotation[:, 2], (70, 140, 255), axis_len)
        draw_piper_gripper_overlay(
            draw,
            intrinsic,
            translation,
            rotation,
            grasp,
            color,
            args,
            idx + 1,
            object_center_camera,
        )

    base.convert("RGB").save(path)


def run_anygrasp(
    args: argparse.Namespace,
    points: np.ndarray,
    colors: np.ndarray,
    camera: dict,
    intrinsic: np.ndarray,
    target_mask: np.ndarray,
    target_points: np.ndarray,
) -> dict:
    missing = anygrasp_missing_reasons(args)
    if missing:
        return {
            "status": "missing_requirements",
            "reasons": missing,
            "expected_license_dir": str(args.anygrasp_root / "grasp_detection/license"),
            "checkpoint_path": str(args.checkpoint_path),
        }

    openssl_messages = preload_openssl11(args)
    detection_dir = args.anygrasp_root / "grasp_detection"
    if str(detection_dir) not in sys.path:
        sys.path.insert(0, str(detection_dir))

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    pad = np.array([0.04, 0.04, 0.08], dtype=np.float32)
    lims = [
        float(mins[0] - pad[0]),
        float(maxs[0] + pad[0]),
        float(mins[1] - pad[1]),
        float(maxs[1] + pad[1]),
        max(0.0, float(mins[2] - pad[2])),
        float(maxs[2] + pad[2]),
    ]

    # The vendor extension follows its demo layout and may resolve license files
    # relative to grasp_detection. Keep import/load/inference in that directory.
    try:
        with working_directory(detection_dir):
            from gsnet import AnyGrasp  # type: ignore

            cfg = argparse.Namespace(
                checkpoint_path=str(args.checkpoint_path),
                max_gripper_width=max(0.0, min(0.1, float(args.max_gripper_width))),
                gripper_height=float(args.gripper_height),
                top_down_grasp=bool(args.top_down_grasp),
                debug=False,
            )
            anygrasp = AnyGrasp(cfg)
            anygrasp.load_net()

            attempt_records = []
            for attempt in grasp_attempts(args):
                try:
                    grasps, _ = anygrasp.get_grasp(
                        points.astype(np.float32),
                        colors.astype(np.float32),
                        lims=lims,
                        apply_object_mask=attempt["apply_object_mask"],
                        dense_grasp=attempt["dense_grasp"],
                        collision_detection=attempt["collision_detection"],
                    )
                    raw_count = int(len(grasps))
                    if raw_count == 0:
                        attempt_records.append({**attempt, "status": "no_grasp", "raw_count": 0})
                        continue
                    grasps = grasps.nms().sort_by_score()
                    nms_count = int(len(grasps))
                    if nms_count == 0:
                        attempt_records.append(
                            {**attempt, "status": "no_grasp_after_nms", "raw_count": raw_count, "nms_count": 0}
                        )
                        continue
                    all_payloads = [
                        serialize_grasp(grasps[idx], camera, rank=idx + 1)
                        for idx in range(nms_count)
                    ]
                    if args.anygrasp_cloud_mode == "full_scene_target_filter":
                        filtered_payloads, target_filter = filter_grasp_payloads_to_target(
                            all_payloads,
                            intrinsic,
                            target_mask,
                            target_points,
                            args,
                        )
                    else:
                        filtered_payloads = all_payloads
                        target_filter = {
                            "enabled": False,
                            "mode": "target_mask_input_cloud",
                            "input_grasp_count": nms_count,
                            "kept_grasp_count": nms_count,
                        }
                    if not filtered_payloads:
                        attempt_records.append(
                            {
                                **attempt,
                                "status": "no_target_grasp_after_filter",
                                "raw_count": raw_count,
                                "nms_count": nms_count,
                                "target_filter": target_filter,
                            }
                        )
                        continue
                    best_payload = filtered_payloads[0]
                    top_count = max(1, min(int(args.save_top_grasps), len(filtered_payloads)))
                    top_grasps = filtered_payloads[:top_count]
                    attempt_records.append(
                        {
                            **attempt,
                            "status": "ok",
                            "raw_count": raw_count,
                            "nms_count": nms_count,
                            "target_filtered_count": len(filtered_payloads),
                            "target_filter": target_filter,
                            "best_score": best_payload["score"],
                        }
                    )
                    return {
                        "status": "ok",
                        "workspace_lims": lims,
                        "checkpoint_path": str(args.checkpoint_path),
                        "license_dir": str(args.anygrasp_root / "grasp_detection/license"),
                        "openssl_preload": openssl_messages,
                        "successful_attempt": attempt,
                        "attempts": attempt_records,
                        "num_grasps_after_nms": nms_count,
                        "num_grasps_after_target_filter": len(filtered_payloads),
                        "target_filter": target_filter,
                        "best": best_payload,
                        "top_grasps": top_grasps,
                        "final_grasp_pose": {
                            "frame": "world",
                            "pose_type": "anygrasp_gripper",
                            "approach_axis": "rotation_matrix[:,0]",
                            "pregrasp_direction": "-rotation_matrix[:,0]",
                            **(best_payload["pose_world"] or {}),
                            "score": best_payload["score"],
                            "width": best_payload["width"],
                            "depth": best_payload["depth"],
                        },
                    }
                except BaseException as exc:
                    attempt_records.append({**attempt, "status": "failed", "error": repr(exc)})
                    continue
            return {
                "status": "no_grasp",
                "workspace_lims": lims,
                "openssl_preload": openssl_messages,
                "attempts": attempt_records,
            }
    except SystemExit as exc:
        return {
            "status": "license_failed",
            "error": repr(exc),
            "workspace_lims": lims,
            "license_dir": str(args.anygrasp_root / "grasp_detection/license"),
            "openssl_preload": openssl_messages,
            "hint": (
                "The AnyGrasp binary loaded but rejected the license. Confirm the "
                "license feature id matches this machine and that licenseCfg.json "
                "is installed under grasp_detection/license."
            ),
        }
    except ImportError as exc:
        error = repr(exc)
        status = "missing_openssl11" if "libcrypto.so.1.1" in error or "libssl.so.1.1" in error else "import_failed"
        return {
            "status": status,
            "error": error,
            "workspace_lims": lims,
            "openssl_preload": openssl_messages,
            "hint": (
                f"Use --openssl-lib-dir {DEFAULT_OPENSSL_LIB_DIR} or set "
                "ANYGRASP_OPENSSL_LIB_DIR if the SDK cannot find OpenSSL 1.1."
            )
            if status == "missing_openssl11"
            else None,
        }
    except BaseException as exc:
        return (
            {
                "status": "import_failed",
                "error": repr(exc),
                "workspace_lims": lims,
                "openssl_preload": openssl_messages,
            }
            if "gsnet" in repr(exc).lower()
            else {
                "status": "load_or_inference_failed",
                "error": repr(exc),
                "workspace_lims": lims,
                "openssl_preload": openssl_messages,
            }
        )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    camera_payload = load_json(args.camera_json)
    camera = camera_payload.get("camera", camera_payload)
    intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float32)

    rgb = np.asarray(Image.open(args.rgb).convert("RGB"), dtype=np.float32) / 255.0
    depth = to_depth_array(args.depth_npy)
    mask = load_mask(args.mask, depth.shape)
    points_organized = backproject(depth, intrinsic)
    target_valid = mask & np.isfinite(depth) & (depth > 0.0) & (depth < args.max_depth)
    target_points = points_organized[target_valid].astype(np.float32)
    target_colors = rgb[target_valid].astype(np.float32)
    ys, xs = np.where(target_valid)
    view_records = [
        {
            "name": "primary",
            "rgb": str(args.rgb.resolve()),
            "depth_npy": str(args.depth_npy.resolve()),
            "mask": str(args.mask.resolve()),
            "camera_json": str(args.camera_json.resolve()),
            "point_count": int(target_points.shape[0]),
            "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if xs.size else None,
            "frame": "primary_camera",
        }
    ]

    target_point_batches = [target_points]
    target_color_batches = [target_colors]
    scene_point_batches = []
    scene_color_batches = []
    scene_view_records = []
    if args.anygrasp_cloud_mode == "full_scene_target_filter":
        scene_points, scene_colors, scene_stats = scene_points_for_view(
            args.rgb,
            args.depth_npy,
            camera,
            args.max_depth,
            args.scene_cloud_stride,
        )
        scene_stats["name"] = "primary"
        scene_stats["camera_json"] = str(args.camera_json.resolve())
        scene_stats["frame"] = "primary_camera"
        scene_point_batches.append(scene_points)
        scene_color_batches.append(scene_colors)
        scene_view_records.append(scene_stats)

    for extra_idx, extra_spec in enumerate(args.extra_view, start=2):
        extra_rgb, extra_depth, extra_mask, extra_camera_json = parse_extra_view(extra_spec)
        extra_camera_payload = load_json(extra_camera_json)
        extra_camera = extra_camera_payload.get("camera", extra_camera_payload)
        extra_points, extra_colors, extra_stats = masked_points_for_view(
            extra_rgb,
            extra_depth,
            extra_mask,
            extra_camera,
            args.max_depth,
        )
        extra_stats["name"] = f"extra_{extra_idx:02d}"
        extra_stats["camera_json"] = str(extra_camera_json.resolve())
        extra_stats["frame"] = "reprojected_to_primary_camera"
        if len(extra_points) > 0:
            extra_world = transform_points_to_world(extra_points, extra_camera)
            extra_primary = transform_points_from_world(extra_world, camera).astype(np.float32)
            keep = np.isfinite(extra_primary).all(axis=1) & (extra_primary[:, 2] > 0.0)
            extra_stats["point_count_after_primary_reprojection"] = int(np.count_nonzero(keep))
            if keep.any():
                target_point_batches.append(extra_primary[keep])
                target_color_batches.append(extra_colors[keep])
        else:
            extra_stats["point_count_after_primary_reprojection"] = 0
        view_records.append(extra_stats)

        if args.anygrasp_cloud_mode == "full_scene_target_filter":
            extra_scene_points, extra_scene_colors, extra_scene_stats = scene_points_for_view(
                extra_rgb,
                extra_depth,
                extra_camera,
                args.max_depth,
                args.scene_cloud_stride,
            )
            extra_scene_stats["name"] = f"extra_{extra_idx:02d}"
            extra_scene_stats["camera_json"] = str(extra_camera_json.resolve())
            extra_scene_stats["frame"] = "reprojected_to_primary_camera"
            if len(extra_scene_points) > 0:
                extra_scene_world = transform_points_to_world(extra_scene_points, extra_camera)
                extra_scene_primary = transform_points_from_world(extra_scene_world, camera).astype(np.float32)
                scene_keep = np.isfinite(extra_scene_primary).all(axis=1) & (extra_scene_primary[:, 2] > 0.0)
                extra_scene_stats["point_count_after_primary_reprojection"] = int(np.count_nonzero(scene_keep))
                if scene_keep.any():
                    scene_point_batches.append(extra_scene_primary[scene_keep])
                    scene_color_batches.append(extra_scene_colors[scene_keep])
            else:
                extra_scene_stats["point_count_after_primary_reprojection"] = 0
            scene_view_records.append(extra_scene_stats)

    if len(target_point_batches) > 1:
        target_points = np.concatenate(target_point_batches, axis=0).astype(np.float32)
        target_colors = np.concatenate(target_color_batches, axis=0).astype(np.float32)

    if args.filter_target_outliers:
        target_points, target_colors, target_filter = largest_voxel_component_filter(
            target_points,
            target_colors,
            args.target_filter_voxel_size,
            args.target_filter_min_points,
        )
    else:
        target_filter = {
            "enabled": False,
            "input_point_count": int(len(target_points)),
            "output_point_count": int(len(target_points)),
            "applied": False,
        }

    observed_target_points = target_points.copy()
    observed_target_colors = target_colors.copy()
    object_center_world = None
    if len(args.overlay_object_center_world) >= 3:
        object_center_world = np.asarray(args.overlay_object_center_world[:3], dtype=np.float64)
    target_points, target_colors, symmetry_completion = complete_symmetric_cloud_camera(
        target_points,
        target_colors,
        camera,
        object_center_world,
        args,
    )

    if args.anygrasp_cloud_mode == "full_scene_target_filter":
        if scene_point_batches:
            anygrasp_points = np.concatenate(scene_point_batches, axis=0).astype(np.float32)
            anygrasp_colors = np.concatenate(scene_color_batches, axis=0).astype(np.float32)
        else:
            anygrasp_points = target_points
            anygrasp_colors = target_colors
        anygrasp_points, anygrasp_colors, downsample_record = deterministic_downsample(
            anygrasp_points,
            anygrasp_colors,
            args.scene_cloud_max_points,
        )
    else:
        anygrasp_points = target_points
        anygrasp_colors = target_colors
        downsample_record = {
            "enabled": False,
            "input_point_count": int(len(anygrasp_points)),
            "output_point_count": int(len(anygrasp_points)),
            "applied": False,
        }

    point_cloud_path = args.output / "masked_cloud.npy"
    color_path = args.output / "masked_cloud_colors.npy"
    ply_path = args.output / "masked_cloud.ply"
    np.save(point_cloud_path, target_points)
    np.save(color_path, target_colors)
    if len(target_points) > 0:
        write_ply(ply_path, target_points, target_colors)
    observed_point_cloud_path = args.output / "observed_masked_cloud.npy"
    observed_color_path = args.output / "observed_masked_cloud_colors.npy"
    observed_ply_path = args.output / "observed_masked_cloud.ply"
    np.save(observed_point_cloud_path, observed_target_points)
    np.save(observed_color_path, observed_target_colors)
    if len(observed_target_points) > 0:
        write_ply(observed_ply_path, observed_target_points, observed_target_colors)

    anygrasp_input_path = args.output / "anygrasp_input_cloud.npy"
    anygrasp_input_color_path = args.output / "anygrasp_input_cloud_colors.npy"
    anygrasp_input_ply_path = args.output / "anygrasp_input_cloud.ply"
    np.save(anygrasp_input_path, anygrasp_points)
    np.save(anygrasp_input_color_path, anygrasp_colors)
    if len(anygrasp_points) > 0:
        write_ply(anygrasp_input_ply_path, anygrasp_points, anygrasp_colors)

    if len(anygrasp_points) < 50:
        anygrasp = {
            "status": "insufficient_points",
            "point_count": int(len(anygrasp_points)),
            "target_point_count": int(len(target_points)),
            "reason": "Need at least 50 valid AnyGrasp input depth points.",
        }
        marker = None
    else:
        anygrasp = run_anygrasp(
            args,
            anygrasp_points,
            anygrasp_colors,
            camera,
            intrinsic,
            mask,
            target_points,
        )
        marker = None
        best = anygrasp.get("best") if isinstance(anygrasp, dict) else None
        if isinstance(best, dict) and best.get("translation_camera") is not None:
            marker = np.asarray(best["translation_camera"], dtype=np.float32)

    viz_path = args.output / "anygrasp_result.png"
    save_cloud_views(viz_path, anygrasp_points, anygrasp_colors, anygrasp.get("status", "unknown"), marker=marker)
    target_viz_path = args.output / "target_cloud.png"
    save_cloud_views(target_viz_path, target_points, target_colors, "target_mask", marker=marker)
    top_grasps = anygrasp.get("top_grasps", []) if isinstance(anygrasp, dict) else []
    top_grasps_path = args.output / "top_grasps.json"
    top_grasps_path.write_text(json.dumps(top_grasps, indent=2), encoding="utf-8")
    top_overlay_path = args.output / "top_grasps_overlay.png"
    save_grasp_overlay(
        top_overlay_path,
        rgb,
        mask,
        intrinsic,
        camera,
        top_grasps if isinstance(top_grasps, list) else [],
        anygrasp.get("status", "unknown") if isinstance(anygrasp, dict) else "unknown",
        args,
    )

    summary = {
        "rgb": str(args.rgb.resolve()),
        "depth_npy": str(args.depth_npy.resolve()),
        "mask": str(args.mask.resolve()),
        "camera_json": str(args.camera_json.resolve()),
        "anygrasp_cloud_mode": args.anygrasp_cloud_mode,
        "symmetric_cloud_mode": args.symmetric_cloud_mode,
        "point_count": int(len(target_points)),
        "target_point_count": int(len(target_points)),
        "observed_target_point_count": int(len(observed_target_points)),
        "point_cloud_npy": point_cloud_path.name,
        "point_cloud_colors_npy": color_path.name,
        "point_cloud_ply": ply_path.name if ply_path.exists() else None,
        "observed_point_cloud_npy": observed_point_cloud_path.name,
        "observed_point_cloud_colors_npy": observed_color_path.name,
        "observed_point_cloud_ply": observed_ply_path.name if observed_ply_path.exists() else None,
        "anygrasp_input_point_count": int(len(anygrasp_points)),
        "anygrasp_input_cloud_npy": anygrasp_input_path.name,
        "anygrasp_input_cloud_colors_npy": anygrasp_input_color_path.name,
        "anygrasp_input_cloud_ply": anygrasp_input_ply_path.name if anygrasp_input_ply_path.exists() else None,
        "scene_cloud_downsample": downsample_record,
        "target_outlier_filter": target_filter,
        "symmetry_completion": symmetry_completion,
        "point_cloud_frame": "primary_camera",
        "fused_view_count": len(view_records),
        "views": view_records,
        "scene_views": scene_view_records,
        "visualization": viz_path.name,
        "target_visualization": target_viz_path.name,
        "top_grasps_json": top_grasps_path.name,
        "top_grasps_overlay": top_overlay_path.name,
        "top_grasps_overlay_geometry": {
            "tool_transform": args.overlay_tool_transform,
            "gripper_base_offset_m": float(args.overlay_gripper_base_offset),
            "gripper_base_offset_mode": args.overlay_gripper_base_offset_mode,
            "object_center_world": list(args.overlay_object_center_world[:3]),
            "finger_length_m": float(args.overlay_finger_length),
            "palm_width_m": float(args.overlay_palm_width),
            "show_full_tool": bool(args.overlay_show_full_tool),
        },
        "anygrasp": anygrasp,
    }
    result_path = args.output / "anygrasp_result.json"
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    final_pose = anygrasp.get("final_grasp_pose") if isinstance(anygrasp, dict) else None
    if isinstance(final_pose, dict) and anygrasp.get("status") == "ok":
        (args.output / "final_grasp_pose.json").write_text(
            json.dumps(final_pose, indent=2), encoding="utf-8"
        )
    else:
        (args.output / "final_grasp_pose_error.json").write_text(
            json.dumps(anygrasp, indent=2), encoding="utf-8"
        )
    print(f"[INFO] Saved {result_path}")
    print(f"[INFO] Saved {viz_path}")
    print(f"[INFO] Saved {top_overlay_path}")


if __name__ == "__main__":
    main()
