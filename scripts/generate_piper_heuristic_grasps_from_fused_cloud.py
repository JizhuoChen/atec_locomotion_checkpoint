#!/usr/bin/env python3
"""Generate Piper-sized heuristic grasp candidates from an existing fused cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from contact_graspnet_from_rgbd_mask import (
    load_json,
    transform_points_from_world,
    transform_points_to_world,
    write_ply,
)


PIPER_FINGER_LENGTH_M = 0.0765
PIPER_FINGER_WIDTH_M = 0.0265
PIPER_FINGER_DEPTH_M = 0.056
PIPER_PALM_APPROACH_THICKNESS_M = 0.035


def parse_vec3(value: str) -> list[float]:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-grasp-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--attempts", type=int, default=25000)
    parser.add_argument("--camera-json", type=Path, default=None)
    parser.add_argument("--object-center-world", type=parse_vec3, default=None)
    parser.add_argument("--jaw-width", type=float, default=0.075)
    parser.add_argument("--centerline-radius", type=float, default=0.018)
    parser.add_argument("--min-gap-points", type=int, default=20)
    parser.add_argument("--max-finger-collision-points", type=int, default=0)
    parser.add_argument("--bbox-trim-quantile", type=float, default=0.01)
    parser.add_argument(
        "--heuristic-profile",
        choices=("observed_centerline", "symmetric_bottle", "mixed", "box_edge", "box_top_center_arm_side"),
        default="observed_centerline",
    )
    parser.add_argument(
        "--heuristic-family-mode",
        choices=("auto", "topdown", "tilted_top", "side_orbit", "side_x", "side_y", "diagonal_side", "mixed_diverse"),
        default="auto",
        help="Restrict or diversify the pose family sampled by the symmetric heuristic.",
    )
    parser.add_argument(
        "--symmetric-cloud-mode",
        choices=("off", "mirror", "bottle_surface"),
        default="bottle_surface",
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
    parser.add_argument("--symmetric-top-grasp-fraction", type=float, default=0.70)
    parser.add_argument("--symmetric-min-z-margin", type=float, default=0.012)
    parser.add_argument("--candidate-filter-max-points", type=int, default=60000)
    parser.add_argument("--obstacle-cloud-max-points", type=int, default=60000)
    parser.add_argument("--obstacle-target-exclusion-radius", type=float, default=0.012)
    parser.add_argument("--piper-offset-modes", default="approach_axis,finger_centerline")
    parser.add_argument("--piper-offset-min", type=float, default=0.0)
    parser.add_argument("--piper-offset-max", type=float, default=0.12)
    parser.add_argument("--piper-offset-step", type=float, default=0.005)
    parser.add_argument("--centerline-half-width", type=float, default=0.012)
    parser.add_argument("--centerline-half-depth", type=float, default=0.018)
    parser.add_argument("--centerline-min-points", type=int, default=10)
    parser.add_argument("--closing-region-min-points", type=int, default=10)
    parser.add_argument("--middle-support-min-points", type=int, default=5)
    parser.add_argument("--middle-support-offset", type=float, default=0.025)
    parser.add_argument("--middle-support-half-length", type=float, default=0.012)
    parser.add_argument("--middle-support-half-width", type=float, default=0.014)
    parser.add_argument("--middle-support-half-depth", type=float, default=0.020)
    parser.add_argument("--root-centerline-clear-length", type=float, default=0.025)
    parser.add_argument("--root-centerline-max-points", type=int, default=0)
    parser.add_argument("--target-solid-max-points", type=int, default=0)
    parser.add_argument("--obstacle-solid-max-points", type=int, default=0)
    parser.add_argument("--approach-clearance", type=float, default=0.002)
    parser.add_argument(
        "--symmetric-ee-roll",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also emit the 180-degree roll-equivalent gripper orientation for each accepted "
            "Piper grasp. This preserves the parallel-jaw contact geometry but may use a "
            "different wrist joint configuration."
        ),
    )
    return parser.parse_args()


def normalize(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        return fallback.astype(np.float64)
    return value.astype(np.float64) / norm


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
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


def robust_bbox(points_world: np.ndarray, trim_quantile: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    q = float(np.clip(trim_quantile, 0.0, 0.20))
    trimmed = points_world
    if q > 0.0 and len(points_world) >= 64:
        lo, hi = np.quantile(points_world, [q, 1.0 - q], axis=0)
        keep = np.all((points_world >= lo) & (points_world <= hi), axis=1)
        if np.count_nonzero(keep) >= max(32, int(0.2 * len(points_world))):
            trimmed = points_world[keep]
    bbox_min = trimmed.min(axis=0)
    bbox_max = trimmed.max(axis=0)
    center = (bbox_min + bbox_max) * 0.5
    return bbox_min, bbox_max, center, {
        "trim_quantile": q,
        "input_point_count": int(len(points_world)),
        "trimmed_point_count": int(len(trimmed)),
        "bbox_min_world": bbox_min.astype(float).tolist(),
        "bbox_max_world": bbox_max.astype(float).tolist(),
        "bbox_size_world": (bbox_max - bbox_min).astype(float).tolist(),
        "bbox_center_world": center.astype(float).tolist(),
    }


def symmetry_center_world(
    points_world: np.ndarray,
    bbox_center: np.ndarray,
    object_center: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if args.symmetry_center_source == "object_center":
        return np.asarray(object_center, dtype=np.float64).copy()
    if args.symmetry_center_source == "mean":
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


def complete_symmetric_bottle_cloud(
    observed_points_world: np.ndarray,
    observed_colors: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    bbox_center: np.ndarray,
    object_center_world: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict]:
    mode = str(args.symmetric_cloud_mode)
    if mode == "off" or str(args.heuristic_profile) == "observed_centerline":
        return observed_points_world, observed_colors, {"enabled": False, "mode": mode}

    center = symmetry_center_world(observed_points_world, bbox_center, object_center_world, args)
    bbox_size = np.maximum(bbox_max - bbox_min, 1e-4)
    rx = float(args.symmetric_body_rx) if float(args.symmetric_body_rx) > 0.0 else float(np.clip(bbox_size[0] * 0.55, 0.028, 0.060))
    ry = float(args.symmetric_body_ry) if float(args.symmetric_body_ry) > 0.0 else float(np.clip(bbox_size[1] * 0.58, 0.020, 0.043))
    z_min = float(bbox_min[2] + max(0.0, float(args.symmetric_min_z_margin)))
    z_max = float(bbox_max[2] - max(0.0, float(args.symmetric_min_z_margin) * 0.5))
    if z_max <= z_min + 0.04:
        z_min = float(bbox_min[2])
        z_max = float(bbox_max[2])

    points = [observed_points_world]
    colors = [observed_colors]
    mirrored = []
    mirrored_colors = []
    if mode in {"mirror", "bottle_surface"}:
        for flip_x, flip_y in ((True, False), (False, True), (True, True)):
            reflected = observed_points_world.copy()
            if flip_x:
                reflected[:, 0] = 2.0 * center[0] - reflected[:, 0]
            if flip_y:
                reflected[:, 1] = 2.0 * center[1] - reflected[:, 1]
            mirrored.append(reflected)
            mirrored_colors.append(observed_colors)
    if mirrored:
        points.extend(mirrored)
        colors.extend(mirrored_colors)

    synthetic_count = max(0, int(args.symmetric_surface_points)) if mode == "bottle_surface" else 0
    if synthetic_count:
        z_count = max(24, int(np.sqrt(synthetic_count) * 0.85))
        theta_count = max(48, int(np.ceil(synthetic_count / z_count)))
        z_values = np.linspace(z_min, z_max, z_count)
        theta_values = np.linspace(-np.pi, np.pi, theta_count, endpoint=False)
        zz, tt = np.meshgrid(z_values, theta_values, indexing="ij")
        z_norm = (zz - z_min) / max(z_max - z_min, 1e-6)
        rxs, rys = bottle_radius_profile(z_norm, rx, ry, float(args.symmetric_neck_radius_scale))
        # Superellipse-style outline: rectangular enough for a mustard bottle, still smooth for sampling.
        c = np.cos(tt)
        s = np.sin(tt)
        power = 0.55
        x = center[0] + rxs * np.sign(c) * (np.abs(c) ** power)
        y = center[1] + rys * np.sign(s) * (np.abs(s) ** power)
        surface = np.stack([x.reshape(-1), y.reshape(-1), zz.reshape(-1)], axis=1)
        top_theta = theta_values
        top_rx, top_ry = bottle_radius_profile(
            np.ones_like(top_theta),
            rx,
            ry,
            float(args.symmetric_neck_radius_scale),
        )
        cap_r = np.linspace(0.0, 1.0, max(4, theta_count // 12))
        cap_points = []
        for scale in cap_r:
            cap_points.append(
                np.stack(
                    [
                        center[0] + top_rx * scale * np.cos(top_theta),
                        center[1] + top_ry * scale * np.sin(top_theta),
                        np.full_like(top_theta, z_max),
                    ],
                    axis=1,
                )
            )
        synthetic = np.concatenate([surface, *cap_points], axis=0)
        synthetic_color = np.tile(np.array([[0.95, 0.78, 0.12]], dtype=np.float64), (len(synthetic), 1))
        points.append(synthetic)
        colors.append(synthetic_color)

    completed_points = np.concatenate(points, axis=0).astype(np.float64)
    completed_colors = np.concatenate(colors, axis=0).astype(np.float64)
    return completed_points, completed_colors, {
        "enabled": True,
        "mode": mode,
        "symmetry_center_source": str(args.symmetry_center_source),
        "center_world": center.astype(float).tolist(),
        "body_rx_m": rx,
        "body_ry_m": ry,
        "z_min_m": z_min,
        "z_max_m": z_max,
        "neck_radius_scale": float(args.symmetric_neck_radius_scale),
        "observed_point_count": int(len(observed_points_world)),
        "completed_point_count": int(len(completed_points)),
        "synthetic_surface_point_budget": int(args.symmetric_surface_points),
    }


def piper_axes_world(approach: np.ndarray, roll: float) -> np.ndarray:
    approach = normalize(approach, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(approach, up))) > 0.94:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    side0 = normalize(up - np.dot(up, approach) * approach, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    jaw0 = normalize(np.cross(approach, side0), np.array([0.0, 1.0, 0.0], dtype=np.float64))
    side = normalize(np.cos(roll) * side0 + np.sin(roll) * jaw0, side0)
    jaw = normalize(np.cross(approach, side), jaw0)
    return np.stack([side, jaw, approach], axis=1)


def graspnet_rotation_from_piper_axes(piper_axes: np.ndarray) -> np.ndarray:
    side = piper_axes[:, 0]
    jaw = piper_axes[:, 1]
    approach = piper_axes[:, 2]
    return np.stack([approach, jaw, -side], axis=1)


def segment_distance_counts(
    points_world: np.ndarray,
    root: np.ndarray,
    tip: np.ndarray,
    radius: float,
) -> int:
    segment = tip - root
    denom = max(float(np.dot(segment, segment)), 1e-12)
    t = ((points_world - root) @ segment) / denom
    keep = (t >= 0.0) & (t <= 1.0)
    closest = root + np.clip(t, 0.0, 1.0)[:, None] * segment
    distances = np.linalg.norm(points_world - closest, axis=1)
    return int(np.count_nonzero(keep & (distances <= radius)))


def deterministic_downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if int(max_points) <= 0 or len(points) <= int(max_points):
        return points
    indices = np.linspace(0, len(points) - 1, int(max_points)).astype(np.int64)
    return points[indices]


def backproject(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    z = depth.astype(np.float32)
    x = (xx - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (yy - intrinsic[1, 2]) * z / intrinsic[1, 1]
    return np.stack([x, y, z], axis=-1)


def reconstruct_scene_points_from_views(source_result: dict, max_depth: float, stride: int) -> tuple[np.ndarray, dict]:
    view_points_world = []
    view_records = []
    stride = max(1, int(stride))
    for index, view in enumerate(source_result.get("views") or [], start=1):
        depth_path_value = view.get("depth_npy")
        camera_path_value = view.get("camera_json")
        if not depth_path_value or not camera_path_value:
            continue
        depth_path = Path(depth_path_value).expanduser().resolve()
        camera_path = Path(camera_path_value).expanduser().resolve()
        if not depth_path.exists() or not camera_path.exists():
            continue
        camera_payload = load_json(camera_path)
        camera = camera_payload.get("camera", camera_payload)
        depth = np.load(depth_path).astype(np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]
        if depth.ndim != 2:
            continue
        if stride > 1:
            depth_sample = depth[::stride, ::stride]
            intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float64).copy()
            intrinsic[0, 0] /= stride
            intrinsic[1, 1] /= stride
            intrinsic[0, 2] /= stride
            intrinsic[1, 2] /= stride
        else:
            depth_sample = depth
            intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float64)
        points_camera = backproject(depth_sample, intrinsic)
        valid = np.isfinite(depth_sample) & (depth_sample > 0.0) & (depth_sample < float(max_depth))
        if not valid.any():
            view_records.append(
                {
                    "name": view.get("name", f"view_{index:02d}"),
                    "camera_json": str(camera_path),
                    "depth_npy": str(depth_path),
                    "point_count": 0,
                }
            )
            continue
        points_world = transform_points_to_world(points_camera[valid].astype(np.float64), camera)
        view_points_world.append(points_world)
        view_records.append(
            {
                "name": view.get("name", f"view_{index:02d}"),
                "camera_json": str(camera_path),
                "depth_npy": str(depth_path),
                "point_count": int(len(points_world)),
            }
        )
    if not view_points_world:
        return np.empty((0, 3), dtype=np.float64), {
            "source": "reconstructed_rgbd_views",
            "point_count": 0,
            "reason": "no_reconstructable_views",
            "view_count": int(len(view_records)),
            "views": view_records,
        }
    points_world = np.concatenate(view_points_world, axis=0)
    return points_world, {
        "source": "reconstructed_rgbd_views",
        "point_count": int(len(points_world)),
        "view_count": int(len(view_records)),
        "views": view_records,
        "scene_cloud_stride": int(stride),
    }


def nearest_target_exclusion_mask(scene_points: np.ndarray, target_points: np.ndarray, radius: float) -> tuple[np.ndarray, str]:
    if len(scene_points) == 0 or len(target_points) == 0:
        return np.ones(len(scene_points), dtype=bool), "empty_scene_or_target"
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(target_points)
        distances, _ = tree.query(scene_points, k=1, workers=-1)
        return distances > float(radius), "scipy_cKDTree"
    except Exception:
        keep = np.ones(len(scene_points), dtype=bool)
        chunk = 2048
        radius_sq = float(radius) * float(radius)
        for start in range(0, len(scene_points), chunk):
            query = scene_points[start : start + chunk]
            min_sq = np.full(len(query), np.inf, dtype=np.float64)
            for target_start in range(0, len(target_points), chunk):
                target = target_points[target_start : target_start + chunk]
                diff = query[:, None, :] - target[None, :, :]
                min_sq = np.minimum(min_sq, np.min(np.sum(diff * diff, axis=2), axis=1))
            keep[start : start + chunk] = min_sq > radius_sq
        return keep, "chunked_numpy"


def load_obstacle_points_world(
    source: Path,
    camera: dict,
    target_points_world: np.ndarray,
    source_result: dict,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    scene_path = None
    for name in ("anygrasp_input_cloud.npy", "graspgen_input_cloud.npy"):
        path = source / name
        if path.exists():
            scene_path = path
            break
    if scene_path is None:
        scene_points_world, scene_meta = reconstruct_scene_points_from_views(
            source_result,
            2.0,
            1,
        )
        scene_meta["scene_cloud_npy"] = None
    else:
        scene_points_world = transform_points_to_world(np.load(scene_path).astype(np.float64), camera)
        scene_meta = {
            "source": "saved_generator_scene_cloud",
            "scene_cloud_npy": str(scene_path),
            "point_count": int(len(scene_points_world)),
        }
    scene_points_world = deterministic_downsample_points(scene_points_world, int(args.obstacle_cloud_max_points))
    keep, method = nearest_target_exclusion_mask(
        scene_points_world,
        target_points_world,
        float(args.obstacle_target_exclusion_radius),
    )
    obstacle_points_world = scene_points_world[keep]
    return obstacle_points_world, {
        **scene_meta,
        "method": method,
        "target_exclusion_radius_m": float(args.obstacle_target_exclusion_radius),
        "scene_point_count_after_downsample": int(len(scene_points_world)),
        "obstacle_point_count": int(len(obstacle_points_world)),
        "max_scene_points": int(args.obstacle_cloud_max_points),
    }


def gripper_stats(
    points_world: np.ndarray,
    root: np.ndarray,
    piper_axes: np.ndarray,
    jaw_width: float,
    centerline_radius: float,
) -> dict:
    local = (points_world - root) @ piper_axes
    side = local[:, 0]
    jaw = local[:, 1]
    along = local[:, 2]
    in_length = (along >= 0.0) & (along <= PIPER_FINGER_LENGTH_M)

    gap = (
        in_length
        & (np.abs(side) <= PIPER_FINGER_DEPTH_M * 0.5)
        & (np.abs(jaw) <= jaw_width * 0.5)
    )
    centerline = in_length & (np.hypot(side, jaw) <= centerline_radius)

    finger_collision = np.zeros(len(points_world), dtype=bool)
    for sign in (-1.0, 1.0):
        finger_center_jaw = sign * (jaw_width * 0.5 + PIPER_FINGER_WIDTH_M * 0.5)
        finger_collision |= (
            in_length
            & (np.abs(side) <= PIPER_FINGER_DEPTH_M * 0.5)
            & (np.abs(jaw - finger_center_jaw) <= PIPER_FINGER_WIDTH_M * 0.5)
        )

    return {
        "gap_point_count": int(np.count_nonzero(gap)),
        "centerline_near_surface_point_count": int(np.count_nonzero(centerline)),
        "finger_collision_point_count": int(np.count_nonzero(finger_collision)),
    }


def pipeline_piper_stats_at_base(
    points_world: np.ndarray,
    piper_axes: np.ndarray,
    base_world: np.ndarray,
    jaw_width: float,
    clearance: float,
    args: argparse.Namespace,
) -> dict:
    side_axis = normalize(piper_axes[:, 0], np.array([1.0, 0.0, 0.0], dtype=np.float64))
    jaw_axis = normalize(piper_axes[:, 1], np.array([0.0, 1.0, 0.0], dtype=np.float64))
    approach_axis = normalize(piper_axes[:, 2], np.array([0.0, 0.0, -1.0], dtype=np.float64))
    points = np.asarray(points_world, dtype=np.float64)
    base = np.asarray(base_world, dtype=np.float64)
    clearance = max(0.0, float(clearance))
    jaw_width = float(np.clip(jaw_width, 0.030, float(args.jaw_width)))
    finger_root = base - approach_axis * PIPER_FINGER_LENGTH_M

    inside_solid = np.zeros(points.shape[0], dtype=bool)
    components: dict[str, int] = {}
    for side_sign, label in [(-1.0, "left_finger"), (1.0, "right_finger")]:
        root_center = finger_root + jaw_axis * side_sign * (jaw_width * 0.5 + PIPER_FINGER_WIDTH_M * 0.5)
        rel = points - root_center[None, :]
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
    rel = points - palm_center[None, :]
    palm_inside = (
        (np.abs(rel @ approach_axis) <= PIPER_PALM_APPROACH_THICKNESS_M * 0.5 + clearance)
        & (np.abs(rel @ jaw_axis) <= (jaw_width * 0.5 + PIPER_FINGER_WIDTH_M) + clearance)
        & (np.abs(rel @ side_axis) <= PIPER_FINGER_DEPTH_M * 0.5 + clearance)
    )
    components["palm_base"] = int(np.count_nonzero(palm_inside))
    inside_solid |= palm_inside

    rel = points - finger_root[None, :]
    along = rel @ approach_axis
    jaw = rel @ jaw_axis
    side = rel @ side_axis
    closing_region = (
        (along >= -clearance)
        & (along <= PIPER_FINGER_LENGTH_M + clearance)
        & (np.abs(jaw) <= jaw_width * 0.5 + clearance)
        & (np.abs(side) <= PIPER_FINGER_DEPTH_M * 0.5 + clearance)
    )
    centerline = (
        (along >= -clearance)
        & (along <= PIPER_FINGER_LENGTH_M + clearance)
        & (np.abs(jaw) <= float(args.centerline_half_width) + clearance)
        & (np.abs(side) <= float(args.centerline_half_depth) + clearance)
    )
    root_clear_length = float(np.clip(args.root_centerline_clear_length, 0.0, PIPER_FINGER_LENGTH_M))
    root_centerline = (
        (along >= -clearance)
        & (along <= root_clear_length + clearance)
        & (np.abs(jaw) <= float(args.centerline_half_width) + clearance)
        & (np.abs(side) <= float(args.centerline_half_depth) + clearance)
    )
    middle_offset = float(np.clip(args.middle_support_offset, 0.0, PIPER_FINGER_LENGTH_M))
    middle_center = base - approach_axis * middle_offset
    rel_mid = points - middle_center[None, :]
    middle_support = (
        (np.abs(rel_mid @ approach_axis) <= float(args.middle_support_half_length) + clearance)
        & (np.abs(rel_mid @ jaw_axis) <= float(args.middle_support_half_width) + clearance)
        & (np.abs(rel_mid @ side_axis) <= float(args.middle_support_half_depth) + clearance)
    )
    return {
        "solid_point_count": int(np.count_nonzero(inside_solid)),
        "component_collision_points": components,
        "closing_region_point_count": int(np.count_nonzero(closing_region)),
        "centerline_point_count": int(np.count_nonzero(centerline)),
        "root_centerline_point_count": int(np.count_nonzero(root_centerline)),
        "root_centerline_clear_length_m": float(root_clear_length),
        "middle_support_point_count": int(np.count_nonzero(middle_support)),
        "middle_support_center_w": middle_center.astype(float).tolist(),
        "middle_support_offset_m": float(middle_offset),
        "middle_support_box_half_extents_m": [
            float(args.middle_support_half_length),
            float(args.middle_support_half_width),
            float(args.middle_support_half_depth),
        ],
        "gripper_base_w": base.astype(float).tolist(),
        "finger_root_center_w": finger_root.astype(float).tolist(),
        "finger_tip_center_w": base.astype(float).tolist(),
        "geometry_reference": "offset_pose_is_finger_tip_contact_end",
        "jaw_width_m": jaw_width,
    }


def piper_offset_values(args: argparse.Namespace) -> list[float]:
    step = max(1e-5, float(args.piper_offset_step))
    minimum = float(args.piper_offset_min)
    maximum = max(minimum, float(args.piper_offset_max))
    values = [float(v) for v in np.arange(minimum, maximum + step * 0.5, step)]
    if not values or values[-1] < maximum:
        values.append(maximum)
    return sorted({round(float(np.clip(v, minimum, maximum)), 6) for v in values})


def pipeline_offset_axis(mode: str, translation: np.ndarray, approach_axis: np.ndarray, object_center: np.ndarray) -> np.ndarray:
    if mode == "towards_object_center":
        return normalize(object_center - translation, -approach_axis)
    if mode in {"finger_centerline", "yellow_line"}:
        return approach_axis
    return -approach_axis


def pipeline_final_geometry_search(
    target_points_world: np.ndarray,
    obstacle_points_world: np.ndarray,
    translation: np.ndarray,
    piper_axes: np.ndarray,
    object_center_world: np.ndarray,
    jaw_width: float,
    args: argparse.Namespace,
) -> tuple[bool, dict]:
    approach_axis = normalize(piper_axes[:, 2], np.array([1.0, 0.0, 0.0], dtype=np.float64))
    modes = [part.strip() for part in str(args.piper_offset_modes).split(",") if part.strip()]
    tested = []
    passing = []
    for mode in modes:
        axis = pipeline_offset_axis(mode, translation, approach_axis, object_center_world)
        for offset in piper_offset_values(args):
            base = translation + axis * float(offset)
            target = pipeline_piper_stats_at_base(
                target_points_world,
                piper_axes,
                base,
                jaw_width,
                float(args.approach_clearance),
                args,
            )
            obstacle = pipeline_piper_stats_at_base(
                obstacle_points_world,
                piper_axes,
                base,
                jaw_width,
                float(args.approach_clearance),
                args,
            )
            target_solid_ok = int(target["solid_point_count"]) <= int(args.target_solid_max_points)
            obstacle_solid_ok = int(obstacle["solid_point_count"]) <= int(args.obstacle_solid_max_points)
            centerline_ok = int(target["centerline_point_count"]) >= int(args.centerline_min_points)
            closing_ok = int(target["closing_region_point_count"]) >= int(args.closing_region_min_points)
            middle_support_ok = int(target["middle_support_point_count"]) >= int(args.middle_support_min_points)
            root_centerline_clear_ok = int(target["root_centerline_point_count"]) <= int(args.root_centerline_max_points)
            entry = {
                "ok": bool(
                    target_solid_ok
                    and obstacle_solid_ok
                    and centerline_ok
                    and closing_ok
                    and middle_support_ok
                    and root_centerline_clear_ok
                ),
                "offset_mode": mode,
                "offset_m": float(offset),
                "offset_axis_w": axis.astype(float).tolist(),
                "target": target,
                "obstacle": obstacle,
                "checks": {
                    "target_solid_ok": bool(target_solid_ok),
                    "obstacle_solid_ok": bool(obstacle_solid_ok),
                    "centerline_ok": bool(centerline_ok),
                    "closing_region_ok": bool(closing_ok),
                    "middle_support_ok": bool(middle_support_ok),
                    "root_centerline_clear_ok": bool(root_centerline_clear_ok),
                    "target_solid_max_points": int(args.target_solid_max_points),
                    "obstacle_solid_max_points": int(args.obstacle_solid_max_points),
                    "centerline_min_points": int(args.centerline_min_points),
                    "closing_region_min_points": int(args.closing_region_min_points),
                    "middle_support_min_points": int(args.middle_support_min_points),
                    "root_centerline_max_points": int(args.root_centerline_max_points),
                },
            }
            tested.append(entry)
            if entry["ok"]:
                passing.append(entry)

    def pass_key(entry: dict) -> tuple:
        return (
            abs(float(entry["offset_m"]) - PIPER_FINGER_LENGTH_M),
            -int(entry["target"]["centerline_point_count"]),
            -int(entry["target"]["closing_region_point_count"]),
            -int(entry["target"]["middle_support_point_count"]),
            int(entry["target"]["root_centerline_point_count"]),
        )

    def fail_key(entry: dict) -> tuple:
        return (
            int(entry["target"]["solid_point_count"]) + int(entry["obstacle"]["solid_point_count"]),
            -int(entry["target"]["centerline_point_count"]),
            -int(entry["target"]["closing_region_point_count"]),
            -int(entry["target"]["middle_support_point_count"]),
            int(entry["target"]["root_centerline_point_count"]),
        )

    selected = sorted(passing, key=pass_key)[0] if passing else sorted(tested, key=fail_key)[0]
    return bool(selected["ok"]), {
        "safe": bool(selected["ok"]),
        "selected": selected,
        "passing_count": int(len(passing)),
        "tested_count": int(len(tested)),
    }


def sample_observed_centerline_candidate(
    rng: np.random.Generator,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    object_center_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    bbox_size = bbox_max - bbox_min
    angle = float(rng.uniform(-np.pi, np.pi))
    z_slope = float(rng.uniform(-0.20, 0.20))
    approach = normalize(np.array([np.cos(angle), np.sin(angle), z_slope], dtype=np.float64), np.array([1, 0, 0]))
    center_noise = np.array(
        [
            rng.uniform(-0.18, 0.18) * bbox_size[0],
            rng.uniform(-0.18, 0.18) * bbox_size[1],
            rng.uniform(-0.28, 0.10) * bbox_size[2],
        ],
        dtype=np.float64,
    )
    center_point = object_center_world + center_noise
    center_point = np.minimum(np.maximum(center_point, bbox_min + bbox_size * 0.08), bbox_max - bbox_size * 0.08)
    return approach, center_point, "observed_side_centerline"


def sample_box_edge_candidate(
    rng: np.random.Generator,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    bbox_center = (bbox_min + bbox_max) * 0.5
    bbox_size = np.maximum(bbox_max - bbox_min, 1e-4)
    corner_x = float(rng.choice([bbox_min[0], bbox_max[0]]))
    corner_y = float(rng.choice([bbox_min[1], bbox_max[1]]))
    inward_x = 1.0 if corner_x <= bbox_center[0] else -1.0
    inward_y = 1.0 if corner_y <= bbox_center[1] else -1.0
    dx = float(rng.uniform(0.006, min(0.036, 0.42 * bbox_size[0])))
    dy = float(rng.uniform(0.004, min(0.026, 0.35 * bbox_size[1])))
    z = float(bbox_max[2] - rng.uniform(0.003, 0.010))
    z = float(np.clip(z, bbox_min[2] + 0.35 * bbox_size[2], bbox_max[2] - 0.05 * bbox_size[2]))
    center_point = np.array(
        [
            corner_x + inward_x * dx,
            corner_y + inward_y * dy,
            z,
        ],
        dtype=np.float64,
    )
    approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    # With top-down approach, this diagonal jaw direction places each finger outside
    # one adjacent box face while the centerline/closing region catches the corner.
    roll_hint = float(np.arctan2(-inward_x, inward_y))
    return approach, center_point, "box_edge_topdown_corner", roll_hint


def sample_box_top_center_arm_side_candidate(
    rng: np.random.Generator,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    bbox_center = (bbox_min + bbox_max) * 0.5
    bbox_size = np.maximum(bbox_max - bbox_min, 1e-4)
    x = float(bbox_center[0] + rng.uniform(-0.12, 0.12) * bbox_size[0])

    # In Task E the box sits at positive Y, while the arm/retract side is near Y=0.
    # Bias strongly toward that near side but keep the sampled centerline inside
    # the top face so the contact-rule filters, not the sampler, decide validity.
    toward_arm_y = -1.0 if bbox_center[1] >= 0.0 else 1.0
    y_shift = float(rng.uniform(0.24, 0.46) * bbox_size[1])
    y = float(bbox_center[1] + toward_arm_y * y_shift)
    y = float(np.clip(y, bbox_min[1] + 0.10 * bbox_size[1], bbox_max[1] - 0.18 * bbox_size[1]))

    z = float(bbox_max[2] - rng.uniform(0.002, 0.008))
    z = float(np.clip(z, bbox_min[2] + 0.50 * bbox_size[2], bbox_max[2] - 0.03 * bbox_size[2]))
    center_point = np.array([x, y, z], dtype=np.float64)

    approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    # The box is long along world Y in Task E, so a center/arm-side top grasp
    # should open across the short X dimension; opening across Y puts both
    # fingers into the target cloud because the Piper jaw cannot straddle it.
    return approach, center_point, "box_top_center_arm_side", 0.5 * np.pi


def sample_symmetric_bottle_candidate(
    rng: np.random.Generator,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    symmetry_meta: dict,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, str]:
    center = np.asarray(symmetry_meta["center_world"], dtype=np.float64)
    rx = float(symmetry_meta["body_rx_m"])
    ry = float(symmetry_meta["body_ry_m"])
    z_min = float(symmetry_meta["z_min_m"])
    z_max = float(symmetry_meta["z_max_m"])
    z_span = max(z_max - z_min, 1e-4)
    top_fraction = float(np.clip(args.symmetric_top_grasp_fraction, 0.0, 1.0))
    family_mode = str(getattr(args, "heuristic_family_mode", "auto"))
    family_draw = float(rng.uniform())

    if family_mode == "mixed_diverse":
        family_mode = str(rng.choice(["tilted_top", "side_orbit", "side_x", "side_y", "diagonal_side"]))
    elif family_mode == "auto":
        if family_draw < top_fraction * 0.55:
            family_mode = "topdown_neck"
        elif family_draw < top_fraction:
            family_mode = "topdown_body"
        elif family_draw < top_fraction + (1.0 - top_fraction) * 0.65:
            family_mode = "tilted_top"
        else:
            family_mode = "side_y"
    elif family_mode == "topdown":
        family_mode = "topdown_neck" if family_draw < 0.55 else "topdown_body"

    if family_mode == "topdown_neck":
        family = "symmetric_topdown_neck"
        approach = normalize(
            np.array([rng.normal(0.0, 0.10), rng.normal(0.0, 0.10), -1.0], dtype=np.float64),
            np.array([0.0, 0.0, -1.0]),
        )
        radius = float(np.clip(args.symmetric_neck_radius_scale, 0.25, 0.85))
        lateral = np.array(
            [
                rng.uniform(-0.35, 0.35) * rx * radius,
                rng.uniform(-0.35, 0.35) * ry * radius,
                0.0,
            ],
            dtype=np.float64,
        )
        z = rng.uniform(z_max - 0.42 * z_span, z_max - 0.10 * z_span)
    elif family_mode == "topdown_body":
        family = "symmetric_topdown_body"
        approach = normalize(
            np.array([rng.normal(0.0, 0.18), rng.normal(0.0, 0.18), -1.0], dtype=np.float64),
            np.array([0.0, 0.0, -1.0]),
        )
        lateral = np.array(
            [
                rng.uniform(-0.18, 0.18) * rx,
                rng.uniform(-0.18, 0.18) * ry,
                0.0,
            ],
            dtype=np.float64,
        )
        z = rng.uniform(z_min + 0.38 * z_span, z_min + 0.72 * z_span)
    elif family_mode == "tilted_top":
        family = "symmetric_tilted_top_front"
        approach = normalize(
            np.array(
                [
                    rng.uniform(-0.55, 0.35),
                    rng.uniform(-0.42, 0.42),
                    -rng.uniform(0.42, 0.88),
                ],
                dtype=np.float64,
            ),
            np.array([0.0, 0.0, -1.0]),
        )
        lateral = np.array(
            [
                rng.uniform(-0.30, 0.30) * rx,
                rng.uniform(-0.30, 0.30) * ry,
                0.0,
            ],
            dtype=np.float64,
        )
        z = rng.uniform(z_min + 0.42 * z_span, z_min + 0.82 * z_span)
    else:
        if family_mode == "side_x":
            angle = float(rng.choice([0.0, np.pi]) + rng.uniform(-0.22, 0.22))
            family = "symmetric_side_x_axis"
        elif family_mode == "side_y":
            angle = float(rng.choice([0.5 * np.pi, -0.5 * np.pi]) + rng.uniform(-0.22, 0.22))
            family = "symmetric_side_y_axis"
        elif family_mode == "diagonal_side":
            angle = float(rng.choice([0.25, 0.75, 1.25, 1.75]) * np.pi + rng.uniform(-0.18, 0.18))
            family = "symmetric_diagonal_side_axis"
        else:
            angle = float(rng.uniform(-np.pi, np.pi))
            family = "symmetric_side_orbit"
        radial = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float64)
        tangent = np.array([-np.sin(angle), np.cos(angle), 0.0], dtype=np.float64)
        z_slope = float(rng.uniform(-0.22, 0.26))
        if family_mode == "diagonal_side":
            z_slope = float(rng.uniform(-0.34, 0.34))
        approach = normalize(
            radial + np.array([0.0, 0.0, z_slope], dtype=np.float64),
            radial,
        )
        lateral = np.array(
            [
                radial[0] * rng.uniform(-0.12, 0.12) * rx + tangent[0] * rng.uniform(-0.22, 0.22) * rx,
                radial[1] * rng.uniform(-0.12, 0.12) * ry + tangent[1] * rng.uniform(-0.22, 0.22) * ry,
                0.0,
            ],
            dtype=np.float64,
        )
        z = rng.uniform(z_min + 0.20 * z_span, z_min + 0.70 * z_span)

    center_point = center + lateral
    center_point[2] = float(np.clip(z, bbox_min[2] + 0.04 * z_span, bbox_max[2] - 0.04 * z_span))
    return approach, center_point, family


def save_inputs(source: Path, output: Path, target_points: np.ndarray, target_colors: np.ndarray) -> None:
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "masked_cloud.npy", target_points.astype(np.float32))
    np.save(output / "masked_cloud_colors.npy", target_colors.astype(np.float32))
    write_ply(output / "masked_cloud.ply", target_points, target_colors)
    for name in ("anygrasp_input_cloud.npy", "anygrasp_input_cloud_colors.npy"):
        src = source / name
        if src.exists():
            np.save(output / name, np.load(src))
    if (output / "anygrasp_input_cloud.npy").exists() and (output / "anygrasp_input_cloud_colors.npy").exists():
        write_ply(
            output / "anygrasp_input_cloud.ply",
            np.load(output / "anygrasp_input_cloud.npy"),
            np.load(output / "anygrasp_input_cloud_colors.npy"),
        )


def main() -> None:
    args = parse_args()
    source = args.source_grasp_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    observed_cloud_path = source / "observed_masked_cloud.npy"
    observed_color_path = source / "observed_masked_cloud_colors.npy"
    if observed_cloud_path.exists() and observed_color_path.exists():
        target_points = np.load(observed_cloud_path).astype(np.float64)
        target_colors = np.load(observed_color_path).astype(np.float64)
    else:
        target_points = np.load(source / "masked_cloud.npy").astype(np.float64)
        target_colors = np.load(source / "masked_cloud_colors.npy").astype(np.float64)

    result_path = source / "graspgen_result.json"
    if not result_path.exists():
        result_path = source / "anygrasp_result.json"
    source_result = load_json(result_path)
    camera_path = args.camera_json or Path(source_result["camera_json"])
    camera_payload = load_json(camera_path.expanduser().resolve())
    camera = camera_payload.get("camera", camera_payload)
    r_wc = quat_wxyz_to_matrix(np.asarray(camera["quat_w_ros"], dtype=np.float64))

    observed_points_world = transform_points_to_world(target_points, camera)
    bbox_min, bbox_max, bbox_center, bbox_details = robust_bbox(observed_points_world, args.bbox_trim_quantile)
    if args.object_center_world is not None:
        object_center_world = np.asarray(args.object_center_world, dtype=np.float64)
    else:
        overlay_center = (source_result.get("top_grasps_overlay_geometry") or {}).get("object_center_world")
        object_center_world = np.asarray(overlay_center if overlay_center else bbox_center, dtype=np.float64)
    points_world, target_colors_for_output, symmetry_meta = complete_symmetric_bottle_cloud(
        observed_points_world,
        target_colors,
        bbox_min,
        bbox_max,
        bbox_center,
        object_center_world,
        args,
    )
    proposal_points_world = deterministic_downsample_points(points_world, int(args.candidate_filter_max_points))
    target_points_for_output = transform_points_from_world(points_world, camera)
    obstacle_points_world, obstacle_meta = load_obstacle_points_world(
        source,
        camera,
        points_world,
        source_result,
        args,
    )
    object_center_camera = transform_points_from_world(object_center_world[None, :], camera)[0]

    rng = np.random.default_rng(args.seed)
    count = max(1, int(args.count))
    jaw_width = float(args.jaw_width)
    candidates: list[dict] = []
    rejected = {
        "finger_collision": 0,
        "insufficient_gap_points": 0,
        "pipeline_final_geometry": 0,
    }

    for attempt in range(1, max(1, int(args.attempts)) + 1):
        if args.heuristic_profile == "symmetric_bottle":
            approach, center_point, family = sample_symmetric_bottle_candidate(
                rng,
                bbox_min,
                bbox_max,
                symmetry_meta,
                args,
            )
        elif args.heuristic_profile == "box_edge":
            approach, center_point, family, box_roll_hint = sample_box_edge_candidate(
                rng,
                bbox_min,
                bbox_max,
            )
        elif args.heuristic_profile == "box_top_center_arm_side":
            approach, center_point, family, box_roll_hint = sample_box_top_center_arm_side_candidate(
                rng,
                bbox_min,
                bbox_max,
            )
        elif args.heuristic_profile == "mixed" and symmetry_meta.get("enabled") and float(rng.uniform()) < 0.70:
            approach, center_point, family = sample_symmetric_bottle_candidate(
                rng,
                bbox_min,
                bbox_max,
                symmetry_meta,
                args,
            )
        else:
            approach, center_point, family = sample_observed_centerline_candidate(
                rng,
                bbox_min,
                bbox_max,
                object_center_world,
            )
        roll = float(rng.uniform(-0.30, 0.30))
        if family.startswith("symmetric_topdown"):
            # Cover both principal jaw orientations on upright bottle grasps.
            roll = float(rng.choice([0.0, np.pi * 0.5, -np.pi * 0.5, np.pi]) + rng.uniform(-0.18, 0.18))
        elif family.startswith("box_edge") or family.startswith("box_top_center"):
            roll = float(box_roll_hint + rng.uniform(-0.18, 0.18))
        axes = piper_axes_world(approach, roll)

        if family.startswith("box_top_center"):
            along_at_center = float(rng.uniform(0.020, 0.038))
        elif family.startswith("box_edge_topdown"):
            along_at_center = float(rng.uniform(0.018, 0.034))
        elif family.startswith("box_edge"):
            along_at_center = float(rng.uniform(0.034, 0.048))
        else:
            along_at_center = float(rng.uniform(0.024, 0.058) if family.startswith("symmetric") else rng.uniform(0.026, 0.050))
        root = center_point - axes[:, 2] * along_at_center
        tip = root + axes[:, 2] * PIPER_FINGER_LENGTH_M
        stats = gripper_stats(proposal_points_world, root, axes, jaw_width, args.centerline_radius)
        if stats["finger_collision_point_count"] > int(args.max_finger_collision_points):
            rejected["finger_collision"] += 1
            continue
        if stats["gap_point_count"] < int(args.min_gap_points):
            rejected["insufficient_gap_points"] += 1
            continue

        geometry_ok, geometry_report = pipeline_final_geometry_search(
            proposal_points_world,
            obstacle_points_world,
            tip,
            axes,
            object_center_world,
            jaw_width,
            args,
        )
        if not geometry_ok:
            rejected["pipeline_final_geometry"] += 1
            continue
        line_count = segment_distance_counts(proposal_points_world, root, tip, args.centerline_radius)
        score = (
            1.0
            + 0.002 * min(stats["gap_point_count"], 300)
            + 0.004 * min(line_count, 100)
            - 0.03 * abs(float(center_point[2] - object_center_world[2]))
        )
        roll_variants = [(roll, axes, "base")]
        if bool(args.symmetric_ee_roll):
            roll_variants.append((roll + np.pi, piper_axes_world(approach, roll + np.pi), "roll_pi"))
        for variant_roll, variant_axes, variant_label in roll_variants:
            rotation_world = graspnet_rotation_from_piper_axes(variant_axes)
            rotation_camera = r_wc.T @ rotation_world
            tip_camera = transform_points_from_world(tip[None, :], camera)[0]
            rank = len(candidates) + 1
            candidates.append(
                {
                    "rank": rank,
                    "source_rank": attempt,
                    "score": float(score),
                    "width": jaw_width,
                    "depth": float(PIPER_FINGER_LENGTH_M),
                    "translation_camera": tip_camera.astype(float).tolist(),
                    "rotation_matrix_camera": rotation_camera.astype(float).tolist(),
                    "pose_world": {
                        "translation": tip.astype(float).tolist(),
                        "rotation_matrix": rotation_world.astype(float).tolist(),
                    },
                    "translation_world": tip.astype(float).tolist(),
                    "rotation_matrix_world": rotation_world.astype(float).tolist(),
                    "translation": tip.astype(float).tolist(),
                    "rotation_matrix": rotation_world.astype(float).tolist(),
                    "generator": "heuristic_piper_centerline",
                    "pose_type": "heuristic_centerline_inside_object_bbox",
                    "heuristic": {
                        "root_world": root.astype(float).tolist(),
                        "tip_world": tip.astype(float).tolist(),
                        "centerline_anchor_world": center_point.astype(float).tolist(),
                        "piper_axes_world": variant_axes.astype(float).tolist(),
                        "base_piper_axes_world": axes.astype(float).tolist(),
                        "centerline_inside_object_bbox": True,
                        "family": family,
                        "symmetry_completion": symmetry_meta,
                        "target_gap_point_count": stats["gap_point_count"],
                        "target_centerline_near_surface_point_count": stats["centerline_near_surface_point_count"],
                        "target_finger_collision_point_count": stats["finger_collision_point_count"],
                        "centerline_radius_m": float(args.centerline_radius),
                        "pipeline_final_geometry": geometry_report,
                        "middle_support_point_count": int(
                            ((geometry_report.get("selected") or {}).get("target") or {}).get(
                                "middle_support_point_count",
                                0,
                            )
                        ),
                        "middle_support_min_points": int(args.middle_support_min_points),
                        "root_centerline_point_count": int(
                            ((geometry_report.get("selected") or {}).get("target") or {}).get(
                                "root_centerline_point_count",
                                0,
                            )
                        ),
                        "root_centerline_max_points": int(args.root_centerline_max_points),
                        "attempt": int(attempt),
                        "roll_rad": float(variant_roll),
                        "symmetric_ee_roll_enabled": bool(args.symmetric_ee_roll),
                        "symmetric_ee_roll_variant": variant_label,
                    },
                }
            )
            if len(candidates) >= count:
                break
        if len(candidates) >= count:
            break

    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank

    save_inputs(source, output, target_points_for_output, target_colors_for_output)
    np.save(output / "observed_masked_cloud.npy", target_points.astype(np.float32))
    np.save(output / "observed_masked_cloud_colors.npy", target_colors.astype(np.float32))
    write_ply(output / "observed_masked_cloud.ply", target_points, target_colors)
    (output / "top_grasps.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    summary = {
        "grasp_generator": "heuristic_piper_centerline",
        "status": "ok" if len(candidates) else "no_candidate",
        "source_grasp_dir": str(source),
        "camera_json": str(camera_path.expanduser().resolve()),
        "point_cloud_frame": "primary_camera",
        "target_point_count": int(len(target_points)),
        "completed_target_point_count": int(len(points_world)),
        "proposal_target_point_count": int(len(proposal_points_world)),
        "fused_view_count": source_result.get("fused_view_count"),
        "views": source_result.get("views"),
        "candidate_count": int(len(candidates)),
        "requested_count": count,
        "attempts": int(args.attempts),
        "rejected": rejected,
        "bbox": bbox_details,
        "symmetry_completion": symmetry_meta,
        "obstacle": obstacle_meta,
        "heuristic_rules": {
            "heuristic_profile": str(args.heuristic_profile),
            "heuristic_family_mode": str(args.heuristic_family_mode),
            "centerline": "sampled centerline passes through the robust object bbox",
            "symmetric_bottle": (
                "when enabled, mirror/synthesize an upright mustard-bottle surface from the partial cloud "
                "and generate top-down, tilted top-front, and thin-axis side grasps on that inferred target"
            ),
            "finger_collision": "reject if segmented bottle points fall inside either Piper finger box",
            "gap": "require segmented bottle points in the open gap between the fingers",
            "pipeline_final_geometry": (
                "also require the same final Piper geometry gate used before CuRobo: "
                "offset search, finger/palm solids, target centerline, target closing region, "
                "2.5 cm middle support, and obstacle solids"
            ),
            "jaw_width_m": jaw_width,
            "finger_length_m": PIPER_FINGER_LENGTH_M,
            "finger_width_m": PIPER_FINGER_WIDTH_M,
            "finger_depth_m": PIPER_FINGER_DEPTH_M,
            "palm_approach_thickness_m": PIPER_PALM_APPROACH_THICKNESS_M,
            "piper_offset_modes": str(args.piper_offset_modes),
            "piper_offset_min_m": float(args.piper_offset_min),
            "piper_offset_max_m": float(args.piper_offset_max),
            "piper_offset_step_m": float(args.piper_offset_step),
            "target_solid_max_points": int(args.target_solid_max_points),
            "obstacle_solid_max_points": int(args.obstacle_solid_max_points),
            "candidate_filter_max_points": int(args.candidate_filter_max_points),
            "centerline_min_points": int(args.centerline_min_points),
            "closing_region_min_points": int(args.closing_region_min_points),
            "middle_support_min_points": int(args.middle_support_min_points),
            "middle_support_offset_m": float(args.middle_support_offset),
            "middle_support_half_length_m": float(args.middle_support_half_length),
            "middle_support_half_width_m": float(args.middle_support_half_width),
            "middle_support_half_depth_m": float(args.middle_support_half_depth),
            "root_centerline_clear_length_m": float(args.root_centerline_clear_length),
            "root_centerline_max_points": int(args.root_centerline_max_points),
            "centerline_half_width_m": float(args.centerline_half_width),
            "centerline_half_depth_m": float(args.centerline_half_depth),
            "symmetric_ee_roll": bool(args.symmetric_ee_roll),
        },
        "top_grasps_overlay_geometry": {
            "object_center_world": object_center_world.astype(float).tolist(),
            "object_center_camera": object_center_camera.astype(float).tolist(),
            "finger_length_m": PIPER_FINGER_LENGTH_M,
            "finger_width_m": PIPER_FINGER_WIDTH_M,
            "finger_depth_m": PIPER_FINGER_DEPTH_M,
            "note": "Heuristic candidates use Piper-sized fingers and are intended for visualization/debug only.",
        },
        "heuristic": {
            "status": "ok" if len(candidates) else "no_candidate",
            "top_grasps": candidates,
        },
        "anygrasp": {
            "status": "ok" if len(candidates) else "no_candidate",
            "generator": "heuristic_piper_centerline",
            "top_grasps": candidates,
            "final_grasp_pose": candidates[0] if candidates else None,
        },
    }
    (output / "anygrasp_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "heuristic_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if candidates:
        final = {
            "frame": "world",
            "generator": "heuristic_piper_centerline",
            "pose_type": candidates[0]["pose_type"],
            "translation": candidates[0]["translation_world"],
            "rotation_matrix": candidates[0]["rotation_matrix_world"],
            "score": candidates[0]["score"],
            "width": candidates[0]["width"],
            "source_rank": candidates[0]["source_rank"],
        }
        (output / "final_grasp_pose.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"[INFO] Saved {len(candidates)} heuristic candidates to {output}")
    print(f"[INFO] Rejected: {json.dumps(rejected, sort_keys=True)}")


if __name__ == "__main__":
    main()
