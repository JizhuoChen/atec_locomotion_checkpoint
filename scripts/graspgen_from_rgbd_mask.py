#!/usr/bin/env python3
"""Run NVLabs GraspGen from the same RGB-D/mask views used by AnyGrasp."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from contact_graspnet_from_rgbd_mask import (
    add_bool_option,
    deterministic_downsample,
    largest_voxel_component_filter,
    load_json,
    load_mask,
    parse_extra_view,
    project_camera_point,
    transform_points_from_world,
    transform_points_to_world,
    transform_pose_to_world,
    view_points,
    write_ply,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRASPGEN_ROOT = REPO_ROOT / "third_party/graspgen"
DEFAULT_GRIPPER_CONFIG = REPO_ROOT / "third_party/graspgen_models/checkpoints/graspgen_robotiq_2f_140.yml"


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", required=True, type=Path)
    parser.add_argument("--depth-npy", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--camera-json", required=True, type=Path)
    parser.add_argument(
        "--extra-view",
        action="append",
        default=[],
        metavar="RGB,DEPTH_NPY,MASK,CAMERA_JSON",
        help="Additional segmented RGB-D view, reprojected into the primary camera frame.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--graspgen-root", type=Path, default=DEFAULT_GRASPGEN_ROOT)
    parser.add_argument("--gripper-config", type=Path, default=DEFAULT_GRIPPER_CONFIG)
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--scene-cloud-stride", type=int, default=1)
    parser.add_argument("--scene-cloud-max-points", type=int, default=180000)
    parser.add_argument("--target-cloud-max-points", type=int, default=2048)
    add_bool_option(
        parser,
        "--filter-target-outliers",
        default=True,
        help_text="Keep only the largest connected voxel component of the fused target cloud.",
    )
    parser.add_argument("--target-filter-voxel-size", type=float, default=0.008)
    parser.add_argument("--target-filter-min-points", type=int, default=50)
    parser.add_argument(
        "--symmetric-cloud-mode",
        choices=("off", "mirror", "bottle_surface"),
        default="off",
        help="Optionally complete an upright symmetric bottle target cloud before GraspGen inference.",
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
    parser.add_argument("--num-grasps", type=int, default=500)
    parser.add_argument("--save-top-grasps", type=int, default=20)
    parser.add_argument("--min-grasps", type=int, default=20)
    parser.add_argument("--max-tries", type=int, default=6)
    parser.add_argument("--grasp-threshold", type=float, default=-1.0)
    parser.add_argument("--default-width", type=float, default=0.14)
    parser.add_argument("--default-depth", type=float, default=0.1034)
    parser.add_argument(
        "--export-tcp-offset",
        type=float,
        default=0.195,
        help=(
            "Distance in metres along GraspGen local +Z from the exported base-link pose "
            "to the grasp/TCP point consumed by the Task E Piper motion contract. "
            "The released Robotiq-2F-140 visualization control points place the fingertips "
            "at z=0.195 m."
        ),
    )
    parser.add_argument("--collision-max-scene-points", type=int, default=8192)
    parser.add_argument("--collision-threshold", type=float, default=0.002)
    parser.add_argument("--target-exclusion-radius", type=float, default=0.006)
    parser.add_argument(
        "--overlay-object-center-world",
        type=parse_float_list,
        default=[],
        help="Accepted for command compatibility with AnyGrasp wrapper.",
    )
    add_bool_option(parser, "--remove-outliers", default=True)
    add_bool_option(parser, "--require-pointnet", default=True)
    add_bool_option(parser, "--filter-collisions", default=True)
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    args.graspgen_root = args.graspgen_root.expanduser().resolve()
    args.gripper_config = args.gripper_config.expanduser().resolve()
    return args


def normalize(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        return fallback.astype(np.float64)
    return np.asarray(value, dtype=np.float64) / norm


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
        draw.text((left + 8, top + 28), "empty cloud", fill=(220, 120, 120))
        return

    sample = points
    sample_colors = colors
    if len(sample) > 20000:
        idx = np.linspace(0, len(sample) - 1, 20000).astype(np.int64)
        sample = sample[idx]
        sample_colors = sample_colors[idx]

    xy = sample[:, list(axes)]
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    if marker is not None:
        marker_xy = marker[list(axes)]
        mins = np.minimum(mins, marker_xy)
        maxs = np.maximum(maxs, marker_xy)
    span = np.maximum(maxs - mins, 1e-4)
    mins -= span * 0.08
    maxs += span * 0.08
    span = maxs - mins

    px = left + 8 + (xy[:, 0] - mins[0]) / span[0] * (right - left - 16)
    py = bottom - 8 - (xy[:, 1] - mins[1]) / span[1] * (bottom - top - 32)
    rgb = np.clip(sample_colors * 255.0, 0, 255).astype(np.uint8)
    for x, y, color in zip(px.astype(int), py.astype(int), rgb, strict=False):
        draw.point((int(x), int(y)), fill=tuple(int(c) for c in color))

    if marker is not None:
        marker_xy = marker[list(axes)]
        mx = left + 8 + (marker_xy[0] - mins[0]) / span[0] * (right - left - 16)
        my = bottom - 8 - (marker_xy[1] - mins[1]) / span[1] * (bottom - top - 32)
        radius = 5
        draw.ellipse((mx - radius, my - radius, mx + radius, my + radius), outline=(255, 40, 40), width=2)


def save_cloud_views(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    status: str,
    marker: np.ndarray | None = None,
    title: str = "Masked RGB-D point cloud | GraspGen",
) -> None:
    canvas = Image.new("RGB", (1260, 520), (18, 22, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), f"{title}: {status}", fill=(255, 255, 255))
    boxes = [(20, 44, 410, 500), (435, 44, 825, 500), (850, 44, 1240, 500)]
    draw_projection(draw, boxes[0], points, colors, (0, 1), "primary camera XY", marker)
    draw_projection(draw, boxes[1], points, colors, (0, 2), "primary camera XZ", marker)
    draw_projection(draw, boxes[2], points, colors, (1, 2), "primary camera YZ", marker)
    canvas.save(path)


def downsample_points_only(points: np.ndarray, max_points: int) -> tuple[np.ndarray, dict]:
    if len(points) == 0:
        return points, {
            "applied": False,
            "input_point_count": 0,
            "output_point_count": 0,
            "max_points": int(max_points),
        }
    dummy = np.zeros((len(points), 3), dtype=np.float32)
    sampled, _, stats = deterministic_downsample(points, dummy, int(max_points))
    return sampled, stats


def symmetry_center_world(
    observed_points_world: np.ndarray,
    bbox_center: np.ndarray,
    object_center_world: np.ndarray | None,
    args: argparse.Namespace,
) -> np.ndarray:
    source = str(args.symmetry_center_source)
    if source == "object_center" and object_center_world is not None and np.isfinite(object_center_world).all():
        center = np.asarray(object_center_world, dtype=np.float64).copy()
        center[2] = bbox_center[2]
        return center
    if source == "mean" and len(observed_points_world):
        center = np.mean(observed_points_world, axis=0)
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


def complete_symmetric_bottle_cloud_camera(
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


def obstacle_cloud_without_target(
    scene_points: np.ndarray,
    target_points: np.ndarray,
    exclusion_radius: float,
    max_points: int,
) -> tuple[np.ndarray, dict]:
    if len(scene_points) == 0 or len(target_points) == 0:
        return scene_points[:0], {
            "input_scene_points": int(len(scene_points)),
            "target_points": int(len(target_points)),
            "output_obstacle_points": 0,
            "target_exclusion_radius_m": float(exclusion_radius),
        }
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(target_points.astype(np.float64))
        distances, _ = tree.query(scene_points.astype(np.float64), k=1, workers=-1)
        keep = distances > float(exclusion_radius)
        obstacles = scene_points[keep].astype(np.float32)
        method = "nearest_target_point_exclusion"
    except Exception as exc:
        obstacles = scene_points.astype(np.float32)
        method = f"fallback_all_scene_points_after_exclusion_failed:{type(exc).__name__}"
    obstacles, downsample = downsample_points_only(obstacles, max_points)
    return obstacles.astype(np.float32), {
        "method": method,
        "input_scene_points": int(len(scene_points)),
        "target_points": int(len(target_points)),
        "output_obstacle_points": int(len(obstacles)),
        "target_exclusion_radius_m": float(exclusion_radius),
        "downsample": downsample,
    }


def graspgen_depth_from_cfg(cfg, fallback: float) -> float:
    try:
        return float(cfg.m2t2.action_decoder.gripper_depth)
    except Exception:
        return float(fallback)


def serialize_graspgen_grasp(
    grasp_matrix: np.ndarray,
    score: float,
    camera: dict,
    rank: int,
    width: float,
    depth: float,
    export_tcp_offset: float,
    collision_free: bool | None,
) -> dict:
    grasp_matrix = np.asarray(grasp_matrix, dtype=np.float64)

    # GraspGen convention: +Z is approach, +X is finger closing direction.
    approach_axis = normalize(grasp_matrix[:3, 2], np.array([0.0, 0.0, 1.0], dtype=np.float64))
    jaw_axis = normalize(grasp_matrix[:3, 0], np.array([1.0, 0.0, 0.0], dtype=np.float64))
    side_axis = normalize(np.cross(approach_axis, jaw_axis), grasp_matrix[:3, 1])
    compat_rotation = np.stack([approach_axis, jaw_axis, side_axis], axis=1)

    raw_translation = grasp_matrix[:3, 3].astype(np.float64)
    translation = raw_translation + approach_axis * float(export_tcp_offset)
    world_pose = transform_pose_to_world(camera, translation, compat_rotation)
    raw_world_pose = transform_pose_to_world(camera, raw_translation, grasp_matrix[:3, :3])
    payload = {
        "rank": int(rank),
        "source_rank": int(rank),
        "score": float(score),
        "width": float(width),
        "depth": float(depth),
        "translation_camera": translation.astype(float).tolist(),
        "rotation_matrix_camera": compat_rotation.astype(float).tolist(),
        "pose_world": world_pose,
        "translation_world": world_pose["translation"] if world_pose else None,
        "rotation_matrix_world": world_pose["rotation_matrix"] if world_pose else None,
        "generator": "graspgen",
        "pose_type": "graspgen_gripper_pose_as_graspnet_compatible",
        "collision_free": collision_free,
        "graspgen": {
            "raw_translation_camera": raw_translation.astype(float).tolist(),
            "raw_rotation_matrix_camera": grasp_matrix[:3, :3].astype(float).tolist(),
            "raw_pose_world": raw_world_pose,
            "export_tcp_offset_m": float(export_tcp_offset),
            "exported_translation_semantics": "raw_base_link_plus_local_positive_z_offset",
            "frame_note": (
                "GraspGen returns a gripper pose in its own convention: local +Z is "
                "approach and local +X is finger closing. The raw pose is a gripper "
                "base-link transform, so the exported translation is shifted along "
                "local +Z to the Robotiq fingertip/TCP region before feeding the Task E "
                "Piper motion contract. Rotation columns are remapped so rotation[:,0] "
                "is approach, matching the AnyGrasp/GraspNet-compatible path."
            ),
        },
    }
    return payload


def default_robotiq_visual_polyline() -> np.ndarray:
    # Matches GraspGen's released Robotiq-2F-140 visualization control line.
    return np.asarray(
        [
            [0.06801729, 0.0, 0.195],
            [0.06801729, 0.0, 0.0975],
            [0.0, 0.0, 0.0975],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0975],
            [-0.06801729, 0.0, 0.0975],
            [-0.06801729, 0.0, 0.195],
        ],
        dtype=np.float64,
    )


def raw_gripper_polyline_camera(grasp: dict) -> np.ndarray | None:
    raw = grasp.get("graspgen") if isinstance(grasp, dict) else None
    if not isinstance(raw, dict):
        return None
    try:
        rotation = np.asarray(raw["raw_rotation_matrix_camera"], dtype=np.float64)
        translation = np.asarray(raw["raw_translation_camera"], dtype=np.float64)
    except Exception:
        return None
    if rotation.shape != (3, 3) or translation.shape != (3,):
        return None
    local = default_robotiq_visual_polyline()
    return local @ rotation.T + translation[None, :]


def save_overlay(
    path: Path,
    rgb_path: Path,
    mask_path: Path,
    intrinsic: np.ndarray,
    grasps: list[dict],
    status: str,
) -> None:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    mask = load_mask(mask_path, rgb.shape[:2])
    base = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    layer = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    layer[mask] = (255, 204, 0, 78)
    base = Image.alpha_composite(base, Image.fromarray(layer, mode="RGBA"))
    draw = ImageDraw.Draw(base)
    draw.rectangle((0, 0, base.width, 38), fill=(0, 0, 0, 180))
    draw.text((8, 10), f"GraspGen target grasps | status={status}", fill=(255, 255, 255, 255))
    palette = [(255, 64, 64), (64, 200, 255), (80, 220, 120), (255, 180, 64), (210, 120, 255)]
    for idx, grasp in enumerate(grasps):
        origin = np.asarray(grasp["translation_camera"], dtype=np.float64)
        rotation = np.asarray(grasp["rotation_matrix_camera"], dtype=np.float64)
        uv = project_camera_point(intrinsic, origin)
        polyline = raw_gripper_polyline_camera(grasp)
        if polyline is not None:
            poly_color = palette[idx % len(palette)]
            projected = [project_camera_point(intrinsic, point) for point in polyline]
            for start, end in zip(projected[:-1], projected[1:]):
                if start is not None and end is not None:
                    draw.line((*start, *end), fill=poly_color, width=3 if idx == 0 else 2)
        if uv is None:
            continue
        color = palette[idx % len(palette)]
        radius = 7 if idx == 0 else 5
        draw.ellipse((uv[0] - radius, uv[1] - radius, uv[0] + radius, uv[1] + radius), outline=color, width=3)
        draw.text((uv[0] + 9, uv[1] - 9), f"{idx + 1}:{float(grasp.get('score', 0.0)):.3f}", fill=color)
        for axis_idx, axis_color in [(0, (255, 80, 80)), (1, (80, 220, 120))]:
            end = project_camera_point(intrinsic, origin + rotation[:, axis_idx] * 0.045)
            if end is not None:
                draw.line((*uv, *end), fill=axis_color, width=2)
    base.convert("RGB").save(path)


def checkpoints_exist(gripper_config: Path) -> tuple[bool, list[str]]:
    if not gripper_config.exists():
        return False, [str(gripper_config)]
    try:
        import yaml

        data = yaml.safe_load(gripper_config.read_text(encoding="utf-8"))
        missing = []
        for section, key in [("eval", "checkpoint"), ("discriminator", "checkpoint")]:
            rel = str(data.get(section, {}).get(key, "")).strip()
            if not rel:
                missing.append(f"{section}.{key}")
                continue
            path = gripper_config.parent / rel
            if not path.exists():
                missing.append(str(path))
        return not missing, missing
    except Exception as exc:
        return False, [f"{gripper_config}: {type(exc).__name__}: {exc}"]


def run_graspgen(
    args: argparse.Namespace,
    scene_points: np.ndarray,
    target_points: np.ndarray,
    camera: dict,
) -> dict:
    if not args.graspgen_root.exists():
        return {"status": "missing_requirements", "reason": f"missing root: {args.graspgen_root}"}
    ready, missing = checkpoints_exist(args.gripper_config)
    if not ready:
        return {
            "status": "missing_checkpoint",
            "gripper_config": str(args.gripper_config),
            "missing": missing,
        }

    sys.path.insert(0, str(args.graspgen_root))
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    try:
        import torch
        from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
        from grasp_gen.robot import get_gripper_info
    except BaseException as exc:
        return {"status": "import_failed", "error": repr(exc), "traceback_tail": traceback.format_exc(limit=8).splitlines()[-24:]}

    try:
        cfg = load_grasp_cfg(str(args.gripper_config))
        if bool(args.require_pointnet):
            diffusion_backbone = str(cfg.diffusion.obs_backbone)
            discriminator_backbone = str(cfg.discriminator.obs_backbone)
            if diffusion_backbone != "pointnet" or discriminator_backbone != "pointnet":
                return {
                    "status": "incompatible_config",
                    "reason": "PointNet++ backbone required for this CUDA 12.8/Blackwell setup.",
                    "diffusion_obs_backbone": diffusion_backbone,
                    "discriminator_obs_backbone": discriminator_backbone,
                }
        sampler = GraspGenSampler(cfg)
        depth = graspgen_depth_from_cfg(cfg, args.default_depth)
        try:
            gripper_info = get_gripper_info(str(cfg.data.gripper_name))
            width = float(getattr(gripper_info, "width", args.default_width))
        except Exception:
            width = float(args.default_width)

        grasps_t, scores_t = GraspGenSampler.run_inference(
            target_points.astype(np.float32),
            sampler,
            grasp_threshold=float(args.grasp_threshold),
            num_grasps=int(args.num_grasps),
            topk_num_grasps=max(1, int(args.save_top_grasps) * 4),
            min_grasps=int(args.min_grasps),
            max_tries=int(args.max_tries),
            remove_outliers=bool(args.remove_outliers),
        )
        if len(grasps_t) == 0:
            return {
                "status": "no_grasp",
                "gripper_config": str(args.gripper_config),
                "target_point_count": int(len(target_points)),
            }
        grasps = grasps_t.detach().cpu().numpy()
        scores = scores_t.detach().cpu().numpy()
        order = np.argsort(scores)[::-1]
        grasps = grasps[order]
        scores = scores[order]

        collision_mask = None
        collision_info = {"enabled": False}
        if bool(args.filter_collisions):
            obstacles, obstacle_info = obstacle_cloud_without_target(
                scene_points,
                target_points,
                args.target_exclusion_radius,
                args.collision_max_scene_points,
            )
            collision_info = {"enabled": True, "obstacle_cloud": obstacle_info}
            if len(obstacles) > 0:
                try:
                    from grasp_gen.robot import get_gripper_info
                    from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps

                    gripper_mesh = get_gripper_info(str(cfg.data.gripper_name)).collision_mesh
                    collision_mask = filter_colliding_grasps(
                        scene_pc=obstacles.astype(np.float32),
                        grasp_poses=grasps.astype(np.float32),
                        gripper_collision_mesh=gripper_mesh,
                        collision_threshold=float(args.collision_threshold),
                    )
                    collision_info["collision_free_count"] = int(np.count_nonzero(collision_mask))
                    collision_info["checked_grasp_count"] = int(len(collision_mask))
                except BaseException as exc:
                    collision_info["status"] = "failed"
                    collision_info["error"] = repr(exc)
                    collision_info["traceback_tail"] = traceback.format_exc(limit=6).splitlines()[-16:]

        if collision_mask is not None and np.any(collision_mask):
            keep = np.where(collision_mask)[0]
            grasps = grasps[keep]
            scores = scores[keep]
            kept_collision_flags: list[bool | None] = [True] * len(grasps)
        else:
            kept_collision_flags = [None if collision_mask is None else False] * len(grasps)
        grasp_count_after_collision = int(len(grasps)) if collision_mask is not None else None

        top_payloads = []
        for rank, (grasp, score, collision_free) in enumerate(
            zip(grasps[: max(1, int(args.save_top_grasps))], scores[: max(1, int(args.save_top_grasps))], kept_collision_flags),
            start=1,
        ):
            top_payloads.append(
                serialize_graspgen_grasp(
                    grasp,
                    float(score),
                    camera,
                    rank,
                    width=width,
                    depth=depth,
                    export_tcp_offset=float(args.export_tcp_offset),
                    collision_free=collision_free,
                )
            )
        if not top_payloads:
            return {
                "status": "no_grasp_after_collision_filter",
                "raw_count": int(len(order)),
                "collision_filter": collision_info,
            }
        best = top_payloads[0]
        return {
            "status": "ok",
            "generator": "graspgen",
            "gripper_config": str(args.gripper_config),
            "gripper_name": str(cfg.data.gripper_name),
            "raw_count": int(len(order)),
            "num_grasps_after_target_filter": int(len(order)),
            "num_grasps_after_collision_filter": grasp_count_after_collision,
            "collision_filter": collision_info,
            "best": best,
            "top_grasps": top_payloads,
            "final_grasp_pose": {
                "frame": "world",
                "pose_type": "graspgen_gripper_pose_as_graspnet_compatible",
                "approach_axis": "rotation_matrix[:,0]",
                "pregrasp_direction": "-rotation_matrix[:,0]",
                "translation": best["translation_world"],
                "rotation_matrix": best["rotation_matrix_world"],
                "score": best["score"],
                "width": best["width"],
                "depth": best["depth"],
                "source_rank": best["rank"],
                "generator": "graspgen",
                "graspgen": best["graspgen"],
            },
        }
    except BaseException as exc:
        return {
            "status": "load_or_inference_failed",
            "error": repr(exc),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback_tail": traceback.format_exc(limit=10).splitlines()[-30:],
            "gripper_config": str(args.gripper_config),
        }
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    camera_payload = load_json(args.camera_json)
    camera = camera_payload.get("camera", camera_payload)
    intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float32)

    scene_batches = []
    scene_color_batches = []
    target_batches = []
    target_color_batches = []
    view_records = []

    scene, scene_colors, target, target_colors, stats = view_points(
        args.rgb,
        args.depth_npy,
        args.mask,
        camera,
        args.max_depth,
        args.scene_cloud_stride,
    )
    stats["name"] = "primary"
    stats["camera_json"] = str(args.camera_json.resolve())
    stats["frame"] = "primary_camera"
    scene_batches.append(scene)
    scene_color_batches.append(scene_colors)
    target_batches.append(target)
    target_color_batches.append(target_colors)
    view_records.append(stats)

    for extra_idx, spec in enumerate(args.extra_view, start=2):
        extra_rgb, extra_depth, extra_mask, extra_camera_json = parse_extra_view(spec)
        extra_camera_payload = load_json(extra_camera_json)
        extra_camera = extra_camera_payload.get("camera", extra_camera_payload)
        e_scene, e_scene_colors, e_target, e_target_colors, e_stats = view_points(
            extra_rgb,
            extra_depth,
            extra_mask,
            extra_camera,
            args.max_depth,
            args.scene_cloud_stride,
        )
        e_stats["name"] = f"extra_{extra_idx:02d}"
        e_stats["camera_json"] = str(extra_camera_json.resolve())
        e_stats["frame"] = "reprojected_to_primary_camera"
        if len(e_scene):
            e_scene_primary = transform_points_from_world(transform_points_to_world(e_scene, extra_camera), camera).astype(np.float32)
            keep = np.isfinite(e_scene_primary).all(axis=1) & (e_scene_primary[:, 2] > 0.0)
            e_stats["scene_point_count_after_primary_reprojection"] = int(np.count_nonzero(keep))
            if keep.any():
                scene_batches.append(e_scene_primary[keep])
                scene_color_batches.append(e_scene_colors[keep])
        if len(e_target):
            e_target_primary = transform_points_from_world(transform_points_to_world(e_target, extra_camera), camera).astype(np.float32)
            keep = np.isfinite(e_target_primary).all(axis=1) & (e_target_primary[:, 2] > 0.0)
            e_stats["target_point_count_after_primary_reprojection"] = int(np.count_nonzero(keep))
            if keep.any():
                target_batches.append(e_target_primary[keep])
                target_color_batches.append(e_target_colors[keep])
        view_records.append(e_stats)

    scene_points = np.concatenate(scene_batches, axis=0).astype(np.float32)
    scene_colors = np.concatenate(scene_color_batches, axis=0).astype(np.float32)
    target_points = np.concatenate(target_batches, axis=0).astype(np.float32)
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
    object_center_world = None
    if len(args.overlay_object_center_world) >= 3:
        object_center_world = np.asarray(args.overlay_object_center_world[:3], dtype=np.float64)
    target_points, target_colors, symmetry_completion = complete_symmetric_bottle_cloud_camera(
        target_points,
        target_colors,
        camera,
        object_center_world,
        args,
    )
    scene_points, scene_colors, scene_downsample = deterministic_downsample(
        scene_points,
        scene_colors,
        args.scene_cloud_max_points,
    )
    target_points_for_model, target_downsample = downsample_points_only(target_points, args.target_cloud_max_points)

    np.save(args.output / "graspgen_input_cloud.npy", scene_points)
    np.save(args.output / "graspgen_input_cloud_colors.npy", scene_colors)
    np.save(args.output / "anygrasp_input_cloud.npy", scene_points)
    np.save(args.output / "anygrasp_input_cloud_colors.npy", scene_colors)
    np.save(args.output / "masked_cloud.npy", target_points)
    np.save(args.output / "masked_cloud_colors.npy", target_colors)
    np.save(args.output / "graspgen_target_model_cloud.npy", target_points_for_model)
    write_ply(args.output / "graspgen_input_cloud.ply", scene_points, scene_colors)
    write_ply(args.output / "anygrasp_input_cloud.ply", scene_points, scene_colors)
    write_ply(args.output / "masked_cloud.ply", target_points, target_colors)
    write_ply(args.output / "graspgen_target_model_cloud.ply", target_points_for_model, np.zeros_like(target_points_for_model))
    np.savez(
        args.output / "graspgen_input.npz",
        xyz=scene_points,
        xyz_color=scene_colors,
        target_xyz=target_points,
        target_xyz_color=target_colors,
        target_xyz_model=target_points_for_model,
        K=intrinsic,
    )

    if len(target_points_for_model) < 50:
        result = {
            "status": "insufficient_points",
            "scene_point_count": int(len(scene_points)),
            "target_point_count": int(len(target_points)),
            "target_model_point_count": int(len(target_points_for_model)),
        }
    else:
        result = run_graspgen(args, scene_points, target_points_for_model, camera)

    top_grasps = result.get("top_grasps", []) if isinstance(result, dict) else []
    marker = None
    best = result.get("best") if isinstance(result, dict) else None
    if isinstance(best, dict) and best.get("translation_camera") is not None:
        marker = np.asarray(best["translation_camera"], dtype=np.float32)
    status = result.get("status", "unknown") if isinstance(result, dict) else "unknown"
    save_cloud_views(
        args.output / "anygrasp_result.png",
        scene_points,
        scene_colors,
        status,
        marker=marker,
        title="Fused RGB-D input cloud | GraspGen",
    )
    save_cloud_views(
        args.output / "graspgen_result.png",
        scene_points,
        scene_colors,
        status,
        marker=marker,
        title="Fused RGB-D input cloud | GraspGen",
    )
    save_cloud_views(
        args.output / "target_cloud.png",
        target_points,
        target_colors,
        "target_mask",
        marker=marker,
        title="Segmented target cloud | GraspGen",
    )
    (args.output / "top_grasps.json").write_text(json.dumps(top_grasps, indent=2), encoding="utf-8")
    save_overlay(
        args.output / "top_grasps_overlay.png",
        args.rgb,
        args.mask,
        intrinsic,
        top_grasps if isinstance(top_grasps, list) else [],
        status,
    )
    save_overlay(
        args.output / "selected_grasp_overlay.png",
        args.rgb,
        args.mask,
        intrinsic,
        top_grasps[:1] if isinstance(top_grasps, list) else [],
        status,
    )

    summary = {
        "grasp_generator": "graspgen",
        "rgb": str(args.rgb.resolve()),
        "depth_npy": str(args.depth_npy.resolve()),
        "mask": str(args.mask.resolve()),
        "camera_json": str(args.camera_json.resolve()),
        "point_cloud_frame": "primary_camera",
        "point_count": int(len(target_points)),
        "target_point_count": int(len(target_points)),
        "target_model_point_count": int(len(target_points_for_model)),
        "point_cloud_npy": "masked_cloud.npy",
        "point_cloud_colors_npy": "masked_cloud_colors.npy",
        "point_cloud_ply": "masked_cloud.ply",
        "graspgen_input_point_count": int(len(scene_points)),
        "graspgen_input_cloud_npy": "graspgen_input_cloud.npy",
        "graspgen_input_cloud_colors_npy": "graspgen_input_cloud_colors.npy",
        "graspgen_input_cloud_ply": "graspgen_input_cloud.ply",
        "graspgen_input_npz": "graspgen_input.npz",
        "anygrasp_input_point_count": int(len(scene_points)),
        "anygrasp_input_cloud_npy": "anygrasp_input_cloud.npy",
        "anygrasp_input_cloud_colors_npy": "anygrasp_input_cloud_colors.npy",
        "anygrasp_input_cloud_ply": "anygrasp_input_cloud.ply",
        "scene_cloud_downsample": scene_downsample,
        "target_cloud_downsample": target_downsample,
        "target_outlier_filter": target_filter,
        "symmetry_completion": symmetry_completion,
        "fused_view_count": int(len(view_records)),
        "views": view_records,
        "scene_views": view_records,
        "visualization": "anygrasp_result.png",
        "graspgen_visualization": "graspgen_result.png",
        "target_visualization": "target_cloud.png",
        "top_grasps_json": "top_grasps.json",
        "top_grasps_overlay": "top_grasps_overlay.png",
        "top_grasps_overlay_geometry": {
            "tool_transform": "graspnet_to_piper_z",
            "gripper_base_offset_m": float(args.export_tcp_offset),
            "gripper_base_offset_mode": "graspgen_raw_base_to_exported_tcp_along_local_positive_z",
            "object_center_world": list(args.overlay_object_center_world[:3]),
            "finger_length_m": 0.0765,
            "palm_width_m": 0.085,
            "show_full_tool": True,
            "note": "GraspGen top_grasps_overlay.png draws the raw Robotiq 2F-140 control-line shape plus the exported TCP/contact marker.",
        },
        "selected_grasp_overlay": "selected_grasp_overlay.png",
        "anygrasp": result,
        "graspgen": result,
    }
    (args.output / "anygrasp_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "graspgen_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    final_pose = result.get("final_grasp_pose") if isinstance(result, dict) else None
    if isinstance(final_pose, dict) and result.get("status") == "ok":
        (args.output / "final_grasp_pose.json").write_text(json.dumps(final_pose, indent=2), encoding="utf-8")
    else:
        (args.output / "final_grasp_pose_error.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[INFO] Saved {args.output / 'graspgen_result.json'}")


if __name__ == "__main__":
    main()
