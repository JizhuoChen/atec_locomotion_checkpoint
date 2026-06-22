#!/usr/bin/env python3
"""Run Contact-GraspNet from the same RGB-D/mask views used by AnyGrasp."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTACT_GRASPNET_ROOT = REPO_ROOT / "third_party/contact_graspnet"
DEFAULT_CKPT_DIR = DEFAULT_CONTACT_GRASPNET_ROOT / "checkpoints/scene_test_2048_bs3_hor_sigma_001"


def add_bool_option(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str = "") -> None:
    parser.add_argument(name, dest=name.lstrip("-").replace("-", "_"), action="store_true", help=help_text)
    parser.add_argument(
        f"--no-{name.lstrip('-')}",
        dest=name.lstrip("-").replace("-", "_"),
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(**{name.lstrip("-").replace("-", "_"): default})


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
    parser.add_argument("--contact-graspnet-root", type=Path, default=DEFAULT_CONTACT_GRASPNET_ROOT)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--scene-cloud-stride", type=int, default=1)
    parser.add_argument("--scene-cloud-max-points", type=int, default=180000)
    parser.add_argument("--save-top-grasps", type=int, default=12)
    parser.add_argument("--forward-passes", type=int, default=1)
    add_bool_option(
        parser,
        "--filter-target-outliers",
        default=True,
        help_text="Keep only the largest connected voxel component of the fused target cloud.",
    )
    parser.add_argument("--target-filter-voxel-size", type=float, default=0.008)
    parser.add_argument("--target-filter-min-points", type=int, default=50)
    add_bool_option(parser, "--local-regions", default=True)
    add_bool_option(parser, "--filter-grasps", default=True)
    parser.add_argument("--target-segmap-id", type=int, default=1)
    parser.add_argument("--z-range", type=parse_float_list, default=[0.05, 2.0])
    parser.add_argument(
        "--gripper-depth",
        type=float,
        default=0.1034,
        help="Contact-GraspNet Panda gripper depth used only for metadata/debug geometry.",
    )
    parser.add_argument(
        "--overlay-object-center-world",
        type=parse_float_list,
        default=[],
        help="Accepted for command compatibility with AnyGrasp wrapper.",
    )
    args = parser.parse_args()
    args.contact_graspnet_root = args.contact_graspnet_root.expanduser().resolve()
    args.ckpt_dir = args.ckpt_dir.expanduser().resolve()
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


def transform_pose_to_world(camera: dict, translation: np.ndarray, rotation: np.ndarray) -> dict | None:
    if camera.get("pos_w") is None or camera.get("quat_w_ros") is None:
        return None
    r_wc = quat_wxyz_to_matrix(np.asarray(camera["quat_w_ros"], dtype=np.float64))
    t_wc = np.asarray(camera["pos_w"], dtype=np.float64)
    return {
        "translation": (r_wc @ np.asarray(translation, dtype=np.float64) + t_wc).tolist(),
        "rotation_matrix": (r_wc @ np.asarray(rotation, dtype=np.float64)).tolist(),
    }


def transform_points_to_world(points_cam: np.ndarray, camera: dict) -> np.ndarray:
    r_wc = quat_wxyz_to_matrix(np.asarray(camera["quat_w_ros"], dtype=np.float64))
    t_wc = np.asarray(camera["pos_w"], dtype=np.float64)
    return np.asarray(points_cam, dtype=np.float64) @ r_wc.T + t_wc


def transform_points_from_world(points_w: np.ndarray, camera: dict) -> np.ndarray:
    r_wc = quat_wxyz_to_matrix(np.asarray(camera["quat_w_ros"], dtype=np.float64))
    t_wc = np.asarray(camera["pos_w"], dtype=np.float64)
    return (np.asarray(points_w, dtype=np.float64) - t_wc) @ r_wc


def parse_extra_view(value: str) -> tuple[Path, Path, Path, Path]:
    parts = [Path(part.strip()).expanduser() for part in value.split(",")]
    if len(parts) != 4 or any(str(part) == "." for part in parts):
        raise ValueError("--extra-view expects RGB,DEPTH_NPY,MASK,CAMERA_JSON")
    return parts[0], parts[1], parts[2], parts[3]


def view_points(
    rgb_path: Path,
    depth_path: Path,
    mask_path: Path,
    camera: dict,
    max_depth: float,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float32)
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
    depth = to_depth_array(depth_path)
    mask = load_mask(mask_path, depth.shape)
    points_organized = backproject(depth, intrinsic)
    valid_scene = np.isfinite(depth) & (depth > 0.0) & (depth < max_depth)
    stride = max(1, int(stride))
    if stride > 1:
        stride_mask = np.zeros_like(valid_scene, dtype=bool)
        stride_mask[::stride, ::stride] = True
        valid_scene &= stride_mask
    valid_target = mask & np.isfinite(depth) & (depth > 0.0) & (depth < max_depth)
    scene_points = points_organized[valid_scene].astype(np.float32)
    scene_colors = rgb[valid_scene].astype(np.float32)
    target_points = points_organized[valid_target].astype(np.float32)
    target_colors = rgb[valid_target].astype(np.float32)
    ys, xs = np.where(valid_target)
    stats = {
        "rgb": str(rgb_path.resolve()),
        "depth_npy": str(depth_path.resolve()),
        "mask": str(mask_path.resolve()),
        "point_count_scene": int(scene_points.shape[0]),
        "point_count_target": int(target_points.shape[0]),
        "target_bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if xs.size else None,
        "stride": int(stride),
    }
    return scene_points, scene_colors, target_points, target_colors, stats


def deterministic_downsample(points: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray, dict]:
    max_points = int(max_points)
    if max_points <= 0 or len(points) <= max_points:
        return points, colors, {
            "applied": False,
            "input_point_count": int(len(points)),
            "output_point_count": int(len(points)),
            "max_points": max_points,
        }
    indices = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
    return points[indices], colors[indices], {
        "applied": True,
        "method": "linspace_deterministic",
        "input_point_count": int(len(points)),
        "output_point_count": int(max_points),
        "max_points": max_points,
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


def normalize(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        return fallback.astype(np.float64)
    return np.asarray(value, dtype=np.float64) / norm


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, rgb):
            f.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def project_camera_point(intrinsic: np.ndarray, point: np.ndarray) -> tuple[float, float] | None:
    point = np.asarray(point, dtype=np.float64)
    if point.shape != (3,) or point[2] <= 1e-6:
        return None
    return (
        float(intrinsic[0, 0] * point[0] / point[2] + intrinsic[0, 2]),
        float(intrinsic[1, 1] * point[1] / point[2] + intrinsic[1, 2]),
    )


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
    draw.rectangle((0, 0, base.width, 36), fill=(0, 0, 0, 180))
    draw.text((8, 10), f"Contact-GraspNet target-filtered grasps | status={status}", fill=(255, 255, 255, 255))
    palette = [(255, 64, 64), (64, 200, 255), (80, 220, 120), (255, 180, 64), (210, 120, 255)]
    for idx, grasp in enumerate(grasps):
        uv = project_camera_point(intrinsic, np.asarray(grasp["translation_camera"], dtype=np.float64))
        if uv is None:
            continue
        color = palette[idx % len(palette)]
        radius = 7 if idx == 0 else 5
        draw.ellipse((uv[0] - radius, uv[1] - radius, uv[0] + radius, uv[1] + radius), outline=color, width=3)
        draw.text((uv[0] + 9, uv[1] - 9), f"{idx + 1}:{float(grasp.get('score', 0.0)):.3f}", fill=color)
    base.convert("RGB").save(path)


@contextlib.contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def checkpoint_ready(path: Path) -> bool:
    return path.exists() and (
        (path / "checkpoint").exists()
        or any(path.glob("*.index"))
        or any(path.glob("model.ckpt*.index"))
    )


def serialize_contact_grasp(
    grasp_matrix: np.ndarray,
    score: float,
    contact_point: np.ndarray,
    opening: float,
    camera: dict,
    rank: int,
) -> dict:
    grasp_matrix = np.asarray(grasp_matrix, dtype=np.float64)
    contact_point = np.asarray(contact_point, dtype=np.float64)
    jaw_axis = normalize(grasp_matrix[:3, 0], np.array([0.0, 1.0, 0.0], dtype=np.float64))
    approach_axis = normalize(grasp_matrix[:3, 2], np.array([0.0, 0.0, -1.0], dtype=np.float64))
    side_axis = normalize(np.cross(approach_axis, jaw_axis), grasp_matrix[:3, 1])
    compat_rotation = np.stack([approach_axis, jaw_axis, side_axis], axis=1)
    world_pose = transform_pose_to_world(camera, contact_point, compat_rotation)
    gripper_base_world = transform_pose_to_world(camera, grasp_matrix[:3, 3], grasp_matrix[:3, :3])
    payload = {
        "rank": int(rank),
        "source_rank": int(rank),
        "score": float(score),
        "width": float(opening) if np.isfinite(opening) else 0.08,
        "depth": 0.1034,
        "translation_camera": contact_point.astype(float).tolist(),
        "rotation_matrix_camera": compat_rotation.astype(float).tolist(),
        "pose_world": world_pose,
        "translation_world": world_pose["translation"] if world_pose else None,
        "rotation_matrix_world": world_pose["rotation_matrix"] if world_pose else None,
        "generator": "contact_graspnet",
        "pose_type": "contact_graspnet_contact_as_graspnet_compatible",
        "contact_graspnet": {
            "raw_gripper_base_camera": grasp_matrix[:3, 3].astype(float).tolist(),
            "raw_rotation_matrix_camera": grasp_matrix[:3, :3].astype(float).tolist(),
            "raw_gripper_base_world": gripper_base_world,
            "contact_point_camera": contact_point.astype(float).tolist(),
            "opening_m": float(opening) if np.isfinite(opening) else None,
            "frame_note": (
                "Contact-GraspNet returns a Panda gripper-base pose whose local z column is "
                "the approach direction. For pipeline compatibility this record exposes the "
                "predicted contact point as translation and remaps columns so rotation[:,0] "
                "is the approach axis, matching the existing graspnet_to_piper_z path."
            ),
        },
    }
    return payload


def run_contact_graspnet(
    args: argparse.Namespace,
    scene_points: np.ndarray,
    scene_colors: np.ndarray,
    target_points: np.ndarray,
    camera: dict,
) -> dict:
    if not args.contact_graspnet_root.exists():
        return {"status": "missing_requirements", "reason": f"missing root: {args.contact_graspnet_root}"}
    if not checkpoint_ready(args.ckpt_dir):
        return {
            "status": "missing_checkpoint",
            "ckpt_dir": str(args.ckpt_dir),
            "reason": "Download Contact-GraspNet trained models into this checkpoint directory.",
        }

    root = args.contact_graspnet_root
    sys.path.append(str(root))
    sys.path.insert(0, str(root / "pointnet2/tf_ops/grouping"))
    sys.path.insert(0, str(root / "pointnet2/utils"))
    sys.path.insert(0, str(root / "contact_graspnet"))

    try:
        import tensorflow.compat.v1 as tf  # type: ignore

        tf.disable_eager_execution()
        from contact_grasp_estimator import GraspEstimator  # type: ignore
        import config_utils  # type: ignore
    except BaseException as exc:
        return {"status": "import_failed", "error": repr(exc), "ckpt_dir": str(args.ckpt_dir)}

    try:
        with working_directory(root):
            global_config = config_utils.load_config(
                str(args.ckpt_dir),
                batch_size=int(args.forward_passes),
                arg_configs=[],
            )
            estimator = GraspEstimator(global_config)
            estimator.build_network()
            saver = tf.train.Saver(save_relative_paths=True)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            with tf.Session(config=config) as sess:
                estimator.load_weights(sess, saver, str(args.ckpt_dir), mode="test")
                pred_grasps, scores, contacts, openings = estimator.predict_scene_grasps(
                    sess,
                    scene_points.astype(np.float32),
                    pc_segments={int(args.target_segmap_id): target_points.astype(np.float32)},
                    local_regions=bool(args.local_regions),
                    filter_grasps=bool(args.filter_grasps),
                    forward_passes=int(args.forward_passes),
                )
    except SystemExit as exc:
        return {
            "status": "load_or_inference_failed",
            "error": repr(exc),
            "error_type": type(exc).__name__,
            "ckpt_dir": str(args.ckpt_dir),
        }
    except BaseException as exc:
        return {
            "status": "load_or_inference_failed",
            "error": repr(exc),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback_tail": traceback.format_exc(limit=8).splitlines()[-24:],
            "ckpt_dir": str(args.ckpt_dir),
        }

    key = int(args.target_segmap_id) if int(args.target_segmap_id) in pred_grasps else -1
    grasps = np.asarray(pred_grasps.get(key, []), dtype=np.float64)
    grasp_scores = np.asarray(scores.get(key, []), dtype=np.float64)
    contact_points = np.asarray(contacts.get(key, []), dtype=np.float64)
    gripper_openings = np.asarray(openings.get(key, []), dtype=np.float64)
    if grasps.size == 0 or grasp_scores.size == 0:
        return {
            "status": "no_grasp",
            "ckpt_dir": str(args.ckpt_dir),
            "segment_key": key,
            "raw_count": int(len(grasps)),
        }

    order = np.argsort(grasp_scores)[::-1]
    top_payloads = []
    for out_rank, grasp_idx in enumerate(order[: max(1, int(args.save_top_grasps))], start=1):
        opening = gripper_openings[grasp_idx] if gripper_openings.size else 0.08
        top_payloads.append(
            serialize_contact_grasp(
                grasps[grasp_idx],
                float(grasp_scores[grasp_idx]),
                contact_points[grasp_idx],
                float(opening),
                camera,
                out_rank,
            )
        )
    best = top_payloads[0]
    return {
        "status": "ok",
        "generator": "contact_graspnet",
        "ckpt_dir": str(args.ckpt_dir),
        "segment_key": key,
        "raw_count": int(len(grasps)),
        "num_grasps_after_target_filter": int(len(grasps)),
        "best": best,
        "top_grasps": top_payloads,
        "final_grasp_pose": {
            "frame": "world",
            "pose_type": "contact_graspnet_contact_as_graspnet_compatible",
            "approach_axis": "rotation_matrix[:,0]",
            "pregrasp_direction": "-rotation_matrix[:,0]",
            "translation": best["translation_world"],
            "rotation_matrix": best["rotation_matrix_world"],
            "score": best["score"],
            "width": best["width"],
            "depth": best["depth"],
            "source_rank": best["rank"],
            "generator": "contact_graspnet",
            "contact_graspnet": best["contact_graspnet"],
        },
    }


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
    scene_points, scene_colors, downsample = deterministic_downsample(
        scene_points,
        scene_colors,
        args.scene_cloud_max_points,
    )

    np.save(args.output / "anygrasp_input_cloud.npy", scene_points)
    np.save(args.output / "anygrasp_input_cloud_colors.npy", scene_colors)
    np.save(args.output / "masked_cloud.npy", target_points)
    np.save(args.output / "masked_cloud_colors.npy", target_colors)
    write_ply(args.output / "anygrasp_input_cloud.ply", scene_points, scene_colors)
    write_ply(args.output / "contact_graspnet_input_cloud.ply", scene_points, scene_colors)
    write_ply(args.output / "masked_cloud.ply", target_points, target_colors)
    np.savez(
        args.output / "contact_graspnet_input.npz",
        xyz=scene_points,
        xyz_color=scene_colors,
        target_xyz=target_points,
        target_xyz_color=target_colors,
        K=intrinsic,
    )

    if len(scene_points) < 50 or len(target_points) < 10:
        result = {
            "status": "insufficient_points",
            "scene_point_count": int(len(scene_points)),
            "target_point_count": int(len(target_points)),
        }
    else:
        result = run_contact_graspnet(args, scene_points, scene_colors, target_points, camera)

    top_grasps = result.get("top_grasps", []) if isinstance(result, dict) else []
    (args.output / "top_grasps.json").write_text(json.dumps(top_grasps, indent=2), encoding="utf-8")
    save_overlay(
        args.output / "top_grasps_overlay.png",
        args.rgb,
        args.mask,
        intrinsic,
        top_grasps if isinstance(top_grasps, list) else [],
        result.get("status", "unknown") if isinstance(result, dict) else "unknown",
    )

    summary = {
        "grasp_generator": "contact_graspnet",
        "rgb": str(args.rgb.resolve()),
        "depth_npy": str(args.depth_npy.resolve()),
        "mask": str(args.mask.resolve()),
        "camera_json": str(args.camera_json.resolve()),
        "point_cloud_frame": "primary_camera",
        "point_count": int(len(target_points)),
        "target_point_count": int(len(target_points)),
        "point_cloud_npy": "masked_cloud.npy",
        "point_cloud_colors_npy": "masked_cloud_colors.npy",
        "point_cloud_ply": "masked_cloud.ply",
        "anygrasp_input_point_count": int(len(scene_points)),
        "anygrasp_input_cloud_npy": "anygrasp_input_cloud.npy",
        "anygrasp_input_cloud_colors_npy": "anygrasp_input_cloud_colors.npy",
        "anygrasp_input_cloud_ply": "anygrasp_input_cloud.ply",
        "contact_graspnet_input_npz": "contact_graspnet_input.npz",
        "scene_cloud_downsample": downsample,
        "target_outlier_filter": target_filter,
        "fused_view_count": int(len(view_records)),
        "views": view_records,
        "top_grasps_json": "top_grasps.json",
        "top_grasps_overlay": "top_grasps_overlay.png",
        "anygrasp": result,
        "contact_graspnet": result,
    }
    (args.output / "anygrasp_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "contact_graspnet_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    final_pose = result.get("final_grasp_pose") if isinstance(result, dict) else None
    if isinstance(final_pose, dict) and result.get("status") == "ok":
        (args.output / "final_grasp_pose.json").write_text(json.dumps(final_pose, indent=2), encoding="utf-8")
    else:
        (args.output / "final_grasp_pose_error.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[INFO] Saved {args.output / 'contact_graspnet_result.json'}")


if __name__ == "__main__":
    main()
