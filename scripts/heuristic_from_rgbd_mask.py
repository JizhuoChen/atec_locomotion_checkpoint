#!/usr/bin/env python3
"""Create a Task E top-down heuristic grasp from segmented RGB-D views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from contact_graspnet_from_rgbd_mask import (
    deterministic_downsample,
    largest_voxel_component_filter,
    load_json,
    parse_extra_view,
    project_camera_point,
    transform_points_from_world,
    transform_points_to_world,
    view_points,
    write_ply,
)
from task_e_full_baseline_request import (
    OBJECTS,
    TABLE_TOP_Z,
    quat_wxyz_to_matrix,
    task_e_grasp_quat,
)


PIPER_FINGER_LENGTH_M = 0.0765
PIPER_FINGER_WIDTH_M = 0.0265
PIPER_FINGER_DEPTH_M = 0.056


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", required=True, type=Path)
    parser.add_argument("--depth-npy", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--camera-json", required=True, type=Path)
    parser.add_argument("--extra-view", action="append", default=[], metavar="RGB,DEPTH_NPY,MASK,CAMERA_JSON")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--object-name", required=True, choices=sorted(OBJECTS))
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--scene-cloud-stride", type=int, default=1)
    parser.add_argument("--scene-cloud-max-points", type=int, default=180000)
    parser.add_argument("--save-top-grasps", type=int, default=5)
    parser.add_argument(
        "--filter-target-outliers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only the largest connected voxel component of the fused target cloud.",
    )
    parser.add_argument("--target-filter-voxel-size", type=float, default=0.008)
    parser.add_argument("--target-filter-min-points", type=int, default=50)
    parser.add_argument("--bbox-trim-quantile", type=float, default=0.005)
    parser.add_argument("--heuristic-width", type=float, default=0.075)
    parser.add_argument("--heuristic-depth", type=float, default=0.03)
    parser.add_argument(
        "--overlay-object-center-world",
        type=parse_float_list,
        default=[],
        help="Fallback object center. The heuristic prefers the fused-cloud bbox center.",
    )
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    return args


def save_cloud_views(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    status: str,
    marker: np.ndarray | None = None,
) -> None:
    canvas = Image.new("RGB", (1260, 520), (18, 22, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), f"Heuristic segmented cloud: {status}", fill=(255, 255, 255))
    boxes = [(20, 44, 410, 500), (435, 44, 825, 500), (850, 44, 1240, 500)]
    labels = [("primary camera XY", (0, 1)), ("primary camera XZ", (0, 2)), ("primary camera YZ", (1, 2))]
    for box, (label, axes) in zip(boxes, labels, strict=False):
        draw_projection(draw, box, points, colors, axes, label, marker)
    canvas.save(path)


def draw_projection(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    points: np.ndarray,
    colors: np.ndarray,
    axes: tuple[int, int],
    title: str,
    marker: np.ndarray | None,
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
        indices = np.linspace(0, len(sample) - 1, 20000).astype(np.int64)
        sample = sample[indices]
        sample_colors = sample_colors[indices]

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
        radius = 6
        draw.ellipse((mx - radius, my - radius, mx + radius, my + radius), outline=(255, 50, 50), width=2)


def robust_bbox(points_world: np.ndarray, trim_quantile: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if len(points_world) == 0:
        raise ValueError("empty target cloud")
    q = float(np.clip(trim_quantile, 0.0, 0.20))
    if q > 0.0 and len(points_world) >= 32:
        q_low, q_high = np.quantile(points_world, [q, 1.0 - q], axis=0)
        keep = np.all((points_world >= q_low) & (points_world <= q_high), axis=1)
        trimmed = points_world[keep]
        if len(trimmed) < max(16, int(0.10 * len(points_world))):
            trimmed = points_world
    else:
        trimmed = points_world
    bbox_min = np.min(trimmed, axis=0)
    bbox_max = np.max(trimmed, axis=0)
    center = (bbox_min + bbox_max) * 0.5
    return bbox_min, bbox_max, center, {
        "method": "trimmed_fused_cloud_bbox_center_world",
        "trim_quantile": q,
        "point_count": int(len(points_world)),
        "trimmed_point_count": int(len(trimmed)),
        "bbox_min_world": bbox_min.astype(float).tolist(),
        "bbox_max_world": bbox_max.astype(float).tolist(),
        "bbox_size_world": (bbox_max - bbox_min).astype(float).tolist(),
        "bbox_center_world": center.astype(float).tolist(),
    }


def save_overlay(
    path: Path,
    rgb_path: Path,
    mask_path: Path,
    camera: dict,
    candidate: dict,
    status: str,
) -> None:
    image = Image.open(rgb_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    mask_arr = np.asarray(mask) > 127
    layer = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    layer[mask_arr] = (255, 204, 0, 70)
    image = Image.alpha_composite(image, Image.fromarray(layer, mode="RGBA"))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 42), fill=(0, 0, 0, 180))
    draw.text((8, 10), f"Heuristic top-down grasp | status={status}", fill=(255, 255, 255, 255))
    intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float64)
    point = np.asarray(candidate["translation_camera"], dtype=np.float64)
    uv = project_camera_point(intrinsic, point)
    if uv is not None:
        x, y = uv
        radius = 8
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(255, 64, 64), width=3)
        draw.text((x + 10, y - 10), "exec", fill=(255, 64, 64, 255))
    contact = np.asarray(candidate["contact_translation_camera"], dtype=np.float64)
    uv_contact = project_camera_point(intrinsic, contact)
    if uv_contact is not None:
        x, y = uv_contact
        radius = 6
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(64, 220, 255), width=2)
        draw.text((x + 8, y - 8), "contact", fill=(64, 220, 255, 255))
    image.convert("RGB").save(path)


def top_down_candidate(args: argparse.Namespace, camera: dict, points_world: np.ndarray) -> tuple[dict, dict, np.ndarray]:
    cfg = OBJECTS[args.object_name]
    bbox_min, bbox_max, bbox_center, bbox_details = robust_bbox(points_world, args.bbox_trim_quantile)
    if not np.isfinite(bbox_center).all() and len(args.overlay_object_center_world) >= 3:
        bbox_center = np.asarray(args.overlay_object_center_world[:3], dtype=np.float64)

    quat = task_e_grasp_quat(cfg["object_quat_wxyz"])
    rotation = quat_wxyz_to_matrix(quat)
    approach_axis = rotation[:, 2].astype(np.float64)

    object_center = bbox_center.astype(np.float64)
    object_center[2] = max(float(object_center[2]), float(TABLE_TOP_Z + 0.01))
    execution_translation = np.asarray(
        [
            object_center[0],
            object_center[1],
            object_center[2] + float(cfg["grasp_z_offset"]),
        ],
        dtype=np.float64,
    )

    bbox_height = max(0.0, float(bbox_max[2] - bbox_min[2]))
    surface_margin = min(0.005, 0.20 * bbox_height)
    contact_z = float(np.clip(object_center[2], bbox_min[2] + surface_margin, bbox_max[2] - surface_margin))
    contact_translation = np.asarray([object_center[0], object_center[1], contact_z], dtype=np.float64)
    visual_finger_root = contact_translation - approach_axis * PIPER_FINGER_LENGTH_M

    exec_camera = transform_points_from_world(execution_translation[None, :], camera)[0]
    contact_camera = transform_points_from_world(contact_translation[None, :], camera)[0]
    rotation_camera = quat_wxyz_to_matrix(camera["quat_w_ros"]).T @ rotation
    candidate = {
        "rank": 1,
        "source_rank": 1,
        "score": 1.0,
        "width": float(args.heuristic_width),
        "depth": float(args.heuristic_depth),
        "translation_camera": exec_camera.astype(float).tolist(),
        "rotation_matrix_camera": rotation_camera.astype(float).tolist(),
        "pose_world": {
            "translation": execution_translation.astype(float).tolist(),
            "rotation_matrix": rotation.astype(float).tolist(),
        },
        "translation_world": execution_translation.astype(float).tolist(),
        "rotation_matrix_world": rotation.astype(float).tolist(),
        "translation": execution_translation.astype(float).tolist(),
        "rotation_matrix": rotation.astype(float).tolist(),
        "generator": "heuristic_top_down",
        "pose_type": "heuristic_top_down_execution_pose",
        "contact_pose_world": {
            "translation": contact_translation.astype(float).tolist(),
            "rotation_matrix": rotation.astype(float).tolist(),
        },
        "contact_translation_camera": contact_camera.astype(float).tolist(),
        "object_center_w": object_center.astype(float).tolist(),
        "center_details": bbox_details,
        "approach_axis_world": approach_axis.astype(float).tolist(),
        "pregrasp_direction_world": (-approach_axis).astype(float).tolist(),
        "visual_finger_root_center_w": visual_finger_root.astype(float).tolist(),
        "visual_finger_root_clearance_above_bbox_top_m": float(visual_finger_root[2] - bbox_max[2]),
        "piper_gripper_visualization_geometry": {
            "model": "simplified_real_scale_link7_link8_boxes",
            "finger_length_m": PIPER_FINGER_LENGTH_M,
            "finger_width_opening_axis_m": PIPER_FINGER_WIDTH_M,
            "finger_depth_side_axis_m": PIPER_FINGER_DEPTH_M,
        },
    }
    final_pose = {
        "frame": "world",
        "pose_type": "heuristic_top_down_execution_pose",
        "approach_axis": "rotation_matrix[:,2]",
        "pregrasp_direction": "-rotation_matrix[:,2]",
        "translation": execution_translation.astype(float).tolist(),
        "rotation_matrix": rotation.astype(float).tolist(),
        "quat_wxyz": [float(v) for v in quat],
        "score": 1.0,
        "width": float(args.heuristic_width),
        "depth": float(args.heuristic_depth),
        "source_rank": 1,
        "generator": "heuristic_top_down",
        "object": args.object_name,
        "object_center_w": object_center.astype(float).tolist(),
        "contact_pose_world": candidate["contact_pose_world"],
        "center_details": bbox_details,
        "translation_convention": "translation is the executed Piper gripper_base target from the current fused object cloud",
    }
    return candidate, final_pose, exec_camera


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    camera_payload = load_json(args.camera_json)
    primary_camera = camera_payload.get("camera", camera_payload)

    scene_batches = []
    scene_color_batches = []
    target_batches = []
    target_color_batches = []
    view_records = []

    scene, scene_colors, target, target_colors, stats = view_points(
        args.rgb,
        args.depth_npy,
        args.mask,
        primary_camera,
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
        extra_payload = load_json(extra_camera_json)
        extra_camera = extra_payload.get("camera", extra_payload)
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
            e_scene_primary = transform_points_from_world(transform_points_to_world(e_scene, extra_camera), primary_camera).astype(np.float32)
            keep = np.isfinite(e_scene_primary).all(axis=1) & (e_scene_primary[:, 2] > 0.0)
            e_stats["scene_point_count_after_primary_reprojection"] = int(np.count_nonzero(keep))
            if keep.any():
                scene_batches.append(e_scene_primary[keep])
                scene_color_batches.append(e_scene_colors[keep])
        if len(e_target):
            e_target_primary = transform_points_from_world(transform_points_to_world(e_target, extra_camera), primary_camera).astype(np.float32)
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
    scene_points, scene_colors, scene_downsample = deterministic_downsample(
        scene_points,
        scene_colors,
        args.scene_cloud_max_points,
    )

    np.save(args.output / "anygrasp_input_cloud.npy", scene_points)
    np.save(args.output / "anygrasp_input_cloud_colors.npy", scene_colors)
    np.save(args.output / "masked_cloud.npy", target_points)
    np.save(args.output / "masked_cloud_colors.npy", target_colors)
    write_ply(args.output / "anygrasp_input_cloud.ply", scene_points, scene_colors)
    write_ply(args.output / "masked_cloud.ply", target_points, target_colors)

    status = "ok" if len(target_points) >= 20 else "insufficient_points"
    result_payload = {
        "status": status,
        "generator": "heuristic_top_down",
        "object": args.object_name,
        "scene_point_count": int(len(scene_points)),
        "target_point_count": int(len(target_points)),
    }
    top_grasps = []
    final_pose = None
    marker = None
    if status == "ok":
        points_world = transform_points_to_world(target_points, primary_camera)
        candidate, final_pose, marker = top_down_candidate(args, primary_camera, points_world)
        top_grasps = [candidate]
        result_payload.update(
            {
                "best": candidate,
                "top_grasps": top_grasps,
                "final_grasp_pose": final_pose,
            }
        )

    save_cloud_views(args.output / "anygrasp_result.png", scene_points, scene_colors, status, marker=marker)
    save_cloud_views(args.output / "target_cloud.png", target_points, target_colors, "target_mask", marker=marker)
    (args.output / "top_grasps.json").write_text(json.dumps(top_grasps, indent=2), encoding="utf-8")
    if top_grasps:
        save_overlay(args.output / "top_grasps_overlay.png", args.rgb, args.mask, primary_camera, top_grasps[0], status)
        save_overlay(args.output / "selected_grasp_overlay.png", args.rgb, args.mask, primary_camera, top_grasps[0], status)
    else:
        Image.new("RGB", (640, 480), (18, 22, 30)).save(args.output / "top_grasps_overlay.png")
        Image.new("RGB", (640, 480), (18, 22, 30)).save(args.output / "selected_grasp_overlay.png")

    summary = {
        "grasp_generator": "heuristic_top_down",
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
        "scene_cloud_downsample": scene_downsample,
        "target_outlier_filter": target_filter,
        "fused_view_count": int(len(view_records)),
        "views": view_records,
        "scene_views": view_records,
        "visualization": "anygrasp_result.png",
        "target_visualization": "target_cloud.png",
        "top_grasps_json": "top_grasps.json",
        "top_grasps_overlay": "top_grasps_overlay.png",
        "selected_grasp_overlay": "selected_grasp_overlay.png",
        "top_grasps_overlay_geometry": {
            "object_center_world": final_pose.get("object_center_w") if isinstance(final_pose, dict) else None,
            "finger_length_m": PIPER_FINGER_LENGTH_M,
            "finger_width_m": PIPER_FINGER_WIDTH_M,
            "finger_depth_m": PIPER_FINGER_DEPTH_M,
            "note": "Heuristic top-down candidate from current fused segmented object cloud.",
        },
        "anygrasp": result_payload,
        "heuristic": result_payload,
    }
    (args.output / "anygrasp_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "heuristic_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if isinstance(final_pose, dict) and status == "ok":
        (args.output / "final_grasp_pose.json").write_text(json.dumps(final_pose, indent=2), encoding="utf-8")
    else:
        (args.output / "final_grasp_pose_error.json").write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    print(f"[INFO] Saved heuristic result: {args.output / 'heuristic_result.json'}")


if __name__ == "__main__":
    main()
