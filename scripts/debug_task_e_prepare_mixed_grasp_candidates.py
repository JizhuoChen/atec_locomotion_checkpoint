#!/usr/bin/env python3
"""Prepare mixed Task E grasp candidates for inspection.

The mixed policy keeps banana on EE-only fused clouds because the far/video
view still creates a ghost cloud, while mustard and box use the full filtered
fusion. The script copies existing AnyGrasp/GraspGen outputs and adds a simple
top-down heuristic grasp at a robust object center.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.task_e_full_baseline_request import OBJECTS, quat_wxyz_to_matrix, task_e_grasp_quat


DEFAULT_SOURCE_ROOT = (
    REPO_ROOT
    / "outputs/task_e_ideal_ee_camera_debug/20260614_same_env_lookat_real_ee"
)
DEFAULT_OUTPUT_NAME = "grasp_candidates_mixed_ee_banana_all_others"
OBJECT_ORDER = ("banana", "mustard_bottle", "box_object")
PIPER_FINGER_LENGTH_M = 0.0765
PIPER_FINGER_WIDTH_M = 0.0265
PIPER_FINGER_DEPTH_M = 0.056
VIEW_POLICY = {
    "banana": {
        "source_run": "grasp_fusion_debug_ee_only",
        "policy": "ee_only",
        "reason": "avoid the known banana far/video-view ghost cloud",
    },
    "mustard_bottle": {
        "source_run": "grasp_fusion_debug_all_filtered",
        "policy": "all_filtered",
        "reason": "full fusion looked geometrically clean for this object",
    },
    "box_object": {
        "source_run": "grasp_fusion_debug_all_filtered",
        "policy": "all_filtered",
        "reason": "full fusion looked geometrically clean for this object",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--heuristic-center-source",
        choices=("auto", "scene_object_pose", "fused_cloud_bbox", "debug_request"),
        default="auto",
        help=(
            "Center used for the heuristic contact target. auto prefers saved simulator "
            "scene object pose, then robust fused-cloud bbox center, then debug request center."
        ),
    )
    parser.add_argument("--heuristic-width", type=float, default=0.075)
    parser.add_argument("--heuristic-depth", type=float, default=0.03)
    parser.add_argument(
        "--heuristic-root-clearance",
        type=float,
        default=0.01,
        help="Minimum world-Z clearance between object cloud top and the visualized finger root/top side.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def object_center_from_debug(source_root: Path, object_name: str) -> list[float]:
    pose_path = source_root / object_name / "intended_camera_pose.json"
    if pose_path.exists():
        payload = load_json(pose_path)
        center = payload.get("request_object_center_w")
        if center is None:
            center = (payload.get("intended_camera_pose") or {}).get("target_object_center_w")
        if center is not None:
            return [float(v) for v in center[:3]]
    raise FileNotFoundError(f"missing object center for {object_name}: {pose_path}")


def scene_object_center_from_debug(source_root: Path, object_name: str) -> list[float] | None:
    pose_path = source_root / object_name / "intended_camera_pose.json"
    if not pose_path.exists():
        return None
    payload = load_json(pose_path)
    scene_pose = payload.get("scene_object_pose") or payload.get("scene_object_root_pose")
    if not isinstance(scene_pose, dict):
        return None
    center = scene_pose.get("center_world") or scene_pose.get("center_w")
    if center is None or len(center) < 3:
        return None
    return [float(v) for v in center[:3]]


def transform_points_to_world(points_camera: np.ndarray, camera: dict) -> np.ndarray:
    rotation_wc = quat_wxyz_to_matrix(np.asarray(camera["quat_w_ros"], dtype=np.float64))
    translation_wc = np.asarray(camera["pos_w"], dtype=np.float64)
    return np.asarray(points_camera, dtype=np.float64) @ rotation_wc.T + translation_wc


def fused_cloud_bbox_center_world(generator_dir: Path) -> tuple[list[float], dict[str, Any]]:
    result = load_json(result_json_path(generator_dir))
    camera_path = Path(result["camera_json"]).expanduser().resolve()
    camera_payload = load_json(camera_path)
    camera = camera_payload.get("camera", camera_payload)
    points_camera = np.load(generator_dir / "masked_cloud.npy").astype(np.float64)
    points_world = transform_points_to_world(points_camera, camera)
    if len(points_world) == 0:
        raise ValueError(f"empty masked_cloud.npy in {generator_dir}")

    q_low, q_high = np.quantile(points_world, [0.005, 0.995], axis=0)
    keep = np.all((points_world >= q_low) & (points_world <= q_high), axis=1)
    trimmed = points_world[keep]
    if len(trimmed) < max(32, int(0.10 * len(points_world))):
        trimmed = points_world
    bbox_min = np.min(trimmed, axis=0)
    bbox_max = np.max(trimmed, axis=0)
    center = (bbox_min + bbox_max) * 0.5
    stats = {
        "source": "fused_cloud_bbox_center_world",
        "generator_dir": str(generator_dir.resolve()),
        "point_count": int(len(points_world)),
        "trimmed_point_count": int(len(trimmed)),
        "trim_quantiles": [0.005, 0.995],
        "bbox_min_world": bbox_min.astype(float).tolist(),
        "bbox_max_world": bbox_max.astype(float).tolist(),
        "bbox_size_world": (bbox_max - bbox_min).astype(float).tolist(),
        "bbox_center_world": center.astype(float).tolist(),
    }
    return center.astype(float).tolist(), stats


def heuristic_center_for_object(
    source_root: Path,
    object_name: str,
    source_generator_dir: Path,
    requested_source: str,
) -> tuple[list[float], dict[str, Any]]:
    fused_bbox: dict[str, Any] | None = None
    try:
        _, fused_bbox = fused_cloud_bbox_center_world(source_generator_dir)
    except Exception:
        fused_bbox = None

    if requested_source in {"auto", "scene_object_pose"}:
        center = scene_object_center_from_debug(source_root, object_name)
        if center is not None:
            details: dict[str, Any] = {"source": "scene_object_pose", "center_world": center}
            if fused_bbox is not None:
                details["fused_cloud_bbox"] = fused_bbox
            return center, details
        if requested_source == "scene_object_pose":
            raise FileNotFoundError(f"missing scene_object_pose for {object_name}")

    if requested_source in {"auto", "fused_cloud_bbox"}:
        try:
            return fused_cloud_bbox_center_world(source_generator_dir)
        except Exception:
            if requested_source == "fused_cloud_bbox":
                raise

    center = object_center_from_debug(source_root, object_name)
    details = {
        "source": "intended_camera_pose.request_object_center_w",
        "center_world": center,
        "warning": "fallback center is derived from the old video-mask median XY request",
    }
    if fused_bbox is not None:
        details["fused_cloud_bbox"] = fused_bbox
    return center, details


def bbox_stats_from_center_details(center_details: dict[str, Any]) -> dict[str, Any] | None:
    if "bbox_min_world" in center_details and "bbox_max_world" in center_details:
        return center_details
    fused_bbox = center_details.get("fused_cloud_bbox")
    if isinstance(fused_bbox, dict) and "bbox_min_world" in fused_bbox and "bbox_max_world" in fused_bbox:
        return fused_bbox
    return None


def result_json_path(generator_dir: Path) -> Path:
    for name in ("graspgen_result.json", "anygrasp_result.json", "heuristic_result.json"):
        path = generator_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"missing result json in {generator_dir}")


def generator_summary(generator_dir: Path) -> dict[str, Any]:
    payload = load_json(result_json_path(generator_dir))
    final_path = generator_dir / "final_grasp_pose.json"
    top_path = generator_dir / "top_grasps.json"
    final = load_json(final_path) if final_path.exists() else {}
    top_grasps = load_json(top_path) if top_path.exists() else []
    return {
        "status": "ok" if final else payload.get("status", "unknown"),
        "dir": str(generator_dir.resolve()),
        "point_count": payload.get("target_point_count") or payload.get("point_count"),
        "fused_view_count": payload.get("fused_view_count"),
        "top_grasp_count": len(top_grasps) if isinstance(top_grasps, list) else None,
        "best_score": final.get("score"),
        "final_grasp_pose_frame": final.get("frame"),
        "final_translation_world": final.get("translation"),
    }


def copy_selected_object(source_root: Path, output_dir: Path, object_name: str, overwrite: bool) -> Path:
    policy = VIEW_POLICY[object_name]
    src = source_root / policy["source_run"] / object_name
    dst = output_dir / object_name
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} already exists; pass --overwrite to replace it")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    write_json(
        dst / "view_policy.json",
        {
            "object": object_name,
            "policy": policy["policy"],
            "source_run": str(src.resolve()),
            "reason": policy["reason"],
        },
    )
    return dst


def copy_cloud_files(source_generator_dir: Path, heuristic_dir: Path) -> None:
    for name in ("masked_cloud.npy", "masked_cloud_colors.npy", "masked_cloud.ply"):
        src = source_generator_dir / name
        if src.exists():
            shutil.copy2(src, heuristic_dir / name)


def build_heuristic_grasp(
    object_name: str,
    center_w: list[float],
    center_details: dict[str, Any],
    heuristic_width: float,
    heuristic_depth: float,
    root_clearance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = OBJECTS[object_name]
    grasp_quat = task_e_grasp_quat(cfg["object_quat_wxyz"])
    rotation = quat_wxyz_to_matrix(grasp_quat).astype(float)
    approach_axis = rotation[:, 2].astype(float)
    contact_translation = np.asarray(center_w, dtype=np.float64)
    contact_height_rule: dict[str, Any] = {
        "mode": "center_z",
        "reason": "no fused-cloud bbox top was available",
    }
    bbox_stats = bbox_stats_from_center_details(center_details)
    if bbox_stats is not None:
        bbox_min = np.asarray(bbox_stats["bbox_min_world"], dtype=np.float64)
        bbox_max = np.asarray(bbox_stats["bbox_max_world"], dtype=np.float64)
        bbox_height = max(0.0, float(bbox_max[2] - bbox_min[2]))
        surface_margin = min(0.005, 0.20 * bbox_height)
        min_contact_z = float(bbox_min[2] + surface_margin)
        max_contact_z = float(bbox_max[2] - surface_margin)
        desired_for_root_clearance = float(
            bbox_max[2] + max(0.0, float(root_clearance)) - PIPER_FINGER_LENGTH_M
        )
        adjusted_z = max(float(contact_translation[2]), desired_for_root_clearance)
        if max_contact_z >= min_contact_z:
            adjusted_z = float(np.clip(adjusted_z, min_contact_z, max_contact_z))
        contact_translation[2] = adjusted_z
        contact_height_rule = {
            "mode": "bbox_top_root_clearance",
            "bbox_min_world": bbox_min.astype(float).tolist(),
            "bbox_max_world": bbox_max.astype(float).tolist(),
            "bbox_height_m": bbox_height,
            "surface_margin_m": surface_margin,
            "requested_root_clearance_m": float(root_clearance),
            "desired_contact_z_for_root_clearance": desired_for_root_clearance,
            "min_contact_z": min_contact_z,
            "max_contact_z": max_contact_z,
            "adjusted_contact_z": adjusted_z,
        }
    visual_finger_root = contact_translation - approach_axis * PIPER_FINGER_LENGTH_M
    execution_translation = np.asarray(
        [center_w[0], center_w[1], center_w[2] + float(cfg["grasp_z_offset"])],
        dtype=np.float64,
    )
    final = {
        "frame": "world",
        "pose_type": "heuristic_top_down_contact_center",
        "generator": "heuristic_top_down",
        "object": object_name,
        "object_key": cfg["object_key"],
        "center_source": center_details.get("source"),
        "center_details": center_details,
        "object_center_w": center_w,
        "translation": contact_translation.astype(float).tolist(),
        "quat_wxyz": [float(v) for v in grasp_quat],
        "rotation_matrix": rotation.tolist(),
        "translation_convention": (
            "translation is the fingertip/contact center used for grasp visualization; "
            "execution_pose_world stores the approximate EE/root target above the object"
        ),
        "approach_axis_convention": "rotation_matrix[:,2] points downward in world",
        "approach_axis_world": approach_axis.astype(float).tolist(),
        "pregrasp_direction_world": (-approach_axis).astype(float).tolist(),
        "visual_finger_root_center_w": visual_finger_root.astype(float).tolist(),
        "visual_finger_root_clearance_above_bbox_top_m": (
            float(visual_finger_root[2] - float(contact_height_rule["bbox_max_world"][2]))
            if "bbox_max_world" in contact_height_rule
            else None
        ),
        "contact_height_rule": contact_height_rule,
        "execution_pose_world": {
            "translation": execution_translation.astype(float).tolist(),
            "quat_wxyz": [float(v) for v in grasp_quat],
            "grasp_z_offset": float(cfg["grasp_z_offset"]),
            "note": "approximate EE/root pose matching the older Task E top-down z-offset convention",
        },
        "score": 1.0,
        "width": float(heuristic_width),
        "depth": float(heuristic_depth),
        "visual_finger_length_m": PIPER_FINGER_LENGTH_M,
        "piper_gripper_visualization_geometry": {
            "model": "simplified_real_scale_link7_link8_boxes",
            "finger_length_m": PIPER_FINGER_LENGTH_M,
            "finger_width_opening_axis_m": PIPER_FINGER_WIDTH_M,
            "finger_depth_side_axis_m": PIPER_FINGER_DEPTH_M,
            "note": "Visualization uses Piper STL-bounds dimensions as boxes, not full STL mesh geometry.",
        },
    }
    top_grasp = {
        "rank": 1,
        "label": "heuristic_top_down_center",
        "source": "heuristic",
        **final,
    }
    return final, top_grasp


def write_heuristic_outputs(
    source_root: Path,
    object_dir: Path,
    object_name: str,
    heuristic_center_source: str,
    heuristic_width: float,
    heuristic_depth: float,
    root_clearance: float,
) -> dict[str, Any]:
    heuristic_dir = object_dir / "heuristic"
    heuristic_dir.mkdir(parents=True, exist_ok=True)

    source_generator_dir = object_dir / "graspgen"
    if not source_generator_dir.exists():
        source_generator_dir = object_dir / "anygrasp"
    copy_cloud_files(source_generator_dir, heuristic_dir)

    source_result = load_json(result_json_path(source_generator_dir))
    camera_json = source_result.get("camera_json")
    point_count = source_result.get("target_point_count") or source_result.get("point_count")
    center_w, center_details = heuristic_center_for_object(
        source_root,
        object_name,
        source_generator_dir,
        heuristic_center_source,
    )
    final, top_grasp = build_heuristic_grasp(
        object_name,
        center_w,
        center_details,
        heuristic_width,
        heuristic_depth,
        root_clearance,
    )
    result = {
        "status": "ok",
        "generator": "heuristic_top_down",
        "object": object_name,
        "camera_json": camera_json,
        "point_cloud_frame": "primary_camera",
        "point_count": point_count,
        "target_point_count": point_count,
        "fused_cloud_source_generator": str(source_generator_dir.resolve()),
        "point_cloud_npy": "masked_cloud.npy",
        "point_cloud_colors_npy": "masked_cloud_colors.npy",
        "point_cloud_ply": "masked_cloud.ply",
        "top_grasps": [top_grasp],
        "final_grasp_pose": final,
        "top_grasps_overlay_geometry": {
            "object_center_world": final["object_center_w"],
        },
        "note": (
            "Heuristic grasp uses a robust object center and Task E top-down gripper "
            "orientation. The saved translation is the fingertip/contact center for "
            "visual inspection; execution_pose_world gives the approximate EE/root target."
        ),
    }
    write_json(heuristic_dir / "final_grasp_pose.json", final)
    write_json(heuristic_dir / "top_grasps.json", [top_grasp])
    write_json(heuristic_dir / "heuristic_result.json", result)
    write_json(heuristic_dir / "anygrasp_result.json", result)
    return generator_summary(heuristic_dir)


def write_diary(output_dir: Path, source_root: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Task E L0 Manipulation Debug Diary - 2026-06-14",
        "",
        "## What we completed",
        "",
        "- Built an EE-camera pose debug pipeline from the video-camera segmentation target.",
        "- Switched all target views to look-at mode, including banana.",
        "- Moved the real arm to the intended EE-camera pose and captured the actual EE-camera RGB/depth there.",
        "- Fixed the largest debug bug: the intended video-camera target and real EE-camera capture are now produced in the same simulator environment reset. The saved summary has `real_ee_camera_same_env_as_video: true`.",
        "- Added fused segmented point-cloud generation in the primary EE-camera frame, then ran AnyGrasp and GraspGen on the fused clouds.",
        "- Added PLY/OBJ visualization outputs for fused cloud plus generated gripper geometry.",
        "- Added largest-connected-component outlier filtering for generated RGB-D mask point clouds.",
        "- Added banana-only diagnostics comparing the real video camera with virtual cameras near the same video-camera pose.",
        "",
        "## Bugs fixed or narrowed",
        "",
        "- Environment mismatch: fixed. Earlier captures could mix video segmentation from one reset with real EE-camera images from another reset. This made camera/fusion debugging invalid.",
        "- EE-to-EE fusion: verified clean. Extra EE views align with the primary EE banana cloud at sub-millimeter median nearest-neighbor distance in the existing diagnostics.",
        "- General outliers: partially fixed. The target-cloud connected-component filter removes small disconnected components, but it does not solve the banana video-view ghost.",
        "- Video-camera extrinsic suspicion: narrowed. A virtual camera cloned from `video_camera.json` produces essentially the same banana cloud discrepancy as the real `video_cam`, so this is not just a special `video_cam` pose-readout mismatch.",
        "",
        "## Current remaining bug",
        "",
        "- Banana still shows a ghost/secondary surface when the far/video view is fused.",
        "- The deterministic banana diagnostic shows the video-view banana cloud is around 1.5 cm median nearest-neighbor distance from the primary EE cloud, with p95 around 4.1 cm. The virtual video clone and virtual video look-at camera show the same scale of discrepancy.",
        "- Because box and mustard look clean with full fusion, the remaining bug is probably specific to banana's thin/curved silhouette from the far view, SAM mask plus depth boundary pixels, or the far-view RGB-D backprojection convention. The next clean isolation is to replace SAM masks with simulator instance/ground-truth object masks for banana.",
        "",
        "## Current grasp-candidate policy",
        "",
        "- Banana uses EE-only fusion to avoid the known far/video-view ghost.",
        "- Mustard bottle and box use full filtered fusion because they do not show the same obvious ghost issue.",
        "- Each object now has three candidate generators in this mixed folder: `anygrasp`, `graspgen`, and `heuristic`.",
        "",
        "## Output folder",
        "",
        f"- `{output_dir}`",
        f"- Source debug root: `{source_root}`",
        "",
        "## Generator summary",
        "",
    ]
    for object_name in OBJECT_ORDER:
        item = summary["objects"][object_name]
        lines.append(f"### {object_name}")
        lines.append(f"- View policy: `{item['view_policy']['policy']}` from `{item['view_policy']['source_run']}`")
        for generator_name in ("anygrasp", "graspgen", "heuristic"):
            gen = item["generators"].get(generator_name, {})
            lines.append(
                "- "
                f"{generator_name}: status `{gen.get('status')}`, "
                f"points `{gen.get('point_count')}`, "
                f"top grasps `{gen.get('top_grasp_count')}`, "
                f"final frame `{gen.get('final_grasp_pose_frame')}`"
            )
        lines.append("")
    (output_dir / "DIARY_20260614.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else source_root / DEFAULT_OUTPUT_NAME
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "policy": {
            "banana": "ee_only",
            "mustard_bottle": "all_filtered",
            "box_object": "all_filtered",
        },
        "objects": {},
    }

    for object_name in OBJECT_ORDER:
        object_dir = copy_selected_object(source_root, output_dir, object_name, args.overwrite)
        view_policy = VIEW_POLICY[object_name]
        generators = {
            "anygrasp": generator_summary(object_dir / "anygrasp"),
            "graspgen": generator_summary(object_dir / "graspgen"),
        }
        generators["heuristic"] = write_heuristic_outputs(
            source_root,
            object_dir,
            object_name,
            args.heuristic_center_source,
            args.heuristic_width,
            args.heuristic_depth,
            args.heuristic_root_clearance,
        )
        summary["objects"][object_name] = {
            "dir": str(object_dir.resolve()),
            "view_policy": {
                "policy": view_policy["policy"],
                "source_run": view_policy["source_run"],
                "reason": view_policy["reason"],
            },
            "generators": generators,
        }

    write_json(output_dir / "summary.json", summary)
    write_diary(output_dir, source_root, summary)
    print(f"[INFO] Wrote mixed grasp candidates to {output_dir}")
    print(f"[INFO] Wrote diary to {output_dir / 'DIARY_20260614.md'}")


if __name__ == "__main__":
    main()
