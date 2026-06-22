#!/usr/bin/env python3
"""Evaluate saved AnyGrasp/GraspGen candidates with simplified Piper geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


PIPER_FINGER_LENGTH_M = 0.0765
PIPER_FINGER_WIDTH_M = 0.0265
PIPER_FINGER_DEPTH_M = 0.056
PIPER_PALM_APPROACH_THICKNESS_M = 0.035
PIPER_MAX_JAW_WIDTH_M = 0.075


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generator_dirs", nargs="+", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--tool-transform",
        choices=("identity", "graspnet_to_piper_z"),
        default="graspnet_to_piper_z",
    )
    parser.add_argument(
        "--offset-modes",
        type=parse_csv,
        default=["approach_axis", "finger_centerline", "towards_object_center"],
        help="Comma-separated modes to evaluate.",
    )
    parser.add_argument("--offset-min", type=float, default=0.0)
    parser.add_argument("--offset-max", type=float, default=0.12)
    parser.add_argument("--offset-step", type=float, default=0.005)
    parser.add_argument("--clearance", type=float, default=0.002)
    parser.add_argument("--target-exclusion-radius", type=float, default=0.012)
    parser.add_argument("--max-scene-points", type=int, default=60000)
    parser.add_argument("--target-solid-max-points", type=int, default=0)
    parser.add_argument("--obstacle-solid-max-points", type=int, default=0)
    parser.add_argument("--centerline-min-points", type=int, default=10)
    parser.add_argument("--closing-region-min-points", type=int, default=10)
    parser.add_argument("--centerline-half-width", type=float, default=0.012)
    parser.add_argument("--centerline-half-depth", type=float, default=0.018)
    parser.add_argument("--piper-max-jaw-width", type=float, default=PIPER_MAX_JAW_WIDTH_M)
    parser.add_argument("--piper-clip-generator-width", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-name", default="piper_grasp_candidate_eval.json")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_points_to_world(points_cam: np.ndarray, camera: dict) -> np.ndarray:
    rotation_wc = quat_wxyz_to_matrix(np.asarray(camera["quat_w_ros"], dtype=np.float64))
    translation_wc = np.asarray(camera["pos_w"], dtype=np.float64)
    return np.asarray(points_cam, dtype=np.float64) @ rotation_wc.T + translation_wc


def normalize(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        return np.asarray(fallback, dtype=np.float64)
    return np.asarray(value, dtype=np.float64) / norm


def result_json_path(generator_dir: Path) -> Path:
    for name in ("anygrasp_result.json", "graspgen_result.json", "heuristic_result.json"):
        path = generator_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"missing result json in {generator_dir}")


def scene_cloud_path(generator_dir: Path) -> Path | None:
    for name in ("anygrasp_input_cloud.npy", "graspgen_input_cloud.npy"):
        path = generator_dir / name
        if path.exists():
            return path
    return None


def deterministic_downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, int(max_points)).astype(np.int64)
    return points[indices]


def obstacle_points_without_target(
    scene_points_w: np.ndarray,
    target_points_w: np.ndarray,
    exclusion_radius: float,
    max_scene_points: int,
) -> tuple[np.ndarray, dict]:
    scene_points_w = deterministic_downsample(scene_points_w, max_scene_points)
    if len(scene_points_w) == 0 or len(target_points_w) == 0:
        return scene_points_w, {
            "method": "empty_scene_or_target",
            "scene_point_count": int(len(scene_points_w)),
            "obstacle_point_count": int(len(scene_points_w)),
        }
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(target_points_w)
        distances, _ = tree.query(scene_points_w, k=1, workers=-1)
        keep = distances > float(exclusion_radius)
        method = "cKDTree_target_exclusion"
    except Exception as exc:
        keep = np.ones(len(scene_points_w), dtype=bool)
        method = f"fallback_all_scene_points:{type(exc).__name__}"
    return scene_points_w[keep], {
        "method": method,
        "scene_point_count": int(len(scene_points_w)),
        "target_point_count": int(len(target_points_w)),
        "obstacle_point_count": int(np.count_nonzero(keep)),
        "target_exclusion_radius_m": float(exclusion_radius),
    }


def load_clouds(generator_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    result_path = result_json_path(generator_dir)
    result = load_json(result_path)
    camera_path = Path(result["camera_json"]).expanduser()
    if not camera_path.is_absolute():
        camera_path = (generator_dir / camera_path).resolve()
    camera_payload = load_json(camera_path)
    camera = camera_payload.get("camera", camera_payload)
    target_cam = np.load(generator_dir / "masked_cloud.npy").astype(np.float64)
    target_w = transform_points_to_world(target_cam, camera)
    scene_path = scene_cloud_path(generator_dir)
    if scene_path is not None:
        scene_w = transform_points_to_world(np.load(scene_path).astype(np.float64), camera)
    else:
        scene_w = np.empty((0, 3), dtype=np.float64)
    meta = {
        "result_json": str(result_path),
        "camera_json": str(camera_path),
        "target_cloud": str(generator_dir / "masked_cloud.npy"),
        "target_point_count": int(len(target_w)),
        "scene_cloud": str(scene_path) if scene_path is not None else None,
        "scene_point_count": int(len(scene_w)),
    }
    return target_w, scene_w, meta


def top_grasp_pose(raw: dict, rank: int) -> dict | None:
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
    if not isinstance(world, dict) or world.get("translation") is None or world.get("rotation_matrix") is None:
        return None
    return {
        "rank": int(raw.get("rank", raw.get("source_rank", rank))),
        "source_rank": int(raw.get("source_rank", raw.get("rank", rank))),
        "score": float(raw.get("score", 0.0)),
        "width": float(raw.get("width", 0.065)),
        "depth": float(raw.get("depth", 0.0)),
        "translation": [float(v) for v in world["translation"]],
        "rotation_matrix": [[float(v) for v in row] for row in world["rotation_matrix"]],
        "raw_collision_free": raw.get("collision_free"),
    }


def load_candidates(generator_dir: Path, top_k: int) -> list[dict]:
    raw = load_json(generator_dir / "top_grasps.json")
    candidates = []
    for idx, item in enumerate(raw, start=1):
        pose = top_grasp_pose(item, idx)
        if pose is not None:
            candidates.append(pose)
    candidates.sort(key=lambda item: (-float(item["score"]), int(item["source_rank"])))
    return candidates[:top_k] if top_k > 0 else candidates


def piper_rotation(rotation: np.ndarray, tool_transform: str) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(rotation, dtype=np.float64)
    if tool_transform == "graspnet_to_piper_z":
        rot = np.stack([-rotation[:, 2], rotation[:, 1], rotation[:, 0]], axis=1)
        approach = normalize(rot[:, 2], np.array([0.0, 0.0, -1.0], dtype=np.float64))
    else:
        rot = rotation
        approach = normalize(rot[:, 2], np.array([0.0, 0.0, -1.0], dtype=np.float64))
    return rot, approach


def offset_axis(mode: str, translation: np.ndarray, approach: np.ndarray, object_center: np.ndarray) -> np.ndarray:
    if mode == "towards_object_center":
        return normalize(object_center - translation, -approach)
    if mode in {"finger_centerline", "yellow_line"}:
        return approach
    return -approach


def gripper_stats(
    points_w: np.ndarray,
    base_w: np.ndarray,
    rotation_w: np.ndarray,
    jaw_width: float,
    clearance: float,
    centerline_half_width: float,
    centerline_half_depth: float,
) -> dict:
    if len(points_w) == 0:
        return {
            "solid_point_count": 0,
            "component_collision_points": {"left_finger": 0, "right_finger": 0, "palm_base": 0},
            "closing_region_point_count": 0,
            "centerline_point_count": 0,
        }
    side_axis = normalize(rotation_w[:, 0], np.array([1.0, 0.0, 0.0], dtype=np.float64))
    jaw_axis = normalize(rotation_w[:, 1], np.array([0.0, 1.0, 0.0], dtype=np.float64))
    approach_axis = normalize(rotation_w[:, 2], np.array([0.0, 0.0, -1.0], dtype=np.float64))
    base_w = np.asarray(base_w, dtype=np.float64)
    points_w = np.asarray(points_w, dtype=np.float64)
    jaw_width = float(np.clip(jaw_width, 0.030, 0.095))
    clearance = max(0.0, float(clearance))

    solid = np.zeros(len(points_w), dtype=bool)
    components: dict[str, int] = {}
    for sign, label in [(-1.0, "left_finger"), (1.0, "right_finger")]:
        root_center = base_w + jaw_axis * sign * (jaw_width * 0.5 + PIPER_FINGER_WIDTH_M * 0.5)
        rel = points_w - root_center[None, :]
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
        solid |= inside

    palm_center = base_w - approach_axis * (PIPER_PALM_APPROACH_THICKNESS_M * 0.5)
    rel = points_w - palm_center[None, :]
    palm_inside = (
        (np.abs(rel @ approach_axis) <= PIPER_PALM_APPROACH_THICKNESS_M * 0.5 + clearance)
        & (np.abs(rel @ jaw_axis) <= (jaw_width * 0.5 + PIPER_FINGER_WIDTH_M) + clearance)
        & (np.abs(rel @ side_axis) <= PIPER_FINGER_DEPTH_M * 0.5 + clearance)
    )
    components["palm_base"] = int(np.count_nonzero(palm_inside))
    solid |= palm_inside

    rel = points_w - base_w[None, :]
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
        & (np.abs(jaw) <= float(centerline_half_width) + clearance)
        & (np.abs(side) <= float(centerline_half_depth) + clearance)
    )
    return {
        "solid_point_count": int(np.count_nonzero(solid)),
        "component_collision_points": components,
        "closing_region_point_count": int(np.count_nonzero(closing_region)),
        "centerline_point_count": int(np.count_nonzero(centerline)),
        "base_w": base_w.astype(float).tolist(),
        "finger_tip_center_w": (base_w + approach_axis * PIPER_FINGER_LENGTH_M).astype(float).tolist(),
        "jaw_width_m": jaw_width,
    }


def evaluate_candidate(
    candidate: dict,
    target_w: np.ndarray,
    obstacle_w: np.ndarray,
    object_center_w: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    translation = np.asarray(candidate["translation"], dtype=np.float64)
    rotation_w, approach = piper_rotation(np.asarray(candidate["rotation_matrix"], dtype=np.float64), args.tool_transform)
    jaw_width_raw = float(candidate.get("width", 0.065))
    jaw_width = min(jaw_width_raw, float(args.piper_max_jaw_width)) if bool(args.piper_clip_generator_width) else jaw_width_raw
    jaw_width = float(np.clip(jaw_width, 0.030, float(args.piper_max_jaw_width)))
    width_info = {
        "raw_width_m": jaw_width_raw,
        "checked_width_m": jaw_width,
        "max_piper_jaw_width_m": float(args.piper_max_jaw_width),
        "width_clipped": bool(jaw_width < jaw_width_raw - 1e-9),
        "clip_enabled": bool(args.piper_clip_generator_width),
    }
    offsets = [
        float(v)
        for v in np.arange(float(args.offset_min), float(args.offset_max) + float(args.offset_step) * 0.5, float(args.offset_step))
    ]
    if not offsets:
        offsets = [float(args.offset_min)]

    tested = []
    best = None
    for mode in args.offset_modes:
        axis = offset_axis(mode, translation, approach, object_center_w)
        for distance in offsets:
            base_w = translation + axis * float(distance)
            target_stats = gripper_stats(
                target_w,
                base_w,
                rotation_w,
                jaw_width,
                args.clearance,
                args.centerline_half_width,
                args.centerline_half_depth,
            )
            obstacle_stats = gripper_stats(
                obstacle_w,
                base_w,
                rotation_w,
                jaw_width,
                args.clearance,
                args.centerline_half_width,
                args.centerline_half_depth,
            )
            ok = (
                int(target_stats["solid_point_count"]) <= int(args.target_solid_max_points)
                and int(obstacle_stats["solid_point_count"]) <= int(args.obstacle_solid_max_points)
                and int(target_stats["centerline_point_count"]) >= int(args.centerline_min_points)
                and int(target_stats["closing_region_point_count"]) >= int(args.closing_region_min_points)
            )
            entry = {
                "ok": bool(ok),
                "offset_mode": mode,
                "offset_m": float(distance),
                "offset_axis_w": axis.astype(float).tolist(),
                "width": width_info,
                "width_ok": True,
                "raw_width_m": jaw_width_raw,
                "checked_width_m": jaw_width,
                "target": target_stats,
                "obstacle": obstacle_stats,
            }
            tested.append(entry)
            key = (
                0 if ok else 1,
                int(target_stats["solid_point_count"]) + int(obstacle_stats["solid_point_count"]),
                -int(target_stats["centerline_point_count"]),
                -int(target_stats["closing_region_point_count"]),
                abs(float(distance) - PIPER_FINGER_LENGTH_M),
            )
            if best is None or key < best[0]:
                best = (key, entry)

    return {
        "rank": int(candidate["source_rank"]),
        "score": float(candidate["score"]),
        "raw_collision_free": candidate.get("raw_collision_free"),
        "best": best[1] if best is not None else None,
        "tested_count": len(tested),
        "tested": tested,
    }


def evaluate_dir(generator_dir: Path, args: argparse.Namespace) -> dict:
    generator_dir = generator_dir.expanduser().resolve()
    target_w, scene_w, cloud_meta = load_clouds(generator_dir)
    object_center_w = (np.quantile(target_w, 0.01, axis=0) + np.quantile(target_w, 0.99, axis=0)) * 0.5
    obstacle_w, obstacle_meta = obstacle_points_without_target(
        scene_w,
        target_w,
        args.target_exclusion_radius,
        args.max_scene_points,
    )
    candidates = load_candidates(generator_dir, args.top_k)
    evaluations = [
        evaluate_candidate(candidate, target_w, obstacle_w, object_center_w, args)
        for candidate in candidates
    ]
    passing = [item for item in evaluations if item.get("best", {}).get("ok")]
    summary = {
        "generator_dir": str(generator_dir),
        "cloud": cloud_meta,
        "obstacle_cloud": obstacle_meta,
        "object_center_w": object_center_w.astype(float).tolist(),
        "criteria": {
            "tool_transform": args.tool_transform,
            "offset_modes": list(args.offset_modes),
            "offset_min_m": float(args.offset_min),
            "offset_max_m": float(args.offset_max),
            "offset_step_m": float(args.offset_step),
            "clearance_m": float(args.clearance),
            "target_solid_max_points": int(args.target_solid_max_points),
            "obstacle_solid_max_points": int(args.obstacle_solid_max_points),
            "centerline_min_points": int(args.centerline_min_points),
            "closing_region_min_points": int(args.closing_region_min_points),
            "piper_geometry_m": {
                "finger_length": PIPER_FINGER_LENGTH_M,
                "finger_width": PIPER_FINGER_WIDTH_M,
                "finger_depth": PIPER_FINGER_DEPTH_M,
                "palm_approach_thickness": PIPER_PALM_APPROACH_THICKNESS_M,
                "max_jaw_width": float(args.piper_max_jaw_width),
                "clip_generator_width": bool(args.piper_clip_generator_width),
            },
        },
        "candidate_count": len(evaluations),
        "passing_count": len(passing),
        "selected": passing[0] if passing else (evaluations[0] if evaluations else None),
        "evaluations": evaluations,
    }
    output_path = generator_dir / args.output_name
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["output_json"] = str(output_path)
    return summary


def main() -> None:
    args = parse_args()
    summaries = [evaluate_dir(path, args) for path in args.generator_dirs]
    for summary in summaries:
        selected = summary.get("selected") or {}
        best = selected.get("best") or {}
        target = (best.get("target") or {})
        obstacle = (best.get("obstacle") or {})
        print(
            "[EVAL] "
            f"{summary['generator_dir']} "
            f"passing={summary['passing_count']}/{summary['candidate_count']} "
            f"selected_rank={selected.get('rank')} "
            f"ok={best.get('ok')} "
            f"mode={best.get('offset_mode')} "
            f"offset={best.get('offset_m')} "
            f"target_solid={target.get('solid_point_count')} "
            f"obstacle_solid={obstacle.get('solid_point_count')} "
            f"centerline={target.get('centerline_point_count')} "
            f"closing={target.get('closing_region_point_count')} "
            f"json={summary['output_json']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
