#!/usr/bin/env python3
"""Run GraspGen on the exact views used by an AnyGrasp result."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anygrasp-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--graspgen-env", default="graspgen_env")
    parser.add_argument("--graspgen-root", type=Path, default=REPO_ROOT / "third_party/graspgen")
    parser.add_argument(
        "--gripper-config",
        type=Path,
        default=REPO_ROOT / "third_party/graspgen_models/checkpoints/graspgen_robotiq_2f_140.yml",
    )
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--scene-cloud-stride", type=int, default=1)
    parser.add_argument("--scene-cloud-max-points", type=int, default=180000)
    parser.add_argument("--target-cloud-max-points", type=int, default=2048)
    parser.add_argument("--collision-max-scene-points", type=int, default=8192)
    parser.add_argument("--num-grasps", type=int, default=500)
    parser.add_argument("--save-top-grasps", type=int, default=20)
    parser.add_argument("--export-tcp-offset", type=float, default=0.195)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-filter-collisions", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_result(result: dict) -> dict:
    payload = result.get("graspgen") or result.get("anygrasp") or {}
    return {
        "generator": result.get("grasp_generator", payload.get("generator", "unknown")),
        "status": payload.get("status"),
        "target_point_count": result.get("target_point_count") or result.get("point_count"),
        "target_model_point_count": result.get("target_model_point_count"),
        "scene_input_point_count": result.get("graspgen_input_point_count") or result.get("anygrasp_input_point_count"),
        "fused_view_count": result.get("fused_view_count"),
        "raw_count": payload.get("raw_count"),
        "num_grasps_after_target_filter": payload.get("num_grasps_after_target_filter"),
        "num_grasps_after_collision_filter": payload.get("num_grasps_after_collision_filter"),
        "best_score": (payload.get("best") or {}).get("score"),
        "best_rank": (payload.get("best") or {}).get("rank"),
        "top_grasp_count": len(payload.get("top_grasps") or []),
        "collision_filter": payload.get("collision_filter"),
    }


def main() -> None:
    args = parse_args()
    anygrasp_dir = args.anygrasp_dir.expanduser().resolve()
    anygrasp_result_path = anygrasp_dir / "anygrasp_result.json"
    anygrasp_result = load_json(anygrasp_result_path)
    output_dir = args.output_dir or (anygrasp_dir.parent / "graspgen_same_views")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    graspgen_result_path = output_dir / "graspgen_result.json"
    if args.force or not graspgen_result_path.exists():
        if args.force and graspgen_result_path.exists():
            graspgen_result_path.unlink()
        views = anygrasp_result.get("views") or []
        primary = views[0] if views else anygrasp_result
        command = [
            "conda",
            "run",
            "-n",
            args.graspgen_env,
            "python",
            "scripts/graspgen_from_rgbd_mask.py",
            "--rgb",
            str(primary["rgb"]),
            "--depth-npy",
            str(primary["depth_npy"]),
            "--mask",
            str(primary["mask"]),
            "--camera-json",
            str(primary["camera_json"]),
            "--output",
            str(output_dir),
            "--graspgen-root",
            str(args.graspgen_root),
            "--gripper-config",
            str(args.gripper_config),
            "--scene-cloud-stride",
            str(args.scene_cloud_stride),
            "--scene-cloud-max-points",
            str(args.scene_cloud_max_points),
            "--target-cloud-max-points",
            str(args.target_cloud_max_points),
            "--collision-max-scene-points",
            str(args.collision_max_scene_points),
            "--num-grasps",
            str(args.num_grasps),
            "--save-top-grasps",
            str(args.save_top_grasps),
            "--export-tcp-offset",
            str(args.export_tcp_offset),
        ]
        if args.no_filter_collisions:
            command.append("--no-filter-collisions")
        if args.max_depth is not None:
            command.extend(["--max-depth", str(args.max_depth)])
        for view in views[1:]:
            if not all(key in view for key in ("rgb", "depth_npy", "mask", "camera_json")):
                continue
            command.extend(
                [
                    "--extra-view",
                    ",".join([str(view["rgb"]), str(view["depth_npy"]), str(view["mask"]), str(view["camera_json"])]),
                ]
            )
        log_path = output_dir / "graspgen.log"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                env=os.environ.copy(),
            )
        if proc.returncode != 0 and not graspgen_result_path.exists():
            graspgen_result_path.write_text(
                json.dumps(
                    {
                        "grasp_generator": "graspgen",
                        "graspgen": {
                            "status": "runner_failed",
                            "returncode": proc.returncode,
                            "log": str(log_path),
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    graspgen_result = load_json(graspgen_result_path)
    comparison = {
        "anygrasp_dir": str(anygrasp_dir),
        "graspgen_dir": str(output_dir),
        "same_views_source": str(anygrasp_result_path),
        "anygrasp": summarize_result(anygrasp_result),
        "graspgen": summarize_result(graspgen_result),
        "artifacts": {
            "graspgen_result": str(graspgen_result_path),
            "graspgen_log": str(output_dir / "graspgen.log"),
            "graspgen_input_cloud": str(output_dir / "graspgen_input_cloud.ply"),
            "graspgen_target_cloud": str(output_dir / "masked_cloud.ply"),
            "graspgen_model_cloud": str(output_dir / "graspgen_target_model_cloud.ply"),
            "graspgen_top_overlay": str(output_dir / "top_grasps_overlay.png"),
            "graspgen_selected_overlay": str(output_dir / "selected_grasp_overlay.png"),
            "graspgen_top_json": str(output_dir / "top_grasps.json"),
        },
    }
    comparison_path = output_dir / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"[INFO] Saved {comparison_path}")


if __name__ == "__main__":
    main()
