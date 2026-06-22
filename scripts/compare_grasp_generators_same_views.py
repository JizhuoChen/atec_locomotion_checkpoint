#!/usr/bin/env python3
"""Run Contact-GraspNet on the exact views used by an AnyGrasp result."""

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
    parser.add_argument("--contact-graspnet-env", default="contact_graspnet_env")
    parser.add_argument("--contact-graspnet-root", type=Path, default=REPO_ROOT / "third_party/contact_graspnet")
    parser.add_argument(
        "--contact-graspnet-ckpt-dir",
        type=Path,
        default=REPO_ROOT / "third_party/contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001",
    )
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--scene-cloud-stride", type=int, default=1)
    parser.add_argument("--scene-cloud-max-points", type=int, default=180000)
    parser.add_argument("--save-top-grasps", type=int, default=12)
    parser.add_argument("--forward-passes", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_result(result: dict) -> dict:
    payload = result.get("anygrasp") or result.get("contact_graspnet") or {}
    return {
        "generator": result.get("grasp_generator", payload.get("generator", "anygrasp")),
        "status": payload.get("status"),
        "target_point_count": result.get("target_point_count") or result.get("point_count"),
        "scene_input_point_count": result.get("anygrasp_input_point_count"),
        "fused_view_count": result.get("fused_view_count"),
        "num_grasps_after_nms": payload.get("num_grasps_after_nms"),
        "num_grasps_after_target_filter": payload.get("num_grasps_after_target_filter"),
        "raw_count": payload.get("raw_count"),
        "best_score": (payload.get("best") or {}).get("score"),
        "best_rank": (payload.get("best") or {}).get("rank"),
        "top_grasp_count": len(payload.get("top_grasps") or []),
    }


def main() -> None:
    args = parse_args()
    anygrasp_dir = args.anygrasp_dir.expanduser().resolve()
    anygrasp_result_path = anygrasp_dir / "anygrasp_result.json"
    anygrasp_result = load_json(anygrasp_result_path)
    output_dir = args.output_dir or (anygrasp_dir.parent / "contact_graspnet_same_views")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    contact_result_path = output_dir / "contact_graspnet_result.json"
    if args.force or not contact_result_path.exists():
        if args.force and contact_result_path.exists():
            contact_result_path.unlink()
        views = anygrasp_result.get("views") or []
        primary = views[0] if views else anygrasp_result
        command = [
            "conda",
            "run",
            "-n",
            args.contact_graspnet_env,
            "python",
            "scripts/contact_graspnet_from_rgbd_mask.py",
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
            "--contact-graspnet-root",
            str(args.contact_graspnet_root),
            "--ckpt-dir",
            str(args.contact_graspnet_ckpt_dir),
            "--scene-cloud-stride",
            str(args.scene_cloud_stride),
            "--scene-cloud-max-points",
            str(args.scene_cloud_max_points),
            "--save-top-grasps",
            str(args.save_top_grasps),
            "--forward-passes",
            str(args.forward_passes),
        ]
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
        log_path = output_dir / "contact_graspnet.log"
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
        if proc.returncode != 0 and not contact_result_path.exists():
            contact_result_path.write_text(
                json.dumps(
                    {
                        "grasp_generator": "contact_graspnet",
                        "contact_graspnet": {
                            "status": "runner_failed",
                            "returncode": proc.returncode,
                            "log": str(log_path),
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    contact_result = load_json(contact_result_path)
    comparison = {
        "anygrasp_dir": str(anygrasp_dir),
        "contact_graspnet_dir": str(output_dir),
        "same_views_source": str(anygrasp_result_path),
        "anygrasp": summarize_result(anygrasp_result),
        "contact_graspnet": summarize_result(contact_result),
        "artifacts": {
            "contact_result": str(contact_result_path),
            "contact_log": str(output_dir / "contact_graspnet.log"),
            "contact_input_cloud": str(output_dir / "contact_graspnet_input_cloud.ply"),
            "contact_target_cloud": str(output_dir / "masked_cloud.ply"),
            "contact_top_overlay": str(output_dir / "top_grasps_overlay.png"),
        },
    }
    comparison_path = output_dir / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"[INFO] Saved {comparison_path}")


if __name__ == "__main__":
    main()
