#!/usr/bin/env python3
"""Run AnyGrasp and GraspGen on fused real EE-camera debug views."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "outputs/task_e_ideal_ee_camera_debug/20260614_all_lookat_real_ee"
OBJECTS = ("banana", "mustard_bottle", "box_object")
OBJECT_PROMPTS = {
    "banana": "banana",
    "mustard_bottle": "mustard bottle",
    "box_object": "yellow and white box",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Real EE-camera debug capture folder.")
    parser.add_argument("--output", type=Path, default=None, help="Output folder. Defaults to <input>/grasp_fusion_debug.")
    parser.add_argument("--objects", nargs="+", choices=OBJECTS, default=list(OBJECTS))
    parser.add_argument("--sam3-env", default="sam3_full")
    parser.add_argument("--sam3-device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--anygrasp-env", default="anygrasp")
    parser.add_argument("--graspgen-env", default="graspgen_env")
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--save-top-grasps", type=int, default=20)
    parser.add_argument("--scene-cloud-stride", type=int, default=1)
    parser.add_argument("--scene-cloud-max-points", type=int, default=180000)
    parser.add_argument("--graspgen-target-cloud-max-points", type=int, default=2048)
    parser.add_argument("--graspgen-num-grasps", type=int, default=500)
    parser.add_argument("--graspgen-export-tcp-offset", type=float, default=0.195)
    parser.add_argument("--anygrasp-cloud-mode", choices=("target_mask", "full_scene_target_filter"), default="target_mask")
    parser.add_argument("--filter-target-outliers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-filter-voxel-size", type=float, default=0.008)
    parser.add_argument("--target-filter-min-points", type=int, default=50)
    parser.add_argument(
        "--extra-view-set",
        choices=("all", "ee_only", "primary_only"),
        default="all",
        help=(
            "Which views to fuse. all = other EE views plus video; ee_only = other EE views only; "
            "primary_only = no extra views."
        ),
    )
    parser.add_argument("--force-masks", action="store_true")
    parser.add_argument("--force-generators", action="store_true")
    parser.add_argument("--skip-sam3", action="store_true", help="Reuse existing masks only.")
    parser.add_argument("--skip-anygrasp", action="store_true")
    parser.add_argument("--skip-graspgen", action="store_true")
    parser.add_argument("--skip-viz", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_command(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=env,
        )
        log.write(f"\n[returncode] {proc.returncode}\n")
        return int(proc.returncode)


def capture_views(input_dir: Path) -> dict[str, dict[str, Any]]:
    views: dict[str, dict[str, Any]] = {
        "video": {
            "role": "video",
            "rgb": input_dir / "video_rgb.png",
            "depth_npy": input_dir / "video_depth.npy",
            "camera_json": input_dir / "video_camera.json",
            "view_label": "eye-to-hand / video_cam",
        }
    }
    for name in OBJECTS:
        object_dir = input_dir / name
        views[f"ee_{name}"] = {
            "role": "ee_camera",
            "source_object": name,
            "rgb": object_dir / "actual_ee_camera_rgb.png",
            "depth_npy": object_dir / "actual_ee_camera_depth.npy",
            "camera_json": object_dir / "actual_ee_camera.json",
            "view_label": f"real EE camera at {name} look-at pose",
        }
    return views


def assert_view_files(views: dict[str, dict[str, Any]]) -> None:
    missing = []
    for view_name, view in views.items():
        for key in ("rgb", "depth_npy", "camera_json"):
            path = Path(view[key])
            if not path.exists():
                missing.append(f"{view_name}.{key}: {path}")
    if missing:
        raise FileNotFoundError("Missing required capture files:\n" + "\n".join(missing))


def ensure_mask(
    args: argparse.Namespace,
    target_object: str,
    view_name: str,
    view: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    mask_dir = output_dir / target_object / "masks" / view_name
    label = f"{target_object}_{view_name}"
    mask_path = mask_dir / f"{label}_mask.png"
    detection_path = mask_dir / f"{label}_detections.json"
    if args.skip_sam3:
        if not mask_path.exists() or not detection_path.exists():
            raise FileNotFoundError(f"Missing reused mask: {mask_path}")
    elif args.force_masks or not mask_path.exists() or not detection_path.exists():
        command = [
            "conda",
            "run",
            "-n",
            args.sam3_env,
            "python",
            "scripts/sam3_single_image_mask.py",
            "--image",
            str(view["rgb"]),
            "--prompt",
            OBJECT_PROMPTS[target_object],
            "--label",
            label,
            "--view-label",
            str(view["view_label"]),
            "--output",
            str(mask_dir),
            "--device",
            args.sam3_device,
        ]
        rc = run_command(command, mask_dir / "sam3.log")
        if rc != 0:
            write_json(
                detection_path,
                {
                    "status": "sam3_failed",
                    "returncode": rc,
                    "log": str(mask_dir / "sam3.log"),
                    "prompt": OBJECT_PROMPTS[target_object],
                    "image": str(Path(view["rgb"]).resolve()),
                },
            )
            raise RuntimeError(f"SAM3 failed for {target_object} in {view_name}: {mask_dir / 'sam3.log'}")
    detection = load_json(detection_path)
    return mask_path, detection


def view_records_for_object(
    args: argparse.Namespace,
    target_object: str,
    all_views: dict[str, dict[str, Any]],
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    primary_name = f"ee_{target_object}"
    ordered_names = [primary_name]
    if args.extra_view_set in ("all", "ee_only"):
        ordered_names.extend(f"ee_{name}" for name in OBJECTS if name != target_object)
    if args.extra_view_set == "all":
        ordered_names.append("video")
    records = []
    for view_name in ordered_names:
        view = all_views[view_name]
        mask_path, detection = ensure_mask(args, target_object, view_name, view, output_dir)
        records.append(
            {
                "view_name": view_name,
                "role": view["role"],
                "source_object": view.get("source_object"),
                "rgb": str(Path(view["rgb"]).resolve()),
                "depth_npy": str(Path(view["depth_npy"]).resolve()),
                "mask": str(mask_path.resolve()),
                "camera_json": str(Path(view["camera_json"]).resolve()),
                "view_label": view["view_label"],
                "sam3": {
                    "prompt": OBJECT_PROMPTS[target_object],
                    "mask_count": detection.get("mask_count"),
                    "best_index": detection.get("best_index"),
                    "areas_px": detection.get("areas_px"),
                    "scores": detection.get("scores"),
                    "detections_json": str((mask_path.parent / f"{target_object}_{view_name}_detections.json").resolve()),
                    "overlay": str((mask_path.parent / f"{target_object}_{view_name}_overlay.png").resolve()),
                },
            }
        )
    return records[0], records[1:], records


def openssl_env() -> dict[str, str]:
    env = os.environ.copy()
    openssl_lib = REPO_ROOT / "third_party/openssl11/lib"
    current_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{openssl_lib}:{current_ld}" if current_ld else str(openssl_lib)
    return env


def generator_command(
    args: argparse.Namespace,
    generator: str,
    primary: dict[str, Any],
    extras: list[dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    if generator == "anygrasp":
        command = [
            "conda",
            "run",
            "-n",
            args.anygrasp_env,
            "python",
            "scripts/anygrasp_from_rgbd_mask.py",
            "--rgb",
            primary["rgb"],
            "--depth-npy",
            primary["depth_npy"],
            "--mask",
            primary["mask"],
            "--camera-json",
            primary["camera_json"],
            "--output",
            str(output_dir),
            "--max-depth",
            str(args.max_depth),
            "--save-top-grasps",
            str(args.save_top_grasps),
            "--anygrasp-cloud-mode",
            args.anygrasp_cloud_mode,
            "--scene-cloud-stride",
            str(args.scene_cloud_stride),
            "--scene-cloud-max-points",
            str(args.scene_cloud_max_points),
            "--target-filter-voxel-size",
            str(args.target_filter_voxel_size),
            "--target-filter-min-points",
            str(args.target_filter_min_points),
        ]
    elif generator == "graspgen":
        command = [
            "conda",
            "run",
            "-n",
            args.graspgen_env,
            "python",
            "scripts/graspgen_from_rgbd_mask.py",
            "--rgb",
            primary["rgb"],
            "--depth-npy",
            primary["depth_npy"],
            "--mask",
            primary["mask"],
            "--camera-json",
            primary["camera_json"],
            "--output",
            str(output_dir),
            "--max-depth",
            str(args.max_depth),
            "--save-top-grasps",
            str(args.save_top_grasps),
            "--scene-cloud-stride",
            str(args.scene_cloud_stride),
            "--scene-cloud-max-points",
            str(args.scene_cloud_max_points),
            "--target-cloud-max-points",
            str(args.graspgen_target_cloud_max_points),
            "--num-grasps",
            str(args.graspgen_num_grasps),
            "--export-tcp-offset",
            str(args.graspgen_export_tcp_offset),
            "--target-filter-voxel-size",
            str(args.target_filter_voxel_size),
            "--target-filter-min-points",
            str(args.target_filter_min_points),
        ]
    else:
        raise ValueError(generator)
    if not args.filter_target_outliers:
        command.append("--no-filter-target-outliers")
    for extra in extras:
        command.extend(
            [
                "--extra-view",
                ",".join([extra["rgb"], extra["depth_npy"], extra["mask"], extra["camera_json"]]),
            ]
        )
    return command


def run_generator(
    args: argparse.Namespace,
    generator: str,
    target_object: str,
    primary: dict[str, Any],
    extras: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    generator_dir = output_dir / target_object / generator
    result_name = f"{generator}_result.json" if generator == "graspgen" else "anygrasp_result.json"
    result_path = generator_dir / result_name
    if args.force_generators or not result_path.exists():
        generator_dir.mkdir(parents=True, exist_ok=True)
        command = generator_command(args, generator, primary, extras, generator_dir)
        rc = run_command(command, generator_dir / f"{generator}.log", env=openssl_env())
        if rc != 0 and not result_path.exists():
            write_json(
                result_path,
                {
                    "grasp_generator": generator,
                    generator: {
                        "status": "runner_failed",
                        "returncode": rc,
                        "log": str(generator_dir / f"{generator}.log"),
                    },
                },
            )
    result = load_json(result_path)
    payload = result.get(generator) or result.get("anygrasp") or result.get("graspgen") or {}
    stale_error = generator_dir / "final_grasp_pose_error.json"
    if payload.get("status") == "ok" and (generator_dir / "final_grasp_pose.json").exists() and stale_error.exists():
        stale_error.unlink()
    if not args.skip_viz and (generator_dir / "masked_cloud.npy").exists():
        viz_dir = generator_dir / "fused_pc_grasp_viz"
        viz_command = [
            "python",
            "scripts/visualize_anygrasp_fused_pc.py",
            "--anygrasp-dir",
            str(generator_dir),
            "--mode",
            "top",
            "--top-k",
            str(min(args.save_top_grasps, 10)),
            "--output-dir",
            str(viz_dir),
        ]
        run_command(viz_command, viz_dir / "visualize.log")
    return result


def summarize_generator(result: dict[str, Any], generator: str) -> dict[str, Any]:
    payload = result.get(generator) or result.get("anygrasp") or result.get("graspgen") or {}
    best = payload.get("best") or {}
    return {
        "status": payload.get("status"),
        "point_cloud_frame": result.get("point_cloud_frame"),
        "fused_view_count": result.get("fused_view_count"),
        "target_point_count": result.get("target_point_count") or result.get("point_count"),
        "input_point_count": result.get("anygrasp_input_point_count") or result.get("graspgen_input_point_count"),
        "top_grasp_count": len(payload.get("top_grasps") or []),
        "best_score": best.get("score"),
        "best_translation_camera": best.get("translation_camera"),
        "best_rotation_matrix_camera": best.get("rotation_matrix_camera"),
        "final_grasp_pose_frame": (payload.get("final_grasp_pose") or result.get("final_grasp_pose") or {}).get("frame"),
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input.expanduser().resolve()
    output_dir = (args.output or (input_dir / "grasp_fusion_debug")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    views = capture_views(input_dir)
    assert_view_files(views)
    summary: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "extra_view_set": args.extra_view_set,
        "target_outlier_filter": {
            "enabled": bool(args.filter_target_outliers),
            "voxel_size_m": float(args.target_filter_voxel_size),
            "min_points": int(args.target_filter_min_points),
        },
        "view_contract": {
            "primary": "target object's actual EE camera image from the real arm pose",
            "extras": {
                "all": "other actual EE camera poses plus the shared video camera view",
                "ee_only": "other actual EE camera poses only",
                "primary_only": "no extra views; target object's actual EE camera image only",
            }[args.extra_view_set],
            "point_cloud_frame": "primary_camera, which is the target object's EE camera frame",
        },
        "objects": {},
    }

    for target_object in args.objects:
        primary, extras, records = view_records_for_object(args, target_object, views, output_dir)
        object_summary: dict[str, Any] = {
            "primary_view": primary,
            "extra_views": extras,
            "all_views": records,
            "generators": {},
        }
        if not args.skip_anygrasp:
            anygrasp = run_generator(args, "anygrasp", target_object, primary, extras, output_dir)
            object_summary["generators"]["anygrasp"] = summarize_generator(anygrasp, "anygrasp")
            object_summary["generators"]["anygrasp"]["dir"] = str((output_dir / target_object / "anygrasp").resolve())
        if not args.skip_graspgen:
            graspgen = run_generator(args, "graspgen", target_object, primary, extras, output_dir)
            object_summary["generators"]["graspgen"] = summarize_generator(graspgen, "graspgen")
            object_summary["generators"]["graspgen"]["dir"] = str((output_dir / target_object / "graspgen").resolve())
        summary["objects"][target_object] = object_summary
        write_json(output_dir / "summary.json", summary)

    write_json(output_dir / "summary.json", summary)
    print(f"[INFO] Saved {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
