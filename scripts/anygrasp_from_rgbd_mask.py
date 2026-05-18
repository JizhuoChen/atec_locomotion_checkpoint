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
    attempts = [
        {
            "name": "configured",
            "apply_object_mask": bool(args.apply_object_mask),
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


def serialize_best_grasp(best, camera: dict) -> dict:
    translation = np.asarray(best.translation, dtype=np.float64)
    rotation = np.asarray(best.rotation_matrix, dtype=np.float64)
    world_pose = transform_pose_to_world(camera, translation, rotation)
    return {
        "score": float(best.score),
        "width": float(best.width),
        "depth": float(best.depth),
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


def run_anygrasp(args: argparse.Namespace, points: np.ndarray, colors: np.ndarray, camera: dict) -> dict:
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
                    best = grasps[0]
                    best_payload = serialize_best_grasp(best, camera)
                    attempt_records.append(
                        {
                            **attempt,
                            "status": "ok",
                            "raw_count": raw_count,
                            "nms_count": nms_count,
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
                        "best": best_payload,
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
    valid = mask & np.isfinite(depth) & (depth > 0.0) & (depth < args.max_depth)
    points = points_organized[valid].astype(np.float32)
    colors = rgb[valid].astype(np.float32)

    point_cloud_path = args.output / "masked_cloud.npy"
    color_path = args.output / "masked_cloud_colors.npy"
    ply_path = args.output / "masked_cloud.ply"
    np.save(point_cloud_path, points)
    np.save(color_path, colors)
    if len(points) > 0:
        write_ply(ply_path, points, colors)

    if len(points) < 50:
        anygrasp = {
            "status": "insufficient_points",
            "point_count": int(len(points)),
            "reason": "Need at least 50 valid masked depth points.",
        }
        marker = None
    else:
        anygrasp = run_anygrasp(args, points, colors, camera)
        marker = None
        best = anygrasp.get("best") if isinstance(anygrasp, dict) else None
        if isinstance(best, dict) and best.get("translation_camera") is not None:
            marker = np.asarray(best["translation_camera"], dtype=np.float32)

    viz_path = args.output / "anygrasp_result.png"
    save_cloud_views(viz_path, points, colors, anygrasp.get("status", "unknown"), marker=marker)

    summary = {
        "rgb": str(args.rgb.resolve()),
        "depth_npy": str(args.depth_npy.resolve()),
        "mask": str(args.mask.resolve()),
        "camera_json": str(args.camera_json.resolve()),
        "point_count": int(len(points)),
        "point_cloud_npy": point_cloud_path.name,
        "point_cloud_colors_npy": color_path.name,
        "point_cloud_ply": ply_path.name if ply_path.exists() else None,
        "visualization": viz_path.name,
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


if __name__ == "__main__":
    main()
