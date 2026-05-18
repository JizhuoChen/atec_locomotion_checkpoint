#!/usr/bin/env python3
"""Create and visualize a pseudo grasp pose for a Task E target object."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "outputs/task_e_banana_pipeline/latest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Task E banana pipeline output directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Defaults to <input>/pseudo_grasp.",
    )
    parser.add_argument(
        "--grasp-height",
        type=float,
        default=0.035,
        help="Z offset above detected target center for the pseudo grasp.",
    )
    parser.add_argument(
        "--pregrasp-height",
        type=float,
        default=0.20,
        help="Z offset above detected target center for the pre-grasp waypoint.",
    )
    parser.add_argument(
        "--lift-height",
        type=float,
        default=0.24,
        help="Z offset above detected target center for the lift waypoint.",
    )
    parser.add_argument(
        "--pose-source",
        choices=("pipeline", "sam3-mask", "ground-truth"),
        default="pipeline",
        help="Where to get the banana center/axis. Use sam3-mask with --sam3-mask after prompt testing.",
    )
    parser.add_argument(
        "--sam3-mask",
        type=Path,
        default=None,
        help="Binary SAM3 mask to back-project with video_depth.npy and video_camera.json.",
    )
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument(
        "--grasp-xy-offset",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("DX", "DY"),
        help="World XY offset added to the selected banana center for pseudo grasp testing.",
    )
    parser.add_argument(
        "--quat-source",
        choices=("axis", "default-topdown"),
        default="axis",
        help="Use mask/object axis orientation or the existing Task E default top-down quaternion.",
    )
    return parser.parse_args()


def resolve_input(path: Path) -> Path:
    if path.exists():
        return path.resolve()
    if path.name == "latest":
        latest_txt = path.with_name("latest.txt")
        if latest_txt.exists():
            return Path(latest_txt.read_text(encoding="utf-8").strip()).resolve()
    raise FileNotFoundError(f"Task E pipeline output does not exist: {path}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0:
        return np.eye(3)
    w, x, y, z = quat / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_wxyz(matrix: np.ndarray) -> list[float]:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(m)))
        if idx == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return quat.astype(float).tolist()


def make_grasp_rotation(long_axis_xy: np.ndarray) -> np.ndarray:
    x_axis = np.asarray([long_axis_xy[0], long_axis_xy[1], 0.0], dtype=np.float64)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def project_world(point_w: np.ndarray, camera: dict) -> tuple[float, float] | None:
    intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float64)
    pos_w = np.asarray(camera["pos_w"], dtype=np.float64)
    rot_wc = quat_wxyz_to_matrix(np.asarray(camera["quat_w_ros"], dtype=np.float64))
    point_c = rot_wc.T @ (np.asarray(point_w, dtype=np.float64) - pos_w)
    if point_c[2] <= 1e-6:
        return None
    u = intrinsic[0, 0] * point_c[0] / point_c[2] + intrinsic[0, 2]
    v = intrinsic[1, 1] * point_c[1] / point_c[2] + intrinsic[1, 2]
    return float(u), float(v)


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


def transform_points_to_world(points_cam: np.ndarray, camera: dict) -> np.ndarray:
    rot_wc = quat_wxyz_to_matrix(np.asarray(camera["quat_w_ros"], dtype=np.float64))
    pos_w = np.asarray(camera["pos_w"], dtype=np.float64)
    return points_cam @ rot_wc.T + pos_w


def estimate_pose_from_mask(mask_path: Path, input_dir: Path, camera: dict, max_depth: float) -> dict:
    depth = np.load(input_dir / "video_depth.npy")
    mask = load_mask(mask_path, depth.shape)
    intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float32)
    points_cam_all = backproject(depth, intrinsic)
    valid = mask & np.isfinite(depth) & (depth > 0.0) & (depth < max_depth)
    if not valid.any():
        raise ValueError(f"SAM3 mask has no valid depth pixels: {mask_path}")

    points_world = transform_points_to_world(points_cam_all[valid], camera)
    center = np.median(points_world, axis=0)
    xy = points_world[:, :2]
    if len(xy) >= 3:
        cov = np.cov((xy - xy.mean(axis=0)).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]
    else:
        axis = np.array([1.0, 0.0], dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-6)
    ys, xs = np.where(valid)
    return {
        "center_world": center.astype(float).tolist(),
        "principal_axis_xy": axis.astype(float).tolist(),
        "yaw_rad": float(np.arctan2(axis[1], axis[0])),
        "valid_depth_points": int(valid.sum()),
        "pixel_center": [float(np.median(xs)), float(np.median(ys))],
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "source": f"sam3_mask_depth:{mask_path}",
    }


def select_pose(args: argparse.Namespace, input_dir: Path, pipeline: dict, camera: dict) -> dict:
    if args.pose_source == "pipeline":
        pose = dict(pipeline["banana_pose_from_video"])
        pose.setdefault("source", "pipeline:banana_pose_from_video")
        return pose
    if args.pose_source == "ground-truth":
        gt = pipeline.get("ground_truth_banana_pose")
        if not gt:
            raise ValueError("ground_truth_banana_pose is not present in pipeline.json")
        pose = {
            "center_world": gt["center_world"],
            "principal_axis_xy": [1.0, 0.0],
            "yaw_rad": 0.0,
            "valid_depth_points": 0,
            "pixel_center": None,
            "bbox_xyxy": None,
            "source": "pipeline:ground_truth_banana_pose",
        }
        return pose

    if args.sam3_mask is None:
        raise ValueError("--pose-source sam3-mask requires --sam3-mask")
    mask_path = args.sam3_mask if args.sam3_mask.is_absolute() else (Path.cwd() / args.sam3_mask)
    max_depth = float(args.max_depth if args.max_depth is not None else pipeline.get("max_depth", 2.0))
    return estimate_pose_from_mask(mask_path, input_dir, camera, max_depth)


def draw_cross(draw: ImageDraw.ImageDraw, point: tuple[float, float], color: tuple[int, int, int], radius: int = 8) -> None:
    x, y = point
    draw.line((x - radius, y, x + radius, y), fill=color, width=3)
    draw.line((x, y - radius, x, y + radius), fill=color, width=3)


def draw_projected_line(
    draw: ImageDraw.ImageDraw,
    camera: dict,
    a_w: np.ndarray,
    b_w: np.ndarray,
    color: tuple[int, int, int],
    width: int = 3,
) -> None:
    a = project_world(a_w, camera)
    b = project_world(b_w, camera)
    if a is not None and b is not None:
        draw.line((*a, *b), fill=color, width=width)


def draw_video_overlay(input_dir: Path, output_dir: Path, pseudo: dict, camera: dict, pose: dict) -> None:
    image = Image.open(input_dir / "video_rgb.png").convert("RGB")
    draw = ImageDraw.Draw(image)

    bbox = pose.get("bbox_xyxy")
    if bbox:
        draw.rectangle(tuple(bbox), outline=(255, 255, 0), width=3)

    center = np.asarray(pseudo["target_center_w"], dtype=np.float64)
    grasp = np.asarray(pseudo["grasp_pose_w"]["position"], dtype=np.float64)
    pre = np.asarray(pseudo["pregrasp_pose_w"]["position"], dtype=np.float64)
    lift = np.asarray(pseudo["lift_pose_w"]["position"], dtype=np.float64)
    long_axis = np.asarray(pseudo["banana_long_axis_w"], dtype=np.float64)
    cross_axis = np.asarray(pseudo["jaw_closing_axis_w"], dtype=np.float64)

    projected = {
        "center": project_world(center, camera),
        "grasp": project_world(grasp, camera),
        "pre": project_world(pre, camera),
        "lift": project_world(lift, camera),
    }
    for label, point, color in [
        ("center", projected["center"], (255, 255, 255)),
        ("grasp", projected["grasp"], (255, 60, 60)),
        ("pre", projected["pre"], (60, 180, 255)),
        ("lift", projected["lift"], (60, 255, 120)),
    ]:
        if point is not None:
            draw_cross(draw, point, color)
            draw.text((point[0] + 10, point[1] - 10), label, fill=color)

    draw_projected_line(draw, camera, pre, grasp, (60, 180, 255), width=3)
    draw_projected_line(draw, camera, grasp, lift, (60, 255, 120), width=3)
    draw_projected_line(draw, camera, grasp - long_axis * 0.08, grasp + long_axis * 0.08, (255, 180, 40), width=4)
    draw_projected_line(draw, camera, grasp - cross_axis * 0.055, grasp + cross_axis * 0.055, (255, 60, 180), width=4)

    draw.rectangle((0, 0, image.width, 34), fill=(0, 0, 0))
    draw.text((8, 9), "Pseudo banana grasp: yellow=banana axis, magenta=jaw closing axis", fill=(255, 255, 255))
    image.save(output_dir / "pseudo_grasp_video_overlay.png")


def draw_cloud_views(output_dir: Path, pseudo: dict) -> None:
    points_path = output_dir.parent / "anygrasp/masked_cloud.npy"
    colors_path = output_dir.parent / "anygrasp/masked_cloud_colors.npy"
    if not points_path.exists() or not colors_path.exists():
        return

    points = np.load(points_path)
    colors = np.load(colors_path)
    if len(points) == 0:
        return
    # Cloud is in EE camera frame. This visualization is diagnostic only, so
    # place the marker at the masked-cloud median.
    marker = np.median(points, axis=0)
    canvas = Image.new("RGB", (1260, 520), (18, 22, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), "Masked EE RGB-D cloud with pseudo grasp center marker", fill=(255, 255, 255))

    def draw_panel(box: tuple[int, int, int, int], axes: tuple[int, int], title: str) -> None:
        left, top, right, bottom = box
        draw.rectangle(box, outline=(80, 80, 80), width=1)
        draw.text((left + 8, top + 6), title, fill=(255, 255, 255))
        pts = points[:, list(axes)]
        max_points = 9000
        if len(pts) > max_points:
            idx = np.linspace(0, len(pts) - 1, max_points).astype(np.int64)
            pts = pts[idx]
            rgb = colors[idx]
        else:
            rgb = colors
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        span = np.maximum(maxs - mins, 1e-5)
        pad = span * 0.08
        mins -= pad
        span = maxs + pad - mins
        px = left + 8 + (pts[:, 0] - mins[0]) / span[0] * (right - left - 16)
        py = bottom - 8 - (pts[:, 1] - mins[1]) / span[1] * (bottom - top - 32)
        rgb255 = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        for x, y, color in zip(px.astype(int), py.astype(int), rgb255, strict=False):
            draw.point((int(x), int(y)), fill=tuple(int(c) for c in color))
        m = marker[list(axes)]
        mx = left + 8 + (m[0] - mins[0]) / span[0] * (right - left - 16)
        my = bottom - 8 - (m[1] - mins[1]) / span[1] * (bottom - top - 32)
        draw.ellipse((mx - 6, my - 6, mx + 6, my + 6), outline=(255, 60, 60), width=3)

    draw_panel((20, 44, 410, 500), (0, 1), "camera XY")
    draw_panel((435, 44, 825, 500), (0, 2), "camera XZ")
    draw_panel((850, 44, 1240, 500), (1, 2), "camera YZ")
    canvas.save(output_dir / "pseudo_grasp_cloud_overlay.png")


def main() -> None:
    args = parse_args()
    input_dir = resolve_input(args.input)
    output_dir = (args.output or input_dir / "pseudo_grasp").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = load_json(input_dir / "pipeline.json")
    video_camera = load_json(input_dir / "video_camera.json")["camera"]
    pose = select_pose(args, input_dir, pipeline, video_camera)
    center = np.asarray(pose["center_world"], dtype=np.float64)
    center[:2] += np.asarray(args.grasp_xy_offset, dtype=np.float64)
    long_axis = np.asarray([*pose.get("principal_axis_xy", [1.0, 0.0]), 0.0], dtype=np.float64)
    if np.linalg.norm(long_axis) < 1e-6:
        long_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    long_axis /= np.linalg.norm(long_axis)
    jaw_axis = np.asarray([-long_axis[1], long_axis[0], 0.0], dtype=np.float64)
    jaw_axis /= np.linalg.norm(jaw_axis)

    if args.quat_source == "default-topdown":
        quat = [0.0, 1.0, 0.0, 0.0]
    else:
        rotation = make_grasp_rotation(long_axis[:2])
        quat = matrix_to_quat_wxyz(rotation)
    grasp = center + np.array([0.0, 0.0, args.grasp_height], dtype=np.float64)
    pre = center + np.array([0.0, 0.0, args.pregrasp_height], dtype=np.float64)
    lift = center + np.array([0.0, 0.0, args.lift_height], dtype=np.float64)
    hover = pipeline.get("hover", {})

    pseudo = {
        "source_dir": str(input_dir),
        "target": "banana",
        "pose_source": pose.get("source", args.pose_source),
        "grasp_xy_offset_w": [float(args.grasp_xy_offset[0]), float(args.grasp_xy_offset[1])],
        "quat_source": args.quat_source,
        "target_center_w": center.astype(float).tolist(),
        "banana_long_axis_w": long_axis.astype(float).tolist(),
        "jaw_closing_axis_w": jaw_axis.astype(float).tolist(),
        "approach_axis_w": [0.0, 0.0, -1.0],
        "grasp_pose_w": {"position": grasp.astype(float).tolist(), "quat_wxyz": quat},
        "pregrasp_pose_w": {"position": pre.astype(float).tolist(), "quat_wxyz": quat},
        "lift_pose_w": {"position": lift.astype(float).tolist(), "quat_wxyz": quat},
        "camera_look_pose_w": {
            "position": hover.get("desired_gripper_pos_w") or hover.get("hover_pos_w"),
            "quat_wxyz": hover.get("desired_gripper_quat_wxyz"),
        },
        "gripper": {
            "open_joint7_joint8": [0.035, -0.035],
            "closed_joint7_joint8": [-0.015, 0.015],
        },
        "note": "Pseudo top-down banana grasp; use for IK/plumbing tests while AnyGrasp license is pending.",
    }

    (output_dir / "pseudo_grasp.json").write_text(json.dumps(pseudo, indent=2), encoding="utf-8")
    draw_video_overlay(input_dir, output_dir, pseudo, video_camera, pose)
    draw_cloud_views(output_dir, pseudo)
    print(f"[INFO] Saved {output_dir / 'pseudo_grasp.json'}")
    print(f"[INFO] Saved {output_dir / 'pseudo_grasp_video_overlay.png'}")
    if (output_dir / "pseudo_grasp_cloud_overlay.png").exists():
        print(f"[INFO] Saved {output_dir / 'pseudo_grasp_cloud_overlay.png'}")


if __name__ == "__main__":
    main()
