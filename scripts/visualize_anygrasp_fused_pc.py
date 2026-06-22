#!/usr/bin/env python3
"""Visualize a fused AnyGrasp point cloud with grasp gripper geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


GRIPPER_COLORS = {
    "palm_base": (245, 245, 245),
    "left_finger_body": (64, 220, 255),
    "right_finger_body": (64, 220, 255),
    "left_fingertip_pad": (255, 80, 220),
    "right_fingertip_pad": (255, 80, 220),
    "finger_centerline": (255, 185, 55),
    "offset_base_reference": (120, 255, 120),
}


def parse_vec3(value: str) -> list[float]:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anygrasp-dir", "--grasp-dir", dest="anygrasp_dir", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("selected", "final", "top", "both"),
        default="selected",
        help="Which grasps to draw. selected uses ik_candidate_selection.json when available.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--camera-json", type=Path, default=None)
    parser.add_argument("--object-center-world", type=parse_vec3, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--tool-transform",
        choices=("identity", "graspnet_to_piper_z"),
        default="graspnet_to_piper_z",
    )
    parser.add_argument(
        "--offset-distance",
        "--gripper-base-offset",
        dest="offset_distance",
        type=float,
        default=0.09,
        help=(
            "Distance in metres applied to the visualized gripper pose by --offset-mode. "
            "With --offset-mode towards_object_center, this offsets from the grasp/contact "
            "pose toward the object center. The green line shows the applied translation. "
            "--gripper-base-offset is kept as a deprecated alias."
        ),
    )
    parser.add_argument(
        "--offset-mode",
        choices=("none", "approach_axis", "finger_centerline", "yellow_line", "towards_object_center"),
        default="towards_object_center",
        help=(
            "Direction for the offset applied to the visualized gripper pose. "
            "none disables the offset; approach_axis offsets opposite the drawn finger centerline; "
            "finger_centerline/yellow_line offsets along the drawn yellow finger centerline; "
            "towards_object_center offsets toward the segmented object center."
        ),
    )
    parser.add_argument(
        "--palm-width",
        type=float,
        default=0.085,
        help="Deprecated compatibility option; real-scale finger boxes now define the palm/root crossbars.",
    )
    parser.add_argument(
        "--finger-length",
        type=float,
        default=0.0765,
        help=(
            "Fixed visualized Piper finger length in metres. Default 0.0765 "
            "comes from the official link7/link8 STL bounds."
        ),
    )
    parser.add_argument(
        "--finger-width",
        type=float,
        default=0.0265,
        help=(
            "Piper finger thickness along the opening/jaw axis in metres. "
            "Default 0.0265 comes from the official link7/link8 STL bounds."
        ),
    )
    parser.add_argument(
        "--finger-depth",
        type=float,
        default=0.056,
        help=(
            "Piper finger width along the side axis in metres. Default 0.056 "
            "comes from the official link7/link8 STL bounds."
        ),
    )
    parser.add_argument(
        "--finger-pad-length",
        type=float,
        default=0.05,
        help="Deprecated compatibility option; fingertip pads now use the real finger width/depth rectangle.",
    )
    parser.add_argument("--show", action="store_true", help="Open an interactive Open3D viewer when available.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def transform_points_from_world(points_w: np.ndarray, camera: dict) -> np.ndarray:
    r_wc = quat_wxyz_to_matrix(np.asarray(camera["quat_w_ros"], dtype=np.float64))
    t_wc = np.asarray(camera["pos_w"], dtype=np.float64)
    return (np.asarray(points_w, dtype=np.float64) - t_wc) @ r_wc


def rotation_world_to_camera(rotation_w: np.ndarray, camera: dict) -> np.ndarray:
    r_wc = quat_wxyz_to_matrix(np.asarray(camera["quat_w_ros"], dtype=np.float64))
    return r_wc.T @ np.asarray(rotation_w, dtype=np.float64)


def normalize(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        return fallback.astype(np.float64)
    return value.astype(np.float64) / norm


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


def anygrasp_pose_in_camera(grasp: dict, camera: dict) -> tuple[np.ndarray, np.ndarray] | None:
    if grasp.get("translation_camera") is not None and grasp.get("rotation_matrix_camera") is not None:
        return (
            np.asarray(grasp["translation_camera"], dtype=np.float64),
            np.asarray(grasp["rotation_matrix_camera"], dtype=np.float64),
        )
    world = grasp.get("pose_world") or grasp
    translation_w = world.get("translation") or grasp.get("translation_world")
    rotation_w = world.get("rotation_matrix") or grasp.get("rotation_matrix_world")
    if translation_w is None or rotation_w is None:
        return None
    translation_c = transform_points_from_world(np.asarray([translation_w], dtype=np.float64), camera)[0]
    rotation_c = rotation_world_to_camera(np.asarray(rotation_w, dtype=np.float64), camera)
    return translation_c, rotation_c


def piper_rotation(rotation: np.ndarray, tool_transform: str) -> np.ndarray:
    if tool_transform == "graspnet_to_piper_z":
        return np.stack([-rotation[:, 2], rotation[:, 1], rotation[:, 0]], axis=1)
    return rotation


def gripper_geometry(
    contact: np.ndarray,
    rotation: np.ndarray,
    width: float,
    args: argparse.Namespace,
    object_center_camera: np.ndarray,
) -> tuple[list[np.ndarray], list[dict]]:
    piper_rot = piper_rotation(rotation, args.tool_transform)
    jaw_axis = normalize(piper_rot[:, 1], np.array([0.0, 1.0, 0.0], dtype=np.float64))
    side_axis = normalize(piper_rot[:, 0], np.array([1.0, 0.0, 0.0], dtype=np.float64))
    approach_axis = normalize(piper_rot[:, 2], np.array([0.0, 0.0, -1.0], dtype=np.float64))
    jaw_width = float(np.clip(width if np.isfinite(width) and width > 0 else 0.065, 0.030, 0.095))
    if args.offset_mode == "none":
        offset_axis = np.zeros(3, dtype=np.float64)
    elif args.offset_mode == "towards_object_center":
        offset_axis = normalize(object_center_camera - contact, -approach_axis)
    elif args.offset_mode in {"finger_centerline", "yellow_line"}:
        offset_axis = approach_axis
    else:
        offset_axis = -approach_axis
    raw_contact = contact
    visualized_contact = raw_contact + offset_axis * max(0.0, float(args.offset_distance))

    finger_length = max(0.005, float(args.finger_length))
    finger_width = max(0.003, float(args.finger_width))
    finger_depth = max(0.003, float(args.finger_depth))
    finger_root = visualized_contact - approach_axis * finger_length

    points: list[np.ndarray] = []
    lines: list[dict] = []

    def add_line(start: int, end: int, label: str, color_key: str, width_px: int, **extra) -> None:
        lines.append(
            {
                "indices": (start, end),
                "label": label,
                "color": GRIPPER_COLORS[color_key],
                "width": width_px,
                **extra,
            }
        )

    finger_root_center_indices = []
    for side_sign, name in [(-1.0, "left"), (1.0, "right")]:
        center_offset = jaw_axis * side_sign * (jaw_width * 0.5 + finger_width * 0.5)
        start_idx = len(points)
        # corner order: root/tip, inner/outer jaw, side -/+
        for along in (0.0, 1.0):
            center = finger_root * (1.0 - along) + visualized_contact * along + center_offset
            for jaw_sign in (-1.0, 1.0):
                for side in (-1.0, 1.0):
                    points.append(
                        center
                        + jaw_axis * jaw_sign * finger_width * 0.5
                        + side_axis * side * finger_depth * 0.5
                    )
        root = [start_idx + idx for idx in range(4)]
        tip = [start_idx + 4 + idx for idx in range(4)]
        finger_root_center_indices.append((root, tip))

        # Root and length edges show the actual finger body.
        for a, b in ((0, 1), (1, 3), (3, 2), (2, 0)):
            add_line(root[a], root[b], f"{name}_finger_root", f"{name}_finger_body", 3)
        for a, b in ((0, 4), (1, 5), (2, 6), (3, 7)):
            add_line(start_idx + a, start_idx + b, f"{name}_finger_body", f"{name}_finger_body", 4)

        # Tip-face rectangle is the fingertip pad at the object/contact end.
        for a, b in ((0, 1), (1, 3), (3, 2), (2, 0)):
            add_line(tip[a], tip[b], f"{name}_fingertip_pad", f"{name}_fingertip_pad", 5)

    root_center_idx = len(points)
    points.append(finger_root)
    visualized_contact_idx = len(points)
    points.append(visualized_contact)
    raw_contact_idx = len(points)
    points.append(raw_contact)

    add_line(root_center_idx, visualized_contact_idx, "finger_centerline", "finger_centerline", 2)
    add_line(
        raw_contact_idx,
        visualized_contact_idx,
        "applied_offset",
        "offset_base_reference",
        2,
        draw_if_nonzero=True,
    )

    # Concise palm/base cues: two crossbars between the roots of the fingers.
    left_root, _ = finger_root_center_indices[0]
    right_root, _ = finger_root_center_indices[1]
    add_line(left_root[0], right_root[2], "palm_base", "palm_base", 3)
    add_line(left_root[1], right_root[3], "palm_base", "palm_base", 3)
    return points, lines


def selected_grasps(anygrasp_dir: Path, camera: dict, mode: str, top_k: int) -> list[dict]:
    grasps: list[dict] = []
    top_path = anygrasp_dir / "top_grasps.json"
    top = load_json(top_path) if top_path.exists() else []
    selection_path = anygrasp_dir / "ik_candidate_selection.json"
    final_path = anygrasp_dir / "final_grasp_pose.json"

    if mode in {"top", "both"}:
        for grasp in top[: max(0, int(top_k))]:
            grasps.append({"label": f"rank_{grasp.get('rank', len(grasps) + 1)}", "source": "top", **grasp})

    if mode in {"selected", "both"} and selection_path.exists():
        selection = load_json(selection_path)
        rank = selection.get("selected_rank")
        grasp = next((item for item in top if int(item.get("rank", -1)) == int(rank or -999)), None)
        if grasp is not None:
            grasps.append({"label": f"selected_rank_{rank}", "source": "selected", **grasp})

    if (mode in {"final", "both"} or not grasps) and final_path.exists():
        final = load_json(final_path)
        if anygrasp_pose_in_camera(final, camera) is not None:
            grasps.append({"label": "final", "source": "final", **final})
    return grasps


def write_gripper_obj(path: Path, gripper_points: list[np.ndarray], gripper_lines: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for point in gripper_points:
            f.write(f"v {point[0]:.7f} {point[1]:.7f} {point[2]:.7f}\n")
        for line in gripper_lines:
            start, end = line["indices"]
            f.write(f"l {start + 1} {end + 1}\n")


def sample_segment(start: np.ndarray, end: np.ndarray, color: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    length = float(np.linalg.norm(end - start))
    count = max(6, int(length / 0.003))
    weights = np.linspace(0.0, 1.0, count, dtype=np.float64)[:, None]
    points = start[None, :] * (1.0 - weights) + end[None, :] * weights
    colors = np.tile(np.asarray(color, dtype=np.float64)[None, :] / 255.0, (count, 1))
    return points, colors


def gripper_sample_cloud(
    gripper_points: list[np.ndarray],
    gripper_lines: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    if not gripper_points or not gripper_lines:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    point_batches = []
    color_batches = []
    for line in gripper_lines:
        start_idx, end_idx = line["indices"]
        if line.get("draw_if_nonzero") and np.linalg.norm(gripper_points[end_idx] - gripper_points[start_idx]) <= 1e-6:
            continue
        points, colors = sample_segment(
            gripper_points[start_idx],
            gripper_points[end_idx],
            tuple(line["color"]),
        )
        point_batches.append(points)
        color_batches.append(colors)
    return np.concatenate(point_batches, axis=0), np.concatenate(color_batches, axis=0)


def draw_projection_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    points: np.ndarray,
    colors: np.ndarray,
    gripper_points: list[np.ndarray],
    gripper_lines: list[dict],
    axes: tuple[int, int],
    title: str,
) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline=(80, 80, 80), width=1)
    draw.text((left + 8, top + 6), title, fill=(255, 255, 255))
    all_points = points
    if gripper_points:
        all_points = np.concatenate([points, np.asarray(gripper_points, dtype=np.float64)], axis=0)
    selected = all_points[:, list(axes)]
    mins = selected.min(axis=0)
    maxs = selected.max(axis=0)
    span = np.maximum(maxs - mins, 1e-4)
    mins -= span * 0.08
    maxs += span * 0.08
    span = maxs - mins

    sample = points
    sample_colors = colors
    if len(sample) > 20000:
        idx = np.linspace(0, len(sample) - 1, 20000).astype(np.int64)
        sample = sample[idx]
        sample_colors = sample_colors[idx]
    xy = sample[:, list(axes)]
    px = left + 8 + (xy[:, 0] - mins[0]) / span[0] * (right - left - 16)
    py = bottom - 8 - (xy[:, 1] - mins[1]) / span[1] * (bottom - top - 32)
    rgb = np.clip(sample_colors * 255.0, 0, 255).astype(np.uint8)
    for x, y, color in zip(px.astype(int), py.astype(int), rgb, strict=False):
        draw.point((int(x), int(y)), fill=tuple(int(c) for c in color))

    def project(point: np.ndarray) -> tuple[float, float]:
        pair = point[list(axes)]
        return (
            left + 8 + (pair[0] - mins[0]) / span[0] * (right - left - 16),
            bottom - 8 - (pair[1] - mins[1]) / span[1] * (bottom - top - 32),
        )

    for line in gripper_lines:
        start, end = line["indices"]
        if line.get("draw_if_nonzero") and np.linalg.norm(gripper_points[end] - gripper_points[start]) <= 1e-6:
            continue
        p0 = project(gripper_points[start])
        p1 = project(gripper_points[end])
        width = int(line.get("width", 3))
        color = tuple(int(value) for value in line["color"])
        draw.line((*p0, *p1), fill=(0, 0, 0), width=width + 3)
        draw.line((*p0, *p1), fill=color, width=width)


def save_projection_png(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    gripper_points: list[np.ndarray],
    gripper_lines: list[dict],
) -> None:
    canvas = Image.new("RGB", (1260, 520), (18, 22, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), "Fused grasp point cloud with gripper geometry", fill=(255, 255, 255))
    legend = [
        ("palm/base", GRIPPER_COLORS["palm_base"]),
        ("finger bodies", GRIPPER_COLORS["left_finger_body"]),
        ("fingertip pads", GRIPPER_COLORS["left_fingertip_pad"]),
        ("finger length", GRIPPER_COLORS["finger_centerline"]),
        ("applied offset", GRIPPER_COLORS["offset_base_reference"]),
    ]
    x = 365
    for label, color in legend:
        draw.rectangle((x, 10, x + 12, 22), fill=color)
        draw.text((x + 18, 9), label, fill=(230, 235, 245))
        x += 145
    panels = [
        ((20, 44, 410, 500), (0, 1), "primary camera XY"),
        ((435, 44, 825, 500), (0, 2), "primary camera XZ"),
        ((850, 44, 1240, 500), (1, 2), "primary camera YZ"),
    ]
    for box, axes, title in panels:
        draw_projection_panel(draw, box, points, colors, gripper_points, gripper_lines, axes, title)
    canvas.save(path)


def maybe_show_open3d(
    points: np.ndarray,
    colors: np.ndarray,
    gripper_points: list[np.ndarray],
    gripper_lines: list[dict],
    gripper_sample_points: np.ndarray | None = None,
    gripper_sample_colors: np.ndarray | None = None,
) -> None:
    try:
        import open3d as o3d
    except Exception as exc:
        print(f"[WARN] Open3D viewer unavailable: {exc!r}")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0).astype(np.float64))
    geometries = [pcd]
    if gripper_sample_points is not None and gripper_sample_colors is not None and len(gripper_sample_points):
        gripper_pcd = o3d.geometry.PointCloud()
        gripper_pcd.points = o3d.utility.Vector3dVector(gripper_sample_points.astype(np.float64))
        gripper_pcd.colors = o3d.utility.Vector3dVector(
            np.clip(gripper_sample_colors, 0.0, 1.0).astype(np.float64)
        )
        geometries.append(gripper_pcd)
    if gripper_points and gripper_lines:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.asarray(gripper_points, dtype=np.float64))
        visible_lines = [
            line
            for line in gripper_lines
            if not line.get("draw_if_nonzero")
            or np.linalg.norm(gripper_points[line["indices"][1]] - gripper_points[line["indices"][0]]) > 1e-6
        ]
        line_indices = np.asarray([line["indices"] for line in visible_lines], dtype=np.int32)
        line_set.lines = o3d.utility.Vector2iVector(line_indices)
        line_set.colors = o3d.utility.Vector3dVector(
            np.asarray([line["color"] for line in visible_lines], dtype=np.float64) / 255.0
        )
        geometries.append(line_set)
    o3d.visualization.draw_geometries(geometries)


def load_result_summary(anygrasp_dir: Path) -> dict:
    for name in ("anygrasp_result.json", "graspgen_result.json"):
        path = anygrasp_dir / name
        if path.exists():
            return load_json(path)
    raise FileNotFoundError(f"missing anygrasp_result.json or graspgen_result.json in {anygrasp_dir}")


def main() -> None:
    args = parse_args()
    anygrasp_dir = args.anygrasp_dir.expanduser().resolve()
    output_dir = args.output_dir or (anygrasp_dir / "fused_pc_grasp_viz")
    output_dir.mkdir(parents=True, exist_ok=True)

    result = load_result_summary(anygrasp_dir)
    camera_path = args.camera_json or Path(result["camera_json"])
    camera_payload = load_json(camera_path.expanduser().resolve())
    camera = camera_payload.get("camera", camera_payload)
    points = np.load(anygrasp_dir / "masked_cloud.npy").astype(np.float64)
    colors = np.load(anygrasp_dir / "masked_cloud_colors.npy").astype(np.float64)

    overlay_center = (result.get("top_grasps_overlay_geometry") or {}).get("object_center_world")
    if args.object_center_world is not None:
        object_center_camera = transform_points_from_world(np.asarray([args.object_center_world]), camera)[0]
        object_center_source = "argument_world"
    elif overlay_center is not None and len(overlay_center) >= 3:
        object_center_camera = transform_points_from_world(np.asarray([overlay_center[:3]], dtype=np.float64), camera)[0]
        object_center_source = "anygrasp_result_overlay_object_center_world"
    else:
        object_center_camera = np.mean(points, axis=0)
        object_center_source = "masked_cloud_mean_camera"

    gripper_points: list[np.ndarray] = []
    gripper_lines: list[dict] = []
    drawn = []
    for grasp in selected_grasps(anygrasp_dir, camera, args.mode, args.top_k):
        pose = anygrasp_pose_in_camera(grasp, camera)
        if pose is None:
            continue
        contact, rotation = pose
        start_index = len(gripper_points)
        points_local, lines_local = gripper_geometry(
            contact,
            rotation,
            float(grasp.get("width", 0.065)),
            args,
            object_center_camera,
        )
        gripper_points.extend(points_local)
        for line in lines_local:
            start, end = line["indices"]
            gripper_lines.append(
                {
                    **line,
                    "indices": (start + start_index, end + start_index),
                }
            )
        root_center = points_local[-3]
        visualized_contact = points_local[-2]
        raw_contact = points_local[-1]
        finger_length = float(np.linalg.norm(visualized_contact - root_center))
        applied_offset_length = float(np.linalg.norm(visualized_contact - raw_contact))
        drawn.append(
            {
                "label": grasp.get("label"),
                "source": grasp.get("source"),
                "rank": grasp.get("rank"),
                "score": grasp.get("score"),
                "contact_camera": contact.astype(float).tolist(),
                "raw_contact_camera": raw_contact.astype(float).tolist(),
                "visualized_contact_camera": visualized_contact.astype(float).tolist(),
                "visualized_finger_length_m": finger_length,
                "applied_offset_length_m": applied_offset_length,
                "offset_base_to_contact_reference_length_m": applied_offset_length,
                "note": (
                    "finger bodies are cyan real-scale link7/link8 boxes; magenta rectangles "
                    "are fingertip pads at the offset-applied contact end; green shows the applied "
                    "translation from the raw grasp contact to the visualized gripper pose."
                ),
            }
        )

    gripper_sample_points, gripper_sample_colors = gripper_sample_cloud(gripper_points, gripper_lines)
    write_ply(output_dir / "fused_cloud_primary_camera.ply", points, colors)
    write_gripper_obj(output_dir / "grippers_primary_camera.obj", gripper_points, gripper_lines)
    if len(gripper_sample_points):
        write_ply(output_dir / "gripper_points_primary_camera.ply", gripper_sample_points, gripper_sample_colors)
        write_ply(
            output_dir / "fused_cloud_with_gripper_primary_camera.ply",
            np.concatenate([points, gripper_sample_points], axis=0),
            np.concatenate([colors, gripper_sample_colors], axis=0),
        )
    save_projection_png(output_dir / "fused_cloud_grasps_projection.png", points, colors, gripper_points, gripper_lines)
    generator = result.get("grasp_generator") or (result.get("graspgen") or result.get("anygrasp") or {}).get("generator")
    metadata = {
        "anygrasp_dir": str(anygrasp_dir),
        "grasp_dir": str(anygrasp_dir),
        "grasp_generator": generator,
        "point_cloud_frame": result.get("point_cloud_frame", "primary_camera"),
        "fused_view_count": result.get("fused_view_count"),
        "views": result.get("views"),
        "camera_json": str(camera_path),
        "mode": args.mode,
        "top_k": int(args.top_k),
        "tool_transform": args.tool_transform,
        "offset_mode": args.offset_mode,
        "offset_distance": float(args.offset_distance),
        "gripper_base_offset": float(args.offset_distance),
        "finger_length": float(args.finger_length),
        "finger_width": float(args.finger_width),
        "finger_depth": float(args.finger_depth),
        "finger_pad_length": float(args.finger_pad_length),
        "piper_dimension_source": (
            "Defaults are from AgileX Piper link7/link8 STL bounds: "
            "0.0765 m length, 0.0265 m opening-axis thickness, 0.056 m side width."
        ),
        "object_center_source": object_center_source,
        "object_center_camera": object_center_camera.astype(float).tolist(),
        "drawn_grasps": drawn,
        "outputs": {
            "cloud_ply": "fused_cloud_primary_camera.ply",
            "cloud_with_gripper_ply": "fused_cloud_with_gripper_primary_camera.ply" if len(gripper_sample_points) else None,
            "gripper_points_ply": "gripper_points_primary_camera.ply" if len(gripper_sample_points) else None,
            "grippers_obj": "grippers_primary_camera.obj",
            "projection_png": "fused_cloud_grasps_projection.png",
        },
    }
    (output_dir / "scene_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[INFO] Saved {output_dir / 'fused_cloud_primary_camera.ply'}")
    if len(gripper_sample_points):
        print(f"[INFO] Saved {output_dir / 'gripper_points_primary_camera.ply'}")
        print(f"[INFO] Saved {output_dir / 'fused_cloud_with_gripper_primary_camera.ply'}")
    print(f"[INFO] Saved {output_dir / 'grippers_primary_camera.obj'}")
    print(f"[INFO] Saved {output_dir / 'fused_cloud_grasps_projection.png'}")
    if args.show:
        maybe_show_open3d(
            points,
            colors,
            gripper_points,
            gripper_lines,
            gripper_sample_points,
            gripper_sample_colors,
        )


if __name__ == "__main__":
    main()
