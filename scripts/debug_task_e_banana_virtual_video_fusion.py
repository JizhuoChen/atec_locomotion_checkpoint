#!/usr/bin/env python3
"""Banana-only video/virtual-video point-cloud fusion diagnostic."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "outputs/task_e_ideal_ee_camera_debug/20260614_same_env_lookat_real_ee"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <source>/banana_virtual_video_fusion_debug/<timestamp>.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sam3-env", default="sam3_full")
    parser.add_argument("--sam3-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--debug-steps", type=int, default=4)
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


args_cli = parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401,E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from contact_graspnet_from_rgbd_mask import (  # noqa: E402
    load_json,
    transform_points_from_world,
    transform_points_to_world,
    view_points,
    write_ply,
)
from task_e_full_baseline_request import deterministic_object_poses  # noqa: E402


def output_dir() -> Path:
    if args_cli.output is not None:
        out = args_cli.output.expanduser().resolve()
    else:
        out = (
            args_cli.source.expanduser().resolve()
            / "banana_virtual_video_fusion_debug"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def to_numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def to_rgb_array(value) -> np.ndarray:
    array = to_numpy(value)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        if array.size and float(np.nanmax(array)) <= 1.0:
            array *= 255.0
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def to_depth_array(value) -> np.ndarray:
    array = to_numpy(value).astype(np.float32)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError(f"Expected depth image, got {array.shape}")
    return np.ascontiguousarray(array)


def tensor_to_list(value) -> list[float]:
    return value.detach().cpu().numpy().astype(float).tolist()


def camera_metadata(env, sensor_name: str) -> dict:
    camera = env.unwrapped.scene.sensors[sensor_name]
    data = camera.data
    return {
        "sensor": sensor_name,
        "image_shape": list(data.image_shape),
        "intrinsic_matrix": tensor_to_list(data.intrinsic_matrices[0]),
        "pos_w": tensor_to_list(data.pos_w[0]),
        "quat_w_world": tensor_to_list(data.quat_w_world[0]),
        "quat_w_ros": tensor_to_list(data.quat_w_ros[0]),
        "quat_w_opengl": tensor_to_list(data.quat_w_opengl[0]),
    }


def quat_wxyz_from_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / s,
                    0.25 * s,
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    (matrix[0, 2] + matrix[2, 0]) / s,
                ],
                dtype=np.float64,
            )
        elif axis == 1:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / s,
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    0.25 * s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                ],
                dtype=np.float64,
            )
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / s,
                    (matrix[0, 2] + matrix[2, 0]) / s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                    0.25 * s,
                ],
                dtype=np.float64,
            )
    norm = float(np.linalg.norm(quat))
    return quat / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def normalize_or_fallback(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        return np.asarray(fallback, dtype=np.float64)
    return np.asarray(value, dtype=np.float64) / norm


def lookat_camera_quat_wxyz(camera_pos_w: np.ndarray, target_pos_w: np.ndarray) -> np.ndarray:
    optical_axis = normalize_or_fallback(target_pos_w - camera_pos_w, np.array([1.0, 0.0, -1.0]))
    image_down_hint = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    image_down = image_down_hint - optical_axis * float(np.dot(image_down_hint, optical_axis))
    if np.linalg.norm(image_down) <= 1e-6:
        image_down_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        image_down = image_down_hint - optical_axis * float(np.dot(image_down_hint, optical_axis))
    image_down = normalize_or_fallback(image_down, np.array([0.0, 1.0, 0.0]))
    image_right = normalize_or_fallback(np.cross(image_down, optical_axis), np.array([1.0, 0.0, 0.0]))
    image_down = normalize_or_fallback(np.cross(optical_axis, image_right), image_down)
    rot = np.stack([image_right, image_down, optical_axis], axis=1)
    return quat_wxyz_from_matrix(rot)


def camera_rgb(sensor) -> np.ndarray:
    output = sensor.data.output
    value = output.get("rgb")
    if value is None:
        value = output.get("rgba")
    if value is None:
        raise KeyError(f"Camera has no rgb/rgba output. Keys: {list(output.keys())}")
    return to_rgb_array(value)


def camera_depth(sensor) -> np.ndarray:
    value = sensor.data.output.get("depth")
    if value is None:
        raise KeyError(f"Camera has no depth output. Keys: {list(sensor.data.output.keys())}")
    return to_depth_array(value)


def add_camera_cfg(env_cfg, sensor_name: str, pos_w: np.ndarray, quat_w_ros: np.ndarray) -> str:
    setattr(
        env_cfg.scene,
        sensor_name,
        CameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{sensor_name}",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=tuple(float(v) for v in pos_w),
                rot=tuple(float(v) for v in quat_w_ros),
                convention="ros",
            ),
        ),
    )
    return sensor_name


def create_env(source_dir: Path):
    video_camera_payload = load_json(source_dir / "video_camera.json")
    video_camera = video_camera_payload.get("camera", video_camera_payload)
    object_poses = deterministic_object_poses(args_cli.seed)
    banana_center = np.asarray(object_poses["banana"]["center_w"], dtype=np.float64)
    video_pos = np.asarray(video_camera["pos_w"], dtype=np.float64)
    video_quat_ros = np.asarray(video_camera["quat_w_ros"], dtype=np.float64)
    lookat_quat_ros = lookat_camera_quat_wxyz(video_pos, banana_center)
    env_cfg = parse_env_cfg(
        "ATEC-TaskE-Piper",
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    virtual_sensor_names = {
        "virtual_video_clone": add_camera_cfg(
            env_cfg,
            "debug_virtual_video_ros_clone",
            video_pos,
            video_quat_ros,
        ),
        "virtual_video_lookat": add_camera_cfg(
            env_cfg,
            "debug_virtual_video_lookat_banana",
            video_pos,
            lookat_quat_ros,
        ),
    }
    env = gym.make("ATEC-TaskE-Piper", cfg=env_cfg)
    obs, _ = env.reset()
    apply_deterministic_object_poses(env, object_poses)
    obs = env.unwrapped.observation_manager.compute()
    return env, obs, virtual_sensor_names, object_poses


def apply_deterministic_object_poses(env, object_poses: dict[str, dict]) -> None:
    scene = env.unwrapped.scene
    for pose in object_poses.values():
        object_key = pose["object_key"]
        obj = scene.rigid_objects[object_key]
        state = obj.data.default_root_state[0:1].clone()
        state[0, 0:3] = torch.tensor(pose["center_w"], dtype=state.dtype, device=state.device)
        state[0, 3:7] = torch.tensor(pose["quat_wxyz"], dtype=state.dtype, device=state.device)
        state[0, 7:] = 0.0
        obj.write_root_state_to_sim(state)
    scene.write_data_to_sim()
    env.unwrapped.sim.forward()


def run_sam3(image_path: Path, out_dir: Path, label: str, view_label: str) -> tuple[Path, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / f"{label}_mask.png"
    detection_path = out_dir / f"{label}_detections.json"
    command = [
        "conda",
        "run",
        "-n",
        args_cli.sam3_env,
        "python",
        "scripts/sam3_single_image_mask.py",
        "--image",
        str(image_path),
        "--prompt",
        "banana",
        "--label",
        label,
        "--view-label",
        view_label,
        "--output",
        str(out_dir),
        "--device",
        args_cli.sam3_device,
    ]
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    (out_dir / "sam3.log").write_text(proc.stdout or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"SAM3 failed for {label}: {out_dir / 'sam3.log'}")
    return mask_path, load_json(detection_path)


def save_capture(out: Path, env, obs, virtual_sensor_names: dict[str, str]) -> dict:
    video_rgb = to_rgb_array(obs["image"]["video_rgb"])
    video_depth = to_depth_array(obs["image"]["video_depth"])
    Image.fromarray(video_rgb, mode="RGB").save(out / "actual_video_rgb.png")
    np.save(out / "actual_video_depth.npy", video_depth)
    write_json(out / "actual_video_camera.json", camera_metadata(env, "video_cam"))

    captures = {
        "actual_video": {
            "rgb": out / "actual_video_rgb.png",
            "depth": out / "actual_video_depth.npy",
            "camera": out / "actual_video_camera.json",
        }
    }
    for capture_name, sensor_name in virtual_sensor_names.items():
        virtual_sensor = env.unwrapped.scene.sensors[sensor_name]
        virtual_rgb = camera_rgb(virtual_sensor)
        virtual_depth = camera_depth(virtual_sensor)
        Image.fromarray(virtual_rgb, mode="RGB").save(out / f"{capture_name}_rgb.png")
        np.save(out / f"{capture_name}_depth.npy", virtual_depth)
        write_json(out / f"{capture_name}_camera.json", camera_metadata(env, sensor_name))
        captures[capture_name] = {
            "rgb": out / f"{capture_name}_rgb.png",
            "depth": out / f"{capture_name}_depth.npy",
            "camera": out / f"{capture_name}_camera.json",
        }
    return captures


def point_cloud_in_primary(
    rgb: Path,
    depth: Path,
    mask: Path,
    camera_json: Path,
    primary_camera: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    camera_payload = load_json(camera_json)
    camera = camera_payload.get("camera", camera_payload)
    _, _, target, target_colors, stats = view_points(
        rgb,
        depth,
        mask,
        camera,
        args_cli.max_depth,
        1,
    )
    if camera_json.name == "actual_ee_camera.json":
        return target, target_colors, stats
    points_primary = transform_points_from_world(transform_points_to_world(target, camera), primary_camera).astype(np.float32)
    return points_primary, target_colors, stats


def nn_stats(primary: np.ndarray, extra: np.ndarray) -> dict:
    try:
        from scipy.spatial import cKDTree

        dists, _ = cKDTree(primary.astype(np.float64)).query(extra.astype(np.float64), k=1, workers=-1)
    except Exception:
        diffs = extra[:, None, :].astype(np.float64) - primary[None, :, :].astype(np.float64)
        dists = np.sqrt(np.sum(diffs * diffs, axis=2)).min(axis=1)
    return {
        "median_m": float(np.median(dists)),
        "p80_m": float(np.quantile(dists, 0.8)),
        "p95_m": float(np.quantile(dists, 0.95)),
        "frac_lt_01m": float(np.mean(dists < 0.01)),
        "frac_lt_02m": float(np.mean(dists < 0.02)),
        "frac_lt_03m": float(np.mean(dists < 0.03)),
    }


def cloud_stats(points: np.ndarray, source_stats: dict) -> dict:
    if len(points) == 0:
        return {"point_count": 0, "source_stats": source_stats}
    q = np.quantile(points, [0.01, 0.5, 0.99], axis=0)
    return {
        "point_count": int(len(points)),
        "bbox_xyxy": source_stats.get("target_bbox_xyxy"),
        "primary_camera_quantile_01_50_99": q.astype(float).tolist(),
        "primary_camera_q99_minus_q01": (q[2] - q[0]).astype(float).tolist(),
        "primary_camera_minmax": (points.max(axis=0) - points.min(axis=0)).astype(float).tolist(),
    }


def save_colored_pair(path: Path, primary: np.ndarray, extra: np.ndarray, extra_color: tuple[float, float, float]) -> None:
    points = np.concatenate([primary, extra], axis=0).astype(np.float32)
    colors = np.concatenate(
        [
            np.tile(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), (len(primary), 1)),
            np.tile(np.asarray(extra_color, dtype=np.float32), (len(extra), 1)),
        ],
        axis=0,
    )
    write_ply(path, points, colors)


def main() -> None:
    source_dir = args_cli.source.expanduser().resolve()
    out = output_dir()

    env, obs, virtual_sensor_names, object_poses = create_env(source_dir)
    for _ in range(max(1, int(args_cli.debug_steps))):
        zero_action = torch.zeros_like(env.unwrapped.scene.articulations["robot"].data.default_joint_pos)
        obs, _, terminated, truncated, _ = env.step(zero_action)
        if terminated.any() or truncated.any():
            break
    captures = save_capture(out, env, obs, virtual_sensor_names)
    env.close()

    actual_mask, actual_detection = run_sam3(
        captures["actual_video"]["rgb"],
        out / "masks" / "actual_video",
        "banana_actual_video",
        "actual video_cam in new same-seed env",
    )
    masks = {"actual_video": actual_mask}
    detections = {"actual_video": actual_detection}
    for capture_name, capture in captures.items():
        if capture_name == "actual_video":
            continue
        mask, detection = run_sam3(
            capture["rgb"],
            out / "masks" / capture_name,
            f"banana_{capture_name}",
            capture_name.replace("_", " "),
        )
        masks[capture_name] = mask
        detections[capture_name] = detection

    primary_camera_payload = load_json(source_dir / "banana" / "actual_ee_camera.json")
    primary_camera = primary_camera_payload.get("camera", primary_camera_payload)
    primary_rgb = source_dir / "banana" / "actual_ee_camera_rgb.png"
    primary_depth = source_dir / "banana" / "actual_ee_camera_depth.npy"
    primary_mask = (
        source_dir
        / "grasp_fusion_debug_all_filtered"
        / "banana"
        / "masks"
        / "ee_banana"
        / "banana_ee_banana_mask.png"
    )
    primary_points, primary_colors, primary_source_stats = point_cloud_in_primary(
        primary_rgb,
        primary_depth,
        primary_mask,
        source_dir / "banana" / "actual_ee_camera.json",
        primary_camera,
    )
    actual_points, _, actual_source_stats = point_cloud_in_primary(
        captures["actual_video"]["rgb"],
        captures["actual_video"]["depth"],
        actual_mask,
        captures["actual_video"]["camera"],
        primary_camera,
    )
    extra_clouds = {}
    for capture_name, capture in captures.items():
        points, _, source_stats = point_cloud_in_primary(
            capture["rgb"],
            capture["depth"],
            masks[capture_name],
            capture["camera"],
            primary_camera,
        )
        extra_clouds[capture_name] = {"points": points, "source_stats": source_stats}

    write_ply(out / "primary_ee_banana_cloud.ply", primary_points, primary_colors)
    colors_by_capture = {
        "actual_video": (1.0, 0.0, 1.0),
        "virtual_video_clone": (0.0, 1.0, 1.0),
        "virtual_video_lookat": (0.0, 1.0, 0.0),
    }
    ply_paths = {"primary": str(out / "primary_ee_banana_cloud.ply")}
    all_points = [primary_points]
    all_colors = [np.tile(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), (len(primary_points), 1))]
    for capture_name, payload in extra_clouds.items():
        path = out / f"banana_primary_plus_{capture_name}.ply"
        color = colors_by_capture.get(capture_name, (1.0, 1.0, 1.0))
        save_colored_pair(path, primary_points, payload["points"], color)
        ply_paths[f"primary_plus_{capture_name}"] = str(path)
        all_points.append(payload["points"])
        all_colors.append(np.tile(np.asarray(color, dtype=np.float32), (len(payload["points"]), 1)))
    write_ply(
        out / "banana_primary_actual_video_virtual_video.ply",
        np.concatenate(all_points, axis=0).astype(np.float32),
        np.concatenate(all_colors, axis=0).astype(np.float32),
    )
    ply_paths["primary_actual_video_virtual_video"] = str(out / "banana_primary_actual_video_virtual_video.ply")

    actual_video_camera = load_json(captures["actual_video"]["camera"])
    virtual_cameras = {
        name: load_json(capture["camera"])
        for name, capture in captures.items()
        if name != "actual_video"
    }
    clouds = {"primary_ee": cloud_stats(primary_points, primary_source_stats)}
    for capture_name, payload in extra_clouds.items():
        clouds[capture_name] = {
            **cloud_stats(payload["points"], payload["source_stats"]),
            "nearest_to_primary": nn_stats(primary_points, payload["points"]),
        }
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(source_dir),
        "output_dir": str(out),
        "deterministic_object_poses_applied": object_poses,
        "color_legend": {
            "primary_ee": "red",
            "actual_video": "magenta",
            "virtual_video_clone": "cyan",
            "virtual_video_lookat": "green",
        },
        "virtual_camera_source": {
            "virtual_video_clone": "spawned at source video_camera.json pos_w + quat_w_ros with CameraCfg convention='ros'",
            "virtual_video_lookat": "spawned at source video_camera.json pos_w and look_at deterministic banana center",
        },
        "camera_pose_delta": {
            name: {
                "actual_vs_virtual_pos_l2_m": float(
                    np.linalg.norm(
                        np.asarray(actual_video_camera["pos_w"], dtype=np.float64)
                        - np.asarray(camera["pos_w"], dtype=np.float64)
                    )
                )
            }
            for name, camera in virtual_cameras.items()
        }
        | {
            "actual_video_camera": actual_video_camera,
            "virtual_cameras": virtual_cameras,
        },
        "sam3": detections,
        "clouds": clouds,
        "ply": ply_paths,
    }
    write_json(out / "summary.json", summary)
    print(f"[INFO] Saved {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
