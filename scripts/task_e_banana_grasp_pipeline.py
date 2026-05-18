#!/usr/bin/env python3
"""Task E banana localization, EE camera capture, RGB-D cloud, and AnyGrasp demo."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher


def prepare_output_dir(path: Path | None) -> Path:
    if path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPO_ROOT / "outputs/task_e_banana_pipeline" / timestamp
    path.mkdir(parents=True, exist_ok=True)
    latest_txt = path.parent / "latest.txt"
    latest_txt.write_text(str(path.resolve()), encoding="utf-8")
    latest_link = path.parent / "latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(path.resolve(), target_is_directory=True)
    except OSError:
        pass
    return path.resolve()


def write_failure(path: Path, name: str, exc: BaseException) -> None:
    payload = {
        "error_type": type(exc).__name__,
        "error": str(exc),
        "repr": repr(exc),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Output directory. Defaults to outputs/task_e_banana_pipeline/<timestamp>.",
)
parser.add_argument("--prompt", default="banana", help="SAM3 prompt for the target object.")
parser.add_argument("--sam3-env", default="sam3_full", help="Conda env used for SAM3.")
parser.add_argument("--anygrasp-env", default="anygrasp", help="Conda env used for AnyGrasp.")
parser.add_argument(
    "--anygrasp-checkpoint-path",
    type=Path,
    default=Path(os.environ["ANYGRASP_CHECKPOINT"]) if os.environ.get("ANYGRASP_CHECKPOINT") else None,
    help="Path to AnyGrasp checkpoint_detection.tar. Defaults to the SDK log path.",
)
parser.add_argument(
    "--anygrasp-license-dir",
    type=Path,
    default=Path(os.environ["ANYGRASP_LICENSE_DIR"]) if os.environ.get("ANYGRASP_LICENSE_DIR") else None,
    help="Path to the machine-bound AnyGrasp license directory.",
)
parser.add_argument(
    "--require-anygrasp",
    action="store_true",
    help="Exit nonzero unless AnyGrasp returns a final grasp pose.",
)
parser.add_argument(
    "--hover-height",
    type=float,
    default=0.25,
    help="World Z offset above table top for EE hover capture.",
)
parser.add_argument(
    "--hover-mode",
    choices=("gripper", "camera"),
    default="gripper",
    help="Use gripper-base top-down hover or calibrated camera-center hover.",
)
parser.add_argument(
    "--gripper-hover-offset",
    type=parse_float_list,
    default=[0.12, 0.04, 0.0],
    help="XYZ world offset added to the SAM3 target for gripper-mode EE camera capture.",
)
parser.add_argument("--settle-steps", type=int, default=180, help="IK settle steps for hover.")
parser.add_argument("--max-depth", type=float, default=2.0, help="Max depth used for masked point clouds.")
parser.add_argument("--no-anygrasp", action="store_true", help="Skip AnyGrasp helper.")
parser.add_argument(
    "--allow-gt-fallback",
    action="store_true",
    help="Use simulator banana pose if SAM3/depth localization fails.",
)
parser.add_argument(
    "--hover-action",
    type=parse_float_list,
    default=None,
    help="Optional raw env action to use instead of IK hover.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
OUTPUT_DIR = prepare_output_dir(args_cli.output)

try:
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
except BaseException as exc:
    write_failure(OUTPUT_DIR, "launch_error.json", exc)
    raise

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401,E402
from atec_rl_lab.tasks.task_e.env_cfg import TABLE_TOP_Z  # noqa: E402
from atec_rl_lab.utils import CartesianController  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINT_NAMES = ["joint7", "joint8"]
GRIPPER_OPEN_POS = [0.035, -0.035]
ACTION_SCALE = 0.5
TOP_DOWN_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)


def make_output_dir(path: Path | None) -> Path:
    return OUTPUT_DIR if path is None else prepare_output_dir(path)


def to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def to_rgb_array(value) -> np.ndarray:
    array = to_numpy(value)
    if array.ndim == 4:
        array = array[0]
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
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


def save_depth_preview(path: Path, depth: np.ndarray, max_depth: float = 2.0) -> None:
    valid = np.isfinite(depth) & (depth > 0.0)
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        clipped = np.clip(depth, 0.0, max_depth)
        scaled = (255.0 * (1.0 - clipped / max_depth)).astype(np.uint8)
        scaled[~valid] = 0
    Image.fromarray(scaled, mode="L").save(path)


def save_camera_frame(output_dir: Path, prefix: str, obs_image: dict) -> dict:
    rgb = to_rgb_array(obs_image[f"{prefix}_rgb"])
    depth = to_depth_array(obs_image[f"{prefix}_depth"])
    Image.fromarray(rgb, mode="RGB").save(output_dir / f"{prefix}_rgb.png")
    np.save(output_dir / f"{prefix}_depth.npy", depth)
    save_depth_preview(output_dir / f"{prefix}_depth_preview.png", depth)
    return {
        "rgb": f"{prefix}_rgb.png",
        "depth_npy": f"{prefix}_depth.npy",
        "depth_preview": f"{prefix}_depth_preview.png",
        "rgb_shape": list(rgb.shape),
        "depth_shape": list(depth.shape),
        "depth_minmax": [
            float(np.nanmin(depth[np.isfinite(depth)])) if np.isfinite(depth).any() else None,
            float(np.nanmax(depth[np.isfinite(depth)])) if np.isfinite(depth).any() else None,
        ],
    }


def tensor_to_list(value) -> list[float]:
    return value.detach().cpu().numpy().astype(float).tolist()


def camera_metadata(env, sensor_name: str) -> dict:
    camera = env.unwrapped.scene.sensors[sensor_name]
    data = camera.data
    payload = {
        "sensor": sensor_name,
        "image_shape": list(data.image_shape),
        "intrinsic_matrix": tensor_to_list(data.intrinsic_matrices[0]),
        "pos_w": tensor_to_list(data.pos_w[0]),
        "quat_w_world": tensor_to_list(data.quat_w_world[0]),
        "quat_w_ros": tensor_to_list(data.quat_w_ros[0]),
        "quat_w_opengl": tensor_to_list(data.quat_w_opengl[0]),
    }
    return payload


def write_camera_json(path: Path, camera: dict) -> None:
    path.write_text(json.dumps({"camera": camera}, indent=2), encoding="utf-8")


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L")
    width, height = shape[1], shape[0]
    if mask.size != (width, height):
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(mask) > 127


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


def quat_normalize(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def quat_inverse(quat: np.ndarray) -> np.ndarray:
    quat = quat_normalize(quat)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = quat_normalize(left)
    w2, x2, y2, z2 = quat_normalize(right)
    return quat_normalize(
        np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )
    )


def backproject(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    z = depth.astype(np.float32)
    x = (xx - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (yy - intrinsic[1, 2]) * z / intrinsic[1, 1]
    return np.stack([x, y, z], axis=-1)


def transform_points_to_world(points_cam: np.ndarray, camera: dict) -> np.ndarray:
    rot = quat_wxyz_to_matrix(np.asarray(camera["quat_w_ros"], dtype=np.float64))
    pos = np.asarray(camera["pos_w"], dtype=np.float64)
    return points_cam @ rot.T + pos


def estimate_pose_from_mask(
    depth: np.ndarray,
    mask: np.ndarray,
    camera: dict,
    max_depth: float,
) -> dict:
    intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float32)
    points_cam_all = backproject(depth, intrinsic)
    valid = mask & np.isfinite(depth) & (depth > 0.0) & (depth < max_depth)
    if not valid.any():
        raise ValueError("SAM3 mask has no valid depth pixels.")

    points_cam = points_cam_all[valid]
    points_world = transform_points_to_world(points_cam, camera)
    center = np.median(points_world, axis=0)
    xy = points_world[:, :2]
    if len(xy) >= 3:
        cov = np.cov((xy - xy.mean(axis=0)).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]
    else:
        axis = np.array([1.0, 0.0])
    yaw = float(np.arctan2(axis[1], axis[0]))
    ys, xs = np.where(valid)
    pixel_center = [float(np.median(xs)), float(np.median(ys))]
    return {
        "center_world": center.astype(float).tolist(),
        "principal_axis_xy": axis.astype(float).tolist(),
        "yaw_rad": yaw,
        "valid_depth_points": int(valid.sum()),
        "pixel_center": pixel_center,
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
    }


def get_ground_truth_banana_pose(env) -> dict | None:
    try:
        banana = env.unwrapped.scene["object_3"]
        pose = banana.data.root_pose_w[0]
        return {"center_world": tensor_to_list(pose[:3]), "quat_wxyz": tensor_to_list(pose[3:])}
    except Exception:
        return None


def run_subprocess(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    print(f"[INFO] Running: {' '.join(command)}", flush=True)
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit {proc.returncode}. See {log_path}")


def run_sam3(image_path: Path, output_dir: Path, label: str, view_label: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
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
        args_cli.prompt,
        "--label",
        label,
        "--view-label",
        view_label,
        "--output",
        str(output_dir),
        "--device",
        "cuda",
    ]
    run_subprocess(command, output_dir / f"{label}_sam3.log")
    detection_path = output_dir / f"{label}_detections.json"
    return json.loads(detection_path.read_text(encoding="utf-8"))


def draw_target_overlay(
    image_path: Path,
    output_path: Path,
    pose: dict,
    title: str,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    bbox = pose.get("bbox_xyxy")
    if bbox is not None:
        draw.rectangle(tuple(bbox), outline=(255, 60, 60), width=3)
    center = pose.get("pixel_center")
    if center is not None:
        x, y = center
        draw.line((x - 14, y, x + 14, y), fill=(255, 255, 0), width=3)
        draw.line((x, y - 14, x, y + 14), fill=(255, 255, 0), width=3)
    center_world = pose.get("center_world")
    text = title
    if center_world is not None:
        text += f" | world {np.round(center_world, 3).tolist()}"
    draw.rectangle((0, 0, image.width, 28), fill=(0, 0, 0))
    draw.text((8, 7), text, fill=(255, 255, 255))
    image.save(output_path)


def action_to_tensor(action: list[float], action_dim: int, device: str) -> torch.Tensor:
    values = list(action[:action_dim])
    if len(values) < action_dim:
        values.extend([0.0] * (action_dim - len(values)))
    return torch.tensor(values, dtype=torch.float32, device=device).view(1, -1)


def desired_gripper_pose_for_camera(
    env,
    robot,
    ee_idx: int,
    desired_camera_pos_w: list[float],
    desired_camera_quat_ros: list[float],
) -> tuple[np.ndarray, np.ndarray, dict]:
    camera = env.unwrapped.scene.sensors["ee_camera"]
    cam_pos_w = np.asarray(tensor_to_list(camera.data.pos_w[0]), dtype=np.float64)
    cam_quat_w_ros = np.asarray(tensor_to_list(camera.data.quat_w_ros[0]), dtype=np.float64)
    grip_pose_w = robot.data.body_pose_w[0, ee_idx]
    grip_pos_w = np.asarray(tensor_to_list(grip_pose_w[:3]), dtype=np.float64)
    grip_quat_w = np.asarray(tensor_to_list(grip_pose_w[3:]), dtype=np.float64)

    grip_to_cam_quat = quat_multiply(quat_inverse(grip_quat_w), cam_quat_w_ros)
    grip_to_cam_pos = quat_wxyz_to_matrix(grip_quat_w).T @ (cam_pos_w - grip_pos_w)

    desired_cam_quat = quat_normalize(np.asarray(desired_camera_quat_ros, dtype=np.float64))
    desired_grip_quat = quat_multiply(desired_cam_quat, quat_inverse(grip_to_cam_quat))
    desired_grip_pos = (
        np.asarray(desired_camera_pos_w, dtype=np.float64)
        - quat_wxyz_to_matrix(desired_grip_quat) @ grip_to_cam_pos
    )

    calibration = {
        "current_camera_pos_w": cam_pos_w.tolist(),
        "current_camera_quat_w_ros": cam_quat_w_ros.tolist(),
        "current_gripper_pos_w": grip_pos_w.tolist(),
        "current_gripper_quat_wxyz": grip_quat_w.tolist(),
        "gripper_to_camera_pos": grip_to_cam_pos.tolist(),
        "gripper_to_camera_quat_ros": grip_to_cam_quat.tolist(),
        "desired_camera_pos_w": list(desired_camera_pos_w),
        "desired_camera_quat_w_ros": list(desired_camera_quat_ros),
        "desired_gripper_pos_w": desired_grip_pos.tolist(),
        "desired_gripper_quat_wxyz": desired_grip_quat.tolist(),
    }
    return desired_grip_pos, desired_grip_quat, calibration


def move_to_hover(env, obs, robot, hover_pos_w: list[float]) -> tuple[dict, dict]:
    if args_cli.hover_action is not None:
        action = action_to_tensor(
            args_cli.hover_action,
            int(robot.data.default_joint_pos.shape[-1]),
            env.unwrapped.device,
        )
        for _ in range(args_cli.settle_steps):
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated.any() or truncated.any():
                break
        return obs, {"mode": "raw_action", "hover_pos_w": hover_pos_w}

    arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
    gripper_ids, _ = robot.find_joints(GRIPPER_JOINT_NAMES)
    controller = CartesianController(
        robot=robot,
        ee_body_name="gripper_base",
        arm_joint_names=ARM_JOINT_NAMES,
        num_envs=1,
        device=env.unwrapped.device,
        command_type="pose",
        lambda_val=0.05,
        max_joint_delta=0.18,
    )
    controller.reset()
    camera_calibration = None
    if args_cli.hover_mode == "camera":
        desired_camera_quat_ros = [0.0, 1.0, 0.0, 0.0]
        grip_pos_des, grip_quat_des, camera_calibration = desired_gripper_pose_for_camera(
            env,
            robot,
            controller.ee_idx,
            hover_pos_w,
            desired_camera_quat_ros,
        )
        mode = "cartesian_ik_camera_center"
    else:
        grip_pos_des = np.asarray(hover_pos_w, dtype=np.float64)
        grip_quat_des = np.asarray(TOP_DOWN_QUAT_WXYZ, dtype=np.float64)
        mode = "cartesian_ik_gripper_base"
    hover = torch.tensor([grip_pos_des], dtype=torch.float32, device=env.unwrapped.device)
    quat = torch.tensor([grip_quat_des], dtype=torch.float32, device=env.unwrapped.device)
    gripper_target = torch.tensor([GRIPPER_OPEN_POS], dtype=torch.float32, device=env.unwrapped.device)
    default_jpos = robot.data.default_joint_pos.clone()
    last_action = None
    for _ in range(args_cli.settle_steps):
        arm_des = controller.compute(hover, quat)
        target = robot.data.joint_pos.clone()
        target[:, arm_ids] = arm_des
        target[:, gripper_ids] = gripper_target
        action = (target - default_jpos) / ACTION_SCALE
        last_action = action
        obs, _, terminated, truncated, _ = env.step(action)
        robot.update(dt=env.unwrapped.physics_dt)
        if terminated.any() or truncated.any():
            break

    ee_pos = robot.data.body_pose_w[0, controller.ee_idx, :3]
    ee_quat = robot.data.body_pose_w[0, controller.ee_idx, 3:]
    record = {
        "mode": mode,
        "hover_pos_w": hover_pos_w,
        "desired_gripper_pos_w": grip_pos_des.astype(float).tolist(),
        "desired_gripper_quat_wxyz": grip_quat_des.astype(float).tolist(),
        "settle_steps": args_cli.settle_steps,
        "final_ee_pos_w": tensor_to_list(ee_pos),
        "final_ee_quat_wxyz": tensor_to_list(ee_quat),
        "last_action": tensor_to_list(last_action[0]) if last_action is not None else None,
    }
    if camera_calibration is not None:
        record["camera_to_gripper_calibration"] = camera_calibration
    return obs, record


def run_anygrasp(output_dir: Path, ee_camera_json: Path) -> dict | None:
    if args_cli.no_anygrasp:
        return None
    anygrasp_dir = output_dir / "anygrasp"
    anygrasp_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    openssl_lib = REPO_ROOT / "third_party/openssl11/lib"
    current_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{openssl_lib}:{current_ld}" if current_ld else str(openssl_lib)
    command = [
        "conda",
        "run",
        "-n",
        args_cli.anygrasp_env,
        "python",
        "scripts/anygrasp_from_rgbd_mask.py",
        "--rgb",
        str(output_dir / "ee_rgb.png"),
        "--depth-npy",
        str(output_dir / "ee_depth.npy"),
        "--mask",
        str(output_dir / "sam3_ee/ee_banana_mask.png"),
        "--camera-json",
        str(ee_camera_json),
        "--output",
        str(anygrasp_dir),
        "--max-depth",
        str(args_cli.max_depth),
    ]
    if args_cli.anygrasp_checkpoint_path is not None:
        command.extend(["--checkpoint-path", str(args_cli.anygrasp_checkpoint_path)])
    if args_cli.anygrasp_license_dir is not None:
        command.extend(["--license-dir", str(args_cli.anygrasp_license_dir)])
    run_subprocess(command, anygrasp_dir / "anygrasp.log", env=env)
    result_path = anygrasp_dir / "anygrasp_result.json"
    return json.loads(result_path.read_text(encoding="utf-8"))


def persist_final_grasp_pose(output_dir: Path, anygrasp_result: dict | None) -> str | None:
    if not isinstance(anygrasp_result, dict):
        return None
    anygrasp_payload = anygrasp_result.get("anygrasp")
    if not isinstance(anygrasp_payload, dict):
        return None
    final_pose = anygrasp_payload.get("final_grasp_pose")
    if isinstance(final_pose, dict) and anygrasp_payload.get("status") == "ok":
        path = output_dir / "final_grasp_pose.json"
        path.write_text(json.dumps(final_pose, indent=2), encoding="utf-8")
        return path.name
    path = output_dir / "final_grasp_pose_error.json"
    path.write_text(json.dumps(anygrasp_payload, indent=2), encoding="utf-8")
    return path.name


def anygrasp_status(anygrasp_result: dict | None) -> str | None:
    if not isinstance(anygrasp_result, dict):
        return None
    payload = anygrasp_result.get("anygrasp")
    if isinstance(payload, dict):
        return payload.get("status")
    return None


def make_summary_image(output_dir: Path, anygrasp_result: dict | None) -> None:
    tiles = [
        ("1 video rgb", output_dir / "video_rgb.png"),
        ("2 video SAM3", output_dir / "sam3_video/video_banana_overlay.png"),
        ("3 video target", output_dir / "video_target_overlay.png"),
        ("4 ee rgb", output_dir / "ee_rgb.png"),
        ("5 ee SAM3", output_dir / "sam3_ee/ee_banana_overlay.png"),
    ]
    if anygrasp_result is not None:
        tiles.append(("6 cloud / AnyGrasp", output_dir / "anygrasp/anygrasp_result.png"))
    else:
        tiles.append(("6 depth", output_dir / "ee_depth_preview.png"))

    tile_w, tile_h = 420, 315
    canvas = Image.new("RGB", (tile_w * 3, tile_h * 2), (18, 22, 30))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, path) in enumerate(tiles):
        row, col = divmod(idx, 3)
        x, y = col * tile_w, row * tile_h
        if path.exists():
            image = Image.open(path).convert("RGB")
            image.thumbnail((tile_w, tile_h - 26), Image.Resampling.LANCZOS)
            canvas.paste(image, (x + (tile_w - image.width) // 2, y + 26))
        else:
            draw.text((x + 12, y + 52), f"missing: {path.name}", fill=(255, 160, 160))
        draw.rectangle((x, y, x + tile_w, y + 24), fill=(0, 0, 0))
        draw.text((x + 8, y + 6), label, fill=(255, 255, 255))
    canvas.save(output_dir / "summary.png")


def main() -> None:
    output_dir = OUTPUT_DIR
    env_cfg = parse_env_cfg(
        "ATEC-TaskE-Piper",
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make("ATEC-TaskE-Piper", cfg=env_cfg)
    obs, _ = env.reset()
    robot = env.unwrapped.scene.articulations["robot"]

    records: dict = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": "ATEC-TaskE-Piper",
        "prompt": args_cli.prompt,
        "output_dir": str(output_dir),
        "max_depth": args_cli.max_depth,
        "ground_truth_banana_pose": get_ground_truth_banana_pose(env),
    }

    records["initial_frames"] = {
        "video": save_camera_frame(output_dir, "video", obs["image"]),
        "ee": save_camera_frame(output_dir, "ee_reset", {"ee_reset_rgb": obs["image"]["ee_rgb"], "ee_reset_depth": obs["image"]["ee_depth"]}),
    }
    video_camera = camera_metadata(env, "video_cam")
    write_camera_json(output_dir / "video_camera.json", video_camera)

    video_detection = run_sam3(
        output_dir / "video_rgb.png",
        output_dir / "sam3_video",
        "video_banana",
        "eye-to-hand / video_cam",
    )
    records["video_sam3"] = video_detection

    video_depth = np.load(output_dir / "video_depth.npy")
    video_mask = load_mask(output_dir / "sam3_video/video_banana_mask.png", video_depth.shape)
    try:
        video_pose = estimate_pose_from_mask(video_depth, video_mask, video_camera, args_cli.max_depth)
        video_pose["source"] = "sam3_video_mask_depth"
    except Exception as exc:
        gt = records.get("ground_truth_banana_pose")
        if not args_cli.allow_gt_fallback or gt is None:
            raise
        video_pose = {
            "source": "ground_truth_fallback",
            "error": repr(exc),
            "center_world": gt["center_world"],
        }
    records["banana_pose_from_video"] = video_pose
    draw_target_overlay(
        output_dir / "video_rgb.png",
        output_dir / "video_target_overlay.png",
        video_pose,
        "banana target from video mask",
    )

    center = np.asarray(video_pose["center_world"], dtype=np.float32)
    hover_target = [
        float(center[0]),
        float(center[1]),
        float(TABLE_TOP_Z + args_cli.hover_height),
    ]
    if args_cli.hover_mode == "gripper":
        offset = list(args_cli.gripper_hover_offset[:3])
        if len(offset) < 3:
            offset.extend([0.0] * (3 - len(offset)))
        hover_target = [float(value + delta) for value, delta in zip(hover_target, offset, strict=True)]
    obs, hover_record = move_to_hover(env, obs, robot, hover_target)
    records["hover"] = hover_record

    records["hover_frames"] = {
        "ee": save_camera_frame(output_dir, "ee", obs["image"]),
        "video": save_camera_frame(output_dir, "video_hover", {"video_hover_rgb": obs["image"]["video_rgb"], "video_hover_depth": obs["image"]["video_depth"]}),
    }
    ee_camera = camera_metadata(env, "ee_camera")
    write_camera_json(output_dir / "ee_camera.json", ee_camera)

    ee_detection = run_sam3(
        output_dir / "ee_rgb.png",
        output_dir / "sam3_ee",
        "ee_banana",
        "eye-in-hand / ee_camera",
    )
    records["ee_sam3"] = ee_detection

    ee_depth = np.load(output_dir / "ee_depth.npy")
    ee_mask = load_mask(output_dir / "sam3_ee/ee_banana_mask.png", ee_depth.shape)
    try:
        records["banana_pose_from_ee"] = estimate_pose_from_mask(
            ee_depth,
            ee_mask,
            ee_camera,
            args_cli.max_depth,
        )
        records["banana_pose_from_ee"]["source"] = "sam3_ee_mask_depth"
    except Exception as exc:
        records["banana_pose_from_ee"] = {"source": "failed", "error": repr(exc)}

    anygrasp_result = run_anygrasp(output_dir, output_dir / "ee_camera.json")
    records["anygrasp"] = anygrasp_result
    records["final_grasp_pose_file"] = persist_final_grasp_pose(output_dir, anygrasp_result)
    make_summary_image(output_dir, anygrasp_result)

    (output_dir / "pipeline.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[INFO] Saved pipeline outputs to {output_dir}")
    print(f"[INFO] Summary: {output_dir / 'summary.png'}")
    print(f"[INFO] Metadata: {output_dir / 'pipeline.json'}")
    if args_cli.require_anygrasp and anygrasp_status(anygrasp_result) != "ok":
        raise RuntimeError(
            "AnyGrasp did not generate a final pose. See "
            f"{output_dir / 'final_grasp_pose_error.json'}"
        )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write_failure(OUTPUT_DIR, "pipeline_error.json", exc)
        try:
            simulation_app.close()
        except Exception:
            pass
        raise
