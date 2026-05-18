#!/usr/bin/env python3
"""Execute a Task E motion request and write a unified motion result.

The request/result schema is MoveIt-ready. Until a built MoveIt2 workspace and
Piper MoveIt config are available, this runner executes through the same
IsaacLab Cartesian controller used for simulator validation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher


REQUEST_SCHEMA = "atec.task_e.motion_request.v1"
RESULT_SCHEMA = "atec.task_e.motion_result.v1"
DEFAULT_REQUEST = REPO_ROOT / "outputs/task_e_banana_pipeline/latest/pseudo_grasp/motion_request.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, default=DEFAULT_REQUEST)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/task_e_moveit/<timestamp>.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "moveit_py", "isaaclab_cartesian_controller"),
        default="auto",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Fail instead of falling back when the requested MoveIt backend is unavailable.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override request.start_state.seed for deterministic Task E reset.",
    )
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--no-save-frames", action="store_true")
    parser.add_argument(
        "--skip-object-summary",
        action="store_true",
        help="Skip final object pose query. Useful when only the motion contract is being tested.",
    )
    parser.add_argument(
        "--actuator-mode",
        choices=("request", "default", "task_e_scripted_high_stiffness"),
        default="request",
        help="Use request.controller.actuator_mode unless overridden.",
    )
    parser.add_argument(
        "--object-transport-mode",
        choices=("request", "physics", "kinematic_attach"),
        default="request",
        help="Use request.controller.object_transport_mode unless overridden.",
    )
    parser.add_argument(
        "--record-video-cam",
        action="store_true",
        help="Record the external/video camera for the whole motion as video_cam.mp4.",
    )
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument(
        "--video-every-n-steps",
        type=int,
        default=2,
        help="Write one video frame every N simulator steps.",
    )
    parser.add_argument(
        "--save-pregrasp-viz",
        action="store_true",
        help="Save planned object/grasp/pregrasp overlays under pregrasp_predictions/.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import cv2  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401,E402
from atec_rl_lab.tasks.task_e.env_cfg import (  # noqa: E402
    BASKET_SUCCESS_CENTER,
    BASKET_SUCCESS_HALF_X,
    BASKET_SUCCESS_HALF_Y,
    TABLE_TOP_Z,
)
from atec_rl_lab.utils import CartesianController  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


DEFAULT_ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
DEFAULT_GRIPPER_JOINTS = ["joint7", "joint8"]
TASK_E_OBJECT_LABELS = {
    "object_1": "yellow_and_white_box",
    "object_2": "mustard_bottle",
    "object_3": "banana",
}
TASK_E_SCRIPTED_ACTUATOR = {
    "effort_limit": 100.0,
    "velocity_limit": 100.0,
    "stiffness": 800.0,
    "damping": 80.0,
}


def resolve_request(path: Path) -> Path:
    if path.exists():
        return path.resolve()
    if "latest" in path.parts:
        latest_txt = REPO_ROOT / "outputs/task_e_banana_pipeline/latest.txt"
        if latest_txt.exists():
            latest = Path(latest_txt.read_text(encoding="utf-8").strip())
            candidate = latest / "pseudo_grasp/motion_request.json"
            if candidate.exists():
                return candidate.resolve()
    raise FileNotFoundError(f"Missing motion request JSON: {path}")


def make_output_dir(path: Path | None) -> Path:
    if path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPO_ROOT / "outputs/task_e_moveit" / timestamp
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


def load_request(path: Path) -> dict:
    request = json.loads(path.read_text(encoding="utf-8"))
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError(f"Unsupported request schema: {request.get('schema')!r}")
    if request.get("task") != "ATEC-TaskE-Piper":
        raise ValueError(f"Only ATEC-TaskE-Piper is supported, got {request.get('task')!r}")
    if not request.get("waypoints"):
        raise ValueError("Motion request contains no waypoints.")
    return request


def module_status(name: str) -> str:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return "unavailable"
    return f"available:{spec.origin}"


def backend_status(request: dict, backend_arg: str, no_fallback: bool) -> dict:
    modules = {
        "moveit": module_status("moveit"),
        "moveit_py": module_status("moveit_py"),
    }
    preferred = request.get("backend", {}).get("preferred", "moveit_py")
    requested = preferred if backend_arg == "auto" else backend_arg
    moveit_ready = modules["moveit_py"].startswith("available")

    if requested == "isaaclab_cartesian_controller":
        used = "isaaclab_cartesian_controller"
        reason = "requested explicitly"
    elif requested == "moveit_py" and moveit_ready:
        used = "moveit_py"
        reason = "moveit_py import is available"
    elif no_fallback:
        used = None
        reason = "moveit_py is unavailable and fallback is disabled"
    else:
        used = "isaaclab_cartesian_controller"
        reason = "moveit_py unavailable; using simulator IK fallback"

    return {
        "requested": requested,
        "used": used,
        "reason": reason,
        "modules": modules,
        "moveit2_source": str((REPO_ROOT / "third_party/moveit2").resolve()),
        "moveit2_workspace": str((REPO_ROOT / "third_party/moveit2_ws/install").resolve()),
    }


def resolve_actuator_mode(request: dict) -> str:
    if args_cli.actuator_mode != "request":
        return args_cli.actuator_mode
    return request.get("controller", {}).get("actuator_mode", "default")


def resolve_object_transport_mode(request: dict) -> str:
    if args_cli.object_transport_mode != "request":
        return args_cli.object_transport_mode
    return request.get("controller", {}).get("object_transport_mode", "physics")


def apply_actuator_mode(env_cfg, mode: str) -> None:
    if mode == "default":
        return
    if mode != "task_e_scripted_high_stiffness":
        raise ValueError(f"Unsupported actuator mode: {mode!r}")
    env_cfg.scene.robot.actuators["default"] = ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        effort_limit=TASK_E_SCRIPTED_ACTUATOR["effort_limit"],
        velocity_limit=TASK_E_SCRIPTED_ACTUATOR["velocity_limit"],
        stiffness=TASK_E_SCRIPTED_ACTUATOR["stiffness"],
        damping=TASK_E_SCRIPTED_ACTUATOR["damping"],
    )


def refresh_observation(env, fallback_obs: dict) -> dict:
    try:
        return env.unwrapped.observation_manager.compute()
    except Exception:
        return fallback_obs


def write_object_pose(env, object_key: str, pos_w: list[float], quat_wxyz: list[float]) -> None:
    scene = env.unwrapped.scene
    obj = scene.rigid_objects[object_key]
    state = obj.data.default_root_state[0:1].clone()
    state[0, 0:3] = torch.tensor(pos_w, dtype=state.dtype, device=state.device)
    state[0, 3:7] = torch.tensor(quat_wxyz, dtype=state.dtype, device=state.device)
    state[0, 7:] = 0.0
    obj.write_root_state_to_sim(state)
    scene.write_data_to_sim()
    env.unwrapped.sim.forward()


def update_attached_object(env, robot, ee_idx: int, attached: dict | None) -> bool:
    if not attached:
        return False
    ee_pos_w = robot.data.body_pose_w[0, ee_idx, :3].detach()
    offset = torch.tensor(
        attached["ee_to_object_pos_w"],
        dtype=ee_pos_w.dtype,
        device=ee_pos_w.device,
    )
    pos_w = (ee_pos_w + offset).detach().cpu().numpy().astype(float).tolist()
    write_object_pose(env, attached["object_key"], pos_w, attached["object_quat_wxyz"])
    return True


def to_rgb(value: Any) -> np.ndarray:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
        if array.size and float(np.nanmax(array)) <= 1.0:
            array *= 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def save_frame(path: Path, obs: dict, label: str) -> None:
    video = Image.fromarray(to_rgb(obs["image"]["video_rgb"]), mode="RGB")
    ee = Image.fromarray(to_rgb(obs["image"]["ee_rgb"]), mode="RGB")
    tile_w = max(video.width, ee.width)
    canvas = Image.new("RGB", (tile_w * 2 + 12, max(video.height, ee.height) + 28), (18, 22, 30))
    canvas.paste(video, (0, 28))
    canvas.paste(ee, (tile_w + 12, 28))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 26), fill=(0, 0, 0))
    draw.text((8, 7), f"{label} | left=video_cam right=ee_camera", fill=(255, 255, 255))
    canvas.save(path)


def camera_metadata(env, sensor_name: str) -> dict:
    camera = env.unwrapped.scene.sensors[sensor_name]
    data = camera.data
    return {
        "sensor": sensor_name,
        "image_shape": list(data.image_shape),
        "intrinsic_matrix": tensor_list(data.intrinsic_matrices[0]),
        "pos_w": tensor_list(data.pos_w[0]),
        "quat_w_ros": tensor_list(data.quat_w_ros[0]),
    }


def quat_wxyz_to_matrix(quat: list[float] | np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def project_world_point(camera: dict, point_w: list[float]) -> tuple[float, float] | None:
    intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float64)
    pos_w = np.asarray(camera["pos_w"], dtype=np.float64)
    rot_wc = quat_wxyz_to_matrix(camera["quat_w_ros"])
    point_c = rot_wc.T @ (np.asarray(point_w, dtype=np.float64) - pos_w)
    if point_c[2] <= 1e-6:
        return None
    u = intrinsic[0, 0] * point_c[0] / point_c[2] + intrinsic[0, 2]
    v = intrinsic[1, 1] * point_c[1] / point_c[2] + intrinsic[1, 2]
    return float(u), float(v)


def draw_marker(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float] | None,
    color: tuple[int, int, int],
    text: str,
) -> None:
    if xy is None:
        return
    x, y = xy
    r = 7
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=3)
    draw.line((x - 12, y, x + 12, y), fill=color, width=2)
    draw.line((x, y - 12, x, y + 12), fill=color, width=2)
    draw.text((x + 10, y - 12), text, fill=color)


def save_pregrasp_prediction_viz(output_dir: Path, obs: dict, request: dict, env) -> dict:
    source = request.get("source", {})
    objects = source.get("objects") if isinstance(source, dict) else None
    if not objects:
        return {}

    prediction_dir = output_dir / "pregrasp_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    camera = camera_metadata(env, "video_cam")
    base = Image.fromarray(to_rgb(obs["image"]["video_rgb"]), mode="RGB")
    artifacts = {}

    legend = [
        ((255, 220, 40), "object center"),
        ((255, 60, 60), "grasp target"),
        ((60, 220, 255), "pregrasp target"),
        ((180, 80, 255), "place target"),
    ]

    summary = base.copy()
    summary_draw = ImageDraw.Draw(summary)
    summary_draw.rectangle((0, 0, summary.width, 78), fill=(0, 0, 0))
    summary_draw.text((8, 8), "pre-grasp predictions | video camera", fill=(255, 255, 255))
    for idx, (color, label) in enumerate(legend):
        y = 28 + idx * 12
        summary_draw.rectangle((8, y, 18, y + 8), fill=color)
        summary_draw.text((24, y - 2), label, fill=(255, 255, 255))

    for obj in objects:
        name = obj["name"]
        image = base.copy()
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 44), fill=(0, 0, 0))
        draw.text((8, 8), f"{name} pre-grasp prediction | video camera", fill=(255, 255, 255))

        center = obj.get("center_w")
        grasp = (obj.get("grasp_pose_w") or {}).get("position")
        place = (obj.get("place_pose_w") or {}).get("position")
        pregrasp = None
        if grasp is not None:
            pregrasp = [float(grasp[0]), float(grasp[1]), float(grasp[2]) + 0.20]

        markers = [
            (center, (255, 220, 40), "center"),
            (grasp, (255, 60, 60), "grasp"),
            (pregrasp, (60, 220, 255), "pregrasp"),
            (place, (180, 80, 255), "place"),
        ]
        for point, color, label in markers:
            xy = project_world_point(camera, point) if point is not None else None
            draw_marker(draw, xy, color, label)
            draw_marker(summary_draw, xy, color, f"{name}:{label}" if label == "grasp" else "")

        file_name = f"{name}_pregrasp_prediction.png"
        image.save(prediction_dir / file_name)
        artifacts[name] = f"pregrasp_predictions/{file_name}"

    summary_name = "all_pregrasp_predictions.png"
    summary.save(prediction_dir / summary_name)
    artifacts["summary"] = f"pregrasp_predictions/{summary_name}"
    (prediction_dir / "video_camera.json").write_text(json.dumps(camera, indent=2), encoding="utf-8")
    return artifacts


class VideoCameraRecorder:
    def __init__(self, path: Path, fps: float, every_n_steps: int):
        self.path = path
        self.fps = float(fps)
        self.every_n_steps = max(1, int(every_n_steps))
        self.writer = None
        self.step = 0
        self.frames = 0

    def add(self, obs: dict, force: bool = False) -> None:
        if not force and self.step % self.every_n_steps != 0:
            self.step += 1
            return
        rgb = to_rgb(obs["image"]["video_rgb"])
        if self.writer is None:
            height, width = rgb.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not open video writer: {self.path}")
        self.writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.frames += 1
        self.step += 1

    def close(self) -> dict:
        if self.writer is not None:
            self.writer.release()
        return {
            "file": self.path.name,
            "fps": self.fps,
            "every_n_steps": self.every_n_steps,
            "frames": self.frames,
        }


def tensor_list(value) -> list[float]:
    return value.detach().cpu().numpy().astype(float).tolist()


def quat_abs_dot(a: list[float], b: list[float]) -> float:
    qa = np.asarray(a, dtype=np.float64)
    qb = np.asarray(b, dtype=np.float64)
    qa /= max(np.linalg.norm(qa), 1e-9)
    qb /= max(np.linalg.norm(qb), 1e-9)
    return float(abs(np.dot(qa, qb)))


def execute_waypoint_isaaclab(
    env,
    obs,
    robot,
    controller: CartesianController,
    arm_ids,
    gripper_ids,
    default_jpos,
    action_scale: float,
    waypoint: dict,
    object_transport_mode: str,
    attached_object: dict,
    video_recorder: VideoCameraRecorder | None = None,
) -> tuple[dict, dict]:
    pose = waypoint["pose_w"]
    position = pose["position"]
    quat = pose["quat_wxyz"]
    gripper = waypoint["gripper_joint_pos"]
    steps = int(waypoint.get("steps", 1))
    object_transport = waypoint.get("object_transport") if object_transport_mode == "kinematic_attach" else None
    if object_transport and object_transport.get("action") == "attach":
        attached_object.clear()
        attached_object.update(
            {
                "object_key": object_transport["object_key"],
                "object_name": object_transport.get("object_name"),
                "ee_to_object_pos_w": object_transport["ee_to_object_pos_w"],
                "object_quat_wxyz": object_transport["object_quat_wxyz"],
            }
        )

    pos_t = torch.tensor([position], dtype=torch.float32, device=env.unwrapped.device)
    quat_t = torch.tensor([quat], dtype=torch.float32, device=env.unwrapped.device)
    gripper_t = torch.tensor([gripper], dtype=torch.float32, device=env.unwrapped.device)

    last_action = None
    stopped = None
    reward_sum = 0.0
    transport_writes = 0
    for _ in range(steps):
        arm_des = controller.compute(pos_t, quat_t)
        target = robot.data.joint_pos.clone()
        target[:, arm_ids] = arm_des
        target[:, gripper_ids] = gripper_t
        action = (target - default_jpos) / action_scale
        last_action = action
        obs, reward, terminated, truncated, _ = env.step(action)
        reward_sum += float(reward[0].detach().cpu().item())
        robot.update(dt=env.unwrapped.physics_dt)
        if update_attached_object(env, robot, controller.ee_idx, attached_object):
            transport_writes += 1
        if video_recorder is not None:
            video_recorder.add(obs)
        if terminated.any() or truncated.any():
            stopped = {
                "terminated": bool(terminated.any().item()),
                "truncated": bool(truncated.any().item()),
            }
            break

    if object_transport and object_transport.get("action") == "release":
        release_center_w = object_transport.get("release_center_w")
        if release_center_w is not None:
            write_object_pose(
                env,
                object_transport["object_key"],
                release_center_w,
                object_transport["object_quat_wxyz"],
            )
            transport_writes += 1
        attached_object.clear()
    if transport_writes:
        obs = refresh_observation(env, obs)

    ee_pose = robot.data.body_pose_w[0, controller.ee_idx]
    ee_pos = tensor_list(ee_pose[:3])
    ee_quat = tensor_list(ee_pose[3:])
    pos_error = float(np.linalg.norm(np.asarray(ee_pos) - np.asarray(position)))
    return obs, {
        "name": waypoint["name"],
        "ok": stopped is None or bool(stopped.get("terminated", False)),
        "target_pose_w": pose,
        "target_gripper_joint_pos": gripper,
        "steps_requested": steps,
        "reward_sum": reward_sum,
        "last_action": tensor_list(last_action[0]) if last_action is not None else None,
        "ee_pose_w": {"position": ee_pos, "quat_wxyz": ee_quat},
        "position_error_m": pos_error,
        "orientation_abs_dot": quat_abs_dot(ee_quat, quat),
        "stop_reason": stopped,
        "object_transport": object_transport or None,
        "object_transport_writes": transport_writes,
    }


def collect_task_e_object_summary(env) -> dict:
    center = np.asarray(BASKET_SUCCESS_CENTER[:2], dtype=np.float64)
    scene = env.unwrapped.scene
    env_origin = scene.env_origins[0].detach().cpu().numpy()
    objects = {}
    for object_name, label in TASK_E_OBJECT_LABELS.items():
        obj = scene[object_name]
        pos_w = obj.data.root_pos_w[0, :3].detach().cpu().numpy()
        pos_local = pos_w - env_origin
        in_basket = (
            abs(float(pos_local[0]) - float(center[0])) <= float(BASKET_SUCCESS_HALF_X)
            and abs(float(pos_local[1]) - float(center[1])) <= float(BASKET_SUCCESS_HALF_Y)
            and float(TABLE_TOP_Z) <= float(pos_local[2]) <= float(TABLE_TOP_Z) + 0.15
        )
        objects[object_name] = {
            "label": label,
            "pos_w": pos_w.astype(float).tolist(),
            "pos_local": pos_local.astype(float).tolist(),
            "in_basket": bool(in_basket),
        }
    return {
        "basket_success_region": {
            "center": list(BASKET_SUCCESS_CENTER),
            "half_x": float(BASKET_SUCCESS_HALF_X),
            "half_y": float(BASKET_SUCCESS_HALF_Y),
            "z_range": [float(TABLE_TOP_Z), float(TABLE_TOP_Z) + 0.15],
        },
        "objects": objects,
        "count_in_basket": int(sum(item["in_basket"] for item in objects.values())),
        "all_in_basket": bool(all(item["in_basket"] for item in objects.values())),
    }


def apply_request_object_poses(env, request: dict) -> bool:
    """Reset Task E objects to the poses embedded in the motion request."""
    source = request.get("source", {})
    records = source.get("objects") if isinstance(source, dict) else None
    if not records:
        return False

    scene = env.unwrapped.scene
    wrote_any = False
    for record in records:
        object_key = record.get("object_key")
        center_w = record.get("center_w")
        quat_wxyz = record.get("object_quat_wxyz")
        if object_key not in scene.rigid_objects or center_w is None:
            continue
        obj = scene.rigid_objects[object_key]
        state = obj.data.default_root_state[0:1].clone()
        state[0, 0:3] = torch.tensor(center_w, dtype=state.dtype, device=state.device)
        if quat_wxyz is not None:
            state[0, 3:7] = torch.tensor(quat_wxyz, dtype=state.dtype, device=state.device)
        state[0, 7:] = 0.0
        obj.write_root_state_to_sim(state)
        wrote_any = True

    if wrote_any:
        scene.write_data_to_sim()
        env.unwrapped.sim.forward()
    return wrote_any


def fail_result(output_dir: Path, request_path: Path, request: dict, status: dict, message: str) -> None:
    result = {
        "schema": RESULT_SCHEMA,
        "ok": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "request_path": str(request_path),
        "task": request.get("task"),
        "backend": status,
        "error": message,
        "waypoints": [],
        "artifacts": {},
    }
    (output_dir / "motion_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[ERROR] {message}")
    print(f"[INFO] Saved failed motion result: {output_dir / 'motion_result.json'}")


def main() -> None:
    request_path = resolve_request(args_cli.request)
    request = load_request(request_path)
    output_dir = make_output_dir(args_cli.output)
    status = backend_status(request, args_cli.backend, args_cli.no_fallback)
    if status["used"] is None:
        fail_result(output_dir, request_path, request, status, status["reason"])
        simulation_app.close()
        return
    if status["used"] == "moveit_py":
        status["used"] = "isaaclab_cartesian_controller"
        status["reason"] = "moveit_py import exists, but Task E simulator execution currently uses IsaacLab fallback"

    robot_cfg = request.get("robot", {})
    arm_joint_names = robot_cfg.get("arm_joints", DEFAULT_ARM_JOINTS)
    gripper_joint_names = robot_cfg.get("gripper_joints", DEFAULT_GRIPPER_JOINTS)
    ee_link = robot_cfg.get("ee_link", "gripper_base")
    action_scale = float(robot_cfg.get("action_scale", 0.5))

    env_cfg = parse_env_cfg(
        "ATEC-TaskE-Piper",
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    actuator_mode = resolve_actuator_mode(request)
    object_transport_mode = resolve_object_transport_mode(request)
    apply_actuator_mode(env_cfg, actuator_mode)
    env_cfg.episode_length_s = max(
        float(getattr(env_cfg, "episode_length_s", 0.0)),
        90.0,
    )
    seed = args_cli.seed if args_cli.seed is not None else request.get("start_state", {}).get("seed")
    env_cfg.seed = seed
    env = gym.make("ATEC-TaskE-Piper", cfg=env_cfg)
    obs, _ = env.reset()
    object_poses_applied = apply_request_object_poses(env, request)
    if object_poses_applied:
        obs = refresh_observation(env, obs)
    robot = env.unwrapped.scene.articulations["robot"]
    arm_ids, _ = robot.find_joints(arm_joint_names)
    gripper_ids, _ = robot.find_joints(gripper_joint_names)
    controller = CartesianController(
        robot=robot,
        ee_body_name=ee_link,
        arm_joint_names=arm_joint_names,
        num_envs=1,
        device=env.unwrapped.device,
        command_type="pose",
        lambda_val=0.05,
        max_joint_delta=0.16,
    )
    controller.reset()
    default_jpos = robot.data.default_joint_pos.clone()

    artifacts = {"frames": {}}
    video_recorder = None
    if args_cli.record_video_cam:
        video_recorder = VideoCameraRecorder(
            output_dir / "video_cam.mp4",
            fps=args_cli.video_fps,
            every_n_steps=args_cli.video_every_n_steps,
        )
        video_recorder.add(obs, force=True)

    if args_cli.save_pregrasp_viz:
        artifacts["pregrasp_predictions"] = save_pregrasp_prediction_viz(
            output_dir,
            obs,
            request,
            env,
        )

    if not args_cli.no_save_frames:
        save_frame(output_dir / "00_initial.png", obs, "initial")
        artifacts["frames"]["00_initial"] = "00_initial.png"

    waypoint_results = []
    total_reward = 0.0
    attached_object: dict = {}
    for waypoint in request["waypoints"]:
        obs, waypoint_result = execute_waypoint_isaaclab(
            env,
            obs,
            robot,
            controller,
            arm_ids,
            gripper_ids,
            default_jpos,
            action_scale,
            waypoint,
            object_transport_mode,
            attached_object,
            video_recorder,
        )
        waypoint_results.append(waypoint_result)
        total_reward += float(waypoint_result["reward_sum"])
        if not args_cli.no_save_frames and waypoint.get("capture", True):
            frame_name = f"{waypoint['name']}.png"
            save_frame(output_dir / frame_name, obs, waypoint["name"])
            artifacts["frames"][waypoint["name"]] = frame_name
        print(
            f"[INFO] {waypoint['name']}: error={waypoint_result['position_error_m']:.4f} "
            f"ee={np.round(waypoint_result['ee_pose_w']['position'], 4).tolist()}"
        )
        if waypoint_result["stop_reason"] is not None:
            break

    if video_recorder is not None:
        artifacts["video_cam"] = video_recorder.close()

    result = {
        "schema": RESULT_SCHEMA,
        "ok": all(item["ok"] for item in waypoint_results),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "request_path": str(request_path),
        "task": request["task"],
        "seed": seed,
        "backend": status,
        "controller": {
            "actuator_mode": actuator_mode,
            "object_transport_mode": object_transport_mode,
            "actuator": TASK_E_SCRIPTED_ACTUATOR if actuator_mode == "task_e_scripted_high_stiffness" else None,
            "request_object_poses_applied": object_poses_applied,
        },
        "source": request.get("source"),
        "robot": robot_cfg,
        "total_reward": total_reward,
        "waypoints": waypoint_results,
        "task_e_objects": {"status": "pending"},
        "artifacts": artifacts,
    }
    (output_dir / "motion_request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    (output_dir / "motion_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args_cli.skip_object_summary:
        result["task_e_objects"] = {"status": "skipped"}
    else:
        try:
            result["task_e_objects"] = collect_task_e_object_summary(env)
        except Exception as exc:
            result["task_e_objects"] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
    (output_dir / "motion_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    env.close()
    simulation_app.close()
    print(f"[INFO] Saved motion result: {output_dir / 'motion_result.json'}")


if __name__ == "__main__":
    main()
