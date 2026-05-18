#!/usr/bin/env python3
"""Capture Task E video and eye-in-hand camera frames as PNG files."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher


TABLE_LOOK_ACTION = [
    -0.41051948070526123,
    1.5470619201660156,
    -0.38924479484558105,
    0.000016453657735837623,
    0.039999961853027344,
    -0.4105374217033386,
    -0.0000010281801223754883,
    0.0000001043081283569336,
]


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/task_e_camera_capture/<timestamp>.",
    )
    parser.add_argument(
        "--look-at-table",
        action="store_true",
        help="Drive the arm to the verified table-looking EE camera pose before capture.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=160,
        help="Number of control steps for --look-at-table or --action.",
    )
    parser.add_argument(
        "--action",
        type=parse_float_list,
        default=None,
        help="Comma-separated raw env action to apply before capture.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Task E environment seed.")
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401,E402
from atec_rl_lab.tasks.task_e.env_cfg import CAM_POS, CAM_ROT  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def make_output_dir(path: Path | None) -> Path:
    if path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("outputs/task_e_camera_capture") / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def to_rgb_array(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)

    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected RGB tensor with 3 or 4 dims, got {array.shape}")

    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))

    if array.shape[-1] == 4:
        array = array[:, :, :3]
    elif array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] != 3:
        raise ValueError(f"Expected RGB/RGBA channels, got {array.shape}")

    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        if array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        array = np.clip(array, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(array)


def save_rgb(path: Path, value) -> dict:
    rgb = to_rgb_array(value)
    Image.fromarray(rgb, mode="RGB").save(path)
    return {"file": path.name, "shape": list(rgb.shape), "dtype": str(rgb.dtype)}


def action_to_tensor(action: list[float], action_dim: int, device: str) -> torch.Tensor:
    values = list(action[:action_dim])
    if len(values) < action_dim:
        values.extend([0.0] * (action_dim - len(values)))
    return torch.tensor(values, dtype=torch.float32, device=device).view(1, -1)


def find_single_body(robot, body_name: str) -> int | None:
    body_ids, _ = robot.find_bodies(body_name)
    return body_ids[0] if len(body_ids) == 1 else None


def tensor_list(value) -> list[float] | None:
    if value is None:
        return None
    return value.detach().cpu().tolist()


def main() -> None:
    output_dir = make_output_dir(args_cli.output)
    env_cfg = parse_env_cfg(
        "ATEC-TaskE-Piper",
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env = gym.make("ATEC-TaskE-Piper", cfg=env_cfg)
    obs, _ = env.reset()

    robot = env.unwrapped.scene.articulations["robot"]
    action_dim = int(robot.data.default_joint_pos.shape[-1])
    applied_action = None
    mode = "reset"

    action = args_cli.action
    if action is None and args_cli.look_at_table:
        action = TABLE_LOOK_ACTION
    if action is not None:
        mode = "custom_action" if args_cli.action is not None else "look_at_table"
        applied_action = list(action[:action_dim])
        action_tensor = action_to_tensor(action, action_dim, env.unwrapped.device)
        for _ in range(args_cli.steps):
            obs, _, terminated, truncated, _ = env.step(action_tensor)
            if terminated.any() or truncated.any():
                break

    image_obs = obs["image"]
    views = {
        "video_rgb": save_rgb(output_dir / "video_rgb.png", image_obs["video_rgb"]),
        "ee_rgb": save_rgb(output_dir / "ee_rgb.png", image_obs["ee_rgb"]),
    }

    ee_body_idx = find_single_body(robot, "gripper_base")
    ee_pose = robot.data.body_pose_w[0, ee_body_idx] if ee_body_idx is not None else None
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": "ATEC-TaskE-Piper",
        "seed": args_cli.seed,
        "mode": mode,
        "steps": args_cli.steps if action is not None else 0,
        "applied_action": applied_action,
        "views": views,
        "video_camera_config": {
            "source": "source/atec_rl_lab/atec_rl_lab/tasks/task_e/env_cfg.py",
            "prim_path": "{ENV_REGEX_NS}/video_cam",
            "pos": list(CAM_POS),
            "rot": list(CAM_ROT),
            "convention": "world",
        },
        "ee_camera_config": {
            "source": "source/atec_rl_lab/atec_rl_lab/assets/robots/piper.py",
            "link": "gripper_base",
            "offset_pos": [-0.05, 0.0, 0.06],
            "offset_rot_euler_xyz": [0.0, 0.0, "-pi/2"],
            "convention": "ros",
        },
        "robot_joint_pos": tensor_list(robot.data.joint_pos[0]),
        "robot_default_joint_pos": tensor_list(robot.data.default_joint_pos[0]),
        "ee_body_pose_w": {
            "pos": tensor_list(ee_pose[:3]) if ee_pose is not None else None,
            "quat_wxyz": tensor_list(ee_pose[3:]) if ee_pose is not None else None,
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"[INFO] Saved Task E camera frames to {output_dir}")
    print(f"[INFO] video_rgb: {output_dir / 'video_rgb.png'}")
    print(f"[INFO] ee_rgb: {output_dir / 'ee_rgb.png'}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
