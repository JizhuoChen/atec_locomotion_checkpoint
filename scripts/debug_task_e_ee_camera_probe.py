"""Probe Task E eye-in-hand camera views at scripted observation poses."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "act"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output",
    default="outputs/task_e_ee_camera_probe",
    help="Directory for probe images.",
)
parser.add_argument("--steps", type=int, default=160, help="Control steps per pose.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401,E402
from atec_rl_lab.tasks.task_e.env_cfg import (  # noqa: E402
    TABLE_CENTER_X,
    TABLE_CENTER_Y,
    TABLE_TOP_Z,
    TABLE_HALF_X,
)
from atec_rl_lab.utils import CartesianController  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINT_NAMES = ["joint7", "joint8"]
GRIPPER_OPEN_POS = [0.035, -0.035]
ACTION_SCALE = 0.5


def to_rgb(value) -> np.ndarray:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def save_labeled(path: Path, image: np.ndarray, text: str) -> None:
    pil = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(pil)
    draw.rectangle((0, 0, pil.width, 22), fill=(0, 0, 0))
    draw.text((6, 4), text, fill=(255, 255, 255))
    pil.save(path)


def main() -> None:
    output_dir = Path(args_cli.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        "ATEC-TaskE-Piper",
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make("ATEC-TaskE-Piper", cfg=env_cfg)
    obs, _ = env.reset()

    robot = env.unwrapped.scene.articulations["robot"]
    arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
    gripper_ids, _ = robot.find_joints(GRIPPER_JOINT_NAMES)
    ik_ctrl = CartesianController(
        robot=robot,
        ee_body_name="gripper_base",
        arm_joint_names=ARM_JOINT_NAMES,
        num_envs=1,
        device=env.unwrapped.device,
        command_type="pose",
        lambda_val=0.05,
        max_joint_delta=0.2,
    )
    default_jpos = robot.data.default_joint_pos.clone()
    gripper_target = torch.tensor([GRIPPER_OPEN_POS], dtype=torch.float32, device=env.unwrapped.device)

    poses = [
        (
            "reset",
            None,
            None,
        ),
        (
            "home_top_down",
            (TABLE_CENTER_X + TABLE_HALF_X - 0.05, TABLE_CENTER_Y, TABLE_TOP_Z + 0.40),
            (0.0, 1.0, 0.0, 0.0),
        ),
        (
            "table_center_top_down",
            (TABLE_CENTER_X + 0.02, TABLE_CENTER_Y + 0.08, TABLE_TOP_Z + 0.38),
            (0.0, 1.0, 0.0, 0.0),
        ),
        (
            "table_center_lower",
            (TABLE_CENTER_X + 0.02, TABLE_CENTER_Y + 0.08, TABLE_TOP_Z + 0.25),
            (0.0, 1.0, 0.0, 0.0),
        ),
        (
            "basket_top_down",
            (TABLE_CENTER_X + 0.08, TABLE_CENTER_Y - 0.22, TABLE_TOP_Z + 0.40),
            (0.0, 1.0, 0.0, 0.0),
        ),
    ]

    records = []
    for pose_name, pos, quat in poses:
        if pos is not None:
            ik_ctrl.reset()
            ee_pos = torch.tensor([pos], dtype=torch.float32, device=env.unwrapped.device)
            ee_quat = torch.tensor([quat], dtype=torch.float32, device=env.unwrapped.device)
            for _ in range(args_cli.steps):
                arm_des = ik_ctrl.compute(ee_pos, ee_quat)
                target = robot.data.joint_pos.clone()
                target[:, arm_ids] = arm_des
                target[:, gripper_ids] = gripper_target
                action = (target - default_jpos) / ACTION_SCALE
                obs, _, terminated, truncated, _ = env.step(action)
                robot.update(dt=env.unwrapped.physics_dt)
                if terminated.any() or truncated.any():
                    break

        full_target = robot.data.joint_pos[0].detach().cpu().numpy()
        env_action = ((robot.data.joint_pos - default_jpos) / ACTION_SCALE)[0].detach().cpu().numpy()
        ee_pos_w = robot.data.body_pose_w[0, ik_ctrl.ee_idx, :3].detach().cpu().numpy()
        ee_quat_w = robot.data.body_pose_w[0, ik_ctrl.ee_idx, 3:].detach().cpu().numpy()
        label = f"{pose_name} q={np.round(full_target, 3).tolist()}"

        save_labeled(output_dir / f"{pose_name}_ee_rgb.png", to_rgb(obs["image"]["ee_rgb"]), label)
        save_labeled(output_dir / f"{pose_name}_video_rgb.png", to_rgb(obs["image"]["video_rgb"]), label)
        records.append(
            {
                "name": pose_name,
                "joint_pos": full_target.tolist(),
                "action": env_action.tolist(),
                "ee_pos_w": ee_pos_w.tolist(),
                "ee_quat_w": ee_quat_w.tolist(),
            }
        )
        print(f"{pose_name}: action={np.round(env_action, 4).tolist()} ee_pos={np.round(ee_pos_w, 4).tolist()}")

    (output_dir / "records.json").write_text(
        __import__("json").dumps(records, indent=2), encoding="utf-8"
    )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
