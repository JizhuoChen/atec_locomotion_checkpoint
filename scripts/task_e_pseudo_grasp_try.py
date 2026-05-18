#!/usr/bin/env python3
"""Try the pseudo banana grasp in Task E using simulator IK."""

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


DEFAULT_GRASP = REPO_ROOT / "outputs/task_e_banana_pipeline/latest/pseudo_grasp/pseudo_grasp.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp-json", type=Path, default=DEFAULT_GRASP)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/task_e_pseudo_grasp_try/<timestamp>.",
    )
    parser.add_argument("--camera-steps", type=int, default=160)
    parser.add_argument("--pregrasp-steps", type=int, default=180)
    parser.add_argument("--descend-steps", type=int, default=140)
    parser.add_argument("--close-steps", type=int, default=70)
    parser.add_argument("--lift-steps", type=int, default=150)
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
from PIL import Image, ImageDraw  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401,E402
from atec_rl_lab.utils import CartesianController  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINT_NAMES = ["joint7", "joint8"]
ACTION_SCALE = 0.5


def resolve_grasp_json(path: Path) -> Path:
    if path.exists():
        return path.resolve()
    if "latest" in path.parts:
        latest_txt = REPO_ROOT / "outputs/task_e_banana_pipeline/latest.txt"
        if latest_txt.exists():
            latest = Path(latest_txt.read_text(encoding="utf-8").strip())
            candidate = latest / "pseudo_grasp/pseudo_grasp.json"
            if candidate.exists():
                return candidate.resolve()
    raise FileNotFoundError(f"Missing pseudo grasp JSON: {path}")


def make_output_dir(path: Path | None) -> Path:
    if path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPO_ROOT / "outputs/task_e_pseudo_grasp_try" / timestamp
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


def to_rgb(value) -> np.ndarray:
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


def tensor_list(value) -> list[float]:
    return value.detach().cpu().numpy().astype(float).tolist()


def move_to_pose(
    env,
    obs,
    robot,
    controller: CartesianController,
    arm_ids,
    gripper_ids,
    default_jpos,
    position: list[float],
    quat: list[float],
    gripper: list[float],
    steps: int,
) -> tuple[dict, list[float] | None]:
    pos_t = torch.tensor([position], dtype=torch.float32, device=env.unwrapped.device)
    quat_t = torch.tensor([quat], dtype=torch.float32, device=env.unwrapped.device)
    gripper_t = torch.tensor([gripper], dtype=torch.float32, device=env.unwrapped.device)
    last_action = None
    for _ in range(steps):
        arm_des = controller.compute(pos_t, quat_t)
        target = robot.data.joint_pos.clone()
        target[:, arm_ids] = arm_des
        target[:, gripper_ids] = gripper_t
        action = (target - default_jpos) / ACTION_SCALE
        last_action = action
        obs, _, terminated, truncated, _ = env.step(action)
        robot.update(dt=env.unwrapped.physics_dt)
        if terminated.any() or truncated.any():
            break
    return obs, tensor_list(last_action[0]) if last_action is not None else None


def moveit_status() -> dict:
    modules = {}
    for name in ("moveit", "moveit_py"):
        try:
            __import__(name)
            modules[name] = "available"
        except Exception as exc:
            modules[name] = f"unavailable: {type(exc).__name__}: {exc}"
    return {
        "moveit2_source": str((REPO_ROOT / "third_party/moveit2").resolve()),
        "python_modules": modules,
        "solver_used": "isaaclab_cartesian_controller_fallback",
    }


def main() -> None:
    output_dir = make_output_dir(args_cli.output)
    grasp_path = resolve_grasp_json(args_cli.grasp_json)
    pseudo = json.loads(grasp_path.read_text(encoding="utf-8"))

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
    controller = CartesianController(
        robot=robot,
        ee_body_name="gripper_base",
        arm_joint_names=ARM_JOINT_NAMES,
        num_envs=1,
        device=env.unwrapped.device,
        command_type="pose",
        lambda_val=0.05,
        max_joint_delta=0.16,
    )
    controller.reset()
    default_jpos = robot.data.default_joint_pos.clone()

    open_gripper = pseudo["gripper"]["open_joint7_joint8"]
    closed_gripper = pseudo["gripper"]["closed_joint7_joint8"]
    topdown_quat = pseudo["grasp_pose_w"]["quat_wxyz"]
    stages = [
        (
            "00_reset",
            pseudo["camera_look_pose_w"]["position"] or pseudo["pregrasp_pose_w"]["position"],
            pseudo["camera_look_pose_w"]["quat_wxyz"] or topdown_quat,
            open_gripper,
            args_cli.camera_steps,
        ),
        (
            "01_pregrasp",
            pseudo["pregrasp_pose_w"]["position"],
            topdown_quat,
            open_gripper,
            args_cli.pregrasp_steps,
        ),
        (
            "02_grasp",
            pseudo["grasp_pose_w"]["position"],
            topdown_quat,
            open_gripper,
            args_cli.descend_steps,
        ),
        (
            "03_close",
            pseudo["grasp_pose_w"]["position"],
            topdown_quat,
            closed_gripper,
            args_cli.close_steps,
        ),
        (
            "04_lift",
            pseudo["lift_pose_w"]["position"],
            topdown_quat,
            closed_gripper,
            args_cli.lift_steps,
        ),
    ]

    records = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pseudo_grasp_json": str(grasp_path),
        "moveit": moveit_status(),
        "stages": [],
    }
    save_frame(output_dir / "00_initial.png", obs, "initial")
    for name, position, quat, gripper, steps in stages:
        obs, last_action = move_to_pose(
            env,
            obs,
            robot,
            controller,
            arm_ids,
            gripper_ids,
            default_jpos,
            position,
            quat,
            gripper,
            steps,
        )
        ee_pose = robot.data.body_pose_w[0, controller.ee_idx]
        record = {
            "name": name,
            "target_position_w": position,
            "target_quat_wxyz": quat,
            "gripper_target": gripper,
            "steps": steps,
            "last_action": last_action,
            "ee_pos_w": tensor_list(ee_pose[:3]),
            "ee_quat_wxyz": tensor_list(ee_pose[3:]),
        }
        records["stages"].append(record)
        save_frame(output_dir / f"{name}.png", obs, name)
        print(f"[INFO] {name}: ee={np.round(record['ee_pos_w'], 4).tolist()}")

    (output_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    env.close()
    simulation_app.close()
    print(f"[INFO] Saved pseudo grasp try output to {output_dir}")


if __name__ == "__main__":
    main()
