"""Scripted oracle data collection for Task E (pick-and-place).

Usage
-----
# Object-1 smoke trial with MP4 output.
python scripts/act/collect_demos_task_e.py \
  --pick_objects 1 --num_demos 2 \
  --headless --enable_cameras --save_video \
  --output_dir datasets/atec_task_e_object1_trial

"""

import argparse
import os
import subprocess
import sys
import traceback

# sys.path.insert(0, os.path.dirname(__file__))

from isaaclab.app import AppLauncher
from cli_args import add_collect_demo_args

parser = argparse.ArgumentParser(description="Collect Task E demonstrations for ACT.")
add_collect_demo_args(parser)
parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--_append", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--_traj_start_idx", type=int, default=0, help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Task E's scene includes camera sensors even when we do not save frames.
args_cli.enable_cameras = True

import h5py
import json


def _load_worker_modules() -> None:
    """Import IsaacLab-dependent modules after AppLauncher starts the app."""
    global ImplicitActuatorCfg, ManagerBasedRLEnv
    global TaskEEnvPiperCfg, CartesianController
    global EE_BODY_NAME, ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES
    global ACT_STIFFNESS, ACT_DAMPING, ACT_EFFORT_LIMIT, ACT_VEL_LIMIT
    global STATE_ORDER
    global check_objects_in_basket, collect_one_demo

    from isaaclab.actuators import ImplicitActuatorCfg as _ImplicitActuatorCfg
    from isaaclab.envs import ManagerBasedRLEnv as _ManagerBasedRLEnv
    from atec_rl_lab.tasks.task_e.env_cfg import TaskEEnvPiperCfg as _TaskEEnvPiperCfg
    from atec_rl_lab.utils import CartesianController as _CartesianController
    from task_e.config import (
        EE_BODY_NAME as _EE_BODY_NAME,
        ARM_JOINT_NAMES as _ARM_JOINT_NAMES,
        GRIPPER_JOINT_NAMES as _GRIPPER_JOINT_NAMES,
        ACT_STIFFNESS as _ACT_STIFFNESS,
        ACT_DAMPING as _ACT_DAMPING,
        ACT_EFFORT_LIMIT as _ACT_EFFORT_LIMIT,
        ACT_VEL_LIMIT as _ACT_VEL_LIMIT,
        STATE_ORDER as _STATE_ORDER,
    )
    from task_e.collector import (
        check_objects_in_basket as _check_objects_in_basket,
        collect_one_demo as _collect_one_demo,
    )

    ImplicitActuatorCfg = _ImplicitActuatorCfg
    ManagerBasedRLEnv = _ManagerBasedRLEnv
    TaskEEnvPiperCfg = _TaskEEnvPiperCfg
    CartesianController = _CartesianController
    EE_BODY_NAME = _EE_BODY_NAME
    ARM_JOINT_NAMES = _ARM_JOINT_NAMES
    GRIPPER_JOINT_NAMES = _GRIPPER_JOINT_NAMES
    ACT_STIFFNESS = _ACT_STIFFNESS
    ACT_DAMPING = _ACT_DAMPING
    ACT_EFFORT_LIMIT = _ACT_EFFORT_LIMIT
    ACT_VEL_LIMIT = _ACT_VEL_LIMIT
    STATE_ORDER = _STATE_ORDER
    check_objects_in_basket = _check_objects_in_basket
    collect_one_demo = _collect_one_demo


def build_env(pick_objects: list[int], need_camera: bool, seed: int | None = None):
    import time
    seed = int(time.time_ns() % (2**31)) if seed is None else int(seed)
    cfg = TaskEEnvPiperCfg()
    cfg.seed              = seed
    cfg.scene.num_envs    = 1
    cfg.episode_length_s  = 40.0 * len(pick_objects) + 10.0
    cfg.scene.robot.actuators["default"] = ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        effort_limit=ACT_EFFORT_LIMIT,
        velocity_limit=ACT_VEL_LIMIT,
        stiffness=ACT_STIFFNESS,
        damping=ACT_DAMPING,
    )
    print(f"[INFO] Building env with seed {seed}")
    return ManagerBasedRLEnv(cfg), seed


def build_runtime(pick_objects: list[int], need_camera: bool, seed: int | None = None):
    """Construct a fresh env and controller for one collection attempt."""
    env, seed = build_env(pick_objects, need_camera, seed=seed)
    dev = env.unwrapped.device
    camera = env.unwrapped.scene["video_cam"] if need_camera else None

    robot = env.unwrapped.scene.articulations["robot"]
    arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
    gripper_ids, _ = robot.find_joints(GRIPPER_JOINT_NAMES)
    ik_ctrl = CartesianController(
        robot=robot, ee_body_name=EE_BODY_NAME,
        arm_joint_names=ARM_JOINT_NAMES,
        num_envs=1, device=dev,
        command_type="pose",
        lambda_val=0.05,
        max_joint_delta=0.2,
    )
    ik_pos_ctrl = CartesianController(
        robot=robot, ee_body_name=EE_BODY_NAME,
        arm_joint_names=ARM_JOINT_NAMES,
        num_envs=1, device=dev,
        command_type="position",
        lambda_val=0.05,
        max_joint_delta=0.2,
    )
    default_jpos = robot.data.default_joint_pos.clone()
    return env, dev, camera, robot, arm_ids, gripper_ids, ik_ctrl, ik_pos_ctrl, default_jpos, seed


def init_output(output_dir: str) -> tuple[str, str]:
    """Create output directory, wipe any existing trajectory.hdf5, write JSON metadata."""
    os.makedirs(output_dir, exist_ok=True)
    traj_path = os.path.join(output_dir, "trajectory.hdf5")
    json_path = os.path.join(output_dir, "trajectory.json")
    with h5py.File(traj_path, "w"):   # truncate / create fresh
        pass
    with open(json_path, "w") as fh:
        json.dump({"env_info": {"env_kwargs": {"control_mode": "pd_joint_pos"}}}, fh)
    return traj_path, json_path


def save_traj(traj_path: str, traj_idx: int, data: dict,
              save_images: bool) -> None:
    """Append one trajectory group to the consolidated HDF5."""
    with h5py.File(traj_path, "a") as f:
        grp = f.create_group(f"traj_{traj_idx}")
        grp.create_dataset("obs",     data=data["qpos"],    compression="gzip")
        grp.create_dataset("actions", data=data["action"],  compression="gzip")
        grp.create_dataset("qvel",    data=data["qvel"],    compression="gzip")
        grp.create_dataset("ee_pos",  data=data["ee_pos"],  compression="gzip")
        grp.create_dataset("ee_quat", data=data["ee_quat"], compression="gzip")
        for key in (
            "target_ee_pos", "target_ee_quat", "target_error", "object_pos",
            "state_id", "object_idx", "controller_mode", "gripper_cmd",
        ):
            if key in data:
                grp.create_dataset(key, data=data[key], compression="gzip")
        if "seed" in data:
            grp.attrs["seed"] = int(data["seed"])
        if "position_priority_orientation_weight" in data:
            grp.attrs["position_priority_orientation_weight"] = float(
                data["position_priority_orientation_weight"]
            )
        if "STATE_ORDER" in globals():
            grp.attrs["state_order"] = json.dumps(STATE_ORDER)
        if save_images and "frames" in data:
            grp.create_group("images").create_dataset(
                "rgb", data=data["frames"], compression="gzip"
            )


_SKIP_ATTEMPT_CODE = 20


def _seed_for_attempt(args: argparse.Namespace, attempt_idx: int) -> int | None:
    """Return the seed candidate for an outer collection attempt, if configured."""
    if args.seeds is not None:
        if attempt_idx >= len(args.seeds):
            raise ValueError(
                f"Ran out of --seeds candidates after {len(args.seeds)} attempts. "
                "Provide more seed candidates or increase the sweep script's seed pool."
            )
        return int(args.seeds[attempt_idx])
    if args.seed is not None:
        return int(args.seed) + int(args.seed_stride) * int(attempt_idx)
    return None


def _worker_command(args: argparse.Namespace, traj_idx: int, attempt_idx: int) -> list[str]:
    script_path = os.path.abspath(__file__)
    cmd = [
        sys.executable, script_path,
        "--_worker",
        "--_append",
        "--_traj_start_idx", str(traj_idx),
        "--num_demos", "1",
        "--output_dir", args.output_dir,
        "--pick_objects", *[str(i) for i in sorted(set(args.pick_objects))],
    ]
    worker_seed = _seed_for_attempt(args, attempt_idx)
    if worker_seed is not None:
        cmd += ["--seed", str(worker_seed)]
    if args.position_priority_orientation_weight is not None:
        cmd += [
            "--position_priority_orientation_weight",
            str(args.position_priority_orientation_weight),
        ]
    if args.save_video:
        cmd.append("--save_video")
    if args.video_dir:
        cmd += ["--video_dir", args.video_dir]
    if args.save_images:
        cmd.append("--save_images")
    if args.only_success:
        cmd.append("--only_success")
    if getattr(args, "headless", False):
        cmd.append("--headless")
    if getattr(args, "enable_cameras", False):
        cmd.append("--enable_cameras")
    if getattr(args, "device", None):
        cmd += ["--device", str(args.device)]
    return cmd


def main_outer() -> None:
    if args_cli.seeds is not None and len(args_cli.seeds) < args_cli.num_demos:
        raise ValueError(
            f"--seeds has {len(args_cli.seeds)} values but --num_demos={args_cli.num_demos}."
        )
    init_output(args_cli.output_dir)
    if args_cli.save_video:
        video_dir = args_cli.video_dir or os.path.join(args_cli.output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

    n_ok = 0
    attempt = 0
    while n_ok < args_cli.num_demos:
        if args_cli.max_attempts and attempt >= args_cli.max_attempts:
            raise RuntimeError(
                f"Reached --max_attempts={args_cli.max_attempts} "
                f"after collecting {n_ok}/{args_cli.num_demos} demos."
            )
        attempt_idx = attempt
        attempt += 1
        worker_seed = _seed_for_attempt(args_cli, attempt_idx)
        seed_note = f", seed {worker_seed}" if worker_seed is not None else ""
        print(
            f"\n[INFO] Demo {n_ok + 1}/{args_cli.num_demos}  "
            f"(fresh-seed attempt {attempt}{seed_note})"
        )
        result = subprocess.run(_worker_command(args_cli, n_ok, attempt_idx))
        if result.returncode == 0:
            n_ok += 1
        elif result.returncode == _SKIP_ATTEMPT_CODE:
            print("[WARN] Rejected attempt — retrying with a fresh Isaac process/seed.")
        else:
            raise subprocess.CalledProcessError(result.returncode, result.args)

    traj_path = os.path.join(args_cli.output_dir, "trajectory.hdf5")
    print(f"\n[INFO] Collected {n_ok} demos → {traj_path}")


def main_worker() -> int:
    pick_objects = sorted(set(args_cli.pick_objects))
    need_camera = args_cli.save_video or args_cli.save_images

    video_dir = None
    imageio   = None
    if args_cli.save_video:
        video_dir = args_cli.video_dir or os.path.join(args_cli.output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        import imageio as _io
        imageio = _io

    if args_cli._append:
        traj_path = os.path.join(args_cli.output_dir, "trajectory.hdf5")
    else:
        traj_path, _ = init_output(args_cli.output_dir)

    n_ok = 0
    attempt = 0
    while n_ok < args_cli.num_demos:
        if args_cli.max_attempts and attempt >= args_cli.max_attempts:
            raise RuntimeError(
                f"Reached --max_attempts={args_cli.max_attempts} "
                f"after collecting {n_ok}/{args_cli.num_demos} demos."
            )
        attempt += 1
        print(f"\n[INFO] Demo {n_ok + 1}/{args_cli.num_demos}  (attempt {attempt})")

        (
            env, dev, camera, robot, arm_ids, gripper_ids,
            ik_ctrl, ik_pos_ctrl, default_jpos, seed,
        ) = build_runtime(pick_objects, need_camera, seed=args_cli.seed)
        try:
            data = collect_one_demo(
                env, robot, ik_ctrl, ik_pos_ctrl,
                arm_ids, gripper_ids,
                pick_objects, dev,
                default_jpos=default_jpos,
                position_priority_orientation_weight=args_cli.position_priority_orientation_weight,
                camera=camera,
            )
            if data is None:
                print("[WARN] Invalid start or early termination — rebuilding with a new seed.")
                return _SKIP_ATTEMPT_CODE

            if args_cli.only_success and not check_objects_in_basket(env, pick_objects):
                print("[WARN] Objects not in basket — skipping (--only_success).")
                return _SKIP_ATTEMPT_CODE

            traj_idx = args_cli._traj_start_idx + n_ok
            data["seed"] = seed
            save_traj(traj_path, traj_idx, data, args_cli.save_images)

            T     = len(data["qpos"])
            notes = [f"{T} steps"]
            if args_cli.save_video and "frames" in data:
                vp = os.path.join(video_dir, f"demo_{traj_idx:04d}.mp4")
                imageio.mimwrite(vp, data["frames"], fps=50, quality=7)
                notes.append(f"video → {vp}")
            if args_cli.save_images and "frames" in data:
                notes.append("images saved")
            print(f"[INFO] traj_{traj_idx}: {', '.join(notes)}")
            n_ok += 1
        finally:
            # Worker mode exits the Python process after this attempt. Avoid
            # env.close() here because Isaac teardown can hang between retries.
            pass

    print(f"\n[INFO] Collected {n_ok} demos → {traj_path}")
    return 0


if __name__ == "__main__":
    if args_cli._worker:
        app_launcher   = AppLauncher(args_cli)
        simulation_app = app_launcher.app
        exit_code = 1
        try:
            _load_worker_modules()
            exit_code = main_worker()
        except SystemExit as exc:
            exit_code = int(exc.code or 0)
        except Exception:
            traceback.print_exc()
            exit_code = 1
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
    main_outer()
