"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import hashlib
import json
import os
import re
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--terrain_showcase",
    action="store_true",
    default=False,
    help="Record one tracked-policy video for every configured rough-terrain family.",
)
parser.add_argument(
    "--showcase_steps_per_terrain",
    type=int,
    default=1000,
    help="Number of policy steps recorded for each terrain in --terrain_showcase mode.",
)
parser.add_argument(
    "--showcase_difficulty",
    type=float,
    default=0.75,
    help="Fixed normalized terrain difficulty in --terrain_showcase mode (0 to 1).",
)
parser.add_argument(
    "--showcase_output_name",
    type=str,
    default=None,
    help=(
        "Relative output folder below videos/play for --terrain_showcase. "
        "Safe slash-separated names are accepted; an automatic difficulty/seed name is used by default."
    ),
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--eval_steps", type=int, default=2000, help="Finite policy steps for quantitative evaluation.")
parser.add_argument("--eval_output", type=str, required=True, help="JSON output path below /data/steven.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.eval_steps <= 0:
    parser.error("--eval_steps must be positive")
args_cli.eval_output = os.path.abspath(os.path.expanduser(args_cli.eval_output))
if os.path.commonpath(("/data/steven", args_cli.eval_output)) != "/data/steven":
    parser.error("--eval_output must remain below /data/steven")
# always enable cameras to record video
if args_cli.terrain_showcase:
    args_cli.video = True
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import atec_rl_lab.train  # noqa: F401  # isort: skip
from atec_rl_lab.train.locomotion.velocity.hybrid_distillation_ppo import HybridOnPolicyRunner
from terrain_camera import (
    TerrainFamilyCameraCycle,
    terrain_family_representatives,
    terrain_showcase_column_allocation,
    terrain_video_manifest,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# PLACEHOLDER: Extension template (do not remove this comment)


SHOWCASE_PATH_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")


def resolve_showcase_output_dir(
    play_video_dir: str, requested_name: str | None, difficulty: float, seed: int
) -> tuple[str, str]:
    """Return a safe, non-overwriting showcase directory and its relative name."""
    if requested_name is None:
        difficulty_tag = f"d{round(difficulty * 100):03d}"
        requested_name = f"terrain-showcase-{difficulty_tag}-seed{seed}"
    components = requested_name.split("/")
    if not components or any(
        SHOWCASE_PATH_COMPONENT_RE.fullmatch(component) is None for component in components
    ):
        raise ValueError(
            "--showcase_output_name must contain only safe slash-separated components made of "
            "letters, numbers, '.', '_', or '-' (no empty, '.' or '..' components)."
        )
    relative_name = os.path.join(*components)
    output_dir = os.path.abspath(os.path.join(play_video_dir, relative_name))
    if os.path.commonpath((os.path.abspath(play_video_dir), output_dir)) != os.path.abspath(
        play_video_dir
    ):
        raise ValueError("--showcase_output_name must remain below the checkpoint's videos/play directory.")
    if os.path.exists(output_dir):
        raise FileExistsError(
            f"Showcase output already exists and will not be overwritten: {output_dir}. "
            "Choose a different --showcase_output_name."
        )
    return output_dir, relative_name


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.terrain_showcase:
        if not 0.0 <= args_cli.showcase_difficulty <= 1.0:
            raise ValueError("--showcase_difficulty must be between 0 and 1.")
        if args_cli.showcase_steps_per_terrain <= 0:
            raise ValueError("--showcase_steps_per_terrain must be positive.")
        showcase_generator_cfg = env_cfg.scene.terrain.terrain_generator
        if env_cfg.scene.terrain.terrain_type != "generator" or showcase_generator_cfg is None:
            raise ValueError("--terrain_showcase requires a generated rough-terrain task.")
        showcase_column_allocation = terrain_showcase_column_allocation(
            showcase_generator_cfg.sub_terrains
        )
        showcase_num_columns = sum(showcase_column_allocation.values())
        # One environment per column guarantees that every configured family
        # has a representative without adding duplicate rendered robots.
        env_cfg.scene.num_envs = showcase_num_columns
        print(
            f"[INFO] Terrain showcase: {showcase_num_columns} columns/environments; allocation "
            f"{showcase_column_allocation}."
        )
    else:
        showcase_column_allocation = None
        showcase_num_columns = None
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = 0 if args_cli.terrain_showcase else None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        if args_cli.terrain_showcase:
            env_cfg.scene.terrain.terrain_generator.num_rows = 1
            env_cfg.scene.terrain.terrain_generator.num_cols = showcase_num_columns
            env_cfg.scene.terrain.terrain_generator.curriculum = True
            env_cfg.scene.terrain.terrain_generator.difficulty_range = (
                args_cli.showcase_difficulty,
                args_cli.showcase_difficulty,
            )
        else:
            env_cfg.scene.terrain.terrain_generator.num_rows = 5
            env_cfg.scene.terrain.terrain_generator.num_cols = 5
            env_cfg.scene.terrain.terrain_generator.curriculum = False
        if env_cfg.scene.terrain.terrain_generator.seed is None:
            env_cfg.scene.terrain.terrain_generator.seed = env_cfg.seed

    if args_cli.terrain_showcase:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (3.5, 3.5, 2.4)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.3)
        # A terrain-performance clip should exercise locomotion. The ordinary
        # task keeps its 2% standing commands for hold-policy evaluation.
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.randomize_push_robot = None
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.terrain_showcase:
        terrain_camera_representatives = terrain_family_representatives(env)
        env = TerrainFamilyCameraCycle(
            env, args_cli.showcase_steps_per_terrain, terrain_camera_representatives
        )
        showcase_steps = args_cli.showcase_steps_per_terrain * len(terrain_camera_representatives)
        showcase_name_prefix = "terrain-showcase"
        showcase_metadata = terrain_video_manifest(
            terrain_camera_representatives, args_cli.showcase_steps_per_terrain, showcase_name_prefix
        )
        showcase_play_dir = os.path.join(log_dir, "videos", "play")
        showcase_video_dir, showcase_relative_name = resolve_showcase_output_dir(
            showcase_play_dir,
            args_cli.showcase_output_name,
            args_cli.showcase_difficulty,
            agent_cfg.seed,
        )
        policy_step_dt = float(env.unwrapped.step_dt)
        showcase_metadata.update(
            {
                "schema_version": 2,
                "output_name": showcase_relative_name,
                "difficulty": args_cli.showcase_difficulty,
                "seed": agent_cfg.seed,
                "standing_command_probability": env_cfg.commands.base_velocity.rel_standing_envs,
                "policy_step_dt_s": policy_step_dt,
                "nominal_segment_duration_s": args_cli.showcase_steps_per_terrain * policy_step_dt,
                "terrain_family_count": len(terrain_camera_representatives),
                "terrain_column_count": showcase_num_columns,
                "terrain_column_allocation": showcase_column_allocation,
            }
        )
    else:
        terrain_camera_representatives = None
        showcase_steps = None
        showcase_name_prefix = None
        showcase_video_dir = None
        showcase_metadata = None

    # wrap for video recording
    if args_cli.video:
        if args_cli.terrain_showcase:
            video_kwargs = {
                "video_folder": showcase_video_dir,
                "step_trigger": lambda step: step % args_cli.showcase_steps_per_terrain == 0,
                "video_length": args_cli.showcase_steps_per_terrain,
                "name_prefix": showcase_name_prefix,
                "disable_logger": True,
            }
        else:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        if args_cli.terrain_showcase:
            dump_yaml(os.path.join(showcase_video_dir, "terrain_showcase.yaml"), showcase_metadata)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "HybridOnPolicyRunner":
        runner = HybridOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    device = env.unwrapped.device
    total_reward = torch.zeros((), device=device)
    total_dones = torch.zeros((), device=device)
    total_linear_velocity_error = torch.zeros((), device=device)
    total_yaw_velocity_error = torch.zeros((), device=device)
    total_action_abs = torch.zeros((), device=device)
    total_action_sq = torch.zeros((), device=device)
    maximum_action_abs = torch.zeros((), device=device)
    nonfinite_action_count = torch.zeros((), device=device)
    termination_counts = {
        name: torch.zeros((), device=device) for name in env.unwrapped.termination_manager.active_terms
    }
    evaluation_started = time.time()
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, extras = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
            total_reward += rewards.sum()
            total_dones += dones.sum()
            total_action_abs += actions.abs().sum()
            total_action_sq += actions.square().sum()
            maximum_action_abs = torch.maximum(maximum_action_abs, actions.abs().max())
            nonfinite_action_count += (~torch.isfinite(actions)).sum()
            command = env.unwrapped.command_manager.get_command("base_velocity")
            robot = env.unwrapped.scene["robot"]
            total_linear_velocity_error += torch.linalg.vector_norm(
                command[:, :2] - robot.data.root_lin_vel_b[:, :2], dim=1
            ).sum()
            total_yaw_velocity_error += (
                command[:, 2] - robot.data.root_ang_vel_b[:, 2]
            ).abs().sum()
            for name in termination_counts:
                termination_counts[name] += env.unwrapped.termination_manager.get_term(name).sum()
        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            target_steps = showcase_steps if showcase_steps is not None else args_cli.video_length
            if timestep == target_steps:
                break
        elif timestep == args_cli.eval_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    elapsed = time.time() - evaluation_started
    num_envs = int(env.unwrapped.num_envs)
    sample_count = num_envs * timestep
    action_count = sample_count * int(actions.shape[-1])
    checkpoint_hash = hashlib.sha256()
    with open(resume_path, "rb") as checkpoint_stream:
        for chunk in iter(lambda: checkpoint_stream.read(1024 * 1024), b""):
            checkpoint_hash.update(chunk)
    term_counts = {name: int(value.item()) for name, value in termination_counts.items()}
    timeout_count = term_counts.get("time_out", 0)
    failure_count = sum(value for name, value in term_counts.items() if name != "time_out")
    result = {
        "schema_version": 1,
        "status": "passed" if int(nonfinite_action_count.item()) == 0 else "failed",
        "task": args_cli.task,
        "agent": args_cli.agent,
        "checkpoint": os.path.abspath(resume_path),
        "checkpoint_sha256": checkpoint_hash.hexdigest(),
        "seed": int(agent_cfg.seed),
        "num_envs": num_envs,
        "policy_steps": timestep,
        "environment_steps": sample_count,
        "simulated_seconds_per_environment": timestep * float(env.unwrapped.step_dt),
        "wall_time_seconds": elapsed,
        "environment_steps_per_second": sample_count / elapsed,
        "mean_reward_per_environment_step": float((total_reward / sample_count).item()),
        "mean_linear_velocity_error_mps": float((total_linear_velocity_error / sample_count).item()),
        "mean_yaw_velocity_error_radps": float((total_yaw_velocity_error / sample_count).item()),
        "mean_action_abs": float((total_action_abs / action_count).item()),
        "action_rms": float(torch.sqrt(total_action_sq / action_count).item()),
        "maximum_action_abs": float(maximum_action_abs.item()),
        "nonfinite_action_count": int(nonfinite_action_count.item()),
        "episode_terminations": int(total_dones.item()),
        "timeout_terminations": timeout_count,
        "failure_terminations": failure_count,
        "termination_counts": term_counts,
        "evaluation_settings": {
            "observation_corruption": False,
            "external_pushes": False,
            "terrain_curriculum": False,
            "terrain_grid_rows": 5,
            "terrain_grid_columns": 5,
        },
        "exports": {
            "jit": os.path.join(export_model_dir, "policy.pt"),
            "onnx": os.path.join(export_model_dir, "policy.onnx"),
        },
    }
    os.makedirs(os.path.dirname(args_cli.eval_output), exist_ok=True)
    with open(args_cli.eval_output, "w", encoding="utf-8") as output_stream:
        json.dump(result, output_stream, indent=2)
        output_stream.write("\n")
    print(f"[EVAL] Wrote quantitative evaluation: {args_cli.eval_output}")
    print(json.dumps(result, indent=2))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
