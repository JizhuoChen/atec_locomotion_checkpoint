
"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--video_terrain_cycle",
    action="store_true",
    default=False,
    help="Rotate successive training videos through every generated terrain family.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--pretrained_actor",
    type=str,
    default=None,
    help="Path to an RSL-RL model_N.pt checkpoint whose actor weights initialize a new training run.",
)
parser.add_argument(
    "--pretrained_privileged_teacher",
    type=str,
    default=None,
    help=(
        "Path to the canonical 45-actor/251-critic checkpoint used to initialize an expanded "
        "privileged teacher. Shared input columns are copied and appended columns are zeroed."
    ),
)
parser.add_argument(
    "--teacher_checkpoint",
    type=str,
    default=None,
    help="Explicit privileged PPO teacher checkpoint to load for a DistillationRunner.",
)
parser.add_argument(
    "--pretrained_student",
    type=str,
    default=None,
    help="PPO checkpoint whose actor initializes the 45-input student before distillation.",
)
parser.add_argument(
    "--spawn_audit",
    action="store_true",
    default=False,
    help="Validate every robot root and foot against its assigned terrain tile before training.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import hashlib
import logging
import os
import time
from datetime import datetime

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
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import atec_rl_lab.train  # noqa: F401  # isort: skip
from terrain_camera import TerrainFamilyCameraCycle, terrain_family_representatives, terrain_video_manifest

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resume_metadata_and_sync_learning_rate(
    runner: OnPolicyRunner,
    checkpoint_path: str,
    additional_iterations: int,
) -> dict[str, object]:
    """Record full-state resume provenance and preserve the checkpoint's LR.

    RSL-RL restores the optimizer parameter group but leaves a separate
    adaptive-learning-rate scalar at its fresh config value. Synchronizing the
    scalar prevents the first PPO update from silently discarding the restored
    learning rate.
    """
    resolved_path = os.path.abspath(checkpoint_path)
    checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=True)
    source_iteration = int(checkpoint.get("iter", -1))
    if source_iteration < 0:
        raise RuntimeError(f"Resume checkpoint has no valid iteration: {resolved_path}")
    if int(runner.current_learning_iteration) != source_iteration:
        raise RuntimeError(
            "RSL-RL loaded an unexpected iteration: "
            f"runner={runner.current_learning_iteration}, checkpoint={source_iteration}."
        )
    optimizer = getattr(runner.alg, "optimizer", None)
    if optimizer is None or not optimizer.param_groups:
        raise RuntimeError("Resumed algorithm does not expose an optimizer parameter group.")
    group_lrs = {float(group["lr"]) for group in optimizer.param_groups}
    if len(group_lrs) != 1:
        raise RuntimeError(f"Cannot synchronize multiple optimizer learning rates: {group_lrs}")
    restored_lr = group_lrs.pop()
    checkpoint_optimizer = checkpoint.get("optimizer_state_dict")
    if not isinstance(checkpoint_optimizer, dict) or not checkpoint_optimizer.get("param_groups"):
        raise RuntimeError(f"Resume checkpoint has no optimizer parameter groups: {resolved_path}")
    checkpoint_lrs = {
        float(group["lr"]) for group in checkpoint_optimizer["param_groups"]
    }
    if checkpoint_lrs != {restored_lr}:
        raise RuntimeError(
            "Loaded optimizer learning rate differs from the checkpoint: "
            f"loaded={restored_lr}, checkpoint={checkpoint_lrs}."
        )
    if hasattr(runner.alg, "learning_rate"):
        runner.alg.learning_rate = restored_lr
    expected_final_iteration = source_iteration + int(additional_iterations) - 1
    print(
        f"[INFO]: Full-state resume source iteration: {source_iteration}; "
        f"restored learning rate: {restored_lr:.12g}; "
        f"expected final iteration: {expected_final_iteration}."
    )
    return {
        "schema_version": 1,
        "status": "running",
        "checkpoint": resolved_path,
        "checkpoint_sha256": _sha256_file(resolved_path),
        "source_iteration": source_iteration,
        "additional_iterations": int(additional_iterations),
        "expected_final_iteration": expected_final_iteration,
        "optimizer_learning_rate": restored_lr,
        "adaptive_learning_rate_synchronized": hasattr(runner.alg, "learning_rate"),
        "started_at": datetime.now().astimezone().isoformat(),
    }


def load_pretrained_actor(runner: OnPolicyRunner, checkpoint_path: str) -> dict[str, str | int]:
    """Initialize only the actor network from an RSL-RL training checkpoint.

    The critic, action-noise parameter, optimizer, and iteration counter intentionally remain freshly initialized.
    This supports transfer between tasks that share an actor interface but use different privileged critic inputs.
    """
    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Pretrained actor checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError(
            "Pretrained actor must be an RSL-RL training checkpoint containing 'model_state_dict' "
            "(for example, model_4999.pt), not an exported policy.pt file."
        )

    actor = getattr(runner.alg.policy, "actor", None)
    if actor is None:
        raise ValueError("The target RSL-RL policy does not expose an actor module.")

    source_state = checkpoint["model_state_dict"]
    source_actor_state = {
        name.removeprefix("actor."): value for name, value in source_state.items() if name.startswith("actor.")
    }
    target_actor_state = actor.state_dict()

    missing_keys = sorted(set(target_actor_state) - set(source_actor_state))
    unexpected_keys = sorted(set(source_actor_state) - set(target_actor_state))
    shape_mismatches = sorted(
        name
        for name in set(source_actor_state) & set(target_actor_state)
        if source_actor_state[name].shape != target_actor_state[name].shape
    )
    if missing_keys or unexpected_keys or shape_mismatches:
        raise ValueError(
            "Pretrained actor is incompatible with the target policy. "
            f"Missing keys: {missing_keys}; unexpected keys: {unexpected_keys}; "
            f"shape mismatches: {shape_mismatches}."
        )

    actor.load_state_dict(source_actor_state, strict=True)
    source_iteration = int(checkpoint.get("iter", -1))
    print(
        f"[INFO]: Initialized {len(source_actor_state)} actor tensors from: {checkpoint_path}\n"
        f"[INFO]: Source iteration: {source_iteration}. Critic, optimizer, action noise, and iteration counter are fresh."
    )
    return {"checkpoint": checkpoint_path, "source_iteration": source_iteration}


def _load_checkpoint_model_state(checkpoint_path: str) -> tuple[str, dict[str, torch.Tensor], int]:
    """Load and validate an RSL-RL training checkpoint's model state."""
    resolved_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Training checkpoint does not exist: {resolved_path}")
    checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError(
            "Expected an RSL-RL model_N.pt checkpoint containing 'model_state_dict', "
            f"not an exported policy: {resolved_path}"
        )
    return resolved_path, checkpoint["model_state_dict"], int(checkpoint.get("iter", -1))


def _expanded_mlp_state(
    target: torch.nn.Module,
    source_state: dict[str, torch.Tensor],
    *,
    network_name: str,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Copy an MLP while zero-initializing appended first-layer input columns."""
    target_state = target.state_dict()
    if set(target_state) != set(source_state):
        raise ValueError(
            f"{network_name} layer keys differ. Source={sorted(source_state)}, "
            f"target={sorted(target_state)}."
        )

    result: dict[str, torch.Tensor] = {}
    expanded_key: str | None = None
    source_input_dim: int | None = None
    target_input_dim: int | None = None
    for name, target_value in target_state.items():
        source_value = source_state[name]
        if source_value.shape == target_value.shape:
            result[name] = source_value
            continue
        if (
            name == "0.weight"
            and source_value.ndim == 2
            and target_value.ndim == 2
            and source_value.shape[0] == target_value.shape[0]
            and source_value.shape[1] < target_value.shape[1]
        ):
            expanded = torch.zeros(target_value.shape, dtype=source_value.dtype, device=source_value.device)
            expanded[:, : source_value.shape[1]] = source_value
            result[name] = expanded
            expanded_key = name
            source_input_dim = int(source_value.shape[1])
            target_input_dim = int(target_value.shape[1])
            continue
        raise ValueError(
            f"Cannot expand {network_name} tensor {name}: source={tuple(source_value.shape)}, "
            f"target={tuple(target_value.shape)}. Only appended columns in 0.weight are supported."
        )

    if expanded_key is None or source_input_dim is None or target_input_dim is None:
        raise ValueError(f"{network_name} did not require the expected first-layer input expansion.")
    return result, {
        "source_input_dim": source_input_dim,
        "target_input_dim": target_input_dim,
        "zero_initialized_input_columns": target_input_dim - source_input_dim,
    }


def load_expanded_privileged_teacher(
    runner: OnPolicyRunner, checkpoint_path: str
) -> dict[str, object]:
    """Warm-start a privileged teacher from the canonical checkpoint."""
    expected_obs_groups = {
        "critic": ["critic", "contact_forces"],
        "policy": ["policy", "teacher_privileged", "contact_forces", "teacher_height_scan"],
    }
    resolved_obs_groups = getattr(runner, "cfg", {}).get("obs_groups")
    policy_groups = tuple(resolved_obs_groups.get("policy", ())) if resolved_obs_groups else None
    critic_groups = tuple(resolved_obs_groups.get("critic", ())) if resolved_obs_groups else None
    legacy_policy_obs_groups = ("policy", "teacher_privileged", "contact_forces")
    height_scan_policy_obs_groups = ("policy", "teacher_privileged", "contact_forces", "teacher_height_scan")
    if (
        resolved_obs_groups is None
        or (policy_groups != legacy_policy_obs_groups and policy_groups != height_scan_policy_obs_groups)
        or critic_groups != tuple(expected_obs_groups["critic"])
    ):
        raise ValueError(
            "Expanded teacher initialization requires the prefix-preserving observation mapping "
            f"{expected_obs_groups} or {legacy_policy_obs_groups}, got {resolved_obs_groups}."
        )

    resolved_path, source_state, source_iteration = _load_checkpoint_model_state(checkpoint_path)
    policy = runner.alg.policy
    actor = getattr(policy, "actor", None)
    critic = getattr(policy, "critic", None)
    if actor is None or critic is None:
        raise ValueError("Expanded teacher initialization requires ActorCritic actor and critic modules.")

    source_actor = {
        name.removeprefix("actor."): value
        for name, value in source_state.items()
        if name.startswith("actor.")
    }
    source_critic = {
        name.removeprefix("critic."): value
        for name, value in source_state.items()
        if name.startswith("critic.")
    }
    actor_state, actor_metadata = _expanded_mlp_state(
        actor, source_actor, network_name="privileged teacher actor"
    )
    critic_state, critic_metadata = _expanded_mlp_state(
        critic, source_critic, network_name="privileged teacher critic"
    )
    if actor_metadata["source_input_dim"] != 45:
        raise ValueError(
            "Unexpected privileged teacher actor expansion: source input dimension "
            f"must stay 45, got {actor_metadata['source_input_dim']}."
        )
    if actor_metadata["target_input_dim"] <= actor_metadata["source_input_dim"]:
        raise ValueError(f"Unexpected privileged teacher actor expansion: {actor_metadata}")
    if (
        actor_metadata["target_input_dim"] != 76
        and "teacher_height_scan" not in policy_groups
    ):
        raise ValueError(
            "Unexpected privileged teacher actor expansion: target input dimension "
            f"{actor_metadata['target_input_dim']} requires the height-scan policy group "
            "(teacher_height_scan)."
        )
    if critic_metadata != {
        "source_input_dim": 251,
        "target_input_dim": 263,
        "zero_initialized_input_columns": 12,
    }:
        raise ValueError(f"Unexpected privileged teacher critic expansion: {critic_metadata}")

    actor.load_state_dict(actor_state, strict=True)
    critic.load_state_dict(critic_state, strict=True)

    std_copied = False
    if hasattr(policy, "std") and "std" in source_state:
        if policy.std.shape != source_state["std"].shape:
            raise ValueError(
                f"Action standard deviation shape mismatch: source={tuple(source_state['std'].shape)}, "
                f"target={tuple(policy.std.shape)}."
            )
        with torch.no_grad():
            policy.std.copy_(source_state["std"].to(device=policy.std.device, dtype=policy.std.dtype))
        std_copied = True

    metadata = {
        "checkpoint": resolved_path,
        "checkpoint_sha256": _sha256_file(resolved_path),
        "source_iteration": source_iteration,
        "actor": actor_metadata,
        "critic": critic_metadata,
        "action_std_copied": std_copied,
        "optimizer": "fresh",
        "iteration_counter": "fresh",
    }
    print(
        "[INFO]: Initialized privileged teacher from canonical PPO checkpoint. "
        f"Actor {actor_metadata['source_input_dim']}->{actor_metadata['target_input_dim']} "
        f"and critic {critic_metadata['source_input_dim']}->{critic_metadata['target_input_dim']}; "
        "all appended input columns are exactly zero."
    )
    return metadata


def load_pretrained_student(runner: DistillationRunner, checkpoint_path: str) -> dict[str, object]:
    """Initialize the distillation student's unchanged MLP from a PPO actor."""
    resolved_path, source_state, source_iteration = _load_checkpoint_model_state(checkpoint_path)
    student = getattr(runner.alg.policy, "student", None)
    if student is None:
        raise ValueError("Pretrained student initialization requires a StudentTeacher policy.")
    source_actor = {
        name.removeprefix("actor."): value
        for name, value in source_state.items()
        if name.startswith("actor.")
    }
    target_state = student.state_dict()
    missing = sorted(set(target_state) - set(source_actor))
    unexpected = sorted(set(source_actor) - set(target_state))
    mismatched = sorted(
        name
        for name in set(target_state) & set(source_actor)
        if target_state[name].shape != source_actor[name].shape
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "PPO actor is incompatible with the distillation student. "
            f"Missing={missing}; unexpected={unexpected}; shape_mismatches={mismatched}."
        )
    student.load_state_dict(source_actor, strict=True)
    print(
        f"[INFO]: Initialized the 45-input distillation student from: {resolved_path} "
        f"(iteration {source_iteration})."
    )
    return {
        "checkpoint": resolved_path,
        "checkpoint_sha256": _sha256_file(resolved_path),
        "source_iteration": source_iteration,
        "student_input_dim": 45,
        "student_output_dim": 12,
        "optimizer": "fresh",
    }


@torch.inference_mode()
def audit_terrain_spawns(env: RslRlVecEnvWrapper) -> dict[str, float | int | bool | None]:
    """Fail fast if a reset root or foot lies outside its assigned terrain tile."""
    base_env = env.unwrapped
    terrain = base_env.scene.terrain
    robot = base_env.scene["robot"]
    if terrain.terrain_origins is None or terrain.cfg.terrain_generator is None:
        raise ValueError("--spawn_audit requires a generated terrain with curriculum origins.")

    # Forward one physics frame so body-link poses reflect the reset pose written by
    # the event manager. This does not advance the RL episode counter.
    base_env.scene.write_data_to_sim()
    base_env.sim.step(render=False)
    base_env.scene.update(base_env.physics_dt)

    levels = terrain.terrain_levels
    types = terrain.terrain_types
    tile_origins = terrain.terrain_origins[levels, types]
    tile_half_extent = 0.5 * torch.tensor(
        terrain.cfg.terrain_generator.size[:2], device=base_env.device
    )
    root_local_xy = robot.data.root_pos_w[:, :2] - tile_origins[:, :2]

    foot_ids, _ = robot.find_bodies(".*_foot")
    if len(foot_ids) == 0:
        raise RuntimeError("--spawn_audit could not find robot bodies matching '.*_foot'.")
    foot_local_xy = robot.data.body_pos_w[:, foot_ids, :2] - tile_origins[:, None, :2]

    root_outside = torch.any(torch.abs(root_local_xy) > tile_half_extent, dim=1)
    foot_outside = torch.any(torch.abs(foot_local_xy) > tile_half_extent, dim=(1, 2))

    num_cols = terrain.terrain_origins.shape[1]
    cell_ids = levels * num_cols + types
    cell_counts = torch.bincount(cell_ids, minlength=terrain.terrain_origins.shape[0] * num_cols)
    pairwise_distances = []
    for cell_id in torch.nonzero(cell_counts > 1, as_tuple=False).flatten():
        pairwise_distances.append(torch.pdist(root_local_xy[cell_ids == cell_id]))
    min_same_cell_separation = (
        torch.cat(pairwise_distances).min().item() if pairwise_distances else float("inf")
    )
    duplicate_roots = min_same_cell_separation < 1.0e-6

    result = {
        "num_envs": int(base_env.num_envs),
        "num_terrain_cells_used": int(torch.count_nonzero(cell_counts).item()),
        "max_envs_per_cell": int(cell_counts.max().item()),
        "initial_min_terrain_level": int(levels.min().item()),
        "initial_max_terrain_level": int(levels.max().item()),
        "min_same_cell_root_separation_m": float(min_same_cell_separation),
        "max_abs_root_offset_x_m": float(torch.abs(root_local_xy[:, 0]).max().item()),
        "max_abs_root_offset_y_m": float(torch.abs(root_local_xy[:, 1]).max().item()),
        "max_abs_foot_offset_x_m": float(torch.abs(foot_local_xy[..., 0]).max().item()),
        "max_abs_foot_offset_y_m": float(torch.abs(foot_local_xy[..., 1]).max().item()),
        "root_outside_tile_count": int(root_outside.sum().item()),
        "robot_with_foot_outside_tile_count": int(foot_outside.sum().item()),
        "duplicate_root_within_cell": bool(duplicate_roots),
        "cross_environment_collision_filtering": bool(base_env.cfg.scene.filter_collisions),
    }
    proxy_mask = getattr(base_env, "_b2_only_proxy_mask", None)
    if isinstance(proxy_mask, torch.Tensor):
        proxy_counts = getattr(base_env, "_b2_only_proxy_counts_by_terrain_type", {})
        proxy_scales = getattr(base_env, "_b2_only_proxy_mass_scales", None)
        finite_proxy_scales = (
            proxy_scales[torch.isfinite(proxy_scales)]
            if isinstance(proxy_scales, torch.Tensor)
            else None
        )
        result.update(
            {
                "b2_only_proxy_count": int(torch.count_nonzero(proxy_mask).item()),
                "b2_only_proxy_fraction": float(torch.mean(proxy_mask.float()).item()),
                "b2_only_proxy_terrain_type_count": len(proxy_counts),
                "b2_only_proxy_min_per_terrain_type": (
                    min(proxy_counts.values()) if proxy_counts else 0
                ),
                "b2_only_proxy_max_per_terrain_type": (
                    max(proxy_counts.values()) if proxy_counts else 0
                ),
                "b2_only_proxy_min_mass_scale": (
                    float(finite_proxy_scales.min().item())
                    if finite_proxy_scales is not None and len(finite_proxy_scales) > 0
                    else None
                ),
                "b2_only_proxy_max_mass_scale": (
                    float(finite_proxy_scales.max().item())
                    if finite_proxy_scales is not None and len(finite_proxy_scales) > 0
                    else None
                ),
            }
        )
    print("[INFO] Terrain spawn audit:")
    print_dict(result, nesting=4)

    if result["root_outside_tile_count"] or result["robot_with_foot_outside_tile_count"] or duplicate_roots:
        bad = torch.nonzero(root_outside | foot_outside, as_tuple=False).flatten()[:10].tolist()
        raise RuntimeError(f"Terrain spawn audit failed; first out-of-bounds environment IDs: {bad}")
    return result


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    if agent_cfg.max_iterations <= 0:
        raise ValueError("--max_iterations must be positive.")
    if args_cli.pretrained_actor is not None and agent_cfg.resume:
        raise ValueError("--pretrained_actor and --resume are mutually exclusive.")
    if args_cli.pretrained_actor is not None and agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError("--pretrained_actor currently supports only the RSL-RL OnPolicyRunner.")
    if args_cli.pretrained_privileged_teacher is not None and agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError("--pretrained_privileged_teacher requires the RSL-RL OnPolicyRunner.")
    if args_cli.pretrained_privileged_teacher is not None and (
        agent_cfg.resume or args_cli.pretrained_actor is not None
    ):
        raise ValueError(
            "--pretrained_privileged_teacher is mutually exclusive with --resume and --pretrained_actor."
        )
    is_distillation = agent_cfg.class_name == "DistillationRunner"
    if args_cli.teacher_checkpoint is not None and not is_distillation:
        raise ValueError("--teacher_checkpoint is only valid for a DistillationRunner.")
    if args_cli.teacher_checkpoint is not None and args_cli.checkpoint is not None:
        raise ValueError("--teacher_checkpoint and --checkpoint are mutually exclusive.")
    if args_cli.pretrained_student is not None and not is_distillation:
        raise ValueError("--pretrained_student is only valid for a DistillationRunner.")
    if args_cli.pretrained_student is not None and agent_cfg.resume:
        raise ValueError("--pretrained_student cannot be combined with --resume.")
    if is_distillation and args_cli.teacher_checkpoint is None and args_cli.checkpoint is None:
        logger.warning(
            "No explicit --teacher_checkpoint was supplied; the teacher will be selected from "
            "the distillation experiment's load_run/load_checkpoint patterns."
        )
    if args_cli.video_terrain_cycle and not args_cli.video:
        raise ValueError("--video_terrain_cycle requires --video.")

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.video_terrain_cycle:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (3.5, 3.5, 2.4)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.3)
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # Persist the procedural terrain seed explicitly. This makes the generated
    # riser/tread combinations reproducible from params/env.yaml instead of
    # relying on an implicit snapshot of NumPy's global RNG state.
    terrain_generator = getattr(env_cfg.scene.terrain, "terrain_generator", None)
    if terrain_generator is not None and terrain_generator.seed is None:
        terrain_generator.seed = env_cfg.seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video_terrain_cycle:
        terrain_camera_representatives = terrain_family_representatives(env)
        env = TerrainFamilyCameraCycle(env, args_cli.video_interval, terrain_camera_representatives)
        terrain_camera_metadata = terrain_video_manifest(
            terrain_camera_representatives, args_cli.video_interval, "terrain-train"
        )
    else:
        terrain_camera_metadata = None

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    elif is_distillation:
        if args_cli.teacher_checkpoint is not None:
            resume_path = os.path.abspath(os.path.expanduser(args_cli.teacher_checkpoint))
            if not os.path.isfile(resume_path):
                raise FileNotFoundError(f"Teacher checkpoint does not exist: {resume_path}")
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "name_prefix": "terrain-train" if args_cli.video_terrain_cycle else "rl-video",
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    spawn_audit_metadata = audit_terrain_spawns(env) if args_cli.spawn_audit else None

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    privileged_teacher_metadata = None
    pretrained_student_metadata = None
    if agent_cfg.resume or is_distillation:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
        pretrained_actor_metadata = None
        resume_metadata = (
            resume_metadata_and_sync_learning_rate(
                runner, resume_path, agent_cfg.max_iterations
            )
            if agent_cfg.resume and agent_cfg.class_name == "OnPolicyRunner"
            else None
        )
        if is_distillation and args_cli.pretrained_student is not None:
            pretrained_student_metadata = load_pretrained_student(runner, args_cli.pretrained_student)
    elif args_cli.pretrained_privileged_teacher is not None:
        privileged_teacher_metadata = load_expanded_privileged_teacher(
            runner, args_cli.pretrained_privileged_teacher
        )
        pretrained_actor_metadata = None
        resume_metadata = None
    elif args_cli.pretrained_actor is not None:
        pretrained_actor_metadata = load_pretrained_actor(runner, args_cli.pretrained_actor)
        resume_metadata = None
    else:
        pretrained_actor_metadata = None
        resume_metadata = None

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    if pretrained_actor_metadata is not None:
        dump_yaml(os.path.join(log_dir, "params", "pretrained_actor.yaml"), pretrained_actor_metadata)
    if privileged_teacher_metadata is not None:
        dump_yaml(
            os.path.join(log_dir, "params", "privileged_teacher_initialization.yaml"),
            privileged_teacher_metadata,
        )
    if pretrained_student_metadata is not None:
        dump_yaml(
            os.path.join(log_dir, "params", "pretrained_student.yaml"),
            pretrained_student_metadata,
        )
    if resume_metadata is not None:
        dump_yaml(os.path.join(log_dir, "params", "resume.yaml"), resume_metadata)
    if spawn_audit_metadata is not None:
        dump_yaml(os.path.join(log_dir, "params", "spawn_audit.yaml"), spawn_audit_metadata)
    if terrain_camera_metadata is not None:
        dump_yaml(os.path.join(log_dir, "params", "terrain_camera.yaml"), terrain_camera_metadata)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    if resume_metadata is not None:
        final_iteration = int(runner.current_learning_iteration)
        if final_iteration != int(resume_metadata["expected_final_iteration"]):
            raise RuntimeError(
                f"Resume finished at iteration {final_iteration}, expected "
                f"{resume_metadata['expected_final_iteration']}."
            )
        resume_metadata.update(
            {
                "status": "completed",
                "final_iteration": final_iteration,
                "final_checkpoint": os.path.join(log_dir, f"model_{final_iteration}.pt"),
                "completed_at": datetime.now().astimezone().isoformat(),
            }
        )
        dump_yaml(os.path.join(log_dir, "params", "resume.yaml"), resume_metadata)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
