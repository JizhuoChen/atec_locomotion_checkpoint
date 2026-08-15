#!/usr/bin/env python3
"""Run B2-Piper flat adaptation and then heading-first rough fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


FLAT_TASK = "ATEC-Isaac-Velocity-Flat-Unitree-B2-Piper-v0"
ROUGH_TASK = "ATEC-Isaac-Velocity-Heading-Rough-Unitree-B2-Piper-v0"
FLAT_EXPERIMENT = "unitree_b2_piper_flat"
ROUGH_EXPERIMENT = "unitree_b2_piper_heading_rough"
ROBUST_FLAT_TASK = "ATEC-Isaac-Velocity-Robust-Flat-Unitree-B2-Piper-v0"
ROBUST_ROUGH_TASK = "ATEC-Isaac-Velocity-Robust-Heading-Rough-Unitree-B2-Piper-v0"
ROBUST_FLAT_EXPERIMENT = "unitree_b2_piper_robust_flat"
ROBUST_ROUGH_EXPERIMENT = "unitree_b2_piper_robust_heading_rough"
BARE_B2_FLAT_EXPERIMENT = "unitree_b2_flat"
PROFILE_SETTINGS: dict[str, dict[str, Any]] = {
    "baseline": {
        "flat_task": FLAT_TASK,
        "rough_task": ROUGH_TASK,
        "flat_experiment": FLAT_EXPERIMENT,
        "rough_experiment": ROUGH_EXPERIMENT,
        "flat_iterations": 5000,
        "rough_iterations": 5000,
        "flat_seed": "auto-flat",
        "run_prefix": "b2_piper_auto",
        "arm_motion_profile": "small_sinusoidal",
    },
    "robust": {
        "flat_task": ROBUST_FLAT_TASK,
        "rough_task": ROBUST_ROUGH_TASK,
        "flat_experiment": ROBUST_FLAT_EXPERIMENT,
        "rough_experiment": ROBUST_ROUGH_EXPERIMENT,
        "flat_iterations": 8000,
        "rough_iterations": 12000,
        "flat_seed": "none",
        "run_prefix": "b2_piper_robust",
        "arm_motion_profile": "smooth_random_waypoints",
    },
}
CHECKPOINT_RE = re.compile(r"model_(\d+)\.pt$")
RUN_PREFIX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
ACTOR_SHAPES = {
    "actor.0.weight": (512, 45),
    "actor.0.bias": (512,),
    "actor.2.weight": (256, 512),
    "actor.2.bias": (256,),
    "actor.4.weight": (128, 256),
    "actor.4.bias": (128,),
    "actor.6.weight": (12, 128),
    "actor.6.bias": (12,),
}


class PipelineInterrupted(RuntimeError):
    """Raised when the launcher receives a termination signal between child runs."""

    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"Pipeline received {signal.Signals(signum).name}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the leg-only B2-Piper actor on flat ground, then automatically "
            "transfer its actor into the heading-first rough-terrain task."
        )
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_SETTINGS),
        default="baseline",
        help=(
            "Training profile. 'baseline' preserves the original 5k/5k warm-start workflow; "
            "'robust' defaults to an 8k from-scratch flat stage and 12k robust-terrain stage."
        ),
    )
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument(
        "--flat-iterations",
        type=int,
        default=None,
        help="Override the selected profile's flat-stage iteration count.",
    )
    parser.add_argument(
        "--rough-iterations",
        type=int,
        default=None,
        help="Override the selected profile's rough-stage iteration count.",
    )
    parser.add_argument(
        "--flat-seed",
        default=None,
        help=(
            "Actor seed for stage 1: 'auto-flat' selects the highest-iteration bare-B2 flat "
            "checkpoint, 'none' starts from scratch, or provide an explicit model_N.pt path. "
            "Defaults to 'auto-flat' for baseline and 'none' for robust."
        ),
    )
    parser.add_argument(
        "--skip-flat-checkpoint",
        type=Path,
        default=None,
        help="Skip stage 1 and launch stage 2 from this completed B2-Piper flat checkpoint.",
    )
    parser.add_argument(
        "--run-prefix",
        default=None,
        help="Pipeline/run prefix; defaults to a profile-specific value.",
    )
    parser.add_argument("--flat-video-interval", type=int, default=10000)
    parser.add_argument("--rough-video-interval", type=int, default=4000)
    parser.add_argument("--video-length", type=int, default=300)
    parser.add_argument(
        "--video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Record videos inside the training process. Disabled by default because rendering "
            "many non-instanceable Piper meshes can exhaust host RAM; use play.py afterwards."
        ),
    )
    parser.add_argument(
        "--allow-high-env-rendering",
        action="store_true",
        help=(
            "Acknowledge the host-memory risk and allow video/live rendering with more than 64 environments."
        ),
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spawn-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--python", default=sys.executable, help="Python executable used for both stages.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profile_defaults = PROFILE_SETTINGS[args.profile]
    if args.flat_iterations is None:
        args.flat_iterations = profile_defaults["flat_iterations"]
    if args.rough_iterations is None:
        args.rough_iterations = profile_defaults["rough_iterations"]
    if args.flat_seed is None:
        args.flat_seed = profile_defaults["flat_seed"]
    if args.run_prefix is None:
        args.run_prefix = profile_defaults["run_prefix"]

    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be non-negative; omit it to use the runner's deterministic default")
    if args.flat_iterations <= 0 or args.rough_iterations <= 0:
        parser.error("both iteration counts must be positive")
    if args.video_length <= 0 or args.flat_video_interval <= 0 or args.rough_video_interval <= 0:
        parser.error("video length and intervals must be positive")
    if (args.video or not args.headless) and args.num_envs > 64 and not args.allow_high_env_rendering:
        parser.error(
            "B2-Piper rendering with more than 64 environments is blocked because RTX ingestion "
            "of the cloned arm meshes can exhaust host RAM. Train headless without --video, then "
            "use scripts/rsl_rl/play.py with a checkpoint. Pass --allow-high-env-rendering only "
            "if the machine has been independently validated."
        )
    if args.skip_flat_checkpoint is not None and args.flat_seed != profile_defaults["flat_seed"]:
        parser.error("--flat-seed is unused and must be omitted when --skip-flat-checkpoint is set")
    if RUN_PREFIX_RE.fullmatch(args.run_prefix) is None:
        parser.error(
            "--run-prefix must be 1-64 characters and contain only letters, numbers, '.', '_', or '-'"
        )
    return args


def checkpoint_iteration(path: Path) -> int:
    match = CHECKPOINT_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Checkpoint must be named model_<iteration>.pt: {path}")
    return int(match.group(1))


def highest_completed_checkpoint(experiment_dir: Path) -> Path | None:
    """Select only exact final checkpoints from completed-looking training runs.

    A periodic checkpoint from a currently running job is deliberately ignored:
    the candidate filename must equal ``model_{max_iterations - 1}.pt`` from
    that run's saved agent configuration.
    """
    candidates: list[tuple[int, float, str, Path]] = []
    if not experiment_dir.is_dir():
        return None
    for run_dir in experiment_dir.iterdir():
        if not run_dir.is_dir():
            continue
        agent_params_path = run_dir / "params/agent.yaml"
        try:
            agent_params = agent_params_path.read_text(encoding="utf-8")
        except OSError:
            continue
        match = re.search(r"^max_iterations:\s*(\d+)\s*$", agent_params, flags=re.MULTILINE)
        if match is None:
            continue
        iteration = int(match.group(1)) - 1
        if iteration < 0:
            continue
        path = run_dir / f"model_{iteration}.pt"
        if not path.is_file():
            continue
        resolved_path = path.resolve()
        candidates.append((iteration, path.stat().st_mtime, str(resolved_path), resolved_path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1], item[2]))[3]


def validate_actor_checkpoint(path: Path, expected_iteration: int | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    filename_iteration = checkpoint_iteration(path)
    if expected_iteration is not None and filename_iteration != expected_iteration:
        raise RuntimeError(
            f"Expected model_{expected_iteration}.pt after a clean stage, found {path.name}."
        )

    initial_stat = path.stat()

    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise RuntimeError(f"Not an RSL-RL training checkpoint: {path}")
    if not isinstance(payload.get("optimizer_state_dict"), dict):
        raise RuntimeError(f"Checkpoint has no optimizer state and appears incomplete: {path}")
    saved_iteration = int(payload.get("iter", -1))
    if saved_iteration != filename_iteration:
        raise RuntimeError(
            f"Checkpoint iteration mismatch: filename={filename_iteration}, payload={saved_iteration}."
        )
    state = payload["model_state_dict"]
    actor_keys = {key for key in state if key.startswith("actor.")}
    if actor_keys != set(ACTOR_SHAPES):
        raise RuntimeError(
            f"Checkpoint actor keys do not match the versioned 45-to-12 contract: {path}"
        )
    mismatched_shapes: dict[str, tuple[Any, tuple[int, ...]]] = {}
    for key, expected_shape in ACTOR_SHAPES.items():
        value = state[key]
        if not isinstance(value, torch.Tensor):
            mismatched_shapes[key] = (type(value).__name__, expected_shape)
        elif tuple(value.shape) != expected_shape:
            mismatched_shapes[key] = (tuple(value.shape), expected_shape)
    if mismatched_shapes:
        raise RuntimeError(f"Checkpoint actor shape mismatch {mismatched_shapes}: {path}")
    actor_tensors = [state[key] for key in ACTOR_SHAPES]
    if any(not torch.isfinite(value).all() for value in actor_tensors):
        raise RuntimeError(f"Checkpoint contains missing or non-finite actor tensors: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    final_stat = path.stat()
    if (initial_stat.st_size, initial_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
        raise RuntimeError(f"Checkpoint changed while it was being validated (possibly still being written): {path}")
    return {
        "path": str(path.resolve()),
        "iteration": saved_iteration,
        "actor_input_dim": 45,
        "actor_output_dim": 12,
        "sha256": digest.hexdigest(),
    }


def validate_run_experiment(path: Path, expected_experiment: str) -> None:
    """Require a checkpoint to retain provenance from the expected experiment."""
    agent_params_path = path.parent / "params/agent.yaml"
    try:
        agent_params = agent_params_path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(
            f"Cannot verify checkpoint provenance because {agent_params_path} is unavailable."
        ) from error
    expected_line = f"experiment_name: {expected_experiment}\n"
    if expected_line not in agent_params:
        raise RuntimeError(
            f"Checkpoint is not from the required '{expected_experiment}' experiment: {path}"
        )


def validate_completed_run_checkpoint(path: Path, expected_experiment: str) -> None:
    """Require the exact final checkpoint declared by a run's saved configuration."""
    validate_run_experiment(path, expected_experiment)
    agent_params = (path.parent / "params/agent.yaml").read_text(encoding="utf-8")
    match = re.search(r"^max_iterations:\s*(\d+)\s*$", agent_params, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"Cannot find max_iterations in the checkpoint's agent metadata: {path}")
    expected_iteration = int(match.group(1)) - 1
    if checkpoint_iteration(path) != expected_iteration:
        raise RuntimeError(
            f"--skip-flat-checkpoint requires the completed run's model_{expected_iteration}.pt, "
            f"not {path.name}."
        )


def atomic_dump_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def stage_command(
    args: argparse.Namespace,
    repo_root: Path,
    task: str,
    run_name: str,
    iterations: int,
    checkpoint: Path | None,
    rough: bool,
) -> list[str]:
    command = [
        str(Path(args.python).expanduser()),
        str(repo_root / "scripts/rsl_rl/train.py"),
        "--task",
        task,
        "--num_envs",
        str(args.num_envs),
        "--max_iterations",
        str(iterations),
        "--run_name",
        run_name,
    ]
    if checkpoint is not None:
        command.extend(["--pretrained_actor", str(checkpoint.resolve())])
    if args.device:
        command.extend(["--device", args.device])
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if args.video:
        command.extend(
            [
                "--video",
                "--video_interval",
                str(args.rough_video_interval if rough else args.flat_video_interval),
                "--video_length",
                str(args.video_length),
            ]
        )
        if rough:
            command.append("--video_terrain_cycle")
    if rough and args.spawn_audit:
        command.append("--spawn_audit")
    if args.headless:
        command.append("--headless")
    return command


def unique_stage_run(experiment_dir: Path, run_name: str) -> Path:
    matches = sorted(path for path in experiment_dir.glob(f"*_{run_name}") if path.is_dir())
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one run ending in '_{run_name}' under {experiment_dir}, found {matches}."
        )
    return matches[0].resolve()


def run_child(command: list[str], cwd: Path) -> tuple[int, int | None]:
    """Run one owned child process and forward terminal signals to its process group."""
    process: subprocess.Popen[Any] | None = None
    received_signal: int | None = None
    signal_received_at: float | None = None
    signal_count = 0
    sent_sigterm = False
    previous_handlers: dict[int, Any] = {}

    def forward_signal(signum: int, _frame: Any) -> None:
        nonlocal received_signal, signal_received_at, signal_count, sent_sigterm
        if received_signal is None:
            received_signal = signum
            signal_received_at = time.monotonic()
        signal_count += 1
        if process is not None and process.poll() is None:
            try:
                if signal_count == 1:
                    os.killpg(process.pid, signum)
                elif signal_count == 2:
                    os.killpg(process.pid, signal.SIGTERM)
                    sent_sigterm = True
                else:
                    os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    forwarded_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        forwarded_signals.append(signal.SIGHUP)
    for signum in forwarded_signals:
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, forward_signal)
    try:
        # Install forwarding before spawning. If a signal arrives while Popen
        # is creating the new session, it is remembered and sent immediately
        # after the process handle becomes available.
        process = subprocess.Popen(command, cwd=cwd, start_new_session=True)
        if received_signal is not None and process.poll() is None:
            try:
                if signal_count == 1:
                    os.killpg(process.pid, received_signal)
                elif signal_count == 2:
                    os.killpg(process.pid, signal.SIGTERM)
                    sent_sigterm = True
                else:
                    os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        while True:
            try:
                return process.wait(timeout=0.5), received_signal
            except subprocess.TimeoutExpired:
                if signal_received_at is None or process.poll() is not None:
                    continue
                elapsed = time.monotonic() - signal_received_at
                try:
                    if elapsed >= 60.0:
                        os.killpg(process.pid, signal.SIGKILL)
                    elif elapsed >= 30.0 and not sent_sigterm:
                        os.killpg(process.pid, signal.SIGTERM)
                        sent_sigterm = True
                except ProcessLookupError:
                    pass
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def run_stage(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    stage_name: str,
    task: str,
    experiment: str,
    iterations: int,
    actor_seed: Path | None,
    rough: bool,
) -> Path:
    run_name = f"{manifest['pipeline_id']}_{stage_name}"
    command = stage_command(args, repo_root, task, run_name, iterations, actor_seed, rough)
    manifest["stages"][stage_name] = {
        "status": "pending",
        "task": task,
        "experiment": experiment,
        "iterations": iterations,
        "actor_seed": str(actor_seed.resolve()) if actor_seed is not None else None,
        "command": command,
    }
    actor_seed_info = None
    atomic_dump_json(manifest_path, manifest)
    print(f"\n[PIPELINE] {stage_name} command:\n  " + " ".join(command), flush=True)
    if args.dry_run:
        return Path(f"<dry-run-{stage_name}-checkpoint>")

    try:
        if actor_seed is not None:
            actor_seed_info = validate_actor_checkpoint(actor_seed)
            manifest["stages"][stage_name]["validated_actor_seed"] = actor_seed_info
    except BaseException as error:
        interrupted = isinstance(error, (KeyboardInterrupt, PipelineInterrupted))
        manifest["stages"][stage_name].update(
            {
                "status": "interrupted" if interrupted else "failed_preflight",
                "error": f"{type(error).__name__}: {error}",
                "finished_at": datetime.now().astimezone().isoformat(),
            }
        )
        atomic_dump_json(manifest_path, manifest)
        raise
    manifest["stages"][stage_name].update(
        {"status": "running", "started_at": datetime.now().astimezone().isoformat()}
    )
    atomic_dump_json(manifest_path, manifest)

    try:
        return_code, received_signal = run_child(command, repo_root)
    except BaseException as error:
        manifest["stages"][stage_name].update(
            {
                "status": (
                    "interrupted"
                    if isinstance(error, (KeyboardInterrupt, PipelineInterrupted))
                    else "failed_start"
                ),
                "error": f"{type(error).__name__}: {error}",
                "finished_at": datetime.now().astimezone().isoformat(),
            }
        )
        atomic_dump_json(manifest_path, manifest)
        raise
    if return_code != 0 or received_signal is not None:
        status = "interrupted" if received_signal is not None else "failed"
        termination: dict[str, Any] = {
            "kind": "exit_code",
            "return_code": return_code,
            "forwarded_by_pipeline": received_signal is not None,
        }
        child_signal_name = None
        if return_code < 0:
            child_signal_number = -return_code
            try:
                child_signal_name = signal.Signals(child_signal_number).name
            except ValueError:
                child_signal_name = f"SIGNAL_{child_signal_number}"
            termination.update(
                {
                    "kind": "signal",
                    "signal": child_signal_name,
                    "signal_number": child_signal_number,
                }
            )
        manifest["stages"][stage_name].update(
            {
                "status": status,
                "return_code": return_code,
                "pipeline_signal": (
                    signal.Signals(received_signal).name if received_signal is not None else None
                ),
                "termination": termination,
                "automatic_retry": False,
                "rerun_command": shlex.join(command),
                "finished_at": datetime.now().astimezone().isoformat(),
            }
        )
        atomic_dump_json(manifest_path, manifest)
        diagnostic = ""
        if child_signal_name == "SIGKILL" and received_signal is None:
            diagnostic = (
                " The child was terminated by SIGKILL. If it was not killed explicitly, host or "
                "cgroup RAM exhaustion is a common cause; check the kernel log. For B2-Piper, "
                "disable inline video and reduce --num-envs."
            )
        raise RuntimeError(
            f"Stage '{stage_name}' {status} with code {return_code}; the next stage was not started."
            f"{diagnostic} No automatic retry was attempted."
        )

    try:
        run_dir = unique_stage_run(repo_root / "logs/rsl_rl" / experiment, run_name)
        agent_params = (run_dir / "params/agent.yaml").read_text(encoding="utf-8")
        expected_params = (
            f"max_iterations: {iterations}\n",
            f"experiment_name: {experiment}\n",
            f"run_name: {run_name}\n",
        )
        missing_params = [entry.strip() for entry in expected_params if entry not in agent_params]
        if missing_params:
            raise RuntimeError(
                f"Stage '{stage_name}' saved unexpected agent parameters: missing {missing_params}."
            )
        if actor_seed is not None:
            transfer_params = (run_dir / "params/pretrained_actor.yaml").read_text(encoding="utf-8")
            expected_transfer_params = (
                f"checkpoint: {actor_seed.resolve()}\n",
                f"source_iteration: {actor_seed_info['iteration']}\n",
            )
            missing_transfer_params = [
                entry.strip() for entry in expected_transfer_params if entry not in transfer_params
            ]
            if missing_transfer_params:
                raise RuntimeError(
                    f"Stage '{stage_name}' did not record the exact validated actor transfer: "
                    f"missing {missing_transfer_params}."
                )
        checkpoint = run_dir / f"model_{iterations - 1}.pt"
        checkpoint_info = validate_actor_checkpoint(checkpoint, expected_iteration=iterations - 1)
    except BaseException as error:
        interrupted = isinstance(error, (KeyboardInterrupt, PipelineInterrupted))
        manifest["stages"][stage_name].update(
            {
                "status": "interrupted" if interrupted else "failed_validation",
                "return_code": return_code,
                "error": f"{type(error).__name__}: {error}",
                "finished_at": datetime.now().astimezone().isoformat(),
            }
        )
        atomic_dump_json(manifest_path, manifest)
        raise
    manifest["stages"][stage_name].update(
        {
            "status": "completed",
            "return_code": return_code,
            "run_dir": str(run_dir),
            "checkpoint": checkpoint_info,
            "finished_at": datetime.now().astimezone().isoformat(),
        }
    )
    atomic_dump_json(manifest_path, manifest)
    print(f"[PIPELINE] {stage_name} completed: {checkpoint}", flush=True)
    return checkpoint.resolve()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    profile = PROFILE_SETTINGS[args.profile]
    flat_task = str(profile["flat_task"])
    rough_task = str(profile["rough_task"])
    flat_experiment = str(profile["flat_experiment"])
    rough_experiment = str(profile["rough_experiment"])
    pipeline_id = f"{args.run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    manifest_path = repo_root / "logs/rsl_rl/pipelines" / pipeline_id / "pipeline.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "pipeline_id": pipeline_id,
        "profile": args.profile,
        "status": "planned" if args.dry_run else "running",
        "created_at": datetime.now().astimezone().isoformat(),
        "configuration": {
            "profile": args.profile,
            "num_envs": args.num_envs,
            "flat_iterations": args.flat_iterations,
            "rough_iterations": args.rough_iterations,
            "flat_seed_mode": args.flat_seed,
            "seed": args.seed,
            "inline_training_video": args.video,
            "selected_tasks": {
                "flat": flat_task,
                "heading_rough": rough_task,
            },
            "selected_experiments": {
                "flat": flat_experiment,
                "heading_rough": rough_experiment,
            },
            "arm_motion_profile": profile["arm_motion_profile"],
            "rough_actor_initialization": "actor_only_from_completed_flat_stage",
        },
        "actor_contract": {
            "policy_observations": 45,
            "leg_actions": 12,
            "arm_policy_observations": False,
            "arm_policy_actions": False,
            "arm_state_privileged_to_critic": True,
            "stage_semantics": {
                "flat": "original body-frame x/y/yaw velocity tracking",
                "heading_rough": "world-heading target with turn-first gated forward velocity",
            },
        },
        "stages": {},
    }

    forwarded_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        forwarded_signals.append(signal.SIGHUP)
    previous_handlers: dict[int, Any] = {}

    def interrupt_pipeline(signum: int, _frame: Any) -> None:
        raise PipelineInterrupted(signum)

    for signum in forwarded_signals:
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, interrupt_pipeline)
    try:
        atomic_dump_json(manifest_path, manifest)
        if args.skip_flat_checkpoint is not None:
            flat_checkpoint = args.skip_flat_checkpoint.expanduser().resolve()
            if not args.dry_run:
                manifest["provided_flat_checkpoint"] = validate_actor_checkpoint(flat_checkpoint)
                validate_completed_run_checkpoint(flat_checkpoint, flat_experiment)
        else:
            if args.flat_seed == "none":
                flat_seed = None
            elif args.flat_seed == "auto-flat":
                flat_seed = highest_completed_checkpoint(
                    repo_root / "logs/rsl_rl" / BARE_B2_FLAT_EXPERIMENT
                )
                if flat_seed is None:
                    print("[PIPELINE] No bare-B2 flat checkpoint found; stage 1 will start from scratch.")
                elif not args.dry_run:
                    manifest["flat_seed_checkpoint"] = validate_actor_checkpoint(flat_seed)
            else:
                flat_seed = Path(args.flat_seed).expanduser().resolve()
                if not args.dry_run:
                    manifest["flat_seed_checkpoint"] = validate_actor_checkpoint(flat_seed)

            flat_checkpoint = run_stage(
                args=args,
                repo_root=repo_root,
                manifest_path=manifest_path,
                manifest=manifest,
                stage_name="flat",
                task=flat_task,
                experiment=flat_experiment,
                iterations=args.flat_iterations,
                actor_seed=flat_seed,
                rough=False,
            )

        rough_checkpoint = run_stage(
            args=args,
            repo_root=repo_root,
            manifest_path=manifest_path,
            manifest=manifest,
            stage_name="heading_rough",
            task=rough_task,
            experiment=rough_experiment,
            iterations=args.rough_iterations,
            actor_seed=flat_checkpoint,
            rough=True,
        )
        if not args.dry_run:
            manifest["status"] = "completed"
            manifest["final_checkpoint"] = str(rough_checkpoint)
            manifest["completed_at"] = datetime.now().astimezone().isoformat()
            atomic_dump_json(manifest_path, manifest)
            print(f"\n[PIPELINE] Complete. Manifest: {manifest_path}")
            print(f"[PIPELINE] Final checkpoint: {rough_checkpoint}")
        else:
            manifest["status"] = "dry_run"
            atomic_dump_json(manifest_path, manifest)
            print(f"\n[PIPELINE] Dry run complete. Planned manifest: {manifest_path}")
    except BaseException as error:
        stage_statuses = {stage.get("status") for stage in manifest["stages"].values()}
        interrupted = isinstance(error, (KeyboardInterrupt, PipelineInterrupted)) or "interrupted" in stage_statuses
        manifest["status"] = "interrupted" if interrupted else "failed"
        if isinstance(error, PipelineInterrupted):
            manifest["signal"] = signal.Signals(error.signum).name
        for stage in manifest["stages"].values():
            if stage.get("status") in {"pending", "running"}:
                stage["status"] = "interrupted" if interrupted else "failed"
                stage["finished_at"] = datetime.now().astimezone().isoformat()
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["failed_at"] = datetime.now().astimezone().isoformat()
        atomic_dump_json(manifest_path, manifest)
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
