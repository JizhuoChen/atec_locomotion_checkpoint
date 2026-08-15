#!/usr/bin/env python3
"""Continue the completed robust B2-Piper rough policy with full PPO state."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from train_b2_piper_pipeline import (
    atomic_dump_json,
    run_child,
    validate_actor_checkpoint,
    validate_run_experiment,
)


TASK = "ATEC-Isaac-Velocity-Robust-Heading-Rough-Unitree-B2-Piper-v0"
EXPERIMENT = "unitree_b2_piper_robust_heading_rough"
RUN_PREFIX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
FULL_STATE_SHAPES = {
    "std": (12,),
    "actor.0.weight": (512, 45),
    "actor.0.bias": (512,),
    "actor.2.weight": (256, 512),
    "actor.2.bias": (256,),
    "actor.4.weight": (128, 256),
    "actor.4.bias": (128,),
    "actor.6.weight": (12, 128),
    "actor.6.bias": (12,),
    "critic.0.weight": (512, 251),
    "critic.0.bias": (512,),
    "critic.2.weight": (256, 512),
    "critic.2.bias": (256,),
    "critic.4.weight": (128, 256),
    "critic.4.bias": (128,),
    "critic.6.weight": (1, 128),
    "critic.6.bias": (1,),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completed_iteration(run_dir: Path) -> int:
    """Resolve fresh and resumed run completion without guessing from filenames."""
    resume_path = run_dir / "params" / "resume.yaml"
    if resume_path.is_file():
        resume = yaml.safe_load(resume_path.read_text(encoding="utf-8"))
        if not isinstance(resume, dict) or resume.get("status") != "completed":
            raise RuntimeError(f"Resume metadata does not mark a completed run: {resume_path}")
        final_iteration = int(resume["final_iteration"])
        if final_iteration != int(resume["expected_final_iteration"]):
            raise RuntimeError(f"Inconsistent final iteration in {resume_path}")
        return final_iteration

    params_path = run_dir / "params" / "agent.yaml"
    params = params_path.read_text(encoding="utf-8")
    match = re.search(r"^max_iterations:\s*(\d+)\s*$", params, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"Cannot find max_iterations in {params_path}")
    return int(match.group(1)) - 1


def _latest_completed_pipeline_checkpoint(repo_root: Path) -> Path:
    candidates: list[tuple[float, str, Path]] = []
    for manifest_path in (repo_root / "logs" / "rsl_rl" / "pipelines").glob("*/pipeline.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "completed" or manifest.get("profile") != "robust":
                continue
            checkpoint = Path(manifest["final_checkpoint"]).expanduser().resolve()
            if checkpoint.is_file():
                candidates.append((manifest_path.stat().st_mtime, str(checkpoint), checkpoint))
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    if not candidates:
        raise FileNotFoundError("No completed robust pipeline checkpoint was found.")
    return max(candidates)[2]


def _validate_full_resume_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    path = path.expanduser().resolve()
    actor_info = validate_actor_checkpoint(path)
    validate_run_experiment(path, EXPERIMENT)
    expected_iteration = _completed_iteration(path.parent)
    if actor_info["iteration"] != expected_iteration:
        raise RuntimeError(
            f"Continuation requires the completed run's model_{expected_iteration}.pt, not {path.name}."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    actual_shapes = {
        name: tuple(value.shape) for name, value in state.items() if hasattr(value, "shape")
    }
    if actual_shapes != FULL_STATE_SHAPES:
        raise RuntimeError(
            "Resume checkpoint actor/critic/std contract changed: "
            f"expected {FULL_STATE_SHAPES}, got {actual_shapes}."
        )
    optimizer = checkpoint.get("optimizer_state_dict")
    if not isinstance(optimizer, dict) or not optimizer.get("param_groups"):
        raise RuntimeError(f"Checkpoint has no resumable optimizer: {path}")
    learning_rates = {float(group["lr"]) for group in optimizer["param_groups"]}
    if len(learning_rates) != 1:
        raise RuntimeError(f"Checkpoint has multiple optimizer learning rates: {learning_rates}")
    return {
        **actor_info,
        "critic_input_dim": 251,
        "critic_output_dim": 1,
        "optimizer_state_count": len(optimizer.get("state", {})),
        "optimizer_learning_rate": learning_rates.pop(),
        "completed_run": str(path.parent),
    }


def _validate_local_checkpoint_location(path: Path, repo_root: Path) -> None:
    """Ensure train.py's regex lookup will resolve the exact validated file."""
    expected_experiment_dir = (repo_root / "logs" / "rsl_rl" / EXPERIMENT).resolve()
    actual_experiment_dir = path.resolve().parent.parent
    if actual_experiment_dir != expected_experiment_dir:
        raise ValueError(
            "Continuation checkpoints must be directly below this repository's experiment tree so "
            "train.py can load the exact validated file. Expected "
            f"{expected_experiment_dir}/<run>/model_N.pt, got {path.resolve()}."
        )


def _source_seed(path: Path) -> int:
    """Read the source runner seed so an omitted override remains explicit."""
    params_path = path.parent / "params" / "agent.yaml"
    params = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    if not isinstance(params, dict) or not isinstance(params.get("seed"), int):
        raise RuntimeError(f"Cannot resolve an integer source seed from {params_path}.")
    seed = int(params["seed"])
    if seed < 0:
        raise RuntimeError(f"Source checkpoint has an invalid negative seed: {seed}.")
    return seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Exact completed robust rough model_N.pt; defaults to the latest completed robust pipeline.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=4000,
        help="Additional PPO updates to run after the saved iteration (default: 4000).",
    )
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--run-prefix", default="b2_piper_robust_finegrain")
    parser.add_argument("--spawn-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.iterations <= 0 or args.num_envs <= 0:
        parser.error("--iterations and --num-envs must be positive")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be non-negative")
    if RUN_PREFIX_RE.fullmatch(args.run_prefix) is None:
        parser.error(
            "--run-prefix must be 1-64 characters containing letters, numbers, '.', '_' or '-'."
        )
    if not args.headless and args.num_envs > 64:
        parser.error("Live rendering is limited to at most 64 B2-Piper environments.")
    return args


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else _latest_completed_pipeline_checkpoint(repo_root)
    )
    _validate_local_checkpoint_location(checkpoint, repo_root)
    checkpoint_info = _validate_full_resume_checkpoint(checkpoint)
    source_seed = _source_seed(checkpoint)
    effective_seed = args.seed if args.seed is not None else source_seed
    continuation_id = (
        f"{args.run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    )
    run_name = continuation_id
    load_run_pattern = f"^{re.escape(checkpoint.parent.name)}$"
    checkpoint_pattern = f"^{re.escape(checkpoint.name)}$"
    command = [
        str(Path(args.python).expanduser()),
        str(repo_root / "scripts" / "rsl_rl" / "train.py"),
        "--task",
        TASK,
        "--num_envs",
        str(args.num_envs),
        "--max_iterations",
        str(args.iterations),
        "--run_name",
        run_name,
        "--resume",
        "--load_run",
        load_run_pattern,
        "--checkpoint",
        checkpoint_pattern,
    ]
    if args.spawn_audit:
        command.append("--spawn_audit")
    if args.headless:
        command.append("--headless")
    if args.device:
        command.extend(("--device", args.device))
    command.extend(("--seed", str(effective_seed)))

    expected_final_iteration = checkpoint_info["iteration"] + args.iterations - 1
    manifest_path = repo_root / "logs" / "rsl_rl" / "pipelines" / continuation_id / "pipeline.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "pipeline_id": continuation_id,
        "workflow": "full_state_rough_continuation",
        "profile": "robust",
        "status": "dry_run" if args.dry_run else "running",
        "created_at": datetime.now().astimezone().isoformat(),
        "task": TASK,
        "experiment": EXPERIMENT,
        "source_checkpoint": checkpoint_info,
        "configuration": {
            "num_envs": args.num_envs,
            "additional_iterations": args.iterations,
            "expected_final_iteration": expected_final_iteration,
            "seed": effective_seed,
            "seed_source": "command_line" if args.seed is not None else "source_checkpoint",
            "spawn_audit": args.spawn_audit,
            "full_state_resume": True,
            "environment_state_restored": False,
            "terrain_geometry": {
                "stair_riser_range_m": [0.04, 0.26],
                "stair_tread_bands_m": [[0.22, 0.28], [0.28, 0.35], [0.35, 0.42]],
                "stair_tread_resolution_m": 0.005,
            },
            "b2_only_payload_proxy": {
                "fraction": 0.25,
                "arm_mass_inertia_scale_range": [0.05, 0.10],
                "persistent_across_resets": True,
                "stratified_by_terrain_type": True,
                "actor_observation_changed": False,
                "critic_observation_changed": False,
            },
        },
        "implementation_sha256": {
            "continue_b2_piper_rough.py": _sha256(Path(__file__).resolve()),
            "train.py": _sha256(repo_root / "scripts" / "rsl_rl" / "train.py"),
            "robust_terrain_cfg.py": _sha256(
                repo_root
                / "source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/"
                "quadruped/unitree_b2/robust_terrain_cfg.py"
            ),
            "robust_piper_env_cfg.py": _sha256(
                repo_root
                / "source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/"
                "quadruped/unitree_b2/robust_piper_env_cfg.py"
            ),
            "commands.py": _sha256(
                repo_root
                / "source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/commands.py"
            ),
            "robust_events.py": _sha256(
                repo_root
                / "source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/robust_events.py"
            ),
            "piper_env_cfg.py": _sha256(
                repo_root
                / "source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/"
                "quadruped/unitree_b2/piper_env_cfg.py"
            ),
            "curriculums.py": _sha256(
                repo_root
                / "source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/curriculums.py"
            ),
        },
        "command": command,
    }
    atomic_dump_json(manifest_path, manifest)
    print("[CONTINUE] Command:\n  " + shlex.join(command), flush=True)
    print(
        f"[CONTINUE] Full PPO state from iteration {checkpoint_info['iteration']}; "
        f"expected final checkpoint model_{expected_final_iteration}.pt.",
        flush=True,
    )
    if args.dry_run:
        print(f"[CONTINUE] Dry-run manifest: {manifest_path}")
        return 0

    try:
        return_code, received_signal = run_child(command, repo_root)
        if return_code != 0 or received_signal is not None:
            child_signal_number = -return_code if return_code < 0 else None
            child_signal_name = None
            if child_signal_number is not None:
                try:
                    child_signal_name = signal.Signals(child_signal_number).name
                except ValueError:
                    child_signal_name = f"SIGNAL_{child_signal_number}"
            status = "interrupted" if received_signal is not None else "failed"
            manifest.update(
                {
                    "status": status,
                    "return_code": return_code,
                    "termination": {
                        "child_signal": child_signal_name,
                        "child_signal_number": child_signal_number,
                        "forwarded_by_launcher": received_signal is not None,
                        "launcher_signal": (
                            signal.Signals(received_signal).name
                            if received_signal is not None
                            else None
                        ),
                    },
                    "rerun_command": command,
                    f"{status}_at": datetime.now().astimezone().isoformat(),
                }
            )
            atomic_dump_json(manifest_path, manifest)
            print(
                f"[CONTINUE] {status}: child return code {return_code}; "
                f"launcher signal={received_signal}. Manifest: {manifest_path}",
                file=sys.stderr,
            )
            if received_signal is not None:
                return 128 + received_signal
            return return_code if return_code > 0 else 128 + (child_signal_number or 1)
        matches = sorted(
            path
            for path in (repo_root / "logs" / "rsl_rl" / EXPERIMENT).glob(f"*_{run_name}")
            if path.is_dir()
        )
        if len(matches) != 1:
            raise RuntimeError(f"Expected one continuation run, found {matches}.")
        run_dir = matches[0].resolve()
        final_checkpoint = run_dir / f"model_{expected_final_iteration}.pt"
        final_info = validate_actor_checkpoint(
            final_checkpoint, expected_iteration=expected_final_iteration
        )
        resume_path = run_dir / "params" / "resume.yaml"
        resume = yaml.safe_load(resume_path.read_text(encoding="utf-8"))
        if (
            not isinstance(resume, dict)
            or resume.get("status") != "completed"
            or int(resume.get("final_iteration", -1)) != expected_final_iteration
            or Path(resume.get("checkpoint", "")).resolve() != checkpoint
        ):
            raise RuntimeError(f"Continuation resume provenance is inconsistent: {resume_path}")
        manifest.update(
            {
                "status": "completed",
                "run_dir": str(run_dir),
                "final_checkpoint": str(final_checkpoint.resolve()),
                "final_checkpoint_info": final_info,
                "completed_at": datetime.now().astimezone().isoformat(),
            }
        )
        atomic_dump_json(manifest_path, manifest)
        print(f"[CONTINUE] Completed: {final_checkpoint}")
        return 0
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "failed_at": datetime.now().astimezone().isoformat(),
            }
        )
        atomic_dump_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
