#!/usr/bin/env python3
"""Fine-tune a distilled 45-D student with PPO and a frozen height-scan teacher."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4


TASK = "ATEC-Isaac-Velocity-Robust-Heading-Rough-Unitree-B2-Piper-TeacherStudent-HeightScan-v0"
VARIANTS = {
    "decay": (
        "rsl_rl_hybrid_cfg_entry_point",
        "unitree_b2_piper_student_hybrid_ppo_heightscan",
    ),
    "fixed": (
        "rsl_rl_hybrid_fixed_cfg_entry_point",
        "unitree_b2_piper_student_hybrid_ppo_fixed_heightscan",
    ),
    "ppo-only": (
        "rsl_rl_student_ppo_cfg_entry_point",
        "unitree_b2_piper_student_ppo_only_heightscan",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="decay")
    parser.add_argument("--num-envs", type=int, default=256, help="Environments per GPU/process.")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--distillation-coef-start", type=float, default=None)
    parser.add_argument("--distillation-coef-end", type=float, default=None)
    parser.add_argument("--distillation-decay-iterations", type=int, default=None)
    parser.add_argument("--spawn-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.num_envs <= 0 or args.iterations <= 0 or args.num_gpus <= 0:
        parser.error("--num-envs, --iterations, and --num-gpus must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.distillation_coef_start is not None and args.distillation_coef_start < 0.0:
        parser.error("--distillation-coef-start must be non-negative")
    if args.distillation_coef_end is not None and args.distillation_coef_end < 0.0:
        parser.error("--distillation-coef-end must be non-negative")
    if args.distillation_decay_iterations is not None and args.distillation_decay_iterations < 0:
        parser.error("--distillation-decay-iterations must be non-negative")

    args.student_checkpoint = args.student_checkpoint.expanduser().resolve()
    args.teacher_checkpoint = args.teacher_checkpoint.expanduser().resolve()
    if not args.dry_run:
        if not args.student_checkpoint.is_file():
            parser.error(f"student checkpoint does not exist: {args.student_checkpoint}")
        if not args.teacher_checkpoint.is_file():
            parser.error(f"teacher checkpoint does not exist: {args.teacher_checkpoint}")
    return args


def training_command(args: argparse.Namespace, train_script: Path, run_name: str) -> list[str]:
    agent_entry_point, _ = VARIANTS[args.variant]
    command = [args.python]
    if args.num_gpus > 1:
        command.extend(
            [
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={args.num_gpus}",
            ]
        )
    command.extend(
        [
            str(train_script),
            "--task",
            TASK,
            "--agent",
            agent_entry_point,
            "--num_envs",
            str(args.num_envs),
            "--max_iterations",
            str(args.iterations),
            "--run_name",
            run_name,
            "--seed",
            str(args.seed),
            "--hybrid-student-checkpoint",
            str(args.student_checkpoint),
            "--hybrid-teacher-checkpoint",
            str(args.teacher_checkpoint),
            "--headless",
        ]
    )
    if args.num_gpus > 1:
        command.append("--distributed")
    if args.spawn_audit:
        command.append("--spawn_audit")
    if args.device is not None:
        command.extend(["--device", args.device])
    if args.distillation_coef_start is not None:
        command.extend(["--hybrid-distillation-coef-start", str(args.distillation_coef_start)])
    if args.distillation_coef_end is not None:
        command.extend(["--hybrid-distillation-coef-end", str(args.distillation_coef_end)])
    if args.distillation_decay_iterations is not None:
        command.extend(
            ["--hybrid-distillation-decay-iterations", str(args.distillation_decay_iterations)]
        )
    return command


def completed_checkpoint(repo_root: Path, experiment: str, run_name: str, iterations: int) -> Path:
    expected_name = f"model_{iterations - 1}.pt"
    candidates = sorted((repo_root / "logs/rsl_rl" / experiment).glob(f"*_{run_name}/{expected_name}"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one completed checkpoint for {run_name}, found {len(candidates)}: {candidates}"
        )
    return candidates[0].resolve()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    train_script = repo_root / "scripts/rsl_rl/train.py"
    run_id = f"hybrid_{args.variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    command = training_command(args, train_script, run_id)

    if args.dry_run:
        print("[DRY-RUN] Hybrid PPO command:")
        print(" ".join(command))
        return

    env = os.environ.copy()
    package_root = str(repo_root / "source/atec_rl_lab")
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = package_root if not old_pythonpath else f"{package_root}{os.pathsep}{old_pythonpath}"

    print(f"[HYBRID] Starting {args.variant} run: {run_id}")
    subprocess.run(command, cwd=repo_root, env=env, check=True)

    _, experiment = VARIANTS[args.variant]
    checkpoint = completed_checkpoint(repo_root, experiment, run_id, args.iterations)
    manifest = {
        "run_id": run_id,
        "variant": args.variant,
        "task": TASK,
        "student_checkpoint": str(args.student_checkpoint),
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "hybrid_checkpoint": str(checkpoint),
        "student_actor_observations": 45,
        "teacher_actor_observations": 263,
        "student_critic_observations": 263,
        "actions": 12,
        "num_gpus": args.num_gpus,
        "num_envs_per_gpu": args.num_envs,
        "num_envs_total": args.num_envs * args.num_gpus,
        "iterations": args.iterations,
        "seed": args.seed,
        "distillation_coef_start_override": args.distillation_coef_start,
        "distillation_coef_end_override": args.distillation_coef_end,
        "distillation_decay_iterations_override": args.distillation_decay_iterations,
    }
    manifest_path = checkpoint.parent / "hybrid_ppo_distillation_pipeline.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[HYBRID] Completed. Checkpoint: {checkpoint}")
    print(f"[HYBRID] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
