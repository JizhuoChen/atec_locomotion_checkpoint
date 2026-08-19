#!/usr/bin/env python3
"""Run privileged-teacher PPO followed by warm-started student distillation."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4


TASK = "ATEC-Isaac-Velocity-Robust-Heading-Rough-Unitree-B2-Piper-TeacherStudent-v0"
TEACHER_EXPERIMENT = "unitree_b2_piper_privileged_teacher"
STUDENT_EXPERIMENT = "unitree_b2_piper_student_distillation"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=repo_root / "reference/model_19998_run/model_19998.pt",
        help="Canonical 45-input actor/251-input critic checkpoint.",
    )
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--teacher-iterations", type=int, default=6000)
    parser.add_argument("--student-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--teacher-resume-checkpoint",
        type=Path,
        default=None,
        help="Resume the teacher from model_N.pt and train until --teacher-iterations total iterations.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--spawn-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_envs <= 0 or args.teacher_iterations <= 0 or args.student_iterations <= 0:
        parser.error("environment and iteration counts must be positive")
    if args.num_gpus <= 0:
        parser.error("--num-gpus must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    args.base_checkpoint = args.base_checkpoint.expanduser().resolve()
    if not args.dry_run and not args.base_checkpoint.is_file():
        parser.error(f"base checkpoint does not exist: {args.base_checkpoint}")
    if args.teacher_resume_checkpoint is not None:
        args.teacher_resume_checkpoint = args.teacher_resume_checkpoint.expanduser().resolve()
        if not args.dry_run and not args.teacher_resume_checkpoint.is_file():
            parser.error(f"teacher resume checkpoint does not exist: {args.teacher_resume_checkpoint}")
    return args


def stage_command(
    *,
    args: argparse.Namespace,
    train_script: Path,
    agent: str,
    iterations: int,
    run_name: str,
    initialization_args: list[str],
) -> list[str]:
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
        agent,
        "--num_envs",
        str(args.num_envs),
        "--max_iterations",
        str(iterations),
        "--run_name",
        run_name,
        "--seed",
        str(args.seed),
        "--headless",
        *initialization_args,
        ]
    )
    if args.num_gpus > 1:
        command.append("--distributed")
    if args.spawn_audit:
        command.append("--spawn_audit")
    if args.device is not None:
        command.extend(["--device", args.device])
    return command


def checkpoint_iteration(checkpoint: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", checkpoint.name)
    if match is None:
        raise ValueError(f"Teacher resume checkpoint must be named model_N.pt: {checkpoint}")
    return int(match.group(1))


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
    pipeline_id = f"teacher_student_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    teacher_run = f"{pipeline_id}_teacher"
    student_run = f"{pipeline_id}_student"

    env = os.environ.copy()
    package_root = str(repo_root / "source/atec_rl_lab")
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = package_root if not old_pythonpath else f"{package_root}{os.pathsep}{old_pythonpath}"

    teacher_resume_iteration = None
    teacher_stage_iterations = args.teacher_iterations
    teacher_initialization_args = ["--pretrained_privileged_teacher", str(args.base_checkpoint)]
    if args.teacher_resume_checkpoint is not None:
        teacher_resume_iteration = checkpoint_iteration(args.teacher_resume_checkpoint)
        teacher_stage_iterations = args.teacher_iterations - teacher_resume_iteration
        if teacher_stage_iterations <= 0:
            raise ValueError(
                f"Teacher checkpoint iteration {teacher_resume_iteration} already reaches "
                f"the requested total {args.teacher_iterations}."
            )
        teacher_initialization_args = [
            "--resume",
            "--load_run",
            args.teacher_resume_checkpoint.parent.name,
            "--checkpoint",
            args.teacher_resume_checkpoint.name,
        ]

    teacher_command = stage_command(
        args=args,
        train_script=train_script,
        agent="rsl_rl_teacher_cfg_entry_point",
        iterations=teacher_stage_iterations,
        run_name=teacher_run,
        initialization_args=teacher_initialization_args,
    )
    if args.dry_run:
        print("[DRY-RUN] Teacher command:")
        print(" ".join(teacher_command))
        print("[DRY-RUN] Student command will use the teacher's final model_N.pt checkpoint.")
        return

    print(f"[PIPELINE] Training privileged teacher: {teacher_run}")
    subprocess.run(teacher_command, cwd=repo_root, env=env, check=True)
    teacher_checkpoint = completed_checkpoint(
        repo_root, TEACHER_EXPERIMENT, teacher_run, args.teacher_iterations
    )

    student_command = stage_command(
        args=args,
        train_script=train_script,
        agent="rsl_rl_distillation_cfg_entry_point",
        iterations=args.student_iterations,
        run_name=student_run,
        initialization_args=[
            "--teacher_checkpoint",
            str(teacher_checkpoint),
            "--pretrained_student",
            str(args.base_checkpoint),
        ],
    )
    print(f"[PIPELINE] Distilling deployment student: {student_run}")
    subprocess.run(student_command, cwd=repo_root, env=env, check=True)
    student_checkpoint = completed_checkpoint(
        repo_root, STUDENT_EXPERIMENT, student_run, args.student_iterations
    )

    manifest = {
        "pipeline_id": pipeline_id,
        "task": TASK,
        "base_checkpoint": str(args.base_checkpoint),
        "teacher_checkpoint": str(teacher_checkpoint),
        "student_checkpoint": str(student_checkpoint),
        "teacher_actor_observations": 76,
        "teacher_critic_observations": 263,
        "student_observations": 45,
        "actions": 12,
        "seed": args.seed,
        "num_gpus": args.num_gpus,
        "num_envs_per_gpu": args.num_envs,
        "num_envs_total": args.num_envs * args.num_gpus,
        "teacher_resume_checkpoint": (
            str(args.teacher_resume_checkpoint) if args.teacher_resume_checkpoint is not None else None
        ),
        "teacher_resume_iteration": teacher_resume_iteration,
        "num_envs": args.num_envs,
        "teacher_iterations": args.teacher_iterations,
        "student_iterations": args.student_iterations,
    }
    manifest_path = student_checkpoint.parent / "teacher_student_pipeline.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[PIPELINE] Completed. Student checkpoint: {student_checkpoint}")
    print(f"[PIPELINE] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
