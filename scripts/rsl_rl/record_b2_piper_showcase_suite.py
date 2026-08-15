#!/usr/bin/env python3
"""Record a long, multi-difficulty terrain showcase for a completed B2-Piper policy."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml


CHECKPOINT_RE = re.compile(r"model_(\d+)\.pt$")
SAFE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")
BASELINE_TERRAIN_FAMILIES = (
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "boxes",
    "random_rough",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
)
ROBUST_TERRAIN_FAMILIES = (
    "stairs_short_tread",
    "stairs_short_tread_inv",
    "stairs_nominal_tread",
    "stairs_nominal_tread_inv",
    "stairs_long_tread",
    "stairs_long_tread_inv",
    "rough_fine",
    "rough_medium",
    "rough_hard",
    "boxes",
    "slope",
    "slope_inv",
)
POLICY_STEP_DT_S = 0.02


@dataclass(frozen=True)
class ShowcaseProfile:
    """Checkpoint provenance and visualization contract for one training profile."""

    task: str
    experiment: str
    terrain_families: tuple[str, ...]
    montage_columns: int


SHOWCASE_PROFILES = {
    "baseline": ShowcaseProfile(
        task="ATEC-Isaac-Velocity-Heading-Rough-Unitree-B2-Piper-v0",
        experiment="unitree_b2_piper_heading_rough",
        terrain_families=BASELINE_TERRAIN_FAMILIES,
        montage_columns=3,
    ),
    "robust": ShowcaseProfile(
        task="ATEC-Isaac-Velocity-Robust-Heading-Rough-Unitree-B2-Piper-v0",
        experiment="unitree_b2_piper_robust_heading_rough",
        terrain_families=ROBUST_TERRAIN_FAMILIES,
        montage_columns=4,
    ),
}

# Backward-compatible baseline aliases for callers that import these helpers.
TASK = SHOWCASE_PROFILES["baseline"].task
EXPERIMENT = SHOWCASE_PROFILES["baseline"].experiment
TERRAIN_FAMILIES = BASELINE_TERRAIN_FAMILIES
TERRAIN_FAMILY_COUNT = len(TERRAIN_FAMILIES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run play.py once per difficulty and record one long clip for every rough-terrain "
            "family in the selected B2-Piper profile."
        )
    )
    parser.add_argument(
        "--profile",
        choices=tuple(SHOWCASE_PROFILES),
        default="baseline",
        help="Training/evaluation profile (default: baseline).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Completed B2-Piper heading-rough model_N.pt for the selected profile; omit to "
            "select its latest completed run."
        ),
    )
    parser.add_argument(
        "--difficulties",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75, 1.00],
        help="Normalized terrain difficulties to record (default: 0.25 0.50 0.75 1.00).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Procedural-scene seeds. If supplied, every difficulty is recorded for every seed. "
            "By default one distinct seed is derived for each difficulty."
        ),
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="First derived seed when --seeds is omitted (default: 42).",
    )
    parser.add_argument(
        "--steps-per-terrain",
        type=int,
        default=1000,
        help=(
            "Policy steps per clip (1000 steps is a 20-second clip; it spans 2.5 robust "
            "episode horizons)."
        ),
    )
    parser.add_argument(
        "--suite-name",
        default=None,
        help="Output folder name below the checkpoint's videos/play directory.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for play.py.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.steps_per_terrain <= 0:
        parser.error("--steps-per-terrain must be positive")
    if any(not 0.0 <= difficulty <= 1.0 for difficulty in args.difficulties):
        parser.error("every --difficulties value must be between 0 and 1")
    if len(set(args.difficulties)) != len(args.difficulties):
        parser.error("--difficulties must not contain duplicates")
    if args.seed_base < 0:
        parser.error("--seed-base must be non-negative")
    if args.seeds is not None and any(seed < 0 for seed in args.seeds):
        parser.error("every --seeds value must be non-negative")
    if args.seeds is not None and len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    if args.suite_name is not None and SAFE_NAME_RE.fullmatch(args.suite_name) is None:
        parser.error(
            "--suite-name must be 1-96 characters and contain only letters, numbers, '.', '_', or '-'"
        )
    return args


def latest_completed_checkpoint(experiment_dir: Path, experiment: str = EXPERIMENT) -> Path:
    """Select the highest-iteration exact final checkpoint, breaking ties by recency."""
    candidates: list[tuple[int, float, str, Path]] = []
    if experiment_dir.is_dir():
        for run_dir in experiment_dir.iterdir():
            if not run_dir.is_dir():
                continue
            params_path = run_dir / "params/agent.yaml"
            try:
                params = params_path.read_text(encoding="utf-8")
            except OSError:
                continue
            if f"experiment_name: {experiment}\n" not in params:
                continue
            try:
                iteration = completed_run_iteration(run_dir, params)
            except (KeyError, OSError, TypeError, ValueError, RuntimeError, yaml.YAMLError):
                continue
            checkpoint = run_dir / f"model_{iteration}.pt"
            if checkpoint.is_file():
                candidates.append(
                    (iteration, checkpoint.stat().st_mtime, str(checkpoint.resolve()), checkpoint.resolve())
                )
    if not candidates:
        raise FileNotFoundError(f"No completed checkpoint found under {experiment_dir}.")
    return max(candidates)[3]


def completed_run_iteration(run_dir: Path, agent_params: str | None = None) -> int:
    """Resolve the final iteration of either a fresh or full-state resumed run."""
    resume_path = run_dir / "params/resume.yaml"
    if resume_path.is_file():
        resume = yaml.safe_load(resume_path.read_text(encoding="utf-8"))
        if not isinstance(resume, dict) or resume.get("status") != "completed":
            raise RuntimeError(f"Run is not marked as a completed resume: {resume_path}")
        final_iteration = int(resume["final_iteration"])
        if final_iteration != int(resume["expected_final_iteration"]):
            raise RuntimeError(f"Resume iteration metadata is inconsistent: {resume_path}")
        final_checkpoint = Path(resume["final_checkpoint"]).expanduser().resolve()
        expected_checkpoint = (run_dir / f"model_{final_iteration}.pt").resolve()
        if final_checkpoint != expected_checkpoint:
            raise RuntimeError(f"Resume final checkpoint path is inconsistent: {resume_path}")
        return final_iteration

    if agent_params is None:
        agent_params = (run_dir / "params/agent.yaml").read_text(encoding="utf-8")
    iterations_match = re.search(r"^max_iterations:\s*(\d+)\s*$", agent_params, re.MULTILINE)
    if iterations_match is None:
        raise RuntimeError(f"Cannot find max_iterations in {run_dir / 'params/agent.yaml'}.")
    return int(iterations_match.group(1)) - 1


def validate_checkpoint(checkpoint: Path, experiment: str = EXPERIMENT) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    match = CHECKPOINT_RE.fullmatch(checkpoint.name)
    if match is None or not checkpoint.is_file():
        raise FileNotFoundError(f"Expected an existing model_N.pt checkpoint, got: {checkpoint}")
    params_path = checkpoint.parent / "params/agent.yaml"
    try:
        params = params_path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"Cannot verify checkpoint metadata at {params_path}.") from error
    if f"experiment_name: {experiment}\n" not in params:
        raise RuntimeError(f"Checkpoint is not from the required '{experiment}' experiment: {checkpoint}")
    expected_iteration = completed_run_iteration(checkpoint.parent, params)
    if int(match.group(1)) != expected_iteration:
        raise RuntimeError(
            f"Showcase requires the completed run's model_{expected_iteration}.pt, not {checkpoint.name}."
        )
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or int(payload.get("iter", -1)) != expected_iteration:
        raise RuntimeError(
            "Checkpoint payload iteration does not match its filename/run metadata: "
            f"{checkpoint}."
        )
    return checkpoint


def atomic_dump_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def difficulty_tag(difficulty: float) -> str:
    """Return a compact percentage label while rejecting ambiguous rounding collisions upstream."""
    return f"d{round(difficulty * 100):03d}"


def probe_duration(video: Path) -> float | None:
    if shutil.which("ffprobe") is None:
        return None
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def verify_recordings(
    output_dir: Path,
    steps_per_terrain: int,
    difficulty: float,
    seed: int,
    terrain_families: tuple[str, ...] = TERRAIN_FAMILIES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cross-check videos against the manifest before assigning terrain labels."""
    manifest_path = output_dir / "terrain_showcase.yaml"
    try:
        showcase = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RuntimeError(f"Missing or unreadable terrain manifest in {output_dir}.") from error
    if not isinstance(showcase, dict):
        raise RuntimeError(f"Malformed terrain showcase manifest: {manifest_path}")

    terrain_family_count = len(terrain_families)
    expected_scalars = {
        "schema_version": 2,
        "interval_steps": steps_per_terrain,
        "cycle_length_steps": steps_per_terrain * terrain_family_count,
        "terrain_family_count": terrain_family_count,
        "seed": seed,
        "standing_command_probability": 0.0,
    }
    mismatches = {
        key: {"expected": expected, "actual": showcase.get(key)}
        for key, expected in expected_scalars.items()
        if showcase.get(key) != expected
    }
    try:
        actual_difficulty = float(showcase["difficulty"])
        policy_step_dt = float(showcase["policy_step_dt_s"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Manifest has invalid difficulty or policy step time: {manifest_path}") from error
    if abs(actual_difficulty - difficulty) > 1.0e-9:
        mismatches["difficulty"] = {"expected": difficulty, "actual": actual_difficulty}
    if policy_step_dt <= 0.0:
        mismatches["policy_step_dt_s"] = {"expected": "> 0", "actual": policy_step_dt}
    if mismatches:
        raise RuntimeError(f"Terrain showcase manifest mismatch in {manifest_path}: {mismatches}")

    segments = showcase.get("segments")
    if not isinstance(segments, list) or len(segments) != terrain_family_count:
        raise RuntimeError(
            f"Expected {terrain_family_count} manifest segments in {manifest_path}, got {segments!r}."
        )
    terrain_names = tuple(segment.get("terrain") for segment in segments if isinstance(segment, dict))
    if terrain_names != terrain_families:
        raise RuntimeError(
            f"Terrain order changed; refusing to mislabel the montage. Expected {terrain_families}, "
            f"got {terrain_names}."
        )

    expected_names = []
    for index, segment in enumerate(segments):
        expected_start = index * steps_per_terrain
        expected_name = f"terrain-showcase-step-{expected_start}.mp4"
        if segment.get("start_step_in_cycle") != expected_start or segment.get("video_file") != expected_name:
            raise RuntimeError(f"Invalid segment {index} in {manifest_path}: {segment}")
        if Path(expected_name).name != expected_name:
            raise RuntimeError(f"Unsafe video filename in {manifest_path}: {expected_name}")
        expected_names.append(expected_name)

    actual_names = sorted(path.name for path in output_dir.glob("*.mp4"))
    if sorted(expected_names) != actual_names:
        raise RuntimeError(
            f"Expected exactly {expected_names} in {output_dir}, found {actual_names}."
        )

    expected_duration = steps_per_terrain * policy_step_dt
    recordings = []
    for segment, name in zip(segments, expected_names, strict=True):
        video = output_dir / name
        if video.stat().st_size == 0:
            raise RuntimeError(f"Recorded an empty video: {video}")
        duration = probe_duration(video)
        if duration is not None and abs(duration - expected_duration) > 0.25:
            raise RuntimeError(
                f"Unexpected duration for {video}: {duration:.3f}s; expected about {expected_duration:.3f}s."
            )
        recordings.append(
            {
                "path": str(video.resolve()),
                "terrain": segment["terrain"],
                "start_step_in_cycle": segment["start_step_in_cycle"],
                "bytes": video.stat().st_size,
                "duration_s": duration,
            }
        )
    return recordings, showcase


def make_all_terrain_montage(
    recordings: list[dict[str, Any]],
    output: Path,
    difficulty: float,
    seed: int,
    terrain_families: tuple[str, ...] = TERRAIN_FAMILIES,
    montage_columns: int = 3,
) -> dict[str, Any]:
    """Combine all labeled family clips into one synchronized grid."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to create the all-terrain montage.")
    terrain_family_count = len(terrain_families)
    if len(recordings) != terrain_family_count:
        raise ValueError(f"Expected {terrain_family_count} recordings, got {len(recordings)}.")
    if montage_columns <= 0 or montage_columns > terrain_family_count:
        raise ValueError(
            f"Montage columns must be between 1 and {terrain_family_count}, got {montage_columns}."
        )

    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    for recording in recordings:
        command.extend(["-i", recording["path"]])
    filters = []
    terrain_order = [recording["terrain"] for recording in recordings]
    if tuple(terrain_order) != terrain_families:
        raise ValueError(
            f"Recording order does not match the selected profile: {terrain_order!r}."
        )
    for index, family in enumerate(terrain_order):
        label = f"{family}  difficulty {difficulty:.2f}  seed {seed}"
        filters.append(
            f"[{index}:v]scale=640:360,"
            f"drawtext=text='{label}':x=12:y=12:fontsize=22:fontcolor=white:"
            f"box=1:boxcolor=black@0.65:boxborderw=7[v{index}]"
        )
    inputs = "".join(f"[v{index}]" for index in range(terrain_family_count))
    layout = "|".join(
        f"{(index % montage_columns) * 640}_{(index // montage_columns) * 360}"
        for index in range(terrain_family_count)
    )
    filters.append(f"{inputs}xstack=inputs={terrain_family_count}:layout={layout}[out]")
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
    )
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to create all-terrain montage: {output}")
    return {
        "path": str(output.resolve()),
        "bytes": output.stat().st_size,
        "duration_s": probe_duration(output),
        "layout": f"{montage_columns}x{(terrain_family_count + montage_columns - 1) // montage_columns}",
        "terrain_order": terrain_order,
    }


def main() -> int:
    args = parse_args()
    profile = SHOWCASE_PROFILES[args.profile]
    terrain_family_count = len(profile.terrain_families)
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint = validate_checkpoint(
        args.checkpoint
        if args.checkpoint is not None
        else latest_completed_checkpoint(
            repo_root / "logs/rsl_rl" / profile.experiment, profile.experiment
        ),
        profile.experiment,
    )
    suite_name = args.suite_name or (
        f"terrain-showcase-suite-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
    )
    suite_dir = checkpoint.parent / "videos/play" / suite_name
    manifest_path = suite_dir / "suite.json"

    difficulty_seed_pairs = (
        [
            (difficulty, args.seed_base + index)
            for index, difficulty in enumerate(args.difficulties)
        ]
        if args.seeds is None
        else [
            (difficulty, seed)
            for difficulty in args.difficulties
            for seed in args.seeds
        ]
    )
    labels = [
        f"{difficulty_tag(difficulty)}-seed{seed}"
        for difficulty, seed in difficulty_seed_pairs
    ]
    if len(set(labels)) != len(labels):
        raise ValueError(
            "Difficulty values collapse to duplicate percentage labels; choose values at least 0.01 apart."
        )

    runs = []
    for difficulty, seed in difficulty_seed_pairs:
        label = f"{difficulty_tag(difficulty)}-seed{seed}"
        relative_output = f"{suite_name}/{label}"
        command = [
            args.python,
            str(repo_root / "scripts/rsl_rl/play.py"),
            "--task",
            profile.task,
            "--checkpoint",
            str(checkpoint),
            "--terrain_showcase",
            "--showcase_difficulty",
            str(difficulty),
            "--showcase_steps_per_terrain",
            str(args.steps_per_terrain),
            "--showcase_output_name",
            relative_output,
            "--seed",
            str(seed),
        ]
        if args.device:
            command.extend(["--device", args.device])
        if args.headless:
            command.append("--headless")
        runs.append(
            {
                "label": label,
                "difficulty": difficulty,
                "seed": seed,
                "status": "planned" if args.dry_run else "pending",
                "output_dir": str((suite_dir / label).resolve()),
                "command": command,
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "suite_name": suite_name,
        "status": "dry_run" if args.dry_run else "running",
        "created_at": datetime.now().astimezone().isoformat(),
        "profile": args.profile,
        "checkpoint": str(checkpoint),
        "experiment": profile.experiment,
        "task": profile.task,
        "steps_per_terrain": args.steps_per_terrain,
        "nominal_clip_duration_s": args.steps_per_terrain * POLICY_STEP_DT_S,
        "terrain_families": list(profile.terrain_families),
        "terrain_families_per_run": terrain_family_count,
        "montage_layout": (
            f"{profile.montage_columns}x"
            f"{(terrain_family_count + profile.montage_columns - 1) // profile.montage_columns}"
        ),
        "expected_individual_video_count": len(runs) * terrain_family_count,
        "expected_montage_count": len(runs),
        "expected_video_count": len(runs) * (terrain_family_count + 1),
        "expected_total_video_duration_s": (
            len(runs)
            * (terrain_family_count + 1)
            * args.steps_per_terrain
            * POLICY_STEP_DT_S
        ),
        "runs": runs,
    }

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    if suite_dir.exists():
        raise FileExistsError(f"Showcase suite already exists and will not be overwritten: {suite_dir}")
    suite_dir.mkdir(parents=True, exist_ok=False)
    atomic_dump_json(manifest_path, manifest)

    try:
        for run in runs:
            run["status"] = "running"
            run["started_at"] = datetime.now().astimezone().isoformat()
            atomic_dump_json(manifest_path, manifest)
            print(f"\n[SHOWCASE] Recording {run['label']}:\n  " + " ".join(run["command"]), flush=True)
            completed = subprocess.run(run["command"], cwd=repo_root, check=False)
            run["return_code"] = completed.returncode
            if completed.returncode != 0:
                run["status"] = "failed"
                raise RuntimeError(
                    f"Showcase run {run['label']} failed with return code {completed.returncode}."
                )
            run["recordings"], showcase_manifest = verify_recordings(
                Path(run["output_dir"]),
                args.steps_per_terrain,
                run["difficulty"],
                run["seed"],
                profile.terrain_families,
            )
            run["verified_showcase_manifest"] = str(
                (Path(run["output_dir"]) / "terrain_showcase.yaml").resolve()
            )
            run["policy_step_dt_s"] = showcase_manifest["policy_step_dt_s"]
            run["all_terrain_montage"] = make_all_terrain_montage(
                run["recordings"],
                suite_dir / f"{run['label']}-all-terrains.mp4",
                run["difficulty"],
                run["seed"],
                profile.terrain_families,
                profile.montage_columns,
            )
            run["status"] = "completed"
            run["finished_at"] = datetime.now().astimezone().isoformat()
            atomic_dump_json(manifest_path, manifest)
    except BaseException as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        manifest["status"] = "interrupted" if interrupted else "failed"
        for run in runs:
            if run["status"] == "running":
                run["status"] = "interrupted" if interrupted else "failed"
                run["finished_at"] = datetime.now().astimezone().isoformat()
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["finished_at"] = datetime.now().astimezone().isoformat()
        atomic_dump_json(manifest_path, manifest)
        raise

    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now().astimezone().isoformat()
    atomic_dump_json(manifest_path, manifest)
    print(f"\n[SHOWCASE] Complete: {manifest_path}")
    print(f"[SHOWCASE] Verified {manifest['expected_video_count']} videos in {suite_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
