#!/usr/bin/env python3
"""Run Task E pipeline seeds with fresh-process retries until success or trial limit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = REPO_ROOT / "scripts/task_e_full_anygrasp_ee_pipeline.py"


def parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(item.strip()) for item in part.split("-", 1)]
            step = 1 if end >= start else -1
            seeds.extend(range(start, end + step, step))
        else:
            seeds.append(int(part))
    if not seeds:
        raise argparse.ArgumentTypeError("Expected at least one seed.")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=parse_seeds, required=True, help="Comma/range list, e.g. 11,12,13 or 11-15.")
    parser.add_argument("--max-failed-trials", type=int, default=3)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, default=PIPELINE)
    parser.add_argument(
        "pipeline_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to task_e_full_anygrasp_ee_pipeline.py after a literal --.",
    )
    args = parser.parse_args()
    if args.pipeline_args and args.pipeline_args[0] == "--":
        args.pipeline_args = args.pipeline_args[1:]
    args.max_failed_trials = max(1, int(args.max_failed_trials))
    return args


def load_trial_result(output_dir: Path) -> dict:
    summary_path = output_dir / "pipeline_summary.json"
    if not summary_path.exists():
        return {
            "status": "missing_summary",
            "target_in_basket": False,
            "summary_path": str(summary_path),
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    objects = (summary.get("execution") or {}).get("objects") or []
    target = objects[-1] if objects else {}
    selection = None
    if summary.get("records"):
        for record in summary["records"].values():
            if record.get("grasp_selection"):
                selection = record.get("grasp_selection")
                break
    return {
        "status": target.get("status", "unknown"),
        "ok": target.get("ok"),
        "target_in_basket": bool(target.get("target_in_basket")),
        "objects_in_basket": target.get("objects_in_basket"),
        "selected_rank": (selection or {}).get("selected_rank"),
        "selected_ok": (selection or {}).get("selected_ok"),
        "selected_score": (selection or {}).get("selected_score"),
        "summary_path": str(summary_path),
    }


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for seed in args.seeds:
        failed_trials = 0
        seed_success = False
        trial = 0
        while failed_trials < args.max_failed_trials and not seed_success:
            trial += 1
            output_dir = args.output_root / f"seed{seed}_trial{trial:02d}"
            command = [
                sys.executable,
                str(args.pipeline),
                "--seed",
                str(seed),
                "--output",
                str(output_dir),
                *args.pipeline_args,
            ]
            log_path = output_dir / "trial.log"
            output_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[INFO] seed={seed} trial={trial} failed_so_far={failed_trials}/{args.max_failed_trials}",
                flush=True,
            )
            proc = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            log_path.write_text(proc.stdout or "", encoding="utf-8")
            trial_result = load_trial_result(output_dir)
            trial_result.update(
                {
                    "seed": seed,
                    "trial": trial,
                    "returncode": int(proc.returncode),
                    "output_dir": str(output_dir),
                    "log_path": str(log_path),
                }
            )
            if proc.returncode != 0:
                trial_result["status"] = "process_failed"
                trial_result["target_in_basket"] = False
            seed_success = bool(trial_result["target_in_basket"])
            if not seed_success:
                failed_trials += 1
            trial_result["failed_trials_after"] = failed_trials
            results.append(trial_result)
            status = "success" if seed_success else "failed"
            print(
                f"[INFO] seed={seed} trial={trial} {status} "
                f"rank={trial_result.get('selected_rank')} summary={trial_result.get('summary_path')}",
                flush=True,
            )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seeds": args.seeds,
        "max_failed_trials": int(args.max_failed_trials),
        "pipeline": str(args.pipeline),
        "pipeline_args": args.pipeline_args,
        "trials": results,
        "seed_successes": {
            str(seed): any(item["seed"] == seed and item.get("target_in_basket") for item in results)
            for seed in args.seeds
        },
    }
    summary["success_count"] = int(sum(1 for ok in summary["seed_successes"].values() if ok))
    summary["seed_count"] = int(len(args.seeds))
    summary_path = args.output_root / "seed_trial_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO] Saved trial summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
