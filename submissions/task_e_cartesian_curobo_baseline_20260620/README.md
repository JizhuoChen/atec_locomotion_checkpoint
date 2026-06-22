# Task E Cartesian/CuRobo Pipeline Baseline - 2026-06-20

This folder records the current validated full-pipeline baseline before further
experiments.

It is a reproducibility snapshot for
`scripts/task_e_full_anygrasp_ee_pipeline.py`, not the old server-style
`solution.py` submission package.

## Strategy

- Object order: `mustard_bottle,box_object,banana`
- Mustard: symmetric-bottle heuristic, CuRobo-selected joint execution.
- Box: top-center arm-side heuristic, CuRobo-selected pose executed through
  IsaacLab `CartesianController`.
- Banana: AnyGrasp, CuRobo-selected pose executed through IsaacLab
  `CartesianController`, up to 10 attempts.
- No full environment reset between objects.
- Re-observe/replan banana attempts from the current scene.

## Validation

The exact settings in `run_seed.sh` were tested on:

- Seed 120: all three objects reached basket; banana succeeded on attempt 3.
- Seed 130: all three objects reached basket; banana succeeded on attempt 2.

Detailed numbers are in `baseline_results.json`.

## Run

From the repository root:

```bash
bash submissions/task_e_cartesian_curobo_baseline_20260620/run_seed.sh 120
```

Optional custom output directory:

```bash
bash submissions/task_e_cartesian_curobo_baseline_20260620/run_seed.sh 130 outputs/my_seed130_baseline
```

The runner uses the active workspace script:

```text
scripts/task_e_full_anygrasp_ee_pipeline.py
```

A snapshot of that script at the time this baseline was saved is in
`snapshots/task_e_full_anygrasp_ee_pipeline.py`.

