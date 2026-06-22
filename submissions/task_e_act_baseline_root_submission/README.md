# Task E ACT + Scripted Fallback Submission

This archive is root-entrypoint compatible. The package root contains:

```text
solution.py
policy_act.pt
act/
server.py
run.sh
requirements.txt
Dockerfile
README.md
```

`solution.py` defines the official `AlgSolution` class and `predicts(obs, current_score)` method.
This final variant keeps the ACT policy and adds a reset-safe scripted fallback
for the remaining manipulation sequence.

## Local Smoke Check

```bash
python -m py_compile solution.py
python -c "from solution import AlgSolution; print(type(AlgSolution()).__name__)"
```

Latest local simulator check:

```bash
python scripts/debug_task_e_submission_policy.py --headless --max_steps 3500 \
  --submission-dir submissions/task_e_act_baseline_root_submission
```

Observed score: `6.00` on a fresh non-deterministic Task E layout.

## Docker Build

```bash
docker build -t atec-task-e-act-baseline-root .
```

The Dockerfile copies the root files into `/home/admin/appspace/atec2026/robot/solution/` for the server runtime.
