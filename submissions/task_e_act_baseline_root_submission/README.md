# Task E ACT Baseline Submission

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

## Local Smoke Check

```bash
python -m py_compile solution.py
python -c "from solution import AlgSolution; print(type(AlgSolution()).__name__)"
```

## Docker Build

```bash
docker build -t atec-task-e-act-baseline-root .
```

The Dockerfile copies the root files into `/home/admin/appspace/atec2026/robot/solution/` for the server runtime.
