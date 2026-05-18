# Task E ACT Baseline Submission

This package follows the official submission entrypoint:

- File: `solution/solution.py`
- Class: `AlgSolution`
- Method: `predicts(obs, current_score)`
- Return: `{"action": action, "giveup": False}`

The packaged policy is the ACT Task E baseline from `demo/solution_act.py`, with the required ACT model code and checkpoint copied into the same `solution/` folder.

## Contents

```text
solution/
  solution.py
  policy_act.pt
  act/
  server.py
  run.sh
  requirements.txt
Dockerfile
README.md
```

## Local Evaluation

The repository-local evaluator imports `demo/solution.py`, so local evaluation of this packaged file requires either copying `solution/solution.py` into `demo/solution.py` or temporarily changing the import path in a throwaway copy of the play script.

Recommended non-destructive smoke check:

```bash
python -m py_compile solution/solution.py
```

Official Task E local command:

```bash
python scripts/play_atec_task.py --task ATEC-TaskE-Piper --enable_cameras
```

## Docker Build

From this directory:

```bash
docker build -t atec-task-e-act-baseline .
```

The Docker image starts `solution/run.sh`, which launches `solution/server.py` and imports `solution/solution.py`.

## Notes

This package is submission-valid because it only returns robot actions from `AlgSolution.predicts`. It does not include the debug-only `kinematic_attach` runner used in `scripts/task_e_moveit_runner.py`.
