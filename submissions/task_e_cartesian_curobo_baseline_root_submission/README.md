# Task E Root-Format Baseline Submission

This folder uses the same root-level submission layout as
`submissions/task_e_heuristic_physics_root_submission`:

```text
Dockerfile
README.md
requirements.txt
run.sh
server.py
solution.py
```

The previous folder,
`submissions/task_e_cartesian_curobo_baseline_20260620`, is a local
reproducibility snapshot for the offline AnyGrasp/CuRobo pipeline. It is not a
submit-ready package because it contains `run_seed.sh`, result JSON, and a
pipeline script snapshot instead of the required server `solution.py` entrypoint.

This root-format folder keeps the submit-compatible `AlgSolution` server policy
structure. It does not depend on the offline motion-request runner, AnyGrasp, or
CuRobo at submission runtime.

## Smoke Check

```bash
python -m py_compile solution.py server.py
python -c "from solution import AlgSolution; s=AlgSolution(); print(len(s._actions))"
```
