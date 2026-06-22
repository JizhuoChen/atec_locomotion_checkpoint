# Task E Heuristic Physics Submission

This root-format package contains a lightweight `solution.py` policy for Task E.

The policy replays the banana and mustard segments of the heuristic sequence
validated with real physics in:

`outputs/task_e_heuristic_physics_run/seed_sweep_debug_layout`

Validation summary for the aligned debug-layout sweep:

- Seeds: `8, 9, 10, 11, 12`
- Average score: `2.0 / 3`
- Banana: `5 / 5`
- Mustard: `5 / 5`
- Box: omitted here because the tested top-down box heuristic was `0 / 5`

This package does not depend on the offline motion-request runner, object pose
writes, AnyGrasp, GraspGen, or MoveIt.

## Smoke Check

```bash
python -m py_compile solution.py server.py
python -c "from solution import AlgSolution; s=AlgSolution(); print(len(s._actions))"
```
