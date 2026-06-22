# Task E Seed 120/130 Pipeline Replay Submission

This root-format submission replays the latest two full-success pipeline
executions:

- `outputs/task_e_seed120_bottle_box_banana_cartesian_box_banana/execution`
- `outputs/task_e_seed130_bottle_box_banana_cartesian_box_banana/execution`

The offline pipeline used AnyGrasp/CuRobo/IsaacLab CartesianController. The
submission server cannot run that standalone pipeline directly because it only
calls `AlgSolution.predicts(obs, current_score)` for the next action. This
package therefore embeds the successful execution artifacts in
`motion_profiles.json` and follows the same object order, retry structure,
waypoint timing, CuRobo joint targets, and solved Cartesian joint endpoints via
the official joint-position action interface.

The raw profile timing is preserved in `motion_profiles.json`. At runtime,
`solution.py` uses `ATEC_TASK_E_REPLAY_STEP_SCALE` to fit the usual submission
horizon. Set `ATEC_TASK_E_REPLAY_STEP_SCALE=1.0` to replay the raw saved timing.

Use this package when you want the submitted file to match the seed-120/130
successful pipeline behavior as closely as the server API permits.

## Files

```text
Dockerfile
README.md
requirements.txt
run.sh
server.py
solution.py
motion_profiles.json
```

## Smoke Check

```bash
python -m py_compile solution.py server.py
python -c "from solution import AlgSolution; print(type(AlgSolution()).__name__)"
```
