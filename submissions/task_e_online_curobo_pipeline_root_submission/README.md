# Task E Online Latest Policy Submission

This is the online submit-format port of the seed-120/130 full-success pipeline.

It does not replay a fixed action sequence. At runtime, `solution.py`:

- reads the current `video_rgb` and `video_depth` observations,
- estimates the current target object center,
- selects or interpolates the latest successful seed-120/130 actuator command profiles,
- follows the same mustard -> box -> banana order,
- advances by `current_score`, so it does not waste steps after an object is scored,
- retries banana with an alternate successful profile if the first one does not score.

This is as close as the submission API permits to the offline pipeline behavior:
the server only gives `obs` and expects the next action, so `solution.py` cannot
own the Isaac environment object, run SAM/AnyGrasp subprocesses, or call the
original pipeline `main()` directly.

## Files

```text
Dockerfile
README.md
requirements.txt
run.sh
server.py
solution.py
action_profiles.json
pose_references.json
task_e_curobo_planner.py
curobo_assets/
```

## Runtime Knobs

- `ATEC_TASK_E_PROFILE_STEP_SCALE=0.62` controls waypoint duration scaling.
- `ATEC_TASK_E_PROFILE_MIN_STEPS=12` controls the shortest transition duration.

## Smoke Check

```bash
python -m py_compile solution.py server.py task_e_curobo_planner.py
python -c "from solution import AlgSolution; print(type(AlgSolution()).__name__)"
```
