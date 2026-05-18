# Task E Grasp Pipeline

The pipeline is split into isolated stages with file-based contracts:

1. Camera capture: [task_e_camera_capture_readme.md](task_e_camera_capture_readme.md)
2. SAM3 masks and pseudo grasp: [task_e_sam3_usage_readme.md](task_e_sam3_usage_readme.md)
3. AnyGrasp grasp generation: [task_e_anygrasp_usage_readme.md](task_e_anygrasp_usage_readme.md)
4. MoveIt/IK request execution: [task_e_moveit_usage_readme.md](task_e_moveit_usage_readme.md)
5. AnyGrasp environment setup: [anygrasp_setup.md](anygrasp_setup.md)

## Full Task E Baseline

This baseline reuses the same motion request/result contract as the isolated banana path, but generates a complete three-object sequence:

1. banana to purple basket
2. mustard bottle to purple basket
3. yellow and white box to purple basket

Generate the request:

```bash
python scripts/task_e_full_baseline_request.py \
  --seed 7 \
  --grasp-tuning 'mustard_bottle:0,0,0.055;box_object:0,0,0.06' \
  --object-transport-mode kinematic_attach
```

Execute it in Task E:

```bash
env -u DISPLAY conda run -n isaaclab \
  python scripts/task_e_moveit_runner.py \
  --request outputs/task_e_full_baseline/latest/motion_request.json \
  --output outputs/task_e_full_baseline/latest/run_visual \
  --headless
```

Execute the same policy while recording the external video camera and saving pre-grasp prediction overlays:

```bash
python scripts/task_e_full_baseline_request.py \
  --output outputs/task_e_whole_policy/latest/motion_request.json \
  --seed 7 \
  --grasp-tuning 'mustard_bottle:0,0,0.055;box_object:0,0,0.06' \
  --object-transport-mode kinematic_attach

env -u DISPLAY conda run -n isaaclab \
  python scripts/task_e_moveit_runner.py \
  --request outputs/task_e_whole_policy/latest/motion_request.json \
  --output outputs/task_e_whole_policy/latest/run \
  --headless \
  --record-video-cam \
  --video-every-n-steps 2 \
  --video-fps 15 \
  --save-pregrasp-viz
```

Main artifacts:

```text
outputs/task_e_whole_policy/latest/run/video_cam.mp4
outputs/task_e_whole_policy/latest/run/pregrasp_predictions/all_pregrasp_predictions.png
outputs/task_e_whole_policy/latest/run/motion_result.json
```

Check the result:

```bash
python - <<'PY'
import json
from pathlib import Path
result = json.loads(Path("outputs/task_e_full_baseline/latest/run_visual/motion_result.json").read_text())
print("ok:", result["ok"])
print("in basket:", result["task_e_objects"]["count_in_basket"], "/ 3")
for key, item in result["task_e_objects"]["objects"].items():
    print(key, item["label"], item["in_basket"], item["pos_local"])
PY
```

The default full-task baseline uses `kinematic_attach`. This is a deterministic integration/debug mode that keeps the selected object attached to the end effector during transport and releases it in the basket. It is useful for validating the camera, SAM3, grasp/motion contract, and waypoint sequencing while AnyGrasp licensing and true contact grasping are still being finished.

Task E can terminate as soon as all three objects satisfy the basket condition. In that case the last saved visualization may be the final object place frame instead of the requested final retract frame.

For contact-only testing, generate or override the request with:

```bash
python scripts/task_e_full_baseline_request.py \
  --seed 7 \
  --grasp-tuning 'mustard_bottle:0,0,0.055;box_object:0,0,0.06' \
  --object-transport-mode physics
```

`physics` uses only simulator contact. With the current pseudo grasp, banana and mustard are reasonable debug targets, but the yellow and white box is too wide for the simple top-down grasp and should be replaced by an AnyGrasp pose or a better object-specific grasp.

## Banana Debug Path

```bash
# 1. Generate or reuse pseudo_grasp.json from SAM3.
python scripts/pseudo_grasp_from_task_e_output.py \
  --input outputs/task_e_banana_pipeline/latest \
  --pose-source sam3-mask \
  --sam3-mask outputs/task_e_banana_pipeline/latest/pseudo_grasp/sam3_prompt_test/video_curved_yellow_banana_mask.png \
  --grasp-xy-offset 0.03 -0.006 \
  --quat-source default-topdown

# 2. Convert grasp to unified motion request.
python scripts/pseudo_grasp_to_motion_request.py \
  --pseudo-grasp outputs/task_e_banana_pipeline/latest/pseudo_grasp/pseudo_grasp.json

# 3. Execute request in Task E simulator.
env -u DISPLAY conda run -n isaaclab \
  python scripts/task_e_moveit_runner.py \
  --request outputs/task_e_banana_pipeline/latest/pseudo_grasp/motion_request.json \
  --headless
```

## Contracts

- Perception output: mask PNG, overlay PNG, detections JSON.
- Grasp output: `pseudo_grasp.json` now; AnyGrasp `final_grasp_pose.json` later.
- Motion input: `atec.task_e.motion_request.v1`.
- Motion output: `atec.task_e.motion_result.v1`.

For isolated reruns, use a fixed Task E seed in capture and in the motion request. Otherwise the simulator reset may place objects differently from the captured scene.
