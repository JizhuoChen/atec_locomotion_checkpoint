# Task E Camera Capture

This stage captures the two official Task E camera streams:

- `video_rgb.png`: external eye-to-hand camera.
- `ee_rgb.png`: eye-in-hand camera mounted on `gripper_base`.
- `metadata.json`: camera mode, seed, robot state, and camera configuration notes.

## Isolated Debug Capture

Use this when you only need PNGs for detection/debugging:

```bash
env -u DISPLAY conda run -n isaaclab \
  python scripts/capture_task_e_cameras.py \
  --headless \
  --seed 7 \
  --look-at-table \
  --output outputs/task_e_camera_capture/seed7_table
```

Important options:

- `--seed`: fixes object randomization. Use the same seed later for motion execution.
- `--look-at-table`: moves the end-effector camera to a verified table-looking pose.
- `--action "a0,a1,...,a7"`: apply a custom raw Task E action before capture.

## Official Evaluation Path

Use the official test entrypoint when validating submission behavior:

```bash
ATEC_SAM3_CAPTURE=1 \
python scripts/play_atec_task.py --task ATEC-TaskE-Piper --enable_cameras --headless
```

That path exposes images through the Task E observation:

- `obs["image"]["video_rgb"]`
- `obs["image"]["ee_rgb"]`

## Output Contract

Downstream SAM3 scripts only require:

```text
<capture_dir>/
  video_rgb.png
  ee_rgb.png
  metadata.json
```

If motion execution is separated from capture, keep `metadata.json["seed"]` and pass the same seed into the motion request. Without a fixed seed, each Task E reset may place objects differently.
