# Task E MoveIt / IK Usage

The motion stage uses a stable JSON request/result contract. The current executable runner can consume the same request format now with IsaacLab Cartesian IK fallback, while MoveIt2 source and build helpers live under `third_party/`.

## Files

- `scripts/pseudo_grasp_to_motion_request.py`: converts `pseudo_grasp.json` into a unified motion request.
- `scripts/task_e_full_baseline_request.py`: creates a full three-object Task E request using the same contract.
- `scripts/task_e_moveit_runner.py`: consumes the motion request and writes a unified motion result.
- `scripts/setup_moveit2_source.sh`: downloads MoveIt2 source to `third_party/moveit2`.
- `scripts/build_moveit2_source.sh`: builds MoveIt2 up to `moveit_py` after `rosdep` is initialized.

## Request Generation

Full Task E baseline:

```bash
python scripts/task_e_full_baseline_request.py \
  --seed 7 \
  --grasp-tuning 'mustard_bottle:0,0,0.055;box_object:0,0,0.06' \
  --object-transport-mode kinematic_attach
```

Default output:

```text
outputs/task_e_full_baseline/latest/motion_request.json
```

Single-object banana debug path:

```bash
python scripts/pseudo_grasp_to_motion_request.py \
  --pseudo-grasp outputs/task_e_banana_pipeline/latest/pseudo_grasp/pseudo_grasp.json \
  --seed 7
```

Default output:

```text
outputs/task_e_banana_pipeline/latest/pseudo_grasp/motion_request.json
```

The seed matters. Task E randomizes object placement during environment creation, so capture and execution must use the same seed if they happen in separate processes.

## Request Schema

```json
{
  "schema": "atec.task_e.motion_request.v1",
  "task": "ATEC-TaskE-Piper",
  "frame": "world",
  "start_state": {"source": "task_reset", "seed": 7},
  "backend": {"preferred": "moveit_py", "fallback": "isaaclab_cartesian_controller"},
  "controller": {
    "actuator_mode": "task_e_scripted_high_stiffness",
    "object_transport_mode": "kinematic_attach",
    "grasp_quat_source": "scripts/act/task_e/state_machine.py"
  },
  "robot": {
    "name": "piper",
    "planning_group": "piper_arm",
    "ee_link": "gripper_base",
    "arm_joints": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
    "gripper_joints": ["joint7", "joint8"],
    "action_scale": 0.5
  },
  "waypoints": [
    {
      "name": "02_grasp",
      "pose_w": {"position": [1.0, 0.0, 0.9], "quat_wxyz": [0.0, 1.0, 0.0, 0.0]},
      "gripper_joint_pos": [0.035, -0.035],
      "steps": 140,
      "capture": true,
      "object_transport": {
        "action": "attach",
        "object_key": "object_3",
        "object_name": "banana",
        "ee_to_object_pos_w": [0.0, 0.0, -0.09],
        "object_quat_wxyz": [0.0, 0.0, -0.707, 0.707]
      }
    }
  ]
}
```

## Execute Request

Full Task E baseline:

```bash
env -u DISPLAY conda run -n isaaclab \
  python scripts/task_e_moveit_runner.py \
  --request outputs/task_e_full_baseline/latest/motion_request.json \
  --output outputs/task_e_full_baseline/latest/run_visual \
  --headless
```

Record the external camera and planned pre-grasp overlays:

```bash
env -u DISPLAY conda run -n isaaclab \
  python scripts/task_e_moveit_runner.py \
  --request outputs/task_e_full_baseline/latest/motion_request.json \
  --output outputs/task_e_full_baseline/latest/run_video \
  --headless \
  --record-video-cam \
  --video-every-n-steps 2 \
  --video-fps 15 \
  --save-pregrasp-viz
```

This writes:

```text
video_cam.mp4
pregrasp_predictions/
```

Fast semantic check without frame saving:

```bash
env -u DISPLAY conda run -n isaaclab \
  python scripts/task_e_moveit_runner.py \
  --request outputs/task_e_full_baseline/latest/motion_request.json \
  --output outputs/task_e_full_baseline/latest/run_check \
  --headless \
  --no-save-frames
```

Single-object banana debug request:

```bash
env -u DISPLAY conda run -n isaaclab \
  python scripts/task_e_moveit_runner.py \
  --request outputs/task_e_banana_pipeline/latest/pseudo_grasp/motion_request.json \
  --headless
```

Default output:

```text
outputs/task_e_moveit/latest/
  motion_request.json
  motion_result.json
  00_initial.png
  00_camera_look.png
  01_pregrasp.png
  02_grasp.png
  03_close.png
  04_lift.png
```

## Result Schema

```json
{
  "schema": "atec.task_e.motion_result.v1",
  "ok": true,
  "seed": 7,
  "backend": {
    "requested": "moveit_py",
    "used": "isaaclab_cartesian_controller",
    "reason": "moveit_py unavailable; using simulator IK fallback"
  },
  "waypoints": [
    {
      "name": "02_grasp",
      "ok": true,
      "target_pose_w": {"position": [1.0, 0.0, 0.9], "quat_wxyz": [0.0, 1.0, 0.0, 0.0]},
      "target_gripper_joint_pos": [0.035, -0.035],
      "ee_pose_w": {"position": [1.0, 0.0, 0.96], "quat_wxyz": [0.0, 1.0, 0.0, 0.0]},
      "position_error_m": 0.05
    }
  ],
  "controller": {
    "actuator_mode": "task_e_scripted_high_stiffness",
    "object_transport_mode": "kinematic_attach",
    "request_object_poses_applied": true
  },
  "task_e_objects": {
    "count_in_basket": 3,
    "all_in_basket": true
  },
  "artifacts": {"frames": {"04_lift": "04_lift.png"}}
}
```

`ok` means the runner did not hit an execution failure. Task E may terminate early after semantic success, so the final saved frame can be the last successful place waypoint rather than the requested final retract.

`task_e_objects` is the semantic Task E check for the three rigid objects. For the full baseline, prefer `task_e_objects.all_in_basket` over `ok`.

`object_transport_mode` has two modes:

- `kinematic_attach`: deterministic integration/debug mode. The runner attaches the requested object to the end effector after the close waypoint and writes it into the basket on release.
- `physics`: contact-only mode. This is closer to a real policy check, but the current pseudo grasps are not robust enough for every object.

## MoveIt2 Setup

Download source:

```bash
bash scripts/setup_moveit2_source.sh
```

Build after `rosdep` is initialized:

```bash
sudo rosdep init
rosdep update
bash scripts/build_moveit2_source.sh
source third_party/moveit2_ws/install/setup.bash
```

Current limitation: IsaacLab is running in Python 3.11, while ROS Jazzy/MoveIt Python modules are normally built for the system Python on Ubuntu 24.04. Until a compatible MoveIt execution process is wired in, `task_e_moveit_runner.py` records MoveIt availability and uses the IsaacLab Cartesian IK fallback. The request/result JSON is the boundary to keep the future true MoveIt process isolated.
