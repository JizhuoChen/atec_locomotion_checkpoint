# Task E AnyGrasp Usage

This stage is isolated from IsaacLab and SAM3. It consumes an already segmented EE-camera RGB-D observation and writes an AnyGrasp grasp pose that can replace the heuristic pseudo grasp in the motion-request pipeline.

## Inputs

The stage expects:

- EE RGB PNG, for example `ee_rgb.png`
- EE depth `.npy` in metres, for example `ee_depth.npy`
- binary SAM3 mask PNG in the same image frame, for example `sam3_ee/ee_banana_mask.png`
- camera metadata JSON with intrinsics and pose, for example `ee_camera.json`

Expected SDK artifacts:

```text
third_party/anygrasp_sdk/checkpoint_detection.tar
third_party/anygrasp_sdk/license/licenseCfg.json
third_party/anygrasp_sdk/license/*.lic
third_party/openssl11/lib/libcrypto.so.1.1
third_party/openssl11/lib/libssl.so.1.1
```

`scripts/anygrasp_from_rgbd_mask.py` links `third_party/anygrasp_sdk/license` into `third_party/anygrasp_sdk/grasp_detection/license` if needed and preloads the isolated OpenSSL 1.1 libraries before importing the vendor SDK.

## Generate Grasp From Segmented EE RGB-D

Run from the repo root:

```bash
conda run -n anygrasp \
  python scripts/anygrasp_from_rgbd_mask.py \
  --rgb outputs/task_e_banana_pipeline/latest/ee_rgb.png \
  --depth-npy outputs/task_e_banana_pipeline/latest/ee_depth.npy \
  --mask outputs/task_e_banana_pipeline/latest/sam3_ee/ee_banana_mask.png \
  --camera-json outputs/task_e_banana_pipeline/latest/ee_camera.json \
  --output outputs/task_e_banana_pipeline/latest/anygrasp
```

Useful optional flags:

```bash
--checkpoint-path third_party/anygrasp_sdk/checkpoint_detection.tar
--license-dir third_party/anygrasp_sdk/license
--openssl-lib-dir third_party/openssl11/lib
--no-collision-detection
--dense-grasp
```

The default mode uses AnyGrasp object-mask filtering and collision detection. `--no-collision-detection` is useful as a diagnostic if the segmented cloud is partial or noisy.

## Outputs

```text
anygrasp/
  anygrasp_result.json
  final_grasp_pose.json
  anygrasp_result.png
  masked_cloud.npy
  masked_cloud_colors.npy
  masked_cloud.ply
```

`anygrasp_result.json` records the full diagnostic payload, including point count, workspace limits, SDK paths, retry attempts, grasp counts, and best grasp score.

`final_grasp_pose.json` is the compact downstream contract:

```json
{
  "frame": "world",
  "pose_type": "anygrasp_gripper",
  "approach_axis": "rotation_matrix[:,0]",
  "pregrasp_direction": "-rotation_matrix[:,0]",
  "translation": [1.14, -0.04, 1.15],
  "rotation_matrix": [[...], [...], [...]],
  "score": 0.34,
  "width": 0.068,
  "depth": 0.020
}
```

The pose is an AnyGrasp/GraspNet gripper-frame pose. In GraspNet geometry, local `+X` points from the gripper tail toward the fingers, so the pregrasp direction is `-rotation_matrix[:,0]`.

## Convert To Motion Request

Convert the AnyGrasp pose to the same unified request format used by the MoveIt/IK runner:

```bash
python scripts/anygrasp_pose_to_motion_request.py \
  --anygrasp-pose outputs/task_e_banana_pipeline/latest/anygrasp/final_grasp_pose.json \
  --output outputs/task_e_banana_pipeline/latest/anygrasp/motion_request.json \
  --target banana \
  --seed 7
```

Then execute it with the existing runner:

```bash
env -u DISPLAY conda run -n isaaclab \
  python scripts/task_e_moveit_runner.py \
  --request outputs/task_e_banana_pipeline/latest/anygrasp/motion_request.json \
  --output outputs/task_e_banana_pipeline/latest/anygrasp/moveit_run \
  --headless
```

## Banana Test Result

Using the current banana EE capture:

```text
outputs/task_e_banana_pipeline/latest/anygrasp_isolated_collision_test/
```

AnyGrasp returned:

```text
point_count: 33563
status: ok
raw grasps: 1198
NMS grasps: 13
best score: 0.34565985202789307
```

The generated files are:

```text
anygrasp_result.json
final_grasp_pose.json
motion_request.json
anygrasp_result.png
masked_cloud.ply
```

The generated motion request was also accepted by `scripts/task_e_moveit_runner.py`:

```text
outputs/task_e_banana_pipeline/latest/anygrasp_isolated_collision_test/moveit_run/motion_result.json
ok: true
waypoints executed: 5
```

This confirms the AnyGrasp output can replace the heuristic grasp pose at the file-contract level.

## Caveat Before Execution

The current converter uses the AnyGrasp gripper pose directly as a Piper `gripper_base` target. This is enough to replace the heuristic pose for pipeline plumbing, but before trusting physical execution you should calibrate or verify the fixed transform between the AnyGrasp gripper frame and the Piper `gripper_base` frame.

If the SDK reports `license_failed`, the binary and checkpoint were found but the license does not match the current machine feature id. Re-check the license folder before debugging the point cloud code.
