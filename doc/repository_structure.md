# Repository Structure

This document records the intended GitHub layout and the local-only artifacts
that are intentionally excluded from version control. `log/` and `outputs/` are
omitted from the local-only tree.

## Tracked Top-Level Layout

```text
.
+-- demo/                  # Submission interface and reference solution code
+-- doc/                   # Setup, camera, SAM3, AnyGrasp, MoveIt, and pipeline notes
+-- envs/                  # Conda environment descriptors
+-- scripts/               # Evaluation, capture, segmentation, grasping, and setup tools
+-- source/                # ATEC Isaac Lab extension source
+-- submissions/           # Lightweight submission source trees; model weights ignored
`-- third_party/
    +-- anygrasp_sdk/      # Git submodule: https://github.com/graspnet/anygrasp_sdk.git
    `-- moveit2/           # Git submodule: https://github.com/moveit/moveit2.git
```

## Local-Only / Ignored Layout

```text
.
+-- ATEC2026_Simulation_Challenge(1).zip
+-- Pasted image.png
+-- atec_robot_model.zip
+-- atec_robot_model/
|   +-- baseline/
|   |   +-- act/policy.pt
|   |   `-- unitree_b2_flat/policy.pt
|   +-- objects/
|   +-- robot/
|   `-- scene/
+-- submissions/
|   +-- *.zip
|   `-- */policy_act.pt
`-- third_party/
    +-- anygrasp_sdk/
    |   +-- checkpoint_detection.tar
    |   +-- checkpoint_tracking.tar
    |   +-- dependencies/MinkowskiEngine/
    |   +-- grasp_detection/
    |   |   +-- example_data/
    |   |   +-- gsnet.so
    |   |   +-- lib_cxx.so
    |   |   `-- license/
    |   +-- grasp_tracking/
    |   |   +-- example_data/
    |   |   +-- lib_cxx.so
    |   |   +-- tracker.so
    |   |   `-- license/
    |   +-- license/
    |   `-- pointnet2/build/
    `-- openssl11/
```

## Fresh Clone Notes

Third-party sources are submodules, while checkpoints and generated binaries are
still local artifacts. After cloning, initialize source links with:

```bash
git submodule update --init --recursive
```

Use the setup scripts in `scripts/` to recreate local AnyGrasp, MoveIt, OpenSSL,
checkpoint, and license artifacts as needed.
