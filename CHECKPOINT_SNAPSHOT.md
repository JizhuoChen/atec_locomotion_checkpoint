# Canonical `model_19998.pt` training snapshot

This parallel repository is a byte-verified reconstruction of the code state that produced:

`logs/rsl_rl/unitree_b2_piper_robust_heading_rough/2026-08-07_19-04-45_b2_piper_robust_finegrain_20260807_190441_a74cf724/model_19998.pt`

Canonical checkpoint SHA-256:

`995b9d11ae99648255e5c213baaa14cfbe31b0380f76e976a3469a593322bfd9`

Do not confuse it with the later one-iteration smoke checkpoint that also happened to be named `model_19998.pt` under `unitree_b2_piper_terrain_turn`; that file has a different hash.

## What was reconstructed

The checkout starts from competition commit `2f4bd998386717bd8e4484db43fc8e3b9c0aee5c`. The August 7 tracked and untracked changes were reconstructed from the run artifacts and the local patch ledger. All eight implementation files recorded by the run's pipeline manifest match their historical SHA-256 values exactly. The original `Clear_ATEC2026_Simulation_Challenge` checkout was not changed.

This folder contains:

- the exact training code and task registration;
- the complete extracted `atec_robot_model` asset tree;
- the exact `model_11999.pt` full-state checkpoint at the log path expected by the original continuation command;
- the canonical final checkpoint, TensorBoard events, exports, resolved YAML, Git patch and provenance under `reference/model_19998_run`;
- the flat `model_7999.pt` and rough `model_11999.pt` lineage endpoints under `reference/lineage`;
- both original pipeline manifests and a package freeze under `reference/provenance`.

The local snapshot has the complete extracted `atec_robot_model` tree. The published repository includes the ten runtime files needed by this B2-Piper locomotion task; unrelated competition assets follow the upstream Git-ignore rule. See `README.md` if you also need those other task assets.

Intermediate checkpoints were intentionally omitted. They are not needed to execute the producer command or inspect the final policy.

## Exact producer command

First verify the snapshot:

```bash
cd /home/user/jz/atec/Clear_ATEC2026_checkpoint
/home/user/miniforge3/envs/isaaclab/bin/python verify_checkpoint_snapshot.py --all-assets
```

For a fresh GitHub clone containing only the required locomotion assets, omit `--all-assets`. Use that flag after copying the complete competition asset tree.

Then launch the same 8,000-update, full-PPO-state continuation that produced iteration 19,998:

```bash
cd /home/user/jz/atec/Clear_ATEC2026_checkpoint
./train_model_19998.sh
```

The launcher prepends this snapshot's package directory to `PYTHONPATH`. This matters because the existing Isaac Lab conda environment has an editable install that otherwise points at the newer repository. It checks the source checkpoint hash before starting and accepts additional CLI arguments, for example `./train_model_19998.sh --device cuda:0`.

The exact underlying historical command, rewritten only to use this parallel path, is:

```bash
PYTHONPATH=/home/user/jz/atec/Clear_ATEC2026_checkpoint/source/atec_rl_lab \
/home/user/miniforge3/envs/isaaclab/bin/python \
  /home/user/jz/atec/Clear_ATEC2026_checkpoint/scripts/rsl_rl/train.py \
  --task ATEC-Isaac-Velocity-Robust-Heading-Rough-Unitree-B2-Piper-v0 \
  --num_envs 2048 \
  --max_iterations 8000 \
  --run_name b2_piper_robust_finegrain_20260807_190441_a74cf724 \
  --resume \
  --load_run '^2026\-08\-06_23\-17\-23_b2_piper_robust_20260806_215402_69aaacaf_heading_rough$' \
  --checkpoint '^model_11999\.pt$' \
  --spawn_audit \
  --headless \
  --seed 42
```

The expected new endpoint is `model_19998.pt`: RSL-RL resumes from iteration 11,999 and labels 8,000 additional updates as `11999 + 8000 - 1 = 19998`.

## Historical three-stage lineage

The canonical model was not trained from random initialization in one run:

1. Flat robust B2-Piper PPO, 2,048 environments and 8,000 updates, produced `model_7999.pt` (SHA `8d9d184d...`).
2. A fresh rough-terrain PPO run copied **actor weights only** from the flat checkpoint. Its critic, optimizer, action standard deviation and iteration counter started fresh. After 12,000 updates it produced `model_11999.pt` (SHA `c7a8b3b...`).
3. The August 7 run restored the **full PPO state** from `model_11999.pt`, changed to the fine-grained terrain/proxy configuration in this snapshot, and trained 8,000 more updates to canonical `model_19998.pt`.

The exact stage-one and stage-two commands are preserved in `reference/provenance/pipeline_aug06_flat_to_rough.json`. Those two stages ran before the August 7 fine-grained terrain edits, so rerunning them with this August 7 code is a new experiment rather than a literal recreation of the historical `model_11999.pt`. The supplied source checkpoint is the correct way to reproduce stage three.

## Task, simulator and model

- Gym task: `ATEC-Isaac-Velocity-Robust-Heading-Rough-Unitree-B2-Piper-v0`.
- Robot: B2 with Piper asset `atec_robot_model/robot/b2/b2_piper.usda`.
- Simulation: 5 ms physics step, decimation 4, hence 50 Hz policy actions; 8 s episodes.
- Parallelism: 2,048 environments from the CLI; environment seed 42.
- Algorithm: online, on-policy PPO with 24 steps/environment/update, 5 learning epochs, 4 minibatches, gamma 0.99, lambda 0.95, adaptive learning rate starting at `3e-4`, entropy coefficient `0.005`, clip parameter `0.2` and desired KL `0.01`.
- Network: ELU MLPs. Actor `45 -> 512 -> 256 -> 128 -> 12`; critic `251 -> 512 -> 256 -> 128 -> 1`. No observation normalization and no symmetry augmentation in this run.

The actor controls only the 12 leg joint-position targets. Hip action scale is 0.125, other leg joints 0.25, with the default pose as offset. A 0-1 policy-tick delay, target bias `+/-0.01 rad`, and per-step target noise `+/-0.003 rad` are applied.

### Actor and critic observations

The 45-dimensional actor input is:

- base angular velocity (3);
- projected gravity (3);
- generated forward/yaw command (3);
- 12 relative leg joint positions;
- 12 relative leg joint velocities;
- previous 12-dimensional action.

The arm is deliberately absent from the actor input and action. The privileged 251-dimensional critic also receives base linear velocity, all 20 leg/arm joint positions and velocities, and a 187-point terrain height scan. Thus arm movement affects the actor only through its physical effect on IMU-like body observations, while the critic can explain that disturbance during training.

## Command and heading-first behavior

Each 8 s episode gets a simultaneous command `[gated forward speed, 0, yaw rate]`, not a separate turn command followed by a translation command.

- Raw desired forward speed is 0.25-1.0 m/s; lateral command is always zero.
- Target world heading spans `[-pi, pi]`, with yaw-rate command clamped to `[-1, 1]` rad/s.
- The Gaussian alignment gate has standard deviation 0.55 rad. Large heading error suppresses the published forward speed; as alignment improves, forward speed smoothly opens.
- Heading targets point approximately through the assigned terrain tile center, with at most 0.35 rad deviation.
- Two percent of environments receive a hold command.

This is why maximizing return means turning toward the desired heading first and then progressing forward while suppressing sideways motion.

## Scene and terrain setup

The generator is a curriculum grid of 10 rows by 80 columns, with 8 m by 8 m tiles and 12 equally weighted terrain families:

- stair ascent/descent with risers continuously curriculum-scaled from 0.04 to 0.26 m;
- three independent tread bands: 0.22-0.28 m, 0.28-0.35 m and 0.35-0.42 m, sampled at 5 mm resolution;
- signed random roughness at maximum amplitudes `+/-0.040`, `+/-0.075` and `+/-0.115` m, starting at 30% amplitude at the easiest level;
- random boxes 0.05-0.22 m high;
- up/down slopes from 0 to 0.45.

Robots spawn on validated flat patches inside their assigned tile. The saved audit reports 2,048 environments across 399 occupied cells, no duplicate spawn positions and no workspace violations. Episodes end as timeouts when the robot leaves the overall map or its assigned tile (0.55 m buffer); `illegal_contact` was disabled in this historical run. The path-aware curriculum promotes after 90% path progress and demotes below 50%, with at least 1 m target travel and a 50% lateral corridor.

Twenty-five percent of environments are persistent B2-only payload proxies. Isaac Lab cannot mix articulation topologies in one vectorized scene, so these still use the Piper articulation but scale the arm mass and inertia to 5-10% and park it at the stow pose. The proxy assignment is stratified by terrain type and does not change actor or critic dimensions.

## Arm motion

Non-proxy Piper arms follow smooth random waypoints rather than one small repeated sinusoid:

- segment durations 1.75-4.5 s;
- 18% pause probability for 0.5-1.8 s;
- 8% fully stationary probability;
- wide per-joint waypoint offsets, with gripper joints held fixed;
- joint-specific speed limits of 0.65-0.9 rad/s for the main arm joints.

The arm has its own actuator, mass, friction and armature randomization. Its motion is not rewarded directly and leg energy/posture costs explicitly select only the 12 leg joints.

## Reward contract

Positive terms:

| Term | Weight | What it rewards |
|---|---:|---|
| Heading-aligned forward tracking | +3.0 | Gaussian tracking of yaw-frame forward speed with zero lateral speed, multiplied by the heading-alignment gate (`std=0.5`). |
| World yaw-rate tracking | +1.5 | Gaussian match to the commanded yaw rate (`std=0.5`). |
| Upright | +3.0 | Base z-axis aligned with world up. |
| Feet contact without command | +0.1 | Stable foot contact during hold commands. |

Penalties:

| Term | Weight | What it discourages |
|---|---:|---|
| Vertical base velocity squared | -2.0 | Bouncing. |
| Roll/pitch angular velocity squared | -0.05 | Body rocking. |
| Yaw-frame lateral velocity squared | -0.5 | Sideways drift. |
| Planar motion while misaligned | -0.5 | Translating before acquiring the target heading. |
| Leg torque squared | -1e-5 | High effort. |
| Leg acceleration squared | -1e-7 | Jerky joint motion. |
| Leg position-limit violation | -5.0 | Unsafe joint excursions. |
| Absolute leg mechanical power | -1e-5 | Energy use. |
| Leg deviation during hold | -2.0 | Moving instead of standing still. |
| Leg pose deviation | -1.0 | Excessive posture deviation, especially at low command/body speed. |
| Diagonal leg joint mismatch | -0.05 | Asymmetric posture. |
| Action-rate squared | -0.01 | Rapid target changes. |
| Non-foot contact | -1.0 | Body, arm or leg collisions with terrain. |
| Foot force above 100 N | -1.5e-4 | Excessive impacts. |
| Foot height/body-frame shaping | -5.0 | Poor swing-foot clearance behavior. |

The complete serialized truth is `reference/model_19998_run/params/env.yaml`; the table is a readable summary, not a replacement for that file.

## Domain randomization

The principal ranges in the producer run were:

| Quantity | Historical range |
|---|---|
| Static / dynamic friction | 0.45-1.25 / 0.35-1.0 |
| Restitution | 0-0.08 |
| Base mass | additive -1 to +3 kg |
| Other body and arm-link masses | 0.85-1.15 times nominal |
| B2-only proxy arm mass/inertia | 0.05-0.10 times the already randomized arm |
| Center of mass | x/y +/-0.03 m, z +/-0.02 m |
| Leg Kp and Kd | independently 0.8-1.2 times nominal; nominal 160/5, so Kp 128-192 and Kd 4-6 |
| Arm Kp and Kd | independently 0.7-1.3 times nominal; nominal 80/4 |
| Joint friction / armature | 0.5-1.5 / 0.8-1.2 times nominal |
| Shared leg effort limit | 0.85-1.10 times nominal |
| Continuous rough-stage force / torque | each axis +/-10 N / +/-3 Nm |
| Velocity push | every 4-8 s; x/y +/-0.25 m/s and yaw +/-0.15 rad/s |
| Rough reset pose | z 0-0.10 m, roll/pitch +/-0.15 rad, yaw +/-3.14 rad |
| Leg reset state | position +/-0.03 rad, velocity +/-0.10 rad/s |

Actor-only observation noise adds bounded white noise and a per-episode bias to angular velocity, projected gravity, leg position and leg velocity. The exact ranges are in `robust_piper_env_cfg.py` and the resolved YAML.

## Reproducibility boundary

This is an exact source-and-artifact snapshot, but a second training run is not promised to produce the same checkpoint bytes. `model_11999.pt` restores actor, critic, action standard deviation, optimizer and iteration. It does **not** restore Isaac environment state or all simulator/RNG state, and GPU/PhysX execution is not generally bit deterministic. The expected claim is the same code, data, hyperparameters and training recipe—not bit-identical stochastic trajectories.

One provenance limit remains: the historical run did not record a dependency lock or hashes for the Git-ignored robot asset tree. This snapshot captures the currently installed Isaac environment and the complete current asset tree (whose files predate the run), then records their hashes. The producing code and checkpoints are independently verified against hashes written by the August 7 pipeline; historical dependency and asset byte identity cannot be proven to the same standard.
