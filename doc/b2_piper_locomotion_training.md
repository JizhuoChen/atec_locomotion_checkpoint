# B2-Piper locomotion training

For the optional privileged-teacher PPO and 45-input student-distillation
extension of the canonical policy, see
[`b2_piper_teacher_student_training.md`](b2_piper_teacher_student_training.md).

Run these commands from the repository root in the Isaac Lab environment. The implemented workflow first adapts a bare-B2 gait to the mounted Piper arm on a plane, then transfers that actor to the heading-first official rough terrains.

## Tasks and policy contract

| Stage | Task | Purpose | Actor / critic | Experiment log |
| --- | --- | --- | --- | --- |
| 1 | `ATEC-Isaac-Velocity-Flat-Unitree-B2-Piper-v0` | Flat-ground embodiment adaptation with the original body-frame velocity command | 45 / 64 observations | `unitree_b2_piper_flat` |
| 2 | `ATEC-Isaac-Velocity-Heading-Rough-Unitree-B2-Piper-v0` | Turn toward a world-heading target, then walk forward over the six official rough-terrain families | 45 / 251 observations | `unitree_b2_piper_heading_rough` |

Both actors receive the same 45 values: base angular velocity (3), projected gravity (3), velocity command (3), 12 leg positions, 12 leg velocities, and the previous 12 leg actions. They output targets for only the 12 leg joints. Arm state and height scans therefore cannot become deployment-time actor dependencies, and a bare-B2 actor is exactly shape-compatible.

Training is asymmetric. The flat critic receives 64 values: base linear velocity plus the actor-side state, with all 20 leg-and-arm joint positions and velocities replacing the actor's leg-only joint state. The rough critic adds the 187-value height scan for 251 values. Giving the critic the actual arm state lets it explain the time-varying payload disturbance while the deployable actor remains leg-only.

The Piper starts in a bent, half-open stow pose. A separate controller gives its first six joints small, slow sinusoidal position-and-velocity targets (approximately 0.05--0.12 Hz); 20% of episodes use a randomized static pose, and the two gripper joints remain half open. This produces physical inertial reactions without asking PPO to control the arm. Joint-cost reward terms are restricted to the legs, while arm contacts and their effects on the base remain physical.

## Automatic two-stage run

The default launch is:

```bash
python scripts/rsl_rl/train_b2_piper_pipeline.py
```

Defaults are 2,048 environments and 5,000 PPO iterations per stage. Training is headless and does **not** enable cameras inside the large training scene. The rough stage audits every initial root and foot position against its assigned terrain tile.

This separation is required for the supplied Piper USD on the current 64 GiB machine. Flat and rough 2,048-environment probes both completed without rendering, while enabling inline video at 2,048 environments caused the kernel OOM killer to terminate Isaac Sim at roughly 53 GiB resident memory. The arm contains large non-instanceable visual meshes, so camera activation makes RTX ingest every cloned copy. Record videos afterwards with `play.py`, which uses only 1 flat environment, 10 baseline showcase environments, or 16 robust showcase environments.

### Robust from-scratch profile

The original command remains the default baseline. To train the expanded sim-to-real profile from scratch, use:

```bash
python scripts/rsl_rl/train_b2_piper_pipeline.py --profile robust
```

The robust defaults are 8,000 flat iterations followed by 12,000 heading-rough iterations. The flat stage deliberately has no old actor seed; after it completes, the launcher validates the final 45-input/12-output checkpoint and copies only its actor into the rough stage. The rough critic, optimizer, exploration state, and iteration counter are new.

| Stage | Robust task | Experiment log |
| --- | --- | --- |
| 1 | `ATEC-Isaac-Velocity-Robust-Flat-Unitree-B2-Piper-v0` | `unitree_b2_piper_robust_flat` |
| 2 | `ATEC-Isaac-Velocity-Robust-Heading-Rough-Unitree-B2-Piper-v0` | `unitree_b2_piper_robust_heading_rough` |

This profile keeps the same deployable policy dimensions while adding controller delay and target error, episode sensor bias, conservative physics variation, varied smooth Piper waypoints, fine-grained stair geometry, signed multi-scale roughness, and assigned-tile containment. The exact resolved environment and runner configurations are saved under each run's `params/`; the pipeline manifest also records `profile: robust`, selected tasks/experiments, iteration overrides, seed mode, and transfer semantics.

The current robust terrain generator retains the same 10-by-80 curriculum and family proportions, but no longer gives every stair family one fixed tread depth. Stair riser height varies continuously with difficulty from 4 to 26 cm. Tread depth is independently hash-sampled at 5 mm resolution inside short (22--28 cm), nominal (28--35 cm), and long (35--42 cm) bands for both ascent and descent. This produces many riser/tread combinations without increasing the terrain-map memory footprint. Other maximum difficulties are also modestly higher: signed roughness reaches 4.0, 7.5, and 11.5 cm; random boxes reach 22 cm; and slopes reach gradient 0.45 (about 24.2 degrees).

The robust tasks also include a continuation-compatible B2-only payload proxy. Exactly 25% of environments are selected once at startup, stratified across terrain columns, and remain in that mode across resets. Their Piper is held at the compact stow with zero target velocity while all ten arm bodies have mass and inertia scaled to 5--10% (about 0.23--0.47 kg residual versus 4.67 kg nominal). The other 75% retain 85--115% Piper mass randomization and diverse moving waypoints. This is not a literal bare-B2 USD: the visual arm and colliders remain because one vectorized articulation cannot mix 12- and 20-joint topologies. It is the closest full-state-resume-safe approximation and exercises both low-payload and moving-payload dynamics in every terrain family.

Neither the proxy mode nor arm state is added to the actor. The deployable network remains 45-to-12. The critic remains 251 inputs and continues to observe actual arm positions and velocities, so the saved critic and optimizer are shape-compatible. TensorBoard reports the fixed assignment as `Metrics/arm_motion/b2_only_proxy_fraction`, which should remain 0.25 even when an episode terminates early. It also reports `Curriculum/terrain_level_<family>` for all twelve terrain families, which should be checked alongside the aggregate terrain level to detect underexposure to hard stairs.

The terrain stage uses 8-second episodes rather than the baseline 20 seconds. Every heading points approximately through the assigned 8-by-8 m tile interior, with up to 0.35 rad (about 20 degrees) of variation. Independent spawn positions and initial robot yaw still expose the policy to the full range of relative turning commands. A 0.55 m inset ends a rollout before the B2 footprint enters a neighboring tile, while the original map-wide boundary remains as a safety net. Terrain promotion uses signed progress along the commanded heading, caps its target at the feasible in-tile path, rejects excessive lateral drift, and treats hold commands and paths shorter than 1 m as curriculum-neutral.

This addresses an ambiguity in the completed baseline terrain run. That run did not save root trajectories, so its exact historical exposure cannot be reconstructed. A geometry audit indicates that its random 20-second commands likely spent only about 47--54% of their time in the originally assigned tile. Most departures would enter another generated terrain tile, not the outer flat plane, but that still mixes terrain family and difficulty. A conservative straight-line audit estimates that the robust 8-second/inward-heading geometry retains about 98.8% of nominal trajectory time in the assigned tile; the boundary term prevents any later outside samples from contributing reward.

The robust rough task logs the actual measured ratios in TensorBoard:

- `Metrics/base_velocity/assigned_tile_fraction`: fraction of each completed trajectory inside its original buffered tile; this should remain close to 1.
- `Metrics/base_velocity/ever_exited_assigned_tile`: fraction of completed episodes that reached the tile boundary; lower is better, but interpret it together with assigned-tile fraction and commanded speed because a brief late boundary contact can mark the whole episode.
- `Metrics/base_velocity/nonflat_trajectory_fraction`: fraction of trajectory samples whose local height scan has at least 2.5 cm of relief, irrespective of tile.
- `Metrics/base_velocity/nonflat_assigned_fraction`: non-flat fraction among samples still inside the assigned tile.
- `Metrics/base_velocity/mean_local_relief_assigned`: mean local height range, in metres, while inside the assigned tile.

Ignore the first few logged points when judging these ratios: RSL-RL randomizes initial episode ages, so startup resets do not represent complete fresh rollouts.

The non-flat ratios need not equal 1 because stairs, slopes, boxes, and spawn patches intentionally contain local platforms. Read them together with the assigned-tile ratio: a falling assigned fraction indicates leakage, whereas a high assigned fraction with a low non-flat fraction indicates too much time on locally flat portions of the intended tile.

### Continue the completed robust policy on the expanded distribution

Use the dedicated launcher to resume the actor, critic, exploration standard deviation, optimizer moments, iteration counter, and adaptive learning rate from the completed robust checkpoint:

```bash
python scripts/rsl_rl/continue_b2_piper_rough.py \
  --checkpoint "$PWD/logs/rsl_rl/unitree_b2_piper_robust_heading_rough/2026-08-06_23-17-23_b2_piper_robust_20260806_215402_69aaacaf_heading_rough/model_11999.pt" \
  --iterations 4000 \
  --num-envs 2048
```

Here `--iterations` means additional PPO updates. RSL-RL resumes at saved iteration 11999, so 4,000 updates produce `model_15998.pt`. Omitting `--checkpoint` selects the latest completed robust pipeline checkpoint. Add `--dry-run` to validate provenance and print the exact command without launching Isaac Sim.

The launcher requires the exact final checkpoint of a completed robust rough run, directly below this repository's `logs/rsl_rl/unitree_b2_piper_robust_heading_rough/<run>/` tree, and validates the 45/251/12 network plus optimizer before starting. It uses anchored run/checkpoint matching, audits all root and foot spawns (including exact proxy count, per-terrain allocation, and mass-scale range), and writes a continuation manifest below `logs/rsl_rl/pipelines/`. If `--seed` is omitted, the source checkpoint's saved seed is inherited and passed explicitly. The new run's `params/resume.yaml` records the source path and SHA-256, source/final iterations, restored learning rate, and completion status. PPO state is restored; simulator episode positions and curriculum levels are deliberately initialized fresh on the expanded terrain distribution.

Inspect the robust commands without starting Isaac Sim:

```bash
python scripts/rsl_rl/train_b2_piper_pipeline.py --profile robust --dry-run
```

The robust iteration counts can be overridden normally. Supplying `--flat-seed PATH` is also supported for an intentional experiment, but omitting it is the recommended from-scratch run:

```bash
python scripts/rsl_rl/train_b2_piper_pipeline.py --profile robust \
  --flat-iterations 10000 --rough-iterations 15000
```

When restarting only stage 2, the supplied checkpoint must be the exact final checkpoint of a completed `unitree_b2_piper_robust_flat` run:

```bash
python scripts/rsl_rl/train_b2_piper_pipeline.py --profile robust \
  --skip-flat-checkpoint /absolute/path/to/model_7999.pt
```

For playback or showcase recording of the final robust checkpoint, use the robust heading-rough task ID from the table rather than the baseline task ID.

For the baseline profile, stage 1 uses actor-only warm-start. By default, `--flat-seed auto-flat` considers only exact final checkpoints (`model_{max_iterations-1}.pt`) from runs with saved agent metadata, then selects the highest iteration below `logs/rsl_rl/unitree_b2_flat/`; at present this resolves to:

```text
logs/rsl_rl/unitree_b2_flat/2026-08-04_14-18-01_b2_flat_baseline_video/model_4999.pt
```

If no bare-B2 flat checkpoint exists, stage 1 starts from scratch. Use `--flat-seed none` to request that explicitly, or `--flat-seed /absolute/path/to/model_N.pt` to select a checkpoint. Actor-only transfer intentionally starts a fresh critic, optimizer, exploration noise, and iteration counter. After stage 1 exits successfully, the launcher validates its final `model_4999.pt` and transfers only that actor into a fresh stage-2 run.

The completed bare-B2 heading-first actor is also directly compatible with the Piper heading-rough task. This is the fastest way to test the new embodiment, but it skips the requested flat embodiment-adaptation stage:

```bash
BARE_HEADING_CHECKPOINT="$PWD/logs/rsl_rl/unitree_b2_heading_rough/2026-08-05_15-26-43_b2_heading_resume100_curriculumfix_4096_allterrain/model_4999.pt"
python scripts/rsl_rl/train.py \
  --task ATEC-Isaac-Velocity-Heading-Rough-Unitree-B2-Piper-v0 \
  --pretrained_actor "$BARE_HEADING_CHECKPOINT" \
  --num_envs 2048 --max_iterations 5000 \
  --spawn_audit --headless \
  --run_name b2_piper_direct_heading_transfer
```

This path was smoke-tested successfully. It still creates a fresh Piper-aware critic and optimizer; only the 45-to-12 actor weights are copied.

Inspect the complete commands without starting Isaac Sim:

```bash
python scripts/rsl_rl/train_b2_piper_pipeline.py --dry-run
```

To start stage 2 again from an already completed stage-1 checkpoint:

```bash
B2P_FLAT_CHECKPOINT=/absolute/path/to/logs/rsl_rl/unitree_b2_piper_flat/RUN/model_4999.pt
python scripts/rsl_rl/train_b2_piper_pipeline.py \
  --skip-flat-checkpoint "$B2P_FLAT_CHECKPOINT"
```

This is a new stage-2 fine-tune with actor-only transfer. To continue a *partially completed* stage-2 run while preserving its critic, optimizer, noise, and adaptive learning rate, resume it directly; `--max_iterations` is the number of additional iterations:

```bash
python scripts/rsl_rl/train.py \
  --task ATEC-Isaac-Velocity-Heading-Rough-Unitree-B2-Piper-v0 \
  --num_envs 2048 \
  --resume \
  --load_run RUN_DIRECTORY_NAME \
  --checkpoint model_N.pt \
  --max_iterations ADDITIONAL_ITERATIONS \
  --spawn_audit --headless \
  --run_name b2_piper_heading_rough_resume
```

Do not enable `--video` or `--no-headless` for the 2,048-environment run on this machine. The launcher blocks rendering above 64 environments unless `--allow-high-env-rendering` is also supplied. That override is intended only for machines whose host-memory behavior has already been validated.

## Play and all-terrain visualization

Use a stage-appropriate checkpoint: a flat checkpoint with the flat task, or a rough checkpoint with the rough task. For interactive rough-terrain play:

```bash
B2P_ROUGH_CHECKPOINT=/absolute/path/to/logs/rsl_rl/unitree_b2_piper_heading_rough/RUN/model_4999.pt
python scripts/rsl_rl/play.py \
  --task ATEC-Isaac-Velocity-Heading-Rough-Unitree-B2-Piper-v0 \
  --checkpoint "$B2P_ROUGH_CHECKPOINT" \
  --num_envs 16
```

For a flat stage-1 checkpoint, change the task to `ATEC-Isaac-Velocity-Flat-Unitree-B2-Piper-v0` and point the variable at that flat checkpoint.

A normal viewport follows one environment. Showcase mode derives the smallest exact curriculum-column allocation from the selected task's normalized sub-terrain proportions, creates one environment per column, follows one representative robot from every configured family, and writes one video per family. For the baseline task this is ten columns with the official `2/2/2/2/1/1` allocation and six clips. For the robust task it is sixteen columns covering all twelve variants. The default 1,000 policy steps give each family a 20-second clip; this is one baseline episode or 2.5 robust 8-second episode horizons, so a robust clip can contain resets:

```bash
python scripts/rsl_rl/play.py \
  --task ATEC-Isaac-Velocity-Heading-Rough-Unitree-B2-Piper-v0 \
  --checkpoint "$B2P_ROUGH_CHECKPOINT" \
  --terrain_showcase \
  --showcase_difficulty 0.75 \
  --showcase_steps_per_terrain 1000 \
  --headless
```

Remove `--headless` to watch the same camera cycle in the simulator. Showcase output is non-overwriting and uses a difficulty/seed-specific folder below `videos/play`; repeating the same single-showcase command requires a different `--showcase_output_name NAME`. Showcase mode forces moving commands so every terrain panel exercises locomotion; ordinary play retains the training task's 2% zero-command hold trials.

The recommended baseline completed-policy review records the six families at four difficulty samples. With no arguments it retains the baseline behavior: it selects the latest exact final B2-Piper rough checkpoint automatically, uses a distinct procedural seed at each difficulty, writes 24 labeled individual clips of about 20 seconds, and creates four synchronized 3-by-2 all-terrain montage videos:

```bash
python scripts/rsl_rl/record_b2_piper_showcase_suite.py
```

After robust training completes, review its twelve variants with the matching task, experiment, and checkpoint validation. This writes 48 individual clips and four synchronized 4-by-3 montages at the default four difficulties:

```bash
python scripts/rsl_rl/record_b2_piper_showcase_suite.py --profile robust
```

To review a specific checkpoint, add `--checkpoint /absolute/path/to/model_N.pt`; it must be the exact final checkpoint from the selected profile's experiment. Supplying `--seeds 42 43` evaluates every requested difficulty at both seeds. The suite manifest records the profile, task, experiment, checkpoint, ordered family labels, command lines, difficulties, seeds, montage layout, output paths, sizes, and probed durations. Procedural terrain has infinitely many realizations, so this matrix covers every configured family and several representative difficulty/seed cases rather than claiming exhaustive geometry coverage.

## Outputs and recovery records

Each launcher invocation writes:

```text
logs/rsl_rl/pipelines/<pipeline-id>/pipeline.json
```

The manifest records the exact commands, stage status, actor source, run directories, final checkpoint, actor dimensions, and checkpoint SHA-256. Stage artifacts are under:

```text
logs/rsl_rl/unitree_b2_piper_flat/<run>/
logs/rsl_rl/unitree_b2_piper_heading_rough/<run>/
logs/rsl_rl/unitree_b2_piper_robust_flat/<run>/
logs/rsl_rl/unitree_b2_piper_robust_heading_rough/<run>/
```

Important files are `model_N.pt`, the TensorBoard event file, `params/agent.yaml`, `params/env.yaml`, `params/pretrained_actor.yaml`, and, on rough terrain, `params/spawn_audit.yaml`. If low-environment inline video is explicitly enabled, training clips are in `videos/train/`. Each showcase is placed below `videos/play/<showcase-name>/` with its own `terrain_showcase.yaml`; multi-difficulty suites additionally have a parent `suite.json` and labeled montage files. Running `play.py` also exports `exported/policy.pt` and `exported/policy.onnx` beside the checkpoint.

## Current bare-B2 convergence reference

The completed bare-B2 heading-first reference is:

```text
logs/rsl_rl/unitree_b2_heading_rough/2026-08-05_15-26-43_b2_heading_resume100_curriculumfix_4096_allterrain/model_4999.pt
```

Its mean reward over successive 250-iteration windows from iteration 4,000 through 4,999 was `273.96`, `273.95`, `274.33`, and `274.03`, so the aggregate objective had plateaued. Over the final 100 logged iterations, mean episode length was the full `1000/1000` steps, timeout fraction was `99.998%`, mean terrain level was `5.96`, mean alignment gate was `0.921`, absolute heading error was `0.139 rad` (about 8 degrees), lateral speed was `0.107 m/s`, planar velocity error was `0.207 m/s`, and yaw-rate error was `0.191 rad/s`.

That is strong aggregate evidence that the *bare-B2* run converged to a stable, survivable heading-first policy, but it is not proof of equal performance on every terrain family. A full B2-Piper two-stage pipeline has since completed at `logs/rsl_rl/pipelines/b2_piper_auto_20260806_023308_e35b4719/pipeline.json`, with its final rough checkpoint under `logs/rsl_rl/unitree_b2_piper_heading_rough/2026-08-06_03-47-07_b2_piper_auto_20260806_023308_e35b4719_heading_rough/model_4999.pt`. Use its TensorBoard curves together with the multi-difficulty suite rather than aggregate reward alone to judge embodiment adaptation and terrain-wide behavior.
