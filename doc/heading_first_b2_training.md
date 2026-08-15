# Heading-first B2 rough-terrain training

This repository registers an isolated task for teaching Unitree B2 to turn toward a requested world direction before walking forward:

```text
ATEC-Isaac-Velocity-Heading-Rough-Unitree-B2-v0
```

The original flat and rough task registrations are unchanged. The new task still uses Isaac Lab's official `ROUGH_TERRAINS_CFG` families: pyramid stairs, inverted pyramid stairs, boxes, random rough ground, pyramid slope, and inverted pyramid slope.

## Command and reward

The original rough task samples planar velocity in the robot's body frame. A nonzero `lin_vel_y` therefore asks for lateral walking, while its absolute heading target is independent. That objective makes crabbing a valid optimum.

The heading-first task instead samples a nominal forward speed `s` in `[0.25, 1.0] m/s` and a target world heading. For wrapped heading error `e`, it publishes the same three-dimensional policy command as before:

```text
gate = exp(-(e / 0.55)^2)
command = [s * gate, 0, clip(e, -1, 1)]
```

Thus the actor input and action dimensions remain 45 and 12, respectively. An existing B2 rough actor is directly compatible.

The active behavior-specific terms are:

- heading-gated forward tracking: weight `+3.0`;
- world-yaw-rate tracking: weight `+1.5`;
- lateral yaw-frame velocity: weight `-0.5`;
- planar translation while misaligned: weight `-0.5`.

There is deliberately no standalone positive heading bonus. Such a bonus can be collected by merely standing at the desired angle. Instead, yaw tracking teaches the turn, and alignment unlocks the forward reward. Terrain curriculum demotion uses the accumulated ungated nominal path length so never turning cannot make the expected travel distance vanish.

Each sampled heading lasts for the complete 20-second episode. This keeps final displacement consistent with the commanded path instead of allowing two independently sampled headings to cancel each other. Advancing to a harder terrain row also requires a mean alignment gate of at least `0.6`.

## Warm-start training

Use actor-only transfer when changing to this reward. It preserves the learned gait while deliberately starting a fresh critic, optimizer, exploration parameter, and iteration count for the new objective.

```bash
python scripts/rsl_rl/train.py \
  --task ATEC-Isaac-Velocity-Heading-Rough-Unitree-B2-v0 \
  --num_envs 4096 \
  --pretrained_actor /absolute/path/to/model_N.pt \
  --video --video_terrain_cycle \
  --video_interval 4000 --video_length 300 \
  --spawn_audit --headless \
  --run_name heading_first_from_rough
```

The task's fine-tuning configuration runs for 5,000 iterations with learning rate `3e-4` and entropy coefficient `0.005`. Success should be judged using heading error, lateral velocity, alignment gate, forward/yaw tracking, and per-terrain evaluation rather than total reward alone.

To continue an interrupted heading-first run without discarding its critic,
optimizer, exploration noise, or adaptive learning rate, use a full resume:

```bash
python scripts/rsl_rl/train.py \
  --task ATEC-Isaac-Velocity-Heading-Rough-Unitree-B2-v0 \
  --num_envs 4096 \
  --resume \
  --load_run 2026-08-05_15-02-18_b2_heading_first_from_rough18700_4096_all_terrain_video \
  --checkpoint model_100.pt \
  --max_iterations 4900 \
  --video --video_terrain_cycle \
  --video_interval 4000 --video_length 300 \
  --spawn_audit --headless \
  --run_name heading_resume100
```

For RSL-RL, `--max_iterations` is the number of *additional* iterations after
loading. Resuming iteration 100 for another 4,900 iterations therefore finishes
at approximately iteration 4,999.

## All-terrain evaluation videos

`--terrain_showcase` creates one deterministic-difficulty terrain row with ten columns, realizing the official `2/2/2/2/1/1` terrain proportions. It follows one representative robot per family and saves six separate videos plus `terrain_showcase.yaml`.

```bash
python scripts/rsl_rl/play.py \
  --task ATEC-Isaac-Velocity-Heading-Rough-Unitree-B2-v0 \
  --checkpoint /absolute/path/to/model_N.pt \
  --terrain_showcase \
  --showcase_difficulty 0.75 \
  --showcase_steps_per_terrain 250 \
  --headless
```

Training videos also rotate through these six terrain families when `--video_terrain_cycle` is enabled. At a 4,000-step interval, one full six-family cycle spans 24,000 policy steps (about 1,000 PPO iterations with 24 rollout steps per iteration). The `params/terrain_camera.yaml` manifest records which terrain each successive video represents.
