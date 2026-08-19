# B2-Piper frozen-teacher PPO + distillation

This document describes the additive hybrid training path introduced after the
privileged height-scan teacher and the 45-observation student were trained.  It
does not replace or modify the behavior of the previous teacher PPO or pure
RSL-RL distillation variants.

## Objective

Stock RSL-RL distillation already lets the student control the environment and
asks the teacher to label student-visited states.  However, that algorithm does
not compute returns or PPO advantages; it optimizes only the teacher/student
behavior loss.  The hybrid stage keeps the frozen teacher as a regularizer and
also optimizes the student's task reward:

```text
sampled student action -> environment -> rewards/returns -> PPO

student policy mean ---- Huber ---- frozen teacher policy mean
```

The optimized loss is:

```text
L = L_PPO + lambda(t) * Huber(mu_student, mu_teacher)
```

The sampled student action is used for environment interaction and PPO log
probabilities.  The deterministic student and teacher means are used for the
distillation term, so exploration noise is not penalized by imitation.

## Network and observation contract

The height-scan variant has the following actual dimensions:

| Component | Observation groups | Input | Output | Deployment |
|---|---|---:|---:|---|
| Student actor | `policy` | 45 | 12 | Yes |
| Student critic | `critic` + `contact_forces` | 263 | 1 | No |
| Frozen teacher actor | `policy` + `teacher_privileged` + `contact_forces` + `teacher_height_scan` | 263 | 12 | No |

The height scan contributes 187 values, so the height-scan teacher actor is
263-input, not 76-input.  A legacy teacher without the height scan remains
76-input and is intentionally rejected by the strict hybrid checkpoint loader.

The deployed actor is unchanged:

```text
45 -> 512 -> 256 -> 128 -> 12
```

The critic and teacher are simulation-only.  The final inference policy still
requires only the original 45 observations.

## Checkpoint initialization

A fresh hybrid run requires two checkpoints:

1. A distilled student checkpoint such as `model_1999.pt`.
2. The matching privileged height-scan teacher checkpoint such as
   `model_5999.pt`.

Initialization is deliberately strict:

- `student.*` weights initialize the new PPO actor;
- the student's `std`/`log_std` initializes PPO exploration noise;
- the teacher's `actor.*` weights initialize a separate frozen deterministic
  teacher;
- the teacher's `critic.*` weights warm-start the new student PPO critic;
- the student critic remains trainable from the first PPO update;
- the PPO optimizer and hybrid iteration counter start fresh.

The warm-started critic estimates returns under student-generated trajectories,
so it is not treated as a frozen teacher value function.

Each run writes `params/hybrid_initialization.yaml` containing paths, SHA-256
hashes, source iterations, input dimensions, and initialization behavior.

Hybrid checkpoints contain:

- `model_state_dict`: the normal 45-input ActorCritic student policy and critic;
- `teacher_state_dict`: the frozen teacher needed only to resume training;
- `hybrid_state_dict`: distillation schedule progress;
- optimizer and iteration state.

Export and deployment continue to use the actor inside `model_state_dict`.

## Default optimization settings

The default decaying variant uses:

| Parameter | Value |
|---|---:|
| PPO learning rate | `1e-4`, adaptive |
| PPO epochs / minibatches | `5 / 4` |
| PPO clip | `0.2` |
| Entropy coefficient | `0.002` |
| Initial distillation coefficient | `0.5` |
| Final distillation coefficient | `0.1` |
| Linear decay | `1500` PPO iterations |
| Distillation loss | Huber |

The nonzero final coefficient is intentional: the already-distilled student is
allowed to improve with PPO while retaining a weak teacher anchor.

## Files added and changed

New files:

- `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/hybrid_distillation_ppo.py`
  implements the frozen teacher, checkpoint initialization, hybrid PPO update,
  and resumable runner.
- `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/agents/rsl_rl_hybrid_distillation_cfg.py`
  defines decaying, fixed-coefficient, and PPO-only configurations.
- `scripts/rsl_rl/train_b2_piper_student_hybrid_ppo_heightscan.py` is the
  standalone single/multi-GPU launcher.
- `tests/test_hybrid_distillation_ppo.py` tests gradient isolation, critic
  training, dimensions, and checkpoint mapping without starting Isaac Sim.

Additive changes:

- the existing height-scan Gym registration exposes three additional agent
  entry points;
- `scripts/rsl_rl/train.py` recognizes the hybrid runner and initialization
  arguments.
- `scripts/rsl_rl/play.py` and `scripts/rsl_rl/evaluate_student_checkpoint.py`
  recognize resumable hybrid checkpoints while exporting only the deployable
  student actor.

The previous teacher, pure-distillation, and height-scan pipeline files remain
available and keep their previous behavior.

## Agent variants and ablation design

The existing pure-distillation student is the first baseline.  Three additional
agent entry points are registered on the same height-scan task:

| Variant | Agent entry point | Experiment directory |
|---|---|---|
| Decaying hybrid | `rsl_rl_hybrid_cfg_entry_point` | `unitree_b2_piper_student_hybrid_ppo_heightscan` |
| Fixed hybrid | `rsl_rl_hybrid_fixed_cfg_entry_point` | `unitree_b2_piper_student_hybrid_ppo_fixed_heightscan` |
| PPO-only | `rsl_rl_student_ppo_cfg_entry_point` | `unitree_b2_piper_student_ppo_only_heightscan` |

The PPO-only variant still loads the same teacher checkpoint so its critic has
the same initialization and teacher/student action error can be logged, but its
distillation coefficient is exactly zero and therefore teacher actions do not
affect gradients.

## Local smoke command

Set the paths to the downloaded height-scan student and teacher:

```bash
cd /home/steven/Projects/ATEC/atec_locomotion_checkpoint

conda run --no-capture-output -n isaaclab51_clean \
  python scripts/rsl_rl/train_b2_piper_student_hybrid_ppo_heightscan.py \
  --student-checkpoint /absolute/path/to/student/model_1999.pt \
  --teacher-checkpoint downloads/teacher_heightscan_explicit_6000/model_5999.pt \
  --variant decay \
  --num-envs 16 \
  --iterations 5
```

Use `--dry-run` to inspect the generated command without launching Isaac Sim or
requiring the checkpoint paths to exist.

## Three-GPU server command

The server checkpoints used in the completed runs were:

```text
/data/steven/atec_locomotion_checkpoint/logs/rsl_rl/unitree_b2_piper_student_distillation_heightscan/2026-08-17_10-14-48_ts7g_student_v2/model_1999.pt
/data/steven/atec_locomotion_checkpoint/logs/rsl_rl/unitree_b2_piper_privileged_teacher_heightscan/2026-08-16_06-12-17_ts7g_teacher_v2/model_5999.pt
```

From `/data/steven/atec_locomotion_checkpoint`:

```bash
export CUDA_VISIBLE_DEVICES=1,2,4

/data/steven/run_atec_env.sh env \
  /data/steven/conda_envs/isaaclab/bin/python \
  scripts/rsl_rl/train_b2_piper_student_hybrid_ppo_heightscan.py \
  --python /data/steven/conda_envs/isaaclab/bin/python \
  --student-checkpoint /data/steven/atec_locomotion_checkpoint/logs/rsl_rl/unitree_b2_piper_student_distillation_heightscan/2026-08-17_10-14-48_ts7g_student_v2/model_1999.pt \
  --teacher-checkpoint /data/steven/atec_locomotion_checkpoint/logs/rsl_rl/unitree_b2_piper_privileged_teacher_heightscan/2026-08-16_06-12-17_ts7g_teacher_v2/model_5999.pt \
  --variant decay \
  --num-gpus 3 \
  --num-envs 256 \
  --iterations 2000 \
  --seed 42
```

`--num-envs` is per GPU/process.  The command above therefore rolls out 768
environments in parallel.

For the fixed and PPO-only ablations, change only:

```bash
--variant fixed
```

or:

```bash
--variant ppo-only
```

Coefficient overrides are also available:

```bash
--distillation-coef-start 0.4 \
--distillation-coef-end 0.05 \
--distillation-decay-iterations 1500
```

## Resume a hybrid checkpoint

Hybrid checkpoints save the frozen teacher and schedule progress.  Resume via
the underlying training command and do not pass the two initialization paths:

```bash
python scripts/rsl_rl/train.py \
  --task ATEC-Isaac-Velocity-Robust-Heading-Rough-Unitree-B2-Piper-TeacherStudent-HeightScan-v0 \
  --agent rsl_rl_hybrid_cfg_entry_point \
  --resume \
  --load_run '<existing_run_directory>' \
  --checkpoint model_900.pt \
  --max_iterations 1100 \
  --headless
```

For this repository's resume logic, `--max_iterations` is the number of
additional iterations.  The resumed optimizer learning rate and hybrid
distillation schedule are restored and checked.

## Visualize or evaluate a hybrid checkpoint

The existing play script accepts the hybrid agent configuration:

```bash
python scripts/rsl_rl/play.py \
  --task ATEC-Isaac-Velocity-Robust-Heading-Rough-Unitree-B2-Piper-TeacherStudent-HeightScan-v0 \
  --agent rsl_rl_hybrid_cfg_entry_point \
  --checkpoint /absolute/path/to/hybrid/model_1999.pt \
  --num_envs 1 \
  --real-time
```

Use the same agent entry point that produced the checkpoint (`hybrid`,
`hybrid_fixed`, or `student_ppo`).  The frozen teacher is restored only because
it is part of the resumable training checkpoint; inference calls the 45-input
student actor.

## TensorBoard signals

In addition to the existing PPO and environment metrics, the hybrid runner
logs:

- `Loss/distillation`: Huber/MSE teacher regularization before multiplying by
  the coefficient;
- `Loss/distillation_coefficient`: the active schedule value;
- `Loss/teacher_student_action_rmse`: deterministic mean-action disagreement;
- `Loss/value_function`, `Loss/surrogate`, `Loss/entropy`;
- `Train/mean_reward` and existing terrain, velocity, termination, and arm
  metrics.

Compare all variants on fixed evaluation terrain distributions.  Curriculum
level alone is not sufficient because different policies may advance through
terrain difficulty at different rates.

## Pure-Python verification

The algorithm and checkpoint mapping tests do not require Isaac Sim:

```bash
cd /home/steven/Projects/ATEC/atec_locomotion_checkpoint
PYTHONPATH=source/atec_rl_lab \
  /home/steven/miniconda3/envs/isaaclab51_clean/bin/python -m pytest -q \
  tests/test_hybrid_distillation_ppo.py
```

The implementation targets the installed RSL-RL 3.1.2 API used by this local
Isaac Lab environment.
