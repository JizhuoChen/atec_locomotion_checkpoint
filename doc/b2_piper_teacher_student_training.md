# B2-Piper privileged teacher and student distillation

This extension appends a privileged-teacher PPO stage and a student-distillation
stage to the canonical `model_19998.pt` lineage.  It does not change the original
task registration or deployment interface.

## Observation and network contract

| Network | Observation groups | Input | Output |
| --- | --- | ---: | ---: |
| Student | `policy` | 45 | 12 leg actions |
| Teacher actor | `policy` + `teacher_privileged` + `contact_forces` | 76 | 12 leg actions |
| Teacher critic | `critic` + `contact_forces` | 263 | 1 value |

Height-map-aware alternative:

| Network | Observation groups | Input | Output |
| --- | --- | ---: | ---: |
| Student | `policy` | 45 | 12 leg actions |
| Teacher actor | `policy` + `teacher_privileged` + `contact_forces` + `teacher_height_scan` | 263 | 12 leg actions |
| Teacher critic | `critic` + `contact_forces` | 263 | 1 value |

`teacher_privileged` contains clean ground-truth base linear velocity (3), Piper
joint positions (8), and Piper joint velocities (8). `contact_forces` contains
the four foot net-normal force vectors rotated into the robot base frame (12),
clipped to +/-500 N per component and scaled by 0.01.

The student remains the original `45 -> 512 -> 256 -> 128 -> 12` ELU MLP.

## Checkpoint initialization

The privileged PPO stage starts from
`reference/model_19998_run/model_19998.pt`:

- actor first-layer columns 0:45 are copied; columns 45:76 are exactly zero;
- critic first-layer columns 0:251 are copied; columns 251:263 are exactly zero;
- all later actor and critic layers, biases, and the action standard deviation are copied;
- optimizer and iteration counter start fresh.

Thus the expanded networks initially implement the old actor and critic exactly;
the new inputs affect their outputs only after PPO learns non-zero weights.

The distillation stage loads the completed privileged PPO actor as a frozen
teacher and separately copies the canonical 45-input actor into the student.
Its optimizer is fresh and its rollout noise standard deviation is 0.1.

Each run writes initialization provenance and checkpoint hashes below `params/`.

## Automatic two-stage run

From this repository root in the Isaac Lab environment:

```bash
conda run --no-capture-output -n isaaclab \
  python scripts/rsl_rl/train_b2_piper_teacher_student.py
```

Defaults are 2,048 environments, 6,000 privileged PPO iterations, 2,000
distillation iterations, and seed 42. A small command-shape check can be printed
without starting Isaac Sim:

```bash
conda run -n isaaclab \
  python scripts/rsl_rl/train_b2_piper_teacher_student.py \
  --dry-run --num-envs 16 --teacher-iterations 2 --student-iterations 2
```

Useful overrides include:

```bash
python scripts/rsl_rl/train_b2_piper_teacher_student.py \
  --base-checkpoint /absolute/path/to/model_19998.pt \
  --num-envs 2048 \
  --teacher-iterations 6000 \
  --student-iterations 2000 \
  --device cuda:0
```

Terrain-aware two-stage run:

```bash
conda run --no-capture-output -n isaaclab \
  python scripts/rsl_rl/train_b2_piper_teacher_student_heightscan.py
```

## Individual stages

Teacher PPO:

```bash
python scripts/rsl_rl/train.py \
  --task ATEC-Isaac-Velocity-Robust-Heading-Rough-Unitree-B2-Piper-TeacherStudent-v0 \
  --agent rsl_rl_teacher_cfg_entry_point \
  --num_envs 2048 \
  --max_iterations 6000 \
  --pretrained_privileged_teacher reference/model_19998_run/model_19998.pt \
  --run_name privileged_teacher \
  --spawn_audit --headless --seed 42
```

Student distillation:

```bash
python scripts/rsl_rl/train.py \
  --task ATEC-Isaac-Velocity-Robust-Heading-Rough-Unitree-B2-Piper-TeacherStudent-v0 \
  --agent rsl_rl_distillation_cfg_entry_point \
  --num_envs 2048 \
  --max_iterations 2000 \
  --teacher_checkpoint /absolute/path/to/teacher/model_5999.pt \
  --pretrained_student reference/model_19998_run/model_19998.pt \
  --run_name distilled_student \
  --spawn_audit --headless --seed 42
```

Manual teacher-student height-map stages use the same `HeightScan` task:

```bash
python scripts/rsl_rl/train.py \
  --task ATEC-Isaac-Velocity-Robust-Heading-Rough-Unitree-B2-Piper-TeacherStudent-HeightScan-v0 \
  --agent rsl_rl_teacher_cfg_entry_point \
  --num_envs 2048 \
  --max_iterations 6000 \
  --pretrained_privileged_teacher reference/model_19998_run/model_19998.pt \
  --run_name privileged_teacher \
  --spawn_audit --headless --seed 42

python scripts/rsl_rl/train.py \
  --task ATEC-Isaac-Velocity-Robust-Heading-Rough-Unitree-B2-Piper-TeacherStudent-HeightScan-v0 \
  --agent rsl_rl_distillation_cfg_entry_point \
  --num_envs 2048 \
  --max_iterations 2000 \
  --teacher_checkpoint /absolute/path/to/teacher/model_5999.pt \
  --pretrained_student reference/model_19998_run/model_19998.pt \
  --run_name distilled_student \
  --spawn_audit --headless --seed 42
```

For playback and student export, use the same task with
`--agent rsl_rl_distillation_cfg_entry_point` and the distilled checkpoint.
`play.py` exports the `student` module as the deployment policy.
