#!/usr/bin/env bash
set -euo pipefail

# Sweep object-1 weighted-pose IK orientation weights while replaying the same
# Task E seed-candidate list for every weight. Override defaults with env vars:
#   OUT_ROOT=datasets/my_sweep SEEDS="430881471 410320436" NUM_DEMOS=2 bash ...

OUT_ROOT="${OUT_ROOT:-datasets/atec_task_e_object1_then3_weight_sweep}"
NUM_DEMOS="${NUM_DEMOS:-2}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-10}"
SEED_STEP="${SEED_STEP:-7919}"
SEEDS_TEXT="${SEEDS:-430881471 410320436}"
WEIGHTS_TEXT="${WEIGHTS:-0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20}"
read -r -a SEEDS <<< "${SEEDS_TEXT}"
read -r -a WEIGHTS <<< "${WEIGHTS_TEXT}"

if ((${#SEEDS[@]} == 0)); then
  echo "[ERROR] Need at least one seed candidate." >&2
  exit 2
fi
if ((MAX_ATTEMPTS < NUM_DEMOS)); then
  echo "[ERROR] MAX_ATTEMPTS=${MAX_ATTEMPTS} must be >= NUM_DEMOS=${NUM_DEMOS}." >&2
  exit 2
fi

ATTEMPT_SEEDS=("${SEEDS[@]}")
seed_cursor=0
while ((${#ATTEMPT_SEEDS[@]} < MAX_ATTEMPTS)); do
  base_seed="${SEEDS[$((seed_cursor % ${#SEEDS[@]}))]}"
  round="$((seed_cursor / ${#SEEDS[@]} + 1))"
  next_seed="$(((base_seed + SEED_STEP * round) % 2147483647))"
  ATTEMPT_SEEDS+=("${next_seed}")
  seed_cursor=$((seed_cursor + 1))
done

mkdir -p "${OUT_ROOT}"
echo "[INFO] Output root: ${OUT_ROOT}"
echo "[INFO] Need ${NUM_DEMOS} demos per weight; allowing ${MAX_ATTEMPTS} attempts."
echo "[INFO] Seed candidates reused for every weight: ${ATTEMPT_SEEDS[*]:0:${MAX_ATTEMPTS}}"

for weight in "${WEIGHTS[@]}"; do
  label="${weight/./p}"
  out_dir="${OUT_ROOT}/weight_${label}"
  mkdir -p "${out_dir}"
  echo
  echo "[INFO] ===== weight=${weight} output=${out_dir} ====="

  conda run --no-capture-output -n isaaclab \
    python scripts/act/collect_demos_task_e.py \
    --pick_objects 1 3 \
    --num_demos "${NUM_DEMOS}" \
    --seeds "${ATTEMPT_SEEDS[@]:0:${MAX_ATTEMPTS}}" \
    --position_priority_orientation_weight "${weight}" \
    --headless --enable_cameras --save_video \
    --output_dir "${out_dir}" \
    --max_attempts "${MAX_ATTEMPTS}" \
    2>&1 | tee "${out_dir}/run.log"
done

summary_csv="${OUT_ROOT}/success_summary.csv"
conda run --no-capture-output -n isaaclab python - "${OUT_ROOT}" "${summary_csv}" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

import h5py
import numpy as np

BASKET_CENTER_X = 1.08
BASKET_CENTER_Y = -0.30
TABLE_TOP_Z = 0.8266426479059951
BASKET_IN_X = 0.20
BASKET_IN_Y = 0.11
BASKET_MAX_Z = TABLE_TOP_Z + 0.10


def weight_from_dir(path: Path) -> float:
    return float(path.name.removeprefix("weight_").replace("p", "."))


def object_success(pos: np.ndarray) -> bool:
    return (
        abs(float(pos[0]) - BASKET_CENTER_X) <= BASKET_IN_X
        and abs(float(pos[1]) - BASKET_CENTER_Y) <= BASKET_IN_Y
        and float(pos[2]) <= BASKET_MAX_Z
    )


root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
rows: list[dict[str, str]] = []

for traj_path in sorted(root.glob("weight_*/trajectory.hdf5"), key=lambda p: weight_from_dir(p.parent)):
    weight = weight_from_dir(traj_path.parent)
    total = 0
    successes = 0
    details: list[str] = []

    with h5py.File(traj_path, "r") as f:
        for traj_name in sorted(f):
            g = f[traj_name]
            total += 1
            obj_idx = g["object_idx"][:]
            seed = int(g.attrs.get("seed", -1))
            object_results: list[str] = []
            traj_ok = True

            for obj in sorted(set(int(x) for x in obj_idx)):
                inds = np.flatnonzero(obj_idx == obj)
                final_pos = g["object_pos"][int(inds[-1])]
                ok = object_success(final_pos)
                traj_ok = traj_ok and ok
                object_results.append(
                    f"obj{obj}={'ok' if ok else 'fail'}"
                    f"@[{final_pos[0]:.3f},{final_pos[1]:.3f},{final_pos[2]:.3f}]"
                )

            successes += int(traj_ok)
            details.append(
                f"{traj_name}:seed={seed}:traj={'ok' if traj_ok else 'fail'}:"
                + ";".join(object_results)
            )

    rate = successes / total if total else 0.0
    rows.append(
        {
            "weight": f"{weight:.2f}",
            "successes": str(successes),
            "total": str(total),
            "success_rate": f"{rate:.3f}",
            "details": " | ".join(details),
        }
    )

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=["weight", "successes", "total", "success_rate", "details"])
    writer.writeheader()
    writer.writerows(rows)

print()
print(f"[INFO] Success summary written to {summary_path}")
print("weight  successes/total  success_rate")
for row in rows:
    print(f"{row['weight']:>6}  {row['successes']}/{row['total']}              {row['success_rate']}")
PY
