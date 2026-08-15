#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$script_dir"
python_bin="${ISAACLAB_PYTHON:-/home/user/miniforge3/envs/isaaclab/bin/python}"
source_run="2026-08-06_23-17-23_b2_piper_robust_20260806_215402_69aaacaf_heading_rough"
source_checkpoint="$repo_root/logs/rsl_rl/unitree_b2_piper_robust_heading_rough/$source_run/model_11999.pt"
expected_source_sha="c7a8b3b0e990c7fb724ea4408c0c2decfee8cfec5e7c0d9938d88bfa416813b8"

if [[ ! -x "$python_bin" ]]; then
    echo "Isaac Lab Python is not executable: $python_bin" >&2
    echo "Set ISAACLAB_PYTHON to the correct interpreter." >&2
    exit 1
fi

if [[ ! -f "$source_checkpoint" ]]; then
    echo "Required source checkpoint is missing: $source_checkpoint" >&2
    exit 1
fi

actual_source_sha="$(sha256sum "$source_checkpoint" | awk '{print $1}')"
if [[ "$actual_source_sha" != "$expected_source_sha" ]]; then
    echo "Source checkpoint SHA-256 mismatch." >&2
    echo "Expected: $expected_source_sha" >&2
    echo "Actual:   $actual_source_sha" >&2
    exit 1
fi

# This is essential when the same conda environment has an editable install of
# the newer Clear_ATEC2026_Simulation_Challenge checkout.
export PYTHONPATH="$repo_root/source/atec_rl_lab${PYTHONPATH:+:$PYTHONPATH}"

resolved_package="$("$python_bin" -c 'from pathlib import Path; import atec_rl_lab; print(Path(atec_rl_lab.__file__).resolve())')"
if [[ "$resolved_package" != "$repo_root"/* ]]; then
    echo "Refusing to train with atec_rl_lab imported from: $resolved_package" >&2
    echo "Expected it below: $repo_root" >&2
    exit 1
fi

cd "$repo_root"
exec "$python_bin" "$repo_root/scripts/rsl_rl/train.py" \
    --task ATEC-Isaac-Velocity-Robust-Heading-Rough-Unitree-B2-Piper-v0 \
    --num_envs 2048 \
    --max_iterations 8000 \
    --run_name b2_piper_robust_finegrain_20260807_190441_a74cf724 \
    --resume \
    --load_run '^2026\-08\-06_23\-17\-23_b2_piper_robust_20260806_215402_69aaacaf_heading_rough$' \
    --checkpoint '^model_11999\.pt$' \
    --spawn_audit \
    --headless \
    --seed 42 \
    "$@"
