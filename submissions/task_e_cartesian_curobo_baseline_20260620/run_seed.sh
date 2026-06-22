#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-120}"
OUTPUT="${2:-outputs/task_e_seed${SEED}_cartesian_curobo_baseline_20260620}"

conda run --no-capture-output -n isaaclab \
  python scripts/task_e_full_anygrasp_ee_pipeline.py \
  --headless \
  --seed "${SEED}" \
  --output "${OUTPUT}" \
  --object-order mustard_bottle,box_object,banana \
  --execute-after-each-object \
  --max-object-attempts 1 \
  --object-max-attempt-overrides banana:10 \
  --grasp-generator heuristic \
  --grasp-generator-overrides 'banana:anygrasp;mustard_bottle:heuristic;box_object:heuristic' \
  --heuristic-profile-overrides 'mustard_bottle:symmetric_bottle;box_object:box_top_center_arm_side' \
  --heuristic-family-mode mixed_diverse \
  --heuristic-symmetric-cloud-mode bottle_surface \
  --heuristic-symmetric-cloud-mode-overrides 'mustard_bottle:bottle_surface;box_object:off' \
  --grasp-selection curobo_feasible \
  --ik-filter-top-k 120 \
  --middle-support-relaxed-objects banana \
  --tcp-support-objects banana \
  --tcp-support-min-points 1 \
  --tcp-support-offset 0.010 \
  --relax-curobo-pregrasp-stage \
  --curobo-first-ik-objects banana \
  --curobo-accept-tolerance-ik-objects box_object \
  --curobo-staged-selection \
  --curobo-staged-final-ik-top-k 120 \
  --curobo-staged-full-ik-top-k 16 \
  --curobo-position-tol-overrides box_object:0.045 \
  --curobo-soft-limit-tolerance-overrides box_object:0.13 \
  --curobo-preclose-gate-objects box_object,banana \
  --curobo-preclose-cartesian-correction-objects box_object,banana \
  --curobo-grasp-cartesian-correction-objects box_object,banana \
  --curobo-grasp-cartesian-correction-steps 480 \
  --curobo-grasp-cartesian-correction-trigger-ee-tol 0.045 \
  --curobo-grasp-cartesian-correction-ee-tol 0.030 \
  --curobo-grasp-cartesian-correction-orientation-dot 0.985 \
  --cartesian-execution-objects box_object,banana \
  --cartesian-execution-stages pregrasp_stage,pregrasp,grasp,close,lift \
  --curobo-long-baseline-objects box_object,banana \
  --no-curobo-scene-collision \
  --curobo-long-baseline-step-scale 2.25 \
  --curobo-preclose-gate-max-settle-steps 650 \
  --record-video-cam \
  --force

