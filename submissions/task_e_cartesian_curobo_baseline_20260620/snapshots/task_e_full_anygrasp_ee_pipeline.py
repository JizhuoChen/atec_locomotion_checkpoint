#!/usr/bin/env python3
"""Full Task E AnyGrasp pipeline with video rough localization and EE top-down refinement."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from isaaclab.app import AppLauncher

from task_e_full_baseline_request import (  # noqa: E402
    BASKET_CENTER_X,
    BASKET_CENTER_Y,
    CARRY_Z,
    DEFAULT_PICK_ORDER,
    DEFAULT_QUAT_WXYZ,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    OBJECTS,
    RETRACT_POS_X,
    RETRACT_POS_Y,
    TABLE_CENTER_X,
    TABLE_CENTER_Y,
    TABLE_DIMS,
    TABLE_TOP_Z,
    deterministic_object_poses,
    parse_grasp_tuning,
    parse_object_order,
    quat_wxyz_from_matrix,
)
from task_e_full_anygrasp_request import build_request, grasp_pose_for_object  # noqa: E402


OBJECT_PROMPTS = {
    "banana": "banana",
    "mustard_bottle": "mustard bottle",
    "box_object": "yellow and white box",
}
ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINT_NAMES = ["joint7", "joint8"]
ACTION_SCALE = 0.5
TASK_E_DEFAULT_ARM_JOINT_POS = np.array([0.0, 1.2, -1.5, 0.0, 1.2, 0.0], dtype=np.float64)
TOP_DOWN_GRIPPER_QUAT_WXYZ = [0.0, 1.0, 0.0, 0.0]
PIPER_FINGER_LENGTH_M = 0.0765
PIPER_FINGER_WIDTH_M = 0.0265
PIPER_FINGER_DEPTH_M = 0.056
PIPER_PALM_APPROACH_THICKNESS_M = 0.035
PIPER_MAX_JAW_WIDTH_M = 0.075

BASELINE_ACTION_X_GRID = np.array([0.90, 0.95, 1.00, 1.05, 1.10], dtype=np.float64)
BASELINE_ACTION_PRIORS = {
    "mustard_bottle": {
        "fallback_x": 1.00,
        "x_bias": 0.04,
        "fallback_y": 0.18,
        "y_grid": (0.145, 0.18),
        "pre_low": [
            [-0.8285, 1.6723, -2.1158, 0.0, 0.36, -0.8285],
            [-0.8285, 1.6723, -2.1158, 0.0, 0.36, -0.8285],
            [-0.8285, 1.6723, -2.1158, 0.0, 0.36, -0.8285],
            [-0.8285, 1.6723, -2.1158, 0.0, 0.36, -0.8285],
            [-0.8285, 1.6723, -2.1158, 0.0, 0.36, -0.8285],
        ],
        "grasp_low": [
            [-0.8276, 1.3949, 0.5402, -0.0063, -0.7241, -0.8244],
            [-0.8276, 1.3949, 0.5402, -0.0063, -0.7241, -0.8244],
            [-0.8276, 1.3949, 0.5402, -0.0063, -0.7241, -0.8244],
            [-0.8276, 1.3949, 0.5402, -0.0063, -0.7241, -0.8244],
            [-0.8276, 1.3949, 0.5402, -0.0063, -0.7241, -0.8244],
        ],
        "pre": [
            [-0.8285, 1.6723, -2.1158, 0.0, 0.36, -0.8285],
            [-0.8285, 1.6723, -2.1158, 0.0, 0.36, -0.8285],
            [-0.8285, 1.6723, -2.1158, 0.0, 0.36, -0.8285],
            [-0.8285, 1.6723, -2.1158, 0.0, 0.36, -0.8285],
            [-0.8285, 1.6723, -2.1158, 0.0, 0.36, -0.8285],
        ],
        "grasp": [
            [-0.8276, 1.3949, 0.5402, -0.0063, -0.7241, -0.8244],
            [-0.8276, 1.3949, 0.5402, -0.0063, -0.7241, -0.8244],
            [-0.8276, 1.3949, 0.5402, -0.0063, -0.7241, -0.8244],
            [-0.8276, 1.3949, 0.5402, -0.0063, -0.7241, -0.8244],
            [-0.8276, 1.3949, 0.5402, -0.0063, -0.7241, -0.8244],
        ],
    },
    "box_object": {
        "fallback_x": 1.00,
        "x_bias": 0.03,
        "pre": [
            [-0.9832, 2.3189, -2.5633, 0.0, 0.36, -0.9831],
            [-1.0726, 2.2173, -2.5327, 0.0, 0.36, -1.0725],
            [-1.1777, 2.1012, -2.4858, 0.0, 0.36, -1.1777],
            [-1.3025, 1.9668, -2.4091, 0.0, 0.36, -1.3025],
            [-1.4516, 1.8068, -2.2738, 0.0, 0.36, -1.4516],
        ],
        "grasp": [
            [-0.9832, 3.2082, -2.4437, 0.0018, 0.3600, -0.9822],
            [-1.0718, 3.0205, -2.0798, -0.0022, 0.3601, -1.0700],
            [-1.1746, 2.5474, -1.0697, 0.0069, 0.0397, -1.1777],
            [-1.3005, 2.2123, -0.4041, 0.0012, -0.2920, -1.3021],
            [-1.4509, 1.9343, 0.1222, -0.0004, -0.5411, -1.4520],
        ],
    },
}


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def json_text(payload: object) -> str:
    return json.dumps(payload, indent=2, default=json_default)


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_object_name_set(value: str) -> list[str]:
    value = value.strip()
    if not value or value.lower() in {"none", "off", "false"}:
        return []
    if value.lower() == "all":
        return list(OBJECTS)
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in OBJECTS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown objects: {unknown}. Valid: {sorted(OBJECTS)}")
    return names


def parse_stage_name_set(value: str) -> list[str]:
    value = value.strip()
    if not value or value.lower() in {"none", "off", "false"}:
        return []
    allowed = {"pregrasp_stage", "pregrasp", "grasp", "lift", "close"}
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown stages: {unknown}. Valid: {sorted(allowed)}")
    return names


def _interp_baseline_action_x(object_name: str, key: str, x: float) -> np.ndarray | None:
    cfg = BASELINE_ACTION_PRIORS.get(object_name)
    if cfg is None or key not in cfg:
        return None
    table = np.asarray(cfg[key], dtype=np.float64)
    x = float(np.clip(x, float(BASELINE_ACTION_X_GRID[0]), float(BASELINE_ACTION_X_GRID[-1])))
    return np.asarray(
        [np.interp(x, BASELINE_ACTION_X_GRID, table[:, dim]) for dim in range(table.shape[1])],
        dtype=np.float64,
    )


def baseline_expected_arm_action(object_name: str, stage_label: str, object_pose: dict) -> tuple[np.ndarray | None, dict]:
    cfg = BASELINE_ACTION_PRIORS.get(object_name)
    if cfg is None:
        return None, {"enabled": False, "reason": "no_baseline_prior_for_object"}

    center = np.asarray(object_pose.get("center_w", []), dtype=np.float64)
    if center.shape[0] < 2 or not np.isfinite(center[:2]).all():
        x = float(cfg.get("fallback_x", 1.0))
        y = cfg.get("fallback_y")
        center_source = "fallback"
    else:
        x = float(center[0]) + float(cfg.get("x_bias", 0.0))
        y = float(center[1])
        center_source = "object_pose_center"

    key = "grasp" if stage_label == "grasp" else "pre"
    low_key = f"{key}_low"
    action = _interp_baseline_action_x(object_name, key, x)
    if action is None:
        return None, {"enabled": False, "reason": f"missing_baseline_key:{key}"}

    if y is not None and low_key in cfg and "y_grid" in cfg:
        low = _interp_baseline_action_x(object_name, low_key, x)
        if low is not None:
            y0, y1 = cfg["y_grid"]
            alpha = float(np.clip((float(y) - float(y0)) / max(float(y1) - float(y0), 1e-6), 0.0, 1.0))
            action = low + alpha * (action - low)

    return action.astype(np.float64), {
        "enabled": True,
        "object": object_name,
        "stage": stage_label,
        "baseline_key": key,
        "x_for_prior": float(x),
        "y_for_prior": None if y is None else float(y),
        "center_source": center_source,
    }


def baseline_action_prior_report(name: str, object_pose: dict, stages: list[dict]) -> dict:
    enabled = bool(args_cli.curobo_baseline_action_prior)
    if not enabled:
        return {"enabled": False, "reason": "disabled_by_cli"}
    if name not in BASELINE_ACTION_PRIORS:
        return {"enabled": False, "reason": "no_baseline_prior_for_object"}

    stage_reports = []
    comparable_l2 = []
    comparable_linf = []
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        label = str(stage.get("label", ""))
        joints = stage.get("joint_position")
        if label not in {"pregrasp_stage", "pregrasp", "grasp", "lift"}:
            continue
        if not isinstance(joints, list) or len(joints) != len(ARM_JOINT_NAMES):
            continue
        expected_action, meta = baseline_expected_arm_action(name, label, object_pose)
        target_action = (np.asarray(joints, dtype=np.float64) - TASK_E_DEFAULT_ARM_JOINT_POS) / float(ACTION_SCALE)
        report = {
            "label": label,
            "target_arm_action": target_action.astype(float).tolist(),
            "prior": meta,
        }
        if expected_action is not None:
            delta = target_action - expected_action
            l2 = float(np.linalg.norm(delta))
            linf = float(np.max(np.abs(delta)))
            comparable_l2.append(l2)
            comparable_linf.append(linf)
            report.update(
                {
                    "baseline_arm_action": expected_action.astype(float).tolist(),
                    "delta_action": delta.astype(float).tolist(),
                    "l2": l2,
                    "linf": linf,
                }
            )
        stage_reports.append(report)

    if not stage_reports:
        return {"enabled": True, "comparable": False, "reason": "no_comparable_stages", "stages": []}

    return {
        "enabled": True,
        "comparable": bool(comparable_l2),
        "mean_stage_l2": float(np.mean(comparable_l2)) if comparable_l2 else None,
        "max_stage_l2": float(np.max(comparable_l2)) if comparable_l2 else None,
        "mean_stage_linf": float(np.mean(comparable_linf)) if comparable_linf else None,
        "max_stage_linf": float(np.max(comparable_linf)) if comparable_linf else None,
        "stages": stage_reports,
    }


def baseline_action_prior_sort_value(evaluation: dict) -> float:
    if not bool(args_cli.curobo_baseline_action_prior_ranking):
        return 0.0
    report = evaluation.get("baseline_action_prior")
    if not isinstance(report, dict) or not report.get("enabled") or not report.get("comparable"):
        return 1e6
    value = report.get("mean_stage_l2")
    return float(value) if value is not None and np.isfinite(float(value)) else 1e6


def box_arm_side_selection_preference(name: str, evaluation: dict, object_pose: dict | None) -> tuple[dict, tuple]:
    if name != "box_object":
        return {"enabled": False, "reason": "not_box_object"}, ()

    pose = evaluation.get("piper_gripper_base_pose_w")
    position = pose.get("position") if isinstance(pose, dict) else None
    if position is None:
        return {
            "enabled": True,
            "comparable": False,
            "reason": "missing_piper_gripper_base_pose_w",
            "sort_key": [1, 1e6, 1e6],
        }, (1, 1e6, 1e6)

    pos = np.asarray(position, dtype=np.float64)
    if pos.shape[0] < 3 or not np.isfinite(pos[:3]).all():
        return {
            "enabled": True,
            "comparable": False,
            "reason": "invalid_piper_gripper_base_position",
            "position_w": [float(v) for v in np.ravel(pos)[:3]] if pos.size >= 3 else None,
            "sort_key": [1, 1e6, 1e6],
        }, (1, 1e6, 1e6)

    arm_side_distance = abs(float(pos[1]))
    height = float(pos[2])
    center = np.asarray((object_pose or {}).get("center_w", []), dtype=np.float64)
    object_center_y = float(center[1]) if center.shape[0] >= 2 and np.isfinite(center[1]) else None
    object_center_arm_distance = abs(object_center_y) if object_center_y is not None else None
    sort_key = (0, arm_side_distance, -height)
    return {
        "enabled": True,
        "comparable": True,
        "policy": "box_object_prefers_nearest_arm_side_then_highest_contact_pose",
        "arm_reference_y_w": 0.0,
        "arm_side_distance_m": float(arm_side_distance),
        "height_m": float(height),
        "object_center_y_w": object_center_y,
        "object_center_arm_distance_m": object_center_arm_distance,
        "closer_to_arm_than_object_center_m": (
            float(object_center_arm_distance - arm_side_distance)
            if object_center_arm_distance is not None
            else None
        ),
        "position_w": pos[:3].astype(float).tolist(),
        "sort_key": [float(v) for v in sort_key],
    }, sort_key


def parse_object_choice_map(value: str, *, choices: set[str], label: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    value = value.strip()
    if not value or value.lower() in {"none", "off", "false"}:
        return mapping
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError(f"Expected entries like box_object:{next(iter(sorted(choices)))}")
        name, payload = item.split(":", 1)
        name = name.strip()
        choice = payload.strip()
        if name not in OBJECTS:
            raise argparse.ArgumentTypeError(f"Unknown object in {label}: {name!r}")
        if choice not in choices:
            raise argparse.ArgumentTypeError(f"Unknown {label} value {choice!r}. Valid: {sorted(choices)}")
        mapping[name] = choice
    return mapping


def parse_heuristic_profile_overrides(value: str) -> dict[str, str]:
    return parse_object_choice_map(
        value,
        choices={"topdown", "observed_centerline", "symmetric_bottle", "mixed", "box_edge", "box_top_center_arm_side"},
        label="heuristic profile override",
    )


def parse_grasp_generator_overrides(value: str) -> dict[str, str]:
    return parse_object_choice_map(
        value,
        choices={"anygrasp", "contact_graspnet", "graspgen", "heuristic"},
        label="grasp generator override",
    )


def parse_heuristic_symmetric_cloud_overrides(value: str) -> dict[str, str]:
    return parse_object_choice_map(
        value,
        choices={"off", "mirror", "bottle_surface"},
        label="heuristic symmetric cloud override",
    )


def parse_object_offsets(value: str) -> dict[str, dict[str, float]]:
    offsets: dict[str, dict[str, float]] = {}
    value = value.strip()
    if not value or value.lower() in {"none", "off", "false"}:
        return offsets
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError("Expected entries like banana:0,0,-0.02")
        name, payload = item.split(":", 1)
        name = name.strip()
        if name not in OBJECTS:
            raise argparse.ArgumentTypeError(f"Unknown object in hover offset: {name!r}")
        parts = [float(part.strip()) for part in payload.split(",") if part.strip()]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(f"Expected dx,dy,dz for {name}, got {payload!r}")
        offsets[name] = {"dx": parts[0], "dy": parts[1], "dz": parts[2]}
    return offsets


def parse_object_int_overrides(value: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    value = value.strip()
    if not value or value.lower() in {"none", "off", "false"}:
        return mapping
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError("Expected entries like banana:3")
        name, payload = item.split(":", 1)
        name = name.strip()
        if name not in OBJECTS:
            raise argparse.ArgumentTypeError(f"Unknown object in integer override: {name!r}")
        try:
            mapping[name] = int(payload.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid integer override for {name}: {payload!r}") from exc
    return mapping


def parse_object_float_overrides(value: str) -> dict[str, float]:
    mapping: dict[str, float] = {}
    value = value.strip()
    if not value or value.lower() in {"none", "off", "false"}:
        return mapping
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError("Expected entries like box_object:0.045")
        name, payload = item.split(":", 1)
        name = name.strip()
        if name not in OBJECTS:
            raise argparse.ArgumentTypeError(f"Unknown object in float override: {name!r}")
        try:
            mapping[name] = float(payload.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid float override for {name}: {payload!r}") from exc
    return mapping


def parse_hover_mode_list(value: str) -> list[str]:
    modes = [part.strip() for part in value.split(",") if part.strip()]
    valid = {"gripper_offset", "camera_center", "look_at"}
    unknown = [mode for mode in modes if mode not in valid]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown hover modes: {unknown}. Valid: {sorted(valid)}")
    if not modes:
        raise argparse.ArgumentTypeError("Expected at least one hover mode.")
    return modes


def parse_piper_offset_modes(value: str) -> list[str]:
    modes = [part.strip() for part in value.split(",") if part.strip()]
    valid = {"approach_axis", "finger_centerline", "yellow_line", "towards_object_center"}
    unknown = [mode for mode in modes if mode not in valid]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown Piper offset modes: {unknown}. Valid: {sorted(valid)}")
    if not modes:
        raise argparse.ArgumentTypeError("Expected at least one Piper offset mode.")
    return modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/task_e_full_anygrasp_ee/<timestamp>.",
    )
    parser.add_argument(
        "--object-order",
        type=parse_object_order,
        default=list(DEFAULT_PICK_ORDER),
        help="Comma-separated pick order. Default: banana,mustard_bottle,box_object.",
    )
    parser.add_argument(
        "--object-center-source",
        choices=("deterministic", "video_mask", "ee_mask"),
        default="ee_mask",
        help="Object centers embedded into the execution request.",
    )
    parser.add_argument(
        "--hover-target-source",
        choices=("video_mask", "deterministic"),
        default="video_mask",
        help="Coarse object center used to move the EE camera above each object.",
    )
    parser.add_argument(
        "--video-color-refine-objects",
        type=parse_object_name_set,
        default=["banana"],
        help=(
            "Comma-separated object names whose video-camera rough pose should be refined by "
            "the color component overlapping the SAM3 mask. Default: banana."
        ),
    )
    parser.add_argument(
        "--grasp-mode",
        choices=("raw_anygrasp", "topdown_position"),
        default="raw_anygrasp",
        help="Passed to the generated motion request.",
    )
    parser.add_argument(
        "--raw-tool-transform",
        choices=("identity", "graspnet_to_piper_z"),
        default="graspnet_to_piper_z",
    )
    parser.add_argument("--gripper-base-offset", type=float, default=0.09)
    parser.add_argument(
        "--gripper-base-offset-mode",
        choices=("approach_axis", "finger_centerline", "yellow_line", "towards_object_center"),
        default="towards_object_center",
        help=(
            "How to translate the executed Piper gripper_base away from the generated "
            "AnyGrasp contact pose."
        ),
    )
    parser.add_argument("--grasp-tuning", type=parse_grasp_tuning, default={})
    parser.add_argument(
        "--object-transport-mode",
        choices=("physics", "kinematic_attach"),
        default="physics",
    )
    parser.add_argument(
        "--actuator-mode",
        choices=("default", "task_e_scripted_high_stiffness"),
        default="default",
    )
    parser.add_argument("--preferred-backend", choices=("moveit_py", "isaaclab_cartesian_controller"), default="moveit_py")
    parser.add_argument("--sam3-env", default="sam3_full")
    parser.add_argument("--anygrasp-env", default="anygrasp")
    parser.add_argument(
        "--grasp-generator",
        choices=("anygrasp", "contact_graspnet", "graspgen", "heuristic"),
        default="graspgen",
        help="Third-party grasp generator used after RGB-D/SAM capture. Defaults to GraspGen.",
    )
    parser.add_argument(
        "--grasp-generator-overrides",
        type=parse_grasp_generator_overrides,
        default={},
        help=(
            "Semicolon-separated per-object generator overrides, e.g. "
            "banana:anygrasp;mustard_bottle:heuristic;box_object:heuristic."
        ),
    )
    parser.add_argument(
        "--anygrasp-symmetric-cloud-mode",
        choices=("off", "mirror", "bottle_surface"),
        default="off",
        help="Complete the target cloud before AnyGrasp inference.",
    )
    parser.add_argument(
        "--anygrasp-symmetry-center-source",
        choices=("bbox_center", "object_center", "mean"),
        default="bbox_center",
    )
    parser.add_argument("--anygrasp-symmetric-surface-points", type=int, default=16000)
    parser.add_argument("--contact-graspnet-env", default="contact_graspnet_env")
    parser.add_argument(
        "--contact-graspnet-root",
        type=Path,
        default=REPO_ROOT / "third_party/contact_graspnet",
    )
    parser.add_argument(
        "--contact-graspnet-ckpt-dir",
        type=Path,
        default=REPO_ROOT / "third_party/contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001",
    )
    parser.add_argument("--contact-graspnet-forward-passes", type=int, default=1)
    parser.add_argument("--contact-graspnet-local-regions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contact-graspnet-filter-grasps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graspgen-env", default="graspgen_env")
    parser.add_argument(
        "--graspgen-root",
        type=Path,
        default=REPO_ROOT / "third_party/graspgen",
    )
    parser.add_argument(
        "--graspgen-gripper-config",
        type=Path,
        default=REPO_ROOT / "third_party/graspgen_models/checkpoints/graspgen_robotiq_2f_140.yml",
    )
    parser.add_argument("--graspgen-target-cloud-max-points", type=int, default=2048)
    parser.add_argument("--graspgen-num-grasps", type=int, default=500)
    parser.add_argument("--graspgen-collision-max-scene-points", type=int, default=8192)
    parser.add_argument("--graspgen-export-tcp-offset", type=float, default=0.195)
    parser.add_argument("--graspgen-filter-collisions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--graspgen-symmetric-cloud-mode",
        choices=("off", "mirror", "bottle_surface"),
        default="off",
        help="Complete the target cloud before GraspGen inference, useful for upright symmetric bottles.",
    )
    parser.add_argument(
        "--graspgen-symmetry-center-source",
        choices=("bbox_center", "object_center", "mean"),
        default="bbox_center",
    )
    parser.add_argument("--graspgen-symmetric-surface-points", type=int, default=16000)
    parser.add_argument(
        "--heuristic-profile",
        choices=("topdown", "observed_centerline", "symmetric_bottle", "mixed", "box_edge", "box_top_center_arm_side"),
        default="topdown",
        help=(
            "topdown uses the legacy single heuristic pose. Other modes post-process the fused cloud "
            "with the Piper-sized heuristic sampler before CuRobo selection."
        ),
    )
    parser.add_argument(
        "--heuristic-profile-overrides",
        type=parse_heuristic_profile_overrides,
        default={},
        help=(
            "Semicolon-separated per-object heuristic profiles, e.g. "
            "box_object:observed_centerline;mustard_bottle:symmetric_bottle."
        ),
    )
    parser.add_argument("--heuristic-attempts", type=int, default=25000)
    parser.add_argument(
        "--heuristic-symmetric-cloud-mode",
        choices=("off", "mirror", "bottle_surface"),
        default="bottle_surface",
    )
    parser.add_argument(
        "--heuristic-symmetric-cloud-mode-overrides",
        type=parse_heuristic_symmetric_cloud_overrides,
        default={},
        help=(
            "Semicolon-separated per-object symmetric cloud modes for the heuristic sampler, "
            "e.g. box_object:off;mustard_bottle:bottle_surface."
        ),
    )
    parser.add_argument(
        "--heuristic-symmetry-center-source",
        choices=("bbox_center", "object_center", "mean"),
        default="bbox_center",
    )
    parser.add_argument("--heuristic-symmetric-surface-points", type=int, default=16000)
    parser.add_argument("--heuristic-symmetric-top-grasp-fraction", type=float, default=0.70)
    parser.add_argument("--heuristic-candidate-filter-max-points", type=int, default=60000)
    parser.add_argument(
        "--heuristic-family-mode",
        choices=("auto", "topdown", "tilted_top", "side_orbit", "side_x", "side_y", "diagonal_side", "mixed_diverse"),
        default="auto",
        help="Family of Piper heuristic candidates generated before CuRobo filtering.",
    )
    parser.add_argument("--heuristic-root-centerline-clear-length", type=float, default=0.025)
    parser.add_argument("--heuristic-root-centerline-max-points", type=int, default=0)
    parser.add_argument(
        "--heuristic-vary-seed-by-attempt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When retrying an object in one environment episode, offset the second-stage Piper "
            "heuristic sampler seed by attempt number so retries explore different candidates."
        ),
    )
    parser.add_argument(
        "--heuristic-attempt-seed-stride",
        type=int,
        default=9973,
        help="Seed stride used by --heuristic-vary-seed-by-attempt.",
    )
    parser.add_argument(
        "--heuristic-symmetric-ee-roll",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For Piper heuristic grasps, also evaluate the 180-degree roll-equivalent "
            "parallel-jaw orientation before CuRobo selection."
        ),
    )
    parser.add_argument("--sam3-device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--ee-mask-source",
        choices=("sam3", "color"),
        default="sam3",
        help="Use SAM3 or a simple top-down color mask for the EE-camera refinement view.",
    )
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument(
        "--hover-mode",
        choices=("gripper_offset", "camera_center", "look_at", "adaptive_lookat"),
        default="adaptive_lookat",
        help=(
            "gripper_offset uses the verified top-down gripper pose plus a camera-centering "
            "offset. camera_center tries to solve the camera pose directly. look_at keeps the "
            "camera near its initial position and pitches/yaws it toward the object. "
            "adaptive_lookat uses look_at for --lookat-objects and gripper_offset otherwise."
        ),
    )
    parser.add_argument("--hover-height", type=float, default=0.23)
    parser.add_argument(
        "--gripper-hover-offset",
        type=parse_float_list,
        default=[0.05, 0.0, 0.0],
        help="XYZ world offset from rough object center to desired EE-camera center for gripper_offset mode.",
    )
    parser.add_argument("--hover-settle-steps", type=int, default=260)
    parser.add_argument(
        "--lookat-objects",
        type=parse_object_name_set,
        default=["mustard_bottle", "box_object"],
        help="Objects that use the initial-position look-at EE camera pose in adaptive_lookat mode.",
    )
    parser.add_argument(
        "--lookat-camera-position",
        type=parse_float_list,
        default=[],
        help="Optional absolute world XYZ camera position for look_at mode. Defaults to the reset EE camera position.",
    )
    parser.add_argument(
        "--lookat-camera-offset",
        type=parse_float_list,
        default=[0.0, 0.0, 0.0],
        help="XYZ world offset added to the look_at camera position.",
    )
    parser.add_argument(
        "--hover-object-offsets",
        type=parse_object_offsets,
        default={},
        help="Semicolon-separated object:dx,dy,dz offsets applied to the EE camera hover target.",
    )
    parser.add_argument("--hover-center-iters", type=int, default=1)
    parser.add_argument("--hover-center-steps", type=int, default=100)
    parser.add_argument("--hover-center-gain", type=float, default=0.75)
    parser.add_argument("--hover-center-max-shift", type=float, default=0.08)
    parser.add_argument("--hover-center-pixel", type=parse_float_list, default=[320.0, 240.0])
    parser.add_argument(
        "--dual-hover-objects",
        type=parse_object_name_set,
        default=["banana"],
        help=(
            "Objects that try hover modes from --dual-hover-modes until one produces a "
            "usable EE-camera SAM3/depth estimate. Default: banana."
        ),
    )
    parser.add_argument(
        "--dual-hover-modes",
        type=parse_hover_mode_list,
        default=["gripper_offset", "look_at"],
        help="Comma-separated EE hover modes tried for --dual-hover-objects.",
    )
    parser.add_argument(
        "--ee-multi-view-objects",
        type=parse_object_name_set,
        default=[],
        help="Objects whose EE-camera AnyGrasp input should fuse a second segmented view.",
    )
    parser.add_argument(
        "--anygrasp-video-view-objects",
        type=parse_object_name_set,
        default=[],
        help=(
            "Objects whose AnyGrasp input should also fuse the video_cam segmented RGB-D view. "
            "Use 'all' to enable it for every object."
        ),
    )
    parser.add_argument(
        "--anygrasp-full-scene-objects",
        type=parse_object_name_set,
        default=[],
        help=(
            "Objects whose AnyGrasp input should use the full RGB-D scene cloud, then filter "
            "generated grasps back to the segmented target object. Use 'all' to enable it for every object."
        ),
    )
    parser.add_argument("--anygrasp-scene-cloud-stride", type=int, default=1)
    parser.add_argument("--anygrasp-scene-cloud-max-points", type=int, default=180000)
    parser.add_argument("--target-grasp-filter-distance", type=float, default=0.035)
    parser.add_argument("--target-grasp-filter-pixel-radius", type=int, default=8)
    parser.add_argument(
        "--ee-second-view-offset",
        type=parse_float_list,
        default=[0.0, 0.14, 0.03],
        help="World XYZ camera-position offset for the second EE look-at view.",
    )
    parser.add_argument(
        "--save-top-grasps",
        type=int,
        default=10,
        help="Number of generator candidates to persist. Raised to --ik-filter-top-k when needed.",
    )
    parser.add_argument(
        "--grasp-selection",
        choices=("score_only", "ik_feasible", "curobo_feasible", "curobo_first_ik"),
        default="ik_feasible",
        help=(
            "score_only keeps the generator's top candidate. ik_feasible uses the legacy "
            "Cartesian IK probe. curobo_feasible filters with clipped Piper geometry and "
            "cuRobo IK. curobo_first_ik uses the same geometry and approach "
            "filters, then accepts the first candidate whose cuRobo waypoints solve in score order."
        ),
    )
    parser.add_argument(
        "--curobo-first-ik-objects",
        type=parse_object_name_set,
        default=[],
        help=(
            "Comma-separated object names that use the curobo_first_ik acceptance policy even when "
            "--grasp-selection is curobo_feasible. This keeps stricter CuRobo tolerances for other objects."
        ),
    )
    parser.add_argument(
        "--ik-filter-raw-anygrasp",
        action="store_true",
        help=(
            "Deprecated compatibility flag. IK probing now always checks the final grasp pose. "
            "When point-cloud offset search is enabled, the offset is applied before IK."
        ),
    )
    parser.add_argument("--ik-filter-top-k", type=int, default=5)
    parser.add_argument("--ik-filter-position-tol", type=float, default=0.045)
    parser.add_argument("--ik-filter-orientation-dot", type=float, default=0.90)
    parser.add_argument("--ik-filter-stage-steps", type=int, default=80)
    parser.add_argument("--ik-filter-target-steps", type=int, default=90)
    parser.add_argument("--piper-max-jaw-width", type=float, default=PIPER_MAX_JAW_WIDTH_M)
    parser.add_argument(
        "--piper-clip-generator-width",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate generator widths using min(generator_width, --piper-max-jaw-width).",
    )
    parser.add_argument(
        "--piper-offset-modes",
        type=parse_piper_offset_modes,
        default=["approach_axis", "finger_centerline"],
        help="Comma-separated offset modes searched by curobo_feasible selection.",
    )
    parser.add_argument("--piper-offset-min", type=float, default=0.0)
    parser.add_argument("--piper-offset-max", type=float, default=0.12)
    parser.add_argument("--piper-offset-step", type=float, default=0.005)
    parser.add_argument("--target-solid-max-points", type=int, default=0)
    parser.add_argument("--obstacle-solid-max-points", type=int, default=0)
    parser.add_argument("--centerline-min-points", type=int, default=10)
    parser.add_argument("--closing-region-min-points", type=int, default=10)
    parser.add_argument(
        "--centerline-relaxed-objects",
        type=parse_object_name_set,
        default=[],
        help="Objects allowed to bypass the target centerline point-count grasp geometry check.",
    )
    parser.add_argument("--centerline-half-width", type=float, default=0.012)
    parser.add_argument("--centerline-half-depth", type=float, default=0.018)
    parser.add_argument(
        "--middle-support-min-points",
        type=int,
        default=5,
        help=(
            "Minimum target-cloud points required around the gripper middle support point. "
            "This is checked before CuRobo and again on the CuRobo reached FK pose."
        ),
    )
    parser.add_argument(
        "--middle-support-relaxed-objects",
        type=parse_object_name_set,
        default=[],
        help="Objects allowed to bypass the 2.5 cm middle-support point-count grasp geometry check.",
    )
    parser.add_argument(
        "--middle-support-offset",
        type=float,
        default=0.025,
        help="Distance in metres from the fingertip/TCP contact point back toward the finger root.",
    )
    parser.add_argument("--middle-support-half-length", type=float, default=0.012)
    parser.add_argument("--middle-support-half-width", type=float, default=0.014)
    parser.add_argument("--middle-support-half-depth", type=float, default=0.020)
    parser.add_argument(
        "--tcp-support-objects",
        type=parse_object_name_set,
        default=[],
        help=(
            "Objects that require target-cloud support near the fingertip/TCP contact point. "
            "Use banana to require the 1 cm upper TCP point to lie inside the target cloud."
        ),
    )
    parser.add_argument("--tcp-support-min-points", type=int, default=1)
    parser.add_argument(
        "--tcp-support-offset",
        type=float,
        default=0.010,
        help="Distance in metres from the fingertip/TCP contact point back toward the finger root.",
    )
    parser.add_argument("--tcp-support-half-length", type=float, default=0.008)
    parser.add_argument("--tcp-support-half-width", type=float, default=0.012)
    parser.add_argument("--tcp-support-half-depth", type=float, default=0.018)
    parser.add_argument("--obstacle-target-exclusion-radius", type=float, default=0.012)
    parser.add_argument("--obstacle-cloud-max-points", type=int, default=60000)
    parser.add_argument("--curobo-num-seeds", type=int, default=96)
    parser.add_argument("--curobo-position-tol", type=float, default=0.025)
    parser.add_argument(
        "--curobo-position-tol-overrides",
        type=parse_object_float_overrides,
        default={},
        help="Per-object CuRobo IK position tolerance overrides, e.g. box_object:0.045.",
    )
    parser.add_argument("--curobo-rotation-tol", type=float, default=0.30)
    parser.add_argument(
        "--curobo-accept-tolerance-ik-objects",
        type=parse_object_name_set,
        default=[],
        help=(
            "Objects whose CuRobo stage can be accepted when measured IK errors are inside "
            "the configured tolerances even if CuRobo's strict success flag is false."
        ),
    )
    parser.add_argument(
        "--relax-curobo-pregrasp-stage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Do not require the high pregrasp_stage waypoint to pass CuRobo candidate "
            "acceptance. The pregrasp, grasp, lift, and reached-grasp point-cloud checks "
            "remain required."
        ),
    )
    parser.add_argument("--curobo-request-timeout", type=float, default=120.0)
    parser.add_argument(
        "--curobo-joint-execution-settle-tol",
        type=float,
        default=0.02,
        help="Max arm joint error in radians required before a CuRobo joint waypoint is treated as reached.",
    )
    parser.add_argument(
        "--curobo-joint-execution-max-settle-steps",
        type=int,
        default=320,
        help="Extra env steps spent holding a CuRobo joint target until joint error is below tolerance.",
    )
    parser.add_argument(
        "--curobo-joint-execution-interp-step",
        type=float,
        default=0.02,
        help="Maximum nominal joint-space interpolation step in radians for CuRobo joint waypoint execution.",
    )
    parser.add_argument(
        "--curobo-joint-execution-mode",
        choices=("settle", "baseline_interp"),
        default="baseline_interp",
        help=(
            "settle uses tight joint-target settle gates. baseline_interp follows the submission "
            "baseline style: long action-space transitions, close holds, and tracking-lag diagnostics."
        ),
    )
    parser.add_argument("--curobo-baseline-pregrasp-stage-steps", type=int, default=140)
    parser.add_argument("--curobo-baseline-pregrasp-steps", type=int, default=190)
    parser.add_argument("--curobo-baseline-grasp-steps", type=int, default=130)
    parser.add_argument("--curobo-baseline-close-steps", type=int, default=90)
    parser.add_argument("--curobo-baseline-close-hold-steps", type=int, default=35)
    parser.add_argument("--curobo-baseline-lift-steps", type=int, default=190)
    parser.add_argument(
        "--curobo-long-baseline-objects",
        type=parse_object_name_set,
        default=["box_object"],
        help="Objects whose CuRobo baseline interpolation waypoint durations are scaled longer.",
    )
    parser.add_argument(
        "--curobo-long-baseline-step-scale",
        type=float,
        default=1.75,
        help="Multiplier applied to baseline interpolation steps for --curobo-long-baseline-objects.",
    )
    parser.add_argument(
        "--curobo-baseline-settle-objects",
        type=parse_object_name_set,
        default=["box_object"],
        help="Objects that hold CuRobo joint targets open after baseline pregrasp/grasp paths.",
    )
    parser.add_argument(
        "--curobo-baseline-settle-stages",
        type=parse_stage_name_set,
        default=["pregrasp", "grasp"],
        help="Comma-separated CuRobo stage labels that receive baseline settle holds.",
    )
    parser.add_argument(
        "--curobo-baseline-settle-ee-tol",
        type=float,
        default=0.045,
        help="Optional EE position tolerance for baseline settle holds; set <=0 to use joint error only.",
    )
    parser.add_argument(
        "--curobo-preclose-gate-objects",
        type=parse_object_name_set,
        default=["box_object"],
        help="Objects whose close waypoint is skipped unless measured EE and joint errors are small.",
    )
    parser.add_argument("--curobo-preclose-gate-joint-tol", type=float, default=0.05)
    parser.add_argument("--curobo-preclose-gate-ee-tol", type=float, default=0.045)
    parser.add_argument(
        "--curobo-preclose-gate-max-settle-steps",
        type=int,
        default=420,
        help="Extra open-gripper hold steps before a gated close is allowed.",
    )
    parser.add_argument(
        "--curobo-preclose-cartesian-correction-objects",
        type=parse_object_name_set,
        default=["box_object"],
        help=(
            "Objects whose close waypoint may run an online Cartesian correction after the "
            "CuRobo joint target fails the preclose tracking gate. The correction still emits "
            "joint-position actions, but it uses measured EE pose feedback before closing."
        ),
    )
    parser.add_argument(
        "--curobo-preclose-cartesian-correction-steps",
        type=int,
        default=240,
        help="Maximum open-gripper Cartesian correction steps before a gated close.",
    )
    parser.add_argument(
        "--curobo-preclose-cartesian-correction-ee-tol",
        type=float,
        default=0.030,
        help="EE position tolerance required after Cartesian correction before closing.",
    )
    parser.add_argument(
        "--curobo-preclose-cartesian-correction-orientation-dot",
        type=float,
        default=0.985,
        help="Minimum absolute quaternion dot required after Cartesian correction before closing.",
    )
    parser.add_argument(
        "--curobo-grasp-cartesian-correction-objects",
        type=parse_object_name_set,
        default=[],
        help=(
            "Objects whose open-gripper grasp waypoint runs a measured-pose Cartesian servo before "
            "the close waypoint is allowed. If accepted, the close waypoint holds the corrected "
            "measured joint state instead of returning to the original CuRobo joint target."
        ),
    )
    parser.add_argument(
        "--curobo-grasp-cartesian-correction-steps",
        type=int,
        default=480,
        help="Maximum open-gripper Cartesian correction steps after the grasp waypoint.",
    )
    parser.add_argument(
        "--curobo-grasp-cartesian-correction-trigger-ee-tol",
        type=float,
        default=0.045,
        help="Run grasp Cartesian correction when measured grasp EE position error exceeds this value.",
    )
    parser.add_argument(
        "--curobo-grasp-cartesian-correction-ee-tol",
        type=float,
        default=0.030,
        help="EE position tolerance required after grasp Cartesian correction before close is allowed.",
    )
    parser.add_argument(
        "--curobo-grasp-cartesian-correction-orientation-dot",
        type=float,
        default=0.985,
        help="Minimum absolute quaternion dot required after grasp Cartesian correction before close is allowed.",
    )
    parser.add_argument(
        "--curobo-grasp-cartesian-correction-fail-on-reject",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort before close when the grasp Cartesian correction cannot reach its measured EE tolerance.",
    )
    parser.add_argument(
        "--cartesian-execution-objects",
        type=parse_object_name_set,
        default=[],
        help=(
            "Objects whose selected CuRobo waypoint poses should be executed with the IsaacLab "
            "CartesianController instead of the selected CuRobo joint positions. The CuRobo joint "
            "targets are kept as diagnostics only."
        ),
    )
    parser.add_argument(
        "--cartesian-execution-stages",
        type=parse_stage_name_set,
        default=["pregrasp_stage", "pregrasp", "grasp", "close", "lift"],
        help=(
            "CuRobo-annotated waypoint stages to execute with CartesianController for "
            "--cartesian-execution-objects. Close is matched from the waypoint name."
        ),
    )
    parser.add_argument(
        "--curobo-joint-execution-action-warn",
        type=float,
        default=1.0,
        help="Record when absolute action values exceed this diagnostic threshold.",
    )
    parser.add_argument(
        "--curobo-joint-execution-abort-on-fail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort execution if a CuRobo joint waypoint does not settle before continuing.",
    )
    parser.add_argument(
        "--curobo-require-action-soft-limits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject CuRobo candidates whose required stage joint targets exceed Isaac/Piper soft joint limits.",
    )
    parser.add_argument(
        "--curobo-soft-limit-exempt-objects",
        type=parse_object_name_set,
        default=[],
        help=(
            "Comma-separated object names that skip the CuRobo soft joint-limit candidate gate. "
            "Useful for thin/awkward objects where the final IK is reachable but just outside "
            "Isaac's soft-limit margin; other objects still obey --curobo-require-action-soft-limits."
        ),
    )
    parser.add_argument(
        "--curobo-soft-limit-tolerance-overrides",
        type=parse_object_float_overrides,
        default={},
        help=(
            "Per-object soft joint-limit violation tolerance in radians, e.g. box_object:0.12. "
            "This relaxes the CuRobo candidate gate without fully exempting the object."
        ),
    )
    parser.add_argument(
        "--curobo-require-action-range",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optional diagnostic gate: reject CuRobo candidates whose required stage joint targets "
            "require normalized Isaac actions outside --curobo-action-limit. This is off by default "
            "because the official Task E action config has clip=None and the baseline uses actions > 1."
        ),
    )
    parser.add_argument(
        "--curobo-action-limit",
        type=float,
        default=1.0,
        help="Absolute normalized action limit used by --curobo-require-action-range.",
    )
    parser.add_argument(
        "--curobo-baseline-action-prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record distance from CuRobo candidate action targets to the official baseline scripted action priors.",
    )
    parser.add_argument(
        "--curobo-baseline-action-prior-ranking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use baseline action-prior distance as a staged-selection tie-breaker before full CuRobo sequence solve.",
    )
    parser.add_argument(
        "--curobo-staged-selection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use a staged CuRobo selector: point-cloud geometry first, final-grasp IK/soft-limit "
            "screening second, and full pregrasp/grasp/lift IK only for the best survivors."
        ),
    )
    parser.add_argument(
        "--curobo-staged-final-ik-top-k",
        type=int,
        default=100,
        help="Maximum geometry-passing candidates to screen with final-grasp IK in staged selection.",
    )
    parser.add_argument(
        "--curobo-staged-full-ik-top-k",
        type=int,
        default=12,
        help="Maximum final-IK survivors to evaluate with full pregrasp/grasp/lift IK in staged selection.",
    )
    parser.add_argument(
        "--curobo-scene-collision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable CuRobo scene collision using voxel cuboids made from the fused scene "
            "point cloud after excluding the segmented target."
        ),
    )
    parser.add_argument("--curobo-obstacle-max-cuboids", type=int, default=48)
    parser.add_argument("--curobo-obstacle-voxel-size", type=float, default=0.045)
    parser.add_argument("--curobo-obstacle-cuboid-padding", type=float, default=0.012)
    parser.add_argument("--curobo-obstacle-min-points-per-voxel", type=int, default=6)
    parser.add_argument("--curobo-obstacle-radius", type=float, default=0.45)
    parser.add_argument("--curobo-obstacle-min-height-above-table", type=float, default=0.018)
    parser.add_argument(
        "--disable-curobo-reached-pose-pc-filter",
        action="store_true",
        help=(
            "Disable the post-IK FK point-cloud filter in curobo_feasible/curobo_first_ik. "
            "By default, the returned CuRobo grasp joint solution is FK'd and the actual "
            "reached Piper gripper pose must still contain target points in the centerline/"
            "closing region without finger/palm solid collision."
        ),
    )
    parser.add_argument(
        "--disable-ik-physics-obstacle-filter",
        action="store_true",
        help=(
            "Disable the physics obstacle-motion check inside IK candidate probing. By default, "
            "candidates are rejected if task objects move before the final grasp waypoint."
        ),
    )
    parser.add_argument(
        "--ik-obstacle-motion-tol",
        type=float,
        default=0.015,
        help="Maximum allowed pre-grasp object displacement in metres during the physics IK probe.",
    )
    parser.add_argument(
        "--pc-offset-search-objects",
        type=parse_object_name_set,
        default=[],
        help=(
            "Comma-separated object names whose selected raw AnyGrasp candidate should search "
            "the fused object point cloud for the first gripper/object collision offset."
        ),
    )
    parser.add_argument(
        "--pc-offset-step",
        type=float,
        default=0.005,
        help="Offset search step in metres. Default 0.005 m = 0.5 cm.",
    )
    parser.add_argument(
        "--pc-offset-max",
        type=float,
        default=0.09,
        help="Maximum offset distance in metres for point-cloud collision search.",
    )
    parser.add_argument(
        "--pc-offset-collision-min-points",
        type=int,
        default=5,
        help="Minimum object cloud points inside the gripper finger solids to count as collision.",
    )
    parser.add_argument(
        "--pc-offset-require-both-fingers",
        action="store_true",
        help=(
            "For point-cloud offset search, require at least --pc-offset-collision-min-points "
            "inside each finger before accepting the offset. If used with IK selection, later "
            "IK-feasible candidates are tried until one satisfies this condition."
        ),
    )
    parser.add_argument(
        "--pc-offset-collision-clearance",
        type=float,
        default=0.002,
        help="Extra metres added around the finger solid boxes during point-cloud collision testing.",
    )
    parser.add_argument(
        "--disable-approach-pc-collision-filter",
        action="store_true",
        help=(
            "Disable swept point-cloud collision filtering for pregrasp/grasp approach segments. "
            "By default, candidates with early gripper/object point-cloud collision are rejected before IK."
        ),
    )
    parser.add_argument("--approach-pc-collision-samples", type=int, default=10)
    parser.add_argument("--approach-pc-collision-final-fraction", type=float, default=0.85)
    parser.add_argument("--approach-pc-collision-min-points", type=int, default=5)
    parser.add_argument("--approach-pc-collision-clearance", type=float, default=0.002)
    parser.add_argument(
        "--disable-straight-approach-pc-search",
        action="store_true",
        help=(
            "Disable adaptive straight-line pregrasp selection from the fused object point cloud. "
            "By default, objects using point-cloud offset search also choose a longer clear "
            "pregrasp along the AnyGrasp/Piper approach axis."
        ),
    )
    parser.add_argument("--straight-approach-max-distance", type=float, default=0.24)
    parser.add_argument("--straight-approach-step", type=float, default=0.01)
    parser.add_argument("--straight-approach-stage-extra", type=float, default=0.06)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-video-sam3", action="store_true")
    parser.add_argument("--force-anygrasp", action="store_true")
    parser.add_argument("--no-sam3", action="store_true", help="Reuse existing SAM3 masks in the output directory.")
    parser.add_argument("--no-anygrasp", action="store_true", help="Reuse existing AnyGrasp outputs in the output directory.")
    parser.add_argument("--approach-axis-column", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--pregrasp-distance", type=float, default=0.12)
    parser.add_argument("--lift-distance", type=float, default=0.18)
    parser.add_argument("--init-steps", type=int, default=100)
    parser.add_argument("--staging-steps", type=int, default=140)
    parser.add_argument("--pregrasp-steps", type=int, default=120)
    parser.add_argument("--grasp-steps", type=int, default=100)
    parser.add_argument("--close-steps", type=int, default=55)
    parser.add_argument("--lift-steps", type=int, default=120)
    parser.add_argument("--transport-steps", type=int, default=210)
    parser.add_argument("--place-steps", type=int, default=100)
    parser.add_argument("--open-steps", type=int, default=70)
    parser.add_argument("--retract-steps", type=int, default=90)
    parser.add_argument(
        "--early-close-failure-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After a close waypoint, abort the remaining lift/transport/place waypoints if the "
            "gripper aperture is fully closed, then let the interleaved runner observe/replan."
        ),
    )
    parser.add_argument("--early-close-failure-aperture-threshold", type=float, default=0.008)
    parser.add_argument("--early-close-failure-target-tol", type=float, default=0.006)
    parser.add_argument(
        "--execute-after-each-object",
        action="store_true",
        help=(
            "Interleave planning and execution: video/SAM3/EE/AnyGrasp/IK for one object, "
            "execute it in the same simulator scene, then continue with the next object."
        ),
    )
    parser.add_argument(
        "--max-object-attempts",
        type=int,
        default=1,
        help=(
            "When --execute-after-each-object is enabled, re-run video/SAM3/EE "
            "AnyGrasp/execution for an object until it reaches the basket or this "
            "attempt limit is hit."
        ),
    )
    parser.add_argument(
        "--object-max-attempt-overrides",
        type=parse_object_int_overrides,
        default={},
        help=(
            "Per-object attempt limits for --execute-after-each-object, e.g. banana:5;box_object:1. "
            "Objects not listed use --max-object-attempts."
        ),
    )
    parser.add_argument(
        "--skip-laid-down-box",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip box_object immediately when its rough video-depth center is below the laid-down threshold.",
    )
    parser.add_argument(
        "--box-laid-down-center-z-threshold",
        type=float,
        default=0.875,
        help="box_object rough center z at or below this value is treated as laid down and skipped.",
    )
    parser.add_argument(
        "--stop-on-object-failure",
        action="store_true",
        help=(
            "When --execute-after-each-object is enabled, stop the remaining object order "
            "if the current object does not reach the basket within --max-object-attempts."
        ),
    )
    parser.add_argument(
        "--record-video-cam",
        action="store_true",
        help="Record the external video camera during interleaved execution as execution/video_cam.mp4.",
    )
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument("--video-every-n-steps", type=int, default=2)
    parser.add_argument("--no-save-execution-frames", action="store_true")
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


args_cli = parse_args()


def prepare_output_dir(path: Path | None) -> Path:
    if path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPO_ROOT / "outputs/task_e_full_anygrasp_ee" / timestamp
    path.mkdir(parents=True, exist_ok=True)
    latest_txt = path.parent / "latest.txt"
    latest_txt.write_text(str(path.resolve()), encoding="utf-8")
    latest_link = path.parent / "latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(path.resolve(), target_is_directory=True)
    except OSError:
        pass
    return path.resolve()


OUTPUT_DIR = prepare_output_dir(args_cli.output)


def write_failure(name: str, exc: BaseException) -> None:
    payload = {
        "error_type": type(exc).__name__,
        "error": str(exc),
        "repr": repr(exc),
        "traceback": traceback.format_exc(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (OUTPUT_DIR / name).write_text(json_text(payload), encoding="utf-8")


try:
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
except BaseException as exc:
    write_failure("launch_error.json", exc)
    raise

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import cv2  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401,E402
from atec_rl_lab.tasks.task_e.env_cfg import (  # noqa: E402
    BASKET_SUCCESS_CENTER,
    BASKET_SUCCESS_HALF_X,
    BASKET_SUCCESS_HALF_Y,
)
from atec_rl_lab.utils import CartesianController  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


RESULT_SCHEMA = "atec.task_e.motion_result.v1"
TASK_E_OBJECT_LABELS = {
    "object_1": "yellow_and_white_box",
    "object_2": "mustard_bottle",
    "object_3": "banana",
}
TASK_E_SCRIPTED_ACTUATOR = {
    "effort_limit": 100.0,
    "velocity_limit": 100.0,
    "stiffness": 800.0,
    "damping": 80.0,
}


def run_subprocess(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    print(f"[INFO] Running: {' '.join(command)}", flush=True)
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit {proc.returncode}. See {log_path}")


def apply_actuator_mode(env_cfg, mode: str) -> None:
    if mode == "default":
        return
    if mode != "task_e_scripted_high_stiffness":
        raise ValueError(f"Unsupported actuator mode: {mode!r}")
    env_cfg.scene.robot.actuators["default"] = ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        effort_limit=TASK_E_SCRIPTED_ACTUATOR["effort_limit"],
        velocity_limit=TASK_E_SCRIPTED_ACTUATOR["velocity_limit"],
        stiffness=TASK_E_SCRIPTED_ACTUATOR["stiffness"],
        damping=TASK_E_SCRIPTED_ACTUATOR["damping"],
    )


def to_numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def to_rgb_array(value) -> np.ndarray:
    array = to_numpy(value)
    if array.ndim == 4:
        array = array[0]
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        if array.size and float(np.nanmax(array)) <= 1.0:
            array *= 255.0
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def to_depth_array(value) -> np.ndarray:
    array = to_numpy(value).astype(np.float32)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError(f"Expected depth image, got {array.shape}")
    return np.ascontiguousarray(array)


def save_depth_preview(path: Path, depth: np.ndarray, max_depth: float = 2.0) -> None:
    valid = np.isfinite(depth) & (depth > 0.0)
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        clipped = np.clip(depth, 0.0, max_depth)
        scaled = (255.0 * (1.0 - clipped / max_depth)).astype(np.uint8)
        scaled[~valid] = 0
    Image.fromarray(scaled, mode="L").save(path)


def save_camera_frame(output_dir: Path, prefix: str, obs_image: dict) -> dict:
    rgb = to_rgb_array(obs_image[f"{prefix}_rgb"])
    depth = to_depth_array(obs_image[f"{prefix}_depth"])
    Image.fromarray(rgb, mode="RGB").save(output_dir / f"{prefix}_rgb.png")
    np.save(output_dir / f"{prefix}_depth.npy", depth)
    save_depth_preview(output_dir / f"{prefix}_depth_preview.png", depth, max_depth=args_cli.max_depth)
    return {
        "rgb": f"{prefix}_rgb.png",
        "depth_npy": f"{prefix}_depth.npy",
        "depth_preview": f"{prefix}_depth_preview.png",
        "rgb_shape": list(rgb.shape),
        "depth_shape": list(depth.shape),
    }


def save_execution_frame(path: Path, obs: dict, label: str) -> None:
    video = Image.fromarray(to_rgb_array(obs["image"]["video_rgb"]), mode="RGB")
    ee = Image.fromarray(to_rgb_array(obs["image"]["ee_rgb"]), mode="RGB")
    tile_w = max(video.width, ee.width)
    canvas = Image.new("RGB", (tile_w * 2 + 12, max(video.height, ee.height) + 28), (18, 22, 30))
    canvas.paste(video, (0, 28))
    canvas.paste(ee, (tile_w + 12, 28))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 26), fill=(0, 0, 0))
    draw.text((8, 7), f"{label} | left=video_cam right=ee_camera", fill=(255, 255, 255))
    canvas.save(path)


class VideoCameraRecorder:
    def __init__(self, path: Path, fps: float, every_n_steps: int) -> None:
        self.path = path
        self.fps = float(fps)
        self.every_n_steps = max(1, int(every_n_steps))
        self.writer = None
        self.step = 0
        self.frames = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, obs: dict, force: bool = False) -> None:
        if not force and self.step % self.every_n_steps != 0:
            self.step += 1
            return
        rgb = to_rgb_array(obs["image"]["video_rgb"])
        if self.writer is None:
            height, width = rgb.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not open video writer: {self.path}")
        self.writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.frames += 1
        self.step += 1

    def close(self) -> dict:
        if self.writer is not None:
            self.writer.release()
        return {
            "file": str(self.path),
            "fps": self.fps,
            "every_n_steps": self.every_n_steps,
            "frames": self.frames,
        }


def tensor_to_list(value) -> list[float]:
    return value.detach().cpu().numpy().astype(float).tolist()


def camera_metadata(env, sensor_name: str) -> dict:
    camera = env.unwrapped.scene.sensors[sensor_name]
    data = camera.data
    return {
        "sensor": sensor_name,
        "image_shape": list(data.image_shape),
        "intrinsic_matrix": tensor_to_list(data.intrinsic_matrices[0]),
        "pos_w": tensor_to_list(data.pos_w[0]),
        "quat_w_world": tensor_to_list(data.quat_w_world[0]),
        "quat_w_ros": tensor_to_list(data.quat_w_ros[0]),
        "quat_w_opengl": tensor_to_list(data.quat_w_opengl[0]),
    }


def find_single_body(robot, body_name: str) -> int:
    body_ids, body_names = robot.find_bodies(body_name)
    if len(body_ids) != 1:
        raise RuntimeError(f"Expected one body named {body_name!r}, found {len(body_ids)}: {body_names}")
    return int(body_ids[0])


def calibrate_ee_camera(env, robot, ee_idx: int) -> dict:
    sensor_camera = camera_metadata(env, "ee_camera")
    grip_pose_w = robot.data.body_pose_w[0, ee_idx]
    grip_pos_w = np.asarray(tensor_to_list(grip_pose_w[:3]), dtype=np.float64)
    grip_quat_w = np.asarray(tensor_to_list(grip_pose_w[3:]), dtype=np.float64)
    cam_pos_w = np.asarray(sensor_camera["pos_w"], dtype=np.float64)
    cam_quat_w_ros = np.asarray(sensor_camera["quat_w_ros"], dtype=np.float64)
    grip_to_cam_pos = quat_wxyz_to_matrix(grip_quat_w).T @ (cam_pos_w - grip_pos_w)
    grip_to_cam_quat = quat_multiply(quat_inverse(grip_quat_w), cam_quat_w_ros)
    return {
        "source": "reset_sensor_pose_and_gripper_body_pose",
        "ee_body_idx": ee_idx,
        "reset_camera_sensor": sensor_camera,
        "reset_gripper_pos_w": grip_pos_w.tolist(),
        "reset_gripper_quat_wxyz": grip_quat_w.tolist(),
        "gripper_to_camera_pos": grip_to_cam_pos.tolist(),
        "gripper_to_camera_quat_ros": grip_to_cam_quat.tolist(),
    }


def ee_camera_metadata_from_gripper(env, robot, ee_idx: int, calibration: dict) -> dict:
    metadata = camera_metadata(env, "ee_camera")
    grip_pose_w = robot.data.body_pose_w[0, ee_idx]
    grip_pos_w = np.asarray(tensor_to_list(grip_pose_w[:3]), dtype=np.float64)
    grip_quat_w = np.asarray(tensor_to_list(grip_pose_w[3:]), dtype=np.float64)
    grip_to_cam_pos = np.asarray(calibration["gripper_to_camera_pos"], dtype=np.float64)
    grip_to_cam_quat = np.asarray(calibration["gripper_to_camera_quat_ros"], dtype=np.float64)
    cam_pos_w = grip_pos_w + quat_wxyz_to_matrix(grip_quat_w) @ grip_to_cam_pos
    cam_quat_w_ros = quat_multiply(grip_quat_w, grip_to_cam_quat)
    metadata["pos_w_sensor_raw"] = metadata["pos_w"]
    metadata["quat_w_ros_sensor_raw"] = metadata["quat_w_ros"]
    metadata["pos_w"] = cam_pos_w.astype(float).tolist()
    metadata["quat_w_ros"] = cam_quat_w_ros.astype(float).tolist()
    metadata["pose_source"] = "gripper_body_pose_plus_reset_camera_calibration"
    metadata["calibration"] = calibration
    return metadata


def write_camera_json(path: Path, camera: dict) -> None:
    path.write_text(json_text({"camera": camera}), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L")
    width, height = shape[1], shape[0]
    if mask.size != (width, height):
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(mask) > 127


def quat_normalize(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def quat_inverse(quat: np.ndarray) -> np.ndarray:
    quat = quat_normalize(quat)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = quat_normalize(left)
    w2, x2, y2, z2 = quat_normalize(right)
    return quat_normalize(
        np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )
    )


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_wxyz_to_xyzw(quat: list[float] | np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return np.asarray([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)


def backproject(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    z = depth.astype(np.float32)
    x = (xx - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (yy - intrinsic[1, 2]) * z / intrinsic[1, 1]
    return np.stack([x, y, z], axis=-1)


def transform_points_to_world(points_cam: np.ndarray, camera: dict) -> np.ndarray:
    rot = quat_wxyz_to_matrix(np.asarray(camera["quat_w_ros"], dtype=np.float64))
    pos = np.asarray(camera["pos_w"], dtype=np.float64)
    return points_cam @ rot.T + pos


def estimate_pose_from_mask(depth: np.ndarray, mask: np.ndarray, camera: dict, max_depth: float) -> dict:
    intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float32)
    points_cam_all = backproject(depth, intrinsic)
    valid = mask & np.isfinite(depth) & (depth > 0.0) & (depth < max_depth)
    if not valid.any():
        raise ValueError("Mask has no valid depth pixels.")

    yy, xx = np.where(valid)
    points_world = transform_points_to_world(points_cam_all[valid], camera)
    center_median = np.median(points_world, axis=0)
    center_mean = np.mean(points_world, axis=0)
    return {
        "center_world": center_median.astype(float).tolist(),
        "center_world_median": center_median.astype(float).tolist(),
        "center_world_mean": center_mean.astype(float).tolist(),
        "point_count": int(points_world.shape[0]),
        "pixel_center": [float(np.median(xx)), float(np.median(yy))],
        "bbox_xyxy": [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())],
    }


def draw_target_overlay(image_path: Path, output_path: Path, pose: dict, title: str) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    bbox = pose.get("bbox_xyxy")
    if bbox is not None:
        draw.rectangle(tuple(bbox), outline=(255, 60, 60), width=3)
    center = pose.get("pixel_center")
    if center is not None:
        x, y = center
        draw.line((x - 14, y, x + 14, y), fill=(255, 255, 0), width=3)
        draw.line((x, y - 14, x, y + 14), fill=(255, 255, 0), width=3)
    center_world = pose.get("center_world_median") or pose.get("center_world")
    text = title
    if center_world is not None:
        text += f" | world {np.round(center_world, 3).tolist()}"
    draw.rectangle((0, 0, image.width, 30), fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255))
    image.save(output_path)


def color_candidate_mask(rgb: np.ndarray, object_name: str) -> np.ndarray:
    rgb_i = rgb.astype(np.int16)
    r, g, b = rgb_i[..., 0], rgb_i[..., 1], rgb_i[..., 2]
    yellow = (r > 145) & (g > 105) & (b < 135) & ((r - b) > 55) & ((g - b) > 35)
    white = (r > 165) & (g > 165) & (b > 145) & (np.abs(r - g) < 45) & (np.abs(g - b) < 65)
    dark_table = (r < 95) & (g < 95) & (b < 95)

    if object_name == "box_object":
        mask = (yellow | white) & ~dark_table
    else:
        mask = yellow & ~dark_table
    return mask.astype(np.uint8)


def select_centered_component(mask: np.ndarray) -> np.ndarray:
    try:
        import cv2

        kernel = np.ones((5, 5), dtype=np.uint8)
        cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, 8)
        if count <= 1:
            return cleaned.astype(bool)

        height, width = mask.shape
        image_center = np.array([width * 0.5, height * 0.5], dtype=np.float64)
        best_idx = None
        best_score = -np.inf
        for idx in range(1, count):
            area = float(stats[idx, cv2.CC_STAT_AREA])
            if area < 80.0:
                continue
            centroid = np.asarray(centroids[idx], dtype=np.float64)
            distance = float(np.linalg.norm(centroid - image_center))
            score = area / ((1.0 + distance / 90.0) ** 2)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            return cleaned.astype(bool)
        return labels == best_idx
    except Exception:
        return mask.astype(bool)


def save_mask_overlay(image: Image.Image, mask: np.ndarray, output_path: Path, label: str, view_label: str) -> None:
    base = image.convert("RGBA")
    layer = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    layer[mask] = (255, 204, 0, 95)
    base = Image.alpha_composite(base, Image.fromarray(layer, mode="RGBA"))
    draw = ImageDraw.Draw(base)
    ys, xs = np.where(mask)
    if xs.size:
        draw.rectangle((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), outline=(255, 60, 60), width=3)
    draw.rectangle((0, 0, image.width, 38), fill=(0, 0, 0, 180))
    draw.text((8, 10), f"{label} color mask | {view_label}", fill=(255, 255, 255, 255))
    base.convert("RGB").save(output_path)


def run_color_mask(image_path: Path, output_dir: Path, object_name: str, label: str, view_label: str) -> tuple[Path, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image)
    mask = select_centered_component(color_candidate_mask(rgb, object_name))
    mask_path = output_dir / f"{label}_mask.png"
    overlay_path = output_dir / f"{label}_overlay.png"
    detection_path = output_dir / f"{label}_detections.json"
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
    save_mask_overlay(image, mask, overlay_path, label, view_label)
    ys, xs = np.where(mask)
    payload = {
        "image": str(image_path.resolve()),
        "prompt": f"{object_name}_color_mask",
        "label": label,
        "view_label": view_label,
        "device": "cpu_numpy",
        "source": "color_threshold_centered_component",
        "mask": mask_path.name,
        "overlay": overlay_path.name,
        "mask_count": 1 if xs.size else 0,
        "best_index": 0 if xs.size else None,
        "areas_px": [int(xs.size)] if xs.size else [],
        "boxes_xyxy": [[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]] if xs.size else [],
        "scores": [None] if xs.size else [],
    }
    detection_path.write_text(json_text(payload), encoding="utf-8")
    return mask_path, payload


def select_color_component_overlapping_guide(color_mask: np.ndarray, guide_mask: np.ndarray) -> np.ndarray | None:
    try:
        import cv2

        kernel = np.ones((3, 3), dtype=np.uint8)
        cleaned = cv2.morphologyEx(color_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
        best_idx = None
        best_overlap = 0
        best_area = 0
        for idx in range(1, count):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < 50:
                continue
            component = labels == idx
            overlap = int(np.count_nonzero(component & guide_mask))
            if overlap > best_overlap or (overlap == best_overlap and overlap > 0 and area > best_area):
                best_idx = idx
                best_overlap = overlap
                best_area = area
        if best_idx is None or best_overlap < 20:
            return None
        return labels == best_idx
    except Exception:
        overlap = color_mask.astype(bool) & guide_mask
        return overlap if np.count_nonzero(overlap) >= 20 else None


def refine_video_pose_with_color(
    image_path: Path,
    depth: np.ndarray,
    camera: dict,
    guide_mask: np.ndarray,
    object_name: str,
    output_dir: Path,
    label: str,
) -> tuple[np.ndarray | None, dict | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image)
    component = select_color_component_overlapping_guide(color_candidate_mask(rgb, object_name), guide_mask)
    if component is None:
        return None, None

    mask_path = output_dir / f"{label}_mask.png"
    overlay_path = output_dir / f"{label}_overlay.png"
    detection_path = output_dir / f"{label}_detections.json"
    Image.fromarray(component.astype(np.uint8) * 255, mode="L").save(mask_path)
    save_mask_overlay(image, component, overlay_path, label, "eye-to-hand / video_cam color refinement")
    pose = estimate_pose_from_mask(depth, component, camera, args_cli.max_depth)
    pose["source"] = "video_cam_color_component_overlapping_sam3_depth"
    draw_target_overlay(
        image_path,
        output_dir / f"{label}_target_overlay.png",
        pose,
        f"{object_name} color-refined video target",
    )
    ys, xs = np.where(component)
    payload = {
        "image": str(image_path.resolve()),
        "prompt": f"{object_name}_video_color_refine",
        "label": label,
        "view_label": "eye-to-hand / video_cam color refinement",
        "device": "cpu_numpy",
        "source": "color_component_overlapping_sam3",
        "mask": mask_path.name,
        "overlay": overlay_path.name,
        "target_overlay": f"{label}_target_overlay.png",
        "mask_count": 1,
        "best_index": 0,
        "areas_px": [int(xs.size)],
        "boxes_xyxy": [[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]],
        "scores": [None],
        "pose_estimate": pose,
    }
    detection_path.write_text(json_text(payload), encoding="utf-8")
    return component, payload


def run_sam3(image_path: Path, output_dir: Path, object_name: str, label: str, view_label: str) -> tuple[Path, dict]:
    mask_path = output_dir / f"{label}_mask.png"
    detection_path = output_dir / f"{label}_detections.json"
    if args_cli.no_sam3:
        if not mask_path.exists() or not detection_path.exists():
            raise FileNotFoundError(f"Missing existing SAM3 output: {mask_path}")
        return mask_path, load_json(detection_path)
    if args_cli.force or args_cli.force_video_sam3 or not mask_path.exists() or not detection_path.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "conda",
            "run",
            "-n",
            args_cli.sam3_env,
            "python",
            "scripts/sam3_single_image_mask.py",
            "--image",
            str(image_path),
            "--prompt",
            OBJECT_PROMPTS[object_name],
            "--label",
            label,
            "--view-label",
            view_label,
            "--output",
            str(output_dir),
            "--device",
            args_cli.sam3_device,
        ]
        run_subprocess(command, output_dir / f"{label}_sam3.log")
    return mask_path, load_json(detection_path)


def desired_gripper_pose_for_camera(
    env,
    robot,
    ee_idx: int,
    desired_camera_pos_w: list[float],
    calibration: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if calibration is None:
        camera = camera_metadata(env, "ee_camera")
    else:
        camera = ee_camera_metadata_from_gripper(env, robot, ee_idx, calibration)
    cam_pos_w = np.asarray(camera["pos_w"], dtype=np.float64)
    cam_quat_w_ros = np.asarray(camera["quat_w_ros"], dtype=np.float64)
    grip_pose_w = robot.data.body_pose_w[0, ee_idx]
    grip_pos_w = np.asarray(tensor_to_list(grip_pose_w[:3]), dtype=np.float64)
    grip_quat_w = np.asarray(tensor_to_list(grip_pose_w[3:]), dtype=np.float64)

    if calibration is None:
        grip_to_cam_quat = quat_multiply(quat_inverse(grip_quat_w), cam_quat_w_ros)
        grip_to_cam_pos = quat_wxyz_to_matrix(grip_quat_w).T @ (cam_pos_w - grip_pos_w)
    else:
        grip_to_cam_quat = np.asarray(calibration["gripper_to_camera_quat_ros"], dtype=np.float64)
        grip_to_cam_pos = np.asarray(calibration["gripper_to_camera_pos"], dtype=np.float64)
    desired_grip_quat = np.asarray(TOP_DOWN_GRIPPER_QUAT_WXYZ, dtype=np.float64)
    desired_cam_quat = quat_multiply(desired_grip_quat, grip_to_cam_quat)
    desired_grip_pos = (
        np.asarray(desired_camera_pos_w, dtype=np.float64)
        - quat_wxyz_to_matrix(desired_grip_quat) @ grip_to_cam_pos
    )
    return desired_grip_pos, desired_grip_quat, {
        "current_camera_pos_w": cam_pos_w.tolist(),
        "current_camera_quat_w_ros": cam_quat_w_ros.tolist(),
        "current_gripper_pos_w": grip_pos_w.tolist(),
        "current_gripper_quat_wxyz": grip_quat_w.tolist(),
        "gripper_to_camera_pos": grip_to_cam_pos.tolist(),
        "gripper_to_camera_quat_ros": grip_to_cam_quat.tolist(),
        "desired_camera_pos_w": list(desired_camera_pos_w),
        "desired_camera_quat_w_ros": desired_cam_quat.tolist(),
        "desired_gripper_pos_w": desired_grip_pos.tolist(),
        "desired_gripper_quat_wxyz": desired_grip_quat.tolist(),
    }


def camera_topdown_alignment(camera: dict) -> dict:
    rot = quat_wxyz_to_matrix(np.asarray(camera["quat_w_ros"], dtype=np.float64))
    optical_axis_w = rot[:, 2]
    desired_axis_w = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    dot = float(np.dot(optical_axis_w, desired_axis_w))
    dot = max(-1.0, min(1.0, dot))
    angle_deg = float(np.degrees(np.arccos(dot)))
    return {
        "optical_axis_w": optical_axis_w.astype(float).tolist(),
        "desired_optical_axis_w": desired_axis_w.astype(float).tolist(),
        "topdown_dot": dot,
        "angle_error_deg": angle_deg,
    }


def normalize_or_fallback(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        return np.asarray(fallback, dtype=np.float64)
    return np.asarray(value, dtype=np.float64) / norm


def lookat_camera_quat_wxyz(camera_pos_w: np.ndarray, target_pos_w: np.ndarray) -> np.ndarray:
    optical_axis = normalize_or_fallback(
        target_pos_w - camera_pos_w,
        np.array([-1.0, 0.0, -1.0], dtype=np.float64),
    )
    image_down_hint = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    image_down = image_down_hint - optical_axis * float(np.dot(image_down_hint, optical_axis))
    if np.linalg.norm(image_down) <= 1e-6:
        image_down_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        image_down = image_down_hint - optical_axis * float(np.dot(image_down_hint, optical_axis))
    image_down = normalize_or_fallback(image_down, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    image_right = normalize_or_fallback(np.cross(image_down, optical_axis), np.array([1.0, 0.0, 0.0]))
    image_down = normalize_or_fallback(np.cross(optical_axis, image_right), image_down)
    rot = np.stack([image_right, image_down, optical_axis], axis=1)
    return np.asarray(quat_wxyz_from_matrix(rot), dtype=np.float64)


def gripper_pose_for_camera_pose(
    desired_camera_pos_w: np.ndarray,
    desired_camera_quat_w_ros: np.ndarray,
    ee_camera_calibration: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    grip_to_cam_pos = np.asarray(ee_camera_calibration["gripper_to_camera_pos"], dtype=np.float64)
    grip_to_cam_quat = np.asarray(ee_camera_calibration["gripper_to_camera_quat_ros"], dtype=np.float64)
    desired_grip_quat = quat_multiply(desired_camera_quat_w_ros, quat_inverse(grip_to_cam_quat))
    desired_grip_pos = desired_camera_pos_w - quat_wxyz_to_matrix(desired_grip_quat) @ grip_to_cam_pos
    return desired_grip_pos, desired_grip_quat, {
        "desired_camera_pos_w": desired_camera_pos_w.astype(float).tolist(),
        "desired_camera_quat_w_ros": desired_camera_quat_w_ros.astype(float).tolist(),
        "desired_gripper_pos_w": desired_grip_pos.astype(float).tolist(),
        "desired_gripper_quat_wxyz": desired_grip_quat.astype(float).tolist(),
        "gripper_to_camera_pos": grip_to_cam_pos.astype(float).tolist(),
        "gripper_to_camera_quat_ros": grip_to_cam_quat.astype(float).tolist(),
    }


def project_world_point(camera: dict, point_w: np.ndarray) -> dict:
    intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float64)
    rot_wc = quat_wxyz_to_matrix(np.asarray(camera["quat_w_ros"], dtype=np.float64))
    cam_pos = np.asarray(camera["pos_w"], dtype=np.float64)
    point_cam = rot_wc.T @ (np.asarray(point_w, dtype=np.float64) - cam_pos)
    z = float(point_cam[2])
    if z <= 1e-6:
        return {"ok": False, "point_cam": point_cam.astype(float).tolist(), "reason": "behind_camera"}
    pixel = np.array(
        [
            intrinsic[0, 0] * point_cam[0] / z + intrinsic[0, 2],
            intrinsic[1, 1] * point_cam[1] / z + intrinsic[1, 2],
        ],
        dtype=np.float64,
    )
    return {
        "ok": True,
        "point_cam": point_cam.astype(float).tolist(),
        "pixel": pixel.astype(float).tolist(),
        "depth_m": z,
    }


def camera_centering_shift(camera: dict, point_w: np.ndarray, target_pixel: list[float]) -> tuple[np.ndarray, dict]:
    projection = project_world_point(camera, point_w)
    if not projection.get("ok"):
        return np.zeros(3, dtype=np.float64), {"projection": projection, "shift_w": [0.0, 0.0, 0.0]}

    intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float64)
    rot_wc = quat_wxyz_to_matrix(np.asarray(camera["quat_w_ros"], dtype=np.float64))
    point_cam = np.asarray(projection["point_cam"], dtype=np.float64)
    target_u, target_v = [float(v) for v in target_pixel[:2]]
    target_cam = np.array(
        [
            (target_u - intrinsic[0, 2]) * point_cam[2] / intrinsic[0, 0],
            (target_v - intrinsic[1, 2]) * point_cam[2] / intrinsic[1, 1],
            0.0,
        ],
        dtype=np.float64,
    )
    shift_cam = np.array([point_cam[0], point_cam[1], 0.0], dtype=np.float64) - target_cam
    shift_w = rot_wc @ shift_cam
    shift_w *= float(args_cli.hover_center_gain)
    max_shift = float(args_cli.hover_center_max_shift)
    norm = float(np.linalg.norm(shift_w))
    if max_shift > 0.0 and norm > max_shift:
        shift_w *= max_shift / norm
    projection["target_pixel"] = [target_u, target_v]
    projection["shift_cam"] = shift_cam.astype(float).tolist()
    projection["shift_w"] = shift_w.astype(float).tolist()
    return shift_w, {"projection": projection, "shift_w": shift_w.astype(float).tolist()}


def move_to_topdown_camera(
    env,
    obs,
    robot,
    target_object_center_w: list[float],
    ee_camera_calibration: dict,
    object_name: str | None = None,
    look_at_target_w: list[float] | None = None,
    camera_position_offset_w: list[float] | None = None,
    forced_hover_mode: str | None = None,
) -> tuple[dict, dict]:
    arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
    gripper_ids, _ = robot.find_joints(GRIPPER_JOINT_NAMES)
    controller = CartesianController(
        robot=robot,
        ee_body_name="gripper_base",
        arm_joint_names=ARM_JOINT_NAMES,
        num_envs=1,
        device=env.unwrapped.device,
        command_type="pose",
        lambda_val=0.05,
        max_joint_delta=0.18,
    )
    controller.reset()
    gripper_t = torch.tensor([GRIPPER_OPEN], dtype=torch.float32, device=env.unwrapped.device)
    default_jpos = robot.data.default_joint_pos.clone()
    last_action = None

    requested_mode = forced_hover_mode or args_cli.hover_mode
    if requested_mode == "adaptive_lookat":
        effective_mode = "look_at" if object_name in set(args_cli.lookat_objects) else "gripper_offset"
    else:
        effective_mode = requested_mode

    look_at_target = (
        np.asarray(look_at_target_w, dtype=np.float64)
        if look_at_target_w is not None
        else np.asarray(target_object_center_w, dtype=np.float64)
    )
    view_offset = np.zeros(3, dtype=np.float64)
    if camera_position_offset_w is not None:
        view_offset_values = list(camera_position_offset_w[:3])
        if len(view_offset_values) < 3:
            view_offset_values.extend([0.0] * (3 - len(view_offset_values)))
        view_offset = np.asarray(view_offset_values, dtype=np.float64)

    def desired_pose_for_camera(desired_camera_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        if effective_mode == "camera_center":
            return desired_gripper_pose_for_camera(
                env,
                robot,
                controller.ee_idx,
                desired_camera_pos.astype(float).tolist(),
                ee_camera_calibration,
            )
        if effective_mode == "look_at":
            desired_camera_quat = lookat_camera_quat_wxyz(desired_camera_pos, look_at_target)
            return gripper_pose_for_camera_pose(desired_camera_pos, desired_camera_quat, ee_camera_calibration)

        grip_quat = np.asarray(TOP_DOWN_GRIPPER_QUAT_WXYZ, dtype=np.float64)
        grip_to_cam_pos = np.asarray(ee_camera_calibration["gripper_to_camera_pos"], dtype=np.float64)
        grip_to_cam_quat = np.asarray(ee_camera_calibration["gripper_to_camera_quat_ros"], dtype=np.float64)
        grip_pos = desired_camera_pos - quat_wxyz_to_matrix(grip_quat) @ grip_to_cam_pos
        return grip_pos, grip_quat, {
            "desired_camera_pos_w": desired_camera_pos.astype(float).tolist(),
            "desired_camera_quat_w_ros": quat_multiply(grip_quat, grip_to_cam_quat).tolist(),
            "desired_gripper_pos_w": grip_pos.astype(float).tolist(),
            "desired_gripper_quat_wxyz": grip_quat.astype(float).tolist(),
            "gripper_to_camera_pos": grip_to_cam_pos.astype(float).tolist(),
            "gripper_to_camera_quat_ros": grip_to_cam_quat.astype(float).tolist(),
        }

    def command_gripper_pose(grip_pos: np.ndarray, grip_quat: np.ndarray, steps: int) -> dict | None:
        nonlocal obs, last_action
        pos_t = torch.tensor([grip_pos], dtype=torch.float32, device=env.unwrapped.device)
        quat_t = torch.tensor([grip_quat], dtype=torch.float32, device=env.unwrapped.device)
        stopped = None
        for _ in range(max(1, int(steps))):
            arm_des = controller.compute(pos_t, quat_t)
            target = robot.data.joint_pos.clone()
            target[:, arm_ids] = arm_des
            target[:, gripper_ids] = gripper_t
            action = (target - default_jpos) / ACTION_SCALE
            last_action = action
            obs, _, terminated, truncated, _ = env.step(action)
            robot.update(dt=env.unwrapped.physics_dt)
            if terminated.any() or truncated.any():
                stopped = {
                    "terminated": bool(terminated.any().item()),
                    "truncated": bool(truncated.any().item()),
                }
                break
        return stopped

    base_camera_pos = np.asarray(target_object_center_w, dtype=np.float64)
    if effective_mode == "look_at":
        if args_cli.lookat_camera_position:
            base_camera_pos = np.asarray(args_cli.lookat_camera_position[:3], dtype=np.float64)
        else:
            base_camera_pos = np.asarray(ee_camera_calibration["reset_camera_sensor"]["pos_w"], dtype=np.float64)
        lookat_offset = list(args_cli.lookat_camera_offset[:3])
        if len(lookat_offset) < 3:
            lookat_offset.extend([0.0] * (3 - len(lookat_offset)))
        base_camera_pos = base_camera_pos + np.asarray(lookat_offset, dtype=np.float64) + view_offset
    elif effective_mode == "gripper_offset":
        offset = list(args_cli.gripper_hover_offset[:3])
        if len(offset) < 3:
            offset.extend([0.0] * (3 - len(offset)))
        base_camera_pos = base_camera_pos + np.asarray(offset, dtype=np.float64) + view_offset
    else:
        base_camera_pos = base_camera_pos + view_offset

    grip_pos_des, grip_quat_des, calibration = desired_pose_for_camera(base_camera_pos)
    stop_reason = command_gripper_pose(grip_pos_des, grip_quat_des, args_cli.hover_settle_steps)

    centering_iterations = []
    desired_camera_pos_for_centering = np.asarray(calibration["desired_camera_pos_w"], dtype=np.float64)
    for iteration in range(max(0, int(args_cli.hover_center_iters))):
        camera_now = ee_camera_metadata_from_gripper(env, robot, controller.ee_idx, ee_camera_calibration)
        shift_w, shift_record = camera_centering_shift(camera_now, look_at_target, args_cli.hover_center_pixel)
        shift_norm = float(np.linalg.norm(shift_w))
        shift_record["iteration"] = iteration + 1
        shift_record["before_camera_pos_w"] = camera_now["pos_w"]
        centering_iterations.append(shift_record)
        if shift_norm <= 1e-4:
            break
        desired_camera_pos_for_centering = desired_camera_pos_for_centering + shift_w
        grip_pos_des, grip_quat_des, calibration = desired_pose_for_camera(desired_camera_pos_for_centering)
        stop_reason = command_gripper_pose(grip_pos_des, grip_quat_des, args_cli.hover_center_steps)

    ee_pose = robot.data.body_pose_w[0, controller.ee_idx]
    camera = ee_camera_metadata_from_gripper(env, robot, controller.ee_idx, ee_camera_calibration)
    desired_camera_final = np.asarray(calibration["desired_camera_pos_w"], dtype=np.float64)
    camera_position_error = np.asarray(camera["pos_w"], dtype=np.float64) - desired_camera_final
    gripper_position_error = np.asarray(tensor_to_list(ee_pose[:3]), dtype=np.float64) - np.asarray(grip_pos_des, dtype=np.float64)
    if effective_mode in {"camera_center", "look_at"}:
        position_error = camera_position_error
    else:
        position_error = gripper_position_error
    topdown_alignment = camera_topdown_alignment(camera)
    record = {
        "mode": (
            "cartesian_ik_ee_camera_look_at"
            if effective_mode == "look_at"
            else "cartesian_ik_ee_camera_center_topdown"
            if effective_mode == "camera_center"
            else "cartesian_ik_reachable_gripper_topdown_with_camera_offset"
        ),
        "requested_mode": requested_mode,
        "effective_mode": effective_mode,
        "object_name": object_name,
        "target_object_center_w": target_object_center_w,
        "look_at_target_w": look_at_target.astype(float).tolist(),
        "desired_camera_pos_w": calibration["desired_camera_pos_w"],
        "desired_camera_quat_w_ros": calibration["desired_camera_quat_w_ros"],
        "desired_gripper_pos_w": grip_pos_des.astype(float).tolist(),
        "desired_gripper_quat_wxyz": grip_quat_des.astype(float).tolist(),
        "final_ee_pos_w": tensor_to_list(ee_pose[:3]),
        "final_ee_quat_wxyz": tensor_to_list(ee_pose[3:]),
        "final_camera_pos_w": camera["pos_w"],
        "final_camera_quat_w_ros": camera["quat_w_ros"],
        "camera_position_error_m": float(np.linalg.norm(camera_position_error)),
        "gripper_position_error_m": float(np.linalg.norm(gripper_position_error)),
        "position_error_m": float(np.linalg.norm(position_error)),
        "camera_topdown_alignment": topdown_alignment,
        "settle_steps": args_cli.hover_settle_steps,
        "center_steps": args_cli.hover_center_steps,
        "center_iterations": centering_iterations,
        "stop_reason": stop_reason,
        "last_action": tensor_to_list(last_action[0]) if last_action is not None else None,
        "gripper_hover_offset": list(args_cli.gripper_hover_offset),
        "lookat_camera_offset": list(args_cli.lookat_camera_offset),
        "camera_position_offset_w": view_offset.astype(float).tolist(),
        "ee_camera_calibration": ee_camera_calibration,
    }
    if calibration is not None:
        record["camera_to_gripper_calibration"] = calibration
    return obs, record


def run_anygrasp(
    capture_dir: Path,
    mask_path: Path,
    camera_json: Path,
    object_name: str,
    extra_views: list[dict] | None = None,
    object_center_w: list[float] | None = None,
    heuristic_seed: int | None = None,
) -> tuple[Path, dict]:
    anygrasp_dir = capture_dir / "anygrasp"
    pose_path = anygrasp_dir / "final_grasp_pose.json"
    result_path = anygrasp_dir / "anygrasp_result.json"
    if args_cli.no_anygrasp:
        if not pose_path.exists() or not result_path.exists():
            raise FileNotFoundError(f"Missing existing AnyGrasp output for {object_name}: {pose_path}")
        return pose_path, load_json(result_path)
    if args_cli.force or args_cli.force_anygrasp or not pose_path.exists() or not result_path.exists():
        anygrasp_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        openssl_lib = REPO_ROOT / "third_party/openssl11/lib"
        current_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{openssl_lib}:{current_ld}" if current_ld else str(openssl_lib)
        grasp_generator = args_cli.grasp_generator_overrides.get(object_name, args_cli.grasp_generator)
        if grasp_generator == "contact_graspnet":
            generator_env = args_cli.contact_graspnet_env
            generator_script = "scripts/contact_graspnet_from_rgbd_mask.py"
        elif grasp_generator == "graspgen":
            generator_env = args_cli.graspgen_env
            generator_script = "scripts/graspgen_from_rgbd_mask.py"
        elif grasp_generator == "heuristic":
            generator_env = None
            generator_script = "scripts/heuristic_from_rgbd_mask.py"
        else:
            generator_env = args_cli.anygrasp_env
            generator_script = "scripts/anygrasp_from_rgbd_mask.py"
        if generator_env is None:
            command = [sys.executable, generator_script]
        else:
            command = ["conda", "run", "-n", generator_env, "python", generator_script]
        saved_top_grasps = int(args_cli.save_top_grasps)
        if args_cli.ik_filter_top_k > 0:
            saved_top_grasps = max(saved_top_grasps, int(args_cli.ik_filter_top_k))
        command.extend(
            [
                "--rgb",
                str(capture_dir / "ee_rgb.png"),
                "--depth-npy",
                str(capture_dir / "ee_depth.npy"),
                "--mask",
                str(mask_path),
                "--camera-json",
                str(camera_json),
                "--output",
                str(anygrasp_dir),
                "--max-depth",
                str(args_cli.max_depth),
                "--save-top-grasps",
                str(saved_top_grasps),
            ]
        )
        if grasp_generator == "anygrasp":
            command.extend(
                [
                    "--overlay-tool-transform",
                    args_cli.raw_tool_transform,
                    "--overlay-gripper-base-offset",
                    str(args_cli.gripper_base_offset),
                    "--overlay-gripper-base-offset-mode",
                    args_cli.gripper_base_offset_mode,
                    "--symmetric-cloud-mode",
                    args_cli.anygrasp_symmetric_cloud_mode,
                    "--symmetry-center-source",
                    args_cli.anygrasp_symmetry_center_source,
                    "--symmetric-surface-points",
                    str(args_cli.anygrasp_symmetric_surface_points),
                ]
            )
        elif grasp_generator == "contact_graspnet":
            command.extend(
                [
                    "--contact-graspnet-root",
                    str(args_cli.contact_graspnet_root),
                    "--ckpt-dir",
                    str(args_cli.contact_graspnet_ckpt_dir),
                    "--forward-passes",
                    str(args_cli.contact_graspnet_forward_passes),
                    "--scene-cloud-stride",
                    str(args_cli.anygrasp_scene_cloud_stride),
                    "--scene-cloud-max-points",
                    str(args_cli.anygrasp_scene_cloud_max_points),
                ]
            )
            command.append("--local-regions" if args_cli.contact_graspnet_local_regions else "--no-local-regions")
            command.append("--filter-grasps" if args_cli.contact_graspnet_filter_grasps else "--no-filter-grasps")
        elif grasp_generator == "graspgen":
            command.extend(
                [
                    "--graspgen-root",
                    str(args_cli.graspgen_root),
                    "--gripper-config",
                    str(args_cli.graspgen_gripper_config),
                    "--scene-cloud-stride",
                    str(args_cli.anygrasp_scene_cloud_stride),
                    "--scene-cloud-max-points",
                    str(args_cli.anygrasp_scene_cloud_max_points),
                    "--target-cloud-max-points",
                    str(args_cli.graspgen_target_cloud_max_points),
                    "--collision-max-scene-points",
                    str(args_cli.graspgen_collision_max_scene_points),
                    "--num-grasps",
                    str(args_cli.graspgen_num_grasps),
                    "--export-tcp-offset",
                    str(args_cli.graspgen_export_tcp_offset),
                    "--symmetric-cloud-mode",
                    args_cli.graspgen_symmetric_cloud_mode,
                    "--symmetry-center-source",
                    args_cli.graspgen_symmetry_center_source,
                    "--symmetric-surface-points",
                    str(args_cli.graspgen_symmetric_surface_points),
                ]
            )
            command.append("--filter-collisions" if args_cli.graspgen_filter_collisions else "--no-filter-collisions")
        elif grasp_generator == "heuristic":
            command.extend(["--object-name", object_name])
        if grasp_generator == "anygrasp" and object_name in args_cli.anygrasp_full_scene_objects:
            command.extend(
                [
                    "--anygrasp-cloud-mode",
                    "full_scene_target_filter",
                    "--scene-cloud-stride",
                    str(args_cli.anygrasp_scene_cloud_stride),
                    "--scene-cloud-max-points",
                    str(args_cli.anygrasp_scene_cloud_max_points),
                    "--target-grasp-filter-distance",
                    str(args_cli.target_grasp_filter_distance),
                    "--target-grasp-filter-pixel-radius",
                    str(args_cli.target_grasp_filter_pixel_radius),
                ]
            )
        if object_center_w is not None:
            command.extend(
                [
                    "--overlay-object-center-world",
                    ",".join(f"{float(value):.9f}" for value in object_center_w[:3]),
                ]
            )
        for extra_view in extra_views or []:
            command.extend(
                [
                    "--extra-view",
                    ",".join(
                        [
                            str(extra_view["rgb"]),
                            str(extra_view["depth_npy"]),
                            str(extra_view["mask"]),
                            str(extra_view["camera_json"]),
                        ]
                    ),
                ]
            )
        run_subprocess(command, anygrasp_dir / "anygrasp.log", env=env)
        heuristic_profile = args_cli.heuristic_profile_overrides.get(object_name, args_cli.heuristic_profile)
        heuristic_symmetric_cloud_mode = args_cli.heuristic_symmetric_cloud_mode_overrides.get(
            object_name,
            args_cli.heuristic_symmetric_cloud_mode,
        )
        if grasp_generator == "heuristic" and heuristic_profile != "topdown":
            heuristic_command = [
                sys.executable,
                "scripts/generate_piper_heuristic_grasps_from_fused_cloud.py",
                "--source-grasp-dir",
                str(anygrasp_dir),
                "--output",
                str(anygrasp_dir),
                "--count",
                str(saved_top_grasps),
                "--seed",
                str(int(args_cli.seed if heuristic_seed is None else heuristic_seed)),
                "--attempts",
                str(args_cli.heuristic_attempts),
                "--camera-json",
                str(camera_json),
                "--jaw-width",
                str(args_cli.piper_max_jaw_width),
                "--heuristic-profile",
                heuristic_profile,
                "--heuristic-family-mode",
                args_cli.heuristic_family_mode,
                "--symmetric-cloud-mode",
                heuristic_symmetric_cloud_mode,
                "--symmetry-center-source",
                args_cli.heuristic_symmetry_center_source,
                "--symmetric-surface-points",
                str(args_cli.heuristic_symmetric_surface_points),
                "--symmetric-top-grasp-fraction",
                str(args_cli.heuristic_symmetric_top_grasp_fraction),
                "--candidate-filter-max-points",
                str(args_cli.heuristic_candidate_filter_max_points),
                "--obstacle-cloud-max-points",
                str(args_cli.obstacle_cloud_max_points),
                "--obstacle-target-exclusion-radius",
                str(args_cli.obstacle_target_exclusion_radius),
                "--piper-offset-modes",
                ",".join(args_cli.piper_offset_modes),
                "--piper-offset-min",
                str(args_cli.piper_offset_min),
                "--piper-offset-max",
                str(args_cli.piper_offset_max),
                "--piper-offset-step",
                str(args_cli.piper_offset_step),
                "--centerline-half-width",
                str(args_cli.centerline_half_width),
                "--centerline-half-depth",
                str(args_cli.centerline_half_depth),
                "--centerline-min-points",
                str(args_cli.centerline_min_points),
                "--closing-region-min-points",
                str(args_cli.closing_region_min_points),
                "--middle-support-min-points",
                str(args_cli.middle_support_min_points),
                "--middle-support-offset",
                str(args_cli.middle_support_offset),
                "--middle-support-half-length",
                str(args_cli.middle_support_half_length),
                "--middle-support-half-width",
                str(args_cli.middle_support_half_width),
                "--middle-support-half-depth",
                str(args_cli.middle_support_half_depth),
                "--root-centerline-clear-length",
                str(args_cli.heuristic_root_centerline_clear_length),
                "--root-centerline-max-points",
                str(args_cli.heuristic_root_centerline_max_points),
                "--target-solid-max-points",
                str(args_cli.target_solid_max_points),
                "--obstacle-solid-max-points",
                str(args_cli.obstacle_solid_max_points),
            ]
            heuristic_command.append(
                "--symmetric-ee-roll" if bool(args_cli.heuristic_symmetric_ee_roll) else "--no-symmetric-ee-roll"
            )
            if object_center_w is not None:
                heuristic_command.extend(
                    [
                        "--object-center-world",
                        ",".join(f"{float(value):.9f}" for value in object_center_w[:3]),
                    ]
                )
            run_subprocess(heuristic_command, anygrasp_dir / "heuristic_piper_sampler.log", env=env)
    return pose_path, load_json(result_path)


def pose_center_w(pose: dict | None) -> list[float] | None:
    if not isinstance(pose, dict):
        return None
    center = pose.get("center_world_median") or pose.get("center_world") or pose.get("center_world_mean")
    if center is None or len(center) < 3:
        return None
    return [float(value) for value in center[:3]]


def laid_down_box_check(name: str, record: dict) -> dict:
    if name != "box_object":
        return {"enabled": False, "reason": "not_box_object"}
    if not bool(args_cli.skip_laid_down_box):
        return {"enabled": False, "reason": "disabled"}
    pose = record.get("video_pose_estimate")
    center = pose_center_w(pose)
    if center is None:
        return {
            "enabled": True,
            "laid_down": False,
            "reason": "missing_rough_center",
            "threshold_z": float(args_cli.box_laid_down_center_z_threshold),
        }
    center_z = float(center[2])
    threshold = float(args_cli.box_laid_down_center_z_threshold)
    laid_down = bool(center_z <= threshold)
    return {
        "enabled": True,
        "laid_down": laid_down,
        "center_world": center,
        "center_z": center_z,
        "threshold_z": threshold,
        "source": pose.get("source") if isinstance(pose, dict) else None,
        "rule": "rough_video_center_z_le_threshold",
    }


def video_anygrasp_extra_view(record: dict) -> dict | None:
    video_dir = Path(record.get("video_rough_dir", ""))
    mask_path = Path(record.get("video_sam3", {}).get("mask_path", ""))
    rgb_path = video_dir / "video_rgb.png"
    depth_path = video_dir / "video_depth.npy"
    camera_path = video_dir / "video_camera.json"
    if not (rgb_path.exists() and depth_path.exists() and camera_path.exists() and mask_path.exists()):
        return None
    return {
        "role": "video_cam_rough",
        "rgb": rgb_path,
        "depth_npy": depth_path,
        "mask": mask_path,
        "camera_json": camera_path,
    }


def quat_abs_dot(left: list[float] | np.ndarray, right: list[float] | np.ndarray) -> float:
    left_q = quat_normalize(np.asarray(left, dtype=np.float64))
    right_q = quat_normalize(np.asarray(right, dtype=np.float64))
    return float(abs(np.dot(left_q, right_q)))


def refresh_observation(env, fallback_obs: dict) -> dict:
    try:
        return env.unwrapped.observation_manager.compute()
    except Exception:
        return fallback_obs


def write_object_pose(env, object_key: str, pos_w: list[float], quat_wxyz: list[float]) -> None:
    scene = env.unwrapped.scene
    obj = scene.rigid_objects[object_key]
    state = obj.data.default_root_state[0:1].clone()
    state[0, 0:3] = torch.tensor(pos_w, dtype=state.dtype, device=state.device)
    state[0, 3:7] = torch.tensor(quat_wxyz, dtype=state.dtype, device=state.device)
    state[0, 7:] = 0.0
    obj.write_root_state_to_sim(state)
    scene.write_data_to_sim()
    env.unwrapped.sim.forward()


def snapshot_rigid_object_states(env) -> dict[str, torch.Tensor]:
    states = {}
    for key, obj in env.unwrapped.scene.rigid_objects.items():
        states[key] = obj.data.root_state_w[0:1].clone()
    return states


def restore_rigid_object_states(env, states: dict[str, torch.Tensor], fallback_obs: dict) -> dict:
    scene = env.unwrapped.scene
    for key, state in states.items():
        if key in scene.rigid_objects:
            scene.rigid_objects[key].write_root_state_to_sim(state)
    scene.write_data_to_sim()
    env.unwrapped.sim.forward()
    return refresh_observation(env, fallback_obs)


def snapshot_robot_joint_state(robot) -> dict[str, torch.Tensor]:
    return {
        "joint_pos": robot.data.joint_pos.clone(),
        "joint_vel": robot.data.joint_vel.clone(),
    }


def restore_robot_and_rigid_object_states(
    env,
    robot,
    robot_state: dict[str, torch.Tensor],
    object_states: dict[str, torch.Tensor],
    fallback_obs: dict,
) -> dict:
    scene = env.unwrapped.scene
    joint_pos = robot_state["joint_pos"].clone()
    joint_vel = robot_state["joint_vel"].clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    for key, state in object_states.items():
        if key in scene.rigid_objects:
            scene.rigid_objects[key].write_root_state_to_sim(state)
    scene.write_data_to_sim()
    env.unwrapped.sim.forward()
    robot.update(dt=env.unwrapped.physics_dt)
    return refresh_observation(env, fallback_obs)


def update_attached_object(env, robot, ee_idx: int, attached: dict | None) -> bool:
    if not attached:
        return False
    ee_pos_w = robot.data.body_pose_w[0, ee_idx, :3].detach()
    offset = torch.tensor(
        attached["ee_to_object_pos_w"],
        dtype=ee_pos_w.dtype,
        device=ee_pos_w.device,
    )
    pos_w = (ee_pos_w + offset).detach().cpu().numpy().astype(float).tolist()
    write_object_pose(env, attached["object_key"], pos_w, attached["object_quat_wxyz"])
    return True


def collect_task_e_object_summary(env) -> dict:
    center = np.asarray(BASKET_SUCCESS_CENTER[:2], dtype=np.float64)
    scene = env.unwrapped.scene
    env_origin = scene.env_origins[0].detach().cpu().numpy()
    objects = {}
    for object_name, label in TASK_E_OBJECT_LABELS.items():
        obj = scene[object_name]
        pos_w = obj.data.root_pos_w[0, :3].detach().cpu().numpy()
        pos_local = pos_w - env_origin
        in_basket = (
            abs(float(pos_local[0]) - float(center[0])) <= float(BASKET_SUCCESS_HALF_X)
            and abs(float(pos_local[1]) - float(center[1])) <= float(BASKET_SUCCESS_HALF_Y)
            and float(TABLE_TOP_Z) <= float(pos_local[2]) <= float(TABLE_TOP_Z) + 0.15
        )
        objects[object_name] = {
            "label": label,
            "pos_w": pos_w.astype(float).tolist(),
            "pos_local": pos_local.astype(float).tolist(),
            "in_basket": bool(in_basket),
        }
    return {
        "basket_success_region": {
            "center": list(BASKET_SUCCESS_CENTER),
            "half_x": float(BASKET_SUCCESS_HALF_X),
            "half_y": float(BASKET_SUCCESS_HALF_Y),
            "z_range": [float(TABLE_TOP_Z), float(TABLE_TOP_Z) + 0.15],
        },
        "objects": objects,
        "count_in_basket": int(sum(item["in_basket"] for item in objects.values())),
        "all_in_basket": bool(all(item["in_basket"] for item in objects.values())),
    }


def task_object_positions_w(env) -> dict[str, np.ndarray]:
    scene = env.unwrapped.scene
    positions = {}
    for name, cfg in OBJECTS.items():
        object_key = cfg["object_key"]
        if object_key not in scene.rigid_objects:
            continue
        positions[name] = (
            scene.rigid_objects[object_key]
            .data.root_pos_w[0, :3]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    return positions


def object_motion_from_start(env, start_positions: dict[str, np.ndarray]) -> dict:
    current_positions = task_object_positions_w(env)
    objects = {}
    max_displacement = 0.0
    for name, start_pos in start_positions.items():
        if name not in current_positions:
            continue
        displacement = float(np.linalg.norm(current_positions[name] - start_pos))
        max_displacement = max(max_displacement, displacement)
        objects[name] = {
            "start_pos_w": start_pos.astype(float).tolist(),
            "current_pos_w": current_positions[name].astype(float).tolist(),
            "displacement_m": displacement,
        }
    return {
        "max_displacement_m": float(max_displacement),
        "objects": objects,
    }


def tracked_object_states(env, tracked_objects: list[dict], ee_pos_w: list[float] | np.ndarray) -> list[dict]:
    scene = env.unwrapped.scene
    ee_pos = np.asarray(ee_pos_w, dtype=np.float64)
    states = []
    seen: set[str] = set()
    for record in tracked_objects:
        object_key = record.get("object_key")
        if not object_key or object_key in seen or object_key not in scene.rigid_objects:
            continue
        seen.add(object_key)
        obj = scene.rigid_objects[object_key]
        pos_w = obj.data.root_pos_w[0, :3].detach().cpu().numpy().astype(np.float64)
        quat_tensor = (
            obj.data.root_quat_w[0, :4]
            if hasattr(obj.data, "root_quat_w")
            else obj.data.root_state_w[0, 3:7]
        )
        quat_wxyz = quat_tensor.detach().cpu().numpy().astype(np.float64)
        states.append(
            {
                "name": record.get("name"),
                "object_key": object_key,
                "label": record.get("label"),
                "pos_w": pos_w.astype(float).tolist(),
                "quat_wxyz": quat_wxyz.astype(float).tolist(),
                "distance_to_ee_m": float(np.linalg.norm(pos_w - ee_pos)),
            }
        )
    return states


def task_e_object_entry(task_e_objects: dict, name: str) -> dict | None:
    label = OBJECTS[name]["label"]
    for item in task_e_objects.get("objects", {}).values():
        if item.get("label") == label:
            return item
    return None


def motion_target_diagnostics(motion_result: dict, name: str) -> dict:
    object_key = OBJECTS[name]["object_key"]
    task_entry = task_e_object_entry(motion_result.get("task_e_objects", {}), name)
    samples = []
    for waypoint in motion_result.get("waypoints", []):
        tracked = next(
            (
                item
                for item in waypoint.get("tracked_objects", [])
                if item.get("object_key") == object_key
            ),
            None,
        )
        if tracked is None:
            continue
        samples.append(
            {
                "waypoint": waypoint.get("name"),
                "object_pos_w": tracked.get("pos_w"),
                "ee_pos_w": waypoint.get("ee_pose_w", {}).get("position"),
                "distance_to_ee_m": tracked.get("distance_to_ee_m"),
            }
        )

    distances = [
        float(item["distance_to_ee_m"])
        for item in samples
        if item.get("distance_to_ee_m") is not None
    ]
    z_values = [
        float(item["object_pos_w"][2])
        for item in samples
        if item.get("object_pos_w") and len(item["object_pos_w"]) >= 3
    ]
    grasp_like = [
        item
        for item in samples
        if str(item.get("waypoint", "")).endswith(("_grasp", "_close"))
        and item.get("distance_to_ee_m") is not None
    ]
    closest_grasp_like = min(grasp_like, key=lambda item: float(item["distance_to_ee_m"]), default=None)
    closest = min(samples, key=lambda item: float(item["distance_to_ee_m"]), default=None) if distances else None
    return {
        "target_in_basket": bool(task_entry.get("in_basket")) if isinstance(task_entry, dict) else False,
        "final_object": task_entry,
        "min_ee_object_distance_m": float(min(distances)) if distances else None,
        "closest_waypoint": closest,
        "closest_grasp_or_close_waypoint": closest_grasp_like,
        "max_object_z_w": float(max(z_values)) if z_values else None,
        "min_object_z_w": float(min(z_values)) if z_values else None,
        "samples": samples,
        "note": "Distance is measured from Piper gripper_base to the object's root pose, not fingertip contact.",
    }


def scene_object_pose_estimate(env, name: str, reason: str) -> dict:
    object_key = OBJECTS[name]["object_key"]
    obj = env.unwrapped.scene.rigid_objects[object_key]
    center = obj.data.root_pos_w[0, :3].detach().cpu().numpy().astype(float).tolist()
    return {
        "center_world": center,
        "center_world_median": center,
        "center_world_mean": center,
        "point_count": 0,
        "pixel_center": None,
        "bbox_xyxy": None,
        "source": "scene_object_pose_fallback",
        "fallback_reason": reason,
    }


def top_grasp_final_pose(candidate: dict, rank: int) -> dict | None:
    world = candidate.get("pose_world")
    if world is None and candidate.get("translation_world") is not None:
        world = {
            "translation": candidate.get("translation_world"),
            "rotation_matrix": candidate.get("rotation_matrix_world"),
        }
    if world is None and candidate.get("translation") is not None:
        world = {
            "translation": candidate.get("translation"),
            "rotation_matrix": candidate.get("rotation_matrix"),
        }
    if not world or world.get("translation") is None or world.get("rotation_matrix") is None:
        return None
    final_pose = {
        "frame": "world",
        "pose_type": "anygrasp_gripper",
        "approach_axis": "rotation_matrix[:,0]",
        "pregrasp_direction": "-rotation_matrix[:,0]",
        "translation": [float(v) for v in world["translation"]],
        "rotation_matrix": [[float(v) for v in row] for row in world["rotation_matrix"]],
        "score": float(candidate.get("score", 0.0)),
        "width": float(candidate.get("width", 0.0)),
        "depth": float(candidate.get("depth", 0.0)),
        "source_rank": int(candidate.get("rank", rank)),
    }
    for key in (
        "generator",
        "heuristic",
        "pose_world",
        "translation_world",
        "rotation_matrix_world",
        "translation_camera",
        "rotation_matrix_camera",
    ):
        if key in candidate:
            final_pose[key] = candidate[key]
    return final_pose


def anygrasp_candidates(record: dict) -> list[dict]:
    payload = (record.get("anygrasp_result") or {}).get("anygrasp") or {}
    candidates = payload.get("top_grasps") or []
    if not candidates:
        final_path = record.get("final_grasp_pose_path")
        if final_path is not None and Path(final_path).exists():
            candidates = [load_json(Path(final_path))]

    converted = []
    for idx, candidate in enumerate(candidates, start=1):
        final_pose = top_grasp_final_pose(candidate, idx)
        if final_pose is not None:
            converted.append(final_pose)
    return converted


def piper_rotation_for_candidate(candidate: dict) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(candidate["rotation_matrix"], dtype=np.float64)
    if args_cli.raw_tool_transform == "graspnet_to_piper_z":
        piper_rotation = np.stack([-rotation[:, 2], rotation[:, 1], rotation[:, 0]], axis=1)
        approach_axis = normalize_or_fallback(piper_rotation[:, 2], np.array([0.0, 0.0, -1.0]))
    else:
        piper_rotation = rotation
        approach_axis = normalize_or_fallback(
            rotation[:, args_cli.approach_axis_column],
            np.array([0.0, 0.0, -1.0]),
        )
    return piper_rotation, approach_axis


def offset_axis_for_mode(
    mode: str,
    translation: np.ndarray,
    approach_axis: np.ndarray,
    object_pose: dict,
) -> np.ndarray:
    if mode == "none":
        return np.zeros(3, dtype=np.float64)
    if mode == "towards_object_center":
        center = np.asarray(object_pose["center_w"], dtype=np.float64)
        return normalize_or_fallback(center - translation, -approach_axis)
    if mode in {"finger_centerline", "yellow_line"}:
        return approach_axis
    return -approach_axis


def load_fused_object_cloud_world(record: dict) -> tuple[np.ndarray, dict]:
    anygrasp_dir = Path(record["anygrasp_result_path"]).parent
    result = record.get("anygrasp_result") or load_json(anygrasp_dir / "anygrasp_result.json")
    points_camera = np.load(anygrasp_dir / "masked_cloud.npy").astype(np.float64)
    camera_path = Path(result["camera_json"]).expanduser().resolve()
    camera_payload = load_json(camera_path)
    camera = camera_payload.get("camera", camera_payload)
    points_world = transform_points_to_world(points_camera, camera)
    return points_world, {
        "anygrasp_dir": str(anygrasp_dir),
        "masked_cloud_npy": str(anygrasp_dir / "masked_cloud.npy"),
        "masked_cloud_ply": str(anygrasp_dir / "masked_cloud.ply"),
        "camera_json": str(camera_path),
        "point_count": int(points_world.shape[0]),
        "point_cloud_frame": result.get("point_cloud_frame", "primary_camera"),
        "fused_view_count": result.get("fused_view_count"),
        "views": result.get("views"),
    }


def load_fused_scene_cloud_world(record: dict) -> tuple[np.ndarray, dict]:
    generator_dir = Path(record["anygrasp_result_path"]).parent
    result = record.get("anygrasp_result") or load_json(generator_dir / "anygrasp_result.json")
    camera_path = Path(result["camera_json"]).expanduser().resolve()
    camera_payload = load_json(camera_path)
    camera = camera_payload.get("camera", camera_payload)
    if result.get("anygrasp_cloud_mode") == "full_scene_target_filter":
        scene_cloud_names = ("anygrasp_input_cloud.npy", "graspgen_input_cloud.npy")
    else:
        scene_cloud_names = ("graspgen_input_cloud.npy",)
    for name in scene_cloud_names:
        path = generator_dir / name
        if path.exists():
            points_world = transform_points_to_world(np.load(path).astype(np.float64), camera)
            return points_world, {
                "generator_dir": str(generator_dir),
                "scene_cloud_npy": str(path),
                "camera_json": str(camera_path),
                "point_count": int(points_world.shape[0]),
                "source": "saved_generator_scene_cloud",
            }
    view_points = []
    view_records = []
    stride = max(1, int(args_cli.anygrasp_scene_cloud_stride))
    max_depth = float(args_cli.max_depth)
    for index, view in enumerate(result.get("views") or [], start=1):
        depth_path_value = view.get("depth_npy")
        camera_path_value = view.get("camera_json")
        if not depth_path_value or not camera_path_value:
            continue
        depth_path = Path(depth_path_value).expanduser().resolve()
        view_camera_path = Path(camera_path_value).expanduser().resolve()
        if not depth_path.exists() or not view_camera_path.exists():
            continue
        view_camera_payload = load_json(view_camera_path)
        view_camera = view_camera_payload.get("camera", view_camera_payload)
        depth = to_depth_array(np.load(depth_path))
        if stride > 1:
            depth_sample = depth[::stride, ::stride]
            intrinsic = np.asarray(view_camera["intrinsic_matrix"], dtype=np.float64).copy()
            intrinsic[0, 0] /= stride
            intrinsic[1, 1] /= stride
            intrinsic[0, 2] /= stride
            intrinsic[1, 2] /= stride
        else:
            depth_sample = depth
            intrinsic = np.asarray(view_camera["intrinsic_matrix"], dtype=np.float64)
        points_camera = backproject(depth_sample, intrinsic)
        valid = np.isfinite(depth_sample) & (depth_sample > 0.0) & (depth_sample < max_depth)
        if not valid.any():
            view_records.append(
                {
                    "name": view.get("name", f"view_{index:02d}"),
                    "camera_json": str(view_camera_path),
                    "depth_npy": str(depth_path),
                    "point_count": 0,
                }
            )
            continue
        points_world = transform_points_to_world(points_camera[valid].astype(np.float64), view_camera)
        view_points.append(points_world)
        view_records.append(
            {
                "name": view.get("name", f"view_{index:02d}"),
                "camera_json": str(view_camera_path),
                "depth_npy": str(depth_path),
                "point_count": int(points_world.shape[0]),
            }
        )
    if view_points:
        points_world = np.concatenate(view_points, axis=0)
        return points_world, {
            "generator_dir": str(generator_dir),
            "scene_cloud_npy": None,
            "camera_json": str(camera_path),
            "point_count": int(points_world.shape[0]),
            "source": "reconstructed_rgbd_views",
            "view_count": int(len(view_records)),
            "views": view_records,
            "scene_cloud_stride": int(stride),
        }
    return np.empty((0, 3), dtype=np.float64), {
        "generator_dir": str(generator_dir),
        "scene_cloud_npy": None,
        "camera_json": str(camera_path),
        "point_count": 0,
        "reason": "generator did not save full scene cloud and RGB-D view reconstruction was unavailable",
    }


def deterministic_downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if int(max_points) <= 0 or len(points) <= int(max_points):
        return points
    indices = np.linspace(0, len(points) - 1, int(max_points)).astype(np.int64)
    return points[indices]


def nearest_target_exclusion_mask(scene_points: np.ndarray, target_points: np.ndarray, radius: float) -> tuple[np.ndarray, str]:
    if len(scene_points) == 0 or len(target_points) == 0:
        return np.ones(len(scene_points), dtype=bool), "empty_scene_or_target"
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(target_points)
        distances, _ = tree.query(scene_points, k=1, workers=-1)
        return distances > float(radius), "scipy_cKDTree"
    except Exception:
        keep = np.ones(len(scene_points), dtype=bool)
        chunk = 2048
        radius_sq = float(radius) * float(radius)
        for start in range(0, len(scene_points), chunk):
            query = scene_points[start : start + chunk]
            min_sq = np.full(len(query), np.inf, dtype=np.float64)
            for target_start in range(0, len(target_points), chunk):
                target = target_points[target_start : target_start + chunk]
                diff = query[:, None, :] - target[None, :, :]
                min_sq = np.minimum(min_sq, np.min(np.sum(diff * diff, axis=2), axis=1))
            keep[start : start + chunk] = min_sq > radius_sq
        return keep, "chunked_numpy"


def load_piper_validation_clouds(record: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    target_points, target_meta = load_fused_object_cloud_world(record)
    scene_points, scene_meta = load_fused_scene_cloud_world(record)
    scene_points = deterministic_downsample_points(scene_points, int(args_cli.obstacle_cloud_max_points))
    keep, method = nearest_target_exclusion_mask(
        scene_points,
        target_points,
        float(args_cli.obstacle_target_exclusion_radius),
    )
    obstacle_points = scene_points[keep]
    return target_points, obstacle_points, {
        "target": target_meta,
        "scene": scene_meta,
        "obstacle": {
            "method": method,
            "target_exclusion_radius_m": float(args_cli.obstacle_target_exclusion_radius),
            "scene_point_count_after_downsample": int(len(scene_points)),
            "obstacle_point_count": int(len(obstacle_points)),
            "max_scene_points": int(args_cli.obstacle_cloud_max_points),
        },
    }


def checked_piper_jaw_width(candidate: dict) -> tuple[float, dict]:
    raw_width = float(candidate.get("width", 0.065))
    max_width = float(args_cli.piper_max_jaw_width)
    if bool(args_cli.piper_clip_generator_width):
        checked_width = float(np.clip(raw_width, 0.030, max_width))
    else:
        checked_width = float(np.clip(raw_width, 0.030, 0.095))
    return checked_width, {
        "raw_width_m": raw_width,
        "checked_width_m": checked_width,
        "max_piper_jaw_width_m": max_width,
        "width_clipped": bool(checked_width < raw_width - 1e-9),
        "clip_enabled": bool(args_cli.piper_clip_generator_width),
    }


def piper_gripper_stats_at_base(
    points_w: np.ndarray,
    piper_rotation: np.ndarray,
    gripper_base_w: np.ndarray,
    jaw_width: float,
    clearance: float,
) -> dict:
    side_axis = normalize_or_fallback(np.asarray(piper_rotation, dtype=np.float64)[:, 0], np.array([1.0, 0.0, 0.0]))
    jaw_axis = normalize_or_fallback(np.asarray(piper_rotation, dtype=np.float64)[:, 1], np.array([0.0, 1.0, 0.0]))
    approach_axis = normalize_or_fallback(np.asarray(piper_rotation, dtype=np.float64)[:, 2], np.array([0.0, 0.0, -1.0]))
    points = np.asarray(points_w, dtype=np.float64)
    base = np.asarray(gripper_base_w, dtype=np.float64)
    clearance = max(0.0, float(clearance))
    jaw_width = float(np.clip(jaw_width, 0.030, float(args_cli.piper_max_jaw_width)))

    # Keep this convention aligned with piper_gripper_collision_stats_at_pose()
    # and scripts/visualize_anygrasp_fused_pc.py: the offset/executed reference
    # point is the contact/tip end of the finger centerline, so the solid finger
    # bodies extend backward by one finger length along -approach_axis.
    finger_root = base - approach_axis * PIPER_FINGER_LENGTH_M

    inside_solid = np.zeros(points.shape[0], dtype=bool)
    components: dict[str, int] = {}
    for side_sign, label in [(-1.0, "left_finger"), (1.0, "right_finger")]:
        root_center = finger_root + jaw_axis * side_sign * (jaw_width * 0.5 + PIPER_FINGER_WIDTH_M * 0.5)
        rel = points - root_center[None, :]
        along = rel @ approach_axis
        jaw = rel @ jaw_axis
        side = rel @ side_axis
        inside = (
            (along >= -clearance)
            & (along <= PIPER_FINGER_LENGTH_M + clearance)
            & (np.abs(jaw) <= PIPER_FINGER_WIDTH_M * 0.5 + clearance)
            & (np.abs(side) <= PIPER_FINGER_DEPTH_M * 0.5 + clearance)
        )
        components[label] = int(np.count_nonzero(inside))
        inside_solid |= inside

    palm_center = finger_root - approach_axis * (PIPER_PALM_APPROACH_THICKNESS_M * 0.5)
    rel = points - palm_center[None, :]
    palm_inside = (
        (np.abs(rel @ approach_axis) <= PIPER_PALM_APPROACH_THICKNESS_M * 0.5 + clearance)
        & (np.abs(rel @ jaw_axis) <= (jaw_width * 0.5 + PIPER_FINGER_WIDTH_M) + clearance)
        & (np.abs(rel @ side_axis) <= PIPER_FINGER_DEPTH_M * 0.5 + clearance)
    )
    components["palm_base"] = int(np.count_nonzero(palm_inside))
    inside_solid |= palm_inside

    rel = points - finger_root[None, :]
    along = rel @ approach_axis
    jaw = rel @ jaw_axis
    side = rel @ side_axis
    closing_region = (
        (along >= -clearance)
        & (along <= PIPER_FINGER_LENGTH_M + clearance)
        & (np.abs(jaw) <= jaw_width * 0.5 + clearance)
        & (np.abs(side) <= PIPER_FINGER_DEPTH_M * 0.5 + clearance)
    )
    centerline = (
        (along >= -clearance)
        & (along <= PIPER_FINGER_LENGTH_M + clearance)
        & (np.abs(jaw) <= float(args_cli.centerline_half_width) + clearance)
        & (np.abs(side) <= float(args_cli.centerline_half_depth) + clearance)
    )
    root_clear_length = float(np.clip(args_cli.heuristic_root_centerline_clear_length, 0.0, PIPER_FINGER_LENGTH_M))
    root_centerline = (
        (along >= -clearance)
        & (along <= root_clear_length + clearance)
        & (np.abs(jaw) <= float(args_cli.centerline_half_width) + clearance)
        & (np.abs(side) <= float(args_cli.centerline_half_depth) + clearance)
    )
    middle_offset = float(np.clip(args_cli.middle_support_offset, 0.0, PIPER_FINGER_LENGTH_M))
    middle_center = base - approach_axis * middle_offset
    rel_mid = points - middle_center[None, :]
    middle_support = (
        (np.abs(rel_mid @ approach_axis) <= float(args_cli.middle_support_half_length) + clearance)
        & (np.abs(rel_mid @ jaw_axis) <= float(args_cli.middle_support_half_width) + clearance)
        & (np.abs(rel_mid @ side_axis) <= float(args_cli.middle_support_half_depth) + clearance)
    )
    tcp_support_offset = float(np.clip(args_cli.tcp_support_offset, 0.0, PIPER_FINGER_LENGTH_M))
    tcp_support_center = base - approach_axis * tcp_support_offset
    rel_tcp = points - tcp_support_center[None, :]
    tcp_support = (
        (np.abs(rel_tcp @ approach_axis) <= float(args_cli.tcp_support_half_length) + clearance)
        & (np.abs(rel_tcp @ jaw_axis) <= float(args_cli.tcp_support_half_width) + clearance)
        & (np.abs(rel_tcp @ side_axis) <= float(args_cli.tcp_support_half_depth) + clearance)
    )
    return {
        "solid_point_count": int(np.count_nonzero(inside_solid)),
        "component_collision_points": components,
        "closing_region_point_count": int(np.count_nonzero(closing_region)),
        "centerline_point_count": int(np.count_nonzero(centerline)),
        "root_centerline_point_count": int(np.count_nonzero(root_centerline)),
        "root_centerline_clear_length_m": float(root_clear_length),
        "middle_support_point_count": int(np.count_nonzero(middle_support)),
        "middle_support_center_w": middle_center.astype(float).tolist(),
        "middle_support_offset_m": float(middle_offset),
        "middle_support_box_half_extents_m": [
            float(args_cli.middle_support_half_length),
            float(args_cli.middle_support_half_width),
            float(args_cli.middle_support_half_depth),
        ],
        "tcp_support_point_count": int(np.count_nonzero(tcp_support)),
        "tcp_support_center_w": tcp_support_center.astype(float).tolist(),
        "tcp_support_offset_m": float(tcp_support_offset),
        "tcp_support_box_half_extents_m": [
            float(args_cli.tcp_support_half_length),
            float(args_cli.tcp_support_half_width),
            float(args_cli.tcp_support_half_depth),
        ],
        "gripper_base_w": base.astype(float).tolist(),
        "finger_root_center_w": finger_root.astype(float).tolist(),
        "finger_tip_center_w": base.astype(float).tolist(),
        "geometry_reference": "offset_pose_is_finger_tip_contact_end",
        "jaw_width_m": jaw_width,
    }


def piper_offset_values() -> list[float]:
    step = max(1e-5, float(args_cli.piper_offset_step))
    minimum = float(args_cli.piper_offset_min)
    maximum = max(minimum, float(args_cli.piper_offset_max))
    values = [float(v) for v in np.arange(minimum, maximum + step * 0.5, step)]
    if not values or values[-1] < maximum:
        values.append(maximum)
    return sorted({round(float(np.clip(v, minimum, maximum)), 6) for v in values})


def effective_centerline_min_points(object_name: str | None) -> int:
    if object_name is not None and object_name in set(args_cli.centerline_relaxed_objects):
        return 0
    return int(args_cli.centerline_min_points)


def effective_middle_support_min_points(object_name: str | None) -> int:
    if object_name is not None and object_name in set(args_cli.middle_support_relaxed_objects):
        return 0
    return int(args_cli.middle_support_min_points)


def effective_tcp_support_min_points(object_name: str | None) -> int:
    if object_name is not None and object_name in set(args_cli.tcp_support_objects):
        return int(args_cli.tcp_support_min_points)
    return 0


def final_piper_geometry_search(
    target_points_w: np.ndarray,
    obstacle_points_w: np.ndarray,
    candidate: dict,
    object_pose: dict,
    object_name: str | None = None,
) -> tuple[dict, dict]:
    translation = np.asarray(candidate["translation"], dtype=np.float64)
    piper_rotation, approach_axis = piper_rotation_for_candidate(candidate)
    checked_width, width_info = checked_piper_jaw_width(candidate)
    centerline_min_points = effective_centerline_min_points(object_name)
    middle_support_min_points = effective_middle_support_min_points(object_name)
    tcp_support_min_points = effective_tcp_support_min_points(object_name)
    tested = []
    passing = []
    for mode in args_cli.piper_offset_modes:
        axis = offset_axis_for_mode(mode, translation, approach_axis, object_pose)
        for offset in piper_offset_values():
            base = translation + axis * float(offset)
            target_stats = piper_gripper_stats_at_base(
                target_points_w,
                piper_rotation,
                base,
                checked_width,
                args_cli.approach_pc_collision_clearance,
            )
            obstacle_stats = piper_gripper_stats_at_base(
                obstacle_points_w,
                piper_rotation,
                base,
                checked_width,
                args_cli.approach_pc_collision_clearance,
            )
            target_solid_ok = int(target_stats["solid_point_count"]) <= int(args_cli.target_solid_max_points)
            obstacle_solid_ok = int(obstacle_stats["solid_point_count"]) <= int(args_cli.obstacle_solid_max_points)
            centerline_ok = int(target_stats["centerline_point_count"]) >= int(centerline_min_points)
            closing_ok = int(target_stats["closing_region_point_count"]) >= int(args_cli.closing_region_min_points)
            middle_support_ok = int(target_stats["middle_support_point_count"]) >= int(middle_support_min_points)
            tcp_support_ok = int(target_stats["tcp_support_point_count"]) >= int(tcp_support_min_points)
            root_centerline_clear_ok = int(target_stats["root_centerline_point_count"]) <= int(
                args_cli.heuristic_root_centerline_max_points
            )
            ok = bool(
                target_solid_ok
                and obstacle_solid_ok
                and centerline_ok
                and closing_ok
                and middle_support_ok
                and tcp_support_ok
                and root_centerline_clear_ok
            )
            entry = {
                "ok": ok,
                "offset_mode": mode,
                "offset_m": float(offset),
                "offset_axis_w": axis.astype(float).tolist(),
                "width": width_info,
                "target": target_stats,
                "obstacle": obstacle_stats,
                "checks": {
                    "target_solid_ok": bool(target_solid_ok),
                    "obstacle_solid_ok": bool(obstacle_solid_ok),
                    "centerline_ok": bool(centerline_ok),
                    "closing_region_ok": bool(closing_ok),
                    "middle_support_ok": bool(middle_support_ok),
                    "tcp_support_ok": bool(tcp_support_ok),
                    "root_centerline_clear_ok": bool(root_centerline_clear_ok),
                    "target_solid_max_points": int(args_cli.target_solid_max_points),
                    "obstacle_solid_max_points": int(args_cli.obstacle_solid_max_points),
                    "centerline_min_points": int(centerline_min_points),
                    "closing_region_min_points": int(args_cli.closing_region_min_points),
                    "middle_support_min_points": int(middle_support_min_points),
                    "tcp_support_min_points": int(tcp_support_min_points),
                    "root_centerline_max_points": int(args_cli.heuristic_root_centerline_max_points),
                },
            }
            tested.append(entry)
            if ok:
                passing.append(entry)

    def rank_key(entry: dict) -> tuple:
        return (
            abs(float(entry["offset_m"]) - PIPER_FINGER_LENGTH_M),
            -int(entry["target"]["centerline_point_count"]),
            -int(entry["target"]["closing_region_point_count"]),
            -int(entry["target"]["middle_support_point_count"]),
            -int(entry["target"]["tcp_support_point_count"]),
        )

    selected = sorted(passing, key=rank_key)[0] if passing else sorted(
        tested,
        key=lambda entry: (
            int(entry["target"]["solid_point_count"]) + int(entry["obstacle"]["solid_point_count"]),
            -int(entry["target"]["centerline_point_count"]),
            -int(entry["target"]["closing_region_point_count"]),
            -int(entry["target"]["middle_support_point_count"]),
            -int(entry["target"]["tcp_support_point_count"]),
        ),
    )[0]
    selected_pose = copy.deepcopy(candidate)
    selected_pose["gripper_base_offset_override_m"] = float(selected["offset_m"])
    selected_pose["gripper_base_offset_mode_override"] = selected["offset_mode"]
    selected_pose["piper_checked_width_m"] = float(width_info["checked_width_m"])
    selected_pose["piper_width_clipped"] = bool(width_info["width_clipped"])
    selected_pose["piper_final_geometry"] = selected
    report = {
        "enabled": True,
        "safe": bool(selected["ok"]),
        "selected": selected,
        "passing_count": int(len(passing)),
        "tested_count": int(len(tested)),
        "tested": tested,
        "piper_geometry_m": {
            "finger_length": PIPER_FINGER_LENGTH_M,
            "finger_width": PIPER_FINGER_WIDTH_M,
            "finger_depth": PIPER_FINGER_DEPTH_M,
            "palm_approach_thickness": PIPER_PALM_APPROACH_THICKNESS_M,
            "max_jaw_width": float(args_cli.piper_max_jaw_width),
        },
    }
    return selected_pose, report


def quat_xyzw_to_wxyz(quat: list[float] | np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return np.asarray([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


def world_pose_to_robot_root(robot, pos_w: np.ndarray, quat_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation as R

    root_pose = robot.data.root_pose_w[0].detach().cpu().numpy()
    root_pos = root_pose[:3].astype(np.float64)
    root_rot = R.from_quat(quat_wxyz_to_xyzw(root_pose[3:7]))
    target_rot_w = R.from_quat(quat_wxyz_to_xyzw(quat_wxyz))
    pos_root = root_rot.inv().apply(np.asarray(pos_w, dtype=np.float64) - root_pos)
    quat_root = quat_xyzw_to_wxyz((root_rot.inv() * target_rot_w).as_quat())
    return pos_root.astype(np.float32), quat_root.astype(np.float32)


def robot_root_pose_to_world(robot, pos_root: np.ndarray, quat_root_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation as R

    root_pose = robot.data.root_pose_w[0].detach().cpu().numpy()
    root_pos = root_pose[:3].astype(np.float64)
    root_rot = R.from_quat(quat_wxyz_to_xyzw(root_pose[3:7]))
    target_rot_root = R.from_quat(quat_wxyz_to_xyzw(quat_root_wxyz))
    pos_w = root_pos + root_rot.apply(np.asarray(pos_root, dtype=np.float64))
    quat_w = quat_xyzw_to_wxyz((root_rot * target_rot_root).as_quat())
    return pos_w.astype(np.float32), quat_w.astype(np.float32)


def rotation_matrix_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    return R.from_quat(quat_wxyz_to_xyzw(quat_wxyz)).as_matrix().astype(np.float64)


def cuboid_world_to_robot_root(robot, name: str, center_w: list[float], dims: list[float]) -> dict:
    pos_root, quat_root = world_pose_to_robot_root(
        robot,
        np.asarray(center_w, dtype=np.float32),
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    return {
        "name": name,
        "pose": pos_root.astype(float).tolist() + quat_root.astype(float).tolist(),
        "dims": [float(v) for v in dims],
    }


def task_e_curobo_world_cuboids(robot) -> list[dict]:
    table_dims = [float(TABLE_DIMS[0] + 0.04), float(TABLE_DIMS[1] + 0.04), float(TABLE_TOP_Z)]
    return [
        cuboid_world_to_robot_root(
            robot,
            "table",
            [float(TABLE_CENTER_X), float(TABLE_CENTER_Y), float(TABLE_TOP_Z * 0.5)],
            table_dims,
        ),
        cuboid_world_to_robot_root(
            robot,
            "basket_outer",
            [float(BASKET_CENTER_X), float(BASKET_CENTER_Y), float(TABLE_TOP_Z + 0.075)],
            [0.46, 0.30, 0.15],
        ),
    ]


def curobo_obstacle_cuboids_from_cloud(
    robot,
    target_points_w: np.ndarray,
    obstacle_points_w: np.ndarray,
) -> tuple[list[dict], dict]:
    if not bool(args_cli.curobo_scene_collision):
        return [], {"enabled": False, "reason": "disabled_by_cli"}

    points = np.asarray(obstacle_points_w, dtype=np.float64)
    target = np.asarray(target_points_w, dtype=np.float64)
    if points.size == 0 or target.size == 0:
        return [], {
            "enabled": True,
            "cuboid_count": 0,
            "reason": "empty_obstacle_or_target_cloud",
        }

    target_center = np.median(target, axis=0)
    radius = max(0.0, float(args_cli.curobo_obstacle_radius))
    min_z = float(TABLE_TOP_Z + args_cli.curobo_obstacle_min_height_above_table)
    max_z = float(TABLE_TOP_Z + 0.55)
    workspace = (
        np.isfinite(points).all(axis=1)
        & (points[:, 2] >= min_z)
        & (points[:, 2] <= max_z)
    )
    if radius > 0.0:
        workspace &= np.linalg.norm(points[:, :2] - target_center[None, :2], axis=1) <= radius
    points = points[workspace]
    if len(points) == 0:
        return [], {
            "enabled": True,
            "cuboid_count": 0,
            "reason": "no_points_after_workspace_filter",
            "target_center_w": target_center.astype(float).tolist(),
            "min_z_m": float(min_z),
            "radius_m": float(radius),
        }

    voxel_size = max(0.005, float(args_cli.curobo_obstacle_voxel_size))
    padding = max(0.0, float(args_cli.curobo_obstacle_cuboid_padding))
    min_points = max(1, int(args_cli.curobo_obstacle_min_points_per_voxel))
    max_cuboids = max(1, int(args_cli.curobo_obstacle_max_cuboids))
    voxel_ids = np.floor(points / voxel_size).astype(np.int64)
    unique_ids, inverse, counts = np.unique(voxel_ids, axis=0, return_inverse=True, return_counts=True)

    groups = []
    for group_index, voxel_id in enumerate(unique_ids):
        count = int(counts[group_index])
        if count < min_points:
            continue
        group_points = points[inverse == group_index]
        center = np.mean(group_points, axis=0)
        extent = np.ptp(group_points, axis=0)
        dims = np.maximum(extent + 2.0 * padding, voxel_size)
        distance_to_target = float(np.linalg.norm(center[:2] - target_center[:2]))
        groups.append(
            {
                "count": count,
                "center": center,
                "dims": dims,
                "voxel_id": voxel_id,
                "distance_to_target_m": distance_to_target,
            }
        )

    groups.sort(key=lambda item: (item["distance_to_target_m"], -item["count"]))
    total_candidate_voxels = int(len(groups))
    groups = groups[:max_cuboids]
    cuboids = []
    for index, group in enumerate(groups):
        cuboids.append(
            cuboid_world_to_robot_root(
                robot,
                f"scene_obstacle_{index:03d}",
                group["center"].astype(float).tolist(),
                group["dims"].astype(float).tolist(),
            )
        )

    return cuboids, {
        "enabled": True,
        "source": "fused_scene_cloud_minus_target_voxel_cuboids",
        "cuboid_count": int(len(cuboids)),
        "candidate_voxel_count": int(total_candidate_voxels),
        "filtered_point_count": int(len(points)),
        "target_center_w": target_center.astype(float).tolist(),
        "radius_m": float(radius),
        "min_z_m": float(min_z),
        "voxel_size_m": float(voxel_size),
        "cuboid_padding_m": float(padding),
        "min_points_per_voxel": int(min_points),
        "max_cuboids": int(max_cuboids),
        "cuboids": cuboids,
    }


def curobo_approach_collision_check(
    target_points_w: np.ndarray,
    obstacle_points_w: np.ndarray,
    name: str,
    candidate: dict,
    object_pose: dict,
) -> dict:
    if args_cli.disable_approach_pc_collision_filter:
        return {"enabled": False, "safe": True, "disabled": True}

    grasp_pos, _, approach_axis, _ = grasp_pose_for_object(args_cli, name, candidate, object_pose)
    piper_rotation, _ = piper_rotation_for_candidate(candidate)
    checked_width, width_info = checked_piper_jaw_width(candidate)
    _, pregrasp_pos, _, stage_pos = pregrasp_layout_for_candidate(candidate, grasp_pos, approach_axis)
    samples_per_segment = max(2, int(args_cli.approach_pc_collision_samples))
    final_fraction = float(np.clip(args_cli.approach_pc_collision_final_fraction, 0.0, 1.0))
    clearance = float(args_cli.approach_pc_collision_clearance)
    tested: list[dict] = []
    first_illegal = None

    def add_segment(label: str, start: np.ndarray, end: np.ndarray, max_fraction: float) -> None:
        nonlocal first_illegal
        fractions = [float(v) for v in np.linspace(0.0, max_fraction, samples_per_segment)]
        for fraction in fractions:
            pos = start * (1.0 - fraction) + end * fraction
            target_stats = piper_gripper_stats_at_base(target_points_w, piper_rotation, pos, checked_width, clearance)
            obstacle_stats = piper_gripper_stats_at_base(obstacle_points_w, piper_rotation, pos, checked_width, clearance)
            target_illegal = int(target_stats["solid_point_count"]) > int(args_cli.target_solid_max_points)
            obstacle_illegal = int(obstacle_stats["solid_point_count"]) > int(args_cli.obstacle_solid_max_points)
            entry = {
                "segment": label,
                "fraction": float(fraction),
                "illegal": bool(target_illegal or obstacle_illegal),
                "target_solid_point_count": int(target_stats["solid_point_count"]),
                "obstacle_solid_point_count": int(obstacle_stats["solid_point_count"]),
                "target_component_collision_points": target_stats["component_collision_points"],
                "obstacle_component_collision_points": obstacle_stats["component_collision_points"],
                "gripper_base_w": pos.astype(float).tolist(),
            }
            tested.append(entry)
            if first_illegal is None and entry["illegal"]:
                first_illegal = entry

    add_segment("stage_to_pregrasp", stage_pos, pregrasp_pos, 1.0)
    add_segment("pregrasp_to_grasp_early", pregrasp_pos, grasp_pos, final_fraction)
    return {
        "enabled": True,
        "safe": first_illegal is None,
        "first_illegal": first_illegal,
        "width": width_info,
        "samples_per_segment": int(samples_per_segment),
        "pregrasp_to_grasp_checked_until_fraction": float(final_fraction),
        "clearance_m": float(clearance),
        "tested_samples": tested,
    }


def piper_geometry_check_at_base(
    target_points_w: np.ndarray,
    obstacle_points_w: np.ndarray,
    piper_rotation: np.ndarray,
    gripper_base_w: np.ndarray,
    jaw_width: float,
    width_info: dict,
    object_name: str | None = None,
) -> dict:
    target_stats = piper_gripper_stats_at_base(
        target_points_w,
        piper_rotation,
        gripper_base_w,
        jaw_width,
        args_cli.approach_pc_collision_clearance,
    )
    obstacle_stats = piper_gripper_stats_at_base(
        obstacle_points_w,
        piper_rotation,
        gripper_base_w,
        jaw_width,
        args_cli.approach_pc_collision_clearance,
    )
    target_solid_ok = int(target_stats["solid_point_count"]) <= int(args_cli.target_solid_max_points)
    obstacle_solid_ok = int(obstacle_stats["solid_point_count"]) <= int(args_cli.obstacle_solid_max_points)
    centerline_min_points = effective_centerline_min_points(object_name)
    middle_support_min_points = effective_middle_support_min_points(object_name)
    tcp_support_min_points = effective_tcp_support_min_points(object_name)
    centerline_ok = int(target_stats["centerline_point_count"]) >= int(centerline_min_points)
    closing_ok = int(target_stats["closing_region_point_count"]) >= int(args_cli.closing_region_min_points)
    middle_support_ok = int(target_stats["middle_support_point_count"]) >= int(middle_support_min_points)
    tcp_support_ok = int(target_stats["tcp_support_point_count"]) >= int(tcp_support_min_points)
    root_centerline_clear_ok = int(target_stats["root_centerline_point_count"]) <= int(
        args_cli.heuristic_root_centerline_max_points
    )
    safe = bool(
        target_solid_ok
        and obstacle_solid_ok
        and centerline_ok
        and closing_ok
        and middle_support_ok
        and tcp_support_ok
        and root_centerline_clear_ok
    )
    return {
        "enabled": True,
        "safe": safe,
        "gripper_base_w": np.asarray(gripper_base_w, dtype=float).tolist(),
        "width": width_info,
        "target": target_stats,
        "obstacle": obstacle_stats,
        "checks": {
            "target_solid_ok": bool(target_solid_ok),
            "obstacle_solid_ok": bool(obstacle_solid_ok),
            "centerline_ok": bool(centerline_ok),
            "closing_region_ok": bool(closing_ok),
            "middle_support_ok": bool(middle_support_ok),
            "tcp_support_ok": bool(tcp_support_ok),
            "root_centerline_clear_ok": bool(root_centerline_clear_ok),
            "target_solid_max_points": int(args_cli.target_solid_max_points),
            "obstacle_solid_max_points": int(args_cli.obstacle_solid_max_points),
            "centerline_min_points": int(centerline_min_points),
            "closing_region_min_points": int(args_cli.closing_region_min_points),
            "middle_support_min_points": int(middle_support_min_points),
            "tcp_support_min_points": int(tcp_support_min_points),
            "root_centerline_max_points": int(args_cli.heuristic_root_centerline_max_points),
        },
    }


def action_soft_limits_for_robot(robot) -> np.ndarray | None:
    if not bool(args_cli.curobo_require_action_soft_limits):
        return None
    try:
        arm_ids_for_limits, _ = robot.find_joints(ARM_JOINT_NAMES)
        soft_limits = getattr(robot.data, "soft_joint_pos_limits", None)
        if soft_limits is None:
            return None
        return soft_limits[0, arm_ids_for_limits].detach().cpu().numpy().astype(np.float64)
    except Exception:
        return None


def object_exempts_action_soft_limits(name: str | None) -> bool:
    return bool(name and name in set(args_cli.curobo_soft_limit_exempt_objects))


def action_soft_limit_tolerance_for_object(name: str | None) -> float:
    if name in getattr(args_cli, "curobo_soft_limit_tolerance_overrides", {}):
        return max(0.0, float(args_cli.curobo_soft_limit_tolerance_overrides[name]))
    return 0.0


def action_soft_limits_for_object(robot, name: str | None) -> np.ndarray | None:
    if object_exempts_action_soft_limits(name):
        return None
    return action_soft_limits_for_robot(robot)


def curobo_position_tolerance_for_object(name: str | None) -> float:
    if name in getattr(args_cli, "curobo_position_tol_overrides", {}):
        return float(args_cli.curobo_position_tol_overrides[name])
    return float(args_cli.curobo_position_tol)


def object_accepts_tolerance_ik_success(name: str | None) -> bool:
    return bool(name and name in set(args_cli.curobo_accept_tolerance_ik_objects))


def action_soft_limit_report(
    joint_position: np.ndarray,
    soft_limits_np: np.ndarray | None,
    object_name: str | None = None,
) -> tuple[bool, dict]:
    exempt = object_exempts_action_soft_limits(object_name)
    enabled = bool(args_cli.curobo_require_action_soft_limits) and not exempt
    if exempt:
        return True, {
            "enabled": False,
            "within_soft_limits": True,
            "reason": "object_exempt",
            "object_name": object_name,
        }
    if soft_limits_np is None:
        return True, {"enabled": enabled, "within_soft_limits": True, "reason": "limits_unavailable"}

    q_for_limits = np.asarray(joint_position, dtype=np.float64)
    lower = soft_limits_np[:, 0]
    upper = soft_limits_np[:, 1]
    tolerance_rad = action_soft_limit_tolerance_for_object(object_name)
    lower_violation = np.maximum(lower - q_for_limits, 0.0)
    upper_violation = np.maximum(q_for_limits - upper, 0.0)
    lower_margin = q_for_limits - lower
    upper_margin = upper - q_for_limits
    min_margin = float(np.min(np.minimum(lower_margin, upper_margin)))
    max_lower_violation = float(np.max(lower_violation))
    max_upper_violation = float(np.max(upper_violation))
    within_soft_limits = bool(
        max_lower_violation <= tolerance_rad + 1e-6
        and max_upper_violation <= tolerance_rad + 1e-6
    )
    return within_soft_limits, {
        "enabled": enabled,
        "object_name": object_name,
        "tolerance_rad": float(tolerance_rad),
        "joint_position": q_for_limits.astype(float).tolist(),
        "lower": lower.astype(float).tolist(),
        "upper": upper.astype(float).tolist(),
        "lower_margin_rad": lower_margin.astype(float).tolist(),
        "upper_margin_rad": upper_margin.astype(float).tolist(),
        "min_margin_rad": min_margin,
        "max_lower_violation_rad": max_lower_violation,
        "max_upper_violation_rad": max_upper_violation,
        "within_soft_limits": within_soft_limits,
    }


def action_default_arm_joint_pos_for_robot(robot) -> np.ndarray | None:
    if not bool(args_cli.curobo_require_action_range):
        return None
    try:
        arm_ids_for_actions, _ = robot.find_joints(ARM_JOINT_NAMES)
        return robot.data.default_joint_pos[0, arm_ids_for_actions].detach().cpu().numpy().astype(np.float64)
    except Exception:
        return None


def action_range_report(joint_position: np.ndarray, default_arm_q: np.ndarray | None) -> tuple[bool, dict]:
    enabled = bool(args_cli.curobo_require_action_range)
    if default_arm_q is None:
        return True, {"enabled": enabled, "within_action_range": True, "reason": "default_joint_pos_unavailable"}

    q_for_action = np.asarray(joint_position, dtype=np.float64)
    default_q = np.asarray(default_arm_q, dtype=np.float64)
    action = (q_for_action - default_q) / float(ACTION_SCALE)
    abs_action = np.abs(action)
    limit = max(0.0, float(args_cli.curobo_action_limit))
    excess = np.maximum(abs_action - limit, 0.0)
    within_action_range = bool(np.max(excess) <= 1e-6)
    return within_action_range, {
        "enabled": enabled,
        "default_joint_position": default_q.astype(float).tolist(),
        "action_scale": float(ACTION_SCALE),
        "action_limit": float(limit),
        "normalized_action": action.astype(float).tolist(),
        "max_abs_action": float(np.max(abs_action)),
        "max_excess": float(np.max(excess)),
        "within_action_range": within_action_range,
    }


def curobo_probe_candidate_final_grasp(
    planner,
    robot,
    current_q: np.ndarray,
    name: str,
    candidate: dict,
    object_pose: dict,
    target_points_w: np.ndarray,
    obstacle_points_w: np.ndarray,
) -> dict:
    grasp_pos, grasp_quat, approach_axis, mode_note = grasp_pose_for_object(
        args_cli,
        name,
        candidate,
        object_pose,
    )
    soft_limits_np = action_soft_limits_for_object(robot, name)
    default_arm_q = action_default_arm_joint_pos_for_robot(robot)
    position_tolerance = curobo_position_tolerance_for_object(name)
    pos_root, quat_root = world_pose_to_robot_root(robot, grasp_pos, np.asarray(grasp_quat, dtype=np.float64))
    ik = planner.solve_ik(np.asarray(current_q, dtype=np.float32), pos_root, quat_root, return_seeds=1)
    soft_joint_limit_ok, soft_joint_limit_check = action_soft_limit_report(ik.joint_position, soft_limits_np, name)
    action_range_ok, action_range_check = action_range_report(ik.joint_position, default_arm_q)

    reached_grasp_geometry = {"enabled": False, "safe": True}
    reached_pose_filter_enabled = not bool(args_cli.disable_curobo_reached_pose_pc_filter)
    checked_width, width_info = checked_piper_jaw_width(candidate)
    if reached_pose_filter_enabled:
        fk_pos_root, fk_quat_root = planner.compute_fk(ik.joint_position)
        fk_pos_w, fk_quat_w = robot_root_pose_to_world(robot, fk_pos_root, fk_quat_root)
        piper_rotation = rotation_matrix_from_quat_wxyz(fk_quat_w)
        reached_grasp_geometry = piper_geometry_check_at_base(
            target_points_w,
            obstacle_points_w,
            piper_rotation,
            fk_pos_w,
            checked_width,
            width_info,
            name,
        )
        reached_grasp_geometry["target_pos_w"] = np.asarray(grasp_pos, dtype=float).tolist()
        reached_grasp_geometry["target_quat_wxyz"] = [float(v) for v in grasp_quat]
        reached_grasp_geometry["fk_pos_root"] = fk_pos_root.astype(float).tolist()
        reached_grasp_geometry["fk_quat_root_wxyz"] = fk_quat_root.astype(float).tolist()
        reached_grasp_geometry["fk_pos_w"] = fk_pos_w.astype(float).tolist()
        reached_grasp_geometry["fk_quat_wxyz"] = fk_quat_w.astype(float).tolist()
        reached_grasp_geometry["fk_position_delta_from_target_m"] = float(
            np.linalg.norm(fk_pos_w.astype(np.float64) - np.asarray(grasp_pos, dtype=np.float64))
        )

    position_ok = float(ik.position_error_m) <= float(position_tolerance)
    rotation_ok = float(ik.rotation_error_rad) <= float(args_cli.curobo_rotation_tol)
    ik_success_for_acceptance = bool(
        ik.success or (object_accepts_tolerance_ik_success(name) and position_ok and rotation_ok)
    )
    reached_grasp_geometry_ok = bool(reached_grasp_geometry.get("safe", True))
    final_action_feasible = bool(ik_success_for_acceptance and soft_joint_limit_ok and action_range_ok)
    ok = bool(final_action_feasible and position_ok and rotation_ok and reached_grasp_geometry_ok)
    stage_record = {
        "label": "grasp",
        "critical_for_acceptance": True,
        "ik_success": bool(ik.success),
        "ik_success_used_for_acceptance": bool(ik_success_for_acceptance),
        "ik_success_by_tolerance": bool((not ik.success) and ik_success_for_acceptance),
        "ik_feasible": None if ik.feasible is None else bool(ik.feasible),
        "action_soft_joint_limit_ok": bool(soft_joint_limit_ok),
        "action_soft_joint_limit_check": soft_joint_limit_check,
        "action_range_ok": bool(action_range_ok),
        "action_range_check": action_range_check,
        "ik_position_error_m": float(ik.position_error_m),
        "ik_rotation_error_rad": float(ik.rotation_error_rad),
        "solve_time_s": float(ik.solve_time_s),
        "target_pos_w": np.asarray(grasp_pos, dtype=float).tolist(),
        "target_pos_root": pos_root.astype(float).tolist(),
        "target_quat_wxyz": [float(v) for v in grasp_quat],
        "target_quat_root_wxyz": quat_root.astype(float).tolist(),
        "joint_position": ik.joint_position.astype(float).tolist(),
    }
    return {
        "rank": int(candidate.get("source_rank", 0)),
        "score": float(candidate.get("score", 0.0)),
        "ok": ok,
        "staged_screen_only": True,
        "stage": "grasp",
        "ik_success": bool(ik.success),
        "ik_success_used_for_acceptance": bool(ik_success_for_acceptance),
        "ik_success_by_tolerance": bool((not ik.success) and ik_success_for_acceptance),
        "ik_feasible": None if ik.feasible is None else bool(ik.feasible),
        "action_soft_joint_limit_ok": bool(soft_joint_limit_ok),
        "action_soft_joint_limit_check": soft_joint_limit_check,
        "action_soft_limit_min_margin_rad": float(soft_joint_limit_check.get("min_margin_rad", 0.0)),
        "action_range_ok": bool(action_range_ok),
        "action_range_check": action_range_check,
        "final_action_feasible": bool(final_action_feasible),
        "position_ok": bool(position_ok),
        "rotation_ok": bool(rotation_ok),
        "reached_grasp_geometry_ok": bool(reached_grasp_geometry_ok),
        "reached_grasp_geometry_check": reached_grasp_geometry,
        "ik_position_error_m": float(ik.position_error_m),
        "ik_rotation_error_rad": float(ik.rotation_error_rad),
        "max_required_ik_position_error_m": float(ik.position_error_m),
        "max_required_ik_rotation_error_rad": float(ik.rotation_error_rad),
        "position_tolerance_m": float(position_tolerance),
        "rotation_tolerance_rad": float(args_cli.curobo_rotation_tol),
        "piper_gripper_base_pose_w": {
            "position": grasp_pos.astype(float).tolist(),
            "quat_wxyz": [float(v) for v in grasp_quat],
        },
        "mode_note": mode_note,
        "baseline_action_prior": baseline_action_prior_report(name, object_pose, [stage_record]),
        "stage_record": stage_record,
    }


def curobo_probe_candidate(
    planner,
    robot,
    current_q: np.ndarray,
    name: str,
    candidate: dict,
    object_pose: dict,
    target_points_w: np.ndarray,
    obstacle_points_w: np.ndarray,
) -> dict:
    grasp_pos, grasp_quat, approach_axis, mode_note = grasp_pose_for_object(
        args_cli,
        name,
        candidate,
        object_pose,
    )
    pregrasp_distance, pregrasp_pos, stage_distance, stage_pos = pregrasp_layout_for_candidate(
        candidate,
        grasp_pos,
        approach_axis,
    )
    lift_pos = grasp_pos - approach_axis * float(args_cli.lift_distance)
    stages_w = [
        ("pregrasp_stage", stage_pos, grasp_quat),
        ("pregrasp", pregrasp_pos, grasp_quat),
        ("grasp", grasp_pos, grasp_quat),
        ("lift", lift_pos, grasp_quat),
    ]
    q = np.asarray(current_q, dtype=np.float32).copy()
    stages = []
    max_position_error = 0.0
    max_rotation_error = 0.0
    all_success = True
    critical_labels = {"pregrasp", "grasp", "lift"} if bool(args_cli.relax_curobo_pregrasp_stage) else {
        "pregrasp_stage",
        "pregrasp",
        "grasp",
        "lift",
    }
    all_required_success = True
    all_soft_joint_limit_ok = True
    all_required_soft_joint_limit_ok = True
    all_action_range_ok = True
    all_required_action_range_ok = True
    max_required_position_error = 0.0
    max_required_rotation_error = 0.0
    reached_grasp_geometry = {"enabled": False, "safe": True}
    reached_pose_filter_enabled = not bool(args_cli.disable_curobo_reached_pose_pc_filter)
    checked_width, width_info = checked_piper_jaw_width(candidate)
    soft_limits_np = action_soft_limits_for_object(robot, name)
    default_arm_q = action_default_arm_joint_pos_for_robot(robot)
    position_tolerance = curobo_position_tolerance_for_object(name)
    for label, pos_w, quat_w in stages_w:
        pos_root, quat_root = world_pose_to_robot_root(robot, pos_w, np.asarray(quat_w, dtype=np.float64))
        ik = planner.solve_ik(q, pos_root, quat_root, return_seeds=1)
        soft_joint_limit_ok, soft_joint_limit_check = action_soft_limit_report(ik.joint_position, soft_limits_np, name)
        action_range_ok, action_range_check = action_range_report(ik.joint_position, default_arm_q)
        stage_position_ok = float(ik.position_error_m) <= float(position_tolerance)
        stage_rotation_ok = float(ik.rotation_error_rad) <= float(args_cli.curobo_rotation_tol)
        ik_success_for_acceptance = bool(
            ik.success
            or (
                object_accepts_tolerance_ik_success(name)
                and stage_position_ok
                and stage_rotation_ok
            )
        )
        stage_entry = {
            "label": label,
            "critical_for_acceptance": bool(label in critical_labels),
            "ik_success": bool(ik.success),
            "ik_success_used_for_acceptance": bool(ik_success_for_acceptance),
            "ik_success_by_tolerance": bool((not ik.success) and ik_success_for_acceptance),
            "ik_position_within_tolerance": bool(stage_position_ok),
            "ik_rotation_within_tolerance": bool(stage_rotation_ok),
            "ik_feasible": None if ik.feasible is None else bool(ik.feasible),
            "action_soft_joint_limit_ok": bool(soft_joint_limit_ok),
            "action_soft_joint_limit_check": soft_joint_limit_check,
            "action_range_ok": bool(action_range_ok),
            "action_range_check": action_range_check,
            "ik_position_error_m": float(ik.position_error_m),
            "ik_rotation_error_rad": float(ik.rotation_error_rad),
            "solve_time_s": float(ik.solve_time_s),
            "target_pos_w": np.asarray(pos_w, dtype=float).tolist(),
            "target_pos_root": pos_root.astype(float).tolist(),
            "target_quat_wxyz": [float(v) for v in quat_w],
            "target_quat_root_wxyz": quat_root.astype(float).tolist(),
            "joint_position": ik.joint_position.astype(float).tolist(),
        }
        if reached_pose_filter_enabled and label == "grasp":
            fk_pos_root, fk_quat_root = planner.compute_fk(ik.joint_position)
            fk_pos_w, fk_quat_w = robot_root_pose_to_world(robot, fk_pos_root, fk_quat_root)
            piper_rotation = rotation_matrix_from_quat_wxyz(fk_quat_w)
            reached_grasp_geometry = piper_geometry_check_at_base(
                target_points_w,
                obstacle_points_w,
                piper_rotation,
                fk_pos_w,
                checked_width,
                width_info,
                name,
            )
            reached_grasp_geometry["target_pos_w"] = np.asarray(pos_w, dtype=float).tolist()
            reached_grasp_geometry["target_quat_wxyz"] = [float(v) for v in quat_w]
            reached_grasp_geometry["fk_pos_root"] = fk_pos_root.astype(float).tolist()
            reached_grasp_geometry["fk_quat_root_wxyz"] = fk_quat_root.astype(float).tolist()
            reached_grasp_geometry["fk_pos_w"] = fk_pos_w.astype(float).tolist()
            reached_grasp_geometry["fk_quat_wxyz"] = fk_quat_w.astype(float).tolist()
            reached_grasp_geometry["fk_position_delta_from_target_m"] = float(
                np.linalg.norm(fk_pos_w.astype(np.float64) - np.asarray(pos_w, dtype=np.float64))
            )
            stage_entry["reached_pose_w"] = {
                "position": fk_pos_w.astype(float).tolist(),
                "quat_wxyz": fk_quat_w.astype(float).tolist(),
            }
            stage_entry["reached_pose_pc_geometry_check"] = reached_grasp_geometry
        stages.append(stage_entry)
        all_success = all_success and bool(ik_success_for_acceptance)
        all_soft_joint_limit_ok = all_soft_joint_limit_ok and bool(soft_joint_limit_ok)
        all_action_range_ok = all_action_range_ok and bool(action_range_ok)
        max_position_error = max(max_position_error, float(ik.position_error_m))
        max_rotation_error = max(max_rotation_error, float(ik.rotation_error_rad))
        if label in critical_labels:
            all_required_success = all_required_success and bool(ik_success_for_acceptance)
            all_required_soft_joint_limit_ok = all_required_soft_joint_limit_ok and bool(soft_joint_limit_ok)
            all_required_action_range_ok = all_required_action_range_ok and bool(action_range_ok)
            max_required_position_error = max(max_required_position_error, float(ik.position_error_m))
            max_required_rotation_error = max(max_required_rotation_error, float(ik.rotation_error_rad))
        q = ik.joint_position.astype(np.float32)

    position_ok = max_required_position_error <= float(position_tolerance)
    rotation_ok = max_required_rotation_error <= float(args_cli.curobo_rotation_tol)
    strict_position_ok = max_position_error <= float(position_tolerance)
    strict_rotation_ok = max_rotation_error <= float(args_cli.curobo_rotation_tol)
    reached_grasp_geometry_ok = bool(reached_grasp_geometry.get("safe", True))
    required_action_feasible = bool(
        all_required_success and all_required_soft_joint_limit_ok and all_required_action_range_ok
    )
    strict_action_feasible = bool(all_success and all_soft_joint_limit_ok and all_action_range_ok)
    required_soft_margins = [
        float((stage.get("action_soft_joint_limit_check") or {}).get("min_margin_rad", 0.0))
        for stage in stages
        if bool(stage.get("critical_for_acceptance"))
    ]
    all_soft_margins = [
        float((stage.get("action_soft_joint_limit_check") or {}).get("min_margin_rad", 0.0))
        for stage in stages
    ]
    baseline_prior = baseline_action_prior_report(name, object_pose, stages)
    return {
        "rank": int(candidate.get("source_rank", 0)),
        "score": float(candidate.get("score", 0.0)),
        "ok": bool(required_action_feasible and position_ok and rotation_ok and reached_grasp_geometry_ok),
        "all_ik_success": bool(all_success),
        "all_required_ik_success": bool(all_required_success),
        "all_soft_joint_limit_ok": bool(all_soft_joint_limit_ok),
        "all_required_soft_joint_limit_ok": bool(all_required_soft_joint_limit_ok),
        "all_action_range_ok": bool(all_action_range_ok),
        "all_required_action_range_ok": bool(all_required_action_range_ok),
        "all_required_action_feasible": bool(required_action_feasible),
        "min_required_action_soft_limit_margin_rad": float(min(required_soft_margins, default=0.0)),
        "min_all_action_soft_limit_margin_rad": float(min(all_soft_margins, default=0.0)),
        "relaxed_pregrasp_stage": bool(args_cli.relax_curobo_pregrasp_stage),
        "critical_stage_labels": sorted(critical_labels),
        "strict_all_stage_ok": bool(
            strict_action_feasible and strict_position_ok and strict_rotation_ok and reached_grasp_geometry_ok
        ),
        "position_ok": bool(position_ok),
        "rotation_ok": bool(rotation_ok),
        "reached_grasp_geometry_ok": bool(reached_grasp_geometry_ok),
        "reached_grasp_geometry_check": reached_grasp_geometry,
        "max_ik_position_error_m": float(max_position_error),
        "max_ik_rotation_error_rad": float(max_rotation_error),
        "max_required_ik_position_error_m": float(max_required_position_error),
        "max_required_ik_rotation_error_rad": float(max_required_rotation_error),
        "position_tolerance_m": float(position_tolerance),
        "rotation_tolerance_rad": float(args_cli.curobo_rotation_tol),
        "piper_gripper_base_pose_w": {
            "position": grasp_pos.astype(float).tolist(),
            "quat_wxyz": [float(v) for v in grasp_quat],
        },
        "pregrasp_pose_w": {
            "position": pregrasp_pos.astype(float).tolist(),
            "quat_wxyz": [float(v) for v in grasp_quat],
        },
        "pregrasp_distance_m": float(pregrasp_distance),
        "pregrasp_stage_pose_w": {
            "position": stage_pos.astype(float).tolist(),
            "quat_wxyz": [float(v) for v in grasp_quat],
        },
        "pregrasp_stage_distance_m": float(stage_distance) if stage_distance is not None else None,
        "lift_pose_w": {
            "position": lift_pos.astype(float).tolist(),
            "quat_wxyz": [float(v) for v in grasp_quat],
        },
        "mode_note": mode_note,
        "baseline_action_prior": baseline_prior,
        "stages": stages,
    }


def piper_finger_collision_stats(
    object_points_w: np.ndarray,
    candidate: dict,
    offset_distance: float,
    offset_axis: np.ndarray,
    clearance: float,
) -> dict:
    translation = np.asarray(candidate["translation"], dtype=np.float64)
    piper_rotation, approach_axis = piper_rotation_for_candidate(candidate)
    side_axis = normalize_or_fallback(piper_rotation[:, 0], np.array([1.0, 0.0, 0.0]))
    jaw_axis = normalize_or_fallback(piper_rotation[:, 1], np.array([0.0, 1.0, 0.0]))
    jaw_width = float(np.clip(float(candidate.get("width", 0.065)), 0.030, 0.095))
    finger_length = PIPER_FINGER_LENGTH_M
    finger_width = PIPER_FINGER_WIDTH_M
    finger_depth = PIPER_FINGER_DEPTH_M
    clearance = max(0.0, float(clearance))

    fingertip_center = translation + np.asarray(offset_axis, dtype=np.float64) * float(offset_distance)
    finger_root = fingertip_center - approach_axis * finger_length
    inside_any = np.zeros(object_points_w.shape[0], dtype=bool)
    per_finger = {}
    for side_sign, label in [(-1.0, "left"), (1.0, "right")]:
        center_offset = jaw_axis * side_sign * (jaw_width * 0.5 + finger_width * 0.5)
        root_center = finger_root + center_offset
        rel = object_points_w - root_center[None, :]
        along = rel @ approach_axis
        jaw = rel @ jaw_axis
        side = rel @ side_axis
        inside = (
            (along >= -clearance)
            & (along <= finger_length + clearance)
            & (np.abs(jaw) <= finger_width * 0.5 + clearance)
            & (np.abs(side) <= finger_depth * 0.5 + clearance)
        )
        per_finger[label] = int(np.count_nonzero(inside))
        inside_any |= inside

    count = int(np.count_nonzero(inside_any))
    return {
        "offset_m": float(offset_distance),
        "collision_point_count": count,
        "per_finger_collision_points": per_finger,
        "fingertip_center_w": fingertip_center.astype(float).tolist(),
        "finger_root_center_w": finger_root.astype(float).tolist(),
    }


def piper_gripper_collision_stats_at_pose(
    object_points_w: np.ndarray,
    candidate: dict,
    gripper_pos_w: np.ndarray,
    clearance: float,
) -> dict:
    piper_rotation, approach_axis = piper_rotation_for_candidate(candidate)
    side_axis = normalize_or_fallback(piper_rotation[:, 0], np.array([1.0, 0.0, 0.0]))
    jaw_axis = normalize_or_fallback(piper_rotation[:, 1], np.array([0.0, 1.0, 0.0]))
    jaw_width = float(np.clip(float(candidate.get("width", 0.065)), 0.030, 0.095))
    finger_length = PIPER_FINGER_LENGTH_M
    finger_width = PIPER_FINGER_WIDTH_M
    finger_depth = PIPER_FINGER_DEPTH_M
    palm_thickness = PIPER_PALM_APPROACH_THICKNESS_M
    clearance = max(0.0, float(clearance))

    gripper_pos = np.asarray(gripper_pos_w, dtype=np.float64)
    finger_root = gripper_pos - approach_axis * finger_length
    inside_any = np.zeros(object_points_w.shape[0], dtype=bool)
    components: dict[str, int] = {}
    for side_sign, label in [(-1.0, "left_finger"), (1.0, "right_finger")]:
        center_offset = jaw_axis * side_sign * (jaw_width * 0.5 + finger_width * 0.5)
        root_center = finger_root + center_offset
        rel = object_points_w - root_center[None, :]
        along = rel @ approach_axis
        jaw = rel @ jaw_axis
        side = rel @ side_axis
        inside = (
            (along >= -clearance)
            & (along <= finger_length + clearance)
            & (np.abs(jaw) <= finger_width * 0.5 + clearance)
            & (np.abs(side) <= finger_depth * 0.5 + clearance)
        )
        components[label] = int(np.count_nonzero(inside))
        inside_any |= inside

    palm_center = finger_root - approach_axis * (palm_thickness * 0.5)
    rel = object_points_w - palm_center[None, :]
    palm_along = rel @ approach_axis
    palm_jaw = rel @ jaw_axis
    palm_side = rel @ side_axis
    palm_inside = (
        (np.abs(palm_along) <= palm_thickness * 0.5 + clearance)
        & (np.abs(palm_jaw) <= (jaw_width * 0.5 + finger_width) + clearance)
        & (np.abs(palm_side) <= finger_depth * 0.5 + clearance)
    )
    components["palm_base"] = int(np.count_nonzero(palm_inside))
    inside_any |= palm_inside

    return {
        "collision_point_count": int(np.count_nonzero(inside_any)),
        "component_collision_points": components,
        "gripper_pos_w": gripper_pos.astype(float).tolist(),
    }


def pregrasp_layout_for_candidate(candidate: dict, grasp_pos: np.ndarray, approach_axis: np.ndarray) -> tuple[float, np.ndarray, float | None, np.ndarray]:
    pregrasp_distance = float(candidate.get("pregrasp_distance_override_m", args_cli.pregrasp_distance))
    pregrasp_pos = grasp_pos - approach_axis * pregrasp_distance
    stage_distance_value = candidate.get("pregrasp_stage_distance_override_m")
    if candidate.get("pregrasp_stage_mode_override") == "straight_approach_axis" and stage_distance_value is not None:
        stage_distance = float(stage_distance_value)
        stage_pos = grasp_pos - approach_axis * stage_distance
    else:
        stage_distance = None
        stage_pos = np.array([pregrasp_pos[0], pregrasp_pos[1], max(float(CARRY_Z), float(pregrasp_pos[2]))])
    return pregrasp_distance, pregrasp_pos, stage_distance, stage_pos


def straight_approach_clearance_search(record: dict, name: str, candidate: dict, object_pose: dict) -> dict:
    if args_cli.disable_straight_approach_pc_search or name not in args_cli.pc_offset_search_objects:
        return {"enabled": False, "accepted": True, "disabled": True}

    object_points_w, cloud_meta = load_fused_object_cloud_world(record)
    grasp_pos, _, approach_axis, _ = grasp_pose_for_object(args_cli, name, candidate, object_pose)
    min_points = max(1, int(args_cli.approach_pc_collision_min_points))
    clearance = float(args_cli.approach_pc_collision_clearance)
    min_distance = max(float(args_cli.pregrasp_distance), float(args_cli.pc_offset_step))
    max_distance = max(min_distance, float(args_cli.straight_approach_max_distance))
    step = max(1e-5, float(args_cli.straight_approach_step))
    samples_per_segment = max(2, int(args_cli.approach_pc_collision_samples))
    final_fraction = float(np.clip(args_cli.approach_pc_collision_final_fraction, 0.0, 1.0))

    distances = [float(value) for value in np.arange(min_distance, max_distance + step * 0.5, step)]
    if not distances or distances[-1] < max_distance:
        distances.append(max_distance)

    tested = []
    accepted = []
    for distance in distances:
        pregrasp_pos = grasp_pos - approach_axis * distance
        pregrasp_stats = piper_gripper_collision_stats_at_pose(object_points_w, candidate, pregrasp_pos, clearance)
        first_illegal = None
        max_collision = 0
        for fraction in np.linspace(0.0, final_fraction, samples_per_segment):
            pos = pregrasp_pos * (1.0 - float(fraction)) + grasp_pos * float(fraction)
            stats = piper_gripper_collision_stats_at_pose(object_points_w, candidate, pos, clearance)
            count = int(stats["collision_point_count"])
            max_collision = max(max_collision, count)
            if first_illegal is None and count >= min_points:
                first_illegal = {
                    "fraction": float(fraction),
                    "collision_point_count": count,
                    "component_collision_points": stats["component_collision_points"],
                    "gripper_pos_w": stats["gripper_pos_w"],
                }

        entry = {
            "pregrasp_distance_m": float(distance),
            "pregrasp_collision_point_count": int(pregrasp_stats["collision_point_count"]),
            "pregrasp_component_collision_points": pregrasp_stats["component_collision_points"],
            "max_early_approach_collision_point_count": int(max_collision),
            "first_illegal": first_illegal,
            "accepted": first_illegal is None,
        }
        tested.append(entry)
        if entry["accepted"]:
            accepted.append(entry)

    selected = accepted[-1] if accepted else None
    if selected is None:
        selected = tested[-1] if tested else {
            "pregrasp_distance_m": min_distance,
            "pregrasp_collision_point_count": 0,
            "max_early_approach_collision_point_count": 0,
            "first_illegal": None,
            "accepted": False,
        }
    selected_distance = float(selected["pregrasp_distance_m"])
    stage_distance = min(max_distance + max(0.0, float(args_cli.straight_approach_stage_extra)), selected_distance + max(0.0, float(args_cli.straight_approach_stage_extra)))
    approach_lift_distance = max(float(args_cli.lift_distance), selected_distance)
    return {
        "enabled": True,
        "accepted": bool(selected["accepted"]),
        "selected_pregrasp_distance_m": selected_distance,
        "selected_stage_distance_m": float(stage_distance),
        "selected_approach_lift_distance_m": float(approach_lift_distance),
        "selected_reason": "longest_clear_straight_approach" if selected["accepted"] else "no_clear_straight_approach_before_max_distance",
        "min_distance_m": float(min_distance),
        "max_distance_m": float(max_distance),
        "step_m": float(step),
        "min_points": int(min_points),
        "clearance_m": float(clearance),
        "checked_until_fraction": float(final_fraction),
        "cloud": cloud_meta,
        "tested_distances": tested,
        "note": (
            "Final gripper/object point-cloud overlap can be valid contact. This search chooses "
            "a longer pregrasp on the straight approach axis whose early approach segment remains "
            "clear until the configured final contact window."
        ),
    }


def point_cloud_approach_collision_check(
    record: dict,
    name: str,
    candidate: dict,
    object_pose: dict,
) -> dict:
    if args_cli.disable_approach_pc_collision_filter:
        return {"enabled": False, "safe": True, "disabled": True}

    object_points_w, cloud_meta = load_fused_object_cloud_world(record)
    grasp_pos, _, approach_axis, _ = grasp_pose_for_object(args_cli, name, candidate, object_pose)
    _, pregrasp_pos, _, stage_pos = pregrasp_layout_for_candidate(candidate, grasp_pos, approach_axis)
    min_points = max(1, int(args_cli.approach_pc_collision_min_points))
    clearance = float(args_cli.approach_pc_collision_clearance)
    samples_per_segment = max(2, int(args_cli.approach_pc_collision_samples))
    final_fraction = float(np.clip(args_cli.approach_pc_collision_final_fraction, 0.0, 1.0))

    tested: list[dict] = []
    first_illegal = None

    def add_segment(name_label: str, start: np.ndarray, end: np.ndarray, max_fraction: float) -> None:
        nonlocal first_illegal
        if max_fraction <= 0.0:
            fractions = [0.0]
        else:
            fractions = [float(v) for v in np.linspace(0.0, max_fraction, samples_per_segment)]
        for fraction in fractions:
            pos = start * (1.0 - fraction) + end * fraction
            stats = piper_gripper_collision_stats_at_pose(object_points_w, candidate, pos, clearance)
            entry = {
                "segment": name_label,
                "fraction": float(fraction),
                "collision_point_count": int(stats["collision_point_count"]),
                "component_collision_points": stats["component_collision_points"],
                "gripper_pos_w": stats["gripper_pos_w"],
                "illegal": bool(stats["collision_point_count"] >= min_points),
            }
            tested.append(entry)
            if first_illegal is None and entry["illegal"]:
                first_illegal = entry

    add_segment("stage_to_pregrasp", stage_pos, pregrasp_pos, 1.0)
    add_segment("pregrasp_to_grasp_early", pregrasp_pos, grasp_pos, final_fraction)

    max_collision = max((int(item["collision_point_count"]) for item in tested), default=0)
    return {
        "enabled": True,
        "safe": first_illegal is None,
        "min_points": min_points,
        "clearance_m": clearance,
        "samples_per_segment": samples_per_segment,
        "pregrasp_to_grasp_checked_until_fraction": final_fraction,
        "max_collision_point_count": max_collision,
        "first_illegal": first_illegal,
        "cloud": cloud_meta,
        "tested_samples": tested,
        "note": (
            "This is a point-cloud swept-gripper filter, not a full MoveIt planning-scene solve. "
            "It rejects candidates whose Piper gripper/finger proxy collides with the segmented "
            "object cloud before the final allowed grasp-contact portion."
        ),
    }


def search_point_cloud_offset(record: dict, name: str, candidate: dict, object_pose: dict) -> dict:
    object_points_w, cloud_meta = load_fused_object_cloud_world(record)
    translation = np.asarray(candidate["translation"], dtype=np.float64)
    _, approach_axis = piper_rotation_for_candidate(candidate)
    offset_mode = args_cli.gripper_base_offset_mode
    offset_axis = offset_axis_for_mode(offset_mode, translation, approach_axis, object_pose)
    step = max(1e-5, float(args_cli.pc_offset_step))
    max_offset = max(0.0, float(args_cli.pc_offset_max))
    min_points = max(1, int(args_cli.pc_offset_collision_min_points))
    offsets = [float(value) for value in np.arange(0.0, max_offset + step * 0.5, step)]
    if not offsets or offsets[-1] < max_offset:
        offsets.append(max_offset)

    tested = []
    selected = None
    for offset in offsets:
        offset = min(float(offset), max_offset)
        stats = piper_finger_collision_stats(
            object_points_w,
            candidate,
            offset,
            offset_axis,
            args_cli.pc_offset_collision_clearance,
        )
        left_count = int(stats["per_finger_collision_points"].get("left", 0))
        right_count = int(stats["per_finger_collision_points"].get("right", 0))
        stats["collides"] = bool(stats["collision_point_count"] >= min_points)
        stats["both_fingers_collide"] = bool(left_count >= min_points and right_count >= min_points)
        tested.append(stats)
        accepts_offset = stats["both_fingers_collide"] if args_cli.pc_offset_require_both_fingers else stats["collides"]
        if accepts_offset:
            selected = stats
            break

    if selected is None:
        selected = tested[-1] if tested else {
            "offset_m": 0.0,
            "collision_point_count": 0,
            "per_finger_collision_points": {"left": 0, "right": 0},
            "collides": False,
            "both_fingers_collide": False,
        }
        reason = "no_both_finger_collision_before_max_offset" if args_cli.pc_offset_require_both_fingers else "no_collision_before_max_offset"
    else:
        reason = "first_both_finger_collision" if args_cli.pc_offset_require_both_fingers else "first_collision"

    return {
        "enabled": True,
        "object": name,
        "mode": offset_mode,
        "require_both_fingers": bool(args_cli.pc_offset_require_both_fingers),
        "step_m": step,
        "max_offset_m": max_offset,
        "selected_offset_m": float(selected["offset_m"]),
        "selected_reason": reason,
        "selected_accepted": bool(
            selected.get("both_fingers_collide")
            if args_cli.pc_offset_require_both_fingers
            else selected.get("collides")
        ),
        "selected_collision_point_count": int(selected["collision_point_count"]),
        "selected_per_finger_collision_points": selected.get("per_finger_collision_points"),
        "collision_min_points": min_points,
        "collision_clearance_m": float(args_cli.pc_offset_collision_clearance),
        "offset_axis_w": offset_axis.astype(float).tolist(),
        "piper_finger_dimensions_m": {
            "length": PIPER_FINGER_LENGTH_M,
            "opening_axis_width": PIPER_FINGER_WIDTH_M,
            "side_axis_depth": PIPER_FINGER_DEPTH_M,
        },
        "cloud": cloud_meta,
        "tested_offsets": tested,
        "note": (
            "The selected raw IK-feasible AnyGrasp pose is translated in this mode until "
            "the real-scale Piper finger solid first contains at least collision_min_points "
            "from the fused segmented object cloud."
        ),
    }


def apply_selected_offset_search(
    record: dict,
    name: str,
    selected_pose: dict,
    selected_eval: dict | None,
    object_pose: dict,
) -> tuple[dict, dict | None, dict | None]:
    if name not in args_cli.pc_offset_search_objects:
        return selected_pose, selected_eval, None

    search = search_point_cloud_offset(record, name, selected_pose, object_pose)
    selected_pose = copy.deepcopy(selected_pose)
    selected_pose["gripper_base_offset_override_m"] = float(search["selected_offset_m"])
    selected_pose["gripper_base_offset_mode_override"] = search["mode"]
    selected_pose["pc_offset_search"] = search

    selected_eval = copy.deepcopy(selected_eval) if selected_eval is not None else {}
    selected_eval["ik_probe_used_zero_offset"] = bool(args_cli.ik_filter_raw_anygrasp)
    selected_eval["ik_probe_offset_applied_after_selection"] = True
    selected_eval["pc_offset_search"] = search
    selected_eval["raw_ik_probe_piper_gripper_base_pose_w"] = selected_eval.get("piper_gripper_base_pose_w")
    selected_eval["raw_ik_probe_pregrasp_pose_w"] = selected_eval.get("pregrasp_pose_w")
    grasp_pos, grasp_quat, approach_axis, mode_note = grasp_pose_for_object(
        args_cli,
        name,
        selected_pose,
        object_pose,
    )
    _, pregrasp_pos, _, _ = pregrasp_layout_for_candidate(selected_pose, grasp_pos, approach_axis)
    selected_eval["piper_gripper_base_pose_w"] = {
        "position": grasp_pos.astype(float).tolist(),
        "quat_wxyz": [float(v) for v in grasp_quat],
    }
    selected_eval["pregrasp_pose_w"] = {
        "position": pregrasp_pos.astype(float).tolist(),
        "quat_wxyz": [float(v) for v in grasp_quat],
    }
    selected_eval["mode_note"] = mode_note
    return selected_pose, selected_eval, search


def apply_straight_approach_search(
    record: dict,
    name: str,
    selected_pose: dict,
    selected_eval: dict | None,
    object_pose: dict,
) -> tuple[dict, dict | None, dict | None]:
    search = straight_approach_clearance_search(record, name, selected_pose, object_pose)
    if not search.get("enabled"):
        return selected_pose, selected_eval, search

    selected_pose = copy.deepcopy(selected_pose)
    selected_pose["pregrasp_distance_override_m"] = float(search["selected_pregrasp_distance_m"])
    selected_pose["pregrasp_stage_distance_override_m"] = float(search["selected_stage_distance_m"])
    selected_pose["pregrasp_stage_mode_override"] = "straight_approach_axis"
    selected_pose["approach_lift_distance_override_m"] = float(search["selected_approach_lift_distance_m"])
    selected_pose["straight_approach_search"] = search

    selected_eval = copy.deepcopy(selected_eval) if selected_eval is not None else {}
    selected_eval["straight_approach_search"] = search
    grasp_pos, grasp_quat, approach_axis, _ = grasp_pose_for_object(args_cli, name, selected_pose, object_pose)
    pregrasp_distance, pregrasp_pos, stage_distance, stage_pos = pregrasp_layout_for_candidate(
        selected_pose,
        grasp_pos,
        approach_axis,
    )
    selected_eval["pregrasp_pose_w"] = {
        "position": pregrasp_pos.astype(float).tolist(),
        "quat_wxyz": [float(v) for v in grasp_quat],
    }
    selected_eval["pregrasp_distance_m"] = float(pregrasp_distance)
    selected_eval["pregrasp_stage_pose_w"] = {
        "position": stage_pos.astype(float).tolist(),
        "quat_wxyz": [float(v) for v in grasp_quat],
    }
    selected_eval["pregrasp_stage_distance_m"] = float(stage_distance) if stage_distance is not None else None
    return selected_pose, selected_eval, search


def save_pc_offset_visualization(record: dict, selection: dict) -> dict | None:
    search = selection.get("pc_offset_search")
    source = "pc_offset_search"
    if not isinstance(search, dict):
        selected_rank = selection.get("selected_rank")
        selected_eval = next(
            (
                candidate
                for candidate in selection.get("candidates", [])
                if int(candidate.get("rank", -1)) == int(selected_rank or -999)
            ),
            None,
        )
        final_geometry = (selected_eval or {}).get("final_geometry_check") or {}
        selected_geometry = final_geometry.get("selected") or {}
        if selected_geometry.get("offset_m") is None or selected_geometry.get("offset_mode") is None:
            return None
        search = {
            "mode": selected_geometry.get("offset_mode"),
            "selected_offset_m": selected_geometry.get("offset_m"),
        }
        source = "final_geometry_check"
    if not isinstance(search, dict):
        return None
    anygrasp_dir = Path(record["anygrasp_result_path"]).parent
    output_dir = anygrasp_dir / "fused_pc_selected_grasp"
    log_path = anygrasp_dir / "visualize_selected_offset.log"
    command = [
        sys.executable,
        "scripts/visualize_anygrasp_fused_pc.py",
        "--anygrasp-dir",
        str(anygrasp_dir),
        "--mode",
        "selected",
        "--offset-mode",
        str(search.get("mode", "finger_centerline")),
        "--offset-distance",
        f"{float(search.get('selected_offset_m', 0.0)):.9f}",
        "--output-dir",
        str(output_dir),
    ]
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    artifact = {
        "source": source,
        "command": command,
        "log": str(log_path),
        "returncode": int(proc.returncode),
        "output_dir": str(output_dir),
        "projection_png": str(output_dir / "fused_cloud_grasps_projection.png"),
        "cloud_with_gripper_ply": str(output_dir / "fused_cloud_with_gripper_primary_camera.ply"),
        "scene_metadata": str(output_dir / "scene_metadata.json"),
    }
    if proc.returncode != 0:
        print(f"[WARN] Selected-offset visualization failed. See {log_path}", flush=True)
    return artifact


def waypoint_object_name(waypoint: dict) -> str | None:
    explicit = waypoint.get("execution_object_name")
    if explicit in OBJECTS:
        return str(explicit)
    waypoint_name = str(waypoint.get("name", ""))
    for object_name in OBJECTS:
        if f"_{object_name}_" in waypoint_name or waypoint_name.endswith(f"_{object_name}"):
            return object_name
    return None


def settle_ik_pose(
    env,
    obs,
    robot,
    controller: CartesianController,
    arm_ids,
    gripper_ids,
    default_jpos,
    position: np.ndarray,
    quat: list[float] | np.ndarray,
    steps: int,
    label: str,
) -> tuple[dict, dict]:
    pos_t = torch.tensor([position], dtype=torch.float32, device=env.unwrapped.device)
    quat_t = torch.tensor([quat], dtype=torch.float32, device=env.unwrapped.device)
    gripper_t = torch.tensor([GRIPPER_OPEN], dtype=torch.float32, device=env.unwrapped.device)
    last_action = None
    stopped = None
    for _ in range(max(1, int(steps))):
        arm_des = controller.compute(pos_t, quat_t)
        target = robot.data.joint_pos.clone()
        target[:, arm_ids] = arm_des
        target[:, gripper_ids] = gripper_t
        action = (target - default_jpos) / ACTION_SCALE
        last_action = action
        obs, _, terminated, truncated, _ = env.step(action)
        robot.update(dt=env.unwrapped.physics_dt)
        if terminated.any() or truncated.any():
            stopped = {
                "terminated": bool(terminated.any().item()),
                "truncated": bool(truncated.any().item()),
            }
            break

    ee_pose = robot.data.body_pose_w[0, controller.ee_idx]
    ee_pos = np.asarray(tensor_to_list(ee_pose[:3]), dtype=np.float64)
    ee_quat = tensor_to_list(ee_pose[3:])
    return obs, {
        "label": label,
        "target_position": position.astype(float).tolist(),
        "target_quat_wxyz": [float(v) for v in quat],
        "steps": int(steps),
        "ee_position": ee_pos.astype(float).tolist(),
        "ee_quat_wxyz": ee_quat,
        "position_error_m": float(np.linalg.norm(ee_pos - position)),
        "orientation_abs_dot": quat_abs_dot(ee_quat, quat),
        "stop_reason": stopped,
        "last_action": tensor_to_list(last_action[0]) if last_action is not None else None,
    }


def reset_robot_for_ik_probe(env, obs, robot, default_jpos) -> dict:
    robot.write_joint_state_to_sim(
        default_jpos,
        torch.zeros_like(robot.data.default_joint_vel),
    )
    robot.set_joint_position_target(default_jpos)
    env.unwrapped.scene.write_data_to_sim()
    env.unwrapped.sim.forward()
    robot.update(dt=env.unwrapped.physics_dt)
    try:
        return env.unwrapped.observation_manager.compute()
    except Exception:
        return obs


def probe_candidate_ik(
    env,
    obs,
    robot,
    controller: CartesianController,
    arm_ids,
    gripper_ids,
    default_jpos,
    name: str,
    candidate: dict,
    object_pose: dict,
    reset_jpos=None,
) -> tuple[dict, dict]:
    grasp_pos, grasp_quat, approach_axis, mode_note = grasp_pose_for_object(
        args_cli,
        name,
        candidate,
        object_pose,
    )
    pregrasp_distance, pregrasp_pos, stage_distance, stage_pos = pregrasp_layout_for_candidate(
        candidate,
        grasp_pos,
        approach_axis,
    )
    initial_pos = np.array([RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z], dtype=np.float64)

    obs = reset_robot_for_ik_probe(env, obs, robot, reset_jpos if reset_jpos is not None else default_jpos)
    object_start_positions = task_object_positions_w(env)
    object_motion_by_waypoint = []
    controller.reset()
    waypoints = []

    def record_waypoint(result: dict) -> None:
        motion = object_motion_from_start(env, object_start_positions)
        result["object_motion_from_probe_start"] = motion
        object_motion_by_waypoint.append(
            {
                "label": result["label"],
                "max_displacement_m": motion["max_displacement_m"],
                "target_displacement_m": motion.get("objects", {}).get(name, {}).get("displacement_m"),
                "objects": motion["objects"],
            }
        )
        waypoints.append(result)

    obs, result = settle_ik_pose(
        env,
        obs,
        robot,
        controller,
        arm_ids,
        gripper_ids,
        default_jpos,
        initial_pos,
        DEFAULT_QUAT_WXYZ,
        args_cli.ik_filter_stage_steps,
        "initial_retract",
    )
    record_waypoint(result)
    obs, result = settle_ik_pose(
        env,
        obs,
        robot,
        controller,
        arm_ids,
        gripper_ids,
        default_jpos,
        stage_pos,
        grasp_quat,
        args_cli.ik_filter_stage_steps,
        "pregrasp_stage",
    )
    record_waypoint(result)
    obs, result = settle_ik_pose(
        env,
        obs,
        robot,
        controller,
        arm_ids,
        gripper_ids,
        default_jpos,
        pregrasp_pos,
        grasp_quat,
        args_cli.ik_filter_target_steps,
        "pregrasp",
    )
    record_waypoint(result)
    obs, result = settle_ik_pose(
        env,
        obs,
        robot,
        controller,
        arm_ids,
        gripper_ids,
        default_jpos,
        grasp_pos,
        grasp_quat,
        args_cli.ik_filter_target_steps,
        "grasp",
    )
    record_waypoint(result)

    critical_waypoints = [item for item in waypoints if item["label"] in {"pregrasp", "grasp"}]
    position_ok = all(item["position_error_m"] <= args_cli.ik_filter_position_tol for item in critical_waypoints)
    orientation_ok = all(item["orientation_abs_dot"] >= args_cli.ik_filter_orientation_dot for item in critical_waypoints)
    stop_ok = all(item["stop_reason"] is None for item in critical_waypoints)
    max_critical_position_error = max(item["position_error_m"] for item in critical_waypoints)
    min_critical_orientation_dot = min(item["orientation_abs_dot"] for item in critical_waypoints)
    pregrasp_labels = {"initial_retract", "pregrasp_stage", "pregrasp"}
    pregrasp_motion = [item for item in object_motion_by_waypoint if item["label"] in pregrasp_labels]
    max_pregrasp_object_motion = max((float(item["max_displacement_m"]) for item in pregrasp_motion), default=0.0)
    max_pregrasp_target_motion = max(
        (
            float(item["target_displacement_m"])
            for item in pregrasp_motion
            if item.get("target_displacement_m") is not None
        ),
        default=0.0,
    )
    physics_obstacle_filter_enabled = not bool(args_cli.disable_ik_physics_obstacle_filter)
    physics_obstacle_ok = (
        (not physics_obstacle_filter_enabled)
        or max_pregrasp_object_motion <= float(args_cli.ik_obstacle_motion_tol)
    )
    return obs, {
        "rank": int(candidate.get("source_rank", 0)),
        "score": float(candidate.get("score", 0.0)),
        "ok": bool(position_ok and orientation_ok and stop_ok and physics_obstacle_ok),
        "position_ok": bool(position_ok),
        "orientation_ok": bool(orientation_ok),
        "stop_ok": bool(stop_ok),
        "physics_obstacle_ok": bool(physics_obstacle_ok),
        "physics_obstacle_filter_enabled": bool(physics_obstacle_filter_enabled),
        "max_pregrasp_object_motion_m": float(max_pregrasp_object_motion),
        "max_pregrasp_target_motion_m": float(max_pregrasp_target_motion),
        "ik_obstacle_motion_tolerance_m": float(args_cli.ik_obstacle_motion_tol),
        "object_motion_by_waypoint": object_motion_by_waypoint,
        "critical_waypoints": ["pregrasp", "grasp"],
        "max_critical_position_error_m": float(max_critical_position_error),
        "min_critical_orientation_abs_dot": float(min_critical_orientation_dot),
        "position_tolerance_m": float(args_cli.ik_filter_position_tol),
        "orientation_abs_dot_min": float(args_cli.ik_filter_orientation_dot),
        "piper_gripper_base_pose_w": {
            "position": grasp_pos.astype(float).tolist(),
            "quat_wxyz": [float(v) for v in grasp_quat],
        },
        "pregrasp_pose_w": {
            "position": pregrasp_pos.astype(float).tolist(),
            "quat_wxyz": [float(v) for v in grasp_quat],
        },
        "pregrasp_distance_m": float(pregrasp_distance),
        "pregrasp_stage_pose_w": {
            "position": stage_pos.astype(float).tolist(),
            "quat_wxyz": [float(v) for v in grasp_quat],
        },
        "pregrasp_stage_distance_m": float(stage_distance) if stage_distance is not None else None,
        "mode_note": mode_note,
        "waypoints": waypoints,
    }


def write_selected_grasp(record: dict, selected_pose: dict, selection: dict) -> None:
    pose_path = Path(record["final_grasp_pose_path"])
    pose_path.write_text(json_text(selected_pose), encoding="utf-8")

    result_path = Path(record["anygrasp_result_path"])
    anygrasp_result = record.get("anygrasp_result") or load_json(result_path)
    payload = anygrasp_result.setdefault("anygrasp", {})
    payload["ik_candidate_selection"] = selection
    payload["final_grasp_pose"] = selected_pose
    if selection.get("selected_rank") is not None:
        payload["selected_top_grasp_rank"] = selection["selected_rank"]
        payload["selected_score"] = selection.get("selected_score")
    result_path.write_text(json_text(anygrasp_result), encoding="utf-8")

    selection_path = pose_path.parent / "ik_candidate_selection.json"
    selection_path.write_text(json_text(selection), encoding="utf-8")
    save_selected_grasp_overlay(record, selected_pose, selection)
    visualization = save_pc_offset_visualization(record, selection)
    if visualization is not None:
        selection["pc_offset_visualization"] = visualization
        payload["ik_candidate_selection"] = selection
        result_path.write_text(json_text(anygrasp_result), encoding="utf-8")
        selection_path.write_text(json_text(selection), encoding="utf-8")
    record["anygrasp_result"] = anygrasp_result
    record["grasp_selection"] = {
        **selection,
        "selection_path": str(selection_path),
    }


def selected_curobo_candidate_from_record(record: dict) -> tuple[dict | None, dict | None]:
    selection = record.get("grasp_selection")
    if not isinstance(selection, dict):
        anygrasp_result = record.get("anygrasp_result") or {}
        selection = (anygrasp_result.get("anygrasp") or {}).get("ik_candidate_selection")
    if not isinstance(selection, dict):
        result_path = Path(record.get("anygrasp_result_path", ""))
        if result_path.exists():
            selection = (load_json(result_path).get("anygrasp") or {}).get("ik_candidate_selection")
    if not isinstance(selection, dict):
        return None, None

    candidates = selection.get("candidates") or []
    selected_rank = selection.get("selected_rank")
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("selected"):
            return selection, candidate
    if selected_rank is not None:
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            try:
                if int(candidate.get("rank", -1)) == int(selected_rank):
                    return selection, candidate
            except (TypeError, ValueError):
                continue
    return selection, None


def attach_curobo_joint_targets_to_request(request: dict, records: dict[str, dict]) -> list[dict]:
    """Use the selected CuRobo IK solutions as executable arm joint targets."""
    attached: list[dict] = []
    stage_to_waypoint_suffixes = {
        "pregrasp_stage": ["pregrasp_stage"],
        "pregrasp": ["pregrasp"],
        "grasp": ["grasp", "close"],
        "lift": ["approach_lift"],
    }

    for name, record in records.items():
        selection, candidate = selected_curobo_candidate_from_record(record)
        if not isinstance(selection, dict) or not isinstance(candidate, dict):
            continue
        if not bool(selection.get("selected_ok", False)):
            continue
        stages = candidate.get("stages") or []
        stage_by_label: dict[str, dict] = {}
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            label = str(stage.get("label", ""))
            joints = stage.get("joint_position")
            if (
                label in stage_to_waypoint_suffixes
                and bool(stage.get("ik_success_used_for_acceptance", stage.get("ik_success")))
                and bool(stage.get("action_soft_joint_limit_ok", True))
                and bool(stage.get("action_range_ok", True))
                and isinstance(joints, list)
                and len(joints) == len(ARM_JOINT_NAMES)
            ):
                stage_by_label[label] = stage

        for stage_label, waypoint_suffixes in stage_to_waypoint_suffixes.items():
            stage = stage_by_label.get(stage_label)
            if stage is None:
                continue
            for waypoint_suffix in waypoint_suffixes:
                waypoint_tail = f"_{name}_{waypoint_suffix}"
                for waypoint in request.get("waypoints", []):
                    waypoint_name = str(waypoint.get("name", ""))
                    if not waypoint_name.endswith(waypoint_tail):
                        continue
                    joints = [float(value) for value in stage["joint_position"]]
                    waypoint["curobo_joint_position"] = joints
                    waypoint["curobo_stage_label"] = stage_label
                    waypoint["curobo_ik_position_error_m"] = float(stage.get("ik_position_error_m", 0.0))
                    waypoint["curobo_ik_rotation_error_rad"] = float(stage.get("ik_rotation_error_rad", 0.0))
                    waypoint["curobo_target_pose_w"] = {
                        "position": stage.get("target_pos_w"),
                        "quat_wxyz": stage.get("target_quat_wxyz"),
                    }
                    waypoint["execution_object_name"] = name
                    attached.append(
                        {
                            "object": name,
                            "waypoint": waypoint_name,
                            "curobo_stage_label": stage_label,
                            "selected_rank": selection.get("selected_rank"),
                            "ik_position_error_m": waypoint["curobo_ik_position_error_m"],
                            "ik_rotation_error_rad": waypoint["curobo_ik_rotation_error_rad"],
                        }
                    )
                    break

    controller = request.setdefault("controller", {})
    source = request.setdefault("source", {})
    controller["curobo_joint_waypoint_execution"] = bool(attached)
    source["curobo_joint_waypoints"] = attached
    if attached:
        source["curobo_joint_waypoint_note"] = (
            "Pregrasp/grasp/approach_lift waypoints with curobo_joint_position are executed "
            "as selected CuRobo arm joint targets. Remaining waypoints keep the Cartesian pose controller."
        )
    return attached


def waypoint_policy_stage_name(waypoint: dict) -> str:
    """Return the execution-policy stage name for a waypoint."""
    waypoint_name = str(waypoint.get("name", ""))
    if waypoint_name.endswith("_close"):
        return "close"
    stage = str(waypoint.get("curobo_stage_label") or "")
    return stage


def apply_cartesian_execution_overrides_to_request(request: dict) -> list[dict]:
    """Move selected CuRobo joint targets to diagnostics for Cartesian execution."""
    objects = set(args_cli.cartesian_execution_objects)
    stages = set(args_cli.cartesian_execution_stages)
    if not objects or not stages:
        return []

    overrides: list[dict] = []
    for waypoint in request.get("waypoints", []):
        object_name = waypoint_object_name(waypoint)
        if object_name not in objects:
            continue
        if "curobo_joint_position" not in waypoint:
            continue
        policy_stage = waypoint_policy_stage_name(waypoint)
        if policy_stage not in stages:
            continue

        reference = waypoint.pop("curobo_joint_position")
        waypoint["cartesian_execution_override"] = True
        waypoint["cartesian_execution_stage"] = policy_stage
        waypoint["curobo_joint_position_reference"] = reference
        waypoint["cartesian_execution_note"] = (
            "CuRobo selected this waypoint and provided the reference joint target, "
            "but execution intentionally uses the IsaacLab CartesianController on the "
            "same target pose for solver/root-cause comparison."
        )
        overrides.append(
            {
                "object": object_name,
                "waypoint": waypoint.get("name"),
                "stage": policy_stage,
                "curobo_stage_label": waypoint.get("curobo_stage_label"),
                "curobo_ik_position_error_m": waypoint.get("curobo_ik_position_error_m"),
                "curobo_ik_rotation_error_rad": waypoint.get("curobo_ik_rotation_error_rad"),
            }
        )

    if overrides:
        controller = request.setdefault("controller", {})
        source = request.setdefault("source", {})
        controller["cartesian_execution_override"] = True
        controller["cartesian_execution_objects"] = sorted(objects)
        controller["cartesian_execution_stages"] = sorted(stages)
        source["cartesian_execution_overrides"] = overrides
        source["cartesian_execution_override_note"] = (
            "Waypoints listed here keep CuRobo selection/IK diagnostics but execute "
            "their target poses through the IsaacLab CartesianController."
        )
    return overrides


def world_to_camera_point(camera: dict, point_w: list[float] | np.ndarray) -> np.ndarray:
    rot_wc = quat_wxyz_to_matrix(np.asarray(camera["quat_w_ros"], dtype=np.float64))
    cam_pos = np.asarray(camera["pos_w"], dtype=np.float64)
    return rot_wc.T @ (np.asarray(point_w, dtype=np.float64) - cam_pos)


def project_camera_point(intrinsic: np.ndarray, point_cam: np.ndarray) -> tuple[float, float] | None:
    point = np.asarray(point_cam, dtype=np.float64)
    if point.shape != (3,) or point[2] <= 1e-6:
        return None
    u = intrinsic[0, 0] * point[0] / point[2] + intrinsic[0, 2]
    v = intrinsic[1, 1] * point[1] / point[2] + intrinsic[1, 2]
    return float(u), float(v)


def draw_camera_cross(
    draw: ImageDraw.ImageDraw,
    intrinsic: np.ndarray,
    point_cam: np.ndarray,
    color: tuple[int, int, int],
    label: str,
    radius: int = 8,
    image_size: tuple[int, int] | None = None,
    clamp_to_image: bool = False,
) -> None:
    uv = project_camera_point(intrinsic, point_cam)
    if uv is None:
        return
    x, y = uv
    if image_size is not None:
        width, height = image_size
        inside = 0 <= x < width and 0 <= y < height
        if not inside:
            if not clamp_to_image:
                return
            x = float(np.clip(x, radius + 2, max(radius + 2, width - radius - 2)))
            y = float(np.clip(y, radius + 2, max(radius + 2, height - radius - 2)))
            label = f"{label} off"
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=4)
    draw.line((x - radius - 7, y, x + radius + 7, y), fill=color, width=3)
    draw.line((x, y - radius - 7, x, y + radius + 7), fill=color, width=3)
    draw.text((x + radius + 5, y - radius), label, fill=color)


def draw_camera_segment(
    draw: ImageDraw.ImageDraw,
    intrinsic: np.ndarray,
    start_cam: np.ndarray,
    end_cam: np.ndarray,
    color: tuple[int, int, int],
    width: int = 4,
) -> None:
    start = project_camera_point(intrinsic, start_cam)
    end = project_camera_point(intrinsic, end_cam)
    if start is None or end is None:
        return
    draw.line((*start, *end), fill=(0, 0, 0), width=width + 3)
    draw.line((*start, *end), fill=color, width=width)


def save_selected_grasp_overlay(record: dict, selected_pose: dict, selection: dict) -> None:
    capture_dir = Path(record["capture_dir"])
    anygrasp_dir = Path(record["anygrasp_result_path"]).parent
    rgb_path = capture_dir / "ee_rgb.png"
    mask_path = Path(record.get("ee_sam3", {}).get("mask_path", ""))
    camera = record.get("ee_camera")
    if not rgb_path.exists() or not isinstance(camera, dict):
        return

    image = Image.open(rgb_path).convert("RGBA")
    if mask_path.exists():
        mask = load_mask(mask_path, (image.height, image.width))
        layer = np.zeros((image.height, image.width, 4), dtype=np.uint8)
        layer[mask] = (255, 204, 0, 70)
        image = Image.alpha_composite(image, Image.fromarray(layer, mode="RGBA"))

    intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float64)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 64), fill=(0, 0, 0, 190))
    selected_rank = selection.get("selected_rank")
    selected_score = selection.get("selected_score")
    draw.text(
        (8, 8),
        f"selected grasp rank={selected_rank} score={selected_score:.4f} | "
        f"IK ok={selection.get('selected_ok')} fallback={selection.get('fallback_used')}",
        fill=(255, 255, 255, 255),
    )
    selected_eval = None
    for candidate in selection.get("candidates", []):
        if int(candidate.get("rank", -1)) == int(selected_rank or -999):
            selected_eval = candidate
            break
    final_geometry = (selected_eval or {}).get("final_geometry_check") or {}
    geom_selected = final_geometry.get("selected") or {}
    width_info = geom_selected.get("width") or {}
    draw.text(
        (8, 34),
        "red/orange: generator contact | cyan: Piper gripper_base | "
        f"offset={geom_selected.get('offset_mode')}:{geom_selected.get('offset_m')} "
        f"width={width_info.get('checked_width_m')} clipped={width_info.get('width_clipped')}",
        fill=(220, 235, 255, 255),
    )

    selected_translation = np.asarray(selected_pose.get("translation", []), dtype=np.float64)
    selected_rotation = np.asarray(selected_pose.get("rotation_matrix", []), dtype=np.float64)
    selected_cam = None
    if selected_translation.shape == (3,):
        selected_cam = world_to_camera_point(camera, selected_translation)
        draw_camera_cross(draw, intrinsic, selected_cam, (255, 96, 48), f"rank {selected_rank}", radius=9)
        if selected_rotation.shape == (3, 3):
            rot_wc = quat_wxyz_to_matrix(np.asarray(camera["quat_w_ros"], dtype=np.float64))
            selected_rot_cam = rot_wc.T @ selected_rotation
            axis_len = 0.045
            draw_camera_segment(
                draw,
                intrinsic,
                selected_cam,
                selected_cam + selected_rot_cam[:, 0] * axis_len,
                (255, 70, 70),
                width=3,
            )
            draw_camera_segment(
                draw,
                intrinsic,
                selected_cam,
                selected_cam + selected_rot_cam[:, 1] * axis_len,
                (70, 230, 90),
                width=3,
            )
            draw_camera_segment(
                draw,
                intrinsic,
                selected_cam,
                selected_cam + selected_rot_cam[:, 2] * axis_len,
                (80, 150, 255),
                width=3,
            )

    if isinstance(selected_eval, dict):
        piper_pos = (
            selected_eval.get("piper_gripper_base_pose_w", {}).get("position")
        )
        if piper_pos is not None:
            piper_cam = world_to_camera_point(camera, piper_pos)
            if selected_cam is not None:
                draw_camera_segment(draw, intrinsic, selected_cam, piper_cam, (64, 220, 255), width=2)
            draw_camera_cross(
                draw,
                intrinsic,
                piper_cam,
                (64, 220, 255),
                "exec",
                radius=7,
                image_size=image.size,
                clamp_to_image=True,
            )

    output_path = anygrasp_dir / "selected_grasp_overlay.png"
    image.convert("RGB").save(output_path)
    record["selected_grasp_overlay_path"] = str(output_path)


def curobo_baseline_steps_for_waypoint(waypoint: dict, default_steps: int) -> tuple[int, dict]:
    name = str(waypoint.get("name", ""))
    stage = str(waypoint.get("curobo_stage_label") or "")
    object_name = waypoint_object_name(waypoint)
    if name.endswith("_close"):
        steps = int(args_cli.curobo_baseline_close_steps)
        source = "close"
    elif stage == "pregrasp_stage":
        steps = int(args_cli.curobo_baseline_pregrasp_stage_steps)
        source = "pregrasp_stage"
    elif stage == "pregrasp":
        steps = int(args_cli.curobo_baseline_pregrasp_steps)
        source = "pregrasp"
    elif stage == "grasp":
        steps = int(args_cli.curobo_baseline_grasp_steps)
        source = "grasp"
    elif stage == "lift":
        steps = int(args_cli.curobo_baseline_lift_steps)
        source = "lift"
    else:
        steps = int(default_steps)
        source = "request_waypoint"
    base_steps = max(1, int(steps))
    scale = 1.0
    if object_name in set(args_cli.curobo_long_baseline_objects):
        scale = max(1.0, float(args_cli.curobo_long_baseline_step_scale))
    scaled_steps = int(math.ceil(float(base_steps) * scale))
    return max(1, scaled_steps), {
        "source": source,
        "object": object_name,
        "request_steps": int(default_steps),
        "baseline_steps": int(base_steps),
        "step_scale": float(scale),
        "scaled_baseline_steps": int(max(1, scaled_steps)),
    }


def execute_waypoint_isaaclab(
    env,
    obs,
    robot,
    controller: CartesianController,
    arm_ids,
    gripper_ids,
    default_jpos,
    waypoint: dict,
    object_transport_mode: str,
    attached_object: dict,
    tracked_objects: list[dict] | None = None,
    video_recorder: VideoCameraRecorder | None = None,
) -> tuple[dict, dict]:
    pose = waypoint["pose_w"]
    position = pose["position"]
    quat = pose["quat_wxyz"]
    arm_joint_position = waypoint.get("curobo_joint_position") or waypoint.get("arm_joint_position")
    joint_target_mode = arm_joint_position is not None
    cartesian_execution_override = bool(waypoint.get("cartesian_execution_override"))
    held_current_pose = False
    if waypoint.get("hold_current_pose") and not joint_target_mode:
        ee_pose_start = robot.data.body_pose_w[0, controller.ee_idx]
        position = tensor_to_list(ee_pose_start[:3])
        quat = tensor_to_list(ee_pose_start[3:])
        pose = {"position": position, "quat_wxyz": quat}
        held_current_pose = True
    gripper = waypoint["gripper_joint_pos"]
    steps = int(waypoint.get("steps", 1))
    execution_object_name = waypoint_object_name(waypoint)
    object_transport = waypoint.get("object_transport") if object_transport_mode == "kinematic_attach" else None
    if object_transport and object_transport.get("action") == "attach":
        attached_object.clear()
        attached_object.update(
            {
                "object_key": object_transport["object_key"],
                "object_name": object_transport.get("object_name"),
                "ee_to_object_pos_w": object_transport["ee_to_object_pos_w"],
                "object_quat_wxyz": object_transport["object_quat_wxyz"],
            }
        )

    arm_joint_target_np = None
    if joint_target_mode:
        arm_joint_target_np = np.asarray(arm_joint_position, dtype=np.float64)
        if arm_joint_target_np.shape != (len(arm_ids),):
            raise ValueError(
                f"{waypoint['name']} arm joint target has shape {arm_joint_target_np.shape}, "
                f"expected {(len(arm_ids),)}"
            )
    curobo_joint_reference_np = None
    curobo_joint_reference = waypoint.get("curobo_joint_position_reference")
    if isinstance(curobo_joint_reference, list):
        curobo_joint_reference_np = np.asarray(curobo_joint_reference, dtype=np.float64)
        if curobo_joint_reference_np.shape != (len(arm_ids),):
            curobo_joint_reference_np = None
    soft_limit_report = None
    if arm_joint_target_np is not None:
        soft_limits = getattr(robot.data, "soft_joint_pos_limits", None)
        if soft_limits is not None:
            try:
                limits_np = soft_limits[0, arm_ids].detach().cpu().numpy().astype(np.float64)
                lower = limits_np[:, 0]
                upper = limits_np[:, 1]
                lower_violation = np.maximum(lower - arm_joint_target_np, 0.0)
                upper_violation = np.maximum(arm_joint_target_np - upper, 0.0)
                soft_limit_report = {
                    "lower": lower.astype(float).tolist(),
                    "upper": upper.astype(float).tolist(),
                    "max_lower_violation_rad": float(np.max(lower_violation)),
                    "max_upper_violation_rad": float(np.max(upper_violation)),
                    "within_soft_limits": bool(
                        np.max(lower_violation) <= 1e-6 and np.max(upper_violation) <= 1e-6
                    ),
                }
            except Exception as exc:
                soft_limit_report = {"error": str(exc)}

    pos_t = torch.tensor([position], dtype=torch.float32, device=env.unwrapped.device)
    quat_t = torch.tensor([quat], dtype=torch.float32, device=env.unwrapped.device)
    gripper_t = torch.tensor([gripper], dtype=torch.float32, device=env.unwrapped.device)
    arm_joint_t = (
        torch.tensor([arm_joint_target_np.astype(np.float32).tolist()], dtype=torch.float32, device=env.unwrapped.device)
        if arm_joint_target_np is not None
        else None
    )

    last_action = None
    stopped = None
    reward_sum = 0.0
    transport_writes = 0
    steps_executed = 0
    joint_path_steps = None
    joint_settle_steps = 0
    joint_close_steps = 0
    joint_reached = None
    joint_error_after_path = None
    joint_error_trace: list[dict] = []
    ee_error_after_path = None
    ee_error_trace: list[dict] = []
    max_action_abs = 0.0
    max_action_excess_over_warn = 0.0
    joint_execution_mode = str(args_cli.curobo_joint_execution_mode)
    joint_timing_report = None
    tracking_lag_exceeded_tolerance = None
    baseline_settle_report = None
    preclose_gate_report = None
    grasp_cartesian_correction_report = None
    cartesian_execution_timing_report = None
    close_arm_target_source = "curobo_joint_target"
    if waypoint.get("grasp_cartesian_correction_hold_override"):
        close_arm_target_source = "grasp_cartesian_correction_measured_joint_hold"
    close_arm_target_t = None

    def current_arm_joint_error() -> float | None:
        if arm_joint_target_np is None:
            return None
        current = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().astype(np.float64)
        return float(np.max(np.abs(current - arm_joint_target_np)))

    def current_curobo_reference_joint_error() -> float | None:
        if curobo_joint_reference_np is None:
            return None
        current = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().astype(np.float64)
        return float(np.max(np.abs(current - curobo_joint_reference_np)))

    def current_ee_pose_error() -> dict:
        ee_pose_now = robot.data.body_pose_w[0, controller.ee_idx]
        ee_pos_now = np.asarray(tensor_to_list(ee_pose_now[:3]), dtype=np.float64)
        ee_quat_now = tensor_to_list(ee_pose_now[3:])
        return {
            "position_error_m": float(np.linalg.norm(ee_pos_now - np.asarray(position, dtype=np.float64))),
            "orientation_abs_dot": float(quat_abs_dot(ee_quat_now, quat)),
            "ee_position": ee_pos_now.astype(float).tolist(),
            "ee_quat_wxyz": ee_quat_now,
        }

    def record_joint_error(label: str, step_index: int) -> None:
        error = current_arm_joint_error()
        if error is None:
            return
        if len(joint_error_trace) < 80:
            joint_error_trace.append({"label": label, "step": int(step_index), "max_error_rad": float(error)})

    def record_ee_error(label: str, step_index: int) -> None:
        if len(ee_error_trace) >= 80:
            return
        error = current_ee_pose_error()
        ee_error_trace.append(
            {
                "label": label,
                "step": int(step_index),
                "position_error_m": float(error["position_error_m"]),
                "orientation_abs_dot": float(error["orientation_abs_dot"]),
            }
        )

    def step_command(arm_des: torch.Tensor, gripper_des: torch.Tensor) -> bool:
        nonlocal obs, last_action, stopped, reward_sum, transport_writes, steps_executed
        nonlocal max_action_abs, max_action_excess_over_warn
        target = robot.data.joint_pos.clone()
        target[:, arm_ids] = arm_des
        target[:, gripper_ids] = gripper_des
        action = (target - default_jpos) / ACTION_SCALE
        last_action = action
        action_abs = float(torch.max(torch.abs(action)).detach().cpu().item())
        max_action_abs = max(max_action_abs, action_abs)
        max_action_excess_over_warn = max(
            max_action_excess_over_warn,
            max(0.0, action_abs - float(args_cli.curobo_joint_execution_action_warn)),
        )
        obs, reward, terminated, truncated, _ = env.step(action)
        steps_executed += 1
        reward_sum += float(reward[0].detach().cpu().item())
        robot.update(dt=env.unwrapped.physics_dt)
        if update_attached_object(env, robot, controller.ee_idx, attached_object):
            transport_writes += 1
        if video_recorder is not None:
            video_recorder.add(obs)
        if terminated.any() or truncated.any():
            stopped = {
                "terminated": bool(terminated.any().item()),
                "truncated": bool(truncated.any().item()),
            }
            return True
        return False

    if joint_target_mode:
        assert arm_joint_t is not None
        settle_tol = max(0.0, float(args_cli.curobo_joint_execution_settle_tol))
        max_settle_steps = max(0, int(args_cli.curobo_joint_execution_max_settle_steps))
        interp_step = max(1e-4, float(args_cli.curobo_joint_execution_interp_step))
        is_joint_close = str(waypoint.get("name", "")).endswith("_close")
        stage_label = str(waypoint.get("curobo_stage_label") or "")
        baseline_settle_enabled = bool(
            joint_execution_mode == "baseline_interp"
            and execution_object_name in set(args_cli.curobo_baseline_settle_objects)
            and stage_label in set(args_cli.curobo_baseline_settle_stages)
            and not is_joint_close
        )
        preclose_gate_enabled = bool(
            joint_execution_mode == "baseline_interp"
            and is_joint_close
            and execution_object_name in set(args_cli.curobo_preclose_gate_objects)
        )
        start_arm_np = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().astype(np.float64)
        start_gripper_np = robot.data.joint_pos[0, gripper_ids].detach().cpu().numpy().astype(np.float64)
        max_delta = float(np.max(np.abs(arm_joint_target_np - start_arm_np)))
        if joint_execution_mode == "baseline_interp":
            minimum_path_steps, joint_timing_report = curobo_baseline_steps_for_waypoint(waypoint, int(steps))
        else:
            minimum_path_steps = 1 if is_joint_close else int(steps)
            joint_timing_report = {
                "source": "request_waypoint",
                "request_steps": int(steps),
                "baseline_steps": None,
            }
        joint_path_steps = max(1, minimum_path_steps, int(math.ceil(max_delta / interp_step)))
        start_arm_t = torch.tensor([start_arm_np.astype(np.float32).tolist()], dtype=torch.float32, device=env.unwrapped.device)
        start_gripper_t = torch.tensor(
            [start_gripper_np.astype(np.float32).tolist()],
            dtype=torch.float32,
            device=env.unwrapped.device,
        )
        target_arm_t = arm_joint_t
        close_arm_target_t = target_arm_t
        motion_gripper_t = gripper_t
        if is_joint_close and (joint_execution_mode != "baseline_interp" or preclose_gate_enabled):
            motion_gripper_t = torch.tensor([GRIPPER_OPEN], dtype=torch.float32, device=env.unwrapped.device)
        record_joint_error("start", 0)
        record_ee_error("start", 0)
        for step_idx in range(joint_path_steps):
            frac = float(step_idx + 1) / float(joint_path_steps)
            arm_des = start_arm_t + (target_arm_t - start_arm_t) * frac
            if is_joint_close and joint_execution_mode == "baseline_interp" and not preclose_gate_enabled:
                gripper_des = start_gripper_t + (gripper_t - start_gripper_t) * frac
            else:
                gripper_des = motion_gripper_t
            if step_command(arm_des, gripper_des):
                break
            if step_idx in {0, joint_path_steps - 1} or (step_idx + 1) % 25 == 0:
                record_joint_error("path", step_idx + 1)
                record_ee_error("path", step_idx + 1)
        joint_error_after_path = current_arm_joint_error()
        ee_error_after_path = current_ee_pose_error()
        if stopped is None and joint_execution_mode != "baseline_interp":
            while (
                current_arm_joint_error() is not None
                and float(current_arm_joint_error()) > settle_tol
                and joint_settle_steps < max_settle_steps
            ):
                if step_command(target_arm_t, motion_gripper_t):
                    break
                joint_settle_steps += 1
                if joint_settle_steps in {1, max_settle_steps} or joint_settle_steps % 25 == 0:
                    record_joint_error("settle", joint_settle_steps)
                    record_ee_error("settle", joint_settle_steps)
        if stopped is None and baseline_settle_enabled:
            settle_ee_tol = float(args_cli.curobo_baseline_settle_ee_tol)
            baseline_start = {
                "joint_error_rad": current_arm_joint_error(),
                "ee": current_ee_pose_error(),
            }
            while joint_settle_steps < max_settle_steps:
                joint_error = current_arm_joint_error()
                ee_error = current_ee_pose_error()
                joint_ok = joint_error is None or float(joint_error) <= settle_tol
                ee_ok = settle_ee_tol <= 0.0 or float(ee_error["position_error_m"]) <= settle_ee_tol
                if joint_ok and ee_ok:
                    break
                if step_command(target_arm_t, motion_gripper_t):
                    break
                joint_settle_steps += 1
                if joint_settle_steps in {1, max_settle_steps} or joint_settle_steps % 25 == 0:
                    record_joint_error("baseline_settle", joint_settle_steps)
                    record_ee_error("baseline_settle", joint_settle_steps)
            baseline_final_joint_error = current_arm_joint_error()
            baseline_final_ee_error = current_ee_pose_error()
            baseline_settle_report = {
                "enabled": True,
                "object": execution_object_name,
                "stage": stage_label,
                "joint_tolerance_rad": float(settle_tol),
                "ee_position_tolerance_m": float(settle_ee_tol),
                "max_settle_steps": int(max_settle_steps),
                "settle_steps": int(joint_settle_steps),
                "start_joint_error_rad": baseline_start["joint_error_rad"],
                "start_ee_error": baseline_start["ee"],
                "final_joint_error_rad": baseline_final_joint_error,
                "final_ee_error": baseline_final_ee_error,
                "accepted": bool(
                    (baseline_final_joint_error is None or float(baseline_final_joint_error) <= settle_tol)
                    and (
                        settle_ee_tol <= 0.0
                        or float(baseline_final_ee_error["position_error_m"]) <= settle_ee_tol
                    )
                ),
            }
        elif joint_target_mode:
            baseline_settle_report = {
                "enabled": False,
                "object": execution_object_name,
                "stage": stage_label,
            }
        grasp_correction_enabled = bool(
            stopped is None
            and joint_execution_mode == "baseline_interp"
            and stage_label == "grasp"
            and execution_object_name in set(args_cli.curobo_grasp_cartesian_correction_objects)
            and int(args_cli.curobo_grasp_cartesian_correction_steps) > 0
        )
        if grasp_correction_enabled:
            correction_steps_max = max(0, int(args_cli.curobo_grasp_cartesian_correction_steps))
            trigger_ee_tol = max(0.0, float(args_cli.curobo_grasp_cartesian_correction_trigger_ee_tol))
            correction_ee_tol = max(0.0, float(args_cli.curobo_grasp_cartesian_correction_ee_tol))
            correction_orientation_dot = float(args_cli.curobo_grasp_cartesian_correction_orientation_dot)
            correction_start = {
                "joint_error_rad": current_arm_joint_error(),
                "ee": current_ee_pose_error(),
            }
            should_correct = bool(
                float(correction_start["ee"]["position_error_m"]) > trigger_ee_tol
                or float(correction_start["ee"]["orientation_abs_dot"]) < correction_orientation_dot
            )
            correction_steps = 0
            correction_stop = None
            correction_trace = []
            if should_correct:
                controller.reset()
                while correction_steps < correction_steps_max:
                    ee_error = current_ee_pose_error()
                    correction_ok = bool(
                        float(ee_error["position_error_m"]) <= correction_ee_tol
                        and float(ee_error["orientation_abs_dot"]) >= correction_orientation_dot
                    )
                    if correction_ok:
                        break
                    arm_des = controller.compute(pos_t, quat_t)
                    if step_command(arm_des, motion_gripper_t):
                        correction_stop = stopped
                        break
                    correction_steps += 1
                    joint_settle_steps += 1
                    if correction_steps in {1, correction_steps_max} or correction_steps % 25 == 0:
                        record_joint_error("grasp_cartesian_correction", correction_steps)
                        record_ee_error("grasp_cartesian_correction", correction_steps)
                        trace_ee = current_ee_pose_error()
                        if len(correction_trace) < 60:
                            correction_trace.append(
                                {
                                    "step": int(correction_steps),
                                    "joint_error_rad": current_arm_joint_error(),
                                    "position_error_m": float(trace_ee["position_error_m"]),
                                    "orientation_abs_dot": float(trace_ee["orientation_abs_dot"]),
                                }
                            )
            correction_final_joint_error = current_arm_joint_error()
            correction_final_ee_error = current_ee_pose_error()
            correction_accepted = bool(
                stopped is None
                and float(correction_final_ee_error["position_error_m"]) <= correction_ee_tol
                and float(correction_final_ee_error["orientation_abs_dot"]) >= correction_orientation_dot
            )
            corrected_arm = robot.data.joint_pos[0, arm_ids].detach().clone().view(1, -1)
            grasp_cartesian_correction_report = {
                "enabled": True,
                "object": execution_object_name,
                "triggered": bool(should_correct),
                "steps": int(correction_steps),
                "max_steps": int(correction_steps_max),
                "trigger_ee_position_tolerance_m": float(trigger_ee_tol),
                "ee_position_tolerance_m": float(correction_ee_tol),
                "orientation_abs_dot_min": float(correction_orientation_dot),
                "start_joint_error_rad": correction_start["joint_error_rad"],
                "start_ee_error": correction_start["ee"],
                "final_joint_error_rad": correction_final_joint_error,
                "final_ee_error": correction_final_ee_error,
                "accepted": bool(correction_accepted),
                "corrected_arm_joint_pos": tensor_to_list(corrected_arm[0]),
                "trace": correction_trace,
                "stop_reason": correction_stop,
            }
            if (
                not correction_accepted
                and stopped is None
                and bool(args_cli.curobo_grasp_cartesian_correction_fail_on_reject)
            ):
                stopped = {
                    "grasp_cartesian_correction_failed": True,
                    "object": execution_object_name,
                    "ee_position_error_m": float(correction_final_ee_error["position_error_m"]),
                    "ee_position_tolerance_m": float(correction_ee_tol),
                    "orientation_abs_dot": float(correction_final_ee_error["orientation_abs_dot"]),
                    "orientation_abs_dot_min": float(correction_orientation_dot),
                    "steps": int(correction_steps),
                    "reason": "close_skipped_because_grasp_cartesian_correction_failed",
                }
        elif joint_target_mode:
            grasp_cartesian_correction_report = {
                "enabled": False,
                "object": execution_object_name,
                "stage": stage_label,
            }
        if stopped is None and preclose_gate_enabled:
            gate_joint_tol = max(0.0, float(args_cli.curobo_preclose_gate_joint_tol))
            gate_ee_tol = max(0.0, float(args_cli.curobo_preclose_gate_ee_tol))
            gate_max_settle_steps = max(0, int(args_cli.curobo_preclose_gate_max_settle_steps))
            gate_start = {
                "joint_error_rad": current_arm_joint_error(),
                "ee": current_ee_pose_error(),
            }
            gate_steps = 0
            while gate_steps < gate_max_settle_steps:
                joint_error = current_arm_joint_error()
                ee_error = current_ee_pose_error()
                joint_ok = joint_error is None or float(joint_error) <= gate_joint_tol
                ee_ok = float(ee_error["position_error_m"]) <= gate_ee_tol
                if joint_ok and ee_ok:
                    break
                if step_command(target_arm_t, motion_gripper_t):
                    break
                gate_steps += 1
                joint_settle_steps += 1
                if gate_steps in {1, gate_max_settle_steps} or gate_steps % 25 == 0:
                    record_joint_error("preclose_gate", gate_steps)
                    record_ee_error("preclose_gate", gate_steps)
            gate_final_joint_error = current_arm_joint_error()
            gate_final_ee_error = current_ee_pose_error()
            gate_joint_ok = gate_final_joint_error is None or float(gate_final_joint_error) <= gate_joint_tol
            gate_ee_ok = float(gate_final_ee_error["position_error_m"]) <= gate_ee_tol
            native_gate_ok = bool(gate_joint_ok and gate_ee_ok)
            gate_ok = native_gate_ok
            correction_report = {
                "enabled": False,
                "reason": "native_preclose_gate_accepted" if native_gate_ok else "disabled",
            }
            correction_objects = set(args_cli.curobo_preclose_cartesian_correction_objects)
            correction_enabled = bool(
                (not native_gate_ok)
                and execution_object_name in correction_objects
                and int(args_cli.curobo_preclose_cartesian_correction_steps) > 0
            )
            if correction_enabled and stopped is None:
                correction_steps_max = max(0, int(args_cli.curobo_preclose_cartesian_correction_steps))
                correction_ee_tol = max(0.0, float(args_cli.curobo_preclose_cartesian_correction_ee_tol))
                correction_orientation_dot = float(args_cli.curobo_preclose_cartesian_correction_orientation_dot)
                correction_start = {
                    "joint_error_rad": current_arm_joint_error(),
                    "ee": current_ee_pose_error(),
                }
                correction_trace = []
                controller.reset()
                correction_steps = 0
                correction_stop = None
                while correction_steps < correction_steps_max:
                    ee_error = current_ee_pose_error()
                    correction_ok = bool(
                        float(ee_error["position_error_m"]) <= correction_ee_tol
                        and float(ee_error["orientation_abs_dot"]) >= correction_orientation_dot
                    )
                    if correction_ok:
                        break
                    arm_des = controller.compute(pos_t, quat_t)
                    if step_command(arm_des, motion_gripper_t):
                        correction_stop = stopped
                        break
                    correction_steps += 1
                    joint_settle_steps += 1
                    if correction_steps in {1, correction_steps_max} or correction_steps % 25 == 0:
                        record_joint_error("preclose_cartesian_correction", correction_steps)
                        record_ee_error("preclose_cartesian_correction", correction_steps)
                        trace_ee = current_ee_pose_error()
                        if len(correction_trace) < 40:
                            correction_trace.append(
                                {
                                    "step": int(correction_steps),
                                    "joint_error_rad": current_arm_joint_error(),
                                    "position_error_m": float(trace_ee["position_error_m"]),
                                    "orientation_abs_dot": float(trace_ee["orientation_abs_dot"]),
                                }
                            )
                correction_final_joint_error = current_arm_joint_error()
                correction_final_ee_error = current_ee_pose_error()
                correction_accepted = bool(
                    stopped is None
                    and float(correction_final_ee_error["position_error_m"]) <= correction_ee_tol
                    and float(correction_final_ee_error["orientation_abs_dot"]) >= correction_orientation_dot
                )
                corrected_close_arm = robot.data.joint_pos[0, arm_ids].detach().clone().view(1, -1)
                if correction_accepted:
                    close_arm_target_t = corrected_close_arm
                    close_arm_target_source = "preclose_cartesian_correction_measured_joint_hold"
                    gate_ok = True
                    gate_ee_ok = True
                correction_report = {
                    "enabled": True,
                    "object": execution_object_name,
                    "steps": int(correction_steps),
                    "max_steps": int(correction_steps_max),
                    "ee_position_tolerance_m": float(correction_ee_tol),
                    "orientation_abs_dot_min": float(correction_orientation_dot),
                    "start_joint_error_rad": correction_start["joint_error_rad"],
                    "start_ee_error": correction_start["ee"],
                    "final_joint_error_rad": correction_final_joint_error,
                    "final_ee_error": correction_final_ee_error,
                    "accepted": bool(correction_accepted),
                    "close_arm_target_source": close_arm_target_source if correction_accepted else None,
                    "close_arm_joint_pos": tensor_to_list(corrected_close_arm[0]),
                    "trace": correction_trace,
                    "stop_reason": correction_stop,
                }
            preclose_gate_report = {
                "enabled": True,
                "object": execution_object_name,
                "joint_tolerance_rad": float(gate_joint_tol),
                "ee_position_tolerance_m": float(gate_ee_tol),
                "max_settle_steps": int(gate_max_settle_steps),
                "settle_steps": int(gate_steps),
                "start_joint_error_rad": gate_start["joint_error_rad"],
                "start_ee_error": gate_start["ee"],
                "final_joint_error_rad": gate_final_joint_error,
                "final_ee_error": gate_final_ee_error,
                "joint_ok": bool(gate_joint_ok),
                "ee_ok": bool(gate_ee_ok),
                "native_gate_accepted": bool(native_gate_ok),
                "accepted": gate_ok,
                "cartesian_correction": correction_report,
            }
            if not gate_ok and stopped is None:
                stop_ee_error = gate_final_ee_error
                if correction_report.get("enabled") and isinstance(correction_report.get("final_ee_error"), dict):
                    stop_ee_error = correction_report["final_ee_error"]
                stop_joint_error = gate_final_joint_error
                if correction_report.get("enabled"):
                    stop_joint_error = correction_report.get("final_joint_error_rad", gate_final_joint_error)
                stopped = {
                    "preclose_tracking_gate_failed": True,
                    "object": execution_object_name,
                    "joint_error_rad": None if stop_joint_error is None else float(stop_joint_error),
                    "joint_tolerance_rad": float(gate_joint_tol),
                    "ee_position_error_m": float(stop_ee_error["position_error_m"]),
                    "ee_position_tolerance_m": float(gate_ee_tol),
                    "settle_steps": int(gate_steps),
                    "reason": "close_skipped_because_grasp_target_tracking_was_not_reached",
                }
        elif is_joint_close:
            preclose_gate_report = {
                "enabled": False,
                "object": execution_object_name,
            }
        final_joint_error = current_arm_joint_error()
        joint_reached = final_joint_error is not None and final_joint_error <= settle_tol
        tracking_lag_exceeded_tolerance = bool(final_joint_error is not None and final_joint_error > settle_tol)
        if stopped is None and is_joint_close and (joint_reached or joint_execution_mode == "baseline_interp"):
            close_hold_steps = (
                int(args_cli.curobo_baseline_close_hold_steps)
                if joint_execution_mode == "baseline_interp"
                else int(steps)
            )
            for close_step in range(max(1, close_hold_steps)):
                if step_command(close_arm_target_t if close_arm_target_t is not None else target_arm_t, gripper_t):
                    break
                joint_close_steps += 1
                if close_step in {0, max(1, close_hold_steps) - 1} or (close_step + 1) % 25 == 0:
                    record_joint_error("close", close_step + 1)
                    record_ee_error("close", close_step + 1)
            final_joint_error = current_arm_joint_error()
            joint_reached = final_joint_error is not None and final_joint_error <= settle_tol
            tracking_lag_exceeded_tolerance = bool(final_joint_error is not None and final_joint_error > settle_tol)
        if (
            stopped is None
            and not joint_reached
            and bool(args_cli.curobo_joint_execution_abort_on_fail)
            and joint_execution_mode != "baseline_interp"
        ):
            stopped = {
                "joint_settle_failed": True,
                "max_joint_error_rad": None if final_joint_error is None else float(final_joint_error),
                "settle_tolerance_rad": float(settle_tol),
                "path_steps": int(joint_path_steps),
                "settle_steps": int(joint_settle_steps),
            }
    else:
        cartesian_steps = max(1, int(steps))
        if cartesian_execution_override:
            baseline_steps, cartesian_execution_timing_report = curobo_baseline_steps_for_waypoint(
                waypoint,
                cartesian_steps,
            )
            cartesian_steps = max(cartesian_steps, int(baseline_steps))
        record_ee_error("cartesian_start", 0)
        for step_idx in range(cartesian_steps):
            arm_des = controller.compute(pos_t, quat_t)
            if step_command(arm_des, gripper_t):
                break
            if step_idx in {0, cartesian_steps - 1} or (step_idx + 1) % 25 == 0:
                record_ee_error("cartesian_path", step_idx + 1)

    if object_transport and object_transport.get("action") == "release":
        release_center_w = object_transport.get("release_center_w")
        if release_center_w is not None:
            write_object_pose(
                env,
                object_transport["object_key"],
                release_center_w,
                object_transport["object_quat_wxyz"],
            )
            transport_writes += 1
        attached_object.clear()
    if transport_writes:
        obs = refresh_observation(env, obs)

    ee_pose = robot.data.body_pose_w[0, controller.ee_idx]
    ee_pos = tensor_to_list(ee_pose[:3])
    ee_quat = tensor_to_list(ee_pose[3:])
    arm_joint_pos = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().astype(np.float64)
    arm_joint_error_max = (
        float(np.max(np.abs(arm_joint_pos - arm_joint_target_np)))
        if arm_joint_target_np is not None
        else None
    )
    curobo_joint_reference_error_max = current_curobo_reference_joint_error()
    gripper_joint_pos = robot.data.joint_pos[0, gripper_ids].detach().cpu().numpy().astype(np.float64)
    gripper_aperture = max(0.0, float(gripper_joint_pos[0] - gripper_joint_pos[1])) if len(gripper_joint_pos) >= 2 else None
    close_target = np.asarray(GRIPPER_CLOSE, dtype=np.float64)
    close_target_error = (
        float(np.max(np.abs(gripper_joint_pos[:2] - close_target[:2])))
        if len(gripper_joint_pos) >= 2
        else None
    )
    pos_error = float(np.linalg.norm(np.asarray(ee_pos) - np.asarray(position)))
    tracked = tracked_object_states(env, tracked_objects or [], ee_pos)
    return obs, {
        "name": waypoint["name"],
        "ok": stopped is None or bool(stopped.get("terminated", False)),
        "execution_mode": (
            f"curobo_joint_position_{joint_execution_mode}"
            if joint_target_mode
            else (
                "cartesian_pose_controller_from_curobo_selected_pose"
                if cartesian_execution_override
                else "cartesian_pose_controller"
            )
        ),
        "target_pose_w": pose,
        "held_current_pose": held_current_pose,
        "curobo_stage_label": waypoint.get("curobo_stage_label"),
        "execution_object_name": execution_object_name,
        "cartesian_execution_override": bool(cartesian_execution_override),
        "cartesian_execution_stage": waypoint.get("cartesian_execution_stage"),
        "cartesian_execution_note": waypoint.get("cartesian_execution_note"),
        "cartesian_execution_timing": cartesian_execution_timing_report,
        "curobo_ik_position_error_m": waypoint.get("curobo_ik_position_error_m"),
        "curobo_ik_rotation_error_rad": waypoint.get("curobo_ik_rotation_error_rad"),
        "curobo_target_pose_w": waypoint.get("curobo_target_pose_w"),
        "curobo_joint_execution_mode": joint_execution_mode if joint_target_mode else None,
        "curobo_joint_timing": joint_timing_report,
        "target_arm_joint_pos": arm_joint_target_np.astype(float).tolist()
        if arm_joint_target_np is not None
        else None,
        "curobo_joint_position_reference": (
            curobo_joint_reference_np.astype(float).tolist()
            if curobo_joint_reference_np is not None
            else None
        ),
        "curobo_joint_reference_error_max": curobo_joint_reference_error_max,
        "actual_arm_joint_pos": arm_joint_pos.astype(float).tolist(),
        "arm_joint_error_max": arm_joint_error_max,
        "arm_joint_reached": joint_reached,
        "arm_joint_tracking_lag_exceeded_tolerance": tracking_lag_exceeded_tolerance,
        "arm_joint_error_after_path": joint_error_after_path,
        "ee_error_after_path": ee_error_after_path,
        "arm_joint_settle_tolerance_rad": float(args_cli.curobo_joint_execution_settle_tol)
        if joint_target_mode
        else None,
        "arm_joint_path_steps": joint_path_steps,
        "arm_joint_settle_steps": int(joint_settle_steps),
        "arm_joint_close_steps": int(joint_close_steps),
        "steps_executed": int(steps_executed),
        "joint_error_trace": joint_error_trace,
        "ee_error_trace": ee_error_trace,
        "baseline_settle": baseline_settle_report,
        "grasp_cartesian_correction": grasp_cartesian_correction_report,
        "preclose_tracking_gate": preclose_gate_report,
        "soft_joint_limit_check": soft_limit_report,
        "max_action_abs": float(max_action_abs),
        "max_action_excess_over_warn": float(max_action_excess_over_warn),
        "action_warn_threshold": float(args_cli.curobo_joint_execution_action_warn),
        "target_gripper_joint_pos": gripper,
        "actual_gripper_joint_pos": gripper_joint_pos.astype(float).tolist(),
        "actual_gripper_aperture_m": gripper_aperture,
        "closed_target_joint_error_max": close_target_error,
        "close_arm_target_source": close_arm_target_source if joint_target_mode else None,
        "close_arm_target_pos": (
            tensor_to_list(close_arm_target_t[0])
            if joint_target_mode and close_arm_target_t is not None
            else None
        ),
        "steps_requested": steps,
        "reward_sum": reward_sum,
        "last_action": tensor_to_list(last_action[0]) if last_action is not None else None,
        "ee_pose_w": {"position": ee_pos, "quat_wxyz": ee_quat},
        "position_error_m": pos_error,
        "orientation_abs_dot": quat_abs_dot(ee_quat, quat),
        "target_tracking_error": {
            "target_source": (
                "curobo_selected_pose_executed_by_cartesian_controller"
                if cartesian_execution_override
                else (
                    "curobo_joint_position_execution"
                    if joint_target_mode
                    else "cartesian_pose_controller"
                )
            ),
            "target_pose_w": pose,
            "actual_pose_w": {"position": ee_pos, "quat_wxyz": ee_quat},
            "position_error_m": pos_error,
            "orientation_abs_dot": quat_abs_dot(ee_quat, quat),
            "curobo_reference_joint_error_max": curobo_joint_reference_error_max,
        },
        "stop_reason": stopped,
        "object_transport": object_transport or None,
        "object_transport_writes": transport_writes,
        "tracked_objects": tracked,
    }


def execute_motion_request_in_env(
    env,
    obs,
    robot,
    request: dict,
    request_path: Path,
    output_dir: Path,
    video_recorder: VideoCameraRecorder | None = None,
) -> tuple[dict, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "motion_request.json").write_text(json_text(request), encoding="utf-8")

    arm_ids, _ = robot.find_joints(request.get("robot", {}).get("arm_joints", ARM_JOINT_NAMES))
    gripper_ids, _ = robot.find_joints(request.get("robot", {}).get("gripper_joints", GRIPPER_JOINT_NAMES))
    controller = CartesianController(
        robot=robot,
        ee_body_name=request.get("robot", {}).get("ee_link", "gripper_base"),
        arm_joint_names=request.get("robot", {}).get("arm_joints", ARM_JOINT_NAMES),
        num_envs=1,
        device=env.unwrapped.device,
        command_type="pose",
        lambda_val=0.05,
        max_joint_delta=0.16,
    )
    controller.reset()
    default_jpos = robot.data.default_joint_pos.clone()
    object_transport_mode = request.get("controller", {}).get("object_transport_mode", "physics")
    tracked_objects = request.get("source", {}).get("objects") or []

    artifacts = {"frames": {}}
    if not args_cli.no_save_execution_frames:
        save_execution_frame(output_dir / "00_initial.png", obs, "initial")
        artifacts["frames"]["00_initial"] = "00_initial.png"

    waypoint_results = []
    total_reward = 0.0
    attached_object: dict = {}
    early_abort = None
    grasp_servo_hold_arm_by_object: dict[str, list[float]] = {}
    for waypoint in request["waypoints"]:
        waypoint_object = waypoint_object_name(waypoint)
        if str(waypoint.get("name", "")).endswith("_close") and waypoint_object in grasp_servo_hold_arm_by_object:
            waypoint = copy.deepcopy(waypoint)
            waypoint["curobo_joint_position"] = grasp_servo_hold_arm_by_object[waypoint_object]
            waypoint["grasp_cartesian_correction_hold_override"] = True
        obs, waypoint_result = execute_waypoint_isaaclab(
            env,
            obs,
            robot,
            controller,
            arm_ids,
            gripper_ids,
            default_jpos,
            waypoint,
            object_transport_mode,
            attached_object,
            tracked_objects,
            video_recorder,
        )
        waypoint_results.append(waypoint_result)
        total_reward += float(waypoint_result["reward_sum"])
        if not args_cli.no_save_execution_frames and waypoint.get("capture", True):
            frame_name = f"{waypoint['name']}.png"
            save_execution_frame(output_dir / frame_name, obs, waypoint["name"])
            artifacts["frames"][waypoint["name"]] = frame_name
        joint_error = waypoint_result.get("arm_joint_error_max")
        joint_error_text = f" joint_err={joint_error:.4f}" if joint_error is not None else ""
        curobo_ref_error = waypoint_result.get("curobo_joint_reference_error_max")
        curobo_ref_text = (
            f" curobo_ref_joint_err={float(curobo_ref_error):.4f}"
            if curobo_ref_error is not None
            else ""
        )
        reached = waypoint_result.get("arm_joint_reached")
        reached_text = f" reached={reached}" if reached is not None else ""
        settle_steps = waypoint_result.get("arm_joint_settle_steps")
        settle_text = f" settle={settle_steps}" if settle_steps else ""
        grasp_correction = waypoint_result.get("grasp_cartesian_correction")
        grasp_corr_text = ""
        if isinstance(grasp_correction, dict) and grasp_correction.get("enabled"):
            final_ee = grasp_correction.get("final_ee_error") or {}
            final_pos = final_ee.get("position_error_m")
            final_pos_text = "unknown" if final_pos is None else f"{float(final_pos):.4f}"
            grasp_corr_text = (
                f" grasp_corr={grasp_correction.get('accepted')} "
                f"corr_err={final_pos_text}"
            )
        print(
            f"[INFO] {waypoint['name']}: mode={waypoint_result.get('execution_mode')} "
            f"error={waypoint_result['position_error_m']:.4f}{joint_error_text}{curobo_ref_text}"
            f"{reached_text}{settle_text}"
            f"{grasp_corr_text} "
            f"ee={np.round(waypoint_result['ee_pose_w']['position'], 4).tolist()}",
            flush=True,
        )
        if (
            bool(args_cli.early_close_failure_check)
            and str(waypoint.get("name", "")).endswith("_close")
            and waypoint_result.get("actual_gripper_aperture_m") is not None
        ):
            aperture = float(waypoint_result["actual_gripper_aperture_m"])
            close_error = waypoint_result.get("closed_target_joint_error_max")
            fully_closed = aperture <= float(args_cli.early_close_failure_aperture_threshold)
            target_reached = (
                close_error is None
                or float(close_error) <= float(args_cli.early_close_failure_target_tol)
            )
            close_check = {
                "enabled": True,
                "failed": bool(fully_closed and target_reached),
                "actual_gripper_aperture_m": float(aperture),
                "aperture_threshold_m": float(args_cli.early_close_failure_aperture_threshold),
                "closed_target_joint_error_max": None if close_error is None else float(close_error),
                "closed_target_tolerance": float(args_cli.early_close_failure_target_tol),
                "actual_gripper_joint_pos": waypoint_result.get("actual_gripper_joint_pos"),
                "closed_target_joint_pos": [float(v) for v in GRIPPER_CLOSE],
            }
            waypoint_result["early_close_failure_check"] = close_check
            if close_check["failed"]:
                waypoint_result["ok"] = False
                early_abort = {
                    "reason": "gripper_fully_closed_after_close_waypoint",
                    "waypoint": waypoint.get("name"),
                    "check": close_check,
                }
                print(
                    "[WARN] "
                    f"{waypoint['name']}: gripper fully closed "
                    f"(aperture={aperture:.4f} m); aborting remaining execution and replanning.",
                    flush=True,
                )
                break
        correction = waypoint_result.get("grasp_cartesian_correction")
        if (
            isinstance(correction, dict)
            and correction.get("enabled")
            and correction.get("accepted")
            and correction.get("corrected_arm_joint_pos") is not None
            and waypoint_object is not None
            and str(waypoint.get("curobo_stage_label") or "") == "grasp"
        ):
            grasp_servo_hold_arm_by_object[waypoint_object] = [
                float(value) for value in correction["corrected_arm_joint_pos"]
            ]
        if str(waypoint.get("name", "")).endswith("_close") and waypoint_object in grasp_servo_hold_arm_by_object:
            grasp_servo_hold_arm_by_object.pop(waypoint_object, None)
        if waypoint_result["stop_reason"] is not None:
            break

    used_curobo_joint_execution = any(
        str(item.get("execution_mode", "")).startswith("curobo_joint_position") for item in waypoint_results
    )
    result = {
        "schema": RESULT_SCHEMA,
        "ok": bool(early_abort is None and all(item["ok"] for item in waypoint_results)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "request_path": str(request_path),
        "task": request.get("task"),
        "seed": args_cli.seed,
        "backend": {
            "used": (
                "curobo_joint_position_targets_plus_isaaclab_controller"
                if used_curobo_joint_execution
                else "isaaclab_cartesian_controller"
            ),
            "reason": (
                "selected CuRobo IK joint solutions executed for annotated waypoints"
                if used_curobo_joint_execution
                else "interleaved pipeline execution"
            ),
        },
        "controller": {
            "actuator_mode": args_cli.actuator_mode,
            "object_transport_mode": object_transport_mode,
        },
        "source": request.get("source"),
        "total_reward": total_reward,
        "waypoints": waypoint_results,
        "early_abort": early_abort,
        "task_e_objects": collect_task_e_object_summary(env),
        "artifacts": artifacts,
    }
    for source_object in tracked_objects:
        name = source_object.get("name")
        if name in OBJECTS:
            result.setdefault("target_diagnostics", {})[name] = motion_target_diagnostics(result, name)
    result_path = output_dir / "motion_result.json"
    result_path.write_text(json_text(result), encoding="utf-8")
    return obs, result


def select_curobo_feasible_grasps(env, obs, robot, object_poses: dict[str, dict], records: dict[str, dict]) -> dict:
    submission_dir = REPO_ROOT / "submissions/task_e_act_baseline_root_submission"
    if str(submission_dir) not in sys.path:
        sys.path.insert(0, str(submission_dir))
    from task_e_curobo_planner import TaskECuRoboPlannerProcess

    arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
    planner = TaskECuRoboPlannerProcess(
        num_seeds=int(args_cli.curobo_num_seeds),
        position_tolerance_m=float(args_cli.curobo_position_tol),
        orientation_tolerance_rad=float(args_cli.curobo_rotation_tol),
        request_timeout_s=float(args_cli.curobo_request_timeout),
        scene_collision_check=bool(args_cli.curobo_scene_collision),
        self_collision_check=False,
        collision_cache={"cuboid": max(1, int(args_cli.curobo_obstacle_max_cuboids))},
    )
    try:
        global_accept_first_ik_success = args_cli.grasp_selection == "curobo_first_ik"
        first_ik_objects = set(args_cli.curobo_first_ik_objects)
        for name in args_cli.object_order:
            if name not in records:
                continue
            accept_first_ik_success = bool(global_accept_first_ik_success or name in first_ik_objects)
            record = records[name]
            candidates = anygrasp_candidates(record)
            if not candidates:
                continue

            top_k = len(candidates) if args_cli.ik_filter_top_k <= 0 else min(args_cli.ik_filter_top_k, len(candidates))
            target_points, obstacle_points, cloud_meta = load_piper_validation_clouds(record)
            obstacle_cuboids, obstacle_world_meta = curobo_obstacle_cuboids_from_cloud(
                robot,
                target_points,
                obstacle_points,
            )
            cloud_meta["curobo_obstacle_world"] = obstacle_world_meta
            if bool(args_cli.curobo_scene_collision):
                planner.update_world_cuboids(obstacle_cuboids)
            current_q = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().astype(np.float32)
            tested = []
            selected_pose = candidates[0]
            selected_eval = None
            best_fallback_pose = candidates[0]
            best_fallback_eval = None
            best_fallback_key = None
            staged_final_survivors: list[tuple[tuple, dict, dict, dict, dict, dict]] = []
            staged_final_screen_count = 0
            staged_final_screen_limit = max(1, int(args_cli.curobo_staged_final_ik_top_k))
            staged_full_sequence_limit = max(1, int(args_cli.curobo_staged_full_ik_top_k))

            for candidate in candidates[:top_k]:
                rank = int(candidate.get("source_rank", 0))
                geometry_pose, geometry_report = final_piper_geometry_search(
                    target_points,
                    obstacle_points,
                    candidate,
                    object_poses[name],
                    name,
                )
                if not geometry_report["safe"]:
                    evaluation = {
                        "rank": rank,
                        "score": float(candidate.get("score", 0.0)),
                        "ok": False,
                        "skipped_curobo": True,
                        "skip_reason": "final_piper_geometry",
                        "final_geometry_check": geometry_report,
                        "candidate": {
                            "translation": candidate.get("translation"),
                            "rotation_matrix": candidate.get("rotation_matrix"),
                            "width": float(candidate.get("width", 0.0)),
                            "depth": float(candidate.get("depth", 0.0)),
                        },
                    }
                    tested.append(evaluation)
                    selected = geometry_report["selected"]
                    fallback_key = (
                        2,
                        int(selected["target"]["solid_point_count"]) + int(selected["obstacle"]["solid_point_count"]),
                        -int(selected["target"]["centerline_point_count"]),
                        -int(selected["target"]["middle_support_point_count"]),
                        -int(selected["target"]["tcp_support_point_count"]),
                        -float(candidate.get("score", 0.0)),
                    )
                    if best_fallback_key is None or fallback_key < best_fallback_key:
                        best_fallback_key = fallback_key
                        best_fallback_pose = geometry_pose
                        best_fallback_eval = evaluation
                    print(
                        "[INFO] "
                        f"{name} cuRobo candidate rank={rank} rejected final geometry "
                        f"target_solid={selected['target']['solid_point_count']} "
                        f"obstacle_solid={selected['obstacle']['solid_point_count']} "
                        f"centerline={selected['target']['centerline_point_count']} "
                        f"middle={selected['target']['middle_support_point_count']} "
                        f"tcp={selected['target']['tcp_support_point_count']}",
                        flush=True,
                    )
                    continue

                approach_report = curobo_approach_collision_check(
                    target_points,
                    obstacle_points,
                    name,
                    geometry_pose,
                    object_poses[name],
                )
                if not approach_report["safe"]:
                    evaluation = {
                        "rank": rank,
                        "score": float(candidate.get("score", 0.0)),
                        "ok": False,
                        "skipped_curobo": True,
                        "skip_reason": "approach_piper_geometry",
                        "final_geometry_check": geometry_report,
                        "approach_collision_check": approach_report,
                        "candidate": {
                            "translation": candidate.get("translation"),
                            "rotation_matrix": candidate.get("rotation_matrix"),
                            "width": float(candidate.get("width", 0.0)),
                            "depth": float(candidate.get("depth", 0.0)),
                        },
                    }
                    tested.append(evaluation)
                    first = approach_report.get("first_illegal") or {}
                    fallback_key = (1, -float(candidate.get("score", 0.0)))
                    if best_fallback_key is None or fallback_key < best_fallback_key:
                        best_fallback_key = fallback_key
                        best_fallback_pose = geometry_pose
                        best_fallback_eval = evaluation
                    print(
                        "[INFO] "
                        f"{name} cuRobo candidate rank={rank} rejected approach "
                        f"segment={first.get('segment')} fraction={first.get('fraction')}",
                        flush=True,
                    )
                    continue

                if bool(args_cli.curobo_staged_selection):
                    if staged_final_screen_count >= staged_final_screen_limit:
                        evaluation = {
                            "rank": rank,
                            "score": float(candidate.get("score", 0.0)),
                            "ok": False,
                            "skipped_curobo": True,
                            "skip_reason": "staged_final_ik_screen_limit_reached",
                            "final_geometry_check": geometry_report,
                            "approach_collision_check": approach_report,
                            "candidate": {
                                "translation": candidate.get("translation"),
                                "rotation_matrix": candidate.get("rotation_matrix"),
                                "width": float(candidate.get("width", 0.0)),
                                "depth": float(candidate.get("depth", 0.0)),
                            },
                        }
                        tested.append(evaluation)
                        continue
                    staged_final_screen_count += 1
                    final_eval = curobo_probe_candidate_final_grasp(
                        planner,
                        robot,
                        current_q,
                        name,
                        geometry_pose,
                        object_poses[name],
                        target_points,
                        obstacle_points,
                    )
                    final_eval["final_geometry_check"] = geometry_report
                    final_eval["approach_collision_check"] = approach_report
                    final_eval["accepted_by_policy"] = False
                    final_eval["candidate"] = {
                        "translation": candidate.get("translation"),
                        "rotation_matrix": candidate.get("rotation_matrix"),
                        "width": float(candidate.get("width", 0.0)),
                        "depth": float(candidate.get("depth", 0.0)),
                        "heuristic": candidate.get("heuristic"),
                    }
                    selection_preference, selection_preference_key = box_arm_side_selection_preference(
                        name,
                        final_eval,
                        object_poses[name],
                    )
                    final_eval["selection_preference"] = selection_preference
                    tested.append(final_eval)
                    prior_key = baseline_action_prior_sort_value(final_eval)
                    final_key = selection_preference_key + (
                        prior_key,
                        -float(final_eval.get("action_soft_limit_min_margin_rad", 0.0)),
                        float(final_eval.get("ik_position_error_m", final_eval.get("max_required_ik_position_error_m", 1e9))),
                        float(final_eval.get("ik_rotation_error_rad", final_eval.get("max_required_ik_rotation_error_rad", 1e9))),
                        -float(final_eval["score"]),
                    )
                    fallback_key = (
                        0 if final_eval.get("final_action_feasible", False) else 1,
                        0 if final_eval.get("reached_grasp_geometry_ok", True) else 1,
                        *selection_preference_key,
                        float(final_eval.get("ik_position_error_m", 1e9)),
                        float(final_eval.get("ik_rotation_error_rad", 1e9)),
                        prior_key,
                        -float(final_eval.get("action_soft_limit_min_margin_rad", 0.0)),
                        -float(final_eval["score"]),
                    )
                    if best_fallback_key is None or fallback_key < best_fallback_key:
                        best_fallback_key = fallback_key
                        best_fallback_pose = geometry_pose
                        best_fallback_eval = final_eval
                    print(
                        "[INFO] "
                        f"{name} staged final-IK rank={rank} score={final_eval['score']:.4f} "
                        f"ok={final_eval['ok']} ik={final_eval['ik_success']} "
                        f"soft={final_eval['action_soft_joint_limit_ok']} "
                        f"action={final_eval.get('action_range_ok', True)} "
                        f"max_action={final_eval.get('action_range_check', {}).get('max_abs_action', 0.0):.3f} "
                        f"prior_l2={prior_key:.3f} "
                        f"margin={final_eval.get('action_soft_limit_min_margin_rad', 0.0):.4f} "
                        f"pos_err={final_eval['ik_position_error_m']:.4f} "
                        f"rot_err={final_eval['ik_rotation_error_rad']:.4f} "
                        f"reached_geom={final_eval.get('reached_grasp_geometry_ok', True)}",
                        flush=True,
                    )
                    if final_eval["ok"]:
                        staged_final_survivors.append(
                            (final_key, candidate, geometry_pose, geometry_report, approach_report, final_eval)
                        )
                    continue

                curobo_eval = curobo_probe_candidate(
                    planner,
                    robot,
                    current_q,
                    name,
                    geometry_pose,
                    object_poses[name],
                    target_points,
                    obstacle_points,
                )
                curobo_eval["final_geometry_check"] = geometry_report
                curobo_eval["approach_collision_check"] = approach_report
                curobo_eval["strict_ok"] = bool(curobo_eval.get("strict_all_stage_ok", curobo_eval["ok"]))
                required_ik_success = bool(curobo_eval.get("all_required_ik_success", curobo_eval["all_ik_success"]))
                required_action_feasible = bool(
                    curobo_eval.get("all_required_action_feasible", required_ik_success)
                )
                curobo_eval["accepted_by_policy"] = bool(
                    (
                        required_action_feasible
                        and curobo_eval.get("reached_grasp_geometry_ok", True)
                    )
                    if accept_first_ik_success
                    else curobo_eval["ok"]
                )
                curobo_eval["candidate"] = {
                    "translation": candidate.get("translation"),
                    "rotation_matrix": candidate.get("rotation_matrix"),
                    "width": float(candidate.get("width", 0.0)),
                    "depth": float(candidate.get("depth", 0.0)),
                }
                selection_preference, selection_preference_key = box_arm_side_selection_preference(
                    name,
                    curobo_eval,
                    object_poses[name],
                )
                curobo_eval["selection_preference"] = selection_preference
                tested.append(curobo_eval)
                prior_key = baseline_action_prior_sort_value(curobo_eval)
                fallback_key = (
                    0 if required_action_feasible else 1,
                    0 if curobo_eval.get("reached_grasp_geometry_ok", True) else 1,
                    *selection_preference_key,
                    prior_key,
                    float(curobo_eval.get("max_required_ik_position_error_m", curobo_eval["max_ik_position_error_m"])),
                    float(curobo_eval.get("max_required_ik_rotation_error_rad", curobo_eval["max_ik_rotation_error_rad"])),
                    -float(curobo_eval["score"]),
                )
                if best_fallback_key is None or fallback_key < best_fallback_key:
                    best_fallback_key = fallback_key
                    best_fallback_pose = geometry_pose
                    best_fallback_eval = curobo_eval
                print(
                    "[INFO] "
                    f"{name} cuRobo candidate rank={rank} score={curobo_eval['score']:.4f} "
                    f"ok={curobo_eval['ok']} accepted={curobo_eval['accepted_by_policy']} "
                    f"pos_err={curobo_eval['max_required_ik_position_error_m']:.4f} "
                    f"rot_err={curobo_eval['max_required_ik_rotation_error_rad']:.4f} "
                    f"all_stage_pos_err={curobo_eval['max_ik_position_error_m']:.4f} "
                    f"soft_limits={curobo_eval.get('all_required_soft_joint_limit_ok', True)} "
                    f"action_range={curobo_eval.get('all_required_action_range_ok', True)} "
                    f"prior_l2={prior_key:.3f} "
                    f"reached_geom={curobo_eval.get('reached_grasp_geometry_ok', True)}",
                    flush=True,
                )
                if curobo_eval["accepted_by_policy"]:
                    selected_pose = geometry_pose
                    selected_eval = curobo_eval
                    tested[-1]["selected"] = True
                    break

            if bool(args_cli.curobo_staged_selection) and selected_eval is None and staged_final_survivors:
                staged_final_survivors.sort(key=lambda item: item[0])
                for full_index, (
                    _final_key,
                    candidate,
                    geometry_pose,
                    geometry_report,
                    approach_report,
                    final_eval,
                ) in enumerate(staged_final_survivors[:staged_full_sequence_limit], start=1):
                    final_eval["promoted_to_full_sequence"] = True
                    rank = int(candidate.get("source_rank", 0))
                    curobo_eval = curobo_probe_candidate(
                        planner,
                        robot,
                        current_q,
                        name,
                        geometry_pose,
                        object_poses[name],
                        target_points,
                        obstacle_points,
                    )
                    curobo_eval["staged_selection"] = {
                        "enabled": True,
                        "full_sequence_order": int(full_index),
                        "final_screen": final_eval,
                    }
                    curobo_eval["final_geometry_check"] = geometry_report
                    curobo_eval["approach_collision_check"] = approach_report
                    curobo_eval["strict_ok"] = bool(curobo_eval.get("strict_all_stage_ok", curobo_eval["ok"]))
                    required_ik_success = bool(
                        curobo_eval.get("all_required_ik_success", curobo_eval["all_ik_success"])
                    )
                    required_action_feasible = bool(
                        curobo_eval.get("all_required_action_feasible", required_ik_success)
                    )
                    curobo_eval["accepted_by_policy"] = bool(
                        (
                            required_action_feasible
                            and curobo_eval.get("reached_grasp_geometry_ok", True)
                        )
                        if accept_first_ik_success
                        else curobo_eval["ok"]
                    )
                    curobo_eval["candidate"] = {
                        "translation": candidate.get("translation"),
                        "rotation_matrix": candidate.get("rotation_matrix"),
                        "width": float(candidate.get("width", 0.0)),
                        "depth": float(candidate.get("depth", 0.0)),
                        "heuristic": candidate.get("heuristic"),
                    }
                    selection_preference, selection_preference_key = box_arm_side_selection_preference(
                        name,
                        curobo_eval,
                        object_poses[name],
                    )
                    curobo_eval["selection_preference"] = selection_preference
                    tested.append(curobo_eval)
                    prior_key = baseline_action_prior_sort_value(curobo_eval)
                    fallback_key = (
                        0 if required_action_feasible else 1,
                        0 if curobo_eval.get("reached_grasp_geometry_ok", True) else 1,
                        *selection_preference_key,
                        prior_key,
                        -float(curobo_eval.get("min_required_action_soft_limit_margin_rad", 0.0)),
                        float(
                            curobo_eval.get(
                                "max_required_ik_position_error_m",
                                curobo_eval["max_ik_position_error_m"],
                            )
                        ),
                        float(
                            curobo_eval.get(
                                "max_required_ik_rotation_error_rad",
                                curobo_eval["max_ik_rotation_error_rad"],
                            )
                        ),
                        -float(curobo_eval["score"]),
                    )
                    if best_fallback_key is None or fallback_key < best_fallback_key:
                        best_fallback_key = fallback_key
                        best_fallback_pose = geometry_pose
                        best_fallback_eval = curobo_eval
                    print(
                        "[INFO] "
                        f"{name} staged full-IK #{full_index} rank={rank} score={curobo_eval['score']:.4f} "
                        f"ok={curobo_eval['ok']} accepted={curobo_eval['accepted_by_policy']} "
                        f"margin={curobo_eval.get('min_required_action_soft_limit_margin_rad', 0.0):.4f} "
                        f"prior_l2={prior_key:.3f} "
                        f"pos_err={curobo_eval['max_required_ik_position_error_m']:.4f} "
                        f"rot_err={curobo_eval['max_required_ik_rotation_error_rad']:.4f} "
                        f"soft_limits={curobo_eval.get('all_required_soft_joint_limit_ok', True)} "
                        f"action_range={curobo_eval.get('all_required_action_range_ok', True)} "
                        f"reached_geom={curobo_eval.get('reached_grasp_geometry_ok', True)}",
                        flush=True,
                    )
                    if curobo_eval["accepted_by_policy"]:
                        selected_pose = geometry_pose
                        selected_eval = curobo_eval
                        tested[-1]["selected"] = True
                        break

            if selected_eval is None and best_fallback_eval is not None:
                selected_pose = best_fallback_pose
                selected_eval = best_fallback_eval
                for item in tested:
                    item["selected"] = bool(item is selected_eval)

            selection = {
                "mode": args_cli.grasp_selection,
                "object_accept_first_ik_success": bool(accept_first_ik_success),
                "backend": (
                    "curobo_ik_scene_collision_plus_piper_point_cloud_geometry"
                    if bool(args_cli.curobo_scene_collision)
                    else "curobo_ik_plus_piper_point_cloud_geometry_scene_collision_disabled"
                ),
                "note": (
                    "Candidates are tried in generator score order. Each generator pose is converted "
                    "to Piper gripper_base, searched over configured offsets, evaluated with clipped "
                    "Piper jaw width, rejected if target/surrounding points penetrate finger or palm "
                    "solids, required to contain target points along the finger centerline/closing "
                    "region and at a middle support cross-section, required to keep the finger-root "
                    "centerline clear of target points, swept through the approach segment, then solved through cuRobo for "
                    "pregrasp stage, pregrasp, grasp, and lift. Required CuRobo stage joints must also "
                    "fit Isaac/Piper action-level soft joint limits. The returned CuRobo grasp joint "
                    "solution is FK'd and the actual reached gripper pose is checked against the "
                    "same target/obstacle point-cloud geometry. When enabled, CuRobo scene collision "
                    "uses voxel cuboids made from the fused scene cloud after excluding the segmented target."
                ),
                "acceptance_policy": (
                    (
                        "first_required_ik_success_relaxed_pregrasp_stage_after_geometry_approach_and_reached_pose_filters"
                        if bool(args_cli.relax_curobo_pregrasp_stage)
                        else "first_all_ik_success_after_geometry_approach_and_reached_pose_filters"
                    )
                    if accept_first_ik_success
                    else "strict_geometry_approach_and_curobo_tolerances"
                ),
                "top_k": int(top_k),
                "selected_rank": int(selected_pose.get("source_rank", 1)),
                "selected_score": float(selected_pose.get("score", 0.0)),
                "selected_ok": bool(selected_eval.get("accepted_by_policy", selected_eval.get("ok", False)))
                if selected_eval is not None
                else False,
                "selected_strict_ok": bool(selected_eval.get("strict_ok", selected_eval.get("ok", False)))
                if selected_eval is not None
                else False,
                "fallback_used": bool(
                    selected_eval is None
                    or not bool(selected_eval.get("accepted_by_policy", selected_eval.get("ok", False)))
                ),
                "clouds": cloud_meta,
                "criteria": {
                    "curobo_acceptance_policy": (
                        (
                            "first_required_ik_success_relaxed_pregrasp_stage_after_geometry_approach_and_reached_pose_filters"
                            if bool(args_cli.relax_curobo_pregrasp_stage)
                            else "first_all_ik_success_after_geometry_approach_and_reached_pose_filters"
                        )
                        if accept_first_ik_success
                        else "strict_geometry_approach_and_curobo_tolerances"
                    ),
                    "relax_curobo_pregrasp_stage": bool(args_cli.relax_curobo_pregrasp_stage),
                    "curobo_reached_pose_pc_filter": not bool(args_cli.disable_curobo_reached_pose_pc_filter),
                    "piper_max_jaw_width_m": float(args_cli.piper_max_jaw_width),
                    "piper_clip_generator_width": bool(args_cli.piper_clip_generator_width),
                    "piper_offset_modes": list(args_cli.piper_offset_modes),
                    "piper_offset_min_m": float(args_cli.piper_offset_min),
                    "piper_offset_max_m": float(args_cli.piper_offset_max),
                    "piper_offset_step_m": float(args_cli.piper_offset_step),
                    "target_solid_max_points": int(args_cli.target_solid_max_points),
                    "obstacle_solid_max_points": int(args_cli.obstacle_solid_max_points),
                    "centerline_min_points": int(args_cli.centerline_min_points),
                    "effective_centerline_min_points": int(effective_centerline_min_points(name)),
                    "centerline_relaxed": bool(name in set(args_cli.centerline_relaxed_objects)),
                    "closing_region_min_points": int(args_cli.closing_region_min_points),
                    "middle_support_min_points": int(args_cli.middle_support_min_points),
                    "effective_middle_support_min_points": int(effective_middle_support_min_points(name)),
                    "middle_support_relaxed": bool(name in set(args_cli.middle_support_relaxed_objects)),
                    "middle_support_offset_m": float(args_cli.middle_support_offset),
                    "tcp_support_objects": list(args_cli.tcp_support_objects),
                    "tcp_support_min_points": int(args_cli.tcp_support_min_points),
                    "effective_tcp_support_min_points": int(effective_tcp_support_min_points(name)),
                    "tcp_support_offset_m": float(args_cli.tcp_support_offset),
                    "tcp_support_half_length_m": float(args_cli.tcp_support_half_length),
                    "tcp_support_half_width_m": float(args_cli.tcp_support_half_width),
                    "tcp_support_half_depth_m": float(args_cli.tcp_support_half_depth),
                    "root_centerline_clear_length_m": float(args_cli.heuristic_root_centerline_clear_length),
                    "root_centerline_max_points": int(args_cli.heuristic_root_centerline_max_points),
                    "curobo_scene_collision": bool(args_cli.curobo_scene_collision),
                    "curobo_require_action_soft_limits": bool(args_cli.curobo_require_action_soft_limits),
                    "curobo_require_action_range": bool(args_cli.curobo_require_action_range),
                    "curobo_action_limit": float(args_cli.curobo_action_limit),
                    "curobo_baseline_action_prior": bool(args_cli.curobo_baseline_action_prior),
                    "curobo_baseline_action_prior_ranking": bool(args_cli.curobo_baseline_action_prior_ranking),
                    "box_selection_preference": (
                        "nearest_arm_side_then_highest_contact_pose"
                        if name == "box_object"
                        else None
                    ),
                    "curobo_staged_selection": bool(args_cli.curobo_staged_selection),
                    "curobo_staged_final_ik_top_k": int(args_cli.curobo_staged_final_ik_top_k),
                    "curobo_staged_full_ik_top_k": int(args_cli.curobo_staged_full_ik_top_k),
                    "curobo_obstacle_max_cuboids": int(args_cli.curobo_obstacle_max_cuboids),
                    "curobo_obstacle_voxel_size_m": float(args_cli.curobo_obstacle_voxel_size),
                    "curobo_position_tol_m": float(args_cli.curobo_position_tol),
                    "curobo_rotation_tol_rad": float(args_cli.curobo_rotation_tol),
                },
                "candidates": tested,
            }
            write_selected_grasp(record, selected_pose, selection)
    finally:
        planner.close()
    return obs


def select_ik_feasible_grasps(env, obs, robot, object_poses: dict[str, dict], records: dict[str, dict]) -> dict:
    if args_cli.grasp_selection in {"curobo_feasible", "curobo_first_ik"}:
        return select_curobo_feasible_grasps(env, obs, robot, object_poses, records)
    if args_cli.grasp_selection != "ik_feasible":
        return obs

    arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
    gripper_ids, _ = robot.find_joints(GRIPPER_JOINT_NAMES)
    controller = CartesianController(
        robot=robot,
        ee_body_name="gripper_base",
        arm_joint_names=ARM_JOINT_NAMES,
        num_envs=1,
        device=env.unwrapped.device,
        command_type="pose",
        lambda_val=0.05,
        max_joint_delta=0.16,
    )
    default_jpos = robot.data.default_joint_pos.clone()

    for name in args_cli.object_order:
        if name not in records:
            continue
        record = records[name]
        candidates = anygrasp_candidates(record)
        if not candidates:
            continue
        object_states = snapshot_rigid_object_states(env)
        controller.reset()
        obs, _ = settle_ik_pose(
            env,
            obs,
            robot,
            controller,
            arm_ids,
            gripper_ids,
            default_jpos,
            np.array([RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z], dtype=np.float64),
            DEFAULT_QUAT_WXYZ,
            args_cli.ik_filter_stage_steps,
            "safe_retract_before_ik_filter",
        )
        robot_state = snapshot_robot_joint_state(robot)
        obs = restore_robot_and_rigid_object_states(env, robot, robot_state, object_states, obs)
        probe_start_jpos = robot_state["joint_pos"].clone()
        top_k = len(candidates) if args_cli.ik_filter_top_k <= 0 else min(args_cli.ik_filter_top_k, len(candidates))
        tested = []
        selected_pose = candidates[0]
        selected_eval = None
        best_fallback_pose = candidates[0]
        best_fallback_eval = None
        best_fallback_key = None
        offset_search = None
        straight_approach_search = None
        for candidate in candidates[:top_k]:
            candidate_pose = candidate
            candidate_offset_search = None
            if name in args_cli.pc_offset_search_objects:
                candidate_pose, _, candidate_offset_search = apply_selected_offset_search(
                    record,
                    name,
                    candidate,
                    None,
                    object_poses[name],
                )
                print(
                    "[INFO] "
                    f"{name} PC offset candidate rank={candidate.get('source_rank')} "
                    f"offset={candidate_offset_search['selected_offset_m']:.4f} "
                    f"reason={candidate_offset_search['selected_reason']} "
                    f"accepted={candidate_offset_search['selected_accepted']} "
                    f"per_finger={candidate_offset_search.get('selected_per_finger_collision_points')}",
                    flush=True,
                )
                if not candidate_offset_search["selected_accepted"]:
                    skipped_eval = {
                        "rank": int(candidate.get("source_rank", 0)),
                        "score": float(candidate.get("score", 0.0)),
                        "ok": False,
                        "skipped_ik": True,
                        "skip_reason": "pc_offset_search_not_accepted",
                        "ik_probe_pose": "not_run",
                        "pc_offset_search": candidate_offset_search,
                    }
                    tested.append(skipped_eval)
                    fallback_key = (2, -float(candidate.get("score", 0.0)))
                    if best_fallback_key is None or fallback_key < best_fallback_key:
                        best_fallback_key = fallback_key
                        best_fallback_pose = candidate_pose
                        best_fallback_eval = skipped_eval
                    continue

            candidate_straight_approach_search = None
            if name in args_cli.pc_offset_search_objects and not args_cli.disable_straight_approach_pc_search:
                candidate_pose, _, candidate_straight_approach_search = apply_straight_approach_search(
                    record,
                    name,
                    candidate_pose,
                    None,
                    object_poses[name],
                )
                print(
                    "[INFO] "
                    f"{name} straight approach rank={candidate.get('source_rank')} "
                    f"accepted={candidate_straight_approach_search['accepted']} "
                    f"pregrasp={candidate_straight_approach_search['selected_pregrasp_distance_m']:.4f} "
                    f"reason={candidate_straight_approach_search['selected_reason']}",
                    flush=True,
                )
                if not candidate_straight_approach_search["accepted"]:
                    skipped_eval = {
                        "rank": int(candidate.get("source_rank", 0)),
                        "score": float(candidate.get("score", 0.0)),
                        "ok": False,
                        "skipped_ik": True,
                        "skip_reason": "straight_approach_not_clear",
                        "ik_probe_pose": "not_run",
                        "straight_approach_search": candidate_straight_approach_search,
                    }
                    if candidate_offset_search is not None:
                        skipped_eval["pc_offset_search"] = candidate_offset_search
                    tested.append(skipped_eval)
                    fallback_key = (1, -float(candidate.get("score", 0.0)))
                    if best_fallback_key is None or fallback_key < best_fallback_key:
                        best_fallback_key = fallback_key
                        best_fallback_pose = candidate_pose
                        best_fallback_eval = skipped_eval
                    continue

            approach_collision_check = None
            if name in args_cli.pc_offset_search_objects and not args_cli.disable_approach_pc_collision_filter:
                approach_collision_check = point_cloud_approach_collision_check(
                    record,
                    name,
                    candidate_pose,
                    object_poses[name],
                )
                print(
                    "[INFO] "
                    f"{name} approach PC collision rank={candidate.get('source_rank')} "
                    f"safe={approach_collision_check['safe']} "
                    f"max_points={approach_collision_check['max_collision_point_count']}",
                    flush=True,
                )
                if not approach_collision_check["safe"]:
                    skipped_eval = {
                        "rank": int(candidate.get("source_rank", 0)),
                        "score": float(candidate.get("score", 0.0)),
                        "ok": False,
                        "skipped_ik": True,
                        "skip_reason": "approach_pc_collision",
                        "ik_probe_pose": "not_run",
                        "approach_pc_collision_check": approach_collision_check,
                    }
                    if candidate_offset_search is not None:
                        skipped_eval["pc_offset_search"] = candidate_offset_search
                    if candidate_straight_approach_search is not None:
                        skipped_eval["straight_approach_search"] = candidate_straight_approach_search
                    tested.append(skipped_eval)
                    fallback_key = (1, -float(candidate.get("score", 0.0)))
                    if best_fallback_key is None or fallback_key < best_fallback_key:
                        best_fallback_key = fallback_key
                        best_fallback_pose = candidate_pose
                        best_fallback_eval = skipped_eval
                    obs = restore_robot_and_rigid_object_states(env, robot, robot_state, object_states, obs)
                    continue

            obs, evaluation = probe_candidate_ik(
                env,
                obs,
                robot,
                controller,
                arm_ids,
                gripper_ids,
                default_jpos,
                name,
                candidate_pose,
                object_poses[name],
                reset_jpos=probe_start_jpos,
            )
            evaluation["ik_probe_used_zero_offset"] = False
            evaluation["ik_probe_pose"] = "final_offset_applied" if candidate_offset_search is not None else "final_configured_pose"
            if candidate_offset_search is not None:
                evaluation["pc_offset_search"] = candidate_offset_search
            if candidate_straight_approach_search is not None:
                evaluation["straight_approach_search"] = candidate_straight_approach_search
            if approach_collision_check is not None:
                evaluation["approach_pc_collision_check"] = approach_collision_check
            tested.append(evaluation)
            print(
                "[INFO] "
                f"{name} IK candidate rank={evaluation['rank']} "
                f"score={evaluation['score']:.4f} ok={evaluation['ok']} "
                f"critical_error={evaluation['max_critical_position_error_m']:.4f} "
                f"pregrasp_object_motion={evaluation.get('max_pregrasp_object_motion_m', 0.0):.4f} "
                f"pose={evaluation['ik_probe_pose']}",
                flush=True,
            )
            fallback_key = (
                0 if evaluation["stop_ok"] else 1,
                float(evaluation["max_critical_position_error_m"]),
                -float(evaluation["min_critical_orientation_abs_dot"]),
                -float(evaluation["score"]),
            )
            if best_fallback_key is None or fallback_key < best_fallback_key:
                best_fallback_key = fallback_key
                best_fallback_pose = candidate_pose
                best_fallback_eval = evaluation
            if evaluation["ok"]:
                candidate_eval = evaluation
                selected_pose = candidate_pose
                selected_eval = candidate_eval
                offset_search = candidate_offset_search
                straight_approach_search = candidate_straight_approach_search
                break
            obs = restore_robot_and_rigid_object_states(env, robot, robot_state, object_states, obs)
        if selected_eval is None and best_fallback_eval is not None:
            selected_pose = best_fallback_pose
            selected_eval = best_fallback_eval
        if offset_search is None and name not in args_cli.pc_offset_search_objects:
            selected_pose, selected_eval, offset_search = apply_selected_offset_search(
                record,
                name,
                selected_pose,
                selected_eval,
                object_poses[name],
            )
            if offset_search is not None and selected_eval is not None:
                for idx, evaluation in enumerate(tested):
                    if int(evaluation.get("rank", -1)) == int(selected_pose.get("source_rank", -999)):
                        tested[idx] = selected_eval
                        break

        selection = {
            "mode": "ik_feasible",
            "backend": "isaaclab_cartesian_controller",
            "note": (
                "Candidates are tried in AnyGrasp score order after converting each "
                "AnyGrasp gripper pose to Piper gripper_base with the configured tool transform. "
                "When point-cloud offset search is enabled for an object, each candidate is first "
                "converted into its final offset-applied grasp pose, and only that final pose is "
                "probed for IK. The fused object point cloud is also used to reject approach paths "
                "that collide with the object before the final contact window. During the physics IK "
                "probe, candidates are rejected if task objects move before the final grasp waypoint. "
                "The robot is reset to the saved safe-retract joint state before each candidate probe."
            ),
            "top_k": int(top_k),
            "ik_filter_raw_anygrasp": bool(args_cli.ik_filter_raw_anygrasp),
            "ik_filter_final_pose_only": True,
            "selected_rank": int(selected_pose.get("source_rank", 1)),
            "selected_score": float(selected_pose.get("score", 0.0)),
            "selected_ok": bool(selected_eval["ok"]) if selected_eval is not None else False,
            "fallback_used": bool(selected_eval is None or not selected_eval["ok"]),
            "pc_offset_search": offset_search,
            "straight_approach_search": straight_approach_search,
            "candidates": tested,
        }
        write_selected_grasp(record, selected_pose, selection)
        obs = restore_robot_and_rigid_object_states(env, robot, robot_state, object_states, obs)
    return obs


def object_poses_for_request(records: dict[str, dict]) -> dict[str, dict]:
    poses = deterministic_object_poses(args_cli.seed)
    if args_cli.object_center_source == "deterministic":
        return poses

    for name, record in records.items():
        source_name = args_cli.object_center_source
        source = record.get("ee_pose_estimate" if source_name == "ee_mask" else "video_pose_estimate")
        if not isinstance(source, dict) or source.get("status") == "failed":
            source_name = "video_mask"
            source = record.get("video_pose_estimate")
        if not isinstance(source, dict) or source.get("status") == "failed":
            continue
        center = source.get("center_world_median") or source.get("center_world")
        if center is None:
            continue
        poses[name]["center_w"] = [float(center[0]), float(center[1]), float(TABLE_TOP_Z + 0.05)]
        poses[name]["source"] = f"{source_name}_depth_median_xy"
    return poses


def build_grasp_records(records: dict[str, dict]) -> dict[str, dict]:
    grasp_records = {}
    for name, record in records.items():
        anygrasp_result = record["anygrasp_result"]
        anygrasp_payload = anygrasp_result.get("anygrasp", {})
        if anygrasp_payload.get("status") != "ok":
            raise RuntimeError(f"AnyGrasp failed for {name}: {anygrasp_payload}")
        grasp_records[name] = {
            "sam3": {
                "video": record["video_sam3"],
                "ee": record["ee_sam3"],
                "selected_view": (
                    "fused_ee_camera_views"
                    if record.get("ee_extra_view_count", 0)
                    else f"ee_camera_{record.get('hover', {}).get('effective_mode', 'adaptive')}"
                ),
            },
            "ee_views": record.get("ee_views"),
            "mask_center_estimate": {
                "video": record["video_pose_estimate"],
                "ee": record["ee_pose_estimate"],
            },
            "anygrasp_result_path": record["anygrasp_result_path"],
            "final_grasp_pose_path": record["final_grasp_pose_path"],
            "anygrasp_status": anygrasp_payload.get("status"),
            "final_grasp_pose": load_json(record["final_grasp_pose_path"]),
        }
    return grasp_records


def run_video_rough_for_object(env, obs: dict, name: str, video_dir: Path) -> dict:
    video_dir.mkdir(parents=True, exist_ok=True)
    save_camera_frame(video_dir, "video", obs["image"])
    video_camera = camera_metadata(env, "video_cam")
    write_camera_json(video_dir / "video_camera.json", video_camera)
    video_depth = np.load(video_dir / "video_depth.npy")

    sam3_dir = video_dir / "sam3" / name
    mask_path, detection = run_sam3(
        video_dir / "video_rgb.png",
        sam3_dir,
        name,
        f"video_{name}",
        "eye-to-hand / video_cam",
    )
    video_mask = load_mask(mask_path, video_depth.shape)
    try:
        video_pose = estimate_pose_from_mask(video_depth, video_mask, video_camera, args_cli.max_depth)
        video_pose["source"] = "video_cam_sam3_depth"
    except Exception as exc:
        video_pose = scene_object_pose_estimate(env, name, repr(exc))
    video_sam3_pose = dict(video_pose)
    video_color_refine = None
    if name in args_cli.video_color_refine_objects and video_pose.get("source") == "video_cam_sam3_depth":
        _, video_color_refine = refine_video_pose_with_color(
            video_dir / "video_rgb.png",
            video_depth,
            video_camera,
            video_mask,
            name,
            video_dir / "color_refine" / name,
            f"video_{name}_color_refine",
        )
        if video_color_refine is not None:
            video_pose = dict(video_color_refine["pose_estimate"])
    target_overlay = video_dir / f"{name}_target_overlay.png"
    draw_target_overlay(
        video_dir / "video_rgb.png",
        target_overlay,
        video_pose,
        f"{name} rough target from video",
    )
    return {
        "video_rough_dir": str(video_dir),
        "video_target_overlay_path": str(target_overlay),
        "video_sam3": {
            "prompt": detection.get("prompt"),
            "mask_count": detection.get("mask_count"),
            "best_index": detection.get("best_index"),
            "areas_px": detection.get("areas_px"),
            "scores": detection.get("scores"),
            "mask_path": str(mask_path),
        },
        "video_sam3_pose_estimate": video_sam3_pose,
        "video_color_refinement": video_color_refine,
        "video_pose_estimate": video_pose,
    }


def estimate_ee_pose_for_view(
    current_mask_path: Path,
    current_depth: np.ndarray,
    current_camera: dict,
    source_label: str,
    current_hover: dict,
) -> dict:
    current_mask = load_mask(current_mask_path, current_depth.shape)
    try:
        pose_estimate = estimate_pose_from_mask(current_depth, current_mask, current_camera, args_cli.max_depth)
        pose_estimate["source"] = (
            f"ee_camera_{current_hover.get('effective_mode', 'adaptive')}_{source_label}_depth"
        )
        return pose_estimate
    except Exception as exc:
        return {
            "status": "failed",
            "error": repr(exc),
            "source": f"ee_camera_{current_hover.get('effective_mode', 'adaptive')}_{source_label}_depth",
        }


def capture_primary_ee_view(
    env,
    obs: dict,
    robot,
    ee_body_idx: int,
    ee_camera_calibration: dict,
    name: str,
    capture_dir: Path,
    desired_camera_pos: list[float],
    rough_center_for_hover: list[float],
    hover_offset: dict[str, float],
    forced_hover_mode: str | None = None,
) -> tuple[dict, dict]:
    obs, hover = move_to_topdown_camera(
        env,
        obs,
        robot,
        desired_camera_pos,
        ee_camera_calibration,
        object_name=name,
        look_at_target_w=rough_center_for_hover,
        forced_hover_mode=forced_hover_mode,
    )
    hover["object_hover_offset"] = {
        "dx": float(hover_offset.get("dx", 0.0)),
        "dy": float(hover_offset.get("dy", 0.0)),
        "dz": float(hover_offset.get("dz", 0.0)),
    }
    if forced_hover_mode is not None:
        hover["forced_hover_mode"] = forced_hover_mode

    capture_dir.mkdir(parents=True, exist_ok=True)
    save_camera_frame(capture_dir, "ee", obs["image"])
    save_camera_frame(
        capture_dir,
        "video_hover",
        {
            "video_hover_rgb": obs["image"]["video_rgb"],
            "video_hover_depth": obs["image"]["video_depth"],
        },
    )
    ee_camera = ee_camera_metadata_from_gripper(env, robot, ee_body_idx, ee_camera_calibration)
    write_camera_json(capture_dir / "ee_camera.json", ee_camera)
    ee_view_label = (
        "eye-in-hand look-at / ee_camera"
        if hover.get("effective_mode") == "look_at"
        else "eye-in-hand top-down / ee_camera"
    )

    ee_mask_dir = capture_dir / ("sam3" if args_cli.ee_mask_source == "sam3" else "color_mask")
    mask_source_used = args_cli.ee_mask_source
    mask_fallback_reason = None
    if args_cli.ee_mask_source == "sam3":
        mask_path, detection = run_sam3(
            capture_dir / "ee_rgb.png",
            ee_mask_dir,
            name,
            f"ee_{name}",
            ee_view_label,
        )
    else:
        mask_path, detection = run_color_mask(
            capture_dir / "ee_rgb.png",
            ee_mask_dir,
            name,
            f"ee_{name}",
            ee_view_label,
        )
    if args_cli.ee_mask_source == "sam3" and not detection.get("mask_count"):
        mask_path, detection = run_color_mask(
            capture_dir / "ee_rgb.png",
            capture_dir / "color_mask",
            name,
            f"ee_{name}",
            f"{ee_view_label} color fallback",
        )
        mask_source_used = "color"
        mask_fallback_reason = "sam3_empty_mask"
    ee_depth = np.load(capture_dir / "ee_depth.npy")
    ee_pose = estimate_ee_pose_for_view(mask_path, ee_depth, ee_camera, mask_source_used, hover)
    draw_target_overlay(
        capture_dir / "ee_rgb.png",
        capture_dir / f"{name}_ee_target_overlay.png",
        ee_pose,
        f"{name} EE {hover.get('effective_mode', 'adaptive')} target",
    )
    view_record = {
        "view_index": 1,
        "role": "primary",
        "capture_dir": str(capture_dir),
        "hover": hover,
        "camera": ee_camera,
        "mask_source_used": mask_source_used,
        "mask_fallback_reason": mask_fallback_reason,
        "sam3": {
            "prompt": detection.get("prompt"),
            "source": detection.get("source", args_cli.ee_mask_source),
            "mask_count": detection.get("mask_count"),
            "best_index": detection.get("best_index"),
            "areas_px": detection.get("areas_px"),
            "scores": detection.get("scores"),
            "mask_path": str(mask_path),
        },
        "pose_estimate": ee_pose,
    }
    usable = bool(detection.get("mask_count")) and ee_pose.get("status") != "failed"
    return obs, {
        "capture_dir": capture_dir,
        "hover": hover,
        "ee_camera": ee_camera,
        "ee_view_label": ee_view_label,
        "mask_path": mask_path,
        "detection": detection,
        "mask_source_used": mask_source_used,
        "mask_fallback_reason": mask_fallback_reason,
        "ee_depth": ee_depth,
        "ee_pose": ee_pose,
        "ee_view": view_record,
        "sam3_usable": usable,
    }


def write_pipeline_summary(
    records: dict[str, dict],
    request: dict | None = None,
    request_path: Path | None = None,
    execution: dict | None = None,
) -> Path:
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(OUTPUT_DIR),
        "request": str(request_path) if request_path is not None else None,
        "seed": args_cli.seed,
        "object_order": list(args_cli.object_order),
        "object_center_source": args_cli.object_center_source,
        "hover_target_source": args_cli.hover_target_source,
        "video_color_refine_objects": list(args_cli.video_color_refine_objects),
        "ee_mask_source": args_cli.ee_mask_source,
        "grasp_generator": args_cli.grasp_generator,
        "grasp_generator_overrides": dict(args_cli.grasp_generator_overrides),
        "anygrasp_symmetric_cloud_mode": args_cli.anygrasp_symmetric_cloud_mode,
        "anygrasp_symmetry_center_source": args_cli.anygrasp_symmetry_center_source,
        "anygrasp_symmetric_surface_points": int(args_cli.anygrasp_symmetric_surface_points),
        "graspgen_symmetric_cloud_mode": args_cli.graspgen_symmetric_cloud_mode,
        "graspgen_symmetry_center_source": args_cli.graspgen_symmetry_center_source,
        "graspgen_symmetric_surface_points": int(args_cli.graspgen_symmetric_surface_points),
        "heuristic_profile": args_cli.heuristic_profile,
        "heuristic_profile_overrides": dict(args_cli.heuristic_profile_overrides),
        "heuristic_attempts": int(args_cli.heuristic_attempts),
        "heuristic_symmetric_cloud_mode": args_cli.heuristic_symmetric_cloud_mode,
        "heuristic_symmetric_cloud_mode_overrides": dict(args_cli.heuristic_symmetric_cloud_mode_overrides),
        "heuristic_symmetry_center_source": args_cli.heuristic_symmetry_center_source,
        "heuristic_symmetric_surface_points": int(args_cli.heuristic_symmetric_surface_points),
        "heuristic_symmetric_top_grasp_fraction": float(args_cli.heuristic_symmetric_top_grasp_fraction),
        "heuristic_candidate_filter_max_points": int(args_cli.heuristic_candidate_filter_max_points),
        "heuristic_family_mode": args_cli.heuristic_family_mode,
        "heuristic_root_centerline_clear_length": float(args_cli.heuristic_root_centerline_clear_length),
        "heuristic_root_centerline_max_points": int(args_cli.heuristic_root_centerline_max_points),
        "heuristic_vary_seed_by_attempt": bool(args_cli.heuristic_vary_seed_by_attempt),
        "heuristic_attempt_seed_stride": int(args_cli.heuristic_attempt_seed_stride),
        "centerline_relaxed_objects": list(args_cli.centerline_relaxed_objects),
        "middle_support_relaxed_objects": list(args_cli.middle_support_relaxed_objects),
        "heuristic_symmetric_ee_roll": bool(args_cli.heuristic_symmetric_ee_roll),
        "contact_graspnet_env": args_cli.contact_graspnet_env,
        "contact_graspnet_root": str(args_cli.contact_graspnet_root),
        "contact_graspnet_ckpt_dir": str(args_cli.contact_graspnet_ckpt_dir),
        "contact_graspnet_forward_passes": int(args_cli.contact_graspnet_forward_passes),
        "contact_graspnet_local_regions": bool(args_cli.contact_graspnet_local_regions),
        "contact_graspnet_filter_grasps": bool(args_cli.contact_graspnet_filter_grasps),
        "hover_mode": args_cli.hover_mode,
        "hover_height": args_cli.hover_height,
        "gripper_hover_offset": list(args_cli.gripper_hover_offset),
        "lookat_objects": list(args_cli.lookat_objects),
        "lookat_camera_position": list(args_cli.lookat_camera_position),
        "lookat_camera_offset": list(args_cli.lookat_camera_offset),
        "hover_object_offsets": args_cli.hover_object_offsets,
        "hover_center_iters": args_cli.hover_center_iters,
        "hover_center_steps": args_cli.hover_center_steps,
        "hover_center_pixel": list(args_cli.hover_center_pixel),
        "dual_hover_objects": list(args_cli.dual_hover_objects),
        "dual_hover_modes": list(args_cli.dual_hover_modes),
        "ee_multi_view_objects": list(args_cli.ee_multi_view_objects),
        "anygrasp_video_view_objects": list(args_cli.anygrasp_video_view_objects),
        "anygrasp_full_scene_objects": list(args_cli.anygrasp_full_scene_objects),
        "anygrasp_scene_cloud_stride": int(args_cli.anygrasp_scene_cloud_stride),
        "anygrasp_scene_cloud_max_points": int(args_cli.anygrasp_scene_cloud_max_points),
        "target_grasp_filter_distance": float(args_cli.target_grasp_filter_distance),
        "target_grasp_filter_pixel_radius": int(args_cli.target_grasp_filter_pixel_radius),
        "ee_second_view_offset": list(args_cli.ee_second_view_offset),
        "grasp_mode": args_cli.grasp_mode,
        "gripper_base_offset": float(args_cli.gripper_base_offset),
        "gripper_base_offset_mode": args_cli.gripper_base_offset_mode,
        "grasp_selection": args_cli.grasp_selection,
        "curobo_first_ik_objects": list(args_cli.curobo_first_ik_objects),
        "ik_filter_raw_anygrasp": bool(args_cli.ik_filter_raw_anygrasp),
        "ik_filter_top_k": int(args_cli.ik_filter_top_k),
        "ik_filter_position_tol": float(args_cli.ik_filter_position_tol),
        "ik_filter_orientation_dot": float(args_cli.ik_filter_orientation_dot),
        "piper_max_jaw_width": float(args_cli.piper_max_jaw_width),
        "piper_clip_generator_width": bool(args_cli.piper_clip_generator_width),
        "piper_offset_modes": list(args_cli.piper_offset_modes),
        "piper_offset_min": float(args_cli.piper_offset_min),
        "piper_offset_max": float(args_cli.piper_offset_max),
        "piper_offset_step": float(args_cli.piper_offset_step),
        "target_solid_max_points": int(args_cli.target_solid_max_points),
        "obstacle_solid_max_points": int(args_cli.obstacle_solid_max_points),
        "centerline_min_points": int(args_cli.centerline_min_points),
        "closing_region_min_points": int(args_cli.closing_region_min_points),
        "centerline_half_width": float(args_cli.centerline_half_width),
        "centerline_half_depth": float(args_cli.centerline_half_depth),
        "middle_support_min_points": int(args_cli.middle_support_min_points),
        "middle_support_offset": float(args_cli.middle_support_offset),
        "middle_support_half_length": float(args_cli.middle_support_half_length),
        "middle_support_half_width": float(args_cli.middle_support_half_width),
        "middle_support_half_depth": float(args_cli.middle_support_half_depth),
        "tcp_support_objects": list(args_cli.tcp_support_objects),
        "tcp_support_min_points": int(args_cli.tcp_support_min_points),
        "tcp_support_offset": float(args_cli.tcp_support_offset),
        "tcp_support_half_length": float(args_cli.tcp_support_half_length),
        "tcp_support_half_width": float(args_cli.tcp_support_half_width),
        "tcp_support_half_depth": float(args_cli.tcp_support_half_depth),
        "obstacle_target_exclusion_radius": float(args_cli.obstacle_target_exclusion_radius),
        "obstacle_cloud_max_points": int(args_cli.obstacle_cloud_max_points),
        "curobo_num_seeds": int(args_cli.curobo_num_seeds),
        "curobo_position_tol": float(args_cli.curobo_position_tol),
        "curobo_position_tol_overrides": dict(args_cli.curobo_position_tol_overrides),
        "curobo_rotation_tol": float(args_cli.curobo_rotation_tol),
        "curobo_accept_tolerance_ik_objects": list(args_cli.curobo_accept_tolerance_ik_objects),
        "relax_curobo_pregrasp_stage": bool(args_cli.relax_curobo_pregrasp_stage),
        "curobo_request_timeout": float(args_cli.curobo_request_timeout),
        "curobo_joint_execution_settle_tol": float(args_cli.curobo_joint_execution_settle_tol),
        "curobo_joint_execution_max_settle_steps": int(args_cli.curobo_joint_execution_max_settle_steps),
        "curobo_joint_execution_interp_step": float(args_cli.curobo_joint_execution_interp_step),
        "curobo_joint_execution_mode": args_cli.curobo_joint_execution_mode,
        "curobo_baseline_pregrasp_stage_steps": int(args_cli.curobo_baseline_pregrasp_stage_steps),
        "curobo_baseline_pregrasp_steps": int(args_cli.curobo_baseline_pregrasp_steps),
        "curobo_baseline_grasp_steps": int(args_cli.curobo_baseline_grasp_steps),
        "curobo_baseline_close_steps": int(args_cli.curobo_baseline_close_steps),
        "curobo_baseline_close_hold_steps": int(args_cli.curobo_baseline_close_hold_steps),
        "curobo_baseline_lift_steps": int(args_cli.curobo_baseline_lift_steps),
        "curobo_long_baseline_objects": list(args_cli.curobo_long_baseline_objects),
        "curobo_long_baseline_step_scale": float(args_cli.curobo_long_baseline_step_scale),
        "curobo_baseline_settle_objects": list(args_cli.curobo_baseline_settle_objects),
        "curobo_baseline_settle_stages": list(args_cli.curobo_baseline_settle_stages),
        "curobo_baseline_settle_ee_tol": float(args_cli.curobo_baseline_settle_ee_tol),
        "curobo_preclose_gate_objects": list(args_cli.curobo_preclose_gate_objects),
        "curobo_preclose_gate_joint_tol": float(args_cli.curobo_preclose_gate_joint_tol),
        "curobo_preclose_gate_ee_tol": float(args_cli.curobo_preclose_gate_ee_tol),
        "curobo_preclose_gate_max_settle_steps": int(args_cli.curobo_preclose_gate_max_settle_steps),
        "curobo_preclose_cartesian_correction_objects": list(
            args_cli.curobo_preclose_cartesian_correction_objects
        ),
        "curobo_preclose_cartesian_correction_steps": int(
            args_cli.curobo_preclose_cartesian_correction_steps
        ),
        "curobo_preclose_cartesian_correction_ee_tol": float(
            args_cli.curobo_preclose_cartesian_correction_ee_tol
        ),
        "curobo_preclose_cartesian_correction_orientation_dot": float(
            args_cli.curobo_preclose_cartesian_correction_orientation_dot
        ),
        "curobo_grasp_cartesian_correction_objects": list(
            args_cli.curobo_grasp_cartesian_correction_objects
        ),
        "curobo_grasp_cartesian_correction_steps": int(
            args_cli.curobo_grasp_cartesian_correction_steps
        ),
        "curobo_grasp_cartesian_correction_trigger_ee_tol": float(
            args_cli.curobo_grasp_cartesian_correction_trigger_ee_tol
        ),
        "curobo_grasp_cartesian_correction_ee_tol": float(
            args_cli.curobo_grasp_cartesian_correction_ee_tol
        ),
        "curobo_grasp_cartesian_correction_orientation_dot": float(
            args_cli.curobo_grasp_cartesian_correction_orientation_dot
        ),
        "curobo_grasp_cartesian_correction_fail_on_reject": bool(
            args_cli.curobo_grasp_cartesian_correction_fail_on_reject
        ),
        "cartesian_execution_objects": list(args_cli.cartesian_execution_objects),
        "cartesian_execution_stages": list(args_cli.cartesian_execution_stages),
        "curobo_joint_execution_action_warn": float(args_cli.curobo_joint_execution_action_warn),
        "curobo_joint_execution_abort_on_fail": bool(args_cli.curobo_joint_execution_abort_on_fail),
        "curobo_require_action_soft_limits": bool(args_cli.curobo_require_action_soft_limits),
        "curobo_soft_limit_exempt_objects": list(args_cli.curobo_soft_limit_exempt_objects),
        "curobo_soft_limit_tolerance_overrides": dict(args_cli.curobo_soft_limit_tolerance_overrides),
        "curobo_require_action_range": bool(args_cli.curobo_require_action_range),
        "curobo_action_limit": float(args_cli.curobo_action_limit),
        "curobo_baseline_action_prior": bool(args_cli.curobo_baseline_action_prior),
        "curobo_baseline_action_prior_ranking": bool(args_cli.curobo_baseline_action_prior_ranking),
        "curobo_staged_selection": bool(args_cli.curobo_staged_selection),
        "curobo_staged_final_ik_top_k": int(args_cli.curobo_staged_final_ik_top_k),
        "curobo_staged_full_ik_top_k": int(args_cli.curobo_staged_full_ik_top_k),
        "curobo_scene_collision": bool(args_cli.curobo_scene_collision),
        "curobo_obstacle_max_cuboids": int(args_cli.curobo_obstacle_max_cuboids),
        "curobo_obstacle_voxel_size": float(args_cli.curobo_obstacle_voxel_size),
        "curobo_obstacle_cuboid_padding": float(args_cli.curobo_obstacle_cuboid_padding),
        "curobo_obstacle_min_points_per_voxel": int(args_cli.curobo_obstacle_min_points_per_voxel),
        "curobo_obstacle_radius": float(args_cli.curobo_obstacle_radius),
        "curobo_obstacle_min_height_above_table": float(args_cli.curobo_obstacle_min_height_above_table),
        "pc_offset_search_objects": list(args_cli.pc_offset_search_objects),
        "pc_offset_step": float(args_cli.pc_offset_step),
        "pc_offset_max": float(args_cli.pc_offset_max),
        "pc_offset_collision_min_points": int(args_cli.pc_offset_collision_min_points),
        "pc_offset_require_both_fingers": bool(args_cli.pc_offset_require_both_fingers),
        "pc_offset_collision_clearance": float(args_cli.pc_offset_collision_clearance),
        "approach_pc_collision_filter": not bool(args_cli.disable_approach_pc_collision_filter),
        "approach_pc_collision_samples": int(args_cli.approach_pc_collision_samples),
        "approach_pc_collision_final_fraction": float(args_cli.approach_pc_collision_final_fraction),
        "approach_pc_collision_min_points": int(args_cli.approach_pc_collision_min_points),
        "approach_pc_collision_clearance": float(args_cli.approach_pc_collision_clearance),
        "straight_approach_pc_search": not bool(args_cli.disable_straight_approach_pc_search),
        "straight_approach_max_distance": float(args_cli.straight_approach_max_distance),
        "straight_approach_step": float(args_cli.straight_approach_step),
        "straight_approach_stage_extra": float(args_cli.straight_approach_stage_extra),
        "ik_physics_obstacle_filter": not bool(args_cli.disable_ik_physics_obstacle_filter),
        "ik_obstacle_motion_tol": float(args_cli.ik_obstacle_motion_tol),
        "object_transport_mode": args_cli.object_transport_mode,
        "early_close_failure_check": bool(args_cli.early_close_failure_check),
        "early_close_failure_aperture_threshold": float(args_cli.early_close_failure_aperture_threshold),
        "early_close_failure_target_tol": float(args_cli.early_close_failure_target_tol),
        "execute_after_each_object": bool(args_cli.execute_after_each_object),
        "max_object_attempts": int(args_cli.max_object_attempts),
        "object_max_attempt_overrides": dict(args_cli.object_max_attempt_overrides),
        "skip_laid_down_box": bool(args_cli.skip_laid_down_box),
        "box_laid_down_center_z_threshold": float(args_cli.box_laid_down_center_z_threshold),
        "stop_on_object_failure": bool(args_cli.stop_on_object_failure),
        "execution": execution,
        "records": records,
        "objects": request["source"]["objects"] if request is not None else None,
    }
    summary_path = OUTPUT_DIR / "pipeline_summary.json"
    summary_path.write_text(json_text(summary), encoding="utf-8")
    return summary_path


def save_object_summary(capture_dir: Path, object_name: str, record: dict | None = None) -> None:
    ee_overlay = capture_dir / "sam3" / f"ee_{object_name}_overlay.png"
    if not ee_overlay.exists():
        ee_overlay = capture_dir / "color_mask" / f"ee_{object_name}_overlay.png"
    video_overlay = OUTPUT_DIR / "video_rough" / f"{object_name}_target_overlay.png"
    if record is not None and record.get("video_target_overlay_path"):
        video_overlay = Path(record["video_target_overlay_path"])
    tiles = [
        ("video rough", video_overlay),
        ("ee rgb", capture_dir / "ee_rgb.png"),
        ("ee mask", ee_overlay),
        ("top grasps", capture_dir / "anygrasp" / "top_grasps_overlay.png"),
        ("cloud", capture_dir / "anygrasp" / "anygrasp_result.png"),
    ]
    tile_w, tile_h = 360, 270
    canvas = Image.new("RGB", (tile_w * len(tiles), tile_h), (18, 22, 30))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, path) in enumerate(tiles):
        x = idx * tile_w
        draw.rectangle((x, 0, x + tile_w, 24), fill=(0, 0, 0))
        draw.text((x + 8, 6), label, fill=(255, 255, 255))
        if path.exists():
            image = Image.open(path).convert("RGB")
            image.thumbnail((tile_w, tile_h - 28), Image.Resampling.LANCZOS)
            canvas.paste(image, (x + (tile_w - image.width) // 2, 26))
        else:
            draw.text((x + 12, 48), f"missing: {path.name}", fill=(255, 160, 160))
    canvas.save(capture_dir / "summary.png")


def attempt_suffix(attempt_number: int, max_attempts: int) -> str | None:
    if int(max_attempts) <= 1:
        return None
    return f"attempt_{int(attempt_number):02d}"


def attempt_dir(base: Path, idx: int, name: str, attempt_number: int, max_attempts: int) -> Path:
    root = base / f"{idx:02d}_{name}"
    suffix = attempt_suffix(attempt_number, max_attempts)
    return root / suffix if suffix is not None else root


def plan_object_attempt(
    env,
    obs,
    robot,
    ee_body_idx: int,
    ee_camera_calibration: dict,
    idx: int,
    name: str,
    attempt_number: int,
    max_attempts: int,
    video_recorder: VideoCameraRecorder | None,
) -> tuple[dict, dict, str | None]:
    record = run_video_rough_for_object(
        env,
        obs,
        name,
        attempt_dir(OUTPUT_DIR / "video_rough", idx, name, attempt_number, max_attempts),
    )
    if video_recorder is not None:
        video_recorder.add(obs, force=True)

    if args_cli.hover_target_source == "deterministic":
        rough_center = deterministic_object_poses(args_cli.seed)[name]["center_w"]
    else:
        rough_center = record["video_pose_estimate"]["center_world_median"]
    record["attempt_number"] = int(attempt_number)
    box_check = laid_down_box_check(name, record)
    if box_check.get("enabled"):
        record["box_laid_down_check"] = box_check
    if bool(box_check.get("laid_down")):
        capture_dir = attempt_dir(OUTPUT_DIR / "ee_refine", idx, name, attempt_number, max_attempts)
        record.update(
            {
                "capture_dir": str(capture_dir),
                "skip_object": True,
                "skip_status": "skipped_box_laid_down",
                "skip_reason": "box_object_rough_center_below_laid_down_threshold",
            }
        )
        print(
            "[WARN] "
            f"{name} attempt={attempt_number}/{max_attempts}: skipped laid-down box "
            f"center_z={box_check.get('center_z'):.4f} "
            f"threshold={box_check.get('threshold_z'):.4f}",
            flush=True,
        )
        return obs, record, "box_laid_down"
    hover_offset = args_cli.hover_object_offsets.get(name, {})
    rough_center_for_hover = [
        float(rough_center[0] + float(hover_offset.get("dx", 0.0))),
        float(rough_center[1] + float(hover_offset.get("dy", 0.0))),
        float(rough_center[2]),
    ]
    desired_camera_pos = [
        float(rough_center_for_hover[0]),
        float(rough_center_for_hover[1]),
        float(TABLE_TOP_Z + args_cli.hover_height + float(hover_offset.get("dz", 0.0))),
    ]
    capture_dir = attempt_dir(OUTPUT_DIR / "ee_refine", idx, name, attempt_number, max_attempts)
    primary_attempts = []
    selected_primary = None
    hover_modes = args_cli.dual_hover_modes if name in args_cli.dual_hover_objects else [None]
    for hover_attempt_index, hover_mode in enumerate(hover_modes, start=1):
        hover_attempt_dir = capture_dir
        if len(hover_modes) > 1:
            attempt_label = hover_mode or "adaptive"
            hover_attempt_dir = capture_dir / f"hover_{hover_attempt_index:02d}_{attempt_label}"
        obs, attempt = capture_primary_ee_view(
            env,
            obs,
            robot,
            ee_body_idx,
            ee_camera_calibration,
            name,
            hover_attempt_dir,
            desired_camera_pos,
            rough_center_for_hover,
            hover_offset,
            forced_hover_mode=hover_mode,
        )
        primary_attempts.append(
            {
                "hover_mode": hover_mode or args_cli.hover_mode,
                "capture_dir": str(attempt["capture_dir"]),
                "sam3_usable": bool(attempt["sam3_usable"]),
                "mask_count": attempt["detection"].get("mask_count"),
                "pose_status": attempt["ee_pose"].get("status", "ok"),
                "pose_error": attempt["ee_pose"].get("error"),
                "effective_mode": attempt["hover"].get("effective_mode"),
            }
        )
        if attempt["sam3_usable"]:
            selected_primary = attempt
            break
    if selected_primary is None:
        selected_primary = attempt

    capture_dir = Path(selected_primary["capture_dir"])
    hover = selected_primary["hover"]
    ee_camera = selected_primary["ee_camera"]
    ee_view_label = selected_primary["ee_view_label"]
    mask_path = selected_primary["mask_path"]
    detection = selected_primary["detection"]
    mask_source_used = selected_primary["mask_source_used"]
    mask_fallback_reason = selected_primary["mask_fallback_reason"]
    ee_depth = selected_primary["ee_depth"]
    ee_pose = selected_primary["ee_pose"]
    ee_views = [selected_primary["ee_view"]]
    extra_anygrasp_views = []
    anygrasp_fused_views = []
    if name in args_cli.anygrasp_video_view_objects:
        video_view = video_anygrasp_extra_view(record)
        if video_view is not None:
            extra_anygrasp_views.append(video_view)
            anygrasp_fused_views.append(
                {
                    "role": video_view["role"],
                    "rgb": str(video_view["rgb"]),
                    "depth_npy": str(video_view["depth_npy"]),
                    "mask": str(video_view["mask"]),
                    "camera_json": str(video_view["camera_json"]),
                }
            )
        else:
            anygrasp_fused_views.append(
                {
                    "role": "video_cam_rough",
                    "status": "missing_required_files",
                }
            )
    if name in args_cli.ee_multi_view_objects:
        second_view_dir = capture_dir / "view_02"
        obs, second_hover = move_to_topdown_camera(
            env,
            obs,
            robot,
            desired_camera_pos,
            ee_camera_calibration,
            object_name=name,
            look_at_target_w=rough_center_for_hover,
            camera_position_offset_w=args_cli.ee_second_view_offset,
        )
        second_hover["object_hover_offset"] = {
            "dx": float(hover_offset.get("dx", 0.0)),
            "dy": float(hover_offset.get("dy", 0.0)),
            "dz": float(hover_offset.get("dz", 0.0)),
        }
        second_view_dir.mkdir(parents=True, exist_ok=True)
        save_camera_frame(second_view_dir, "ee", obs["image"])
        save_camera_frame(
            second_view_dir,
            "video_hover",
            {
                "video_hover_rgb": obs["image"]["video_rgb"],
                "video_hover_depth": obs["image"]["video_depth"],
            },
        )
        second_camera = ee_camera_metadata_from_gripper(env, robot, ee_body_idx, ee_camera_calibration)
        write_camera_json(second_view_dir / "ee_camera.json", second_camera)
        second_label = (
            "eye-in-hand second look-at / ee_camera"
            if second_hover.get("effective_mode") == "look_at"
            else "eye-in-hand second top-down / ee_camera"
        )
        second_mask_dir = second_view_dir / ("sam3" if args_cli.ee_mask_source == "sam3" else "color_mask")
        second_mask_source = args_cli.ee_mask_source
        second_fallback_reason = None
        if args_cli.ee_mask_source == "sam3":
            second_mask_path, second_detection = run_sam3(
                second_view_dir / "ee_rgb.png",
                second_mask_dir,
                name,
                f"ee_{name}_view_02",
                second_label,
            )
        else:
            second_mask_path, second_detection = run_color_mask(
                second_view_dir / "ee_rgb.png",
                second_mask_dir,
                name,
                f"ee_{name}_view_02",
                second_label,
            )
        if args_cli.ee_mask_source == "sam3" and not second_detection.get("mask_count"):
            second_mask_path, second_detection = run_color_mask(
                second_view_dir / "ee_rgb.png",
                second_view_dir / "color_mask",
                name,
                f"ee_{name}_view_02",
                f"{second_label} color fallback",
            )
            second_mask_source = "color"
            second_fallback_reason = "sam3_empty_mask"
        second_depth = np.load(second_view_dir / "ee_depth.npy")
        second_pose = estimate_ee_pose_for_view(
            second_mask_path,
            second_depth,
            second_camera,
            second_mask_source,
            second_hover,
        )
        draw_target_overlay(
            second_view_dir / "ee_rgb.png",
            second_view_dir / f"{name}_ee_target_overlay.png",
            second_pose,
            f"{name} EE second {second_hover.get('effective_mode', 'adaptive')} target",
        )
        ee_views.append(
            {
                "view_index": 2,
                "role": "extra",
                "capture_dir": str(second_view_dir),
                "hover": second_hover,
                "camera": second_camera,
                "mask_source_used": second_mask_source,
                "mask_fallback_reason": second_fallback_reason,
                "sam3": {
                    "prompt": second_detection.get("prompt"),
                    "source": second_detection.get("source", args_cli.ee_mask_source),
                    "mask_count": second_detection.get("mask_count"),
                    "best_index": second_detection.get("best_index"),
                    "areas_px": second_detection.get("areas_px"),
                    "scores": second_detection.get("scores"),
                    "mask_path": str(second_mask_path),
                },
                "pose_estimate": second_pose,
            }
        )
        extra_anygrasp_views.append(
            {
                "role": "ee_camera_view_02",
                "rgb": second_view_dir / "ee_rgb.png",
                "depth_npy": second_view_dir / "ee_depth.npy",
                "mask": second_mask_path,
                "camera_json": second_view_dir / "ee_camera.json",
            }
        )
        anygrasp_fused_views.append(
            {
                "role": "ee_camera_view_02",
                "rgb": str(second_view_dir / "ee_rgb.png"),
                "depth_npy": str(second_view_dir / "ee_depth.npy"),
                "mask": str(second_mask_path),
                "camera_json": str(second_view_dir / "ee_camera.json"),
            }
        )

    object_center_for_anygrasp = pose_center_w(ee_pose) or rough_center_for_hover
    heuristic_attempt_seed = None
    if bool(args_cli.heuristic_vary_seed_by_attempt):
        heuristic_attempt_seed = int(args_cli.seed) + (int(attempt_number) - 1) * int(args_cli.heuristic_attempt_seed_stride)
    pose_path, anygrasp_result = run_anygrasp(
        capture_dir,
        mask_path,
        capture_dir / "ee_camera.json",
        name,
        extra_views=extra_anygrasp_views,
        object_center_w=object_center_for_anygrasp,
        heuristic_seed=heuristic_attempt_seed,
    )
    sam3_anygrasp_result = None
    anygrasp_status = (anygrasp_result.get("anygrasp") or {}).get("status")
    if args_cli.ee_mask_source == "sam3" and mask_source_used == "sam3" and anygrasp_status != "ok":
        sam3_anygrasp_result = anygrasp_result
        mask_path, detection = run_color_mask(
            capture_dir / "ee_rgb.png",
            capture_dir / "color_mask",
            name,
            f"ee_{name}",
            f"{ee_view_label} color fallback",
        )
        mask_source_used = "color"
        mask_fallback_reason = f"sam3_anygrasp_{anygrasp_status}"
        ee_pose = estimate_ee_pose_for_view(mask_path, ee_depth, ee_camera, mask_source_used, hover)
        draw_target_overlay(
            capture_dir / "ee_rgb.png",
            capture_dir / f"{name}_ee_target_overlay.png",
            ee_pose,
            f"{name} EE {hover.get('effective_mode', 'adaptive')} target",
        )
        ee_views[0]["mask_source_used"] = mask_source_used
        ee_views[0]["mask_fallback_reason"] = mask_fallback_reason
        ee_views[0]["sam3"] = {
            "prompt": detection.get("prompt"),
            "source": detection.get("source", args_cli.ee_mask_source),
            "mask_count": detection.get("mask_count"),
            "best_index": detection.get("best_index"),
            "areas_px": detection.get("areas_px"),
            "scores": detection.get("scores"),
            "mask_path": str(mask_path),
        }
        ee_views[0]["pose_estimate"] = ee_pose
        pose_path, anygrasp_result = run_anygrasp(
            capture_dir,
            mask_path,
            capture_dir / "ee_camera.json",
            name,
            extra_views=extra_anygrasp_views,
            object_center_w=object_center_for_anygrasp,
            heuristic_seed=heuristic_attempt_seed,
        )

    record.update(
        {
            "attempt_number": int(attempt_number),
            "heuristic_attempt_seed": heuristic_attempt_seed,
            "hover": hover,
            "primary_hover_attempts": primary_attempts,
            "hover_target_source": args_cli.hover_target_source,
            "capture_dir": str(capture_dir),
            "ee_camera": ee_camera,
            "ee_views": ee_views,
            "ee_extra_view_count": len(extra_anygrasp_views),
            "anygrasp_fused_extra_views": anygrasp_fused_views,
            "ee_sam3": {
                "prompt": detection.get("prompt"),
                "source": detection.get("source", args_cli.ee_mask_source),
                "mask_count": detection.get("mask_count"),
                "best_index": detection.get("best_index"),
                "areas_px": detection.get("areas_px"),
                "scores": detection.get("scores"),
                "mask_path": str(mask_path),
            },
            "ee_mask_source_used": mask_source_used,
            "ee_mask_fallback_reason": mask_fallback_reason,
            "ee_pose_estimate": ee_pose,
            "sam3_anygrasp_result": sam3_anygrasp_result,
            "anygrasp_result": anygrasp_result,
            "anygrasp_result_path": capture_dir / "anygrasp" / "anygrasp_result.json",
            "final_grasp_pose_path": pose_path,
        }
    )
    save_object_summary(capture_dir, name, record)
    final_pose = load_json(pose_path) if anygrasp_status == "ok" and pose_path.exists() else {}
    print(
        "[INFO] "
        f"{name} attempt={attempt_number}/{max_attempts}: "
        f"hover_error={hover['position_error_m']:.4f} "
        f"anygrasp={anygrasp_status} "
        f"score={final_pose.get('score')} "
        f"grasp={np.round(final_pose.get('translation', []), 4).tolist()}",
        flush=True,
    )
    return obs, record, anygrasp_status


def execute_interleaved_attempt(
    env,
    obs,
    robot,
    idx: int,
    name: str,
    attempt_number: int,
    max_attempts: int,
    record: dict,
    anygrasp_status: str | None,
    video_recorder: VideoCameraRecorder | None,
) -> tuple[dict, dict, dict | None, Path | None]:
    suffix = attempt_suffix(attempt_number, max_attempts)
    execution_record = {
        "name": name,
        "attempt": int(attempt_number),
        "status": "pending",
    }
    if bool(record.get("skip_object")):
        execution_record.update(
            {
                "status": record.get("skip_status", "skipped_object"),
                "anygrasp_status": anygrasp_status,
                "target_in_basket": False,
                "skip_reason": record.get("skip_reason"),
                "box_laid_down_check": record.get("box_laid_down_check"),
            }
        )
        return obs, execution_record, None, None
    if anygrasp_status != "ok":
        execution_record.update(
            {
                "status": "skipped_anygrasp_failed",
                "anygrasp_status": anygrasp_status,
                "target_in_basket": False,
            }
        )
        return obs, execution_record, None, None

    current_records = {name: record}
    object_poses = object_poses_for_request(current_records)
    obs = select_ik_feasible_grasps(env, obs, robot, object_poses, current_records)
    selection = record.get("grasp_selection") or {}
    if selection and not bool(selection.get("selected_ok")):
        execution_record.update(
            {
                "status": "skipped_no_safe_grasp_candidate",
                "anygrasp_status": anygrasp_status,
                "target_in_basket": False,
                "selected_rank": selection.get("selected_rank"),
                "selected_score": selection.get("selected_score"),
                "selection_path": selection.get("selection_path"),
                "skip_reason": (
                    "No AnyGrasp candidate passed point-cloud offset, early approach collision, "
                    "and final-pose IK filters."
                ),
            }
        )
        return obs, execution_record, None, None
    grasp_records = build_grasp_records(current_records)
    single_args = copy.copy(args_cli)
    single_args.input = OUTPUT_DIR
    single_args.object_order = [name]
    request = build_request(single_args, OUTPUT_DIR, object_poses, grasp_records)
    request["source"]["type"] = "full_task_anygrasp_video_to_ee_interleaved"
    request["source"]["attempt_number"] = int(attempt_number)
    request["source"]["grasp_pose_source"] = (
        "current_scene_video_cam_rough_center_then_ee_camera_sam3_anygrasp"
    )
    request["source"]["note"] = (
        "This request was generated and executed immediately before planning the next object, "
        "so later video/SAM3/AnyGrasp passes see the updated scene. When retries are enabled, "
        "a failed target is re-detected from the current scene before the next attempt."
    )
    attach_curobo_joint_targets_to_request(request, current_records)
    apply_cartesian_execution_overrides_to_request(request)
    request_dir = OUTPUT_DIR / "requests"
    request_dir.mkdir(parents=True, exist_ok=True)
    request_name = f"{idx:02d}_{name}_motion_request.json" if suffix is None else f"{idx:02d}_{name}_{suffix}_motion_request.json"
    request_path = request_dir / request_name
    request_path.write_text(json_text(request), encoding="utf-8")
    exec_dir = attempt_dir(OUTPUT_DIR / "execution", idx, name, attempt_number, max_attempts)
    if video_recorder is not None:
        video_recorder.add(obs, force=True)
    obs, motion_result = execute_motion_request_in_env(
        env,
        obs,
        robot,
        request,
        request_path,
        exec_dir,
        video_recorder=video_recorder,
    )
    record["motion_request_path"] = str(request_path)
    record["motion_result_path"] = str(exec_dir / "motion_result.json")
    record["motion_result"] = motion_result
    diagnostics = motion_target_diagnostics(motion_result, name)
    record["motion_diagnostics"] = diagnostics
    execution_record.update(
        {
            "status": "executed",
            "request": str(request_path),
            "result": str(exec_dir / "motion_result.json"),
            "ok": bool(motion_result.get("ok")),
            "objects_in_basket": motion_result.get("task_e_objects", {}).get("count_in_basket"),
            "target_in_basket": bool(diagnostics.get("target_in_basket")),
            "min_ee_object_distance_m": diagnostics.get("min_ee_object_distance_m"),
            "closest_grasp_or_close_waypoint": diagnostics.get("closest_grasp_or_close_waypoint"),
            "final_object": diagnostics.get("final_object"),
        }
    )
    return obs, execution_record, request, request_path


def main() -> None:
    stale_error = OUTPUT_DIR / "pipeline_error.json"
    if stale_error.exists():
        stale_error.unlink()

    env_cfg = parse_env_cfg(
        "ATEC-TaskE-Piper",
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    apply_actuator_mode(env_cfg, args_cli.actuator_mode)
    env_cfg.seed = args_cli.seed
    env = gym.make("ATEC-TaskE-Piper", cfg=env_cfg)
    obs, _ = env.reset()
    robot = env.unwrapped.scene.articulations["robot"]
    ee_body_idx = find_single_body(robot, "gripper_base")
    ee_camera_calibration = calibrate_ee_camera(env, robot, ee_body_idx)

    records: dict[str, dict] = {}
    execution_records: list[dict] = []
    video_recorder = None
    if args_cli.execute_after_each_object:
        execution_root = OUTPUT_DIR / "execution"
        execution_root.mkdir(parents=True, exist_ok=True)
        if args_cli.record_video_cam:
            video_recorder = VideoCameraRecorder(
                execution_root / "video_cam.mp4",
                fps=args_cli.video_fps,
                every_n_steps=args_cli.video_every_n_steps,
            )
            video_recorder.add(obs, force=True)
    else:
        video_dir = OUTPUT_DIR / "video_rough"
        for name in args_cli.object_order:
            records[name] = run_video_rough_for_object(env, obs, name, video_dir)

    for idx, name in enumerate(args_cli.object_order, start=1):
        if args_cli.execute_after_each_object:
            max_attempts = max(1, int(args_cli.object_max_attempt_overrides.get(name, args_cli.max_object_attempts)))
            attempt_summaries = []
            last_request = None
            last_request_path = None
            object_succeeded = False
            for attempt_number in range(1, max_attempts + 1):
                obs, attempt_record, anygrasp_status = plan_object_attempt(
                    env,
                    obs,
                    robot,
                    ee_body_idx,
                    ee_camera_calibration,
                    idx,
                    name,
                    attempt_number,
                    max_attempts,
                    video_recorder,
                )
                obs, execution_record, last_request, last_request_path = execute_interleaved_attempt(
                    env,
                    obs,
                    robot,
                    idx,
                    name,
                    attempt_number,
                    max_attempts,
                    attempt_record,
                    anygrasp_status,
                    video_recorder,
                )
                execution_records.append(execution_record)
                attempt_summaries.append(execution_record)
                attempt_record["attempts"] = list(attempt_summaries)
                records[name] = attempt_record
                write_pipeline_summary(
                    records,
                    request=last_request,
                    request_path=last_request_path,
                    execution={
                        "mode": "execute_after_each_object",
                        "objects": execution_records,
                        "current_task_e_objects": collect_task_e_object_summary(env),
                    },
                )
                if execution_record.get("target_in_basket"):
                    object_succeeded = True
                    break
                if execution_record.get("status") == "skipped_box_laid_down":
                    break
                if attempt_number < max_attempts:
                    distance = execution_record.get("min_ee_object_distance_m")
                    distance_text = "unknown" if distance is None else f"{distance:.4f} m"
                    print(
                        "[WARN] "
                        f"{name} attempt {attempt_number}/{max_attempts} did not finish in basket; "
                        f"closest gripper_base-object distance={distance_text}. Retrying detection and grasp.",
                        flush=True,
                    )
            if not object_succeeded and args_cli.stop_on_object_failure:
                print(
                    "[WARN] "
                    f"{name} did not reach the basket after {max_attempts} attempt(s); "
                    "stopping remaining object order due to --stop-on-object-failure.",
                    flush=True,
                )
                write_pipeline_summary(
                    records,
                    request=last_request,
                    request_path=last_request_path,
                    execution={
                        "mode": "execute_after_each_object",
                        "objects": execution_records,
                        "current_task_e_objects": collect_task_e_object_summary(env),
                        "stop_reason": f"{name}_failed_after_{max_attempts}_attempts",
                    },
                )
                break
            continue

        if args_cli.execute_after_each_object:
            records[name] = run_video_rough_for_object(
                env,
                obs,
                name,
                OUTPUT_DIR / "video_rough" / f"{idx:02d}_{name}",
            )
            if video_recorder is not None:
                video_recorder.add(obs, force=True)
        if args_cli.hover_target_source == "deterministic":
            rough_center = deterministic_object_poses(args_cli.seed)[name]["center_w"]
        else:
            rough_center = records[name]["video_pose_estimate"]["center_world_median"]
        hover_offset = args_cli.hover_object_offsets.get(name, {})
        rough_center_for_hover = [
            float(rough_center[0] + float(hover_offset.get("dx", 0.0))),
            float(rough_center[1] + float(hover_offset.get("dy", 0.0))),
            float(rough_center[2]),
        ]
        desired_camera_pos = [
            float(rough_center_for_hover[0]),
            float(rough_center_for_hover[1]),
            float(TABLE_TOP_Z + args_cli.hover_height + float(hover_offset.get("dz", 0.0))),
        ]
        capture_dir = OUTPUT_DIR / "ee_refine" / f"{idx:02d}_{name}"
        primary_attempts = []
        selected_primary = None
        hover_modes = args_cli.dual_hover_modes if name in args_cli.dual_hover_objects else [None]
        for attempt_index, hover_mode in enumerate(hover_modes, start=1):
            attempt_dir = capture_dir
            if len(hover_modes) > 1:
                attempt_label = hover_mode or "adaptive"
                attempt_dir = capture_dir / f"hover_{attempt_index:02d}_{attempt_label}"
            obs, attempt = capture_primary_ee_view(
                env,
                obs,
                robot,
                ee_body_idx,
                ee_camera_calibration,
                name,
                attempt_dir,
                desired_camera_pos,
                rough_center_for_hover,
                hover_offset,
                forced_hover_mode=hover_mode,
            )
            primary_attempts.append(
                {
                    "hover_mode": hover_mode or args_cli.hover_mode,
                    "capture_dir": str(attempt["capture_dir"]),
                    "sam3_usable": bool(attempt["sam3_usable"]),
                    "mask_count": attempt["detection"].get("mask_count"),
                    "pose_status": attempt["ee_pose"].get("status", "ok"),
                    "pose_error": attempt["ee_pose"].get("error"),
                    "effective_mode": attempt["hover"].get("effective_mode"),
                }
            )
            if attempt["sam3_usable"]:
                selected_primary = attempt
                break
        if selected_primary is None:
            selected_primary = attempt

        capture_dir = Path(selected_primary["capture_dir"])
        hover = selected_primary["hover"]
        ee_camera = selected_primary["ee_camera"]
        ee_view_label = selected_primary["ee_view_label"]
        mask_path = selected_primary["mask_path"]
        detection = selected_primary["detection"]
        mask_source_used = selected_primary["mask_source_used"]
        mask_fallback_reason = selected_primary["mask_fallback_reason"]
        ee_depth = selected_primary["ee_depth"]
        ee_pose = selected_primary["ee_pose"]
        ee_views = [selected_primary["ee_view"]]
        extra_anygrasp_views = []
        if name in args_cli.ee_multi_view_objects:
            second_view_dir = capture_dir / "view_02"
            obs, second_hover = move_to_topdown_camera(
                env,
                obs,
                robot,
                desired_camera_pos,
                ee_camera_calibration,
                object_name=name,
                look_at_target_w=rough_center_for_hover,
                camera_position_offset_w=args_cli.ee_second_view_offset,
            )
            second_hover["object_hover_offset"] = {
                "dx": float(hover_offset.get("dx", 0.0)),
                "dy": float(hover_offset.get("dy", 0.0)),
                "dz": float(hover_offset.get("dz", 0.0)),
            }
            second_view_dir.mkdir(parents=True, exist_ok=True)
            save_camera_frame(second_view_dir, "ee", obs["image"])
            save_camera_frame(
                second_view_dir,
                "video_hover",
                {
                    "video_hover_rgb": obs["image"]["video_rgb"],
                    "video_hover_depth": obs["image"]["video_depth"],
                },
            )
            second_camera = ee_camera_metadata_from_gripper(env, robot, ee_body_idx, ee_camera_calibration)
            write_camera_json(second_view_dir / "ee_camera.json", second_camera)
            second_label = (
                "eye-in-hand second look-at / ee_camera"
                if second_hover.get("effective_mode") == "look_at"
                else "eye-in-hand second top-down / ee_camera"
            )
            second_mask_dir = second_view_dir / ("sam3" if args_cli.ee_mask_source == "sam3" else "color_mask")
            second_mask_source = args_cli.ee_mask_source
            second_fallback_reason = None
            if args_cli.ee_mask_source == "sam3":
                second_mask_path, second_detection = run_sam3(
                    second_view_dir / "ee_rgb.png",
                    second_mask_dir,
                    name,
                    f"ee_{name}_view_02",
                    second_label,
                )
            else:
                second_mask_path, second_detection = run_color_mask(
                    second_view_dir / "ee_rgb.png",
                    second_mask_dir,
                    name,
                    f"ee_{name}_view_02",
                    second_label,
                )
            if args_cli.ee_mask_source == "sam3" and not second_detection.get("mask_count"):
                second_mask_path, second_detection = run_color_mask(
                    second_view_dir / "ee_rgb.png",
                    second_view_dir / "color_mask",
                    name,
                    f"ee_{name}_view_02",
                    f"{second_label} color fallback",
                )
                second_mask_source = "color"
                second_fallback_reason = "sam3_empty_mask"
            second_depth = np.load(second_view_dir / "ee_depth.npy")
            second_pose = estimate_ee_pose_for_view(
                second_mask_path,
                second_depth,
                second_camera,
                second_mask_source,
                second_hover,
            )
            draw_target_overlay(
                second_view_dir / "ee_rgb.png",
                second_view_dir / f"{name}_ee_target_overlay.png",
                second_pose,
                f"{name} EE second {second_hover.get('effective_mode', 'adaptive')} target",
            )
            ee_views.append(
                {
                    "view_index": 2,
                    "role": "extra",
                    "capture_dir": str(second_view_dir),
                    "hover": second_hover,
                    "camera": second_camera,
                    "mask_source_used": second_mask_source,
                    "mask_fallback_reason": second_fallback_reason,
                    "sam3": {
                        "prompt": second_detection.get("prompt"),
                        "source": second_detection.get("source", args_cli.ee_mask_source),
                        "mask_count": second_detection.get("mask_count"),
                        "best_index": second_detection.get("best_index"),
                        "areas_px": second_detection.get("areas_px"),
                        "scores": second_detection.get("scores"),
                        "mask_path": str(second_mask_path),
                    },
                    "pose_estimate": second_pose,
                }
            )
            extra_anygrasp_views.append(
                {
                    "rgb": second_view_dir / "ee_rgb.png",
                    "depth_npy": second_view_dir / "ee_depth.npy",
                    "mask": second_mask_path,
                    "camera_json": second_view_dir / "ee_camera.json",
                }
            )

        pose_path, anygrasp_result = run_anygrasp(
            capture_dir,
            mask_path,
            capture_dir / "ee_camera.json",
            name,
            extra_views=extra_anygrasp_views,
        )
        sam3_anygrasp_result = None
        anygrasp_status = (anygrasp_result.get("anygrasp") or {}).get("status")
        if args_cli.ee_mask_source == "sam3" and mask_source_used == "sam3" and anygrasp_status != "ok":
            sam3_anygrasp_result = anygrasp_result
            mask_path, detection = run_color_mask(
                capture_dir / "ee_rgb.png",
                capture_dir / "color_mask",
                name,
                f"ee_{name}",
                f"{ee_view_label} color fallback",
            )
            mask_source_used = "color"
            mask_fallback_reason = f"sam3_anygrasp_{anygrasp_status}"
            ee_pose = estimate_ee_pose_for_view(mask_path, ee_depth, ee_camera, mask_source_used, hover)
            draw_target_overlay(
                capture_dir / "ee_rgb.png",
                capture_dir / f"{name}_ee_target_overlay.png",
                ee_pose,
                f"{name} EE {hover.get('effective_mode', 'adaptive')} target",
            )
            ee_views[0]["mask_source_used"] = mask_source_used
            ee_views[0]["mask_fallback_reason"] = mask_fallback_reason
            ee_views[0]["sam3"] = {
                "prompt": detection.get("prompt"),
                "source": detection.get("source", args_cli.ee_mask_source),
                "mask_count": detection.get("mask_count"),
                "best_index": detection.get("best_index"),
                "areas_px": detection.get("areas_px"),
                "scores": detection.get("scores"),
                "mask_path": str(mask_path),
            }
            ee_views[0]["pose_estimate"] = ee_pose
            pose_path, anygrasp_result = run_anygrasp(
                capture_dir,
                mask_path,
                capture_dir / "ee_camera.json",
                name,
                extra_views=extra_anygrasp_views,
            )
        records[name].update(
            {
                "hover": hover,
                "primary_hover_attempts": primary_attempts,
                "hover_target_source": args_cli.hover_target_source,
                "capture_dir": str(capture_dir),
                "ee_camera": ee_camera,
                "ee_views": ee_views,
                "ee_extra_view_count": len(extra_anygrasp_views),
                "ee_sam3": {
                    "prompt": detection.get("prompt"),
                    "source": detection.get("source", args_cli.ee_mask_source),
                    "mask_count": detection.get("mask_count"),
                    "best_index": detection.get("best_index"),
                    "areas_px": detection.get("areas_px"),
                    "scores": detection.get("scores"),
                    "mask_path": str(mask_path),
                },
                "ee_mask_source_used": mask_source_used,
                "ee_mask_fallback_reason": mask_fallback_reason,
                "ee_pose_estimate": ee_pose,
                "sam3_anygrasp_result": sam3_anygrasp_result,
                "anygrasp_result": anygrasp_result,
                "anygrasp_result_path": capture_dir / "anygrasp" / "anygrasp_result.json",
                "final_grasp_pose_path": pose_path,
            }
        )
        save_object_summary(capture_dir, name, records[name])
        anygrasp_status = (anygrasp_result.get("anygrasp") or {}).get("status")
        final_pose = load_json(pose_path) if anygrasp_status == "ok" and pose_path.exists() else {}
        print(
            "[INFO] "
            f"{name}: hover_error={hover['position_error_m']:.4f} "
            f"anygrasp={anygrasp_status} "
            f"score={final_pose.get('score')} "
            f"grasp={np.round(final_pose.get('translation', []), 4).tolist()}",
            flush=True,
        )

        if args_cli.execute_after_each_object:
            current_records = {name: records[name]}
            if anygrasp_status != "ok":
                execution_records.append(
                    {
                        "name": name,
                        "status": "skipped_anygrasp_failed",
                        "anygrasp_status": anygrasp_status,
                    }
                )
                write_pipeline_summary(
                    records,
                    execution={
                        "mode": "execute_after_each_object",
                        "objects": execution_records,
                        "current_task_e_objects": collect_task_e_object_summary(env),
                    },
                )
                continue

            object_poses = object_poses_for_request(current_records)
            obs = select_ik_feasible_grasps(env, obs, robot, object_poses, current_records)
            selection = records[name].get("grasp_selection") or {}
            if selection and not bool(selection.get("selected_ok")):
                execution_records.append(
                    {
                        "name": name,
                        "status": "skipped_no_safe_grasp_candidate",
                        "anygrasp_status": anygrasp_status,
                        "target_in_basket": False,
                        "selected_rank": selection.get("selected_rank"),
                        "selected_score": selection.get("selected_score"),
                        "selection_path": selection.get("selection_path"),
                        "skip_reason": (
                            "No AnyGrasp candidate passed point-cloud offset, early approach collision, "
                            "and final-pose IK filters."
                        ),
                    }
                )
                write_pipeline_summary(
                    records,
                    execution={
                        "mode": "execute_after_each_object",
                        "objects": execution_records,
                        "current_task_e_objects": collect_task_e_object_summary(env),
                    },
                )
                continue
            grasp_records = build_grasp_records(current_records)
            single_args = copy.copy(args_cli)
            single_args.input = OUTPUT_DIR
            single_args.object_order = [name]
            request = build_request(single_args, OUTPUT_DIR, object_poses, grasp_records)
            request["source"]["type"] = "full_task_anygrasp_video_to_ee_interleaved"
            request["source"]["grasp_pose_source"] = (
                "current_scene_video_cam_rough_center_then_ee_camera_sam3_anygrasp"
            )
            request["source"]["note"] = (
                "This request was generated and executed immediately before planning the next object, "
                "so later video/SAM3/AnyGrasp passes see the updated scene."
            )
            attach_curobo_joint_targets_to_request(request, current_records)
            apply_cartesian_execution_overrides_to_request(request)
            request_dir = OUTPUT_DIR / "requests"
            request_dir.mkdir(parents=True, exist_ok=True)
            request_path = request_dir / f"{idx:02d}_{name}_motion_request.json"
            request_path.write_text(json_text(request), encoding="utf-8")
            exec_dir = OUTPUT_DIR / "execution" / f"{idx:02d}_{name}"
            if video_recorder is not None:
                video_recorder.add(obs, force=True)
            obs, motion_result = execute_motion_request_in_env(
                env,
                obs,
                robot,
                request,
                request_path,
                exec_dir,
                video_recorder=video_recorder,
            )
            records[name]["motion_request_path"] = str(request_path)
            records[name]["motion_result_path"] = str(exec_dir / "motion_result.json")
            records[name]["motion_result"] = motion_result
            execution_records.append(
                {
                    "name": name,
                    "request": str(request_path),
                    "result": str(exec_dir / "motion_result.json"),
                    "ok": bool(motion_result.get("ok")),
                    "objects_in_basket": motion_result.get("task_e_objects", {}).get("count_in_basket"),
                    "target_in_basket": next(
                        (
                            item.get("in_basket")
                            for item in motion_result.get("task_e_objects", {}).get("objects", {}).values()
                            if item.get("label") == OBJECTS[name]["label"]
                        ),
                        None,
                    ),
                }
            )
            write_pipeline_summary(
                records,
                request=request,
                request_path=request_path,
                execution={
                    "mode": "execute_after_each_object",
                    "objects": execution_records,
                    "current_task_e_objects": collect_task_e_object_summary(env),
                },
            )

    if args_cli.execute_after_each_object:
        video_artifact = video_recorder.close() if video_recorder is not None else None
        final_objects = collect_task_e_object_summary(env)
        execution_summary = {
            "mode": "execute_after_each_object",
            "objects": execution_records,
            "video_cam": video_artifact,
            "final_task_e_objects": final_objects,
        }
        summary_path = write_pipeline_summary(records, execution=execution_summary)
        env.close()
        print(f"[INFO] Saved interleaved execution summary: {summary_path}")
        if video_artifact is not None:
            print(f"[INFO] Saved interleaved video: {video_artifact['file']}")
        simulation_app.close()
        return

    failed_anygrasp = [
        name
        for name, record in records.items()
        if (record.get("anygrasp_result", {}).get("anygrasp") or {}).get("status") != "ok"
    ]
    if failed_anygrasp:
        summary_path = write_pipeline_summary(records)
        env.close()
        print(f"[WARN] Skipping motion request because AnyGrasp failed for: {failed_anygrasp}")
        print(f"[INFO] Saved summary: {summary_path}")
        simulation_app.close()
        return

    object_poses = object_poses_for_request(records)
    obs = select_ik_feasible_grasps(env, obs, robot, object_poses, records)
    unsafe_selected = [
        {
            "name": name,
            "status": "skipped_no_safe_grasp_candidate",
            "selected_rank": (record.get("grasp_selection") or {}).get("selected_rank"),
            "selected_score": (record.get("grasp_selection") or {}).get("selected_score"),
            "selection_path": (record.get("grasp_selection") or {}).get("selection_path"),
            "target_in_basket": False,
            "skip_reason": (
                "No AnyGrasp candidate passed point-cloud offset, early approach collision, "
                "and final-pose IK filters."
            ),
        }
        for name, record in records.items()
        if (record.get("grasp_selection") or {}) and not bool((record.get("grasp_selection") or {}).get("selected_ok"))
    ]
    if unsafe_selected:
        summary_path = write_pipeline_summary(
            records,
            execution={
                "mode": "request_generation_skipped_no_safe_grasp_candidate",
                "objects": unsafe_selected,
                "current_task_e_objects": collect_task_e_object_summary(env),
            },
        )
        env.close()
        print(f"[WARN] Skipping motion request because no safe grasp candidate was selected: {unsafe_selected}")
        print(f"[INFO] Saved summary: {summary_path}")
        simulation_app.close()
        return
    env.close()
    grasp_records = build_grasp_records(records)
    args_cli.input = OUTPUT_DIR
    request = build_request(args_cli, OUTPUT_DIR, object_poses, grasp_records)
    request["source"]["type"] = "full_task_anygrasp_video_to_ee_adaptive"
    request["source"]["grasp_pose_source"] = "video_cam_rough_center_then_ee_camera_adaptive_sam3_anygrasp"
    request["source"]["note"] = (
        "External video camera is used only for rough object localization. "
        "For each object, the EE camera is moved to an adaptive reachable view, optionally "
        "pitched/yawed from the initial camera position toward the object, then EE RGB-D + "
        "SAM3 mask are passed to AnyGrasp."
    )
    attach_curobo_joint_targets_to_request(request, records)
    apply_cartesian_execution_overrides_to_request(request)
    request_path = OUTPUT_DIR / "motion_request.json"
    request_path.write_text(json_text(request), encoding="utf-8")

    summary_path = write_pipeline_summary(records, request=request, request_path=request_path)
    print(f"[INFO] Saved full EE-refined AnyGrasp request: {request_path}")
    print(f"[INFO] Saved summary: {summary_path}")
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write_failure("pipeline_error.json", exc)
        try:
            simulation_app.close()
        except Exception:
            pass
        raise
