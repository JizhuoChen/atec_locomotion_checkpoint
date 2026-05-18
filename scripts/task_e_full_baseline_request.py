#!/usr/bin/env python3
"""Create a full Task E pseudo-grasp pick-place motion request."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUEST_SCHEMA = "atec.task_e.motion_request.v1"

# Kept in sync with source/atec_rl_lab/atec_rl_lab/tasks/task_e/env_cfg.py
TABLE_CENTER_X = 1.00
TABLE_CENTER_Y = 0.00
TABLE_CENTER_Z = 0.00
TABLE_SCALE = 0.01
TABLE_DIMS_AT_0P008 = (0.6468062441005529, 0.9084968693231588, 0.6613141183247961)
TABLE_DIMS = tuple(dim * (TABLE_SCALE / 0.008) for dim in TABLE_DIMS_AT_0P008)
TABLE_HALF_X = TABLE_DIMS[0] * 0.5
TABLE_TOP_Z = TABLE_CENTER_Z + TABLE_DIMS[2]
BASKET_CENTER_X = TABLE_CENTER_X + 0.08
BASKET_CENTER_Y = TABLE_CENTER_Y - 0.30
BASKET_SUCCESS_HALF_X = 0.20
BASKET_SUCCESS_HALF_Y = 0.11

RETRACT_POS_X = TABLE_CENTER_X + TABLE_HALF_X - 0.05
RETRACT_POS_Y = TABLE_CENTER_Y
CARRY_Z = TABLE_TOP_Z + 0.40
PLACE_HEIGHT = TABLE_TOP_Z + 0.15
DEFAULT_QUAT_WXYZ = [0.0, 1.0, 0.0, 0.0]
GRIPPER_OPEN = [0.035, -0.035]
GRIPPER_CLOSE = [-0.015, 0.015]

OBJECTS = {
    "box_object": {
        "object_key": "object_1",
        "label": "yellow_and_white_box",
        "spawn_band_y": (TABLE_CENTER_Y + 0.25, TABLE_CENTER_Y + 0.29),
        "half_extents_xy": (0.050, 0.044),
        "grasp_z_offset": 0.09,
        "place_offset_xy": (-0.07, 0.00),
        "object_quat_wxyz": [0.0, 0.707, 0.0, 0.707],
    },
    "mustard_bottle": {
        "object_key": "object_2",
        "label": "mustard_bottle",
        "spawn_band_y": (TABLE_CENTER_Y + 0.14, TABLE_CENTER_Y + 0.20),
        "half_extents_xy": (0.050, 0.030),
        "grasp_z_offset": 0.09,
        "place_offset_xy": (0.06, 0.00),
        "object_quat_wxyz": [0.0, 0.0, -0.707, 0.707],
    },
    "banana": {
        "object_key": "object_3",
        "label": "banana",
        "spawn_band_y": (TABLE_CENTER_Y + 0.03, TABLE_CENTER_Y + 0.09),
        "half_extents_xy": (0.100, 0.040),
        "grasp_z_offset": 0.09,
        "place_offset_xy": (0.00, 0.03),
        "object_quat_wxyz": [0.0, 0.0, -0.707, 0.707],
    },
}
ENV_OBJECT_ORDER = ("box_object", "mustard_bottle", "banana")
DEFAULT_PICK_ORDER = ("banana", "mustard_bottle", "box_object")
OBJ_BBOX_MARGIN = 0.015


def parse_object_order(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in OBJECTS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown objects: {unknown}. Valid: {sorted(OBJECTS)}")
    return names


def parse_grasp_tuning(value: str) -> dict[str, dict[str, float]]:
    """Parse object:dx,dy,dz_override entries separated by semicolons."""
    tuning: dict[str, dict[str, float]] = {}
    if not value:
        return tuning
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError(
                "Expected grasp tuning entries like mustard_bottle:0,0,0.06"
            )
        name, payload = item.split(":", 1)
        name = name.strip()
        if name not in OBJECTS:
            raise argparse.ArgumentTypeError(f"Unknown object in grasp tuning: {name!r}")
        parts = [float(part.strip()) for part in payload.split(",") if part.strip()]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"Expected dx,dy,dz_override for {name}, got {payload!r}"
            )
        tuning[name] = {"dx": parts[0], "dy": parts[1], "z_offset": parts[2]}
    return tuning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Task E environment seed. Use the same seed when executing the request.",
    )
    parser.add_argument(
        "--object-order",
        type=parse_object_order,
        default=list(DEFAULT_PICK_ORDER),
        help="Comma-separated pick order. Default: banana,mustard_bottle,box_object.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON. Defaults to outputs/task_e_full_baseline/<timestamp>/motion_request.json.",
    )
    parser.add_argument(
        "--preferred-backend",
        choices=("moveit_py", "isaaclab_cartesian_controller"),
        default="moveit_py",
    )
    parser.add_argument(
        "--actuator-mode",
        choices=("default", "task_e_scripted_high_stiffness"),
        default="task_e_scripted_high_stiffness",
        help="Controller settings requested from task_e_moveit_runner.py.",
    )
    parser.add_argument(
        "--object-transport-mode",
        choices=("physics", "kinematic_attach"),
        default="kinematic_attach",
        help=(
            "physics uses only simulator contact; kinematic_attach keeps the object attached to "
            "the EE during transport for a deterministic whole-task baseline."
        ),
    )
    parser.add_argument(
        "--grasp-tuning",
        type=parse_grasp_tuning,
        default={},
        help=(
            "Semicolon-separated object:dx,dy,dz_override entries, for example "
            "'mustard_bottle:0,0,0.06;box_object:0,0,0.07'."
        ),
    )
    parser.add_argument("--init-steps", type=int, default=100)
    parser.add_argument("--pregrasp-steps", type=int, default=200)
    parser.add_argument("--grasp-steps", type=int, default=110)
    parser.add_argument("--close-steps", type=int, default=55)
    parser.add_argument("--lift-steps", type=int, default=160)
    parser.add_argument("--transport-steps", type=int, default=210)
    parser.add_argument("--place-steps", type=int, default=100)
    parser.add_argument("--open-steps", type=int, default=70)
    parser.add_argument("--retract-steps", type=int, default=90)
    return parser.parse_args()


def make_output(path: Path | None) -> Path:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = REPO_ROOT / "outputs/task_e_full_baseline" / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    latest_txt = output_dir.parent / "latest.txt"
    latest_txt.write_text(str(output_dir.resolve()), encoding="utf-8")
    latest_link = output_dir.parent / "latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(output_dir.resolve(), target_is_directory=True)
    except OSError:
        pass
    return output_dir / "motion_request.json"


def deterministic_object_poses(seed: int | None) -> dict[str, dict]:
    rng = np.random.default_rng(seed=seed)
    x_min, x_max = TABLE_CENTER_X - 0.10, TABLE_CENTER_X + 0.10
    z = TABLE_TOP_Z + 0.05
    placed: dict[str, tuple[float, float]] = {}
    poses = {}

    for name in ENV_OBJECT_ORDER:
        cfg = OBJECTS[name]
        y_min, y_max = cfg["spawn_band_y"]
        hx, hy = cfg["half_extents_xy"]
        x = y = None
        for _ in range(200):
            cx = float(rng.uniform(x_min, x_max))
            cy = float(rng.uniform(y_min, y_max))
            ok = all(
                abs(cx - px) >= hx + OBJECTS[pn]["half_extents_xy"][0] + OBJ_BBOX_MARGIN
                or abs(cy - py) >= hy + OBJECTS[pn]["half_extents_xy"][1] + OBJ_BBOX_MARGIN
                for pn, (px, py) in placed.items()
            )
            if ok:
                x, y = cx, cy
                break
        if x is None:
            x = (x_min + x_max) / 2.0
            y = (y_min + y_max) / 2.0
        placed[name] = (x, y)
        poses[name] = {
            "object_key": cfg["object_key"],
            "label": cfg["label"],
            "center_w": [float(x), float(y), float(z)],
            "quat_wxyz": cfg["object_quat_wxyz"],
            "source": "deterministic_task_e_spawn_model",
        }
    return poses


def quat_normalize(quat: list[float] | np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def quat_wxyz_to_matrix(quat: list[float] | np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_wxyz_from_matrix(matrix: np.ndarray) -> list[float]:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 0.0)) * 2.0
        quat = np.array(
            [
                (m[2, 1] - m[1, 2]) / s,
                0.25 * s,
                (m[0, 1] + m[1, 0]) / s,
                (m[0, 2] + m[2, 0]) / s,
            ],
            dtype=np.float64,
        )
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 0.0)) * 2.0
        quat = np.array(
            [
                (m[0, 2] - m[2, 0]) / s,
                (m[0, 1] + m[1, 0]) / s,
                0.25 * s,
                (m[1, 2] + m[2, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        s = np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 0.0)) * 2.0
        quat = np.array(
            [
                (m[1, 0] - m[0, 1]) / s,
                (m[0, 2] + m[2, 0]) / s,
                (m[1, 2] + m[2, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )
    return quat_normalize(quat).astype(float).tolist()


def build_grasp_matrix(long_axis: np.ndarray, grip_z: np.ndarray) -> np.ndarray:
    jaw_dir = np.cross(long_axis, grip_z)
    jaw_dir = jaw_dir / max(np.linalg.norm(jaw_dir), 1e-9)
    align_dir = np.cross(jaw_dir, grip_z)
    align_dir = align_dir / max(np.linalg.norm(align_dir), 1e-9)
    return np.stack([align_dir, jaw_dir, grip_z], axis=1)


def task_e_grasp_quat(object_quat_wxyz: list[float]) -> list[float]:
    """Pure-numpy copy of scripts/act/task_e/state_machine.compute_grasp_quat."""
    rot_obj = quat_wxyz_to_matrix(object_quat_wxyz)
    grip_z = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    default_quat = np.asarray(DEFAULT_QUAT_WXYZ, dtype=np.float64)

    norms: list[float] = []
    axes_xy: list[np.ndarray] = []
    for col in range(3):
        axis = np.array([rot_obj[0, col], rot_obj[1, col], 0.0], dtype=np.float64)
        norms.append(float(np.linalg.norm(axis)))
        axes_xy.append(axis)

    best_norm = max(norms)
    candidates = [
        axes_xy[idx] / max(norms[idx], 1e-9)
        for idx in range(3)
        if norms[idx] >= best_norm - 1e-3
    ]

    best_axis = candidates[0]
    best_cos = -2.0
    for candidate in candidates:
        quat = np.asarray(quat_wxyz_from_matrix(build_grasp_matrix(candidate, grip_z)), dtype=np.float64)
        cos_sim = abs(float(np.dot(quat, default_quat)))
        if cos_sim > best_cos:
            best_cos = cos_sim
            best_axis = candidate

    return quat_wxyz_from_matrix(build_grasp_matrix(best_axis, grip_z))


def pose(position: list[float], quat: list[float] | None = None) -> dict:
    return {"position": [float(v) for v in position], "quat_wxyz": list(quat or DEFAULT_QUAT_WXYZ)}


def waypoint(
    name: str,
    position: list[float],
    gripper: list[float],
    steps: int,
    capture: bool = True,
    quat: list[float] | None = None,
    object_transport: dict | None = None,
) -> dict:
    item = {
        "name": name,
        "pose_w": pose(position, quat),
        "gripper_joint_pos": list(gripper),
        "steps": int(steps),
        "capture": bool(capture),
    }
    if object_transport is not None:
        item["object_transport"] = object_transport
    return item


def build_request(args: argparse.Namespace, object_poses: dict[str, dict]) -> dict:
    waypoints = [
        waypoint(
            "00_initial_retract",
            [RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z],
            GRIPPER_OPEN,
            args.init_steps,
        )
    ]
    object_records = []

    for obj_index, name in enumerate(args.object_order, start=1):
        cfg = OBJECTS[name]
        obj = object_poses[name]
        x, y, z = obj["center_w"]
        tuning = args.grasp_tuning.get(name, {})
        grasp_x = x + float(tuning.get("dx", 0.0))
        grasp_y = y + float(tuning.get("dy", 0.0))
        grasp_z_offset = float(tuning.get("z_offset", cfg["grasp_z_offset"]))
        place_x = BASKET_CENTER_X + cfg["place_offset_xy"][0]
        place_y = BASKET_CENTER_Y + cfg["place_offset_xy"][1]
        prefix = f"{obj_index:02d}_{name}"
        grasp_quat = task_e_grasp_quat(obj["quat_wxyz"])
        ee_to_object_pos = [x - grasp_x, y - grasp_y, -grasp_z_offset]
        attach_payload = {
            "action": "attach",
            "object_key": cfg["object_key"],
            "object_name": name,
            "ee_to_object_pos_w": ee_to_object_pos,
            "object_quat_wxyz": obj["quat_wxyz"],
        }
        release_payload = {
            "action": "release",
            "object_key": cfg["object_key"],
            "object_name": name,
            "release_center_w": [place_x, place_y, TABLE_TOP_Z + 0.05],
            "object_quat_wxyz": obj["quat_wxyz"],
        }

        object_records.append(
            {
                "name": name,
                "object_key": cfg["object_key"],
                "label": cfg["label"],
                "center_w": obj["center_w"],
                "object_quat_wxyz": obj["quat_wxyz"],
                "grasp_tuning": {
                    "dx": grasp_x - x,
                    "dy": grasp_y - y,
                    "z_offset": grasp_z_offset,
                },
                "grasp_pose_w": pose([grasp_x, grasp_y, z + grasp_z_offset], grasp_quat),
                "place_pose_w": pose([place_x, place_y, PLACE_HEIGHT]),
            }
        )

        waypoints.extend(
            [
                waypoint(f"{prefix}_pregrasp", [grasp_x, grasp_y, CARRY_Z], GRIPPER_OPEN, args.pregrasp_steps),
                waypoint(
                    f"{prefix}_grasp",
                    [grasp_x, grasp_y, z + grasp_z_offset],
                    GRIPPER_OPEN,
                    args.grasp_steps,
                    quat=grasp_quat,
                ),
                waypoint(
                    f"{prefix}_close",
                    [grasp_x, grasp_y, z + grasp_z_offset],
                    GRIPPER_CLOSE,
                    args.close_steps,
                    quat=grasp_quat,
                    object_transport=attach_payload,
                ),
                waypoint(f"{prefix}_lift", [grasp_x, grasp_y, CARRY_Z], GRIPPER_CLOSE, args.lift_steps, quat=grasp_quat),
                waypoint(f"{prefix}_transport", [place_x, place_y, CARRY_Z], GRIPPER_CLOSE, args.transport_steps),
                waypoint(f"{prefix}_place", [place_x, place_y, PLACE_HEIGHT], GRIPPER_CLOSE, args.place_steps),
                waypoint(
                    f"{prefix}_open",
                    [place_x, place_y, PLACE_HEIGHT],
                    GRIPPER_OPEN,
                    args.open_steps,
                    object_transport=release_payload,
                ),
                waypoint(f"{prefix}_lift_retract", [place_x, place_y, CARRY_Z], GRIPPER_OPEN, args.retract_steps),
            ]
        )

    waypoints.append(
        waypoint(
            "99_final_retract",
            [RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z],
            GRIPPER_OPEN,
            args.retract_steps,
        )
    )

    return {
        "schema": REQUEST_SCHEMA,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": "ATEC-TaskE-Piper",
        "frame": "world",
        "units": {"position": "meter", "quaternion": "wxyz"},
        "source": {
            "type": "full_task_pseudo_baseline",
            "object_pose_source": "deterministic_task_e_spawn_model",
            "object_order": list(args.object_order),
            "objects": object_records,
            "basket": {
                "center_w": [BASKET_CENTER_X, BASKET_CENTER_Y, PLACE_HEIGHT],
                "success_half_x": BASKET_SUCCESS_HALF_X,
                "success_half_y": BASKET_SUCCESS_HALF_Y,
            },
        },
        "backend": {
            "preferred": args.preferred_backend,
            "fallback": "isaaclab_cartesian_controller",
        },
        "controller": {
            "actuator_mode": args.actuator_mode,
            "object_transport_mode": args.object_transport_mode,
            "grasp_quat_source": "scripts/act/task_e/state_machine.py",
        },
        "robot": {
            "name": "piper",
            "planning_group": "piper_arm",
            "ee_link": "gripper_base",
            "arm_joints": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            "gripper_joints": ["joint7", "joint8"],
            "action_scale": 0.5,
        },
        "start_state": {"source": "task_reset", "seed": args.seed},
        "waypoints": waypoints,
    }


def main() -> None:
    args = parse_args()
    output = make_output(args.output)
    object_poses = deterministic_object_poses(args.seed)
    request = build_request(args, object_poses)
    output.write_text(json.dumps(request, indent=2), encoding="utf-8")
    print(f"[INFO] Saved full Task E baseline request: {output}")
    print(f"[INFO] Seed: {args.seed}")
    print(f"[INFO] Object order: {', '.join(args.object_order)}")


if __name__ == "__main__":
    main()
