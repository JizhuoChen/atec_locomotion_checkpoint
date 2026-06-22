"""Online Task E submission policy built from the latest full-success runs.

The debug pipeline that succeeded on seeds 120/130 cannot be submitted as-is:
the submission server only calls ``AlgSolution.predicts(obs, current_score)``
and accepts joint-position actions.  This policy therefore keeps the submission
side online by estimating the current object location from RGB-D, choosing or
interpolating the actuator-friendly command profiles from those successful
runs, and advancing by score.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch


ACTION_DIM = 8
TABLE_TOP_Z = 0.8266426479059952
VIDEO_INTRINSIC = np.asarray(
    [[732.999267578125, 0.0, 320.0], [0.0, 732.999267578125, 240.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
VIDEO_POS_W = np.asarray([-0.20000000298023224, 0.0, 1.6266427040100098], dtype=np.float32)
VIDEO_QUAT_W_ROS = np.asarray(
    [-0.3335084915161133, 0.6235159635543823, -0.6235158443450928, 0.3335084915161133],
    dtype=np.float32,
)
OBJECT_ORDER = ("mustard_bottle", "box_object", "banana")
SCORE_DONE = {"mustard_bottle": 5.5, "box_object": 11.5, "banana": 17.5}
OBJECT_LABELS = {"mustard_bottle": "mustard", "box_object": "box", "banana": "banana"}
BOX_X_GRID = np.asarray([0.90, 0.95, 1.00, 1.05, 1.10], dtype=np.float32)
BOX_TABLE = {
    "pre": [
        [-0.9832, 2.3189, -2.5633, 0.0, 0.36, -0.9831, 0.0, 0.0],
        [-1.0726, 2.2173, -2.5327, 0.0, 0.36, -1.0725, 0.0, 0.0],
        [-1.1777, 2.1012, -2.4858, 0.0, 0.36, -1.1777, 0.0, 0.0],
        [-1.3025, 1.9668, -2.4091, 0.0, 0.36, -1.3025, 0.0, 0.0],
        [-1.4516, 1.8068, -2.2738, 0.0, 0.36, -1.4516, 0.0, 0.0],
    ],
    "grasp": [
        [-0.9832, 3.2082, -2.4437, 0.0018, 0.3600, -0.9822, 0.0, 0.0],
        [-1.0718, 3.0205, -2.0798, -0.0022, 0.3601, -1.0700, 0.0, 0.0],
        [-1.1746, 2.5474, -1.0697, 0.0069, 0.0397, -1.1777, 0.0, 0.0],
        [-1.3005, 2.2123, -0.4041, 0.0012, -0.2920, -1.3021, 0.0, 0.0],
        [-1.4509, 1.9343, 0.1222, -0.0004, -0.5411, -1.4520, 0.0, 0.0],
    ],
    "transport": [1.3011, 2.1273, -2.4964, -0.0005, 0.36, 1.2999, -0.1, 0.1],
    "place": [1.3002, 2.5984, -1.5558, -0.0004, 0.36, 1.2997, -0.1, 0.1],
    "lift_open": [1.3011, 2.1273, -2.4964, -0.0005, 0.36, 1.2999, 0.0, 0.0],
}


class AlgSolution:
    def __init__(self):
        self._profiles = self._load_profiles()
        self._step_scale = self._read_float_env("ATEC_TASK_E_PROFILE_STEP_SCALE", 0.62)
        self._object_step_scales = {
            "mustard_bottle": self._read_float_env("ATEC_TASK_E_MUSTARD_PROFILE_STEP_SCALE", self._step_scale),
            "box_object": self._read_float_env("ATEC_TASK_E_BOX_PROFILE_STEP_SCALE", 1.0),
            "banana": self._read_float_env("ATEC_TASK_E_BANANA_PROFILE_STEP_SCALE", 0.68),
        }
        self._min_steps = self._read_int_env("ATEC_TASK_E_PROFILE_MIN_STEPS", 12)
        self._object_index = 0
        self._active_object: str | None = None
        self._actions: list[list[float]] = []
        self._action_step = 0
        self._attempt_counts = {name: 0 for name in OBJECT_ORDER}
        self._last_action = [0.0] * ACTION_DIM

        # Names used by the local debug harness heartbeat.
        self._scripted_actions: list[list[float]] = []
        self._scripted_step = 0
        self._scripted_started = False
        self._scripted_targets: list[tuple[str, float | None, float]] = []

    def reset(self, **_kwargs):
        self._object_index = 0
        self._active_object = None
        self._actions = []
        self._action_step = 0
        self._attempt_counts = {name: 0 for name in OBJECT_ORDER}
        self._last_action = [0.0] * ACTION_DIM
        self._scripted_actions = []
        self._scripted_step = 0
        self._scripted_started = False
        self._scripted_targets = []

    def predicts(self, obs, current_score):
        proprio = self._as_numpy(obs["proprio"]) if isinstance(obs, dict) and "proprio" in obs else np.zeros((1, 24))
        if proprio.ndim == 1:
            proprio = proprio.reshape(1, -1)
        num_envs = int(proprio.shape[0])
        action_dim = self._infer_action_dim(obs, proprio)
        score = float(current_score or 0.0)

        self._advance_by_score(score)
        if self._active_object is not None and score >= SCORE_DONE[self._active_object]:
            self._finish_active_object()
            self._advance_by_score(score)

        if not self._actions:
            object_name = self._next_object()
            if object_name is None:
                action = self._fit_action(self._last_action, action_dim)
                return {"action": [action for _ in range(num_envs)], "giveup": False}
            self._start_object_profile(obs, object_name)

        if self._action_step >= len(self._actions):
            # The profile did not score. Retry the same object with another
            # observation-selected profile on the next call.
            if self._active_object is not None:
                self._attempt_counts[self._active_object] += 1
            self._active_object = None
            self._actions = []
            self._action_step = 0
            action = self._fit_action(self._last_action, action_dim)
            return {"action": [action for _ in range(num_envs)], "giveup": False}

        action = self._fit_action(self._actions[self._action_step], action_dim)
        self._last_action = list(action[:ACTION_DIM]) + [0.0] * max(0, ACTION_DIM - len(action))
        self._action_step += 1
        self._scripted_step = self._action_step
        return {"action": [action for _ in range(num_envs)], "giveup": False}

    def _start_object_profile(self, obs, object_name: str) -> None:
        estimated_center = self._estimate_object_center(obs, object_name)
        waypoints, used_x, strategy = self._waypoints_for_object(object_name, estimated_center)
        initial = self._last_action_from_obs(obs) or self._last_action

        actions: list[list[float]] = [self._fit_action(initial, ACTION_DIM)]
        scale = float(self._object_step_scales.get(object_name, self._step_scale))
        for waypoint in waypoints:
            steps = max(self._min_steps, int(round(int(waypoint["steps"]) * scale)))
            self._append_transition(actions, waypoint["action"], steps)

        self._active_object = object_name
        self._actions = actions
        self._action_step = 0
        self._scripted_actions = actions
        self._scripted_step = 0
        self._scripted_started = True
        estimated_x = None if estimated_center is None else float(estimated_center[0])
        self._scripted_targets = [(f"{OBJECT_LABELS[object_name]}:{strategy}", estimated_x, float(used_x))]

    def _waypoints_for_object(self, object_name: str, center: np.ndarray | None) -> tuple[list[dict], float, str]:
        candidates = [p for p in self._profiles["profiles"] if p["object"] == object_name]
        if not candidates:
            raise RuntimeError(f"No action profile for {object_name}")
        live_x = self._fallback_live_x(object_name) if center is None else float(center[0])

        if object_name in {"mustard_bottle", "box_object"}:
            if object_name == "box_object" and self._attempt_counts.get(object_name, 0) == 0:
                return self._box_table_waypoints(live_x + 0.03), live_x + 0.03, "table_online_x"
            successful = [p for p in candidates if p.get("success")]
            return self._blend_success_profiles(
                successful or candidates,
                live_x,
                extrapolate=object_name == "box_object",
            )

        successful = [p for p in candidates if p.get("success")] or candidates
        successful = sorted(successful, key=lambda p: self._profile_x(p))
        attempt = self._attempt_counts.get(object_name, 0)
        if attempt == 0 and len(successful) > 1:
            # After box transport, the banana video mask often includes nearby
            # yellow/white clutter and overestimates x.  Prefer the left
            # successful profile unless the estimate is clearly on the right.
            profile = successful[-1] if live_x >= 1.10 else successful[0]
            self._banana_first_profile_index = successful.index(profile)
        elif attempt == 0:
            profile = successful[0]
            self._banana_first_profile_index = 0
        else:
            first_index = int(getattr(self, "_banana_first_profile_index", 0))
            ordered = successful[first_index:] + successful[:first_index]
            profile = ordered[attempt % len(ordered)]
        return [dict(wp) for wp in profile["waypoints"]], self._profile_x(profile), f"{profile['seed']}_attempt{profile['attempt']}"

    def _box_table_waypoints(self, x: float) -> list[dict]:
        pre = self._interp_box_table("pre", x)
        grasp = self._interp_box_table("grasp", x)
        close = list(grasp)
        close[-2], close[-1] = -0.1, 0.1
        lift = list(pre)
        lift[-2], lift[-1] = -0.1, 0.1
        place_open = list(BOX_TABLE["place"])
        place_open[-2], place_open[-1] = 0.0, 0.0
        return [
            {"name": "box_table_pre", "steps": 190, "action": pre},
            {"name": "box_table_grasp", "steps": 130, "action": grasp},
            {"name": "box_table_close", "steps": 125, "action": close},
            {"name": "box_table_lift", "steps": 190, "action": lift},
            {"name": "box_table_transport", "steps": 230, "action": list(BOX_TABLE["transport"])},
            {"name": "box_table_place", "steps": 120, "action": list(BOX_TABLE["place"])},
            {"name": "box_table_open", "steps": 90, "action": place_open},
            {"name": "box_table_lift_open", "steps": 100, "action": list(BOX_TABLE["lift_open"])},
        ]

    @staticmethod
    def _interp_box_table(key: str, x: float) -> list[float]:
        table = np.asarray(BOX_TABLE[key], dtype=np.float32)
        x_clamped = float(np.clip(x, float(BOX_X_GRID[0]), float(BOX_X_GRID[-1])))
        return [float(np.interp(x_clamped, BOX_X_GRID, table[:, dim])) for dim in range(table.shape[1])]

    def _blend_success_profiles(
        self,
        profiles: list[dict],
        live_x: float,
        *,
        extrapolate: bool = False,
    ) -> tuple[list[dict], float, str]:
        profiles = sorted(profiles, key=lambda p: self._profile_x(p))
        if len(profiles) == 1:
            profile = profiles[0]
            return [dict(wp) for wp in profile["waypoints"]], self._profile_x(profile), profile["seed"]

        left, right = profiles[0], profiles[-1]
        for idx in range(len(profiles) - 1):
            lo, hi = profiles[idx], profiles[idx + 1]
            if self._profile_x(lo) <= live_x <= self._profile_x(hi):
                left, right = lo, hi
                break
        x0, x1 = self._profile_x(left), self._profile_x(right)
        if abs(x1 - x0) <= 1e-6:
            alpha = 0.0
        elif extrapolate:
            # The box often appears slightly farther toward the arm than the
            # two successful references.  Extrapolating the action profile is
            # closer to the original online planner than clamping to seed130.
            alpha = float(np.clip((live_x - x0) / (x1 - x0), -0.5, 2.4))
        else:
            alpha = float(np.clip((live_x - x0) / (x1 - x0), 0.0, 1.0))

        if len(left["waypoints"]) != len(right["waypoints"]):
            profile = min((left, right), key=lambda p: abs(self._profile_x(p) - live_x))
            return [dict(wp) for wp in profile["waypoints"]], self._profile_x(profile), profile["seed"]

        waypoints = []
        for wp_l, wp_r in zip(left["waypoints"], right["waypoints"], strict=False):
            action_l = np.asarray(wp_l["action"], dtype=np.float32)
            action_r = np.asarray(wp_r["action"], dtype=np.float32)
            steps = int(round((1.0 - alpha) * int(wp_l["steps"]) + alpha * int(wp_r["steps"])))
            name_l, name_r = str(wp_l.get("name")), str(wp_r.get("name"))
            waypoints.append(
                {
                    "name": name_l if name_l == name_r else f"{name_l}|{name_r}",
                    "steps": max(1, steps),
                    "action": ((1.0 - alpha) * action_l + alpha * action_r).astype(float).tolist(),
                }
            )
        used_x = (1.0 - alpha) * x0 + alpha * x1
        return waypoints, used_x, f"blend_{left['seed']}_{right['seed']}"

    def _advance_by_score(self, score: float) -> None:
        while self._object_index < len(OBJECT_ORDER):
            name = OBJECT_ORDER[self._object_index]
            if score < SCORE_DONE[name]:
                break
            self._object_index += 1
            if self._active_object == name:
                self._finish_active_object()

    def _finish_active_object(self) -> None:
        self._active_object = None
        self._actions = []
        self._action_step = 0
        self._scripted_actions = []
        self._scripted_step = 0
        self._scripted_started = False

    def _next_object(self) -> str | None:
        if self._object_index >= len(OBJECT_ORDER):
            return None
        return OBJECT_ORDER[self._object_index]

    @staticmethod
    def _profile_x(profile: dict) -> float:
        center = profile.get("video_reference_center_w") or profile.get("reference_center_w")
        if not center:
            return 1.0
        return float(center[0])

    @staticmethod
    def _fallback_live_x(object_name: str) -> float:
        return {"mustard_bottle": 0.95, "box_object": 1.02, "banana": 1.02}.get(object_name, 1.0)

    def _estimate_object_center(self, obs, object_name: str) -> np.ndarray | None:
        image = obs.get("image") if isinstance(obs, dict) else None
        if not isinstance(image, dict) or image.get("video_rgb") is None or image.get("video_depth") is None:
            return None
        rgb = self._to_rgb_array(image["video_rgb"]).astype(np.int16)
        depth = self._to_depth_array(image["video_depth"])
        if depth.shape[:2] != rgb.shape[:2]:
            return None

        points = self._video_points_world(depth)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        yellow = (r > 135) & (g > 105) & (b < 130) & (r > b + 35) & (g > b + 25)
        white = (r > 170) & (g > 170) & (b > 135) & (np.abs(r - g) < 85)
        valid_depth = np.isfinite(depth) & (depth > 0.05) & (depth < 2.2)

        y_ranges = {
            "banana": (0.00, 0.12),
            "mustard_bottle": (0.09, 0.24),
            "box_object": (0.20, 0.36),
        }
        mask = (yellow | white) if object_name == "box_object" else yellow
        selected = points[(mask & valid_depth).reshape(-1)]
        if selected.shape[0] < 20:
            return None
        y_min, y_max = y_ranges[object_name]
        workspace = (
            (selected[:, 0] >= 0.80)
            & (selected[:, 0] <= 1.25)
            & (selected[:, 1] >= y_min)
            & (selected[:, 1] <= y_max)
            & (selected[:, 2] >= TABLE_TOP_Z - 0.05)
            & (selected[:, 2] <= TABLE_TOP_Z + 0.36)
        )
        selected = selected[workspace]
        if selected.shape[0] < 20:
            return None
        return np.median(selected, axis=0).astype(np.float32)

    def _video_points_world(self, depth: np.ndarray) -> np.ndarray:
        h, w = depth.shape[:2]
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        z = depth.astype(np.float32)
        points_cam = np.stack(
            [
                (xs - VIDEO_INTRINSIC[0, 2]) * z / VIDEO_INTRINSIC[0, 0],
                (ys - VIDEO_INTRINSIC[1, 2]) * z / VIDEO_INTRINSIC[1, 1],
                z,
            ],
            axis=-1,
        ).reshape(-1, 3)
        return points_cam @ self._quat_wxyz_to_matrix(VIDEO_QUAT_W_ROS).T + VIDEO_POS_W

    @staticmethod
    def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float32)
        quat = quat / max(float(np.linalg.norm(quat)), 1e-8)
        w, x, y, z = quat
        return np.asarray(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )

    def _last_action_from_obs(self, obs) -> list[float] | None:
        try:
            proprio = self._as_numpy(obs["proprio"])
            if proprio.ndim == 1:
                proprio = proprio.reshape(1, -1)
            action = proprio[0, 16:24]
            if action.shape[0] == ACTION_DIM and np.isfinite(action).all():
                return action.astype(float).tolist()
        except Exception:
            return None
        return None

    @staticmethod
    def _append_transition(sequence: list[list[float]], action: list[float], steps: int) -> None:
        target = np.asarray(action[:ACTION_DIM], dtype=np.float32)
        start = np.asarray(sequence[-1] if sequence else [0.0] * ACTION_DIM, dtype=np.float32)
        for idx in range(max(1, int(steps))):
            alpha = float(idx + 1) / float(max(1, int(steps)))
            sequence.append((start + alpha * (target - start)).astype(float).tolist())

    def _load_profiles(self) -> dict:
        with (Path(__file__).resolve().parent / "action_profiles.json").open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _read_float_env(name: str, default: float) -> float:
        value = os.environ.get(name, "").strip()
        if not value:
            return float(default)
        try:
            return float(value)
        except ValueError:
            return float(default)

    @staticmethod
    def _read_int_env(name: str, default: int) -> int:
        value = os.environ.get(name, "").strip()
        if not value:
            return int(default)
        try:
            return int(value)
        except ValueError:
            return int(default)

    @staticmethod
    def _as_numpy(value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _to_rgb_array(value) -> np.ndarray:
        array = AlgSolution._as_numpy(value)
        if array.ndim == 4:
            array = array[0]
        if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
            array = np.transpose(array, (1, 2, 0))
        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.dtype != np.uint8:
            array = array.astype(np.float32)
            if array.size and np.nanmax(array) <= 1.0:
                array *= 255.0
            array = np.clip(np.nan_to_num(array), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    @staticmethod
    def _to_depth_array(value) -> np.ndarray:
        array = AlgSolution._as_numpy(value)
        if array.ndim == 4:
            array = array[0]
        if array.ndim == 3:
            if array.shape[0] == 1:
                array = array[0]
            elif array.shape[-1] == 1:
                array = array[..., 0]
        return array.astype(np.float32)

    @staticmethod
    def _infer_action_dim(obs, proprio: np.ndarray) -> int:
        image = obs.get("image") if isinstance(obs, dict) else None
        dim = int(proprio.shape[-1])
        if isinstance(image, dict) and "video_rgb" in image:
            return max(1, dim // 3)
        return ACTION_DIM if dim >= 24 else max(1, dim // 3)

    @staticmethod
    def _fit_action(action: list[float], action_dim: int) -> list[float]:
        fitted = list(action[:action_dim])
        if len(fitted) < action_dim:
            fitted.extend([0.0] * (action_dim - len(fitted)))
        return fitted
