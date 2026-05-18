import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


CAPTURE_PROMPTS = [
    {"name": "banana", "prompt": "banana"},
    {"name": "mustard_bottle", "prompt": "mustard bottle"},
    {"name": "box_object", "prompt": "yellow and white box"},
    {"name": "purple_basket", "prompt": "purple basket"},
]

# Task E reset leaves the eye-in-hand camera looking at the sky.  In capture
# mode, move to a fixed table-observation joint target before saving frames.
# Raw action convention: action = (joint_target - default_joint_pos) / 0.5.
TASK_E_CAPTURE_CAMERA_ACTION = [
    -0.41051948070526123,
    1.5470619201660156,
    -0.38924479484558105,
    0.000016453657735837623,
    0.039999961853027344,
    -0.4105374217033386,
    -0.0000010281801223754883,
    0.0000001043081283569336,
]
TASK_E_CAPTURE_SETTLE_STEPS = 160


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _env_float_list(name: str, default: list[float]) -> list[float]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return list(default)
    return [float(part.strip()) for part in value.split(",") if part.strip()]


class AlgSolution:

    def __init__(self):
        self.capture_enabled = _env_flag("ATEC_SAM3_CAPTURE")
        self.capture_root = Path(
            os.environ.get("ATEC_SAM3_CAPTURE_DIR", "outputs/task_e_sam3_capture")
        )
        self.capture_settle_steps = _env_int(
            "ATEC_SAM3_CAPTURE_SETTLE_STEPS", TASK_E_CAPTURE_SETTLE_STEPS
        )
        self.capture_action = _env_float_list(
            "ATEC_SAM3_CAPTURE_ACTION", TASK_E_CAPTURE_CAMERA_ACTION
        )
        self.capture_dir: Path | None = None
        self._captured = False
        self._giveup_next = False
        self._capture_step = 0

    def reset(self, **_kwargs):
        self._captured = False
        self._giveup_next = False
        self._capture_step = 0

    def predicts(self, obs, current_score):
        action = self._zero_action(obs)

        if self.capture_enabled:
            if self._giveup_next:
                return {"action": action, "giveup": True}
            if not self._captured:
                action = self._capture_motion_action(obs)
            if (
                not self._captured
                and self._capture_step >= self.capture_settle_steps
                and self._has_task_e_images(obs)
            ):
                self._capture_task_e_images(obs, current_score)
                self._captured = True
                self._giveup_next = True
                action = self._zero_action(obs)
            elif not self._captured:
                self._capture_step += 1
            return {"action": action, "giveup": False}

        return {"action": action, "giveup": False}

    def _zero_action(self, obs):
        proprio = obs["proprio"]
        action_dim = self._infer_action_dim(obs, proprio)
        action = [0 for _ in range(action_dim)]
        return action

    def _infer_action_dim(self, obs, proprio) -> int:
        image_obs = obs.get("image") if isinstance(obs, dict) else None
        proprio_dim = int(proprio.shape[-1])
        if isinstance(image_obs, dict) and "video_rgb" in image_obs:
            return max(1, proprio_dim // 3)
        return max(1, (proprio_dim - 12) // 3)

    def _has_task_e_images(self, obs) -> bool:
        image_obs = obs.get("image") if isinstance(obs, dict) else None
        return (
            isinstance(image_obs, dict)
            and image_obs.get("video_rgb") is not None
            and image_obs.get("ee_rgb") is not None
        )

    def _capture_motion_action(self, obs) -> list[float]:
        action_dim = len(self._zero_action(obs))
        action = list(self.capture_action[:action_dim])
        if len(action) < action_dim:
            action.extend([0.0] * (action_dim - len(action)))
        return action

    def _capture_task_e_images(self, obs, current_score) -> None:
        from PIL import Image

        image_obs = obs["image"]
        capture_dir = self._make_capture_dir()

        view_records = {}
        for view_name, file_name in (
            ("video_rgb", "video_rgb.png"),
            ("ee_rgb", "ee_rgb.png"),
        ):
            rgb = self._to_rgb_array(image_obs[view_name])
            path = capture_dir / file_name
            Image.fromarray(rgb, mode="RGB").save(path)
            view_records[view_name] = {
                "file": file_name,
                "shape": list(rgb.shape),
                "dtype": str(rgb.dtype),
            }

        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": "demo.solution.AlgSolution.predicts",
            "entrypoint": "python scripts/play_atec_task.py --task ATEC-TaskE-Piper --enable_cameras",
            "current_score": float(current_score) if current_score is not None else None,
            "capture_settle_steps": self.capture_settle_steps,
            "capture_action": self.capture_action,
            "views": view_records,
            "prompts": CAPTURE_PROMPTS,
            "object_identity": {
                "object_1": "sugar box",
                "object_2": "mustard bottle",
                "object_3": "banana",
                "basket": "purple basket",
            },
        }
        (capture_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        self._update_latest(capture_dir)
        print(f"[ATEC_SAM3_CAPTURE] Saved Task E camera frames to {capture_dir}")

    def _make_capture_dir(self) -> Path:
        if self.capture_dir is not None:
            return self.capture_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.capture_root.mkdir(parents=True, exist_ok=True)
        capture_dir = self.capture_root / timestamp
        suffix = 1
        while capture_dir.exists():
            capture_dir = self.capture_root / f"{timestamp}_{suffix:02d}"
            suffix += 1
        capture_dir.mkdir(parents=True, exist_ok=False)
        self.capture_dir = capture_dir
        return capture_dir

    def _update_latest(self, capture_dir: Path) -> None:
        latest = self.capture_root / "latest"
        try:
            if latest.is_symlink() or latest.is_file():
                latest.unlink()
            if not latest.exists():
                latest.symlink_to(capture_dir.name, target_is_directory=True)
                return
        except OSError:
            pass
        (self.capture_root / "latest.txt").write_text(
            str(capture_dir), encoding="utf-8"
        )

    def _to_rgb_array(self, value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)

        if array.ndim == 4:
            array = array[0]
        if array.ndim != 3:
            raise ValueError(
                f"Expected RGB tensor with 3 or 4 dimensions, got {array.shape}"
            )

        if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))

        if array.shape[-1] == 4:
            array = array[:, :, :3]
        elif array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        elif array.shape[-1] != 3:
            raise ValueError(f"Expected RGB/RGBA channels, got {array.shape}")

        if array.dtype != np.uint8:
            array = array.astype(np.float32)
            if array.size and float(np.nanmax(array)) <= 1.0:
                array = array * 255.0
            array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
            array = np.clip(array, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(array)
