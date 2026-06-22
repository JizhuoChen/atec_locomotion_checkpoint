"""cuRobo IK/planning utilities for Task E submission-compatible control."""

from __future__ import annotations

from dataclasses import dataclass
import json
import multiprocessing as mp
import os
from pathlib import Path
import select
import subprocess
import sys
import traceback
from typing import Iterable, Sequence

import numpy as np
import yaml


ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
GRIPPER_JOINT_NAMES = ("joint7", "joint8")
DEFAULT_JOINT_POS = np.array([0.0, 1.2, -1.5, 0.0, 1.2, 0.0, 0.035, -0.035], dtype=np.float32)
ACTION_SCALE = 0.5


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


@dataclass(frozen=True)
class CuRoboIKResult:
    success: bool
    joint_position: np.ndarray
    position_error_m: float
    rotation_error_rad: float
    solve_time_s: float
    feasible: bool | None = None
    joint_names: tuple[str, ...] = ARM_JOINT_NAMES


class TaskECuRoboPlanner:
    """Lazy cuRobo wrapper that only depends on submission-available state."""

    def __init__(
        self,
        asset_dir: str | Path | None = None,
        *,
        num_seeds: int = 64,
        position_tolerance_m: float = 0.02,
        orientation_tolerance_rad: float = 0.20,
        use_cuda_graph: bool = False,
        self_collision_check: bool = False,
        scene_collision_check: bool = False,
        collision_cache: dict[str, int] | None = None,
        optimizer_collision_activation_distance_m: float = 0.015,
    ) -> None:
        self.asset_dir = Path(asset_dir) if asset_dir is not None else Path(__file__).resolve().parent / "curobo_assets"
        self.num_seeds = int(num_seeds)
        self.position_tolerance_m = float(position_tolerance_m)
        self.orientation_tolerance_rad = float(orientation_tolerance_rad)
        self.use_cuda_graph = bool(use_cuda_graph)
        self.self_collision_check = bool(self_collision_check)
        self.scene_collision_check = bool(scene_collision_check)
        self.collision_cache = dict(collision_cache or {"cuboid": 16})
        self.optimizer_collision_activation_distance_m = float(optimizer_collision_activation_distance_m)

        self._torch = None
        self._ik = None
        self._target_link = "gripper_base"

    @staticmethod
    def arm_joints_from_proprio(proprio: np.ndarray) -> np.ndarray:
        """Recover absolute arm joint positions from submission proprio."""
        arr = np.asarray(proprio, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[0]
        if arr.shape[0] < 8:
            raise ValueError(f"Expected proprio with at least 8 qpos entries, got shape {arr.shape}")
        qpos = arr[:8] + DEFAULT_JOINT_POS
        return qpos[:6].astype(np.float32)

    @staticmethod
    def joint_target_to_action(
        arm_joint_position: Iterable[float],
        gripper_joint_position: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        """Convert absolute joint targets to the Task E normalized action."""
        target = DEFAULT_JOINT_POS.copy()
        target[:6] = np.asarray(list(arm_joint_position), dtype=np.float32)
        target[6:] = np.asarray(gripper_joint_position, dtype=np.float32)
        return ((target - DEFAULT_JOINT_POS) / ACTION_SCALE).astype(np.float32)

    @staticmethod
    def interpolate_joint_waypoints(
        start: Iterable[float],
        goal: Iterable[float],
        *,
        max_step_rad: float = 0.06,
    ) -> list[np.ndarray]:
        """Create bounded joint-space waypoints for the submission action follower."""
        start_arr = np.asarray(list(start), dtype=np.float32)
        goal_arr = np.asarray(list(goal), dtype=np.float32)
        max_delta = float(np.max(np.abs(goal_arr - start_arr)))
        steps = max(1, int(np.ceil(max_delta / float(max_step_rad))))
        return [
            (start_arr * (1.0 - alpha) + goal_arr * alpha).astype(np.float32)
            for alpha in np.linspace(1.0 / steps, 1.0, steps, dtype=np.float32)
        ]

    def _load_robot_config(self) -> dict:
        config_path = self.asset_dir / "piper_arm.yml"
        urdf_path = self.asset_dir / "piper_arm.urdf"
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        if not urdf_path.exists():
            raise FileNotFoundError(urdf_path)
        with config_path.open("r", encoding="utf-8") as f:
            robot_cfg = yaml.safe_load(f)
        kinematics = robot_cfg["robot_cfg"]["kinematics"]
        kinematics["urdf_path"] = str(urdf_path)
        kinematics["asset_root_path"] = str(self.asset_dir)
        return robot_cfg

    def _ensure_solver(self):
        if self._ik is not None:
            return self._ik

        import torch
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg

        if not torch.cuda.is_available():
            raise RuntimeError("cuRobo planner requires CUDA, but torch.cuda.is_available() is false.")

        cfg = InverseKinematicsCfg.create(
            robot=self._load_robot_config(),
            scene_model={"cuboid": {}} if self.scene_collision_check else None,
            collision_cache=self.collision_cache if self.scene_collision_check else None,
            num_seeds=self.num_seeds,
            position_tolerance=self.position_tolerance_m,
            orientation_tolerance=self.orientation_tolerance_rad,
            optimizer_collision_activation_distance=self.optimizer_collision_activation_distance_m,
            self_collision_check=self.self_collision_check,
            use_cuda_graph=self.use_cuda_graph,
            load_collision_spheres=self.scene_collision_check or self.self_collision_check,
        )
        self._torch = torch
        self._ik = InverseKinematics(cfg)
        self._target_link = self._ik.tool_frames[0]
        return self._ik

    def compute_fk(self, arm_joint_position: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
        """Return gripper_base pose for a 6-DOF arm state as (position, quat_wxyz)."""
        ik = self._ensure_solver()
        torch = self._torch
        from curobo.types import JointState

        q = torch.as_tensor(np.asarray(list(arm_joint_position), dtype=np.float32), device="cuda").view(1, 6)
        state = ik.compute_kinematics(JointState.from_position(q, joint_names=list(ARM_JOINT_NAMES)))
        pose = state.tool_poses.get_link_pose(self._target_link)
        pos = pose.position.detach().cpu().numpy().reshape(-1, 3)[0]
        quat = pose.quaternion.detach().cpu().numpy().reshape(-1, 4)[0]
        return pos.astype(np.float32), quat.astype(np.float32)

    def update_world_cuboids(self, cuboids: Sequence[dict]) -> None:
        """Update cuRobo's collision world from cuboids in robot-root frame."""
        if not self.scene_collision_check:
            raise RuntimeError("TaskECuRoboPlanner was created without scene_collision_check=True.")
        ik = self._ensure_solver()
        from curobo.scene import Cuboid, Scene

        obstacles = []
        for cuboid in cuboids:
            obstacles.append(
                Cuboid(
                    name=str(cuboid["name"]),
                    pose=[float(v) for v in cuboid["pose"]],
                    dims=[float(v) for v in cuboid["dims"]],
                )
            )
        ik.update_world(Scene(cuboid=obstacles))

    def solve_ik(
        self,
        current_arm_joint_position: Iterable[float],
        target_position: Iterable[float],
        target_quat_wxyz: Iterable[float],
        *,
        return_seeds: int = 1,
    ) -> CuRoboIKResult:
        """Solve a gripper_base IK target from current absolute arm joints."""
        ik = self._ensure_solver()
        torch = self._torch
        from curobo.types import GoalToolPose, JointState, Pose

        current_q = torch.as_tensor(
            np.asarray(list(current_arm_joint_position), dtype=np.float32), device="cuda"
        ).view(1, 6)
        goal_pos = torch.as_tensor(
            np.asarray(list(target_position), dtype=np.float32), device="cuda"
        ).view(1, 3)
        goal_quat = torch.as_tensor(
            np.asarray(list(target_quat_wxyz), dtype=np.float32), device="cuda"
        ).view(1, 4)

        current_state = JointState.from_position(current_q, joint_names=list(ARM_JOINT_NAMES))
        goal_pose = Pose(position=goal_pos, quaternion=goal_quat)
        result = ik.solve_pose(
            GoalToolPose.from_poses({self._target_link: goal_pose}, num_goalset=1),
            current_state=current_state,
            return_seeds=return_seeds,
        )

        pos = result.js_solution.position.detach().cpu().numpy()
        if pos.ndim == 3:
            joint_position = pos[0, 0]
        elif pos.ndim == 2:
            joint_position = pos[0]
        else:
            joint_position = pos.reshape(-1)[:6]

        position_error = float(result.position_error.detach().cpu().reshape(-1)[0].item())
        rotation_error = float(result.rotation_error.detach().cpu().reshape(-1)[0].item())
        success = bool(result.success.detach().cpu().reshape(-1)[0].item())
        feasible = None
        if getattr(result, "feasible", None) is not None:
            feasible = bool(result.feasible.detach().cpu().reshape(-1)[0].item())
        return CuRoboIKResult(
            success=success,
            joint_position=np.asarray(joint_position[:6], dtype=np.float32),
            position_error_m=position_error,
            rotation_error_rad=rotation_error,
            solve_time_s=float(getattr(result, "solve_time", 0.0)),
            feasible=feasible,
        )


def _prepare_curobo_warp_import() -> None:
    """Prefer the conda Warp package over Isaac's bundled omni.warp in the worker."""
    sys.path[:] = [
        path
        for path in sys.path
        if "omni.warp.core" not in path.replace("\\", "/")
    ]
    for module_name in list(sys.modules):
        if module_name == "warp" or module_name.startswith("warp."):
            del sys.modules[module_name]

    try:
        import inspect
        import warp as wp
    except Exception:
        return

    try:
        signature = inspect.signature(wp.func)
    except (TypeError, ValueError):
        return
    if "module" in signature.parameters:
        return

    original_func = wp.func
    if getattr(original_func, "_task_e_module_kwarg_compat", False):
        return

    def func_compat(f=None, *args, **kwargs):
        kwargs.pop("module", None)
        if f is None:
            return lambda wrapped: original_func(wrapped, *args, **kwargs)
        return original_func(f, *args, **kwargs)

    func_compat._task_e_module_kwarg_compat = True
    wp.func = func_compat


def _handle_planner_command(planner: TaskECuRoboPlanner, command: str, payload: dict) -> tuple[object, bool]:
    if command == "close":
        return None, True
    if command == "compute_fk":
        pos, quat = planner.compute_fk(payload["arm_joint_position"])
        return {"position": pos.tolist(), "quat_wxyz": quat.tolist()}, False
    if command == "solve_ik":
        result = planner.solve_ik(
            payload["current_arm_joint_position"],
            payload["target_position"],
            payload["target_quat_wxyz"],
            return_seeds=payload.get("return_seeds", 1),
        )
        return {
            "success": result.success,
            "joint_position": result.joint_position.tolist(),
            "position_error_m": result.position_error_m,
            "rotation_error_rad": result.rotation_error_rad,
            "solve_time_s": result.solve_time_s,
            "feasible": result.feasible,
            "joint_names": list(result.joint_names),
        }, False
    if command == "update_world_cuboids":
        planner.update_world_cuboids(payload["cuboids"])
        return None, False
    raise ValueError(f"Unknown planner worker command: {command}")


def _planner_worker_main(conn, asset_dir: str, kwargs: dict) -> None:
    planner = None
    try:
        _prepare_curobo_warp_import()
        planner = TaskECuRoboPlanner(asset_dir=asset_dir, **kwargs)
        while True:
            command, payload = conn.recv()
            result, should_close = _handle_planner_command(planner, command, payload)
            conn.send({"ok": True, "result": result})
            if should_close:
                return
    except Exception:
        conn.send({"ok": False, "error": traceback.format_exc()})
    finally:
        conn.close()


def _planner_stdio_worker_main(asset_dir: str, kwargs: dict) -> None:
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    protocol_out = os.fdopen(protocol_fd, "w", buffering=1, encoding="utf-8")
    planner = None
    try:
        _prepare_curobo_warp_import()
        planner = TaskECuRoboPlanner(asset_dir=asset_dir, **kwargs)
        for line in sys.stdin:
            command, payload = json.loads(line)
            try:
                result, should_close = _handle_planner_command(planner, command, payload or {})
                protocol_out.write(json.dumps({"ok": True, "result": result}) + "\n")
                if should_close:
                    return
            except Exception:
                protocol_out.write(json.dumps({"ok": False, "error": traceback.format_exc()}) + "\n")
                return
    except Exception:
        protocol_out.write(json.dumps({"ok": False, "error": traceback.format_exc()}) + "\n")
    finally:
        protocol_out.close()


class TaskECuRoboPlannerProcess:
    """Run cuRobo in a clean subprocess to avoid Isaac/Warp import conflicts."""

    def __init__(
        self,
        asset_dir: str | Path | None = None,
        *,
        request_timeout_s: float = 60.0,
        **planner_kwargs,
    ) -> None:
        self.asset_dir = Path(asset_dir) if asset_dir is not None else Path(__file__).resolve().parent / "curobo_assets"
        self.request_timeout_s = float(request_timeout_s)
        self._stdio = sys.platform.startswith("linux")
        self._ctx = None
        self._parent_conn = None
        self._stdin = None
        self._stdout = None
        if self._stdio:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--stdio-worker",
                str(self.asset_dir),
                json.dumps(dict(planner_kwargs)),
            ]
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                cwd=str(Path(__file__).resolve().parents[2]),
            )
            self._stdin = self._process.stdin
            self._stdout = self._process.stdout
        else:
            self._ctx = mp.get_context("spawn")
            self._parent_conn, child_conn = self._ctx.Pipe()
            self._process = self._ctx.Process(
                target=_planner_worker_main,
                args=(child_conn, str(self.asset_dir), dict(planner_kwargs)),
                daemon=True,
            )
            self._process.start()
            child_conn.close()

    def _request(self, command: str, payload: dict | None = None) -> dict:
        if self._stdio:
            if self._process.poll() is not None:
                raise RuntimeError("cuRobo planner worker is not alive.")
            assert self._stdin is not None and self._stdout is not None
            self._stdin.write(json.dumps([command, payload or {}], default=_json_default) + "\n")
            self._stdin.flush()
            ready, _, _ = select.select([self._stdout], [], [], self.request_timeout_s)
            if not ready:
                raise TimeoutError(f"cuRobo planner worker timed out on command {command!r}.")
            line = self._stdout.readline()
            if not line:
                raise RuntimeError("cuRobo planner worker closed stdout.")
            response = json.loads(line)
        else:
            if not self._process.is_alive():
                raise RuntimeError("cuRobo planner worker is not alive.")
            self._parent_conn.send((command, payload or {}))
            if not self._parent_conn.poll(self.request_timeout_s):
                raise TimeoutError(f"cuRobo planner worker timed out on command {command!r}.")
            response = self._parent_conn.recv()
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "cuRobo planner worker failed."))
        return response["result"]

    def compute_fk(self, arm_joint_position: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
        result = self._request("compute_fk", {"arm_joint_position": list(arm_joint_position)})
        return (
            np.asarray(result["position"], dtype=np.float32),
            np.asarray(result["quat_wxyz"], dtype=np.float32),
        )

    def solve_ik(
        self,
        current_arm_joint_position: Iterable[float],
        target_position: Iterable[float],
        target_quat_wxyz: Iterable[float],
        *,
        return_seeds: int = 1,
    ) -> CuRoboIKResult:
        result = self._request(
            "solve_ik",
            {
                "current_arm_joint_position": list(current_arm_joint_position),
                "target_position": list(target_position),
                "target_quat_wxyz": list(target_quat_wxyz),
                "return_seeds": int(return_seeds),
            },
        )
        return CuRoboIKResult(
            success=bool(result["success"]),
            joint_position=np.asarray(result["joint_position"], dtype=np.float32),
            position_error_m=float(result["position_error_m"]),
            rotation_error_rad=float(result["rotation_error_rad"]),
            solve_time_s=float(result["solve_time_s"]),
            feasible=result.get("feasible"),
            joint_names=tuple(result.get("joint_names", ARM_JOINT_NAMES)),
        )

    def update_world_cuboids(self, cuboids: Sequence[dict]) -> None:
        self._request("update_world_cuboids", {"cuboids": list(cuboids)})

    def close(self) -> None:
        if self._stdio:
            try:
                self._request("close")
            except Exception:
                pass
            if self._stdin is not None:
                self._stdin.close()
            if self._stdout is not None:
                self._stdout.close()
            if self._process.poll() is None:
                try:
                    self._process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
            if self._process.poll() is None:
                self._process.terminate()
            return
        if self._process.is_alive():
            try:
                self._request("close")
            except Exception:
                pass
        self._parent_conn.close()
        if self._process.is_alive():
            self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()

    def __enter__(self) -> "TaskECuRoboPlannerProcess":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


if __name__ == "__main__" and len(sys.argv) >= 2 and sys.argv[1] == "--stdio-worker":
    _planner_stdio_worker_main(sys.argv[2], json.loads(sys.argv[3]))
