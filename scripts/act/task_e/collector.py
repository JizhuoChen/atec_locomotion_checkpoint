"""Single-episode demo collection and success checking for Task E."""

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import matrix_from_quat
from atec_rl_lab.utils import CartesianController
from atec_rl_lab.tasks.task_e.env_cfg import (
    TABLE_TOP_Z,
    BASKET_CENTER_X, BASKET_CENTER_Y,
)

from .config import (
    ACTION_SCALE,
    GRIPPER_OPEN_POS, GRIPPER_CLOSE_POS,
    RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z,
    DEFAULT_PLACE_QUAT_W,
    BASKET_IN_X, BASKET_IN_Y,
    OBJ_UPRIGHT_AXIS_CHECKS,
    STATE_ORDER,
    POSITION_PRIORITY_GRASP_OBJECTS, POSITION_PRIORITY_GRASP_STATES,
    POSITION_PRIORITY_ORIENTATION_WEIGHT as DEFAULT_POSITION_PRIORITY_ORIENTATION_WEIGHT,
    WARMUP_STEPS, SETTLE_STEPS,
)
from .state_machine import PickPlaceStateMachine


def _valid_start_scene(env: ManagerBasedRLEnv, pick_objects: list[int]) -> bool:
    """Reject starts that are bad for scripted demonstrations."""
    for obj_idx in pick_objects:
        axis_check = OBJ_UPRIGHT_AXIS_CHECKS.get(obj_idx)
        if axis_check is None:
            continue
        axis_idx, min_vertical = axis_check
        quat = env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"].data.root_state_w[0, 3:7]
        rot = matrix_from_quat(quat.unsqueeze(0)).squeeze(0)
        vertical_alignment = torch.abs(rot[2, axis_idx]).item()
        print(
            f"[INFO] object_{obj_idx} start orientation local_axis_{axis_idx}_vertical="
            f"{vertical_alignment:.3f} (min valid {min_vertical:.3f})",
            flush=True,
        )
        if vertical_alignment < min_vertical:
            print(
                f"[WARN] object_{obj_idx} start pose is laid down "
                f"(vertical alignment {vertical_alignment:.3f} < {min_vertical:.3f}) - skipping scene.",
                flush=True,
            )
            return False
    return True


_BASKET_MAX_Z = TABLE_TOP_Z + 0.1   # object must be below this to count as inside

def check_objects_in_basket(env: ManagerBasedRLEnv, pick_objects: list[int]) -> bool:
    """Return True only if every picked object is inside the basket region and settled."""
    for obj_idx in pick_objects:
        pos = env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"].data.root_pos_w[0]
        if (abs(pos[0].item() - BASKET_CENTER_X) > BASKET_IN_X or
                abs(pos[1].item() - BASKET_CENTER_Y) > BASKET_IN_Y or
                pos[2].item() > _BASKET_MAX_Z):
            return False
    return True


def collect_one_demo(
    env:         ManagerBasedRLEnv,
    robot,
    ik_ctrl:     CartesianController,
    ik_pos_ctrl: CartesianController | None,
    arm_ids:     list[int],
    gripper_ids: list[int],
    pick_objects: list[int],
    device:      str,
    default_jpos: torch.Tensor,
    position_priority_orientation_weight: float | None = None,
    camera=None,
) -> dict | None:
    """Run one full episode and return recorded data, or None on early termination.

    Returns a dict with keys:
      qpos    (T, 8)        absolute joint positions
      qvel    (T, 8)        joint velocities
      ee_pos  (T, 3)        end-effector position (world frame)
      ee_quat (T, 4)        end-effector quaternion (w,x,y,z)
      action  (T, 8)        env action = (joint_target - default_jpos) / ACTION_SCALE
      target_ee_pos  (T, 3) commanded EE position (world frame)
      target_ee_quat (T, 4) commanded EE quaternion (w,x,y,z)
      target_error   (T, 3) actual EE position minus commanded EE position
      object_pos     (T, 3) current target object position (world frame)
      state_id       (T,)   index into task_e.config.STATE_ORDER
      object_idx     (T,)   target object index
      controller_mode (T,)  0 = pose IK, 1 = position IK fallback, 2 = weighted pose IK
      gripper_cmd    (T,)   0 = open, 1 = close
      position_priority_orientation_weight float, object-1 weighted-pose orientation weight
      frames  (T, H, W, 3)  RGB uint8 — only present when camera is given
    """
    orientation_weight = (
        DEFAULT_POSITION_PRIORITY_ORIENTATION_WEIGHT
        if position_priority_orientation_weight is None
        else float(position_priority_orientation_weight)
    )
    print("[INFO] Resetting env", flush=True)
    env.reset()
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos,
        torch.zeros_like(robot.data.default_joint_vel),
    )

    default_jpos = robot.data.default_joint_pos.clone()

    ee_home = torch.tensor([[RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z]],
                            dtype=torch.float32, device=device)
    eq_home = torch.tensor([DEFAULT_PLACE_QUAT_W], dtype=torch.float32, device=device)
    g_open  = torch.tensor([GRIPPER_OPEN_POS], dtype=torch.float32, device=device)

    robot.update(dt=env.unwrapped.physics_dt)
    ik_ctrl.reset()
    if ik_pos_ctrl is not None:
        ik_pos_ctrl.reset()

    # Warm-up: drive arm to HOME position (not recorded)
    print("[INFO] Warm-up to home", flush=True)
    for _ in range(WARMUP_STEPS):
        _step_to(env, robot, ik_ctrl, arm_ids, gripper_ids,
                 ee_home, eq_home, g_open, default_jpos)

    # Settle
    print("[INFO] Settling before validation", flush=True)
    for _ in range(SETTLE_STEPS):
        _step_to(env, robot, ik_ctrl, arm_ids, gripper_ids,
                 ee_home, eq_home, g_open, default_jpos)

    if not _valid_start_scene(env, pick_objects):
        return None

    # Pre-compute grasp quaternions from actual object orientations after reset.
    sm = PickPlaceStateMachine(pick_objects, device)
    for obj_idx in pick_objects:
        obj_quat = env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"] \
                       .data.root_state_w[0, 3:7]
        sm.set_grasp_quat(obj_idx, obj_quat)

    ik_ctrl.reset()
    if ik_pos_ctrl is not None:
        ik_pos_ctrl.reset()

    # ---- Recording loop ---- #
    print("[INFO] Recording scripted rollout", flush=True)
    qpos_buf, qvel_buf, ee_pos_buf, ee_quat_buf, action_buf = [], [], [], [], []
    target_ee_pos_buf, target_ee_quat_buf, target_error_buf, object_pos_buf = [], [], [], []
    state_id_buf, object_idx_buf, controller_mode_buf, gripper_cmd_buf = [], [], [], []
    frames_buf = [] if camera is not None else None

    while not sm.done:
        state_name = sm.state
        obj_idx = sm.current_object_idx
        obj_pos_w = env.unwrapped.scene.rigid_objects[sm.current_object_key] \
                        .data.root_pos_w[0].clone()
        ee_pos_des, ee_quat_des, gripper_cmd = sm.tick(obj_pos_w)

        use_position_priority = _use_position_priority(obj_idx, state_name, ik_pos_ctrl)
        if use_position_priority and hasattr(ik_ctrl, "compute_weighted_pose"):
            arm_jpos_des = ik_ctrl.compute_weighted_pose(
                ee_pos_des.unsqueeze(0),
                ee_quat_des.unsqueeze(0),
                orientation_weight=orientation_weight,
            )
            controller_mode = 2
        elif use_position_priority:
            arm_jpos_des = ik_pos_ctrl.compute(ee_pos_des.unsqueeze(0), None)
            controller_mode = 1
        else:
            arm_jpos_des = ik_ctrl.compute(ee_pos_des.unsqueeze(0), ee_quat_des.unsqueeze(0))
            controller_mode = 0
        gripper_vals   = GRIPPER_OPEN_POS if gripper_cmd == "open" else GRIPPER_CLOSE_POS
        gripper_target = torch.tensor([gripper_vals], dtype=torch.float32, device=device)

        full_target = robot.data.joint_pos.clone()
        full_target[:, arm_ids]     = arm_jpos_des
        full_target[:, gripper_ids] = gripper_target
        env_action = (full_target - default_jpos) / ACTION_SCALE

        # Record BEFORE stepping (obs at time t, action at time t)
        qpos_buf.append(robot.data.joint_pos[0].cpu().numpy())
        qvel_buf.append(robot.data.joint_vel[0].cpu().numpy())
        ee_pos_buf.append(ik_ctrl.ee_pos_w[0].cpu().numpy())
        ee_quat_buf.append(ik_ctrl.ee_quat_w[0].cpu().numpy())
        action_buf.append(env_action[0].cpu().numpy())
        target_ee_pos_buf.append(ee_pos_des.cpu().numpy())
        target_ee_quat_buf.append(ee_quat_des.cpu().numpy())
        target_error_buf.append((ik_ctrl.ee_pos_w[0] - ee_pos_des).cpu().numpy())
        object_pos_buf.append(obj_pos_w.cpu().numpy())
        state_id_buf.append(STATE_ORDER.index(state_name))
        object_idx_buf.append(obj_idx)
        controller_mode_buf.append(controller_mode)
        gripper_cmd_buf.append(1 if gripper_cmd == "close" else 0)
        if frames_buf is not None:
            rgba = camera.data.output["rgb"][0].cpu().numpy()
            frames_buf.append(rgba[:, :, :3])

        _, _, terminated, truncated, _ = env.step(env_action)

        if terminated.any() or truncated.any():
            print("[WARN] Episode ended early — skipping demo.")
            return None

    result = {
        "qpos":    np.stack(qpos_buf),
        "qvel":    np.stack(qvel_buf),
        "ee_pos":  np.stack(ee_pos_buf),
        "ee_quat": np.stack(ee_quat_buf),
        "action":  np.stack(action_buf),
        "target_ee_pos":  np.stack(target_ee_pos_buf),
        "target_ee_quat": np.stack(target_ee_quat_buf),
        "target_error":   np.stack(target_error_buf),
        "object_pos":     np.stack(object_pos_buf),
        "state_id":       np.asarray(state_id_buf, dtype=np.int16),
        "object_idx":     np.asarray(object_idx_buf, dtype=np.int16),
        "controller_mode": np.asarray(controller_mode_buf, dtype=np.int8),
        "gripper_cmd":    np.asarray(gripper_cmd_buf, dtype=np.int8),
        "position_priority_orientation_weight": orientation_weight,
    }
    if frames_buf is not None:
        result["frames"] = np.stack(frames_buf)
    return result


# ------------------------------------------------------------------ #
# Internal helper
# ------------------------------------------------------------------ #

def _use_position_priority(
    obj_idx: int, state_name: str, ik_pos_ctrl: CartesianController | None
) -> bool:
    """Return True when the current phase should use position-only IK."""
    return (
        ik_pos_ctrl is not None
        and obj_idx in POSITION_PRIORITY_GRASP_OBJECTS
        and state_name in POSITION_PRIORITY_GRASP_STATES
    )


def _step_to(env, robot, ik_ctrl, arm_ids, gripper_ids,
             ee_pos, ee_quat, gripper_target, default_jpos):
    """Single IK step toward a target pose (utility used during warm-up/settle)."""
    arm_des = ik_ctrl.compute(ee_pos, ee_quat)
    tgt = robot.data.joint_pos.clone()
    tgt[:, arm_ids]     = arm_des
    tgt[:, gripper_ids] = gripper_target
    env.step((tgt - default_jpos) / ACTION_SCALE)
    robot.update(dt=env.unwrapped.physics_dt)
