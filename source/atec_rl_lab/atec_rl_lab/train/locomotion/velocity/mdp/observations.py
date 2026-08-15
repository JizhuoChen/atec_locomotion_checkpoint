# Reference: https://github.com/fan-ziqi/robot_lab

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


def contact_forces_b(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return selected net normal contact-force vectors in the robot base frame.

    The contact sensor reports world-frame forces.  Rotating them into the base
    frame preserves the locomotion task's yaw invariance and gives one stable
    ``[fx, fy, fz]`` triplet per selected body.  Observation configuration is
    responsible for clipping and scaling the returned forces.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    robot: Articulation = env.scene[asset_cfg.name]
    forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    root_quat_w = robot.data.root_link_quat_w.unsqueeze(1).expand(-1, forces_w.shape[1], -1)
    forces_b = quat_apply_inverse(root_quat_w, forces_w)
    return forces_b.reshape(env.num_envs, -1)
