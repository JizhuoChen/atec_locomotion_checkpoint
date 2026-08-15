"""Heading-first rough-terrain locomotion configuration for Unitree B2."""

import math

from isaaclab.utils import configclass

import atec_rl_lab.train.locomotion.velocity.mdp as mdp
from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.rough_env_cfg import UnitreeB2RoughEnvCfg


def configure_heading_first(env_cfg: UnitreeB2RoughEnvCfg) -> None:
    """Apply the shared turn-first command, reward, and curriculum settings."""
    # Preserve the three-value actor command interface, but change its
    # semantics to [gated forward speed, 0 lateral speed, desired yaw rate].
    env_cfg.commands.base_velocity.heading_aligned = True
    env_cfg.commands.base_velocity.alignment_std = 0.55
    env_cfg.commands.base_velocity.resampling_time_range = (env_cfg.episode_length_s, env_cfg.episode_length_s)
    env_cfg.commands.base_velocity.rel_heading_envs = 1.0
    env_cfg.commands.base_velocity.heading_command = True
    env_cfg.commands.base_velocity.heading_control_stiffness = 1.0
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.25, 1.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
    env_cfg.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

    # Track horizontal velocity in the gravity-aligned yaw frame. The
    # alignment gate removes the incentive to stand still while misaligned;
    # explicit penalties suppress crabbing and premature translation.
    env_cfg.rewards.track_lin_vel_xy_exp.func = mdp.track_heading_aligned_forward_exp
    env_cfg.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp
    env_cfg.rewards.lateral_velocity_yaw_frame_l2.weight = -0.5
    env_cfg.rewards.misaligned_planar_motion_l2.weight = -0.5

    # A single heading lasts for the full episode, making final displacement
    # a meaningful terrain-progress measure. Promotion additionally requires
    # sustained alignment, so unrelated lateral drift cannot unlock harder rows.
    env_cfg.curriculum.terrain_levels.params["heading_alignment_threshold"] = 0.6


@configclass
class UnitreeB2HeadingRoughEnvCfg(UnitreeB2RoughEnvCfg):
    """Train B2 to turn toward a world-heading target before moving forward."""

    def __post_init__(self):
        super().__post_init__()
        configure_heading_first(self)

        if self.__class__.__name__ == "UnitreeB2HeadingRoughEnvCfg":
            self.disable_zero_weight_rewards()
