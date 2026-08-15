"""Opt-in sim-to-real B2-Piper locomotion environments."""

from __future__ import annotations

from copy import deepcopy

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import atec_rl_lab.train.locomotion.velocity.mdp as mdp
from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.piper_env_cfg import (
    UnitreeB2PiperFlatEnvCfg,
    UnitreeB2PiperHeadingRoughEnvCfg,
    configure_diverse_arm_motion,
)
from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.robust_terrain_cfg import (
    configure_robust_terrain,
)


B2_ONLY_PROXY_MASK_ATTR = "_b2_only_proxy_mask"
B2_ONLY_PROXY_FRACTION = 0.25
B2_ONLY_PROXY_MASS_SCALE_RANGE = (0.05, 0.10)


def _noise_with_episode_bias(
    white_range: tuple[float, float],
    bias_range: tuple[float, float],
) -> NoiseModelWithAdditiveBiasCfg:
    """Build bounded white noise plus a bias re-sampled independently at reset."""
    return NoiseModelWithAdditiveBiasCfg(
        noise_cfg=Unoise(n_min=white_range[0], n_max=white_range[1], operation="add"),
        # ``abs`` replaces the previous bias. ``add`` would create an unbounded
        # random walk each time an environment resets.
        bias_noise_cfg=Unoise(n_min=bias_range[0], n_max=bias_range[1], operation="abs"),
        sample_bias_per_component=True,
    )


def configure_robust_randomization(env_cfg, *, rough: bool) -> None:
    """Install conservative provisional sim-to-real randomization bands."""
    leg_joints = list(env_cfg.joint_names)
    arm_joints = list(env_cfg.arm_joint_names)

    # Preserve the action shape while modeling 0--20 ms controller latency,
    # joint-target calibration error, and small command jitter.
    action = env_cfg.actions.joint_pos
    env_cfg.actions.joint_pos = mdp.DelayedNoisyJointPositionActionCfg(
        asset_name=action.asset_name,
        joint_names=deepcopy(action.joint_names),
        scale=deepcopy(action.scale),
        offset=deepcopy(action.offset),
        preserve_order=action.preserve_order,
        use_default_offset=action.use_default_offset,
        clip=deepcopy(action.clip),
        debug_vis=action.debug_vis,
        min_delay=0,
        max_delay=1,
        target_bias_range=(-0.01, 0.01),
        target_noise_range=(-0.003, 0.003),
    )

    # Actor-only noise: the critic remains privileged. Values are in raw sensor
    # units and are applied before each observation term's configured scale.
    env_cfg.observations.policy.base_ang_vel.noise = _noise_with_episode_bias(
        (-0.10, 0.10), (-0.03, 0.03)
    )
    env_cfg.observations.policy.projected_gravity.noise = _noise_with_episode_bias(
        (-0.02, 0.02), (-0.01, 0.01)
    )
    env_cfg.observations.policy.joint_pos.noise = _noise_with_episode_bias(
        (-0.005, 0.005), (-0.01, 0.01)
    )
    env_cfg.observations.policy.joint_vel.noise = _noise_with_episode_bias(
        (-0.5, 0.5), (-0.1, 0.1)
    )

    events = env_cfg.events
    events.randomize_rigid_body_material.params.update(
        {
            "static_friction_range": (0.45, 1.25),
            "dynamic_friction_range": (0.35, 1.0),
            "restitution_range": (0.0, 0.08),
            "num_buckets": 64,
            "make_consistent": True,
        }
    )

    # Split chassis/leg and arm uncertainty so a tiny arm link is not subjected
    # to the old blanket 0.7--1.3 scaling used by every non-base body.
    events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-1.0, 3.0)
    events.randomize_rigid_body_mass_others.params.update(
        {
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[r"^(?!(?:base_link|arm_base|arm_link[1-8]|gripper_base)$).*"],
            ),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "recompute_inertia": True,
        }
    )
    events.randomize_arm_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[r"^(?:arm_base|arm_link[1-8]|gripper_base)$"]
            ),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    # A vectorized scene cannot mix different articulation topologies. Assign a
    # persistent low-payload subset after full Piper mass randomization; the arm
    # command term uses the same mask to park these clones at the compact stow.
    events.assign_b2_only_proxy_payload = EventTerm(
        func=mdp.assign_b2_only_proxy_payload,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[r"^(?:arm_base|arm_link[1-8]|gripper_base)$"]
            ),
            "proxy_fraction": B2_ONLY_PROXY_FRACTION,
            "proxy_mass_scale_range": B2_ONLY_PROXY_MASS_SCALE_RANGE,
            "mask_attribute": B2_ONLY_PROXY_MASK_ATTR,
        },
    )
    events.randomize_com_positions.params["com_range"] = {
        "x": (-0.03, 0.03),
        "y": (-0.03, 0.03),
        "z": (-0.02, 0.02),
    }

    # Fixed-per-environment joint mechanics approximate manufacturing and
    # calibration differences. Reset-time gain changes would make one simulated
    # robot change hardware identity every episode and require costly CPU writes.
    events.randomize_actuator_gains.mode = "startup"
    events.randomize_actuator_gains.params.update(
        {
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=leg_joints, preserve_order=True
            ),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        }
    )
    events.randomize_arm_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=arm_joints, preserve_order=True
            ),
            "stiffness_distribution_params": (0.7, 1.3),
            "damping_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    events.randomize_leg_joint_mechanics = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=leg_joints, preserve_order=True
            ),
            "friction_distribution_params": (0.5, 1.5),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    events.randomize_arm_joint_mechanics = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=arm_joints, preserve_order=True
            ),
            "friction_distribution_params": (0.5, 1.5),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    events.randomize_leg_effort_limits = EventTerm(
        func=mdp.randomize_joint_effort_limits,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=leg_joints, preserve_order=True
            ),
            "scale_range": (0.85, 1.10),
            "per_joint": False,
        },
    )

    events.randomize_reset_joints.func = mdp.reset_joints_by_offset
    events.randomize_reset_joints.params = {
        "position_range": (-0.03, 0.03),
        "velocity_range": (-0.10, 0.10),
        "asset_cfg": SceneEntityCfg("robot", joint_names=leg_joints, preserve_order=True),
    }
    events.randomize_apply_external_force_torque.params.update(
        {
            "force_range": (-10.0, 10.0) if rough else (-5.0, 5.0),
            "torque_range": (-3.0, 3.0) if rough else (-1.5, 1.5),
        }
    )
    # The rough profile uses an eight-second horizon, so its interval must be
    # shorter than the episode or pushes would silently never occur.
    events.randomize_push_robot.interval_range_s = (4.0, 8.0) if rough else (8.0, 15.0)
    events.randomize_push_robot.params["velocity_range"] = {
        "x": (-0.25, 0.25),
        "y": (-0.25, 0.25),
        "yaw": (-0.15, 0.15),
    }

    if rough:
        # Upright deployment starts are the actual objective; full +/-pi roll and
        # pitch reset ranges turn most samples into an unrelated self-righting task.
        events.randomize_reset_base.params.update(
            {
                "pose_range": {
                    "z": (0.0, 0.10),
                    "roll": (-0.15, 0.15),
                    "pitch": (-0.15, 0.15),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {
                    "x": (-0.20, 0.20),
                    "y": (-0.20, 0.20),
                    "z": (-0.15, 0.15),
                    "roll": (-0.20, 0.20),
                    "pitch": (-0.20, 0.20),
                    "yaw": (-0.20, 0.20),
                },
            }
        )
    else:
        events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.04),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.15, 0.15),
                "y": (-0.15, 0.15),
                "z": (-0.10, 0.10),
                "roll": (-0.15, 0.15),
                "pitch": (-0.15, 0.15),
                "yaw": (-0.15, 0.15),
            },
        }


@configclass
class UnitreeB2PiperRobustFlatEnvCfg(UnitreeB2PiperFlatEnvCfg):
    """Flat embodiment learning with deployable sim-to-real disturbances."""

    def __post_init__(self):
        super().__post_init__()
        configure_diverse_arm_motion(self, stationary_env_mask_attr=B2_ONLY_PROXY_MASK_ATTR)
        configure_robust_randomization(self, rough=False)
        self.disable_zero_weight_rewards()


@configclass
class UnitreeB2PiperRobustHeadingRoughEnvCfg(UnitreeB2PiperHeadingRoughEnvCfg):
    """Diverse terrain fine-tuning constrained to each assigned terrain tile."""

    def __post_init__(self):
        super().__post_init__()
        configure_diverse_arm_motion(self, stationary_env_mask_attr=B2_ONLY_PROXY_MASK_ATTR)
        configure_robust_terrain(self)
        self.scene.terrain.max_init_terrain_level = 4

        # Log the achieved level of every family independently. These terms do
        # not change the curriculum; they reveal whether a nominally supported
        # stair/riser combination actually received enough hard-level exposure.
        for terrain_name in self.scene.terrain.terrain_generator.sub_terrains:
            setattr(
                self.curriculum,
                f"terrain_level_{terrain_name}",
                CurrTerm(func=mdp.terrain_family_level, params={"terrain_names": (terrain_name,)}),
            )

        # At 0.25--1.0 m/s an eight-second local-locomotion episode exercises
        # several metres of terrain without permitting a 20 m cross-map walk.
        self.episode_length_s = 8.0
        self.commands.base_velocity.resampling_time_range = (8.0, 8.0)
        self.commands.arm_motion.resampling_time_range = (8.0, 8.0)
        # Every route points broadly through the tile interior. Spawn positions
        # and initial robot yaw are independently random, so the policy still
        # sees the full range of relative turn commands without sacrificing
        # terrain exposure to outward-pointing trajectories.
        self.commands.base_velocity.terrain_center_heading_probability = 1.0
        self.commands.base_velocity.terrain_center_heading_max_deviation = 0.35
        self.commands.base_velocity.track_terrain_exposure = True
        self.commands.base_velocity.terrain_tile_buffer = 0.55
        self.commands.base_velocity.terrain_relief_threshold = 0.025

        # Keep Isaac Lab's map-wide check as a final safety net. This additional
        # timeout prevents reward collection on a neighbouring tile and keeps the
        # full B2 footprint inside its assigned workspace.
        self.terminations.assigned_terrain_out_of_bounds = DoneTerm(
            func=mdp.assigned_terrain_out_of_bounds,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "distance_buffer": 0.55,
            },
            time_out=True,
        )

        # Promotion/demotion uses feasible signed progress inside the same
        # buffered tile. Hold commands are intentionally neutral to difficulty.
        self.curriculum.terrain_levels.params.update(
            {
                "assigned_tile_path_aware": True,
                "assigned_tile_buffer": 0.55,
                "path_promotion_fraction": 0.9,
                "path_demotion_fraction": 0.5,
                "path_minimum_target_distance": 1.0,
                "path_max_lateral_fraction": 0.5,
            }
        )

        configure_robust_randomization(self, rough=True)
        self.disable_zero_weight_rewards()
