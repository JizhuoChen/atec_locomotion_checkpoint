"""Transfer-compatible locomotion tasks for Unitree B2 with a Piper arm."""

from copy import deepcopy

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import atec_rl_lab.train.locomotion.velocity.mdp as mdp
from atec_rl_lab.assets.robots import UNITREE_B2_PIPER_CFG
from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.heading_rough_env_cfg import (
    configure_heading_first,
)
from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.rough_env_cfg import UnitreeB2RoughEnvCfg


ARM_STOW_JOINT_POS = {
    "arm_joint1": 0.0,
    "arm_joint2": 0.75,
    "arm_joint3": -1.15,
    "arm_joint4": 0.0,
    "arm_joint5": 0.40,
    "arm_joint6": 0.0,
    "arm_joint7": 0.02,
    "arm_joint8": -0.02,
}
"""A bent, half-open nominal pose safely inside every Piper joint limit."""


def configure_diverse_arm_motion(env_cfg, *, stationary_env_mask_attr: str | None = None) -> None:
    """Opt into broad, smooth, limit-safe Piper payload motion."""
    env_cfg.commands.arm_motion = mdp.SinusoidalArmCommandCfg(
        asset_name="robot",
        joint_names=env_cfg.arm_joint_names,
        resampling_time_range=(env_cfg.episode_length_s, env_cfg.episode_length_s),
        center_offset_range=(-0.12, 0.12),
        center_offset_scales=(1.0, 0.8, 0.8, 0.5, 0.5, 0.4, 0.0, 0.0),
        amplitude_range=(0.03, 0.08),
        amplitude_scales=(1.0, 0.8, 0.8, 0.5, 0.5, 0.4, 0.0, 0.0),
        frequency_range=(0.05, 0.12),
        stationary_probability=0.08,
        stationary_env_mask_attr=stationary_env_mask_attr,
        limit_margin_fraction=0.08,
        trajectory_mode="smooth_waypoint",
        # Offsets are relative to ARM_STOW_JOINT_POS. Joint 2 remains
        # positive and joint 3 negative so the arm stays bent above the body;
        # wrist variation changes payload inertia without extreme poses.
        waypoint_offset_ranges=(
            (-0.50, 0.50),
            (-0.22, 0.40),
            (-0.40, 0.25),
            (-0.35, 0.35),
            (-0.22, 0.28),
            (-0.45, 0.45),
            (0.0, 0.0),
            (0.0, 0.0),
        ),
        segment_duration_range=(1.75, 4.50),
        pause_probability=0.18,
        pause_duration_range=(0.50, 1.80),
        max_joint_speeds=(0.75, 0.65, 0.65, 0.80, 0.80, 0.90, 0.10, 0.10),
        debug_vis=False,
    )


@configclass
class UnitreeB2PiperRoughEnvCfg(UnitreeB2RoughEnvCfg):
    """B2 rough locomotion with a Piper payload and a 45-to-12 leg actor.

    The actor receives only base/leg state and controls only the twelve leg
    joints, preserving exact compatibility with bare-B2 actor checkpoints. The
    critic observes all twenty joints, including the moving arm and gripper.
    """

    arm_joint_names = [
        "arm_joint1",
        "arm_joint2",
        "arm_joint3",
        "arm_joint4",
        "arm_joint5",
        "arm_joint6",
        "arm_joint7",
        "arm_joint8",
    ]

    def __post_init__(self):
        super().__post_init__()

        robot_cfg = deepcopy(UNITREE_B2_PIPER_CFG)
        robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
        robot_cfg.init_state.joint_pos.update(ARM_STOW_JOINT_POS)
        self.scene.robot = robot_cfg

        # Actor: leg-only proprioception and leg-only actions (45 observations,
        # 12 actions). Critic: all leg and arm joint states (251 observations on
        # rough terrain, 64 on flat terrain).
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = [
            *self.joint_names,
            *self.arm_joint_names,
        ]
        self.observations.critic.joint_pos.params["asset_cfg"].preserve_order = True
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = [
            *self.joint_names,
            *self.arm_joint_names,
        ]
        self.observations.critic.joint_vel.params["asset_cfg"].preserve_order = True
        self.actions.joint_pos.joint_names = self.joint_names

        # Preserve the original small sinusoidal disturbance for the baseline
        # task. The opt-in robust profile replaces it with diverse waypoints via
        # ``configure_diverse_arm_motion``.
        self.commands.arm_motion = mdp.SinusoidalArmCommandCfg(
            asset_name="robot",
            joint_names=self.arm_joint_names,
            resampling_time_range=(self.episode_length_s, self.episode_length_s),
            center_offset_range=(-0.12, 0.12),
            center_offset_scales=(1.0, 0.8, 0.8, 0.5, 0.5, 0.4, 0.0, 0.0),
            amplitude_range=(0.03, 0.08),
            amplitude_scales=(1.0, 0.8, 0.8, 0.5, 0.5, 0.4, 0.0, 0.0),
            frequency_range=(0.05, 0.12),
            stationary_probability=0.2,
            debug_vis=False,
        )

        # Remove joint-indexed energy, limit, and pose costs for motion the actor
        # cannot control. The inherited non-foot contact term intentionally still
        # treats an arm/ground strike as a fall-safety penalty; the upright swept
        # arm profile does not contact the ground in the simulator probe.
        leg_only_reward_terms = (
            self.rewards.joint_torques_l2,
            self.rewards.joint_vel_l2,
            self.rewards.joint_acc_l2,
            self.rewards.joint_pos_limits,
            self.rewards.joint_vel_limits,
            self.rewards.joint_power,
            self.rewards.stand_still,
            self.rewards.joint_pos_penalty,
            self.rewards.applied_torque_limits,
        )
        for reward_term in leg_only_reward_terms:
            if reward_term is not None and "asset_cfg" in reward_term.params:
                reward_term.params["asset_cfg"] = SceneEntityCfg(
                    "robot", joint_names=self.joint_names, preserve_order=True
                )

        if self.__class__.__name__ == "UnitreeB2PiperRoughEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class UnitreeB2PiperFlatEnvCfg(UnitreeB2PiperRoughEnvCfg):
    """Original body-frame velocity task on flat ground for B2-Piper."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

        # The rough task uses terrain patches for spread spawning. A plane has
        # no generated patches, so restore the standard independent flat reset.
        self.events.randomize_reset_base.func = mdp.reset_root_state_uniform
        self.events.randomize_reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }

        if self.__class__.__name__ == "UnitreeB2PiperFlatEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class UnitreeB2PiperHeadingRoughEnvCfg(UnitreeB2PiperRoughEnvCfg):
    """Heading-first rough-terrain fine-tuning for the arm-mounted B2."""

    def __post_init__(self):
        super().__post_init__()
        configure_heading_first(self)

        if self.__class__.__name__ == "UnitreeB2PiperHeadingRoughEnvCfg":
            self.disable_zero_weight_rewards()
