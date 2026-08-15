# Reference: https://github.com/fan-ziqi/robot_lab

from isaaclab.utils import configclass

import atec_rl_lab.train.locomotion.velocity.mdp as mdp
from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.rough_env_cfg import UnitreeB2RoughEnvCfg


@configclass
class UnitreeB2FlatEnvCfg(UnitreeB2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        # Rough training uses generated flat patches for spread spawning. A
        # plane has no patch sampler, so restore the standard flat reset.
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

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeB2FlatEnvCfg":
            self.disable_zero_weight_rewards()
