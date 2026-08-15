# Reference: https://github.com/fan-ziqi/robot_lab

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="ATEC-Isaac-Velocity-Flat-Unitree-B2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeB2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{ agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2FlatPPORunnerCfg"
    },
)

gym.register(
    id="ATEC-Isaac-Velocity-Rough-Unitree-B2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeB2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{ agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2RoughPPORunnerCfg"
    },
)

gym.register(
    id="ATEC-Isaac-Velocity-Heading-Rough-Unitree-B2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.heading_rough_env_cfg:UnitreeB2HeadingRoughEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2HeadingRoughPPORunnerCfg"
        ),
    },
)

gym.register(
    id="ATEC-Isaac-Velocity-Flat-Unitree-B2-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.piper_env_cfg:UnitreeB2PiperFlatEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2PiperFlatPPORunnerCfg"
        ),
    },
)

gym.register(
    id="ATEC-Isaac-Velocity-Heading-Rough-Unitree-B2-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.piper_env_cfg:UnitreeB2PiperHeadingRoughEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2PiperHeadingRoughPPORunnerCfg"
        ),
    },
)

gym.register(
    id="ATEC-Isaac-Velocity-Robust-Flat-Unitree-B2-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.robust_piper_env_cfg:UnitreeB2PiperRobustFlatEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2PiperRobustFlatPPORunnerCfg"
        ),
    },
)

gym.register(
    id="ATEC-Isaac-Velocity-Robust-Heading-Rough-Unitree-B2-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.robust_piper_env_cfg:UnitreeB2PiperRobustHeadingRoughEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2PiperRobustHeadingRoughPPORunnerCfg"
        ),
    },
)
