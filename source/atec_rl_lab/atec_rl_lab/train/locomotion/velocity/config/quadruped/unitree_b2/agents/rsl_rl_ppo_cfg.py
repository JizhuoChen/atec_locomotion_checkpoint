# Reference: https://github.com/fan-ziqi/robot_lab

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeB2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "unitree_b2_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UnitreeB2FlatPPORunnerCfg(UnitreeB2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "unitree_b2_flat"


@configclass
class UnitreeB2HeadingRoughPPORunnerCfg(UnitreeB2RoughPPORunnerCfg):
    """Conservative fine-tuning settings for the heading-first rough task."""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "unitree_b2_heading_rough"
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.entropy_coef = 0.005


@configclass
class UnitreeB2PiperFlatPPORunnerCfg(UnitreeB2FlatPPORunnerCfg):
    """Flat-ground embodiment adaptation for B2 with the mounted Piper arm."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_b2_piper_flat"


@configclass
class UnitreeB2PiperHeadingRoughPPORunnerCfg(UnitreeB2HeadingRoughPPORunnerCfg):
    """Heading-first rough-terrain fine-tuning for B2 with Piper."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_b2_piper_heading_rough"


@configclass
class UnitreeB2PiperRobustFlatPPORunnerCfg(UnitreeB2PiperFlatPPORunnerCfg):
    """Longer from-scratch adaptation for the robust B2-Piper profile."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 8000
        self.experiment_name = "unitree_b2_piper_robust_flat"


@configclass
class UnitreeB2PiperRobustHeadingRoughPPORunnerCfg(UnitreeB2PiperHeadingRoughPPORunnerCfg):
    """Longer actor-transfer fine-tuning on the robust terrain profile."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 12000
        self.experiment_name = "unitree_b2_piper_robust_heading_rough"
