"""Configurations for student PPO fine-tuning with a frozen privileged teacher."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlHybridDistillationPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO parameters plus deterministic teacher-mean regularization."""

    class_name: str = "HybridDistillationPPO"
    teacher_hidden_dims: list[int] = [512, 256, 128]
    teacher_activation: str = "elu"
    distillation_coef_start: float = 0.5
    distillation_coef_end: float = 0.1
    distillation_decay_iterations: int = 1500
    distillation_loss_type: str = "huber"


@configclass
class UnitreeB2PiperHeightScanHybridDistillationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Student-controlled PPO with a decaying frozen-teacher regularizer."""

    class_name = "HybridOnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "unitree_b2_piper_student_hybrid_ppo_heightscan"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic", "contact_forces"],
        "teacher": ["policy", "teacher_privileged", "contact_forces", "teacher_height_scan"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlHybridDistillationPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        distillation_coef_start=0.5,
        distillation_coef_end=0.1,
        distillation_decay_iterations=1500,
        distillation_loss_type="huber",
    )


@configclass
class UnitreeB2PiperHeightScanHybridFixedPPORunnerCfg(
    UnitreeB2PiperHeightScanHybridDistillationPPORunnerCfg
):
    """Ablation with a fixed teacher regularizer throughout PPO."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_b2_piper_student_hybrid_ppo_fixed_heightscan"
        self.algorithm.distillation_coef_start = 0.25
        self.algorithm.distillation_coef_end = 0.25
        self.algorithm.distillation_decay_iterations = 0


@configclass
class UnitreeB2PiperHeightScanStudentPPOOnlyRunnerCfg(
    UnitreeB2PiperHeightScanHybridDistillationPPORunnerCfg
):
    """Ablation with student PPO and no teacher action regularization."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_b2_piper_student_ppo_only_heightscan"
        self.algorithm.distillation_coef_start = 0.0
        self.algorithm.distillation_coef_end = 0.0
        self.algorithm.distillation_decay_iterations = 0
