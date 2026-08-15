"""RSL-RL configurations for privileged teacher PPO and student distillation."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)

from .rsl_rl_ppo_cfg import UnitreeB2PiperRobustHeadingRoughPPORunnerCfg


@configclass
class UnitreeB2PiperPrivilegedTeacherPPORunnerCfg(
    UnitreeB2PiperRobustHeadingRoughPPORunnerCfg
):
    """Fine-tune an expanded privileged actor and critic with PPO."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 6000
        self.experiment_name = "unitree_b2_piper_privileged_teacher"
        # Prefix order is part of the checkpoint expansion contract:
        # actor = old 45 policy values, then 19 state values, then 12 forces;
        # critic = old 251 critic values, then the same 12 forces.
        self.obs_groups = {
            "policy": ["policy", "teacher_privileged", "contact_forces"],
            "critic": ["critic", "contact_forces"],
        }
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.entropy_coef = 0.002


@configclass
class UnitreeB2PiperStudentDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Distill the frozen 76-input teacher into the original 45-input MLP."""

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "unitree_b2_piper_student_distillation"
    obs_groups = {
        "policy": ["policy"],
        "teacher": ["policy", "teacher_privileged", "contact_forces"],
    }
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1.0e-4,
        gradient_length=15,
        max_grad_norm=1.0,
        optimizer="adam",
        loss_type="huber",
    )
