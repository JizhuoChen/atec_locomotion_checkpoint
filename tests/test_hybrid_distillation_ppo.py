"""Pure-Python tests for the additive hybrid PPO-distillation path."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic

from atec_rl_lab.train.locomotion.velocity.hybrid_distillation_ppo import (
    FrozenPrivilegedTeacherActor,
    HybridDistillationPPO,
    initialize_hybrid_from_checkpoints,
)


OBS_GROUPS = {
    "policy": ["policy"],
    "critic": ["critic", "contact_forces"],
    "teacher": ["policy", "teacher_privileged", "contact_forces", "teacher_height_scan"],
}


def make_observations(num_envs: int) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(num_envs, 45),
            "critic": torch.randn(num_envs, 251),
            "teacher_privileged": torch.randn(num_envs, 19),
            "contact_forces": torch.randn(num_envs, 12),
            "teacher_height_scan": torch.randn(num_envs, 187),
        },
        batch_size=[num_envs],
    )


def make_policy_and_teacher(num_envs: int = 8) -> tuple[ActorCritic, FrozenPrivilegedTeacherActor, TensorDict]:
    obs = make_observations(num_envs)
    policy = ActorCritic(
        obs,
        OBS_GROUPS,
        12,
        actor_hidden_dims=[32, 16],
        critic_hidden_dims=[32, 16],
        init_noise_std=0.1,
    )
    teacher = FrozenPrivilegedTeacherActor(obs, OBS_GROUPS["teacher"], 12, [32, 16], "elu")
    return policy, teacher, obs


def test_hybrid_update_uses_frozen_teacher_and_trainable_critic() -> None:
    torch.manual_seed(7)
    policy, teacher, obs = make_policy_and_teacher()
    teacher.mark_loaded()
    algorithm = HybridDistillationPPO(
        policy,
        teacher,
        distillation_coef_start=0.5,
        distillation_coef_end=0.1,
        distillation_decay_iterations=2,
        num_learning_epochs=1,
        num_mini_batches=2,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1.0e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
    )
    algorithm.init_storage("rl", 8, 4, obs, [12])

    teacher_before = {name: value.clone() for name, value in teacher.state_dict().items()}
    critic_before = {name: value.clone() for name, value in policy.critic.state_dict().items()}
    for _ in range(4):
        actions = algorithm.act(obs)
        next_obs = make_observations(8)
        algorithm.process_env_step(
            next_obs,
            torch.randn(8, 1),
            torch.zeros(8, 1, dtype=torch.bool),
            {},
        )
        obs = next_obs
    algorithm.compute_returns(obs)
    losses = algorithm.update()

    assert losses["distillation_coefficient"] == 0.5
    assert losses["distillation"] >= 0.0
    assert losses["teacher_student_action_rmse"] >= 0.0
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert all(torch.equal(value, teacher.state_dict()[name]) for name, value in teacher_before.items())
    assert any(not torch.equal(value, policy.critic.state_dict()[name]) for name, value in critic_before.items())


def test_checkpoint_initialization_preserves_student_actor_and_warm_starts_critic(tmp_path) -> None:
    torch.manual_seed(11)
    source_student, _, obs = make_policy_and_teacher(num_envs=2)
    source_teacher = ActorCritic(
        obs,
        {"policy": OBS_GROUPS["teacher"], "critic": OBS_GROUPS["critic"]},
        12,
        actor_hidden_dims=[32, 16],
        critic_hidden_dims=[32, 16],
        init_noise_std=0.1,
    )
    student_state = {
        **{f"student.{name}": value.clone() for name, value in source_student.actor.state_dict().items()},
        "std": source_student.std.detach().clone(),
    }
    student_path = tmp_path / "model_1999.pt"
    teacher_path = tmp_path / "model_5999.pt"
    torch.save({"model_state_dict": student_state, "iter": 1999}, student_path)
    torch.save({"model_state_dict": source_teacher.state_dict(), "iter": 5999}, teacher_path)

    target_policy, target_teacher, _ = make_policy_and_teacher(num_envs=2)
    metadata = initialize_hybrid_from_checkpoints(
        target_policy,
        target_teacher,
        str(student_path),
        str(teacher_path),
    )

    assert metadata["student_actor_input_dim"] == 45
    assert metadata["teacher_actor_input_dim"] == 263
    assert metadata["student_critic_input_dim"] == 263
    assert target_teacher.loaded
    assert all(not parameter.requires_grad for parameter in target_teacher.parameters())
    for name, value in source_student.actor.state_dict().items():
        assert torch.equal(value, target_policy.actor.state_dict()[name])
    for name, value in source_teacher.critic.state_dict().items():
        assert torch.equal(value, target_policy.critic.state_dict()[name])
    for name, value in source_teacher.actor.state_dict().items():
        assert torch.equal(value, target_teacher.actor.state_dict()[name])
