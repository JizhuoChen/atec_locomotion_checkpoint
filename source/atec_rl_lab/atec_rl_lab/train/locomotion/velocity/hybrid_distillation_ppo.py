"""Frozen-teacher PPO with an action-distillation regularizer.

This module deliberately builds on RSL-RL's PPO implementation.  The deployed
actor remains an ordinary ``ActorCritic`` actor; a privileged teacher exists
only inside the training algorithm and is stored separately in hybrid
checkpoints so it cannot accidentally become part of the exported policy.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP
from rsl_rl.runners import OnPolicyRunner


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model_state(checkpoint_path: str) -> tuple[str, dict[str, torch.Tensor], int]:
    resolved_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {resolved_path}")
    checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no model_state_dict: {resolved_path}")
    return resolved_path, state, int(checkpoint.get("iter", -1))


def _prefixed_state(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {name.removeprefix(prefix): value for name, value in state.items() if name.startswith(prefix)}


def _load_matching_module_state(
    module: nn.Module,
    source_state: dict[str, torch.Tensor],
    *,
    label: str,
) -> None:
    target_state = module.state_dict()
    missing = sorted(set(target_state) - set(source_state))
    unexpected = sorted(set(source_state) - set(target_state))
    mismatched = sorted(
        name
        for name in set(target_state) & set(source_state)
        if target_state[name].shape != source_state[name].shape
    )
    if missing or unexpected or mismatched:
        mismatch_shapes = {
            name: (tuple(source_state[name].shape), tuple(target_state[name].shape))
            for name in mismatched
        }
        raise ValueError(
            f"{label} checkpoint is incompatible. Missing={missing}; unexpected={unexpected}; "
            f"shape_mismatches={mismatch_shapes}."
        )
    module.load_state_dict(source_state, strict=True)


class FrozenPrivilegedTeacherActor(nn.Module):
    """Deterministic teacher actor over explicitly selected observation groups."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: list[str],
        num_actions: int,
        hidden_dims: list[int] | tuple[int, ...],
        activation: str,
    ) -> None:
        super().__init__()
        self.obs_groups = tuple(obs_groups)
        input_dim = 0
        for group in self.obs_groups:
            if group not in obs:
                raise ValueError(f"Teacher observation group is unavailable: {group}")
            if len(obs[group].shape) != 2:
                raise ValueError(f"Teacher observation group must be 2-D: {group}={tuple(obs[group].shape)}")
            input_dim += int(obs[group].shape[-1])
        self.input_dim = input_dim
        self.num_actions = int(num_actions)
        self.actor = MLP(input_dim, num_actions, hidden_dims, activation)
        self.loaded = False
        self.freeze()

    def get_observations(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.obs_groups], dim=-1)

    def forward(self, obs: TensorDict) -> torch.Tensor:
        return self.actor(self.get_observations(obs))

    def freeze(self) -> None:
        self.requires_grad_(False)
        self.eval()

    def mark_loaded(self) -> None:
        self.loaded = True
        self.freeze()


def initialize_hybrid_from_checkpoints(
    policy: ActorCritic,
    teacher_actor: FrozenPrivilegedTeacherActor,
    student_checkpoint_path: str,
    teacher_checkpoint_path: str,
) -> dict[str, Any]:
    """Initialize the 45-D actor, trainable critic, and frozen privileged teacher.

    The student actor and action-noise parameter come from a distillation
    checkpoint.  A normal PPO actor checkpoint is accepted as a convenience
    for the PPO-only ablation.  The new critic is warm-started from the teacher
    critic but remains part of ``policy`` and is optimized by PPO immediately.
    """

    student_path, student_state, student_iteration = _load_model_state(student_checkpoint_path)
    teacher_path, teacher_state, teacher_iteration = _load_model_state(teacher_checkpoint_path)

    if any(name.startswith("student.") for name in student_state):
        student_actor_state = _prefixed_state(student_state, "student.")
        student_source = "distillation_student"
    elif any(name.startswith("actor.") for name in student_state):
        student_actor_state = _prefixed_state(student_state, "actor.")
        student_source = "ppo_actor"
    else:
        raise ValueError(
            "Student checkpoint must contain either student.* (distillation) or actor.* (PPO) parameters."
        )
    _load_matching_module_state(policy.actor, student_actor_state, label="student actor")

    noise_parameter = None
    for name in ("std", "log_std"):
        target = getattr(policy, name, None)
        source = student_state.get(name)
        if target is None or source is None:
            continue
        if target.shape != source.shape:
            raise ValueError(
                f"Student action-noise shape mismatch for {name}: "
                f"source={tuple(source.shape)}, target={tuple(target.shape)}."
            )
        with torch.no_grad():
            target.copy_(source.to(device=target.device, dtype=target.dtype))
        noise_parameter = name
        break
    if noise_parameter is None:
        raise ValueError("Student checkpoint does not provide the action-noise parameter expected by the PPO actor.")

    teacher_actor_state = _prefixed_state(teacher_state, "actor.")
    teacher_critic_state = _prefixed_state(teacher_state, "critic.")
    if not teacher_actor_state or not teacher_critic_state:
        raise ValueError("Teacher checkpoint must contain both actor.* and critic.* parameters.")
    _load_matching_module_state(teacher_actor.actor, teacher_actor_state, label="privileged teacher actor")
    _load_matching_module_state(policy.critic, teacher_critic_state, label="student PPO critic warm start")
    teacher_actor.mark_loaded()

    if not all(parameter.requires_grad for parameter in policy.critic.parameters()):
        raise RuntimeError("The hybrid student critic must remain trainable.")

    actor_input_dim = int(policy.actor.state_dict()["0.weight"].shape[1])
    critic_input_dim = int(policy.critic.state_dict()["0.weight"].shape[1])
    metadata = {
        "student_checkpoint": student_path,
        "student_checkpoint_sha256": _sha256_file(student_path),
        "student_checkpoint_iteration": student_iteration,
        "student_checkpoint_source": student_source,
        "student_actor_input_dim": actor_input_dim,
        "student_action_noise_parameter": noise_parameter,
        "teacher_checkpoint": teacher_path,
        "teacher_checkpoint_sha256": _sha256_file(teacher_path),
        "teacher_checkpoint_iteration": teacher_iteration,
        "teacher_actor_input_dim": teacher_actor.input_dim,
        "student_critic_input_dim": critic_input_dim,
        "critic_initialization": "teacher critic warm start; trainable from first PPO update",
        "teacher_status": "frozen; deterministic labels only",
        "optimizer": "fresh",
        "iteration_counter": "fresh",
    }
    print(
        "[INFO]: Initialized hybrid PPO-distillation. "
        f"Student actor input={actor_input_dim}, frozen teacher input={teacher_actor.input_dim}, "
        f"trainable critic input={critic_input_dim}."
    )
    return metadata


class HybridDistillationPPO(PPO):
    """PPO on student rollouts plus deterministic teacher-mean regularization."""

    def __init__(
        self,
        policy: ActorCritic,
        teacher_actor: FrozenPrivilegedTeacherActor,
        distillation_coef_start: float = 0.5,
        distillation_coef_end: float = 0.1,
        distillation_decay_iterations: int = 1500,
        distillation_loss_type: str = "huber",
        **ppo_kwargs: Any,
    ) -> None:
        if policy.is_recurrent:
            raise ValueError("HybridDistillationPPO currently supports only the unchanged feed-forward student actor.")
        if distillation_coef_start < 0.0 or distillation_coef_end < 0.0:
            raise ValueError("Distillation coefficients must be non-negative.")
        if distillation_decay_iterations < 0:
            raise ValueError("distillation_decay_iterations must be non-negative.")
        if distillation_loss_type not in {"huber", "mse"}:
            raise ValueError("distillation_loss_type must be 'huber' or 'mse'.")
        if ppo_kwargs.get("rnd_cfg") is not None or ppo_kwargs.get("symmetry_cfg") is not None:
            raise ValueError("RND and symmetry extensions are not enabled for the first hybrid experiment.")

        super().__init__(policy, **ppo_kwargs)
        self.teacher_actor = teacher_actor.to(self.device)
        self.teacher_actor.freeze()
        self.distillation_coef_start = float(distillation_coef_start)
        self.distillation_coef_end = float(distillation_coef_end)
        self.distillation_decay_iterations = int(distillation_decay_iterations)
        self.distillation_loss_type = distillation_loss_type
        self.distillation_updates = 0

    @property
    def distillation_coefficient(self) -> float:
        if self.distillation_decay_iterations == 0:
            return self.distillation_coef_end
        progress = min(self.distillation_updates / self.distillation_decay_iterations, 1.0)
        return self.distillation_coef_start + progress * (
            self.distillation_coef_end - self.distillation_coef_start
        )

    def hybrid_state_dict(self) -> dict[str, Any]:
        return {
            "distillation_updates": self.distillation_updates,
            "distillation_coef_start": self.distillation_coef_start,
            "distillation_coef_end": self.distillation_coef_end,
            "distillation_decay_iterations": self.distillation_decay_iterations,
            "distillation_loss_type": self.distillation_loss_type,
        }

    def load_hybrid_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "distillation_coef_start": self.distillation_coef_start,
            "distillation_coef_end": self.distillation_coef_end,
            "distillation_decay_iterations": self.distillation_decay_iterations,
            "distillation_loss_type": self.distillation_loss_type,
        }
        mismatches = {key: (state.get(key), value) for key, value in expected.items() if state.get(key) != value}
        if mismatches:
            raise ValueError(f"Hybrid resume configuration differs from the checkpoint: {mismatches}")
        self.distillation_updates = int(state["distillation_updates"])

    def update(self) -> dict[str, float]:
        if not self.teacher_actor.loaded:
            raise RuntimeError("Frozen teacher parameters were not loaded before hybrid PPO training.")
        if self.storage is None:
            raise RuntimeError("PPO rollout storage was not initialized.")

        coefficient = self.distillation_coefficient
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_distillation_loss = 0.0
        mean_action_rmse = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1.0e-8
                    )

            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(
                obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1]
            )
            student_mean_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - student_mean_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                        elif 0.0 < kl_mean < self.desired_kl / 2.0:
                            self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # ``no_grad`` is intentional here.  An inference-mode tensor cannot
            # be retained by Huber/MSE autograd as a constant target.
            with torch.no_grad():
                teacher_mean_batch = self.teacher_actor(obs_batch)
            if self.distillation_loss_type == "huber":
                distillation_loss = F.huber_loss(student_mean_batch, teacher_mean_batch)
            else:
                distillation_loss = F.mse_loss(student_mean_batch, teacher_mean_batch)

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + coefficient * distillation_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_distillation_loss += distillation_loss.item()
            mean_action_rmse += torch.sqrt(F.mse_loss(student_mean_batch, teacher_mean_batch)).item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        self.distillation_updates += 1
        return {
            "value_function": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "distillation": mean_distillation_loss / num_updates,
            "teacher_student_action_rmse": mean_action_rmse / num_updates,
            "distillation_coefficient": coefficient,
        }

    def broadcast_parameters(self) -> None:
        super().broadcast_parameters()
        teacher_payload = [self.teacher_actor.state_dict(), self.teacher_actor.loaded]
        torch.distributed.broadcast_object_list(teacher_payload, src=0)
        self.teacher_actor.load_state_dict(teacher_payload[0], strict=True)
        self.teacher_actor.loaded = bool(teacher_payload[1])
        self.teacher_actor.freeze()


class HybridOnPolicyRunner(OnPolicyRunner):
    """On-policy runner that owns a separately checkpointed frozen teacher."""

    def _construct_algorithm(self, obs: TensorDict) -> HybridDistillationPPO:
        policy_class_name = self.policy_cfg.pop("class_name")
        if policy_class_name != "ActorCritic":
            raise ValueError(
                "HybridOnPolicyRunner preserves the existing feed-forward actor and requires class_name='ActorCritic'."
            )
        policy = ActorCritic(obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg).to(self.device)

        algorithm_class_name = self.alg_cfg.pop("class_name")
        if algorithm_class_name != "HybridDistillationPPO":
            raise ValueError("HybridOnPolicyRunner requires class_name='HybridDistillationPPO'.")
        teacher_hidden_dims = self.alg_cfg.pop("teacher_hidden_dims")
        teacher_activation = self.alg_cfg.pop("teacher_activation")
        teacher_actor = FrozenPrivilegedTeacherActor(
            obs,
            self.cfg["obs_groups"]["teacher"],
            self.env.num_actions,
            teacher_hidden_dims,
            teacher_activation,
        ).to(self.device)
        algorithm = HybridDistillationPPO(
            policy,
            teacher_actor,
            device=self.device,
            multi_gpu_cfg=self.multi_gpu_cfg,
            **self.alg_cfg,
        )
        algorithm.init_storage("rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])
        return algorithm

    def initialize_from_checkpoints(
        self, student_checkpoint_path: str, teacher_checkpoint_path: str
    ) -> dict[str, Any]:
        return initialize_hybrid_from_checkpoints(
            self.alg.policy,
            self.alg.teacher_actor,
            student_checkpoint_path,
            teacher_checkpoint_path,
        )

    def save(self, path: str, infos: dict | None = None) -> None:
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "teacher_state_dict": self.alg.teacher_actor.state_dict(),
            "hybrid_state_dict": self.alg.hybrid_state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict | None:
        checkpoint = torch.load(path, weights_only=False, map_location=map_location)
        self.alg.policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
        teacher_state = checkpoint.get("teacher_state_dict")
        hybrid_state = checkpoint.get("hybrid_state_dict")
        if teacher_state is None or hybrid_state is None:
            raise ValueError(
                "This is not a resumable hybrid checkpoint: teacher_state_dict or hybrid_state_dict is missing."
            )
        self.alg.teacher_actor.load_state_dict(teacher_state, strict=True)
        self.alg.teacher_actor.mark_loaded()
        self.alg.load_hybrid_state_dict(hybrid_state)
        if load_optimizer:
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_learning_iteration = int(checkpoint["iter"])
        return checkpoint.get("infos")

    def train_mode(self) -> None:
        super().train_mode()
        self.alg.teacher_actor.eval()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.alg.teacher_actor.eval()
