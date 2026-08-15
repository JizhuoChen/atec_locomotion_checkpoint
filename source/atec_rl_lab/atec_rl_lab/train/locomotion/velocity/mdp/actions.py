"""Action terms used by the locomotion sim-to-real training profiles."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import DelayBuffer, configclass


class DelayedNoisyJointPositionAction(JointPositionAction):
    """Apply a per-environment delayed joint target with bounded controller error.

    Delay is expressed in policy steps, not physics steps.  The action buffer stores
    raw policy actions so its zero-filled reset state maps to the robot's configured
    default joint pose instead of to a zero-radian joint target.
    """

    cfg: DelayedNoisyJointPositionActionCfg

    def __init__(self, cfg: DelayedNoisyJointPositionActionCfg, env):
        if cfg.min_delay < 0 or cfg.max_delay < cfg.min_delay:
            raise ValueError(
                "Action delay must satisfy 0 <= min_delay <= max_delay; "
                f"received ({cfg.min_delay}, {cfg.max_delay})."
            )
        if cfg.target_bias_range[0] > cfg.target_bias_range[1]:
            raise ValueError(f"Invalid target_bias_range: {cfg.target_bias_range}")
        if cfg.target_noise_range[0] > cfg.target_noise_range[1]:
            raise ValueError(f"Invalid target_noise_range: {cfg.target_noise_range}")

        super().__init__(cfg, env)
        self._delay_buffer = DelayBuffer(cfg.max_delay, self.num_envs, device=self.device)
        self._target_bias = torch.zeros_like(self._raw_actions)
        self._ALL_INDICES = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.reset()

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        delayed_actions = self._delay_buffer.compute(self._raw_actions)
        self._processed_actions = delayed_actions * self._scale + self._offset + self._target_bias

        if self.cfg.target_noise_range != (0.0, 0.0):
            self._processed_actions += torch.empty_like(self._processed_actions).uniform_(
                *self.cfg.target_noise_range
            )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            resolved_env_ids = self._ALL_INDICES
        elif isinstance(env_ids, slice):
            resolved_env_ids = self._ALL_INDICES[env_ids]
        else:
            resolved_env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        super().reset(resolved_env_ids)
        self._delay_buffer.reset(resolved_env_ids)
        time_lags = torch.randint(
            self.cfg.min_delay,
            self.cfg.max_delay + 1,
            (len(resolved_env_ids),),
            dtype=torch.int,
            device=self.device,
        )
        self._delay_buffer.set_time_lag(time_lags, resolved_env_ids)
        # Advanced indexing returns a copy, so sample into a temporary tensor
        # and assign it back instead of calling ``uniform_`` on the indexed view.
        self._target_bias[resolved_env_ids] = torch.empty(
            (len(resolved_env_ids), self.action_dim), device=self.device
        ).uniform_(*self.cfg.target_bias_range)


@configclass
class DelayedNoisyJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for delayed, imperfect position targets."""

    class_type: type = DelayedNoisyJointPositionAction

    min_delay: int = 0
    """Minimum policy-step delay sampled independently at reset."""

    max_delay: int = 1
    """Maximum policy-step delay sampled independently at reset."""

    target_bias_range: tuple[float, float] = (-0.01, 0.01)
    """Per-episode, per-joint position-target bias in radians."""

    target_noise_range: tuple[float, float] = (-0.003, 0.003)
    """Per-policy-step position-target jitter in radians."""
