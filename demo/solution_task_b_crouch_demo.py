import os

import torch


class AlgSolution:
    """Task B B2Piper locomotion-to-crouch smoke-test.

    Timeline at 50 Hz:
      - walk forward with the trained locomotion policy
      - stop and settle
      - interpolate to a crouch leg pose
      - hold crouch while sweeping the arm through large reach poses
      - interpolate back to nominal stand
      - resume locomotion
    """

    def __init__(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        policy_path = os.environ.get(
            "ATEC_B2_POLICY_PATH",
            os.path.join(
                repo_root,
                "logs/rsl_rl/unitree_b2_flat/2026-06-21_12-51-48/exported/policy.pt",
            ),
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

        self.step = 0
        self.leg_action_dim = 12
        self.arm_action_dim = 8
        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))

        self.train_to_env_action_scale = torch.tensor(
            [
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
            ],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)
        self.env_to_train_action_scale = torch.tensor(
            [
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
            ],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)

        self.walk_command = torch.tensor([0.35, 0.0, 0.0], device=self.device).view(1, 3)
        self.stop_command = torch.tensor([0.0, 0.0, 0.0], device=self.device).view(1, 3)
        self.arm_fixed_action = torch.zeros((1, self.arm_action_dim), device=self.device)
        self.arm_motion_targets = self._make_arm_motion_targets()

        # Joint order: FR, FL, RR, RL, each [hip, thigh, calf].
        # Action is target offset / 0.5, so this is a moderate stationary crouch.
        self.stand_leg_action = torch.zeros((1, self.leg_action_dim), device=self.device)
        self.crouch_leg_action = torch.tensor(
            [
                0.0, 0.75, -1.00,
                0.0, 0.75, -1.00,
                0.0, 0.65, -0.90,
                0.0, 0.65, -0.90,
            ],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)

        self.walk_steps_1 = 250
        self.stop_steps = 50
        self.crouch_down_steps = 100
        self.crouch_hold_steps = 360
        self.stand_up_steps = 100
        self.walk_steps_2 = 250

    def reset(self, **_kwargs):
        self.step = 0

    def predicts(self, obs, current_score):
        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3

        leg_action = self._leg_action_for_step(proprio, action_dim)
        arm_action = self._arm_action_for_step(proprio.shape[0])
        action_env = torch.zeros((proprio.shape[0], action_dim), device=self.device)
        action_env[:, self.leg_joint_indices] = leg_action
        if action_dim >= 20:
            action_env[:, self.arm_joint_indices] = arm_action

        self.step += 1
        return {"action": action_env.cpu().numpy().tolist(), "giveup": False}

    def _leg_action_for_step(self, proprio: torch.Tensor, action_dim: int) -> torch.Tensor:
        s = self.step
        a = self.walk_steps_1
        b = a + self.stop_steps
        c = b + self.crouch_down_steps
        d = c + self.crouch_hold_steps
        e = d + self.stand_up_steps
        f = e + self.walk_steps_2

        if s < a:
            return self._policy_leg_action(proprio, action_dim, self.walk_command)
        if s < b:
            return self.stand_leg_action.repeat(proprio.shape[0], 1)
        if s < c:
            alpha = (s - b + 1) / self.crouch_down_steps
            return self._interp(self.stand_leg_action, self.crouch_leg_action, alpha, proprio.shape[0])
        if s < d:
            return self.crouch_leg_action.repeat(proprio.shape[0], 1)
        if s < e:
            alpha = (s - d + 1) / self.stand_up_steps
            return self._interp(self.crouch_leg_action, self.stand_leg_action, alpha, proprio.shape[0])
        if s < f:
            return self._policy_leg_action(proprio, action_dim, self.walk_command)
        return self._policy_leg_action(proprio, action_dim, self.stop_command)

    def _arm_action_for_step(self, num_envs: int) -> torch.Tensor:
        s = self.step
        hold_start = self.walk_steps_1 + self.stop_steps + self.crouch_down_steps
        hold_end = hold_start + self.crouch_hold_steps
        if s < hold_start or s >= hold_end:
            return self.arm_fixed_action.repeat(num_envs, 1)

        local_step = s - hold_start
        segment_len = 45
        segment_idx = min(local_step // segment_len, len(self.arm_motion_targets) - 2)
        alpha = (local_step % segment_len) / segment_len
        start = self.arm_motion_targets[segment_idx]
        end = self.arm_motion_targets[segment_idx + 1]
        return self._interp(start, end, alpha, num_envs)

    def _make_arm_motion_targets(self) -> list[torch.Tensor]:
        # Raw action convention is target_offset / 0.5.  These targets are
        # intentionally large to visibly stress balance while crouched.
        targets = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06, -0.06],
            [0.0, 4.0, -4.0, 0.0, 1.2, 0.0, 0.06, -0.06],      # long forward/down
            [3.2, 3.6, -3.8, -1.2, 0.8, 1.2, 0.06, -0.06],     # left reach
            [-3.2, 3.6, -3.8, 1.2, -0.8, -1.2, 0.06, -0.06],   # right reach
            [0.0, 2.4, -1.2, 2.0, 1.4, 2.0, 0.06, -0.06],      # high/forward
            [2.4, 4.2, -4.4, 1.6, -1.0, -2.0, 0.06, -0.06],    # diagonal left/down
            [-2.4, 4.2, -4.4, -1.6, 1.0, 2.0, 0.06, -0.06],    # diagonal right/down
            [0.0, 4.4, -4.6, 0.0, 0.0, 0.0, 0.06, -0.06],      # maximum straight reach
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        targets = [torch.tensor(target, dtype=torch.float32).view(1, -1) for target in targets]
        return [target.to(self.device) for target in targets]

    def _interp(self, start: torch.Tensor, end: torch.Tensor, alpha: float, num_envs: int) -> torch.Tensor:
        alpha = max(0.0, min(1.0, float(alpha)))
        return (start + (end - start) * alpha).repeat(num_envs, 1)

    def _policy_leg_action(
        self,
        proprio: torch.Tensor,
        action_dim: int,
        velocity_command: torch.Tensor,
    ) -> torch.Tensor:
        policy_obs = self._extract_policy_obs(proprio, action_dim, velocity_command)
        with torch.inference_mode():
            action_train = self.policy(policy_obs)
        action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)
        return action_train * self.train_to_env_action_scale

    def _extract_policy_obs(
        self,
        proprio: torch.Tensor,
        action_dim: int,
        velocity_command: torch.Tensor,
    ) -> torch.Tensor:
        idx = 0
        idx += 3
        base_ang_vel = proprio[:, idx:idx + 3]
        idx += 3
        idx += 3
        projected_gravity = proprio[:, idx:idx + 3]
        idx += 3

        joint_pos_all = proprio[:, idx:idx + action_dim]
        idx += action_dim
        joint_vel_all = proprio[:, idx:idx + action_dim]
        idx += action_dim
        actions_all = proprio[:, idx:idx + action_dim]

        joint_pos_leg = joint_pos_all[:, self.leg_joint_indices]
        joint_vel_leg = joint_vel_all[:, self.leg_joint_indices]
        actions_train_leg = actions_all[:, self.leg_joint_indices] * self.env_to_train_action_scale
        velocity_commands = velocity_command.repeat(proprio.shape[0], 1).to(dtype=proprio.dtype)

        return torch.cat(
            [
                base_ang_vel * 0.25,
                projected_gravity,
                velocity_commands,
                joint_pos_leg,
                joint_vel_leg * 0.05,
                actions_train_leg,
            ],
            dim=-1,
        )
