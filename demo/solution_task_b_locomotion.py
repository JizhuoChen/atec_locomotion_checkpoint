import os

import torch


class AlgSolution:
    """Task B locomotion smoke-test solution.

    Runs a 12-DoF B2 velocity policy on B2Piper and keeps the Piper arm fixed.
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

        self.leg_action_dim = 12
        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))
        self.arm_action_dim = 8

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

        self.velocity_command = torch.tensor(
            [0.35, 0.0, 0.0],
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)
        self.arm_fixed_action = torch.zeros(
            (1, self.arm_action_dim),
            device=self.device,
            dtype=torch.float32,
        )

    def reset(self, **_kwargs):
        pass

    def predicts(self, obs, current_score):
        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3
        policy_obs = self._extract_policy_obs(proprio, action_dim)

        with torch.inference_mode():
            action_train = self.policy(policy_obs)

        action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)

        action_env = torch.zeros((action_train.shape[0], action_dim), device=self.device)
        action_env[:, self.leg_joint_indices] = action_train * self.train_to_env_action_scale
        if action_dim >= 20:
            action_env[:, self.arm_joint_indices] = self.arm_fixed_action.repeat(action_train.shape[0], 1)

        return {"action": action_env.cpu().numpy().tolist(), "giveup": False}

    def _extract_policy_obs(self, proprio: torch.Tensor, action_dim: int) -> torch.Tensor:
        idx = 0
        idx += 3  # base linear velocity is not used by this policy.
        base_ang_vel = proprio[:, idx:idx + 3]
        idx += 3
        idx += 3  # ignore env command; local validation supplies our command below.
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
        velocity_commands = self.velocity_command.repeat(proprio.shape[0], 1).to(dtype=proprio.dtype)

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
