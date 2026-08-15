# Reference: https://github.com/fan-ziqi/robot_lab

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import atec_rl_lab.train.locomotion.velocity.mdp as mdp

from .utils import is_robot_on_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold.

    This command generator automatically detects "pits" terrain and applies restrictions:
    - For pit terrains: only allow forward movement (no lateral or rotational movement)
    """

    cfg: mdp.UniformThresholdVelocityCommandCfg  # type: ignore
    """The configuration of the command generator."""

    def __init__(self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)
        # Track which robots were on pit terrain in the previous step
        self.was_on_pit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Buffers used only by the opt-in heading-aligned command mode. Keeping
        # these separate from ``vel_command_b`` prevents the forward target from
        # being repeatedly attenuated every control step.
        self.nominal_forward_speed = torch.zeros(self.num_envs, device=self.device)
        self.heading_alignment_gate = torch.ones(self.num_envs, device=self.device)
        self.nominal_command_distance = torch.zeros(self.num_envs, device=self.device)
        self.heading_alignment_integral = torch.zeros(self.num_envs, device=self.device)
        self.heading_command_time = torch.zeros(self.num_envs, device=self.device)

        if self.cfg.heading_aligned:
            if not self.cfg.heading_command or self.cfg.rel_heading_envs != 1.0:
                raise ValueError("Heading-aligned commands require heading_command=True and rel_heading_envs=1.0.")
            if self.cfg.ranges.lin_vel_x[0] < 0.0:
                raise ValueError("Heading-aligned commands require a non-negative lin_vel_x range.")
            if tuple(self.cfg.ranges.lin_vel_y) != (0.0, 0.0):
                raise ValueError("Heading-aligned commands require lin_vel_y=(0.0, 0.0).")
            if self.cfg.alignment_std <= 0.0:
                raise ValueError("alignment_std must be positive.")
            if not 0.0 <= self.cfg.terrain_center_heading_probability <= 1.0:
                raise ValueError("terrain_center_heading_probability must be in [0, 1].")
            if not 0.0 <= self.cfg.terrain_center_heading_max_deviation <= torch.pi:
                raise ValueError("terrain_center_heading_max_deviation must be in [0, pi].")

            self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["lateral_velocity"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["alignment_gate"] = torch.zeros(self.num_envs, device=self.device)
            if self.cfg.track_terrain_exposure:
                if self._env.scene.cfg.terrain.terrain_type != "generator":
                    raise ValueError("Terrain exposure metrics require generated terrain.")
                terrain_size = torch.tensor(
                    self._env.scene.cfg.terrain.terrain_generator.size[:2], device=self.device
                )
                if self.cfg.terrain_tile_buffer < 0.0 or torch.any(
                    terrain_size <= 2.0 * self.cfg.terrain_tile_buffer
                ):
                    raise ValueError(
                        f"terrain_tile_buffer={self.cfg.terrain_tile_buffer} is invalid for "
                        f"terrain tile size {tuple(terrain_size.tolist())}."
                    )

                # CommandTerm.reset() logs the mean of each entry in ``metrics``.
                # Keep raw per-environment counters separately, then materialize
                # episode ratios immediately before the base reset logs them.
                self.metrics["assigned_tile_fraction"] = torch.zeros(self.num_envs, device=self.device)
                self.metrics["nonflat_trajectory_fraction"] = torch.zeros(self.num_envs, device=self.device)
                self.metrics["nonflat_assigned_fraction"] = torch.zeros(self.num_envs, device=self.device)
                self.metrics["mean_local_relief_assigned"] = torch.zeros(self.num_envs, device=self.device)
                self.metrics["ever_exited_assigned_tile"] = torch.zeros(self.num_envs, device=self.device)
                self._terrain_exposure_steps = torch.zeros(self.num_envs, device=self.device)
                self._terrain_assigned_steps = torch.zeros(self.num_envs, device=self.device)
                self._terrain_nonflat_steps = torch.zeros(self.num_envs, device=self.device)
                self._terrain_assigned_nonflat_steps = torch.zeros(self.num_envs, device=self.device)
                self._terrain_assigned_relief_sum = torch.zeros(self.num_envs, device=self.device)
                self._terrain_ever_exited = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _env_ids_tensor(self, env_ids: Sequence[int] | slice | torch.Tensor | None) -> torch.Tensor:
        """Resolve manager-style environment IDs to a one-dimensional device tensor."""
        if env_ids is None:
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)[env_ids]
        resolved = torch.as_tensor(env_ids, device=self.device)
        if resolved.dtype == torch.bool:
            resolved = torch.nonzero(resolved, as_tuple=False).flatten()
        return resolved.to(dtype=torch.long).flatten()

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset command state while retaining the completed episode's path target for curriculum use."""
        reset_ids = self._env_ids_tensor(env_ids)
        if self.cfg.heading_aligned and self.cfg.track_terrain_exposure and len(reset_ids) > 0:
            # Command metrics are updated after environment resets, so the state
            # that triggered a tile-exit timeout would otherwise be absent from
            # both the denominator and the ever-exited flag. The termination
            # manager still holds its just-computed term values at this point.
            termination_manager = self._env.termination_manager
            if "assigned_terrain_out_of_bounds" in termination_manager.active_terms:
                terminal_tile_exit = termination_manager.get_term(
                    "assigned_terrain_out_of_bounds"
                )[reset_ids]
                self._terrain_exposure_steps[reset_ids] += terminal_tile_exit.float()
                self._terrain_ever_exited[reset_ids] |= terminal_tile_exit

            total_steps = self._terrain_exposure_steps[reset_ids]
            assigned_steps = self._terrain_assigned_steps[reset_ids]
            total_denominator = torch.clamp(total_steps, min=1.0)
            assigned_denominator = torch.clamp(assigned_steps, min=1.0)
            has_samples = total_steps > 0.0
            has_assigned_samples = assigned_steps > 0.0

            self.metrics["assigned_tile_fraction"][reset_ids] = torch.where(
                has_samples, assigned_steps / total_denominator, 0.0
            )
            self.metrics["nonflat_trajectory_fraction"][reset_ids] = torch.where(
                has_samples,
                self._terrain_nonflat_steps[reset_ids] / total_denominator,
                0.0,
            )
            self.metrics["nonflat_assigned_fraction"][reset_ids] = torch.where(
                has_assigned_samples,
                self._terrain_assigned_nonflat_steps[reset_ids] / assigned_denominator,
                0.0,
            )
            self.metrics["mean_local_relief_assigned"][reset_ids] = torch.where(
                has_assigned_samples,
                self._terrain_assigned_relief_sum[reset_ids] / assigned_denominator,
                0.0,
            )
            self.metrics["ever_exited_assigned_tile"][reset_ids] = self._terrain_ever_exited[
                reset_ids
            ].float()

        # The base implementation logs and clears metrics before resampling the
        # next command. Passing resolved tensor IDs also handles None and slices
        # consistently in the custom state cleanup below.
        extras = super().reset(reset_ids)
        if self.cfg.heading_aligned and self.cfg.track_terrain_exposure and len(reset_ids) > 0:
            self._terrain_exposure_steps[reset_ids] = 0.0
            self._terrain_assigned_steps[reset_ids] = 0.0
            self._terrain_nonflat_steps[reset_ids] = 0.0
            self._terrain_assigned_nonflat_steps[reset_ids] = 0.0
            self._terrain_assigned_relief_sum[reset_ids] = 0.0
            self._terrain_ever_exited[reset_ids] = False
        self.nominal_command_distance[reset_ids] = 0.0
        self.heading_alignment_integral[reset_ids] = 0.0
        self.heading_command_time[reset_ids] = 0.0
        self.heading_alignment_gate[reset_ids] = 1.0
        self.was_on_pit[reset_ids] = False
        return extras

    def _update_metrics(self):
        """Add heading-specific diagnostics without changing the baseline metrics."""
        super()._update_metrics()
        if not self.cfg.heading_aligned:
            return

        # These buffers reset once per episode, so normalize by episode length
        # rather than the command resampling horizon.
        normalization_steps = self._env.max_episode_length
        heading_error = math_utils.wrap_to_pi(self.heading_target - self.robot.data.heading_w)
        vel_yaw = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self.robot.data.root_quat_w), self.robot.data.root_lin_vel_w
        )
        moving = ~self.is_standing_env
        self.metrics["error_heading"] += torch.where(moving, torch.abs(heading_error), 0.0) / normalization_steps
        self.metrics["lateral_velocity"] += (
            torch.where(moving, torch.abs(vel_yaw[:, 1]), 0.0) / normalization_steps
        )
        self.metrics["alignment_gate"] += self.heading_alignment_gate / normalization_steps
        if self.cfg.track_terrain_exposure:
            terrain = self._env.scene.terrain
            tile_origins = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
            tile_half_extent = 0.5 * torch.tensor(
                terrain.cfg.terrain_generator.size[:2], device=self.device
            ) - self.cfg.terrain_tile_buffer
            local_xy = self.robot.data.root_pos_w[:, :2] - tile_origins[:, :2]
            inside_assigned_tile = torch.all(torch.abs(local_xy) <= tile_half_extent, dim=1)

            height_sensor = self._env.scene[self.cfg.terrain_height_sensor]
            hit_heights = height_sensor.data.ray_hits_w[..., 2]
            fallback_height = self.robot.data.root_pos_w[:, 2].unsqueeze(1)
            hit_heights = torch.where(torch.isfinite(hit_heights), hit_heights, fallback_height)
            local_relief = torch.amax(hit_heights, dim=1) - torch.amin(hit_heights, dim=1)
            nonflat = local_relief >= self.cfg.terrain_relief_threshold

            self._terrain_exposure_steps += 1.0
            self._terrain_assigned_steps += inside_assigned_tile.float()
            self._terrain_nonflat_steps += nonflat.float()
            self._terrain_assigned_nonflat_steps += (inside_assigned_tile & nonflat).float()
            self._terrain_assigned_relief_sum += torch.where(inside_assigned_tile, local_relief, 0.0)
            self._terrain_ever_exited |= ~inside_assigned_tile

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity commands with threshold."""
        env_ids_tensor = self._env_ids_tensor(env_ids)
        super()._resample_command(env_ids_tensor)
        # set small commands to zero
        self.vel_command_b[env_ids_tensor, :2] *= (
            torch.norm(self.vel_command_b[env_ids_tensor, :2], dim=1) > 0.2
        ).unsqueeze(1)
        if self.cfg.heading_aligned:
            self.nominal_forward_speed[env_ids_tensor] = self.vel_command_b[env_ids_tensor, 0]
            if self.cfg.terrain_center_heading_probability > 0.0:
                terrain = self._env.scene.terrain
                if terrain.terrain_origins is None:
                    raise RuntimeError("Inward-biased headings require generated terrain origins.")
                tile_origins = terrain.terrain_origins[
                    terrain.terrain_levels[env_ids_tensor], terrain.terrain_types[env_ids_tensor]
                ]
                if hasattr(self._env, "_terrain_spawn_positions"):
                    spawn_xy = self._env._terrain_spawn_positions[env_ids_tensor, :2]
                else:
                    # This fallback is used only during initialization if the
                    # terrain-aware reset event has not populated its buffer yet.
                    spawn_xy = self.robot.data.root_pos_w[env_ids_tensor, :2]
                delta_to_center = tile_origins[:, :2] - spawn_xy
                center_heading = torch.atan2(delta_to_center[:, 1], delta_to_center[:, 0])
                deviation = torch.empty_like(center_heading).uniform_(
                    -self.cfg.terrain_center_heading_max_deviation,
                    self.cfg.terrain_center_heading_max_deviation,
                )
                inward_heading = math_utils.wrap_to_pi(center_heading + deviation)
                use_inward = (
                    torch.rand(len(env_ids_tensor), device=self.device)
                    < self.cfg.terrain_center_heading_probability
                ) & (torch.linalg.vector_norm(delta_to_center, dim=1) > 0.25)
                self.heading_target[env_ids_tensor] = torch.where(
                    use_inward, inward_heading, self.heading_target[env_ids_tensor]
                )

    def _update_command(self):
        """Update commands and apply terrain-aware restrictions in real-time.

        This function:
        1. Calls parent's update to handle heading and standing envs
        2. Checks which robots are currently on pit terrain
        3. For robots leaving pits: resamples their commands
        4. For robots on pits: restricts to forward-only movement and sets heading to 0
        """
        # First, call parent's update command
        super()._update_command()

        # Check which robots are currently on pit terrain (real-time check every step)
        on_pits = is_robot_on_terrain(self._env, "pits")

        # Find robots that just left pit terrain (need to resample)
        left_pit_mask = self.was_on_pit & ~on_pits
        if left_pit_mask.any():
            left_pit_env_ids = torch.where(left_pit_mask)[0]
            # Resample commands for robots that left pits
            self._resample_command(left_pit_env_ids)

        if self.cfg.heading_aligned:
            # The policy receives a forward/yaw command with unchanged shape. A
            # Gaussian gate makes the desired forward speed nearly zero for a
            # large heading error and smoothly restores it after the robot turns.
            heading_error = math_utils.wrap_to_pi(self.heading_target - self.robot.data.heading_w)
            gate = torch.exp(-torch.square(heading_error / self.cfg.alignment_std))
            moving = ~self.is_standing_env
            self.heading_alignment_gate[:] = torch.where(moving, gate, 1.0)
            self.vel_command_b[:, 0] = torch.where(moving, self.nominal_forward_speed * gate, 0.0)
            self.vel_command_b[:, 1] = 0.0
            self.vel_command_b[self.is_standing_env, 2] = 0.0

            # This ungated path-length target prevents a policy that never turns
            # from avoiding terrain-curriculum demotion via a near-zero command.
            step_dt = self._env.step_dt
            self.nominal_command_distance += torch.where(moving, self.nominal_forward_speed, 0.0) * step_dt
            self.heading_alignment_integral += torch.where(moving, gate, 0.0) * step_dt
            self.heading_command_time += moving * step_dt

        # For robots currently on pits: restrict to forward-only movement with min/max speed
        if on_pits.any():
            pit_env_ids = torch.where(on_pits)[0]
            # Force forward-only movement with min and max speed limits
            self.vel_command_b[pit_env_ids, 0] = torch.clamp(
                torch.abs(self.vel_command_b[pit_env_ids, 0]), min=0.3, max=0.6
            )
            self.vel_command_b[pit_env_ids, 1] = 0.0  # no lateral movement
            self.vel_command_b[pit_env_ids, 2] = 0.0  # no yaw rotation
            # Set heading to 0 for pit robots
            if self.cfg.heading_command:
                self.heading_target[pit_env_ids] = 0.0
            if self.cfg.heading_aligned:
                self.heading_alignment_gate[pit_env_ids] = 1.0

        # Update tracking state
        self.was_on_pit = on_pits

    def _debug_vis_callback(self, event):
        """Draw the fixed world-heading target in heading-aligned mode."""
        if not self.cfg.heading_aligned:
            return super()._debug_vis_callback(event)
        if not self.robot.is_initialized:
            return

        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        target_scale = torch.tensor(default_scale, device=self.device).repeat(self.num_envs, 1)
        target_scale[:, 0] *= self.nominal_forward_speed * 3.0
        target_scale[self.is_standing_env, 0] = 0.0
        zeros = torch.zeros_like(self.heading_target)
        target_quat = math_utils.quat_from_euler_xyz(zeros, zeros, self.heading_target)

        current_scale, current_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.goal_vel_visualizer.visualize(base_pos_w, target_quat, target_scale)
        self.current_vel_visualizer.visualize(base_pos_w, current_quat, current_scale)


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand

    heading_aligned: bool = False
    """Whether to convert a world-heading target into turn-first, forward-only commands."""

    alignment_std: float = 0.55
    """Heading-error scale in radians for the Gaussian forward-speed gate."""

    terrain_center_heading_probability: float = 0.0
    """Probability of aiming a new world heading approximately toward the assigned tile center."""

    terrain_center_heading_max_deviation: float = 0.0
    """Maximum absolute angular jitter around a terrain-center heading, in radians."""

    track_terrain_exposure: bool = False
    """Whether to log assigned-tile and local-relief exposure during each episode."""

    terrain_tile_buffer: float = 0.0
    """Inset from each tile edge used when measuring assigned-terrain exposure, in metres."""

    terrain_height_sensor: str = "height_scanner"
    """Ray-caster sensor used to estimate whether the local surface is non-flat."""

    terrain_relief_threshold: float = 0.025
    """Minimum local height range, in metres, counted as non-flat exposure."""


class SinusoidalArmCommand(CommandTerm):
    """Drive a mounted arm through smooth, randomized joint trajectories.

    This command is intentionally not part of the locomotion policy observation or
    action.  It acts as a physically simulated, time-varying payload disturbance.
    The asymmetric critic may still observe the actual arm joint state.  The default
    ``sinusoidal`` mode preserves the original behavior.  ``smooth_waypoint`` mode
    visits independently sampled, limit-safe poses using a minimum-jerk interpolation
    whose position, velocity, and acceleration are continuous at every waypoint.
    """

    cfg: SinusoidalArmCommandCfg

    def __init__(self, cfg: SinusoidalArmCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.joint_ids, self.joint_names = self.robot.find_joints(cfg.joint_names, preserve_order=True)
        if len(self.joint_ids) != len(cfg.amplitude_scales):
            raise ValueError(
                "SinusoidalArmCommand amplitude_scales must contain one value per resolved joint: "
                f"got {len(cfg.amplitude_scales)} scales for {len(self.joint_ids)} joints {self.joint_names}."
            )
        if len(cfg.center_offset_scales) != len(self.joint_ids):
            raise ValueError(
                "SinusoidalArmCommand center_offset_scales must contain one value per resolved joint: "
                f"got {len(cfg.center_offset_scales)} scales for {len(self.joint_ids)} joints {self.joint_names}."
            )
        if not 0.0 <= cfg.limit_margin_fraction < 0.5:
            raise ValueError("limit_margin_fraction must be in [0, 0.5).")
        if cfg.amplitude_range[0] < 0.0 or cfg.amplitude_range[1] < cfg.amplitude_range[0]:
            raise ValueError("amplitude_range must be non-negative and ordered.")
        if cfg.frequency_range[0] <= 0.0 or cfg.frequency_range[1] < cfg.frequency_range[0]:
            raise ValueError("frequency_range must be positive and ordered.")
        if not 0.0 <= cfg.stationary_probability <= 1.0:
            raise ValueError("stationary_probability must be in [0, 1].")
        if cfg.trajectory_mode not in ("sinusoidal", "smooth_waypoint"):
            raise ValueError(
                "trajectory_mode must be either 'sinusoidal' or 'smooth_waypoint', "
                f"got {cfg.trajectory_mode!r}."
            )

        num_joints = len(self.joint_ids)
        shape = (self.num_envs, num_joints)
        self._target = torch.zeros(shape, device=self.device)
        self._target_velocity = torch.zeros(shape, device=self.device)
        self._center = torch.zeros(shape, device=self.device)
        self._amplitude = torch.zeros(shape, device=self.device)
        self._frequency = torch.zeros(shape, device=self.device)
        self._phase = torch.zeros(shape, device=self.device)
        self._motion_time = torch.zeros(self.num_envs, 1, device=self.device)
        self._amplitude_scales = torch.tensor(cfg.amplitude_scales, device=self.device).unsqueeze(0)
        self._center_offset_scales = torch.tensor(cfg.center_offset_scales, device=self.device).unsqueeze(0)
        self._waypoint_start = torch.zeros(shape, device=self.device)
        self._waypoint_goal = torch.zeros(shape, device=self.device)
        self._waypoint_duration = torch.ones(self.num_envs, 1, device=self.device)
        self._moving_episode = torch.ones(self.num_envs, 1, dtype=torch.bool, device=self.device)

        if cfg.trajectory_mode == "smooth_waypoint":
            if cfg.waypoint_offset_ranges is None:
                raise ValueError("smooth_waypoint mode requires waypoint_offset_ranges.")
            if len(cfg.waypoint_offset_ranges) != num_joints:
                raise ValueError(
                    "waypoint_offset_ranges must contain one (lower, upper) pair per resolved joint: "
                    f"got {len(cfg.waypoint_offset_ranges)} ranges for {num_joints} joints {self.joint_names}."
                )
            waypoint_ranges = torch.tensor(cfg.waypoint_offset_ranges, device=self.device, dtype=torch.float)
            if waypoint_ranges.shape != (num_joints, 2) or torch.any(
                waypoint_ranges[:, 0] > waypoint_ranges[:, 1]
            ):
                raise ValueError("Every waypoint_offset_ranges entry must be an ordered (lower, upper) pair.")
            if (
                cfg.segment_duration_range[0] <= 0.0
                or cfg.segment_duration_range[1] < cfg.segment_duration_range[0]
            ):
                raise ValueError("segment_duration_range must be positive and ordered.")
            if (
                cfg.pause_duration_range[0] <= 0.0
                or cfg.pause_duration_range[1] < cfg.pause_duration_range[0]
            ):
                raise ValueError("pause_duration_range must be positive and ordered.")
            if not 0.0 <= cfg.pause_probability <= 1.0:
                raise ValueError("pause_probability must be in [0, 1].")
            if cfg.max_joint_speeds is not None:
                if len(cfg.max_joint_speeds) != num_joints or min(cfg.max_joint_speeds) <= 0.0:
                    raise ValueError("max_joint_speeds must contain one positive value per resolved joint.")
                self._max_joint_speeds = torch.tensor(
                    cfg.max_joint_speeds, device=self.device, dtype=torch.float
                ).unsqueeze(0)
            else:
                self._max_joint_speeds = None
            self._waypoint_offset_ranges = waypoint_ranges.unsqueeze(0)
            active_mask = waypoint_ranges[:, 1] > waypoint_ranges[:, 0]
        else:
            self._waypoint_offset_ranges = None
            self._max_joint_speeds = None
            active_mask = self._amplitude_scales[0] > 0.0
        if not torch.any(active_mask):
            raise ValueError("At least one arm joint must be configured to move.")
        self._active_joint_ids = torch.where(active_mask)[0]

        self.metrics["position_tracking_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["joint_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["target_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["target_offset"] = torch.zeros(self.num_envs, device=self.device)
        if cfg.stationary_env_mask_attr is not None:
            self.metrics["b2_only_proxy_fraction"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Return the current absolute arm joint-position target."""
        return self._target

    def _as_env_ids(self, env_ids: Sequence[int] | slice | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _externally_stationary(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return the persistent parked-arm mask selected by a startup event."""
        if self.cfg.stationary_env_mask_attr is None:
            return torch.zeros((len(env_ids), 1), dtype=torch.bool, device=self.device)
        mask = getattr(self._env, self.cfg.stationary_env_mask_attr, None)
        if mask is None:
            # Command terms are constructed before startup events are applied.
            return torch.zeros((len(env_ids), 1), dtype=torch.bool, device=self.device)
        if mask.shape != (self.num_envs,) or mask.dtype != torch.bool:
            raise RuntimeError(
                f"{self.cfg.stationary_env_mask_attr} must be a boolean "
                f"({self.num_envs},) tensor, got shape={tuple(mask.shape)}, dtype={mask.dtype}."
            )
        return mask.to(device=self.device)[env_ids, None]

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Resample a trajectory and initialize the arm without a target discontinuity."""
        reset_ids = self._as_env_ids(env_ids)
        if "b2_only_proxy_fraction" in self.metrics:
            all_env_ids = torch.arange(self.num_envs, device=self.device)
            proxy_fraction = torch.mean(self._externally_stationary(all_env_ids).float())
            # CommandTerm.reset logs then clears the selected metric entries.
            # Materialize the fixed startup assignment directly so early falls
            # cannot make this provenance metric look smaller than it is.
            self.metrics["b2_only_proxy_fraction"][reset_ids] = proxy_fraction
        extras = super().reset(reset_ids)
        self.robot.write_joint_state_to_sim(
            self._target[reset_ids],
            self._target_velocity[reset_ids],
            joint_ids=self.joint_ids,
            env_ids=reset_ids,
        )
        self.robot.set_joint_position_target(
            self._target[reset_ids], joint_ids=self.joint_ids, env_ids=reset_ids
        )
        self.robot.set_joint_velocity_target(
            self._target_velocity[reset_ids], joint_ids=self.joint_ids, env_ids=reset_ids
        )
        return extras

    def _update_metrics(self):
        active = self._active_joint_ids
        joint_pos = self.robot.data.joint_pos[:, self.joint_ids][:, active]
        joint_vel = self.robot.data.joint_vel[:, self.joint_ids][:, active]
        target = self._target[:, active]
        target_velocity = self._target_velocity[:, active]
        default = self.robot.data.default_joint_pos[:, self.joint_ids][:, active]
        normalization_steps = self._env.max_episode_length
        self.metrics["position_tracking_error"] += (
            torch.mean(torch.abs(joint_pos - target), dim=1) / normalization_steps
        )
        self.metrics["joint_speed"] += torch.mean(torch.abs(joint_vel), dim=1) / normalization_steps
        self.metrics["target_speed"] += torch.mean(torch.abs(target_velocity), dim=1) / normalization_steps
        self.metrics["target_offset"] += torch.mean(torch.abs(target - default), dim=1) / normalization_steps

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = self._as_env_ids(env_ids)
        num_resets = len(env_ids)
        if num_resets == 0:
            return

        if self.cfg.trajectory_mode == "smooth_waypoint":
            self._resample_waypoint_trajectory(env_ids)
            return

        default = self.robot.data.default_joint_pos[env_ids[:, None], self.joint_ids]
        center_offset = torch.empty_like(default).uniform_(*self.cfg.center_offset_range)
        center_offset *= self._center_offset_scales
        amplitude = torch.empty_like(default).uniform_(*self.cfg.amplitude_range)
        amplitude *= self._amplitude_scales
        moving_episode = torch.rand((num_resets, 1), device=self.device) >= self.cfg.stationary_probability
        externally_stationary = self._externally_stationary(env_ids)
        moving_episode &= ~externally_stationary
        amplitude *= moving_episode
        # A shared per-environment frequency produces a coordinated slow arm
        # motion; independent phases keep it from degenerating into one rigid arc.
        frequency = torch.empty((num_resets, 1), device=self.device).uniform_(*self.cfg.frequency_range)
        frequency = frequency.expand_as(default)
        phase = torch.empty_like(default).uniform_(0.0, 2.0 * torch.pi)

        limits = self.robot.data.soft_joint_pos_limits[env_ids[:, None], self.joint_ids]
        limit_span = limits[..., 1] - limits[..., 0]
        margin = self.cfg.limit_margin_fraction * limit_span
        safe_lower = limits[..., 0] + margin + amplitude
        safe_upper = limits[..., 1] - margin - amplitude
        if torch.any(safe_lower > safe_upper):
            raise RuntimeError("Configured arm amplitude and limit margin leave no valid joint center.")
        center = torch.maximum(torch.minimum(default + center_offset, safe_upper), safe_lower)
        center = torch.where(externally_stationary, default, center)

        self._center[env_ids] = center
        self._amplitude[env_ids] = amplitude
        self._frequency[env_ids] = frequency
        self._phase[env_ids] = phase
        self._motion_time[env_ids] = 0.0
        self._target[env_ids] = center + amplitude * torch.sin(phase)
        self._target_velocity[env_ids] = amplitude * (2.0 * torch.pi * frequency) * torch.cos(phase)

    def _sample_safe_waypoint(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Sample absolute poses within the configured envelope and soft limits."""
        if self._waypoint_offset_ranges is None:
            raise RuntimeError("Waypoint sampling is only available in smooth_waypoint mode.")

        default = self.robot.data.default_joint_pos[env_ids[:, None], self.joint_ids]
        limits = self.robot.data.soft_joint_pos_limits[env_ids[:, None], self.joint_ids]
        limit_span = limits[..., 1] - limits[..., 0]
        margin = self.cfg.limit_margin_fraction * limit_span
        envelope_lower = default + self._waypoint_offset_ranges[..., 0]
        envelope_upper = default + self._waypoint_offset_ranges[..., 1]
        safe_lower = torch.maximum(envelope_lower, limits[..., 0] + margin)
        safe_upper = torch.minimum(envelope_upper, limits[..., 1] - margin)
        if torch.any(safe_lower > safe_upper):
            raise RuntimeError(
                "Configured waypoint envelope and limit margin leave no valid pose for at least one arm joint."
            )
        return safe_lower + torch.rand_like(safe_lower) * (safe_upper - safe_lower)

    def _sample_waypoint_segment(self, env_ids: torch.Tensor):
        """Start a new moving or paused minimum-jerk segment from the current target."""
        if len(env_ids) == 0:
            return

        num_envs = len(env_ids)
        start = self._target[env_ids].clone()
        sampled_goal = self._sample_safe_waypoint(env_ids)
        paused = torch.rand((num_envs, 1), device=self.device) < self.cfg.pause_probability
        paused |= ~self._moving_episode[env_ids]
        goal = torch.where(paused, start, sampled_goal)

        moving_duration = torch.empty((num_envs, 1), device=self.device).uniform_(
            *self.cfg.segment_duration_range
        )
        pause_duration = torch.empty((num_envs, 1), device=self.device).uniform_(
            *self.cfg.pause_duration_range
        )
        duration = torch.where(paused, pause_duration, moving_duration)

        # The maximum derivative of 10u^3-15u^4+6u^5 is 1.875.  Stretch
        # segments as needed so every requested joint velocity remains bounded.
        if self._max_joint_speeds is not None:
            speed_limited_duration = torch.amax(
                1.875 * torch.abs(goal - start) / self._max_joint_speeds, dim=1, keepdim=True
            )
            duration = torch.maximum(duration, speed_limited_duration)

        self._waypoint_start[env_ids] = start
        self._waypoint_goal[env_ids] = goal
        self._waypoint_duration[env_ids] = duration
        self._motion_time[env_ids] = 0.0

    def _resample_waypoint_trajectory(self, env_ids: torch.Tensor):
        """Initialize each episode at a varied pose and schedule its first segment."""
        initial_pose = self._sample_safe_waypoint(env_ids)
        externally_stationary = self._externally_stationary(env_ids)
        default = self.robot.data.default_joint_pos[env_ids[:, None], self.joint_ids]
        initial_pose = torch.where(externally_stationary, default, initial_pose)
        self._moving_episode[env_ids] = (
            torch.rand((len(env_ids), 1), device=self.device) >= self.cfg.stationary_probability
        ) & ~externally_stationary
        self._target[env_ids] = initial_pose
        self._target_velocity[env_ids] = 0.0
        self._center[env_ids] = initial_pose
        self._amplitude[env_ids] = 0.0
        self._frequency[env_ids] = 0.0
        self._phase[env_ids] = 0.0
        self._sample_waypoint_segment(env_ids)

    def _update_command(self):
        if self.cfg.trajectory_mode == "smooth_waypoint":
            self._update_waypoint_command()
            return

        self._motion_time += self._env.step_dt
        angle = 2.0 * torch.pi * self._frequency * self._motion_time + self._phase
        self._target[:] = self._center + self._amplitude * torch.sin(angle)
        self._target_velocity[:] = self._amplitude * (2.0 * torch.pi * self._frequency) * torch.cos(angle)
        self.robot.set_joint_position_target(self._target, joint_ids=self.joint_ids)
        self.robot.set_joint_velocity_target(self._target_velocity, joint_ids=self.joint_ids)

    def _update_waypoint_command(self):
        """Advance all waypoint trajectories by one control step."""
        self._motion_time += self._env.step_dt
        completed = self._motion_time[:, 0] >= self._waypoint_duration[:, 0]
        if torch.any(completed):
            completed_ids = torch.where(completed)[0]
            self._target[completed_ids] = self._waypoint_goal[completed_ids]
            self._target_velocity[completed_ids] = 0.0
            self._sample_waypoint_segment(completed_ids)

        progress = torch.clamp(self._motion_time / self._waypoint_duration, 0.0, 1.0)
        progress_sq = torch.square(progress)
        progress_cu = progress_sq * progress
        blend = progress_cu * (10.0 - 15.0 * progress + 6.0 * progress_sq)
        blend_derivative = 30.0 * progress_sq * torch.square(1.0 - progress)
        delta = self._waypoint_goal - self._waypoint_start
        self._target[:] = self._waypoint_start + delta * blend
        self._target_velocity[:] = delta * blend_derivative / self._waypoint_duration
        self.robot.set_joint_position_target(self._target, joint_ids=self.joint_ids)
        self.robot.set_joint_velocity_target(self._target_velocity, joint_ids=self.joint_ids)


@configclass
class SinusoidalArmCommandCfg(CommandTermCfg):
    """Configuration for randomized motion of a mounted arm.

    New trajectory options default to the original sinusoidal behavior so existing
    configurations and checkpoints remain compatible.
    """

    class_type: type = SinusoidalArmCommand
    asset_name: str = MISSING
    joint_names: list[str] = MISSING
    center_offset_range: tuple[float, float] = (-0.12, 0.12)
    center_offset_scales: tuple[float, ...] = MISSING
    amplitude_range: tuple[float, float] = (0.03, 0.08)
    amplitude_scales: tuple[float, ...] = MISSING
    frequency_range: tuple[float, float] = (0.05, 0.12)
    stationary_probability: float = 0.2
    stationary_env_mask_attr: str | None = None
    """Optional environment boolean-mask attribute whose arms remain parked at their default pose."""
    limit_margin_fraction: float = 0.02
    trajectory_mode: Literal["sinusoidal", "smooth_waypoint"] = "sinusoidal"
    waypoint_offset_ranges: tuple[tuple[float, float], ...] | None = None
    segment_duration_range: tuple[float, float] = (2.0, 5.0)
    pause_probability: float = 0.2
    pause_duration_range: tuple[float, float] = (0.5, 2.0)
    max_joint_speeds: tuple[float, ...] | None = None


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """
