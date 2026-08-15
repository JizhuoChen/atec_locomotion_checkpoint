# Reference: https://github.com/fan-ziqi/robot_lab

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ray_distance_to_inner_tile(
    spawn_local_xy: torch.Tensor,
    heading_direction: torch.Tensor,
    inner_half_extent: torch.Tensor,
) -> torch.Tensor:
    """Return forward ray distance from each spawn to an axis-aligned inner tile boundary."""
    positive_boundary = inner_half_extent.unsqueeze(0) - spawn_local_xy
    negative_boundary = -inner_half_extent.unsqueeze(0) - spawn_local_xy
    boundary_delta = torch.where(heading_direction >= 0.0, positive_boundary, negative_boundary)
    valid_axis = torch.abs(heading_direction) > 1.0e-6
    axis_distance = torch.where(
        valid_axis,
        boundary_delta / torch.where(valid_axis, heading_direction, torch.ones_like(heading_direction)),
        torch.full_like(heading_direction, torch.inf),
    )
    return torch.clamp(torch.amin(axis_distance, dim=1), min=0.0)


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    heading_alignment_threshold: float = 0.0,
    assigned_tile_path_aware: bool = False,
    assigned_tile_buffer: float = 0.0,
    path_promotion_fraction: float = 0.9,
    path_demotion_fraction: float = 0.5,
    path_minimum_target_distance: float = 1.0,
    path_max_lateral_fraction: float | None = None,
) -> torch.Tensor:
    """Update terrain difficulty from distance travelled since the spread reset.

    On the initial RSL-RL reset, robot roots still contain GridCloner poses while
    terrain origins already contain curriculum-cell poses. The stock curriculum
    interprets that artificial offset as distance travelled and advances every
    environment one level before training starts. Later, measuring from the cell
    center would also count the deliberately spread spawn offset as locomotion.
    Skip the initial update and then measure from each environment's saved reset
    position while retaining Isaac Lab's original up/down thresholds by default.

    When ``assigned_tile_path_aware`` is enabled for a heading-aligned task, use
    signed progress along the commanded world heading. The attainable target is
    capped by the ray distance from the saved spawn to the buffered boundary of
    its assigned terrain tile. Standing commands never change terrain level.
    """
    terrain: TerrainImporter = env.scene.terrain
    if env.common_step_counter == 0 or not hasattr(env, "_terrain_spawn_positions"):
        return torch.mean(terrain.terrain_levels.float())

    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term("base_velocity")
    command = command_term.command
    spawn_xy = env._terrain_spawn_positions[env_ids, :2]
    displacement = asset.data.root_pos_w[env_ids, :2] - spawn_xy
    heading_aligned = getattr(command_term.cfg, "heading_aligned", False)

    if assigned_tile_path_aware:
        if not heading_aligned:
            raise ValueError("assigned_tile_path_aware requires a heading-aligned velocity command.")
        if terrain.cfg.terrain_generator is None or terrain.terrain_origins is None:
            raise ValueError("assigned_tile_path_aware requires generated terrain origins.")
        if not 0.0 < path_promotion_fraction <= 1.0:
            raise ValueError("path_promotion_fraction must be in (0, 1].")
        if not 0.0 <= path_demotion_fraction < path_promotion_fraction:
            raise ValueError("path_demotion_fraction must be in [0, path_promotion_fraction).")
        if path_minimum_target_distance < 0.0:
            raise ValueError("path_minimum_target_distance must be non-negative.")
        if path_max_lateral_fraction is not None and path_max_lateral_fraction <= 0.0:
            raise ValueError("path_max_lateral_fraction must be positive when provided.")

        tile_size = torch.tensor(terrain.cfg.terrain_generator.size[:2], device=env.device)
        if assigned_tile_buffer < 0.0 or torch.any(tile_size <= 2.0 * assigned_tile_buffer):
            raise ValueError(
                f"assigned_tile_buffer={assigned_tile_buffer} is invalid for terrain tile size "
                f"{tuple(tile_size.tolist())}."
            )
        inner_half_extent = 0.5 * tile_size - assigned_tile_buffer
        tile_origins = terrain.terrain_origins[
            terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
        ]
        spawn_local_xy = spawn_xy - tile_origins[:, :2]
        target_heading = command_term.heading_target[env_ids]
        heading_direction = torch.stack((torch.cos(target_heading), torch.sin(target_heading)), dim=1)
        ray_distance = _ray_distance_to_inner_tile(
            spawn_local_xy, heading_direction, inner_half_extent
        )

        projected_progress = torch.sum(displacement * heading_direction, dim=1)
        lateral_direction = torch.stack((-heading_direction[:, 1], heading_direction[:, 0]), dim=1)
        lateral_progress = torch.abs(torch.sum(displacement * lateral_direction, dim=1))
        expected_distance = torch.clamp(command_term.nominal_command_distance[env_ids], min=0.0)
        target_distance = torch.minimum(expected_distance, ray_distance)
        moving = ~command_term.is_standing_env[env_ids]
        has_sufficient_target = target_distance >= path_minimum_target_distance
        if path_max_lateral_fraction is None:
            lateral_progress_ok = torch.ones_like(moving)
        else:
            lateral_progress_ok = lateral_progress <= path_max_lateral_fraction * target_distance
        promotion_eligible = moving & has_sufficient_target & lateral_progress_ok
        move_up = promotion_eligible & (
            projected_progress >= path_promotion_fraction * target_distance
        )
        if heading_alignment_threshold > 0.0:
            alignment_ratio = command_term.heading_alignment_integral[env_ids] / torch.clamp(
                command_term.heading_command_time[env_ids], min=env.step_dt
            )
            move_up &= alignment_ratio >= heading_alignment_threshold
        move_down = (
            moving
            & has_sufficient_target
            & (projected_progress < path_demotion_fraction * target_distance)
            & ~move_up
        )
    else:
        distance = torch.norm(displacement, dim=1)
        move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
        if heading_aligned:
            expected_distance = command_term.nominal_command_distance[env_ids]
            if heading_alignment_threshold > 0.0:
                alignment_ratio = command_term.heading_alignment_integral[env_ids] / torch.clamp(
                    command_term.heading_command_time[env_ids], min=env.step_dt
                )
                move_up &= alignment_ratio >= heading_alignment_threshold
        else:
            expected_distance = torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s
        move_down = distance < expected_distance * 0.5
        move_down &= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def terrain_family_level(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    terrain_names: Sequence[str],
) -> torch.Tensor:
    """Report mean curriculum level for environments assigned to named families.

    This is a logging-only curriculum term: it deliberately ignores the reset
    subset in ``env_ids`` and never changes terrain levels. Reporting every
    family separately prevents a high map-wide mean from hiding sparse exposure
    to a difficult stair/tread combination.
    """
    del env_ids
    from .utils import is_env_assigned_to_terrain

    terrain: TerrainImporter = env.scene.terrain
    family_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for terrain_name in terrain_names:
        family_mask |= is_env_assigned_to_terrain(env, terrain_name)
    if not torch.any(family_mask):
        # Reduced play/smoke scenes may intentionally have fewer columns than
        # configured families. Keep those scenes usable and mark the absent
        # family as unavailable instead of silently reporting level zero.
        return torch.tensor(float("nan"), device=env.device)
    return torch.mean(terrain.terrain_levels[family_mask].float())


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_lin_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_ang_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original angular velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)
        env._initial_ang_vel_z = env._original_ang_vel_z * range_multiplier[0]
        env._final_ang_vel_z = env._original_ang_vel_z * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.ang_vel_z = env._initial_ang_vel_z.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_ang_vel_z = torch.clamp(new_ang_vel_z, min=env._final_ang_vel_z[0], max=env._final_ang_vel_z[1])

            # Update ranges
            base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)
