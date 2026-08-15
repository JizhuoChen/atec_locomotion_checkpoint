# Reference: https://github.com/fan-ziqi/robot_lab

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

from .utils import is_env_assigned_to_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the inertia tensors of the bodies by adding, scaling, or setting random values.

    This function allows randomizing only the diagonal inertia tensor components (xx, yy, zz) of the bodies.
    The function samples random values from the given distribution parameters and adds, scales, or sets the values
    into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body inertias. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current inertia tensors of the bodies (num_assets, num_bodies, 9 for articulations or 9 for rigid objects)
    inertias = asset.root_physx_view.get_inertias()

    # apply randomization on default values
    inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

    # randomize each diagonal element (xx, yy, zz -> indices 0, 4, 8)
    for idx in [0, 4, 8]:
        # Extract and randomize the specific diagonal element
        randomized_inertias = _randomize_prop_by_op(
            inertias[:, :, idx],
            inertia_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        # Assign the randomized values back to the inertia tensor
        inertias[env_ids[:, None], body_ids, idx] = randomized_inertias

    # set the inertia tensors into the physics simulation
    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_com_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the center of mass (COM) positions for the rigid bodies.

    This function allows randomizing the COM positions of the bodies in the physics simulation. The positions can be
    randomized by adding, scaling, or setting random values sampled from the specified distribution.

    .. tip::
        This function is intended for initialization or offline adjustments, as it modifies physics properties directly.

    Args:
        env (ManagerBasedEnv): The simulation environment.
        env_ids (torch.Tensor | None): Specific environment indices to apply randomization,
            or None for all environments.
        asset_cfg (SceneEntityCfg): The configuration for the target asset whose COM will be randomized.
        com_distribution_params (tuple[float, float]): Parameters of the distribution (e.g., min and max for uniform).
        operation (Literal["add", "scale", "abs"]): The operation to apply for randomization.
        distribution (Literal["uniform", "log_uniform", "gaussian"]): The distribution to sample random values from.
    """
    # Extract the asset (Articulation or RigidObject)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # Resolve environment indices
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # Resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # Get the current COM offsets (num_assets, num_bodies, 3)
    com_offsets = asset.root_physx_view.get_coms()

    for dim_idx in range(3):  # Randomize x, y, z independently
        randomized_offset = _randomize_prop_by_op(
            com_offsets[:, :, dim_idx],
            com_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        com_offsets[env_ids[:, None], body_ids, dim_idx] = randomized_offset[env_ids[:, None], body_ids]

    # Set the randomized COM offsets into the simulation
    asset.root_physx_view.set_coms(com_offsets, env_ids)


"""
Internal helper functions.
"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data


def reset_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.

    Note: If "pits" terrain exists, environments on pit terrain will be reset to default state without random
    perturbations to avoid the robot falling into the pit.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # Separate pit and non-pit environments
    # Check which environments are assigned to pit terrain (not random reset)
    assigned_to_pits = is_env_assigned_to_terrain(env, "pits")
    pit_env_ids = env_ids[assigned_to_pits[env_ids]]
    non_pit_env_ids = env_ids[~assigned_to_pits[env_ids]]

    # Reset pit environments to default state (no random perturbations)
    if len(pit_env_ids) > 0:
        root_states = asset.data.default_root_state[pit_env_ids].clone()
        positions = root_states[:, 0:3] + env.scene.env_origins[pit_env_ids]
        orientations = root_states[:, 3:7]
        velocities = torch.zeros_like(root_states[:, 7:13])
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=pit_env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=pit_env_ids)

    # Reset non-pit environments with random perturbations
    if len(non_pit_env_ids) > 0:
        root_states = asset.data.default_root_state[non_pit_env_ids].clone()

        # poses
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(non_pit_env_ids), 6), device=asset.device
        )

        positions = root_states[:, 0:3] + env.scene.env_origins[non_pit_env_ids] + rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
        # velocities
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(non_pit_env_ids), 6), device=asset.device
        )

        velocities = root_states[:, 7:13] + rand_samples

        # set into the physics simulation
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=non_pit_env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=non_pit_env_ids)


def reset_root_state_from_terrain_spread(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    patch_key: str = "init_pos",
    workspace_margin: float = 0.0,
):
    """Reset roots on distinct, terrain-validated patches within each terrain cell.

    Isaac Lab intentionally assigns many vectorized environments to the same terrain
    cell. Randomly choosing a patch with replacement can therefore put several visual
    clones at exactly the same point. This reset ranks environments within their
    current ``(terrain level, terrain type)`` cell and maps those ranks to distinct
    pre-sampled patches. Cross-environment collision filtering remains unchanged.

    The patch sampler supplies terrain-correct XYZ coordinates. ``pose_range`` may
    contain roll, pitch, yaw, and an optional positive Z offset. X/Y are determined by
    the validated patch and are checked against the assigned terrain-tile boundary.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    valid_positions: torch.Tensor | None = terrain.flat_patches.get(patch_key)
    if valid_positions is None:
        raise ValueError(
            f"reset_root_state_from_terrain_spread requires flat patches named '{patch_key}'. "
            f"Found: {list(terrain.flat_patches.keys())}"
        )

    num_rows, num_cols, num_patches = valid_positions.shape[:3]
    cell_ids = terrain.terrain_levels * num_cols + terrain.terrain_types
    cell_counts = torch.bincount(cell_ids, minlength=num_rows * num_cols)
    max_cell_count = int(cell_counts.max().item())
    if max_cell_count > num_patches:
        raise RuntimeError(
            "Not enough distinct terrain spawn patches for the current environment density: "
            f"a cell contains {max_cell_count} environments but only {num_patches} patches exist."
        )

    # Order each cell's random candidate patches with batched farthest-point
    # sampling. The first N entries are therefore a well-spaced subset for a group
    # of size N, instead of simply accepting arbitrarily clustered candidates.
    if not hasattr(env, "_terrain_spread_patch_order"):
        patch_xy = valid_positions[..., :2].reshape(-1, num_patches, 2)
        num_cells = patch_xy.shape[0]
        patch_order = torch.empty((num_cells, num_patches), dtype=torch.long, device=env.device)
        selected = torch.zeros((num_cells, num_patches), dtype=torch.bool, device=env.device)
        centroid = patch_xy.mean(dim=1, keepdim=True)
        next_patch = torch.sum((patch_xy - centroid) ** 2, dim=-1).argmax(dim=1)
        min_distance_sq = torch.full((num_cells, num_patches), torch.inf, device=env.device)
        cell_indices = torch.arange(num_cells, device=env.device)
        for slot in range(num_patches):
            patch_order[:, slot] = next_patch
            selected[cell_indices, next_patch] = True
            selected_xy = patch_xy[cell_indices, next_patch].unsqueeze(1)
            distance_sq = torch.sum((patch_xy - selected_xy) ** 2, dim=-1)
            min_distance_sq = torch.minimum(min_distance_sq, distance_sq)
            min_distance_sq.masked_fill_(selected, -1.0)
            next_patch = min_distance_sq.argmax(dim=1)
        env._terrain_spread_patch_order = patch_order.reshape(num_rows, num_cols, num_patches)

    # Assign a stable rank to every environment within its current terrain cell.
    order = torch.argsort(cell_ids, stable=True)
    sorted_cells = cell_ids[order]
    sorted_indices = torch.arange(len(cell_ids), device=env.device)
    group_start_mask = torch.ones_like(sorted_cells, dtype=torch.bool)
    group_start_mask[1:] = sorted_cells[1:] != sorted_cells[:-1]
    group_starts = torch.where(group_start_mask, sorted_indices, torch.zeros_like(sorted_indices))
    group_starts = torch.cummax(group_starts, dim=0).values
    sorted_ranks = sorted_indices - group_starts
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks

    levels = terrain.terrain_levels[env_ids]
    types = terrain.terrain_types[env_ids]
    patch_ids = env._terrain_spread_patch_order[levels, types, ranks[env_ids]]
    positions = valid_positions[levels, types, patch_ids].clone()
    positions += asset.data.default_root_state[env_ids, :3]

    # Optional clearance above the sampled terrain surface.
    z_range = torch.tensor(pose_range.get("z", (0.0, 0.0)), device=asset.device)
    positions[:, 2] += math_utils.sample_uniform(
        z_range[0], z_range[1], (len(env_ids),), device=asset.device
    )

    # Verify roots against the physical 8x8 m (configurable) terrain tile, not
    # against what happens to be visible from the recording camera.
    tile_origins = terrain.terrain_origins[levels, types]
    tile_half_extent = 0.5 * torch.tensor(
        terrain.cfg.terrain_generator.size[:2], device=asset.device
    )
    local_xy = positions[:, :2] - tile_origins[:, :2]
    outside = torch.any(torch.abs(local_xy) > (tile_half_extent - workspace_margin), dim=1)
    if torch.any(outside):
        bad_env_ids = env_ids[outside][:10].tolist()
        raise RuntimeError(
            f"Terrain spawn workspace violation for {int(outside.sum().item())} environments; "
            f"first environment IDs: {bad_env_ids}."
        )

    # Sample root orientation and velocity using the baseline ranges.
    orientation_ranges = torch.tensor(
        [pose_range.get(key, (0.0, 0.0)) for key in ("roll", "pitch", "yaw")], device=asset.device
    )
    orientation_samples = math_utils.sample_uniform(
        orientation_ranges[:, 0], orientation_ranges[:, 1], (len(env_ids), 3), device=asset.device
    )
    orientation_delta = math_utils.quat_from_euler_xyz(
        orientation_samples[:, 0], orientation_samples[:, 1], orientation_samples[:, 2]
    )
    orientations = math_utils.quat_mul(asset.data.default_root_state[env_ids, 3:7], orientation_delta)

    velocity_ranges = torch.tensor(
        [velocity_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")],
        device=asset.device,
    )
    velocity_samples = math_utils.sample_uniform(
        velocity_ranges[:, 0], velocity_ranges[:, 1], (len(env_ids), 6), device=asset.device
    )
    velocities = asset.data.default_root_state[env_ids, 7:13] + velocity_samples

    if not hasattr(env, "_terrain_spawn_positions"):
        env._terrain_spawn_positions = torch.zeros(env.scene.num_envs, 3, device=env.device)
    env._terrain_spawn_positions[env_ids] = positions

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
