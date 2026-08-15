"""Physics randomization terms for locomotion sim-to-real profiles."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg


def stratified_boolean_assignment(group_ids: torch.Tensor, fraction: float) -> torch.Tensor:
    """Sample an exact-size boolean subset, apportioned across integer groups."""
    if group_ids.ndim != 1:
        raise ValueError(f"group_ids must be one-dimensional, got {tuple(group_ids.shape)}.")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}.")
    assignment = torch.zeros_like(group_ids, dtype=torch.bool)
    target_count = int(round(fraction * len(group_ids)))
    if target_count == 0:
        return assignment

    unique_groups, inverse, counts = torch.unique(
        group_ids, sorted=True, return_inverse=True, return_counts=True
    )
    ideal_counts = counts.to(dtype=torch.float64) * fraction
    allocated = torch.floor(ideal_counts).to(dtype=torch.long)
    remainder = target_count - int(torch.sum(allocated).item())
    if remainder > 0:
        fractional_parts = ideal_counts - allocated
        ranked_groups = sorted(
            range(len(unique_groups)),
            key=lambda index: (
                -float(fractional_parts[index].item()),
                int(unique_groups[index].item()),
            ),
        )
        allocated[ranked_groups[:remainder]] += 1

    for group_index, group_quota in enumerate(allocated.tolist()):
        if group_quota == 0:
            continue
        member_indices = torch.where(inverse == group_index)[0]
        chosen = member_indices[
            torch.randperm(len(member_indices), device=group_ids.device)[:group_quota]
        ]
        assignment[chosen] = True
    if int(torch.count_nonzero(assignment).item()) != target_count:
        raise RuntimeError("Stratified proxy assignment did not produce the requested exact count.")
    return assignment


class randomize_joint_effort_limits(ManagerTermBase):
    """Scale simulated joint effort limits from a cached nominal value.

    A single scale can be shared by all selected joints in an environment to
    approximate battery/thermal motor-strength variation.  The term is intended
    for startup use: each parallel environment then represents one fixed robot.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self._nominal_effort_limits = self.asset.data.joint_effort_limits.clone()

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        scale_range: tuple[float, float],
        per_joint: bool = False,
    ) -> None:
        if scale_range[0] <= 0.0 or scale_range[1] < scale_range[0]:
            raise ValueError(f"Effort-limit scale_range must be positive and ordered: {scale_range}")

        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, dtype=torch.long, device=self.asset.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.asset.device)

        joint_ids = self.asset_cfg.joint_ids
        if isinstance(joint_ids, slice):
            nominal = self._nominal_effort_limits[env_ids]
            resolved_joint_ids = None
        else:
            resolved_joint_ids = torch.as_tensor(joint_ids, dtype=torch.long, device=self.asset.device)
            nominal = self._nominal_effort_limits[env_ids[:, None], resolved_joint_ids]

        scale_shape = nominal.shape if per_joint else (len(env_ids), 1)
        scales = torch.empty(scale_shape, device=self.asset.device).uniform_(*scale_range)
        limits = nominal * scales
        self.asset.write_joint_effort_limit_to_sim(
            limits,
            joint_ids=resolved_joint_ids,
            env_ids=env_ids,
        )


class assign_b2_only_proxy_payload(ManagerTermBase):
    """Assign a persistent low-payload, parked-arm subset of B2-Piper clones.

    Isaac Lab clones one articulation topology across a vectorized scene, so a
    12-DOF B2 USD cannot be mixed directly with a 20-DOF B2-Piper USD while
    retaining one fixed policy/critic interface. This startup term provides a
    continuation-compatible proxy: a fixed fraction of environments keeps the
    arm topology but scales its mass and inertia to a small positive residual.

    The term is intended to run *after* ordinary Piper arm-mass randomization.
    Only the selected proxy environments are overwritten; all other clones keep
    the full-payload mass samples produced by the preceding event.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        proxy_fraction: float = 0.25,
        proxy_mass_scale_range: tuple[float, float] = (0.05, 0.10),
        mask_attribute: str = "_b2_only_proxy_mask",
    ) -> None:
        if not 0.0 <= proxy_fraction <= 1.0:
            raise ValueError(f"proxy_fraction must be in [0, 1], got {proxy_fraction}.")
        if (
            proxy_mass_scale_range[0] <= 0.0
            or proxy_mass_scale_range[1] < proxy_mass_scale_range[0]
            or proxy_mass_scale_range[1] > 1.0
        ):
            raise ValueError(
                "proxy_mass_scale_range must be positive, ordered, and at most one, "
                f"got {proxy_mass_scale_range}."
            )
        if not mask_attribute.startswith("_") or not mask_attribute.isidentifier():
            raise ValueError(
                "mask_attribute must be a private valid Python identifier, "
                f"got {mask_attribute!r}."
            )

        if env_ids is None:
            env_ids_device = torch.arange(env.scene.num_envs, dtype=torch.long, device=env.device)
        else:
            env_ids_device = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()
        terrain_types = getattr(env.scene.terrain, "terrain_types", None)
        if isinstance(terrain_types, torch.Tensor) and terrain_types.shape == (env.scene.num_envs,):
            candidate_groups = terrain_types.to(device=env.device)[env_ids_device]
        else:
            candidate_groups = torch.zeros(len(env_ids_device), dtype=torch.long, device=env.device)
        assignment = stratified_boolean_assignment(candidate_groups, proxy_fraction)

        persistent_mask = getattr(env, mask_attribute, None)
        if persistent_mask is None:
            persistent_mask = torch.zeros(env.scene.num_envs, dtype=torch.bool, device=env.device)
            setattr(env, mask_attribute, persistent_mask)
        elif persistent_mask.shape != (env.scene.num_envs,) or persistent_mask.dtype != torch.bool:
            raise RuntimeError(
                f"Existing {mask_attribute} must be a boolean ({env.scene.num_envs},) tensor."
            )
        persistent_mask[env_ids_device] = assignment

        proxy_env_ids = env_ids_device[assignment]
        if len(proxy_env_ids) > 0:
            proxy_env_ids_cpu = proxy_env_ids.cpu()
            if self.asset_cfg.body_ids == slice(None):
                body_ids = torch.arange(self.asset.num_bodies, dtype=torch.int, device="cpu")
            else:
                body_ids = torch.as_tensor(self.asset_cfg.body_ids, dtype=torch.int, device="cpu")
            if len(body_ids) != 10:
                raise RuntimeError(
                    "B2-only proxy expected exactly 10 Piper rigid bodies "
                    f"(arm_base, arm_link1-8, gripper_base), resolved {len(body_ids)}."
                )

            proxy_scales = torch.empty((len(proxy_env_ids_cpu), 1), device="cpu").uniform_(
                *proxy_mass_scale_range
            )

            masses = self.asset.root_physx_view.get_masses()
            masses[proxy_env_ids_cpu[:, None], body_ids] = (
                self.asset.data.default_mass[proxy_env_ids_cpu[:, None], body_ids]
                * proxy_scales
            )
            self.asset.root_physx_view.set_masses(masses, proxy_env_ids_cpu)

            inertias = self.asset.root_physx_view.get_inertias()
            inertias[proxy_env_ids_cpu[:, None], body_ids] = (
                self.asset.data.default_inertia[proxy_env_ids_cpu[:, None], body_ids]
                * proxy_scales[..., None]
            )
            self.asset.root_physx_view.set_inertias(inertias, proxy_env_ids_cpu)

            payload_scales = getattr(env, "_b2_only_proxy_mass_scales", None)
            if payload_scales is None:
                payload_scales = torch.full(
                    (env.scene.num_envs,), torch.nan, dtype=torch.float, device=env.device
                )
                env._b2_only_proxy_mass_scales = payload_scales
            payload_scales[proxy_env_ids] = proxy_scales[:, 0].to(device=env.device)

        env._b2_only_proxy_count = int(torch.count_nonzero(persistent_mask).item())
        unique_groups, inverse = torch.unique(candidate_groups, sorted=True, return_inverse=True)
        env._b2_only_proxy_counts_by_terrain_type = {
            int(group.item()): int(torch.count_nonzero(assignment & (inverse == index)).item())
            for index, group in enumerate(unique_groups)
        }
        print(
            "[INFO] B2-only payload proxy assignment: "
            f"{env._b2_only_proxy_count}/{env.scene.num_envs} environments "
            f"({torch.mean(persistent_mask.float()).item():.3f}); "
            f"arm mass/inertia scale range={proxy_mass_scale_range}; "
            f"stratified across {len(unique_groups)} terrain types."
        )


def assigned_terrain_out_of_bounds(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_buffer: float = 0.35,
) -> torch.Tensor:
    """Return true after a root leaves its currently assigned sub-terrain tile."""
    terrain = env.scene.terrain
    if terrain.cfg.terrain_type != "generator" or terrain.terrain_origins is None:
        return torch.zeros(env.scene.num_envs, dtype=torch.bool, device=env.device)

    tile_size = torch.tensor(terrain.cfg.terrain_generator.size[:2], device=env.device)
    if distance_buffer < 0.0 or torch.any(tile_size <= 2.0 * distance_buffer):
        raise ValueError(
            f"distance_buffer={distance_buffer} is invalid for terrain tile size {tuple(tile_size.tolist())}."
        )

    asset: Articulation = env.scene[asset_cfg.name]
    tile_origins = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    local_xy = asset.data.root_pos_w[:, :2] - tile_origins[:, :2]
    return torch.any(torch.abs(local_xy) > (0.5 * tile_size - distance_buffer), dim=1)
