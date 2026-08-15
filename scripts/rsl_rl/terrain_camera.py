"""Utilities for recording one representative robot from every generated terrain family."""

from __future__ import annotations

import math
from collections.abc import Mapping

import gymnasium as gym


def _normalized_terrain_proportions(sub_terrains: Mapping[str, object]) -> tuple[list[str], list[float]]:
    """Validate and normalize an ordered sub-terrain mapping."""
    if not sub_terrains:
        raise ValueError("Terrain showcase requires at least one configured sub-terrain.")

    terrain_names = list(sub_terrains.keys())
    proportions = [float(cfg.proportion) for cfg in sub_terrains.values()]
    invalid = [
        name
        for name, proportion in zip(terrain_names, proportions, strict=True)
        if not math.isfinite(proportion) or proportion <= 0.0
    ]
    if invalid:
        raise ValueError(
            "Terrain showcase proportions must be finite and positive; invalid families: "
            + ", ".join(invalid)
        )
    proportion_sum = math.fsum(proportions)
    return terrain_names, [proportion / proportion_sum for proportion in proportions]


def terrain_family_column_indices(
    sub_terrains: Mapping[str, object], num_cols: int
) -> list[int]:
    """Return the sub-terrain index assigned to every curriculum column.

    This intentionally mirrors :class:`isaaclab.terrains.TerrainGenerator`'s
    ``column / num_cols + 0.001`` selection rule.
    """
    if num_cols <= 0:
        raise ValueError(f"Terrain column count must be positive, got {num_cols}.")
    _, proportions = _normalized_terrain_proportions(sub_terrains)
    cumulative = []
    running = 0.0
    for proportion in proportions:
        running += proportion
        cumulative.append(running)
    # Avoid a final-boundary miss caused only by floating-point summation.
    cumulative[-1] = 1.0

    family_for_column = []
    for column in range(num_cols):
        sample = column / num_cols + 0.001
        try:
            family_for_column.append(
                next(index for index, boundary in enumerate(cumulative) if sample < boundary)
            )
        except StopIteration as error:
            raise ValueError(
                f"Isaac Lab's curriculum-column rule cannot assign column {column} of {num_cols}; "
                "use fewer showcase columns."
            ) from error
    return family_for_column


def terrain_showcase_column_allocation(
    sub_terrains: Mapping[str, object], *, max_columns: int = 999, tolerance: float = 1.0e-9
) -> dict[str, int]:
    """Find the smallest column count that exactly realizes all normalized weights.

    The returned ordered mapping gives the number of generated columns per
    sub-terrain. A candidate is accepted only when Isaac Lab's actual
    curriculum-column selection produces the same allocation.
    """
    if max_columns <= 0:
        raise ValueError(f"max_columns must be positive, got {max_columns}.")
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}.")

    terrain_names, proportions = _normalized_terrain_proportions(sub_terrains)
    for num_cols in range(len(terrain_names), max_columns + 1):
        expected_counts = [proportion * num_cols for proportion in proportions]
        rounded_counts = [round(count) for count in expected_counts]
        if any(count < 1 for count in rounded_counts):
            continue
        if any(
            abs(expected - rounded) > tolerance
            for expected, rounded in zip(expected_counts, rounded_counts, strict=True)
        ):
            continue
        if sum(rounded_counts) != num_cols:
            continue

        actual_indices = terrain_family_column_indices(sub_terrains, num_cols)
        actual_counts = [actual_indices.count(index) for index in range(len(terrain_names))]
        if actual_counts == rounded_counts:
            return dict(zip(terrain_names, rounded_counts, strict=True))

    raise ValueError(
        "Could not realize the normalized terrain proportions exactly while representing every "
        f"family in at most {max_columns} curriculum columns."
    )


def terrain_family_representatives(env: gym.Env) -> list[dict[str, int | str]]:
    """Resolve a representative environment ID for each configured sub-terrain family."""
    base_env = env.unwrapped
    terrain = base_env.scene.terrain
    generator_cfg = terrain.cfg.terrain_generator
    if terrain.cfg.terrain_type != "generator" or generator_cfg is None:
        raise ValueError("Terrain-family camera cycling requires a generated terrain.")
    if not generator_cfg.curriculum:
        raise ValueError("Terrain-family camera cycling requires deterministic curriculum columns.")

    terrain_names = list(generator_cfg.sub_terrains.keys())
    family_for_column = terrain_family_column_indices(
        generator_cfg.sub_terrains, generator_cfg.num_cols
    )

    terrain_types = terrain.terrain_types.detach().cpu().tolist()
    terrain_levels = terrain.terrain_levels.detach().cpu().tolist()
    target_level = 0.6 * max(generator_cfg.num_rows - 1, 0)
    representatives = []
    for family_index, terrain_name in enumerate(terrain_names):
        candidates = [
            env_id
            for env_id, column in enumerate(terrain_types)
            if family_for_column[column] == family_index
        ]
        if not candidates:
            raise RuntimeError(f"No environment was assigned to terrain family '{terrain_name}'.")
        env_id = min(candidates, key=lambda index: abs(terrain_levels[index] - target_level))
        representatives.append(
            {
                "terrain": terrain_name,
                "env_id": int(env_id),
                "initial_column": int(terrain_types[env_id]),
                "initial_level": int(terrain_levels[env_id]),
            }
        )
    return representatives


class TerrainFamilyCameraCycle(gym.Wrapper):
    """Switch the tracked viewport environment at fixed global step intervals."""

    def __init__(self, env: gym.Env, interval: int, representatives: list[dict[str, int | str]]):
        if interval <= 0:
            raise ValueError("Camera-cycle interval must be positive.")
        if not representatives:
            raise ValueError("At least one terrain representative is required.")
        super().__init__(env)
        self.interval = interval
        self.representatives = representatives
        self.step_id = -1
        self._active_index = -1
        self._select_for_step(0)

    def _select_for_step(self, step: int):
        cycle_index = (step // self.interval) % len(self.representatives)
        if cycle_index == self._active_index:
            return
        representative = self.representatives[cycle_index]
        camera_controller = self.env.unwrapped.viewport_camera_controller
        camera_controller.set_view_env_index(int(representative["env_id"]))
        # Asset tracking normally updates on Kit's next post-render callback.
        # Update once synchronously so the first frame of each clip is not from
        # the previously selected terrain.
        camera_controller.update_view_to_asset_root("robot")
        self._active_index = cycle_index
        print(
            f"[INFO] Terrain camera: step {step}, family '{representative['terrain']}', "
            f"environment {representative['env_id']}."
        )

    def reset(self, **kwargs):
        self.step_id = -1
        self._active_index = -1
        result = super().reset(**kwargs)
        self._select_for_step(0)
        return result

    def step(self, action):
        next_step = self.step_id + 1
        self._select_for_step(next_step)
        result = self.env.step(action)
        self.step_id = next_step
        return result


def terrain_video_manifest(
    representatives: list[dict[str, int | str]], interval: int, name_prefix: str
) -> dict[str, object]:
    """Build a YAML-friendly mapping from recordings to terrain families."""
    return {
        "interval_steps": interval,
        "cycle_length_steps": interval * len(representatives),
        "segments": [
            {
                **representative,
                "start_step_in_cycle": index * interval,
                "video_file": f"{name_prefix}-step-{index * interval}.mp4",
                "video_name_pattern": f"{name_prefix}-step-<global-step>.mp4",
            }
            for index, representative in enumerate(representatives)
        ],
    }
