"""Robust terrain distribution for B2 locomotion sim-to-real training.

This module is deliberately opt-in: the Isaac Lab baseline terrain and the
existing B2 environment classes are left unchanged.  Call
``configure_robust_terrain`` from a derived environment configuration to use
this distribution.

The stock :class:`HfRandomUniformTerrainCfg` ignores the terrain difficulty
argument.  ``HfCurriculumRandomUniformTerrainCfg`` fixes that limitation by
scaling each signed noise range from ``minimum_noise_scale`` to its configured
maximum as the curriculum advances.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import TYPE_CHECKING

import isaaclab.terrains as terrain_gen
import numpy as np
import scipy.interpolate as interpolate
from isaaclab.terrains import FlatPatchSamplingCfg, SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.terrains.trimesh import mesh_terrains
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from atec_rl_lab.train.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg


# Binary fractions make the sum exactly 1.0 in IEEE-754 and map cleanly onto
# 80 columns (1/16 -> 5 columns and 1/8 -> 10 columns).
_STAIR_VARIANT_PROPORTION = 1.0 / 16.0
_ROUGH_VARIANT_PROPORTION = 1.0 / 8.0
_BOX_PROPORTION = 1.0 / 8.0
_SLOPE_VARIANT_PROPORTION = 1.0 / 16.0
_GEOMETRY_VERSION = 2
_NONCE_UPPER_BOUND = np.iinfo(np.int64).max


def _stable_uint64(*parts: object) -> int:
    """Hash procedural inputs without depending on process-randomized Python hashes."""
    encoded = "\x1f".join(
        value.hex() if isinstance(value, float) else str(value) for value in parts
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], byteorder="big")


def _spawn_patch_sampling() -> dict[str, FlatPatchSamplingCfg]:
    """Create independent reset-patch settings matching the B2 spawn logic."""
    return {
        "init_pos": FlatPatchSamplingCfg(
            num_patches=64,
            patch_radius=0.55,
            x_range=(-2.8, 2.8),
            y_range=(-2.8, 2.8),
            max_height_diff=0.08,
        )
    }


@height_field_to_mesh
def difficulty_scaled_random_uniform_terrain(
    difficulty: float, cfg: HfCurriculumRandomUniformTerrainCfg
) -> np.ndarray:
    """Generate signed roughness whose amplitude increases with difficulty.

    Isaac Lab's built-in random-uniform height field ignores ``difficulty``
    and its cubic interpolation can overshoot the requested height range. This
    implementation uses bounded linear interpolation so the configured maximum
    remains a real geometric limit.
    """
    difficulty = min(max(float(difficulty), 0.0), 1.0)
    amplitude_scale = cfg.minimum_noise_scale + difficulty * (1.0 - cfg.minimum_noise_scale)
    if not 0.0 <= cfg.minimum_noise_scale <= 1.0:
        raise ValueError(f"minimum_noise_scale must be in [0, 1], got {cfg.minimum_noise_scale}.")
    if cfg.noise_range[0] >= 0.0 or cfg.noise_range[1] <= 0.0:
        raise ValueError(f"noise_range must contain signed heights around zero, got {cfg.noise_range}.")
    if not np.isclose(abs(cfg.noise_range[0]), cfg.noise_range[1]):
        raise ValueError(f"noise_range must be symmetric to avoid a height bias, got {cfg.noise_range}.")

    sample_scale = cfg.downsampled_scale or cfg.horizontal_scale
    if sample_scale < cfg.horizontal_scale:
        raise ValueError(
            "downsampled_scale must be greater than or equal to horizontal_scale: "
            f"{sample_scale} < {cfg.horizontal_scale}."
        )

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    width_downsampled = max(2, int(cfg.size[0] / sample_scale))
    length_downsampled = max(2, int(cfg.size[1] / sample_scale))

    max_amplitude = cfg.noise_range[1] * amplitude_scale
    amplitude_units = max(1, int(np.floor(max_amplitude / cfg.vertical_scale + 1.0e-9)))
    step_units = max(1, int(round(cfg.noise_step / cfg.vertical_scale)))
    positive_levels = np.arange(0, amplitude_units + 1, step_units, dtype=np.int16)
    if positive_levels[-1] != amplitude_units:
        positive_levels = np.append(positive_levels, np.int16(amplitude_units))
    # Construct levels symmetrically so random samples do not bias the mean
    # surface above or below zero.
    height_levels = np.concatenate((-positive_levels[:0:-1], positive_levels))
    # Isaac Lab seeds NumPy before scene creation. Consuming one global nonce
    # preserves seed-dependent, per-tile geometry while the local generator
    # keeps all remaining draws isolated from other terrain implementations.
    sampling_nonce = int(np.random.randint(0, _NONCE_UPPER_BOUND))
    rng = np.random.default_rng(
        _stable_uint64(
            cfg.geometry_version,
            cfg.sampling_salt,
            sampling_nonce,
            float(difficulty),
            *cfg.noise_range,
            cfg.noise_step,
            sample_scale,
            *cfg.size,
        )
    )
    height_field_downsampled = rng.choice(
        height_levels, size=(width_downsampled, length_downsampled)
    )

    source_x = np.linspace(0.0, cfg.size[0], width_downsampled)
    source_y = np.linspace(0.0, cfg.size[1], length_downsampled)
    interpolator = interpolate.RectBivariateSpline(
        source_x,
        source_y,
        height_field_downsampled,
        kx=1,
        ky=1,
    )
    target_x = np.linspace(0.0, cfg.size[0], width_pixels)
    target_y = np.linspace(0.0, cfg.size[1], length_pixels)
    height_field = np.rint(interpolator(target_x, target_y))
    return np.clip(height_field, -amplitude_units, amplitude_units).astype(np.int16)


@configclass
class HfCurriculumRandomUniformTerrainCfg(terrain_gen.HfRandomUniformTerrainCfg):
    """Random-uniform height field with difficulty-scaled noise amplitude."""

    function = difficulty_scaled_random_uniform_terrain

    minimum_noise_scale: float = 0.25
    """Fraction of ``noise_range`` used at difficulty zero."""

    sampling_salt: str = "rough"
    """Stable family salt used to reproduce the procedural height samples."""

    geometry_version: int = _GEOMETRY_VERSION
    """Bump when the procedural geometry algorithm changes."""


def variable_pyramid_stairs_terrain(
    difficulty: float, cfg: MeshVariablePyramidStairsTerrainCfg
):
    """Generate normal or inverted stairs with an independently varied tread."""
    resolved_cfg = deepcopy(cfg)
    sampling_nonce = int(np.random.randint(0, _NONCE_UPPER_BOUND))
    resolved_cfg.step_width = resolve_stair_tread_width(
        difficulty, cfg, sampling_nonce=sampling_nonce
    )
    generator = (
        mesh_terrains.inverted_pyramid_stairs_terrain
        if cfg.inverted
        else mesh_terrains.pyramid_stairs_terrain
    )
    return generator(difficulty, resolved_cfg)


@configclass
class MeshVariablePyramidStairsTerrainCfg(SubTerrainBaseCfg):
    """Pyramid stairs with tread depth sampled independently from riser difficulty."""

    function = variable_pyramid_stairs_terrain
    step_height_range: tuple[float, float] = (0.04, 0.26)
    step_width_range: tuple[float, float] = (0.22, 0.28)
    step_width_resolution: float = 0.005
    platform_width: float = 3.0
    border_width: float = 1.0
    holes: bool = False
    inverted: bool = False
    sampling_salt: str = "stairs"
    geometry_version: int = _GEOMETRY_VERSION


def resolve_stair_tread_width(
    difficulty: float,
    cfg: MeshVariablePyramidStairsTerrainCfg,
    *,
    sampling_nonce: int = 0,
) -> float:
    """Resolve one deterministic, quantized tread depth for a generated tile.

    Isaac Lab already jitters ``difficulty`` independently for every row/column
    tile. Hash-whitening it together with a seeded per-tile nonce makes tread
    selection non-monotonic with the riser curriculum while retaining exact
    reproducibility from the saved environment and terrain seed.
    """
    lower, upper = (float(value) for value in cfg.step_width_range)
    resolution = float(cfg.step_width_resolution)
    if not 0.0 < lower <= upper:
        raise ValueError(f"Invalid stair tread range {cfg.step_width_range!r}.")
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError(f"Stair tread resolution must be positive, got {resolution!r}.")
    span_in_bins = (upper - lower) / resolution
    rounded_bins = int(round(span_in_bins))
    if not np.isclose(span_in_bins, rounded_bins, rtol=0.0, atol=1.0e-9):
        raise ValueError(
            f"Stair tread range {cfg.step_width_range!r} is not aligned to {resolution} m."
        )
    sample = _stable_uint64(
        cfg.geometry_version,
        cfg.sampling_salt,
        int(sampling_nonce),
        float(difficulty),
        lower,
        upper,
        resolution,
        cfg.inverted,
    )
    # Clamp removes benign binary floating-point overshoot at an inclusive
    # upper endpoint (for example 0.42000000000000004 for a 0.42 m band).
    return float(np.clip(lower + (sample % (rounded_bins + 1)) * resolution, lower, upper))


def _stairs(
    step_width_range: tuple[float, float], *, inverted: bool, sampling_salt: str
) -> MeshVariablePyramidStairsTerrainCfg:
    return MeshVariablePyramidStairsTerrainCfg(
        proportion=_STAIR_VARIANT_PROPORTION,
        step_height_range=(0.04, 0.26),
        step_width_range=step_width_range,
        step_width_resolution=0.005,
        platform_width=3.0,
        border_width=1.0,
        holes=False,
        inverted=inverted,
        sampling_salt=sampling_salt,
        flat_patch_sampling=_spawn_patch_sampling(),
    )


def _rough(
    noise_amplitude: float,
    noise_step: float,
    downsampled_scale: float,
    sampling_salt: str,
) -> HfCurriculumRandomUniformTerrainCfg:
    return HfCurriculumRandomUniformTerrainCfg(
        proportion=_ROUGH_VARIANT_PROPORTION,
        noise_range=(-noise_amplitude, noise_amplitude),
        noise_step=noise_step,
        downsampled_scale=downsampled_scale,
        minimum_noise_scale=0.30,
        sampling_salt=sampling_salt,
        border_width=0.25,
        flat_patch_sampling=_spawn_patch_sampling(),
    )


_SUB_TERRAINS = {
    # Tread depth is sampled at 5 mm resolution within each band, independently
    # of the 4--26 cm riser curriculum. Overlapping boundary values avoid gaps.
    "stairs_short_tread": _stairs(
        (0.22, 0.28), inverted=False, sampling_salt="stairs_short"
    ),
    "stairs_short_tread_inv": _stairs(
        (0.22, 0.28), inverted=True, sampling_salt="stairs_short_inv"
    ),
    "stairs_nominal_tread": _stairs(
        (0.28, 0.35), inverted=False, sampling_salt="stairs_nominal"
    ),
    "stairs_nominal_tread_inv": _stairs(
        (0.28, 0.35), inverted=True, sampling_salt="stairs_nominal_inv"
    ),
    "stairs_long_tread": _stairs(
        (0.35, 0.42), inverted=False, sampling_salt="stairs_long"
    ),
    "stairs_long_tread_inv": _stairs(
        (0.35, 0.42), inverted=True, sampling_salt="stairs_long_inv"
    ),
    # Signed depressions and protrusions at three amplitudes/wavelengths.
    # Amplitudes shown here are maxima; each is curriculum-scaled from 30%.
    "rough_fine": _rough(
        noise_amplitude=0.040,
        noise_step=0.005,
        downsampled_scale=0.10,
        sampling_salt="rough_fine",
    ),
    "rough_medium": _rough(
        noise_amplitude=0.075,
        noise_step=0.010,
        downsampled_scale=0.20,
        sampling_salt="rough_medium",
    ),
    "rough_hard": _rough(
        noise_amplitude=0.115,
        noise_step=0.020,
        downsampled_scale=0.40,
        sampling_salt="rough_hard",
    ),
    # Retain the useful baseline obstacle and slope families.
    "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        proportion=_BOX_PROPORTION,
        grid_width=0.43,
        grid_height_range=(0.05, 0.22),
        platform_width=2.0,
        flat_patch_sampling=_spawn_patch_sampling(),
    ),
    "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        proportion=_SLOPE_VARIANT_PROPORTION,
        slope_range=(0.0, 0.45),
        platform_width=2.0,
        border_width=0.25,
        flat_patch_sampling=_spawn_patch_sampling(),
    ),
    "slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        proportion=_SLOPE_VARIANT_PROPORTION,
        slope_range=(0.0, 0.45),
        platform_width=2.0,
        border_width=0.25,
        flat_patch_sampling=_spawn_patch_sampling(),
    ),
}

_PROPORTION_SUM = sum(cfg.proportion for cfg in _SUB_TERRAINS.values())
if _PROPORTION_SUM != 1.0:
    raise RuntimeError(f"Robust B2 terrain proportions must sum exactly to 1.0, got {_PROPORTION_SUM!r}.")


B2_ROBUST_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=80,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=True,
    use_cache=False,
    sub_terrains=_SUB_TERRAINS,
)
"""Opt-in 10-by-80 robust terrain curriculum for B2/B2-Piper."""


def configure_robust_terrain(env_cfg: LocomotionVelocityRoughEnvCfg) -> None:
    """Install an independent robust generator on a derived environment config."""
    env_cfg.scene.terrain.terrain_generator = deepcopy(B2_ROBUST_TERRAINS_CFG)
