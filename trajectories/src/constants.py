"""
constants.py
============
Shared constants for room and trajectory generation.

Values follow Wang et al. (NeurIPS 2024) Supplemental Section 1.1 and the
project README trajectory-generation presets.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Simulation timing (Suppl. Table 2: dt = 50 ms)
# ---------------------------------------------------------------------------

DT: Final[float] = 0.05
DEFAULT_DURATION_S: Final[float] = 2048.0
DEFAULT_N_TRAJECTORIES: Final[int] = 128
MIN_SPEED: Final[float] = 0.1  # cm/s floor when resampling velocity

# ---------------------------------------------------------------------------
# Arena map format (matches reference trajectories/100x100_square files)
# ---------------------------------------------------------------------------

DEFAULT_PADDING: Final[int] = 5  # pixels of wall border on each side
FREE_SPACE: Final[float] = 0.0
WALL: Final[float] = 1.0
ARENA_MAP_KEY: Final[str] = 'arena_map'
COORD_KEY: Final[str] = 'coord'

# ---------------------------------------------------------------------------
# Room geometry
# ---------------------------------------------------------------------------

VALID_SHAPES: Final[frozenset[str]] = frozenset({'square', 'circle', 'triangle'})
TRIANGLE_KINDS: Final[frozenset[str]] = frozenset({'right_isoceles', 'equilateral'})
DEFAULT_SHAPE: Final[str] = 'square'
DEFAULT_TRIANGLE_KIND: Final[str] = 'right_isoceles'

# ---------------------------------------------------------------------------
# Output defaults
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR: Final[str] = 'trajectories'
DEFAULT_SPEEDS: Final[tuple[str, ...]] = ('slow', 'med', 'fast')
DEFAULT_BOUNDARY_AVOIDANCE: Final[tuple[bool, bool]] = (True, False)

# Sentinel: disable boundary-avoidance steering (non-_ba trajectory files)
NO_BOUNDARY_AVOIDANCE: Final[int] = -1

# ---------------------------------------------------------------------------
# Speed presets (from project README)
# ---------------------------------------------------------------------------

SPEED_PRESETS: Final[dict[str, dict[str, float]]] = {
    'slow': {
        'velocity_mean': 4.0,
        'velocity_sd': 1.0,
        'random_drift_magnitude': 0.02,
        'switch_direction_prob': 0.06,
        'switch_velocity_prob': 0.03,
        'avoid_boundary_dist': 5.0, # Use -1 to disable boundary avoidance
    },
    'med': {
        'velocity_mean': 10.0,
        'velocity_sd': 2.0,
        'random_drift_magnitude': 0.05,
        'switch_direction_prob': 0.15,
        'switch_velocity_prob': 0.05,
        'avoid_boundary_dist': 10.0, # Use -1 to disable boundary avoidance
    },
    'fast': {
        'velocity_mean': 20.0,
        'velocity_sd': 5.0,
        'random_drift_magnitude': 0.10,
        'switch_direction_prob': 0.3,
        'switch_velocity_prob': 0.1,
        'avoid_boundary_dist': 30.0, # Use -1 to disable boundary avoidance
    },
}

SPEED_NAMES: Final[tuple[str, ...]] = tuple(SPEED_PRESETS.keys())
