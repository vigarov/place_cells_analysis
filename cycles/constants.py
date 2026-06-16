"""
Constants for the cycles experiment.

Trajectory timing, arena I/O keys, and padding come from
``trajectories.src.constants``; values below are specific to the 20-room
× 30-cycle protocol (Wang et al. NeurIPS 2024, Sec. 3.4 / Suppl. Sec. 3.2).
"""

from __future__ import annotations

from trajectories.src.constants import (
    ARENA_MAP_KEY,
    COORD_KEY,
    DEFAULT_N_TRAJECTORIES,
    DEFAULT_PADDING,
    DEFAULT_SHAPE,
    DT,
    SPEED_NAMES,
)

# Paper: 10 min per room visit during multi-room training
CYCLES_TRIAL_DURATION_S = 600.0

# Paper: 1 s episodic memory segments (Suppl. Table 2, Ts)
EPISODIC_SEGMENT_S = 1.0
STEP_SIZE = int(round(EPISODIC_SEGMENT_S / DT))

# Paper: 5 cm bins on 100 cm room → 20×20 population vector (Suppl. Sec. 3.1)
POPULATION_BIN_SIZE_CM = 5

# WSM files written by ``generate-cycles-rooms`` (not part of trajectory npz format)
WSM_RESPONSE_KEY = "response_map"

# Default trajectory preset for cycles rooms (matches demo / cell_evolution notebooks)
DEFAULT_CYCLES_SPEED = "fast"
DEFAULT_CYCLES_BOUNDARY_AVOIDANCE = False

__all__ = [
    "ARENA_MAP_KEY",
    "COORD_KEY",
    "CYCLES_TRIAL_DURATION_S",
    "DEFAULT_CYCLES_BOUNDARY_AVOIDANCE",
    "DEFAULT_CYCLES_SPEED",
    "DEFAULT_N_TRAJECTORIES",
    "DEFAULT_PADDING",
    "DEFAULT_SHAPE",
    "DT",
    "EPISODIC_SEGMENT_S",
    "POPULATION_BIN_SIZE_CM",
    "SPEED_NAMES",
    "STEP_SIZE",
    "WSM_RESPONSE_KEY",
]
