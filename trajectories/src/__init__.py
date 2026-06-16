"""
trajectories.src
================
Room and trajectory generation utilities for the Wang et al. NeurIPS 2024
place-cell episodic RNN project.

Quickstart
----------
>>> from trajectories.src import create_room
>>> create_room(shape='square', width=150)            # 150 x 150 cm room
>>> create_room(shape='circle', width=200)            # circle in 200 x 200 box
>>> create_room(shape='triangle', width=100)          # right-isoceles triangle

See :func:`create_room` for full parameter documentation.
"""

from .constants import (
    ARENA_MAP_KEY,
    COORD_KEY,
    DEFAULT_BOUNDARY_AVOIDANCE,
    DEFAULT_DURATION_S,
    DEFAULT_N_TRAJECTORIES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PADDING,
    DEFAULT_SHAPE,
    DEFAULT_SPEEDS,
    DEFAULT_TRIANGLE_KIND,
    DT,
    FREE_SPACE,
    MIN_SPEED,
    NO_BOUNDARY_AVOIDANCE,
    SPEED_NAMES,
    SPEED_PRESETS,
    TRIANGLE_KINDS,
    VALID_SHAPES,
    WALL,
)
from .generate_room import create_room
from .room_generator import (
    apply_circle_mask,
    apply_triangle_mask,
    make_square_room,
    save_arena_map,
)
from .trajectory_generator import TrajectoryGenerator

__all__ = [
    # Constants
    'ARENA_MAP_KEY',
    'COORD_KEY',
    'DEFAULT_BOUNDARY_AVOIDANCE',
    'DEFAULT_DURATION_S',
    'DEFAULT_N_TRAJECTORIES',
    'DEFAULT_OUTPUT_DIR',
    'DEFAULT_PADDING',
    'DEFAULT_SHAPE',
    'DEFAULT_SPEEDS',
    'DEFAULT_TRIANGLE_KIND',
    'DT',
    'FREE_SPACE',
    'MIN_SPEED',
    'NO_BOUNDARY_AVOIDANCE',
    'SPEED_NAMES',
    'SPEED_PRESETS',
    'TRIANGLE_KINDS',
    'VALID_SHAPES',
    'WALL',
    # Room generation
    'make_square_room',
    'apply_circle_mask',
    'apply_triangle_mask',
    'save_arena_map',
    # Trajectory generation
    'TrajectoryGenerator',
    # High-level API
    'create_room',
]
