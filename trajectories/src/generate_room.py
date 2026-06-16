"""
generate_room.py
================
Python API and CLI for creating rooms compatible with the Wang et al.
NeurIPS 2024 codebase.

Python API
----------
>>> from trajectories.src.generate_room import create_room

>>> # 150 x 150 cm square room (all 6 trajectory files)
>>> create_room(shape='square', width=150)

>>> # 200 x 200 cm circle (inscribed in 200 x 200 bounding box)
>>> create_room(shape='circle', width=200, name='circle_200')

>>> # Right-isoceles triangle, only medium and fast speeds, BA only
>>> create_room(shape='triangle', width=100, height=100,
...             speeds=['med', 'fast'], boundary_avoidance=[True])

CLI
---
    uv run generate-room --shape square --width 150

    uv run python -m trajectories.src.generate_room --shape square --width 150

    uv run python -m trajectories.src.generate_room \\
        --shape circle --width 200 --name circle_200 \\
        --speeds med fast --ba-only

Each call writes into `<output_dir>/<name>/`:
  • `arena_map.npz`
  • `traj_<speed>.npz`         (non-BA, when requested)
  • `traj_<speed>_ba.npz`      (BA,     when requested)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, Union

import numpy as np

from .constants import (
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
    SPEED_NAMES,
    SPEED_PRESETS,
    TRIANGLE_KINDS,
    VALID_SHAPES,
)
from .room_generator import (
    apply_circle_mask,
    apply_triangle_mask,
    make_square_room,
    save_arena_map,
)
from .trajectory_generator import TrajectoryGenerator

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------

def create_room(
    shape: str = DEFAULT_SHAPE,
    width: int = 100,
    height: int = None,
    name: str = None,
    speeds: Iterable[str] = DEFAULT_SPEEDS,
    boundary_avoidance: Union[Iterable[bool], bool] = DEFAULT_BOUNDARY_AVOIDANCE,
    n_trajectories: int = DEFAULT_N_TRAJECTORIES,
    duration_s: float = DEFAULT_DURATION_S,
    padding: int = DEFAULT_PADDING,
    seed: int = None,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    triangle_kind: str = DEFAULT_TRIANGLE_KIND,
    verbose: bool = True,
) -> str:
    """
    Create a new room: generate and save an arena map plus trajectory files.

    The output exactly matches the format of the Wang et al. NeurIPS 2024
    reference files stored in `trajectories/100x100_square/` and
    `trajectories/200x200_square/`.

    Output files
    ------------
    All files are written to `<output_dir>/<name>/`:

    `arena_map.npz`
        Key `'arena_map'`, shape `(width + 2*padding, height + 2*padding)`,
        dtype float64.  0 = accessible, 1 = wall.

    `traj_<speed>.npz` / `traj_<speed>_ba.npz`
        Key `'coord'`, shape `(n_trajectories, n_steps, 2)`, dtype float64.
        n_steps = round(duration_s / 0.05) = 40 960 for the default 2048 s.

    Parameters
    ----------
    shape     : Room geometry.

                `'square'`   -- plain rectangular room (default).
                `'circle'`   -- largest circle inscribed in the bounding rectangle.
                `'triangle'` -- triangle inscribed in the bounding rectangle
                                 (see *triangle_kind*).

    width     : Room width in cm (1 cm = 1 pixel).
    height    : Room height in cm.  Defaults to *width* (square bounding box).
    name      : Sub-directory name under *output_dir*.  Auto-generated as
                `'<width>x<height>_<shape>'` if `None`.
    speeds    : Iterable of speed preset names.  Subset of
                `{'slow', 'med', 'fast'}`.
    boundary_avoidance : Which BA variants to generate.

                `(True, False)` (default) -- generate both `_ba` and
                plain variants for each speed.

                `True` / `[True]`  -- BA only.
                `False` / `[False]` -- non-BA only.

    n_trajectories : Independent trajectories per file (default 128).
    duration_s     : Seconds per trajectory (default 2048 → 40 960 steps).
    padding        : Wall-padding width in pixels on every side (default 5).
    seed           : Integer RNG seed.  The same seed is reused across all
                     (speed, BA) combinations -- results are fully reproducible
                     given the same seed.
    output_dir     : Root directory that holds all room sub-directories.
                     Defaults to `'trajectories'` (relative to CWD).
    triangle_kind  : Passed to :func:`apply_triangle_mask`.
                     `'right_isoceles'` (default) or `'equilateral'`.
    verbose        : Print progress to stdout when `True`.

    Returns
    -------
    str
        Absolute path to the created room directory.

    Examples
    --------
    >>> create_room(shape='square', width=150)
    # Creates trajectories/150x150_square/

    >>> create_room(shape='circle', width=200, speeds=['med'],
    ...             boundary_avoidance=[True], seed=0)
    # Creates trajectories/200x200_circle/traj_med_ba.npz only
    """
    # ---- Validate arguments ----
    if height is None:
        height = width

    speeds = list(speeds)
    for s in speeds:
        if s not in SPEED_PRESETS:
            raise ValueError(
                f"Unknown speed preset '{s}'.  Valid: {list(SPEED_PRESETS)}."
            )

    if isinstance(boundary_avoidance, bool):
        ba_list = [boundary_avoidance]
    else:
        ba_list = list(boundary_avoidance)

    shape = shape.lower()
    if shape not in VALID_SHAPES:
        raise ValueError(
            f"Unknown shape '{shape}'.  Valid shapes: {sorted(VALID_SHAPES)}."
        )
    if triangle_kind not in TRIANGLE_KINDS:
        raise ValueError(
            f"Unknown triangle kind '{triangle_kind}'.  "
            f"Valid options: {sorted(TRIANGLE_KINDS)}."
        )

    # ---- Build arena map ----
    arena = make_square_room(width, height, padding=padding)

    if shape == 'circle':
        arena = apply_circle_mask(arena)
    elif shape == 'triangle':
        arena = apply_triangle_mask(arena, kind=triangle_kind)

    # ---- Output directory ----
    if name is None:
        name = f'{width}x{height}_{shape}'

    room_dir = Path(output_dir) / name
    arena_path = save_arena_map(arena, room_dir)

    if verbose:
        n_accessible = int((arena == FREE_SPACE).sum())
        n_total = arena.size
        print(
            f"[create_room] arena_map saved  → {arena_path}\n"
            f"  shape={shape!r}, size={width}x{height} cm, "
            f"padding={padding}, arena_map.shape={arena.shape}\n"
            f"  accessible pixels: {n_accessible} / {n_total} "
            f"({100 * n_accessible / n_total:.1f} %)"
        )

    # ---- Generate trajectories ----
    gen = TrajectoryGenerator(
        arena_map=arena,
        n_trajectories=n_trajectories,
        duration_s=duration_s,
        dt=DT,
        seed=seed,
    )

    tasks = [(spd, ba) for spd in speeds for ba in ba_list]
    if verbose:
        print(
            f"  Generating {len(tasks)} trajectory file(s): "
            + ", ".join(
                f"traj_{s}{'_ba' if ba else ''}" for s, ba in tasks
            )
        )

    iterator = (
        _tqdm(tasks, desc='Trajectories', unit='file')
        if _HAS_TQDM and verbose
        else tasks
    )

    for spd, ba in iterator:
        t0 = time.perf_counter()
        coord = gen.generate(speed=spd, boundary_avoidance=ba)
        traj_path = gen.save(coord, room_dir, speed=spd, boundary_avoidance=ba)
        elapsed = time.perf_counter() - t0

        if verbose and not _HAS_TQDM:
            ba_tag = 'BA' if ba else 'no-BA'
            print(
                f"  [{ba_tag:5s} {spd:4s}] {traj_path.name}  "
                f"shape={coord.shape}  ({elapsed:.1f} s)"
            )

    if verbose:
        print(f"[create_room] Done → {room_dir.resolve()}")

    return str(room_dir.resolve())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='generate_room',
        description=(
            'Generate a room (arena map + trajectories) that is compatible\n'
            'with the Wang et al. NeurIPS 2024 codebase.\n\n'
            'Examples:\n'
            '  uv run generate-room --width 150\n'
            '  uv run python -m trajectories.src.generate_room \\\n'
            '      --shape circle --width 200 --name circle_200 --ba-only\n'
            '  uv run python -m trajectories.src.generate_room \\\n'
            '      --shape triangle --width 100 --speeds med fast --seed 42'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Room geometry
    geo = p.add_argument_group('Room geometry')
    geo.add_argument(
        '--shape',
        choices=sorted(VALID_SHAPES),
        default=DEFAULT_SHAPE,
        help=(
            "Geometry of the accessible region.  'circle' and 'triangle' apply "
            "a mask to the circumscribed rectangle.  (default: square)"
        ),
    )
    geo.add_argument(
        '--width', type=int, default=100, metavar='CM',
        help='Room width in cm  (default: 100).',
    )
    geo.add_argument(
        '--height', type=int, default=None, metavar='CM',
        help='Room height in cm.  Defaults to --width.',
    )
    geo.add_argument(
        '--triangle-kind',
        choices=sorted(TRIANGLE_KINDS),
        default=DEFAULT_TRIANGLE_KIND,
        dest='triangle_kind',
        help=(
            "Triangle variant when --shape=triangle.  "
            "'right_isoceles': right angle at (x_min, y_min).  "
            "'equilateral': apex at (center_x, y_max).  "
            "(default: right_isoceles)"
        ),
    )
    geo.add_argument(
        '--padding', type=int, default=DEFAULT_PADDING, metavar='PX',
        help=f'Wall-padding width in pixels on every side  (default: {DEFAULT_PADDING}).',
    )

    # Output
    out = p.add_argument_group('Output')
    out.add_argument(
        '--name', type=str, default=None,
        help=(
            "Sub-directory name inside --output.  "
            "Defaults to '<width>x<height>_<shape>'."
        ),
    )
    out.add_argument(
        '--output', type=str, default=DEFAULT_OUTPUT_DIR, dest='output_dir',
        metavar='DIR',
        help=f'Root output directory  (default: {DEFAULT_OUTPUT_DIR}).',
    )

    # Trajectory options
    traj = p.add_argument_group('Trajectory options')
    traj.add_argument(
        '--speeds', nargs='+',
        choices=list(SPEED_NAMES),
        default=list(DEFAULT_SPEEDS),
        metavar='SPEED',
        help='Speed presets to generate  (default: slow med fast).',
    )

    ba_group = traj.add_mutually_exclusive_group()
    ba_group.add_argument(
        '--ba-only', action='store_true',
        help='Generate only boundary-avoidance (_ba) variants.',
    )
    ba_group.add_argument(
        '--no-ba-only', action='store_true',
        help='Generate only non-boundary-avoidance variants.',
    )

    traj.add_argument(
        '--n-trajectories', type=int, default=DEFAULT_N_TRAJECTORIES,
        dest='n_trajectories',
        metavar='N',
        help=f'Independent trajectories per file  (default: {DEFAULT_N_TRAJECTORIES}).',
    )
    traj.add_argument(
        '--duration', type=float, default=DEFAULT_DURATION_S, dest='duration_s',
        metavar='S',
        help=f'Duration per trajectory in seconds  (default: {DEFAULT_DURATION_S}).',
    )
    traj.add_argument(
        '--seed', type=int, default=None,
        help='Integer RNG seed for reproducibility  (default: none).',
    )

    # Verbosity
    p.add_argument(
        '--quiet', action='store_true',
        help='Suppress progress messages.',
    )

    return p


def main(argv: list[str] = None) -> None:
    """Entry point for CLI usage."""
    args = _build_parser().parse_args(argv)

    if args.ba_only:
        ba = (True,)
    elif args.no_ba_only:
        ba = (False,)
    else:
        ba = DEFAULT_BOUNDARY_AVOIDANCE

    create_room(
        shape=args.shape,
        width=args.width,
        height=args.height,
        name=args.name,
        speeds=args.speeds,
        boundary_avoidance=ba,
        n_trajectories=args.n_trajectories,
        duration_s=args.duration_s,
        padding=args.padding,
        seed=args.seed,
        output_dir=args.output_dir,
        triangle_kind=args.triangle_kind,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
