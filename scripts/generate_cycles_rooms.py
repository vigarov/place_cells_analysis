#!/usr/bin/env python3
"""
Generate rooms for the Wang et al. (NeurIPS 2024) multi-room cycles experiment.

This setup reproduces Supplemental Figure 2 (place-field drift across 30 cycles
in Room 1) and the related orthogonality analysis (Suppl. Sec. 3.2 / Fig. 1):
20 unique 100×100 cm square enclosures, each with its own weakly spatially
modulated (WSM) experience map and trajectories.

Output layout
-------------
<output_dir>/
  manifest.json
  room_01/
    arena_map.npz
    traj_<speed>.npz          # rodent-like random walk
    wsm_response_map.npz      # unique WSM per room (key: response_map)
  ...
  room_20/

Paper defaults (Sec. 3.4, Suppl. Sec. 3.2)
------------------------------------------
- 20 rooms, 100 cm × 100 cm squares
- 10 min per room visit during training → 600 s → 12 000 steps at dt = 50 ms
- Unique WSM set per room (independent RNG seeds)

Usage
-----
    uv run python scripts/generate_cycles_rooms.py

    uv run python scripts/generate_cycles_rooms.py --output trajectories/cycles --seed 0

    # Shorter smoke test
    uv run python scripts/generate_cycles_rooms.py --n-rooms 2 --trial-duration-s 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Repository root on sys.path (script may be invoked without editable install).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cycles.constants import (  # noqa: E402
    CYCLES_TRIAL_DURATION_S,
    DEFAULT_CYCLES_BOUNDARY_AVOIDANCE,
    DEFAULT_CYCLES_SPEED,
    DEFAULT_N_TRAJECTORIES,
    DT,
    WSM_RESPONSE_KEY,
)
from cycles.cycles_data import trial_steps_from_duration  # noqa: E402
from trajectories.src.constants import DEFAULT_PADDING  # noqa: E402
from trajectories.src.room_generator import make_square_room, save_arena_map  # noqa: E402
from trajectories.src.trajectory_generator import TrajectoryGenerator  # noqa: E402
from weak_sm_cell import WeakSMCell  # noqa: E402

# Cycles experiment (main text Sec. 3.4, Suppl. Sec. 3.2)
DEFAULT_N_ROOMS = 20
DEFAULT_ROOM_WIDTH_CM = 100
DEFAULT_TRIAL_DURATION_S = CYCLES_TRIAL_DURATION_S
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "trajectories" / "cycles"
DEFAULT_SPEED = DEFAULT_CYCLES_SPEED
DEFAULT_BOUNDARY_AVOIDANCE = DEFAULT_CYCLES_BOUNDARY_AVOIDANCE

# WSM / EV defaults (match demo.ipynb and cell_evolutions/cell_evolution.ipynb)
DEFAULT_N_WSM_CELLS = 200
DEFAULT_WSM_SIGMA = 12
DEFAULT_WSM_SSIGMA = 0
DEFAULT_WSM_MAGNITUDE = 4.0

MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class RoomSpec:
    """Metadata for one cycles-experiment room."""

    index: int
    name: str
    wsm_seed: int
    traj_seed: int
    rel_dir: str


@dataclass(frozen=True)
class CyclesManifest:
    """Top-level manifest written beside room directories."""

    experiment: str
    n_rooms: int
    room_width_cm: int
    trial_duration_s: float
    trial_steps: int
    dt_s: float
    speed: str
    boundary_avoidance: bool
    n_trajectories: int
    wsm_n_cells: int
    wsm_sigma: int
    wsm_ssigma: int
    wsm_magnitude: float
    base_seed: int | None
    rooms: list[RoomSpec]


def _room_name(index: int) -> str:
    return f"room_{index:02d}"


def _save_wsm(response_map: np.ndarray, room_dir: Path) -> Path:
    path = room_dir / "wsm_response_map.npz"
    np.savez_compressed(path, **{WSM_RESPONSE_KEY: response_map})
    return path


def generate_cycles_rooms(
    *,
    n_rooms: int = DEFAULT_N_ROOMS,
    room_width_cm: int = DEFAULT_ROOM_WIDTH_CM,
    trial_duration_s: float = DEFAULT_TRIAL_DURATION_S,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    speed: str = DEFAULT_SPEED,
    boundary_avoidance: bool = DEFAULT_BOUNDARY_AVOIDANCE,
    n_trajectories: int = DEFAULT_N_TRAJECTORIES,
    n_wsm_cells: int = DEFAULT_N_WSM_CELLS,
    wsm_sigma: int = DEFAULT_WSM_SIGMA,
    wsm_ssigma: int = DEFAULT_WSM_SSIGMA,
    wsm_magnitude: float = DEFAULT_WSM_MAGNITUDE,
    base_seed: int | None = 0,
    verbose: bool = True,
) -> Path:
    """
    Create *n_rooms* square rooms with unique WSM maps and trajectory files.

    Returns
    -------
    Path
        The output directory containing ``manifest.json`` and room sub-folders.
    """
    if n_rooms < 1:
        raise ValueError("n_rooms must be at least 1.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arena = make_square_room(room_width_cm, room_width_cm, padding=DEFAULT_PADDING)
    trial_steps = trial_steps_from_duration(trial_duration_s)

    room_specs: list[RoomSpec] = []

    if verbose:
        print(
            f"[cycles] Generating {n_rooms} rooms → {output_dir.resolve()}\n"
            f"  geometry: {room_width_cm}×{room_width_cm} cm square\n"
            f"  trial duration: {trial_duration_s}s ({trial_steps} steps)\n"
            f"  trajectories: speed={speed!r}, BA={boundary_avoidance}, "
            f"n={n_trajectories}\n"
            f"  WSM: n_cells={n_wsm_cells}, sigma={wsm_sigma}, "
            f"magnitude={wsm_magnitude}"
        )

    t_all = time.perf_counter()

    for i in range(1, n_rooms + 1):
        name = _room_name(i)
        room_dir = output_dir / name
        room_dir.mkdir(parents=True, exist_ok=True)

        wsm_seed = (base_seed + i) if base_seed is not None else None
        traj_seed = (base_seed + 10_000 + i) if base_seed is not None else None

        t0 = time.perf_counter()

        save_arena_map(arena, room_dir)

        gen = TrajectoryGenerator(
            arena_map=arena,
            n_trajectories=n_trajectories,
            duration_s=trial_duration_s,
            dt=DT,
            seed=traj_seed,
        )
        coord = gen.generate(speed=speed, boundary_avoidance=boundary_avoidance)
        traj_path = gen.save(coord, room_dir, speed=speed, boundary_avoidance=boundary_avoidance)

        wsm = WeakSMCell(
            arena_map=arena,
            n_cells=n_wsm_cells,
            sigma=wsm_sigma,
            ssigma=wsm_ssigma,
            magnitude=wsm_magnitude,
            seed=wsm_seed,
        )
        wsm_path = _save_wsm(wsm.response_map, room_dir)

        room_specs.append(
            RoomSpec(
                index=i,
                name=name,
                wsm_seed=wsm_seed if wsm_seed is not None else -1,
                traj_seed=traj_seed if traj_seed is not None else -1,
                rel_dir=name,
            )
        )

        if verbose:
            elapsed = time.perf_counter() - t0
            print(
                f"  [{i:2d}/{n_rooms}] {name}  "
                f"traj {coord.shape} → {traj_path.name}  "
                f"wsm {wsm.response_map.shape}  ({elapsed:.1f}s)"
            )

    manifest = CyclesManifest(
        experiment="cycles_20rooms_30cycles",
        n_rooms=n_rooms,
        room_width_cm=room_width_cm,
        trial_duration_s=trial_duration_s,
        trial_steps=trial_steps,
        dt_s=DT,
        speed=speed,
        boundary_avoidance=boundary_avoidance,
        n_trajectories=n_trajectories,
        wsm_n_cells=n_wsm_cells,
        wsm_sigma=wsm_sigma,
        wsm_ssigma=wsm_ssigma,
        wsm_magnitude=wsm_magnitude,
        base_seed=base_seed,
        rooms=room_specs,
    )

    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(
            {
                **{k: v for k, v in asdict(manifest).items() if k != "rooms"},
                "rooms": [asdict(r) for r in manifest.rooms],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if verbose:
        total = time.perf_counter() - t_all
        print(f"[cycles] Done in {total:.1f}s — manifest → {manifest_path}")

    return output_dir.resolve()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate 20-room dataset for the cycles / Suppl. Fig. 2 experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory for manifest and room_* sub-folders.",
    )
    p.add_argument("--n-rooms", type=int, default=DEFAULT_N_ROOMS, metavar="N")
    p.add_argument("--width", type=int, default=DEFAULT_ROOM_WIDTH_CM, metavar="CM")
    p.add_argument(
        "--trial-duration-s",
        type=float,
        default=DEFAULT_TRIAL_DURATION_S,
        dest="trial_duration_s",
        metavar="S",
        help="Trajectory length per file (paper: 600 s = 10 min per room visit).",
    )
    p.add_argument("--speed", choices=("slow", "med", "fast"), default=DEFAULT_SPEED)
    p.add_argument(
        "--boundary-avoidance",
        action="store_true",
        default=DEFAULT_BOUNDARY_AVOIDANCE,
        help="Use traj_<speed>_ba.npz (default: plain traj_<speed>.npz).",
    )
    p.add_argument(
        "--n-trajectories",
        type=int,
        default=DEFAULT_N_TRAJECTORIES,
        metavar="N",
    )
    p.add_argument("--n-wsm-cells", type=int, default=DEFAULT_N_WSM_CELLS, metavar="D")
    p.add_argument("--wsm-sigma", type=int, default=DEFAULT_WSM_SIGMA)
    p.add_argument("--wsm-ssigma", type=int, default=DEFAULT_WSM_SSIGMA)
    p.add_argument("--wsm-magnitude", type=float, default=DEFAULT_WSM_MAGNITUDE)
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed; room i uses seed+i (WSM) and seed+10000+i (trajectories). "
        "Pass a negative value for non-deterministic generation.",
    )
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    base_seed = None if args.seed < 0 else args.seed

    out = generate_cycles_rooms(
        n_rooms=args.n_rooms,
        room_width_cm=args.width,
        trial_duration_s=args.trial_duration_s,
        output_dir=args.output,
        speed=args.speed,
        boundary_avoidance=args.boundary_avoidance,
        n_trajectories=args.n_trajectories,
        n_wsm_cells=args.n_wsm_cells,
        wsm_sigma=args.wsm_sigma,
        wsm_ssigma=args.wsm_ssigma,
        wsm_magnitude=args.wsm_magnitude,
        base_seed=base_seed,
        verbose=not args.quiet,
    )
    if args.quiet:
        print(out)


if __name__ == "__main__":
    main()
