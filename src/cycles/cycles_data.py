"""Load cycles rooms, manifest, and visit schedules."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cycles.cycles_paths import MANIFEST_PATH, ROOMS_DIR
from cycles.constants import (
    ARENA_MAP_KEY,
    COORD_KEY,
    DT,
    WSM_RESPONSE_KEY,
)
from core.weak_sm_cell import WeakSMCell


@dataclass(frozen=True)
class RoomSpec:
    index: int
    name: str
    wsm_seed: int
    traj_seed: int
    rel_dir: str


@dataclass(frozen=True)
class CyclesManifest:
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
    rooms: tuple[RoomSpec, ...]

    @property
    def traj_filename(self) -> str:
        ba = "_ba" if self.boundary_avoidance else ""
        return f"traj_{self.speed}{ba}.npz"


def trial_steps_from_duration(duration_s: float, dt: float = DT) -> int:
    return int(round(duration_s / dt))


def load_manifest(path: Path | None = None) -> CyclesManifest:
    path = Path(path or MANIFEST_PATH)
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    rooms = tuple(
        RoomSpec(
            index=r["index"],
            name=r["name"],
            wsm_seed=r["wsm_seed"],
            traj_seed=r["traj_seed"],
            rel_dir=r["rel_dir"],
        )
        for r in raw["rooms"]
    )
    return CyclesManifest(
        experiment=raw["experiment"],
        n_rooms=raw["n_rooms"],
        room_width_cm=raw["room_width_cm"],
        trial_duration_s=raw["trial_duration_s"],
        trial_steps=raw["trial_steps"],
        dt_s=raw["dt_s"],
        speed=raw["speed"],
        boundary_avoidance=raw["boundary_avoidance"],
        n_trajectories=raw["n_trajectories"],
        wsm_n_cells=raw["wsm_n_cells"],
        wsm_sigma=raw["wsm_sigma"],
        wsm_ssigma=raw["wsm_ssigma"],
        wsm_magnitude=raw["wsm_magnitude"],
        base_seed=raw.get("base_seed"),
        rooms=rooms,
    )


def load_room(
    room: RoomSpec | str | int,
    *,
    manifest: CyclesManifest | None = None,
    rooms_dir: Path | None = None,
    cycles_dir: Path | None = None,
):
    """
    Load arena map, trajectory coordinates, and WSM for one cycles room.

    Room files are read from ``data/cycles/<room_name>/``.

    Returns
    -------
    arena_map : np.ndarray
    traj_coord : np.ndarray, shape `(B, T, 2)`
    wsm : WeakSMCell
    room_spec : RoomSpec
    """
    manifest = manifest or load_manifest()
    rooms_dir = Path(rooms_dir or cycles_dir or ROOMS_DIR)

    if isinstance(room, int):
        room_spec = manifest.rooms[room - 1]
    elif isinstance(room, str):
        room_spec = next(r for r in manifest.rooms if r.name == room)
    else:
        room_spec = room

    room_dir = rooms_dir / room_spec.rel_dir
    arena_map = np.load(room_dir / "arena_map.npz")[ARENA_MAP_KEY]
    traj_coord = np.load(room_dir / manifest.traj_filename)[COORD_KEY]
    response_map = np.load(room_dir / "wsm_response_map.npz")[WSM_RESPONSE_KEY]

    wsm = WeakSMCell(
        arena_map=arena_map,
        n_cells=manifest.wsm_n_cells,
        sigma=manifest.wsm_sigma,
        ssigma=manifest.wsm_ssigma,
        magnitude=manifest.wsm_magnitude,
        seed=room_spec.wsm_seed,
    )
    wsm.response_map = response_map
    return arena_map, traj_coord, wsm, room_spec


def build_visit_schedule(
    n_cycles: int,
    n_rooms: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """
    Shuffled room order within each cycle.

    Returns
    -------
    schedule : ndarray, shape `(n_cycles, n_rooms)`, dtype int
        `schedule[c, k]` is the 1-based room index visited k-th in cycle `c`.
    """
    rng = np.random.default_rng(seed)
    schedule = np.zeros((n_cycles, n_rooms), dtype=np.int32)
    for c in range(n_cycles):
        schedule[c] = rng.permutation(np.arange(1, n_rooms + 1))
    return schedule


def flatten_schedule(schedule: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten `(n_cycles, n_rooms)` schedule to per-visit arrays.

    Returns
    -------
    cycle_ids, room_ids, visit_indices : each shape `(n_visits,)`
    """
    n_cycles, n_rooms = schedule.shape
    n_visits = n_cycles * n_rooms
    cycle_ids = np.repeat(np.arange(n_cycles), n_rooms)
    room_ids = schedule.reshape(-1)
    visit_indices = np.arange(n_visits, dtype=np.int32)
    return cycle_ids, room_ids, visit_indices
