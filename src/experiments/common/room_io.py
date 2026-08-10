import json
from pathlib import Path
from typing import Any

import numpy as np

from core.eval_trajectories import generate_eval_trajectories, print_coverage
from core.weak_sm_cell import WeakSMCell
from trajectories.constants import ARENA_MAP_KEY, COORD_KEY, DT
from trajectories.room_generator import save_arena_map
from trajectories.trajectory_generator import TrajectoryGenerator

WSM_RESPONSE_KEY = "response_map"


def room_dir_name(index: int) -> str:
    return f"room_{index + 1:02d}"


def write_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def read_manifest(data_dir: Path) -> dict[str, Any]:
    return json.loads((data_dir / "manifest.json").read_text())


def resolve_room_dir(data_dir: Path, manifest: dict[str, Any], room_index: int) -> Path:
    """Directory holding one room's `arena_map.npz`, `traj.npz`, etc."""
    if "rooms" in manifest:
        if room_index < 0 or room_index >= len(manifest["rooms"]):
            raise IndexError(
                f"room_index {room_index} out of range for {len(manifest['rooms'])} rooms"
            )
        return data_dir / manifest["rooms"][room_index]["rel_dir"]
    if room_index != 0:
        raise IndexError("single-room layout only supports room_index=0")
    return data_dir


def room_spec(manifest: dict[str, Any], room_index: int) -> dict[str, Any]:
    if "rooms" in manifest:
        return manifest["rooms"][room_index]
    return manifest


def generate_train_trajectories(
    arena_map: np.ndarray,
    *,
    n_trajectories: int,
    duration_s: float,
    seed: int,
    speed: str,
    boundary_avoidance: bool,
    dt: float = DT,
) -> np.ndarray:
    gen = TrajectoryGenerator(
        arena_map,
        n_trajectories=n_trajectories,
        duration_s=duration_s,
        dt=dt,
        seed=seed,
    )
    return gen.generate(speed=speed, boundary_avoidance=boundary_avoidance)


def write_train_trajectories(room_dir: Path, traj_coord: np.ndarray) -> None:
    np.savez_compressed(room_dir / "traj.npz", **{COORD_KEY: traj_coord})


def generate_and_write_eval_trajectories(
    arena_map: np.ndarray,
    room_dir: Path,
    *,
    n_traj: int,
    duration_s: float,
    seed: int,
    speed: str,
    boundary_avoidance: bool,
    coverage_label: str | None = None,
    dt: float = DT,
) -> np.ndarray:
    eval_traj_coord = generate_eval_trajectories(
        arena_map,
        n_traj=n_traj,
        duration_s=duration_s,
        dt=dt,
        seed=seed,
        speed=speed,
        boundary_avoidance=boundary_avoidance,
    )
    np.savez_compressed(room_dir / "eval_traj.npz", **{COORD_KEY: eval_traj_coord})
    if coverage_label is not None:
        print_coverage(arena_map, eval_traj_coord, label=coverage_label)
    return eval_traj_coord


def build_wsm(
    arena_map: np.ndarray,
    *,
    n_cells: int,
    sigma: int,
    ssigma: int,
    magnitude: float,
    seed: int,
    bias_strength: float | None = None,
    bias_corner: str | None = None,
    bias_type: str | None = None,
) -> WeakSMCell:
    kwargs: dict[str, Any] = {}
    if bias_strength is not None:
        kwargs["bias_strength"] = bias_strength
    if bias_corner is not None:
        kwargs["bias_corner"] = bias_corner
    if bias_type is not None:
        kwargs["bias_type"] = bias_type
    return WeakSMCell(
        arena_map=arena_map,
        n_cells=n_cells,
        sigma=sigma,
        ssigma=ssigma,
        magnitude=magnitude,
        seed=seed,
        **kwargs,
    )


def write_wsm(wsm: WeakSMCell, room_dir: Path) -> None:
    np.savez_compressed(room_dir / "wsm_response_map.npz", **{WSM_RESPONSE_KEY: wsm.response_map})


def write_room_bundle(
    room_dir: Path,
    *,
    arena_map: np.ndarray,
    traj_coord: np.ndarray,
    eval_traj_coord: np.ndarray,
    wsm: WeakSMCell,
) -> None:
    """Write one room's arena, trajectories, and WSM to `room_dir`."""
    room_dir.mkdir(parents=True, exist_ok=True)
    save_arena_map(arena_map, room_dir)
    write_train_trajectories(room_dir, traj_coord)
    np.savez_compressed(room_dir / "eval_traj.npz", **{COORD_KEY: eval_traj_coord})
    write_wsm(wsm, room_dir)


def load_room_arrays(room_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arena_map = np.load(room_dir / "arena_map.npz")[ARENA_MAP_KEY]
    traj_coord = np.load(room_dir / "traj.npz")[COORD_KEY]
    eval_traj_coord = np.load(room_dir / "eval_traj.npz")[COORD_KEY]
    response_map = np.load(room_dir / "wsm_response_map.npz")[WSM_RESPONSE_KEY]
    return arena_map, traj_coord, eval_traj_coord, response_map


def load_wsm(
    arena_map: np.ndarray,
    manifest: dict[str, Any],
    spec: dict[str, Any],
    response_map: np.ndarray,
) -> WeakSMCell:
    kwargs: dict[str, Any] = {}
    if "bias_strength" in manifest:
        kwargs["bias_strength"] = manifest["bias_strength"]
    if "bias_type" in manifest:
        kwargs["bias_type"] = manifest["bias_type"]
    if "bias_corner" in spec:
        kwargs["bias_corner"] = spec["bias_corner"]
    wsm = build_wsm(
        arena_map,
        n_cells=manifest["n_wsm_cells"],
        sigma=manifest["wsm_sigma"],
        ssigma=manifest["wsm_ssigma"],
        magnitude=manifest["wsm_magnitude"],
        seed=spec["wsm_seed"],
        **kwargs,
    )
    wsm.response_map = response_map
    return wsm


def load_room_from_disk(
    data_dir: Path,
    *,
    room_index: int = 0,
    manifest: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, WeakSMCell]:
    """Load arena, train traj, eval traj, and WSM for one room (0-based index)."""
    data_dir = Path(data_dir)
    manifest = manifest or read_manifest(data_dir)
    room_dir = resolve_room_dir(data_dir, manifest, room_index)
    spec = room_spec(manifest, room_index)
    arena_map, traj_coord, eval_traj_coord, response_map = load_room_arrays(room_dir)
    wsm = load_wsm(arena_map, manifest, spec, response_map)
    return arena_map, traj_coord, eval_traj_coord, wsm
