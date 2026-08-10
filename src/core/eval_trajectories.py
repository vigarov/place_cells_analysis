"""Held-out evaluation trajectories for activation (rate-map) capture.

For a full spatial rate map, a room must be visited at
* every position,
* from multiple directions,
* under multiple input-masking patterns.

Instead of  reusing training trajectories for this we use a dedicated
set of "evaluation" trajectories per room (not trained on) use only to compute rate maps.

(Ref: `local/ratemap_traj_impact.ipynb` --> ~128 trajectories of ~350s each
.
"""
import numpy as np

from trajectories.constants import DT, FREE_SPACE
from trajectories.trajectory_generator import TrajectoryGenerator

DEFAULT_N_EVAL_TRAJ = 128
DEFAULT_EVAL_TRAJ_DURATION_S = 350.0


def generate_eval_trajectories(
    arena_map: np.ndarray,
    *,
    n_traj: int = DEFAULT_N_EVAL_TRAJ,
    duration_s: float = DEFAULT_EVAL_TRAJ_DURATION_S,
    dt: float = DT,
    seed: int | None = None,
    speed: str = "fast",
    boundary_avoidance: bool = True,
) -> np.ndarray:
    """Generates `n_traj` trajectories of `duration_s` seconds each.

    Use wit a `seed` distinct from training-trajectory seed used for the same room...

    Returns
    -------
    np.ndarray, shape (n_traj, round(duration_s / dt), 2)
    """
    generator = TrajectoryGenerator(
        arena_map, n_trajectories=n_traj, duration_s=duration_s, dt=dt, seed=seed
    )
    return generator.generate(speed=speed, boundary_avoidance=boundary_avoidance)


def coverage_fraction(arena_map: np.ndarray, traj_coord: np.ndarray) -> float:
    """Fraction of accessible (non-wall) pixels visited at least once by `traj_coord`."""
    accessible = arena_map == FREE_SPACE
    n_accessible = int(accessible.sum())
    if n_accessible == 0:
        return 0.0

    idx = np.round(traj_coord).astype(int).reshape(-1, 2)
    xi = np.clip(idx[:, 0], 0, arena_map.shape[0] - 1)
    yi = np.clip(idx[:, 1], 0, arena_map.shape[1] - 1)

    visited = np.zeros(arena_map.shape, dtype=bool)
    visited[xi, yi] = True
    visited &= accessible
    return float(visited.sum()) / n_accessible


def print_coverage(
    arena_map: np.ndarray,
    traj_coord: np.ndarray,
    *,
    label: str = "eval trajectories",
) -> float:
    """Print, and return, the % of accessible room pixels visited by `traj_coord`."""
    frac = coverage_fraction(arena_map, traj_coord)
    print(
        f"[{label}] visited {frac * 100:.2f}% of accessible room pixels "
        f"({traj_coord.shape[0]} trajectories x {traj_coord.shape[1]} steps each)"
    )
    return frac
