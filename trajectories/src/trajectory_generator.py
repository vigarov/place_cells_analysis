"""
trajectory_generator.py
=======================
Biased random-walk trajectory simulator.

Reproduces the trajectory generation described in:

    Wang et al. (NeurIPS 2024) "Time Makes Space: Emergence of Place Fields in
    Networks Encoding Temporally Continuous Sensory Experiences"
    Supplemental Section 1.1 + README speed / boundary-avoidance presets.

Algorithm (per timestep, dt = 50 ms)
-------------------------------------
1.  heading  += drift
2.  [BA only] Steer heading away from nearby walls proportionally to
    `(1 - dist_to_wall / avoid_boundary_dist)`.
3.  new_pos   = pos + speed * [cos(heading), sin(heading)] * dt
4.  If new_pos is inside a wall pixel: reflect heading about the boundary
    normal (gradient of the precomputed distance field) and stay at pos.
5.  With prob `switch_velocity_prob` : resample speed ~ max(0.1, N(μ, σ)).
6.  With prob `switch_direction_prob`: resample drift ~ N(0, drift_magnitude).

Output format
-------------
Key `'coord'`, shape `(B, n_steps, 2)`, dtype `float64`.
Coordinates are in centimetres (1 px = 1 cm) with the 5 px padding offset
included.  Range: `[padding, padding + room_dim]` per axis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from scipy.ndimage import distance_transform_edt

from .constants import (
    COORD_KEY,
    DEFAULT_DURATION_S,
    DEFAULT_N_TRAJECTORIES,
    DT,
    MIN_SPEED,
    NO_BOUNDARY_AVOIDANCE,
    SPEED_PRESETS,
)


# ---------------------------------------------------------------------------
# Trajectory generator
# ---------------------------------------------------------------------------

class TrajectoryGenerator:
    """
    Generate biased random-walk trajectories over an arena map.

    All `B` trajectories are simulated in a single vectorised pass
    (one numpy operation per timestep across all trajectories), so
    generation of the full 128 x 40 960-step dataset completes in a
    few seconds on a modern CPU.

    Parameters
    ----------
    arena_map       : np.ndarray, shape `(n_x, n_y)`.
                      `0` = accessible, `1` = wall.
    n_trajectories  : Number of independent trajectories (default 128).
    duration_s      : Duration of each trajectory in seconds (default 2048,
                      i.e., ≈ 34 minutes; gives 40 960 steps at dt=0.05).
    dt              : Simulation timestep in seconds (default 0.05 = 50 ms).
    seed            : Integer seed passed to `np.random.default_rng`.
                      `None` produces non-deterministic results.
    """

    def __init__(
        self,
        arena_map: np.ndarray,
        n_trajectories: int = DEFAULT_N_TRAJECTORIES,
        duration_s: float = DEFAULT_DURATION_S,
        dt: float = DT,
        seed: int = None,
    ) -> None:
        self.arena_map = arena_map.astype(np.float64)
        self.n_trajectories = n_trajectories
        self.duration_s = duration_s
        self.dt = dt
        self.n_steps = int(round(duration_s / dt))
        self.seed = seed

        self._n_x, self._n_y = arena_map.shape

        # Accessible-pixel indices for random initialisation
        self._accessible = np.argwhere(arena_map == 0)
        if self._accessible.size == 0:
            raise ValueError("arena_map contains no accessible pixels (all walls).")

        # Distance field: dist_field[i, j] = Euclidean distance from pixel
        # (i, j) to the nearest wall pixel.  Zero at wall pixels, positive
        # inside the accessible region.
        acc_mask = (arena_map == 0).astype(np.uint8)
        self._dist_field = distance_transform_edt(acc_mask).astype(np.float32)

        # Gradient of the distance field: points away from the nearest wall.
        self._grad_x = np.gradient(self._dist_field, axis=0).astype(np.float32)
        self._grad_y = np.gradient(self._dist_field, axis=1).astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        speed: str = 'med',
        boundary_avoidance: bool = True,
    ) -> np.ndarray:
        """
        Simulate all trajectories and return the coordinate array.

        Parameters
        ----------
        speed             : Speed preset -- one of `'slow'`, `'med'`,
                            `'fast'`.
        boundary_avoidance: If `True`, uses `avoid_boundary_dist` from the
                            preset to steer agents away from walls (generates the
                            `_ba` file variant).  If `False`, sets
                            `avoid_boundary_dist = -1` so agents rely on
                            pure reflection (generates the non-`_ba` variant).

        Returns
        -------
        coord : np.ndarray, shape `(n_trajectories, n_steps, 2)`,
                dtype `float64`.  Last axis is `(x, y)` in cm.
        """
        if speed not in SPEED_PRESETS:
            raise ValueError(
                f"speed must be one of {list(SPEED_PRESETS)}; got '{speed}'."
            )

        params = dict(SPEED_PRESETS[speed])
        if not boundary_avoidance:
            params['avoid_boundary_dist'] = NO_BOUNDARY_AVOIDANCE

        rng = np.random.default_rng(self.seed)
        return self._simulate(params, rng)

    def save(
        self,
        coord: np.ndarray,
        output_dir: Union[str, Path],
        speed: str,
        boundary_avoidance: bool,
    ) -> Path:
        """
        Save a coordinate array as `traj_<speed>[_ba].npz`.

        Parameters
        ----------
        coord             : np.ndarray, shape `(B, n_steps, 2)`.
        output_dir        : Directory in which to write the file.
                            Created recursively if absent.
        speed             : Speed label used in the file name.
        boundary_avoidance: Appends `_ba` to the file name when `True`.

        Returns
        -------
        Path to the saved file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = '_ba' if boundary_avoidance else ''
        path = output_dir / f'traj_{speed}{suffix}.npz'
        np.savez_compressed(path, **{COORD_KEY: coord})
        return path

    # ------------------------------------------------------------------
    # Core simulation (vectorised over the B-trajectory batch)
    # ------------------------------------------------------------------

    def _simulate(self, params: dict, rng: np.random.Generator) -> np.ndarray:
        B = self.n_trajectories
        n_steps = self.n_steps
        dt = self.dt

        v_mean = float(params['velocity_mean'])
        v_sd = float(params['velocity_sd'])
        drift_mag = float(params['random_drift_magnitude'])
        p_dir = float(params['switch_direction_prob'])
        p_vel = float(params['switch_velocity_prob'])
        avoid_dist = float(params['avoid_boundary_dist'])

        # Max valid array indices (0-based)
        max_x = self._n_x - 1
        max_y = self._n_y - 1

        # ---- Initialise state ----
        idx = rng.integers(0, len(self._accessible), B)
        positions = self._accessible[idx].astype(np.float64)   # (B, 2)

        headings = rng.uniform(0.0, 2.0 * np.pi, B)            # (B,)
        speeds   = np.maximum(MIN_SPEED, rng.normal(v_mean, v_sd, B))
        drifts   = rng.normal(0.0, drift_mag, B)

        all_coords = np.empty((B, n_steps, 2), dtype=np.float64)

        # ---- Main loop ----
        for t in range(n_steps):
            all_coords[:, t] = positions

            # 1. Accumulate angular momentum drift
            headings += drifts

            # 2. Boundary avoidance: smoothly steer away from walls (BA only)
            if avoid_dist > 0:
                xi = np.round(positions[:, 0]).astype(np.intp).clip(0, max_x)
                yi = np.round(positions[:, 1]).astype(np.intp).clip(0, max_y)
                dist_wall = self._dist_field[xi, yi]

                ba_mask = dist_wall < avoid_dist
                if ba_mask.any():
                    # Weight: 0 at avoid_dist, 1 at the wall surface
                    weight = np.where(ba_mask, 1.0 - dist_wall / avoid_dist, 0.0)

                    gx = self._grad_x[xi, yi]
                    gy = self._grad_y[xi, yi]
                    gnorm = np.sqrt(gx ** 2 + gy ** 2).clip(1e-8)
                    repulsion = np.arctan2(gy / gnorm, gx / gnorm)

                    # Angular difference, wrapped to (−π, π]
                    dtheta = repulsion - headings
                    dtheta = (dtheta + np.pi) % (2.0 * np.pi) - np.pi
                    headings += weight * dtheta

            # 3. Proposed displacement
            cos_h = np.cos(headings)
            sin_h = np.sin(headings)
            new_positions = positions + np.stack(
                [speeds * cos_h * dt, speeds * sin_h * dt], axis=1
            )

            # 4. Boundary reflection: if proposed position is in a wall,
            #    reflect the heading about the local wall normal and stay put.
            new_xi = np.round(new_positions[:, 0]).astype(np.intp).clip(0, max_x)
            new_yi = np.round(new_positions[:, 1]).astype(np.intp).clip(0, max_y)
            in_wall = self.arena_map[new_xi, new_yi] == 1.0

            if in_wall.any():
                # Evaluate gradient (= outward wall normal) at current position
                xi_c = np.round(positions[:, 0]).astype(np.intp).clip(0, max_x)
                yi_c = np.round(positions[:, 1]).astype(np.intp).clip(0, max_y)

                gx = self._grad_x[xi_c, yi_c]
                gy = self._grad_y[xi_c, yi_c]
                gnorm = np.sqrt(gx ** 2 + gy ** 2).clip(1e-8)
                nx_ = gx / gnorm
                ny_ = gy / gnorm

                # Specular reflection: v_ref = v − 2(v·n̂)n̂
                dot = cos_h * nx_ + sin_h * ny_
                vx_ref = cos_h - 2.0 * dot * nx_
                vy_ref = sin_h - 2.0 * dot * ny_
                headings_ref = np.arctan2(vy_ref, vx_ref)

                headings     = np.where(in_wall, headings_ref, headings)
                new_positions = np.where(
                    in_wall[:, np.newaxis], positions, new_positions
                )

            positions = new_positions

            # 5. Stochastic parameter resampling
            mask_vel = rng.random(B) < p_vel
            if mask_vel.any():
                new_spd = np.maximum(MIN_SPEED, rng.normal(v_mean, v_sd, B))
                speeds = np.where(mask_vel, new_spd, speeds)

            mask_dir = rng.random(B) < p_dir
            if mask_dir.any():
                new_dft = rng.normal(0.0, drift_mag, B)
                drifts = np.where(mask_dir, new_dft, drifts)

        return all_coords
