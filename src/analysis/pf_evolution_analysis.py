"""
Computations for the thorough place-field (PF) cycle analysis notebook.

A cell's place field for a given ``(room, cycle)`` is the **sum of its fitted
2D Gaussians**. This module turns the per-visit Gaussian RF table into:

* rendered PF images across rooms x cycles for selected cells,
* amplitude-weighted scalar time-series (amplitude / x-centre / y-centre)
  describing how the PF of an active ``(cell, room)`` evolves over cycles,
* dispersion (std) and drift summaries of those time-series.

The amplitude weighting follows the spec: each Gaussian gets a weight
proportional to its amplitude relative to the largest component, normalised so
the weights sum to 1. This reduces to ``w_k = amp_k / sum_j amp_j``.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from analysis.sum_gaussians_core import _gaussian_2d
from analysis.dead_cells_analysis import DEFAULT_DEAD_THRESHOLD
from analysis.gaussian_fit_cycle_analysis import (
    DEFAULT_GAUSSIAN_FIELD_SHAPE,
    GAUSSIAN_PARAM_NAMES,
    params_array_from_row,
)

WEIGHTED_METRICS = ("amplitude", "x_center", "y_center")


def select_active_cells(
    dead_df: pd.DataFrame,
    df_receptive_fields: pd.DataFrame,
    *,
    n: int = 3,
    signal_col: str = "signal_max",
) -> list[int]:
    """
    Pick *n* representative active cells.

    Cells are ranked by the number of distinct rooms they are ever active in
    (so the rooms x cycles PF grid is interesting), with mean ``signal_max``
    as a tie-breaker.
    """
    rooms_active = (
        dead_df[dead_df["active"]].groupby("cell_idx")["room_id"].nunique()
    )
    mean_max = df_receptive_fields.reset_index().groupby("cell_idx")[
        signal_col
    ].mean()
    score = pd.DataFrame({"n_active_rooms": rooms_active, "mean_signal_max": mean_max})
    score["n_active_rooms"] = score["n_active_rooms"].fillna(0).astype(int)
    score = score.sort_values(
        ["n_active_rooms", "mean_signal_max"], ascending=False
    )
    return [int(c) for c in score.index[:n]]


def render_pf_sum_field(
    params: np.ndarray,
    *,
    render_shape: tuple[int, int] = (48, 48),
    field_extent: tuple[int, int] = DEFAULT_GAUSSIAN_FIELD_SHAPE,
) -> np.ndarray | None:
    """
    Render the place field (sum of Gaussians) on a ``render_shape`` grid.

    The Gaussians are evaluated over the full ``field_extent`` coordinate range
    but sampled at ``render_shape`` resolution (coarser is faster for big
    grids). Returns ``None`` when any parameter is non-finite.
    """
    params = np.asarray(params, dtype=np.float64)
    if params.ndim != 2 or params.shape[1] != 5:
        raise ValueError(f"Expected params (n_gaussians, 5), got {params.shape}")
    if not np.all(np.isfinite(params)):
        return None

    h_extent, w_extent = field_extent
    h_render, w_render = render_shape
    xs = np.linspace(0, w_extent - 1, w_render)
    ys = np.linspace(0, h_extent - 1, h_render)
    xx, yy = np.meshgrid(xs, ys)
    total = np.zeros((h_render, w_render), dtype=np.float64)
    for k in range(params.shape[0]):
        total += _gaussian_2d(params[k], xx, yy)
    return total


class CellPfGrid(NamedTuple):
    """Rendered PF grid (rooms x cycles) for one cell."""

    cell_idx: int
    cycles: list[int]
    rooms: list[int]
    fields: dict[tuple[int, int], np.ndarray | None]
    signal_max: dict[tuple[int, int], float]
    dead: dict[tuple[int, int], bool]
    vmax: float


def build_cell_pf_grid(
    df_receptive_fields: pd.DataFrame,
    cell_idx: int,
    *,
    n_gaussians: int,
    dead_threshold: float = DEFAULT_DEAD_THRESHOLD,
    render_shape: tuple[int, int] = (48, 48),
    field_extent: tuple[int, int] = DEFAULT_GAUSSIAN_FIELD_SHAPE,
    param_names: tuple[str, ...] = GAUSSIAN_PARAM_NAMES,
    show_progress: bool = True,
) -> CellPfGrid:
    """
    Render the PF (Gaussian sum) for one cell across every room and cycle.

    The shared colour scale ``vmax`` is the 99th percentile of the peak PF
    value across *active* (cell visited & above threshold) panels, so a few
    pathological fits do not wash out the grid.
    """
    sub = df_receptive_fields.xs(cell_idx, level="cell_idx").reset_index()
    cycles = sorted(int(c) for c in sub["cycle_id"].unique())
    rooms = sorted(int(r) for r in sub["room_id"].unique())

    fields: dict[tuple[int, int], np.ndarray | None] = {}
    signal_max: dict[tuple[int, int], float] = {}
    dead: dict[tuple[int, int], bool] = {}
    active_peaks: list[float] = []

    iterator = tqdm(
        sub.itertuples(index=False),
        total=len(sub),
        desc=f"render PF grid cell {cell_idx}",
        disable=not show_progress,
    )
    for row in iterator:
        row_series = pd.Series(row._asdict())
        key = (int(row_series["room_id"]), int(row_series["cycle_id"]))
        params = params_array_from_row(row_series, n_gaussians, param_names)
        field = render_pf_sum_field(
            params, render_shape=render_shape, field_extent=field_extent
        )
        sig = float(row_series["signal_max"])
        is_dead = sig < dead_threshold
        fields[key] = field
        signal_max[key] = sig
        dead[key] = is_dead
        if field is not None and not is_dead:
            active_peaks.append(float(np.nanmax(field)))

    if active_peaks:
        vmax = float(np.percentile(active_peaks, 99))
    else:
        finite_peaks = [
            float(np.nanmax(f)) for f in fields.values() if f is not None
        ]
        vmax = float(np.percentile(finite_peaks, 99)) if finite_peaks else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return CellPfGrid(cell_idx, cycles, rooms, fields, signal_max, dead, vmax)


def compute_weighted_pf_df(
    df_receptive_fields: pd.DataFrame,
    *,
    n_gaussians: int,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Amplitude-weighted scalar PF descriptors for every visit (vectorised).

    For each ``(cell, cycle, room)`` the per-Gaussian amplitude defines weights
    ``w_k = amp_k / sum_j amp_j`` (clipping negative amplitudes to 0). The
    returned descriptors are the weighted means

    * ``w_amplitude`` = sum_k w_k * amp_k
    * ``w_mu_x``      = sum_k w_k * mu_x_k
    * ``w_mu_y``      = sum_k w_k * mu_y_k

    Rows with non-finite parameters or zero total amplitude get ``NaN``
    descriptors. Columns: ``cell_idx, cycle_id, room_id, signal_max,
    w_amplitude, w_mu_x, w_mu_y``.
    """
    amp_cols = [f"g{k + 1}_amplitude" for k in range(n_gaussians)]
    mux_cols = [f"g{k + 1}_mu_x" for k in range(n_gaussians)]
    muy_cols = [f"g{k + 1}_mu_y" for k in range(n_gaussians)]

    amps = df_receptive_fields[amp_cols].to_numpy(dtype=np.float64)
    mux = df_receptive_fields[mux_cols].to_numpy(dtype=np.float64)
    muy = df_receptive_fields[muy_cols].to_numpy(dtype=np.float64)

    finite = (
        np.isfinite(amps).all(axis=1)
        & np.isfinite(mux).all(axis=1)
        & np.isfinite(muy).all(axis=1)
    )
    amps_pos = np.where(finite[:, None], np.clip(amps, 0.0, None), np.nan)
    sum_amp = amps_pos.sum(axis=1)
    valid = finite & (sum_amp > eps)

    safe_sum = np.where(valid, sum_amp, np.nan)
    weights = amps_pos / safe_sum[:, None]
    w_amplitude = np.where(valid, np.nansum(weights * amps_pos, axis=1), np.nan)
    w_mu_x = np.where(valid, np.nansum(weights * mux, axis=1), np.nan)
    w_mu_y = np.where(valid, np.nansum(weights * muy, axis=1), np.nan)

    source = df_receptive_fields.reset_index()
    out = source[["cell_idx", "cycle_id", "room_id", "signal_max"]].copy()
    out["cell_idx"] = out["cell_idx"].astype(int)
    out["cycle_id"] = out["cycle_id"].astype(int)
    out["room_id"] = out["room_id"].astype(int)
    out["w_amplitude"] = w_amplitude
    out["w_mu_x"] = w_mu_x
    out["w_mu_y"] = w_mu_y
    return out


def weighted_pf_timeseries(
    weighted_pf_df: pd.DataFrame,
    cell_idx: int,
    room_id: int,
    *,
    dead_threshold: float = DEFAULT_DEAD_THRESHOLD,
    active_only: bool = True,
) -> pd.DataFrame:
    """
    Weighted PF descriptor time-series over cycles for one ``(cell, room)``.

    Sorted by cycle. With ``active_only`` (default) only cycles where the cell
    is active in the room (and the fit is finite) are kept, i.e. the evolution
    of the field while it is "alive".
    """
    sub = weighted_pf_df[
        (weighted_pf_df["cell_idx"] == cell_idx)
        & (weighted_pf_df["room_id"] == room_id)
    ].copy()
    sub = sub.sort_values("cycle_id")
    sub["active"] = sub["signal_max"] >= dead_threshold
    keep = sub["w_amplitude"].notna()
    if active_only:
        keep = keep & sub["active"]
    return sub[keep].reset_index(drop=True)


def pf_variation_std_table(
    weighted_pf_df: pd.DataFrame,
    *,
    dead_threshold: float = DEFAULT_DEAD_THRESHOLD,
    min_active_cycles: int = 2,
    active_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per ``(cell, room)`` dispersion of the weighted PF time-series.

    Returns ``(long, wide)``:

    * ``wide``: ``cell_idx, room_id, amplitude, x_center, y_center`` (the std
      of each weighted time-series), plus ``n_cycles``.
    * ``long``: melted version with columns ``cell_idx, room_id, metric, std``
      (``metric`` in ``amplitude``/``x_center``/``y_center``) for seaborn.

    Only ``(cell, room)`` pairs with at least *min_active_cycles* qualifying
    cycles are included.
    """
    work = weighted_pf_df.copy()
    work["active"] = work["signal_max"] >= dead_threshold
    keep = work["w_amplitude"].notna()
    if active_only:
        keep = keep & work["active"]
    work = work[keep]

    sizes = work.groupby(["cell_idx", "room_id"])["cycle_id"].transform("size")
    work = work[sizes >= min_active_cycles]

    wide = (
        work.groupby(["cell_idx", "room_id"])
        .agg(
            amplitude=("w_amplitude", "std"),
            x_center=("w_mu_x", "std"),
            y_center=("w_mu_y", "std"),
            n_cycles=("cycle_id", "size"),
        )
        .reset_index()
    )
    long = wide.melt(
        id_vars=["cell_idx", "room_id", "n_cycles"],
        value_vars=list(WEIGHTED_METRICS),
        var_name="metric",
        value_name="std",
    )
    return long, wide


def weighted_center_drift(
    weighted_pf_df: pd.DataFrame,
    *,
    dead_threshold: float = DEFAULT_DEAD_THRESHOLD,
    min_active_cycles: int = 2,
    active_only: bool = True,
) -> pd.DataFrame:
    """
    Total path length travelled by the weighted PF centre over cycles.

    For each ``(cell, room)`` the centre ``(w_mu_x, w_mu_y)`` is tracked across
    sorted cycles and the cumulative Euclidean step length is summed. Returns
    ``cell_idx, room_id, total_drift, mean_step, n_cycles``.
    """
    work = weighted_pf_df.copy()
    work["active"] = work["signal_max"] >= dead_threshold
    keep = work["w_amplitude"].notna()
    if active_only:
        keep = keep & work["active"]
    work = work[keep].sort_values(["cell_idx", "room_id", "cycle_id"])

    grouped = work.groupby(["cell_idx", "room_id"], sort=False)
    dx = grouped["w_mu_x"].diff()
    dy = grouped["w_mu_y"].diff()
    work = work.assign(step=np.sqrt(dx**2 + dy**2))

    agg = (
        work.groupby(["cell_idx", "room_id"], sort=True)
        .agg(
            total_drift=("step", "sum"),
            mean_step=("step", "mean"),
            n_cycles=("cycle_id", "size"),
        )
        .reset_index()
    )
    return agg[agg["n_cycles"] >= min_active_cycles].reset_index(drop=True)


def rooms_active_per_cell(dead_df: pd.DataFrame) -> pd.Series:
    """Number of distinct rooms each cell is ever active in. Index ``cell_idx``."""
    return (
        dead_df[dead_df["active"]]
        .groupby("cell_idx")["room_id"]
        .nunique()
        .rename("n_rooms_active")
    )
