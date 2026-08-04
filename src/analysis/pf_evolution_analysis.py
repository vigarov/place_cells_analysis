"""
Computations for the thorough place-field (PF) cycle analysis notebook.

A cell's place field for a given `(room, cycle)` is the **sum of its fitted
2D Gaussians**. This module turns the per-visit Gaussian RF table into:

* rendered PF images across rooms x cycles for selected cells,
* amplitude-weighted scalar time-series (amplitude / x-centre / y-centre)
  describing how the PF of an active `(cell, room)` evolves over cycles,
* dispersion (std) and drift summaries of those time-series.

The amplitude weighting follows the spec: each Gaussian gets a weight
proportional to its amplitude relative to the largest component, normalised so
the weights sum to 1. This reduces to `w_k = amp_k / sum_j amp_j`.
"""

import itertools
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
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
from analysis.hk_distance import custom_hk

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
    (so the rooms x cycles PF grid is interesting), with mean `signal_max`
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
    Render the place field (sum of Gaussians) on a `render_shape` grid.

    The Gaussians are evaluated over the full `field_extent` coordinate range
    but sampled at `render_shape` resolution (coarser is faster for big
    grids). Returns `None` when any parameter is non-finite.
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

    The shared colour scale `vmax` is the 99th percentile of the peak PF
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

    For each `(cell, cycle, room)` the per-Gaussian amplitude defines weights
    `w_k = amp_k / sum_j amp_j` (clipping negative amplitudes to 0). The
    returned descriptors are the weighted means

    * `w_amplitude` = sum_k w_k * amp_k
    * `w_mu_x`      = sum_k w_k * mu_x_k
    * `w_mu_y`      = sum_k w_k * mu_y_k

    Rows with non-finite parameters or zero total amplitude get `NaN`
    descriptors. Columns: `cell_idx, cycle_id, room_id, signal_max, r2,
    w_amplitude, w_mu_x, w_mu_y`.
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
    out = source[["cell_idx", "cycle_id", "room_id", "signal_max", "r2"]].copy()
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
    Weighted PF descriptor time-series over cycles for one `(cell, room)`.

    Sorted by cycle. With `active_only` (default) only cycles where the cell
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
    Per `(cell, room)` dispersion of the weighted PF time-series.

    Returns `(long, wide)`:

    * `wide`: `cell_idx, room_id, amplitude, x_center, y_center` (the std
      of each weighted time-series), plus `n_cycles`.
    * `long`: melted version with columns `cell_idx, room_id, metric, std`
      (`metric` in `amplitude`/`x_center`/`y_center`) for seaborn.

    Only `(cell, room)` pairs with at least *min_active_cycles* qualifying
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

    For each `(cell, room)` the centre `(w_mu_x, w_mu_y)` is tracked across
    sorted cycles and the cumulative Euclidean step length is summed. Returns
    `cell_idx, room_id, total_drift, mean_step, n_cycles`.
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
    """Number of distinct rooms each cell is ever active in. Index `cell_idx`."""
    return (
        dead_df[dead_df["active"]]
        .groupby("cell_idx")["room_id"]
        .nunique()
        .rename("n_rooms_active")
    )


def count_place_fields_per_visit(
    df_receptive_fields: pd.DataFrame,
    *,
    n_gaussians: int,
    r2_min: float = 0.8,
    dead_threshold: float = DEFAULT_DEAD_THRESHOLD,
    pf_fraction_min: float = 0.1,
) -> pd.DataFrame:
    """
    Number of place fields per selected `(cell, cycle, room)` visit.

    A visit is *selected* when its fit is good (`r2 > r2_min`) and the cell is
    active (`signal_max > dead_threshold`). For a selected visit the number of
    place fields is the count of fitted Gaussians whose amplitude is at least
    `pf_fraction_min` of the largest Gaussian amplitude. Only visits with at
    least 2 such place fields are returned (by construction of the selection).

    Returns `cell_idx, cycle_id, room_id, n_place_fields`.
    """
    amp_cols = [f"g{k + 1}_amplitude" for k in range(n_gaussians)]
    source = df_receptive_fields.reset_index()
    selected = source[
        (source["r2"] > r2_min) & (source["signal_max"] > dead_threshold)
    ].copy()

    amps = selected[amp_cols].to_numpy(dtype=np.float64)
    finite = np.isfinite(amps).all(axis=1)
    selected = selected[finite]
    amps = amps[finite]

    max_amp = amps.max(axis=1)
    n_pf = (amps >= pf_fraction_min * max_amp[:, None]).sum(axis=1)

    out = selected[["cell_idx", "cycle_id", "room_id"]].copy()
    out["cell_idx"] = out["cell_idx"].astype(int)
    out["cycle_id"] = out["cycle_id"].astype(int)
    out["room_id"] = out["room_id"].astype(int)
    out["n_place_fields"] = n_pf.astype(int)
    out = out[out["n_place_fields"] >= 2].reset_index(drop=True)
    return out


def pf_count_cycle_stats(pf_count_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-cycle place-field count averaged over rooms then over cells.

    For each `(cell, cycle)` the place-field count is averaged over the cell's
    selected rooms; the per-cycle `mean` and `sem` are then taken across
    cells. Returns `cycle_id, mean, sem`.
    """
    per_cell_cycle = (
        pf_count_df.groupby(["cell_idx", "cycle_id"])["n_place_fields"]
        .mean()
        .reset_index(name="mean_n_place_fields")
    )
    stats = (
        per_cell_cycle.groupby("cycle_id")["mean_n_place_fields"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats["sem"] = stats["std"] / np.sqrt(stats["count"])
    return stats[["cycle_id", "mean", "sem"]]


def pre_post_pf_creations_df(
    df_rf_pre: pd.DataFrame,
    df_rf_post: pd.DataFrame,
    *,
    r2_min: float = 0.8,
    dead_threshold: float = DEFAULT_DEAD_THRESHOLD,
) -> pd.DataFrame:
    """
    Pre→post place-field *creations* from paired Gaussian RF tables.

    Uses the same visit selection as :func:`count_place_fields_per_visit`
    on the post-training fits (`r2 > r2_min`, `signal_max > dead_threshold`).
    A creation is a `(cell, cycle, room)` visit that is **dead** before
    training (`signal_max < dead_threshold`) and **selected/active** after.

    Returns ``visit_idx, cell_idx, cycle_id, room_id, signal_max_pre,
    signal_max_post, r2_pre, r2_post``.
    """
    pre = df_rf_pre.reset_index()
    post = df_rf_post.reset_index()
    merged = pre.merge(
        post,
        on=["visit_idx", "cell_idx"],
        suffixes=("_pre", "_post"),
        validate="one_to_one",
    )
    if not (merged["cycle_id_pre"] == merged["cycle_id_post"]).all():
        raise ValueError("cycle_id mismatch between pre and post RF tables")
    if not (merged["room_id_pre"] == merged["room_id_post"]).all():
        raise ValueError("room_id mismatch between pre and post RF tables")

    dead_pre = merged["signal_max_pre"] < dead_threshold
    active_post = (merged["r2_post"] > r2_min) & (
        merged["signal_max_post"] > dead_threshold
    )
    out = merged.loc[dead_pre & active_post].copy()
    out = out.rename(
        columns={
            "cycle_id_pre": "cycle_id",
            "room_id_pre": "room_id",
        }
    )
    return (
        out[
            [
                "visit_idx",
                "cell_idx",
                "cycle_id",
                "room_id",
                "signal_max_pre",
                "signal_max_post",
                "r2_pre",
                "r2_post",
            ]
        ]
        .astype(
            {
                "visit_idx": int,
                "cell_idx": int,
                "cycle_id": int,
                "room_id": int,
            }
        )
        .reset_index(drop=True)
    )


def pre_post_pf_creation_cycle_stats(
    creations_df: pd.DataFrame,
) -> pd.DataFrame:
    """Per-cycle counts of pre→post PF creations. ``cycle_id, n_creations``."""
    if creations_df.empty:
        return pd.DataFrame(columns=["cycle_id", "n_creations"])
    return (
        creations_df.groupby("cycle_id", sort=True)
        .size()
        .reset_index(name="n_creations")
        .astype({"cycle_id": int, "n_creations": int})
    )


def select_pre_post_activations_per_cycle(
    creations_df: pd.DataFrame,
    *,
    cycles: list[int],
    n_per_cycle: int,
    rank_col: str = "signal_max_post",
) -> pd.DataFrame:
    """
    Top *n_per_cycle* pre→post activations for each cycle.

    Ranked by descending *rank_col* (post-training ``signal_max`` by default).
    """
    frames: list[pd.DataFrame] = []
    for cycle_id in cycles:
        sub = creations_df[creations_df["cycle_id"] == cycle_id]
        sub = sub.sort_values(rank_col, ascending=False).head(n_per_cycle)
        frames.append(sub)
    if not frames:
        return creations_df.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=True)


def last_shown_activation(
    activations_df: pd.DataFrame,
    *,
    cycles: list[int],
    n_per_cycle: int,
) -> pd.Series | None:
    """Bottom-right populated slot in the per-cycle activation show grid."""
    for cycle_id in reversed(cycles):
        cycle_rows = (
            activations_df[activations_df["cycle_id"] == cycle_id]
            .sort_values("signal_max_post", ascending=False)
        )
        if not cycle_rows.empty:
            slot = min(n_per_cycle, len(cycle_rows)) - 1
            return cycle_rows.iloc[slot]
    return None


def visit_lookup_for_cell(
    df_receptive_fields: pd.DataFrame,
    cell_idx: int,
) -> pd.Series:
    """Map ``(cycle_id, room_id)`` to ``visit_idx`` for one cell."""
    sub = df_receptive_fields.xs(cell_idx, level="cell_idx").reset_index()
    return sub.set_index(["cycle_id", "room_id"])["visit_idx"].astype(int)


def render_visit_pf_field(
    df_receptive_fields: pd.DataFrame,
    visit_idx: int,
    cell_idx: int,
    *,
    n_gaussians: int,
    render_shape: tuple[int, int] = (40, 40),
    field_extent: tuple[int, int] = DEFAULT_GAUSSIAN_FIELD_SHAPE,
    param_names: tuple[str, ...] = GAUSSIAN_PARAM_NAMES,
) -> np.ndarray | None:
    """Render the Gaussian-sum PF for one ``(visit_idx, cell_idx)`` visit."""
    row = df_receptive_fields.loc[(visit_idx, cell_idx)]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    params = params_array_from_row(row, n_gaussians, param_names)
    return render_pf_sum_field(
        params,
        render_shape=render_shape,
        field_extent=field_extent,
    )


class _PfHkConfig(NamedTuple):
    render_shape: tuple[int, int]
    field_extent: tuple[int, int]
    hk_scale: float
    sinkhorn_error: float
    eps_target: float
    eps_init: float
    max_iter: int
    mass_eps: float


def _pf_cross_room_hk_one(
    cell_idx: int,
    cycle_id: int,
    room_params: dict[int, np.ndarray],
    config: _PfHkConfig,
) -> dict[str, float] | None:
    maps: dict[int, np.ndarray] = {}
    masses: dict[int, float] = {}
    for room, params in room_params.items():
        field = render_pf_sum_field(
            params,
            render_shape=config.render_shape,
            field_extent=config.field_extent,
        )
        if field is None:
            continue
        maps[room] = field
        masses[room] = float(np.nansum(field))
    valid_rooms = list(maps.keys())
    if len(valid_rooms) < 2:
        return None
    dists = [
        custom_hk(
            maps[r_a],
            maps[r_b],
            hk_scale=config.hk_scale,
            sinkhorn_error=config.sinkhorn_error,
            eps_target=config.eps_target,
            eps_init=config.eps_init,
            mass_eps=config.mass_eps,
            max_iter=config.max_iter,
        )
        for r_a, r_b in itertools.combinations(valid_rooms, 2)
    ]
    return {
        "cell_idx": cell_idx,
        "cycle_id": cycle_id,
        "mean_pairwise_hk": float(np.mean(dists)),
        "mean_mass": float(np.mean([masses[r] for r in valid_rooms])),
    }


def pf_cross_room_hk_stats(
    df_receptive_fields: pd.DataFrame,
    *,
    n_gaussians: int,
    r2_min: float = 0.8,
    dead_threshold: float = DEFAULT_DEAD_THRESHOLD,
    pf_fraction_min: float = 0.1,
    render_shape: tuple[int, int] = (40, 40),
    field_extent: tuple[int, int] = DEFAULT_GAUSSIAN_FIELD_SHAPE,
    param_names: tuple[str, ...] = GAUSSIAN_PARAM_NAMES,
    hk_scale: float = 1.0,
    sinkhorn_error: float = 1e-3,
    eps_target: float = 1e-2,
    eps_init: float = 1.0,
    max_iter: int = 1000,
    mass_eps: float = 1e-12,
    n_workers: int | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cross-room place-field divergence and mass per cycle.

    For the visits selected by :func:`count_place_fields_per_visit`, and for
    each `(cell, cycle)` active in at least 2 rooms, render the PF (Gaussian
    sum) of every active room and compute

    * the mean pairwise :func:`custom_hk` distance between rooms,
    * the mean total map mass (sum of rates) across rooms.

    These per `(cell, cycle)` values are then aggregated across cells.

    `n_workers` controls process parallelism over `(cell, cycle)` groups
    (defaults to all CPUs). Set `n_workers=1` for a sequential run.

    Returns `(hk_stats, mass_stats)` with columns `cycle_id, mean_hk,
    sem_hk` and `cycle_id, mean_mass, sem_mass` respectively.
    """
    selected = count_place_fields_per_visit(
        df_receptive_fields,
        n_gaussians=n_gaussians,
        r2_min=r2_min,
        dead_threshold=dead_threshold,
        pf_fraction_min=pf_fraction_min,
    )
    lookup = df_receptive_fields.reset_index().set_index(
        ["cell_idx", "cycle_id", "room_id"]
    )

    config = _PfHkConfig(
        render_shape=render_shape,
        field_extent=field_extent,
        hk_scale=hk_scale,
        sinkhorn_error=sinkhorn_error,
        eps_target=eps_target,
        eps_init=eps_init,
        max_iter=max_iter,
        mass_eps=mass_eps,
    )

    work_items: list[tuple[int, int, dict[int, np.ndarray]]] = []
    for (cell_idx, cycle_id), grp in selected.groupby(
        ["cell_idx", "cycle_id"], sort=True
    ):
        rooms = sorted(int(r) for r in grp["room_id"].unique())
        if len(rooms) < 2:
            continue
        room_params: dict[int, np.ndarray] = {}
        for room in rooms:
            row = lookup.loc[(cell_idx, cycle_id, room)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            room_params[room] = params_array_from_row(row, n_gaussians, param_names)
        work_items.append((int(cell_idx), int(cycle_id), room_params))

    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(n_workers, len(work_items)))

    per_cell_rows: list[dict[str, float]] = []
    if n_workers == 1:
        iterator = tqdm(
            work_items,
            desc="cross-room HK",
            disable=not show_progress,
            total=len(work_items),
        )
        for cell_idx, cycle_id, room_params in iterator:
            row = _pf_cross_room_hk_one(cell_idx, cycle_id, room_params, config)
            if row is not None:
                per_cell_rows.append(row)
    else:
        worker = partial(_pf_cross_room_hk_one, config=config)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(worker, cell_idx, cycle_id, room_params)
                for cell_idx, cycle_id, room_params in work_items
            ]
            iterator = tqdm(
                as_completed(futures),
                desc="cross-room HK",
                disable=not show_progress,
                total=len(futures),
            )
            for future in iterator:
                row = future.result()
                if row is not None:
                    per_cell_rows.append(row)

    per_cell = pd.DataFrame(
        per_cell_rows,
        columns=["cell_idx", "cycle_id", "mean_pairwise_hk", "mean_mass"],
    )

    hk_stats = (
        per_cell.groupby("cycle_id")["mean_pairwise_hk"]
        .agg(mean_hk="mean", _std="std", _n="count")
        .reset_index()
    )
    hk_stats["sem_hk"] = hk_stats["_std"] / np.sqrt(hk_stats["_n"])
    hk_stats = hk_stats[["cycle_id", "mean_hk", "sem_hk"]]

    mass_stats = (
        per_cell.groupby("cycle_id")["mean_mass"]
        .agg(mean_mass="mean", _std="std", _n="count")
        .reset_index()
    )
    mass_stats["sem_mass"] = mass_stats["_std"] / np.sqrt(mass_stats["_n"])
    mass_stats = mass_stats[["cycle_id", "mean_mass", "sem_mass"]]

    return hk_stats, mass_stats


class _PrePostHkConfig(NamedTuple):
    hk_scale: float
    sinkhorn_error: float
    eps_target: float
    eps_init: float
    max_iter: int
    mass_eps: float


def _pre_post_visit_hk_one(
    visit_idx: int,
    cell_idx: int,
    *,
    ratemaps_pre: np.ndarray,
    ratemaps_post: np.ndarray,
    cycle_ids: np.ndarray,
    room_ids: np.ndarray,
    config: _PrePostHkConfig,
) -> dict[str, float | int]:
    hk2 = custom_hk(
        ratemaps_pre[visit_idx, cell_idx],
        ratemaps_post[visit_idx, cell_idx],
        hk_scale=config.hk_scale,
        sinkhorn_error=config.sinkhorn_error,
        eps_target=config.eps_target,
        eps_init=config.eps_init,
        max_iter=config.max_iter,
        mass_eps=config.mass_eps,
    )
    return {
        "visit_idx": visit_idx,
        "cell_idx": cell_idx,
        "cycle_id": int(cycle_ids[visit_idx]),
        "room_id": int(room_ids[visit_idx]),
        "hk2": float(hk2),
        "hk": float(np.sqrt(hk2)),
    }


def pre_post_visit_hk_df(
    ratemaps_pre: np.ndarray,
    ratemaps_post: np.ndarray,
    cycle_ids: np.ndarray,
    room_ids: np.ndarray,
    *,
    hk_scale: float = 20.0,
    sinkhorn_error: float = 1e-2,
    eps_target: float = 1e-1,
    eps_init: float = 1.0,
    max_iter: int = 100,
    mass_eps: float = 1e-3,
    n_workers: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Pre- vs post-training HK distance for each `(visit, cell)` pair.

    `ratemaps_pre` and `ratemaps_post` are the arrays saved by
    `cycles_train.py` (shape `(n_visits, n_hidden, H, W)`). Pairs whose
    combined map mass falls below `mass_eps` are skipped.

    Returns a DataFrame sorted by descending `hk2` with columns
    `visit_idx, cell_idx, cycle_id, room_id, hk2, hk`.
    """
    if ratemaps_pre.shape != ratemaps_post.shape:
        raise ValueError(
            f"ratemaps_pre shape {ratemaps_pre.shape} != "
            f"ratemaps_post shape {ratemaps_post.shape}"
        )
    n_visits, n_cells, _, _ = ratemaps_post.shape
    config = _PrePostHkConfig(
        hk_scale=hk_scale,
        sinkhorn_error=sinkhorn_error,
        eps_target=eps_target,
        eps_init=eps_init,
        max_iter=max_iter,
        mass_eps=mass_eps,
    )
    pairs = [
        (v, c)
        for v in range(n_visits)
        for c in range(n_cells)
        if ratemaps_pre[v, c].sum() + ratemaps_post[v, c].sum() >= mass_eps
    ]
    if n_workers is None:
        n_workers = min(16, (os.cpu_count() or 1) * 2)
    n_workers = max(1, min(n_workers, len(pairs)))

    worker = partial(
        _pre_post_visit_hk_one,
        ratemaps_pre=ratemaps_pre,
        ratemaps_post=ratemaps_post,
        cycle_ids=cycle_ids,
        room_ids=room_ids,
        config=config,
    )
    rows: list[dict[str, float | int]] = []
    if n_workers == 1:
        iterator = tqdm(
            pairs,
            desc="pre/post visit HK",
            disable=not show_progress,
            total=len(pairs),
        )
        for visit_idx, cell_idx in iterator:
            rows.append(worker(visit_idx, cell_idx))
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(worker, visit_idx, cell_idx)
                for visit_idx, cell_idx in pairs
            ]
            iterator = tqdm(
                as_completed(futures),
                desc="pre/post visit HK",
                disable=not show_progress,
                total=len(futures),
            )
            for future in iterator:
                rows.append(future.result())

    if not rows:
        return pd.DataFrame(
            columns=[
                "visit_idx",
                "cell_idx",
                "cycle_id",
                "room_id",
                "hk2",
                "hk",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values("hk2", ascending=False)
        .reset_index(drop=True)
    )
