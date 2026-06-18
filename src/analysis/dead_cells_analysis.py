"""
Reusable "dead cell" characterization for the cycles experiment.

A cell is **dead** in a given ``(room, cycle)`` when the rate-map signal it
produces *after visiting that room* falls below a threshold. Activity is
recorded once per ``(cell, cycle, room)`` (one rate-map per room visit), so the
``(cell, cycle, room)`` key is unique.

The helpers here build the long-form dead/active dataframe and the aggregate
quantities derived from it (per-room/per-cycle dead proportions, recovery
counts, active-room counts). They are deliberately decoupled from the Gaussian
RF machinery so they can be reused elsewhere in the codebase.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

DEFAULT_DEAD_THRESHOLD: float = 0.1
DEFAULT_DEAD_THRESHOLDS: tuple[float, ...] = (0.1, 0.15, 0.2, 0.25, 0.3, 0.5)

DEAD_KEY_COLUMNS = ["cell_idx", "cycle_id", "room_id"]


def infer_grid_shape(df: pd.DataFrame) -> tuple[list[int], list[int]]:
    """
    Return ``(cycles, rooms)`` present in *df* (sorted unique values).

    The number of cycles and rooms is inferred from the data so that the
    same code works for any result granularity (e.g. 10 vs 30 cycles).
    """
    source = df.reset_index() if df.index.names != [None] else df
    cycles = sorted(int(c) for c in pd.unique(source["cycle_id"]))
    rooms = sorted(int(r) for r in pd.unique(source["room_id"]))
    return cycles, rooms


def build_signal_long_df(
    df_receptive_fields: pd.DataFrame,
    *,
    signal_col: str = "signal_max",
) -> pd.DataFrame:
    """
    Long table of the per-visit activity signal.

    Parameters
    ----------
    df_receptive_fields
        Per-visit RF dataframe (index ``(visit_idx, cell_idx)``) as built by
        :func:`analysis.gaussian_fit_cycle_analysis.build_cell_receptive_fields_df`.
    signal_col
        Column used as the activity signal (``signal_max`` by default).

    Returns
    -------
    DataFrame with columns ``cell_idx, cycle_id, room_id, signal_max`` and a
    unique ``(cell_idx, cycle_id, room_id)`` key.
    """
    source = df_receptive_fields.reset_index()
    missing = {"cell_idx", "cycle_id", "room_id", signal_col} - set(source.columns)
    if missing:
        raise KeyError(f"Missing columns for dead analysis: {sorted(missing)}")

    long = source[["cell_idx", "cycle_id", "room_id", signal_col]].copy()
    long = long.rename(columns={signal_col: "signal_max"})
    long["cell_idx"] = long["cell_idx"].astype(int)
    long["cycle_id"] = long["cycle_id"].astype(int)
    long["room_id"] = long["room_id"].astype(int)

    if long.duplicated(DEAD_KEY_COLUMNS).any():
        raise ValueError(
            "(cell_idx, cycle_id, room_id) is not unique; expected one visit "
            "per cell/cycle/room."
        )
    return long


def mark_dead(
    signal_long_df: pd.DataFrame,
    *,
    dead_threshold: float = DEFAULT_DEAD_THRESHOLD,
) -> pd.DataFrame:
    """Add boolean ``dead``/``active`` columns from a ``signal_max`` threshold."""
    out = signal_long_df.copy()
    out["dead"] = out["signal_max"] < dead_threshold
    out["active"] = ~out["dead"]
    out.attrs["dead_threshold"] = float(dead_threshold)
    return out


def build_dead_cells_df(
    df_receptive_fields: pd.DataFrame,
    *,
    dead_threshold: float = DEFAULT_DEAD_THRESHOLD,
    signal_col: str = "signal_max",
) -> pd.DataFrame:
    """
    Binary dead/active table with a unique ``(cell, cycle, room)`` key.

    Columns: ``cell_idx, cycle_id, room_id, signal_max, dead, active``.
    """
    long = build_signal_long_df(df_receptive_fields, signal_col=signal_col)
    return mark_dead(long, dead_threshold=dead_threshold)


def dead_proportion_by_room_cycle(dead_df: pd.DataFrame) -> pd.DataFrame:
    """
    Proportion of dead cells for every ``(room, cycle)``.

    Returns a long table ``room_id, cycle_id, proportion_dead, n_cells``.
    """
    grouped = dead_df.groupby(["room_id", "cycle_id"], sort=True)
    prop = grouped["dead"].mean().reset_index(name="proportion_dead")
    prop["n_cells"] = grouped["dead"].size().to_numpy()
    return prop


def dead_proportion_matrix(dead_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot of dead proportions: index ``room_id``, columns ``cycle_id``."""
    prop = dead_proportion_by_room_cycle(dead_df)
    return prop.pivot(index="room_id", columns="cycle_id", values="proportion_dead")


def dead_proportion_multi_threshold(
    signal_long_df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_DEAD_THRESHOLDS,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Dead proportions per ``(room, cycle)`` for several thresholds.

    Returns a long table ``dead_threshold, room_id, cycle_id, proportion_dead``.
    """
    frames: list[pd.DataFrame] = []
    iterator = tqdm(
        thresholds, desc="dead proportions / threshold", disable=not show_progress
    )
    for threshold in iterator:
        flagged = signal_long_df.assign(
            dead=signal_long_df["signal_max"] < threshold
        )
        prop = (
            flagged.groupby(["room_id", "cycle_id"], sort=True)["dead"]
            .mean()
            .reset_index(name="proportion_dead")
        )
        prop["dead_threshold"] = float(threshold)
        frames.append(prop)
    return pd.concat(frames, ignore_index=True)


def count_recoveries_per_cell_room(dead_df: pd.DataFrame) -> pd.DataFrame:
    """
    Recoveries per ``(cell, room)``.

    A *recovery* is a transition ``dead -> active`` between consecutive cycles
    for the same cell and room (the cell re-activates for a room after having
    gone dead). Returns ``cell_idx, room_id, n_recoveries``.
    """
    ordered = dead_df.sort_values(["cell_idx", "room_id", "cycle_id"])
    prev_dead = ordered.groupby(["cell_idx", "room_id"], sort=False)["dead"].shift(1)
    is_recovery = prev_dead.fillna(False).astype(bool) & (~ordered["dead"])
    ordered = ordered.assign(recovery=is_recovery)
    out = (
        ordered.groupby(["cell_idx", "room_id"], sort=True)["recovery"]
        .sum()
        .astype(int)
        .reset_index(name="n_recoveries")
    )
    return out


def count_recoveries_per_cell(dead_df: pd.DataFrame) -> pd.Series:
    """Total recoveries per cell (summed over rooms). Index: ``cell_idx``."""
    per_cell_room = count_recoveries_per_cell_room(dead_df)
    return (
        per_cell_room.groupby("cell_idx")["n_recoveries"]
        .sum()
        .astype(int)
        .rename("n_recoveries")
    )


def recoveries_multi_threshold(
    signal_long_df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_DEAD_THRESHOLDS,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Per-cell recovery counts for several thresholds.

    Returns a long table ``dead_threshold, cell_idx, n_recoveries`` (one row
    per cell per threshold), suitable for a seaborn violin plot.
    """
    frames: list[pd.DataFrame] = []
    iterator = tqdm(
        thresholds, desc="recoveries / threshold", disable=not show_progress
    )
    for threshold in iterator:
        dead_df = mark_dead(signal_long_df, dead_threshold=threshold)
        per_cell = count_recoveries_per_cell(dead_df).reset_index()
        per_cell["dead_threshold"] = float(threshold)
        frames.append(per_cell)
    return pd.concat(frames, ignore_index=True)


def active_rooms_per_cell_cycle(dead_df: pd.DataFrame) -> pd.DataFrame:
    """
    Number of rooms each cell is active for, per cycle.

    Within a cycle every room is visited once; this counts, per ``(cell,
    cycle)``, how many of those room visits were active. Returns ``cell_idx,
    cycle_id, n_active_rooms``.
    """
    out = (
        dead_df.groupby(["cell_idx", "cycle_id"], sort=True)["active"]
        .sum()
        .astype(int)
        .reset_index(name="n_active_rooms")
    )
    return out


def active_cell_room_pairs(
    dead_df: pd.DataFrame,
    *,
    min_active_cycles: int = 1,
) -> pd.DataFrame:
    """
    ``(cell, room)`` pairs active in at least *min_active_cycles* cycles.

    Returns ``cell_idx, room_id, n_active_cycles``.
    """
    counts = (
        dead_df[dead_df["active"]]
        .groupby(["cell_idx", "room_id"], sort=True)
        .size()
        .reset_index(name="n_active_cycles")
    )
    return counts[counts["n_active_cycles"] >= min_active_cycles].reset_index(
        drop=True
    )


def pf_onset_cycle(dead_df: pd.DataFrame) -> pd.DataFrame:
    """First cycle each ``(cell, room)`` becomes active. ``onset_cycle`` col."""
    active = dead_df[dead_df["active"]]
    return (
        active.groupby(["cell_idx", "room_id"], sort=True)["cycle_id"]
        .min()
        .reset_index(name="onset_cycle")
    )


def active_fraction_by_cycle(dead_df: pd.DataFrame) -> pd.DataFrame:
    """
    Population activity per cycle.

    Returns ``cycle_id, active_fraction, n_active_cell_rooms,
    n_distinct_active_cells``. ``active_fraction`` is over all ``(cell, room)``
    visits in the cycle.
    """
    grouped = dead_df.groupby("cycle_id", sort=True)
    out = grouped["active"].mean().reset_index(name="active_fraction")
    out["n_active_cell_rooms"] = grouped["active"].sum().astype(int).to_numpy()
    distinct = (
        dead_df[dead_df["active"]]
        .groupby("cycle_id")["cell_idx"]
        .nunique()
        .reindex(out["cycle_id"], fill_value=0)
        .to_numpy()
    )
    out["n_distinct_active_cells"] = distinct.astype(int)
    return out
