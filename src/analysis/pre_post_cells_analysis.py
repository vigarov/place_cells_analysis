"""
Pre/post paired activity states for the cycles experiment.

At each ``(cell, cycle, room)`` visit we compare pre- and post-training
Gaussian fits and label the pair as one of AA, AD, DA, DD (active/dead in
pre vs post). Active requires ``signal_max >= dead_threshold``; when
``r2_min_selection`` is set, ``r2 > r2_min_selection`` is applied per
``r2_selection_policy`` (``"pre"``, ``"post"``, or ``"pre-post"``).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from analysis.dead_cells_analysis import (
    DEFAULT_DEAD_THRESHOLD,
    DEFAULT_DEAD_THRESHOLDS,
)

PRE_POST_STATES: tuple[str, ...] = ("AA", "AD", "DA", "DD")
DEFAULT_R2_MIN_SELECTION: float = 0.75
R2SelectionPolicy = Literal["pre", "post", "pre-post"]
R2_SELECTION_POLICIES: tuple[R2SelectionPolicy, ...] = ("pre", "post", "pre-post")
DEFAULT_R2_SELECTION_POLICY: R2SelectionPolicy = "pre-post"


def _state_label(active_pre: pd.Series, active_post: pd.Series) -> pd.Series:
    """Map paired active flags to AA / AD / DA / DD strings."""
    pre = active_pre.astype(bool)
    post = active_post.astype(bool)
    labels = np.where(
        pre & post,
        "AA",
        np.where(pre & ~post, "AD", np.where(~pre & post, "DA", "DD")),
    )
    return pd.Series(labels, index=active_pre.index, dtype="string")


def _apply_r2_gate(phase: Literal["pre", "post"], policy: R2SelectionPolicy) -> bool:
    """Whether the r2 gate applies to *phase* under *policy*."""
    if policy not in R2_SELECTION_POLICIES:
        raise ValueError(
            f"r2_selection_policy must be one of {R2_SELECTION_POLICIES}, "
            f"got {policy!r}"
        )
    return policy == "pre-post" or policy == phase


def _phase_active(
    signal_max: pd.Series,
    r2: pd.Series,
    *,
    dead_threshold: float,
    r2_min_selection: float | None,
    apply_r2: bool,
) -> pd.Series:
    """Active when ``signal_max >= dead_threshold`` (and optionally ``r2`` gate)."""
    active = signal_max >= dead_threshold
    if r2_min_selection is not None and apply_r2:
        active = active & (r2 > r2_min_selection)
    return active


def active_criterion_label(
    *,
    dead_threshold: float,
    r2_min_selection: float | None,
    r2_selection_policy: R2SelectionPolicy = DEFAULT_R2_SELECTION_POLICY,
) -> str:
    """Human-readable summary of the active/dead selection rule."""
    label = f"threshold={dead_threshold}"
    if r2_min_selection is None:
        return label
    if r2_selection_policy == "pre-post":
        return f"{label}, r2>{r2_min_selection} (pre & post)"
    if r2_selection_policy == "pre":
        return f"{label}, r2>{r2_min_selection} (pre only)"
    return f"{label}, r2>{r2_min_selection} (post only)"


def pre_post_state_df(
    df_rf_pre: pd.DataFrame,
    df_rf_post: pd.DataFrame,
    *,
    dead_threshold: float = DEFAULT_DEAD_THRESHOLD,
    r2_min_selection: float | None = DEFAULT_R2_MIN_SELECTION,
    r2_selection_policy: R2SelectionPolicy = DEFAULT_R2_SELECTION_POLICY,
) -> pd.DataFrame:
    """
    Paired pre/post state table with a unique ``(cell, cycle, room)`` key.

    Active when ``signal_max >= dead_threshold``. If *r2_min_selection* is
    not ``None``, also require ``r2 > r2_min_selection`` on phases selected
    by *r2_selection_policy*.

    Returns columns ``visit_idx, cell_idx, cycle_id, room_id, signal_max_pre,
    signal_max_post, r2_pre, r2_post, active_pre, active_post, state``.
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

    out = merged.rename(
        columns={
            "cycle_id_pre": "cycle_id",
            "room_id_pre": "room_id",
        }
    ).copy()
    out["active_pre"] = _phase_active(
        out["signal_max_pre"],
        out["r2_pre"],
        dead_threshold=dead_threshold,
        r2_min_selection=r2_min_selection,
        apply_r2=_apply_r2_gate("pre", r2_selection_policy),
    )
    out["active_post"] = _phase_active(
        out["signal_max_post"],
        out["r2_post"],
        dead_threshold=dead_threshold,
        r2_min_selection=r2_min_selection,
        apply_r2=_apply_r2_gate("post", r2_selection_policy),
    )
    out["state"] = _state_label(out["active_pre"], out["active_post"])
    out.attrs["dead_threshold"] = float(dead_threshold)
    out.attrs["r2_min_selection"] = r2_min_selection
    out.attrs["r2_selection_policy"] = r2_selection_policy

    return out[
        [
            "visit_idx",
            "cell_idx",
            "cycle_id",
            "room_id",
            "signal_max_pre",
            "signal_max_post",
            "r2_pre",
            "r2_post",
            "active_pre",
            "active_post",
            "state",
        ]
    ].astype(
        {
            "visit_idx": int,
            "cell_idx": int,
            "cycle_id": int,
            "room_id": int,
        }
    )


def state_proportion_by_room_cycle(state_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fraction of cells in each state for every ``(room, cycle)``.

    Returns ``room_id, cycle_id, state, proportion, n_cells``.
    """
    grouped = state_df.groupby(["room_id", "cycle_id", "state"], sort=True)
    counts = grouped.size().reset_index(name="n_state")
    totals = (
        state_df.groupby(["room_id", "cycle_id"], sort=True)
        .size()
        .reset_index(name="n_cells")
    )
    out = counts.merge(totals, on=["room_id", "cycle_id"], how="left")
    out["proportion"] = out["n_state"] / out["n_cells"]
    return out[["room_id", "cycle_id", "state", "proportion", "n_cells"]]


def state_proportion_matrix(state_df: pd.DataFrame, state: str) -> pd.DataFrame:
    """Pivot of one state's proportions: index ``room_id``, columns ``cycle_id``."""
    prop = state_proportion_by_room_cycle(state_df)
    sub = prop[prop["state"] == state]
    return sub.pivot(index="room_id", columns="cycle_id", values="proportion")


def dd_proportion_multi_threshold(
    df_rf_pre: pd.DataFrame,
    df_rf_post: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_DEAD_THRESHOLDS,
    r2_min_selection: float | None = DEFAULT_R2_MIN_SELECTION,
    r2_selection_policy: R2SelectionPolicy = DEFAULT_R2_SELECTION_POLICY,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    DD proportions per ``(room, cycle)`` for several thresholds.

    Returns ``dead_threshold, room_id, cycle_id, proportion_dd``.
    """
    frames: list[pd.DataFrame] = []
    iterator = tqdm(
        thresholds, desc="DD proportions / threshold", disable=not show_progress
    )
    for threshold in iterator:
        state_df = pre_post_state_df(
            df_rf_pre,
            df_rf_post,
            dead_threshold=threshold,
            r2_min_selection=r2_min_selection,
            r2_selection_policy=r2_selection_policy,
        )
        dd = (
            state_df[state_df["state"] == "DD"]
            .groupby(["room_id", "cycle_id"], sort=True)
            .size()
            .reset_index(name="n_dd")
        )
        totals = (
            state_df.groupby(["room_id", "cycle_id"], sort=True)
            .size()
            .reset_index(name="n_cells")
        )
        prop = totals.merge(dd, on=["room_id", "cycle_id"], how="left")
        prop["n_dd"] = prop["n_dd"].fillna(0).astype(int)
        prop["proportion_dd"] = prop["n_dd"] / prop["n_cells"]
        prop["dead_threshold"] = float(threshold)
        frames.append(prop[["dead_threshold", "room_id", "cycle_id", "proportion_dd"]])
    return pd.concat(frames, ignore_index=True)


def da_activations_per_cell(state_df: pd.DataFrame) -> pd.Series:
    """
    Count of DA visits per cell (summed over rooms and cycles).

    Index: ``cell_idx``; name: ``n_da_activations``.
    """
    da = state_df[state_df["state"] == "DA"]
    return (
        da.groupby("cell_idx")
        .size()
        .astype(int)
        .rename("n_da_activations")
    )


def da_activations_multi_threshold(
    df_rf_pre: pd.DataFrame,
    df_rf_post: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_DEAD_THRESHOLDS,
    r2_min_selection: float | None = DEFAULT_R2_MIN_SELECTION,
    r2_selection_policy: R2SelectionPolicy = DEFAULT_R2_SELECTION_POLICY,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Per-cell DA activation counts for several thresholds.

    Returns ``dead_threshold, cell_idx, n_da_activations``.
    """
    frames: list[pd.DataFrame] = []
    iterator = tqdm(
        thresholds, desc="DA activations / threshold", disable=not show_progress
    )
    for threshold in iterator:
        state_df = pre_post_state_df(
            df_rf_pre,
            df_rf_post,
            dead_threshold=threshold,
            r2_min_selection=r2_min_selection,
            r2_selection_policy=r2_selection_policy,
        )
        per_cell = da_activations_per_cell(state_df).reset_index()
        per_cell["dead_threshold"] = float(threshold)
        frames.append(per_cell)
    return pd.concat(frames, ignore_index=True)


def ever_da_pairs(state_df: pd.DataFrame) -> pd.DataFrame:
    """
    ``(cell, room)`` pairs with at least one DA visit.

    Returns ``cell_idx, room_id, first_da_cycle``.
    """
    da = state_df[state_df["state"] == "DA"]
    return (
        da.groupby(["cell_idx", "room_id"], sort=True)["cycle_id"]
        .min()
        .reset_index(name="first_da_cycle")
        .astype({"cell_idx": int, "room_id": int, "first_da_cycle": int})
    )


def pre_da_aa_ad_count_by_cycle(state_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count ever-DA ``(cell, room)`` pairs in AA or AD strictly before first DA.

    At cycle *c*, counts pairs with ``first_da_cycle > c`` whose state at *c*
    is AA or AD. Returns ``cycle_id, count``.
    """
    pairs = ever_da_pairs(state_df)
    if pairs.empty:
        return pd.DataFrame(columns=["cycle_id", "count"])

    merged = state_df.merge(pairs, on=["cell_idx", "room_id"], how="inner")
    before = merged[merged["cycle_id"] < merged["first_da_cycle"]]
    aa_ad = before[before["state"].isin(["AA", "AD"])]
    counts = (
        aa_ad.groupby("cycle_id", sort=True)
        .size()
        .astype(int)
        .reset_index(name="count")
    )
    all_cycles = sorted(int(c) for c in state_df["cycle_id"].unique())
    return (
        counts.set_index("cycle_id")
        .reindex(all_cycles, fill_value=0)
        .reset_index()
        .astype({"cycle_id": int, "count": int})
    )


def pre_da_aa_ad_count_multi_threshold(
    df_rf_pre: pd.DataFrame,
    df_rf_post: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_DEAD_THRESHOLDS,
    r2_min_selection: float | None = DEFAULT_R2_MIN_SELECTION,
    r2_selection_policy: R2SelectionPolicy = DEFAULT_R2_SELECTION_POLICY,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Pre-first-DA AA/AD pair counts per cycle for several thresholds.

    Returns ``dead_threshold, cycle_id, count``.
    """
    frames: list[pd.DataFrame] = []
    iterator = tqdm(
        thresholds,
        desc="pre-DA AA/AD counts / threshold",
        disable=not show_progress,
    )
    for threshold in iterator:
        state_df = pre_post_state_df(
            df_rf_pre,
            df_rf_post,
            dead_threshold=threshold,
            r2_min_selection=r2_min_selection,
            r2_selection_policy=r2_selection_policy,
        )
        counts = pre_da_aa_ad_count_by_cycle(state_df)
        counts["dead_threshold"] = float(threshold)
        frames.append(counts)
    return pd.concat(frames, ignore_index=True)


def ever_da_state_counts_by_cycle(state_df: pd.DataFrame) -> pd.DataFrame:
    """
    State counts per cycle among ``(cell, room)`` pairs with ≥1 DA.

    Returns ``cycle_id, state, count``.
    """
    pairs = ever_da_pairs(state_df)[["cell_idx", "room_id"]]
    if pairs.empty:
        return pd.DataFrame(columns=["cycle_id", "state", "count"])

    subset = state_df.merge(pairs, on=["cell_idx", "room_id"], how="inner")
    return (
        subset.groupby(["cycle_id", "state"], sort=True)
        .size()
        .astype(int)
        .reset_index(name="count")
    )


def _all_dd_before_first_da(
    state_df: pd.DataFrame,
    cell_idx: int,
    room_id: int,
    first_da_cycle: int,
) -> bool:
    """True when every cycle before *first_da_cycle* is DD (or there are none)."""
    sub = state_df[
        (state_df["cell_idx"] == cell_idx) & (state_df["room_id"] == room_id)
    ]
    before = sub[sub["cycle_id"] < first_da_cycle]
    return before.empty or (before["state"] == "DD").all()


def _is_perfect_cell_room(
    state_df: pd.DataFrame,
    cell_idx: int,
    room_id: int,
    first_da_cycle: int,
) -> bool:
    """
    Perfect ``(cell, room)``: DD at every cycle before first DA, DA at first DA,
    AA at every cycle after.
    """
    sub = state_df[
        (state_df["cell_idx"] == cell_idx) & (state_df["room_id"] == room_id)
    ]
    at_da = sub[sub["cycle_id"] == first_da_cycle]
    if len(at_da) != 1 or str(at_da.iloc[0]["state"]) != "DA":
        return False
    return _all_dd_before_first_da(
        state_df, cell_idx, room_id, first_da_cycle
    ) and _all_aa_after_first_da(state_df, cell_idx, room_id, first_da_cycle)


def perfect_cell_rooms_df(state_df: pd.DataFrame) -> pd.DataFrame:
    """
    ``(cell, room)`` pairs with a perfect DD…→DA→AA… trajectory.

    Returns ``cell_idx, room_id, first_da_cycle``.
    """
    pairs = ever_da_pairs(state_df)
    if pairs.empty:
        return pd.DataFrame(columns=["cell_idx", "room_id", "first_da_cycle"])

    rows: list[dict[str, int]] = []
    for _, row in pairs.iterrows():
        cell_idx = int(row["cell_idx"])
        room_id = int(row["room_id"])
        first_da = int(row["first_da_cycle"])
        if _is_perfect_cell_room(state_df, cell_idx, room_id, first_da):
            rows.append(
                {
                    "cell_idx": cell_idx,
                    "room_id": room_id,
                    "first_da_cycle": first_da,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["cell_idx", "room_id", "first_da_cycle"])

    return pd.DataFrame(rows).astype(
        {"cell_idx": int, "room_id": int, "first_da_cycle": int}
    )


def perfect_da_activation_count_by_cycle(state_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count perfect ``(cell, room)`` pairs by first-DA activation cycle.

    Returns ``first_da_cycle, count``.
    """
    perfect = perfect_cell_rooms_df(state_df)
    if perfect.empty:
        return pd.DataFrame(columns=["first_da_cycle", "count"])

    all_cycles = sorted(int(c) for c in state_df["cycle_id"].unique())
    counts = (
        perfect.groupby("first_da_cycle", sort=True)
        .size()
        .astype(int)
        .reset_index(name="count")
    )
    return (
        counts.set_index("first_da_cycle")
        .reindex(all_cycles, fill_value=0)
        .reset_index()
        .astype({"first_da_cycle": int, "count": int})
    )


def perfect_da_activation_count_multi_threshold(
    df_rf_pre: pd.DataFrame,
    df_rf_post: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_DEAD_THRESHOLDS,
    r2_min_selection: float | None = DEFAULT_R2_MIN_SELECTION,
    r2_selection_policy: R2SelectionPolicy = DEFAULT_R2_SELECTION_POLICY,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Perfect-pair counts by first-DA cycle for several thresholds.

    Returns ``dead_threshold, first_da_cycle, count``.
    """
    frames: list[pd.DataFrame] = []
    iterator = tqdm(
        thresholds,
        desc="perfect DA activations / threshold",
        disable=not show_progress,
    )
    for threshold in iterator:
        state_df = pre_post_state_df(
            df_rf_pre,
            df_rf_post,
            dead_threshold=threshold,
            r2_min_selection=r2_min_selection,
            r2_selection_policy=r2_selection_policy,
        )
        counts = perfect_da_activation_count_by_cycle(state_df)
        counts["dead_threshold"] = float(threshold)
        frames.append(counts)
    return pd.concat(frames, ignore_index=True)


def select_perfect_pairs(
    state_df: pd.DataFrame,
    *,
    n: int = 4,
) -> pd.DataFrame:
    """First *n* perfect ``(cell, room)`` pairs by ``cell_idx``, ``room_id``."""
    perfect = perfect_cell_rooms_df(state_df)
    if perfect.empty:
        return perfect
    return (
        perfect.sort_values(["cell_idx", "room_id"])
        .head(n)
        .reset_index(drop=True)
    )


def _all_aa_after_first_da(
    state_df: pd.DataFrame,
    cell_idx: int,
    room_id: int,
    first_da_cycle: int,
) -> bool:
    """True when every cycle after *first_da_cycle* is AA (or there are none)."""
    sub = state_df[
        (state_df["cell_idx"] == cell_idx) & (state_df["room_id"] == room_id)
    ]
    after = sub[sub["cycle_id"] > first_da_cycle]
    return after.empty or (after["state"] == "AA").all()


def select_timeline_pairs(
    state_df: pd.DataFrame,
    *,
    n_violators: int = 3,
) -> pd.DataFrame:
    """
    Pick ``(cell, room)`` pairs for state-timeline plots.

    Returns the first *n_violators* ever-DA pairs (by ``cell_idx``, ``room_id``)
    with a non-AA state after first DA, plus one pair whose post-first-DA
    trajectory is AA-only (when available).

    Columns: ``cell_idx, room_id, first_da_cycle``.
    """
    pairs = ever_da_pairs(state_df)
    if pairs.empty:
        return pd.DataFrame(columns=["cell_idx", "room_id", "first_da_cycle"])

    rows: list[dict[str, int | bool]] = []
    for _, row in pairs.iterrows():
        cell_idx = int(row["cell_idx"])
        room_id = int(row["room_id"])
        first_da = int(row["first_da_cycle"])
        sub = state_df[
            (state_df["cell_idx"] == cell_idx) & (state_df["room_id"] == room_id)
        ]
        n_after = int((sub["cycle_id"] > first_da).sum())
        rows.append(
            {
                "cell_idx": cell_idx,
                "room_id": room_id,
                "first_da_cycle": first_da,
                "all_aa_after_first_da": _all_aa_after_first_da(
                    state_df, cell_idx, room_id, first_da
                ),
                "n_cycles_after_da": n_after,
            }
        )

    info = pd.DataFrame(rows).sort_values(["cell_idx", "room_id"])
    violators = info[~info["all_aa_after_first_da"]].head(n_violators)
    aa_only = info[info["all_aa_after_first_da"] & (info["n_cycles_after_da"] > 0)].head(
        1
    )
    if aa_only.empty:
        aa_only = info[info["all_aa_after_first_da"]].head(1)

    picked = (
        pd.concat([violators, aa_only], ignore_index=True)
        .drop_duplicates(subset=["cell_idx", "room_id"])
        .sort_values(["cell_idx", "room_id"])
    )
    return picked[["cell_idx", "room_id", "first_da_cycle"]].astype(
        {"cell_idx": int, "room_id": int, "first_da_cycle": int}
    )
