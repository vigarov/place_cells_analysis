"""Plotting helpers for multi-optimizer two_rooms comparison."""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, NullFormatter, StrMethodFormatter

from single_room_multi_opt_compute import (
    OPTIMIZER_COLORS,
    OPTIMIZER_LABELS,
    OPTIMIZER_ORDER,
    TIME_UNIT_SECONDS,
    pf_metric_bar_panel_specs,
    pf_metric_boxplot_panels,
)
from single_room_multi_opt_plots import (
    GAUSS_TRANSITION_COLORS,
    TRANSITION_COLORS,
    _bin_transition_series,
    _handle_optimizer_significance,
    _mean_sem_ci95,
    _optimizer_legend_handles,
    _plot_drift_bars,
    _plot_peak_amp_tma_bars,
    _plot_spontaneous_pf_bars,
    _windowed_mean_sem,
    save_plot,
)
from single_room_pf_analysis import (
    TIME_AXIS_LABEL,
    bin_proportion_histogram,
    index_to_time_s,
    traj_boundary_vlines,
)
from two_rooms_analysis_plots import ROOM_TITLES
from two_rooms_multi_opt_compute import (
    OPTIMIZER_COLORS_DARK,
    OPTIMIZER_COLORS_LIGHT,
    REP_SWITCH_COLOR,
    REP_SWITCH_LW,
    ROOM_SWITCH_COLOR,
    ROOM_SWITCH_LW,
    TRAIN_OTHER_COLOR,
    TwoRoomsMultiOptResults,
    aggregate_train_same_metrics,
    to_cohort_multi_opt_view,
    to_multi_opt_view,
)
from two_rooms_pf_analysis import (
    ROOM_IDXS,
    TwoRoomsData,
    TwoRoomsPFResults,
    _capture_context_lookup,
    _last_train_same_capture,
    load_gaussian_capture,
    rep_switch_vlines,
    room_switch_vlines,
)

__all__ = [
    "save_plot",
    "plot_training_loss_by_optimizer_two_rooms",
    "plot_dual_room_metric_by_optimizer",
    "plot_train_same_wsm_fr_product_by_optimizer",
    "plot_state_transitions_grid_by_optimizer_two_rooms",
    "plot_pf_metric_boxplots_by_optimizer_two_rooms",
    "plot_pf_metric_bars_by_optimizer_two_rooms",
    "plot_formation_probability_by_optimizer",
    "plot_cell_displacement_by_optimizer_two_rooms",
    "plot_sustained_total_ratio_by_optimizer",
    "plot_pf_onset_cdf_by_optimizer",
    "plot_consistent_pf_cohorts_by_optimizer",
    "plot_pf_signal_group_by_optimizer_cohort",
    "plot_lifecycle_outlier_summary_by_optimizer_cohort",
]

PC_BIRTH_COHORT_LABELS = {
    0: "Born day 0",
    1: "Born day 1",
    2: "Born day 2",
    3: "Born day 3",
}


def _two_rooms_boundary_tick_positions_s(
    timeline: pd.DataFrame,
    *,
    x_col: str,
    segment_duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Major ticks at room/rep switches; minor ticks at midpoints between them."""
    room_x = index_to_time_s(
        room_switch_vlines(timeline, x_col=x_col),
        segment_duration_s,
    )
    rep_x = index_to_time_s(
        rep_switch_vlines(timeline, x_col=x_col),
        segment_duration_s,
    )
    major = np.sort(np.unique(np.concatenate([room_x, rep_x])))
    if major.size < 2:
        return major, np.array([], dtype=float)
    minor = 0.5 * (major[:-1] + major[1:])
    return major, minor


def _apply_two_rooms_boundary_ticks(
    ax,
    timeline: pd.DataFrame,
    *,
    x_col: str,
    segment_duration_s: float,
) -> None:
    major, minor = _two_rooms_boundary_tick_positions_s(
        timeline,
        x_col=x_col,
        segment_duration_s=segment_duration_s,
    )
    if major.size == 0:
        return
    ax.xaxis.set_major_locator(FixedLocator(major))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    if minor.size:
        ax.xaxis.set_minor_locator(FixedLocator(minor))
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.tick_params(axis="x", which="major", length=7, direction="out")
    ax.tick_params(
        axis="x",
        which="minor",
        length=5,
        width=0.9,
        direction="out",
    )


def add_two_rooms_timeseries_boundaries(
    ax,
    timeline: pd.DataFrame,
    *,
    x_col: str = "global_capture_idx",
    segment_duration_s: float,
    show_traj_boundaries: bool = True,
    set_boundary_ticks: bool = True,
) -> None:
    if show_traj_boundaries:
        for xv in index_to_time_s(
            traj_boundary_vlines(timeline, x_col=x_col),
            segment_duration_s,
        ):
            ax.axvline(xv, color="0.5", linestyle="--", linewidth=1, alpha=0.8, zorder=1)
    for xv in index_to_time_s(
        room_switch_vlines(timeline, x_col=x_col),
        segment_duration_s,
    ):
        ax.axvline(
            xv,
            color=ROOM_SWITCH_COLOR,
            linestyle="--",
            linewidth=ROOM_SWITCH_LW,
            alpha=1.0,
            zorder=2,
        )
    for xv in index_to_time_s(
        rep_switch_vlines(timeline, x_col=x_col),
        segment_duration_s,
    ):
        ax.axvline(
            xv,
            color=REP_SWITCH_COLOR,
            linestyle="--",
            linewidth=REP_SWITCH_LW,
            alpha=1.0,
            zorder=3,
        )
    if set_boundary_ticks:
        _apply_two_rooms_boundary_ticks(
            ax,
            timeline,
            x_col=x_col,
            segment_duration_s=segment_duration_s,
        )
        ax._two_rooms_boundary_tick_spec = (timeline, x_col, segment_duration_s)


def _refresh_two_rooms_boundary_ticks(ax) -> None:
    """Re-apply boundary ticks after layout/restyle so minors are not clipped."""
    spec = getattr(ax, "_two_rooms_boundary_tick_spec", None)
    if spec is None:
        return
    timeline, x_col, segment_duration_s = spec
    _apply_two_rooms_boundary_ticks(
        ax,
        timeline,
        x_col=x_col,
        segment_duration_s=segment_duration_s,
    )


def _plot_trace_with_train_other(
    ax,
    timeline: pd.DataFrame,
    y_col: str,
    *,
    color: str,
    eval_room_id: int,
    segment_duration_s: float,
    x_col: str = "global_capture_idx",
    label: str | None = None,
    with_fill: bool = False,
    sem_col: str | None = None,
    alpha: float = 0.85,
) -> None:
    x = index_to_time_s(timeline[x_col], segment_duration_s)
    y = timeline[y_col].to_numpy(dtype=float)
    if "visit_room_id" not in timeline.columns:
        same_mask = np.ones(len(timeline), dtype=bool)
    else:
        other = timeline["visit_room_id"].astype(int).to_numpy() != int(eval_room_id)
        same_mask = ~other
    if with_fill and sem_col is not None and sem_col in timeline.columns:
        sem = timeline[sem_col].to_numpy(dtype=float)
        ci = 1.96 * sem
        ax.fill_between(
            x[same_mask],
            (y - ci)[same_mask],
            (y + ci)[same_mask],
            color=color,
            alpha=0.2,
            zorder=2,
        )
        if (~same_mask).any():
            ax.fill_between(
                x[~same_mask],
                (y - ci)[~same_mask],
                (y + ci)[~same_mask],
                color=TRAIN_OTHER_COLOR,
                alpha=0.12,
                zorder=2,
            )
    if same_mask.any():
        ax.plot(
            x[same_mask],
            y[same_mask],
            "o-",
            color=color,
            markersize=2,
            linewidth=1.2,
            label=label,
            alpha=alpha,
            zorder=4,
        )
    if (~same_mask).any():
        ax.plot(
            x[~same_mask],
            y[~same_mask],
            "o-",
            color=TRAIN_OTHER_COLOR,
            markersize=2,
            linewidth=1.0,
            alpha=0.65,
            zorder=3,
        )
    if x.size:
        ax.set_xlim([0, x.max()])


def _room_color_for_optimizer(optimizer: str, room_idx: int) -> str:
    if room_idx == 0:
        return OPTIMIZER_COLORS_LIGHT[optimizer]
    return OPTIMIZER_COLORS_DARK[optimizer]


def _boundary_switch_count(timeline: pd.DataFrame) -> int:
    x_col = (
        "global_segment_idx"
        if "global_segment_idx" in timeline.columns
        else "global_capture_idx"
    )
    return len(room_switch_vlines(timeline, x_col=x_col)) + len(
        rep_switch_vlines(timeline, x_col=x_col)
    )


def _two_rooms_boundary_timeline(
    results: TwoRoomsMultiOptResults,
) -> pd.DataFrame | None:
    """Return a timeline with full room/rep switches for boundary markers."""
    best: pd.DataFrame | None = None
    best_score = -1
    for opt in OPTIMIZER_ORDER:
        training = results.training.get(opt)
        if training is not None and not training.loss_ts.empty:
            ts = training.loss_ts
            score = _boundary_switch_count(ts)
            if score > best_score:
                best_score = score
                best = ts
        pf_results = results.pf.get(opt)
        if pf_results is None:
            continue
        for room_idx in ROOM_IDXS:
            room = pf_results.by_room[room_idx]
            for ts in (room.mean_r2_ts, room.alive_ts):
                if ts.empty or "visit_room_id" not in ts.columns:
                    continue
                score = _boundary_switch_count(ts)
                if score > best_score:
                    best_score = score
                    best = ts
    return best


def plot_training_loss_by_optimizer_two_rooms(
    results: TwoRoomsMultiOptResults,
    *,
    segment_duration_s: float | None = None,
    trace_alpha: float = 0.75,
    subsample_win_size: int = 5,
    show_legend: bool = True,
    show_room_rep_boundaries: bool = False,
    title: str | None = None,
) -> tuple[Figure, plt.Axes]:
    if segment_duration_s is None:
        segment_duration_s = next(iter(results.contexts.values())).segment_duration_s

    fig, ax = plt.subplots(figsize=(14, 4))
    for opt in OPTIMIZER_ORDER:
        if opt not in results.training:
            continue
        loss_ts = results.training[opt].loss_ts
        x = index_to_time_s(
            loss_ts["global_segment_idx"].to_numpy(dtype=float),
            segment_duration_s,
        )
        y = loss_ts["loss"].to_numpy(dtype=float)
        color = OPTIMIZER_COLORS[opt]
        if subsample_win_size > 1:
            x_w, y_w, sem_w = _windowed_mean_sem(x, y, win_size=subsample_win_size)
            ci = 1.96 * sem_w
            ax.fill_between(
                x_w,
                y_w - ci,
                y_w + ci,
                color=color,
                alpha=trace_alpha * 0.35,
                linewidth=0,
            )
            ax.plot(x_w, y_w, "-", color=color, linewidth=1.2, alpha=trace_alpha)
        else:
            ax.plot(x, y, "-", color=color, linewidth=1.0, alpha=trace_alpha)
    ref = results.training[OPTIMIZER_ORDER[0]].loss_ts
    if show_room_rep_boundaries:
        add_two_rooms_timeseries_boundaries(
            ax,
            ref,
            x_col="global_segment_idx",
            segment_duration_s=segment_duration_s,
            show_traj_boundaries=False,
            set_boundary_ticks=False,
        )
    else:
        for xv in index_to_time_s(
            traj_boundary_vlines(ref, x_col="global_segment_idx"),
            segment_duration_s,
        ):
            ax.axvline(xv, color="0.5", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xlim(0, None)
    ax.set_xlabel(TIME_AXIS_LABEL)
    ax.set_ylabel("Loss")
    win_label = (
        f"; {subsample_win_size * segment_duration_s:g}s window mean ± 1.96 SEM"
        if subsample_win_size > 1
        else ""
    )
    if title is None:
        title = f"Training loss by optimizer ({win_label.lstrip('; ')})"
    ax.set_title(title)
    if show_legend:
        ax.legend(
            handles=_optimizer_legend_handles(linestyle="-", linewidth=1.2),
            frameon=False,
            loc="upper right",
            ncol=2,
        )
    fig.tight_layout()
    if show_room_rep_boundaries:
        ax._two_rooms_boundary_tick_spec = (ref, "global_segment_idx", segment_duration_s)
        _refresh_two_rooms_boundary_ticks(ax)
    return fig, ax


def plot_dual_room_metric_by_optimizer(
    results: TwoRoomsMultiOptResults,
    *,
    y_col: str,
    ylabel: str,
    title: str,
    segment_duration_s: float | None = None,
    with_fill: bool = False,
    sem_col: str | None = None,
    ylim: tuple[float, float] | None = None,
    trace_alpha: float = 0.85,
    show_traj_boundaries: bool = True,
    show_legend: bool = True,
) -> tuple[Figure, plt.Axes]:
    if segment_duration_s is None:
        segment_duration_s = next(iter(results.contexts.values())).segment_duration_s

    fig, ax = plt.subplots(figsize=(14, 4))
    ref_timeline = None
    for opt in OPTIMIZER_ORDER:
        if opt not in results.pf:
            continue
        pf_results = results.pf[opt]
        for room_idx in ROOM_IDXS:
            if y_col == "mean_amplitude":
                ts = pf_results.by_room[room_idx].mean_pf_amplitude_ts.copy()
            elif y_col == "mean_r2":
                ts = pf_results.by_room[room_idx].mean_r2_ts.copy()
            elif y_col == "proportion_alive":
                ts = pf_results.by_room[room_idx].alive_ts.copy()
            else:
                raise ValueError(f"Unsupported y_col {y_col!r}")
            if ref_timeline is None:
                ref_timeline = ts
            _plot_trace_with_train_other(
                ax,
                ts,
                y_col,
                color=_room_color_for_optimizer(opt, room_idx),
                eval_room_id=room_idx,
                segment_duration_s=segment_duration_s,
                label=f"{OPTIMIZER_LABELS[opt]} — {ROOM_TITLES[room_idx]}",
                with_fill=with_fill,
                sem_col=sem_col,
                alpha=trace_alpha,
            )
    boundary_timeline = _two_rooms_boundary_timeline(results)
    boundary_x_col = (
        "global_segment_idx"
        if boundary_timeline is not None
        and "global_segment_idx" in boundary_timeline.columns
        else "global_capture_idx"
    )
    if boundary_timeline is not None:
        add_two_rooms_timeseries_boundaries(
            ax,
            boundary_timeline,
            x_col=boundary_x_col,
            segment_duration_s=segment_duration_s,
            show_traj_boundaries=show_traj_boundaries,
            set_boundary_ticks=False,
        )
    ax.set_xlabel(TIME_AXIS_LABEL)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if show_legend:
        ax.legend(frameon=False, fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    if boundary_timeline is not None:
        ax._two_rooms_boundary_tick_spec = (
            boundary_timeline,
            boundary_x_col,
            segment_duration_s,
        )
        _refresh_two_rooms_boundary_ticks(ax)
    return fig, ax


def _rep_train_same_caps(meta: pd.DataFrame, n_repetitions: int) -> dict[int, frozenset[int]]:
    out: dict[int, frozenset[int]] = {}
    for rep_id in range(n_repetitions):
        mask = (
            (meta["rep_id"].astype(int) == rep_id)
            & (meta["train_context"] == "train_same")
            & (meta["segment_id"].astype(int) >= 0)
        )
        out[rep_id] = (
            frozenset(meta.loc[mask, "capture_idx"].astype(int).tolist())
            if mask.any()
            else frozenset()
        )
    return out


def _active_pc_cell_indices(pf_segment_df: pd.DataFrame, capture_idx: int) -> np.ndarray:
    cells = pf_segment_df.loc[
        pf_segment_df["is_segment"]
        & (pf_segment_df["state"] == "active")
        & (pf_segment_df["capture_idx"] == capture_idx),
        "cell_idx",
    ]
    if cells.empty:
        return np.array([], dtype=int)
    return np.unique(cells.astype(int))


def _mean_ratemap_at_capture(
    data: TwoRoomsData,
    *,
    eval_room_id: int,
    capture_idx: int | None,
    pf_segment_df: pd.DataFrame | None = None,
    active_only: bool = False,
) -> np.ndarray | None:
    if capture_idx is None:
        return None
    rm = load_gaussian_capture(
        capture_idx,
        gaussian_params=data.gaussian_params_by_room[eval_room_id],
        field_shape=data.field_shape,
    )
    if active_only:
        if pf_segment_df is None:
            raise ValueError("pf_segment_df is required when active_only=True")
        cell_indices = _active_pc_cell_indices(pf_segment_df, capture_idx)
        if cell_indices.size == 0:
            return None
        rm = rm[cell_indices]
    return rm.mean(axis=0)


def _mean_ratemap_at_last_train_same(
    data: TwoRoomsData,
    *,
    eval_room_id: int,
    rep_id: int,
    pf_segment_df: pd.DataFrame | None = None,
    active_only: bool = False,
) -> np.ndarray | None:
    meta = _capture_context_lookup(data.master_df_by_room[eval_room_id], eval_room_id)
    rep_caps = _rep_train_same_caps(meta, data.n_repetitions)
    capture_idx = _last_train_same_capture(meta, rep_caps, rep_id)
    return _mean_ratemap_at_capture(
        data,
        eval_room_id=eval_room_id,
        capture_idx=capture_idx,
        pf_segment_df=pf_segment_df,
        active_only=active_only,
    )


def _warmup_end_capture_idx(meta: pd.DataFrame) -> int | None:
    sub = meta.loc[
        (meta["rep_id"].astype(int) == 0) & (meta["segment_id"].astype(int) == -1)
    ]
    if sub.empty:
        return None
    return int(sub.iloc[0]["capture_idx"])


def build_train_same_panel_data(
    results: TwoRoomsMultiOptResults,
    optimizer: str,
    *,
    active_only: bool,
) -> dict[str, object]:
    data = results.contexts[optimizer]
    pf_results = results.pf[optimizer]
    mean_wsm_by_room = results.mean_wsm_by_room
    if not mean_wsm_by_room:
        raise ValueError("mean_wsm_by_room missing; pass to compute_all_two_rooms_optimizer_results")

    prod_last_train_same: dict[int, list[np.ndarray | None]] = {r: [] for r in ROOM_IDXS}
    row_labels = ["Warmup end", *[f"Rep {rep_id}" for rep_id in range(data.n_repetitions)]]

    for room_idx in ROOM_IDXS:
        pf_segment_df = (
            pf_results.by_room[room_idx].pf_segment_df if active_only else None
        )
        meta = _capture_context_lookup(data.master_df_by_room[room_idx], room_idx)
        warmup_capture_idx = _warmup_end_capture_idx(meta)
        warmup_mean_rm = _mean_ratemap_at_capture(
            data,
            eval_room_id=room_idx,
            capture_idx=warmup_capture_idx,
            pf_segment_df=pf_segment_df,
            active_only=active_only,
        )
        mean_wsm = mean_wsm_by_room[room_idx]
        if warmup_mean_rm is None:
            prod_last_train_same[room_idx].append(None)
        else:
            prod_last_train_same[room_idx].append(mean_wsm * warmup_mean_rm)

        for rep_id in range(data.n_repetitions):
            mean_rm = _mean_ratemap_at_last_train_same(
                data,
                eval_room_id=room_idx,
                rep_id=rep_id,
                pf_segment_df=pf_segment_df,
                active_only=active_only,
            )
            if mean_rm is None:
                prod_last_train_same[room_idx].append(None)
            else:
                prod_last_train_same[room_idx].append(mean_wsm * mean_rm)

    n_rows = len(row_labels)
    if any(len(prod_last_train_same[r]) != n_rows for r in ROOM_IDXS):
        raise ValueError(
            f"train_same panel row mismatch for {optimizer!r}: "
            f"expected {n_rows} rows, got "
            f"{ {r: len(prod_last_train_same[r]) for r in ROOM_IDXS} }"
        )
    prod_vmax: list[float] = []
    shared_vmax = results.shared_ratemap_vmax
    for row_idx in range(n_rows):
        prod_peaks = [
            float(np.nanmax(prod_last_train_same[room_idx][row_idx]))
            for room_idx in ROOM_IDXS
            if prod_last_train_same[room_idx][row_idx] is not None
        ]
        prod_vmax.append(max(prod_peaks) if prod_peaks else shared_vmax)

    return {
        "prod": prod_last_train_same,
        "prod_vmax": prod_vmax,
        "row_labels": row_labels,
    }


def plot_train_same_wsm_fr_product_by_optimizer(
    results: TwoRoomsMultiOptResults,
    *,
    active_only: bool = True,
) -> Figure:
    opt_cols = [o for o in OPTIMIZER_ORDER if o in results.pf]
    n_opts = len(opt_cols)
    n_rows = 1 + next(iter(results.contexts.values())).n_repetitions
    fig, axes = plt.subplots(
        n_rows,
        n_opts * len(ROOM_IDXS),
        figsize=(2.3 * n_opts * len(ROOM_IDXS), 2.4 * n_rows),
        squeeze=False,
    )
    for opt_i, opt in enumerate(opt_cols):
        panel = build_train_same_panel_data(results, opt, active_only=active_only)
        prod = panel["prod"]
        prod_vmax = panel["prod_vmax"]
        row_labels = panel["row_labels"]
        for room_i, room_idx in enumerate(ROOM_IDXS):
            col = opt_i * len(ROOM_IDXS) + room_i
            for row_idx in range(n_rows):
                ax = axes[row_idx, col]
                field = prod[room_idx][row_idx]
                if field is None:
                    ax.text(
                        0.5,
                        0.5,
                        "no active PCs" if active_only else "no train_same",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=8,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.imshow(field, cmap="jet", vmin=0, vmax=prod_vmax[row_idx])
                    ax.set_xticks([])
                    ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title(
                        f"{OPTIMIZER_LABELS[opt]}\n{ROOM_TITLES[room_idx]}\nmean WSM × mean FR",
                        fontsize=8,
                    )
                if col == 0:
                    ax.set_ylabel(row_labels[row_idx], fontsize=8)
    fig.suptitle(
        "Mean WSM × mean FR at end of train_same blocks (active PCs only)"
        if active_only
        else "Mean WSM × mean FR at end of train_same blocks",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def _plot_pf_transitions_on_ax(
    ax,
    transition_counts: pd.DataFrame,
    *,
    title: str,
    show_revives: bool,
    segment_duration_s: float,
    bin_window: int = 10,
) -> None:
    x_raw = index_to_time_s(
        transition_counts["global_transition_idx"],
        segment_duration_s,
    )
    series = [
        ("n_dead_to_alive", TRANSITION_COLORS["inactive_to_active"], "Inactive → active"),
        ("n_alive_to_dead", TRANSITION_COLORS["active_to_inactive"], "Active → inactive"),
    ]
    if show_revives:
        series.append(("n_revives", TRANSITION_COLORS["revives"], "Revives"))
    for col, color, label in series:
        x, y, sem = _bin_transition_series(
            x_raw,
            transition_counts[col].to_numpy(),
            bin_window=bin_window,
        )
        if sem is None:
            ax.plot(x, y, "o--", color=color, markersize=3, linewidth=1.2, label=label)
        else:
            ci = 1.96 * sem
            ax.fill_between(x, y - ci, y + ci, alpha=0.2, color=color)
            ax.plot(x, y, "-", color=color, linewidth=1.5, label=label)
    for xv in index_to_time_s(
        traj_boundary_vlines(transition_counts, x_col="global_transition_idx"),
        segment_duration_s,
    ):
        ax.axvline(xv, color="0.5", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xlabel(TIME_AXIS_LABEL)
    ax.set_ylabel("PF count")
    ax.set_title(title, fontsize=9)


def _plot_gauss_transitions_on_ax(
    ax,
    gauss_props: pd.DataFrame,
    *,
    title: str,
    segment_duration_s: float,
) -> None:
    x = index_to_time_s(gauss_props["global_transition_idx"], segment_duration_s)
    series = [
        ("prop_1_to_2", GAUSS_TRANSITION_COLORS["prop_1_to_2"], "1→2"),
        ("prop_2_to_1", GAUSS_TRANSITION_COLORS["prop_2_to_1"], "2→1"),
        ("prop_1_to_1", GAUSS_TRANSITION_COLORS["prop_1_to_1"], "1→1"),
        ("prop_2_to_2", GAUSS_TRANSITION_COLORS["prop_2_to_2"], "2→2"),
    ]
    for col, color, label in series:
        if col not in gauss_props.columns:
            continue
        ax.plot(
            x,
            gauss_props[col].to_numpy(dtype=float),
            "-",
            color=color,
            linewidth=1.2,
            label=label,
            alpha=0.85,
        )
    for xv in index_to_time_s(
        traj_boundary_vlines(gauss_props, x_col="global_transition_idx"),
        segment_duration_s,
    ):
        ax.axvline(xv, color="0.5", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xlabel(TIME_AXIS_LABEL)
    ax.set_ylabel("Cell proportion")
    ax.set_title(title, fontsize=9)


def plot_state_transitions_grid_by_optimizer_two_rooms(
    results: TwoRoomsMultiOptResults,
    *,
    pf_tracking_method: str,
    segment_duration_s: float | None = None,
    bin_window: int = 10,
) -> tuple[Figure, np.ndarray]:
    if segment_duration_s is None:
        segment_duration_s = next(iter(results.contexts.values())).segment_duration_s

    opt_cols = [o for o in OPTIMIZER_ORDER if o in results.pf]
    n_opts = len(opt_cols)
    fig = plt.figure(figsize=(5.5 * n_opts, 10))
    gs = fig.add_gridspec(4, n_opts, hspace=0.35, wspace=0.25)
    axes = np.empty((4, n_opts), dtype=object)
    for col in range(n_opts):
        for row in range(4):
            if row in (0, 2):
                if col == 0:
                    axes[row, col] = fig.add_subplot(gs[row, col])
                else:
                    axes[row, col] = fig.add_subplot(gs[row, col], sharey=axes[row, 0])
            else:
                axes[row, col] = fig.add_subplot(gs[row, col], sharey=axes[row - 1, col])

    show_revives = pf_tracking_method != "no_revive"
    panel_specs = [
        (0, 0, "pf"),
        (1, 1, "pf"),
        (2, 0, "gauss"),
        (3, 1, "gauss"),
    ]

    for col, opt in enumerate(opt_cols):
        pf_results = results.pf[opt]
        for row, room_idx, kind in panel_specs:
            room = pf_results.by_room[room_idx]
            title = f"{OPTIMIZER_LABELS[opt]}\n{ROOM_TITLES[room_idx]}"
            if kind == "pf":
                _plot_pf_transitions_on_ax(
                    axes[row, col],
                    room.pf_transition_counts,
                    title=title,
                    show_revives=show_revives,
                    segment_duration_s=segment_duration_s,
                    bin_window=bin_window,
                )
            else:
                _plot_gauss_transitions_on_ax(
                    axes[row, col],
                    room.gauss_transition_props,
                    title=title,
                    segment_duration_s=segment_duration_s,
                )

    axes[0, 0].set_ylabel("PF count")
    axes[2, 0].set_ylabel("Cell proportion")
    fig.suptitle(
        "PF state transitions (rows 0–1) and Gaussian-state transitions (rows 2–3) by optimizer",
        y=1.01,
    )
    fig.tight_layout()
    return fig, axes


def _metric_values_two_rooms_by_optimizer(
    results: TwoRoomsMultiOptResults,
    metric_col: str,
    *,
    split: Literal["by_room", "aggregated"],
) -> dict[str, dict[int, np.ndarray]] | dict[str, np.ndarray]:
    if split == "aggregated":
        out: dict[str, np.ndarray] = {}
        for opt in OPTIMIZER_ORDER:
            if opt not in results.pf:
                continue
            agg = aggregate_train_same_metrics(results.pf[opt], by_room=False)
            if isinstance(agg, dict) and metric_col in agg:
                out[opt] = np.asarray(agg[metric_col], dtype=float)
            else:
                out[opt] = np.array([], dtype=float)
        return out

    out_by_room: dict[str, dict[int, np.ndarray]] = {}
    for opt in OPTIMIZER_ORDER:
        if opt not in results.pf:
            continue
        agg = aggregate_train_same_metrics(results.pf[opt], by_room=True)
        out_by_room[opt] = {}
        if isinstance(agg, dict):
            for room_idx in ROOM_IDXS:
                room_metrics = agg.get(room_idx, {})
                out_by_room[opt][room_idx] = np.asarray(
                    room_metrics.get(metric_col, []),
                    dtype=float,
                )
    return out_by_room


def _boxplot_flierprops() -> dict[str, object]:
    return {
        "marker": "o",
        "markersize": 2,
        "alpha": 0.35,
        "markerfacecolor": "0.45",
        "markeredgecolor": "0.45",
        "linestyle": "none",
    }


def _optimizer_bar_tops(
    opts: list[str],
    values_by_opt: dict[str, np.ndarray] | dict[str, dict[int, np.ndarray]],
    *,
    plot_kind: Literal["boxplot", "bars"],
    by_room: bool,
) -> np.ndarray:
    tops: list[float] = []
    for opt in opts:
        opt_top = 0.0
        if by_room:
            room_values = values_by_opt[opt]
            room_items = room_values.items()
        else:
            room_items = [(0, values_by_opt.get(opt, np.array([])))]
        for _room_idx, vals in room_items:
            vals = np.asarray(vals, dtype=float)
            if plot_kind == "boxplot":
                if vals.size:
                    opt_top = max(opt_top, float(np.nanmax(vals)))
            else:
                mean, _sem, ci = _mean_sem_ci95(vals)
                opt_top = max(opt_top, mean + ci)
        tops.append(opt_top)
    return np.asarray(tops, dtype=float)


def _plot_two_room_metric_panel(
    ax,
    results: TwoRoomsMultiOptResults,
    metric_col: str,
    *,
    ylabel: str,
    plot_kind: Literal["boxplot", "bars"],
    split: Literal["by_room", "aggregated"],
    show_significance: bool,
    show_insignificance: bool,
    bracket_step_mult: float = 1.5,
) -> None:
    opts = [o for o in OPTIMIZER_ORDER if o in results.pf]
    n_opts = len(opts)
    positions = {o: float(i) for i, o in enumerate(opts)}
    x = np.arange(n_opts, dtype=float)

    if split == "aggregated":
        values_by_opt = _metric_values_two_rooms_by_optimizer(
            results,
            metric_col,
            split="aggregated",
        )
        if plot_kind == "boxplot":
            box_data = [values_by_opt.get(opt, np.array([])) for opt in opts]
            bp = ax.boxplot(
                box_data,
                positions=x,
                widths=0.55,
                patch_artist=True,
                flierprops=_boxplot_flierprops(),
            )
            for patch, opt in zip(bp["boxes"], opts, strict=True):
                patch.set_facecolor(OPTIMIZER_COLORS[opt])
                patch.set_alpha(0.75)
                patch.set_edgecolor("k")
            for median in bp["medians"]:
                median.set_color("k")
                median.set_linewidth(1.5)
        else:
            means, cis = [], []
            for opt in opts:
                m, _s, ci = _mean_sem_ci95(values_by_opt.get(opt, np.array([])))
                means.append(m)
                cis.append(ci)
            ax.bar(
                x,
                means,
                width=0.55,
                color=[OPTIMIZER_COLORS[o] for o in opts],
                alpha=0.75,
                edgecolor="k",
            )
            ax.errorbar(x, means, yerr=cis, fmt="none", ecolor="k", capsize=4, linewidth=1.2)
            bar_tops = np.asarray(means) + np.asarray(cis)
            data_top = float(np.nanmax(bar_tops)) if bar_tops.size else 0.0
            ax.set_ylim(0, data_top * 1.05 if np.isfinite(data_top) and data_top > 0 else 1)
        bar_tops = _optimizer_bar_tops(
            opts,
            values_by_opt,
            plot_kind=plot_kind,
            by_room=False,
        )
        sig_values = values_by_opt
    else:
        values_by_room = _metric_values_two_rooms_by_optimizer(
            results,
            metric_col,
            split="by_room",
        )
        sig_values = _metric_values_two_rooms_by_optimizer(
            results,
            metric_col,
            split="aggregated",
        )
        group_width = 0.75
        bar_width = group_width / len(ROOM_IDXS)
        offsets = np.linspace(
            -group_width / 2 + bar_width / 2,
            group_width / 2 - bar_width / 2,
            len(ROOM_IDXS),
        )

        if plot_kind == "boxplot":
            for room_idx in ROOM_IDXS:
                offset = offsets[room_idx]
                box_data = [values_by_room[opt][room_idx] for opt in opts]
                bar_positions = x + offset
                bp = ax.boxplot(
                    box_data,
                    positions=bar_positions,
                    widths=bar_width * 0.9,
                    patch_artist=True,
                    flierprops=_boxplot_flierprops(),
                )
                for patch, opt in zip(bp["boxes"], opts, strict=True):
                    patch.set_facecolor(OPTIMIZER_COLORS[opt])
                    patch.set_alpha(0.75)
                    patch.set_edgecolor("k")
                    if room_idx == 1:
                        patch.set_hatch("//")
                for median in bp["medians"]:
                    median.set_color("k")
                    median.set_linewidth(1.5)
        else:
            for room_idx in ROOM_IDXS:
                offset = offsets[room_idx]
                means, cis = [], []
                for opt in opts:
                    m, _s, ci = _mean_sem_ci95(values_by_room[opt][room_idx])
                    means.append(m)
                    cis.append(ci)
                bar_positions = x + offset
                ax.bar(
                    bar_positions,
                    means,
                    width=bar_width * 0.9,
                    color=[OPTIMIZER_COLORS[o] for o in opts],
                    alpha=0.75,
                    edgecolor="k",
                    hatch=None if room_idx == 0 else "//",
                    label=ROOM_TITLES[room_idx],
                )
                ax.errorbar(
                    bar_positions,
                    means,
                    yerr=cis,
                    fmt="none",
                    ecolor="k",
                    capsize=3,
                    linewidth=1.2,
                )
            ax.legend(frameon=False, fontsize=8)
            bar_tops_arr = []
            for opt in opts:
                opt_top = 0.0
                for room_idx in ROOM_IDXS:
                    m, _s, ci = _mean_sem_ci95(values_by_room[opt][room_idx])
                    opt_top = max(opt_top, m + ci)
                bar_tops_arr.append(opt_top)
            data_top = float(np.nanmax(bar_tops_arr)) if bar_tops_arr else 0.0
            ax.set_ylim(0, data_top * 1.05 if np.isfinite(data_top) and data_top > 0 else 1)
        bar_tops = _optimizer_bar_tops(
            opts,
            values_by_room,
            plot_kind=plot_kind,
            by_room=True,
        )

    _handle_optimizer_significance(
        ax,
        sig_values,
        positions=positions,
        bar_tops=bar_tops,
        panel_label=ylabel,
        show_significance=show_significance,
        show_insignificance=show_insignificance,
        bracket_step_mult=bracket_step_mult,
        test="ks" if plot_kind == "boxplot" else "welch",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([OPTIMIZER_LABELS[o] for o in opts], rotation=15, ha="right")
    ax.set_ylabel(ylabel)


def plot_pf_metric_boxplots_by_optimizer_two_rooms(
    results: TwoRoomsMultiOptResults,
    *,
    time_unit: str = TIME_UNIT_SECONDS,
    split: Literal["by_room", "aggregated"] = "by_room",
    show_significance: bool = True,
    show_insignificance: bool = False,
) -> Figure:
    panels = pf_metric_boxplot_panels(time_unit)
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 3.2 * len(panels)))
    if len(panels) == 1:
        axes = [axes]
    for ax, (col, ylabel) in zip(axes, panels, strict=True):
        _plot_two_room_metric_panel(
            ax,
            results,
            col,
            ylabel=ylabel,
            plot_kind="boxplot",
            split=split,
            show_significance=show_significance,
            show_insignificance=show_insignificance,
        )
        ax.set_title(ylabel, fontsize=10)
    title_suffix = "by room" if split == "by_room" else "rooms aggregated"
    fig.suptitle(
        f"Place-field metrics by optimizer — train_same only ({title_suffix})",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_pf_metric_bars_by_optimizer_two_rooms(
    results: TwoRoomsMultiOptResults,
    *,
    time_unit: str = TIME_UNIT_SECONDS,
    split: Literal["by_room", "aggregated"] = "by_room",
    show_significance: bool = True,
    show_insignificance: bool = False,
) -> Figure:
    panel_specs = pf_metric_bar_panel_specs(time_unit)
    fig, axes = plt.subplots(len(panel_specs), 1, figsize=(15, 3.5 * len(panel_specs)))
    if len(panel_specs) == 1:
        axes = [axes]
    multi_view = to_multi_opt_view(results, pool_rooms=(split == "aggregated"))
    for ax, (kind, metric_col, ylabel) in zip(axes, panel_specs, strict=True):
        if kind == "simple":
            _plot_two_room_metric_panel(
                ax,
                results,
                metric_col,
                ylabel=ylabel,
                plot_kind="bars",
                split=split,
                show_significance=show_significance,
                show_insignificance=show_insignificance,
                bracket_step_mult=1.65,
            )
            ax.set_title(ylabel, fontsize=10)
        elif kind == "amp_tma":
            _plot_peak_amp_tma_bars(
                ax,
                multi_view,
                time_unit=time_unit,
                show_significance=show_significance,
                show_insignificance=show_insignificance,
                bracket_step_mult=1.65,
            )
        elif kind == "spontaneous":
            _plot_spontaneous_pf_bars(
                ax,
                multi_view,
                show_significance=show_significance,
                show_insignificance=show_insignificance,
                bracket_step_mult=1.65,
            )
        elif kind == "drift":
            _plot_drift_bars(
                ax,
                multi_view,
                show_significance=show_significance,
                show_insignificance=show_insignificance,
            )
        else:
            raise ValueError(f"Unknown bar panel kind: {kind!r}")
    title_suffix = "by room" if split == "by_room" else "rooms aggregated"
    fig.suptitle(
        f"Place-field metrics by optimizer (train_same)",
        y=1.01,
    )
    fig.tight_layout()
    return fig


# --- Section 2: extended analysis ---


def _pool_curve_with_sem(dfs: list[pd.DataFrame], x_col: str, y_col: str) -> pd.DataFrame:
    pooled = pd.concat(dfs, ignore_index=True)
    return (
        pooled.groupby(x_col, sort=True)[y_col]
        .agg(["mean", "sem"])
        .reset_index()
        .rename(columns={"sem": "sem_raw"})
        .assign(sem=lambda d: d["sem_raw"].fillna(0.0))
    )


def plot_formation_probability_by_optimizer(
    results: TwoRoomsMultiOptResults,
) -> tuple[Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(9, 4))
    for opt in OPTIMIZER_ORDER:
        if opt not in results.pf:
            continue
        pf_results = results.pf[opt]
        dfs = [
            pf_results.formation_probability[
                pf_results.formation_probability["eval_room_id"] == room_idx
            ]
            for room_idx in ROOM_IDXS
        ]
        pooled = _pool_curve_with_sem(dfs, "n_prior_days_with_pf", "formation_probability")
        x = pooled["n_prior_days_with_pf"].to_numpy(dtype=float)
        y = pooled["mean"].to_numpy(dtype=float)
        sem = pooled["sem"].to_numpy(dtype=float)
        ci = 1.96 * sem
        color = OPTIMIZER_COLORS[opt]
        ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.2)
        ax.plot(x, y, "o-", color=color, linewidth=1.5, label=OPTIMIZER_LABELS[opt])
    ax.set_xlabel("Prior repetitions with active PC")
    ax.set_ylabel("Formation probability")
    ax.set_title("PF formation probability vs prior activity (pooled rooms; mean ± 1.96 SEM)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, ax


def _plot_displacement_proportion(
    ax,
    centers: np.ndarray,
    prop: np.ndarray,
    *,
    bin_width: float,
    color: str,
    xlabel: str,
    xlim: tuple[float, float],
    show_zero_line: bool = True,
    label: str | None = None,
) -> None:
    ax.bar(
        centers,
        prop,
        width=bin_width,
        color=color,
        alpha=0.25,
        edgecolor="k",
        linewidth=0.5,
        align="center",
        zorder=1,
        label=label,
    )
    ax.plot(centers, prop, color=color, linewidth=2, zorder=2)
    if show_zero_line:
        ax.axvline(0, color="k", linestyle="--", linewidth=1.5, zorder=0)
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Proportion per bin")
    ymax = float(np.max(prop)) if len(prop) else 0.0
    ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1)


def _plot_displacement_row(
    ax,
    results: TwoRoomsMultiOptResults,
    opts: list[str],
    *,
    room_idx: int | None,
    kind: str,
    col: str,
    xlabel: str,
    xlim: tuple[float, float],
    show_zero: bool,
    zoom_bin_width: float,
) -> None:
    for opt in opts:
        sub = results.pf[opt].cell_displacement
        if room_idx is not None:
            sub = sub[
                (sub["eval_room_id"] == room_idx) & (sub["kind"] == kind)
            ]
        else:
            sub = sub[sub["kind"] == kind]
        vals = sub[col].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        centers, prop, _ = bin_proportion_histogram(
            vals,
            bin_width=zoom_bin_width,
            bin_range=xlim,
        )
        _plot_displacement_proportion(
            ax,
            centers,
            prop,
            bin_width=zoom_bin_width,
            color=OPTIMIZER_COLORS[opt],
            xlabel=xlabel,
            xlim=xlim,
            show_zero_line=show_zero,
            label=OPTIMIZER_LABELS[opt],
        )


def plot_cell_displacement_by_optimizer_two_rooms(
    results: TwoRoomsMultiOptResults,
    *,
    layout: Literal["by_room", "aggregated"] = "by_room",
    zoom_bin_width: float = 0.25,
) -> Figure:
    disp_panels = [
        ("displacement_l2", "L2 (cm)", (0, 15), False),
        ("displacement_x", "Δx (cm)", (-8, 8), True),
        ("displacement_y", "Δy (cm)", (-8, 8), True),
    ]
    if layout == "by_room":
        row_specs: list[tuple[int | None, str, str]] = [
            (0, "within", "Room 0 within"),
            (0, "cross_rep", "Room 0 cross-rep"),
            (1, "within", "Room 1 within"),
            (1, "cross_rep", "Room 1 cross-rep"),
        ]
        fig, axes = plt.subplots(4, 3, figsize=(14, 14), layout="constrained")
    else:
        row_specs = [
            (None, "within", "Within repetition"),
            (None, "cross_rep", "Cross repetition"),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")

    opts = [o for o in OPTIMIZER_ORDER if o in results.pf]
    for row_idx, (room_idx, kind, row_title) in enumerate(row_specs):
        for metric_idx, (col, xlabel, xlim, show_zero) in enumerate(disp_panels):
            ax = axes[row_idx, metric_idx]
            _plot_displacement_row(
                ax,
                results,
                opts,
                room_idx=room_idx,
                kind=kind,
                col=col,
                xlabel=xlabel,
                xlim=xlim,
                show_zero=show_zero,
                zoom_bin_width=zoom_bin_width,
            )
            ax.set_title(f"{row_title} — {xlabel}")
            if metric_idx == 0:
                ax.legend(frameon=False, fontsize=7)

    fig.suptitle(
        "Cell displacement by optimizer (train_same PFs)"
        if layout == "by_room"
        else "Cell displacement by optimizer — rooms aggregated",
        y=1.01,
    )
    return fig


def _aggregate_stability_ratio(
    pf_results: TwoRoomsPFResults,
    *,
    entity: Literal["pc", "pf"],
) -> pd.DataFrame:
    if entity == "pc":
        src = pf_results.pc_stability_by_repetition
        total_col, sustained_col = "n_total_pcs", "n_sustained_pcs"
    else:
        src = pf_results.pf_stability_by_repetition
        total_col, sustained_col = "n_total_pfs", "n_sustained_pfs"
    pooled = (
        src.groupby("repetition", sort=True)
        .agg(n_total=(total_col, "sum"), n_sustained=(sustained_col, "sum"))
        .reset_index()
    )
    pooled["ratio"] = np.where(
        pooled["n_total"] > 0,
        pooled["n_sustained"] / pooled["n_total"],
        np.nan,
    )
    return pooled


def plot_sustained_total_ratio_by_optimizer(
    results: TwoRoomsMultiOptResults,
    *,
    entity: Literal["pc", "pf"],
    layout: Literal["by_room", "aggregated"] = "by_room",
) -> Figure:
    entity_label = "place cells" if entity == "pc" else "place fields"
    if layout == "by_room":
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True)
        for ax, room_idx in zip(axes, ROOM_IDXS):
            for opt in OPTIMIZER_ORDER:
                if opt not in results.pf:
                    continue
                sub = (
                    results.pf[opt].pc_stability_by_repetition
                    if entity == "pc"
                    else results.pf[opt].pf_stability_by_repetition
                )
                sub = sub[sub["eval_room_id"] == room_idx].sort_values("repetition")
                total = sub["n_total_pcs" if entity == "pc" else "n_total_pfs"].to_numpy(
                    dtype=float
                )
                sustained = sub[
                    "n_sustained_pcs" if entity == "pc" else "n_sustained_pfs"
                ].to_numpy(dtype=float)
                ratio = np.where(total > 0, sustained / total, np.nan)
                x = sub["repetition"].to_numpy(dtype=float)
                ax.plot(
                    x,
                    ratio,
                    "o-",
                    color=OPTIMIZER_COLORS[opt],
                    linewidth=1.5,
                    label=OPTIMIZER_LABELS[opt],
                )
            ax.set_xlabel("Repetition (day)")
            ax.set_ylabel("Sustained / total")
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{ROOM_TITLES[room_idx]} (train_same)")
            ax.legend(frameon=False, fontsize=7)
        fig.suptitle(f"Sustained / total {entity_label} by repetition", y=1.02)
    else:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        for opt in OPTIMIZER_ORDER:
            if opt not in results.pf:
                continue
            pooled = _aggregate_stability_ratio(results.pf[opt], entity=entity)
            x = pooled["repetition"].to_numpy(dtype=float)
            y = pooled["ratio"].to_numpy(dtype=float)
            sem = np.zeros_like(y)
            for i, rep in enumerate(x.astype(int)):
                ratios = []
                for room_idx in ROOM_IDXS:
                    sub = (
                        results.pf[opt].pc_stability_by_repetition
                        if entity == "pc"
                        else results.pf[opt].pf_stability_by_repetition
                    )
                    row = sub[
                        (sub["eval_room_id"] == room_idx)
                        & (sub["repetition"] == rep)
                    ]
                    if row.empty:
                        continue
                    total = float(
                        row["n_total_pcs" if entity == "pc" else "n_total_pfs"].iloc[0]
                    )
                    sustained = float(
                        row[
                            "n_sustained_pcs" if entity == "pc" else "n_sustained_pfs"
                        ].iloc[0]
                    )
                    if total > 0:
                        ratios.append(sustained / total)
                if len(ratios) > 1:
                    sem[i] = float(np.std(ratios, ddof=1) / np.sqrt(len(ratios)))
            ci = 1.96 * sem
            color = OPTIMIZER_COLORS[opt]
            ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.2)
            ax.plot(x, y, "o-", color=color, linewidth=1.5, label=OPTIMIZER_LABELS[opt])
        ax.set_xlabel("Repetition (day)")
        ax.set_ylabel("Sustained / total")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"Sustained / total {entity_label} — rooms aggregated (train_same)")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def _plot_onset_ecdf(
    ax,
    values: np.ndarray,
    *,
    label: str,
    color: str,
    linestyle: str = "-",
) -> None:
    values = np.sort(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    y = np.arange(1, values.size + 1, dtype=float) / values.size
    ax.step(values, y, where="post", color=color, linestyle=linestyle, linewidth=1.8, label=label)


def plot_pf_onset_cdf_by_optimizer(
    results: TwoRoomsMultiOptResults,
) -> Figure:
    cdf_panels = [
        ("birth_day", "Birth day (sustained = ever sustained later)"),
        ("subsequent_days", "Subsequent days"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (cohort, title) in zip(axes, cdf_panels):
        for opt in OPTIMIZER_ORDER:
            if opt not in results.pf:
                continue
            cohort_df = results.pf[opt].pf_onset_cdf_samples[
                results.pf[opt].pf_onset_cdf_samples["cohort"] == cohort
            ]
            for pf_label, linestyle, shade in [
                ("sustained", "-", OPTIMIZER_COLORS_DARK[opt]),
                ("transient", "--", OPTIMIZER_COLORS_LIGHT[opt]),
            ]:
                vals = cohort_df.loc[cohort_df["label"] == pf_label, "onset_s"].to_numpy(
                    dtype=float
                )
                _plot_onset_ecdf(
                    ax,
                    vals,
                    label=f"{OPTIMIZER_LABELS[opt]} — {pf_label}",
                    color=shade,
                    linestyle=linestyle,
                )
        ax.set_xlabel("Onset time within train_same block (s)")
        ax.set_ylabel("CDF")
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=6, loc="lower right")
    fig.suptitle("PF onset time CDF (train_same) by optimizer", y=1.02)
    fig.tight_layout()
    return fig


def _plot_pc_cohort_retention_on_ax(
    ax,
    retention: pd.DataFrame,
    *,
    show_room_legend: bool = False,
    show_cohort_legend: bool = True,
) -> None:
    birth_cohorts = sorted(retention["birth_cohort"].unique())
    for birth_cohort in birth_cohorts:
        color = retention.loc[retention["birth_cohort"] == birth_cohort, "color"].iloc[0]
        for room_idx in ROOM_IDXS:
            sub = retention[
                (retention["eval_room_id"] == room_idx)
                & (retention["birth_cohort"] == birth_cohort)
            ].sort_values("recording_day")
            marker = "o" if room_idx == 0 else "s"
            ax.plot(
                sub["recording_day"],
                sub["count"],
                f"{marker}-",
                color=color,
                markersize=6,
                linewidth=1.5,
                markeredgecolor="0.2",
                markeredgewidth=0.6,
                label=(
                    ROOM_TITLES[room_idx]
                    if show_room_legend and birth_cohort == birth_cohorts[0]
                    else None
                ),
            )
        if show_cohort_legend:
            ax.plot(
                [],
                [],
                "o-",
                color=color,
                linewidth=1.5,
                markersize=6,
                label=PC_BIRTH_COHORT_LABELS.get(
                    birth_cohort,
                    f"Born day {birth_cohort}",
                ),
            )
    ax.set_xticks(birth_cohorts)
    ax.set_xlabel("Recording day")
    ax.set_ylabel("# place cells")


def plot_consistent_pf_cohorts_by_optimizer(
    results: TwoRoomsMultiOptResults,
) -> tuple[Figure, np.ndarray]:
    opts = [o for o in OPTIMIZER_ORDER if o in results.pf]
    fig, axes = plt.subplots(1, len(opts), figsize=(3.2 * len(opts), 4.5), squeeze=False)
    for col, opt in enumerate(opts):
        ax = axes[0, col]
        _plot_pc_cohort_retention_on_ax(
            ax,
            results.pf[opt].pc_cohort_retention,
            show_room_legend=(col == 0),
            show_cohort_legend=(col == 0),
        )
        ax.set_title(OPTIMIZER_LABELS[opt])
        if col != 0:
            ax.set_ylabel("")
    fig.suptitle(
        "PC cohort retention (train_same; circles=room0, squares=room1)",
        y=1.02,
    )
    if opts:
        axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig, axes


# --- Section 3: cohort-split wrappers ---


def plot_pf_signal_group_by_optimizer_cohort(
    results: TwoRoomsMultiOptResults,
    *,
    cohort: Literal["sustained", "transient"],
    variant: str = "normalized_abs",
    groups_attr: str = "pf_eff_lr_groups",
    stat_prefix: str = "effective_lr",
    signal_label: str = "effective LR",
    show_significance: bool = True,
) -> Figure:
    from single_room_multi_opt_plots import plot_pf_signal_group_by_optimizer

    view = to_cohort_multi_opt_view(results, cohort=cohort)
    fig = plot_pf_signal_group_by_optimizer(
        view,
        variant=variant,
        groups_attr=groups_attr,
        stat_prefix=stat_prefix,
        signal_label=signal_label,
        show_significance=show_significance,
    )
    fig.suptitle(
        fig._suptitle.get_text() + f" — {cohort} PFs only"
        if fig._suptitle is not None
        else f"{signal_label} by segment group ({cohort} PFs)",
        y=1.02,
    )
    return fig


def plot_lifecycle_outlier_summary_by_optimizer_cohort(
    results: TwoRoomsMultiOptResults,
    *,
    cohort: Literal["sustained", "transient"],
    fit_type: str | None = None,
    probability_kinds: tuple[str, ...] = ("pmf", "cdf"),
    probs_attr: str = "lifecycle_probs",
    show_significance: bool = True,
) -> Figure:
    from single_room_multi_opt_plots import plot_lifecycle_outlier_summary_by_optimizer

    view = to_cohort_multi_opt_view(results, cohort=cohort)
    fig = plot_lifecycle_outlier_summary_by_optimizer(
        view,
        fit_type=fit_type,
        probability_kinds=probability_kinds,
        probs_attr=probs_attr,
        show_significance=show_significance,
    )
    fig.suptitle(
        f"Lifecycle outlier probabilities ({cohort} PFs) — "
        + (" / ".join(probability_kinds)),
        y=1.02,
    )
    return fig
