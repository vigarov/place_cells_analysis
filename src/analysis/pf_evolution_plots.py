"""Plot helpers for the thorough place-field (PF) cycle analysis notebook."""
from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize

from analysis.dead_cells_analysis import (
    active_fraction_by_cycle,
    dead_proportion_matrix,
)
from analysis.gaussian_fit_cycle_analysis import (
    DEFAULT_GAUSSIAN_FIELD_SHAPE,
    GAUSSIAN_PARAM_NAMES,
)
from analysis.gaussian_fit_cycle_plots import save_figure  # re-exported for the nb
from analysis.pf_evolution_analysis import (
    CellPfGrid,
    WEIGHTED_METRICS,
    render_visit_pf_field,
    visit_lookup_for_cell,
)
from analysis.pre_post_cells_analysis import PRE_POST_STATES

__all__ = [
    "save_figure",
    "plot_dead_proportion_grouped_bars",
    "plot_dead_proportion_multithreshold_lines",
    "plot_dead_proportion_multithreshold_scatter",
    "plot_recoveries_violin",
    "plot_recoveries_boxplot",
    "plot_cell_pf_grid",
    "plot_active_rooms_violin",
    "plot_weighted_timeseries",
    "plot_pf_std_violin",
    "plot_pf_std_boxplot",
    "plot_active_fraction_heatmap",
    "plot_population_activity_curve",
    "plot_onset_cycle_hist",
    "plot_center_drift_hist",
    "plot_rooms_active_hist",
    "plot_pf_count_timeseries",
    "plot_pre_post_pf_creation_cycle_bars",
    "plot_pre_post_pf_activation_gaussians",
    "plot_pre_post_cell_room_cycle_evolution",
    "plot_pf_hk_timeseries",
    "plot_pre_post_hk_ratemaps",
    "plot_pre_post_state_stacked_bars",
    "plot_dd_proportion_multithreshold_lines",
    "plot_dd_proportion_multithreshold_scatter",
    "plot_da_activations_violin",
    "plot_da_activations_boxplot",
    "plot_pre_da_aa_ad_count_multithreshold_lines",
    "plot_perfect_da_activation_count_multithreshold_lines",
    "plot_perfect_da_activation_count_multithreshold_ecdf",
    "plot_ever_da_state_count_stacked_bars",
    "plot_pre_post_state_timeline",
]

_STATE_HATCHES: dict[str, str] = {
    "AA": "",
    "AD": "///",
    "DA": "...",
    "DD": "xxx",
}
_STATE_COLORS: dict[str, str] = {
    "AA": "#2ca02c",
    "AD": "#ff7f0e",
    "DA": "#1f77b4",
    "DD": "#7f7f7f",
}
_STACK_ORDER: tuple[str, ...] = ("DD", "DA", "AD", "AA")

_METRIC_LABELS = {
    "amplitude": "amplitude",
    "x_center": "x centre (px)",
    "y_center": "y centre (px)",
}


def _pf_cmap() -> plt.Colormap:
    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad(color="0.9")
    return cmap


def plot_dead_proportion_grouped_bars(
    prop_df: pd.DataFrame,
    *,
    title: str = "Dead-cell proportion per cycle, grouped by room",
    cmap_name: str = "viridis",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Grouped bar chart: x = cycle, one bar per room, y = proportion dead.

    Rooms are encoded by colour (sequential colormap + colorbar) instead of a
    huge legend, so it stays readable with many rooms.
    """
    matrix = prop_df.pivot(
        index="cycle_id", columns="room_id", values="proportion_dead"
    ).sort_index()
    cycles = matrix.index.to_numpy()
    rooms = matrix.columns.to_numpy()
    n_cycles = len(cycles)
    n_rooms = len(rooms)

    if figsize is None:
        figsize = (max(12.0, n_cycles * 1.15), 5.5)
    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=float(rooms.min()), vmax=float(rooms.max()))
    group_width = 0.82
    bar_width = group_width / max(1, n_rooms)
    x = np.arange(n_cycles)

    for j, room in enumerate(rooms):
        offset = (j - (n_rooms - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            matrix[room].to_numpy(),
            width=bar_width,
            color=cmap(norm(room)),
            linewidth=0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c) + 1) for c in cycles])
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Proportion of dead cells")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.margins(x=0.01)

    scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.01)
    cbar.set_label("Room")
    fig.tight_layout()
    return fig


def plot_dead_proportion_multithreshold_lines(
    multi_df: pd.DataFrame,
    *,
    show_band: bool = True,
    title: str = "Dead-cell proportion per cycle (mean over rooms)",
    figsize: tuple[float, float] = (11, 5.5),
) -> plt.Figure:
    """
    One line per dead threshold: x = cycle, y = mean-over-rooms dead proportion.

    The optional shaded band shows +/-1 SEM across rooms.
    """
    stats = (
        multi_df.groupby(["dead_threshold", "cycle_id"])["proportion_dead"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats["sem"] = stats["std"] / np.sqrt(stats["count"])
    thresholds = sorted(stats["dead_threshold"].unique())
    cmap = plt.get_cmap("plasma")

    fig, ax = plt.subplots(figsize=figsize)
    for i, threshold in enumerate(thresholds):
        sub = stats[stats["dead_threshold"] == threshold].sort_values("cycle_id")
        color = cmap(i / max(1, len(thresholds) - 1))
        cycles = sub["cycle_id"].to_numpy()
        mean = sub["mean"].to_numpy()
        ax.plot(cycles, mean, marker="o", ms=4, color=color, label=f"{threshold:g}")
        if show_band:
            sem = sub["sem"].fillna(0).to_numpy()
            ax.fill_between(cycles, mean - sem, mean + sem, color=color, alpha=0.12)

    sorted_cycles = sorted(int(c) for c in stats["cycle_id"].unique())
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Proportion of dead cells")
    ax.set_ylim(0, 1)
    ax.set_xticks(sorted_cycles)
    ax.set_xticklabels([str(c + 1) for c in sorted_cycles])
    ax.set_title(title)
    ax.legend(title="dead threshold", ncol=2, fontsize=9)
    fig.tight_layout()
    return fig


def plot_dead_proportion_multithreshold_scatter(
    multi_df: pd.DataFrame,
    *,
    title: str = "Dead-cell proportion per room and cycle",
    figsize: tuple[float, float] = (12, 5.5),
    jitter: float = 0.7,
) -> plt.Figure:
    """
    Raw dead proportions without averaging: one dot per ``(room, cycle)``.

    One colour per dead threshold; within each cycle the rooms are sorted in
    increasing order of dead proportion and spread over a small x-jitter so the
    thresholds do not overlap.
    """
    thresholds = sorted(multi_df["dead_threshold"].unique())
    cycles = sorted(int(c) for c in multi_df["cycle_id"].unique())
    cmap = plt.get_cmap("plasma")
    n_thresh = len(thresholds)

    fig, ax = plt.subplots(figsize=figsize)
    for i, threshold in enumerate(thresholds):
        color = cmap(i / max(1, n_thresh - 1))
        offset = (i - (n_thresh - 1) / 2) / max(1, n_thresh) * jitter
        xs: list[float] = []
        ys: list[float] = []
        for cycle in cycles:
            sub = multi_df[
                (multi_df["dead_threshold"] == threshold)
                & (multi_df["cycle_id"] == cycle)
            ].sort_values("proportion_dead")
            props = sub["proportion_dead"].to_numpy()
            n = len(props)
            if n == 0:
                continue
            spread = np.linspace(-0.5, 0.5, n) * jitter if n > 1 else np.zeros(1)
            xs.extend(cycle + offset + spread)
            ys.extend(props)
        ax.scatter(xs, ys, s=10, color=color, alpha=0.6, label=f"{threshold:g}")

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Proportion of dead cells")
    ax.set_ylim(0, 1)
    ax.set_xticks(cycles)
    ax.set_xticklabels([str(c + 1) for c in cycles])
    ax.set_title(title)
    ax.legend(title="dead threshold", ncol=2, fontsize=9)
    fig.tight_layout()
    return fig


def plot_recoveries_violin(
    recoveries_long: pd.DataFrame,
    *,
    title: str = "Per-cell recoveries (dead → active) vs dead threshold",
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Seaborn violin of per-cell recovery counts, one violin per threshold."""
    data = recoveries_long.copy()
    data["threshold"] = data["dead_threshold"].map(lambda t: f"{t:g}")
    order = [f"{t:g}" for t in sorted(data["dead_threshold"].unique())]

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(
        data=data,
        x="threshold",
        y="n_recoveries",
        order=order,
        cut=0,
        density_norm="width",
        inner="quartile",
        ax=ax,
    )
    ax.set_xlabel("Dead threshold")
    ax.set_ylabel("Recoveries per cell")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_recoveries_boxplot(
    recoveries_long: pd.DataFrame,
    *,
    title: str = "Per-cell recoveries (dead → active) vs dead threshold",
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Seaborn boxplot of per-cell recovery counts, one box per threshold."""
    data = recoveries_long.copy()
    data["threshold"] = data["dead_threshold"].map(lambda t: f"{t:g}")
    order = [f"{t:g}" for t in sorted(data["dead_threshold"].unique())]

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=data,
        x="threshold",
        y="n_recoveries",
        order=order,
        flierprops={"marker": ".", "markersize": 3},
        ax=ax,
    )
    ax.set_xlabel("Dead threshold")
    ax.set_ylabel("Recoveries per cell")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_cell_pf_grid(
    grid: CellPfGrid,
    *,
    im_size: float = 0.42,
    mark_dead: bool = True,
) -> plt.Figure:
    """
    Place field (Gaussian sum) for one cell across rooms (rows) x cycles (cols).

    Failed fits show a grey panel; dead panels are outlined faintly in red when
    ``mark_dead`` is set.
    """
    rooms = grid.rooms
    cycles = grid.cycles
    n_rows = len(rooms)
    n_cols = len(cycles)
    cmap = _pf_cmap()

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(im_size * n_cols + 1.2, im_size * n_rows + 1.0),
        squeeze=False,
    )

    blank = None
    im = None
    for i, room in enumerate(rooms):
        for j, cycle in enumerate(cycles):
            ax = axes[i, j]
            field = grid.fields.get((room, cycle))
            if field is None:
                if blank is None:
                    blank = np.full((2, 2), np.nan)
                ax.imshow(blank, cmap=cmap, vmin=0, vmax=grid.vmax)
            else:
                im = ax.imshow(field, cmap=cmap, vmin=0, vmax=grid.vmax)
            if mark_dead and grid.dead.get((room, cycle), False):
                for spine in ax.spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(0.6)
                    spine.set_alpha(0.5)
            else:
                for spine in ax.spines.values():
                    spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(
                    f"R{int(room)}",
                    fontsize=6,
                    rotation=0,
                    ha="right",
                    va="center",
                )
            if i == 0:
                ax.set_title(f"C{int(cycle) + 1}", fontsize=6)

    fig.suptitle(
        f"Cell {grid.cell_idx}: place field across rooms x cycles "
        f"(red outline = dead)",
        fontsize=11,
        y=1.005,
    )
    if im is not None:
        fig.subplots_adjust(right=0.9)
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, pad=0.01)
        cbar.set_label("PF amplitude")
    return fig


def plot_active_rooms_violin(
    active_df: pd.DataFrame,
    *,
    title: str = "Number of rooms a cell is active for, per cycle",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Seaborn violin (one per cycle) of per-cell active-room counts."""
    cycles = sorted(int(c) for c in active_df["cycle_id"].unique())
    if figsize is None:
        figsize = (max(9.0, len(cycles) * 0.45), 5.0)

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(
        data=active_df,
        x="cycle_id",
        y="n_active_rooms",
        order=cycles,
        cut=0,
        density_norm="width",
        inner="quartile",
        ax=ax,
    )
    means = active_df.groupby("cycle_id")["n_active_rooms"].mean().reindex(cycles)
    ax.plot(
        range(len(cycles)),
        means.to_numpy(),
        color="crimson",
        marker="o",
        ms=4,
        lw=1.5,
        label="mean",
    )
    ax.set_xticklabels([f"C{c + 1}" for c in cycles], rotation=45, ha="right")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("# rooms active")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_weighted_timeseries(
    timeseries: pd.DataFrame,
    *,
    cell_idx: int,
    room_id: int,
    figsize: tuple[float, float] = (7, 4),
) -> plt.Figure:
    """
    Weighted PF descriptor time-series for one ``(cell, room)``.

    Amplitude on the left axis; x/y centres (pixel coords) on a twin right axis.
    """
    cycles = timeseries["cycle_id"].to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    (l_amp,) = ax.plot(
        cycles,
        timeseries["w_amplitude"].to_numpy(),
        color="C0",
        marker="o",
        ms=4,
        label="amplitude",
    )
    (l_r2,) = ax.plot(
        cycles,
        timeseries["r2"].to_numpy(),
        color="C3",
        marker="d",
        ms=3,
        label="R²",
    )
    ax.set_xlabel("Cycle")
    ax.set_ylabel("weighted amplitude", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")

    ax2 = ax.twinx()
    (l_x,) = ax2.plot(
        cycles,
        timeseries["w_mu_x"].to_numpy(),
        color="C1",
        marker="s",
        ms=4,
        label="x centre",
    )
    (l_y,) = ax2.plot(
        cycles,
        timeseries["w_mu_y"].to_numpy(),
        color="C2",
        marker="^",
        ms=4,
        label="y centre",
    )
    ax2.set_ylabel("centre (px)")

    ax.set_title(f"Cell {cell_idx}, room {room_id}: weighted PF evolution")
    ax.legend(handles=[l_amp, l_r2, l_x, l_y], loc="best", fontsize=9)
    if len(cycles):
        ax.set_xticks(cycles)
        ax.set_xticklabels([str(int(c) + 1) for c in cycles])
    fig.tight_layout()
    return fig


def plot_pf_std_violin(
    std_long: pd.DataFrame,
    *,
    log_scale: bool = True,
    title: str = "PF time-series dispersion (std over cycles) per active cell-room",
    figsize: tuple[float, float] = (7, 5),
) -> plt.Figure:
    """
    Seaborn violin of the per cell-room std for the 3 weighted descriptors.

    Amplitude and centre coordinates live on different scales, so a symmetric
    log y-axis (default) keeps all three distributions visible in one plot.
    """
    order = list(WEIGHTED_METRICS)
    labels = [_METRIC_LABELS[m] for m in order]

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(
        data=std_long,
        x="metric",
        y="std",
        order=order,
        cut=0,
        density_norm="width",
        inner="quartile",
        ax=ax,
    )
    if log_scale:
        positive = std_long.loc[std_long["std"] > 0, "std"]
        linthresh = float(positive.min()) if not positive.empty else 1e-3
        ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Descriptor")
    ax.set_ylabel("Std over active cycles")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pf_std_boxplot(
    std_long: pd.DataFrame,
    *,
    log_scale: bool = True,
    title: str = "PF time-series dispersion (std over cycles) per active cell-room",
    figsize: tuple[float, float] = (7, 5),
) -> plt.Figure:
    """
    Seaborn boxplot of the per cell-room std for the 3 weighted descriptors.

    Amplitude and centre coordinates live on different scales, so a symmetric
    log y-axis (default) keeps all three distributions visible in one plot.
    """
    order = list(WEIGHTED_METRICS)
    labels = [_METRIC_LABELS[m] for m in order]

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=std_long,
        x="metric",
        y="std",
        order=order,
        flierprops={"marker": ".", "markersize": 3},
        ax=ax,
    )
    if log_scale:
        positive = std_long.loc[std_long["std"] > 0, "std"]
        linthresh = float(positive.min()) if not positive.empty else 1e-3
        ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Descriptor")
    ax.set_ylabel("Std over active cycles")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_active_fraction_heatmap(
    dead_df: pd.DataFrame,
    *,
    title: str = "Active-cell fraction per room x cycle",
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Heatmap of active-cell fraction (1 - dead proportion) over room x cycle."""
    dead_matrix = dead_proportion_matrix(dead_df)
    active_matrix = 1.0 - dead_matrix
    active_matrix = active_matrix.rename(columns=lambda c: int(c) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        active_matrix,
        ax=ax,
        cmap="magma",
        vmin=0,
        vmax=float(np.nanmax(active_matrix.to_numpy())) or 1.0,
        cbar_kws={"label": "active-cell fraction"},
    )
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Room")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_population_activity_curve(
    dead_df: pd.DataFrame,
    *,
    title: str = "Population activity over cycles",
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Active fraction and number of distinct active cells per cycle."""
    pop = active_fraction_by_cycle(dead_df)
    cycles = pop["cycle_id"].to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    (l_frac,) = ax.plot(
        cycles,
        pop["active_fraction"].to_numpy(),
        color="C0",
        marker="o",
        ms=4,
        label="active (cell,room) fraction",
    )
    ax.set_xlabel("Cycle")
    ax.set_ylabel("active (cell,room) fraction", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.set_ylim(bottom=0)

    ax2 = ax.twinx()
    (l_cells,) = ax2.plot(
        cycles,
        pop["n_distinct_active_cells"].to_numpy(),
        color="C3",
        marker="s",
        ms=4,
        label="# distinct active cells",
    )
    ax2.set_ylabel("# distinct active cells", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    ax.set_title(title)
    ax.legend(handles=[l_frac, l_cells], loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def plot_onset_cycle_hist(
    onset_df: pd.DataFrame,
    *,
    title: str = "When place fields first form (onset cycle)",
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Histogram of the first active cycle per (cell, room)."""
    onset = onset_df["onset_cycle"].to_numpy()
    cycles = np.arange(int(onset.min()), int(onset.max()) + 2)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(onset, bins=cycles - 0.5, color="C0", edgecolor="white")
    ax.set_xlabel("Onset cycle (first cycle active)")
    ax.set_ylabel("# (cell, room) pairs")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_center_drift_hist(
    drift_df: pd.DataFrame,
    *,
    title: str = "Total PF centre drift over active cycles",
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Histogram of cumulative weighted-centre path length per (cell, room)."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(
        drift_df["total_drift"].to_numpy(),
        bins=40,
        color="C2",
        edgecolor="white",
    )
    median = float(np.nanmedian(drift_df["total_drift"].to_numpy()))
    ax.axvline(median, color="crimson", lw=1.5, label=f"median = {median:.1f}px")
    ax.set_xlabel("Total centre drift (px, summed over cycles)")
    ax.set_ylabel("# (cell, room) pairs")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_rooms_active_hist(
    rooms_active: pd.Series,
    *,
    n_rooms: int,
    title: str = "Room generalisation: # rooms each active cell fires in",
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Histogram of the number of distinct rooms each active cell is active in."""
    counts = rooms_active.to_numpy()
    bins = np.arange(0.5, n_rooms + 1.5, 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(counts, bins=bins, color="C4", edgecolor="white")
    ax.set_xlabel("# distinct rooms active")
    ax.set_ylabel("# cells")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pf_count_timeseries(
    cycle_stats: pd.DataFrame,
    *,
    title: str = "Number of place fields per selected cell",
    figsize: tuple[float, float] = (9, 5),
) -> plt.Figure:
    """
    Bar chart of the per-cycle place-field count (mean +/- SEM over cells).

    ``cycle_stats`` has columns ``cycle_id, mean, sem``.
    """
    stats = cycle_stats.sort_values("cycle_id")
    cycles = stats["cycle_id"].to_numpy()
    x = np.arange(len(cycles))

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        x,
        stats["mean"].to_numpy(),
        yerr=stats["sem"].to_numpy(),
        color="C0",
        capsize=3,
        linewidth=0,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c) + 1) for c in cycles])
    ax.set_xlabel("Cycle")
    ax.set_ylabel("# place fields (mean ± sem over cells)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pre_post_pf_creation_cycle_bars(
    cycle_stats: pd.DataFrame,
    *,
    dead_threshold: float,
    r2_min: float,
    title: str | None = None,
    figsize: tuple[float, float] = (9, 5),
) -> plt.Figure:
    """Bar chart of pre→post PF creation counts per cycle."""
    stats = cycle_stats.sort_values("cycle_id")
    cycles = stats["cycle_id"].to_numpy()
    x = np.arange(len(cycles))

    if title is None:
        title = (
            "Pre→post place-field creations per cycle "
            f"(dead if signal_max < {dead_threshold}, "
            f"active if r2 > {r2_min} and signal_max > {dead_threshold})"
        )

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, stats["n_creations"].to_numpy(), color="C5", linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels(cycles)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("# (cell, room) creations")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pre_post_pf_activation_gaussians(
    activations_df: pd.DataFrame,
    df_rf_pre: pd.DataFrame,
    df_rf_post: pd.DataFrame,
    *,
    cycles: list[int],
    n_per_cycle: int,
    n_gaussians: int,
    render_shape: tuple[int, int] = (40, 40),
    field_extent: tuple[int, int] = DEFAULT_GAUSSIAN_FIELD_SHAPE,
    param_names: tuple[str, ...] = GAUSSIAN_PARAM_NAMES,
    title: str | None = None,
    im_width: float = 1.6,
    row_height: float = 1.6,
) -> plt.Figure:
    """
    Pre/post Gaussian-sum blobs for pre→post activations, grouped by cycle.

    One row per cycle in *cycles*; each row shows up to *n_per_cycle*
    ``(cell, room)`` activations as pre | post pairs (ranked strongest post
    ``signal_max`` first). Panel titles include cell, room, and cycle.
    """
    cmap = _pf_cmap()
    n_rows = len(cycles)
    n_cols = n_per_cycle * 2
    blank = np.full(render_shape, np.nan)

    shown_fields: list[np.ndarray] = []
    for cycle_id in cycles:
        cycle_rows = activations_df[activations_df["cycle_id"] == cycle_id]
        for _, row in cycle_rows.iterrows():
            v, c = int(row.visit_idx), int(row.cell_idx)
            for df_rf in (df_rf_pre, df_rf_post):
                field = render_visit_pf_field(
                    df_rf,
                    v,
                    c,
                    n_gaussians=n_gaussians,
                    render_shape=render_shape,
                    field_extent=field_extent,
                    param_names=param_names,
                )
                if field is not None:
                    shown_fields.append(field)
    vmax = float(np.nanmax(shown_fields)) if shown_fields else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(im_width * n_cols + 0.8, row_height * n_rows + 0.6),
        squeeze=False,
    )

    im = None
    for row_idx, cycle_id in enumerate(cycles):
        cycle_rows = (
            activations_df[activations_df["cycle_id"] == cycle_id]
            .sort_values("signal_max_post", ascending=False)
            .head(n_per_cycle)
        )
        for slot in range(n_per_cycle):
            ax_pre = axes[row_idx, slot * 2]
            ax_post = axes[row_idx, slot * 2 + 1]
            if slot >= len(cycle_rows):
                ax_pre.imshow(blank, cmap=cmap, vmin=0, vmax=vmax)
                ax_post.imshow(blank, cmap=cmap, vmin=0, vmax=vmax)
                ax_pre.axis("off")
                ax_post.axis("off")
                continue

            visit_row = cycle_rows.iloc[slot]
            v = int(visit_row.visit_idx)
            c = int(visit_row.cell_idx)
            rid = int(visit_row.room_id)
            field_pre = render_visit_pf_field(
                df_rf_pre,
                v,
                c,
                n_gaussians=n_gaussians,
                render_shape=render_shape,
                field_extent=field_extent,
                param_names=param_names,
            )
            field_post = render_visit_pf_field(
                df_rf_post,
                v,
                c,
                n_gaussians=n_gaussians,
                render_shape=render_shape,
                field_extent=field_extent,
                param_names=param_names,
            )
            label = f"cell {c}, R{rid} C{cycle_id}"
            for ax, field, phase in (
                (ax_pre, field_pre, "pre"),
                (ax_post, field_post, "post"),
            ):
                if field is None:
                    ax.imshow(blank, cmap=cmap, vmin=0, vmax=vmax)
                else:
                    im = ax.imshow(field, cmap=cmap, vmin=0, vmax=vmax)
                ax.set_title(f"{label}\n{phase}", fontsize=5)
                ax.set_xticks([])
                ax.set_yticks([])

        axes[row_idx, 0].set_ylabel(
            f"C{int(cycle_id) + 1}",
            fontsize=7,
            rotation=0,
            ha="right",
            va="center",
        )

    if title is None:
        title = (
            f"Pre→post PF activations "
            f"({n_per_cycle} per cycle, first {len(cycles)} cycles)"
        )
    fig.suptitle(title, fontsize=10, y=1.0)
    fig.tight_layout()
    if im is not None:
        fig.subplots_adjust(right=0.9)
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)
        cbar.set_label("Gaussian-sum amplitude")
    return fig


def plot_pre_post_cell_room_cycle_evolution(
    cell_idx: int,
    df_rf_pre: pd.DataFrame,
    df_rf_post: pd.DataFrame,
    *,
    cycles: list[int],
    rooms: list[int],
    n_gaussians: int,
    render_shape: tuple[int, int] = (40, 40),
    field_extent: tuple[int, int] = DEFAULT_GAUSSIAN_FIELD_SHAPE,
    param_names: tuple[str, ...] = GAUSSIAN_PARAM_NAMES,
    title: str | None = None,
    cell_width: float = 1.35,
    row_height: float = 1.35,
) -> plt.Figure:
    """
    Pre/post Gaussian-sum evolution for one cell across rooms and cycles.

    Outer grid: ``len(rooms) // 2`` rows (two rooms per row) x ``len(cycles)``
    columns. Each ``(room, cycle)`` entry is a pre | post pair.
    """
    cmap = _pf_cmap()
    blank = np.full(render_shape, np.nan)
    lookup_pre = visit_lookup_for_cell(df_rf_pre, cell_idx)
    lookup_post = visit_lookup_for_cell(df_rf_post, cell_idx)

    shown_fields: list[np.ndarray] = []
    for room_id in rooms:
        for cycle_id in cycles:
            key = (int(cycle_id), int(room_id))
            if key not in lookup_pre.index or key not in lookup_post.index:
                continue
            for df_rf, lookup in ((df_rf_pre, lookup_pre), (df_rf_post, lookup_post)):
                field = render_visit_pf_field(
                    df_rf,
                    int(lookup.loc[key]),
                    cell_idx,
                    n_gaussians=n_gaussians,
                    render_shape=render_shape,
                    field_extent=field_extent,
                    param_names=param_names,
                )
                if field is not None:
                    shown_fields.append(field)
    vmax = float(np.nanmax(shown_fields)) if shown_fields else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    n_room_rows = max(1, len(rooms) // 2)
    n_cycles = len(cycles)
    fig = plt.figure(
        figsize=(cell_width * n_cycles + 0.8, row_height * n_room_rows + 0.8),
    )
    outer = gridspec.GridSpec(
        n_room_rows,
        n_cycles,
        figure=fig,
        wspace=0.12,
        hspace=0.22,
    )

    im = None
    for pair_idx in range(n_room_rows):
        pair_rooms = rooms[2 * pair_idx : 2 * pair_idx + 2]
        for col_idx, cycle_id in enumerate(cycles):
            inner = gridspec.GridSpecFromSubplotSpec(
                len(pair_rooms),
                2,
                subplot_spec=outer[pair_idx, col_idx],
                wspace=0.04,
                hspace=0.08,
            )
            for room_idx, room_id in enumerate(pair_rooms):
                key = (int(cycle_id), int(room_id))
                for phase_idx, (df_rf, lookup, phase) in enumerate(
                    (
                        (df_rf_pre, lookup_pre, "pre"),
                        (df_rf_post, lookup_post, "post"),
                    )
                ):
                    ax = fig.add_subplot(inner[room_idx, phase_idx])
                    if key in lookup.index:
                        field = render_visit_pf_field(
                            df_rf,
                            int(lookup.loc[key]),
                            cell_idx,
                            n_gaussians=n_gaussians,
                            render_shape=render_shape,
                            field_extent=field_extent,
                            param_names=param_names,
                        )
                    else:
                        field = None
                    if field is None:
                        ax.imshow(blank, cmap=cmap, vmin=0, vmax=vmax)
                    else:
                        im = ax.imshow(field, cmap=cmap, vmin=0, vmax=vmax)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if room_idx == 0 and pair_idx == 0:
                        ax.set_title(f"C{int(cycle_id) + 1}", fontsize=6)
                    if col_idx == 0:
                        ax.set_ylabel(
                            f"R{int(room_id)}\n{phase}",
                            fontsize=5,
                            rotation=0,
                            ha="right",
                            va="center",
                        )

    if title is None:
        title = (
            f"Cell {cell_idx}: pre→post PF evolution "
            f"({len(rooms)} rooms x {len(cycles)} cycles)"
        )
    fig.suptitle(title, fontsize=10, y=1.01)
    if im is not None:
        fig.subplots_adjust(right=0.92)
        cbar = fig.colorbar(im, ax=fig.axes, shrink=0.5, pad=0.02)
        cbar.set_label("Gaussian-sum amplitude")
    fig.tight_layout()
    return fig


def plot_pf_hk_timeseries(
    hk_stats: pd.DataFrame,
    mass_stats: pd.DataFrame,
    *,
    title: str = "Cross-room place-field divergence over cycles",
    figsize: tuple[float, float] = (9, 5),
) -> plt.Figure:
    """
    Twin-axis time-series of pairwise HK distance and total mass over cycles.

    ``hk_stats`` has columns ``cycle_id, mean_hk, sem_hk``; ``mass_stats`` has
    ``cycle_id, mean_mass, sem_mass``. Both are mean +/- SEM over cells.
    """
    hk = hk_stats.sort_values("cycle_id")
    mass = mass_stats.sort_values("cycle_id")
    hk_cycles = hk["cycle_id"].to_numpy()
    mass_cycles = mass["cycle_id"].to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    (l_hk,) = ax.plot(
        hk_cycles,
        hk["mean_hk"].to_numpy(),
        color="C0",
        marker="o",
        ms=4,
        label="pairwise HK distance",
    )
    hk_sem = hk["sem_hk"].fillna(0).to_numpy()
    ax.fill_between(
        hk_cycles,
        hk["mean_hk"].to_numpy() - hk_sem,
        hk["mean_hk"].to_numpy() + hk_sem,
        color="C0",
        alpha=0.15,
    )
    ax.set_xlabel("Cycle")
    ax.set_ylabel("mean pairwise HK distance", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")

    ax2 = ax.twinx()
    (l_mass,) = ax2.plot(
        mass_cycles,
        mass["mean_mass"].to_numpy(),
        color="C3",
        marker="s",
        ms=4,
        label="total mass",
    )
    mass_sem = mass["sem_mass"].fillna(0).to_numpy()
    ax2.fill_between(
        mass_cycles,
        mass["mean_mass"].to_numpy() - mass_sem,
        mass["mean_mass"].to_numpy() + mass_sem,
        color="C3",
        alpha=0.15,
    )
    ax2.set_ylabel("mean total mass across rooms", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    all_cycles = sorted(set(int(c) for c in hk_cycles) | set(int(c) for c in mass_cycles))
    ax.set_xticks(all_cycles)
    ax.set_xticklabels([str(c + 1) for c in all_cycles])
    ax.set_title(title)
    ax.legend(handles=[l_hk, l_mass], loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def plot_pre_post_hk_ratemaps(
    hk_df: pd.DataFrame,
    ratemaps_pre: np.ndarray,
    ratemaps_post: np.ndarray,
    *,
    n_show: int = 50,
    entry_cols: int = 5,
    title: str | None = None,
    col_width: float = 2.2,
    row_height: float = 2.0,
) -> plt.Figure:
    """
    Grid of pre/post rate maps for the top ``n_show`` rows of *hk_df*.

    Each entry is a pre | post pair; ``entry_cols`` pairs are shown per row.
    Titles include cell, room, cycle, and HK distance.
    """
    show_df = hk_df.head(n_show)
    n_plot_rows = int(np.ceil(n_show / entry_cols))
    n_plot_cols = entry_cols * 2
    cmap = _pf_cmap()

    shown_maps: list[np.ndarray] = []
    for _, row in show_df.iterrows():
        v, c = int(row.visit_idx), int(row.cell_idx)
        shown_maps.extend([ratemaps_pre[v, c], ratemaps_post[v, c]])
    vmax = float(np.nanmax(shown_maps)) if shown_maps else 1.0
    if vmax <= 0:
        vmax = 1.0

    fig, axes = plt.subplots(
        n_plot_rows,
        n_plot_cols,
        figsize=(col_width * n_plot_cols, row_height * n_plot_rows),
        squeeze=False,
    )

    for idx, (_, row) in enumerate(show_df.iterrows()):
        entry_row = idx // entry_cols
        entry_col = idx % entry_cols
        v, c = int(row.visit_idx), int(row.cell_idx)
        rid, cid, hk_val = int(row.room_id), int(row.cycle_id), float(row.hk)

        ax_pre = axes[entry_row, entry_col * 2]
        ax_post = axes[entry_row, entry_col * 2 + 1]
        ax_pre.imshow(ratemaps_pre[v, c], cmap=cmap, vmin=0, vmax=vmax)
        ax_post.imshow(ratemaps_post[v, c], cmap=cmap, vmin=0, vmax=vmax)

        label = f"#{idx + 1} cell {c}, R{rid} C{cid}"
        ax_pre.set_title(f"{label}\npre", fontsize=5)
        ax_post.set_title(f"HK={hk_val:.1f}\npost", fontsize=5)
        for ax in (ax_pre, ax_post):
            ax.set_xticks([])
            ax.set_yticks([])

    for idx in range(len(show_df), n_plot_rows * entry_cols):
        entry_row = idx // entry_cols
        entry_col = idx % entry_cols
        axes[entry_row, entry_col * 2].axis("off")
        axes[entry_row, entry_col * 2 + 1].axis("off")

    if title is None:
        title = f"Pre vs post-training rate maps (top {n_show} by HK distance)"
    fig.suptitle(title, fontsize=10, y=1.01)
    fig.tight_layout()
    return fig


def plot_pre_post_state_stacked_bars(
    prop_df: pd.DataFrame,
    *,
    title: str = "Pre/post state distribution per cycle, grouped by room",
    cmap_name: str = "viridis",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Stacked bar chart: x = cycle, one bar per room, y = state proportions.

    Room colour is fixed per room; hatch pattern encodes state (AA/AD/DA/DD).
    """
    matrix = prop_df.pivot_table(
        index=["room_id", "cycle_id"],
        columns="state",
        values="proportion",
        fill_value=0.0,
    )
    for state in PRE_POST_STATES:
        if state not in matrix.columns:
            matrix[state] = 0.0
    matrix = matrix[list(PRE_POST_STATES)]

    rooms = sorted(int(r) for r in prop_df["room_id"].unique())
    cycles = sorted(int(c) for c in prop_df["cycle_id"].unique())
    n_cycles = len(cycles)
    n_rooms = len(rooms)

    if figsize is None:
        figsize = (max(12.0, n_cycles * 1.15), 5.5)
    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=float(min(rooms)), vmax=float(max(rooms)))
    group_width = 0.82
    bar_width = group_width / max(1, n_rooms)
    x = np.arange(n_cycles)

    for j, room in enumerate(rooms):
        offset = (j - (n_rooms - 1) / 2) * bar_width
        bottom = np.zeros(n_cycles)
        room_color = cmap(norm(room))
        for state in _STACK_ORDER:
            heights = np.array(
                [
                    float(matrix.loc[(room, cycle), state])
                    if (room, cycle) in matrix.index
                    else 0.0
                    for cycle in cycles
                ]
            )
            ax.bar(
                x + offset,
                heights,
                width=bar_width,
                bottom=bottom,
                color=room_color,
                hatch=_STATE_HATCHES[state],
                edgecolor="white",
                linewidth=0.4,
            )
            bottom = bottom + heights

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c) + 1) for c in cycles])
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Fraction of cells")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.margins(x=0.01)

    scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.01)
    cbar.set_label("Room")

    from matplotlib.patches import Patch

    hatch_handles = [
        Patch(
            facecolor="0.85",
            edgecolor="0.3",
            hatch=_STATE_HATCHES[s],
            label=s,
        )
        for s in PRE_POST_STATES
    ]
    ax.legend(handles=hatch_handles, title="State", loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_dd_proportion_multithreshold_lines(
    multi_df: pd.DataFrame,
    *,
    show_band: bool = True,
    title: str = "DD proportion per cycle (mean over rooms)",
    figsize: tuple[float, float] = (11, 5.5),
) -> plt.Figure:
    """One line per threshold: x = cycle, y = mean-over-rooms DD proportion."""
    stats = (
        multi_df.groupby(["dead_threshold", "cycle_id"])["proportion_dd"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats["sem"] = stats["std"] / np.sqrt(stats["count"])
    thresholds = sorted(stats["dead_threshold"].unique())
    cmap = plt.get_cmap("plasma")

    fig, ax = plt.subplots(figsize=figsize)
    for i, threshold in enumerate(thresholds):
        sub = stats[stats["dead_threshold"] == threshold].sort_values("cycle_id")
        color = cmap(i / max(1, len(thresholds) - 1))
        cycles = sub["cycle_id"].to_numpy()
        mean = sub["mean"].to_numpy()
        ax.plot(cycles, mean, marker="o", ms=4, color=color, label=f"{threshold:g}")
        if show_band:
            sem = sub["sem"].fillna(0).to_numpy()
            ax.fill_between(cycles, mean - sem, mean + sem, color=color, alpha=0.12)

    sorted_cycles = sorted(int(c) for c in stats["cycle_id"].unique())
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Proportion of DD (dead pre & post)")
    ax.set_ylim(0, 1)
    ax.set_xticks(sorted_cycles)
    ax.set_xticklabels([str(c + 1) for c in sorted_cycles])
    ax.set_title(title)
    ax.legend(title="dead threshold", ncol=2, fontsize=9)
    fig.tight_layout()
    return fig


def plot_dd_proportion_multithreshold_scatter(
    multi_df: pd.DataFrame,
    *,
    title: str = "DD proportion per room and cycle",
    figsize: tuple[float, float] = (12, 5.5),
    jitter: float = 0.7,
) -> plt.Figure:
    """Raw DD proportions: one dot per ``(room, cycle)`` per threshold."""
    thresholds = sorted(multi_df["dead_threshold"].unique())
    cycles = sorted(int(c) for c in multi_df["cycle_id"].unique())
    cmap = plt.get_cmap("plasma")
    n_thresh = len(thresholds)

    fig, ax = plt.subplots(figsize=figsize)
    for i, threshold in enumerate(thresholds):
        color = cmap(i / max(1, n_thresh - 1))
        offset = (i - (n_thresh - 1) / 2) / max(1, n_thresh) * jitter
        xs: list[float] = []
        ys: list[float] = []
        for cycle in cycles:
            sub = multi_df[
                (multi_df["dead_threshold"] == threshold)
                & (multi_df["cycle_id"] == cycle)
            ].sort_values("proportion_dd")
            props = sub["proportion_dd"].to_numpy()
            n = len(props)
            if n == 0:
                continue
            spread = np.linspace(-0.5, 0.5, n) * jitter if n > 1 else np.zeros(1)
            xs.extend(cycle + offset + spread)
            ys.extend(props)
        ax.scatter(xs, ys, s=10, color=color, alpha=0.6, label=f"{threshold:g}")

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Proportion of DD (dead pre & post)")
    ax.set_ylim(0, 1)
    ax.set_xticks(cycles)
    ax.set_xticklabels([str(c + 1) for c in cycles])
    ax.set_title(title)
    ax.legend(title="dead threshold", ncol=2, fontsize=9)
    fig.tight_layout()
    return fig


def plot_da_activations_violin(
    da_multi_df: pd.DataFrame,
    *,
    title: str = "Per-cell DA activations (dead pre, active post) vs threshold",
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Seaborn violin of per-cell DA counts, one violin per threshold."""
    data = da_multi_df.copy()
    data["threshold"] = data["dead_threshold"].map(lambda t: f"{t:g}")
    order = [f"{t:g}" for t in sorted(data["dead_threshold"].unique())]

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(
        data=data,
        x="threshold",
        y="n_da_activations",
        order=order,
        cut=0,
        density_norm="width",
        inner="quartile",
        ax=ax,
    )
    ax.set_xlabel("Dead threshold")
    ax.set_ylabel("DA activations per cell")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_da_activations_boxplot(
    da_multi_df: pd.DataFrame,
    *,
    title: str = "Per-cell DA activations (dead pre, active post) vs threshold",
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Seaborn boxplot of per-cell DA counts, one box per threshold."""
    data = da_multi_df.copy()
    data["threshold"] = data["dead_threshold"].map(lambda t: f"{t:g}")
    order = [f"{t:g}" for t in sorted(data["dead_threshold"].unique())]

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=data,
        x="threshold",
        y="n_da_activations",
        order=order,
        flierprops={"marker": ".", "markersize": 3},
        ax=ax,
    )
    ax.set_xlabel("Dead threshold")
    ax.set_ylabel("DA activations per cell")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pre_da_aa_ad_count_multithreshold_lines(
    multi_df: pd.DataFrame,
    *,
    show_band: bool = False,
    title: str = (
        "Ever-DA (cell, room) pairs in AA or AD before first DA (by cycle)"
    ),
    figsize: tuple[float, float] = (11, 5.5),
) -> plt.Figure:
    """
    One line per dead threshold: count of ever-DA pairs in AA/AD at each cycle
    while still before their first DA cycle.
    """
    thresholds = sorted(multi_df["dead_threshold"].unique())
    cmap = plt.get_cmap("plasma")
    cycles = sorted(int(c) for c in multi_df["cycle_id"].unique())

    fig, ax = plt.subplots(figsize=figsize)
    for i, threshold in enumerate(thresholds):
        sub = multi_df[multi_df["dead_threshold"] == threshold].sort_values(
            "cycle_id"
        )
        color = cmap(i / max(1, len(thresholds) - 1))
        xs = sub["cycle_id"].to_numpy()
        ys = sub["count"].to_numpy()
        ax.plot(xs, ys, marker="o", ms=4, color=color, label=f"{threshold:g}")

    ax.set_xlabel("Cycle")
    ax.set_ylabel("# (cell, room) pairs in AA or AD (pre-first-DA)")
    ax.set_xticks(cycles)
    ax.set_xticklabels([str(c + 1) for c in cycles])
    ax.set_title(title)
    ax.legend(title="dead threshold", ncol=2, fontsize=9)
    fig.tight_layout()
    return fig


def plot_perfect_da_activation_count_multithreshold_lines(
    multi_df: pd.DataFrame,
    *,
    title: str = (
        "Perfect (cell, room) pairs by first-DA activation cycle "
        "(DD before, AA after)"
    ),
    figsize: tuple[float, float] = (11, 5.5),
) -> plt.Figure:
    """
    One line per dead threshold: count of perfect pairs vs first-DA cycle.

    A perfect pair is DD at every cycle before first DA and AA at every cycle
    after (with DA at the activation cycle).
    """
    thresholds = sorted(multi_df["dead_threshold"].unique())
    cmap = plt.get_cmap("plasma")
    cycles = sorted(int(c) for c in multi_df["first_da_cycle"].unique())

    fig, ax = plt.subplots(figsize=figsize)
    for i, threshold in enumerate(thresholds):
        sub = multi_df[multi_df["dead_threshold"] == threshold].sort_values(
            "first_da_cycle"
        )
        color = cmap(i / max(1, len(thresholds) - 1))
        xs = sub["first_da_cycle"].to_numpy()
        ys = sub["count"].to_numpy()
        ax.plot(xs, ys, marker="o", ms=4, color=color, label=f"{threshold:g}")

    ax.set_xlabel("First DA cycle (activation cycle)")
    ax.set_ylabel("# perfect (cell, room) pairs")
    ax.set_xticks(cycles)
    ax.set_xticklabels([str(c + 1) for c in cycles])
    ax.set_title(title)
    ax.legend(title="dead threshold", ncol=2, fontsize=9)
    fig.tight_layout()
    return fig


def plot_perfect_da_activation_count_multithreshold_ecdf(
    multi_df: pd.DataFrame,
    *,
    title: str = (
        "ECDF of perfect-pair first-DA activation cycle "
        "(DD before, AA after)"
    ),
    figsize: tuple[float, float] = (11, 5.5),
) -> plt.Figure:
    """
    One ECDF per dead threshold: cumulative fraction of perfect pairs activated
    by each first-DA cycle.
    """
    thresholds = sorted(multi_df["dead_threshold"].unique())
    cmap = plt.get_cmap("plasma")
    cycles = sorted(int(c) for c in multi_df["first_da_cycle"].unique())

    fig, ax = plt.subplots(figsize=figsize)
    for i, threshold in enumerate(thresholds):
        sub = multi_df[multi_df["dead_threshold"] == threshold].sort_values(
            "first_da_cycle"
        )
        counts = sub["count"].to_numpy(dtype=float)
        total = counts.sum()
        if total <= 0:
            continue
        color = cmap(i / max(1, len(thresholds) - 1))
        xs = sub["first_da_cycle"].to_numpy()
        ecdf = np.cumsum(counts) / total
        ax.step(
            xs,
            ecdf,
            where="post",
            color=color,
            label=f"{threshold:g}",
            linewidth=1.5,
        )
        ax.scatter(xs, ecdf, s=16, color=color, zorder=3)

    ax.set_xlabel("First DA cycle (activation cycle)")
    ax.set_ylabel("Cumulative fraction of perfect pairs")
    ax.set_ylim(0, 1)
    ax.set_xticks(cycles)
    ax.set_xticklabels([str(c + 1) for c in cycles])
    ax.set_title(title)
    ax.legend(title="dead threshold", ncol=2, fontsize=9)
    fig.tight_layout()
    return fig


def plot_ever_da_state_count_stacked_bars(
    counts_df: pd.DataFrame,
    *,
    title: str = "State counts per cycle (cell-room pairs with ≥1 DA)",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Stacked bar chart of raw state counts among ever-DA pairs."""
    matrix = counts_df.pivot_table(
        index="cycle_id",
        columns="state",
        values="count",
        fill_value=0,
        aggfunc="sum",
    ).sort_index()
    for state in PRE_POST_STATES:
        if state not in matrix.columns:
            matrix[state] = 0
    matrix = matrix[list(PRE_POST_STATES)]

    cycles = matrix.index.to_numpy()
    n_cycles = len(cycles)
    if figsize is None:
        figsize = (max(9.0, n_cycles * 1.0), 5.5)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_cycles)
    bottom = np.zeros(n_cycles)
    for state in _STACK_ORDER:
        heights = matrix[state].to_numpy()
        ax.bar(
            x,
            heights,
            bottom=bottom,
            label=state,
            color=_STATE_COLORS[state],
            edgecolor="white",
            linewidth=0.4,
        )
        bottom = bottom + heights

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c) + 1) for c in cycles])
    ax.set_xlabel("Cycle")
    ax.set_ylabel("# (cell, room) pairs")
    ax.set_title(title)
    ax.legend(title="State", loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_pre_post_state_timeline(
    state_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    cycles: list[int],
    *,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Small multiples: coloured state strip per cycle for selected cell-room pairs.
    """
    n_pairs = len(pairs_df)
    if n_pairs == 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.set_title(title or "Pre/post state timelines")
        ax.text(0.5, 0.5, "No pairs selected", ha="center", va="center")
        return fig

    if figsize is None:
        figsize = (max(8.0, len(cycles) * 0.45), max(2.5, n_pairs * 0.55 + 1.0))
    fig, axes = plt.subplots(n_pairs, 1, figsize=figsize, squeeze=False)

    for row_idx, (_, pair) in enumerate(pairs_df.iterrows()):
        ax = axes[row_idx, 0]
        cell_idx = int(pair["cell_idx"])
        room_id = int(pair["room_id"])
        sub = state_df[
            (state_df["cell_idx"] == cell_idx) & (state_df["room_id"] == room_id)
        ].set_index("cycle_id")

        for j, cycle in enumerate(cycles):
            state = str(sub.loc[cycle, "state"]) if cycle in sub.index else "?"
            color = _STATE_COLORS.get(state, "#cccccc")
            ax.bar(
                j,
                1.0,
                width=0.9,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )

        ax.set_ylabel(f"c{cell_idx}\nR{room_id}", fontsize=7, rotation=0, ha="right")
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, len(cycles) - 0.5)
        if row_idx == 0:
            ax.set_xticks(range(len(cycles)))
            ax.set_xticklabels([str(c + 1) for c in cycles], fontsize=7)
        else:
            ax.set_xticks([])
        if "first_da_cycle" in pair.index and pd.notna(pair["first_da_cycle"]):
            first_da = int(pair["first_da_cycle"])
            ax.text(
                len(cycles) - 0.3,
                0.5,
                f"1st DA C{first_da + 1}",
                ha="right",
                va="center",
                fontsize=6,
                transform=ax.transData,
            )

    if title is None:
        title = "Pre/post state timelines for selected (cell, room) pairs"
    fig.suptitle(title, fontsize=10, y=1.02)

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=_STATE_COLORS[s], label=s) for s in PRE_POST_STATES
    ]
    fig.legend(
        handles=legend_handles,
        title="State",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=4,
        fontsize=8,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    return fig
