"""Plot helpers for the thorough place-field (PF) cycle analysis notebook."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize

from analysis.dead_cells_analysis import (
    active_fraction_by_cycle,
    dead_proportion_matrix,
)
from analysis.gaussian_fit_cycle_plots import save_figure  # re-exported for the nb
from analysis.pf_evolution_analysis import CellPfGrid, WEIGHTED_METRICS

__all__ = [
    "save_figure",
    "plot_dead_proportion_grouped_bars",
    "plot_dead_proportion_multithreshold_lines",
    "plot_recoveries_violin",
    "plot_cell_pf_grid",
    "plot_active_rooms_violin",
    "plot_weighted_timeseries",
    "plot_pf_std_violin",
    "plot_active_fraction_heatmap",
    "plot_population_activity_curve",
    "plot_onset_cycle_hist",
    "plot_center_drift_hist",
    "plot_rooms_active_hist",
]

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
    title: str = "Dead-cell proportion per room, grouped by cycle",
    cmap_name: str = "viridis",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Grouped bar chart: x = room, one bar per cycle, y = proportion dead.

    Cycles are encoded by colour (sequential colormap + colorbar) instead of a
    huge legend, so it stays readable with many cycles.
    """
    matrix = prop_df.pivot(
        index="room_id", columns="cycle_id", values="proportion_dead"
    ).sort_index()
    rooms = matrix.index.to_numpy()
    cycles = matrix.columns.to_numpy()
    n_rooms = len(rooms)
    n_cycles = len(cycles)

    if figsize is None:
        figsize = (max(12.0, n_rooms * 1.15), 5.5)
    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=float(cycles.min()), vmax=float(cycles.max()))
    group_width = 0.82
    bar_width = group_width / max(1, n_cycles)
    x = np.arange(n_rooms)

    for j, cycle in enumerate(cycles):
        offset = (j - (n_cycles - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            matrix[cycle].to_numpy(),
            width=bar_width,
            color=cmap(norm(cycle)),
            linewidth=0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(r)) for r in rooms])
    ax.set_xlabel("Room")
    ax.set_ylabel("Proportion of dead cells")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.margins(x=0.01)

    scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.01)
    cbar.set_label("Cycle (0 = first)")
    fig.tight_layout()
    return fig


def plot_dead_proportion_multithreshold_lines(
    multi_df: pd.DataFrame,
    *,
    show_band: bool = True,
    title: str = "Dead-cell proportion per room (mean over cycles)",
    figsize: tuple[float, float] = (11, 5.5),
) -> plt.Figure:
    """
    One line per dead threshold: x = room, y = mean-over-cycles dead proportion.

    The optional shaded band shows +/-1 std across cycles.
    """
    stats = (
        multi_df.groupby(["dead_threshold", "room_id"])["proportion_dead"]
        .agg(["mean", "std"])
        .reset_index()
    )
    thresholds = sorted(stats["dead_threshold"].unique())
    cmap = plt.get_cmap("plasma")

    fig, ax = plt.subplots(figsize=figsize)
    for i, threshold in enumerate(thresholds):
        sub = stats[stats["dead_threshold"] == threshold].sort_values("room_id")
        color = cmap(i / max(1, len(thresholds) - 1))
        rooms = sub["room_id"].to_numpy()
        mean = sub["mean"].to_numpy()
        ax.plot(rooms, mean, marker="o", ms=4, color=color, label=f"{threshold:g}")
        if show_band:
            std = sub["std"].fillna(0).to_numpy()
            ax.fill_between(rooms, mean - std, mean + std, color=color, alpha=0.12)

    ax.set_xlabel("Room")
    ax.set_ylabel("Proportion of dead cells")
    ax.set_ylim(0, 1)
    ax.set_xticks(sorted(int(r) for r in stats["room_id"].unique()))
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
    ax.legend(handles=[l_amp, l_x, l_y], loc="best", fontsize=9)
    if len(cycles):
        ax.set_xticks(cycles)
        ax.set_xticklabels([f"C{int(c) + 1}" for c in cycles], rotation=45, ha="right")
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


def plot_active_fraction_heatmap(
    dead_df: pd.DataFrame,
    *,
    title: str = "Active-cell fraction per room x cycle",
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Heatmap of active-cell fraction (1 - dead proportion) over room x cycle."""
    dead_matrix = dead_proportion_matrix(dead_df)
    active_matrix = 1.0 - dead_matrix

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        active_matrix,
        ax=ax,
        cmap="magma",
        vmin=0,
        vmax=float(np.nanmax(active_matrix.to_numpy())) or 1.0,
        cbar_kws={"label": "active-cell fraction"},
    )
    ax.set_xlabel("Cycle (0 = first)")
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
