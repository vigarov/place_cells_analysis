"""Plots for Supplemental Figures 1 and 2."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_suppl_fig1_correlation(
    corr: np.ndarray,
    *,
    cycle_ids: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    cmap: str = "jet",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str = "Suppl. Fig. 1 — cross-correlation of all trials",
) -> plt.Figure:
    """
    Heatmap of trial–trial Pearson correlations (600×600 when fully run).

    Expects trials sorted cycle-major (see ``reorder_trials_by_cycle``). Cycle 0
    is at the top-left (``origin='upper'``). If *cycle_ids* is given, draw
    cycle block boundaries and label axes from the cycles present in the data
    (e.g. truncated runs only show C0 … C9, not the full protocol length).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure

    if vmin is None:
        vmin = float(np.nanmin(corr))
    if vmax is None:
        vmax = float(np.nanmax(corr))

    im = ax.imshow(corr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", aspect="equal")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Cycle")
    ax.set_title(title)

    if cycle_ids is not None:
        cycles = np.sort(np.unique(cycle_ids))
        boundaries: list[float] = []
        tick_positions: list[float] = []
        tick_labels: list[str] = []
        count = 0
        for i, c in enumerate(cycles):
            n_in_cycle = int(np.sum(cycle_ids == c))
            tick_positions.append(count + n_in_cycle / 2 - 0.5)
            tick_labels.append(f"C{int(c)}")
            count += n_in_cycle
            if i < len(cycles) - 1:
                boundaries.append(count - 0.5)
        for b in boundaries:
            ax.axhline(b, color="white", linewidth=0.4, alpha=0.7)
            ax.axvline(b, color="white", linewidth=0.4, alpha=0.7)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    return fig


def plot_suppl_fig2_drift(
    ratemaps_by_cycle: np.ndarray,
    cell_ids: np.ndarray,
    *,
    cycles: np.ndarray | None = None,
    room_label: str = "Room 1",
    vmax_percentile: float = 99.0,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Grid of rate maps: rows = cells, columns = cycles (Suppl. Fig. 2).

    Parameters
    ----------
    ratemaps_by_cycle : (n_cycles, n_units, H, W)
    cell_ids : (n_cells,)
    """
    n_cycles, _, h, w = ratemaps_by_cycle.shape
    n_cells = len(cell_ids)
    if cycles is None:
        cycles = np.arange(1, n_cycles + 1)

    if ax is None:
        fig, axes = plt.subplots(
            n_cells,
            n_cycles,
            figsize=(0.35 * n_cycles + 1, 0.35 * n_cells + 1),
            squeeze=False,
        )
    else:
        fig = ax.figure
        axes = np.array([[ax]])

    stack = ratemaps_by_cycle[:, cell_ids]
    vmax = np.nanpercentile(stack, vmax_percentile)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    for row, uid in enumerate(cell_ids):
        for col in range(n_cycles):
            ax_ij = axes[row, col]
            rm = ratemaps_by_cycle[col, uid]
            ax_ij.imshow(rm, cmap="jet", vmin=0, vmax=vmax)
            ax_ij.set_xticks([])
            ax_ij.set_yticks([])
            if row == 0:
                ax_ij.set_title(f"C{cycles[col] + 1}", fontsize=7)
            if col == 0:
                ax_ij.set_ylabel(f"U{uid}", fontsize=7)

    fig.suptitle(
        f"{room_label}: place-field drift across {n_cycles} cycles",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: Path | str, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
