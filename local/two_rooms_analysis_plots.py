"""Side-by-side plotting helpers for two_rooms_analysis.ipynb."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ROOM_TITLES = (
    "Room 0 (top-left bias)",
    "Room 1 (bottom-right bias)",
)


def side_by_side(
    plot_fn: Callable[[Axes, Any], None],
    left_data: Any,
    right_data: Any,
    *,
    figsize: tuple[float, float] = (14, 5),
    titles: tuple[str, str] = ROOM_TITLES,
    sharex: bool = True,
    sharey: bool = False,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Run ``plot_fn(ax, data)`` on two panels."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        constrained_layout=True,
    )
    plot_fn(axes[0], left_data, title=titles[0])
    plot_fn(axes[1], right_data, title=titles[1])
    return fig, (axes[0], axes[1])


def side_by_side_grids(
    plot_fn: Callable[[Axes, Any], None],
    left_data: Any,
    right_data: Any,
    *,
    figsize: tuple[float, float] = (18, 10),
    titles: tuple[str, str] = ROOM_TITLES,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Side-by-side panels without shared axes (e.g. ratemap grids)."""
    return side_by_side(
        plot_fn,
        left_data,
        right_data,
        figsize=figsize,
        titles=titles,
        sharex=False,
        sharey=False,
    )
