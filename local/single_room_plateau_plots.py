"""Paired normal-vs-plateau plots for the `single_room` plateau comparison.

The multi-optimizer helpers in `single_room_multi_opt_plots.py` compare optimizers
*within* one run. Here every panel compares two training modes across optimizers, so
each optimizer gets two bars: normal (solid) and plateau (hatched).

Two conventions differ from the multi-optimizer plots:

- Percentage changes are coloured by whether the change is an *improvement*, which
  depends on the metric. Lower is better for acquisition time, drift and revives;
  higher is better for spontaneity and field lifetime; and some metrics (peak
  amplitude, fields per neuron) have no preferred direction and are left neutral.
- Significance is tested *within* an optimizer, between the two modes, rather than
  between optimizers.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy import stats

from single_room_multi_opt_compute import (
    N_COLS,
    N_ROWS,
    OPTIMIZER_COLORS,
    OPTIMIZER_LABELS,
    _p_value_to_stars,
)

MODE_NORMAL = "normal"
MODE_PLATEAU = "plateau"
MODE_HATCH = {MODE_NORMAL: "", MODE_PLATEAU: "///"}

IMPROVE_COLOR = "#1a7f37"
WORSEN_COLOR = "#b3261e"
NEUTRAL_COLOR = "#444444"

LOWER_IS_BETTER = "lower"
HIGHER_IS_BETTER = "higher"
NO_PREFERENCE = "neutral"

METRIC_DIRECTION: dict[str, str] = {
    # Task error and how long a representation takes to appear: lower is better.
    "eval_error": LOWER_IS_BETTER,
    "train_error": LOWER_IS_BETTER,
    "tma_s": LOWER_IS_BETTER,
    "tma_iterations": LOWER_IS_BETTER,
    "formation_onset_s": LOWER_IS_BETTER,
    "start_iteration_offset": LOWER_IS_BETTER,
    # Instability: lower is better.
    "robustness": LOWER_IS_BETTER,
    "drift_x": LOWER_IS_BETTER,
    "drift_y": LOWER_IS_BETTER,
    "revives_per_pf": LOWER_IS_BETTER,
    # Spontaneity and persistence: higher is better.
    "one_segment": HIGHER_IS_BETTER,
    "spontaneous_frac": HIGHER_IS_BETTER,
    "spont_len_ge3": HIGHER_IS_BETTER,
    "spont_len_ge5": HIGHER_IS_BETTER,
    "spont_len_ge10": HIGHER_IS_BETTER,
    "spont_len_ge20": HIGHER_IS_BETTER,
    "length_s": HIGHER_IS_BETTER,
    "period_len_seg": HIGHER_IS_BETTER,
    # Descriptive: no direction is "better" on its own.
    "peak_amplitude": NO_PREFERENCE,
    "pfs_per_neuron": NO_PREFERENCE,
    "alive_frac": NO_PREFERENCE,
    "proportion_alive": NO_PREFERENCE,
}


def metric_direction(metric: str) -> str:
    return METRIC_DIRECTION.get(metric, NO_PREFERENCE)


def change_color(metric: str, delta_pct: float) -> str:
    """Colour a percentage change green when it improves the metric, red when not."""
    direction = metric_direction(metric)
    if direction == NO_PREFERENCE or not np.isfinite(delta_pct) or delta_pct == 0.0:
        return NEUTRAL_COLOR
    improved = delta_pct < 0 if direction == LOWER_IS_BETTER else delta_pct > 0
    return IMPROVE_COLOR if improved else WORSEN_COLOR


def _mean_ci95(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), 0.0
    sem = float(values.std(ddof=1) / np.sqrt(values.size))
    return float(values.mean()), 1.96 * sem


@dataclass(frozen=True)
class ModeComparison:
    """Within-optimizer normal-vs-plateau comparison for one metric."""

    optimizer: str
    metric: str
    normal_mean: float
    plateau_mean: float
    normal_ci: float
    plateau_ci: float
    delta_pct: float
    p_value: float
    stars: str
    n_normal: int
    n_plateau: int


def compare_modes(
    optimizer: str,
    metric: str,
    normal_values: np.ndarray,
    plateau_values: np.ndarray,
    *,
    test: str = "welch",
) -> ModeComparison:
    """Compare one metric between modes for one optimizer.

    `test` is `welch` (matches `pairwise_welch_holm` used elsewhere in this
    codebase) or `mannwhitney` (rank-based; these distributions are heavy-tailed
    and several have a median of zero, so it is the more defensible choice when
    the means are not the quantity of interest).
    """
    a = np.asarray(normal_values, dtype=float)
    b = np.asarray(plateau_values, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]

    if a.size < 2 or b.size < 2:
        p = float("nan")
    elif test == "mannwhitney":
        p = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    elif test == "welch":
        p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    else:
        raise ValueError(f"Unknown test {test!r}; expected 'welch' or 'mannwhitney'")

    mean_a, ci_a = _mean_ci95(a)
    mean_b, ci_b = _mean_ci95(b)
    delta = (mean_b / mean_a - 1.0) * 100 if mean_a else float("nan")
    return ModeComparison(
        optimizer=optimizer,
        metric=metric,
        normal_mean=mean_a,
        plateau_mean=mean_b,
        normal_ci=ci_a,
        plateau_ci=ci_b,
        delta_pct=delta,
        p_value=p,
        stars=_p_value_to_stars(p),
        n_normal=int(a.size),
        n_plateau=int(b.size),
    )


def _annotation(comparison: ModeComparison) -> str:
    stars = "" if comparison.stars == "ns" else comparison.stars
    return f"$^{{{stars}}}${comparison.delta_pct:+.1f}%" if stars else (
        f"{comparison.delta_pct:+.1f}%"
    )


def plot_paired_metric_panel(
    ax,
    comparisons: list[ModeComparison],
    *,
    ylabel: str,
    optimizer_order: list[str],
    bar_width: float = 0.38,
    show_ci: bool = True,
) -> None:
    """One panel: two bars per optimizer, annotated with a signed, coloured change."""
    x = np.arange(len(optimizer_order))
    by_opt = {c.optimizer: c for c in comparisons}

    for i, mode in enumerate((MODE_NORMAL, MODE_PLATEAU)):
        means, cis, colors = [], [], []
        for opt in optimizer_order:
            c = by_opt.get(opt)
            means.append(
                np.nan if c is None
                else (c.normal_mean if mode == MODE_NORMAL else c.plateau_mean)
            )
            cis.append(
                0.0 if c is None
                else (c.normal_ci if mode == MODE_NORMAL else c.plateau_ci)
            )
            colors.append(OPTIMIZER_COLORS.get(opt, "0.5"))
        ax.bar(
            x + (i - 0.5) * bar_width,
            means,
            bar_width,
            color=colors,
            alpha=0.85 if mode == MODE_NORMAL else 0.55,
            edgecolor="k",
            linewidth=0.6,
            hatch=MODE_HATCH[mode],
            yerr=cis if show_ci else None,
            error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "0.25"},
        )

    for xi, opt in enumerate(optimizer_order):
        c = by_opt.get(opt)
        if c is None or not np.isfinite(c.delta_pct):
            continue
        top = np.nanmax([c.normal_mean + c.normal_ci, c.plateau_mean + c.plateau_ci])
        ax.annotate(
            _annotation(c),
            xy=(xi, top),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=change_color(c.metric, c.delta_pct),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [OPTIMIZER_LABELS.get(o, o) for o in optimizer_order],
        rotation=22, ha="right", fontsize=9,
    )
    ax.set_ylabel(ylabel)
    ax.margins(y=0.18)
    ax.spines[["top", "right"]].set_visible(False)


def add_mode_legend(
    fig_or_ax,
    *,
    loc: str = "lower center",
    bbox_to_anchor: tuple[float, float] | None = (0.5, -0.03),
    ncol: int = 2,
    fontsize: int = 10,
) -> None:
    """Mode legend, placed outside the panels so it cannot cover an annotation."""
    handles = [
        Patch(facecolor="0.75", edgecolor="k", label="normal"),
        Patch(facecolor="0.75", edgecolor="k", hatch=MODE_HATCH[MODE_PLATEAU],
              label="plateau"),
    ]
    fig_or_ax.legend(
        handles=handles, loc=loc, bbox_to_anchor=bbox_to_anchor,
        ncol=ncol, frameon=False, fontsize=fontsize,
    )


def plot_paired_pf_metric_bars(
    metric_values: dict[tuple[str, str], dict[str, np.ndarray]],
    *,
    panel_specs: list[tuple[str, str]],
    optimizer_order: list[str],
    test: str = "welch",
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> tuple[Figure, list[ModeComparison]]:
    """Grid of paired metric panels, one per entry in `panel_specs`.

    `metric_values` maps `(mode, optimizer) -> {metric: values}`. Significance is
    tested within an optimizer between modes; no cross-optimizer brackets are drawn.
    """
    n = len(panel_specs)
    n_cols = 2 if n > 1 else 1
    n_rows = int(np.ceil(n / n_cols))
    if figsize is None:
        figsize = (7.2 * n_cols, 3.5 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    all_comparisons: list[ModeComparison] = []
    for ax, (metric, ylabel) in zip(axes, panel_specs):
        comparisons = []
        for opt in optimizer_order:
            normal = metric_values.get((MODE_NORMAL, opt), {}).get(metric)
            plateau = metric_values.get((MODE_PLATEAU, opt), {}).get(metric)
            if normal is None or plateau is None:
                continue
            comparisons.append(compare_modes(opt, metric, normal, plateau, test=test))
        plot_paired_metric_panel(
            ax, comparisons, ylabel=ylabel, optimizer_order=optimizer_order
        )
        all_comparisons.extend(comparisons)

    for ax in axes[n:]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title, y=1.01)
    fig.tight_layout()
    add_mode_legend(fig)
    return fig, all_comparisons


def _draw_ratemap_panel(fig, gs, final_ratemap, *, peak_rates, vmax,
                        n_rows: int, n_cols: int):
    """Grid of per-unit fitted fields for one run, sorted by peak rate."""
    if peak_rates is not None:
        order_desc = np.argsort(np.asarray(peak_rates, dtype=np.float64))[::-1]
    else:
        order_desc = np.argsort(np.nanmax(final_ratemap, axis=(1, 2)))[::-1]

    inner = gridspec.GridSpecFromSubplotSpec(
        n_rows, n_cols, subplot_spec=gs, wspace=0.02, hspace=0.02
    )
    im = None
    for rank, unit in enumerate(order_desc[: n_rows * n_cols]):
        r, c = rank // n_cols, rank % n_cols
        ax = fig.add_subplot(inner[r, c])
        im = ax.imshow(final_ratemap[unit], cmap="jet", vmin=0, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
    return im


def plot_final_ratemap_grid(
    meta_by_optimizer: dict[str, object],
    *,
    optimizer_order: list[str],
    vmax: float,
    n_rows: int = N_ROWS,
    n_cols: int = N_COLS,
    title: str = "Final-epoch Gaussian fits of ratemaps, sorted by peak",
    panel_width: float = 6.5,
    panel_height: float = 14.0,
) -> Figure:
    """One column of per-unit fitted fields per optimizer, sharing a colour scale."""
    n_opt = len(optimizer_order)
    fig = plt.figure(figsize=(panel_width * n_opt, panel_height))
    outer = gridspec.GridSpec(
        1, n_opt, figure=fig, wspace=0.05,
        top=0.94, bottom=0.04, left=0.02, right=0.92,
    )
    ims = []
    for col, opt in enumerate(optimizer_order):
        meta = meta_by_optimizer[opt]
        im = _draw_ratemap_panel(
            fig, outer[col], meta.final_ratemap,
            peak_rates=meta.final_peak_rates, vmax=vmax,
            n_rows=n_rows, n_cols=n_cols,
        )
        if im is not None:
            ims.append(im)
    if ims:
        fig.colorbar(ims[0], ax=fig.axes, shrink=0.5, label="Firing rate", pad=0.01)

    fig.canvas.draw()
    cells_per_panel = n_rows * n_cols
    for col, opt in enumerate(optimizer_order):
        top_axes = fig.axes[col * cells_per_panel : col * cells_per_panel + n_cols]
        if not top_axes:
            continue
        x0 = min(ax.get_position().x0 for ax in top_axes)
        x1 = max(ax.get_position().x1 for ax in top_axes)
        y1 = max(ax.get_position().y1 for ax in top_axes)
        fig.text((x0 + x1) / 2, y1 + 0.004, OPTIMIZER_LABELS.get(opt, opt),
                 ha="center", va="bottom", fontsize=12)
    fig.suptitle(title, y=0.98, ha="center")
    return fig
