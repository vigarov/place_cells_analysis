"""Single-room place-field figures for supplementary report notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from report_final.notebook_cache import (
    SINGLE_ROOM_CACHE,
    load_pkl_gz,
    report_final_cache_dir,
    save_pkl_gz,
)
from single_room_multi_opt_compute import (
    MultiOptResults,
    build_optimizer_contexts,
    compute_all_optimizer_results,
)
from single_room_multi_opt_plots import _handle_optimizer_significance, _mean_sem_ci95
from report_final.style import (
    report_font,
    report_grid,
    report_optimizer_colors,
    report_optimizer_labels,
    report_optimizer_order,
    report_optimizer_xtick_labels,
    shared_optimizer_legend,
)


def load_single_room_optimizer_results(
    *,
    project_root: Path,
    config_path: Path,
    results_parent_override: str | None = None,
    dead_threshold_frac: float = 0.4,
    r2_threshold: float = 0.8,
    num_gauss_threshold: float = 1.5,
    pf_tracking_method: str = "revive_mean_displacement",
    constant_l2_similarity: float = 10.0,
    max_factor_pct: float = 0.95,
    peak_analysis_technique: str = "local",
    zoom_bin_width: float = 0.25,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    force_recompute: bool = False,
    **compute_kwargs: Any,
) -> MultiOptResults:
    """Load full single-room optimizer results (same pipeline as the main report).

    Reads/writes ``local/_cache/single_room.pkl.gz`` shared with
    ``REPORT_final_FINAL.ipynb`` when ``use_cache`` is True.
    """
    if cache_dir is None:
        cache_dir = report_final_cache_dir(project_root)
    cache_path = cache_dir / SINGLE_ROOM_CACHE

    if use_cache and not force_recompute and cache_path.is_file():
        cached = load_pkl_gz(cache_path)
        if cached is not None:
            print(f"Loaded single-room results from cache: {cache_path}")
            return cached

    contexts = build_optimizer_contexts(
        project_root=project_root,
        config_path=config_path,
        results_parent_override=results_parent_override,
    )
    results = compute_all_optimizer_results(
        contexts,
        dead_threshold_frac=dead_threshold_frac,
        r2_threshold=r2_threshold,
        num_gauss_threshold=num_gauss_threshold,
        pf_tracking_method=pf_tracking_method,
        zoom_bin_width=zoom_bin_width,
        constant_l2_similarity=constant_l2_similarity,
        max_factor_pct=max_factor_pct,
        peak_analysis_technique=peak_analysis_technique,
        **compute_kwargs,
    )
    if use_cache:
        save_pkl_gz(cache_path, results)
        print(f"Saved single-room results to cache: {cache_path}")
    return results


def _revives_per_pf(values: pd.DataFrame) -> np.ndarray:
    return np.maximum(values["n_life_periods"].astype(int).to_numpy() - 1, 0)


def _per_cell_revives_over_unique_pfs(
    global_pf: pd.DataFrame,
    *,
    n_neurons: int,
) -> np.ndarray:
    """Per-cell total revives divided by that cell's number of distinct PFs."""
    if n_neurons <= 0:
        raise ValueError("n_neurons must be positive")
    if global_pf.empty:
        return np.zeros(n_neurons, dtype=float)

    tmp = global_pf[["cell_idx", "pf_idx", "n_life_periods"]].copy()
    tmp["revives"] = _revives_per_pf(tmp)
    by_cell = tmp.groupby("cell_idx", sort=False).agg(
        total_revives=("revives", "sum"),
        n_unique_pfs=("pf_idx", "nunique"),
    )
    ratio = by_cell["total_revives"] / by_cell["n_unique_pfs"]
    return ratio.reindex(range(n_neurons), fill_value=0.0).to_numpy(dtype=float)


def per_cell_revives_normalized_values_by_optimizer(
    results: MultiOptResults,
) -> dict[str, np.ndarray]:
    """Per-cell revives / unique PFs, keyed by optimizer (Welch t-test input)."""
    values: dict[str, np.ndarray] = {}
    for opt in report_optimizer_order():
        global_pf = results.pf[opt].global_pf_metrics
        n_neurons = results.contexts[opt].n_neurons
        values[opt] = _per_cell_revives_over_unique_pfs(global_pf, n_neurons=n_neurons)
    return values


def revives_per_cell_normalized_table(results: MultiOptResults) -> pd.DataFrame:
    """Mean over cells of (total revives / distinct PFs for that cell)."""
    rows: list[dict[str, float | int | str]] = []
    for opt in report_optimizer_order():
        global_pf = results.pf[opt].global_pf_metrics
        n_neurons = results.contexts[opt].n_neurons
        per_cell = _per_cell_revives_over_unique_pfs(global_pf, n_neurons=n_neurons)
        mean, sem, _ci95 = _mean_sem_ci95(per_cell)
        rows.append(
            {
                "optimizer": opt,
                "mean_revives_over_unique_pfs_per_cell": mean,
                "sem_revives_over_unique_pfs_per_cell": sem,
                "n_neurons": int(n_neurons),
            }
        )
    return pd.DataFrame(rows)


def revives_per_pf_normalized_table(results: MultiOptResults) -> pd.DataFrame:
    """Mean revives per PF divided by the number of distinct tracked place fields."""
    rows: list[dict[str, float | int | str]] = []
    for opt in report_optimizer_order():
        global_pf = results.pf[opt].global_pf_metrics
        revives = _revives_per_pf(global_pf)
        n_distinct = int(len(global_pf))
        mean_revives = float(np.mean(revives)) if n_distinct else float("nan")
        rows.append(
            {
                "optimizer": opt,
                "mean_revives_per_pf": mean_revives,
                "n_distinct_place_fields": n_distinct,
                "mean_revives_per_pf_over_n_distinct": (
                    mean_revives / n_distinct if n_distinct else float("nan")
                ),
                "total_revives": int(revives.sum()),
            }
        )
    return pd.DataFrame(rows)


def _values_by_optimizer(table: pd.DataFrame, column: str) -> np.ndarray:
    order = report_optimizer_order()
    return (
        table.set_index("optimizer")
        .reindex(order)[column]
        .to_numpy(dtype=float)
    )


def _style_axis(ax, *, xlabel: str, ylabel: str, title: str | None = None) -> None:
    f = report_font()
    ax.set_xlabel(xlabel, fontsize=f["axis_label"])
    ax.set_ylabel(ylabel, fontsize=f["axis_label"])
    if title:
        ax.set_title(title, fontsize=f["panel_title"])
    ax.tick_params(labelsize=f["tick"])
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which="major", **report_grid())
    ax.xaxis.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_revives_per_pf_normalized_by_optimizer(
    table: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (8.5, 5.0),
    annotate: bool = True,
) -> Figure:
    """Bar chart of mean revives per PF / number of distinct place fields."""
    order = report_optimizer_order()
    colors = report_optimizer_colors()
    labels = report_optimizer_xtick_labels()
    f = report_font()
    x = np.arange(len(order), dtype=float)

    values = _values_by_optimizer(table, "mean_revives_per_pf_over_n_distinct")

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        x,
        values,
        width=0.62,
        color=[colors[o] for o in order],
        edgecolor="k",
        linewidth=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([labels[o] for o in order])
    y_hi = float(np.nanmax(values))
    ax.set_ylim(0.0, y_hi + max(0.04 * y_hi, 0.001))
    _style_axis(
        ax,
        xlabel="",
        ylabel="Mean revives per PF / distinct PFs",
        title="Revives normalized by place-field catalog size",
    )
    if annotate:
        for xi, val in zip(x, values, strict=True):
            ax.annotate(
                f"{val:.4f}",
                (xi, val),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=f["annotation"],
            )
    fig.tight_layout()
    return fig


def plot_revives_per_cell_normalized_by_optimizer(
    table: pd.DataFrame,
    *,
    values_by_optimizer: dict[str, np.ndarray] | None = None,
    figsize: tuple[float, float] = (8.5, 5.0),
    annotate: bool = True,
    show_significance: bool = True,
    show_insignificance: bool = False,
    bracket_step_mult: float = 1.65,
) -> Figure:
    """Bar chart of mean cell-level revives / unique PFs."""
    order = report_optimizer_order()
    colors = report_optimizer_colors()
    labels = report_optimizer_xtick_labels()
    f = report_font()
    x = np.arange(len(order), dtype=float)

    values = _values_by_optimizer(table, "mean_revives_over_unique_pfs_per_cell")
    if "sem_revives_over_unique_pfs_per_cell" in table.columns:
        sems = _values_by_optimizer(table, "sem_revives_over_unique_pfs_per_cell")
    else:
        sems = np.zeros_like(values)
    cis = 1.96 * sems

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        x,
        values,
        width=0.62,
        color=[colors[o] for o in order],
        edgecolor="k",
        linewidth=0.8,
    )
    ax.errorbar(
        x,
        values,
        yerr=cis,
        fmt="none",
        ecolor="k",
        capsize=4,
        linewidth=1.2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([labels[o] for o in order])
    y_hi = float(np.nanmax(values + cis))
    ax.set_ylim(0.0, y_hi + max(0.04 * y_hi, 0.02))
    _style_axis(
        ax,
        xlabel="",
        ylabel="Mean (revives / unique PFs) per cell",
        title="Normalized per-cell revives",
    )
    if annotate:
        for xi, val, ci in zip(x, values, cis, strict=True):
            ax.annotate(
                f"{val:.2f}",
                (xi, val + ci),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=f["annotation"],
            )
    if values_by_optimizer is not None:
        positions = {opt: float(i) for i, opt in enumerate(order)}
        bar_tops = values + cis
        _handle_optimizer_significance(
            ax,
            values_by_optimizer,
            positions=positions,
            bar_tops=bar_tops,
            panel_label="Revives per cell (normalized)",
            show_significance=show_significance,
            show_insignificance=show_insignificance,
            bracket_step_mult=bracket_step_mult,
        )
    fig.tight_layout()
    return fig


def _life_period_segment_counts(
    periods: pd.DataFrame,
    *,
    segment_duration_s: float,
) -> np.ndarray:
    if "n_segments" in periods.columns:
        return periods["n_segments"].to_numpy(dtype=float)
    return periods["length_s"].to_numpy(dtype=float) / segment_duration_s


def conditional_spontaneity_table(
    results: MultiOptResults,
    *,
    segment_duration_s: float | None = None,
    min_segments: range | tuple[int, ...] | None = None,
    min_periods: int = 30,
) -> pd.DataFrame:
    """Spontaneous fraction conditioned on field lifetime, by optimizer.

    For each survival threshold *k* (segments), only life periods with
    ``n_segments >= k`` enter the mean of the ``one_segment`` flag. Thresholds
    are also reported in seconds via ``segment_duration_s``.
    """
    if segment_duration_s is None:
        segment_duration_s = next(iter(results.contexts.values())).segment_duration_s
    if min_segments is None:
        min_segments = range(1, 51)

    rows: list[dict[str, float | int | str]] = []
    for opt in report_optimizer_order():
        periods = results.pf[opt].life_period_metrics
        if periods.empty or "one_segment" not in periods.columns:
            continue
        spont = periods["one_segment"].astype(float).to_numpy()
        n_seg = _life_period_segment_counts(
            periods,
            segment_duration_s=segment_duration_s,
        )
        for k in min_segments:
            keep = n_seg >= float(k)
            n_keep = int(keep.sum())
            rows.append(
                {
                    "optimizer": opt,
                    "min_segments": int(k),
                    "min_survival_s": float(k) * segment_duration_s,
                    "spontaneous_frac": (
                        float(spont[keep].mean()) if n_keep >= min_periods else np.nan
                    ),
                    "n_periods": n_keep,
                }
            )
    return pd.DataFrame(rows)


def plot_conditional_spontaneity_vs_survival(
    table: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7.2, 4.6),
    title: str | None = None,
    x_tick_step_s: float | None = None,
    show_legend: bool = True,
    marker_every: int = 5,
) -> Figure:
    """Line plot: conditional spontaneous fraction vs minimum field survival."""
    order = report_optimizer_order()
    colors = report_optimizer_colors()
    labels = report_optimizer_labels()

    fig, ax = plt.subplots(figsize=figsize)
    for opt in order:
        part = table.loc[table["optimizer"] == opt].sort_values("min_survival_s")
        if part.empty:
            continue
        x = part["min_survival_s"].to_numpy(dtype=float)
        y = part["spontaneous_frac"].to_numpy(dtype=float)
        markevery = max(1, marker_every) if marker_every else None
        ax.plot(
            x,
            y,
            "-o",
            color=colors[opt],
            label=labels[opt],
            linewidth=2.0,
            markersize=4.5,
            markerfacecolor=colors[opt],
            markeredgecolor="0.15",
            markeredgewidth=0.5,
            markevery=markevery,
        )

    x_vals = table["min_survival_s"].to_numpy(dtype=float)
    x_lo = float(np.nanmin(x_vals))
    x_hi = float(np.nanmax(x_vals))
    if x_tick_step_s is None:
        x_tick_step_s = 50.0 if (x_hi - x_lo) > 120.0 else 20.0
    tick_lo = np.ceil(x_lo / x_tick_step_s) * x_tick_step_s
    ax.set_xticks(np.arange(tick_lo, x_hi + 0.5 * x_tick_step_s, x_tick_step_s))
    ax.set_xlim(x_lo - 0.02 * (x_hi - x_lo), x_hi + 0.02 * (x_hi - x_lo))

    y_vals = table["spontaneous_frac"].to_numpy(dtype=float)
    y_lo = 0.0
    y_hi = float(np.nanmax(y_vals)) if np.isfinite(y_vals).any() else 1.0
    ax.set_ylim(y_lo, y_hi + max(0.06 * y_hi, 0.04))

    _style_axis(
        ax,
        xlabel="Field survived ≥ (s)",
        ylabel="Spontaneous fraction",
        title=title,
    )
    if show_legend:
        shared_optimizer_legend(fig, optimizers=order, ncol=len(order), y=-0.14)
        fig.subplots_adjust(bottom=0.22)
    fig.tight_layout()
    return fig
