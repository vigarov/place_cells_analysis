"""Single-room ratemap summaries for supplementary report figures.

Prefer ``final_fields_from_results`` / ``max_signal_table_from_results`` on a
``MultiOptResults`` object from ``compute_all_optimizer_results`` (main-report
pipeline). The ``load_final_*`` helpers reload NPZs and are memory-heavy.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from tqdm.auto import tqdm

from experiments.common.paths import gaussian_rf_fits_path
from experiments.common.ratemaps_io import (
    discover_ratemap_captures,
    load_ratemap_capture,
    load_trajectory_capture_tags,
)
from experiments.common.run import create_experiment, load_experiment_config
from optimizers.defaults import build_optimizer_config
from single_room_multi_opt_compute import (
    OPTIMIZER_ORDER,
    MultiOptResults,
    apply_results_parent_override,
    load_gaussian_capture,
)
from report_final.style import (
    report_font,
    report_grid,
    report_optimizer_colors,
    report_optimizer_labels,
    report_optimizer_order,
    report_optimizer_xtick_labels,
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


def final_fields_from_results(
    results: MultiOptResults,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Final Gaussian ratemaps and peak rates already attached to ``results.meta``."""
    ratemaps: dict[str, np.ndarray] = {}
    peak_rates: dict[str, np.ndarray] = {}
    for opt in report_optimizer_order():
        if opt not in results.meta:
            continue
        meta = results.meta[opt]
        ratemaps[opt] = meta.final_ratemap
        peak_rates[opt] = meta.final_peak_rates
    return ratemaps, peak_rates


def max_signal_table_from_results(results: MultiOptResults) -> pd.DataFrame:
    """Per-cell Gaussian ``signal_max`` at the final capture (main-report pipeline)."""
    rows: list[dict[str, object]] = []
    for opt in report_optimizer_order():
        if opt not in results.meta:
            continue
        peaks = results.meta[opt].final_peak_rates
        for cell_idx, value in enumerate(peaks):
            rows.append(
                {
                    "optimizer": opt,
                    "cell_idx": int(cell_idx),
                    "max_signal": float(value),
                }
            )
    return pd.DataFrame(rows)


def _load_optimizer_final_ratemap(
    *,
    run_config,
    optimizer_type: str,
    project_root: Path,
    results_parent_override: str | None,
) -> tuple[np.ndarray, dict[str, object]]:
    if optimizer_type not in run_config.optimizers:
        raise ValueError(
            f"Optimizer {optimizer_type!r} not in config optimizers "
            f"{run_config.optimizers}"
        )
    optimizer_config = build_optimizer_config(optimizer_type)
    experiment = create_experiment(run_config, optimizer_config=optimizer_config)
    paths = experiment.resolve_paths()
    paths = apply_results_parent_override(
        paths,
        experiment_name=experiment.name,
        project_root=project_root,
        results_parent_override=results_parent_override,
    )
    if not paths.ratemaps_dir.is_dir():
        raise FileNotFoundError(
            f"Missing ratemaps directory for {optimizer_type}: {paths.ratemaps_dir}"
        )

    captures = discover_ratemap_captures(
        paths.ratemaps_dir,
        experiment_name=experiment.name,
    )
    if not captures:
        raise FileNotFoundError(
            f"No ratemap captures found for {optimizer_type} under {paths.ratemaps_dir}"
        )
    last_capture = captures[-1]
    tags = load_trajectory_capture_tags(last_capture.path)
    capture_tag = tags[last_capture.capture_idx]
    ratemap = load_ratemap_capture(last_capture)
    meta = {
        "capture_path": str(last_capture.path),
        "capture_idx": int(last_capture.capture_idx),
        "capture_tag": capture_tag,
    }
    return ratemap, meta


def _load_optimizer_final_gaussian_ratemap(
    *,
    run_config,
    optimizer_type: str,
    project_root: Path,
    results_parent_override: str | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if optimizer_type not in run_config.optimizers:
        raise ValueError(
            f"Optimizer {optimizer_type!r} not in config optimizers "
            f"{run_config.optimizers}"
        )
    optimizer_config = build_optimizer_config(optimizer_type)
    experiment = create_experiment(run_config, optimizer_config=optimizer_config)
    paths = experiment.resolve_paths()
    paths = apply_results_parent_override(
        paths,
        experiment_name=experiment.name,
        project_root=project_root,
        results_parent_override=results_parent_override,
    )
    if not paths.ratemaps_dir.is_dir():
        raise FileNotFoundError(
            f"Missing ratemaps directory for {optimizer_type}: {paths.ratemaps_dir}"
        )

    fits_path = gaussian_rf_fits_path(paths.results_dir, experiment.name, room_idx=0)
    if not fits_path.is_file():
        raise FileNotFoundError(
            f"Missing Gaussian fits for {optimizer_type}: {fits_path}. "
            "Run: uv run estimate-gaussians-rf --input <results_dir>"
        )

    captures = discover_ratemap_captures(
        paths.ratemaps_dir,
        experiment_name=experiment.name,
    )
    if not captures:
        raise FileNotFoundError(
            f"No ratemap captures found for {optimizer_type} under {paths.ratemaps_dir}"
        )
    last_capture = captures[-1]
    tags = load_trajectory_capture_tags(last_capture.path)
    capture_tag = tags[last_capture.capture_idx]
    final_capture_idx = len(captures) - 1

    with np.load(fits_path) as fits:
        gaussian_params = np.asarray(fits["gaussian_params"])
        signal_max = np.asarray(fits["signal_max"], dtype=np.float64)
    if gaussian_params.shape[0] != len(captures):
        raise ValueError(
            f"{optimizer_type}: Gaussian fits have {gaussian_params.shape[0]} captures "
            f"but found {len(captures)} ratemap captures"
        )

    sample_rm = load_ratemap_capture(captures[0])
    field_shape = (int(sample_rm.shape[1]), int(sample_rm.shape[2]))
    ratemap = load_gaussian_capture(
        final_capture_idx,
        gaussian_params=gaussian_params,
        field_shape=field_shape,
    )
    peak_rates = signal_max[final_capture_idx]
    meta = {
        "capture_path": str(last_capture.path),
        "capture_idx": int(last_capture.capture_idx),
        "capture_tag": capture_tag,
        "fits_path": str(fits_path),
    }
    return ratemap, peak_rates, meta


def load_final_gaussian_ratemaps_by_optimizer(
    *,
    project_root: Path,
    config_path: Path,
    results_parent_override: str | None = None,
    optimizer_order: tuple[str, ...] = tuple(OPTIMIZER_ORDER),
    show_progress: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Rendered Gaussian RF fits at the final training capture."""
    run_config = load_experiment_config(config_path)
    ratemaps: dict[str, np.ndarray] = {}
    peak_rates: dict[str, np.ndarray] = {}
    opt_iter = tqdm(
        optimizer_order,
        desc="Loading final single-room Gaussian fits",
        disable=not show_progress,
    )
    for optimizer_type in opt_iter:
        opt_iter.set_postfix(optimizer=optimizer_type)
        ratemap, peaks, _ = _load_optimizer_final_gaussian_ratemap(
            run_config=run_config,
            optimizer_type=optimizer_type,
            project_root=project_root,
            results_parent_override=results_parent_override,
        )
        ratemaps[optimizer_type] = ratemap
        peak_rates[optimizer_type] = peaks
    return ratemaps, peak_rates


def load_final_ratemaps_by_optimizer(
    *,
    project_root: Path,
    config_path: Path,
    results_parent_override: str | None = None,
    optimizer_order: tuple[str, ...] = tuple(OPTIMIZER_ORDER),
    show_progress: bool = True,
) -> dict[str, np.ndarray]:
    """Final training-capture ratemaps keyed by optimizer."""
    run_config = load_experiment_config(config_path)
    ratemaps: dict[str, np.ndarray] = {}
    opt_iter = tqdm(
        optimizer_order,
        desc="Loading final single-room ratemaps",
        disable=not show_progress,
    )
    for optimizer_type in opt_iter:
        opt_iter.set_postfix(optimizer=optimizer_type)
        ratemap, _ = _load_optimizer_final_ratemap(
            run_config=run_config,
            optimizer_type=optimizer_type,
            project_root=project_root,
            results_parent_override=results_parent_override,
        )
        ratemaps[optimizer_type] = ratemap
    return ratemaps


def max_signal_table_from_ratemaps(
    ratemaps: dict[str, np.ndarray],
    *,
    capture_meta: dict[str, dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Long-form peak signal table from loaded final ratemaps."""
    rows: list[dict[str, object]] = []
    for optimizer_type, ratemap in ratemaps.items():
        max_signal = np.nanmax(ratemap, axis=(1, 2))
        meta = (capture_meta or {}).get(optimizer_type, {})
        for cell_idx, value in enumerate(max_signal):
            rows.append(
                {
                    "optimizer": optimizer_type,
                    "cell_idx": int(cell_idx),
                    "max_signal": float(value),
                    **meta,
                }
            )
    return pd.DataFrame(rows)


def load_final_max_signal_by_cell(
    *,
    project_root: Path,
    config_path: Path,
    results_parent_override: str | None = None,
    optimizer_order: tuple[str, ...] = tuple(OPTIMIZER_ORDER),
    show_progress: bool = True,
) -> pd.DataFrame:
    """Peak ratemap signal per cell at the last training capture for each optimizer."""
    run_config = load_experiment_config(config_path)
    rows: list[dict[str, object]] = []
    opt_iter = tqdm(
        optimizer_order,
        desc="Loading final single-room ratemaps",
        disable=not show_progress,
    )
    for optimizer_type in opt_iter:
        opt_iter.set_postfix(optimizer=optimizer_type)
        ratemap, meta = _load_optimizer_final_ratemap(
            run_config=run_config,
            optimizer_type=optimizer_type,
            project_root=project_root,
            results_parent_override=results_parent_override,
        )
        max_signal = np.nanmax(ratemap, axis=(1, 2))
        for cell_idx, value in enumerate(max_signal):
            rows.append(
                {
                    "optimizer": optimizer_type,
                    "cell_idx": int(cell_idx),
                    "max_signal": float(value),
                    **meta,
                }
            )
    return pd.DataFrame(rows)


def _activity_rank_order(
    ratemap: np.ndarray,
    *,
    peak_rates: np.ndarray | None = None,
) -> np.ndarray:
    if peak_rates is not None:
        return np.argsort(peak_rates)[::-1]
    peak = np.nanmax(ratemap, axis=(1, 2))
    return np.argsort(peak)[::-1]


def activity_rank_cell_table(
    ratemaps: dict[str, np.ndarray],
    *,
    ranks: tuple[int, ...] = (0, 7),
    peak_rates: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Map activity rank to cell index for each optimizer."""
    rows: list[dict[str, int | float | str]] = []
    for opt in report_optimizer_order():
        if opt not in ratemaps:
            continue
        peaks = None if peak_rates is None else peak_rates[opt]
        order = _activity_rank_order(ratemaps[opt], peak_rates=peaks)
        if peaks is None:
            peaks = np.nanmax(ratemaps[opt], axis=(1, 2))
        for rank in ranks:
            if rank < 0 or rank >= len(order):
                raise IndexError(
                    f"Activity rank {rank} out of range for {opt!r} ({len(order)} cells)"
                )
            cell_idx = int(order[rank])
            rows.append(
                {
                    "optimizer": opt,
                    "activity_rank": int(rank),
                    "cell_idx": cell_idx,
                    "max_signal": float(peaks[cell_idx]),
                }
            )
    return pd.DataFrame(rows)


def _shared_ratemap_vmax(ratemaps: dict[str, np.ndarray], *, pct: float = 99.9) -> float:
    pooled = np.concatenate(
        [np.nanmax(ratemap, axis=(1, 2)).ravel() for ratemap in ratemaps.values()]
    )
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return 1.0
    return float(np.nanpercentile(pooled, pct))


def plot_final_ratemap_activity_ranks(
    ratemaps: dict[str, np.ndarray],
    *,
    ranks: tuple[int, ...] = (0, 7),
    peak_rates: dict[str, np.ndarray] | None = None,
    figsize: tuple[float, float] | None = None,
    cmap: str = "jet",
    vmax: float | None = None,
    vmax_pct: float = 99.9,
    field_label: str = "ratemap",
    title: str | None = None,
) -> Figure:
    """Mosaic of final fields for fixed activity ranks, one column per optimizer."""
    order = report_optimizer_order()
    labels = report_optimizer_labels()
    f = report_font()
    n_rows = len(ranks)
    n_cols = len(order)
    if figsize is None:
        figsize = (2.4 * n_cols, 2.2 * n_rows)

    if vmax is None:
        if peak_rates is not None:
            pooled = np.concatenate(
                [peaks.ravel() for peaks in peak_rates.values() if peaks is not None]
            )
            pooled = pooled[np.isfinite(pooled)]
            vmax = float(np.nanpercentile(pooled, vmax_pct)) if pooled.size else 1.0
        else:
            vmax = _shared_ratemap_vmax(ratemaps, pct=vmax_pct)

    rank_labels = {
        0: "Most active",
        200: "200th",
    }

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
        layout="constrained",
    )
    im = None
    for col, opt in enumerate(order):
        ratemap = ratemaps[opt]
        peaks = None if peak_rates is None else peak_rates[opt]
        activity_order = _activity_rank_order(ratemap, peak_rates=peaks)
        for row, rank in enumerate(ranks):
            ax = axes[row, col]
            cell_idx = int(activity_order[rank])
            im = ax.imshow(ratemap[cell_idx], cmap=cmap, vmin=0.0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(
                    rank_labels.get(rank, f"Rank {rank}"),
                    fontsize=f["axis_label"],
                )
        axes[n_rows - 1, col].set_xlabel(
            labels[opt],
            fontsize=f["panel_title"]+2,
            labelpad=8,
        )
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
        cbar.set_label("Firing rate", fontsize=f["colorbar_label"])
        cbar.ax.tick_params(labelsize=f["colorbar_tick"])
    if title is None:
        title = f"Final training {field_label}s by activity rank (shared vmax={vmax:.2f})"
    fig.suptitle(title, fontsize=f["suptitle"])
    return fig


def final_max_signal_quantile_table(
    table: pd.DataFrame,
    *,
    q: float = 0.995,
) -> pd.DataFrame:
    """Per-optimizer quantile of the per-cell peak ratemap signal."""
    order = report_optimizer_order()
    rows = []
    for opt in order:
        vals = table.loc[table["optimizer"] == opt, "max_signal"].to_numpy(dtype=float)
        rows.append(
            {
                "optimizer": opt,
                "quantile": float(q),
                "max_signal_quantile": float(np.nanquantile(vals, q)),
            }
        )
    return pd.DataFrame(rows)


def plot_final_max_signal_boxplot(
    table: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (14.0, 5.0),
    title: str = "Final training segment",
    quantile: float = 0.995,
    annotate_bars: bool = True,
) -> Figure:
    """Per-cell boxplots plus a bar chart of the per-optimizer quantile."""
    order = report_optimizer_order()
    colors = report_optimizer_colors()
    labels = report_optimizer_xtick_labels()
    x = np.arange(len(order), dtype=float)
    f = report_font()

    data = [
        table.loc[table["optimizer"] == opt, "max_signal"].to_numpy(dtype=float)
        for opt in order
    ]
    quantiles = final_max_signal_quantile_table(table, q=quantile)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    bp = ax.boxplot(
        data,
        positions=x,
        widths=0.62,
        patch_artist=True,
        flierprops={
            "marker": "o",
            "markersize": 3,
            "alpha": 0.35,
            "markerfacecolor": "0.45",
            "markeredgecolor": "0.45",
            "linestyle": "none",
        },
    )
    for patch, opt in zip(bp["boxes"], order, strict=True):
        patch.set_facecolor(colors[opt])
        patch.set_alpha(0.75)
        patch.set_edgecolor("k")
        patch.set_linewidth(0.8)
    for median in bp["medians"]:
        median.set_color("k")
        median.set_linewidth(1.5)
    for whisker in bp["whiskers"]:
        whisker.set_color("0.25")
        whisker.set_linewidth(1.0)
    for cap in bp["caps"]:
        cap.set_color("0.25")
        cap.set_linewidth(1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([labels[opt] for opt in order])
    _style_axis(
        ax,
        xlabel="",
        ylabel="Max signal (ratemap peak)",
        title="Distribution across cells",
    )

    ax = axes[1]
    q_values = quantiles["max_signal_quantile"].to_numpy(dtype=float)
    ax.bar(
        x,
        q_values,
        width=0.62,
        color=[colors[o] for o in order],
        edgecolor="k",
        linewidth=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([labels[opt] for opt in order])
    q_pct = int(round(100 * quantile))
    _style_axis(
        ax,
        xlabel="",
        ylabel=f"{q_pct}th percentile",
        title=f"{q_pct}th percentile across cells",
    )
    y_hi = float(np.max(q_values))
    ax.set_ylim(0.0, y_hi + max(0.04 * y_hi, 0.02))
    if annotate_bars:
        for xi, val in zip(x, q_values, strict=True):
            ax.annotate(
                f"{val:.2f}",
                (xi, val),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=f["annotation"],
            )

    if title:
        fig.suptitle(title, fontsize=f["suptitle"], y=1.02)
    fig.tight_layout()
    return fig
