"""Plot helpers for `analyse_cycles.ipynb` (Gaussian RF evolution)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cycles.gaussian_fit_cycle_analysis import (
    DEFAULT_GAUSSIAN_FIELD_SHAPE,
    GAUSSIAN_PARAM_NAMES,
    AicR2Correlation,
    OutlierPoint,
    build_truncated_segments,
    cell_room_pairs_with_nan_r2,
    collect_ratemap_requests,
    fit_dict_from_visit_row,
    gaussian_fields_from_params,
    load_ratemaps_by_visit,
    nan_r2_affected_counts,
    n_rooms_from_truncated_file,
    params_array_from_row,
    select_nan_r2_pairs_cyclical,
    visits_for_cell_in_room,
)
from analysis.cell_evolution_plots import _format_fit_metrics


def save_figure(
    fig: plt.Figure,
    path: Path | str,
    *,
    dpi: int = 150,
    close: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path


def plot_aic_r2_correlation(
    stats: AicR2Correlation,
    *,
    r2_threshold: float | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (6, 5),
) -> plt.Figure:
    """Scatter AIC vs `r2` with a linear fit and annotated coefficients."""
    r2, aic = stats.r2, stats.aic
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if len(r2) == 0:
        ax.set_xlabel(r"$R^2$")
        ax.set_ylabel("AIC")
        ax.set_title("AIC vs $R^2$ (no finite fits)")
        fig.tight_layout()
        return fig

    ax.scatter(r2, aic, s=8, alpha=0.35, edgecolors="none")
    if len(r2) >= 2 and np.isfinite(stats.slope):
        x_line = np.linspace(float(r2.min()), float(r2.max()), 100)
        ax.plot(
            x_line,
            stats.slope * x_line + stats.intercept,
            color="C1",
            lw=2,
        )
    ax.set_xlabel(r"$R^2$")
    ax.set_ylabel("AIC")
    title = f"AIC vs $R^2$ ({len(r2):,} fits)"
    if r2_threshold is not None:
        title += f", $R^2 > {r2_threshold:g}$"
    ax.set_title(title)
    if np.isfinite(stats.correlation):
        ax.text(
            0.05,
            0.95,
            r"$\rho"+f" = {stats.correlation:.3f}$\n"
            f"slope = {stats.slope:.2f}\n"
            f"intercept = {stats.intercept:.1f}",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )
    fig.tight_layout()
    return fig


def _gaussian_evolution_cmap() -> plt.Colormap:
    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad(color="0.92")
    return cmap


def plot_gaussian_evolution_for_room(
    visits: pd.DataFrame,
    *,
    cell_idx: int,
    room_id: int,
    n_gaussians: int = 2,
    param_names: tuple[str, ...] = GAUSSIAN_PARAM_NAMES,
    field_shape: tuple[int, int] = (110, 110),
    im_width: float = 2,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """
    Plot Gaussian components and their sum across visits to one room.

    Rows: `g1`, `g2`, … then `sum`. Columns: cycles (visits) in order.
    Visits with failed fits (`NaN` params) show a grey panel.
    """
    if visits.empty:
        return None

    n_visits = len(visits)
    n_rows = n_gaussians + 1
    row_labels = [f"g{k + 1}" for k in range(n_gaussians)] + ["sum"]

    if ax is None:
        fig, axes = plt.subplots(
            n_rows,
            n_visits,
            figsize=(im_width * n_visits + 0.8, im_width * n_rows + 0.6),
            squeeze=False,
        )
    else:
        fig = ax.figure
        axes = np.array([[ax]])

    rendered: list[tuple[np.ndarray, np.ndarray] | None] = []
    for _, row in visits.iterrows():
        params = params_array_from_row(row, n_gaussians, param_names)
        rendered.append(gaussian_fields_from_params(params, field_shape=field_shape))

    finite_vals = [
        arr.ravel()
        for fit in rendered
        if fit is not None
        for arr in fit
    ]
    if finite_vals:
        stack = np.concatenate(finite_vals)
        vmin = float(np.nanmin(stack))
        vmax = float(np.nanmax(stack))
    else:
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        vmax = vmin + 1e-6

    cmap = _gaussian_evolution_cmap()
    im = None
    for col, (_, row) in enumerate(visits.iterrows()):
        cycle_id = int(row["cycle_id"])
        fit = rendered[col]
        r2 = row["r2"]
        r2_label = "NaN" if pd.isna(r2) else f"{r2:.2f}"
        aic = row["aic"]
        aic_label = "NaN" if pd.isna(aic) else f"{aic:.0f}"
        col_title = f"C{cycle_id + 1}\n$R^2$={r2_label}\nAIC={aic_label}"

        for row_idx in range(n_gaussians):
            ax_ij = axes[row_idx, col]
            if fit is None:
                ax_ij.imshow(
                    np.full(field_shape, np.nan),
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                if row_idx == 0:
                    ax_ij.set_title(col_title, fontsize=7)
            else:
                components, _ = fit
                im = ax_ij.imshow(components[row_idx], cmap=cmap, vmin=vmin, vmax=vmax)
                if row_idx == 0:
                    ax_ij.set_title(col_title, fontsize=7)
            ax_ij.set_xticks([])
            ax_ij.set_yticks([])
            if col == 0:
                ax_ij.set_ylabel(row_labels[row_idx], fontsize=8)

        ax_sum = axes[n_rows - 1, col]
        if fit is None:
            ax_sum.imshow(
                np.full(field_shape, np.nan),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax_sum.text(
                0.5,
                0.5,
                "fit failed",
                transform=ax_sum.transAxes,
                ha="center",
                va="center",
                fontsize=7,
                color="0.35",
            )
        else:
            _, sum_field = fit
            im = ax_sum.imshow(sum_field, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_sum.set_xticks([])
        ax_sum.set_yticks([])
        if col == 0:
            ax_sum.set_ylabel("sum", fontsize=8)

    fig.suptitle(
        f"Cell {cell_idx}, Room {room_id}: Gaussian evolution ({n_visits} visits)",
        fontsize=11,
        y=1.02,
    )
    if im is not None:
        fig.subplots_adjust(right=0.88, top=0.86)
        fig.colorbar(
            im,
            ax=axes.ravel().tolist(),
            shrink=0.55,
            label="Amplitude",
            pad=0.02,
        )
    else:
        fig.tight_layout()
    return fig


def plot_nan_r2_gaussian_evolutions(
    df: pd.DataFrame,
    *,
    signal_max_threshold: float,
    n_gaussians: int = 2,
    param_names: tuple[str, ...] = GAUSSIAN_PARAM_NAMES,
    field_shape: tuple[int, int] = (110, 110),
    max_plots: int | None = 12,
    im_width: float = 2.0,
) -> list[tuple[int, int, plt.Figure]]:
    """
    Plot Gaussian evolution for cells with failed fits.

    Returns a list of `(cell_idx, room_id, figure)` for each panel drawn.
    """
    pairs = cell_room_pairs_with_nan_r2(
        df,
        signal_max_threshold=signal_max_threshold,
    )
    if pairs.empty:
        print(
            f"No (cell, room) pairs with signal_max > {signal_max_threshold} "
            "and NaN r2."
        )
        return []

    n_cells, n_rooms, n_pairs = nan_r2_affected_counts(pairs)
    print(
        f"NaN r2: {n_cells} cells, {n_rooms} rooms, {n_pairs} (cell, room) pairs"
    )

    pairs = select_nan_r2_pairs_cyclical(pairs, max_plots)
    if max_plots is not None:
        print(f"Plotting {len(pairs)} of {n_pairs} pairs (cyclical, max_plots={max_plots})")

    figures: list[tuple[int, int, plt.Figure]] = []
    for _, pair in pairs.iterrows():
        cell_idx = int(pair["cell_idx"])
        room_id = int(pair["room_id"])
        visits = visits_for_cell_in_room(df, cell_idx, room_id)
        fig = plot_gaussian_evolution_for_room(
            visits,
            cell_idx=cell_idx,
            room_id=room_id,
            n_gaussians=n_gaussians,
            param_names=param_names,
            field_shape=field_shape,
            im_width=im_width,
        )
        if fig is not None:
            figures.append((cell_idx, room_id, fig))
    return figures


def _outlier_highlight_color(point: OutlierPoint) -> str:
    return "green" if point.r2_band == "top" else "red"


def _outlier_linestyle(point: OutlierPoint) -> str:
    return "-" if point.aic_extremum == "high" else "--"


def _highlight_column_spines(axes: np.ndarray, col: int, color: str) -> None:
    for row in range(axes.shape[0]):
        ax = axes[row, col]
        if not ax.images and not ax.texts and not ax.patches:
            continue
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)


def _outlier_decomposition_row_layout(
    n_gaussians: int,
) -> list[tuple[str, str]]:
    """
    Vertical decomposition layout: gt, sum, `=`, g1, `+`, g2, …

    Returns `(kind, key)` pairs with `kind` `'image'` or `'op'`.
    """
    rows: list[tuple[str, str]] = [("image", "gt"), ("image", "sum"), ("op", "=")]
    for k in range(n_gaussians):
        if k > 0:
            rows.append(("op", "+"))
        rows.append(("image", f"g{k + 1}"))
    return rows


def _align_decomposition_op_axes(
    axes: np.ndarray,
    row_layout: list[tuple[str, str]],
) -> None:
    """Match `=` / `+` row horizontal extent to image panels in each column."""
    image_rows = [i for i, (kind, _) in enumerate(row_layout) if kind == "image"]
    if not image_rows:
        return
    ref_row = image_rows[0]
    for col in range(axes.shape[1]):
        ref = axes[ref_row, col].get_position()
        for row_idx, (kind, _) in enumerate(row_layout):
            if kind != "op":
                continue
            op_ax = axes[row_idx, col]
            op_pos = op_ax.get_position()
            op_ax.set_position([ref.x0, op_pos.y0, ref.width, op_pos.height])


def _ordered_gaussian_components(
    fit: dict,
) -> tuple[list[np.ndarray], list[dict]]:
    """Sort components by peak amplitude (largest first)."""
    components = fit["components"]
    component_params = fit["component_params"]
    if len(components) <= 1:
        return components, component_params
    order = sorted(
        range(len(components)),
        key=lambda k: np.nanmax(components[k]),
        reverse=True,
    )
    return [components[k] for k in order], [component_params[k] for k in order]


def plot_outlier_metric_evolution(
    point: OutlierPoint,
    visits: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (7, 4),
) -> plt.Figure:
    """
    Plot AIC evolution across room visits for one outlier point.

    Color: green = top R² tail, red = bottom R² tail.
    Linestyle: solid = highest AIC in tail, dashed = lowest AIC in tail.
    """
    cycles = visits["cycle_id"].to_numpy(dtype=int)
    values = visits["aic"].to_numpy(dtype=np.float64)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    line_color = _outlier_highlight_color(point)
    ax.plot(
        cycles,
        values,
        marker="o",
        markersize=4,
        linestyle=_outlier_linestyle(point),
        color=line_color,
        lw=1.5,
    )

    highlight_mask = visits["visit_idx"].to_numpy(dtype=int) == point.visit_idx
    if np.any(highlight_mask):
        hi_cycles = cycles[highlight_mask]
        hi_values = values[highlight_mask]
        ax.scatter(
            hi_cycles,
            hi_values,
            s=120,
            facecolors="none",
            edgecolors=line_color,
            linewidths=2.5,
            zorder=5,
        )
        for c in hi_cycles:
            ax.axvline(c, color=line_color, alpha=0.25, lw=1, zorder=1)

    ax.set_xlabel("Cycle")
    ax.set_ylabel("AIC")
    ax.set_title(
        f"Cell {point.cell_idx}, room {point.room_id}: "
        f"{point.r2_band} 5% $R^2$, {point.aic_extremum} AIC "
        f"($R^2$={point.r2:.3f}, AIC={point.aic:.0f} at C{point.cycle_id + 1})"
    )
    ax.set_xticks(cycles)
    ax.set_xticklabels([f"C{c + 1}" for c in cycles], rotation=45, ha="right")
    fig.tight_layout()
    return fig


def plot_outlier_gaussian_room_evolution(
    point: OutlierPoint,
    visits: pd.DataFrame,
    ratemaps_by_visit: dict[tuple[int, int], np.ndarray],
    *,
    n_gaussians: int = 2,
    param_names: tuple[str, ...] = GAUSSIAN_PARAM_NAMES,
    field_shape: tuple[int, int] = DEFAULT_GAUSSIAN_FIELD_SHAPE,
    im_width: float = 1.5,
) -> plt.Figure | None:
    """
    GT + Gaussian decomposition across room visits for one outlier point.

    Rows: ground truth, Gaussian sum, `=`, `g1`, `+`, `g2`, …
    Highlights the selection cycle column: green = top R² tail, red = bottom.
    """
    if visits.empty:
        return None

    n_visits = len(visits)
    row_layout = _outlier_decomposition_row_layout(n_gaussians)
    n_rows = len(row_layout)
    height_ratios = [1.0 if kind == "image" else 0.6 for kind, _ in row_layout]

    fig, axes = plt.subplots(
        n_rows,
        n_visits,
        figsize=(im_width * n_visits + 0.8, im_width * (n_gaussians + 2) + 0.8),
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios},
    )

    gt_fields: list[np.ndarray | None] = []
    fits: list[dict | None] = []
    for _, row in visits.iterrows():
        visit_idx = int(row["visit_idx"])
        cell_idx = int(row["cell_idx"])
        gt_fields.append(ratemaps_by_visit.get((visit_idx, cell_idx)))
        fits.append(
            fit_dict_from_visit_row(
                row,
                n_gaussians,
                field_shape=field_shape,
                param_names=param_names,
            )
        )

    finite_vals = [
        arr.ravel()
        for gt in gt_fields
        if gt is not None
        for arr in (gt,)
    ]
    finite_vals.extend(
        arr.ravel()
        for fit in fits
        if fit is not None
        for arr in (fit["sum_field"], *fit["components"])
    )
    if finite_vals:
        stack = np.concatenate(finite_vals)
        vmin = float(np.nanmin(stack))
        vmax = float(np.nanmax(stack))
    else:
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        vmax = vmin + 1e-6

    cmap = _gaussian_evolution_cmap()
    im = None
    highlight_col: int | None = None
    image_axes: list[plt.Axes] = []

    for col, (_, row) in enumerate(visits.iterrows()):
        cycle_id = int(row["cycle_id"])
        visit_idx = int(row["visit_idx"])
        if visit_idx == point.visit_idx:
            highlight_col = col

        r2 = row["r2"]
        r2_label = "NaN" if pd.isna(r2) else f"{r2:.2f}"
        aic = row["aic"]
        aic_label = "NaN" if pd.isna(aic) else f"{aic:.0f}"
        col_title = f"C{cycle_id + 1}\n$R^2$={r2_label}\nAIC={aic_label}"

        gt = gt_fields[col]
        fit = fits[col]
        components: list[np.ndarray] = []
        component_params: list[dict] = []
        if fit is not None:
            components, component_params = _ordered_gaussian_components(fit)

        for row_idx, (kind, key) in enumerate(row_layout):
            ax = axes[row_idx, col]
            if kind == "op":
                ax.axis("off")
                ax.text(0.5, 0.25, key, ha="center", va="center", fontsize=12)
                continue

            image_axes.append(ax)
            ylabel = key
            if key == "gt":
                if gt is None:
                    ax.imshow(
                        np.full(field_shape, np.nan),
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                    )
                else:
                    im = ax.imshow(gt, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(col_title, fontsize=7)
            elif key == "sum":
                if fit is None:
                    ax.imshow(
                        np.full(field_shape, np.nan),
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                    )
                    ax.text(
                        0.5,
                        0.5,
                        "fit failed",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="0.35",
                    )
                else:
                    im = ax.imshow(
                        fit["sum_field"],
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                    )
                    metrics_text = _format_fit_metrics(fit)
                    if metrics_text:
                        ax.text(
                            0.03,
                            -0.03,
                            metrics_text,
                            transform=ax.transAxes,
                            fontsize=6,
                            va="top",
                            ha="left",
                            color="white",
                            bbox={"facecolor": "black", "alpha": 0.45, "pad": 1},
                        )
            else:
                g_idx = int(key[1:]) - 1
                if fit is None or g_idx >= len(components):
                    ax.imshow(
                        np.full(field_shape, np.nan),
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                    )
                else:
                    im = ax.imshow(
                        components[g_idx],
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                    )
                    p = component_params[g_idx]
                    ax.set_xlabel(
                        r"$\mu$" + f"=({p['mu_x']:.1f},{p['mu_y']:.1f})\n"
                        r"$\sigma$" + f"=({p['sigma_x']:.1f},{p['sigma_y']:.1f})",
                        fontsize=6,
                    )

            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8)

    if highlight_col is not None:
        for row_idx, (kind, _) in enumerate(row_layout):
            if kind != "image":
                continue
            ax = axes[row_idx, highlight_col]
            color = _outlier_highlight_color(point)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)

    fig.suptitle(
        f"Cell {point.cell_idx}, room {point.room_id}: "
        f"{point.r2_band} 5% $R^2$, {point.aic_extremum} AIC "
        f"($R^2$={point.r2:.3f}, AIC={point.aic:.0f} at C{point.cycle_id + 1})",
        fontsize=11,
        y=1.02,
    )
    if im is not None:
        fig.subplots_adjust(right=0.88, top=0.86)
        fig.colorbar(
            im,
            ax=image_axes,
            shrink=0.55,
            label="Firing rate",
            pad=0.02,
        )
        _align_decomposition_op_axes(axes, row_layout)
    else:
        fig.tight_layout()
    return fig


def plot_all_outlier_analysis(
    df: pd.DataFrame,
    truncated_paths: list[Path | str],
    points: list[OutlierPoint],
    *,
    ratemaps_by_visit: dict[tuple[int, int], np.ndarray] | None = None,
    n_gaussians: int = 2,
    param_names: tuple[str, ...] = GAUSSIAN_PARAM_NAMES,
    field_shape: tuple[int, int] = DEFAULT_GAUSSIAN_FIELD_SHAPE,
    im_width: float = 1.5,
) -> tuple[
    list[tuple[OutlierPoint, plt.Figure]],
    list[tuple[OutlierPoint, plt.Figure]],
    dict[tuple[int, int], np.ndarray] | None,
]:
    """
    Build metric-evolution and GT+Gaussian figures for all outlier points.

    Loads raw rate maps once (memory-safe) for the union of required visits,
    unless *ratemaps_by_visit* is provided (e.g. from a prior call in the
    notebook). Returns the loaded map dict so callers can cache it.
    """
    if not points:
        return [], [], ratemaps_by_visit

    if ratemaps_by_visit is None:
        segments = build_truncated_segments(truncated_paths)
        n_rooms = n_rooms_from_truncated_file(segments[0][2])
        requests = collect_ratemap_requests(df, points, segments, n_rooms)
        ratemaps_by_visit = load_ratemaps_by_visit(requests)

    metric_figs: list[tuple[OutlierPoint, plt.Figure]] = []
    gaussian_figs: list[tuple[OutlierPoint, plt.Figure]] = []

    for point in points:
        visits = visits_for_cell_in_room(df, point.cell_idx, point.room_id)
        metric_fig = plot_outlier_metric_evolution(point, visits)
        metric_figs.append((point, metric_fig))

        gaussian_fig = plot_outlier_gaussian_room_evolution(
            point,
            visits,
            ratemaps_by_visit,
            n_gaussians=n_gaussians,
            param_names=param_names,
            field_shape=field_shape,
            im_width=im_width,
        )
        if gaussian_fig is not None:
            gaussian_figs.append((point, gaussian_fig))

    return metric_figs, gaussian_figs, ratemaps_by_visit
