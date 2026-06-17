import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from .cell_evolution_analysis import select_cells_sensitive_to_n


def _ratemap_cmap():
    cmap = plt.cm.YlOrBr.copy()
    cmap.set_under("white")
    cmap.set_over("black")
    return cmap


def plot_ratemap_evolution(
    epoch_ratemaps,
    cell_indices,
    peak,
    threshold,
    im_width=2,
    output_dir=None,
):
    """Plot one row per cell and one column per logged epoch."""
    if len(cell_indices) == 0:
        print(
            f"No cells with peak activation > {threshold}. "
            "Try lowering the activation threshold."
        )
        return

    n_rows = len(cell_indices)
    n_cols = len(epoch_ratemaps)
    last_epoch_maps = epoch_ratemaps[-1][cell_indices]
    vmin = np.nanmin(last_epoch_maps)
    vmax = np.nanmax(last_epoch_maps)
    if vmin == vmax:
        vmax = vmin + 1e-6

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(im_width * n_cols, im_width * n_rows),
        squeeze=False,
    )
    im = None
    cmap = _ratemap_cmap()
    for row, cell_idx in enumerate(cell_indices):
        for col, rm in enumerate(epoch_ratemaps):
            ax = axes[row, col]
            im = ax.imshow(rm[cell_idx], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"epoch {col - 1}" if col > 0 else "initial", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"cell {cell_idx}\npeak {peak[cell_idx]:.2f}", fontsize=8)

    fig.suptitle(
        f"Rate map evolution ({n_rows} cells, peak > {threshold})",
        fontsize=12,
    )
    fig.subplots_adjust(right=0.92, top=0.94)
    if im is not None:
        fig.colorbar(
            im, ax=axes.ravel().tolist(), shrink=0.4, label="Firing rate", extend="both"
        )
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, "ratemap_evolution.pdf"), bbox_inches="tight")
    plt.show()


def _gaussian_fit_col_layout(n_gaussians):
    gt_col, sum_col, eq_col = 0, 1, 2
    gaussian_cols = []
    op_cols = []
    col = 3
    for k in range(n_gaussians):
        if k > 0:
            op_cols.append(col)
            col += 1
        gaussian_cols.append(col)
        col += 1
    return gt_col, sum_col, eq_col, gaussian_cols, op_cols


def _format_fit_metrics(fit: dict) -> str:
    """One- or two-line annotation for R² and optional AIC."""
    parts: list[str] = []
    r2 = fit.get("r2")
    if r2 is not None and np.isfinite(r2):
        parts.append(rf"$R^2$={r2:.3f}")
    aic = fit.get("aic")
    if aic is not None and np.isfinite(aic):
        parts.append(f"AIC={aic:.0f}")
    return "\n".join(parts)


def plot_gaussian_fits(
    final_rm,
    gaussian_fits,
    n_gaussians,
    im_width=2,
    output_dir=None,
    save_name=None,
):
    """Plot ground-truth rate maps next to their Gaussian-sum decompositions."""
    if not gaussian_fits:
        print("No Gaussian fits to plot.")
        return

    gaussian_fits = sorted(
        gaussian_fits,
        key=lambda fit: np.nanmax(final_rm[fit["cell_idx"]]),
        reverse=True,
    )
    cell_indices = [fit["cell_idx"] for fit in gaussian_fits]
    n_rows = len(gaussian_fits)
    n_cols = 2 + 1 + n_gaussians + (n_gaussians - 1)
    gt_col, sum_col, eq_col, gaussian_cols, op_cols = _gaussian_fit_col_layout(n_gaussians)

    width_ratios = [1.0, 1.0, 0.12]
    for k in range(n_gaussians):
        if k > 0:
            width_ratios.append(0.12)
        width_ratios.append(1.0)

    vmin = np.nanmin(final_rm[cell_indices])
    vmax = np.nanmax(final_rm[cell_indices])
    if vmin == vmax:
        vmax = vmin + 1e-6
    cmap = plt.cm.YlGnBu.copy()

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(im_width * (2 + 1.2 * n_gaussians), im_width * n_rows),
        squeeze=False,
        gridspec_kw={"width_ratios": width_ratios},
    )
    im = None
    for row, fit in enumerate(gaussian_fits):
        cell_idx = fit["cell_idx"]
        data = final_rm[cell_idx]

        ax = axes[row, gt_col]
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title("ground truth", fontsize=9)
        peak = np.nanmax(data)
        ax.set_ylabel(f"cell {cell_idx}\npeak {peak:.2f}", fontsize=8)

        ax = axes[row, sum_col]
        ax.imshow(fit["sum_field"], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title("Gaussian sum", fontsize=9)
        metrics_text = _format_fit_metrics(fit)
        if metrics_text:
            ax.text(
                0.03,
                -0.03,
                metrics_text,
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                ha="left",
                color="white",
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 2},
            )

        ax = axes[row, eq_col]
        ax.axis("off")
        ax.text(0.5, 0.5, "=", ha="center", va="center", fontsize=14)

        components = fit["components"]
        component_params = fit["component_params"]
        if n_gaussians > 1:
            order = sorted(
                range(len(components)),
                key=lambda k: np.nanmax(components[k]),
                reverse=True,
            )
            components = [components[k] for k in order]
            component_params = [component_params[k] for k in order]

        for k, comp in enumerate(components):
            ax = axes[row, gaussian_cols[k]]
            ax.imshow(comp, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"g{k + 1}", fontsize=9)
            p = component_params[k]
            ax.set_xlabel(
                r"$\mu$" + f"=({p['mu_x']:.1f},{p['mu_y']:.1f})\n"
                r"$\sigma$" + f"=({p['sigma_x']:.1f},{p['sigma_y']:.1f})",
                fontsize=7,
            )

        for op_col in op_cols:
            ax = axes[row, op_col]
            ax.axis("off")
            ax.text(0.5, 0.5, "+", ha="center", va="center", fontsize=14)

    fig.suptitle(
        f"Ground truth vs sum of {n_gaussians} independent 2D Gaussians",
        fontsize=12,
    )
    fig.subplots_adjust(right=0.88, top=0.94)
    if im is not None:
        fig.colorbar(
            im, ax=axes.ravel().tolist(), shrink=0.4, label="Firing rate", pad=0.2
        )
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        filename = save_name or f"gaussian_fits_n{n_gaussians}.pdf"
        fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.show()


def plot_mean_r2_vs_n_gaussians(r2_by_n, n_cells=None, output_dir=None):
    """Plot mean Gaussian-sum R² vs number of Gaussians from ``mean_r2_by_n_gaussians``."""
    n_values = r2_by_n["n_values"]
    mean_r2 = r2_by_n["mean_r2"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(n_values, mean_r2, marker="o")
    ax.set_xlabel("N Gaussians")
    ax.set_ylabel("Mean $R^2$ across cells")
    ax.set_xticks(n_values)
    title = "Mean Gaussian-sum fit quality"
    if n_cells is not None:
        title += f" ({n_cells} cells)"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(
            os.path.join(output_dir, "mean_r2_vs_n_gaussians.pdf"), bbox_inches="tight"
        )
    plt.show()


def _plot_n_maxfev_grid_surface(
    grid,
    *,
    z_key,
    z_title,
    plot_title,
    colorscale,
    hover_z_format,
    n_cells=None,
    log_scale=True,
):
    n_values = grid["n_values"]
    maxfev_values = grid["maxfev_values"]
    z = grid[z_key]
    y_fev = np.log10(maxfev_values) if log_scale else maxfev_values
    z_t = z.T

    fig = go.Figure(
        data=[
            go.Surface(
                x=n_values,
                y=y_fev,
                z=z_t,
                colorscale=colorscale,
                colorbar=dict(title=z_title),
                hovertemplate=(
                    f"N=%{{x}}<br>maxfev=%{{customdata}}<br>"
                    f"{z_title}=%{{z:{hover_z_format}}}<extra></extra>"
                ),
                customdata=np.broadcast_to(
                    maxfev_values.reshape(-1, 1), z_t.shape
                ),
            )
        ]
    )
    title = plot_title
    if n_cells is not None:
        title += f" ({n_cells} cells)"
    fig.update_layout(
        title=title,
        width=900,
        height=600,
        scene=dict(
            xaxis_title="N Gaussians",
            yaxis_title="log₁₀(maxfev)" if log_scale else "maxfev",
            zaxis_title=z_title,
            xaxis=dict(tickmode="array", tickvals=n_values.tolist()),
            yaxis=dict(
                tickmode="array",
                tickvals=y_fev.tolist(),
                ticktext=[str(v) for v in maxfev_values],
            ),
        ),
    )
    fig.show()


def plot_mean_r2_by_n_and_maxfev(r2_grid, n_cells=None, log_scale=True):
    """Interactive 3D surface of mean R² vs N Gaussians and maxfev (from ``mean_r2_by_n_and_maxfev``)."""
    _plot_n_maxfev_grid_surface(
        r2_grid,
        z_key="mean_r2",
        z_title="Mean R²",
        plot_title="Gaussian-sum fit quality",
        colorscale="Viridis",
        hover_z_format=".4f",
        n_cells=n_cells,
        log_scale=log_scale,
    )


def plot_wall_time_by_n_and_maxfev(r2_grid, n_cells=None, log_scale=True):
    """Interactive 3D surface of fit wall time vs N Gaussians and maxfev."""
    _plot_n_maxfev_grid_surface(
        r2_grid,
        z_key="wall_time_s",
        z_title="Wall time (s)",
        plot_title="Gaussian-sum fit wall time",
        colorscale="Plasma",
        hover_z_format=".2f",
        n_cells=n_cells,
        log_scale=log_scale,
    )


def plot_cells_sensitive_to_n(
    r2_by_n,
    sem_threshold=0.2,
    output_dir=None,
):
    """Plot R² vs N for cells with SEM(R²) over N above a threshold."""
    cell_indices, sem_by_cell, r2_by_cell, n_values = select_cells_sensitive_to_n(
        r2_by_n, sem_threshold=sem_threshold
    )
    if len(cell_indices) == 0:
        print(f"No cells with SEM($R^2$) > {sem_threshold} across N.")
        return

    n_rows = len(cell_indices)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(6, 2.2 * n_rows), squeeze=False, sharex=True
    )
    for row, cell_idx in enumerate(cell_indices):
        ax = axes[row, 0]
        r2_vals = r2_by_cell[cell_idx]
        sem = sem_by_cell[cell_idx]
        ax.plot(n_values, r2_vals, marker="o")
        ax.set_ylabel(f"cell {cell_idx}\nSEM={sem:.3f}", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("N Gaussians")
    axes[-1, 0].set_xticks(n_values)
    fig.suptitle(
        f"Cells sensitive to N (SEM($R^2$) > {sem_threshold}, n={n_rows})",
        fontsize=12,
    )
    plt.tight_layout()
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(
            os.path.join(output_dir, "cells_sensitive_to_n.pdf"), bbox_inches="tight"
        )
    plt.show()
