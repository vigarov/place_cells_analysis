import time

import numpy as np
from tqdm.auto import tqdm

from analysis.sum_gaussians import fit_sum_gaussians
from analysis.sum_gaussians_core import (
    _aic_from_least_squares,
    _gaussian_2d,
    _make_sum_gaussians_model,
)


def select_active_cells(final_rm, threshold, max_cells=None):
    """
    Select cells whose peak activation exceeds a threshold.

    Returns
    -------
    cell_indices : np.ndarray
        Indices sorted by descending peak activation.
    peak : np.ndarray
        Peak activation for every cell in `final_rm`.
    """
    peak = np.nanmax(final_rm, axis=(1, 2))
    cell_indices = np.where(peak > threshold)[0]

    if len(cell_indices) == 0:
        return cell_indices, peak

    order = np.argsort(peak[cell_indices])[::-1]
    cell_indices = cell_indices[order]

    if max_cells is not None and len(cell_indices) > max_cells:
        print(
            f"{len(cell_indices)} cells pass filter; showing top {max_cells} by peak activation."
        )
        cell_indices = cell_indices[:max_cells]

    return cell_indices, peak


def fit_gaussian_sums(
    final_rm,
    cell_indices,
    n_gaussians=2,
    verbose=True,
    *,
    show_progress=True,
    device="cpu",
    maxfev=20_000,
    n_steps=3_000,
    lr=0.05,
    early_stop_patience=30,
    early_stop_eps=1e-8,
):
    """Fit Gaussian sums for each selected cell."""
    gaussian_fits = []
    cells = cell_indices
    if show_progress:
        cells = tqdm(cells, desc=f"Gaussian fits (N={n_gaussians}, {device})")
    for cell_idx in cells:
        fit = fit_sum_gaussians(
            final_rm[cell_idx],
            n_gaussians=n_gaussians,
            device=device,
            maxfev=maxfev,
            n_steps=n_steps,
            lr=lr,
            early_stop_patience=early_stop_patience,
            early_stop_eps=early_stop_eps,
        )
        if fit is None:
            if verbose:
                msg = f"cell {cell_idx}: could not fit Gaussian sum"
                if show_progress:
                    tqdm.write(msg)
                else:
                    print(msg)
            continue
        fit["cell_idx"] = cell_idx
        gaussian_fits.append(fit)
    return gaussian_fits


def mean_r2_by_n_and_maxfev(
    final_rm,
    cell_indices,
    n_gaussians_values=(1, 2, 3),
    maxfev_values=None,
    *,
    show_progress=True,
):
    """
    Fit Gaussian sums on a grid of ``n_gaussians`` and ``maxfev`` values.

    Uses ``scipy.optimize.curve_fit`` (CPU) with the given ``maxfev`` limit.

    Returns
    -------
    dict
        ``n_values``, ``maxfev_values``, ``mean_r2`` and ``wall_time_s``
        (each shape ``n × maxfev``), ``total_wall_time_s``, and
        ``gaussian_fits_by_n_maxfev`` mapping ``(n, maxfev)`` to fit lists.
    """
    if maxfev_values is None:
        maxfev_values = np.sort(np.r_[np.logspace(2, 4, 5), 2 * np.logspace(2, 4, 5)])
    n_values = np.asarray(n_gaussians_values, dtype=int)
    maxfev_values = np.asarray(maxfev_values, dtype=int)
    mean_r2 = np.full((len(n_values), len(maxfev_values)), np.nan)
    wall_time_s = np.full((len(n_values), len(maxfev_values)), np.nan)
    gaussian_fits_by_n_maxfev = {}

    t_start_total = time.perf_counter()
    n_iter = tqdm(n_values, desc="R² grid vs N and maxfev") if show_progress else n_values
    for i, n in enumerate(n_iter):
        fev_iter = (
            tqdm(maxfev_values, desc=f"maxfev (N={n})", leave=False)
            if show_progress
            else maxfev_values
        )
        for j, maxfev in enumerate(fev_iter):
            t0 = time.perf_counter()
            fits = fit_gaussian_sums(
                final_rm,
                cell_indices,
                n_gaussians=int(n),
                verbose=False,
                show_progress=False,
                device="cpu",
                maxfev=int(maxfev),
            )
            elapsed = time.perf_counter() - t0
            wall_time_s[i, j] = elapsed
            gaussian_fits_by_n_maxfev[(int(n), int(maxfev))] = fits
            if fits:
                mean_r2[i, j] = np.mean([fit["r2"] for fit in fits])
            if show_progress and hasattr(fev_iter, "set_postfix"):
                fev_iter.set_postfix(time=f"{elapsed:.1f}s")

    total_wall_time_s = time.perf_counter() - t_start_total
    msg = f"Gaussian grid total wall time: {total_wall_time_s:.1f} s"
    if show_progress:
        tqdm.write(msg)
    else:
        print(msg)

    return {
        "n_values": n_values,
        "maxfev_values": maxfev_values,
        "mean_r2": mean_r2,
        "wall_time_s": wall_time_s,
        "total_wall_time_s": total_wall_time_s,
        "gaussian_fits_by_n_maxfev": gaussian_fits_by_n_maxfev,
    }


def mean_r2_by_n_gaussians(
    final_rm,
    cell_indices,
    n_gaussians_range=6,
    device="cpu",
    *,
    n_steps=3_000,
    lr=0.05,
    early_stop_patience=30,
    early_stop_eps=1e-8,
):
    """
    Fit Gaussian sums for each `n` in `range(n_gaussians_range)` and return
    the mean R² across selected cells.

    Parameters
    ----------
    device : {'cpu', 'gpu'}
        `'cpu'` fits with scipy; `'gpu'` fits with PyTorch gradient descent.
    early_stop_patience, early_stop_eps
        GPU only. Stop when train MSE changes by at most `early_stop_eps` for
        `early_stop_patience` consecutive steps.

    Returns
    -------
    dict
        `n_values` and `mean_r2` arrays, one entry per `n`;
        `gaussian_fits_by_n` maps each `n` to the list of per-cell fit dicts
        from `fit_gaussian_sums`; `wall_time_by_n_s` maps each `n` to elapsed
        seconds for that sweep; `wall_time_s` is the total elapsed seconds;
        `device` is the fitting backend used.
    """
    n_values = np.arange(n_gaussians_range)
    mean_r2 = np.full(n_gaussians_range, np.nan)
    gaussian_fits_by_n = {}
    wall_time_by_n_s = {}
    t0 = time.perf_counter()

    for n in tqdm(n_values, desc=f"Mean R² vs N Gaussians ({device})"):
        if n == 0:
            continue
        t_n = time.perf_counter()
        fits = fit_gaussian_sums(
            final_rm,
            cell_indices,
            n_gaussians=n,
            verbose=False,
            show_progress=False,
            device=device,
            n_steps=n_steps,
            lr=lr,
            early_stop_patience=early_stop_patience,
            early_stop_eps=early_stop_eps,
            maxfev = int(n*40)
        )
        end = time.perf_counter()
        wall_time_by_n_s[int(n)] = end - t_n
        gaussian_fits_by_n[int(n)] = fits
        if fits:
            mean_r2[n] = np.mean([fit["r2"] for fit in fits])

    return {
        "n_values": n_values,
        "mean_r2": mean_r2,
        "gaussian_fits_by_n": gaussian_fits_by_n,
        "wall_time_by_n_s": wall_time_by_n_s,
        "wall_time_s": time.perf_counter() - t0,
        "device": device,
    }


def collect_r2_by_cell_and_n(r2_by_n):
    """
    Collect per-cell R² values across Gaussian counts.

    Returns
    -------
    n_values : np.ndarray
    r2_by_cell : dict[int, np.ndarray]
        Maps each cell index to its R² values over `n_values`.
    """
    gaussian_fits_by_n = r2_by_n["gaussian_fits_by_n"]
    n_values = np.array(sorted(n for n in gaussian_fits_by_n if n > 0))
    cell_indices = sorted(
        {
            fit["cell_idx"]
            for n in n_values
            for fit in gaussian_fits_by_n[int(n)]
        }
    )
    r2_by_cell = {}
    for cell_idx in cell_indices:
        r2_vals = []
        for n in n_values:
            fit = next(
                (f for f in gaussian_fits_by_n[int(n)] if f["cell_idx"] == cell_idx),
                None,
            )
            r2_vals.append(fit["r2"] if fit is not None else np.nan)
        r2_by_cell[cell_idx] = np.array(r2_vals, dtype=np.float64)
    return n_values, r2_by_cell


def sem_r2_over_n_by_cell(r2_by_n):
    """
    Compute the SEM of R² across Gaussian counts for each cell.

    Returns
    -------
    n_values : np.ndarray
    r2_by_cell : dict[int, np.ndarray]
    sem_by_cell : dict[int, float]
    """
    n_values, r2_by_cell = collect_r2_by_cell_and_n(r2_by_n)
    sem_by_cell = {}
    for cell_idx, r2_vals in r2_by_cell.items():
        valid = r2_vals[np.isfinite(r2_vals)]
        if len(valid) < 2:
            sem_by_cell[cell_idx] = np.nan
        else:
            sem_by_cell[cell_idx] = np.std(valid, ddof=1) / np.sqrt(len(valid))
    return n_values, r2_by_cell, sem_by_cell


def select_cells_sensitive_to_n(r2_by_n, sem_threshold=0.2):
    """
    Select cells whose R² SEM across N exceeds a threshold.

    Returns
    -------
    cell_indices : np.ndarray
        Sensitive cells ordered by descending SEM.
    sem_by_cell : dict[int, float]
    r2_by_cell : dict[int, np.ndarray]
    n_values : np.ndarray
    """
    n_values, r2_by_cell, sem_by_cell = sem_r2_over_n_by_cell(r2_by_n)
    sensitive = [
        cell_idx
        for cell_idx, sem in sem_by_cell.items()
        if np.isfinite(sem) and sem > sem_threshold
    ]
    order = np.argsort([sem_by_cell[cell_idx] for cell_idx in sensitive])[::-1]
    cell_indices = np.array(sensitive, dtype=int)[order]
    return cell_indices, sem_by_cell, r2_by_cell, n_values


def print_gaussian_fit_summary(gaussian_fits, n_gaussians):
    """Print fit quality and component parameters for each cell."""
    has_aic = any("aic" in fit for fit in gaussian_fits)
    print(f"Gaussian sum fits (N={n_gaussians}, final epoch):")
    if has_aic:
        print(f"{'cell':>6} {'R²':>8} {'AIC':>12}")
    else:
        print(f"{'cell':>6} {'R²':>8}")
    for fit in gaussian_fits:
        if has_aic:
            aic = fit.get("aic", np.nan)
            print(f"{fit['cell_idx']:6d} {fit['r2']:8.3f} {aic:12.1f}")
        else:
            print(f"{fit['cell_idx']:6d} {fit['r2']:8.3f}")
        for k, p in enumerate(fit["component_params"], start=1):
            print(
                f"       g{k}: amp={p['amplitude']:.3f} "
                f"μ=({p['mu_x']:.1f},{p['mu_y']:.1f}) "
                f"σ=({p['sigma_x']:.1f},{p['sigma_y']:.1f})"
            )
