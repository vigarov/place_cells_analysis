#!/usr/bin/env python3
"""
Temporary patch: backfill ``aic`` and per-rate-map signal stats in ``gaussian_rf_fits.npz``.

Loads each ``cycles_ratemaps_truncated_*.npz`` one at a time (rate maps are large),
records ``mean`` / ``max`` / ``std`` of finite pixels in each rate map, and for rows
with saved ``gaussian_params`` re-evaluates the Gaussian sum to compute AIC (same
formula as ``analysis/cell_evolution_analysis.py``).

Usage::

    uv run python cycles/patch_gaussian_rf_aic_stats.py

    uv run python cycles/patch_gaussian_rf_aic_stats.py --results-dir cycles/results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from analysis.cell_evolution_analysis import (
    _aic_from_least_squares,
    _make_sum_gaussians_model,
)
from cycles.cycles_paths import RESULTS_DIR
from cycles.cycles_train import discover_truncated_ratemaps_series
from cycles.estimate_gaussians_rf import (
    DEFAULT_OUTPUT_NAME,
    DEFAULT_TOTAL_CYCLES,
    _n_rooms_from_truncated_file,
)


def _signal_stats(field: np.ndarray) -> tuple[float, float, float]:
    """``mean``, ``max``, ``std`` over finite pixels (``nan`` if none)."""
    vals = field[np.isfinite(field)]
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    return float(vals.mean()), float(vals.max()), float(vals.std())


def _aic_from_saved_params(
    field: np.ndarray,
    params: np.ndarray,
    *,
    n_gaussians: int,
) -> float:
    """AIC for a fixed parameter vector ``(n_gaussians, 5)``."""
    if not np.all(np.isfinite(params)):
        return np.nan

    mask = np.isfinite(field)
    if not np.any(mask):
        return np.nan

    h, w = field.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x_pts = xx[mask]
    y_pts = yy[mask]
    y_true = field[mask].astype(np.float64)

    model = _make_sum_gaussians_model(n_gaussians)
    popt = params.reshape(-1)
    y_pred = model((x_pts, y_pts), *popt)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    n_obs = int(y_true.size)
    n_params = 5 * n_gaussians
    return _aic_from_least_squares(n_obs, ss_res, n_params)


def _load_fits_payload(fits_path: Path) -> dict[str, np.ndarray | int | str]:
    with np.load(fits_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _atomic_save_npz(path: Path, payload: dict[str, np.ndarray | int | str]) -> None:
    """
    Write ``path`` atomically via a sibling ``.npz`` temp file.

    ``np.savez_compressed`` appends ``.npz`` when the path lacks that suffix
    (e.g. ``foo.tmp`` → ``foo.tmp.npz``), so the temp name must end in ``.npz``.
    """
    path = Path(path)
    tmp_path = path.with_name(f"{path.stem}.patch.tmp.npz")
    np.savez_compressed(tmp_path, **payload)
    tmp_path.replace(path)


def patch_gaussian_rf_metrics(
    results_dir: Path,
    *,
    fits_name: str = DEFAULT_OUTPUT_NAME,
    total_cycles: int = DEFAULT_TOTAL_CYCLES,
    show_progress: bool = True,
) -> Path:
    results_dir = Path(results_dir)
    fits_path = results_dir / fits_name
    if not fits_path.is_file():
        raise FileNotFoundError(fits_path)

    payload = _load_fits_payload(fits_path)
    for key in ("aic", "signal_mean", "signal_max", "signal_std"):
        if key in payload:
            print(f"Note: replacing existing '{key}' in {fits_path}")

    r2 = np.asarray(payload["r2"])
    gaussian_params = np.asarray(payload["gaussian_params"])
    n_gaussians = int(payload["n_gaussians"])
    n_visits, n_cells = r2.shape

    segments = discover_truncated_ratemaps_series(
        results_dir, total_cycles=total_cycles
    )
    n_rooms = _n_rooms_from_truncated_file(segments[0][2])

    aic = np.full((n_visits, n_cells), np.nan, dtype=np.float64)
    signal_mean = np.full((n_visits, n_cells), np.nan, dtype=np.float64)
    signal_max = np.full((n_visits, n_cells), np.nan, dtype=np.float64)
    signal_std = np.full((n_visits, n_cells), np.nan, dtype=np.float64)

    for start_cycle, end_cycle, segment_path in segments:
        visit_start = start_cycle * n_rooms
        visit_end = end_cycle * n_rooms
        n_seg_visits = visit_end - visit_start

        with np.load(segment_path) as data:
            ratemaps = np.asarray(data["ratemaps"])
        if ratemaps.shape[0] != n_seg_visits:
            raise ValueError(
                f"{segment_path.name}: expected {n_seg_visits} visits, "
                f"got {ratemaps.shape[0]}"
            )

        seg_params = gaussian_params[visit_start:visit_end]
        valid = np.isfinite(seg_params).all(axis=(2, 3))
        total_curves = n_seg_visits * n_cells

        progress = tqdm(
            total=total_curves,
            desc=f"Patch {segment_path.name} (cycles {start_cycle}:{end_cycle})",
            disable=not show_progress,
        )
        try:
            for local_v in range(n_seg_visits):
                global_v = visit_start + local_v
                for cell_idx in range(n_cells):
                    field = ratemaps[local_v, cell_idx]
                    mean_v, max_v, std_v = _signal_stats(field)
                    signal_mean[global_v, cell_idx] = mean_v
                    signal_max[global_v, cell_idx] = max_v
                    signal_std[global_v, cell_idx] = std_v
                    if valid[local_v, cell_idx]:
                        aic[global_v, cell_idx] = _aic_from_saved_params(
                            field,
                            seg_params[local_v, cell_idx],
                            n_gaussians=n_gaussians,
                        )
                    progress.update(1)
        finally:
            progress.close()

        del ratemaps

    payload["aic"] = aic
    payload["signal_mean"] = signal_mean
    payload["signal_max"] = signal_max
    payload["signal_std"] = signal_std

    _atomic_save_npz(fits_path, payload)
    return fits_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill AIC and signal mean/max/std in gaussian_rf_fits.npz "
            "from truncated rate maps and saved Gaussian parameters."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Directory with truncated NPZs and {DEFAULT_OUTPUT_NAME} (default: results/cycles).",
    )
    parser.add_argument(
        "--fits",
        type=str,
        default=DEFAULT_OUTPUT_NAME,
        help=f"Gaussian fits filename (default: {DEFAULT_OUTPUT_NAME}).",
    )
    parser.add_argument(
        "--total-cycles",
        type=int,
        default=DEFAULT_TOTAL_CYCLES,
        help=f"Expected cycle count for truncated series (default: {DEFAULT_TOTAL_CYCLES}).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    args = parser.parse_args(argv)

    out = patch_gaussian_rf_metrics(
        args.results_dir,
        fits_name=args.fits,
        total_cycles=args.total_cycles,
        show_progress=not args.no_progress,
    )
    with np.load(out) as data:
        for key in ("aic", "signal_mean", "signal_max", "signal_std"):
            arr = data[key]
            n_finite = int(np.isfinite(arr).sum())
            print(f"  {key}: shape {arr.shape}, {n_finite} finite values")
        print(f"Wrote {out}")


# Backward-compatible alias
patch_aic = patch_gaussian_rf_metrics


if __name__ == "__main__":
    main()
