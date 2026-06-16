#!/usr/bin/env python3
"""
Fit N-Gaussian receptive fields to cycles experiment rate maps.

For each visit and hidden unit, fits a sum of ``n_gaussians`` independent 2D
Gaussians (same routine as ``cell_evolutions/cell_evolution_analysis.py``).
Writes ``gaussian_rf_fits.npz`` next to the input results file or truncated-dir.
Each output includes ``r2``, ``gaussian_params``, ``aic``, and per-rate-map
``signal_mean`` / ``signal_max`` / ``signal_std``.

Usage::

    uv run estimate-gaussians-rf

    uv run estimate-gaussians-rf --input cycles/results/cycles_ratemaps_truncated_10.npz

    uv run estimate-gaussians-rf --truncated-dir --input cycles/results

    uv run estimate-gaussians-rf --device cpu --n-processes auto
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from cell_evolutions.cell_evolution_analysis import (
    _aic_from_least_squares,
    _make_sum_gaussians_model,
    fit_sum_gaussians,
)
from cycles.cycles_paths import RESULTS_DIR
from cycles.cycles_train import (
    DEFAULT_CYCLES_RESULTS_NAME,
    discover_truncated_ratemaps_series,
)

DEFAULT_OUTPUT_NAME = "gaussian_rf_fits.npz"
DEFAULT_TOTAL_CYCLES = 30
CPU_WORKER_RESERVE = 4
CURVES_PER_WORKER_CHUNK = 1000
_PARAM_NAMES = ("amplitude", "mu_x", "mu_y", "sigma_x", "sigma_y")


def default_cpu_process_count(*, reserve: int = CPU_WORKER_RESERVE) -> int:
    """Worker count for ``--n-processes auto``: ``max(1, cpu_count() - reserve)``."""
    count = os.cpu_count() or 1
    return max(1, count - reserve)


def resolve_n_processes(
    n_processes: int | str,
    *,
    device: str,
) -> int:
    """Return process count (1 = serial). GPU always uses 1."""
    if device != "cpu":
        return 1
    if n_processes == "auto":
        return default_cpu_process_count()
    if isinstance(n_processes, str):
        raise ValueError(f"n_processes must be 'auto' or a positive int, got {n_processes!r}")
    if n_processes < 1:
        raise ValueError(f"n_processes must be >= 1, got {n_processes}")
    return int(n_processes)


def _pool_worker_init() -> None:
    """Avoid BLAS/thread oversubscription inside each worker process."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _default_input_path() -> Path:
    return RESULTS_DIR / DEFAULT_CYCLES_RESULTS_NAME


def _output_path(base: Path) -> Path:
    return base / DEFAULT_OUTPUT_NAME


def _partial_output_path(base: Path, part: int, n_parts: int) -> Path:
    return base / f"gaussian_rf_fits.part{part:03d}_of_{n_parts:03d}.npz"


def visit_chunk_bounds(n_visits: int, k_parts: int) -> list[tuple[int, int]]:
    """Split ``n_visits`` into ``k_parts`` contiguous slices (last gets remainder)."""
    if k_parts < 1:
        raise ValueError(f"memory-split must be >= 1, got {k_parts}")
    chunk = n_visits // k_parts
    bounds: list[tuple[int, int]] = []
    for i in range(k_parts):
        start = i * chunk
        end = start + chunk if i < k_parts - 1 else n_visits
        bounds.append((start, end))
    return bounds


def _read_ratemap_shape(input_path: Path) -> tuple[int, int, int, int]:
    with np.load(input_path, mmap_mode="r") as data:
        shape = tuple(int(s) for s in data["ratemaps"].shape)
    if len(shape) != 4:
        raise ValueError(f"Expected ratemaps (n_visits, n_cells, H, W), got shape {shape}")
    return shape  # type: ignore[return-value]


def _load_ratemaps_file(input_path: Path) -> np.ndarray:
    with np.load(input_path) as data:
        return np.asarray(data["ratemaps"])


def _load_metadata_arrays(
    input_path: Path,
    visit_start: int = 0,
    visit_end: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(input_path) as data:
        cycle_ids = np.asarray(data["cycle_ids"])
        room_ids = np.asarray(data["room_ids"])
    if visit_end is not None:
        cycle_ids = cycle_ids[visit_start:visit_end]
        room_ids = room_ids[visit_start:visit_end]
    elif visit_start:
        cycle_ids = cycle_ids[visit_start:]
        room_ids = room_ids[visit_start:]
    return cycle_ids, room_ids


def _curve_chunk_bounds(total_curves: int, chunk_size: int) -> list[tuple[int, int]]:
    """``(flat_start, flat_end)`` pairs covering ``[0, total_curves)``."""
    bounds: list[tuple[int, int]] = []
    for start in range(0, total_curves, chunk_size):
        bounds.append((start, min(start + chunk_size, total_curves)))
    return bounds


def _fit_curves_chunk_worker(
    flat_start: int,
    flat_end: int,
    shm_name: str,
    shape: tuple[int, int, int, int],
    dtype_str: str,
    n_gaussians: int,
) -> tuple[
    int,
    int,
    list[tuple[int, int, float, np.ndarray, float, float, float, float]],
]:
    """
    Fit up to ``CURVES_PER_WORKER_CHUNK`` (visit, cell) rate maps (module-level).

    Returns ``(flat_start, flat_end, [(visit, cell, r2, params, aic,
    signal_mean, signal_max, signal_std), ...])``.
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        ratemaps = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        n_cells = shape[1]
        results: list[
            tuple[int, int, float, np.ndarray, float, float, float, float]
        ] = []
        for flat_idx in range(flat_start, flat_end):
            visit_idx = flat_idx // n_cells
            cell_idx = flat_idx % n_cells
            results.append(
                (
                    visit_idx,
                    cell_idx,
                    *_fit_one_curve(
                        ratemaps[visit_idx, cell_idx],
                        n_gaussians=n_gaussians,
                        device="cpu",
                    ),
                )
            )
        return flat_start, flat_end, results
    finally:
        shm.close()


def _safe_fit_sum_gaussians(
    field: np.ndarray,
    *,
    n_gaussians: int,
    device: str,
) -> dict | None:
    """Call :func:`fit_sum_gaussians`, returning ``None`` on any fit failure."""
    try:
        return fit_sum_gaussians(field, n_gaussians=n_gaussians, device=device)
    except Exception:
        return None


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


def _params_from_fit(fit: dict, n_gaussians: int) -> np.ndarray:
    """Shape ``(n_gaussians, 5)``."""
    out = np.full((n_gaussians, 5), np.nan, dtype=np.float64)
    for k, comp in enumerate(fit["component_params"]):
        if k >= n_gaussians:
            break
        out[k, 0] = comp["amplitude"]
        out[k, 1] = comp["mu_x"]
        out[k, 2] = comp["mu_y"]
        out[k, 3] = comp["sigma_x"]
        out[k, 4] = comp["sigma_y"]
    return out


def _fit_one_curve(
    field: np.ndarray,
    *,
    n_gaussians: int,
    device: str,
) -> tuple[float, np.ndarray, float, float, float, float]:
    """
    Fit one (visit, cell) rate map.

    Returns ``r2``, ``params``, ``aic``, ``signal_mean``, ``signal_max``,
    ``signal_std``. Failed fits leave ``r2``, ``params``, and ``aic`` as NaN
    but still record signal stats when the field has finite pixels.
    """
    signal_mean, signal_max, signal_std = _signal_stats(field)
    fit = _safe_fit_sum_gaussians(field, n_gaussians=n_gaussians, device=device)
    if fit is None:
        return (
            np.nan,
            np.full((n_gaussians, 5), np.nan, dtype=np.float64),
            np.nan,
            signal_mean,
            signal_max,
            signal_std,
        )

    params = _params_from_fit(fit, n_gaussians)
    r2 = float(fit["r2"])
    aic_val = fit.get("aic")
    if aic_val is None or not np.isfinite(aic_val):
        aic = _aic_from_saved_params(field, params, n_gaussians=n_gaussians)
    else:
        aic = float(aic_val)
    return r2, params, aic, signal_mean, signal_max, signal_std


def _fit_ratemaps_serial(
    ratemaps: np.ndarray,
    *,
    n_gaussians: int,
    device: str,
    show_progress: bool,
    desc: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_visits, n_cells, _, _ = ratemaps.shape
    r2 = np.full((n_visits, n_cells), np.nan, dtype=np.float64)
    gaussian_params = np.full(
        (n_visits, n_cells, n_gaussians, 5),
        np.nan,
        dtype=np.float64,
    )
    aic = np.full((n_visits, n_cells), np.nan, dtype=np.float64)
    signal_mean = np.full((n_visits, n_cells), np.nan, dtype=np.float64)
    signal_max = np.full((n_visits, n_cells), np.nan, dtype=np.float64)
    signal_std = np.full((n_visits, n_cells), np.nan, dtype=np.float64)

    total = n_visits * n_cells
    progress = tqdm(
        total=total,
        desc=desc or f"Gaussian RF fits (N={n_gaussians}, {device})",
        disable=not show_progress,
    )
    try:
        for cell_idx in range(n_cells):
            for visit_idx in range(n_visits):
                (
                    r2[visit_idx, cell_idx],
                    gaussian_params[visit_idx, cell_idx],
                    aic[visit_idx, cell_idx],
                    signal_mean[visit_idx, cell_idx],
                    signal_max[visit_idx, cell_idx],
                    signal_std[visit_idx, cell_idx],
                ) = _fit_one_curve(
                    ratemaps[visit_idx, cell_idx],
                    n_gaussians=n_gaussians,
                    device=device,
                )
                progress.update(1)
    finally:
        progress.close()

    return r2, gaussian_params, aic, signal_mean, signal_max, signal_std


def _fit_ratemaps_parallel(
    ratemaps: np.ndarray,
    *,
    n_gaussians: int,
    n_processes: int,
    show_progress: bool,
    desc: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_visits, n_cells, _, _ = ratemaps.shape
    r2 = np.full((n_visits, n_cells), np.nan, dtype=np.float64)
    gaussian_params = np.full(
        (n_visits, n_cells, n_gaussians, 5),
        np.nan,
        dtype=np.float64,
    )
    aic = np.full((n_visits, n_cells), np.nan, dtype=np.float64)
    signal_mean = np.full((n_visits, n_cells), np.nan, dtype=np.float64)
    signal_max = np.full((n_visits, n_cells), np.nan, dtype=np.float64)
    signal_std = np.full((n_visits, n_cells), np.nan, dtype=np.float64)

    shm = shared_memory.SharedMemory(create=True, size=ratemaps.nbytes)
    try:
        shared = np.ndarray(ratemaps.shape, dtype=ratemaps.dtype, buffer=shm.buf)
        shared[:] = ratemaps[:]
        del shared

        shape = ratemaps.shape
        dtype_str = str(ratemaps.dtype)
        progress = tqdm(
            total=n_visits * n_cells,
            desc=desc
            or (
                f"Gaussian RF fits (N={n_gaussians}, cpu, "
                f"{n_processes} processes)"
            ),
            disable=not show_progress,
        )
        total_curves = n_visits * n_cells
        chunk_bounds = _curve_chunk_bounds(total_curves, CURVES_PER_WORKER_CHUNK)
        try:
            with ProcessPoolExecutor(
                max_workers=n_processes,
                initializer=_pool_worker_init,
            ) as executor:
                futures = [
                    executor.submit(
                        _fit_curves_chunk_worker,
                        flat_start,
                        flat_end,
                        shm.name,
                        shape,
                        dtype_str,
                        n_gaussians,
                    )
                    for flat_start, flat_end in chunk_bounds
                ]
                for future in as_completed(futures):
                    flat_start, flat_end, chunk_results = future.result()
                    for (
                        visit_idx,
                        cell_idx,
                        r2_val,
                        params,
                        aic_val,
                        mean_val,
                        max_val,
                        std_val,
                    ) in chunk_results:
                        r2[visit_idx, cell_idx] = r2_val
                        gaussian_params[visit_idx, cell_idx] = params
                        aic[visit_idx, cell_idx] = aic_val
                        signal_mean[visit_idx, cell_idx] = mean_val
                        signal_max[visit_idx, cell_idx] = max_val
                        signal_std[visit_idx, cell_idx] = std_val
                    progress.update(flat_end - flat_start)
        finally:
            progress.close()
    finally:
        shm.close()
        shm.unlink()

    return r2, gaussian_params, aic, signal_mean, signal_max, signal_std


def fit_ratemaps(
    ratemaps: np.ndarray,
    *,
    n_gaussians: int,
    device: str,
    n_processes: int = 1,
    show_progress: bool = True,
    desc: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit Gaussians for each (visit, cell).

    With ``device='cpu'`` and ``n_processes > 1``, fits one visit per worker using
    a shared-memory view of ``ratemaps``.

    Returns
    -------
    r2
        ``(n_visits, n_cells)``
    gaussian_params
        ``(n_visits, n_cells, n_gaussians, 5)``
    aic
        ``(n_visits, n_cells)``
    signal_mean, signal_max, signal_std
        Per-rate-map signal stats over finite pixels, each ``(n_visits, n_cells)``
    """
    if device == "cpu" and n_processes > 1:
        return _fit_ratemaps_parallel(
            ratemaps,
            n_gaussians=n_gaussians,
            n_processes=n_processes,
            show_progress=show_progress,
            desc=desc,
        )
    return _fit_ratemaps_serial(
        ratemaps,
        n_gaussians=n_gaussians,
        device=device,
        show_progress=show_progress,
        desc=desc,
    )


def _save_fits(
    path: Path,
    *,
    r2: np.ndarray,
    gaussian_params: np.ndarray,
    aic: np.ndarray,
    signal_mean: np.ndarray,
    signal_max: np.ndarray,
    signal_std: np.ndarray,
    visit_start: int,
    visit_end: int,
    n_gaussians: int,
    source_path: Path,
    cycle_ids: np.ndarray | None = None,
    room_ids: np.ndarray | None = None,
) -> None:
    payload: dict[str, np.ndarray | int | str] = {
        "r2": r2,
        "gaussian_params": gaussian_params,
        "aic": aic,
        "signal_mean": signal_mean,
        "signal_max": signal_max,
        "signal_std": signal_std,
        "visit_start": visit_start,
        "visit_end": visit_end,
        "n_gaussians": n_gaussians,
        "param_names": np.array(_PARAM_NAMES),
        "source_path": str(source_path),
    }
    if cycle_ids is not None:
        payload["cycle_ids"] = cycle_ids
    if room_ids is not None:
        payload["room_ids"] = room_ids
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _merge_partial_files(
    partial_paths: list[Path],
    *,
    n_visits: int,
    n_cells: int,
    n_gaussians: int,
    output_path: Path,
    source_path: Path,
) -> None:
    r2 = np.full((n_visits, n_cells), np.nan, dtype=np.float64)
    gaussian_params = np.full(
        (n_visits, n_cells, n_gaussians, 5),
        np.nan,
        dtype=np.float64,
    )
    aic = np.full((n_visits, n_cells), np.nan, dtype=np.float64)
    signal_mean = np.full((n_visits, n_cells), np.nan, dtype=np.float64)
    signal_max = np.full((n_visits, n_cells), np.nan, dtype=np.float64)
    signal_std = np.full((n_visits, n_cells), np.nan, dtype=np.float64)
    cycle_parts: list[np.ndarray] = []
    room_parts: list[np.ndarray] = []

    for partial_path in partial_paths:
        with np.load(partial_path) as part:
            start = int(part["visit_start"])
            end = int(part["visit_end"])
            r2[start:end] = part["r2"]
            gaussian_params[start:end] = part["gaussian_params"]
            aic[start:end] = part["aic"]
            signal_mean[start:end] = part["signal_mean"]
            signal_max[start:end] = part["signal_max"]
            signal_std[start:end] = part["signal_std"]
            if "cycle_ids" in part:
                cycle_parts.append((start, np.asarray(part["cycle_ids"])))
            if "room_ids" in part:
                room_parts.append((start, np.asarray(part["room_ids"])))

    cycle_ids = None
    room_ids = None
    if cycle_parts:
        cycle_ids = np.empty(n_visits, dtype=cycle_parts[0][1].dtype)
        for start, values in cycle_parts:
            cycle_ids[start : start + len(values)] = values
    if room_parts:
        room_ids = np.empty(n_visits, dtype=room_parts[0][1].dtype)
        for start, values in room_parts:
            room_ids[start : start + len(values)] = values

    _save_fits(
        output_path,
        r2=r2,
        gaussian_params=gaussian_params,
        aic=aic,
        signal_mean=signal_mean,
        signal_max=signal_max,
        signal_std=signal_std,
        visit_start=0,
        visit_end=n_visits,
        n_gaussians=n_gaussians,
        source_path=source_path,
        cycle_ids=cycle_ids,
        room_ids=room_ids,
    )


def _n_rooms_from_truncated_file(path: Path) -> int:
    with np.load(path, mmap_mode="r") as data:
        schedule = np.asarray(data["schedule"])
    return int(schedule.shape[1])


def run_estimate_single_file(
    input_path: Path,
    *,
    n_gaussians: int = 2,
    memory_split: int = 1,
    device: str = "cpu",
    n_processes: int = 1,
    show_progress: bool = True,
) -> Path:
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    n_visits, n_cells, _, _ = _read_ratemap_shape(input_path)
    output_path = _output_path(input_path.parent)

    if memory_split <= 1:
        ratemaps = _load_ratemaps_file(input_path)
        r2, gaussian_params, aic, signal_mean, signal_max, signal_std = fit_ratemaps(
            ratemaps,
            n_gaussians=n_gaussians,
            device=device,
            n_processes=n_processes,
            show_progress=show_progress,
        )
        del ratemaps
        cycle_ids, room_ids = _load_metadata_arrays(input_path)
        _save_fits(
            output_path,
            r2=r2,
            gaussian_params=gaussian_params,
            aic=aic,
            signal_mean=signal_mean,
            signal_max=signal_max,
            signal_std=signal_std,
            visit_start=0,
            visit_end=n_visits,
            n_gaussians=n_gaussians,
            source_path=input_path,
            cycle_ids=cycle_ids,
            room_ids=room_ids,
        )
        return output_path

    bounds = visit_chunk_bounds(n_visits, memory_split)
    partial_paths: list[Path] = []
    for part_idx, (visit_start, visit_end) in enumerate(bounds):
        if visit_start >= visit_end:
            continue
        with np.load(input_path) as data:
            ratemaps = np.asarray(data["ratemaps"][visit_start:visit_end])
        r2, gaussian_params, aic, signal_mean, signal_max, signal_std = fit_ratemaps(
            ratemaps,
            n_gaussians=n_gaussians,
            device=device,
            n_processes=n_processes,
            show_progress=show_progress,
            desc=(
                f"Gaussian RF fits part {part_idx + 1}/{memory_split} "
                f"(visits {visit_start}:{visit_end}, N={n_gaussians}, {device})"
            ),
        )
        del ratemaps

        partial_path = _partial_output_path(input_path.parent, part_idx, memory_split)
        cycle_ids, room_ids = _load_metadata_arrays(
            input_path, visit_start, visit_end
        )
        _save_fits(
            partial_path,
            r2=r2,
            gaussian_params=gaussian_params,
            aic=aic,
            signal_mean=signal_mean,
            signal_max=signal_max,
            signal_std=signal_std,
            visit_start=visit_start,
            visit_end=visit_end,
            n_gaussians=n_gaussians,
            source_path=input_path,
            cycle_ids=cycle_ids,
            room_ids=room_ids,
        )
        partial_paths.append(partial_path)
        del r2, gaussian_params, aic, signal_mean, signal_max, signal_std, cycle_ids, room_ids

    _merge_partial_files(
        partial_paths,
        n_visits=n_visits,
        n_cells=n_cells,
        n_gaussians=n_gaussians,
        output_path=output_path,
        source_path=input_path,
    )
    for partial_path in partial_paths:
        partial_path.unlink(missing_ok=True)

    return output_path


def _print_truncated_segments(
    segments: list[tuple[int, int, Path]],
    *,
    total_cycles: int,
) -> None:
    print(f"Truncated segments (cycles [0, {total_cycles})):")
    for start_cycle, end_cycle, path in segments:
        print(f"  [{start_cycle}, {end_cycle}): {path}")


def run_estimate_truncated_dir(
    truncated_dir: Path,
    *,
    total_cycles: int = DEFAULT_TOTAL_CYCLES,
    n_gaussians: int = 2,
    device: str = "cpu",
    n_processes: int = 1,
    show_progress: bool = True,
) -> Path:
    truncated_dir = Path(truncated_dir)
    segments = discover_truncated_ratemaps_series(
        truncated_dir, total_cycles=total_cycles
    )
    _print_truncated_segments(segments, total_cycles=total_cycles)
    n_rooms = _n_rooms_from_truncated_file(segments[0][2])
    n_visits = total_cycles * n_rooms
    _, n_cells, _, _ = _read_ratemap_shape(segments[0][2])
    output_path = _output_path(truncated_dir)

    partial_paths: list[Path] = []
    n_parts = len(segments)
    for part_idx, (start_cycle, end_cycle, segment_path) in enumerate(segments):
        visit_start = start_cycle * n_rooms
        visit_end = end_cycle * n_rooms
        ratemaps = _load_ratemaps_file(segment_path)
        if ratemaps.shape[0] != visit_end - visit_start:
            raise ValueError(
                f"{segment_path.name}: expected {visit_end - visit_start} visits, "
                f"got {ratemaps.shape[0]}"
            )

        r2, gaussian_params, aic, signal_mean, signal_max, signal_std = fit_ratemaps(
            ratemaps,
            n_gaussians=n_gaussians,
            device=device,
            n_processes=n_processes,
            show_progress=show_progress,
            desc=(
                f"Gaussian RF fits {segment_path.name} "
                f"(cycles {start_cycle}:{end_cycle}, N={n_gaussians}, {device})"
            ),
        )
        del ratemaps

        cycle_ids, room_ids = _load_metadata_arrays(segment_path)
        partial_path = _partial_output_path(truncated_dir, part_idx, n_parts)
        _save_fits(
            partial_path,
            r2=r2,
            gaussian_params=gaussian_params,
            aic=aic,
            signal_mean=signal_mean,
            signal_max=signal_max,
            signal_std=signal_std,
            visit_start=visit_start,
            visit_end=visit_end,
            n_gaussians=n_gaussians,
            source_path=segment_path,
            cycle_ids=cycle_ids,
            room_ids=room_ids,
        )
        partial_paths.append(partial_path)
        del r2, gaussian_params, aic, signal_mean, signal_max, signal_std, cycle_ids, room_ids

    _merge_partial_files(
        partial_paths,
        n_visits=n_visits,
        n_cells=n_cells,
        n_gaussians=n_gaussians,
        output_path=output_path,
        source_path=truncated_dir,
    )
    for partial_path in partial_paths:
        partial_path.unlink(missing_ok=True)

    return output_path


def run_estimate(
    input_path: Path,
    *,
    truncated_dir: bool = False,
    total_cycles: int = DEFAULT_TOTAL_CYCLES,
    n_gaussians: int = 2,
    memory_split: int = 1,
    device: str = "cpu",
    n_processes: int | str = "auto",
    show_progress: bool = True,
) -> Path:
    input_path = Path(input_path)
    workers = resolve_n_processes(n_processes, device=device)
    if truncated_dir:
        if not input_path.is_dir():
            raise NotADirectoryError(
                f"--truncated-dir requires --input to be a directory, got {input_path}"
            )
        if memory_split != 1:
            raise ValueError(
                "--memory-split cannot be used with --truncated-dir "
                "(each truncated NPZ is already one segment)."
            )
        return run_estimate_truncated_dir(
            input_path,
            total_cycles=total_cycles,
            n_gaussians=n_gaussians,
            device=device,
            n_processes=workers,
            show_progress=show_progress,
        )
    return run_estimate_single_file(
        input_path,
        n_gaussians=n_gaussians,
        memory_split=memory_split,
        device=device,
        n_processes=workers,
        show_progress=show_progress,
    )


def _parse_n_processes_arg(value: str) -> int | str:
    if value == "auto":
        return "auto"
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(
            f"--n-processes must be 'auto' or a positive integer, got {value!r}"
        )
    return n


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fit N-Gaussian sums to cycles rate maps (per visit, per cell).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            f"Cycles results NPZ, or directory of truncated NPZs with --truncated-dir "
            f"(default file: results/{DEFAULT_CYCLES_RESULTS_NAME})."
        ),
    )
    parser.add_argument(
        "--truncated-dir",
        action="store_true",
        dest="truncated_dir",
        help=(
            "Treat --input as a directory of cycles_ratemaps_truncated_*.npz files "
            "that contiguously cover cycles [0, total_cycles)."
        ),
    )
    parser.add_argument(
        "--total-cycles",
        type=int,
        default=DEFAULT_TOTAL_CYCLES,
        metavar="N",
        help=f"Expected final cycle index for --truncated-dir (default: {DEFAULT_TOTAL_CYCLES}).",
    )
    parser.add_argument(
        "--n-gaussians",
        type=int,
        default=2,
        metavar="N",
        help="Number of Gaussians per fit (default: 2).",
    )
    parser.add_argument(
        "--memory-split",
        type=int,
        default=1,
        metavar="K",
        help=(
            "Split visits into K chunks: reload the NPZ K times, keep only "
            "len(visits)//K visits per pass, write partial NPZs, then merge."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="cpu",
        help="cpu: scipy curve_fit; gpu: PyTorch on CUDA/MPS.",
    )
    parser.add_argument(
        "--n-processes",
        type=_parse_n_processes_arg,
        default="auto",
        metavar="N",
        help=(
            "CPU worker processes (default: auto = cpu_count() - 4, minimum 1). "
            "Ignored when --device gpu."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    args = parser.parse_args(argv)

    if args.n_gaussians < 1:
        parser.error("--n-gaussians must be >= 1")
    if args.memory_split < 1:
        parser.error("--memory-split must be >= 1")
    if args.total_cycles < 1:
        parser.error("--total-cycles must be >= 1")

    if args.truncated_dir:
        if args.input is None:
            input_path = RESULTS_DIR
        else:
            input_path = args.input
    else:
        input_path = args.input or _default_input_path()

    workers = resolve_n_processes(args.n_processes, device=args.device)
    if args.device == "cpu" and workers > 1 and not args.quiet:
        print(f"Using {workers} CPU worker processes")

    out = run_estimate(
        input_path,
        truncated_dir=args.truncated_dir,
        total_cycles=args.total_cycles,
        n_gaussians=args.n_gaussians,
        memory_split=args.memory_split,
        device=args.device,
        n_processes=args.n_processes,
        show_progress=not args.quiet,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
