#!/usr/bin/env python3
"""
Fit N-Gaussian receptive fields to room-experiment rate maps.

Reads per-trajectory rate-map NPZs under ``<results_dir>/ratemaps/`` produced by
``single_room``, ``two_rooms``, and ``many_rooms`` experiments. For each capture
timepoint and hidden unit, fits a sum of *n_gaussians* independent 2D Gaussians
(same routine as ``analysis.sum_gaussians``).

Writes ``gaussian_rf_fits.npz`` in the experiment results directory. The output
includes ``r2``, ``indiv_r2``, ``gaussian_params``, ``aic``, per-rate-map signal
stats, and capture metadata (epoch/cycle, trajectory, segment, tags, source files).

Usage::

    uv run estimate-gaussians-rf --input results/single_room/600s_warm15x60s_train10x10s_ep1

    uv run estimate-gaussians-rf --input results/two_rooms/<suffix> --room-idx 1

    uv run estimate-gaussians-rf --device cpu --n-processes auto --memory-split 4
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from experiments.common.ratemaps_io import (
    RatemapCaptureRef,
    discover_ratemap_captures,
    load_stacked_capture_ratemaps,
    load_trajectory_ratemaps,
    parse_capture_tag,
)
from experiments.common.run import EXPERIMENT_TYPES
from experiments.many_rooms.experiment import MANY_ROOMS_NAME
from experiments.single_room.experiment import SINGLE_ROOM_NAME
from experiments.two_rooms.experiment import TWO_ROOMS_NAME
from scripts.old_estimate_gaussians_rf import (
    _PARAM_NAMES,
    fit_ratemaps,
    resolve_n_processes,
    visit_chunk_bounds,
)

DEFAULT_OUTPUT_NAME = "gaussian_rf_fits.npz"
_RATEMAPS_SUBDIR = "ratemaps"


def _partial_output_path(base: Path, part: int, n_parts: int) -> Path:
    return base / f"gaussian_rf_fits.part{part:03d}_of_{n_parts:03d}.npz"


def _infer_experiment_from_path(path: Path) -> str | None:
    """Return the nearest ancestor directory named like a room experiment."""
    for parent in (path.resolve(), *path.resolve().parents):
        if parent.name in EXPERIMENT_TYPES:
            return parent.name
    return None


def resolve_input(
    input_path: Path,
    *,
    experiment: str | None,
    room_idx: int,
) -> tuple[Path, Path, str, int]:
    """
    Resolve ``(results_dir, ratemaps_dir, experiment_name, room_idx)``.

    Accepts either a run results directory or its ``ratemaps/`` subdirectory.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if input_path.is_file():
        raise ValueError(
            f"--input must be a results directory or ratemaps directory, got file: {input_path}"
        )

    if input_path.name == _RATEMAPS_SUBDIR:
        ratemaps_dir = input_path
        results_dir = input_path.parent
    elif (input_path / _RATEMAPS_SUBDIR).is_dir():
        results_dir = input_path
        ratemaps_dir = input_path / _RATEMAPS_SUBDIR
    else:
        raise FileNotFoundError(
            f"Expected {_RATEMAPS_SUBDIR}/ under {input_path} (or pass the ratemaps dir directly)"
        )

    experiment_name = experiment or _infer_experiment_from_path(results_dir)
    if experiment_name is None:
        raise ValueError(
            "Could not infer experiment type from --input path "
            f"({results_dir}). Pass --experiment one of {sorted(EXPERIMENT_TYPES)}."
        )
    if experiment_name not in EXPERIMENT_TYPES:
        raise ValueError(
            f"Unsupported --experiment {experiment_name!r}; "
            f"expected one of {sorted(EXPERIMENT_TYPES)}"
        )
    if experiment_name == TWO_ROOMS_NAME and room_idx not in (0, 1):
        raise ValueError(f"--room-idx must be 0 or 1 for two_rooms, got {room_idx}")
    return results_dir, ratemaps_dir, experiment_name, room_idx


def _trajectory_stem(ref: RatemapCaptureRef) -> str:
    stem = ref.path.stem
    if stem.endswith("_room0") or stem.endswith("_room1"):
        return stem.rsplit("_room", 1)[0]
    return stem


def _parse_trajectory_ids(stem: str, experiment_name: str) -> tuple[int, ...]:
    if experiment_name == SINGLE_ROOM_NAME:
        m = re.fullmatch(r"epoch(\d+)_traj(\d+)", stem)
        if m is None:
            raise ValueError(f"Unexpected single_room trajectory file stem: {stem!r}")
        return int(m.group(1)), int(m.group(2))
    if experiment_name == TWO_ROOMS_NAME:
        m = re.fullmatch(r"rep(\d+)_room(\d+)_traj(\d+)", stem)
        if m is None:
            raise ValueError(f"Unexpected two_rooms trajectory file stem: {stem!r}")
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    if experiment_name == MANY_ROOMS_NAME:
        m = re.fullmatch(r"cyc(\d+)_room(\d+)_traj(\d+)", stem)
        if m is None:
            raise ValueError(f"Unexpected many_rooms trajectory file stem: {stem!r}")
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    raise ValueError(f"Unsupported experiment: {experiment_name}")


def build_capture_metadata(
    captures: list[RatemapCaptureRef],
    *,
    experiment_name: str,
    ratemaps_dir: Path,
    room_idx: int,
    tag_cache: dict[Path, list[str]] | None = None,
) -> dict[str, np.ndarray | str]:
    """Metadata arrays aligned with capture index (same order as ``captures``)."""
    if not captures:
        raise ValueError("No captures to describe")

    tags_by_path = tag_cache if tag_cache is not None else {}
    capture_tags: list[str] = []
    segment_ids: list[int] = []
    trajectory_files: list[str] = []
    file_capture_idx: list[int] = []

    if experiment_name == SINGLE_ROOM_NAME:
        epoch_ids: list[int] = []
        traj_ids: list[int] = []
    elif experiment_name == TWO_ROOMS_NAME:
        rep_ids: list[int] = []
        visit_room_ids: list[int] = []
        traj_ids = []
    else:
        cycle_ids: list[int] = []
        room_ids: list[int] = []
        traj_ids = []

    for ref in captures:
        if ref.path not in tags_by_path:
            _, tags_by_path[ref.path] = load_trajectory_ratemaps(ref.path)
        tag = tags_by_path[ref.path][ref.capture_idx]
        stem = _trajectory_stem(ref)
        parsed = _parse_trajectory_ids(stem, experiment_name)

        capture_tags.append(tag)
        segment_ids.append(parse_capture_tag(tag))
        trajectory_files.append(str(ref.path.relative_to(ratemaps_dir)))
        file_capture_idx.append(ref.capture_idx)

        if experiment_name == SINGLE_ROOM_NAME:
            epoch_ids.append(parsed[0])
            traj_ids.append(parsed[1])
        elif experiment_name == TWO_ROOMS_NAME:
            rep_ids.append(parsed[0])
            visit_room_ids.append(parsed[1])
            traj_ids.append(parsed[2])
        else:
            cycle_ids.append(parsed[0])
            room_ids.append(parsed[1])
            traj_ids.append(parsed[2])

    metadata: dict[str, np.ndarray | str] = {
        "experiment_name": experiment_name,
        "ratemaps_dir": str(ratemaps_dir),
        "capture_tags": np.asarray(capture_tags),
        "segment_ids": np.asarray(segment_ids, dtype=np.int32),
        "trajectory_files": np.asarray(trajectory_files),
        "file_capture_idx": np.asarray(file_capture_idx, dtype=np.int32),
        "traj_ids": np.asarray(traj_ids, dtype=np.int32),
    }
    if experiment_name == SINGLE_ROOM_NAME:
        metadata["epoch_ids"] = np.asarray(epoch_ids, dtype=np.int32)
    elif experiment_name == TWO_ROOMS_NAME:
        metadata["rep_ids"] = np.asarray(rep_ids, dtype=np.int32)
        metadata["visit_room_ids"] = np.asarray(visit_room_ids, dtype=np.int32)
        metadata["room_idx"] = np.int32(room_idx)
    else:
        metadata["cycle_ids"] = np.asarray(cycle_ids, dtype=np.int32)
        metadata["room_ids"] = np.asarray(room_ids, dtype=np.int32)
    return metadata


def _slice_metadata(
    metadata: dict[str, np.ndarray | str],
    start: int,
    end: int,
) -> dict[str, np.ndarray | str]:
    sliced: dict[str, np.ndarray | str] = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            sliced[key] = value
        else:
            sliced[key] = np.asarray(value)[start:end]
    return sliced


def _save_fits(
    path: Path,
    *,
    r2: np.ndarray,
    gaussian_params: np.ndarray,
    indiv_r2: np.ndarray,
    aic: np.ndarray,
    signal_mean: np.ndarray,
    signal_max: np.ndarray,
    signal_std: np.ndarray,
    capture_start: int,
    capture_end: int,
    n_gaussians: int,
    source_path: Path,
    metadata: dict[str, np.ndarray | str],
) -> None:
    payload: dict[str, np.ndarray | int | str] = {
        "r2": r2,
        "gaussian_params": gaussian_params,
        "indiv_r2": indiv_r2,
        "aic": aic,
        "signal_mean": signal_mean,
        "signal_max": signal_max,
        "signal_std": signal_std,
        "capture_start": capture_start,
        "capture_end": capture_end,
        "n_gaussians": n_gaussians,
        "param_names": np.array(_PARAM_NAMES),
        "source_path": str(source_path),
    }
    payload.update(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _merge_partial_files(
    partial_paths: list[Path],
    *,
    n_captures: int,
    n_cells: int,
    n_gaussians: int,
    output_path: Path,
    source_path: Path,
    metadata: dict[str, np.ndarray | str],
) -> None:
    r2 = np.full((n_captures, n_cells), np.nan, dtype=np.float64)
    gaussian_params = np.full(
        (n_captures, n_cells, n_gaussians, 5),
        np.nan,
        dtype=np.float64,
    )
    indiv_r2 = np.full((n_captures, n_cells, n_gaussians), np.nan, dtype=np.float64)
    aic = np.full((n_captures, n_cells), np.nan, dtype=np.float64)
    signal_mean = np.full((n_captures, n_cells), np.nan, dtype=np.float64)
    signal_max = np.full((n_captures, n_cells), np.nan, dtype=np.float64)
    signal_std = np.full((n_captures, n_cells), np.nan, dtype=np.float64)

    for partial_path in partial_paths:
        with np.load(partial_path) as part:
            start = int(part["capture_start"])
            end = int(part["capture_end"])
            r2[start:end] = part["r2"]
            gaussian_params[start:end] = part["gaussian_params"]
            if "indiv_r2" in part:
                indiv_r2[start:end] = part["indiv_r2"]
            aic[start:end] = part["aic"]
            signal_mean[start:end] = part["signal_mean"]
            signal_max[start:end] = part["signal_max"]
            signal_std[start:end] = part["signal_std"]

    _save_fits(
        output_path,
        r2=r2,
        gaussian_params=gaussian_params,
        indiv_r2=indiv_r2,
        aic=aic,
        signal_mean=signal_mean,
        signal_max=signal_max,
        signal_std=signal_std,
        capture_start=0,
        capture_end=n_captures,
        n_gaussians=n_gaussians,
        source_path=source_path,
        metadata=metadata,
    )


def run_estimate(
    input_path: Path,
    *,
    experiment: str | None = None,
    room_idx: int = 0,
    n_gaussians: int = 2,
    memory_split: int = 1,
    device: str = "cpu",
    n_processes: int | str = "auto",
    show_progress: bool = True,
) -> Path:
    results_dir, ratemaps_dir, experiment_name, room_idx = resolve_input(
        input_path,
        experiment=experiment,
        room_idx=room_idx,
    )
    tag_cache: dict[Path, list[str]] = {}
    captures = discover_ratemap_captures(
        ratemaps_dir,
        experiment_name=experiment_name,
        room_idx=room_idx,
        show_progress=show_progress,
        tag_cache=tag_cache,
    )
    if not captures:
        raise FileNotFoundError(f"No rate map captures found under {ratemaps_dir}")

    n_captures = len(captures)
    metadata = build_capture_metadata(
        captures,
        experiment_name=experiment_name,
        ratemaps_dir=ratemaps_dir,
        room_idx=room_idx,
        tag_cache=tag_cache,
    )
    output_path = results_dir / DEFAULT_OUTPUT_NAME
    workers = resolve_n_processes(n_processes, device=device)

    if memory_split <= 1:
        ratemaps = load_stacked_capture_ratemaps(captures, show_progress=show_progress)
        r2, gaussian_params, indiv_r2, aic, signal_mean, signal_max, signal_std = fit_ratemaps(
            ratemaps,
            n_gaussians=n_gaussians,
            device=device,
            n_processes=workers,
            show_progress=show_progress,
        )
        del ratemaps
        _save_fits(
            output_path,
            r2=r2,
            gaussian_params=gaussian_params,
            indiv_r2=indiv_r2,
            aic=aic,
            signal_mean=signal_mean,
            signal_max=signal_max,
            signal_std=signal_std,
            capture_start=0,
            capture_end=n_captures,
            n_gaussians=n_gaussians,
            source_path=results_dir,
            metadata=metadata,
        )
        return output_path

    bounds = visit_chunk_bounds(n_captures, memory_split)
    partial_paths: list[Path] = []
    n_cells: int | None = None

    for part_idx, (capture_start, capture_end) in enumerate(bounds):
        if capture_start >= capture_end:
            continue
        chunk_captures = captures[capture_start:capture_end]
        ratemaps = load_stacked_capture_ratemaps(
            chunk_captures, show_progress=show_progress
        )
        if n_cells is None:
            n_cells = ratemaps.shape[1]
        r2, gaussian_params, indiv_r2, aic, signal_mean, signal_max, signal_std = fit_ratemaps(
            ratemaps,
            n_gaussians=n_gaussians,
            device=device,
            n_processes=workers,
            show_progress=show_progress,
            desc=(
                f"Gaussian RF fits part {part_idx + 1}/{memory_split} "
                f"(captures {capture_start}:{capture_end}, N={n_gaussians}, {device})"
            ),
        )
        del ratemaps

        partial_path = _partial_output_path(results_dir, part_idx, memory_split)
        _save_fits(
            partial_path,
            r2=r2,
            gaussian_params=gaussian_params,
            indiv_r2=indiv_r2,
            aic=aic,
            signal_mean=signal_mean,
            signal_max=signal_max,
            signal_std=signal_std,
            capture_start=capture_start,
            capture_end=capture_end,
            n_gaussians=n_gaussians,
            source_path=results_dir,
            metadata=_slice_metadata(metadata, capture_start, capture_end),
        )
        partial_paths.append(partial_path)
        del r2, gaussian_params, indiv_r2, aic, signal_mean, signal_max, signal_std

    if n_cells is None:
        raise RuntimeError("memory_split produced no non-empty capture chunks")

    _merge_partial_files(
        partial_paths,
        n_captures=n_captures,
        n_cells=n_cells,
        n_gaussians=n_gaussians,
        output_path=output_path,
        source_path=results_dir,
        metadata=metadata,
    )
    for partial_path in partial_paths:
        partial_path.unlink(missing_ok=True)
    return output_path


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
        description=(
            "Fit N-Gaussian sums to room-experiment rate maps "
            "(one fit per capture timepoint, per hidden unit)."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Experiment results directory (containing ratemaps/) or the ratemaps/ "
            "directory itself, e.g. results/single_room/<suffix>."
        ),
    )
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENT_TYPES),
        default=None,
        help="Experiment type (inferred from --input when omitted).",
    )
    parser.add_argument(
        "--room-idx",
        type=int,
        default=0,
        choices=(0, 1),
        help="For two_rooms: which room's rate maps to fit (default: 0).",
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
            "Split captures into K chunks: load K capture slices sequentially, "
            "write partial NPZs, then merge."
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

    workers = resolve_n_processes(args.n_processes, device=args.device)
    if args.device == "cpu" and workers > 1 and not args.quiet:
        print(f"Using {workers} CPU worker processes")

    out = run_estimate(
        args.input,
        experiment=args.experiment,
        room_idx=args.room_idx,
        n_gaussians=args.n_gaussians,
        memory_split=args.memory_split,
        device=args.device,
        n_processes=args.n_processes,
        show_progress=not args.quiet,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
