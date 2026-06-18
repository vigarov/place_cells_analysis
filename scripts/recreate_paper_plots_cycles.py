#!/usr/bin/env python3
"""
Recreate Supplemental Figures 1 & 2 from the cycles experiment notebook.

Wang et al. (NeurIPS 2024): 20 rooms × 30 cycles → 600 recorded rate-map trials.

- **Suppl. Fig. 1**: cross-correlation of population vectors (Pearson *r*), trials
  re-ordered by cycle (rooms sorted within each cycle).
- **Suppl. Fig. 2**: drift of hidden units in Room 1 across cycles.

Prerequisites: generate rooms under ``data/cycles/`` with::

    uv run generate-cycles-rooms

Training (no plotting): ``uv run cycles-experiment``.

Usage::

    uv run recreate-paper-plots-cycles

    uv run recreate-paper-plots-cycles --results results/cycles/!DUR_600s_!NSEG_8_!SS_20/cycles_ratemaps_truncated_10.npz

    uv run recreate-paper-plots-cycles --config input_configs/all_cycles.json --smoke-test
"""
import argparse
from pathlib import Path

from core.paths import ROOT
from cycles.cycles_data import load_manifest
from cycles.cycles_paths import ROOMS_DIR, resolve_cycles_paths
from cycles.cycles_train import (
    CyclesConfig,
    load_cycles_experiment_config,
    load_cycles_result,
)
from analysis.cycles_analysis import (
    pearson_correlation_matrix,
    population_vectors_from_ratemaps,
    ratemaps_for_room_across_cycles,
    reorder_trials_by_cycle,
    select_drift_cells,
)
from analysis.cycles_plots import (
    plot_suppl_fig1_correlation,
    plot_suppl_fig2_drift,
    save_figure,
)

DEFAULT_CONFIG_PATH = ROOT / "input_configs" / "all_cycles.json"
DEFAULT_TRUNCATED_RESULTS_NAME = "cycles_ratemaps_truncated_10.npz"

# Notebook defaults (`src/cycles/notebook/cycles_experiment.ipynb`).
NOTEBOOK_CYCLES = CyclesConfig(
    n_cycles=30,
    n_rooms=20,
    record_n_segments=None,
    train_mode="indiv_traj",
    trajectory_duration_s=600.0,
    n_segments=8,
    step_size=20,
)

SMOKE_CYCLES = CyclesConfig(
    n_cycles=2,
    n_rooms=2,
    record_n_segments=50,
    train_mode="indiv_traj",
    trajectory_duration_s=600.0,
    n_segments=8,
    step_size=20,
)


def _resolve_config(args: argparse.Namespace) -> CyclesConfig:
    if args.smoke_test:
        return SMOKE_CYCLES
    if args.config is not None:
        if not args.config.is_file():
            raise SystemExit(f"Config file not found: {args.config}")
        return load_cycles_experiment_config(args.config).cycles
    return NOTEBOOK_CYCLES


def _resolve_results_path(args: argparse.Namespace, results_dir: Path) -> Path:
    if args.results is not None:
        path = Path(args.results)
        if not path.is_file():
            raise SystemExit(f"Results file not found: {path}")
        return path
    path = results_dir / DEFAULT_TRUNCATED_RESULTS_NAME
    if not path.is_file():
        raise SystemExit(
            f"Results file not found: {path}\n"
            "Pass --results to an existing .npz (e.g. cycles_ratemaps.npz)."
        )
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Recreate cycles experiment Supplemental Figures 1 & 2.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            f"JSON config for path resolution (default: notebook preset, "
            f"not {DEFAULT_CONFIG_PATH})."
        ),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help=(
            "Explicit .npz results file "
            f"(default: <results_dir>/{DEFAULT_TRUNCATED_RESULTS_NAME})."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick run: 2 rooms × 2 cycles (overrides --config).",
    )
    parser.add_argument(
        "--drift-room",
        type=int,
        default=1,
        metavar="N",
        help="Room index for Supplemental Figure 2 (default: 1).",
    )
    parser.add_argument(
        "--n-drift-cells",
        type=int,
        default=10,
        metavar="N",
        help="Number of units to show in Supplemental Figure 2 (default: 10).",
    )
    parser.add_argument(
        "--drift-seed",
        type=int,
        default=0,
        help="RNG seed for drift-unit selection (default: 0).",
    )
    args = parser.parse_args(argv)

    config = _resolve_config(args)
    paths = resolve_cycles_paths(config)
    paths.plots_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    results_path = _resolve_results_path(args, paths.results_dir)

    print(f"Room data: {ROOMS_DIR}")
    print(f"Results: {results_path}")
    print(f"Plots: {paths.plots_dir}")
    print(f"Rooms in manifest: {manifest.n_rooms}, trial steps: {manifest.trial_steps}")
    print(f"Config: {config}")

    result = load_cycles_result(path=results_path)
    print("Rate maps:", result.ratemaps.shape)
    print("Visits:", len(result.visit_indices))

    rm_sorted, cyc_sorted, _room_sorted = reorder_trials_by_cycle(
        result.ratemaps,
        result.cycle_ids,
        result.room_ids,
    )
    pop_vecs = population_vectors_from_ratemaps(
        rm_sorted,
        room_size_cm=manifest.room_width_cm,
    )
    corr = pearson_correlation_matrix(pop_vecs)

    fig1_path = paths.plots_dir / "suppl_fig1_correlation.pdf"
    fig1 = plot_suppl_fig1_correlation(corr, cycle_ids=cyc_sorted)
    save_figure(fig1, fig1_path)
    print(f"Wrote {fig1_path}")

    rm_room, cycles_present = ratemaps_for_room_across_cycles(
        result.ratemaps,
        result.cycle_ids,
        result.room_ids,
        args.drift_room,
    )
    cell_ids = select_drift_cells(
        rm_room,
        n_cells=args.n_drift_cells,
        seed=args.drift_seed,
    )
    print("Selected units:", cell_ids)

    fig2_path = paths.plots_dir / "suppl_fig2_room1_drift.pdf"
    fig2 = plot_suppl_fig2_drift(
        rm_room,
        cell_ids,
        cycles=cycles_present,
        room_label=f"Room {args.drift_room}",
    )
    save_figure(fig2, fig2_path)
    print(f"Wrote {fig2_path}")


if __name__ == "__main__":
    main()
