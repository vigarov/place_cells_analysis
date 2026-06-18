#!/usr/bin/env python3
"""
Run the Wang et al. (NeurIPS 2024) multi-room cycles experiment (training only).

20 rooms x 30 cycles → 600 recorded rate-map trials. Saves checkpoints and
`results/cycles/cycles_ratemaps.npz`. Use `src/cycles/notebook/cycles_experiment.ipynb`
for Supplemental Figures 1 & 2.

Prerequisites: generate rooms under `data/cycles/` with::

    uv run generate-cycles-rooms

Usage::

    uv run cycles-experiment

    uv run cycles-experiment --config input_configs/all_cycles.json

    uv run cycles-experiment --smoke-test

    uv run cycles-experiment --load-only
"""
import argparse
from pathlib import Path

from core.paths import ROOT
from cycles.cycles_data import load_manifest
from cycles.cycles_paths import ROOMS_DIR, resolve_cycles_paths
from cycles.cycles_train import (
    CyclesConfig,
    CyclesExperimentRunConfig,
    DEFAULT_CHECKPOINT_EVERY_K_ROOMS,
    load_cycles_experiment_config,
    load_cycles_result,
    run_cycles_experiment,
    validate_checkpoint_every_k_rooms,
)

DEFAULT_CONFIG_PATH = ROOT / "input_configs" / "all_cycles.json"

SMOKE_RUN = CyclesExperimentRunConfig(
    cycles=CyclesConfig(
        n_cycles=2,
        n_rooms=2,
        record_n_segments=50,
    ),
    checkpoint_every_k_rooms=2,
)


def _apply_cli_overrides(
    run_config: CyclesExperimentRunConfig,
    args: argparse.Namespace,
) -> CyclesExperimentRunConfig:
    cycles = run_config.cycles
    if args.train_mode is not None:
        cycles = CyclesConfig(**{**cycles.__dict__, "train_mode": args.train_mode})
    return CyclesExperimentRunConfig(
        cycles=cycles,
        save_every_k_cycles=args.save_every_k_cycles,
        checkpoint_every_k_rooms=args.checkpoint_every_k_rooms,
        resume=run_config.resume and not args.no_resume,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train RAE across room cycles and record rate maps.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"JSON config for CyclesConfig and run options (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick run: 2 rooms x 2 cycles, 50 record segments (overrides config).",
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Skip training; load saved results from disk.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore partial results and start from visit 0.",
    )
    parser.add_argument(
        "--save-every-k-cycles",
        type=int,
        default=None,
        metavar="K",
        help="Write partial NPZ every K completed cycles (default: from config or 3).",
    )
    parser.add_argument(
        "--checkpoint-every-k-rooms",
        type=int,
        default=None,
        metavar="K",
        help=(
            "Save model, optimizer, and masking RNG every K rooms within a cycle "
            f"(default: from config or {DEFAULT_CHECKPOINT_EVERY_K_ROOMS}; "
            "requires n_rooms %% K == 0)."
        ),
    )
    parser.add_argument(
        "--train-mode",
        choices=["default", "indiv_traj"],
        default=None,
        help=(
            "Training batching: default (128x4 parallel, batch 512) or "
            "indiv_traj (one trajectory, 8 segments, batch 8)."
        ),
    )
    args = parser.parse_args(argv)

    if args.smoke_test:
        run_config = SMOKE_RUN
        config_label = "(smoke-test preset)"
    else:
        if not args.config.is_file():
            raise SystemExit(f"Config file not found: {args.config}")
        run_config = load_cycles_experiment_config(args.config)
        config_label = str(args.config)

    if args.save_every_k_cycles is None:
        args.save_every_k_cycles = run_config.save_every_k_cycles
    if args.checkpoint_every_k_rooms is None:
        args.checkpoint_every_k_rooms = run_config.checkpoint_every_k_rooms

    run_config = _apply_cli_overrides(run_config, args)
    config = run_config.cycles
    paths = resolve_cycles_paths(config)
    validate_checkpoint_every_k_rooms(config.n_rooms, args.checkpoint_every_k_rooms)

    paths.ckpt_dir.mkdir(parents=True, exist_ok=True)
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    paths.plots_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    print(f"Config: {config_label}")
    print(f"Room data: {ROOMS_DIR}")
    print(f"Checkpoints: {paths.ckpt_dir}")
    print(f"Results: {paths.results_dir}")
    print(f"Plots: {paths.plots_dir}")
    print(f"Rooms in manifest: {manifest.n_rooms}, trial steps: {manifest.trial_steps}")
    print(f"Config: {config}")
    print(f"Indiv checkpoint every {args.checkpoint_every_k_rooms} room(s) per cycle")

    if args.load_only:
        result = load_cycles_result(results_dir=paths.results_dir, config=config)
    else:
        result = run_cycles_experiment(
            config,
            results_dir=paths.results_dir,
            resume=run_config.resume,
            save_every_k_cycles=args.save_every_k_cycles,
            checkpoint_every_k_rooms=args.checkpoint_every_k_rooms,
        )

    print("Rate maps:", result.ratemaps.shape)
    print("Visits:", len(result.visit_indices))


if __name__ == "__main__":
    main()
