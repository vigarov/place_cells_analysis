#!/usr/bin/env python3
"""
Run a room-based experiment from an input config JSON file.

The config must include a top-level `experiment_type` field
(`single_room`, `two_rooms`, or `many_rooms`) plus the usual
experiment-specific sections.

Usage::

    uv run run-experiment --config input_configs/single_room.json
    uv run run-experiment --config input_configs/single_room.json --alt-training
"""
import argparse
from dataclasses import fields
from pathlib import Path

from core.plateau_training import PlateauConfig
from core.training import run_experiment
from experiments.common.run import (
    create_experiment,
    load_experiment_config,
    run_is_complete,
)
from optimizers.defaults import build_optimizer_config


def plateau_cli_fields() -> list[tuple[str, type]]:
    """Expose every scalar `PlateauConfig` field as a `--plateau-<name>` flag."""
    out: list[tuple[str, type]] = []
    for f in fields(PlateauConfig):
        if f.type in ("bool", bool):
            out.append((f.name, lambda v: v.lower() in ("1", "true", "yes")))
        elif f.type in ("int", int):
            out.append((f.name, int))
        elif f.type in ("float", float):
            out.append((f.name, float))
        elif f.type in ("str", str):
            out.append((f.name, str))
    return out


def plateau_config_from_args(args: argparse.Namespace) -> PlateauConfig:
    overrides = {
        name: getattr(args, f"plateau_{name}")
        for name, _ in plateau_cli_fields()
        if getattr(args, f"plateau_{name}", None) is not None
    }
    return PlateauConfig(**overrides)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a room-based experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON config, e.g. input_configs/single_room.json",
    )
    parser.add_argument(
        "--progress-level",
        type=int,
        choices=[0, 1, 2, 3],
        default=None,
        help="Nested progress detail: 0=none, 1=group, 2=+room/traj, 3=+segments "
        "(default: 3 for single_room, 2 for two_rooms/many_rooms)",
    )
    parser.add_argument(
        "--alt-training",
        action="store_true",
        help=(
            "Use on-optimizer alt training: scale all gradients after backward by "
            "clipped kappa * inverse z-score from g_local_mse. Results go under "
            "<experiment_type>_altTOO/ (e.g. results/single_room_altTOO/)."
        ),
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=2.7,
        help="On-opt grad scale factor (default: 2.7; requires --alt-training)",
    )
    parser.add_argument(
        "--plateau",
        action="store_true",
        help=(
            "Use plateau training: BTSP-style one-shot place-field allocation plus "
            "per-unit consolidation. Results go under <experiment_type>_plateau/."
        ),
    )
    for name, field_type in plateau_cli_fields():
        parser.add_argument(
            f"--plateau-{name.replace('_', '-')}",
            dest=f"plateau_{name}",
            type=field_type,
            default=None,
            help=f"Override PlateauConfig.{name} (requires --plateau)",
        )
    args = parser.parse_args(argv)

    if args.kappa != 2.7 and not args.alt_training:
        raise SystemExit("--kappa requires --alt-training")
    if args.alt_training and args.plateau:
        raise SystemExit("--alt-training and --plateau are mutually exclusive")

    if not args.config.is_file():
        raise SystemExit(f"Config file not found: {args.config}")

    run_config = load_experiment_config(args.config)
    if args.plateau:
        alt_training = "plateau"
    elif args.alt_training:
        alt_training = "on_opt"
    else:
        alt_training = None
    plateau_config = plateau_config_from_args(args) if args.plateau else None

    for opt_type in run_config.optimizers:
        optimizer_config = build_optimizer_config(opt_type)
        experiment = create_experiment(
            run_config,
            optimizer_config=optimizer_config,
            alt_training=alt_training,
        )
        paths = experiment.resolve_paths()
        if run_is_complete(paths):
            print(f"Skipping {opt_type} (results exist at {paths.results_dir})")
            continue
        run_experiment(
            experiment,
            show_progress_level=args.progress_level,
            source_config_path=args.config,
            experiment_type=run_config.experiment_type,
            alt_training=alt_training,
            kappa=args.kappa,
            plateau_config=plateau_config,
        )


if __name__ == "__main__":
    main()
