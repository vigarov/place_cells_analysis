#!/usr/bin/env python3
"""
Run a room-based experiment from an input config JSON file.

The config must include a top-level `experiment_type` field
(`single_room`, `two_rooms`, or `many_rooms`) plus the usual
experiment-specific sections.

Usage::

    uv run run-experiment --config input_configs/single_room.json
"""
import argparse
from pathlib import Path

from core.training import run_experiment
from experiments.common.run import create_experiment, load_experiment_config


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
    args = parser.parse_args(argv)

    if not args.config.is_file():
        raise SystemExit(f"Config file not found: {args.config}")

    run_config = load_experiment_config(args.config)
    run_experiment(create_experiment(run_config), show_progress_level=args.progress_level)


if __name__ == "__main__":
    main()
