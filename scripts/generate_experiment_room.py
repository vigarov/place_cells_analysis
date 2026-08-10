#!/usr/bin/env python3
"""
Generate room data for single_room, two_rooms, or many_rooms experiments.

The config's top-level `experiment_type` selects which generator to run.

Usage::

    uv run generate-experiment-room --config input_configs/single_room.json
    uv run generate-experiment-room --config input_configs/two_rooms.json --output data/two_rooms
"""
import argparse
from pathlib import Path

from experiments.common.run import generate_experiment_rooms, load_experiment_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate rooms/trajectories/WSMs for a room-based experiment.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON config, e.g. input_configs/single_room.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: data/<experiment_type>/)",
    )
    args = parser.parse_args(argv)

    if not args.config.is_file():
        raise SystemExit(f"Config file not found: {args.config}")

    run_config = load_experiment_config(args.config)
    generate_experiment_rooms(run_config, output_dir=args.output)


if __name__ == "__main__":
    main()
