#!/usr/bin/env python3
"""
Truncate a completed cycles rate-map checkpoint to a cycle range.

Keeps all room visits for cycles `start .. end-1`. Output is written next to
the input file:

- `cycles_ratemaps_truncated_<end>.npz` when `start` is 0
- `cycles_ratemaps_truncated_<start>_<end>.npz` otherwise

Usage::

    uv run truncate-cycles-results 10

    uv run truncate-cycles-results 20 --start 10

    uv run truncate-cycles-results 30 --start 20 --input results/cycles/cycles_ratemaps.npz
"""
import argparse
from pathlib import Path

from experiments.old_cycles.cycles_paths import RESULTS_DIR
from experiments.old_cycles.cycles_train import (
    DEFAULT_CYCLES_RESULTS_NAME,
    truncate_cycles_results,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Truncate cycles_ratemaps.npz to a cycle range [start, end).",
    )
    parser.add_argument(
        "end_cycle",
        type=int,
        help="Exclusive end cycle (e.g. 20 → cycles start..19).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        dest="start_cycle",
        metavar="CYCLE",
        help="Inclusive start cycle (default: 0).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS_DIR / DEFAULT_CYCLES_RESULTS_NAME,
        help=f"Source NPZ (default: results/{DEFAULT_CYCLES_RESULTS_NAME}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination NPZ (default: truncated name in <input_dir>).",
    )
    args = parser.parse_args(argv)

    out = truncate_cycles_results(
        args.end_cycle,
        start_cycle=args.start_cycle,
        input_path=args.input,
        output_path=args.output,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
