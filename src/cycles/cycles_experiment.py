#!/usr/bin/env python3
"""
Run the Wang et al. (NeurIPS 2024) multi-room cycles experiment (training only).

20 rooms x 30 cycles → 600 recorded rate-map trials. Saves checkpoints and
`results/cycles_ratemaps.npz`. Use `cycles/cycles_experiment.ipynb` for
Supplemental Figures 1 & 2.

Prerequisites: generate rooms under ``data/cycles/`` with::

    uv run generate-cycles-rooms

Usage::

    uv run cycles-experiment

    uv run cycles-experiment --smoke-test

    uv run cycles-experiment --load-only
"""

from __future__ import annotations

import argparse

from cycles.cycles_data import load_manifest
from cycles.cycles_paths import CKPT_DIR, RESULTS_DIR, ROOMS_DIR
from cycles.cycles_train import (
    CyclesConfig,
    DEFAULT_CHECKPOINT_EVERY_K_ROOMS,
    load_cycles_result,
    run_cycles_experiment,
    validate_checkpoint_every_k_rooms,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train RAE across room cycles and record rate maps.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick run: 2 rooms x 2 cycles, 50 record segments.",
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
        default=3,
        metavar="K",
        help="Write partial NPZ every K completed cycles (default: 3).",
    )
    parser.add_argument(
        "--checkpoint-every-k-rooms",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY_K_ROOMS,
        metavar="K",
        help=(
            "Save model, optimizer, and masking RNG every K rooms within a cycle "
            f"(default: {DEFAULT_CHECKPOINT_EVERY_K_ROOMS}; requires n_rooms %% K == 0)."
        ),
    )
    parser.add_argument(
        "--train-mode",
        choices=["default", "indiv_traj"],
        default="default",
        help=(
            "Training batching: default (128x4 parallel, batch 512) or "
            "indiv_traj (one trajectory, 8 segments, batch 8)."
        ),
    )
    args = parser.parse_args(argv)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    print(f"Room data: {ROOMS_DIR}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Rooms in manifest: {manifest.n_rooms}, trial steps: {manifest.trial_steps}")

    if args.smoke_test:
        config = CyclesConfig(
            n_cycles=2,
            n_rooms=2,
            record_n_segments=50,
            train_mode=args.train_mode,
        )
        checkpoint_every_k_rooms = config.n_rooms
    else:
        config = CyclesConfig(
            n_cycles=30,
            n_rooms=20,
            record_n_segments=None,
            train_mode=args.train_mode,
        )
        checkpoint_every_k_rooms = args.checkpoint_every_k_rooms
    validate_checkpoint_every_k_rooms(config.n_rooms, checkpoint_every_k_rooms)
    print(f"Config: {config}")
    print(f"Indiv checkpoint every {checkpoint_every_k_rooms} room(s) per cycle")

    if args.load_only:
        result = load_cycles_result()
    else:
        result = run_cycles_experiment(
            config,
            resume=not args.no_resume,
            save_every_k_cycles=args.save_every_k_cycles,
            checkpoint_every_k_rooms=checkpoint_every_k_rooms,
        )

    print("Rate maps:", result.ratemaps.shape)
    print("Visits:", len(result.visit_indices))


if __name__ == "__main__":
    main()
