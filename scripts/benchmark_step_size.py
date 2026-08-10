#!/usr/bin/env python3
"""
Benchmark RAE training time vs BPTT step size on the first three cycles rooms.

Trains with `indiv_traj` mode (same protocol as `cycles_experiment.ipynb`) on
rooms 1-3 in fixed order, sweeping step sizes suitable for plotting
`train_time` vs `step_size`. Large step sizes may OOM; those runs are logged
and skipped.

Usage
-----
    uv run benchmark-step-size
    uv run python -m scripts.benchmark_step_size --output results/cycles/step_size_benchmark.json
"""
import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from core.old_training import train_room_visit
from experiments.old_cycles.constants import STEP_SIZE
from experiments.old_cycles.cycles_data import load_manifest, load_room
from experiments.old_cycles.cycles_paths import RESULTS_DIR, ROOMS_DIR
from experiments.old_cycles.cycles_train import CyclesConfig
from models.utils import build_rae

# 75 s segment length → 75 * (1 s / STEP_SIZE) timesteps at full BPTT
STEP_SIZE_GRID = np.r_[1, 20 * np.linspace(1, 75, 6)]
DEFAULT_ROOM_IDS = (1, 2, 3)
DEFAULT_OUTPUT = RESULTS_DIR / "step_size_benchmark.json"


@dataclass
class StepSizeResult:
    step_size: int
    train_time_s: float | None
    status: str  # "ok" | "oom" | "error"
    error: str | None = None


def _clear_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _train_three_rooms(
    *,
    step_size: int,
    config: CyclesConfig,
    device: torch.device,
    rooms_dir: Path,
    room_ids: tuple[int, ...],
) -> None:
    """Fresh weights; train one visit per room with `indiv_traj` batching."""
    manifest = load_manifest(rooms_dir / "manifest.json")
    rae = build_rae(config.n_wsm_cells, config.n_hidden, device)
    optimizer = torch.optim.Adam(rae.parameters(), lr=config.learning_rate)
    mask_generator = torch.Generator(device="cpu")
    mask_generator.manual_seed(config.mask_rng_seed)

    train_config = CyclesConfig(
        n_wsm_cells=config.n_wsm_cells,
        n_hidden=config.n_hidden,
        learning_rate=config.learning_rate,
        lambda_mse=config.lambda_mse,
        lambda_fr=config.lambda_fr,
        mask_rate=config.mask_rate,
        step_size=step_size,
        train_mode="indiv_traj",
        mask_rng_seed=config.mask_rng_seed,
    )

    for room_id in tqdm(room_ids, desc=f"step_size={step_size}", leave=False):
        _arena, traj_coord, wsm, _ = load_room(
            room_id, manifest=manifest, rooms_dir=rooms_dir
        )
        train_room_visit(
            rae,
            optimizer,
            traj_coord,
            wsm,
            device,
            train_config,
            mask_generator,
        )


def benchmark_step_sizes(
    *,
    step_sizes: np.ndarray | None = None,
    room_ids: tuple[int, ...] = DEFAULT_ROOM_IDS,
    rooms_dir: Path | None = None,
    device: str | None = None,
    config: CyclesConfig | None = None,
) -> list[StepSizeResult]:
    """Sweep `step_size` and record wall-clock training time per setting."""
    step_sizes = np.asarray(step_sizes if step_sizes is not None else STEP_SIZE_GRID)
    config = config or CyclesConfig(
        n_cycles=1,
        n_rooms=len(room_ids),
        train_mode="indiv_traj",
    )
    rooms_dir = Path(rooms_dir or ROOMS_DIR)
    manifest_path = rooms_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Generate rooms with: uv run generate-oldcycles-rooms"
        )

    manifest = load_manifest(manifest_path)
    for room_id in room_ids:
        if room_id < 1 or room_id > manifest.n_rooms:
            raise ValueError(
                f"room_id {room_id} out of range (manifest has {manifest.n_rooms} rooms)"
            )

    dev = torch.device(device) if device is not None else config.resolve_device()
    results: list[StepSizeResult] = []

    for raw_step in tqdm(step_sizes, desc="Step sizes", unit="run"):
        step_size = int(round(float(raw_step)))
        _clear_device_cache(dev)
        t0 = time.perf_counter()
        try:
            _train_three_rooms(
                step_size=step_size,
                config=config,
                device=dev,
                rooms_dir=rooms_dir,
                room_ids=room_ids,
            )
            elapsed = time.perf_counter() - t0
            results.append(
                StepSizeResult(step_size=step_size, train_time_s=elapsed, status="ok")
            )
            tqdm.write(f"step_size={step_size:4d}  train_time={elapsed:.2f}s  ok")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            if _is_oom(exc):
                results.append(
                    StepSizeResult(
                        step_size=step_size,
                        train_time_s=None,
                        status="oom",
                        error=str(exc),
                    )
                )
                tqdm.write(f"step_size={step_size:4d}  OOM after {elapsed:.2f}s — skipped")
            else:
                results.append(
                    StepSizeResult(
                        step_size=step_size,
                        train_time_s=None,
                        status="error",
                        error=str(exc),
                    )
                )
                tqdm.write(f"step_size={step_size:4d}  error: {exc}")
        finally:
            _clear_device_cache(dev)

    return results


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark indiv_traj training time on the first three cycles rooms "
            "across BPTT step sizes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--rooms-dir",
        type=Path,
        default=ROOMS_DIR,
        help="Directory containing manifest.json and room_* folders.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON file for benchmark results (step_size vs train_time).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda > mps > cpu).",
    )
    p.add_argument(
        "--paper-step-size",
        type=int,
        default=STEP_SIZE,
        metavar="N",
        help="Reference step size from the paper (1 s segments); stored in metadata.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config = CyclesConfig(
        n_cycles=1,
        n_rooms=len(DEFAULT_ROOM_IDS),
        train_mode="indiv_traj",
    )

    results = benchmark_step_sizes(
        room_ids=DEFAULT_ROOM_IDS,
        rooms_dir=args.rooms_dir,
        device=args.device,
        config=config,
    )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(config.resolve_device() if args.device is None else torch.device(args.device)),
        "train_mode": "indiv_traj",
        "room_ids": list(DEFAULT_ROOM_IDS),
        "rooms_dir": str(args.rooms_dir.resolve()),
        "paper_step_size": args.paper_step_size,
        "step_size_grid": STEP_SIZE_GRID.tolist(),
        "results": [asdict(r) for r in results],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\nWrote {out_path}")
    print("Summary (step_size → train_time_s):")
    for r in results:
        if r.status == "ok":
            print(f"  {r.step_size:4d}  {r.train_time_s:.2f}s")
        else:
            print(f"  {r.step_size:4d}  [{r.status}]")


if __name__ == "__main__":
    main()
