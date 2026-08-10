"""Train RAE across multi-room cycles and record rate maps (600 trials)."""
import json
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from tqdm.auto import tqdm

from experiments.old_cycles.cycles_paths import (
    CyclesPathConfig,
    RESULTS_DIR,
    ROOMS_DIR,
    resolve_cycles_paths,
)
from experiments.old_cycles.constants import STEP_SIZE
from core.utils import compute_ratemap
from models.utils import RaeModelConfig, build_rae, rae_model_config_from_dict
from core.old_training import (
    TrainMode,
    rebatch_trajectories,
    resolve_n_segments,
    train_room_visit,
)
from experiments.old_cycles.cycles_data import (
    build_visit_schedule,
    flatten_schedule,
    load_manifest,
    load_room,
    trial_steps_from_duration,
    truncate_trajectory_to_duration,
)
from experiments.old_cycles.ratemaps_io import (
    DEFAULT_CYCLES_RESULTS_NAME,
    DEFAULT_CYCLES_RESULTS_PRE_NAME,
    TRUNCATED_RATEMAPS_PREFIX,
    TRUNCATED_RATEMAPS_SUFFIX,
    discover_truncated_ratemaps_series,
    parse_truncated_ratemaps_filename,
    truncated_cycles_results_path,
)

DEFAULT_CHECKPOINT_EVERY_K_ROOMS = 10


@dataclass
class CyclesConfig(CyclesPathConfig):
    """Hyperparameters for the 20-room x 30-cycle experiment."""

    # Paper Suppl. Table 1--2 (cycles use same architecture)
    n_wsm_cells: int = 200
    n_hidden: int = 1000
    learning_rate: float = 5e-4
    lambda_mse: float = 1.0
    lambda_fr: float = 200.0
    mask_rate: float = 0.5
    step_size: int = STEP_SIZE  # EPISODIC_SEGMENT_S / DT → 1 s segments

    n_cycles: int = 30
    n_rooms: int = 20
    schedule_seed: int = 0
    mask_rng_seed: int = 0

    record_n_segments: int | None = None  # None -> derive from trajectory or use all

    # If set, train and record rate maps on the first N seconds of each visit
    # (e.g. 400 s of the 600 s trajectories from `generate-oldcycles-rooms`).
    trajectory_duration_s: float | None = None

    # Sub-divisions per trajectory for rebatching (default: 4 for `default`, 8 for `indiv_traj`).
    n_segments: int | None = None

    train_mode: TrainMode = "default"

    device: str | None = None  # resolved in run_cycles_experiment

    gradient_clip_max: float | None = None

    # When True, carry hidden state across segments (truncated BPTT); when False,
    # each segment starts from a zero hidden state (original cycles behavior).
    carry_state: bool = False

    model: RaeModelConfig = field(default_factory=RaeModelConfig)

    def resolve_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


@dataclass
class CyclesExperimentRunConfig:
    """Cycles hyperparameters plus CLI/runtime options for `cycles-experiment`."""

    cycles: CyclesConfig
    save_every_k_cycles: int = 3
    checkpoint_every_k_rooms: int = DEFAULT_CHECKPOINT_EVERY_K_ROOMS
    resume: bool = True


_CYCLES_CONFIG_FIELDS = {f.name for f in fields(CyclesConfig)}
_RUN_CONFIG_FIELDS = {"save_every_k_cycles", "checkpoint_every_k_rooms", "resume"}


def cycles_config_from_dict(raw: dict[str, Any]) -> CyclesConfig:
    """Build `CyclesConfig` from a JSON object (unknown keys are ignored)."""
    kwargs = {k: v for k, v in raw.items() if k in _CYCLES_CONFIG_FIELDS}
    return CyclesConfig(**kwargs)


def load_cycles_experiment_config(path: Path | str) -> CyclesExperimentRunConfig:
    """Load `input_configs/*.json` for the cycles training CLI."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a JSON object: {path}")

    cycles_raw = payload.get("cycles", payload)
    if not isinstance(cycles_raw, dict):
        raise ValueError(f"Config 'cycles' section must be a JSON object: {path}")

    run_raw = payload.get("run", {})
    if not isinstance(run_raw, dict):
        raise ValueError(f"Config 'run' section must be a JSON object: {path}")

    cycles = cycles_config_from_dict(cycles_raw)
    model = rae_model_config_from_dict(payload.get("model", {}))
    cycles = replace(cycles, model=model)
    run_kwargs = {k: run_raw[k] for k in _RUN_CONFIG_FIELDS if k in run_raw}
    return CyclesExperimentRunConfig(cycles=cycles, **run_kwargs)


def validate_trajectory_duration(
    duration_s: float | None,
    *,
    manifest,
    n_segments: int,
) -> None:
    if duration_s is None:
        return
    if duration_s <= 0:
        raise ValueError(f"trajectory_duration_s must be > 0, got {duration_s}")
    if duration_s > manifest.trial_duration_s:
        raise ValueError(
            f"trajectory_duration_s={duration_s} exceeds manifest trial duration "
            f"({manifest.trial_duration_s}s)."
        )
    n_steps = trial_steps_from_duration(duration_s, manifest.dt_s)
    if n_steps % n_segments != 0:
        raise ValueError(
            f"trajectory_duration_s={duration_s} → {n_steps} steps, not divisible by "
            f"n_segments={n_segments}."
        )


def resolve_record_n_segments(
    config: CyclesConfig,
    *,
    traj_timesteps: int,
) -> int:
    """
    Episodic steps to aggregate when recording rate maps.

    `traj_timesteps` must match the time axis of the tensor passed to
    `compute_ratemap` (per-row length after rebatching), not the raw visit
    length from `trajectory_duration_s`.
    """
    max_steps = traj_timesteps // config.step_size
    if config.record_n_segments is not None:
        return min(config.record_n_segments, max_steps)
    return max_steps


def _record_visit_ratemap(
    rae,
    traj_for_ratemap,
    wsm,
    arena_map,
    device,
    config: CyclesConfig,
    *,
    record_segments: int,
    mask_generator: torch.Generator,
) -> np.ndarray:
    """Compute and return a single visit rate map (float32)."""
    rae.eval()
    rm = compute_ratemap(
        rae,
        traj_for_ratemap,
        wsm,
        arena_map,
        device,
        config.mask_rate,
        config.step_size,
        n_test=record_segments,
        mask_generator=mask_generator,
        show_progress=False,
    )
    return np.asarray(rm, dtype=np.float32)


def _load_ratemaps_prefix_from_partial(
    path: Path,
    ratemaps: np.ndarray,
    start_visit: int,
) -> None:
    """Copy saved rate maps from a partial NPZ into `ratemaps[:start_visit]`."""
    with np.load(path) as data:
        n_completed = int(data["n_completed"])
        if n_completed != start_visit:
            raise ValueError(
                f"{path.name}: n_completed={n_completed} != expected start_visit={start_visit}"
            )
        saved = data["ratemaps"]
        if saved.shape[0] != n_completed:
            raise ValueError(
                f"{path.name}: ratemaps length {saved.shape[0]} != n_completed={n_completed}"
            )
        np.copyto(ratemaps[:start_visit], saved)


def _save_partial_pair(
    post_path: Path,
    pre_path: Path,
    *,
    ratemaps: np.ndarray,
    ratemaps_pre: np.ndarray,
    cycle_ids: np.ndarray,
    room_ids: np.ndarray,
    visit_indices: np.ndarray,
    schedule: np.ndarray,
    n_completed: int,
    cycles_since_last_save: int = 0,
    train_mode: TrainMode = "default",
) -> None:
    """Write post and pre partial rate-map NPZs with identical metadata."""
    meta = dict(
        cycle_ids=cycle_ids,
        room_ids=room_ids,
        visit_indices=visit_indices,
        schedule=schedule,
        n_completed=n_completed,
        cycles_since_last_save=cycles_since_last_save,
        train_mode=train_mode,
    )
    _save_partial_results(post_path, ratemaps=ratemaps, **meta)
    _save_partial_results(pre_path, ratemaps=ratemaps_pre, **meta)


def _save_final_ratemaps_pair(
    results_dir: Path,
    *,
    ratemaps: np.ndarray,
    ratemaps_pre: np.ndarray,
    cycle_ids: np.ndarray,
    room_ids: np.ndarray,
    visit_indices: np.ndarray,
    schedule: np.ndarray,
    config: CyclesConfig,
) -> None:
    """Write final post and pre rate-map NPZs."""
    payload = dict(
        cycle_ids=cycle_ids,
        room_ids=room_ids,
        visit_indices=visit_indices,
        schedule=schedule,
        n_hidden=config.n_hidden,
        n_wsm_cells=config.n_wsm_cells,
        train_mode=config.train_mode,
    )
    np.savez_compressed(
        results_dir / DEFAULT_CYCLES_RESULTS_NAME,
        ratemaps=ratemaps,
        **payload,
    )
    np.savez_compressed(
        results_dir / DEFAULT_CYCLES_RESULTS_PRE_NAME,
        ratemaps=ratemaps_pre,
        **payload,
    )


@dataclass
class CyclesResult:
    """Outputs of `run_cycles_experiment`."""

    ratemaps: np.ndarray  # (n_visits, n_hidden, H, W)
    cycle_ids: np.ndarray
    room_ids: np.ndarray
    visit_indices: np.ndarray
    schedule: np.ndarray
    config: CyclesConfig
    ratemaps_pre: np.ndarray | None = None  # (n_visits, n_hidden, H, W)


IndivComponent = Literal["model", "optim", "rng"]


@dataclass(frozen=True)
class IndivCheckpointRef:
    """Training snapshot after `ridx` rooms completed in `cycle_id`."""

    cycle_id: int
    room_id: int
    ridx: int


def validate_checkpoint_every_k_rooms(n_rooms: int, checkpoint_every_k_rooms: int) -> None:
    if checkpoint_every_k_rooms < 1:
        raise ValueError(
            f"checkpoint_every_k_rooms must be >= 1, got {checkpoint_every_k_rooms}"
        )
    if n_rooms % checkpoint_every_k_rooms != 0:
        raise ValueError(
            f"n_rooms ({n_rooms}) must be divisible by checkpoint_every_k_rooms "
            f"({checkpoint_every_k_rooms}) so each cycle ends on a checkpoint boundary."
        )


def indiv_checkpoint_path(
    indiv_dir: Path,
    ref: IndivCheckpointRef,
    component: IndivComponent,
) -> Path:
    """`indiv/c<C>/r<R>_ridx<I>_{model,optim,rng}.pth`."""
    name = f"r{ref.room_id}_ridx{ref.ridx}_{component}.pth"
    return indiv_dir / f"c{ref.cycle_id}" / name


def rooms_completed_in_cycle(visit_index: int, n_rooms: int) -> int:
    """1-based count of rooms finished in the visit's cycle (after `visit_index`)."""
    return (visit_index % n_rooms) + 1


def indiv_checkpoint_ref_after_visit(
    visit_index: int,
    room_id: int,
    cycle_id: int,
    n_rooms: int,
) -> IndivCheckpointRef:
    return IndivCheckpointRef(
        cycle_id=cycle_id,
        room_id=room_id,
        ridx=rooms_completed_in_cycle(visit_index, n_rooms),
    )


def should_save_indiv_checkpoint(
    visit_index: int,
    n_rooms: int,
    checkpoint_every_k_rooms: int,
) -> bool:
    return rooms_completed_in_cycle(visit_index, n_rooms) % checkpoint_every_k_rooms == 0


def resolve_indiv_checkpoint_for_resume(
    start_visit: int,
    *,
    cycle_ids: np.ndarray,
    room_ids: np.ndarray,
    n_rooms: int,
    checkpoint_every_k_rooms: int,
) -> IndivCheckpointRef | None:
    """
    Pick the latest per-cycle indiv snapshot at or before `start_visit`.

    Returns `None` when training should begin from freshly initialized weights
    (visit 0). Requires `n_rooms % checkpoint_every_k_rooms == 0`.
    """
    validate_checkpoint_every_k_rooms(n_rooms, checkpoint_every_k_rooms)
    if start_visit <= 0:
        return None

    last_v = start_visit - 1
    cyc = int(cycle_ids[last_v])
    rooms_done = rooms_completed_in_cycle(last_v, n_rooms)
    target_ridx = (rooms_done // checkpoint_every_k_rooms) * checkpoint_every_k_rooms

    if target_ridx == 0:
        if cyc == 0:
            return None
        prev_cyc = cyc - 1
        prev_last_v = prev_cyc * n_rooms + (n_rooms - 1)
        return IndivCheckpointRef(
            cycle_id=prev_cyc,
            room_id=int(room_ids[prev_last_v]),
            ridx=n_rooms,
        )

    ckpt_v = cyc * n_rooms + (target_ridx - 1)
    return IndivCheckpointRef(
        cycle_id=cyc,
        room_id=int(room_ids[ckpt_v]),
        ridx=target_ridx,
    )


def save_indiv_checkpoint(
    indiv_dir: Path,
    ref: IndivCheckpointRef,
    *,
    rae: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    mask_generator: torch.Generator,
) -> None:
    indiv_dir.mkdir(parents=True, exist_ok=True)
    model_path = indiv_checkpoint_path(indiv_dir, ref, "model")
    optim_path = indiv_checkpoint_path(indiv_dir, ref, "optim")
    rng_path = indiv_checkpoint_path(indiv_dir, ref, "rng")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rae": rae.state_dict()}, model_path)
    torch.save(optimizer.state_dict(), optim_path)
    torch.save({"mask_generator": mask_generator.get_state()}, rng_path)


def load_indiv_checkpoint(
    indiv_dir: Path,
    ref: IndivCheckpointRef,
    *,
    rae: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    mask_generator: torch.Generator,
    device: torch.device,
) -> None:
    model_path = indiv_checkpoint_path(indiv_dir, ref, "model")
    optim_path = indiv_checkpoint_path(indiv_dir, ref, "optim")
    rng_path = indiv_checkpoint_path(indiv_dir, ref, "rng")
    for path in (model_path, optim_path, rng_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path.name} for resume at cycle {ref.cycle_id}, "
                f"ridx {ref.ridx} (expected under {indiv_dir / f'c{ref.cycle_id}'})"
            )
    rae.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)["rae"]
    )
    optimizer.load_state_dict(torch.load(optim_path, map_location=device, weights_only=True))
    mask_generator.set_state(
        torch.load(rng_path, map_location="cpu", weights_only=True)["mask_generator"]
    )


def _load_room_maps(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _save_room_maps(
    path: Path,
    *,
    schedule: np.ndarray,
    schedule_seed: int,
    through_cycle: int,
) -> None:
    """Write permutations for cycles `0 .. through_cycle` (inclusive)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    per_cycle = [schedule[c].tolist() for c in range(through_cycle + 1)]
    payload = {
        "schedule_seed": schedule_seed,
        "n_cycles": int(schedule.shape[0]),
        "n_rooms": int(schedule.shape[1]),
        "per_cycle": per_cycle,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _maybe_flush_room_maps(
    path: Path,
    *,
    schedule: np.ndarray,
    schedule_seed: int,
    visit_index: int,
    cycle_ids: np.ndarray,
) -> None:
    """Append the current cycle's permutation once all its rooms have been visited."""
    cyc = int(cycle_ids[visit_index])
    is_last_visit = visit_index == len(cycle_ids) - 1
    cycle_done = is_last_visit or int(cycle_ids[visit_index + 1]) != cyc
    if not cycle_done:
        return
    existing = _load_room_maps(path)
    if existing is not None and len(existing.get("per_cycle", [])) > cyc:
        return
    _save_room_maps(
        path,
        schedule=schedule,
        schedule_seed=schedule_seed,
        through_cycle=cyc,
    )


def _visit_ends_cycle(visit_index: int, cycle_ids: np.ndarray, n_visits: int) -> bool:
    """True when `visit_index` is the last visit of its cycle."""
    if visit_index >= n_visits - 1:
        return True
    return int(cycle_ids[visit_index + 1]) != int(cycle_ids[visit_index])


def _resume_is_mid_cycle(start_visit: int, cycle_ids: np.ndarray, n_visits: int) -> bool:
    """
    True when `start_visit` (next visit index) lies mid-cycle.

    `start_visit` equals the number of visits already completed (`n_completed` in
    the partial checkpoint).
    """
    if start_visit == 0 or start_visit >= n_visits:
        return False
    return int(cycle_ids[start_visit - 1]) == int(cycle_ids[start_visit])


def _load_partial_for_resume(
    path: Path,
    *,
    n_visits: int,
    n_hidden: int,
    arena_shape: tuple[int, ...],
) -> tuple[
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int | None,
]:
    """
    Load a partial checkpoint and release the NPZ handle before returning.

    Allocates the full rate-map buffer first, copies the saved prefix with
    `np.copyto` inside a `with np.load(...)` block, then closes the archive.
    """
    with np.load(path) as data:
        n_completed = int(data["n_completed"])
        if n_completed > n_visits:
            raise ValueError(
                f"Partial checkpoint has n_completed={n_completed} but run has "
                f"only {n_visits} visits; check config matches the saved schedule."
            )
        saved_ratemaps = data["ratemaps"]
        if saved_ratemaps.shape[0] != n_completed:
            raise ValueError(
                f"Partial ratemaps length {saved_ratemaps.shape[0]} != "
                f"n_completed={n_completed}"
            )
        schedule = np.asarray(data["schedule"])
        cycle_ids, room_ids, visit_indices = flatten_schedule(schedule)
        if len(visit_indices) != n_visits:
            raise ValueError(
                f"Saved schedule has {len(visit_indices)} visits, expected {n_visits}"
            )
        start_visit = n_completed
        cycles_since_last_save = (
            int(data["cycles_since_last_save"])
            if "cycles_since_last_save" in data
            else None
        )
        ratemaps = np.zeros((n_visits, n_hidden, *arena_shape), dtype=np.float32)
        np.copyto(ratemaps[:start_visit], saved_ratemaps)

    return (
        start_visit,
        ratemaps,
        cycle_ids,
        room_ids,
        visit_indices,
        schedule,
        cycles_since_last_save,
    )


def _save_partial_results(
    path: Path,
    *,
    ratemaps: np.ndarray,
    cycle_ids: np.ndarray,
    room_ids: np.ndarray,
    visit_indices: np.ndarray,
    schedule: np.ndarray,
    n_completed: int,
    cycles_since_last_save: int = 0,
    train_mode: TrainMode = "default",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ratemaps=ratemaps[:n_completed],
        cycle_ids=cycle_ids[:n_completed],
        room_ids=room_ids[:n_completed],
        visit_indices=visit_indices[:n_completed],
        schedule=schedule,
        n_completed=n_completed,
        cycles_since_last_save=cycles_since_last_save,
        train_mode=train_mode,
    )


def run_cycles_experiment(
    config: CyclesConfig | None = None,
    *,
    rooms_dir: Path | None = None,
    cycles_dir: Path | None = None,
    ckpt_dir: Path | None = None,
    results_dir: Path | None = None,
    resume: bool = True,
    save_every_k_cycles: int = 3,
    checkpoint_every_k_rooms: int = DEFAULT_CHECKPOINT_EVERY_K_ROOMS,
    show_progress: bool = True,
) -> CyclesResult:
    """
    Run the full cycles protocol: train in each shuffled room visit, record after each.

    Saves `results/cycles/<suffix>/cycles_ratemaps.npz` (post-training) and
    `cycles_ratemaps_pre.npz` (pre-training), periodic training snapshots under
    snapshots under
    `ckpts/cycles/<suffix>/indiv/c<C>/r<R>_ridx<I>_{model,optim,rng}.pth` every
    `checkpoint_every_k_rooms` rooms within a cycle (requires
    `n_rooms % checkpoint_every_k_rooms == 0`), and
    `ckpts/cycles/<suffix>/room_maps.json`.

    Partial rate-map NPZs are written every `save_every_k_cycles` completed cycles
    (unchanged). Resume loads the partial rate maps to find `n_completed`, then
    restores model, optimizer, and masking RNG from the latest matching indiv snapshot
    (not `cycles_latest.pth`).
    """
    if save_every_k_cycles < 1:
        raise ValueError(f"save_every_k_cycles must be >= 1, got {save_every_k_cycles}")
    config = config or CyclesConfig()
    validate_checkpoint_every_k_rooms(config.n_rooms, checkpoint_every_k_rooms)
    device = config.resolve_device()
    paths = resolve_cycles_paths(config)
    rooms_dir = Path(rooms_dir or cycles_dir or ROOMS_DIR)
    ckpt_dir = Path(ckpt_dir or paths.ckpt_dir)
    results_dir = Path(results_dir or paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(rooms_dir / "manifest.json")
    if config.n_rooms > manifest.n_rooms:
        raise ValueError(
            f"config.n_rooms={config.n_rooms} exceeds manifest ({manifest.n_rooms} rooms)."
        )
    n_seg = resolve_n_segments(
        n_segments=config.n_segments,
        train_mode=config.train_mode,
    )
    validate_trajectory_duration(
        config.trajectory_duration_s,
        manifest=manifest,
        n_segments=n_seg,
    )

    schedule = build_visit_schedule(config.n_cycles, config.n_rooms, seed=config.schedule_seed)
    cycle_ids, room_ids, visit_indices = flatten_schedule(schedule)
    n_visits = len(visit_indices)

    # Probe one room for tensor shapes.
    arena0, traj0, wsm0, _ = load_room(1, manifest=manifest, rooms_dir=rooms_dir)
    traj0 = truncate_trajectory_to_duration(
        traj0,
        config.trajectory_duration_s,
        dt=manifest.dt_s,
    )
    traj0_rebatched = rebatch_trajectories(traj0, n_segments=n_seg)
    rae = build_rae(
        config.n_wsm_cells,
        config.n_hidden,
        device,
        model=config.model,
    )
    optimizer = torch.optim.Adam(rae.parameters(), lr=config.learning_rate)
    mask_generator = torch.Generator(device="cpu")
    mask_generator.manual_seed(config.mask_rng_seed)

    indiv_dir = ckpt_dir / "indiv"
    room_maps_path = ckpt_dir / "room_maps.json"
    partial_path = results_dir / DEFAULT_CYCLES_RESULTS_NAME
    partial_pre_path = results_dir / DEFAULT_CYCLES_RESULTS_PRE_NAME
    start_visit = 0
    ratemaps = None
    ratemaps_pre = None
    cycles_since_last_save = 0
    need_align_save = False

    if resume and partial_path.is_file():
        (
            start_visit,
            ratemaps,
            cycle_ids,
            room_ids,
            visit_indices,
            schedule,
            loaded_cycles_since_last_save,
        ) = _load_partial_for_resume(
            partial_path,
            n_visits=n_visits,
            n_hidden=config.n_hidden,
            arena_shape=arena0.shape,
        )
        ckpt_ref = resolve_indiv_checkpoint_for_resume(
            start_visit,
            cycle_ids=cycle_ids,
            room_ids=room_ids,
            n_rooms=config.n_rooms,
            checkpoint_every_k_rooms=checkpoint_every_k_rooms,
        )
        if ckpt_ref is not None:
            load_indiv_checkpoint(
                indiv_dir,
                ckpt_ref,
                rae=rae,
                optimizer=optimizer,
                mask_generator=mask_generator,
                device=device,
            )
            if show_progress:
                tqdm.write(
                    f"Loaded indiv checkpoint c{ckpt_ref.cycle_id} "
                    f"r{ckpt_ref.room_id}_ridx{ckpt_ref.ridx} "
                    f"(model, optim, rng) for visit {start_visit}/{n_visits}"
                )
        elif start_visit > 0 and show_progress:
            tqdm.write(
                f"Resuming visit {start_visit}/{n_visits} from initial weights "
                f"(no indiv snapshot before this point)"
            )
        if _resume_is_mid_cycle(start_visit, cycle_ids, n_visits):
            need_align_save = True
            cycles_since_last_save = 0
            if show_progress:
                cyc = int(cycle_ids[start_visit])
                tqdm.write(
                    f"Resuming mid-cycle {cyc} at visit {start_visit}/{n_visits} "
                    f"(will save partial at end of cycle)"
                )
        else:
            cycles_since_last_save = (
                0 if loaded_cycles_since_last_save is None else loaded_cycles_since_last_save
            )
            if show_progress:
                tqdm.write(f"Resuming from visit {start_visit}/{n_visits}")
        _maybe_flush_room_maps(
            room_maps_path,
            schedule=schedule,
            schedule_seed=config.schedule_seed,
            visit_index=start_visit - 1,
            cycle_ids=cycle_ids,
        )

    if ratemaps is None:
        ratemaps = np.zeros((n_visits, config.n_hidden, *arena0.shape), dtype=np.float32)
    if ratemaps_pre is None:
        ratemaps_pre = np.full(
            (n_visits, config.n_hidden, *arena0.shape), np.nan, dtype=np.float32
        )
    if start_visit > 0 and partial_pre_path.is_file():
        _load_ratemaps_prefix_from_partial(partial_pre_path, ratemaps_pre, start_visit)
    elif start_visit > 0 and show_progress:
        tqdm.write(
            f"No {DEFAULT_CYCLES_RESULTS_PRE_NAME} found; pre rate maps before visit "
            f"{start_visit} will remain NaN."
        )

    record_segments = resolve_record_n_segments(
        config,
        traj_timesteps=traj0_rebatched.shape[1],
    )

    visit_iter = range(start_visit, n_visits)
    if show_progress:
        visit_iter = tqdm(visit_iter, desc="Cycles visits", unit="visit")

    for v in visit_iter:
        cyc = int(cycle_ids[v])
        rid = int(room_ids[v])
        arena_map, traj_coord, wsm, _ = load_room(rid, manifest=manifest, rooms_dir=rooms_dir)
        traj_coord = truncate_trajectory_to_duration(
            traj_coord,
            config.trajectory_duration_s,
            dt=manifest.dt_s,
        )

        traj_for_ratemap = rebatch_trajectories(traj_coord, n_segments=n_seg)
        ratemaps_pre[v] = _record_visit_ratemap(
            rae,
            traj_for_ratemap,
            wsm,
            arena_map,
            device,
            config,
            record_segments=record_segments,
            mask_generator=mask_generator,
        )

        train_room_visit(
            rae, optimizer, traj_coord, wsm, device, config, mask_generator
        )

        ratemaps[v] = _record_visit_ratemap(
            rae,
            traj_for_ratemap,
            wsm,
            arena_map,
            device,
            config,
            record_segments=record_segments,
            mask_generator=mask_generator,
        )

        if should_save_indiv_checkpoint(v, config.n_rooms, checkpoint_every_k_rooms):
            ref = indiv_checkpoint_ref_after_visit(v, rid, cyc, config.n_rooms)
            save_indiv_checkpoint(
                indiv_dir,
                ref,
                rae=rae,
                optimizer=optimizer,
                mask_generator=mask_generator,
            )
        _maybe_flush_room_maps(
            room_maps_path,
            schedule=schedule,
            schedule_seed=config.schedule_seed,
            visit_index=v,
            cycle_ids=cycle_ids,
        )
        if _visit_ends_cycle(v, cycle_ids, n_visits):
            if need_align_save:
                _save_partial_pair(
                    partial_path,
                    partial_pre_path,
                    ratemaps=ratemaps,
                    ratemaps_pre=ratemaps_pre,
                    cycle_ids=cycle_ids,
                    room_ids=room_ids,
                    visit_indices=visit_indices,
                    schedule=schedule,
                    n_completed=v + 1,
                    cycles_since_last_save=0,
                    train_mode=config.train_mode,
                )
                need_align_save = False
                cycles_since_last_save = 0
                if show_progress:
                    tqdm.write(f"Saved partial results after cycle {int(cycle_ids[v])}")
            else:
                cycles_since_last_save += 1
                if cycles_since_last_save >= save_every_k_cycles:
                    _save_partial_pair(
                        partial_path,
                        partial_pre_path,
                        ratemaps=ratemaps,
                        ratemaps_pre=ratemaps_pre,
                        cycle_ids=cycle_ids,
                        room_ids=room_ids,
                        visit_indices=visit_indices,
                        schedule=schedule,
                        n_completed=v + 1,
                        cycles_since_last_save=0,
                        train_mode=config.train_mode,
                    )
                    cycles_since_last_save = 0
                    if show_progress:
                        tqdm.write(f"Saved partial results after cycle {int(cycle_ids[v])}")

    _save_final_ratemaps_pair(
        results_dir,
        ratemaps=ratemaps,
        ratemaps_pre=ratemaps_pre,
        cycle_ids=cycle_ids,
        room_ids=room_ids,
        visit_indices=visit_indices,
        schedule=schedule,
        config=config,
    )

    return CyclesResult(
        ratemaps=ratemaps,
        cycle_ids=cycle_ids,
        room_ids=room_ids,
        visit_indices=visit_indices,
        schedule=schedule,
        config=config,
        ratemaps_pre=ratemaps_pre,
    )


def truncate_cycles_results(
    end_cycle: int,
    *,
    start_cycle: int = 0,
    input_path: Path | None = None,
    output_path: Path | None = None,
    input_pre_path: Path | None = None,
    output_pre_path: Path | None = None,
) -> Path:
    """
    Write a smaller NPZ keeping visits for cycles `start_cycle .. end_cycle-1`.

    Default input: `results/cycles_ratemaps.npz`. Default output: same directory as
    input, `truncated_cycles_results_path(end_cycle, start_cycle)`.

    When a sibling `cycles_ratemaps_pre.npz` exists (or `input_pre_path` is set),
    also writes the matching truncated pre file.
    """
    if end_cycle < 1:
        raise ValueError(f"end_cycle must be >= 1, got {end_cycle}")
    if start_cycle < 0:
        raise ValueError(f"start_cycle must be >= 0, got {start_cycle}")
    if start_cycle >= end_cycle:
        raise ValueError(
            f"start_cycle ({start_cycle}) must be < end_cycle ({end_cycle})"
        )

    input_path = Path(input_path or RESULTS_DIR / DEFAULT_CYCLES_RESULTS_NAME)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    if output_path is None:
        output_path = truncated_cycles_results_path(
            end_cycle, start_cycle, results_dir=input_path.parent
        )
    else:
        output_path = Path(output_path)

    if input_pre_path is None:
        sibling_pre = input_path.parent / DEFAULT_CYCLES_RESULTS_PRE_NAME
        input_pre_path = sibling_pre if sibling_pre.is_file() else None
    elif not Path(input_pre_path).is_file():
        raise FileNotFoundError(input_pre_path)

    if output_pre_path is None and input_pre_path is not None:
        output_pre_path = truncated_cycles_results_path(
            end_cycle, start_cycle, results_dir=input_path.parent, pre=True
        )

    output_path = _truncate_ratemaps_file(
        input_path,
        output_path,
        start_cycle=start_cycle,
        end_cycle=end_cycle,
    )
    if input_pre_path is not None and output_pre_path is not None:
        _truncate_ratemaps_file(
            input_pre_path,
            output_pre_path,
            start_cycle=start_cycle,
            end_cycle=end_cycle,
        )
    return output_path


def _truncate_ratemaps_file(
    input_path: Path,
    output_path: Path,
    *,
    start_cycle: int,
    end_cycle: int,
) -> Path:
    """Slice visits for `[start_cycle, end_cycle)` from `input_path` into `output_path`."""
    with np.load(input_path) as data:
        schedule = np.asarray(data["schedule"])
        file_n_cycles, n_rooms = schedule.shape
        if end_cycle > file_n_cycles:
            raise ValueError(
                f"Requested end cycle {end_cycle} but input only has {file_n_cycles} "
                f"(schedule shape {schedule.shape})."
            )
        n_start = start_cycle * n_rooms
        n_end = end_cycle * n_rooms
        n_saved = int(data["ratemaps"].shape[0])
        if n_saved < n_end:
            raise ValueError(
                f"Input has {n_saved} visits ({n_saved // n_rooms} complete cycles); "
                f"need at least {n_end} visits for cycles [{start_cycle}, {end_cycle})."
            )

        schedule_out = schedule[start_cycle:end_cycle]
        payload: dict[str, np.ndarray | int] = {
            "ratemaps": np.asarray(data["ratemaps"][n_start:n_end]),
            "cycle_ids": np.asarray(data["cycle_ids"][n_start:n_end]),
            "room_ids": np.asarray(data["room_ids"][n_start:n_end]),
            "visit_indices": np.asarray(data["visit_indices"][n_start:n_end]),
            "schedule": schedule_out,
        }
        if "n_hidden" in data:
            payload["n_hidden"] = int(np.asarray(data["n_hidden"]))
        if "n_wsm_cells" in data:
            payload["n_wsm_cells"] = int(np.asarray(data["n_wsm_cells"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return output_path


def load_cycles_result(
    results_dir: Path | None = None,
    *,
    path: Path | str | None = None,
    config: CyclesConfig | None = None,
) -> CyclesResult:
    """
    Load experiment results from disk.

    Parameters
    ----------
    results_dir
        Directory containing `cycles_ratemaps.npz` (ignored if `path` is set).
    path
        Explicit `.npz` file, e.g. `results/cycles_ratemaps_truncated_20.npz`.
    config
        When set, resolve tagged `results/cycles/<suffix>/` from run hyperparameters.
    """
    if path is not None:
        npz_path = Path(path)
    else:
        if results_dir is None and config is not None:
            results_dir = resolve_cycles_paths(config).results_dir
        npz_path = Path(results_dir or RESULTS_DIR) / DEFAULT_CYCLES_RESULTS_NAME

    with np.load(npz_path) as data:
        ratemaps = np.asarray(data["ratemaps"])
        n_visits = ratemaps.shape[0]
        cycle_ids = np.asarray(data["cycle_ids"][:n_visits])
        room_ids = np.asarray(data["room_ids"][:n_visits])
        visit_indices = np.asarray(data["visit_indices"][:n_visits])
        schedule = np.asarray(data["schedule"])
        config_kw: dict[str, int | str] = {
            "n_cycles": int(schedule.shape[0]),
            "n_rooms": int(schedule.shape[1]),
        }
        if "n_hidden" in data:
            config_kw["n_hidden"] = int(data["n_hidden"])
        if "n_wsm_cells" in data:
            config_kw["n_wsm_cells"] = int(data["n_wsm_cells"])
        if "train_mode" in data:
            config_kw["train_mode"] = str(np.asarray(data["train_mode"]))
        config = CyclesConfig(**config_kw)

    ratemaps_pre = None
    pre_path = npz_path.parent / DEFAULT_CYCLES_RESULTS_PRE_NAME
    if pre_path.is_file() and npz_path.name == DEFAULT_CYCLES_RESULTS_NAME:
        with np.load(pre_path) as pre_data:
            ratemaps_pre = np.asarray(pre_data["ratemaps"])

    return CyclesResult(
        ratemaps=ratemaps,
        cycle_ids=cycle_ids,
        room_ids=room_ids,
        visit_indices=visit_indices,
        schedule=schedule,
        config=config,
        ratemaps_pre=ratemaps_pre,
    )
