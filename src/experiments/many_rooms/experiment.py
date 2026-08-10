from pathlib import Path
from typing import Any

from core.paths import CKPTS_DIR, DATA_DIR, PLOTS_DIR as _ROOT_PLOTS_DIR, RESULTS_DIR as _ROOT_RESULTS_DIR
from core.experiment import ExperimentPaths, ExpProtocol, RatemapCaptureFrequency, RoomExperiment, Visit
from core.training import run_experiment
from core.eval_trajectories import generate_eval_trajectories, print_coverage
from experiments.common.room_io import (
    build_wsm,
    generate_train_trajectories,
    room_dir_name,
    write_manifest,
    write_room_bundle,
)
from trajectories.constants import DEFAULT_PADDING, DT
from trajectories.room_generator import make_square_room

MANY_ROOMS_NAME = "many_rooms"

RESULTS_BASE = _ROOT_RESULTS_DIR / MANY_ROOMS_NAME
PLOTS_BASE = _ROOT_PLOTS_DIR / MANY_ROOMS_NAME
CKPT_BASE = CKPTS_DIR / MANY_ROOMS_NAME
DATA_DIR_DEFAULT = DATA_DIR / MANY_ROOMS_NAME

# must be under the `MANY_ROOMS_NAME` definition to avoid circular import
from experiments.many_rooms.config import ManyRoomsConfig, ManyRoomsRunConfig
from experiments.old_cycles.cycles_data import build_visit_schedule, flatten_schedule


class ManyRoomsExperiment(RoomExperiment):
    """many_rooms experiment: multiple random rooms, visited in shuffled ordered for many cycles.

    Rooms are generated in the same unbiased fashion as in the original `cycles` (Sup. Fig. 1&2).
    
    Shorter trajectories (default 200s). 
    
    After warmup, training alternates `n_traj` trajectories of
    each room (room order shuffled for each cycle) ; the whole thing repeated `n_cycles` times.

    Training is done in `train_step_size_s` truncated-BPTT segments (no batching; segment by segment),
    capturing gradient/optimizer signals after every segment but activation rate maps only once
    at the end of each trajectory.
    """

    config: ManyRoomsConfig

    results_base = RESULTS_BASE
    ckpt_base = CKPT_BASE

    @property
    def name(self) -> str:
        return MANY_ROOMS_NAME

    @property
    def suffix(self) -> str:
        config = self.config
        return (
            f"{int(config.trajectories.trajectory_duration_s)}s_"
            f"r{config.room.n_rooms}_"
            f"traj{config.trajectories.n_traj}x{int(config.training.train_step_size_s)}s_"
            f"cyc{config.n_cycles}"
        )

    @classmethod
    def data_dir(cls) -> Path:
        return DATA_DIR_DEFAULT

    @property
    def n_rooms(self) -> int:
        return self.config.room.n_rooms

    @property
    def n_warm_per_room(self) -> int:
        return self.config.n_warm_traj_per_room

    @property
    def progress_group_unit(self) -> str:
        return "cycle"

    @classmethod
    def generate(cls, config: ManyRoomsConfig, *, output_dir: Path | None = None) -> Path:
        output_dir = Path(output_dir or cls.data_dir())
        output_dir.mkdir(parents=True, exist_ok=True)

        room_cfg = config.room
        traj_cfg = config.trajectories
        arena_map = make_square_room(
            room_cfg.room_width_cm, room_cfg.room_height, padding=DEFAULT_PADDING
        )

        room_specs: list[dict[str, Any]] = []
        for i in range(room_cfg.n_rooms):
            rel_dir = room_dir_name(i)
            room_dir = output_dir / rel_dir

            wsm_seed = room_cfg.base_seed + i
            traj_seed = room_cfg.base_seed + 10_000 + i
            eval_traj_seed = room_cfg.base_seed + 20_000 + i

            traj_coord = generate_train_trajectories(
                arena_map,
                n_trajectories=config.n_total_traj_per_room,
                duration_s=traj_cfg.trajectory_duration_s,
                seed=traj_seed,
                speed=room_cfg.speed,
                boundary_avoidance=room_cfg.boundary_avoidance,
            )
            eval_traj_coord = generate_eval_trajectories(
                arena_map,
                n_traj=traj_cfg.n_eval_traj,
                duration_s=traj_cfg.eval_traj_duration_s,
                dt=DT,
                seed=eval_traj_seed,
                speed=room_cfg.speed,
                boundary_avoidance=room_cfg.boundary_avoidance,
            )
            wsm = build_wsm(
                arena_map,
                n_cells=room_cfg.n_wsm_cells,
                sigma=room_cfg.wsm_sigma,
                ssigma=room_cfg.wsm_ssigma,
                magnitude=room_cfg.wsm_magnitude,
                seed=wsm_seed,
            )
            write_room_bundle(
                room_dir,
                arena_map=arena_map,
                traj_coord=traj_coord,
                eval_traj_coord=eval_traj_coord,
                wsm=wsm,
            )
            print_coverage(
                arena_map, traj_coord, label=f"{MANY_ROOMS_NAME} {rel_dir} train trajectories"
            )
            print_coverage(
                arena_map, eval_traj_coord, label=f"{MANY_ROOMS_NAME} {rel_dir} eval trajectories"
            )

            room_specs.append(
                {
                    "index": i,
                    "rel_dir": rel_dir,
                    "wsm_seed": wsm_seed,
                    "traj_seed": traj_seed,
                    "eval_traj_seed": eval_traj_seed,
                }
            )
            print(
                f"  [{i + 1:2d}/{room_cfg.n_rooms}] {rel_dir}  "
                f"traj {traj_coord.shape}  eval {eval_traj_coord.shape}"
            )

        manifest: dict[str, Any] = {
            "experiment": MANY_ROOMS_NAME,
            "n_rooms": room_cfg.n_rooms,
            "room_width_cm": room_cfg.room_width_cm,
            "room_height_cm": room_cfg.room_height,
            "dt_s": DT,
            "speed": room_cfg.speed,
            "boundary_avoidance": room_cfg.boundary_avoidance,
            "trajectory_duration_s": traj_cfg.trajectory_duration_s,
            "n_warm_traj": traj_cfg.n_warm_traj,
            "n_traj": traj_cfg.n_traj,
            "n_eval_traj": traj_cfg.n_eval_traj,
            "eval_traj_duration_s": traj_cfg.eval_traj_duration_s,
            "n_wsm_cells": room_cfg.n_wsm_cells,
            "wsm_sigma": room_cfg.wsm_sigma,
            "wsm_ssigma": room_cfg.wsm_ssigma,
            "wsm_magnitude": room_cfg.wsm_magnitude,
            "base_seed": room_cfg.base_seed,
            "rooms": room_specs,
        }
        write_manifest(output_dir, manifest)
        print(f"Wrote many_rooms manifest ({room_cfg.n_rooms} rooms) to {output_dir}")
        return output_dir

    def build_protocol(self) -> ExpProtocol:
        config = self.config
        visit_order = build_visit_schedule(
            config.n_cycles, config.room.n_rooms, seed=config.schedule_seed
        )
        cycle_ids, room_ids, _ = flatten_schedule(visit_order)
        n_visits = len(cycle_ids)
        traj_indices = list(range(config.trajectories.n_traj))

        protocol: ExpProtocol = []
        for v in range(n_visits):
            cyc = int(cycle_ids[v])
            room_idx = int(room_ids[v]) - 1
            is_last_of_cycle = v == n_visits - 1 or int(cycle_ids[v + 1]) != cyc
            protocol.append(
                Visit(
                    tag=f"cyc{cyc}_room{room_idx}",
                    room_index=room_idx,
                    traj_indices=traj_indices,
                    group_tag=f"cycle{cyc}",
                    is_last_in_group=is_last_of_cycle,
                )
            )
        return protocol

    def ratemap_capture(self) -> RatemapCaptureFrequency:
        return RatemapCaptureFrequency(
            capture_before_trajectory=False,
            capture_after_segment=lambda seg_idx, n_segments: seg_idx == n_segments - 1,
        )


def run_many_rooms_experiment(
    run_config: ManyRoomsRunConfig,
    *,
    show_progress_level: int | None = None,
) -> ExperimentPaths:
    experiment = ManyRoomsExperiment(run_config.many_rooms, run_config.optimizer)
    return run_experiment(experiment, show_progress_level=show_progress_level)
