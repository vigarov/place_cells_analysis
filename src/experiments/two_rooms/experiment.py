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

TWO_ROOMS_NAME = "two_rooms"

RESULTS_BASE = _ROOT_RESULTS_DIR / TWO_ROOMS_NAME
PLOTS_BASE = _ROOT_PLOTS_DIR / TWO_ROOMS_NAME
CKPT_BASE = CKPTS_DIR / TWO_ROOMS_NAME
DATA_DIR_DEFAULT = DATA_DIR / TWO_ROOMS_NAME

# must be under the `TWO_ROOMS_NAME` definition to avoid circular import
from experiments.two_rooms.config import TwoRoomsConfig, TwoRoomsRunConfig


class TwoRoomsExperiment(RoomExperiment):
    """two_rooms experiment: two activity-biased rooms, alternating visits in each room.

    Room 1's WSM signal is biased toward the top-left of the room, room 2's toward
    the bottom-right (default). 
    After warmup, training alternates `n_traj_per_room` trajectories of room 1,
    then `n_traj_per_room` room 2; the whole thing repeated `n_repetitions` times
    
    Training is done in `train_step_size_s` truncated-BPTT segments (no batching; segment by segment), 
    capturing gradient/optimizer signals and activation rate maps at every segment.
    """

    config: TwoRoomsConfig

    results_base = RESULTS_BASE
    ckpt_base = CKPT_BASE

    @property
    def name(self) -> str:
        return TWO_ROOMS_NAME

    @property
    def suffix(self) -> str:
        config = self.config
        return (
            f"{int(config.trajectories.trajectory_duration_s)}s_"
            f"warm{config.trajectories.n_warm_traj}x{int(config.warmup.warmup_step_size_s)}s_"
            f"perroom{config.trajectories.n_traj_per_room}x{int(config.training.train_step_size_s)}s_"
            f"rep{config.n_repetitions}"
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
        return "rep"

    @classmethod
    def generate(cls, config: TwoRoomsConfig, *, output_dir: Path | None = None) -> Path:
        output_dir = Path(output_dir or cls.data_dir())
        output_dir.mkdir(parents=True, exist_ok=True)

        room_cfg = config.room
        traj_cfg = config.trajectories
        room_specs: list[dict[str, Any]] = []

        for i in range(room_cfg.n_rooms):
            rel_dir = room_dir_name(i)
            room_dir = output_dir / rel_dir

            arena_map = make_square_room(
                room_cfg.room_width_cm, room_cfg.room_height, padding=DEFAULT_PADDING
            )
            traj_coord = generate_train_trajectories(
                arena_map,
                n_trajectories=config.n_total_traj_per_room,
                duration_s=traj_cfg.trajectory_duration_s,
                seed=traj_cfg.traj_seeds[i],
                speed=room_cfg.speed,
                boundary_avoidance=room_cfg.boundary_avoidance,
            )
            eval_traj_coord = generate_eval_trajectories(
                arena_map,
                n_traj=traj_cfg.n_eval_traj,
                duration_s=traj_cfg.eval_traj_duration_s,
                dt=DT,
                seed=traj_cfg.eval_traj_seeds[i],
                speed=room_cfg.speed,
                boundary_avoidance=room_cfg.boundary_avoidance,
            )
            wsm = build_wsm(
                arena_map,
                n_cells=room_cfg.n_wsm_cells,
                sigma=room_cfg.wsm_sigma,
                ssigma=room_cfg.wsm_ssigma,
                magnitude=room_cfg.wsm_magnitude,
                seed=room_cfg.wsm_seeds[i],
                bias_strength=room_cfg.bias_strength,
                bias_corner=room_cfg.bias_corners[i],
                bias_type=room_cfg.bias_type,
            )
            write_room_bundle(
                room_dir,
                arena_map=arena_map,
                traj_coord=traj_coord,
                eval_traj_coord=eval_traj_coord,
                wsm=wsm,
            )
            print_coverage(
                arena_map, traj_coord, label=f"{TWO_ROOMS_NAME} {rel_dir} train trajectories"
            )
            print_coverage(
                arena_map, eval_traj_coord, label=f"{TWO_ROOMS_NAME} {rel_dir} eval trajectories"
            )

            room_specs.append(
                {
                    "index": i,
                    "rel_dir": rel_dir,
                    "wsm_seed": room_cfg.wsm_seeds[i],
                    "traj_seed": traj_cfg.traj_seeds[i],
                    "eval_traj_seed": traj_cfg.eval_traj_seeds[i],
                    "bias_corner": room_cfg.bias_corners[i],
                }
            )
            print(
                f"Wrote two_rooms {rel_dir} (bias={room_cfg.bias_corners[i]}) to {room_dir}: "
                f"arena {arena_map.shape}, traj {traj_coord.shape}, "
                f"eval traj {eval_traj_coord.shape}, WSM {wsm.response_map.shape}"
            )

        manifest: dict[str, Any] = {
            "experiment": TWO_ROOMS_NAME,
            "n_rooms": room_cfg.n_rooms,
            "room_width_cm": room_cfg.room_width_cm,
            "room_height_cm": room_cfg.room_height,
            "dt_s": DT,
            "speed": room_cfg.speed,
            "boundary_avoidance": room_cfg.boundary_avoidance,
            "bias_strength": room_cfg.bias_strength,
            "bias_type": room_cfg.bias_type,
            "trajectory_duration_s": traj_cfg.trajectory_duration_s,
            "n_warm_traj": traj_cfg.n_warm_traj,
            "n_traj_per_room": traj_cfg.n_traj_per_room,
            "n_eval_traj": traj_cfg.n_eval_traj,
            "eval_traj_duration_s": traj_cfg.eval_traj_duration_s,
            "n_wsm_cells": room_cfg.n_wsm_cells,
            "wsm_sigma": room_cfg.wsm_sigma,
            "wsm_ssigma": room_cfg.wsm_ssigma,
            "wsm_magnitude": room_cfg.wsm_magnitude,
            "rooms": room_specs,
        }
        write_manifest(output_dir, manifest)
        return output_dir

    def build_protocol(self) -> ExpProtocol:
        traj_indices = list(range(self.config.trajectories.n_traj_per_room))
        n_rooms = self.config.room.n_rooms
        return [
            Visit(
                tag=f"rep{rep}_room{room_idx}",
                room_index=room_idx,
                traj_indices=traj_indices,
                group_tag=f"rep{rep}",
                is_last_in_group=(room_idx == n_rooms - 1),
            )
            for rep in range(self.config.n_repetitions)
            for room_idx in range(n_rooms)
        ]

    def ratemap_capture(self) -> RatemapCaptureFrequency:
        n_rooms = self.config.room.n_rooms
        return RatemapCaptureFrequency(
            capture_every_n_segments=self.config.training.capture_every_n_segments,
            activation_capture_room_indices=lambda _: list(range(n_rooms)),
        )


def run_two_rooms_experiment(
    run_config: TwoRoomsRunConfig,
    *,
    show_progress_level: int | None = None,
) -> ExperimentPaths:
    experiment = TwoRoomsExperiment(run_config.two_rooms, run_config.optimizer)
    return run_experiment(experiment, show_progress_level=show_progress_level)
