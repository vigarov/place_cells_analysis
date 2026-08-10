from pathlib import Path
from typing import Any

from core.paths import CKPTS_DIR, DATA_DIR, PLOTS_DIR as _ROOT_PLOTS_DIR, RESULTS_DIR as _ROOT_RESULTS_DIR
from core.experiment import ExperimentPaths, ExpProtocol, RoomExperiment, Visit
from core.training import run_experiment
from core.eval_trajectories import generate_eval_trajectories, print_coverage
from experiments.common.room_io import (
    build_wsm,
    generate_train_trajectories,
    write_manifest,
    write_room_bundle,
)
from trajectories.constants import DEFAULT_PADDING, DT
from trajectories.room_generator import make_square_room

SINGLE_ROOM_NAME = "single_room"

RESULTS_BASE = _ROOT_RESULTS_DIR / SINGLE_ROOM_NAME
PLOTS_BASE = _ROOT_PLOTS_DIR / SINGLE_ROOM_NAME
CKPT_BASE = CKPTS_DIR / SINGLE_ROOM_NAME
DATA_DIR_DEFAULT = DATA_DIR / SINGLE_ROOM_NAME

# must be under the `SINGLE_ROOM_NAME` definition to avoid circular import
from experiments.single_room.config import SingleRoomConfig, SingleRoomRunConfig


class SingleRoomExperiment(RoomExperiment):
    """single_room experiment: one unbiased square room, repeated passes over training trajectories.

    Longer trajectories (default 600s).

    After warmup, training runs `n_train_traj` trajectories in the single room;
    the whole pass repeated `n_epochs` times.

    Training is done in `train_step_size_s` truncated-BPTT segments (no batching; segment by segment),
    capturing gradient/optimizer signals and activation rate maps at every segment.
    """

    config: SingleRoomConfig

    results_base = RESULTS_BASE
    ckpt_base = CKPT_BASE

    @property
    def name(self) -> str:
        return SINGLE_ROOM_NAME

    @property
    def suffix(self) -> str:
        config = self.config
        return (
            f"{int(config.trajectories.trajectory_duration_s)}s_"
            f"warm{config.trajectories.n_warm_traj}x{int(config.warmup.warmup_step_size_s)}s_"
            f"train{config.trajectories.n_train_traj}x{int(config.training.train_step_size_s)}s_"
            f"ep{config.n_epochs}"
        )

    @classmethod
    def data_dir(cls) -> Path:
        return DATA_DIR_DEFAULT

    @property
    def n_rooms(self) -> int:
        return 1

    @property
    def n_warm_per_room(self) -> int:
        return self.config.trajectories.n_warm_traj

    @property
    def progress_group_unit(self) -> str:
        return "epoch"

    @property
    def default_show_progress_level(self) -> int:
        return 3

    @classmethod
    def generate(cls, config: SingleRoomConfig, *, output_dir: Path | None = None) -> Path:
        output_dir = Path(output_dir or cls.data_dir())
        output_dir.mkdir(parents=True, exist_ok=True)

        room_cfg = config.room
        traj_cfg = config.trajectories
        arena_map = make_square_room(
            room_cfg.room_width_cm, room_cfg.room_height, padding=DEFAULT_PADDING
        )

        traj_coord = generate_train_trajectories(
            arena_map,
            n_trajectories=traj_cfg.n_total_traj,
            duration_s=traj_cfg.trajectory_duration_s,
            seed=traj_cfg.traj_seed,
            speed=room_cfg.speed,
            boundary_avoidance=room_cfg.boundary_avoidance,
        )
        eval_traj_coord = generate_eval_trajectories(
            arena_map,
            n_traj=traj_cfg.n_eval_traj,
            duration_s=traj_cfg.eval_traj_duration_s,
            dt=DT,
            seed=traj_cfg.eval_traj_seed,
            speed=room_cfg.speed,
            boundary_avoidance=room_cfg.boundary_avoidance,
        )
        wsm = build_wsm(
            arena_map,
            n_cells=room_cfg.n_wsm_cells,
            sigma=room_cfg.wsm_sigma,
            ssigma=room_cfg.wsm_ssigma,
            magnitude=room_cfg.wsm_magnitude,
            seed=room_cfg.wsm_seed,
        )
        write_room_bundle(
            output_dir,
            arena_map=arena_map,
            traj_coord=traj_coord,
            eval_traj_coord=eval_traj_coord,
            wsm=wsm,
        )
        print_coverage(arena_map, traj_coord, label=f"{SINGLE_ROOM_NAME} train trajectories")
        print_coverage(arena_map, eval_traj_coord, label=f"{SINGLE_ROOM_NAME} eval trajectories")

        manifest: dict[str, Any] = {
            "experiment": SINGLE_ROOM_NAME,
            "room_width_cm": room_cfg.room_width_cm,
            "room_height_cm": room_cfg.room_height,
            "dt_s": DT,
            "speed": room_cfg.speed,
            "boundary_avoidance": room_cfg.boundary_avoidance,
            "trajectory_duration_s": traj_cfg.trajectory_duration_s,
            "n_warm_traj": traj_cfg.n_warm_traj,
            "n_train_traj": traj_cfg.n_train_traj,
            "traj_seed": traj_cfg.traj_seed,
            "n_eval_traj": traj_cfg.n_eval_traj,
            "eval_traj_duration_s": traj_cfg.eval_traj_duration_s,
            "eval_traj_seed": traj_cfg.eval_traj_seed,
            "n_wsm_cells": room_cfg.n_wsm_cells,
            "wsm_sigma": room_cfg.wsm_sigma,
            "wsm_ssigma": room_cfg.wsm_ssigma,
            "wsm_magnitude": room_cfg.wsm_magnitude,
            "wsm_seed": room_cfg.wsm_seed,
        }
        write_manifest(output_dir, manifest)
        print(
            f"Wrote single_room room to {output_dir}: arena {arena_map.shape}, "
            f"train traj {traj_coord.shape}, eval traj {eval_traj_coord.shape}, "
            f"WSM {wsm.response_map.shape}"
        )
        return output_dir

    def build_protocol(self) -> ExpProtocol:
        traj_indices = list(range(self.config.trajectories.n_train_traj))
        return [
            Visit(
                tag=f"epoch{epoch}",
                room_index=0,
                traj_indices=traj_indices,
                group_tag=f"epoch{epoch}",
                is_last_in_group=True,
            )
            for epoch in range(self.config.n_epochs)
        ]


def run_single_room_experiment(
    run_config: SingleRoomRunConfig,
    *,
    show_progress_level: int | None = None,
) -> ExperimentPaths:
    experiment = SingleRoomExperiment(run_config.single_room, run_config.optimizer)
    return run_experiment(experiment, show_progress_level=show_progress_level)
