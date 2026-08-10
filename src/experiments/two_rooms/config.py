import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from experiments.common.config import (
    ExperimentConfig,
    RoomConfig,
    TrajectoriesConfig,
    TrainingConfig,
    WarmupConfig,
    config_from_dict,
    config_subsection,
    require_config_dict,
)
from core.weak_sm_cell import VALID_BIAS_TYPES
from models.utils import RaeModelConfig, rae_model_config_from_dict


@dataclass
class TwoRoomsRoomConfig(RoomConfig):
    n_rooms: int = 2
    wsm_seeds: list[int] = field(default_factory=lambda: [3003, 3004])
    # Room 1 biased toward top_left, room 2 toward bottom_right (see
    # `core.weak_sm_cell.WeakSMCell`'s `bias_strength`/`bias_corner`/`bias_type`).
    bias_corners: list[str] = field(default_factory=lambda: ["top_left", "bottom_right"])
    bias_strength: float = 0.6
    bias_type: str = "post-noise"


@dataclass
class TwoRoomsTrajectoriesConfig(TrajectoriesConfig):
    # Total warmup trajectories, split evenly across rooms (n_warm_traj // n_rooms each).
    n_warm_traj: int = 16
    n_traj_per_room: int = 10
    traj_seeds: list[int] = field(default_factory=lambda: [13003, 13004])
    eval_traj_seeds: list[int] = field(default_factory=lambda: [23003, 23004])
    trajectory_duration_s: float = 300.0


@dataclass
class TwoRoomsTrainingConfig(TrainingConfig):
    train_step_size_s: float = 20.0


@dataclass
class TwoRoomsConfig(ExperimentConfig):
    """Hyperparameters for the two_rooms experiment."""

    room: TwoRoomsRoomConfig = field(default_factory=TwoRoomsRoomConfig)
    trajectories: TwoRoomsTrajectoriesConfig = field(default_factory=TwoRoomsTrajectoriesConfig)
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    training: TwoRoomsTrainingConfig = field(default_factory=TwoRoomsTrainingConfig)
    model: RaeModelConfig = field(default_factory=RaeModelConfig)

    # Repeat "n_traj_per_room of room 1, then n_traj_per_room of room 2" this many times.
    n_repetitions: int = 3

    def __post_init__(self) -> None:
        n_rooms = self.room.n_rooms
        if n_rooms != 2:
            raise ValueError(f"n_rooms must be 2, got {n_rooms}.")
        if self.room.bias_type not in VALID_BIAS_TYPES:
            raise ValueError(
                f"bias_type must be one of {sorted(VALID_BIAS_TYPES)}, "
                f"got {self.room.bias_type!r}"
            )
        if self.trajectories.n_warm_traj % n_rooms != 0:
            raise ValueError(
                f"n_warm_traj ({self.trajectories.n_warm_traj}) must be divisible by n_rooms "
                f"({n_rooms}) so warmup trajectories split evenly."
            )
        for name, seq in (
            ("wsm_seeds", self.room.wsm_seeds),
            ("bias_corners", self.room.bias_corners),
            ("traj_seeds", self.trajectories.traj_seeds),
            ("eval_traj_seeds", self.trajectories.eval_traj_seeds),
        ):
            if len(seq) != n_rooms:
                raise ValueError(
                    f"{name} must have exactly n_rooms={n_rooms} entries, got {len(seq)}"
                )

    @property
    def n_warm_traj_per_room(self) -> int:
        return self.trajectories.n_warm_traj // self.room.n_rooms

    @property
    def n_total_traj_per_room(self) -> int:
        return self.n_warm_traj_per_room + self.trajectories.n_traj_per_room


@dataclass
class TwoRoomsRunConfig:
    two_rooms: TwoRoomsConfig
    optimizer: dict[str, Any] = field(default_factory=lambda: {"type": "adam"})


def load_two_rooms_experiment_config(path: Path | str) -> TwoRoomsRunConfig:
    """
    Sections: `two_rooms` (with nested `room`, `trajectories`, `warmup`,
    `training`), `model`, `optimizer`.
    """
    # import here to avoid circular import
    from experiments.two_rooms.experiment import TWO_ROOMS_NAME

    path = Path(path)
    with open(path) as f:
        config_json = json.load(f)
    config_json = require_config_dict(config_json, label="root", path=str(path))

    section = require_config_dict(
        config_json.get(TWO_ROOMS_NAME, config_json), label=f"'{TWO_ROOMS_NAME}'", path=str(path)
    )

    config = TwoRoomsConfig(
        room=config_from_dict(TwoRoomsRoomConfig, config_subsection(section, "room")),
        trajectories=config_from_dict(
            TwoRoomsTrajectoriesConfig, config_subsection(section, "trajectories")
        ),
        warmup=config_from_dict(WarmupConfig, config_subsection(section, "warmup")),
        training=config_from_dict(
            TwoRoomsTrainingConfig, config_subsection(section, "training")
        ),
        **{k: section[k] for k in ("n_repetitions",) if k in section},
    )
    config = replace(config, model=rae_model_config_from_dict(config_json.get("model", {})))

    optimizer = require_config_dict(
        config_json.get("optimizer", {"type": "adam"}), label="'optimizer'", path=str(path)
    )
    return TwoRoomsRunConfig(two_rooms=config, optimizer=dict(optimizer))
