import json
from dataclasses import dataclass, field, replace
from pathlib import Path

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
from experiments.common.optimizers_config import load_optimizers_from_config_json
from models.utils import RaeModelConfig, rae_model_config_from_dict


@dataclass
class ManyRoomsRoomConfig(RoomConfig):
    n_rooms: int = 20
    base_seed: int = 3003


@dataclass
class ManyRoomsTrajectoriesConfig(TrajectoriesConfig):
    n_traj: int = 4  # training trajectories per room
    n_warm_traj: int = 20  # total, split evenly across rooms
    trajectory_duration_s: float = 200.0


@dataclass
class ManyRoomsTrainingConfig(TrainingConfig):
    train_step_size_s: float = 20.0


@dataclass
class ManyRoomsConfig(ExperimentConfig):
    """Hyperparameters for the many_rooms experiment."""

    room: ManyRoomsRoomConfig = field(default_factory=ManyRoomsRoomConfig)
    trajectories: ManyRoomsTrajectoriesConfig = field(default_factory=ManyRoomsTrajectoriesConfig)
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    training: ManyRoomsTrainingConfig = field(default_factory=ManyRoomsTrainingConfig)
    model: RaeModelConfig = field(default_factory=RaeModelConfig)

    n_cycles: int = 10  # repeat the full room-visit schedule this many times
    schedule_seed: int = 3003  # room-visit-order shuffle RNG, reused each cycle

    def __post_init__(self) -> None:
        n_rooms = self.room.n_rooms
        if self.trajectories.n_warm_traj % n_rooms != 0:
            raise ValueError(
                f"n_warm_traj ({self.trajectories.n_warm_traj}) must be divisible by n_rooms "
                f"({n_rooms}) so warmup trajectories split evenly."
            )

    @property
    def n_warm_traj_per_room(self) -> int:
        return self.trajectories.n_warm_traj // self.room.n_rooms

    @property
    def n_total_traj_per_room(self) -> int:
        return self.n_warm_traj_per_room + self.trajectories.n_traj


@dataclass
class ManyRoomsRunConfig:
    many_rooms: ManyRoomsConfig
    optimizers: list[str] = field(default_factory=lambda: ["adam"])


def load_many_rooms_experiment_config(path: Path | str) -> ManyRoomsRunConfig:
    """
    Sections: `many_rooms` (with nested `room`, `trajectories`, `warmup`,
    `training`), `model`, `optimizers`.
    """
    # import here to avoid circular import
    from experiments.many_rooms.experiment import MANY_ROOMS_NAME

    path = Path(path)
    with open(path) as f:
        config_json = json.load(f)
    config_json = require_config_dict(config_json, label="root", path=str(path))

    section = require_config_dict(
        config_json.get(MANY_ROOMS_NAME, config_json), label=f"'{MANY_ROOMS_NAME}'", path=str(path)
    )

    config = ManyRoomsConfig(
        room=config_from_dict(ManyRoomsRoomConfig, config_subsection(section, "room")),
        trajectories=config_from_dict(
            ManyRoomsTrajectoriesConfig, config_subsection(section, "trajectories")
        ),
        warmup=config_from_dict(WarmupConfig, config_subsection(section, "warmup")),
        training=config_from_dict(
            ManyRoomsTrainingConfig, config_subsection(section, "training")
        ),
        **{k: section[k] for k in ("n_cycles", "schedule_seed") if k in section},
    )
    config = replace(config, model=rae_model_config_from_dict(config_json.get("model", {})))

    optimizers = load_optimizers_from_config_json(config_json, path=path)
    return ManyRoomsRunConfig(many_rooms=config, optimizers=optimizers)
