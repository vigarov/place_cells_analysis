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
from models.utils import RaeModelConfig, rae_model_config_from_dict


@dataclass
class SingleRoomRoomConfig(RoomConfig):
    wsm_seed: int = 3003


@dataclass
class SingleRoomTrajectoriesConfig(TrajectoriesConfig):
    n_train_traj: int = 15
    traj_seed: int = 13003
    eval_traj_seed: int = 23003

    @property
    def n_total_traj(self) -> int:
        return self.n_warm_traj + self.n_train_traj


@dataclass
class SingleRoomConfig(ExperimentConfig):
    """Hyperparameters for the single_room experiment."""

    room: SingleRoomRoomConfig = field(default_factory=SingleRoomRoomConfig)
    trajectories: SingleRoomTrajectoriesConfig = field(default_factory=SingleRoomTrajectoriesConfig)
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: RaeModelConfig = field(default_factory=RaeModelConfig)

    n_epochs: int = 1  # repeat the full training-trajectory pass this many times


@dataclass
class SingleRoomRunConfig:
    single_room: SingleRoomConfig
    optimizer: dict[str, Any] = field(default_factory=lambda: {"type": "adam"})


def load_single_room_experiment_config(path: Path | str) -> SingleRoomRunConfig:
    """
    Sections: `single_room` (with nested `room`, `trajectories`, `warmup`,
    `training`), `model`, `optimizer`.
    """
    # import here to avoid circular import
    from experiments.single_room.experiment import SINGLE_ROOM_NAME

    path = Path(path)
    with open(path) as f:
        config_json = json.load(f)
    config_json = require_config_dict(config_json, label="root", path=str(path))

    section = require_config_dict(
        config_json.get(SINGLE_ROOM_NAME, config_json), label=f"'{SINGLE_ROOM_NAME}'", path=str(path)
    )

    config = SingleRoomConfig(
        room=config_from_dict(SingleRoomRoomConfig, config_subsection(section, "room")),
        trajectories=config_from_dict(
            SingleRoomTrajectoriesConfig, config_subsection(section, "trajectories")
        ),
        warmup=config_from_dict(WarmupConfig, config_subsection(section, "warmup")),
        training=config_from_dict(TrainingConfig, config_subsection(section, "training")),
        **{k: section[k] for k in ("n_epochs",) if k in section},
    )
    config = replace(config, model=rae_model_config_from_dict(config_json.get("model", {})))

    optimizer = require_config_dict(
        config_json.get("optimizer", {"type": "adam"}), label="'optimizer'", path=str(path)
    )
    return SingleRoomRunConfig(single_room=config, optimizer=dict(optimizer))
