"""Load experiment configs and construct the matching training driver."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.experiment import RoomExperiment
from experiments.common.config import ExperimentConfig, require_config_dict
from experiments.common.paths import ExperimentPaths
from experiments.many_rooms.config import load_many_rooms_experiment_config
from experiments.many_rooms.experiment import MANY_ROOMS_NAME, ManyRoomsExperiment
from experiments.single_room.config import load_single_room_experiment_config
from experiments.single_room.experiment import SINGLE_ROOM_NAME, SingleRoomExperiment
from experiments.two_rooms.config import load_two_rooms_experiment_config
from experiments.two_rooms.experiment import TWO_ROOMS_NAME, TwoRoomsExperiment
from optimizers.defaults import (
    build_optimizer_config,
    optimizer_tag_from_config,
)

EXPERIMENT_TYPES = frozenset({SINGLE_ROOM_NAME, TWO_ROOMS_NAME, MANY_ROOMS_NAME})


@dataclass
class ExperimentRunConfig:
    """Unified view of a loaded experiment config plus optimizer settings."""

    experiment_type: str
    config: ExperimentConfig
    optimizers: list[str]


def run_is_complete(paths: ExperimentPaths) -> bool:
    """Return True when a prior run finished (``final.pth`` exists)."""
    return (paths.ckpt_dir / "final.pth").is_file()


def load_experiment_config(path: Path | str) -> ExperimentRunConfig:
    """Load an input config JSON and return a unified run configuration."""
    path = Path(path)
    payload = json.loads(path.read_text())
    payload = require_config_dict(payload, label="root", path=str(path))

    experiment_type = payload.get("experiment_type")
    if not isinstance(experiment_type, str) or not experiment_type:
        raise ValueError(f"Config must include a non-empty 'experiment_type' field: {path}")
    if experiment_type not in EXPERIMENT_TYPES:
        raise ValueError(
            f"Unknown experiment_type {experiment_type!r} in {path}; "
            f"expected one of {sorted(EXPERIMENT_TYPES)}"
        )

    if experiment_type == SINGLE_ROOM_NAME:
        run_config = load_single_room_experiment_config(path)
        return ExperimentRunConfig(
            experiment_type=experiment_type,
            config=run_config.single_room,
            optimizers=run_config.optimizers,
        )
    if experiment_type == TWO_ROOMS_NAME:
        run_config = load_two_rooms_experiment_config(path)
        return ExperimentRunConfig(
            experiment_type=experiment_type,
            config=run_config.two_rooms,
            optimizers=run_config.optimizers,
        )
    run_config = load_many_rooms_experiment_config(path)
    return ExperimentRunConfig(
        experiment_type=experiment_type,
        config=run_config.many_rooms,
        optimizers=run_config.optimizers,
    )


def create_experiment(
    run_config: ExperimentRunConfig,
    *,
    optimizer_config: dict[str, Any] | None = None,
) -> RoomExperiment:
    """Instantiate the experiment driver for a loaded run configuration."""
    if optimizer_config is None:
        opt_type = run_config.optimizers[0]
        optimizer_config = build_optimizer_config(opt_type)
    optimizer_tag = optimizer_tag_from_config(optimizer_config)

    experiment_type = run_config.experiment_type
    if experiment_type == SINGLE_ROOM_NAME:
        return SingleRoomExperiment(
            run_config.config,
            optimizer_config,
            optimizer_tag=optimizer_tag,
        )
    if experiment_type == TWO_ROOMS_NAME:
        return TwoRoomsExperiment(
            run_config.config,
            optimizer_config,
            optimizer_tag=optimizer_tag,
        )
    if experiment_type == MANY_ROOMS_NAME:
        return ManyRoomsExperiment(
            run_config.config,
            optimizer_config,
            optimizer_tag=optimizer_tag,
        )
    raise ValueError(f"Unknown experiment_type: {experiment_type}")


def generate_experiment_rooms(
    run_config: ExperimentRunConfig,
    *,
    output_dir: Path | None = None,
) -> Path:
    """Generate room data for a loaded experiment configuration."""
    experiment = create_experiment(run_config)
    return experiment.generate(run_config.config, output_dir=output_dir)
