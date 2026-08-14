"""Shared config base classes and JSON parsing for room-based experiments."""
from abc import ABC
from dataclasses import dataclass, field, fields
from typing import Any

import torch

from core.utils import MaskMethod
from models.utils import RaeModelConfig


@dataclass
class RoomConfig(ABC):
    """Common room / WSM geometry and generation parameters."""

    room_width_cm: int = 100
    room_height_cm: int | None = None  # None -> square (== room_width_cm)
    n_wsm_cells: int = 200
    wsm_sigma: int = 12
    wsm_ssigma: int = 0
    wsm_magnitude: float = 4.0
    speed: str = "fast"
    boundary_avoidance: bool = False

    @property
    def room_height(self) -> int:
        return self.room_height_cm or self.room_width_cm


@dataclass
class TrajectoriesConfig(ABC):
    """Common trajectory-generation parameters."""

    trajectory_duration_s: float = 600.0
    n_warm_traj: int = 15
    n_eval_traj: int = 128
    eval_traj_duration_s: float = 350.0


@dataclass
class WarmupConfig(ABC):
    """Per-cell Gaussian-smoothed WSM warmup (see `core.warmup`)."""

    warmup_step_size_s: float = 60.0
    warmup_gaussian_sigma: float = 15.0
    warmup_shuffle: bool = False
    warmup_shuffle_seed: int = 3003


@dataclass
class TrainingConfig(ABC):
    """Main truncated-BPTT training and capture parameters."""

    train_step_size_s: float = 10.0
    n_hidden: int = 1000
    lambda_mse: float = 1.0
    lambda_fr: float = 200.0
    mask_rate: float = 0.5
    mask_method: MaskMethod = "cell"
    mask_rng_seed: int = 3003
    init_seed: int = 3003
    gradient_clip_max: float | None = None
    carry_state: bool = False
    capture_every_n_segments: int = 1
    max_units_per_layer: int | None = None
    record_n_eval_segments: int | None = None
    device: str | None = None


@dataclass
class ExperimentConfig(ABC):
    """Base experiment config: nested room/trajectory/warmup/training sections."""

    room: RoomConfig
    trajectories: TrajectoriesConfig
    warmup: WarmupConfig
    training: TrainingConfig
    model: RaeModelConfig = field(default_factory=RaeModelConfig)

    def resolve_device(self) -> torch.device:
        if self.training.device is not None:
            return torch.device(self.training.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


def config_from_dict(cls: type, raw: dict[str, Any] | None) -> Any:
    """Build a dataclass from a JSON object (unknown keys are ignored)."""
    raw = raw or {}
    allowed = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in raw.items() if k in allowed}
    return cls(**kwargs)


def require_config_dict(raw: Any, *, label: str, path: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Config {label} must be a JSON object: {path}")
    return raw


def config_subsection(section: dict[str, Any], key: str) -> dict[str, Any]:
    raw = section.get(key)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config subsection {key!r} must be a JSON object")
    return raw
