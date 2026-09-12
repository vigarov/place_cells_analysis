"""Shared types and base class for room-based experiments."""
import abc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from core.weak_sm_cell import WeakSMCell
from experiments.common.config import ExperimentConfig
from experiments.common.paths import ExperimentPaths, resolve_experiment_paths
from experiments.common.room_io import load_room_from_disk, read_manifest

__all__ = [
    "RatemapCaptureFrequency",
    "ExperimentPaths",
    "ExpProtocol",
    "Room",
    "RoomExperiment",
    "Visit",
]


@dataclass
class Room:
    """One room's arena, WSM, trajectories (warmup + main), and eval trajectories."""

    arena_map: np.ndarray
    wsm: WeakSMCell
    traj_coord: np.ndarray  # (n_warm + n_main, T, 2); first n_warm rows are for warmup
    n_warm: int
    eval_traj_coord: np.ndarray

    @property
    def main_traj(self) -> np.ndarray:
        return self.traj_coord[self.n_warm :]


@dataclass
class Visit:
    """One room visit: train on room `room_index`'s trajectories `traj_indices`."""

    tag: str
    room_index: int
    traj_indices: list[int]
    group_tag: str
    is_last_in_group: bool = False


ExpProtocol = list[Visit]


@dataclass
class RatemapCaptureFrequency:
    """When to capture activation rate maps during main training."""

    capture_before_trajectory: bool = True
    capture_every_n_segments: int = 1
    capture_after_segment: Callable[[int, int], bool] | None = None
    activation_capture_room_indices: Callable[[int], list[int]] | None = None

    def should_capture_before_trajectory(self) -> bool:
        return self.capture_before_trajectory

    def should_capture_after_segment(self, seg_idx: int, n_segments: int) -> bool:
        if self.capture_after_segment is not None:
            return self.capture_after_segment(seg_idx, n_segments)
        return (seg_idx + 1) % self.capture_every_n_segments == 0

    def rooms_for_capture(self, room_index: int) -> list[int]:
        if self.activation_capture_room_indices is not None:
            return self.activation_capture_room_indices(room_index)
        return [room_index]


class RoomExperiment(abc.ABC):
    """Base class for `single_room` / `two_rooms` / `many_rooms` experiments."""

    results_base: ClassVar[Path]
    ckpt_base: ClassVar[Path]

    def __init__(
        self,
        config: ExperimentConfig,
        optimizer_config: dict[str, Any],
        *,
        optimizer_tag: str | None = None,
        alt_training_tag: str | None = None,
    ) -> None:
        self.config = config
        self.optimizer_config = optimizer_config
        self.optimizer_tag = optimizer_tag
        self.alt_training_tag = alt_training_tag
        self.dt: float = 0.05  # overwritten by `load_rooms()` from the room manifest

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short experiment name, used for progress-bar labels and logging."""

    @property
    @abc.abstractmethod
    def suffix(self) -> str:
        """Tag using hyperparameters."""

    @classmethod
    @abc.abstractmethod
    def data_dir(cls) -> Path:
        """Default directory for generated room data."""

    @property
    @abc.abstractmethod
    def n_rooms(self) -> int:
        """Number of distinct room environments."""

    @property
    @abc.abstractmethod
    def n_warm_per_room(self) -> int:
        """Warmup trajectory count."""

    @property
    @abc.abstractmethod
    def progress_group_unit(self) -> str:
        """Label for the top-level progress bar (`epoch`, `rep`, or `cycle`)."""

    @property
    def default_show_progress_level(self) -> int:
        """Default nested progress detail (override in experiment subclasses)."""
        return 2

    @classmethod
    def load_manifest(cls, *, data_dir: Path | None = None) -> dict[str, Any]:
        return read_manifest(data_dir or cls.data_dir())

    @classmethod
    def load_room(
        cls,
        room_index: int = 0,
        *,
        data_dir: Path | None = None,
        manifest: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, WeakSMCell]:
        return load_room_from_disk(
            data_dir or cls.data_dir(),
            room_index=room_index,
            manifest=manifest,
        )

    @classmethod
    @abc.abstractmethod
    def generate(cls, config: ExperimentConfig, *, output_dir: Path | None = None) -> Path:
        """Generate and write room data under `output_dir` (default: `data_dir()`)."""

    @abc.abstractmethod
    def build_protocol(self) -> ExpProtocol:
        """Build the ordered training protocol (warmup handled through src/core/warmup.py)."""

    def load_rooms(self) -> list[Room]:
        manifest = type(self).load_manifest()
        self.dt = manifest["dt_s"]
        cls = type(self)
        rooms: list[Room] = []
        for room_index in range(self.n_rooms):
            arena_map, traj_coord, eval_traj_coord, wsm = cls.load_room(
                room_index, manifest=manifest
            )
            rooms.append(
                Room(
                    arena_map=arena_map,
                    wsm=wsm,
                    traj_coord=traj_coord,
                    n_warm=self.n_warm_per_room,
                    eval_traj_coord=eval_traj_coord,
                )
            )
        return rooms

    def ratemap_capture(self) -> RatemapCaptureFrequency:
        """Override to customize activation capture timing."""
        return RatemapCaptureFrequency(
            capture_every_n_segments=self.config.training.capture_every_n_segments,
        )

    def resolve_paths(self) -> ExperimentPaths:
        """Resolve this run's results/signals/ratemaps/checkpoint directories."""
        if self.alt_training_tag:
            family = f"{self.name}{self.alt_training_tag}"
            results_base = self.results_base.parent / family
            ckpt_base = self.ckpt_base.parent / family
        else:
            results_base = self.results_base
            ckpt_base = self.ckpt_base
        return resolve_experiment_paths(
            results_base,
            ckpt_base,
            self.suffix,
            optimizer_tag=self.optimizer_tag,
        )
