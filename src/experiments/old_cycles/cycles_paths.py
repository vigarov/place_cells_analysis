"""Paths for the cycles experiment (code vs room data)."""
import abc
from dataclasses import dataclass
from pathlib import Path

from core.paths import (
    CKPTS_DIR,
    DATA_DIR,
    PLOTS_DIR as _ROOT_PLOTS_DIR,
    RESULTS_DIR as _ROOT_RESULTS_DIR,
)
from core.old_training import TrainMode, resolve_n_segments
from experiments.old_cycles.constants import CYCLES_TRIAL_DURATION_S, STEP_SIZE

ROOMS_DIR = DATA_DIR / "cycles"
MANIFEST_PATH = ROOMS_DIR / "manifest.json"

# Backward-compatible alias (room data directory)
CYCLES_DIR = ROOMS_DIR

CYCLES_SUFFIX_TEMPLATE = "!DUR_!NSEG_!SS"
CYCLES_RESULTS_BASE = _ROOT_RESULTS_DIR / "cycles"
CYCLES_PLOTS_BASE = _ROOT_PLOTS_DIR / "cycles"
CYCLES_CKPT_BASE = CKPTS_DIR / "cycles"

# Unresolved template paths (use `resolve_cycles_paths` at runtime).
RESULTS_DIR = CYCLES_RESULTS_BASE / CYCLES_SUFFIX_TEMPLATE
PLOTS_DIR = CYCLES_PLOTS_BASE / CYCLES_SUFFIX_TEMPLATE
CKPT_DIR = CYCLES_CKPT_BASE / CYCLES_SUFFIX_TEMPLATE
INDIV_CKPT_DIR = CKPT_DIR / "indiv"
ROOM_MAPS_PATH = CKPT_DIR / "room_maps.json"
GAUSSIAN_EVOLUTION_PLOTS_DIR = PLOTS_DIR / "gaussian_evolution"


class CyclesPathConfig(abc.ABC):
    """Fields needed to resolve cycles experiment output paths."""

    @property
    @abc.abstractmethod
    def trajectory_duration_s(self) -> float | None: ...

    @property
    @abc.abstractmethod
    def n_segments(self) -> int | None: ...

    @property
    @abc.abstractmethod
    def step_size(self) -> int: ...

    @property
    @abc.abstractmethod
    def train_mode(self) -> TrainMode: ...


@dataclass(frozen=True)
class CyclesPaths:
    """Resolved results/plots/checkpoint directories for one cycles run configuration."""

    suffix: str
    results_dir: Path
    plots_dir: Path
    gaussian_evolution_plots_dir: Path
    ckpt_dir: Path
    indiv_ckpt_dir: Path
    room_maps_path: Path


def resolve_cycles_suffix(
    *,
    trajectory_duration_s: float | None = None,
    n_segments: int | None = None,
    step_size: int | None = None,
    train_mode: TrainMode = "default",
) -> str:
    """
    Expand `!DUR_!NSEG_!SS` using run hyperparameters.

    Defaults: 600 s duration, 4/8 segments by train mode, step size 20.
    """
    dur_s = int(
        trajectory_duration_s
        if trajectory_duration_s is not None
        else CYCLES_TRIAL_DURATION_S
    )
    n_seg = resolve_n_segments(n_segments=n_segments, train_mode=train_mode)
    ss = step_size if step_size is not None else STEP_SIZE
    return (
        CYCLES_SUFFIX_TEMPLATE.replace("!DUR", f"{dur_s}s")
        .replace("!NSEG", str(n_seg))
        .replace("!SS", str(ss))
    )


def resolve_cycles_paths(config: CyclesPathConfig) -> CyclesPaths:
    """Resolve tagged results/plots/checkpoint dirs for a `CyclesConfig`."""
    suffix = resolve_cycles_suffix(
        trajectory_duration_s=config.trajectory_duration_s,
        n_segments=config.n_segments,
        step_size=config.step_size,
        train_mode=config.train_mode,
    )
    plots_dir = CYCLES_PLOTS_BASE / suffix
    ckpt_dir = CYCLES_CKPT_BASE / suffix
    return CyclesPaths(
        suffix=suffix,
        results_dir=CYCLES_RESULTS_BASE / suffix,
        plots_dir=plots_dir,
        gaussian_evolution_plots_dir=plots_dir / "gaussian_evolution",
        ckpt_dir=ckpt_dir,
        indiv_ckpt_dir=ckpt_dir / "indiv",
        room_maps_path=ckpt_dir / "room_maps.json",
    )
