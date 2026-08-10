"""Shared results/signals/ratemaps/checkpoint path layout for room experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentPaths:
    """Resolved results/signals/ratemaps/checkpoint directories for one run configuration."""

    suffix: str
    results_dir: Path
    signals_dir: Path
    ratemaps_dir: Path
    ckpt_dir: Path

    def mkdirs(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.signals_dir.mkdir(parents=True, exist_ok=True)
        self.ratemaps_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)


def resolve_experiment_paths(results_base: Path, ckpt_base: Path, suffix: str) -> ExperimentPaths:
    results_dir = results_base / suffix
    return ExperimentPaths(
        suffix=suffix,
        results_dir=results_dir,
        signals_dir=results_dir / "signals",
        ratemaps_dir=results_dir / "ratemaps",
        ckpt_dir=ckpt_base / suffix,
    )
