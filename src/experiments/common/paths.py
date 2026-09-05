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


def resolve_experiment_paths(
    results_base: Path,
    ckpt_base: Path,
    suffix: str,
    *,
    optimizer_tag: str | None = None,
) -> ExperimentPaths:
    run_suffix = f"{suffix}/{optimizer_tag}" if optimizer_tag else suffix
    results_dir = results_base / run_suffix
    return ExperimentPaths(
        suffix=run_suffix,
        results_dir=results_dir,
        signals_dir=results_dir / "signals",
        ratemaps_dir=results_dir / "ratemaps",
        ckpt_dir=ckpt_base / run_suffix,
    )


_RATEMAPS_SUBDIR = "ratemaps"
GAUSSIAN_RF_FITS_FILENAME = "gaussian_rf_fits.npz"
_TWO_ROOMS = "two_rooms"
_EXPERIMENT_TYPE_NAMES = frozenset({"single_room", "two_rooms", "many_rooms"})
_ALT_TOO_SUFFIX = "_altTOO"


def _is_experiment_family_dir(name: str) -> bool:
    if name in _EXPERIMENT_TYPE_NAMES:
        return True
    return any(name == f"{exp}{_ALT_TOO_SUFFIX}" for exp in _EXPERIMENT_TYPE_NAMES)


def is_experiment_results_dir(path: Path) -> bool:
    """True when `path/ratemaps/` exists."""
    return (Path(path) / _RATEMAPS_SUBDIR).is_dir()


def is_optimizer_results_dir(path: Path) -> bool:
    """True for `.../<experiment_type>/<suffix>/<optimizer_tag>/`."""
    if not is_experiment_results_dir(path):
        return False
    suffix_dir = path.parent
    if suffix_dir.name in _EXPERIMENT_TYPE_NAMES:
        return False
    return _is_experiment_family_dir(suffix_dir.parent.name)


def gaussian_rf_fits_filename(experiment_name: str, room_idx: int) -> str:
    """Output NPZ name for Gaussian RF fits (per-room for `two_rooms`)."""
    if experiment_name == _TWO_ROOMS:
        return f"gaussian_rf_fits_room{room_idx}.npz"
    return GAUSSIAN_RF_FITS_FILENAME


def gaussian_rf_fits_path(
    results_dir: Path,
    experiment_name: str,
    room_idx: int,
) -> Path:
    return Path(results_dir) / gaussian_rf_fits_filename(experiment_name, room_idx)


def optimizer_run_complete(
    results_dir: Path,
    *,
    experiment_name: str,
    room_idx: int | None = None,
) -> bool:
    """True when all required Gaussian RF fit files exist for this optimizer run."""
    if experiment_name == _TWO_ROOMS:
        if room_idx is None:
            return all(
                gaussian_rf_fits_path(results_dir, experiment_name, r).is_file()
                for r in (0, 1)
            )
        return gaussian_rf_fits_path(results_dir, experiment_name, room_idx).is_file()
    return gaussian_rf_fits_path(results_dir, experiment_name, 0).is_file()


def filter_optimizer_results_dirs(
    results_dirs: list[Path],
    optimizers: list[str] | None,
) -> list[Path]:
    """Keep only runs whose directory name is in `optimizers` (preserves request order)."""
    if not optimizers:
        return results_dirs
    by_name = {path.name: path for path in results_dirs}
    missing = [name for name in optimizers if name not in by_name]
    if missing:
        raise FileNotFoundError(
            f"No optimizer results directories named {missing!r} "
            f"(available: {sorted(by_name)})"
        )
    return [by_name[name] for name in optimizers]


def discover_optimizer_results_dirs(input_path: Path) -> list[Path]:
    """
    Resolve one or more optimizer run directories under `input_path`.

    Accepts:

    - `.../<experiment_type>/<suffix>/` with optimizer subdirs each containing `ratemaps/`
    - `.../<experiment_type>/<suffix>/<optimizer_tag>/` (contains `ratemaps/`)
    - `.../<experiment_type>/<suffix>/<optimizer_tag>/ratemaps/`
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.is_file():
        raise ValueError(
            f"Expected a results directory or experiment suffix directory, got file: {input_path}"
        )

    if input_path.name == _RATEMAPS_SUBDIR:
        parent = input_path.parent
        if not is_optimizer_results_dir(parent):
            raise FileNotFoundError(
                f"Expected .../<suffix>/<optimizer_tag>/{_RATEMAPS_SUBDIR}/, got {input_path}"
            )
        return [parent]

    if is_optimizer_results_dir(input_path):
        return [input_path]

    runs = sorted(
        child
        for child in input_path.iterdir()
        if child.is_dir() and is_optimizer_results_dir(child)
    )
    if runs:
        return runs

    raise FileNotFoundError(
        f"No optimizer {_RATEMAPS_SUBDIR}/ directories under {input_path}"
    )
