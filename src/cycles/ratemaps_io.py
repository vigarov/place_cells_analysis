"""Cycles rate-map NPZ paths and truncated-series discovery (no PyTorch)."""

from pathlib import Path

from core.paths import RESULTS_DIR as ROOT_RESULTS_DIR

DEFAULT_CYCLES_RESULTS_NAME = "cycles_ratemaps.npz"
DEFAULT_CYCLES_RESULTS_PRE_NAME = "cycles_ratemaps_pre.npz"
TRUNCATED_RATEMAPS_PREFIX = "cycles_ratemaps_truncated_"
TRUNCATED_RATEMAPS_PRE_PREFIX = "cycles_ratemaps_pre_truncated_"
TRUNCATED_RATEMAPS_SUFFIX = ".npz"

# Unresolved template path (same suffix as `cycles.cycles_paths.RESULTS_DIR`).
CYCLES_RESULTS_SUFFIX = "!DUR_!NSEG_!SS"
DEFAULT_CYCLES_RESULTS_DIR = ROOT_RESULTS_DIR / "cycles" / CYCLES_RESULTS_SUFFIX
DEFAULT_CYCLES_RESULTS_PATH = DEFAULT_CYCLES_RESULTS_DIR / DEFAULT_CYCLES_RESULTS_NAME
DEFAULT_CYCLES_RESULTS_PRE_PATH = DEFAULT_CYCLES_RESULTS_DIR / DEFAULT_CYCLES_RESULTS_PRE_NAME


def discover_cycles_ratemaps_in_dir(directory: Path) -> list[Path]:
    """
    Return post and pre rate-map NPZ paths that exist under `directory`.

    Post is listed first when present, then pre. Missing files are skipped.
    """
    directory = Path(directory)
    paths: list[Path] = []
    post = directory / DEFAULT_CYCLES_RESULTS_NAME
    pre = directory / DEFAULT_CYCLES_RESULTS_PRE_NAME
    if post.is_file():
        paths.append(post)
    if pre.is_file():
        paths.append(pre)
    if not paths:
        raise FileNotFoundError(
            f"No {DEFAULT_CYCLES_RESULTS_NAME} or {DEFAULT_CYCLES_RESULTS_PRE_NAME} "
            f"in {directory}"
        )
    return paths


def truncated_cycles_results_path(
    end_cycle: int,
    start_cycle: int = 0,
    results_dir: Path | None = None,
    *,
    pre: bool = False,
) -> Path:
    """
    Path for a truncated rate-map NPZ under `results_dir`.

    `start_cycle == 0` → `cycles_ratemaps[_pre]_truncated_<end_cycle>.npz`
    (cycles `0 .. end_cycle-1`). Otherwise
    `cycles_ratemaps[_pre]_truncated_<start>_<end>.npz`.
    """
    prefix = TRUNCATED_RATEMAPS_PRE_PREFIX if pre else TRUNCATED_RATEMAPS_PREFIX
    base = Path(results_dir or DEFAULT_CYCLES_RESULTS_DIR)
    if start_cycle == 0:
        return base / f"{prefix}{end_cycle}{TRUNCATED_RATEMAPS_SUFFIX}"
    return base / (
        f"{prefix}{start_cycle}_{end_cycle}{TRUNCATED_RATEMAPS_SUFFIX}"
    )


def _parse_truncated_ratemaps_filename_with_prefix(
    name: str,
    prefix: str,
) -> tuple[int, int] | None:
    """
    Parse `<prefix><end>.npz` or `<prefix><start>_<end>.npz`.

    Returns `(start_cycle, end_cycle)` with exclusive `end_cycle`, or `None`.
    """
    if not name.startswith(prefix) or not name.endswith(TRUNCATED_RATEMAPS_SUFFIX):
        return None
    stem = name[len(prefix) : -len(TRUNCATED_RATEMAPS_SUFFIX)]
    if not stem:
        return None
    parts = stem.split("_")
    if len(parts) == 1:
        if not parts[0].isdigit():
            return None
        end_cycle = int(parts[0])
        return 0, end_cycle
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        return int(parts[0]), int(parts[1])
    return None


def parse_truncated_ratemaps_filename(name: str) -> tuple[int, int] | None:
    """
    Parse `cycles_ratemaps_truncated_<end>.npz` or `..._<start>_<end>.npz`.

    Returns `(start_cycle, end_cycle)` with exclusive `end_cycle`, or `None`.
    """
    return _parse_truncated_ratemaps_filename_with_prefix(
        name, TRUNCATED_RATEMAPS_PREFIX
    )


def parse_truncated_pre_ratemaps_filename(name: str) -> tuple[int, int] | None:
    """
    Parse `cycles_ratemaps_pre_truncated_<end>.npz` or `..._<start>_<end>.npz`.

    Returns `(start_cycle, end_cycle)` with exclusive `end_cycle`, or `None`.
    """
    return _parse_truncated_ratemaps_filename_with_prefix(
        name, TRUNCATED_RATEMAPS_PRE_PREFIX
    )


def _discover_truncated_ratemaps_series_with_prefix(
    directory: Path,
    *,
    prefix: str,
    total_cycles: int = 30,
) -> list[tuple[int, int, Path]]:
    """
    Find truncated NPZs with `prefix` that contiguously cover cycles
    `[0, total_cycles)`.

    Returns sorted `(start_cycle, end_cycle, path)` triples.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    segments: list[tuple[int, int, Path]] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        bounds = _parse_truncated_ratemaps_filename_with_prefix(path.name, prefix)
        if bounds is None:
            continue
        start_cycle, end_cycle = bounds
        segments.append((start_cycle, end_cycle, path))

    if not segments:
        raise FileNotFoundError(
            f"No {prefix}*{TRUNCATED_RATEMAPS_SUFFIX} files in {directory}"
        )

    segments.sort(key=lambda item: item[0])
    expected_start = 0
    for start_cycle, end_cycle, path in segments:
        if start_cycle != expected_start:
            raise ValueError(
                f"Gap in truncated series at cycle {expected_start}: "
                f"next file is {path.name} (starts at cycle {start_cycle})."
            )
        if end_cycle <= start_cycle:
            raise ValueError(
                f"Invalid cycle range in {path.name}: [{start_cycle}, {end_cycle})"
            )
        expected_start = end_cycle

    if expected_start != total_cycles:
        raise ValueError(
            f"Truncated series ends at cycle {expected_start}, expected {total_cycles}. "
            f"Files: {[p.name for _, _, p in segments]}"
        )

    return segments


def discover_truncated_ratemaps_series(
    directory: Path,
    *,
    total_cycles: int = 30,
) -> list[tuple[int, int, Path]]:
    """
    Find truncated NPZs that contiguously cover cycles `[0, total_cycles)`.

    Returns sorted `(start_cycle, end_cycle, path)` triples.
    """
    return _discover_truncated_ratemaps_series_with_prefix(
        directory,
        prefix=TRUNCATED_RATEMAPS_PREFIX,
        total_cycles=total_cycles,
    )


def discover_truncated_pre_ratemaps_series(
    directory: Path,
    *,
    total_cycles: int = 30,
) -> list[tuple[int, int, Path]]:
    """
    Find pre truncated NPZs that contiguously cover cycles `[0, total_cycles)`.

    Returns sorted `(start_cycle, end_cycle, path)` triples.
    """
    return _discover_truncated_ratemaps_series_with_prefix(
        directory,
        prefix=TRUNCATED_RATEMAPS_PRE_PREFIX,
        total_cycles=total_cycles,
    )
