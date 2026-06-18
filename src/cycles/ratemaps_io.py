"""Cycles rate-map NPZ paths and truncated-series discovery (no PyTorch)."""

from pathlib import Path

from core.paths import RESULTS_DIR as ROOT_RESULTS_DIR

DEFAULT_CYCLES_RESULTS_NAME = "cycles_ratemaps.npz"
TRUNCATED_RATEMAPS_PREFIX = "cycles_ratemaps_truncated_"
TRUNCATED_RATEMAPS_SUFFIX = ".npz"

# Unresolved template path (same suffix as ``cycles.cycles_paths.RESULTS_DIR``).
CYCLES_RESULTS_SUFFIX = "!DUR_!NSEG_!SS"
DEFAULT_CYCLES_RESULTS_DIR = ROOT_RESULTS_DIR / "cycles" / CYCLES_RESULTS_SUFFIX
DEFAULT_CYCLES_RESULTS_PATH = DEFAULT_CYCLES_RESULTS_DIR / DEFAULT_CYCLES_RESULTS_NAME


def truncated_cycles_results_path(
    end_cycle: int,
    start_cycle: int = 0,
    results_dir: Path | None = None,
) -> Path:
    """
    Path for a truncated rate-map NPZ under ``results_dir``.

    ``start_cycle == 0`` → ``cycles_ratemaps_truncated_<end_cycle>.npz`` (cycles
    ``0 .. end_cycle-1``). Otherwise ``cycles_ratemaps_truncated_<start>_<end>.npz``.
    """
    base = Path(results_dir or DEFAULT_CYCLES_RESULTS_DIR)
    if start_cycle == 0:
        return base / f"{TRUNCATED_RATEMAPS_PREFIX}{end_cycle}{TRUNCATED_RATEMAPS_SUFFIX}"
    return base / (
        f"{TRUNCATED_RATEMAPS_PREFIX}{start_cycle}_{end_cycle}{TRUNCATED_RATEMAPS_SUFFIX}"
    )


def parse_truncated_ratemaps_filename(name: str) -> tuple[int, int] | None:
    """
    Parse ``cycles_ratemaps_truncated_<end>.npz`` or ``..._<start>_<end>.npz``.

    Returns ``(start_cycle, end_cycle)`` with exclusive ``end_cycle``, or ``None``.
    """
    if not name.startswith(TRUNCATED_RATEMAPS_PREFIX) or not name.endswith(
        TRUNCATED_RATEMAPS_SUFFIX
    ):
        return None
    stem = name[len(TRUNCATED_RATEMAPS_PREFIX) : -len(TRUNCATED_RATEMAPS_SUFFIX)]
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


def discover_truncated_ratemaps_series(
    directory: Path,
    *,
    total_cycles: int = 30,
) -> list[tuple[int, int, Path]]:
    """
    Find truncated NPZs that contiguously cover cycles ``[0, total_cycles)``.

    Returns sorted ``(start_cycle, end_cycle, path)`` triples.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    segments: list[tuple[int, int, Path]] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        bounds = parse_truncated_ratemaps_filename(path.name)
        if bounds is None:
            continue
        start_cycle, end_cycle = bounds
        segments.append((start_cycle, end_cycle, path))

    if not segments:
        raise FileNotFoundError(
            f"No {TRUNCATED_RATEMAPS_PREFIX}*<{TRUNCATED_RATEMAPS_SUFFIX}> files in {directory}"
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
            raise ValueError(f"Invalid cycle range in {path.name}: [{start_cycle}, {end_cycle})")
        expected_start = end_cycle

    if expected_start != total_cycles:
        raise ValueError(
            f"Truncated series ends at cycle {expected_start}, expected {total_cycles}. "
            f"Files: {[p.name for _, _, p in segments]}"
        )

    return segments
