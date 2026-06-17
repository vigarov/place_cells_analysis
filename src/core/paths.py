"""Repository-root paths for data, plots, results, and checkpoints."""

from pathlib import Path


def project_root() -> Path:
    """Directory containing ``pyproject.toml``."""
    for path in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (path / "pyproject.toml").is_file():
            return path
    raise RuntimeError("Could not find project root (pyproject.toml)")


ROOT = project_root()
DATA_DIR = ROOT / "data"
PLOTS_DIR = ROOT / "plots"
RESULTS_DIR = ROOT / "results"
CKPTS_DIR = ROOT / "ckpts"
