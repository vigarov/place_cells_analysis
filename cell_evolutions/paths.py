"""Project paths for cell-evolution notebooks and helpers."""

import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent


def project_root() -> Path:
    """Repository root (directory containing ``pyproject.toml``)."""
    for path in (_PKG_DIR, *_PKG_DIR.parents):
        if (path / "pyproject.toml").is_file():
            return path
    return _PKG_DIR.parent


ROOT = project_root()
CKPT_DIR = _PKG_DIR / "ckpts"
PLOTS_DIR = _PKG_DIR / "plots"

_root_str = str(ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)
