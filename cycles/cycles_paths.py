"""Paths for the cycles experiment (code vs room data)."""

import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent


def project_root() -> Path:
    for path in (_PKG_DIR, *_PKG_DIR.parents):
        if (path / "pyproject.toml").is_file():
            return path
    return _PKG_DIR.parent


ROOT = project_root()

# Python package (notebooks, training code, checkpoints, plots)
PKG_DIR = _PKG_DIR

# Room data: manifest + room_* (generated under trajectories/cycles)
ROOMS_DIR = ROOT / "trajectories" / "cycles"

CKPT_DIR = PKG_DIR / "ckpts"
INDIV_CKPT_DIR = CKPT_DIR / "indiv"
ROOM_MAPS_PATH = CKPT_DIR / "room_maps.json"
RESULTS_DIR = PKG_DIR / "results"
PLOTS_DIR = PKG_DIR / "plots"
GAUSSIAN_EVOLUTION_PLOTS_DIR = PLOTS_DIR / "gaussian_evolution"
MANIFEST_PATH = ROOMS_DIR / "manifest.json"

# Backward-compatible alias (room data directory)
CYCLES_DIR = ROOMS_DIR

# Repo-root modules (``weak_sm_cell``, ``utils``, …) live beside ``pyproject.toml``.
_root_str = str(ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)
