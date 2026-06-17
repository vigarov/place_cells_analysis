"""Paths for the cycles experiment (code vs room data)."""

from core.paths import (
    CKPTS_DIR,
    DATA_DIR,
    PLOTS_DIR as _ROOT_PLOTS_DIR,
    RESULTS_DIR as _ROOT_RESULTS_DIR,
    ROOT,
)

ROOMS_DIR = DATA_DIR / "cycles"
CKPT_DIR = CKPTS_DIR / "cycles"
INDIV_CKPT_DIR = CKPT_DIR / "indiv"
ROOM_MAPS_PATH = CKPT_DIR / "room_maps.json"
RESULTS_DIR = _ROOT_RESULTS_DIR / "cycles"
PLOTS_DIR = _ROOT_PLOTS_DIR / "cycles"
GAUSSIAN_EVOLUTION_PLOTS_DIR = PLOTS_DIR / "gaussian_evolution"
MANIFEST_PATH = ROOMS_DIR / "manifest.json"

# Backward-compatible alias (room data directory)
CYCLES_DIR = ROOMS_DIR
