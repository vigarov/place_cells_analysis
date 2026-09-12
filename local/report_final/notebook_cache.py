"""Disk cache helpers for report notebooks (`REPORT_final_FINAL`, `REPORT_final_aide`)."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Any

SINGLE_ROOM_CACHE = "single_room.pkl.gz"
TWO_ROOMS_CACHE = "two_rooms.pkl.gz"
SINGLE_ROOM_EFF_LR_CACHE = "single_room_eff_lr.pkl.gz"


def report_final_cache_dir(project_root: Path) -> Path:
    return project_root / "local" / "_cache"


def save_pkl_gz(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_pkl_gz(path: Path) -> Any | None:
    if not path.is_file():
        return None
    with gzip.open(path, "rb") as f:
        return pickle.load(f)
