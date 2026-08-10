# To avoid having up to ~1000 saved files, we batch all ratemap-evaluated segments (e.g.: 20 per trajectory) 
# into one (saved) "trajectory-array".
# 
# Furthermore, to save space, we save it as fp16 (less that 1e-4 error...)
# -->  
# Keys:
#   ratemaps     (K, n_hidden, H, W) float16
#   capture_tags (K,) str — e.g. "pre", "seg0", "seg1", ...
# (where one ratemap is, logically, (n_hidden, H, W) )

import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np


def capture_tag_for_segment(*, pre: bool = False, seg_idx: int | None = None) -> str:
    if pre:
        return "pre"
    if seg_idx is None:
        raise ValueError("seg_idx is required when pre=False")
    return f"seg{seg_idx}"


def parse_capture_tag(tag: str) -> int:
    """Sort order: pre -> -1, segN -> N (0, 1, ...)"""
    if tag == "pre":
        return -1
    m = re.fullmatch(r"seg(\d+)", tag)
    if m is None:
        raise ValueError(f"Unknown capture tag: {tag!r}")
    return int(m.group(1))


@dataclass(frozen=True)
class RatemapCaptureRef:
    path: Path
    capture_idx: int = 0


class TrajectoryRatemapBatch:
    """Accumulates fp16 rate maps for one trajectory (one room) until flush."""

    def __init__(self) -> None:
        self._ratemaps: list[np.ndarray] = []
        self._tags: list[str] = []

    def __len__(self) -> int:
        return len(self._ratemaps)

    def append(self, tag: str, ratemap: np.ndarray) -> None:
        self._tags.append(tag)
        self._ratemaps.append(np.asarray(ratemap, dtype=np.float16))

    def save(self, path: Path) -> None:
        if not self._ratemaps:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            ratemaps=np.stack(self._ratemaps, axis=0),
            capture_tags=np.asarray(self._tags),
        )


def trajectory_ratemap_path(
    ratemaps_dir: Path,
    base_tag: str,
    *,
    room_idx: int | None = None,
    multi_room: bool = False,
) -> Path:
    if multi_room:
        if room_idx is None:
            raise ValueError("room_idx is required when multi_room=True")
        return ratemaps_dir / f"{base_tag}_room{room_idx}.npz"
    return ratemaps_dir / f"{base_tag}.npz"


def load_trajectory_ratemaps(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path) as data:
        ratemaps = np.asarray(data["ratemaps"])
        tags = [str(t) for t in data["capture_tags"]]
    return ratemaps, tags


def load_ratemap_capture(ref: RatemapCaptureRef) -> np.ndarray:
    """Return one `(n_hidden, H, W)` slice, upcast to float32."""
    ratemaps, _ = load_trajectory_ratemaps(ref.path)
    if ref.capture_idx < 0 or ref.capture_idx >= ratemaps.shape[0]:
        raise IndexError(
            f"capture_idx {ref.capture_idx} out of range for {ref.path} "
            f"(K={ratemaps.shape[0]})"
        )
    return np.asarray(ratemaps[ref.capture_idx], dtype=np.float32)


def load_ratemap_capture_cells(
    ref: RatemapCaptureRef,
    cell_indices: np.ndarray | list[int],
) -> np.ndarray:
    return load_ratemap_capture(ref)[list(cell_indices)].copy()


def _single_room_trajectory_key(stem: str) -> tuple[int, int] | None:
    m = re.fullmatch(r"epoch(\d+)_traj(\d+)", stem)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


def _two_rooms_trajectory_key(stem: str, room_idx: int) -> tuple[int, int, int] | None:
    m = re.fullmatch(rf"rep(\d+)_room(\d+)_traj(\d+)_room{room_idx}", stem)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _many_rooms_trajectory_key(stem: str) -> tuple[int, int, int] | None:
    m = re.fullmatch(r"cyc(\d+)_room(\d+)_traj(\d+)", stem)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def discover_ratemap_captures(
    ratemaps_dir: Path,
    *,
    experiment_name: str,
    room_idx: int = 0,
) -> list[RatemapCaptureRef]:
    """Return capture refs in training order (one ref per capture timepoint)."""
    ratemaps_dir = Path(ratemaps_dir)
    captures: list[tuple[tuple[int, ...], RatemapCaptureRef]] = []

    if experiment_name == "single_room":
        for path in ratemaps_dir.glob("*.npz"):
            traj_key = _single_room_trajectory_key(path.stem)
            if traj_key is None:
                continue
            _, tags = load_trajectory_ratemaps(path)
            epoch, traj = traj_key
            for capture_idx, tag in enumerate(tags):
                captures.append(
                    (
                        (epoch, traj, parse_capture_tag(tag)),
                        RatemapCaptureRef(path, capture_idx),
                    )
                )

    elif experiment_name == "two_rooms":
        for path in ratemaps_dir.glob(f"*_room{room_idx}.npz"):
            traj_key = _two_rooms_trajectory_key(path.stem, room_idx)
            if traj_key is None:
                continue
            _, tags = load_trajectory_ratemaps(path)
            rep, visit_room, traj = traj_key
            for capture_idx, tag in enumerate(tags):
                captures.append(
                    (
                        (rep, visit_room, traj, parse_capture_tag(tag)),
                        RatemapCaptureRef(path, capture_idx),
                    )
                )

    elif experiment_name == "many_rooms":
        for path in ratemaps_dir.glob("*.npz"):
            traj_key = _many_rooms_trajectory_key(path.stem)
            if traj_key is None:
                continue
            _, tags = load_trajectory_ratemaps(path)
            cyc, room, traj = traj_key
            for capture_idx, tag in enumerate(tags):
                captures.append(
                    (
                        (cyc, room, traj, parse_capture_tag(tag)),
                        RatemapCaptureRef(path, capture_idx),
                    )
                )

    else:
        raise ValueError(f"Unsupported experiment: {experiment_name}")

    captures.sort(key=lambda item: item[0])
    return [ref for _, ref in captures]


def capture_column_label(ref: RatemapCaptureRef) -> str:
    stem = ref.path.stem
    _, tags = load_trajectory_ratemaps(ref.path)
    tag = tags[ref.capture_idx]

    traj_tag = re.search(r"traj(\d+)", stem)
    traj_num = traj_tag.group(1) if traj_tag else "?"
    if tag == "pre":
        return f"t{traj_num}\npre"
    m = re.fullmatch(r"seg(\d+)", tag)
    return f"t{traj_num}\ns{m.group(1)}" if m else tag
