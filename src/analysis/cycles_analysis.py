"""Analysis for Suppl. Figs. 1-2 (population vectors and correlations)."""
import numpy as np

from experiments.old_cycles.constants import DEFAULT_PADDING, POPULATION_BIN_SIZE_CM


def trial_metadata_from_schedule(
    schedule: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """`cycle_ids` and `room_ids` in visit order (cycle-major, shuffled rooms)."""
    n_cycles, n_rooms = schedule.shape
    cycle_ids = np.repeat(np.arange(n_cycles), n_rooms)
    room_ids = schedule.reshape(-1)
    return cycle_ids, room_ids


def reorder_trials_by_room(
    ratemaps: np.ndarray,
    cycle_ids: np.ndarray,
    room_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort trials so all visits to room 1 appear first, then room 2, etc.

    Within each room, trials are ordered by increasing cycle index.
    """
    order = np.lexsort((cycle_ids, room_ids))
    return ratemaps[order], cycle_ids[order], room_ids[order]


def reorder_trials_by_cycle(
    ratemaps: np.ndarray,
    cycle_ids: np.ndarray,
    room_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort trials cycle-major for Supplemental Figure 1.

    Cycle 0 (all rooms) first, then cycle 1, etc. Within each cycle, rooms are
    ordered by increasing room index (not shuffled visit order).
    """
    order = np.lexsort((room_ids, cycle_ids))
    return ratemaps[order], cycle_ids[order], room_ids[order]


def ratemap_to_binned_field(
    ratemap: np.ndarray,
    *,
    padding: int = DEFAULT_PADDING,
    room_size_cm: int = 100,
    bin_size_cm: int = POPULATION_BIN_SIZE_CM,
) -> np.ndarray:
    """
    Average firing rates in `room_size_cm / bin_size_cm` bins per unit.

    Parameters
    ----------
    ratemap : ndarray, shape `(n_units, H, W)`

    Returns
    -------
    ndarray, shape `(n_units, n_bins, n_bins)`
    """
    if room_size_cm % bin_size_cm != 0:
        raise ValueError("room_size_cm must be divisible by bin_size_cm")
    n_bins = room_size_cm // bin_size_cm
    y0, x0 = padding, padding
    y1, x1 = y0 + room_size_cm, x0 + room_size_cm
    inner = ratemap[:, y0:y1, x0:x1]
    n_units, h, w = inner.shape
    assert h == room_size_cm and w == room_size_cm
    b = bin_size_cm
    return inner.reshape(n_units, n_bins, b, n_bins, b).mean(axis=(2, 4))


def population_vectors_from_ratemaps(
    ratemaps: np.ndarray,
    *,
    padding: int = DEFAULT_PADDING,
    room_size_cm: int = 100,
    bin_size_cm: int = POPULATION_BIN_SIZE_CM,
) -> np.ndarray:
    """
    Build population coding vectors (Alme et al. / paper Suppl. Sec. 3.1).

    Returns
    -------
    ndarray, shape `(n_trials, n_units * n_bins * n_bins)`
    """
    n_bins = room_size_cm // bin_size_cm
    b = bin_size_cm
    y0, x0 = padding, padding
    y1, x1 = y0 + room_size_cm, x0 + room_size_cm
    inner = ratemaps[:, :, y0:y1, x0:x1]
    binned = inner.reshape(
        inner.shape[0], inner.shape[1], n_bins, b, n_bins, b
    ).mean(axis=(3, 5))
    return binned.reshape(inner.shape[0], -1)


def pearson_correlation_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    Pairwise Pearson correlation between population vectors.

    Parameters
    ----------
    vectors : ndarray, shape `(n_trials, n_features)`
    """
    x = vectors.astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    x = x / norms
    return x @ x.T


def ratemaps_for_room_across_cycles(
    ratemaps: np.ndarray,
    cycle_ids: np.ndarray,
    room_ids: np.ndarray,
    room: int | str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract rate maps for one room, ordered by cycle.

    Returns
    -------
    room_ratemaps : (n_cycles_present, n_units, H, W)
    cycles : sorted cycle indices
    """
    if isinstance(room, str):
        room_num = int(room.replace("room_", ""))
    else:
        room_num = int(room)

    mask = room_ids == room_num
    cycles = np.sort(np.unique(cycle_ids[mask]))
    out = []
    for c in cycles:
        idx = np.where(mask & (cycle_ids == c))[0]
        if len(idx) != 1:
            raise ValueError(f"Expected one visit for room {room_num} cycle {c}, got {len(idx)}")
        out.append(ratemaps[idx[0]])
    return np.stack(out, axis=0), cycles


def select_drift_cells(
    ratemaps_room: np.ndarray,
    n_cells: int = 6,
    *,
    seed: int = 0,
    min_peak: float = 0.05,
) -> np.ndarray:
    """Random active units for Supplemental Figure 2."""
    peak = np.nanmax(ratemaps_room, axis=(0, 2, 3))
    active = np.where(peak >= min_peak)[0]
    if len(active) < n_cells:
        active = np.arange(ratemaps_room.shape[1])
    rng = np.random.default_rng(seed)
    return rng.choice(active, size=min(n_cells, len(active)), replace=False)
