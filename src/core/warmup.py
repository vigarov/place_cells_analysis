"""Warmup pretraining on a spatially-smoothed ("coarse") version of the WSM signal.

Before training on the actual rooms, `single_room`/`two_rooms`/
`many_rooms` first pretrain on a per-cell Gaussian-smoothed copy of the room's
`WeakSMCell` response fields. Each cell's field is independently blurred and
rescaled back to its original mean firing rate, so the warmup target has the
same overall magnitude per cell but coarser (lower spatial frequency) spatial
structure.

This creates a bio-realistic warmup for cases (e.g.: Vaidya et al.) where mice are
habituated to an environment before a "reward" (which, in our case, is higher WSM
overall sensory signal) is delivered. Since our rooms are uniquely defined by their WSM,
we want to keep a similar:
* mean room WSM magnitude
* "overall" spatial frequency
--> gaussian smooth + scaling
"""

import copy

from dataclasses import replace

import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm

from core.training import TrainConfig, train_trajectory_segments
from core.weak_sm_cell import WeakSMCell

DEFAULT_WARMUP_GAUSSIAN_SIGMA = 15.0


def build_warmup_wsm(
    wsm: WeakSMCell, sigma: float = DEFAULT_WARMUP_GAUSSIAN_SIGMA
) -> WeakSMCell:
    """Return a copy of `wsm` whose `response_map` is per-cell Gaussian-smoothed.

    Each cell's field `response_map[i]` is filtered independently, then rescaled so its
    mean matches the original (unfiltered) field's mean
    """
    warm = copy.copy(wsm)  # shares rng/arena_map, we will only change response_map
    original = wsm.response_map
    smoothed = np.empty_like(original)
    for i in range(original.shape[0]):
        cell = original[i]
        filtered = gaussian_filter(cell, sigma=sigma, mode="nearest")
        filtered_mean = filtered.mean()
        cell_mean = cell.mean()
        if filtered_mean != 0:
            filtered = filtered / filtered_mean * cell_mean
        smoothed[i] = filtered
    warm.response_map = smoothed
    return warm


def run_warmup(
    rae,
    optimizer,
    rooms,
    device,
    config: TrainConfig,
    *,
    gaussian_sigma: float,
    step_size: int,
    warmup_shuffle: bool = False,
    warmup_shuffle_seed: int = 3003,
    mask_generator=None,
    show_progress_level: int = 0,
    experiment_name: str = "",
) -> None:
    """Pretrain on warmup trajectories for all rooms, one trajectory at a time.

    No gradient/optimizer-signal capture and no activation capture during warmup

    When `warmup_shuffle` is true, trajectories from all rooms are pooled and
    shuffled with `warmup_shuffle_seed` before training.
    """
    warmup_config = replace(config, step_size=step_size)
    warmup_wsm_cache: dict[int, WeakSMCell] = {}

    def warmup_wsm_for(room_idx: int, wsm: WeakSMCell) -> WeakSMCell:
        if room_idx not in warmup_wsm_cache:
            warmup_wsm_cache[room_idx] = build_warmup_wsm(wsm, sigma=gaussian_sigma)
        return warmup_wsm_cache[room_idx]

    def train_step(room_idx: int, traj_idx: int) -> None:
        room = rooms[room_idx]
        train_trajectory_segments(
            rae,
            optimizer,
            room.traj_coord[traj_idx : traj_idx + 1],
            warmup_wsm_for(room_idx, room.wsm),
            device,
            warmup_config,
            mask_generator=mask_generator,
        )

    desc = f"{experiment_name} warmup" if experiment_name else "Warmup"

    if warmup_shuffle or len(rooms) == 1:
        trajectories_to_run: list[tuple[int, int]] = [
            (room_idx, traj_idx)
            for room_idx, room in enumerate(rooms)
            for traj_idx in range(room.n_warm)
        ]
        if warmup_shuffle:
            rng = np.random.default_rng(warmup_shuffle_seed)
            rng.shuffle(trajectories_to_run)
        step_iter = trajectories_to_run
        if show_progress_level >= 1:
            step_iter = tqdm(trajectories_to_run, desc=desc, unit="traj", leave=warmup_shuffle)
        for room_idx, traj_idx in step_iter:
            train_step(room_idx, traj_idx)
    else:
        room_iter = enumerate(rooms)
        if show_progress_level >= 1:
            room_iter = tqdm(list(enumerate(rooms)), desc=desc, unit="room")
        for room_idx, _room in room_iter:
            traj_iter = range(rooms[room_idx].n_warm)
            if show_progress_level >= 1:
                traj_iter = tqdm(traj_iter, desc="Warmup", unit="traj", leave=False)
            for traj_idx in traj_iter:
                train_step(room_idx, traj_idx)
