"""Shared RAE training loops and trajectory batching."""
import os
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def masking(x, mask_rate, *, generator: torch.Generator | None = None):
    """Bernoulli input masking; optional CPU `generator` for reproducible draws."""
    batch_size, timesteps, _ = x.shape
    probs = torch.full((batch_size, timesteps), 1 - mask_rate)
    if generator is None:
        mask = torch.bernoulli(probs)
    else:
        mask = torch.bernoulli(probs, generator=generator)
    return x * mask.unsqueeze(2).to(x.device)


def fr_loss(states):
    mean_fr = torch.mean(states, dim=(0, 1))
    return torch.pow(mean_fr, 2).mean()

TrainMode = Literal["default", "indiv_traj"]


class _TrainParams(Protocol):
    mask_rate: float
    step_size: int
    lambda_mse: float
    lambda_fr: float
    train_mode: TrainMode
    n_segments: int | None
    gradient_clip_max: float | None


@dataclass
class TrainConfig:
    """Hyperparameters for episodic RAE training."""

    mask_rate: float = 0.5
    step_size: int = 20
    lambda_mse: float = 1.0
    lambda_fr: float = 200.0
    learning_rate: float = 5e-4
    train_mode: TrainMode = "default"
    n_segments: int | None = None
    gradient_clip_max: float | None = None


def rebatch_trajectories(
    traj_coord: np.ndarray,
    *,
    n_segments: int = 4,
) -> np.ndarray:
    """(B, T, 2) → (B * n_segments, T // n_segments, 2) when T % n_segments == 0."""
    b, t, _ = traj_coord.shape
    if t % n_segments != 0:
        return traj_coord
    return traj_coord.reshape(b, n_segments, t // n_segments, 2).reshape(
        b * n_segments, t // n_segments, 2
    )


def trajectory_rebatch_params(train_mode: TrainMode) -> int:
    return 4 if train_mode == "default" else 8


def resolve_n_segments(config: _TrainParams) -> int:
    """Rebatch count per trajectory; falls back to ``train_mode`` defaults (4 or 8)."""
    if config.n_segments is not None:
        return config.n_segments
    return trajectory_rebatch_params(config.train_mode)


def rebatch_for_mode(
    traj_coord: np.ndarray,
    train_mode: TrainMode,
) -> np.ndarray:
    """Rebatch raw room trajectories for training or rate-map evaluation."""
    return rebatch_trajectories(
        traj_coord,
        n_segments=trajectory_rebatch_params(train_mode),
    )


def prepare_default_trajectory(traj_coord: np.ndarray) -> np.ndarray:
    """
    Rebatch demo-room trajectories into batch 512 (128 x 4 segments).

    Accepts already-rebatched `(512, T, 2)` input unchanged.
    """
    if traj_coord.shape[0] == 512:
        return traj_coord
    if traj_coord.ndim == 4:
        return traj_coord.reshape(-1, traj_coord.shape[-2], 2)
    return rebatch_trajectories(traj_coord, n_segments=4)


def train_steps(
    rae: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    traj_coord: np.ndarray,
    wsm,
    device: torch.device,
    config: _TrainParams,
    mask_generator: torch.Generator | None = None,
) -> None:
    """Run one pass over all episodic steps on a pre-batched trajectory."""
    n_steps = traj_coord.shape[1] // config.step_size
    rae.train()
    for step in range(n_steps):
        optimizer.zero_grad()
        tc = traj_coord[:, step * config.step_size : (step + 1) * config.step_size]
        gt_res = torch.as_tensor(wsm.get_response(tc), dtype=torch.float32, device=device)
        masked_res = masking(
            gt_res[:, :-1], config.mask_rate, generator=mask_generator
        )
        gt_res = gt_res[:, 1:]
        pred, states = rae(masked_res)
        loss = (
            F.mse_loss(pred, gt_res) * config.lambda_mse
            + fr_loss(states[0]) * config.lambda_fr
        )
        loss.backward()
        if config.gradient_clip_max is not None:
            torch.nn.utils.clip_grad_norm_(rae.parameters(), config.gradient_clip_max)
        optimizer.step()


def train_room_visit(
    rae: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    raw_traj_coord: np.ndarray,
    wsm,
    device: torch.device,
    config: _TrainParams,
    mask_generator: torch.Generator | None = None,
) -> None:
    """
    Train on one room visit.

    `default`: rebatch all trajectories (4 segments, batch 512).
    `indiv_traj`: one trajectory at a time (8 segments, batch 8).
    """
    # the following comments assume we have 128 trajectories of 600s each, 
    # and that we have dt = 0.5 and step_size = 20 (-->1s of training per weight update)
    n_segments = resolve_n_segments(config)
    if config.train_mode == "default":
        # by default, we:
        # 1) sub-divide each 600s trajectory in 4 150s segments that we consider independend (-> can train in parallel)
        # 2) Assume each (of the 128) trajectory is independent --> can be trained in parallel
        # This creates a batch of 128*4 = 512 segments, that we train on in parallel (1s per 1 second)
        # hence, each training step encopasses gradient updates from different trajectories at the same time
        #
        # in total, for one room with 1s train segments, we have 150 updates only

        traj = rebatch_trajectories(raw_traj_coord, n_segments=n_segments) # by default, 4
        train_steps(rae, optimizer, traj, wsm, device, config, mask_generator)
    else:
        # In indiv traj, we don't want to mix trajectories within each other. However, we
        # 1) sub-divide each 600s trajectory in 8 75s segments that we consider independend
        # (to increase training speed)
        # --> we train on all trajectories sequentially (hence this for loop), 1s of training per weight update, 
        # on all 8 segments of the given trajectory at once (batch size = 8)
        # 
        # in total, for one room with 1s train segments, we have 75*128 = 9600 updates
        for i in range(raw_traj_coord.shape[0]):
            traj_i = rebatch_trajectories(
                raw_traj_coord[i : i + 1], n_segments=n_segments # by default, 8
            )
            train_steps(rae, optimizer, traj_i, wsm, device, config, mask_generator)


def train_rae_epochs(
    rae: torch.nn.Module,
    traj_coord: np.ndarray,
    wsm,
    arena_map: np.ndarray,
    device: torch.device,
    *,
    config: TrainConfig | None = None,
    mask_rate: float | None = None,
    learning_rate: float | None = None,
    n_epochs: int = 10,
    step_size: int | None = None,
    lambda_mse: float | None = None,
    lambda_fr: float | None = None,
    train_mode: TrainMode | None = None,
    ckpt_dir: str | os.PathLike | None = None,
    log_every: int = 100,
    mask_rng_seed: int = 0,
) -> list[np.ndarray]:
    """
    Train the RAE for multiple epochs and log rate maps after each epoch.

    Supports both `default` and `indiv_traj` batching. If `ckpt_dir/latest.pth`
    exists, load it and skip training.

    Returns
    -------
    list[np.ndarray]
        Rate maps after the initial (untrained) state and after each epoch,
        or a single final rate map when loading from checkpoint.
    """
    config = config or TrainConfig()
    if mask_rate is not None:
        config.mask_rate = mask_rate
    if learning_rate is not None:
        config.learning_rate = learning_rate
    if step_size is not None:
        config.step_size = step_size
    if lambda_mse is not None:
        config.lambda_mse = lambda_mse
    if lambda_fr is not None:
        config.lambda_fr = lambda_fr
    if train_mode is not None:
        config.train_mode = train_mode

    from core.utils import compute_ratemap

    traj_for_ratemap = (
        prepare_default_trajectory(traj_coord)
        if config.train_mode == "default"
        else rebatch_for_mode(traj_coord, config.train_mode)
    )

    if ckpt_dir is not None:
        latest_path = os.path.join(ckpt_dir, "latest.pth")
        if os.path.isfile(latest_path):
            rae.load_state_dict(
                torch.load(latest_path, map_location=device, weights_only=True)["rae"]
            )
            rm = compute_ratemap(
                rae,
                traj_for_ratemap,
                wsm,
                arena_map,
                device,
                config.mask_rate,
                config.step_size,
                show_progress=True,
            )
            tqdm.write(f"Loaded {latest_path}. Skipping training.")
            return [rm]
        os.makedirs(ckpt_dir, exist_ok=True)

    optimizer = torch.optim.Adam(rae.parameters(), lr=config.learning_rate)
    mask_generator = torch.Generator(device="cpu")
    mask_generator.manual_seed(mask_rng_seed)

    epoch_ratemaps = [
        compute_ratemap(
            rae,
            traj_for_ratemap,
            wsm,
            arena_map,
            device,
            config.mask_rate,
            config.step_size,
            mask_generator=mask_generator,
            show_progress=True,
        )
    ]

    for epoch in tqdm(range(n_epochs), desc="Training"):
        train_room_visit(
            rae,
            optimizer,
            traj_coord,
            wsm,
            device,
            config,
            mask_generator,
        )

        if log_every > 0 and config.train_mode == "default":
            traj = prepare_default_trajectory(traj_coord)
            n_steps = traj.shape[1] // config.step_size
            tqdm.write(f"  epoch {epoch}: {n_steps} steps, batch {traj.shape[0]}")

        if ckpt_dir is not None:
            torch.save(
                {"rae": rae.state_dict()},
                os.path.join(ckpt_dir, "latest.pth"),
            )
            torch.save(
                {"rae": rae.state_dict()},
                os.path.join(ckpt_dir, f"rae_{epoch}.pth"),
            )

        rm = compute_ratemap(
            rae,
            traj_for_ratemap,
            wsm,
            arena_map,
            device,
            config.mask_rate,
            config.step_size,
            mask_generator=mask_generator,
        )
        epoch_ratemaps.append(rm)
        tqdm.write(
            f"  epoch {epoch} ratemaps: shape={rm.shape}, "
            f"mean peak activation={np.nanmax(rm, axis=(1, 2)).mean():.4f}"
        )

    tqdm.write(f"Done. Logged {len(epoch_ratemaps)} epoch ratemaps.")
    return epoch_ratemaps
