"""
Bayesian Hyperparameter search trial def (bounds, constraints, ...)
Runs warmup + {2 epochs, 1 repetition, 1 cycle} training protocols (depending on the selected config experiment)
"""
import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from core.experiment import Room, RoomExperiment
from core.progress import group_protocol
from core.training import TrainConfig, fr_loss, train_trajectory_segments, unif_loss
from core.utils import MaskMethod, apply_input_mask, compute_ratemap
from core.warmup import run_warmup
from experiments.common.run import ExperimentRunConfig, create_experiment
from experiments.many_rooms.experiment import MANY_ROOMS_NAME
from experiments.single_room.experiment import SINGLE_ROOM_NAME
from experiments.two_rooms.experiment import TWO_ROOMS_NAME
from models.custom_learnable_activation import CustomLearnableActivation
from models.utils import build_rae
from optimizers import optimizer_extractor_from_dict

BoVariant = Literal["a", "b"]
SearchBounds = dict[str, tuple[float, float]]

ACTIVITY_TARGET = 0.40
ACTIVITY_PEAK_FRACTION = 0.40
BAD_RUN_NAN_THRESHOLD = 0.5
BAD_RUN_LOSS = 1e6
BAD_RUN_ACTIVITY_SCORE = 1.0
BAD_RUN_TRAIN_ERROR = 1e6
BAD_RUN_METRIC_DEFAULTS: dict[str, float] = {
    "loss": BAD_RUN_LOSS,
    "train_error": BAD_RUN_TRAIN_ERROR,
    "eval_error": BAD_RUN_TRAIN_ERROR,
    "activity_score": BAD_RUN_ACTIVITY_SCORE,
}


def replace_bad_run_metrics(metrics: dict[str, float]) -> dict[str, float]:
    out = dict(metrics)
    for key, sentinel in BAD_RUN_METRIC_DEFAULTS.items():
        value = out.get(key)
        if value is None:
            continue
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            out[key] = sentinel
    return out

CELL_MASK_SEARCH_BOUNDS: dict[str, tuple[float, float]] = {
    "lambda_mse": (0.01, 100.0),
    "lambda_fr": (0.0, 2000.0),
    "mask_rate": (0.25, 0.75),
    "gradient_clip_max": (1.0, 100.0),
}

VARIANT_A_BOUNDS: dict[str, tuple[float, float]] = {
    **CELL_MASK_SEARCH_BOUNDS,
    "alpha": (0.0001, 0.2),
}

VARIANT_B_BOUNDS: dict[str, tuple[float, float]] = {
    **CELL_MASK_SEARCH_BOUNDS,
    "lambda_ab": (0.0, 100.0),
    "alpha": (0.0001, 0.2),
}


MAX_FR_FACTOR = 100.0
MAX_AB_FACTOR = MAX_FR_FACTOR / 10.0
def lambda_fr_mse_parameter_constraints(
    *,
    enabled: bool = True,
    max_fr_factor: float = MAX_FR_FACTOR,
    max_ab_factor: float = MAX_AB_FACTOR,
    bounds: SearchBounds | None = None,
) -> list[str]:
    """Ax parameter constraints

    `lambda_fr <= max_fr_factor * lambda_mse` when enabled.
    `lambda_ab <= max_ab_factor * lambda_mse` for variant B.
    """
    if not enabled:
        return []
    constraints = [f"lambda_fr - {max_fr_factor:g}*lambda_mse <= 0"]
    if bounds is not None and "lambda_ab" in bounds:
        constraints.append(f"lambda_ab - {max_ab_factor:g}*lambda_mse <= 0")
    return constraints


def search_bounds_from_config(
    run_config: ExperimentRunConfig,
    *,
    variant: BoVariant = "a",
    lambda_mse_bounds: tuple[float, float] | None = None,
    lambda_fr_bounds: tuple[float, float] | None = None,
    gradient_clip_bounds: tuple[float, float] | None = None,
    mask_rate_bounds: tuple[float, float] | None = None,
    lambda_ab_bounds: tuple[float, float] | None = None,
    alpha_bounds: tuple[float, float] | None = None,
) -> SearchBounds:
    """Compute Ax search bounds for the given BO variant """
    if run_config.config.training.mask_method != "cell":
        raise ValueError(
            f"We only perform BO for 'cell' mask type; got {run_config.config.training.mask_method!r}"
        )

    base_bounds = VARIANT_A_BOUNDS if variant == "a" else dict(VARIANT_B_BOUNDS)
    bounds: SearchBounds = dict(base_bounds)
    if lambda_mse_bounds is not None:
        bounds["lambda_mse"] = lambda_mse_bounds
    if lambda_fr_bounds is not None:
        bounds["lambda_fr"] = lambda_fr_bounds
    if gradient_clip_bounds is not None:
        bounds["gradient_clip_max"] = gradient_clip_bounds
    if mask_rate_bounds is not None:
        bounds["mask_rate"] = mask_rate_bounds
    if alpha_bounds is not None:
        bounds["alpha"] = alpha_bounds
    if variant == "b" and lambda_ab_bounds is not None:
        bounds["lambda_ab"] = lambda_ab_bounds
    return bounds


def apply_trial_params(
    base_config: ExperimentRunConfig,
    params: dict[str, Any],
    *,
    variant: BoVariant = "a",
) -> ExperimentRunConfig:
    """Override trial hyperparameters"""
    if base_config.config.training.mask_method != "cell":
        raise ValueError(
            f"We only perform BO for 'cell' mask type; got {base_config.config.training.mask_method!r}"
        )

    run_config = copy.deepcopy(base_config)
    config = run_config.config

    config.training = replace(
        config.training,
        lambda_mse=float(params["lambda_mse"]),
        lambda_fr=float(params["lambda_fr"]),
        gradient_clip_max=float(params["gradient_clip_max"]),
        mask_rate=float(params["mask_rate"]),
    )
    config.warmup = replace(config.warmup, warmup_shuffle=True)

    model_kwargs: dict[str, Any] = {"alpha": float(params["alpha"])}
    if variant == "b":
        model_kwargs["activation"] = "custom_learnable"
    config.model = replace(config.model, **model_kwargs)

    if run_config.experiment_type == SINGLE_ROOM_NAME:
        config.n_epochs = 2
    elif run_config.experiment_type == TWO_ROOMS_NAME:
        config.n_repetitions = 1
    elif run_config.experiment_type == MANY_ROOMS_NAME:
        config.n_cycles = 1
    else:
        raise ValueError(f"Unknown experiment_type: {run_config.experiment_type!r}")

    return run_config


def trial_lambda_ab(params: dict[str, Any], *, variant: BoVariant) -> float:
    if variant == "b":
        return float(params["lambda_ab"])
    return 0.0


def _train_config(
    experiment: RoomExperiment,
    step_size: int,
    *,
    lambda_ab: float = 0.0,
    lambda_unif: float = 0.0,
) -> TrainConfig:
    training = experiment.config.training
    return TrainConfig(
        mask_rate=training.mask_rate,
        mask_method=training.mask_method,
        step_size=step_size,
        lambda_mse=training.lambda_mse,
        lambda_fr=training.lambda_fr,
        learning_rate=training.learning_rate,
        gradient_clip_max=training.gradient_clip_max,
        carry_state=training.carry_state,
        lambda_ab=lambda_ab,
        lambda_unif=lambda_unif,
    )


def _room_ratemap_metrics(ratemap: np.ndarray) -> tuple[float, float, float]:
    """Return (nan_fraction, activity_fraction, activity_score) for one room."""
    peak = np.nanmax(ratemap, axis=(1, 2))
    nan_fraction = float(np.mean(np.isnan(peak)))

    finite_mask = ~np.isnan(peak)
    if not np.any(finite_mask):
        return nan_fraction, 0.0, BAD_RUN_ACTIVITY_SCORE

    finite_peak = peak[finite_mask]
    max_peak = float(np.max(finite_peak))
    if max_peak <= 0:
        activity_fraction = 0.0
    else:
        threshold = ACTIVITY_PEAK_FRACTION * max_peak
        active = (~np.isnan(peak)) & (peak > threshold)
        activity_fraction = float(np.mean(active))

    activity_score = abs(activity_fraction - ACTIVITY_TARGET)
    return nan_fraction, activity_fraction, activity_score


def mean_prediction_mse(
    rae: torch.nn.Module,
    traj_coord: np.ndarray,
    wsm,
    device: torch.device,
    *,
    mask_rate: float,
    mask_method: MaskMethod,
    step_size: int,
    n_segments: int | None,
    mask_generator: torch.Generator | None,
    carry_state: bool,
) -> float:
    """Mean unweighted MSE between predicted and ground-truth responses."""
    max_steps = traj_coord.shape[1] // step_size
    if max_steps == 0:
        return float("nan")

    if n_segments is None:
        n_steps = max_steps
    else:
        n_steps = min(n_segments, max_steps)

    losses: list[float] = []
    rae.eval()
    init_states = None
    with torch.no_grad():
        for step in range(n_steps):
            tc = traj_coord[:, step * step_size : (step + 1) * step_size]
            gt_res_full = torch.as_tensor(
                wsm.get_response(tc), dtype=torch.float32, device=device
            )
            masked_res = apply_input_mask(
                gt_res_full[:, :-1],
                mask_rate,
                mask_method=mask_method,
                generator=mask_generator,
            )
            gt_res = gt_res_full[:, 1:]
            segment_init = [init_states] if init_states is not None else None
            pred, states = rae(masked_res, init_states=segment_init)
            losses.append(float(F.mse_loss(pred, gt_res).item()))
            if carry_state:
                init_states = states[0][:, -1, :].clone()
    return float(np.mean(losses)) if losses else float("nan")


def custom_activation_parameter_stats(rae: torch.nn.Module) -> dict[str, float]:
    """Mean + SEM a/b parameters for variant B"""
    a_parts: list[np.ndarray] = []
    b_parts: list[np.ndarray] = []
    # extract all ab
    for module in rae.modules():
        if isinstance(module, CustomLearnableActivation):
            a_parts.append(module.a.detach().cpu().numpy().ravel())
            b_parts.append(module.b.detach().cpu().numpy().ravel())
    if not a_parts:
        return {}

    a = np.concatenate(a_parts)
    b = np.concatenate(b_parts)
    n = a.size
    # then compute stats
    sem_a = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    sem_b = float(np.std(b, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {
        "activation_a_mean": float(np.mean(a)),
        "activation_a_sem": sem_a,
        "activation_b_mean": float(np.mean(b)),
        "activation_b_sem": sem_b,
    }


def compute_eval_loss(
    rae: torch.nn.Module,
    room: Room,
    device: torch.device,
    *,
    lambda_mse: float,
    lambda_fr: float,
    mask_rate: float,
    mask_method: MaskMethod,
    step_size: int,
    n_eval_segments: int | None,
    mask_generator: torch.Generator | None,
    carry_state: bool,
    lambda_unif: float = 0.0,
) -> float:
    """Mean eval loss on held-out trajectories (no gradient)."""
    traj_coord = room.eval_traj_coord
    max_steps = traj_coord.shape[1] // step_size
    if max_steps == 0:
        return float("nan")

    if n_eval_segments is None:
        n_steps = min(500, max_steps)
    else:
        n_steps = min(n_eval_segments, max_steps)
    losses: list[float] = []

    rae.eval()
    init_states = None
    with torch.no_grad():
        for step in range(n_steps):
            tc = traj_coord[:, step * step_size : (step + 1) * step_size]
            gt_res_full = torch.as_tensor(
                room.wsm.get_response(tc), dtype=torch.float32, device=device
            )
            masked_res = apply_input_mask(
                gt_res_full[:, :-1],
                mask_rate,
                mask_method=mask_method,
                generator=mask_generator,
            )
            gt_res = gt_res_full[:, 1:]
            segment_init = [init_states] if init_states is not None else None
            pred, states = rae(masked_res, init_states=segment_init)
            h = states[0]
            loss = F.mse_loss(pred, gt_res) * lambda_mse + fr_loss(h) * lambda_fr
            if lambda_unif > 0:
                loss = loss + unif_loss(h) * lambda_unif
            losses.append(float(loss.item()))
            if carry_state:
                init_states = h[:, -1, :].clone()
    return float(np.mean(losses))


def compute_trial_metrics(
    rae: torch.nn.Module,
    rooms: list[Room],
    experiment: RoomExperiment,
    *,
    mask_generator: torch.Generator | None,
    return_ratemaps: bool = False,
    variant: BoVariant = "a",
    lambda_unif: float = 0.0,
) -> dict[str, float] | tuple[dict[str, float], list[np.ndarray]]:
    config = experiment.config
    device = config.resolve_device()
    train_step_size = round(config.training.train_step_size_s / experiment.dt)

    activity_fractions: list[float] = []
    activity_scores: list[float] = []
    losses: list[float] = []
    train_errors: list[float] = []
    eval_errors: list[float] = []
    ratemaps: list[np.ndarray] = []

    train_mask_generator = torch.Generator(device="cpu")
    train_mask_generator.manual_seed(config.training.mask_rng_seed + 1)

    for room in rooms:
        ratemap_result = compute_ratemap(
            rae,
            room.eval_traj_coord,
            room.wsm,
            room.arena_map,
            device,
            config.training.mask_rate,
            train_step_size,
            n_test=config.training.record_n_eval_segments,
            mask_method=config.training.mask_method,
            mask_generator=mask_generator,
            show_progress=False,
            return_mse=True,
        )
        ratemap, eval_error = ratemap_result
        if return_ratemaps:
            ratemaps.append(ratemap)
        _, activity_fraction, activity_score = _room_ratemap_metrics(ratemap)
        activity_fractions.append(activity_fraction)
        activity_scores.append(activity_score)
        eval_errors.append(eval_error)

        train_mses: list[float] = []
        for traj_idx in range(room.main_traj.shape[0]):
            tc_full = room.main_traj[traj_idx : traj_idx + 1]
            train_mses.append(
                mean_prediction_mse(
                    rae,
                    tc_full,
                    room.wsm,
                    device,
                    mask_rate=config.training.mask_rate,
                    mask_method=config.training.mask_method,
                    step_size=train_step_size,
                    n_segments=None,
                    mask_generator=train_mask_generator,
                    carry_state=config.training.carry_state,
                )
            )
        train_errors.append(float(np.nanmean(train_mses)))

        loss = compute_eval_loss(
            rae,
            room,
            device,
            lambda_mse=config.training.lambda_mse,
            lambda_fr=config.training.lambda_fr,
            mask_rate=config.training.mask_rate,
            mask_method=config.training.mask_method,
            step_size=train_step_size,
            n_eval_segments=config.training.record_n_eval_segments,
            mask_generator=mask_generator,
            carry_state=config.training.carry_state,
            lambda_unif=lambda_unif,
        )
        losses.append(loss)

    activity_fraction = float(np.nanmean(activity_fractions)) if activity_fractions else 0.0
    if np.isnan(activity_fraction):
        activity_fraction = 0.0
    activity_score = float(np.nanmean(activity_scores))
    if np.isnan(activity_score):
        activity_score = BAD_RUN_ACTIVITY_SCORE
    loss = float(np.nanmean(losses)) if losses else float("nan")
    train_error = float(np.nanmean(train_errors)) if train_errors else float("nan")
    eval_error = float(np.nanmean(eval_errors)) if eval_errors else float("nan")

    metrics: dict[str, float] = replace_bad_run_metrics(
        {
            "loss": loss,
            "train_error": train_error,
            "eval_error": eval_error,
            "activity_score": activity_score,
            "activity_fraction": activity_fraction,
        }
    )
    if variant == "b":
        metrics.update(custom_activation_parameter_stats(rae))
    if return_ratemaps:
        return metrics, ratemaps
    return metrics


def save_trial_ratemap(
    output_dir: Path,
    trial_index: int,
    ratemap: np.ndarray,
    params: dict[str, Any],
    metrics: dict[str, float] | None = None,
) -> Path:
    ratemaps_dir = output_dir / "ratemaps"
    ratemaps_dir.mkdir(parents=True, exist_ok=True)
    out_path = ratemaps_dir / f"trial_{trial_index:04d}.npz"
    payload: dict[str, Any] = {
        "ratemap": ratemap,
        "trial_index": np.int32(trial_index),
        "params": np.array([json.dumps(params)], dtype=object),
    }
    if metrics is not None:
        for key, value in metrics.items():
            if key == "ratemap_path":
                continue
            if isinstance(value, (int, float, np.floating, np.integer)):
                payload[key] = np.float64(value)
    np.savez_compressed(out_path, **payload)
    return out_path


def estimate_trial_segments(experiment: RoomExperiment, rooms: list[Room]) -> int:
    """Count TBPTT segments for warmup plus the BO trial training protocol."""
    config = experiment.config
    warmup_step_size = round(config.warmup.warmup_step_size_s / experiment.dt)
    train_step_size = round(config.training.train_step_size_s / experiment.dt)

    warmup_segments = sum(
        room.n_warm * (room.traj_coord.shape[1] // warmup_step_size) for room in rooms
    )

    protocol = experiment.build_protocol()
    groups = group_protocol(protocol)
    if not groups:
        return warmup_segments

    train_segments = 0
    for _, group_visits in groups:
        for visit in group_visits:
            room = rooms[visit.room_index]
            n_segments_per_traj = room.main_traj.shape[1] // train_step_size
            train_segments += len(visit.traj_indices) * n_segments_per_traj

    return warmup_segments + train_segments


def _build_trial_model(
    experiment: RoomExperiment,
    device: torch.device,
) -> torch.nn.Module:
    config = experiment.config
    return build_rae(
        config.room.n_wsm_cells,
        config.training.n_hidden,
        device,
        model=config.model,
    )


def run_hyperparam_trial(
    run_config: ExperimentRunConfig,
    params: dict[str, Any],
    *,
    variant: BoVariant = "a",
    trial_index: int | None = None,
    output_dir: Path | None = None,
    show_progress: bool = False,
    lambda_unif: float = 0.0,
) -> dict[str, float]:
    """Run warmup plus the configured training protocol and return Ax trial metrics."""
    experiment = create_experiment(run_config)
    config = experiment.config
    device = config.resolve_device()
    rooms = experiment.load_rooms()
    lambda_ab = trial_lambda_ab(params, variant=variant)

    rae = _build_trial_model(experiment, device)
    extractor = optimizer_extractor_from_dict(dict(experiment.optimizer_config))
    optimizer = extractor.create_optimizer(rae.parameters())

    mask_generator = torch.Generator(device="cpu")
    mask_generator.manual_seed(config.training.mask_rng_seed)

    warmup_step_size = round(config.warmup.warmup_step_size_s / experiment.dt)
    warmup_params = _train_config(
        experiment, warmup_step_size, lambda_ab=lambda_ab, lambda_unif=lambda_unif
    )
    progress_level = 1 if show_progress else 0

    run_warmup(
        rae,
        optimizer,
        rooms,
        device,
        warmup_params,
        gaussian_sigma=config.warmup.warmup_gaussian_sigma,
        step_size=warmup_step_size,
        warmup_shuffle=config.warmup.warmup_shuffle,
        warmup_shuffle_seed=config.warmup.warmup_shuffle_seed,
        mask_generator=mask_generator,
        show_progress_level=progress_level,
        experiment_name=experiment.name,
    )

    train_step_size = round(config.training.train_step_size_s / experiment.dt)
    train_params = _train_config(
        experiment, train_step_size, lambda_ab=lambda_ab, lambda_unif=lambda_unif
    )

    protocol = experiment.build_protocol()
    groups = group_protocol(protocol)

    for _, group_visits in groups:
        for visit in group_visits:
            room = rooms[visit.room_index]
            for traj_idx in visit.traj_indices:
                tc_full = room.main_traj[traj_idx : traj_idx + 1]
                train_trajectory_segments(
                    rae,
                    optimizer,
                    tc_full,
                    room.wsm,
                    device,
                    train_params,
                    mask_generator=mask_generator,
                )

    need_ratemaps = output_dir is not None and trial_index is not None
    if need_ratemaps:
        metrics, ratemaps = compute_trial_metrics(
            rae,
            rooms,
            experiment,
            mask_generator=mask_generator,
            return_ratemaps=True,
            variant=variant,
            lambda_unif=lambda_unif,
        )
        ratemap_path = save_trial_ratemap(
            output_dir, trial_index, ratemaps[0], params, metrics=metrics
        )
        metrics["ratemap_path"] = str(ratemap_path)
        return metrics

    metrics = compute_trial_metrics(
        rae,
        rooms,
        experiment,
        mask_generator=mask_generator,
        variant=variant,
        lambda_unif=lambda_unif,
    )
    return metrics
