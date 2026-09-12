"""RAE training and room experiments"""
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from pathlib import Path

from core.experiment import RoomExperiment
from core.gradient_signals import StepGradientSignals, capture_step_signals
from core.progress import TrainingProgress, group_protocol
from core.utils import MaskMethod, apply_input_mask, compute_ratemap
from experiments.common.paths import ExperimentPaths
from experiments.common.ratemaps_io import (
    TrajectoryRatemapBatch,
    capture_tag_for_segment,
    trajectory_ratemap_path,
)
from models.custom_learnable_activation import custom_ab_reg_loss
from models.utils import build_rae, build_tracked_units, order_node_ids, seed_model_init
from optimizers import optimizer_extractor_from_dict
from optimizers.base import OptimizerSignalExtractor

if TYPE_CHECKING:
    from core.plateau_training import PlateauConfig


def fr_loss(states):
    mean_fr = torch.mean(states, dim=(0, 1))
    return torch.pow(mean_fr, 2).mean()


def unif_loss(states: torch.Tensor) -> torch.Tensor:
    std_fr = torch.std(states, dim=(0, 1), unbiased=False)
    return torch.pow(1.0 - std_fr, 2).mean()


def weight_l2_reg_loss(rae: torch.nn.Module) -> torch.Tensor:
    """Mean squared L2 over trainable parameters (per-parameter mean, then averaged)."""
    parts = [p.square().mean() for p in rae.parameters() if p.requires_grad]
    if not parts:
        device = next(rae.parameters()).device
        return torch.tensor(0.0, device=device)
    return torch.stack(parts).mean()


@dataclass
class TrainConfig:
    """Hyperparameters for episodic RAE training."""

    mask_rate: float = 0.5
    mask_method: MaskMethod = "cell"
    step_size: int = 20
    lambda_mse: float = 1.0
    lambda_fr: float = 200.0
    gradient_clip_max: float | None = None
    lambda_ab: float = 0.0 # ! experimental, leave 0/unused for current experiments
    lambda_unif: float = 0.0 # ! experimental, leave 0/unused for current experiments
    lambda_weight: float = 0.0 # ! experimental, leave 0/unused for current experiments
    # True <=> the ending hidden state of one segment is propagated to the initial state of the next segment of tBPTT
    # /!\ changes behavior of training
    carry_state: bool = False


@dataclass
class SegmentResult:
    """Outcome of one truncated-BPTT training step (`train_one_segment`)."""

    loss: float
    final_state: torch.Tensor | None = None  # (B, H), detached; set iff carry_state
    gradient_signals: StepGradientSignals | None = None
    optimizer_signals: dict[str, dict[str, Any]] = field(default_factory=dict)
    weight_snapshots: dict[str, np.ndarray] = field(default_factory=dict)
    shampoo_blocks: dict[str, np.ndarray] = field(default_factory=dict)
    on_opt_scale_factors: np.ndarray | None = None  # (H,) per-hidden-unit multipliers
    plateau_units: np.ndarray | None = None  # hidden units rewritten by a plateau
    plateau_damping: np.ndarray | None = None  # (H,) incoming-grad multipliers


def model_has_nonfinite(rae: torch.nn.Module) -> bool:
    """Return True when any model parameter is non-finite."""
    for param in rae.parameters():
        if not torch.isfinite(param).all():
            return True
    return False


def tensor_has_nonfinite(tensor: torch.Tensor) -> bool:
    """Return True when any element of a tensor is non-finite."""
    return not torch.isfinite(tensor).all()


def forward_has_nonfinite(
    rae: torch.nn.Module,
    pred: torch.Tensor,
    h: torch.Tensor,
    loss: torch.Tensor,
) -> bool:
    """Return True when the forward pass produced non-finite values."""
    if not torch.isfinite(loss):
        return True
    if tensor_has_nonfinite(pred) or tensor_has_nonfinite(h):
        return True
    return model_has_nonfinite(rae)


def segment_has_nonfinite(result: SegmentResult) -> bool:
    """Return True when a segment loss is non-finite."""
    return not math.isfinite(result.loss)


def _is_numerical_optimizer_failure(exc: BaseException) -> bool:
    """Return True when an optimizer step failed due to NaN/Inf divergence."""
    if type(exc).__qualname__ == "PreconditionerValueError":
        return True
    message = str(exc).casefold()
    return "nan" in message or "inf" in message


def _capture_weight_snapshots(model: torch.nn.Module) -> dict[str, np.ndarray]:
    """Return full post-step weights as `weight__<param_name>` float32 arrays."""
    snapshots: dict[str, np.ndarray] = {}
    for name, param in model.named_parameters():
        safe = name.replace(".", "__")
        snapshots[f"weight__{safe}"] = param.detach().float().cpu().numpy()
    return snapshots


def train_one_segment(
    rae: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tc: np.ndarray,
    wsm,
    device: torch.device,
    config: TrainConfig,
    *,
    mask_generator: torch.Generator | None = None,
    init_state: torch.Tensor | None = None,
    capture_gradients: bool = False,
    extractor: OptimizerSignalExtractor | None = None,
    on_activity: Callable[[np.ndarray, torch.Tensor], None] | None = None,
    abort_on_nan: bool = False,
) -> SegmentResult:
    """Run exactly one truncated-BPTT gradient update on one pre-sliced chunk.

    Captures gradients and optimizer signals (see extractors).

    `tc` is pre-sliced `(B=1, step_size, 2)` coordinate chunk 
    when `config.carry_state` is true, hidden state is carried between segments of tBPTT
    """
    rae.train()
    optimizer.zero_grad()
    gt_res_full = torch.as_tensor(wsm.get_response(tc), dtype=torch.float32, device=device)
    masked_res = apply_input_mask(
        gt_res_full[:, :-1],
        config.mask_rate,
        mask_method=config.mask_method,
        generator=mask_generator,
    )
    gt_res = gt_res_full[:, 1:]

    init_states = [init_state] if init_state is not None else None
    pred, states = rae(masked_res, init_states=init_states)
    h = states[0]
    if capture_gradients:
        h.retain_grad()

    loss = (
        F.mse_loss(pred, gt_res) * config.lambda_mse
        + fr_loss(h) * config.lambda_fr
    )
    if config.lambda_unif > 0:
        loss = loss + unif_loss(h) * config.lambda_unif
    if config.lambda_ab > 0:
        loss = loss + config.lambda_ab * custom_ab_reg_loss(rae)
    if config.lambda_weight > 0:
        loss = loss + config.lambda_weight * weight_l2_reg_loss(rae)

    if abort_on_nan and forward_has_nonfinite(rae, pred, h, loss):
        loss_value = float(loss.detach().item())
        if not math.isfinite(loss_value):
            loss_value = float("nan")
        return SegmentResult(loss=loss_value)

    loss.backward()

    gradient_signals = None
    if capture_gradients:
        gradient_signals = capture_step_signals(
            states=h,
            states_grad=h.grad,
            pred=pred,
            gt_res=gt_res,
            lambda_mse=config.lambda_mse,
            lambda_fr=config.lambda_fr,
            readout_weight=rae.readout_layer.weight,
        )

    optimizer_signals: dict[str, dict[str, Any]] = {}
    if extractor is not None:
        optimizer_signals.update(extractor.on_before_step(rae, optimizer))

    if config.gradient_clip_max is not None:
        torch.nn.utils.clip_grad_norm_(rae.parameters(), config.gradient_clip_max)
    try:
        optimizer.step()
    except Exception as exc:
        if abort_on_nan and _is_numerical_optimizer_failure(exc):
            loss_value = float(loss.detach().item())
            if not math.isfinite(loss_value):
                loss_value = float("nan")
            return SegmentResult(loss=loss_value)
        raise

    shampoo_blocks: dict[str, np.ndarray] = {}
    if extractor is not None:
        optimizer_signals.update(extractor.on_after_step(rae, optimizer))
        shampoo_blocks = extractor.on_after_step_shampoo_blocks(rae, optimizer)

    if on_activity is not None:
        on_activity(tc[:, 1:], h.detach())

    final_state = h[:, -1, :].detach().clone() if config.carry_state else None

    return SegmentResult(
        loss=float(loss.detach().item()),
        final_state=final_state,
        gradient_signals=gradient_signals,
        optimizer_signals=optimizer_signals,
        weight_snapshots=_capture_weight_snapshots(rae),
        shampoo_blocks=shampoo_blocks,
    )


def train_trajectory_segments(
    rae: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tc_full: np.ndarray,
    wsm,
    device: torch.device,
    config: TrainConfig,
    *,
    mask_generator: torch.Generator | None = None,
    capture_gradients: bool = False,
    extractor: OptimizerSignalExtractor | None = None,
    on_segment: Callable[[int, int, SegmentResult], None] | None = None,
    on_activity: Callable[[np.ndarray, torch.Tensor], None] | None = None,
    segment_progress: tqdm | None = None,
    abort_on_nan: bool = False,
) -> bool:
    """Train all truncated-BPTT segments of one `(B, T, 2)` trajectory chunk.

    Returns False when `abort_on_nan` is set and non-finite values appear in the
    forward pass (before backward), after an optimizer step, or when an optimizer
    step raises a numerical failure (e.g. Shampoo preconditioner NaN).
    """
    n_segments = tc_full.shape[1] // config.step_size
    state: torch.Tensor | None = None
    for seg_idx in range(n_segments):
        tc = tc_full[:, seg_idx * config.step_size : (seg_idx + 1) * config.step_size]
        result = train_one_segment(
            rae,
            optimizer,
            tc,
            wsm,
            device,
            config,
            mask_generator=mask_generator,
            init_state=state if config.carry_state else None,
            capture_gradients=capture_gradients,
            extractor=extractor,
            on_activity=on_activity,
            abort_on_nan=abort_on_nan,
        )
        if abort_on_nan and (
            segment_has_nonfinite(result) or model_has_nonfinite(rae)
        ):
            return False
        if on_segment is not None:
            on_segment(seg_idx, n_segments, result)
        if segment_progress is not None:
            segment_progress.update(1)
        if config.carry_state:
            state = result.final_state
    return True


def _train_config(experiment: RoomExperiment, step_size: int) -> TrainConfig:
    training = experiment.config.training
    return TrainConfig(
        mask_rate=training.mask_rate,
        mask_method=training.mask_method,
        step_size=step_size,
        lambda_mse=training.lambda_mse,
        lambda_fr=training.lambda_fr,
        gradient_clip_max=training.gradient_clip_max,
        carry_state=training.carry_state,
    )


def _save_segment_signals(
    signals_dir : Path,
    *,
    tag: str,
    seg_idx: int,
    result: SegmentResult,
    node_ids: list[str],
) -> None:
    traj_dir = signals_dir / tag
    traj_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, np.ndarray] = {"loss": np.float32(result.loss)}
    if result.gradient_signals is not None:
        payload.update(result.gradient_signals.to_numpy())

    payload["opt_signals"] = np.array([result.optimizer_signals], dtype=object)
    payload["node_ids"] = np.array(node_ids)
    payload.update(result.weight_snapshots)
    payload.update(result.shampoo_blocks)
    if result.on_opt_scale_factors is not None:
        payload["on_opt_scale_factors"] = result.on_opt_scale_factors.astype(np.float32)
    if result.plateau_units is not None:
        payload["plateau_units"] = result.plateau_units.astype(np.int32)
    if result.plateau_damping is not None:
        payload["plateau_damping"] = result.plateau_damping.astype(np.float32)

    np.savez_compressed(traj_dir / f"segment_{seg_idx}.npz", **payload)


def _print_plan_estimate(experiment: RoomExperiment, rooms, protocol) -> None:
    capture_frequency = experiment.ratemap_capture()
    train_step_size = round(experiment.config.training.train_step_size_s / experiment.dt)
    total_segments = 0
    total_captures = 0
    for visit in protocol:
        room = rooms[visit.room_index]
        n_segments_per_traj = room.main_traj.shape[1] // train_step_size
        total_segments += len(visit.traj_indices) * n_segments_per_traj
        n_rooms_captured = len(capture_frequency.rooms_for_capture(visit.room_index))
        captures_per_traj = (1 if capture_frequency.should_capture_before_trajectory() else 0) + sum(
            1
            for seg_idx in range(n_segments_per_traj)
            if capture_frequency.should_capture_after_segment(seg_idx, n_segments_per_traj)
        )
        total_captures += len(visit.traj_indices) * captures_per_traj * n_rooms_captured

    n_hidden = experiment.config.training.n_hidden
    arena_size = rooms[0].arena_map.size
    mb_per_capture_fp16 = 2 * n_hidden * arena_size / 1e6
    print(
        f"[{experiment.name}] planned: {total_segments} gradient/optimizer-signal segments, "
        f"{total_captures} activation (rate-map) captures across {len(protocol)} visit(s) "
        f"(~{mb_per_capture_fp16:.1f} MB fp16 per capture; saved batched per trajectory) -- "
        f"tune capture_every_n_segments/visit counts if this is too much disk I/O."
    )


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


def _write_run_config_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n")


def _build_run_config_payload(
    experiment: RoomExperiment,
    *,
    source_config_path: Path | None,
    experiment_type: str | None,
    started_at: str,
    completed_at: str | None = None,
    alt_training: Literal["on_opt", "plateau"] | None = None,
    kappa: float | None = None,
    plateau_config: Any | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source_config": str(source_config_path.resolve()) if source_config_path else None,
        "experiment_type": experiment_type or experiment.name,
        "optimizer": dict(experiment.optimizer_config),
        "experiment_config": asdict(experiment.config),
        "started_at": started_at,
    }
    if alt_training is not None:
        payload["alt_training"] = alt_training
    if kappa is not None:
        payload["kappa"] = kappa
    if plateau_config is not None:
        payload["plateau_config"] = asdict(plateau_config)
    if completed_at is not None:
        payload["completed_at"] = completed_at
    return payload


def run_experiment(
    experiment: RoomExperiment,
    *,
    show_progress_level: int | None = None,
    source_config_path: Path | None = None,
    experiment_type: str | None = None,
    alt_training: Literal["on_opt", "plateau"] | None = None,
    kappa: float = 2.7,
    plateau_config: "PlateauConfig | None" = None,
) -> ExperimentPaths:
    """Run warmup + main training for a room-based experiment."""
    # import here to avoid circular imports
    from core.alt_training import LocalMseUnitScaler, train_trajectory_segments_on_opt
    from core.plateau_training import (
        PlateauConfig,
        PlateauState,
        resolve_plateau_layers,
        train_trajectory_segments_plateau,
    )
    from core.warmup import run_warmup

    if show_progress_level is None:
        show_progress_level = experiment.default_show_progress_level
    if show_progress_level not in range(4):
        raise ValueError(f"show_progress_level must be 0-3, got {show_progress_level}")
    if alt_training == "plateau" and plateau_config is None:
        plateau_config = PlateauConfig()

    config = experiment.config
    device = config.resolve_device()
    paths = experiment.resolve_paths()
    paths.mkdirs()

    started_at = datetime.now(timezone.utc).isoformat()
    config_json_path = paths.results_dir / "config.json"
    _write_run_config_json(
        config_json_path,
        _build_run_config_payload(
            experiment,
            source_config_path=source_config_path,
            experiment_type=experiment_type,
            started_at=started_at,
            alt_training=alt_training,
            kappa=kappa if alt_training == "on_opt" else None,
            plateau_config=plateau_config,
        ),
    )

    rooms = experiment.load_rooms()
    protocol = experiment.build_protocol()
    groups = group_protocol(protocol)
    capture_frequency = experiment.ratemap_capture()
    _print_plan_estimate(experiment, rooms, protocol)

    seed_model_init(config.training.init_seed)
    rae = build_rae(
        config.room.n_wsm_cells,
        config.training.n_hidden,
        device,
        model=config.model,
    )
    extractor = optimizer_extractor_from_dict(dict(experiment.optimizer_config))
    units = build_tracked_units(rae, max_units_per_layer=config.training.max_units_per_layer)
    extractor.set_units(units)
    node_ids = order_node_ids(units)
    optimizer = extractor.create_optimizer(rae.parameters())

    mask_generator = torch.Generator(device="cpu")
    mask_generator.manual_seed(config.training.mask_rng_seed)

    warmup_step_size = round(config.warmup.warmup_step_size_s / experiment.dt)
    warmup_params = _train_config(experiment, warmup_step_size)
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
        show_progress_level=show_progress_level,
        experiment_name=experiment.name,
    )

    train_step_size = round(config.training.train_step_size_s / experiment.dt)
    train_params = _train_config(experiment, train_step_size)
    glocal_scaler = (
        LocalMseUnitScaler(n_hidden=config.training.n_hidden)
        if alt_training == "on_opt"
        else None
    )
    plateau_state = None
    plateau_layers = None
    if alt_training == "plateau":
        plateau_layers = resolve_plateau_layers(rae)
        plateau_state = PlateauState(
            config.training.n_hidden,
            config=plateau_config,
            device=device,
            dt=experiment.dt,
        )

    multi_room = len(rooms) > 1

    def _flush_trajectory_ratemaps(
        base_tag: str,
        batches: dict[int, TrajectoryRatemapBatch],
        room_indices: list[int],
    ) -> None:
        for ridx in room_indices:
            batch = batches[ridx]
            if len(batch) == 0:
                continue
            out_path = trajectory_ratemap_path(
                paths.ratemaps_dir,
                base_tag,
                room_idx=ridx,
                multi_room=multi_room,
            )
            batch.save(out_path)

    def _capture_into_batch(
        tag: str,
        room_indices: list[int],
        batches: dict[int, TrajectoryRatemapBatch],
    ) -> None:
        rae.eval()
        for ridx in room_indices:
            room = rooms[ridx]
            rm = compute_ratemap(
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
            )
            batches[ridx].append(tag, rm)
        rae.train()

    progress = TrainingProgress(
        show_progress_level,
        experiment_name=experiment.name,
        group_unit=experiment.progress_group_unit,
        n_groups=len(groups),
    )
    for _, group_visits in groups:
        progress.begin_group(len(group_visits))
        for visit in group_visits:
            room = rooms[visit.room_index]
            main_traj = room.main_traj
            n_traj = len(visit.traj_indices)
            progress.begin_room(n_traj)
            for traj_idx in visit.traj_indices:
                tc_full = main_traj[traj_idx : traj_idx + 1]  # (1, T, 2), no batching
                base_tag = f"{visit.tag}_traj{traj_idx}"
                capture_rooms = capture_frequency.rooms_for_capture(visit.room_index)
                n_segments = tc_full.shape[1] // train_step_size
                segment_bar = progress.begin_trajectory(n_segments)

                ratemap_batches = {ridx: TrajectoryRatemapBatch() for ridx in capture_rooms}

                if capture_frequency.should_capture_before_trajectory():
                    _capture_into_batch(
                        capture_tag_for_segment(pre=True),
                        capture_rooms,
                        ratemap_batches,
                    )

                def _on_segment(seg_idx: int, n_segs: int, result: SegmentResult) -> None:
                    _save_segment_signals(
                        paths.signals_dir,
                        tag=base_tag,
                        seg_idx=seg_idx,
                        result=result,
                        node_ids=node_ids,
                    )
                    if capture_frequency.should_capture_after_segment(seg_idx, n_segs):
                        _capture_into_batch(
                            capture_tag_for_segment(seg_idx=seg_idx),
                            capture_rooms,
                            ratemap_batches,
                        )

                if alt_training == "plateau":
                    assert plateau_state is not None and plateau_layers is not None
                    train_trajectory_segments_plateau(
                        rae,
                        optimizer,
                        tc_full,
                        room.wsm,
                        device,
                        train_params,
                        state=plateau_state,
                        layers=plateau_layers,
                        mask_generator=mask_generator,
                        capture_gradients=True,
                        extractor=extractor,
                        on_segment=_on_segment,
                        segment_progress=segment_bar,
                    )
                elif alt_training == "on_opt":
                    assert glocal_scaler is not None
                    train_trajectory_segments_on_opt(
                        rae,
                        optimizer,
                        tc_full,
                        room.wsm,
                        device,
                        train_params,
                        scaler=glocal_scaler,
                        kappa=kappa,
                        mask_generator=mask_generator,
                        capture_gradients=True,
                        extractor=extractor,
                        on_segment=_on_segment,
                        segment_progress=segment_bar,
                    )
                else:
                    train_trajectory_segments(
                        rae,
                        optimizer,
                        tc_full,
                        room.wsm,
                        device,
                        train_params,
                        mask_generator=mask_generator,
                        capture_gradients=True,
                        extractor=extractor,
                        on_segment=_on_segment,
                        segment_progress=segment_bar,
                    )
                _flush_trajectory_ratemaps(base_tag, ratemap_batches, capture_rooms)
                progress.end_trajectory()
            progress.end_room()

            if visit.is_last_in_group:
                torch.save({"rae": rae.state_dict()}, paths.ckpt_dir / f"{visit.group_tag}.pth")
        progress.end_group()
    progress.close()

    torch.save({"rae": rae.state_dict()}, paths.ckpt_dir / "final.pth")
    _write_run_config_json(
        config_json_path,
        _build_run_config_payload(
            experiment,
            source_config_path=source_config_path,
            experiment_type=experiment_type,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
            alt_training=alt_training,
            kappa=kappa if alt_training == "on_opt" else None,
            plateau_config=plateau_config,
        ),
    )
    if show_progress_level >= 1:
        tqdm.write(f"Done. Results in {paths.results_dir}")
    return paths


_DEPRECATED_REEXPORTS = frozenset({
    "TrainMode",
    "rebatch_trajectories",
    "rebatch_for_mode",
    "prepare_default_trajectory",
    "resolve_n_segments",
    "train_steps",
    "train_room_visit",
    "train_rae_epochs",
})


def __getattr__(name: str):
    if name in _DEPRECATED_REEXPORTS:
        import core.old_training as old_training

        return getattr(old_training, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
