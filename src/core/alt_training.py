"""Alt training variants: LoRA BTSP adapter or on-optimizer grad scaling."""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from core.gradient_signals import capture_step_signals, compute_g_local_mse
from core.split_weight_wrapper import SplitWeightWrapper
from core.training import (
    SegmentResult,
    TrainConfig,
    forward_has_nonfinite,
    fr_loss,
    model_has_nonfinite,
    segment_has_nonfinite,
    unif_loss,
    weight_l2_reg_loss,
    _capture_weight_snapshots,
    _is_numerical_optimizer_failure,
)
from core.utils import apply_input_mask
from models.custom_learnable_activation import custom_ab_reg_loss
from optimizers.base import OptimizerSignalExtractor

GLOCAL_N_RUNNING_SEGMENTS = 10
GLOCAL_SIGMA_EPS = 1e-8
ON_OPT_SCALE_CLIP_LO = 0.0
ON_OPT_SCALE_CLIP_HI = 5.0

AltTrainingVariant = Literal["lora", "on_opt", "plateau"]


def alt_training_path_tag(
    variant: AltTrainingVariant,
    *,
    rank: int | None = None,
) -> str:
    """Directory/experiment suffix for an alt-training run."""
    if variant == "lora":
        if rank is None:
            raise ValueError("rank is required for alt_training variant 'lora'")
        return f"_altTLORA_r{rank}"
    if variant == "on_opt":
        return "_altTOO"
    if variant == "plateau":
        return "_plateau"
    raise ValueError(f"Unknown alt_training variant: {variant!r}")


@dataclass
class LocalMseUnitScaler:
    """Per-hidden-unit grad multipliers from rolling g_local_mse z-scores."""

    n_hidden: int
    n_running: int = GLOCAL_N_RUNNING_SEGMENTS
    sigma_eps: float = GLOCAL_SIGMA_EPS
    n_segments: int = 0

    def __post_init__(self) -> None:
        self._history: deque[np.ndarray] = deque(maxlen=self.n_running)

    def unit_scales_from_g_local(self, g_local_mse: torch.Tensor) -> torch.Tensor:
        unit_means = g_local_mse[0].mean(dim=0).detach().cpu().numpy().astype(np.float64)
        if unit_means.shape[0] != self.n_hidden:
            raise ValueError(
                f"Expected {self.n_hidden} hidden units, got g_local_mse with H={unit_means.shape[0]}"
            )

        if len(self._history) < self.n_running:
            scales = np.ones(self.n_hidden, dtype=np.float64)
        else:
            hist = np.stack(list(self._history), axis=0)
            mu_h = hist.mean(axis=0)
            sigma_h = hist.std(axis=0, ddof=0)
            z = (unit_means - mu_h) / np.maximum(sigma_h, self.sigma_eps)
            scales = 1.0 / (1.0 + np.abs(z))

        self._history.append(unit_means)
        self.n_segments += 1
        return torch.as_tensor(scales, dtype=torch.float32, device=g_local_mse.device)

    def clipped_on_opt_scale_factors(
        self,
        g_local_mse: torch.Tensor,
        kappa: float,
        *,
        clip_lo: float = ON_OPT_SCALE_CLIP_LO,
        clip_hi: float = ON_OPT_SCALE_CLIP_HI,
    ) -> torch.Tensor:
        """Per-unit grad multipliers: clip(kappa * inverse_z_score, clip_lo, clip_hi)."""
        unit_scales = self.unit_scales_from_g_local(g_local_mse)
        return unit_scales.mul(float(kappa)).clamp(clip_lo, clip_hi)


def scale_all_parameter_grads(
    model: torch.nn.Module,
    scale_factors: torch.Tensor,
) -> None:
    """Scale every parameter gradient using per-hidden-unit factors where shapes match."""
    n_hidden = int(scale_factors.shape[0])
    mean_scale = float(scale_factors.mean().item())
    for param in model.parameters():
        grad = param.grad
        if grad is None:
            continue
        if grad.ndim >= 2 and grad.shape[0] == n_hidden:
            grad.mul_(scale_factors.view(-1, *([1] * (grad.ndim - 1))))
        elif grad.ndim >= 2 and grad.shape[-1] == n_hidden:
            grad.mul_(scale_factors.view(*([1] * (grad.ndim - 1)), -1))
        elif grad.ndim == 1 and grad.shape[0] == n_hidden:
            grad.mul_(scale_factors)
        else:
            grad.mul_(mean_scale)


def train_one_segment_alt(
    rae: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tc: np.ndarray,
    wsm,
    device: torch.device,
    config: TrainConfig,
    *,
    split_wrapper: SplitWeightWrapper,
    scaler: LocalMseUnitScaler,
    mask_generator: torch.Generator | None = None,
    init_state: torch.Tensor | None = None,
    abort_on_nan: bool = False,
) -> SegmentResult:
    """One TBPTT segment: z-score scaling on LoRA grads, clip, then wrapper step."""
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

    loss = F.mse_loss(pred, gt_res) * config.lambda_mse + fr_loss(h) * config.lambda_fr
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

    g_local_mse = compute_g_local_mse(
        pred.detach(),
        gt_res.detach(),
        config.lambda_mse,
        rae.readout_layer.weight.detach(),
    )
    unit_scales = scaler.unit_scales_from_g_local(g_local_mse)
    split_wrapper.scale_btsp_grads(unit_scales)

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

    final_state = h[:, -1, :].detach().clone() if config.carry_state else None
    return SegmentResult(loss=float(loss.detach().item()), final_state=final_state)


def train_trajectory_segments_alt(
    rae: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tc_full: np.ndarray,
    wsm,
    device: torch.device,
    config: TrainConfig,
    *,
    split_wrapper: SplitWeightWrapper,
    scaler: LocalMseUnitScaler,
    mask_generator: torch.Generator | None = None,
    on_segment: Callable[[int, int, SegmentResult], None] | None = None,
    segment_progress: tqdm | None = None,
    abort_on_nan: bool = False,
) -> bool:
    """Train all TBPTT segments with alt (split + g_local) scaling on W_btsp."""
    n_segments = tc_full.shape[1] // config.step_size
    state: torch.Tensor | None = None
    for seg_idx in range(n_segments):
        tc = tc_full[:, seg_idx * config.step_size : (seg_idx + 1) * config.step_size]
        result = train_one_segment_alt(
            rae,
            optimizer,
            tc,
            wsm,
            device,
            config,
            split_wrapper=split_wrapper,
            scaler=scaler,
            mask_generator=mask_generator,
            init_state=state if config.carry_state else None,
            abort_on_nan=abort_on_nan,
        )
        if abort_on_nan and (segment_has_nonfinite(result) or model_has_nonfinite(rae)):
            return False
        if on_segment is not None:
            on_segment(seg_idx, n_segments, result)
        if segment_progress is not None:
            segment_progress.update(1)
        if config.carry_state:
            state = result.final_state
    return True


def train_one_segment_on_opt(
    rae: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tc: np.ndarray,
    wsm,
    device: torch.device,
    config: TrainConfig,
    *,
    scaler: LocalMseUnitScaler,
    kappa: float,
    mask_generator: torch.Generator | None = None,
    init_state: torch.Tensor | None = None,
    capture_gradients: bool = False,
    extractor: OptimizerSignalExtractor | None = None,
    on_activity: Callable[[np.ndarray, torch.Tensor], None] | None = None,
    abort_on_nan: bool = False,
) -> SegmentResult:
    """One TBPTT segment: scale all grads by clipped kappa * inverse z-score, then step."""
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

    loss = F.mse_loss(pred, gt_res) * config.lambda_mse + fr_loss(h) * config.lambda_fr
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

    g_local_mse = compute_g_local_mse(
        pred.detach(),
        gt_res.detach(),
        config.lambda_mse,
        rae.readout_layer.weight.detach(),
    )
    scale_factors = scaler.clipped_on_opt_scale_factors(g_local_mse, kappa)
    scale_all_parameter_grads(rae, scale_factors)

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
        on_opt_scale_factors=scale_factors.detach().cpu().numpy().astype(np.float32),
    )


def train_trajectory_segments_on_opt(
    rae: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tc_full: np.ndarray,
    wsm,
    device: torch.device,
    config: TrainConfig,
    *,
    scaler: LocalMseUnitScaler,
    kappa: float,
    mask_generator: torch.Generator | None = None,
    capture_gradients: bool = False,
    extractor: OptimizerSignalExtractor | None = None,
    on_segment: Callable[[int, int, SegmentResult], None] | None = None,
    on_activity: Callable[[np.ndarray, torch.Tensor], None] | None = None,
    segment_progress: tqdm | None = None,
    abort_on_nan: bool = False,
) -> bool:
    """Train all TBPTT segments with on-optimizer grad scaling from g_local_mse."""
    n_segments = tc_full.shape[1] // config.step_size
    state: torch.Tensor | None = None
    for seg_idx in range(n_segments):
        tc = tc_full[:, seg_idx * config.step_size : (seg_idx + 1) * config.step_size]
        result = train_one_segment_on_opt(
            rae,
            optimizer,
            tc,
            wsm,
            device,
            config,
            scaler=scaler,
            kappa=kappa,
            mask_generator=mask_generator,
            init_state=state if config.carry_state else None,
            capture_gradients=capture_gradients,
            extractor=extractor,
            on_activity=on_activity,
            abort_on_nan=abort_on_nan,
        )
        if abort_on_nan and (segment_has_nonfinite(result) or model_has_nonfinite(rae)):
            return False
        if on_segment is not None:
            on_segment(seg_idx, n_segments, result)
        if segment_progress is not None:
            segment_progress.update(1)
        if config.carry_state:
            state = result.final_state
    return True
