"""Plateau training: discrete one-shot field allocation plus per-unit consolidation.

Motivation
----------
Gradient descent on the RAE objective grows place fields *continuously*: a unit's
field ramps up over many TBPTT segments (nonzero Representation Acquisition Time),
and because the firing-rate penalty keeps every unit in a permanent tug-of-war with
the reconstruction term, fields also drift and die/revive repeatedly.

Biology does not work that way. In CA1, a single dendritic plateau potential
(behavioral timescale synaptic plasticity, Bittner et al. 2017) installs a complete
place field in one traversal, and the field is then comparatively stable. This
module factorises learning the same way:

1. **Allocation (discrete, one-shot).** Every segment, a few plateau events may
   fire. A plateau picks a moment `t_p` where reconstruction is poor, forms a
   behavioural-timescale eligibility trace of the presynaptic input around `t_p`,
   and *writes* that trace into the input weights of an unused hidden unit,
   together with a threshold that makes the unit fire only near that location.
   The readout column for the unit is seeded with the exact coordinate
   least-squares solution against the current residual, so a plateau is a greedy
   error-reducing step (matching pursuit) rather than a perturbation.

2. **Consolidation (slow, continuous).** Units that carry a useful field get their
   *incoming* weight gradients progressively damped (metaplasticity), while the
   readout stays fully plastic. Established fields therefore stop drifting and
   stop being recycled, but the network can still fit the task.

Unlike the `on_opt` variant in `core.alt_training`, this is not a rescaling of the
gradient: plateaus are discrete writes applied outside the optimizer, and the
optimizer's own state for the rewritten rows is reset so it does not fight them.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from core.gradient_signals import capture_step_signals
from core.training import (
    SegmentResult,
    TrainConfig,
    _capture_weight_snapshots,
    _is_numerical_optimizer_failure,
    forward_has_nonfinite,
    fr_loss,
    model_has_nonfinite,
    segment_has_nonfinite,
    unif_loss,
    weight_l2_reg_loss,
)
from core.utils import apply_input_mask
from models.custom_learnable_activation import custom_ab_reg_loss
from optimizers.base import OptimizerSignalExtractor

EPS = 1e-8


@dataclass
class PlateauConfig:
    """Hyperparameters for plateau allocation and consolidation.

    The two mechanisms are independently switchable so the contribution of each
    can be ablated.
    """

    # --- what is enabled ---
    allocate: bool = True
    consolidate: bool = True

    # --- allocation ---
    plateau_rate: float = 1.0
    """Expected number of plateau events per TBPTT segment (Poisson-like)."""
    max_plateaus_per_segment: int = 3
    start_segment: int = 2
    """Skip allocation for this many segments so usefulness statistics settle."""
    refractory_segments: int = 20
    """A unit cannot receive a second plateau within this many segments."""
    eligibility_tau_pre_s: float = 1.5
    """Decay of the eligibility kernel for input *preceding* the plateau."""
    eligibility_tau_post_s: float = 0.75
    """Decay of the eligibility kernel for input *following* the plateau."""
    target_sparsity: float = 0.10
    """Fraction of recent positions at which the new field should be active."""
    peak_scale: float = 1.0
    """New field peak rate, relative to the median peak of currently active units."""
    readout_seed_gain: float = 1.0
    """Fraction of the exact coordinate least-squares readout column to install."""
    readout_seed_max_norm_ratio: float = 1.0
    """Trust region on the seeded readout column, in units of the median column norm.

    The exact coordinate least-squares solution minimizes *this* segment's error but
    is estimated from only the handful of timesteps where the new field is active, so
    its variance is enormous (columns tens of times the population norm were observed).
    Shrinking it toward zero keeps the error-reduction guarantee -- the objective is a
    parabola in the column scale, minimized at 1 and below its value at 0 for any
    scale in (0, 2) -- while keeping the write in distribution.

    Zero installs no readout column at all, leaving the plateau as pure feature
    allocation: the field still forms in one shot (it is set by the input weights),
    and the gradient discovers how to read it out.
    """
    zero_outgoing_recurrent: bool = True
    """Also clear the recruited unit's outgoing recurrent column.

    Its incoming row is zeroed so the unit's own response is exactly a leaky filter.
    Leaving the outgoing column would inject a newly loud, spatially coherent signal
    into every other unit through weights that were tuned while the unit was silent.
    """
    candidate_quantile: float = 0.25
    """Recruit only from the least useful this fraction of units."""
    residual_temperature: float = 1.0
    """Softmax temperature (relative to residual std) for sampling plateau times."""
    template_buffer_segments: int = 8
    template_stride: int = 4
    """Subsampling of buffered inputs used to calibrate the field threshold."""
    reset_optimizer_state: bool = True

    # --- consolidation ---
    usefulness_ema: float = 0.9
    usefulness_mode: str = "readout"
    """How a unit's usefulness is scored: 'readout' or 'activity'.

    'readout' uses peak rate x readout-column norm, a first-order estimate of how
    much the reconstruction would degrade if the unit were silenced. That couples
    consolidation to the readout seed: a plateau unit whose column was seeded small
    reads as useless, is never consolidated, and has its fresh field eroded by the
    gradient. 'activity' scores by peak rate alone, so a unit is protected for
    carrying a field regardless of how well the readout already exploits it.
    """
    commit_quantile: float = 0.5
    """Units above this usefulness quantile accrue commitment; the rest shed it."""
    commit_rate: float = 0.15
    commit_decay: float = 0.05
    max_damping: float = 0.9
    """Maximum fraction of the incoming-weight gradient removed for a committed unit."""
    grace_segments: int = 3
    """Incoming weights of a just-plateaued unit are frozen for this many segments."""

    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_damping <= 1.0:
            raise ValueError(f"max_damping must be in [0, 1], got {self.max_damping}")
        if not 0.0 < self.target_sparsity < 1.0:
            raise ValueError(
                f"target_sparsity must be in (0, 1), got {self.target_sparsity}"
            )
        if not 0.0 < self.candidate_quantile <= 1.0:
            raise ValueError(
                f"candidate_quantile must be in (0, 1], got {self.candidate_quantile}"
            )
        if self.usefulness_mode not in ("readout", "activity"):
            raise ValueError(
                f"usefulness_mode must be 'readout' or 'activity', "
                f"got {self.usefulness_mode!r}"
            )


@dataclass
class PlateauLayers:
    """The parameter tensors a plateau writes to, resolved once per run."""

    w_in: torch.nn.Parameter
    b_in: torch.nn.Parameter
    w_rec: torch.nn.Parameter
    b_rec: torch.nn.Parameter
    w_out: torch.nn.Parameter
    alpha: torch.Tensor


def resolve_plateau_layers(rae: torch.nn.Module) -> PlateauLayers:
    """Locate input/recurrent/readout weights of the single-layer leaky RAE."""
    recurrent = rae.recurrent_layers[0]
    leaky = recurrent.leaky_layer
    return PlateauLayers(
        w_in=recurrent.projection_layer.weight,
        b_in=recurrent.projection_layer.bias,
        w_rec=leaky.linear_layer.weight,
        b_rec=leaky.linear_layer.bias,
        w_out=rae.readout_layer.weight,
        alpha=leaky.alpha.detach(),
    )


def eligibility_trace(
    inputs: torch.Tensor,
    t_p: int,
    *,
    dt: float,
    tau_pre_s: float,
    tau_post_s: float,
) -> torch.Tensor:
    """Behavioural-timescale weighted average of presynaptic input around `t_p`.

    `inputs` is `(T, n_in)`. The kernel is asymmetric: input arriving before the
    plateau is bound over a longer window than input arriving after it, matching
    the measured BTSP plasticity kernel.
    """
    n_t = inputs.shape[0]
    offsets = (torch.arange(n_t, device=inputs.device, dtype=inputs.dtype) - t_p) * dt
    weights = torch.where(
        offsets < 0,
        torch.exp(offsets / tau_pre_s),
        torch.exp(-offsets / tau_post_s),
    )
    weights = weights / weights.sum().clamp_min(EPS)
    return weights @ inputs


def leaky_relu_response(drive: torch.Tensor, alpha: float) -> torch.Tensor:
    """Response of one isolated leaky-integrator ReLU unit to a scalar drive.

    Mirrors `LeakyLinearLayer.forward` with the recurrent input removed and
    `v_0 = 0`, which is exact for a unit whose recurrent row has been zeroed.
    """
    response = torch.empty_like(drive)
    v = torch.zeros((), device=drive.device, dtype=drive.dtype)
    for t in range(drive.shape[0]):
        v = (1.0 - alpha) * v + alpha * drive[t]
        response[t] = torch.relu(v)
    return response


def reset_optimizer_slice(
    optimizer: torch.optim.Optimizer,
    param: torch.nn.Parameter,
    index: int,
    *,
    dim: int = 0,
) -> None:
    """Zero one row/column of the optimizer moments for `param`.

    A plateau writes weights discretely; stale first/second moments would
    otherwise pull the new field straight back out. Only state entries shaped
    like the parameter are touched, so structured state (e.g. Shampoo
    preconditioners) is left alone.
    """
    state = optimizer.state.get(param)
    if not state:
        return
    for value in state.values():
        if isinstance(value, torch.Tensor) and value.shape == param.shape:
            value.select(dim, index).zero_()


class PlateauState:
    """Per-unit usefulness/commitment bookkeeping and the plateau mechanism itself."""

    def __init__(
        self,
        n_hidden: int,
        *,
        config: PlateauConfig,
        device: torch.device | str = "cpu",
        dt: float = 0.05,
    ) -> None:
        self.config = config
        self.n_hidden = int(n_hidden)
        self.dt = float(dt)
        self.device = torch.device(device)
        self.n_segments = 0
        self.n_plateaus = 0
        self.usefulness = torch.zeros(self.n_hidden, device=self.device)
        self.commitment = torch.zeros(self.n_hidden, device=self.device)
        self.last_plateau = torch.full(
            (self.n_hidden,), -(10**9), dtype=torch.long, device=self.device
        )
        self._templates: deque[torch.Tensor] = deque(
            maxlen=max(1, config.template_buffer_segments)
        )
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(config.seed)
        self.last_events: list[int] = []

    # --- consolidation -----------------------------------------------------

    def damping(self) -> torch.Tensor:
        """Per-unit multiplier applied to incoming-weight gradients."""
        if not self.config.consolidate:
            return torch.ones(self.n_hidden, device=self.device)
        damp = 1.0 - self.config.max_damping * self.commitment
        age = self.n_segments - self.last_plateau
        damp = torch.where(age < self.config.grace_segments, torch.zeros_like(damp), damp)
        return damp

    def apply_consolidation(self, layers: PlateauLayers) -> torch.Tensor:
        """Damp incoming (input + recurrent) weight gradients; leave the readout free."""
        damp = self.damping()
        if layers.w_in.grad is not None:
            layers.w_in.grad.mul_(damp.unsqueeze(1))
        if layers.b_in.grad is not None:
            layers.b_in.grad.mul_(damp)
        if layers.w_rec.grad is not None:
            layers.w_rec.grad.mul_(damp.unsqueeze(1))
        if layers.b_rec.grad is not None:
            layers.b_rec.grad.mul_(damp)
        return damp

    def update_usefulness(self, h: torch.Tensor, w_out: torch.Tensor) -> None:
        """Track how much each unit contributes to the reconstruction."""
        with torch.no_grad():
            contribution = h[0].amax(dim=0)
            if self.config.usefulness_mode == "readout":
                contribution = contribution * w_out.detach().norm(dim=0)
            ema = self.config.usefulness_ema
            self.usefulness.mul_(ema).add_(contribution, alpha=1.0 - ema)

    def update_commitment(self) -> None:
        with torch.no_grad():
            threshold = torch.quantile(self.usefulness, self.config.commit_quantile)
            useful = self.usefulness > threshold
            delta = torch.where(
                useful,
                torch.full_like(self.commitment, self.config.commit_rate),
                torch.full_like(self.commitment, -self.config.commit_decay),
            )
            self.commitment.add_(delta).clamp_(0.0, 1.0)

    # --- allocation --------------------------------------------------------

    def push_template(self, masked_input: torch.Tensor) -> None:
        """Buffer recent presynaptic input used to calibrate field thresholds."""
        self._templates.append(
            masked_input[0, :: self.config.template_stride].detach().clone()
        )

    def _template_pool(self, fallback: torch.Tensor) -> torch.Tensor:
        if not self._templates:
            return fallback
        return torch.cat(list(self._templates), dim=0)

    def _sample_plateau_times(self, residual_energy: torch.Tensor, n: int) -> list[int]:
        """Sample moments in proportion to how badly they are reconstructed."""
        scale = residual_energy.std().clamp_min(EPS) * self.config.residual_temperature
        logits = (residual_energy - residual_energy.mean()) / scale
        probs = torch.softmax(logits, dim=0).cpu()
        n = min(n, int((probs > 0).sum().item()))
        if n <= 0:
            return []
        idx = torch.multinomial(probs, n, replacement=False, generator=self._rng)
        return [int(i) for i in idx]

    def _candidate_units(self, n_wanted: int) -> list[int]:
        """Randomly draw recruitable units from the least useful tail.

        Plateaus in vivo occur in a small random subset of silent cells, so the
        draw is uniform within the tail rather than strictly least-useful-first;
        that also avoids a deterministic unit ordering while usefulness is flat.
        """
        age = self.n_segments - self.last_plateau
        eligible = age >= self.config.refractory_segments
        cutoff = torch.quantile(self.usefulness, min(1.0, self.config.candidate_quantile))
        pool = torch.nonzero(eligible & (self.usefulness <= cutoff), as_tuple=False)
        pool = pool.flatten()
        if pool.numel() == 0:
            return []
        n = min(n_wanted, int(pool.numel()))
        picks = torch.randperm(pool.numel(), generator=self._rng)[:n]
        return [int(pool[i]) for i in picks]

    def _n_plateaus_this_segment(self) -> int:
        rate = self.config.plateau_rate
        if rate <= 0.0:
            return 0
        whole = int(math.floor(rate))
        frac = rate - whole
        extra = int(torch.rand((), generator=self._rng).item() < frac)
        return min(whole + extra, self.config.max_plateaus_per_segment)

    @torch.no_grad()
    def maybe_plateau(
        self,
        layers: PlateauLayers,
        optimizer: torch.optim.Optimizer,
        *,
        masked_input: torch.Tensor,
        residual: torch.Tensor,
        h: torch.Tensor,
    ) -> list[int]:
        """Fire zero or more plateau events; return the units that were rewritten.

        `masked_input` is `(1, T, n_in)` (what the network actually received),
        `residual` is `(1, T, n_out)` = target minus prediction, and `h` is the
        segment's hidden activity `(1, T, H)`.
        """
        cfg = self.config
        if not cfg.allocate or self.n_segments < cfg.start_segment:
            return []
        n_events = self._n_plateaus_this_segment()
        if n_events <= 0:
            return []

        x_seg = masked_input[0]
        # Cloned because each plateau subtracts what it explains, so several
        # plateaus in one segment act as successive matching-pursuit steps
        # instead of all fitting the same residual and jointly overshooting.
        residual_seg = residual[0].clone()
        residual_energy = residual_seg.pow(2).sum(dim=1)
        pool = self._template_pool(x_seg)
        median_readout_norm = float(layers.w_out.norm(dim=0).median())

        peaks = h[0].amax(dim=0)
        active_peaks = peaks[peaks > 0]
        target_peak = cfg.peak_scale * (
            float(active_peaks.median()) if active_peaks.numel() else 1.0
        )
        if not math.isfinite(target_peak) or target_peak <= 0.0:
            return []

        candidates = self._candidate_units(n_events)
        if not candidates:
            return []

        events: list[int] = []
        for t_p in self._sample_plateau_times(residual_energy, len(candidates)):
            trace = eligibility_trace(
                x_seg,
                t_p,
                dt=self.dt,
                tau_pre_s=cfg.eligibility_tau_pre_s,
                tau_post_s=cfg.eligibility_tau_post_s,
            )
            norm = trace.norm()
            if not torch.isfinite(norm) or norm <= EPS:
                continue
            direction = trace / norm

            # Threshold so the unit is driven above zero at ~target_sparsity of
            # recently visited positions: a localized field, not a global mode.
            similarity_pool = pool @ direction
            theta = torch.quantile(similarity_pool, 1.0 - cfg.target_sparsity)

            unit = candidates[len(events)]
            alpha = float(layers.alpha[unit])
            unit_response = leaky_relu_response(x_seg @ direction - theta, alpha)
            peak = float(unit_response.max())
            if peak <= EPS:
                continue
            gain = target_peak / peak
            activity = unit_response * gain

            layers.w_in[unit] = direction * gain
            layers.b_in[unit] = -theta * gain
            layers.w_rec[unit].zero_()
            layers.b_rec[unit].zero_()
            if cfg.zero_outgoing_recurrent:
                layers.w_rec[:, unit].zero_()

            # Coordinate least-squares for the new feature's readout column,
            # shrunk into a trust region so the plateau still reduces this
            # segment's error without installing an out-of-distribution column.
            denom = activity.dot(activity).clamp_min(EPS)
            column = cfg.readout_seed_gain * (
                residual_seg.transpose(0, 1) @ activity
            ) / denom
            max_norm = cfg.readout_seed_max_norm_ratio * median_readout_norm
            column_norm = float(column.norm())
            if max_norm <= 0.0:
                column = torch.zeros_like(column)
            elif column_norm > max_norm:
                column = column * (max_norm / column_norm)
            layers.w_out[:, unit] = column
            residual_seg -= torch.outer(activity, column)

            if cfg.reset_optimizer_state:
                reset_optimizer_slice(optimizer, layers.w_in, unit)
                reset_optimizer_slice(optimizer, layers.b_in, unit)
                reset_optimizer_slice(optimizer, layers.w_rec, unit)
                reset_optimizer_slice(optimizer, layers.b_rec, unit)
                reset_optimizer_slice(optimizer, layers.w_out, unit, dim=1)
                if cfg.zero_outgoing_recurrent:
                    reset_optimizer_slice(optimizer, layers.w_rec, unit, dim=1)

            self.last_plateau[unit] = self.n_segments
            self.commitment[unit] = 1.0
            self.usefulness[unit] = float(self.usefulness.median())
            events.append(unit)

        self.n_plateaus += len(events)
        return events

    def end_segment(self) -> None:
        self.n_segments += 1


def train_one_segment_plateau(
    rae: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tc: np.ndarray,
    wsm,
    device: torch.device,
    config: TrainConfig,
    *,
    state: PlateauState,
    layers: PlateauLayers,
    mask_generator: torch.Generator | None = None,
    init_state: torch.Tensor | None = None,
    capture_gradients: bool = False,
    extractor: OptimizerSignalExtractor | None = None,
    on_activity: Callable[[np.ndarray, torch.Tensor], None] | None = None,
    abort_on_nan: bool = False,
) -> SegmentResult:
    """One TBPTT segment: consolidated gradient step, then plateau allocation."""
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

    state.update_usefulness(h, layers.w_out)
    damping = state.apply_consolidation(layers)

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

    # The plateau is driven by this segment's residual. Reusing the pre-step
    # forward pass keeps the cost of plateau training identical to normal
    # training; one optimizer step changes the residual negligibly, and the
    # recruited unit was silent either way.
    events = state.maybe_plateau(
        layers,
        optimizer,
        masked_input=masked_res.detach(),
        residual=(gt_res - pred).detach(),
        h=h.detach(),
    )
    state.push_template(masked_res)
    state.update_commitment()
    state.end_segment()

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
        plateau_units=np.asarray(events, dtype=np.int32),
        plateau_damping=damping.detach().cpu().numpy().astype(np.float32),
    )


def train_trajectory_segments_plateau(
    rae: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tc_full: np.ndarray,
    wsm,
    device: torch.device,
    config: TrainConfig,
    *,
    state: PlateauState,
    layers: PlateauLayers,
    mask_generator: torch.Generator | None = None,
    capture_gradients: bool = False,
    extractor: OptimizerSignalExtractor | None = None,
    on_segment: Callable[[int, int, SegmentResult], None] | None = None,
    on_activity: Callable[[np.ndarray, torch.Tensor], None] | None = None,
    segment_progress: tqdm | None = None,
    abort_on_nan: bool = False,
) -> bool:
    """Train all TBPTT segments of one trajectory with plateau training."""
    n_segments = tc_full.shape[1] // config.step_size
    carried: torch.Tensor | None = None
    for seg_idx in range(n_segments):
        tc = tc_full[:, seg_idx * config.step_size : (seg_idx + 1) * config.step_size]
        result = train_one_segment_plateau(
            rae,
            optimizer,
            tc,
            wsm,
            device,
            config,
            state=state,
            layers=layers,
            mask_generator=mask_generator,
            init_state=carried if config.carry_state else None,
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
            carried = result.final_state
    return True
