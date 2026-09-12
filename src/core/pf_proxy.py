"""Cheap online place-field statistics, for screening training variants.

The full analysis path (held-out eval ratemaps -> 2D Gaussian fits -> PF tracking)
is far too expensive to run inside a multi-seed sweep. This module estimates the
same *shape* of metrics -- acquisition time, spontaneity, drift, revives -- from
the hidden activity that training already produces, at essentially zero extra
cost: an exponentially-decaying, occupancy-normalized coarse rate map is
accumulated from each TBPTT segment and reduced to five numbers per unit at
regular probes.

Numbers from here are *not* comparable to the Gaussian-fit pipeline (coarser
bins, coarser time resolution, one field per unit instead of two). They are only
meant to rank training variants against each other before committing to a full
run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

SNAPSHOT_FIELDS = ("peak", "com_x", "com_y", "mean_rate", "field_frac")


@dataclass
class PfProxyConfig:
    bin_size_cm: float = 8.0
    decay: float = 0.85
    """Per-segment decay of the rate-map accumulator; memory is ~1/(1-decay) segments."""
    probe_every_segments: int = 5
    min_visits: float = 1.5
    """Bins with less accumulated occupancy than this are treated as unvisited.

    One segment supplies ~1 sample per bin at these defaults, so the decayed
    counts settle around 10 per bin and most of the arena stays covered. Finer
    bins or a faster decay drop coverage far enough that fields drop out of the
    map spuriously, which inflates the revive and drift estimates.
    """
    dead_threshold_frac: float = 0.4
    """Dead if peak rate <= frac * 99.5th percentile peak (mirrors the full pipeline)."""
    field_level: float = 0.5
    """Bins above this fraction of a unit's peak are counted as inside its field."""
    max_field_frac: float = 0.35
    """A unit whose field covers more than this fraction of visited bins is not a PF."""
    peak_factor: float = 0.95
    """Acquisition time is the first probe reaching this fraction of the period peak."""


class PfProxyTracker:
    """Accumulate coarse rate maps from training activity and probe them periodically."""

    def __init__(
        self,
        n_hidden: int,
        arena_map: np.ndarray,
        *,
        config: PfProxyConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config or PfProxyConfig()
        self.device = torch.device(device)
        self.n_hidden = int(n_hidden)
        n_x, n_y = arena_map.shape
        self.n_bins_x = max(1, int(np.ceil(n_x / self.config.bin_size_cm)))
        self.n_bins_y = max(1, int(np.ceil(n_y / self.config.bin_size_cm)))
        self.n_bins = self.n_bins_x * self.n_bins_y
        self._sums = torch.zeros((self.n_hidden, self.n_bins), device=self.device)
        self._counts = torch.zeros(self.n_bins, device=self.device)
        self._n_segments = 0
        self._snapshots: list[np.ndarray] = []
        self._coverage: list[float] = []

    @torch.no_grad()
    def update(self, coords: np.ndarray, states: torch.Tensor) -> None:
        """Accumulate one segment. `coords` is `(1, T, 2)`, `states` is `(1, T, H)`."""
        cfg = self.config
        self._sums.mul_(cfg.decay)
        self._counts.mul_(cfg.decay)

        pos = torch.as_tensor(np.asarray(coords), dtype=torch.float32, device=self.device)
        pos = pos.reshape(-1, 2)
        bin_x = (pos[:, 0] / cfg.bin_size_cm).long().clamp_(0, self.n_bins_x - 1)
        bin_y = (pos[:, 1] / cfg.bin_size_cm).long().clamp_(0, self.n_bins_y - 1)
        flat = bin_x * self.n_bins_y + bin_y

        activity = states.detach().reshape(-1, self.n_hidden).to(self.device)
        self._sums.index_add_(1, flat, activity.T)
        self._counts.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))

        self._n_segments += 1
        if self._n_segments % cfg.probe_every_segments == 0:
            self._probe()

    @torch.no_grad()
    def _probe(self) -> None:
        cfg = self.config
        visited = self._counts >= cfg.min_visits
        if not bool(visited.any()):
            return
        rate = self._sums[:, visited] / self._counts[visited].clamp_min(1e-6)

        centers_x = (
            (torch.arange(self.n_bins_x, device=self.device) + 0.5) * cfg.bin_size_cm
        ).repeat_interleave(self.n_bins_y)[visited]
        centers_y = (
            (torch.arange(self.n_bins_y, device=self.device) + 0.5) * cfg.bin_size_cm
        ).repeat(self.n_bins_x)[visited]

        peak = rate.amax(dim=1)
        in_field = (rate >= cfg.field_level * peak.unsqueeze(1)) & (peak.unsqueeze(1) > 0)
        weights = rate * in_field
        mass = weights.sum(dim=1).clamp_min(1e-8)
        snapshot = np.stack(
            [
                peak.cpu().numpy(),
                ((weights * centers_x).sum(dim=1) / mass).cpu().numpy(),
                ((weights * centers_y).sum(dim=1) / mass).cpu().numpy(),
                rate.mean(dim=1).cpu().numpy(),
                (in_field.sum(dim=1).float() / int(visited.sum())).cpu().numpy(),
            ],
            axis=1,
        ).astype(np.float32)
        self._snapshots.append(snapshot)
        self._coverage.append(float(visited.float().mean()))

    @property
    def snapshots(self) -> np.ndarray:
        """`(n_probes, n_hidden, 5)` array, columns as in `SNAPSHOT_FIELDS`."""
        if not self._snapshots:
            return np.zeros((0, self.n_hidden, len(SNAPSHOT_FIELDS)), dtype=np.float32)
        return np.stack(self._snapshots, axis=0)

    @property
    def mean_coverage(self) -> float:
        return float(np.mean(self._coverage)) if self._coverage else float("nan")


def summarize_pf_proxy(
    snapshots: np.ndarray,
    *,
    config: PfProxyConfig | None = None,
    min_period_probes: tuple[int, ...] = (1, 2, 3),
) -> dict[str, float]:
    """Reduce probe snapshots to acquisition/spontaneity/drift/revive statistics.

    `min_period_probes` selects the survival thresholds at which spontaneity is
    also reported. Reporting spontaneity unconditionally is misleading: a field
    that lives for a single probe is "spontaneous" by definition, so a variant
    that merely makes fields flicker faster scores well on the raw number.
    """
    cfg = config or PfProxyConfig()
    out: dict[str, float] = {}
    if snapshots.size == 0:
        return out

    peak = snapshots[..., 0]
    com_x = snapshots[..., 1]
    com_y = snapshots[..., 2]
    field_frac = snapshots[..., 4]

    dead_threshold = cfg.dead_threshold_frac * float(np.nanpercentile(peak, 99.5))
    alive = (peak > dead_threshold) & (field_frac < cfg.max_field_frac)
    # Instantaneous fraction of the population carrying a field, as opposed to the
    # cumulative `n_units_with_pf` below. Only this one is comparable to the ~30-50%
    # of CA1 cells that show place-field activity at a given time.
    out["alive_frac_mean"] = float(alive.mean())
    out["alive_frac_final"] = float(alive[-1].mean())

    rat: list[float] = []
    spontaneous: list[bool] = []
    lengths: list[int] = []
    drifts: list[float] = []
    revives: list[int] = []

    n_probes, n_units = alive.shape
    for unit in range(n_units):
        flags = alive[:, unit]
        n_periods = 0
        start = None
        for t in range(n_probes + 1):
            active = bool(flags[t]) if t < n_probes else False
            if active and start is None:
                start = t
            elif not active and start is not None:
                idx = np.arange(start, t)
                amps = peak[idx, unit]
                threshold = cfg.peak_factor * float(amps.max())
                reached = np.flatnonzero(amps >= threshold)
                peak_idx = int(reached[0]) if reached.size else int(np.argmax(amps))
                rat.append(float(peak_idx))
                spontaneous.append(peak_idx == 0)
                lengths.append(int(idx.size))
                drifts.append(
                    (float(np.std(com_x[idx, unit])) + float(np.std(com_y[idx, unit]))) / 2.0
                    if idx.size > 1
                    else 0.0
                )
                n_periods += 1
                start = None
        if n_periods:
            revives.append(n_periods - 1)

    if not rat:
        return out

    rat_arr = np.asarray(rat)
    spont_arr = np.asarray(spontaneous)
    len_arr = np.asarray(lengths)
    out["n_life_periods"] = float(rat_arr.size)
    out["n_units_with_pf"] = float(len(revives))
    out["rat_probes_mean"] = float(rat_arr.mean())
    out["rat_probes_median"] = float(np.median(rat_arr))
    out["spontaneous_frac"] = float(spont_arr.mean())
    out["period_len_probes_mean"] = float(len_arr.mean())
    out["period_len_probes_median"] = float(np.median(len_arr))
    out["drift_mean_cm"] = float(np.mean(drifts))
    out["revives_per_unit_mean"] = float(np.mean(revives))
    for min_len in min_period_probes:
        keep = len_arr > min_len
        out[f"spont_len_gt{min_len}"] = (
            float(spont_arr[keep].mean()) if keep.any() else float("nan")
        )
        out[f"frac_periods_len_gt{min_len}"] = float(keep.mean())
    return out
