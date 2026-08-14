"""Load per-segment training signals saved under `results/.../signals/`.

Each `segment_*.npz` stores scalar `loss`, hidden-state gradient arrays
(`g_bptt_mse`, `g_local_mse`, `g_fr`), per-unit `opt_signals`, and
post-step full model weights as top-level `weight__*` arrays. Shampoo runs
also store inverse Kronecker factors as top-level `h_inv__*` arrays.
"""
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

GRADIENT_SIGNAL_KEYS = ("g_bptt_mse", "g_local_mse", "g_fr")

_HIDDEN_UNIT_NODE_ID_RE = re.compile(r"^(recurrent|input)\[(\d+)\]$")


def _parse_traj_dir_name(name: str) -> tuple[int, int] | None:
    m = re.fullmatch(r"epoch(\d+)_traj(\d+)", name)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


def _segment_idx_from_path(path: Path) -> int:
    m = re.fullmatch(r"segment_(\d+)", path.stem)
    if m is None:
        raise ValueError(f"Unexpected segment file name: {path.name!r}")
    return int(m.group(1))


def _iter_training_segment_files(
    signals_dir: Path,
) -> Iterator[tuple[int, int, int, Path]]:
    """Yield `(epoch_id, traj_id, segment_id, segment_path)` in training order."""
    traj_dirs: list[tuple[tuple[int, int], Path]] = []
    for path in signals_dir.iterdir():
        if not path.is_dir():
            continue
        parsed = _parse_traj_dir_name(path.name)
        if parsed is not None:
            traj_dirs.append((parsed, path))

    if not traj_dirs:
        raise FileNotFoundError(f"No epoch*_traj* directories under {signals_dir}")

    traj_dirs.sort(key=lambda item: item[0])
    for (epoch_id, traj_id), traj_dir in traj_dirs:
        seg_paths = sorted(
            traj_dir.glob("segment_*.npz"),
            key=_segment_idx_from_path,
        )
        for seg_idx, seg_path in enumerate(seg_paths):
            expected_seg = _segment_idx_from_path(seg_path)
            if expected_seg != seg_idx:
                raise ValueError(
                    f"Non-contiguous segment index in {seg_path} (expected {seg_idx})"
                )
            yield epoch_id, traj_id, seg_idx, seg_path


def _validate_g_local_mse_array(arr: np.ndarray) -> np.ndarray:
    """Return `g_local_mse` as `(T, H)` float64."""
    if arr.ndim != 3 or arr.shape[0] != 1:
        raise ValueError(f"Expected gradient shape (1, T, H), got {arr.shape}")
    return np.asarray(arr[0], dtype=np.float64)


def _mean_over_hidden_per_timepoint(arr: np.ndarray) -> np.ndarray:
    """Reduce `(1, T, H)` MSE gradient arrays to `(T,)` means over hidden units."""
    return _validate_g_local_mse_array(arr).mean(axis=1)


def _g_local_mse_stats_over_timesteps(values: np.ndarray) -> tuple[float, float, float]:
    """Return mean, max, and std of a per-timestep scalar series."""
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.max()), float(values.std())


def _network_excl_unit_mean_over_timesteps(g: np.ndarray) -> np.ndarray:
    """Mean over timesteps of the leave-one-out hidden mean, shape `(H,)`."""
    h = g.shape[1]
    if h <= 1:
        return np.full(h, np.nan, dtype=np.float64)
    sum_over_h = g.sum(axis=1, keepdims=True)
    excl_timeline = (sum_over_h - g) / (h - 1)
    return excl_timeline.mean(axis=0)


@dataclass
class EffectiveLrSegmentStatsBundle:
    """Per-segment effective LR for network-wide and per-hidden-unit views.

    `network` has one row per segment with `effective_lr_mean` (mean over
    hidden units). `unit_mean` and `network_excl_mean` have shape
    `(n_segments, n_hidden)` with one scalar per unit per segment (no
    within-segment temporal aggregation).
    """

    network: pd.DataFrame
    unit_mean: np.ndarray
    network_excl_mean: np.ndarray


@dataclass
class GLocalMseSegmentStatsBundle:
    """Per-segment `g_local_mse` stats for network-wide and per-hidden-unit views.

    `network` has one row per segment with columns `g_local_mse_{mean,max,std}`
    computed from the mean-over-hidden timeline. Unit arrays have shape
    `(n_segments, n_hidden)` with the same three stats taken on each unit's own
    timeline (signed or `|.|` depending on the field).

    `network_excl_signed_mean` and `network_excl_abs_mean` store, for each
    segment and hidden unit, the mean-over-timesteps of the network mean with that
    unit excluded (signed and `|.|` respectively).
    """

    network: pd.DataFrame
    unit_signed_mean: np.ndarray
    unit_signed_max: np.ndarray
    unit_signed_std: np.ndarray
    unit_abs_mean: np.ndarray
    unit_abs_max: np.ndarray
    unit_abs_std: np.ndarray
    network_excl_signed_mean: np.ndarray
    network_excl_abs_mean: np.ndarray


def _mean_over_hidden_units(arr: np.ndarray) -> float:
    """Reduce `(H,)` FR gradient vector to a scalar mean over hidden units."""
    return float(np.asarray(arr, dtype=np.float64).mean())


def _load_opt_signals_dict(raw: np.ndarray) -> dict:
    """Unwrap the object array written as `opt_signals` in segment NPZ files."""
    if raw.shape == ():
        return dict(raw.item())
    if raw.size != 1:
        raise ValueError(f"Expected scalar object array for opt_signals, got shape {raw.shape}")
    return dict(raw[0])


def _mean_per_weight_effective_lr(arr: np.ndarray) -> float:
    """Mean effective LR over incoming weights for one tracked unit."""
    values = np.asarray(arr, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.nanmean(values))


def _effective_lr_per_hidden_unit(
    effective_lr_by_node: dict[str, np.ndarray],
) -> np.ndarray:
    """Per-hidden-unit effective LR, shape `(H,)`.

    For each unit index, per-weight values are averaged within each tracked
    layer (`recurrent`, `input`), then combined across layers.
    """
    by_unit_index: dict[int, list[float]] = {}
    for node_id, eff in effective_lr_by_node.items():
        m = _HIDDEN_UNIT_NODE_ID_RE.fullmatch(node_id)
        if m is None:
            continue
        unit_idx = int(m.group(2))
        by_unit_index.setdefault(unit_idx, []).append(_mean_per_weight_effective_lr(eff))

    if not by_unit_index:
        return np.empty(0, dtype=np.float64)

    n_hidden = max(by_unit_index) + 1
    out = np.full(n_hidden, np.nan, dtype=np.float64)
    for unit_idx, layer_means in by_unit_index.items():
        out[unit_idx] = float(np.nanmean(layer_means))
    return out


def _network_excl_unit_mean_from_vector(values: np.ndarray) -> np.ndarray:
    """Leave-one-out network mean for one per-unit vector, shape `(H,)`."""
    values = np.asarray(values, dtype=np.float64)
    h = values.shape[0]
    if h <= 1:
        return np.full(h, np.nan, dtype=np.float64)
    sum_over_h = np.nansum(values)
    return (sum_over_h - values) / (h - 1)


def _mean_effective_lr_over_neurons(effective_lr_by_node: dict[str, np.ndarray]) -> float:
    """Mean effective LR across hidden units at one segment.

    For each hidden unit index, per-weight effective LRs are averaged within
    each tracked layer (`recurrent`, `input`), then combined across layers
    for that unit. The segment scalar is the mean over units.
    """
    unit_vals = _effective_lr_per_hidden_unit(effective_lr_by_node)
    if unit_vals.size == 0:
        return float("nan")
    return float(np.nanmean(unit_vals))


def load_training_loss_timeline(signals_dir: Path | str) -> pd.DataFrame:
    """
    Load the full training loss per TBPTT segment (warmup excluded).

    Each `signals/<epoch_tag>_traj<N>/segment_<M>.npz` stores scalar `loss`
    (combined MSE + firing-rate regularization from training).

    Returns columns `global_segment_idx`, `epoch_id`, `traj_id`,
    `segment_id`, `loss`, `is_traj_start`.
    """
    signals_dir = Path(signals_dir)
    if not signals_dir.is_dir():
        raise FileNotFoundError(f"Missing signals directory: {signals_dir}")

    rows: list[dict[str, int | float | bool]] = []
    global_idx = 0
    for epoch_id, traj_id, seg_idx, seg_path in _iter_training_segment_files(signals_dir):
        with np.load(seg_path) as data:
            if "loss" not in data:
                raise KeyError(f"{seg_path} missing 'loss'")
            loss = float(np.asarray(data["loss"]).item())

        rows.append(
            {
                "global_segment_idx": global_idx,
                "epoch_id": epoch_id,
                "traj_id": traj_id,
                "segment_id": seg_idx,
                "loss": loss,
                "is_traj_start": seg_idx == 0,
            }
        )
        global_idx += 1

    return pd.DataFrame(rows)


def load_training_gradient_timeline(
    signals_dir: Path | str,
    *,
    segment_duration_s: float,
) -> pd.DataFrame:
    """
    Load mean gradient signals over training (warmup excluded).

    For `g_bptt_mse` and `g_local_mse` (shape `(1, T, H)`), the mean is
    taken over hidden units at each timestep, then over timesteps within the
    segment. `g_fr` (shape `(H,)`) is averaged over hidden units only.

    Returns one row per segment with `global_segment_idx`, `time_s` (segment
    midpoint), the three gradient means, and `is_traj_start`.
    """
    signals_dir = Path(signals_dir)
    if not signals_dir.is_dir():
        raise FileNotFoundError(f"Missing signals directory: {signals_dir}")

    rows: list[dict[str, int | float | bool]] = []
    global_idx = 0
    for epoch_id, traj_id, seg_idx, seg_path in _iter_training_segment_files(signals_dir):
        with np.load(seg_path) as data:
            for key in GRADIENT_SIGNAL_KEYS:
                if key not in data:
                    raise KeyError(f"{seg_path} missing {key!r}")

            bptt_t = _mean_over_hidden_per_timepoint(data["g_bptt_mse"])
            local_t = _mean_over_hidden_per_timepoint(data["g_local_mse"])
            fr_mean = _mean_over_hidden_units(data["g_fr"])

        rows.append(
            {
                "global_segment_idx": global_idx,
                "epoch_id": epoch_id,
                "traj_id": traj_id,
                "segment_id": seg_idx,
                "time_s": (global_idx + 0.5) * segment_duration_s,
                "g_bptt_mse": float(bptt_t.mean()),
                "g_local_mse": float(local_t.mean()),
                "g_fr": fr_mean,
                "is_traj_start": seg_idx == 0,
            }
        )
        global_idx += 1

    return pd.DataFrame(rows)


def load_effective_lr_segment_stats_bundle(
    signals_dir: Path | str,
    *,
    segment_duration_s: float | None = None,
) -> EffectiveLrSegmentStatsBundle:
    """
    Per-segment effective LR for each hidden unit (one scalar per segment).

    `effective_lr` is stored per tracked unit in `opt_signals` (AdaGrad and
    grafted Shampoo). For each hidden unit, per-weight values are averaged
    within `recurrent` and `input` layers, then combined across layers.
    """
    signals_dir = Path(signals_dir)
    if not signals_dir.is_dir():
        raise FileNotFoundError(f"Missing signals directory: {signals_dir}")

    rows: list[dict[str, int | float | bool]] = []
    unit_mean_rows: list[np.ndarray] = []
    network_excl_rows: list[np.ndarray] = []
    global_idx = 0
    for epoch_id, traj_id, seg_idx, seg_path in _iter_training_segment_files(signals_dir):
        with np.load(seg_path, allow_pickle=True) as data:
            if "opt_signals" not in data:
                raise KeyError(f"{seg_path} missing 'opt_signals'")
            opt_signals = _load_opt_signals_dict(data["opt_signals"])
            effective_lr_by_node = opt_signals.get("effective_lr")
            if not effective_lr_by_node:
                raise KeyError(
                    f"{seg_path} opt_signals missing 'effective_lr' "
                    "(optimizer may not log effective LR)"
                )

        unit_vals = _effective_lr_per_hidden_unit(effective_lr_by_node)
        rows.append(
            {
                "global_segment_idx": global_idx,
                "epoch_id": epoch_id,
                "traj_id": traj_id,
                "segment_id": seg_idx,
                "effective_lr_mean": float(np.nanmean(unit_vals)),
                "is_traj_start": seg_idx == 0,
            }
        )
        unit_mean_rows.append(unit_vals)
        network_excl_rows.append(_network_excl_unit_mean_from_vector(unit_vals))
        global_idx += 1

    if not rows:
        empty = pd.DataFrame(
            columns=[
                "global_segment_idx",
                "epoch_id",
                "traj_id",
                "segment_id",
                "effective_lr_mean",
                "is_traj_start",
            ]
        )
        empty_arr = np.empty((0, 0), dtype=np.float64)
        return EffectiveLrSegmentStatsBundle(
            network=empty,
            unit_mean=empty_arr,
            network_excl_mean=empty_arr,
        )

    network = pd.DataFrame(rows)
    if segment_duration_s is not None:
        network["time_s"] = (network["global_segment_idx"] + 0.5) * segment_duration_s

    return EffectiveLrSegmentStatsBundle(
        network=network,
        unit_mean=np.stack(unit_mean_rows, axis=0),
        network_excl_mean=np.stack(network_excl_rows, axis=0),
    )


def load_effective_lr_timeline(
    signals_dir: Path | str,
    *,
    segment_duration_s: float | None = None,
) -> pd.DataFrame:
    """
    Load mean effective learning rate over hidden units per TBPTT segment.

    Returns one row per segment with `global_segment_idx`, trajectory ids,
    `effective_lr_mean`, and `is_traj_start`. When `segment_duration_s` is
    given, `time_s` is the segment midpoint (same convention as
    `load_training_gradient_timeline`).
    """
    return load_effective_lr_segment_stats_bundle(
        signals_dir,
        segment_duration_s=segment_duration_s,
    ).network


def load_g_local_mse_segment_stats(
    signals_dir: Path | str,
    *,
    segment_duration_s: float,
) -> pd.DataFrame:
    """
    Per-segment `g_local_mse` statistics over timesteps (after mean over H).

    Returns one row per segment with `global_segment_idx`, trajectory ids, and
    `g_local_mse_mean`, `g_local_mse_max`, `g_local_mse_std` computed over
    timesteps within the segment.
    """
    _ = segment_duration_s
    return load_g_local_mse_segment_stats_bundle(
        signals_dir, segment_duration_s=segment_duration_s
    ).network


def load_g_local_mse_segment_stats_bundle(
    signals_dir: Path | str,
    *,
    segment_duration_s: float,
) -> GLocalMseSegmentStatsBundle:
    """
    Per-segment signed and magnitude `g_local_mse` stats for network and units.

    For each segment, per-unit stats are computed over timesteps on
    `g_local_mse[0, t, h]` (signed) or `|g_local_mse[0, t, h]|` (magnitude).
    Network stats use the mean-over-hidden timeline at each timestep.
    """
    _ = segment_duration_s
    signals_dir = Path(signals_dir)
    if not signals_dir.is_dir():
        raise FileNotFoundError(f"Missing signals directory: {signals_dir}")

    rows: list[dict[str, int | float | bool]] = []
    unit_signed_mean: list[np.ndarray] = []
    unit_signed_max: list[np.ndarray] = []
    unit_signed_std: list[np.ndarray] = []
    unit_abs_mean: list[np.ndarray] = []
    unit_abs_max: list[np.ndarray] = []
    unit_abs_std: list[np.ndarray] = []
    network_excl_signed_mean: list[np.ndarray] = []
    network_excl_abs_mean: list[np.ndarray] = []
    global_idx = 0
    for epoch_id, traj_id, seg_idx, seg_path in _iter_training_segment_files(signals_dir):
        with np.load(seg_path) as data:
            if "g_local_mse" not in data:
                raise KeyError(f"{seg_path} missing 'g_local_mse'")
            g = _validate_g_local_mse_array(data["g_local_mse"])
            local_t_network = g.mean(axis=1)
            net_mean, net_max, net_std = _g_local_mse_stats_over_timesteps(local_t_network)
            g_abs = np.abs(g)

        rows.append(
            {
                "global_segment_idx": global_idx,
                "epoch_id": epoch_id,
                "traj_id": traj_id,
                "segment_id": seg_idx,
                "g_local_mse_mean": net_mean,
                "g_local_mse_max": net_max,
                "g_local_mse_std": net_std,
                "is_traj_start": seg_idx == 0,
            }
        )
        unit_signed_mean.append(g.mean(axis=0))
        unit_signed_max.append(g.max(axis=0))
        unit_signed_std.append(g.std(axis=0))
        unit_abs_mean.append(g_abs.mean(axis=0))
        unit_abs_max.append(g_abs.max(axis=0))
        unit_abs_std.append(g_abs.std(axis=0))
        network_excl_signed_mean.append(_network_excl_unit_mean_over_timesteps(g))
        network_excl_abs_mean.append(_network_excl_unit_mean_over_timesteps(g_abs))
        global_idx += 1

    if not rows:
        empty = pd.DataFrame(
            columns=[
                "global_segment_idx",
                "epoch_id",
                "traj_id",
                "segment_id",
                "g_local_mse_mean",
                "g_local_mse_max",
                "g_local_mse_std",
                "is_traj_start",
            ]
        )
        empty_arr = np.empty((0, 0), dtype=np.float64)
        return GLocalMseSegmentStatsBundle(
            network=empty,
            unit_signed_mean=empty_arr,
            unit_signed_max=empty_arr,
            unit_signed_std=empty_arr,
            unit_abs_mean=empty_arr,
            unit_abs_max=empty_arr,
            unit_abs_std=empty_arr,
            network_excl_signed_mean=empty_arr,
            network_excl_abs_mean=empty_arr,
        )

    return GLocalMseSegmentStatsBundle(
        network=pd.DataFrame(rows),
        unit_signed_mean=np.stack(unit_signed_mean, axis=0),
        unit_signed_max=np.stack(unit_signed_max, axis=0),
        unit_signed_std=np.stack(unit_signed_std, axis=0),
        unit_abs_mean=np.stack(unit_abs_mean, axis=0),
        unit_abs_max=np.stack(unit_abs_max, axis=0),
        unit_abs_std=np.stack(unit_abs_std, axis=0),
        network_excl_signed_mean=np.stack(network_excl_signed_mean, axis=0),
        network_excl_abs_mean=np.stack(network_excl_abs_mean, axis=0),
    )
