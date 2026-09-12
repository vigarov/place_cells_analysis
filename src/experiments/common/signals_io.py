"""Load per-segment training signals saved under `results/.../signals/`.

Each `segment_*.npz` stores scalar `loss`, hidden-state gradient arrays
(`g_bptt_mse`, `g_local_mse`, `g_fr`), per-unit `opt_signals`, and
post-step full model weights as top-level `weight__*` arrays. Shampoo runs
also store inverse Kronecker factors as top-level `h_inv__*` arrays.
Alt-training on_opt runs also store `on_opt_scale_factors` (shape `(H,)`,
per-hidden-unit gradient multipliers applied before the optimizer step).
"""
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from optimizers.defaults import (
    DEFAULT_ADAM_EPS,
    OPTIMIZER_LR_KEY,
    OPTIMIZER_TYPE_KEY,
    optimizer_lr,
)

GRADIENT_SIGNAL_KEYS = ("g_bptt_mse", "g_local_mse", "g_fr")

TRACKED_LAYER_NAMES: tuple[str, ...] = ("recurrent", "input", "readout")

_HIDDEN_UNIT_NODE_ID_RE = re.compile(r"^(recurrent|input)\[(\d+)\]$")
_LAYER_NODE_ID_RE = re.compile(r"^(recurrent|input|readout)\[(\d+)\]$")
_READOUT_NODE_ID_RE = re.compile(r"^readout\[(\d+)\]$")


SINGLE_ROOM_NAME = "single_room"
TWO_ROOMS_NAME = "two_rooms"


def _parse_two_rooms_traj_dir_name(name: str) -> tuple[int, int, int] | None:
    m = re.fullmatch(r"rep(\d+)_room(\d+)_traj(\d+)", name)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def detect_signals_experiment_name(signals_dir: Path) -> str:
    """Infer experiment type from signal trajectory directory names."""
    for path in signals_dir.iterdir():
        if path.is_dir() and _parse_two_rooms_traj_dir_name(path.name) is not None:
            return TWO_ROOMS_NAME
    return SINGLE_ROOM_NAME


@dataclass(frozen=True)
class TrainingSegmentRef:
    traj_id: int
    segment_id: int
    path: Path
    epoch_id: int | None = None
    rep_id: int | None = None
    visit_room_id: int | None = None

    def metadata(self) -> dict[str, int]:
        out = {"traj_id": self.traj_id, "segment_id": self.segment_id}
        if self.rep_id is not None:
            out["rep_id"] = self.rep_id
            out["visit_room_id"] = int(self.visit_room_id)
        else:
            out["epoch_id"] = int(self.epoch_id)
        return out

    def sort_key(self) -> tuple[int, ...]:
        if self.rep_id is not None:
            return (self.rep_id, int(self.visit_room_id), self.traj_id, self.segment_id)
        return (int(self.epoch_id), self.traj_id, self.segment_id)


def _filter_last_rep(df: pd.DataFrame) -> pd.DataFrame:
    if "rep_id" not in df.columns:
        return df
    rep = int(df["rep_id"].max())
    return df[df["rep_id"] == rep].copy().reset_index(drop=True)


def _apply_last_rep_filter(df: pd.DataFrame, last_rep_only: bool) -> pd.DataFrame:
    if not last_rep_only:
        return df
    filtered = _filter_last_rep(df)
    if filtered.empty:
        return filtered
    filtered = filtered.copy()
    filtered["global_segment_idx"] = np.arange(len(filtered), dtype=int)
    if "time_s" in filtered.columns:
        segment_duration_s = filtered["time_s"].iloc[1] - filtered["time_s"].iloc[0] if len(filtered) > 1 else 0.0
        if segment_duration_s > 0:
            filtered["time_s"] = (filtered["global_segment_idx"] + 0.5) * segment_duration_s
    return filtered


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
    *,
    experiment_name: str | None = None,
    last_rep_only: bool = False,
) -> Iterator[TrainingSegmentRef]:
    """Yield training segment refs in protocol order."""
    experiment_name = experiment_name or detect_signals_experiment_name(signals_dir)
    traj_dirs: list[tuple[tuple[int, ...], Path]] = []

    for path in signals_dir.iterdir():
        if not path.is_dir():
            continue
        if experiment_name == TWO_ROOMS_NAME:
            parsed = _parse_two_rooms_traj_dir_name(path.name)
            if parsed is not None:
                traj_dirs.append((parsed, path))
        else:
            parsed = _parse_traj_dir_name(path.name)
            if parsed is not None:
                traj_dirs.append((parsed, path))

    if not traj_dirs:
        if experiment_name == TWO_ROOMS_NAME:
            raise FileNotFoundError(
                f"No rep*_room*_traj* directories under {signals_dir}"
            )
        raise FileNotFoundError(f"No epoch*_traj* directories under {signals_dir}")

    traj_dirs.sort(key=lambda item: item[0])
    refs: list[TrainingSegmentRef] = []
    for key, traj_dir in traj_dirs:
        seg_paths = sorted(
            traj_dir.glob("segment_*.npz"),
            key=_segment_idx_from_path,
        )
        if experiment_name == TWO_ROOMS_NAME:
            rep_id, visit_room_id, traj_id = key
            for seg_idx, seg_path in enumerate(seg_paths):
                expected_seg = _segment_idx_from_path(seg_path)
                if expected_seg != seg_idx:
                    raise ValueError(
                        f"Non-contiguous segment index in {seg_path} (expected {seg_idx})"
                    )
                refs.append(
                    TrainingSegmentRef(
                        rep_id=rep_id,
                        visit_room_id=visit_room_id,
                        traj_id=traj_id,
                        segment_id=seg_idx,
                        path=seg_path,
                    )
                )
        else:
            epoch_id, traj_id = key
            for seg_idx, seg_path in enumerate(seg_paths):
                expected_seg = _segment_idx_from_path(seg_path)
                if expected_seg != seg_idx:
                    raise ValueError(
                        f"Non-contiguous segment index in {seg_path} (expected {seg_idx})"
                    )
                refs.append(
                    TrainingSegmentRef(
                        epoch_id=epoch_id,
                        traj_id=traj_id,
                        segment_id=seg_idx,
                        path=seg_path,
                    )
                )

    if last_rep_only and experiment_name == TWO_ROOMS_NAME and refs:
        last_rep = max(ref.rep_id for ref in refs if ref.rep_id is not None)
        refs = [ref for ref in refs if ref.rep_id == last_rep]

    yield from refs


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
class DeltaWSegmentStatsBundle:
    """Per-segment weight change for network-wide and per-hidden-unit views.

    ``unit_mean[g, h]`` is the mean signed ΔW over incoming weights to hidden
    unit ``h`` after segment ``g`` (i.e. ``W[g+1] - W[g]``). The last segment
    is NaN. Aggregation matches effective LR: mean over incoming weights within
    ``recurrent`` and ``input``, then mean across those layers.
    """

    network: pd.DataFrame
    unit_mean: np.ndarray
    network_excl_mean: np.ndarray


_WEIGHT_SNAPSHOT_KEYS: dict[str, str] = {
    "recurrent": "weight__recurrent_layers__0__leaky_layer__linear_layer__weight",
    "input": "weight__recurrent_layers__0__projection_layer__weight",
    "readout": "weight__readout_layer__weight",
}


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


def _load_weight_snapshots_by_layer(data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    """Load tracked weight matrices from one segment NPZ."""
    out: dict[str, np.ndarray] = {}
    for layer, key in _WEIGHT_SNAPSHOT_KEYS.items():
        if key not in data:
            raise KeyError(f"segment NPZ missing {key!r}")
        out[layer] = np.asarray(data[key], dtype=np.float64)
    return out


def _delta_w_by_node_from_weight_snapshots(
    prev_by_layer: Mapping[str, np.ndarray],
    curr_by_layer: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Per-node signed ΔW between consecutive post-step weight snapshots."""
    out: dict[str, np.ndarray] = {}
    for layer in ("recurrent", "input"):
        delta = np.asarray(curr_by_layer[layer], dtype=np.float64) - np.asarray(
            prev_by_layer[layer], dtype=np.float64
        )
        for unit_idx in range(delta.shape[0]):
            out[f"{layer}[{unit_idx}]"] = delta[unit_idx]
    readout_delta = np.asarray(curr_by_layer["readout"], dtype=np.float64) - np.asarray(
        prev_by_layer["readout"], dtype=np.float64
    )
    for unit_idx in range(readout_delta.shape[0]):
        out[f"readout[{unit_idx}]"] = readout_delta[unit_idx]
    return out


def _node_signal_to_unit_scalar(value: np.ndarray | float) -> float:
    """Reduce one tracked node's optimizer signal to a scalar effective LR."""
    if isinstance(value, np.ndarray):
        return _mean_per_weight_effective_lr(value)
    return float(value)


def _adam_effective_lr_by_node(
    exp_avg_sq_by_node: Mapping[str, Any],
    *,
    lr: float,
    eps: float = DEFAULT_ADAM_EPS,
) -> dict[str, np.ndarray]:
    """Per-node per-weight Adam effective LR: ``lr / (sqrt(v) + eps)``."""
    out: dict[str, np.ndarray] = {}
    for node_id, val in exp_avg_sq_by_node.items():
        arr = np.asarray(val, dtype=np.float64)
        if arr.size == 0:
            out[str(node_id)] = np.array([float("nan")])
            continue
        out[str(node_id)] = lr / (np.sqrt(arr) + eps)
    return out


def _effective_lr_signal_key(optimizer_type: str) -> str:
    if optimizer_type == "adam":
        return "exp_avg_sq"
    if optimizer_type == "pure_shampoo":
        return "h_inv_norm"
    if optimizer_type in ("adagrad", "grafted_shampoo"):
        return "effective_lr"
    raise ValueError(
        f"Unsupported optimizer type {optimizer_type!r} for effective LR; "
        "supported: adagrad, adam, pure_shampoo, grafted_shampoo"
    )


def _resolve_optimizer_type(
    optimizer_config: Mapping[str, Any] | None,
    opt_signals: Mapping[str, Any],
) -> str:
    if optimizer_config is not None and OPTIMIZER_TYPE_KEY in optimizer_config:
        return str(optimizer_config[OPTIMIZER_TYPE_KEY])
    if "h_inv_norm" in opt_signals and "effective_lr" in opt_signals:
        return "grafted_shampoo"
    if "h_inv_norm" in opt_signals:
        return "pure_shampoo"
    if "effective_lr" in opt_signals:
        return "adagrad"
    if "exp_avg_sq" in opt_signals:
        return "adam"
    raise KeyError(
        "Could not infer optimizer type from opt_signals "
        f"(keys: {sorted(opt_signals)}); pass optimizer_config"
    )


def _learning_rate_for_signals(
    optimizer_config: Mapping[str, Any] | None,
    optimizer_type: str,
) -> float:
    if optimizer_config is not None and OPTIMIZER_LR_KEY in optimizer_config:
        return float(optimizer_config[OPTIMIZER_LR_KEY])
    return float(optimizer_lr(optimizer_type))


def _effective_lr_by_node_from_opt_signals(
    opt_signals: Mapping[str, Any],
    *,
    optimizer_config: Mapping[str, Any] | None = None,
    adam_eps: float = DEFAULT_ADAM_EPS,
) -> dict[str, np.ndarray | float]:
    """Build per-node effective LR signals for aggregation (AdaGrad-style mean)."""
    optimizer_type = _resolve_optimizer_type(optimizer_config, opt_signals)
    signal_key = _effective_lr_signal_key(optimizer_type)

    raw_by_node = opt_signals.get(signal_key)
    if not raw_by_node:
        raise KeyError(
            f"opt_signals missing {signal_key!r} for optimizer {optimizer_type!r}"
        )

    if optimizer_type == "adam":
        lr = _learning_rate_for_signals(optimizer_config, optimizer_type)
        return _adam_effective_lr_by_node(raw_by_node, lr=lr, eps=adam_eps)

    return dict(raw_by_node)


def _effective_lr_mean_by_layer(
    effective_lr_by_node: Mapping[str, np.ndarray | float],
) -> dict[str, float]:
    """Mean effective LR over tracked units within each layer at one segment."""
    by_layer: dict[str, list[float]] = {layer: [] for layer in TRACKED_LAYER_NAMES}
    for node_id, eff in effective_lr_by_node.items():
        m = _LAYER_NODE_ID_RE.fullmatch(str(node_id))
        if m is None:
            continue
        layer = m.group(1)
        if layer not in by_layer:
            continue
        by_layer[layer].append(_node_signal_to_unit_scalar(eff))
    return {
        layer: float(np.nanmean(vals)) if vals else float("nan")
        for layer, vals in by_layer.items()
    }


def _per_weight_effective_lr_by_tracked_layer_for_hidden_unit(
    effective_lr_by_node: Mapping[str, np.ndarray | float],
    cell_idx: int,
    *,
    layers: tuple[str, ...] = TRACKED_LAYER_NAMES,
) -> dict[str, np.ndarray]:
    """Per-weight effective LR for one hidden unit, keyed by tracked layer."""
    out: dict[str, np.ndarray] = {layer: np.empty(0, dtype=np.float64) for layer in layers}
    unit_idx = int(cell_idx)

    for layer in layers:
        if layer in ("recurrent", "input"):
            val = effective_lr_by_node.get(f"{layer}[{unit_idx}]")
            if val is None:
                continue
            out[layer] = np.asarray(val, dtype=np.float64).ravel()
            continue

        if layer != "readout":
            continue

        readout_vals: list[tuple[int, float]] = []
        for node_id, eff in effective_lr_by_node.items():
            m = _READOUT_NODE_ID_RE.fullmatch(str(node_id))
            if m is None:
                continue
            arr = np.asarray(eff, dtype=np.float64).ravel()
            if unit_idx >= arr.size:
                continue
            readout_vals.append((int(m.group(1)), float(arr[unit_idx])))
        if readout_vals:
            readout_vals.sort(key=lambda item: item[0])
            out[layer] = np.array([val for _, val in readout_vals], dtype=np.float64)

    return out


def _effective_lr_per_hidden_unit(
    effective_lr_by_node: Mapping[str, np.ndarray | float],
) -> np.ndarray:
    """Per-hidden-unit effective LR, shape `(H,)`.

    For each unit index, per-weight values are averaged within each tracked
    layer (`recurrent`, `input`), then combined across layers.
    """
    by_unit_index: dict[int, list[float]] = {}
    for node_id, eff in effective_lr_by_node.items():
        m = _HIDDEN_UNIT_NODE_ID_RE.fullmatch(str(node_id))
        if m is None:
            continue
        unit_idx = int(m.group(2))
        by_unit_index.setdefault(unit_idx, []).append(_node_signal_to_unit_scalar(eff))

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


def load_training_loss_timeline(
    signals_dir: Path | str,
    *,
    experiment_name: str | None = None,
    last_rep_only: bool = False,
) -> pd.DataFrame:
    """
    Load the full training loss per TBPTT segment (warmup excluded).

    Each `signals/<epoch_tag>_traj<N>/segment_<M>.npz` stores scalar `loss`
    (combined MSE + firing-rate regularization from training).

    Returns columns `global_segment_idx`, trajectory metadata
    (`epoch_id`/`traj_id` or `rep_id`/`visit_room_id`/`traj_id`),
    `segment_id`, `loss`, `is_traj_start`.
    """
    signals_dir = Path(signals_dir)
    if not signals_dir.is_dir():
        raise FileNotFoundError(f"Missing signals directory: {signals_dir}")

    rows: list[dict[str, int | float | bool]] = []
    global_idx = 0
    for seg_ref in _iter_training_segment_files(
        signals_dir,
        experiment_name=experiment_name,
        last_rep_only=last_rep_only,
    ):
        with np.load(seg_ref.path) as data:
            if "loss" not in data:
                raise KeyError(f"{seg_ref.path} missing 'loss'")
            loss = float(np.asarray(data["loss"]).item())

        rows.append(
            {
                "global_segment_idx": global_idx,
                **seg_ref.metadata(),
                "loss": loss,
                "is_traj_start": seg_ref.segment_id == 0,
            }
        )
        global_idx += 1

    return pd.DataFrame(rows)


def load_training_gradient_timeline(
    signals_dir: Path | str,
    *,
    segment_duration_s: float,
    experiment_name: str | None = None,
    last_rep_only: bool = False,
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
    for seg_ref in _iter_training_segment_files(
        signals_dir,
        experiment_name=experiment_name,
        last_rep_only=last_rep_only,
    ):
        with np.load(seg_ref.path) as data:
            for key in GRADIENT_SIGNAL_KEYS:
                if key not in data:
                    raise KeyError(f"{seg_ref.path} missing {key!r}")

            bptt_t = _mean_over_hidden_per_timepoint(data["g_bptt_mse"])
            local_t = _mean_over_hidden_per_timepoint(data["g_local_mse"])
            fr_mean = _mean_over_hidden_units(data["g_fr"])

        rows.append(
            {
                "global_segment_idx": global_idx,
                **seg_ref.metadata(),
                "time_s": (global_idx + 0.5) * segment_duration_s,
                "g_bptt_mse": float(bptt_t.mean()),
                "g_local_mse": float(local_t.mean()),
                "g_fr": fr_mean,
                "is_traj_start": seg_ref.segment_id == 0,
            }
        )
        global_idx += 1

    return pd.DataFrame(rows)


def load_effective_lr_segment_stats_bundle(
    signals_dir: Path | str,
    *,
    segment_duration_s: float | None = None,
    optimizer_config: Mapping[str, Any] | None = None,
    adam_eps: float = DEFAULT_ADAM_EPS,
    experiment_name: str | None = None,
    last_rep_only: bool = False,
) -> EffectiveLrSegmentStatsBundle:
    """
    Per-segment effective LR for each hidden unit (one scalar per segment).

    Per-node signals are reduced to one scalar per hidden unit (mean over
    incoming weights within ``recurrent`` and ``input``, then mean across
    layers), matching the AdaGrad aggregation:

    - **AdaGrad / Grafted Shampoo (grafting LR):** stored ``effective_lr``
    - **Adam:** ``lr / (sqrt(exp_avg_sq) + eps)`` per weight, then mean
    - **Pure Shampoo:** stored ``h_inv_norm`` (scalar per unit)
    """
    signals_dir = Path(signals_dir)
    if not signals_dir.is_dir():
        raise FileNotFoundError(f"Missing signals directory: {signals_dir}")

    rows: list[dict[str, int | float | bool]] = []
    unit_mean_rows: list[np.ndarray] = []
    network_excl_rows: list[np.ndarray] = []
    global_idx = 0
    for seg_ref in _iter_training_segment_files(
        signals_dir,
        experiment_name=experiment_name,
        last_rep_only=last_rep_only,
    ):
        with np.load(seg_ref.path, allow_pickle=True) as data:
            if "opt_signals" not in data:
                raise KeyError(f"{seg_ref.path} missing 'opt_signals'")
            opt_signals = _load_opt_signals_dict(data["opt_signals"])
            effective_lr_by_node = _effective_lr_by_node_from_opt_signals(
                opt_signals,
                optimizer_config=optimizer_config,
                adam_eps=adam_eps,
            )

        unit_vals = _effective_lr_per_hidden_unit(effective_lr_by_node)
        layer_means = _effective_lr_mean_by_layer(effective_lr_by_node)
        rows.append(
            {
                "global_segment_idx": global_idx,
                **seg_ref.metadata(),
                "effective_lr_mean": float(np.nanmean(unit_vals)),
                **{
                    f"effective_lr_mean_{layer}": layer_means[layer]
                    for layer in TRACKED_LAYER_NAMES
                },
                "is_traj_start": seg_ref.segment_id == 0,
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


def load_delta_w_segment_stats_bundle(
    signals_dir: Path | str,
    *,
    segment_duration_s: float | None = None,
    experiment_name: str | None = None,
    last_rep_only: bool = False,
) -> DeltaWSegmentStatsBundle:
    """
    Per-segment signed weight change for each hidden unit.

    For global segment index ``g``, ``unit_mean[g, h]`` is the mean ΔW over
    incoming weights to unit ``h`` with ``ΔW = W[g+1] - W[g]`` (post-step
    snapshots). The final segment is all NaN. Layer means in ``network`` use
    the same per-node aggregation as effective LR.
    """
    signals_dir = Path(signals_dir)
    if not signals_dir.is_dir():
        raise FileNotFoundError(f"Missing signals directory: {signals_dir}")

    segment_files = list(
        _iter_training_segment_files(
            signals_dir,
            experiment_name=experiment_name,
            last_rep_only=last_rep_only,
        )
    )
    if not segment_files:
        empty = pd.DataFrame(
            columns=[
                "global_segment_idx",
                "epoch_id",
                "traj_id",
                "segment_id",
                "delta_w_mean",
                "is_traj_start",
            ]
        )
        empty_arr = np.empty((0, 0), dtype=np.float64)
        return DeltaWSegmentStatsBundle(
            network=empty,
            unit_mean=empty_arr,
            network_excl_mean=empty_arr,
        )

    rows: list[dict[str, int | float | bool]] = []
    unit_mean_rows: list[np.ndarray] = []
    network_excl_rows: list[np.ndarray] = []
    n_hidden: int | None = None

    for global_idx, seg_ref in enumerate(segment_files):
        if global_idx + 1 >= len(segment_files):
            if n_hidden is None:
                with np.load(seg_ref.path) as data:
                    n_hidden = _load_weight_snapshots_by_layer(data)["recurrent"].shape[0]
            unit_vals = np.full(n_hidden, np.nan, dtype=np.float64)
            layer_means = {layer: float("nan") for layer in TRACKED_LAYER_NAMES}
        else:
            next_ref = segment_files[global_idx + 1]
            with np.load(seg_ref.path) as data, np.load(next_ref.path) as next_data:
                prev_by_layer = _load_weight_snapshots_by_layer(data)
                curr_by_layer = _load_weight_snapshots_by_layer(next_data)
            delta_w_by_node = _delta_w_by_node_from_weight_snapshots(
                prev_by_layer,
                curr_by_layer,
            )
            unit_vals = _effective_lr_per_hidden_unit(delta_w_by_node)
            layer_means = _effective_lr_mean_by_layer(delta_w_by_node)
            n_hidden = unit_vals.shape[0]

        rows.append(
            {
                "global_segment_idx": global_idx,
                **seg_ref.metadata(),
                "delta_w_mean": float(np.nanmean(unit_vals))
                if np.isfinite(unit_vals).any()
                else float("nan"),
                **{f"delta_w_mean_{layer}": layer_means[layer] for layer in TRACKED_LAYER_NAMES},
                "is_traj_start": seg_ref.segment_id == 0,
            }
        )
        unit_mean_rows.append(unit_vals)
        network_excl_rows.append(_network_excl_unit_mean_from_vector(unit_vals))

    network = pd.DataFrame(rows)
    if segment_duration_s is not None:
        network["time_s"] = (network["global_segment_idx"] + 0.5) * segment_duration_s

    return DeltaWSegmentStatsBundle(
        network=network,
        unit_mean=np.stack(unit_mean_rows, axis=0),
        network_excl_mean=np.stack(network_excl_rows, axis=0),
    )


def load_delta_w_timeline(
    signals_dir: Path | str,
    *,
    segment_duration_s: float | None = None,
    experiment_name: str | None = None,
    last_rep_only: bool = False,
) -> pd.DataFrame:
    """Load mean signed ΔW over hidden units per TBPTT segment."""
    return load_delta_w_segment_stats_bundle(
        signals_dir,
        segment_duration_s=segment_duration_s,
        experiment_name=experiment_name,
        last_rep_only=last_rep_only,
    ).network


def load_effective_lr_timeline(
    signals_dir: Path | str,
    *,
    segment_duration_s: float | None = None,
    optimizer_config: Mapping[str, Any] | None = None,
    adam_eps: float = DEFAULT_ADAM_EPS,
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
        optimizer_config=optimizer_config,
        adam_eps=adam_eps,
    ).network


def _segment_path_at_global_idx(
    signals_dir: Path,
    global_segment_idx: int,
    *,
    experiment_name: str | None = None,
    last_rep_only: bool = False,
) -> Path:
    """Return the NPZ path for a training segment by global index."""
    for idx, seg_ref in enumerate(
        _iter_training_segment_files(
            signals_dir,
            experiment_name=experiment_name,
            last_rep_only=last_rep_only,
        )
    ):
        if idx == global_segment_idx:
            return seg_ref.path
    raise IndexError(
        f"global_segment_idx {global_segment_idx} out of range under {signals_dir}"
    )


def load_hidden_unit_per_weight_effective_lr(
    signals_dir: Path | str,
    *,
    global_segment_idx: int,
    cell_idx: int,
    optimizer_config: Mapping[str, Any] | None = None,
    adam_eps: float = DEFAULT_ADAM_EPS,
) -> np.ndarray:
    """
    Per-weight effective LR for one hidden unit at one training segment.

    Concatenates incoming-weight effective LRs from ``recurrent[cell_idx]`` and
    ``input[cell_idx]`` (empty array if neither node is present).
    """
    by_layer = load_hidden_unit_per_weight_effective_lr_by_layer(
        signals_dir,
        global_segment_idx=global_segment_idx,
        cell_idx=cell_idx,
        optimizer_config=optimizer_config,
        adam_eps=adam_eps,
    )
    parts = [by_layer[layer] for layer in ("recurrent", "input") if by_layer[layer].size]
    if not parts:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(parts)


def load_hidden_unit_per_weight_effective_lr_by_layer(
    signals_dir: Path | str,
    *,
    global_segment_idx: int,
    cell_idx: int,
    optimizer_config: Mapping[str, Any] | None = None,
    adam_eps: float = DEFAULT_ADAM_EPS,
    layers: tuple[str, ...] = TRACKED_LAYER_NAMES,
) -> dict[str, np.ndarray]:
    """Per-weight effective LR for one hidden unit, split by tracked layer.

    ``recurrent`` and ``input`` use incoming weights to the hidden unit.
    ``readout`` uses outgoing weights from the hidden unit to each readout row.
    """
    signals_dir = Path(signals_dir)
    seg_path = _segment_path_at_global_idx(signals_dir, global_segment_idx)
    with np.load(seg_path, allow_pickle=True) as data:
        if "opt_signals" not in data:
            raise KeyError(f"{seg_path} missing 'opt_signals'")
        opt_signals = _load_opt_signals_dict(data["opt_signals"])
        effective_lr_by_node = _effective_lr_by_node_from_opt_signals(
            opt_signals,
            optimizer_config=optimizer_config,
            adam_eps=adam_eps,
        )

    return _per_weight_effective_lr_by_tracked_layer_for_hidden_unit(
        effective_lr_by_node,
        cell_idx,
        layers=layers,
    )


def _per_weight_delta_w_by_tracked_layer_for_hidden_unit(
    prev_by_layer: Mapping[str, np.ndarray],
    curr_by_layer: Mapping[str, np.ndarray],
    cell_idx: int,
    *,
    layers: tuple[str, ...] = TRACKED_LAYER_NAMES,
) -> dict[str, np.ndarray]:
    """Per-weight signed ΔW for one hidden unit, keyed by tracked layer."""
    out: dict[str, np.ndarray] = {layer: np.empty(0, dtype=np.float64) for layer in layers}
    unit_idx = int(cell_idx)

    for layer in layers:
        if layer in ("recurrent", "input"):
            prev = np.asarray(prev_by_layer[layer], dtype=np.float64)
            curr = np.asarray(curr_by_layer[layer], dtype=np.float64)
            if unit_idx >= prev.shape[0]:
                continue
            out[layer] = curr[unit_idx] - prev[unit_idx]
            continue

        if layer != "readout":
            continue

        prev = np.asarray(prev_by_layer["readout"], dtype=np.float64)
        curr = np.asarray(curr_by_layer["readout"], dtype=np.float64)
        if unit_idx >= prev.shape[1]:
            continue
        out[layer] = curr[:, unit_idx] - prev[:, unit_idx]

    return out


def load_hidden_unit_per_weight_delta_w_by_layer(
    signals_dir: Path | str,
    *,
    global_segment_idx: int,
    cell_idx: int,
    layers: tuple[str, ...] = TRACKED_LAYER_NAMES,
) -> dict[str, np.ndarray]:
    """Per-weight signed ΔW for one hidden unit after one training segment.

    Returns ``W[g+1] - W[g]`` for incoming/outgoing weights at global segment
    index ``g``. The final segment returns NaN vectors.
    """
    signals_dir = Path(signals_dir)
    gidx = int(global_segment_idx)
    seg_path = _segment_path_at_global_idx(signals_dir, gidx)
    try:
        next_path = _segment_path_at_global_idx(signals_dir, gidx + 1)
    except IndexError:
        with np.load(seg_path) as data:
            prev_by_layer = _load_weight_snapshots_by_layer(data)
        unit_idx = int(cell_idx)
        out: dict[str, np.ndarray] = {}
        for layer in layers:
            if layer in ("recurrent", "input"):
                prev = prev_by_layer[layer]
                if unit_idx >= prev.shape[0]:
                    out[layer] = np.empty(0, dtype=np.float64)
                else:
                    out[layer] = np.full(prev.shape[1], np.nan, dtype=np.float64)
            elif layer == "readout":
                prev = prev_by_layer["readout"]
                if unit_idx >= prev.shape[1]:
                    out[layer] = np.empty(0, dtype=np.float64)
                else:
                    out[layer] = np.full(prev.shape[0], np.nan, dtype=np.float64)
        return out

    with np.load(seg_path) as data, np.load(next_path) as next_data:
        prev_by_layer = _load_weight_snapshots_by_layer(data)
        curr_by_layer = _load_weight_snapshots_by_layer(next_data)

    return _per_weight_delta_w_by_tracked_layer_for_hidden_unit(
        prev_by_layer,
        curr_by_layer,
        cell_idx,
        layers=layers,
    )


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
    experiment_name: str | None = None,
    last_rep_only: bool = False,
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
    for seg_ref in _iter_training_segment_files(
        signals_dir,
        experiment_name=experiment_name,
        last_rep_only=last_rep_only,
    ):
        with np.load(seg_ref.path) as data:
            if "g_local_mse" not in data:
                raise KeyError(f"{seg_ref.path} missing 'g_local_mse'")
            g = _validate_g_local_mse_array(data["g_local_mse"])
            local_t_network = g.mean(axis=1)
            net_mean, net_max, net_std = _g_local_mse_stats_over_timesteps(local_t_network)
            g_abs = np.abs(g)

        rows.append(
            {
                "global_segment_idx": global_idx,
                **seg_ref.metadata(),
                "g_local_mse_mean": net_mean,
                "g_local_mse_max": net_max,
                "g_local_mse_std": net_std,
                "is_traj_start": seg_ref.segment_id == 0,
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
