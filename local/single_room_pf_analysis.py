"""PF formation and evolution analysis for single_room and two_rooms Gaussian fits."""

from __future__ import annotations

import json
import multiprocessing
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import gammaln, zeta
from scipy.stats import lognorm, nbinom, weibull_min
from scipy.stats.qmc import Sobol
from tqdm.auto import tqdm

from experiments.common.ratemaps_io import RatemapCaptureRef, parse_capture_tag
from experiments.common.signals_io import (
    DeltaWSegmentStatsBundle,
    EffectiveLrSegmentStatsBundle,
    GLocalMseSegmentStatsBundle,
)
from trajectories.constants import DEFAULT_PADDING

PIXEL_TO_CM_OFFSET = float(DEFAULT_PADDING)

def _resid_window_size(window_size: int) -> int:
    """Extended residual window: one extra segment on each side of the anchor."""
    return int(window_size) + 1

PF_TRACKING_METHODS = ("no_revive", "revive_constant", "revive_mean_displacement")
DEFAULT_PF_TRACKING_METHOD = "no_revive"
DEFAULT_CONSTANT_L2_SIMILARITY = 4.0
MIN_MEAN_DISPLACEMENT_SAMPLES = 3
MEAN_DISPLACEMENT_REVIVE_SEM_Z = 1.96
PF_LIFECYCLE_EFF_LR_META_KEY = "_meta"

PF_EFF_LR_GROUP_NAMES = (
    "birth",
    "peak",
    "other_revives",
    "absent",
    "pre_vicinity",
)

PF_GLOCAL_VARIANTS = ("signed", "abs", "normalized", "normalized_abs")
GLocalVariant = Literal["signed", "abs", "normalized", "normalized_abs"]

_PF_GLOCAL_WORKER_BUNDLE: GLocalMseSegmentStatsBundle | None = None
_PF_GLOCAL_WORKER_CAPTURE_TO_GLOBAL: dict[int, int] | None = None
_PF_GLOCAL_WORKER_PF_SEGMENT_DF: pd.DataFrame | None = None
_PF_GLOCAL_WORKER_CELL_SEGMENT_CAPS: dict[int, np.ndarray] | None = None

_PF_EFF_LR_WORKER_BUNDLE: EffectiveLrSegmentStatsBundle | None = None
_PF_EFF_LR_WORKER_CAPTURE_TO_GLOBAL: dict[int, int] | None = None
_PF_EFF_LR_WORKER_PF_SEGMENT_DF: pd.DataFrame | None = None
_PF_EFF_LR_WORKER_CELL_SEGMENT_CAPS: dict[int, np.ndarray] | None = None
_PF_EFF_LR_WORKER_WINDOW_SIZE: int | None = None

_PF_DELTA_W_WORKER_BUNDLE: DeltaWSegmentStatsBundle | None = None
_PF_DELTA_W_WORKER_CAPTURE_TO_GLOBAL: dict[int, int] | None = None
_PF_DELTA_W_WORKER_PF_SEGMENT_DF: pd.DataFrame | None = None
_PF_DELTA_W_WORKER_CELL_SEGMENT_CAPS: dict[int, np.ndarray] | None = None
_PF_DELTA_W_WORKER_WINDOW_SIZE: int | None = None

_PF_LIFECYCLE_OUTLIER_WORKER: dict[str, object] | None = None

_GAUSS_COLS = (
    "g1_amplitude",
    "g1_mu_x",
    "g1_mu_y",
    "g2_amplitude",
    "g2_mu_x",
    "g2_mu_y",
)


def pixel_to_cm(pixel: np.ndarray | float) -> np.ndarray | float:
    """Convert rate-map pixel coordinates to cm (100 cm accessible, 5 px padding)."""
    return np.asarray(pixel, dtype=np.float64) - PIXEL_TO_CM_OFFSET


def _parse_epoch_traj(stem: str) -> tuple[int, int]:
    m = re.fullmatch(r"epoch(\d+)_traj(\d+)", stem)
    if m is None:
        raise ValueError(f"Unexpected single_room trajectory file stem: {stem!r}")
    return int(m.group(1)), int(m.group(2))



SINGLE_ROOM_NAME = "single_room"
TWO_ROOMS_NAME = "two_rooms"


def infer_experiment_name(df: pd.DataFrame) -> str:
    """Infer experiment type from master/pf dataframe columns."""
    if "rep_id" in df.columns:
        return TWO_ROOMS_NAME
    return SINGLE_ROOM_NAME


def timeline_keys(df: pd.DataFrame) -> list[str]:
    if infer_experiment_name(df) == TWO_ROOMS_NAME:
        return ["rep_id", "visit_room_id", "traj_id"]
    return ["epoch_id", "traj_id"]


def timeline_keys_for(experiment_name: str) -> list[str]:
    if experiment_name == TWO_ROOMS_NAME:
        return ["rep_id", "visit_room_id", "traj_id"]
    return ["epoch_id", "traj_id"]


def capture_metadata_keys(df: pd.DataFrame) -> list[str]:
    return timeline_keys(df) + ["segment_id"]


def capture_dedupe_keys(df: pd.DataFrame) -> list[str]:
    return ["capture_idx"] + capture_metadata_keys(df)


def transition_event_keys(df: pd.DataFrame) -> list[str]:
    return timeline_keys(df) + ["segment_from", "segment_to"]


def _parse_two_rooms_stem(stem: str, *, eval_room_id: int) -> tuple[int, int, int]:
    m = re.fullmatch(rf"rep(\d+)_room(\d+)_traj(\d+)_room{eval_room_id}", stem)
    if m is None:
        raise ValueError(f"Unexpected two_rooms trajectory file stem: {stem!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _filter_last_training_block(df: pd.DataFrame) -> pd.DataFrame:
    if infer_experiment_name(df) == TWO_ROOMS_NAME:
        rep = int(df["rep_id"].max())
        return df[df["rep_id"] == rep].copy()
    epoch = int(df["epoch_id"].max())
    return df[df["epoch_id"] == epoch].copy()


def _apply_last_block_filter(df: pd.DataFrame, last_epoch_only: bool) -> pd.DataFrame:
    return _filter_last_training_block(df) if last_epoch_only else df.copy()


def _row_event_fields(row, keys: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in keys:
        val = getattr(row, key) if hasattr(row, key) else row[key]
        out[key] = int(val)
    return out


def visit_room_switch_vlines(
    timeline: pd.DataFrame,
    *,
    x_col: str,
) -> np.ndarray:
    """X positions where visit_room_id changes (two_rooms timelines only)."""
    if "visit_room_id" not in timeline.columns:
        return np.array([], dtype=float)
    room = timeline["visit_room_id"].to_numpy()
    xs = timeline[x_col].to_numpy(dtype=float)
    if room.size < 2:
        return np.array([], dtype=float)
    switches = np.where(room[1:] != room[:-1])[0] + 1
    return xs[switches]



def build_master_fits_df(
    fits_path: Path | str,
    capture_refs: list[RatemapCaptureRef],
    tag_cache: dict[Path, list[str]],
    *,
    experiment_name: str = SINGLE_ROOM_NAME,
) -> pd.DataFrame:
    """
    Long-form table keyed by ``(capture_idx, cell_idx)``.

    Columns include trajectory metadata, ``signal_max``, ``r2``, and Gaussian
    parameters ``g1_*``, ``g2_*``.

    ``r2`` is clipped to ``[0, ∞)`` for analysis (raw fits may be negative).
    """
    fits_path = Path(fits_path)
    with np.load(fits_path) as fits:
        r2 = np.maximum(np.asarray(fits["r2"], dtype=np.float64), 0.0)
        gaussian_params = np.asarray(fits["gaussian_params"], dtype=np.float64)
        signal_max = np.asarray(fits["signal_max"], dtype=np.float64)

    n_captures, n_cells, n_gaussians, n_params = gaussian_params.shape
    if n_gaussians < 2:
        raise ValueError(f"Expected at least 2 Gaussians, got {n_gaussians}")
    if n_params != 5:
        raise ValueError(f"Expected 5 params per Gaussian, got {n_params}")
    if len(capture_refs) != n_captures:
        raise ValueError(
            f"Gaussian fits have {n_captures} captures but found "
            f"{len(capture_refs)} ratemap captures"
        )

    traj_ids = np.empty(n_captures, dtype=np.int32)
    segment_ids = np.empty(n_captures, dtype=np.int32)
    rep_ids: np.ndarray | None = None
    visit_room_ids: np.ndarray | None = None
    eval_room_ids: np.ndarray | None = None
    epoch_ids: np.ndarray | None = None

    if experiment_name == TWO_ROOMS_NAME:
        rep_ids = np.empty(n_captures, dtype=np.int32)
        visit_room_ids = np.empty(n_captures, dtype=np.int32)
        eval_room_ids = np.empty(n_captures, dtype=np.int32)
        eval_match = re.search(r"_room(\d+)(?:\.npz)?$", Path(fits_path).name)
        if eval_match is None:
            raise ValueError(f"Could not infer eval room from fits path: {fits_path}")
        eval_room_id = int(eval_match.group(1))
    else:
        epoch_ids = np.empty(n_captures, dtype=np.int32)

    for capture_idx, ref in enumerate(capture_refs):
        if ref.path not in tag_cache:
            raise KeyError(f"Missing tag_cache entry for {ref.path}")
        tag = tag_cache[ref.path][ref.capture_idx]
        if experiment_name == TWO_ROOMS_NAME:
            assert rep_ids is not None and visit_room_ids is not None and eval_room_ids is not None
            rep, visit_room, traj = _parse_two_rooms_stem(
                ref.path.stem, eval_room_id=eval_room_id
            )
            rep_ids[capture_idx] = rep
            visit_room_ids[capture_idx] = visit_room
            traj_ids[capture_idx] = traj
            eval_room_ids[capture_idx] = eval_room_id
        else:
            assert epoch_ids is not None
            epoch_ids[capture_idx], traj_ids[capture_idx] = _parse_epoch_traj(ref.path.stem)
        segment_ids[capture_idx] = parse_capture_tag(tag)

    capture_idx_grid, cell_idx_grid = np.indices((n_captures, n_cells))
    flat_capture = capture_idx_grid.ravel()
    flat_cell = cell_idx_grid.ravel()

    data: dict[str, np.ndarray | list[int]] = {
        "capture_idx": flat_capture.astype(int),
        "cell_idx": flat_cell.astype(int),
        "traj_id": traj_ids[flat_capture],
        "segment_id": segment_ids[flat_capture],
        "signal_max": signal_max.ravel(),
        "r2": r2.ravel(),
    }
    if experiment_name == TWO_ROOMS_NAME:
        assert rep_ids is not None and visit_room_ids is not None and eval_room_ids is not None
        data["rep_id"] = rep_ids[flat_capture]
        data["visit_room_id"] = visit_room_ids[flat_capture]
        data["eval_room_id"] = eval_room_ids[flat_capture]
    else:
        assert epoch_ids is not None
        data["epoch_id"] = epoch_ids[flat_capture]

    df = pd.DataFrame(data)

    param_names = ("amplitude", "mu_x", "mu_y", "sigma_x", "sigma_y")
    for g in range(min(2, n_gaussians)):
        for p_idx, pname in enumerate(param_names):
            col = f"g{g + 1}_{pname}"
            df[col] = gaussian_params[:, :, g, p_idx].ravel()

    return df


def is_alive(
    df: pd.DataFrame,
    *,
    dead_threshold: float,
    r2_threshold: float,
) -> pd.Series:
    """True when ``signal_max > dead_threshold`` and ``r2 > r2_threshold``."""
    return (df["signal_max"] > dead_threshold) & (df["r2"] > r2_threshold)


def _symmetric_amplitude_ratio(
    g1_amp: np.ndarray | pd.Series | float,
    g2_amp: np.ndarray | pd.Series | float,
) -> np.ndarray:
    """Return ``max(|g1|, |g2|) / min(|g1|, |g2|)``; ``inf`` when min amplitude is 0."""
    g1 = np.asarray(g1_amp, dtype=np.float64)
    g2 = np.asarray(g2_amp, dtype=np.float64)
    numer = np.maximum(np.abs(g1), np.abs(g2))
    denom = np.minimum(np.abs(g1), np.abs(g2))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denom > 0, numer / denom, np.inf)
    return np.where(np.isfinite(ratio), ratio, np.inf)


def _dominant_gaussian_index(g1_amp: float, g2_amp: float) -> int:
    """Return 1 or 2 for the larger-amplitude fitted component (ties -> 1)."""
    return 2 if g2_amp > g1_amp else 1


def count_n_gaussians(
    g1_amp: np.ndarray | pd.Series,
    g2_amp: np.ndarray | pd.Series,
    gauss_threshold: float,
) -> np.ndarray:
    """Return 2 when amplitudes are similar (``max/min <= threshold``), else 1."""
    ratio = _symmetric_amplitude_ratio(g1_amp, g2_amp)
    return np.where(ratio <= gauss_threshold, 2, 1).astype(int)




def sobol_alive_proportion_last_epoch(
    df: pd.DataFrame,
    *,
    n_samples: int = 100,
    dead_range: tuple[float, float] = (0.1, 1.0),
    r2_range: tuple[float, float] = (0.5, 0.9),
    seed: int = 0,
) -> pd.DataFrame:
    """Sobol sample threshold pairs; proportion alive at last epoch."""
    sub = _filter_last_training_block(df)
    n_cells = sub["cell_idx"].nunique()
    if n_cells == 0:
        raise ValueError("No cells in last epoch")

    sampler = Sobol(d=2, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(n_samples)))
    unit = sampler.random_base2(m)[:n_samples]
    dead_lo, dead_hi = dead_range
    r2_lo, r2_hi = r2_range
    dead_thresholds = dead_lo + unit[:, 0] * (dead_hi - dead_lo)
    r2_thresholds = r2_lo + unit[:, 1] * (r2_hi - r2_lo)

    signal = sub["signal_max"].to_numpy()
    r2 = sub["r2"].to_numpy()
    rows: list[dict[str, float]] = []
    for dead_thr, r2_thr in zip(dead_thresholds, r2_thresholds, strict=True):
        alive = (signal > dead_thr) & (r2 > r2_thr)
        rows.append(
            {
                "dead_threshold": float(dead_thr),
                "r2_threshold": float(r2_thr),
                "proportion_alive": float(alive.mean()),
            }
        )
    return pd.DataFrame(rows)


def _capture_timeline(
    sub: pd.DataFrame,
    *,
    value_col: str = "alive",
    agg: str = "mean",
) -> pd.DataFrame:
    """One row per ratemap capture in training order."""
    group_keys = ["capture_idx"] + capture_metadata_keys(sub)
    sort_keys = capture_metadata_keys(sub)
    traj_keys = timeline_keys(sub)
    timeline = (
        sub.groupby(group_keys, sort=False)[value_col]
        .agg(agg)
        .reset_index()
        .sort_values(sort_keys, kind="mergesort")
        .reset_index(drop=True)
    )
    timeline["global_capture_idx"] = np.arange(len(timeline), dtype=int)
    timeline["is_traj_start"] = (
        timeline.groupby(traj_keys, sort=False).cumcount() == 0
    )
    return timeline


def _transition_timeline(
    pairs: pd.DataFrame,
    *,
    flag_cols: tuple[str, ...],
    out_names: tuple[str, ...],
) -> pd.DataFrame:
    """One row per within-trajectory segment transition in training order."""
    event_keys = transition_event_keys(pairs)
    traj_keys = timeline_keys(pairs)
    timeline = (
        pairs.groupby(event_keys, sort=False)
        .agg(**{name: (col, "sum") for name, col in zip(out_names, flag_cols, strict=True)})
        .reset_index()
        .sort_values(event_keys, kind="mergesort")
        .reset_index(drop=True)
    )
    timeline["global_transition_idx"] = np.arange(len(timeline), dtype=int)
    timeline["is_traj_start"] = (
        timeline.groupby(traj_keys, sort=False).cumcount() == 0
    )
    timeline["segment_from"] = timeline["segment_from"].astype(int)
    timeline["segment_to"] = timeline["segment_to"].astype(int)
    for name in out_names:
        timeline[name] = timeline[name].astype(float)
    return timeline


def traj_boundary_vlines(
    timeline: pd.DataFrame,
    *,
    x_col: str,
    skip_first: bool = True,
) -> np.ndarray:
    """X positions of trajectory starts for vertical dashed lines."""
    starts = timeline.loc[timeline["is_traj_start"], x_col].to_numpy(dtype=float)
    if skip_first and starts.size > 0:
        return starts[1:]
    return starts


def mean_r2_by_segment(
    df: pd.DataFrame,
    *,
    last_epoch_only: bool = True,
) -> pd.DataFrame:
    """Network-wide mean R² at each recorded capture (global training timeline)."""
    sub = _apply_last_block_filter(df, last_epoch_only)
    r2_col_name = "r2"
    timeline = _capture_timeline(sub, value_col=r2_col_name, agg="mean")
    return timeline.rename(columns={r2_col_name: "mean_r2"})


def alive_proportion_by_segment(
    df: pd.DataFrame,
    *,
    dead_threshold: float = 0.1,
    r2_threshold: float = 0.8,
    last_epoch_only: bool = True,
) -> pd.DataFrame:
    """Proportion alive at each recorded capture (global training timeline)."""
    sub = _apply_last_block_filter(df, last_epoch_only)
    sub = sub.assign(
        alive=is_alive(sub, dead_threshold=dead_threshold, r2_threshold=r2_threshold)
    )
    timeline = _capture_timeline(sub, value_col="alive", agg="mean")
    return timeline.rename(columns={"alive": "proportion_alive"})


def _build_transition_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Consecutive within-trajectory segment pairs for each cell."""
    tkeys = timeline_keys(df)
    ordered = df.sort_values(tkeys + ["cell_idx", "segment_id"]).copy()
    grouped = ordered.groupby(tkeys + ["cell_idx"], sort=False)
    seg_next = grouped["segment_id"].shift(-1)
    ordered["segment_next"] = seg_next
    ordered["alive_next"] = grouped["alive"].shift(-1)
    if "n_gaussians" in ordered.columns:
        ordered["n_gauss_next"] = grouped["n_gaussians"].shift(-1)
    for col in _GAUSS_COLS:
        if col in ordered.columns:
            ordered[f"{col}_next"] = grouped[col].shift(-1)

    # shift() on grouped booleans can yield object dtype; ~ breaks on object
    ordered["alive"] = ordered["alive"].astype(bool)
    ordered["alive_next"] = ordered["alive_next"].fillna(False).astype(bool)

    # segment_id goes -1, 0, 1, ... so consecutive diff is 1
    same_traj = seg_next.notna() & ((seg_next - ordered["segment_id"]) == 1)
    pairs = ordered[same_traj].copy()
    pairs = pairs.rename(
        columns={
            "segment_id": "segment_from",
            "segment_next": "segment_to",
            "alive": "alive_from",
            "alive_next": "alive_to",
            "n_gaussians": "n_gauss_from",
            "n_gauss_next": "n_gauss_to",
        }
    )
    return pairs.reset_index(drop=True)


def _per_trajectory_transition_counts(
    pairs: pd.DataFrame,
    *,
    flag_cols: tuple[str, ...],
    out_names: tuple[str, ...],
) -> pd.DataFrame:
    """Alias kept for internal call sites; returns global transition timeline."""
    return _transition_timeline(pairs, flag_cols=flag_cols, out_names=out_names)


def alive_state_transition_counts(
    df: pd.DataFrame,
    *,
    dead_threshold: float = 0.1,
    r2_threshold: float = 0.8,
    last_epoch_only: bool = True,
) -> pd.DataFrame:
    """Dead→alive and alive→dead counts for each actual segment transition.

    Returns one row per within-trajectory transition in training order
    (``n_train_traj × 60`` events; 610 captures yield 600 transitions).
    """
    sub = _apply_last_block_filter(df, last_epoch_only)
    sub = sub.assign(
        alive=is_alive(sub, dead_threshold=dead_threshold, r2_threshold=r2_threshold)
    )
    pairs = _build_transition_pairs(sub)

    pairs["dead_to_alive"] = (~pairs["alive_from"]) & pairs["alive_to"]
    pairs["alive_to_dead"] = pairs["alive_from"] & (~pairs["alive_to"])

    return _transition_timeline(
        pairs,
        flag_cols=("dead_to_alive", "alive_to_dead"),
        out_names=("n_dead_to_alive", "n_alive_to_dead"),
    )


def mean_n_gaussians_vs_threshold(
    df: pd.DataFrame,
    thresholds: np.ndarray | None = None,
    *,
    dead_threshold: float = 0.1,
    r2_threshold: float = 0.8,
    last_epoch_only: bool = True,
) -> pd.DataFrame:
    """Mean ± SEM of Gaussian count over alive cell×capture rows."""
    if thresholds is None:
        thresholds = np.linspace(1.0, 2.0, 11)
    sub = _apply_last_block_filter(df, last_epoch_only)
    sub = sub.assign(
        alive=is_alive(sub, dead_threshold=dead_threshold, r2_threshold=r2_threshold)
    )
    alive_rows = sub[sub["alive"]]
    if alive_rows.empty:
        return pd.DataFrame(columns=["gauss_threshold", "mean", "sem", "n_obs"])

    g1 = alive_rows["g1_amplitude"].to_numpy()
    g2 = alive_rows["g2_amplitude"].to_numpy()
    rows: list[dict[str, float | int]] = []
    for thr in thresholds:
        n_gauss = count_n_gaussians(g1, g2, float(thr))
        rows.append(
            {
                "gauss_threshold": float(thr),
                "mean": float(n_gauss.mean()),
                "sem": float(n_gauss.std(ddof=1) / np.sqrt(n_gauss.size))
                if n_gauss.size > 1
                else 0.0,
                "n_obs": int(n_gauss.size),
            }
        )
    return pd.DataFrame(rows)


def _prepare_alive_transition_df(
    df: pd.DataFrame,
    *,
    dead_threshold: float,
    r2_threshold: float,
    gauss_threshold: float,
    last_epoch_only: bool,
) -> pd.DataFrame:
    sub = _apply_last_block_filter(df, last_epoch_only)
    sub = sub.assign(
        alive=is_alive(sub, dead_threshold=dead_threshold, r2_threshold=r2_threshold),
        n_gaussians=count_n_gaussians(
            sub["g1_amplitude"],
            sub["g2_amplitude"],
            gauss_threshold,
        ),
    )
    pairs = _build_transition_pairs(sub)
    return pairs[pairs["alive_from"] & pairs["alive_to"]].copy()


def mean_n_gaussians_by_transition(
    df: pd.DataFrame,
    *,
    gauss_threshold: float = 1.5,
    dead_threshold: float = 0.1,
    r2_threshold: float = 0.8,
    last_epoch_only: bool = True,
) -> pd.DataFrame:
    """Mean ± SEM of Gaussian count per alive→alive segment transition event."""
    pairs = _prepare_alive_transition_df(
        df,
        dead_threshold=dead_threshold,
        r2_threshold=r2_threshold,
        gauss_threshold=gauss_threshold,
        last_epoch_only=last_epoch_only,
    )
    if pairs.empty:
        return pd.DataFrame(
            columns=transition_event_keys(pairs)
            + [
                "mean",
                "sem",
                "n_cells",
                "global_transition_idx",
                "is_traj_start",
            ]
        )

    pairs["n_gauss_mean"] = (pairs["n_gauss_from"] + pairs["n_gauss_to"]) / 2.0
    event_keys = transition_event_keys(pairs)
    stats = (
        pairs.groupby(event_keys, sort=False)["n_gauss_mean"]
        .agg(mean="mean", sem=lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0)
        .reset_index()
    )
    counts = (
        pairs.groupby(event_keys, sort=False)
        .size()
        .reset_index(name="n_cells")
    )
    timeline = stats.merge(counts, on=event_keys).sort_values(
        event_keys, kind="mergesort"
    ).reset_index(drop=True)
    timeline["global_transition_idx"] = np.arange(len(timeline), dtype=int)
    timeline["is_traj_start"] = (
        timeline.groupby(timeline_keys(pairs), sort=False).cumcount() == 0
    )
    return timeline


def gauss_state_transition_counts(
    df: pd.DataFrame,
    *,
    gauss_threshold: float = 1.5,
    dead_threshold: float = 0.1,
    r2_threshold: float = 0.8,
    last_epoch_only: bool = True,
) -> pd.DataFrame:
    """Counts of 1→2, 2→1, 1→1, 2→2 among alive→alive transitions."""
    pairs = _prepare_alive_transition_df(
        df,
        dead_threshold=dead_threshold,
        r2_threshold=r2_threshold,
        gauss_threshold=gauss_threshold,
        last_epoch_only=last_epoch_only,
    )
    if pairs.empty:
        return pd.DataFrame(
            columns=[
                "segment_from",
                "segment_to",
                "n_1_to_2",
                "n_2_to_1",
                "n_1_to_1",
                "n_2_to_2",
            ]
        )

    pairs["is_1_to_2"] = (pairs["n_gauss_from"] == 1) & (pairs["n_gauss_to"] == 2)
    pairs["is_2_to_1"] = (pairs["n_gauss_from"] == 2) & (pairs["n_gauss_to"] == 1)
    pairs["is_1_to_1"] = (pairs["n_gauss_from"] == 1) & (pairs["n_gauss_to"] == 1)
    pairs["is_2_to_2"] = (pairs["n_gauss_from"] == 2) & (pairs["n_gauss_to"] == 2)

    return _per_trajectory_transition_counts(
        pairs,
        flag_cols=("is_1_to_2", "is_2_to_1", "is_1_to_1", "is_2_to_2"),
        out_names=("n_1_to_2", "n_2_to_1", "n_1_to_1", "n_2_to_2"),
    )


def _gaussian_positions_cm(row: pd.Series, *, n_gauss: int, suffix: str = "") -> np.ndarray:
    """Return (n_gauss, 2) array of (x_cm, y_cm) for active Gaussians."""
    if n_gauss == 1:
        g1_amp = float(row[f"g1_amplitude{suffix}"])
        g2_amp = float(row[f"g2_amplitude{suffix}"])
        k = _dominant_gaussian_index(g1_amp, g2_amp)
        x = pixel_to_cm(row[f"g{k}_mu_x{suffix}"])
        y = pixel_to_cm(row[f"g{k}_mu_y{suffix}"])
        return np.asarray([[x, y]], dtype=np.float64)

    positions = []
    for k in range(1, n_gauss + 1):
        x = pixel_to_cm(row[f"g{k}_mu_x{suffix}"])
        y = pixel_to_cm(row[f"g{k}_mu_y{suffix}"])
        positions.append([x, y])
    return np.asarray(positions, dtype=np.float64)


def _match_displacements(
    pos_from: np.ndarray,
    pos_to: np.ndarray,
    n_from: int,
    n_to: int,
) -> list[tuple[float, float, float]]:
    """Return list of (l2, dx, dy) displacements in cm for matched Gaussians."""
    if n_from == 1 and n_to == 1:
        d = pos_to[0] - pos_from[0]
        return [(float(np.linalg.norm(d)), float(d[0]), float(d[1]))]

    if n_from == 1 and n_to == 2:
        dists = [np.linalg.norm(pos_to[j] - pos_from[0]) for j in range(2)]
        j = int(np.argmin(dists))
        d = pos_to[j] - pos_from[0]
        return [(float(np.linalg.norm(d)), float(d[0]), float(d[1]))]

    if n_from == 2 and n_to == 1:
        dists = [np.linalg.norm(pos_to[0] - pos_from[i]) for i in range(2)]
        i = int(np.argmin(dists))
        d = pos_to[0] - pos_from[i]
        return [(float(np.linalg.norm(d)), float(d[0]), float(d[1]))]

    if n_from == 2 and n_to == 2:
        cost = np.zeros((2, 2), dtype=np.float64)
        for i in range(2):
            for j in range(2):
                cost[i, j] = np.linalg.norm(pos_to[j] - pos_from[i])
        row_ind, col_ind = linear_sum_assignment(cost)
        out: list[tuple[float, float, float]] = []
        for i, j in zip(row_ind, col_ind, strict=True):
            d = pos_to[j] - pos_from[i]
            out.append((float(np.linalg.norm(d)), float(d[0]), float(d[1])))
        return out

    raise ValueError(f"Unsupported Gaussian count transition: {n_from} -> {n_to}")


def pf_displacement_samples(
    df: pd.DataFrame,
    *,
    gauss_threshold: float = 1.5,
    dead_threshold: float = 0.1,
    r2_threshold: float = 0.8,
    last_epoch_only: bool = True,
) -> pd.DataFrame:
    """All matched-PF displacement samples (cm) across alive→alive transitions."""
    pairs = _prepare_alive_transition_df(
        df,
        dead_threshold=dead_threshold,
        r2_threshold=r2_threshold,
        gauss_threshold=gauss_threshold,
        last_epoch_only=last_epoch_only,
    )
    if pairs.empty:
        return pd.DataFrame(
            columns=[
                "segment_from",
                "segment_to",
                "cell_idx",
                "displacement_l2",
                "displacement_x",
                "displacement_y",
            ]
        )

    rows: list[dict[str, float | int]] = []
    for _, row in pairs.iterrows():
        n_from = int(row["n_gauss_from"])
        n_to = int(row["n_gauss_to"])
        pos_from = _gaussian_positions_cm(row, n_gauss=n_from)
        pos_to = _gaussian_positions_cm(row, n_gauss=n_to, suffix="_next")
        for l2, dx, dy in _match_displacements(pos_from, pos_to, n_from, n_to):
            event_keys = transition_event_keys(pairs)
            rows.append(
                {
                    **{k: int(row[k]) for k in event_keys},
                    "cell_idx": int(row["cell_idx"]),
                    "displacement_l2": l2,
                    "displacement_x": dx,
                    "displacement_y": dy,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    event_keys = transition_event_keys(out)
    key_to_idx = (
        out[event_keys]
        .drop_duplicates()
        .sort_values(event_keys, kind="mergesort")
        .reset_index(drop=True)
    )
    key_to_idx["global_transition_idx"] = np.arange(len(key_to_idx), dtype=int)
    out = out.merge(key_to_idx, on=event_keys, how="left")
    return out


def bin_proportion_histogram(
    values: np.ndarray,
    *,
    bin_width: float = 3.6,
    bin_range: tuple[float, float] = (-90.0, 90.0),
) -> tuple[np.ndarray, np.ndarray, float]:
    """Histogram bin centers and proportion of samples per bin (sums to 1)."""
    lo, hi = bin_range
    edges = np.arange(lo, hi + bin_width, bin_width)
    centers = edges[:-1] + bin_width / 2.0
    if values.size == 0:
        return centers, np.zeros(len(centers), dtype=np.float64), float(bin_width)
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    if total == 0:
        return centers, np.zeros(len(centers), dtype=np.float64), float(bin_width)
    prop = counts.astype(np.float64) / float(total)
    return centers, prop, float(bin_width)


def displacement_histogram(
    samples: pd.DataFrame,
    *,
    bin_width: float = 3.6,
    bin_range: tuple[float, float] = (-90.0, 90.0),
) -> dict[str, np.ndarray]:
    """Proportion-per-bin histograms for L2, x, and y displacements."""
    centers, pdf_l2, width = bin_proportion_histogram(
        samples["displacement_l2"].to_numpy(),
        bin_width=bin_width,
        bin_range=bin_range,
    )
    _, pdf_x, _ = bin_proportion_histogram(
        samples["displacement_x"].to_numpy(),
        bin_width=bin_width,
        bin_range=bin_range,
    )
    _, pdf_y, _ = bin_proportion_histogram(
        samples["displacement_y"].to_numpy(),
        bin_width=bin_width,
        bin_range=bin_range,
    )
    return {
        "centers": centers,
        "pdf_l2": pdf_l2,
        "pdf_x": pdf_x,
        "pdf_y": pdf_y,
        "bin_width": np.float64(width),
    }


def transition_x_index(df: pd.DataFrame, *, prefix: str = "") -> pd.Series:
    """Deprecated: prefer ``global_transition_idx`` from timeline builders."""
    if "global_transition_idx" in df.columns:
        return df["global_transition_idx"]
    cols = ["segment_from", "segment_to"]
    if prefix:
        cols = [f"{prefix}{c}" if not c.startswith(prefix) else c for c in cols]
    unique = df[cols].drop_duplicates().sort_values(cols, kind="mergesort")
    mapping = {
        tuple(int(getattr(r, c)) for c in cols): i
        for i, r in enumerate(unique.itertuples(index=False))
    }
    return df.apply(
        lambda r: mapping[tuple(int(r[c]) for c in cols)],
        axis=1,
    )


def load_segment_duration_s(config_json_path: Path | str) -> float:
    """Load training segment duration in seconds from a run ``config.json``."""
    with Path(config_json_path).open(encoding="utf-8") as fh:
        config = json.load(fh)
    return float(config["experiment_config"]["training"]["train_step_size_s"])


TIME_AXIS_LABEL = "Time (s)"


def index_to_time_s(
    index: np.ndarray | pd.Series | float,
    segment_duration_s: float,
) -> np.ndarray:
    """Map segment/capture/transition indices to elapsed wall-clock seconds."""
    return np.asarray(index, dtype=float) * float(segment_duration_s)


def _match_gaussian_indices(
    pos_from: np.ndarray,
    pos_to: np.ndarray,
    n_from: int,
    n_to: int,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Return matched (from_idx, to_idx) pairs and unmatched from/to indices."""
    if n_from == 0:
        return [], [], list(range(n_to))
    if n_to == 0:
        return [], list(range(n_from)), []

    if n_from == 1 and n_to == 1:
        return [(0, 0)], [], []

    if n_from == 1 and n_to == 2:
        d0 = pos_to[0] - pos_from[0]
        d1 = pos_to[1] - pos_from[0]
        j = 0 if d0 @ d0 <= d1 @ d1 else 1
        return [(0, j)], [], [1 - j]

    if n_from == 2 and n_to == 1:
        d0 = pos_to[0] - pos_from[0]
        d1 = pos_to[0] - pos_from[1]
        i = 0 if d0 @ d0 <= d1 @ d1 else 1
        return [(i, 0)], [1 - i], []

    if n_from == 2 and n_to == 2:
        cost = np.empty((2, 2), dtype=np.float64)
        for i in range(2):
            diff = pos_to - pos_from[i]
            cost[i] = np.einsum("ij,ij->i", diff, diff)
        row_ind, col_ind = linear_sum_assignment(cost)
        matched = [(int(i), int(j)) for i, j in zip(row_ind, col_ind, strict=True)]
        unmatched_from = [i for i in range(2) if i not in row_ind]
        unmatched_to = [j for j in range(2) if j not in col_ind]
        return matched, unmatched_from, unmatched_to

    raise ValueError(f"Unsupported Gaussian count transition: {n_from} -> {n_to}")


def _validate_pf_tracking_method(method: str) -> str:
    if method not in PF_TRACKING_METHODS:
        raise ValueError(
            f"pf_tracking_method must be one of {PF_TRACKING_METHODS}, got {method!r}"
        )
    return method


@dataclass
class _PFRecord:
    pf_idx: int
    is_active: bool
    last_mu_x: float
    last_mu_y: float
    displacements_x: list[float] = field(default_factory=list)
    displacements_y: list[float] = field(default_factory=list)
    life_period_idx: int = 0
    life_periods: list[tuple[int, int | None]] = field(default_factory=list)


def _mean_displacement_revive_radius_from_step_l2(
    step_l2: np.ndarray,
    *,
    constant_l2: float,
    min_samples: int = MIN_MEAN_DISPLACEMENT_SAMPLES,
) -> float:
    """Revive search radius from per-step L2 displacements (cm)."""
    n = int(step_l2.size)
    if n < min_samples:
        return float(constant_l2)
    mean_radius = float(step_l2.mean())
    sem = float(step_l2.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean_radius + MEAN_DISPLACEMENT_REVIVE_SEM_Z * abs(sem)


def _mean_displacement_revive_radius(
    displacements_x: list[float],
    displacements_y: list[float],
    *,
    constant_l2: float,
    min_samples: int = MIN_MEAN_DISPLACEMENT_SAMPLES,
) -> float:
    """Revive search radius from tracked component displacements (cm)."""
    step_l2 = np.hypot(displacements_x, displacements_y)
    return _mean_displacement_revive_radius_from_step_l2(
        step_l2,
        constant_l2=constant_l2,
        min_samples=min_samples,
    )


def _mean_displacement_revive_radius_from_l2(
    displacements: list[float],
    *,
    constant_l2: float,
    min_samples: int = MIN_MEAN_DISPLACEMENT_SAMPLES,
) -> float:
    """Revive search radius from per-step L2 displacements (cm)."""
    return _mean_displacement_revive_radius_from_step_l2(
        np.asarray(displacements, dtype=np.float64),
        constant_l2=constant_l2,
        min_samples=min_samples,
    )


def _revive_threshold(
    pf_record: _PFRecord,
    *,
    method: str,
    constant_l2: float,
) -> float:
    if method == "revive_constant":
        return constant_l2
    if method == "revive_mean_displacement":
        return _mean_displacement_revive_radius(
            pf_record.displacements_x,
            pf_record.displacements_y,
            constant_l2=constant_l2,
        )
    raise ValueError(f"Unsupported revive method: {method!r}")


def _try_revive_pf(
    inactive_records: list[_PFRecord],
    mu_x: float,
    mu_y: float,
    *,
    method: str,
    constant_l2: float,
) -> _PFRecord | None:
    if method == "no_revive" or not inactive_records:
        return None

    best: _PFRecord | None = None
    best_dist_sq = float("inf")
    for rec in inactive_records:
        dx = mu_x - rec.last_mu_x
        dy = mu_y - rec.last_mu_y
        dist_sq = dx * dx + dy * dy
        thresh = _revive_threshold(rec, method=method, constant_l2=constant_l2)
        if dist_sq < thresh * thresh and dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best = rec
    return best


def _prepare_tracking_arrays(
    df: pd.DataFrame,
    *,
    dead_threshold: float,
    r2_threshold: float,
    gauss_threshold: float,
    last_epoch_only: bool,
) -> dict[str, np.ndarray]:
    """Return per-(cell, capture) arrays shaped (n_cells, n_captures)."""
    sub = _apply_last_block_filter(df, last_epoch_only)
    n_captures = int(sub["capture_idx"].nunique())
    n_cells = int(sub["cell_idx"].nunique())
    if n_captures * n_cells != len(sub):
        sub = sub.sort_values(["capture_idx", "cell_idx"], kind="mergesort").reset_index(
            drop=True
        )
        if n_captures * n_cells != len(sub):
            raise ValueError(
                f"Expected {n_captures * n_cells} rows, got {len(sub)} "
                "(master_df must be a full cell×capture grid)"
            )

    def _as_cell_capture(col: str) -> np.ndarray:
        return sub[col].to_numpy().reshape(n_captures, n_cells).T

    alive = (
        (sub["signal_max"].to_numpy() > dead_threshold)
        & (sub["r2"].to_numpy() > r2_threshold)
    ).reshape(n_captures, n_cells).T

    g1_amp = _as_cell_capture("g1_amplitude")
    g2_amp = _as_cell_capture("g2_amplitude")
    n_gauss = count_n_gaussians(g1_amp, g2_amp, gauss_threshold)

    g1_mu_x = pixel_to_cm(_as_cell_capture("g1_mu_x"))
    g1_mu_y = pixel_to_cm(_as_cell_capture("g1_mu_y"))
    g2_mu_x = pixel_to_cm(_as_cell_capture("g2_mu_x"))
    g2_mu_y = pixel_to_cm(_as_cell_capture("g2_mu_y"))

    arrays: dict[str, np.ndarray] = {
        "alive": alive,
        "n_gauss": n_gauss,
        "g1_amp": g1_amp,
        "g2_amp": g2_amp,
        "g1_mu_x": g1_mu_x,
        "g1_mu_y": g1_mu_y,
        "g2_mu_x": g2_mu_x,
        "g2_mu_y": g2_mu_y,
        "capture_idx": _as_cell_capture("capture_idx").astype(np.int32),
        "traj_id": _as_cell_capture("traj_id").astype(np.int32),
        "segment_id": _as_cell_capture("segment_id").astype(np.int32),
    }
    for col in timeline_keys(sub):
        arrays[col] = _as_cell_capture(col).astype(np.int32)
    return arrays


def _track_one_cell(
    cell_idx: int,
    *,
    alive: np.ndarray,
    n_gauss: np.ndarray,
    g1_amp: np.ndarray,
    g2_amp: np.ndarray,
    g1_mu_x: np.ndarray,
    g1_mu_y: np.ndarray,
    g2_mu_x: np.ndarray,
    g2_mu_y: np.ndarray,
    capture_idx: np.ndarray,
    traj_id: np.ndarray,
    segment_id: np.ndarray,
    timeline_meta: dict[str, np.ndarray],
    pf_tracking_method: str = DEFAULT_PF_TRACKING_METHOD,
    constant_l2_similarity: float = DEFAULT_CONSTANT_L2_SIMILARITY,
) -> dict[str, np.ndarray]:
    """Track PFs for one cell; returns column arrays for pf_segment rows."""
    pf_tracking_method = _validate_pf_tracking_method(pf_tracking_method)
    n_captures = alive.shape[0]
    cap_cell = int(cell_idx)
    cap_pf: list[int] = []
    cap_life_period: list[int] = []
    cap_capture: list[int] = []
    cap_traj: list[int] = []
    cap_segment: list[int] = []
    cap_is_segment: list[bool] = []
    cap_state: list[str] = []
    cap_amplitude: list[float] = []
    cap_mu_x: list[float] = []
    cap_mu_y: list[float] = []
    cap_timeline_meta: dict[str, list[int]] = {
        key: [] for key in timeline_meta
    }

    next_pf_idx = 0
    registry: dict[int, _PFRecord] = {}
    prev_active: list[tuple[int, float, float, float]] = []
    pos_buf = np.empty((2, 2), dtype=np.float64)

    def _append(
        t: int,
        pf_idx: int,
        life_period_idx: int,
        state: str,
        amp: float,
        mu_x: float,
        mu_y: float,
    ) -> None:
        cap_pf.append(pf_idx)
        cap_life_period.append(life_period_idx)
        cap_capture.append(int(capture_idx[t]))
        cap_traj.append(int(traj_id[t]))
        cap_segment.append(int(segment_id[t]))
        for key, arr in timeline_meta.items():
            cap_timeline_meta[key].append(int(arr[t]))
        cap_is_segment.append(int(segment_id[t]) >= 0)
        cap_state.append(state)
        cap_amplitude.append(amp)
        cap_mu_x.append(mu_x)
        cap_mu_y.append(mu_y)

    def _curr_gaussians(t: int) -> list[tuple[float, float, float]]:
        ng = int(n_gauss[t])
        if ng == 2:
            return [
                (float(g1_amp[t]), float(g1_mu_x[t]), float(g1_mu_y[t])),
                (float(g2_amp[t]), float(g2_mu_x[t]), float(g2_mu_y[t])),
            ]
        dominant = _dominant_gaussian_index(float(g1_amp[t]), float(g2_amp[t]))
        if dominant == 1:
            return [(float(g1_amp[t]), float(g1_mu_x[t]), float(g1_mu_y[t]))]
        return [(float(g2_amp[t]), float(g2_mu_x[t]), float(g2_mu_y[t]))]

    def _inactive_records(exclude: set[int] | None = None) -> list[_PFRecord]:
        excluded = exclude or set()
        return [
            rec
            for rec in registry.values()
            if not rec.is_active and rec.pf_idx not in excluded
        ]

    def _close_life_period(rec: _PFRecord, death_capture: int) -> None:
        if rec.life_periods and rec.life_periods[-1][1] is None:
            birth, _ = rec.life_periods[-1]
            rec.life_periods[-1] = (birth, death_capture)

    def _mark_pf_dead(rec: _PFRecord, t: int) -> None:
        rec.is_active = False
        _close_life_period(rec, int(capture_idx[t]))

    def _create_pf(
        pf_idx: int,
        amp: float,
        mu_x: float,
        mu_y: float,
        t: int,
    ) -> _PFRecord:
        rec = _PFRecord(
            pf_idx=pf_idx,
            is_active=True,
            last_mu_x=mu_x,
            last_mu_y=mu_y,
            life_period_idx=0,
        )
        rec.life_periods.append((int(capture_idx[t]), None))
        registry[pf_idx] = rec
        return rec

    def _revive_pf(rec: _PFRecord, amp: float, mu_x: float, mu_y: float, t: int) -> None:
        rec.is_active = True
        rec.life_period_idx += 1
        rec.last_mu_x = mu_x
        rec.last_mu_y = mu_y
        rec.life_periods.append((int(capture_idx[t]), None))

    def _activate_gaussian(
        amp: float,
        mu_x: float,
        mu_y: float,
        t: int,
        revived_this_capture: set[int],
    ) -> tuple[int, _PFRecord]:
        revived = _try_revive_pf(
            _inactive_records(exclude=revived_this_capture),
            mu_x,
            mu_y,
            method=pf_tracking_method,
            constant_l2=constant_l2_similarity,
        )
        if revived is not None:
            _revive_pf(revived, amp, mu_x, mu_y, t)
            revived_this_capture.add(revived.pf_idx)
            return revived.pf_idx, revived

        nonlocal next_pf_idx
        pf_idx = next_pf_idx
        next_pf_idx += 1
        rec = _create_pf(pf_idx, amp, mu_x, mu_y, t)
        return pf_idx, rec

    def _empty_result() -> dict[str, np.ndarray]:
        empty = {
            "cell_idx": np.empty(0, dtype=np.int32),
            "pf_idx": np.empty(0, dtype=np.int32),
            "life_period_idx": np.empty(0, dtype=np.int32),
            "capture_idx": np.empty(0, dtype=np.int32),
            "traj_id": np.empty(0, dtype=np.int32),
            "segment_id": np.empty(0, dtype=np.int32),
            "is_segment": np.empty(0, dtype=bool),
            "state": np.empty(0, dtype=object),
            "amplitude": np.empty(0, dtype=np.float64),
            "mu_x_cm": np.empty(0, dtype=np.float64),
            "mu_y_cm": np.empty(0, dtype=np.float64),
        }
        for key in timeline_meta:
            empty[key] = np.empty(0, dtype=np.int32)
        return empty

    for t in range(n_captures):
        if not alive[t]:
            for pf_idx, *_ in prev_active:
                rec = registry[pf_idx]
                _mark_pf_dead(rec, t)
                _append(
                    t,
                    pf_idx,
                    rec.life_period_idx,
                    "inactive",
                    np.nan,
                    np.nan,
                    np.nan,
                )
            prev_active = []
            continue

        curr = _curr_gaussians(t)
        n_to = len(curr)

        if not prev_active:
            prev_active = []
            revived_this_capture: set[int] = set()
            for amp, mu_x, mu_y in curr:
                pf_idx, rec = _activate_gaussian(
                    amp, mu_x, mu_y, t, revived_this_capture
                )
                prev_active.append((pf_idx, amp, mu_x, mu_y))
                _append(
                    t,
                    pf_idx,
                    rec.life_period_idx,
                    "active",
                    amp,
                    mu_x,
                    mu_y,
                )
            continue

        n_from = len(prev_active)
        for i in range(n_from):
            pos_buf[i, 0] = prev_active[i][2]
            pos_buf[i, 1] = prev_active[i][3]
        for j in range(n_to):
            pos_buf[j, 0] = curr[j][1]
            pos_buf[j, 1] = curr[j][2]

        matched, dead_from, born_to = _match_gaussian_indices(
            pos_buf[:n_from], pos_buf[:n_to], n_from, n_to
        )

        new_active: list[tuple[int, float, float, float]] = []
        for i, j in matched:
            pf_idx = prev_active[i][0]
            amp, mu_x, mu_y = curr[j]
            rec = registry[pf_idx]
            rec.displacements_x.append(mu_x - rec.last_mu_x)
            rec.displacements_y.append(mu_y - rec.last_mu_y)
            rec.last_mu_x = mu_x
            rec.last_mu_y = mu_y
            new_active.append((pf_idx, amp, mu_x, mu_y))
            _append(
                t,
                pf_idx,
                rec.life_period_idx,
                "active",
                amp,
                mu_x,
                mu_y,
            )

        for i in dead_from:
            pf_idx = prev_active[i][0]
            rec = registry[pf_idx]
            _mark_pf_dead(rec, t)
            _append(
                t,
                pf_idx,
                rec.life_period_idx,
                "inactive",
                np.nan,
                np.nan,
                np.nan,
            )

        revived_this_capture = set()
        for j in born_to:
            amp, mu_x, mu_y = curr[j]
            pf_idx, rec = _activate_gaussian(
                amp, mu_x, mu_y, t, revived_this_capture
            )
            new_active.append((pf_idx, amp, mu_x, mu_y))
            _append(
                t,
                pf_idx,
                rec.life_period_idx,
                "active",
                amp,
                mu_x,
                mu_y,
            )

        prev_active = new_active

    n_rows = len(cap_pf)
    if n_rows == 0:
        return _empty_result()

    result = {
        "cell_idx": np.full(n_rows, cap_cell, dtype=np.int32),
        "pf_idx": np.asarray(cap_pf, dtype=np.int32),
        "life_period_idx": np.asarray(cap_life_period, dtype=np.int32),
        "capture_idx": np.asarray(cap_capture, dtype=np.int32),
        "traj_id": np.asarray(cap_traj, dtype=np.int32),
        "segment_id": np.asarray(cap_segment, dtype=np.int32),
        "is_segment": np.asarray(cap_is_segment, dtype=bool),
        "state": np.asarray(cap_state, dtype=object),
        "amplitude": np.asarray(cap_amplitude, dtype=np.float64),
        "mu_x_cm": np.asarray(cap_mu_x, dtype=np.float64),
        "mu_y_cm": np.asarray(cap_mu_y, dtype=np.float64),
    }
    for key in timeline_meta:
        result[key] = np.asarray(cap_timeline_meta[key], dtype=np.int32)
    return result


def track_place_fields(
    df: pd.DataFrame,
    *,
    dead_threshold: float = 0.1,
    r2_threshold: float = 0.8,
    gauss_threshold: float = 1.5,
    segment_duration_s: float,
    last_epoch_only: bool = True,
    pf_tracking_method: str = DEFAULT_PF_TRACKING_METHOD,
    constant_l2_similarity: float = DEFAULT_CONSTANT_L2_SIMILARITY,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Longitudinal PF identity tracking per cell across all captures.

    Returns one row per PF per capture where the PF is active, plus one inactive
    row at the capture where it disappears. Each row includes ``life_period_idx``
    (0-based per ``pf_idx``; increments on revival).
    """
    _ = segment_duration_s
    pf_tracking_method = _validate_pf_tracking_method(pf_tracking_method)
    arrays = _prepare_tracking_arrays(
        df,
        dead_threshold=dead_threshold,
        r2_threshold=r2_threshold,
        gauss_threshold=gauss_threshold,
        last_epoch_only=last_epoch_only,
    )
    timeline_meta = {
        key: arrays[key]
        for key in timeline_keys(df)
    }
    n_cells = arrays["alive"].shape[0]

    chunks: list[dict[str, np.ndarray]] = []
    cell_iter: range | tqdm = range(n_cells)
    if show_progress:
        cell_iter = tqdm(cell_iter, desc="Track place fields", leave=False)
    for cell_idx in cell_iter:
        chunks.append(
            _track_one_cell(
                cell_idx,
                alive=arrays["alive"][cell_idx],
                n_gauss=arrays["n_gauss"][cell_idx],
                g1_amp=arrays["g1_amp"][cell_idx],
                g2_amp=arrays["g2_amp"][cell_idx],
                g1_mu_x=arrays["g1_mu_x"][cell_idx],
                g1_mu_y=arrays["g1_mu_y"][cell_idx],
                g2_mu_x=arrays["g2_mu_x"][cell_idx],
                g2_mu_y=arrays["g2_mu_y"][cell_idx],
                capture_idx=arrays["capture_idx"][cell_idx],
                traj_id=arrays["traj_id"][cell_idx],
                segment_id=arrays["segment_id"][cell_idx],
                timeline_meta={key: timeline_meta[key][cell_idx] for key in timeline_meta},
                pf_tracking_method=pf_tracking_method,
                constant_l2_similarity=constant_l2_similarity,
            )
        )

    if not chunks or all(len(c["pf_idx"]) == 0 for c in chunks):
        empty_cols = [
            "cell_idx",
            "pf_idx",
            "life_period_idx",
            "capture_idx",
            *timeline_keys(df),
            "traj_id",
            "segment_id",
            "is_segment",
            "state",
            "amplitude",
            "mu_x_cm",
            "mu_y_cm",
        ]
        return pd.DataFrame(columns=empty_cols)

    merged = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    out = pd.DataFrame(merged)
    return out.sort_values(
        ["cell_idx", "capture_idx", "pf_idx", "life_period_idx"],
        kind="mergesort",
    ).reset_index(drop=True)


def _gaussians_from_capture_row(
    row: pd.Series,
    *,
    gauss_threshold: float,
) -> list[tuple[float, float, float]]:
    """Return ``[(amplitude, mu_x_cm, mu_y_cm), ...]`` for one cell×capture row."""
    g1_amp = float(row["g1_amplitude"])
    g2_amp = float(row["g2_amplitude"])
    n_gauss = int(
        count_n_gaussians(
            np.asarray([g1_amp]),
            np.asarray([g2_amp]),
            gauss_threshold,
        )[0]
    )
    g1_mu_x = float(pixel_to_cm(row["g1_mu_x"]))
    g1_mu_y = float(pixel_to_cm(row["g1_mu_y"]))
    g2_mu_x = float(pixel_to_cm(row["g2_mu_x"]))
    g2_mu_y = float(pixel_to_cm(row["g2_mu_y"]))
    if n_gauss == 2:
        return [
            (g1_amp, g1_mu_x, g1_mu_y),
            (g2_amp, g2_mu_x, g2_mu_y),
        ]
    dominant = _dominant_gaussian_index(g1_amp, g2_amp)
    if dominant == 1:
        return [(g1_amp, g1_mu_x, g1_mu_y)]
    return [(g2_amp, g2_mu_x, g2_mu_y)]


def _genesis_search_radius(
    displacements: list[float],
    *,
    method: str,
    constant_l2: float,
) -> float:
    """Displacement search radius for genesis backtracking (mirrors revive logic)."""
    method = _validate_pf_tracking_method(method)
    if method == "revive_constant" or method == "no_revive":
        return float(constant_l2)
    if method == "revive_mean_displacement":
        return _mean_displacement_revive_radius_from_l2(
            displacements,
            constant_l2=constant_l2,
        )
    raise ValueError(f"Unsupported pf_tracking_method: {method!r}")


def _match_gaussian_within_radius(
    mu_x: float,
    mu_y: float,
    candidates: list[tuple[float, float, float]],
    *,
    radius: float,
) -> tuple[float, float, float] | None:
    """Pick closest Gaussian bump within ``radius`` of ``(mu_x, mu_y)``."""
    radius_sq = float(radius) * float(radius)
    best: tuple[float, float, float] | None = None
    best_dist_sq = float("inf")
    for amp, cx, cy in candidates:
        dx = cx - mu_x
        dy = cy - mu_y
        dist_sq = dx * dx + dy * dy
        if dist_sq <= radius_sq and dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best = (amp, cx, cy)
    return best


def _cell_training_capture_indices(
    master_df: pd.DataFrame,
    cell_idx: int,
) -> np.ndarray:
    caps = (
        master_df.loc[
            (master_df["cell_idx"] == cell_idx) & (master_df["segment_id"] >= 0)
        ]
        .drop_duplicates("capture_idx")
        .sort_values("capture_idx", kind="mergesort")["capture_idx"]
        .to_numpy(dtype=np.int32)
    )
    return caps


def _build_genesis_segment_row(
    master_row: pd.Series,
    *,
    cell_idx: int,
    pf_idx: int,
    life_period_idx: int,
    amp: float,
    mu_x: float,
    mu_y: float,
    timeline_cols: list[str],
) -> dict[str, object]:
    row: dict[str, object] = {
        "cell_idx": int(cell_idx),
        "pf_idx": int(pf_idx),
        "life_period_idx": int(life_period_idx),
        "capture_idx": int(master_row["capture_idx"]),
        "traj_id": int(master_row["traj_id"]),
        "segment_id": int(master_row["segment_id"]),
        "is_segment": int(master_row["segment_id"]) >= 0,
        "state": "active",
        "amplitude": float(amp),
        "mu_x_cm": float(mu_x),
        "mu_y_cm": float(mu_y),
    }
    for col in timeline_cols:
        row[col] = int(master_row[col])
    return row


def _canonical_pf_idx(
    cell_idx: int,
    pf_idx: int,
    pf_redirect: dict[tuple[int, int], int],
) -> int:
    """Follow merge redirects to the survivor ``pf_idx``."""
    key = (int(cell_idx), int(pf_idx))
    while key in pf_redirect:
        pf_idx = int(pf_redirect[key])
        key = (int(cell_idx), pf_idx)
    return int(pf_idx)


def _period_displacement_radius(
    period_rows: pd.DataFrame,
    *,
    pf_tracking_method: str,
    constant_l2: float,
) -> float:
    period_rows = period_rows.sort_values("capture_idx", kind="mergesort")
    mu_x = period_rows["mu_x_cm"].to_numpy(dtype=np.float64)
    mu_y = period_rows["mu_y_cm"].to_numpy(dtype=np.float64)
    displacements = [
        float(np.hypot(mu_x[i] - mu_x[i - 1], mu_y[i] - mu_y[i - 1]))
        for i in range(1, len(period_rows))
    ]
    return _genesis_search_radius(
        displacements,
        method=pf_tracking_method,
        constant_l2=constant_l2,
    )


def _genesis_active_period_rows(
    df: pd.DataFrame,
    *,
    cell_idx: int,
    pf_idx: int,
    life_period_idx: int,
) -> pd.DataFrame:
    return df.loc[
        (df["cell_idx"] == cell_idx)
        & (df["pf_idx"] == pf_idx)
        & (df["life_period_idx"] == life_period_idx)
        & df["is_segment"]
        & (df["state"] == "active")
    ].sort_values("capture_idx", kind="mergesort")


def _genesis_position_at_capture(
    df: pd.DataFrame,
    *,
    cell_idx: int,
    pf_idx: int,
    life_period_idx: int,
    capture_idx: int,
) -> tuple[float, float] | None:
    rows = _genesis_active_period_rows(
        df,
        cell_idx=cell_idx,
        pf_idx=pf_idx,
        life_period_idx=life_period_idx,
    )
    match = rows.loc[rows["capture_idx"] == int(capture_idx)]
    if match.empty:
        return None
    row = match.iloc[0]
    return float(row["mu_x_cm"]), float(row["mu_y_cm"])


def _apply_genesis_pf_merge(
    df: pd.DataFrame,
    existing_keys: set[tuple[int, int, int, int]],
    prepended_rows: list[dict[str, object]],
    *,
    cell_idx: int,
    survivor_pf_idx: int,
    victim_pf_idx: int,
) -> None:
    """Remap ``victim_pf_idx`` rows to ``survivor_pf_idx`` on one cell."""
    cell_idx = int(cell_idx)
    survivor_pf_idx = int(survivor_pf_idx)
    victim_pf_idx = int(victim_pf_idx)
    if survivor_pf_idx == victim_pf_idx:
        return

    victim_mask = (df["cell_idx"] == cell_idx) & (df["pf_idx"] == victim_pf_idx)
    df.loc[victim_mask, "pf_idx"] = survivor_pf_idx

    for row in prepended_rows:
        if int(row["cell_idx"]) == cell_idx and int(row["pf_idx"]) == victim_pf_idx:
            row["pf_idx"] = survivor_pf_idx

    updated_keys: set[tuple[int, int, int, int]] = set()
    for key in existing_keys:
        c, cap, p, life = key
        if c == cell_idx and p == victim_pf_idx:
            updated_keys.add((c, cap, survivor_pf_idx, life))
        else:
            updated_keys.add(key)
    existing_keys.clear()
    existing_keys.update(updated_keys)


def _genesis_try_merge_at_death(
    df: pd.DataFrame,
    existing_keys: set[tuple[int, int, int, int]],
    prepended_rows: list[dict[str, object]],
    pf_redirect: dict[tuple[int, int], int],
    death_endpoints: pd.DataFrame,
    *,
    cell_idx: int,
    pf_idx: int,
    life_period_idx: int,
    capture_idx: int,
    current_mu_x: float,
    current_mu_y: float,
    radius: float,
) -> bool:
    """
    Merge with another PF whose life period ends at ``capture_idx``.

    Returns True if a merge occurred (caller should stop backtracking).
    """
    candidates = death_endpoints.loc[
        (death_endpoints["cell_idx"] == cell_idx)
        & (death_endpoints["death_capture_idx"] == int(capture_idx))
    ]
    radius_sq = float(radius) * float(radius)
    best: tuple[int, int, float] | None = None

    for row in candidates.itertuples(index=False):
        other_pf = int(row.pf_idx)
        other_life = int(row.life_period_idx)
        if other_pf == int(pf_idx) and other_life == int(life_period_idx):
            continue
        other_pf = _canonical_pf_idx(cell_idx, other_pf, pf_redirect)
        if other_pf == int(pf_idx):
            continue
        pos = _genesis_position_at_capture(
            df,
            cell_idx=cell_idx,
            pf_idx=other_pf,
            life_period_idx=other_life,
            capture_idx=capture_idx,
        )
        if pos is None:
            continue
        dx = pos[0] - current_mu_x
        dy = pos[1] - current_mu_y
        dist_sq = dx * dx + dy * dy
        if dist_sq <= radius_sq and (best is None or dist_sq < best[2]):
            best = (other_pf, other_life, dist_sq)

    if best is None:
        return False

    other_pf = best[0]
    survivor = min(int(pf_idx), int(other_pf))
    victim = max(int(pf_idx), int(other_pf))
    _apply_genesis_pf_merge(
        df,
        existing_keys,
        prepended_rows,
        cell_idx=cell_idx,
        survivor_pf_idx=survivor,
        victim_pf_idx=victim,
    )
    pf_redirect[(cell_idx, victim)] = survivor
    return True


def extend_pf_genesis_segments(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    r2_threshold: float,
    gauss_threshold: float,
    pf_tracking_method: str = DEFAULT_PF_TRACKING_METHOD,
    constant_l2_similarity: float = DEFAULT_CONSTANT_L2_SIMILARITY,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Prepend genesis segments before each PF life-period birth/revive.

    Starting from the strict birth/revive anchor, backtrack through earlier
    training captures using displacement matching (per ``pf_tracking_method``)
    while keeping the R² gate and dropping the amplitude/dead threshold gate.

    Life periods are processed in reverse death order (last dead first). When
    backtracking reaches the final active capture of another nearby PF on the
    same cell, the two identities merge (lowest ``pf_idx`` survives) and
    backtracking for that period stops.
    """
    if pf_segment_df.empty:
        return pf_segment_df.copy()

    pf_tracking_method = _validate_pf_tracking_method(pf_tracking_method)
    timeline_cols = timeline_keys(master_df)
    master_lookup = master_df.set_index(["cell_idx", "capture_idx"], drop=False)

    df = pf_segment_df.copy()
    existing_keys = set(
        zip(
            df["cell_idx"].to_numpy(dtype=np.int64),
            df["capture_idx"].to_numpy(dtype=np.int64),
            df["pf_idx"].to_numpy(dtype=np.int64),
            df["life_period_idx"].to_numpy(dtype=np.int64),
            strict=True,
        )
    )
    prepended_rows: list[dict[str, object]] = []
    pf_redirect: dict[tuple[int, int], int] = {}

    life_periods = extract_pf_life_periods(df)
    if life_periods.empty:
        return df

    tasks = life_periods.sort_values(
        ["cell_idx", "death_capture_idx", "pf_idx", "life_period_idx"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )

    if show_progress:
        task_source = tqdm(
            tasks.itertuples(index=False),
            total=len(tasks),
            desc="Genesis extension",
            leave=False,
        )
    else:
        task_source = tasks.itertuples(index=False)

    for row in task_source:
        cell_idx = int(row.cell_idx)
        orig_pf_idx = int(row.pf_idx)
        life_period_idx = int(row.life_period_idx)

        if _canonical_pf_idx(cell_idx, orig_pf_idx, pf_redirect) != orig_pf_idx:
            continue

        pf_idx = _canonical_pf_idx(cell_idx, orig_pf_idx, pf_redirect)
        period_rows = _genesis_active_period_rows(
            df,
            cell_idx=cell_idx,
            pf_idx=pf_idx,
            life_period_idx=life_period_idx,
        )
        if period_rows.empty:
            continue

        radius = _period_displacement_radius(
            period_rows,
            pf_tracking_method=pf_tracking_method,
            constant_l2=constant_l2_similarity,
        )

        anchor_cap = int(period_rows.iloc[0]["capture_idx"])
        current_mu_x = float(period_rows.iloc[0]["mu_x_cm"])
        current_mu_y = float(period_rows.iloc[0]["mu_y_cm"])

        cell_caps = _cell_training_capture_indices(master_df, cell_idx)
        if cell_caps.size == 0:
            continue
        pos = int(np.searchsorted(cell_caps, anchor_cap))
        if pos >= cell_caps.size or int(cell_caps[pos]) != anchor_cap:
            continue

        death_endpoints = extract_pf_life_periods(df)

        for cap_idx in range(pos - 1, -1, -1):
            cap = int(cell_caps[cap_idx])
            key = (cell_idx, cap, pf_idx, life_period_idx)
            if key in existing_keys:
                break

            if _genesis_try_merge_at_death(
                df,
                existing_keys,
                prepended_rows,
                pf_redirect,
                death_endpoints,
                cell_idx=cell_idx,
                pf_idx=pf_idx,
                life_period_idx=life_period_idx,
                capture_idx=cap,
                current_mu_x=current_mu_x,
                current_mu_y=current_mu_y,
                radius=radius,
            ):
                pf_idx = _canonical_pf_idx(cell_idx, pf_idx, pf_redirect)
                break

            try:
                master_row = master_lookup.loc[(cell_idx, cap)]
            except KeyError:
                break
            if isinstance(master_row, pd.DataFrame):
                master_row = master_row.iloc[0]
            if float(master_row["r2"]) <= r2_threshold:
                break

            candidates = _gaussians_from_capture_row(
                master_row,
                gauss_threshold=gauss_threshold,
            )
            matched = _match_gaussian_within_radius(
                current_mu_x,
                current_mu_y,
                candidates,
                radius=radius,
            )
            if matched is None:
                break

            amp, match_mu_x, match_mu_y = matched
            prepended_rows.append(
                _build_genesis_segment_row(
                    master_row,
                    cell_idx=cell_idx,
                    pf_idx=pf_idx,
                    life_period_idx=life_period_idx,
                    amp=amp,
                    mu_x=match_mu_x,
                    mu_y=match_mu_y,
                    timeline_cols=timeline_cols,
                )
            )
            existing_keys.add(key)
            current_mu_x = match_mu_x
            current_mu_y = match_mu_y

    if prepended_rows:
        df = pd.concat([pd.DataFrame(prepended_rows), df], ignore_index=True)

    return df.sort_values(
        ["cell_idx", "capture_idx", "pf_idx", "life_period_idx"],
        kind="mergesort",
    ).reset_index(drop=True)


def genesis_backtrack_depths(
    strict_pf_segment_df: pd.DataFrame,
    genesis_pf_segment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Count prepended genesis segments per ``(cell_idx, pf_idx, life_period_idx)``.

    Returns columns ``cell_idx, pf_idx, life_period_idx, n_prepended``.
    """
    def _active_keys(df: pd.DataFrame) -> set[tuple[int, int, int, int]]:
        active = df.loc[df["is_segment"] & (df["state"] == "active")]
        return {
            (int(r.cell_idx), int(r.pf_idx), int(r.life_period_idx), int(r.capture_idx))
            for r in active.itertuples(index=False)
        }

    strict_keys = _active_keys(strict_pf_segment_df)
    genesis_keys = _active_keys(genesis_pf_segment_df)
    added = genesis_keys - strict_keys
    if not added:
        return pd.DataFrame(
            columns=["cell_idx", "pf_idx", "life_period_idx", "n_prepended"]
        )

    counts: dict[tuple[int, int, int], int] = defaultdict(int)
    for cell_idx, pf_idx, life_period_idx, _cap in added:
        counts[(cell_idx, pf_idx, life_period_idx)] += 1

    rows = [
        {
            "cell_idx": cell_idx,
            "pf_idx": pf_idx,
            "life_period_idx": life_period_idx,
            "n_prepended": n_prepended,
        }
        for (cell_idx, pf_idx, life_period_idx), n_prepended in sorted(counts.items())
    ]
    return pd.DataFrame(rows)


def extract_pf_life_periods(pf_segment_df: pd.DataFrame) -> pd.DataFrame:
    """Return birth/death capture indices for each PF life period."""
    if pf_segment_df.empty:
        return pd.DataFrame(
            columns=[
                "cell_idx",
                "pf_idx",
                "life_period_idx",
                "birth_capture_idx",
                "death_capture_idx",
            ]
        )

    active = pf_segment_df[pf_segment_df["state"] == "active"].copy()
    if active.empty:
        return pd.DataFrame(
            columns=[
                "cell_idx",
                "pf_idx",
                "life_period_idx",
                "birth_capture_idx",
                "death_capture_idx",
            ]
        )

    grouped = (
        active.sort_values(
            ["cell_idx", "pf_idx", "life_period_idx", "capture_idx"],
            kind="mergesort",
        )
        .groupby(["cell_idx", "pf_idx", "life_period_idx"], sort=False)["capture_idx"]
        .agg(birth_capture_idx="min", death_capture_idx="max")
        .reset_index()
    )
    return grouped.sort_values(
        ["cell_idx", "pf_idx", "life_period_idx"],
        kind="mergesort",
    ).reset_index(drop=True)


def compute_pf_dead_segments_before_revival(
    pf_segment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Count training segments between each PF death and its subsequent revival.

    One row per death→revival pair (same ``cell_idx``, ``pf_idx``). A PF's final
    death with no later revival is excluded.
    """
    life_periods = extract_pf_life_periods(pf_segment_df)
    if life_periods.empty:
        return pd.DataFrame(
            columns=[
                "cell_idx",
                "pf_idx",
                "life_period_idx",
                "death_capture_idx",
                "revival_capture_idx",
                "n_dead_segments",
            ]
        )

    seg_caps = (
        pf_segment_df.loc[pf_segment_df["is_segment"]]
        .drop_duplicates("capture_idx")["capture_idx"]
        .sort_values(kind="mergesort")
        .to_numpy(dtype=np.int32)
    )

    rows: list[dict] = []
    grouped = life_periods.sort_values(
        ["cell_idx", "pf_idx", "life_period_idx"], kind="mergesort"
    ).groupby(["cell_idx", "pf_idx"], sort=False)

    for (cell_idx, pf_idx), periods in grouped:
        periods = periods.reset_index(drop=True)
        for i in range(len(periods) - 1):
            death_cap = int(periods.iloc[i]["death_capture_idx"])
            revival_cap = int(periods.iloc[i + 1]["birth_capture_idx"])
            n_dead = int(np.sum((seg_caps > death_cap) & (seg_caps < revival_cap)))
            rows.append(
                {
                    "cell_idx": int(cell_idx),
                    "pf_idx": int(pf_idx),
                    "life_period_idx": int(periods.iloc[i]["life_period_idx"]),
                    "death_capture_idx": death_cap,
                    "revival_capture_idx": revival_cap,
                    "n_dead_segments": n_dead,
                }
            )

    return pd.DataFrame(rows)


def extract_pf_revive_events(pf_segment_df: pd.DataFrame) -> pd.DataFrame:
    """First active capture for each revival (``life_period_idx > 0``)."""
    active_seg = pf_segment_df[
        pf_segment_df["is_segment"] & (pf_segment_df["state"] == "active")
    ]
    if active_seg.empty:
        return pd.DataFrame(
            columns=["cell_idx", "pf_idx", "life_period_idx", "capture_idx"]
        )

    revives = active_seg[active_seg["life_period_idx"] > 0].sort_values(
        ["cell_idx", "pf_idx", "life_period_idx", "capture_idx"],
        kind="mergesort",
    )
    if revives.empty:
        return pd.DataFrame(
            columns=["cell_idx", "pf_idx", "life_period_idx", "capture_idx"]
        )

    first = (
        revives.groupby(["cell_idx", "pf_idx", "life_period_idx"], sort=False)
        .first()
        .reset_index()
    )
    return first[["cell_idx", "pf_idx", "life_period_idx", "capture_idx"]].sort_values(
        ["cell_idx", "pf_idx", "life_period_idx"],
        kind="mergesort",
    ).reset_index(drop=True)


def training_capture_to_global_segment_map(
    pf_segment_df: pd.DataFrame,
) -> dict[int, int]:
    """Map ``capture_idx`` to global training segment index (``seg*`` order)."""
    meta_keys = capture_metadata_keys(pf_segment_df)
    caps = (
        pf_segment_df.loc[pf_segment_df["is_segment"]]
        .drop_duplicates(capture_dedupe_keys(pf_segment_df))
        .sort_values(meta_keys, kind="mergesort")
        .reset_index(drop=True)
    )
    return {int(row.capture_idx): int(idx) for idx, row in caps.iterrows()}


def summarize_place_field_metrics(
    pf_segment_df: pd.DataFrame,
    *,
    segment_duration_s: float,
    max_factor_pct: float = 1.0,
) -> pd.DataFrame:
    """
    Aggregate per-PF metrics over active training segments (seg* only).

    One row per ``(cell_idx, pf_idx)`` summed across all life periods:
    - ``length_s`` — total active duration across every life period
    - ``start_iteration_offset`` — global training segment index at first birth
    - ``peak_amplitude`` — global maximum amplitude across all periods
    - ``tma_s`` — time from the **start of the life period containing the peak**
      to the first segment whose amplitude reaches ``max_factor_pct`` times the
      global peak amplitude (``1.0`` = time to the true peak; not from the PF's
      first birth)
    """
    active_seg = pf_segment_df[
        pf_segment_df["is_segment"] & (pf_segment_df["state"] == "active")
    ]
    if active_seg.empty:
        return pd.DataFrame(
            columns=[
                "cell_idx",
                "pf_idx",
                "birth_capture_idx",
                "death_capture_idx",
                "n_life_periods",
                "peak_life_period_idx",
                "length_s",
                "start_iteration_offset",
                "peak_amplitude",
                "tma_s",
                "one_segment",
                "drift_x",
                "drift_y",
                "robustness",
            ]
        )

    capture_to_global = training_capture_to_global_segment_map(pf_segment_df)

    def _metrics_from_pf_group(group: pd.DataFrame) -> pd.Series:
        group = group.sort_values(
            ["life_period_idx", "capture_idx"],
            kind="mergesort",
        )
        amps = group["amplitude"].to_numpy(dtype=np.float64)
        peak_row_idx = int(np.argmax(amps))
        peak_row = group.iloc[peak_row_idx]
        peak_life = int(peak_row["life_period_idx"])
        n_seg = len(group)
        drift_x = float(np.std(group["mu_x_cm"], ddof=0)) if n_seg > 1 else 0.0
        drift_y = float(np.std(group["mu_y_cm"], ddof=0)) if n_seg > 1 else 0.0

        period_rows = group[group["life_period_idx"] == peak_life].sort_values(
            "capture_idx",
            kind="mergesort",
        )
        period_amps = period_rows["amplitude"].to_numpy(dtype=np.float64)
        true_peak_amp = float(amps[peak_row_idx])
        if max_factor_pct >= 1.0:
            period_peak_idx = int(np.argmax(period_amps))
        else:
            threshold = max_factor_pct * true_peak_amp
            reached = period_amps >= threshold
            period_peak_idx = (
                int(np.argmax(reached))
                if np.any(reached)
                else int(np.argmax(period_amps))
            )

        birth_capture_idx = int(group["capture_idx"].iloc[0])
        start_iteration_offset = capture_to_global.get(birth_capture_idx)

        return pd.Series(
            {
                "birth_capture_idx": birth_capture_idx,
                "death_capture_idx": int(group["capture_idx"].iloc[-1]),
                "n_life_periods": int(group["life_period_idx"].nunique()),
                "peak_life_period_idx": peak_life,
                "length_s": float(n_seg * segment_duration_s),
                "start_iteration_offset": (
                    float(start_iteration_offset)
                    if start_iteration_offset is not None
                    else float("nan")
                ),
                "peak_amplitude": true_peak_amp,
                "tma_s": float(period_peak_idx * segment_duration_s),
                "one_segment": bool(period_peak_idx == 0),
                "drift_x": drift_x,
                "drift_y": drift_y,
                "robustness": (drift_x + drift_y) / 2.0,
            }
        )

    sorted_seg = active_seg.sort_values(
        ["cell_idx", "pf_idx", "life_period_idx", "capture_idx"],
        kind="mergesort",
    )
    out = (
        sorted_seg.groupby(["cell_idx", "pf_idx"], sort=False)
        .apply(_metrics_from_pf_group, include_groups=False)
        .reset_index()
    )
    return out.sort_values(["cell_idx", "pf_idx"], kind="mergesort").reset_index(drop=True)


def summarize_place_field_life_period_metrics(
    pf_segment_df: pd.DataFrame,
    *,
    segment_duration_s: float,
    max_factor_pct: float = 1.0,
) -> pd.DataFrame:
    """Per ``(cell_idx, pf_idx, life_period_idx)`` metrics (one row per life period)."""
    active_seg = pf_segment_df[
        pf_segment_df["is_segment"] & (pf_segment_df["state"] == "active")
    ]
    if active_seg.empty:
        return pd.DataFrame(
            columns=[
                "cell_idx",
                "pf_idx",
                "life_period_idx",
                "birth_capture_idx",
                "death_capture_idx",
                "length_s",
                "peak_amplitude",
                "tma_s",
                "one_segment",
                "drift_x",
                "drift_y",
                "robustness",
            ]
        )

    def _metrics_from_life_period(group: pd.DataFrame) -> pd.Series:
        amps = group["amplitude"].to_numpy(dtype=np.float64)
        local_peak = float(amps.max())
        if max_factor_pct >= 1.0:
            peak_idx = int(np.argmax(amps))
        else:
            threshold = max_factor_pct * local_peak
            reached = amps >= threshold
            peak_idx = (
                int(np.argmax(reached))
                if np.any(reached)
                else int(np.argmax(amps))
            )
        n_seg = len(group)
        drift_x = float(np.std(group["mu_x_cm"], ddof=0)) if n_seg > 1 else 0.0
        drift_y = float(np.std(group["mu_y_cm"], ddof=0)) if n_seg > 1 else 0.0
        return pd.Series(
            {
                "birth_capture_idx": int(group["capture_idx"].iloc[0]),
                "death_capture_idx": int(group["capture_idx"].iloc[-1]),
                "length_s": float(n_seg * segment_duration_s),
                "peak_amplitude": local_peak,
                "tma_s": float(peak_idx * segment_duration_s),
                "one_segment": bool(peak_idx == 0),
                "drift_x": drift_x,
                "drift_y": drift_y,
                "robustness": (drift_x + drift_y) / 2.0,
            }
        )

    sorted_seg = active_seg.sort_values(
        ["cell_idx", "pf_idx", "life_period_idx", "capture_idx"],
        kind="mergesort",
    )
    out = (
        sorted_seg.groupby(["cell_idx", "pf_idx", "life_period_idx"], sort=False)
        .apply(_metrics_from_life_period, include_groups=False)
        .reset_index()
    )
    return out.sort_values(
        ["cell_idx", "pf_idx", "life_period_idx"],
        kind="mergesort",
    ).reset_index(drop=True)


def attach_first_birth_formation_onset(
    plot_metrics: pd.DataFrame,
    global_pf_metrics: pd.DataFrame,
    *,
    segment_duration_s: float,
    pf_segment_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add first-birth ``start_iteration_offset`` and ``formation_onset_s``.

    When ``plot_metrics`` has one row per life period, only ``life_period_idx == 0``
    rows receive a finite onset; revive periods are left as NaN so downstream
    aggregations exclude them.
    """
    out = plot_metrics.copy()
    if global_pf_metrics.empty:
        out["start_iteration_offset"] = np.nan
        out["formation_onset_s"] = np.nan
        return out

    first_birth = global_pf_metrics[["cell_idx", "pf_idx"]].copy()
    if "start_iteration_offset" in global_pf_metrics.columns:
        first_birth["first_birth_start_iteration_offset"] = (
            global_pf_metrics["start_iteration_offset"].astype(float)
        )
    else:
        if pf_segment_df is None:
            raise ValueError(
                "pf_segment_df is required when global_pf_metrics lacks "
                "start_iteration_offset"
            )
        capture_to_global = training_capture_to_global_segment_map(pf_segment_df)
        first_birth["first_birth_start_iteration_offset"] = (
            global_pf_metrics["birth_capture_idx"]
            .map(capture_to_global)
            .astype(float)
        )

    out = out.drop(columns=["start_iteration_offset", "formation_onset_s"], errors="ignore")
    out = out.merge(first_birth, on=["cell_idx", "pf_idx"], how="left")

    if "life_period_idx" in out.columns:
        is_first_birth = out["life_period_idx"].astype(int).eq(0)
        out["start_iteration_offset"] = np.where(
            is_first_birth,
            out["first_birth_start_iteration_offset"],
            np.nan,
        )
    else:
        out["start_iteration_offset"] = out["first_birth_start_iteration_offset"]

    out["formation_onset_s"] = (
        out["start_iteration_offset"] * float(segment_duration_s)
    )
    return out.drop(columns=["first_birth_start_iteration_offset"])


def mean_pfs_per_neuron_stats(
    pf_metrics_df: pd.DataFrame,
    *,
    n_neurons: int,
) -> dict[str, float]:
    """Mean place fields per neuron ± 1.96 SEM (includes neurons with zero PFs)."""
    if n_neurons <= 0:
        raise ValueError("n_neurons must be positive")

    if pf_metrics_df.empty:
        counts = np.zeros(n_neurons, dtype=np.float64)
    else:
        counts_by_cell = (
            pf_metrics_df.groupby("cell_idx", sort=False)["pf_idx"]
            .nunique()
            .reindex(range(n_neurons), fill_value=0)
            .astype(np.float64)
        )
        counts = counts_by_cell.to_numpy()

    mean = float(counts.mean())
    sem = float(counts.std(ddof=1) / np.sqrt(counts.size)) if counts.size > 1 else 0.0
    return {
        "mean": mean,
        "sem": sem,
        "ci95": 1.96 * sem,
        "counts": counts,
    }


def _attach_capture_indices(
    pairs: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    last_epoch_only: bool,
) -> pd.DataFrame:
    """Add ``capture_from`` / ``capture_to`` columns via merge (not row-wise apply)."""
    sub = _apply_last_block_filter(master_df, last_epoch_only)
    meta_keys = capture_metadata_keys(sub)
    cap_df = sub.drop_duplicates(meta_keys)[meta_keys + ["capture_idx"]]
    merge_keys = timeline_keys(sub)
    out = pairs.merge(
        cap_df.rename(
            columns={"segment_id": "segment_from", "capture_idx": "capture_from"}
        ),
        on=merge_keys + ["segment_from"],
        how="left",
    )
    out = out.merge(
        cap_df.rename(columns={"segment_id": "segment_to", "capture_idx": "capture_to"}),
        on=merge_keys + ["segment_to"],
        how="left",
    )
    return out


def _build_pf_capture_lookups(
    pf_segment_df: pd.DataFrame,
) -> tuple[
    dict[tuple[int, int], frozenset[tuple[int, int]]],
    dict[tuple[int, int], int],
    dict[tuple[int, int, int, int], tuple[float, float]],
]:
    """
    One-pass indexes for PF events keyed by capture.

    Returns ``active_keys``, ``death_counts``, and ``positions`` where positions
    maps ``(cell_idx, capture_idx, pf_idx, life_period_idx)`` to ``(mu_x, mu_y)``.
    """
    active_keys: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    death_counts: dict[tuple[int, int], int] = defaultdict(int)
    positions: dict[tuple[int, int, int, int], tuple[float, float]] = {}

    if pf_segment_df.empty:
        return {}, {}, {}

    cell_arr = pf_segment_df["cell_idx"].to_numpy(dtype=np.int32)
    cap_arr = pf_segment_df["capture_idx"].to_numpy(dtype=np.int32)
    pf_arr = pf_segment_df["pf_idx"].to_numpy(dtype=np.int32)
    life_arr = pf_segment_df["life_period_idx"].to_numpy(dtype=np.int32)
    state_arr = pf_segment_df["state"].to_numpy(dtype=object)
    mu_x_arr = pf_segment_df["mu_x_cm"].to_numpy(dtype=np.float64)
    mu_y_arr = pf_segment_df["mu_y_cm"].to_numpy(dtype=np.float64)

    for i in range(len(pf_segment_df)):
        cell_idx = int(cell_arr[i])
        capture_idx = int(cap_arr[i])
        cap_key = (cell_idx, capture_idx)
        if state_arr[i] == "active":
            pf_key = (int(pf_arr[i]), int(life_arr[i]))
            active_keys[cap_key].add(pf_key)
            positions[(cell_idx, capture_idx, pf_key[0], pf_key[1])] = (
                float(mu_x_arr[i]),
                float(mu_y_arr[i]),
            )
        else:
            death_counts[cap_key] += 1

    frozen_active = {k: frozenset(v) for k, v in active_keys.items()}
    return frozen_active, dict(death_counts), positions


def _transition_timeline_from_events(events: pd.DataFrame) -> pd.DataFrame:
    event_keys = transition_event_keys(events)
    traj_keys = timeline_keys(events)
    timeline = (
        events.groupby(event_keys, sort=False)
        .agg(
            n_dead_to_alive=("n_dead_to_alive", "sum"),
            n_alive_to_dead=("n_alive_to_dead", "sum"),
            n_revives=("n_revives", "sum"),
        )
        .reset_index()
        .sort_values(event_keys, kind="mergesort")
        .reset_index(drop=True)
    )
    timeline["global_transition_idx"] = np.arange(len(timeline), dtype=int)
    timeline["is_traj_start"] = (
        timeline.groupby(traj_keys, sort=False).cumcount() == 0
    )
    timeline["segment_from"] = timeline["segment_from"].astype(int)
    timeline["segment_to"] = timeline["segment_to"].astype(int)
    return timeline


def pf_state_transition_counts(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    dead_threshold: float = 0.1,
    r2_threshold: float = 0.8,
    last_epoch_only: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    PF inactive→active and active→inactive counts per segment transition.

    Uses the same transition grid as :func:`alive_state_transition_counts`.
    Column names mirror cell-level transitions for plotting reuse; here they
    count PF appearances and disappearances. ``n_revives`` counts appearances
    where ``life_period_idx > 0`` (coordinate-based revival).
    """
    empty_cols = (
        timeline_keys_for(infer_experiment_name(master_df))
        + [
            "segment_from",
            "segment_to",
            "global_transition_idx",
            "is_traj_start",
            "n_dead_to_alive",
            "n_alive_to_dead",
            "n_revives",
        ]
    )
    if pf_segment_df.empty:
        return pd.DataFrame(columns=empty_cols)

    sub = _apply_last_block_filter(master_df, last_epoch_only).copy()
    sub = sub.assign(
        alive=is_alive(sub, dead_threshold=dead_threshold, r2_threshold=r2_threshold)
    )
    pairs = _build_transition_pairs(sub)
    if pairs.empty:
        return pd.DataFrame(columns=empty_cols)

    pairs = _attach_capture_indices(pairs, master_df, last_epoch_only=last_epoch_only)
    active_keys, death_counts, _ = _build_pf_capture_lookups(pf_segment_df)

    empty_active: frozenset[tuple[int, int]] = frozenset()
    n_pairs = len(pairs)
    n_dead_to_alive = np.zeros(n_pairs, dtype=np.float64)
    n_alive_to_dead = np.zeros(n_pairs, dtype=np.float64)
    n_revives = np.zeros(n_pairs, dtype=np.float64)

    pair_iter = pairs.itertuples(index=False)
    if show_progress:
        pair_iter = tqdm(
            pair_iter,
            total=n_pairs,
            desc="PF state transitions",
            leave=False,
        )

    for i, row in enumerate(pair_iter):
        cell_idx = int(row.cell_idx)
        cap_from = int(row.capture_from)
        cap_to = int(row.capture_to)
        active_from = active_keys.get((cell_idx, cap_from), empty_active)
        active_to = active_keys.get((cell_idx, cap_to), empty_active)
        new_active = active_to - active_from
        n_alive_to_dead[i] = float(death_counts.get((cell_idx, cap_to), 0))
        n_dead_to_alive[i] = float(len(new_active))
        n_revives[i] = float(sum(1 for _pf_idx, life_period_idx in new_active if life_period_idx > 0))

    events = pairs[transition_event_keys(pairs)].copy()
    events["n_dead_to_alive"] = n_dead_to_alive
    events["n_alive_to_dead"] = n_alive_to_dead
    events["n_revives"] = n_revives
    return _transition_timeline_from_events(events)


def pf_tracked_displacement_samples(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    gauss_threshold: float = 1.5,
    dead_threshold: float = 0.1,
    r2_threshold: float = 0.8,
    last_epoch_only: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Matched-PF displacement samples using tracked PF identity.

    For each alive→alive segment transition and each PF active in both
    consecutive captures (same ``pf_idx`` and ``life_period_idx``), records
    centre displacement in cm.
    """
    empty_cols = (
        timeline_keys_for(infer_experiment_name(master_df))
        + [
            "segment_from",
            "segment_to",
            "cell_idx",
            "pf_idx",
            "life_period_idx",
            "displacement_l2",
            "displacement_x",
            "displacement_y",
        ]
    )
    if pf_segment_df.empty:
        return pd.DataFrame(columns=empty_cols)

    pairs = _prepare_alive_transition_df(
        master_df,
        dead_threshold=dead_threshold,
        r2_threshold=r2_threshold,
        gauss_threshold=gauss_threshold,
        last_epoch_only=last_epoch_only,
    )
    if pairs.empty:
        return pd.DataFrame(columns=empty_cols)

    pairs = _attach_capture_indices(pairs, master_df, last_epoch_only=last_epoch_only)
    active_keys, _, positions = _build_pf_capture_lookups(pf_segment_df)

    empty_active: frozenset[tuple[int, int]] = frozenset()
    rows: list[dict[str, float | int]] = []
    pair_iter = pairs.itertuples(index=False)
    if show_progress:
        pair_iter = tqdm(
            pair_iter,
            total=len(pairs),
            desc="PF tracked displacements",
            leave=False,
        )

    for row in pair_iter:
        cell_idx = int(row.cell_idx)
        cap_from = int(row.capture_from)
        cap_to = int(row.capture_to)
        shared = active_keys.get((cell_idx, cap_from), empty_active) & active_keys.get(
            (cell_idx, cap_to), empty_active
        )
        for pf_idx, life_period_idx in shared:
            pos_from = positions[(cell_idx, cap_from, pf_idx, life_period_idx)]
            pos_to = positions[(cell_idx, cap_to, pf_idx, life_period_idx)]
            dx = pos_to[0] - pos_from[0]
            dy = pos_to[1] - pos_from[1]
            row_fields = _row_event_fields(row, transition_event_keys(pairs))
            rows.append(
                {
                    **row_fields,
                    "cell_idx": cell_idx,
                    "pf_idx": int(pf_idx),
                    "life_period_idx": int(life_period_idx),
                    "displacement_l2": float(np.hypot(dx, dy)),
                    "displacement_x": float(dx),
                    "displacement_y": float(dy),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    event_keys = transition_event_keys(out)
    key_to_idx = (
        out[event_keys]
        .drop_duplicates()
        .sort_values(event_keys, kind="mergesort")
        .reset_index(drop=True)
    )
    key_to_idx["global_transition_idx"] = np.arange(len(key_to_idx), dtype=int)
    out = out.merge(key_to_idx, on=event_keys, how="left")
    return out


def build_capture_to_global_segment_map(
    master_df: pd.DataFrame,
    grad_ts: pd.DataFrame,
    *,
    experiment_name: str | None = None,
) -> pd.Series:
    """Map ``capture_idx`` to ``global_segment_idx`` for training segments only."""
    exp = experiment_name or infer_experiment_name(master_df)
    merge_keys = timeline_keys_for(exp) + ["segment_id"]
    seg_master = (
        master_df.loc[master_df["segment_id"] >= 0]
        .drop_duplicates(capture_dedupe_keys(master_df))
        .loc[:, ["capture_idx"] + merge_keys]
    )
    seg_grad = grad_ts.loc[:, ["global_segment_idx"] + merge_keys]
    merged = seg_master.merge(
        seg_grad,
        on=merge_keys,
        how="inner",
        validate="many_to_one",
    )
    dup = merged["capture_idx"].duplicated(keep=False)
    if dup.any():
        raise ValueError(
            "Multiple global segments map to the same capture_idx: "
            f"{merged.loc[dup, 'capture_idx'].unique().tolist()}"
        )
    return merged.set_index("capture_idx")["global_segment_idx"].sort_index()


def _init_pf_glocal_worker(
    g_local_bundle: GLocalMseSegmentStatsBundle,
    capture_to_global: dict[int, int],
    pf_segment_df: pd.DataFrame,
    cell_segment_caps: dict[int, np.ndarray],
) -> None:
    global _PF_GLOCAL_WORKER_BUNDLE, _PF_GLOCAL_WORKER_CAPTURE_TO_GLOBAL
    global _PF_GLOCAL_WORKER_PF_SEGMENT_DF, _PF_GLOCAL_WORKER_CELL_SEGMENT_CAPS
    _PF_GLOCAL_WORKER_BUNDLE = g_local_bundle
    _PF_GLOCAL_WORKER_CAPTURE_TO_GLOBAL = capture_to_global
    _PF_GLOCAL_WORKER_PF_SEGMENT_DF = pf_segment_df
    _PF_GLOCAL_WORKER_CELL_SEGMENT_CAPS = cell_segment_caps


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = numerator / denominator
    invalid = ~np.isfinite(denominator) | (np.abs(denominator) < 1e-12) | ~np.isfinite(numerator)
    out = out.astype(np.float64, copy=False)
    out[invalid] = np.nan
    return out


def _unit_segment_stats_for_variant(
    g_local_bundle: GLocalMseSegmentStatsBundle,
    gidxs: np.ndarray,
    cell_idx: int,
    variant: GLocalVariant,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-segment mean/max/std for one hidden unit under the requested variant."""
    if variant == "signed":
        return (
            g_local_bundle.unit_signed_mean[gidxs, cell_idx],
            g_local_bundle.unit_signed_max[gidxs, cell_idx],
            g_local_bundle.unit_signed_std[gidxs, cell_idx],
        )
    if variant == "abs":
        return (
            g_local_bundle.unit_abs_mean[gidxs, cell_idx],
            g_local_bundle.unit_abs_max[gidxs, cell_idx],
            g_local_bundle.unit_abs_std[gidxs, cell_idx],
        )
    if variant == "normalized":
        denom = g_local_bundle.network_excl_signed_mean[gidxs, cell_idx]
        return (
            _safe_ratio(g_local_bundle.unit_signed_mean[gidxs, cell_idx], denom),
            _safe_ratio(g_local_bundle.unit_signed_max[gidxs, cell_idx], denom),
            _safe_ratio(g_local_bundle.unit_signed_std[gidxs, cell_idx], denom),
        )
    denom = g_local_bundle.network_excl_abs_mean[gidxs, cell_idx]
    return (
        _safe_ratio(g_local_bundle.unit_abs_mean[gidxs, cell_idx], denom),
        _safe_ratio(g_local_bundle.unit_abs_max[gidxs, cell_idx], denom),
        _safe_ratio(g_local_bundle.unit_abs_std[gidxs, cell_idx], denom),
    )


def _aggregate_segment_g_local_stats(
    capture_indices: list[int] | np.ndarray,
    *,
    capture_to_global: dict[int, int] | pd.Series,
    g_local_bundle: GLocalMseSegmentStatsBundle,
    cell_idx: int,
    variant: GLocalVariant,
    stat_prefix: str = "g_local_mse",
) -> dict[str, float]:
    """Return mean/max/std of per-segment stats, averaged over the given captures."""
    nan_stats = {
        f"{stat_prefix}_mean": float("nan"),
        f"{stat_prefix}_max": float("nan"),
        f"{stat_prefix}_std": float("nan"),
    }
    if len(capture_indices) == 0:
        return nan_stats

    if isinstance(capture_to_global, pd.Series):
        capture_to_global_map = {
            int(cap): int(gidx)
            for cap, gidx in capture_to_global.items()
            if np.isfinite(gidx)
        }
    else:
        capture_to_global_map = capture_to_global

    gidxs = np.asarray(
        [
            capture_to_global_map[int(cap)]
            for cap in capture_indices
            if int(cap) in capture_to_global_map
        ],
        dtype=np.intp,
    )
    if gidxs.size == 0:
        return nan_stats

    seg_mean, seg_max, seg_std = _unit_segment_stats_for_variant(
        g_local_bundle,
        gidxs,
        cell_idx,
        variant,
    )
    return {
        f"{stat_prefix}_mean": float(np.nanmean(seg_mean)),
        f"{stat_prefix}_max": float(np.nanmean(seg_max)),
        f"{stat_prefix}_std": float(np.nanmean(seg_std)),
    }


def _compute_pf_glocal_rows_for_keys(
    pf_keys: list[tuple[int, int]],
    *,
    pf_segment_df: pd.DataFrame,
    cell_segment_caps: dict[int, np.ndarray],
    g_local_bundle: GLocalMseSegmentStatsBundle,
    capture_to_global: dict[int, int],
    group_names: tuple[str, ...] = ("birth", "peak", "other_revives", "absent"),
    variants: tuple[GLocalVariant, ...] = PF_GLOCAL_VARIANTS,
) -> dict[str, list[dict[str, int | float | str]]]:
    rows_by_variant: dict[str, list[dict[str, int | float | str]]] = {
        variant: [] for variant in variants
    }
    for cell_idx, pf_idx in pf_keys:
        groups = classify_pf_segment_groups(
            pf_segment_df,
            cell_idx=int(cell_idx),
            pf_idx=int(pf_idx),
            cell_segment_capture_indices=cell_segment_caps[int(cell_idx)],
        )
        for group_name in group_names:
            caps = groups[group_name]
            for variant in variants:
                stats = _aggregate_segment_g_local_stats(
                    caps,
                    capture_to_global=capture_to_global,
                    g_local_bundle=g_local_bundle,
                    cell_idx=int(cell_idx),
                    variant=variant,
                )
                rows_by_variant[variant].append(
                    {
                        "cell_idx": int(cell_idx),
                        "pf_idx": int(pf_idx),
                        "segment_group": group_name,
                        "n_segments": len(caps),
                        **stats,
                    }
                )
    return rows_by_variant


def _compute_pf_glocal_worker(
    pf_keys: list[tuple[int, int]],
) -> dict[str, list[dict[str, int | float | str]]]:
    if (
        _PF_GLOCAL_WORKER_BUNDLE is None
        or _PF_GLOCAL_WORKER_CAPTURE_TO_GLOBAL is None
        or _PF_GLOCAL_WORKER_PF_SEGMENT_DF is None
        or _PF_GLOCAL_WORKER_CELL_SEGMENT_CAPS is None
    ):
        raise RuntimeError("PF g_local worker not initialized")
    return _compute_pf_glocal_rows_for_keys(
        pf_keys,
        pf_segment_df=_PF_GLOCAL_WORKER_PF_SEGMENT_DF,
        cell_segment_caps=_PF_GLOCAL_WORKER_CELL_SEGMENT_CAPS,
        g_local_bundle=_PF_GLOCAL_WORKER_BUNDLE,
        capture_to_global=_PF_GLOCAL_WORKER_CAPTURE_TO_GLOBAL,
    )


def classify_pf_segment_groups(
    pf_segment_df: pd.DataFrame,
    *,
    cell_idx: int,
    pf_idx: int,
    cell_segment_capture_indices: np.ndarray | list[int],
) -> dict[str, list[int]]:
    """
    Classify training segments for one PF into four analysis groups.

    Groups (capture indices, training segments only):
    - ``birth`` — first active training segment in life period 0 (empty if birth
      was at ``pre`` only)
    - ``peak`` — segment with global peak amplitude
    - ``other_revives`` — first active segment of each revival (life period > 0),
      excluding the peak segment if it coincides with a revival
    - ``absent`` — training segments for the cell where the PF is not active
    """
    pf_rows = pf_segment_df[
        (pf_segment_df["cell_idx"] == cell_idx) & (pf_segment_df["pf_idx"] == pf_idx)
    ]
    active = pf_rows[pf_rows["is_segment"] & (pf_rows["state"] == "active")].sort_values(
        ["life_period_idx", "capture_idx"],
        kind="mergesort",
    )
    if active.empty:
        return {"birth": [], "peak": [], "other_revives": [], "absent": []}

    birth_rows = active.loc[active["life_period_idx"] == 0]
    birth_caps = [int(birth_rows.iloc[0]["capture_idx"])] if not birth_rows.empty else []

    peak_cap = int(active.iloc[int(active["amplitude"].to_numpy().argmax())]["capture_idx"])

    other_revives: list[int] = []
    for life_period_idx, period_rows in active.groupby("life_period_idx", sort=False):
        if int(life_period_idx) == 0:
            continue
        revive_cap = int(period_rows["capture_idx"].iloc[0])
        if revive_cap != peak_cap:
            other_revives.append(revive_cap)

    active_caps = set(active["capture_idx"].astype(int).tolist())
    absent_caps = [
        int(cap)
        for cap in cell_segment_capture_indices
        if int(cap) not in active_caps
    ]

    return {
        "birth": birth_caps,
        "peak": [peak_cap],
        "other_revives": other_revives,
        "absent": absent_caps,
    }


def _absent_segments_before_period_start(
    period_start_cap: int,
    *,
    absent_caps: set[int],
    cell_segment_capture_indices: np.ndarray | list[int],
    max_count: int,
) -> list[int]:
    """Up to ``max_count`` PF-absent training segments immediately before a period start."""
    caps = np.asarray(cell_segment_capture_indices, dtype=np.int32)
    pos = int(np.searchsorted(caps, int(period_start_cap)))
    if pos >= caps.size or int(caps[pos]) != int(period_start_cap):
        return []

    collected: list[int] = []
    for i in range(pos - 1, -1, -1):
        cap = int(caps[i])
        if cap not in absent_caps:
            break
        collected.append(cap)
        if len(collected) >= max_count:
            break
    return sorted(collected)


def _absent_segments_after_anchor(
    anchor_cap: int,
    *,
    absent_caps: set[int],
    cell_segment_capture_indices: np.ndarray | list[int],
    max_count: int,
) -> list[int]:
    """Up to ``max_count`` PF-absent training segments immediately after an anchor."""
    caps = np.asarray(cell_segment_capture_indices, dtype=np.int32)
    pos = int(np.searchsorted(caps, int(anchor_cap)))
    if pos >= caps.size or int(caps[pos]) != int(anchor_cap):
        return []

    collected: list[int] = []
    for i in range(pos + 1, caps.size):
        cap = int(caps[i])
        if cap not in absent_caps:
            break
        collected.append(cap)
        if len(collected) >= max_count:
            break
    return collected


def _training_segments_after_anchor(
    anchor_cap: int,
    cell_segment_capture_indices: np.ndarray | list[int],
    *,
    max_count: int,
) -> list[int]:
    """Up to ``max_count`` training segments immediately after an anchor (any PF state)."""
    caps = np.asarray(cell_segment_capture_indices, dtype=np.int32)
    pos = int(np.searchsorted(caps, int(anchor_cap)))
    if pos >= caps.size or int(caps[pos]) != int(anchor_cap):
        return []
    end = min(caps.size, pos + 1 + max_count)
    return [int(cap) for cap in caps[pos + 1 : end]]


def _training_segments_before_anchor(
    anchor_cap: int,
    cell_segment_capture_indices: np.ndarray | list[int],
    *,
    max_count: int,
) -> list[int]:
    """Up to ``max_count`` training segments immediately before an anchor (any PF state)."""
    caps = np.asarray(cell_segment_capture_indices, dtype=np.int32)
    pos = int(np.searchsorted(caps, int(anchor_cap)))
    if pos >= caps.size or int(caps[pos]) != int(anchor_cap):
        return []
    start = max(0, pos - max_count)
    return [int(cap) for cap in caps[start:pos]]


def _pf_lifecycle_vicinity_window_capture_indices(
    anchor_cap: int,
    *,
    absent_caps: set[int],
    cell_segment_capture_indices: np.ndarray | list[int],
    max_count: int,
) -> tuple[list[int], list[int]]:
    """Lifecycle plot windows: PF-absent pre, any-state post."""
    pre = _absent_segments_before_period_start(
        int(anchor_cap),
        absent_caps=absent_caps,
        cell_segment_capture_indices=cell_segment_capture_indices,
        max_count=max_count,
    )
    post = _training_segments_after_anchor(
        int(anchor_cap),
        cell_segment_capture_indices,
        max_count=max_count,
    )
    return pre, post


def pf_lifecycle_vicinity_window_capture_indices_for_event(
    anchor_cap: int,
    *,
    event_name: str,
    absent_caps: set[int],
    cell_segment_capture_indices: np.ndarray | list[int],
    max_count: int,
) -> tuple[list[int], list[int]]:
    """
    Lifecycle vicinity windows around an anchor capture.

    PF lifecycle events (``birth``, ``first_revival``) use PF-absent pre segments
    and any-state post segments. ``random`` / ``dead_random`` use consecutive
    training segments on both sides; ``dead_random`` anchors are PF-absent segments.
    """
    if event_name in PF_LIFECYCLE_EFF_LR_MULTI_WINDOW_EVENTS:
        return (
            _training_segments_before_anchor(
                int(anchor_cap),
                cell_segment_capture_indices,
                max_count=max_count,
            ),
            _training_segments_after_anchor(
                int(anchor_cap),
                cell_segment_capture_indices,
                max_count=max_count,
            ),
        )
    return _pf_lifecycle_vicinity_window_capture_indices(
        int(anchor_cap),
        absent_caps=absent_caps,
        cell_segment_capture_indices=cell_segment_capture_indices,
        max_count=max_count,
    )


def _pf_lifecycle_anchor_extended_window_bounds(
    anchor_pos: int,
    *,
    pre_count: int,
    post_count: int,
) -> tuple[int, int]:
    """Inclusive capture-index bounds for the extended residual window."""
    return int(anchor_pos) - int(pre_count), int(anchor_pos) + int(post_count)


def _pf_lifecycle_capture_spans_overlap(
    span_a: tuple[int, int],
    span_b: tuple[int, int],
) -> bool:
    return not (span_a[1] < span_b[0] or span_b[1] < span_a[0])


def pf_lifecycle_non_overlapping_window_capture_indices(
    cell_segment_capture_indices: np.ndarray | list[int],
    *,
    n_windows: int,
    rng: np.random.Generator | None = None,
    min_pre: int,
    min_post: int,
    anchor_positions: set[int] | None = None,
) -> list[int]:
    """
    Pick up to ``n_windows`` anchors whose extended windows do not overlap.

    Overlap is defined on the extended residual window:
    ``min_pre`` segments before the anchor through ``min_post`` after it.
    """
    caps = np.asarray(cell_segment_capture_indices, dtype=np.int32)
    if caps.size == 0 or int(n_windows) <= 0:
        return []
    if rng is None:
        rng = np.random.default_rng()

    valid_positions = [
        pos
        for pos in range(caps.size)
        if pos >= int(min_pre)
        and pos + int(min_post) < caps.size
        and (anchor_positions is None or pos in anchor_positions)
    ]
    if not valid_positions:
        return []

    candidate_order = np.asarray(valid_positions, dtype=np.int32)
    rng.shuffle(candidate_order)

    chosen_positions: list[int] = []
    chosen_spans: list[tuple[int, int]] = []
    for pos in candidate_order:
        pos = int(pos)
        span = _pf_lifecycle_anchor_extended_window_bounds(
            pos,
            pre_count=min_pre,
            post_count=min_post,
        )
        if any(
            _pf_lifecycle_capture_spans_overlap(span, other_span)
            for other_span in chosen_spans
        ):
            continue
        chosen_positions.append(pos)
        chosen_spans.append(span)
        if len(chosen_positions) >= int(n_windows):
            break

    return [int(caps[pos]) for pos in chosen_positions]


def pf_lifecycle_random_event_capture_indices(
    cell_segment_capture_indices: np.ndarray | list[int],
    *,
    n_windows: int,
    rng: np.random.Generator | None = None,
    min_pre: int,
    min_post: int,
) -> list[int]:
    """Pick non-overlapping random training-segment anchors."""
    return pf_lifecycle_non_overlapping_window_capture_indices(
        cell_segment_capture_indices,
        n_windows=n_windows,
        rng=rng,
        min_pre=min_pre,
        min_post=min_post,
    )


def pf_lifecycle_dead_random_event_capture_indices(
    cell_segment_capture_indices: np.ndarray | list[int],
    *,
    absent_caps: set[int],
    n_windows: int,
    rng: np.random.Generator | None = None,
    min_pre: int,
    min_post: int,
) -> list[int]:
    """Pick non-overlapping anchors on PF-absent (dead) training segments."""
    caps = np.asarray(cell_segment_capture_indices, dtype=np.int32)
    dead_positions = {
        pos for pos in range(caps.size) if int(caps[pos]) in absent_caps
    }
    return pf_lifecycle_non_overlapping_window_capture_indices(
        cell_segment_capture_indices,
        n_windows=n_windows,
        rng=rng,
        min_pre=min_pre,
        min_post=min_post,
        anchor_positions=dead_positions,
    )


def pf_lifecycle_random_event_capture_index(
    cell_segment_capture_indices: np.ndarray | list[int],
    *,
    rng: np.random.Generator | None = None,
    min_pre: int,
    min_post: int,
) -> int | None:
    """Pick one random training-segment anchor with enough pre/post context."""
    caps = pf_lifecycle_random_event_capture_indices(
        cell_segment_capture_indices,
        n_windows=1,
        rng=rng,
        min_pre=min_pre,
        min_post=min_post,
    )
    return caps[0] if caps else None


def _pf_absent_vicinity_window_capture_indices(
    anchor_cap: int,
    *,
    absent_caps: set[int],
    cell_segment_capture_indices: np.ndarray | list[int],
    max_count: int,
) -> tuple[list[int], list[int]]:
    """PF-absent pre/post windows around one anchor (same logic as pre-vicinity group)."""
    pre = _absent_segments_before_period_start(
        int(anchor_cap),
        absent_caps=absent_caps,
        cell_segment_capture_indices=cell_segment_capture_indices,
        max_count=max_count,
    )
    post = _absent_segments_after_anchor(
        int(anchor_cap),
        absent_caps=absent_caps,
        cell_segment_capture_indices=cell_segment_capture_indices,
        max_count=max_count,
    )
    return pre, post


def classify_pf_pre_vicinity_segments(
    pf_segment_df: pd.DataFrame,
    *,
    cell_idx: int,
    pf_idx: int,
    cell_segment_capture_indices: np.ndarray | list[int],
    window_size: int,
) -> list[int]:
    """
    PF-absent training segments in the pre-vicinity window before each life period.

    For every life period (birth and each revival), take up to
    ``VICINITY_SEGMENTS_SIZE`` consecutive PF-absent training segments
    immediately before that period's first active segment.
    """
    groups = classify_pf_segment_groups(
        pf_segment_df,
        cell_idx=cell_idx,
        pf_idx=pf_idx,
        cell_segment_capture_indices=cell_segment_capture_indices,
    )
    absent_caps = set(groups["absent"])
    pf_rows = pf_segment_df[
        (pf_segment_df["cell_idx"] == cell_idx) & (pf_segment_df["pf_idx"] == pf_idx)
    ]
    active = pf_rows[pf_rows["is_segment"] & (pf_rows["state"] == "active")].sort_values(
        ["life_period_idx", "capture_idx"],
        kind="mergesort",
    )
    if active.empty:
        return []

    pre_vicinity: set[int] = set()
    for _life_period_idx, period_rows in active.groupby("life_period_idx", sort=False):
        period_start_cap = int(period_rows.iloc[0]["capture_idx"])
        pre_vicinity.update(
            _absent_segments_before_period_start(
                period_start_cap,
                absent_caps=absent_caps,
                cell_segment_capture_indices=cell_segment_capture_indices,
                max_count=int(window_size),
            )
        )
    return sorted(pre_vicinity)


def pf_lifecycle_event_capture_indices(
    pf_segment_df: pd.DataFrame,
    *,
    cell_idx: int,
    pf_idx: int,
    cell_segment_capture_indices: np.ndarray | list[int],
    window_size: int,
    n_random_windows: int,
    rng: np.random.Generator | None = None,
    random_seed: int | None = 42,
) -> dict[str, int | list[int] | None]:
    """Capture indices for PF birth / first-revival / random anchor milestones."""
    caps = np.asarray(cell_segment_capture_indices, dtype=np.int32)
    _rng = rng if rng is not None else np.random.default_rng(random_seed)

    groups = classify_pf_segment_groups(
        pf_segment_df,
        cell_idx=cell_idx,
        pf_idx=pf_idx,
        cell_segment_capture_indices=cell_segment_capture_indices,
    )
    absent_caps = set(groups["absent"])
    resid = _resid_window_size(window_size)
    random_caps = pf_lifecycle_random_event_capture_indices(
        caps,
        n_windows=int(n_random_windows),
        rng=_rng,
        min_pre=resid,
        min_post=resid,
    )
    dead_random_caps = pf_lifecycle_dead_random_event_capture_indices(
        caps,
        absent_caps=absent_caps,
        n_windows=int(n_random_windows),
        rng=_rng,
        min_pre=resid,
        min_post=resid,
    )
    random_cap = random_caps[0] if random_caps else None
    dead_random_cap = dead_random_caps[0] if dead_random_caps else None

    birth_caps = groups["birth"]
    if not birth_caps:
        return {
            "pre_birth": None,
            "birth": None,
            "post_birth_absent": None,
            "first_revival": None,
            "random": random_cap,
            "random_windows": random_caps,
            "dead_random": dead_random_cap,
            "dead_random_windows": dead_random_caps,
        }

    birth_cap = int(birth_caps[0])
    caps = np.asarray(cell_segment_capture_indices, dtype=np.int32)
    birth_pos = int(np.searchsorted(caps, birth_cap))
    pre_birth = int(caps[birth_pos - 1]) if birth_pos > 0 else None

    pf_rows = pf_segment_df[
        (pf_segment_df["cell_idx"] == cell_idx) & (pf_segment_df["pf_idx"] == pf_idx)
    ]
    active_caps = set(
        pf_rows.loc[
            pf_rows["is_segment"] & (pf_rows["state"] == "active"),
            "capture_idx",
        ]
        .astype(int)
        .tolist()
    )
    post_birth_absent: int | None = None
    if birth_pos + 1 < caps.size:
        for cap in caps[birth_pos + 1 :]:
            if int(cap) not in active_caps:
                post_birth_absent = int(cap)
                break

    revival_rows = pf_rows.loc[
        pf_rows["is_segment"]
        & (pf_rows["state"] == "active")
        & (pf_rows["life_period_idx"] > 0)
    ].sort_values(["life_period_idx", "capture_idx"], kind="mergesort")
    first_revival = (
        int(revival_rows.iloc[0]["capture_idx"]) if not revival_rows.empty else None
    )

    return {
        "pre_birth": pre_birth,
        "birth": birth_cap,
        "post_birth_absent": post_birth_absent,
        "first_revival": first_revival,
        "random": random_cap,
        "random_windows": random_caps,
        "dead_random": dead_random_cap,
        "dead_random_windows": dead_random_caps,
    }


def select_pf_with_birth_and_revive(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    window_size: int,
    n_random_windows: int,
    min_pre_absent: int = 2,
) -> tuple[int, int]:
    """Pick one PF with training birth, a revival, and enough pre-event absent segments."""
    cell_segment_caps = (
        master_df.loc[master_df["segment_id"] >= 0]
        .drop_duplicates(["cell_idx", "capture_idx"])
        .sort_values(["cell_idx", "capture_idx"], kind="mergesort")
        .groupby("cell_idx", sort=False)["capture_idx"]
        .apply(lambda s: s.to_numpy(dtype=np.int32))
        .to_dict()
    )

    active_seg = pf_segment_df.loc[
        pf_segment_df["is_segment"] & (pf_segment_df["state"] == "active")
    ]
    if active_seg.empty:
        raise ValueError("No place field with both a training birth and a revival")

    pf_max_life = (
        active_seg.groupby(["cell_idx", "pf_idx"], sort=False)["life_period_idx"]
        .max()
        .reset_index(name="max_life_period_idx")
    )
    candidates = pf_max_life.loc[
        pf_max_life["max_life_period_idx"] > 0, ["cell_idx", "pf_idx"]
    ].sort_values(["cell_idx", "pf_idx"], kind="mergesort")

    for cell_idx, pf_idx in candidates.itertuples(index=False, name=None):
        cell_caps = cell_segment_caps.get(int(cell_idx))
        if cell_caps is None or cell_caps.size == 0:
            continue
        events = pf_lifecycle_event_capture_indices(
            pf_segment_df,
            cell_idx=int(cell_idx),
            pf_idx=int(pf_idx),
            cell_segment_capture_indices=cell_caps,
            window_size=int(window_size),
            n_random_windows=int(n_random_windows),
        )
        if events["birth"] is None or events["first_revival"] is None:
            continue
        absent_caps = set(
            classify_pf_segment_groups(
                pf_segment_df,
                cell_idx=int(cell_idx),
                pf_idx=int(pf_idx),
                cell_segment_capture_indices=cell_caps,
            )["absent"]
        )
        birth_pre, _ = _pf_lifecycle_vicinity_window_capture_indices(
            int(events["birth"]),
            absent_caps=absent_caps,
            cell_segment_capture_indices=cell_caps,
            max_count=int(window_size),
        )
        revival_pre, _ = _pf_lifecycle_vicinity_window_capture_indices(
            int(events["first_revival"]),
            absent_caps=absent_caps,
            cell_segment_capture_indices=cell_caps,
            max_count=int(window_size),
        )
        if (
            len(birth_pre) >= int(min_pre_absent)
            and len(revival_pre) >= int(min_pre_absent)
        ):
            return int(cell_idx), int(pf_idx)
    raise ValueError(
        "No place field with training birth, a revival, and "
        f">={int(min_pre_absent)} PF-absent pre-window segments at both events"
    )


def select_pfs_with_birth_and_revive(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    window_size: int,
    n_random_windows: int,
    n: int = 5,
    min_pre_absent: int = 2,
    distinct_cells: bool = True,
) -> list[tuple[int, int]]:
    """Pick up to ``n`` PFs with training birth, a revival, and pre-event absent windows."""
    cell_segment_caps = (
        master_df.loc[master_df["segment_id"] >= 0]
        .drop_duplicates(["cell_idx", "capture_idx"])
        .sort_values(["cell_idx", "capture_idx"], kind="mergesort")
        .groupby("cell_idx", sort=False)["capture_idx"]
        .apply(lambda s: s.to_numpy(dtype=np.int32))
        .to_dict()
    )

    active_seg = pf_segment_df.loc[
        pf_segment_df["is_segment"] & (pf_segment_df["state"] == "active")
    ]
    if active_seg.empty:
        raise ValueError("No place field with both a training birth and a revival")

    pf_max_life = (
        active_seg.groupby(["cell_idx", "pf_idx"], sort=False)["life_period_idx"]
        .max()
        .reset_index(name="max_life_period_idx")
    )
    candidates = pf_max_life.loc[
        pf_max_life["max_life_period_idx"] > 0, ["cell_idx", "pf_idx"]
    ].sort_values(["cell_idx", "pf_idx"], kind="mergesort")

    selected: list[tuple[int, int]] = []
    seen_cells: set[int] = set()
    for cell_idx, pf_idx in candidates.itertuples(index=False, name=None):
        cell_idx = int(cell_idx)
        pf_idx = int(pf_idx)
        if distinct_cells and cell_idx in seen_cells:
            continue
        cell_caps = cell_segment_caps.get(cell_idx)
        if cell_caps is None or cell_caps.size == 0:
            continue
        events = pf_lifecycle_event_capture_indices(
            pf_segment_df,
            cell_idx=cell_idx,
            pf_idx=pf_idx,
            cell_segment_capture_indices=cell_caps,
        )
        if events["birth"] is None or events["first_revival"] is None:
            continue
        absent_caps = set(
            classify_pf_segment_groups(
                pf_segment_df,
                cell_idx=cell_idx,
                pf_idx=pf_idx,
                cell_segment_capture_indices=cell_caps,
            )["absent"]
        )
        birth_pre, _ = _pf_lifecycle_vicinity_window_capture_indices(
            int(events["birth"]),
            absent_caps=absent_caps,
            cell_segment_capture_indices=cell_caps,
            max_count=int(window_size),
        )
        revival_pre, _ = _pf_lifecycle_vicinity_window_capture_indices(
            int(events["first_revival"]),
            absent_caps=absent_caps,
            cell_segment_capture_indices=cell_caps,
            max_count=int(window_size),
        )
        if (
            len(birth_pre) >= int(min_pre_absent)
            and len(revival_pre) >= int(min_pre_absent)
        ):
            selected.append((cell_idx, pf_idx))
            if distinct_cells:
                seen_cells.add(cell_idx)
            if len(selected) >= int(n):
                break

    if len(selected) < int(n):
        raise ValueError(
            f"Found only {len(selected)} qualifying place fields; requested {int(n)}"
        )
    return selected


def _iter_birth_revive_qualified_pfs(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    window_size: int,
    n_random_windows: int,
    min_pre_absent: int = 2,
    distinct_cells: bool = False,
) -> list[tuple[int, int]]:
    """All PFs with training birth, a revival, and enough pre-event absent segments."""
    cell_segment_caps = (
        master_df.loc[master_df["segment_id"] >= 0]
        .drop_duplicates(["cell_idx", "capture_idx"])
        .sort_values(["cell_idx", "capture_idx"], kind="mergesort")
        .groupby("cell_idx", sort=False)["capture_idx"]
        .apply(lambda s: s.to_numpy(dtype=np.int32))
        .to_dict()
    )

    active_seg = pf_segment_df.loc[
        pf_segment_df["is_segment"] & (pf_segment_df["state"] == "active")
    ]
    if active_seg.empty:
        return []

    pf_max_life = (
        active_seg.groupby(["cell_idx", "pf_idx"], sort=False)["life_period_idx"]
        .max()
        .reset_index(name="max_life_period_idx")
    )
    candidates = pf_max_life.loc[
        pf_max_life["max_life_period_idx"] > 0, ["cell_idx", "pf_idx"]
    ].sort_values(["cell_idx", "pf_idx"], kind="mergesort")

    qualified: list[tuple[int, int]] = []
    seen_cells: set[int] = set()
    for cell_idx, pf_idx in candidates.itertuples(index=False, name=None):
        cell_idx = int(cell_idx)
        pf_idx = int(pf_idx)
        if distinct_cells and cell_idx in seen_cells:
            continue
        cell_caps = cell_segment_caps.get(cell_idx)
        if cell_caps is None or cell_caps.size == 0:
            continue
        events = pf_lifecycle_event_capture_indices(
            pf_segment_df,
            cell_idx=cell_idx,
            pf_idx=pf_idx,
            cell_segment_capture_indices=cell_caps,
            window_size=int(window_size),
            n_random_windows=int(n_random_windows),
        )
        if events["birth"] is None or events["first_revival"] is None:
            continue
        absent_caps = set(
            classify_pf_segment_groups(
                pf_segment_df,
                cell_idx=cell_idx,
                pf_idx=pf_idx,
                cell_segment_capture_indices=cell_caps,
            )["absent"]
        )
        birth_pre, _ = _pf_lifecycle_vicinity_window_capture_indices(
            int(events["birth"]),
            absent_caps=absent_caps,
            cell_segment_capture_indices=cell_caps,
            max_count=int(window_size),
        )
        revival_pre, _ = _pf_lifecycle_vicinity_window_capture_indices(
            int(events["first_revival"]),
            absent_caps=absent_caps,
            cell_segment_capture_indices=cell_caps,
            max_count=int(window_size),
        )
        if (
            len(birth_pre) >= int(min_pre_absent)
            and len(revival_pre) >= int(min_pre_absent)
        ):
            qualified.append((cell_idx, pf_idx))
            if distinct_cells:
                seen_cells.add(cell_idx)
    return qualified


def select_pfs_random_from_birth_revive_qualified(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    window_size: int,
    n_random_windows: int,
    n: int,
    min_pre_absent: int = 2,
    distinct_cells: bool = True,
    random_seed: int | None = 42,
) -> list[tuple[int, int]]:
    """Randomly sample ``n`` birth+revive-qualified place fields."""
    qualified = _iter_birth_revive_qualified_pfs(
        pf_segment_df,
        master_df,
        window_size=int(window_size),
        n_random_windows=int(n_random_windows),
        min_pre_absent=int(min_pre_absent),
        distinct_cells=bool(distinct_cells),
    )
    if len(qualified) < int(n):
        raise ValueError(
            f"Found only {len(qualified)} qualifying place fields; requested {int(n)}"
        )
    rng = np.random.default_rng(random_seed)
    idx = rng.choice(len(qualified), size=int(n), replace=False)
    return [qualified[int(i)] for i in np.sort(idx)]


PF_LIFECYCLE_EFF_LR_EVENTS = ("birth", "first_revival", "random", "dead_random")
PF_LIFECYCLE_EFF_LR_MULTI_WINDOW_EVENTS = ("random", "dead_random")
PF_LIFECYCLE_EFF_LR_EVENT_LABELS: dict[str, str] = {
    "birth": "birth",
    "first_revival": "first revival",
    "random": "random",
    "dead_random": "dead random",
}
PF_LIFECYCLE_EFF_LR_BASELINES = ("pre", "post")
PF_LIFECYCLE_EFF_LR_BASELINE_LABELS: dict[str, str] = {
    "pre": "vs pre",
    "post": "vs post",
}
PF_LIFECYCLE_EFF_LR_OUTLIER_METHODS = (
    "raw_zscore",
    "raw_robustz",
    "resid_zscore",
    "resid_robustz",
)
OutlierMethod = Literal[
    "raw_zscore",
    "raw_robustz",
    "resid_zscore",
    "resid_robustz",
]
PF_LIFECYCLE_EFF_LR_OUTLIER_METHOD_LABELS: dict[str, str] = {
    "raw_zscore": "raw z-score",
    "raw_robustz": "raw robust z",
    "resid_zscore": "residual z-score",
    "resid_robustz": "residual robust z",
}
PF_LIFECYCLE_EFF_LR_ZSCORE_OUTLIER_THRESHOLD = 3.0
PF_LIFECYCLE_EFF_LR_ROBUSTZ_OUTLIER_THRESHOLD = 3.5
_ROBUST_Z_MAD_SCALE = 0.6745


def _pf_lifecycle_eff_lr_result_meta(
    zscores_by_method: dict[str, object],
) -> dict[str, object]:
    meta = zscores_by_method.get(PF_LIFECYCLE_EFF_LR_META_KEY)
    return meta if isinstance(meta, dict) else {}


def _iter_pf_lifecycle_eff_lr_score_methods(
    zscores_by_method: dict[str, object],
    methods: tuple[str, ...] = PF_LIFECYCLE_EFF_LR_OUTLIER_METHODS,
) -> tuple[str, ...]:
    return tuple(
        method
        for method in methods
        if method in zscores_by_method and method != PF_LIFECYCLE_EFF_LR_META_KEY
    )


def _is_pf_lifecycle_multi_window_event(event_name: str) -> bool:
    return event_name in PF_LIFECYCLE_EFF_LR_MULTI_WINDOW_EVENTS


def _pf_lifecycle_multi_window_event_meta(
    meta: dict[str, object],
    event_name: str,
) -> dict[str, object]:
    multi = meta.get("multi_window_events", {})
    if isinstance(multi, dict) and event_name in multi:
        event_meta = multi[event_name]
        return event_meta if isinstance(event_meta, dict) else {}
    if event_name == "random":
        return {
            "n_windows": meta.get("n_random_windows"),
            "capture_indices": meta.get("random_window_capture_indices", []),
            "window_scores": meta.get("random_window_scores", {}),
            "window_outlier_counts": meta.get("random_window_outlier_counts", {}),
        }
    return {}


def _align_per_weight_segments(
    segments: list[np.ndarray],
    *,
    n_weights: int,
) -> list[np.ndarray]:
    return [
        np.asarray(seg, dtype=np.float64).ravel()
        for seg in segments
        if np.asarray(seg, dtype=np.float64).ravel().size == n_weights
    ]


def _per_weight_consecutive_residuals(
    raw_segments: list[np.ndarray],
) -> list[np.ndarray]:
    """First-order residuals ``seg[t+1] - seg[t]`` along a temporal segment chain."""
    if len(raw_segments) < 2:
        return []
    return [
        np.asarray(raw_segments[idx + 1], dtype=np.float64)
        - np.asarray(raw_segments[idx], dtype=np.float64)
        for idx in range(len(raw_segments) - 1)
    ]


def _per_weight_scores_from_vicinity(
    event_values: np.ndarray,
    vicinity_segments: list[np.ndarray],
    *,
    outlier_method: OutlierMethod,
) -> np.ndarray:
    """Score each weight at the event against a vicinity window."""
    event_values = np.asarray(event_values, dtype=np.float64).ravel()
    if event_values.size == 0:
        return np.empty(0, dtype=np.float64)
    if not vicinity_segments:
        return np.full(event_values.size, np.nan, dtype=np.float64)

    n_weights = event_values.size
    aligned = _align_per_weight_segments(vicinity_segments, n_weights=n_weights)
    if not aligned:
        return np.full(n_weights, np.nan, dtype=np.float64)

    stack = np.stack(aligned, axis=0)
    if outlier_method == "raw_zscore":
        center = np.nanmean(stack, axis=0)
        if stack.shape[0] >= 2:
            scale = np.nanstd(stack, axis=0, ddof=1)
        else:
            scale = np.full(n_weights, np.nan, dtype=np.float64)
    elif outlier_method == "raw_robustz":
        center = np.nanmedian(stack, axis=0)
        scale = np.nanmedian(np.abs(stack - center), axis=0)
        scale = _ROBUST_Z_MAD_SCALE * scale
    else:
        raise ValueError(f"Expected a raw outlier method, got {outlier_method!r}")

    with np.errstate(divide="ignore", invalid="ignore"):
        scores = (event_values - center) / scale
    invalid = (
        ~np.isfinite(scale)
        | (np.abs(scale) < 1e-12)
        | ~np.isfinite(event_values)
        | ~np.isfinite(center)
    )
    scores = scores.astype(np.float64, copy=False)
    scores[invalid] = np.nan
    return scores


def _per_weight_scores_from_extended_residual_window(
    raw_segments: list[np.ndarray],
    *,
    n_pre: int,
    baseline: str,
    outlier_method: OutlierMethod,
) -> np.ndarray:
    """
    Residual outlier score on the extended lifecycle window.

    Given raw segments ``[pre_k, ..., pre1, event, post1, ..., post_m]`` the
    consecutive residuals are::

        r_pre1, ..., r_pre5, r_pre_birth, r_post_birth, r_post1, ..., r_post5

    when ``k == m == RESID_VICINITY_SEGMENTS_SIZE`` (two more raw samples than
    the raw-z vicinity on each side).

    - **pre:** score ``r_pre_birth`` against ``r_pre1..r_pre5``
    - **post:** score ``r_post_birth`` against ``r_post1..r_post5``
    """
    if outlier_method not in {"resid_zscore", "resid_robustz"}:
        raise ValueError(f"Expected a residual outlier method, got {outlier_method!r}")

    if not raw_segments:
        return np.empty(0, dtype=np.float64)

    n_pre = int(n_pre)
    n_weights = np.asarray(raw_segments[0], dtype=np.float64).ravel().size
    aligned = _align_per_weight_segments(raw_segments, n_weights=n_weights)
    n_post = len(aligned) - n_pre - 1
    if n_pre < 1 or n_post < 1 or len(aligned) < 3:
        return np.full(n_weights, np.nan, dtype=np.float64)

    residuals = _per_weight_consecutive_residuals(aligned)
    if not residuals:
        return np.full(n_weights, np.nan, dtype=np.float64)

    if baseline == "pre":
        if n_pre < 2:
            return np.full(n_weights, np.nan, dtype=np.float64)
        event_residual = np.asarray(residuals[n_pre - 1], dtype=np.float64)
        baseline_residuals = residuals[: n_pre - 1]
    elif baseline == "post":
        if n_post < 2:
            return np.full(n_weights, np.nan, dtype=np.float64)
        event_residual = np.asarray(residuals[n_pre], dtype=np.float64)
        baseline_residuals = residuals[n_pre + 1 : n_pre + n_post]
    else:
        raise ValueError(f"Unexpected baseline {baseline!r}")

    if not baseline_residuals:
        return np.full(n_weights, np.nan, dtype=np.float64)

    stack = np.stack(
        [np.asarray(seg, dtype=np.float64).ravel() for seg in baseline_residuals],
        axis=0,
    )
    if outlier_method == "resid_zscore":
        center = np.nanmean(stack, axis=0)
        if stack.shape[0] >= 2:
            scale = np.nanstd(stack, axis=0, ddof=1)
        else:
            scale = np.full(n_weights, np.nan, dtype=np.float64)
    else:
        center = np.nanmedian(stack, axis=0)
        scale = np.nanmedian(np.abs(stack - center), axis=0)
        scale = _ROBUST_Z_MAD_SCALE * scale

    with np.errstate(divide="ignore", invalid="ignore"):
        scores = (event_residual - center) / scale
    invalid = (
        ~np.isfinite(scale)
        | (np.abs(scale) < 1e-12)
        | ~np.isfinite(event_residual)
        | ~np.isfinite(center)
    )
    scores = scores.astype(np.float64, copy=False)
    scores[invalid] = np.nan
    return scores


def _residual_names_for_extended_window(n_pre: int, n_post: int) -> list[str]:
    """Human-readable residual labels for the extended lifecycle window."""
    if n_pre <= 0 or n_post <= 0:
        return []
    names = [f"r_pre{i}" for i in range(1, n_pre)]
    names.append("r_pre_birth")
    names.append("r_post_birth")
    names.extend(f"r_post{i}" for i in range(1, n_post))
    return names


def _safe_per_weight_scalar(arr: np.ndarray, weight_idx: int) -> float:
    """Return one per-weight signal value, or NaN when out of bounds."""
    values = np.asarray(arr, dtype=np.float64).ravel()
    wi = int(weight_idx)
    if wi < 0 or wi >= values.size:
        return float("nan")
    val = float(values[wi])
    return val if np.isfinite(val) else float("nan")


def explain_pf_lifecycle_weight_residual_scores(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    cell_idx: int,
    pf_idx: int,
    signals_dir: Path | str,
    capture_to_global: pd.Series,
    optimizer_config: dict | None = None,
    weight_idx: int,
    layer: str,
    event_name: str = "birth",
    window_size: int,
) -> dict[str, object]:
    """
    Print and return a step-by-step residual breakdown for one weight.

    Useful for debugging cases like near-linear ramps (tiny MAD -> huge
    robust-z) or isolated event spikes (zero post baseline MAD -> NaN z).
    """
    from experiments.common.signals_io import (
        load_hidden_unit_per_weight_effective_lr_by_layer,
    )

    resid_window = int(window_size) + 1
    cell_caps = (
        master_df.loc[
            (master_df["cell_idx"] == int(cell_idx)) & (master_df["segment_id"] >= 0)
        ]
        .drop_duplicates(["capture_idx"])
        .sort_values("capture_idx", kind="mergesort")["capture_idx"]
        .to_numpy(dtype=np.int32)
    )
    absent_caps = set(
        classify_pf_segment_groups(
            pf_segment_df,
            cell_idx=int(cell_idx),
            pf_idx=int(pf_idx),
            cell_segment_capture_indices=cell_caps,
        )["absent"]
    )
    events = pf_lifecycle_event_capture_indices(
        pf_segment_df,
        cell_idx=int(cell_idx),
        pf_idx=int(pf_idx),
        cell_segment_capture_indices=cell_caps,
        window_size=int(window_size),
        n_random_windows=1,
    )
    event_cap = events.get(event_name)
    if event_cap is None:
        raise ValueError(f"No capture for event {event_name!r}")

    capture_to_global_map = {int(cap): int(gidx) for cap, gidx in capture_to_global.items()}
    pre_caps, post_caps = pf_lifecycle_vicinity_window_capture_indices_for_event(
        int(event_cap),
        event_name=event_name,
        absent_caps=absent_caps,
        cell_segment_capture_indices=cell_caps,
        max_count=resid_window,
    )

    def _load_layer_values(caps: list[int]) -> list[float]:
        vals: list[float] = []
        for cap in caps:
            gidx = capture_to_global_map.get(int(cap))
            if gidx is None:
                vals.append(float("nan"))
                continue
            arr = load_hidden_unit_per_weight_effective_lr_by_layer(
                signals_dir,
                global_segment_idx=int(gidx),
                cell_idx=int(cell_idx),
                optimizer_config=optimizer_config,
                layers=(layer,),
            )[layer]
            vals.append(_safe_per_weight_scalar(arr, weight_idx))
        return vals

    pre_vals = _load_layer_values(pre_caps)
    event_val = _load_layer_values([int(event_cap)])[0]
    post_vals = _load_layer_values(post_caps)
    raw_vals = np.asarray(pre_vals + [event_val] + post_vals, dtype=np.float64)
    n_pre = len(pre_vals)
    n_post = len(post_vals)
    residual_names = _residual_names_for_extended_window(n_pre, n_post)
    residuals = np.diff(raw_vals)

    out: dict[str, object] = {
        "event_name": event_name,
        "layer": layer,
        "weight_idx": int(weight_idx),
        "pre_caps": pre_caps,
        "event_cap": int(event_cap),
        "post_caps": post_caps,
        "raw_vals": raw_vals,
        "residual_names": residual_names,
        "residuals": residuals,
        "baselines": {},
    }

    print(
        f"Residual breakdown: cell {cell_idx}, PF {pf_idx}, "
        f"{event_name}, layer={layer}, weight={weight_idx}"
    )
    print(f"raw ({len(raw_vals)} segs): {np.array2string(raw_vals, precision=6)}")
    if len(residual_names) == residuals.size:
        for name, val in zip(residual_names, residuals):
            print(f"  {name:>12}: {val:.9g}")
    else:
        print(f"residuals: {residuals}")

    for baseline in PF_LIFECYCLE_EFF_LR_BASELINES:
        if baseline == "pre" and n_pre < 2:
            continue
        if baseline == "post" and n_post < 2:
            continue
        if baseline == "pre":
            event_r = float(residuals[n_pre - 1])
            baseline_rs = residuals[: n_pre - 1]
            baseline_names = residual_names[: n_pre - 1]
        else:
            event_r = float(residuals[n_pre])
            baseline_rs = residuals[n_pre + 1 : n_pre + n_post]
            baseline_names = residual_names[n_pre + 1 : n_pre + n_post]

        mean = float(np.nanmean(baseline_rs)) if baseline_rs.size else float("nan")
        std = (
            float(np.nanstd(baseline_rs, ddof=1))
            if baseline_rs.size >= 2
            else float("nan")
        )
        median = float(np.nanmedian(baseline_rs)) if baseline_rs.size else float("nan")
        mad = (
            float(np.nanmedian(np.abs(baseline_rs - median)))
            if baseline_rs.size
            else float("nan")
        )
        robust_scale = _ROBUST_Z_MAD_SCALE * mad
        z = (event_r - mean) / std if np.isfinite(std) and abs(std) >= 1e-12 else float("nan")
        robust_z = (
            (event_r - median) / robust_scale
            if np.isfinite(robust_scale) and abs(robust_scale) >= 1e-12
            else float("nan")
        )

        baseline_info = {
            "event_residual": event_r,
            "baseline_residuals": baseline_rs,
            "baseline_names": baseline_names,
            "mean": mean,
            "std": std,
            "z": z,
            "median": median,
            "mad": mad,
            "robust_scale": robust_scale,
            "robust_z": robust_z,
        }
        out["baselines"][baseline] = baseline_info

        print(f"\nvs {baseline}:")
        print(f"  event residual = {event_r:.9g}")
        print(f"  baseline residuals ({baseline_names}): {baseline_rs}")
        print(f"  mean={mean:.9g}, std={std:.9g} -> resid_zscore={z:.3g}")
        print(
            f"  median={median:.9g}, MAD={mad:.9g}, "
            f"robust_scale={robust_scale:.9g} -> resid_robustz={robust_z:.3g}"
        )

    return out


def summarize_pf_lifecycle_eff_lr_outlier_methods(
    zscores_by_method: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    *,
    layers: tuple[str, ...] | None = None,
    event_order: tuple[str, ...] = PF_LIFECYCLE_EFF_LR_EVENTS,
    baselines: tuple[str, ...] = PF_LIFECYCLE_EFF_LR_BASELINES,
    methods: tuple[str, ...] = PF_LIFECYCLE_EFF_LR_OUTLIER_METHODS,
    print_summary: bool = True,
    signal_label: str = "Eff-LR",
) -> pd.DataFrame:
    """
    Count per-weight outliers for each eff-LR scoring method.

    Returns one row per (layer, event, baseline, method) with finite-score and
    outlier counts. For multi-window events, outlier counts are means across windows.
    """
    from experiments.common.signals_io import TRACKED_LAYER_NAMES

    if layers is None:
        layers = TRACKED_LAYER_NAMES

    score_methods = _iter_pf_lifecycle_eff_lr_score_methods(
        zscores_by_method,
        methods=methods,
    )
    meta = _pf_lifecycle_eff_lr_result_meta(zscores_by_method)

    rows: list[dict[str, object]] = []
    for layer in layers:
        for event_name in event_order:
            for baseline in baselines:
                for method in score_methods:
                    scores = (
                        zscores_by_method.get(method, {})
                        .get(event_name, {})
                        .get(baseline, {})
                        .get(layer, np.empty(0, dtype=np.float64))
                    )
                    scores = np.asarray(scores, dtype=np.float64)
                    finite = scores[np.isfinite(scores)]
                    threshold = _outlier_threshold_for_method(method)
                    if _is_pf_lifecycle_multi_window_event(event_name):
                        event_meta = _pf_lifecycle_multi_window_event_meta(
                            meta,
                            event_name,
                        )
                        window_counts = (
                            event_meta.get("window_outlier_counts", {})
                            .get(method, {})
                            .get(baseline, {})
                            .get(layer, np.empty(0, dtype=np.float64))
                        )
                        window_counts = np.asarray(window_counts, dtype=np.float64)
                        window_counts = window_counts[np.isfinite(window_counts)]
                        n_outliers = (
                            float(window_counts.mean())
                            if window_counts.size
                            else float("nan")
                        )
                        window_scores = (
                            event_meta.get("window_scores", {})
                            .get(method, {})
                            .get(baseline, {})
                            .get(layer, np.empty((0, 0), dtype=np.float64))
                        )
                        window_scores = np.asarray(window_scores, dtype=np.float64)
                        n_finite = (
                            int(window_scores.shape[1])
                            if window_scores.ndim == 2 and window_scores.size
                            else int(
                                finite.size
                                // max(len(event_meta.get("capture_indices", []) or []), 1)
                            )
                        )
                    else:
                        n_finite = int(finite.size)
                        n_outliers = float(_count_eff_lr_outliers(finite, method=method))
                    rows.append(
                        {
                            "layer": layer,
                            "event": event_name,
                            "baseline": baseline,
                            "method": method,
                            "method_label": PF_LIFECYCLE_EFF_LR_OUTLIER_METHOD_LABELS.get(
                                method, method
                            ),
                            "threshold": threshold,
                            "n_finite": n_finite,
                            "n_outliers": n_outliers,
                            "frac_outliers": (
                                float(n_outliers / n_finite)
                                if n_finite and np.isfinite(n_outliers)
                                else float("nan")
                            ),
                        }
                    )

    summary = pd.DataFrame(rows)
    if print_summary and not summary.empty:
        print(f"{signal_label} outlier counts by method (|score| > threshold; prefer lower frac):")
        for (layer, event_name), group in summary.groupby(["layer", "event"], sort=False):
            print(f"\n  {layer} / {event_name}")
            pivot = group.pivot(index="method_label", columns="baseline", values="n_outliers")
            frac = group.pivot(
                index="method_label", columns="baseline", values="frac_outliers"
            )
            n_finite = int(group["n_finite"].max())
            print(f"  n_weights={n_finite}")
            print("  outlier counts:")
            print(pivot.to_string())
            print("  outlier fractions:")
            print(frac.map(lambda x: f"{x:.3f}" if np.isfinite(x) else "nan").to_string())
            thresholds = group.drop_duplicates("method").set_index("method_label")[
                "threshold"
            ]
            print(f"  thresholds: {thresholds.to_dict()}")
    return summary


def pf_lifecycle_eff_lr_outlier_flag_table(
    zscores_by_method: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    *,
    layer: str,
    event_name: str,
    methods: tuple[str, ...] = PF_LIFECYCLE_EFF_LR_OUTLIER_METHODS,
    baselines: tuple[str, ...] = PF_LIFECYCLE_EFF_LR_BASELINES,
) -> pd.DataFrame:
    """Per-weight scores and outlier flags for method selection."""
    score_methods = _iter_pf_lifecycle_eff_lr_score_methods(
        zscores_by_method,
        methods=methods,
    )
    meta = _pf_lifecycle_eff_lr_result_meta(zscores_by_method)

    score_cols: dict[str, np.ndarray] = {}
    flag_cols: dict[str, np.ndarray] = {}
    n_weights = 0
    for method in score_methods:
        for baseline in baselines:
            if _is_pf_lifecycle_multi_window_event(event_name):
                event_meta = _pf_lifecycle_multi_window_event_meta(meta, event_name)
                window_scores = (
                    event_meta.get("window_scores", {})
                    .get(method, {})
                    .get(baseline, {})
                    .get(layer, np.empty((0, 0), dtype=np.float64))
                )
                window_scores = np.asarray(window_scores, dtype=np.float64)
                if window_scores.ndim == 2 and window_scores.size:
                    scores = np.nanmax(np.abs(window_scores), axis=0)
                    flags = np.any(
                        _eff_lr_outlier_mask(window_scores, method=method),
                        axis=0,
                    )
                    n_weights = max(n_weights, window_scores.shape[1])
                else:
                    scores = np.empty(0, dtype=np.float64)
                    flags = np.empty(0, dtype=bool)
            else:
                scores = np.asarray(
                    zscores_by_method.get(method, {})
                    .get(event_name, {})
                    .get(baseline, {})
                    .get(layer, np.empty(0, dtype=np.float64)),
                    dtype=np.float64,
                )
                flags = _eff_lr_outlier_mask(scores, method=method)
                if scores.size > n_weights:
                    n_weights = scores.size
            col = f"{method}_{baseline}"
            score_cols[col] = scores
            flag_cols[f"{col}_outlier"] = flags

    if n_weights == 0:
        return pd.DataFrame()

    out = pd.DataFrame({"weight_idx": np.arange(n_weights, dtype=np.int32)})
    for col, values in score_cols.items():
        padded = np.full(n_weights, np.nan, dtype=np.float64)
        padded[: values.size] = values
        out[col] = padded
    for col, flags in flag_cols.items():
        padded = np.zeros(n_weights, dtype=bool)
        padded[: flags.size] = flags
        out[col] = padded

    outlier_flag_cols = [c for c in out.columns if c.endswith("_outlier")]
    out["n_outlier_slots"] = out[outlier_flag_cols].sum(axis=1)
    zscore_flag_cols = [
        c
        for c in outlier_flag_cols
        if c.startswith("raw_zscore_") or c.startswith("resid_zscore_")
    ]
    robust_flag_cols = [
        c
        for c in outlier_flag_cols
        if c.startswith("raw_robustz_") or c.startswith("resid_robustz_")
    ]
    out["any_zscore_outlier"] = out[zscore_flag_cols].any(axis=1)
    out["any_robustz_outlier"] = out[robust_flag_cols].any(axis=1)
    out["robustz_only_outlier"] = out["any_robustz_outlier"] & ~out["any_zscore_outlier"]
    out["all_methods_outlier"] = out[outlier_flag_cols].all(axis=1)
    return out


def explain_pf_lifecycle_weight_eff_lr_scores(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    zscores_by_method: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    cell_idx: int,
    pf_idx: int,
    signals_dir: Path | str,
    capture_to_global: pd.Series,
    optimizer_config: dict | None = None,
    weight_idx: int,
    layer: str,
    event_name: str = "birth",
    window_size: int,
    signal_kind: str = "effective_lr",
    signal_label: str = "eff-LR",
) -> dict[str, object]:
    """
    Print a full eff-LR outlier breakdown for one weight across all methods.

    Includes an outlier flag table, raw-z vicinity stats, and the extended
    residual window breakdown.
    """
    from experiments.common.signals_io import (
        load_hidden_unit_per_weight_delta_w_by_layer,
        load_hidden_unit_per_weight_effective_lr_by_layer,
    )

    weight_idx = int(weight_idx)
    print(
        f"\n{'#' * 80}\n"
        f"Weight {weight_idx} @ {event_name} / {layer} "
        f"(cell {cell_idx}, PF {pf_idx})\n"
        f"{'#' * 80}"
    )

    score_rows: list[dict[str, object]] = []
    for baseline in PF_LIFECYCLE_EFF_LR_BASELINES:
        for method in PF_LIFECYCLE_EFF_LR_OUTLIER_METHODS:
            scores = (
                zscores_by_method.get(method, {})
                .get(event_name, {})
                .get(baseline, {})
                .get(layer, np.empty(0, dtype=np.float64))
            )
            score = (
                float(scores[weight_idx])
                if weight_idx < scores.size and np.isfinite(scores[weight_idx])
                else float("nan")
            )
            threshold = _outlier_threshold_for_method(method)
            is_outlier = (
                bool(np.isfinite(score) and abs(score) > threshold)
                if np.isfinite(score)
                else False
            )
            score_rows.append(
                {
                    "baseline": PF_LIFECYCLE_EFF_LR_BASELINE_LABELS.get(
                        baseline, baseline
                    ),
                    "method": PF_LIFECYCLE_EFF_LR_OUTLIER_METHOD_LABELS.get(
                        method, method
                    ),
                    "score": score,
                    "threshold": threshold,
                    "outlier": is_outlier,
                }
            )
    score_df = pd.DataFrame(score_rows)
    print("\nOutlier flags (|score| > threshold):")
    print(
        score_df.to_string(
            index=False,
            formatters={
                "score": lambda x: f"{x:.3g}" if np.isfinite(x) else "nan",
                "threshold": lambda x: f"{x:.1f}",
                "outlier": lambda x: "YES" if x else "no",
            },
        )
    )

    cell_caps = (
        master_df.loc[
            (master_df["cell_idx"] == int(cell_idx)) & (master_df["segment_id"] >= 0)
        ]
        .drop_duplicates(["capture_idx"])
        .sort_values("capture_idx", kind="mergesort")["capture_idx"]
        .to_numpy(dtype=np.int32)
    )
    absent_caps = set(
        classify_pf_segment_groups(
            pf_segment_df,
            cell_idx=int(cell_idx),
            pf_idx=int(pf_idx),
            cell_segment_capture_indices=cell_caps,
        )["absent"]
    )
    events = pf_lifecycle_event_capture_indices(
        pf_segment_df,
        cell_idx=int(cell_idx),
        pf_idx=int(pf_idx),
        cell_segment_capture_indices=cell_caps,
        window_size=int(window_size),
        n_random_windows=1,
    )
    event_cap = events.get(event_name)
    if event_cap is None:
        raise ValueError(f"No capture for event {event_name!r}")

    capture_to_global_map = {int(cap): int(gidx) for cap, gidx in capture_to_global.items()}

    def _load_values(caps: list[int]) -> np.ndarray:
        vals: list[float] = []
        for cap in caps:
            gidx = capture_to_global_map.get(int(cap))
            if gidx is None:
                vals.append(float("nan"))
                continue
            arr = (
                load_hidden_unit_per_weight_delta_w_by_layer(
                    signals_dir,
                    global_segment_idx=int(gidx),
                    cell_idx=int(cell_idx),
                    layers=(layer,),
                )[layer]
                if signal_kind == "delta_w"
                else load_hidden_unit_per_weight_effective_lr_by_layer(
                    signals_dir,
                    global_segment_idx=int(gidx),
                    cell_idx=int(cell_idx),
                    optimizer_config=optimizer_config,
                    layers=(layer,),
                )[layer]
            )
            vals.append(_safe_per_weight_scalar(arr, weight_idx))
        return np.asarray(vals, dtype=np.float64)

    pre_caps, post_caps = pf_lifecycle_vicinity_window_capture_indices_for_event(
        int(event_cap),
        event_name=event_name,
        absent_caps=absent_caps,
        cell_segment_capture_indices=cell_caps,
        max_count=int(window_size),
    )
    pre_vals = _load_values(pre_caps)
    event_val = float(_load_values([int(event_cap)])[0])
    post_vals = _load_values(post_caps)
    event_label = PF_LIFECYCLE_EFF_LR_EVENT_LABELS.get(event_name, event_name)
    pre_labels = [f"pre-{i}" for i in range(len(pre_caps), 0, -1)]
    post_labels = [f"post+{i}" for i in range(1, len(post_caps) + 1)]

    print(
        f"\nRaw {signal_label} window for raw-z ({len(pre_caps)} pre + {event_label} + "
        f"{len(post_caps)} post):"
    )
    print(f"  pre ({pre_labels}): {np.array2string(pre_vals, precision=6)}")
    print(f"  {event_label}: {event_val:.6g}")
    print(f"  post ({post_labels}): {np.array2string(post_vals, precision=6)}")

    raw_baselines: dict[str, object] = {}
    for baseline, baseline_vals, baseline_labels in (
        ("pre", pre_vals, pre_labels),
        ("post", post_vals, post_labels),
    ):
        if baseline_vals.size == 0:
            continue
        mean = float(np.nanmean(baseline_vals))
        std = (
            float(np.nanstd(baseline_vals, ddof=1))
            if baseline_vals.size >= 2
            else float("nan")
        )
        median = float(np.nanmedian(baseline_vals))
        mad = float(np.nanmedian(np.abs(baseline_vals - median)))
        robust_scale = _ROBUST_Z_MAD_SCALE * mad
        raw_z = (
            (event_val - mean) / std
            if np.isfinite(std) and abs(std) >= 1e-12
            else float("nan")
        )
        raw_robust_z = (
            (event_val - median) / robust_scale
            if np.isfinite(robust_scale) and abs(robust_scale) >= 1e-12
            else float("nan")
        )
        raw_baselines[baseline] = {
            "baseline_vals": baseline_vals,
            "baseline_labels": baseline_labels,
            "mean": mean,
            "std": std,
            "raw_zscore": raw_z,
            "median": median,
            "mad": mad,
            "raw_robustz": raw_robust_z,
        }
        baseline_label = PF_LIFECYCLE_EFF_LR_BASELINE_LABELS.get(baseline, baseline)
        print(f"\nRaw-z vs {baseline_label} (event={event_label} vs {baseline} segments):")
        print(f"  baseline mean={mean:.6g}, std={std:.6g} -> raw_zscore={raw_z:.3g}")
        print(
            f"  baseline median={median:.6g}, MAD={mad:.6g} -> "
            f"raw_robustz={raw_robust_z:.3g}"
        )

    residual_info = explain_pf_lifecycle_weight_residual_scores(
        pf_segment_df,
        master_df,
        cell_idx=cell_idx,
        pf_idx=pf_idx,
        signals_dir=signals_dir,
        capture_to_global=capture_to_global,
        optimizer_config=optimizer_config,
        weight_idx=weight_idx,
        layer=layer,
        event_name=event_name,
        window_size=window_size,
    )

    return {
        "weight_idx": weight_idx,
        "layer": layer,
        "event_name": event_name,
        "score_table": score_df,
        "event_val": event_val,
        "pre_vals": pre_vals,
        "post_vals": post_vals,
        "raw_baselines": raw_baselines,
        "residual_info": residual_info,
    }


def scan_pf_lifecycle_eff_lr_jumps(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    cell_idx: int,
    pf_idx: int,
    signals_dir: Path | str,
    capture_to_global: pd.Series,
    optimizer_config: dict | None = None,
    layer: str,
    window_size: int,
    n_random_windows: int,
    event_name: str = "birth",
    top_k: int = 10,
    signal_kind: str = "effective_lr",
) -> pd.DataFrame:
    """
    Rank hidden-unit weights by birth/post event jumps in the extended window.

    Useful to locate the weight behind a visible plot outlier when a guessed
    index (e.g. 494 vs 464) does not match the spike you remember.
    """
    from experiments.common.signals_io import (
        load_hidden_unit_per_weight_delta_w_by_layer,
        load_hidden_unit_per_weight_effective_lr_by_layer,
    )

    resid_window = _resid_window_size(window_size)
    cell_caps = (
        master_df.loc[
            (master_df["cell_idx"] == int(cell_idx)) & (master_df["segment_id"] >= 0)
        ]
        .drop_duplicates(["capture_idx"])
        .sort_values("capture_idx", kind="mergesort")["capture_idx"]
        .to_numpy(dtype=np.int32)
    )
    absent_caps = set(
        classify_pf_segment_groups(
            pf_segment_df,
            cell_idx=int(cell_idx),
            pf_idx=int(pf_idx),
            cell_segment_capture_indices=cell_caps,
        )["absent"]
    )
    events = pf_lifecycle_event_capture_indices(
        pf_segment_df,
        cell_idx=int(cell_idx),
        pf_idx=int(pf_idx),
        cell_segment_capture_indices=cell_caps,
        window_size=int(window_size),
        n_random_windows=int(n_random_windows),
    )
    event_cap = events.get(event_name)
    if event_cap is None:
        return pd.DataFrame()

    capture_to_global_map = {int(cap): int(gidx) for cap, gidx in capture_to_global.items()}
    pre_caps, post_caps = _pf_lifecycle_vicinity_window_capture_indices(
        int(event_cap),
        absent_caps=absent_caps,
        cell_segment_capture_indices=cell_caps,
        max_count=resid_window,
    )
    caps = [int(c) for c in pre_caps] + [int(event_cap)] + [int(c) for c in post_caps]
    gidxs = [capture_to_global_map.get(cap) for cap in caps]
    if any(g is None for g in gidxs):
        return pd.DataFrame()

    def _load_layer_at_gidx(gidx: int) -> np.ndarray:
        if signal_kind == "delta_w":
            return load_hidden_unit_per_weight_delta_w_by_layer(
                signals_dir,
                global_segment_idx=int(gidx),
                cell_idx=int(cell_idx),
                layers=(layer,),
            )[layer]
        return load_hidden_unit_per_weight_effective_lr_by_layer(
            signals_dir,
            global_segment_idx=int(gidx),
            cell_idx=int(cell_idx),
            optimizer_config=optimizer_config,
            layers=(layer,),
        )[layer]

    layers_at_caps = [_load_layer_at_gidx(int(gidx)) for gidx in gidxs]
    stack = np.stack(layers_at_caps, axis=0)
    n_pre = len(pre_caps)
    birth_vals = stack[n_pre]
    post1_vals = stack[n_pre + 1] if stack.shape[0] > n_pre + 1 else np.full_like(birth_vals, np.nan)
    pre1_vals = stack[n_pre - 1] if n_pre >= 1 else np.full_like(birth_vals, np.nan)

    r_pre_birth = birth_vals - pre1_vals
    r_post_birth = post1_vals - birth_vals
    ratio_post_birth = post1_vals / np.maximum(np.abs(birth_vals), 1e-12)

    df = pd.DataFrame(
        {
            "weight_idx": np.arange(stack.shape[1], dtype=np.int32),
            "pre1": pre1_vals,
            "birth": birth_vals,
            "post1": post1_vals,
            "r_pre_birth": r_pre_birth,
            "r_post_birth": r_post_birth,
            "post1_over_birth": ratio_post_birth,
        }
    )
    df["abs_r_post_birth"] = np.abs(df["r_post_birth"])
    df = df.sort_values("abs_r_post_birth", ascending=False, kind="mergesort")
    return df.head(int(top_k)).reset_index(drop=True)


def _compute_pf_lifecycle_per_weight_signal_zscores(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    cell_idx: int,
    pf_idx: int,
    signals_dir: Path | str,
    capture_to_global: pd.Series,
    load_layer_segments_for_gidx,
    window_size: int,
    outlier_methods: tuple[OutlierMethod, ...] = PF_LIFECYCLE_EFF_LR_OUTLIER_METHODS,
    random_seed: int | None = 42,
    n_random_windows: int,
    layers: tuple[str, ...] | None = None,
    include_single_lifecycle_events: bool = True,
    first_revival_capture_idx: int | None = None,
    score_first_revival: bool = True,
) -> dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]:
    """
    Per-weight outlier scores at birth, first revival, random, and dead_random.

    ``load_layer_segments_for_gidx(gidx)`` must return per-layer incoming/outgoing
    weight signal vectors for the hidden unit (effective LR, ΔW, etc.).
    """
    from experiments.common.signals_io import TRACKED_LAYER_NAMES

    tracked_layers = layers or TRACKED_LAYER_NAMES
    needs_resid_methods = any(
        method in {"resid_zscore", "resid_robustz"} for method in outlier_methods
    )
    segment_by_gidx_cache: dict[int, dict[str, np.ndarray]] = {}

    def _load_layer_segments_for_gidx(gidx: int) -> dict[str, np.ndarray]:
        gidx = int(gidx)
        cached = segment_by_gidx_cache.get(gidx)
        if cached is not None:
            return cached
        loaded = load_layer_segments_for_gidx(gidx)
        segment_by_gidx_cache[gidx] = loaded
        return loaded

    cell_caps = (
        master_df.loc[
            (master_df["cell_idx"] == int(cell_idx)) & (master_df["segment_id"] >= 0)
        ]
        .drop_duplicates(["capture_idx"])
        .sort_values("capture_idx", kind="mergesort")["capture_idx"]
        .to_numpy(dtype=np.int32)
    )
    absent_caps = set(
        classify_pf_segment_groups(
            pf_segment_df,
            cell_idx=int(cell_idx),
            pf_idx=int(pf_idx),
            cell_segment_capture_indices=cell_caps,
        )["absent"]
    )
    events = pf_lifecycle_event_capture_indices(
        pf_segment_df,
        cell_idx=int(cell_idx),
        pf_idx=int(pf_idx),
        cell_segment_capture_indices=cell_caps,
        window_size=int(window_size),
        n_random_windows=int(n_random_windows),
        random_seed=random_seed,
    )
    if first_revival_capture_idx is not None:
        events["first_revival"] = int(first_revival_capture_idx)
    elif not score_first_revival:
        events["first_revival"] = None
    capture_to_global_map = {int(cap): int(gidx) for cap, gidx in capture_to_global.items()}
    multi_window_caps = {
        event_name: [
            int(cap)
            for cap in (events.get(f"{event_name}_windows") or [])
        ]
        for event_name in PF_LIFECYCLE_EFF_LR_MULTI_WINDOW_EVENTS
    }

    out: dict[str, object] = {
        method: {
            event_name: {baseline: {} for baseline in PF_LIFECYCLE_EFF_LR_BASELINES}
            for event_name in PF_LIFECYCLE_EFF_LR_EVENTS
        }
        for method in outlier_methods
    }
    multi_window_scores: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
        event_name: {
            method: {
                baseline: {layer: [] for layer in tracked_layers}
                for baseline in PF_LIFECYCLE_EFF_LR_BASELINES
            }
            for method in outlier_methods
        }
        for event_name in PF_LIFECYCLE_EFF_LR_MULTI_WINDOW_EVENTS
    }
    multi_window_outlier_counts: dict[str, dict[str, dict[str, list[int]]]] = {
        event_name: {
            method: {
                baseline: {layer: [] for layer in tracked_layers}
                for baseline in PF_LIFECYCLE_EFF_LR_BASELINES
            }
            for method in outlier_methods
        }
        for event_name in PF_LIFECYCLE_EFF_LR_MULTI_WINDOW_EVENTS
    }

    def _layer_segments_for_caps(caps: list[int]) -> dict[int, dict[str, np.ndarray]]:
        by_gidx: dict[int, dict[str, np.ndarray]] = {}
        for cap in caps:
            gidx = capture_to_global_map.get(int(cap))
            if gidx is None:
                continue
            by_gidx[int(gidx)] = _load_layer_segments_for_gidx(int(gidx))
        return by_gidx

    def _layer_segments_in_capture_order(
        caps: list[int],
        by_gidx: dict[int, dict[str, np.ndarray]],
    ) -> dict[str, list[np.ndarray]]:
        by_layer: dict[str, list[np.ndarray]] = {
            layer: [] for layer in tracked_layers
        }
        for cap in caps:
            gidx = capture_to_global_map.get(int(cap))
            if gidx is None:
                continue
            layers_at_gidx = by_gidx.get(int(gidx))
            if layers_at_gidx is None:
                continue
            for layer in tracked_layers:
                by_layer[layer].append(
                    layers_at_gidx.get(layer, np.empty(0, dtype=np.float64))
                )
        return by_layer

    def _scores_for_anchor(
        capture_idx: int,
        event_name: str,
    ) -> dict[str, dict[str, dict[str, np.ndarray]]]:
        event_global_idx = capture_to_global_map.get(int(capture_idx))
        if event_global_idx is None:
            return {}

        pre_caps, post_caps = pf_lifecycle_vicinity_window_capture_indices_for_event(
            int(capture_idx),
            event_name=event_name,
            absent_caps=absent_caps,
            cell_segment_capture_indices=cell_caps,
            max_count=window_size,
        )
        pre_by_layer = _layer_segments_in_capture_order(
            pre_caps,
            _layer_segments_for_caps(pre_caps),
        )
        post_by_layer = _layer_segments_in_capture_order(
            post_caps,
            _layer_segments_for_caps(post_caps),
        )
        if needs_resid_methods:
            resid_pre_caps, resid_post_caps = (
                pf_lifecycle_vicinity_window_capture_indices_for_event(
                    int(capture_idx),
                    event_name=event_name,
                    absent_caps=absent_caps,
                    cell_segment_capture_indices=cell_caps,
                    max_count=window_size + 1,
                )
            )
            resid_pre_by_layer = _layer_segments_in_capture_order(
                resid_pre_caps,
                _layer_segments_for_caps(resid_pre_caps),
            )
            resid_post_by_layer = _layer_segments_in_capture_order(
                resid_post_caps,
                _layer_segments_for_caps(resid_post_caps),
            )
        else:
            resid_pre_by_layer = {layer: [] for layer in tracked_layers}
            resid_post_by_layer = {layer: [] for layer in tracked_layers}
        event_by_layer = _load_layer_segments_for_gidx(int(event_global_idx))

        scores_by_method: dict[str, dict[str, dict[str, np.ndarray]]] = {
            method: {
                baseline: {} for baseline in PF_LIFECYCLE_EFF_LR_BASELINES
            }
            for method in outlier_methods
        }
        for layer in tracked_layers:
            event_vals = event_by_layer.get(layer, np.empty(0, dtype=np.float64))
            if event_vals.size == 0:
                continue

            pre_segments = pre_by_layer[layer]
            post_segments = post_by_layer[layer]
            resid_pre_segments = resid_pre_by_layer[layer]
            resid_post_segments = resid_post_by_layer[layer]
            resid_raw_chain = resid_pre_segments + [event_vals] + resid_post_segments
            n_pre = len(resid_pre_segments)

            for method in outlier_methods:
                if method in {"raw_zscore", "raw_robustz"}:
                    scores_by_method[method]["pre"][layer] = (
                        _per_weight_scores_from_vicinity(
                            event_vals,
                            pre_segments,
                            outlier_method=method,
                        )
                    )
                    scores_by_method[method]["post"][layer] = (
                        _per_weight_scores_from_vicinity(
                            event_vals,
                            post_segments,
                            outlier_method=method,
                        )
                    )
                else:
                    scores_by_method[method]["pre"][layer] = (
                        _per_weight_scores_from_extended_residual_window(
                            resid_raw_chain,
                            n_pre=n_pre,
                            baseline="pre",
                            outlier_method=method,
                        )
                    )
                    scores_by_method[method]["post"][layer] = (
                        _per_weight_scores_from_extended_residual_window(
                            resid_raw_chain,
                            n_pre=n_pre,
                            baseline="post",
                            outlier_method=method,
                        )
                    )
        return scores_by_method

    lifecycle_events = [
        event_name
        for event_name in PF_LIFECYCLE_EFF_LR_EVENTS
        if not _is_pf_lifecycle_multi_window_event(event_name)
    ]
    if include_single_lifecycle_events:
        for event_name in lifecycle_events:
            capture_idx = events.get(event_name)
            if capture_idx is None:
                continue
            scores_by_method = _scores_for_anchor(int(capture_idx), event_name)
            for method in outlier_methods:
                for baseline in PF_LIFECYCLE_EFF_LR_BASELINES:
                    for layer, scores in scores_by_method.get(method, {}).get(
                        baseline, {}
                    ).items():
                        out[method][event_name][baseline][layer] = scores

    if int(n_random_windows) > 0:
        for event_name in PF_LIFECYCLE_EFF_LR_MULTI_WINDOW_EVENTS:
            for capture_idx in multi_window_caps.get(event_name, []):
                scores_by_method = _scores_for_anchor(int(capture_idx), event_name)
                for method in outlier_methods:
                    for baseline in PF_LIFECYCLE_EFF_LR_BASELINES:
                        for layer in tracked_layers:
                            scores = scores_by_method.get(method, {}).get(baseline, {}).get(
                                layer, np.empty(0, dtype=np.float64)
                            )
                            scores = np.asarray(scores, dtype=np.float64)
                            if scores.size == 0:
                                continue
                            multi_window_scores[event_name][method][baseline][layer].append(
                                scores
                            )
                            multi_window_outlier_counts[event_name][method][baseline][
                                layer
                            ].append(_count_eff_lr_outliers(scores, method=method))

        multi_window_events_meta: dict[str, dict[str, object]] = {}
        for event_name in PF_LIFECYCLE_EFF_LR_MULTI_WINDOW_EVENTS:
            for method in outlier_methods:
                for baseline in PF_LIFECYCLE_EFF_LR_BASELINES:
                    for layer in tracked_layers:
                        window_scores = multi_window_scores[event_name][method][baseline][layer]
                        if window_scores:
                            out[method][event_name][baseline][layer] = np.concatenate(
                                [
                                    np.asarray(scores, dtype=np.float64).ravel()
                                    for scores in window_scores
                                ]
                            )

            window_scores_stack: dict[str, dict[str, dict[str, np.ndarray]]] = {
                method: {baseline: {} for baseline in PF_LIFECYCLE_EFF_LR_BASELINES}
                for method in outlier_methods
            }
            window_outlier_counts_arr: dict[str, dict[str, dict[str, np.ndarray]]] = {
                method: {baseline: {} for baseline in PF_LIFECYCLE_EFF_LR_BASELINES}
                for method in outlier_methods
            }
            for method in outlier_methods:
                for baseline in PF_LIFECYCLE_EFF_LR_BASELINES:
                    for layer in tracked_layers:
                        window_scores = multi_window_scores[event_name][method][baseline][layer]
                        if window_scores:
                            window_scores_stack[method][baseline][layer] = np.stack(
                                window_scores,
                                axis=0,
                            )
                        counts = multi_window_outlier_counts[event_name][method][baseline][
                            layer
                        ]
                        if counts:
                            window_outlier_counts_arr[method][baseline][layer] = (
                                np.asarray(counts, dtype=np.float64)
                            )

            multi_window_events_meta[event_name] = {
                "n_windows_requested": int(n_random_windows),
                "n_windows": len(multi_window_caps.get(event_name, [])),
                "capture_indices": multi_window_caps.get(event_name, []),
                "window_scores": window_scores_stack,
                "window_outlier_counts": window_outlier_counts_arr,
            }
    else:
        multi_window_events_meta = {}

    out[PF_LIFECYCLE_EFF_LR_META_KEY] = {
        "n_random_windows": int(n_random_windows),
        "multi_window_events": multi_window_events_meta,
        "random_window_capture_indices": multi_window_caps.get("random", []),
        "random_window_scores": multi_window_events_meta.get("random", {}).get(
            "window_scores",
            {},
        ),
        "random_window_outlier_counts": multi_window_events_meta.get("random", {}).get(
            "window_outlier_counts",
            {},
        ),
    }
    return out


def compute_pf_lifecycle_per_weight_eff_lr_zscores(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    cell_idx: int,
    pf_idx: int,
    signals_dir: Path | str,
    capture_to_global: pd.Series,
    optimizer_config: dict | None = None,
    window_size: int,
    outlier_methods: tuple[OutlierMethod, ...] = PF_LIFECYCLE_EFF_LR_OUTLIER_METHODS,
    random_seed: int | None = 42,
    n_random_windows: int,
    layers: tuple[str, ...] | None = None,
    include_single_lifecycle_events: bool = True,
    first_revival_capture_idx: int | None = None,
    score_first_revival: bool = True,
) -> dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]:
    """Per-weight effective-LR outlier scores (see ``_compute_pf_lifecycle_per_weight_signal_zscores``)."""
    from experiments.common.signals_io import load_hidden_unit_per_weight_effective_lr_by_layer

    tracked_layers = layers
    if tracked_layers is None:
        from experiments.common.signals_io import TRACKED_LAYER_NAMES

        tracked_layers = TRACKED_LAYER_NAMES

    def _load(gidx: int) -> dict[str, np.ndarray]:
        return load_hidden_unit_per_weight_effective_lr_by_layer(
            signals_dir,
            global_segment_idx=int(gidx),
            cell_idx=int(cell_idx),
            optimizer_config=optimizer_config,
            layers=tracked_layers,
        )

    return _compute_pf_lifecycle_per_weight_signal_zscores(
        pf_segment_df,
        master_df,
        cell_idx=cell_idx,
        pf_idx=pf_idx,
        signals_dir=signals_dir,
        capture_to_global=capture_to_global,
        load_layer_segments_for_gidx=_load,
        window_size=window_size,
        outlier_methods=outlier_methods,
        random_seed=random_seed,
        n_random_windows=n_random_windows,
        layers=layers,
        include_single_lifecycle_events=include_single_lifecycle_events,
        first_revival_capture_idx=first_revival_capture_idx,
        score_first_revival=score_first_revival,
    )


def compute_pf_lifecycle_per_weight_delta_w_zscores(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    cell_idx: int,
    pf_idx: int,
    signals_dir: Path | str,
    capture_to_global: pd.Series,
    window_size: int,
    outlier_methods: tuple[OutlierMethod, ...] = PF_LIFECYCLE_EFF_LR_OUTLIER_METHODS,
    random_seed: int | None = 42,
    n_random_windows: int,
    layers: tuple[str, ...] | None = None,
    include_single_lifecycle_events: bool = True,
    first_revival_capture_idx: int | None = None,
    score_first_revival: bool = True,
) -> dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]:
    """Per-weight signed ΔW outlier scores (``W[g+1]-W[g]`` at each anchor)."""
    from experiments.common.signals_io import load_hidden_unit_per_weight_delta_w_by_layer

    tracked_layers = layers
    if tracked_layers is None:
        from experiments.common.signals_io import TRACKED_LAYER_NAMES

        tracked_layers = TRACKED_LAYER_NAMES

    def _load(gidx: int) -> dict[str, np.ndarray]:
        return load_hidden_unit_per_weight_delta_w_by_layer(
            signals_dir,
            global_segment_idx=int(gidx),
            cell_idx=int(cell_idx),
            layers=tracked_layers,
        )

    return _compute_pf_lifecycle_per_weight_signal_zscores(
        pf_segment_df,
        master_df,
        cell_idx=cell_idx,
        pf_idx=pf_idx,
        signals_dir=signals_dir,
        capture_to_global=capture_to_global,
        load_layer_segments_for_gidx=_load,
        window_size=window_size,
        outlier_methods=outlier_methods,
        random_seed=random_seed,
        n_random_windows=n_random_windows,
        layers=layers,
        include_single_lifecycle_events=include_single_lifecycle_events,
        first_revival_capture_idx=first_revival_capture_idx,
        score_first_revival=score_first_revival,
    )


def _compute_pf_outlier_zscores_for_pair(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    cell_idx: int,
    pf_idx: int,
    signals_dir: Path | str,
    capture_to_global: pd.Series | dict[int, int],
    optimizer_config: dict | None,
    window_size: int,
    method: str,
    layer: str,
    n_random_windows: int,
    random_seed: int | None,
    include_single_lifecycle_events: bool,
    first_revival_capture_idx: int | None = None,
    score_first_revival: bool = True,
    signal_kind: str = "effective_lr",
) -> dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]:
    """Single zscore pass for window and/or lifecycle outlier scoring."""
    if signal_kind == "delta_w":
        return compute_pf_lifecycle_per_weight_delta_w_zscores(
            pf_segment_df,
            master_df,
            cell_idx=int(cell_idx),
            pf_idx=int(pf_idx),
            signals_dir=signals_dir,
            capture_to_global=capture_to_global,
            window_size=int(window_size),
            outlier_methods=(method,),  # type: ignore[arg-type]
            random_seed=random_seed,
            n_random_windows=int(n_random_windows),
            layers=(layer,),
            include_single_lifecycle_events=include_single_lifecycle_events,
            first_revival_capture_idx=first_revival_capture_idx,
            score_first_revival=score_first_revival,
        )
    return compute_pf_lifecycle_per_weight_eff_lr_zscores(
        pf_segment_df,
        master_df,
        cell_idx=int(cell_idx),
        pf_idx=int(pf_idx),
        signals_dir=signals_dir,
        capture_to_global=capture_to_global,
        optimizer_config=optimizer_config,
        window_size=int(window_size),
        outlier_methods=(method,),  # type: ignore[arg-type]
        random_seed=random_seed,
        n_random_windows=int(n_random_windows),
        layers=(layer,),
        include_single_lifecycle_events=include_single_lifecycle_events,
        first_revival_capture_idx=first_revival_capture_idx,
        score_first_revival=score_first_revival,
    )


def _window_outlier_rows_from_zscores(
    zscores_by_method: dict[str, object],
    *,
    cell_idx: int,
    pf_idx: int,
    method: str,
    layer: str,
    baselines: tuple[str, ...],
) -> list[dict[str, object]]:
    """Extract per-window outlier-count rows from one zscore result."""
    meta = _pf_lifecycle_eff_lr_result_meta(zscores_by_method)
    rows: list[dict[str, object]] = []
    for event_name in PF_LIFECYCLE_EFF_LR_MULTI_WINDOW_EVENTS:
        event_meta = _pf_lifecycle_multi_window_event_meta(meta, event_name)
        capture_indices = event_meta.get("capture_indices") or []
        per_baseline_counts: list[np.ndarray] = []
        for baseline in baselines:
            counts = (
                event_meta.get("window_outlier_counts", {})
                .get(method, {})
                .get(baseline, {})
                .get(layer, np.empty(0, dtype=np.float64))
            )
            counts = np.asarray(counts, dtype=np.float64)
            if counts.size:
                per_baseline_counts.append(counts)
        if not per_baseline_counts:
            continue
        n_event_windows = int(min(len(arr) for arr in per_baseline_counts))
        total_counts = sum(arr[:n_event_windows] for arr in per_baseline_counts)
        for window_idx in range(n_event_windows):
            cap = (
                int(capture_indices[window_idx])
                if window_idx < len(capture_indices)
                else None
            )
            rows.append(
                {
                    "cell_idx": int(cell_idx),
                    "pf_idx": int(pf_idx),
                    "event_name": event_name,
                    "window_idx": int(window_idx),
                    "capture_idx": cap,
                    "n_outliers": int(total_counts[window_idx]),
                    "method": method,
                    "layer": layer,
                }
            )
    return rows


def _lifecycle_outlier_row_from_zscores(
    zscores_by_method: dict[str, object],
    *,
    cell_idx: int,
    pf_idx: int,
    eligibility: dict[str, object],
    method: str,
    layer: str,
) -> dict[str, object]:
    """Extract birth/revive lifecycle outlier counts from one zscore result."""
    revive_cap = eligibility.get("first_eligible_revive_capture_idx")
    score_revive = revive_cap is not None and not pd.isna(revive_cap)
    birth_pre, birth_post = _lifecycle_event_outlier_counts_from_zscores(
        zscores_by_method,
        event_name="birth",
        method=str(method),
        layer=str(layer),
    )
    if score_revive:
        revive_pre, revive_post = _lifecycle_event_outlier_counts_from_zscores(
            zscores_by_method,
            event_name="first_revival",
            method=str(method),
            layer=str(layer),
        )
    else:
        revive_pre, revive_post = float("nan"), float("nan")
    return {
        **eligibility,
        "cell_idx": int(cell_idx),
        "pf_idx": int(pf_idx),
        "method": str(method),
        "layer": str(layer),
        "birth_pre_n_outliers": int(birth_pre),
        "birth_post_n_outliers": int(birth_post),
        "revive_pre_n_outliers": revive_pre,
        "revive_post_n_outliers": revive_post,
    }


def _collect_pf_window_and_lifecycle_outlier_counts_for_pair(
    cell_idx: int,
    pf_idx: int,
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    signals_dir: Path | str,
    capture_to_global: pd.Series | dict[int, int],
    optimizer_config: dict | None,
    window_size: int,
    n_windows: int,
    method: str,
    layer: str,
    baselines: tuple[str, ...],
    random_seed: int | None,
    eligibility: dict[str, object] | None,
    *,
    collect_windows: bool,
    signal_kind: str = "effective_lr",
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    """Score one PF once; return window rows and optional lifecycle row."""
    cell_idx = int(cell_idx)
    pf_idx = int(pf_idx)
    collect_lifecycle = eligibility is not None
    seed = None if random_seed is None else int(random_seed) + cell_idx * 1000 + pf_idx
    revive_cap = (
        eligibility.get("first_eligible_revive_capture_idx") if eligibility else None
    )
    score_revive = (
        collect_lifecycle
        and revive_cap is not None
        and not pd.isna(revive_cap)
    )
    zscores_by_method = _compute_pf_outlier_zscores_for_pair(
        pf_segment_df,
        master_df,
        cell_idx=cell_idx,
        pf_idx=pf_idx,
        signals_dir=signals_dir,
        capture_to_global=capture_to_global,
        optimizer_config=optimizer_config,
        window_size=int(window_size),
        method=str(method),
        layer=str(layer),
        n_random_windows=int(n_windows) if collect_windows else 0,
        random_seed=seed,
        include_single_lifecycle_events=collect_lifecycle,
        first_revival_capture_idx=int(revive_cap) if score_revive else None,
        score_first_revival=score_revive,
        signal_kind=signal_kind,
    )
    window_rows = (
        _window_outlier_rows_from_zscores(
            zscores_by_method,
            cell_idx=cell_idx,
            pf_idx=pf_idx,
            method=str(method),
            layer=str(layer),
            baselines=baselines,
        )
        if collect_windows
        else []
    )
    lifecycle_row = (
        _lifecycle_outlier_row_from_zscores(
            zscores_by_method,
            cell_idx=cell_idx,
            pf_idx=pf_idx,
            eligibility=eligibility,
            method=str(method),
            layer=str(layer),
        )
        if collect_lifecycle
        else None
    )
    return window_rows, lifecycle_row


def _collect_pf_multi_window_outlier_counts_for_pair(
    cell_idx: int,
    pf_idx: int,
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    signals_dir: Path | str,
    capture_to_global: pd.Series,
    optimizer_config: dict | None,
    window_size: int,
    n_windows: int,
    method: str,
    layer: str,
    baselines: tuple[str, ...],
    random_seed: int | None,
    signal_kind: str = "effective_lr",
) -> list[dict[str, object]]:
    """Score one PF and return per-window outlier-count rows."""
    window_rows, _lifecycle_row = _collect_pf_window_and_lifecycle_outlier_counts_for_pair(
        cell_idx,
        pf_idx,
        pf_segment_df,
        master_df,
        signals_dir,
        capture_to_global,
        optimizer_config,
        window_size,
        n_windows,
        method,
        layer,
        baselines,
        random_seed,
        eligibility=None,
        collect_windows=True,
        signal_kind=signal_kind,
    )
    return window_rows


def _collect_pf_multi_window_outlier_counts_task(
    task: tuple[object, ...],
) -> list[dict[str, object]]:
    """Pool task wrapper for ``imap_unordered`` progress updates."""
    return _collect_pf_multi_window_outlier_counts_for_pair(*task)


def collect_pf_multi_window_outlier_counts(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    pf_pairs: list[tuple[int, int]],
    *,
    signals_dir: Path | str,
    capture_to_global: pd.Series,
    optimizer_config: dict | None = None,
    window_size: int,
    n_windows: int = 50,
    method: OutlierMethod | str = "raw_zscore",
    layer: str = "recurrent",
    baselines: tuple[str, ...] = PF_LIFECYCLE_EFF_LR_BASELINES,
    random_seed: int | None = 42,
    show_progress: bool = True,
    n_process: int = 8,
    signal_kind: str = "effective_lr",
) -> pd.DataFrame:
    """
    Per-window outlier counts for random and dead_random multi-window events.

    For each PF, draws up to ``n_windows`` non-overlapping anchors per event type,
    scores incoming weights with ``method``, and returns one row per window with
    ``n_outliers`` equal to the sum of per-weight outlier counts across ``baselines``.

    PF pairs are processed in parallel when ``n_process > 1``.
    """
    pf_pairs = [(int(cell_idx), int(pf_idx)) for cell_idx, pf_idx in pf_pairs]
    n_pairs = len(pf_pairs)
    if n_pairs == 0:
        return pd.DataFrame()

    pool_tasks: list[tuple[object, ...]] = []
    for cell_idx, pf_idx in pf_pairs:
        cell_pf_segment_df = pf_segment_df.loc[
            (pf_segment_df["cell_idx"] == cell_idx)
            & (pf_segment_df["pf_idx"] == pf_idx)
        ]
        pool_tasks.append(
            (
                cell_idx,
                pf_idx,
                cell_pf_segment_df,
                master_df,
                str(signals_dir),
                capture_to_global,
                optimizer_config,
                int(window_size),
                int(n_windows),
                str(method),
                str(layer),
                baselines,
                random_seed,
                signal_kind,
            )
        )

    rows: list[dict[str, object]] = []
    if int(n_process) <= 1:
        if show_progress:
            pbar = tqdm(
                total=n_pairs,
                desc="PF window outlier counts",
                unit="PF",
            )
            for task in pool_tasks:
                pair_rows = _collect_pf_multi_window_outlier_counts_for_pair(*task)
                rows.extend(pair_rows)
                cell_idx, pf_idx = int(task[0]), int(task[1])
                pbar.set_postfix_str(
                    f"last: cell {cell_idx}, PF {pf_idx}",
                    refresh=True,
                )
                pbar.update(1)
            pbar.close()
        else:
            for task in pool_tasks:
                rows.extend(_collect_pf_multi_window_outlier_counts_for_pair(*task))
    else:
        with multiprocessing.Pool(processes=int(n_process)) as pool:
            result_iter = pool.imap_unordered(
                _collect_pf_multi_window_outlier_counts_task,
                pool_tasks,
            )
            if show_progress:
                pbar = tqdm(
                    total=n_pairs,
                    desc="PF window outlier counts",
                    unit="PF",
                )
                for pair_rows in result_iter:
                    rows.extend(pair_rows)
                    if pair_rows:
                        cell_idx = int(pair_rows[0]["cell_idx"])
                        pf_idx = int(pair_rows[0]["pf_idx"])
                        pbar.set_postfix_str(
                            f"last: cell {cell_idx}, PF {pf_idx}",
                            refresh=True,
                        )
                    pbar.update(1)
                pbar.close()
            else:
                for pair_rows in result_iter:
                    rows.extend(pair_rows)
    return pd.DataFrame(rows)


def _fit_power_law_mle(
    samples: np.ndarray,
    *,
    x_min: float | None = None,
) -> dict[str, float] | None:
    """
    MLE fit of continuous power law ``p(x) = (alpha-1)/x_min * (x/x_min)^{-alpha}``
    on raw samples with ``x >= x_min``.
    """
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 2:
        return None

    x_min_val = float(np.min(x)) if x_min is None else float(x_min)
    x = x[x >= x_min_val]
    if x.size < 2 or x_min_val <= 0:
        return None

    alpha = 1.0 + x.size / np.sum(np.log(x / x_min_val))
    if not np.isfinite(alpha) or alpha <= 1.0:
        return None
    pdf_normalization = (alpha - 1.0) / x_min_val
    return {
        "alpha": float(alpha),
        "x_min": float(x_min_val),
        "n_fit": int(x.size),
        "pdf_normalization": float(pdf_normalization),
    }


def _power_law_fit_count_curve(
    x: np.ndarray,
    *,
    fit: dict[str, float],
    n_samples: int,
) -> np.ndarray:
    """Expected window-count curve ``n_samples * p(x)`` from fit coefficients."""
    return float(n_samples) * _power_law_pdf(
        x,
        alpha=float(fit["alpha"]),
        x_min=float(fit["x_min"]),
    )


def _power_law_pdf(
    x: np.ndarray,
    *,
    alpha: float,
    x_min: float,
) -> np.ndarray:
    """Continuous power-law PDF ``p(x) = (alpha-1)/x_min * (x/x_min)^{-alpha}``."""
    x = np.asarray(x, dtype=np.float64)
    pdf = np.zeros_like(x, dtype=np.float64)
    mask = x >= float(x_min)
    if not np.any(mask):
        return pdf
    pdf[mask] = (alpha - 1.0) / float(x_min) * (x[mask] / float(x_min)) ** (-alpha)
    return pdf


def _power_law_cutoff_normalization(
    alpha: float,
    lambda_: float,
    x_min: float,
) -> float:
    """Normalization of ``x^{-alpha} exp(-lambda x)`` on ``[x_min, inf)``."""
    alpha = float(alpha)
    lambda_ = float(lambda_)
    x_min = float(x_min)
    if alpha <= 1.0 or x_min <= 0.0 or lambda_ < 0.0:
        return float("nan")

    def integrand(t: float) -> float:
        return float(t ** (-alpha) * np.exp(-lambda_ * t))

    if lambda_ == 0.0:
        return float(x_min ** (1.0 - alpha) / (alpha - 1.0))

    norm, _ = quad(integrand, x_min, np.inf, limit=200)
    return float(norm)


def _power_law_cutoff_pdf(
    x: np.ndarray,
    *,
    alpha: float,
    lambda_: float,
    x_min: float,
) -> np.ndarray:
    """Continuous power law with exponential cutoff ``p(x) ∝ x^{-alpha} exp(-lambda x)``."""
    x = np.asarray(x, dtype=np.float64)
    pdf = np.zeros_like(x, dtype=np.float64)
    mask = x >= float(x_min)
    if not np.any(mask):
        return pdf
    norm = _power_law_cutoff_normalization(alpha, lambda_, x_min)
    if not np.isfinite(norm) or norm <= 0.0:
        return pdf
    pdf[mask] = (
        x[mask] ** (-float(alpha))
        * np.exp(-float(lambda_) * x[mask])
        / norm
    )
    return pdf


def _fit_power_law_cutoff_mle(
    samples: np.ndarray,
    *,
    x_min: float | None = None,
) -> dict[str, float] | None:
    """
    MLE fit of ``p(x) ∝ x^{-alpha} exp(-lambda x)`` on raw samples with ``x >= x_min``.
    """
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0.0)]
    if x.size < 3:
        return None

    x_min_val = float(np.min(x)) if x_min is None else float(x_min)
    x = x[x >= x_min_val]
    if x.size < 3 or x_min_val <= 0.0:
        return None

    power_init = _fit_power_law_mle(x, x_min=x_min_val)
    alpha0 = float(power_init["alpha"]) if power_init is not None else 2.5
    lambda0 = max(1.0 / float(x.mean()), 1e-6)

    def neg_log_likelihood(params: np.ndarray) -> float:
        alpha = float(params[0])
        lambda_ = float(params[1])
        if alpha <= 1.0 or lambda_ < 0.0:
            return 1e300
        norm = _power_law_cutoff_normalization(alpha, lambda_, x_min_val)
        if not np.isfinite(norm) or norm <= 0.0:
            return 1e300
        log_likelihood = (
            x.size * (-np.log(norm))
            - alpha * float(np.sum(np.log(x)))
            - lambda_ * float(np.sum(x))
        )
        return -log_likelihood

    result = minimize(
        neg_log_likelihood,
        x0=np.asarray([alpha0, lambda0], dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(1.001, 50.0), (0.0, None)],
    )
    if not result.success:
        return None

    alpha = float(result.x[0])
    lambda_ = float(result.x[1])
    norm = _power_law_cutoff_normalization(alpha, lambda_, x_min_val)
    if not np.isfinite(alpha) or not np.isfinite(lambda_) or not np.isfinite(norm):
        return None
    if alpha <= 1.0 or lambda_ < 0.0 or norm <= 0.0:
        return None

    return {
        "alpha": alpha,
        "lambda": lambda_,
        "x_min": float(x_min_val),
        "n_fit": int(x.size),
        "pdf_normalization": float(1.0 / norm),
    }


def _lognormal_pdf(
    x: np.ndarray,
    *,
    mu: float,
    sigma: float,
) -> np.ndarray:
    """Lognormal PDF on positive ``x``."""
    x = np.asarray(x, dtype=np.float64)
    pdf = np.zeros_like(x, dtype=np.float64)
    mask = x > 0.0
    if not np.any(mask):
        return pdf
    z = (np.log(x[mask]) - float(mu)) / float(sigma)
    pdf[mask] = (
        np.exp(-0.5 * z * z)
        / (x[mask] * float(sigma) * np.sqrt(2.0 * np.pi))
    )
    return pdf


def _exponential_pdf(
    x: np.ndarray,
    *,
    rate: float,
) -> np.ndarray:
    """Exponential PDF on non-negative ``x``."""
    x = np.asarray(x, dtype=np.float64)
    pdf = np.zeros_like(x, dtype=np.float64)
    mask = x >= 0.0
    pdf[mask] = float(rate) * np.exp(-float(rate) * x[mask])
    return pdf


def _poisson_pmf(
    k: np.ndarray,
    *,
    mu: float,
) -> np.ndarray:
    """Poisson PMF on non-negative integer counts."""
    k_arr = np.asarray(k, dtype=np.float64)
    pmf = np.zeros_like(k_arr, dtype=np.float64)
    k_int = np.rint(k_arr).astype(np.int64)
    mask = (k_int >= 0) & (np.abs(k_arr - k_int) <= 1e-9)
    if not np.any(mask):
        return pmf
    mu = float(mu)
    if mu <= 0.0:
        return pmf
    counts = k_int[mask]
    pmf[mask] = np.exp(counts * np.log(mu) - mu - gammaln(counts + 1.0))
    return pmf


def _fit_poisson_mle(samples: np.ndarray) -> dict[str, float] | None:
    """MLE fit of Poisson distribution on non-negative integer counts."""
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x) & (x >= 0.0)]
    if x.size < 1:
        return None
    counts = np.rint(x).astype(np.int64)
    if np.any(np.abs(x - counts) > 1e-9):
        return None
    mu = float(counts.mean())
    if not np.isfinite(mu) or mu <= 0.0:
        return None
    return {
        "mu": mu,
        "n_fit": int(x.size),
    }


def _geometric_pmf(
    k: np.ndarray,
    *,
    p: float,
) -> np.ndarray:
    """0-indexed geometric PMF ``P(X=k) = (1-p)^k p`` on non-negative integers."""
    k_arr = np.asarray(k, dtype=np.float64)
    pmf = np.zeros_like(k_arr, dtype=np.float64)
    k_int = np.rint(k_arr).astype(np.int64)
    mask = (k_int >= 0) & (np.abs(k_arr - k_int) <= 1e-9)
    if not np.any(mask):
        return pmf
    p = float(p)
    if p <= 0.0 or p > 1.0:
        return pmf
    q = 1.0 - p
    counts = k_int[mask]
    pmf[mask] = (q ** counts) * p
    return pmf


def _fit_geometric_mle(samples: np.ndarray) -> dict[str, float] | None:
    """MLE fit of 0-indexed geometric distribution on non-negative integer counts."""
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x) & (x >= 0.0)]
    if x.size < 1:
        return None
    counts = np.rint(x).astype(np.int64)
    if np.any(np.abs(x - counts) > 1e-9):
        return None
    mean_k = float(counts.mean())
    if not np.isfinite(mean_k) or mean_k < 0.0:
        return None
    p = 1.0 / (1.0 + mean_k)
    if not np.isfinite(p) or p <= 0.0 or p > 1.0:
        return None
    return {
        "p": float(p),
        "mean": mean_k,
        "n_fit": int(x.size),
    }


def _negbinom_pmf(
    k: np.ndarray,
    *,
    n: float,
    p: float,
) -> np.ndarray:
    """Negative-binomial PMF (scipy ``nbinom`` parameterization) on non-negative integers."""
    k_arr = np.asarray(k, dtype=np.float64)
    pmf = np.zeros_like(k_arr, dtype=np.float64)
    k_int = np.rint(k_arr).astype(np.int64)
    mask = (k_int >= 0) & (np.abs(k_arr - k_int) <= 1e-9)
    if not np.any(mask):
        return pmf
    n = float(n)
    p = float(p)
    if n <= 0.0 or p <= 0.0 or p >= 1.0:
        return pmf
    pmf[mask] = nbinom.pmf(k_int[mask], n, p)
    return pmf


def _fit_negbinom_mle(samples: np.ndarray) -> dict[str, float] | None:
    """MLE fit of negative-binomial distribution on non-negative integer counts."""
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x) & (x >= 0.0)]
    if x.size < 2:
        return None
    counts = np.rint(x).astype(np.int64)
    if np.any(np.abs(x - counts) > 1e-9):
        return None
    sample_var = float(counts.var(ddof=0))
    if sample_var <= 0.0:
        return None

    mean_k = float(counts.mean())
    if sample_var > mean_k and mean_k > 0.0:
        p0 = mean_k / sample_var
        n0 = mean_k * p0 / max(1.0 - p0, 1e-6)
    else:
        p0 = 0.5
        n0 = max(mean_k, 1.0)
    n0 = float(np.clip(n0, 1e-3, 1e6))
    p0 = float(np.clip(p0, 1e-3, 1.0 - 1e-3))

    def neg_log_likelihood(params: np.ndarray) -> float:
        n_param = float(np.exp(params[0]))
        p_param = float(1.0 / (1.0 + np.exp(-params[1])))
        if n_param <= 0.0 or p_param <= 0.0 or p_param >= 1.0:
            return 1e300
        log_pmf = (
            gammaln(counts.astype(np.float64) + n_param)
            - gammaln(counts.astype(np.float64) + 1.0)
            - gammaln(n_param)
            + n_param * np.log(p_param)
            + counts.astype(np.float64) * np.log(1.0 - p_param)
        )
        if not np.all(np.isfinite(log_pmf)):
            return 1e300
        return float(-np.sum(log_pmf))

    result = minimize(
        neg_log_likelihood,
        x0=np.asarray([np.log(n0), np.log(p0 / (1.0 - p0))], dtype=np.float64),
        method="L-BFGS-B",
    )
    if not result.success:
        return None

    n_param = float(np.exp(result.x[0]))
    p_param = float(1.0 / (1.0 + np.exp(-result.x[1])))
    if (
        not np.isfinite(n_param)
        or not np.isfinite(p_param)
        or n_param <= 0.0
        or p_param <= 0.0
        or p_param >= 1.0
    ):
        return None
    mean_fit = n_param * (1.0 - p_param) / p_param
    var_fit = n_param * (1.0 - p_param) / (p_param ** 2)
    return {
        "n": n_param,
        "p": p_param,
        "mean": float(mean_fit),
        "var": float(var_fit),
        "n_fit": int(x.size),
    }


def _zipf_normalization(alpha: float, k_min: float) -> float:
    """Normalization ``sum_{k=k_min}^inf k^{-alpha}`` for discrete Zipf / zeta PMF."""
    alpha = float(alpha)
    k_min = float(k_min)
    if alpha <= 1.0 or k_min <= 0.0:
        return float("nan")
    return float(zeta(alpha, k_min))


def _zipf_pmf(
    k: np.ndarray,
    *,
    alpha: float,
    k_min: float,
) -> np.ndarray:
    """Discrete Zipf PMF ``P(X=k) = k^{-alpha} / zeta(alpha, k_min)`` on integers ``k >= k_min``."""
    k_arr = np.asarray(k, dtype=np.float64)
    pmf = np.zeros_like(k_arr, dtype=np.float64)
    k_int = np.rint(k_arr).astype(np.int64)
    k_min_val = float(k_min)
    mask = (
        (k_int >= int(np.ceil(k_min_val)))
        & (np.abs(k_arr - k_int) <= 1e-9)
    )
    if not np.any(mask):
        return pmf
    norm = _zipf_normalization(alpha, k_min_val)
    if not np.isfinite(norm) or norm <= 0.0:
        return pmf
    counts = k_int[mask].astype(np.float64)
    pmf[mask] = counts ** (-float(alpha)) / norm
    return pmf


def _fit_zipf_mle(
    samples: np.ndarray,
    *,
    k_min: float | None = None,
) -> dict[str, float] | None:
    """
    MLE fit of discrete Zipf ``P(k) ∝ k^{-alpha}`` on positive integer counts
    with ``k >= k_min``.
    """
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0.0)]
    if x.size < 2:
        return None
    counts = np.rint(x).astype(np.int64)
    if np.any(np.abs(x - counts) > 1e-9):
        return None

    k_min_val = int(np.min(counts)) if k_min is None else int(k_min)
    counts = counts[counts >= k_min_val]
    if counts.size < 2 or k_min_val <= 0:
        return None

    alpha = 1.0 + counts.size / np.sum(np.log(counts / float(k_min_val)))
    if not np.isfinite(alpha) or alpha <= 1.0:
        return None
    norm = _zipf_normalization(alpha, float(k_min_val))
    if not np.isfinite(norm) or norm <= 0.0:
        return None
    return {
        "alpha": float(alpha),
        "k_min": float(k_min_val),
        "n_fit": int(counts.size),
        "pmf_normalization": float(1.0 / norm),
    }


def _fit_lognormal_mle(samples: np.ndarray) -> dict[str, float] | None:
    """MLE fit of lognormal distribution on strictly positive samples."""
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0.0)]
    if x.size < 2:
        return None
    log_x = np.log(x)
    mu = float(log_x.mean())
    sigma = float(log_x.std(ddof=0))
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0.0:
        return None
    return {
        "mu": mu,
        "sigma": sigma,
        "n_fit": int(x.size),
    }


def _fit_exponential_mle(samples: np.ndarray) -> dict[str, float] | None:
    """MLE fit of exponential distribution on strictly positive samples."""
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0.0)]
    if x.size < 1:
        return None
    mean_x = float(x.mean())
    if not np.isfinite(mean_x) or mean_x <= 0.0:
        return None
    rate = 1.0 / mean_x
    return {
        "rate": float(rate),
        "n_fit": int(x.size),
    }


def _weibull_cdf_scalar(x: float, *, shape: float, scale: float) -> float:
    """Weibull CDF with ``loc=0`` (``scipy.stats.weibull_min`` parameterization)."""
    if x <= 0.0:
        return 0.0
    shape = float(shape)
    scale = float(scale)
    if shape <= 0.0 or scale <= 0.0:
        return 0.0
    return float(weibull_min.cdf(x, shape, loc=0.0, scale=scale))


def _fit_weibull_mle(samples: np.ndarray) -> dict[str, float] | None:
    """MLE fit of Weibull distribution on strictly positive samples."""
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0.0)]
    if x.size < 2:
        return None
    try:
        shape, _loc, scale = weibull_min.fit(x, floc=0.0)
    except (ValueError, FloatingPointError, RuntimeError):
        return None
    shape = float(shape)
    scale = float(scale)
    if not np.isfinite(shape) or not np.isfinite(scale):
        return None
    if shape <= 0.0 or scale <= 0.0:
        return None
    return {
        "shape": shape,
        "scale": scale,
        "n_fit": int(x.size),
    }


def _model_log_likelihood_from_pdf(
    samples: np.ndarray,
    pdf_values: np.ndarray,
) -> float:
    pdf_values = np.maximum(np.asarray(pdf_values, dtype=np.float64), 1e-300)
    return float(np.sum(np.log(pdf_values)))


def _information_criteria(
    log_likelihood: float,
    *,
    n_params: int,
    n_samples: int,
) -> tuple[float, float]:
    n_params = int(n_params)
    n_samples = max(int(n_samples), 1)
    aic = 2.0 * n_params - 2.0 * float(log_likelihood)
    bic = n_params * np.log(n_samples) - 2.0 * float(log_likelihood)
    return float(aic), float(bic)


def _integer_outlier_bincount(samples: np.ndarray) -> dict[int, int]:
    """Return ``{x: n_x}`` for non-negative integer outlier counts."""
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x) & (x >= 0.0)]
    if x.size == 0:
        return {}
    counts = np.rint(x).astype(np.int64)
    if np.any(np.abs(x - counts) > 1e-9):
        raise ValueError("Outlier counts must be non-negative integers")
    uniq, freq = np.unique(counts, return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq, freq, strict=True)}


def _log_likelihood_from_bincount(
    bincount: dict[int, int],
    pmf_fn,
) -> float:
    """``sum_x n_x log P(X=x)`` for a discrete PMF."""
    if not bincount:
        return float("-inf")
    log_likelihood = 0.0
    for x, n_x in bincount.items():
        prob = float(pmf_fn(int(x)))
        if prob <= 0.0 or not np.isfinite(prob):
            return float("-inf")
        log_likelihood += float(n_x) * np.log(prob)
    return float(log_likelihood)


def _zi_log_likelihood_from_bincount(
    bincount: dict[int, int],
    *,
    pi: float,
    positive_pmf_fn,
) -> float:
    """
    ``N0 log pi + N+ log(1-pi) + sum_{x>0} n_x log P_positive(x)`` on integer bins.
    """
    if not bincount:
        return float("-inf")
    n_zero = int(bincount.get(0, 0))log_likelihood
            return float("-inf")
        log_likelihood += float(n_x) * np.log(prob)
    return float(log_likelihood)


def _fit_metrics_from_bincount(
    coefficients: dict[str, float],
    bincount: dict[int, int],
    *,
    n_params: int,
    log_likelihood: float,
) -> dict[str, object]:
    n_total = int(sum(bincount.values()))
    aic, bic = _information_criteria(
        log_likelihood,
        n_params=int(n_params),
        n_samples=n_total,
    )
    return {
        **coefficients,
        "log_likelihood": float(log_likelihood),
        "aic": float(aic),
        "bic": float(bic),
        "n_params": int(n_params),
        "n_fit": n_total,
        "x_max": float(max(bincount) if bincount else 0),
    }


def _exponential_cdf_scalar(x: float, *, rate: float) -> float:
    x = max(float(x), 0.0)
    return float(1.0 - np.exp(-float(rate) * x))


def _lognormal_cdf_scalar(x: float, *, mu: float, sigma: float) -> float:
    if x <= 0.0:
        return 0.0
    return float(lognorm.cdf(x, s=float(sigma), scale=float(np.exp(mu))))


def _power_law_cdf_scalar(x: float, *, alpha: float, x_min: float) -> float:
    x = float(x)
    x_min = float(x_min)
    if x < x_min:
        return 0.0
    return float(1.0 - (x / x_min) ** (1.0 - float(alpha)))


def _power_law_cutoff_cdf_scalar(
    x: float,
    *,
    alpha: float,
    lambda_: float,
    x_min: float,
) -> float:
    x = float(x)
    x_min = float(x_min)
    if x < x_min:
        return 0.0
    norm = _power_law_cutoff_normalization(alpha, lambda_, x_min)
    if not np.isfinite(norm) or norm <= 0.0:
        return 0.0

    def integrand(t: float) -> float:
        return float(t ** (-float(alpha)) * np.exp(-float(lambda_) * t))

    mass, _ = quad(integrand, x_min, x, limit=200)
    return float(np.clip(mass / norm, 0.0, 1.0))


def _discretized_bin_pmf_from_cdf(x: int, cdf_fn) -> float:
    """``P(X=x) = F(x+1) - F(x)`` for integer bin ``[x, x+1)``."""
    if x < 0:
        return 0.0
    prob = float(cdf_fn(float(x + 1)) - cdf_fn(float(x)))
    return max(prob, 0.0)


def _positive_tail_discretized_mass(
    x: int,
    model: str,
    fit: dict[str, float],
) -> float:
    """Unnormalized discrete mass for the positive-tail component at integer ``x``."""
    if x <= 0:
        return 0.0
    if model == "exponential":
        return _discretized_bin_pmf_from_cdf(
            x,
            lambda t: _exponential_cdf_scalar(t, rate=float(fit["rate"])),
        )
    if model == "lognormal":
        return _discretized_bin_pmf_from_cdf(
            x,
            lambda t: _lognormal_cdf_scalar(t, mu=float(fit["mu"]), sigma=float(fit["sigma"])),
        )
    if model == "weibull":
        return _discretized_bin_pmf_from_cdf(
            x,
            lambda t: _weibull_cdf_scalar(
                t,
                shape=float(fit["shape"]),
                scale=float(fit["scale"]),
            ),
        )
    if model == "power_law":
        k_min = int(np.ceil(float(fit["x_min"])))
        if x < k_min:
            return 0.0
        return _discretized_bin_pmf_from_cdf(
            x,
            lambda t: _power_law_cdf_scalar(
                t,
                alpha=float(fit["alpha"]),
                x_min=float(fit["x_min"]),
            ),
        )
    if model == "power_law_cutoff":
        k_min = int(np.ceil(float(fit["x_min"])))
        if x < k_min:
            return 0.0
        return _discretized_bin_pmf_from_cdf(
            x,
            lambda t: _power_law_cutoff_cdf_scalar(
                t,
                alpha=float(fit["alpha"]),
                lambda_=float(fit["lambda"]),
                x_min=float(fit["x_min"]),
            ),
        )
    if model == "zipf":
        k_min = int(np.ceil(float(fit["k_min"])))
        if x < k_min:
            return 0.0
        return float(
            _zipf_pmf(
                np.asarray([float(x)], dtype=np.float64),
                alpha=float(fit["alpha"]),
                k_min=float(fit["k_min"]),
            )[0]
        )
    if model == "geometric":
        return float(
            _geometric_pmf(
                np.asarray([float(x)], dtype=np.float64),
                p=float(fit["p"]),
            )[0]
        )
    if model == "negbinom":
        return float(
            _negbinom_pmf(
                np.asarray([float(x)], dtype=np.float64),
                n=float(fit["n"]),
                p=float(fit["p"]),
            )[0]
        )
    raise ValueError(f"Unexpected positive-tail model {model!r}")


_OUTLIER_POSITIVE_PMF_COEFF_KEYS: dict[str, tuple[str, ...]] = {
    "exponential": ("rate",),
    "lognormal": ("mu", "sigma"),
    "weibull": ("shape", "scale"),
    "power_law": ("alpha", "x_min"),
    "power_law_cutoff": ("alpha", "lambda", "x_min"),
    "zipf": ("alpha", "k_min"),
    "geometric": ("p",),
    "negbinom": ("n", "p"),
}


def _outlier_positive_pmf_cache_key(
    model: str,
    fit: dict[str, float],
    *,
    x_max: int,
) -> tuple[object, ...]:
    coeff_keys = _OUTLIER_POSITIVE_PMF_COEFF_KEYS.get(model, ())
    coeff_items = tuple(
        (key, float(fit[key]))
        for key in coeff_keys
        if key in fit and np.isfinite(float(fit[key]))
    )
    return (model, int(x_max), coeff_items)


@lru_cache(maxsize=256)
def _positive_tail_discretized_pmf_table_cached(
    cache_key: tuple[object, ...],
) -> tuple[float, ...]:
    model = str(cache_key[0])
    x_max = int(cache_key[1])
    fit = {str(key): float(value) for key, value in cache_key[2]}
    if x_max <= 0:
        return tuple()

    masses = [
        _positive_tail_discretized_mass(k, model, fit)
        for k in range(1, x_max + 1)
    ]
    norm = float(np.sum(masses))
    if norm <= 0.0:
        return tuple(0.0 for _ in masses)
    return tuple(float(m / norm) for m in masses)


def _positive_tail_discretized_pmf_table(
    model: str,
    fit: dict[str, float],
    *,
    x_max: int,
) -> np.ndarray:
    """Normalized ``P_positive(x)`` for ``x = 1, …, x_max`` (cached)."""
    cache_key = _outlier_positive_pmf_cache_key(model, fit, x_max=int(x_max))
    return np.asarray(
        _positive_tail_discretized_pmf_table_cached(cache_key),
        dtype=np.float64,
    )


def _positive_tail_discretized_pmf(
    x: int,
    model: str,
    fit: dict[str, float],
    *,
    x_max: int,
    pmf_table: np.ndarray | None = None,
) -> float:
    """``P_positive(x)`` normalized over integers ``x >= 1`` (or ``x >= k_min``)."""
    if x <= 0:
        return 0.0
    table = pmf_table
    if table is None:
        table = _positive_tail_discretized_pmf_table(model, fit, x_max=int(x_max))
    if x > int(table.size):
        return 0.0
    return float(table[x - 1])


def _zi_log_likelihood_from_positive_pmf_table(
    bincount: dict[int, int],
    *,
    pi: float,
    positive_pmf_table: np.ndarray,
) -> float:
    """Fast ZI log-likelihood using a precomputed positive-tail PMF table."""
    if not bincount:
        return float("-inf")
    n_zero = int(bincount.get(0, 0))
    n_total = int(sum(bincount.values()))
    n_pos = n_total - n_zero
    pi = float(np.clip(pi, 1e-300, 1.0 - 1e-300))
    log_likelihood = float(n_zero) * np.log(pi)
    if n_pos > 0:
        log_likelihood += float(n_pos) * np.log(1.0 - pi)
    for x, n_x in bincount.items():
        x_int = int(x)
        if x_int <= 0:
            continue
        if x_int > int(positive_pmf_table.size):
            return float("-inf")
        prob = float(positive_pmf_table[x_int - 1])
        if prob <= 0.0 or not np.isfinite(prob):
            return float("-inf")
        log_likelihood += float(n_x) * np.log(prob)
    return float(log_likelihood)


def _full_discretized_pmf(
    x: int,
    model: str,
    fit: dict[str, float],
    *,
    x_max: int,
) -> float:
    """Full discrete PMF on ``x=0,1,...`` via bin masses ``F(x+1)-F(x)``."""
    if x < 0 or x > int(x_max):
        return 0.0
    if model == "exponential":
        return _discretized_bin_pmf_from_cdf(
            x,
            lambda t: _exponential_cdf_scalar(t, rate=float(fit["rate"])),
        )
    if model == "lognormal":
        return _discretized_bin_pmf_from_cdf(
            x,
            lambda t: _lognormal_cdf_scalar(t, mu=float(fit["mu"]), sigma=float(fit["sigma"])),
        )
    if model == "weibull":
        return _discretized_bin_pmf_from_cdf(
            x,
            lambda t: _weibull_cdf_scalar(
                t,
                shape=float(fit["shape"]),
                scale=float(fit["scale"]),
            ),
        )
    if model == "poisson":
        return float(
            _poisson_pmf(np.asarray([float(x)], dtype=np.float64), mu=float(fit["mu"]))[0]
        )
    if model == "geometric":
        return float(
            _geometric_pmf(np.asarray([float(x)], dtype=np.float64), p=float(fit["p"]))[0]
        )
    if model == "negbinom":
        return float(
            _negbinom_pmf(
                np.asarray([float(x)], dtype=np.float64),
                n=float(fit["n"]),
                p=float(fit["p"]),
            )[0]
        )
    raise ValueError(f"Unexpected full discrete model {model!r}")


def _zero_inflated_discrete_pmf(
    x: int,
    model: str,
    fit: dict[str, float],
    *,
    pi: float,
    x_max: int,
    pmf_table: np.ndarray | None = None,
) -> float:
    if x == 0:
        return float(pi)
    if x < 0:
        return 0.0
    table = pmf_table
    if table is None:
        stored = fit.get("positive_pmf_table")
        if isinstance(stored, np.ndarray):
            table = stored
    return float(1.0 - pi) * _positive_tail_discretized_pmf(
        x,
        model,
        fit,
        x_max=int(x_max),
        pmf_table=table,
    )


def _discrete_pmf_probability(
    x: int,
    model: str,
    fit: dict[str, float],
) -> float:
    """Evaluate the final discrete ``P(X=x)`` for a stored fit result."""
    x_max = int(fit.get("x_max", max(int(x), 1)))
    base_model, zero_inflated = _resolve_outlier_count_model(model)
    if zero_inflated:
        return _zero_inflated_discrete_pmf(
            int(x),
            base_model,
            fit,
            pi=float(fit["pi"]),
            x_max=x_max,
        )
    return _full_discretized_pmf(int(x), base_model, fit, x_max=x_max)


def _outlier_count_observed_expected_table(
    bincount: dict[int, int],
    model: str,
    fit: dict[str, float],
    *,
    x_max: int | None = None,
) -> pd.DataFrame:
    """Observed vs expected counts ``n_x`` and ``N * P(X=x)`` for integer bins."""
    if not bincount:
        return pd.DataFrame(
            columns=["x", "observed", "expected", "residual", "model", "model_label"]
        )
    x_max_val = int(x_max if x_max is not None else fit.get("x_max", max(bincount)))
    n_total = int(sum(bincount.values()))
    rows: list[dict[str, object]] = []
    for x in range(0, x_max_val + 1):
        observed = int(bincount.get(x, 0))
        expected = float(n_total) * _discrete_pmf_probability(x, model, fit)
        rows.append(
            {
                "x": int(x),
                "observed": observed,
                "expected": expected,
                "residual": float(observed) - expected,
                "model": model,
                "model_label": OUTLIER_COUNT_FIT_MODEL_LABELS.get(model, model),
            }
        )
    return pd.DataFrame(rows)


def plot_pf_multi_window_outlier_count_gof(
    counts_df: pd.DataFrame,
    fit_results_by_event: dict[str, dict[str, object]],
    *,
    event_order: tuple[str, ...] = PF_LIFECYCLE_EFF_LR_MULTI_WINDOW_EVENTS,
    model: str = "negbinom",
    x_max: int = 50,
):
    """
    Observed vs expected count plot and residual plot for one fitted model.

    Returns ``(fig, axes, gof_df_by_event)``.
    """
    import matplotlib.pyplot as plt

    present_events = [
        event_name
        for event_name in event_order
        if event_name in fit_results_by_event
        and isinstance((fit_results_by_event[event_name] or {}).get(model), dict)
    ]
    if not present_events:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.set_axis_off()
        ax.set_title(f"No {model!r} fit results to plot")
        fig.tight_layout()
        return fig, ax, {}

    fig, axes = plt.subplots(
        len(present_events),
        2,
        figsize=(11.0, 3.8 * len(present_events)),
        squeeze=False,
    )
    gof_df_by_event: dict[str, pd.DataFrame] = {}
    model_label = OUTLIER_COUNT_FIT_MODEL_LABELS.get(model, model)

    for row_i, event_name in enumerate(present_events):
        ev = counts_df.loc[counts_df["event_name"] == event_name]
        fit = fit_results_by_event[event_name][model]
        bincount = _integer_outlier_bincount(ev["n_outliers"].to_numpy(dtype=np.float64))
        gof_df = _outlier_count_observed_expected_table(
            bincount,
            model,
            fit,
            x_max=int(x_max),
        )
        gof_df_by_event[event_name] = gof_df
        event_label = PF_LIFECYCLE_EFF_LR_EVENT_LABELS.get(event_name, event_name)
        ax_obs = axes[row_i, 0]
        ax_res = axes[row_i, 1]
        ax_obs.plot(
            gof_df["x"],
            gof_df["observed"],
            "o",
            color="C0",
            label="observed",
            zorder=3,
        )
        ax_obs.plot(
            gof_df["x"],
            gof_df["expected"],
            "-",
            color="C3",
            label="expected",
            zorder=2,
        )
        ax_obs.set_title(f"{event_label}: observed vs expected ({model_label})")
        ax_obs.set_xlabel("Outliers per window")
        ax_obs.set_ylabel("Count")
        ax_obs.grid(True, axis="y", alpha=0.25)
        ax_obs.legend(frameon=False, fontsize=9)

        ax_res.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax_res.plot(gof_df["x"], gof_df["residual"], "o-", color="C2")
        ax_res.set_title(f"{event_label}: residuals (observed − expected)")
        ax_res.set_xlabel("Outliers per window")
        ax_res.set_ylabel("Residual")
        ax_res.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    return fig, axes, gof_df_by_event


OUTLIER_COUNT_TAIL_MODELS: tuple[str, ...] = (
    "power_law",
    "power_law_cutoff",
    "lognormal",
    "exponential",
    "weibull",
    "zipf",
)
OUTLIER_COUNT_TAIL_MODELS_ZI: tuple[str, ...] = tuple(
    f"{model}_zi" for model in OUTLIER_COUNT_TAIL_MODELS
)
OUTLIER_COUNT_DISCRETE_ZI_MODELS: tuple[str, ...] = (
    "geometric_zi",
    "negbinom_zi",
)
OUTLIER_COUNT_ALL_ZI_MODELS: tuple[str, ...] = (
    OUTLIER_COUNT_TAIL_MODELS_ZI + OUTLIER_COUNT_DISCRETE_ZI_MODELS
)
OUTLIER_COUNT_COMPARABLE_MODELS: tuple[str, ...] = (
    "exponential",
    "weibull",
    "lognormal",
    "poisson",
    "geometric",
    "negbinom",
)
OUTLIER_COUNT_FIT_MODEL_LABELS: dict[str, str] = {
    "power_law": "power law",
    "power_law_cutoff": "power law + exp cutoff",
    "lognormal": "lognormal",
    "exponential": "exponential",
    "weibull": "Weibull",
    "poisson": "Poisson",
    "geometric": "geometric",
    "negbinom": "neg. binomial",
    "geometric_zi": "geometric + π",
    "negbinom_zi": "neg. binomial + π",
    "zipf": "Zipf",
    "power_law_zi": "power law + π",
    "power_law_cutoff_zi": "power law + exp cutoff + π",
    "lognormal_zi": "lognormal + π",
    "exponential_zi": "exponential + π",
    "weibull_zi": "Weibull + π",
    "zipf_zi": "Zipf + π",
}


def _resolve_outlier_count_model(model: str) -> tuple[str, bool]:
    """Return ``(base_model, zero_inflated)`` for plain or ``*_zi`` model keys."""
    if str(model).endswith("_zi"):
        return str(model)[:-3], True
    return str(model), False


def _attach_zero_inflated_tail_fits(
    bincount: dict[int, int],
    results: dict[str, dict[str, object] | None],
    *,
    x_max: int,
) -> None:
    """Add ``*_zi`` entries using discrete bins and ``pi = n_0 / N``."""
    if not bincount:
        for model in OUTLIER_COUNT_TAIL_MODELS:
            results[f"{model}_zi"] = None
        return

    n_zero = int(bincount.get(0, 0))
    n_total = int(sum(bincount.values()))
    pi = float(n_zero / n_total)

    for model in OUTLIER_COUNT_TAIL_MODELS:
        tail_fit = results.get(model)
        if not isinstance(tail_fit, dict):
            results[f"{model}_zi"] = None
            continue
        tail_n_params = int(tail_fit["n_params"])
        positive_pmf_table = _positive_tail_discretized_pmf_table(
            model,
            tail_fit,
            x_max=int(x_max),
        )
        log_likelihood = _zi_log_likelihood_from_positive_pmf_table(
            bincount,
            pi=pi,
            positive_pmf_table=positive_pmf_table,
        )
        results[f"{model}_zi"] = _fit_metrics_from_bincount(
            {
                **tail_fit,
                "pi": pi,
                "n_zero": n_zero,
                "tail_model": model,
                "zero_inflated": True,
                "x_max": float(x_max),
                "positive_pmf_table": positive_pmf_table,
            },
            bincount,
            n_params=tail_n_params + 1,
            log_likelihood=log_likelihood,
        )


def _attach_zero_inflated_discrete_fits(
    bincount: dict[int, int],
    results: dict[str, dict[str, object] | None],
    *,
    x_max: int,
    base_models: tuple[str, ...] = ("geometric", "negbinom"),
) -> None:
    """Add ``geometric_zi`` / ``negbinom_zi`` using conditional discrete tails."""
    if not bincount:
        for model in base_models:
            results[f"{model}_zi"] = None
        return

    n_zero = int(bincount.get(0, 0))
    pi = float(n_zero / sum(bincount.values()))

    for model in base_models:
        base_fit = results.get(model)
        if not isinstance(base_fit, dict):
            results[f"{model}_zi"] = None
            continue
        tail_n_params = int(base_fit["n_params"])
        positive_pmf_table = _positive_tail_discretized_pmf_table(
            model,
            base_fit,
            x_max=int(x_max),
        )
        log_likelihood = _zi_log_likelihood_from_positive_pmf_table(
            bincount,
            pi=pi,
            positive_pmf_table=positive_pmf_table,
        )
        results[f"{model}_zi"] = _fit_metrics_from_bincount(
            {
                **base_fit,
                "pi": pi,
                "n_zero": n_zero,
                "tail_model": model,
                "zero_inflated": True,
                "x_max": float(x_max),
                "positive_pmf_table": positive_pmf_table,
            },
            bincount,
            n_params=tail_n_params + 1,
            log_likelihood=log_likelihood,
        )


def _tail_fit_n_params(model: str) -> int:
    """Number of estimated parameters for tail / count models."""
    if model in {
        "power_law",
        "power_law_cutoff",
        "lognormal",
        "weibull",
        "negbinom",
        "zipf",
    }:
        return 2
    if model in {"exponential", "poisson", "geometric"}:
        return 1
    raise ValueError(f"Unexpected tail model {model!r}")


def _param_only_tail_fit(
    fit: dict[str, float],
    *,
    model: str,
    x_max: int,
) -> dict[str, object]:
    """Store positive-sample coefficients used by zero-inflated discrete fits."""
    return {
        **fit,
        "x_max": float(x_max),
        "n_params": _tail_fit_n_params(model),
        "param_only": True,
    }


def _compare_outlier_count_distribution_fits(
    samples: np.ndarray,
) -> dict[str, dict[str, object] | None]:
    """
    Fit outlier-count models and evaluate discrete log-likelihoods on the same
    integer bins for every comparable model (``n_fit = N`` for all).

    Continuous candidates (exponential, lognormal) use bin masses
    ``P(X=x) = F(x+1) - F(x)``. Tail-only fits (power law, Zipf, …) store
    positive-sample coefficients for their ``*_zi`` counterparts.
    """
    raw = np.asarray(samples, dtype=np.float64)
    raw = raw[np.isfinite(raw)]
    bincount = _integer_outlier_bincount(raw)
    x_pos = raw[raw > 0.0]
    x_max = int(max(bincount) if bincount else 0)

    empty_results: dict[str, dict[str, object] | None] = {
        "power_law": None,
        "power_law_cutoff": None,
        "lognormal": None,
        "exponential": None,
        "weibull": None,
        "poisson": None,
        "geometric": None,
        "negbinom": None,
        "zipf": None,
        "n_samples": 0,
    }
    if not bincount:
        _attach_zero_inflated_tail_fits(bincount, empty_results, x_max=x_max)
        _attach_zero_inflated_discrete_fits(bincount, empty_results, x_max=x_max)
        return empty_results

    results: dict[str, dict[str, object] | None] = {
        "power_law": None,
        "power_law_cutoff": None,
        "lognormal": None,
        "exponential": None,
        "weibull": None,
        "poisson": None,
        "geometric": None,
        "negbinom": None,
        "zipf": None,
        "n_samples": int(sum(bincount.values())),
    }

    power_fit = _fit_power_law_mle(x_pos)
    if power_fit is not None:
        results["power_law"] = _param_only_tail_fit(
            power_fit,
            model="power_law",
            x_max=x_max,
        )

    power_cutoff_fit = _fit_power_law_cutoff_mle(x_pos)
    if power_cutoff_fit is not None:
        results["power_law_cutoff"] = _param_only_tail_fit(
            power_cutoff_fit,
            model="power_law_cutoff",
            x_max=x_max,
        )

    zipf_fit = _fit_zipf_mle(x_pos)
    if zipf_fit is not None:
        results["zipf"] = _param_only_tail_fit(
            zipf_fit,
            model="zipf",
            x_max=x_max,
        )

    lognormal_fit = _fit_lognormal_mle(x_pos)
    if lognormal_fit is not None:
        fit = {**lognormal_fit, "x_max": float(x_max)}
        log_likelihood = _log_likelihood_from_bincount(
            bincount,
            lambda x, _fit=fit: _full_discretized_pmf(
                x,
                "lognormal",
                _fit,
                x_max=x_max,
            ),
        )
        results["lognormal"] = _fit_metrics_from_bincount(
            fit,
            bincount,
            n_params=_tail_fit_n_params("lognormal"),
            log_likelihood=log_likelihood,
        )

    exponential_fit = _fit_exponential_mle(x_pos)
    if exponential_fit is not None:
        fit = {**exponential_fit, "x_max": float(x_max)}
        log_likelihood = _log_likelihood_from_bincount(
            bincount,
            lambda x, _fit=fit: _full_discretized_pmf(
                x,
                "exponential",
                _fit,
                x_max=x_max,
            ),
        )
        results["exponential"] = _fit_metrics_from_bincount(
            fit,
            bincount,
            n_params=_tail_fit_n_params("exponential"),
            log_likelihood=log_likelihood,
        )

    weibull_fit = _fit_weibull_mle(x_pos)
    if weibull_fit is not None:
        fit = {**weibull_fit, "x_max": float(x_max)}
        log_likelihood = _log_likelihood_from_bincount(
            bincount,
            lambda x, _fit=fit: _full_discretized_pmf(
                x,
                "weibull",
                _fit,
                x_max=x_max,
            ),
        )
        results["weibull"] = _fit_metrics_from_bincount(
            fit,
            bincount,
            n_params=_tail_fit_n_params("weibull"),
            log_likelihood=log_likelihood,
        )

    poisson_fit = _fit_poisson_mle(raw[raw >= 0.0])
    if poisson_fit is not None:
        fit = {**poisson_fit, "x_max": float(x_max)}
        log_likelihood = _log_likelihood_from_bincount(
            bincount,
            lambda x, _fit=fit: _full_discretized_pmf(
                x,
                "poisson",
                _fit,
                x_max=x_max,
            ),
        )
        results["poisson"] = _fit_metrics_from_bincount(
            fit,
            bincount,
            n_params=_tail_fit_n_params("poisson"),
            log_likelihood=log_likelihood,
        )

    geometric_fit = _fit_geometric_mle(raw[raw >= 0.0])
    if geometric_fit is not None:
        fit = {**geometric_fit, "x_max": float(x_max)}
        log_likelihood = _log_likelihood_from_bincount(
            bincount,
            lambda x, _fit=fit: _full_discretized_pmf(
                x,
                "geometric",
                _fit,
                x_max=x_max,
            ),
        )
        results["geometric"] = _fit_metrics_from_bincount(
            fit,
            bincount,
            n_params=_tail_fit_n_params("geometric"),
            log_likelihood=log_likelihood,
        )

    negbinom_fit = _fit_negbinom_mle(raw[raw >= 0.0])
    if negbinom_fit is not None:
        fit = {**negbinom_fit, "x_max": float(x_max)}
        log_likelihood = _log_likelihood_from_bincount(
            bincount,
            lambda x, _fit=fit: _full_discretized_pmf(
                x,
                "negbinom",
                _fit,
                x_max=x_max,
            ),
        )
        results["negbinom"] = _fit_metrics_from_bincount(
            fit,
            bincount,
            n_params=_tail_fit_n_params("negbinom"),
            log_likelihood=log_likelihood,
        )

    _attach_zero_inflated_tail_fits(bincount, results, x_max=x_max)
    _attach_zero_inflated_discrete_fits(bincount, results, x_max=x_max)
    return results


def _distribution_fit_count_curve(
    x: np.ndarray,
    *,
    model: str,
    fit: dict[str, float],
    n_samples: int,
) -> np.ndarray:
    """Expected count curve ``n_samples * P(X=x)`` from the fitted discrete PMF."""
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x, dtype=np.float64)
    for i, xi in enumerate(x):
        if not np.isfinite(xi) or xi < 0.0:
            continue
        k = int(np.rint(xi))
        if abs(float(xi) - k) > 1e-9:
            continue
        y[i] = float(n_samples) * _discrete_pmf_probability(k, model, fit)
    return y


_OUTLIER_COUNT_FIT_METRIC_KEYS = frozenset(
    {
        "log_likelihood",
        "aic",
        "bic",
        "n_params",
        "n_fit",
        "pdf_normalization",
        "pmf_normalization",
        "tail_model",
        "zero_inflated",
        "param_only",
        "x_max",
        "positive_pmf_table",
    }
)
OUTLIER_COUNT_FIT_PARAM_COLUMNS: tuple[str, ...] = (
    "pi",
    "n_zero",
    "alpha",
    "x_min",
    "lambda",
    "mu",
    "sigma",
    "rate",
    "shape",
    "scale",
    "p",
    "mean",
    "var",
    "n",
    "k_min",
)


def _outlier_count_fit_params_for_row(
    model_fit: dict[str, object],
) -> dict[str, float]:
    """Extract fitted coefficient values for metrics-table rows."""
    params: dict[str, float] = {}
    for key, value in model_fit.items():
        if key in _OUTLIER_COUNT_FIT_METRIC_KEYS:
            continue
        if isinstance(value, (bool, str)):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            if np.isfinite(float(value)):
                params[str(key)] = float(value)
    return {
        col: params.get(col, np.nan)
        for col in OUTLIER_COUNT_FIT_PARAM_COLUMNS
    }


def _outlier_count_fit_metrics_table(
    fit_results_by_event: dict[str, dict[str, object]],
    *,
    event_order: tuple[str, ...] = PF_LIFECYCLE_EFF_LR_MULTI_WINDOW_EVENTS,
    model_order: tuple[str, ...] = OUTLIER_COUNT_COMPARABLE_MODELS,
    model_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Tabular fit metrics plus model-specific coefficient columns.

    Coefficient columns (``alpha``, ``pi``, ``mu``, ``n``, ``p``, etc.) are filled
    per model; unused entries are ``NaN``.
    """
    rows: list[dict[str, object]] = []
    model_labels = model_labels or OUTLIER_COUNT_FIT_MODEL_LABELS
    for event_name in event_order:
        event_fit = fit_results_by_event.get(event_name, {})
        event_label = PF_LIFECYCLE_EFF_LR_EVENT_LABELS.get(event_name, event_name)
        for model in model_order:
            model_fit = event_fit.get(model)
            if not isinstance(model_fit, dict):
                continue
            if "log_likelihood" not in model_fit:
                continue
            rows.append(
                {
                    "event_name": event_name,
                    "event_label": event_label,
                    "model": model,
                    "model_label": model_labels.get(model, model),
                    "log_likelihood": float(model_fit["log_likelihood"]),
                    "aic": float(model_fit["aic"]),
                    "bic": float(model_fit["bic"]),
                    "n_params": int(model_fit["n_params"]),
                    "n_fit": int(model_fit.get("n_fit", event_fit.get("n_samples", 0))),
                    **_outlier_count_fit_params_for_row(model_fit),
                }
            )
    return pd.DataFrame(rows)


def _plot_outlier_count_fit_metrics_figure(
    metrics_df: pd.DataFrame,
    *,
    model_order: tuple[str, ...],
    event_order: tuple[str, ...],
    suptitle: str,
):
    """
    Grouped bar chart with models on the x-axis and one bar per event.

    Returns ``(fig, axes, metrics_df)``.
    """
    import matplotlib.pyplot as plt

    if metrics_df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.set_axis_off()
        ax.set_title("No outlier-count fit metrics to plot")
        fig.tight_layout()
        return fig, ax, metrics_df

    metric_specs = (
        ("log_likelihood", "Log-likelihood\n(higher is better)"),
        ("aic", "AIC\n(lower is better)"),
        ("bic", "BIC\n(lower is better)"),
    )
    present_models = [m for m in model_order if m in metrics_df["model"].unique()]
    if not present_models:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.set_axis_off()
        ax.set_title("No outlier-count fit metrics to plot")
        fig.tight_layout()
        return fig, ax, metrics_df

    present_events = [e for e in event_order if e in metrics_df["event_name"].unique()]
    model_labels = {
        model: OUTLIER_COUNT_FIT_MODEL_LABELS.get(model, model)
        for model in present_models
    }
    event_labels = {
        event: PF_LIFECYCLE_EFF_LR_EVENT_LABELS.get(event, event)
        for event in present_events
    }
    x_models = np.arange(len(present_models), dtype=np.float64)
    n_events = max(len(present_events), 1)
    bar_width = min(0.36, 0.8 / n_events)
    tab10 = plt.get_cmap("tab10")

    fig_w = max(11.0, 1.35 * len(present_models))
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, 4.8), sharey=False)

    for ax, (metric_col, metric_title) in zip(axes, metric_specs, strict=True):
        for event_i, event_name in enumerate(present_events):
            offset = (event_i - (n_events - 1) / 2.0) * bar_width
            heights: list[float] = []
            for model in present_models:
                row = metrics_df.loc[
                    (metrics_df["event_name"] == event_name)
                    & (metrics_df["model"] == model),
                    metric_col,
                ]
                heights.append(float(row.iloc[0]) if not row.empty else np.nan)
            ax.bar(
                x_models + offset,
                heights,
                width=bar_width * 0.92,
                label=event_labels[event_name],
                color=tab10(event_i),
                alpha=0.88,
            )
        ax.set_xticks(x_models)
        ax.set_xticklabels(
            [model_labels[model] for model in present_models],
            rotation=35,
            ha="right",
        )
        ax.set_title(metric_title)
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()
    return fig, axes, metrics_df


def plot_pf_multi_window_outlier_count_fit_metrics(
    fit_results_by_event: dict[str, dict[str, object]],
    *,
    event_order: tuple[str, ...] = PF_LIFECYCLE_EFF_LR_MULTI_WINDOW_EVENTS,
    model_order: tuple[str, ...] = OUTLIER_COUNT_COMPARABLE_MODELS,
):
    """
    Grouped bar chart of log-likelihood, AIC, and BIC for outlier-count fits.

    All models use discrete PMFs evaluated on the same ``N`` observations.

    Returns ``(fig, axes, metrics_df)``.
    """
    metrics_df = _outlier_count_fit_metrics_table(
        fit_results_by_event,
        event_order=event_order,
        model_order=model_order,
    )
    return _plot_outlier_count_fit_metrics_figure(
        metrics_df,
        model_order=model_order,
        event_order=event_order,
        suptitle="Outlier-count model comparison (tail / full-count models)",
    )


def plot_pf_multi_window_outlier_count_zi_fit_metrics(
    fit_results_by_event: dict[str, dict[str, object]],
    *,
    event_order: tuple[str, ...] = PF_LIFECYCLE_EFF_LR_MULTI_WINDOW_EVENTS,
    model_order: tuple[str, ...] = OUTLIER_COUNT_ALL_ZI_MODELS,
):
    """
    Grouped bar chart of log-likelihood, AIC, and BIC for zero-inflated fits.

    Includes tail ``*_zi`` models plus ``geometric_zi`` and ``negbinom_zi``.

    Returns ``(fig, axes, metrics_df)``.
    """
    metrics_df = _outlier_count_fit_metrics_table(
        fit_results_by_event,
        event_order=event_order,
        model_order=model_order,
    )
    return _plot_outlier_count_fit_metrics_figure(
        metrics_df,
        model_order=model_order,
        event_order=event_order,
        suptitle="Outlier-count model comparison (zero-inflated tail models)",
    )


def _value_counts_sorted(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted unique values and their raw counts (no binning)."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
    uniq, counts = np.unique(values.astype(np.int64, copy=False), return_counts=True)
    return uniq.astype(np.float64), counts.astype(np.int64)


def _plot_outlier_count_bincount_on_axes(
    ax,
    vals: np.ndarray,
    *,
    event_name: str,
    event_marker: str,
    event_color: str,
    alpha: float,
    fit_points: int,
    model_linestyles: dict[str, str],
    ylim_top: float | None,
    method: str,
    layer: str,
    n_cells: int,
) -> dict[str, object]:
    """Plot one event's bincount data (markers only) and fit overlays on ``ax``."""
    import matplotlib.pyplot as plt

    label = PF_LIFECYCLE_EFF_LR_EVENT_LABELS.get(event_name, event_name)
    vals = np.asarray(vals, dtype=np.float64)
    x_vals, y_counts = _value_counts_sorted(vals)
    scatter_kwargs: dict[str, object] = {
        "marker": event_marker,
        "alpha": alpha,
        "s": 36,
        "zorder": 3,
    }
    if event_marker in {"x", "+", ".", "|", "_", "1", "2", "3", "4"}:
        scatter_kwargs["color"] = event_color
    else:
        scatter_kwargs["facecolors"] = event_color
        scatter_kwargs["edgecolors"] = "k"
        scatter_kwargs["linewidths"] = 0.3
    ax.scatter(
        x_vals,
        y_counts,
        label=f"data (n={vals.size} windows)",
        **scatter_kwargs,
    )

    event_fits = _compare_outlier_count_distribution_fits(vals)
    positive_vals = vals[np.isfinite(vals) & (vals > 0.0)]
    if positive_vals.size == 0:
        print(f"{label}: all fits skipped (no positive outlier counts)")
        ax.set_xlabel("Outliers per window")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(
            f"{label}: pooled outlier-count distribution "
            f"({method}, {layer}, |z|>{_outlier_threshold_for_method(method):.1f}, "
            f"{n_cells} cells)"
        )
        if ylim_top is not None:
            ax.set_ylim(None, ylim_top)
        return event_fits

    x_min = float(np.min(positive_vals))
    x_max = float(np.max(x_vals)) if x_vals.size else x_min
    if x_max <= x_min:
        x_max = x_min + 1.0
    x_fit_discrete = np.arange(
        0,
        int(np.floor(x_max)) + 1,
        dtype=np.float64,
    )
    model_names = (
        "negbinom_zi",
        "negbinom",
        "weibull_zi",
        "geometric_zi",
        "lognormal_zi",
        "power_law_zi",
        "power_law_cutoff_zi",
        "exponential_zi",
        "zipf_zi",
        "weibull",
        "poisson",
        "geometric",
    )
    tab10 = plt.get_cmap("tab10")
    model_colors = {
        "negbinom_zi": tab10(7),
        "negbinom": tab10(7),
        "weibull_zi": tab10(8),
        "geometric_zi": tab10(5),
        "lognormal_zi": tab10(0),
        "power_law_zi": tab10(1),
        "power_law_cutoff_zi": tab10(3),
        "exponential_zi": tab10(2),
        "zipf_zi": tab10(6),
        "weibull": tab10(8),
        "poisson": tab10(4),
        "geometric": tab10(5),
    }
    model_labels = OUTLIER_COUNT_FIT_MODEL_LABELS

    for model_name in model_names:
        model_fit = event_fits.get(model_name)
        if not isinstance(model_fit, dict):
            continue
        if "log_likelihood" not in model_fit:
            continue
        base_model, zero_inflated = _resolve_outlier_count_model(model_name)
        if base_model == "zipf":
            k_start = 0 if zero_inflated else int(model_fit["k_min"])
            x_plot = np.arange(
                k_start,
                int(np.floor(x_max)) + 1,
                dtype=np.float64,
            )
        else:
            x_plot = x_fit_discrete
        y_fit = _distribution_fit_count_curve(
            x_plot,
            model=model_name,
            fit=model_fit,
            n_samples=vals.size,
        )
        legend_parts: list[str] = []
        if zero_inflated:
            legend_parts.append(f"π={float(model_fit['pi']):.3g}")
        if base_model == "power_law":
            legend_parts.append(f"α={float(model_fit['alpha']):.2f}")
        elif base_model == "power_law_cutoff":
            legend_parts.extend(
                [
                    f"α={float(model_fit['alpha']):.2f}",
                    f"λ={float(model_fit['lambda']):.3g}",
                ]
            )
        elif base_model == "lognormal":
            legend_parts.extend(
                [
                    f"μ={float(model_fit['mu']):.2f}",
                    f"σ={float(model_fit['sigma']):.2f}",
                ]
            )
        elif base_model == "poisson":
            legend_parts.append(f"μ={float(model_fit['mu']):.3g}")
        elif base_model == "geometric":
            legend_parts.append(f"p={float(model_fit['p']):.3g}")
        elif base_model == "negbinom":
            legend_parts.extend(
                [
                    f"n={float(model_fit['n']):.3g}",
                    f"p={float(model_fit['p']):.3g}",
                ]
            )
        elif base_model == "zipf":
            legend_parts.append(f"α={float(model_fit['alpha']):.2f}")
        elif base_model == "exponential":
            legend_parts.append(f"λ={float(model_fit['rate']):.3g}")
        elif base_model == "weibull":
            legend_parts.extend(
                [
                    f"k={float(model_fit['shape']):.2f}",
                    f"scale={float(model_fit['scale']):.3g}",
                ]
            )
        legend_suffix = ", ".join(legend_parts)
        ax.plot(
            x_plot,
            y_fit,
            linestyle=model_linestyles.get(model_name, "--"),
            color=model_colors[model_name],
            linewidth=1.4,
            alpha=0.7,
            label=f"{model_labels[model_name]} ({legend_suffix})",
            zorder=5,
        )
        print(
            f"{label} / {model_labels[model_name]}: "
            f"logL={float(model_fit['log_likelihood']):.2f}, "
            f"AIC={float(model_fit['aic']):.2f}, "
            f"BIC={float(model_fit['bic']):.2f} "
            f"(n_fit={int(model_fit['n_fit'])})"
        )

    if ylim_top is not None:
        ax.set_ylim(None, ylim_top)
    ax.set_xlabel("Outliers per window")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        f"{label}: pooled outlier-count distribution "
        f"({method}, {layer}, |z|>{_outlier_threshold_for_method(method):.1f}, "
        f"{n_cells} cells)"
    )
    return event_fits



def fit_outlier_count_distribution(
    values: np.ndarray,
    *,
    fit_type: str,
) -> dict[str, object] | None:
    """Fit one outlier-count distribution model to pooled window counts."""
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    all_fits = _compare_outlier_count_distribution_fits(vals)
    fit = all_fits.get(str(fit_type))
    if not isinstance(fit, dict) or "log_likelihood" not in fit:
        return None
    return fit


def fit_outlier_count_mean_residual_ci95(
    values: np.ndarray,
    *,
    fit_type: str,
) -> tuple[float, float, float]:
    """Return (mean residual, lower, upper) at 1.96× residual std."""
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    fit = fit_outlier_count_distribution(vals, fit_type=fit_type)
    if fit is None:
        return float("nan"), float("nan"), float("nan")
    x_vals, y_counts = _value_counts_sorted(vals)
    expected = _distribution_fit_count_curve(
        x_vals,
        model=str(fit_type),
        fit=fit,
        n_samples=int(vals.size),
    )
    residuals = y_counts.astype(np.float64) - expected
    mean_res = float(np.nanmean(residuals))
    std_res = float(np.nanstd(residuals)) if residuals.size > 1 else 0.0
    half = 1.96 * std_res
    return mean_res, mean_res - half, mean_res + half


def outlier_count_empirical_and_fit_curve(
    values: np.ndarray,
    *,
    fit_type: str,
    fit: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object] | None]:
    """Return (x_data, y_data, x_fit, y_fit, fit_dict) for scatter+curve plots."""
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    x_data, y_data = _value_counts_sorted(vals)
    if fit is None:
        fit = fit_outlier_count_distribution(vals, fit_type=fit_type)
    if fit is None or vals.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return x_data, y_data, empty, empty, fit
    x_max = float(np.max(x_data)) if x_data.size else 0.0
    x_fit = np.arange(0, int(np.floor(x_max)) + 1, dtype=np.float64)
    y_fit = _distribution_fit_count_curve(
        x_fit,
        model=str(fit_type),
        fit=fit,
        n_samples=int(vals.size),
    )
    return x_data, y_data, x_fit, y_fit, fit


def format_outlier_count_fit_annotation(
    fit: dict[str, object] | None,
    *,
    fit_type: str,
    mean_residual: float,
    residual_lo: float,
    residual_hi: float,
) -> str:
    """One-line fit summary for plot annotations."""
    if fit is None:
        return "no fit"
    parts: list[str] = [str(fit_type)]
    if "pi" in fit:
        parts.append(f"π={float(fit['pi']):.3g}")
    if "n" in fit and "p" in fit:
        parts.append(f"n={float(fit['n']):.3g}")
        parts.append(f"p={float(fit['p']):.3g}")
    if np.isfinite(mean_residual):
        parts.append(f"res={mean_residual:.2f}")
        if np.isfinite(residual_lo) and np.isfinite(residual_hi):
            parts.append(f"[{residual_lo:.2f},{residual_hi:.2f}]")
    return "\n".join(parts)


def _discrete_cdf_probability(
    x: int,
    model: str,
    fit: dict[str, float],
) -> float:
    """``P(X <= x)`` for the fitted discrete model."""
    x = int(x)
    if x < 0:
        return 0.0
    return float(
        sum(_discrete_pmf_probability(k, model, fit) for k in range(x + 1))
    )


def _discrete_ppf_probability(
    p: float,
    model: str,
    fit: dict[str, float],
) -> float:
    """Smallest ``k`` with ``P(X <= k) >= p``."""
    p = float(np.clip(p, 0.0, 1.0))
    x_max = int(fit.get("x_max", 0))
    for k in range(max(x_max, 0) + 1):
        if _discrete_cdf_probability(k, model, fit) >= p:
            return float(k)
    return float(max(x_max, 0))


def _outlier_count_fit_support_residuals(
    values: np.ndarray,
    fit_type: str,
    fit: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x_vals, y_counts, expected, residuals)`` on integer support."""
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    x_vals, y_counts = _value_counts_sorted(vals)
    if x_vals.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty
    expected = _distribution_fit_count_curve(
        x_vals,
        model=str(fit_type),
        fit=fit,
        n_samples=int(vals.size),
    )
    residuals = y_counts.astype(np.float64) - expected
    return x_vals, y_counts.astype(np.float64), expected, residuals


def augment_outlier_count_fit_with_residual_metrics(
    values: np.ndarray,
    fit_type: str,
    fit: dict[str, object],
) -> dict[str, object]:
    """Attach MRS/MRE (first/last decile of support residuals) to a fit dict."""
    out = dict(fit)
    _x_vals, _y_counts, _expected, residuals = _outlier_count_fit_support_residuals(
        values,
        fit_type,
        fit,
    )
    if residuals.size == 0:
        out.update(
            {
                "mrs": float("nan"),
                "mre": float("nan"),
                "mrs_sem": float("nan"),
                "mre_sem": float("nan"),
            }
        )
        return out
    n = int(residuals.size)
    n_start = max(1, int(np.ceil(n * 0.1)))
    mrs_slice = residuals[:n_start]
    mre_slice = residuals[n_start:] if n_start < n else residuals[-1:]
    out["mrs"] = float(np.mean(mrs_slice))
    out["mre"] = float(np.mean(mre_slice))
    out["mrs_sem"] = (
        float(np.std(mrs_slice, ddof=1) / np.sqrt(mrs_slice.size))
        if mrs_slice.size > 1
        else 0.0
    )
    out["mre_sem"] = (
        float(np.std(mre_slice, ddof=1) / np.sqrt(mre_slice.size))
        if mre_slice.size > 1
        else 0.0
    )
    return out


def build_outlier_count_fit_comparison(
    values: np.ndarray,
    fit_types: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    """Fit and compare multiple outlier-count models on the same samples."""
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {}
    all_fits = _compare_outlier_count_distribution_fits(vals)
    models: dict[str, dict[str, object]] = {}
    for fit_type in fit_types:
        fit = all_fits.get(str(fit_type))
        if isinstance(fit, dict) and "log_likelihood" in fit:
            models[str(fit_type)] = augment_outlier_count_fit_with_residual_metrics(
                vals,
                str(fit_type),
                fit,
            )
    return models


def outlier_count_fit_pp_points(
    values: np.ndarray,
    fit_type: str,
    fit: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Probability-probability coordinates on the integer support."""
    x_vals, y_counts, _expected, _residuals = _outlier_count_fit_support_residuals(
        values,
        fit_type,
        fit,
    )
    if x_vals.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty
    n_total = float(y_counts.sum())
    theo_cdf = np.asarray(
        [_discrete_cdf_probability(int(x), str(fit_type), fit) for x in x_vals],
        dtype=np.float64,
    )
    emp_cdf = np.cumsum(y_counts) / n_total
    return theo_cdf, emp_cdf


def outlier_count_fit_qq_points(
    values: np.ndarray,
    fit_type: str,
    fit: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Quantile-quantile coordinates using raw samples vs fitted quantiles."""
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty
    n_quant = int(min(50, max(10, vals.size // 20)))
    ps = np.linspace(0.01, 0.99, n_quant)
    emp_q = np.quantile(vals, ps)
    theo_q = np.asarray(
        [_discrete_ppf_probability(p, str(fit_type), fit) for p in ps],
        dtype=np.float64,
    )
    return theo_q, emp_q



def plot_pf_multi_window_outlier_count_bincount(
    counts_df: pd.DataFrame,
    *,
    bin_width: int | None = None,
    event_order: tuple[str, ...] = PF_LIFECYCLE_EFF_LR_MULTI_WINDOW_EVENTS,
    event_markers: dict[str, str] | None = None,
    event_colors: dict[str, str] | None = None,
    alpha: float = 0.9,
    fit_points: int = 200,
    model_linestyles: dict[str, str] | None = None,
    ylim_top: float | None = 300.0,
):
    """
    Pooled per-window outlier-count distribution, one panel per event (no binning).

    Plots ``random`` and ``dead_random`` on separate axes. Empirical counts are
    markers only (no connecting lines); model fits are overlaid as curves.

    Returns
    -------
    fig, axes, fit_results_by_event
        ``axes`` is an array with one axis per event in ``event_order``.
        ``fit_results_by_event`` maps each event name to fit dicts (including
        ``*_zi`` variants), plus ``n_samples``.
    """
    _ = bin_width  # kept for backward compatibility; binning removed
    import matplotlib.pyplot as plt

    fit_results_by_event: dict[str, dict[str, object]] = {
        event_name: {"n_samples": 0} for event_name in event_order
    }
    if counts_df.empty:
        fig, axes = plt.subplots(1, len(event_order), figsize=(8.5 * len(event_order), 5.0))
        axes = np.atleast_1d(axes)
        for ax in axes:
            ax.set_axis_off()
        fig.suptitle("No multi-window outlier counts to plot", y=1.02)
        fig.tight_layout()
        return fig, axes, fit_results_by_event

    event_markers = event_markers or {
        "random": "o",
        "dead_random": "x",
    }
    event_colors = event_colors or {
        "random": "C0",
        "dead_random": "C3",
    }
    model_linestyles = model_linestyles or {
        "power_law_zi": "--",
        "power_law_cutoff_zi": "-.",
        "lognormal_zi": "--",
        "exponential_zi": ":",
        "zipf_zi": (0, (5, 2)),
        "poisson": (0, (3, 1, 1, 1)),
        "geometric": (0, (1, 1)),
        "negbinom": (0, (4, 1, 2, 1)),
    }

    values = counts_df["n_outliers"].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        fig, axes = plt.subplots(1, len(event_order), figsize=(8.5 * len(event_order), 5.0))
        axes = np.atleast_1d(axes)
        for ax in axes:
            ax.set_axis_off()
        fig.suptitle("No finite multi-window outlier counts to plot", y=1.02)
        fig.tight_layout()
        return fig, axes, fit_results_by_event

    method = counts_df["method"].iloc[0] if "method" in counts_df.columns else "raw_zscore"
    layer = counts_df["layer"].iloc[0] if "layer" in counts_df.columns else "recurrent"
    n_cells = counts_df[["cell_idx", "pf_idx"]].drop_duplicates().shape[0]

    present_events = [
        event_name
        for event_name in event_order
        if not counts_df.loc[counts_df["event_name"] == event_name].empty
    ]
    if not present_events:
        present_events = list(event_order)

    fig, axes = plt.subplots(
        1,
        len(present_events),
        figsize=(8.5 * len(present_events), 5.0),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, event_name in zip(axes, present_events, strict=True):
        ev = counts_df.loc[counts_df["event_name"] == event_name]
        fit_results_by_event[event_name] = _plot_outlier_count_bincount_on_axes(
            ax,
            ev["n_outliers"].to_numpy(dtype=np.float64),
            event_name=event_name,
            event_marker=event_markers.get(event_name, "o"),
            event_color=event_colors.get(event_name, "C0"),
            alpha=alpha,
            fit_points=fit_points,
            model_linestyles=model_linestyles,
            ylim_top=ylim_top,
            method=method,
            layer=layer,
            n_cells=n_cells,
        )

    fig.tight_layout()
    return fig, axes, fit_results_by_event


def _count_absent_segments_before_period_start(
    period_start_cap: int,
    *,
    absent_caps: set[int],
    cell_segment_capture_indices: np.ndarray | list[int],
) -> int:
    """Count consecutive PF-absent training segments immediately before a period start."""
    return len(
        _absent_segments_before_period_start(
            int(period_start_cap),
            absent_caps=absent_caps,
            cell_segment_capture_indices=cell_segment_capture_indices,
            max_count=10**9,
        )
    )


def _first_eligible_revival_for_pf(
    pf_segment_df: pd.DataFrame,
    *,
    cell_idx: int,
    pf_idx: int,
    cell_segment_capture_indices: np.ndarray | list[int],
    absent_caps: set[int],
    min_pre_absent: int,
    min_active: int,
) -> tuple[int | None, int | None, int | None, int | None]:
    """
    First revival life period with enough pre-revive absent and active segments.

    Returns ``(capture_idx, life_period_idx, n_pre_absent, n_active)`` or four
    ``None`` values when no period qualifies.
    """
    pf_rows = pf_segment_df[
        (pf_segment_df["cell_idx"] == int(cell_idx))
        & (pf_segment_df["pf_idx"] == int(pf_idx))
    ]
    active = pf_rows[pf_rows["is_segment"] & (pf_rows["state"] == "active")].sort_values(
        ["life_period_idx", "capture_idx"],
        kind="mergesort",
    )
    if active.empty:
        return None, None, None, None

    for life_period_idx, period_rows in active.groupby("life_period_idx", sort=False):
        if int(life_period_idx) <= 0:
            continue
        period_start_cap = int(period_rows.iloc[0]["capture_idx"])
        n_pre_absent = _count_absent_segments_before_period_start(
            period_start_cap,
            absent_caps=absent_caps,
            cell_segment_capture_indices=cell_segment_capture_indices,
        )
        n_active = int(len(period_rows))
        if n_pre_absent >= int(min_pre_absent) and n_active >= int(min_active):
            return (
                period_start_cap,
                int(life_period_idx),
                n_pre_absent,
                n_active,
            )
    return None, None, None, None


def assess_pf_eff_lr_eligibility(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    cell_idx: int,
    pf_idx: int,
    min_pre_birth_absent: int,
    min_post_birth_active: int,
    min_revive_pre_absent: int,
    min_revive_active: int,
) -> dict[str, object]:
    """Segment-count eligibility for birth / first-eligible-revive eff-LR analysis."""
    cell_idx = int(cell_idx)
    pf_idx = int(pf_idx)
    cell_caps = (
        master_df.loc[
            (master_df["cell_idx"] == cell_idx) & (master_df["segment_id"] >= 0)
        ]
        .drop_duplicates(["capture_idx"])
        .sort_values("capture_idx", kind="mergesort")["capture_idx"]
        .to_numpy(dtype=np.int32)
    )
    pf_rows = pf_segment_df[
        (pf_segment_df["cell_idx"] == cell_idx) & (pf_segment_df["pf_idx"] == pf_idx)
    ]
    active = pf_rows[pf_rows["is_segment"] & (pf_rows["state"] == "active")].sort_values(
        ["life_period_idx", "capture_idx"],
        kind="mergesort",
    )
    birth_rows = active.loc[active["life_period_idx"] == 0]
    birth_cap = int(birth_rows.iloc[0]["capture_idx"]) if not birth_rows.empty else None

    absent_caps = set(
        classify_pf_segment_groups(
            pf_segment_df,
            cell_idx=cell_idx,
            pf_idx=pf_idx,
            cell_segment_capture_indices=cell_caps,
        )["absent"]
    )
    n_pre_birth_absent = (
        _count_absent_segments_before_period_start(
            birth_cap,
            absent_caps=absent_caps,
            cell_segment_capture_indices=cell_caps,
        )
        if birth_cap is not None
        else 0
    )
    n_post_birth_active = (
        int((birth_rows["capture_idx"].astype(int) > birth_cap).sum())
        if birth_cap is not None and not birth_rows.empty
        else 0
    )
    (
        revive_cap,
        revive_life_period,
        n_pre_revive_absent,
        n_revive_active,
    ) = _first_eligible_revival_for_pf(
        pf_segment_df,
        cell_idx=cell_idx,
        pf_idx=pf_idx,
        cell_segment_capture_indices=cell_caps,
        absent_caps=absent_caps,
        min_pre_absent=min_revive_pre_absent,
        min_active=min_revive_active,
    )

    eligible_birth = (
        birth_cap is not None
        and n_pre_birth_absent >= int(min_pre_birth_absent)
        and n_post_birth_active >= int(min_post_birth_active)
    )
    eligible_revive = revive_cap is not None
    return {
        "cell_idx": cell_idx,
        "pf_idx": pf_idx,
        "eligible_birth": bool(eligible_birth),
        "eligible_revive": bool(eligible_revive),
        "eligible": bool(eligible_birth),
        "birth_capture_idx": birth_cap,
        "n_pre_birth_absent": int(n_pre_birth_absent),
        "n_post_birth_active": int(n_post_birth_active),
        "first_eligible_revive_capture_idx": revive_cap,
        "first_eligible_revive_life_period": revive_life_period,
        "n_pre_revive_absent": n_pre_revive_absent,
        "n_revive_active": n_revive_active,
    }


def summarize_pf_eff_lr_eligibility(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    **eligibility_kwargs: object,
) -> pd.DataFrame:
    """One eligibility row per tracked ``(cell_idx, pf_idx)``."""
    pf_pairs = (
        pf_segment_df[["cell_idx", "pf_idx"]]
        .drop_duplicates()
        .sort_values(["cell_idx", "pf_idx"], kind="mergesort")
        .itertuples(index=False, name=None)
    )
    rows = [
        assess_pf_eff_lr_eligibility(
            pf_segment_df,
            master_df,
            cell_idx=int(cell_idx),
            pf_idx=int(pf_idx),
            **eligibility_kwargs,
        )
        for cell_idx, pf_idx in pf_pairs
    ]
    return pd.DataFrame(rows)


def _lifecycle_event_outlier_counts_from_zscores(
    zscores_by_method: dict[str, object],
    *,
    event_name: str,
    method: str,
    layer: str,
) -> tuple[int, int]:
    """Return ``(n_pre_outliers, n_post_outliers)`` for one lifecycle event."""
    pre_scores = np.asarray(
        zscores_by_method.get(method, {})
        .get(event_name, {})
        .get("pre", {})
        .get(layer, np.empty(0, dtype=np.float64)),
        dtype=np.float64,
    )
    post_scores = np.asarray(
        zscores_by_method.get(method, {})
        .get(event_name, {})
        .get("post", {})
        .get(layer, np.empty(0, dtype=np.float64)),
        dtype=np.float64,
    )
    return (
        _count_eff_lr_outliers(pre_scores, method=method),
        _count_eff_lr_outliers(post_scores, method=method),
    )


def _pf_eligibility_lookup_from_df(
    eligibility_df: pd.DataFrame,
) -> dict[tuple[int, int], dict[str, object]]:
    """Build ``(cell_idx, pf_idx) -> eligibility row`` lookup."""
    lookup: dict[tuple[int, int], dict[str, object]] = {}
    for row in eligibility_df.to_dict(orient="records"):
        key = (int(row["cell_idx"]), int(row["pf_idx"]))
        lookup[key] = row
    return lookup


def _collect_pf_lifecycle_birth_revive_outlier_counts_for_pair(
    cell_idx: int,
    pf_idx: int,
    eligibility: dict[str, object],
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    signals_dir: Path | str,
    capture_to_global: pd.Series | dict[int, int],
    optimizer_config: dict | None,
    window_size: int,
    method: str,
    layer: str,
    *,
    signal_kind: str = "effective_lr",
) -> dict[str, object]:
    """Score birth; score first eligible revival only when one exists."""
    _window_rows, lifecycle_row = _collect_pf_window_and_lifecycle_outlier_counts_for_pair(
        cell_idx,
        pf_idx,
        pf_segment_df,
        master_df,
        signals_dir,
        capture_to_global,
        optimizer_config,
        window_size,
        0,
        method,
        layer,
        PF_LIFECYCLE_EFF_LR_BASELINES,
        None,
        eligibility,
        collect_windows=False,
        signal_kind=signal_kind,
    )
    if lifecycle_row is None:
        raise RuntimeError(
            f"Missing lifecycle outlier row for cell {cell_idx}, PF {pf_idx}"
        )
    return lifecycle_row


def _init_pf_lifecycle_outlier_worker(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    signals_dir: str,
    capture_to_global: dict[int, int],
    optimizer_config: dict | None,
    window_size: int,
    method: str,
    layer: str,
    eligibility_lookup: dict[tuple[int, int], dict[str, object]],
    signal_kind: str = "effective_lr",
) -> None:
    global _PF_LIFECYCLE_OUTLIER_WORKER
    _PF_LIFECYCLE_OUTLIER_WORKER = {
        "pf_segment_df": pf_segment_df,
        "master_df": master_df,
        "signals_dir": signals_dir,
        "capture_to_global": capture_to_global,
        "optimizer_config": optimizer_config,
        "window_size": int(window_size),
        "method": str(method),
        "layer": str(layer),
        "eligibility_lookup": eligibility_lookup,
        "signal_kind": str(signal_kind),
    }


def _collect_pf_lifecycle_birth_revive_outlier_counts_batch(
    pf_pairs_batch: list[tuple[int, int]],
) -> tuple[int, list[dict[str, object]], tuple[int, int] | None]:
    """Process up to one batch of PF keys using worker-local shared data."""
    if _PF_LIFECYCLE_OUTLIER_WORKER is None:
        raise RuntimeError("PF lifecycle outlier worker not initialized")

    state = _PF_LIFECYCLE_OUTLIER_WORKER
    eligibility_lookup = state["eligibility_lookup"]  # type: ignore[assignment]
    rows: list[dict[str, object]] = []
    last_pair: tuple[int, int] | None = None
    for cell_idx, pf_idx in pf_pairs_batch:
        cell_idx = int(cell_idx)
        pf_idx = int(pf_idx)
        last_pair = (cell_idx, pf_idx)
        eligibility = eligibility_lookup.get((cell_idx, pf_idx))
        if eligibility is None:
            continue
        row = _collect_pf_lifecycle_birth_revive_outlier_counts_for_pair(
            cell_idx,
            pf_idx,
            eligibility,
            state["pf_segment_df"],  # type: ignore[arg-type]
            state["master_df"],  # type: ignore[arg-type]
            state["signals_dir"],  # type: ignore[arg-type]
            state["capture_to_global"],  # type: ignore[arg-type]
            state["optimizer_config"],  # type: ignore[arg-type]
            state["window_size"],  # type: ignore[arg-type]
            state["method"],  # type: ignore[arg-type]
            state["layer"],  # type: ignore[arg-type]
            signal_kind=str(state.get("signal_kind", "effective_lr")),
        )
        rows.append(row)
    return len(pf_pairs_batch), rows, last_pair


def collect_pf_lifecycle_birth_revive_outlier_counts(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    eligibility_df: pd.DataFrame,
    *,
    signals_dir: Path | str,
    capture_to_global: pd.Series,
    optimizer_config: dict | None = None,
    window_size: int,
    method: OutlierMethod | str = "raw_zscore",
    layer: str = "recurrent",
    show_progress: bool = True,
    n_process: int = 12,
    batch_size: int,
    signal_kind: str = "effective_lr",
) -> pd.DataFrame:
    """
    Per-PF birth / first-eligible-revival outlier counts (pre and post separately).

    Expects ``eligibility_df`` from ``summarize_pf_eff_lr_eligibility`` (typically
    pre-filtered to ``eligible_birth == True``). Revive counts are NaN when no
    eligible revival exists. Eligibility is not recomputed here.

    Parallel mode initializes one worker copy of the shared inputs, then dispatches
    PF keys in batches of ``batch_size`` to reduce pickling overhead.
    """
    if eligibility_df.empty:
        return pd.DataFrame()

    eligibility_lookup = _pf_eligibility_lookup_from_df(eligibility_df)
    pf_pairs = list(eligibility_lookup.keys())
    if not pf_pairs:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    n_pairs = len(pf_pairs)
    if int(n_process) <= 1:
        iterator: object = pf_pairs
        if show_progress:
            iterator = tqdm(
                pf_pairs,
                total=n_pairs,
                desc="PF lifecycle outlier counts",
                unit="PF",
            )
        for cell_idx, pf_idx in iterator:
            eligibility = eligibility_lookup[(cell_idx, pf_idx)]
            row = _collect_pf_lifecycle_birth_revive_outlier_counts_for_pair(
                cell_idx,
                pf_idx,
                eligibility,
                pf_segment_df,
                master_df,
                str(signals_dir),
                capture_to_global,
                optimizer_config,
                int(window_size),
                str(method),
                str(layer),
                signal_kind=signal_kind,
            )
            rows.append(row)
            if show_progress and isinstance(iterator, tqdm):
                iterator.set_postfix_str(
                    f"last: cell {cell_idx}, PF {pf_idx}",
                    refresh=True,
                )
    else:
        batch_size = max(1, int(batch_size))
        pf_batches = [
            pf_pairs[start : start + batch_size]
            for start in range(0, n_pairs, batch_size)
        ]
        capture_to_global_map = {
            int(cap): int(gidx) for cap, gidx in capture_to_global.items()
        }
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(
            processes=min(int(n_process), len(pf_batches)),
            initializer=_init_pf_lifecycle_outlier_worker,
            initargs=(
                pf_segment_df,
                master_df,
                str(signals_dir),
                capture_to_global_map,
                optimizer_config,
                int(window_size),
                str(method),
                str(layer),
                eligibility_lookup,
                signal_kind,
            ),
        ) as pool:
            result_iter = pool.imap_unordered(
                _collect_pf_lifecycle_birth_revive_outlier_counts_batch,
                pf_batches,
            )
            if show_progress:
                pbar = tqdm(
                    total=n_pairs,
                    desc="PF lifecycle outlier counts",
                    unit="PF",
                )
                for n_processed, batch_rows, last_pair in result_iter:
                    rows.extend(batch_rows)
                    if last_pair is not None:
                        pbar.set_postfix_str(
                            f"last: cell {last_pair[0]}, PF {last_pair[1]}",
                            refresh=True,
                        )
                    pbar.update(int(n_processed))
                pbar.close()
            else:
                for _n_processed, batch_rows, _last_pair in result_iter:
                    rows.extend(batch_rows)
    return pd.DataFrame(rows)


_PF_COMBINED_OUTLIER_WORKER: dict[str, object] | None = None


def _init_pf_combined_outlier_worker(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    signals_dir: str,
    capture_to_global: dict[int, int],
    optimizer_config: dict | None,
    window_size: int,
    n_windows: int,
    method: str,
    layer: str,
    baselines: tuple[str, ...],
    random_seed: int | None,
    window_set: set[tuple[int, int]],
    eligibility_lookup: dict[tuple[int, int], dict[str, object]],
    signal_kind: str = "effective_lr",
) -> None:
    global _PF_COMBINED_OUTLIER_WORKER
    _PF_COMBINED_OUTLIER_WORKER = {
        "pf_segment_df": pf_segment_df,
        "master_df": master_df,
        "signals_dir": signals_dir,
        "capture_to_global": capture_to_global,
        "optimizer_config": optimizer_config,
        "window_size": int(window_size),
        "n_windows": int(n_windows),
        "method": str(method),
        "layer": str(layer),
        "baselines": baselines,
        "random_seed": random_seed,
        "window_set": window_set,
        "eligibility_lookup": eligibility_lookup,
        "signal_kind": str(signal_kind),
    }


def _collect_pf_window_and_lifecycle_outlier_counts_batch(
    pf_pairs_batch: list[tuple[int, int]],
) -> tuple[int, list[dict[str, object]], list[dict[str, object]], tuple[int, int] | None]:
    if _PF_COMBINED_OUTLIER_WORKER is None:
        raise RuntimeError("PF combined outlier worker not initialized")

    state = _PF_COMBINED_OUTLIER_WORKER
    window_set = state["window_set"]  # type: ignore[assignment]
    eligibility_lookup = state["eligibility_lookup"]  # type: ignore[assignment]
    window_rows: list[dict[str, object]] = []
    lifecycle_rows: list[dict[str, object]] = []
    last_pair: tuple[int, int] | None = None
    for cell_idx, pf_idx in pf_pairs_batch:
        cell_idx = int(cell_idx)
        pf_idx = int(pf_idx)
        last_pair = (cell_idx, pf_idx)
        pair_key = (cell_idx, pf_idx)
        pair_window_rows, lifecycle_row = (
            _collect_pf_window_and_lifecycle_outlier_counts_for_pair(
                cell_idx,
                pf_idx,
                state["pf_segment_df"],  # type: ignore[arg-type]
                state["master_df"],  # type: ignore[arg-type]
                state["signals_dir"],  # type: ignore[arg-type]
                state["capture_to_global"],  # type: ignore[arg-type]
                state["optimizer_config"],  # type: ignore[arg-type]
                state["window_size"],  # type: ignore[arg-type]
                state["n_windows"],  # type: ignore[arg-type]
                state["method"],  # type: ignore[arg-type]
                state["layer"],  # type: ignore[arg-type]
                state["baselines"],  # type: ignore[arg-type]
                state["random_seed"],  # type: ignore[arg-type]
                eligibility_lookup.get(pair_key),
                collect_windows=pair_key in window_set,
                signal_kind=str(state.get("signal_kind", "effective_lr")),
            )
        )
        window_rows.extend(pair_window_rows)
        if lifecycle_row is not None:
            lifecycle_rows.append(lifecycle_row)
    return len(pf_pairs_batch), window_rows, lifecycle_rows, last_pair


def collect_pf_window_and_lifecycle_outlier_counts(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    sample_pf_pairs: list[tuple[int, int]],
    eligibility_df: pd.DataFrame,
    *,
    signals_dir: Path | str,
    capture_to_global: pd.Series,
    optimizer_config: dict | None = None,
    window_size: int,
    n_windows: int = 50,
    method: OutlierMethod | str = "raw_zscore",
    layer: str = "recurrent",
    baselines: tuple[str, ...] = PF_LIFECYCLE_EFF_LR_BASELINES,
    random_seed: int | None = 42,
    show_progress: bool = True,
    n_process: int = 8,
    batch_size: int = 64,
    signal_kind: str = "effective_lr",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Window and lifecycle outlier counts in one zscore pass per PF.

    ``sample_pf_pairs`` supplies PFs for random-window scoring; ``eligibility_df``
    (typically birth-eligible rows) supplies lifecycle targets. PFs in both sets
    are scored once with both window and lifecycle modes enabled.
    """
    window_set = {(int(cell_idx), int(pf_idx)) for cell_idx, pf_idx in sample_pf_pairs}
    eligibility_lookup = (
        _pf_eligibility_lookup_from_df(eligibility_df) if not eligibility_df.empty else {}
    )
    all_pairs = sorted(window_set | set(eligibility_lookup.keys()))
    if not all_pairs:
        return pd.DataFrame(), pd.DataFrame()

    window_rows: list[dict[str, object]] = []
    lifecycle_rows: list[dict[str, object]] = []
    n_pairs = len(all_pairs)
    if int(n_process) <= 1:
        iterator: object = all_pairs
        if show_progress:
            iterator = tqdm(
                all_pairs,
                total=n_pairs,
                desc="PF outlier counts",
                unit="PF",
            )
        for cell_idx, pf_idx in iterator:
            pair_key = (cell_idx, pf_idx)
            pair_window_rows, lifecycle_row = (
                _collect_pf_window_and_lifecycle_outlier_counts_for_pair(
                    cell_idx,
                    pf_idx,
                    pf_segment_df,
                    master_df,
                    str(signals_dir),
                    capture_to_global,
                    optimizer_config,
                    int(window_size),
                    int(n_windows),
                    str(method),
                    str(layer),
                    baselines,
                    random_seed,
                    eligibility_lookup.get(pair_key),
                    collect_windows=pair_key in window_set,
                    signal_kind=signal_kind,
                )
            )
            window_rows.extend(pair_window_rows)
            if lifecycle_row is not None:
                lifecycle_rows.append(lifecycle_row)
            if show_progress and isinstance(iterator, tqdm):
                iterator.set_postfix_str(
                    f"last: cell {cell_idx}, PF {pf_idx}",
                    refresh=True,
                )
    else:
        batch_size = max(1, int(batch_size))
        pf_batches = [
            all_pairs[start : start + batch_size]
            for start in range(0, n_pairs, batch_size)
        ]
        capture_to_global_map = {
            int(cap): int(gidx) for cap, gidx in capture_to_global.items()
        }
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(
            processes=min(int(n_process), len(pf_batches)),
            initializer=_init_pf_combined_outlier_worker,
            initargs=(
                pf_segment_df,
                master_df,
                str(signals_dir),
                capture_to_global_map,
                optimizer_config,
                int(window_size),
                int(n_windows),
                str(method),
                str(layer),
                baselines,
                random_seed,
                window_set,
                eligibility_lookup,
                signal_kind,
            ),
        ) as pool:
            result_iter = pool.imap_unordered(
                _collect_pf_window_and_lifecycle_outlier_counts_batch,
                pf_batches,
            )
            if show_progress:
                pbar = tqdm(
                    total=n_pairs,
                    desc="PF outlier counts",
                    unit="PF",
                )
                for n_processed, batch_window_rows, batch_lifecycle_rows, last_pair in (
                    result_iter
                ):
                    window_rows.extend(batch_window_rows)
                    lifecycle_rows.extend(batch_lifecycle_rows)
                    if last_pair is not None:
                        pbar.set_postfix_str(
                            f"last: cell {last_pair[0]}, PF {last_pair[1]}",
                            refresh=True,
                        )
                    pbar.update(int(n_processed))
                pbar.close()
            else:
                for _n_processed, batch_window_rows, batch_lifecycle_rows, _last_pair in (
                    result_iter
                ):
                    window_rows.extend(batch_window_rows)
                    lifecycle_rows.extend(batch_lifecycle_rows)

    return pd.DataFrame(window_rows), pd.DataFrame(lifecycle_rows)


def _power_law_probability_at_count(
    count: int,
    fit: dict[str, float] | None,
) -> float:
    """Evaluate fitted continuous power-law PDF at integer ``count``."""
    return _outlier_count_probability_at_count(count, fit, fit_type="power_law")


OutlierCountFitType = Literal[
    "power_law",
    "power_law_cutoff",
    "lognormal",
    "exponential",
    "weibull",
    "poisson",
    "geometric",
    "negbinom",
    "zipf",
    "power_law_zi",
    "power_law_cutoff_zi",
    "lognormal_zi",
    "exponential_zi",
    "weibull_zi",
    "zipf_zi",
    "geometric_zi",
    "negbinom_zi",
]
OUTLIER_COUNT_FIT_TYPE_LABELS: dict[str, str] = OUTLIER_COUNT_FIT_MODEL_LABELS


def _outlier_count_probability_at_count(
    count: int,
    fit: dict[str, float] | None,
    *,
    fit_type: OutlierCountFitType | str = "exponential",
) -> float:
    """Evaluate fitted outlier-count PMF ``P(X=count)`` at integer ``count``."""
    if fit is None:
        return float("nan")
    if "log_likelihood" not in fit and not fit.get("zero_inflated"):
        return float("nan")
    return _discrete_pmf_probability(int(count), str(fit_type), fit)


def _outlier_count_cdf_at_count(
    count: int,
    fit: dict[str, float] | None,
    *,
    fit_type: OutlierCountFitType | str = "exponential",
) -> float:
    """Evaluate fitted outlier-count CDF ``P(X <= count)`` at integer ``count``."""
    if fit is None:
        return float("nan")
    if "log_likelihood" not in fit and not fit.get("zero_inflated"):
        return float("nan")
    count = int(count)
    if count < 0:
        return 0.0
    fit_type = str(fit_type)
    _base_model, zero_inflated = _resolve_outlier_count_model(fit_type)
    stored_table = fit.get("positive_pmf_table")
    if zero_inflated and isinstance(stored_table, np.ndarray):
        pi = float(fit["pi"])
        if count == 0:
            return pi
        table = np.asarray(stored_table, dtype=np.float64)
        k = min(count, int(table.size))
        return pi + (1.0 - pi) * float(np.sum(table[:k]))
    return float(
        sum(
            _discrete_pmf_probability(x, fit_type, fit)
            for x in range(count + 1)
        )
    )


OutlierCountProbabilityKind = Literal["pmf", "cdf", "epmf", "ecdf"]


def _outlier_count_fit_from_results(
    fit_results_by_event: dict[str, dict[str, object]],
    event_name: str,
    fit_type: OutlierCountFitType | str,
) -> dict[str, float] | None:
    event_results = fit_results_by_event.get(event_name)
    if not isinstance(event_results, dict):
        return None
    fit = event_results.get(str(fit_type))
    return fit if isinstance(fit, dict) else None


def print_outlier_count_fit_parameters(
    fit: dict[str, object],
    fit_type: OutlierCountFitType | str,
    *,
    event: str | None = None,
) -> None:
    """Print fitted outlier-count distribution coefficients and fit metrics."""
    fit_type = str(fit_type)
    _base_model, zero_inflated = _resolve_outlier_count_model(fit_type)
    label = OUTLIER_COUNT_FIT_MODEL_LABELS.get(fit_type, fit_type)
    header = f"Fitted parameters — {label}"
    if event is not None:
        header += f" @ {event}"
    print(header)

    if zero_inflated and fit.get("tail_model") is not None:
        tail_label = OUTLIER_COUNT_FIT_MODEL_LABELS.get(
            str(fit["tail_model"]),
            str(fit["tail_model"]),
        )
        print(f"  tail model: {tail_label}")

    params = _outlier_count_fit_params_for_row(fit)
    param_labels = {
        "pi": "π (zero-inflation)",
        "n_zero": "n_zero",
        "alpha": "α",
        "x_min": "x_min",
        "lambda": "λ",
        "mu": "μ",
        "sigma": "σ",
        "rate": "rate",
        "shape": "shape",
        "scale": "scale",
        "p": "p",
        "mean": "mean",
        "var": "var",
        "n": "n",
        "k_min": "k_min",
    }
    printed_any = False
    for col in OUTLIER_COUNT_FIT_PARAM_COLUMNS:
        val = params.get(col, np.nan)
        if np.isfinite(val):
            print(f"  {param_labels.get(col, col)} = {float(val):.6g}")
            printed_any = True
    if not printed_any:
        print("  (no finite coefficient parameters stored)")

    metric_labels = {
        "n_fit": "n_fit",
        "n_params": "n_params",
        "log_likelihood": "log-likelihood",
        "aic": "AIC",
        "bic": "BIC",
    }
    for key, metric_label in metric_labels.items():
        if key not in fit:
            continue
        value = fit[key]
        if isinstance(value, (int, float, np.integer, np.floating)):
            if np.isfinite(float(value)):
                if key in {"n_fit", "n_params"}:
                    print(f"  {metric_label} = {int(value)}")
                else:
                    print(f"  {metric_label} = {float(value):.6g}")


def _empirical_outlier_bincount_from_counts_df(
    window_counts_df: pd.DataFrame,
    event_name: str,
) -> dict[int, int]:
    """Pooled integer bincount for one multi-window event."""
    if window_counts_df.empty or "event_name" not in window_counts_df.columns:
        return {}
    ev = window_counts_df.loc[window_counts_df["event_name"] == str(event_name)]
    if ev.empty or "n_outliers" not in ev.columns:
        return {}
    return _integer_outlier_bincount(ev["n_outliers"].to_numpy(dtype=np.float64))


def empirical_outlier_count_pmf_at(
    bincount: dict[int, int],
    x: int,
) -> float:
    """Empirical ``P(X = x)`` from pooled window counts."""
    if not bincount:
        return float("nan")
    n_total = int(sum(bincount.values()))
    if n_total <= 0:
        return float("nan")
    return float(bincount.get(int(x), 0) / n_total)


def empirical_outlier_count_cdf_at(
    bincount: dict[int, int],
    x: int,
) -> float:
    """Empirical ``P(X <= x)`` from pooled window counts."""
    if not bincount:
        return float("nan")
    n_total = int(sum(bincount.values()))
    if n_total <= 0:
        return float("nan")
    return float(sum(v for k, v in bincount.items() if int(k) <= int(x)) / n_total)


def add_pf_lifecycle_outlier_empirical_probabilities(
    counts_df: pd.DataFrame,
    window_counts_df: pd.DataFrame,
    *,
    distribution_order: tuple[str, ...] = PF_LIFECYCLE_EFF_LR_MULTI_WINDOW_EVENTS,
) -> pd.DataFrame:
    """
    Map lifecycle outlier counts to empirical reference probabilities.

    For each birth/revive × pre/post count, adds ``eprob_*`` and ``ecdf_*``
    columns by lookup in the pooled random/dead-random window bincounts (no
    parametric fit).
    """
    if counts_df.empty:
        return counts_df.copy()

    bincount_by_event = {
        dist_name: _empirical_outlier_bincount_from_counts_df(
            window_counts_df,
            dist_name,
        )
        for dist_name in distribution_order
    }

    out = counts_df.copy()
    event_specs = (
        ("birth", "birth_pre_n_outliers", "birth_post_n_outliers"),
        ("revive", "revive_pre_n_outliers", "revive_post_n_outliers"),
    )
    for event_key, pre_col, post_col in event_specs:
        for baseline_key, count_col in (("pre", pre_col), ("post", post_col)):
            for dist_name in distribution_order:
                bincount = bincount_by_event[dist_name]
                eprob_col = f"eprob_{event_key}_{baseline_key}_{dist_name}"
                ecdf_col = f"ecdf_{event_key}_{baseline_key}_{dist_name}"
                out[eprob_col] = out[count_col].map(
                    lambda c, _bc=bincount: (
                        empirical_outlier_count_pmf_at(_bc, int(c))
                        if pd.notna(c)
                        else float("nan")
                    )
                )
                out[ecdf_col] = out[count_col].map(
                    lambda c, _bc=bincount: (
                        empirical_outlier_count_cdf_at(_bc, int(c))
                        if pd.notna(c)
                        else float("nan")
                    )
                )
    return out


def plot_outlier_count_fitted_and_empirical_pmf_cdf(
    fit_results_by_event: dict[str, dict[str, object]],
    window_counts_df: pd.DataFrame,
    *,
    event: str,
    fit_type: OutlierCountFitType | str,
    cdf_target: float = 0.90,
    show_continuous: bool = True,
    x_max_cap: int | None = None,
    title_prefix: str = "",
) -> tuple[object, np.ndarray]:
    """
    Four-panel PMF/CDF diagnostic: fitted PMF/CDF plus empirical ePDF/eCDF.

    Empirical panels use the pooled multi-window outlier-count bincount for
    ``event`` (no parametric fit).
    """
    import matplotlib.pyplot as plt
    from scipy.stats import weibull_min

    event = str(event)
    fit_type = str(fit_type)
    fit = _outlier_count_fit_from_results(fit_results_by_event, event, fit_type)
    if fit is None:
        raise ValueError(f"No fit for event={event!r}, fit_type={fit_type!r}")

    base_model, zero_inflated = _resolve_outlier_count_model(fit_type)
    label = OUTLIER_COUNT_FIT_MODEL_LABELS.get(fit_type, fit_type)
    support_max = int(fit.get("x_max", 0))
    print_outlier_count_fit_parameters(fit, fit_type, event=event)

    def pmf_at(x: int) -> float:
        return float(_discrete_pmf_probability(x, fit_type, fit))

    cdf = 0.0
    x_cut = 0
    for x in range(0, support_max + 1):
        cdf += pmf_at(x)
        x_cut = x
        if cdf >= float(cdf_target):
            break
    else:
        print(
            f"Warning: CDF reached only {cdf:.4f} by x={support_max}. "
            "Plotting full stored support."
        )

    if x_max_cap is not None:
        x_cut = min(x_cut, int(x_max_cap))

    x_int = np.arange(0, x_cut + 1, dtype=int)
    pmf = np.array([pmf_at(int(x)) for x in x_int], dtype=float)
    cdf_vals = np.cumsum(pmf)

    bincount = _empirical_outlier_bincount_from_counts_df(window_counts_df, event)
    n_emp = int(sum(bincount.values()))
    if bincount:
        emp_x_max = max(bincount)
        emp_x_cut = min(x_cut, emp_x_max)
    else:
        emp_x_cut = x_cut
    emp_x = np.arange(0, emp_x_cut + 1, dtype=int)
    epdf = np.array(
        [empirical_outlier_count_pmf_at(bincount, int(x)) for x in emp_x],
        dtype=float,
    )
    ecdf = np.array(
        [empirical_outlier_count_cdf_at(bincount, int(x)) for x in emp_x],
        dtype=float,
    )

    prefix = f"{title_prefix} " if title_prefix else ""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.stem(x_int, pmf, basefmt=" ", linefmt="C0-", markerfmt="C0o")
    ax.set_xlabel("Outliers per window (x)")
    ax.set_ylabel("P(X = x)")
    ax.set_title(f"{prefix}PMF — {event}, {label}")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.step(x_int, cdf_vals, where="post", color="C1", linewidth=1.8)
    ax.axhline(float(cdf_target), color="k", linestyle=":", alpha=0.6, label=f"{cdf_target:.0%} level")
    ax.axvline(x_cut, color="k", linestyle=":", alpha=0.6, label=f"x={x_cut}")
    ax.set_xlabel("Outliers per window (x)")
    ax.set_ylabel("P(X ≤ x)")
    ax.set_title(f"{prefix}CDF — {event}, {label}")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    continuous_models = {"exponential", "lognormal", "weibull"}
    if show_continuous and base_model in continuous_models:
        x_cont = np.linspace(0, max(x_cut, 1), 400)
        if base_model == "exponential":
            pdf_cont = _exponential_pdf(x_cont, rate=float(fit["rate"]))
            cdf_cont = np.array([
                _exponential_cdf_scalar(t, rate=float(fit["rate"])) for t in x_cont
            ])
        elif base_model == "lognormal":
            pdf_cont = _lognormal_pdf(x_cont, mu=float(fit["mu"]), sigma=float(fit["sigma"]))
            cdf_cont = np.array([
                _lognormal_cdf_scalar(t, mu=float(fit["mu"]), sigma=float(fit["sigma"]))
                for t in x_cont
            ])
        else:
            shape, scale = float(fit["shape"]), float(fit["scale"])
            pdf_cont = weibull_min.pdf(x_cont, shape, loc=0.0, scale=scale)
            cdf_cont = np.array([
                _weibull_cdf_scalar(t, shape=shape, scale=scale) for t in x_cont
            ])
        if zero_inflated:
            pi = float(fit["pi"])
            pdf_cont = (1.0 - pi) * pdf_cont
            cdf_cont = pi + (1.0 - pi) * cdf_cont
        axes[0, 0].plot(
            x_cont, pdf_cont, "--", color="C3", alpha=0.8, label="continuous PDF (underlying)"
        )
        axes[0, 1].plot(
            x_cont, cdf_cont, "--", color="C3", alpha=0.8, label="continuous CDF (underlying)"
        )
        axes[0, 0].legend(frameon=False, fontsize=9)
        axes[0, 1].legend(frameon=False, fontsize=9)

    ax = axes[1, 0]
    if n_emp > 0:
        ax.stem(emp_x, epdf, basefmt=" ", linefmt="C2-", markerfmt="C2o")
    ax.set_xlabel("Outliers per window (x)")
    ax.set_ylabel("eP(X = x)")
    ax.set_title(f"{prefix}ePDF — {event} (n={n_emp} windows)")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    if n_emp > 0:
        ax.step(emp_x, ecdf, where="post", color="C2", linewidth=1.8)
    ax.set_xlabel("Outliers per window (x)")
    ax.set_ylabel("eP(X ≤ x)")
    ax.set_title(f"{prefix}eCDF — {event} (n={n_emp} windows)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"{prefix}Outlier-count distribution @ {event} — fitted {label}; "
        f"empirical lookup from pooled windows",
        y=1.02,
    )
    fig.tight_layout()

    print(f"x_cut = {x_cut}  (first x with fitted CDF >= {cdf_target})")
    if n_emp > 0:
        print(f"Empirical sample size: n={n_emp} windows")
        print(f"Empirical support: 0..{max(bincount)}")
    return fig, axes


def add_pf_lifecycle_outlier_fit_probabilities(
    counts_df: pd.DataFrame,
    fit_results_by_event: dict[str, dict[str, object]],
    *,
    fit_type: OutlierCountFitType | str = "exponential",
    probability_kind: OutlierCountProbabilityKind = "pmf",
    distribution_order: tuple[str, ...] = PF_LIFECYCLE_EFF_LR_MULTI_WINDOW_EVENTS,
) -> pd.DataFrame:
    """
    Map pre/post outlier counts to fitted reference-distribution probabilities.

    Uses the selected ``fit_type`` coefficients from ``fit_results_by_event``
    (output of ``plot_pf_multi_window_outlier_count_bincount``). For each
    lifecycle event (birth, revive) and baseline (pre, post), adds one column
    per reference window type (``random``, ``dead_random``).

    With ``probability_kind="pmf"`` (default), columns are ``prob_*`` and hold
    ``P(X=x)``. With ``probability_kind="cdf"``, columns are ``cdf_*`` and hold
    ``P(X <= x)``.
    """
    if counts_df.empty:
        return counts_df.copy()

    probability_kind = str(probability_kind)
    if probability_kind not in {"pmf", "cdf"}:
        raise ValueError(f"Unexpected probability_kind {probability_kind!r}")

    col_prefix = "prob" if probability_kind == "pmf" else "cdf"
    prob_fn = (
        _outlier_count_probability_at_count
        if probability_kind == "pmf"
        else _outlier_count_cdf_at_count
    )

    out = counts_df.copy()
    event_specs = (
        ("birth", "birth_pre_n_outliers", "birth_post_n_outliers"),
        ("revive", "revive_pre_n_outliers", "revive_post_n_outliers"),
    )
    for event_key, pre_col, post_col in event_specs:
        for baseline_key, count_col in (("pre", pre_col), ("post", post_col)):
            for dist_name in distribution_order:
                fit = _outlier_count_fit_from_results(
                    fit_results_by_event,
                    dist_name,
                    fit_type,
                )
                col = f"{col_prefix}_{event_key}_{baseline_key}_{dist_name}"
                out[col] = out[count_col].map(
                    lambda c, _fit=fit, _fit_type=fit_type, _prob_fn=prob_fn: (
                        _prob_fn(
                            int(c),
                            _fit,
                            fit_type=_fit_type,
                        )
                        if pd.notna(c)
                        else float("nan")
                    )
                )
    return out


def add_pf_lifecycle_outlier_power_law_probabilities(
    counts_df: pd.DataFrame,
    power_law_fit_coefficients: dict[str, dict[str, float] | None],
    *,
    distribution_order: tuple[str, ...] = PF_LIFECYCLE_EFF_LR_MULTI_WINDOW_EVENTS,
) -> pd.DataFrame:
    """
    Map pre/post outlier counts to power-law occurrence probabilities.

    Backward-compatible wrapper around ``add_pf_lifecycle_outlier_fit_probabilities``.
    """
    fit_results_by_event = {
        event_name: {"power_law": fit}
        for event_name, fit in power_law_fit_coefficients.items()
    }
    return add_pf_lifecycle_outlier_fit_probabilities(
        counts_df,
        fit_results_by_event,
        fit_type="power_law",
        distribution_order=distribution_order,
    )


def plot_pf_lifecycle_outlier_probability_summary(
    prob_df: pd.DataFrame,
    *,
    fit_type: OutlierCountFitType | str = "exponential",
    probability_kind: OutlierCountProbabilityKind = "pmf",
    distribution_order: tuple[str, ...] = PF_LIFECYCLE_EFF_LR_MULTI_WINDOW_EVENTS,
    distribution_labels: dict[str, str] | None = None,
):
    """
    Two-row summary of fitted reference-distribution probabilities across PFs.

    Row 1: boxplots; row 2: mean ± 1.96 SEM bars. X groups are
    birth/revive × pre/post; hue is reference distribution.

    Expects ``prob_*`` columns when ``probability_kind="pmf"`` (default) or
    ``cdf_*`` columns when ``probability_kind="cdf"``. For empirical lookup
    panels use ``probability_kind="epmf"`` (``eprob_*``) or ``"ecdf"`` (``ecdf_*``).
    """
    import matplotlib.pyplot as plt

    distribution_labels = distribution_labels or PF_LIFECYCLE_EFF_LR_EVENT_LABELS
    group_specs = (
        ("birth", "pre", "Birth\npre"),
        ("birth", "post", "Birth\npost"),
        ("revive", "pre", "Revive\npre"),
        ("revive", "post", "Revive\npost"),
    )
    dist_colors = {
        "random": "C0",
        "dead_random": "C3",
    }

    probability_kind = str(probability_kind)
    kind_specs = {
        "pmf": ("prob", "P(X = x)", str(fit_type)),
        "cdf": ("cdf", "P(X ≤ x)", str(fit_type)),
        "epmf": ("eprob", "eP(X = x)", "empirical"),
        "ecdf": ("ecdf", "eP(X ≤ x)", "empirical"),
    }
    if probability_kind not in kind_specs:
        raise ValueError(f"Unexpected probability_kind {probability_kind!r}")
    col_prefix, prob_label, ref_label = kind_specs[probability_kind]

    if prob_df.empty:
        fig, axes = plt.subplots(2, 1, figsize=(8, 5))
        for ax in axes:
            ax.set_axis_off()
        axes[0].set_title("No eligible PF lifecycle probabilities to plot")
        fig.tight_layout()
        return fig, axes

    n_groups = len(group_specs)
    n_dists = len(distribution_order)
    box_width = 0.18
    group_centers = np.arange(n_groups, dtype=np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)

    for dist_i, dist_name in enumerate(distribution_order):
        offset = (dist_i - (n_dists - 1) / 2.0) * box_width
        color = dist_colors.get(dist_name, f"C{dist_i}")
        label = distribution_labels.get(dist_name, dist_name)
        for group_i, (event_key, baseline_key, _tick) in enumerate(group_specs):
            col = f"{col_prefix}_{event_key}_{baseline_key}_{dist_name}"
            vals = prob_df[col].to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            x_pos = group_centers[group_i] + offset
            if vals.size:
                axes[0].boxplot(
                    vals,
                    positions=[x_pos],
                    widths=box_width * 0.85,
                    patch_artist=True,
                    boxprops={"facecolor": color, "alpha": 0.35, "linewidth": 1.0},
                    medianprops={"color": "k", "linewidth": 1.2},
                    whiskerprops={"color": color, "linewidth": 1.0},
                    capprops={"color": color, "linewidth": 1.0},
                    flierprops={
                        "marker": "o",
                        "markersize": 3,
                        "alpha": 0.5,
                        "markerfacecolor": color,
                    },
                    showfliers=True,
                )
            mean, _sem, ci95 = _mean_sem_ci95(vals)
            axes[1].bar(
                x_pos,
                mean,
                width=box_width * 0.85,
                color=color,
                alpha=0.75,
                label=label if group_i == 0 else None,
            )
            if np.isfinite(mean) and np.isfinite(ci95):
                axes[1].errorbar(
                    x_pos,
                    mean,
                    yerr=ci95,
                    fmt="none",
                    ecolor="k",
                    elinewidth=1.0,
                    capsize=3,
                )

    fit_label = (
        OUTLIER_COUNT_FIT_TYPE_LABELS.get(ref_label, ref_label)
        if ref_label != "empirical"
        else "empirical"
    )
    axes[0].set_ylabel(f"{fit_label}\n{prob_label}")
    axes[1].set_ylabel(f"Mean {prob_label}\n± 1.96 SEM")
    n_revive = 0
    if "revive_pre_n_outliers" in prob_df.columns:
        n_revive = int(prob_df["revive_pre_n_outliers"].notna().sum())
    axes[0].set_title(
        f"PF lifecycle outlier {prob_label} ({fit_label}; "
        f"n={len(prob_df)} birth, n={n_revive} revive)"
    )
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].set_xticks(group_centers)
    axes[1].set_xticklabels([tick for *_rest, tick in group_specs])
    axes[0].set_xlim(group_centers[0] - 0.6, group_centers[-1] + 0.6)
    axes[1].legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig, axes


def plot_pf_multi_window_outlier_count_traces(
    counts_df: pd.DataFrame,
    *,
    event_order: tuple[str, ...] = PF_LIFECYCLE_EFF_LR_MULTI_WINDOW_EVENTS,
    event_labels: dict[str, str] | None = None,
    event_colors: dict[str, str] | None = None,
    ylabel: str = "Outlier count per window\n(raw z-score, pre+post sum)",
):
    """
    One subplot per PF: scatter/line traces of per-window outlier counts.

    Expects output from ``collect_pf_multi_window_outlier_counts``.
    """
    import matplotlib.pyplot as plt

    if counts_df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.set_axis_off()
        ax.set_title("No multi-window outlier counts to plot")
        fig.tight_layout()
        return fig, ax

    event_labels = event_labels or PF_LIFECYCLE_EFF_LR_EVENT_LABELS
    event_colors = event_colors or {
        "random": "C0",
        "dead_random": "C3",
    }
    pf_pairs = (
        counts_df[["cell_idx", "pf_idx"]]
        .drop_duplicates()
        .sort_values(["cell_idx", "pf_idx"], kind="mergesort")
        .itertuples(index=False, name=None)
    )
    pf_pairs = [(int(c), int(p)) for c, p in pf_pairs]
    n_rows = len(pf_pairs)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(8.0, 2.4 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for row_i, (cell_idx, pf_idx) in enumerate(pf_pairs):
        ax = axes[row_i, 0]
        sub = counts_df[
            (counts_df["cell_idx"] == cell_idx) & (counts_df["pf_idx"] == pf_idx)
        ]
        for event_name in event_order:
            ev = sub.loc[sub["event_name"] == event_name].sort_values(
                "window_idx",
                kind="mergesort",
            )
            if ev.empty:
                continue
            x = ev["window_idx"].to_numpy(dtype=np.int32)
            y = ev["n_outliers"].to_numpy(dtype=np.float64)
            color = event_colors.get(event_name, "C0")
            label = event_labels.get(event_name, event_name)
            ax.plot(
                x,
                y,
                "o-",
                color=color,
                markersize=4,
                linewidth=1.0,
                alpha=0.85,
                label=f"{label} (n={len(ev)})",
            )
        ax.set_ylabel("n outliers")
        ax.set_title(f"cell {cell_idx}, PF {pf_idx}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel("Window index (within event type)")
    method = counts_df["method"].iloc[0] if "method" in counts_df.columns else "raw_zscore"
    layer = counts_df["layer"].iloc[0] if "layer" in counts_df.columns else "recurrent"
    fig.suptitle(
        f"Per-window outlier counts ({method}, {layer}, |z|>{_outlier_threshold_for_method(method):.1f})",
        y=1.01,
    )
    fig.tight_layout()
    return fig, axes


def _mean_sem_ci95(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
    return mean, sem, 1.96 * sem


def _outlier_threshold_for_method(method: OutlierMethod | str) -> float:
    if method in {"raw_zscore", "resid_zscore"}:
        return PF_LIFECYCLE_EFF_LR_ZSCORE_OUTLIER_THRESHOLD
    return PF_LIFECYCLE_EFF_LR_ROBUSTZ_OUTLIER_THRESHOLD


def _eff_lr_outlier_mask(
    values: np.ndarray,
    *,
    method: OutlierMethod | str | None = None,
    threshold: float | None = None,
) -> np.ndarray:
    """Return a boolean mask for per-weight eff-LR outlier scores."""
    values = np.asarray(values, dtype=np.float64)
    if threshold is None:
        if method is None:
            raise ValueError("Provide either method or threshold")
        threshold = _outlier_threshold_for_method(method)
    return np.isfinite(values) & (np.abs(values) > float(threshold))


def _count_eff_lr_outliers(
    values: np.ndarray,
    *,
    method: OutlierMethod | str | None = None,
    threshold: float | None = None,
) -> int:
    if threshold is None and method is None:
        raise ValueError("Provide either method or threshold")
    return int(np.sum(_eff_lr_outlier_mask(values, method=method, threshold=threshold)))


def plot_pf_lifecycle_per_weight_eff_lr_zscore_panels(
    zscores_by_method: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    *,
    cell_idx: int,
    pf_idx: int,
    window_size: int,
    layers: tuple[str, ...] | None = None,
    outlier_methods: tuple[str, ...] = PF_LIFECYCLE_EFF_LR_OUTLIER_METHODS,
    outlier_method_labels: dict[str, str] | None = None,
    event_order: tuple[str, ...] = PF_LIFECYCLE_EFF_LR_EVENTS,
    event_labels: dict[str, str] | None = None,
    baseline_labels: dict[str, str] | None = None,
    score_ylabel: str = "Per-weight eff LR score",
    signal_name: str = "eff-LR",
):
    """Boxplot + mean±SEM bars of per-weight eff-LR scores, one column per layer.

    After each layer column, an adjacent column counts per-weight outliers in the
    event/baseline categories (birth / first revival / random × pre/post).
    Outliers use |score| > 3.0 for z-score methods and |score| > 3.5 for robust-z.

    Random and dead_random use ``N_RANDOM_EFF_LR_WINDOWS`` non-overlapping anchors:
    boxplots and mean score bars pool all per-weight scores across windows; outlier
    bars show the mean outlier count across windows ± 1.96 SEM.
    """
    import matplotlib.pyplot as plt
    from experiments.common.signals_io import TRACKED_LAYER_NAMES

    if layers is None:
        layers = TRACKED_LAYER_NAMES

    score_methods = _iter_pf_lifecycle_eff_lr_score_methods(
        zscores_by_method,
        methods=outlier_methods,
    )
    meta = _pf_lifecycle_eff_lr_result_meta(zscores_by_method)

    outlier_method_labels = (
        outlier_method_labels or PF_LIFECYCLE_EFF_LR_OUTLIER_METHOD_LABELS
    )
    event_labels = event_labels or PF_LIFECYCLE_EFF_LR_EVENT_LABELS
    baseline_labels = baseline_labels or PF_LIFECYCLE_EFF_LR_BASELINE_LABELS
    active_layers = [
        layer
        for layer in layers
        if any(
            zscores_by_method.get(method, {})
            .get(event, {})
            .get(baseline, {})
            .get(layer, np.empty(0, dtype=np.float64))
            .size
            > 0
            for method in score_methods
            for event in event_order
            for baseline in PF_LIFECYCLE_EFF_LR_BASELINES
        )
    ]
    if not active_layers:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.set_axis_off()
        ax.set_title(
            f"No lifecycle {signal_name} scores to plot (cell {cell_idx}, PF {pf_idx})"
        )
        fig.tight_layout()
        return fig, np.array([[ax]])

    n_event_groups = len(event_order)
    group_stride = len(PF_LIFECYCLE_EFF_LR_BASELINES) + 1
    n_method_rows = len(score_methods)
    n_layer_cols = len(active_layers)
    n_cols = 2 * n_layer_cols
    fig, axes = plt.subplots(
        2 * n_method_rows,
        n_cols,
        figsize=(5.0 * n_cols, 3.5 * 2 * n_method_rows),
        squeeze=False,
    )
    event_boundaries = [
        float(event_i * group_stride + 0.5) for event_i in range(1, n_event_groups)
    ]

    for layer_i, layer in enumerate(active_layers):
        layer_col = 2 * layer_i
        outlier_col = 2 * layer_i + 1
        xmax = n_event_groups * group_stride + 0.6
        for method_i, method in enumerate(score_methods):
            zscores_by_event = zscores_by_method.get(method, {})
            ax_box = axes[2 * method_i, layer_col]
            ax_bar = axes[2 * method_i + 1, layer_col]
            ax_outlier_box = axes[2 * method_i, outlier_col]
            ax_outlier_count = axes[2 * method_i + 1, outlier_col]
            ax_outlier_box.set_axis_off()

            box_data: list[np.ndarray] = []
            positions: list[float] = []
            tick_labels: list[str] = []
            bar_means: list[float] = []
            bar_cis: list[float] = []
            bar_positions: list[float] = []
            outlier_positions: list[float] = []
            outlier_counts: list[float] = []
            outlier_cis: list[float] = []
            outlier_tick_labels: list[str] = []

            for event_i, event_name in enumerate(event_order):
                group_base = event_i * group_stride + 1
                for baseline_i, baseline in enumerate(PF_LIFECYCLE_EFF_LR_BASELINES):
                    pos = float(group_base + baseline_i)
                    vals = zscores_by_event.get(event_name, {}).get(
                        baseline, {}
                    ).get(layer, np.empty(0, dtype=np.float64))
                    vals = np.asarray(vals, dtype=np.float64)
                    finite_vals = vals[np.isfinite(vals)]
                    if finite_vals.size == 0:
                        continue
                    box_data.append(finite_vals)
                    positions.append(pos)
                    tick_labels.append(
                        f"{event_labels[event_name]}\n{baseline_labels[baseline]}"
                    )
                    mean, _sem, ci = _mean_sem_ci95(finite_vals)
                    bar_means.append(mean)
                    bar_cis.append(ci)
                    bar_positions.append(pos)
                    outlier_positions.append(pos)
                    if _is_pf_lifecycle_multi_window_event(event_name):
                        event_meta = _pf_lifecycle_multi_window_event_meta(
                            meta,
                            event_name,
                        )
                        window_counts = np.asarray(
                            event_meta.get("window_outlier_counts", {})
                            .get(method, {})
                            .get(baseline, {})
                            .get(layer, np.empty(0, dtype=np.float64)),
                            dtype=np.float64,
                        )
                        if window_counts.size:
                            mean_count, _sem, ci = _mean_sem_ci95(window_counts)
                            outlier_counts.append(mean_count)
                            outlier_cis.append(ci)
                        else:
                            outlier_counts.append(0.0)
                            outlier_cis.append(0.0)
                    else:
                        outlier_counts.append(
                            float(
                                _count_eff_lr_outliers(
                                    finite_vals,
                                    method=method,
                                )
                            )
                        )
                        outlier_cis.append(0.0)
                    outlier_tick_labels.append(
                        f"{event_labels[event_name]}\n{baseline_labels[baseline]}"
                    )

            if box_data:
                ax_box.boxplot(
                    box_data,
                    positions=positions,
                    widths=0.55,
                    patch_artist=True,
                    boxprops={"facecolor": "0.85", "edgecolor": "k", "linewidth": 1.0},
                    medianprops={"color": "k", "linewidth": 1.4},
                    whiskerprops={"color": "k", "linewidth": 1.0},
                    capprops={"color": "k", "linewidth": 1.0},
                    flierprops={"marker": "o", "markersize": 3, "alpha": 0.35},
                )
                ax_box.axhline(0.0, color="0.55", linestyle="--", linewidth=0.9)
                ax_box.set_xticks(positions)
                ax_box.set_xticklabels(tick_labels, rotation=20, ha="right")
                ax_box.set_ylabel(score_ylabel)

                bar_means_arr = np.asarray(bar_means, dtype=float)
                bar_cis_arr = np.asarray(bar_cis, dtype=float)
                ax_bar.bar(
                    bar_positions,
                    bar_means_arr,
                    width=0.55,
                    color="k",
                    alpha=0.35,
                    edgecolor="k",
                )
                ax_bar.errorbar(
                    bar_positions,
                    bar_means_arr,
                    yerr=bar_cis_arr,
                    fmt="none",
                    ecolor="k",
                    capsize=4,
                    linewidth=1.2,
                )
                ax_bar.axhline(0.0, color="0.55", linestyle="--", linewidth=0.9)
                ax_bar.set_xticks(bar_positions)
                ax_bar.set_xticklabels(tick_labels, rotation=20, ha="right")
                ax_bar.set_ylabel("Mean score\n(± 1.96 SEM)")

            if outlier_positions:
                outlier_heights = np.asarray(outlier_counts, dtype=float)
                outlier_cis_arr = np.asarray(outlier_cis, dtype=float)
                ax_outlier_count.bar(
                    outlier_positions,
                    outlier_heights,
                    width=0.55,
                    color="0.55",
                    alpha=0.55,
                    edgecolor="k",
                )
                ci_mask = outlier_cis_arr > 0.0
                if np.any(ci_mask):
                    ax_outlier_count.errorbar(
                        np.asarray(outlier_positions)[ci_mask],
                        outlier_heights[ci_mask],
                        yerr=outlier_cis_arr[ci_mask],
                        fmt="none",
                        ecolor="k",
                        capsize=4,
                        linewidth=1.2,
                    )
                for pos, count, ci in zip(
                    outlier_positions,
                    outlier_counts,
                    outlier_cis,
                ):
                    label_y = float(count + ci) if ci > 0 else float(count)
                    label_y = label_y if label_y > 0 else 0.05
                    label = f"{count:.1f}" if ci > 0 else str(int(count))
                    ax_outlier_count.text(
                        pos,
                        label_y,
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )
                ax_outlier_count.set_xticks(outlier_positions)
                ax_outlier_count.set_xticklabels(
                    outlier_tick_labels,
                    rotation=20,
                    ha="right",
                )
                ymax = max(
                    (outlier_heights + outlier_cis_arr).max(initial=0.0),
                    outlier_heights.max(initial=0.0),
                    1.0,
                )
                ax_outlier_count.set_ylim(0.0, ymax * 1.15)

            method_title = outlier_method_labels.get(method, method)
            if layer_col == 0:
                ax_box.set_ylabel(f"{method_title}\nper-weight eff LR score")
                ax_bar.set_ylabel(f"{method_title}\nmean score\n(± 1.96 SEM)")
            if outlier_col == 1:
                ax_outlier_count.set_ylabel(
                    f"{method_title}\noutlier count\n"
                    f"(random: mean±1.96 SEM / {int(meta.get('n_random_windows', 0))} windows)"
                )
            ax_box.set_title(layer if method_i == 0 else "")
            if method_i == 0:
                ax_outlier_count.set_title(f"{layer}\noutliers")
            ax_box.set_xlim(0.4, xmax)
            ax_bar.set_xlim(0.4, xmax)
            ax_outlier_count.set_xlim(0.4, xmax)
            for x_sep in event_boundaries:
                ax_box.axvline(x_sep, color="0.82", linestyle=":", linewidth=1.0)
                ax_bar.axvline(x_sep, color="0.82", linestyle=":", linewidth=1.0)
                ax_outlier_count.axvline(x_sep, color="0.82", linestyle=":", linewidth=1.0)

    event_summary = ", ".join(event_labels.get(event_name, event_name) for event_name in event_order)
    fig.suptitle(
        f"Per-weight eff-LR outlier scores ({event_summary})\n"
        f"cell {cell_idx}, PF {pf_idx} | "
        f"birth/revival pre: ±{window_size} PF-absent, "
        f"random: {int(meta.get('n_random_windows', 0))} windows × "
        f"±{window_size} consecutive segments",
        y=1.02,
    )
    fig.tight_layout()
    return fig, axes


def compute_pf_g_local_mse_group_metrics(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    g_local_bundle: GLocalMseSegmentStatsBundle,
    capture_to_global: pd.Series,
    *,
    n_workers: int,
) -> dict[str, pd.DataFrame]:
    """
    Per-PF ``g_local_mse`` stats in four segment groups (see ``classify_pf_segment_groups``).

    Returns four DataFrames keyed by ``signed``, ``abs``, ``normalized``, and
    ``normalized_abs``:

    - ``signed``: per-neuron ``g_local_mse[cell_idx]`` (mean/max/std over timesteps)
    - ``abs``: same, but ``|g_local_mse|`` before temporal aggregation
    - ``normalized``: per-neuron stat divided by the leave-one-out network mean
    - ``normalized_abs``: ``|unit| / |network|`` with the same leave-one-out network mean

    For groups with multiple segments (``other_revives``, ``absent``), per-segment
    stats are computed first, then averaged across segments in the group.
    """
    active_seg = pf_segment_df[
        pf_segment_df["is_segment"] & (pf_segment_df["state"] == "active")
    ]
    empty = {variant: pd.DataFrame() for variant in PF_GLOCAL_VARIANTS}
    if active_seg.empty:
        return empty

    cell_segment_caps = (
        master_df.loc[master_df["segment_id"] >= 0]
        .drop_duplicates(["cell_idx", "capture_idx"])
        .sort_values(["cell_idx", "capture_idx"], kind="mergesort")
        .groupby("cell_idx", sort=False)["capture_idx"]
        .apply(lambda s: s.to_numpy(dtype=np.int32))
        .to_dict()
    )

    pf_keys = [
        (int(cell_idx), int(pf_idx))
        for cell_idx, pf_idx in active_seg[["cell_idx", "pf_idx"]]
        .drop_duplicates()
        .sort_values(["cell_idx", "pf_idx"], kind="mergesort")
        .itertuples(index=False, name=None)
    ]
    capture_to_global_map = {
        int(cap): int(gidx) for cap, gidx in capture_to_global.items()
    }

    if n_workers <= 1 or len(pf_keys) <= 1:
        rows_by_variant = _compute_pf_glocal_rows_for_keys(
            pf_keys,
            pf_segment_df=pf_segment_df,
            cell_segment_caps=cell_segment_caps,
            g_local_bundle=g_local_bundle,
            capture_to_global=capture_to_global_map,
        )
    else:
        n_workers = min(n_workers, len(pf_keys))
        chunks = np.array_split(np.asarray(pf_keys, dtype=object), n_workers)
        chunk_lists = [chunk.tolist() for chunk in chunks if len(chunk) > 0]
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(
            processes=len(chunk_lists),
            initializer=_init_pf_glocal_worker,
            initargs=(
                g_local_bundle,
                capture_to_global_map,
                pf_segment_df,
                cell_segment_caps,
            ),
        ) as pool:
            partial_rows = pool.map(_compute_pf_glocal_worker, chunk_lists)

        rows_by_variant = {variant: [] for variant in PF_GLOCAL_VARIANTS}
        for chunk_rows in partial_rows:
            for variant in PF_GLOCAL_VARIANTS:
                rows_by_variant[variant].extend(chunk_rows[variant])

    return {
        variant: pd.DataFrame(rows) for variant, rows in rows_by_variant.items()
    }


def _unit_segment_eff_lr_for_variant(
    eff_lr_bundle: EffectiveLrSegmentStatsBundle,
    gidxs: np.ndarray,
    cell_idx: int,
    variant: GLocalVariant,
) -> np.ndarray:
    """Per-segment effective LR scalar for one hidden unit."""
    unit_vals = eff_lr_bundle.unit_mean[gidxs, cell_idx]
    if variant == "signed":
        return unit_vals
    if variant == "abs":
        return np.abs(unit_vals)
    if variant == "normalized":
        return _safe_ratio(unit_vals, eff_lr_bundle.network_excl_mean[gidxs, cell_idx])
    return _safe_ratio(
        np.abs(unit_vals),
        np.abs(eff_lr_bundle.network_excl_mean[gidxs, cell_idx]),
    )


def _aggregate_segment_eff_lr_stats(
    capture_indices: list[int] | np.ndarray,
    *,
    capture_to_global: dict[int, int] | pd.Series,
    eff_lr_bundle: EffectiveLrSegmentStatsBundle,
    cell_idx: int,
    variant: GLocalVariant,
    stat_prefix: str = "effective_lr",
) -> dict[str, float]:
    """Return mean/max/std of per-segment effective LR over the given captures."""
    nan_stats = {
        f"{stat_prefix}_mean": float("nan"),
        f"{stat_prefix}_max": float("nan"),
        f"{stat_prefix}_std": float("nan"),
    }
    if len(capture_indices) == 0:
        return nan_stats

    if isinstance(capture_to_global, pd.Series):
        capture_to_global_map = {
            int(cap): int(gidx)
            for cap, gidx in capture_to_global.items()
            if np.isfinite(gidx)
        }
    else:
        capture_to_global_map = capture_to_global

    gidxs = np.asarray(
        [
            capture_to_global_map[int(cap)]
            for cap in capture_indices
            if int(cap) in capture_to_global_map
        ],
        dtype=np.intp,
    )
    if gidxs.size == 0:
        return nan_stats

    seg_vals = _unit_segment_eff_lr_for_variant(
        eff_lr_bundle,
        gidxs,
        cell_idx,
        variant,
    )
    group_std = float(np.nanstd(seg_vals)) if gidxs.size >= 2 else float("nan")
    return {
        f"{stat_prefix}_mean": float(np.nanmean(seg_vals)),
        f"{stat_prefix}_max": float(np.nanmax(seg_vals)),
        f"{stat_prefix}_std": group_std,
    }


def _compute_pf_eff_lr_rows_for_keys(
    pf_keys: list[tuple[int, int]],
    *,
    pf_segment_df: pd.DataFrame,
    cell_segment_caps: dict[int, np.ndarray],
    eff_lr_bundle: EffectiveLrSegmentStatsBundle,
    capture_to_global: dict[int, int],
    window_size: int,
    group_names: tuple[str, ...] = PF_EFF_LR_GROUP_NAMES,
    variants: tuple[GLocalVariant, ...] = PF_GLOCAL_VARIANTS,
) -> dict[str, list[dict[str, int | float | str]]]:
    rows_by_variant: dict[str, list[dict[str, int | float | str]]] = {
        variant: [] for variant in variants
    }
    for cell_idx, pf_idx in pf_keys:
        groups = classify_pf_segment_groups(
            pf_segment_df,
            cell_idx=int(cell_idx),
            pf_idx=int(pf_idx),
            cell_segment_capture_indices=cell_segment_caps[int(cell_idx)],
        )
        groups["pre_vicinity"] = classify_pf_pre_vicinity_segments(
            pf_segment_df,
            cell_idx=int(cell_idx),
            pf_idx=int(pf_idx),
            cell_segment_capture_indices=cell_segment_caps[int(cell_idx)],
            window_size=int(window_size),
        )
        for group_name in group_names:
            caps = groups[group_name]
            for variant in variants:
                stats = _aggregate_segment_eff_lr_stats(
                    caps,
                    capture_to_global=capture_to_global,
                    eff_lr_bundle=eff_lr_bundle,
                    cell_idx=int(cell_idx),
                    variant=variant,
                )
                rows_by_variant[variant].append(
                    {
                        "cell_idx": int(cell_idx),
                        "pf_idx": int(pf_idx),
                        "segment_group": group_name,
                        "n_segments": len(caps),
                        **stats,
                    }
                )
    return rows_by_variant


def _init_pf_eff_lr_worker(
    eff_lr_bundle: EffectiveLrSegmentStatsBundle,
    capture_to_global: dict[int, int],
    pf_segment_df: pd.DataFrame,
    cell_segment_caps: dict[int, np.ndarray],
    window_size: int,
) -> None:
    global _PF_EFF_LR_WORKER_BUNDLE, _PF_EFF_LR_WORKER_CAPTURE_TO_GLOBAL
    global _PF_EFF_LR_WORKER_PF_SEGMENT_DF, _PF_EFF_LR_WORKER_CELL_SEGMENT_CAPS
    global _PF_EFF_LR_WORKER_WINDOW_SIZE
    _PF_EFF_LR_WORKER_BUNDLE = eff_lr_bundle
    _PF_EFF_LR_WORKER_CAPTURE_TO_GLOBAL = capture_to_global
    _PF_EFF_LR_WORKER_PF_SEGMENT_DF = pf_segment_df
    _PF_EFF_LR_WORKER_CELL_SEGMENT_CAPS = cell_segment_caps
    _PF_EFF_LR_WORKER_WINDOW_SIZE = int(window_size)


def _compute_pf_eff_lr_worker(
    pf_keys: list[tuple[int, int]],
) -> dict[str, list[dict[str, int | float | str]]]:
    if (
        _PF_EFF_LR_WORKER_BUNDLE is None
        or _PF_EFF_LR_WORKER_CAPTURE_TO_GLOBAL is None
        or _PF_EFF_LR_WORKER_PF_SEGMENT_DF is None
        or _PF_EFF_LR_WORKER_CELL_SEGMENT_CAPS is None
    ):
        raise RuntimeError("PF effective-LR worker not initialized")
    if _PF_EFF_LR_WORKER_WINDOW_SIZE is None:
        raise RuntimeError("PF effective-LR worker window_size not initialized")
    return _compute_pf_eff_lr_rows_for_keys(
        pf_keys,
        pf_segment_df=_PF_EFF_LR_WORKER_PF_SEGMENT_DF,
        cell_segment_caps=_PF_EFF_LR_WORKER_CELL_SEGMENT_CAPS,
        eff_lr_bundle=_PF_EFF_LR_WORKER_BUNDLE,
        capture_to_global=_PF_EFF_LR_WORKER_CAPTURE_TO_GLOBAL,
        window_size=_PF_EFF_LR_WORKER_WINDOW_SIZE,
    )


def compute_pf_effective_lr_group_metrics(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    eff_lr_bundle: EffectiveLrSegmentStatsBundle,
    capture_to_global: pd.Series,
    *,
    window_size: int,
    n_workers: int,
) -> dict[str, pd.DataFrame]:
    """
    Per-PF effective learning rate in five segment groups (see
    ``classify_pf_segment_groups`` and ``classify_pf_pre_vicinity_segments``).

    Returns four DataFrames keyed by ``signed``, ``abs``, ``normalized``, and
    ``normalized_abs``:

    - ``signed``: per-neuron effective LR on the PF's hidden unit
    - ``abs``: ``|effective_lr|`` (identical to signed when LR is non-negative)
    - ``normalized``: per-neuron LR divided by the leave-one-out network mean
    - ``normalized_abs``: ``|unit| / |network|`` with the same leave-one-out mean

    Segment groups are ``birth``, ``peak``, ``other_revives``, ``absent``, and
    ``pre_vicinity``. The last collects, for each life period, up to
    ``VICINITY_SEGMENTS_SIZE`` PF-absent training segments immediately before
    that period's first active segment.

    Each segment contributes one scalar per unit (mean over incoming weights).
    For groups with multiple segments (``other_revives``, ``absent``,
    ``pre_vicinity``), mean/max/std are taken across segment scalars in the
    group. ``std`` is NaN when a group has fewer than two segments (e.g.
    ``birth`` and ``peak``).
    """
    active_seg = pf_segment_df[
        pf_segment_df["is_segment"] & (pf_segment_df["state"] == "active")
    ]
    empty = {variant: pd.DataFrame() for variant in PF_GLOCAL_VARIANTS}
    if active_seg.empty:
        return empty

    cell_segment_caps = (
        master_df.loc[master_df["segment_id"] >= 0]
        .drop_duplicates(["cell_idx", "capture_idx"])
        .sort_values(["cell_idx", "capture_idx"], kind="mergesort")
        .groupby("cell_idx", sort=False)["capture_idx"]
        .apply(lambda s: s.to_numpy(dtype=np.int32))
        .to_dict()
    )

    pf_keys = [
        (int(cell_idx), int(pf_idx))
        for cell_idx, pf_idx in active_seg[["cell_idx", "pf_idx"]]
        .drop_duplicates()
        .sort_values(["cell_idx", "pf_idx"], kind="mergesort")
        .itertuples(index=False, name=None)
    ]
    capture_to_global_map = {
        int(cap): int(gidx) for cap, gidx in capture_to_global.items()
    }

    if n_workers <= 1 or len(pf_keys) <= 1:
        rows_by_variant = _compute_pf_eff_lr_rows_for_keys(
            pf_keys,
            pf_segment_df=pf_segment_df,
            cell_segment_caps=cell_segment_caps,
            eff_lr_bundle=eff_lr_bundle,
            capture_to_global=capture_to_global_map,
            window_size=int(window_size),
        )
    else:
        n_workers = min(n_workers, len(pf_keys))
        chunks = np.array_split(np.asarray(pf_keys, dtype=object), n_workers)
        chunk_lists = [chunk.tolist() for chunk in chunks if len(chunk) > 0]
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(
            processes=len(chunk_lists),
            initializer=_init_pf_eff_lr_worker,
            initargs=(
                eff_lr_bundle,
                capture_to_global_map,
                pf_segment_df,
                cell_segment_caps,
                int(window_size),
            ),
        ) as pool:
            partial_rows = pool.map(_compute_pf_eff_lr_worker, chunk_lists)

        rows_by_variant = {variant: [] for variant in PF_GLOCAL_VARIANTS}
        for chunk_rows in partial_rows:
            for variant in PF_GLOCAL_VARIANTS:
                rows_by_variant[variant].extend(chunk_rows[variant])

    return {
        variant: pd.DataFrame(rows) for variant, rows in rows_by_variant.items()
    }


def _unit_segment_delta_w_for_variant(
    delta_w_bundle: DeltaWSegmentStatsBundle,
    gidxs: np.ndarray,
    cell_idx: int,
    variant: GLocalVariant,
) -> np.ndarray:
    """Per-segment signed ΔW scalar for one hidden unit."""
    unit_vals = delta_w_bundle.unit_mean[gidxs, cell_idx]
    if variant == "signed":
        return unit_vals
    if variant == "abs":
        return np.abs(unit_vals)
    if variant == "normalized":
        return _safe_ratio(unit_vals, delta_w_bundle.network_excl_mean[gidxs, cell_idx])
    return _safe_ratio(
        np.abs(unit_vals),
        np.abs(delta_w_bundle.network_excl_mean[gidxs, cell_idx]),
    )


def _aggregate_segment_delta_w_stats(
    capture_indices: list[int] | np.ndarray,
    *,
    capture_to_global: dict[int, int] | pd.Series,
    delta_w_bundle: DeltaWSegmentStatsBundle,
    cell_idx: int,
    variant: GLocalVariant,
    stat_prefix: str = "delta_w",
) -> dict[str, float]:
    """Return mean/max/std of per-segment ΔW over the given captures."""
    nan_stats = {
        f"{stat_prefix}_mean": float("nan"),
        f"{stat_prefix}_max": float("nan"),
        f"{stat_prefix}_std": float("nan"),
    }
    if len(capture_indices) == 0:
        return nan_stats

    if isinstance(capture_to_global, pd.Series):
        capture_to_global_map = {
            int(cap): int(gidx)
            for cap, gidx in capture_to_global.items()
            if np.isfinite(gidx)
        }
    else:
        capture_to_global_map = capture_to_global

    gidxs = np.asarray(
        [
            capture_to_global_map[int(cap)]
            for cap in capture_indices
            if int(cap) in capture_to_global_map
        ],
        dtype=np.intp,
    )
    if gidxs.size == 0:
        return nan_stats

    seg_vals = _unit_segment_delta_w_for_variant(
        delta_w_bundle,
        gidxs,
        cell_idx,
        variant,
    )
    group_std = float(np.nanstd(seg_vals)) if gidxs.size >= 2 else float("nan")
    return {
        f"{stat_prefix}_mean": float(np.nanmean(seg_vals)),
        f"{stat_prefix}_max": float(np.nanmax(seg_vals)),
        f"{stat_prefix}_std": group_std,
    }


def _compute_pf_delta_w_rows_for_keys(
    pf_keys: list[tuple[int, int]],
    *,
    pf_segment_df: pd.DataFrame,
    cell_segment_caps: dict[int, np.ndarray],
    delta_w_bundle: DeltaWSegmentStatsBundle,
    capture_to_global: dict[int, int],
    window_size: int,
    group_names: tuple[str, ...] = PF_EFF_LR_GROUP_NAMES,
    variants: tuple[GLocalVariant, ...] = PF_GLOCAL_VARIANTS,
) -> dict[str, list[dict[str, int | float | str]]]:
    rows_by_variant: dict[str, list[dict[str, int | float | str]]] = {
        variant: [] for variant in variants
    }
    for cell_idx, pf_idx in pf_keys:
        groups = classify_pf_segment_groups(
            pf_segment_df,
            cell_idx=int(cell_idx),
            pf_idx=int(pf_idx),
            cell_segment_capture_indices=cell_segment_caps[int(cell_idx)],
        )
        groups["pre_vicinity"] = classify_pf_pre_vicinity_segments(
            pf_segment_df,
            cell_idx=int(cell_idx),
            pf_idx=int(pf_idx),
            cell_segment_capture_indices=cell_segment_caps[int(cell_idx)],
            window_size=int(window_size),
        )
        for group_name in group_names:
            caps = groups[group_name]
            for variant in variants:
                stats = _aggregate_segment_delta_w_stats(
                    caps,
                    capture_to_global=capture_to_global,
                    delta_w_bundle=delta_w_bundle,
                    cell_idx=int(cell_idx),
                    variant=variant,
                )
                rows_by_variant[variant].append(
                    {
                        "cell_idx": int(cell_idx),
                        "pf_idx": int(pf_idx),
                        "segment_group": group_name,
                        "n_segments": len(caps),
                        **stats,
                    }
                )
    return rows_by_variant


def _init_pf_delta_w_worker(
    delta_w_bundle: DeltaWSegmentStatsBundle,
    capture_to_global: dict[int, int],
    pf_segment_df: pd.DataFrame,
    cell_segment_caps: dict[int, np.ndarray],
    window_size: int,
) -> None:
    global _PF_DELTA_W_WORKER_BUNDLE, _PF_DELTA_W_WORKER_CAPTURE_TO_GLOBAL
    global _PF_DELTA_W_WORKER_PF_SEGMENT_DF, _PF_DELTA_W_WORKER_CELL_SEGMENT_CAPS
    global _PF_DELTA_W_WORKER_WINDOW_SIZE
    _PF_DELTA_W_WORKER_BUNDLE = delta_w_bundle
    _PF_DELTA_W_WORKER_CAPTURE_TO_GLOBAL = capture_to_global
    _PF_DELTA_W_WORKER_PF_SEGMENT_DF = pf_segment_df
    _PF_DELTA_W_WORKER_CELL_SEGMENT_CAPS = cell_segment_caps
    _PF_DELTA_W_WORKER_WINDOW_SIZE = int(window_size)


def _compute_pf_delta_w_worker(
    pf_keys: list[tuple[int, int]],
) -> dict[str, list[dict[str, int | float | str]]]:
    if (
        _PF_DELTA_W_WORKER_BUNDLE is None
        or _PF_DELTA_W_WORKER_CAPTURE_TO_GLOBAL is None
        or _PF_DELTA_W_WORKER_PF_SEGMENT_DF is None
        or _PF_DELTA_W_WORKER_CELL_SEGMENT_CAPS is None
    ):
        raise RuntimeError("PF ΔW worker not initialized")
    if _PF_DELTA_W_WORKER_WINDOW_SIZE is None:
        raise RuntimeError("PF ΔW worker window_size not initialized")
    return _compute_pf_delta_w_rows_for_keys(
        pf_keys,
        pf_segment_df=_PF_DELTA_W_WORKER_PF_SEGMENT_DF,
        cell_segment_caps=_PF_DELTA_W_WORKER_CELL_SEGMENT_CAPS,
        delta_w_bundle=_PF_DELTA_W_WORKER_BUNDLE,
        capture_to_global=_PF_DELTA_W_WORKER_CAPTURE_TO_GLOBAL,
        window_size=_PF_DELTA_W_WORKER_WINDOW_SIZE,
    )


def compute_pf_delta_w_group_metrics(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    delta_w_bundle: DeltaWSegmentStatsBundle,
    capture_to_global: pd.Series,
    *,
    window_size: int,
    n_workers: int,
) -> dict[str, pd.DataFrame]:
    """
    Per-PF signed weight change in five segment groups (same grouping as
    ``compute_pf_effective_lr_group_metrics``).

    Returns four DataFrames keyed by ``signed``, ``abs``, ``normalized``, and
    ``normalized_abs``. Each segment contributes one scalar per unit: the mean
    signed ΔW over incoming weights with ``ΔW = W[t+1] - W[t]``.
    """
    active_seg = pf_segment_df[
        pf_segment_df["is_segment"] & (pf_segment_df["state"] == "active")
    ]
    empty = {variant: pd.DataFrame() for variant in PF_GLOCAL_VARIANTS}
    if active_seg.empty:
        return empty

    cell_segment_caps = (
        master_df.loc[master_df["segment_id"] >= 0]
        .drop_duplicates(["cell_idx", "capture_idx"])
        .sort_values(["cell_idx", "capture_idx"], kind="mergesort")
        .groupby("cell_idx", sort=False)["capture_idx"]
        .apply(lambda s: s.to_numpy(dtype=np.int32))
        .to_dict()
    )

    pf_keys = [
        (int(cell_idx), int(pf_idx))
        for cell_idx, pf_idx in active_seg[["cell_idx", "pf_idx"]]
        .drop_duplicates()
        .sort_values(["cell_idx", "pf_idx"], kind="mergesort")
        .itertuples(index=False, name=None)
    ]
    capture_to_global_map = {
        int(cap): int(gidx) for cap, gidx in capture_to_global.items()
    }

    if n_workers <= 1 or len(pf_keys) <= 1:
        rows_by_variant = _compute_pf_delta_w_rows_for_keys(
            pf_keys,
            pf_segment_df=pf_segment_df,
            cell_segment_caps=cell_segment_caps,
            delta_w_bundle=delta_w_bundle,
            capture_to_global=capture_to_global_map,
            window_size=int(window_size),
        )
    else:
        n_workers = min(n_workers, len(pf_keys))
        chunks = np.array_split(np.asarray(pf_keys, dtype=object), n_workers)
        chunk_lists = [chunk.tolist() for chunk in chunks if len(chunk) > 0]
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(
            processes=len(chunk_lists),
            initializer=_init_pf_delta_w_worker,
            initargs=(
                delta_w_bundle,
                capture_to_global_map,
                pf_segment_df,
                cell_segment_caps,
                int(window_size),
            ),
        ) as pool:
            partial_rows = pool.map(_compute_pf_delta_w_worker, chunk_lists)

        rows_by_variant = {variant: [] for variant in PF_GLOCAL_VARIANTS}
        for chunk_rows in partial_rows:
            for variant in PF_GLOCAL_VARIANTS:
                rows_by_variant[variant].extend(chunk_rows[variant])

    return {
        variant: pd.DataFrame(rows) for variant, rows in rows_by_variant.items()
    }


def summarize_pf_lifecycle_delta_w_outlier_methods(
    zscores_by_method: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    **kwargs: object,
) -> pd.DataFrame:
    return summarize_pf_lifecycle_eff_lr_outlier_methods(
        zscores_by_method,
        signal_label="ΔW",
        **kwargs,
    )


def pf_lifecycle_delta_w_outlier_flag_table(
    zscores_by_method: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    **kwargs: object,
) -> pd.DataFrame:
    return pf_lifecycle_eff_lr_outlier_flag_table(zscores_by_method, **kwargs)


def plot_pf_lifecycle_per_weight_delta_w_zscore_panels(
    zscores_by_method: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    **kwargs: object,
):
    return plot_pf_lifecycle_per_weight_eff_lr_zscore_panels(
        zscores_by_method,
        score_ylabel="Per-weight ΔW score",
        signal_name="ΔW",
        **kwargs,
    )


def collect_pf_multi_window_delta_w_outlier_counts(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    pf_pairs: list[tuple[int, int]],
    **kwargs: object,
) -> pd.DataFrame:
    return collect_pf_multi_window_outlier_counts(
        pf_segment_df,
        master_df,
        pf_pairs,
        signal_kind="delta_w",
        **kwargs,
    )


def collect_pf_lifecycle_birth_revive_delta_w_outlier_counts(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    eligibility_df: pd.DataFrame,
    **kwargs: object,
) -> pd.DataFrame:
    return collect_pf_lifecycle_birth_revive_outlier_counts(
        pf_segment_df,
        master_df,
        eligibility_df,
        signal_kind="delta_w",
        **kwargs,
    )


def collect_pf_window_and_lifecycle_delta_w_outlier_counts(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    sample_pf_pairs: list[tuple[int, int]],
    eligibility_df: pd.DataFrame,
    **kwargs: object,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return collect_pf_window_and_lifecycle_outlier_counts(
        pf_segment_df,
        master_df,
        sample_pf_pairs,
        eligibility_df,
        signal_kind="delta_w",
        **kwargs,
    )
