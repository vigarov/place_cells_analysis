"""Compute pipeline for two_rooms PF analysis (no plotting, no hardcoded constants)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from analysis.gaussian_fit_cycle_analysis import gaussian_fields_from_params
from experiments.common.paths import ExperimentPaths, gaussian_rf_fits_path
from experiments.common.ratemaps_io import (
    RatemapCaptureRef,
    discover_ratemap_captures,
    load_ratemap_capture,
)
from experiments.common.run import create_experiment, load_experiment_config
from optimizers.defaults import build_optimizer_config
from single_room_pf_analysis import (
    TWO_ROOMS_NAME,
    alive_proportion_by_segment,
    build_master_fits_df,
    capture_metadata_keys,
    displacement_histogram,
    gauss_state_transition_counts,
    load_segment_duration_s,
    mean_n_gaussians_by_transition,
    mean_pfs_per_neuron_stats,
    mean_r2_by_segment,
    pf_state_transition_counts,
    pf_tracked_displacement_samples,
    summarize_place_field_metrics,
    timeline_keys,
    track_place_fields,
    training_capture_to_global_segment_map,
)
from single_room_multi_opt_compute import (
    apply_results_parent_override,
    compute_displacement_curves,
    gauss_transition_counts_to_proportions,
)

TrainContext = Literal["train_same", "train_other"]

ROOM_IDXS = (0, 1)


@dataclass(frozen=True)
class TwoRoomsData:
    paths: ExperimentPaths
    suffix: str
    optimizer_tag: str
    tag_cache: dict[Path, list[str]]
    capture_refs_by_room: dict[int, list[RatemapCaptureRef]]
    master_df_by_room: dict[int, pd.DataFrame]
    gaussian_params_by_room: dict[int, np.ndarray]
    gaussian_signal_max_by_room: dict[int, np.ndarray]
    field_shape: tuple[int, int]
    segment_duration_s: float
    n_neurons: int
    n_repetitions: int
    percentile99_signal_max: float


def percentile99_signal_max(
    gaussian_signal_max_by_room: dict[int, np.ndarray],
) -> float:
    """99.5th percentile of signal_max pooled across eval rooms (multi-opt convention)."""
    pooled = np.concatenate(
        [gaussian_signal_max_by_room[r].ravel() for r in ROOM_IDXS]
    )
    return float(np.nanpercentile(pooled, 99.5))


def dead_threshold_from_frac(
    gaussian_signal_max_by_room: dict[int, np.ndarray],
    *,
    dead_threshold_frac: float,
) -> float:
    return dead_threshold_frac * percentile99_signal_max(gaussian_signal_max_by_room)


@dataclass
class RoomPFResults:
    eval_room_id: int
    master_df: pd.DataFrame
    pf_segment_df: pd.DataFrame
    pf_metrics_df: pd.DataFrame
    plot_metrics: pd.DataFrame
    pf_per_neuron: dict[str, float]
    mean_r2_ts: pd.DataFrame
    alive_ts: pd.DataFrame
    pf_transition_counts: pd.DataFrame
    gauss_ts: pd.DataFrame
    gauss_trans: pd.DataFrame
    gauss_transition_props: pd.DataFrame
    disp_samples: pd.DataFrame
    disp_hist: dict[str, Any]
    disp_hist_by_context: dict[str, dict[str, Any]]
    disp_samples_by_context: dict[str, pd.DataFrame]
    displacement_curves: dict[str, Any]
    pf_birth_room: pd.Series
    mean_pf_amplitude_ts: pd.DataFrame


@dataclass
class TwoRoomsPFResults:
    data: TwoRoomsData
    by_room: dict[int, RoomPFResults]
    birth_room_counts: pd.DataFrame
    repetition_activity: pd.DataFrame
    formation_probability: pd.DataFrame
    cell_displacement: pd.DataFrame
    consistent_pf_counts: pd.DataFrame
    pc_cohort_retention: pd.DataFrame
    pc_stability_by_repetition: pd.DataFrame
    pf_stability_by_repetition: pd.DataFrame
    pf_onset_cdf_samples: pd.DataFrame


def add_train_context(
    df: pd.DataFrame,
    *,
    eval_room_id: int,
) -> pd.DataFrame:
    """Label rows train_same vs train_other for a given eval room."""
    out = df.copy()
    if "visit_room_id" not in out.columns:
        raise ValueError("Expected visit_room_id column for two_rooms data")
    same = out["visit_room_id"].astype(int) == int(eval_room_id)
    out["train_context"] = np.where(same, "train_same", "train_other")
    return out


def train_other_intervals(
    timeline: pd.DataFrame,
    *,
    eval_room_id: int,
    x_col: str = "global_capture_idx",
) -> list[tuple[float, float]]:
    """Contiguous x-ranges where visit_room_id != eval_room_id."""
    if timeline.empty or "visit_room_id" not in timeline.columns:
        return []
    xs = timeline[x_col].to_numpy(dtype=float)
    other = timeline["visit_room_id"].astype(int).to_numpy() != int(eval_room_id)
    intervals: list[tuple[float, float]] = []
    in_interval = False
    start = 0.0
    for i, is_other in enumerate(other):
        if is_other and not in_interval:
            in_interval = True
            start = xs[i]
        elif not is_other and in_interval:
            in_interval = False
            end = xs[i - 1] if i > 0 else start
            intervals.append((start, end))
    if in_interval:
        intervals.append((start, float(xs[-1])))
    return intervals


def room_switch_vlines(
    timeline: pd.DataFrame,
    *,
    x_col: str,
) -> np.ndarray:
    """X positions where visit_room_id switches from 0 to 1."""
    if "visit_room_id" not in timeline.columns or len(timeline) < 2:
        return np.array([], dtype=float)
    room = timeline["visit_room_id"].astype(int).to_numpy()
    xs = timeline[x_col].to_numpy(dtype=float)
    switches = np.where((room[:-1] == 0) & (room[1:] == 1))[0] + 1
    return xs[switches]


def rep_switch_vlines(
    timeline: pd.DataFrame,
    *,
    x_col: str,
) -> np.ndarray:
    """X positions where visit_room_id switches from 1 to 0 (new repetition)."""
    if "visit_room_id" not in timeline.columns or "rep_id" not in timeline.columns:
        return np.array([], dtype=float)
    room = timeline["visit_room_id"].astype(int).to_numpy()
    xs = timeline[x_col].to_numpy(dtype=float)
    switches = np.where((room[:-1] == 1) & (room[1:] == 0))[0] + 1
    return xs[switches]


def _render_gaussian_cells(
    params: np.ndarray,
    field_shape: tuple[int, int],
) -> np.ndarray:
    n_cells = params.shape[0]
    h, w = field_shape
    out = np.zeros((n_cells, h, w), dtype=np.float32)
    for i in range(n_cells):
        rendered = gaussian_fields_from_params(params[i], field_shape=field_shape)
        if rendered is not None:
            out[i] = rendered[1]
    return out


def load_gaussian_capture(
    capture_idx: int,
    *,
    gaussian_params: np.ndarray,
    field_shape: tuple[int, int],
) -> np.ndarray:
    return _render_gaussian_cells(gaussian_params[capture_idx], field_shape)


def load_two_rooms_data(
    *,
    project_root: Path,
    config_path: Path,
    optimizer: str,
    results_parent_override: str | None = None,
) -> TwoRoomsData:
    run_config = load_experiment_config(config_path)
    if optimizer not in run_config.optimizers:
        raise ValueError(
            f"optimizer={optimizer!r} not in config optimizers {run_config.optimizers}"
        )
    optimizer_config = build_optimizer_config(optimizer)
    experiment = create_experiment(run_config, optimizer_config=optimizer_config)
    paths = experiment.resolve_paths()
    paths = apply_results_parent_override(
        paths,
        experiment_name=TWO_ROOMS_NAME,
        project_root=project_root,
        results_parent_override=results_parent_override,
    )

    tag_cache: dict[Path, list[str]] = {}
    capture_refs_by_room = {
        room_idx: discover_ratemap_captures(
            paths.ratemaps_dir,
            experiment_name=TWO_ROOMS_NAME,
            room_idx=room_idx,
            tag_cache=tag_cache,
        )
        for room_idx in ROOM_IDXS
    }
    for room_idx, refs in capture_refs_by_room.items():
        if not refs:
            raise FileNotFoundError(
                f"No ratemaps found for room {room_idx} under {paths.ratemaps_dir}"
            )

    gaussian_params_by_room: dict[int, np.ndarray] = {}
    gaussian_signal_max_by_room: dict[int, np.ndarray] = {}
    master_df_by_room: dict[int, pd.DataFrame] = {}

    for room_idx in ROOM_IDXS:
        fits_path = gaussian_rf_fits_path(paths.results_dir, TWO_ROOMS_NAME, room_idx)
        if not fits_path.is_file():
            raise FileNotFoundError(f"Missing Gaussian fits: {fits_path}")
        with np.load(fits_path) as fits:
            gaussian_params_by_room[room_idx] = np.asarray(fits["gaussian_params"])
            gaussian_signal_max_by_room[room_idx] = np.asarray(fits["signal_max"])
        if gaussian_params_by_room[room_idx].shape[0] != len(
            capture_refs_by_room[room_idx]
        ):
            raise ValueError(f"Room {room_idx}: fits/capture count mismatch")
        master_df_by_room[room_idx] = build_master_fits_df(
            fits_path,
            capture_refs_by_room[room_idx],
            tag_cache,
            experiment_name=TWO_ROOMS_NAME,
        )

    sample_rm = load_ratemap_capture(capture_refs_by_room[0][0])
    field_shape = (int(sample_rm.shape[1]), int(sample_rm.shape[2]))
    segment_duration_s = load_segment_duration_s(paths.results_dir / "config.json")
    n_neurons = int(master_df_by_room[0]["cell_idx"].nunique())
    n_repetitions = int(master_df_by_room[0]["rep_id"].max()) + 1
    percentile99_signal_max_val = percentile99_signal_max(gaussian_signal_max_by_room)

    return TwoRoomsData(
        paths=paths,
        suffix=paths.suffix,
        optimizer_tag=paths.results_dir.name,
        tag_cache=tag_cache,
        capture_refs_by_room=capture_refs_by_room,
        master_df_by_room=master_df_by_room,
        gaussian_params_by_room=gaussian_params_by_room,
        gaussian_signal_max_by_room=gaussian_signal_max_by_room,
        field_shape=field_shape,
        segment_duration_s=segment_duration_s,
        n_neurons=n_neurons,
        n_repetitions=n_repetitions,
        percentile99_signal_max=percentile99_signal_max_val,
    )


def visit_room_block_formation_onset_offsets(
    pf_metrics_df: pd.DataFrame,
    pf_segment_df: pd.DataFrame,
) -> pd.Series:
    """Training-segment offset from the visited room's block start in the birth rep.

    Uses ``visit_room_id`` at ``birth_capture_idx`` (not eval room / train_same).
    A PF born while visiting room 1 in rep *r* is measured from the first training
    segment of room 1's block in rep *r*, even when analyzed from room 0's fits.
    """
    cap_to_global = training_capture_to_global_segment_map(pf_segment_df)
    meta_keys = capture_metadata_keys(pf_segment_df)
    caps = (
        pf_segment_df.loc[pf_segment_df["is_segment"]]
        .drop_duplicates(["capture_idx"])
        .sort_values(meta_keys, kind="mergesort")
    )
    cap_meta = caps.set_index("capture_idx")
    block_starts = (
        caps.groupby(["rep_id", "visit_room_id"], sort=False)["capture_idx"]
        .first()
        .map(cap_to_global)
        .astype(float)
    )

    birth_caps = pf_metrics_df["birth_capture_idx"].astype(int)
    global_idx = birth_caps.map(cap_to_global).astype(float)
    birth_rep = birth_caps.map(cap_meta["rep_id"]).astype(float)
    birth_visit = birth_caps.map(cap_meta["visit_room_id"]).astype(float)
    block_idx = pd.MultiIndex.from_arrays(
        [birth_rep.astype(int), birth_visit.astype(int)],
    )
    block_start = block_idx.map(block_starts).astype(float)
    return (global_idx - block_start).where(
        global_idx.notna() & birth_rep.notna() & birth_visit.notna() & block_start.notna()
    )


def plot_metrics_train_same(room: RoomPFResults) -> pd.DataFrame:
    """Place-field metrics for PFs born during train_same for this eval room."""
    return room.plot_metrics.loc[room.pf_birth_room.notna()].copy()


def filter_metrics_for_plot(
    pf_metrics_df: pd.DataFrame,
    pf_segment_df: pd.DataFrame,
    *,
    segment_duration_s: float,
) -> pd.DataFrame:
    plot_metrics = pf_metrics_df[pf_metrics_df["length_s"] > 0].copy()
    plot_metrics["revives_per_pf"] = np.maximum(
        plot_metrics["n_life_periods"].astype(int) - 1,
        0,
    )
    rep_offsets = visit_room_block_formation_onset_offsets(
        pf_metrics_df,
        pf_segment_df,
    )
    plot_metrics["start_iteration_offset"] = rep_offsets.loc[plot_metrics.index]
    plot_metrics["formation_onset_s"] = (
        plot_metrics["start_iteration_offset"] * segment_duration_s
    )
    plot_metrics["tma_iterations"] = plot_metrics["tma_s"] / segment_duration_s
    return plot_metrics


def _capture_context_lookup(master_df: pd.DataFrame, eval_room_id: int) -> pd.DataFrame:
    keys = ["capture_idx"] + capture_metadata_keys(master_df)
    meta = master_df[keys].drop_duplicates().sort_values(
        capture_metadata_keys(master_df),
        kind="mergesort",
    ).reset_index(drop=True)
    meta["train_context"] = np.where(
        meta["visit_room_id"].astype(int) == int(eval_room_id),
        "train_same",
        "train_other",
    )
    meta["global_capture_idx"] = np.arange(len(meta), dtype=int)
    traj_keys = timeline_keys(master_df)
    meta["is_traj_start"] = meta.groupby(traj_keys, sort=False).cumcount() == 0
    return meta


def assign_pf_birth_room(
    pf_metrics_df: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    eval_room_id: int,
) -> pd.Series:
    """Mark PFs born during train_same for this eval room with birth_room=eval_room_id."""
    meta = _capture_context_lookup(master_df, eval_room_id)
    ctx_by_cap = meta.set_index("capture_idx")["train_context"]
    birth_room = pd.Series(index=pf_metrics_df.index, dtype="Int64")
    for idx, row in pf_metrics_df.iterrows():
        birth_cap = int(row["birth_capture_idx"])
        if birth_cap in ctx_by_cap.index and ctx_by_cap.loc[birth_cap] == "train_same":
            birth_room.loc[idx] = int(eval_room_id)
    return birth_room


def count_pfs_by_birth_room(pf_birth_room: pd.Series) -> pd.DataFrame:
    counts = pf_birth_room.dropna().astype(int).value_counts().sort_index()
    rows = [
        {"room_id": room_idx, "n_pfs": int(counts.get(room_idx, 0))}
        for room_idx in ROOM_IDXS
    ]
    return pd.DataFrame(rows)


def split_displacement_by_train_context(
    disp_samples: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    eval_room_id: int,
) -> dict[str, pd.DataFrame]:
    if disp_samples.empty:
        empty = disp_samples.copy()
        return {"train_same": empty, "train_other": empty}
    out = disp_samples.copy()
    if "visit_room_id" in out.columns:
        same_mask = out["visit_room_id"].astype(int) == int(eval_room_id)
    else:
        meta = _capture_context_lookup(master_df, eval_room_id)
        cap_ctx = meta.set_index("capture_idx")["train_context"]
        ctx_from = out["capture_from"].map(cap_ctx)
        ctx_to = out["capture_to"].map(cap_ctx)
        same_mask = (ctx_from == "train_same") & (ctx_to == "train_same")
    return {
        "train_same": out[same_mask].copy(),
        "train_other": out[~same_mask].copy(),
    }


def mean_pf_amplitude_timeseries(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    eval_room_id: int,
) -> pd.DataFrame:
    active = pf_segment_df[
        pf_segment_df["is_segment"] & (pf_segment_df["state"] == "active")
    ].copy()
    if active.empty:
        return pd.DataFrame(
            columns=[
                "capture_idx",
                "global_capture_idx",
                "visit_room_id",
                "rep_id",
                "train_context",
                "is_traj_start",
                "mean_amplitude",
                "sem",
            ]
        )
    meta = _capture_context_lookup(master_df, eval_room_id)
    stats = (
        active.groupby("capture_idx", sort=False)["amplitude"]
        .agg(mean_amplitude="mean", sem=lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0)
        .reset_index()
    )
    out = meta.merge(stats, on="capture_idx", how="left")
    return out.sort_values("global_capture_idx", kind="mergesort").reset_index(drop=True)


def _train_same_active_by_rep(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    eval_room_id: int,
) -> dict[int, set[int]]:
    active = pf_segment_df[
        pf_segment_df["is_segment"] & (pf_segment_df["state"] == "active")
    ].copy()
    meta = _capture_context_lookup(master_df, eval_room_id)
    cap_meta = meta.set_index("capture_idx")
    rep_cells: dict[int, set[int]] = {}
    for _, row in active.iterrows():
        cap = int(row["capture_idx"])
        if cap not in cap_meta.index:
            continue
        if cap_meta.loc[cap, "train_context"] != "train_same":
            continue
        rep = int(cap_meta.loc[cap, "rep_id"])
        rep_cells.setdefault(rep, set()).add(int(row["cell_idx"]))
    return rep_cells


def repetition_activity_curve(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    eval_room_id: int,
    n_neurons: int,
    n_repetitions: int,
) -> pd.DataFrame:
    rep_cells = _train_same_active_by_rep(
        pf_segment_df, master_df, eval_room_id=eval_room_id
    )
    cell_rep_count: dict[int, int] = {}
    for rep in range(n_repetitions):
        for cell in rep_cells.get(rep, set()):
            cell_rep_count[cell] = cell_rep_count.get(cell, 0) + 1

    hist = np.zeros(n_repetitions + 1, dtype=int)
    for cell in range(n_neurons):
        n_active_reps = cell_rep_count.get(cell, 0)
        hist[n_active_reps] += 1

    rows = [
        {
            "n_days_with_pf": i,
            "n_cells": int(hist[i]),
            "pct_neurons": 100.0 * hist[i] / n_neurons,
            "eval_room_id": eval_room_id,
        }
        for i in range(n_repetitions + 1)
    ]
    return pd.DataFrame(rows)


def formation_probability_curve(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    eval_room_id: int,
    n_neurons: int,
    n_repetitions: int,
) -> pd.DataFrame:
    """P(active PC on repetition d | j prior repetitions with an active PC).

    Paper-style Fig. 2c: pool every (neuron, repetition) opportunity, bin by the
    count of earlier repetitions where the neuron already had an active PF, and
    ask whether it is active on the current repetition (including comebacks after
    inactive repetitions).
    """
    rep_cells = _train_same_active_by_rep(
        pf_segment_df, master_df, eval_room_id=eval_room_id
    )
    numer = np.zeros(n_repetitions, dtype=int)
    denom = np.zeros(n_repetitions, dtype=int)

    for cell in range(n_neurons):
        for rep in range(n_repetitions):
            prior_count = sum(
                1
                for prior_rep in range(rep)
                if cell in rep_cells.get(prior_rep, set())
            )
            if prior_count >= n_repetitions:
                continue
            denom[prior_count] += 1
            if cell in rep_cells.get(rep, set()):
                numer[prior_count] += 1

    rows = []
    for i in range(n_repetitions):
        prob = float(numer[i] / denom[i]) if denom[i] > 0 else float("nan")
        rows.append(
            {
                "n_prior_days_with_pf": i,
                "n_opportunities": int(denom[i]),
                "n_active": int(numer[i]),
                "formation_probability": prob,
                "eval_room_id": eval_room_id,
            }
        )
    return pd.DataFrame(rows)


def _active_positions_train_same(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    eval_room_id: int,
) -> pd.DataFrame:
    active = pf_segment_df[
        pf_segment_df["is_segment"] & (pf_segment_df["state"] == "active")
    ].copy()
    meta = _capture_context_lookup(master_df, eval_room_id)
    cap_meta = meta.set_index("capture_idx")
    rows: list[dict[str, float | int]] = []
    for _, row in active.iterrows():
        cap = int(row["capture_idx"])
        if cap not in cap_meta.index:
            continue
        if cap_meta.loc[cap, "train_context"] != "train_same":
            continue
        rows.append(
            {
                "capture_idx": cap,
                "global_capture_idx": int(cap_meta.loc[cap, "global_capture_idx"]),
                "rep_id": int(cap_meta.loc[cap, "rep_id"]),
                "cell_idx": int(row["cell_idx"]),
                "mu_x_cm": float(row["mu_x_cm"]),
                "mu_y_cm": float(row["mu_y_cm"]),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "capture_idx",
                "global_capture_idx",
                "rep_id",
                "cell_idx",
                "mu_x_cm",
                "mu_y_cm",
            ]
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["cell_idx", "global_capture_idx"],
        kind="mergesort",
    ).reset_index(drop=True)


def compute_cell_displacement(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    eval_room_id: int,
    n_repetitions: int,
    show_progress: bool = True,
) -> pd.DataFrame:
    positions = _active_positions_train_same(
        pf_segment_df, master_df, eval_room_id=eval_room_id
    )
    if positions.empty:
        return pd.DataFrame(
            columns=[
                "eval_room_id",
                "kind",
                "cell_idx",
                "displacement_l2",
                "displacement_x",
                "displacement_y",
            ]
        )

    rows: list[dict[str, float | int | str]] = []

    cell_groups = list(positions.groupby("cell_idx", sort=False))
    cell_iter = tqdm(
        cell_groups,
        desc=f"Room {eval_room_id} cell displacement",
        disable=not show_progress,
        leave=False,
    )
    for cell_idx, group in cell_iter:
        g = group.sort_values("global_capture_idx", kind="mergesort").reset_index(drop=True)
        for i in range(len(g) - 1):
            r0 = g.iloc[i]
            for j in range(i + 1, len(g)):
                r1 = g.iloc[j]
                if int(r1["rep_id"]) != int(r0["rep_id"]):
                    break
                dx = float(r1["mu_x_cm"] - r0["mu_x_cm"])
                dy = float(r1["mu_y_cm"] - r0["mu_y_cm"])
                rows.append(
                    {
                        "eval_room_id": eval_room_id,
                        "kind": "within",
                        "cell_idx": int(cell_idx),
                        "displacement_x": dx,
                        "displacement_y": dy,
                        "displacement_l2": float(np.hypot(dx, dy)),
                    }
                )
                break

        by_rep: dict[int, pd.DataFrame] = {
            int(rep): grp.sort_values("global_capture_idx", kind="mergesort")
            for rep, grp in g.groupby("rep_id", sort=False)
        }
        for rep in range(n_repetitions - 1):
            if rep not in by_rep or (rep + 1) not in by_rep:
                continue
            last_row = by_rep[rep].iloc[-1]
            first_next = by_rep[rep + 1].iloc[0]
            dx = float(first_next["mu_x_cm"] - last_row["mu_x_cm"])
            dy = float(first_next["mu_y_cm"] - last_row["mu_y_cm"])
            rows.append(
                {
                    "eval_room_id": eval_room_id,
                    "kind": "cross_rep",
                    "cell_idx": int(cell_idx),
                    "displacement_x": dx,
                    "displacement_y": dy,
                    "displacement_l2": float(np.hypot(dx, dy)),
                }
            )

    return pd.DataFrame(rows)


@dataclass(frozen=True)
class PFLifePresence:
    cell_idx: int
    pf_idx: int
    life_period_idx: int
    birth_capture_idx: int
    birth_rep_id: int
    birth_train_same: bool
    active_captures: frozenset[int]


@dataclass(frozen=True)
class PFPresenceIndex:
    eval_room_id: int
    n_repetitions: int
    rep_train_same_caps: dict[int, frozenset[int]]
    pf_lives: tuple[PFLifePresence, ...]
    primary_life_by_pf: dict[tuple[int, int], PFLifePresence]


def build_pf_presence_index(
    pf_segment_df: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    eval_room_id: int,
    n_repetitions: int,
) -> PFPresenceIndex:
    """Precompute per-life active capture sets and train_same rep capture blocks."""
    meta = _capture_context_lookup(master_df, eval_room_id)
    rep_train_same_caps: dict[int, frozenset[int]] = {}
    for rep_id in range(n_repetitions):
        mask = (
            (meta["rep_id"].astype(int) == rep_id)
            & (meta["train_context"] == "train_same")
            & (meta["segment_id"].astype(int) >= 0)
        )
        if mask.any():
            rep_train_same_caps[rep_id] = frozenset(
                meta.loc[mask, "capture_idx"].astype(int).tolist()
            )
        else:
            rep_train_same_caps[rep_id] = frozenset()

    active = pf_segment_df[
        pf_segment_df["is_segment"] & (pf_segment_df["state"] == "active")
    ]
    cap_meta = meta.set_index("capture_idx")
    pf_lives: list[PFLifePresence] = []
    if not active.empty:
        grouped = active.groupby(
            ["cell_idx", "pf_idx", "life_period_idx"],
            sort=False,
        )
        for (cell_idx, pf_idx, life_period_idx), grp in grouped:
            caps = frozenset(grp["capture_idx"].astype(int).tolist())
            birth_cap = int(grp["capture_idx"].min())
            if birth_cap in cap_meta.index:
                birth_row = cap_meta.loc[birth_cap]
                birth_rep = int(birth_row["rep_id"])
                birth_same = birth_row["train_context"] == "train_same"
            else:
                birth_rep = -1
                birth_same = False
            pf_lives.append(
                PFLifePresence(
                    cell_idx=int(cell_idx),
                    pf_idx=int(pf_idx),
                    life_period_idx=int(life_period_idx),
                    birth_capture_idx=birth_cap,
                    birth_rep_id=birth_rep,
                    birth_train_same=birth_same,
                    active_captures=caps,
                )
            )

    primary_life_by_pf: dict[tuple[int, int], PFLifePresence] = {}
    for life in pf_lives:
        key = (life.cell_idx, life.pf_idx)
        prev = primary_life_by_pf.get(key)
        if prev is None or life.birth_capture_idx < prev.birth_capture_idx:
            primary_life_by_pf[key] = life

    return PFPresenceIndex(
        eval_room_id=eval_room_id,
        n_repetitions=n_repetitions,
        rep_train_same_caps=rep_train_same_caps,
        pf_lives=tuple(pf_lives),
        primary_life_by_pf=primary_life_by_pf,
    )


def _covers_reps(
    active_captures: frozenset[int],
    rep_train_same_caps: dict[int, frozenset[int]],
    required_reps: set[int] | frozenset[int],
) -> bool:
    for rep_id in required_reps:
        req = rep_train_same_caps.get(int(rep_id), frozenset())
        if not req or not req.issubset(active_captures):
            return False
    return True


def build_presence_index_by_room(
    pf_results_by_room: dict[int, RoomPFResults],
    *,
    n_repetitions: int,
) -> dict[int, PFPresenceIndex]:
    return {
        room_idx: build_pf_presence_index(
            pf_results_by_room[room_idx].pf_segment_df,
            pf_results_by_room[room_idx].master_df,
            eval_room_id=room_idx,
            n_repetitions=n_repetitions,
        )
        for room_idx in ROOM_IDXS
    }


def compute_consistent_pf_counts(
    presence_by_room: dict[int, PFPresenceIndex],
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    cohort_specs = [
        ("days_1_4", 0, frozenset({0, 1, 2, 3}), "#d62728"),
        ("days_2_4", 1, frozenset({1, 2, 3}), "#000000"),
        ("days_3_4", 2, frozenset({2, 3}), "#1f3a5f"),
        ("day_4_only", 3, frozenset({3}), "#98df8a"),
    ]
    rows: list[dict[str, Any]] = []

    room_ids = sorted(presence_by_room)
    room_iter = tqdm(
        room_ids,
        desc="Consistent PF cohorts",
        disable=not show_progress,
        leave=False,
    )
    for eval_room_id in room_iter:
        room_iter.set_postfix(room=eval_room_id)
        index = presence_by_room[eval_room_id]
        for cohort_id, first_rep, required_reps, color in cohort_specs:
            count = 0
            for life in index.primary_life_by_pf.values():
                if not life.birth_train_same or life.birth_rep_id != first_rep:
                    continue
                if _covers_reps(
                    life.active_captures,
                    index.rep_train_same_caps,
                    required_reps,
                ):
                    count += 1
            rows.append(
                {
                    "eval_room_id": eval_room_id,
                    "cohort": cohort_id,
                    "color": color,
                    "count": count,
                }
            )
    return pd.DataFrame(rows)


def _build_cell_active_captures(
    pf_segment_df: pd.DataFrame,
) -> dict[int, frozenset[int]]:
    active = pf_segment_df[
        pf_segment_df["is_segment"] & (pf_segment_df["state"] == "active")
    ]
    if active.empty:
        return {}
    return {
        int(cell_idx): frozenset(grp["capture_idx"].astype(int).tolist())
        for cell_idx, grp in active.groupby("cell_idx", sort=False)
    }


def _last_train_same_capture(
    meta: pd.DataFrame,
    rep_train_same_caps: dict[int, frozenset[int]],
    rep_id: int,
) -> int | None:
    caps = rep_train_same_caps.get(int(rep_id), frozenset())
    if not caps:
        return None
    sub = meta.loc[meta["capture_idx"].astype(int).isin(caps)]
    if sub.empty:
        return None
    last_idx = sub["segment_id"].astype(int).idxmax()
    return int(sub.loc[last_idx, "capture_idx"])


def compute_pc_stability_by_repetition(
    presence_by_room: dict[int, PFPresenceIndex],
    pf_results_by_room: dict[int, RoomPFResults],
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Count total, sustained, and transient PCs per repetition (train_same).

    A PC is active on day i when it has any PF on train_same that day.
    Sustained at day i (i >= 1): active at the last train_same segment of day
    i-1 and at every train_same segment of day i.
    Transient at day i: active that day but not sustained (includes birth day).
    """
    rows: list[dict[str, Any]] = []
    room_ids = sorted(presence_by_room)
    room_iter = tqdm(
        room_ids,
        desc="PC stability by repetition",
        disable=not show_progress,
        leave=False,
    )
    for eval_room_id in room_iter:
        room_iter.set_postfix(room=eval_room_id)
        index = presence_by_room[eval_room_id]
        room_results = pf_results_by_room[eval_room_id]
        meta = _capture_context_lookup(room_results.master_df, eval_room_id)
        cell_caps = _build_cell_active_captures(room_results.pf_segment_df)
        for rep_id in range(index.n_repetitions):
            day_caps = index.rep_train_same_caps.get(int(rep_id), frozenset())
            active_cells = [
                cell_idx
                for cell_idx, caps in cell_caps.items()
                if day_caps and caps & day_caps
            ]
            n_total = len(active_cells)
            n_sustained = 0
            if rep_id >= 1 and active_cells:
                last_prev = _last_train_same_capture(
                    meta,
                    index.rep_train_same_caps,
                    rep_id - 1,
                )
                if last_prev is not None:
                    for cell_idx in active_cells:
                        caps = cell_caps[cell_idx]
                        if day_caps.issubset(caps) and last_prev in caps:
                            n_sustained += 1
            rows.append(
                {
                    "eval_room_id": eval_room_id,
                    "repetition": rep_id,
                    "n_total_pcs": n_total,
                    "n_sustained_pcs": n_sustained,
                    "n_transient_pcs": n_total - n_sustained,
                }
            )
    return pd.DataFrame(rows)


PFKey = tuple[int, int]


def _build_pf_active_captures(
    pf_segment_df: pd.DataFrame,
) -> dict[PFKey, frozenset[int]]:
    active = pf_segment_df[
        pf_segment_df["is_segment"] & (pf_segment_df["state"] == "active")
    ]
    if active.empty:
        return {}
    out: dict[PFKey, set[int]] = {}
    for (cell_idx, pf_idx), grp in active.groupby(["cell_idx", "pf_idx"], sort=False):
        key = (int(cell_idx), int(pf_idx))
        out.setdefault(key, set()).update(grp["capture_idx"].astype(int).tolist())
    return {key: frozenset(caps) for key, caps in out.items()}


def _ordered_train_same_captures(
    meta: pd.DataFrame,
    rep_train_same_caps: dict[int, frozenset[int]],
    rep_id: int,
) -> list[int]:
    caps = rep_train_same_caps.get(int(rep_id), frozenset())
    if not caps:
        return []
    sub = meta.loc[meta["capture_idx"].astype(int).isin(caps)]
    sort_keys = capture_metadata_keys(meta)
    sub = sub.sort_values(sort_keys, kind="mergesort")
    return sub["capture_idx"].astype(int).tolist()


def _last_onset_index(
    active_caps: frozenset[int],
    ordered_caps: list[int],
) -> int | None:
    if not ordered_caps:
        return None
    last_start: int | None = None
    for i, cap in enumerate(ordered_caps):
        prev_active = i > 0 and ordered_caps[i - 1] in active_caps
        curr_active = cap in active_caps
        if curr_active and not prev_active:
            last_start = i
    return last_start


def _pf_active_in_block(
    active_caps: frozenset[int],
    ordered_caps: list[int],
) -> bool:
    return bool(ordered_caps) and any(cap in active_caps for cap in ordered_caps)


def _pf_stable_from_last_onset(
    active_caps: frozenset[int],
    ordered_caps: list[int],
) -> bool:
    onset_idx = _last_onset_index(active_caps, ordered_caps)
    if onset_idx is None:
        return False
    tail = ordered_caps[onset_idx:]
    return bool(tail) and all(cap in active_caps for cap in tail)


def _is_sustained_pf_at_rep(
    active_caps: frozenset[int],
    ordered_prev: list[int],
    ordered_day: list[int],
) -> bool:
    if not _pf_active_in_block(active_caps, ordered_prev):
        return False
    if not _pf_active_in_block(active_caps, ordered_day):
        return False
    return _pf_stable_from_last_onset(active_caps, ordered_day)


def _pf_birth_rep(
    active_caps: frozenset[int],
    rep_train_same_caps: dict[int, frozenset[int]],
    n_repetitions: int,
) -> int | None:
    for rep_id in range(n_repetitions):
        day_caps = rep_train_same_caps.get(rep_id, frozenset())
        if day_caps and active_caps & day_caps:
            return rep_id
    return None


PC_COHORT_COLORS: dict[int, str] = {
    0: "#d62728",
    1: "#000000",
    2: "#87CEEB",
    3: "#98df8a",
}


def compute_pc_cohort_retention(
    presence_by_room: dict[int, PFPresenceIndex],
    pf_results_by_room: dict[int, RoomPFResults],
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    """PC cohort retention matrix (Vaidya Fig 2e style, train_same only).

    Cohort *b*: place cells whose **first** PF birth day is *b*.
    At recording day *r* >= *b*: count cells in cohort *b* with at least one
    displacement-tracked PF born on day *b* active for some train_same time on
    day *r* (each cell counted once).
    """
    rows: list[dict[str, Any]] = []
    room_ids = sorted(presence_by_room)
    room_iter = tqdm(
        room_ids,
        desc="PC cohort retention",
        disable=not show_progress,
        leave=False,
    )
    for eval_room_id in room_iter:
        room_iter.set_postfix(room=eval_room_id)
        index = presence_by_room[eval_room_id]
        room_results = pf_results_by_room[eval_room_id]
        meta = _capture_context_lookup(room_results.master_df, eval_room_id)
        pf_caps = _build_pf_active_captures(room_results.pf_segment_df)
        n_reps = index.n_repetitions

        ordered_by_rep = {
            rep_id: _ordered_train_same_captures(meta, index.rep_train_same_caps, rep_id)
            for rep_id in range(n_reps)
        }

        pf_birth_rep: dict[PFKey, int] = {}
        for pf_key, caps in pf_caps.items():
            birth = _pf_birth_rep(caps, index.rep_train_same_caps, n_reps)
            if birth is not None:
                pf_birth_rep[pf_key] = birth

        first_birth_by_cell: dict[int, int] = {}
        for (cell_idx, _pf_idx), birth in pf_birth_rep.items():
            prev = first_birth_by_cell.get(cell_idx)
            if prev is None or birth < prev:
                first_birth_by_cell[cell_idx] = birth

        pfs_by_cell_birth: dict[tuple[int, int], list[PFKey]] = {}
        for pf_key, birth in pf_birth_rep.items():
            pfs_by_cell_birth.setdefault((pf_key[0], birth), []).append(pf_key)

        for birth_cohort in range(n_reps):
            cohort_cells = {
                cell for cell, first in first_birth_by_cell.items() if first == birth_cohort
            }
            color = PC_COHORT_COLORS.get(birth_cohort, "#888888")
            for recording_day in range(birth_cohort, n_reps):
                ordered_day = ordered_by_rep[recording_day]
                count = 0
                for cell_idx in cohort_cells:
                    pf_keys = pfs_by_cell_birth.get((cell_idx, birth_cohort), [])
                    if any(
                        _pf_active_in_block(pf_caps[pk], ordered_day)
                        for pk in pf_keys
                        if pk in pf_caps
                    ):
                        count += 1
                rows.append(
                    {
                        "eval_room_id": eval_room_id,
                        "birth_cohort": birth_cohort,
                        "recording_day": recording_day,
                        "count": count,
                        "color": color,
                    }
                )
    return pd.DataFrame(rows)


def _onset_s_in_block(
    active_caps: frozenset[int],
    ordered_caps: list[int],
    *,
    segment_duration_s: float,
) -> float | None:
    onset_idx = _last_onset_index(active_caps, ordered_caps)
    if onset_idx is None:
        return None
    return float(onset_idx) * float(segment_duration_s)


def compute_pf_stability_analysis(
    presence_by_room: dict[int, PFPresenceIndex],
    pf_results_by_room: dict[int, RoomPFResults],
    *,
    segment_duration_s: float,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PF-level sustained/transient counts and onset samples (train_same only).

    A tracked PF (cell_idx, pf_idx) is sustained at repetition i (i >= 1) when it
    was active on train_same at some point in i-1, active on train_same at some
    point in i, and from its last (re)appearance in i remains active through the
    end of that train_same block. Transient at i: active that day but not sustained.
    """
    count_rows: list[dict[str, Any]] = []
    onset_rows: list[dict[str, Any]] = []
    room_ids = sorted(presence_by_room)
    room_iter = tqdm(
        room_ids,
        desc="PF stability by repetition",
        disable=not show_progress,
        leave=False,
    )
    for eval_room_id in room_iter:
        room_iter.set_postfix(room=eval_room_id)
        index = presence_by_room[eval_room_id]
        room_results = pf_results_by_room[eval_room_id]
        meta = _capture_context_lookup(room_results.master_df, eval_room_id)
        pf_caps = _build_pf_active_captures(room_results.pf_segment_df)
        ordered_by_rep = {
            rep_id: _ordered_train_same_captures(meta, index.rep_train_same_caps, rep_id)
            for rep_id in range(index.n_repetitions)
        }

        birth_rep: dict[PFKey, int] = {}
        sustained_at_rep: dict[tuple[PFKey, int], bool] = {}
        for pf_key, caps in pf_caps.items():
            birth = _pf_birth_rep(caps, index.rep_train_same_caps, index.n_repetitions)
            if birth is None:
                continue
            birth_rep[pf_key] = birth
            for rep_id in range(index.n_repetitions):
                ordered_day = ordered_by_rep[rep_id]
                if not _pf_active_in_block(caps, ordered_day):
                    continue
                if rep_id >= 1:
                    sustained = _is_sustained_pf_at_rep(
                        caps,
                        ordered_by_rep[rep_id - 1],
                        ordered_day,
                    )
                else:
                    sustained = False
                sustained_at_rep[(pf_key, rep_id)] = sustained

        ever_sustained_after_birth = {
            pf_key: any(
                sustained_at_rep.get((pf_key, rep_id), False)
                for rep_id in range(birth + 1, index.n_repetitions)
            )
            for pf_key, birth in birth_rep.items()
        }

        for rep_id in range(index.n_repetitions):
            ordered_day = ordered_by_rep[rep_id]
            active_keys = [
                pf_key
                for pf_key, caps in pf_caps.items()
                if _pf_active_in_block(caps, ordered_day)
            ]
            n_total = len(active_keys)
            n_sustained = sum(
                1 for pf_key in active_keys if sustained_at_rep.get((pf_key, rep_id), False)
            )
            count_rows.append(
                {
                    "eval_room_id": eval_room_id,
                    "repetition": rep_id,
                    "n_total_pfs": n_total,
                    "n_sustained_pfs": n_sustained,
                    "n_transient_pfs": n_total - n_sustained,
                }
            )

        for pf_key, birth in birth_rep.items():
            caps = pf_caps[pf_key]
            ordered_birth = ordered_by_rep[birth]
            onset_s = _onset_s_in_block(
                caps,
                ordered_birth,
                segment_duration_s=segment_duration_s,
            )
            if onset_s is None:
                continue
            cohort = "birth_day"
            label = (
                "sustained"
                if ever_sustained_after_birth.get(pf_key, False)
                else "transient"
            )
            onset_rows.append(
                {
                    "eval_room_id": eval_room_id,
                    "cohort": cohort,
                    "label": label,
                    "repetition": birth,
                    "cell_idx": pf_key[0],
                    "pf_idx": pf_key[1],
                    "onset_s": onset_s,
                }
            )

        for (pf_key, rep_id), is_sustained in sustained_at_rep.items():
            if rep_id <= birth_rep.get(pf_key, rep_id):
                continue
            caps = pf_caps[pf_key]
            ordered_day = ordered_by_rep[rep_id]
            onset_s = _onset_s_in_block(
                caps,
                ordered_day,
                segment_duration_s=segment_duration_s,
            )
            if onset_s is None:
                continue
            onset_rows.append(
                {
                    "eval_room_id": eval_room_id,
                    "cohort": "subsequent_days",
                    "label": "sustained" if is_sustained else "transient",
                    "repetition": rep_id,
                    "cell_idx": pf_key[0],
                    "pf_idx": pf_key[1],
                    "onset_s": onset_s,
                }
            )

    return pd.DataFrame(count_rows), pd.DataFrame(onset_rows)


def compute_room_pf_results(
    data: TwoRoomsData,
    *,
    eval_room_id: int,
    dead_threshold: float,
    r2_threshold: float,
    num_gauss_threshold: float,
    pf_tracking_method: str,
    constant_l2_similarity: float,
    zoom_bin_width: float,
    max_factor_pct: float = 1.0,
    last_epoch_only: bool = False,
    show_progress: bool = True,
) -> RoomPFResults:
    master_df = data.master_df_by_room[eval_room_id]
    step = tqdm(
        total=7,
        desc=f"Room {eval_room_id} PF pipeline",
        disable=not show_progress,
        leave=False,
    )
    step.set_postfix_str("track PFs")
    pf_segment_df = track_place_fields(
        master_df,
        dead_threshold=dead_threshold,
        r2_threshold=r2_threshold,
        gauss_threshold=num_gauss_threshold,
        segment_duration_s=data.segment_duration_s,
        last_epoch_only=last_epoch_only,
        pf_tracking_method=pf_tracking_method,
        constant_l2_similarity=constant_l2_similarity,
        show_progress=show_progress,
    )
    step.update(1)
    step.set_postfix_str("PF metrics")
    pf_metrics_df = summarize_place_field_metrics(
        pf_segment_df,
        segment_duration_s=data.segment_duration_s,
        max_factor_pct=max_factor_pct,
    )
    plot_metrics = filter_metrics_for_plot(
        pf_metrics_df,
        pf_segment_df,
        segment_duration_s=data.segment_duration_s,
    )
    pf_per_neuron = mean_pfs_per_neuron_stats(
        pf_metrics_df,
        n_neurons=data.n_neurons,
    )
    step.update(1)
    step.set_postfix_str("timeseries")
    mean_r2_ts = mean_r2_by_segment(master_df, last_epoch_only=last_epoch_only)
    alive_ts = alive_proportion_by_segment(
        master_df,
        dead_threshold=dead_threshold,
        r2_threshold=r2_threshold,
        last_epoch_only=last_epoch_only,
    )
    step.update(1)
    step.set_postfix_str("PF transitions")
    pf_transition_counts = pf_state_transition_counts(
        pf_segment_df,
        master_df,
        dead_threshold=dead_threshold,
        r2_threshold=r2_threshold,
        last_epoch_only=last_epoch_only,
        show_progress=show_progress,
    )
    step.update(1)
    step.set_postfix_str("Gaussian transitions")
    gauss_ts = mean_n_gaussians_by_transition(
        master_df,
        gauss_threshold=num_gauss_threshold,
        dead_threshold=dead_threshold,
        r2_threshold=r2_threshold,
        last_epoch_only=last_epoch_only,
    )
    gauss_trans = gauss_state_transition_counts(
        master_df,
        gauss_threshold=num_gauss_threshold,
        dead_threshold=dead_threshold,
        r2_threshold=r2_threshold,
        last_epoch_only=last_epoch_only,
    )
    gauss_transition_props = gauss_transition_counts_to_proportions(
        gauss_trans,
        n_neurons=data.n_neurons,
    )
    step.update(1)
    step.set_postfix_str("displacements")
    disp_samples = pf_tracked_displacement_samples(
        pf_segment_df,
        master_df,
        gauss_threshold=num_gauss_threshold,
        dead_threshold=dead_threshold,
        r2_threshold=r2_threshold,
        last_epoch_only=last_epoch_only,
        show_progress=show_progress,
    )
    disp_hist = displacement_histogram(disp_samples)
    disp_by_ctx = split_displacement_by_train_context(
        disp_samples, master_df, eval_room_id=eval_room_id
    )
    disp_hist_by_context = {
        key: displacement_histogram(df) if not df.empty else disp_hist
        for key, df in disp_by_ctx.items()
    }
    displacement_curves = compute_displacement_curves(
        disp_samples,
        zoom_bin_width=zoom_bin_width,
    )
    step.update(1)
    step.set_postfix_str("birth room + amplitude")
    pf_birth_room = assign_pf_birth_room(
        pf_metrics_df,
        master_df,
        eval_room_id=eval_room_id,
    )
    mean_pf_amplitude_ts = mean_pf_amplitude_timeseries(
        pf_segment_df,
        master_df,
        eval_room_id=eval_room_id,
    )
    step.update(1)
    step.close()

    return RoomPFResults(
        eval_room_id=eval_room_id,
        master_df=master_df,
        pf_segment_df=pf_segment_df,
        pf_metrics_df=pf_metrics_df,
        plot_metrics=plot_metrics,
        pf_per_neuron=pf_per_neuron,
        mean_r2_ts=mean_r2_ts,
        alive_ts=alive_ts,
        pf_transition_counts=pf_transition_counts,
        gauss_ts=gauss_ts,
        gauss_trans=gauss_trans,
        gauss_transition_props=gauss_transition_props,
        disp_samples=disp_samples,
        disp_hist=disp_hist,
        disp_hist_by_context=disp_hist_by_context,
        disp_samples_by_context=disp_by_ctx,
        displacement_curves=displacement_curves,
        pf_birth_room=pf_birth_room,
        mean_pf_amplitude_ts=mean_pf_amplitude_ts,
    )


def compute_all_two_rooms_pf_results(
    data: TwoRoomsData,
    *,
    dead_threshold: float,
    r2_threshold: float,
    num_gauss_threshold: float,
    pf_tracking_method: str,
    constant_l2_similarity: float,
    zoom_bin_width: float,
    max_factor_pct: float = 1.0,
    last_epoch_only: bool = False,
    show_progress: bool = True,
) -> TwoRoomsPFResults:
    by_room: dict[int, RoomPFResults] = {}
    room_iter = tqdm(
        ROOM_IDXS,
        desc="Two-rooms PF analysis",
        disable=not show_progress,
    )
    for room_idx in room_iter:
        room_iter.set_postfix(room=room_idx)
        by_room[room_idx] = compute_room_pf_results(
            data,
            eval_room_id=room_idx,
            dead_threshold=dead_threshold,
            r2_threshold=r2_threshold,
            num_gauss_threshold=num_gauss_threshold,
            pf_tracking_method=pf_tracking_method,
            constant_l2_similarity=constant_l2_similarity,
            zoom_bin_width=zoom_bin_width,
            max_factor_pct=max_factor_pct,
            last_epoch_only=last_epoch_only,
            show_progress=show_progress,
        )

    birth_counts = pd.DataFrame(
        [
            {
                "room_id": r,
                "n_pfs": int(by_room[r].pf_birth_room.notna().sum()),
            }
            for r in ROOM_IDXS
        ]
    )

    presence_by_room = build_presence_index_by_room(
        by_room,
        n_repetitions=data.n_repetitions,
    )

    extended_steps = [
        (
            "repetition activity",
            lambda: pd.concat(
                [
                    repetition_activity_curve(
                        by_room[r].pf_segment_df,
                        by_room[r].master_df,
                        eval_room_id=r,
                        n_neurons=data.n_neurons,
                        n_repetitions=data.n_repetitions,
                    )
                    for r in ROOM_IDXS
                ],
                ignore_index=True,
            ),
        ),
        (
            "formation probability",
            lambda: pd.concat(
                [
                    formation_probability_curve(
                        by_room[r].pf_segment_df,
                        by_room[r].master_df,
                        eval_room_id=r,
                        n_neurons=data.n_neurons,
                        n_repetitions=data.n_repetitions,
                    )
                    for r in ROOM_IDXS
                ],
                ignore_index=True,
            ),
        ),
        (
            "cell displacement",
            lambda: pd.concat(
                [
                    compute_cell_displacement(
                        by_room[r].pf_segment_df,
                        by_room[r].master_df,
                        eval_room_id=r,
                        n_repetitions=data.n_repetitions,
                        show_progress=show_progress,
                    )
                    for r in ROOM_IDXS
                ],
                ignore_index=True,
            ),
        ),
        (
            "consistent PFs",
            lambda: compute_consistent_pf_counts(
                presence_by_room,
                show_progress=show_progress,
            ),
        ),
        (
            "PC cohort retention",
            lambda: compute_pc_cohort_retention(
                presence_by_room,
                by_room,
                show_progress=show_progress,
            ),
        ),
        (
            "PC stability by repetition",
            lambda: compute_pc_stability_by_repetition(
                presence_by_room,
                by_room,
                show_progress=show_progress,
            ),
        ),
        (
            "PF stability by repetition",
            lambda: compute_pf_stability_analysis(
                presence_by_room,
                by_room,
                segment_duration_s=data.segment_duration_s,
                show_progress=show_progress,
            ),
        ),
    ]
    extended: dict[str, pd.DataFrame] = {}
    ext_iter = tqdm(
        extended_steps,
        desc="Extended PF analysis",
        disable=not show_progress,
        leave=False,
    )
    for step_name, step_fn in ext_iter:
        ext_iter.set_postfix_str(step_name)
        extended[step_name] = step_fn()

    rep_activity = extended["repetition activity"]
    form_prob = extended["formation probability"]
    cell_disp = extended["cell displacement"]
    consistent = extended["consistent PFs"]
    pc_cohort_retention = extended["PC cohort retention"]
    pc_stability = extended["PC stability by repetition"]
    pf_stability, pf_onsets = extended["PF stability by repetition"]

    return TwoRoomsPFResults(
        data=data,
        by_room=by_room,
        birth_room_counts=birth_counts,
        repetition_activity=rep_activity,
        formation_probability=form_prob,
        cell_displacement=cell_disp,
        consistent_pf_counts=consistent,
        pc_cohort_retention=pc_cohort_retention,
        pc_stability_by_repetition=pc_stability,
        pf_stability_by_repetition=pf_stability,
        pf_onset_cdf_samples=pf_onsets,
    )


def final_ratemaps_by_room(data: TwoRoomsData) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for room_idx in ROOM_IDXS:
        refs = data.capture_refs_by_room[room_idx]
        final_capture_idx = len(refs) - 1
        out[room_idx] = load_gaussian_capture(
            final_capture_idx,
            gaussian_params=data.gaussian_params_by_room[room_idx],
            field_shape=data.field_shape,
        )
    return out


def final_peak_rates_by_room(data: TwoRoomsData) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for room_idx in ROOM_IDXS:
        refs = data.capture_refs_by_room[room_idx]
        final_capture_idx = len(refs) - 1
        out[room_idx] = data.gaussian_signal_max_by_room[room_idx][final_capture_idx]
    return out
