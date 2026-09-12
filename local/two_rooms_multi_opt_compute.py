"""Compute pipeline for multi-optimizer two_rooms comparison."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiments.common.paths import gaussian_rf_fits_path
from experiments.common.run import create_experiment, load_experiment_config
from experiments.common.signals_io import (
    TWO_ROOMS_NAME,
    load_delta_w_segment_stats_bundle,
    load_effective_lr_segment_stats_bundle,
    load_training_gradient_timeline,
    load_training_loss_timeline,
)
from optimizers.defaults import build_optimizer_config
from single_room_multi_opt_compute import (
    OPTIMIZER_ORDER,
    OptimizerEffLrAnalysis,
    OptimizerGaussianData,
    OptimizerMetaData,
    OptimizerPFData,
    OptimizerRunContext,
    OptimizerTrainingData,
    MultiOptResults,
    compute_optimizer_eff_lr_analysis,
    optimizer_supports_eff_lr,
)
from two_rooms_pf_analysis import (
    ROOM_IDXS,
    RoomPFResults,
    TwoRoomsData,
    TwoRoomsPFResults,
    compute_all_two_rooms_pf_results,
    dead_threshold_from_frac,
    final_peak_rates_by_room,
    load_two_rooms_data,
    plot_metrics_train_same,
)
from single_room_pf_analysis import (
    TWO_ROOMS_NAME as PF_TWO_ROOMS_NAME,
    extract_pf_revive_events,
)

OPTIMIZER_COLORS_LIGHT = {
    "sgd": "#6c8fe9",
    "adagrad": "#b0de82",
    "adam": "#e38749",
    "pure_shampoo": "#9feaea",
    "grafted_shampoo": "#e96c84",
}
OPTIMIZER_COLORS_DARK = {
    "sgd": "#2a5cdf",
    "adagrad": "#89ce46",
    "adam": "#c25f1e",
    "pure_shampoo": "#60dcdc",
    "grafted_shampoo": "#df2a4c",
}
TRAIN_OTHER_COLOR = "0.55"
ROOM_SWITCH_COLOR = "#ffbf29"
REP_SWITCH_COLOR = "k"
REP_SWITCH_LW = 2.5
ROOM_SWITCH_LW = 1.5


@dataclass
class TwoRoomsMultiOptResults:
    contexts: dict[str, TwoRoomsData]
    pf: dict[str, TwoRoomsPFResults] = field(default_factory=dict)
    training: dict[str, OptimizerTrainingData] = field(default_factory=dict)
    dead_threshold_by_optimizer: dict[str, float] = field(default_factory=dict)
    eff_lr: dict[str, OptimizerEffLrAnalysis] = field(default_factory=dict)
    eff_lr_sustained: dict[str, OptimizerEffLrAnalysis] = field(default_factory=dict)
    eff_lr_transient: dict[str, OptimizerEffLrAnalysis] = field(default_factory=dict)
    pf_cohort_labels: dict[str, pd.DataFrame] = field(default_factory=dict)
    shared_ratemap_vmax: float = 1.0
    mean_wsm_by_room: dict[int, np.ndarray] = field(default_factory=dict)


def load_mean_wsm_by_room(*, config_path: Path) -> dict[int, np.ndarray]:
    run_config = load_experiment_config(config_path)
    experiment = create_experiment(run_config)
    return {
        room_idx: experiment.load_room(room_idx)[3].response_map.mean(axis=0)
        for room_idx in ROOM_IDXS
    }


def build_two_rooms_optimizer_contexts(
    *,
    project_root: Path,
    config_path: Path,
    optimizer_order: tuple[str, ...] = tuple(OPTIMIZER_ORDER),
    results_parent_override: str | None = None,
    show_progress: bool = True,
) -> dict[str, TwoRoomsData]:
    contexts: dict[str, TwoRoomsData] = {}
    opt_iter = tqdm(
        optimizer_order,
        desc="Loading two-rooms optimizer data",
        disable=not show_progress,
    )
    for optimizer_type in opt_iter:
        opt_iter.set_postfix(optimizer=optimizer_type)
        contexts[optimizer_type] = load_two_rooms_data(
            project_root=project_root,
            config_path=config_path,
            optimizer=optimizer_type,
            results_parent_override=results_parent_override,
        )
    return contexts


def compute_two_rooms_training_timeseries(
    data: TwoRoomsData,
    *,
    optimizer_type: str,
) -> OptimizerTrainingData:
    loss_ts = load_training_loss_timeline(
        data.paths.signals_dir,
        experiment_name=TWO_ROOMS_NAME,
        last_rep_only=True,
    )
    grad_ts = load_training_gradient_timeline(
        data.paths.signals_dir,
        segment_duration_s=data.segment_duration_s,
        experiment_name=TWO_ROOMS_NAME,
        last_rep_only=True,
    )
    delta_w_ts = load_delta_w_segment_stats_bundle(
        data.paths.signals_dir,
        segment_duration_s=data.segment_duration_s,
        experiment_name=TWO_ROOMS_NAME,
        last_rep_only=True,
    ).network
    if optimizer_supports_eff_lr(optimizer_type):
        try:
            eff_lr_bundle = load_effective_lr_segment_stats_bundle(
                data.paths.signals_dir,
                segment_duration_s=data.segment_duration_s,
                experiment_name=TWO_ROOMS_NAME,
                last_rep_only=True,
            )
            eff_lr_ts = eff_lr_bundle.network
        except KeyError:
            eff_lr_ts = grad_ts[
                [
                    "global_segment_idx",
                    "rep_id",
                    "visit_room_id",
                    "traj_id",
                    "segment_id",
                    "is_traj_start",
                ]
            ].copy()
            eff_lr_ts["effective_lr_mean"] = np.nan
            eff_lr_ts["time_s"] = (
                (eff_lr_ts["global_segment_idx"] + 0.5) * data.segment_duration_s
            )
    else:
        eff_lr_ts = grad_ts[
            [
                "global_segment_idx",
                "rep_id",
                "visit_room_id",
                "traj_id",
                "segment_id",
                "is_traj_start",
            ]
        ].copy()
        eff_lr_ts["effective_lr_mean"] = np.nan
        eff_lr_ts["time_s"] = (
            (eff_lr_ts["global_segment_idx"] + 0.5) * data.segment_duration_s
        )
    return OptimizerTrainingData(
        loss_ts=loss_ts,
        grad_ts=grad_ts,
        eff_lr_ts=eff_lr_ts,
        delta_w_ts=delta_w_ts,
    )


def aggregate_train_same_metrics(
    pf_results: TwoRoomsPFResults,
    *,
    by_room: bool,
) -> dict[str, np.ndarray] | dict[int, dict[str, np.ndarray]]:
    if by_room:
        out: dict[int, dict[str, np.ndarray]] = {}
        for room_idx in ROOM_IDXS:
            metrics = plot_metrics_train_same(pf_results.by_room[room_idx])
            out[room_idx] = {
                col: metrics[col].dropna().to_numpy(dtype=float)
                for col in metrics.columns
                if col not in {"cell_idx", "pf_idx", "birth_capture_idx"}
            }
        return out
    pooled = pd.concat(
        [
            plot_metrics_train_same(pf_results.by_room[room_idx]).assign(
                eval_room_id=room_idx
            )
            for room_idx in ROOM_IDXS
        ],
        ignore_index=True,
    )
    return {
        col: pooled[col].dropna().to_numpy(dtype=float)
        for col in pooled.columns
        if col not in {"cell_idx", "pf_idx", "birth_capture_idx", "eval_room_id"}
    }


def build_pooled_pf_cohort_labels(pf_results: TwoRoomsPFResults) -> pd.DataFrame:
    """One row per train_same PF with sustained/transient label (birth-day cohort)."""
    samples = pf_results.pf_onset_cdf_samples
    birth = samples[samples["cohort"] == "birth_day"].copy()
    return birth[
        ["eval_room_id", "cell_idx", "pf_idx", "label"]
    ].drop_duplicates(subset=["eval_room_id", "cell_idx", "pf_idx"])


def _two_rooms_data_to_run_context(
    data: TwoRoomsData,
    *,
    optimizer_type: str,
    eval_room_id: int = 0,
) -> OptimizerRunContext:
    return OptimizerRunContext(
        optimizer_type=optimizer_type,
        optimizer_tag=data.optimizer_tag,
        paths=data.paths,
        capture_refs=data.capture_refs_by_room[eval_room_id],
        tag_cache=data.tag_cache,
        fits_path=gaussian_rf_fits_path(
            data.paths.results_dir,
            PF_TWO_ROOMS_NAME,
            eval_room_id,
        ),
        segment_duration_s=data.segment_duration_s,
        n_neurons=data.n_neurons,
    )


def _room_pf_to_optimizer_pf_data(room: RoomPFResults) -> OptimizerPFData:
    return OptimizerPFData(
        alive_ts=room.alive_ts,
        pf_segment_df=room.pf_segment_df,
        pf_transition_counts=room.pf_transition_counts,
        gauss_transition_props=room.gauss_transition_props,
        pf_per_neuron=room.pf_per_neuron,
        plot_metrics=plot_metrics_train_same(room),
        global_pf_metrics=room.pf_metrics_df,
        life_period_metrics=room.pf_metrics_df,
        revive_events=extract_pf_revive_events(room.pf_segment_df),
        displacement_curves=room.displacement_curves,
    )


def _load_gaussian_data_for_room(
    data: TwoRoomsData,
    *,
    eval_room_id: int,
    dead_threshold: float,
) -> OptimizerGaussianData:
    return OptimizerGaussianData(
        master_df=data.master_df_by_room[eval_room_id],
        gaussian_params=data.gaussian_params_by_room[eval_room_id],
        gaussian_signal_max=data.gaussian_signal_max_by_room[eval_room_id],
        field_shape=data.field_shape,
        dead_threshold=dead_threshold,
        percentile99_signal_max=data.percentile99_signal_max,
    )


def _pf_key_set(labels: pd.DataFrame, cohort: str) -> set[tuple[int, int, int]]:
    sub = labels.loc[labels["label"] == cohort]
    return {
        (int(r.eval_room_id), int(r.cell_idx), int(r.pf_idx))
        for r in sub.itertuples(index=False)
    }


def _add_eval_room_id(groups: dict[str, pd.DataFrame], eval_room_id: int) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for variant, df in groups.items():
        if df.empty:
            out[variant] = df.copy()
            continue
        tagged = df.copy()
        tagged["eval_room_id"] = int(eval_room_id)
        out[variant] = tagged
    return out


def _merge_group_variants(
    left: dict[str, pd.DataFrame],
    right: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for variant in left:
        frames = [left[variant]]
        if variant in right and not right[variant].empty:
            frames.append(right[variant])
        out[variant] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out


def _filter_group_variants_by_keys(
    groups: dict[str, pd.DataFrame],
    keys: set[tuple[int, int, int]],
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for variant, df in groups.items():
        if df.empty or not keys:
            out[variant] = df.iloc[0:0].copy()
            continue
        if "eval_room_id" not in df.columns:
            filtered = df[
                df.apply(
                    lambda row: (0, int(row["cell_idx"]), int(row["pf_idx"])) in keys,
                    axis=1,
                )
            ]
        else:
            filtered = df[
                df.apply(
                    lambda row: (
                        int(row["eval_room_id"]),
                        int(row["cell_idx"]),
                        int(row["pf_idx"]),
                    )
                    in keys,
                    axis=1,
                )
            ]
        out[variant] = filtered.reset_index(drop=True)
    return out


def _filter_lifecycle_counts(
    counts: pd.DataFrame,
    keys: set[tuple[int, int, int]],
) -> pd.DataFrame:
    if counts.empty or not keys:
        return counts.iloc[0:0].copy()
    if "eval_room_id" in counts.columns:
        mask = counts.apply(
            lambda row: (
                int(row["eval_room_id"]),
                int(row["cell_idx"]),
                int(row["pf_idx"]),
            )
            in keys,
            axis=1,
        )
    else:
        mask = counts.apply(
            lambda row: (0, int(row["cell_idx"]), int(row["pf_idx"])) in keys,
            axis=1,
        )
    return counts.loc[mask].reset_index(drop=True)


def _filter_eligibility_df(
    eligibility_df: pd.DataFrame,
    keys: set[tuple[int, int, int]],
    *,
    eval_room_id: int,
) -> pd.DataFrame:
    if eligibility_df.empty or not keys:
        return eligibility_df.iloc[0:0].copy()
    key_pairs = {(cell, pf) for room, cell, pf in keys if room == eval_room_id}
    if not key_pairs:
        return eligibility_df.iloc[0:0].copy()
    mask = eligibility_df.apply(
        lambda row: (int(row["cell_idx"]), int(row["pf_idx"])) in key_pairs,
        axis=1,
    )
    return eligibility_df.loc[mask].reset_index(drop=True)


def _filter_sample_pf_pairs(
    sample_pf_pairs: list[tuple[int, int]],
    keys: set[tuple[int, int, int]],
    *,
    eval_room_id: int,
) -> list[tuple[int, int]]:
    key_pairs = {(cell, pf) for room, cell, pf in keys if room == eval_room_id}
    return [pair for pair in sample_pf_pairs if pair in key_pairs]


def _merge_eff_lr_analyses(
    room0: OptimizerEffLrAnalysis,
    room1: OptimizerEffLrAnalysis,
) -> OptimizerEffLrAnalysis:
    def _tag_counts(df: pd.DataFrame, eval_room_id: int) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        return df.assign(eval_room_id=int(eval_room_id))

    lifecycle_outlier_counts = pd.concat(
        [
            _tag_counts(room0.lifecycle_outlier_counts, 0),
            _tag_counts(room1.lifecycle_outlier_counts, 1),
        ],
        ignore_index=True,
    )
    dw_lifecycle_outlier_counts = pd.concat(
        [
            _tag_counts(room0.dw_lifecycle_outlier_counts, 0),
            _tag_counts(room1.dw_lifecycle_outlier_counts, 1),
        ],
        ignore_index=True,
    )
    lifecycle_probs = pd.concat(
        [room0.lifecycle_probs, room1.lifecycle_probs],
        ignore_index=True,
    )
    dw_lifecycle_probs = pd.concat(
        [room0.dw_lifecycle_probs, room1.dw_lifecycle_probs],
        ignore_index=True,
    )
    return OptimizerEffLrAnalysis(
        capture_to_global=room0.capture_to_global,
        pf_eff_lr_groups=_merge_group_variants(
            room0.pf_eff_lr_groups,
            room1.pf_eff_lr_groups,
        ),
        pf_delta_w_groups=_merge_group_variants(
            room0.pf_delta_w_groups,
            room1.pf_delta_w_groups,
        ),
        eligibility_df=pd.concat(
            [
                room0.eligibility_df.assign(eval_room_id=0),
                room1.eligibility_df.assign(eval_room_id=1),
            ],
            ignore_index=True,
        ),
        sample_pf_pairs=room0.sample_pf_pairs + room1.sample_pf_pairs,
        window_outlier_counts=pd.concat(
            [room0.window_outlier_counts, room1.window_outlier_counts],
            ignore_index=True,
        ),
        fit_results_random=room0.fit_results_random,
        lifecycle_outlier_counts=lifecycle_outlier_counts,
        lifecycle_probs=lifecycle_probs,
        dw_window_outlier_counts=pd.concat(
            [room0.dw_window_outlier_counts, room1.dw_window_outlier_counts],
            ignore_index=True,
        ),
        dw_fit_results_random=room0.dw_fit_results_random,
        dw_lifecycle_outlier_counts=dw_lifecycle_outlier_counts,
        dw_lifecycle_probs=dw_lifecycle_probs,
    )


def _filter_eff_lr_analysis_by_cohort(
    analysis: OptimizerEffLrAnalysis,
    keys: set[tuple[int, int, int]],
) -> OptimizerEffLrAnalysis:
    filtered = deepcopy(analysis)
    filtered.pf_eff_lr_groups = _filter_group_variants_by_keys(
        analysis.pf_eff_lr_groups,
        keys,
    )
    filtered.pf_delta_w_groups = _filter_group_variants_by_keys(
        analysis.pf_delta_w_groups,
        keys,
    )
    filtered.lifecycle_outlier_counts = _filter_lifecycle_counts(
        analysis.lifecycle_outlier_counts,
        keys,
    )
    filtered.dw_lifecycle_outlier_counts = _filter_lifecycle_counts(
        analysis.dw_lifecycle_outlier_counts,
        keys,
    )
    if not analysis.lifecycle_probs.empty:
        pf_cols = ["eval_room_id", "cell_idx", "pf_idx"]
        if all(c in analysis.lifecycle_probs.columns for c in pf_cols):
            mask = analysis.lifecycle_probs.apply(
                lambda row: (
                    int(row["eval_room_id"]),
                    int(row["cell_idx"]),
                    int(row["pf_idx"]),
                )
                in keys,
                axis=1,
            )
            filtered.lifecycle_probs = analysis.lifecycle_probs.loc[mask].reset_index(
                drop=True
            )
    if not analysis.dw_lifecycle_probs.empty:
        pf_cols = ["eval_room_id", "cell_idx", "pf_idx"]
        if all(c in analysis.dw_lifecycle_probs.columns for c in pf_cols):
            mask = analysis.dw_lifecycle_probs.apply(
                lambda row: (
                    int(row["eval_room_id"]),
                    int(row["cell_idx"]),
                    int(row["pf_idx"]),
                )
                in keys,
                axis=1,
            )
            filtered.dw_lifecycle_probs = analysis.dw_lifecycle_probs.loc[
                mask
            ].reset_index(drop=True)
    filtered.eligibility_df = analysis.eligibility_df
    return filtered


def _compute_single_optimizer_two_rooms(
    task: tuple[str, TwoRoomsData, float, float, float, str, float, float, float, str, bool],
) -> tuple[str, TwoRoomsPFResults, OptimizerTrainingData, float, np.ndarray]:
    (
        optimizer_type,
        data,
        dead_threshold_frac,
        r2_threshold,
        num_gauss_threshold,
        pf_tracking_method,
        zoom_bin_width,
        constant_l2_similarity,
        max_factor_pct,
        peak_analysis_technique,
        show_progress,
    ) = task
    dead_threshold = dead_threshold_from_frac(
        data.gaussian_signal_max_by_room,
        dead_threshold_frac=dead_threshold_frac,
    )
    pf_results = compute_all_two_rooms_pf_results(
        data,
        dead_threshold=dead_threshold,
        r2_threshold=r2_threshold,
        num_gauss_threshold=num_gauss_threshold,
        pf_tracking_method=pf_tracking_method,
        constant_l2_similarity=constant_l2_similarity,
        zoom_bin_width=zoom_bin_width,
        max_factor_pct=max_factor_pct,
        last_epoch_only=False,
        show_progress=show_progress,
    )
    training = compute_two_rooms_training_timeseries(
        data,
        optimizer_type=optimizer_type,
    )
    peaks = final_peak_rates_by_room(data)
    pooled_peaks = np.concatenate([peaks[r].ravel() for r in ROOM_IDXS])
    pooled_peaks = pooled_peaks[np.isfinite(pooled_peaks)]
    return optimizer_type, pf_results, training, dead_threshold, pooled_peaks


def compute_all_two_rooms_optimizer_results(
    contexts: dict[str, TwoRoomsData],
    *,
    r2_threshold: float,
    num_gauss_threshold: float,
    pf_tracking_method: str,
    dead_threshold_frac: float,
    zoom_bin_width: float,
    constant_l2_similarity: float = 4.0,
    max_factor_pct: float = 1.0,
    peak_analysis_technique: str = "local",
    mean_wsm_by_room: dict[int, np.ndarray] | None = None,
    show_progress: bool = True,
) -> TwoRoomsMultiOptResults:
    tasks = [
        (
            optimizer_type,
            contexts[optimizer_type],
            dead_threshold_frac,
            r2_threshold,
            num_gauss_threshold,
            pf_tracking_method,
            zoom_bin_width,
            constant_l2_similarity,
            max_factor_pct,
            peak_analysis_technique,
            show_progress,
        )
        for optimizer_type in OPTIMIZER_ORDER
        if optimizer_type in contexts
    ]

    results = TwoRoomsMultiOptResults(contexts=contexts)
    if mean_wsm_by_room is not None:
        results.mean_wsm_by_room = mean_wsm_by_room

    all_peaks: list[np.ndarray] = []
    opt_iter = tqdm(
        tasks,
        total=len(tasks),
        desc="Computing two-rooms optimizer results",
        disable=not show_progress,
    )
    for task in opt_iter:
        optimizer_type = task[0]
        opt_iter.set_postfix(optimizer=optimizer_type)
        (
            optimizer_type,
            pf_results,
            training,
            dead_threshold,
            peaks,
        ) = _compute_single_optimizer_two_rooms(task)
        results.pf[optimizer_type] = pf_results
        results.training[optimizer_type] = training
        results.dead_threshold_by_optimizer[optimizer_type] = dead_threshold
        results.pf_cohort_labels[optimizer_type] = build_pooled_pf_cohort_labels(
            pf_results
        )
        all_peaks.append(peaks)

    pooled = np.concatenate(all_peaks) if all_peaks else np.array([], dtype=float)
    pooled = pooled[np.isfinite(pooled)]
    results.shared_ratemap_vmax = (
        float(np.nanpercentile(pooled, 99.9)) if pooled.size else 1.0
    )
    return results


def compute_all_two_rooms_eff_lr_analyses(
    results: TwoRoomsMultiOptResults,
    *,
    window_size: int,
    n_random_windows: int,
    n_network_random_pfs: int,
    n_window_samples: int,
    outlier_method: str,
    outlier_layer: str,
    eff_lr_outlier_fit_type_by_optimizer: dict[str, str],
    delta_w_outlier_fit_type_by_optimizer: dict[str, str],
    eff_lr_n_workers: int,
    aggregate_n_process: int,
    aggregate_batch_size: int,
    window_outlier_seed: int,
    min_pre_birth_absent: int,
    min_post_birth_active: int,
    min_revive_pre_absent: int,
    min_revive_active: int,
    eff_lr_outlier_fit_comparison: dict[str, tuple[str, ...]] | None = None,
    delta_w_outlier_fit_comparison: dict[str, tuple[str, ...]] | None = None,
    show_progress: bool = True,
) -> TwoRoomsMultiOptResults:
    optimizer_types = [opt for opt in OPTIMIZER_ORDER if opt in results.contexts]
    opt_iter = tqdm(
        optimizer_types,
        desc="Two-rooms effective-LR analysis",
        disable=not show_progress,
    )
    for optimizer_type in opt_iter:
        opt_iter.set_postfix(optimizer=optimizer_type)
        data = results.contexts[optimizer_type]
        pf_results = results.pf[optimizer_type]
        training = results.training[optimizer_type]
        dead_threshold = results.dead_threshold_by_optimizer[optimizer_type]
        optimizer_config = build_optimizer_config(optimizer_type)
        include_eff_lr = optimizer_supports_eff_lr(optimizer_type)

        room_analyses: list[OptimizerEffLrAnalysis] = []
        room_iter = tqdm(
            ROOM_IDXS,
            desc=f"eff_lr rooms ({optimizer_type})",
            leave=False,
            disable=not show_progress,
        )
        for eval_room_id in room_iter:
            room_iter.set_postfix(room=eval_room_id)
            ctx = _two_rooms_data_to_run_context(
                data,
                optimizer_type=optimizer_type,
                eval_room_id=eval_room_id,
            )
            gaussian_data = _load_gaussian_data_for_room(
                data,
                eval_room_id=eval_room_id,
                dead_threshold=dead_threshold,
            )
            pf_data = _room_pf_to_optimizer_pf_data(pf_results.by_room[eval_room_id])
            try:
                analysis = compute_optimizer_eff_lr_analysis(
                    ctx,
                    pf_data,
                    gaussian_data,
                    training,
                    optimizer_config=optimizer_config,
                    window_size=int(window_size),
                    n_random_windows=int(n_random_windows),
                    n_network_random_pfs=int(n_network_random_pfs),
                    n_window_samples=int(n_window_samples),
                    outlier_method=str(outlier_method),
                    outlier_layer=str(outlier_layer),
                    eff_lr_outlier_fit_type_by_optimizer=eff_lr_outlier_fit_type_by_optimizer,
                    delta_w_outlier_fit_type_by_optimizer=delta_w_outlier_fit_type_by_optimizer,
                    eff_lr_n_workers=int(eff_lr_n_workers),
                    aggregate_n_process=int(aggregate_n_process),
                    aggregate_batch_size=int(aggregate_batch_size),
                    window_outlier_seed=int(window_outlier_seed),
                    min_pre_birth_absent=int(min_pre_birth_absent),
                    min_post_birth_active=int(min_post_birth_active),
                    min_revive_pre_absent=int(min_revive_pre_absent),
                    min_revive_active=int(min_revive_active),
                    eff_lr_outlier_fit_comparison=eff_lr_outlier_fit_comparison,
                    delta_w_outlier_fit_comparison=delta_w_outlier_fit_comparison,
                    include_eff_lr=include_eff_lr,
                )
            except (KeyError, FileNotFoundError, ValueError) as exc:
                if include_eff_lr:
                    print(
                        f"Skipping room {eval_room_id} eff_lr for {optimizer_type}: {exc}"
                    )
                    continue
                raise
            analysis.pf_eff_lr_groups = _add_eval_room_id(
                analysis.pf_eff_lr_groups,
                eval_room_id,
            )
            analysis.pf_delta_w_groups = _add_eval_room_id(
                analysis.pf_delta_w_groups,
                eval_room_id,
            )
            if not analysis.lifecycle_outlier_counts.empty:
                analysis.lifecycle_outlier_counts = (
                    analysis.lifecycle_outlier_counts.assign(
                        eval_room_id=eval_room_id
                    )
                )
            if not analysis.dw_lifecycle_outlier_counts.empty:
                analysis.dw_lifecycle_outlier_counts = (
                    analysis.dw_lifecycle_outlier_counts.assign(
                        eval_room_id=eval_room_id
                    )
                )
            room_analyses.append(analysis)

        if not room_analyses:
            continue
        merged = room_analyses[0]
        for extra in room_analyses[1:]:
            merged = _merge_eff_lr_analyses(merged, extra)
        results.eff_lr[optimizer_type] = merged

        labels = results.pf_cohort_labels[optimizer_type]
        sustained_keys = _pf_key_set(labels, "sustained")
        transient_keys = _pf_key_set(labels, "transient")
        results.eff_lr_sustained[optimizer_type] = _filter_eff_lr_analysis_by_cohort(
            merged,
            sustained_keys,
        )
        results.eff_lr_transient[optimizer_type] = _filter_eff_lr_analysis_by_cohort(
            merged,
            transient_keys,
        )

    return results


def to_multi_opt_view(
    results: TwoRoomsMultiOptResults,
    *,
    pool_rooms: bool = False,
) -> MultiOptResults:
    """Adapter so single-room Section 3 plot functions can be reused."""
    view = MultiOptResults(contexts={})
    for optimizer_type in OPTIMIZER_ORDER:
        if optimizer_type not in results.contexts:
            continue
        data = results.contexts[optimizer_type]
        pf_results = results.pf[optimizer_type]
        dead_threshold = results.dead_threshold_by_optimizer[optimizer_type]
        room0 = pf_results.by_room[0]

        view.contexts[optimizer_type] = _two_rooms_data_to_run_context(
            data,
            optimizer_type=optimizer_type,
            eval_room_id=0,
        )
        view.training[optimizer_type] = results.training[optimizer_type]
        view.gaussian[optimizer_type] = _load_gaussian_data_for_room(
            data,
            eval_room_id=0,
            dead_threshold=dead_threshold,
        )
        view.meta[optimizer_type] = OptimizerMetaData(
            mean_r2_ts=room0.mean_r2_ts,
            final_ratemap=np.zeros(data.field_shape, dtype=float),
            final_peak_rates=np.array([], dtype=float),
        )
        pf_data = _room_pf_to_optimizer_pf_data(room0)
        if pool_rooms:
            pooled_plot_metrics = pd.concat(
                [
                    plot_metrics_train_same(pf_results.by_room[room_idx])
                    for room_idx in ROOM_IDXS
                ],
                ignore_index=True,
            )
            pooled_life_period_metrics = pd.concat(
                [pf_results.by_room[room_idx].pf_metrics_df for room_idx in ROOM_IDXS],
                ignore_index=True,
            )
            pf_data = replace(
                pf_data,
                plot_metrics=pooled_plot_metrics,
                global_pf_metrics=pooled_life_period_metrics,
                life_period_metrics=pooled_life_period_metrics,
            )
        view.pf[optimizer_type] = pf_data
        if optimizer_type in results.eff_lr:
            view.eff_lr[optimizer_type] = results.eff_lr[optimizer_type]
    return view


def to_cohort_multi_opt_view(
    results: TwoRoomsMultiOptResults,
    *,
    cohort: str,
) -> MultiOptResults:
    """MultiOptResults view with eff_lr dict swapped to a sustained/transient cohort."""
    view = to_multi_opt_view(results)
    cohort_attr = "eff_lr_sustained" if cohort == "sustained" else "eff_lr_transient"
    cohort_dict = getattr(results, cohort_attr)
    for optimizer_type in OPTIMIZER_ORDER:
        if optimizer_type in cohort_dict:
            view.eff_lr[optimizer_type] = cohort_dict[optimizer_type]
    return view
