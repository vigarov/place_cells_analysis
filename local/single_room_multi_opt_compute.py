"""Compute pipeline for multi-optimizer single_room comparison."""

from __future__ import annotations

import multiprocessing
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from analysis.gaussian_fit_cycle_analysis import gaussian_fields_from_params
from experiments.common.paths import ExperimentPaths, gaussian_rf_fits_path
from experiments.common.ratemaps_io import (
    RatemapCaptureRef,
    discover_ratemap_captures,
    load_ratemap_capture,
)
from experiments.common.run import create_experiment, load_experiment_config
from experiments.common.signals_io import (
    load_delta_w_segment_stats_bundle,
    load_effective_lr_segment_stats_bundle,
    load_training_gradient_timeline,
    load_training_loss_timeline,
)
from optimizers.defaults import build_optimizer_config
from single_room_pf_analysis import (
    PF_GLOCAL_VARIANTS,
    add_pf_lifecycle_outlier_empirical_probabilities,
    add_pf_lifecycle_outlier_fit_probabilities,
    attach_first_birth_formation_onset,
    alive_proportion_by_segment,
    bin_proportion_histogram,
    build_capture_to_global_segment_map,
    build_master_fits_df,
    build_outlier_count_fit_comparison,
    collect_pf_window_and_lifecycle_delta_w_outlier_counts,
    collect_pf_window_and_lifecycle_outlier_counts,
    compute_pf_delta_w_group_metrics,
    compute_pf_effective_lr_group_metrics,
    extract_pf_revive_events,
    gauss_state_transition_counts,
    load_segment_duration_s,
    mean_pfs_per_neuron_stats,
    mean_r2_by_segment,
    pf_state_transition_counts,
    pf_tracked_displacement_samples,
    select_pfs_random_from_birth_revive_qualified,
    summarize_pf_eff_lr_eligibility,
    summarize_place_field_life_period_metrics,
    summarize_place_field_metrics,
    track_place_fields,
)

USE_GAUSSIANS = True
OPTIMIZER_ORDER = ["sgd", "adagrad", "adam", "pure_shampoo", "grafted_shampoo"]
OPTIMIZERS_WITHOUT_EFF_LR = frozenset({"sgd"})
EFF_LR_OPTIMIZER_ORDER = tuple(
    opt for opt in OPTIMIZER_ORDER if opt not in OPTIMIZERS_WITHOUT_EFF_LR
)


def optimizer_supports_eff_lr(optimizer_type: str) -> bool:
    return optimizer_type not in OPTIMIZERS_WITHOUT_EFF_LR


OPTIMIZER_COLORS = {
    "sgd": "#567ee6",
    "adagrad": "#a3d96e",
    "adam": "#e07833",
    "pure_shampoo": "#8ae5e5",
    "grafted_shampoo": "#e65671",
}
OPTIMIZER_LABELS = {
    "sgd": "SGD",
    "adagrad": "AdaGrad",
    "adam": "Adam",
    "pure_shampoo": "Pure Shampoo",
    "grafted_shampoo": "Grafted Shampoo",
}
EXTRA_COLORS = ["#97b4ed", "#f1c3b1", "#fffe93"]
N_COLS, N_ROWS = 25, 40

TIME_UNIT_SECONDS = "seconds"
TIME_UNIT_ITERATIONS = "iterations"

PEAK_ANALYSIS_TECHNIQUE_GLOBAL = "global"
PEAK_ANALYSIS_TECHNIQUE_LOCAL = "local"
PEAK_ANALYSIS_TECHNIQUES = (
    PEAK_ANALYSIS_TECHNIQUE_GLOBAL,
    PEAK_ANALYSIS_TECHNIQUE_LOCAL,
)


def formation_metric_spec(time_unit: str) -> tuple[str, str]:
    if time_unit == TIME_UNIT_SECONDS:
        return "formation_onset_s", "Formation onset (s)"
    if time_unit == TIME_UNIT_ITERATIONS:
        return "start_iteration_offset", "Formation onset (iterations)"
    raise ValueError(
        f"Unknown time_unit {time_unit!r}; expected "
        f"{TIME_UNIT_SECONDS!r} or {TIME_UNIT_ITERATIONS!r}"
    )


def tma_metric_spec(time_unit: str) -> tuple[str, str]:
    if time_unit == TIME_UNIT_SECONDS:
        return "tma_s", "RAT (s)"
    if time_unit == TIME_UNIT_ITERATIONS:
        return "tma_iterations", "Representation Acquisition Time (iterations)"
    raise ValueError(
        f"Unknown time_unit {time_unit!r}; expected "
        f"{TIME_UNIT_SECONDS!r} or {TIME_UNIT_ITERATIONS!r}"
    )


def pf_metric_boxplot_panels(time_unit: str) -> list[tuple[str, str]]:
    formation_col, formation_label = formation_metric_spec(time_unit)
    tma_col, tma_label = tma_metric_spec(time_unit)
    return [
        ("length_s", "PF length (s)"),
        (formation_col, formation_label),
        (tma_col, tma_label),
        ("peak_amplitude", "Peak amplitude"),
        ("revives_per_pf", "Revives per place field"),
        ("drift_x", "Drift x (cm)"),
        ("drift_y", "Drift y (cm)"),
        ("robustness", "Mean drift (cm)"),
    ]


def pf_metric_bar_panel_specs(time_unit: str) -> list[tuple[str, str | None, str | None]]:
    """Return panel specs for bar plots: kind, metric_col, ylabel."""
    formation_col, formation_label = formation_metric_spec(time_unit)
    return [
        ("simple", "length_s", "PF length (s)"),
        ("simple", formation_col, formation_label),
        ("amp_tma", None, None),
        ("spontaneous", None, None),
        ("simple", "revives_per_pf", "Revives per place field"),
        ("drift", None, None),
    ]

_GAUSS_COUNT_COLS = ("n_1_to_2", "n_2_to_1", "n_1_to_1", "n_2_to_2")
_GAUSS_PROP_COLS = ("prop_1_to_2", "prop_2_to_1", "prop_1_to_1", "prop_2_to_2")

_DISPLACEMENT_PANELS = (
    {
        "col": "displacement_l2",
        "label": "PF shift L2 (cm)",
        "zoom_range": (0.0, 15.0),
    },
    {
        "col": "displacement_x",
        "label": "PF shift Δx (cm)",
        "zoom_range": (-8.0, 8.0),
    },
    {
        "col": "displacement_y",
        "label": "PF shift Δy (cm)",
        "zoom_range": (-8.0, 8.0),
    },
)


@dataclass(frozen=True)
class OptimizerRunContext:
    optimizer_type: str
    optimizer_tag: str
    paths: ExperimentPaths
    capture_refs: list[RatemapCaptureRef]
    tag_cache: dict[Path, list[str]]
    fits_path: Path
    segment_duration_s: float
    n_neurons: int


@dataclass
class OptimizerGaussianData:
    master_df: pd.DataFrame
    gaussian_params: np.ndarray
    gaussian_signal_max: np.ndarray
    field_shape: tuple[int, int]
    dead_threshold: float
    percentile99_signal_max: float


@dataclass
class OptimizerTrainingData:
    loss_ts: pd.DataFrame
    grad_ts: pd.DataFrame
    eff_lr_ts: pd.DataFrame
    delta_w_ts: pd.DataFrame


@dataclass
class OptimizerEffLrAnalysis:
    capture_to_global: pd.Series
    pf_eff_lr_groups: dict[str, pd.DataFrame]
    pf_delta_w_groups: dict[str, pd.DataFrame]
    eligibility_df: pd.DataFrame
    sample_pf_pairs: list[tuple[int, int]]
    window_outlier_counts: pd.DataFrame
    fit_results_random: dict[str, object]
    lifecycle_outlier_counts: pd.DataFrame
    lifecycle_probs: pd.DataFrame
    dw_window_outlier_counts: pd.DataFrame
    dw_fit_results_random: dict[str, object]
    dw_lifecycle_outlier_counts: pd.DataFrame
    dw_lifecycle_probs: pd.DataFrame


@dataclass
class OptimizerMetaData:
    mean_r2_ts: pd.DataFrame
    final_ratemap: np.ndarray
    final_peak_rates: np.ndarray


@dataclass
class OptimizerPFData:
    alive_ts: pd.DataFrame
    pf_segment_df: pd.DataFrame
    pf_transition_counts: pd.DataFrame
    gauss_transition_props: pd.DataFrame
    pf_per_neuron: dict[str, Any]
    plot_metrics: pd.DataFrame
    global_pf_metrics: pd.DataFrame
    life_period_metrics: pd.DataFrame
    revive_events: pd.DataFrame
    displacement_curves: dict[str, tuple[np.ndarray, np.ndarray]]


@dataclass
class MultiOptResults:
    contexts: dict[str, OptimizerRunContext]
    gaussian: dict[str, OptimizerGaussianData] = field(default_factory=dict)
    training: dict[str, OptimizerTrainingData] = field(default_factory=dict)
    meta: dict[str, OptimizerMetaData] = field(default_factory=dict)
    pf: dict[str, OptimizerPFData] = field(default_factory=dict)
    eff_lr: dict[str, OptimizerEffLrAnalysis] = field(default_factory=dict)
    shared_ratemap_vmax: float = 1.0


@dataclass(frozen=True)
class SignificantPair:
    opt_a: str
    opt_b: str
    raw_p: float
    holm_p: float
    stars: str


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
    params = gaussian_params[capture_idx]
    return _render_gaussian_cells(params, field_shape)


def apply_results_parent_override(
    paths: ExperimentPaths,
    *,
    experiment_name: str,
    project_root: Path,
    results_parent_override: str | None,
) -> ExperimentPaths:
    if results_parent_override is None:
        return paths
    default_results_root = project_root / "results" / experiment_name
    override_results_dir = (
        project_root / "results" / results_parent_override
    ) / paths.results_dir.relative_to(default_results_root)
    return ExperimentPaths(
        suffix=paths.suffix,
        results_dir=override_results_dir,
        signals_dir=override_results_dir / "signals",
        ratemaps_dir=override_results_dir / "ratemaps",
        ckpt_dir=paths.ckpt_dir,
    )


def build_optimizer_contexts(
    *,
    config_path: Path,
    project_root: Path,
    optimizer_order: tuple[str, ...] = tuple(OPTIMIZER_ORDER),
    results_parent_override: str | None = None,
) -> dict[str, OptimizerRunContext]:
    run_config = load_experiment_config(config_path)
    contexts: dict[str, OptimizerRunContext] = {}

    for optimizer_type in optimizer_order:
        if optimizer_type not in run_config.optimizers:
            raise ValueError(
                f"Optimizer {optimizer_type!r} not in config optimizers "
                f"{run_config.optimizers}"
            )
        optimizer_config = build_optimizer_config(optimizer_type)
        experiment = create_experiment(run_config, optimizer_config=optimizer_config)
        paths = experiment.resolve_paths()
        paths = apply_results_parent_override(
            paths,
            experiment_name=experiment.name,
            project_root=project_root,
            results_parent_override=results_parent_override,
        )

        for required in (paths.ratemaps_dir, paths.signals_dir):
            if not required.is_dir():
                raise FileNotFoundError(
                    f"Missing directory for {optimizer_type}: {required}"
                )

        fits_path = gaussian_rf_fits_path(paths.results_dir, experiment.name, room_idx=0)
        if not fits_path.is_file():
            raise FileNotFoundError(
                f"Missing Gaussian fits for {optimizer_type}: {fits_path}. "
                f"Run: uv run estimate-gaussians-rf --input {paths.results_dir}"
            )

        tag_cache: dict[Path, list[str]] = {}
        capture_refs = discover_ratemap_captures(
            paths.ratemaps_dir,
            experiment_name=experiment.name,
            tag_cache=tag_cache,
        )
        if not capture_refs:
            raise FileNotFoundError(f"No ratemaps under {paths.ratemaps_dir}")

        with np.load(fits_path) as fits:
            n_captures = int(np.asarray(fits["gaussian_params"]).shape[0])
        if n_captures != len(capture_refs):
            raise ValueError(
                f"{optimizer_type}: Gaussian fits have {n_captures} captures but "
                f"found {len(capture_refs)} ratemap captures"
            )

        segment_duration_s = load_segment_duration_s(paths.results_dir / "config.json")
        sample_rm = load_ratemap_capture(capture_refs[0])
        n_neurons = int(sample_rm.shape[0])

        contexts[optimizer_type] = OptimizerRunContext(
            optimizer_type=optimizer_type,
            optimizer_tag=experiment.optimizer_tag,
            paths=paths,
            capture_refs=capture_refs,
            tag_cache=tag_cache,
            fits_path=fits_path,
            segment_duration_s=segment_duration_s,
            n_neurons=n_neurons,
        )
    return contexts


def load_optimizer_gaussian_data(
    ctx: OptimizerRunContext,
    *,
    dead_threshold_frac: float,
) -> OptimizerGaussianData:
    with np.load(ctx.fits_path) as fits:
        gaussian_params = np.asarray(fits["gaussian_params"])
        gaussian_signal_max = np.asarray(fits["signal_max"], dtype=np.float64)

    sample_rm = load_ratemap_capture(ctx.capture_refs[0])
    field_shape = (int(sample_rm.shape[1]), int(sample_rm.shape[2]))
    master_df = build_master_fits_df(ctx.fits_path, ctx.capture_refs, ctx.tag_cache)
    percentile99_signal_max = float(np.nanpercentile(gaussian_signal_max, 99.5))
    dead_threshold = dead_threshold_frac * percentile99_signal_max
    return OptimizerGaussianData(
        master_df=master_df,
        gaussian_params=gaussian_params,
        gaussian_signal_max=gaussian_signal_max,
        field_shape=field_shape,
        dead_threshold=dead_threshold,
        percentile99_signal_max=percentile99_signal_max,
    )


def compute_training_timeseries(ctx: OptimizerRunContext) -> OptimizerTrainingData:
    loss_ts = load_training_loss_timeline(ctx.paths.signals_dir)
    grad_ts = load_training_gradient_timeline(
        ctx.paths.signals_dir,
        segment_duration_s=ctx.segment_duration_s,
    )
    delta_w_ts = load_delta_w_segment_stats_bundle(
        ctx.paths.signals_dir,
        segment_duration_s=ctx.segment_duration_s,
    ).network
    if optimizer_supports_eff_lr(ctx.optimizer_type):
        try:
            eff_lr_bundle = load_effective_lr_segment_stats_bundle(
                ctx.paths.signals_dir,
                segment_duration_s=ctx.segment_duration_s,
            )
            eff_lr_ts = eff_lr_bundle.network
        except KeyError:
            eff_lr_ts = grad_ts[
                ["global_segment_idx", "epoch_id", "traj_id", "segment_id", "is_traj_start"]
            ].copy()
            eff_lr_ts["effective_lr_mean"] = np.nan
            eff_lr_ts["time_s"] = (
                (eff_lr_ts["global_segment_idx"] + 0.5) * ctx.segment_duration_s
            )
    else:
        eff_lr_ts = grad_ts[
            ["global_segment_idx", "epoch_id", "traj_id", "segment_id", "is_traj_start"]
        ].copy()
        eff_lr_ts["effective_lr_mean"] = np.nan
        eff_lr_ts["time_s"] = (
            (eff_lr_ts["global_segment_idx"] + 0.5) * ctx.segment_duration_s
        )
    return OptimizerTrainingData(
        loss_ts=loss_ts,
        grad_ts=grad_ts,
        eff_lr_ts=eff_lr_ts,
        delta_w_ts=delta_w_ts,
    )


def compute_meta_data(
    ctx: OptimizerRunContext,
    gaussian_data: OptimizerGaussianData,
) -> OptimizerMetaData:
    mean_r2_ts = mean_r2_by_segment(gaussian_data.master_df)
    final_capture_idx = len(ctx.capture_refs) - 1
    final_ratemap = load_gaussian_capture(
        final_capture_idx,
        gaussian_params=gaussian_data.gaussian_params,
        field_shape=gaussian_data.field_shape,
    )
    final_peak_rates = gaussian_data.gaussian_signal_max[final_capture_idx]
    return OptimizerMetaData(
        mean_r2_ts=mean_r2_ts,
        final_ratemap=final_ratemap,
        final_peak_rates=final_peak_rates,
    )


def gauss_transition_counts_to_proportions(
    gauss_trans: pd.DataFrame,
    *,
    n_neurons: int,
) -> pd.DataFrame:
    if gauss_trans.empty:
        return gauss_trans.copy()
    out = gauss_trans.copy()
    denom = float(n_neurons)
    for count_col, prop_col in zip(_GAUSS_COUNT_COLS, _GAUSS_PROP_COLS, strict=True):
        out[prop_col] = out[count_col].astype(float) / denom
    return out


def compute_displacement_curves(
    disp_samples: pd.DataFrame,
    *,
    zoom_bin_width: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for panel in _DISPLACEMENT_PANELS:
        values = disp_samples[panel["col"]].dropna().to_numpy(dtype=float)
        centers, prop, _ = bin_proportion_histogram(
            values,
            bin_width=zoom_bin_width,
            bin_range=panel["zoom_range"],
        )
        curves[panel["col"]] = (centers, prop)
    return curves


def compute_pf_data(
    ctx: OptimizerRunContext,
    gaussian_data: OptimizerGaussianData,
    *,
    r2_threshold: float,
    num_gauss_threshold: float,
    pf_tracking_method: str,
    zoom_bin_width: float,
    constant_l2_similarity: float = 4.0,
    max_factor_pct: float = 1.0,
    peak_analysis_technique: str = PEAK_ANALYSIS_TECHNIQUE_GLOBAL,
) -> OptimizerPFData:
    master_df = gaussian_data.master_df
    dead_threshold = gaussian_data.dead_threshold

    alive_ts = alive_proportion_by_segment(
        master_df,
        dead_threshold=dead_threshold,
        r2_threshold=r2_threshold,
    )
    pf_segment_df = track_place_fields(
        master_df,
        dead_threshold=dead_threshold,
        r2_threshold=r2_threshold,
        gauss_threshold=num_gauss_threshold,
        segment_duration_s=ctx.segment_duration_s,
        pf_tracking_method=pf_tracking_method,
        constant_l2_similarity=constant_l2_similarity,
    )
    global_pf_metrics = summarize_place_field_metrics(
        pf_segment_df,
        segment_duration_s=ctx.segment_duration_s,
        max_factor_pct=max_factor_pct,
    )
    life_period_metrics = summarize_place_field_life_period_metrics(
        pf_segment_df,
        segment_duration_s=ctx.segment_duration_s,
        max_factor_pct=max_factor_pct,
    )
    if peak_analysis_technique == PEAK_ANALYSIS_TECHNIQUE_LOCAL:
        pf_metrics_df = life_period_metrics.merge(
            global_pf_metrics[["cell_idx", "pf_idx", "n_life_periods"]],
            on=["cell_idx", "pf_idx"],
            how="left",
        )
    elif peak_analysis_technique == PEAK_ANALYSIS_TECHNIQUE_GLOBAL:
        pf_metrics_df = global_pf_metrics.copy()
    else:
        raise ValueError(
            f"Unknown peak_analysis_technique {peak_analysis_technique!r}; "
            f"expected {PEAK_ANALYSIS_TECHNIQUE_GLOBAL!r} or "
            f"{PEAK_ANALYSIS_TECHNIQUE_LOCAL!r}"
        )

    pf_count_df = pf_metrics_df
    if peak_analysis_technique == PEAK_ANALYSIS_TECHNIQUE_LOCAL:
        pf_count_df = life_period_metrics.drop_duplicates(
            ["cell_idx", "pf_idx"],
            keep="first",
        )
    pf_per_neuron = mean_pfs_per_neuron_stats(
        pf_count_df,
        n_neurons=ctx.n_neurons,
    )
    pf_transition_counts = pf_state_transition_counts(
        pf_segment_df,
        master_df,
        dead_threshold=dead_threshold,
        r2_threshold=r2_threshold,
        show_progress=False,
    )
    gauss_trans = gauss_state_transition_counts(
        master_df,
        gauss_threshold=num_gauss_threshold,
        dead_threshold=dead_threshold,
        r2_threshold=r2_threshold,
    )
    gauss_transition_props = gauss_transition_counts_to_proportions(
        gauss_trans,
        n_neurons=ctx.n_neurons,
    )
    disp_samples = pf_tracked_displacement_samples(
        pf_segment_df,
        master_df,
        gauss_threshold=num_gauss_threshold,
        dead_threshold=dead_threshold,
        r2_threshold=r2_threshold,
        show_progress=False,
    )
    displacement_curves = compute_displacement_curves(
        disp_samples,
        zoom_bin_width=zoom_bin_width,
    )
    revive_events = extract_pf_revive_events(pf_segment_df)
    plot_metrics = pf_metrics_df[pf_metrics_df["length_s"] > 0].copy()
    global_revives = global_pf_metrics[["cell_idx", "pf_idx", "n_life_periods"]].copy()
    global_revives["revives_per_pf"] = np.maximum(
        global_revives["n_life_periods"].astype(int) - 1,
        0,
    )
    plot_metrics = plot_metrics.merge(
        global_revives[["cell_idx", "pf_idx", "revives_per_pf"]],
        on=["cell_idx", "pf_idx"],
        how="left",
    )
    plot_metrics = attach_first_birth_formation_onset(
        plot_metrics,
        global_pf_metrics,
        segment_duration_s=ctx.segment_duration_s,
        pf_segment_df=pf_segment_df,
    )
    plot_metrics["tma_iterations"] = (
        plot_metrics["tma_s"] / ctx.segment_duration_s
    )

    return OptimizerPFData(
        alive_ts=alive_ts,
        pf_segment_df=pf_segment_df,
        pf_transition_counts=pf_transition_counts,
        gauss_transition_props=gauss_transition_props,
        pf_per_neuron=pf_per_neuron,
        plot_metrics=plot_metrics,
        global_pf_metrics=global_pf_metrics[global_pf_metrics["length_s"] > 0].copy(),
        life_period_metrics=life_period_metrics[
            life_period_metrics["length_s"] > 0
        ].copy(),
        revive_events=revive_events,
        displacement_curves=displacement_curves,
    )


def _p_value_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def holm_adjusted_pvalues(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values for a family of tests."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * p[idx])
        prev = max(prev, adj)
        adjusted[idx] = prev
    return adjusted


def pairwise_welch_holm(
    values_by_optimizer: dict[str, np.ndarray],
    *,
    optimizer_order: tuple[str, ...] = tuple(OPTIMIZER_ORDER),
    alpha: float = 0.05,
) -> list[SignificantPair]:
    groups = [opt for opt in optimizer_order if opt in values_by_optimizer]
    pair_meta: list[tuple[str, str, float]] = []
    raw_p_values: list[float] = []

    for g1, g2 in combinations(groups, 2):
        v1 = np.asarray(values_by_optimizer[g1], dtype=float)
        v2 = np.asarray(values_by_optimizer[g2], dtype=float)
        v1 = v1[np.isfinite(v1)]
        v2 = v2[np.isfinite(v2)]
        if v1.size < 2 or v2.size < 2:
            continue
        _, p = stats.ttest_ind(v1, v2, equal_var=False)
        if not np.isfinite(p):
            continue
        pair_meta.append((g1, g2, float(p)))
        raw_p_values.append(float(p))

    if not pair_meta:
        return []

    holm_p = holm_adjusted_pvalues(np.asarray(raw_p_values, dtype=float))
    out: list[SignificantPair] = []
    for (g1, g2, raw_p), adj_p in zip(pair_meta, holm_p, strict=True):
        stars = _p_value_to_stars(adj_p) if adj_p < alpha else "ns"
        out.append(
            SignificantPair(
                opt_a=g1,
                opt_b=g2,
                raw_p=raw_p,
                holm_p=float(adj_p),
                stars=stars,
            )
        )
    return out


def pairwise_ks_holm(
    values_by_optimizer: dict[str, np.ndarray],
    *,
    optimizer_order: tuple[str, ...] = tuple(OPTIMIZER_ORDER),
    alpha: float = 0.05,
) -> list[SignificantPair]:
    groups = [opt for opt in optimizer_order if opt in values_by_optimizer]
    pair_meta: list[tuple[str, str, float]] = []
    raw_p_values: list[float] = []

    for g1, g2 in combinations(groups, 2):
        v1 = np.asarray(values_by_optimizer[g1], dtype=float)
        v2 = np.asarray(values_by_optimizer[g2], dtype=float)
        v1 = v1[np.isfinite(v1)]
        v2 = v2[np.isfinite(v2)]
        if v1.size < 2 or v2.size < 2:
            continue
        _, p = stats.ks_2samp(v1, v2)
        if not np.isfinite(p):
            continue
        pair_meta.append((g1, g2, float(p)))
        raw_p_values.append(float(p))

    if not pair_meta:
        return []

    holm_p = holm_adjusted_pvalues(np.asarray(raw_p_values, dtype=float))
    out: list[SignificantPair] = []
    for (g1, g2, raw_p), adj_p in zip(pair_meta, holm_p, strict=True):
        stars = _p_value_to_stars(adj_p) if adj_p < alpha else "ns"
        out.append(
            SignificantPair(
                opt_a=g1,
                opt_b=g2,
                raw_p=raw_p,
                holm_p=float(adj_p),
                stars=stars,
            )
        )
    return out


def build_meta_summary_table(results: MultiOptResults) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for opt in OPTIMIZER_ORDER:
        ctx = results.contexts[opt]
        training = results.training[opt]
        meta = results.meta[opt]
        gaussian = results.gaussian[opt]
        rows.append(
            {
                "optimizer": opt,
                "optimizer_tag": ctx.optimizer_tag,
                "dead_threshold": gaussian.dead_threshold,
                "max_signal_max": gaussian.percentile99_signal_max,
                "final_loss": float(training.loss_ts["loss"].iloc[-1]),
                "mean_r2_last": float(meta.mean_r2_ts["mean_r2"].iloc[-1]),
                "proportion_alive_last": float(
                    results.pf[opt].alive_ts["proportion_alive"].iloc[-1]
                )
                if opt in results.pf
                else np.nan,
                "mean_pfs_per_neuron": float(
                    results.pf[opt].pf_per_neuron["mean"]
                )
                if opt in results.pf
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_pf_per_neuron_summary_table(results: MultiOptResults) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for opt in OPTIMIZER_ORDER:
        pf = results.pf[opt]
        rows.append(
            {
                "optimizer": opt,
                "mean": pf.pf_per_neuron["mean"],
                "sem": pf.pf_per_neuron["sem"],
                "ci95": pf.pf_per_neuron["ci95"],
            }
        )
    return pd.DataFrame(rows)


def _compute_single_optimizer_results(
    task: tuple[
        str,
        OptimizerRunContext,
        float,
        float,
        float,
        str,
        float,
        float,
        float,
        float,
        str,
    ],
) -> tuple[
    str,
    OptimizerGaussianData,
    OptimizerTrainingData,
    OptimizerMetaData,
    OptimizerPFData,
    np.ndarray,
]:
    (
        optimizer_type,
        ctx,
        dead_threshold_frac,
        r2_threshold,
        num_gauss_threshold,
        pf_tracking_method,
        zoom_bin_width,
        constant_l2_similarity,
        max_factor_pct,
        peak_analysis_technique,
    ) = task
    gaussian_data = load_optimizer_gaussian_data(
        ctx,
        dead_threshold_frac=dead_threshold_frac,
    )
    training = compute_training_timeseries(ctx)
    meta = compute_meta_data(ctx, gaussian_data)
    neuron_peaks = np.nanmax(meta.final_ratemap, axis=(1, 2))
    pf_data = compute_pf_data(
        ctx,
        gaussian_data,
        r2_threshold=r2_threshold,
        num_gauss_threshold=num_gauss_threshold,
        pf_tracking_method=pf_tracking_method,
        zoom_bin_width=zoom_bin_width,
        constant_l2_similarity=constant_l2_similarity,
        max_factor_pct=max_factor_pct,
        peak_analysis_technique=peak_analysis_technique,
    )
    return (
        optimizer_type,
        gaussian_data,
        training,
        meta,
        pf_data,
        neuron_peaks,
    )


def compute_all_optimizer_results(
    contexts: dict[str, OptimizerRunContext],
    *,
    r2_threshold: float,
    num_gauss_threshold: float,
    pf_tracking_method: str,
    dead_threshold_frac: float,
    zoom_bin_width: float,
    constant_l2_similarity: float = 4.0,
    max_factor_pct: float = 1.0,
    peak_analysis_technique: str = PEAK_ANALYSIS_TECHNIQUE_GLOBAL,
) -> MultiOptResults:
    from tqdm.auto import tqdm

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
        )
        for optimizer_type in OPTIMIZER_ORDER
    ]

    results = MultiOptResults(contexts=contexts)
    neuron_peaks: list[np.ndarray] = []

    mp_ctx = multiprocessing.get_context("fork")
    with mp_ctx.Pool(processes=len(tasks)) as pool:
        for (
            optimizer_type,
            gaussian_data,
            training,
            meta,
            pf_data,
            peaks,
        ) in tqdm(
            pool.imap(_compute_single_optimizer_results, tasks),
            total=len(tasks),
            desc="Computing optimizer results",
        ):
            results.gaussian[optimizer_type] = gaussian_data
            results.training[optimizer_type] = training
            results.meta[optimizer_type] = meta
            results.pf[optimizer_type] = pf_data
            neuron_peaks.append(peaks)

    pooled_peaks = np.concatenate(neuron_peaks)
    pooled_peaks = pooled_peaks[np.isfinite(pooled_peaks)]
    if pooled_peaks.size:
        results.shared_ratemap_vmax = float(np.nanpercentile(pooled_peaks, 99.9))
    else:
        results.shared_ratemap_vmax = 1.0
    return results


def _outlier_fit_type_for_optimizer(
    optimizer_type: str,
    fit_type_by_optimizer: dict[str, str],
    *,
    default: str = "negbinom_zi",
) -> str:
    return str(fit_type_by_optimizer.get(optimizer_type, default))


def _fit_random_window_outlier_distribution(
    window_outlier_counts: pd.DataFrame,
    *,
    fit_type: str,
    comparison_fit_types: tuple[str, ...] = (),
) -> dict[str, object]:
    if window_outlier_counts.empty:
        return {"n_samples": 0, "models": {}}
    random_vals = window_outlier_counts.loc[
        window_outlier_counts["event_name"] == "random",
        "n_outliers",
    ].to_numpy(dtype=float)
    fit_types = tuple(
        dict.fromkeys((str(fit_type), *(str(ft) for ft in comparison_fit_types)))
    )
    models = build_outlier_count_fit_comparison(random_vals, fit_types)
    primary = models.get(str(fit_type))
    return {
        "random": primary if primary is not None else {},
        "fit_type": str(fit_type),
        "n_samples": int(random_vals.size),
        "models": models,
    }


def _empty_pf_group_variants() -> dict[str, pd.DataFrame]:
    return {variant: pd.DataFrame() for variant in PF_GLOCAL_VARIANTS}


def _build_lifecycle_outlier_probabilities(
    lifecycle_counts: pd.DataFrame,
    fit_results: dict[str, object],
    window_outlier_counts: pd.DataFrame,
    *,
    fit_type: str,
) -> pd.DataFrame:
    if lifecycle_counts.empty:
        return lifecycle_counts.copy()
    distribution_order = ("random",)
    fit_by_event = {
        dist_name: fit_results.get("models", {})
        for dist_name in distribution_order
    }
    probs = add_pf_lifecycle_outlier_fit_probabilities(
        lifecycle_counts,
        fit_by_event,
        fit_type=fit_type,
        probability_kind="pmf",
        distribution_order=distribution_order,
    )
    probs = add_pf_lifecycle_outlier_fit_probabilities(
        probs,
        fit_by_event,
        fit_type=fit_type,
        probability_kind="cdf",
        distribution_order=distribution_order,
    )
    return add_pf_lifecycle_outlier_empirical_probabilities(
        probs,
        window_outlier_counts,
        distribution_order=distribution_order,
    )


def compute_optimizer_eff_lr_analysis(
    ctx: OptimizerRunContext,
    pf_data: OptimizerPFData,
    gaussian_data: OptimizerGaussianData,
    training: OptimizerTrainingData,
    *,
    optimizer_config: dict,
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
    include_eff_lr: bool = True,
) -> OptimizerEffLrAnalysis:
    eff_lr_outlier_fit_comparison = eff_lr_outlier_fit_comparison or {}
    delta_w_outlier_fit_comparison = delta_w_outlier_fit_comparison or {}
    eff_lr_fit_type = _outlier_fit_type_for_optimizer(
        ctx.optimizer_type,
        eff_lr_outlier_fit_type_by_optimizer,
    )
    delta_w_fit_type = _outlier_fit_type_for_optimizer(
        ctx.optimizer_type,
        delta_w_outlier_fit_type_by_optimizer,
    )
    master_df = gaussian_data.master_df
    pf_segment_df = pf_data.pf_segment_df
    capture_to_global = build_capture_to_global_segment_map(
        master_df,
        training.grad_ts,
    )

    delta_w_bundle = load_delta_w_segment_stats_bundle(
        ctx.paths.signals_dir,
        segment_duration_s=ctx.segment_duration_s,
    )

    pf_delta_w_groups = compute_pf_delta_w_group_metrics(
        pf_segment_df,
        master_df,
        delta_w_bundle,
        capture_to_global,
        window_size=int(window_size),
        n_workers=int(eff_lr_n_workers),
    )

    eligibility_df = summarize_pf_eff_lr_eligibility(
        pf_segment_df,
        master_df,
        min_pre_birth_absent=int(min_pre_birth_absent),
        min_post_birth_active=int(min_post_birth_active),
        min_revive_pre_absent=int(min_revive_pre_absent),
        min_revive_active=int(min_revive_active),
    )

    sample_pf_pairs = select_pfs_random_from_birth_revive_qualified(
        pf_segment_df,
        master_df,
        window_size=int(window_size),
        n_random_windows=int(n_random_windows),
        n=int(n_network_random_pfs),
        random_seed=int(window_outlier_seed),
    )

    birth_eligible_df = eligibility_df.loc[eligibility_df["eligible_birth"]].copy()

    if include_eff_lr:
        eff_lr_bundle = load_effective_lr_segment_stats_bundle(
            ctx.paths.signals_dir,
            segment_duration_s=ctx.segment_duration_s,
            optimizer_config=optimizer_config,
        )
        pf_eff_lr_groups = compute_pf_effective_lr_group_metrics(
            pf_segment_df,
            master_df,
            eff_lr_bundle,
            capture_to_global,
            window_size=int(window_size),
            n_workers=int(eff_lr_n_workers),
        )
        window_outlier_counts, lifecycle_outlier_counts = (
            collect_pf_window_and_lifecycle_outlier_counts(
                pf_segment_df,
                master_df,
                sample_pf_pairs,
                birth_eligible_df,
                signals_dir=ctx.paths.signals_dir,
                capture_to_global=capture_to_global,
                optimizer_config=optimizer_config,
                window_size=int(window_size),
                n_windows=int(n_window_samples),
                method=str(outlier_method),
                layer=str(outlier_layer),
                random_seed=int(window_outlier_seed),
                n_process=int(aggregate_n_process),
                batch_size=int(aggregate_batch_size),
                show_progress=True,
            )
        )
        window_outlier_counts = window_outlier_counts.loc[
            window_outlier_counts["event_name"] == "random"
        ].copy()
        fit_results_random = _fit_random_window_outlier_distribution(
            window_outlier_counts,
            fit_type=eff_lr_fit_type,
            comparison_fit_types=eff_lr_outlier_fit_comparison.get(
                ctx.optimizer_type,
                (),
            ),
        )
        lifecycle_probs = _build_lifecycle_outlier_probabilities(
            lifecycle_outlier_counts,
            fit_results_random,
            window_outlier_counts,
            fit_type=eff_lr_fit_type,
        )
    else:
        pf_eff_lr_groups = _empty_pf_group_variants()
        window_outlier_counts = pd.DataFrame()
        fit_results_random = {"n_samples": 0}
        lifecycle_outlier_counts = pd.DataFrame()
        lifecycle_probs = pd.DataFrame()

    dw_window_outlier_counts, dw_lifecycle_outlier_counts = (
        collect_pf_window_and_lifecycle_delta_w_outlier_counts(
            pf_segment_df,
            master_df,
            sample_pf_pairs,
            birth_eligible_df,
            signals_dir=ctx.paths.signals_dir,
            capture_to_global=capture_to_global,
            window_size=int(window_size),
            n_windows=int(n_window_samples),
            method=str(outlier_method),
            layer=str(outlier_layer),
            random_seed=int(window_outlier_seed),
            n_process=int(aggregate_n_process),
            batch_size=int(aggregate_batch_size),
            show_progress=True,
        )
    )
    dw_window_outlier_counts = dw_window_outlier_counts.loc[
        dw_window_outlier_counts["event_name"] == "random"
    ].copy()
    dw_fit_results_random = _fit_random_window_outlier_distribution(
        dw_window_outlier_counts,
        fit_type=delta_w_fit_type,
        comparison_fit_types=delta_w_outlier_fit_comparison.get(
            ctx.optimizer_type,
            (),
        ),
    )
    dw_lifecycle_probs = _build_lifecycle_outlier_probabilities(
        dw_lifecycle_outlier_counts,
        dw_fit_results_random,
        dw_window_outlier_counts,
        fit_type=delta_w_fit_type,
    )

    return OptimizerEffLrAnalysis(
        capture_to_global=capture_to_global,
        pf_eff_lr_groups=pf_eff_lr_groups,
        pf_delta_w_groups=pf_delta_w_groups,
        eligibility_df=eligibility_df,
        sample_pf_pairs=sample_pf_pairs,
        window_outlier_counts=window_outlier_counts,
        fit_results_random=fit_results_random,
        lifecycle_outlier_counts=lifecycle_outlier_counts,
        lifecycle_probs=lifecycle_probs,
        dw_window_outlier_counts=dw_window_outlier_counts,
        dw_fit_results_random=dw_fit_results_random,
        dw_lifecycle_outlier_counts=dw_lifecycle_outlier_counts,
        dw_lifecycle_probs=dw_lifecycle_probs,
    )


def compute_all_eff_lr_analyses(
    results: MultiOptResults,
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
) -> MultiOptResults:
    from tqdm.auto import tqdm

    for optimizer_type in tqdm(OPTIMIZER_ORDER, desc="Effective-LR analysis"):
        ctx = results.contexts[optimizer_type]
        optimizer_config = build_optimizer_config(optimizer_type)
        include_eff_lr = optimizer_supports_eff_lr(optimizer_type)
        try:
            results.eff_lr[optimizer_type] = compute_optimizer_eff_lr_analysis(
                ctx,
                results.pf[optimizer_type],
                results.gaussian[optimizer_type],
                results.training[optimizer_type],
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
                    f"Skipping effective-LR analysis for {optimizer_type}: {exc}"
                )
            else:
                raise
    return results
