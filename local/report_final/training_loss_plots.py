"""Training-loss figures for the supplementary report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from tqdm.auto import tqdm

from experiments.common.run import create_experiment, load_experiment_config
from experiments.common.signals_io import TWO_ROOMS_NAME, load_training_loss_timeline
from optimizers.defaults import build_optimizer_config
from single_room_multi_opt_compute import (
    OPTIMIZER_ORDER,
    OptimizerTrainingData,
    apply_results_parent_override,
)
from single_room_pf_analysis import load_segment_duration_s
from two_rooms_multi_opt_compute import TwoRoomsMultiOptResults
from two_rooms_multi_opt_plots import plot_training_loss_by_optimizer_two_rooms

_EMPTY_TS = pd.DataFrame()


def load_two_rooms_training_loss_results(
    *,
    project_root: Path,
    config_path: Path,
    results_parent_override: str | None = None,
    optimizer_order: tuple[str, ...] = tuple(OPTIMIZER_ORDER),
    show_progress: bool = True,
) -> tuple[TwoRoomsMultiOptResults, float]:
    """Load only the loss timelines needed for the training-loss figure."""
    run_config = load_experiment_config(config_path)
    results = TwoRoomsMultiOptResults(contexts={})
    segment_duration_s: float | None = None

    opt_iter = tqdm(
        optimizer_order,
        desc="Loading two-rooms training loss",
        disable=not show_progress,
    )
    for optimizer_type in opt_iter:
        opt_iter.set_postfix(optimizer=optimizer_type)
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
            experiment_name=TWO_ROOMS_NAME,
            project_root=project_root,
            results_parent_override=results_parent_override,
        )
        if not paths.signals_dir.is_dir():
            raise FileNotFoundError(
                f"Missing signals directory for {optimizer_type}: {paths.signals_dir}"
            )

        opt_segment_duration_s = load_segment_duration_s(paths.results_dir / "config.json")
        if segment_duration_s is None:
            segment_duration_s = opt_segment_duration_s
        elif segment_duration_s != opt_segment_duration_s:
            raise ValueError(
                f"segment_duration_s mismatch for {optimizer_type}: "
                f"{opt_segment_duration_s} vs {segment_duration_s}"
            )

        loss_ts = load_training_loss_timeline(
            paths.signals_dir,
            experiment_name=TWO_ROOMS_NAME,
            last_rep_only=False,
        )
        results.training[optimizer_type] = OptimizerTrainingData(
            loss_ts=loss_ts,
            grad_ts=_EMPTY_TS,
            eff_lr_ts=_EMPTY_TS,
            delta_w_ts=_EMPTY_TS,
        )

    if segment_duration_s is None:
        raise ValueError("No optimizers loaded.")
    return results, segment_duration_s


def two_rooms_training_loss_table(results: TwoRoomsMultiOptResults) -> pd.DataFrame:
    """Long-form loss timeline for every optimizer."""
    rows: list[pd.DataFrame] = []
    for opt in OPTIMIZER_ORDER:
        if opt not in results.training:
            continue
        part = results.training[opt].loss_ts.copy()
        part.insert(0, "optimizer", opt)
        rows.append(part)
    if not rows:
        raise ValueError("No training loss timelines found in results.")
    return pd.concat(rows, ignore_index=True)


def plot_two_rooms_training_loss(
    results: TwoRoomsMultiOptResults,
    *,
    segment_duration_s: float,
    trace_alpha: float = 0.75,
    subsample_win_size: int = 5,
    title: str | None = None,
) -> Figure:
    """Build the raw figure from `two_rooms_multi_opt_plots`.

    Typography is intentionally left at notebook defaults here. Call
    `finalize(fig, ...)` afterward (default ``restyle=True``), as in
    `REPORT_final_FINAL.ipynb`, to apply `report_font()` sizes.
    """
    import matplotlib.pyplot as plt
    with plt.rc_context({"mathtext.fontset": "dejavusans"}):
        fig, _ = plot_training_loss_by_optimizer_two_rooms(
            results,
            segment_duration_s=segment_duration_s,
            trace_alpha=trace_alpha,
            subsample_win_size=subsample_win_size,
            show_legend=False,
            show_room_rep_boundaries=True,
            title=title,
        )
    return fig
