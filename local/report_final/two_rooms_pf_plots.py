"""Two-room place-field figures for supplementary report notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

from report_final.notebook_cache import (
    TWO_ROOMS_CACHE,
    load_pkl_gz,
    report_final_cache_dir,
    save_pkl_gz,
)
from single_room_multi_opt_compute import OPTIMIZER_ORDER
from two_rooms_multi_opt_compute import (
    TwoRoomsMultiOptResults,
    build_two_rooms_optimizer_contexts,
    compute_all_two_rooms_optimizer_results,
    load_mean_wsm_by_room,
)
from two_rooms_multi_opt_plots import plot_dual_room_metric_by_optimizer
from two_rooms_pf_analysis import ROOM_IDXS


def load_two_rooms_optimizer_results(
    *,
    project_root: Path,
    config_path: Path,
    results_parent_override: str | None = None,
    dead_threshold_frac: float = 0.4,
    r2_threshold: float = 0.8,
    num_gauss_threshold: float = 1.5,
    pf_tracking_method: str = "revive_mean_displacement",
    constant_l2_similarity: float = 10.0,
    max_factor_pct: float = 0.95,
    peak_analysis_technique: str = "local",
    zoom_bin_width: float = 0.25,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    force_recompute: bool = False,
    **compute_kwargs: Any,
) -> TwoRoomsMultiOptResults:
    """Load full two-room optimizer results (same pipeline as the main report).

    Reads/writes ``local/_cache/two_rooms.pkl.gz`` shared with
    ``REPORT_final_FINAL.ipynb`` when ``use_cache`` is True.
    """
    if cache_dir is None:
        cache_dir = report_final_cache_dir(project_root)
    cache_path = cache_dir / TWO_ROOMS_CACHE

    if use_cache and not force_recompute and cache_path.is_file():
        cached = load_pkl_gz(cache_path)
        if cached is not None:
            print(f"Loaded two-room results from cache: {cache_path}")
            return cached

    mean_wsm_by_room = load_mean_wsm_by_room(config_path=config_path)
    contexts = build_two_rooms_optimizer_contexts(
        project_root=project_root,
        config_path=config_path,
        results_parent_override=results_parent_override,
    )
    results = compute_all_two_rooms_optimizer_results(
        contexts,
        dead_threshold_frac=dead_threshold_frac,
        r2_threshold=r2_threshold,
        num_gauss_threshold=num_gauss_threshold,
        pf_tracking_method=pf_tracking_method,
        zoom_bin_width=zoom_bin_width,
        constant_l2_similarity=constant_l2_similarity,
        max_factor_pct=max_factor_pct,
        peak_analysis_technique=peak_analysis_technique,
        mean_wsm_by_room=mean_wsm_by_room,
        **compute_kwargs,
    )
    if use_cache:
        save_pkl_gz(cache_path, results)
        print(f"Saved two-room results to cache: {cache_path}")
    return results


def two_rooms_alive_proportion_table(
    results: TwoRoomsMultiOptResults,
) -> pd.DataFrame:
    """Long-form alive proportion timeline for every optimizer and eval room."""
    rows: list[pd.DataFrame] = []
    for opt in OPTIMIZER_ORDER:
        if opt not in results.pf:
            continue
        for room_idx in ROOM_IDXS:
            part = results.pf[opt].by_room[room_idx].alive_ts.copy()
            part.insert(0, "optimizer", opt)
            part.insert(1, "eval_room_id", room_idx)
            rows.append(part)
    if not rows:
        raise ValueError("No alive proportion timelines found in results.")
    return pd.concat(rows, ignore_index=True)


def plot_two_rooms_alive_proportion(
    results: TwoRoomsMultiOptResults,
    *,
    segment_duration_s: float | None = None,
    trace_alpha: float = 0.75,
    y_lims: tuple[float, float] | None = (0.0, 0.5),
    show_traj_boundaries: bool = False,
    show_legend: bool = True,
    title: str = "",
) -> Figure:
    """Alive proportion by optimizer and eval room (muted = train_other).

    Thin wrapper around ``plot_dual_room_metric_by_optimizer``. Call
    ``finalize(fig, ...)`` afterward to apply report typography.
    """
    fig, _ = plot_dual_room_metric_by_optimizer(
        results,
        y_col="proportion_alive",
        ylabel="Proportion of cells",
        title=title,
        segment_duration_s=segment_duration_s,
        ylim=y_lims,
        trace_alpha=trace_alpha,
        show_traj_boundaries=show_traj_boundaries,
        show_legend=show_legend,
    )
    return fig
