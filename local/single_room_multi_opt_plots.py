"""Plotting helpers for multi-optimizer single_room comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from scipy.stats import gaussian_kde
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from single_room_multi_opt_compute import (
    EFF_LR_OPTIMIZER_ORDER,
    N_COLS,
    N_ROWS,
    OPTIMIZER_COLORS,
    OPTIMIZER_LABELS,
    OPTIMIZER_ORDER,
    TIME_UNIT_SECONDS,
    MultiOptResults,
    SignificantPair,
    pf_metric_bar_panel_specs,
    pf_metric_boxplot_panels,
    pairwise_ks_holm,
    pairwise_welch_holm,
    tma_metric_spec,
)
from experiments.common.signals_io import (
    TRACKED_LAYER_NAMES,
    _effective_lr_by_node_from_opt_signals,
    _iter_training_segment_files,
    _load_opt_signals_dict,
    _node_signal_to_unit_scalar,
)
from optimizers.defaults import build_optimizer_config
from single_room_pf_analysis import (
    OUTLIER_COUNT_FIT_MODEL_LABELS,
    TIME_AXIS_LABEL,
    format_outlier_count_fit_annotation,
    fit_outlier_count_distribution,
    fit_outlier_count_mean_residual_ci95,
    index_to_time_s,
    outlier_count_empirical_and_fit_curve,
    outlier_count_fit_qq_points,
    traj_boundary_vlines,
)

TRANSITION_COLORS = {
    "inactive_to_active": "#a3d96e",
    "active_to_inactive": "#e65671",
    "revives": "#97b4ed",
}
GAUSS_TRANSITION_COLORS = {
    "prop_1_to_2": "#567ee6",
    "prop_2_to_1": "#e07833",
    "prop_1_to_1": "#a3d96e",
    "prop_2_to_2": "#e65671",
}
GRADIENT_COLORS = {
    "g_bptt_mse": "#567ee6",
    "g_local_mse": "#e07833",
    "g_fr": "#a3d96e",
}
LAYER_TIMESERIES_COLORS = {
    "recurrent": "tab:blue",
    "input": "tab:orange",
    "readout": "tab:green",
}
EFF_LR_GROUP_ORDER = (
    "pre_vicinity",
    "birth",
    "peak",
    "other_revives",
    "absent",
)
EFF_LR_GROUP_LABELS = {
    "pre_vicinity": "pre-vicinity",
    "birth": "birth",
    "peak": "peak amp",
    "other_revives": "other revives",
    "absent": "PF absent",
}
EFF_LR_GROUP_SHORT_TAGS = {
    "pre_vicinity": "pv",
    "birth": "b",
    "peak": "pk",
    "other_revives": "or",
    "absent": "ab",
}
EFF_LR_GROUP_HATCHES = {
    "pre_vicinity": "",
    "birth": "///",
    "peak": "...",
    "other_revives": "xx",
    "absent": "++",
}
LIFECYCLE_PROB_SPECS = (
    ("birth", "pre", r"b_\mathbf{pre}"),
    ("birth", "post", r"b_\mathbf{post}"),
    ("revive", "pre", r"r_\mathbf{pre}"),
    ("revive", "post", r"r_\mathbf{post}"),
)
LIFECYCLE_PROB_LEGEND_LABELS = (
    "Birth, pre",
    "Birth, post",
    "Revive, pre",
    "Revive, post",
)


def save_plot(fig: Figure, plot_dir: Path | str, name: str) -> Path:
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"{name}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    return out_path


def _add_traj_boundaries(
    ax,
    timeline: pd.DataFrame,
    *,
    x_col: str,
    segment_duration_s: float,
) -> None:
    starts = index_to_time_s(
        timeline.loc[timeline["is_traj_start"], x_col].to_numpy(dtype=float),
        segment_duration_s,
    )
    if starts.size > 1:
        for xv in starts[1:]:
            ax.axvline(xv, color="0.5", linestyle="--", linewidth=1, alpha=0.8)


def _bin_transition_series(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bin_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if bin_window <= 1:
        return x, y, None

    n_bins = int(np.ceil(len(x) / bin_window))
    x_out, mean_out, sem_out = [], [], []
    for b in range(n_bins):
        sl = slice(b * bin_window, min((b + 1) * bin_window, len(x)))
        yb = y[sl]
        if yb.size == 0:
            continue
        x_out.append(float(x[sl].mean()))
        mean_out.append(float(yb.mean()))
        sem_out.append(
            float(yb.std(ddof=1) / np.sqrt(yb.size)) if yb.size > 1 else 0.0
        )
    return np.asarray(x_out), np.asarray(mean_out), np.asarray(sem_out)


def _mean_sem_ci95(values: np.ndarray) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
    return mean, sem, 1.96 * sem


def _format_significance_pair(pair: SignificantPair) -> str:
    return (
        f"{OPTIMIZER_LABELS[pair.opt_a]} vs {OPTIMIZER_LABELS[pair.opt_b]}: "
        f"raw p={pair.raw_p:.2e}, Holm p={pair.holm_p:.2e} ({pair.stars})"
    )


SignificanceTest = Literal["welch", "ks"]

SIGNIFICANCE_TEST_LABELS: dict[SignificanceTest, str] = {
    "welch": "Welch t-test",
    "ks": "Kolmogorov–Smirnov",
}


def _pairwise_significance(
    values_by_optimizer: dict[str, np.ndarray],
    *,
    test: SignificanceTest,
    alpha: float = 0.05,
) -> list[SignificantPair]:
    if test == "ks":
        return pairwise_ks_holm(values_by_optimizer, alpha=alpha)
    return pairwise_welch_holm(values_by_optimizer, alpha=alpha)


def _print_significance_results(
    *,
    panel_label: str,
    pairs: list[SignificantPair],
    test: SignificanceTest = "welch",
) -> None:
    sig = [p for p in pairs if p.stars != "ns"]
    test_label = SIGNIFICANCE_TEST_LABELS[test]
    print(
        f"[{panel_label}] {len(sig)} significant pair(s) "
        f"({test_label}, Holm corrected):"
    )
    if not sig:
        print("  (none)")
        return
    for pair in sig:
        print(f"  {_format_significance_pair(pair)}")


def _metric_values_by_optimizer(
    results: MultiOptResults,
    metric_col: str,
) -> dict[str, np.ndarray]:
    values_by_optimizer: dict[str, np.ndarray] = {}
    for opt in OPTIMIZER_ORDER:
        pf_data = results.pf[opt]
        if metric_col == "revives_per_pf":
            src = pf_data.global_pf_metrics
            values = np.maximum(src["n_life_periods"].astype(int) - 1, 0).astype(float)
        else:
            values = (
                pf_data.plot_metrics[metric_col]
                .dropna()
                .to_numpy(dtype=float)
            )
        values_by_optimizer[opt] = values
    return values_by_optimizer


def _optimizer_positions() -> dict[str, float]:
    return {opt: float(i) for i, opt in enumerate(OPTIMIZER_ORDER)}


def _optimizer_positions_at(
    x: np.ndarray,
    optimizer_order: tuple[str, ...] | list[str] | None = None,
) -> dict[str, float]:
    order = optimizer_order or OPTIMIZER_ORDER
    return {opt: float(x[i]) for i, opt in enumerate(order)}


def _pair_key(opt_a: str, opt_b: str) -> tuple[str, str]:
    i_a = OPTIMIZER_ORDER.index(opt_a)
    i_b = OPTIMIZER_ORDER.index(opt_b)
    return (opt_a, opt_b) if i_a < i_b else (opt_b, opt_a)


def _format_significance_stars(stars: str) -> str:
    """LaTeX superscript stars with ordinary-atom spacing (TeX SE #68947)."""
    if stars == "ns":
        return "ns"
    n = stars.count("*")
    if n <= 0:
        return stars
    return "".join("{*}" for _ in range(n))


def _multi_metric_tag_label(stars: str, tag: str) -> str:
    """Format significance as LaTeX, e.g. $*{*}_{\\mathrm{time}}$."""
    return rf"${_format_significance_stars(stars)}_{{\mathrm{{{tag}}}}}$"


SIGNIFICANCE_INVERT_FRACTION = 0.7
ALL_PAIRWISE_SIGNIFICANT_LABEL = "all p-s"


def _spanning_optimizer_pair(positions: dict[str, float]) -> tuple[str, str]:
    opts = sorted(positions, key=lambda opt: positions[opt])
    return opts[0], opts[-1]


def _all_pairs_significant(pairs: list[SignificantPair]) -> bool:
    return bool(pairs) and all(pair.stars != "ns" for pair in pairs)


def _multi_metric_pair_stars(
    values_by_metric: dict[str, dict[str, np.ndarray]],
    metric_tags: dict[str, str],
    *,
    alpha: float = 0.05,
) -> dict[tuple[str, str], dict[str, str]]:
    pair_stars: dict[tuple[str, str], dict[str, str]] = {}
    for metric_col, tag in metric_tags.items():
        for pair in pairwise_welch_holm(values_by_metric[metric_col], alpha=alpha):
            key = _pair_key(pair.opt_a, pair.opt_b)
            pair_stars.setdefault(key, {})[tag] = pair.stars
    return pair_stars


def _pair_fully_significant(
    stars_by_tag: dict[str, str],
    required_tags: tuple[str, ...],
) -> bool:
    return all(stars_by_tag.get(tag, "ns") != "ns" for tag in required_tags)


def _mostly_fully_significant_multi_metric(
    pair_stars: dict[tuple[str, str], dict[str, str]],
    required_tags: tuple[str, ...],
    *,
    fraction: float = SIGNIFICANCE_INVERT_FRACTION,
) -> bool:
    if not pair_stars:
        return False
    n_fully = sum(
        1
        for stars_by_tag in pair_stars.values()
        if _pair_fully_significant(stars_by_tag, required_tags)
    )
    return n_fully / len(pair_stars) >= fraction


def _multi_metric_pair_labels(
    values_by_metric: dict[str, dict[str, np.ndarray]],
    metric_tags: dict[str, str],
    *,
    show_insignificance: bool = False,
    alpha: float = 0.05,
    invert_fraction: float = SIGNIFICANCE_INVERT_FRACTION,
    positions: dict[str, float] | None = None,
) -> dict[tuple[str, str], str]:
    required_tags = tuple(metric_tags.values())
    pair_stars = _multi_metric_pair_stars(
        values_by_metric,
        metric_tags,
        alpha=alpha,
    )
    invert_brackets = show_insignificance and _mostly_fully_significant_multi_metric(
        pair_stars,
        required_tags,
        fraction=invert_fraction,
    )

    if invert_brackets:
        labels: dict[tuple[str, str], str] = {}
        for key, stars_by_tag in pair_stars.items():
            if _pair_fully_significant(stars_by_tag, required_tags):
                continue
            parts = [
                _multi_metric_tag_label(stars_by_tag.get(tag, "ns"), tag)
                for tag in required_tags
            ]
            labels[key] = ", ".join(parts)
        if (
            not labels
            and positions is not None
            and pair_stars
            and all(
                _pair_fully_significant(stars_by_tag, required_tags)
                for stars_by_tag in pair_stars.values()
            )
        ):
            labels = {
                _spanning_optimizer_pair(positions): ALL_PAIRWISE_SIGNIFICANT_LABEL
            }
        return labels

    labels_by_pair: dict[tuple[str, str], list[str]] = {}
    for key, stars_by_tag in pair_stars.items():
        for tag in required_tags:
            stars = stars_by_tag.get(tag, "ns")
            if stars == "ns":
                continue
            labels_by_pair.setdefault(key, []).append(
                _multi_metric_tag_label(stars, tag)
            )
    return {key: ", ".join(parts) for key, parts in labels_by_pair.items()}


def _print_multi_metric_significance(
    *,
    panel_label: str,
    values_by_metric: dict[str, dict[str, np.ndarray]],
    metric_tags: dict[str, str],
    alpha: float = 0.05,
) -> None:
    for metric_col, tag in metric_tags.items():
        pairs = pairwise_welch_holm(values_by_metric[metric_col], alpha=alpha)
        print(
            f"[{panel_label} / {tag}] "
            f"{sum(1 for p in pairs if p.stars != 'ns')} significant pair(s):"
        )
        if not pairs:
            print("  (none)")
            continue
        for pair in pairs:
            if pair.stars == "ns":
                continue
            print(f"  {_format_significance_pair(pair)}")


def _add_labeled_pairwise_brackets(
    ax,
    *,
    positions: dict[str, float],
    bar_top: float,
    pair_labels: dict[tuple[str, str], str],
    bracket_step_mult: float = 1.5,
) -> None:
    if not pair_labels:
        return

    y_base = float(bar_top) if np.isfinite(bar_top) else 0.0
    y_step = max(y_base * 0.08, 0.05)
    bracket_height = y_step * 0.4
    required_top = y_base + y_step + len(pair_labels) * y_step * bracket_step_mult
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, max(y1, required_top * 1.05))

    y_level = y_base + y_step
    for (opt_a, opt_b), label in pair_labels.items():
        x1, x2 = positions[opt_a], positions[opt_b]
        ax.plot(
            [x1, x1, x2, x2],
            [y_level, y_level + bracket_height, y_level + bracket_height, y_level],
            color="k",
            linewidth=1,
        )
        ax.text(
            (x1 + x2) / 2,
            y_level + bracket_height + y_step * 0.05,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )
        y_level += y_step * bracket_step_mult


def _mostly_significant_pairs(
    pairs: list[SignificantPair],
    *,
    fraction: float = SIGNIFICANCE_INVERT_FRACTION,
) -> bool:
    if not pairs:
        return False
    n_sig = sum(1 for p in pairs if p.stars != "ns")
    return n_sig / len(pairs) >= fraction


def _pairwise_bracket_labels(
    values_by_optimizer: dict[str, np.ndarray],
    *,
    tag: str | None = None,
    show_insignificance: bool = False,
    alpha: float = 0.05,
    test: SignificanceTest = "welch",
    positions: dict[str, float] | None = None,
) -> dict[tuple[str, str], str]:
    all_pairs = _pairwise_significance(values_by_optimizer, test=test, alpha=alpha)
    invert_brackets = show_insignificance and _mostly_significant_pairs(all_pairs)
    if invert_brackets:
        labels = {
            _pair_key(p.opt_a, p.opt_b): "$ns$"
            for p in all_pairs
            if p.stars == "ns"
        }
        if (
            not labels
            and positions is not None
            and _all_pairs_significant(all_pairs)
        ):
            labels = {
                _spanning_optimizer_pair(positions): ALL_PAIRWISE_SIGNIFICANT_LABEL
            }
        return labels
    return {
        _pair_key(p.opt_a, p.opt_b): (
            _multi_metric_tag_label(p.stars, tag)
            if tag is not None
            else f"${_format_significance_stars(p.stars)}$"
        )
        for p in all_pairs
        if p.stars != "ns"
    }


def _handle_optimizer_significance(
    ax,
    values_by_optimizer: dict[str, np.ndarray],
    *,
    positions: dict[str, float],
    bar_tops: np.ndarray,
    panel_label: str,
    show_significance: bool = True,
    show_insignificance: bool = False,
    alpha: float = 0.05,
    bracket_step_mult: float = 1.5,
    test: SignificanceTest = "welch",
) -> None:
    all_pairs = _pairwise_significance(values_by_optimizer, test=test, alpha=alpha)
    if not show_significance:
        _print_significance_results(
            panel_label=panel_label,
            pairs=all_pairs,
            test=test,
        )
        return

    if show_insignificance and _mostly_significant_pairs(all_pairs):
        _print_significance_results(
            panel_label=panel_label,
            pairs=all_pairs,
            test=test,
        )
    pair_labels = _pairwise_bracket_labels(
        values_by_optimizer,
        show_insignificance=show_insignificance,
        alpha=alpha,
        test=test,
        positions=positions,
    )

    bar_top = float(np.nanmax(bar_tops)) if bar_tops.size else 0.0
    _add_labeled_pairwise_brackets(
        ax,
        positions=positions,
        bar_top=bar_top,
        pair_labels=pair_labels,
        bracket_step_mult=bracket_step_mult,
    )


def _windowed_mean_sem(
    x: np.ndarray,
    y: np.ndarray,
    *,
    win_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_bins = int(np.ceil(len(y) / win_size))
    x_out, mean_out, sem_out = [], [], []
    for b in range(n_bins):
        sl = slice(b * win_size, min((b + 1) * win_size, len(y)))
        yb = y[sl]
        if yb.size == 0:
            continue
        x_out.append(float(x[sl].mean()))
        mean_out.append(float(yb.mean()))
        sem_out.append(
            float(yb.std(ddof=1) / np.sqrt(yb.size)) if yb.size > 1 else 0.0
        )
    return np.asarray(x_out), np.asarray(mean_out), np.asarray(sem_out)


def _optimizer_legend_handles(
    *,
    linestyle: str = "-",
    linewidth: float = 1.2,
    marker: str | None = None,
    markersize: float = 2.0,
) -> list[Line2D]:
    """Legend proxies at full opacity while plotted traces may use trace_alpha."""
    handles: list[Line2D] = []
    for opt in OPTIMIZER_ORDER:
        kwargs: dict[str, object] = {
            "color": OPTIMIZER_COLORS[opt],
            "label": OPTIMIZER_LABELS[opt],
            "alpha": 1.0,
            "linestyle": linestyle,
            "linewidth": linewidth,
        }
        if marker is not None:
            kwargs["marker"] = marker
            kwargs["markersize"] = markersize
        handles.append(Line2D([0], [0], **kwargs))
    return handles


def _optimizer_patch_legend_handles(*, alpha: float = 0.75) -> list[Patch]:
    return [
        Patch(
            facecolor=OPTIMIZER_COLORS[opt],
            edgecolor="k",
            alpha=alpha,
            label=OPTIMIZER_LABELS[opt],
        )
        for opt in OPTIMIZER_ORDER
    ]


def _add_shared_optimizer_legend_between_axes(
    fig: Figure,
    axes: list,
    *,
    ncol: int = 5,
    fontsize: float = 8,
) -> None:
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    if not axes:
        return
    if len(axes) == 1:
        legend_y = axes[0].get_position().y0
    else:
        top_bbox = axes[0].get_position()
        bottom_bbox = axes[-1].get_position()
        legend_y = (top_bbox.y0 + bottom_bbox.y1) / 2.0
    fig.legend(
        handles=_optimizer_patch_legend_handles(),
        loc="center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=ncol,
        frameon=True,
        framealpha=0.92,
        fontsize=fontsize,
    )


def _shared_pf_colormap(pf_segment_dfs: dict[str, pd.DataFrame]) -> dict[int, tuple]:
    pf_vals = sorted(
        {
            int(pf)
            for opt in OPTIMIZER_ORDER
            if opt in pf_segment_dfs
            for pf in pf_segment_dfs[opt]["pf_idx"].unique()
        }
    )
    cmap = plt.get_cmap("twilight", max(len(pf_vals), 1))
    return {pf: cmap(i / max(len(pf_vals), 1)) for i, pf in enumerate(pf_vals)}


def plot_training_loss_by_optimizer(
    results: MultiOptResults,
    *,
    segment_duration_s: float | None = None,
    trace_alpha: float = 0.4,
    subsample_win_size: int = 5,
    title: str = "",
) -> tuple[Figure, plt.Axes]:
    if segment_duration_s is None:
        segment_duration_s = next(iter(results.contexts.values())).segment_duration_s

    fig, ax = plt.subplots(figsize=(14, 4))
    for opt in OPTIMIZER_ORDER:
        loss_ts = results.training[opt].loss_ts
        x = index_to_time_s(
            loss_ts["global_segment_idx"].to_numpy(dtype=float),
            segment_duration_s,
        )
        y = loss_ts["loss"].to_numpy(dtype=float)
        color = OPTIMIZER_COLORS[opt]
        if subsample_win_size > 1:
            x_w, y_w, sem_w = _windowed_mean_sem(x, y, win_size=subsample_win_size)
            ci = 1.96 * sem_w
            ax.fill_between(
                x_w,
                y_w - ci,
                y_w + ci,
                color=color,
                alpha=trace_alpha * 0.35,
                linewidth=0,
            )
            ax.plot(
                x_w,
                y_w,
                "-",
                color=color,
                linewidth=1.2,
                alpha=trace_alpha,
            )
        else:
            ax.plot(
                x,
                y,
                "-",
                color=color,
                linewidth=1.0,
                alpha=trace_alpha,
            )
    _add_traj_boundaries(
        ax,
        results.training[OPTIMIZER_ORDER[0]].loss_ts,
        x_col="global_segment_idx",
        segment_duration_s=segment_duration_s,
    )
    ax.set_xlim(0, None)
    ax.set_xlabel(TIME_AXIS_LABEL)
    ax.set_ylabel("Loss")
    win_label = (
        f"; {subsample_win_size * segment_duration_s:g}s window mean ± 1.96 SEM"
        if subsample_win_size > 1
        else ""
    )
    title = title if title is not None else f"Training loss by optimizer ({win_label.lstrip('; ')}{'; ' if win_label else ''}dashed = new trajectory)"
    ax.set_title(
        title
    )
    ax.legend(
        handles=_optimizer_legend_handles(linestyle="-", linewidth=1.2),
        frameon=False,
        loc="upper center",
        ncol=5,
    )
    fig.tight_layout()
    return fig, ax


def plot_mean_r2_by_optimizer(
    results: MultiOptResults,
    *,
    segment_duration_s: float | None = None,
    trace_alpha: float = 0.4,
) -> tuple[Figure, plt.Axes]:
    if segment_duration_s is None:
        segment_duration_s = next(iter(results.contexts.values())).segment_duration_s

    fig, ax = plt.subplots(figsize=(14, 4))
    for opt in OPTIMIZER_ORDER:
        mean_r2_ts = results.meta[opt].mean_r2_ts
        ax.plot(
            index_to_time_s(mean_r2_ts["global_capture_idx"], segment_duration_s),
            mean_r2_ts["mean_r2"],
            "o-",
            color=OPTIMIZER_COLORS[opt],
            markersize=2,
            linewidth=1,
            alpha=trace_alpha,
        )
    _add_traj_boundaries(
        ax,
        results.meta[OPTIMIZER_ORDER[0]].mean_r2_ts,
        x_col="global_capture_idx",
        segment_duration_s=segment_duration_s,
    )
    ax.set_xlabel(TIME_AXIS_LABEL)
    ax.set_ylabel("Mean R² (network-wide)")
    ax.set_title("Mean Gaussian-fit R² per segment by optimizer")
    ax.set_ylim(0, 1)
    ax.legend(
        handles=_optimizer_legend_handles(
            linestyle="-",
            linewidth=1.0,
            marker="o",
            markersize=2.0,
        ),
        frameon=False,
        loc="lower right",
        ncol=2,
    )
    fig.tight_layout()
    return fig, ax


def _plot_layer_mean_timeseries(
    ax,
    timeline: pd.DataFrame,
    *,
    prefix: str,
    segment_duration_s: float,
    layers: tuple[str, ...] = TRACKED_LAYER_NAMES,
    layer_colors: dict[str, str] | None = None,
    fallback_label: str = "hidden mean",
) -> None:
    layer_colors = layer_colors or LAYER_TIMESERIES_COLORS
    x = index_to_time_s(timeline["global_segment_idx"], segment_duration_s)
    plotted = False
    for layer in layers:
        col = f"{prefix}_{layer}"
        if col not in timeline.columns:
            continue
        y = timeline[col].to_numpy(dtype=float)
        if not np.isfinite(y).any():
            continue
        ax.plot(
            x,
            y,
            "-",
            color=layer_colors.get(layer),
            linewidth=1.0,
            alpha=0.85,
            label=layer,
        )
        plotted = True
    if not plotted:
        mean_col = f"{prefix.rstrip('_')}_mean" if prefix.endswith("_") else f"{prefix}_mean"
        if mean_col not in timeline.columns:
            mean_col = prefix
        if mean_col in timeline.columns:
            y = timeline[mean_col].to_numpy(dtype=float)
            if np.isfinite(y).any():
                ax.plot(
                    x,
                    y,
                    "-",
                    color="tab:purple",
                    linewidth=1.0,
                    alpha=0.85,
                    label=fallback_label,
                )
                plotted = True
    if not plotted:
        ax.text(
            0.5,
            0.5,
            "No eff. lr",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="0.4",
        )
    ax.set_xlim(0, None)
    _add_traj_boundaries(
        ax,
        timeline,
        x_col="global_segment_idx",
        segment_duration_s=segment_duration_s,
    )


def plot_gradients_delta_w_eff_lr_grid(
    results: MultiOptResults,
    *,
    segment_duration_s: float | None = None,
) -> tuple[Figure, np.ndarray]:
    if segment_duration_s is None:
        segment_duration_s = next(iter(results.contexts.values())).segment_duration_s

    n_cols = len(OPTIMIZER_ORDER)
    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(3, n_cols, hspace=0.38, wspace=0.25)
    axes = np.empty((3, n_cols), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[0, 0])
    axes[1, 0] = fig.add_subplot(gs[1, 0])
    axes[2, 0] = fig.add_subplot(gs[2, 0])
    for col in range(1, n_cols):
        axes[0, col] = fig.add_subplot(gs[0, col], sharey=axes[0, 0])
        axes[1, col] = fig.add_subplot(gs[1, col])
        axes[2, col] = fig.add_subplot(gs[2, col])

    grad_series = [
        ("g_bptt_mse", GRADIENT_COLORS["g_bptt_mse"], "g_bptt_mse"),
        ("g_local_mse", GRADIENT_COLORS["g_local_mse"], "g_local_mse"),
        ("g_fr", GRADIENT_COLORS["g_fr"], "g_fr"),
    ]

    for col, opt in enumerate(OPTIMIZER_ORDER):
        training = results.training[opt]
        grad_ts = training.grad_ts
        ax_grad = axes[0, col]
        x = index_to_time_s(grad_ts["global_segment_idx"], segment_duration_s)
        for col_name, color, label in grad_series:
            ax_grad.plot(
                x,
                grad_ts[col_name],
                "-",
                color=color,
                linewidth=1.0,
                alpha=0.85,
                label=label,
            )
        _add_traj_boundaries(
            ax_grad,
            grad_ts,
            x_col="global_segment_idx",
            segment_duration_s=segment_duration_s,
        )
        ax_grad.set_title(OPTIMIZER_LABELS[opt], fontsize=10)
        if col == n_cols - 1:
            ax_grad.legend(frameon=False, loc="upper right", fontsize=8)

        ax_dw = axes[1, col]
        _plot_layer_mean_timeseries(
            ax_dw,
            training.delta_w_ts,
            prefix="delta_w_mean",
            segment_duration_s=segment_duration_s,
            fallback_label="hidden mean ΔW",
        )
        if col == n_cols - 1:
            ax_dw.legend(frameon=False, loc="lower right", fontsize=8)

        ax_lr = axes[2, col]
        _plot_layer_mean_timeseries(
            ax_lr,
            training.eff_lr_ts,
            prefix="effective_lr_mean",
            segment_duration_s=segment_duration_s,
            fallback_label="hidden mean LR",
        )
        ax_lr.set_xlabel(TIME_AXIS_LABEL)
        if col == n_cols - 1:
            ax_lr.legend(frameon=False, loc="upper right", fontsize=8)

    axes[0, 0].set_ylabel("Mean gradient signal")
    axes[1, 0].set_ylabel("Mean signed ΔW")
    axes[2, 0].set_ylabel("Mean effective LR")
    fig.suptitle(
        "Gradients, ΔW, and effective LR by layer (dashed = new trajectory)",
        y=1.01,
    )
    fig.tight_layout()
    return fig, axes


def plot_gradients_and_eff_lr_grid(
    results: MultiOptResults,
    *,
    segment_duration_s: float | None = None,
) -> tuple[Figure, np.ndarray]:
    """Backward-compatible alias for the 3-row training-signal grid."""
    return plot_gradients_delta_w_eff_lr_grid(
        results,
        segment_duration_s=segment_duration_s,
    )


def _draw_ratemap_panel(
    fig: Figure,
    gs: gridspec.SubplotSpec,
    final_ratemap: np.ndarray,
    *,
    peak_rates: np.ndarray,
    vmax: float,
) -> None:
    if peak_rates is not None:
        order_desc = np.argsort(np.asarray(peak_rates, dtype=np.float64))[::-1]
    else:
        peak_fr = np.nanmax(final_ratemap, axis=(1, 2))
        order_desc = np.argsort(peak_fr)[::-1]

    inner = gridspec.GridSpecFromSubplotSpec(
        N_ROWS,
        N_COLS,
        subplot_spec=gs,
        wspace=0.02,
        hspace=0.02,
    )
    im = None
    for rank, unit in enumerate(order_desc):
        r, c = rank // N_COLS, rank % N_COLS
        ax = fig.add_subplot(inner[r, c])
        im = ax.imshow(final_ratemap[unit], cmap="jet", vmin=0, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
    return im


def plot_final_ratemap_grid_by_optimizer(
    results: MultiOptResults,
) -> Figure:
    vmax = results.shared_ratemap_vmax
    n_opt = len(OPTIMIZER_ORDER)
    margin_left = 0.02
    margin_right = 0.92
    fig = plt.figure(figsize=(6.5 * n_opt, 14))
    outer = gridspec.GridSpec(
        1,
        n_opt,
        figure=fig,
        wspace=0.05,
        top=0.94,
        bottom=0.04,
        left=margin_left,
        right=margin_right,
    )
    ims = []
    for col, opt in enumerate(OPTIMIZER_ORDER):
        meta = results.meta[opt]
        im = _draw_ratemap_panel(
            fig,
            outer[col],
            meta.final_ratemap,
            peak_rates=meta.final_peak_rates,
            vmax=vmax,
        )
        if im is not None:
            ims.append(im)

    if ims:
        fig.colorbar(
            ims[0],
            ax=fig.axes,
            shrink=0.5,
            label="Firing rate",
            pad=0.01,
        )

    fig.canvas.draw()
    cells_per_panel = N_ROWS * N_COLS
    col_bboxes: list[tuple[float, float, float]] = []
    for col in range(n_opt):
        top_axes = fig.axes[col * cells_per_panel : col * cells_per_panel + N_COLS]
        x0 = min(ax.get_position().x0 for ax in top_axes)
        x1 = max(ax.get_position().x1 for ax in top_axes)
        y1 = max(ax.get_position().y1 for ax in top_axes)
        col_bboxes.append((x0, x1, y1))

    grid_center_x = (col_bboxes[0][0] + col_bboxes[-1][1]) / 2
    for col, opt in enumerate(OPTIMIZER_ORDER):
        x0, x1, y1 = col_bboxes[col]
        fig.text(
            (x0 + x1) / 2,
            y1 + 0.004,
            OPTIMIZER_LABELS[opt],
            ha="center",
            va="bottom",
            fontsize=12,
        )

    fig.suptitle(
        f"Final-epoch Gaussian fits of ratemaps, sorted by peak",
        x=grid_center_x,
        y=0.98,
        ha="center",
    )
    return fig


def plot_alive_proportion_by_optimizer(
    results: MultiOptResults,
    *,
    r2_threshold: float,
    segment_duration_s: float | None = None,
    trace_alpha: float = 0.4,
    y_lims: tuple[float, float] | None = (0.0, 0.6),
) -> tuple[Figure, plt.Axes]:
    if segment_duration_s is None:
        segment_duration_s = next(iter(results.contexts.values())).segment_duration_s

    fig, ax = plt.subplots(figsize=(14, 4))
    for opt in OPTIMIZER_ORDER:
        alive_ts = results.pf[opt].alive_ts
        ax.plot(
            index_to_time_s(alive_ts["global_capture_idx"], segment_duration_s),
            alive_ts["proportion_alive"],
            "o-",
            color=OPTIMIZER_COLORS[opt],
            markersize=2,
            linewidth=1,
            alpha=trace_alpha,
        )
    _add_traj_boundaries(
        ax,
        results.pf[OPTIMIZER_ORDER[0]].alive_ts,
        x_col="global_capture_idx",
        segment_duration_s=segment_duration_s,
    )
    ax.set_xlabel(TIME_AXIS_LABEL)
    ax.set_ylabel("Proportion of cells")
    ax.set_xlim(0,None)
    # ax.set_title(
    #     f"Alive cells per capture by optimizer (R²>{r2_threshold:.0%}; per-run dead threshold)"
    # )
    if y_lims is not None:
        ax.set_ylim(*y_lims)
    ax.legend(
        handles=_optimizer_legend_handles(
            linestyle="-",
            linewidth=1.0,
            marker="o",
            markersize=2.0,
        ),
        frameon=False,
        loc="lower center",
        ncol=len(OPTIMIZER_ORDER),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig, ax


def _plot_pf_raster_on_ax(
    ax,
    pf_segment_df: pd.DataFrame,
    *,
    segment_duration_s: float,
    color_by_pf: dict[int, tuple],
    title: str,
    revive_events: pd.DataFrame | None = None,
) -> None:
    active = pf_segment_df[
        pf_segment_df["is_segment"] & (pf_segment_df["state"] == "active")
    ].copy()
    if active.empty:
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Neuron id")
        return

    seg_captures = np.sort(active["capture_idx"].unique())
    cap_to_time = {
        int(cap): i * segment_duration_s for i, cap in enumerate(seg_captures)
    }
    active["time_s"] = active["capture_idx"].map(cap_to_time).astype(np.float64)
    point_colors = active["pf_idx"].map(lambda pf: color_by_pf[int(pf)]).to_numpy()
    n_neurons = int(active["cell_idx"].max()) + 1
    ax.scatter(
        active["time_s"],
        active["cell_idx"],
        c=point_colors,
        s=4.0,
        alpha=0.4,
        linewidths=0,
        rasterized=True,
    )
    if revive_events is not None and not revive_events.empty:
        revives = revive_events.copy()
        revives["time_s"] = revives["capture_idx"].map(cap_to_time).astype(np.float64)
        ax.scatter(
            revives["time_s"],
            revives["cell_idx"],
            marker="x",
            c="red",
            s=1.5,
            linewidths=1.2,
            zorder=3,
            rasterized=True,
        )
    ax.set_xlabel("Time (s)")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-0.5, n_neurons - 0.5)
    ax.set_xlim(active["time_s"].min(), active["time_s"].max())


def _pf_active_scatter_xy(
    pf_segment_df: pd.DataFrame,
    *,
    segment_duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    active = pf_segment_df[
        pf_segment_df["is_segment"] & (pf_segment_df["state"] == "active")
    ]
    if active.empty:
        return np.array([]), np.array([])
    seg_captures = np.sort(active["capture_idx"].unique())
    cap_to_time = {
        int(cap): i * segment_duration_s for i, cap in enumerate(seg_captures)
    }
    time_s = active["capture_idx"].map(cap_to_time).astype(np.float64).to_numpy()
    cell_idx = active["cell_idx"].to_numpy()
    return time_s, cell_idx


def _pf_revive_scatter_xy(
    revive_events: pd.DataFrame | None,
    pf_segment_df: pd.DataFrame,
    *,
    segment_duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    if revive_events is None or revive_events.empty:
        return np.array([]), np.array([])
    active = pf_segment_df[
        pf_segment_df["is_segment"] & (pf_segment_df["state"] == "active")
    ]
    if active.empty:
        return np.array([]), np.array([])
    seg_captures = np.sort(active["capture_idx"].unique())
    cap_to_time = {
        int(cap): i * segment_duration_s for i, cap in enumerate(seg_captures)
    }
    revives = revive_events.copy()
    revives["time_s"] = revives["capture_idx"].map(cap_to_time).astype(np.float64)
    revives = revives.dropna(subset=["time_s"])
    if revives.empty:
        return np.array([]), np.array([])
    return revives["time_s"].to_numpy(), revives["cell_idx"].to_numpy()


def _plot_truncated_time_kde(
    ax_time,
    time_values: np.ndarray,
    *,
    bins: np.ndarray,
    time_lo: float,
    time_hi: float,
    color: str,
    linewidth: float = 1.5,
) -> None:
    if time_values.size < 2 or np.ptp(time_values) <= 0:
        return
    kde_time = gaussian_kde(time_values)
    bw = float(kde_time.factor * np.std(time_values, ddof=1))
    bin_width = bins[1] - bins[0]
    kde_inset = max(0.85 * bw, 2.5 * bin_width)
    x_kde_lo = time_lo + kde_inset
    x_kde_hi = time_hi - kde_inset
    if x_kde_hi <= x_kde_lo:
        return
    x_kde = np.linspace(x_kde_lo, x_kde_hi, 500)
    y_kde = kde_time(x_kde)
    ax_time.plot(x_kde, y_kde, color=color, linewidth=linewidth, clip_on=True)


def _plot_scatter_marginals_on_axes(
    ax_time,
    ax_cell=None,
    *,
    time_s: np.ndarray,
    cell_idx: np.ndarray,
    revive_time_s: np.ndarray | None = None,
    revive_cell_idx: np.ndarray | None = None,
    show_cell_marginals: bool = True,
    set_time_ylim: bool = True,
    time_bins: int = 50,
) -> float:
    revive_time_s = np.asarray([] if revive_time_s is None else revive_time_s, dtype=float)
    revive_cell_idx = np.asarray(
        [] if revive_cell_idx is None else revive_cell_idx,
        dtype=float,
    )
    if (
        time_s.size == 0
        and cell_idx.size == 0
        and revive_time_s.size == 0
        and revive_cell_idx.size == 0
    ):
        ax_time.set_visible(False)
        if show_cell_marginals and ax_cell is not None:
            ax_cell.set_visible(False)
        return 0.0

    time_density_peak = 0.0
    time_for_bins = (
        np.concatenate([time_s, revive_time_s])
        if time_s.size and revive_time_s.size
        else (time_s if time_s.size else revive_time_s)
    )
    if time_for_bins.size:
        time_lo, time_hi = float(time_for_bins.min()), float(time_for_bins.max())
        bins = np.linspace(time_lo, time_hi, time_bins + 1)
        ax_time.set_xlim(time_lo, time_hi)
        density_peaks: list[float] = []

        if time_s.size:
            activity_counts, _ = np.histogram(time_s, bins=bins, density=True)
            ax_time.hist(
                time_s,
                bins=bins,
                density=True,
                alpha=0.45,
                color="#A589D1",
                edgecolor="#5E3B97",
                linewidth=0.4,
                label="activity",
            )
            density_peaks.append(float(activity_counts.max()))
            _plot_truncated_time_kde(
                ax_time,
                time_s,
                bins=bins,
                time_lo=time_lo,
                time_hi=time_hi,
                color="#5E3B97",
            )

        if revive_time_s.size:
            revive_counts, _ = np.histogram(revive_time_s, bins=bins, density=True)
            ax_time.hist(
                revive_time_s,
                bins=bins,
                density=True,
                alpha=0.25,
                color="#d62728",
                edgecolor="#9a1b1b",
                linewidth=0.4,
                label="revives",
            )
            density_peaks.append(float(revive_counts.max()))
            _plot_truncated_time_kde(
                ax_time,
                revive_time_s,
                bins=bins,
                time_lo=time_lo,
                time_hi=time_hi,
                color="#d62728",
            )

        if density_peaks:
            time_density_peak = max(density_peaks)
            if set_time_ylim:
                ax_time.set_ylim(0, time_density_peak * 1.12)
    else:
        ax_time.set_visible(False)

    if not show_cell_marginals or ax_cell is None:
        ax_time.tick_params(labelbottom=False)
        return time_density_peak

    cells_for_bins = (
        np.concatenate([cell_idx, revive_cell_idx])
        if cell_idx.size and revive_cell_idx.size
        else (cell_idx if cell_idx.size else revive_cell_idx)
    )
    if cells_for_bins.size:
        y_min, y_max = int(cells_for_bins.min()), int(cells_for_bins.max())
        cell_bin_width = 1
        cell_bins = np.arange(
            y_min - 0.5,
            y_max + cell_bin_width + 0.5,
            cell_bin_width,
        )
        cell_density_peaks: list[float] = []

        if cell_idx.size:
            cell_counts, _ = np.histogram(cell_idx, bins=cell_bins, density=True)
            ax_cell.hist(
                cell_idx,
                bins=cell_bins,
                density=True,
                orientation="horizontal",
                alpha=0.45,
                color="#C3B1E1",
                edgecolor="#5E3B97",
                linewidth=0.4,
                label="activity",
            )
            cell_density_peaks.append(float(cell_counts.max()))

        if revive_cell_idx.size:
            revive_cell_counts, _ = np.histogram(
                revive_cell_idx,
                bins=cell_bins,
                density=True,
            )
            ax_cell.hist(
                revive_cell_idx,
                bins=cell_bins,
                density=True,
                orientation="horizontal",
                alpha=0.25,
                color="#d62728",
                edgecolor="#9a1b1b",
                linewidth=0.4,
                label="revives",
            )
            cell_density_peaks.append(float(revive_cell_counts.max()))

        if cell_density_peaks:
            ax_cell.set_xlim(0, max(cell_density_peaks) * 0.75)
    else:
        ax_cell.set_visible(False)

    ax_time.tick_params(labelbottom=False)
    ax_cell.tick_params(labelleft=False)
    return time_density_peak


def plot_pf_activity_raster_by_optimizer(
    results: MultiOptResults,
    *,
    show_cell_marginals: bool = True,
) -> tuple[Figure, list]:
    pf_segment_dfs = {opt: results.pf[opt].pf_segment_df for opt in OPTIMIZER_ORDER}
    color_by_pf = _shared_pf_colormap(pf_segment_dfs)
    sample = pf_segment_dfs[OPTIMIZER_ORDER[0]]
    sample_active = sample[sample["is_segment"] & (sample["state"] == "active")]
    n_neurons = int(sample_active["cell_idx"].max()) + 1 if not sample_active.empty else 1
    fig_h = max(8.0, n_neurons * 0.012)
    fig_w = 22 if show_cell_marginals else 20
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer_gs = fig.add_gridspec(1, len(OPTIMIZER_ORDER), wspace=0.35)
    scatter_axes: list = []
    time_axes: list = []
    max_time_density = 0.0

    for i, opt in enumerate(OPTIMIZER_ORDER):
        if show_cell_marginals:
            inner_gs = outer_gs[i].subgridspec(
                2,
                2,
                width_ratios=(5, 1.4),
                height_ratios=(1, 5),
                hspace=0.05,
                wspace=0.05,
            )
            scatter_spec = inner_gs[1, 0]
            time_spec = inner_gs[0, 0]
            cell_spec = inner_gs[1, 1]
        else:
            inner_gs = outer_gs[i].subgridspec(
                2,
                1,
                height_ratios=(1, 5),
                hspace=0.05,
            )
            scatter_spec = inner_gs[1, 0]
            time_spec = inner_gs[0, 0]
            cell_spec = None

        if i == 0:
            ax_scatter = fig.add_subplot(scatter_spec)
        else:
            ax_scatter = fig.add_subplot(scatter_spec, sharey=scatter_axes[0])
        if i == 0:
            ax_time = fig.add_subplot(time_spec, sharex=ax_scatter)
        else:
            ax_time = fig.add_subplot(time_spec, sharex=ax_scatter, sharey=time_axes[0])
            ax_time.tick_params(labelleft=False)
        ax_cell = (
            fig.add_subplot(cell_spec, sharey=ax_scatter)
            if cell_spec is not None
            else None
        )
        scatter_axes.append(ax_scatter)
        time_axes.append(ax_time)

        ctx = results.contexts[opt]
        pf_data = results.pf[opt]
        _plot_pf_raster_on_ax(
            ax_scatter,
            pf_data.pf_segment_df,
            segment_duration_s=ctx.segment_duration_s,
            color_by_pf=color_by_pf,
            title=OPTIMIZER_LABELS[opt],
            revive_events=pf_data.revive_events,
        )
        panel_title = ax_scatter.get_title()
        ax_scatter.set_title("")
        ax_time.set_title(panel_title, fontsize=10)

        time_s, cell_idx = _pf_active_scatter_xy(
            pf_data.pf_segment_df,
            segment_duration_s=ctx.segment_duration_s,
        )
        revive_time_s, revive_cell_idx = _pf_revive_scatter_xy(
            pf_data.revive_events,
            pf_data.pf_segment_df,
            segment_duration_s=ctx.segment_duration_s,
        )
        time_density_peak = _plot_scatter_marginals_on_axes(
            ax_time,
            ax_cell,
            time_s=time_s,
            cell_idx=cell_idx,
            revive_time_s=revive_time_s,
            revive_cell_idx=revive_cell_idx if show_cell_marginals else None,
            show_cell_marginals=show_cell_marginals,
            set_time_ylim=False,
        )
        max_time_density = max(max_time_density, time_density_peak)

        if i == 0:
            ax_scatter.set_ylabel("Neuron id")

    if time_axes and max_time_density > 0:
        time_axes[0].set_ylim(0, max_time_density * 1.12)

    fig.suptitle("Active PFs and revivals by optimizer", y=0.95)
    fig.tight_layout()
    return fig, scatter_axes


def plot_mean_pfs_per_neuron_by_optimizer(
    results: MultiOptResults,
    *,
    show_significance: bool = True,
    show_insignificance: bool = False,
) -> tuple[Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(OPTIMIZER_ORDER))
    means, cis, values_by_optimizer, positions = [], [], {}, {}
    for i, opt in enumerate(OPTIMIZER_ORDER):
        stats_dict = results.pf[opt].pf_per_neuron
        means.append(stats_dict["mean"])
        cis.append(stats_dict["ci95"])
        values_by_optimizer[opt] = stats_dict["counts"]
        positions[opt] = float(i)

    ax.bar(
        x,
        means,
        width=0.6,
        color=[OPTIMIZER_COLORS[opt] for opt in OPTIMIZER_ORDER],
        alpha=0.75,
        edgecolor="k",
    )
    ax.errorbar(x, means, yerr=cis, fmt="none", ecolor="k", capsize=8, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([OPTIMIZER_LABELS[opt] for opt in OPTIMIZER_ORDER])
    ax.set_ylabel("PF / neuron")
    ax.set_title("Mean PF counts per neuron")
    bar_tops = np.asarray(means) + np.asarray(cis)
    ax.set_ylim(0, float(np.nanmax(bar_tops)) * 1.05 if bar_tops.size else 1)
    _handle_optimizer_significance(
        ax,
        values_by_optimizer,
        positions=positions,
        bar_tops=bar_tops,
        panel_label="Mean PFs per neuron",
        show_significance=show_significance,
        show_insignificance=show_insignificance,
    )
    fig.tight_layout()
    return fig, ax


def _transition_line_legend_handle(color: str, label: str) -> Line2D:
    return Line2D(
        [0],
        [0],
        color=color,
        marker="o",
        linestyle="--",
        markersize=3,
        linewidth=1.2,
        label=label,
    )


def _pf_transition_legend_handles(*, show_revives: bool) -> list[Line2D]:
    handles = [
        _transition_line_legend_handle(
            TRANSITION_COLORS["inactive_to_active"],
            "Inactive → active",
        ),
        _transition_line_legend_handle(
            TRANSITION_COLORS["active_to_inactive"],
            "Active → inactive",
        ),
    ]
    if show_revives:
        handles.append(
            _transition_line_legend_handle(TRANSITION_COLORS["revives"], "Revives")
        )
    return handles


def _gauss_transition_legend_handles() -> list[Line2D]:
    return [
        _transition_line_legend_handle(GAUSS_TRANSITION_COLORS[col], label)
        for col, label in (
            ("prop_1_to_2", "1 → 2"),
            ("prop_2_to_1", "2 → 1"),
            ("prop_1_to_1", "1 → 1"),
            ("prop_2_to_2", "2 → 2"),
        )
    ]


def _plot_pf_transitions_on_ax(
    ax,
    transition_counts: pd.DataFrame,
    *,
    title: str,
    show_revives: bool,
    segment_duration_s: float,
    bin_window: int = 10,
    show_legend: bool = True,
) -> None:
    x_raw = index_to_time_s(
        transition_counts["global_transition_idx"],
        segment_duration_s,
    )
    x_in, y_in, sem_in = _bin_transition_series(
        x_raw,
        transition_counts["n_dead_to_alive"].to_numpy(),
        bin_window=bin_window,
    )
    x_out, y_out, sem_out = _bin_transition_series(
        x_raw,
        transition_counts["n_alive_to_dead"].to_numpy(),
        bin_window=bin_window,
    )

    color_out = TRANSITION_COLORS["active_to_inactive"]
    if sem_out is None:
        ax.plot(
            x_out,
            y_out,
            "o--",
            color=color_out,
            markersize=3,
            linewidth=1.2,
            label="Active → inactive",
            zorder=4,
        )
    else:
        ci_out = 1.96 * sem_out
        ax.fill_between(
            x_out,
            y_out - ci_out,
            y_out + ci_out,
            alpha=0.2,
            color=color_out,
            zorder=1,
        )
        ax.plot(
            x_out,
            y_out,
            "-",
            color=color_out,
            linewidth=1.5,
            label="Active → inactive",
            zorder=4,
        )

    color_in = TRANSITION_COLORS["inactive_to_active"]
    ax.fill_between(x_in, 0, y_in, alpha=0.28, color=color_in, zorder=2)
    if sem_in is None:
        ax.plot(
            x_in,
            y_in,
            "o--",
            color=color_in,
            markersize=3,
            linewidth=1.2,
            label="Inactive → active",
            zorder=5,
        )
    else:
        ax.plot(
            x_in,
            y_in,
            "-",
            color=color_in,
            linewidth=1.5,
            label="Inactive → active",
            zorder=5,
        )

    if show_revives:
        x_rev, y_rev, sem_rev = _bin_transition_series(
            x_raw,
            transition_counts["n_revives"].to_numpy(),
            bin_window=bin_window,
        )
        color_rev = TRANSITION_COLORS["revives"]
        ax.fill_between(x_rev, 0, y_rev, alpha=0.42, color=color_rev, zorder=3)
        if sem_rev is None:
            ax.plot(
                x_rev,
                y_rev,
                "o--",
                color=color_rev,
                markersize=3,
                linewidth=1.2,
                label="Revives",
                zorder=6,
            )
        else:
            ax.plot(
                x_rev,
                y_rev,
                "-",
                color=color_rev,
                linewidth=1.5,
                label="Revives",
                zorder=6,
            )

    for xv in index_to_time_s(
        traj_boundary_vlines(transition_counts, x_col="global_transition_idx"),
        segment_duration_s,
    ):
        ax.axvline(xv, color="0.5", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xlabel(TIME_AXIS_LABEL)
    ax.set_ylabel("PF count")
    ax.set_title(title, fontsize=10)
    if show_legend and show_revives:
        ax.legend(frameon=False, fontsize=7)


def _plot_gauss_transitions_on_ax(
    ax,
    gauss_props: pd.DataFrame,
    *,
    title: str,
    segment_duration_s: float,
) -> None:
    x = index_to_time_s(gauss_props["global_transition_idx"], segment_duration_s)
    styles = [
        ("prop_1_to_2", GAUSS_TRANSITION_COLORS["prop_1_to_2"], "1 → 2"),
        ("prop_2_to_1", GAUSS_TRANSITION_COLORS["prop_2_to_1"], "2 → 1"),
        ("prop_1_to_1", GAUSS_TRANSITION_COLORS["prop_1_to_1"], "1 → 1"),
        ("prop_2_to_2", GAUSS_TRANSITION_COLORS["prop_2_to_2"], "2 → 2"),
    ]
    for col, color, label in styles:
        ax.plot(
            x,
            gauss_props[col],
            "o--",
            color=color,
            markersize=3,
            linewidth=1.2,
            label=label,
            alpha=0.7,
        )
    for xv in index_to_time_s(
        traj_boundary_vlines(gauss_props, x_col="global_transition_idx"),
        segment_duration_s,
    ):
        ax.axvline(xv, color="0.5", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xlabel(TIME_AXIS_LABEL)
    ax.set_ylabel("Cell proportion")
    ax.set_title(title, fontsize=10)


def plot_state_transitions_grid_by_optimizer(
    results: MultiOptResults,
    *,
    gauss_threshold: float,
    pf_tracking_method: str,
    segment_duration_s: float | None = None,
    bin_window: int = 10,
) -> tuple[Figure, np.ndarray]:
    if segment_duration_s is None:
        segment_duration_s = next(iter(results.contexts.values())).segment_duration_s

    n_cols = len(OPTIMIZER_ORDER)
    fig = plt.figure(figsize=(30, 7.5))
    gs = fig.add_gridspec(
        3,
        n_cols,
        height_ratios=[1, 0.14, 1],
        hspace=0.35,
        wspace=0.25,
    )
    axes = np.empty((2, n_cols), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[0, 0])
    axes[1, 0] = fig.add_subplot(gs[2, 0], sharex=axes[0, 0])
    for col in range(1, n_cols):
        axes[0, col] = fig.add_subplot(gs[0, col], sharey=axes[0, 0])
        axes[1, col] = fig.add_subplot(
            gs[2, col],
            sharey=axes[1, 0],
            sharex=axes[0, col],
        )

    show_revives = pf_tracking_method != "no_revive"

    for col, opt in enumerate(OPTIMIZER_ORDER):
        pf_data = results.pf[opt]
        _plot_pf_transitions_on_ax(
            axes[0, col],
            pf_data.pf_transition_counts,
            title=OPTIMIZER_LABELS[opt],
            show_revives=show_revives,
            segment_duration_s=segment_duration_s,
            bin_window=bin_window,
            show_legend=False,
        )
        _plot_gauss_transitions_on_ax(
            axes[1, col],
            pf_data.gauss_transition_props,
            title=OPTIMIZER_LABELS[opt],
            segment_duration_s=segment_duration_s,
        )
        axes[0, col].set_xlabel("")
        plt.setp(axes[0, col].get_xticklabels(), visible=False)

    legend_gs = gs[1, :].subgridspec(2, 1, hspace=0.08)
    legend_ax_pf = fig.add_subplot(legend_gs[0, 0])
    legend_ax_gauss = fig.add_subplot(legend_gs[1, 0])
    legend_ax_pf.axis("off")
    legend_ax_gauss.axis("off")
    legend_ax_pf.legend(
        handles=_pf_transition_legend_handles(show_revives=show_revives),
        loc="center",
        ncol=3 if show_revives else 2,
        frameon=False,
        fontsize=8,
    )
    legend_ax_gauss.legend(
        handles=_gauss_transition_legend_handles(),
        loc="center",
        ncol=4,
        frameon=False,
        fontsize=8,
    )

    axes[0, 0].set_ylabel("PF count")
    axes[1, 0].set_ylabel("Cell proportion")
    fig.suptitle(
        "Field (top) and cell (bottom) state transitions over training",
        y=1.02,
    )
    fig.tight_layout()
    return fig, axes


def plot_displacement_curves_by_optimizer(
    results: MultiOptResults,
    *,
    dashed: bool = False,
    overlay_vaidya: bool = False,
    room_extent_cm: float = 100.0,
    zoom_bin_width: float = 0.25,
) -> tuple[Figure, np.ndarray]:
    line_style = "--" if dashed else "-"
    panels = [
        ("displacement_l2", "PF shift L2 (cm)", (0, 15)),
        ("displacement_x", "PF shift Δx (cm)", (-8, 8)),
        ("displacement_y", "PF shift Δy (cm)", (-8, 8)),
    ]
    vaidya_overlays: dict[str, dict[str, np.ndarray]] = {}
    if overlay_vaidya:
        from report_final import vaidya_targets as vt
        from report_final.style import report_reference_style

        vaidya_ref_style = report_reference_style()
        vaidya_overlays = {
            "displacement_x": vt.pf_shift_signed_proportion_overlay(
                model_bin_width=zoom_bin_width,
                room_extent_cm=room_extent_cm,
            ),
            "displacement_y": vt.pf_shift_signed_proportion_overlay(
                model_bin_width=zoom_bin_width,
                room_extent_cm=room_extent_cm,
            ),
            "displacement_l2": vt.pf_shift_abs_proportion_overlay(
                model_bin_width=zoom_bin_width,
                room_extent_cm=room_extent_cm,
            ),
        }
    else:
        vaidya_ref_style = None

    fig, axes = plt.subplots(len(panels), 1, figsize=(8.4, 3.2 * len(panels)))
    for ax, (col, xlabel, xlim) in zip(axes, panels, strict=True):
        for opt in OPTIMIZER_ORDER:
            centers, prop = results.pf[opt].displacement_curves[col]
            ax.plot(
                centers,
                prop,
                line_style,
                color=OPTIMIZER_COLORS[opt],
                linewidth=1.5,
                alpha=0.85,
                label=OPTIMIZER_LABELS[opt],
            )
        ymax = max(
            float(results.pf[opt].displacement_curves[col][1].max())
            for opt in OPTIMIZER_ORDER
        )
        if overlay_vaidya:
            ref = vaidya_overlays[col]
            mask = (ref["centers"] >= xlim[0]) & (ref["centers"] <= xlim[1])
            ax.plot(
                ref["centers"][mask],
                ref["proportion"][mask],
                label=vaidya_ref_style["label"],
                **{k: v for k, v in vaidya_ref_style.items() if k != "label"},
            )
            ymax = max(ymax, float(ref["proportion"][mask].max()))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Proportion per bin")
        ax.set_xlim(xlim)
        ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1)
    axes[0].set_ylabel("Proportion per bin")
    axes[0].legend(frameon=False, loc="upper right", fontsize=8)
    title = "Tracked PF drift"
    # if overlay_vaidya:
    #     title += "; Vaidya Fig 2d rescaled to room extent"
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, axes


def _plot_pf_metric_panel_by_optimizer(
    ax,
    results: MultiOptResults,
    metric_col: str,
    *,
    ylabel: str,
    plot_kind: str,
    show_significance: bool = True,
    show_insignificance: bool = False,
    bracket_step_mult: float = 1.5,
    bar_width: float = 0.55,
) -> None:
    x = np.arange(len(OPTIMIZER_ORDER))
    values_by_optimizer = _metric_values_by_optimizer(results, metric_col)
    positions = _optimizer_positions()

    if plot_kind == "boxplot":
        data = [values_by_optimizer[opt] for opt in OPTIMIZER_ORDER]
        bp = ax.boxplot(
            data,
            positions=x,
            widths=0.55,
            patch_artist=True,
            flierprops={
                "marker": "o",
                "markersize": 2,
                "alpha": 0.35,
                "markerfacecolor": "0.45",
                "markeredgecolor": "0.45",
                "linestyle": "none",
            },
        )
        for patch, opt in zip(bp["boxes"], OPTIMIZER_ORDER, strict=True):
            patch.set_facecolor(OPTIMIZER_COLORS[opt])
            patch.set_alpha(0.75)
            patch.set_edgecolor("k")
        for median in bp["medians"]:
            median.set_color("k")
            median.set_linewidth(1.5)
        bar_tops = np.array(
            [
                np.nanmax(values_by_optimizer[opt])
                if values_by_optimizer[opt].size
                else 0.0
                for opt in OPTIMIZER_ORDER
            ]
        )
    else:
        means, cis = [], []
        for opt in OPTIMIZER_ORDER:
            mean, _sem, ci = _mean_sem_ci95(values_by_optimizer[opt])
            means.append(mean)
            cis.append(ci)
        ax.bar(
            x,
            means,
            width=bar_width,
            color=[OPTIMIZER_COLORS[opt] for opt in OPTIMIZER_ORDER],
            alpha=0.75,
            edgecolor="k",
        )
        ax.errorbar(x, means, yerr=cis, fmt="none", ecolor="k", capsize=4, linewidth=1.2)
        bar_tops = np.asarray(means) + np.asarray(cis)
        data_top = float(np.nanmax(bar_tops)) if bar_tops.size else 0.0
        ax.set_ylim(0, data_top * 1.05 if np.isfinite(data_top) and data_top > 0 else 1)

    _handle_optimizer_significance(
        ax,
        values_by_optimizer,
        positions=positions,
        bar_tops=bar_tops,
        panel_label=ylabel,
        show_significance=show_significance,
        show_insignificance=show_insignificance,
        bracket_step_mult=bracket_step_mult,
        test="ks" if plot_kind == "boxplot" else "welch",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([OPTIMIZER_LABELS[opt] for opt in OPTIMIZER_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel, fontsize=10)


def _plot_peak_amp_tma_bars(
    ax,
    results: MultiOptResults,
    *,
    time_unit: str,
    show_significance: bool = True,
    show_insignificance: bool = False,
    bracket_step_mult: float = 1.65,
    shorten_rat_label: bool = True,
    show_legend: bool = True,
) -> None:
    tma_col, tma_label = tma_metric_spec(time_unit)
    time_values = _metric_values_by_optimizer(results, tma_col)
    amp_values = _metric_values_by_optimizer(results, "peak_amplitude")
    n_opts = len(OPTIMIZER_ORDER)
    group_gap = 2.5
    bar_width = 0.95
    x_time = np.arange(n_opts, dtype=float)
    x_amp = np.arange(n_opts, dtype=float) + n_opts + group_gap

    time_means, time_cis, amp_means, amp_cis = [], [], [], []
    for opt in OPTIMIZER_ORDER:
        t_mean, _t_sem, t_ci = _mean_sem_ci95(time_values[opt])
        a_mean, _a_sem, a_ci = _mean_sem_ci95(amp_values[opt])
        time_means.append(t_mean)
        time_cis.append(t_ci)
        amp_means.append(a_mean)
        amp_cis.append(a_ci)

    time_colors = [OPTIMIZER_COLORS[opt] for opt in OPTIMIZER_ORDER]
    ax.bar(
        x_time,
        time_means,
        width=bar_width,
        color=time_colors,
        alpha=0.75,
        edgecolor="k",
    )
    ax.errorbar(
        x_time,
        time_means,
        yerr=time_cis,
        fmt="none",
        ecolor="k",
        capsize=3,
        linewidth=1.0,
    )

    ax2 = ax.twinx()
    ax2.bar(
        x_amp,
        amp_means,
        width=bar_width,
        color=time_colors,
        alpha=0.45,
        edgecolor="k",
        hatch="//",
    )
    ax2.errorbar(
        x_amp,
        amp_means,
        yerr=amp_cis,
        fmt="none",
        ecolor="k",
        capsize=3,
        linewidth=1.0,
    )
    ylabel = tma_label
    if shorten_rat_label:
        ylabel = ylabel.replace("Representation Acquisition Time", "RAT")
    ax.set_ylabel(ylabel)
    ax2.set_ylabel("Peak amplitude")
    ax.set_title(f"Peak amplitude and {ylabel}", fontsize=10)
    ax.set_xticks(np.concatenate([x_time, x_amp]))
    ax.set_xticklabels(
        [OPTIMIZER_LABELS[opt] for opt in OPTIMIZER_ORDER]
        + [OPTIMIZER_LABELS[opt] for opt in OPTIMIZER_ORDER]
    )
    ax.tick_params(axis="x", pad=6)
    for x_group, group_label in (
        (x_time, tma_label),
        (x_amp, "Peak amplitude"),
    ):
        ax.text(
            float(np.mean(x_group)),
            -0.05,
            group_label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            clip_on=False,
        )
    ax.set_xlim(x_time[0] - bar_width / 1.5, x_amp[-1] + bar_width / 1.5)
    ax2.set_xticks([])

    time_tops = np.asarray(time_means, dtype=float) + np.asarray(time_cis, dtype=float)
    amp_tops = np.asarray(amp_means, dtype=float) + np.asarray(amp_cis, dtype=float)
    time_top = float(np.nanmax(time_tops)) if time_tops.size else 0.0
    amp_top = float(np.nanmax(amp_tops)) if amp_tops.size else 0.0
    ax.set_ylim(0, time_top * 1.05 if np.isfinite(time_top) and time_top > 0 else 1)
    ax2.set_ylim(0, amp_top * 1.05 if np.isfinite(amp_top) and amp_top > 0 else 1)

    if show_legend:
        ax.legend(
            handles=_optimizer_patch_legend_handles(),
            loc="upper center",
            ncol=1,
            frameon=True,
            framealpha=0.92,
            fontsize=8,
        )

    values_by_metric = {tma_col: time_values, "peak_amplitude": amp_values}
    metric_tags = {tma_col: "time", "peak_amplitude": "amplitude"}
    panel_label = f"Peak amplitude and {'RAT' if shorten_rat_label else 'Representation Acquisition Time'}"
    if not show_significance:
        _print_multi_metric_significance(
            panel_label=panel_label,
            values_by_metric=values_by_metric,
            metric_tags=metric_tags,
        )
    else:
        time_positions = _optimizer_positions_at(x_time)
        amp_positions = _optimizer_positions_at(x_amp)
        time_pair_labels = _pairwise_bracket_labels(
            time_values,
            tag="time",
            show_insignificance=show_insignificance,
            positions=time_positions,
        )
        amp_pair_labels = _pairwise_bracket_labels(
            amp_values,
            tag="amplitude",
            show_insignificance=show_insignificance,
            positions=amp_positions,
        )
        _add_labeled_pairwise_brackets(
            ax,
            positions=time_positions,
            bar_top=time_top,
            pair_labels=time_pair_labels,
            bracket_step_mult=bracket_step_mult,
        )
        _add_labeled_pairwise_brackets(
            ax2,
            positions=amp_positions,
            bar_top=amp_top,
            pair_labels=amp_pair_labels,
            bracket_step_mult=bracket_step_mult,
        )


def _spontaneous_values_by_optimizer(
    results: MultiOptResults,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return period-level one_segment flags and per-cell spontaneous rates."""
    period_values: dict[str, np.ndarray] = {}
    cell_rate_values: dict[str, np.ndarray] = {}
    for opt in OPTIMIZER_ORDER:
        periods = results.pf[opt].life_period_metrics
        if periods.empty:
            period_values[opt] = np.asarray([], dtype=float)
            cell_rate_values[opt] = np.asarray([], dtype=float)
            continue
        period_values[opt] = periods["one_segment"].astype(float).to_numpy()
        cell_rates = (
            periods.groupby("cell_idx", sort=False)["one_segment"]
            .mean()
            .to_numpy(dtype=float)
        )
        cell_rate_values[opt] = cell_rates
    return period_values, cell_rate_values


def _plot_spontaneous_pf_bars(
    ax,
    results: MultiOptResults,
    *,
    show_significance: bool = True,
    show_insignificance: bool = False,
    bracket_step_mult: float = 1.65,
    show_legend: bool = True,
) -> None:
    period_values, cell_rate_values = _spontaneous_values_by_optimizer(results)
    n_opts = len(OPTIMIZER_ORDER)
    group_gap = 2.5
    bar_width = 0.95
    x_pct = np.arange(n_opts, dtype=float)
    x_rate = np.arange(n_opts, dtype=float) + n_opts + group_gap

    pct_means, pct_cis, rate_means, rate_cis = [], [], [], []
    for opt in OPTIMIZER_ORDER:
        pct_vals = period_values[opt] * 100.0
        rate_vals = cell_rate_values[opt] * 100.0
        p_mean, _p_sem, p_ci = _mean_sem_ci95(pct_vals)
        r_mean, _r_sem, r_ci = _mean_sem_ci95(rate_vals)
        pct_means.append(p_mean)
        pct_cis.append(p_ci)
        rate_means.append(r_mean)
        rate_cis.append(r_ci)

    bar_colors = [OPTIMIZER_COLORS[opt] for opt in OPTIMIZER_ORDER]
    ax.bar(
        x_pct,
        pct_means,
        width=bar_width,
        color=bar_colors,
        alpha=0.75,
        edgecolor="k",
    )
    ax.errorbar(
        x_pct,
        pct_means,
        yerr=pct_cis,
        fmt="none",
        ecolor="k",
        capsize=3,
        linewidth=1.0,
    )

    ax2 = ax.twinx()
    ax2.bar(
        x_rate,
        rate_means,
        width=bar_width,
        color=bar_colors,
        alpha=0.45,
        edgecolor="k",
        hatch="//",
    )
    ax2.errorbar(
        x_rate,
        rate_means,
        yerr=rate_cis,
        fmt="none",
        ecolor="k",
        capsize=3,
        linewidth=1.0,
    )

    ax.set_ylabel("Spontaneous PF periods")
    ax2.set_ylabel("Spontaneous rate per cell")
    ax.set_title("Spontaneous PF formations (%)", fontsize=10)
    ax.set_xticks(np.concatenate([x_pct, x_rate]))
    ax.set_xticklabels(
        [OPTIMIZER_LABELS[opt] for opt in OPTIMIZER_ORDER]
        + [OPTIMIZER_LABELS[opt] for opt in OPTIMIZER_ORDER]
    )
    ax.tick_params(axis="x", pad=6)
    for x_group, group_label in (
        (x_pct, "Spontaneous PF periods"),
        (x_rate, "Spontaneous rate per cell"),
    ):
        ax.text(
            float(np.mean(x_group)),
            -0.05,
            group_label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            clip_on=False,
        )
    ax.set_xlim(x_pct[0] - bar_width / 1.5, x_rate[-1] + bar_width / 1.5)
    ax2.set_xticks([])

    pct_tops = np.asarray(pct_means, dtype=float) + np.asarray(pct_cis, dtype=float)
    rate_tops = np.asarray(rate_means, dtype=float) + np.asarray(rate_cis, dtype=float)
    pct_top = float(np.nanmax(pct_tops)) if pct_tops.size else 0.0
    rate_top = float(np.nanmax(rate_tops)) if rate_tops.size else 0.0
    ax.set_ylim(0, 100)
    ax2.set_ylim(0, 100)

    if show_legend:
        ax.legend(
            handles=_optimizer_patch_legend_handles(),
            loc="upper center",
            ncol=1,
            frameon=True,
            framealpha=0.92,
            fontsize=8,
        )

    values_by_metric = {
        "one_segment_pct": {opt: period_values[opt] * 100.0 for opt in OPTIMIZER_ORDER},
        "one_segment_cell_rate": cell_rate_values,
    }
    metric_tags = {
        "one_segment_pct": "pct",
        "one_segment_cell_rate": "rate",
    }
    panel_label = "Spontaneous PF formations (%)"
    if not show_significance:
        _print_multi_metric_significance(
            panel_label=panel_label,
            values_by_metric=values_by_metric,
            metric_tags=metric_tags,
        )
    else:
        pct_positions = _optimizer_positions_at(x_pct)
        rate_positions = _optimizer_positions_at(x_rate)
        pct_pair_labels = _pairwise_bracket_labels(
            values_by_metric["one_segment_pct"],
            tag="pct",
            show_insignificance=show_insignificance,
            positions=pct_positions,
        )
        rate_pair_labels = _pairwise_bracket_labels(
            values_by_metric["one_segment_cell_rate"],
            tag="rate",
            show_insignificance=show_insignificance,
            positions=rate_positions,
        )
        _add_labeled_pairwise_brackets(
            ax,
            positions=pct_positions,
            bar_top=pct_top,
            pair_labels=pct_pair_labels,
            bracket_step_mult=bracket_step_mult,
        )
        _add_labeled_pairwise_brackets(
            ax2,
            positions=rate_positions,
            bar_top=rate_top,
            pair_labels=rate_pair_labels,
            bracket_step_mult=bracket_step_mult,
        )


def _plot_drift_bars(
    ax,
    results: MultiOptResults,
    *,
    show_significance: bool = True,
    show_insignificance: bool = False,
    bracket_step_mult: float = 1.65,
    bar_width: float = 0.18,
) -> None:
    drift_specs = [
        ("drift_x", "x", "///"),
        ("drift_y", "y", "..."),
        ("robustness", "mean", None),
    ]
    drift_labels = ["Drift x (cm)", "Drift y (cm)", "Mean drift (cm)"]
    positions = _optimizer_positions()
    x = np.arange(len(OPTIMIZER_ORDER))
    offsets = (-bar_width, 0.0, bar_width)

    values_by_metric: dict[str, dict[str, np.ndarray]] = {}
    metric_tags: dict[str, str] = {}
    bar_tops: list[float] = []

    for (col, tag, hatch), offset, ylabel in zip(
        drift_specs, offsets, drift_labels, strict=True
    ):
        values = _metric_values_by_optimizer(results, col)
        values_by_metric[col] = values
        metric_tags[col] = tag
        means, cis = [], []
        for opt in OPTIMIZER_ORDER:
            mean, _sem, ci = _mean_sem_ci95(values[opt])
            means.append(mean)
            cis.append(ci)
        means_arr = np.asarray(means, dtype=float)
        cis_arr = np.asarray(cis, dtype=float)
        bar_tops.extend((means_arr + cis_arr).tolist())
        bar_kwargs: dict[str, object] = {
            "width": bar_width,
            "color": [OPTIMIZER_COLORS[opt] for opt in OPTIMIZER_ORDER],
            "alpha": 0.75,
            "edgecolor": "k",
        }
        if hatch is not None:
            bar_kwargs["hatch"] = hatch
        ax.bar(x + offset, means_arr, **bar_kwargs)
        ax.errorbar(
            x + offset,
            means_arr,
            yerr=cis_arr,
            fmt="none",
            ecolor="k",
            capsize=3,
            linewidth=1.0,
        )

    ax.set_ylabel("Drift (cm)")
    ax.set_title("PF drift (x, y, mean)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([OPTIMIZER_LABELS[opt] for opt in OPTIMIZER_ORDER])
    data_top = float(np.nanmax(bar_tops)) if bar_tops else 0.0
    ax.set_ylim(0, data_top * 1.05 if np.isfinite(data_top) and data_top > 0 else 1)
    ax.legend(
        handles=[
            Patch(facecolor="0.92", edgecolor="k", hatch="///", label="Drift x"),
            Patch(facecolor="0.92", edgecolor="k", hatch="...", label="Drift y"),
            Patch(facecolor="0.92", edgecolor="k", label="Mean drift"),
        ],
        loc="upper left",
        frameon=False,
        fontsize=8,
    )

    panel_label = "PF drift (x, y, mean)"
    if not show_significance:
        _print_multi_metric_significance(
            panel_label=panel_label,
            values_by_metric=values_by_metric,
            metric_tags=metric_tags,
        )
    else:
        pair_stars = _multi_metric_pair_stars(values_by_metric, metric_tags)
        if show_insignificance and _mostly_fully_significant_multi_metric(
            pair_stars,
            tuple(metric_tags.values()),
        ):
            _print_multi_metric_significance(
                panel_label=panel_label,
                values_by_metric=values_by_metric,
                metric_tags=metric_tags,
            )
        pair_labels = _multi_metric_pair_labels(
            values_by_metric,
            metric_tags,
            show_insignificance=show_insignificance,
            positions=positions,
        )
        _add_labeled_pairwise_brackets(
            ax,
            positions=positions,
            bar_top=data_top,
            pair_labels=pair_labels,
            bracket_step_mult=bracket_step_mult,
        )


def plot_pf_metric_boxplots_by_optimizer(
    results: MultiOptResults,
    *,
    time_unit: str = TIME_UNIT_SECONDS,
    show_significance: bool = True,
    show_insignificance: bool = False,
) -> Figure:
    panels = pf_metric_boxplot_panels(time_unit)
    n_metrics = len(panels)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3.2 * n_metrics))
    if n_metrics == 1:
        axes = [axes]
    for ax, (col, ylabel) in zip(axes, panels, strict=True):
        _plot_pf_metric_panel_by_optimizer(
            ax,
            results,
            col,
            ylabel=ylabel,
            plot_kind="boxplot",
            show_significance=show_significance,
            show_insignificance=show_insignificance,
        )
    fig.suptitle("Place-field metrics by optimizer", y=1.01)
    fig.tight_layout()
    return fig


def plot_pf_metric_bars_by_optimizer(
    results: MultiOptResults,
    *,
    time_unit: str = TIME_UNIT_SECONDS,
    show_significance: bool = True,
    show_insignificance: bool = False,
    panel_specs: list[tuple[str, str | None, str | None]] | None = None,
    title: str | None = None,
    bar_width: float = 0.55,
    drift_bar_width: float = 0.18,
    drift_bracket_step_mult: float = 1.65,
    figsize: tuple[float, float] | None = None,
    hspace: float | None = None,
    shared_optimizer_legend: bool = False,
    shared_optimizer_legend_ncol: int = 5,
) -> Figure:
    if panel_specs is None:
        panel_specs = pf_metric_bar_panel_specs(time_unit)
    n_metrics = len(panel_specs)
    if figsize is None:
        figsize = (15.0, 3.5 * n_metrics)
    gridspec_kw = {"hspace": hspace} if hspace is not None else None
    fig, axes = plt.subplots(
        n_metrics,
        1,
        figsize=figsize,
        gridspec_kw=gridspec_kw,
    )
    if n_metrics == 1:
        axes = [axes]
    panel_show_legend = not shared_optimizer_legend
    for ax, (kind, metric_col, ylabel) in zip(axes, panel_specs, strict=True):
        if kind == "simple":
            _plot_pf_metric_panel_by_optimizer(
                ax,
                results,
                metric_col,
                ylabel=ylabel,
                plot_kind="bars",
                show_significance=show_significance,
                show_insignificance=show_insignificance,
                bracket_step_mult=1.65,
                bar_width=bar_width,
            )
        elif kind == "amp_tma":
            _plot_peak_amp_tma_bars(
                ax,
                results,
                time_unit=time_unit,
                show_significance=show_significance,
                show_insignificance=show_insignificance,
                bracket_step_mult=1.65,
                show_legend=panel_show_legend,
            )
        elif kind == "spontaneous":
            _plot_spontaneous_pf_bars(
                ax,
                results,
                show_significance=show_significance,
                show_insignificance=show_insignificance,
                bracket_step_mult=1.65,
                show_legend=panel_show_legend,
            )
        elif kind == "drift":
            _plot_drift_bars(
                ax,
                results,
                show_significance=show_significance,
                show_insignificance=show_insignificance,
                bracket_step_mult=drift_bracket_step_mult,
                bar_width=drift_bar_width,
            )
        else:
            raise ValueError(f"Unknown bar panel kind: {kind!r}")
    if title is None:
        title = (
            "Place-field metrics by optimizer"
        )
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    if hspace is not None:
        fig.subplots_adjust(hspace=hspace)
        fig._report_hspace = hspace  # noqa: SLF001 — reapplied after report restyle
    if shared_optimizer_legend:
        _add_shared_optimizer_legend_between_axes(
            fig,
            list(axes),
            ncol=shared_optimizer_legend_ncol,
        )
    return fig


def _group_metric_values_by_optimizer(
    results: MultiOptResults,
    *,
    variant: str,
    metric_col: str,
    group: str,
    groups_attr: str,
    optimizer_order: tuple[str, ...] | list[str] | None = None,
) -> dict[str, np.ndarray]:
    order = optimizer_order or OPTIMIZER_ORDER
    values: dict[str, np.ndarray] = {}
    for opt in order:
        analysis = results.eff_lr.get(opt)
        if analysis is None:
            values[opt] = np.array([], dtype=float)
            continue
        groups_dict = getattr(analysis, groups_attr)
        df = groups_dict.get(variant, pd.DataFrame())
        if df.empty:
            values[opt] = np.array([], dtype=float)
            continue
        vals = (
            df.loc[df["segment_group"] == group, metric_col]
            .dropna()
            .to_numpy(dtype=float)
        )
        values[opt] = vals
    return values


def _optimizer_order_for_analysis(
    *,
    groups_attr: str | None = None,
    counts_attr: str | None = None,
    probs_attr: str | None = None,
    optimizer_order: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if optimizer_order is not None:
        return optimizer_order
    if groups_attr == "pf_eff_lr_groups":
        return EFF_LR_OPTIMIZER_ORDER
    if counts_attr == "window_outlier_counts":
        return EFF_LR_OPTIMIZER_ORDER
    if probs_attr == "lifecycle_probs":
        return EFF_LR_OPTIMIZER_ORDER
    return OPTIMIZER_ORDER


def _random_window_outlier_values(counts_df: pd.DataFrame) -> np.ndarray:
    if counts_df.empty:
        return np.empty(0, dtype=np.float64)
    return counts_df.loc[
        counts_df["event_name"] == "random",
        "n_outliers",
    ].to_numpy(dtype=float)


def _plot_outlier_count_qq_overlay(
    ax,
    vals: np.ndarray,
    *,
    selected_fit_type: str,
    selected_fit: dict[str, object] | None,
    comparison_fit_types: tuple[str, ...],
    models: dict[str, dict[str, object]],
    color: str,
) -> None:
    all_fit_types = tuple(
        dict.fromkeys((*comparison_fit_types, str(selected_fit_type)))
    )
    grey_alphas = np.linspace(0.12, 0.32, max(len(all_fit_types) - 1, 1))

    ref_lo, ref_hi = np.inf, -np.inf
    for fit_type in all_fit_types:
        if str(fit_type) == str(selected_fit_type):
            fit = selected_fit
        else:
            fit = models.get(str(fit_type)) if isinstance(models, dict) else None
            if fit is None or "log_likelihood" not in fit:
                fit = fit_outlier_count_distribution(vals, fit_type=str(fit_type))
        if not isinstance(fit, dict) or "log_likelihood" not in fit:
            continue

        theo_q, emp_q = outlier_count_fit_qq_points(vals, str(fit_type), fit)
        if not theo_q.size:
            continue
        ref_lo = min(ref_lo, float(theo_q.min()), float(emp_q.min()))
        ref_hi = max(ref_hi, float(theo_q.max()), float(emp_q.max()))

        is_selected = str(fit_type) == str(selected_fit_type)
        if is_selected:
            ax.plot(
                theo_q,
                emp_q,
                "o",
                ms=5,
                color=color,
                alpha=0.9,
                zorder=3,
                label=OUTLIER_COUNT_FIT_MODEL_LABELS.get(str(fit_type), str(fit_type)),
            )
        else:
            grey_idx = max(
                0,
                [str(ft) for ft in all_fit_types if str(ft) != str(selected_fit_type)].index(
                    str(fit_type)
                ),
            )
            alpha = float(grey_alphas[min(grey_idx, len(grey_alphas) - 1)])
            ax.plot(
                theo_q,
                emp_q,
                "o",
                ms=2.5,
                color="0.45",
                alpha=alpha,
                zorder=2,
            )

    if np.isfinite(ref_lo) and np.isfinite(ref_hi):
        ax.plot([ref_lo, ref_hi], [ref_lo, ref_hi], "--", color="0.55", lw=0.9, zorder=1)
    ax.set_xlabel("Theoretical quantile")
    ax.set_ylabel("Empirical quantile")
    ax.grid(True, alpha=0.25)


def plot_outlier_count_fit_scatter_by_optimizer(
    results: MultiOptResults,
    *,
    fit_type_by_optimizer: dict[str, str],
    comparison_by_optimizer: dict[str, tuple[str, ...]] | None = None,
    counts_attr: str = "window_outlier_counts",
    fit_attr: str = "fit_results_random",
    optimizer_order: tuple[str, ...] | None = None,
    signal_label: str = "outlier count",
) -> Figure:
    comparison_by_optimizer = comparison_by_optimizer or {}
    optimizer_order = _optimizer_order_for_analysis(
        counts_attr=counts_attr,
        optimizer_order=optimizer_order,
    )
    n_opts = len(optimizer_order)
    fig, axes = plt.subplots(
        2,
        n_opts,
        figsize=(4.2 * n_opts, 7.8),
        sharex="col",
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.35},
    )
    if n_opts == 1:
        axes = np.array(axes).reshape(2, 1)

    for col, opt in enumerate(optimizer_order):
        ax_scatter = axes[0, col]
        ax_qq = axes[1, col]
        analysis = results.eff_lr.get(opt)
        color = OPTIMIZER_COLORS[opt]
        fit_type = str(fit_type_by_optimizer.get(opt, "negbinom_zi"))
        comparison_fit_types = tuple(comparison_by_optimizer.get(opt, ()))

        if analysis is None:
            ax_scatter.set_axis_off()
            ax_qq.set_axis_off()
            continue

        counts_df = getattr(analysis, counts_attr)
        fit_results = getattr(analysis, fit_attr)
        models = fit_results.get("models", {}) if isinstance(fit_results, dict) else {}
        fit = fit_results.get("random") if isinstance(fit_results, dict) else None
        if not isinstance(fit, dict) or not fit:
            fit = models.get(fit_type) if isinstance(models, dict) else None

        vals = _random_window_outlier_values(counts_df)
        x_data, y_data, x_fit, y_fit, fit = outlier_count_empirical_and_fit_curve(
            vals,
            fit_type=fit_type,
            fit=fit if isinstance(fit, dict) else None,
        )
        if x_data.size:
            ax_scatter.scatter(
                x_data,
                y_data,
                s=36,
                alpha=0.12,
                color=color,
                edgecolors="none",
                zorder=1,
            )
        if x_fit.size:
            ax_scatter.plot(x_fit, y_fit, "-", color=color, linewidth=2.0, alpha=0.95, zorder=2)
        mean_res, res_lo, res_hi = fit_outlier_count_mean_residual_ci95(
            vals,
            fit_type=fit_type,
        )
        annotation = format_outlier_count_fit_annotation(
            fit if isinstance(fit, dict) else None,
            fit_type=fit_type,
            mean_residual=mean_res,
            residual_lo=res_lo,
            residual_hi=res_hi,
        )
        ax_scatter.set_title(
            f"{OPTIMIZER_LABELS[opt]} ({OUTLIER_COUNT_FIT_MODEL_LABELS.get(fit_type, fit_type)})",
            fontsize=10,
        )
        ax_scatter.text(
            0.5,
            0.72,
            annotation,
            transform=ax_scatter.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax_scatter.set_xlabel("Outliers per window")
        ax_scatter.grid(True, axis="y", alpha=0.25)

        _plot_outlier_count_qq_overlay(
            ax_qq,
            vals,
            selected_fit_type=fit_type,
            selected_fit=fit if isinstance(fit, dict) else None,
            comparison_fit_types=comparison_fit_types,
            models=models if isinstance(models, dict) else {},
            color=color,
        )
        ax_qq.legend(frameon=False, fontsize=6, loc="lower right")

    axes[0, 0].set_ylabel("Count")
    fig.suptitle(f"Pooled random-window {signal_label} fits by optimizer", y=1.02)
    fig.tight_layout()
    return fig


def plot_pf_signal_group_by_optimizer(
    results: MultiOptResults,
    *,
    variant: str = "normalized_abs",
    groups_attr: str = "pf_eff_lr_groups",
    stat_prefix: str = "effective_lr",
    signal_label: str = "effective LR",
    group_order: tuple[str, ...] = EFF_LR_GROUP_ORDER,
    show_significance: bool = True,
    bracket_step_mult: float = 1.65,
    optimizer_order: tuple[str, ...] | None = None,
) -> Figure:
    optimizer_order = _optimizer_order_for_analysis(
        groups_attr=groups_attr,
        optimizer_order=optimizer_order,
    )
    metric_specs = (
        (f"{stat_prefix}_mean", "mean"),
        (f"{stat_prefix}_max", "max"),
        (f"{stat_prefix}_std", "std"),
    )
    n_groups = len(group_order)
    n_opts = len(optimizer_order)
    group_width = 0.75
    bar_width = group_width / max(n_groups, 1)
    offsets = np.linspace(
        -group_width / 2 + bar_width / 2,
        group_width / 2 - bar_width / 2,
        n_groups,
    )

    fig, axes = plt.subplots(2, len(metric_specs), figsize=(16, 7), sharex="col")
    if len(metric_specs) == 1:
        axes = np.array(axes).reshape(2, 1)

    x = np.arange(n_opts, dtype=float)
    for col, (metric_col, stat_label) in enumerate(metric_specs):
        ax_box = axes[0, col]
        ax_bar = axes[1, col]
        bar_tops_all: list[float] = []

        for gi, group in enumerate(group_order):
            offset = offsets[gi]
            hatch = EFF_LR_GROUP_HATCHES.get(group, "")
            box_data = []
            means, cis = [], []
            for opt in optimizer_order:
                vals = _group_metric_values_by_optimizer(
                    results,
                    variant=variant,
                    metric_col=metric_col,
                    group=group,
                    groups_attr=groups_attr,
                    optimizer_order=optimizer_order,
                )[opt]
                box_data.append(vals)
                mean, _sem, ci = _mean_sem_ci95(vals)
                means.append(mean)
                cis.append(ci)
            positions = x + offset
            bp = ax_box.boxplot(
                box_data,
                positions=positions,
                widths=bar_width * 0.9,
                patch_artist=True,
                showfliers=True,
                flierprops={"marker": "o", "markersize": 2, "alpha": 0.35},
            )
            for patch, opt in zip(bp["boxes"], optimizer_order, strict=True):
                patch.set_facecolor(OPTIMIZER_COLORS[opt])
                patch.set_alpha(0.55)
                patch.set_hatch(hatch)
                patch.set_edgecolor("k")

            means_arr = np.asarray(means, dtype=float)
            cis_arr = np.asarray(cis, dtype=float)
            for opt_i, opt in enumerate(optimizer_order):
                ax_bar.bar(
                    positions[opt_i],
                    means_arr[opt_i],
                    width=bar_width * 0.9,
                    color=OPTIMIZER_COLORS[opt],
                    alpha=0.75,
                    edgecolor="k",
                    hatch=hatch,
                )
            ax_bar.errorbar(
                positions,
                means_arr,
                yerr=cis_arr,
                fmt="none",
                ecolor="k",
                capsize=3,
                linewidth=1.0,
            )
            bar_tops_all.extend((means_arr + cis_arr).tolist())

        if show_significance:
            values_by_group = {
                group: _group_metric_values_by_optimizer(
                    results,
                    variant=variant,
                    metric_col=metric_col,
                    group=group,
                    groups_attr=groups_attr,
                    optimizer_order=optimizer_order,
                )
                for group in group_order
            }
            group_tags = {
                group: EFF_LR_GROUP_SHORT_TAGS[group] for group in group_order
            }
            pair_labels = _multi_metric_pair_labels(values_by_group, group_tags)
            bar_top = float(np.nanmax(bar_tops_all)) if bar_tops_all else 0.0
            _add_labeled_pairwise_brackets(
                ax_bar,
                positions=_optimizer_positions_at(x, optimizer_order),
                bar_top=bar_top,
                pair_labels=pair_labels,
                bracket_step_mult=bracket_step_mult,
            )

        ax_box.set_ylabel(f"{stat_label} {signal_label}")
        ax_box.set_title(f"{stat_label}", fontsize=10)
        ax_box.set_xticks(x)
        ax_box.set_xticklabels([OPTIMIZER_LABELS[o] for o in optimizer_order], rotation=15)
        ax_bar.set_ylabel(f"{stat_label} (mean ± 1.96 SEM)")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([OPTIMIZER_LABELS[o] for o in optimizer_order], rotation=15)

    handles = [
        Patch(
            facecolor="0.92",
            edgecolor="k",
            hatch=EFF_LR_GROUP_HATCHES[g],
            label=EFF_LR_GROUP_LABELS[g],
        )
        for g in group_order
    ]
    axes[0, -1].legend(handles=handles, loc="upper right", fontsize=7, frameon=False)
    fig.suptitle(
        f"|{signal_label}| / |network| (excl. unit) by PF segment group and optimizer",
        y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_pf_eff_lr_eligibility_stacked_by_optimizer(
    results: MultiOptResults,
) -> Figure:
    n_opts = len(OPTIMIZER_ORDER)
    x = np.arange(n_opts, dtype=float)
    ineligible = np.zeros(n_opts, dtype=float)
    birth_only = np.zeros(n_opts, dtype=float)
    revive = np.zeros(n_opts, dtype=float)

    for i, opt in enumerate(OPTIMIZER_ORDER):
        analysis = results.eff_lr.get(opt)
        if analysis is None:
            continue
        df = analysis.eligibility_df
        n_total = len(df)
        n_birth = int(df["eligible_birth"].sum())
        n_revive = int((df["eligible_birth"] & df["eligible_revive"]).sum())
        ineligible[i] = n_total - n_birth
        birth_only[i] = n_birth - n_revive
        revive[i] = n_revive

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x, ineligible, width=0.6, color="lightgray", edgecolor="k", label="ineligible")
    ax.bar(
        x,
        birth_only,
        width=0.6,
        bottom=ineligible,
        color="#98df8a",
        edgecolor="k",
        label="birth only",
    )
    ax.bar(
        x,
        revive,
        width=0.6,
        bottom=ineligible + birth_only,
        color="tab:green",
        edgecolor="k",
        label="revive-eligible",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([OPTIMIZER_LABELS[o] for o in OPTIMIZER_ORDER])
    ax.set_ylabel("Place fields (count)")
    ax.set_title("PF eligibility for lifecycle outlier analysis")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


def _load_recurrent_unit_eff_lr_window(
    signals_dir: Path,
    *,
    segment_indices: np.ndarray,
    unit_indices: np.ndarray,
    optimizer_config: dict,
) -> np.ndarray:
    """Recurrent-layer mean effective LR, shape ``(n_segments, n_units)``."""
    refs = list(_iter_training_segment_files(signals_dir))
    n_seg = len(segment_indices)
    n_units = len(unit_indices)
    out = np.full((n_seg, n_units), np.nan, dtype=np.float64)

    for row, gidx in enumerate(segment_indices):
        seg_ref = refs[int(gidx)]
        with np.load(seg_ref.path, allow_pickle=True) as data:
            opt_signals = _load_opt_signals_dict(data["opt_signals"])
            eff_by_node = _effective_lr_by_node_from_opt_signals(
                opt_signals,
                optimizer_config=optimizer_config,
            )
        for col, unit_idx in enumerate(unit_indices):
            val = eff_by_node.get(f"recurrent[{int(unit_idx)}]")
            if val is not None:
                out[row, col] = _node_signal_to_unit_scalar(val)

    return out


def plot_recurrent_unit_eff_lr_midtraining_zoom_by_optimizer(
    results: MultiOptResults,
    *,
    optimizer: str = "pure_shampoo",
    n_segments: int = 10,
    n_cells: int = 5,
    cell_rng_seed: int = 42,
    segment_duration_s: float | None = None,
) -> Figure:
    """Zoom panel: per-unit recurrent effective LR over a mid-training window."""
    if segment_duration_s is None:
        segment_duration_s = next(iter(results.contexts.values())).segment_duration_s

    ctx = results.contexts[optimizer]
    n_neurons = ctx.n_neurons
    rng = np.random.default_rng(cell_rng_seed)
    unit_indices = rng.choice(n_neurons, size=n_cells, replace=False)

    n_total_segments = len(list(_iter_training_segment_files(ctx.paths.signals_dir)))
    start = max(0, n_total_segments // 2 - n_segments // 2)
    segment_indices = np.arange(start, start + n_segments, dtype=int)
    x = index_to_time_s(segment_indices, segment_duration_s)

    optimizer_config = build_optimizer_config(optimizer)
    eff_lr = _load_recurrent_unit_eff_lr_window(
        ctx.paths.signals_dir,
        segment_indices=segment_indices,
        unit_indices=unit_indices,
        optimizer_config=optimizer_config,
    )

    tab10_colors = plt.cm.tab10_r.colors
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for trace_idx, unit_idx in enumerate(unit_indices):
        color = tab10_colors[trace_idx % len(tab10_colors)]
        ax.plot(
            x,
            eff_lr[:, trace_idx],
            "-o",
            color=color,
            linewidth=1.4,
            markersize=4,
        )
    ax.set_xlabel(TIME_AXIS_LABEL)
    ax.set_ylabel(r"$\tilde{\gamma}_{t,j}$")

    legend_handles = [
        Line2D(
            [],
            [],
            color=tab10_colors[i % len(tab10_colors)],
            marker="o",
            linestyle="-",
            linewidth=1.4,
            markersize=4,
            label=f"neuron {int(unit_idx)}",
        )
        for i, unit_idx in enumerate(unit_indices)
    ]
    fig.suptitle(
        f"{OPTIMIZER_LABELS[optimizer]}",
        y=1.04,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=n_cells//2,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def _lifecycle_prob_column_prefix(probability_kind: str) -> str:
    return {
        "pmf": "prob",
        "cdf": "cdf",
        "epmf": "eprob",
        "ecdf": "ecdf",
    }[probability_kind]


def plot_lifecycle_outlier_summary_by_optimizer(
    results: MultiOptResults,
    *,
    fit_type: str | None = None,
    probability_kinds: tuple[str, ...] = ("pmf", "cdf"),
    probs_attr: str = "lifecycle_probs",
    show_significance: bool = True,
    bracket_step_mult: float = 1.35,
    optimizer_order: tuple[str, ...] | None = None,
) -> Figure:
    optimizer_order = _optimizer_order_for_analysis(
        probs_attr=probs_attr,
        optimizer_order=optimizer_order,
    )
    kind_titles = {
        "pmf": "PDF",
        "cdf": "CDF",
        "epmf": "ePDF",
        "ecdf": "eCDF",
    }
    n_cols = len(probability_kinds)
    fig, axes = plt.subplots(2, n_cols, figsize=(6.5 * n_cols, 7), sharex="col")
    if n_cols == 1:
        axes = np.array(axes).reshape(2, 1)

    n_opts = len(optimizer_order)
    n_specs = len(LIFECYCLE_PROB_SPECS)
    group_width = 0.72
    bar_width = group_width / n_specs
    offsets = np.linspace(
        -group_width / 2 + bar_width / 2,
        group_width / 2 - bar_width / 2,
        n_specs,
    )
    x = np.arange(n_opts, dtype=float)

    for col, prob_kind in enumerate(probability_kinds):
        col_prefix = _lifecycle_prob_column_prefix(prob_kind)
        ax_box = axes[0, col]
        ax_bar = axes[1, col]
        bar_tops: list[float] = []
        values_by_spec: dict[str, dict[str, np.ndarray]] = {}
        spec_tags: dict[str, str] = {}

        for si, (event_key, baseline_key, short_label) in enumerate(LIFECYCLE_PROB_SPECS):
            offset = offsets[si]
            col_name = f"{col_prefix}_{event_key}_{baseline_key}_random"
            box_data = []
            means, cis = [], []
            for opt in optimizer_order:
                analysis = results.eff_lr.get(opt)
                if analysis is None:
                    vals = np.array([], dtype=float)
                else:
                    probs_df = getattr(analysis, probs_attr)
                    vals = (
                        probs_df[col_name].dropna().to_numpy(dtype=float)
                        if col_name in probs_df.columns
                        else np.array([], dtype=float)
                    )
                box_data.append(vals)
                mean, _sem, ci = _mean_sem_ci95(vals)
                means.append(mean)
                cis.append(ci)

            positions = x + offset
            is_revive = event_key == "revive"
            is_post = baseline_key == "post"
            for opt_i, (opt, vals) in enumerate(zip(optimizer_order, box_data, strict=True)):
                alpha_box = 0.25 if is_revive else 0.45
                bp = ax_box.boxplot(
                    [vals],
                    positions=[positions[opt_i]],
                    widths=bar_width * 0.85,
                    patch_artist=True,
                    showfliers=True,
                    flierprops={"marker": "o", "markersize": 2, "alpha": 0.35},
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(OPTIMIZER_COLORS[opt])
                    patch.set_alpha(alpha_box)
                    patch.set_edgecolor("k")
                    if is_post:
                        patch.set_hatch("//")

            means_arr = np.asarray(means, dtype=float)
            cis_arr = np.asarray(cis, dtype=float)
            for opt_i, opt in enumerate(optimizer_order):
                ax_bar.bar(
                    positions[opt_i],
                    means_arr[opt_i],
                    width=bar_width * 0.85,
                    color=OPTIMIZER_COLORS[opt],
                    alpha=0.55 if is_revive else 0.85,
                    edgecolor="k",
                    hatch="//" if is_post else None,
                )
            ax_bar.errorbar(
                positions,
                means_arr,
                yerr=cis_arr,
                fmt="none",
                ecolor="k",
                capsize=2,
                linewidth=1.0,
            )
            bar_tops.extend((means_arr + cis_arr).tolist())
            values_by_spec[col_name] = {
                opt: box_data[i] for i, opt in enumerate(optimizer_order)
            }
            spec_tags[col_name] = short_label

        if show_significance:
            pair_labels = _multi_metric_pair_labels(values_by_spec, spec_tags)
            bar_top = float(np.nanmax(bar_tops)) if bar_tops else 0.0
            _add_labeled_pairwise_brackets(
                ax_bar,
                positions=_optimizer_positions_at(x, optimizer_order),
                bar_top=bar_top,
                pair_labels=pair_labels,
                bracket_step_mult=bracket_step_mult,
            )

        ax_box.set_title(kind_titles.get(prob_kind, prob_kind))
        y_suffix = f" ({fit_type})" if fit_type else ""
        ax_box.set_ylabel(f"{kind_titles.get(prob_kind, prob_kind)}{y_suffix}")
        ax_box.set_xticks(x)
        ax_box.set_xticklabels([OPTIMIZER_LABELS[o] for o in optimizer_order], rotation=15)
        ax_bar.set_ylabel("Mean ± 1.96 SEM")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([OPTIMIZER_LABELS[o] for o in optimizer_order], rotation=15)

    spec_legend_handles = [
        Patch(
            facecolor="0.92",
            edgecolor="k",
            alpha=0.55 if event_key == "revive" else 0.85,
            hatch="//" if baseline_key == "post" else None,
            label=label,
        )
        for (event_key, baseline_key, _short_label), label in zip(
            LIFECYCLE_PROB_SPECS,
            LIFECYCLE_PROB_LEGEND_LABELS,
            strict=True,
        )
    ]
    fig.legend(
        handles=spec_legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=len(LIFECYCLE_PROB_SPECS),
        frameon=False,
        fontsize=8,
    )
    fig.suptitle("Lifecycle outlier probabilities by optimizer", y=1.02)
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    return fig
