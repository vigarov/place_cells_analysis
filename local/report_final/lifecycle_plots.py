"""Report figures for the PF-lifecycle effective-LR / delta-W analysis.

The extensive notebook's ``plot_pf_signal_group_by_optimizer`` renders three
statistics per signal (mean / max / std). Only ``mean`` is comparable across
lifecycle groups: ``birth`` and ``peak`` are single-capture windows, so their
``max`` collapses onto their ``mean`` and their ``std`` is undefined. These
helpers therefore keep the per-window mean only, report medians (the
distributions are heavy-tailed), and test each group against ``pre_vicinity``
*within* each optimizer using a paired signed-rank test over place fields.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

from single_room_multi_opt_compute import (
    EFF_LR_OPTIMIZER_ORDER,
    OPTIMIZER_COLORS,
    OPTIMIZER_ORDER,
    _p_value_to_stars,
    holm_adjusted_pvalues,
)
from single_room_multi_opt_plots import (
    EFF_LR_GROUP_HATCHES,
    EFF_LR_GROUP_LABELS,
    EFF_LR_GROUP_ORDER,
)

from report_final.style import report_font, report_optimizer_xtick_labels

BASELINE_GROUP = "pre_vicinity"


def _add_pair_brackets(
    ax,
    *,
    pair_specs: list[tuple[float, float, str]],
    y_base: float,
    fontsize: float,
    bracket_step_mult: float = 1.5,
) -> None:
    """Draw stacked comparison brackets (same geometry as other report figures)."""
    if not pair_specs:
        return
    y_base = float(y_base) if np.isfinite(y_base) else 0.0
    y_step = max(y_base * 0.08, 0.05)
    bracket_height = y_step * 0.4
    required_top = y_base + y_step + len(pair_specs) * y_step * bracket_step_mult
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, max(y1, required_top * 1.05))
    y_level = y_base + y_step
    for x1, x2, label in pair_specs:
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
            fontsize=fontsize,
            fontweight="bold",
        )
        y_level += y_step * bracket_step_mult


def _boxplot_top(vals: np.ndarray, *, log_y: bool) -> float:
    if vals.size == 0:
        return np.nan
    if log_y:
        return float(np.max(vals))
    q25, q75 = np.percentile(vals, [25, 75])
    inside = vals[vals <= q75 + 1.5 * (q75 - q25)]
    return float(inside.max()) if inside.size else float(q75)


SIGNAL_SPECS: dict[str, dict[str, Any]] = {
    "effective_lr": {
        "groups_attr": "pf_eff_lr_groups",
        "metric_col": "effective_lr_mean",
        "label": r"normalized $\tilde{\gamma}_{t,j}$",
        "optimizer_order": tuple(EFF_LR_OPTIMIZER_ORDER),
        "log_y": False,
    },
    "delta_w": {
        "groups_attr": "pf_delta_w_groups",
        "metric_col": "delta_w_mean",
        "label": r"normalized $|\Delta W|$",
        "optimizer_order": tuple(OPTIMIZER_ORDER),
        "log_y": True,
    },
}


def _group_frame(results, signal: str, *, variant: str = "normalized_abs") -> pd.DataFrame:
    """Wide frame, one row per place field, one column per lifecycle group."""
    spec = SIGNAL_SPECS[signal]
    frames = []
    for opt in spec["optimizer_order"]:
        analysis = results.eff_lr.get(opt)
        if analysis is None:
            continue
        df = getattr(analysis, spec["groups_attr"]).get(variant, pd.DataFrame())
        if df.empty:
            continue
        wide = df.pivot_table(
            index=["cell_idx", "pf_idx"],
            columns="segment_group",
            values=spec["metric_col"],
        )
        frames.append(wide.assign(optimizer=opt).reset_index())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def lifecycle_group_table(
    results,
    *,
    signals: tuple[str, ...] = ("effective_lr", "delta_w"),
    variant: str = "normalized_abs",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-(signal, optimizer, group) medians plus a paired test vs pre-vicinity.

    ``p_vs_pre_holm`` is a Wilcoxon signed-rank test over the place fields with
    a finite value in *both* the group and ``pre_vicinity``, Holm-corrected
    across all (optimizer, group) comparisons of that signal.
    """
    rows: list[dict[str, Any]] = []
    for signal in signals:
        spec = SIGNAL_SPECS[signal]
        wide = _group_frame(results, signal, variant=variant)
        if wide.empty:
            continue
        raw: list[float] = []
        for opt in spec["optimizer_order"]:
            sub = wide.loc[wide["optimizer"] == opt]
            base = sub[BASELINE_GROUP].to_numpy(dtype=float)
            base_med = float(np.nanmedian(base)) if np.isfinite(base).any() else np.nan
            for group in EFF_LR_GROUP_ORDER:
                vals = sub[group].to_numpy(dtype=float)
                finite = vals[np.isfinite(vals)]
                paired = np.isfinite(vals) & np.isfinite(base)
                p = np.nan
                if group != BASELINE_GROUP and paired.sum() >= 10:
                    if np.any(vals[paired] != base[paired]):
                        p = float(stats.wilcoxon(vals[paired], base[paired]).pvalue)
                raw.append(p)
                med = float(np.median(finite)) if finite.size else np.nan
                rows.append(
                    {
                        "signal": signal,
                        "optimizer": opt,
                        "group": group,
                        "n_pfs": int(finite.size),
                        "n_paired": int(paired.sum()),
                        "median": med,
                        "q25": float(np.percentile(finite, 25)) if finite.size else np.nan,
                        "q75": float(np.percentile(finite, 75)) if finite.size else np.nan,
                        "mean": float(finite.mean()) if finite.size else np.nan,
                        "median_ratio_to_pre": med / base_med if base_med else np.nan,
                    }
                )
        raw_arr = np.asarray(raw, dtype=float)
        mask = np.isfinite(raw_arr)
        adj = np.full(raw_arr.shape, np.nan)
        if mask.any():
            adj[mask] = holm_adjusted_pvalues(raw_arr[mask])
        for row, p_raw, p_adj in zip(rows[-raw_arr.size :], raw_arr, adj, strict=True):
            row["p_vs_pre_raw"] = p_raw
            row["p_vs_pre_holm"] = p_adj
            row["stars_vs_pre"] = (
                _p_value_to_stars(p_adj) if np.isfinite(p_adj) and p_adj < alpha else "ns"
            )
    return pd.DataFrame(rows)


def plot_lifecycle_group_signal(
    results,
    *,
    signal: str = "delta_w",
    variant: str = "normalized_abs",
    table: pd.DataFrame | None = None,
    figsize: tuple[float, float] = (7.4, 4.6),
    show_fliers: bool = False,
    show_unity_line: bool = True,
    mark_insignificant: bool = True,
    show_legend: bool = True,
    legend_ncol: int = 3,
    ylabel: str | None = None,
    title: str = "",
) -> Figure:
    """Boxplots of one per-window signal, grouped by optimizer and lifecycle stage.

    Every group differs from that optimizer's own ``pre_vicinity`` window at
    ``p < 0.001`` (paired signed-rank, Holm-corrected) except where marked
    ``ns``; :func:`lifecycle_group_table` holds the full test output.
    """
    spec = SIGNAL_SPECS[signal]
    opt_order = spec["optimizer_order"]
    log_y = bool(spec["log_y"])
    if table is None:
        table = lifecycle_group_table(results, signals=(signal,), variant=variant)

    wide = _group_frame(results, signal, variant=variant)
    font = report_font()
    xtick_labels = report_optimizer_xtick_labels()

    n_groups = len(EFF_LR_GROUP_ORDER)
    group_width = 0.78
    box_width = group_width / n_groups
    offsets = np.linspace(
        -group_width / 2 + box_width / 2, group_width / 2 - box_width / 2, n_groups
    )
    x = np.arange(len(opt_order), dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    if log_y:
        ax.set_yscale("log")

    pre_gi = EFF_LR_GROUP_ORDER.index(BASELINE_GROUP)
    ns_brackets: list[tuple[float, float, str]] = []
    box_tops: list[float] = []

    for gi, group in enumerate(EFF_LR_GROUP_ORDER):
        hatch = EFF_LR_GROUP_HATCHES.get(group, "")
        for oi, opt in enumerate(opt_order):
            vals = wide.loc[wide["optimizer"] == opt, group].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if log_y:
                vals = vals[vals > 0]
            if vals.size == 0:
                continue
            pos = x[oi] + offsets[gi]
            bp = ax.boxplot(
                [vals],
                positions=[pos],
                widths=box_width * 0.86,
                patch_artist=True,
                showfliers=show_fliers,
                flierprops={"marker": "o", "markersize": 1.5, "alpha": 0.25},
                medianprops={"color": "k", "linewidth": 1.3},
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(OPTIMIZER_COLORS[opt])
                patch.set_alpha(0.7)
                patch.set_edgecolor("k")
                patch.set_hatch(hatch)

            top = _boxplot_top(vals, log_y=log_y)
            if np.isfinite(top):
                box_tops.append(top)

            if mark_insignificant and group != BASELINE_GROUP:
                row = table[
                    (table["signal"] == signal)
                    & (table["optimizer"] == opt)
                    & (table["group"] == group)
                ]
                if not row.empty and str(row["stars_vs_pre"].iloc[0]) == "ns":
                    x_pre = x[oi] + offsets[pre_gi]
                    ns_brackets.append((float(x_pre), float(pos), r"$ns$"))

    if ns_brackets:
        y_base = float(np.nanmax(box_tops)) if box_tops else 0.0
        _add_pair_brackets(
            ax,
            pair_specs=ns_brackets,
            y_base=y_base,
            fontsize=font["sig"],
        )

    if show_unity_line:
        ax.axhline(1.0, color="k", linestyle=":", linewidth=1.3, zorder=1)
        if not log_y:
            y0, y1 = ax.get_ylim()
            ax.set_ylim(y0, max(y1, 1.06))

    ax.set_xticks(x)
    ax.set_xticklabels(
        [xtick_labels.get(opt, opt) for opt in opt_order], fontsize=font["tick"]
    )
    ax.set_ylabel(ylabel or spec["label"], fontsize=font["axis_label"])
    if title:
        ax.set_title(title, fontsize=font["panel_title"])

    if show_legend:
        handles: list[Any] = [
            Patch(
                facecolor="0.85",
                edgecolor="k",
                hatch=EFF_LR_GROUP_HATCHES.get(g, ""),
                label=EFF_LR_GROUP_LABELS[g],
            )
            for g in EFF_LR_GROUP_ORDER
        ]
        if show_unity_line:
            handles.append(
                Line2D(
                    [], [], color="k", linestyle=":", linewidth=1.3, label="network average"
                )
            )
        ax.legend(
            handles=handles,
            fontsize=font["legend"],
            ncol=legend_ncol,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            frameon=False,
            columnspacing=1.4,
            handlelength=1.8,
        )

    fig.tight_layout()
    return fig


LIFECYCLE_ANCHOR_LABELS = {
    "birth_pre": "birth, pre",
    "birth_post": "birth, post",
    "revive_pre": "revive, pre",
    "revive_post": "revive, post",
}

_PROBS_ATTR = {"effective_lr": "lifecycle_probs", "delta_w": "dw_lifecycle_probs"}


def lifecycle_outlier_table(
    results,
    *,
    signals: tuple[str, ...] = ("effective_lr", "delta_w"),
    kind: str = "cdf",
) -> pd.DataFrame:
    """Mean per-PF outlier-count probability at each lifecycle anchor."""
    prefix = "cdf" if kind == "cdf" else "prob"
    rows = []
    for signal in signals:
        for opt in SIGNAL_SPECS[signal]["optimizer_order"]:
            analysis = results.eff_lr.get(opt)
            if analysis is None:
                continue
            df = getattr(analysis, _PROBS_ATTR[signal])
            for anchor in LIFECYCLE_ANCHOR_LABELS:
                event, baseline = anchor.split("_")
                col = f"{prefix}_{event}_{baseline}_random"
                if col not in df.columns:
                    continue
                vals = df[col].dropna().to_numpy(dtype=float)
                if vals.size == 0:
                    continue
                rows.append(
                    {
                        "signal": signal,
                        "optimizer": opt,
                        "anchor": anchor,
                        "kind": kind,
                        "n": int(vals.size),
                        "mean": float(vals.mean()),
                        "median": float(np.median(vals)),
                    }
                )
    return pd.DataFrame(rows)


def plot_lifecycle_outlier_cdf(
    results,
    *,
    signals: tuple[str, ...] = ("delta_w", "effective_lr"),
    anchors: tuple[str, ...] = ("birth_pre", "birth_post", "revive_pre", "revive_post"),
    figsize: tuple[float, float] | None = None,
    table: pd.DataFrame | None = None,
) -> Figure:
    """One panel per signal: P(X <= observed outlier count) under random windows.

    A discrete induction event would push the birth anchors towards 1; a value
    near 0.5 means the window is indistinguishable from a randomly placed one.
    """
    if table is None:
        table = lifecycle_outlier_table(results, signals=signals, kind="cdf")
    font = report_font()
    xtick_labels = report_optimizer_xtick_labels()
    hatches = ("", "///", "xx", "++")

    n = len(signals)
    fig, axes = plt.subplots(1, n, figsize=figsize or (6.9 * n, 4.5), squeeze=False)
    for col, signal in enumerate(signals):
        ax = axes[0, col]
        opt_order = SIGNAL_SPECS[signal]["optimizer_order"]
        x = np.arange(len(opt_order), dtype=float)
        width = 0.78 / len(anchors)
        offsets = np.linspace(-0.39 + width / 2, 0.39 - width / 2, len(anchors))
        for ai, anchor in enumerate(anchors):
            vals = []
            for opt in opt_order:
                row = table[
                    (table["signal"] == signal)
                    & (table["optimizer"] == opt)
                    & (table["anchor"] == anchor)
                ]
                vals.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
            ax.bar(
                x + offsets[ai],
                vals,
                width=width * 0.9,
                color=[OPTIMIZER_COLORS[o] for o in opt_order],
                alpha=0.85,
                edgecolor="k",
                hatch=hatches[ai % len(hatches)],
            )
        ax.axhline(0.5, color="k", linestyle=":", linewidth=1.3, zorder=1)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [xtick_labels.get(o, o) for o in opt_order], fontsize=font["tick"]
        )
        ax.set_ylabel(
            r"$P(N_{OL} \leq N_{OL}^{(event)})$",
            fontsize=font["axis_label"],
        )
        ax.set_title(
            r"$|\Delta W|$" if signal == "delta_w" else r"$\tilde{\gamma}_{t,j}$",
            fontsize=font["panel_title"],
        )

    handles = [
        Patch(
            facecolor="0.85",
            edgecolor="k",
            hatch=hatches[ai % len(hatches)],
            label=LIFECYCLE_ANCHOR_LABELS[a],
        )
        for ai, a in enumerate(anchors)
    ]
    handles.append(
        Line2D([], [], color="k", linestyle=":", linewidth=1.3, label="median random window")
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.legend(
        handles=handles,
        fontsize=font["legend"],
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        columnspacing=1.4,
        handlelength=1.8,
    )
    return fig
