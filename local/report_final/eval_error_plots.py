"""Bar charts of checkpoint eval error by optimizer."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from report_final.eval_errors import EXPERIMENT_EVAL_ERRORS, EXPERIMENT_LABELS
from report_final.style import (
    report_font,
    report_grid,
    report_optimizer_colors,
    report_optimizer_labels,
    report_optimizer_order,
    report_optimizer_xtick_labels,
)


def _style_axis(ax, *, xlabel: str, ylabel: str, title: str | None = None) -> None:
    f = report_font()
    ax.set_xlabel(xlabel, fontsize=f["axis_label"])
    ax.set_ylabel(ylabel, fontsize=f["axis_label"])
    if title:
        ax.set_title(title, fontsize=f["panel_title"])
    ax.tick_params(labelsize=f["tick"])
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which="major", **report_grid())
    ax.xaxis.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def eval_error_table(experiment: str) -> pd.DataFrame:
    """Long-form table of eval errors for one experiment."""
    if experiment not in EXPERIMENT_EVAL_ERRORS:
        known = ", ".join(sorted(EXPERIMENT_EVAL_ERRORS))
        raise KeyError(f"Unknown experiment {experiment!r}; known: {known}")
    order = report_optimizer_order()
    errors = EXPERIMENT_EVAL_ERRORS[experiment]
    rows = []
    for opt in order:
        if opt not in errors:
            raise KeyError(f"Missing eval error for optimizer {opt!r} in {experiment!r}")
        rows.append(
            {
                "experiment": experiment,
                "optimizer": opt,
                "optimizer_label": report_optimizer_labels()[opt],
                "eval_error": float(errors[opt]),
            }
        )
    return pd.DataFrame(rows)


def plot_eval_error_by_optimizer(
    experiment: str,
    *,
    figsize: tuple[float, float] = (8.5, 5.0),
    annotate: bool = True,
) -> Figure:
    """One bar per optimizer: checkpoint eval error for a single experiment."""
    table = eval_error_table(experiment)
    colors = report_optimizer_colors()
    labels = report_optimizer_xtick_labels()
    f = report_font()

    opts = table["optimizer"].tolist()
    values = table["eval_error"].to_numpy(dtype=float)
    x = np.arange(len(opts), dtype=float)
    with plt.rc_context({"mathtext.fontset": "dejavusans"}):
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(
            x,
            values,
            width=0.85,
            color=[colors[o] for o in opts],
            edgecolor="k",
            linewidth=0.8,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([labels[o] for o in opts])
        y_hi = float(np.max(values))
        y_lo = 0.0
        headroom = max(0.04 * y_hi, 0.02)
        ax.set_ylim(y_lo, y_hi + headroom)
        _style_axis(
            ax,
            xlabel="",
            ylabel="Eval error",
            title=EXPERIMENT_LABELS.get(experiment, experiment),
        )
        if annotate:
            for xi, val in zip(x, values, strict=True):
                ax.annotate(
                    f"{val:.3f}",
                    (xi, val),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=f["annotation"],
                )
        fig.tight_layout()
    return fig


def list_eval_error_experiments() -> tuple[str, ...]:
    return tuple(EXPERIMENT_EVAL_ERRORS)
