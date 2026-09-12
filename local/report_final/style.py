"""Report styling for the final thesis figures.

Every constant is exposed through a function rather than a module-level
binding so that `%autoreload 2` picks up edits without a kernel restart, and
so callers can override individual entries at the call site.

The restyle layer is deliberately non-destructive: it takes a `Figure` that an
existing plotting helper already produced and rewrites its typography, grid and
legend placement. Nothing in `single_room_multi_opt_plots` or
`two_rooms_multi_opt_plots` needs to change, so the older notebooks keep
rendering exactly as before.

X tick labels: do not rotate category axes in report figures except the Vaidya
scorecard heatmap (`bio.plot_scorecard`, finalized with ``restyle=False``).
Everywhere else, keep ticks horizontal and abbreviate the Shampoo optimizers as
``Pure\\nShampoo`` / ``Grafted\\nShampoo`` via `report_optimizer_xtick_labels()`.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


def report_font() -> dict[str, float]:
    """Typography for report figures, in points.

    Deliberately larger than the notebook defaults: these figures are shrunk
    to roughly half a text width in the thesis, so 7 pt ticks become illegible.
    """
    return {
        "tick": 13,
        "axis_label": 15,
        "panel_title": 14,
        "row_title": 15,
        "suptitle": 18,
        "legend": 12,
        "legend_title": 13,
        "annotation": 12,
        "sig": 14,
        "colorbar_label": 14,
        "colorbar_tick": 12,
    }


def report_rc() -> dict[str, Any]:
    """rcParams applied by `apply_report_style`."""
    font = report_font()
    return {
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,  # embed TrueType so text stays selectable/editable
        "ps.fonttype": 42,
        "axes.titlesize": font["panel_title"],
        "axes.labelsize": font["axis_label"],
        "xtick.labelsize": font["tick"],
        "ytick.labelsize": font["tick"],
        "legend.fontsize": font["legend"],
        "legend.title_fontsize": font["legend_title"],
        "figure.titlesize": font["suptitle"],
        "axes.axisbelow": True,
        "axes.grid": False,
        "grid.color": "0.85",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    }


def report_grid() -> dict[str, Any]:
    """Styling for the horizontal reference grid drawn behind data."""
    return {"color": "0.85", "linewidth": 0.6, "alpha": 0.9}


def apply_report_style(*, font_scale: float = 1.0) -> None:
    """Set the seaborn theme and rcParams used by every report figure."""
    import seaborn as sns

    sns.set_theme(context="paper", style="whitegrid", font_scale=font_scale)
    rc = report_rc()
    if font_scale != 1.0:
        rc = {
            key: (value * font_scale if _is_size_key(key) else value)
            for key, value in rc.items()
        }
    mpl.rcParams.update(rc)


def _is_size_key(key: str) -> bool:
    return key.endswith(("size", "labelsize")) and not key.startswith(("figure.dpi",))


def report_optimizer_order() -> tuple[str, ...]:
    from single_room_multi_opt_compute import OPTIMIZER_ORDER

    return tuple(OPTIMIZER_ORDER)


def report_optimizer_colors() -> dict[str, str]:
    from single_room_multi_opt_compute import OPTIMIZER_COLORS

    return dict(OPTIMIZER_COLORS)


def report_optimizer_labels() -> dict[str, str]:
    from single_room_multi_opt_compute import OPTIMIZER_LABELS

    return dict(OPTIMIZER_LABELS)


def report_optimizer_xtick_labels() -> dict[str, str]:
    """Short horizontal x tick labels; Shampoo variants split across two lines."""
    labels = report_optimizer_labels()
    return {
        key: value.replace(" Shampoo", "\nShampoo") if " Shampoo" in value else value
        for key, value in labels.items()
    }


def format_optimizer_xtick_label(text: str) -> str:
    """Map a rendered tick string to the report x tick convention."""
    if not text:
        return text
    if "\n" in text:
        return text
    return text.replace(" Shampoo", "\nShampoo")


def report_extra_colors() -> list[str]:
    """Non-optimizer accent colors, shared with the other thesis repo."""
    return ["#97b4ed", "#f1c3b1", "#fffe93", "#edb1ff", "#27f16e"]


def report_reference_style() -> dict[str, Any]:
    """Line style for experimental (Vaidya) reference curves drawn over models."""
    return {
        "color": "0.15",
        "linewidth": 2.4,
        "linestyle": (0, (5, 2)),
        "marker": "D",
        "markersize": 6,
        "markerfacecolor": "white",
        "markeredgewidth": 1.6,
        "zorder": 10,
        "label": "BTSP (CA1, Vaidya et al.)",
    }


def restyle_figure(
    fig: Figure,
    *,
    font: Mapping[str, float] | None = None,
    grid: bool | str = "y",
    legend_loc: str | None = None,
    legend_bbox: tuple[float, ...] | None = None,
    legend_ncol: int | None = None,
    drop_panel_titles: bool = False,
    suptitle: str | None = None,
    drop_suptitle: bool = False,
    tighten: bool = True,
    rect: tuple[float, float, float, float] | None = None,
    despine: bool = True,
    auto_rotate_xticks: bool = False,
    xtick_rotation: float = 20.0,
    max_axes: int = 200,
) -> Figure:
    """Rewrite typography, grid and legends on an already-built figure.

    `grid` is `"y"` (horizontal only), `"both"`, `"x"`, or False. Passing
    `legend_loc="outside"` moves every axis legend to the right of its axes,
    which is the usual fix when a five-optimizer legend covers the data.

    Category x ticks stay horizontal. The only report figure with rotated
    optimizer ticks is the Vaidya scorecard heatmap, which sets rotation in
    `plot_scorecard` and is finalized with ``restyle=False``.

    Figures with more than `max_axes` panels (the ratemap mosaics, which have
    one axis per neuron) skip the per-axis pass; only figure-level text is
    touched, since those panels carry no ticks or labels anyway.
    """
    f = {**report_font(), **(font or {})}
    grid_kw = report_grid()

    if len(fig.axes) <= max_axes:
        for ax in fig.axes:
            _restyle_axis(
                ax,
                font=f,
                grid=grid,
                grid_kw=grid_kw,
                legend_loc=legend_loc,
                legend_bbox=legend_bbox,
                legend_ncol=legend_ncol,
                drop_panel_titles=drop_panel_titles,
                despine=despine,
            )
    else:
        for ax in fig.axes:
            if getattr(ax, "_colorbar", None) is not None:
                ax.tick_params(labelsize=f["colorbar_tick"])
                ax.yaxis.label.set_fontsize(f["colorbar_label"])

    for text in fig.texts:
        # Column/row banners added with `fig.text` by the source helpers.
        if fig._suptitle is not None and text is fig._suptitle:
            continue
        text.set_fontsize(f["row_title"])

    if drop_suptitle:
        if fig._suptitle is not None:
            fig._suptitle.set_visible(False)
    elif suptitle is not None:
        fig.suptitle(suptitle, fontsize=f["suptitle"], y=1.0, va="bottom")
    elif fig._suptitle is not None:
        fig._suptitle.set_fontsize(f["suptitle"])

    legend = getattr(fig, "legends", None)
    if legend:
        for leg in fig.legends:
            _restyle_legend(leg, f)

    if tighten:
        try:
            fig.tight_layout(rect=rect)
        except (ValueError, RuntimeError):
            pass
        report_hspace = getattr(fig, "_report_hspace", None)
        if report_hspace is not None:
            fig.subplots_adjust(hspace=report_hspace)
    if len(fig.axes) <= max_axes:
        apply_report_optimizer_xtick_labels(fig)
        _refresh_two_rooms_boundary_ticks(fig)
    if auto_rotate_xticks and len(fig.axes) <= max_axes:
        rotate_colliding_xticks(fig, rotation=xtick_rotation)
    return fig


def _refresh_two_rooms_boundary_ticks(fig: Figure) -> None:
    """Keep two-rooms boundary minor ticks after layout/restyle."""
    try:
        from two_rooms_multi_opt_plots import _refresh_two_rooms_boundary_ticks as refresh
    except ImportError:
        return
    for ax in fig.axes:
        if getattr(ax, "_colorbar", None) is not None:
            continue
        refresh(ax)


def apply_report_optimizer_xtick_labels(fig: Figure) -> None:
    """Horizontal optimizer ticks with two-line Shampoo labels."""
    optimizer_labels = set(report_optimizer_labels().values())
    for ax in fig.axes:
        if getattr(ax, "_colorbar", None) is not None:
            continue
        tick_labels = [t for t in ax.get_xticklabels() if t.get_text()]
        if not tick_labels:
            continue
        texts = [t.get_text() for t in tick_labels]
        if not any(text in optimizer_labels or "Shampoo" in text for text in texts):
            continue
        ax.set_xticklabels(
            [format_optimizer_xtick_label(text) for text in texts],
            rotation=0,
            ha="center",
        )


def rotate_colliding_xticks(
    fig: Figure, *, rotation: float = 20.0, ha: str = "right", pad_px: float = 1.0
) -> None:
    """Rotate x tick labels only on axes where they actually overlap.

    Enlarging tick fonts is what makes report figures readable, but it is also
    what makes crowded category axes collide. Measuring the rendered label boxes
    and rotating only the offending axes avoids tilting labels that were fine.
    """
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        return
    for ax in fig.axes:
        labels = [t for t in ax.get_xticklabels() if t.get_text()]
        if len(labels) < 2 or any(t.get_rotation() % 360 != 0 for t in labels):
            continue
        try:
            boxes = sorted(
                (t.get_window_extent(renderer=renderer) for t in labels),
                key=lambda b: b.x0,
            )
        except Exception:
            continue
        if any(boxes[i].x1 > boxes[i + 1].x0 + pad_px for i in range(len(boxes) - 1)):
            for t in labels:
                t.set_rotation(rotation)
                t.set_ha(ha)
                t.set_rotation_mode("anchor")


def _restyle_axis(
    ax,
    *,
    font: Mapping[str, float],
    grid: bool | str,
    grid_kw: Mapping[str, Any],
    legend_loc: str | None,
    legend_bbox: tuple[float, ...] | None,
    legend_ncol: int | None,
    drop_panel_titles: bool,
    despine: bool,
) -> None:
    is_colorbar = getattr(ax, "_colorbar", None) is not None
    is_polar = ax.name == "polar"

    if is_colorbar:
        ax.tick_params(labelsize=font["colorbar_tick"])
        ax.yaxis.label.set_fontsize(font["colorbar_label"])
        ax.xaxis.label.set_fontsize(font["colorbar_label"])
        return

    ax.tick_params(axis="both", which="major", labelsize=font["tick"])
    ax.xaxis.label.set_fontsize(font["axis_label"])
    ax.yaxis.label.set_fontsize(font["axis_label"])

    if drop_panel_titles:
        ax.set_title("")
    elif ax.get_title():
        ax.title.set_fontsize(font["panel_title"])

    # Significance stars / inline annotations added by the source helpers.
    for txt in ax.texts:
        current = txt.get_fontsize()
        txt.set_fontsize(max(current, font["annotation"]))

    if not is_polar:
        ax.set_axisbelow(True)
        if grid in (True, "both", "y"):
            ax.yaxis.grid(True, which="major", **grid_kw)
        else:
            ax.yaxis.grid(False)
        if grid in (True, "both", "x"):
            ax.xaxis.grid(True, which="major", **grid_kw)
        else:
            ax.xaxis.grid(False)

        if despine:
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

    leg = ax.get_legend()
    if leg is None:
        return
    if legend_loc == "outside":
        handles, labels = ax.get_legend_handles_labels()
        leg.remove()
        leg = ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=legend_bbox or (1.01, 1.0),
            ncol=legend_ncol or 1,
            frameon=False,
            borderaxespad=0.0,
            handlelength=1.6,
            columnspacing=1.0,
        )
    elif legend_loc is not None:
        handles, labels = ax.get_legend_handles_labels()
        leg.remove()
        leg = ax.legend(
            handles,
            labels,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox,
            ncol=legend_ncol or 1,
            frameon=False,
        )
    elif legend_ncol is not None:
        handles, labels = ax.get_legend_handles_labels()
        leg.remove()
        leg = ax.legend(handles, labels, ncol=legend_ncol, frameon=False)
    _restyle_legend(leg, font)


def _restyle_legend(leg, font: Mapping[str, float]) -> None:
    for text in leg.get_texts():
        text.set_fontsize(font["legend"])
    title = leg.get_title()
    if title is not None and title.get_text():
        title.set_fontsize(font["legend_title"])


def shared_optimizer_legend(
    fig: Figure,
    *,
    optimizers: Iterable[str] | None = None,
    ncol: int | None = None,
    y: float = -0.01,
) -> None:
    """Replace per-axis optimizer legends with one centred figure legend."""
    from matplotlib.lines import Line2D

    order = tuple(optimizers) if optimizers is not None else report_optimizer_order()
    colors = report_optimizer_colors()
    labels = report_optimizer_labels()
    handles = [
        Line2D([], [], color=colors[opt], linewidth=2.6, label=labels[opt])
        for opt in order
    ]
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol or len(handles),
        frameon=False,
        fontsize=report_font()["legend"],
        columnspacing=1.6,
        handlelength=1.8,
    )


def data_yticks_with_headroom(
    data_lo: float, data_hi: float, *, nbins: int = 6
) -> list[float]:
    """Y ticks spanning the data range only, so bracket headroom stays unlabelled."""
    if not np.isfinite(data_hi) or not np.isfinite(data_lo) or data_hi <= data_lo:
        return [data_lo, data_hi] if data_hi > data_lo else [0.0, 1.0]
    locator = MaxNLocator(nbins=nbins, min_n_ticks=3)
    raw = [float(t) for t in locator.tick_values(data_lo, data_hi) if np.isfinite(t)]
    in_range = sorted({t for t in raw if data_lo - 1e-9 <= t <= data_hi + 1e-9})
    if len(in_range) >= 2:
        return in_range
    return [float(data_lo), float(data_hi)]


def set_ylim_with_brackets(
    ax, data_lo: float, data_hi: float, bracket_top: float, *, nbins: int = 6
) -> None:
    """Reserve vertical headroom for significance brackets without wasting space."""
    bracket_top = max(bracket_top, data_hi)
    ax.set_ylim(data_lo, bracket_top)
    yticks = data_yticks_with_headroom(data_lo, data_hi, nbins=nbins)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:g}" for t in yticks])
    ax.tick_params(axis="y", which="major", labelsize=report_font()["tick"])
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which="major", **report_grid())
    ax.xaxis.grid(False)


def save_report_figure(fig: Figure, plot_dir: Path | str, name: str) -> Path:
    """Write `{plot_dir}/{name}.pdf` without triggering a second inline render."""
    path = (Path(plot_dir) / f"{name}.pdf").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight")
    path.write_bytes(buf.getvalue())
    return path


def save_report_table(df, plot_dir: Path | str, name: str, *, verbose: bool = True) -> Path:
    """Write the numbers behind a figure next to it, for quoting in the text."""
    path = (Path(plot_dir) / f"{name}.csv").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if verbose:
        print(f"Saved {path}")
    return path


def finalize(
    fig: Figure,
    plot_dir: Path | str,
    name: str,
    *,
    save: bool = True,
    show: bool = True,
    restyle: bool = True,
    verbose: bool = True,
    **restyle_kwargs: Any,
) -> Figure:
    """Restyle, save as PDF, display once, and close.

    Closing after an explicit `display` is what stops the inline backend from
    rendering the same figure a second time at the end of the cell.
    """
    if restyle:
        restyle_figure(fig, **restyle_kwargs)
    if save:
        path = save_report_figure(fig, plot_dir, name)
        if verbose:
            print(f"Saved {path}")
    if show:
        from IPython.display import display

        display(fig)
    plt.close(fig)
    return fig
