"""Model-vs-CA1 comparison: statistics, Vaidya overlays, and the scorecard.

Everything here consumes a `TwoRoomsMultiOptResults` and the frozen Vaidya
targets. The plots are new rather than restyled, because each one overlays an
experimental reference curve that the original helpers know nothing about.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.figure import Figure

from report_final import vaidya_targets as vt
from report_final.style import (
    report_extra_colors,
    report_font,
    report_optimizer_colors,
    report_optimizer_labels,
    report_optimizer_order,
    report_optimizer_xtick_labels,
    report_reference_style,
)

PoolMode = Literal["counts", "mean"]


def _optimizers(results, optimizers: Iterable[str] | None) -> tuple[str, ...]:
    if optimizers is not None:
        return tuple(optimizers)
    available = (
        set(results["optimizer"].unique())
        if isinstance(results, pd.DataFrame)
        else set(results.pf)
    )
    return tuple(o for o in report_optimizer_order() if o in available)


# --------------------------------------------------------------------------
# Model-side statistics
# --------------------------------------------------------------------------


def model_repetition_activity(results, *, optimizers=None) -> pd.DataFrame:
    """% of neurons holding a field on exactly N repetitions (rooms averaged)."""
    rows = []
    for opt in _optimizers(results, optimizers):
        df = results.pf[opt].repetition_activity
        agg = df.groupby("n_days_with_pf")["pct_neurons"].mean()
        for n_days, pct in agg.items():
            rows.append({"optimizer": opt, "n_days_with_pf": int(n_days), "pct_neurons": float(pct)})
    return pd.DataFrame(rows)


def model_always_active_fraction(results, *, optimizers=None) -> pd.Series:
    """Share of ever-active cells that hold a field on every repetition."""
    act = model_repetition_activity(results, optimizers=optimizers)
    wide = act.pivot(index="optimizer", columns="n_days_with_pf", values="pct_neurons")
    n_max = int(wide.columns.max())
    return (wide[n_max] / (100.0 - wide[0])).rename("always_active_fraction")


def model_formation_probability(
    results, *, optimizers=None, pool: PoolMode = "counts"
) -> pd.DataFrame:
    """P(field present on repetition d | j prior repetitions with a field).

    `pool="counts"` sums successes and opportunities across rooms, which is the
    correct estimator when the two rooms contribute unequal sample sizes;
    `pool="mean"` averages the two per-room probabilities.
    """
    rows = []
    for opt in _optimizers(results, optimizers):
        df = results.pf[opt].formation_probability
        if pool == "counts":
            g = df.groupby("n_prior_days_with_pf")[["n_active", "n_opportunities"]].sum()
            prob = g["n_active"] / g["n_opportunities"].replace(0, np.nan)
        else:
            prob = df.groupby("n_prior_days_with_pf")["formation_probability"].mean()
        for j, p in prob.items():
            rows.append(
                {"optimizer": opt, "n_prior_days_with_pf": int(j), "formation_probability": float(p)}
            )
    return pd.DataFrame(rows)


def model_pc_pools(results, *, optimizers=None) -> pd.DataFrame:
    """Total / sustained / newly-recruited place cells per repetition, rooms pooled."""
    rows = []
    for opt in _optimizers(results, optimizers):
        pf = results.pf[opt]
        stab = pf.pc_stability_by_repetition.groupby("repetition")[
            ["n_total_pcs", "n_sustained_pcs", "n_transient_pcs"]
        ].sum()
        ret = pf.pc_cohort_retention
        new = (
            ret[ret["birth_cohort"] == ret["recording_day"]]
            .groupby("recording_day")["count"]
            .sum()
        )
        for rep, r in stab.iterrows():
            total = float(r["n_total_pcs"])
            rows.append(
                {
                    "optimizer": opt,
                    "repetition": int(rep),
                    "n_total_pcs": total,
                    "n_sustained_pcs": float(r["n_sustained_pcs"]),
                    "n_new_pcs": float(new.get(int(rep), np.nan)),
                    "sustained_fraction": float(r["n_sustained_pcs"]) / total if total else np.nan,
                    "new_fraction": float(new.get(int(rep), np.nan)) / total if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def model_onset_anchors(
    results, *, optimizers=None, laps: tuple[int, ...] = (1, 5, 10, 20)
) -> pd.DataFrame:
    """Onset CDF sampled at the block fractions matching Vaidya's laps."""
    seconds = vt.onset_lap_seconds(laps)
    rows = []
    for opt in _optimizers(results, optimizers):
        df = results.pf[opt].pf_onset_cdf_samples
        for (cohort, label), grp in df.groupby(["cohort", "label"]):
            v = np.sort(grp["onset_s"].to_numpy(dtype=float))
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            row = {"optimizer": opt, "cohort": cohort, "label": label, "n": int(v.size)}
            for name, cutoff in seconds.items():
                row[name] = float((v <= cutoff).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def model_shift_density(
    results,
    *,
    optimizers=None,
    room_extent_cm: float = 100.0,
    kind: str = "cross_rep",
) -> pd.DataFrame:
    """Cross-repetition PF drift, binned to match Vaidya Fig 2d exactly.

    Vaidya measures a signed shift along a 1D track; our rooms are 2D, so the
    comparable quantity is the signed shift along a single axis. Pooling dx and
    dy doubles the sample and averages over any axis asymmetry, while the L2
    displacement would not be comparable at all (it is a non-negative
    magnitude in two dimensions).

    Both axes are put on a common footing by expressing the shift as a fraction
    of the environment's linear extent and binning at Vaidya's *relative*
    resolution (3.6 cm on a 183.6 cm track, i.e. 1.96% of it). The density is
    renormalised over the plotted window, as in the published figure, so the
    two curves integrate to the same total.
    """
    ref = vt.pf_shift_density_normalized()
    bw = ref["bin_width"]
    half = float(np.abs(ref["centers"]).max()) + bw / 2

    rows = []
    for opt in _optimizers(results, optimizers):
        shifts = _shift_fractions(results, opt, room_extent_cm=room_extent_cm, kind=kind)
        shifts = shifts[np.abs(shifts) <= half]
        if shifts.size == 0:
            continue
        centers = ref["centers"]
        edges = np.append(centers - bw / 2, centers[-1] + bw / 2)
        counts, _ = np.histogram(shifts, bins=edges)
        density = counts / (counts.sum() * bw)
        for c, d in zip(centers, density):
            rows.append(
                {"optimizer": opt, "center": float(c), "density": float(d), "n": int(shifts.size)}
            )
    return pd.DataFrame(rows)


SHIFT_BAND_EDGES = (0.0, 0.02, 0.10, 0.50)
SHIFT_BAND_LABELS = (
    "Same place\n(<2% of extent)",
    "Graded shift\n(2-10%)",
    "Relocated\n(>10%)",
)
# Zoom window matches ``plot_cell_displacement_by_optimizer_two_rooms`` Δx/Δy panels.
SHIFT_DISPLACEMENT_XLIM = (-8.0, 8.0)


def model_shift_density_spatial_only(
    results,
    *,
    optimizers=None,
    room_extent_cm: float = 100.0,
    kind: str = "cross_rep",
    pool_xy: bool = True,
    bin_width: float = 0.25,
    bin_range: tuple[float, float] = SHIFT_DISPLACEMENT_XLIM,
) -> pd.DataFrame:
    """Cross-repetition shift histogram on the model room cm axis.

    By default pools ``displacement_x`` and ``displacement_y`` (two samples per
    reappearance). Uses the same bin grid as
    ``plot_cell_displacement_by_optimizer_two_rooms`` and reports probability
    density (``count / (n × bin_width)``) to match Vaidya Fig 2d units.
    """
    lo, hi = bin_range
    edges = np.arange(lo, hi + bin_width, bin_width)
    centers = edges[:-1] + bin_width / 2.0
    rows = []
    for opt in _optimizers(results, optimizers):
        shifts = _shift_cm(results, opt, kind=kind, pool_xy=pool_xy)
        if shifts.size == 0:
            continue
        counts, _ = np.histogram(shifts, bins=edges)
        density = counts / (shifts.size * bin_width)
        for c, d in zip(centers, density):
            rows.append(
                {
                    "optimizer": opt,
                    "center": float(c),
                    "density": float(d),
                    "n": int(shifts.size),
                }
            )
    return pd.DataFrame(rows)


def model_shift_bands_spatial_only(
    results,
    *,
    optimizers=None,
    room_extent_cm: float = 100.0,
    edges_cm: tuple[float, ...] | None = None,
    kind: str = "cross_rep",
    pool_xy: bool = True,
) -> pd.DataFrame:
    """Three-band shift composition on the model room cm axis."""
    if edges_cm is None:
        edges_cm = vt.shift_band_edges_cm(room_extent_cm)
    rows = []
    for opt in _optimizers(results, optimizers):
        shifts = np.abs(_shift_cm(results, opt, kind=kind, pool_xy=pool_xy))
        if shifts.size == 0:
            continue
        n = int(shifts.size)
        for lo, hi in zip(edges_cm[:-1], edges_cm[1:]):
            if lo > 0:
                count = int(((shifts > lo) & (shifts <= hi)).sum())
            else:
                count = int((shifts <= hi).sum())
            rows.append(
                {
                    "source": opt,
                    "band_lo": lo,
                    "band_hi": hi,
                    "fraction": count / n,
                    "n": n,
                }
            )
    for (lo, hi), frac in zip(
        zip(edges_cm[:-1], edges_cm[1:]),
        vt.pf_shift_band_fractions_on_model_cm(room_extent_cm, edges_cm),
    ):
        rows.append(
            {
                "source": "vaidya",
                "band_lo": lo,
                "band_hi": hi,
                "fraction": float(frac),
                "n": np.nan,
            }
        )
    return pd.DataFrame(rows)


def model_shift_return_fraction(
    results,
    *,
    optimizers=None,
    room_extent_cm: float = 100.0,
    window_frac: float = 0.05,
    kind: str = "cross_rep",
) -> pd.Series:
    """Share of reappearing fields landing within `window_frac` of their old centre."""
    out = {}
    for opt in _optimizers(results, optimizers):
        shifts = _shift_fractions(results, opt, room_extent_cm=room_extent_cm, kind=kind)
        if shifts.size:
            out[opt] = float((np.abs(shifts) <= window_frac).mean())
    return pd.Series(out, name="shift_return_fraction")


def model_shift_bands(
    results,
    *,
    optimizers=None,
    room_extent_cm: float = 100.0,
    edges: tuple[float, ...] = SHIFT_BAND_EDGES,
    kind: str = "cross_rep",
) -> pd.DataFrame:
    """Composition of reappearance shifts across three bands, model and CA1.

    This is the statistic the Fig 2d overlay actually argues: CA1 carries a
    substantial *intermediate* population of fields that come back a little
    displaced, and that is the part a model with no discrete formation event
    has no reason to produce.
    """
    rows = []
    for opt in _optimizers(results, optimizers):
        shifts = np.abs(_shift_fractions(results, opt, room_extent_cm=room_extent_cm, kind=kind))
        shifts = shifts[shifts <= edges[-1]]
        if shifts.size == 0:
            continue
        for lo, hi in zip(edges[:-1], edges[1:]):
            frac = float(((shifts > lo) & (shifts <= hi)).mean()) if lo > 0 else float(
                (shifts <= hi).mean()
            )
            rows.append(
                {"source": opt, "band_lo": lo, "band_hi": hi, "fraction": frac, "n": int(shifts.size)}
            )
    for (lo, hi), frac in zip(zip(edges[:-1], edges[1:]), vt.pf_shift_band_fractions(edges)):
        rows.append(
            {"source": "vaidya", "band_lo": lo, "band_hi": hi, "fraction": float(frac), "n": np.nan}
        )
    return pd.DataFrame(rows)


def model_shift_gradedness(
    results,
    *,
    optimizers=None,
    room_extent_cm: float = 100.0,
    edges: tuple[float, ...] = SHIFT_BAND_EDGES,
    kind: str = "cross_rep",
) -> pd.Series:
    """Share of reappearances in the intermediate band, i.e. the graded population."""
    bands = model_shift_bands(
        results, optimizers=optimizers, room_extent_cm=room_extent_cm, edges=edges, kind=kind
    )
    mid = bands[(bands["band_lo"] == edges[1]) & (bands["source"] != "vaidya")]
    return mid.set_index("source")["fraction"].rename("shift_gradedness")


def _shift_cm(
    results,
    opt: str,
    *,
    kind: str,
    pool_xy: bool = True,
    column: str = "displacement_x",
) -> np.ndarray:
    """Signed single-axis shifts for one optimizer, in model room cm."""
    if isinstance(results, pd.DataFrame):
        df = results[(results["kind"] == kind) & (results["optimizer"] == opt)]
    else:
        df = results.pf[opt].cell_displacement
        df = df[df["kind"] == kind]
    if pool_xy:
        shifts = np.concatenate(
            [df["displacement_x"].to_numpy(float), df["displacement_y"].to_numpy(float)]
        )
    else:
        shifts = df[column].to_numpy(dtype=float)
    return shifts[np.isfinite(shifts)]


def _shift_fractions(results, opt: str, *, room_extent_cm: float, kind: str) -> np.ndarray:
    """Signed single-axis shifts for one optimizer, as a fraction of room extent.

    Accepts either a `TwoRoomsMultiOptResults` or a tidy displacement frame
    carrying an `optimizer` column, so the figure can be rebuilt from the
    cached CSV without re-running the 20-minute pipeline.
    """
    if isinstance(results, pd.DataFrame):
        df = results[(results["kind"] == kind) & (results["optimizer"] == opt)]
    else:
        df = results.pf[opt].cell_displacement
        df = df[df["kind"] == kind]
    shifts = np.concatenate(
        [df["displacement_x"].to_numpy(float), df["displacement_y"].to_numpy(float)]
    )
    return shifts[np.isfinite(shifts)] / room_extent_cm


# --------------------------------------------------------------------------
# Scorecard
# --------------------------------------------------------------------------


def scorecard_statistic_specs() -> list[dict[str, Any]]:
    """Definition of every statistic compared against CA1.

    `family` splits the across-day (consolidation) statistics from the
    within-session (BTSP timing) ones; `shared_failure` marks statistics where
    every optimizer misses by so much that the normalised score is meaningless.
    """
    return [
        {
            "key": "formation_probability",
            "label": "P(form | prior history)",
            "family": "across_day",
            "shared_failure": False,
        },
        {
            "key": "sustained_fraction",
            "label": "Sustained / total PCs",
            "family": "across_day",
            "shared_failure": False,
        },
        {
            "key": "always_active",
            "label": "Always-active share",
            "family": "across_day",
            "shared_failure": False,
        },
        {
            "key": "birth_discriminability",
            "label": "Birth-day sustained/transient gap",
            "family": "within_session",
            "shared_failure": False,
        },
        {
            "key": "established_onset",
            "label": "Established-field onset",
            "family": "within_session",
            "shared_failure": False,
        },
        {
            "key": "onset_gradedness",
            "label": "Onset gradedness",
            "family": "within_session",
            "shared_failure": True,
        },
        {
            "key": "new_pc_fraction",
            "label": "Daily PC recruitment",
            "family": "across_day",
            "shared_failure": True,
        },
        {
            "key": "shift_gradedness",
            "label": "Reappearance shift gradedness",
            "family": "spatial",
            "shared_failure": False,
        },
    ]


CONSOLIDATION_SPATIAL_SCORECARD_STATISTICS = (
    "formation_probability",
    "sustained_fraction",
    "always_active",
    "new_pc_fraction",
    "shift_gradedness",
)


def subset_scorecard(scorecard: pd.DataFrame, statistics: Iterable[str]) -> pd.DataFrame:
    """Return rows for ``statistics`` only, in the given order."""
    order = list(statistics)
    out = scorecard[scorecard["statistic"].isin(order)].copy()
    out["statistic"] = pd.Categorical(out["statistic"], categories=order, ordered=True)
    return out.sort_values("statistic")


def build_scorecard(
    results,
    *,
    optimizers=None,
    pool: PoolMode = "counts",
    n_prior: int = 4,
    model_reps: tuple[int, ...] = (1, 2, 3),
    vaidya_days: tuple[int, ...] = (2, 3, 4),
    room_extent_cm: float = 100.0,
) -> pd.DataFrame:
    """Tidy model-vs-CA1 table: one row per (statistic, optimizer).

    Model repetitions 1-3 are aligned to Vaidya recording days 2-4 because our
    `sustained` flag is backward-looking (it needs a preceding repetition),
    so repetition 0 is zero by construction.
    """
    opts = _optimizers(results, optimizers)

    fp = model_formation_probability(results, optimizers=opts, pool=pool)
    fp_wide = fp.pivot(index="optimizer", columns="n_prior_days_with_pf", values="formation_probability")
    pools = model_pc_pools(results, optimizers=opts)
    always = model_always_active_fraction(results, optimizers=opts)
    anchors = model_onset_anchors(results, optimizers=opts).set_index(["cohort", "label", "optimizer"])

    t_formation = vt.formation_probability(n_prior)
    t_sustained = vt.sustained_fraction(vaidya_days)
    t_new = vt.new_pc_fraction(vaidya_days)
    t_gap = vt.birth_day_discriminability(lap=1)
    t_onset = vt.established_onset(lap=1)
    t_rise = vt.established_onset_rise(1, 20)
    t_always = float(vt.days_with_pf()["data"][-1] / (1.0 - vt.days_with_pf()["data"][0]))
    t_shift = float(vt.pf_shift_band_fractions(SHIFT_BAND_EDGES)[1])
    shift = model_shift_gradedness(
        results, optimizers=opts, room_extent_cm=room_extent_cm
    )

    def mad(a, b) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        return float(np.nanmean(np.abs(a - b)))

    rows = []
    for opt in opts:
        sub = pools[pools["optimizer"] == opt].set_index("repetition")
        m_sustained = sub.loc[list(model_reps), "sustained_fraction"].to_numpy()
        m_new = sub.loc[list(model_reps), "new_fraction"].to_numpy()
        m_formation = fp_wide.loc[opt].to_numpy()[:n_prior]

        gap = float(
            anchors.loc[("birth_day", "sustained", opt), "lap1"]
            - anchors.loc[("birth_day", "transient", opt), "lap1"]
        )
        onset = float(anchors.loc[("subsequent_days", "sustained", opt), "lap1"])
        rise = float(anchors.loc[("subsequent_days", "sustained", opt), "lap20"]) - onset

        rows += [
            _row(opt, "formation_probability", np.nan, np.nan, mad(m_formation, t_formation)),
            _row(opt, "sustained_fraction", np.nan, np.nan, mad(m_sustained, t_sustained)),
            _row(opt, "always_active", float(always[opt]), t_always, abs(float(always[opt]) - t_always)),
            _row(opt, "birth_discriminability", gap, t_gap, abs(gap - t_gap)),
            _row(opt, "established_onset", onset, t_onset, abs(onset - t_onset)),
            _row(opt, "onset_gradedness", rise, t_rise, abs(rise - t_rise)),
            _row(opt, "new_pc_fraction", np.nan, np.nan, mad(m_new, t_new)),
        ]
        if opt in shift.index:
            rows.append(
                _row(
                    opt,
                    "shift_gradedness",
                    float(shift[opt]),
                    t_shift,
                    abs(float(shift[opt]) - t_shift),
                )
            )

    df = pd.DataFrame(rows)
    meta = pd.DataFrame(scorecard_statistic_specs()).rename(columns={"key": "statistic"})
    df = df.merge(meta, on="statistic", how="left")

    df["normalized_deviation"] = df.groupby("statistic")["abs_deviation"].transform(_minmax)
    return df


def _row(optimizer, statistic, model_value, target_value, abs_deviation) -> dict[str, Any]:
    return {
        "optimizer": optimizer,
        "statistic": statistic,
        "model_value": model_value,
        "target_value": target_value,
        "abs_deviation": abs_deviation,
    }


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = float(s.min()), float(s.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def scorecard_family_means(scorecard: pd.DataFrame, *, drop_shared_failures: bool = True) -> pd.DataFrame:
    """Mean normalised deviation per optimizer, split by statistic family."""
    df = scorecard
    if drop_shared_failures:
        df = df[~df["shared_failure"]]
    out = (
        df.pivot_table(
            index="optimizer", columns="family", values="normalized_deviation", aggfunc="mean"
        )
        .reindex(report_optimizer_order())
        .dropna(how="all")
    )
    out["composite"] = out.mean(axis=1)
    return out


# --------------------------------------------------------------------------
# Overlay figures
# --------------------------------------------------------------------------


def _style_axis(ax, *, xlabel, ylabel, title=None):
    f = report_font()
    ax.set_xlabel(xlabel, fontsize=f["axis_label"])
    ax.set_ylabel(ylabel, fontsize=f["axis_label"])
    if title:
        ax.set_title(title, fontsize=f["panel_title"])
    ax.tick_params(labelsize=f["tick"])
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="0.85", linewidth=0.6)
    ax.xaxis.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


class _PartialColorText:
    """Horizontally adjacent text segments in distinct colors (see SO 9169052)."""

    def __init__(self, fig: Figure, ax) -> None:
        self._fig = fig
        self._ax = ax

    def _segment_width(self, text: str, transform, **kw) -> float:
        temp = self._ax.text(0, 0, text, transform=transform, **kw)
        temp.draw(self._fig.canvas.get_renderer())
        width = transform.inverted().transform_bbox(temp.get_window_extent()).width
        temp.remove()
        return width

    def horizontal(
        self,
        x: float,
        y: float,
        segments: Iterable[tuple[str, str]],
        *,
        transform=None,
        ha: str = "left",
        **kw,
    ) -> None:
        if transform is None:
            transform = self._ax.transData
        kw = dict(kw)
        kw.setdefault("ha", "left")
        kw.setdefault("va", "bottom")
        parts = list(segments)
        if ha == "center":
            x -= sum(self._segment_width(text, transform, **kw) for text, _ in parts) / 2.0
        elif ha == "right":
            x -= sum(self._segment_width(text, transform, **kw) for text, _ in parts)

        t = transform
        for text, color in parts:
            txt = self._ax.text(x, y, text, color=color, transform=t, **kw)
            txt.draw(self._fig.canvas.get_renderer())
            width_pts = txt.get_window_extent().width / self._fig.dpi * 72
            t = transforms.offset_copy(txt._transform, x=width_pts, units="points", fig=self._fig)


@contextmanager
def _partial_color_text(fig: Figure, ax):
    """Scope partial-color text helpers to one figure build."""
    yield _PartialColorText(fig, ax)


def _map_vaidya_to_model_axis(counts, *, n_days_max: int, n_reps_max: int) -> np.ndarray:
    """Map Vaidya day/prior counts onto the model's integer repetition axis."""
    return np.asarray(counts, dtype=float) / float(n_days_max) * float(n_reps_max)


def _map_vaidya_recording_days_to_model_axis(
    days, *, n_days_max: int, n_reps_max: int
) -> np.ndarray:
    """Map Vaidya 1-indexed recording days onto the model's 1-indexed repetition axis."""
    days = np.asarray(days, dtype=float)
    if n_days_max <= 1 or n_reps_max <= 1:
        return days
    return (days - 1) / float(n_days_max - 1) * float(n_reps_max - 1) + 1.0


def _vaidya_new_pc_fraction_by_day() -> tuple[np.ndarray, np.ndarray]:
    pools = vt.pools()
    return pools["days"], pools["new"] / pools["total"]


def _vaidya_sustained_fraction_by_day() -> tuple[np.ndarray, np.ndarray]:
    pools = vt.pools()
    return pools["days"], pools["sustained_fraction"]


def _normalized_btsp_style() -> dict[str, Any]:
    """Full Vaidya curve on a protocol-normalised x mapping, for truncated plots."""
    return {
        "color": "0.55",
        "linewidth": 2.0,
        "linestyle": "-",
        "marker": "D",
        "markersize": 5,
        "markerfacecolor": "white",
        "markeredgewidth": 1.2,
        "alpha": 0.8,
        "zorder": 9,
        "label": "BTSP (normalized)",
    }


def _dual_days_reps_xticks(
    ax,
    fig,
    *,
    n_days_max: int,
    n_reps_max: int,
    xlabel_prefix: str = "",
    xlabel_suffix: str = "",
) -> None:
    """Two tick rows on a normalised [0, 1] protocol axis."""
    days_color = "black"
    reps_color = "#e65671"
    zero_color = "black"
    right_pad = 0.045

    f = report_font()
    trans = ax.get_xaxis_transform()
    tick_kw = {"transform": trans, "clip_on": False, "solid_capstyle": "butt", "linewidth": 0.9}
    lbl_kw = {
        "transform": trans,
        "ha": "center",
        "va": "top",
        "fontsize": f["tick"],
        "clip_on": False,
    }

    ax.set_xlim(0, 1 + right_pad)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    day_vals = np.arange(n_days_max + 1)
    rep_vals = np.arange(n_reps_max + 1)
    for d, pos in zip(day_vals, day_vals / float(n_days_max)):
        ax.plot([pos, pos], [0, -0.022], color=days_color, **tick_kw)
        ax.text(pos, -0.035, str(int(d)), color=days_color, **lbl_kw)
    for r, pos in zip(rep_vals, rep_vals / float(n_reps_max)):
        if r == 0:
            continue
        ax.plot([pos, pos], [0, -0.072], color=reps_color, **tick_kw)
        ax.text(pos, -0.085, str(int(r)), color=reps_color, **lbl_kw)

    row_lbl_kw = {**lbl_kw, "ha": "right"}
    with _partial_color_text(fig, ax) as pc:
        pc.horizontal(-0.01, -0.035, [("days", days_color)], **row_lbl_kw)
        pc.horizontal(-0.01, -0.085, [("repetitions", reps_color)], **row_lbl_kw)
        xlabel_segments = [
            (xlabel_prefix, zero_color),
            ("days", days_color),
            (" or ", zero_color),
            ("repetitions", reps_color),
            (xlabel_suffix, zero_color),
        ]
        pc.horizontal(
            0.5,
            -0.13,
            [(text, color) for text, color in xlabel_segments if text],
            transform=ax.transAxes,
            ha="center",
            fontsize=f["axis_label"],
            va="top",
        )


def plot_formation_probability_vs_vaidya(
    results,
    *,
    optimizers=None,
    pool: PoolMode = "counts",
    figsize=(8.5, 5.0),
    show_proportion: bool = False,
    show_normalized: bool = False,
) -> Figure:
    """Fig 2c overlay: how fast does a prior field guarantee the next one?

    With ``show_proportion=False`` (default), Vaidya is truncated to the model's
    prior-count range on a shared integer axis. With ``show_proportion=True``,
    both curves use a normalised axis with dual day/repetition tick rows.
    ``show_normalized`` adds the full protocol-normalised Vaidya curve on the
    integer axis; it applies only when ``show_proportion=False``.
    """
    opts = _optimizers(results, optimizers)
    fp = model_formation_probability(results, optimizers=opts, pool=pool)
    colors, labels = report_optimizer_colors(), report_optimizer_labels()

    fig, ax = plt.subplots(figsize=figsize)
    if show_proportion:
        prior_days = vt.load_targets()["fig2c_formation_probability"]["prior_days"]
        n_days_max = int(max(prior_days))
        n_reps_max = int(fp["n_prior_days_with_pf"].max())
        vaidya_x = np.arange(len(prior_days)) / float(n_days_max)
        for opt in opts:
            sub = fp[fp["optimizer"] == opt].sort_values("n_prior_days_with_pf")
            ax.plot(
                sub["n_prior_days_with_pf"] / n_reps_max,
                sub["formation_probability"],
                "o-",
                color=colors[opt],
                linewidth=2.0,
                markersize=7,
                label=labels[opt],
            )
        ax.plot(vaidya_x, vt.formation_probability(), **report_reference_style())
        _style_axis(ax, xlabel="", ylabel="P(active this repetition)")
        _dual_days_reps_xticks(
            ax,
            fig,
            n_days_max=n_days_max,
            n_reps_max=n_reps_max,
            xlabel_prefix="# prior ",
            xlabel_suffix=" with a place field",
        )
    else:
        prior_days = vt.load_targets()["fig2c_formation_probability"]["prior_days"]
        n_days_max = int(max(prior_days))
        n_reps_max = int(fp["n_prior_days_with_pf"].max())
        n_prior = n_reps_max + 1
        for opt in opts:
            sub = fp[fp["optimizer"] == opt].sort_values("n_prior_days_with_pf")
            ax.plot(
                sub["n_prior_days_with_pf"],
                sub["formation_probability"],
                "o-",
                color=colors[opt],
                linewidth=2.0,
                markersize=7,
                label=labels[opt],
            )
        x = np.arange(n_prior)
        if show_normalized:
            ax.plot(
                _map_vaidya_to_model_axis(prior_days, n_days_max=n_days_max, n_reps_max=n_reps_max),
                vt.formation_probability(),
                **_normalized_btsp_style(),
            )
        ax.plot(x, vt.formation_probability(n_prior), **report_reference_style())
        ax.set_xticks(x)
        _style_axis(
            ax,
            xlabel="# prior repetitions with a place field",
            ylabel="P(active this repetition)",
        )
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=report_font()["legend"], loc="lower right", ncol=2)
    fig.tight_layout()
    if show_proportion:
        fig.subplots_adjust(bottom=0.22)
    return fig


def plot_sustained_fraction_vs_vaidya(
    results,
    *,
    optimizers=None,
    vaidya_days=(2, 3, 4),
    figsize=(8.5, 5.0),
    show_proportion: bool = False,
    show_normalized: bool = False,
) -> Figure:
    """Fig 2k overlay: the sustained/transient equilibrium across days.

    Model repetitions 1-3 align to Vaidya recording days 2-4. With
    ``show_proportion=False`` (default), both curves share a 1-indexed integer
    repetition axis. With ``show_proportion=True``, both curves use a
    normalised axis with dual day/repetition tick rows. ``show_normalized`` adds
    the full protocol-normalised Vaidya curve on the integer axis; it applies
    only when ``show_proportion=False``.
    """
    opts = _optimizers(results, optimizers)
    pools = model_pc_pools(results, optimizers=opts)
    colors, labels = report_optimizer_colors(), report_optimizer_labels()
    model_reps = sorted(pools.loc[pools["repetition"] > 0, "repetition"].unique())
    n_reps_max = int(max(model_reps))
    vaidya_days_all, vaidya_sustained = _vaidya_sustained_fraction_by_day()
    n_days_max = int(vaidya_days_all.max())

    fig, ax = plt.subplots(figsize=figsize)
    if show_proportion:
        for opt in opts:
            sub = pools[(pools["optimizer"] == opt) & (pools["repetition"] > 0)].sort_values("repetition")
            ax.plot(
                sub["repetition"] / n_reps_max,
                sub["sustained_fraction"],
                "o-",
                color=colors[opt],
                linewidth=2.0,
                markersize=7,
                label=labels[opt],
            )
        ax.plot(vaidya_days_all / n_days_max, vaidya_sustained, **report_reference_style())
        _style_axis(ax, xlabel="", ylabel="Sustained / total place cells")
        _dual_days_reps_xticks(ax, fig, n_days_max=n_days_max, n_reps_max=n_reps_max)
    else:
        for opt in opts:
            sub = pools[(pools["optimizer"] == opt) & (pools["repetition"] > 0)].sort_values("repetition")
            ax.plot(
                sub["repetition"],
                sub["sustained_fraction"],
                "o-",
                color=colors[opt],
                linewidth=2.0,
                markersize=7,
                label=labels[opt],
            )
        if show_normalized:
            ax.plot(
                _map_vaidya_recording_days_to_model_axis(
                    vaidya_days_all, n_days_max=n_days_max, n_reps_max=n_reps_max
                ),
                vaidya_sustained,
                **_normalized_btsp_style(),
            )
        ax.plot(
            list(range(1, len(vaidya_days) + 1)),
            vt.sustained_fraction(vaidya_days),
            **report_reference_style(),
        )
        ax.set_xticks(model_reps)
        _style_axis(ax, xlabel="Repetition (day)", ylabel="Sustained / total place cells")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=report_font()["legend"], loc="upper left", ncol=2)
    fig.tight_layout()
    if show_proportion:
        fig.subplots_adjust(bottom=0.22)
    return fig


def plot_recruitment_vs_vaidya(
    results,
    *,
    optimizers=None,
    vaidya_days=(2, 3, 4),
    figsize=(8.5, 5.0),
    show_proportion: bool = False,
    show_normalized: bool = False,
    include_first: bool = False,
) -> Figure:
    """Fig 2e/2j overlay: what share of each day's place cells is brand new?

    The reference numerator is the Fig 2e cohort-count diagonal (cells whose
    onset day is the recording day); the denominator is the Fig 2j total pool.

    Model repetitions 1-3 align to Vaidya recording days 2-4. With
    ``show_proportion=False`` (default), both curves share a 1-indexed integer
    repetition axis. With ``show_proportion=True``, both curves use a
    normalised axis with dual day/repetition tick rows. ``show_normalized`` adds
    the full protocol-normalised Vaidya curve on the integer axis; it applies
    only when ``show_proportion=False``.

    ``include_first`` extends the integer axis left to model repetition 0 and
    Vaidya day 1, where every field is new by construction. Both curves then
    start from a common point, which makes the subsequent divergence readable.
    """
    if include_first and show_proportion:
        raise ValueError("include_first applies to the integer repetition axis only")
    rep_min = 0 if include_first else 1
    opts = _optimizers(results, optimizers)
    pools = model_pc_pools(results, optimizers=opts)
    colors, labels = report_optimizer_colors(), report_optimizer_labels()
    model_reps = sorted(pools.loc[pools["repetition"] >= rep_min, "repetition"].unique())
    n_reps_max = int(max(model_reps))
    vaidya_days_all, vaidya_new = _vaidya_new_pc_fraction_by_day()
    n_days_max = int(vaidya_days_all.max())

    fig, ax = plt.subplots(figsize=figsize)
    if show_proportion:
        for opt in opts:
            sub = pools[(pools["optimizer"] == opt) & (pools["repetition"] > 0)].sort_values("repetition")
            ax.plot(
                sub["repetition"] / n_reps_max,
                sub["new_fraction"],
                "o-",
                color=colors[opt],
                linewidth=2.0,
                markersize=7,
                label=labels[opt],
            )
        ax.plot(
            vaidya_days_all / n_days_max,
            vaidya_new,
            **report_reference_style(),
        )
        _style_axis(ax, xlabel="", ylabel="Newly recruited / total place cells")
        _dual_days_reps_xticks(ax, fig, n_days_max=n_days_max, n_reps_max=n_reps_max)
    else:
        for opt in opts:
            sub = pools[(pools["optimizer"] == opt) & (pools["repetition"] >= rep_min)].sort_values("repetition")
            ax.plot(
                sub["repetition"],
                sub["new_fraction"],
                "o-",
                color=colors[opt],
                linewidth=2.0,
                markersize=7,
                label=labels[opt],
            )
        if show_normalized:
            ax.plot(
                _map_vaidya_recording_days_to_model_axis(
                    vaidya_days_all, n_days_max=n_days_max, n_reps_max=n_reps_max
                ),
                vaidya_new,
                **_normalized_btsp_style(),
            )
        days_used = (1, *vaidya_days) if include_first else tuple(vaidya_days)
        ax.plot(
            [d - 1 for d in days_used],
            vt.new_pc_fraction(days_used),
            **report_reference_style(),
        )
        ax.set_xticks(model_reps)
        _style_axis(ax, xlabel="Repetition (day)", ylabel="Newly recruited / total place cells")
    ax.set_ylim(0, 1.05 if include_first else 0.75)
    ax.legend(frameon=False, fontsize=report_font()["legend"], loc="upper right", ncol=2)
    fig.tight_layout()
    if show_proportion:
        fig.subplots_adjust(bottom=0.22)
    return fig


def plot_days_with_pf_vs_vaidya(
    results,
    *,
    optimizers=None,
    figsize=(9.0, 5.0),
    show_proportion: bool = False,
    show_normalized: bool = False,
) -> Figure:
    """Fig 2b overlay: graded lifetimes in CA1 versus bimodal ones in the model.

    With ``show_proportion=False`` (default), Vaidya is truncated to the model's
    repetition range on a shared integer axis. With ``show_proportion=True``,
    both curves use a normalised axis with dual day/repetition tick rows.
    ``show_normalized`` adds the full protocol-normalised Vaidya curve on the
    integer axis; it applies only when ``show_proportion=False``.
    """
    opts = _optimizers(results, optimizers)
    act = model_repetition_activity(results, optimizers=opts)
    colors, labels = report_optimizer_colors(), report_optimizer_labels()
    tgt = vt.days_with_pf()
    n_reps_max = int(act["n_days_with_pf"].max())

    fig, ax = plt.subplots(figsize=figsize)
    if show_proportion:
        n_days_max = int(tgt["days"].max())
        for opt in opts:
            sub = act[act["optimizer"] == opt].sort_values("n_days_with_pf")
            ax.plot(
                sub["n_days_with_pf"] / n_reps_max,
                sub["pct_neurons"] / 100.0,
                "o-",
                color=colors[opt],
                linewidth=2.0,
                markersize=7,
                label=labels[opt],
            )
        ax.plot(tgt["days"] / n_days_max, tgt["data"], **report_reference_style())
        _style_axis(ax, xlabel="", ylabel="Proportion of neurons")
        _dual_days_reps_xticks(ax, fig, n_days_max=n_days_max, n_reps_max=n_reps_max)
    else:
        n_days_max = int(tgt["days"].max())
        n_points = n_reps_max + 1
        for opt in opts:
            sub = act[act["optimizer"] == opt].sort_values("n_days_with_pf")
            ax.plot(
                sub["n_days_with_pf"],
                sub["pct_neurons"] / 100.0,
                "o-",
                color=colors[opt],
                linewidth=2.0,
                markersize=7,
                label=labels[opt],
            )
        if show_normalized:
            ax.plot(
                _map_vaidya_to_model_axis(tgt["days"], n_days_max=n_days_max, n_reps_max=n_reps_max),
                tgt["data"],
                **_normalized_btsp_style(),
            )
        ax.plot(tgt["days"][:n_points], tgt["data"][:n_points], **report_reference_style())
        ax.set_xticks(np.arange(n_points))
        _style_axis(
            ax,
            xlabel="Repetitions with a place field",
            ylabel="Proportion of neurons",
        )
    ax.set_ylim(0, 0.62)
    ax.legend(frameon=False, fontsize=report_font()["legend"], loc="upper center", ncol=2)
    fig.tight_layout()
    if show_proportion:
        fig.subplots_adjust(bottom=0.22)
    return fig


def plot_onset_anchors_vs_vaidya(
    results, *, optimizers=None, laps=(1, 5, 10, 20), figsize=(13.5, 5.4)
) -> Figure:
    """Fig 3f/3g overlay: the two within-session BTSP signatures.

    Left panel is the birth-day null result. In CA1 the sustained and transient
    onset curves are superimposed, so the gap between them is ~0: at birth you
    cannot tell which fields will persist. Plotting the gap rather than both
    curves keeps ten model traces off the panel.

    Right panel is the fast-recall result: established fields turn on earlier on
    a later day, but still climb gradually across the session.
    """
    opts = _optimizers(results, optimizers)
    anchors = model_onset_anchors(results, optimizers=opts, laps=laps).set_index(
        ["cohort", "label", "optimizer"]
    )
    colors, labels = report_optimizer_colors(), report_optimizer_labels()
    lap_cols = [f"lap{lap}" for lap in laps]
    x = np.array(laps, dtype=float)
    f = report_font()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    v_new = vt.onset_cdf("stable_new")["mean"]
    v_tr = vt.onset_cdf("transient")["mean"]
    v_gap = np.array([v_new[int(lap) - 1] - v_tr[int(lap) - 1] for lap in laps])
    ax.axhspan(-0.05, 0.05, color="0.85", zorder=0)
    for opt in opts:
        try:
            gap = (
                anchors.loc[("birth_day", "sustained", opt), lap_cols].to_numpy(dtype=float)
                - anchors.loc[("birth_day", "transient", opt), lap_cols].to_numpy(dtype=float)
            )
        except KeyError:
            continue
        ax.plot(x, gap, "o-", color=colors[opt], linewidth=2.0, markersize=7, label=labels[opt])
    ax.plot(x, v_gap, **report_reference_style())
    ax.axhline(0.0, color="0.4", linewidth=1.0, zorder=1)
    ax.set_xticks(x)
    ax.set_ylim(-0.08, 0.42)
    _style_axis(
        ax,
        xlabel="Lap-equivalent within block",
        ylabel="Sustained − transient onset CDF",
        title="Birth day: is persistence predictable at birth?",
    )
    ax.legend(frameon=False, fontsize=f["legend"], loc="upper left", ncol=2)

    ax = axes[1]
    for opt in opts:
        try:
            vals = anchors.loc[("subsequent_days", "sustained", opt), lap_cols].to_numpy(dtype=float)
        except KeyError:
            continue
        ax.plot(x, vals, "o-", color=colors[opt], linewidth=2.0, markersize=7, label=labels[opt])
    v_old = vt.onset_cdf("stable_old")["mean"]
    ax.plot(x, [v_old[int(lap) - 1] for lap in laps], **report_reference_style())
    ax.set_xticks(x)
    ax.set_ylim(0, 1.0)
    _style_axis(
        ax,
        xlabel="Lap-equivalent within block",
        ylabel="Onset CDF of established fields",
        title="Subsequent days: how abrupt is recall?",
    )
    ax.legend(frameon=False, fontsize=f["legend"], loc="lower right")

    fig.tight_layout()
    return fig


def _onset_ecdf_step(ax, values, *, color, linestyle, zorder=3):
    v = np.sort(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    y = np.arange(1, v.size + 1, dtype=float) / v.size
    ax.step(v, y, where="post", color=color, linestyle=linestyle, linewidth=1.7, zorder=zorder)


def _plot_vaidya_onset_curve(ax, kind: str, *, linestyle: str) -> None:
    """CA1 onset CDF on the model's seconds axis, with its SEM band."""
    d = vt.onset_cdf(kind)
    x = d["laps"] / vt.VAIDYA_LAPS_PER_SESSION * vt.TRAIN_SAME_BLOCK_S
    ax.fill_between(
        x, d["mean"] - d["sem"], d["mean"] + d["sem"], color="0.15", alpha=0.18, zorder=19
    )
    ax.plot(x, d["mean"], color="0.05", linestyle=linestyle, linewidth=2.6, zorder=20)


def plot_onset_cdf_vs_vaidya(
    results, *, optimizers=None, figsize=(13.5, 5.6)
) -> Figure:
    """Fig 3f/3g overlay: the full onset CDFs rather than four sampled laps.

    The shape is the result, not the gap. CA1 fields switch on gradually across
    a session, so its curves are concave ramps; model fields are present from
    the first moments of the block, so theirs are near-flat from the origin.

    Vaidya splits sustained fields into new (birth day) and old (subsequent
    days) but pools transient ones over days 1-6, so the same transient curve
    is the reference in both panels.
    """
    opts = _optimizers(results, optimizers)
    colors, labels = report_optimizer_colors(), report_optimizer_labels()
    f = report_font()
    panels = [
        ("birth_day", "stable_new", "Birth day\n(field appears)"),
        ("subsequent_days", "stable_old", "Subsequent days\n(established field)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, (cohort, vaidya_sustained, title) in zip(axes, panels):
        for opt in opts:
            df = results.pf[opt].pf_onset_cdf_samples
            sub = df[df["cohort"] == cohort]
            for pf_label, linestyle in (("sustained", "-"), ("transient", "--")):
                _onset_ecdf_step(
                    ax,
                    sub.loc[sub["label"] == pf_label, "onset_s"],
                    color=colors[opt],
                    linestyle=linestyle,
                )
        _plot_vaidya_onset_curve(ax, vaidya_sustained, linestyle="-")
        _plot_vaidya_onset_curve(ax, "transient", linestyle="--")
        ax.set_xlim(0, vt.TRAIN_SAME_BLOCK_S)
        ax.set_ylim(0, 1.02)
        _style_axis(
            ax,
            xlabel="Onset time in ANNs (s)",# within train_same block (s)",
            ylabel="Onset CDF" if ax is axes[0] else "",
        )
        lap_axis = ax.secondary_xaxis(
            "top",
            functions=(
                lambda s: s / vt.TRAIN_SAME_BLOCK_S * vt.VAIDYA_LAPS_PER_SESSION,
                lambda l: l / vt.VAIDYA_LAPS_PER_SESSION * vt.TRAIN_SAME_BLOCK_S,
            ),
        )
        lap_axis.set_xlabel("Equivalent CA1 lap", fontsize=f["axis_label"])
        lap_axis.tick_params(labelsize=f["tick"])
        ax.set_title(title, fontsize=f["panel_title"]+8, pad=120,fontweight="bold")

    handles = [
        plt.Line2D([], [], color=colors[o], linewidth=1.9, label=labels[o]) for o in opts
    ]
    handles += [
        plt.Line2D([], [], color="0.05", linewidth=2.6, label="CA1 (Vaidya et al.)"),
        plt.Line2D([], [], color="0.45", linewidth=1.9, linestyle="-", label="sustained"),
        plt.Line2D([], [], color="0.45", linewidth=1.9, linestyle="--", label="transient"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=4,
        frameon=False,
        fontsize=f["legend"],
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig


def plot_pf_shift_vs_vaidya(
    results,
    *,
    optimizers=None,
    room_extent_cm: float = 100.0,
    figsize=(13.5, 5.4),
) -> Figure:
    """Fig 2d overlay: *where* a field comes back, not just whether it comes back.

    Left panel is the shift density on a log axis, because the model peak sits
    several times above the CA1 one and a linear axis would flatten every model
    curve into a single spike. The horizontal line at 1.0 is chance (a field
    relocated uniformly at random), which is why the density is expressed per
    unit *fractional* shift: it makes the two environments directly comparable
    and puts chance at the same height for both.

    Right panel is the same data as a three-band composition, which is what
    carries the argument: the models match CA1 reasonably well on outright
    relocation but are missing most of its intermediate, partially-drifted
    population.
    """
    opts = _optimizers(results, optimizers)
    dens = model_shift_density(results, optimizers=opts, room_extent_cm=room_extent_cm)
    bands = model_shift_bands(results, optimizers=opts, room_extent_cm=room_extent_cm)
    ref = vt.pf_shift_density_normalized()
    colors, labels = report_optimizer_colors(), report_optimizer_labels()
    f = report_font()

    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.5, 1.0]})

    ax = axes[0]
    ax.axhline(ref["chance"], color="0.55", linestyle=":", linewidth=1.6, zorder=1)
    ax.annotate(
        "chance (uniform relocation)",
        (0.02, ref["chance"]),
        xycoords=("axes fraction", "data"),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=f["annotation"] - 1,
        color="0.4",
    )
    ax.axvline(0.0, color="0.75", linewidth=1.0, zorder=1)
    for opt in opts:
        sub = dens[dens["optimizer"] == opt].sort_values("center")
        ax.plot(
            100 * sub["center"], sub["density"],
            "-", color=colors[opt], linewidth=2.0, label=labels[opt],
        )
    ax.plot(100 * ref["centers"], ref["density"], **report_reference_style())
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 3e2)
    ax.set_xticks([-40, -20, 0, 20, 40])
    _style_axis(
        ax,
        xlabel="PF shift on reappearance (% of environment extent)",
        ylabel="Density (per unit fractional shift)",
        title="Where does the field come back?",
    )
    ax.legend(frameon=False, fontsize=f["legend"], loc="upper right", ncol=2)

    ax = axes[1]
    n_bar = len(opts) + 1
    width = 0.86 / n_bar
    for k, src in enumerate(("vaidya", *opts)):
        sub = bands[bands["source"] == src].sort_values("band_lo")
        x = np.arange(len(sub), dtype=float) + (k - (n_bar - 1) / 2) * width
        ax.bar(
            x, sub["fraction"], width=width,
            color="0.35" if src == "vaidya" else colors[src],
            edgecolor="k", linewidth=0.6,
        )
        if src == "vaidya":
            # The optimizer colours are already keyed in the left panel, so the
            # only bar needing a name here is the reference one.
            ax.annotate(
                "CA1", (x[0], sub["fraction"].iloc[0]),
                xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=f["annotation"], color="0.2",
            )
    ax.set_xticks(np.arange(len(SHIFT_BAND_LABELS), dtype=float))
    ax.set_xticklabels(SHIFT_BAND_LABELS)
    ax.set_ylim(0, 0.78)
    _style_axis(
        ax,
        xlabel="",
        ylabel="Fraction of reappearances",
        title="The missing intermediate population",
    )

    fig.tight_layout()
    return fig


def _shift_band_labels_cm(room_extent_cm: float = 100.0) -> tuple[str, ...]:
    e2, e10, e50 = (0.02 * room_extent_cm, 0.10 * room_extent_cm, 0.50 * room_extent_cm)
    return (
        f"Same place\n(<{e2:g} cm)",
        f"Graded shift\n({e2:g}-{e10:g} cm)",
        f"Relocated\n(>{e10:g} cm)",
    )


def plot_pf_shift_vs_vaidya_spatial_only(
    results,
    *,
    optimizers=None,
    room_extent_cm: float = 100.0,
    bin_width: float = 0.25,
    bin_range: tuple[float, float] = SHIFT_DISPLACEMENT_XLIM,
    pool_xy: bool = True,
    figsize=(13.5, 5.4),
    show_second_ax = True
) -> Figure:
    """Fig 2d overlay on the model room cm axis (−8 to +8 cm zoom).

    Model curves pool ``displacement_x`` and ``displacement_y`` by default.
    Vaidya cm/PDF are mapped onto that axis (183.6 → 100 cm with the Jacobian).
    """
    opts = _optimizers(results, optimizers)
    dens = model_shift_density_spatial_only(
        results,
        optimizers=opts,
        room_extent_cm=room_extent_cm,
        pool_xy=pool_xy,
        bin_width=bin_width,
        bin_range=bin_range,
    )
    bands = model_shift_bands_spatial_only(
        results,
        optimizers=opts,
        room_extent_cm=room_extent_cm,
        pool_xy=pool_xy,
    )
    ref = vt.pf_shift_native_density_on_model_cm(
        room_extent_cm=room_extent_cm,
        bin_range=bin_range,
    )
    ref_centers = ref["centers"]
    ref_density = ref["density"]
    colors, labels = report_optimizer_colors(), report_optimizer_labels()
    f = report_font()
    xlim = bin_range

    extra_kw = {}
    if show_second_ax:
        extra_kw = {"width_ratios": [1.5, 1.0]}

    fig, axes = plt.subplots(1, 2 if show_second_ax else 1, figsize=figsize, **extra_kw)

    ax = axes[0] if show_second_ax else axes
    ax.axvline(0.0, color="0.75", linewidth=1.0, zorder=1)
    ymax = float(ref_density.max()) if ref_density.size else 0.0
    for opt in opts:
        sub = dens[dens["optimizer"] == opt].sort_values("center")
        y = sub["density"].to_numpy(dtype=float)
        ymax = max(ymax, float(y.max()) if y.size else 0.0)
        ax.plot(
            sub["center"],
            sub["density"],
            "-",
            color=colors[opt],
            linewidth=2.0,
            label=labels[opt],
        )
    ax.plot(ref_centers, ref_density, **{**report_reference_style(), "markersize": 4, "alpha": 0.65})
    ax.set_xlim(xlim)
    ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1.0)
    _style_axis(
        ax,
        xlabel="PF shift on reappearance (cm)"
        if pool_xy
        else "PF shift on reappearance, Δx (cm)",
        ylabel="Probability density (per cm)",
        title="Field-displacement across days",
    )
    ax.legend(frameon=False, fontsize=f["legend"], loc="upper right", ncol=2)

    if show_second_ax:
        ax = axes[1]
        n_bar = len(opts) + 1
        width = 0.86 / n_bar
        ymax = float(bands["fraction"].max()) if not bands.empty else 0.5
        band_labels = _shift_band_labels_cm(room_extent_cm)
        for k, src in enumerate(("vaidya", *opts)):
            sub = bands[bands["source"] == src].sort_values("band_lo")
            x = np.arange(len(sub), dtype=float) + (k - (n_bar - 1) / 2) * width
            ax.bar(
                x,
                sub["fraction"],
                width=width,
                color="0.35" if src == "vaidya" else colors[src],
                edgecolor="k",
                linewidth=0.6,
            )
            if src == "vaidya":
                ax.annotate(
                    "CA1",
                    (x[0], sub["fraction"].iloc[0]),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=f["annotation"],
                    color="0.2",
                )
        ax.set_xticks(np.arange(len(band_labels), dtype=float))
        ax.set_xticklabels(band_labels)
        ax.set_ylim(0, min(0.95, ymax * 1.25))
        _style_axis(
            ax,
            xlabel="",
            ylabel="Fraction of reappearances",
            title="Shift bands (model cm)",
        )

    fig.tight_layout()
    return fig


def plot_detected_btsp_gap(results=None, *, optimizers=None, figsize=(7.5, 5.0)) -> Figure:
    """Fig 4b: plateau events drive most field formation in CA1, none in the model."""
    d = vt.detected_btsp_pct()
    opts = _optimizers(results, optimizers) if results is not None else report_optimizer_order()
    labels = report_optimizer_xtick_labels()

    fig, ax = plt.subplots(figsize=figsize)
    x_data = np.arange(len(d["categories"]), dtype=float)
    ax.bar(
        x_data, d["mean"], yerr=1.96 * d["sem"],
        width=0.62, color="0.35", edgecolor="k", linewidth=0.8,
        capsize=5, error_kw={"elinewidth": 1.3},
        label="Vaidya et al. (CA1)",
    )
    x_model = np.arange(len(opts), dtype=float) + len(d["categories"]) + 0.6
    colors = report_optimizer_colors()
    ax.bar(
        x_model, np.zeros(len(opts)),
        width=0.62, color=[colors[o] for o in opts], edgecolor="k", linewidth=0.8,
    )
    for xi in x_model:
        ax.annotate(
            "0", (xi, 0), xytext=(0, 6), textcoords="offset points",
            ha="center", fontsize=report_font()["annotation"],
        )
    ax.axvline(len(d["categories"]) - 0.2, color="0.6", linestyle="--", linewidth=1.2)
    ax.set_xticks(np.concatenate([x_data, x_model]))
    ax.set_xticklabels(
        [c.replace(" (", "\n(") for c in d["categories"]] + [labels[o] for o in opts],
    )
    ax.set_ylim(0, 100)
    _style_axis(ax, xlabel="", ylabel="% cells with a detected plateau")
    ax.legend(frameon=False, fontsize=report_font()["legend"], loc="upper right")
    fig.tight_layout()
    return fig


def plot_scorecard(
    scorecard: pd.DataFrame, *, figsize=(10.5, 5.6), annotate: bool = True
) -> Figure:
    """Heatmap of normalised deviation from CA1: 0 = closest, 1 = furthest."""
    specs = pd.DataFrame(scorecard_statistic_specs())
    order = [s for s in specs["key"] if s in set(scorecard["statistic"])]
    opts = [o for o in report_optimizer_order() if o in set(scorecard["optimizer"])]
    mat = (
        scorecard.pivot(index="statistic", columns="optimizer", values="normalized_deviation")
        .reindex(index=order, columns=opts)
    )
    row_labels = specs.set_index("key")["label"]
    shared = specs.set_index("key")["shared_failure"]
    f = report_font()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat.to_numpy(), cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(opts)))
    ax.set_xticklabels([report_optimizer_labels()[o] for o in opts], rotation=25, ha="right",
                       fontsize=f["tick"])
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(
        [row_labels[k] + ("  †" if shared[k] else "") for k in order], fontsize=f["tick"]
    )
    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.to_numpy()[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=f["annotation"], color="0.1")
    ax.set_xticks(np.arange(len(opts) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(order) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", length=0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Normalised deviation from CA1", fontsize=f["colorbar_label"])
    cbar.ax.tick_params(labelsize=f["colorbar_tick"])
    ax.set_title("† every optimizer fails; ranking is not meaningful on that row",
                 fontsize=f["annotation"], loc="left", color="0.35", pad=10)
    fig.tight_layout()
    return fig


def plot_scorecard_dissociation(
    scorecard: pd.DataFrame, *, figsize=(8.5, 5.2)
) -> Figure:
    """The headline: across-day and within-session accuracy trade off exactly."""
    fam = scorecard_family_means(scorecard, drop_shared_failures=True)
    colors, labels = report_optimizer_colors(), report_optimizer_labels()
    f = report_font()

    lo, hi = -0.14, 1.14
    fig, ax = plt.subplots(figsize=figsize)
    placed: list[tuple[float, float]] = []
    candidates = [(0, 17), (0, -26), (0, 36), (0, -45)]
    for opt, row in fam.iterrows():
        xv, yv = float(row["across_day"]), float(row["within_session"])
        ax.scatter(xv, yv, s=260, color=colors[opt], edgecolor="k", linewidth=1.0, zorder=3)
        # Anchor sideways near the left/right edges so the text stays in frame,
        # then take the first vertical offset that clears already-placed labels.
        if xv < 0.15:
            ha, dx0 = "left", 12
        elif xv > 0.85:
            ha, dx0 = "right", -12
        else:
            ha, dx0 = "center", 0
        offset = candidates[-1]
        for dx, dy in candidates:
            y_label = yv + dy / 260.0
            if all(abs(xv - px) > 0.24 or abs(y_label - py) > 0.10 for px, py in placed):
                offset = (dx0 + dx, dy)
                placed.append((xv, y_label))
                break
        else:
            placed.append((xv, yv + offset[1] / 260.0))
        ax.annotate(
            labels[opt], (xv, yv),
            xytext=offset, textcoords="offset points",
            ha=ha, fontsize=f["annotation"], zorder=4,
        )
    ax.plot([lo, hi], [hi, lo], color="0.7", linestyle="--", linewidth=1.2, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    _style_axis(
        ax,
        xlabel="Deviation on across-day statistics",
        ylabel="Deviation on within-session statistics",
    )
    ax.xaxis.grid(True, color="0.85", linewidth=0.6)
    fig.tight_layout()
    return fig
