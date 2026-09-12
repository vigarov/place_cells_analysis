"""Experimental reference values from Vaidya et al., for model comparison.

The numbers are frozen into `vaidya_targets.json` at build time from
`data/vaidya_code_and_data/FigureData/FiguresSourceData/` via `local/vaidya_helpers`,
so the report notebook runs without the Excel sources present. Call
`refresh_targets_json()` to regenerate them.

Accessors return plain dicts rather than module-level constants so that
`%autoreload 2` picks up edits and callers can tweak values inline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# The two-room protocol trains one `train_same` block per repetition:
# 10 trajectories x 15 segments x 20 s.
TRAIN_SAME_BLOCK_S = 3000.0
# Vaidya sessions are 50 laps; a lap maps onto a fraction of our block.
VAIDYA_LAPS_PER_SESSION = 50


def targets_path() -> Path:
    return Path(__file__).with_name("vaidya_targets.json")


def load_targets() -> dict[str, Any]:
    return json.loads(targets_path().read_text())


def _arr(values) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in values], dtype=float)


def days_with_pf() -> dict[str, np.ndarray]:
    """Fig 2b: proportion of neurons having a PF on exactly N of 7 days."""
    t = load_targets()["fig2b_days_with_pf"]
    return {k: _arr(v) for k, v in t.items()}


def formation_probability(n_prior: int | None = None) -> np.ndarray:
    """Fig 2c: P(field present | j prior days with a field), averaged over RL1/RL2."""
    t = load_targets()["fig2c_formation_probability"]
    mean = (_arr(t["rl1_mean"]) + _arr(t["rl2_mean"])) / 2.0
    return mean if n_prior is None else mean[:n_prior]


def formation_probability_sem(n_prior: int | None = None) -> np.ndarray:
    t = load_targets()["fig2c_formation_probability"]
    sem = np.sqrt(_arr(t["rl1_sem"]) ** 2 + _arr(t["rl2_sem"]) ** 2) / 2.0
    return sem if n_prior is None else sem[:n_prior]


def pools() -> dict[str, np.ndarray]:
    """Fig 2j/2k: total / new / past / sustained place-cell pools by day."""
    t = load_targets()["fig2jk_pools"]
    return {k: _arr(v) for k, v in t.items()}


def sustained_fraction(days: tuple[int, ...] = (2, 3, 4)) -> np.ndarray:
    """Sustained / total place cells on the given 1-indexed recording days."""
    p = pools()
    idx = [int(d) - 1 for d in days]
    return p["sustained_fraction"][idx]


def cohort_counts() -> dict[str, np.ndarray]:
    """Fig 2e: onset-day x recording-day cohort count matrices."""
    t = load_targets()["fig2e_cohort_counts"]
    return {
        "days": _arr(t["days"]),
        "rl1": np.array([_arr(r) for r in t["rl1"]], dtype=float),
        "rl2": np.array([_arr(r) for r in t["rl2"]], dtype=float),
    }


def new_pc_fraction(days: tuple[int, ...] = (2, 3, 4)) -> np.ndarray:
    """Share of each day's place cells that were newly recruited that day.

    Uses the RL1 cohort matrix diagonal (cells whose onset day is the recording
    day) over the Fig 2j total pool.
    """
    diag = np.diag(cohort_counts()["rl1"])
    total = pools()["total"]
    frac = diag / total
    return frac[[int(d) - 1 for d in days]]


def onset_cdf(kind: str) -> dict[str, np.ndarray]:
    """Fig 3f/3g onset CDFs. `kind` is stable_new, transient, or stable_old."""
    t = load_targets()["fig3fg_onset_cdf"][kind]
    return {k: _arr(v) for k, v in t.items()}


def onset_cdf_at_lap(kind: str, lap: int) -> float:
    d = onset_cdf(kind)
    return float(d["mean"][int(lap) - 1])


def onset_lap_fractions(laps: tuple[int, ...] = (1, 5, 10, 20)) -> dict[str, float]:
    """Vaidya laps expressed as a fraction of one `train_same` block."""
    return {f"lap{lap}": lap / VAIDYA_LAPS_PER_SESSION for lap in laps}


def onset_lap_seconds(laps: tuple[int, ...] = (1, 5, 10, 20)) -> dict[str, float]:
    return {
        name: frac * TRAIN_SAME_BLOCK_S
        for name, frac in onset_lap_fractions(laps).items()
    }


def birth_day_discriminability(lap: int = 1) -> float:
    """Fig 3f: sustained minus transient onset CDF on the day a field appears.

    Near zero in the data: at birth you cannot tell which fields will persist.
    """
    return onset_cdf_at_lap("stable_new", lap) - onset_cdf_at_lap("transient", lap)


def established_onset(lap: int = 1) -> float:
    """Fig 3g: onset CDF of an already-established field on a subsequent day."""
    return onset_cdf_at_lap("stable_old", lap)


def established_onset_rise(lap_lo: int = 1, lap_hi: int = 20) -> float:
    """How much the established-field onset CDF climbs across the session."""
    return onset_cdf_at_lap("stable_old", lap_hi) - onset_cdf_at_lap("stable_old", lap_lo)


def detected_btsp_pct() -> dict[str, Any]:
    """Fig 4b: % of cells with a detected plateau event, by field category."""
    t = load_targets()["fig4b_detected_btsp_pct"]
    return {
        "categories": list(t["categories"]),
        "mean": _arr(t["mean"]),
        "sem": _arr(t["sem"]),
    }


def pf_shift_pdf() -> dict[str, np.ndarray]:
    """Fig 2d: day-to-day PF shift density on the 183.6 cm track."""
    t = load_targets()["fig2d_pf_shift"]
    return {k: _arr(v) for k, v in t.items()}


# Vaidya's linear track. Shifts are compared to our 2D room by expressing both
# as a fraction of the environment's linear extent, which also makes the
# uniform-chance density equal to 1.0 in both.
VAIDYA_TRACK_CM = 183.6


def pf_shift_density_normalized() -> dict[str, np.ndarray]:
    """Fig 2d expressed per unit fractional shift rather than per cm.

    Returns bin centres as a fraction of track length and a density in units of
    "per unit fractional shift", so a value of 1.0 is exactly chance and the
    curve reads as a fold-enrichment over a uniformly relocated field.
    """
    d = pf_shift_pdf()
    centers = d["centers_cm"] / VAIDYA_TRACK_CM
    return {
        "centers": centers,
        "density": d["pdf"] * VAIDYA_TRACK_CM,
        "bin_width": float(np.median(np.diff(centers))),
        "chance": 1.0,
    }


def pf_shift_band_fractions(
    edges: tuple[float, ...] = (0.0, 0.02, 0.10, 0.50),
) -> np.ndarray:
    """Share of CA1 reappearances whose |shift| falls in each fractional band.

    The default bands split "came back in the same place" (<2 % of the track)
    from a graded intermediate population (2-10 %) and outright relocation
    (>10 %). The published density is truncated at |shift| < 49 % of the track
    and renormalised there, so the last edge covers everything shown.
    """
    d = pf_shift_pdf()
    centers = np.abs(d["centers_cm"]) / VAIDYA_TRACK_CM
    bw = float(np.median(np.diff(d["centers_cm"] / VAIDYA_TRACK_CM)))
    mass = d["pdf"] * VAIDYA_TRACK_CM * bw
    return np.array(
        [
            float(mass[(centers > lo) & (centers <= hi)].sum())
            if lo > 0
            else float(mass[centers <= hi].sum())
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
    )


def pf_shift_native_density_on_model_cm(
    *,
    room_extent_cm: float = 100.0,
    bin_range: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Native Fig 2d bins (3.6 cm on the Vaidya track) mapped to model cm.

    Matches ``vaidya_helpers.fig2.plot_fig2d``: the source stores a PDF per
    Vaidya cm, plotted on the y-axis as probability density (not ``pdf × bin
    width``). Bin centres are scaled by ``room_extent_cm / VAIDYA_TRACK_CM``;
    density is scaled by ``VAIDYA_TRACK_CM / room_extent_cm`` so mass is
    conserved on the model cm axis.
    """
    d = pf_shift_pdf()
    scale = room_extent_cm / VAIDYA_TRACK_CM
    bw_vaidya = float(np.median(np.diff(d["centers_cm"])))
    centers = d["centers_cm"] * scale
    density = d["pdf"] * (VAIDYA_TRACK_CM / room_extent_cm)
    bw_model = bw_vaidya * scale
    if bin_range is not None:
        lo, hi = bin_range
        mask = (centers >= lo - bw_model / 2) & (centers <= hi + bw_model / 2)
        centers = centers[mask]
        density = density[mask]
    return {
        "centers": centers,
        "density": density,
        "bin_width": bw_model,
        "chance": 1.0 / room_extent_cm,
    }


def pf_shift_on_model_cm(room_extent_cm: float = 100.0) -> dict[str, np.ndarray]:
    """Fig 2d mapped from Vaidya track cm onto the model room cm axis.

    Bin centres are scaled by ``room_extent_cm / VAIDYA_TRACK_CM``; the PDF
    (per Vaidya cm) is multiplied by ``VAIDYA_TRACK_CM / room_extent_cm`` so
    probability mass is conserved. No other normalization is applied.
    """
    d = pf_shift_pdf()
    scale = room_extent_cm / VAIDYA_TRACK_CM
    bw_vaidya = float(np.median(np.diff(d["centers_cm"])))
    return {
        "centers": d["centers_cm"] * scale,
        "density": d["pdf"] * (VAIDYA_TRACK_CM / room_extent_cm),
        "bin_width": bw_vaidya * scale,
        "chance": 1.0 / room_extent_cm,
    }


def shift_band_edges_cm(
    room_extent_cm: float = 100.0,
    edges_frac: tuple[float, ...] = (0.0, 0.02, 0.10, 0.50),
) -> tuple[float, ...]:
    """Default Fig 2d band edges expressed on the model room cm axis."""
    return tuple(float(e * room_extent_cm) for e in edges_frac)


def pf_shift_band_fractions_on_model_cm(
    room_extent_cm: float = 100.0,
    edges_cm: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Band fractions after mapping Vaidya cm bands onto the model room cm axis.

    Band edges in model cm are converted back to Vaidya cm for integration;
    each fraction is integrated PDF mass divided by the total stored mass.
    """
    if edges_cm is None:
        edges_cm = shift_band_edges_cm(room_extent_cm)
    d = pf_shift_pdf()
    abs_vaidya = np.abs(d["centers_cm"])
    bw = float(np.median(np.diff(d["centers_cm"])))
    mass = d["pdf"] * bw
    total = float(mass.sum())
    vaidya_per_model = VAIDYA_TRACK_CM / room_extent_cm
    return np.array(
        [
            float(
                mass[
                    (abs_vaidya > lo * vaidya_per_model)
                    & (abs_vaidya <= hi * vaidya_per_model)
                ].sum()
                / total
            )
            if lo > 0
            else float(mass[abs_vaidya <= hi * vaidya_per_model].sum() / total)
            for lo, hi in zip(edges_cm[:-1], edges_cm[1:])
        ]
    )


def _pdf_cm_to_proportion_per_bin(
    centers_cm: np.ndarray,
    pdf_per_cm: np.ndarray,
    *,
    model_bin_width: float,
    room_extent_cm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a Vaidya PDF (per cm on the track) onto the model room cm axis.

    Returns bin centres in model cm and the matching proportion-per-bin values
    for histograms that use ``model_bin_width``, i.e. ``pdf_model_cm * width``.
    """
    scale = room_extent_cm / VAIDYA_TRACK_CM
    centers_model = centers_cm * scale
    pdf_model_cm = pdf_per_cm * (VAIDYA_TRACK_CM / room_extent_cm)
    return centers_model, pdf_model_cm * model_bin_width


def pf_shift_signed_proportion_overlay(
    *,
    model_bin_width: float,
    room_extent_cm: float = 100.0,
) -> dict[str, np.ndarray]:
    """Signed Fig 2d shift as proportion-per-bin on the model room cm axis."""
    d = pf_shift_pdf()
    centers, prop = _pdf_cm_to_proportion_per_bin(
        d["centers_cm"],
        d["pdf"],
        model_bin_width=model_bin_width,
        room_extent_cm=room_extent_cm,
    )
    return {"centers": centers, "proportion": prop}


def pf_shift_abs_proportion_overlay(
    *,
    model_bin_width: float,
    room_extent_cm: float = 100.0,
) -> dict[str, np.ndarray]:
    """|shift| PDF derived from Fig 2d, rescaled to the model room cm axis."""
    d = pf_shift_pdf()
    signed_centers = d["centers_cm"]
    signed_pdf = d["pdf"]
    bw = float(np.median(np.diff(signed_centers)))
    max_abs = float(np.max(np.abs(signed_centers)))
    abs_centers = np.arange(0.0, max_abs + bw / 2, bw)
    pdf_by_center = dict(zip(signed_centers, signed_pdf))
    abs_pdf = np.array(
        [
            float(
                pdf_by_center.get(c, 0.0)
                + (pdf_by_center.get(-c, 0.0) if c > 0 else 0.0)
            )
            for c in abs_centers
        ],
        dtype=float,
    )
    centers, prop = _pdf_cm_to_proportion_per_bin(
        abs_centers,
        abs_pdf,
        model_bin_width=model_bin_width,
        room_extent_cm=room_extent_cm,
    )
    return {"centers": centers, "proportion": prop}


def pf_shift_return_fraction(window_frac: float = 0.05) -> float:
    """P(a reappearing field lands within `window_frac` of the extent of where it was).

    Integrating the published density is more robust than reading its peak,
    which depends on the bin width.
    """
    d = pf_shift_pdf()
    centers = d["centers_cm"]
    bw = float(np.median(np.diff(centers)))
    mask = np.abs(centers) <= window_frac * VAIDYA_TRACK_CM
    return float((d["pdf"][mask] * bw).sum())


def refresh_targets_json(project_root: Path | None = None) -> Path:
    """Re-extract every target from the Vaidya source data and rewrite the JSON."""
    import sys

    local_dir = Path(__file__).resolve().parent.parent
    if str(local_dir) not in sys.path:
        sys.path.insert(0, str(local_dir))
    from vaidya_helpers import fig2, fig3, fig4

    def enc(a) -> list:
        a = np.asarray(a, dtype=float).ravel()
        return [None if not np.isfinite(v) else round(float(v), 6) for v in a]

    b = fig2.load_fig2b_data()
    c = fig2.load_fig2c_data()
    jk = fig2.load_fig2jk_data()
    e = fig2.load_fig2e_data()
    d = fig2.load_fig2d_data()
    f3 = fig3.load_fig3fgh_data()
    b4 = fig4.load_fig4b_data()

    payload = {
        "source": "Vaidya et al., source data (FigureData/FiguresSourceData)",
        "fig2b_days_with_pf": {
            "days": enc(b["days_with_pf"]),
            "data": enc(b["data"]),
            "random": enc(b["random"]),
        },
        "fig2c_formation_probability": {
            "prior_days": enc(c["history"]),
            "rl1_mean": enc(c["rl1_mean"]),
            "rl2_mean": enc(c["rl2_mean"]),
            "rl1_sem": enc(c["rl1_sem"]),
            "rl2_sem": enc(c["rl2_sem"]),
        },
        "fig2d_pf_shift": {
            "centers_cm": enc(d["centers"]),
            "pdf": enc(d["pdf"]),
            "random_pdf": enc(d["random_pdf"]),
        },
        "fig2e_cohort_counts": {
            "days": enc(e["days"]),
            "rl1": [enc(r) for r in np.asarray(e["rl1"])],
            "rl2": [enc(r) for r in np.asarray(e["rl2"])],
        },
        "fig2jk_pools": {
            k: enc(jk[k])
            for k in (
                "days",
                "total",
                "new",
                "past",
                "sustained",
                "sustained_fraction",
                "past_new_ratio",
            )
        },
        "fig3fg_onset_cdf": {
            k: {"laps": enc(f3[k]["laps"]), "mean": enc(f3[k]["mean"]), "sem": enc(f3[k]["sem"])}
            for k in ("stable_new", "transient", "stable_old")
        },
        "fig4b_detected_btsp_pct": {
            "categories": [str(x) for x in b4["categories"]],
            "mean": enc(b4["mean"]),
            "sem": enc(b4["sem"]),
        },
    }
    path = targets_path()
    path.write_text(json.dumps(payload, indent=1))
    return path
