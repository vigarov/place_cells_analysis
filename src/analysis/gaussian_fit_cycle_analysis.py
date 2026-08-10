"""Analysis helpers for `analyse_cycles.ipynb` (Gaussian RF fits)."""
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from analysis.sum_gaussians_core import _gaussian_2d
from experiments.old_cycles.ratemaps_io import parse_truncated_ratemaps_filename

GAUSSIAN_PARAM_NAMES = ("amplitude", "mu_x", "mu_y", "sigma_x", "sigma_y")
R2BandName = str  # "top" | "bottom"
AicExtremumName = str  # "high" | "low"
DEFAULT_GAUSSIAN_FIELD_SHAPE = (110, 110)


def gaussian_param_column_names(
    n_gaussians: int,
    param_names: tuple[str, ...] = GAUSSIAN_PARAM_NAMES,
) -> list[str]:
    """Column names `g1_amplitude`, ... for a flattened Gaussian-params table."""
    return [
        f"g{k + 1}_{name}" for k in range(n_gaussians) for name in param_names
    ]


def build_cell_receptive_fields_df(path: Path | str) -> pd.DataFrame:
    """
    Long-form table of per-visit Gaussian RF fits.

    Index: `(visit_idx, cell_idx)`. Gaussian parameters appear as columns
    `g1_amplitude`, `g1_mu_x`, ...
    """
    path = Path(path)
    with np.load(path) as data:
        r2 = np.asarray(data["r2"])
        gaussian_params = np.asarray(data["gaussian_params"])
        cycle_ids = np.asarray(data["cycle_ids"])
        room_ids = np.asarray(data["room_ids"])
        aic = np.asarray(data["aic"])
        signal_mean = np.asarray(data["signal_mean"])
        signal_max = np.asarray(data["signal_max"])
        signal_std = np.asarray(data["signal_std"])
        param_names = tuple(
            p.decode() if isinstance(p, bytes) else str(p)
            for p in data["param_names"]
        )

    n_visits, n_cells = r2.shape
    n_gaussians = gaussian_params.shape[2]
    n_rows = n_visits * n_cells

    visit_idx, cell_idx = np.indices((n_visits, n_cells))
    visit_idx = visit_idx.ravel()
    cell_idx = cell_idx.ravel()

    df = pd.DataFrame(
        {
            "visit_idx": visit_idx,
            "cell_idx": cell_idx,
            "cycle_id": cycle_ids[visit_idx],
            "room_id": room_ids[visit_idx],
            "r2": r2.ravel(),
            "aic": aic.ravel(),
            "signal_mean": signal_mean.ravel(),
            "signal_max": signal_max.ravel(),
            "signal_std": signal_std.ravel(),
        }
    )

    param_cols = gaussian_param_column_names(n_gaussians, param_names)
    params_flat = gaussian_params.reshape(n_rows, n_gaussians * len(param_names))
    df[param_cols] = params_flat
    df = df.set_index(["visit_idx", "cell_idx"])
    assert df.index.is_unique
    return df


def cell_room_pairs_with_nan_r2(
    df: pd.DataFrame,
    *,
    signal_max_threshold: float,
) -> pd.DataFrame:
    """
    `(cell_idx, room_id)` pairs to inspect when Gaussian fitting failed.

    Keeps cells whose peak `signal_max` exceeds *signal_max_threshold* in at
    least one visit and that have `NaN` `r2` in at least one visit to the
    same *room_id*.
    """
    peak_by_cell = df.groupby("cell_idx")["signal_max"].max()
    active_cells = peak_by_cell.index[peak_by_cell > signal_max_threshold]

    sub = df.loc[df.index.get_level_values("cell_idx").isin(active_cells)]
    nan_visits = sub[sub["r2"].isna()]
    if nan_visits.empty:
        return pd.DataFrame(columns=["cell_idx", "room_id"])

    pairs = (
        nan_visits.reset_index()[["cell_idx", "room_id"]]
        .drop_duplicates()
        .sort_values(["cell_idx", "room_id"])
        .reset_index(drop=True)
    )
    return pairs


def cell_room_pairs_with_mean_signal_max_above(
    df: pd.DataFrame,
    *,
    signal_max_threshold: float,
) -> pd.DataFrame:
    """
    `(cell_idx, room_id)` pairs whose mean `signal_max` over cycles exceeds
    *signal_max_threshold*.
    """
    mean_max = (
        df.reset_index()
        .groupby(["cell_idx", "room_id"])["signal_max"]
        .mean()
    )
    active = mean_max[mean_max > signal_max_threshold]
    if active.empty:
        return pd.DataFrame(columns=["cell_idx", "room_id"])
    return (
        active.reset_index()[["cell_idx", "room_id"]]
        .sort_values(["cell_idx", "room_id"])
        .reset_index(drop=True)
    )


def filter_df_to_cell_room_pairs(
    df: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Restrict *df* to rows whose `(cell_idx, room_id)` appears in *pairs*."""
    if pairs.empty:
        return df.iloc[0:0].copy()
    sub = df.reset_index()
    keys = pairs[["cell_idx", "room_id"]].drop_duplicates()
    sub = sub.merge(keys, on=["cell_idx", "room_id"], how="inner")
    return sub.set_index(["visit_idx", "cell_idx"])


def nan_r2_affected_counts(pairs: pd.DataFrame) -> tuple[int, int, int]:
    """
    Count affected cells, rooms, and `(cell, room)` pairs.

    Returns `(n_cells, n_rooms, n_pairs)`.
    """
    if pairs.empty:
        return 0, 0, 0
    return (
        int(pairs["cell_idx"].nunique()),
        int(pairs["room_id"].nunique()),
        len(pairs),
    )


def select_nan_r2_pairs_cyclical(
    pairs: pd.DataFrame,
    max_plots: int | None,
) -> pd.DataFrame:
    """
    Pick one room per cell per round, cycling cells until *max_plots* pairs.

    Each round takes the next unplotted room for every cell (in `cell_idx`
    order). Rounds repeat until *max_plots* pairs are chosen or all pairs are
    exhausted. `max_plots=None` selects every pair in this order.
    """
    if pairs.empty:
        return pairs.copy()

    rooms_by_cell: dict[int, list[int]] = {}
    for cell_idx, group in pairs.groupby("cell_idx", sort=True):
        rooms_by_cell[int(cell_idx)] = sorted(group["room_id"].unique().astype(int))

    cells = sorted(rooms_by_cell)
    room_index = dict.fromkeys(cells, 0)
    selected: list[tuple[int, int]] = []

    while max_plots is None or len(selected) < max_plots:
        added = False
        for cell_idx in cells:
            if max_plots is not None and len(selected) >= max_plots:
                break
            rooms = rooms_by_cell[cell_idx]
            idx = room_index[cell_idx]
            if idx < len(rooms):
                selected.append((cell_idx, rooms[idx]))
                room_index[cell_idx] = idx + 1
                added = True
        if not added:
            break

    return pd.DataFrame(selected, columns=["cell_idx", "room_id"])


def visits_for_cell_in_room(
    df: pd.DataFrame,
    cell_idx: int,
    room_id: int,
) -> pd.DataFrame:
    """All visits for one cell in one room, sorted by cycle."""
    mask = (df.index.get_level_values("cell_idx") == cell_idx) & (
        df["room_id"] == room_id
    )
    out = df.loc[mask].reset_index().sort_values("cycle_id").reset_index(drop=True)
    return out


def gaussian_fields_from_params(
    params: np.ndarray,
    *,
    field_shape: tuple[int, int] = DEFAULT_GAUSSIAN_FIELD_SHAPE,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Render independent 2D Gaussian components and their sum.

    Parameters
    ----------
    params
        `(n_gaussians, 5)` with `amplitude, mu_x, mu_y, sigma_x, sigma_y`.

    Returns
    -------
    components, sum_field
        Shapes `(n_gaussians, H, W)` and `(H, W)`. `None` if params are
        not finite.
    """
    if params.ndim != 2 or params.shape[1] != 5:
        raise ValueError(f"Expected params (n_gaussians, 5), got {params.shape}")
    if not np.all(np.isfinite(params)):
        return None

    h, w = field_shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    n_gaussians = params.shape[0]
    components = np.stack(
        [_gaussian_2d(params[k], xx, yy) for k in range(n_gaussians)],
        axis=0,
    )
    return components, components.sum(axis=0)


def params_array_from_row(
    row: pd.Series,
    n_gaussians: int,
    param_names: tuple[str, ...] = GAUSSIAN_PARAM_NAMES,
) -> np.ndarray:
    """Extract `(n_gaussians, 5)` params from a dataframe row."""
    cols = gaussian_param_column_names(n_gaussians, param_names)
    return row[cols].to_numpy(dtype=np.float64).reshape(n_gaussians, len(param_names))


class AicR2Correlation(NamedTuple):
    """Finite `(r2, aic)` pairs, their statistics, and the source rows."""

    r2: np.ndarray
    aic: np.ndarray
    correlation: float
    slope: float
    intercept: float
    filtered_df: pd.DataFrame


def filter_df_for_aic_r2_analysis(
    df: pd.DataFrame,
    *,
    r2_threshold: float | None = None,
    signal_max_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Rows used for AIC–R² correlation (and downstream outlier selection).

    If `r2_threshold` is set, only rows with `r2` strictly greater than it
    are included. If `signal_max_threshold` is set, only rows belonging to
    `(cell_idx, room_id)` pairs whose mean `signal_max` over cycles exceeds
    that threshold are included. Only rows with finite `r2` and `aic` are
    kept.
    """
    if signal_max_threshold is not None:
        active_pairs = cell_room_pairs_with_mean_signal_max_above(
            df,
            signal_max_threshold=signal_max_threshold,
        )
        df = filter_df_to_cell_room_pairs(df, active_pairs)
    mask = df["r2"].notna()
    sub = df.loc[mask].copy()
    finite = np.isfinite(sub["r2"].to_numpy()) & np.isfinite(
        sub["aic"].to_numpy(dtype=np.float64)
    )
    sub = sub.loc[finite]
    if r2_threshold is not None:
        sub = sub.loc[sub["r2"] > r2_threshold]
    return sub


def finite_aic_r2_pairs(
    df: pd.DataFrame,
    *,
    r2_threshold: float | None = None,
    signal_max_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    `r2` and `aic` for fits with non-NaN `r2` and finite `aic`.

    See :func:`filter_df_for_aic_r2_analysis` for filtering rules.

    Returns
    -------
    r2, aic
        1-D arrays of equal length (may be empty).
    """
    sub = filter_df_for_aic_r2_analysis(
        df,
        r2_threshold=r2_threshold,
        signal_max_threshold=signal_max_threshold,
    )
    r2 = sub["r2"].to_numpy(dtype=np.float64)
    aic = sub["aic"].to_numpy(dtype=np.float64)
    return r2, aic


def compute_aic_r2_correlation(
    df: pd.DataFrame,
    *,
    r2_threshold: float | None = None,
    signal_max_threshold: float | None = None,
) -> AicR2Correlation:
    """Pearson correlation and least-squares linear fit of AIC vs `r2`."""
    filtered_df = filter_df_for_aic_r2_analysis(
        df,
        r2_threshold=r2_threshold,
        signal_max_threshold=signal_max_threshold,
    )
    r2 = filtered_df["r2"].to_numpy(dtype=np.float64)
    aic = filtered_df["aic"].to_numpy(dtype=np.float64)
    if len(r2) < 2:
        return AicR2Correlation(
            r2, aic, np.nan, np.nan, np.nan, filtered_df
        )
    correlation = float(np.corrcoef(r2, aic)[0, 1])
    slope, intercept = np.polyfit(r2, aic, 1)
    return AicR2Correlation(
        r2, aic, correlation, float(slope), float(intercept), filtered_df
    )


class OutlierPoint(NamedTuple):
    """One `(visit, cell)` fit selected from an R² tail by AIC extremum."""

    visit_idx: int
    cell_idx: int
    room_id: int
    cycle_id: int
    r2_band: R2BandName
    aic_extremum: AicExtremumName
    r2: float
    aic: float


def _outlier_candidate_frame(
    df: pd.DataFrame,
    *,
    r2_threshold: float | None,
) -> pd.DataFrame:
    """Rows with finite `r2` and `aic`, optionally above *r2_threshold*."""
    sub = df.reset_index()
    finite_r2 = sub["r2"].notna() & np.isfinite(sub["r2"])
    finite_aic = sub["aic"].notna() & np.isfinite(sub["aic"])
    sub = sub.loc[finite_r2 & finite_aic].copy()
    if r2_threshold is not None:
        sub = sub.loc[sub["r2"] > r2_threshold]
    return sub


def _r2_tail_pool(
    sub: pd.DataFrame,
    *,
    band: R2BandName,
    pool_fraction: float,
) -> pd.DataFrame:
    """Top or bottom `pool_fraction` of rows by `r2`."""
    if sub.empty:
        return sub.copy()
    if not 0.0 < pool_fraction <= 1.0:
        raise ValueError(f"pool_fraction must be in (0, 1], got {pool_fraction}")

    pool_size = max(1, int(np.ceil(len(sub) * pool_fraction)))
    sort_cols = ["r2", "visit_idx", "cell_idx"]
    sub = sub.sort_values(sort_cols, ascending=[True, True, True])
    if band == "bottom":
        return sub.nsmallest(pool_size, "r2", keep="all").head(pool_size)
    if band == "top":
        return sub.nlargest(pool_size, "r2", keep="all").head(pool_size)
    raise ValueError(f"band must be 'top' or 'bottom', got {band!r}")


def _rows_to_outlier_points(
    rows: pd.DataFrame,
    *,
    r2_band: R2BandName,
    aic_extremum: AicExtremumName,
) -> list[OutlierPoint]:
    points: list[OutlierPoint] = []
    for _, row in rows.iterrows():
        points.append(
            OutlierPoint(
                visit_idx=int(row["visit_idx"]),
                cell_idx=int(row["cell_idx"]),
                room_id=int(row["room_id"]),
                cycle_id=int(row["cycle_id"]),
                r2_band=r2_band,
                aic_extremum=aic_extremum,
                r2=float(row["r2"]),
                aic=float(row["aic"]),
            )
        )
    return points


def _aic_extremes_in_pool(
    pool: pd.DataFrame,
    *,
    r2_band: R2BandName,
    n: int,
) -> list[OutlierPoint]:
    """Up to `2 * n` points with highest and lowest AIC inside one R² tail."""
    if pool.empty:
        return []

    sort_cols = ["aic", "visit_idx", "cell_idx"]
    pool = pool.sort_values(sort_cols, ascending=[True, True, True])
    points: list[OutlierPoint] = []
    low_rows = pool.nsmallest(n, "aic", keep="all").head(n)
    high_rows = pool.nlargest(n, "aic", keep="all").head(n)
    points.extend(
        _rows_to_outlier_points(low_rows, r2_band=r2_band, aic_extremum="low")
    )
    points.extend(
        _rows_to_outlier_points(high_rows, r2_band=r2_band, aic_extremum="high")
    )
    return points


def select_aic_r2_outlier_points(
    df: pd.DataFrame,
    *,
    n_per_group: int = 5,
    r2_threshold: float | None = None,
    r2_pool_fraction: float = 0.05,
) -> list[OutlierPoint]:
    """
    Select AIC extremes within the top and bottom R² tails.

    Builds two pools (top / bottom `r2_pool_fraction` of eligible rows by
    `r2`), then picks `n_per_group` highest- and lowest-AIC points in each
    pool (up to `4 * n_per_group` total).
    """
    sub = _outlier_candidate_frame(df, r2_threshold=r2_threshold)
    points: list[OutlierPoint] = []
    for band in ("top", "bottom"):
        pool = _r2_tail_pool(sub, band=band, pool_fraction=r2_pool_fraction)
        points.extend(
            _aic_extremes_in_pool(pool, r2_band=band, n=n_per_group)
        )
    return points


def outlier_points_summary_df(points: list[OutlierPoint]) -> pd.DataFrame:
    """Tabular summary of selected outlier points."""
    if not points:
        return pd.DataFrame(
            columns=[
                "r2_band",
                "aic_extremum",
                "visit_idx",
                "cell_idx",
                "room_id",
                "cycle_id",
                "r2",
                "aic",
            ]
        )
    return pd.DataFrame(
        [
            {
                "r2_band": p.r2_band,
                "aic_extremum": p.aic_extremum,
                "visit_idx": p.visit_idx,
                "cell_idx": p.cell_idx,
                "room_id": p.room_id,
                "cycle_id": p.cycle_id,
                "r2": p.r2,
                "aic": p.aic,
            }
            for p in points
        ]
    )


def build_truncated_segments(
    truncated_paths: list[Path | str],
) -> list[tuple[int, int, Path]]:
    """
    Parse and validate a contiguous truncated rate-map series.

    Returns sorted `(start_cycle, end_cycle, path)` triples.
    """
    segments: list[tuple[int, int, Path]] = []
    for raw_path in truncated_paths:
        path = Path(raw_path)
        bounds = parse_truncated_ratemaps_filename(path.name)
        if bounds is None:
            raise ValueError(f"Not a truncated ratemaps file: {path.name}")
        start_cycle, end_cycle = bounds
        segments.append((start_cycle, end_cycle, path))

    segments.sort(key=lambda item: item[0])
    expected_start = 0
    for start_cycle, end_cycle, path in segments:
        if start_cycle != expected_start:
            raise ValueError(
                f"Gap in truncated series at cycle {expected_start}: "
                f"next file is {path.name} (starts at cycle {start_cycle})."
            )
        if end_cycle <= start_cycle:
            raise ValueError(
                f"Invalid cycle range in {path.name}: [{start_cycle}, {end_cycle})"
            )
        expected_start = end_cycle
    return segments


def n_rooms_from_truncated_file(path: Path | str) -> int:
    """Number of rooms per cycle from a truncated NPZ `schedule` array."""
    with np.load(path) as data:
        schedule = np.asarray(data["schedule"])
    return int(schedule.shape[1])


def visit_to_segment(
    visit_idx: int,
    segments: list[tuple[int, int, Path]],
    n_rooms: int,
) -> tuple[Path, int]:
    """Map a global visit index to `(segment_path, local_visit_index)`."""
    for start_cycle, end_cycle, path in segments:
        visit_start = start_cycle * n_rooms
        visit_end = end_cycle * n_rooms
        if visit_start <= visit_idx < visit_end:
            return path, visit_idx - visit_start
    raise ValueError(f"visit_idx {visit_idx} not covered by truncated segments")


def collect_ratemap_requests(
    df: pd.DataFrame,
    points: list[OutlierPoint],
    segments: list[tuple[int, int, Path]],
    n_rooms: int,
) -> dict[Path, list[tuple[int, int, int]]]:
    """
    Group required `(local_v, cell_idx, visit_idx)` tuples by truncated NPZ.

    Includes every visit for each point's `(cell_idx, room_id)` pair.
    """
    needed: set[tuple[int, int]] = set()
    for point in points:
        visits = visits_for_cell_in_room(df, point.cell_idx, point.room_id)
        for _, row in visits.iterrows():
            needed.add((int(row["visit_idx"]), int(row["cell_idx"])))

    by_path: dict[Path, list[tuple[int, int, int]]] = {}
    for visit_idx, cell_idx in needed:
        path, local_v = visit_to_segment(visit_idx, segments, n_rooms)
        by_path.setdefault(path, []).append((local_v, cell_idx, visit_idx))
    return by_path


def load_ratemaps_by_visit(
    requests_by_path: dict[Path, list[tuple[int, int, int]]],
) -> dict[tuple[int, int], np.ndarray]:
    """
    Load only the requested rate maps, one truncated file at a time.

    Returns `{(visit_idx, cell_idx): field}` with copied arrays so NPZ files
    can be closed before loading the next segment.
    """
    result: dict[tuple[int, int], np.ndarray] = {}
    for path in sorted(requests_by_path, key=lambda p: str(p)):
        items = requests_by_path[path]
        with np.load(path, mmap_mode="r") as data:
            ratemaps = data["ratemaps"]
            for local_v, cell_idx, visit_idx in items:
                result[(visit_idx, cell_idx)] = np.asarray(
                    ratemaps[local_v, cell_idx]
                ).copy()
    return result


def fit_dict_from_visit_row(
    row: pd.Series,
    n_gaussians: int,
    *,
    field_shape: tuple[int, int] = DEFAULT_GAUSSIAN_FIELD_SHAPE,
    param_names: tuple[str, ...] = GAUSSIAN_PARAM_NAMES,
) -> dict | None:
    """
    Build a `plot_gaussian_fits`-compatible fit dict from a dataframe row.

    Returns `None` when Gaussian parameters are not finite.
    """
    params = params_array_from_row(row, n_gaussians, param_names)
    rendered = gaussian_fields_from_params(params, field_shape=field_shape)
    if rendered is None:
        return None

    components_arr, sum_field = rendered
    component_params = [
        {
            "amplitude": params[k, 0],
            "mu_x": params[k, 1],
            "mu_y": params[k, 2],
            "sigma_x": params[k, 3],
            "sigma_y": params[k, 4],
        }
        for k in range(n_gaussians)
    ]
    fit: dict = {
        "cell_idx": int(row["cell_idx"]),
        "components": [components_arr[k] for k in range(n_gaussians)],
        "component_params": component_params,
        "sum_field": sum_field,
        "r2": float(row["r2"]) if pd.notna(row["r2"]) else np.nan,
    }
    if "aic" in row.index and pd.notna(row["aic"]) and np.isfinite(row["aic"]):
        fit["aic"] = float(row["aic"])
    return fit
