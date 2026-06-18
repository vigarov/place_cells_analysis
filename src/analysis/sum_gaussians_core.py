"""SciPy/NumPy fitting for sums of 2D Gaussians (no PyTorch)."""
import numpy as np
from scipy.optimize import curve_fit


def _gaussian_2d(params, x, y):
    amp, mu_x, mu_y, sigma_x, sigma_y = params
    return amp * np.exp(
        -0.5 * ((x - mu_x) / sigma_x) ** 2 - 0.5 * ((y - mu_y) / sigma_y) ** 2
    )


def _make_sum_gaussians_model(n_gaussians):
    def model(xy, *flat_params):
        x, y = xy
        total = np.zeros_like(x, dtype=np.float64)
        for k in range(n_gaussians):
            total += _gaussian_2d(flat_params[k * 5 : (k + 1) * 5], x, y)
        return total

    return model


def _init_sum_gaussian_params(field, mask, xx, yy, n_gaussians):
    h, w = field.shape
    residual = np.zeros(field.shape, dtype=np.float64)
    residual[mask] = np.maximum(field[mask].astype(np.float64), 0.0)
    x_pts = xx[mask]
    y_pts = yy[mask]
    params = []

    for _ in range(n_gaussians):
        v = residual[mask]
        if v.sum() <= 0:
            params.extend([0.1, w / 2, h / 2, max(w / 4, 1.0), max(h / 4, 1.0)])
            continue

        wsum = v.sum()
        mu_x = (x_pts * v).sum() / wsum
        mu_y = (y_pts * v).sum() / wsum
        var_x = ((x_pts - mu_x) ** 2 * v).sum() / wsum
        var_y = ((y_pts - mu_y) ** 2 * v).sum() / wsum
        sigma_x = np.sqrt(max(var_x, 1.0))
        sigma_y = np.sqrt(max(var_y, 1.0))
        amp = v.max()
        comp = _gaussian_2d([amp, mu_x, mu_y, sigma_x, sigma_y], xx, yy)
        residual = np.maximum(residual - comp, 0.0)
        params.extend([amp, mu_x, mu_y, sigma_x, sigma_y])

    return params


def _aic_from_least_squares(n_obs: int, ss_res: float, n_params: int) -> float:
    """
    Akaike Information Criterion for least-squares fits.
    `AIC = n * ln(RSS / n) + 2 * k` with `k` free parameters and
    `RSS` the sum of squared residuals.
    """
    if n_obs <= n_params or ss_res <= 0 or not np.isfinite(ss_res):
        return np.nan
    return float(n_obs * np.log(ss_res / n_obs) + 2 * n_params)


def fit_sum_gaussians(field, n_gaussians=2, *, maxfev=20_000):
    """Fit a sum of independent 2D Gaussians with `scipy.optimize.curve_fit`."""
    mask = np.isfinite(field)
    if not np.any(mask):
        return None

    h, w = field.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x_pts = xx[mask]
    y_pts = yy[mask]
    y_true = field[mask].astype(np.float64)
    if np.all(y_true <= 0):
        return None

    p0 = _init_sum_gaussian_params(field, mask, xx, yy, n_gaussians)
    lower, upper = [], []
    for _ in range(n_gaussians):
        lower.extend([0.0, 0.0, 0.0, 0.5, 0.5])
        upper.extend([np.inf, w - 1, h - 1, w, h])

    model = _make_sum_gaussians_model(n_gaussians)
    try:
        popt, _ = curve_fit(
            model,
            (x_pts, y_pts),
            y_true,
            p0=p0,
            bounds=(lower, upper),
            maxfev=maxfev,
        )
    except (RuntimeError, ValueError):
        return None

    y_pred = model((x_pts, y_pts), *popt)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    n_obs = int(y_true.size)
    n_params = 5 * n_gaussians
    aic = _aic_from_least_squares(n_obs, float(ss_res), n_params)

    components = []
    component_params = []
    for k in range(n_gaussians):
        p = popt[k * 5 : (k + 1) * 5]
        comp = _gaussian_2d(p, xx, yy)
        components.append(comp)
        component_params.append(
            {
                "amplitude": p[0],
                "mu_x": p[1],
                "mu_y": p[2],
                "sigma_x": p[3],
                "sigma_y": p[4],
            }
        )

    return {
        "components": components,
        "component_params": component_params,
        "sum_field": np.sum(components, axis=0),
        "r2": r2,
        "aic": aic,
    }
