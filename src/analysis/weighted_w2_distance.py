"""Weighted W2 distance between maps via matched sum-of-Gaussian fits."""
import numpy as np
import ot

from analysis.sum_gaussians_core import fit_sum_gaussians

__all__ = ["weighted_w2_distance"]


def _fit_params(field: np.ndarray, n_gaussians: int) -> np.ndarray:
    """Return fitted parameters as (n_gaussians, 5)."""
    fit = fit_sum_gaussians(field, n_gaussians=n_gaussians)
    if fit is None:
        raise RuntimeError(f"Gaussian fit failed for K={n_gaussians}")

    params = np.array(
        [
            [
                p["amplitude"],
                p["mu_x"],
                p["mu_y"],
                p["sigma_x"],
                p["sigma_y"],
            ]
            for p in fit["component_params"]
        ],
        dtype=np.float64,
    )
    return params


def _params_to_mean_cov(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert one component to mean (2,) and covariance (2, 2)."""
    amp, mu_x, mu_y, sigma_x, sigma_y = params
    mean = np.array([mu_y, mu_x], dtype=np.float64)
    cov = np.diag([sigma_y**2, sigma_x**2]).astype(np.float64)
    return mean, cov


def _center_mse(params_a: np.ndarray, params_b: np.ndarray) -> float:
    """Squared distance between Gaussian centers."""
    dx = params_a[1] - params_b[1]
    dy = params_a[2] - params_b[2]
    return float(dx * dx + dy * dy)


def _moment_match(params_list: list[np.ndarray], amplitudes: np.ndarray) -> np.ndarray:
    """Amplitude-weighted moment matching for a cluster of components."""
    amps = np.asarray(amplitudes, dtype=np.float64)
    total = amps.sum()
    if total <= 0:
        raise ValueError("Cluster amplitudes must sum to a positive value")

    weights = amps / total
    means = np.array([_params_to_mean_cov(p)[0] for p in params_list], dtype=np.float64)
    covs = np.array([_params_to_mean_cov(p)[1] for p in params_list], dtype=np.float64)

    mean_agg = np.sum(weights[:, None] * means, axis=0)
    cov_agg = np.zeros((2, 2), dtype=np.float64)
    for w, mean, cov in zip(weights, means, covs, strict=True):
        diff = mean - mean_agg
        cov_agg += w * (cov + np.outer(diff, diff))

    sigma_y = np.sqrt(max(cov_agg[0, 0], 1e-6))
    sigma_x = np.sqrt(max(cov_agg[1, 1], 1e-6))
    amp_agg = float(total)
    return np.array(
        [amp_agg, mean_agg[1], mean_agg[0], sigma_x, sigma_y],
        dtype=np.float64,
    )


def _w2_between(params_a: np.ndarray, params_b: np.ndarray) -> float:
    """Closed-form W2 between two diagonal 2D Gaussians."""
    mean_a, cov_a = _params_to_mean_cov(params_a)
    mean_b, cov_b = _params_to_mean_cov(params_b)
    return float(ot.gaussian.bures_wasserstein_distance(mean_a, mean_b, cov_a, cov_b))


def _greedy_match_equal(
    params_a: np.ndarray,
    params_b: np.ndarray,
) -> list[tuple[int, int]]:
    """Greedy 1-to-1 matching sorted by A amplitudes (descending)."""
    k = params_a.shape[0]
    order_a = np.argsort(-params_a[:, 0])
    used_b: set[int] = set()
    pairs: list[tuple[int, int]] = []

    for i_a in order_a:
        best_j = None
        best_cost = np.inf
        for j_b in range(k):
            if j_b in used_b:
                continue
            cost = _center_mse(params_a[i_a], params_b[j_b])
            if cost < best_cost:
                best_cost = cost
                best_j = j_b
        if best_j is None:
            raise RuntimeError("Failed to find unmatched B component")
        used_b.add(best_j)
        pairs.append((int(i_a), int(best_j)))

    return pairs


def _match_a_to_b_unequal(
    params_a: np.ndarray,
    params_b: np.ndarray,
) -> list[list[int]]:
    """Match when K_A <= K_B: clusters of B indices assigned to each A index."""
    k_a, k_b = params_a.shape[0], params_b.shape[0]
    pairs = _greedy_match_equal(params_a, params_b)
    clusters: list[list[int]] = [[] for _ in range(k_a)]
    matched_b: set[int] = set()

    for i_a, j_b in pairs:
        clusters[i_a].append(j_b)
        matched_b.add(j_b)

    for j_b in range(k_b):
        if j_b in matched_b:
            continue
        best_i = min(range(k_a), key=lambda i: _center_mse(params_a[i], params_b[j_b]))
        clusters[best_i].append(j_b)

    return clusters


def _match_b_to_a_unequal(
    params_a: np.ndarray,
    params_b: np.ndarray,
) -> list[list[int]]:
    """Match when K_A > K_B: clusters of A indices assigned to each B index."""
    k_a, k_b = params_a.shape[0], params_b.shape[0]
    order_b = np.argsort(-params_b[:, 0])
    used_a: set[int] = set()
    pairs: list[tuple[int, int]] = []

    for j_b in order_b:
        best_i = None
        best_cost = np.inf
        for i_a in range(k_a):
            if i_a in used_a:
                continue
            cost = _center_mse(params_a[i_a], params_b[j_b])
            if cost < best_cost:
                best_cost = cost
                best_i = i_a
        if best_i is None:
            raise RuntimeError("Failed to find unmatched A component")
        used_a.add(best_i)
        pairs.append((best_i, int(j_b)))

    clusters: list[list[int]] = [[] for _ in range(k_b)]
    matched_a: set[int] = set()
    for i_a, j_b in pairs:
        clusters[j_b].append(i_a)
        matched_a.add(i_a)

    for i_a in range(k_a):
        if i_a in matched_a:
            continue
        best_j = min(range(k_b), key=lambda j: _center_mse(params_a[i_a], params_b[j]))
        clusters[best_j].append(i_a)

    return clusters


def weighted_w2_distance(
    map_a: np.ndarray,
    map_b: np.ndarray,
    *,
    n_gaussians_a: int,
    n_gaussians_b: int,
) -> float:
    """
    Amplitude-weighted W2 distance between two maps via Gaussian matching.

    Each map is fit with a sum of ``n_gaussians_*`` independent 2D Gaussians.
    Components are matched by center distance (greedy, amplitude-sorted on the
    smaller side for unequal K). When multiple components map to one reference
    component, they are merged via amplitude-weighted moment matching before
    computing W2 with POT's closed-form Bures-Wasserstein distance.
    """
    params_a = _fit_params(map_a, n_gaussians_a)
    params_b = _fit_params(map_b, n_gaussians_b)

    k_a, k_b = n_gaussians_a, n_gaussians_b
    distances: list[float] = []
    weights: list[float] = []

    if k_a == k_b:
        for i_a, j_b in _greedy_match_equal(params_a, params_b):
            distances.append(_w2_between(params_a[i_a], params_b[j_b]))
            weights.append(float(params_a[i_a, 0]))
    elif k_a < k_b:
        for i_a, cluster_b in enumerate(_match_a_to_b_unequal(params_a, params_b)):
            cluster_params = [params_b[j] for j in cluster_b]
            cluster_amps = params_b[cluster_b, 0]
            if len(cluster_params) == 1:
                target = cluster_params[0]
            else:
                target = _moment_match(cluster_params, cluster_amps)
            distances.append(_w2_between(params_a[i_a], target))
            weights.append(float(params_a[i_a, 0]))
    else:
        for j_b, cluster_a in enumerate(_match_b_to_a_unequal(params_a, params_b)):
            cluster_params = [params_a[i] for i in cluster_a]
            cluster_amps = params_a[cluster_a, 0]
            if len(cluster_params) == 1:
                source = cluster_params[0]
            else:
                source = _moment_match(cluster_params, cluster_amps)
            distances.append(_w2_between(source, params_b[j_b]))
            weights.append(float(params_b[j_b, 0]))

    weights_arr = np.asarray(weights, dtype=np.float64)
    total_weight = weights_arr.sum()
    if total_weight <= 0:
        return float(np.nan)
    norm_weights = weights_arr / total_weight
    return float(np.sum(norm_weights * np.asarray(distances, dtype=np.float64)))
