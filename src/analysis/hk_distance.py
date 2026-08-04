"""
Independent Hellinger-Kantorovich (HK) distance implementation.

This module provides a *from-scratch* HK / Wasserstein-Fisher-Rao distance
computation that does **not** rely on the external `SolveHK` solver. It exists
so the external solver (which can be added to this module as `SolveHK`) can be
validated against an independent reference implementation.

The squared HK distance is computed as an entropy-transport problem
(Liero-Mielke-Savare 2018; Chizat et al. 2018):

    HK^2(a, b) = min_gamma <C, gamma> + KL(gamma_1 | a) + KL(gamma_2 | b)

with the "cone" cost `C(x, y) = -2 log cos_+(d(x, y))` where
`cos_+(s) = cos(min(s, pi/2))` (transport beyond `pi/2` is forbidden, so it
becomes cheaper to create/destroy mass via the KL marginal terms). A length
scale `hk_scale` is handled exactly like the external solver: every distance
is divided by `hk_scale` and the final squared distance is multiplied by
`hk_scale**2`.

The minimisation is solved with an entropic, epsilon-scaled, log-domain
unbalanced Sinkhorn iteration (KL marginals with coefficient `rho = 1`).

Additional methods with **hard source / soft target** semantics
---------------------------------------------------------------
Both variants enforce π₁ = a exactly (all mass from A is transported) while
relaxing the target marginal.

`solve_suot_sinkhorn` — Regularised Semi-Unbalanced OT (RSUOT):

    RSUOT_{ρ,ε}(a, b) = min_{π≥0} <C, π> + ε KL(π | a⊗b) + ι_a(π₁) + ρ KL(π₂|b)

Uses the asymmetric Sinkhorn of Theorem 3.4 in Mignon et al. (SIAM J. Imaging
Sci. 2025).  The target damping factor is λ_g = ρ/(ρ+ε); the source update is
the exact balanced step (λ_f = 1).

`solve_rot_l2` — Semi-relaxed smooth OT (ROT) with squared-L2 target penalty:

    ROT_γ(a, b) = min_{π≥0, π₁=a} <C, π> + (γ/2) ‖π₂ - b‖²

Solved via semi-dual gradient ascent (AGAA).  Produces **sparse** coupling
plans.  Ref: Blondel, Seguy, Rolet (AISTATS 2018, arXiv:1710.06276).

Both methods use the same HK cone cost and are accessible through
`maps_hk_with_plan` via `method="suot"` and `method="rot"`.
"""
import numpy as np
from scipy.special import logsumexp

__all__ = [
    "custom_hk",
    "custom_hk_with_plan",
    "maps_hk_with_plan",
    "solve_hk_sinkhorn",
    "solve_suot_sinkhorn",
    "solve_rot_l2",
    "external_hk_with_plan",
    "cone_cost_matrix",
]

# Optional external solver — assign or import `SolveHK` here when available.
SolveHK = None  # type: ignore[assignment, misc]

_HALF_PI = np.pi / 2.0


def cone_cost_matrix(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    *,
    cos_clip: float = 1e-12,
) -> np.ndarray:
    """
    Cone cost `C_ij = -2 log cos_+(||x_i - y_j||)` between two point sets.

    Positions are assumed already divided by the HK length scale. Pairs at
    distance `>= pi/2` get `+inf` (transport forbidden).
    """
    pos_x = np.asarray(pos_x, dtype=np.float64)
    pos_y = np.asarray(pos_y, dtype=np.float64)
    diff = pos_x[:, None, :] - pos_y[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))

    cost = np.full(dist.shape, np.inf, dtype=np.float64)
    near = dist < _HALF_PI
    cos_vals = np.clip(np.cos(dist[near]), cos_clip, 1.0)
    cost[near] = -2.0 * np.log(cos_vals)
    return cost


def _eps_schedule(
    eps_init: float, eps_target: float, factor: float
) -> list[float]:
    """Descending geometric epsilon schedule ending exactly at `eps_target`."""
    eps = max(float(eps_init), float(eps_target))
    schedule: list[float] = []
    while eps > eps_target:
        schedule.append(eps)
        eps *= factor
    schedule.append(float(eps_target))
    return schedule


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """Generalised KL divergence `sum p log(p/q) - sum p + sum q` (p, q >= 0)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    pos = p > 0
    div = float(np.sum(p[pos] * (np.log(p[pos]) - np.log(q[pos]))))
    return div - float(p.sum()) + float(q.sum())


def _hk_primal_value(
    plan: np.ndarray, cost: np.ndarray, a: np.ndarray, b: np.ndarray
) -> float:
    """Unregularised entropy-transport objective for a coupling `plan`."""
    mask = plan > 0
    transport = float(np.sum(plan[mask] * cost[mask])) if mask.any() else 0.0
    gamma1 = plan.sum(axis=1)
    gamma2 = plan.sum(axis=0)
    return transport + _kl(gamma1, a) + _kl(gamma2, b)


def solve_hk_sinkhorn(
    mu_x: np.ndarray,
    pos_x: np.ndarray,
    mu_y: np.ndarray,
    pos_y: np.ndarray,
    hk_scale: float = 1.0,
    sinkhorn_error: float = 1e-3,
    eps_target: float = 1e-2,
    eps_init: float = 1.0,
    *,
    rho: float = 1.0,
    max_iter: int = 1000,
    eps_scaling_factor: float = 0.5,
    mass_eps: float = 1e-12,
    return_plan: bool = False,
):
    """
    Squared HK distance between weighted point clouds via unbalanced Sinkhorn.

    The positional arguments mirror the external `SolveHK`: `(muX, posX, muY,
    posY, HKScale, SinkhornError, epsTarget, epsInit)`. Positions are divided by
    `hk_scale` and the final value multiplied by `hk_scale**2`.

    The problem is solved in the log domain with epsilon scaling from
    `eps_init` down to `eps_target` (reduction factor `eps_scaling_factor`,
    ~0.5). The inner loop stops when the largest change of the `f` potential
    falls below `sinkhorn_error` (or after `max_iter` sweeps). The KL
    marginal coefficient `rho` is 1 for the standard HK distance.

    Returns `value` (approximate squared HK distance), or `(value, plan)`
    with the dense coupling matrix when `return_plan` is set.
    """
    mu_x = np.asarray(mu_x, dtype=np.float64).ravel()
    mu_y = np.asarray(mu_y, dtype=np.float64).ravel()
    pos_x = np.asarray(pos_x, dtype=np.float64).reshape(mu_x.shape[0], -1)
    pos_y = np.asarray(pos_y, dtype=np.float64).reshape(mu_y.shape[0], -1)

    keep_x = mu_x > mass_eps
    keep_y = mu_y > mass_eps
    mu_x, pos_x = mu_x[keep_x], pos_x[keep_x]
    mu_y, pos_y = mu_y[keep_y], pos_y[keep_y]
    if mu_x.shape[0] == 0 or mu_y.shape[0] == 0:
        empty = np.zeros((mu_x.shape[0], mu_y.shape[0]))
        value = (float(mu_x.sum()) + float(mu_y.sum())) * hk_scale**2
        return (value, empty) if return_plan else value

    cost = cone_cost_matrix(pos_x / hk_scale, pos_y / hk_scale)
    log_a = np.log(mu_x)
    log_b = np.log(mu_y)
    f = np.zeros(mu_x.shape[0])
    g = np.zeros(mu_y.shape[0])

    for eps in _eps_schedule(eps_init, eps_target, eps_scaling_factor):
        lam = rho / (rho + eps)
        for _ in range(max_iter):
            f_prev = f
            with np.errstate(invalid="ignore"):
                log_kv = logsumexp((g[None, :] - cost) / eps, axis=1)
                log_kv = np.where(np.isfinite(log_kv), log_kv, -np.inf)
                f = np.where(
                    np.isfinite(log_kv), lam * eps * (log_a - log_kv), 0.0
                )
                log_ku = logsumexp((f[:, None] - cost) / eps, axis=0)
                log_ku = np.where(np.isfinite(log_ku), log_ku, -np.inf)
                g = np.where(
                    np.isfinite(log_ku), lam * eps * (log_b - log_ku), 0.0
                )
            if np.max(np.abs(f - f_prev)) < sinkhorn_error * max(1.0, eps):
                break

    plan = np.exp((f[:, None] + g[None, :] - cost) / eps_target)
    value = _hk_primal_value(plan, cost, mu_x, mu_y) * hk_scale**2
    return (value, plan) if return_plan else value


def _suot_primal_value(
    plan: np.ndarray,
    cost: np.ndarray,
    b: np.ndarray,
    rho: float,
) -> float:
    """SUOT primal: ⟨C, π⟩ + ρ·KL(π₂|b).  Source KL term is 0 (hard π₁=a)."""
    mask = plan > 0
    transport = float(np.sum(plan[mask] * cost[mask])) if mask.any() else 0.0
    gamma2 = plan.sum(axis=0)
    return transport + rho * _kl(gamma2, b)


def solve_suot_sinkhorn(
    mu_x: np.ndarray,
    pos_x: np.ndarray,
    mu_y: np.ndarray,
    pos_y: np.ndarray,
    hk_scale: float = 1.0,
    sinkhorn_error: float = 1e-3,
    eps_target: float = 1e-2,
    eps_init: float = 1.0,
    *,
    rho: float = 1.0,
    max_iter: int = 1000,
    eps_scaling_factor: float = 0.5,
    mass_eps: float = 1e-12,
    return_plan: bool = False,
):
    """
    Regularised Semi-Unbalanced OT (RSUOT) with cone cost — entropy Sinkhorn.

    **Hard source** (π₁ = a exactly) and **soft target** (ρ·KL(π₂|b)):

        RSUOT_{ρ,ε}(a,b) = min_{π≥0} ⟨C,π⟩ + ε KL(π|a⊗b) + ι_a(π₁) + ρ KL(π₂|b)

    The asymmetric Sinkhorn updates (Theorem 3.4 of Mignon et al. 2025) are:
      - source potential f: balanced update (λ_f = 1, hard source enforced)
      - target potential g: unbalanced update (λ_g = ρ/(ρ+ε), soft KL target)

    The cost C is the HK cone cost `C_{ij} = -2 log cos_+(d(x_i,y_j)/hk_scale)`.
    Larger `rho` → closer to balanced transport; rho → ∞ recovers standard HK.

    When M(A) > M(B) the extra source mass is absorbed at B (local creation).
    When M(A) < M(B) some B mass is locally destroyed (not reached by A).

    Returns `value` ≈ ⟨C,π⟩ + ρ·KL(π₂|b) scaled by `hk_scale²`, or
    `(value, plan)` when `return_plan` is set.

    Reference:
        Mignon, Galerne, Hidane, Louchet, Mille (2025).
        "Semi-Unbalanced OT for Reference-Based Image Restoration and Synthesis."
        SIAM J. Imaging Sci. 18(2):1372-1416.  Theorem 3.4.
    """
    mu_x = np.asarray(mu_x, dtype=np.float64).ravel()
    mu_y = np.asarray(mu_y, dtype=np.float64).ravel()
    pos_x = np.asarray(pos_x, dtype=np.float64).reshape(mu_x.shape[0], -1)
    pos_y = np.asarray(pos_y, dtype=np.float64).reshape(mu_y.shape[0], -1)

    keep_x = mu_x > mass_eps
    keep_y = mu_y > mass_eps
    mu_x, pos_x = mu_x[keep_x], pos_x[keep_x]
    mu_y, pos_y = mu_y[keep_y], pos_y[keep_y]

    if mu_x.shape[0] == 0 or mu_y.shape[0] == 0:
        empty = np.zeros((mu_x.shape[0], mu_y.shape[0]))
        # Source is hard: KL(0|a) = sum(a); target is soft: rho*KL(0|b) = rho*sum(b)
        value = (float(mu_x.sum()) + rho * float(mu_y.sum())) * hk_scale**2
        return (value, empty) if return_plan else value

    cost = cone_cost_matrix(pos_x / hk_scale, pos_y / hk_scale)
    log_a = np.log(mu_x)
    log_b = np.log(mu_y)
    f = np.zeros(mu_x.shape[0])
    g = np.zeros(mu_y.shape[0])

    for eps in _eps_schedule(eps_init, eps_target, eps_scaling_factor):
        lam_g = rho / (rho + eps)   # soft-target factor; source uses lam_f = 1
        for _ in range(max_iter):
            f_prev = f
            with np.errstate(invalid="ignore"):
                # Hard source: balanced Sinkhorn step (λ_f = 1)
                log_kv = logsumexp((g[None, :] - cost) / eps, axis=1)
                log_kv = np.where(np.isfinite(log_kv), log_kv, -np.inf)
                f = np.where(
                    np.isfinite(log_kv), eps * (log_a - log_kv), 0.0
                )
                # Soft target: unbalanced KL step (λ_g = ρ/(ρ+ε))
                log_ku = logsumexp((f[:, None] - cost) / eps, axis=0)
                log_ku = np.where(np.isfinite(log_ku), log_ku, -np.inf)
                g = np.where(
                    np.isfinite(log_ku), lam_g * eps * (log_b - log_ku), 0.0
                )
            if np.max(np.abs(f - f_prev)) < sinkhorn_error * max(1.0, eps):
                break

    # One final balanced f-update using the last g so that π₁ = a exactly.
    with np.errstate(invalid="ignore"):
        log_kv = logsumexp((g[None, :] - cost) / eps_target, axis=1)
        log_kv = np.where(np.isfinite(log_kv), log_kv, -np.inf)
        f = np.where(np.isfinite(log_kv), eps_target * (log_a - log_kv), 0.0)

    plan = np.exp((f[:, None] + g[None, :] - cost) / eps_target)
    value = _suot_primal_value(plan, cost, mu_y, rho) * hk_scale**2
    return (value, plan) if return_plan else value


def _proj_rows_scaled_simplex(V: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Project each row of V [n_rows × n_cols] onto the scaled simplex
    {t ≥ 0 : ∑ t = z_i} in a single vectorised pass O(n_rows · n_cols · log n_cols).
    """
    n_rows, n_cols = V.shape
    z = np.asarray(z, dtype=np.float64)
    out = np.zeros_like(V)
    active = z > 0
    if not active.any():
        return out
    Va = V[active]
    za = z[active]
    na = Va.shape[0]
    # Sort rows descending
    U = np.sort(Va, axis=1)[:, ::-1]
    cssv = np.cumsum(U, axis=1)
    idx = np.arange(1, n_cols + 1, dtype=np.float64)
    # rho[i] = number of elements to keep for row i
    rho = np.maximum(np.sum(U * idx - cssv + za[:, None] > 0, axis=1), 1)
    theta = (cssv[np.arange(na), rho - 1] - za) / rho.astype(np.float64)
    out[active] = np.maximum(Va - theta[:, None], 0.0)
    return out


def solve_rot_l2(
    mu_x: np.ndarray,
    pos_x: np.ndarray,
    mu_y: np.ndarray,
    pos_y: np.ndarray,
    hk_scale: float = 1.0,
    gamma: float = 1.0,
    *,
    max_iter: int = 300,
    tol: float = 1e-4,
    mass_eps: float = 1e-12,
    return_plan: bool = False,
):
    """
    Semi-relaxed smooth OT (ROT) with cone cost and squared-L2 target penalty.

    **Hard source** (π₁ = a exactly at every iterate), **soft target** with
    squared-Euclidean penalty:

        ROT_γ(a,b) = min_{π≥0, π₁=a} ⟨C, π⟩ + (γ/2) ‖π₂ - b‖²

    Larger `gamma` → target marginal is penalised more strongly → plan closer to
    balanced transport.  gamma → ∞ recovers hard balanced OT.

    Solved by **FISTA projected gradient** (Beck & Teboulle 2009): gradient descent
    on the smooth objective F(π) = ⟨C,π⟩ + (γ/2)‖π₂-b‖² followed by projection
    of each row onto the scaled simplex {t≥0, ∑t=a_i}.  The hard source constraint
    is satisfied exactly at every iteration.  Forbidden cone-cost pairs (C_{ij}=∞)
    are masked to 0 after each projection step.

    Lipschitz constant: L = γ·|A|  (Hessian spectral norm), step size η = 1/L.
    Convergence: O(1/T²) in primal gap (Nesterov-accelerated).

    The cost C is the HK cone cost `C_{ij} = -2 log cos_+(d(x_i,y_j)/hk_scale)`.

    Returns `value` = ⟨C,π⟩ + (γ/2)‖π₂-b‖² scaled by `hk_scale²`, or
    `(value, plan)` when `return_plan` is set.

    Reference:
        Blondel, Seguy, Rolet (2018). "Smooth and Sparse Optimal Transport."
        AISTATS 2018.  arXiv:1710.06276.  Definition 3 (Semi-relaxed primal).
    """
    mu_x = np.asarray(mu_x, dtype=np.float64).ravel()
    mu_y = np.asarray(mu_y, dtype=np.float64).ravel()
    pos_x = np.asarray(pos_x, dtype=np.float64).reshape(mu_x.shape[0], -1)
    pos_y = np.asarray(pos_y, dtype=np.float64).reshape(mu_y.shape[0], -1)

    keep_x = mu_x > mass_eps
    keep_y = mu_y > mass_eps
    mu_x, pos_x = mu_x[keep_x], pos_x[keep_x]
    mu_y, pos_y = mu_y[keep_y], pos_y[keep_y]

    if mu_x.shape[0] == 0 or mu_y.shape[0] == 0:
        empty = np.zeros((mu_x.shape[0], mu_y.shape[0]))
        # Hard source has no target to flow into; L2 target penalty: (γ/2)‖b‖²
        value = (gamma / 2.0) * float(np.sum(mu_y**2)) * hk_scale**2
        return (value, empty) if return_plan else value

    cost = cone_cost_matrix(pos_x / hk_scale, pos_y / hk_scale)
    n_x, n_y = mu_x.shape[0], mu_y.shape[0]
    feasible = np.isfinite(cost)  # pairs reachable within cone

    # Replace inf costs with a large finite sentinel for gradient arithmetic.
    # Infeasible pairs are forced to 0 after every projection step.
    cost_finite = np.where(feasible, cost, 1e8)

    # Lipschitz constant of ∇F.
    # ∇F(π)_{ij} = C_{ij} + γ(∑_k π_{kj} - b_j): the change γ·Δ_j broadcasts to
    # all n_x rows, so the spectral norm of the Hessian is γ·n_x.
    L = gamma * float(n_x)
    eta = 1.0 / L

    # Warm start: distribute each source uniformly over its feasible targets
    n_feasible = feasible.sum(axis=1).astype(np.float64)  # [n_x]
    valid_rows = n_feasible > 0
    pi = np.zeros((n_x, n_y))
    pi[valid_rows] = (
        feasible[valid_rows].astype(np.float64)
        * (mu_x[valid_rows] / n_feasible[valid_rows])[:, None]
    )
    # Sources with no feasible target contribute only the L2 penalty on b

    y = pi.copy()
    t_fista = 1.0

    for _ in range(max_iter):
        # Gradient: ∇F(y)_{ij} = C_{ij} + γ(∑_i y_{ij} - b_j)
        tau = y.sum(axis=0)                                  # [n_y]
        grad = cost_finite + gamma * (tau - mu_y)[None, :]  # [n_x, n_y]

        # Gradient step + row-wise simplex projection
        pi_new = _proj_rows_scaled_simplex(y - eta * grad, mu_x)

        # Zero out forbidden pairs and renormalise rows to restore π₁ = a
        pi_new = np.where(feasible, pi_new, 0.0)
        row_sums = pi_new.sum(axis=1)
        valid = row_sums > mass_eps
        if valid.any():
            pi_new[valid] *= (mu_x[valid] / row_sums[valid])[:, None]

        # FISTA momentum
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_fista * t_fista))
        y = pi_new + ((t_fista - 1.0) / t_new) * (pi_new - pi)

        if np.max(np.abs(pi_new - pi)) < tol:
            pi = pi_new
            break
        pi = pi_new
        t_fista = t_new

    # Primal value: ⟨C, π⟩ + (γ/2) ‖π₂ - b‖²
    tau = pi.sum(axis=0)
    mask = pi > 0
    transport = float(np.sum(pi[mask] * cost[mask])) if mask.any() else 0.0
    l2_pen = 0.5 * gamma * float(np.sum((tau - mu_y) ** 2))
    value = (transport + l2_pen) * hk_scale**2
    return (value, pi) if return_plan else value


def _maps_to_point_clouds(
    map1: np.ndarray,
    map2: np.ndarray,
    *,
    mass_eps: float = 1e-12,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract positive-mass pixel particles from two maps."""
    map1 = np.asarray(map1, dtype=np.float64)
    map2 = np.asarray(map2, dtype=np.float64)

    pos1 = np.indices(map1.shape).reshape(map1.ndim, -1).T.astype(np.float64)
    pos2 = np.indices(map2.shape).reshape(map2.ndim, -1).T.astype(np.float64)
    flat1 = np.clip(map1.ravel(), 0.0, None)
    flat2 = np.clip(map2.ravel(), 0.0, None)

    if normalize:
        if flat1.sum() > mass_eps:
            flat1 = flat1 / flat1.sum()
        if flat2.sum() > mass_eps:
            flat2 = flat2 / flat2.sum()

    keep1 = flat1 > mass_eps
    keep2 = flat2 > mass_eps
    return flat1[keep1], pos1[keep1], flat2[keep2], pos2[keep2]


def external_hk_with_plan(
    mu_x: np.ndarray,
    pos_x: np.ndarray,
    mu_y: np.ndarray,
    pos_y: np.ndarray,
    hk_scale: float = 1.0,
    sinkhorn_error: float = 1e-3,
    eps_target: float = 1e-2,
    eps_init: float = 1.0,
) -> tuple[float, np.ndarray]:
    """
    Squared HK distance and coupling plan via the external `SolveHK` solver.

    Expects `SolveHK` to be defined in this module (same signature as documented
    in the notebook). Returns `(hk2, plan)` with a dense coupling matrix.
    """
    if SolveHK is None:
        raise RuntimeError(
            "SolveHK is not available; define it in analysis.hk_distance "
            "before using method='solve_hk'."
        )
    value, pi_csr, *_ = SolveHK(
        mu_x,
        pos_x,
        mu_y,
        pos_y,
        hk_scale,
        sinkhorn_error,
        eps_target,
        eps_init,
    )
    if hasattr(pi_csr, "toarray"):
        plan = pi_csr.toarray()
    else:
        plan = np.asarray(pi_csr, dtype=np.float64)
    return float(value), plan


def maps_hk_with_plan(
    map1: np.ndarray,
    map2: np.ndarray,
    *,
    method: str = "custom",
    hk_scale: float = 1.0,
    sinkhorn_error: float = 1e-3,
    eps_target: float = 1e-2,
    eps_init: float = 1.0,
    mass_eps: float = 1e-12,
    normalize: bool = False,
    max_iter: int = 1000,
    **solver_kwargs,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    HK (or semi-unbalanced) distance between two maps with the coupling plan.

    `method` selects the backend:

    * `"custom"`   — independent :func:`solve_hk_sinkhorn` (symmetric HK).
    * `"suot"`     — :func:`solve_suot_sinkhorn` (hard source, soft KL target).
      Pass `rho=<float>` via kwargs to control the KL target penalty strength.
    * `"rot"`      — :func:`solve_rot_l2` (hard source, soft L2 target, sparse plan).
      Pass `gamma=<float>` via kwargs to control the L2 target penalty strength.
    * `"solve_hk"` — external :func:`external_hk_with_plan` / `SolveHK`.

    For `"suot"` and `"rot"`, all mass from map1 (source A) is forced into
    the transport plan; map2 (target B) may locally gain or lose mass depending
    on M(A) vs M(B) and the penalty strength.

    Returns `(value, plan, pos_a, pos_b, mu_a, mu_b)`.
    """
    mu_a, pos_a, mu_b, pos_b = _maps_to_point_clouds(
        map1, map2, mass_eps=mass_eps, normalize=normalize
    )
    if method == "custom":
        hk2, plan = solve_hk_sinkhorn(
            mu_a,
            pos_a,
            mu_b,
            pos_b,
            hk_scale=hk_scale,
            sinkhorn_error=sinkhorn_error,
            eps_target=eps_target,
            eps_init=eps_init,
            mass_eps=mass_eps,
            max_iter=max_iter,
            return_plan=True,
            **solver_kwargs,
        )
    elif method == "suot":
        hk2, plan = solve_suot_sinkhorn(
            mu_a,
            pos_a,
            mu_b,
            pos_b,
            hk_scale=hk_scale,
            sinkhorn_error=sinkhorn_error,
            eps_target=eps_target,
            eps_init=eps_init,
            mass_eps=mass_eps,
            max_iter=max_iter,
            return_plan=True,
            **solver_kwargs,
        )
    elif method == "rot":
        hk2, plan = solve_rot_l2(
            mu_a,
            pos_a,
            mu_b,
            pos_b,
            hk_scale=hk_scale,
            mass_eps=mass_eps,
            max_iter=max_iter,
            return_plan=True,
            **solver_kwargs,
        )
    elif method == "solve_hk":
        hk2, plan = external_hk_with_plan(
            mu_a,
            pos_a,
            mu_b,
            pos_b,
            hk_scale=hk_scale,
            sinkhorn_error=sinkhorn_error,
            eps_target=eps_target,
            eps_init=eps_init,
        )
    else:
        raise ValueError(
            f"Unknown HK method {method!r};"
            " expected 'custom', 'suot', 'rot', or 'solve_hk'."
        )
    return float(hk2), plan, pos_a, pos_b, mu_a, mu_b


def custom_hk(
    map1: np.ndarray,
    map2: np.ndarray,
    *,
    hk_scale: float = 1.0,
    sinkhorn_error: float = 1e-3,
    eps_target: float = 1e-2,
    eps_init: float = 1.0,
    mass_eps: float = 1e-12,
    normalize: bool = False,
    **solver_kwargs,
) -> float:
    """
    Squared HK distance between two 2D maps, computed independently of `SolveHK`.

    Each map becomes a weighted point cloud: one particle per pixel positioned at
    its grid index (render-pixel units) with mass equal to the (non-negative)
    pixel value. Negligible-mass pixels are dropped. When `normalize` is set,
    each map's masses are scaled to sum to 1 before solving.
    """
    hk2, *_ = maps_hk_with_plan(
        map1,
        map2,
        method="custom",
        hk_scale=hk_scale,
        sinkhorn_error=sinkhorn_error,
        eps_target=eps_target,
        eps_init=eps_init,
        mass_eps=mass_eps,
        normalize=normalize,
        **solver_kwargs,
    )
    return hk2


def custom_hk_with_plan(
    map1: np.ndarray,
    map2: np.ndarray,
    *,
    hk_scale: float = 1.0,
    sinkhorn_error: float = 1e-3,
    eps_target: float = 1e-2,
    eps_init: float = 1.0,
    mass_eps: float = 1e-12,
    normalize: bool = False,
    **solver_kwargs,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Like :func:`custom_hk` but also returns the entropic coupling plan.

    Returns `(hk2, plan, pos_a, pos_b, mu_a, mu_b)` where `plan[i, j]` is
    the mass transported from source particle `i` to target `j`.
    """
    return maps_hk_with_plan(
        map1,
        map2,
        method="custom",
        hk_scale=hk_scale,
        sinkhorn_error=sinkhorn_error,
        eps_target=eps_target,
        eps_init=eps_init,
        mass_eps=mass_eps,
        normalize=normalize,
        **solver_kwargs,
    )
