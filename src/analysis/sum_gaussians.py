"""Device-aware entry point for sum-of-Gaussians receptive-field fitting."""


def fit_sum_gaussians(
    field,
    n_gaussians=2,
    device="cpu",
    *,
    maxfev=20_000,
    n_steps=3_000,
    lr=0.05,
    early_stop_patience=30,
    early_stop_eps=1e-8,
):
    """
    Fit a sum of independent 2D Gaussians.

    Parameters
    ----------
    device : {'cpu', 'gpu'}
        `'cpu'` uses `scipy.optimize.curve_fit`; `'gpu'` trains
        :class:`~analysis.sum_gaussians_torch.SumGaussians2D` with gradient
        descent on CUDA/MPS (requires PyTorch).
    early_stop_patience, early_stop_eps
        GPU only. Stop when train MSE changes by at most `early_stop_eps` for
        `early_stop_patience` consecutive steps.
    """
    if device == "cpu":
        from analysis.sum_gaussians_core import fit_sum_gaussians as fit_cpu

        return fit_cpu(field, n_gaussians, maxfev=maxfev)
    if device == "gpu":
        from analysis.sum_gaussians_torch import fit_sum_gaussians as fit_gpu

        return fit_gpu(
            field,
            n_gaussians,
            n_steps=n_steps,
            lr=lr,
            early_stop_patience=early_stop_patience,
            early_stop_eps=early_stop_eps,
        )
    raise ValueError("device must be 'cpu' or 'gpu'")
