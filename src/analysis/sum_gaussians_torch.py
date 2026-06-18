"""PyTorch gradient-descent fitting for sums of 2D Gaussians."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from analysis.sum_gaussians_core import _init_sum_gaussian_params


def _resolve_torch_device(device):
    if device == "cpu":
        return torch.device("cpu")
    if device == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("device='gpu' requested but no CUDA or MPS backend is available")
    raise ValueError("device must be 'cpu' or 'gpu'")


class SumGaussians2D(nn.Module):
    """Sum of independent 2D Gaussians, trainable with gradient descent."""

    def __init__(self, n_gaussians, h, w, init_params):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.h = h
        self.w = w
        p = torch.tensor(init_params, dtype=torch.float32).view(n_gaussians, 5)
        self._raw_amp = nn.Parameter(self._inv_softplus(p[:, 0].clamp(min=1e-6)))
        self._raw_mu_x = nn.Parameter(self._logit(p[:, 1].clamp(min=0.0, max=w - 1) / max(w - 1, 1.0)))
        self._raw_mu_y = nn.Parameter(self._logit(p[:, 2].clamp(min=0.0, max=h - 1) / max(h - 1, 1.0)))
        self._raw_sigma_x = nn.Parameter(self._inv_softplus(p[:, 3].clamp(min=0.5)))
        self._raw_sigma_y = nn.Parameter(self._inv_softplus(p[:, 4].clamp(min=0.5)))

    @staticmethod
    def _inv_softplus(y):
        return y + torch.log(-torch.expm1(-y.clamp(min=1e-6)))

    @staticmethod
    def _logit(p):
        p = p.clamp(1e-6, 1.0 - 1e-6)
        return torch.log(p / (1.0 - p))

    def _constrained_params(self):
        amp = F.softplus(self._raw_amp)
        mu_x = torch.sigmoid(self._raw_mu_x) * max(self.w - 1, 1.0)
        mu_y = torch.sigmoid(self._raw_mu_y) * max(self.h - 1, 1.0)
        sigma_x = F.softplus(self._raw_sigma_x).clamp(min=0.5, max=float(self.w))
        sigma_y = F.softplus(self._raw_sigma_y).clamp(min=0.5, max=float(self.h))
        return amp, mu_x, mu_y, sigma_x, sigma_y

    def forward(self, x, y):
        amp, mu_x, mu_y, sigma_x, sigma_y = self._constrained_params()
        total = torch.zeros_like(x, dtype=torch.float32)
        for k in range(self.n_gaussians):
            total = total + amp[k] * torch.exp(
                -0.5 * ((x - mu_x[k]) / sigma_x[k]) ** 2
                - 0.5 * ((y - mu_y[k]) / sigma_y[k]) ** 2
            )
        return total

    def component_fields(self, xx, yy):
        amp, mu_x, mu_y, sigma_x, sigma_y = self._constrained_params()
        components = []
        component_params = []
        for k in range(self.n_gaussians):
            comp = (
                amp[k]
                * torch.exp(
                    -0.5 * ((xx - mu_x[k]) / sigma_x[k]) ** 2
                    - 0.5 * ((yy - mu_y[k]) / sigma_y[k]) ** 2
                )
            ).detach()
            components.append(comp)
            component_params.append(
                {
                    "amplitude": float(amp[k].detach().cpu()),
                    "mu_x": float(mu_x[k].detach().cpu()),
                    "mu_y": float(mu_y[k].detach().cpu()),
                    "sigma_x": float(sigma_x[k].detach().cpu()),
                    "sigma_y": float(sigma_y[k].detach().cpu()),
                }
            )
        return components, component_params


def fit_sum_gaussians(
    field,
    n_gaussians=2,
    *,
    n_steps=3_000,
    lr=0.05,
    early_stop_patience=50,
    early_stop_eps=1e-8,
):
    """Fit a sum of independent 2D Gaussians with PyTorch gradient descent."""
    torch_device = _resolve_torch_device("gpu")
    mask = np.isfinite(field)
    if not np.any(mask):
        return None

    h, w = field.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x_pts = torch.tensor(xx[mask], dtype=torch.float32, device=torch_device)
    y_pts = torch.tensor(yy[mask], dtype=torch.float32, device=torch_device)
    y_true = torch.tensor(field[mask], dtype=torch.float32, device=torch_device)
    if torch.all(y_true <= 0):
        return None

    p0 = _init_sum_gaussian_params(field, mask, xx, yy, n_gaussians)
    model = SumGaussians2D(n_gaussians, h, w, p0).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    prev_loss = None
    stagnant_steps = 0
    for _ in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        y_pred = model(x_pts, y_pts)
        loss = torch.mean((y_pred - y_true) ** 2)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if prev_loss is not None and abs(prev_loss - loss_val) <= early_stop_eps:
            stagnant_steps += 1
            if stagnant_steps >= early_stop_patience:
                break
        else:
            stagnant_steps = 0
        prev_loss = loss_val

    with torch.no_grad():
        y_pred = model(x_pts, y_pts)
        ss_res = torch.sum((y_true - y_pred) ** 2).item()
        ss_tot = torch.sum((y_true - y_true.mean()) ** 2).item()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    xx_t = torch.tensor(xx, dtype=torch.float32, device=torch_device)
    yy_t = torch.tensor(yy, dtype=torch.float32, device=torch_device)
    components_t, component_params = model.component_fields(xx_t, yy_t)
    components = [comp.cpu().numpy() for comp in components_t]
    sum_field = np.sum(components, axis=0)

    return {
        "components": components,
        "component_params": component_params,
        "sum_field": sum_field,
        "r2": r2,
    }
