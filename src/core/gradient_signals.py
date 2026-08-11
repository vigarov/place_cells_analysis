"""Per-timestep and per-update gradient credit-assignment signals for TBPTT training.

Three quantities are captured at every truncated-BPTT training step (segment):

- `G_bptt_MSE[t] = d L_MSE / d h_t`: the true, full backprop-through-time gradient
  of the segment's MSE reconstruction loss w.r.t. the hidden state at each timestep.
  This includes both the direct contribution of `h_t` (through the readout at
  time `t`) and its indirect contribution to every later timestep's loss (through
  recurrence).

- `G_local_MSE[t] = d L_MSE_t / d h_t`: the gradient of only timestep `t`'s own
  reconstruction loss w.r.t. `h_t` -- i.e. the direct/instantaneous contribution
  through the readout only, ignoring recurrence into the future.

- `G_FR = d L_FR / d h`: the gradient of the firing-rate regularization loss.
  Identical for every `(b, t)`, so it is a single `(H,)` vector per update 
  (since we use mean firing rate as regularization)

Subtracting the two MSE gradients, `G_future_MSE = G_bptt_MSE - G_local_MSE`,
isolates the portion of `h_t`'s credit that flows to future timesteps through
recurrence

Normalization note: `G_local_MSE` is normalized to match `F.mse_loss`'s default
`reduction="mean"` (i.e. divided by `B*T*O`, not just `O`), so that the
`G_bptt_MSE - G_local_MSE` subtraction is a valid decomposition of the *same*
training loss (both terms otherwise being on a per-instantaneous-loss scale would
introduce a large, spurious constant offset unrelated to recurrent credit).
"""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class StepGradientSignals:
    """Gradient-based credit-assignment signals for one truncated-BPTT segment.

    `g_bptt_mse` and `g_local_mse` have shape `(B=1, T, H)`; `g_fr` -- `(H,)`
    """

    g_bptt_mse: torch.Tensor
    g_local_mse: torch.Tensor
    g_fr: torch.Tensor

    def g_future_mse(self) -> torch.Tensor:
        return self.g_bptt_mse - self.g_local_mse

    def to_numpy(self) -> dict[str, np.ndarray]:
        return {
            "g_bptt_mse": self.g_bptt_mse.detach().cpu().numpy().astype(np.float32),
            "g_local_mse": self.g_local_mse.detach().cpu().numpy().astype(np.float32),
            "g_fr": self.g_fr.detach().cpu().numpy().astype(np.float32),
        }


def compute_g_fr(states: torch.Tensor, lambda_fr: float) -> torch.Tensor:
    """`d L_FR / d h`, shape `(H,)`.

    `L_FR = lambda_fr * mean_h(mean_{b,t}(h)^2)`, 
    --> gradient w.r.t. `h[b, t, h]` is identical across `(b, t)`:
            2 * lambda_fr * mean_fr[h] / (H * B * T)
    """
    b, t, h = states.shape
    mean_fr = states.mean(dim=(0, 1))  # (H,)
    return 2.0 * lambda_fr * mean_fr / (h * b * t)


def compute_g_local_mse(
    pred: torch.Tensor,
    gt_res: torch.Tensor,
    lambda_mse: float,
    readout_weight: torch.Tensor,
) -> torch.Tensor:
    """`d L_MSE_t / d h_t` via the direct (non-recurrent) readout path only.

    Shape `(B, T, H)`. Matches the normalization of
    `F.mse_loss(pred, gt_res)` (`reduction="mean"`, i.e. divides by `B*T*O`)
    so that `G_bptt_MSE - G_local_MSE` is a valid future-credit decomposition.
    """
    b, t, o = pred.shape
    d_pred = 2.0 * lambda_mse * (pred - gt_res) / (b * t * o)  # (B, T, O)
    return d_pred @ readout_weight  # (B, T, O) @ (O, H) -> (B, T, H)


def compute_g_bptt_mse(states_grad: torch.Tensor, g_fr: torch.Tensor) -> torch.Tensor:
    """Recover the MSE-only BPTT gradient from the combined retained gradient.

    `states_grad` is `states.grad` after
    `(lambda_mse * MSE + lambda_fr * FR).backward()`.

    Since `g_fr` is identical for every `(b, t)`, subtracting it
    (broadcasting over `B, T`) recovers the MSE contribution.
    """
    return states_grad - g_fr


def capture_step_signals(
    *,
    states: torch.Tensor,
    states_grad: torch.Tensor,
    pred: torch.Tensor,
    gt_res: torch.Tensor,
    lambda_mse: float,
    lambda_fr: float,
    readout_weight: torch.Tensor,
) -> StepGradientSignals:
    r"""/!\ Call after `loss.backward()` (with `states.retain_grad()` set before the
    forward pass, on the hidden-state tensor whose `.grad` is passed in as
    `states_grad`) and before `optimizer.step()`.
    """
    g_fr = compute_g_fr(states.detach(), lambda_fr)
    g_local_mse = compute_g_local_mse(
        pred.detach(), gt_res.detach(), lambda_mse, readout_weight.detach()
    )
    g_bptt_mse = compute_g_bptt_mse(states_grad.detach(), g_fr)
    return StepGradientSignals(
        g_bptt_mse=g_bptt_mse, g_local_mse=g_local_mse, g_fr=g_fr
    )
