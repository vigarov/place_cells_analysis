"""Low-rank BTSP adapter on selected linear weights for alt training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

DEFAULT_ALT_SPLIT_TARGETS: tuple[str, ...] = (
    "recurrent_layers.0.leaky_layer.linear_layer",
)


@dataclass(frozen=True)
class SplitWeightTarget:
    """Dotted module path to a linear layer receiving a low-rank BTSP adapter."""

    module_path: str


@dataclass
class _InstalledSplit:
    linear: nn.Module
    module_path: str
    rank: int


class SplitWeightWrapper:
    """Add a low-rank BTSP pathway: W_eff = W + beta * (B @ A).

    W (base weight) and bias are updated by the trial optimizer. LoRA factors B (out×rank)
    and A (rank×in) are updated via plain SGD in `AltTrainingOptimizerWrapper`, optionally
    after per-unit grad scaling from g_local_mse.
    """

    def __init__(
        self,
        model: nn.Module,
        targets: Sequence[SplitWeightTarget | str],
        beta: float,
        rank: int,
    ) -> None:
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        self._model = model
        self._targets = [
            SplitWeightTarget(module_path=t) if isinstance(t, str) else t
            for t in targets
        ]
        self._beta = float(beta)
        self._rank = int(rank)
        self._installed: list[_InstalledSplit] = []

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def rank(self) -> int:
        return self._rank

    def set_beta(self, beta: float) -> None:
        self._beta = float(beta)
        for entry in self._installed:
            entry.linear._split_beta = self._beta  # type: ignore[attr-defined]

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("SplitWeightWrapper.install() called more than once")

        for target in self._targets:
            linear = self._model.get_submodule(target.module_path)
            if hasattr(linear, "weight_lora_A"):
                raise RuntimeError(
                    f"Low-rank BTSP adapter already installed on {target.module_path!r}"
                )
            if not hasattr(linear, "weight") or not hasattr(linear, "bias"):
                raise TypeError(
                    f"Expected linear module with weight/bias at {target.module_path!r}, "
                    f"got {type(linear).__name__}"
                )

            out_dim, in_dim = linear.weight.shape
            if self._rank > min(out_dim, in_dim):
                raise ValueError(
                    f"rank={self._rank} exceeds min(out, in)={min(out_dim, in_dim)} "
                    f"for {target.module_path!r}"
                )

            lora_A = torch.empty(self._rank, in_dim, device=linear.weight.device)
            nn.init.kaiming_uniform_(lora_A, a=5**0.5)
            lora_B = torch.zeros(out_dim, self._rank, device=linear.weight.device)

            linear.register_parameter("weight_lora_A", nn.Parameter(lora_A))
            linear.register_parameter("weight_lora_B", nn.Parameter(lora_B))
            linear._split_beta = self._beta  # type: ignore[attr-defined]

            def split_forward(x: torch.Tensor, *, _linear=linear) -> torch.Tensor:
                beta_t = _linear._split_beta  # type: ignore[attr-defined]
                x_f = x.float()
                # Fused low-rank path: x @ (B @ A).T == (x @ A.T) @ B.T — avoids (H, H) w_delta.
                lora_out = (x_f @ _linear.weight_lora_A.T) @ _linear.weight_lora_B.T
                return x_f @ _linear.weight.T + beta_t * lora_out + _linear.bias

            linear.forward = split_forward  # type: ignore[method-assign]
            self._installed.append(
                _InstalledSplit(
                    linear=linear,
                    module_path=target.module_path,
                    rank=self._rank,
                )
            )

    def lora_parameters(self) -> list[nn.Parameter]:
        """LoRA BTSP factors managed outside the main trial optimizer."""
        params: list[nn.Parameter] = []
        for entry in self._installed:
            params.append(entry.linear.weight_lora_A)
            params.append(entry.linear.weight_lora_B)
        return params

    def scale_btsp_grads(self, scale: float | torch.Tensor) -> None:
        """Scale LoRA BTSP parameter gradients for all installed targets."""
        for entry in self._installed:
            linear = entry.linear
            if linear.weight_lora_B.grad is None and linear.weight_lora_A.grad is None:
                continue

            if isinstance(scale, (float, int)):
                s = float(scale)
                if linear.weight_lora_B.grad is not None:
                    linear.weight_lora_B.grad.mul_(s)
                if linear.weight_lora_A.grad is not None:
                    linear.weight_lora_A.grad.mul_(s)
                continue

            unit_scales = scale
            if unit_scales.ndim != 1:
                raise ValueError(
                    f"Per-unit BTSP scales must be 1-D, got shape {tuple(unit_scales.shape)}"
                )
            n_out = linear.weight_lora_B.shape[0]
            if unit_scales.shape[0] != n_out:
                raise ValueError(
                    f"BTSP scale length {unit_scales.shape[0]} does not match "
                    f"{entry.module_path} output dim {n_out}"
                )
            if linear.weight_lora_B.grad is not None:
                linear.weight_lora_B.grad.mul_(unit_scales[:, None])
            if linear.weight_lora_A.grad is not None:
                mean_scale = float(unit_scales.mean().item())
                linear.weight_lora_A.grad.mul_(mean_scale)


class AltTrainingOptimizerWrapper:
    """Send base weights to the trial optimizer; update LoRA with plain SGD."""

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        split_wrapper: SplitWeightWrapper,
        *,
        lora_lr: float,
    ) -> None:
        self.base_optimizer = base_optimizer
        self.split_wrapper = split_wrapper
        self.lora_lr = float(lora_lr)
        self._lora_params = split_wrapper.lora_parameters()

    @property
    def param_groups(self) -> list[dict]:
        return self.base_optimizer.param_groups

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
        for param in self._lora_params:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()

    def step(self, closure=None):  # noqa: ANN001 — matches torch.optim.Optimizer
        loss = closure() if closure is not None else None
        self.base_optimizer.step()
        with torch.no_grad():
            for param in self._lora_params:
                if param.grad is None:
                    continue
                param.add_(param.grad, alpha=-self.lora_lr)
        return loss


def base_parameters_for_alt_training(
    model: nn.Module,
    split_wrapper: SplitWeightWrapper,
) -> list[nn.Parameter]:
    """Model parameters excluding LoRA BTSP factors."""
    lora_ids = {id(p) for p in split_wrapper.lora_parameters()}
    return [p for p in model.parameters() if id(p) not in lora_ids]
