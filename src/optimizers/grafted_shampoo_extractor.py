"""GraftedShampoo optimizer signal extractor (AdaGrad-grafted Shampoo).

Shampoo supplies the direction; AdaGrad supplies the step size.

Per-unit signals (gradient signals inherited from base):
- grad_norm, grad_cosine_sim
- h_inv_norm: per-unit ||L^{-1/4}[i, :]||_2 * ||R^{-1/4}||_F (same as PureShampoo)
- effective_lr: per-weight lr / (sqrt(accumulator) + eps) from the
  AdaGrad grafter, analogous to AdaGradExtractor.

Global block signals (via on_after_step_shampoo_blocks):
- h_inv__{param}__L / __R: inverse Kronecker factor matrices (paper notation)
  for each preconditioned parameter (weights and biases).
"""
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from distributed_shampoo import (
    AdaGradPreconditionerConfig,
    DistributedShampoo,
    RootInvShampooPreconditionerConfig,
)
from distributed_shampoo.shampoo_types import SingleDeviceDistributedConfig

from optimizers.base import OptimizerSignalExtractor, register_extractor
from optimizers.defaults import (
    DEFAULT_GRAFTING_EPS,
    DEFAULT_LR,
    DEFAULT_SHAMPOO_BETAS,
    DEFAULT_SHAMPOO_PRECONDITIONER_EPSILON,
    GRAFTING_EPS_KEY,
    OPTIMIZER_BETAS_KEY,
    OPTIMIZER_LR_KEY,
    OPTIMIZER_TYPE_KEY,
    SHAMPOO_PRECONDITIONER_EPSILON_KEY,
)
from optimizers.pure_shampoo_extractor import (
    _compute_h_inv_norms_by_unit,
    _extract_h_inv_blocks,
)

ADAGRAD_KEY = "adagrad"


@register_extractor
class GraftedShampooExtractor(OptimizerSignalExtractor):

    def __init__(
        self,
        *,
        lr: float = DEFAULT_LR,
        betas: tuple[float, float] = DEFAULT_SHAMPOO_BETAS,
        shampoo_preconditioner_epsilon: float = DEFAULT_SHAMPOO_PRECONDITIONER_EPSILON,
        grafting_eps: float = DEFAULT_GRAFTING_EPS,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.betas = betas
        self.shampoo_preconditioner_epsilon = shampoo_preconditioner_epsilon
        self.grafting_eps = grafting_eps
        self._pending_block_signals: dict[str, np.ndarray] = {}

    def optimizer_id(self) -> str:
        return f"grafted_shampoo_lr{self.lr}"

    def optimizer_config(self) -> dict[str, Any]:
        return {
            OPTIMIZER_TYPE_KEY: "grafted_shampoo",
            OPTIMIZER_LR_KEY: self.lr,
            OPTIMIZER_BETAS_KEY: list(self.betas),
            SHAMPOO_PRECONDITIONER_EPSILON_KEY: self.shampoo_preconditioner_epsilon,
            GRAFTING_EPS_KEY: self.grafting_eps,
        }

    def create_optimizer(self, params) -> torch.optim.Optimizer:
        return DistributedShampoo(
            params,
            lr=self.lr,
            betas=self.betas,
            weight_decay=0.0,
            epsilon=self.shampoo_preconditioner_epsilon,
            grafting_config=AdaGradPreconditionerConfig(epsilon=self.grafting_eps),
            preconditioner_config=RootInvShampooPreconditionerConfig(
                inv_factor_matrix_dtype=torch.float64,
            ),
            max_preconditioner_dim=1024,
            distributed_config=SingleDeviceDistributedConfig(
                target_parameter_dimensionality=2,
            ),
        )

    def signal_names(self) -> list[str]:
        return [
            "grad", "grad_norm", "grad_cosine_sim", "h_inv_norm", "effective_lr",
        ]

    def on_before_step(
        self, model: nn.Module, optimizer: torch.optim.Optimizer,
    ) -> dict[str, dict[str, float]]:
        return self._compute_grad_signals(model)

    def _extract_grafting_effective_lr(
        self, model: nn.Module, optimizer: DistributedShampoo,
    ) -> dict[str, np.ndarray]:
        """Per-unit effective LR from the AdaGrad grafting preconditioner.

        The grafting accumulator is stored per blocked parameter inside the
        optimizer state at `state[param]["block_N"]["adagrad"]` as a 1-D
        tensor (the block is flattened). We reconstruct the full
        accumulator by concatenating all blocks, reshape back to the
        original parameter shape, and extract the unit slice.
        """
        eff_lrs: dict[str, np.ndarray] = {}

        for u in self._units:
            nid = u["node_id"]
            param = self._find_weight_param(model, u["layer_name"])
            if param is None:
                eff_lrs[nid] = np.array([float("nan")])
                continue

            param_state = optimizer.state.get(param)
            if not param_state:
                eff_lrs[nid] = np.array([self.lr])
                continue

            block_keys = sorted(
                k for k in param_state
                if isinstance(k, str) and k.startswith("block_")
            )
            accum_slices: list[torch.Tensor] = []
            for bk in block_keys:
                block_state = param_state[bk]
                adagrad_tensor = block_state.get(ADAGRAD_KEY)
                if adagrad_tensor is None:
                    continue
                accum_slices.append(adagrad_tensor.detach().flatten())

            if not accum_slices:
                eff_lrs[nid] = np.array([self.lr])
                continue

            accum_flat = torch.cat(accum_slices)
            accum = accum_flat.reshape(param.shape)
            idx = u["unit_index"]
            unit_accum = accum[idx].flatten().float()
            eff = self.lr / (unit_accum.sqrt() + self.grafting_eps)
            eff_lrs[nid] = eff.cpu().numpy()

        return eff_lrs

    def on_after_step(
        self, model: nn.Module, optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        assert isinstance(optimizer, DistributedShampoo)
        self._pending_block_signals = _extract_h_inv_blocks(
            optimizer, model, units=self._units,
        )
        return {
            "h_inv_norm": _compute_h_inv_norms_by_unit(
                optimizer, model, self._units,
            ),
            "effective_lr": self._extract_grafting_effective_lr(model, optimizer),
        }

    def on_after_step_shampoo_blocks(
        self, model: nn.Module, optimizer: torch.optim.Optimizer,
    ) -> dict[str, np.ndarray]:
        return dict(self._pending_block_signals)
