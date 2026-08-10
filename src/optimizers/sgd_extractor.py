"""SGD optimizer signal extractor.

Per-unit signals:
- grad_norm: L2 norm of the gradient for each unit's incoming weights.
- grad_cosine_sim: cosine similarity between current and previous gradient.
"""
from typing import Any

import torch
from torch import nn

from optimizers.base import OptimizerSignalExtractor, register_extractor
from optimizers.defaults import (
    DEFAULT_LR,
    DEFAULT_SGD_MOMENTUM,
    OPTIMIZER_LR_KEY,
    OPTIMIZER_MOMENTUM_KEY,
    OPTIMIZER_TYPE_KEY,
)


@register_extractor
class SGDExtractor(OptimizerSignalExtractor):

    def __init__(
        self,
        *,
        lr: float = DEFAULT_LR,
        momentum: float = DEFAULT_SGD_MOMENTUM,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.momentum = momentum

    def optimizer_id(self) -> str:
        return f"sgd_lr{self.lr}"

    def optimizer_config(self) -> dict[str, Any]:
        return {
            OPTIMIZER_TYPE_KEY: "sgd",
            OPTIMIZER_LR_KEY: self.lr,
            OPTIMIZER_MOMENTUM_KEY: self.momentum,
        }

    def create_optimizer(self, params) -> torch.optim.Optimizer:
        return torch.optim.SGD(params, lr=self.lr, momentum=self.momentum)

    def signal_names(self) -> list[str]:
        return ["grad", "grad_norm", "grad_cosine_sim"]

    def on_before_step(
        self, model: nn.Module, optimizer: torch.optim.Optimizer,
    ) -> dict[str, dict[str, float]]:
        return self._compute_grad_signals(model)

    def on_after_step(
        self, model: nn.Module, optimizer: torch.optim.Optimizer,
    ) -> dict[str, dict[str, float]]:
        return {}
