"""Adam optimizer signal extractor.

Per-unit signals (gradient signals inherited from base):
- grad_norm, grad_cosine_sim
- exp_avg: full first-moment slice for each unit's incoming weights.
- exp_avg_sq: full second-moment slice for each unit.
- moment_cosine_sim: cosine similarity between consecutive first-moment
  snapshots for each unit.
"""
from typing import Any

import numpy as np
import torch

from optimizers.base import OptimizerSignalExtractor, register_extractor
from optimizers.defaults import (
    DEFAULT_ADAM_BETAS,
    DEFAULT_ADAM_EPS,
    DEFAULT_LR,
    OPTIMIZER_BETAS_KEY,
    OPTIMIZER_EPS_KEY,
    OPTIMIZER_LR_KEY,
    OPTIMIZER_TYPE_KEY,
)


@register_extractor
class AdamExtractor(OptimizerSignalExtractor):

    def __init__(
        self,
        *,
        lr: float = DEFAULT_LR,
        betas: tuple[float, float] = DEFAULT_ADAM_BETAS,
        eps: float = DEFAULT_ADAM_EPS,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self._prev_exp_avg: dict[str, torch.Tensor] = {}

    def optimizer_id(self) -> str:
        return f"adam_lr{self.lr}"

    def optimizer_config(self) -> dict[str, Any]:
        return {
            OPTIMIZER_TYPE_KEY: "adam",
            OPTIMIZER_LR_KEY: self.lr,
            OPTIMIZER_BETAS_KEY: list(self.betas),
            OPTIMIZER_EPS_KEY: self.eps,
        }

    def create_optimizer(self, params) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=self.lr, betas=self.betas, eps=self.eps)

    def signal_names(self) -> list[str]:
        return [
            "grad",
            "grad_norm",
            "grad_cosine_sim",
            "exp_avg",
            "exp_avg_sq",
            "moment_cosine_sim",
        ]

    def on_before_step(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    ) -> dict[str, dict[str, float]]:
        return self._compute_grad_signals(model)

    def on_after_step(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    ) -> dict[str, dict[str, Any]]:
        exp_avgs: dict[str, np.ndarray] = {}
        exp_avg_sqs: dict[str, np.ndarray] = {}
        moment_cos: dict[str, float] = {}

        for u in self._units:
            nid = u["node_id"]
            param = self._find_weight_param(model, u["layer_name"])
            if param is None:
                exp_avgs[nid] = np.array([float("nan")])
                exp_avg_sqs[nid] = np.array([float("nan")])
                moment_cos[nid] = float("nan")
                continue

            state = optimizer.state.get(param)
            if not state or "exp_avg" not in state:
                exp_avgs[nid] = np.array([float("nan")])
                exp_avg_sqs[nid] = np.array([float("nan")])
                moment_cos[nid] = float("nan")
                continue

            idx = u["unit_index"]
            ea = state["exp_avg"][idx].detach().flatten().float()
            eas = state["exp_avg_sq"][idx].detach().flatten().float()

            exp_avgs[nid] = ea.cpu().numpy()
            exp_avg_sqs[nid] = eas.cpu().numpy()

            prev = self._prev_exp_avg.get(nid)
            if prev is not None and prev.shape == ea.shape:
                moment_cos[nid] = torch.nn.functional.cosine_similarity(
                    ea.unsqueeze(0), prev.unsqueeze(0),
                ).item()
            else:
                moment_cos[nid] = 0.0
            self._prev_exp_avg[nid] = ea.clone()

        return {
            "exp_avg": exp_avgs,
            "exp_avg_sq": exp_avg_sqs,
            "moment_cosine_sim": moment_cos,
        }
