"""PureShampoo optimizer signal extractor (no grafting).

Per-unit signals (gradient signals inherited from base):
- grad_norm, grad_cosine_sim
- h_inv_norm: per-unit ||L^{-1/4}[i, :]||_2 * ||R^{-1/4}||_F from inverse Kronecker factors
  (L shape is (out, out), R -- (in, in) (for Linear of shape (out, in)); unit i is row i).

Layer-wise block signals (via on_after_step_shampoo_blocks):
- h_inv__{param}__L / __R: inverse Kronecker factor matrices (paper notation)
  for each preconditioned parameter (weights and biases).
"""
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn

from distributed_shampoo import DistributedShampoo, RootInvShampooPreconditionerConfig
from distributed_shampoo.shampoo_types import SHAMPOO_PRECONDITIONER_LIST, SingleDeviceDistributedConfig

from optimizers.base import OptimizerSignalExtractor, register_extractor
from optimizers.defaults import (
    DEFAULT_LR,
    DEFAULT_SHAMPOO_BETAS,
    DEFAULT_SHAMPOO_PRECONDITIONER_EPSILON,
    OPTIMIZER_BETAS_KEY,
    OPTIMIZER_LR_KEY,
    OPTIMIZER_TYPE_KEY,
    SHAMPOO_PRECONDITIONER_EPSILON_KEY,
)

_SHAMPOO_FACTOR_SUFFIXES = ("L", "R")


def _param_names_in_optimizer_order(
    model: nn.Module, optimizer: torch.optim.Optimizer,
) -> list[str]:
    param_to_name = {id(p): name for name, p in model.named_parameters()}
    names: list[str] = []
    for group in optimizer.param_groups:
        for param in group["params"]:
            names.append(param_to_name.get(id(param), f"param_{len(names)}"))
    return names


def _tracked_layer_names(units: list[dict[str, Any]]) -> set[str]:
    return {str(u["layer_name"]) for u in units}


def _layer_name_from_weight_param(param_name: str) -> str | None:
    if param_name.endswith(".weight"):
        return param_name[: -len(".weight")]
    return None


def _iter_param_inv_factor_tensors(
    optimizer: DistributedShampoo, model: nn.Module,
) -> Iterator[tuple[str, tuple[torch.Tensor, ...]]]:
    """Iterator returning `(param_name, inv_factor_matrices)` in optimizer param order."""
    param_names = _param_names_in_optimizer_order(model, optimizer)
    name_idx = 0
    for state_lists in optimizer._per_group_state_lists:
        shampoo_list = state_lists[SHAMPOO_PRECONDITIONER_LIST]
        kf_list = shampoo_list._local_kronecker_factors_unwrapped
        for kf in kf_list:
            param_name = (
                param_names[name_idx]
                if name_idx < len(param_names)
                else f"param_{name_idx}"
            )
            name_idx += 1
            inv_mats = getattr(kf, "inv_factor_matrices", None)
            if not inv_mats:
                continue
            yield param_name, inv_mats


def _compute_h_inv_norms_by_unit(
    optimizer: DistributedShampoo,
    model: nn.Module,
    units: list[dict[str, Any]],
) -> dict[str, float]:
    """Level-2 per-unit h_inv norm; cache row/R norms once per layer per step."""
    tracked_layers = _tracked_layer_names(units)
    layer_row_scales: dict[str, torch.Tensor] = {}
    layer_r_fro: dict[str, float] = {}

    for param_name, inv_mats in _iter_param_inv_factor_tensors(optimizer, model):
        layer_name = _layer_name_from_weight_param(param_name)
        if layer_name is None or layer_name not in tracked_layers:
            continue
        if len(inv_mats) < 2:
            continue
        layer_row_scales[layer_name] = inv_mats[0].detach().norm(dim=1)  # not "fro" (we take norm on a row --> L2)
        layer_r_fro[layer_name] = float(inv_mats[1].detach().norm(p="fro").item())

    out: dict[str, float] = {}
    for u in units:
        nid = u["node_id"]
        layer_name = str(u["layer_name"])
        row_scales = layer_row_scales.get(layer_name)
        if row_scales is None or layer_name not in layer_r_fro:
            print(f"{__file__} row_scales is None or layer_name not in layer_r_fro")
            out[nid] = float("nan")
            continue
        idx = int(u["unit_index"])
        out[nid] = float(row_scales[idx].item() * layer_r_fro[layer_name])
    return out


def _blocks_from_inv_factors(
    param_name: str, inv_mats: tuple[torch.Tensor, ...],
) -> dict[str, np.ndarray]:
    safe = param_name.replace(".", "__")
    blocks: dict[str, np.ndarray] = {}
    for fi, mat in enumerate(inv_mats):
        suffix = (
            _SHAMPOO_FACTOR_SUFFIXES[fi]
            if fi < len(_SHAMPOO_FACTOR_SUFFIXES)
            else f"f{fi}"
        )
        blocks[f"h_inv__{safe}__{suffix}"] = mat.detach().float().cpu().numpy()
    return blocks


def _extract_h_inv_blocks(
    optimizer: DistributedShampoo,
    model: nn.Module,
    *,
    units: list[dict[str, Any]] | None = None,
) -> dict[str, np.ndarray]:
    """Extract inverse-factor matrices for saving (optionally tracked layers only)."""
    tracked_layers = _tracked_layer_names(units) if units is not None else None
    blocks: dict[str, np.ndarray] = {}
    for param_name, inv_mats in _iter_param_inv_factor_tensors(optimizer, model):
        if tracked_layers is not None:
            layer_name = _layer_name_from_weight_param(param_name)
            if layer_name is None or layer_name not in tracked_layers:
                continue
        blocks.update(_blocks_from_inv_factors(param_name, inv_mats))
    return blocks


@register_extractor
class PureShampooExtractor(OptimizerSignalExtractor):

    def __init__(
        self,
        *,
        lr: float = DEFAULT_LR,
        betas: tuple[float, float] = DEFAULT_SHAMPOO_BETAS,
        shampoo_preconditioner_epsilon: float = DEFAULT_SHAMPOO_PRECONDITIONER_EPSILON,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.betas = betas
        self.shampoo_preconditioner_epsilon = shampoo_preconditioner_epsilon
        self._pending_block_signals: dict[str, np.ndarray] = {}

    def optimizer_id(self) -> str:
        return f"pure_shampoo_lr{self.lr}"

    def optimizer_config(self) -> dict[str, Any]:
        return {
            OPTIMIZER_TYPE_KEY: "pure_shampoo",
            OPTIMIZER_LR_KEY: self.lr,
            OPTIMIZER_BETAS_KEY: list(self.betas),
            SHAMPOO_PRECONDITIONER_EPSILON_KEY: self.shampoo_preconditioner_epsilon,
        }

    def create_optimizer(self, params) -> torch.optim.Optimizer:
        return DistributedShampoo(
            params,
            lr=self.lr,
            betas=self.betas,
            weight_decay=0.0,
            epsilon=self.shampoo_preconditioner_epsilon,
            grafting_config=None,
            preconditioner_config=RootInvShampooPreconditionerConfig(
                inv_factor_matrix_dtype=torch.float64,
            ),
            max_preconditioner_dim=1024,
            distributed_config=SingleDeviceDistributedConfig(
                target_parameter_dimensionality=2,
            ),
        )

    def signal_names(self) -> list[str]:
        return ["grad", "grad_norm", "grad_cosine_sim", "h_inv_norm"]

    def on_before_step(
        self, model: nn.Module, optimizer: torch.optim.Optimizer,
    ) -> dict[str, dict[str, float]]:
        return self._compute_grad_signals(model)

    def on_after_step(
        self, model: nn.Module, optimizer: torch.optim.Optimizer,
    ) -> dict[str, dict[str, float]]:
        assert isinstance(optimizer, DistributedShampoo)
        self._pending_block_signals = _extract_h_inv_blocks(
            optimizer, model, units=self._units,
        )
        return {
            "h_inv_norm": _compute_h_inv_norms_by_unit(
                optimizer, model, self._units,
            ),
        }

    def on_after_step_shampoo_blocks(
        self, model: nn.Module, optimizer: torch.optim.Optimizer,
    ) -> dict[str, np.ndarray]:
        return dict(self._pending_block_signals)
