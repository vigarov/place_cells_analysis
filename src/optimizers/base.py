"""Base class and registry for optimizer signal extractors.

Ported from `network_analyzer/optimizers/base.py`. Each concrete extractor both
constructs a `torch.optim`-compatible optimizer for the RAE's parameters *and*
knows how to pull optimizer-internal signals (moments, accumulators, Shampoo
preconditioner factors, etc.) out of it around each `optimizer.step()` call, so
that training runs can be compared across optimizer choice.

Per-unit signals map `{node_id: scalar | 1-D ndarray}` (e.g. per-weight
effective LR, one entry per tracked hidden/output unit -- see `models.utils.build_tracked_units`).
Global block signals (Shampoo inverse-factor matrices) are returned from
`on_after_step_shampoo_blocks` as `{block_key: 2-D ndarray}`.
"""
import abc
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from optimizers.defaults import DEFAULT_LR, OPTIMIZER_SHORTHAND_TO_CLASS, OPTIMIZER_TYPE_KEY


class OptimizerSignalExtractor(abc.ABC):
    """Interface for extracting optimizer-specific signals during training."""

    def __init__(self) -> None:
        self._units: list[dict[str, Any]] = []
        self._prev_grads: dict[str, torch.Tensor] = {}

    def set_units(self, units: list[dict[str, Any]]) -> None:
        """Register the tracked units from the model (call before training)."""
        self._units = units

    @staticmethod
    def _find_weight_param(
        model: nn.Module, layer_name: str,
    ) -> nn.Parameter | None:
        """Find the weight parameter for a named layer."""
        params = dict(model.named_parameters())
        for candidate in (f"{layer_name}.weight", f"{layer_name}.0.weight"):
            if candidate in params:
                return params[candidate]
        for name, param in params.items():
            if name.startswith(layer_name) and "weight" in name:
                return param
        return None

    def _compute_grad_signals(
        self, model: nn.Module,
    ) -> dict[str, dict[str, Any]]:
        """Per-unit gradient vectors, norms, and cosine similarity vs previous iteration.

        Shared by all extractors - call from `on_before_step`.
        """
        grads: dict[str, np.ndarray] = {}
        norms: dict[str, float] = {}
        cosines: dict[str, float] = {}

        for u in self._units:
            nid = u["node_id"]
            param = self._find_weight_param(model, u["layer_name"])
            if param is None or param.grad is None:
                grads[nid] = np.array([float("nan")])
                norms[nid] = float("nan")
                cosines[nid] = float("nan")
                continue

            grad = param.grad[u["unit_index"]].detach().flatten().float()
            grads[nid] = grad.cpu().numpy()
            norms[nid] = grad.norm().item()

            prev = self._prev_grads.get(nid)
            if prev is not None and prev.shape == grad.shape:
                cosines[nid] = torch.nn.functional.cosine_similarity(
                    grad.unsqueeze(0), prev.unsqueeze(0),
                ).item()
            else:
                cosines[nid] = 0.0

            self._prev_grads[nid] = grad.clone()

        return {"grad": grads, "grad_norm": norms, "grad_cosine_sim": cosines}

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def optimizer_id(self) -> str:
        """Unique identifier (e.g. 'adam_lr1e-3')."""

    @abc.abstractmethod
    def optimizer_config(self) -> dict[str, Any]:
        """Optimizer-specific hyperparams to store / compare."""

    @abc.abstractmethod
    def create_optimizer(self, params) -> torch.optim.Optimizer:
        """Instantiate the underlying optimizer."""

    @abc.abstractmethod
    def signal_names(self) -> list[str]:
        """Names of per-unit, per-iteration signals this extractor logs."""

    @abc.abstractmethod
    def on_before_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, dict[str, float]]:
        """Called after `loss.backward()` but before `optimizer.step()`.

        Return `{signal_name: {node_id: value}}`.
        """

    @abc.abstractmethod
    def on_after_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, dict[str, float]]:
        """Called after `optimizer.step()`.

        Return `{signal_name: {node_id: value}}`.
        """

    def on_after_step_shampoo_blocks(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, np.ndarray]:
        """Called after `on_after_step` each iteration.
        Only non `{}` defined for Shampoo extractors
        Return global block signals `{block_key: matrix}` not keyed by unit.
        """
        return {}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_EXTRACTOR_REGISTRY: dict[str, type[OptimizerSignalExtractor]] = {}


def register_extractor(cls: type[OptimizerSignalExtractor]) -> type[OptimizerSignalExtractor]:
    key = cls.__name__
    _EXTRACTOR_REGISTRY[key] = cls
    return cls


def get_extractor(name: str, **kwargs: Any) -> OptimizerSignalExtractor:
    if name not in _EXTRACTOR_REGISTRY:
        raise KeyError(
            f"Unknown extractor '{name}'. "
            f"Available: {sorted(_EXTRACTOR_REGISTRY)}"
        )
    return _EXTRACTOR_REGISTRY[name](**kwargs)


def list_extractors() -> list[str]:
    return sorted(_EXTRACTOR_REGISTRY)


def optimizer_extractor_from_dict(raw: dict[str, Any] | None) -> OptimizerSignalExtractor:
    """Instantiate the extractor named by `raw["type"]` (default: `"adam"`).

    `raw` is passed through as keyword arguments to the extractor's
    constructor (minus the `"type"` key), e.g.
    `{"type": "sgd", "lr": 1e-3, "momentum": 0.9}`.
    """
    raw = dict(raw or {})
    type_name = raw.pop(OPTIMIZER_TYPE_KEY, "adam")
    if type_name not in OPTIMIZER_SHORTHAND_TO_CLASS:
        raise ValueError(
            f"Unknown optimizer type {type_name!r}; supported: "
            f"{sorted(OPTIMIZER_SHORTHAND_TO_CLASS)}"
        )
    class_name = OPTIMIZER_SHORTHAND_TO_CLASS[type_name]
    raw.setdefault("lr", DEFAULT_LR)
    if "betas" in raw:
        raw["betas"] = tuple(raw["betas"])
    return get_extractor(class_name, **raw)
