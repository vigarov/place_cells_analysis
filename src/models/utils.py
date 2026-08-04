from dataclasses import dataclass, fields
from typing import Any

import torch
from models.nn4n.nn import (
    RNN,
    LeakyLinearLayer,
    RecurrentLayer,
    LinearLayer,
)

_ACTIVATION_FACTORIES: dict[str, type[torch.nn.Module]] = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
    "leaky_relu": torch.nn.LeakyReLU,
}


@dataclass
class RaeModelConfig:
    """Leaky RAE recurrent-layer hyperparameters (see ``build_rae``)."""

    activation: str = "relu"
    alpha: float = 0.1
    learn_alpha: bool = False
    preact_noise: float = 0.0
    postact_noise: float = 0.0


_RAE_MODEL_CONFIG_FIELDS = {f.name for f in fields(RaeModelConfig)}


def rae_model_config_from_dict(raw: dict[str, Any]) -> RaeModelConfig:
    """Build ``RaeModelConfig`` from a JSON object (unknown keys are ignored)."""
    kwargs = {k: v for k, v in raw.items() if k in _RAE_MODEL_CONFIG_FIELDS}
    return RaeModelConfig(**kwargs)


def resolve_activation(name: str) -> torch.nn.Module:
    key = name.lower()
    factory = _ACTIVATION_FACTORIES.get(key)
    if factory is None:
        supported = ", ".join(sorted(_ACTIVATION_FACTORIES))
        raise ValueError(f"Unknown activation {name!r}; supported: {supported}")
    return factory()


def build_rae(
    n_cells: int,
    n_hidden: int,
    device: torch.device | str,
    *,
    model: RaeModelConfig | None = None,
) -> RNN:
    """Build and move a readout autoencoder RNN to the target device."""
    model = model or RaeModelConfig()
    input_layer = LinearLayer(input_dim=n_cells, output_dim=n_hidden)
    output_layer = LinearLayer(input_dim=n_hidden, output_dim=n_cells)
    leaky_layer = LeakyLinearLayer(
        linear_layer=LinearLayer(input_dim=n_hidden, output_dim=n_hidden),
        activation=resolve_activation(model.activation),
        alpha=model.alpha,
        learn_alpha=model.learn_alpha,
        preact_noise=model.preact_noise,
        postact_noise=model.postact_noise,
    )
    rae = RNN(
        readout_layer=output_layer,
        recurrent_layers=[
            RecurrentLayer(
                leaky_layer=leaky_layer,
                projection_layer=input_layer,
            )
        ],
    )
    return rae.to(device)
