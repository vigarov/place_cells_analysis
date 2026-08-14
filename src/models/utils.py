from dataclasses import dataclass, fields
from typing import Any

import torch
from models.bump_activation import BumpActivation
from models.custom_learnable_activation import CustomLearnableActivation
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

_SIZE_DEPENDENT_ACTIVATIONS: dict[str, type[torch.nn.Module]] = {
    "custom_learnable": CustomLearnableActivation,
}


@dataclass
class RaeModelConfig:
    """Leaky RAE recurrent-layer hyperparameters (see `build_rae`)."""

    activation: str = "relu"
    alpha: float = 0.1
    learn_alpha: bool = False
    preact_noise: float = 0.0
    postact_noise: float = 0.0
    bump_nl: float | None = None
    bump_nr: float | None = None
    bump_s: float = 3.0


_RAE_MODEL_CONFIG_FIELDS = {f.name for f in fields(RaeModelConfig)}


def rae_model_config_from_dict(raw: dict[str, Any]) -> RaeModelConfig:
    """Build `RaeModelConfig` from a JSON object (unknown keys are ignored)."""
    kwargs = {k: v for k, v in raw.items() if k in _RAE_MODEL_CONFIG_FIELDS}
    return RaeModelConfig(**kwargs)


def resolve_activation(
    name: str,
    hidden_size: int,
    *,
    model: RaeModelConfig | None = None,
) -> torch.nn.Module:
    key = name.lower()
    if key == "bump_activation":
        if model is None or model.bump_nl is None or model.bump_nr is None:
            raise ValueError(
                "bump_activation requires bump_nl and bump_nr in RaeModelConfig"
            )
        return BumpActivation(
            hidden_size,
            nl=model.bump_nl,
            nr=model.bump_nr,
            s=model.bump_s,
        )
    sized_factory = _SIZE_DEPENDENT_ACTIVATIONS.get(key)
    if sized_factory is not None:
        return sized_factory(hidden_size)
    factory = _ACTIVATION_FACTORIES.get(key)
    if factory is None:
        supported = ", ".join(
            sorted(
                _ACTIVATION_FACTORIES
                | _SIZE_DEPENDENT_ACTIVATIONS
                | {"bump_activation"}
            )
        )
        raise ValueError(f"Unknown activation {name!r}; supported: {supported}")
    return factory()



def seed_model_init(init_seed: int) -> None:
    """Seed PyTorch's global RNG before ``build_rae`` for reproducible init."""
    torch.manual_seed(init_seed)


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
        activation=resolve_activation(model.activation, n_hidden, model=model),
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


# (short display name, fully-qualified `named_parameters()` layer prefix)
_TRACKED_LAYER_SPECS: tuple[tuple[str, str], ...] = (
    ("recurrent", "recurrent_layers.0.leaky_layer.linear_layer"),
    ("input", "recurrent_layers.0.projection_layer"),
    ("readout", "readout_layer"),
)


def build_tracked_units(
    rae: torch.nn.Module,
    *,
    max_units_per_layer: int | None = None,
) -> list[dict[str, Any]]:
    """Build `[{"node_id", "layer_name", "unit_index"}, ...]` for the RAE.

    One entry per row of each of the 3 weight matrices (hidden units for
    `recurrent`/`input`, output/sensory-cell units for `readout`). Set
    `max_units_per_layer` to track only the first N rows of each layer
    """
    params = dict(rae.named_parameters())
    units: list[dict[str, Any]] = []
    for short_name, layer_name in _TRACKED_LAYER_SPECS:
        weight = params.get(f"{layer_name}.weight")
        if weight is None:
            continue
        n_units = weight.shape[0]
        if max_units_per_layer is not None:
            n_units = min(n_units, max_units_per_layer)
        for i in range(n_units):
            units.append(
                {
                    "node_id": f"{short_name}[{i}]",
                    "layer_name": layer_name,
                    "unit_index": i,
                }
            )
    return units


def order_node_ids(units: list[dict[str, Any]]) -> list[str]:
    """Stable node-id ordering matching `build_tracked_units`'s output order."""
    return [u["node_id"] for u in units]
