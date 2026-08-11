"""Dict/JSON keys and default hyperparameters for optimizer signal extractors.

Same as `network_analyzer/compute_results/defaults.py`
"""
from typing import Any

# dict/json keys for the saved "optimizer_config" section
OPTIMIZER_TYPE_KEY = "type"
OPTIMIZER_LR_KEY = "lr"
OPTIMIZER_BETAS_KEY = "betas"
OPTIMIZER_EPS_KEY = "eps"
OPTIMIZER_MOMENTUM_KEY = "momentum"
SHAMPOO_PRECONDITIONER_EPSILON_KEY = "shampoo_preconditioner_epsilon"
GRAFTING_EPS_KEY = "grafting_eps"

# Default hyperparameters
DEFAULT_LR = 1e-3  # matches TrainingConfig.learning_rate default
REFERENCE_LR = 0.01  # denominator in per-optimizer LR scaling

# Per-type base LR at REFERENCE_LR (Adam -> 0.0005 when base_lr = 0.001).
OPTIMIZER_LR_BASE: dict[str, float] = {
    "adam": 0.005,
    "sgd": 0.01,
    "adagrad": 0.01,
    "pure_shampoo": 0.005,
    "grafted_shampoo": 0.005,
}
DEFAULT_ADAM_BETAS: tuple[float, float] = (0.9, 0.999)
DEFAULT_ADAM_EPS = 1e-8
DEFAULT_ADAGRAD_EPS = 1e-10
DEFAULT_SGD_MOMENTUM = 0.0
DEFAULT_SHAMPOO_PRECONDITIONER_EPSILON = 1e-12
DEFAULT_SHAMPOO_BETAS: tuple[float, float] = (0.9, 0.999)
DEFAULT_GRAFTING_EPS = 1e-10

# Human-friendly config "type" shorthand -> registry extractor class name.
OPTIMIZER_SHORTHAND_TO_CLASS: dict[str, str] = {
    "sgd": "SGDExtractor",
    "adam": "AdamExtractor",
    "adagrad": "AdaGradExtractor",
    "pure_shampoo": "PureShampooExtractor",
    "grafted_shampoo": "GraftedShampooExtractor",
}


def resolve_optimizer_lr(optimizer_type: str, base_lr: float) -> float:
    """Scale a type-specific base LR by ``base_lr / REFERENCE_LR``."""
    if optimizer_type not in OPTIMIZER_LR_BASE:
        raise ValueError(
            f"Unknown optimizer type {optimizer_type!r}; "
            f"supported: {sorted(OPTIMIZER_LR_BASE)}"
        )
    return OPTIMIZER_LR_BASE[optimizer_type] * (base_lr / REFERENCE_LR)


def build_optimizer_config(optimizer_type: str, base_lr: float) -> dict[str, Any]:
    """Build an optimizer config dict with resolved ``type`` and ``lr``."""
    return {
        OPTIMIZER_TYPE_KEY: optimizer_type,
        OPTIMIZER_LR_KEY: resolve_optimizer_lr(optimizer_type, base_lr),
    }


def optimizer_tag_from_config(optimizer_config: dict[str, Any]) -> str:
    """Results/checkpoint subdir name, e.g. ``adam_0.0005``."""
    opt_type = optimizer_config[OPTIMIZER_TYPE_KEY]
    lr = optimizer_config[OPTIMIZER_LR_KEY]
    return f"{opt_type}_{lr:g}"
