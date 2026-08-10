"""Dict/JSON keys and default hyperparameters for optimizer signal extractors.

Same as `network_analyzer/compute_results/defaults.py`
"""

# dict/json keys for the saved "optimizer_config" section
OPTIMIZER_TYPE_KEY = "type"
OPTIMIZER_LR_KEY = "lr"
OPTIMIZER_BETAS_KEY = "betas"
OPTIMIZER_EPS_KEY = "eps"
OPTIMIZER_MOMENTUM_KEY = "momentum"
SHAMPOO_PRECONDITIONER_EPSILON_KEY = "shampoo_preconditioner_epsilon"
GRAFTING_EPS_KEY = "grafting_eps"

# Default hyperparameters
DEFAULT_LR = 5e-4  # matches this repo's RAE learning_rate default (CyclesConfig/TrainConfig)
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
