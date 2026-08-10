"""
Importing this package registers all 5 extractors (side-effect imports below)
so `get_extractor`/`list_extractors` see the full set.
"""

from optimizers.base import (
    OptimizerSignalExtractor,
    get_extractor,
    list_extractors,
    optimizer_extractor_from_dict,
    register_extractor,
)

from optimizers import (  # noqa: F401  (imported for registration side effects)
    adagrad_extractor,
    adam_extractor,
    grafted_shampoo_extractor,
    pure_shampoo_extractor,
    sgd_extractor,
)
__all__ = [
    "OptimizerSignalExtractor",
    "get_extractor",
    "list_extractors",
    "register_extractor",
    "optimizer_extractor_from_dict",
]
