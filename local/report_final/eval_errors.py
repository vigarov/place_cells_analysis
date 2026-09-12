"""Hardcoded checkpoint eval errors for supplementary report figures.

Values come from `local/eval_single_room_ckpt_eval_error.py` runs. Extend
`EXPERIMENT_EVAL_ERRORS` when new experiments are evaluated (e.g. two_rooms).
"""

from __future__ import annotations

# eval_error by optimizer, keyed by experiment name.
EXPERIMENT_EVAL_ERRORS: dict[str, dict[str, float]] = {
    "single_room": {
        "sgd": 0.536354,
        "adagrad": 0.322921,
        "adam": 0.310226,
        "pure_shampoo": 0.170075,
        "grafted_shampoo": 0.138351,
    },
    "single_room_plateau": {
        "sgd": 0.537793,
        "adagrad": 0.286435,
        "adam": 0.294879,
        "pure_shampoo": 0.181952,
        "grafted_shampoo": 0.13911,
    },
    "two_rooms": {
        "sgd": 0.371588,
        "adagrad": 0.313995,
        "adam": 0.324671,
        "pure_shampoo": 0.204561,
        "grafted_shampoo": 0.147806,    
    },
}

EXPERIMENT_LABELS: dict[str, str] = {
    "single_room": r"$\mathtt{(single)}$",
    "single_room_plateau": r"$\mathtt{(single)}$ (plateau opt.)",
    "two_rooms": r"$\mathtt{(two\ rooms)}$",
}
