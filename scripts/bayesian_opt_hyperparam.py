#!/usr/bin/env python3
"""
Bayesian hyperparameter optimization for room-based experiments.
Two variants -- normal RNN, or RNN with learnable activation function: 

Variant A: ReLU RAE; tunes lambda_mse, lambda_fr, mask_rate, gradient_clip_max, alpha.
Variant B: custom activation RAE; same plus lambda_ab;
Variant C: BumpActivation RAE; training hyperparams fixed from config; tunes nl, nr, s (nr > nl); 

Each trial runs full warmup plus two training epochs, then evaluates on held-out
trajectories and saves ratemaps. Log train and eval-set MSE autoencoding error.

Objectives: 
* minimize loss or train-set prediction MSE (`train_error`, use --error) and activity_score,
* subject to activity_fraction constraints, 
* and lambda_fr <= MAX_FR_FACTOR * lambda_mse ; lambda_ab <= MAX_AB_FACTOR * lambda_mse (for variant B).


Usage::

    uv run generate-experiment-room --config input_configs/single_room.json
    uv run bayesian-opt-hyperparam --config input_configs/single_room.json --variant a --n-trials 30
    uv run bayesian-opt-hyperparam --config input_configs/single_room.json --variant b --n-trials 30
    uv run bayesian-opt-hyperparam --config input_configs/single_room.json --variant c --n-trials 30
    uv run bayesian-opt-hyperparam --config input_configs/single_room.json --lr-only --n-trials 20
    uv run bayesian-opt-hyperparam --config input_configs/single_room.json --variant b --ab-only --n-trials 20
    uv run bayesian-opt-hyperparam --config input_configs/single_room.json --variant a --alt-training lora --rank 16 --n-trials 30
    uv run bayesian-opt-hyperparam --config input_configs/single_room.json --variant a --alt-training on_opt --n-trials 30
    uv run bayesian-opt-hyperparam --config input_configs/single_room.json --lr-only --exclude-opt sgd --n-trials 20
"""
import argparse
import copy
import csv
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from ax.exceptions.core import AxError
from ax.service.ax_client import AxClient, ObjectiveProperties

from core.hyperparam_trial import (
    ACTIVITY_TARGET,
    BAD_RUN_ACTIVITY_SCORE,
    BAD_RUN_LOSS,
    BAD_RUN_TRAIN_ERROR,
    MAX_FR_FACTOR,
    AltTrainingVariant,
    BoVariant,
    alt_training_path_tag,
    apply_trial_params,
    bo_parameter_constraints,
    estimate_trial_segments,
    fixed_trial_params,
    lambda_fr_mse_parameter_constraints,
    normalize_alt_training_variant,
    run_hyperparam_trial,
    search_bounds_from_config,
)
from experiments.common.optimizers_config import parse_optimizers, raw_optimizers_value
from experiments.common.run import ExperimentRunConfig, create_experiment, load_experiment_config
from optimizers.defaults import OPTIMIZER_SHORTHAND_TO_CLASS

ACTIVITY_FRACTION_LO = ACTIVITY_TARGET - 0.3
ACTIVITY_FRACTION_HI = ACTIVITY_TARGET + 0.2

TRIAL_METRIC_FIELDS = [
    "activity_fraction",
    "activity_score",
    "loss",
    "train_error",
    "eval_error",
    "aborted_nan",
]

VARIANT_B_METRIC_FIELDS = [
    "activation_a_mean",
    "activation_a_sem",
    "activation_b_mean",
    "activation_b_sem",
]

VARIANT_C_METRIC_FIELDS = [
    "activation_A_mean",
    "activation_A_sem",
    "activation_m_mean",
    "activation_m_sem",
]

TRIALS_CSV_FIELDS_A = [
    "trial_index",
    "timestamp",
    "lambda_mse",
    "lambda_fr",
    "gradient_clip_max",
    "mask_rate",
    "alpha",
    *TRIAL_METRIC_FIELDS,
    "ratemap_path",
]

TRIALS_CSV_FIELDS_B = [
    "trial_index",
    "timestamp",
    "lambda_mse",
    "lambda_fr",
    "gradient_clip_max",
    "mask_rate",
    "alpha",
    "lambda_ab",
    *TRIAL_METRIC_FIELDS,
    *VARIANT_B_METRIC_FIELDS,
    "ratemap_path",
]

TRIALS_CSV_FIELDS_C = [
    "trial_index",
    "timestamp",
    "nl",
    "nr",
    "s",
    *TRIAL_METRIC_FIELDS,
    *VARIANT_C_METRIC_FIELDS,
    "ratemap_path",
]


def _parse_bounds(value: str, name: str) -> tuple[float, float]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"{name} must be 'low,high', got {value!r}")
    try:
        low, high = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} bounds must be floats: {value!r}") from exc
    if low >= high:
        raise argparse.ArgumentTypeError(f"{name} lower bound must be < upper bound: {value!r}")
    return low, high


def _default_output_dir(
    experiment_type: str,
    variant: BoVariant,
    *,
    optimizer: str | None = None,
    stamp: str | None = None,
    alt_training: AltTrainingVariant | None = None,
    alt_rank: int | None = None,
) -> Path:
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    alt_tag = alt_training_path_tag(alt_training, rank=alt_rank) if alt_training else ""
    if optimizer is not None:
        run_name = f"variant_{variant}{alt_tag}_{optimizer}_{stamp}"
    else:
        run_name = f"variant_{variant}{alt_tag}_{stamp}"
    return Path("results") / "hyperparam_opt" / experiment_type / run_name


def _trials_csv_fields(
    variant: BoVariant,
    *,
    lr_only: bool = False,
    ab_only: bool = False,
    alt_training: AltTrainingVariant | None = None,
) -> list[str]:
    if alt_training:
        tuned_param = "beta" if alt_training == "lora" else "kappa"
        fields = [
            "trial_index",
            "timestamp",
            tuned_param,
            "lambda_mse",
            "lambda_fr",
            "gradient_clip_max",
            "mask_rate",
            "alpha",
        ]
        if variant == "b":
            fields.append("lambda_ab")
        if variant == "c":
            fields.extend(["nl", "nr", "s"])
        fields.extend(TRIAL_METRIC_FIELDS)
        if variant == "b":
            fields.extend(VARIANT_B_METRIC_FIELDS)
        if variant == "c":
            fields.extend(VARIANT_C_METRIC_FIELDS)
        fields.append("ratemap_path")
        return fields

    if lr_only:
        fields = [
            "trial_index",
            "timestamp",
            "learning_rate",
            "lambda_mse",
            "lambda_fr",
            "gradient_clip_max",
            "mask_rate",
            "alpha",
        ]
        if variant == "b":
            fields.append("lambda_ab")
        fields.extend(TRIAL_METRIC_FIELDS)
        if variant == "b":
            fields.extend(VARIANT_B_METRIC_FIELDS)
        fields.append("ratemap_path")
        return fields
    if ab_only:
        fields = [
            "trial_index",
            "timestamp",
            "lambda_ab",
            "lambda_mse",
            "lambda_fr",
            "gradient_clip_max",
            "mask_rate",
            "alpha",
            *TRIAL_METRIC_FIELDS,
            *VARIANT_B_METRIC_FIELDS,
            "ratemap_path",
        ]
        return fields
    if variant == "c":
        return TRIALS_CSV_FIELDS_C
    return TRIALS_CSV_FIELDS_B if variant == "b" else TRIALS_CSV_FIELDS_A


def _ax_parameters(bounds: dict[str, Any]) -> list[dict[str, Any]]:
    parameters: list[dict[str, Any]] = []
    for name, bound in bounds.items():
        low, high = bound
        spec: dict[str, Any] = {
            "name": name,
            "type": "range",
            "bounds": [low, high],
            "value_type": "float",
        }
        if name == "learning_rate":
            spec["log_scale"] = True
        parameters.append(spec)
    return parameters


def _verify_room_data(run_config) -> None:
    experiment = create_experiment(run_config)
    manifest_path = experiment.data_dir() / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(
            f"Room data not found at {manifest_path}.\n"
            "Generate it first, e.g.:\n"
            f"  uv run generate-experiment-room --config <your-config.json>"
        )


def _ax_setup(
    *, optimize_error: bool
) -> tuple[dict[str, ObjectiveProperties], list[str], list[str]]:
    activity_score = ObjectiveProperties(
        minimize=True, threshold=BAD_RUN_ACTIVITY_SCORE
    )
    if optimize_error:
        objectives = {
            "train_error": ObjectiveProperties(
                minimize=True,
                threshold=min(BAD_RUN_TRAIN_ERROR, 100.0),
            ),
            "activity_score": activity_score,
        }
        tracking_metric_names = ["loss", "eval_error"]
    else:
        objectives = {
            "loss": ObjectiveProperties(minimize=True, threshold=min(BAD_RUN_LOSS, 100.0)),
            "activity_score": activity_score,
        }
        tracking_metric_names = ["train_error", "eval_error"]
    outcome_constraints = [
        f"activity_fraction >= {ACTIVITY_FRACTION_LO}",
        f"activity_fraction <= {ACTIVITY_FRACTION_HI}",
    ]
    return objectives, outcome_constraints, tracking_metric_names


def _primary_objective_names(*, optimize_error: bool) -> tuple[str, ...]:
    objectives, _, _ = _ax_setup(optimize_error=optimize_error)
    return tuple(objectives.keys())


def _current_objective_names(*, optimize_error: bool) -> frozenset[str]:
    objectives, _, _ = _ax_setup(optimize_error=optimize_error)
    return frozenset(objectives.keys())


def _ax_objective_names(ax_client: AxClient) -> frozenset[str]:
    return frozenset(ax_client.experiment.optimization_config.objective.metric_names)


def _objectives_match(ax_client: AxClient, *, optimize_error: bool) -> bool:
    return _ax_objective_names(ax_client) == _current_objective_names(
        optimize_error=optimize_error
    )


def _is_multi_objective(ax_client: AxClient) -> bool:
    return ax_client.experiment.optimization_config.objective.is_multi_objective


def _ax_registered_metrics(*, optimize_error: bool) -> set[str]:
    objectives, outcome_constraints, tracking_metric_names = _ax_setup(
        optimize_error=optimize_error
    )
    names = set(objectives) | set(tracking_metric_names)
    for constraint in outcome_constraints:
        names.add(constraint.split()[0])
    return names


def _ax_raw_data(metrics: dict[str, float], *, optimize_error: bool) -> dict[str, float]:
    from core.hyperparam_trial import replace_bad_run_metrics

    registered = _ax_registered_metrics(optimize_error=optimize_error)
    sanitized = replace_bad_run_metrics(metrics)
    return {name: float(value) for name, value in sanitized.items() if name in registered}


def _create_ax_client(
    *,
    experiment_type: str,
    variant: BoVariant,
    bounds: dict[str, Any],
    random_seed: int | None,
    sobol_trials: int,
    use_lambda_fr_constraint: bool,
    optimize_error: bool,
    optimizer: str | None = None,
    alt_training: AltTrainingVariant | None = None,
    alt_rank: int | None = None,
) -> AxClient:
    objectives, outcome_constraints, tracking_metric_names = _ax_setup(
        optimize_error=optimize_error
    )
    parameter_constraints = bo_parameter_constraints(
        variant=variant,
        enabled=use_lambda_fr_constraint,
        bounds=bounds,
    )
    experiment_name = f"hyperparam_{experiment_type}_variant_{variant}"
    if alt_training is not None:
        experiment_name = f"{experiment_name}{alt_training_path_tag(alt_training, rank=alt_rank)}"
    if optimizer is not None:
        experiment_name = f"{experiment_name}_{optimizer}"
    ax_client = AxClient(random_seed=random_seed, verbose_logging=True)
    ax_client.create_experiment(
        name=experiment_name,
        parameters=_ax_parameters(bounds),
        objectives=objectives,
        parameter_constraints=parameter_constraints or None,
        outcome_constraints=outcome_constraints,
        tracking_metric_names=tracking_metric_names,
        choose_generation_strategy_kwargs={"num_initialization_trials": sobol_trials},
        overwrite_existing_experiment=True,
    )
    return ax_client


def _attach_completed_trials(
    ax_client: AxClient,
    source_df: pd.DataFrame,
    param_names: list[str],
    *,
    optimize_error: bool,
) -> int:
    """Replay completed trials from a prior Ax run or CSV export."""
    registered_metrics = _ax_registered_metrics(optimize_error=optimize_error)
    attached = 0
    sort_cols = ["trial_index"] if "trial_index" in source_df.columns else None
    rows = source_df.sort_values(sort_cols) if sort_cols is not None else source_df
    for _, row in rows.iterrows():
        if row.get("trial_status") not in (None, "COMPLETED"):
            continue
        params = {name: float(row[name]) for name in param_names}
        raw_data: dict[str, float] = {}
        for name in registered_metrics:
            if name not in row:
                continue
            value = row[name]
            if pd.isna(value):
                continue
            raw_data[name] = float(value)
        _, trial_index = ax_client.attach_trial(params)
        ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)
        attached += 1
    return attached


def _copy_run_artifacts(source_dir: Path, output_dir: Path) -> None:
    trials_csv = source_dir / "trials.csv"
    if trials_csv.is_file():
        shutil.copy2(trials_csv, output_dir / "trials.csv")
    ratemaps_dir = source_dir / "ratemaps"
    if ratemaps_dir.is_dir():
        shutil.copytree(ratemaps_dir, output_dir / "ratemaps", dirs_exist_ok=True)


def _load_ax_client(resume_path: Path) -> AxClient:
    return AxClient.load_from_json_file(str(resume_path))


def _write_run_meta(
    output_dir: Path,
    *,
    variant: BoVariant,
    config_path: Path,
    bounds: dict[str, Any],
    use_lambda_fr_constraint: bool,
    optimize_error: bool,
    lr_only: bool = False,
    ab_only: bool = False,
    optimizer: str | None = None,
    alt_training: AltTrainingVariant | None = None,
    alt_rank: int | None = None,
    migrated_from: Path | None = None,
) -> None:
    meta: dict[str, Any] = {
        "variant": variant,
        "config_path": str(config_path.resolve()),
        "bounds": {k: list(v) for k, v in bounds.items()},
        "objectives": sorted(_current_objective_names(optimize_error=optimize_error)),
        "lambda_fr_constraint": use_lambda_fr_constraint,
        "parameter_constraints": bo_parameter_constraints(
            variant=variant,
            enabled=use_lambda_fr_constraint,
            bounds=bounds,
        ),
        "optimize_error": optimize_error,
        "lr_only": lr_only,
        "ab_only": ab_only,
        "alt_training": alt_training is not None,
        "alt_training_variant": alt_training,
        "max_fr_factor": MAX_FR_FACTOR,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if alt_training == "lora":
        if alt_rank is None:
            raise ValueError("alt_rank is required when alt_training='lora'")
        meta["rank"] = alt_rank
    if optimizer is not None:
        meta["optimizer"] = optimizer
    if migrated_from is not None:
        meta["migrated_from"] = str(migrated_from.resolve())
        meta["migrated_at"] = datetime.now(timezone.utc).isoformat()
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def _load_run_meta(output_dir: Path) -> dict[str, Any]:
    meta_path = output_dir / "run_meta.json"
    if not meta_path.is_file():
        raise SystemExit(
            f"Missing run_meta.json in {output_dir}. "
            "Cannot resume without variant metadata."
        )
    return json.loads(meta_path.read_text())


def _append_trial_csv(
    output_dir: Path,
    row: dict[str, Any],
    *,
    variant: BoVariant,
    lr_only: bool = False,
    ab_only: bool = False,
    alt_training: AltTrainingVariant | None = None,
) -> None:
    csv_path = output_dir / "trials.csv"
    fieldnames = _trials_csv_fields(
        variant, lr_only=lr_only, ab_only=ab_only, alt_training=alt_training
    )
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def _feasible_trial_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "activity_fraction" not in df.columns:
        return df.iloc[0:0]
    activity = pd.to_numeric(df["activity_fraction"], errors="coerce")
    mask = (activity >= ACTIVITY_FRACTION_LO) & (activity <= ACTIVITY_FRACTION_HI)
    return df.loc[mask].copy()


def _pareto_optimal_rows(
    df: pd.DataFrame,
    *,
    optimize_error: bool,
    objective_names: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    if objective_names is None:
        objective_names = _primary_objective_names(optimize_error=optimize_error)
    feasible = _feasible_trial_rows(df)
    if feasible.empty:
        return feasible

    keep: list[int] = []
    for idx, row in feasible.iterrows():
        dominated = False
        for other_idx, other in feasible.iterrows():
            if other_idx == idx:
                continue
            if all(
                float(other[name]) <= float(row[name]) for name in objective_names
            ) and any(
                float(other[name]) < float(row[name]) for name in objective_names
            ):
                dominated = True
                break
        if not dominated:
            keep.append(int(row["trial_index"]))

    if not keep:
        return feasible.iloc[0:0]
    pareto = feasible[feasible["trial_index"].isin(keep)].copy()
    sort_cols = [*objective_names, "trial_index"]
    return pareto.sort_values(sort_cols)


def _serialize_best_trials(
    ax_client: AxClient, *, optimize_error: bool
) -> list[dict[str, Any]]:
    df = ax_client.get_trials_data_frame()
    completed = df[df["trial_status"] == "COMPLETED"] if "trial_status" in df.columns else df
    if completed.empty:
        return []

    registered_metrics = _ax_registered_metrics(optimize_error=optimize_error)
    if _is_multi_objective(ax_client):
        pareto = _pareto_optimal_rows(completed, optimize_error=optimize_error)
        rows: list[dict[str, Any]] = []
        for _, row in pareto.iterrows():
            metrics = {
                name: float(row[name])
                for name in registered_metrics
                if name in row and pd.notna(row[name])
            }
            params = {
                name: float(row[name])
                for name in row.index
                if name
                not in {
                    "trial_index",
                    "arm_name",
                    "trial_status",
                    "generation_node",
                    *registered_metrics,
                }
                and pd.notna(row[name])
            }
            rows.append(
                {
                    "trial_index": int(row["trial_index"]),
                    "parameters": params,
                    "metrics": metrics,
                }
            )
        return rows

    try:
        best = ax_client.get_best_trial(use_model_predictions=False)
        if best is None:
            return []
        trial_index, parameters, metrics = best
        results = {trial_index: (parameters, metrics)}
    except (AxError, NotImplementedError, TypeError, ValueError, RuntimeError):
        return []

    rows = []
    for trial_index, (parameters, metrics) in results.items():
        mean_metrics, _ = metrics
        rows.append(
            {
                "trial_index": trial_index,
                "parameters": parameters,
                "metrics": {k: float(v) for k, v in mean_metrics.items() if v == v},
            }
        )
    return rows


def _write_summary(
    output_dir: Path, ax_client: AxClient, *, optimize_error: bool
) -> None:
    best_trials = _serialize_best_trials(ax_client, optimize_error=optimize_error)
    summary = {
        "n_completed_trials": len(ax_client.get_trials_data_frame()),
        "best_trial": best_trials,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def _trial_is_infeasible(metrics: dict[str, float]) -> bool:
    activity_fraction = metrics.get("activity_fraction", float("nan"))
    return (
        activity_fraction < ACTIVITY_FRACTION_LO
        or activity_fraction > ACTIVITY_FRACTION_HI
    )


def _print_trial_summary(
    trial_num: int,
    n_trials: int,
    trial_index: int,
    params: dict[str, Any],
    metrics: dict[str, float],
    ax_client: AxClient,
    *,
    variant: BoVariant,
    optimize_error: bool,
    lr_only: bool = False,
    ab_only: bool = False,
    optimizer: str | None = None,
    alt_training: AltTrainingVariant | None = None,
) -> None:
    opt_label = f", optimizer={optimizer}" if optimizer is not None else ""
    alt_label = f", alt-training={alt_training}" if alt_training else ""
    print(
        f"\n=== Trial {trial_num}/{n_trials} (index {trial_index}) "
        f"[variant {variant}{opt_label}{alt_label}] ==="
    )
    if lr_only:
        param_parts = [f"learning_rate={params['learning_rate']:.4g}"]
    elif alt_training == "lora":
        param_parts = [f"beta={params['beta']:.4g}"]
    elif alt_training == "on_opt":
        param_parts = [f"kappa={params['kappa']:.4g}"]
    elif ab_only:
        param_parts = [f"lambda_ab={params['lambda_ab']:.4g}"]
    elif variant == "c":
        param_parts = [
            f"nl={params['nl']:.4g}",
            f"nr={params['nr']:.4g}",
            f"s={params['s']:.4g}",
        ]
    else:
        param_parts = [
            f"lambda_mse={params['lambda_mse']:.4g}",
            f"lambda_fr={params['lambda_fr']:.4g}",
            f"gradient_clip_max={params['gradient_clip_max']:.4g}",
            f"mask_rate={params['mask_rate']:.4g}",
        ]
        param_parts.append(f"alpha={params['alpha']:.4g}")
        if variant == "b":
            param_parts.append(f"lambda_ab={params['lambda_ab']:.4g}")
    print("Params:", ", ".join(param_parts))
    activity_fraction = metrics.get("activity_fraction", float("nan"))
    metric_parts = [
        f"activity_fraction={activity_fraction:.4f}",
        f"activity_score={metrics['activity_score']:.4f}",
        f"train_error={metrics.get('train_error', float('nan')):.4f}",
        f"eval_error={metrics.get('eval_error', float('nan')):.4f}",
        f"loss={metrics['loss']:.4f}",
    ]
    if metrics.get("aborted_nan", 0.0) >= 1.0:
        metric_parts.append("aborted_nan=1")
    if variant == "b" and "activation_a_mean" in metrics:
        metric_parts.append(
            "a="
            f"{metrics['activation_a_mean']:.4g}"
            f"±{metrics.get('activation_a_sem', 0.0):.4g}"
        )
        metric_parts.append(
            "b="
            f"{metrics['activation_b_mean']:.4g}"
            f"±{metrics.get('activation_b_sem', 0.0):.4g}"
        )
    if variant == "c" and "activation_A_mean" in metrics:
        metric_parts.append(
            "A="
            f"{metrics['activation_A_mean']:.4g}"
            f"±{metrics.get('activation_A_sem', 0.0):.4g}"
        )
        metric_parts.append(
            "m="
            f"{metrics['activation_m_mean']:.4g}"
            f"±{metrics.get('activation_m_sem', 0.0):.4g}"
        )
    if "ratemap_path" in metrics:
        metric_parts.append(f"ratemap_path={metrics['ratemap_path']}")
    print("Metrics:", ", ".join(metric_parts))

    best_trials = _serialize_best_trials(ax_client, optimize_error=optimize_error)
    if best_trials:
        label = (
            "Pareto-optimal feasible trials"
            if _is_multi_objective(ax_client)
            else "Best feasible trial"
        )
        print(f"{label}:")
        objective_names = _primary_objective_names(optimize_error=optimize_error)
        for row in best_trials[:5]:
            m = row["metrics"]
            objective_parts = ", ".join(
                f"{name}={m.get(name, float('nan')):.4g}" for name in objective_names
            )
            print(
                f"  #{row['trial_index']}: {objective_parts}, "
                f"activity_fraction={m.get('activity_fraction', float('nan')):.4g}"
            )
    elif _trial_is_infeasible(metrics):
        print(
            "Best feasible trial: none yet "
            f"(no feasible trials with {ACTIVITY_FRACTION_LO:.2f} "
            f"≤ activity_fraction ≤ {ACTIVITY_FRACTION_HI:.2f})"
        )


def _probe_params(base_config: ExperimentRunConfig, *, variant: BoVariant) -> dict[str, Any]:
    return fixed_trial_params(base_config, variant=variant)


def _session_config(
    base_config: ExperimentRunConfig,
    optimizer: str | None,
) -> ExperimentRunConfig:
    if optimizer is None:
        return base_config
    session = copy.deepcopy(base_config)
    session.optimizers = [optimizer]
    return session


def _resolve_output_dir(
    *,
    base_output_dir: Path | None,
    experiment_type: str,
    variant: BoVariant,
    optimizer: str | None,
    stamp: str,
    multi_optimizer: bool,
    alt_training: AltTrainingVariant | None = None,
    alt_rank: int | None = None,
) -> Path:
    if base_output_dir is None:
        return _default_output_dir(
            experiment_type,
            variant,
            optimizer=optimizer if multi_optimizer else None,
            stamp=stamp,
            alt_training=alt_training,
            alt_rank=alt_rank,
        )
    if multi_optimizer and optimizer is not None:
        return base_output_dir / optimizer
    return base_output_dir


def _run_bo_session(
    *,
    base_config: ExperimentRunConfig,
    config_path: Path,
    variant: BoVariant,
    optimizer: str | None,
    output_dir: Path,
    n_trials: int,
    resume_path: Path | None,
    args: argparse.Namespace,
    lr_only: bool,
    ab_only: bool,
    use_lambda_fr_constraint: bool,
    optimize_error: bool,
    alt_training: AltTrainingVariant | None = None,
    alt_rank: int | None = None,
) -> None:
    session_config = _session_config(base_config, optimizer)
    output_dir.mkdir(parents=True, exist_ok=True)

    if resume_path is not None:
        if not resume_path.is_file():
            raise SystemExit(f"Resume file not found: {resume_path}")
        source_dir = resume_path.parent
        if output_dir.resolve() != source_dir.resolve():
            output_dir.mkdir(parents=True, exist_ok=True)
        run_meta = _load_run_meta(source_dir)
        saved_variant = run_meta.get("variant")
        if saved_variant != variant:
            raise SystemExit(
                f"Resume variant mismatch: run_meta.json has variant={saved_variant!r}, "
                f"but --variant={variant!r}"
            )
        saved_lr_only = run_meta.get("lr_only", False)
        if saved_lr_only != lr_only:
            raise SystemExit(
                f"Resume lr_only mismatch: run_meta.json has lr_only={saved_lr_only!r}, "
                f"but current run has lr_only={lr_only!r}"
            )
        saved_ab_only = run_meta.get("ab_only", False)
        if saved_ab_only != ab_only:
            raise SystemExit(
                f"Resume ab_only mismatch: run_meta.json has ab_only={saved_ab_only!r}, "
                f"but current run has ab_only={ab_only!r}"
            )
        saved_alt_training = normalize_alt_training_variant(
            run_meta.get("alt_training_variant", run_meta.get("alt_training"))
        )
        if saved_alt_training != alt_training:
            raise SystemExit(
                f"Resume alt_training mismatch: run_meta.json has "
                f"alt_training_variant={saved_alt_training!r}, "
                f"but current run has alt_training={alt_training!r}"
            )
        saved_rank = run_meta.get("rank")
        if alt_training == "lora" and saved_rank != alt_rank:
            raise SystemExit(
                f"Resume rank mismatch: run_meta.json has rank={saved_rank!r}, "
                f"but current run has rank={alt_rank!r}"
            )
        if alt_training == "on_opt" and alt_rank is not None:
            raise SystemExit("--rank is not used with --alt-training on_opt")
        saved_optimizer = run_meta.get("optimizer")
        if optimizer is not None and saved_optimizer not in (None, optimizer):
            raise SystemExit(
                f"Resume optimizer mismatch: run_meta.json has optimizer={saved_optimizer!r}, "
                f"but current session uses optimizer={optimizer!r}"
            )
        bounds = {name: tuple(values) for name, values in run_meta["bounds"].items()}
        param_names = list(bounds.keys())
        saved_lambda_fr_constraint = run_meta.get("lambda_fr_constraint", False)
        saved_optimize_error = run_meta.get("optimize_error", False)
        current_parameter_constraints = bo_parameter_constraints(
            variant=variant,
            enabled=use_lambda_fr_constraint,
            bounds=bounds,
        )
        saved_parameter_constraints = run_meta.get("parameter_constraints")
        if saved_parameter_constraints is None:
            if variant == "c":
                saved_parameter_constraints = bo_parameter_constraints(
                    variant=variant,
                    bounds=bounds,
                )
            else:
                legacy_bounds = {
                    name: bound for name, bound in bounds.items() if name != "lambda_ab"
                }
                saved_parameter_constraints = lambda_fr_mse_parameter_constraints(
                    enabled=saved_lambda_fr_constraint,
                    bounds=legacy_bounds,
                )
        source_client = _load_ax_client(resume_path)

        if (
            _objectives_match(source_client, optimize_error=optimize_error)
            and saved_lambda_fr_constraint == use_lambda_fr_constraint
            and saved_optimize_error == optimize_error
            and saved_parameter_constraints == current_parameter_constraints
        ):
            ax_client = source_client
            completed = len(ax_client.get_trials_data_frame())
            print(f"Resuming from {resume_path} ({completed} completed trials)")
        else:
            source_df = source_client.get_trials_data_frame()
            n_source = len(
                source_df[source_df["trial_status"] == "COMPLETED"]
                if "trial_status" in source_df.columns
                else source_df
            )
            if not _objectives_match(source_client, optimize_error=optimize_error):
                old_objectives = sorted(_ax_objective_names(source_client))
                new_objectives = sorted(
                    _current_objective_names(optimize_error=optimize_error)
                )
                print(
                    f"Objective mismatch ({old_objectives} -> {new_objectives}); "
                    f"migrating {n_source} completed trials from {resume_path}"
                )
            elif saved_lambda_fr_constraint != use_lambda_fr_constraint:
                print(
                    "Parameter-constraint mismatch "
                    f"(saved={saved_lambda_fr_constraint}, "
                    f"current={use_lambda_fr_constraint}); "
                    f"migrating {n_source} completed trials from {resume_path}"
                )
            elif saved_optimize_error != optimize_error:
                print(
                    "Objective-mode mismatch "
                    f"(saved optimize_error={saved_optimize_error}, "
                    f"current={optimize_error}); "
                    f"migrating {n_source} completed trials from {resume_path}"
                )
            elif saved_parameter_constraints != current_parameter_constraints:
                print(
                    "Parameter-constraint mismatch "
                    f"(saved={saved_parameter_constraints}, "
                    f"current={current_parameter_constraints}); "
                    f"migrating {n_source} completed trials from {resume_path}"
                )
            if output_dir.resolve() != source_dir.resolve():
                _copy_run_artifacts(source_dir, output_dir)
            ax_client = _create_ax_client(
                experiment_type=session_config.experiment_type,
                variant=variant,
                bounds=bounds,
                random_seed=args.random_seed,
                sobol_trials=args.sobol_trials,
                use_lambda_fr_constraint=use_lambda_fr_constraint,
                optimize_error=optimize_error,
                optimizer=optimizer,
                alt_training=alt_training,
                alt_rank=alt_rank,
            )
            completed = _attach_completed_trials(
                ax_client,
                source_df,
                param_names,
                optimize_error=optimize_error,
            )
            _write_run_meta(
                output_dir,
                variant=variant,
                config_path=config_path,
                bounds=bounds,
                use_lambda_fr_constraint=use_lambda_fr_constraint,
                optimize_error=optimize_error,
                lr_only=lr_only,
                ab_only=ab_only,
                optimizer=optimizer,
                alt_training=alt_training,
                alt_rank=alt_rank,
                migrated_from=resume_path,
            )
            ax_client.save_to_json_file(str(output_dir / "ax_experiment.json"))
            print(
                f"Attached {completed} trials; continuing in {output_dir} "
                f"from trial {completed + 1}"
            )

        start_trial = completed + 1
    else:
        bounds = search_bounds_from_config(
            session_config,
            variant=variant,
            lr_only=lr_only,
            ab_only=ab_only,
            lambda_mse_bounds=args.lambda_mse_bounds,
            lambda_fr_bounds=args.lambda_fr_bounds,
            gradient_clip_bounds=args.gradient_clip_bounds,
            lambda_ab_bounds=args.lambda_ab_bounds,
            alpha_bounds=args.alpha_bounds,
            nl_bounds=args.nl_bounds,
            nr_bounds=args.nr_bounds,
            s_bounds=args.s_bounds,
            apply_ab_constraint=use_lambda_fr_constraint,
            alt_training=alt_training,
            beta_bounds=args.beta_bounds,
            kappa_bounds=args.kappa_bounds,
        )
        _write_run_meta(
            output_dir,
            variant=variant,
            config_path=config_path,
            bounds=bounds,
            use_lambda_fr_constraint=use_lambda_fr_constraint,
            optimize_error=optimize_error,
            lr_only=lr_only,
            ab_only=ab_only,
            optimizer=optimizer,
            alt_training=alt_training,
            alt_rank=alt_rank,
        )
        ax_client = _create_ax_client(
            experiment_type=session_config.experiment_type,
            variant=variant,
            bounds=bounds,
            random_seed=args.random_seed,
            sobol_trials=args.sobol_trials,
            use_lambda_fr_constraint=use_lambda_fr_constraint,
            optimize_error=optimize_error,
            optimizer=optimizer,
            alt_training=alt_training,
            alt_rank=alt_rank,
        )
        ax_client.save_to_json_file(str(output_dir / "ax_experiment.json"))
        start_trial = 1

        probe_params = _probe_params(session_config, variant=variant)
        probe_config = apply_trial_params(
            session_config,
            probe_params,
            variant=variant,
            lr_only=lr_only,
            ab_only=ab_only,
        )
        probe_experiment = create_experiment(probe_config)
        probe_rooms = probe_experiment.load_rooms()
        n_segments = estimate_trial_segments(probe_experiment, probe_rooms)
        if session_config.experiment_type == "single_room":
            n_epochs = probe_config.config.n_epochs
            training_unit = f"{n_epochs} epoch{'s' if n_epochs != 1 else ''}"
        elif session_config.experiment_type == "two_rooms":
            training_unit = "1 repetition"
        else:
            training_unit = "1 cycle"
        opt_label = f", optimizer={optimizer}" if optimizer is not None else ""
        if lr_only:
            mode_label = "lr-only"
        elif ab_only:
            mode_label = "ab-only"
        elif alt_training == "lora":
            mode_label = "altTLORA-beta-only"
        elif alt_training == "on_opt":
            mode_label = "altTOO-kappa-only"
        elif variant == "c":
            mode_label = "bump-shape"
        else:
            mode_label = "full"
        print(
            f"[{session_config.experiment_type}] variant {variant} ({mode_label}{opt_label}): "
            f"each trial warmup + {training_unit} (~{n_segments} TBPTT segments). "
            f"Output: {output_dir}"
        )
        print("Search bounds:", json.dumps(bounds, indent=2, default=list))
        objectives, outcome_constraints, tracking_metrics = _ax_setup(
            optimize_error=optimize_error
        )
        parameter_constraints = bo_parameter_constraints(
            variant=variant,
            enabled=use_lambda_fr_constraint,
            bounds=bounds,
        )
        print("Objectives:", ", ".join(objectives))
        print("Tracking metrics:", tracking_metrics or "none")
        print("Outcome constraints:", outcome_constraints or "none")
        print(
            "Parameter constraints:",
            parameter_constraints or "none",
        )

    if start_trial > n_trials:
        print(f"All {n_trials} trials already completed.")
        return

    fixed_params = fixed_trial_params(session_config, variant=variant)

    def _run_one_trial(
        trial_index: int, params: dict[str, Any]
    ) -> tuple[int, dict[str, Any], dict[str, float]]:
        trial_config = apply_trial_params(
            session_config,
            params,
            variant=variant,
            lr_only=lr_only,
            ab_only=ab_only,
            alt_training=alt_training,
        )
        metrics = run_hyperparam_trial(
            trial_config,
            params,
            variant=variant,
            trial_index=trial_index,
            output_dir=output_dir,
            show_progress=args.show_progress,
            alt_training=alt_training,
            alt_rank=alt_rank,
        )
        if lr_only or ab_only or alt_training:
            merged_params = {**fixed_params, **params}
        elif variant == "c":
            merged_params = {**fixed_params, **params}
        else:
            merged_params = params
        return trial_index, merged_params, metrics

    trial_num = start_trial
    while trial_num <= n_trials:
        batch_size = min(args.parallel_trials, n_trials - trial_num + 1)
        trials, _ = ax_client.get_next_trials(max_trials=batch_size)

        if batch_size == 1:
            trial_index, params = next(iter(trials.items()))
            completed = [_run_one_trial(trial_index, params)]
        else:
            completed = []
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {
                    executor.submit(_run_one_trial, trial_index, params): trial_index
                    for trial_index, params in trials.items()
                }
                for future in as_completed(futures):
                    completed.append(future.result())
            completed.sort(key=lambda row: row[0])

        for trial_index, params, metrics in completed:
            ax_client.complete_trial(
                trial_index=trial_index,
                raw_data=_ax_raw_data(metrics, optimize_error=optimize_error),
            )

            timestamp = datetime.now(timezone.utc).isoformat()
            row = {
                "trial_index": trial_index,
                "timestamp": timestamp,
                **params,
                **metrics,
            }
            _append_trial_csv(
                output_dir,
                row,
                variant=variant,
                lr_only=lr_only,
                ab_only=ab_only,
                alt_training=alt_training,
            )
            _print_trial_summary(
                trial_num,
                n_trials,
                trial_index,
                params,
                metrics,
                ax_client,
                variant=variant,
                optimize_error=optimize_error,
                lr_only=lr_only,
                ab_only=ab_only,
                optimizer=optimizer,
                alt_training=alt_training,
            )
            trial_num += 1

        ax_client.save_to_json_file(str(output_dir / "ax_experiment.json"))
        _write_summary(output_dir, ax_client, optimize_error=optimize_error)

    print(f"\nDone. Results in {output_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter optimization (Ax + BoTorch, cell masking only)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Base experiment JSON, e.g. input_configs/single_room.json",
    )
    parser.add_argument(
        "--variant",
        choices=("a", "b", "c"),
        default="a",
        help=(
            "BO variant: a=ReLU RAE, b=custom activation RAE, "
            "c=BumpActivation RAE (default: a)"
        ),
    )
    parser.add_argument("--n-trials", type=int, default=30, help="Total BO trials to run")
    parser.add_argument(
        "--sobol-trials",
        type=int,
        default=5,
        help="Initial quasi-random Sobol trials before GP (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for trials.csv, ax_experiment.json, summary.json",
    )
    parser.add_argument("--random-seed", type=int, default=None, help="Ax random seed")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Resume from a saved ax_experiment.json (uses its parent dir as output-dir). "
            "If objectives differ from the current setup, completed trials are attached "
            "into a new experiment and optimization continues."
        ),
    )
    parser.add_argument(
        "--lambda-mse-bounds",
        type=lambda s: _parse_bounds(s, "lambda_mse"),
        default=None,
        help="Override lambda_mse search bounds as 'low,high'",
    )
    parser.add_argument(
        "--lambda-fr-bounds",
        type=lambda s: _parse_bounds(s, "lambda_fr"),
        default=None,
        help="Override lambda_fr search bounds as 'low,high'",
    )
    parser.add_argument(
        "--gradient-clip-bounds",
        type=lambda s: _parse_bounds(s, "gradient_clip_max"),
        default=None,
        help="Override gradient_clip_max search bounds as 'low,high'",
    )
    parser.add_argument(
        "--lambda-ab-bounds",
        type=lambda s: _parse_bounds(s, "lambda_ab"),
        default=None,
        help="Override lambda_ab search bounds as 'low,high' (variant B only)",
    )
    parser.add_argument(
        "--alpha-bounds",
        type=lambda s: _parse_bounds(s, "alpha"),
        default=None,
        help="Override alpha search bounds as 'low,high'",
    )
    parser.add_argument(
        "--nl-bounds",
        type=lambda s: _parse_bounds(s, "nl"),
        default=None,
        help="Override nl search bounds as 'low,high' (variant C only)",
    )
    parser.add_argument(
        "--nr-bounds",
        type=lambda s: _parse_bounds(s, "nr"),
        default=None,
        help="Override nr search bounds as 'low,high' (variant C only)",
    )
    parser.add_argument(
        "--s-bounds",
        type=lambda s: _parse_bounds(s, "s"),
        default=None,
        help="Override s search bounds as 'low,high' (variant C only)",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show warmup trajectory progress bars during each trial",
    )
    parser.add_argument(
        "--parallel-trials",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Run up to N BO trials concurrently on the same GPU (default: 1). "
            "Use 2 when GPU utilization is low; watch VRAM usage."
        ),
    )
    parser.add_argument(
        "--no-constraint",
        action="store_true",
        help=(
            "Disable the lambda_fr <= MAX_FR_FACTOR * lambda_mse parameter constraint "
            f"(default factor: {MAX_FR_FACTOR:g}). Applies to variants A and B."
        ),
    )
    parser.add_argument(
        "--error",
        action="store_true",
        help=(
            "Optimize train_error (mean train-set prediction MSE) instead of loss; "
            "still logs loss and eval_error."
        ),
    )
    parser.add_argument(
        "--lr-only",
        action="store_true",
        help="Tune only optimizer lr in [1e-5, 0.1] (log scale); fix other hyperparams from config.",
    )
    parser.add_argument(
        "--ab-only",
        action="store_true",
        help=(
            "Variant B only: tune only lambda_ab; fix other hyperparams from config."
        ),
    )
    parser.add_argument(
        "--alt-training",
        choices=["lora", "on_opt"],
        default=None,
        metavar="VARIANT",
        help=(
            "Alt-training variant: 'lora' uses a low-rank BTSP adapter (W + beta * B @ A) "
            "with per-unit g_local_mse scaling on LoRA factors (tune beta in [0, 1]; "
            "requires --rank; paths include altTLORA_r<rank>). "
            "'on_opt' scales all gradients after backward by clipped kappa * inverse "
            "z-score from g_local_mse (tune kappa in [1, 3]; paths include altTOO). "
            "Other hyperparameters are fixed from config."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        metavar="R",
        help="LoRA rank for --alt-training lora (required with lora variant).",
    )
    parser.add_argument(
        "--beta-bounds",
        type=lambda s: _parse_bounds(s, "beta"),
        default=None,
        help="Override beta search bounds as 'low,high' (default: 0.0,1.0; lora only)",
    )
    parser.add_argument(
        "--kappa-bounds",
        type=lambda s: _parse_bounds(s, "kappa"),
        default=None,
        help="Override kappa search bounds as 'low,high' (default: 1.0,3.0; on_opt only)",
    )
    parser.add_argument(
        "--exclude-opt",
        choices=sorted(OPTIMIZER_SHORTHAND_TO_CLASS),
        default=None,
        help=(
            "Skip one optimizer when the config lists multiple optimizers "
            "(or optimizers: all). Remaining optimizers each get their own BO session."
        ),
    )
    args = parser.parse_args(argv)
    variant: BoVariant = args.variant

    if args.parallel_trials < 1:
        raise SystemExit("--parallel-trials must be >= 1")
    if args.parallel_trials > 1 and args.show_progress:
        print("Note: --show-progress is disabled when --parallel-trials > 1")
        args.show_progress = False

    if not args.config.is_file():
        raise SystemExit(f"Config file not found: {args.config}")

    raw_optimizers = raw_optimizers_value(args.config)
    configured_optimizers = parse_optimizers(raw_optimizers, path=args.config)
    if args.exclude_opt is not None:
        if args.exclude_opt not in configured_optimizers:
            raise SystemExit(
                f"--exclude-opt {args.exclude_opt!r} is not among config optimizers "
                f"{configured_optimizers}"
            )
        configured_optimizers = [
            opt for opt in configured_optimizers if opt != args.exclude_opt
        ]
        if not configured_optimizers:
            raise SystemExit(
                "--exclude-opt would leave no optimizers to tune."
            )

    if args.lr_only and args.ab_only:
        raise SystemExit("--lr-only and --ab-only are mutually exclusive")
    if args.alt_training and (args.lr_only or args.ab_only):
        raise SystemExit("--alt-training is mutually exclusive with --lr-only and --ab-only")
    if args.alt_training == "lora" and args.rank is None:
        raise SystemExit("--alt-training lora requires --rank")
    if args.rank is not None and args.alt_training != "lora":
        raise SystemExit("--rank requires --alt-training lora")
    if args.rank is not None and args.rank < 1:
        raise SystemExit("--rank must be >= 1")
    if args.beta_bounds is not None and args.alt_training != "lora":
        raise SystemExit("--beta-bounds requires --alt-training lora")
    if args.kappa_bounds is not None and args.alt_training != "on_opt":
        raise SystemExit("--kappa-bounds requires --alt-training on_opt")
    if args.ab_only and variant != "b":
        raise SystemExit("--ab-only requires --variant b")
    if variant == "c" and (args.lr_only or args.ab_only):
        raise SystemExit("variant c does not support --lr-only or --ab-only")

    if args.alt_training:
        bound_overrides = [
            name
            for name, value in (
                ("--lambda-mse-bounds", args.lambda_mse_bounds),
                ("--lambda-fr-bounds", args.lambda_fr_bounds),
                ("--gradient-clip-bounds", args.gradient_clip_bounds),
                ("--lambda-ab-bounds", args.lambda_ab_bounds),
                ("--alpha-bounds", args.alpha_bounds),
                ("--nl-bounds", args.nl_bounds),
                ("--nr-bounds", args.nr_bounds),
                ("--s-bounds", args.s_bounds),
            )
            if value is not None
        ]
        if bound_overrides:
            print(
                f"Note: ignoring hyperparameter bound overrides in --alt-training "
                f"{args.alt_training} mode: "
                + ", ".join(bound_overrides)
            )
    elif args.lr_only:
        bound_overrides = [
            name
            for name, value in (
                ("--lambda-mse-bounds", args.lambda_mse_bounds),
                ("--lambda-fr-bounds", args.lambda_fr_bounds),
                ("--gradient-clip-bounds", args.gradient_clip_bounds),
                ("--lambda-ab-bounds", args.lambda_ab_bounds),
                ("--alpha-bounds", args.alpha_bounds),
            )
            if value is not None
        ]
        if bound_overrides:
            print(
                "Note: ignoring hyperparameter bound overrides in --lr-only mode: "
                + ", ".join(bound_overrides)
            )
    elif args.ab_only:
        bound_overrides = [
            name
            for name, value in (
                ("--lambda-mse-bounds", args.lambda_mse_bounds),
                ("--lambda-fr-bounds", args.lambda_fr_bounds),
                ("--gradient-clip-bounds", args.gradient_clip_bounds),
                ("--alpha-bounds", args.alpha_bounds),
            )
            if value is not None
        ]
        if bound_overrides:
            print(
                "Note: ignoring hyperparameter bound overrides in --ab-only mode: "
                + ", ".join(bound_overrides)
            )

    base_config = load_experiment_config(args.config)
    if base_config.config.training.mask_method != "cell":
        raise SystemExit(
            "Bayesian optimization requires mask_method='cell' in the config; "
            f"got {base_config.config.training.mask_method!r}"
        )
    _verify_room_data(base_config)
    lr_only = args.lr_only
    ab_only = args.ab_only
    alt_training: AltTrainingVariant | None = args.alt_training
    alt_rank = args.rank if alt_training == "lora" else None
    use_lambda_fr_constraint = (
        not args.no_constraint
        and not lr_only
        and not ab_only
        and alt_training is None
        and variant != "c"
    )
    optimize_error = args.error

    if args.resume is not None:
        resume_meta = _load_run_meta(args.resume.parent)
        optimizers_to_run: list[str | None] = [resume_meta.get("optimizer")]
    elif len(configured_optimizers) > 1:
        optimizers_to_run = configured_optimizers
    else:
        optimizers_to_run = [None]

    multi_optimizer = len(optimizers_to_run) > 1
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    for optimizer in optimizers_to_run:
        if multi_optimizer and optimizer is not None:
            print(f"\n=== Optimizer: {optimizer} ===")
        if args.resume is not None:
            output_dir = args.output_dir or args.resume.parent
            resume_path = args.resume
        else:
            output_dir = _resolve_output_dir(
                base_output_dir=args.output_dir,
                experiment_type=base_config.experiment_type,
                variant=variant,
                optimizer=optimizer,
                stamp=stamp,
                multi_optimizer=multi_optimizer,
                alt_training=alt_training,
                alt_rank=alt_rank,
            )
            resume_path = None
        _run_bo_session(
            base_config=base_config,
            config_path=args.config,
            variant=variant,
            optimizer=optimizer,
            output_dir=output_dir,
            n_trials=args.n_trials,
            resume_path=resume_path,
            args=args,
            lr_only=lr_only,
            ab_only=ab_only,
            use_lambda_fr_constraint=use_lambda_fr_constraint,
            optimize_error=optimize_error,
            alt_training=alt_training,
            alt_rank=alt_rank,
        )


if __name__ == "__main__":
    main()
