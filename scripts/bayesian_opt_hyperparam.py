#!/usr/bin/env python3
"""
Bayesian hyperparameter optimization for room-based experiments.
Two variants -- normal RNN, or RNN with learnable activation function: 

Variant A: ReLU RAE; tunes lambda_mse, lambda_fr, mask_rate, gradient_clip_max, alpha.
Variant B: custom activation RAE; same plus lambda_ab; 

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
    uv run bayesian-opt-hyperparam --config input_configs/single_room.json --variant b --n-trials 30 --parallel-trials 2
"""
import argparse
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
    BoVariant,
    apply_trial_params,
    estimate_trial_segments,
    lambda_fr_mse_parameter_constraints,
    run_hyperparam_trial,
    search_bounds_from_config,
)
from experiments.common.run import create_experiment, load_experiment_config

ACTIVITY_FRACTION_LO = ACTIVITY_TARGET - 0.3
ACTIVITY_FRACTION_HI = ACTIVITY_TARGET + 0.2

TRIAL_METRIC_FIELDS = [
    "activity_fraction",
    "activity_score",
    "loss",
    "train_error",
    "eval_error",
]

VARIANT_B_METRIC_FIELDS = [
    "activation_a_mean",
    "activation_a_sem",
    "activation_b_mean",
    "activation_b_sem",
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


def _default_output_dir(experiment_type: str, variant: BoVariant) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("results") / "hyperparam_opt" / experiment_type / f"variant_{variant}_{stamp}"


def _trials_csv_fields(variant: BoVariant) -> list[str]:
    return TRIALS_CSV_FIELDS_B if variant == "b" else TRIALS_CSV_FIELDS_A


def _ax_parameters(bounds: dict[str, Any]) -> list[dict[str, Any]]:
    parameters: list[dict[str, Any]] = []
    for name, bound in bounds.items():
        low, high = bound
        parameters.append(
            {
                "name": name,
                "type": "range",
                "bounds": [low, high],
                "value_type": "float",
            }
        )
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
) -> AxClient:
    objectives, outcome_constraints, tracking_metric_names = _ax_setup(
        optimize_error=optimize_error
    )
    parameter_constraints = lambda_fr_mse_parameter_constraints(
        enabled=use_lambda_fr_constraint,
        bounds=bounds,
    )
    ax_client = AxClient(random_seed=random_seed, verbose_logging=True)
    ax_client.create_experiment(
        name=f"hyperparam_{experiment_type}_variant_{variant}",
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
    migrated_from: Path | None = None,
) -> None:
    meta: dict[str, Any] = {
        "variant": variant,
        "config_path": str(config_path.resolve()),
        "bounds": {k: list(v) for k, v in bounds.items()},
        "objectives": sorted(_current_objective_names(optimize_error=optimize_error)),
        "lambda_fr_constraint": use_lambda_fr_constraint,
        "parameter_constraints": lambda_fr_mse_parameter_constraints(
            enabled=use_lambda_fr_constraint,
            bounds=bounds,
        ),
        "optimize_error": optimize_error,
        "max_fr_factor": MAX_FR_FACTOR,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
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


def _append_trial_csv(output_dir: Path, row: dict[str, Any], *, variant: BoVariant) -> None:
    csv_path = output_dir / "trials.csv"
    fieldnames = _trials_csv_fields(variant)
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
) -> None:
    print(f"\n=== Trial {trial_num}/{n_trials} (index {trial_index}) [variant {variant}] ===")
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


def _probe_params(base_config, *, variant: BoVariant) -> dict[str, Any]:
    training = base_config.config.training
    params: dict[str, Any] = {
        "lambda_mse": training.lambda_mse,
        "lambda_fr": training.lambda_fr,
        "gradient_clip_max": training.gradient_clip_max or 10.0,
        "mask_rate": training.mask_rate,
    }
    params["alpha"] = base_config.config.model.alpha
    if variant == "b":
        params["lambda_ab"] = training.lambda_fr / 100.0
    return params


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
        choices=("a", "b"),
        default="a",
        help="BO variant: a=ReLU RAE, b=custom activation RAE (default: a)",
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
    args = parser.parse_args(argv)
    variant: BoVariant = args.variant

    if args.parallel_trials < 1:
        raise SystemExit("--parallel-trials must be >= 1")
    if args.parallel_trials > 1 and args.show_progress:
        print("Note: --show-progress is disabled when --parallel-trials > 1")
        args.show_progress = False

    if not args.config.is_file():
        raise SystemExit(f"Config file not found: {args.config}")

    base_config = load_experiment_config(args.config)
    if base_config.config.training.mask_method != "cell":
        raise SystemExit(
            "Bayesian optimization requires mask_method='cell' in the config; "
            f"got {base_config.config.training.mask_method!r}"
        )
    _verify_room_data(base_config)
    use_lambda_fr_constraint = not args.no_constraint
    optimize_error = args.error

    if args.resume is not None:
        if not args.resume.is_file():
            raise SystemExit(f"Resume file not found: {args.resume}")
        source_dir = args.resume.parent
        output_dir = args.output_dir or source_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        run_meta = _load_run_meta(source_dir)
        saved_variant = run_meta.get("variant")
        if saved_variant != variant:
            raise SystemExit(
                f"Resume variant mismatch: run_meta.json has variant={saved_variant!r}, "
                f"but --variant={variant!r}"
            )
        bounds = {name: tuple(values) for name, values in run_meta["bounds"].items()}
        param_names = list(bounds.keys())
        saved_lambda_fr_constraint = run_meta.get("lambda_fr_constraint", False)
        saved_optimize_error = run_meta.get("optimize_error", False)
        current_parameter_constraints = lambda_fr_mse_parameter_constraints(
            enabled=use_lambda_fr_constraint,
            bounds=bounds,
        )
        saved_parameter_constraints = run_meta.get("parameter_constraints")
        if saved_parameter_constraints is None:
            legacy_bounds = {
                name: bound for name, bound in bounds.items() if name != "lambda_ab"
            }
            saved_parameter_constraints = lambda_fr_mse_parameter_constraints(
                enabled=saved_lambda_fr_constraint,
                bounds=legacy_bounds,
            )
        source_client = _load_ax_client(args.resume)

        if (
            _objectives_match(source_client, optimize_error=optimize_error)
            and saved_lambda_fr_constraint == use_lambda_fr_constraint
            and saved_optimize_error == optimize_error
            and saved_parameter_constraints == current_parameter_constraints
        ):
            ax_client = source_client
            completed = len(ax_client.get_trials_data_frame())
            print(f"Resuming from {args.resume} ({completed} completed trials)")
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
                    f"migrating {n_source} completed trials from {args.resume}"
                )
            elif saved_lambda_fr_constraint != use_lambda_fr_constraint:
                print(
                    "Parameter-constraint mismatch "
                    f"(saved={saved_lambda_fr_constraint}, "
                    f"current={use_lambda_fr_constraint}); "
                    f"migrating {n_source} completed trials from {args.resume}"
                )
            elif saved_optimize_error != optimize_error:
                print(
                    "Objective-mode mismatch "
                    f"(saved optimize_error={saved_optimize_error}, "
                    f"current={optimize_error}); "
                    f"migrating {n_source} completed trials from {args.resume}"
                )
            elif saved_parameter_constraints != current_parameter_constraints:
                print(
                    "Parameter-constraint mismatch "
                    f"(saved={saved_parameter_constraints}, "
                    f"current={current_parameter_constraints}); "
                    f"migrating {n_source} completed trials from {args.resume}"
                )
            if output_dir.resolve() != source_dir.resolve():
                _copy_run_artifacts(source_dir, output_dir)
            ax_client = _create_ax_client(
                experiment_type=base_config.experiment_type,
                variant=variant,
                bounds=bounds,
                random_seed=args.random_seed,
                sobol_trials=args.sobol_trials,
                use_lambda_fr_constraint=use_lambda_fr_constraint,
                optimize_error=optimize_error,
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
                config_path=args.config,
                bounds=bounds,
                use_lambda_fr_constraint=use_lambda_fr_constraint,
                optimize_error=optimize_error,
                migrated_from=args.resume,
            )
            ax_client.save_to_json_file(str(output_dir / "ax_experiment.json"))
            print(
                f"Attached {completed} trials; continuing in {output_dir} "
                f"from trial {completed + 1}"
            )

        n_trials = args.n_trials
        start_trial = completed + 1
    else:
        output_dir = args.output_dir or _default_output_dir(base_config.experiment_type, variant)
        output_dir.mkdir(parents=True, exist_ok=True)
        bounds = search_bounds_from_config(
            base_config,
            variant=variant,
            lambda_mse_bounds=args.lambda_mse_bounds,
            lambda_fr_bounds=args.lambda_fr_bounds,
            gradient_clip_bounds=args.gradient_clip_bounds,
            lambda_ab_bounds=args.lambda_ab_bounds,
            alpha_bounds=args.alpha_bounds,
        )
        _write_run_meta(
            output_dir,
            variant=variant,
            config_path=args.config,
            bounds=bounds,
            use_lambda_fr_constraint=use_lambda_fr_constraint,
            optimize_error=optimize_error,
        )
        ax_client = _create_ax_client(
            experiment_type=base_config.experiment_type,
            variant=variant,
            bounds=bounds,
            random_seed=args.random_seed,
            sobol_trials=args.sobol_trials,
            use_lambda_fr_constraint=use_lambda_fr_constraint,
            optimize_error=optimize_error,
        )
        ax_client.save_to_json_file(str(output_dir / "ax_experiment.json"))
        n_trials = args.n_trials
        start_trial = 1

        probe_params = _probe_params(base_config, variant=variant)
        probe_config = apply_trial_params(base_config, probe_params, variant=variant)
        probe_experiment = create_experiment(probe_config)
        probe_rooms = probe_experiment.load_rooms()
        n_segments = estimate_trial_segments(probe_experiment, probe_rooms)
        if base_config.experiment_type == "single_room":
            n_epochs = probe_config.config.n_epochs
            training_unit = f"{n_epochs} epoch{'s' if n_epochs != 1 else ''}"
        elif base_config.experiment_type == "two_rooms":
            training_unit = "1 repetition"
        else:
            training_unit = "1 cycle"
        print(
            f"[{base_config.experiment_type}] variant {variant}: each trial warmup + {training_unit} "
            f"(~{n_segments} TBPTT segments). Output: {output_dir}"
        )
        print("Search bounds:", json.dumps(bounds, indent=2, default=list))
        objectives, outcome_constraints, tracking_metrics = _ax_setup(
            optimize_error=optimize_error
        )
        parameter_constraints = lambda_fr_mse_parameter_constraints(
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

    def _run_one_trial(
        trial_index: int, params: dict[str, Any]
    ) -> tuple[int, dict[str, Any], dict[str, float]]:
        trial_config = apply_trial_params(base_config, params, variant=variant)
        metrics = run_hyperparam_trial(
            trial_config,
            params,
            variant=variant,
            trial_index=trial_index,
            output_dir=output_dir,
            show_progress=args.show_progress,
        )
        return trial_index, params, metrics

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
            _append_trial_csv(output_dir, row, variant=variant)
            _print_trial_summary(
                trial_num,
                n_trials,
                trial_index,
                params,
                metrics,
                ax_client,
                variant=variant,
                optimize_error=optimize_error,
            )
            trial_num += 1

        ax_client.save_to_json_file(str(output_dir / "ax_experiment.json"))
        _write_summary(output_dir, ax_client, optimize_error=optimize_error)

    print(f"\nDone. Results in {output_dir}")


if __name__ == "__main__":
    main()
