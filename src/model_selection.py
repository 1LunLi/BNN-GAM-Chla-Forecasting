"""
Hyperparameter selection for the BNN-GAM workflow.

Only the training and validation sets are used in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .bnn_model import train_bnn, predict_bnn


@dataclass
class SearchRecord:
    """Results from one hyperparameter trial."""

    trial_id: int
    configuration: Dict[str, Any]
    validation_rmse: Optional[float] = None
    validation_mae: Optional[float] = None
    validation_r2: Optional[float] = None
    coverage_95: Optional[float] = None
    selection_score: Optional[float] = None
    status: str = "completed"
    error: Optional[str] = None


@dataclass
class SelectionResult:
    """Output of the validation-based model selection procedure."""

    best_configuration: Dict[str, Any]
    best_score: float
    best_training_result: Any
    best_prediction_result: Any
    history: List[SearchRecord] = field(default_factory=list)

    def history_dataframe(self) -> pd.DataFrame:
        """Return the search history as a table."""
        return pd.DataFrame(
            [
                {
                    "trial_id": record.trial_id,
                    **record.configuration,
                    "validation_rmse": record.validation_rmse,
                    "validation_mae": record.validation_mae,
                    "validation_r2": record.validation_r2,
                    "coverage_95": record.coverage_95,
                    "selection_score": record.selection_score,
                    "status": record.status,
                    "error": record.error,
                }
                for record in self.history
            ]
        )


def default_search_space() -> Dict[str, List[Any]]:
    """
    Return a compact default search space.

    The values can be modified according to computational resources.
    `num_samples` is intentionally excluded because it is an evaluation
    setting rather than a model-training hyperparameter.
    """
    return {
        "hidden_sizes": [(100, 50, 25)],
        "sigma_range": [
            (0.001, 0.05),
            (0.01, 0.10),
            (0.05, 0.20),
        ],
        "learning_rate": [0.001, 0.003],
        "batch_size": [64, 128],
        "patience": [100],
        "dropout_rate": [0.10, 0.20],
    }


def expand_search_space(
    search_space: Dict[str, List[Any]],
) -> List[Dict[str, Any]]:
    """Expand a dictionary-based search space into configurations."""
    keys = list(search_space.keys())
    values = [search_space[key] for key in keys]

    return [
        dict(zip(keys, combination))
        for combination in product(*values)
    ]


def _as_float(value: Any) -> Optional[float]:
    """Convert scalar-like values to Python floats."""
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value):
        return None

    return value


def _get_metric(metrics: Any, *names: str) -> Optional[float]:
    """Read a metric from either a dictionary or an object."""
    if metrics is None:
        return None

    if isinstance(metrics, dict):
        for name in names:
            if name in metrics:
                return _as_float(metrics[name])

    for name in names:
        if hasattr(metrics, name):
            return _as_float(getattr(metrics, name))

    return None


def _extract_metrics(prediction_result: Any) -> Dict[str, Optional[float]]:
    """Extract validation metrics from PredictionResult."""
    metrics = getattr(prediction_result, "metrics", prediction_result)

    return {
        "rmse": _get_metric(metrics, "rmse", "RMSE", "root_mean_squared_error"),
        "mae": _get_metric(metrics, "mae", "MAE", "mean_absolute_error"),
        "r2": _get_metric(metrics, "r2", "R2", "r_squared"),
        "coverage_95": _get_metric(
            metrics,
            "coverage_95",
            "coverage",
            "interval_coverage",
        ),
    }


def calculate_validation_score(
    metrics: Dict[str, Optional[float]],
    target_coverage: float = 0.95,
    coverage_weight: float = 0.30,
) -> float:
    """
    Calculate a validation-only model selection score.

    The score rewards:
    - lower validation RMSE;
    - higher validation R2;
    - 95% prediction-interval coverage close to the nominal level.

    The score is used only for ranking candidate models. It must not be
    calculated using the test set.
    """
    rmse = metrics.get("rmse")
    r2 = metrics.get("r2")
    coverage = metrics.get("coverage_95")

    if rmse is None or r2 is None:
        raise ValueError(
            "Validation RMSE and R2 are required for model selection."
        )

    # Bounded accuracy component.
    r2_component = np.clip((r2 + 1.0) / 2.0, 0.0, 1.0)

    # Smaller RMSE receives a larger score.
    rmse_component = 1.0 / (1.0 + max(rmse, 0.0))

    accuracy_component = (
        0.50 * r2_component
        + 0.50 * rmse_component
    )

    if coverage is None:
        coverage_component = 0.0
    else:
        coverage_component = max(
            0.0,
            1.0 - abs(coverage - target_coverage) / target_coverage,
        )

    score = (
        (1.0 - coverage_weight) * accuracy_component
        + coverage_weight * coverage_component
    )

    return float(score)


def select_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    search_space: Optional[Dict[str, List[Any]]] = None,
    num_samples: int = 500,
    target_coverage: float = 0.95,
    coverage_weight: float = 0.30,
    random_seed: int = 42,
    continue_on_error: bool = True,
    **fixed_training_kwargs: Any,
) -> SelectionResult:
    """
    Select BNN hyperparameters using training and validation data only.

    Parameters
    ----------
    X_train, y_train:
        Training data.

    X_validation, y_validation:
        Validation data used for hyperparameter selection.

    search_space:
        Dictionary containing candidate values for each hyperparameter.

    num_samples:
        Number of posterior predictive samples used for validation metrics.

    target_coverage:
        Nominal prediction interval coverage, normally 0.95.

    coverage_weight:
        Weight assigned to interval calibration.

    random_seed:
        Base random seed. A different deterministic seed is used per trial.

    continue_on_error:
        If True, failed trials are recorded and the search continues.

    fixed_training_kwargs:
        Additional fixed arguments passed to `train_bnn`.

    Returns
    -------
    SelectionResult
        Best configuration, best training result, best validation prediction,
        and the complete search history.
    """
    if search_space is None:
        search_space = default_search_space()

    configurations = expand_search_space(search_space)

    if not configurations:
        raise ValueError("The search space contains no configurations.")

    history: List[SearchRecord] = []
    best_score = -np.inf
    best_configuration = None
    best_training_result = None
    best_prediction_result = None

    for trial_id, configuration in enumerate(configurations, start=1):
        record = SearchRecord(
            trial_id=trial_id,
            configuration=configuration.copy(),
        )

        try:
            trial_seed = random_seed + trial_id - 1

            training_kwargs = {
                **configuration,
                **fixed_training_kwargs,
                "random_seed": trial_seed,
            }

            # The test set is deliberately absent from this call.
            training_result = train_bnn(
                X_train,
                y_train,
                X_validation,
                y_validation,
                **training_kwargs,
            )

            prediction_result = predict_bnn(
                training_result,
                X_validation,
                y_validation,
                num_samples=num_samples,
                random_seed=trial_seed,
            )

            metrics = _extract_metrics(prediction_result)

            record.validation_rmse = metrics["rmse"]
            record.validation_mae = metrics["mae"]
            record.validation_r2 = metrics["r2"]
            record.coverage_95 = metrics["coverage_95"]

            record.selection_score = calculate_validation_score(
                metrics=metrics,
                target_coverage=target_coverage,
                coverage_weight=coverage_weight,
            )

            if record.selection_score > best_score:
                best_score = record.selection_score
                best_configuration = configuration.copy()
                best_training_result = training_result
                best_prediction_result = prediction_result

        except Exception as exc:
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"

            if not continue_on_error:
                raise

        history.append(record)

    if best_configuration is None:
        error_messages = [
            record.error
            for record in history
            if record.error is not None
        ]

        raise RuntimeError(
            "All hyperparameter trials failed. "
            + " | ".join(error_messages)
        )

    return SelectionResult(
        best_configuration=best_configuration,
        best_score=float(best_score),
        best_training_result=best_training_result,
        best_prediction_result=best_prediction_result,
        history=history,
    )
