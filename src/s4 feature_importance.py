"""
Permutation feature importance for the BNN-GAM workflow.

Feature importance is calculated on the validation set by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


@dataclass
class PermutationImportanceResult:
    """Permutation importance results."""

    importance: pd.DataFrame
    baseline_rmse: float
    baseline_mae: float
    baseline_r2: float


def _to_numpy(value: Any) -> np.ndarray:
    """Convert tensors, lists, or arrays to NumPy arrays."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()

    return np.asarray(value)


def _get_predictions(prediction_result: Any) -> np.ndarray:
    """
    Extract posterior mean predictions from a prediction result.

    The function supports either:
    - an object with a `mean` attribute;
    - a dictionary containing `mean`;
    - a direct prediction array.
    """
    if isinstance(prediction_result, dict):
        for key in ("mean", "mean_prediction", "predictions"):
            if key in prediction_result:
                return _to_numpy(prediction_result[key]).reshape(-1)

    for attribute in ("mean", "mean_prediction", "predictions"):
        if hasattr(prediction_result, attribute):
            return _to_numpy(
                getattr(prediction_result, attribute)
            ).reshape(-1)

    return _to_numpy(prediction_result).reshape(-1)


def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate RMSE, MAE, and R2."""
    y_true = _to_numpy(y_true).reshape(-1)
    y_pred = _to_numpy(y_pred).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    errors = y_true - y_pred
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))

    denominator = np.sum((y_true - np.mean(y_true)) ** 2)

    if denominator == 0:
        r2 = float("nan")
    else:
        r2 = float(1.0 - np.sum(errors**2) / denominator)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def calculate_permutation_importance(
    model: Any,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    predict_function: Callable[..., Any],
    feature_names: Optional[list[str]] = None,
    n_repeats: int = 10,
    random_seed: int = 42,
    prediction_kwargs: Optional[dict[str, Any]] = None,
) -> PermutationImportanceResult:
    """
    Calculate permutation feature importance on the validation set.

    Parameters
    ----------
    model:
        Trained BNN model or training result.

    X_validation:
        Validation features with shape
        `(n_samples, n_features)`.

    y_validation:
        Validation targets.

    predict_function:
        Prediction function compatible with the trained model.
        For example, `predict_bnn` from `src.bnn_model`.

    feature_names:
        Names of the input features.

    n_repeats:
        Number of random permutations for each feature.

    random_seed:
        Seed for reproducible permutations.

    prediction_kwargs:
        Additional keyword arguments passed to predict_function.

    Returns
    -------
    PermutationImportanceResult
        Importance scores based on increases in RMSE and MAE.
    """
    X_validation = _to_numpy(X_validation).copy()
    y_validation = _to_numpy(y_validation).reshape(-1)

    if X_validation.ndim != 2:
        raise ValueError("X_validation must be a two-dimensional array.")

    n_samples, n_features = X_validation.shape

    if len(y_validation) != n_samples:
        raise ValueError(
            "X_validation and y_validation have incompatible lengths."
        )

    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")

    if feature_names is None:
        feature_names = [
            f"feature_{index}"
            for index in range(n_features)
        ]

    if len(feature_names) != n_features:
        raise ValueError(
            "feature_names must have one name for each feature."
        )

    prediction_kwargs = prediction_kwargs or {}

    baseline_result = predict_function(
        model,
        X_validation,
        y_validation,
        **prediction_kwargs,
    )

    baseline_prediction = _get_predictions(baseline_result)
    baseline_metrics = _calculate_metrics(
        y_validation,
        baseline_prediction,
    )

    rng = np.random.default_rng(random_seed)
    records = []

    for feature_index, feature_name in enumerate(feature_names):
        rmse_changes = []
        mae_changes = []
        r2_changes = []

        for _ in range(n_repeats):
            X_permuted = X_validation.copy()

            permutation = rng.permutation(n_samples)
            X_permuted[:, feature_index] = (
                X_permuted[permutation, feature_index]
            )

            permuted_result = predict_function(
                model,
                X_permuted,
                y_validation,
                **prediction_kwargs,
            )

            permuted_prediction = _get_predictions(permuted_result)
            permuted_metrics = _calculate_metrics(
                y_validation,
                permuted_prediction,
            )

            rmse_changes.append(
                permuted_metrics["rmse"] - baseline_metrics["rmse"]
            )
            mae_changes.append(
                permuted_metrics["mae"] - baseline_metrics["mae"]
            )
            r2_changes.append(
                baseline_metrics["r2"] - permuted_metrics["r2"]
            )

        records.append(
            {
                "feature": feature_name,
                "rmse_increase_mean": float(np.mean(rmse_changes)),
                "rmse_increase_std": float(np.std(rmse_changes)),
                "mae_increase_mean": float(np.mean(mae_changes)),
                "mae_increase_std": float(np.std(mae_changes)),
                "r2_decrease_mean": float(np.mean(r2_changes)),
                "r2_decrease_std": float(np.std(r2_changes)),
            }
        )

    importance = pd.DataFrame(records)

    importance = importance.sort_values(
        by="rmse_increase_mean",
        ascending=False,
    ).reset_index(drop=True)

    importance.insert(
        0,
        "rank",
        np.arange(1, len(importance) + 1),
    )

    return PermutationImportanceResult(
        importance=importance,
        baseline_rmse=baseline_metrics["rmse"],
        baseline_mae=baseline_metrics["mae"],
        baseline_r2=baseline_metrics["r2"],
    )
