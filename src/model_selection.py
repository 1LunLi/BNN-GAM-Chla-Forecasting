"""Validation-based BNN hyperparameter selection."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Optional

import numpy as np
import pandas as pd

from .bnn_model import (
    PredictionResult,
    TrainingResult,
    predict_bnn,
    train_bnn,
)


@dataclass
class SelectionResult:
    best_configuration: dict[str, Any]
    best_score: float
    best_training_result: TrainingResult
    best_validation_prediction: PredictionResult
    search_results: pd.DataFrame


def default_search_space() -> dict[str, list[Any]]:
    return {
        "hidden_sizes": [
            (100, 50, 25),
            (150, 75, 35),
            (200, 100, 50),
        ],
        "sigma_range": [
            (0.001, 0.05),
            (0.01, 0.10),
            (0.05, 0.20),
        ],
        "learning_rate": [0.001, 0.003, 0.01],
        "dropout_rate": [0.10, 0.20, 0.30],
        "patience": [150, 200, 250],
        "batch_size": [64, 128],
    }


def _expand_search_space(
    search_space: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    keys = list(search_space)

    return [
        dict(zip(keys, values))
        for values in product(*(search_space[key] for key in keys))
    ]


def _inverse_minmax_score(values: pd.Series) -> pd.Series:
    value_range = values.max() - values.min()

    if np.isclose(value_range, 0.0):
        return pd.Series(1.0, index=values.index)

    return (values.max() - values) / value_range


def _score_results(
    results: pd.DataFrame,
    target_coverage: float = 0.95,
) -> pd.DataFrame:
    results = results.copy()

    results["rmse_score"] = _inverse_minmax_score(
        results["validation_rmse"]
    )
    results["mae_score"] = _inverse_minmax_score(
        results["validation_mae"]
    )

    results["accuracy_score"] = (
        results["validation_r2"]
        + results["rmse_score"]
        + results["mae_score"]
    ) / 3.0

    results["uncertainty_score"] = (
        1.0 - (results["validation_coverage_95"] - target_coverage).abs()
    )

    results["generalization_score"] = (
        1.0 - (results["training_r2"] - results["validation_r2"]).abs()
    )

    results["composite_score"] = (
        0.50 * results["accuracy_score"]
        + 0.30 * results["uncertainty_score"]
        + 0.20 * results["generalization_score"]
    )

    return results.sort_values(
        "composite_score",
        ascending=False,
    ).reset_index(drop=True)


def select_bnn_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    y_scaler: Any,
    search_space: Optional[dict[str, list[Any]]] = None,
    num_samples: int = 500,
    max_epochs: int = 3000,
    random_seed: int = 42,
    verbose: bool = True,
) -> SelectionResult:
    """
    Select BNN hyperparameters using training and validation data only.

    The test set must not be passed to this function.
    """
    if search_space is None:
        search_space = default_search_space()

    configurations = _expand_search_space(search_space)

    if not configurations:
        raise ValueError("The search space is empty.")

    records = []
    trial_configurations = {}

    for trial_id, configuration in enumerate(configurations, start=1):
        trial_seed = random_seed + trial_id - 1
        trial_configurations[trial_id] = configuration.copy()

        if verbose:
            print(
                f"Trial {trial_id}/{len(configurations)}: "
                f"{configuration}"
            )

        try:
            trained_model = train_bnn(
                X_train=X_train,
                y_train=y_train,
                X_validation=X_validation,
                y_validation=y_validation,
                max_epochs=max_epochs,
                random_seed=trial_seed,
                verbose=False,
                **configuration,
            )

            train_prediction = predict_bnn(
                training_result=trained_model,
                X_data=X_train,
                y_scaler=y_scaler,
                y_data=y_train,
                num_samples=num_samples,
                random_seed=trial_seed,
            )

            validation_prediction = predict_bnn(
                training_result=trained_model,
                X_data=X_validation,
                y_scaler=y_scaler,
                y_data=y_validation,
                num_samples=num_samples,
                random_seed=trial_seed + 1,
            )

            train_metrics = train_prediction.metrics
            validation_metrics = validation_prediction.metrics

            records.append(
                {
                    "trial_id": trial_id,
                    "trial_seed": trial_seed,
                    **configuration,
                    "training_r2": train_metrics["r2"],
                    "validation_r2": validation_metrics["r2"],
                    "validation_rmse": validation_metrics["rmse"],
                    "validation_mae": validation_metrics["mae"],
                    "validation_coverage_95": validation_metrics[
                        "coverage_95"
                    ],
                    "best_validation_loss": trained_model.best_validation_loss,
                    "epochs_trained": trained_model.epochs_trained,
                    "status": "completed",
                    "error": None,
                }
            )

        except Exception as error:
            records.append(
                {
                    "trial_id": trial_id,
                    "trial_seed": trial_seed,
                    **configuration,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )

            if verbose:
                print(f"Trial {trial_id} failed: {error}")

    all_results = pd.DataFrame(records)
    successful_results = all_results.loc[
        all_results["status"] == "completed"
    ].copy()

    if successful_results.empty:
        raise RuntimeError("All hyperparameter trials failed.")

    search_results = _score_results(successful_results)
    best_row = search_results.iloc[0]

    best_trial_id = int(best_row["trial_id"])
    best_seed = int(best_row["trial_seed"])
    best_configuration = trial_configurations[best_trial_id]

    best_training_result = train_bnn(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        max_epochs=max_epochs,
        random_seed=best_seed,
        verbose=False,
        **best_configuration,
    )

    best_validation_prediction = predict_bnn(
        training_result=best_training_result,
        X_data=X_validation,
        y_scaler=y_scaler,
        y_data=y_validation,
        num_samples=num_samples,
        random_seed=best_seed + 1,
    )

    return SelectionResult(
        best_configuration=best_configuration,
        best_score=float(best_row["composite_score"]),
        best_training_result=best_training_result,
        best_validation_prediction=best_validation_prediction,
        search_results=search_results,
    )