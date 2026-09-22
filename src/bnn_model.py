"""
Bayesian Neural Network for probabilistic chlorophyll-a forecasting.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Sequence

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from torch.utils.data import DataLoader, TensorDataset

from .preprocess import inverse_transform_target


@dataclass
class TrainingResult:
    """Trained BNN, configuration, and saved variational parameter state."""

    guide: Any
    hidden_sizes: tuple[int, ...]
    sigma_range: tuple[float, float]
    dropout_rate: float
    learning_rate: float
    batch_size: int
    patience: int
    max_epochs: int
    random_seed: int
    best_validation_loss: float
    epochs_trained: int
    training_losses: list[float]
    validation_losses: list[float]
    param_store_state: dict[str, Any]


@dataclass
class PredictionResult:
    """Posterior predictive results in original Chl-a concentration units."""

    posterior_samples: np.ndarray
    mean: np.ndarray
    standard_deviation: np.ndarray
    lower_95: np.ndarray
    upper_95: np.ndarray
    actual: Optional[np.ndarray]
    metrics: Optional[dict[str, float]]


def set_random_seed(random_seed: int = 42) -> None:
    """Set random seeds for reproducible model training and prediction."""
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    pyro.set_rng_seed(random_seed)


def _validate_configuration(
    hidden_sizes: Sequence[int],
    sigma_range: tuple[float, float],
    dropout_rate: float,
) -> None:
    """Validate core BNN settings."""
    if not hidden_sizes or any(size < 1 for size in hidden_sizes):
        raise ValueError("hidden_sizes must contain positive integers.")

    if len(sigma_range) != 2 or sigma_range[0] <= 0:
        raise ValueError("sigma_range must contain two positive values.")

    if sigma_range[0] >= sigma_range[1]:
        raise ValueError(
            "The lower sigma bound must be smaller than the upper bound."
        )

    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in the interval [0, 1).")


def _as_feature_tensor(X: np.ndarray) -> torch.Tensor:
    """Convert feature data to a two-dimensional float tensor."""
    X_array = np.asarray(X, dtype=np.float32)

    if X_array.ndim != 2:
        raise ValueError("Feature data must have shape (n_samples, n_features).")

    return torch.tensor(X_array, dtype=torch.float32)


def _as_target_tensor(y: np.ndarray) -> torch.Tensor:
    """Convert standardized target data to a column-vector float tensor."""
    y_array = np.asarray(y, dtype=np.float32)

    if y_array.ndim == 1:
        y_array = y_array.reshape(-1, 1)

    if y_array.ndim != 2 or y_array.shape[1] != 1:
        raise ValueError("Target data must have shape (n_samples, 1).")

    return torch.tensor(y_array, dtype=torch.float32)


def bnn_model(
    X: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    hidden_sizes: Sequence[int] = (150, 75, 35),
    dropout_rate: float = 0.1,
    sigma_range: tuple[float, float] = (0.01, 0.10),
    weight_prior_scale: float = 0.08,
    bias_prior_scale: float = 0.05,
) -> torch.Tensor:
    """
    Define a fully connected Bayesian neural network.
    """
    _validate_configuration(hidden_sizes, sigma_range, dropout_rate)

    input_size = X.shape[1]
    hidden_sizes = tuple(hidden_sizes)

    hidden_layers = []
    previous_size = input_size

    for layer_index, hidden_size in enumerate(hidden_sizes, start=1):
        weights = pyro.sample(
            f"w{layer_index}",
            dist.Normal(
                torch.zeros(previous_size, hidden_size),
                torch.full(
                    (previous_size, hidden_size),
                    weight_prior_scale,
                ),
            ).to_event(2),
        )

        biases = pyro.sample(
            f"b{layer_index}",
            dist.Normal(
                torch.zeros(hidden_size),
                torch.full((hidden_size,), bias_prior_scale),
            ).to_event(1),
        )

        hidden_layers.append((weights, biases))
        previous_size = hidden_size

    output_weights = pyro.sample(
        "w_out",
        dist.Normal(
            torch.zeros(hidden_sizes[-1], 1),
            torch.full((hidden_sizes[-1], 1), bias_prior_scale),
        ).to_event(2),
    )

    output_bias = pyro.sample(
        "b_out",
        dist.Normal(
            torch.zeros(1),
            torch.full((1,), bias_prior_scale),
        ).to_event(1),
    )

    sigma = pyro.sample(
        "sigma",
        dist.Uniform(sigma_range[0], sigma_range[1]),
    )

    hidden = X

    for weights, biases in hidden_layers:
        hidden = torch.matmul(hidden, weights) + biases
        hidden = torch.relu(hidden)

        # This remains active during posterior prediction to retain the
        # dropout regularization used in the probabilistic model.
        hidden = torch.nn.functional.dropout(
            hidden,
            p=dropout_rate,
            training=True,
        )

    mean = torch.matmul(hidden, output_weights) + output_bias

    with pyro.plate("data", X.shape[0]):
        pyro.sample(
            "obs",
            dist.Normal(mean, sigma).to_event(1),
            obs=y,
        )

    return mean


def train_bnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    hidden_sizes: Sequence[int] = (150, 75, 35),
    sigma_range: tuple[float, float] = (0.01, 0.10),
    learning_rate: float = 0.003,
    batch_size: int = 128,
    patience: int = 150,
    dropout_rate: float = 0.1,
    max_epochs: int = 3000,
    random_seed: int = 42,
    verbose: bool = False,
) -> TrainingResult:
    """
    Train a BNN using training data and validation-loss early stopping.
    """
    _validate_configuration(hidden_sizes, sigma_range, dropout_rate)

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if patience < 1:
        raise ValueError("patience must be at least 1.")

    if max_epochs < 1:
        raise ValueError("max_epochs must be at least 1.")

    set_random_seed(random_seed)
    pyro.clear_param_store()

    X_train_tensor = _as_feature_tensor(X_train)
    y_train_tensor = _as_target_tensor(y_train)
    X_validation_tensor = _as_feature_tensor(X_validation)
    y_validation_tensor = _as_target_tensor(y_validation)

    if len(X_train_tensor) != len(y_train_tensor):
        raise ValueError("Training features and targets have different lengths.")

    if len(X_validation_tensor) != len(y_validation_tensor):
        raise ValueError(
            "Validation features and targets have different lengths."
        )

    model_function = partial(
        bnn_model,
        hidden_sizes=tuple(hidden_sizes),
        dropout_rate=dropout_rate,
        sigma_range=sigma_range,
    )

    guide = AutoDiagonalNormal(model_function)

    optimizer = pyro.optim.ClippedAdam(
        {
            "lr": learning_rate,
            "clip_norm": 10.0,
        }
    )

    svi = SVI(
        model_function,
        guide,
        optimizer,
        loss=Trace_ELBO(),
    )

    data_generator = torch.Generator()
    data_generator.manual_seed(random_seed)

    training_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=data_generator,
    )

    best_validation_loss = float("inf")
    best_param_store_state = None
    patience_counter = 0

    training_losses: list[float] = []
    validation_losses: list[float] = []

    for epoch in range(max_epochs):
        epoch_loss = 0.0

        for batch_X, batch_y in training_loader:
            epoch_loss += svi.step(batch_X, batch_y)

        epoch_loss /= len(training_loader)
        validation_loss = float(
            svi.evaluate_loss(X_validation_tensor, y_validation_tensor)
        )

        training_losses.append(float(epoch_loss))
        validation_losses.append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_param_store_state = copy.deepcopy(
                pyro.get_param_store().get_state()
            )
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and ((epoch + 1) % 100 == 0 or epoch == 0):
            print(
                f"Epoch {epoch + 1}: "
                f"training loss = {epoch_loss:.4f}, "
                f"validation loss = {validation_loss:.4f}"
            )

        if patience_counter >= patience:
            if verbose:
                print(
                    f"Early stopping at epoch {epoch + 1}; "
                    f"best validation loss = {best_validation_loss:.4f}"
                )
            break

    if best_param_store_state is None:
        raise RuntimeError("BNN training did not produce a valid parameter state.")

    pyro.get_param_store().set_state(
        copy.deepcopy(best_param_store_state)
    )

    return TrainingResult(
        guide=guide,
        hidden_sizes=tuple(hidden_sizes),
        sigma_range=sigma_range,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,
        max_epochs=max_epochs,
        random_seed=random_seed,
        best_validation_loss=float(best_validation_loss),
        epochs_trained=len(training_losses),
        training_losses=training_losses,
        validation_losses=validation_losses,
        param_store_state=best_param_store_state,
    )


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    lower_95: Optional[np.ndarray] = None,
    upper_95: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Calculate prediction metrics in original Chl-a concentration units."""
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)

    if len(actual) != len(predicted):
        raise ValueError("actual and predicted values must have the same length.")

    errors = actual - predicted
    mse = float(np.mean(errors**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))

    total_variation = np.sum((actual - np.mean(actual)) ** 2)
    r2 = (
        float(1.0 - np.sum(errors**2) / total_variation)
        if total_variation > 0
        else float("nan")
    )

    mape = float(
        np.mean(np.abs(errors / np.maximum(np.abs(actual), 1e-8))) * 100.0
    )

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
    }

    if lower_95 is not None and upper_95 is not None:
        lower_95 = np.asarray(lower_95, dtype=float).reshape(-1)
        upper_95 = np.asarray(upper_95, dtype=float).reshape(-1)

        coverage = np.mean(
            (actual >= lower_95) & (actual <= upper_95)
        )

        metrics["coverage_95"] = float(coverage)

    return metrics


def predict_bnn(
    training_result: TrainingResult,
    X_data: np.ndarray,
    y_scaler: Any,
    y_data: Optional[np.ndarray] = None,
    num_samples: int = 500,
    random_seed: Optional[int] = 42,
) -> PredictionResult:
    """
    Generate posterior predictive samples and uncertainty intervals.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")

    if random_seed is not None:
        set_random_seed(random_seed)

    X_tensor = _as_feature_tensor(X_data)

    if y_data is not None and len(X_tensor) != len(y_data):
        raise ValueError("X_data and y_data have different lengths.")

    # Restoring the saved state ensures a selected model remains valid even
    # after later trials have cleared or overwritten Pyro's global parameter store.
    pyro.clear_param_store()
    pyro.get_param_store().set_state(
        copy.deepcopy(training_result.param_store_state)
    )

    model_function = partial(
        bnn_model,
        hidden_sizes=training_result.hidden_sizes,
        dropout_rate=training_result.dropout_rate,
        sigma_range=training_result.sigma_range,
    )

    predictive = Predictive(
        model_function,
        guide=training_result.guide,
        num_samples=num_samples,
        return_sites=("obs",),
    )

    with torch.no_grad():
        samples_scaled = predictive(X_tensor)["obs"].detach().cpu().numpy()

    samples_original = inverse_transform_target(
        samples_scaled,
        y_scaler,
    )

    posterior_samples = np.asarray(samples_original, dtype=float).squeeze(-1)

    mean = np.mean(posterior_samples, axis=0)
    standard_deviation = np.std(posterior_samples, axis=0)
    lower_95 = np.percentile(posterior_samples, 2.5, axis=0)
    upper_95 = np.percentile(posterior_samples, 97.5, axis=0)

    actual = None
    metrics = None

    if y_data is not None:
        actual = inverse_transform_target(
            np.asarray(y_data, dtype=float),
            y_scaler,
        ).reshape(-1)

        metrics = calculate_metrics(
            actual=actual,
            predicted=mean,
            lower_95=lower_95,
            upper_95=upper_95,
        )

    return PredictionResult(
        posterior_samples=posterior_samples,
        mean=mean,
        standard_deviation=standard_deviation,
        lower_95=lower_95,
        upper_95=upper_95,
        actual=actual,
        metrics=metrics,
    )


def exceedance_probability(
    prediction_result: PredictionResult,
    threshold: float,
) -> np.ndarray:
    """
    Calculate the posterior probability that Chl-a exceeds a threshold.
    """
    if not np.isfinite(threshold):
        raise ValueError("threshold must be a finite number.")

    return np.mean(
        prediction_result.posterior_samples > threshold,
        axis=0,
    )
