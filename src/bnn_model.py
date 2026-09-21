"""
Bayesian Neural Network model utilities for Chl-a forecasting.

This module defines the Bayesian Neural Network (BNN), training routine,
posterior prediction, and basic evaluation metrics used in the BNN-GAM
framework.

"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainingResult:
    """Container for a trained BNN model and training history."""

    guide: AutoDiagonalNormal
    train_losses: list[float]
    val_losses: list[float]
    best_val_loss: float
    config: Dict


@dataclass
class PredictionResult:
    """Container for posterior predictive results on the original target scale."""

    y_true: np.ndarray
    y_pred_mean: np.ndarray
    y_pred_std: np.ndarray
    lower_95: np.ndarray
    upper_95: np.ndarray
    posterior_samples: np.ndarray
    metrics: Dict[str, float]


def set_random_seed(seed: int = 42) -> None:
    """Set NumPy, PyTorch, and Pyro random seeds."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)


def bnn_model(
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    hidden_sizes: Sequence[int] = (100, 50, 25),
    dropout_rate: float = 0.1,
    sigma_range: Tuple[float, float] = (0.01, 0.1),
    training: bool = True,
) -> torch.Tensor:
    """
    Bayesian Neural Network for scaled log-transformed Chl-a prediction.

    Parameters
    ----------
    x:
        Predictor matrix with shape (n_samples, n_features).
    y:
        Optional target vector with shape (n_samples, 1), already transformed
        and standardized.
    hidden_sizes:
        Number of neurons in each hidden layer.
    dropout_rate:
        Dropout probability used during training. Dropout is disabled during
        posterior prediction by passing training=False.
    sigma_range:
        Lower and upper bounds for the observation noise prior.
    training:
        Whether to apply dropout. Use True during SVI training and False
        during posterior prediction.

    Returns
    -------
    torch.Tensor
        Predictive mean on the standardized log-target scale.
    """

    input_size = x.shape[1]

    layers = []
    for i, hidden_size in enumerate(hidden_sizes):
        previous_size = input_size if i == 0 else hidden_sizes[i - 1]

        weight = pyro.sample(
            f"w{i + 1}",
            dist.Normal(
                torch.zeros(previous_size, hidden_size),
                torch.ones(previous_size, hidden_size) * 0.08,
            ).to_event(2),
        )
        bias = pyro.sample(
            f"b{i + 1}",
            dist.Normal(
                torch.zeros(hidden_size),
                torch.ones(hidden_size) * 0.05,
            ).to_event(1),
        )
        layers.append((weight, bias))

    w_out = pyro.sample(
        "w_out",
        dist.Normal(
            torch.zeros(hidden_sizes[-1], 1),
            torch.ones(hidden_sizes[-1], 1) * 0.05,
        ).to_event(2),
    )
    b_out = pyro.sample(
        "b_out",
        dist.Normal(torch.zeros(1), torch.ones(1) * 0.05).to_event(1),
    )

    sigma = pyro.sample("sigma", dist.Uniform(sigma_range[0], sigma_range[1]))

    hidden = x
    for i, (weight, bias) in enumerate(layers):
        hidden = torch.matmul(hidden, weight) + bias
        if i < len(layers) - 1:
            hidden = torch.relu(hidden)
            hidden = torch.nn.functional.dropout(
                hidden,
                p=dropout_rate,
                training=training,
            )

    mean = torch.matmul(hidden, w_out) + b_out

    with pyro.plate("data", x.shape[0]):
        pyro.sample("obs", dist.Normal(mean, sigma).to_event(1), obs=y)

    return mean


def train_bnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_sizes: Sequence[int] = (100, 50, 25),
    sigma_range: Tuple[float, float] = (0.01, 0.1),
    learning_rate: float = 0.003,
    batch_size: int = 128,
    patience: int = 150,
    dropout_rate: float = 0.1,
    max_epochs: int = 3000,
    seed: int = 42,
    verbose: bool = True,
) -> TrainingResult:
    """
    Train a Bayesian Neural Network using stochastic variational inference.

    Validation loss is used for early stopping. The test set must not be used
    in this function, because the test set should remain an independent final
    evaluation dataset.
    """

    set_random_seed(seed)
    pyro.clear_param_store()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )

    train_model = partial(
        bnn_model,
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate,
        sigma_range=sigma_range,
        training=True,
    )

    guide = AutoDiagonalNormal(train_model)

    optimizer = pyro.optim.ClippedAdam(
        {"lr": learning_rate, "clip_norm": 10.0}
    )
    svi = SVI(train_model, guide, optimizer, loss=Trace_ELBO())

    # Initialize guide parameters.
    guide(X_train_t[: min(10, len(X_train_t))], y_train_t[: min(10, len(y_train_t))])

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(max_epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            epoch_loss += svi.step(xb, yb)

        epoch_loss /= max(1, len(loader))

        with torch.no_grad():
            val_loss = svi.evaluate_loss(X_val_t, y_val_t)

        train_losses.append(float(epoch_loss))
        val_losses.append(float(val_loss))

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_state = pyro.get_param_store().get_state()
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch == 0 or (epoch + 1) % 200 == 0):
            print(
                f"Epoch {epoch + 1:04d} | "
                f"train loss = {epoch_loss:.4f} | "
                f"val loss = {val_loss:.4f}"
            )

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}.")
            break

    if best_state is not None:
        pyro.get_param_store().set_state(best_state)

    config = {
        "hidden_sizes": tuple(hidden_sizes),
        "sigma_range": tuple(sigma_range),
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "patience": patience,
        "dropout_rate": dropout_rate,
        "max_epochs": max_epochs,
        "seed": seed,
    }

    return TrainingResult(
        guide=guide,
        train_losses=train_losses,
        val_losses=val_losses,
        best_val_loss=best_val_loss,
        config=config,
    )


def inverse_transform_target(
    y_scaled: np.ndarray,
    y_scaler: StandardScaler,
    log1p_target: bool = True,
) -> np.ndarray:
    """Convert standardized model outputs back to the original Chl-a scale."""

    y_scaled = np.asarray(y_scaled).reshape(-1, 1)
    y_unscaled = y_scaler.inverse_transform(y_scaled)

    if log1p_target:
        return np.expm1(y_unscaled)

    return y_unscaled


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower_95: Optional[np.ndarray] = None,
    upper_95: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculate common regression and uncertainty-coverage metrics."""

    y_true = np.asarray(y_true).reshape(-1, 1)
    y_pred = np.asarray(y_pred).reshape(-1, 1)

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100.0)

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE_percent": mape,
    }

    if lower_95 is not None and upper_95 is not None:
        lower_95 = np.asarray(lower_95).reshape(-1, 1)
        upper_95 = np.asarray(upper_95).reshape(-1, 1)
        coverage = np.mean((y_true >= lower_95) & (y_true <= upper_95))
        metrics["Coverage_95"] = float(coverage)

    return metrics


def predict_bnn(
    guide: AutoDiagonalNormal,
    X_data: np.ndarray,
    y_data: np.ndarray,
    y_scaler: StandardScaler,
    hidden_sizes: Sequence[int] = (100, 50, 25),
    sigma_range: Tuple[float, float] = (0.01, 0.1),
    dropout_rate: float = 0.1,
    num_samples: int = 1000,
    log1p_target: bool = True,
) -> PredictionResult:
    """
    Generate posterior predictive samples and evaluate predictions.

    Dropout is disabled during prediction. Predictive uncertainty is obtained
    from posterior samples and observation noise represented in the Pyro model.
    """

    X_t = torch.tensor(X_data, dtype=torch.float32)

    prediction_model = partial(
        bnn_model,
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate,
        sigma_range=sigma_range,
        training=False,
    )

    predictive = Predictive(
        prediction_model,
        guide=guide,
        num_samples=num_samples,
        return_sites=["obs"],
    )

    with torch.no_grad():
        samples_scaled = predictive(X_t)["obs"].detach().cpu().numpy()

    samples_scaled = np.squeeze(samples_scaled)
    if samples_scaled.ndim == 1:
        samples_scaled = samples_scaled[:, None]

    # Shape after transpose: n_samples_observations x n_posterior_samples
    samples_scaled = samples_scaled.reshape(num_samples, X_data.shape[0]).T

    posterior_samples_original = np.column_stack(
        [
            inverse_transform_target(
                samples_scaled[:, i],
                y_scaler=y_scaler,
                log1p_target=log1p_target,
            ).flatten()
            for i in range(samples_scaled.shape[1])
        ]
    )

    y_true = inverse_transform_target(
        y_data,
        y_scaler=y_scaler,
        log1p_target=log1p_target,
    )

    y_pred_mean = np.mean(posterior_samples_original, axis=1, keepdims=True)
    y_pred_std = np.std(posterior_samples_original, axis=1, keepdims=True)

    lower_95 = np.percentile(posterior_samples_original, 2.5, axis=1).reshape(-1, 1)
    upper_95 = np.percentile(posterior_samples_original, 97.5, axis=1).reshape(-1, 1)

    metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred_mean,
        lower_95=lower_95,
        upper_95=upper_95,
    )

    return PredictionResult(
        y_true=y_true,
        y_pred_mean=y_pred_mean,
        y_pred_std=y_pred_std,
        lower_95=lower_95,
        upper_95=upper_95,
        posterior_samples=posterior_samples_original,
        metrics=metrics,
    )


def exceedance_probability(
    posterior_samples: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Calculate threshold exceedance probability from posterior samples.

    Parameters
    ----------
    posterior_samples:
        Array with shape (n_observations, n_posterior_samples) on the original
        Chl-a scale.
    threshold:
        Alert threshold on the original Chl-a scale.

    Returns
    -------
    np.ndarray
        Exceedance probability for each observation.
    """

    posterior_samples = np.asarray(posterior_samples)
    return np.mean(posterior_samples > threshold, axis=1)
