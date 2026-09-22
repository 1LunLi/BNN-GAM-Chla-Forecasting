"""
Data loading and preprocessing utilities for the BNN-GAM framework.

This module implements the recommended preprocessing workflow for time-series Chl-a forecasting:
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame]

@dataclass
class DatasetBundle:
    """Container for leakage-safe model inputs and preprocessing objects."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    X_train_raw: np.ndarray
    X_val_raw: np.ndarray
    X_test_raw: np.ndarray
    y_train_raw: np.ndarray
    y_val_raw: np.ndarray
    y_test_raw: np.ndarray
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    feature_names: list[str]
    target_name: str
    split_indices: Dict[str, np.ndarray]
    log1p_target: bool = True


def read_table(data_path: Union[str, Path]) -> pd.DataFrame:
    """Read a CSV or Excel table."""

    data_path = Path(data_path)
    suffix = data_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(data_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(data_path)

    raise ValueError(
        f"Unsupported file format: {suffix}. Use .csv, .xlsx, or .xls."
    )


def chronological_split_indices(
    n_samples: int,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> Dict[str, np.ndarray]:
    """
    Generate train/validation/test split indices.

    """

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be smaller than 1.")

    train_end = int(np.floor(n_samples * train_fraction))
    val_end = int(np.floor(n_samples * (train_fraction + val_fraction)))

    if train_end <= 0 or val_end <= train_end or val_end >= n_samples:
        raise ValueError(
            "Invalid split sizes. Check the number of samples and split fractions."
        )

    return {
        "train": np.arange(0, train_end),
        "val": np.arange(train_end, val_end),
        "test": np.arange(val_end, n_samples),
    }


def _resolve_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    date_column: Optional[str],
    feature_columns: Optional[Sequence[str]],
    exclude_columns: Optional[Iterable[str]],
) -> list[str]:
    """Determine which columns should be used as model predictors."""

    if feature_columns is not None:
        missing = [col for col in feature_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found in data: {missing}")
        return list(feature_columns)

    excluded = {target_column}
    if date_column is not None:
        excluded.add(date_column)
    if exclude_columns is not None:
        excluded.update(exclude_columns)

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [col for col in numeric_columns if col not in excluded]

    if not feature_columns:
        raise ValueError("No numeric feature columns were found.")

    return feature_columns


def load_lagged_forecasting_data(
    data_path: Union[str, Path],
    target_column: Optional[str] = None,
    date_column: Optional[str] = None,
    feature_columns: Optional[Sequence[str]] = None,
    exclude_columns: Optional[Iterable[str]] = None,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    log1p_target: bool = True,
    dropna: bool = True,
) -> DatasetBundle:
    """
    Load a prepared lagged-feature table and perform leakage-safe preprocessing.

    Parameters
    ----------
    data_path:
        Path to a CSV or Excel file. The file should already contain lagged
        predictor columns and the forecast target column.
    target_column:
        Name of the target column.
    date_column:
        Optional datetime column used for chronological sorting.
    feature_columns:
        Optional explicit list of feature columns. If None, all numeric columns
        except the target/date/excluded columns are used.
    exclude_columns:
        Optional columns to exclude from predictors, such as ID or sequence columns.
    train_fraction:
        Fraction of samples used for training.
    val_fraction:
        Fraction of samples used for validation.
    log1p_target:
        Whether to apply log1p transformation to the target before scaling.
    dropna:
        Whether to drop rows with missing values.

    Returns
    -------
    DatasetBundle
        Scaled and raw train/validation/test arrays, fitted scalers, feature
        names, target name, and split indices.
    """

    df = read_table(data_path)

    if dropna:
        df = df.dropna().reset_index(drop=True)

    if target_column is None:
        target_column = df.columns[-1]

    if target_column not in df.columns:
        raise ValueError(f"Target column not found in data: {target_column}")

    if date_column is not None:
        if date_column not in df.columns:
            raise ValueError(f"Date column not found in data: {date_column}")
        df = df.sort_values(date_column).reset_index(drop=True)

    selected_features = _resolve_feature_columns(
        df=df,
        target_column=target_column,
        date_column=date_column,
        feature_columns=feature_columns,
        exclude_columns=exclude_columns,
    )

    X_raw = df[selected_features].to_numpy(dtype=np.float32)
    y_raw = df[[target_column]].to_numpy(dtype=np.float32)

    if log1p_target:
        if np.any(y_raw < 0):
            raise ValueError("log1p_target=True requires non-negative target values.")
        y_for_scaling = np.log1p(y_raw)
    else:
        y_for_scaling = y_raw.copy()

    indices = chronological_split_indices(
        n_samples=len(df),
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )

    X_train_raw = X_raw[indices["train"]]
    X_val_raw = X_raw[indices["val"]]
    X_test_raw = X_raw[indices["test"]]

    y_train_raw = y_for_scaling[indices["train"]]
    y_val_raw = y_for_scaling[indices["val"]]
    y_test_raw = y_for_scaling[indices["test"]]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = x_scaler.fit_transform(X_train_raw)
    y_train = y_scaler.fit_transform(y_train_raw)

    X_val = x_scaler.transform(X_val_raw)
    X_test = x_scaler.transform(X_test_raw)
    y_val = y_scaler.transform(y_val_raw)
    y_test = y_scaler.transform(y_test_raw)

    return DatasetBundle(
        X_train=X_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_train=y_train.astype(np.float32),
        y_val=y_val.astype(np.float32),
        y_test=y_test.astype(np.float32),
        X_train_raw=X_train_raw.astype(np.float32),
        X_val_raw=X_val_raw.astype(np.float32),
        X_test_raw=X_test_raw.astype(np.float32),
        y_train_raw=y_train_raw.astype(np.float32),
        y_val_raw=y_val_raw.astype(np.float32),
        y_test_raw=y_test_raw.astype(np.float32),
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_names=selected_features,
        target_name=target_column,
        split_indices=indices,
        log1p_target=log1p_target,
    )


def inverse_transform_target(
    y_scaled: ArrayLike,
    y_scaler: StandardScaler,
    log1p_target: bool = True,
) -> np.ndarray:
    """
    Convert scaled target values back to the original chlorophyll-a scale.
    """

    y_scaled = np.asarray(y_scaled).reshape(-1, 1)
    y_unscaled = y_scaler.inverse_transform(y_scaled)

    if log1p_target:
        return np.expm1(y_unscaled)

    return y_unscaled


def transform_new_features(
    X_raw: ArrayLike,
    x_scaler: StandardScaler,
) -> np.ndarray:
    """
    Transform new predictor data using a scaler fitted on the training set.
    """

    X_raw = np.asarray(X_raw, dtype=np.float32)
    return x_scaler.transform(X_raw).astype(np.float32)
