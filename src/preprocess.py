"""
Data preprocessing for BNN chlorophyll-a forecasting.
AT: ahead time in days.
LT: lagged time window in days.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetBundle:
    """Processed datasets and metadata for one AT-LT forecasting scenario."""

    X_train_raw: np.ndarray
    X_val_raw: np.ndarray
    X_test_raw: np.ndarray

    X_train_scaled: np.ndarray
    X_val_scaled: np.ndarray
    X_test_scaled: np.ndarray

    y_train_original: np.ndarray
    y_val_original: np.ndarray
    y_test_original: np.ndarray

    y_train_log: np.ndarray
    y_val_log: np.ndarray
    y_test_log: np.ndarray

    y_train_scaled: np.ndarray
    y_val_scaled: np.ndarray
    y_test_scaled: np.ndarray

    train_dates: pd.Series
    val_dates: pd.Series
    test_dates: pd.Series

    train_origins: pd.Series
    val_origins: pd.Series
    test_origins: pd.Series

    feature_names: list[str]
    target_name: str
    ahead_time: int
    lagged_time: int

    X_scaler: StandardScaler
    y_scaler: StandardScaler


def load_daily_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load a daily dataset from a CSV or Excel file.

    The input file should contain one row per observation date, a date column,
    the target Chl-a column, and all predictor columns.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path)

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)

    raise ValueError(
        "Unsupported file type. Please provide a CSV, XLSX, or XLS file."
    )


def create_lagged_dataset(
    daily_data: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    date_column: str,
    ahead_time: int,
    lagged_time: int,
    require_daily_continuity: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series, pd.Series, list[str]]:
    """
    Create supervised forecasting samples from daily observations.

    For each forecast origin t, input features span t-LT+1 to t, and the
    target is Chl-a at t+AT. For example, AT=1 and LT=7 uses observations
    from t-6 to t to forecast Chl-a at t+1.
    """
    if ahead_time < 1:
        raise ValueError("ahead_time must be at least 1.")

    if lagged_time < 1:
        raise ValueError("lagged_time must be at least 1.")

    required_columns = list(
    dict.fromkeys([date_column, target_column, *feature_columns])
)
    missing_columns = [
        column
        for column in required_columns
        if column not in daily_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )

    data = daily_data.loc[:, required_columns].copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data = data.dropna().sort_values(date_column).reset_index(drop=True)

    if data.empty:
        raise ValueError("No complete observations remain after removing NA values.")

    if data[target_column].lt(0).any():
        raise ValueError(
            "The target contains negative values and cannot use log1p."
        )

    feature_names = []
    for lag in range(lagged_time - 1, -1, -1):
        for feature in feature_columns:
            feature_names.append(f"{feature}_t-{lag}" if lag else f"{feature}_t")

    X_rows = []
    y_rows = []
    target_dates = []
    forecast_origins = []

    dates = data[date_column].reset_index(drop=True)
    feature_values = data.loc[:, feature_columns].to_numpy(dtype=float)
    target_values = data[target_column].to_numpy(dtype=float)

    final_origin_index = len(data) - ahead_time

    for origin_index in range(lagged_time - 1, final_origin_index):
        start_index = origin_index - lagged_time + 1
        target_index = origin_index + ahead_time

        if require_daily_continuity:
            date_window = dates.iloc[start_index:target_index + 1]
            day_differences = date_window.diff().dropna().dt.days

            if not day_differences.eq(1).all():
                continue

        lagged_features = feature_values[
            start_index:origin_index + 1
        ].reshape(-1)

        X_rows.append(lagged_features)
        y_rows.append(target_values[target_index])
        target_dates.append(dates.iloc[target_index])
        forecast_origins.append(dates.iloc[origin_index])

    if not X_rows:
        raise ValueError(
            "No forecasting samples were created. Check AT, LT, missing data, "
            "or daily date continuity."
        )

    X = pd.DataFrame(X_rows, columns=feature_names)
    y = np.asarray(y_rows, dtype=float).reshape(-1, 1)

    return (
        X,
        y,
        pd.Series(target_dates, name="target_date"),
        pd.Series(forecast_origins, name="forecast_origin"),
        feature_names,
    )


def chronological_split(
    X: pd.DataFrame,
    y: np.ndarray,
    target_dates: pd.Series,
    forecast_origins: pd.Series,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, np.ndarray | pd.Series]:
    """
    Split samples into training, validation, and test sets.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    if train_fraction + validation_fraction >= 1:
        raise ValueError(
            "train_fraction + validation_fraction must be less than 1."
        )

    n_samples = len(X)

    if n_samples < 10:
        raise ValueError(
            "At least 10 supervised samples are required for chronological splitting."
        )

    train_end = int(n_samples * train_fraction)
    validation_end = train_end + int(n_samples * validation_fraction)

    if train_end == 0 or validation_end <= train_end or validation_end >= n_samples:
        raise ValueError("The chosen split fractions create an empty subset.")

    return {
        "X_train": X.iloc[:train_end].to_numpy(dtype=float),
        "X_val": X.iloc[train_end:validation_end].to_numpy(dtype=float),
        "X_test": X.iloc[validation_end:].to_numpy(dtype=float),
        "y_train": y[:train_end],
        "y_val": y[train_end:validation_end],
        "y_test": y[validation_end:],
        "train_dates": target_dates.iloc[:train_end].reset_index(drop=True),
        "val_dates": target_dates.iloc[train_end:validation_end].reset_index(
            drop=True
        ),
        "test_dates": target_dates.iloc[validation_end:].reset_index(drop=True),
        "train_origins": forecast_origins.iloc[:train_end].reset_index(drop=True),
        "val_origins": forecast_origins.iloc[
            train_end:validation_end
        ].reset_index(drop=True),
        "test_origins": forecast_origins.iloc[validation_end:].reset_index(
            drop=True
        ),
    }


def inverse_transform_target(
    y_scaled: np.ndarray,
    y_scaler: StandardScaler,
) -> np.ndarray:
    """
    Convert standardized log1p Chl-a predictions back to original units, for example μg/L.
    """
    values = np.asarray(y_scaled, dtype=float)
    original_shape = values.shape

    values_log = y_scaler.inverse_transform(values.reshape(-1, 1))
    values_original = np.expm1(values_log)

    return values_original.reshape(original_shape)


def transform_features(
    X_raw: np.ndarray,
    X_scaler: StandardScaler,
) -> np.ndarray:
    """Apply a training-fitted feature scaler to new feature data."""
    return X_scaler.transform(np.asarray(X_raw, dtype=float))


def prepare_bnn_data(
    file_path: str | Path,
    feature_columns: Sequence[str],
    target_column: str = "Chla",
    date_column: str = "date",
    ahead_time: int = 1,
    lagged_time: int = 7,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    require_daily_continuity: bool = True,
) -> DatasetBundle:
    """
    Prepare leakage-safe BNN data for one forecasting scenario.
    Standardization follows this sequence:
    1. Create lagged samples from ordered daily data.
    2. Split samples into training, validation, and test periods.
    3. Fit scalers on the training subset.
    4. Transform validation and test subsets using training-fitted scalers.
    """
    daily_data = load_daily_data(file_path)

    (
        X,
        y_original,
        target_dates,
        forecast_origins,
        feature_names,
    ) = create_lagged_dataset(
        daily_data=daily_data,
        feature_columns=feature_columns,
        target_column=target_column,
        date_column=date_column,
        ahead_time=ahead_time,
        lagged_time=lagged_time,
        require_daily_continuity=require_daily_continuity,
    )

    split_data = chronological_split(
        X=X,
        y=y_original,
        target_dates=target_dates,
        forecast_origins=forecast_origins,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )

    y_train_log = np.log1p(split_data["y_train"])
    y_val_log = np.log1p(split_data["y_val"])
    y_test_log = np.log1p(split_data["y_test"])

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = X_scaler.fit_transform(split_data["X_train"])
    X_val_scaled = X_scaler.transform(split_data["X_val"])
    X_test_scaled = X_scaler.transform(split_data["X_test"])

    y_train_scaled = y_scaler.fit_transform(y_train_log)
    y_val_scaled = y_scaler.transform(y_val_log)
    y_test_scaled = y_scaler.transform(y_test_log)

    return DatasetBundle(
        X_train_raw=split_data["X_train"],
        X_val_raw=split_data["X_val"],
        X_test_raw=split_data["X_test"],
        X_train_scaled=X_train_scaled,
        X_val_scaled=X_val_scaled,
        X_test_scaled=X_test_scaled,
        y_train_original=split_data["y_train"],
        y_val_original=split_data["y_val"],
        y_test_original=split_data["y_test"],
        y_train_log=y_train_log,
        y_val_log=y_val_log,
        y_test_log=y_test_log,
        y_train_scaled=y_train_scaled,
        y_val_scaled=y_val_scaled,
        y_test_scaled=y_test_scaled,
        train_dates=split_data["train_dates"],
        val_dates=split_data["val_dates"],
        test_dates=split_data["test_dates"],
        train_origins=split_data["train_origins"],
        val_origins=split_data["val_origins"],
        test_origins=split_data["test_origins"],
        feature_names=feature_names,
        target_name=target_column,
        ahead_time=ahead_time,
        lagged_time=lagged_time,
        X_scaler=X_scaler,
        y_scaler=y_scaler,
    )
