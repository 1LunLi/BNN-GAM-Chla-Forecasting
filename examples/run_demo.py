from pathlib import Path
import pandas as pd
from src.bnn_model import exceedance_probability, predict_bnn
from src.model_selection import select_bnn_model
from src.preprocess import prepare_bnn_data


DATA_PATH = Path("data/sample_daily_data.xlsx")
OUTPUT_DIR = Path("outputs")

AT = 1
LT = 7
BLOOM_THRESHOLD = 20.0
NUM_SAMPLES = 500

FEATURE_COLUMNS = [
    "sin_month",
    "cos_month",
    "Chla",
    "CODMn",
    "NH4_N",
    "TP",
    "TN",
    "T",
    "PAR",
    "PAR_ratio",
    "Q",
    "H",
    "CVQ7",
    "V",
    "h",
    "FR",
    "HR7",
    "tau_b",
]

# A compact configuration for testing the workflow.
# Replace this dictionary with default_search_space() for a full search.
DEMO_SEARCH_SPACE = {
    "hidden_sizes": [(150, 75, 35)],
    "sigma_range": [(0.01, 0.10)],
    "learning_rate": [0.003],
    "dropout_rate": [0.10],
    "patience": [150],
    "batch_size": [64],
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bundle = prepare_bnn_data(
        file_path=DATA_PATH,
        feature_columns=FEATURE_COLUMNS,
        target_column="Chla",
        date_column="date",
        ahead_time=AT,
        lagged_time=LT,
    )

    selection = select_bnn_model(
        X_train=bundle.X_train_scaled,
        y_train=bundle.y_train_scaled,
        X_validation=bundle.X_val_scaled,
        y_validation=bundle.y_val_scaled,
        y_scaler=bundle.y_scaler,
        search_space=DEMO_SEARCH_SPACE,
        num_samples=NUM_SAMPLES,
    )

    test_prediction = predict_bnn(
        training_result=selection.best_training_result,
        X_data=bundle.X_test_scaled,
        y_scaler=bundle.y_scaler,
        y_data=bundle.y_test_scaled,
        num_samples=NUM_SAMPLES,
    )

    exceedance = exceedance_probability(
        prediction_result=test_prediction,
        threshold=BLOOM_THRESHOLD,
    )

    predictions = pd.DataFrame(
        {
            "forecast_origin": bundle.test_origins,
            "target_date": bundle.test_dates,
            "observed_chla": test_prediction.actual,
            "predicted_chla": test_prediction.mean,
            "prediction_std": test_prediction.standard_deviation,
            "lower_95": test_prediction.lower_95,
            "upper_95": test_prediction.upper_95,
            "probability_chla_above_20": exceedance,
        }
    )

    selection.search_results.to_csv(
        OUTPUT_DIR / f"search_results_AT{AT}_LT{LT}.csv",
        index=False,
    )

    predictions.to_csv(
        OUTPUT_DIR / f"test_predictions_AT{AT}_LT{LT}.csv",
        index=False,
    )

    print(f"Best configuration: {selection.best_configuration}")
    print(f"Validation composite score: {selection.best_score:.4f}")
    print("Independent test metrics:")

    for name, value in test_prediction.metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
