# BNN-GAM-Chla-Forecasting

Reference implementation of the Bayesian Neural Network (BNN) forecasting component described in the manuscript, *A probabilistic and interpretable Bayesian Neural Network framework for chlorophyll-a forecasting in highly regulated rivers*.

## Scope

This repository provides a leakage-safe workflow for probabilistic chlorophyll-a forecasting in regulated rivers. It includes:

- Chronological data splitting and training-set-only standardization
- Automatic lagged-feature construction using ahead time (AT) and lagged time (LT)
- Bayesian Neural Network training with Pyro and stochastic variational inference
- Posterior predictive sampling and 95% prediction intervals
- Exceedance probabilities for user-defined Chl-a alert thresholds
- Validation-based hyperparameter selection using accuracy, uncertainty, and generalization scores

The current repository provides the BNN forecasting component. GAM-based nonlinear attribution is not included in this release.

## Input data format

The workflow uses a daily dataset with one row per date. The sample dataset contains the following columns:

```text
date, sin_month, cos_month, Chla, CODMn, NH4_N, TP, TN, T, PAR,
PAR_ratio, Q, H, CVQ7, V, h, FR, HR7, tau_b
```

`Chla` is used both as the forecast target and as an antecedent predictor. For a scenario with `AT = 1` and `LT = 7`, observations from `t-6` to `t` are used to forecast Chl-a at `t+1`.

## Installation

```bash
git clone https://github.com/1LunLi/BNN-GAM-Chla-Forecasting.git
cd BNN-GAM-Chla-Forecasting
python -m pip install -r requirements.txt
```

## Run the demonstration

From the repository root, run:

```bash
python -m examples.run_demo
```

The demonstration uses synthetic daily data and a compact one-configuration search. Outputs are saved in:

```text
outputs/search_results_AT1_LT7.csv
outputs/test_predictions_AT1_LT7.csv
```

To conduct the complete hyperparameter search, replace `DEMO_SEARCH_SPACE` in `examples/run_demo.py` with the full search space defined in `src/model_selection.py`.

## Reproducibility notes

The workflow splits data chronologically into training, validation, and test subsets. Feature and target scalers are fitted using the training subset only. Hyperparameters are selected using training and validation data only; the test subset is reserved for final independent evaluation.

## Data availability

The original in-situ chlorophyll-a and water-quality monitoring data are subject to data-sharing restrictions from the monitoring authority and are therefore not publicly deposited. They may be available from the corresponding author upon reasonable request and with permission from the data provider.

The repository includes a synthetic dataset with the same daily input format solely to demonstrate the execution of the workflow. It does not contain original monitoring records, and it cannot reproduce the numerical results reported in the manuscript.

Public meteorological and radiation data sources are described in the manuscript and its Supplementary Material.

## License

This project is distributed under the license included in this repository.
