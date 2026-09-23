# BNN-GAM-Chla-Forecasting

Reference implementation of a Bayesian Neural Network (BNN) coupled with a Generalized Additive Model (GAM) for probabilistic chlorophyll-a forecasting and nonlinear attribution in regulated rivers.

## Scope

This repository provides a compact, reproducible demonstration of the modelling workflow described in the associated manuscript:

1. Time-ordered data preprocessing and lagged-feature construction.
2. Training-set-only feature standardization.
3. Bayesian neural network training and probabilistic prediction.
4. Hyperparameter/model selection using validation data.
5. Predictive uncertainty quantification.
6. 95% predictive intervals and exceedance probabilities.
7. GAM-based nonlinear attribution of the BNN-predicted chlorophyll-a response.

The public examples demonstrate the `AT=1, LT=7` setting, where the model uses the previous 7 days of information to forecast chlorophyll-a one day ahead.

## Repository Structure

```text
BNN-GAM-Chla-Forecasting/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   ├── sample_daily_data.xlsx
│   └── sample_gam_input.xlsx
├── examples/
│   ├── __init__.py
│   ├── run_demo.py
│   └── run_gam_demo.py
└── src/
    ├── __init__.py
    ├── preprocess.py
    ├── bnn_model.py
    ├── model_selection.py
    └── gam_attribution.py

## Input data format

The workflow uses a daily dataset with one row per date. The sample dataset contains the following columns:

```text
date, sin_month, cos_month, Chla, CODMn, NH4_N, TP, TN, T, PAR,
PAR_ratio, Q, H, CVQ7, V, h, FR, HR7, tau_b
```

`Chla` is used both as the forecast target and as an antecedent predictor. For a scenario with `AT = 1` and `LT = 7`, observations from `t-6` to `t` are used to forecast Chl-a at `t+1`.

## Installation

```bash
Python 3.10 or later is recommended.
git clone https://github.com/1LunLi/BNN-GAM-Chla-Forecasting.git
cd BNN-GAM-Chla-Forecasting
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the demonstration

From the repository root, run:

```bash
python -m examples.run_demo
```

The demonstration reads the synthetic daily dataset in data/sample_daily_data.xlsx and writes the results to outputs/.
Typical outputs include:

```text
outputs/
├── search_results_AT1_LT7.csv
└── test_predictions_AT1_LT7.csv
```
The prediction file contains forecast origins, target dates, observed values, predictive means, predictive standard deviations, 95% predictive intervals, and the probability that chlorophyll-a exceeds the demonstration threshold.

To conduct the complete hyperparameter search, replace `DEMO_SEARCH_SPACE` in `examples/run_demo.py` with the full search space defined in `src/model_selection.py`.

## Run the GAM Demonstration

The GAM example uses the synthetic F1L7 worksheet in:

```text
data/sample_gam_input.xlsx
```
Run:
```bash
python -m examples.run_gam_demo
```

The GAM results are written to:
```text
outputs/gam_demo/
├── gam_AT1LT7.png
└── gam_curves_AT1LT7.csv
```

The figure shows the fitted GAM response, sample points, and 95% confidence intervals for six environmental variables (sin(month), TP, T, PAR, tau_b, HR7).
For the demonstration, Chla_t is the synthetic BNN-predicted chlorophyll-a response.

The GAM transformation rules are:
- Chla_t: log10-transformed response.
- TP_avg, PAR_avg, and tau_b_avg: log10-transformed predictors.
- sin(month)_avg, T_avg, and HR7_avg: used without logarithmic transformation.

## Reproducibility notes

The workflow splits data chronologically into training, validation, and test subsets. Feature and target scalers are fitted using the training subset only. Hyperparameters are selected using training and validation data only; the test subset is reserved for final independent evaluation.

## Data availability

The original in-situ chlorophyll-a and water-quality monitoring data are subject to data-sharing restrictions from the monitoring authority and are therefore not publicly deposited. They may be available from the corresponding author upon reasonable request and with permission from the data provider.

The repository includes a synthetic dataset with the same daily input format solely to demonstrate the execution of the workflow. It does not contain original monitoring records, and it cannot reproduce the numerical results reported in the manuscript.

Public meteorological and radiation data sources are described in the manuscript and its Supplementary Material.

## Output files
Runtime outputs are written to outputs/. This directory is excluded by .gitignore and should not be committed to the repository.
Because the public examples use synthetic data, the resulting prediction metrics and GAM curves are intended only for software verification.

## Citation
If this repository is used, please cite the associated manuscript and acknowledge the repository version used for the analysis.

## License

This project is distributed under the license included in this repository.
