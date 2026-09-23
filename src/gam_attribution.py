"""GAM-based post-hoc interpretation of BNN predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pygam import LinearGAM, s


SCENARIOS = {
    "F1L7": "BNN_AT1LT7",
    "F3L2": "BNN_AT3LT2",
    "F7L4": "BNN_AT7LT4",
}

FACTORS = [
    {
        "name": "sin_month",
        "columns": ("sin(month)_avg",),
        "log10": False,
        "label": r"$\sin(\mathrm{month})$",
    },
    {
        "name": "TP",
        "columns": ("TP_avg",),
        "log10": True,
        "label": r"$\log_{10}$[TP (mg/L)]",
    },
    {
        "name": "T",
        "columns": ("T_avg",),
        "log10": False,
        "label": r"$T$ ($^\circ$C)",
    },
    {
        "name": "PAR",
        "columns": ("PAR_avg",),
        "log10": True,
        "label": r"$\log_{10}$[PAR (MJ/m$^2$/day)]",
    },
    {
        "name": "tau_b",
        "columns": (
            "tau_b_avg",
            "bed_shear_stress_avg",
            "τ_avg",
            "��_avg",
        ),
        "log10": True,
        "label": r"$\log_{10}(\tau_b)$ (N/m$^2$)",
    },
    {
        "name": "HR7",
        "columns": ("HR7_avg",),
        "log10": False,
        "label": r"$HR7$ (m)",
    },
]

# Optional turning points in original units:
# ("F1L7", "TP"): [(0.042, "red")]
TURNING_POINTS = {}


def _find_column(
    data: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str:
    for column in candidates:
        if column in data.columns:
            return column

    raise KeyError(
        f"None of the expected columns were found: {candidates}"
    )


def _transform(values: np.ndarray, use_log10: bool) -> np.ndarray:
    return np.log10(values) if use_log10 else values


def run_gam_analysis(
    input_path: str | Path,
    output_dir: str | Path,
    response_column: str = "Chla_t",
    n_splines: int = 12,
    lam: float = 0.6,
    grid_points: int = 500,
    dpi: int = 600,
) -> pd.DataFrame:
    """
    Fit univariate GAMs between selected predictors and BNN-predicted Chl-a.

    The response and selected positive predictors are log10-transformed.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input workbook not found: {input_path}")

    available_sheets = set(pd.ExcelFile(input_path).sheet_names)
    missing_sheets = set(SCENARIOS) - available_sheets

    if missing_sheets:
        raise ValueError(
            f"Missing worksheets: {sorted(missing_sheets)}"
        )

    figure, axes = plt.subplots(
        len(SCENARIOS),
        len(FACTORS),
        figsize=(18, 9),
        squeeze=False,
    )

    curve_tables = []

    for row, (sheet, scenario_label) in enumerate(SCENARIOS.items()):
        data = pd.read_excel(input_path, sheet_name=sheet)

        if response_column not in data.columns:
            raise KeyError(
                f"{response_column!r} was not found in worksheet {sheet!r}."
            )

        predicted_chla = pd.to_numeric(
            data[response_column],
            errors="coerce",
        ).to_numpy(dtype=float)

        for column_index, factor in enumerate(FACTORS):
            source_column = _find_column(data, factor["columns"])

            raw_x = pd.to_numeric(
                data[source_column],
                errors="coerce",
            ).to_numpy(dtype=float)

            valid = (
                np.isfinite(raw_x)
                & np.isfinite(predicted_chla)
                & (predicted_chla > 0)
            )

            if factor["log10"]:
                valid &= raw_x > 0

            x = _transform(raw_x[valid], factor["log10"])
            y = np.log10(predicted_chla[valid])

            if len(x) < n_splines + 1 or len(np.unique(x)) < 4:
                raise ValueError(
                    f"Insufficient data for {sheet} and {factor['name']}."
                )

            X = x.reshape(-1, 1)

            gam = LinearGAM(
                s(0, n_splines=n_splines),
                lam=lam,
            ).fit(X, y)

            x_grid = np.linspace(x.min(), x.max(), grid_points)
            X_grid = x_grid.reshape(-1, 1)

            fitted = gam.predict(X_grid)
            confidence = gam.confidence_intervals(
                X_grid,
                width=0.95,
            )

            original_x_grid = (
                10**x_grid
                if factor["log10"]
                else x_grid
            )

            curve_tables.append(
                pd.DataFrame(
                    {
                        "scenario": scenario_label,
                        "factor": factor["name"],
                        "x_original": original_x_grid,
                        "x_transformed": x_grid,
                        "predicted_log10_chla": fitted,
                        "lower_95": confidence[:, 0],
                        "upper_95": confidence[:, 1],
                    }
                )
            )

            axis = axes[row, column_index]
            axis.scatter(
                x,
                y,
                s=9,
                color="#1478B8",
                alpha=0.55,
                edgecolors="none",
            )
            axis.fill_between(
                x_grid,
                confidence[:, 0],
                confidence[:, 1],
                color="#BDBDBD",
                alpha=0.65,
            )
            axis.plot(
                x_grid,
                fitted,
                color="black",
                linewidth=1.6,
            )

            for value, color in TURNING_POINTS.get(
                (sheet, factor["name"]),
                [],
            ):
                plotted_value = (
                    np.log10(value)
                    if factor["log10"]
                    else value
                )
                axis.axvline(
                    plotted_value,
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                )

            axis.grid(
                color="#D9D9D9",
                linewidth=0.6,
                alpha=0.8,
            )
            axis.set_xlabel(factor["label"], fontsize=9)
            axis.tick_params(labelsize=8)

            if row == 0:
                axis.set_title(
                    factor["name"],
                    fontsize=10,
                    fontweight="bold",
                )

        axes[row, 0].annotate(
            scenario_label,
            xy=(-0.55, 0.5),
            xycoords="axes fraction",
            ha="right",
            va="center",
            fontsize=10,
        )

    figure.supylabel(
        r"BNN-predicted $\log_{10}$[Chl-a ($\mu$g/L)]",
        fontsize=11,
    )

    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color="#1478B8",
            markersize=4,
            label="Samples",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.6,
            label="GAM smooth function",
        ),
        Patch(
            facecolor="#BDBDBD",
            alpha=0.65,
            label="95% confidence interval",
        ),
    ]

    figure.legend(
        handles=legend_items,
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0.07, 0.03, 1.0, 0.94))

    figure_path = output_dir / "gam_nonlinear_responses.png"
    figure.savefig(
        figure_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)

    curves = pd.concat(curve_tables, ignore_index=True)
    curves.to_csv(
        output_dir / "gam_fitted_curves.csv",
        index=False,
    )

    return curves


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit GAMs to BNN-predicted Chl-a."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/gam")
    args = parser.parse_args()

    run_gam_analysis(
        input_path=args.input,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
